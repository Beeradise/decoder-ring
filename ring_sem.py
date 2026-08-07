r"""ring_sem.py — Decoder Ring v1.4 semantic evidence channel + miss-targeted rerank.

WHAT THIS IS. A NEW module layered ON TOP of the frozen ring_core + the untouched
ring_store (decision 4.0: ring_store.py gets ZERO edits in v1.4, so the v1.3 lexical path
is byte-identical because the file is byte-identical). It adds a second, orthogonal
evidence channel to the coverage retriever: F9-style binary keys over nomic-embed-text
window embeddings, plus three recovery mechanisms (semantic injection, exact-text credit,
neighbor credit) that the frozen 25-question eval attributes per-mechanism.

THE SIDECAR (store\semantic\, page-set-versioned, entirely additive):
  * per-doc window embeddings (fp16, L2-normalized) under emb\, one npz per manifest doc.
    emb\ is BUILD-TIME-ONLY state: queries read keys+mu+provenance+text, never emb\ (grep-
    verified: emb_dir is touched only by _key_phase / _doc_entry_fresh's existence check /
    tests). `sembuild --slim` (v1.7, opt-in) rewrites each emb npz WITHOUT the fp16 `emb`
    member (keeping win_start/win_len so _doc_entry_fresh stays true), reclaiming ~57%% of the
    sidecar with ZERO query-time effect. Un-slim = `sembuild --rebuild` (full re-embed).
    Incremental builds after a slim reuse the frozen keys row-for-row (sem cache_meta doc_ids
    is the reuse map) and embed only new/replaced docs;
  * drifted-doc chunk texts under text\ (README.md/CHANGELOG.md today: their source files
    drifted from the ingested pages by design, so their text is read straight from the v3
    page's stored token ids (page_chunk_ids) instead of the disk source — no NN-decode);
  * per-token UTF-8 byte lengths under toklens\ (O(1) chunk/window byte slicing forever);
  * a DERIVED cache\ (packed keys matrix + window provenance) mirroring store\cache\
    discipline (safe to delete; rebuilds when sha256(meta.json) moves).

THE KEY CONTRACT (F9-style, frozen after the first build; receipts S3/S5 in the plan):
  key(x) = pack_LSB( ((x - mu) @ R.T) > 0 )  as (128,) uint64 = 8192 bits, x L2-normalized.
  * R : (8192, 768) Rademacher +-1, regenerated (NOT stored) from the frozen SHA256-CTR
    stream rc.drbg_vector(b"SEMR", i), i=0..383 (8192*768/16384 = exactly 384 vectors). Its
    sign pattern is checksummed (r_sha256) and re-verified at load (mismatch = hard error).
  * mu : float32 mean of ALL window embeddings at first full build; centering removes the
    large common component of nomic embeddings (raw cosines cluster 0.7-0.9) so the sign
    projection spreads the space. Frozen after first build (mu_sha256 guards it).
  * query similarity: sem = cos(pi * hamming(kq, kw) / 8192) in [-1, 1] (angular-LSH
    estimate, rank-monotone in hamming).

NORMALIZATION IS LOAD-BEARING (receipt S12): Ollama returns UNnormalized vectors -> L2 on
receipt before storing fp16; the nomic task prefixes are mandatory (documents get
"search_document: ", queries "search_query: ").

ABSTENTION + DIVERSITY (v1.6). Two query-time code constants, no new persistent state:
  * SEM_ABSTAIN_TAU (0.505): the full stack abstains ("no grounded answer in the store") when
    the top doc_keep-masked window cosine max(sem_win) falls below tau. The semantic channel is
    the only signal a co-occurring-words phantom cannot forge; the measured empty gap
    [0.499, 0.512] separates all 45 answerable eval questions from 5/7 unanswerable ones. Honest
    residual: near-domain U2/U5 (tungsten / A380 vocabulary present, the fact absent) escape at
    ~0.64-0.66 — catching them needs an answer-level reader, out of scope. Abstention is a pure
    decision layer: it never changes ranking, and fires ONLY on the full-stack path with a real
    semantic scan (never on the embed-degrade->cov / sidecar-absent / all-stop->v12 paths).
  * SEM_DOC_CAP (2): at most 2 chunks per document in the returned top-k (cap-then-backfill so
    the count never shrinks; rank 1 is always admitted). Recovers a lexically-buried gold that a
    single document's own chunks would otherwise crowd out of the top-3.

No new pip deps (numpy + stdlib + ring_core/ring_store). Never restarts / auto-starts the
island; embeds via HTTP POST /api/embed only.
"""

import gc
import gzip
import hashlib
import io
import json
import os
import re
import struct
import time
import urllib.error
import urllib.request

import numpy as np

import ring_core as rc
import ring_store as rs

# ---------------------------------------------------------------- semantic contract constants
SEM_VERSION = 1
SEM_MODEL = "nomic-embed-text"
SEM_DIM = 768                       # nomic-embed-text embedding_length
SEM_DS = 8192                       # binary key width in bits (= 128 uint64 words); receipt S5
SEM_WORDS = SEM_DS // 64            # 128 packed uint64 words per key
SEM_WIN = 8                         # window width in chunks (~128 Llama tokens); receipt S2
SEM_STRIDE = 4                      # window stride in chunks
SEM_R_STREAM = b"SEMR"              # rc.drbg_vector label for R (a new derived use, zero rc edit)
SEM_R_NVEC = SEM_DS * SEM_DIM // rc.D  # 384 drbg vectors compose the R bit stream (exact)
SEM_DOC_PREFIX = "search_document: "
SEM_QUERY_PREFIX = "search_query: "
SEM_SEED = "0xDEC0DE01"             # documented provenance (rc.GENESIS_SEED)

# recovery-mechanism knobs (receipts S7/S8/S9; the shipped calibration is measured by eval)
SEM_NB_GAMMA = 0.4                  # neighbor-credit discount (absent-native only)
SEM_NB_NATIVE_MAX = 0.1            # only lift chunks whose native term_phi_chunk < this
SEM_PHANTOM_FLOOR = 0.25           # else-branch cred cap = min(phi, floor); kills phantoms
SEM_TIE_A = 0.25                   # nbcov tie-break weight in the final score
SEM_WEIGHT_DEFAULT = 0.75          # semantic-channel weight in the final score
SEM_C_SHORT = 1024                 # lexical shortlist size (top nbcov)
SEM_S_INJECT = 256                 # semantic injection: top windows whose chunks enter cands
SEM_EMBED_BATCH = 128              # /api/embed batch (receipt S1: ~105 texts/s, flat 32-256)

# v1.6 query-time calibration (code constants; recalibrated only when the scoring pipeline moves)
SEM_ABSTAIN_TAU = 0.505            # abstain iff max kept sem_win < tau; measured gap [0.499, 0.512]
SEM_DOC_CAP = 2                    # max chunks per doc in the returned top-k (backfilled, rank-1-safe)

# derived-cache members (a missing one => cache invalid, mirrors rs._CACHE_ARRAYS)
_SEM_CACHE_ARRAYS = ("keys_matrix.npy", "win_doc.npy", "win_start.npy", "win_len.npy",
                     "win_first_chunk.npy", "cov_w0.npy", "cov_w1.npy")

_R_CACHE = None                    # module-level memo for the regenerated R


class SemError(Exception):
    """Recoverable semantic-channel failure (island down, embed timeout, sidecar corruption)."""


# ---------------------------------------------------------------- paths (mirror rs.store_paths)
def semantic_paths(ring_dir):
    """(sem_dir, emb_dir, toklens_dir, text_dir, cache_dir, meta_path, mu_path)."""
    sem_dir = os.path.join(rs.store_paths(ring_dir)[0], "semantic")
    return (
        sem_dir,
        os.path.join(sem_dir, "emb"),
        os.path.join(sem_dir, "toklens"),
        os.path.join(sem_dir, "text"),
        os.path.join(sem_dir, "cache"),
        os.path.join(sem_dir, "meta.json"),
        os.path.join(sem_dir, "mu.npy"),
    )


def _emb_name(entry):
    slug = rs._SLUG_RE.sub("", entry["name"])[:40] or "doc"
    return f"{slug}-{entry['doc_id']}.npz"


# ---------------------------------------------------------------- R / mu contract (4.2)
def make_R():
    """(8192, 768) float32 Rademacher +-1, regenerated from rc.drbg_vector(b"SEMR", i).

    Cached module-level (24 MB). The bit stream is exactly SEM_R_NVEC=384 frozen 16384-bit
    drbg vectors (384*16384 = 8192*768), unpacked LSB-first, reshaped row-major, bit->+-1.
    """
    global _R_CACHE
    if _R_CACHE is None:
        words = np.concatenate([rc.drbg_vector(SEM_R_STREAM, i) for i in range(SEM_R_NVEC)])
        bits = np.unpackbits(words.view(np.uint8), bitorder="little")[:SEM_DS * SEM_DIM]
        _R_CACHE = (bits.astype(np.float32) * 2.0 - 1.0).reshape(SEM_DS, SEM_DIM)
    return _R_CACHE


def r_sha256(R=None):
    """sha256 over R's packed sign bits (LSB-first) — the frozen-R guard checksum."""
    if R is None:
        R = make_R()
    packed = np.packbits((R > 0).reshape(-1).astype(np.uint8), bitorder="little")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def mu_sha256(mu):
    return hashlib.sha256(np.ascontiguousarray(mu, dtype="<f4").tobytes()).hexdigest()


def key_of(emb, mu, R, block=8192):
    """L2-normalized embeddings (n,768) -> packed (n,128) uint64 keys = sign((x-mu) @ R.T).

    Row-blocked so the transient projection (block, 8192) f32 stays < ~300 MB (receipt S11)."""
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb[None, :]
    n = emb.shape[0]
    Rt = np.ascontiguousarray(R.T)                      # (768, 8192)
    out = np.empty((n, SEM_WORDS), dtype="<u8")
    for s in range(0, n, block):
        proj = (emb[s:s + block] - mu) @ Rt             # (b, 8192)
        bits = (proj > 0)
        out[s:s + block] = np.packbits(bits, axis=1, bitorder="little").view("<u8")
    return out


def _key_hamming(keys, q_key, block=65536):
    """Blocked Hamming of a (128,) uint64 query key vs an (Nw,128) uint64 key matrix -> (Nw,)."""
    q = np.ascontiguousarray(q_key, dtype="<u8")
    nw = keys.shape[0]
    h = np.empty(nw, dtype=np.int64)
    for s in range(0, nw, block):
        blk = np.ascontiguousarray(keys[s:s + block])
        h[s:s + blk.shape[0]] = np.bitwise_count(blk ^ q).sum(axis=1, dtype=np.int64)
    return h


# ---------------------------------------------------------------- embedding (HTTP /api/embed)
def embed_texts(endpoint, texts, batch=SEM_EMBED_BATCH, timeout=600):
    """POST /api/embed for `texts`, L2-normalize each vector, return (n,768) float32.

    Single in-flight batch (keeps the 2070 polite; receipt: do not parallelize). Any HTTP /
    transport / shape failure raises SemError so callers degrade to lexical-only."""
    endpoint = endpoint.rstrip("/")
    out = np.empty((len(texts), SEM_DIM), dtype=np.float32)
    for s in range(0, len(texts), batch):
        grp = texts[s:s + batch]
        body = json.dumps({"model": SEM_MODEL, "input": grp}).encode("utf-8")
        req = urllib.request.Request(f"{endpoint}/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                embs = json.load(r).get("embeddings")
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as e:
            raise SemError(f"embed failed at {endpoint}/api/embed: {e}") from e
        if not embs or len(embs) != len(grp):
            raise SemError(f"embed returned {0 if not embs else len(embs)} of {len(grp)} vectors")
        v = np.asarray(embs, dtype=np.float32)
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        out[s:s + len(grp)] = v
    return out


def island_model_digest(endpoint, timeout=5):
    """(digest, ok) for SEM_MODEL from /api/tags; ok=False (digest='') if the island is down."""
    try:
        with urllib.request.urlopen(f"{endpoint.rstrip('/')}/api/tags", timeout=timeout) as r:
            data = json.load(r)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return "", False
    for m in data.get("models", []):
        if m.get("name", "").split(":")[0] == SEM_MODEL or m.get("name") == SEM_MODEL:
            return m.get("digest", ""), True
    return "", False


# ---------------------------------------------------------------- page chunk ids (v3)
def page_chunk_ids(ring_dir, entry):
    """Per-chunk token-id lists (+ chunk lengths) of a doc's v3 page, sliced from token_ids.

    v3 replacement for the old batched NN page-decode: the page stores the exact ingestion
    token IDs, so a drifted doc's per-chunk ids are just consecutive fanout-K slices of
    token_ids delimited by chunk_len (the last chunk is partial). No Hamming scan, no
    losslessness caveat — the ids ARE the ground truth. Returns (id_lists, chunk_len). Used
    by build_sidecar's page-decode provider branch and asserted by T27."""
    with np.load(rs.page_fs_path(ring_dir, entry["page"])) as npz:
        ids = np.asarray(npz["token_ids"], dtype=np.int64)
        clen = np.asarray(npz["chunk_len"], dtype=np.int64)
    assert int(clen.sum()) == int(ids.shape[0]), \
        f"page {entry['page']}: chunk_len sum {int(clen.sum())} != n_tokens {int(ids.shape[0])}"
    id_lists, off = [], 0
    for cl in clen:
        cl = int(cl)
        id_lists.append([int(x) for x in ids[off:off + cl]])
        off += cl
    return id_lists, clen


# ---------------------------------------------------------------- window geometry
def doc_window_starts(nc):
    """Doc-local window starts for a doc of nc chunks (short docs get one covering window)."""
    return list(range(0, max(nc - SEM_STRIDE + 1, 1), SEM_STRIDE))


# ---------------------------------------------------------------- sidecar meta / staleness
def _stale_triple(entry):
    return (entry["doc_id"], entry["source_sha256"], entry["ingested"], int(entry["n_chunks"]))


def _load_meta(meta_path):
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _doc_entry_fresh(sem_doc, entry, emb_dir):
    if sem_doc is None:
        return False
    if (sem_doc.get("doc_id"), sem_doc.get("source_sha256"), sem_doc.get("ingested"),
            int(sem_doc.get("n_chunks", -1))) != _stale_triple(entry):
        return False
    return os.path.exists(os.path.join(emb_dir, _emb_name(entry)))


def _is_slimmed(sem_doc):
    """True if a prior sidecar meta entry has had its fp16 emb removed (v1.7 --slim)."""
    return bool(sem_doc and sem_doc.get("slimmed"))


# ---------------------------------------------------------------- sidecar build (4.3)
def build_sidecar(ring_dir, cache, token_matrix, codec, endpoint, embedder=None,
                  rebuild=False, status_only=False, log=print):
    """Build (or resume, or --status, or --rebuild) the semantic sidecar. Doc-granular resume.

    embedder(texts) -> (n,768) normalized float32; defaults to the live island embed_texts.
    Phase 1 (per doc, ~25 min embed + ~20 min one-time drifted-page decode): pick the text
    provider (disk if source sha matches the manifest, else page-decode), build windows, embed
    with the doc prefix, store fp16 emb + toklens/text. meta.json is rewritten after EACH doc
    (the resume point). Phase 2 (~2 min pure numpy): mu + R + binarize all windows -> derived
    cache. Returns the final meta dict.

    --rebuild (plan 4.5): re-derive keys + mu in phase 2 (mu is re-estimated from all current
    embeddings) but re-embed ONLY stale docs in phase 1 -- it never wipes emb\\ nor re-decodes
    fresh pages. So on a complete sidecar it is a cheap phase-2-only pass (no island). To pick up
    stale docs, run with no flags (staleness is always respected)."""
    sem_dir, emb_dir, toklens_dir, text_dir, cache_dir, meta_path, mu_path = semantic_paths(ring_dir)
    if embedder is None:
        embedder = lambda texts: embed_texts(endpoint, texts)
    manifest = cache.manifest
    docs = manifest["docs"]
    man_sha = rs.manifest_sha(ring_dir)

    prev = _load_meta(meta_path)
    prev_docs = {d["doc_id"]: d for d in prev["docs"]} if prev else {}

    if status_only:
        log(f"sidecar status ({sem_dir}):")
        have = 0
        for d in docs:
            pd = prev_docs.get(d["doc_id"])
            fresh = _doc_entry_fresh(pd, d, emb_dir)
            have += fresh
            nw = len(doc_window_starts(int(d["n_chunks"])))
            mark = "  [slim]" if _is_slimmed(pd) else ""
            log(f"  {d['name']:<24} n_win~{nw:>5}  {'ok' if fresh else 'MISSING/stale'}{mark}")
        cache_ok = _cache_fresh(cache_dir, meta_path)
        log(f"docs fresh: {have}/{len(docs)}   derived cache: {'fresh' if cache_ok else 'stale'}")
        if prev and prev.get("slim"):
            log("sidecar is SLIM (embeds removed; queries unaffected; keys rebuild / un-slim = "
                "re-embed via sembuild --rebuild)")
        return prev

    for d in (sem_dir, emb_dir, toklens_dir, text_dir, cache_dir):
        os.makedirs(d, exist_ok=True)

    # island reachability check (never auto-start): only needed if any doc must be (re)embedded.
    # --rebuild re-derives keys+mu (phase 2) and is stale-only (plan 4.5), EXCEPT on a SLIM
    # sidecar where it force-re-embeds the slimmed docs (un-slim; the frozen keys cannot be
    # recomputed once the fp16 embeds are gone). A rebuild on a complete NON-slim sidecar still
    # needs no island (pure numpy phase 2).
    unslim = [d for d in docs if rebuild and _is_slimmed(prev_docs.get(d["doc_id"]))]
    if unslim:
        log(f"--rebuild on a SLIM sidecar: force-re-embedding {len(unslim)} slimmed doc(s) "
            f"(UN-SLIM; island required)")
        # un-slim is the ONLY path with a TOTAL key shift: it re-embeds every slimmed doc on the
        # GPU (not bit-reproducible) AND re-estimates mu, moving ALL keys ~25-57 bits on the
        # 8192-bit keys -- the same order as the abstention margins (+/-0.006-0.007 vs tau 0.505).
        # No error is raised, so a borderline U/B question can silently cross tau.
        log("  UN-SLIM ABSTENTION HAZARD: after this completes, re-run  python eval_ring.py "
            "--set AB  and compare the abstention block (margins are +/-0.006-0.007); if any U/B "
            "verdict flipped, recalibrate tau via  eval_ring.py --sweep-cal.")
    need_embed = bool(unslim) or any(
        not _doc_entry_fresh(prev_docs.get(d["doc_id"]), d, emb_dir) for d in docs)
    if need_embed:
        digest, up = island_model_digest(endpoint) if endpoint else ("", True)
        if endpoint and not up:
            raise SemError("island down - start it first (Start Ring Island.bat)")
    else:
        digest = (prev or {}).get("model_digest", "")

    # offsets (global chunk base per doc, manifest order = cache chunk order)
    off, offsets = 0, {}
    for d in docs:
        offsets[d["doc_id"]] = off
        off += int(d["n_chunks"])

    sem_docs, t_all = [], time.perf_counter()
    for d in docs:
        doc_id = d["doc_id"]
        emb_path = os.path.join(emb_dir, _emb_name(d))
        # staleness is respected even under --rebuild (plan 4.5: --rebuild re-derives keys+mu in
        # phase 2 but re-embeds ONLY stale docs; it never wipes emb\ nor re-decodes fresh pages) --
        # EXCEPT a slimmed doc under --rebuild, which is forced stale here to un-slim it.
        force = rebuild and _is_slimmed(prev_docs.get(doc_id))
        if not force and _doc_entry_fresh(prev_docs.get(doc_id), d, emb_dir):
            sem_docs.append(prev_docs[doc_id])           # resume: keep the prior entry verbatim
            log(f"  {d['name']:<24} n_win={prev_docs[doc_id]['n_windows']:>5} SKIP (fresh)")
            continue

        nc = int(d["n_chunks"])
        starts = doc_window_starts(nc)
        win_start = np.array(starts, dtype=np.int32)
        win_len = np.array([min(SEM_WIN, nc - s) for s in starts], dtype=np.int32)

        ok_sha, _ = _source_sha_ok(d)
        t0 = time.perf_counter()
        if ok_sha:
            provider = "disk"
            # READ AS TEXT (universal newlines): the store ingested via extract_text's
            # open(encoding="utf-8"), which normalizes CRLF->LF. Reading binary here would keep
            # \r\n, drifting the tokenization/byte offsets vs the store's \n chunks (deep chunks
            # then slice to the wrong text). src bytes = the \n-normalized text bytes.
            with open(d["source_path"], encoding="utf-8", errors="replace") as f:
                text = f.read()
            src = text.encode("utf-8")
            ids = codec.encode_ids(text)
            toklens = np.fromiter((len(b) for b in codec.ids_to_bytes(ids)),
                                  dtype=np.int64, count=len(ids))
            cum = np.empty(len(ids) + 1, dtype=np.int64)
            cum[0] = 0
            np.cumsum(toklens, out=cum[1:])
            ntok = len(ids)
            win_texts = [src[cum[s * 16]:cum[min((s + SEM_WIN) * 16, ntok)]].decode("utf-8", "replace")
                         for s in starts]
            rs._save_npy_atomic(os.path.join(toklens_dir, f"{doc_id}.npy"),
                                toklens.astype(np.uint16))
        else:
            provider = "page-decode"
            dec_ids, clen = page_chunk_ids(ring_dir, d)   # v3: ids are ground truth (no NN decode)
            chunk_texts = [codec.ids_to_text(ids) for ids in dec_ids]
            _write_gz(os.path.join(text_dir, f"{doc_id}.json.gz"), {
                "chunk_texts": chunk_texts, "page": d["page"], "ingested": d["ingested"],
                "manifest_source_sha256": d["source_sha256"],
            })
            win_texts = ["".join(chunk_texts[s:s + int(win_len[i])]) for i, s in enumerate(starts)]

        emb = embedder([SEM_DOC_PREFIX + t for t in win_texts]).astype(np.float16)
        _save_npz_atomic(emb_path, emb=emb, win_start=win_start, win_len=win_len)
        entry = {"doc_id": doc_id, "name": d["name"], "n_chunks": nc, "n_windows": len(starts),
                 "source_sha256": d["source_sha256"], "ingested": d["ingested"],
                 "text_provider": provider}
        sem_docs.append(entry)
        log(f"  {d['name']:<24} n_win={len(starts):>5} embed={time.perf_counter()-t0:5.1f}s "
            f"[{provider}]")
        # rewrite meta after each doc = the resume point (mu/R/checksums finalized in phase 2)
        _write_meta(meta_path, sem_docs, digest, man_sha, mu_path, rebuild)

    log(f"phase 1 done ({time.perf_counter()-t_all:.0f}s). phase 2: mu + R + keys ...")
    meta = _key_phase(ring_dir, docs, offsets, sem_docs, digest, man_sha, rebuild, log)
    log(f"sidecar complete: {len(docs)} docs, {meta['n_windows']} windows "
        f"(total {time.perf_counter()-t_all:.0f}s)")
    return meta


def slim_sidecar(ring_dir, cache, log=print):
    r"""Opt-in (v1.7): REMOVE the fp16 `emb` arrays from a finalized sidecar, keeping keys +
    provenance + text + mu. Queries read keys+mu+provenance+text only — never emb\ — so
    retrieval is byte-identical; slim just reclaims the ~57%% of the sidecar that is
    build-time-only embedding state (measured: 242.7 MB of 431.5 MB on the live store).

    One-way cheap; un-slim = `sembuild --rebuild` (full re-embed, island required). Idempotent
    ("already slim" no-op). REFUSES unless the sidecar is finalized, fresh (manifest sha
    matches), its derived cache is fresh, and every manifest doc is fresh (no doc needs
    embedding) -> "run sembuild first". Explain-before-removing: prints exactly what is removed
    (the fp16 embeds), the bytes reclaimed, and the un-slim pointer.

    Each emb npz is rewritten WITHOUT `emb` (win_start/win_len kept so _doc_entry_fresh stays
    true with zero logic change). Then (CRITICAL) the sem cache_meta is re-chained to the
    rewritten meta.json -- without this _cache_fresh fails, load_sidecar returns None, and
    `auto` silently degrades to cov. The re-chain also records `doc_ids` (build-order) as the
    row-reuse map for later incremental builds."""
    sem_dir, emb_dir, _, _, cache_dir, meta_path, mu_path = semantic_paths(ring_dir)
    meta = _load_meta(meta_path)
    if meta is None or not meta.get("finalized"):
        raise SemError("no finalized sidecar to slim - run: python ring_cli.py sembuild")
    if meta.get("slim"):
        log("sidecar is already SLIM (embeds removed) - nothing to do")
        return meta
    if meta.get("manifest_sha256") != rs.manifest_sha(ring_dir):
        raise SemError("sidecar is stale (store changed since build) - run sembuild first, then --slim")
    if not _cache_fresh(cache_dir, meta_path):
        raise SemError("sidecar derived cache is stale - run sembuild first, then --slim")
    docs = cache.manifest["docs"]
    prev_docs = {d["doc_id"]: d for d in meta["docs"]}
    for d in docs:
        if not _doc_entry_fresh(prev_docs.get(d["doc_id"]), d, emb_dir):
            raise SemError(f"{d['name']}: doc needs embedding - run sembuild first, then --slim")

    # rewrite each emb npz WITHOUT the fp16 `emb` member (win_start/win_len kept)
    reclaimed, n = 0, 0
    for d in docs:
        emb_path = os.path.join(emb_dir, _emb_name(d))
        before = os.path.getsize(emb_path)
        with np.load(emb_path) as npz:
            if "emb" not in npz.files:                 # defensive: an already-slim file, leave it
                continue
            ws = np.asarray(npz["win_start"], dtype=np.int32)
            wl = np.asarray(npz["win_len"], dtype=np.int32)
        _save_npz_atomic(emb_path, win_start=ws, win_len=wl)
        reclaimed += before - os.path.getsize(emb_path)
        n += 1

    # meta: keep EVERY field (r_sha256/mu_sha256/manifest_sha256/finalized/... so load_sidecar's
    # checksum gates still pass), just add the slim flags
    meta = dict(meta)
    meta["slim"] = True
    meta["docs"] = [dict(e, slimmed=True) for e in meta["docs"]]
    rs.atomic_write_bytes(meta_path, json.dumps(meta, indent=2).encode("utf-8"))

    # re-chain the sem cache_meta to the rewritten meta.json (+ record the doc_ids reuse map)
    cm_path = os.path.join(cache_dir, "cache_meta.json")
    cm = _load_meta(cm_path) or {}
    cm["sem_meta_sha256"] = hashlib.sha256(open(meta_path, "rb").read()).hexdigest()
    cm["doc_ids"] = [d["doc_id"] for d in docs]
    rs.atomic_write_bytes(cm_path, json.dumps(cm, indent=2).encode("utf-8"))

    log(f"sidecar SLIM: removed the fp16 embed arrays from {n} emb npz, reclaimed "
        f"{reclaimed:,} B ({reclaimed/1e6:.1f} MB). keys/provenance/text kept; queries byte-identical.")
    log("un-slim = python ring_cli.py sembuild --rebuild  (full re-embed, island required, "
        "~25 min at this scale)")
    log("  UN-SLIM ABSTENTION HAZARD: un-slim re-embeds on the GPU + re-estimates mu, so ALL keys "
        "shift ~25-57 bits (~the abstention margin). After any un-slim, re-run  python eval_ring.py "
        "--set AB  and compare the abstention block (margins +/-0.006-0.007); if any U/B verdict "
        "flipped, recalibrate tau via  eval_ring.py --sweep-cal.")
    return meta


def _key_phase(ring_dir, docs, offsets, sem_docs, digest, man_sha, rebuild, log):
    """Phase 2: assemble all window keys, freeze mu (first build), write the derived cache +
    final meta. Pure numpy, re-runnable, never re-embeds.

    SLIM-AWARE (v1.7): a slimmed doc has no fp16 `emb` in its npz, so its key rows are REUSED
    from the prior keys_matrix (sem cache_meta `doc_ids` is the row-reuse map) instead of
    recomputed. New/replaced docs still carry `emb` and are keyed normally with the FROZEN mu
    (slim never re-estimates mu). Every build writes `doc_ids` into the sem cache_meta so the
    next incremental build can reuse rows."""
    sem_dir, emb_dir, _, _, cache_dir, meta_path, mu_path = semantic_paths(ring_dir)
    sem_by_id = {e["doc_id"]: e for e in sem_docs}
    reuse = any(_is_slimmed(sem_by_id.get(d["doc_id"])) for d in docs)

    old_keys = old_win_doc = old_index = None
    mu = R = None
    if reuse:
        # slim key-reuse guards -> hard SemError naming the fix (a full re-embed restores keys)
        if not os.path.exists(mu_path):
            raise SemError("slim sidecar has no mu.npy -> keys unrecoverable; "
                           "run: python ring_cli.py sembuild --rebuild (full re-embed)")
        cm_path = os.path.join(cache_dir, "cache_meta.json")
        keys_path = os.path.join(cache_dir, "keys_matrix.npy")
        wd_path = os.path.join(cache_dir, "win_doc.npy")
        old_cm = _load_meta(cm_path) or {}
        old_doc_ids = old_cm.get("doc_ids")
        if not old_doc_ids or not os.path.exists(keys_path) or not os.path.exists(wd_path):
            raise SemError("slim key-reuse needs the prior derived cache "
                           "(keys_matrix/win_doc + sem cache_meta doc_ids); "
                           "run: python ring_cli.py sembuild --rebuild (full re-embed)")
        old_keys = np.load(keys_path)            # FULL read (not mmap): about to be os.replace'd
        old_win_doc = np.load(wd_path)
        if int(old_keys.shape[0]) != int(old_cm.get("n_windows", -1)):
            raise SemError("slim key-reuse: prior keys row count != cache_meta n_windows; "
                           "run: python ring_cli.py sembuild --rebuild (full re-embed)")
        old_index = {doc_id: i for i, doc_id in enumerate(old_doc_ids)}
        mu = np.load(mu_path).astype(np.float32)   # slim implies FROZEN mu (never re-estimate)
        R = make_R()

    key_parts, emb_parts = [], []
    win_doc, win_start_all, win_len_all, win_first = [], [], [], []
    for di, d in enumerate(docs):
        with np.load(os.path.join(emb_dir, _emb_name(d))) as npz:
            has_emb = "emb" in npz.files
            ws = np.asarray(npz["win_start"], dtype=np.int32)
            wl = np.asarray(npz["win_len"], dtype=np.int32)
            e = np.asarray(npz["emb"], dtype=np.float32) if has_emb else None
        if reuse:
            if has_emb:                          # new/replaced doc -> key with the frozen mu
                kd = key_of(e, mu, R)
            elif not _is_slimmed(sem_by_id.get(d["doc_id"])):
                raise SemError(f"{d['name']}: emb npz has no 'emb' member but the sidecar meta "
                               f"does not mark it slimmed (corruption); "
                               f"run: python ring_cli.py sembuild --rebuild")
            elif d["doc_id"] not in old_index:
                raise SemError(f"{d['name']}: slimmed doc absent from the prior key map (doc_ids); "
                               f"run: python ring_cli.py sembuild --rebuild (full re-embed)")
            else:                                # slimmed doc -> reuse its frozen rows verbatim
                kd = old_keys[old_win_doc == old_index[d["doc_id"]]]
                if kd.shape[0] != ws.shape[0]:
                    raise SemError(f"{d['name']}: slim key-reuse row count {kd.shape[0]} != "
                                   f"windows {ws.shape[0]}; run: sembuild --rebuild")
            key_parts.append(kd)
        else:
            emb_parts.append(e)
        win_doc.append(np.full(ws.shape[0], di, dtype=np.int32))
        win_start_all.append(ws)
        win_len_all.append(wl)
        win_first.append(offsets[d["doc_id"]] + ws)

    win_doc = np.concatenate(win_doc) if win_doc else np.empty(0, dtype=np.int32)
    win_start_all = np.concatenate(win_start_all) if win_start_all else np.empty(0, dtype=np.int32)
    win_len_all = np.concatenate(win_len_all) if win_len_all else np.empty(0, dtype=np.int32)
    win_first = np.concatenate(win_first) if win_first else np.empty(0, dtype=np.int32)

    if reuse:
        keys = np.concatenate(key_parts, axis=0) if key_parts \
            else np.empty((0, SEM_WORDS), dtype="<u8")
    else:
        all_emb = np.concatenate(emb_parts, axis=0)
        # mu frozen after first build (documented: adding docs does not re-estimate mu)
        if rebuild or not os.path.exists(mu_path):
            mu = all_emb.mean(axis=0).astype(np.float32)
            rs._save_npy_atomic(mu_path, mu)
        else:
            mu = np.load(mu_path).astype(np.float32)
        R = make_R()
        keys = key_of(all_emb, mu, R)
    nwin = int(keys.shape[0])

    # cov_w0/cov_w1: per global chunk, the <=2 covering window indices (consistent with the
    # actual stored windows -> no formula/off-by-one drift)
    n_chunks = int(offsets[docs[-1]["doc_id"]]) + int(docs[-1]["n_chunks"]) if docs else 0
    cov_w0 = np.full(n_chunks, -1, dtype=np.int32)
    cov_w1 = np.full(n_chunks, -1, dtype=np.int32)
    for wi in range(nwin):
        c0 = int(win_first[wi]); L = int(win_len_all[wi])
        seg0 = cov_w0[c0:c0 + L]
        empty = seg0 < 0
        seg0[empty] = wi
        cov_w1[c0:c0 + L][~empty] = wi
    assert int((cov_w0 < 0).sum()) == 0, "a chunk is covered by no window (window-grid bug)"

    os.makedirs(cache_dir, exist_ok=True)
    gc.collect()                                          # drop any stale keys mmap before replace
    rs._save_npy_atomic(os.path.join(cache_dir, "keys_matrix.npy"), keys)
    rs._save_npy_atomic(os.path.join(cache_dir, "win_doc.npy"), win_doc)
    rs._save_npy_atomic(os.path.join(cache_dir, "win_start.npy"), win_start_all)
    rs._save_npy_atomic(os.path.join(cache_dir, "win_len.npy"), win_len_all)
    rs._save_npy_atomic(os.path.join(cache_dir, "win_first_chunk.npy"), win_first)
    rs._save_npy_atomic(os.path.join(cache_dir, "cov_w0.npy"), cov_w0)
    rs._save_npy_atomic(os.path.join(cache_dir, "cov_w1.npy"), cov_w1)

    meta = _write_meta(meta_path, sem_docs, digest, man_sha, mu_path, rebuild,
                       n_windows=nwin, mu=mu, finalize=True)
    sem_meta_sha = hashlib.sha256(open(meta_path, "rb").read()).hexdigest()
    rs.atomic_write_bytes(os.path.join(cache_dir, "cache_meta.json"), json.dumps(
        {"sem_meta_sha256": sem_meta_sha, "n_windows": int(nwin), "n_chunks": int(n_chunks),
         "doc_ids": [d["doc_id"] for d in docs],       # v1.7 row-reuse map for slim incremental
         "built": rs._now_iso()}, indent=2).encode("utf-8"))
    return meta


def _write_meta(meta_path, sem_docs, digest, man_sha, mu_path, rebuild,
                n_windows=None, mu=None, finalize=False):
    meta = {
        "sem_version": SEM_VERSION, "model": SEM_MODEL, "model_digest": digest,
        "dim": SEM_DIM, "d_s": SEM_DS, "win": SEM_WIN, "stride": SEM_STRIDE,
        "doc_prefix": SEM_DOC_PREFIX, "query_prefix": SEM_QUERY_PREFIX,
        "seed": SEM_SEED, "r_stream": SEM_R_STREAM.decode("ascii"),
        "r_sha256": r_sha256(), "mu_sha256": mu_sha256(mu) if mu is not None else "",
        "manifest_sha256": man_sha, "built": rs._now_iso(),
        "n_windows": int(n_windows) if n_windows is not None else -1,
        "finalized": bool(finalize), "docs": sem_docs,
    }
    if mu is None and os.path.exists(mu_path):
        try:
            meta["mu_sha256"] = mu_sha256(np.load(mu_path))
        except OSError:
            pass
    rs.atomic_write_bytes(meta_path, json.dumps(meta, indent=2).encode("utf-8"))
    return meta


def _source_sha_ok(entry):
    try:
        with open(entry["source_path"], "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return False, ""
    return actual == entry["source_sha256"], actual


def _save_npz_atomic(path, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    rs.atomic_write_bytes(path, buf.getvalue())


def _write_gz(path, obj):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(json.dumps(obj).encode("utf-8"))
    rs.atomic_write_bytes(path, buf.getvalue())


def _read_gz(path):
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def _cache_fresh(cache_dir, meta_path):
    if not all(os.path.exists(os.path.join(cache_dir, a)) for a in _SEM_CACHE_ARRAYS):
        return False
    cm = os.path.join(cache_dir, "cache_meta.json")
    if not (os.path.exists(cm) and os.path.exists(meta_path)):
        return False
    try:
        with open(cm, encoding="utf-8") as f:
            want = json.load(f).get("sem_meta_sha256")
        return want == hashlib.sha256(open(meta_path, "rb").read()).hexdigest()
    except (json.JSONDecodeError, OSError):
        return False


# ---------------------------------------------------------------- sidecar load + providers
class Sidecar:
    """Loaded semantic sidecar: keys mmap + window provenance + lazy per-doc text providers.

    Drop it (`sidecar.close()`) before any sidecar-cache rebuild: keys is an mmap and a
    memory-mapped file cannot be os.replace'd on Windows (same discipline as rs bag_matrix)."""

    def __init__(self, ring_dir, cache, meta, mu, keys, win_doc, win_start, win_len,
                 win_first, cov_w0, cov_w1):
        self.ring_dir = ring_dir
        self.cache = cache
        self.meta = meta
        self.mu = mu
        self.keys = keys
        self.win_doc = win_doc
        self.win_start = win_start
        self.win_len = win_len
        self.win_first = win_first
        self.cov_w0 = cov_w0
        self.cov_w1 = cov_w1
        self.docs = cache.manifest["docs"]
        self.by_id = {d["doc_id"]: d for d in self.docs}
        off, self.offset = 0, {}
        for d in self.docs:
            self.offset[d["doc_id"]] = off
            off += int(d["n_chunks"])
        self._sem_docs = {d["doc_id"]: d for d in meta["docs"]}
        self._prov = {}                                  # doc_id -> lazy provider dict

    def close(self):
        self.keys = None
        gc.collect()

    def _provider(self, doc_id):
        p = self._prov.get(doc_id)
        if p is not None:
            return p
        d = self.by_id[doc_id]
        sd = self._sem_docs.get(doc_id, {})
        kind = sd.get("text_provider", "disk")
        p = {"kind": kind}
        if kind == "page-decode":
            gz = os.path.join(semantic_paths(self.ring_dir)[3], f"{doc_id}.json.gz")
            p["chunk_texts"] = _read_gz(gz)["chunk_texts"] if os.path.exists(gz) else None
        else:
            tl = os.path.join(semantic_paths(self.ring_dir)[2], f"{doc_id}.npy")
            src_ok, _ = _source_sha_ok(d)
            if src_ok and os.path.exists(tl):
                toklens = np.load(tl).astype(np.int64)
                cum = np.empty(len(toklens) + 1, dtype=np.int64)
                cum[0] = 0
                np.cumsum(toklens, out=cum[1:])
                try:
                    # text-mode read (CRLF->LF) to match the ingested \n tokenization + toklens
                    with open(d["source_path"], encoding="utf-8", errors="replace") as f:
                        p["src"] = f.read().encode("utf-8")
                    p["cum"] = cum if len(p["src"]) == int(cum[-1]) else None
                except OSError:
                    p["src"], p["cum"] = None, None
            else:
                p["src"], p["cum"] = None, None
        self._prov[doc_id] = p
        return p

    def chunk_text(self, c):
        """Decoded/sliced text of a global chunk c, or None when no provider is available."""
        di = int(self.cache.chunk_doc[c])
        doc = self.docs[di]
        lc = int(self.cache.chunk_local[c])
        p = self._provider(doc["doc_id"])
        if p["kind"] == "page-decode":
            ct = p.get("chunk_texts")
            return ct[lc] if ct is not None and lc < len(ct) else None
        if p.get("src") is None or p.get("cum") is None:
            return None
        cum, src = p["cum"], p["src"]
        ntok = len(cum) - 1
        a = int(cum[lc * 16]); b = int(cum[min((lc + 1) * 16, ntok)])
        return src[a:b].decode("utf-8", "replace")


def load_sidecar(ring_dir, cache):
    """Load the sidecar or return None when absent/stale (manifest moved). Raises SemError on
    a genuine corruption (missing R/mu checksum mismatch)."""
    sem_dir, emb_dir, _, _, cache_dir, meta_path, mu_path = semantic_paths(ring_dir)
    meta = _load_meta(meta_path)
    if meta is None or not meta.get("finalized"):
        return None
    if meta.get("manifest_sha256") != rs.manifest_sha(ring_dir):
        return None                                       # store changed since build -> stale
    if not _cache_fresh(cache_dir, meta_path):
        return None
    if not os.path.exists(mu_path):
        return None
    mu = np.load(mu_path).astype(np.float32)
    if meta.get("r_sha256") != r_sha256():
        raise SemError("sidecar R checksum mismatch (frozen-seed contract violated)")
    if meta.get("mu_sha256") and meta["mu_sha256"] != mu_sha256(mu):
        raise SemError("sidecar mu checksum mismatch (mu.npy corrupted)")
    keys = np.load(os.path.join(cache_dir, "keys_matrix.npy"), mmap_mode="r")
    win_doc = np.load(os.path.join(cache_dir, "win_doc.npy"))
    win_start = np.load(os.path.join(cache_dir, "win_start.npy"))
    win_len = np.load(os.path.join(cache_dir, "win_len.npy"))
    win_first = np.load(os.path.join(cache_dir, "win_first_chunk.npy"))
    cov_w0 = np.load(os.path.join(cache_dir, "cov_w0.npy"))
    cov_w1 = np.load(os.path.join(cache_dir, "cov_w1.npy"))
    return Sidecar(ring_dir, cache, meta, mu, keys, win_doc, win_start, win_len,
                   win_first, cov_w0, cov_w1)


# ---------------------------------------------------------------- shared text helper (5.7)
_WB_CACHE = {}


def word_present(word, text):
    """Left-word-boundary containment: (?<![a-z0-9]) + re.escape(word) in text.lower().

    Keeps tall->taller, rejects largest->large and use->because (audit r2 nit). Shared by the
    rerank exact-credit (4.4 step 6) and the eval autopsy/attribution (5.7)."""
    if not text:
        return False
    w = word.lower()
    pat = _WB_CACHE.get(w)
    if pat is None:
        pat = _WB_CACHE[w] = re.compile(r"(?<![a-z0-9])" + re.escape(w))
    return pat.search(text.lower()) is not None


# ---------------------------------------------------------------- neighbor credit (4.4 step 2)
def neighbor_credit(res, chunk_doc, gamma=SEM_NB_GAMMA, native_max=SEM_NB_NATIVE_MAX):
    """(tpc_n, nbcov): lift each term's chunk membership by gamma*max(left,right) ONLY where
    the native term_phi_chunk < native_max, neighbors masked to the same doc (receipt S8)."""
    tpc = res.term_phi_chunk
    N = tpc.shape[1]
    left = np.zeros_like(tpc); right = np.zeros_like(tpc)
    sl = np.zeros(N, dtype=bool); sr = np.zeros(N, dtype=bool)
    sl[1:] = chunk_doc[1:] == chunk_doc[:-1]
    sr[:-1] = chunk_doc[:-1] == chunk_doc[1:]
    left[:, 1:] = tpc[:, :-1]
    right[:, :-1] = tpc[:, 1:]
    nb = np.maximum(left * sl[None, :], right * sr[None, :])
    tpc_n = np.maximum(tpc, np.where(tpc < native_max, gamma * nb, 0.0))
    w = np.array([t["weight"] for t in res.terms], dtype=np.float32)
    nbcov = (tpc_n * w[:, None]).sum(axis=0)
    return tpc_n.astype(np.float32), nbcov.astype(np.float32)


# ---------------------------------------------------------------- semantic scan (4.4 step 4)
def semantic_scan(sidecar, q_emb, doc_keep=None):
    """(sem_chunk (N,), sem_win (Nw,)): per-chunk semantic score = max cos over its <=2
    covering windows. q_emb is the L2-normalized query embedding."""
    q_key = key_of(q_emb, sidecar.mu, make_R())[0]
    h = _key_hamming(sidecar.keys, q_key)
    sem_win = np.cos(np.pi * h.astype(np.float64) / SEM_DS).astype(np.float32)
    if doc_keep is not None:
        allowed = np.zeros(len(sidecar.docs), dtype=bool)
        for di in doc_keep:
            allowed[di] = True
        sem_win = np.where(allowed[sidecar.win_doc], sem_win, -1e9).astype(np.float32)
    sem_chunk = sem_win[sidecar.cov_w0].copy()
    m1 = sidecar.cov_w1 >= 0
    sem_chunk[m1] = np.maximum(sem_chunk[m1], sem_win[sidecar.cov_w1[m1]])
    return sem_chunk, sem_win


# ---------------------------------------------------------------- rerank core (4.4 steps 5-7)
def sem_candidates(cache, sidecar, sem_win, nbcov, doc_keep=None):
    """Candidate chunks (mechanism 3): top SEM_C_SHORT by nbcov UNION the chunks of the top
    SEM_S_INJECT windows by sem (semantic injection rescues lexically-hopeless golds, S4).

    With doc_keep, nbcov is masked to the kept docs BEFORE the argsort so the lexical shortlist
    is per-doc, matching retrieve_cov's mask-then-rank (plan 4.5: --doc applies exactly as cov)."""
    nb = nbcov
    if doc_keep is not None:
        keep = np.isin(cache.chunk_doc, np.array(sorted(doc_keep), dtype=np.int32))
        nb = np.where(keep, nbcov, -np.inf)
    cand = set(int(c) for c in np.argsort(-nb, kind="stable")[:SEM_C_SHORT])
    for wi in np.argsort(-sem_win, kind="stable")[:SEM_S_INJECT]:
        wi = int(wi)
        if sem_win[wi] <= -1e8:                          # doc_keep-masked window
            continue
        c0 = int(sidecar.win_first[wi]); L = int(sidecar.win_len[wi])
        cand.update(range(c0, c0 + L))
    if doc_keep is not None:
        cand = {c for c in cand if int(cache.chunk_doc[c]) in doc_keep}
    return sorted(cand)


def exact_credit(cache, sidecar, res, cand, floors=(SEM_PHANTOM_FLOOR,)):
    """Per-candidate exact-text credit (mechanism 2) evaluated at one or more phantom floors.

    cred_t(c) = 1.0 if the term word is in text(c) (left-word-boundary), 0.4 if in c+-1
    (same doc), else min(phi_t(c), floor) (phantom floor, receipt S7). Only the else-branch
    depends on floor, so all floors are computed in ONE text pass. Provider missing/stale ->
    cred falls back to min(phi, 1.0) for that chunk (counted so the caller can note it in the
    trace, plan 4.4 step 6). Returns ({floor: {c: cred_norm}}, Sw, n_no_text)."""
    w = np.array([t["weight"] for t in res.terms], dtype=np.float32)
    Sw = float(w.sum()) or 1.0
    words = [t["word"] for t in res.terms]
    tpc = res.term_phi_chunk
    N = cache.chunk_doc.shape[0]
    out = {float(f): {} for f in floors}
    n_no_text = 0
    for c in cand:
        di = int(cache.chunk_doc[c])
        txt = sidecar.chunk_text(c)
        if txt is None:
            n_no_text += 1
        nb_txt = ""
        for dc in (c - 1, c + 1):
            if 0 <= dc < N and int(cache.chunk_doc[dc]) == di:
                nb_txt += " " + (sidecar.chunk_text(dc) or "")
        base, else_terms = 0.0, []
        for ti, word in enumerate(words):
            wt = float(w[ti])
            if txt is not None and word_present(word, txt):
                base += wt
            elif txt is not None and word_present(word, nb_txt):
                base += 0.4 * wt
            elif txt is None:                            # provider missing/stale -> cred=min(phi,1)
                base += min(float(tpc[ti, c]), 1.0) * wt
            else:
                else_terms.append((float(tpc[ti, c]), wt))
        for f in out:
            extra = sum(min(phi, f) * wt for phi, wt in else_terms)
            out[f][c] = (base + extra) / Sw
    return out, Sw, n_no_text


def score_and_rank(cand, cred, nbcov, sem_chunk, Sw, topk, tie_a, sem_weight,
                   chunk_doc=None, doc_cap=None):
    """Final dimensionless blend (4.4 step 7): cred + tie_a*nbcov/Sw + sem_weight*sem, desc,
    drop score<=0, top-k (stable lower-index tiebreak). Returns (ordered, score_dict).

    v1.6 SEM_DOC_CAP diversity (only when chunk_doc is not None and doc_cap): at most doc_cap
    chunks per document in the returned top-k. Cap-then-backfill — walk the score-desc order,
    skip a chunk whose doc already holds doc_cap picks; if the capped walk cannot fill topk
    (single-doc scopes, e.g. --doc) backfill with the highest-ranked skipped chunks so the count
    never shrinks. Rank 1 is always admitted (the top scorer's doc is empty), so every @1 metric
    is frozen. Defaults (chunk_doc=None) reproduce the pre-cap order byte-for-byte."""
    score = {c: cred[c] + tie_a * float(nbcov[c]) / Sw + sem_weight * float(sem_chunk[c])
             for c in cand}
    ranked = [c for c in sorted(cand, key=lambda c: (-score[c], c)) if score[c] > 0]
    if chunk_doc is None or not doc_cap:
        return ranked[:topk], score
    ordered, per_doc, skipped = [], {}, []
    for c in ranked:
        d = int(chunk_doc[c])
        if per_doc.get(d, 0) >= doc_cap:
            skipped.append(c)
            continue
        per_doc[d] = per_doc.get(d, 0) + 1
        ordered.append(c)
        if len(ordered) >= topk:
            break
    if len(ordered) < topk:                              # backfill: never shrink the result count
        ordered += skipped[:topk - len(ordered)]
    return ordered, score


def rerank_full(cache, sidecar, res, sem_chunk, sem_win, topk,
                floor=SEM_PHANTOM_FLOOR, tie_a=SEM_TIE_A, sem_weight=SEM_WEIGHT_DEFAULT,
                doc_keep=None):
    """Full-stack rerank over candidates. Returns
    (ordered, per_hit, cand, cred_norm, nbcov, n_no_text)."""
    _tpc_n, nbcov = neighbor_credit(res, cache.chunk_doc)
    cand = sem_candidates(cache, sidecar, sem_win, nbcov, doc_keep)
    creds, Sw, n_no_text = exact_credit(cache, sidecar, res, cand, (floor,))
    cred_norm = creds[float(floor)]
    ordered, score = score_and_rank(cand, cred_norm, nbcov, sem_chunk, Sw, topk, tie_a, sem_weight,
                                    cache.chunk_doc, SEM_DOC_CAP)
    per_hit = [{"chunk": c, "score": float(score[c]), "chunk_cov": float(res.chunk_cov[c]),
                "cred": float(cred_norm[c]), "sem": float(sem_chunk[c]),
                "nbcov": float(nbcov[c])} for c in ordered]
    return ordered, per_hit, cand, cred_norm, nbcov, n_no_text


def retrieve_full(cache, token_matrix, codec, sidecar, question, topk, doc_keep=None,
                  sem_weight=None, embed_fn=None, floor=None, tie_a=None, abstain_tau=None):
    """Shipped `full` retrieval (4.4). embed_fn(list_of_texts)->normalized (n,768); on ANY
    embed failure prints one note and degrades to rs.retrieve_cov (exit 0 preserved).

    `abstain_tau` overrides the global SEM_ABSTAIN_TAU for this call (per-collection tau, v1.8);
    None reproduces the shipped constant byte-for-byte. Returns (ordered, per_hit, trace) parallel
    to rs.retrieve_cov, or None on all-stop (caller falls back to v12 exactly like cov)."""
    sem_weight = SEM_WEIGHT_DEFAULT if sem_weight is None else float(sem_weight)
    floor = SEM_PHANTOM_FLOOR if floor is None else float(floor)
    tie_a = SEM_TIE_A if tie_a is None else float(tie_a)
    tau = SEM_ABSTAIN_TAU if abstain_tau is None else float(abstain_tau)

    res = rs.cov_compute(cache, token_matrix, codec, question, doc_keep)
    if res is None:
        return None

    t0 = time.perf_counter()
    try:
        q_emb = embed_fn([SEM_QUERY_PREFIX + question])[0]
    except SemError as e:
        print(f"semantic: embed unavailable -> lexical cov only ({e})")
        return rs.retrieve_cov(cache, token_matrix, codec, question, topk, None, doc_keep)
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    sem_chunk, sem_win = semantic_scan(sidecar, q_emb, doc_keep)
    t_scan = time.perf_counter() - t0
    sem_top = float(sem_win.max())                       # best (doc_keep-masked) window = abstain signal

    ordered, per_hit, cand, _cred, _nb, n_no_text = rerank_full(
        cache, sidecar, res, sem_chunk, sem_win, topk, floor, tie_a, sem_weight, doc_keep)
    if n_no_text:                                        # plan 4.4 step 6: note it once in the trace
        print(f"semantic: {n_no_text} candidate(s) had no text provider -> cred=min(phi,1) for them")
    trace = {
        "terms": [{"word": t["word"], "weight": t["weight"]} for t in res.terms],
        "stopped": res.stopped, "skipped": res.skipped,
        "n_scans": len(res.scan_ids), "scan_secs": res.scan_secs,
        "n_windows": int(sidecar.keys.shape[0]), "embed_secs": t_embed, "sem_scan_secs": t_scan,
        "n_cand": len(cand), "inject": SEM_S_INJECT, "no_text_cand": n_no_text,
        "sem_top": sem_top, "abstain": bool(sem_top < tau), "abstain_tau": tau,
    }
    return ordered, per_hit, trace
