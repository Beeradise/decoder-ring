r"""ring_cli.py — Decoder Ring demo CLI (info / encode / decode / ingest / remove / stats / query).

The RAG demo: ingest a handful of short docs -> encode tokens -> build chunk vectors ->
store them in the paged store v2 (ring_store) -> retrieve by Hamming over bag vectors ->
decode chunks back to token IDs -> detokenize -> hand the reconstructed context to
llama3.2:1b via the local Ollama HTTP API. Endpoint precedence: `--ollama URL` flag, else
env RING_OLLAMA, else http://127.0.0.1:11436 — decoder-ring's OWN standalone RTX 2070 SUPER
Ollama island (start it with `Start Ring Island.bat`, or let the `ring.bat` wrapper cold-start
it). If the model is absent at the chosen endpoint the demo degrades gracefully, printing the
assembled context and exactly `demo needs: ollama pull llama3.2:1b`, then exiting 0. It never
pulls or substitutes a model.

Run:  python ring_cli.py info atoms.dring
      python ring_cli.py encode "The cat sat."
      python ring_cli.py ingest docs\hdc.txt docs\tokenizer.txt docs\beavers.txt
      python ring_cli.py ingest --add notes.md paper.pdf     # incremental; same path REPLACES
      python ring_cli.py stats
      python ring_cli.py query "What binds two hypervectors?" --topk 2
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np

import ring_core as rc
import ring_sem as sm
import ring_store as rs

OLLAMA = os.environ.get("RING_OLLAMA", "http://127.0.0.1:11436").rstrip("/")
DEMO_MODEL = "llama3.2:1b"
PROMPT_TMPL = (
    "Use ONLY the following context to answer.\n\n"
    "Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
)


def _reconfig_stdout():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- ring loading
def ring_paths(ring_dir):
    return (
        os.path.join(ring_dir, "atoms.dring"),
        os.path.join(ring_dir, "tokens.dring"),
        os.path.join(ring_dir, "tokenizer.json"),
    )


def load_ring(ring_dir, verify_tokens=True):
    """Load atoms (bits+T), the token matrix, and the tokenizer codec.

    verify_tokens defaults ON per plan 3.4 (payload sha256 over tokens.dring, ~1 s for
    250 MB) — atoms are always verified inside rc.load_atoms.
    """
    atoms_path, tokens_path, tok_path = ring_paths(ring_dir)
    atoms_bits, T_bits = rc.load_atoms(atoms_path)
    token_matrix = rc.load_tokens(tokens_path, verify=verify_tokens)
    codec = rc.TokenCodec(tok_path)
    return atoms_bits, T_bits, token_matrix, codec


def _codec_only(ring_dir):
    """Just the tokenizer codec (no 250 MB token matrix) — enough for remove/stats to render
    df_summary token labels without the full ring load."""
    return rc.TokenCodec(ring_paths(ring_dir)[2])


# ---------------------------------------------------------------- collections (v1.8)
def _resolve_root(args, create_ok=False):
    r"""(store_root, col_or_None) for a command's --collection. None/""/"default" -> (ring, None)
    = the legacy alias. A named collection resolves to <ring>\stores\<name>; unless create_ok it
    must already exist (store\manifest.json present) else SystemExit lists the existing ones. A
    bad name -> SystemExit naming the charset. Creation itself (announce + collection.json stamp)
    is handled by cmd_ingest, the only command with create_ok=True."""
    col = getattr(args, "collection", None)
    try:
        name = rs.sanitize_collection(col) if col else "default"
    except rs.StoreError as e:
        raise SystemExit(str(e))
    if name == "default":
        return args.ring, None
    store_root = rs.collection_root(args.ring, name)
    if not os.path.exists(rs.store_paths(store_root)[2]) and not create_ok:
        have = ", ".join(rs.list_collections(args.ring))
        raise SystemExit(f"collection {name!r} does not exist; have: {have} "
                         f"(create it with: ingest --add --collection {name})")
    return store_root, name


def _effective_tau(store_root):
    """The abstention tau for a collection root: collection.json sem_abstain_tau when present and
    a sane 0 < t < 1, else the global SEM_ABSTAIN_TAU (one warning on a bad value). Never crashes
    a query (freshness-neutral file; see ring_store.collection_meta_path)."""
    meta = rs.load_collection_meta(store_root)
    if "sem_abstain_tau" in meta:
        try:
            t = float(meta["sem_abstain_tau"])
        except (TypeError, ValueError):
            t = None
        if t is not None and 0.0 < t < 1.0:
            return t
        print(f"warning: collection.json sem_abstain_tau {meta['sem_abstain_tau']!r} out of "
              f"(0,1) - using global default {sm.SEM_ABSTAIN_TAU}")
    return sm.SEM_ABSTAIN_TAU


def _sidecar_state(store_root):
    """fresh|stale|ABSENT (+ ' [slim]') for a collection's semantic sidecar, via the same chain
    as load_sidecar but WITHOUT loading the keys mmap (collections must stay instant)."""
    _, _, _, _, sem_cache_dir, meta_path, _ = sm.semantic_paths(store_root)
    meta = sm._load_meta(meta_path)
    if meta is None or not meta.get("finalized"):
        return "ABSENT"
    slim = " [slim]" if meta.get("slim") else ""
    if meta.get("manifest_sha256") != rs.manifest_sha(store_root):
        return "stale" + slim
    if not sm._cache_fresh(sem_cache_dir, meta_path):
        return "stale" + slim
    return "fresh" + slim


def collections_report(ring_dir):
    r"""Per-collection summary rows (testable core of cmd_collections): name, docs, chunks, pages
    bytes, bag-cache state, sidecar state, tau. Reads manifests as plain JSON (NO ring load — must
    stay instant); a version-gated manifest (StoreError from load_manifest) is reported as
    'store vN (needs migration)' for that row rather than dying (watch-out: collections never
    crashes on a v2/v3 collection)."""
    rows = []
    for name in rs.list_collections(ring_dir):
        root = rs.collection_root(ring_dir, name)
        _, pages_dir, manifest_path, cache_dir = rs.store_paths(root)
        row = {"name": name, "note": ""}
        try:
            m = rs.load_manifest(root)
        except rs.StoreError:
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    v = json.load(f).get("store_version")
            except (OSError, json.JSONDecodeError, ValueError):
                v = "?"
            row["note"] = f"store v{v} (needs migration)"
            rows.append(row)
            continue
        row["docs"] = len(m["docs"])
        row["chunks"] = sum(int(d["n_chunks"]) for d in m["docs"])
        row["pages_bytes"] = sum(
            os.path.getsize(os.path.join(dp, fn))
            for dp, _, fns in os.walk(pages_dir) for fn in fns) if os.path.isdir(pages_dir) else 0
        cm_path = os.path.join(cache_dir, "cache_meta.json")
        if not os.path.exists(cm_path):
            row["cache"] = "absent"
        else:
            try:
                with open(cm_path, encoding="utf-8") as f:
                    fresh = json.load(f).get("manifest_sha256") == rs.manifest_sha(root)
                row["cache"] = "fresh" if fresh else "stale"
            except (json.JSONDecodeError, OSError):
                row["cache"] = "unreadable"
        row["sidecar"] = _sidecar_state(root)
        meta = rs.load_collection_meta(root)
        raw = meta.get("sem_abstain_tau")
        if name != "default" and isinstance(raw, (int, float)) and not isinstance(raw, bool) \
                and 0.0 < float(raw) < 1.0:
            row["tau"] = f"{float(raw):.3f} (collection override)"
        elif name != "default" and "sem_abstain_tau" in meta:   # present but out of (0,1)/not a number
            row["tau"] = f"{sm.SEM_ABSTAIN_TAU:.3f} (global default; invalid override ignored)"
        else:
            row["tau"] = f"{sm.SEM_ABSTAIN_TAU:.3f} (global default, wiki-calibrated)"
        rows.append(row)
    return rows


# ---------------------------------------------------------------- info
def cmd_info(args):
    path = args.file if os.path.isabs(args.file) else os.path.join(args.ring, args.file)
    h = rc.read_header(path)
    print(f"file            {path}")
    print(f"magic           {h['magic'].decode('latin1')}")
    print(f"version         {h['version']}")
    print(f"D               {h['D']}")
    print(f"words_per_vec   {h['words_per_vec']}")
    print(f"prng_id         {h['prng_id'].decode('latin1')}")
    print(f"seed            0x{h['seed']:08X}")
    print(f"n_vectors       {h['n_vectors']}")
    print(f"source_hash     {h['source_hash'].hex()}")
    print(f"payload_sha256  {h['payload_sha256'].hex()}")
    with open(path, "rb") as f:
        f.seek(rc.HEADER_SIZE)
        payload = f.read()
    ok = hashlib.sha256(payload).digest() == h["payload_sha256"]
    print(f"payload verify  {'OK' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------- encode
def cmd_encode(args):
    _, _, token_matrix, codec = load_ring(args.ring)
    clean = rc.CleanDecoder(token_matrix)
    ids = codec.encode_ids(args.text)
    print(f"text   {args.text!r}")
    print(f"ids    {ids}")
    print(f"{'id':>7}  {'bytes':<20} clean-decode")
    for i in ids:
        b = codec.ids_to_bytes([i])[0]
        rt = clean.decode(token_matrix[i])
        print(f"{i:>7}  {repr(b):<20} {'OK' if rt == i else 'FAIL(' + str(rt) + ')'}")
    print(f"detok  {codec.ids_to_text(ids)!r}")


# ---------------------------------------------------------------- decode
def cmd_decode(args):
    _, _, token_matrix, codec = load_ring(args.ring)
    words = np.load(args.vecfile)
    assert words.ndim == 2 and words.shape[1] == rc.WORDS, words.shape
    words = np.ascontiguousarray(words, dtype="<u8")
    clean = rc.CleanDecoder(token_matrix)
    ids = []
    for row in words:
        if args.noisy:
            tid, ham, z = rc.nn_decode(row, token_matrix)
            print(f"  noisy id={tid} ham={ham} z={z:.1f}")
        else:
            tid = clean.decode(row)
            if tid is None:
                tid, ham, z = rc.nn_decode(row, token_matrix)
                print(f"  clean MISS -> noisy id={tid} ham={ham} z={z:.1f}")
        ids.append(tid)
    print(f"ids    {ids}")
    print(f"text   {codec.ids_to_text(ids)!r}")


# ---------------------------------------------------------------- ingest
def cmd_ingest(args):
    """Bare `ingest` = fresh rebuild of store\\ (explain-then-wipe); `ingest --add` appends
    (same resolved path REPLACES). Per-file StoreError is printed and skipped; exit 0 if at
    least one doc was ingested, else 1."""
    atoms_bits, T_bits, token_matrix, codec = load_ring(args.ring)
    K = args.fanout
    root, col = _resolve_root(args, create_ok=True)
    _, _, manifest_path, _ = rs.store_paths(root)

    # implicit creation is on ingest ONLY, announced loudly (plan 3.2). The collection.json stamp
    # is deferred to the FIRST successful add (below) so a totally-failed ingest leaves no orphan
    # stamp/dir (audit r1 nit; deviation from plan 3.3 which stamped before the add loop).
    creating = col is not None and not os.path.exists(manifest_path)
    if creating:
        print(f"creating NEW collection {col!r} at stores\\{col}\\store\\")

    if not args.add:
        if os.path.exists(manifest_path):
            what = "store\\" if col is None else f"collection {col!r}"
            try:
                m = rs.load_manifest(root)
                nc = sum(int(d["n_chunks"]) for d in m["docs"])
                print(f"{what} exists ({len(m['docs'])} docs, {nc} chunks) -> rebuilding fresh "
                      f"(use --add to append)")
            except rs.StoreError:
                print(f"{what} exists (unreadable manifest) -> rebuilding fresh (use --add to append)")
        rs.fresh_store(root)

    ingested = 0
    for path in args.paths:
        try:
            entry = rs.add_document(
                root, path=path, token_matrix=token_matrix, T_bits=T_bits, codec=codec, K=K
            )
        except rs.StoreError as e:
            print(str(e))
            continue
        if creating and ingested == 0:                # stamp on the FIRST success (no orphan on fail)
            rs.save_collection_meta(root, {"name": col, "created": rs._now_iso()})
        ingested += 1
        print(f"{entry['name']:<20} tokens={entry['n_tokens']:>6} chunks={entry['n_chunks']:>4} "
              f"tree_levels={entry['tree_levels']} kind={entry['source_kind']}")

    m = rs.load_manifest(root)
    total = sum(int(d["n_chunks"]) for d in m["docs"])
    what = "store\\" if col is None else f"collection {col!r}"
    print(f"{what} now: {len(m['docs'])} docs, {total} chunks (cache rebuilds on next query)")
    sys.exit(0 if ingested else 1)


# ---------------------------------------------------------------- remove
def cmd_remove(args):
    root, col = _resolve_root(args)
    _, _, manifest_path, _ = rs.store_paths(root)
    if not os.path.exists(manifest_path):
        raise SystemExit("no store\\ found - run ingest first")
    codec = _codec_only(args.ring)
    try:
        rs.remove_document(root, args.ident, codec=codec)
    except rs.StoreError as e:
        raise SystemExit(str(e))
    m = rs.load_manifest(root)
    total = sum(int(d["n_chunks"]) for d in m["docs"])
    what = "store\\" if col is None else f"collection {col!r}"
    print(f"{what} now: {len(m['docs'])} docs, {total} chunks (cache rebuilds on next query)")


# ---------------------------------------------------------------- stats
def cmd_stats(args):
    root, col = _resolve_root(args)
    store_dir, _, manifest_path, cache_dir = rs.store_paths(root)
    if not os.path.exists(manifest_path):
        raise SystemExit("no store\\ found - run ingest first")
    m = rs.load_manifest(root)
    docs = m["docs"]
    if col is not None:
        print(f"collection {col!r}  (stores\\{col}\\store)")
    print(f"store   {store_dir}")
    print(f"format  {m['format']} v{m['store_version']}  D={m['D']}  fanout={m['fanout']}")
    print(f"created {m['created']}  updated {m['updated']}")
    print(f"\n{'name':<24} {'kind':>4} {'tokens':>8} {'chunks':>7} {'levels':>7}  page")
    total_t = total_c = 0
    for d in docs:
        print(f"{d['name']:<24} {d['source_kind']:>4} {d['n_tokens']:>8} {d['n_chunks']:>7} "
              f"{d['tree_levels']:>7}  {d['page']}")
        total_t += int(d["n_tokens"]); total_c += int(d["n_chunks"])
    print(f"{'TOTAL':<24} {'':>4} {total_t:>8} {total_c:>7}")

    ds = m.get("df_summary", {})
    print(f"\ndf_summary: {ds.get('n_chunks', 0)} chunks, {ds.get('n_distinct_tokens', 0)} distinct tokens")
    if ds.get("top"):
        print("top tokens by chunk-DF fraction:")
        for label, frac in ds["top"]:
            print(f"  {frac:>6.3f}  {label}")

    meta_path = os.path.join(cache_dir, "cache_meta.json")
    state = "absent (rebuilds on next query)"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                cm = json.load(f)
            state = "fresh" if cm.get("manifest_sha256") == rs.manifest_sha(root) \
                else "stale (rebuilds on next query)"
        except (json.JSONDecodeError, OSError):
            state = "unreadable (rebuilds on next query)"
    print(f"\ncache: {state}")

    if col is not None:                                   # tau line for named collections only
        cmeta = rs.load_collection_meta(root)
        raw = cmeta.get("sem_abstain_tau")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and 0.0 < float(raw) < 1.0:
            print(f"abstention tau: {float(raw):.3f} (collection override)")
        elif "sem_abstain_tau" in cmeta:                  # present but out of (0,1)/not a number
            print(f"abstention tau: {sm.SEM_ABSTAIN_TAU:.3f} (global default; invalid override ignored)")
        else:
            print(f"abstention tau: {sm.SEM_ABSTAIN_TAU:.3f} (global default)")


# ---------------------------------------------------------------- query
def ollama_models(timeout=5):
    """Installed model names via HTTP GET /api/tags (NEVER shell out to `ollama list`)."""
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=timeout) as r:
            data = json.load(r)
        return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None


def ollama_generate(prompt, timeout=120):
    body = json.dumps({
        "model": DEMO_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r).get("response", "")


def cmd_sembuild(args):
    """Build/resume the semantic sidecar (store\\semantic\\). --status prints coverage without
    building; --rebuild re-derives keys+mu and re-embeds only docs whose staleness triple moved
    (and force-re-embeds a SLIM sidecar = un-slim); --slim removes the fp16 embed arrays (opt-in,
    queries byte-identical). Requires the ring island up for any embed (never auto-starts it)."""
    global OLLAMA
    if args.ollama:
        OLLAMA = args.ollama.rstrip("/")
    if args.slim and (args.status or args.rebuild):
        raise SystemExit("sembuild --slim cannot be combined with --status/--rebuild "
                         "(un-slim = --rebuild; --slim only removes the fp16 embed arrays)")
    root, col = _resolve_root(args)
    _, _, manifest_path, _ = rs.store_paths(root)
    if not os.path.exists(manifest_path):
        raise SystemExit("no store\\ found - run ingest first")
    _, T_bits, token_matrix, codec = load_ring(args.ring, verify_tokens=False)
    cache = rs.ensure_cache(root, token_matrix, T_bits)
    if int(cache.bag.shape[0]) == 0:
        raise SystemExit("store is empty - ingest at least one document")
    try:
        if args.slim:
            sm.slim_sidecar(root, cache)
        else:
            sm.build_sidecar(root, cache, token_matrix, codec, OLLAMA,
                             rebuild=args.rebuild, status_only=args.status)
    except sm.SemError as e:
        raise SystemExit(str(e))
    sys.exit(0)


def cmd_query(args):
    global OLLAMA
    if args.ollama:
        OLLAMA = args.ollama.rstrip("/")
    root, col = _resolve_root(args)
    _, _, manifest_path, _ = rs.store_paths(root)
    if not os.path.exists(manifest_path):
        raise SystemExit("no store\\ found - run ingest first")
    atoms_bits, T_bits, token_matrix, codec = load_ring(args.ring)
    cache = rs.ensure_cache(root, token_matrix, T_bits)
    tau = _effective_tau(root)                            # per-collection abstention tau (v1.8)
    docs = cache.manifest["docs"]
    n_chunks = int(cache.bag.shape[0])
    K = int(cache.manifest["fanout"])                 # v3: chunk text detokenizes page ids
    if n_chunks == 0:
        raise SystemExit("store is empty - ingest at least one document")

    # --doc: case-insensitive substring on doc name -> set of allowed doc indices
    doc_keep = None
    if args.doc:
        needle = args.doc.lower()
        doc_keep = {i for i, d in enumerate(docs) if needle in d["name"].lower()}
        if not doc_keep:
            raise SystemExit(f"--doc {args.doc!r} matched no doc; have: {[d['name'] for d in docs]}")

    q_ids = codec.encode_ids(args.question)
    if not q_ids:
        # maj([], T) returns exactly T (every bit ties) -> retrieval on a nonsense key
        raise SystemExit("empty question: nothing to encode")

    # Retrieval dispatch (v1.4). auto (default) = the full semantic+coverage stack when a fresh
    # sidecar exists, else cov; full forces the stack; cov = v1.3 term coverage; v12 = the frozen
    # v1.2 majority-bag path (byte-identical output). cov/full fall back to v12 on an all-stop
    # query; full degrades to cov if the island embed is unavailable (exit 0 preserved).
    full = None
    cov = None
    if args.retriever in ("auto", "full"):
        sidecar, sem_err = None, None
        try:
            sidecar = sm.load_sidecar(root, cache)
        except sm.SemError as e:
            sem_err = e                                   # corruption -> one loud note below
        if sidecar is not None:
            # short embed timeout on the interactive query path (a hung island degrades to cov
            # in ~10 s, not the 600 s build timeout)
            full = sm.retrieve_full(
                cache, token_matrix, codec, sidecar, args.question, args.topk, doc_keep,
                args.sem_weight, lambda texts: sm.embed_texts(OLLAMA, texts, timeout=10),
                abstain_tau=tau)
            sidecar.close()
            if full is None:
                print("coverage: no usable content terms (all stop words) -> falling back to v12")
        else:
            if sem_err is not None:
                print(f"semantic: {sem_err} -> cov")
            else:
                print("semantic sidecar absent/stale -> cov (run: python ring_cli.py sembuild)")
            cov = rs.retrieve_cov(cache, token_matrix, codec, args.question, args.topk, args.topw, doc_keep)
            if cov is None:
                print("coverage: no usable content terms (all stop words) -> falling back to v12")
    elif args.retriever == "cov":
        cov = rs.retrieve_cov(cache, token_matrix, codec, args.question, args.topk, args.topw, doc_keep)
        if cov is None:
            print("coverage: no usable content terms (all stop words) -> falling back to v12")

    chunks_text = []
    if full is not None:
        ordered, per_hit, trace = full
        semantic = "n_windows" in trace                 # False => embed-degrade cov output
        # abstention (v1.6): full-stack only, real semantic scan only. Never fires on the
        # embed-degrade->cov output (no "abstain" key + semantic False). --no-abstain overrides.
        abstain = bool(semantic and trace.get("abstain") and not args.no_abstain)
        if semantic:
            print(f"semantic: {trace['n_windows']} windows, query embed {trace['embed_secs']:.2f}s, "
                  f"scan {trace['sem_scan_secs']:.2f}s, inject {trace['inject']}, cand {trace['n_cand']}")
            if args.sem_top:                              # flag-gated so default stdout is frozen
                print(f"sem_top {trace['sem_top']:.3f} (tau {trace['abstain_tau']:.3f})")
        tstr = " ".join(f"{t['word']}:{t['weight']:.0f}" for t in trace["terms"])
        print(f"coverage: {len(trace['terms'])} terms [{tstr}], {trace['n_scans']} scans "
              f"in {trace['scan_secs']:.2f}s")
        notes = []
        if trace["stopped"]:
            notes.append("stop=" + ",".join(trace["stopped"]))
        if trace["skipped"]:
            notes.append("no-term=" + ",".join(trace["skipped"]))
        if notes:
            print("coverage: " + "  ".join(notes))
        print(f"question  {args.question!r}")
        if abstain:
            print("=" * 60)
            print(f"NO GROUNDED ANSWER IN THE STORE (top semantic score {trace['sem_top']:.3f} < "
                  f"{trace['abstain_tau']:.3f} abstention threshold)")
            print("The candidates below are UNGROUNDED - shown for transparency only.")
            print("No context was sent to the model. Override with --no-abstain.")
            print("=" * 60)
        tag = "UNGROUNDED  " if abstain else ""
        for hit in per_hit:
            c = hit["chunk"]
            d = docs[int(cache.chunk_doc[c])]
            local = int(cache.chunk_local[c])
            klen = int(cache.chunk_len[c])
            with np.load(rs.page_fs_path(root, d["page"])) as npz:
                ids = npz["token_ids"][local * K:local * K + klen]   # v3: exact ingestion ids
            text = codec.ids_to_text([int(i) for i in ids])
            chunks_text.append(text)
            if semantic:
                print(f"\n[{tag}{d['name']}  chunk {local}/{d['n_chunks']}  page {d['page']}  "
                      f"score={hit['score']:.2f} cov={hit['chunk_cov']:.1f} "
                      f"cred={hit['cred']:.2f} sem={hit['sem']:.2f}]")
            else:
                print(f"\n[{d['name']}  chunk {local}/{d['n_chunks']}  page {d['page']}  "
                      f"cov={hit['chunk_cov']:.1f}]")
            print(text.strip())
        if abstain:                                     # skip context assembly + generation; exit 3
            sys.exit(3)
    elif cov is not None:
        ordered, per_hit, trace = cov
        tstr = " ".join(f"{t['word']}:{t['weight']:.0f}" for t in trace["terms"])
        print(f"coverage: {len(trace['terms'])} terms [{tstr}], {trace['n_scans']} scans "
              f"in {trace['scan_secs']:.2f}s")
        notes = []
        if trace["stopped"]:
            notes.append("stop=" + ",".join(trace["stopped"]))
        if trace["skipped"]:
            notes.append("no-term=" + ",".join(trace["skipped"]))
        if notes:
            print("coverage: " + "  ".join(notes))
        print(f"question  {args.question!r}")
        for hit in per_hit:
            c = hit["chunk"]
            d = docs[int(cache.chunk_doc[c])]
            local = int(cache.chunk_local[c])
            klen = int(cache.chunk_len[c])
            with np.load(rs.page_fs_path(root, d["page"])) as npz:
                ids = npz["token_ids"][local * K:local * K + klen]   # v3: exact ingestion ids
            text = codec.ids_to_text([int(i) for i in ids])
            chunks_text.append(text)
            print(f"\n[{d['name']}  chunk {local}/{d['n_chunks']}  page {d['page']}  "
                  f"cov={hit['chunk_cov']:.1f}]")
            print(text.strip())
    else:
        order, hams, dropped_uniq, fell_back = rs.retrieve_v12(
            cache, token_matrix, T_bits, codec, args.question, args.topk, args.df_max, doc_keep
        )
        if dropped_uniq:
            labels = ", ".join(repr(codec.ids_to_text([t])) for t in dropped_uniq)
            if fell_back:
                print(f"df-filter would empty the query -> using unfiltered bag (all high-DF: {labels})")
            else:
                print(f"df-filter dropped high-DF tokens (>{args.df_max:.2f} of chunks): {labels}")
        print(f"question  {args.question!r}")
        for c in order:
            c = int(c)
            h = int(hams[c])
            if h >= (1 << 62):
                continue                               # masked-out (--doc) sentinel row
            d = docs[int(cache.chunk_doc[c])]
            local = int(cache.chunk_local[c])
            klen = int(cache.chunk_len[c])
            z = rc.z_of(h)
            with np.load(rs.page_fs_path(root, d["page"])) as npz:
                ids = npz["token_ids"][local * K:local * K + klen]   # v3: exact ingestion ids
            text = codec.ids_to_text([int(i) for i in ids])
            chunks_text.append(text)
            print(f"\n[{d['name']}  chunk {local}/{d['n_chunks']}  page {d['page']}  ham={h}  z={z:.1f}]")
            print(text.strip())

    ctx = "\n\n".join(t.strip() for t in chunks_text)
    print("\n" + "=" * 60)
    models = ollama_models()
    if models is not None and DEMO_MODEL in models:
        prompt = PROMPT_TMPL.format(ctx=ctx, q=args.question)
        print(ollama_generate(prompt).strip())
    else:
        print("Assembled context (llama3.2:1b not available -> no generation):\n")
        print(ctx)
        print("\ndemo needs: ollama pull llama3.2:1b")
    sys.exit(0)


# ---------------------------------------------------------------- collections
def cmd_collections(args):
    """List collections (default + named) or set/clear a named collection's abstention tau.

    Listing is instant (NO ring load): docs/chunks/pages from plain JSON, cache/sidecar freshness
    from the metas. --set-tau/--clear-tau require a NAME, refuse 'default' (the wiki-calibrated
    legacy collection), and validate 0 < tau < 1."""
    if args.set_tau is not None and args.clear_tau:
        raise SystemExit("collections: --set-tau and --clear-tau are mutually exclusive")
    if args.set_tau is not None or args.clear_tau:
        if not args.name:
            raise SystemExit("collections --set-tau/--clear-tau needs a collection NAME "
                             f"(have: {', '.join(rs.list_collections(args.ring))})")
        try:
            name = rs.sanitize_collection(args.name)
        except rs.StoreError as e:
            raise SystemExit(str(e))
        if name == "default":
            raise SystemExit("refusing to set tau on 'default' (the legacy wiki-calibrated "
                             "collection) - tau overrides are for named collections only")
        root = rs.collection_root(args.ring, name)
        if not os.path.exists(rs.store_paths(root)[2]):
            raise SystemExit(f"collection {name!r} does not exist; "
                             f"have: {', '.join(rs.list_collections(args.ring))}")
        meta = rs.load_collection_meta(root)
        if args.clear_tau:
            meta.pop("sem_abstain_tau", None)
            rs.save_collection_meta(root, meta)
            print(f"collection {name!r} tau cleared -> {sm.SEM_ABSTAIN_TAU:.3f} (global default)")
        else:
            t = float(args.set_tau)
            if not (0.0 < t < 1.0):
                raise SystemExit(f"--set-tau must be in (0, 1); got {t}")
            meta["sem_abstain_tau"] = t
            rs.save_collection_meta(root, meta)
            print(f"collection {name!r} tau -> {t:.3f} (collection override)")
        return

    rows = collections_report(args.ring)
    if args.name:
        try:
            want = rs.sanitize_collection(args.name)
        except rs.StoreError as e:
            raise SystemExit(str(e))
        rows = [r for r in rows if r["name"] == want]
        if not rows:
            raise SystemExit(f"collection {args.name!r} does not exist; "
                             f"have: {', '.join(rs.list_collections(args.ring))}")
    print(f"{'collection':<16} {'docs':>5} {'chunks':>10} {'pages':>11}  {'cache':<9} "
          f"{'sidecar':<15} tau")
    for r in rows:
        if r["note"]:
            print(f"{r['name']:<16} {r['note']}")
            continue
        pages = f"{r['pages_bytes'] / 1048576:.1f} MiB"
        print(f"{r['name']:<16} {r['docs']:>5} {r['chunks']:>10,} {pages:>11}  {r['cache']:<9} "
              f"{r['sidecar']:<15} {r['tau']}")


def main():
    _reconfig_stdout()
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Decoder Ring demo CLI.")
    ap.add_argument("--ring", default=here, help="ring directory (default: script dir)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info"); p.add_argument("file"); p.set_defaults(fn=cmd_info)
    p = sub.add_parser("encode"); p.add_argument("text"); p.set_defaults(fn=cmd_encode)
    p = sub.add_parser("decode"); p.add_argument("vecfile"); p.add_argument("--noisy", action="store_true"); p.set_defaults(fn=cmd_decode)
    COL_HELP = "named collection (default: the legacy store\\; created on first ingest)"
    p = sub.add_parser("ingest"); p.add_argument("paths", nargs="+")
    p.add_argument("--add", action="store_true", help="incremental: append instead of a fresh rebuild (same resolved path REPLACES)")
    p.add_argument("--fanout", type=int, default=rc.DEFAULT_FANOUT)
    p.add_argument("--collection", default=None, help=COL_HELP); p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("remove"); p.add_argument("ident", help="doc name | doc_id | source path")
    p.add_argument("--collection", default=None, help=COL_HELP); p.set_defaults(fn=cmd_remove)
    p = sub.add_parser("stats")
    p.add_argument("--collection", default=None, help=COL_HELP); p.set_defaults(fn=cmd_stats)
    p = sub.add_parser("sembuild")
    p.add_argument("--status", action="store_true", help="print per-doc sidecar coverage without building (marks [slim] docs)")
    p.add_argument("--rebuild", action="store_true", help="re-derive keys + re-estimate mu (phase 2); re-embeds ONLY stale docs (never wipes emb/ or re-decodes fresh pages). On a SLIM sidecar it force-re-embeds every slimmed doc = un-slim (island required; shifts ALL keys, so re-run eval_ring.py --set AB and re-check the abstention block afterwards)")
    p.add_argument("--slim", action="store_true", help="remove the fp16 embed arrays (keys/provenance/text kept; queries byte-identical). Opt-in, one-way cheap; un-slim = --rebuild re-embed")
    p.add_argument("--collection", default=None, help=COL_HELP)
    p.add_argument("--ollama", default=None, help="Ollama endpoint override (else env RING_OLLAMA, else http://127.0.0.1:11436)")
    p.set_defaults(fn=cmd_sembuild)
    p = sub.add_parser("query"); p.add_argument("question"); p.add_argument("--topk", type=int, default=2)
    p.add_argument("--retriever", choices=["auto", "full", "cov", "v12"], default="auto",
                   help="auto = full semantic+coverage stack if the sidecar is fresh else cov (default); "
                        "full = force the stack; cov = v1.3 term coverage; v12 = frozen v1.2 majority-bag path")
    p.add_argument("--sem-weight", type=float, default=None,
                   help=f"semantic-channel weight in the full-stack score (default {sm.SEM_WEIGHT_DEFAULT})")
    p.add_argument("--topw", type=int, default=rs.COV_TOPW,
                   help=f"reserved: window-site count for the eval's query-time window diagnostic (default {rs.COV_TOPW}; unused by the shipped chunk-coverage path)")
    p.add_argument("--doc", default=None, help="restrict retrieval to docs whose name contains this substring (case-insensitive). Note: if no in-doc chunk scores any term coverage, cov returns no hits (v12 always returns the best-in-doc chunks)")
    p.add_argument("--no-abstain", action="store_true", help="answer even when the store shows no grounded answer (abstention off)")
    p.add_argument("--df-max", type=float, default=rs.DF_MAX_DEFAULT,
                   help=f"v12 retriever only: drop query tokens present in > this fraction of chunks (default {rs.DF_MAX_DEFAULT}; 1.0 disables). The cov retriever uses its stop list + IDF weights instead.")
    p.add_argument("--ollama", default=None, help="Ollama endpoint override (else env RING_OLLAMA, else http://127.0.0.1:11436)")
    p.add_argument("--collection", default=None, help=COL_HELP)
    p.add_argument("--sem-top", action="store_true",
                   help="print the top semantic window score + effective tau on the full-stack path "
                        "(nothing on cov/v12/degrade); handy for calibrating --collection tau")
    p.set_defaults(fn=cmd_query)
    p = sub.add_parser("collections", help="list collections, or set/clear a named collection's abstention tau")
    p.add_argument("name", nargs="?", default=None, help="one collection to show/modify (omit to list all)")
    p.add_argument("--set-tau", type=float, default=None, help="set this collection's abstention tau (0 < X < 1)")
    p.add_argument("--clear-tau", action="store_true", help="drop the tau override back to the global default")
    p.set_defaults(fn=cmd_collections)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
