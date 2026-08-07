r"""ring_store.py — Decoder Ring paged store v4 (manifest + per-doc ID pages + mmap cache).

WHAT THIS IS. The on-disk RAG store for the Decoder Ring, split out of ring_cli so
ring_core.py stays frozen. It replaces the v1 single `store.npz` blob with a
Minecraft-region-style paged layout (operator standing preference: paged snapshots, not a
growing monolith):

  store\
    manifest.json                  authoritative small index (human-readable)
    pages\<slug>-<doc_id>.npz      one self-contained page per document (v4: DEFLATE-compressed)
    cache\                         DERIVED, lazily rebuilt, safe to delete anytime
      bag_matrix.npy               (N,256) <u8 all level-0 bags, COMPOSED from ids (mmap target)
      chunk_doc / chunk_local / chunk_len .npy   per-chunk provenance arrays
      df.npy                       (128256,) uint32 corpus chunk document-frequency
      cache_meta.json              {manifest_sha256, n_chunks, doc_ids, built}

WHY IT LOOKS LIKE THIS.
  * v3/v4 pages store the document's exact ingestion token IDs (`token_ids` uint32) — NOT the
    hypervectors. Level-0 bag/seq vectors and the chunk tree are all DERIVED: the bag scan
    matrix is composed from ids at cache-rebuild time (compose_bags, a bit-sliced packed
    majority that is bit-exact vs rc.build_chunks); query/eval chunk text detokenizes the
    ids directly (killing the ~2 s/chunk NN decode); the pure-hypervector seq path lives on
    only as a verification mode (tests compose a seq bundle from ids and round-trip it). A
    page is 52x smaller than v2's stored-vector page yet still self-contained provenance:
    lose the manifest and a page still explains itself (ids + lens + sparse DF + meta).
  * DF is a hybrid: sparse (token_ids/counts) per page = the incremental source of truth
    (add/remove a doc = add/remove its contribution, no corpus rescan); a dense 513 KB
    df.npy in the cache gives O(1) per-query lookup. See ring_cli df filtering.
  * doc_id = blake2b-8 of the normcased absolute path => re-ingesting the same resolved
    path REPLACES (same id => same page filename => os.replace overwrite). Synthetic
    corpora use an `ids://<name>` pseudo-path (deterministic, no filesystem lookup).
  * Every write goes through atomic_write_bytes (serialize to memory, write .tmp-<pid>,
    os.replace with a WinError 5/32 retry backoff) — this box has transient AV/indexer
    file locks. np.savez APPENDS `.npz` to bare paths, so we never hand it a path; we
    savez into an io.BytesIO and hand the bytes to the atomic helper.

STORE v4 (compression-only, identical members). v4 writes the SAME v3 member names, dtypes,
shapes and semantics — but via np.savez_compressed (zipfile ZIP_DEFLATED/zlib) instead of
np.savez. np.load decompresses npz members transparently, so EVERY reader is byte-untouched
(the only cost is ~+1.3 ms/hit on the per-hit query page slice, measured). Measured on the
live 103-page store: 53.14 MB -> 18.63 MB (0.351x, 2.85x smaller). Doc-level random access is
preserved (one file per doc; members decompress independently on access). Page FILES are not
byte-reproducible (npz zip headers embed a timestamp — already true in v2/v3); all gates are
ARRAY equality + downstream stdout, never page-file bytes. 17-bit bit-packing was measured and
REJECTED (compressed-uint32 is ~11% SMALLER than compressed-17-bit-packed on every real page,
and needs no custom codec — see CHANGELOG v1.7).

HONEST CAVEATS. Single-writer (no cross-process locking). The chunk tree is no longer
stored (tree_levels is computed arithmetically); the flat bag scan is fast enough through
the design scale (see README). DF filtering is query-side only — the composed bag vectors
stay pure majority (bit-exact vs the frozen rc path), so the .dring contract is untouched.
A v2 store is one-way migrated to v3 by migrate_store_v3.py, then v3 -> v4 by
migrate_store_v4.py.

NAMED COLLECTIONS (v1.8). The legacy store\ is the DEFAULT collection; a named collection lives
at stores\<name>\store\{manifest.json, collection.json, pages\, cache\, semantic\} — a full,
isolated store tree under the SAME ring dir (one shared codebook). It is reached by handing the
COLLECTION ROOT (collection_root) to any rs/sm function in place of the ring dir. A collection is
DATA isolation, never a contract fork: K, D, window/stride and every retrieval constant stay the
frozen global contract. New collections are born store_version 4 (the migrate tools are
legacy-only). See collection_root / list_collections / *_collection_meta below.
"""

import collections
import datetime
import gc
import hashlib
import io
import json
import math
import os
import re
import time

import numpy as np

import ring_core as rc

# ---------------------------------------------------------------- store contract
STORE_VERSION = 4
STORE_FORMAT = "dring-store"
DF_MAX_DEFAULT = 0.25              # query-side high-DF token drop threshold (fraction)
WRITE_RETRIES = 5                  # os.replace attempts on transient WinError 5/32 locks
# the derived cache array members (besides cache_meta.json); a missing one => cache invalid
_CACHE_ARRAYS = ("bag_matrix.npy", "chunk_doc.npy", "chunk_local.npy", "chunk_len.npy", "df.npy")

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]")

# ---------------------------------------------------------------- coverage retriever (v1.3)
# Query-side term-coverage retrieval ("cov", the default in ring_cli). NO cache/page/manifest
# change: the stored keys stay binary (bark), all grading is query-side (trunk). See the v1.3
# plan sections 3-4 for the measured receipts behind every constant here.
#
# Closed-class WORD stop set (4.1): naive corpus IDF ranks rare interrogatives ("Which" w=42)
# above content words ("France" w=21), so a query-side stop list is mandatory alongside IDF.
COV_STOP_WORDS = frozenset("""
a an the of in on at to for from by with as it is its this that these those and or but
what which who whom whose where when why how are was were be been being does do did has
have had can could will would should shall may might must not no yes there here then than
into onto about over under most more many much very also only such same other another each
every all any some few between you your we our i me my he his she her they them their s t
""".split())
COV_RAW_SCALE = 25.0     # raw-arm membership scale (analytic member z at K=16 ~ 25.5, R8).
COV_WHITE_SCALE = 3.0    # whitened-arm membership scale. RECALIBRATED from the plan's 6.0 to
                         # the value MEASURED on the live store (v1.3 handoff): the corpus has
                         # a large ambient bag correlation (mean raw z 15-51 per token from
                         # global-T + byte composition), so a true single-occurrence member
                         # sits at whitened z ~ 2.5-3, not the ~4-6 the plan assumed. At 6.0
                         # membership never reached 1.0 and config D scored 0/20; at 3.0 it
                         # calibrates to member z. The raw arm binds only on repeat-heavy
                         # competitors (whitened z > ~5), gently capping them.
COV_WIN = 4              # query-time sliding window width, in consecutive chunks (R10)
COV_TOPW = 8             # sites picked by non-max suppression before final chunk ranking
COV_MAX_SCANS = 24       # cap on unique per-token scans per query (~0.55 s each, measured)
COV_MAX_PIECES = 8       # a term's exact form must encode to 1..8 token pieces to be kept


class StoreError(Exception):
    """Recoverable store-level failure (per-file ingest skip, bad ident, torn write)."""


Cache = collections.namedtuple("Cache", "bag chunk_doc chunk_local chunk_len df doc_ids manifest")


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- paths
def store_paths(ring_dir):
    """(store_dir, pages_dir, manifest_path, cache_dir) for a ring directory."""
    store_dir = os.path.join(ring_dir, "store")
    return (
        store_dir,
        os.path.join(store_dir, "pages"),
        os.path.join(store_dir, "manifest.json"),
        os.path.join(store_dir, "cache"),
    )


def page_fs_path(ring_dir, page_rel):
    """Resolve a manifest `page` value ("pages/<file>") to a real filesystem path."""
    store_dir = store_paths(ring_dir)[0]
    return os.path.join(store_dir, *page_rel.split("/"))


# ---------------------------------------------------------------- atomic write / remove (5.4)
def atomic_write_bytes(path, data, retries=WRITE_RETRIES):
    """Write `data` to a temp file in the same dir, then os.replace onto `path`.

    Retries os.replace on transient Windows locks (PermissionError, or OSError with
    winerror 5/32) with backoff 0.2*2**attempt (0.2/0.4/0.8/1.6/3.2 s). A non-transient
    OSError raises immediately. On final failure the temp is best-effort removed and a
    StoreError is raised.
    """
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + f".tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    last = None
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last = e
        except OSError as e:
            if getattr(e, "winerror", None) not in (5, 32):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise StoreError(f"atomic write failed for {path}: {e}") from e
            last = e
        if attempt < retries - 1:
            time.sleep(0.2 * (2 ** attempt))
    try:
        os.remove(tmp)
    except OSError:
        pass
    raise StoreError(f"atomic write failed for {path} after {retries} retries: {last}")


def atomic_remove(path, retries=WRITE_RETRIES):
    """Delete a file with the same retry backoff. Missing file = success; a final failure
    prints a warning (orphan pages/cache files are ignored by every reader) and returns
    False rather than raising."""
    last = None
    for attempt in range(retries):
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except OSError as e:
            last = e
        if attempt < retries - 1:
            time.sleep(0.2 * (2 ** attempt))
    print(f"warning: could not remove {os.path.basename(path)}: {last} (orphan ignored by readers)")
    return False


def _save_npy_atomic(path, arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    atomic_write_bytes(path, buf.getvalue())


# ---------------------------------------------------------------- text extraction (5.7)
def extract_text(path):
    """(text, kind) for .txt/.md/.pdf. Raises StoreError for unsupported types or a
    missing pypdf; the CLI catches StoreError per-file and continues."""
    ext = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)
    if ext in (".txt", ".md"):
        kind = "md" if ext == ".md" else "txt"
        try:
            with open(path, encoding="utf-8") as f:
                return f.read(), kind
        except UnicodeDecodeError:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            print(f"note: {name}: not clean UTF-8, ingested with replacements")
            return text, kind
        except OSError as e:                        # missing/unreadable file -> per-file skip
            raise StoreError(f"{name}: skipped - {e.strerror or e}") from e
    if ext == ".pdf":
        try:
            import pypdf
        except ImportError:
            raise StoreError(f"{name}: skipped - .pdf needs pypdf (pip install pypdf)")
        try:
            reader = pypdf.PdfReader(path)
            return "\n".join((p.extract_text() or "") for p in reader.pages), "pdf"
        except OSError as e:
            raise StoreError(f"{name}: skipped - {e.strerror or e}") from e
        except Exception as e:                      # corrupt/unparsable PDF -> per-file skip
            raise StoreError(f"{name}: skipped - unreadable pdf ({e})") from e
    raise StoreError(f"{name}: skipped - unsupported type {ext!r}")


# ---------------------------------------------------------------- identity / naming (5.2)
def doc_identity(source):
    """(doc_id, normpath). Real paths -> normcase(abspath) (case-insensitive FS identity);
    `ids://<name>` synthetic sources are kept verbatim (deterministic, no fs lookup)."""
    if source.startswith("ids://"):
        normpath = source
    else:
        normpath = os.path.normcase(os.path.abspath(source))
    doc_id = hashlib.blake2b(normpath.encode("utf-8"), digest_size=8).hexdigest()
    return doc_id, normpath


def page_name(name, doc_id):
    """Relative page path "pages/<slug>-<doc_id>.npz"; slug is name filtered to
    [A-Za-z0-9._-], max 40 chars (human debuggability only — identity lives in doc_id)."""
    slug = _SLUG_RE.sub("", name)[:40] or "doc"
    return f"pages/{slug}-{doc_id}.npz"


# ---------------------------------------------------------------- named collections (v1.8)
# A collection is DATA isolation only via root aliasing: the legacy store\ stays the DEFAULT
# collection (ZERO migration/writes), and a named collection lives at stores\<name>\store\...
# with its own manifest/pages/cache (incl. the DF table)/semantic sidecar/abstention tau. Every
# rs/sm function keeps its exact signature and is simply handed the COLLECTION ROOT (the dir that
# holds a store\ tree) where it used to be handed the ring dir — because rs/sm resolve store paths
# ONLY through store_paths/page_fs_path/manifest_sha, the isolation falls out by construction. The
# codebook (atoms/tokens/tokenizer), K, D, window/stride and every retrieval constant are the
# frozen global contract shared by all collections: a collection is DATA isolation, never a
# contract fork.
COLLECTIONS_DIR = "stores"
_COLLECTION_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
# Windows reserved device basenames (all regex-legal): a stores\con\ etc. would fail oddly at
# directory-creation time, so reject them up front (audit r1 nit).
_WIN_RESERVED = frozenset(
    ["con", "prn", "aux", "nul"]
    + [f"com{i}" for i in range(1, 10)]
    + [f"lpt{i}" for i in range(1, 10)]
)


def sanitize_collection(name):
    """Normalize + validate a collection name: strip, case-fold to lowercase. "default" passes
    through as the reserved legacy alias. Anything else must match _COLLECTION_RE (and not be a
    Windows reserved device name) or raise StoreError naming the rule (reject, never munge)."""
    n = name.strip().casefold()
    if n == "default":
        return "default"
    if not _COLLECTION_RE.match(n):
        raise StoreError(
            f"invalid collection name {name!r}: use 1-32 chars of a-z 0-9 _ - , starting with "
            f"a letter or digit ('default' is reserved for the legacy store)"
        )
    if n in _WIN_RESERVED:
        raise StoreError(
            f"collection name {name!r} is a reserved Windows device name ("
            f"con/prn/aux/nul/com1-9/lpt1-9) - pick another"
        )
    return n


def collection_root(ring_dir, collection=None):
    r"""The directory that holds a store\ tree for `collection`.

    None / "" / "default" -> ring_dir itself (the legacy DEFAULT collection; zero migration).
    A named collection -> <ring_dir>\stores\<name> (name sanitized). Hand the result to any
    rs/sm function where it expects a ring dir (store_paths/semantic_paths join store\ under it).
    """
    if collection is None or collection == "":
        return ring_dir
    name = sanitize_collection(collection)
    if name == "default":
        return ring_dir
    return os.path.join(ring_dir, COLLECTIONS_DIR, name)


def list_collections(ring_dir):
    r"""["default"] + sorted named collections: subdirs of <ring_dir>\stores\ that already hold a
    store\manifest.json (a bare stores\<name>\ dir without a manifest is not yet a collection).
    Missing stores\ -> ["default"]."""
    names = ["default"]
    base = os.path.join(ring_dir, COLLECTIONS_DIR)
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            if os.path.isfile(os.path.join(base, name, "store", "manifest.json")):
                names.append(name)
    return names


def collection_meta_path(store_root):
    r"""Path to <store_root>\store\collection.json (freshness-NEUTRAL sidecar for a collection's
    name/created stamp + optional sem_abstain_tau — deliberately NOT the store manifest, whose
    sha chains the caches)."""
    return os.path.join(store_paths(store_root)[0], "collection.json")


def load_collection_meta(store_root):
    """Read collection.json for a collection root; absent -> {}; unparseable -> one printed
    warning + {} (never crash a query on a torn/hand-edited file)."""
    path = collection_meta_path(store_root)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"warning: could not read {os.path.basename(path)}: {e} - using collection defaults")
        return {}


def save_collection_meta(store_root, meta):
    """Atomic-write collection.json (plain JSON via atomic_write_bytes; NOT np.savez which would
    append .npz to a bare path)."""
    atomic_write_bytes(collection_meta_path(store_root),
                       json.dumps(meta, indent=2).encode("utf-8"))


# ---------------------------------------------------------------- manifest (5.2)
def manifest_sha(ring_dir):
    """sha256 over manifest.json bytes — the cache-invalidation key (5.5)."""
    manifest_path = store_paths(ring_dir)[2]
    with open(manifest_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_manifest(ring_dir):
    """Load manifest.json; a missing manifest returns a fresh skeleton (fanout=None until
    the first doc). Wrong format/store_version => StoreError."""
    manifest_path = store_paths(ring_dir)[2]
    if not os.path.exists(manifest_path):
        now = _now_iso()
        return {
            "format": STORE_FORMAT, "store_version": STORE_VERSION,
            "D": rc.D, "fanout": None, "created": now, "updated": now,
            "docs": [], "df_summary": {"n_chunks": 0, "n_distinct_tokens": 0, "top": []},
        }
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    if m.get("format") != STORE_FORMAT or m.get("store_version") != STORE_VERSION:
        hint = ""
        if m.get("format") == STORE_FORMAT and m.get("store_version") == 2:
            hint = " -- v2 store? run: python migrate_store_v3.py then python migrate_store_v4.py"
        elif m.get("format") == STORE_FORMAT and m.get("store_version") == 3:
            hint = " -- v3 store? run: python migrate_store_v4.py"
        raise StoreError(
            f"manifest format/version mismatch: {m.get('format')}/{m.get('store_version')} "
            f"(need {STORE_FORMAT}/{STORE_VERSION}){hint}"
        )
    return m


def save_manifest(ring_dir, manifest, codec=None):
    """Recompute df_summary from the pages' sparse DF members, stamp `updated`, atomic write.

    df_summary is HUMAN provenance only (the authoritative dense DF lives in the cache); it
    is cheap to rebuild — random-access reads of two small zip members per page. `codec`
    renders the top-10 token labels; without it the labels are raw ids."""
    manifest_path = store_paths(ring_dir)[2]
    dense = np.zeros(rc.VOCAB, dtype=np.int64)
    total_chunks = 0
    for d in manifest["docs"]:
        with np.load(page_fs_path(ring_dir, d["page"])) as npz:
            ids = npz["df_token_ids"]
            cnts = npz["df_counts"]
            if ids.size:
                dense[ids] += cnts
        total_chunks += int(d["n_chunks"])
    nz = np.nonzero(dense)[0]
    top = []
    if nz.size and total_chunks:
        for tid in nz[np.argsort(dense[nz])[::-1][:10]]:
            frac = round(float(dense[tid]) / total_chunks, 4)
            label = repr(codec.ids_to_text([int(tid)])) if codec is not None else str(int(tid))
            top.append([label, frac])
    manifest["df_summary"] = {
        "n_chunks": total_chunks, "n_distinct_tokens": int(nz.size), "top": top,
    }
    manifest["updated"] = _now_iso()
    atomic_write_bytes(manifest_path, json.dumps(manifest, indent=2).encode("utf-8"))


# ---------------------------------------------------------------- bag composition (v3, 3.4)
def compose_bags(ids, token_matrix, T_packed, T_bits, K=rc.DEFAULT_FANOUT, block=65536):
    """(n_chunks, WORDS) <u8 level-0 bag matrix composed from one doc's token ids.

    Bit-exact vs rc.build_chunks's bag rows (rc._bundle_bag = pack(maj(members, T))). The
    hot path is a bit-sliced ripple-carry adder over the PACKED uint64 token words (no
    unpacking): five carry planes count the K members per bit; count>8 => 1, count==8 (the
    K=16 even tie) => T's bit. The final partial tail chunk (len < K, or odd/one-element
    semantics) defers to rc.maj so tie/majority rules are inherited, never re-derived.
    Measured ~10.6k chunks/s (~4x the naive unpackbits path); public so migrate_store_v3
    and the tests can compose bags directly. `T_packed = rc.pack(T_bits)`."""
    ids = np.asarray(ids, dtype=np.int64)
    n = len(ids)
    n_chunks = (n + K - 1) // K
    n_full = n // K
    out = np.empty((n_chunks, rc.WORDS), dtype="<u8")
    for s in range(0, n_full, block):                       # bound transient memory
        e = min(s + block, n_full)
        toks = token_matrix[ids[s * K:e * K]].reshape(e - s, K, rc.WORDS)
        c0 = np.zeros((e - s, rc.WORDS), dtype=np.uint64)
        c1 = np.zeros_like(c0); c2 = np.zeros_like(c0); c3 = np.zeros_like(c0); c4 = np.zeros_like(c0)
        for k in range(K):                                  # 5-plane bit-sliced counter, max 16
            x = toks[:, k, :].astype(np.uint64)
            for c in (c0, c1, c2, c3):
                carry = c & x
                np.bitwise_xor(c, x, out=c)
                x = carry
            np.bitwise_or(c4, x, out=c4)
        low = c2 | c1 | c0
        maj = c4 | (c3 & low)                               # count > 8
        tie = c3 & ~low & ~c4                               # count == 8 (K even) -> T bits
        out[s:e] = (maj | (tie & T_packed)).view("<u8")
    if n_chunks > n_full:                                   # partial tail chunk: frozen rc path
        tail = [int(i) for i in ids[n_full * K:]]
        out[n_full] = rc.pack(rc.maj([rc.unpack(token_matrix[i]) for i in tail], T_bits))
    return out


# ---------------------------------------------------------------- page building (5.2)
def build_page_arrays(ids, K=rc.DEFAULT_FANOUT):
    """All page npz array members for a document's token ids (store v3).

    v3 stores the raw ingestion token IDs (`token_ids` uint32) + per-chunk lengths + a
    sorted sparse per-doc DF. It stores NO hypervectors: level-0 bag/seq and the chunk tree
    are DERIVED (bags via compose_bags at cache-rebuild; seq via rc._bundle_seq in the
    verification mode). `tree_levels` is the fan-out-K reduction depth, computed
    arithmetically (verified == rc.build_tree's length for every live doc). Callers pass
    token_matrix/T_bits into add_document for the store contract, but page-building no
    longer needs them (3.1/3.4)."""
    ids = np.asarray(ids, dtype=np.int64)
    n = int(ids.shape[0])
    n_chunks = (n + K - 1) // K if n else 0
    chunk_len = np.array([min(K, n - start) for start in range(0, n, K)], dtype=np.int32)

    df = {}
    for start in range(0, n, K):
        for tid in np.unique(ids[start:start + K]):
            t = int(tid)
            df[t] = df.get(t, 0) + 1
    if df:
        df_ids = np.array(sorted(df), dtype=np.int32)
        df_counts = np.array([df[int(t)] for t in df_ids], dtype=np.uint32)
    else:
        df_ids = np.empty((0,), dtype=np.int32)
        df_counts = np.empty((0,), dtype=np.uint32)

    lv, nc = 0, n_chunks                                    # depth of the fan-out-K tree
    while nc > 1:
        nc = (nc + K - 1) // K
        lv += 1

    return {
        "token_ids": ids.astype("<u4"), "chunk_len": chunk_len,
        "df_token_ids": df_ids, "df_counts": df_counts,
        "n_chunks": int(n_chunks), "tree_levels": int(lv),
    }


def add_document(ring_dir, *, path=None, ids=None, name=None, token_matrix, T_bits, codec,
                 K=rc.DEFAULT_FANOUT):
    """Ingest ONE document (path mode or ids mode) into the store; returns its manifest entry.

    Exactly one of path/ids. Same resolved identity => REPLACE (same doc_id/page filename).
    Crash ordering (5.4): page written first, manifest second — an orphan page is harmless.
    """
    if (path is None) == (ids is None):
        raise StoreError("add_document: pass exactly one of path=/ids=")
    manifest = load_manifest(ring_dir)
    if manifest.get("fanout") is not None and int(manifest["fanout"]) != int(K):
        raise StoreError(
            f"store fanout is {manifest['fanout']}, refusing to add with fanout {K} "
            f"(mixed-K stores are not supported)"
        )

    if path is not None:
        text, kind = extract_text(path)                     # OSError already -> StoreError here
        doc_name = name or os.path.basename(path)
        token_ids = codec.encode_ids(text)
        source = path
        try:
            with open(path, "rb") as f:
                source_sha = hashlib.sha256(f.read()).hexdigest()
        except OSError as e:                                 # e.g. file removed between read + sha
            raise StoreError(f"{doc_name}: skipped - {e.strerror or e}") from e
    else:
        if name is None:
            raise StoreError("add_document: ids mode requires name=")
        doc_name = name
        token_ids = np.asarray(ids, dtype=np.int64)
        kind = "ids"
        source = f"ids://{name}"
        source_sha = hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()

    n_tokens = int(len(token_ids))
    if n_tokens == 0:
        raise StoreError(f"{doc_name}: skipped - 0 tokens (empty extraction)")

    doc_id, normpath = doc_identity(source)
    arrays = build_page_arrays(token_ids, K)      # v3: token_matrix/T_bits unused by page-building
    ingested = _now_iso()
    meta = {
        "doc_id": doc_id, "name": doc_name, "source_path": normpath,
        "source_sha256": source_sha, "source_kind": kind, "n_tokens": n_tokens,
        "n_chunks": arrays["n_chunks"], "tree_levels": arrays["tree_levels"],
        "fanout": int(K), "D": rc.D, "store_version": STORE_VERSION, "ingested": ingested,
    }
    page_rel = page_name(doc_name, doc_id)

    buf = io.BytesIO()
    np.savez_compressed(                      # v4: deflate the identical v3 members (0.351x measured;
        buf,                                  # np.load decompresses transparently, so no reader changes)
        token_ids=arrays["token_ids"], chunk_len=arrays["chunk_len"],
        df_token_ids=arrays["df_token_ids"], df_counts=arrays["df_counts"],
        meta_json=np.array(json.dumps(meta)),
    )
    atomic_write_bytes(page_fs_path(ring_dir, page_rel), buf.getvalue())

    entry = {
        "doc_id": doc_id, "name": doc_name, "source_path": normpath,
        "source_sha256": source_sha, "source_kind": kind, "n_tokens": n_tokens,
        "n_chunks": arrays["n_chunks"], "tree_levels": arrays["tree_levels"],
        "page": page_rel, "ingested": ingested,
    }
    if manifest.get("fanout") is None:
        manifest["fanout"] = int(K)
    manifest["D"] = rc.D
    docs = manifest["docs"]
    stale_page = None
    for i, d in enumerate(docs):
        if d["doc_id"] == doc_id:
            if d["page"] != page_rel:            # REPLACE whose display name (slug) changed
                stale_page = d["page"]           # -> new page filename; drop the old one
            docs[i] = entry
            break
    else:
        docs.append(entry)
    save_manifest(ring_dir, manifest, codec=codec)
    if stale_page is not None:                    # manifest no longer references it; delete after save
        atomic_remove(page_fs_path(ring_dir, stale_page))
    return entry


def remove_document(ring_dir, ident, codec=None):
    """Remove a doc by ident (priority: exact doc_id, exact name, normcase abspath).

    Manifest first, then page delete (5.4): a removed-then-orphaned page is ignored by all
    readers. Prints the removal line BEFORE acting (explain-before-removing)."""
    manifest = load_manifest(ring_dir)
    docs = manifest["docs"]

    match = next((d for d in docs if d["doc_id"] == ident), None)
    if match is None:
        named = [d for d in docs if d["name"] == ident]
        if len(named) > 1:
            raise StoreError(
                f"ambiguous name {ident!r}: matches doc_ids {[d['doc_id'] for d in named]} - "
                f"remove by doc_id"
            )
        if named:
            match = named[0]
    if match is None:
        np_ident = os.path.normcase(os.path.abspath(ident))
        match = next((d for d in docs if d["source_path"] == np_ident), None)
    if match is None:
        raise StoreError(f"no doc matches {ident!r} (name|doc_id|path); have {[d['name'] for d in docs]}")

    print(f"removing doc {match['name']} ({match['page']}, {match['n_chunks']} chunks)")
    manifest["docs"] = [d for d in docs if d["doc_id"] != match["doc_id"]]
    if not manifest["docs"]:
        manifest["fanout"] = None
    save_manifest(ring_dir, manifest, codec=codec)
    atomic_remove(page_fs_path(ring_dir, match["page"]))
    return match


def fresh_store(ring_dir):
    """Delete pages/cache/manifest (retry-tolerant, per file) and recreate empty dirs.
    NEVER rmtree anything outside store\\; legacy store.npz/docs.json at the ring root are
    untouched."""
    store_dir, pages_dir, manifest_path, cache_dir = store_paths(ring_dir)
    for d in (pages_dir, cache_dir):
        if os.path.isdir(d):
            for fn in os.listdir(d):
                atomic_remove(os.path.join(d, fn))
    atomic_remove(manifest_path)
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)


# ---------------------------------------------------------------- cache (5.5)
def _rebuild_cache(ring_dir, manifest, sha, token_matrix, T_bits):
    _, _, _, cache_dir = store_paths(ring_dir)
    os.makedirs(cache_dir, exist_ok=True)
    docs = manifest["docs"]
    total = sum(int(d["n_chunks"]) for d in docs)
    K = int(manifest["fanout"]) if manifest.get("fanout") else rc.DEFAULT_FANOUT
    print(f"cache stale -> rebuilding ({len(docs)} docs, {total} chunks)")

    T_packed = rc.pack(T_bits)                              # v3: bags COMPOSED from ids per doc
    bag_parts = []
    chunk_doc = np.empty(total, dtype=np.int32)
    chunk_local = np.empty(total, dtype=np.int32)
    chunk_len = np.empty(total, dtype=np.int32)
    df = np.zeros(rc.VOCAB, dtype=np.uint32)
    doc_ids = []
    off = 0
    for di, d in enumerate(docs):
        with np.load(page_fs_path(ring_dir, d["page"])) as npz:
            b = compose_bags(npz["token_ids"], token_matrix, T_packed, T_bits, K)
            n = b.shape[0]
            bag_parts.append(b)
            chunk_doc[off:off + n] = di
            chunk_local[off:off + n] = np.arange(n, dtype=np.int32)
            chunk_len[off:off + n] = npz["chunk_len"]
            dfi = npz["df_token_ids"]
            if dfi.size:
                df[dfi] += npz["df_counts"]
        doc_ids.append(d["doc_id"])
        off += n
    bag = np.concatenate(bag_parts, axis=0) if bag_parts else np.empty((0, rc.WORDS), dtype="<u8")

    # Free any unreferenced prior bag mmap before we os.replace over bag_matrix.npy (a still
    # memory-mapped file cannot be replaced on Windows; see ensure_cache's note).
    gc.collect()
    # arrays first, meta LAST -> a torn cache always looks stale and rebuilds
    _save_npy_atomic(os.path.join(cache_dir, "bag_matrix.npy"), bag)
    _save_npy_atomic(os.path.join(cache_dir, "chunk_doc.npy"), chunk_doc)
    _save_npy_atomic(os.path.join(cache_dir, "chunk_local.npy"), chunk_local)
    _save_npy_atomic(os.path.join(cache_dir, "chunk_len.npy"), chunk_len)
    _save_npy_atomic(os.path.join(cache_dir, "df.npy"), df)
    meta = {"manifest_sha256": sha, "n_chunks": int(total), "doc_ids": doc_ids, "built": _now_iso()}
    atomic_write_bytes(
        os.path.join(cache_dir, "cache_meta.json"), json.dumps(meta, indent=2).encode("utf-8")
    )


def ensure_cache(ring_dir, token_matrix=None, T_bits=None):
    """Return a Cache namedtuple, rebuilding the derived cache from pages if the manifest
    content-hash moved (5.5). `bag` is an mmap over cache\\bag_matrix.npy.

    v3: a rebuild COMPOSES the bag matrix from the pages' token ids, so it needs
    token_matrix/T_bits. A fresh-cache load never rebuilds and works without them (stats /
    remove / sembuild-status paths). If a rebuild is needed and the matrices are absent, a
    StoreError is raised rather than silently returning a stale cache.

    WINDOWS NOTE: the returned `bag` mmap keeps bag_matrix.npy open, and a memory-mapped
    file cannot be os.replace'd. The CLI is single-shot (one ensure_cache per process) so it
    is unaffected; a long-lived caller (tests, bench) that mutates the store and re-ensures
    must drop the previous Cache first (`del cache; gc.collect()`) or the rebuild's atomic
    write of bag_matrix.npy will exhaust its retries and raise StoreError. Only bag_matrix is
    mmap'd — the other cache arrays are read fully and hold no handle."""
    _, _, manifest_path, cache_dir = store_paths(ring_dir)
    if not os.path.exists(manifest_path):
        raise StoreError("no store found - run ingest first")
    manifest = load_manifest(ring_dir)
    sha = manifest_sha(ring_dir)
    meta_path = os.path.join(cache_dir, "cache_meta.json")

    # cache is fresh only if meta parses, matches the manifest sha, AND every array member
    # is present (a hand-deleted array => rebuild, so "delete any cache file" is safe — F7)
    fresh = False
    have_arrays = all(os.path.exists(os.path.join(cache_dir, a)) for a in _CACHE_ARRAYS)
    if have_arrays and os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                fresh = json.load(f).get("manifest_sha256") == sha
        except (json.JSONDecodeError, OSError):
            fresh = False
    if not fresh:
        if token_matrix is None or T_bits is None:
            raise StoreError("cache is stale and ensure_cache was called without "
                             "token_matrix/T_bits (store v3 composes bags from ids)")
        _rebuild_cache(ring_dir, manifest, sha, token_matrix, T_bits)

    bag = np.load(os.path.join(cache_dir, "bag_matrix.npy"), mmap_mode="r")
    chunk_doc = np.load(os.path.join(cache_dir, "chunk_doc.npy"))
    chunk_local = np.load(os.path.join(cache_dir, "chunk_local.npy"))
    chunk_len = np.load(os.path.join(cache_dir, "chunk_len.npy"))
    df = np.load(os.path.join(cache_dir, "df.npy"))
    with open(meta_path, encoding="utf-8") as f:
        doc_ids = json.load(f)["doc_ids"]
    return Cache(bag, chunk_doc, chunk_local, chunk_len, df, doc_ids, manifest)


# ---------------------------------------------------------------- retrieval (5.6)
def df_filter(q_ids, df, n_chunks, df_max):
    """(kept_ids, dropped_ids): drop query tokens whose corpus chunk-DF fraction exceeds
    df_max. Multiplicity is preserved. If filtering would empty the query, fall back to ALL
    q_ids (the dropped list is still reported). df_max >= 1.0 disables filtering."""
    if n_chunks <= 0:
        return list(q_ids), []
    kept, dropped = [], []
    for tid in q_ids:
        if float(df[tid]) / n_chunks <= df_max:
            kept.append(tid)
        else:
            dropped.append(tid)
    if not kept:
        return list(q_ids), dropped
    return kept, dropped


def flat_scan(bag_mmap, q_bag_words, block=rc.NN_BLOCK):
    """Blocked Hamming scan of a query bag against the mmap bag matrix (mirrors nn_decode).
    Returns an int64 (N,) hamming array."""
    q = np.ascontiguousarray(q_bag_words, dtype="<u8")
    n = bag_mmap.shape[0]
    hams = np.empty(n, dtype=np.int64)
    for start in range(0, n, block):
        blk = bag_mmap[start:start + block]
        hams[start:start + blk.shape[0]] = np.bitwise_count(blk ^ q).sum(axis=1, dtype=np.int64)
    return hams


def retrieve_v12(cache, token_matrix, T_bits, codec, question, topk, df_max, doc_keep):
    """v1.2 retrieval core, extracted verbatim from cmd_query (df filter + majority bag +
    flat Hamming scan + optional --doc mask + argsort topk). Returns
    (order, hams, dropped_uniq, fell_back) so the CLI renders byte-identical v1.2 output and
    T23 can assert the ordering. The empty-question guard stays in the CLI (as in v1.2)."""
    q_ids = codec.encode_ids(question)
    n_chunks = int(cache.bag.shape[0])
    kept, dropped = df_filter(q_ids, cache.df, n_chunks, df_max)
    dropped_uniq, fell_back = [], False
    if dropped:
        seen = set()
        dropped_uniq = [t for t in dropped if not (t in seen or seen.add(t))]
        fell_back = len(kept) == len(q_ids)   # df_filter returns full q_ids on empty-fallback
    q_bag = rc.pack(rc.maj([rc.unpack(token_matrix[i]) for i in kept], T_bits))
    hams = flat_scan(cache.bag, q_bag)
    if doc_keep is not None:
        keep_arr = np.isin(cache.chunk_doc, np.array(sorted(doc_keep), dtype=np.int32))
        hams = np.where(keep_arr, hams, np.int64(1 << 62))
    order = np.argsort(hams)[:topk]
    return order, hams, dropped_uniq, fell_back


# ---------------------------------------------------------------- coverage retriever core (4.1-4.4)
CovResult = collections.namedtuple(
    "CovResult",
    "terms stopped skipped scan_ids tok2idx scan_secs Zraw phi "
    "term_phi_chunk term_phi_win chunk_cov win_cov win_ccsum valid Nw",
)


def _cov_prep_terms(question, codec, df, n_chunks):
    """Extract query terms, one EXACT surface form each (4.1).

    Returns (terms, stopped, skipped): `terms` is a list of dicts
    {word, weight, variants:[tuple(ids)], vdf:[int]} in question order; `stopped` is the
    closed-class words removed; `skipped` is content words that produced no in-corpus token.

    DEVIATION FROM PLAN 4.2 (measured, documented in the v1.3 handoff): the plan's surface
    variant expansion (case + plural s-toggle, up to 6 per term, term = MAX over variants) was
    MEASURED to LOWER accuracy on this store — natural hit@3 fell 4/20 -> 2/20 as soon as any
    variant beyond the exact running-text form was added, because MAX-over-variants credits a
    term at chunks that merely contain a *different* word ("Specials"/"seeds" crediting
    "special"/"seed"). So each term keeps a single exact form: the running-text " "+word, or
    the bare word only when the space form has an out-of-corpus (df=0) piece. This also halves
    scan cost. Multi-piece words (e.g. " kilometres" -> " kilom"+"etres") stay one variant with
    several pieces (soft-AND over pieces), preserving the plan's piece handling."""
    N = max(1, int(n_chunks))
    words = re.findall(r"[A-Za-z0-9_.]+", question)
    terms, stopped, skipped, seen = [], [], [], set()
    for raw in words:
        w = raw.rstrip(".")                          # strip trailing dots (4.1)
        if not w:
            continue
        key = w.lower()
        if key in seen:                              # dedupe preserving order (case-insensitive)
            continue
        seen.add(key)
        if key in COV_STOP_WORDS:                    # word-level stop test (4.1)
            stopped.append(w)
            continue
        chosen = None
        for probe in (" " + w, w):                   # running-text form first, bare as fallback
            ids = codec.encode_ids(probe)
            if not (1 <= len(ids) <= COV_MAX_PIECES):
                continue
            dfs = [int(df[i]) for i in ids]
            if any(d == 0 for d in dfs):             # a df=0 piece can never match -> skip form
                continue
            chosen = (tuple(int(i) for i in ids), min(dfs))
            break
        if chosen is None:
            skipped.append(w)
            continue
        ids_tuple, term_df = chosen
        terms.append({
            "word": w, "weight": max(1.0, 4.0 * math.log(N / term_df)),
            "variants": [ids_tuple], "vdf": [term_df],
        })

    # defensive scan cap: with one variant per term this rarely binds; if a query's unique
    # token pieces still exceed COV_MAX_SCANS, drop the highest-df (least discriminative) terms.
    def uniq_tokens(ts):
        return {i for t in ts for ids in t["variants"] for i in ids}

    if len(uniq_tokens(terms)) > COV_MAX_SCANS:
        keep, acc = [], set()
        for t in sorted(terms, key=lambda t: t["vdf"][0]):        # rarest terms first
            nxt = acc | set(t["variants"][0])
            if len(nxt) > COV_MAX_SCANS and keep:
                continue
            acc, _ = nxt, keep.append(t)
        kept_ids = {id(t) for t in keep}
        terms = [t for t in terms if id(t) in kept_ids]           # restore question order
    return terms, stopped, skipped


def _cov_scan(cache, token_matrix, ids):
    """Batched Hamming z-scan of many single tokens against the bag mmap (R13).

    `ids` is a list of unique token ids. Returns Zraw float32 (nt, N) in z units:
    (D/2 - ham) / (sqrt(D)/2). Each 65,536-row block is read once and shared across all
    tokens (the block read, not the popcount, is the win)."""
    bag = cache.bag
    N = int(bag.shape[0])
    nt = len(ids)
    q = np.ascontiguousarray(token_matrix[np.asarray(ids, dtype=np.int64)], dtype="<u8")
    Zraw = np.empty((nt, N), dtype=np.float32)
    half = rc.D / 2.0
    scale = np.sqrt(rc.D) / 2.0
    block = 65536
    for start in range(0, N, block):
        blk = np.ascontiguousarray(bag[start:start + block])     # (b, WORDS), read once
        b = blk.shape[0]
        for ti in range(nt):
            ham = np.bitwise_count(blk ^ q[ti]).sum(axis=1, dtype=np.int64)
            Zraw[ti, start:start + b] = (half - ham).astype(np.float32) / scale
    return Zraw


def cov_compute(cache, token_matrix, codec, question, doc_keep=None,
                raw_scale=None, white_scale=None):
    """One scan pass -> the full coverage state (4.3): dual-normalized soft membership,
    per-term chunk/window coverage, sliding-window co-occurrence, and the valid-window mask.

    Returns a CovResult, or None when the query has no usable content term (caller falls back
    to v12). Config C (chunk-only) and config D (windows) both derive from ONE CovResult."""
    raw_scale = COV_RAW_SCALE if raw_scale is None else float(raw_scale)
    white_scale = COV_WHITE_SCALE if white_scale is None else float(white_scale)
    N = int(cache.bag.shape[0])
    terms, stopped, skipped = _cov_prep_terms(question, codec, cache.df, N)
    if not terms:
        return None

    scan_ids = sorted({int(i) for t in terms for ids in t["variants"] for i in ids})
    tok2idx = {tid: k for k, tid in enumerate(scan_ids)}
    t0 = time.perf_counter()
    Zraw = _cov_scan(cache, token_matrix, scan_ids)
    scan_secs = time.perf_counter() - t0

    mean = Zraw.mean(axis=1, keepdims=True)
    std = Zraw.std(axis=1, keepdims=True)
    Zw = (Zraw - mean) / (std + 1e-9)
    # dual-normalized soft membership: the raw arm suppresses whitened noise on ultra-rare
    # tokens, the whitened arm suppresses ambient-correlated short tokens (R6/R7). min() is
    # load-bearing.
    phi = np.clip(np.minimum(Zraw / raw_scale, Zw / white_scale), 0.0, 1.0).astype(np.float32)

    Nw = N - (COV_WIN - 1)
    wm = phi[:, 0:Nw].copy()                          # per-token max over the 4 window chunks
    for shift in range(1, COV_WIN):
        np.maximum(wm, phi[:, shift:shift + Nw], out=wm)

    n_terms = len(terms)
    term_phi_chunk = np.zeros((n_terms, N), dtype=np.float32)
    term_phi_win = np.zeros((n_terms, Nw), dtype=np.float32)
    for ti, t in enumerate(terms):
        tc, tw = term_phi_chunk[ti], term_phi_win[ti]
        for ids in t["variants"]:
            rows = [tok2idx[int(i)] for i in ids]     # variant = soft-AND over pieces (min)
            if len(rows) == 1:
                vc, vw = phi[rows[0]], wm[rows[0]]
            else:
                vc = np.minimum.reduce([phi[r] for r in rows])
                vw = np.minimum.reduce([wm[r] for r in rows])
            np.maximum(tc, vc, out=tc)                # term = best variant (max)
            np.maximum(tw, vw, out=tw)

    weights = np.array([t["weight"] for t in terms], dtype=np.float32)
    chunk_cov = (term_phi_chunk * weights[:, None]).sum(axis=0)
    win_cov = (term_phi_win * weights[:, None]).sum(axis=0)
    win_ccsum = chunk_cov[0:Nw].copy()                # sliding chunk_cov sum for the tiebreak
    for shift in range(1, COV_WIN):
        win_ccsum += chunk_cov[shift:shift + Nw]

    cd = cache.chunk_doc
    valid = cd[0:Nw] == cd[COV_WIN - 1:COV_WIN - 1 + Nw]   # never span a doc boundary
    if doc_keep is not None:
        allowed = np.zeros(len(cache.manifest["docs"]), dtype=bool)
        for di in doc_keep:
            allowed[di] = True
        valid = valid & allowed[cd[0:Nw]]

    return CovResult(terms=terms, stopped=stopped, skipped=skipped, scan_ids=scan_ids,
                     tok2idx=tok2idx, scan_secs=scan_secs, Zraw=Zraw, phi=phi,
                     term_phi_chunk=term_phi_chunk, term_phi_win=term_phi_win,
                     chunk_cov=chunk_cov, win_cov=win_cov, win_ccsum=win_ccsum,
                     valid=valid, Nw=Nw)


def cov_sites(res, topw):
    """Greedy non-max suppression over windows (4.4): sort by win_cov desc (tiebreak
    + 1e-6 * in-window chunk_cov sum, then lower s), suppress radius WIN-1, pick <= topw
    sites. Returns a list of window-start global chunk indices."""
    combined = res.win_cov + 1e-6 * res.win_ccsum
    combined = np.where(res.valid, combined, -np.inf)
    order = np.argsort(-combined, kind="stable")      # stable -> lower s wins exact ties
    picked, radius = [], COV_WIN - 1
    for s in order:
        s = int(s)
        if not res.valid[s] or res.win_cov[s] <= 0:   # order is desc -> nothing useful remains
            break
        if all(abs(s - p) > radius for p in picked):
            picked.append(s)
            if len(picked) >= topw:
                break
    return picked


def cov_hits(res, picked, topk):
    """Final chunk ranking over the picked sites (4.4): candidates = all chunks of the picked
    windows; score(c) = best win_cov over sites containing c + chunk_cov(c); tiebreak lower
    chunk index. Returns (ordered_chunk_ids, per_hit) where per_hit carries the site range."""
    best = {}
    for s in picked:
        wc = float(res.win_cov[s])
        for c in range(s, s + COV_WIN):
            if c not in best or wc > best[c]:
                best[c] = wc
    if not best:
        return [], []
    cand = np.array(sorted(best.keys()), dtype=np.int64)          # ascending -> stable tiebreak
    scores = np.array([best[int(c)] + float(res.chunk_cov[int(c)]) for c in cand])
    order = np.argsort(-scores, kind="stable")[:topk]
    ordered = [int(cand[i]) for i in order]
    per_hit = []
    for c in ordered:
        site_s, site_wc = c, -1.0
        for s in picked:
            if s <= c <= s + COV_WIN - 1 and float(res.win_cov[s]) > site_wc:
                site_wc, site_s = float(res.win_cov[s]), s
        per_hit.append({
            "chunk": c, "site": int(site_s), "win_cov": site_wc,
            "chunk_cov": float(res.chunk_cov[c]), "total": site_wc + float(res.chunk_cov[c]),
        })
    return ordered, per_hit


def retrieve_cov(cache, token_matrix, codec, question, topk, topw=None, doc_keep=None,
                 raw_scale=None, white_scale=None):
    """Shipping coverage retrieval. Ranks chunks by term coverage chunk_cov (the eval's
    config C), which MEASURED strictly better than the query-time window path (config D) on
    this store: windows spread credit to neighbours and let multi-term competitors win
    (v1.3 handoff; e.g. W5's gold is chunk_cov rank 1 but the window selection missed it). The
    window machinery (cov_sites/cov_hits) is retained for the eval's honest negative-lift
    measurement only. `topw` is accepted for API/CLI compatibility and is unused here.

    Returns (ordered_chunk_ids, per_hit, trace) or None when the query has no usable content
    term (caller falls back to v12)."""
    res = cov_compute(cache, token_matrix, codec, question, doc_keep, raw_scale, white_scale)
    if res is None:
        return None
    cov = res.chunk_cov
    if doc_keep is not None:
        mask = np.isin(cache.chunk_doc, np.array(sorted(doc_keep), dtype=np.int32))
        cov = np.where(mask, cov, -1.0)
    order = np.argsort(-cov, kind="stable")[:topk]
    ordered = [int(c) for c in order if cov[int(c)] > 0]
    per_hit = [{"chunk": c, "chunk_cov": float(res.chunk_cov[c])} for c in ordered]
    trace = {
        "terms": [{"word": t["word"], "weight": t["weight"]} for t in res.terms],
        "stopped": res.stopped, "skipped": res.skipped,
        "n_scans": len(res.scan_ids), "scan_secs": res.scan_secs,
    }
    return ordered, per_hit, trace


def retrieve_graded(cache, token_matrix, T_bits, codec, question, topk):
    """Eval config B (receipt R5): stop list + integer IDF weights (cap 15) -> weighted-Hamming
    full chunk scan via bit-sliced magnitude planes (sign vector + 4 planes). Isolates graded
    weighting alone (no variant expansion, no coverage). Returns (order, dist)."""
    N = int(cache.bag.shape[0])
    acc = np.zeros(rc.D, dtype=np.int32)
    used, seen = 0, set()
    for raw in re.findall(r"[A-Za-z0-9_.]+", question):
        w = raw.rstrip(".")
        if not w:
            continue
        key = w.lower()
        if key in seen or key in COV_STOP_WORDS:
            continue
        seen.add(key)
        ids = codec.encode_ids(" " + w)
        dfs = [int(cache.df[i]) for i in ids] if ids else []
        if not ids or any(d == 0 for d in dfs):
            ids = codec.encode_ids(w)                 # fall back to the bare form
            dfs = [int(cache.df[i]) for i in ids] if ids else []
            if not ids or any(d == 0 for d in dfs):
                continue
        wt = int(min(15, max(1, round(4.0 * math.log(N / min(dfs))))))
        for i in ids:
            acc += wt * (2 * rc.unpack(token_matrix[int(i)]).astype(np.int32) - 1)
        used += 1
    if used == 0:                                     # all stopped -> plain unweighted bag
        for i in codec.encode_ids(question):
            acc += 2 * rc.unpack(token_matrix[int(i)]).astype(np.int32) - 1

    sign = (acc > 0).astype(np.uint8)
    mag = np.clip(np.abs(acc), 0, 15).astype(np.uint8)
    sign_words = rc.pack(sign)
    planes = [rc.pack(((mag >> j) & 1).astype(np.uint8)) for j in range(4)]

    bag = cache.bag
    n = int(bag.shape[0])
    dist = np.empty(n, dtype=np.int64)
    block = 65536
    for start in range(0, n, block):
        blk = np.ascontiguousarray(bag[start:start + block])
        sr = blk ^ sign_words
        d = np.zeros(blk.shape[0], dtype=np.int64)
        for j in range(4):
            d += (1 << j) * np.bitwise_count(sr & planes[j]).sum(axis=1, dtype=np.int64)
        dist[start:start + blk.shape[0]] = d
    order = np.argsort(dist)[:topk]
    return order, dist
