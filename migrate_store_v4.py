r"""migrate_store_v4.py — one-shot in-place migration of a paged store v3 -> v4.

WHAT IT DOES. v3 and v4 pages carry the EXACT SAME members (token_ids uint32 / chunk_len /
df_token_ids / df_counts / meta_json) — v4 only re-writes each page with np.savez_compressed
(zipfile ZIP_DEFLATED/zlib) instead of np.savez. np.load decompresses npz members
transparently, so every reader is byte-untouched; this tool is pure compression. Measured on
the live 103-page store: 53.14 MB -> 18.63 MB (0.351x). The 17-bit bit-packing the task asked
for was measured and REJECTED (compressed-uint32 is ~11% smaller than compressed-17-bit-packed
on every real page, and needs no custom codec) — see CHANGELOG v1.7.

Unlike the v2->v3 tool, this one recovers NO vectors and re-encodes NOTHING: the ids are
already the truth, so migration = recompress + verify-by-equality. It therefore needs no
.dring files (numpy + json only), and ids:// synthetic docs migrate fine.

TWO PHASES (nothing live is mutated until every doc verifies):
  A. build+verify -> store\pages_v4\ : per doc, read the v3 page's five members, set the meta's
     store_version to 4 (all other fields verbatim), np.savez_compressed into pages_v4\<same
     name>, then RE-OPEN the written file and assert np.array_equal + dtype-preserved on all four
     id arrays, the meta round-trips, len(token_ids)==manifest n_tokens, sum(chunk_len)==n_tokens.
     Any failure -> HARD STOP, nothing swapped.
  B. swap+re-chain (fast, ordered): back up manifest + the 3 freshness metas (*.v3.bak.json,
     distinct from the v2 tool's *.v2.bak.*); rename pages -> pages.v3.bak and pages_v4 -> pages;
     write the v4 manifest (docs + df_summary VERBATIM, store_version 4); re-chain the main
     cache_meta + sem meta + sem cache_meta to the new manifest sha. The derived cache ARRAYS are
     untouched — the ids are byte-identical, so a rebuild would reproduce them bit-for-bit; patch,
     never rebuild.

pages.v3.bak (and pages.v2.bak) are NEVER deleted by this tool (the operator removes them
later). Safe to re-run: a v4 manifest exits 0 ("already migrated"); a leftover pages_v4\ from a
crash is rebuilt.

Run:  python migrate_store_v4.py            # migrates the store next to this script
      python migrate_store_v4.py --ring DIR
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time

import numpy as np

import ring_sem as sm
import ring_store as rs

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_ID_MEMBERS = ("token_ids", "chunk_len", "df_token_ids", "df_counts")


class MigrateError(Exception):
    """A migration precondition or per-doc verification failed (nothing is swapped)."""


# ---------------------------------------------------------------- v4 page assembly + verify
def _recompress_and_verify(v3_page_path, out_path, entry):
    """Read the v3 page, deflate-recompress its identical members into out_path, RE-OPEN and
    prove every array is bit-identical (dtype preserved) + the invariants hold. Returns
    (old_bytes, new_bytes). Raises MigrateError on any mismatch (nothing live is touched)."""
    with np.load(v3_page_path) as npz:
        members = {m: np.array(npz[m]) for m in _ID_MEMBERS}     # copies, npz closes below
        v3_meta = json.loads(npz["meta_json"].item())
    v4_meta = dict(v3_meta)
    v4_meta["store_version"] = 4                                 # ONLY field that changes

    buf = io.BytesIO()
    np.savez_compressed(buf, meta_json=np.array(json.dumps(v4_meta)), **members)
    new_bytes = buf.getvalue()
    rs.atomic_write_bytes(out_path, new_bytes)

    # re-open the file actually on disk and verify against the in-memory source arrays
    with np.load(out_path) as npz:
        if set(npz.files) != set(_ID_MEMBERS) | {"meta_json"}:
            raise MigrateError(f"{entry['name']}: v4 page member set {sorted(npz.files)} != v3 set")
        for m in _ID_MEMBERS:
            got = npz[m]
            if not np.array_equal(got, members[m]):
                raise MigrateError(f"{entry['name']}: v4 member {m!r} != v3 (array mismatch)")
            if got.dtype != members[m].dtype:
                raise MigrateError(f"{entry['name']}: v4 member {m!r} dtype {got.dtype} "
                                   f"!= v3 {members[m].dtype}")
        rt_meta = json.loads(npz["meta_json"].item())
        n_tok = int(npz["token_ids"].shape[0])
        clen_sum = int(npz["chunk_len"].sum())
    if rt_meta != v4_meta:
        raise MigrateError(f"{entry['name']}: v4 meta_json did not round-trip")
    if n_tok != int(entry["n_tokens"]):
        raise MigrateError(f"{entry['name']}: v4 token_ids len {n_tok} != manifest "
                           f"n_tokens {entry['n_tokens']}")
    if clen_sum != n_tok:
        raise MigrateError(f"{entry['name']}: chunk_len sum {clen_sum} != n_tokens {n_tok}")
    return os.path.getsize(v3_page_path), len(new_bytes)


# ---------------------------------------------------------------- freshness re-chain (phase B)
def _patch_json(path, updates):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    obj.update(updates)
    rs.atomic_write_bytes(path, json.dumps(obj, indent=2).encode("utf-8"))


# ---------------------------------------------------------------- driver
def migrate(ring_dir, log=print):
    """Migrate the v3 store under `ring_dir` to v4 in place. Returns a summary dict.

    No ring load is needed (no vectors touched). Raises MigrateError on any precondition or
    per-doc verification failure, leaving the live store untouched (v3)."""
    store_dir, pages_dir, manifest_path, cache_dir = rs.store_paths(ring_dir)
    sem_dir, _, _, _, sem_cache_dir, sem_meta_path, _ = sm.semantic_paths(ring_dir)
    if not os.path.exists(manifest_path):
        raise MigrateError(f"no manifest at {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:                 # plain json (NOT load_manifest)
        manifest = json.load(f)
    if manifest.get("store_version") == 4:
        log("store is already store_version 4 -> nothing to do (idempotent exit)")
        return {"status": "already-migrated"}
    if manifest.get("format") != rs.STORE_FORMAT or manifest.get("store_version") != 3:
        hint = ""
        if manifest.get("format") == rs.STORE_FORMAT and manifest.get("store_version") == 2:
            hint = " -- run migrate_store_v3.py first"
        raise MigrateError(f"expected a v3 {rs.STORE_FORMAT} store, found "
                           f"{manifest.get('format')}/{manifest.get('store_version')}{hint}")

    # a v3 manifest with pages.v3.bak already present = a prior run was interrupted mid-swap
    # (a clean v3 store never has this name). Fail fast with the recovery pointer rather than
    # letting phase B's os.rename(pages -> pages.v3.bak) raise an uncaught OSError post-work.
    pages_bak = os.path.join(store_dir, "pages.v3.bak")
    if os.path.exists(pages_bak):
        raise MigrateError(
            f"{pages_bak} already exists -> a prior migration was interrupted mid-swap. Recover "
            f"first (rollback recipe: ren pages.v3.bak pages + restore the *.v3.bak.json metas), "
            f"or delete pages.v3.bak if 'pages' is already the correct v4 set, then re-run.")

    docs = manifest["docs"]
    total_chunks = sum(int(d["n_chunks"]) for d in docs)
    log(f"migrating {len(docs)} docs / {total_chunks} chunks (v3 -> v4, deflate-compress pages)")
    log("WARNING: single-writer store — do not run queries/ingest against it during migration")

    pages_v4_dir = os.path.join(store_dir, "pages_v4")
    if os.path.isdir(pages_v4_dir):
        log(f"removing leftover {pages_v4_dir} from a prior run")
        shutil.rmtree(pages_v4_dir)
    os.makedirs(pages_v4_dir)

    # ---- Phase A: build + verify every doc into pages_v4 (no live mutation) ----
    t_all = time.perf_counter()
    log("\nphase A: recompress + verify (array-equal per doc)")
    log(f"  {'name':<24}{'chunks':>8}{'old B':>12}{'new B':>12}{'ratio':>8}")
    old_total = new_total = 0
    for d in docs:
        v3_page = rs.page_fs_path(ring_dir, d["page"])
        out_name = os.path.basename(d["page"])
        old_b, new_b = _recompress_and_verify(v3_page, os.path.join(pages_v4_dir, out_name), d)
        old_total += old_b
        new_total += new_b
        log(f"  {d['name']:<24}{int(d['n_chunks']):>8}{old_b:>12}{new_b:>12}"
            f"{new_b / old_b if old_b else 0:>8.3f}")
    ratio = new_total / old_total if old_total else 0.0
    log(f"phase A done: {len(docs)}/{len(docs)} docs verified array-equal in "
        f"{time.perf_counter()-t_all:.0f}s ({old_total} -> {new_total} B, ratio {ratio:.3f})")

    # ---- Phase B: swap + re-chain (fast, ordered) ----
    log("\nphase B: swap + re-chain")
    main_cache_meta = os.path.join(cache_dir, "cache_meta.json")
    sem_cache_meta = os.path.join(sem_cache_dir, "cache_meta.json")
    shutil.copy2(manifest_path, os.path.join(store_dir, "manifest.v3.bak.json"))
    rechain = []                                          # (path, kind) of freshness metas present
    if os.path.exists(main_cache_meta):
        shutil.copy2(main_cache_meta, os.path.join(cache_dir, "cache_meta.v3.bak.json"))
        rechain.append((main_cache_meta, "main"))
    if os.path.exists(sem_meta_path):
        shutil.copy2(sem_meta_path, os.path.join(sem_dir, "meta.v3.bak.json"))
        rechain.append((sem_meta_path, "sem_meta"))
    if os.path.exists(sem_cache_meta):
        shutil.copy2(sem_cache_meta, os.path.join(sem_cache_dir, "cache_meta.v3.bak.json"))
        rechain.append((sem_cache_meta, "sem_cache"))
    log("  backed up manifest + present freshness metas (*.v3.bak.*)")

    os.rename(pages_dir, pages_bak)                       # pages_bak pre-checked absent above
    os.rename(pages_v4_dir, pages_dir)
    log(f"  swapped pages -> pages.v3.bak (kept), pages_v4 -> pages ({len(docs)} pages)")

    manifest["store_version"] = 4
    manifest["updated"] = rs._now_iso()                              # docs + df_summary VERBATIM
    rs.atomic_write_bytes(manifest_path, json.dumps(manifest, indent=2).encode("utf-8"))
    new_man_sha = rs.manifest_sha(ring_dir)

    done = []
    for path, kind in rechain:                            # patch, NOT rebuild (ids are bit-identical)
        if kind == "sem_cache":
            if not os.path.exists(sem_meta_path):         # can't chain to an absent sem meta.json
                log("  (skipped sem cache_meta re-chain: semantic/meta.json absent)")
                continue
            new_sem_meta_sha = hashlib.sha256(open(sem_meta_path, "rb").read()).hexdigest()
            _patch_json(path, {"sem_meta_sha256": new_sem_meta_sha})
        else:
            _patch_json(path, {"manifest_sha256": new_man_sha})
        done.append(kind)
    log("  re-chained freshness: " + ", ".join(done))
    log("  (derived cache ARRAYS untouched — ids are byte-identical so a rebuild would reproduce "
        "them bit-for-bit; patched, not rebuilt)")

    log(f"\nMIGRATION COMPLETE in {time.perf_counter()-t_all:.0f}s: store_version 4, "
        f"{len(docs)} pages, {old_total} -> {new_total} B ({ratio:.3f}x), pages.v3.bak retained.")
    _flag_baks(store_dir, log)
    _print_rollback(store_dir, log)
    return {"status": "migrated", "docs": len(docs), "old_bytes": old_total,
            "new_bytes": new_total, "ratio": ratio, "manifest_sha256": new_man_sha}


def _flag_baks(store_dir, log):
    parts, total = [], 0
    for bak in ("pages.v2.bak", "pages.v3.bak"):
        p = os.path.join(store_dir, bak)
        if os.path.isdir(p):
            b = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(p) for fn in fns)
            parts.append(f"{bak} {b:,} B")
            total += b
    if parts:
        log(f"  BAK FLAG (operator's call to delete, never this tool's): {' + '.join(parts)} "
            f"~ {total/1e9:.2f} GB total baks")


def _print_rollback(store_dir, log):
    log("\nROLLBACK (restores the byte-for-byte v1.6 state; then revert the code):")
    log(f"  cd {store_dir}")
    log("  ren pages pages.v4.failed        (or delete it)")
    log("  ren pages.v3.bak pages")
    log("  copy /Y manifest.v3.bak.json manifest.json")
    log("  copy /Y cache\\cache_meta.v3.bak.json cache\\cache_meta.json")
    log("  copy /Y semantic\\meta.v3.bak.json semantic\\meta.json")
    log("  copy /Y semantic\\cache\\cache_meta.v3.bak.json semantic\\cache\\cache_meta.json")


def main():
    ap = argparse.ArgumentParser(description="Migrate a Decoder Ring paged store v3 -> v4 "
                                             "(deflate-compress pages; identical members).")
    ap.add_argument("--ring", default=os.path.dirname(os.path.abspath(__file__)),
                    help="ring directory holding store\\ (default: script dir)")
    args = ap.parse_args()
    try:
        summary = migrate(args.ring)
    except MigrateError as e:
        print(f"\nMIGRATION ABORTED (store left on v3, nothing swapped): {e}")
        sys.exit(1)
    sys.exit(0 if summary.get("status") in ("migrated", "already-migrated") else 1)


if __name__ == "__main__":
    main()
