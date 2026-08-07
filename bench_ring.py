r"""bench_ring.py — bounded ~1M-token synthetic benchmark for the paged store v2 (Goal E).

No downloads, no Ollama, deterministic from one seed. The corpus is synthesised directly as
token IDs (explicitly allowed): a Zipf(1.3) body over a permuted regular-token vocab gives a
real high-DF head + a heavy distinct-token tail; a 15% common-word overlay makes English
function words genuinely df-filterable; and docs 0..4 get one chunk-aligned needle sentence
each so retrieval has a ground-truth target. Everything runs in a throwaway temp store
(deleted at the end unless --keep).

Phases (each timed): generate -> ingest -> cold cache -> flat-query latency -> round-trip
decode -> needle retrieval (df on vs off) -> incremental add/replace/remove. Prints a metric
report and asserts the acceptance gates -> BENCH PASS, or BENCH FAIL <gate> + exit 1.

Run:  python bench_ring.py                 (defaults: 1,000,000 tokens over 20 docs)
      python bench_ring.py --tokens 200000 --docs 8 --keep
"""

import argparse
import gc
import os
import shutil
import sys
import tempfile
import time

import numpy as np

import ring_core as rc
import ring_store as rs

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_SEED = 0xB0B57011                 # bench-local; NOT part of any frozen contract
OVERLAY_P = 0.15                        # fraction of positions replaced by a uniform pool token
HOT_P = 0.05                            # extra concentration on ONE function word (' in', tid 304)
POOL_TEXT = "the of and is in a to for with this that benchmark document section"
NEEDLE_CHUNK = 10                       # chunk-aligned local index of each needle (docs 0..4)
MARKER_CHUNK = 20                       # doc-0-only sentinel chunk (special-token, df->0 on replace)
OLD_MARKER_ID = rc.SPECIAL_BASE + 7     # a special id used nowhere else in the corpus


def _dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def gen_corpus(tokens, ndocs, K, codec, seed):
    """Return (docs_ids list, needle_chunk dict d->local, needle_text dict d->str)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(rc.SPECIAL_BASE)
    pool = np.array(codec.encode_ids(POOL_TEXT), dtype=np.int64)
    hot = codec.encode_ids(" in")[0]        # single common word, also present in every needle
    per_doc = tokens // ndocs
    docs_ids, needle_chunk, needle_text = [], {}, {}
    for d in range(ndocs):
        body = perm[(rng.zipf(1.3, per_doc) - 1) % rc.SPECIAL_BASE].astype(np.int64)
        overlay = rng.random(per_doc) < OVERLAY_P
        body[overlay] = pool[rng.integers(0, len(pool), int(overlay.sum()))]
        # concentrate ' in' so it lands in >25% of chunks (scale-independent) -> df-droppable
        body[rng.random(per_doc) < HOT_P] = hot
        if d < 5:
            text = f"The secret code hidden in benchmark document {d} is zebra-{d}-kumquat."
            nids = np.array(codec.encode_ids(text), dtype=np.int64)
            off = K * NEEDLE_CHUNK
            body[off:off + len(nids)] = nids          # chunk-aligned splice
            needle_chunk[d] = NEEDLE_CHUNK
            needle_text[d] = text
        if d == 0:
            body[K * MARKER_CHUNK] = OLD_MARKER_ID     # doc-0-only sentinel special token
        docs_ids.append(body)
    return docs_ids, needle_chunk, needle_text


def needle_rank(cache, hams, d, local):
    """1-based rank of the (doc d, local chunk) row in ascending-hamming order."""
    order = np.argsort(hams)
    gidx = np.where((cache.chunk_doc == d) & (cache.chunk_local == local))[0][0]
    return int(np.where(order == gidx)[0][0]) + 1


def main():
    ap = argparse.ArgumentParser(description="Decoder Ring paged-store benchmark.")
    ap.add_argument("--tokens", type=int, default=1_000_000)
    ap.add_argument("--docs", type=int, default=20)
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=BENCH_SEED)
    ap.add_argument("--keep", action="store_true", help="keep the temp store dir after the run")
    args = ap.parse_args()
    seed = args.seed

    K = rc.DEFAULT_FANOUT
    t_all = time.perf_counter()
    print(f"loading ring (D={rc.D}, VOCAB={rc.VOCAB}) ...")
    atoms_bits, T_bits = rc.load_atoms(os.path.join(HERE, "atoms.dring"))
    tm = rc.load_tokens(os.path.join(HERE, "tokens.dring"), verify=False)
    codec = rc.TokenCodec(os.path.join(HERE, "tokenizer.json"))
    ring = tempfile.mkdtemp(prefix="ringbench-")
    print(f"temp store: {ring}\n")

    metrics = {}
    gates = {}
    try:
        # --- 1. generate ------------------------------------------------------------
        t = time.perf_counter()
        docs_ids, needle_chunk, needle_text = gen_corpus(args.tokens, args.docs, K, codec, seed)
        total_tokens = sum(len(x) for x in docs_ids)
        metrics["gen_s"] = time.perf_counter() - t
        distinct = len(np.unique(np.concatenate(docs_ids)))
        print(f"[1] generated {total_tokens:,} tokens over {args.docs} docs "
              f"({distinct:,} distinct ids, needles chunk-aligned at local {NEEDLE_CHUNK}) "
              f"in {metrics['gen_s']:.1f}s")

        # --- 2. ingest --------------------------------------------------------------
        t = time.perf_counter()
        for d in range(args.docs):
            rs.add_document(ring, ids=docs_ids[d], name=f"bench-doc{d:02d}",
                            token_matrix=tm, T_bits=T_bits, codec=codec, K=K)
        metrics["ingest_s"] = time.perf_counter() - t
        metrics["ingest_tok_s"] = total_tokens / metrics["ingest_s"]
        print(f"[2] ingested {args.docs} pages in {metrics['ingest_s']:.1f}s "
              f"=> {metrics['ingest_tok_s']:,.0f} tok/s")

        # --- 3. cold cache ----------------------------------------------------------
        t = time.perf_counter()
        cache = rs.ensure_cache(ring, tm, T_bits)
        metrics["cache_cold_s"] = time.perf_counter() - t
        n_chunks = cache.bag.shape[0]
        store_dir, pages_dir, _, cache_dir = rs.store_paths(ring)
        metrics["pages_mb"] = _dir_size(pages_dir) / 1e6
        metrics["cache_mb"] = _dir_size(cache_dir) / 1e6
        print(f"[3] cold cache build {metrics['cache_cold_s']:.1f}s for {n_chunks:,} L0 chunks; "
              f"pages {metrics['pages_mb']:.0f} MB + cache {metrics['cache_mb']:.0f} MB on disk")

        # --- 4. flat query latency (warm) -------------------------------------------
        rng = np.random.default_rng(seed ^ 0x5F)
        qbags = []
        for d in range(5):                                   # needle queries
            qids = codec.encode_ids(f"the secret code hidden in benchmark document {d}")
            qbags.append(rc.pack(rc.maj([rc.unpack(tm[i]) for i in qids], T_bits)))
        for _ in range(5):                                   # random-token queries
            qids = rng.integers(0, rc.SPECIAL_BASE, 8)
            qbags.append(rc.pack(rc.maj([rc.unpack(tm[int(i)]) for i in qids], T_bits)))
        rs.flat_scan(cache.bag, qbags[0]); rs.flat_scan(cache.bag, qbags[1])   # warm the mmap
        lat = []
        for qb in qbags:
            t = time.perf_counter()
            h = rs.flat_scan(cache.bag, qb)
            np.argsort(h)[:3]
            lat.append(time.perf_counter() - t)
        metrics["flat_median_s"] = float(np.median(lat))
        metrics["flat_x10_s"] = metrics["flat_median_s"] * 10
        print(f"[4] flat query median {metrics['flat_median_s']*1000:.1f} ms over {n_chunks:,} chunks "
              f"(x10 -> ~{metrics['flat_x10_s']*1000:.0f} ms at the 625k-chunk design point)")

        # --- 5. round-trip decode ---------------------------------------------------
        pairs = []
        rng2 = np.random.default_rng(seed ^ 0xA5)
        picked_docs = set()
        while len(pairs) < 10:
            d = int(rng2.integers(0, args.docs))
            nloc = int((cache.chunk_doc == d).sum())
            local = int(rng2.integers(0, nloc))
            pairs.append((d, local)); picked_docs.add(d)
        ok = tot = 0
        for d, local in pairs:
            page = rs.load_manifest(ring)["docs"][d]["page"]
            with np.load(rs.page_fs_path(ring, page)) as npz:
                klen = int(npz["chunk_len"][local])
                chunk_ids = [int(i) for i in npz["token_ids"][local * K:local * K + klen]]
            expect = [int(i) for i in docs_ids[d][local * K:local * K + klen]]
            assert chunk_ids == expect, f"page ids != generated ground truth at doc{d} c{local}"
            # verification mode: derive the seq bundle from the stored ids and round-trip decode it
            # (v3 pages store no seq vectors; the composed bundle is bit-identical to v2's stored
            # seq, so the decode==100% gate is exactly preserved as derived-data verification)
            seq = rc._bundle_seq([rc.unpack(tm[i]) for i in chunk_ids], T_bits)
            dec = rc.decode_chunk(seq, klen, tm)
            ok += sum(int(a == b) for a, b in zip(dec, expect)); tot += klen
        metrics["decode_acc"] = ok / tot
        print(f"[5] round-trip decode {ok}/{tot} tokens across {len(picked_docs)} docs "
              f"=> {metrics['decode_acc']*100:.1f}%")

        # --- 6. needle retrieval (df on vs off) -------------------------------------
        top3 = 0
        dropped_total = 0
        print("[6] needle retrieval (df filter on, df-max=0.25):")
        for d in range(5):
            qids = codec.encode_ids(needle_text[d])
            kept, dropped = rs.df_filter(qids, cache.df, n_chunks, rs.DF_MAX_DEFAULT)
            dropped_total += len(dropped)
            qb = rc.pack(rc.maj([rc.unpack(tm[i]) for i in kept], T_bits))
            r_on = needle_rank(cache, rs.flat_scan(cache.bag, qb), d, needle_chunk[d])
            qb_off = rc.pack(rc.maj([rc.unpack(tm[i]) for i in qids], T_bits))
            r_off = needle_rank(cache, rs.flat_scan(cache.bag, qb_off), d, needle_chunk[d])
            hit = r_on <= 3
            top3 += int(hit)
            drops = ", ".join(sorted({repr(codec.ids_to_text([t])) for t in dropped}))
            print(f"    doc {d}: rank {r_on} (df on) vs {r_off} (df off)  {'TOP3' if hit else 'miss'}"
                  f"   dropped: {drops or '(none)'}")
        metrics["needles_top3"] = top3
        metrics["df_dropped_total"] = dropped_total

        # --- 7. incremental add / replace / remove ----------------------------------
        del cache
        gc.collect()
        arithmetic_ok = True

        m0 = sum(int(d["n_chunks"]) for d in rs.load_manifest(ring)["docs"])
        extra = np.random.default_rng(seed ^ 0x11).integers(0, rc.SPECIAL_BASE, 20000).astype(np.int64)
        e_extra = rs.add_document(ring, ids=extra, name="bench-extra", token_matrix=tm,
                                  T_bits=T_bits, codec=codec, K=K)
        cache = rs.ensure_cache(ring, tm, T_bits)
        arithmetic_ok &= cache.bag.shape[0] == m0 + int(e_extra["n_chunks"])
        sha_a = rs.manifest_sha(ring)
        del cache; gc.collect()

        # REPLACE doc 0 with a fresh body carrying a NEW needle and NO old sentinel
        new_text = "The updated secret code hidden in benchmark document 0 is wolfram-0-persimmon."
        new0 = np.random.default_rng(seed ^ 0x22).integers(0, rc.SPECIAL_BASE, 30000).astype(np.int64)
        nn = np.array(codec.encode_ids(new_text), dtype=np.int64)
        new0[K * 5:K * 5 + len(nn)] = nn
        e_rep = rs.add_document(ring, ids=new0, name="bench-doc00", token_matrix=tm,
                                T_bits=T_bits, codec=codec, K=K)
        assert len(rs.load_manifest(ring)["docs"]) == args.docs + 1, "replace changed doc count"
        cache = rs.ensure_cache(ring, tm, T_bits)
        marker_df0 = int(cache.df[OLD_MARKER_ID])
        di0 = next(i for i, dd in enumerate(cache.manifest["docs"]) if dd["name"] == "bench-doc00")
        qb_new = rc.pack(rc.maj([rc.unpack(tm[i]) for i in codec.encode_ids("wolfram persimmon")], T_bits))
        new_hit_doc = int(cache.chunk_doc[int(np.argsort(rs.flat_scan(cache.bag, qb_new))[0])])
        sha_b = rs.manifest_sha(ring)
        n_after_rep = cache.bag.shape[0]
        del cache; gc.collect()

        # REMOVE the last original doc
        rem_name = f"bench-doc{args.docs - 1:02d}"
        rem_chunks = next(int(d["n_chunks"]) for d in rs.load_manifest(ring)["docs"]
                          if d["name"] == rem_name)
        rs.remove_document(ring, rem_name, codec=codec)
        cache = rs.ensure_cache(ring, tm, T_bits)
        arithmetic_ok &= cache.bag.shape[0] == n_after_rep - rem_chunks
        sha_c = rs.manifest_sha(ring)
        del cache; gc.collect()

        metrics["marker_df_after_replace"] = marker_df0
        metrics["new_needle_hit"] = (new_hit_doc == di0)
        metrics["cache_invalidated_3x"] = len({sha_a, sha_b, sha_c}) == 3
        arithmetic_ok &= metrics["new_needle_hit"] and marker_df0 == 0 and metrics["cache_invalidated_3x"]
        metrics["arithmetic_ok"] = bool(arithmetic_ok)
        print(f"[7] incremental: +extra(+{int(e_extra['n_chunks'])}) replace-doc0(new needle "
              f"{'HIT' if metrics['new_needle_hit'] else 'MISS'}, old-sentinel df={marker_df0}) "
              f"-remove(-{rem_chunks})  arithmetic {'OK' if arithmetic_ok else 'WRONG'}  "
              f"cache-invalidated-3x {metrics['cache_invalidated_3x']}")

        metrics["total_s"] = time.perf_counter() - t_all

        # --- gates ------------------------------------------------------------------
        gates["total_wall<600s"] = metrics["total_s"] < 600
        gates["ingest>=2000tok/s"] = metrics["ingest_tok_s"] >= 2000
        gates["cache_cold<60s"] = metrics["cache_cold_s"] < 60
        gates["flat_median<0.5s"] = metrics["flat_median_s"] < 0.5
        gates["decode==100%"] = metrics["decode_acc"] == 1.0
        gates["needles>=4/5_top3"] = metrics["needles_top3"] >= 4
        gates["df_filter_exercised>=1"] = metrics["df_dropped_total"] >= 1
        gates["add/replace/remove_exact"] = metrics["arithmetic_ok"]

        print("\n" + "=" * 64)
        print(f"total wall time     {metrics['total_s']:.1f} s")
        print(f"ingest throughput   {metrics['ingest_tok_s']:,.0f} tok/s")
        print(f"cold cache build    {metrics['cache_cold_s']:.1f} s")
        print(f"flat query median   {metrics['flat_median_s']*1000:.1f} ms  (x10 ~{metrics['flat_x10_s']*1000:.0f} ms)")
        print(f"round-trip decode   {metrics['decode_acc']*100:.1f} %")
        print(f"needles in top-3    {metrics['needles_top3']}/5")
        print(f"df tokens dropped   {metrics['df_dropped_total']} (across needle queries)")
        print(f"store on disk       {metrics['pages_mb']:.0f} MB pages + {metrics['cache_mb']:.0f} MB cache")
        print("=" * 64)
        for name, ok in gates.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        failed = [n for n, ok in gates.items() if not ok]
        if failed:
            print(f"\nBENCH FAIL {failed[0]}")
            sys.exit(1)
        print("\nBENCH PASS")
    finally:
        cache = None            # drop the last bag mmap so rmtree can delete bag_matrix.npy (Windows)
        gc.collect()
        if args.keep:
            print(f"\n(kept temp store: {ring})")
        else:
            try:
                shutil.rmtree(ring)
            except OSError as e:
                print(f"\nwarning: temp store not fully removed: {ring} ({e}) - delete it manually")


if __name__ == "__main__":
    main()
