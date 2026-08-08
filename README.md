# Decoder Ring v1.8

A universal, deterministic **token → hypervector codebook** for RAG in HDC space, plus the
encode / decode / retrieve machinery to use it. Everything regenerates from one 32-bit seed
through a library-agnostic PRNG (SHA-256 counter mode), so the codebook is byte-reproducible
on any machine and re-implementable in any language.

- **VSA model:** BSC (binary spatter codes). bind = XOR, bundle = bitwise majority with a
  seeded tie-break vector, position = bit rotation, similarity = Hamming.
- **Width:** `D = 16384` bits = 256 × uint64 words.
- **Vocabulary:** the Llama-3-family BPE tokenizer, 128,256 IDs (128,000 BPE + 256 specials).

## Files

| file | what |
|---|---|
| `ring_core.py` | the library: PRNG, BSC ops, byte map, token composition, file I/O, decoders, chunk tree (**frozen**) |
| `ring_store.py` | **v1.5 / v1.7** paged store **v4** (compact token-id pages, DEFLATE-compressed; `compose_bags` bit-sliced derived-bag rebuild) + term-coverage retriever (`retrieve_cov`/`cov_compute`) + `retrieve_v12` + `retrieve_graded` (eval-only) |
| `migrate_store_v3.py` | **v1.5 NEW** one-shot v2→v3 migration (offline): re-encode undrifted docs + recover drifted `README`/`CHANGELOG` per chunk, per-doc bag bit-exact verify, two-phase swap, freshness re-chain, `pages.v2.bak` kept for rollback |
| `migrate_store_v4.py` | **v1.7 NEW** one-shot v3→v4 migration (offline, no ring load): recompress pages + **per-doc `np.array_equal` verify** into `pages_v4/`, two-phase swap → `pages.v3.bak`, freshness re-chain (cache arrays patched, not rebuilt) |
| `ring_sem.py` | **v1.4 NEW / v1.6 / v1.7** semantic plane: F9-style binary keys over nomic-embed-text window embeddings, persistent resumable sidecar, full-stack rerank (semantic injection + exact-text credit + neighbor credit); **v1.6 query-time abstention (`SEM_ABSTAIN_TAU`) + doc-cap-2 (`SEM_DOC_CAP`)**; **v1.7 `slim_sidecar` (opt-in embed removal) + slim-aware key-reuse** |
| `gen_ring.py` | generator CLI — regenerates `atoms.dring` + `tokens.dring` from the seed |
| `ring_cli.py` | **v1.4 / v1.6 / v1.7** demo CLI — `query --retriever {auto,full,cov,v12}` (default `auto`) + `--sem-weight`; **v1.6 `--no-abstain`** (full stack abstains with exit 3 when no grounded answer); **v1.7 `sembuild --slim`** (opt-in embed removal); `sembuild`; `info`/`encode`/`decode`/`ingest`/`remove`/`stats` |
| `eval_ring.py` | **v1.4 / v1.6** scored retrieval eval: **`--set {A,B,AB}`** (frozen set A + held-out set B + 7 unanswerable), 6 rows (A/C/S/F/**G15**/G), per-mechanism attribution, **32-combo calibration sweep**, **abstention block + A-tuned/B-validated summary**, page-decode gold re-resolution (retrieval-only; `/api/embed` on cache miss only) |
| `gui.py` | **v1.2** dark-theme Tk chat front-end (plain text = query, drops/`/ingest` = add, `/rebuild`, knobs) — forked from foreman, subprocess-drives `ring_cli.py` |
| `Ring GUI.bat` | **v1.2** one-click launcher (`pythonw gui.py`; the GUI itself owns the island) |
| `wiki_ingest.py` | **v1.2** Simple English Wikipedia corpus: download → strip → volume-ize → bounded ingest |
| `Start Ring Island.bat` | **v1.1** standalone Ollama island (port 11436, RTX 2070 SUPER pinned by UUID) |
| `ring.bat` | **v1.1** wrapper — ensures the island is up, then forwards args to `ring_cli.py` |
| `test_ring.py` | headless acceptance tests T1-T5 for the codebook (plain asserts) |
| `test_store.py` | store tests T6-T11 (v1.1) + T20-T24 (cov term prep, membership, windows, v12 identity, eval units) + **T30 (v2→v3→v4 migration chain)** + **T35 (v1.7 v4 compressed pages + v3→v4 array-equal migration + verify-fail rollback)** |
| `test_sem.py` | semantic tests T25-T29 (R/mu contract, window grid, sidecar build/resume, rerank mechanics, gold re-resolution) + **T31-T34 (v1.6 abstention/tau, doc-cap, CLI abstain e2e, set-B units)** + **T36/T37 (v1.7 slim byte-identity + post-slim key-reuse + un-slim + doc_ids guard)** |
| `test_sem.py` | **v1.4 NEW** semantic tests **T25-T29** (R/mu contract, window grid, sidecar build/resume/page-decode, rerank mechanics, eval gold re-resolution), fake embedder, zero island contact |
| `test_gui.py` | **v1.2** headless GUI tests T12-T15 (routing, drop parse, URL segments, busy/status) |
| `test_wiki.py` | **v1.2** wiki tests T16-T19 (stripper, mini-dump, resume/skip, bound/clean), zero network |
| `bench_ring.py` | **v1.1** bounded ~1M-token synthetic benchmark (seeded, no downloads, ~90 s) |
| `docs/*.txt` | 3 short sample docs (hyperdimensional computing, the Llama tokenizer, beaver dams) |
| `wiki/` | **v1.2** wiki corpus workspace: the `.bz2` dump, `dump.meta.json`, and `volumes/vol-*.txt` |
| `atoms.dring` | **File 1** — 256 byte atoms + 1 tie-break vector (526,464 bytes) |
| `tokens.dring` | **File 2** — 128,256 token vectors (262,668,416 bytes) |
| `store/` | **v1.5 / v1.7** paged RAG store **v4**: `manifest.json` + `pages/` (DEFLATE-compressed token-id pages) + `cache/` (see "Store v4" below); `pages.v2.bak` (v2 pages) + `pages.v3.bak` (v3 pages) kept for rollback, ~2.81 GB, operator's to delete |
| `store.npz` + `docs.json` | legacy v1 demo store — regenerable, **unused by v1.1**, safe to delete |

## Retrieval quality — v1.6 (abstention + rerank calibration + held-out eval set B)

v1.6 is **compression-neutral** — no new persistent state, no store writes. Three deliverables,
all measured live on the 628,657-chunk store + 157,127-window sidecar (2026-08-07):

**1. Held-out eval set B (`eval_ring.py --set {A,B,AB}`).** Set A stays the frozen comparability
anchor. Set B is **15 natural + 5 keyword** grep-verified questions (planner-authored and frozen
*before* any calibration was simulated) on fresh topics + **7 unanswerable** questions whose
answers are grep-verified *absent* from the corpus (e.g. "Who is the CEO of OpenAI?", "melting
point of tungsten?" — the bait words are present, the fact never is). Unanswerable questions carry
empty gold and **bypass gold resolution entirely** (they are scored only by the abstention block).

**2. Abstention (headline).** A calibrated query-time "no grounded answer" decision:
**abstain iff `max(sem_win) < SEM_ABSTAIN_TAU` (0.505)**, the top doc_keep-masked window cosine.
The semantic channel is the only signal a co-occurring-words *phantom* cannot forge — U2 (tungsten)
scores `cred 1.00 / score 1.66` on the lexical channels yet the fact is absent. Measured empty gap
`[0.499, 0.512]` cleanly separates **all 45 answerable questions** (min sem_top 0.512, B14) from
**5/7 unanswerable** (max abstained 0.499, U4); tau = the midpoint. On abstain, `ring_cli query`
prints a banner, labels the best candidates **`UNGROUNDED`**, sends **no context to the model**,
and exits **3**; `--no-abstain` restores normal behavior byte-for-byte. The GUI subprocess-drives
the CLI, so on abstain it shows the banner + ungrounded candidates + one `ring: exited with code 3.`
system line (transparent by design). **Honest residual:** near-domain U2 (0.638) and U5 (0.656)
*escape* — the corpus discusses tungsten/the A380 with the right vocabulary but never states the
fact, so no retrieval-level signal separates them from real hits. Catching them needs an
answer-level reader model (out of scope). Result: **5/7 true-abstain at ZERO false-abstain cost**
(0/25 on set A, 0/20 on set B, hits *and* misses).

**3. Rerank calibration — doc-cap-2 diversity.** Tuned on set A only, validated on set B. The
shipped constants are unchanged (floor 0.25 / sem_weight 0.75 / tie_a 0.25) plus **one** new rule:
**`SEM_DOC_CAP = 2`** — at most 2 chunks per document in the returned top-k (cap-then-backfill so
the count never shrinks; rank 1 is always admitted, freezing every @1 metric). Set-A naturals
**9/20 → 10/20** (recovers W5, whose old top-3 was 3× vol-0012 — the cap admits gold vol-0003 at
slot 3), keyword controls held **5/5**. Two honest negative results back this: channel reweighting
(208 combos, semantic-rank bonuses + per-question-type weights) **cannot beat 9/20 with kw 5/5**;
the doccap1 / window-cap / adjacent-suppression diversity shapes reach no 10/20 point either. The
A-tied `sem_weight=1.0`+cap alternative also reaches 10/20 kw 5/5 on set A; it is **reported but not
shipped** — the calibration is chosen on set A alone, and its held-out set-B score (9/15,
*observed only, never used to choose* — doing so would be tuning on the validation set) does not
justify promoting it over the minimal-change shipped point. Confirming it would need a future set C.

**Measured scoreboards (`python eval_ring.py --set AB`, 2026-08-07, 236 s, hit@3).** `G15` = the
v1.5 calibration (uncapped) continuity row; `G` = shipped v1.6 (capped). On set A, rows A/C/S/F are
byte-identical to v1.5 and **G15 reproduces v1.5's G exactly**.

| set | row | nat@1 | nat@3 | kw@1 | kw@3 | all@3 |
|---|---|---|---|---|---|---|
| A | G15 (v1.5 cal) | 0.30 | 0.45 (9/20) | 0.80 | 1.00 | 0.56 |
| A | **G shipped (v1.6)** | 0.30 | **0.50 (10/20)** | 0.80 | **1.00** | 0.60 |
| B | G15 (v1.5 cal) | 0.33 | 0.47 (7/15) | 0.80 | 1.00 | 0.60 |
| B | **G shipped (v1.6)** | 0.33 | **0.47 (7/15)** | 0.80 | **1.00** | 0.60 |

Set A improves (+1 natural, W5 recovered, no hit→miss); set B holds at its own baseline (7/15,
kw 5/5) — the calibration **generalizes without regressing** the held-out set. Abstention block:

```
== abstention (rule: top semantic window score < 0.505) ==
  set A answerable: abstained 0/25   false-abstains on hits: 0/15   [hard gate <=1 on A hits]
  set B answerable: abstained 0/20   false-abstains on hits: 0/12
  unanswerable: true-abstain 5/7   [gate >=5]
    U1 0.487 ABSTAIN | U2 0.638 ANSWERS-UNGROUNDED | U3 0.454 ABSTAIN | U4 0.499 ABSTAIN
    U5 0.656 ANSWERS-UNGROUNDED | U6 0.478 ABSTAIN | U7 0.467 ABSTAIN
  margins: nearest answerable sem_top 0.512 (B14, +0.007); nearest escape 0.499 (U4, -0.006)
```

The tau margins are thin by design (±0.006-0.007) — the honest truth of the data; sem_top is
deterministic given the frozen keys + cached embeds, so any answerable abstain or U-verdict flip
is a bug, not drift. All 52 query embeds cache in `store/semantic/cache/eval_queries.npz`; the
first `--set AB` run contacts the island once for the 27 new embeds, then runs fully offline.

## Retrieval quality — v1.4 (semantic plane + attributed recovery)

v1.4 adds a second, orthogonal evidence channel to the v1.3 lexical retriever: **F9-style
binary keys over nomic-embed-text window embeddings**, plus three miss-targeted recovery
mechanisms, all attributed by the frozen 25-question eval. **All new logic lives in the new
`ring_sem.py`; `ring_store.py` is byte-identical to v1.3** (a stronger invariant — the lexical
path is unchanged because the file is unchanged). `.dring` files, `ring_core.py`, the pages,
manifest, cache format, and `store_version` (2) are untouched. The whole semantic plane is an
additive sidecar under `store/semantic/`.

**The three mechanisms.** (1) *Semantic injection* — the query is embedded, keyed, and Hamming-
scanned against all 157,127 window keys (~0.15 s); the chunks of the top-256 windows are added
to the candidate set, rescuing golds whose lexical rank is hopeless. (2) *Exact-text credit* —
each candidate chunk is decoded and each query term gets credit 1.0 if the word is in the chunk
text (left-word-boundary regex), 0.4 if in an adjacent same-doc chunk, else a capped phantom
floor `min(phi, 0.25)`; the exact-text check kills the ambient-noise "phantom" credit that
capped v1.3. (3) *Neighbor credit* — a term present in one chunk lifts its same-doc neighbours
at γ=0.4, absent-native only. Final score is dimensionless: `cred/Sw + 0.25·nbcov/Sw + 0.75·sem`.

**Measured scoreboard (`python eval_ring.py`, live 628,657-chunk store + 157,127-window sidecar,
2026-08-07, 132 s).** 25 frozen grep-verified questions (20 natural + 5 keyword controls),
byte-offset gold, hit@3:

| row | nat@1 | nat@3 | kw@1 | kw@3 | all@3 | med ms | p90 ms |
|---|---|---|---|---|---|---|---|
| A v12-baseline (majority-bag Hamming) | 0.15 | 0.15 | 0.80 | 0.80 | 0.28 | 599 | 608 |
| C cov-chunk (**= v1.3 exactly**) | 0.10 | 0.25 | 0.40 | 0.80 | 0.36 | 1945 | 5090 |
| S semantic-only (window expansion) | 0.15 | 0.40 | 0.40 | 0.60 | 0.44 | 133 | 136 |
| F cov + semantic (additive channel) | 0.20 | 0.40 | 0.60 | **1.00** | 0.52 | 2084 | 5202 |
| **G full stack — shipped `--retriever auto`** | **0.30** | **0.45** | 0.80 | **1.00** | **0.56** | 2019 | 5137 |

**Gate result.** Pre-set hard floor natural hit@3 ≥ **0.45**; **G reaches 0.45 (9/20) → PASS**
(at the floor; the 0.50-0.65 expected band was not reached). Controls: G keyword hit@3 =
**5/5** — G even recovers **K1**, which C missed, so controls *improve* (no regression). No
regression vs C (G nat@3 9 ≥ C 5; G all@3 14 ≥ C 9). Net **+4 naturals** over the v1.3 lexical
baseline, all attributed below.

**Per-mechanism attribution (of the 16 v1.3 config-C misses, G recovers 7).**

| recovered | mechanism | still miss | note |
|---|---|---|---|
| W3, W4, W7 | semantic-rerank | W1, W8, W9 | gold semantically weak (sem-rank 9-25 in candidates) |
| W11, W12, P4 | semantic-inject | W2 | **self-referential-bait**: the pre-edit README chunk *is* the question text (receipt S10) |
| K1 | combination | W14, P2 | injected (sem-rank 1) but a competitor out-scored it |
| | | P1, P3, P5 | private-doc semantic recovery partial (P-recovery = **1/5**; only P4) |

**New misses G introduces (the known trade, receipt S9).** **W5** and **W10** were C hits and
are G misses: both have gold at chunk-coverage rank 1-3 but the semantic channel pulls
neighbouring chunks above them (sem-rank 25 / 109). This is the pre-flagged W10 regression risk;
the net is still +4 naturals. The full per-miss autopsy prints for every G miss.

**Calibration (measured, `python eval_ring.py --sweep-cal`).** A 16-combo sweep
(`floor {0, 0.25} × sem_weight {0.5, 0.75, 1.0, 1.5} × tie_a {0, 0.25}`) runs in-memory on
precomputed channel primitives. Shipped: **floor 0.25 / sem_weight 0.75 / tie_a 0.25**
(controls-first: it is the only region giving keyword **5/5**, at natural 9/20). The literal
max-natural combo (floor 0 / 0.5 / 0) reaches 10/20 naturals but drops a keyword control to 4/5
— it trades a control for one natural, so it is **not** shipped. All 16 combos are reported.

**The semantic sidecar (`store/semantic/`, ~412 MB, built once).** Per-doc fp16 window
embeddings (~230 MB) + a packed `D_s = 8192`-bit key matrix (160 MB) + window provenance. Keys
are `sign((x − mu) @ Rᵀ)` where `x` is the L2-normalized embedding, **mu** is the frozen global
window mean and **R** is a Rademacher projection regenerated from the frozen `SHA256-CTR`
stream (label `SEMR`); both are sha256-checksummed in `meta.json` and re-verified at load
(mismatch = hard error). Windows are **8 chunks** (~128 tokens), stride 4, never spanning docs
(157,127 total). nomic task prefixes (`search_document:` / `search_query:`) and L2-normalization
are load-bearing. Build once:

```
python ring_cli.py sembuild            # ~45 min once (embed ~25 min + README/CHANGELOG decode ~24 min); resumable
python ring_cli.py sembuild --status   # per-doc coverage, no build (marks [slim] docs)
python ring_cli.py sembuild --slim      # v1.7 opt-in: REMOVE the fp16 embeds (queries byte-identical)
python eval_ring.py --sweep-cal        # 16-combo calibration (in-memory, seconds after primitives)
```

**Slimming the sidecar (`sembuild --slim`, v1.7, opt-in).** Queries read only keys + mu +
provenance + text — **never** the fp16 window embeddings under `emb/`, which are build-time-only
state. `--slim` rewrites each `emb/*.npz` **without** the `emb` member (keeping the tiny
`win_start`/`win_len` so freshness is unchanged), reclaiming **~242 MB** (the sidecar drops
**431.5 MB → 190.2 MB**, measured) with **byte-identical retrieval** (proven: full-stack eval +
abstention + CLI probes diff empty). It re-chains the derived cache so `load_sidecar` still
returns a live sidecar — without that the store would silently fall back to `cov`. Opt-in and
one-way cheap; a later incremental `sembuild` reuses the frozen keys row-for-row (via the sem
`cache_meta` `doc_ids` map) and embeds only new/replaced docs.

**Un-slim** = `sembuild --rebuild` (a full re-embed — island required, ~25 min at this scale).
This is the **first and only** path with a *total* key shift, and it carries a real
**abstention-tau hazard**: a non-slim `--rebuild` re-derives keys from the *same* stored embeds →
deterministic `mu` → bit-stable keys, but un-slim re-embeds every doc on the GPU (not
bit-reproducible) **and** re-estimates `mu`, moving **all** keys by ~25–57 bits on the 8192-bit
keys — the same order as the live abstention margins (±0.006–0.007 vs `tau` 0.505). No error is
raised (the rebuilt sidecar is internally consistent), so a borderline unanswerable/set-B question
can silently cross `tau`. **After any un-slim, re-run `python eval_ring.py --set AB` and compare the
abstention block (margins are ±0.006–0.007); if any U/B verdict flipped, recalibrate `tau` via
`eval_ring.py --sweep-cal`.** The live sidecar is **not** slimmed by default; run the flag when you
want the space back.

**Drift is now resolved from the pages (no re-ingest).** `README.md`/`CHANGELOG.md` are drifted
from their ingested pages (the v1.3 doc edits, never re-ingested). v1.4 replaces v1.3's hard
sha-guard *exit* with a **page-decode provider**: the eval resolves their gold from the store
pages (batched NN-decode of the stored chunk vectors, cached in `store/semantic/text/`), so the
private-doc questions (P1-P5, K1) run without re-ingesting. Re-ingesting is still discouraged
(self-referential bait, W2).

**Latency & the GUI.** Full-stack retrieval is ~2-5 s (cov 2-6 s + query embed ~0.06 s warm,
one-time ~25 s if the island cold-loads the model, + scan ~0.15 s + text-credit rerank); `cov`
2-6 s; `v12` ~0.6 s. The GUI flows through `--retriever auto` unchanged (no argv change,
`gui.py` is byte-identical) — a turn gains ~0.5-1 s. `eval_ring` may call `/api/embed`
(never `/api/generate`) on a cache miss only; the 25 query embeds are cached in
`store/semantic/cache/eval_queries.npz`, so steady-state eval runs are offline.

## Retrieval quality — v1.3 (term-coverage retriever + scored eval)

v1.3 is a **retrieval-quality-first, measured** release. It adds a query-side **term-coverage
retriever** and, more importantly, `eval_ring.py` — a scored harness that makes the retrieval
claim honest. **No cache / page / manifest change** (`store_version` stays 2); the frozen
`.dring` files and `ring_core.py` are untouched. The v1.2 path is preserved **byte-identical**
behind `--retriever v12`.

**What the coverage retriever does (query side only).** Extract content words (a closed-class
stop list drops interrogatives/function words that raw IDF would otherwise rank as top
content), weight each by capped IDF, encode one exact surface form per word, batch-scan the bag
mmap, and turn each token's Hamming into a **dual-normalized soft membership**
`phi = clip(min(Zraw/25, Zwhitened/3), 0, 1)` — the whitened arm is load-bearing because the
corpus has a large ambient bag correlation (mean raw z 15-51 per token from the global-T +
byte-composition limitation), so absolute Hamming barely separates members; whitening removes
the ambient and the raw arm caps repeat-heavy competitors. Chunks are ranked by IDF-weighted
coverage `sum_terms w · phi_term`.

**Measured scoreboard (`python eval_ring.py`, live 628,657-chunk store, 2026-08-06, ~161 s).**
25 frozen grep-verified questions (20 natural + 5 keyword controls), byte-offset gold, hit@3:

| config | nat@1 | nat@3 | kw@1 | kw@3 | all@3 | med ms | p90 ms |
|---|---|---|---|---|---|---|---|
| A v12-baseline (majority-bag Hamming) | 0.15 | 0.15 | 0.80 | 0.80 | 0.28 | 557 | 598 |
| B graded-scan (stop list + IDF weighted-Hamming) | 0.05 | 0.10 | **1.00** | **1.00** | 0.28 | 2684 | 2733 |
| **C cov-chunk — shipped `--retriever cov`** | 0.10 | **0.25** | 0.40 | 0.80 | **0.36** | 1847 | 4848 |
| D cov-window (query-time windows) | 0.10 | 0.15 | 0.60 | 0.60 | 0.24 | 1844 | 4853 |

**Honest gate result.** The pre-set gate was natural hit@3 ≥ **0.85**. The best measured config
(**C**) reaches **0.25 (5/20)** — the floor is **not** met. This was a pre-authorized outcome:
the release ships the **honest measured maximum + full per-miss autopsy**, and the eval set is
**frozen and unchanged**. Coverage (C) still **beats the v1.2 baseline** on naturals
(0.25 vs 0.15) and overall (0.36 vs 0.28) and does **not regress** the keyword controls
(both 4/5). Two measured design findings drove what shipped:

- **Windows hurt.** The plan's query-time sliding windows (config D) measured *worse* than plain
  chunk coverage (D 0.24 < C 0.36): windows spread credit to neighbours and let multi-term
  competitors win (e.g. W5's gold is chunk-coverage rank **1** but the window selection missed
  it). So the shipped `cov` retriever ranks by **chunk coverage**; the window code is retained
  for the eval's honest (negative) lift measurement only.
- **Surface-variant expansion hurts.** Adding case/plural variants (term = max over variants)
  credited chunks that merely contain a *different* word ("Specials"→"special"), dropping
  natural hit@3 from 4/20 to 2/20. So each term keeps a **single exact form**.

**Why the ceiling is here (autopsy).** Autopsy of the 10 wiki-re-runnable config-C misses
(corrected categorizer, `--subset wiki`): **7 `no-anchor-tie`** (the decoded winner genuinely
contains every query term) + **3 `ambient-noise`** (whitened credit on absent terms — W4
"largest", W8 "turn/sunlight", W11 "long/marathon"). The remaining 6 misses (P1-P5, K1) cannot
be re-categorized post-edit (sha guard); round-1 live forensics on P1 indicates ambient-noise
(whitened-noise ties), not surface-mismatch. `no-anchor-tie` is structural (per receipt R11): a 16-token gold chunk
mentions each query word once, while competitor chunks that are *about* one of those words
cover the query terms just as well — the answer's discriminating anchor isn't in the query
("What is the capital of Japan?" has no "Tokyo"). A genuinely compact/anchored corpus retriever
is future work; a lexical bag/coverage retriever is tie-bound at 628 k scale.

**Latency & the GUI.** `cov` retrieval is ~2-6 s (typically 4-16 batched per-token scans at
~0.55 s each) vs ~0.6 s for `v12`; a GUI turn goes from ~9-12 s to ~11-16 s. The GUI flows
through unchanged (new flags default sensibly). Use `--retriever v12` for the fast v1.2 path.

**Usage.**

```
python ring_cli.py query "What seed does the decoder ring use?" --topk 3          # cov (default)
python ring_cli.py query "GENESIS_SEED 0xDEC0DE01" --retriever v12 --topk 3        # fast v1.2 path
python eval_ring.py            # full 4-config scoreboard + autopsy
python eval_ring.py --config C --subset natural     # one config, gate subset
```

> **Store note (one edit stale).** Editing this README + CHANGELOG for v1.3 changes them on
> disk but **not** in the store: the ingested pages for `README.md`/`CHANGELOG.md` still carry
> the pre-edit bytes. This is intentional (v1.3 does **not** re-ingest). `eval_ring.py`'s sha
> integrity guard will detect the drift on the *next* run and print the exact realign command
> (`python ring_cli.py ingest --add <path>`); the v1.3 scoreboard above was captured **before**
> these doc edits, so its private-doc gold offsets are exact.
>
> **Do not blindly re-ingest these docs.** These v1.3 sections now embed the frozen eval's
> question texts and answer strings verbatim, so `ingest --add README.md CHANGELOG.md` would
> make the eval **self-referential** — the ingested pages would become lexical bait for the
> very questions they answer (e.g. W2's cov winner is already a v1.2 README chunk covering
> "chemical/symbol/gold"), silently inflating future private-doc scores. Re-ingest only after
> moving the eval question texts out of the docs, or knowingly accept the drift.

## Loading your own corpus (quickstart)

The complete path from "a folder of documents" to "grounded answers", start to finish:

```
ring.bat ingest --add "C:\path\to\manual.pdf" "C:\path\to\policies\*.md"
python ring_cli.py sembuild
ring.bat query "your natural-language question" --topk 3
```

1. **`ring.bat ingest --add <files…>`** — accepts `.txt` / `.md` / `.pdf`; each document becomes
   one atomic page in `store\`; re-ingesting the same resolved path REPLACES its page; per-file
   errors skip and continue. (Bare `ingest` without `--add` wipes and rebuilds the whole store —
   only use it deliberately.) `ring.bat stats` shows the doc table afterwards.
2. **`python ring_cli.py sembuild`** — **do not skip this.** It embeds the new documents into the
   semantic sidecar (island must be up; `ring.bat` starts it). It is resumable and only touches
   stale/new docs, so it is fast after the first build. **If the sidecar is stale, `--retriever
   auto` silently falls back to lexical-only retrieval** — natural-language questions drop to the
   v1.3-ceiling hit rate and nothing errors. `sembuild --status` shows per-doc coverage.
3. **`ring.bat query "…" --topk 3`** — full-stack retrieval + generation. Trust habit: read the
   retrieved chunks above the `====` separator; if they don't contain the answer, the model is
   free-styling (see "Retrieval quality").
4. Housekeeping: `ring.bat remove <doc>` drops one document (page + manifest, no rebuild);
   `--doc SUBSTR` scopes a query to matching docs. **The GUI now runs the FULL pipeline: a drag-drop
   or `/ingest` does step 1 AND step 2 automatically** (ingest, then auto-builds the semantic index
   in the background worker, progress streamed) — so from the GUI you skip the manual `sembuild`. If
   the island is down at drop time the ingest still lands and the build is deferred loudly, then
   auto-retried on the next query once the island returns.

## Named collections (v1.8)

A **collection** is an isolated store — its own manifest, pages, DF table, semantic sidecar (own
`mu`, own keys) and its own optional abstention tau — living under one shared ring (one codebook).
The driving use case: load an HR employee-handbook RAG and a customer-facing inventory RAG on the
same machine, switchable and never cross-contaminating.

```
ring.bat ingest --add "C:\hr\*.md" --collection hr     # creates stores\hr\ on first ingest (announced)
python ring_cli.py sembuild --collection hr            # (or let the GUI auto-build it on drop)
python ring_cli.py collections hr --set-tau 0.12       # SMALL corpora need this — see the warning below
ring.bat query "notice period for resignation?" --collection hr --topk 3
python ring_cli.py collections                         # list every collection + docs/chunks/cache/sidecar/tau
```

> **Small collections abstain at the default tau — this is by design, not a bug.** The default
> `0.505` was measured on the ~628k-chunk wiki corpus. A *tiny* collection (e.g. two HR docs, ~5
> semantic windows) scores far lower on its own grounded question — measured `sem_top ≈ 0.17` for
> "notice period for resignation?" on the 2-doc HR corpus — so the query above will show the
> `NO GROUNDED ANSWER IN THE STORE (0.17 < 0.505)` banner and **exit 3** until you calibrate. The
> per-collection tau IS the fix (`collections NAME --set-tau X`, measured working at `0.12` for that
> corpus), or use `--no-abstain` for a one-off. On tiny corpora the answerable/unanswerable `sem_top`
> ranges can even invert (a measured bait question scored `0.223 > 0.166`), so follow the honest
> **calibration recipe below** rather than trusting a guessed threshold.

- **Layout.** The legacy `store\` is the **DEFAULT** collection (zero migration, zero writes). A
  named collection lives at `stores\<name>\store\{manifest.json, collection.json, pages\, cache\,
  semantic\}` — a full store tree reached by *root aliasing* (`ring_store.collection_root`). Add
  `--collection NAME` to `ingest` / `remove` / `stats` / `sembuild` / `query`; omit it (or use
  `--collection default`) for the legacy store.
- **Data isolation, never a contract fork.** K, D, window/stride and every retrieval constant stay
  the frozen global contract — a collection isolates DATA only. The DF table and abstention state
  are per-collection: the word "termination" can saturate an HR corpus and appear once in an
  inventory corpus without either poisoning the other's high-DF filtering.
- **Creation rule.** A collection is created **implicitly on `ingest` only** (both `--add` and bare),
  announced loudly (`creating NEW collection 'hr' at stores\hr\store\`). Every other command errors
  on a nonexistent collection and lists the existing ones — a typo'd *query* can never spawn a store.
  Names are case-folded and must match `^[a-z0-9][a-z0-9_-]{0,31}$`; `default` is reserved.
- **Per-collection abstention tau.** Lives in the freshness-neutral `stores\<name>\store\collection.json`
  (NOT the manifest — a tau in the manifest would move its sha, rebuild the caches, and silently
  disable abstention). Absent → the global `0.505`. Set/clear with `python ring_cli.py collections
  NAME --set-tau X` / `--clear-tau` (refuses `default`). It survives `/rebuild` (tau calibrates the
  corpus DOMAIN, not the page set).

  **Calibration recipe (the honest minimal procedure).** Without labeled unanswerables a new
  collection **runs at the global default 0.505** (measured on the wiki corpus — a different corpus
  may sit elsewhere; the `collections` listing prints the `wiki-calibrated` caveat, and abstention
  never silently invents a threshold). To calibrate: author ≥10 answerable questions (answers
  verifiably present) and ≥5 unanswerable bait questions (vocabulary present, fact absent); run each
  through `python ring_cli.py query "<q>" --collection NAME --sem-top --no-abstain` and record the
  printed `sem_top`. If `max(unanswerable) < min(answerable)`, set tau to the midpoint via
  `collections NAME --set-tau X`; if the ranges overlap, keep the default and record the overlap —
  abstention on that corpus is honestly weaker.

- **GUI pipeline walkthrough.** `/collection <name>` switches (confirm-to-create on a new name);
  `/collections` lists; `/sembuild` rebuilds the active index. A drop or `/ingest` into the active
  collection = ingest + **auto incremental sembuild** (streamed). The status line shows
  `col: <name> · N docs · M chunks · sem: fresh|STALE|none|STALE (build pending)`.
- **Sembuild cost (measured 2026-08-07, ~80 texts/s).** A small collection auto-builds in seconds
  to ~20 s per 100k tokens. A drop into the **big DEFAULT collection** re-keys all ~157k windows
  when the sidecar is not slim (~2–3 min numpy + a ~160 MB `keys_matrix.npy` rewrite; streamed, the
  UI never freezes). Plan around that if you drop into `default`.
- **Migration tools are legacy-only.** New collections are born `store_version 4`; `migrate_store_v3.py`
  / `migrate_store_v4.py` target the legacy layout and never apply to a fresh collection.

## Usage

```
python gen_ring.py                     # (re)generate both .dring files; downloads tokenizer.json if absent
python ring_cli.py info atoms.dring    # dump + verify a file header
python ring_cli.py encode "The cat sat."
ring.bat ingest docs\hdc.txt docs\tokenizer.txt docs\beavers.txt   # fresh paged store (cold-starts the island)
ring.bat ingest --add notes.md paper.pdf                          # incremental; same resolved path REPLACES
ring.bat stats                                                     # docs table + df_summary + cache state
ring.bat query "Why must you disable the beginning of text marker when encoding a sentence?" --topk 2
ring.bat remove beavers.txt                                        # drop one doc (ident = name | doc_id | path)
python ring_cli.py query "..." --ollama http://127.0.0.1:11435    # point at another island / env RING_OLLAMA
python test_ring.py                    # T1-T4 codebook (~5 min; T4 dominates)
python test_store.py                   # T6-T11 store (~15 s)
python bench_ring.py                   # ~1M-token benchmark (~90 s; prints acceptance gates)
```

`ring.bat <args…>` is the front door: it ensures decoder-ring's own Ollama island is
listening on `127.0.0.1:11436` (cold-starting it minimized if not), then forwards to
`ring_cli.py`. The `query` demo hands the retrieved+reconstructed context to `llama3.2:1b` via
that island (RTX 2070 SUPER — see "Standalone Ollama island" below). Endpoint precedence:
`--ollama` flag, then env `RING_OLLAMA`, then `http://127.0.0.1:11436`. If the model is absent
at the chosen endpoint, the demo degrades gracefully: it prints the assembled context and
exactly `demo needs: ollama pull llama3.2:1b`, then exits 0 — it never pulls or substitutes a
model.

### Example queries (measured 2026-08-06 on the shipped 3-doc store, `--topk 2`, DF filter default 0.25)

> After running `wiki_ingest.py` the store also holds ~100 wiki volumes (it is **mixed**);
> scope a query to the demo docs with `--doc tokenizer` / `--doc beavers` / `--doc hdc`, or
> run `wiki_ingest.py --clean` to remove the wiki volumes again. See "Wikipedia test corpus".

| query | top-2 chunks (page provenance printed) | retrieved fact? |
|---|---|---|
| `Why must you disable the beginning of text marker when encoding a sentence?` | both `tokenizer.txt` — chunk 10 (z 78.7), chunk 9 (z 72.6); df-dropped `' the' ' of' ' a'` | **yes** — "disable the automatic beginning of text marker, or an extra special token" |
| `Which operation encodes order and position in hyperdimensional computing?` | both `hdc.txt` — chunk 6 (z 82.5), chunk 0 (z 79.8) | **yes** — "bit rotation … encodes order and position" |
| `What is the largest beaver dam?` | both `beavers.txt` — chunk 4 (z 77.6), chunk 0 (z 75.4); df-dropped `' is' ' the'` | **partial** — chunk 4 (the "…largest beaver dam ever recorded" lead-in) is rank 1 with *or* without DF; the ~850 m / Alberta *fact* lives in chunk 5, still not in the top-k (34/36 df-off → 32/36 df-on). The v1 doc-only limitation stands. |

Retrieval is by Hamming distance over per-chunk **bag** vectors; the retrieved chunk text is
reconstructed *from the hypervectors* (nearest-neighbour decode of the **seq** bundle, opened
from the hit's page), not from any cached token IDs. Chunks are `DEFAULT_FANOUT = 16` tokens,
so a full answer can span 2 chunks — use `--topk 2` (or 3). **DF query filtering** (default
`--df-max 0.25`) drops query tokens that occur in more than that fraction of corpus chunks
(function words — see the `df-dropped` notes above) before building the query bag; it is
purely query-side, so chunk bags stay pure majority. Its measured effect at this 3-doc scale
is to remove function-word dilution: on the tokenizer query it drops `' the' ' of' ' a'` and
trades top-1 z 92.6 → 78.7 (the *content* words now dominate the bag) while keeping the same
two chunks; it does **not** change which chunks rank top-2 on any of the three demo queries.
`--df-max 1.0` disables it and reproduces the v1 numbers exactly (the tokenizer query returns
z 92.6 / 92.1); `--doc SUBSTR` restricts retrieval to docs whose name matches.

## Standalone Ollama island (v1.1)

decoder-ring runs its **own** Ollama instance — zero runtime dependency on the `foreman` tree
(the env lines are *copied*, not called). `Start Ring Island.bat` launches `ollama serve` on
`127.0.0.1:11436` pinned to the **RTX 2070 SUPER**:

- Pinned by **GPU UUID** (`nvidia-smi -L`), not integer index — the index mismapped live once
  and loaded onto the 4080. Update the UUID in the bat if the 2070 is replaced.
- `OLLAMA_LLM_LIBRARY=cuda_v13` is **load-bearing**: this Ollama build also discovers GPUs
  through a Vulkan backend that ignores `CUDA_VISIBLE_DEVICES`; without the pin the 4080
  re-enters as `Vulkan0` and models land on it anyway.
- The model store is user-level and shared, so `llama3.2:1b` (already pulled) is visible with
  no new pull. The island **coexists** with the foreman island on 11435 on the same 8 GB card
  (gemma3:4b ~3.6 GB + llama3.2:1b ~1.5 GB fit); if VRAM is ever tight Ollama part-offloads
  layers to CPU — slower, but never the wrong GPU.

`ring.bat` netstat-checks 11436, cold-starts the island minimized and polls `/api/tags` until
ready (~20 s cap), then forwards to `ring_cli.py`. The endpoint default moved **11434 → 11436**
in v1.1: the bare 11434 default was the unpinned main instance (the 4080) — wrong GPU for this
pipeline. `--ollama` / `RING_OLLAMA` still override.

**Stopping the island — kill the process tree.** `ollama serve` spawns a child
`llama-server.exe` runner that holds the model in VRAM. Closing the "Ring Ollama Island"
window stops both; if you kill it by PID instead, use a **tree** kill
(`taskkill /PID <serve_pid> /T /F`) — a bare kill orphans the runner child, which keeps ~1.5 GB
of `llama3.2:1b` resident on the 2070 while `/api/ps` reports empty (the known Ollama orphan
gotcha). Verify with `nvidia-smi` that the 2070 returns to its baseline afterward.

## GUI (v1.2, full pipeline + collections in v1.8)

`gui.py` is a one-window dark-theme Tk chat over the store, forked (behavior copied, never
imported) from foreman's battle-tested shell. Launch it with **`Ring GUI.bat`** (double-click;
starts `pythonw gui.py`, no console) or `python gui.py`.

- **Semantics.** Plain typed text = a `query` (streamed live as each hit decodes). File drops
  or `/ingest <paths>` = `ingest --add` **followed automatically by an incremental `sembuild`** in
  the same background worker (v1.8 — the drag-drop now runs the FULL pipeline, not just ingest).
  `/rebuild <paths>` **confirms, then wipes the store and re-ingests fresh** (also auto-builds; it
  refuses locally if any path is missing — a rebuild with only bad paths would wipe the store to
  empty). `/collection <name>` switches the active collection (confirm-to-create), `/collections`
  lists them, `/sembuild` rebuilds the active index. `/remove <ident>`, `/stats`, `/topk N` (1–10),
  `/df X` (0–1.0), `/help`. Enter sends, Shift+Enter is a newline, quote paths with spaces.
- **Island-down deferral.** If the island is down when you drop files, the ingest still lands and the
  semantic build is deferred with a LOUD chat notice (status shows `sem: STALE (build pending)`);
  `/sembuild` retries, and the next query auto-runs the pending build once the island returns.
- **Process model.** Every command spawns `python -u ring_cli.py …` and streams its stdout
  into the chat (`-u` is load-bearing — without it a block-buffered child looks dead). The
  ring load is **~1.5 s warm** (tokenizers-codec dominated, *not* the 250 MB matrix) ≈ 10–15 %
  of a query turn; a **query turn is ~8–10 s** retrieval+decode + generation (measured 9.7 s /
  9.1 s on the demo store against the 11436 island), and the top hits appear as they decode.
  A resident ring was rejected (it would fork shipped `ring_cli`, lose crash isolation, and
  fight the Windows mmap-vs-cache-rebuild discipline for a ~1.5 s/turn saving). No "instant"
  claims — the numbers above are what it does.
- **Island ownership.** The GUI itself health-checks `127.0.0.1:11436` (HTTP `GET /api/tags`,
  never a subprocess) and cold-starts `Start Ring Island.bat` minimized only when the **default**
  island is down — never for a `RING_OLLAMA` override (that just prints context-only). **`Ring
  GUI.bat` deliberately does NOT pre-start the island** (single owner = no double-`serve` bind
  race — a deliberate deviation from foreman's `GUI.bat`, which pre-starts its servers). The
  status bar shows island state, the active **collection** + its docs/chunks + semantic-index
  freshness (`fresh`/`STALE`/`none`/`STALE (build pending)`), and the live `topk`/`df` knobs.
  Freshness is computed GUI-side with plain `json`+`hashlib` — the GUI never imports ring modules.

## Wikipedia test corpus (v1.2)

`wiki_ingest.py` builds a **retrieval stress corpus** from Simple English Wikipedia and ingests
a bounded slice into the shared `store/`. It is stdlib-only (no `mwparserfromhell`, no new pip
deps): download → stream-parse the `bz2` multistream with `xml.etree.iterparse` (`root.clear()`
per page, bounded RAM) → strip wikitext → pack ~100k-token `wiki/volumes/vol-NNNN.txt` files →
incrementally ingest.

- **Source** (verified live 2026-08-06):
  `https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles-multistream.xml.bz2`
  — **384,058,867 bytes** (366.3 MiB), `ETag "6a72308c-16e445f3"`, Last-Modified 2026-08-04,
  `Accept-Ranges: bytes` (so a partial `.part` Range-resumes). The download sends a **descriptive
  `User-Agent`** — Wikimedia returns HTTP 403 to the generic `Python-urllib` UA.
- **Bounds.** Default `--tokens 10_000_000` parses only the first ~15 % of the dump (bz2 reads
  lazily). `--all` builds/ingests the whole dump and prints a measured disk/time estimate first
  (opt-in; roughly 60–70 GB and ~1½ h — never run in the verification below). `--volumes-only`
  stops after the build; `--ingest-only` skips download/build.
- **Resume/skip.** A completed dump is reused offline (no network). Volumes are deterministic
  derived data; a volume is **skipped iff its normcased abspath AND sha256 already match the
  manifest** (skips still count toward the token bound). `doc_id = blake2b-8(path)`, so a
  changed volume REPLACES in place.
- **Mixed store + `--clean`.** After a run the store holds demo docs **and** wiki volumes.
  Scope with `ring_cli query --doc SUBSTR`; `wiki_ingest.py --clean` removes exactly the wiki
  volumes (by source-path prefix) and leaves the demo docs and the `wiki/` files on disk.
- **The stripper is deliberately approximate** (known-limitation 6): two tiny nesting scanners
  (`{{…}}` depth, `{|…|}` line-wise) + ordered regexes remove templates, tables, refs, and
  media/category links; entities are unescaped last. **Infoboxes are templates, so structured
  facts that live only in an infobox (e.g. a country's "capital = …" field) do not survive into
  the plain text** — they are only retrievable when the article's prose also states them.

### Measured run (2026-08-06, default `--tokens 10_000_000`)

| stage | measurement |
|---|---|
| download | 366.3 MiB @ ~2.5 MB/s ≈ **2.7 min** (Range-resumable) |
| build | **99 volumes · 14,144 articles · 10.00 M tokens** (~40 s; volumes are exact-token-counted) |
| ingest | **98 volumes** ingested at the bound · 10,046,447 wiki tokens · **627,952 wiki chunks** · ~15.9 k tok/s ≈ **10 min** |
| **total Goal-B wall** | **13 min 19 s** (gate: < 25 min) |
| resume re-run | **3.08 s** — all 98 volumes SKIP, +0 added, `manifest.json` sha unchanged (zero writes) |
| store now | **101 docs** (98 wiki + 3 demo) · 10,047,008 tokens · 627,988 chunks · df_summary 60,055 distinct tokens |
| footprint | dump 366.3 MiB + volumes 40.9 MiB + `store/pages` 2.57 GiB + `store/cache` 1.21 GiB ≈ **4.17 GiB** (C: 363 GiB free) |
| top wiki DF tokens | `','` 0.50 · `'.'` 0.42 · `' the'` 0.38 · `' '` 0.31 · `' of'` 0.29 · `' and'` 0.25 |

### Fact queries at 628 k-chunk scale (`--topk 3`, warm cache, ~9–12 s/turn incl. generation)

Retrieval was measured against the live 11436 island (cache pre-warmed by an operator session;
warm flat-scan over 627,988 chunks lands near the v1.1 design-point extrapolation of ~0.6 s). The
results are an **honest stress-test finding, not a success advertisement** — at this scale the
pure-majority BSC bag is dominated by high-frequency tokens and the global-T self-correlation:

| query (`--df-max` 0.25 unless noted) | top-3 (page · z) | fact retrieved? |
|---|---|---|
| *What is the capital of France?* | **vol-0094 z 72.0 "The capital city is Paris."**; then two `What is …` scaffold chunks (z 71.5) | **yes** — "Paris"/"capital" in top-1 (from a France-region article, not the main `France` article whose capital lived in a stripped infobox) |
| *What is the chemical symbol of gold?* | all 3 = a philosophy article's `What is truth?/mind?` chunks (z 80–82) | **no** — the `Au` fact (vol-0008) is out-ranked by the `What is … of` scaffold. At `--df-max 0.10` (drops `' is' ' the' ' of'`) the top-3 becomes on-topic chemistry — the *Chemical symbol* article (vol-0095 z 63) + Xenon `Xe` (vol-0012 z 62) — but still not the exact gold chunk |
| *What is the highest mountain on Earth?* | vol-0002 earth-science (z 79.5), vol-0065 nickel (z 79.2), vol-0082 *"highest mountain in North Island"* (z 79.0) | **no** — "Earth" dilutes and "highest mountain" matches the wrong mountain; the `Mount Everest` chunk (vol-0021) does not reach top-3 |

**Two findings the operator hit live, folded in honestly:**
1. **The 1b model answers from pretraining despite the "use ONLY the context" prompt.** On the
   gold/Everest queries above the assembled context did **not** contain the fact, yet
   `llama3.2:1b` still replied "Au" / "Mount Everest … Himalayas". A query whose distinctive terms
   are simply **absent** from the first-15 %-of-dump slice (e.g. *"Cabernet Sauvignon"*, article
   not in the subset) retrieves confidently-wrong chunks at z ≈ 83 and the model answers anyway —
   the demo is a retrieval+reconstruction showcase, **not** a grounded-QA guarantee.
2. **Question-scaffold tokens dilute the bag.** Interrogative/scaffold tokens (`What`, `' is'`)
   sit **under** the 0.25 DF threshold and survive the filter, so a natural-language question can
   be dominated by chunks that merely share its scaffold (the `What is …?` philosophy chunks
   above). Rare content words retrieve far better — a short keyword bag can still go noisy at 628 k
   chunks (the global-T artifact), but a *distinctive* term (the operator's `Voyager`, ×2) lands on
   the right chunk. Tightening `--df-max` and/or scoping with `--doc` both help.

### Mixed-store dilution (demo query after the wiki ingest)

`query "Why must you disable the beginning of text marker when encoding a sentence?" --topk 2`:
**bare** on the mixed store, the correct `tokenizer.txt` chunk (z 86.3) is pushed to **rank 2** by
a wiki "sentence/betray" chunk (vol-0062, z 87.8) and the 1b model is misled by the off-topic top
context; adding **`--doc tokenizer`** restores the exact v1.1 result (chunks 10 + 9, correct
answer). Dilution on a mixed store is expected and is what `--doc` / `--clean` are for.

## Store v4 (v1.7) — compressed token-id pages

`ingest` writes a **paged store** under `store/` instead of the single v1 `store.npz` blob
(operator preference: Minecraft-region-style paged snapshots, not a growing monolith). Pages
store the document's exact ingestion **token IDs**, not hypervectors — every vector is DERIVED.
**v1.7** DEFLATE-compresses each page (same members, `np.savez_compressed`); the live pages
shrank **53.14 MB → 18.63 MB (0.351×, 2.85× smaller)** with **zero reader changes** (`np.load`
decompresses npz members transparently):

```
store/
  manifest.json                  authoritative small index (format+store_version gate, df_summary)
  pages/<slug>-<doc_id>.npz      one self-contained page per doc — token_ids (<u4), chunk_len,
                                 sparse per-doc DF, JSON meta — DEFLATE-compressed (v4);
                                 provenance even if manifest is lost
  cache/                         DERIVED, lazily rebuilt, safe to delete anytime
    bag_matrix.npy               (N,256) <u8 all level-0 bags, COMPOSED from ids (mmap'd at query)
    chunk_doc/chunk_local/chunk_len.npy   per-chunk provenance
    df.npy                       (128256,) uint32 corpus chunk document-frequency (O(1) lookup)
    cache_meta.json              {manifest_sha256, n_chunks, doc_ids, built}
```

Chunk *c* = `token_ids[c·K : c·K + chunk_len[c]]` (K = fan-out 16). The bag scan matrix is
composed from the ids at cache-rebuild by `ring_store.compose_bags` — a bit-sliced ripple-carry
majority over packed uint64 words, **bit-exact** vs the frozen `rc.build_chunks` and ~4× faster
than an unpackbits path (~60 s over the whole corpus). Query/eval chunk text detokenizes the ids
directly; the pure-hypervector decode path survives as a verification mode (`bench` [5] composes
a seq bundle from the ids and round-trips it).

**Why compressed-uint32, not 17-bit packing** (the operator asked for bit-packing; measurement
overruled it). 128,256 IDs need 17 bits (uint16 impossible), so the two candidates were 17-bit
LSB packing vs plain DEFLATE of the uint32 members. Both were implemented and roundtrip-verified;
DEFLATE-on-uint32 won on **size AND simplicity** on every real page — bit-packing destroys
DEFLATE's byte-aligned repeat structure (a repeated token run stops being a repeated byte string
at 17-bit granularity), so `zlib(pack17)` is ~11 % *larger* than `zlib(uint32)`, and even raw
pack17 loses to it:

| page | tokens | raw u4 | **zlib(u4)** | zlib(pack17) | raw pack17 |
|---|---|---|---|---|---|
| vol-0001 (wiki) | 102,697 | 410,788 | **163,046** | 183,326 | 218,232 |
| README.md (drifted) | 7,256 | 29,024 | **11,837** | 13,052 | 15,419 |

Bit-packing would additionally force a custom codec into all 11 page-reader sites; DEFLATE needs
none. `zstandard` is NOT installed and beats zlib by too little here to justify a new pip dep
(zlib already delivers 0.351×) — noted as future work. Cost of compression: the per-hit chunk
render (`np.load` member slice) rises **0.5 ms → 1.8 ms** (+1.3 ms/hit ≈ +4 ms on a top-3 query,
invisible against the 2–5 s stack); ingest adds ~10 ms DEFLATE per wiki volume. Page *files* are
not byte-reproducible (npz zip headers embed a timestamp — already true in v2/v3), so every gate
is **array equality + downstream stdout**, never page-file bytes.

**Migrate** a store in place: `python migrate_store_v3.py` (v2→v3, if needed) then `python
migrate_store_v4.py` (v3→v4). The v4 tool is two-phase — recompress + **per-doc `np.array_equal`
on all four id arrays** into `store/pages_v4/`, then swap `pages → pages.v3.bak` and re-chain the
three freshness metas; the derived cache arrays are byte-identical (ids unchanged), so they are
**patched, never rebuilt**. Nothing swaps until all pages verify (any failure leaves the store on
v3). `pages.v2.bak` (v2 pages) and `pages.v3.bak` (v3 pages) are kept for rollback and are the
**operator's to delete** (~2.81 GB combined). Rollback recipe:

```
cd store
ren pages pages.v4.failed
ren pages.v3.bak pages
copy /Y manifest.v3.bak.json manifest.json
copy /Y cache\cache_meta.v3.bak.json cache\cache_meta.json
copy /Y semantic\meta.v3.bak.json semantic\meta.json
copy /Y semantic\cache\cache_meta.v3.bak.json semantic\cache\cache_meta.json
```

- `doc_id = blake2b-8(normcase(abspath(source)))`, so re-ingesting the **same resolved path
  REPLACES** it (same page filename, atomic overwrite). Synthetic corpora use `ids://<name>`.
- **Incremental:** bare `ingest` rebuilds fresh (printing what it replaces first); `ingest
  --add` appends/replaces; `remove <ident>` drops one doc (`ident` = name | doc_id | path);
  `stats` shows the docs table, `df_summary` top tokens and cache freshness.
- **Ingestion:** `.txt` / `.md` (UTF-8, replacement-decode fallback with a note) and `.pdf`
  (via `pypdf`; a missing pypdf skips just that file and the run continues). Unsupported types
  and empty extractions are per-file skips; the run exits 0 if at least one doc ingested.
- **Atomicity:** every write serialises to memory then `os.replace`s a `.tmp-<pid>` file, with
  a 5-try backoff on Windows `WinError 5/32` locks. Pages are written before the manifest;
  removes rewrite the manifest before deleting the page; the cache writes arrays before its
  `cache_meta.json` — so any torn write self-heals (an orphan page/stale cache is ignored and
  rebuilt). The cache rebuilds automatically whenever `sha256(manifest.json)` changes.
- The legacy `store.npz` + `docs.json` are left on disk, unused by v1.1, and safe to delete.

## File format

128-byte little-endian header (`struct` `<8sIII16sQI32s32s16x`) + payload of `n_vectors`
back-to-back vectors (2048 bytes each = 256 LE uint64, bit `b` at word `b//64`, bit `b%64`,
LSB-first).

| field | type | atoms.dring | tokens.dring |
|---|---|---|---|
| magic | 8s | `DRINGATM` | `DRINGTOK` |
| version | u32 | 1 | 1 |
| D | u32 | 16384 | 16384 |
| words_per_vec | u32 | 256 | 256 |
| prng_id | 16s | `SHA256-CTR-V1` | `SHA256-CTR-V1` |
| seed | u64 | 0xDEC0DE01 | 0xDEC0DE01 |
| n_vectors | u32 | 257 | 128256 |
| source_hash | 32s | zeros | sha256(tokenizer.json) |
| payload_sha256 | 32s | payload digest | payload digest |

## Frozen-seed contract

`GENESIS_SEED = 0xDEC0DE01`, `PRNG_ID = SHA256-CTR-V1`, the bit-packing layout above, and
`VERSION = 1` are the contract. A raw D-bit vector for stream `label` at `index` is the
concatenation of `sha256(b"DRINGv1|" + label + b"|" + struct.pack("<QQQ", seed, index, c))`
for `c` in 0..63 (64 × 32 B = 2048 B). Streams: `ATOM` 0..255 (byte atoms), `TIE` 0
(tie-break T), `SPEC` 0..255 (special token `128000+index`). Regular tokens are **composed**
from byte atoms (positional + bag + bigram planes, W=1/1/1); single-byte tokens degenerate
exactly to their byte atom. Change any contract value ⇒ bump `VERSION` and the magics.

### Recorded checksums (this build, 2026-08-06)

```
tokenizer.json  sha256  6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b
atoms.dring     payload e310cd28c99c3b9923eb6f781972502c635ce88b69d1ae8de641e41bd2f80c59
atoms.dring     file    eb379ee94dc3c50d4d7215c41a44245f24e1c836c9b899901140c363488e2214
tokens.dring    payload 14158d5be56e6c7fcc976f5176cd8d1b646836c695e4aee6fd00671e01f4a625
tokens.dring    file    ca0edec9ee00a1369c6401e32ceeb294b0ab38f24eaca047e62a1200e7c749e9
```

`python test_ring.py --full` regenerates both files into a temp dir and asserts these
whole-file digests match (T5). Generation wall-time on this box: **85.7 s** (composition of
128,256 token vectors = 84.1 s, vectorized numpy with a rotation cache).

## Measured on this machine (2026-08-06, numpy 2.3.2, Python 3.13.14)

- **T2 margins.** 5,000 random token pairs: mean ham 7406, median 7529, p1 5835, min 3411.
  Targeted worst-case surface pairs (`a`/`aa` … `intern`/`internal`) worst ham 2073 — all
  ≥ 2048 (sim ≤ 0.75 decode margin).
- **T3 robustness.** ` cat`~`cat` z=44.2, `teh`~`the` z=67.7, `catt`~`cat` z=77.1 — all
  well above the ratified z ≥ 10 and ≥ 3× the composed-token floor.
- **T4 capacity.** decode accuracy by fan-out K on **random** tokens: 8→1.000, 16→1.000,
  32→0.997, 64→0.992, 128→0.081, 256→0.003. Knee at **K = 16** ⇒ the shipped `DEFAULT_FANOUT`.
  **Honesty note (v1.5, measured on the live corpus):** the stored-seq Hamming NN decode is NOT
  lossless on *real text* — it errs on **~0.78 % of tokens / ~12 % of chunks** (surface-similar
  neighbours: a chunk's `'#'` decoding as `'##'`, `'.'` as `'..'`, `` '`' `` as `` '``' ``). This
  is why v1.5 pages store the exact token IDs: **ID reconstruction is exact** (the ids ARE the
  ingestion ids, proven per doc by the migration's bag bit-exact check). The pure-hypervector
  decode remains available as a verification mode (compose a seq bundle from the ids → identical
  vectors → identical decode), which is what keeps the `bench` decode==100 % gate exact.
- **Store v1.1 (`bench_ring.py`, 1 M tokens / 20 docs / 62,500 chunks).** ingest 16,602 tok/s,
  cold cache build 0.6 s, flat query median 59.6 ms (×10 ≈ 596 ms at the 625 k design point),
  round-trip decode 100 %, all 5 needles rank 1 — 88 s total, all 8 gates PASS. Store tests
  T6-T11 (`test_store.py`) pass in ~15 s.

## Known limitations (v1, by design)

1. **Composed tokens are self-correlated.** The single global tie-break vector T fills every
   *tied* plane bit with the same T, so unrelated composed tokens sit at median Hamming z ≈ 10
   (up to ≈ 32 for 1-2 byte tokens), not z ≈ 0. Pure-random vectors (the 256 specials) *do*
   sit at z ≈ 0, confirming the generator is unbiased — the elevation is purely the ratified
   global-T design. It does not break decode (clean = exact hash; K=16 bundles = 100%).
2. **DF filtering is query-side only** — chunk **bag** vectors stay pure majority (the frozen
   format is untouched). The dense corpus DF (`cache/df.npy`) drops high-frequency query
   tokens before building the query bag (`--df-max`, default 0.25), which sharpens within-doc
   ranking but does not reweight the stored chunks themselves.
3. **The chunk tree is stored but still unqueried.** Each page keeps its per-doc upper levels,
   but retrieval is a **flat** Hamming scan over the mmap'd bag matrix. Measured on the
   benchmark: **59.6 ms** at 62.5 k chunks (1 M tokens), linearly ~**0.6 s** at the 625 k-chunk
   design point — fast enough that a routed beam descent cannot show a clear win inside the
   benchmark bound, so routed mode is intentionally not implemented in v1.1.
4. Brute-force Hamming NN for chunk decode (~0.11 s / full-vocab scan) — no ANN index.
5. Single-writer store, no cross-process locking.
6. **The wikitext stripper (`wiki_ingest.py`) is deliberately approximate.** Templates,
   tables, refs, and media/category links are removed by a staged stdlib pipeline (two tiny
   nesting scanners + ordered regexes, no `mwparserfromhell`); rare markup residue is harmless
   to a retrieval stress test and is left as-is. See the "Wikipedia test corpus" section.
   
Feedback & Usage
If you use this project, I'd love to hear about it! Feel free to open an issue, 
drop a message, or reference this repository to let me know what cool things you
are building with it.

email: beeradise@gmail.com

## Acknowledgements & References

Decoder Ring builds upon the foundational mathematics and theoretical frameworks of Hyperdimensional 
Computing (HDC) and Vector Symbolic Architectures (VSA). 

Special acknowledgement goes to the researchers at UC Berkeley—particularly Pentti Kanerva and the Redwood 
Center for Theoretical Neuroscience—whose pioneering work on Sparse Distributed Memory and high-dimensional 
representational spaces laid the groundwork for this entire computing paradigm. Their research proved that 
high-dimensional random vectors could form robust, scalable architectures for cognitive computing. 
If you are interested in the academic roots of the deterministic token-to-hypervector approaches used in 
this engine, I highly recommend exploring their literature:
*   **Kanerva, P. (2009).** *Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors.* Cognitive Computation.
*   *Also thanks to Dennis Kleyko and Ryan Moughan*

The open-source release of this deterministic codebook is intended to further the practical, local, and 
reproducible application of these concepts in modern Retrieval-Augmented Generation (RAG) systems.


   
