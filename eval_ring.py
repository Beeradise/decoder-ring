r"""eval_ring.py — Decoder Ring v1.6 scored retrieval eval (semantic-attributed + abstention).

Retrieval-only. Imports ring_core + ring_store + ring_sem, loads the ring ONCE, ensure_cache
ONCE, and loads the semantic sidecar ONCE. Hits are scored by chunk index (no generation,
decode only to quote a miss winner in the autopsy).

EVAL SETS (v1.6, plan 3.1/3.6). `--set {A,B,AB}` (default A). Set A = the 25 FROZEN v1.4
questions, verbatim, the comparability anchor. Set B = 15 natural + 5 keyword held-out
validation questions (grep-verified, planner-authored+frozen 2026-08-07 BEFORE any calibration
sim) + 7 UNANSWERABLE questions (answers grep-verified ABSENT, absence proofs in 01-plan.md
section 3.2). Unanswerable questions never touch GoldResolver (they have empty docs/answers and
would exit 2); they are scored ONLY by the abstention metrics.

ABSTENTION (v1.6, plan 3.3, REPORTING layer here): abstain iff max(sem_win) < SEM_ABSTAIN_TAU
(0.505). Printed as its own block after the controls; retrieval rows are unchanged.

ISLAND-CONTACT AMENDMENT (v1.4, extended v1.6): eval_ring may call /api/embed ONLY (never
/api/generate), and only on a cache MISS. All 52 query embeds (set A 25 + set B answerable 20 +
7 unanswerable) are cached in store\semantic\cache\eval_queries.npz after the first --set AB run,
so steady-state eval runs are fully offline. A missing sidecar prints the exact sembuild command
and exits 2 (the semantic rows and the drifted-doc gold both need it).

Six attributable rows share scan work so the whole run stays under budget (S13):
  A v12-baseline : the exact v1.2 majority-bag Hamming path (retrieve_v12)          [history]
  C cov-chunk    : term-coverage membership, chunk_cov ranking (MUST equal v1.3 exactly)
  S sem-only     : rank chunks by the semantic channel alone (window expansion)     [isolation]
  F cov+sem      : chunk_cov/Sw + sem_weight*sem(c), full corpus, NO rerank/exact/neighbor
  G15 v1.5-cal   : ring_sem full rerank, v1.5 calibration (uncapped) — the continuity row
  G full-stack   : G15 + doc-cap-2 diversity in top-k selection (SEM_DOC_CAP)        [shipped]
C/F/G share one cov_compute; S/F/G15/G share one query embed + one key scan per question.

GOLD RE-RESOLUTION (v1.4, plan 5.1): the v1.3 hard sha-guard is replaced by a PROVIDER SWITCH.
An undrifted gold doc resolves from DISK exactly as before (byte cumsum + whitespace-flex regex
+ span->chunks). A drifted doc (README.md/CHANGELOG.md today, edited in v1.3 but never
re-ingested by design) resolves from the STORE PAGES via the sidecar's decoded-text cache, so
its gold chunk set matches the ingested page. The sha mismatch becomes an INFO line, not exit 2.

GOLD RESOLUTION NOTE (carried from v1.3): answer substrings match with WHITESPACE-RUN
FLEXIBILITY (runs collapse to \s+), folding the wiki source's NBSP inside numeric quantities
(e.g. "8,849\xa0meters" for W1). The frozen questions/answers are unchanged; only the match
mechanic tolerates the source whitespace.

Run:  python eval_ring.py                    # set A: 6-row scoreboard + attribution + autopsy
      python eval_ring.py --set AB           # set A + set B tables + abstention + A/B-validated
      python eval_ring.py --set B            # held-out set B + unanswerable abstention block
      python eval_ring.py --config G         # one row
      python eval_ring.py --subset natural   # gate subset (natural questions only)
      python eval_ring.py --sweep-cal        # calibration sweep (32 combos, set A only, in-memory)
"""

import argparse
import collections
import hashlib
import io
import os
import re
import sys
import time

import numpy as np

import ring_core as rc
import ring_sem as sm
import ring_store as rs

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
OLLAMA = os.environ.get("RING_OLLAMA", "http://127.0.0.1:11436").rstrip("/")

Q = collections.namedtuple("Q", "id question docs answers kind")

# ---------------------------------------------------------------- frozen eval set (FROZEN)
NATURAL = [
    Q("W1", "How tall is Mount Everest?", ["vol-0050.txt"],
      ["8,849 meters", "8,848 metres"], "natural"),
    Q("W2", "What is the chemical symbol of gold?", ["vol-0008.txt"],
      ["symbol is Au"], "natural"),                                   # * dev-overlap
    Q("W3", "What is the capital of France?", ["vol-0094.txt"],
      ["capital city is Paris"], "natural"),                          # * dev-overlap
    Q("W4", "Which ocean is the largest on Earth?", ["vol-0006.txt"],
      ["Pacific Ocean is the largest ocean"], "natural"),
    Q("W5", "What is the capital of Japan?", ["vol-0003.txt"],
      ["Tokyo is the capital of Japan"], "natural"),                  # no-anchor structural hard
    Q("W6", "Who was the first person to walk on the Moon?",
      ["vol-0003.txt", "vol-0032.txt", "vol-0039.txt", "vol-0041.txt"],
      ["first person to walk on the Moon"], "natural"),
    Q("W7", "What is the capital city of Australia?", ["vol-0014.txt"],
      ["Canberra is the capital city of Australia"], "natural"),
    Q("W8", "What process do plants use to turn sunlight into food?",
      ["vol-0001.txt", "vol-0009.txt"],
      ["Photosynthesis is the process", "Photosynthesis is a process"], "natural"),
    Q("W9", "What is the speed of light?",
      ["vol-0004.txt", "vol-0006.txt", "vol-0009.txt", "vol-0011.txt", "vol-0057.txt"],
      ["299,792"], "natural"),
    Q("W10", "Which river is the longest in the world?", ["vol-0005.txt"],
      ["Nile is the longest river"], "natural"),
    Q("W11", "How many kilometres long is a marathon?", ["vol-0019.txt"],
      ["run 42.195 kilometres"], "natural"),                         # * dev-overlap
    Q("W12", "Which planet is the hottest?", ["vol-0003.txt"],
      ["hottest planet is Venus"], "natural"),
    Q("W13", "Which planet is closest to the Sun?", ["vol-0003.txt", "vol-0012.txt"],
      ["closest planet to the Sun"], "natural"),                     # no-anchor structural hard
    Q("W14", "At what temperature does water boil?",
      ["vol-0001.txt", "vol-0003.txt", "vol-0005.txt", "vol-0018.txt", "vol-0052.txt", "vol-0098.txt"],
      ["boils at 100"], "natural"),                                  # * dev-overlap
    Q("W15", "What is the highest mountain on Earth?", ["vol-0021.txt", "vol-0050.txt"],
      ["is the highest mountain on Earth"], "natural"),              # * dev-overlap
    Q("P1", "What seed does the decoder ring use?", ["README.md", "CHANGELOG.md"],
      ["0xDEC0DE01"], "natural"),                                    # * dev-overlap
    Q("P2", "Which port does the decoder ring Ollama island listen on?",
      ["README.md", "CHANGELOG.md"], ["11436"], "natural"),          # * dev-overlap
    Q("P3", "Which GPU does the decoder ring island run on?",
      ["README.md", "CHANGELOG.md"], ["RTX 2070 SUPER"], "natural"),
    Q("P4", "How many tokens are in a chunk by default?", ["README.md", "CHANGELOG.md"],
      ["DEFAULT_FANOUT = 16", "K=16", "K = 16"], "natural"),         # * dev-overlap
    Q("P5", "How many bits wide are the ring hypervectors?", ["README.md", "CHANGELOG.md"],
      ["D = 16384", "D=16384", "16384 bits"], "natural"),
]
KEYWORD = [
    Q("K1", "GENESIS_SEED 0xDEC0DE01 SHA256-CTR-V1", ["README.md", "CHANGELOG.md"],
      ["0xDEC0DE01"], "keyword"),                                    # * dev-overlap
    Q("K2", "largest beaver dam ever recorded Alberta", ["beavers.txt"],
      ["largest beaver dam ever recorded", "northern Alberta"], "keyword"),
    Q("K3", "bit rotation encodes order and position hypervector", ["hdc.txt"],
      ["encodes order and position"], "keyword"),
    Q("K4", "Mount Everest Himalayas Nepal China", ["vol-0021.txt"],
      ["Mount Everest is in the Himalayas"], "keyword"),
    Q("K5", "disable automatic beginning of text marker extra special token", ["tokenizer.txt"],
      ["disable the automatic beginning of text marker"], "keyword"),  # * dev-overlap
]
ALL_Q = NATURAL + KEYWORD
WIKI = [q for q in ALL_Q if not any(d in ("README.md", "CHANGELOG.md") for d in q.docs)]

# the 16 v1.3 config-C misses (audit r2 verified) for the attribution matrix (5.5)
V13_MISSES = ["W1", "W2", "W3", "W4", "W7", "W8", "W9", "W11", "W12", "W14",
              "P1", "P2", "P3", "P4", "P5", "K1"]

# ---------------------------------------------------------------- FROZEN eval set B (v1.6, plan 3.1)
# Held-out validation + abstention corpus. Authored+grep-verified 2026-08-07 BEFORE any calibration
# was simulated. Every (docs, answers) pair was verified with the resolver's own whitespace-flex
# regex against the exact texts it reads (disk for sha-intact docs; decoded text\*.json.gz for the
# drifted README/CHANGELOG). Two-way: every gold doc contains an answer AND no answer leaks into an
# unlisted ingested doc. DO NOT EDIT — any wording change invalidates the planner's measured receipts.
NATURAL_B = [
    Q("B1",  "What do bees make honey from?", ["vol-0013.txt"],
      ["honey from nectar"], "natural"),
    Q("B2",  "How many legs does a spider have?", ["vol-0060.txt"],
      ["They have eight legs, and mouthparts"], "natural"),
    Q("B3",  "Which language do people speak in Brazil?",
      ["vol-0001.txt", "vol-0020.txt", "vol-0074.txt"],
      ["official language of Brazil", "Portuguese, the language of Brazil"], "natural"),
    Q("B4",  "Who invented the telephone?", ["vol-0014.txt"],
      ["invented the telephone"], "natural"),
    Q("B5",  "What do giant pandas eat?", ["vol-0073.txt"],
      ["giant pandas eat bamboo"], "natural"),
    Q("B6",  "In which year did World War II end?",
      ["vol-0007.txt", "vol-0009.txt", "vol-0012.txt", "vol-0013.txt", "vol-0020.txt",
       "vol-0021.txt", "vol-0038.txt", "vol-0055.txt", "vol-0066.txt", "vol-0072.txt",
       "vol-0078.txt", "vol-0088.txt"],
      ["ended in 1945", "World War II ends", "end of World War II in 1945"], "natural"),
    Q("B7",  "How many bones are in the adult human skeleton?",
      ["vol-0012.txt", "vol-0030.txt", "vol-0055.txt"], ["206 bones"], "natural"),
    Q("B8",  "How long does it take the Earth to orbit the Sun once?", ["vol-0006.txt"],
      ["365 days (except in a leap year)", "takes the Earth to go completely around"],
      "natural"),
    Q("B9",  "Who wrote the play Romeo and Juliet?", ["vol-0048.txt"],
      ["Romeo and Juliet is a play written by William Shakespeare"], "natural"),
    Q("B10", "How many arms does an octopus have?", ["vol-0065.txt"],
      ["octopus trails its eight arms"], "natural"),
    Q("B11", "Who discovered penicillin?", ["vol-0001.txt", "vol-0083.txt"],
      ["penicillin in 1928 by Alexander Fleming",
       "discovering the antibiotic substance penicillin"], "natural"),
    Q("B12", "How many token IDs are in the ring vocabulary?",
      ["README.md", "CHANGELOG.md"], ["128,256"], "natural"),
    Q("B13", "How long did generating the token vector file take?",
      ["README.md", "CHANGELOG.md"], ["85.7 s"], "natural"),
    Q("B14", "How big is the Simple English Wikipedia dump the ring downloads?",
      ["README.md", "CHANGELOG.md"], ["384,058,867", "366.3 MiB"], "natural"),
    Q("B15", "How many byte atoms does the atoms file store?",
      ["README.md", "CHANGELOG.md"],
      ["256 byte atoms", "256 pure-random byte atoms"], "natural"),
]
KEYWORD_B = [
    Q("BK1", "RMS Titanic iceberg Newfoundland 1912", ["vol-0001.txt"],
      ["Titanic sinks near Newfoundland after hitting an iceberg"], "keyword"),
    Q("BK2", "Eiffel Tower Paris metal girders concrete",
      ["vol-0001.txt", "vol-0005.txt", "vol-0069.txt"],
      ["Eiffel Tower in Paris"], "keyword"),
    Q("BK3", "Great Wall of China mountains Beijing",
      ["vol-0001.txt", "vol-0004.txt", "vol-0006.txt", "vol-0008.txt", "vol-0012.txt",
       "vol-0026.txt", "vol-0032.txt", "vol-0036.txt", "vol-0053.txt", "vol-0071.txt",
       "vol-0072.txt", "vol-0088.txt", "vol-0096.txt"],
      ["Great Wall of China"], "keyword"),
    Q("BK4", "atomic write os.replace tmp backoff", ["README.md", "CHANGELOG.md"],
      ["os.replace"], "keyword"),
    Q("BK5", "running water triggers beavers patch leak", ["beavers.txt"],
      ["sound of running water triggers"], "keyword"),
]
# Unanswerable (7): answers grep-verified ABSENT from all 101 ingested sources + the two decoded
# gz texts (absence proofs in 01-plan.md 3.2). kind="unanswerable", empty docs/answers -> BYPASS
# GoldResolver entirely; scored only by the abstention block.
UNANSWERABLE_B = [
    Q("U1", "Who is the CEO of OpenAI?", [], [], "unanswerable"),
    Q("U2", "What is the melting point of tungsten?", [], [], "unanswerable"),
    Q("U3", "Who created the cryptocurrency Ethereum?", [], [], "unanswerable"),
    Q("U4", "What does the band name BTS stand for?", [], [], "unanswerable"),
    Q("U5", "What is the wingspan of the Airbus A380?", [], [], "unanswerable"),
    Q("U6", "Who won the men's 100 metres at the 2024 Summer Olympics?", [], [],
      "unanswerable"),
    Q("U7", "In which year was the RTX 2070 SUPER released?", [], [], "unanswerable"),
]
ALL_B = NATURAL_B + KEYWORD_B


# ---------------------------------------------------------------- gold resolution helpers (5.1)
def _char_byte_cumsum(text):
    """Cumulative UTF-8 byte offset per character: int64 array of length len(text)+1."""
    cp = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    blen = np.ones(cp.shape[0], dtype=np.int64)
    blen[cp >= 0x80] = 2
    blen[cp >= 0x800] = 3
    blen[cp >= 0x10000] = 4
    cum = np.empty(cp.shape[0] + 1, dtype=np.int64)
    cum[0] = 0
    np.cumsum(blen, out=cum[1:])
    return cum


def _tok_byte_cumsum(codec, ids):
    """Cumulative byte offset per token (exact ingestion tokenization): length len(ids)+1."""
    lens = np.fromiter((len(b) for b in codec.ids_to_bytes(ids)), dtype=np.int64, count=len(ids))
    cum = np.empty(len(ids) + 1, dtype=np.int64)
    cum[0] = 0
    np.cumsum(lens, out=cum[1:])
    return cum


def _page_tok_byte_cumsum(codec, chunk_texts, chunk_lens):
    """Token-byte cumsum for a DRIFTED doc reconstructed from decoded chunk texts (5.1).

    Each chunk contributes exactly chunk_len token slots (16, last = its len) so token index
    t maps to chunk t//16 by the SAME _span_to_chunks machinery. Real per-token byte lengths
    when the chunk re-encodes to chunk_len tokens (the common case); otherwise the chunk's byte
    length is split across its chunk_len slots (chunk boundaries stay exact -> gold chunk mapping
    is preserved)."""
    lens = []
    for txt, cl in zip(chunk_texts, chunk_lens):
        ids = codec.encode_ids(txt)
        if len(ids) == cl:
            lens.extend(len(b) for b in codec.ids_to_bytes(ids))
        else:                                             # fallback: even split over cl slots
            nb = len(txt.encode("utf-8"))
            base, rem = divmod(nb, cl)
            lens.extend([base + (1 if j < rem else 0) for j in range(cl)])
    cum = np.empty(len(lens) + 1, dtype=np.int64)
    cum[0] = 0
    np.cumsum(np.asarray(lens, dtype=np.int64), out=cum[1:])
    return cum


def _ws_regex(sub):
    """Case-insensitive matcher whose whitespace runs match \\s+ (folds the source's NBSP)."""
    parts = [re.escape(p) for p in re.split(r"\s+", sub.strip()) if p]
    return re.compile(r"\s+".join(parts), re.IGNORECASE)


def _span_to_chunks(tokcum, doc_offset, K, a, b):
    """Byte span [a,b) -> set of global chunk indexes (a straddle yields BOTH chunks)."""
    ta = int(np.searchsorted(tokcum, a, side="right")) - 1
    tz = int(np.searchsorted(tokcum, b, side="left"))
    ta = max(ta, 0)
    return {doc_offset + (t // K) for t in range(ta, max(tz - 1, ta) + 1)}


def check_source_sha(doc_entry):
    """(ok, actual_sha): the manifest's frozen source_sha256 vs the file on disk right now."""
    with open(doc_entry["source_path"], "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    return actual == doc_entry["source_sha256"], actual


def hit_at_k(order, gold, k):
    """True iff at least one of the first k returned chunks is in the gold set."""
    return any(int(c) in gold for c in list(order)[:k])


class GoldResolver:
    """Resolve questions to gold chunk sets. Undrifted docs resolve from disk; drifted docs
    resolve from the store pages via the sidecar's decoded-text cache (provider switch, 5.1)."""

    def __init__(self, cache, codec, sidecar=None, ring_dir=None):
        self.codec = codec
        self.sidecar = sidecar
        self.ring_dir = ring_dir or (sidecar.ring_dir if sidecar is not None else HERE)
        self.K = int(cache.manifest["fanout"])
        self.by_name = {d["name"]: d for d in cache.manifest["docs"]}
        off, self.offset = 0, {}
        for d in cache.manifest["docs"]:
            self.offset[d["doc_id"]] = off
            off += int(d["n_chunks"])
        self._doc_cache = {}

    def _doc(self, name):
        if name in self._doc_cache:
            return self._doc_cache[name]
        d = self.by_name.get(name)
        if d is None:
            print(f"INTEGRITY: gold doc {name!r} is not in the manifest", file=sys.stderr)
            sys.exit(2)
        ok, actual = check_source_sha(d)
        if ok:
            text = open(d["source_path"], encoding="utf-8").read()
            cum_char = _char_byte_cumsum(text)
            tokcum = _tok_byte_cumsum(self.codec, self.codec.encode_ids(text))
        else:
            print(f"gold doc {name} drifted (manifest {d['source_sha256'][:12]}.. vs disk "
                  f"{actual[:12]}..) -> resolving from store pages")
            chunk_texts = self._decoded_chunks(name, d)
            text = "".join(chunk_texts)
            cum_char = _char_byte_cumsum(text)
            with np.load(rs.page_fs_path(self.ring_dir, d["page"])) as npz:
                clen = [int(x) for x in npz["chunk_len"]]
            tokcum = _page_tok_byte_cumsum(self.codec, chunk_texts, clen)
        rec = (d, text, cum_char, tokcum)
        self._doc_cache[name] = rec
        return rec

    def _decoded_chunks(self, name, d):
        gz = os.path.join(sm.semantic_paths(self.ring_dir)[3], d["doc_id"] + ".json.gz")
        if self.sidecar is None or not os.path.exists(gz):
            print(f"INTEGRITY: gold doc {name} drifted and no decoded cache; run:\n"
                  f"  python ring_cli.py sembuild", file=sys.stderr)
            sys.exit(2)
        return sm._read_gz(gz)["chunk_texts"]

    def resolve(self, q):
        gold = set()
        for name in q.docs:
            d, text, cum_char, tokcum = self._doc(name)
            doc_off = self.offset[d["doc_id"]]
            for sub in q.answers:
                for m in _ws_regex(sub).finditer(text):
                    a = int(cum_char[m.start()]); b = int(cum_char[m.end()])
                    gold |= _span_to_chunks(tokcum, doc_off, self.K, a, b)
        if not gold:
            print(f"INTEGRITY: {q.id} resolved 0 gold chunks (answers not found in gold docs)",
                  file=sys.stderr)
            sys.exit(2)
        return gold

    def doc_text(self, name):
        return self._doc(name)[1]


# ---------------------------------------------------------------- query embed cache (5.6)
class QueryEmbedder:
    """25-query embed cache in store\\semantic\\cache\\eval_queries.npz (key = sha256 of the
    prefixed question). Island /api/embed on cache miss ONLY; offline once warm."""

    def __init__(self, ring_dir, endpoint):
        self.endpoint = endpoint
        self.path = os.path.join(sm.semantic_paths(ring_dir)[4], "eval_queries.npz")
        self.cache, self.dirty = {}, False
        if os.path.exists(self.path):
            with np.load(self.path) as z:
                self.cache = {k: z[k] for k in z.files}

    def __call__(self, question):
        key = hashlib.sha256((sm.SEM_QUERY_PREFIX + question).encode("utf-8")).hexdigest()
        v = self.cache.get(key)
        if v is None:
            v = sm.embed_texts(self.endpoint, [sm.SEM_QUERY_PREFIX + question])[0]
            self.cache[key] = v
            self.dirty = True
        return v

    def save(self):
        if not self.dirty:
            return
        buf = io.BytesIO()
        np.savez(buf, **self.cache)
        rs.atomic_write_bytes(self.path, buf.getvalue())
        self.dirty = False


# ---------------------------------------------------------------- per-question primitives
def compute_primitives(cache, tm, T, codec, sidecar, qembed, q, topk, sem_weight):
    """One shared scan pass per question -> the primitives every row/sweep/attribution reuses:
    cov_compute, query embed, key scan, neighbor credit, candidates, exact credit at floors
    {0, 0.25}. Returns a dict (res is None only on an all-stop query -> rows fall back to A)."""
    P = {"id": q.id}
    t0 = time.perf_counter()
    A_order = list(int(c) for c in rs.retrieve_v12(cache, tm, T, codec, q.question, topk, 0.25, None)[0])
    P["A_lat"] = (time.perf_counter() - t0) * 1000.0
    P["A"] = A_order

    t0 = time.perf_counter()
    res = rs.cov_compute(cache, tm, codec, q.question, None)
    P["t_cov"] = time.perf_counter() - t0
    P["res"] = res
    if res is None:                                       # all-stop (not in the frozen set)
        for r in ("C", "S", "F", "G15", "G"):
            P[r] = A_order
            P[r + "_lat"] = P["A_lat"]
        P["sem_top"] = None                               # no scan -> an all-stop question never abstains
        P["all_stop"] = True
        return P
    P["all_stop"] = False

    q_emb = qembed(q.question)
    t0 = time.perf_counter()
    sem_chunk, sem_win = sm.semantic_scan(sidecar, q_emb)
    P["t_scan"] = time.perf_counter() - t0
    P["sem_chunk"], P["sem_win"] = sem_chunk, sem_win
    P["sem_top"] = float(sem_win.max())                   # abstention signal (full-corpus, no doc_keep)

    _tpc_n, nbcov = sm.neighbor_credit(res, cache.chunk_doc)
    cand = sm.sem_candidates(cache, sidecar, sem_win, nbcov)
    # credits at the sweep grid floors {0, 0.25} AND the shipped floor (deduped), so G tracks the
    # module constant and the sweep still has its fixed {0, 0.25} grid
    creds, Sw, _ = sm.exact_credit(cache, sidecar, res, cand, (0.0, 0.25, sm.SEM_PHANTOM_FLOOR))
    P["nbcov"], P["cand"], P["creds"], P["Sw"] = nbcov, cand, creds, Sw

    # C: chunk_cov ranking (MUST equal v1.3)
    t0 = time.perf_counter()
    C_order = list(int(c) for c in np.argsort(-res.chunk_cov, kind="stable")[:topk])
    P["C"], P["C_lat"] = C_order, (P["t_cov"] + (time.perf_counter() - t0)) * 1000.0
    # S: semantic-only
    t0 = time.perf_counter()
    S_order = list(int(c) for c in np.argsort(-sem_chunk, kind="stable")[:topk])
    P["S"], P["S_lat"] = S_order, (P["t_scan"] + (time.perf_counter() - t0)) * 1000.0
    # F: additive channel over the full corpus (no rerank/exact/neighbor)
    t0 = time.perf_counter()
    fscore = res.chunk_cov / Sw + sem_weight * sem_chunk
    F_order = list(int(c) for c in np.argsort(-fscore, kind="stable")[:topk])
    P["F"], P["F_lat"] = F_order, (P["t_cov"] + P["t_scan"] + (time.perf_counter() - t0)) * 1000.0
    # G15: full-stack rerank at the v1.5 calibration (uncapped) — the continuity row
    t0 = time.perf_counter()
    G15_order, _sc15 = sm.score_and_rank(cand, creds[float(sm.SEM_PHANTOM_FLOOR)], nbcov, sem_chunk,
                                         Sw, topk, sm.SEM_TIE_A, sem_weight)
    P["G15"], P["G15_lat"] = G15_order, (P["t_cov"] + P["t_scan"] + (time.perf_counter() - t0)) * 1000.0
    # G: shipped v1.6 rerank = G15 + doc-cap-2 diversity in top-k selection (SEM_DOC_CAP)
    t0 = time.perf_counter()
    G_order, _sc = sm.score_and_rank(cand, creds[float(sm.SEM_PHANTOM_FLOOR)], nbcov, sem_chunk,
                                     Sw, topk, sm.SEM_TIE_A, sem_weight,
                                     cache.chunk_doc, sm.SEM_DOC_CAP)
    P["G"], P["G_lat"] = G_order, (P["t_cov"] + P["t_scan"] + (time.perf_counter() - t0)) * 1000.0
    return P


# ---------------------------------------------------------------- attribution (5.5)
def _g_hit_with(cache, sidecar, P, gold, topk, floor, tie_a, sem_weight, gamma):
    """Ablation probe: does the full rerank land a gold chunk in top-k under these knobs?
    Reuses the question primitives (sem_chunk/sem_win/res) — no new scans/embeds."""
    res, sem_chunk, sem_win = P["res"], P["sem_chunk"], P["sem_win"]
    _tpc, nbcov = sm.neighbor_credit(res, cache.chunk_doc, gamma=gamma)
    cand = sm.sem_candidates(cache, sidecar, sem_win, nbcov)
    creds, Sw, _ = sm.exact_credit(cache, sidecar, res, cand, (floor,))
    order, _s = sm.score_and_rank(cand, creds[float(floor)], nbcov, sem_chunk, Sw, topk,
                                  tie_a, sem_weight)
    return hit_at_k(order, gold, topk)


def _gold_semantic_rank(P, gold):
    """Rank of the best gold chunk by the semantic channel among the candidate set (1-based),
    plus whether a gold chunk entered ONLY through injection (not the top-1024 nbcov shortlist)."""
    cand, sem_chunk, nbcov = P["cand"], P["sem_chunk"], P["nbcov"]
    gold_c = [c for c in gold if c in set(cand)]
    if not gold_c:
        return None, "not-in-candidates"
    best = max(gold_c, key=lambda c: float(sem_chunk[c]))
    rank = int(sum(1 for c in cand if float(sem_chunk[c]) > float(sem_chunk[best]))) + 1
    short = set(int(c) for c in np.argsort(-nbcov, kind="stable")[:sm.SEM_C_SHORT])
    entry = "inject" if not any(c in short for c in gold_c) else "lexical-shortlist"
    return rank, entry


def recovered_by(cache, sidecar, P, gold, topk, sem_weight):
    """First matching rule (5.5) for a G-recovered v1.3 miss. Reuses P's primitives."""
    rank, entry = _gold_semantic_rank(P, gold)
    if entry == "inject":
        return "semantic-inject"
    ship = dict(floor=sm.SEM_PHANTOM_FLOOR, tie_a=sm.SEM_TIE_A, sem_weight=sem_weight,
                gamma=sm.SEM_NB_GAMMA)
    if not _g_hit_with(cache, sidecar, P, gold, topk, **{**ship, "sem_weight": 0.0}):
        return "semantic-rerank"
    if not _g_hit_with(cache, sidecar, P, gold, topk, **{**ship, "floor": 1.0}):
        return "exact-credit"
    if not _g_hit_with(cache, sidecar, P, gold, topk, **{**ship, "gamma": 0.0}):
        return "neighbor-credit"
    return "combination"


# ---------------------------------------------------------------- autopsy (5.5)
def _decode_chunk_text(ring_dir, cache, tm, codec, c):
    docs = cache.manifest["docs"]
    di = int(cache.chunk_doc[c]); loc = int(cache.chunk_local[c])
    K = int(cache.manifest["fanout"])
    with np.load(rs.page_fs_path(ring_dir, docs[di]["page"])) as npz:
        klen = int(npz["chunk_len"][loc])
        ids = npz["token_ids"][loc * K:loc * K + klen]     # v3: exact ingestion ids (was NN decode)
    return codec.ids_to_text([int(i) for i in ids])


def autopsy_g(q, P, gold, cache, tm, codec, ring_dir):
    """Per-miss diagnosis for a shipped-config-G miss: decoded top-3, best-gold lexical +
    semantic rank + entry path, and a category (adds self-referential-bait for W2-style)."""
    docs = cache.manifest["docs"]
    order = P["G"]
    res = P["res"]
    words = [t["word"] for t in res.terms] if res is not None else []
    print(f"\n  MISS {q.id}: {q.question!r}")
    print("    top-3 returned:")
    for c in order[:3]:
        di = int(cache.chunk_doc[c]); loc = int(cache.chunk_local[c])
        mark = " <-GOLD" if c in gold else ""
        cc = float(res.chunk_cov[c]) if res is not None else 0.0
        sem = float(P["sem_chunk"][c]) if not P["all_stop"] else 0.0
        print(f"      {docs[di]['name']} c{loc} (g{c})  cov={cc:.1f} sem={sem:.2f}{mark}")
    if P["all_stop"] or res is None:
        print("    category: all-stop-fallback (no content term)")
        return "all-stop-fallback"
    best_g = max(gold, key=lambda g: float(res.chunk_cov[g]))
    g_lex = int((res.chunk_cov > res.chunk_cov[best_g]).sum()) + 1
    g_semrank, entry = _gold_semantic_rank(P, gold)
    dbg = int(cache.chunk_doc[best_g]); lbg = int(cache.chunk_local[best_g])
    print(f"    best gold: {docs[dbg]['name']} c{lbg} (g{best_g})  chunk_cov-rank={g_lex}  "
          f"sem-rank-in-cand={g_semrank}  entry={entry}")
    winner = int(order[0]) if order else -1
    cat = "other"
    if winner >= 0:
        wtext = _decode_chunk_text(ring_dir, cache, tm, codec, winner)
        clip = wtext.strip().replace("\n", " ")[:110]
        # self-referential-bait: the winner literally embeds the question text (W2, receipt S10)
        qcore = re.sub(r"[^a-z0-9 ]", "", q.question.lower()).strip()
        if qcore and qcore in re.sub(r"[^a-z0-9 ]", "", wtext.lower()):
            cat = "self-referential-bait"
        elif g_semrank is None or g_semrank > 3:
            cat = "semantic-weak"                         # gold not near the query in embed space
        elif entry == "inject":
            cat = "reranked-below"                        # injected but a competitor out-scored it
        else:
            cat = "lexical-competitor"
        print(f"    winner text: {clip!r}")
    print(f"    category: {cat}")
    return cat


# ---------------------------------------------------------------- scoreboard
def _pct(hits, n):
    return f"{(sum(hits) / n):.2f}" if n else "  - "


ROW_LABEL = {"A": "A v12-baseline", "C": "C cov-chunk *", "S": "S sem-only",
             "F": "F cov+sem", "G15": "G15 v1.5-cal", "G": "G full-stack **"}


def run_eval(cache, tm, T, codec, sidecar, qembed, questions, rows, topk, sem_weight,
             do_autopsy, ring_dir, set_name="A"):
    resolver = GoldResolver(cache, codec, sidecar, ring_dir)
    gold_map = {q.id: resolver.resolve(q) for q in questions}
    nat = [q for q in questions if q.kind == "natural"]
    kw = [q for q in questions if q.kind == "keyword"]

    acc = {r: {"n1": [], "n3": [], "k1": [], "k3": [], "a3": [], "lat": []} for r in rows}
    prims = {}
    for q in questions:
        P = compute_primitives(cache, tm, T, codec, sidecar, qembed, q, topk, sem_weight)
        prims[q.id] = P
        gold = gold_map[q.id]
        for r in rows:
            order = P[r]
            acc[r]["a3"].append(hit_at_k(order, gold, topk))
            acc[r]["lat"].append(P.get(r + "_lat", 0.0))
            if q.kind == "natural":
                acc[r]["n1"].append(hit_at_k(order, gold, 1)); acc[r]["n3"].append(hit_at_k(order, gold, topk))
            else:
                acc[r]["k1"].append(hit_at_k(order, gold, 1)); acc[r]["k3"].append(hit_at_k(order, gold, topk))
    qembed.save()

    print("\n" + "=" * 82)
    print(f"{'row':<17}{'nat@1':>7}{'nat@3':>7}{'kw@1':>7}{'kw@3':>7}{'all@3':>7}{'med_ms':>9}{'p90_ms':>9}")
    print("-" * 82)
    for r in rows:
        a = acc[r]
        lat = np.array(a["lat"]) if a["lat"] else np.array([0.0])
        print(f"{ROW_LABEL[r]:<17}{_pct(a['n1'], len(nat)):>7}{_pct(a['n3'], len(nat)):>7}"
              f"{_pct(a['k1'], len(kw)):>7}{_pct(a['k3'], len(kw)):>7}{_pct(a['a3'], len(questions)):>7}"
              f"{int(np.median(lat)):>9}{int(np.percentile(lat, 90)):>9}")
    print("-" * 82)
    print(f"natural n={len(nat)}  keyword n={len(kw)}   (nat@3 gate subset, kw@3 controls)")
    print("* C = shipped-lexical (must equal v1.3);  ** G = shipped default (semantic full stack)")

    # gate (5.3): row G naturals — 0.45 hard floor, expected 0.50-0.65, then the v1.3 ladder
    if "G" in rows and nat:
        h = sum(acc["G"]["n3"]); n = len(nat); frac = h / n
        print(f"\nGATE — row G naturals (shipped full stack): {h}/{n} = {frac:.2f}")
        print(f"  hard floor 0.45 : {'PASS' if frac >= 0.45 else 'FAIL'}   (expected band 0.50-0.65)")
        print(f"  floor 0.85      : {'met' if frac >= 0.85 else 'not met'}")
        print(f"  aspiration 0.95 : {'met' if frac >= 0.95 else 'not met'}")
        print(f"  reference 0.99  : {'met' if frac >= 0.99 else 'not met'}")
        if "G15" in rows:                                 # v1.6: shipped G vs the v1.5-cal continuity row
            g15n = sum(acc["G15"]["n3"])
            if set_name == "A":
                print(f"  vs G15 (v1.5 cal): G nat@3 {h}/{n} > {g15n}/{n} -> "
                      f"{'PASS' if h > g15n else 'FAIL'}   [strict target >9/20]")
            else:
                print(f"  vs G15 (v1.5 cal): G nat@3 {h}/{n} >= {g15n}/{n} -> "
                      f"{'PASS' if h >= g15n else 'FAIL'}   [not below baseline]")
        if frac < 0.45:
            print("  NOTE: 0.45 not reached; operator pre-authorized shipping the honest measured "
                  "maximum + full attribution/autopsy (plan 5.4/9). The eval set is FROZEN.")

    # controls + no-regression (5.3)
    if "G" in rows and "C" in rows:
        cn, gn = sum(acc["C"]["n3"]), sum(acc["G"]["n3"])
        ca, ga = sum(acc["C"]["a3"]), sum(acc["G"]["a3"])
        print(f"no-regression: G nat@3 {gn} >= C nat@3 {cn} -> {'OK' if gn >= cn else 'REGRESSION'}; "
              f"G all@3 {ga} >= C all@3 {ca} -> {'OK' if ga >= ca else 'REGRESSION'}")
    if "G" in rows and kw:
        gk = sum(acc["G"]["k3"]); nk = len(kw)
        rule = "PASS(>=4/5)" if gk >= 4 else ("HARD-MIN(3/5)" if gk >= 3 else "FAIL(<3/5)")
        print(f"controls: G kw@3 = {gk}/{nk} -> {rule}")
        if gk < 4:
            miss_k = [q for q in kw if not hit_at_k(prims[q.id]["G"], gold_map[q.id], topk)]
            print("  forced keyword-miss autopsy:")
            for q in miss_k:
                autopsy_g(q, prims[q.id], gold_map[q.id], cache, tm, codec, ring_dir)

    # attribution matrix (5.5) over the 16 named v1.3 misses + any new G-vs-C miss
    if "G" in rows and "C" in rows:
        present = {q.id: q for q in questions}
        ids = [i for i in V13_MISSES if i in present]
        new_miss = [q.id for q in questions
                    if hit_at_k(prims[q.id]["C"], gold_map[q.id], topk)
                    and not hit_at_k(prims[q.id]["G"], gold_map[q.id], topk)]
        for i in new_miss:
            if i not in ids:
                ids.append(i)
        print("\n" + "=" * 82)
        print(f"attribution matrix (hit@{topk}; 1=hit 0=miss) over {len(ids)} rows "
              f"[16 v1.3 misses + {len(new_miss)} new G-vs-C miss]")
        cols = [r for r in ("A", "C", "S", "F", "G") if r in rows]
        print(f"  {'id':<5}" + "".join(f"{c[0]:>3}" for c in cols) + "   recovered-by / note")
        for i in ids:
            q = present[i]; gold = gold_map[i]; P = prims[i]
            cells = "".join(f"{1 if hit_at_k(P[c], gold, topk) else 0:>3}" for c in cols)
            note = ""
            c_hit = hit_at_k(P["C"], gold, topk); g_hit = hit_at_k(P["G"], gold, topk)
            if not c_hit and g_hit:
                note = "recovered-by: " + recovered_by(cache, sidecar, P, gold, topk, sem_weight)
            elif c_hit and not g_hit:
                note = "NEW MISS (C hit, G miss) -> autopsy below"
            elif not g_hit:
                _r, entry = _gold_semantic_rank(P, gold)
                note = f"still miss (sem-rank={_r}, entry={entry})"
            print(f"  {i:<5}{cells}   {note}")

    # per-miss autopsy for row G (5.5)
    if do_autopsy and "G" in rows:
        g_miss = [q for q in questions if not hit_at_k(prims[q.id]["G"], gold_map[q.id], topk)]
        print("\n" + "=" * 82)
        print(f"per-miss autopsy (row G, shipped), {len(g_miss)} miss(es):")
        cats = collections.Counter()
        for q in g_miss:
            cats[autopsy_g(q, prims[q.id], gold_map[q.id], cache, tm, codec, ring_dir)] += 1
        print("\n  autopsy categories:", dict(cats))
    return acc, prims, gold_map


# ---------------------------------------------------------------- abstention report (v1.6, plan 3.3/3.6)
def abstention_report(set_results, unanswerable, prims_u, topk):
    """Print the abstention block (rule: max sem_win < SEM_ABSTAIN_TAU) + the per-set G+abst rows.

    set_results = [(set_name, questions, prims, gold_map), ...] for the answerable sets that ran;
    unanswerable = the Q list (empty for --set A); prims_u[id] holds each unanswerable's primitives
    (sem_top only, no gold). This is a pure REPORTING layer — retrieval rows are untouched."""
    tau = sm.SEM_ABSTAIN_TAU
    print("\n" + "=" * 82)
    print(f"== abstention (rule: top semantic window score < {tau:g}) ==")
    ans_sem, gabst_lines = [], []
    for set_name, questions, prims, gold_map in set_results:
        nat = [q for q in questions if q.kind == "natural"]
        kw = [q for q in questions if q.kind == "keyword"]
        def _abst(q):
            st = prims[q.id]["sem_top"]
            return st is not None and st < tau
        abst = [q for q in questions if _abst(q)]
        hits = [q for q in questions if hit_at_k(prims[q.id]["G"], gold_map[q.id], topk)]
        fa_hits = [q for q in hits if _abst(q)]
        gate = "   [hard gate <=1 on A hits]" if set_name == "A" else ""
        print(f"  set {set_name} answerable: abstained {len(abst)}/{len(questions)}   "
              f"false-abstains on hits: {len(fa_hits)}/{len(hits)}{gate}")
        for q in questions:
            if prims[q.id]["sem_top"] is not None:
                ans_sem.append((prims[q.id]["sem_top"], q.id))
        # G+abst: G hit AND not abstained (measured abstains on answerable are 0 -> equals G)
        def _gabst(rows):
            return sum(1 for q in rows
                       if hit_at_k(prims[q.id]["G"], gold_map[q.id], topk) and not _abst(q))
        gabst_lines.append(f"  G+abst set {set_name}: nat@3 {_gabst(nat)}/{len(nat)}  "
                           f"kw@3 {_gabst(kw)}/{len(kw)}   (G with abstained hits removed)")
    if unanswerable:
        verdicts = [(q.id, prims_u[q.id]["sem_top"],
                     prims_u[q.id]["sem_top"] is not None and prims_u[q.id]["sem_top"] < tau)
                    for q in unanswerable]
        tabst = sum(1 for _i, _s, a in verdicts if a)
        print(f"  unanswerable: true-abstain {tabst}/{len(unanswerable)}   [gate >=5]")
        for i in range(0, len(verdicts), 4):
            print("    " + " | ".join(
                f"{qid} {st:.3f} {'ABSTAIN' if a else 'ANSWERS-UNGROUNDED'}"
                for qid, st, a in verdicts[i:i + 4]))
        if ans_sem:
            an_s, an_id = min(ans_sem)
            line = f"  margins: nearest answerable sem_top {an_s:.3f} ({an_id}, {an_s - tau:+.3f})"
            esc = [(s, qid) for qid, s, a in verdicts if a]
            if esc:
                e_s, e_id = max(esc)
                line += f"; nearest escape {e_s:.3f} ({e_id}, {e_s - tau:+.3f})"
            print(line)
    for ln in gabst_lines:
        print(ln)


def ab_summary(set_results, topk):
    """A-tuned / B-validated summary (prints on --set AB): the calibration was chosen on set A
    only (controls-first); set B is reported as validation, never tuned on."""
    by = {sn: (qs, pr, gm) for sn, qs, pr, gm in set_results}
    if "A" not in by or "B" not in by:
        return

    def cnt(sn, rowkey, kind):
        qs, pr, gm = by[sn]
        return sum(hit_at_k(pr[q.id][rowkey], gm[q.id], topk) for q in qs if q.kind == kind)

    def tot(sn, kind):
        return sum(1 for q in by[sn][0] if q.kind == kind)

    gA, g15A, kA = cnt("A", "G", "natural"), cnt("A", "G15", "natural"), cnt("A", "G", "keyword")
    gB, g15B, kB = cnt("B", "G", "natural"), cnt("B", "G15", "natural"), cnt("B", "G", "keyword")
    nA, kwA, nB, kwB = tot("A", "natural"), tot("A", "keyword"), tot("B", "natural"), tot("B", "keyword")
    print("\n" + "=" * 82)
    print("== A-tuned / B-validated (calibration chosen on set A only, controls-first) ==")
    print(f"  calibration: floor={sm.SEM_PHANTOM_FLOOR} sem_weight={sm.SEM_WEIGHT_DEFAULT} "
          f"tie_a={sm.SEM_TIE_A} doc_cap={sm.SEM_DOC_CAP}")
    print(f"  set A: G nat@3 {gA}/{nA} (G15 {g15A}/{nA})  kw@3 {kA}/{kwA}   -> target >9/20: "
          f"{'PASS' if gA > g15A else 'FAIL'}")
    print(f"  set B: G nat@3 {gB}/{nB}  (G15 {g15B}/{nB})  kw@3 {kB}/{kwB}   -> not below baseline: "
          f"{'PASS' if gB >= g15B else 'FAIL'}")


# ---------------------------------------------------------------- calibration sweep (5.4)
def sweep_cal(cache, tm, T, codec, sidecar, qembed, topk, ring_dir):
    """32-combo in-memory calibration on precomputed channel primitives (no re-scan/re-embed):
    {none, doccap2} x SEM_PHANTOM_FLOOR {0, 0.25} x sem_weight {0.5, 0.75, 1.0, 1.5}
    x SEM_TIE_A {0, 0.25}. Reports EVERY combo; selection is controls-first (max kw@3, then max
    nat@3, ties -> minimal change vs the shipped point). SET A ONLY — sweeping on the held-out
    set B is forbidden (validation only)."""
    resolver = GoldResolver(cache, codec, sidecar, ring_dir)
    gold = {q.id: resolver.resolve(q) for q in ALL_Q}
    nat = [q for q in ALL_Q if q.kind == "natural"]
    kw = [q for q in ALL_Q if q.kind == "keyword"]

    prim = {}
    print(f"computing primitives for {len(ALL_Q)} questions (one scan/embed each) ...")
    t0 = time.perf_counter()
    for q in ALL_Q:
        P = compute_primitives(cache, tm, T, codec, sidecar, qembed, q, topk, sm.SEM_WEIGHT_DEFAULT)
        prim[q.id] = P
    qembed.save()
    print(f"primitives ready in {time.perf_counter()-t0:.1f}s; sweeping 32 combos (set A) in memory ...")

    dedups = ("none", "doccap2"); floors = (0.0, 0.25); sws = (0.5, 0.75, 1.0, 1.5); ties = (0.0, 0.25)
    print(f"\n{'dedup':>8}{'floor':>6}{'semW':>6}{'tieA':>6}{'nat@3':>8}{'kw@3':>7}  misses(nat)")
    results = []
    for dedup in dedups:
        cdoc = cache.chunk_doc if dedup == "doccap2" else None
        cap = sm.SEM_DOC_CAP if dedup == "doccap2" else None
        for floor in floors:
            for sw in sws:
                for tie in ties:
                    nh, kh, nm = 0, 0, []
                    for q in ALL_Q:
                        P = prim[q.id]
                        if P["all_stop"]:
                            order = P["A"]
                        else:
                            order, _s = sm.score_and_rank(P["cand"], P["creds"][floor], P["nbcov"],
                                                          P["sem_chunk"], P["Sw"], topk, tie, sw,
                                                          cdoc, cap)
                        h = hit_at_k(order, gold[q.id], topk)
                        if q.kind == "natural":
                            nh += h
                            if not h:
                                nm.append(q.id)
                        else:
                            kh += h
                    results.append((dedup, floor, sw, tie, nh, kh))
                    print(f"{dedup:>8}{floor:>6}{sw:>6}{tie:>6}{nh:>5}/{len(nat):<2}"
                          f"{kh:>5}/{len(kw):<1}  {nm}")

    # shipped point = (floor 0.25, sem_weight 0.75, tie_a 0.25, doccap2). Selection is
    # controls-first (max kw@3), then max nat@3, ties -> minimal change (prefer the shipped point).
    default = (0.25, 0.75, 0.25, "doccap2")               # (floor, sem_weight, tie_a, dedup)
    max_kw = max(r[5] for r in results)
    ctrl = [r for r in results if r[5] == max_kw]
    best_cn = max(r[4] for r in ctrl)
    tied = [r for r in ctrl if r[4] == best_cn]
    pick = next((r for r in tied if (r[1], r[2], r[3], r[0]) == default), tied[0])
    alts = [r for r in tied if (r[1], r[2], r[3], r[0]) != default]
    print(f"\nselection rule: controls-first (max kw@3={max_kw}/{len(kw)}), then max nat@3, "
          f"ties -> minimal change vs the shipped point")
    print(f"chosen: dedup={pick[0]} floor={pick[1]} sem_weight={pick[2]} tie_a={pick[3]} -> "
          f"nat@3 {pick[4]}/{len(nat)} kw@3 {pick[5]}/{len(kw)}")
    if alts:
        a = alts[0]
        print(f"  A-tied alternative (NOT shipped — preferring it would require citing set-B "
              f"numbers = validation-set tuning): dedup={a[0]} floor={a[1]} sem_weight={a[2]} "
              f"tie_a={a[3]} -> nat@3 {a[4]}/{len(nat)} kw@3 {a[5]}/{len(kw)}")
    print(f"shipped constants: floor={sm.SEM_PHANTOM_FLOOR} sem_weight={sm.SEM_WEIGHT_DEFAULT} "
          f"tie_a={sm.SEM_TIE_A} doc_cap={sm.SEM_DOC_CAP}")


# ---------------------------------------------------------------- main
def _subset(qlist, name):
    """Apply --subset within a set (same idiom for A and B). wiki = no README/CHANGELOG gold."""
    if name == "natural":
        return [q for q in qlist if q.kind == "natural"]
    if name == "keyword":
        return [q for q in qlist if q.kind == "keyword"]
    if name == "wiki":
        return [q for q in qlist if not any(d in ("README.md", "CHANGELOG.md") for d in q.docs)]
    return list(qlist)


def main():
    ap = argparse.ArgumentParser(description="Decoder Ring v1.6 semantic-attributed eval (+abstention).")
    ap.add_argument("--ring", default=HERE)
    ap.add_argument("--set", choices=["A", "B", "AB"], default="A", dest="eval_set",
                    help="frozen eval set (default A; B/AB add held-out set B + 7 unanswerable)")
    ap.add_argument("--config", choices=["A", "C", "S", "F", "G15", "G"], default=None,
                    help="run a single row (default: all six)")
    ap.add_argument("--subset", choices=["natural", "keyword", "all", "wiki"], default="all",
                    help="wiki = the questions with no README/CHANGELOG gold (within each set)")
    ap.add_argument("--sweep-cal", action="store_true", help="calibration sweep (32 combos, set A only)")
    ap.add_argument("--sem-weight", type=float, default=sm.SEM_WEIGHT_DEFAULT)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--ollama", default=None, help="Ollama endpoint override (embed cache miss only)")
    args = ap.parse_args()
    global OLLAMA
    if args.ollama:
        OLLAMA = args.ollama.rstrip("/")

    t_boot = time.perf_counter()
    print("loading ring + cache + semantic sidecar ...")
    _atoms, T = rc.load_atoms(os.path.join(args.ring, "atoms.dring"))
    tm = rc.load_tokens(os.path.join(args.ring, "tokens.dring"), verify=False)
    codec = rc.TokenCodec(os.path.join(args.ring, "tokenizer.json"))
    cache = rs.ensure_cache(args.ring, tm, T)
    try:
        sidecar = sm.load_sidecar(args.ring, cache)
    except sm.SemError as e:
        raise SystemExit(f"semantic sidecar error: {e}")
    if sidecar is None:
        raise SystemExit("semantic sidecar absent/stale. Build it first:\n"
                         "  python ring_cli.py sembuild")
    qembed = QueryEmbedder(args.ring, OLLAMA)
    print(f"loaded ring + cache ({int(cache.bag.shape[0])} chunks) + sidecar "
          f"({int(sidecar.keys.shape[0])} windows) in {time.perf_counter()-t_boot:.1f}s")

    if args.sweep_cal:                                    # set A only (validation set is never swept)
        sweep_cal(cache, tm, T, codec, sidecar, qembed, args.topk, args.ring)
        sidecar.close()
        return

    rows = [args.config] if args.config else ["A", "C", "S", "F", "G15", "G"]
    answerable = {"A": ALL_Q, "B": ALL_B}
    run_order = {"A": ["A"], "B": ["B"], "AB": ["A", "B"]}[args.eval_set]
    unanswerable = list(UNANSWERABLE_B) if args.eval_set in ("B", "AB") else []

    t0 = time.perf_counter()
    set_results, n_scored = [], 0
    for sn in run_order:
        qs = _subset(answerable[sn], args.subset)
        n_scored += len(qs)
        print(f"\n{'#' * 82}")
        print(f"# SET {sn} — {len(qs)} answerable questions (subset={args.subset})")
        print(f"{'#' * 82}")
        _acc, prims, gold_map = run_eval(cache, tm, T, codec, sidecar, qembed, qs, rows, args.topk,
                                         args.sem_weight, True, args.ring, sn)
        set_results.append((sn, qs, prims, gold_map))

    prims_u = {}
    if unanswerable:                                      # sem_top only; NEVER through GoldResolver
        print(f"\n{'#' * 82}")
        print(f"# UNANSWERABLE — {len(unanswerable)} questions (abstention only, no gold)")
        print(f"{'#' * 82}")
        for q in unanswerable:
            prims_u[q.id] = compute_primitives(cache, tm, T, codec, sidecar, qembed, q,
                                               args.topk, args.sem_weight)
            st = prims_u[q.id]["sem_top"]
            print(f"  {q.id:<4} sem_top {st:.3f} -> "
                  f"{'ABSTAIN' if st is not None and st < sm.SEM_ABSTAIN_TAU else 'answers-ungrounded'}")
        qembed.save()
        n_scored += len(unanswerable)

    abstention_report(set_results, unanswerable, prims_u, args.topk)
    if args.eval_set == "AB":
        ab_summary(set_results, args.topk)

    print(f"\ntotal eval wall time: {time.perf_counter()-t_boot:.1f}s "
          f"(set {args.eval_set}: scored {n_scored} questions x {len(rows)} rows in "
          f"{time.perf_counter()-t0:.1f}s)")
    sidecar.close()


if __name__ == "__main__":
    main()
