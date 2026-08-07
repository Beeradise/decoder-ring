r"""Decoder Ring v1 — universal BSC token->hypervector codebook for RAG in HDC space.

WHAT THIS IS. A deterministic, plug-and-play map from Llama-3-family token IDs to
16,384-bit binary spatter-code (BSC) hypervectors, plus the encode/decode/retrieve
machinery a RAG system needs on top of it. "Universal" is the load-bearing word: the
whole codebook is regenerable from one 32-bit seed through a library-agnostic PRNG
(SHA-256 in counter mode), so the frozen-seed contract survives any numpy/tokenizers
upgrade and can be re-implemented in any language. Two artifacts carry the contract:

  * atoms.dring  — 256 BYTE atoms (one per possible byte value) + 1 tie-break vector T,
    pure seeded-random, checksummed header. The irreducible alphabet.
  * tokens.dring — one vector per token ID (128,000 BPE + 256 specials = 128,256).
    Specials are pure random; regular tokens are COMPOSED from byte atoms so that
    surface-similar tokens land Hamming-near (typo/whitespace robustness) while the
    exact-hash decode stays lossless.

VSA MODEL: BSC (binary spatter codes). One representation, four ops:
  * bind  = XOR (self-inverse, distributes over bundling)
  * bundle= bitwise MAJORITY over a multiset, ties broken by a single global seeded
            vector T (identical to appending T as an extra input under strict majority)
  * order = bit ROTATION (circular shift)
  * sim   = HAMMING distance, reported as z = (D/2 - ham) / (sqrt(D)/2) so a random pair
            sits at z ~ N(0,1) and family/self at z >> 10 (matches the operator's z idiom)

D = 16384 bits = 256 x uint64 words (mirrors foreman DIMENSION_B = 16384).

TOKEN COMPOSITION (regular tokens, byte string B[0..L-1], atoms A[0..255]):
  * L == 1: token_vec = A[B[0]] EXACTLY. The 256 single-byte tokens' vectors ARE the byte
    atoms — the degenerate case falls out cleanly (positional plane at pos 0 and the bag
    plane are the same vector, bigram plane absent). Intentional and elegant.
  * L >= 2: three superposed planes, each collapsed by majority first, then combined by a
    W=1/1/1 majority across planes (mirrors foreman F9 BEAGLE per-plane-then-weighted):
      P_pos = maj{ rot(A[B[i]], i)              : i }   absolute byte position (stride 1)
      P_bag = maj{ A[B[i]]                       : i }   unordered multiset (dups count)
      P_big = maj{ rot(A[B[i]],1) XOR A[B[i+1]]  : i }   directed bigrams (F9 ORDER plane)
      token_vec = maj_w([P_pos, P_bag, P_big], [1,1,1], T)

WHY IT IS ROBUST (analytic; test_ring T2/T3 MEASURE it): " cat" vs "cat" share the whole
bag plane and 2/3 of bigrams while the positional plane decorrelates -> strong 2-of-3
correlation. "teh" vs "the" share the bag plane exactly -> per-bit agreement 0.625. Decode
distinctness is still guaranteed: clean decode is an exact hash map; noisy/bundled decode
leans on the measured margin vs bundle noise (T4).

RAG SIDE (chunk tree, bounded lossy compression):
  * A chunk of k<=K consecutive tokens gets a SEQ vector (order-faithful, rotation by
    position*257 so it decodes) and a BAG vector (position-invariant retrieval key).
  * A hierarchical tree bundles child SEQ/BAG vectors with fan-out K. v1 BUILDS and STORES
    the tree (the compression story) but the demo query touches level-0 BAG vectors only;
    upper levels surface in ingest/info stats. Said honestly in the README.

HONEST CAVEATS (v1, ratified out of scope): one GLOBAL tie-break T means every even-count
bundle shares a weak T component (per-bundle T rejected as scope creep); no stopword/DF
weighting in bag retrieval (function words dilute a query — fine at 3-doc demo scale);
brute-force Hamming NN (0.11 s/full-vocab scan on this box, no ANN index).

CONSTANTS PROVENANCE: D and the BSC model are operator-RATIFIED. GENESIS_SEED, PRNG_ID and
the header layout ARE the frozen contract (never change without a version bump).
DEFAULT_FANOUT=32 has an analytic basis (member-recovery z ~ sqrt(2D/(pi*K)) ~ 18 at K=32)
and is MEASURED by test_ring T4; SEQ_ROT_STRIDE=257 is a distinct prime rotation family so
sequence positions never alias the byte-level rot-by-i / rot-by-1 constructions.

No frameworks, no git idiom: verification = running gen_ring.py / test_ring.py / ring_cli.py.
"""

import hashlib
import json
import struct
import sys

import numpy as np

from tokenizers import Tokenizer

# ---------------------------------------------------------------- geometry / model
D = 16384                      # hypervector width in bits (ratified; = foreman DIMENSION_B)
WORDS = 256                    # D // 64 packed uint64 words per vector
assert D == WORDS * 64

# ---------------------------------------------------------------- frozen-seed contract
# These four values ARE the universal contract. Change any of them and every downstream
# vector changes -> bump VERSION and the magics. SHA256-CTR is deliberately not numpy PCG64:
# numpy's Generator method streams are explicitly NOT guaranteed stable across versions, and
# a "universal" ring must be reimplementable in any language from the spec alone.
GENESIS_SEED = 0xDEC0DE01      # uint64 root seed
PRNG_ID = b"SHA256-CTR-V1"     # id stamped in the header (NUL-padded to 16 bytes)
VERSION = 1

# ---------------------------------------------------------------- vocabulary geometry
VOCAB = 128256                 # total token IDs (128000 BPE + 256 specials)
N_SPECIALS = 256               # special tokens ...
SPECIAL_BASE = 128000          # ... occupying contiguous IDs 128000..128255
assert VOCAB == SPECIAL_BASE + N_SPECIALS

# ---------------------------------------------------------------- RAG / chunk-tree knobs
DEFAULT_FANOUT = 16            # K: chunk fan-out. The analytic member-recovery z ~ 18 at K=32
                               # assumed INDEPENDENT vectors; the ratified global-T tie-break
                               # correlates composed tokens (see maj/T notes), lowering real
                               # capacity. test_ring T4 MEASURED on this box: K=16 -> 100%,
                               # K=32 -> 99.7% (~1 miss/320). So the default is the plan's
                               # ratified fallback K=16 (plan 3.5/assumption 3), which decodes
                               # losslessly. This is a RAG-side knob only: it does not touch
                               # the atom/token vectors, so .dring files are unchanged.
SEQ_ROT_STRIDE = 257           # prime, odd -> coprime to 2^14. Sequence position p binds via
                               # rot(v, (p*257) mod D); a distinct stride family so sequence
                               # positions never alias byte-level rot-by-i / rot-by-1.
NN_BLOCK = 16384               # rows per block in the brute-force Hamming NN scan

# ---------------------------------------------------------------- tokenizer source (pinned)
# unsloth mirror, gated:false, commit-pinned; verified 302 -> public CDN, no auth. The
# generator MUST verify TOKENIZER_SHA256 and refuse on mismatch (drift detection).
TOKENIZER_URL = (
    "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/"
    "5a8abab4a5d6f164389b1079fb721cfab8d7126c/tokenizer.json"
)
TOKENIZER_SHA256 = "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b"

# ---------------------------------------------------------------- file format
MAGIC_ATOMS = b"DRINGATM"      # 8 bytes exactly
MAGIC_TOKENS = b"DRINGTOK"
HEADER_FMT = "<8sIII16sQI32s32s16x"   # magic, version, D, words, prng_id, seed,
                                      # n_vectors, source_hash, payload_sha256, 16x reserved
HEADER_SIZE = 128
assert struct.calcsize(HEADER_FMT) == HEADER_SIZE
VEC_BYTES = WORDS * 8          # 2048 bytes on disk per vector


# ================================================================ frozen PRNG (3.1)
def drbg_vector(label, index):
    """One raw D-bit vector as (256,) little-endian uint64 words.

    SHA256-CTR-V1: 64 SHA-256 blocks (64 x 32 B = 2048 B = 16384 bits), block c hashing
    b"DRINGv1|" + label + b"|" + struct.pack("<QQQ", seed, index, c). Library-agnostic and
    reproducible from the spec alone. `label` is bytes (b"ATOM" / b"TIE" / b"SPEC");
    `index` is a uint64 counter within the stream.
    """
    raw = b"".join(
        hashlib.sha256(b"DRINGv1|" + label + b"|" + struct.pack("<QQQ", GENESIS_SEED, index, c)).digest()
        for c in range(64)
    )
    return np.frombuffer(raw, dtype="<u8").copy()


# ================================================================ bit ops (3.2)
# Working forms: unpacked uint8[16384] 0/1 for construction; packed uint64[256] for files,
# storage and Hamming. Packing layout (part of the contract): bit b -> byte b//8, bit b%8,
# LSB-first; the 2048 bytes viewed as 256 LE uint64.

def pack(bits):
    """Unpacked uint8[16384] 0/1 -> packed (256,) <u8 words (LSB-first)."""
    return np.packbits(bits, bitorder="little").view("<u8")


def unpack(words):
    """Packed (256,) <u8 words -> unpacked uint8[16384] 0/1 (LSB-first)."""
    words = np.ascontiguousarray(words, dtype="<u8")
    return np.unpackbits(words.view(np.uint8), bitorder="little")


def rot(bits, r):
    """Circular shift on the UNPACKED array: bit b -> bit (b + r) mod D (np.roll)."""
    return np.roll(bits, r)


def maj(vecs, T):
    """Bitwise majority over a multiset of unpacked vectors; ties (even n) -> T's bit.

    Identical to appending T as an (n+1)-th input under strict majority. `vecs` is a list
    of uint8[16384] 0/1 arrays (duplicates count); returns uint8[16384].
    """
    n = len(vecs)
    if n == 1:
        return vecs[0].copy()
    acc = np.zeros(D, dtype=np.int32)
    for v in vecs:
        acc += v
    out = (2 * acc > n).astype(np.uint8)
    if n % 2 == 0:
        tie = 2 * acc == n
        out[tie] = T[tie]
    return out


def maj_w(planes, weights, T):
    """Weighted majority across whole planes; ties (even total weight) -> T's bit.

    per-bit s = sum(w_i * bit_i), W = sum(w_i); 1 if 2s>W, 0 if 2s<W, T if 2s==W. With
    W=1/1/1 over 3 present planes W=3 is odd -> no ties ever fire.
    """
    W = int(sum(weights))
    acc = np.zeros(D, dtype=np.int32)
    for w, p in zip(weights, planes):
        acc += int(w) * p.astype(np.int32)
    out = (2 * acc > W).astype(np.uint8)
    if W % 2 == 0:
        tie = 2 * acc == W
        out[tie] = T[tie]
    return out


def hamming(a_words, b_words):
    """Hamming distance between two packed (256,) <u8 vectors."""
    return int(np.bitwise_count(a_words ^ b_words).sum())


def z_of(ham):
    """Similarity statistic: random baseline ham ~ N(D/2=8192, sqrt(D)/2=64)."""
    return (D / 2 - ham) / (np.sqrt(D) / 2)


# ================================================================ byte map (2.2)
def bytes_to_unicode():
    """Canonical GPT-2 byte-level alphabet: byte value -> printable unicode char."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def load_id2bytes(tokenizer_json_path):
    """dict id->bytes for the 128,000 REGULAR tokens (specials have no byte spelling).

    Parses model.vocab directly and inverts the GPT-2 byte-level alphabet char-by-char.
    """
    with open(tokenizer_json_path, encoding="utf-8") as f:
        vocab = json.load(f)["model"]["vocab"]
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    return {tid: bytes(u2b[ch] for ch in tokstr) for tokstr, tid in vocab.items()}


# ================================================================ token composition (3.3)
def _roll_cached(atoms_bits, b, r, cache):
    """rot(atom_b, r) with an optional (byte, r) memo. Cached arrays are never mutated."""
    if cache is None:
        return np.roll(atoms_bits[b], r)
    key = (b, r)
    v = cache.get(key)
    if v is None:
        v = np.roll(atoms_bits[b], r)
        cache[key] = v
    return v


def compose_token(byte_string, atoms_bits, T_bits, rot_cache=None):
    """Compose one regular-token vector from its byte string. Returns uint8[16384].

    Accepts ARBITRARY byte strings, not just vocab tokens (T3 composes the typo b"teh"
    directly). `rot_cache` is an optional dict memoising rot(atom, r) across a generation
    run (a per-(byte,position) memo -> few thousand distinct rolls over the whole vocab).
    """
    L = len(byte_string)
    if L == 0:
        raise ValueError("compose_token: empty byte string")
    if L == 1:
        return atoms_bits[byte_string[0]].copy()
    p_pos = maj([_roll_cached(atoms_bits, byte_string[i], i, rot_cache) for i in range(L)], T_bits)
    p_bag = maj([atoms_bits[b] for b in byte_string], T_bits)
    p_big = maj(
        [_roll_cached(atoms_bits, byte_string[i], 1, rot_cache) ^ atoms_bits[byte_string[i + 1]] for i in range(L - 1)],
        T_bits,
    )
    return maj_w([p_pos, p_bag, p_big], [1, 1, 1], T_bits)


# ================================================================ file I/O (3.4)
def write_ring(path, magic, n_vectors, words_matrix, source_hash):
    """Write a .dring file: 128-byte header + n_vectors x 2048-byte payload. Returns payload sha256."""
    assert sys.byteorder == "little"
    words_matrix = np.ascontiguousarray(words_matrix, dtype="<u8")
    assert words_matrix.shape == (n_vectors, WORDS), words_matrix.shape
    assert len(source_hash) == 32
    payload = words_matrix.tobytes()
    payload_sha = hashlib.sha256(payload).digest()
    header = struct.pack(
        HEADER_FMT, magic, VERSION, D, WORDS, PRNG_ID, GENESIS_SEED, n_vectors, source_hash, payload_sha
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)
    return payload_sha


def read_header(path):
    """Parse and return the 128-byte header of a .dring file as a dict."""
    with open(path, "rb") as f:
        hdr = f.read(HEADER_SIZE)
    magic, version, d, words, prng, seed, n, src, paysha = struct.unpack(HEADER_FMT, hdr)
    return {
        "magic": magic,
        "version": version,
        "D": d,
        "words_per_vec": words,
        "prng_id": prng.rstrip(b"\x00"),
        "seed": seed,
        "n_vectors": n,
        "source_hash": src,
        "payload_sha256": paysha,
    }


def _load_ring(path, expect_magic, expect_n, verify):
    """Shared loader: validate the header, optionally verify the payload sha, return (header, words)."""
    assert sys.byteorder == "little", "little-endian host required (header declares LE layout)"
    h = read_header(path)
    assert h["magic"] == expect_magic, h["magic"]
    assert h["version"] == VERSION, h["version"]
    assert h["D"] == D and h["words_per_vec"] == WORDS
    assert h["prng_id"] == PRNG_ID, h["prng_id"]
    assert h["seed"] == GENESIS_SEED, h["seed"]
    assert h["n_vectors"] == expect_n, h["n_vectors"]
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE)
        payload = f.read()
    assert len(payload) == expect_n * VEC_BYTES, len(payload)
    if verify:
        assert hashlib.sha256(payload).digest() == h["payload_sha256"], "payload sha256 mismatch"
    words = np.frombuffer(payload, dtype="<u8").reshape(expect_n, WORDS)
    return h, words


def load_atoms(path):
    """Load atoms.dring -> (atoms_bits (256,16384) uint8, T_bits (16384,) uint8). Always verifies."""
    _, words = _load_ring(path, MAGIC_ATOMS, N_SPECIALS + 1, verify=True)
    atoms_bits = np.stack([unpack(words[i]) for i in range(N_SPECIALS)])
    T_bits = unpack(words[N_SPECIALS])
    return atoms_bits, T_bits


def load_tokens(path, verify=True):
    """Load tokens.dring -> (128256, 256) <u8 token matrix. Payload verify default on."""
    _, words = _load_ring(path, MAGIC_TOKENS, VOCAB, verify=verify)
    return np.ascontiguousarray(words)


# ================================================================ tokenizer wrapper (3.6)
class TokenCodec:
    """Thin wrapper over tokenizers.Tokenizer that ALWAYS suppresses the BOS post-processor."""

    def __init__(self, tokenizer_json_path):
        self.tok = Tokenizer.from_file(tokenizer_json_path)
        self.id2bytes = load_id2bytes(tokenizer_json_path)
        with open(tokenizer_json_path, encoding="utf-8") as f:
            for a in json.load(f)["added_tokens"]:
                # specials have no byte spelling; keep their literal content for display only
                self.id2bytes[a["id"]] = a["content"].encode("utf-8")

    def encode_ids(self, text):
        """text -> token IDs, add_special_tokens=False (default encode injects BOS 128000)."""
        return self.tok.encode(text, add_special_tokens=False).ids

    def ids_to_bytes(self, ids):
        """Per-token byte spelling (specials -> their literal content bytes)."""
        return [self.id2bytes[i] for i in ids]

    def ids_to_text(self, ids):
        """Detokenize IDs back to text (utf-8, replacement on malformed byte fragments)."""
        return self.tok.decode(list(ids), skip_special_tokens=False)


# ================================================================ decode paths (3.6)
class CleanDecoder:
    """O(1) lossless decode: blake2b-16(vec bytes) -> id. Never silently NN-falls-back."""

    def __init__(self, token_matrix):
        self.map = {}
        for tid in range(token_matrix.shape[0]):
            key = hashlib.blake2b(token_matrix[tid].tobytes(), digest_size=16).digest()
            self.map[key] = tid
        # a collision would shrink the map -> doubles as a global uniqueness check
        assert len(self.map) == token_matrix.shape[0], "token vectors not unique"

    def decode(self, words):
        """Exact packed vector -> id, or None if not a clean single-token vector."""
        key = hashlib.blake2b(np.ascontiguousarray(words, dtype="<u8").tobytes(), digest_size=16).digest()
        return self.map.get(key)


def nn_decode(words, token_matrix, block=NN_BLOCK):
    """Brute-force Hamming nearest neighbour over the token matrix -> (id, ham, z)."""
    q = np.ascontiguousarray(words, dtype="<u8")
    best_id, best_ham = -1, 1 << 62
    n = token_matrix.shape[0]
    for start in range(0, n, block):
        blk = token_matrix[start:start + block]
        hams = np.bitwise_count(blk ^ q).sum(axis=1, dtype=np.int64)
        j = int(np.argmin(hams))
        h = int(hams[j])
        if h < best_ham:
            best_ham, best_id = h, start + j
    return best_id, best_ham, float(z_of(best_ham))


# ================================================================ chunk tree (3.5)
def _bundle_seq(bit_vecs, T_bits):
    """Order-faithful bundle: rotate member p by (p*257) mod D, then majority. -> packed words."""
    inputs = [rot(bit_vecs[p], (p * SEQ_ROT_STRIDE) % D) for p in range(len(bit_vecs))]
    return pack(maj(inputs, T_bits))


def _bundle_bag(bit_vecs, T_bits):
    """Position-invariant bundle: majority of members unrotated. -> packed words."""
    return pack(maj(bit_vecs, T_bits))


def build_chunks(ids, token_matrix, T_bits, K=DEFAULT_FANOUT):
    """Level-0 chunks of <=K consecutive tokens. Returns (seq (n,256), bag (n,256), lens (n,))."""
    seqs, bags, lens = [], [], []
    for start in range(0, len(ids), K):
        chunk = ids[start:start + K]
        tvs = [unpack(token_matrix[i]) for i in chunk]
        seqs.append(_bundle_seq(tvs, T_bits))
        bags.append(_bundle_bag(tvs, T_bits))
        lens.append(len(chunk))
    seq = np.array(seqs, dtype="<u8") if seqs else np.empty((0, WORDS), dtype="<u8")
    bag = np.array(bags, dtype="<u8") if bags else np.empty((0, WORDS), dtype="<u8")
    return seq, bag, np.array(lens, dtype=np.int32)


def build_tree(level0_seq, level0_bag, T_bits, K=DEFAULT_FANOUT):
    """Bundle child SEQ/BAG vectors with fan-out K until <=1 root. Returns a list of
    (seq, bag, lens) upper levels (empty if the input already fits in one chunk)."""
    levels = []
    cur_seq, cur_bag = level0_seq, level0_bag
    while cur_seq.shape[0] > 1:
        nseq, nbag, nlen = [], [], []
        for start in range(0, cur_seq.shape[0], K):
            gs = cur_seq[start:start + K]
            gb = cur_bag[start:start + K]
            k = gs.shape[0]
            nseq.append(_bundle_seq([unpack(gs[p]) for p in range(k)], T_bits))
            nbag.append(_bundle_bag([unpack(gb[p]) for p in range(k)], T_bits))
            nlen.append(k)
        cur_seq = np.array(nseq, dtype="<u8")
        cur_bag = np.array(nbag, dtype="<u8")
        levels.append((cur_seq, cur_bag, np.array(nlen, dtype=np.int32)))
    return levels


def decode_chunk(seq_words, k, token_matrix):
    """Recover the k token IDs from a SEQ bundle: per-position unrotate -> Hamming NN."""
    bits = unpack(seq_words)
    ids = []
    for p in range(k):
        unrot = rot(bits, -((p * SEQ_ROT_STRIDE) % D))
        tid, _, _ = nn_decode(pack(unrot), token_matrix)
        ids.append(tid)
    return ids
