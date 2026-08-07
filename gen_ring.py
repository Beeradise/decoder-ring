r"""gen_ring.py — regenerate the Decoder Ring artifacts from the frozen seed.

Writes atoms.dring (256 byte atoms + tie-break vector T) and tokens.dring (128,256 token
vectors) into --out. The ONE network touch is downloading the pinned tokenizer.json if it
is absent; its sha256 is ALWAYS verified against TOKENIZER_SHA256 (drift detection) and the
build hard-fails on mismatch. Everything else is pure seeded computation, so regenerating
on any machine yields byte-identical files (test_ring.py T5 proves it).

Run:  python gen_ring.py
"""

import argparse
import hashlib
import os
import sys
import time
import urllib.request

import numpy as np

import ring_core as rc


def _reconfig_stdout():
    # cp1252 consoles crash printing token strings ("Gcat" etc.) -> force UTF-8.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def ensure_tokenizer(path):
    """Download the pinned tokenizer.json if absent; ALWAYS verify its sha256."""
    if not os.path.exists(path):
        print(f"tokenizer.json absent -> downloading {rc.TOKENIZER_URL}")
        # stream to a .part file with a timeout (a stalled CDN cannot hang gen_ring forever),
        # then os.replace on success -> an interrupted download never leaves a partial
        # tokenizer.json that would sha-fail on the next run with no hint to delete it.
        part = path + ".part"
        with urllib.request.urlopen(rc.TOKENIZER_URL, timeout=60) as resp, open(part, "wb") as f:
            while True:
                buf = resp.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
        os.replace(part, path)
    data = open(path, "rb").read()
    digest = hashlib.sha256(data).hexdigest()
    print(f"tokenizer.json size={len(data)} sha256={digest}")
    if digest != rc.TOKENIZER_SHA256:
        raise SystemExit(f"FATAL: tokenizer.json sha256 mismatch (expected {rc.TOKENIZER_SHA256})")


def validate_vocab(id2bytes):
    """Assert the vocab invariants the composition relies on (2.2)."""
    assert len(id2bytes) == rc.SPECIAL_BASE, len(id2bytes)                 # 128000 regular IDs
    assert set(id2bytes) == set(range(rc.SPECIAL_BASE)), "regular IDs not 0..127999"
    single = {v[0] for v in id2bytes.values() if len(v) == 1}
    assert single == set(range(256)), "not all 256 byte values present as single-byte tokens"
    seqs = list(id2bytes.values())
    assert len(seqs) == len(set(seqs)), "byte sequences not unique"
    print(f"vocab OK: 128000 regular tokens, 256 single-byte, all byte-seqs unique")


def main():
    _reconfig_stdout()
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Regenerate the Decoder Ring from seed.")
    ap.add_argument("--out", default=here, help="output directory (default: script dir)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json path (default: <out>/tokenizer.json)")
    args = ap.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)
    tok_path = args.tokenizer or os.path.join(out, "tokenizer.json")
    atoms_path = os.path.join(out, "atoms.dring")
    tokens_path = os.path.join(out, "tokens.dring")

    t0 = time.time()
    ensure_tokenizer(tok_path)
    tok_sha = hashlib.sha256(open(tok_path, "rb").read()).digest()

    # ---- File 1: atoms + tie-break ----
    atom_words = np.stack([rc.drbg_vector(b"ATOM", i) for i in range(256)])   # (256,256) <u8
    T_words = rc.drbg_vector(b"TIE", 0)
    atoms_matrix = np.vstack([atom_words, T_words[None, :]])                   # (257,256)
    a_sha = rc.write_ring(atoms_path, rc.MAGIC_ATOMS, rc.N_SPECIALS + 1, atoms_matrix, b"\x00" * 32)
    a_size = os.path.getsize(atoms_path)
    print(f"atoms.dring written: {a_size} bytes  payload_sha256={a_sha.hex()}")
    assert a_size == 526464, a_size

    # atoms as bits for composition (+ tie-break bits)
    atoms_bits = np.stack([rc.unpack(atom_words[i]) for i in range(256)])
    T_bits = rc.unpack(T_words)

    # ---- File 2: token vectors, ID order ----
    id2bytes = rc.load_id2bytes(tok_path)
    validate_vocab(id2bytes)

    matrix = np.empty((rc.VOCAB, rc.WORDS), dtype="<u8")
    rot_cache = {}                       # (byte, position) -> rolled atom, memoised across vocab
    tc = time.time()
    for tid in range(rc.SPECIAL_BASE):   # 0..127999 regular tokens
        bits = rc.compose_token(id2bytes[tid], atoms_bits, T_bits, rot_cache)
        matrix[tid] = rc.pack(bits)
        if tid and tid % 20000 == 0:
            print(f"  composed {tid}/128000  ({time.time() - tc:.1f}s, cache={len(rot_cache)})")
    for k in range(rc.N_SPECIALS):       # 128000..128255 specials
        matrix[rc.SPECIAL_BASE + k] = rc.drbg_vector(b"SPEC", k)
    print(f"composed 128,256 token vectors in {time.time() - tc:.1f}s (rot cache size {len(rot_cache)})")

    # global uniqueness (blake2b-16 set) -> hard-fail on collision
    seen = set()
    for row in matrix:
        seen.add(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
    assert len(seen) == rc.VOCAB, f"vector collision: {rc.VOCAB - len(seen)} duplicate(s)"
    print("uniqueness OK: all 128,256 vectors distinct")

    t_sha = rc.write_ring(tokens_path, rc.MAGIC_TOKENS, rc.VOCAB, matrix, tok_sha)
    t_size = os.path.getsize(tokens_path)
    print(f"tokens.dring written: {t_size} bytes  payload_sha256={t_sha.hex()}")
    assert t_size == 262668416, t_size

    a_file = hashlib.sha256(open(atoms_path, "rb").read()).hexdigest()
    t_file = hashlib.sha256(open(tokens_path, "rb").read()).hexdigest()
    print(f"atoms.dring  whole-file sha256={a_file}")
    print(f"tokens.dring whole-file sha256={t_file}")
    print(f"elapsed {time.time() - t0:.1f}s")
    print("GEN OK")


if __name__ == "__main__":
    main()
