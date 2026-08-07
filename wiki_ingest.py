r"""wiki_ingest.py — Simple English Wikipedia retrieval stress corpus for the Decoder Ring.

PIPELINE.  download (Range-resumable) -> stream-parse the bz2 multistream dump with stdlib
xml.etree.iterparse (root.clear() per page, bounded RAM) -> strip wikitext to plain text
(staged stdlib regex + two tiny nesting scanners) -> pack into ~100k-token volume files ->
incrementally ingest those volumes into the SHARED store\ (same paged store as the demo
docs). Everything is bounded by default (--tokens 10_000_000 ~ the first ~15% of the dump)
and resumable (skip a volume iff its path AND sha256 already match the manifest).

The store ends MIXED (3 demo docs + wiki volumes). Scope queries with `ring_cli query --doc`;
`wiki_ingest.py --clean` removes exactly the wiki volumes again (by source-path prefix).

The wikitext stripper is deliberately APPROXIMATE (templates/tables/refs/media removed;
rare markup residue is harmless to a retrieval stress test) — see strip_wikitext / README.

Dump (verified live 2026-08-06): 384,058,867 bytes, ETag "6a72308c-16e445f3",
Last-Modified 2026-08-04, Accept-Ranges: bytes.

Run:  python wiki_ingest.py                 # download -> volumes (bounded) -> ingest (bounded)
      python wiki_ingest.py --volumes-only  # stop after building volumes
      python wiki_ingest.py --ingest-only   # skip download/build; ingest existing volumes
      python wiki_ingest.py --clean         # remove wiki volumes from the store (keep demo docs)
      python wiki_ingest.py --all           # full dump (prints a disk/time estimate; opt-in)
"""

import argparse
import bz2
import datetime
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import ring_core as rc
import ring_store as rs

# ---------------------------------------------------------------- constants
DUMP_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/"
    "simplewiki-latest-pages-articles-multistream.xml.bz2"
)
# Wikimedia blocks the generic "Python-urllib/x" UA with HTTP 403 and requires a descriptive
# one that identifies the tool (their bot policy). No auth, no cookies — just this header.
USER_AGENT = "decoder-ring-wiki-ingest/1.2 (local RAG research; contact beeradise@gmail.com)"
RING_DIR = Path(__file__).resolve().parent
WIKI_DIR = RING_DIR / "wiki"
DUMP_PATH = WIKI_DIR / "simplewiki-latest-pages-articles-multistream.xml.bz2"
PART = str(DUMP_PATH) + ".part"
META = WIKI_DIR / "dump.meta.json"
VOLUMES_DIR = WIKI_DIR / "volumes"
MARKER = VOLUMES_DIR / "_built.json"

VOLUME_TOKENS = 100_000
DEFAULT_BOUND = 10_000_000
MIN_ARTICLE_CHARS = 200
STORE_BYTES_PER_TOKEN = 273      # measured store growth per token (README)
EST_TOKS_PER_SEC = 16_600        # benched ingest rate (README)


def _reconfig_stdout():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ================================================================ wikitext stripper (4.4.3)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_SELF_RE = re.compile(r"<ref[^>]*?/>", re.DOTALL | re.IGNORECASE)
_REF_PAIR_RE = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_BLOCK_RE = re.compile(
    r"<(math|gallery|timeline|syntaxhighlight|source|score)\b[^>]*>.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
_MEDIA_RE = re.compile(
    r"\[\[\s*(?:File|Image|Category)\s*:[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]",
    re.IGNORECASE,
)
_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
_EXTLINK_LABELED_RE = re.compile(r"\[https?://[^\s\]]+\s+([^\]]*)\]")
_EXTLINK_BARE_RE = re.compile(r"\[https?://[^\]]*\]")
_TAG_RE = re.compile(r"</?[A-Za-z][^>]{0,200}>")
_EMPH_RE = re.compile(r"'{2,}")
_HEADING_RE = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$", re.MULTILINE)
_LIST_RE = re.compile(r"^[\*#:;]+\s*", re.MULTILINE)
_MAGIC_RE = re.compile(r"__[A-Z_]+__")
_TRAIL_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_MULTINL_RE = re.compile(r"\n{3,}")


def _strip_templates(text):
    """Remove {{...}} templates via a single-pass depth scanner (real nesting, no regex
    recursion). An unbalanced '{{' drops the remainder of the article (acceptable)."""
    out = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        two = text[i:i + 2]
        if two == "{{":
            depth += 1
            i += 2
        elif two == "}}" and depth > 0:
            depth -= 1
            i += 2
        else:
            if depth == 0:
                out.append(text[i])
            i += 1
    return "".join(out)


def _strip_tables(text):
    """Drop {|...|} tables with a line-wise depth counter (handles nested tables)."""
    out = []
    depth = 0
    for line in text.split("\n"):
        ls = line.lstrip()
        if ls.startswith("{|"):
            depth += 1
            continue
        if depth > 0:
            if ls.startswith("|}"):
                depth -= 1
            continue
        out.append(line)
    return "\n".join(out)


def strip_wikitext(text):
    """Wikitext -> plain text via the exact ordered rules in the plan (4.4.3). Imperfect
    residue is fine and stated in the README."""
    text = _COMMENT_RE.sub("", text)                                    # 1 comments
    text = _strip_templates(text)                                       # 2 templates
    text = _strip_tables(text)                                          # 3 tables
    text = _REF_SELF_RE.sub("", text)                                   # 4a self-closing refs
    text = _REF_PAIR_RE.sub("", text)                                   # 4b paired refs
    text = _BLOCK_RE.sub("", text)                                      # 4c paired content blocks
    text = _MEDIA_RE.sub("", text)                                      # 5 media/category links
    text = _WIKILINK_RE.sub(lambda m: m.group(1).split("|")[-1], text)  # 6 wiki links (last seg)
    text = _EXTLINK_LABELED_RE.sub(lambda m: m.group(1), text)          # 7a labeled ext links
    text = _EXTLINK_BARE_RE.sub("", text)                               # 7b bare ext links
    text = _TAG_RE.sub("", text)                                        # 8 remaining tags
    text = _EMPH_RE.sub("", text)                                       # 9 '' / ''' emphasis
    text = _HEADING_RE.sub(lambda m: m.group(1), text)                  # 10 headings -> plain lines
    text = _LIST_RE.sub("", text)                                       # 11 list/indent markers
    text = _MAGIC_RE.sub("", text)                                      # 12 magic words
    text = html.unescape(text)                                          # 13 entities LAST
    text = _TRAIL_WS_RE.sub("", text)                                   # 14a trailing spaces
    text = _MULTINL_RE.sub("\n\n", text)                                # 14b collapse blank runs
    return text.strip()                                                 # 14c


# ================================================================ dump streaming (4.4.2)
def _local(tag):
    return tag.rsplit("}", 1)[-1]


def iter_articles(dump_path):
    """Yield (title, plain_text) for every ns-0, non-redirect article whose stripped body is
    >= MIN_ARTICLE_CHARS. Streams the bz2 lazily; root.clear() per page keeps RAM bounded."""
    with bz2.open(dump_path, "rb") as f:
        ctx = ET.iterparse(f, events=("start", "end"))
        _, root = next(ctx)                       # export root; namespace lives on its tag
        ns = root.tag[1:root.tag.index("}")] if root.tag.startswith("{") else ""
        q = ("{" + ns + "}") if ns else ""
        for event, elem in ctx:
            if event != "end" or _local(elem.tag) != "page":
                continue
            try:
                if elem.findtext(q + "ns") != "0":
                    continue
                if elem.find(q + "redirect") is not None:
                    continue
                wikitext = elem.findtext(q + "revision/" + q + "text")
                if not wikitext:
                    continue
                if wikitext.lstrip()[:9].upper() == "#REDIRECT":
                    continue
                plain = strip_wikitext(wikitext)
                if len(plain) >= MIN_ARTICLE_CHARS:
                    yield (elem.findtext(q + "title") or ""), plain
            finally:
                root.clear()                      # REQUIRED — else the whole dump piles up in RAM


# ================================================================ meta / marker helpers
def _read_json(path):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _write_json_atomic(path, obj):
    rs.atomic_write_bytes(str(path), json.dumps(obj, indent=2).encode("utf-8"))


# ================================================================ download (4.4.1)
def download(force=False):
    """Ensure the dump is present and complete. Offline-friendly: a completed dump whose size
    matches META is used with no network. Range-resumes a matching .part. Returns dump meta."""
    os.makedirs(WIKI_DIR, exist_ok=True)
    meta = _read_json(META) or {}
    if (not force and os.path.exists(DUMP_PATH) and meta.get("completed")
            and os.path.getsize(DUMP_PATH) == meta.get("content_length")):
        print(f"download: dump present and complete ({os.path.getsize(DUMP_PATH):,} bytes) — skipping",
              flush=True)
        return meta

    head = urllib.request.Request(DUMP_URL, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(head, timeout=30) as r:
        content_length = int(r.headers["Content-Length"])
        etag = r.headers.get("ETag")
        last_modified = r.headers.get("Last-Modified")
    total_mib = content_length / 1048576

    # Persist META BEFORE streaming so a mid-download crash leaves a resumable marker: the next
    # run reads {etag, completed:false} and Range-resumes the .part. `meta` (read above) still
    # holds the OLD etag for the resume comparison just below; completed:false also defeats the
    # offline-skip check at the top on a re-run. The completed:true write happens on success.
    _write_json_atomic(META, {"url": DUMP_URL, "content_length": content_length,
                              "etag": etag, "last_modified": last_modified, "completed": False})

    resume_from, mode = 0, "wb"
    if not force and os.path.exists(PART) and meta.get("etag") == etag:
        resume_from, mode = os.path.getsize(PART), "ab"
    req = urllib.request.Request(DUMP_URL, headers={"User-Agent": USER_AGENT})
    if resume_from:
        req.add_header("Range", f"bytes={resume_from}-")

    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            if resume_from and getattr(r, "status", None) != 206:
                resume_from, mode = 0, "wb"          # server ignored Range -> restart clean
            got = resume_from
            next_mark = got + 16 * 1048576
            with open(PART, mode) as out:
                if mode == "wb" and resume_from == 0:
                    out.seek(0)
                while True:
                    chunk = r.read(1048576)
                    if not chunk:
                        break
                    out.write(chunk)
                    got += len(chunk)
                    if got >= next_mark:
                        print(f"download: {got / 1048576:.1f} / {total_mib:.1f} MiB "
                              f"({100 * got // content_length}%)", flush=True)
                        next_mark += 16 * 1048576
    except (urllib.error.URLError, OSError) as e:
        kept = os.path.getsize(PART) if os.path.exists(PART) else 0
        print(f"download: network error mid-stream ({e}); keeping .part ({kept:,} B) for resume",
              flush=True)
        sys.exit(1)

    size = os.path.getsize(PART)
    if size != content_length:
        os.remove(PART)
        print(f"download: completed but size {size:,} != expected {content_length:,}; removed .part",
              flush=True)
        sys.exit(1)
    os.replace(PART, DUMP_PATH)
    meta = {"url": DUMP_URL, "content_length": content_length, "etag": etag,
            "last_modified": last_modified, "completed": True}
    _write_json_atomic(META, meta)
    print(f"download: complete, {size:,} bytes -> {DUMP_PATH.name}", flush=True)
    return meta


# ================================================================ volume build (4.4.4)
def _marker_fresh(marker, dump_meta, bound, all_flag):
    if not marker:
        return False
    if marker.get("dump_content_length") != dump_meta.get("content_length"):
        return False
    if marker.get("dump_etag") != dump_meta.get("etag"):
        return False
    if marker.get("complete"):
        return True
    if all_flag:
        return marker.get("built_bound") == "all"
    bb = marker.get("built_bound")
    if bb == "all":
        return True
    try:
        return int(bb) >= int(bound)
    except (TypeError, ValueError):
        return False


def _stale_reason(marker, dump_meta, bound, all_flag):
    if not marker:
        return "no marker"
    if (marker.get("dump_content_length") != dump_meta.get("content_length")
            or marker.get("dump_etag") != dump_meta.get("etag")):
        return "dump changed (etag/size mismatch)"
    want = "all" if all_flag else format(bound, ",")
    return f"built_bound {marker.get('built_bound')} insufficient for requested {want}"


def _flush_volume(volumes_dir, idx, blocks):
    path = os.path.join(volumes_dir, f"vol-{idx:04d}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(blocks))
    return path


def build_volumes(dump_path, volumes_dir, marker_path, bound, *, all_flag, codec, dump_meta,
                  volume_tokens=VOLUME_TOKENS):
    """Parse the dump into ~volume_tokens-sized vol-NNNN.txt files (dump order, deterministic).
    Rebuilds from scratch when the existing marker is stale/insufficient. Returns the marker."""
    existing = _read_json(marker_path)
    if _marker_fresh(existing, dump_meta, bound, all_flag):
        print(f"volumes: fresh ({existing['n_volumes']} volumes · {existing['n_articles']:,} "
              f"articles · {existing['total_tokens'] / 1e6:.2f}M tokens · built_bound="
              f"{existing['built_bound']}) — skipping build", flush=True)
        return existing

    print(f"volumes stale/insufficient ({_stale_reason(existing, dump_meta, bound, all_flag)}) — "
          f"rebuilding {volumes_dir} from scratch", flush=True)
    os.makedirs(volumes_dir, exist_ok=True)
    # Remove the marker FIRST, then the stale volumes: a crash mid-delete then leaves NO marker
    # (the next run rebuilds) rather than a fresh marker over a half-deleted volume set.
    if os.path.exists(marker_path):
        os.remove(marker_path)
    for fn in os.listdir(volumes_dir):
        if fn.startswith("vol-") and fn.endswith(".txt"):
            os.remove(os.path.join(volumes_dir, fn))

    blocks, buf_tokens = [], 0
    vol_idx = n_articles = total_tokens = 0
    complete = True
    for title, plain in iter_articles(dump_path):
        n_tok = len(codec.encode_ids(plain))
        blocks.append(f"= {title} =\n\n{plain}\n\n")
        buf_tokens += n_tok
        n_articles += 1
        total_tokens += n_tok
        if buf_tokens >= volume_tokens:
            vol_idx += 1
            _flush_volume(volumes_dir, vol_idx, blocks)
            blocks, buf_tokens = [], 0
            if vol_idx % 10 == 0:
                print(f"volumes: {vol_idx} built · {n_articles:,} articles · "
                      f"{total_tokens / 1e6:.2f}M tokens", flush=True)
        if not all_flag and total_tokens >= bound:
            complete = False
            break
    if blocks:
        vol_idx += 1
        _flush_volume(volumes_dir, vol_idx, blocks)

    marker = {
        "dump_content_length": dump_meta.get("content_length"),
        "dump_etag": dump_meta.get("etag"),
        "built_bound": "all" if all_flag else bound,
        "complete": complete,
        "n_volumes": vol_idx, "n_articles": n_articles, "total_tokens": total_tokens,
        "volume_tokens": volume_tokens, "built": _now_iso(),
    }
    _write_json_atomic(marker_path, marker)
    print(f"volumes: built {vol_idx} volumes · {n_articles:,} articles · "
          f"{total_tokens / 1e6:.2f}M tokens (bound={'all' if all_flag else format(bound, ',')}, "
          f"complete={complete})", flush=True)
    return marker


# ================================================================ ingest (4.4.5)
def _fmt_eta(seconds):
    seconds = int(max(seconds, 0))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _volume_prefix(volumes_dir):
    return os.path.normcase(os.path.abspath(volumes_dir)) + os.sep


def ingest_volumes(ring_dir, volumes_dir, bound, *, all_flag, ring, K=None):
    """Incrementally ingest vol-*.txt into ring_dir's store. Skip a volume iff its normcased
    abspath AND sha256 already match the manifest; skips still count toward the token bound.
    `ring` = (atoms_bits, T_bits, token_matrix, codec) loaded once by the caller."""
    _, T_bits, token_matrix, codec = ring
    if K is None:
        K = rc.DEFAULT_FANOUT
    by_path = {d["source_path"]: d for d in rs.load_manifest(ring_dir)["docs"]}
    vols = sorted(p for p in os.listdir(volumes_dir)
                  if p.startswith("vol-") and p.endswith(".txt"))

    cumulative = added = skipped = 0
    t0 = time.time()
    for name in vols:
        vol_path = os.path.join(volumes_dir, name)
        np_path = os.path.normcase(os.path.abspath(vol_path))
        entry = by_path.get(np_path)
        if entry is not None:
            with open(vol_path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            if entry["source_sha256"] == sha:
                cumulative += int(entry["n_tokens"])
                skipped += 1
                print(f"skip {name} (already ingested, {int(entry['n_tokens']):,} tokens)", flush=True)
                continue
        if not all_flag and cumulative >= bound:
            break
        e = rs.add_document(ring_dir, path=vol_path, token_matrix=token_matrix, T_bits=T_bits,
                            codec=codec, K=K)
        cumulative += int(e["n_tokens"])
        added += 1
        elapsed = max(time.time() - t0, 1e-6)
        rate = cumulative / elapsed
        target = "all" if all_flag else f"{bound / 1e6:.1f}M"
        eta = "--:--" if all_flag else _fmt_eta((bound - cumulative) / max(rate, 1e-6))
        print(f"{name}  tokens={int(e['n_tokens']):,}  chunks={int(e['n_chunks']):,}  "
              f"total={cumulative / 1e6:.2f}M/{target}  {rate / 1000:.1f}k tok/s  ETA {eta}",
              flush=True)

    docs = rs.load_manifest(ring_dir)["docs"]
    prefix = _volume_prefix(volumes_dir)
    wiki = [d for d in docs if d["source_path"].startswith(prefix)]
    wiki_tokens = sum(int(d["n_tokens"]) for d in wiki)
    wiki_chunks = sum(int(d["n_chunks"]) for d in wiki)
    # only an add moves the manifest sha -> only then does the cache go stale
    suffix = " (cache rebuilds on next query)" if added else ""
    print(f"ingest: {len(wiki)} wiki volumes in store · {wiki_tokens:,} wiki tokens · "
          f"{wiki_chunks:,} wiki chunks · +{added} added / {skipped} skipped{suffix}", flush=True)
    return {"added": added, "skipped": skipped, "wiki_docs": len(wiki),
            "wiki_tokens": wiki_tokens, "wiki_chunks": wiki_chunks}


# ================================================================ clean (4.4.6)
def clean(ring_dir, volumes_dir, *, codec=None):
    """Remove wiki-volume docs from the store (demo docs untouched). Prints the count FIRST.
    The wiki\\ files on disk are the reproducible source and are NOT deleted."""
    docs = rs.load_manifest(ring_dir)["docs"]
    prefix = _volume_prefix(volumes_dir)
    targets = [d for d in docs if d["source_path"].startswith(prefix)]
    if not targets:
        print("clean: no wiki volumes in the store (demo docs untouched)", flush=True)
        return 0
    print(f"removing {len(targets)} wiki volumes from the store (demo docs untouched)", flush=True)
    for d in targets:
        rs.remove_document(ring_dir, d["doc_id"], codec=codec)
    print(f"clean: removed {len(targets)} wiki volume docs. The wiki\\ files on disk "
          f"(dump + volumes) are the reproducible source and were NOT deleted.", flush=True)
    return len(targets)


# ================================================================ CLI
def main():
    _reconfig_stdout()
    ap = argparse.ArgumentParser(description="Simple English Wikipedia corpus for the Decoder Ring.")
    ap.add_argument("--tokens", type=int, default=DEFAULT_BOUND,
                    help=f"ingest/build token bound (default {DEFAULT_BOUND:,})")
    ap.add_argument("--all", action="store_true",
                    help="full dump: build + ingest everything (prints a disk/time estimate first)")
    ap.add_argument("--volumes-only", action="store_true", help="stop after the volume build")
    ap.add_argument("--ingest-only", action="store_true",
                    help="skip download/build; ingest existing volumes only")
    ap.add_argument("--clean", action="store_true",
                    help="remove all wiki volumes from the store (keeps demo docs), then exit")
    ap.add_argument("--force-download", action="store_true",
                    help="re-download even if the dump looks complete")
    args = ap.parse_args()

    ring_dir = str(RING_DIR)
    tok_path = os.path.join(ring_dir, "tokenizer.json")

    if args.clean:
        clean(ring_dir, str(VOLUMES_DIR), codec=rc.TokenCodec(tok_path))
        return

    bound = args.tokens

    if not args.ingest_only:
        meta = download(force=args.force_download)
        if args.all:
            print("building ALL volumes first — full-dump parse, tens of minutes", flush=True)
        codec = rc.TokenCodec(tok_path)                  # only the codec is needed to size volumes
        build_volumes(str(DUMP_PATH), str(VOLUMES_DIR), str(MARKER), bound,
                      all_flag=args.all, codec=codec, dump_meta=meta)
    else:
        if not os.path.isdir(VOLUMES_DIR):
            sys.exit("--ingest-only: no wiki\\volumes to ingest (run a build first)")

    if args.volumes_only:
        print("volumes-only: stopping before ingest.", flush=True)
        return

    if args.all:
        marker = _read_json(MARKER) or {}
        tot = int(marker.get("total_tokens", 0))
        if tot:
            print(f"--all ingest: ~{tot:,} tokens, est ~{tot / EST_TOKS_PER_SEC / 60:.0f} min at "
                  f"{EST_TOKS_PER_SEC:,} tok/s, est ~{tot * STORE_BYTES_PER_TOKEN / 1024**3:.1f} GiB "
                  f"store growth at {STORE_BYTES_PER_TOKEN} B/token", flush=True)

    from ring_cli import load_ring
    t0 = time.time()
    ring = load_ring(ring_dir)
    print(f"ring loaded ({time.time() - t0:.1f}s) — ingesting volumes", flush=True)
    ingest_volumes(ring_dir, str(VOLUMES_DIR), bound, all_flag=args.all, ring=ring)


if __name__ == "__main__":
    main()
