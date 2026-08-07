r"""gui.py — Decoder Ring dark-theme chat GUI (query + ingest front-end).

WHAT THIS IS. A one-window Tk chat over the paged store. Plain typed text is a `query`
(streamed live as each hit decodes); file drops or `/ingest <paths>` run `ingest --add` AND
THEN auto-build the semantic index in the same background worker (progress streamed); `/rebuild
<paths>` (confirmed) wipes and re-ingests (also auto-builds); `/remove`, `/stats`, `/topk`,
`/df`, `/help` cover the rest. The GUI owns the Ring Ollama island on 127.0.0.1:11436 (it health-
checks /api/tags and cold-starts `Start Ring Island.bat` minimized only when the default
island is down — never for a RING_OLLAMA override).

NAMED COLLECTIONS (v1.8). `/collection <name>` switches the active collection (confirm-to-create
on a new name); `/collections` lists them; `/sembuild` rebuilds the active index. Every query/
ingest/remove/stats routes through the active collection (default = the legacy store\). A drop or
/ingest = ingest + AUTO incremental sembuild; if the island is down the ingest still lands and the
build is deferred LOUDLY (status shows `sem: STALE (build pending)`) and auto-retried on the next
query/sembuild once the island returns. Cost (measured 2026-08-07, ~80 texts/s): ~20 s per 100k
tokens for a small collection; a drop into the big DEFAULT collection re-keys ~157k windows =
~2-3 min (streamed; never freezes). Freshness in the status line is computed with plain
json+hashlib — the GUI still NEVER imports ring modules.

PROCESS MODEL. Every command spawns `python -u ring_cli.py ...` as a subprocess and streams
its stdout into the chat. `-u` is load-bearing (block-buffered children look dead for
minutes). The ring load is ~1.5 s warm (tokenizers-codec dominated, not the 250 MB matrix)
= ~10-15% of a query turn; a resident ring was rejected (would fork shipped ring_cli, lose
crash isolation, and fight the Windows mmap-vs-rebuild discipline). The GUI process itself
stays light: it reads store\manifest.json as plain JSON and never imports ring_core/_store.

Forked from foreman's gui.py — behavior COPIED, never imported or called. decoder-ring has
zero runtime dependency on the foreman tree.

LAUNCH.  Ring GUI.bat            (double-click; starts pythonw gui.py, no console)
         pythonw gui.py          (same, from a shell)
         python gui.py           (with a console, handy for watching gui.log)
"""

import faulthandler
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    print("decoder-ring GUI needs tkinterdnd2 (pip install tkinterdnd2).", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------- palette (from foreman)
DARK_BG = "#150b29"
DARK_BG_INPUT = "#1e1140"
DARK_BG_UI = "#241448"
DARK_BG_ACTIVE = "#32205f"
DARK_FG = "#ffffff"
DARK_FG_DIM = "#d8d2e8"
LINK_BLUE = "#6c9eff"
FILE_GREEN = "#7cd992"
HINT_GRAY = "#8c82a8"
SELECT_BG = "#3a2a6a"

# ---------------------------------------------------------------- paths / endpoints
RING_DIR = Path(__file__).resolve().parent
RING_CLI = RING_DIR / "ring_cli.py"
ISLAND_BAT = RING_DIR / "Start Ring Island.bat"
GUI_LOG = RING_DIR / "gui.log"
FAULT_LOG = RING_DIR / "gui-fault.log"

OLLAMA = os.environ.get("RING_OLLAMA", "http://127.0.0.1:11436").rstrip("/")
DEFAULT_ISLAND = "http://127.0.0.1:11436"
DEMO_MODEL = "llama3.2:1b"
INGEST_EXTENSIONS = (".txt", ".md", ".pdf")

CHAT_URL_RE = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
_URL_TRAIL = ".,;:!?)]}\"'>"

# Mirror ring_store.sanitize_collection locally — the GUI NEVER imports ring modules (it must
# stay a light plain-JSON front-end). Same rule: casefold, 1-32 chars a-z 0-9 _ - starting alnum.
_COLLECTION_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def _sanitize_collection(name):
    """Normalized name / 'default' / None (invalid), mirroring ring_store.sanitize_collection."""
    n = name.strip().casefold()
    if n == "default":
        return "default"
    return n if _COLLECTION_RE.match(n) else None

# pythonw has no console; land crash tracebacks in a file rather than /dev/null.
try:
    _FAULT_FH = open(FAULT_LOG, "a", encoding="utf-8")
    faulthandler.enable(file=_FAULT_FH)
except OSError:
    pass


# ---------------------------------------------------------------- module helpers (unit-testable)
def gui_log(msg):
    """Append one timestamped line to gui.log. Best-effort — never raises (pythonw silence)."""
    try:
        with open(GUI_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
    except OSError:
        pass


def _url_segments(text):
    """Split text into (segment, is_url) pairs so append_to_chat can link only the URLs.

    Trailing sentence punctuation is peeled back off a URL into the following plain segment
    (so "see https://x.org." does not linkify the dot). No URL -> a single passthrough pair.
    """
    out = []
    pos = 0
    for m in CHAT_URL_RE.finditer(text):
        start = m.start()
        url = m.group(0)
        trail = ""
        while url and url[-1] in _URL_TRAIL:
            trail = url[-1] + trail
            url = url[:-1]
        if start > pos:
            out.append((text[pos:start], False))
        if url:
            out.append((url, True))
        if trail:
            out.append((trail, False))
        pos = m.end()          # trail chars are part of the match; emitted as their own segment
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def _parse_drop_paths(data):
    r"""tkdnd <<Drop>> event.data -> list of paths, WITHOUT Tcl splitlist.

    splitlist backslash-unescapes bare Windows paths (C:\y\c.py -> C:yc.py), so we honor the
    brace grouping (paths with spaces) manually and split bare tokens on ASCII whitespace
    only. Backslashes always survive verbatim. An unbalanced "{" takes the rest of the string.
    """
    paths = []
    i, n = 0, len(data)
    while i < n:
        while i < n and data[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        if data[i] == "{":
            j = data.find("}", i + 1)
            if j == -1:
                paths.append(data[i + 1:])
                break
            paths.append(data[i + 1:j])
            i = j + 1
        else:
            j = i
            while j < n and data[j] not in " \t\r\n":
                j += 1
            paths.append(data[i:j])
            i = j
    return paths


def _split_args(s):
    """Split a command tail like a shell (posix=False keeps Windows backslashes), then strip
    matched surrounding quotes off each token. Returns None on unbalanced quotes so the caller
    can print a usage line."""
    try:
        toks = shlex.split(s, posix=False)
    except ValueError:
        return None
    out = []
    for t in toks:
        if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
            t = t[1:-1]
        out.append(t)
    return out


def _partition_ingestible(paths):
    """(accepted, rejected) by extension, case-insensitively — .txt/.md/.pdf accepted."""
    accepted, rejected = [], []
    for p in paths:
        ext = os.path.splitext(str(p).lower())[1]
        (accepted if ext in INGEST_EXTENSIONS else rejected).append(p)
    return accepted, rejected


def _api_tags(url, timeout=2):
    """Installed model names at an Ollama endpoint via HTTP GET /api/tags, or None if the
    endpoint is unreachable. NEVER shells out (mirrors ring_cli.ollama_models)."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as r:
            data = json.load(r)
        return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None


# ---------------------------------------------------------------- the app
class RingGuiApp:
    def __init__(self, root, start_island_check=True):
        self.root = root
        root.title("Decoder Ring — query & ingest")
        root.geometry("800x600")

        # state
        self._busy = False
        self.topk = 2
        self.df_max = 0.25
        self._island_state = "checking"      # checking | up | model-missing | down | remote
        self._confirm = None                 # test seam: a stub(title, msg) overrides askyesno
        self._link_seq = 0
        self._first_msg_done = False
        self.collection = None               # None = the DEFAULT (legacy store\) collection
        self._pending_sembuild = set()       # display-names with a deferred (island-down) build
        self._store_manifest_path = str(RING_DIR / "store" / "manifest.json")

        style = ttk.Style(root)
        style.theme_use("clam")              # the only built-in ttk theme that honors bg overrides
        style.configure("Dark.TFrame", background=DARK_BG)
        style.configure("Status.TLabel", background=DARK_BG, foreground=HINT_GRAY, padding=(8, 3))
        style.configure("Dark.Vertical.TScrollbar", background=DARK_BG_UI, troughcolor=DARK_BG,
                        bordercolor=DARK_BG, arrowcolor=DARK_FG_DIM)
        style.map("Dark.Vertical.TScrollbar", background=[("active", DARK_BG_ACTIVE)])

        # status bar FIRST (side=bottom) so it sits under everything
        self._status = ttk.Label(root, style="Status.TLabel", anchor="w",
                                 text="island: checking... | store: ... | topk 2 · df 0.25")
        self._status.pack(side="bottom", fill="x")

        main = ttk.Frame(root, style="Dark.TFrame")
        main.pack(side="top", fill="both", expand=True)

        # input row (bottom of main), built before chat so pack order keeps chat expanding
        input_frame = ttk.Frame(main, style="Dark.TFrame")
        input_frame.pack(side="bottom", fill="x")
        self._input = tk.Text(input_frame, height=4, wrap="word", bg=DARK_BG_INPUT, fg=DARK_FG,
                              insertbackground=DARK_FG, selectbackground=SELECT_BG,
                              relief="flat", padx=8, pady=6, font=("Segoe UI", 10))
        self._input.pack(side="left", fill="both", expand=True, padx=(6, 3), pady=6)
        self._send_btn = tk.Button(input_frame, text="Send", command=self.handle_send,
                                   bg=DARK_BG_UI, fg=DARK_FG, activebackground=DARK_BG_ACTIVE,
                                   activeforeground=DARK_FG, relief="flat", width=8,
                                   highlightthickness=0, bd=0)
        self._send_btn.pack(side="right", padx=(3, 6), pady=6)

        # chat area
        chat_frame = ttk.Frame(main, style="Dark.TFrame")
        chat_frame.pack(side="top", fill="both", expand=True)
        scroll = ttk.Scrollbar(chat_frame, orient="vertical", style="Dark.Vertical.TScrollbar")
        scroll.pack(side="right", fill="y")
        self.chat = tk.Text(chat_frame, wrap="word", bg=DARK_BG, fg=DARK_FG_DIM,
                            insertbackground=DARK_FG, selectbackground=SELECT_BG, relief="flat",
                            padx=10, pady=8, state="disabled", font=("Segoe UI", 10),
                            yscrollcommand=scroll.set)
        self.chat.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.chat.yview)

        self.chat.tag_configure("system", foreground=LINK_BLUE)
        self.chat.tag_configure("user", foreground=DARK_FG)
        self.chat.tag_configure("ring", foreground=DARK_FG_DIM)
        self.chat.tag_configure("file_info", foreground=FILE_GREEN, font=("Segoe UI", 10, "italic"))
        self.chat.tag_configure("hint", foreground=HINT_GRAY, font=("Segoe UI", 9, "italic"))
        self.chat.tag_configure("link", foreground=LINK_BLUE, underline=True)
        self.chat.tag_bind("link", "<Enter>", lambda e: self.chat.config(cursor="hand2"))
        self.chat.tag_bind("link", "<Leave>", lambda e: self.chat.config(cursor=""))

        # drag-and-drop = ingest --add
        self.chat.drop_target_register(DND_FILES)
        self.chat.dnd_bind("<<Drop>>", self.handle_file_drop)

        # key bindings: Enter sends, Shift+Enter / KP_Enter behave per the fork contract
        self._input.bind("<Return>", self._on_return)
        self._input.bind("<KP_Enter>", self._on_return)
        self._input.bind("<Shift-Return>", lambda e: None)   # newline; wins over <Return>

        self._welcome()
        self._refresh_status()

        if start_island_check:
            threading.Thread(target=self._startup, daemon=True).start()

    # ------------------------------------------------------------ chat plumbing
    def _welcome(self):
        self.chat.config(state="normal")
        for text in (
            "Decoder Ring — plain text queries the store; commands start with /.",
            "/ingest <paths> · /remove <ident> · /stats · /rebuild <paths> (confirms; wipes "
            "store first) · /topk N · /df X · /help",
            "/collection <name> switch/create · /collections list · /sembuild rebuild the "
            "semantic index",
            "Enter sends · Shift+Enter newline · quote paths with spaces",
            "Drop .txt/.md/.pdf to ingest (auto-builds the semantic index into the active "
            "collection)",
        ):
            self.chat.insert("end", text + "\n", ("system",))
        self.chat.mark_set("hint_start", "end")
        self.chat.mark_gravity("hint_start", "left")
        self.chat.insert("end", "Drag .txt / .md / .pdf here to ingest (--add)...\n", ("hint",))
        self.chat.config(state="disabled")

    def append_to_chat(self, text, tag):
        """Marshal a chat line onto the Tk thread via root.after (foreman's worker->UI channel;
        no explicit queue). URLs become clickable per-URL link tags; the drop hint is removed on
        the first real message."""
        def update():
            self.chat.config(state="normal")
            if not self._first_msg_done:
                try:
                    self.chat.delete("hint_start", "end")
                except tk.TclError:
                    pass
                self._first_msg_done = True
            for seg, is_url in _url_segments(text):
                if is_url:
                    self._link_seq += 1
                    ltag = f"link{self._link_seq}"
                    self.chat.tag_bind(ltag, "<Button-1>", lambda e, u=seg: webbrowser.open(u))
                    self.chat.insert("end", seg, (tag, "link", ltag))
                else:
                    self.chat.insert("end", seg, (tag,))
            self.chat.insert("end", "\n", (tag,))
            self.chat.tag_raise("link")      # load-bearing: else the base tag fg wins over blue
            self.chat.see("end")
            self.chat.config(state="disabled")
        self.root.after(0, update)

    def _sys(self, text):
        self.append_to_chat(text, "system")

    def _help(self):
        self._sys("commands: plain text = query the store · /ingest <paths> = add (--add, then "
                  "auto-builds the semantic index) · /remove <ident> · /stats · /rebuild <paths> "
                  "(confirms; wipes first) · /collection <name> (switch/create) · /collections "
                  "(list) · /sembuild (rebuild the semantic index) · /topk N (1..10) · /df X "
                  "(0<X<=1.0) · /help")

    # ------------------------------------------------------------ collections (v1.8)
    def _col_args(self):
        """CLI args for the active collection: [] for default, else ['--collection', name]."""
        return [] if self.collection is None else ["--collection", self.collection]

    def _collection_exists(self, name):
        r"""A collection exists on disk once its store\manifest.json is present (default always)."""
        if name == "default":
            return True
        return (RING_DIR / "stores" / name / "store" / "manifest.json").is_file()

    def _recompute_store_paths(self):
        r"""Point the T15 manifest-path seam at the active collection's store\manifest.json."""
        if self.collection is None:
            self._store_manifest_path = str(RING_DIR / "store" / "manifest.json")
        else:
            self._store_manifest_path = str(
                RING_DIR / "stores" / self.collection / "store" / "manifest.json")

    def _switch_collection(self, target):
        """Switch the active collection (target=None -> default), recompute the manifest seam,
        announce, refresh the status line."""
        self.collection = target
        self._recompute_store_paths()
        disp = "default" if target is None else target
        self._sys(f"switched to collection {disp!r}")
        self._refresh_status()

    def _sem_state(self):
        r"""fresh | STALE | none | STALE (build pending) for the active collection's sidecar,
        computed with plain json+hashlib (never imports ring modules). Mirrors load_sidecar's
        chain: meta finalized AND meta.manifest_sha256 == sha256(manifest.json) AND
        cache_meta.sem_meta_sha256 == sha256(meta.json bytes)."""
        disp = "default" if self.collection is None else self.collection
        if disp in self._pending_sembuild:
            return "STALE (build pending)"
        store_dir = os.path.dirname(self._store_manifest_path)
        sem_dir = os.path.join(store_dir, "semantic")
        meta_path = os.path.join(sem_dir, "meta.json")
        try:
            with open(meta_path, "rb") as f:
                meta_bytes = f.read()
            meta = json.loads(meta_bytes)
        except (OSError, json.JSONDecodeError, ValueError):
            return "none"
        if not meta.get("finalized"):
            return "none"
        try:
            with open(self._store_manifest_path, "rb") as f:
                man_sha = hashlib.sha256(f.read()).hexdigest()
            with open(os.path.join(sem_dir, "cache", "cache_meta.json"), encoding="utf-8") as f:
                cm = json.load(f)
            if (meta.get("manifest_sha256") == man_sha
                    and cm.get("sem_meta_sha256") == hashlib.sha256(meta_bytes).hexdigest()):
                return "fresh"
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        return "STALE"

    # ------------------------------------------------------------ routing (main thread)
    def _route_command(self, text):
        """Map a submitted line to a ring_cli argv list, or None if handled locally (chat
        message + knob mutation only — no subprocess, so T12 can table-drive this)."""
        s = text.strip()
        if not s:
            return None
        if not s.startswith("/"):
            # plain text = query; the edge-trimmed (multi-line) text stays ONE argv element
            return ["query", text, "--topk", str(self.topk), "--df-max", str(self.df_max),
                    *self._col_args()]

        parts = s.split(None, 1)
        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            self._help()
            return None

        if cmd == "/topk":
            try:
                n = int(rest.strip())
            except ValueError:
                n = None
            if n is not None and 1 <= n <= 10:
                self.topk = n
                self._sys(f"topk = {n}")
                self._refresh_status()
                return None
            self._sys("Usage: /topk N  (integer 1..10)")
            return None

        if cmd == "/df":
            try:
                x = float(rest.strip())
            except ValueError:
                x = None
            if x is not None and 0.0 < x <= 1.0:
                self.df_max = x
                note = " (1.0 disables filtering)" if x == 1.0 else ""
                self._sys(f"df-max = {x}{note}")
                self._refresh_status()
                return None
            self._sys("Usage: /df X  (0 < X <= 1.0; 1.0 disables)")
            return None

        if cmd == "/stats":
            return ["stats", *self._col_args()]

        if cmd == "/collections":
            return ["collections"]              # subprocess = single source of truth for the list

        if cmd == "/sembuild":
            return ["sembuild", *self._col_args()]

        if cmd == "/collection":
            arg = rest.strip()
            if not arg:
                self._sys("Usage: /collection <name>  ('default' = the legacy store\\)")
                return None
            name = _sanitize_collection(arg)
            if name is None:
                self._sys("Usage: /collection <name>  (1-32 chars of a-z 0-9 _ - , starts alnum; "
                          "'default' = the legacy store)")
                return None
            target = None if name == "default" else name
            if self._collection_exists(name):
                self._switch_collection(target)
            elif self._ask("Create collection?",
                           f"'{name}' does not exist. Create it empty and switch? "
                           f"(the first ingest/drop creates it on disk)"):
                self._switch_collection(target)
            else:
                cur = "default" if self.collection is None else self.collection
                self._sys(f"staying on collection {cur!r}")
            return None

        if cmd == "/remove":
            args = _split_args(rest)
            if args is None or len(args) != 1:
                self._sys("Usage: /remove <name | doc_id | path>  (quote if spaced)")
                return None
            return ["remove", args[0], *self._col_args()]

        if cmd == "/ingest":
            args = _split_args(rest)
            if not args:
                self._sys("Usage: /ingest <path> [path...]  (quote paths with spaces)")
                return None
            # never a bare (wiping) ingest from the GUI; the worker auto-builds the sidecar after
            return ["ingest", "--add", *args, *self._col_args()]

        if cmd == "/rebuild":
            args = _split_args(rest)
            if not args:
                self._sys("Usage: /rebuild <path> [path...]  (WIPES the store, re-ingests fresh)")
                return None
            missing = [p for p in args if not Path(p).is_file()]
            if missing:
                # refuse BEFORE the confirm dialog — a rebuild with only bad paths would wipe
                # the store to empty (ring_cli's bare ingest fresh-rebuilds first).
                self._sys(f"/rebuild: file not found: {missing[0]} — refusing (a rebuild with "
                          f"only bad paths would wipe the store to empty)")
                return None
            disp = "default" if self.collection is None else self.collection
            ndocs, nchunks = self._store_counts() or (0, 0)
            ok = self._ask("Rebuild store?",
                           f"This WIPES collection '{disp}' ({ndocs} docs, {nchunks} chunks) and "
                           f"re-ingests {len(args)} file(s) fresh. Continue?")
            if ok:
                # bare = fresh; ring_cli prints its own explain-line; worker auto-builds the sidecar
                return ["ingest", *args, *self._col_args()]
            self._sys("rebuild cancelled")
            return None

        # any other "/..." (including multi-line slash lines) -> help
        self._help()
        return None

    def _ask(self, title, msg):
        if self._confirm is not None:
            return self._confirm(title, msg)
        return messagebox.askyesno(title, msg)

    # ------------------------------------------------------------ send / run
    def _on_return(self, event):
        if event.state & 0x0001:              # Shift held -> let the default newline insert
            return None
        self.handle_send()
        return "break"

    def handle_send(self, event=None):
        if self._busy:
            return "break"
        raw = self._input.get("1.0", "end")
        text = raw.strip("\r\n").rstrip()      # edge-trim only (keep interior newlines)
        if not text:
            return "break"
        self.append_to_chat(f"You: {text}", "user")
        argv = self._route_command(text)       # route on the main thread (no busy flash on usage)
        self._input.delete("1.0", "end")
        if argv is None:
            return "break"
        self._set_busy(True)
        auto = bool(argv and argv[0] == "ingest")   # ingest/rebuild -> auto-build the sidecar
        threading.Thread(target=self._run, args=(argv,),
                         kwargs={"auto_sembuild": auto}, daemon=True).start()
        return "break"

    def _stream(self, argv):
        """Spawn `python -u ring_cli.py *argv`, stream stdout into the chat, return the exit code.
        Worker-thread only; NO busy management (the caller holds the lock across a whole chain so
        nothing interleaves). `-u` is load-bearing (block-buffered children look dead for minutes)."""
        cmd = [sys.executable, "-u", str(RING_CLI), *argv]
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # no console flash under pythonw
        proc = subprocess.Popen(
            cmd, cwd=str(RING_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, **kwargs,
        )
        for line in proc.stdout:
            self.append_to_chat(line.rstrip("\n"), "ring")
        proc.wait()
        if proc.returncode == 0:
            self.append_to_chat("ring: done.", "system")
        else:
            self.append_to_chat(f"ring: exited with code {proc.returncode}.", "system")
        gui_log(f"cmd {argv} exit {proc.returncode}")
        return proc.returncode

    def _run(self, argv, auto_sembuild=False):
        """Worker: hold the busy-lock across the whole chain. query/sembuild ensure the island
        first (and flush any pending semantic builds); an ingest with auto_sembuild then builds
        the collection's semantic index (or defers it loudly when the island is down)."""
        try:
            if argv and argv[0] in ("query", "sembuild"):
                up = self._ensure_island()     # heal-or-degrade; the CLI degrades gracefully
                if up and self._pending_sembuild:
                    self._sys("island is back — running the pending semantic build(s) now...")
                    for disp in sorted(self._pending_sembuild):
                        col = [] if disp == "default" else ["--collection", disp]
                        if self._stream(["sembuild", *col]) == 0:
                            self._pending_sembuild.discard(disp)
            code = self._stream(argv)
            if auto_sembuild and argv and argv[0] == "ingest" and code == 0:
                self._auto_sembuild()
        except Exception as e:                 # broad: worker must never die silently
            self.append_to_chat(f"System Error: {e}", "system")
            gui_log(f"cmd {argv} EXC {e!r}")
        finally:
            self._set_busy(False)
            self._refresh_status()

    def _auto_sembuild(self):
        """After a successful ingest into the ACTIVE collection: build its semantic index if the
        island is up, else defer LOUDLY + mark it pending for auto-retry when the island returns.
        Measured cost (2026-08-07): ~20 s per 100k tokens at ~80 texts/s; the big default index
        re-keys ~157k windows = ~2-3 min (streamed; the UI never freezes)."""
        disp = "default" if self.collection is None else self.collection
        if self._ensure_island():
            self._sys(f"auto-building semantic index for {disp!r} (small drops ~seconds; "
                      f"~20 s per 100k tokens; the big default index ~2-3 min)...")
            if self._stream(["sembuild", *self._col_args()]) == 0:
                self._pending_sembuild.discard(disp)
                self._sys("semantic index fresh.")
            else:
                self._pending_sembuild.add(disp)
                self._sys(f"SEMANTIC INDEX STALE for {disp!r} — the semantic build did not "
                          f"complete. Natural-language quality drops to lexical-only until it is "
                          f"rebuilt. /sembuild retries.")
        else:
            self._pending_sembuild.add(disp)
            self._sys(f"SEMANTIC INDEX STALE for {disp!r} — files ingested, but the semantic index "
                      f"was NOT built (island down). Natural-language quality drops to lexical-only "
                      f"until it is rebuilt. /sembuild retries; I will auto-run it when the island "
                      f"returns.")

    # ------------------------------------------------------------ island ensure
    def _startup(self):
        self._ensure_island()
        self._refresh_status()

    def _ensure_island(self):
        """Health-check the island; cold-start the DEFAULT one (only) if it is down. Worker/
        startup threads only — never the Tk thread. Returns True if the island is reachable."""
        models = _api_tags(OLLAMA)
        if models is not None:
            self._island_state = "up" if DEMO_MODEL in models else "model-missing"
            self._refresh_status()
            return True
        if OLLAMA != DEFAULT_ISLAND:
            self._sys(f"endpoint {OLLAMA} unreachable — queries will print context only")
            self._island_state = "remote"
            self._refresh_status()
            return False                       # never cold-start a remote override
        self._sys("ring island not listening on 11436 — cold-starting minimized...")
        try:
            subprocess.Popen(["cmd", "/c", "start", "Ring Ollama Island", "/min", str(ISLAND_BAT)],
                             cwd=str(RING_DIR))
        except Exception as e:
            gui_log(f"island cold-start Popen failed: {e!r}")
        for _ in range(25):
            time.sleep(1)
            models = _api_tags(OLLAMA)
            if models is not None:
                self._island_state = "up" if DEMO_MODEL in models else "model-missing"
                self._sys("island is up.")
                self._refresh_status()
                return True
        self._sys("island did not come up in ~25 s — check the Ring Ollama Island window; "
                  "queries will print context only")
        self._island_state = "down"
        self._refresh_status()
        return False

    # ------------------------------------------------------------ status / busy
    def _store_counts(self):
        """(docs, chunks) for the manifest, or None if it is missing/unreadable (a valid but
        genuinely-empty manifest returns (0, 0), distinct from None)."""
        try:
            with open(self._store_manifest_path, encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        docs = m.get("docs", [])
        return len(docs), sum(int(d.get("n_chunks", 0)) for d in docs)

    def _refresh_status(self):
        disp = "default" if self.collection is None else self.collection
        counts = self._store_counts()
        if counts is None:
            # keep the "no store" substring (T15); still name the collection (T40)
            store_txt = f"col: {disp} · no store"
        else:
            ndocs, nchunks = counts
            store_txt = f"col: {disp} · {ndocs} docs · {nchunks:,} chunks · sem: {self._sem_state()}"
        st = self._island_state
        if st == "up":
            isl = f"island: up · {DEMO_MODEL}"
        elif st == "model-missing":
            isl = f"island: up · model missing (ollama pull {DEMO_MODEL})"
        elif st == "down":
            isl = "island: DOWN"
        elif st == "remote":
            isl = f"island: remote {OLLAMA}"
        else:
            isl = "island: checking..."
        text = f"{isl} | {store_txt} | topk {self.topk} · df {self.df_max}"
        self.root.after(0, lambda: self._status.config(text=text))

    def _set_busy(self, busy):
        self._busy = busy
        def update():
            state = "disabled" if busy else "normal"
            self._send_btn.config(state=state)
            self._input.config(state=state)
        self.root.after(0, update)

    # ------------------------------------------------------------ drag-and-drop
    def handle_file_drop(self, event):
        try:
            paths = _parse_drop_paths(event.data)
        except Exception:
            paths = [event.data.strip("{}")]
        accepted, rejected = _partition_ingestible(paths)
        for r in rejected:
            self._sys(f"rejected '{Path(r).name}' — drop .txt/.md/.pdf only")
        if not accepted:
            return
        if self._busy:
            self._sys("still working — drop again when the current run finishes")
            return
        disp = "default" if self.collection is None else self.collection
        self._sys(f"ingesting {len(accepted)} dropped file(s) into {disp!r} (--add, then "
                  f"auto-building the semantic index)...")
        self._set_busy(True)
        argv = ["ingest", "--add", *accepted, *self._col_args()]
        threading.Thread(target=self._run, args=(argv,),
                         kwargs={"auto_sembuild": True}, daemon=True).start()


def main():
    root = TkinterDnD.Tk()
    RingGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
