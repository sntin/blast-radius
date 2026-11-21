"""Blast Radius selector.

For every patch in /data/commits:
  1. apply it in memory to the pinned sqlglot tree (/opt/sqlglot),
  2. work out which tests could see the change (per-test line coverage, fixture reads,
     import graph and name references, all derived from the pinned commit),
  3. run the cheap candidates in a forked child against the patched code and keep the
     ones that fail; select the expensive candidates blindly,
  4. split the selection over workers to minimise busiest worker + 2 s per worker.

Logs go to stderr; stdout carries one JSON line per commit.
"""
import ast
import gzip
import io
import json
import os
import pickle
import re
import select as _select
import sys
import time
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("BR_SQLGLOT", "/opt/sqlglot")
DATA = os.environ.get("BR_DATA", "/data")
MAP = os.environ.get("BR_MAP", os.path.join(HERE, "..", "data", "map.json.gz"))
PAR = int(os.environ.get("BR_PAR", "2"))
T0 = time.time()
# The scorer's host may have many cores; keep numeric libraries from starting (and spinning) one
# thread per core. CPU seconds count, wall time barely does.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

FAMILY_DIRS = ("dialects", "parsers", "generators", "tokenizers", "typing")
# tests that start a separate interpreter: the in-memory patch cannot reach them
LAZY_SETUP = {"tests/test_executor.py::TestExecutor"}
NO_RUN_IDS = {"tests/dialects/test_dialect.py::TestDialect::test_lazy_load"}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- data


class _LazySets:
    """List of test sets, converted to frozensets (and renumbered) on first use."""

    def __init__(self, raw, remap):
        self.raw, self.remap, self.cache = raw, remap, {}

    def __getitem__(self, i):
        v = self.cache.get(i)
        if v is None:
            if self.remap is None:
                v = frozenset(self.raw[i])
            else:
                v = frozenset(x for x in (self.remap[j] for j in self.raw[i]) if x is not None)
            self.cache[i] = v
        return v

    def __len__(self):
        return len(self.raw)


MAP_PICKLE = os.path.join(HERE, "..", "data", "map.pickle")


def load_map():
    """The map as decoded at image build time (pickle), else from the gzip'd JSON."""
    try:
        if os.environ.get("BR_MAP") is None and os.path.getmtime(MAP_PICKLE) >= os.path.getmtime(MAP):
            with open(MAP_PICKLE, "rb") as fh:
                return pickle.load(fh)
    except (OSError, pickle.PickleError, EOFError):
        pass
    with gzip.open(MAP, "rt") as fh:
        return json.load(fh)


class Maps:
    def __init__(self, tests):
        m = load_map()
        self.rt_tests = tests  # runtime tests.json entries
        self.n = len(tests)
        id2num = {t["id"]: t["number"] for t in tests}
        # map-index -> runtime number (tests may be renumbered or missing)
        self.idx2num = [id2num.get(i) for i in m["ids"]]
        self.unmapped = [t["number"] for t in tests if t["id"] not in set(m["ids"])]
        self.identity = self.idx2num == list(range(len(self.idx2num)))
        self.sets = _LazySets(m["sets"], None if self.identity else self.idx2num)
        self.lines = {f: {int(k): v for k, v in d.items()} for f, d in m["lines"].items()}
        self.setup_lines = {f: {int(k): v for k, v in d.items()} for f, d in m["setup_lines"].items()}
        self.import_lines = {f: set(v) for f, v in m["import_lines"].items()}
        self.fopen = m["fopen"]
        self.fopen_setup = m["fopen_setup"]
        self.seconds = {t["number"]: t["seconds"] for t in tests}
        cpu = {}
        for i, c in enumerate(m["cpu"]):
            n = self.idx2num[i]
            if n is not None and c is not None:
                cpu[n] = c
        self.cpu = cpu
        self.by_file = {}
        for t in tests:
            self.by_file.setdefault(t["file"], set()).add(t["number"])
        self.id_of = {t["number"]: t["id"] for t in tests}
        self.no_run = {t["number"] for t in tests if t["id"] in NO_RUN_IDS}
        self._fileusers = {}
        self.touched_setup = set()
        self.partial = {}
        for k, v in m.get("partial", {}).items():
            n = self.idx2num[int(k)]
            if n is not None:
                self.partial[n] = v
        self.base_lines = {f: {int(k): v for k, v in d.items()} for f, d in m.get("base_lines", {}).items()}
        self.ent_lines = {f: {int(k): v for k, v in d.items()} for f, d in m.get("ent_lines", {}).items()}
        self.warm = {f: set(v) for f, v in m.get("warm", {}).items()}
        self.import_exec = m.get("import_exec") or {"mods": [], "sets": [], "lines": {}}
        self.ent_sets = m.get("ent_sets", [])
        self.modusers = m.get("modusers", {})
        self.attrusers = m.get("attrusers", {})
        self.attr_global = set(m.get("attr_global", []))
        self.attr_import = m.get("attr_import", {})
        self.attr_sites = m.get("attr_sites", {})
        self.dial_sites = m.get("dial_sites", {})
        self.import_dialects = m.get("import_dialects", {})
        self.imp_users = m.get("imp_users", {})
        self.iso = {}
        for i, c in enumerate(m.get("iso") or []):
            n = self.idx2num[i]
            if n is not None and c is not None:
                self.iso[n] = c
        self.file_fixed = {}
        over = {}
        for n, c in self.iso.items():
            if n in self.cpu:
                over.setdefault(self.id_of[n].split("::")[0], []).append(c - self.cpu[n])
        if over:
            base = min(sorted(v)[len(v) // 2] for v in over.values())
            for f, v in over.items():
                self.file_fixed[f] = max(sorted(v)[len(v) // 2] - base, 0.0)

    def tests_at(self, f, line):
        out = set()
        d = self.lines.get(f)
        if d and line in d:
            out |= self.sets[d[line]]
        d = self.setup_lines.get(f)
        if d and line in d:
            for tf in d[line]:
                self.touched_setup.add(tf)
                out |= self.by_file.get(tf, set())
        return out

    def split_at(self, f, line):
        """For partial-capable tests covering (f, line): (tests needing a full run, {test: job ks})."""
        full, ks = set(), {}
        d = self.base_lines.get(f)
        if d and line in d:
            full |= self.sets[d[line]]
        rows = (self.ent_lines.get(f) or {}).get(line)
        if rows is not None:
            rows = self.ent_sets[rows]
        if rows:
            for row in rows:
                t = self.idx2num[row[0]]
                if t is not None:
                    ks.setdefault(t, set()).update(row[1:])
        return full, ks

    def file_users(self, f):
        """Tests that execute any line of file f (outside import), or instantiate a class defined in it."""
        if f not in self._fileusers:
            out = set()
            mod = f[:-3].replace("/", ".").removesuffix(".__init__")
            if mod in self.modusers:
                out |= self.sets[self.modusers[mod]]
            for idx in set((self.lines.get(f) or {}).values()):
                out |= self.sets[idx]
            for tfs in (self.setup_lines.get(f) or {}).values():
                for tf in tfs:
                    out |= self.by_file.get(tf, set())
            if f.startswith("tests/") and f in self.by_file:
                out |= self.by_file[f]
            self._fileusers[f] = out
        return self._fileusers[f]


class Source:
    """Original tree: file text, ASTs, import graph, identifier index."""

    def __init__(self, root):
        self.root = root
        self.text = {}
        self._ast = {}
        self.pyfiles = []
        for base in ("sqlglot", "tests"):
            for dp, dn, fn in os.walk(os.path.join(root, base)):
                dn[:] = [d for d in dn if d not in ("__pycache__", "fixtures")]
                for f in fn:
                    if f.endswith(".py"):
                        rel = os.path.relpath(os.path.join(dp, f), root)
                        self.pyfiles.append(rel)
        self.pyfiles.sort()
        self._graph = None
        self._core = None
        self.hier = None
        self.spans = {}
        self.entry_refs = {}
        st = os.path.join(os.path.dirname(MAP), "static.pickle")
        if os.path.exists(st) and os.environ.get("BR_NO_STATIC") != "1":
            import pickle

            with open(st, "rb") as fh:
                d = pickle.load(fh)
            self._graph = (d["fwd"], d["rev"])
            self._core = d["core"]
            self.hier = d["hier"]
            self.spans = d["spans"]
            self.entry_refs = d.get("entry_refs", {})
            self._import_names = d.get("import_names")
            self._import_sites = d.get("import_sites")
            self._classfile = d.get("classfile")

    def span_of(self, f, line):
        """Innermost statement (start, end) holding line of original file f."""
        sp = self.spans.get(f)
        if sp is None:
            tree = self.tree(f)
            idx = StmtIndex(tree)
            spans = [(a, b) for a, b, n in idx.spans]
            # entries of a dict built at import (e.g. {exp.X: {...}, ...} tables): each entry on
            # its own, so a name in one entry does not stand for the whole table
            # an entry whose value names a function of this file (`exp.X: _x_sql`) also
            # stands for that function's code
            funcs = {}
            for st in getattr(tree, "body", []):
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    funcs[st.name] = (st.lineno, st.end_lineno)
            refs = self.entry_refs.setdefault(f, {})
            stack = [(st, None) for st in getattr(tree, "body", [])]
            while stack:
                st, cls = stack.pop()
                if isinstance(st, ast.ClassDef):
                    stack.extend((x, st) for x in st.body)
                elif isinstance(st, (ast.Assign, ast.AnnAssign)) and isinstance(st.value, ast.Dict):
                    own = dict(funcs)
                    if cls is not None:
                        for x in cls.body:
                            if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                own[x.name] = (x.lineno, x.end_lineno)
                    for k, v in zip(st.value.keys, st.value.values):
                        a, b = (k.lineno if k is not None else v.lineno), v.end_lineno
                        spans.append((a, b))
                        r = [own[x.id] for x in ast.walk(v) if isinstance(x, ast.Name) and x.id in own]
                        r += [own[x.attr] for x in ast.walk(v) if isinstance(x, ast.Attribute) and x.attr in own
                              and isinstance(x.value, ast.Name) and x.value.id in ("self", "cls")]
                        if r:
                            refs[(a, b)] = r
            best = {}
            for a, b in sorted(spans, key=lambda x: -(x[1] - x[0])):
                for x in range(a, b + 1):
                    best[x] = (a, b)
            sp = self.spans[f] = best
        return sp.get(line)

    def span_ranges(self, f, line):
        """Line ranges that stand for `line` of f: its statement (or dict entry), plus the
        functions that entry names."""
        sp = self.span_of(f, line)
        if sp is None:
            return [(line, line)]
        return [sp] + self.entry_refs.get(f, {}).get(sp, [])

    @property
    def import_names(self):
        """Names read while modules/classes are being built: class bodies, module bodies,
        metaclass methods and __init_subclass__ hooks (Load context, attributes, string keys)."""
        if getattr(self, "_import_names", None) is None:
            self._build_import_names()
        return self._import_names

    @property
    def classfile(self):
        """class name -> files of sqlglot/ defining a class of that name (any nesting)."""
        if getattr(self, "_classfile", None) is None:
            out = {}
            for f in self.pyfiles:
                if f.startswith("sqlglot/"):
                    tree = self.tree(f)
                    for n in ast.walk(tree) if tree is not None else ():
                        if isinstance(n, ast.ClassDef):
                            out.setdefault(n.name, set()).add(f)
            self._classfile = out
        return self._classfile

    @property
    def import_sites(self):
        """name -> set of sites reading it at import: ("meta", file, line of the metaclass
        method) or ("other", file, line)."""
        if getattr(self, "_import_sites", None) is None:
            self._build_import_names()
        return self._import_sites

    def _build_import_names(self):
        sites = {}

        def grab(node, f, site):
            for t in ast.walk(node):
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Load):
                    nm = t.id
                elif isinstance(t, ast.Attribute) and isinstance(t.ctx, ast.Load):
                    nm = t.attr
                elif isinstance(t, ast.Constant) and isinstance(t.value, str) and t.value.isidentifier():
                    nm = t.value
                else:
                    continue
                sites.setdefault(nm, set()).add(site)

        def visit(body, f, in_meta=False):
            for st in body:
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if in_meta:
                        grab(st, f, ("meta", f, st.lineno))
                    elif st.name == "__init_subclass__":
                        grab(st, f, ("meta", f, st.lineno))
                    else:
                        for d in st.decorator_list + st.args.defaults + st.args.kw_defaults:
                            if d is not None:
                                grab(d, f, ("other", f, st.lineno))
                elif isinstance(st, ast.ClassDef):
                    meta = any(isinstance(b, ast.Name) and b.id == "type" for b in st.bases)
                    for b in st.bases + st.decorator_list + [k.value for k in st.keywords]:
                        grab(b, f, ("other", f, st.lineno))
                    visit(st.body, f, meta)
                else:
                    # executed at import; for assignments only the value side is a read
                    if isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        if st.value is not None:
                            grab(st.value, f, ("other", f, st.lineno))
                        if isinstance(st, ast.AugAssign):
                            grab(st.target, f, ("other", f, st.lineno))
                    else:
                        grab(st, f, ("other", f, st.lineno))

        for f in self.pyfiles:
            if f.startswith("sqlglot/"):
                tree = self.tree(f)
                if tree is not None:
                    visit(tree.body, f)
        self._import_sites = sites
        self._import_names = set(sites)

    def prepare(self, dest):
        import pickle

        fwd, rev = self.graph
        hier = {}
        for f in self.pyfiles:
            if not f.startswith("sqlglot/"):
                continue
            tree = self.tree(f)
            if tree is None:
                continue
            for n in ast.walk(tree):
                if isinstance(n, ast.ClassDef):
                    for b in n.bases:
                        bn = b.attr if isinstance(b, ast.Attribute) else (b.id if isinstance(b, ast.Name) else None)
                        if bn:
                            hier.setdefault(bn, set()).add(n.name)
        for f in self.pyfiles:
            self.span_of(f, 0)
        with open(dest, "wb") as fh:
            pickle.dump({"fwd": fwd, "rev": rev, "core": self.core, "hier": hier, "spans": self.spans, "entry_refs": self.entry_refs,
                         "import_names": self.import_names, "import_sites": self.import_sites,
                         "classfile": self.classfile}, fh, protocol=4)

    def read(self, rel):
        if rel not in self.text:
            try:
                with open(os.path.join(self.root, rel), encoding="utf-8") as fh:
                    self.text[rel] = fh.read()
            except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError):
                self.text[rel] = None
        return self.text[rel]

    def tree(self, rel):
        if rel not in self._ast:
            src = self.read(rel)
            try:
                self._ast[rel] = ast.parse(src) if src is not None else None
            except SyntaxError:
                self._ast[rel] = None
        return self._ast[rel]

    def modfile(self, mod):
        p = mod.replace(".", "/")
        for cand in (p + ".py", p + "/__init__.py"):
            if os.path.exists(os.path.join(self.root, cand)):
                return cand
        return None

    def imports_of(self, rel):
        tree = self.tree(rel)
        out = set()
        if tree is None:
            return out
        pkg = rel[:-3].replace("/", ".")
        if rel.endswith("__init__.py"):
            pkg = pkg[: -len(".__init__")]
        else:
            pkg = pkg.rsplit(".", 1)[0]
        def module_level(body):
            # statements executed at import: module body, including if/try/with blocks, not defs/classes
            for st in body:
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                yield st
                for field in ("body", "orelse", "finalbody", "handlers"):
                    sub = getattr(st, field, None)
                    if isinstance(sub, list):
                        yield from module_level([x for x in sub if isinstance(x, (ast.stmt, ast.excepthandler))])
                if isinstance(st, ast.ExceptHandler):
                    yield from module_level(st.body)

        for node in module_level(tree.body):
            if isinstance(node, ast.Import):
                for a in node.names:
                    f = self.modfile(a.name)
                    if f:
                        out.add(f)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg.split(".")
                    base = base[: len(base) - node.level + 1]
                    mod = ".".join(base + ([node.module] if node.module else []))
                else:
                    mod = node.module or ""
                f = self.modfile(mod)
                if f:
                    out.add(f)
                for a in node.names:
                    f2 = self.modfile(mod + "." + a.name)
                    if f2:
                        out.add(f2)
                    elif mod == "sqlglot" and a.name == "exp":
                        out.add("sqlglot/expressions/__init__.py")
        # a package's parents are imported first
        parts = rel.split("/")
        for i in range(1, len(parts)):
            init = "/".join(parts[:i]) + "/__init__.py"
            if init != rel and os.path.exists(os.path.join(self.root, init)):
                out.add(init)
        return out

    @property
    def graph(self):
        if self._graph is None:
            fwd = {f: self.imports_of(f) for f in self.pyfiles}
            rev = {}
            for f, deps in fwd.items():
                for d in deps:
                    rev.setdefault(d, set()).add(f)
            self._graph = (fwd, rev)
        return self._graph

    @property
    def core(self):
        """Files loaded by `import sqlglot`."""
        if self._core is None:
            fwd, _ = self.graph
            seen, stack = set(), ["sqlglot/__init__.py"]
            while stack:
                f = stack.pop()
                if f in seen:
                    continue
                seen.add(f)
                stack.extend(fwd.get(f, ()))
            self._core = seen
        return self._core

    def rev_closure(self, files):
        _, rev = self.graph
        seen, stack = set(), list(files)
        while stack:
            f = stack.pop()
            if f in seen:
                continue
            seen.add(f)
            stack.extend(rev.get(f, ()))
        return seen

    def name_lines(self, name):
        """(file, line) of every line in sqlglot/ and tests/ that mentions identifier `name`."""
        if not hasattr(self, "_nl"):
            self._nl = {}
        if name not in self._nl:
            self._nl[name] = self._name_lines(name)
        return self._nl[name]

    def _name_lines(self, name):
        pat = re.compile(r"\b%s\b" % re.escape(name))
        out = []
        for f in self.pyfiles:
            src = self.read(f)
            if src is None or name not in src:
                continue
            for i, line in enumerate(src.split("\n"), 1):
                if name in line and pat.search(line):
                    out.append((f, i))
        return out


# --------------------------------------------------------------------------- patch


def parse_patch(text):
    """-> list of dicts {old, new, hunks:[(old_start, old_len, new_start, new_len, lines)]}"""
    files = []
    cur = None
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("diff --git "):
            cur = {"old": None, "new": None, "hunks": [], "git": l}
            files.append(cur)
            m = re.match(r"diff --git a/(.*) b/(.*)$", l)
            if m:
                cur["old"], cur["new"] = m.group(1), m.group(2)
        elif l.startswith("--- ") and (i + 1 < len(lines) and lines[i + 1].startswith("+++ ")):
            if cur is None or cur["hunks"]:
                cur = {"old": None, "new": None, "hunks": []}
                files.append(cur)
            a = l[4:].split("\t")[0].strip()
            b = lines[i + 1][4:].split("\t")[0].strip()
            cur["old"] = None if a == "/dev/null" else re.sub(r"^a/", "", a)
            cur["new"] = None if b == "/dev/null" else re.sub(r"^b/", "", b)
            i += 1
        elif l.startswith("new file mode") and cur is not None:
            cur["old"] = None
            cur["created"] = True
        elif l.startswith("deleted file mode") and cur is not None:
            cur["deleted"] = True
        elif l.startswith("rename from ") and cur is not None:
            cur["old"] = l[len("rename from "):]
        elif l.startswith("rename to ") and cur is not None:
            cur["new"] = l[len("rename to "):]
        elif l.startswith("@@") and cur is not None:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", l)
            if not m:
                raise ValueError("bad hunk header")
            os_, ol, ns, nl = int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1)
            if m.group(2) is None:
                ol = 1
            if m.group(4) is None:
                nl = 1
            body = []
            o_seen = n_seen = 0
            i += 1
            while i < len(lines) and (o_seen < ol or n_seen < nl):
                b = lines[i]
                if b.startswith("\\"):
                    body.append(b)
                    i += 1
                    continue
                tag = b[:1] if b else " "
                if tag == " ":
                    o_seen += 1
                    n_seen += 1
                elif tag == "-":
                    o_seen += 1
                elif tag == "+":
                    n_seen += 1
                else:
                    break
                body.append(b if b else " ")
                i += 1
            while i < len(lines) and lines[i].startswith("\\"):
                body.append(lines[i])
                i += 1
            cur["hunks"].append((os_, ol, ns, nl, body))
            continue
        i += 1
    for f in files:
        if f.get("deleted"):
            f["new"] = None
        if f.get("created"):
            f["old"] = None
    return files


def apply_hunks(old_text, hunks):
    """Apply hunks by line number. Returns (new_text, changed_old_lines, added_after_old_lines)."""
    old = old_text.split("\n") if old_text is not None else []
    # keep a trailing "" if file ends with newline
    out = []
    pos = 0  # index into old (0-based)
    changed = set()   # old line numbers (1-based) removed/replaced
    inserted = set()  # old line numbers after which lines were inserted (0 = top)
    no_nl_new = None
    for os_, ol, ns, nl, body in hunks:
        start = os_ - 1 if ol > 0 else os_
        out.extend(old[pos:start])
        p = start
        last = None
        for b in body:
            if b.startswith("\\"):
                if last == "+" or last == " ":
                    no_nl_new = True
                if last == "-":
                    no_nl_new = False if no_nl_new is None else no_nl_new
                continue
            tag, content = b[0], b[1:]
            if tag == " ":
                out.append(old[p] if p < len(old) else content)
                p += 1
            elif tag == "-":
                changed.add(p + 1)
                p += 1
            elif tag == "+":
                out.append(content)
                inserted.add(p)
            last = tag
        pos = p
    out.extend(old[pos:])
    text = "\n".join(out)
    if old_text is None:
        text = "\n".join(out) + ("" if no_nl_new else "\n")
    return text, changed, inserted


# --------------------------------------------------------------------------- analysis


def fixture_pairs(text):
    """Same split as tests/helpers.py load_sql_fixture_pairs: list of (meta-sql, expected)."""
    s = "\n".join([line for line in text.splitlines() if line and not line.startswith("--")])
    st = s.split(";")
    return [(st[i].strip(), st[i + 1].strip()) for i in range(0, len(st), 2) if i + 1 < len(st)]


def fixture_changed_positions(old, new):
    a, b = fixture_pairs(old), fixture_pairs(new)
    out = {i for i in range(min(len(a), len(b))) if a[i] != b[i]}
    out |= set(range(min(len(a), len(b)), max(len(a), len(b))))
    return out


def tokens_sig(src):
    """Token stream with comments and non-logical newlines removed; None if untokenizable."""
    try:
        toks = tokenize.generate_tokens(io.StringIO(src).readline)
        return [
            (t.type, t.string)
            for t in toks
            if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING)
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None


def derived_names(tree, names):
    """Module-level names computed (at import) from any of `names`, transitively."""
    names = set(names)
    out = set()
    body = [st for st in tree.body if isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign))]
    changed = True
    while changed:
        changed = False
        for st in body:
            loads = {t.id for t in ast.walk(st) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Load)}
            if not loads & (names | out):
                continue
            tgts = st.targets if isinstance(st, ast.Assign) else [st.target]
            for t in (x for tg in tgts for x in ast.walk(tg)):
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store) and t.id not in names | out:
                    out.add(t.id)
                    changed = True
    return out


class StmtIndex:
    """Innermost statement spans of a module."""

    def __init__(self, tree):
        self.spans = []
        self.tree = tree
        if tree is None:
            return
        parents = {}
        AST, stmt_t = ast.AST, ast.stmt
        stmts = []
        stack = [tree]
        while stack:
            node = stack.pop()
            if isinstance(node, stmt_t):
                stmts.append(node)
            for f in node._fields:
                v = getattr(node, f, None)
                if type(v) is list:
                    for x in v:
                        if isinstance(x, AST):
                            parents[x] = node
                            stack.append(x)
                elif isinstance(v, AST):
                    parents[v] = node
                    stack.append(v)
        self.parents = parents
        for n in stmts:
            if True:
                start = n.lineno
                if getattr(n, "decorator_list", None):
                    start = min([start] + [d.lineno for d in n.decorator_list])
                end = n.end_lineno
                body = getattr(n, "body", None)
                if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
                    end = max(start, body[0].lineno - 1)
                    # docstring-only first statement belongs to body, fine
                self.spans.append((start, end, n))

    def find(self, line):
        if not hasattr(self, "_best"):
            best = {}
            for a, b, n in sorted(self.spans, key=lambda x: -(x[1] - x[0])):
                for x in range(a, b + 1):
                    best[x] = (a, b, n)
            self._best = best
        return self._best.get(line)

    def scope_chain(self, node):
        """Enclosing def/class nodes, innermost first."""
        out = []
        p = self.parents.get(node)
        while p is not None:
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                out.append(p)
            p = self.parents.get(p)
        return out



def ast_diff(a, b, field=None, parent=None):
    """Yield (old_node, field, parent) at the innermost places where two ASTs differ."""
    if type(a) is not type(b):
        yield (parent if parent is not None else a), field, None
        return
    if isinstance(a, ast.AST):
        for f in a._fields:
            va, vb = getattr(a, f, None), getattr(b, f, None)
            if isinstance(va, list) and isinstance(vb, list):
                if len(va) != len(vb):
                    # align elements; report replaced/removed ones, and the owner for insertions
                    import difflib

                    da = [ast.dump(x) if isinstance(x, ast.AST) else repr(x) for x in va]
                    db = [ast.dump(x) if isinstance(x, ast.AST) else repr(x) for x in vb]
                    sm = difflib.SequenceMatcher(a=da, b=db, autojunk=False)
                    for tag, i1, i2, j1, j2 in sm.get_opcodes():
                        if tag == "equal":
                            continue
                        if tag == "replace" and i2 - i1 == j2 - j1:
                            for x, y in zip(va[i1:i2], vb[j1:j2]):
                                yield from ast_diff(x, y, f, a)
                            continue
                        yield a, f, parent
                        for x in va[i1:i2]:
                            if isinstance(x, ast.AST):
                                yield x, None, a
                else:
                    for x, y in zip(va, vb):
                        yield from ast_diff(x, y, f, a)
            elif isinstance(va, ast.AST) or isinstance(vb, ast.AST):
                if isinstance(va, ast.AST) and isinstance(vb, ast.AST):
                    yield from ast_diff(va, vb, f, a)
                else:
                    yield a, f, parent
            else:
                if va != vb or type(va) is not type(vb):
                    yield a, f, parent
        return
    if a != b:
        yield parent, field, None


ALL = "ALL"


class Planner:
    def __init__(self, maps, src):
        self.m = maps
        self.s = src
        self._stmt = {}

    def stmt_index(self, rel):
        if rel not in self._stmt:
            self._stmt[rel] = StmtIndex(self.s.tree(rel))
        return self._stmt[rel]

    def family_files(self, rel):
        """Dialect family: same basename across dialects/parsers/generators/..., closed over subclasses."""
        parts = rel.split("/")
        if len(parts) != 3 or parts[0] != "sqlglot" or parts[1] not in FAMILY_DIRS:
            return {rel}
        todo, seen = [rel], set()
        while todo:
            f = todo.pop()
            if f in seen:
                continue
            seen.add(f)
            p = f.split("/")
            if len(p) == 3 and p[0] == "sqlglot" and p[1] in FAMILY_DIRS and p[2] != "__init__.py":
                for d in FAMILY_DIRS:
                    sib = f"sqlglot/{d}/{p[2]}"
                    if sib not in seen and self.s.read(sib) is not None:
                        todo.append(sib)
            for r in self.s.rev_closure([f]):
                rp = r.split("/")
                if len(rp) == 3 and rp[0] == "sqlglot" and rp[1] in FAMILY_DIRS and r not in seen:
                    todo.append(r)
        return seen

    def module_users(self, rel):
        """Tests that could observe an import-time change of module `rel`."""
        if rel.startswith("sqlglot/"):
            if rel in self.s.core:
                # check whether non-family core module: everything imports it
                p = rel.split("/")
                if not (len(p) == 3 and p[1] in FAMILY_DIRS and p[2] not in ("__init__.py", "dialect.py")):
                    return ALL
            files = set()
            for f in self.s.rev_closure([rel]):
                if f.startswith("sqlglot/"):
                    if f in self.s.core and f != rel:
                        p = f.split("/")
                        if not (len(p) == 3 and p[1] in FAMILY_DIRS):
                            return ALL
                    files |= self.family_files(f)
                else:
                    files.add(f)
            out = set()
            for f in files:
                out |= self.m.file_users(f)
                if f.startswith("tests/"):
                    out |= self.m.by_file.get(f, set())
            return out
        if rel.startswith("tests/"):
            out = set()
            for f in self.s.rev_closure([rel]):
                out |= self.m.by_file.get(f, set())
                out |= self.m.file_users(f)
            return out
        return ALL

    def too_common(self, name):
        if len(name) < 3:
            return True
        return len(self.s.name_lines(name)) > 400

    def subclasses(self, cname):
        if not hasattr(self, "_hier") and self.s.hier is not None:
            self._hier = self.s.hier
        if not hasattr(self, "_hier"):
            hier = {}
            for f in self.s.pyfiles:
                if not f.startswith("sqlglot/"):
                    continue
                tree = self.s.tree(f)
                if tree is None:
                    continue
                for n in ast.walk(tree):
                    if isinstance(n, ast.ClassDef):
                        for b in n.bases:
                            bn = b.attr if isinstance(b, ast.Attribute) else (b.id if isinstance(b, ast.Name) else None)
                            if bn:
                                hier.setdefault(bn, set()).add(n.name)
            self._hier = hier
        out, todo = set(), [cname]
        while todo:
            c = todo.pop()
            for d in self._hier.get(c, ()):
                if d not in out:
                    out.add(d)
                    todo.append(d)
        return out

    def class_users(self, cname):
        """Tests that touch class cname or a subclass by name (incl. generator <name>_sql methods)."""
        names = {cname} | self.subclasses(cname)
        if len(names) > 60:
            return ALL
        out = set()
        for nm in names:
            if self.too_common(nm):
                return ALL
            hits = self.s.name_lines(nm) + self.s.name_lines(nm.lower() + "_sql")
            for f, l in hits:
                for a_, b_ in self.s.span_ranges(f, l):
                    for L in range(a_, b_ + 1):
                        out |= self.m.tests_at(f, L)
        return out

    def name_users(self, names):
        out = set()
        for name in names:
            if len(name) < 3:
                return ALL
            hits = self.s.name_lines(name)
            if len(hits) > 400:
                return ALL
            for f, l in hits:
                for a_, b_ in self.s.span_ranges(f, l):
                    for L in range(a_, b_ + 1):
                        out |= self.m.tests_at(f, L)
        return out

    def plan_py(self, rel, old, new, changed, inserted):
        """Candidate tests for a change to a python file. Returns (set|ALL, why)."""
        why = []
        new_tree = None
        if new is not None:
            try:
                new_tree = ast.parse(new)
            except (SyntaxError, ValueError):
                why.append("syntax-error")
                if rel.startswith("sqlglot/") and rel in self.s.core:
                    return ALL, why + ["core-broken"]
                return self.module_users(rel), why
        if old is None:
            # new module: only matters if something imports it, which needs another change
            if rel.startswith("tests/") and rel.endswith("conftest.py"):
                return ALL, ["new-conftest"]
            return set(), ["new-file"]
        if new is None:
            return self.module_users(rel), ["deleted"]

        old_tree = self.s.tree(rel)
        if old_tree is None or new_tree is None:
            return self.module_users(rel), ["unparsable"]
        idx = self.stmt_index(rel)
        diffs = list(ast_diff(old_tree, new_tree))
        if not diffs:
            return set(), ["comment-only"]
        future_ann = any(
            isinstance(n, ast.ImportFrom) and n.module == "__future__" and any(a.name == "annotations" for a in n.names)
            for n in old_tree.body
        )
        cand = set()
        imp_lines = self.m.import_lines.get(rel, set())
        done = set()
        for node, field, parent in diffs:
            key = (id(node), field)
            if key in done:
                continue
            done.add(key)
            c, w = self.plan_node(rel, idx, node, field, parent, future_ann, imp_lines)
            why += w
            if c == ALL:
                return ALL, why
            if not any(x.startswith(("rt:", "lambda:")) for x in w):
                self.full |= c
            cand |= c
        return cand, why

    def scope_of(self, idx, node):
        """Innermost function/lambda whose *body* contains node, else None. Also says if node
        sits in a function's signature (defaults, decorators) or annotation."""
        child = node
        p = idx.parents.get(node)
        while p is not None:
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child in p.body:
                    return p, "body"
                if child is p.returns or child is p.args:
                    return p, "sig"
                if child in p.decorator_list:
                    return p, "sig"
                return p, "sig"
            if isinstance(p, ast.Lambda):
                if child is p.body:
                    return p, "body"
                return p, "sig"
            if isinstance(p, ast.arguments):
                pass
            child = p
            p = idx.parents.get(p)
        return None, None

    def in_annotation(self, idx, node):
        child = node
        p = idx.parents.get(node)
        while p is not None:
            if isinstance(p, ast.arg) and child is p.annotation:
                return True
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is p.returns:
                return True
            if isinstance(p, ast.AnnAssign) and child is p.annotation:
                return True
            if isinstance(p, ast.stmt) and not isinstance(p, ast.AnnAssign):
                return False
            child = p
            p = idx.parents.get(p)
        return False

    def stmt_of(self, idx, node):
        p = node
        while p is not None and not isinstance(p, ast.stmt):
            p = idx.parents.get(p)
        return p

    def lines_tests(self, rel, a, b):
        out = set()
        for x in range(a, b + 1):
            out |= self.m.tests_at(rel, x)
        return out

    def rt_tests(self, rel, a, b):
        """Like lines_tests for a runtime change, recording which pooled jobs can see it."""
        out = set()
        for x in range(a, b + 1):
            ts = self.m.tests_at(rel, x)
            out |= ts
            full, ks = self.m.split_at(rel, x)
            for t in ts:
                if t in self.m.partial and t not in full and t in ks:
                    self.ks.setdefault(t, set()).update(ks[t])
                else:
                    self.full.add(t)
        return out

    def import_only_users(self, overrides):
        """Tests in files that import a patched sqlglot module (or a module importing it) at
        module level; they only notice the patch if that import fails."""
        files = set()
        pats = [f for f in overrides if f.startswith("sqlglot/") and f.endswith(".py")]
        if not pats:
            return set()
        for f in self.s.rev_closure(pats):
            if f.startswith("sqlglot/"):
                mod = f[:-3].replace("/", ".").removesuffix(".__init__")
                files |= set(self.m.imp_users.get(mod, ()))
        out = set()
        for tf in files:
            out |= self.m.by_file.get(tf, set())
        return out

    def check_modules(self, overrides):
        """Modules to import to check that a patch does not break importing; None if too many."""
        pats = [f for f in overrides if f.startswith("sqlglot/") and f.endswith(".py")]
        for f, new in overrides.items():
            if f in pats and new is None:
                return None  # deleted module: importers fail
        mods = {f[:-3].replace("/", ".").removesuffix(".__init__") for f in pats}
        for f in self.s.rev_closure(pats):
            if f.startswith("sqlglot/") and f.endswith(".py"):
                mod = f[:-3].replace("/", ".").removesuffix(".__init__")
                if mod in self.m.imp_users:
                    mods.add(mod)
        if len(mods) > 80:
            return None
        return sorted(mods)

    def import_parse_users(self, rel):
        """Users of modules that, while they import, instantiate dialect machinery defined in
        rel (e.g. templates parsed in a class body): a setting change there can break or
        change that import."""
        mod = rel[:-3].replace("/", ".").removesuffix(".__init__")
        out = set()
        for m_, fams in sorted(self.m.import_dialects.items()):
            if mod in fams:
                sites = [x for x in self.m.dial_sites.get(mod, ()) if x.rpartition(":")[0] == m_]
                su = self.sites_users(sites, m_)
                if su is not None:
                    # only what the instantiating statements built can differ
                    out |= su
                    continue
                if m_.split("/")[1] in ("dialects", "parsers", "generators", "tokenizers"):
                    # values computed there live in class attributes, whose readers are recorded;
                    # a failing import is caught by the import check
                    pre = m_[:-3].replace("/", ".") + ":"
                    for k, si in self.m.attrusers.items():
                        if k.startswith(pre):
                            out |= self.m.sets[si]
                    continue
                mu = self.module_users(m_)
                if mu == ALL:
                    return ALL
                out |= mu
        return out

    def owner_keys(self, classes, attrs):
        """attrusers keys for attributes `attrs` of the changed class chain (innermost first),
        the same chain under every subclass of the outermost class."""
        if not hasattr(self, "_akidx"):
            idx = {}
            for k in self.m.attrusers:
                q = k.split(":", 1)[1]
                qual, _, a = q.rpartition(".")
                idx.setdefault((qual, a), []).append(k)
            self._akidx = idx
        outer = classes[-1].name
        inner = ".".join(c.name for c in reversed(classes[:-1]))
        owners = set()
        for d in {outer} | self.subclasses(outer):
            owners.add(d)
            if inner:
                owners.add(d + "." + inner)
        out = []
        for q in owners:
            for a in attrs:
                out.extend(self._akidx.get((q, a), ()))
        return out

    def import_derivations(self, names, rel, classes):
        """Import-time reads of class attributes `names` of the changed class chain that can be
        followed: class bodies elsewhere that copy them into their own attribute of the same
        name (`X = {**Changed.X, ...}`), and metaclass methods. Returns (attributes derived by
        metaclasses, qualnames of the copying classes), or None when a read is not understood."""
        outer = classes[-1].name
        cf = self.s.classfile
        # class names defined in several files (nested Parser, Tokenizer, ...) are ambiguous
        relevant = {c.name for c in classes if len(cf.get(c.name, ())) <= 1} | {
            c for c in self.subclasses(outer) if len(cf.get(c, ())) <= 1
        }
        stem = rel.rsplit("/", 1)[-1][:-3]
        dial = self.s.tree(f"sqlglot/dialects/{stem}.py")
        if dial is not None:
            relevant |= {c.name for c in dial.body if isinstance(c, ast.ClassDef)}
        extra, meta, own_derived = set(), set(), set()
        site_list = [site for n in names for site in self.s.import_sites.get(n, ())]
        handled = set()
        for _ in range(6):
            grew = False
            for site in site_list:
                if site in handled:
                    continue
                if site[0] == "meta":
                    meta.add(site)
                    handled.add(site)
                    continue
                found = self._class_stmt(site[1], site[2])
                if found is None:
                    self.none_why = ("r1", locals().get("site"), locals().get("st")); return None
                st, chain = found
                refs = []
                for t in ast.walk(st):
                    if isinstance(t, ast.Attribute) and t.attr in names and isinstance(t.ctx, ast.Load):
                        comps, v = [], t.value
                        while isinstance(v, ast.Attribute):
                            comps.append(v.attr)
                            v = v.value
                        if isinstance(v, ast.Name):
                            comps.append(v.id)
                        refs.append(set(comps))
                    elif isinstance(t, ast.Name) and t.id in names and isinstance(t.ctx, ast.Load):
                        # a bare name is the enclosing class body's own attribute (or a module
                        # global), not the changed class's: only the changed class's own body counts
                        refs.append(set())
                    elif (
                        isinstance(t, ast.Call)
                        and isinstance(t.func, ast.Name)
                        and t.func.id in ("getattr", "setattr", "hasattr", "delattr")
                        and len(t.args) >= 2
                        and isinstance(t.args[1], ast.Constant)
                        and t.args[1].value in names
                    ):
                        refs.append(None)
                if not refs:
                    continue
                own = bool(chain) and site[1] == rel and chain[0].name == classes[0].name and chain[0].lineno == classes[0].lineno
                if own:
                    # part of the changed class's own definition; other attributes it builds
                    # from the changed one are derived values of the same class
                    handled.add(site)
                    if isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        for t in (st.targets if isinstance(st, ast.Assign) else [st.target]):
                            for x in ast.walk(t):
                                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store) and x.id not in names:
                                    own_derived.add(x.id)
                    continue
                if not any(r is None or r & relevant for r in refs):
                    continue  # reads that attribute of some other class
                handled.add(site)
                tgt = st.targets[0] if isinstance(st, ast.Assign) and len(st.targets) == 1 else None
                if not chain or not isinstance(tgt, ast.Name) or tgt.id not in names or any(r is None for r in refs):
                    self.none_why = ("r2", locals().get("site"), locals().get("st")); return None
                extra.add(".".join(c.name for c in reversed(chain)))
                if chain[0].name not in relevant and len(cf.get(chain[0].name, ())) <= 1:
                    relevant.add(chain[0].name)
                    relevant |= {c for c in self.subclasses(chain[0].name) if len(cf.get(c, ())) <= 1}
                    grew = True
            if not grew:
                break
        else:
            self.none_why = ("r3", locals().get("site"), locals().get("st")); return None
        if own_derived:
            # their own import-time readers would need following too
            if own_derived & self.s.import_names - set(names):
                for n in own_derived:
                    for site in self.s.import_sites.get(n, ()):
                        if site not in handled and site[0] != "meta":
                            self.none_why = ("own-derived", site, None); return None
        D = set(own_derived)
        if meta:
            D2 = self.meta_derived(names, sites=meta)
            if D2 is None:
                self.none_why = ("meta", None, None); return None
            D |= D2
        if True:
            if D - own_derived:
                # metaclass results land on the dialect classes that use these classes
                stems = {stem} | {f.rsplit("/", 1)[-1][:-3] for c in relevant if len(cf.get(c, ())) == 1
                                  for f in cf.get(c, ()) if f.split("/")[1] in FAMILY_DIRS}
                for st_ in stems:
                    t = self.s.tree(f"sqlglot/dialects/{st_}.py")
                    for c in t.body if t is not None else ():
                        if isinstance(c, ast.ClassDef):
                            extra.add(c.name)
                            extra |= self.subclasses(c.name)
        return D, extra

    def _class_stmt(self, f, line):
        """(statement starting at `line`, enclosing classes innermost first) at class or module
        level of file f."""
        tree = self.s.tree(f)
        if tree is None:
            self.none_why = ("r5", locals().get("site"), locals().get("st")); return None

        def walk(body, chain):
            for st in body:
                if st.lineno == line and not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    return st, chain
                if isinstance(st, ast.ClassDef) and st.lineno <= line <= st.end_lineno:
                    r = walk(st.body, [st] + chain)
                    if r is not None:
                        return r
                if isinstance(st, (ast.If, ast.Try)) and st.lineno <= line <= st.end_lineno:
                    for b in (st.body, st.orelse, getattr(st, "finalbody", [])):
                        r = walk(b, chain)
                        if r is not None:
                            return r
            self.none_why = ("r6", locals().get("site"), locals().get("st")); return None

        return walk(tree.body, [])

    def meta_derived(self, names, sites=None):
        """When every import-time read of `names` is in metaclass methods, the class attributes
        those methods derive from them (taint through locals, control flow and klass.X stores).
        None when some read is elsewhere or the flow is not understood."""
        def refs(node):
            out = set()
            for t in ast.walk(node):
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Load):
                    out.add(t.id)
                elif isinstance(t, ast.Attribute) and isinstance(t.ctx, ast.Load):
                    out.add(t.attr)
                elif isinstance(t, ast.Constant) and isinstance(t.value, str) and t.value.isidentifier():
                    out.add(t.value)
            return out

        sites_ok = set()
        pending = set(names)
        D = set()
        funcs = {}
        for it in range(6):
            new_sites = set()
            if it == 0 and sites is not None:
                new_sites = set(sites)
                pending = set()
            for n in pending:
                for site in self.s.import_sites.get(n, ()):
                    if site[0] != "meta":
                        self.none_why = ("r7", locals().get("site"), locals().get("st")); return None
                    new_sites.add(site)
            for site in new_sites - sites_ok:
                tree = self.s.tree(site[1])
                fn = None
                for x in ast.walk(tree) if tree is not None else ():
                    if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.lineno == site[2]:
                        fn = x
                        break
                if fn is None:
                    self.none_why = ("r8", locals().get("site"), locals().get("st")); return None
                funcs[site] = fn
            sites_ok |= new_sites
            seeds = set(names) | D
            for fn in funcs.values():
                T = set()

                def tainted(node):
                    return bool(refs(node) & (seeds | T))

                def store(target):
                    for x in ast.walk(target):
                        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                            T.add(x.id)
                        elif isinstance(x, ast.Attribute) and isinstance(x.ctx, ast.Store):
                            D.add(x.attr)
                            seeds.add(x.attr)
                        elif isinstance(x, ast.Subscript) and isinstance(x.value, ast.Attribute):
                            D.add(x.value.attr)
                            seeds.add(x.value.attr)

                def run(body, ctl):
                    for st in body:
                        if isinstance(st, (ast.If, ast.While)):
                            c = ctl or tainted(st.test)
                            run(st.body, c)
                            run(st.orelse, c)
                        elif isinstance(st, (ast.For, ast.AsyncFor)):
                            c = ctl or tainted(st.iter)
                            if c:
                                store(st.target)
                            run(st.body, c)
                            run(st.orelse, c)
                        elif isinstance(st, (ast.With, ast.AsyncWith)):
                            run(st.body, ctl or any(tainted(i.context_expr) for i in st.items))
                        elif isinstance(st, ast.Try):
                            run(st.body, ctl)
                            for h in st.handlers:
                                run(h.body, ctl)
                            run(st.orelse, ctl)
                            run(st.finalbody, ctl)
                        elif ctl or tainted(st):
                            if isinstance(st, ast.Assign):
                                for t in st.targets:
                                    store(t)
                            elif isinstance(st, (ast.AnnAssign, ast.AugAssign)):
                                store(st.target)
                            elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
                                fnc = st.value.func
                                if isinstance(fnc, ast.Attribute) and isinstance(fnc.value, ast.Attribute):
                                    D.add(fnc.value.attr)  # klass.X.update(...)
                                    seeds.add(fnc.value.attr)
                                elif isinstance(fnc, ast.Name) and fnc.id == "setattr" and len(st.value.args) >= 2:
                                    a1 = st.value.args[1]
                                    if isinstance(a1, ast.Constant) and isinstance(a1.value, str):
                                        D.add(a1.value)
                                        seeds.add(a1.value)
                                    else:
                                        raise ValueError("dynamic setattr")
                            elif isinstance(st, (ast.Return, ast.Pass, ast.Expr, ast.Delete, ast.Raise, ast.Assert)):
                                pass
                            else:
                                raise ValueError("unknown statement")

                try:
                    for _ in range(4):
                        before = (len(T), len(D))
                        run(fn.body, False)
                        if (len(T), len(D)) == before:
                            break
                except ValueError:
                    self.none_why = ("r9", locals().get("site"), locals().get("st")); return None
            pending = D - set(names)
            if all(site in sites_ok for n in pending for site in self.s.import_sites.get(n, ())):
                return D - set(names)
        self.none_why = ("r10", locals().get("site"), locals().get("st")); return None

    def tokenizer_field_users(self, name):
        """If Tokenizer setting `name` is only read (in sqlglot) to build the tokenizer core,
        as `field=self.<name>` in tokens.py, the tests that run core lines reading that field.
        None when it is read anywhere else."""
        field = None
        for f, l in self.s.name_lines(name):
            if not f.startswith("sqlglot/"):
                continue
            line = (self.s.read(f) or "").split("\n")[l - 1]
            m_ = re.match(r"\s*(\w+)\s*=\s*self\.%s\s*,?\s*$" % re.escape(name), line)
            if f == "sqlglot/tokens.py" and m_:
                if field is not None and field != m_.group(1):
                    return None
                field = m_.group(1)
                continue
            if re.match(r"\s*%s\s*(:[^=]*)?(=|$)" % re.escape(name), line):
                # a class-level definition; one that reads another class's value was already
                # ruled out (import-time use)
                continue
            return None
        if field is None:
            return None
        out = set()
        core = "sqlglot/tokenizer_core.py"
        pat = re.compile(r"\bself\.%s\b(?!\s*=[^=])" % re.escape(field))
        for i, line in enumerate((self.s.read(core) or "").split("\n"), 1):
            if pat.search(line):
                out |= self.m.tests_at(core, i)
        return out

    def expr_attr_users(self, rel, cls_node, names):
        """Tests that can see a changed argument spec (arg_types, ...) of expression class C.

        Readers were recorded at run time for C and every subclass, including the attributes
        derived from it when classes are created (required_args, and subclasses that spread
        C.arg_types). Code outside the expression modules that reads C.<attr> at import time is
        found statically: its module-level results are followed to the tests that use them."""
        cname = cls_node.name
        allc = {cname} | self.subclasses(cname)
        keys = [f"sqlglot.expressions:{c}.{a}" for c in allc for a in ("arg_types", "required_args", "is_var_len_args")]
        if any(k in self.m.attr_global for k in keys):
            return None
        out = set()
        for k in keys:
            if k in self.m.attrusers:
                out |= self.m.sets[self.m.attrusers[k]]
            for mod in self.m.attr_import.get(k, ()):
                # read while that module imports (e.g. a template parsed in a class body)
                su = self.attr_site_users(k, mod)
                if su is not None:
                    out |= su
                    continue
                mu = self.module_users(mod)
                if mu == ALL:
                    return ALL
                out |= mu
        out |= self.lines_tests(rel, cls_node.lineno, cls_node.end_lineno)
        for c in sorted(allc):
            hits = self.s.name_lines(c)
            if len(hits) > 2000:
                return None
            for f, l in hits:
                if not f.startswith("sqlglot/") or f.startswith("sqlglot/expressions/"):
                    continue
                text = self.s.read(f) or ""
                if not any(a in text for a in names | {"required_args"}):
                    continue  # (the statement below must mention one of them)
                tree = self.s.tree(f)
                if tree is None:
                    continue
                idx = self._stmt_index(f, tree)
                found = idx.find(l)
                if found is None:
                    continue
                st = found[2]
                chain = idx.scope_chain(st)
                if any(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) for x in chain):
                    continue  # run time: covered by the recorded readers
                seg = ast.get_source_segment(self.s.read(f), st) or ""
                if not any(a in seg for a in names | {"required_args"}):
                    continue
                if chain:
                    # a class attribute built from it at import: users of that module
                    mu = self.module_users(f)
                    if mu == ALL:
                        return ALL
                    out |= mu
                    continue
                tg = set()
                for t in (st.targets if isinstance(st, ast.Assign) else [getattr(st, "target", None)]):
                    for x in ast.walk(t) if t is not None else ():
                        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                            tg.add(x.id)
                if not tg:
                    mu = self.module_users(f)
                    if mu == ALL:
                        return ALL
                    out |= mu
                    continue
                tg |= derived_names(tree, tg)
                nu = self.name_users(tg)
                if nu == ALL:
                    return ALL
                out |= nu
        return out

    def attr_site_users(self, key, mod):
        """Tests that see what the import of `mod` built from expression attribute `key`: the
        recorded statements reading it are class-body or module-level assignments, so their
        results are followed to the readers of those class attributes / names. A failing import
        is caught by the import check (users of `mod` are selected then). None if not understood."""
        sites = [x for x in self.m.attr_sites.get(key, ()) if x.rpartition(":")[0] == mod]
        return self.sites_users(sites, mod)

    def sites_users(self, sites, mod):
        if not sites or mod.split("/")[1] not in FAMILY_DIRS:
            return None
        out = set()
        for site in sites:
            f, _, l = site.rpartition(":")
            tree = self.s.tree(f)
            if tree is None:
                return None
            idx = self._stmt_index(f, tree)
            found = idx.find(int(l))
            if found is None:
                return None
            st = found[2]
            # the outermost statement at that level (a multi-line assignment)
            chain = idx.scope_chain(st)
            if any(not isinstance(x, ast.ClassDef) for x in chain):
                return None
            while idx.parents.get(st) is not None and not isinstance(idx.parents.get(st), (ast.ClassDef, ast.Module)):
                st = idx.parents[st]
            if not isinstance(st, (ast.Assign, ast.AnnAssign)):
                return None
            tg = set()
            for t in (st.targets if isinstance(st, ast.Assign) else [st.target]):
                if not isinstance(t, ast.Name):
                    return None
                tg.add(t.id)
            if chain:
                # (no recorded key: no test read that attribute)
                for kk in self.owner_keys(chain, tg):
                    out |= self.m.sets[self.m.attrusers[kk]]
            else:
                tg |= derived_names(tree, tg)
                nu = self.name_users(tg)
                if nu == ALL:
                    return None
                out |= nu
                # class attributes aliasing those names at import (`X = _X` in a class body)
                for name in tg:
                    for f2, l2 in self.s.name_lines(name):
                        t2 = self.s.tree(f2)
                        if t2 is None:
                            return None
                        i2 = self._stmt_index(f2, t2)
                        fd = i2.find(l2)
                        if fd is None:
                            continue
                        c2 = i2.scope_chain(fd[2])
                        if not c2 or not all(isinstance(x, ast.ClassDef) for x in c2):
                            continue
                        s2 = fd[2]
                        while i2.parents.get(s2) is not None and not isinstance(i2.parents.get(s2), ast.ClassDef):
                            s2 = i2.parents[s2]
                        if not isinstance(s2, (ast.Assign, ast.AnnAssign)):
                            return None
                        t2s = st_targets = (s2.targets if isinstance(s2, ast.Assign) else [s2.target])
                        if not all(isinstance(t, ast.Name) for t in t2s):
                            return None
                        for kk in self.owner_keys(c2, {t.id for t in st_targets}):
                            out |= self.m.sets[self.m.attrusers[kk]]
        mu = self.module_users(mod)
        if mu == ALL:
            return None
        self.extra_imp |= mu
        self.extra_check.add(mod[:-3].replace("/", "."))
        return out

    def _stmt_index(self, f, tree):
        if not hasattr(self, "_sidx"):
            self._sidx = {}
        if f not in self._sidx:
            self._sidx[f] = StmtIndex(tree)
        return self._sidx[f]

    def import_exec_users(self, rel, a, b):
        """Tests that import a module whose import runs lines a..b of rel (a broken line there
        breaks that import, e.g. at collection)."""
        ie = self.m.import_exec
        lines = ie["lines"].get(rel)
        if not lines:
            return set(), []
        mods = set()
        for x in range(a, b + 1):
            k = lines.get(str(x))
            if k is not None:
                mods.update(ie["mods"][i] for i in ie["sets"][k])
        out = set()
        for mod in sorted(mods):
            mu = self.module_users(mod)
            if mu == ALL:
                return ALL, sorted(mods)
            out |= mu
        return out, sorted(mods)

    def is_warm(self, rel, a, b):
        w = self.m.warm.get(rel)
        return bool(w) and any(x in w for x in range(a, b + 1))

    def plan_node(self, rel, idx, node, field, parent, future_ann, imp_lines):
        why = []
        if isinstance(node, ast.Module):
            # top-level statement list changed (statement added/removed at module level)
            return self.module_users(rel), ["module-body"]
        if future_ann and self.in_annotation(idx, node):
            return set(), ["annotation"]
        scope, where = self.scope_of(idx, node)
        if scope is not None and where == "body" and not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and field is None and node is scope
        ):
            if isinstance(scope, ast.Lambda):
                b = scope.body
                a_, b_ = b.lineno, b.end_lineno
                if self.is_warm(rel, a_, b_):
                    mu = self.module_users(rel)
                    return (ALL if mu == ALL else mu | self.lines_tests(rel, a_, b_)), [f"warm:{a_}"]
                rt = self.rt_tests(rel, a_, b_)
                ie, iem = self.import_exec_users(rel, a_, b_)
                if iem:
                    return (ALL if ie == ALL else rt | ie), [f"lambda:{a_}", "import-exec:" + ",".join(iem)[:80]]
                # the line holding the lambda body may also be counted as import (lambda creation)
                return rt, [f"lambda:{a_}"] + ([] if rt else [f"uncovered:{a_}"])
            st = self.stmt_of(idx, node)
            if st is None or st is scope:
                # body list of the function itself changed: use the whole function
                a_, b_ = scope.lineno, scope.end_lineno
            else:
                sp = None
                for a, b, n in idx.spans:
                    if n is st:
                        sp = (a, b)
                        break
                a_, b_ = sp if sp else (st.lineno, st.end_lineno)
                if field in ("body", "orelse", "finalbody", "handlers") and node is st:
                    # statements inserted/removed inside a compound statement: whole statement
                    a_, b_ = st.lineno, st.end_lineno
            if self.is_warm(rel, a_, b_):
                mu = self.module_users(rel)
                return (ALL if mu == ALL else mu | self.lines_tests(rel, a_, b_)), [f"warm:{a_}"]
            rt = self.rt_tests(rel, a_, b_)
            ie, iem = self.import_exec_users(rel, a_, b_)
            if iem:
                why.append("import-exec:" + ",".join(iem)[:80])
                if ie == ALL:
                    return ALL, why + [f"rt:{a_}-{b_}"]
                rt = rt | ie
            if not rt:
                why.append(f"uncovered:{a_}")
            return rt, why + [f"rt:{a_}-{b_}"]
        # import-time change
        st = self.stmt_of(idx, node)
        if st is None:
            return self.module_users(rel), ["no-stmt"]
        why.append(f"import:{st.lineno}")
        extra = set()
        if rel.startswith("tests/"):
            return self.module_users(rel), why
        if scope is not None and where == "sig":
            fn = scope
            extra |= self.lines_tests(rel, fn.lineno, fn.end_lineno)
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nu = self.name_users([fn.name])
                if nu == ALL:
                    return ALL, why + ["name:" + fn.name]
                extra |= nu
            else:
                extra |= self.lines_tests(rel, st.lineno, st.end_lineno)
        names = set()
        target_nodes = [st] if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else []
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(st.name)
            extra |= self.lines_tests(rel, st.lineno, st.end_lineno)
        for tn in target_nodes:
            for t in ast.walk(tn):
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                    names.add(t.id)
                elif isinstance(t, ast.Attribute) and isinstance(t.ctx, ast.Store):
                    names.add(t.attr)
                elif isinstance(t, (ast.Import, ast.ImportFrom)):
                    for al in t.names:
                        names.add((al.asname or al.name).split(".")[0])
        if names and isinstance(idx.parents.get(st), ast.Module):
            names |= derived_names(idx.tree, names)
        # enclosing classes (and their subclasses) stand in for generic attribute names
        classes = [c for c in idx.scope_chain(st) if isinstance(c, ast.ClassDef)]
        if isinstance(st, ast.ClassDef):
            classes.insert(0, st)
        if (
            rel.startswith("sqlglot/expressions/")
            and len(classes) == 1
            and names
            and names <= EXPR_ARG_ATTRS
            and not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ):
            r = self.expr_attr_users(rel, classes[0], names)
            if r is not None:
                return r, why + ["expr-attr"]
        p_ = rel.split("/")
        family_mod = len(p_) == 3 and p_[1] in FAMILY_DIRS and p_[2] not in ("__init__.py", "dialect.py")
        if classes and family_mod and not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # A dialect class attribute is seen through instances of that class (or a subclass), or
            # by code that names the attribute on the class directly. Generic readers of a common
            # attribute name only see this value through an instance, which module users cover.
            why.append("dialect-attr")
            mod = rel[:-3].replace("/", ".")
            qual = ".".join(c.name for c in reversed(classes))
            keys = [f"{mod}:{qual}.{nm}" for nm in names]
            derived = None
            if names and names & self.s.import_names:
                res = self.import_derivations(names, rel, classes)
                derived = None
                if res is not None:
                    derived, extra = res
                    dkeys = self.owner_keys(classes, names | derived)
                    if extra:
                        self.owner_keys(classes, set())  # builds the key index
                        for q in extra:
                            for a in names | derived:
                                dkeys.extend(self._akidx.get((q, a), ()))
                    if any(k in self.m.attr_global for k in dkeys):
                        derived = None
            if derived is not None:
                # consumed only by metaclass code, into the attributes `derived` of the same
                # classes: readers of all of them (recorded at run time)
                out = set()
                for k in dkeys:
                    out |= self.m.sets[self.m.attrusers[k]]
                out |= self.lines_tests(rel, classes[0].lineno, classes[0].end_lineno)
                for nm in names:
                    if not self.too_common(nm):
                        nu = self.name_users([nm])
                        if nu != ALL and len(nu) <= 150:
                            out |= nu
                ip = self.import_parse_users(rel)
                if ip == ALL:
                    return ALL, why + ["import-parse"]
                return out | ip, why + ["meta-derived:" + ",".join(sorted(derived))[:80]]
            if names and all(k in self.m.attrusers and k not in self.m.attr_global for k in keys) and not (
                names & self.s.import_names
            ):
                # every reader of this attribute was seen at run time: those tests, plus code of the class
                out = set()
                for k in keys:
                    out |= self.m.sets[self.m.attrusers[k]]
                narrowed = None
                if classes[0].name == "Tokenizer":
                    cov = [self.tokenizer_field_users(nm) for nm in sorted(names)]
                    if all(c is not None for c in cov):
                        narrowed = set().union(*cov)
                if narrowed is not None:
                    # read only to configure the tokenizer core: tests that read it and run the
                    # core code that consults that setting
                    out &= narrowed
                    why.append("core-field")
                for c in classes:
                    out |= self.lines_tests(rel, c.lineno, c.end_lineno) if c is classes[0] else set()
                for nm in names:
                    if not self.too_common(nm):
                        nu = self.name_users([nm])
                        if nu != ALL and len(nu) <= 150:
                            out |= nu
                ip = self.import_parse_users(rel)
                if ip == ALL:
                    return ALL, why + ["import-parse"]
                return out | ip, why + ["attr-readers"]
            for c in classes:
                extra |= self.lines_tests(rel, c.lineno, c.end_lineno)
            for nm in names:
                if not self.too_common(nm):
                    nu = self.name_users([nm])
                    if nu != ALL and len(nu) <= 150:
                        extra |= nu
            mu = self.module_users(rel)
            if mu == ALL:
                return ALL, why + ["module-core"]
            return extra | mu, why
        generic = {n for n in names if self.too_common(n)}
        if generic and not classes:
            return ALL, why + ["names:" + ",".join(sorted(generic))]
        nu = self.name_users(names - generic)
        if nu == ALL:
            return ALL, why + ["names:" + ",".join(sorted(names))]
        extra |= nu
        for i, c in enumerate(classes):
            extra |= self.lines_tests(rel, c.lineno, c.end_lineno)
            if i < len(classes) - 1 and self.too_common(c.name):
                continue  # nested helper class such as Snowflake.Tokenizer: the outer class decides
            cu = self.class_users(c.name)
            if cu == ALL:
                return ALL, why + ["class:" + c.name]
            extra |= cu
        mu = self.module_users(rel)
        if mu == ALL:
            if not classes and not names:
                return ALL, why + ["module-core"]
            if not classes and generic:
                return ALL, why + ["module-core"]
        else:
            extra |= mu
        return extra, why

    def plan_fixture(self, rel, old, new):
        out = set()
        hit = False
        if rel in self.m.fopen:
            readers = self.m.sets[self.m.fopen[rel]]
            out |= readers
            hit = True
            fx = rel[len("tests/fixtures/"):] if rel.startswith("tests/fixtures/") else None
            changed = fixture_changed_positions(old, new) if fx and old is not None and new is not None else None
            for t in readers:
                cfg = self.m.partial.get(t)
                if changed is not None and cfg and cfg.get("jobs"):
                    ks = {k for k, j in enumerate(cfg["jobs"]) if j[0] == fx and j[1] in changed}
                    if ks:
                        self.ks.setdefault(t, set()).update(ks)
                    elif any(j[0] == fx for j in cfg["jobs"]):
                        # entries of this fixture exist but none changed: nothing to run
                        self.ks.setdefault(t, set())
                    else:
                        self.full.add(t)
                else:
                    self.full.add(t)
        for tf in self.m.fopen_setup.get(rel, ()):
            self.m.touched_setup.add(tf)
            out |= self.m.by_file.get(tf, set())
            self.full |= self.m.by_file.get(tf, set())
            hit = True
        if not hit:
            # unknown fixture: test files that mention its name
            base = os.path.basename(rel)
            stem = base.rsplit(".", 1)[0]
            for f in self.s.pyfiles:
                if f.startswith("tests/"):
                    t = self.s.read(f) or ""
                    if base in t or (len(stem) > 3 and stem in t):
                        out |= self.m.by_file.get(f, set())
                        out |= self.m.file_users(f) if not f.startswith("tests/test") else set()
                        hit = True
            if not hit:
                return set(), ["fixture-unused"]
            self.full |= out
        return out, ["fixture"]

    def plan(self, files):
        """files: list of (rel_old, rel_new, old_text, new_text, changed, inserted)."""
        cand = set()
        why = []
        self.m.touched_setup = set()
        self.full, self.ks = set(), {}
        self.extra_imp, self.extra_check = set(), set()
        for old_rel, new_rel, old, new, changed, inserted in files:
            rel = old_rel or new_rel
            if rel.endswith(".py") and (rel.startswith("sqlglot/") or rel.startswith("tests/")):
                if old_rel and new_rel and old_rel != new_rel:
                    c1, w1 = self.plan_py(old_rel, old, None, set(), set())
                    c, w = (ALL, ["rename"]) if c1 == ALL else (c1, w1)
                else:
                    c, w = self.plan_py(rel, old, new, changed, inserted)
                if c != ALL and not any(x.startswith(("rt:", "lambda:", "import:")) for x in w):
                    self.full |= c
            elif rel.startswith("tests/"):
                c, w = self.plan_fixture(rel, old, new)
                if c != ALL:
                    c = set(c)
            elif rel in ("pyproject.toml", "setup.cfg", "pytest.ini", "tox.ini", "conftest.py"):
                c, w = (ALL, ["config"]) if re.search(r"pytest", (old or "") + (new or "")) else (set(), ["config-other"])
            else:
                c, w = set(), ["other:" + rel]
            why += [f"{rel}:{x}" for x in w]
            if c == ALL:
                return ALL, why
            cand |= c
        # tests that only need some of their pooled jobs re-run
        self.partial_cfg = {}
        for t in list(cand):
            if t in self.full or t not in self.m.partial:
                continue
            ks = self.ks.get(t)
            if ks is None:
                continue
            if not ks:
                cand.discard(t)
                continue
            cfg = self.m.partial[t]
            if cfg.get("attr"):
                self.partial_cfg[t] = {"attr": (cfg["attr"], sorted(ks)), "frac": len(ks) / max(cfg["n"], 1)}
            else:
                fx = {}
                for k in ks:
                    if k < len(cfg["jobs"]):
                        f_, pos = cfg["jobs"][k]
                        fx.setdefault(f_, set()).add(pos)
                self.partial_cfg[t] = {"fixtures": {k: sorted(v) for k, v in fx.items()}, "frac": len(ks) / max(len(cfg["jobs"]), 1)}
        return cand, why


# --------------------------------------------------------------------------- cost


def split(nums, secs, k):
    ws = [[] for _ in range(k)]
    loads = [0] * k
    for n in sorted(nums, key=lambda n: -secs[n]):
        i = loads.index(min(loads))
        ws[i].append(n)
        loads[i] += secs[n]
    return ws, max(loads)


def best_split(nums, secs_s):
    """secs_s: number -> seconds. Returns (workers, cost seconds)."""
    if not nums:
        return [], 0.0
    ms = {n: round(secs_s[n] * 1000) for n in nums}
    best = None
    for k in range(1, min(8, len(nums)) + 1):
        ws, mk = split(nums, ms, k)
        c = mk + 2000 * k
        if best is None or c < best[1]:
            best = (ws, c)
    return [w for w in best[0] if w], best[1] / 1000.0


GAMBLE_SECONDS = 5.0     # recorded seconds and runner cpu above which a test is not run
GAMBLE_P_FAIL = 0.3      # rough chance that a cheaper test fails, which then costs the blind selection
GAMBLE_EVIDENCE = 5      # cheaper candidates that must run (and pass) to leave such a test out
EXPR_ARG_ATTRS = {"arg_types", "is_var_len_args"}
RUN_OVERHEAD = 0.12      # fork + import sqlglot in a child, seconds of CPU
FILE_OVERHEAD = 0.03     # per test module imported
CPU_SAFETY = float(os.environ.get("BR_CPU_SAFETY", "0.9"))
RUN_P_FAIL = float(os.environ.get("BR_RUN_P_FAIL", "0"))  # chance a run candidate fails and is selected anyway


def choose(cand, maps, partial_cfg=None, min_run=0, with_cost=False):
    """Split candidates into blind (always selected) and run (decided by running)."""
    partial_cfg = partial_cfg or {}
    secs = maps.seconds

    def cpu(n):
        c = max(maps.cpu.get(n, secs[n] * 1.5), 0.001)
        cfg = partial_cfg.get(n)
        if cfg is not None:
            c = 0.05 + c * cfg["frac"]
        return c * CPU_SAFETY

    order = sorted(cand, key=lambda n: (-(cpu(n) if n not in maps.no_run else 1e9), n))
    forced = [n for n in order if n in maps.no_run]
    rest = [n for n in order if n not in maps.no_run]
    best = None
    suffix = [0.0] * (len(rest) + 1)
    fixed = [0.0] * (len(rest) + 1)
    seen = set()
    acc = 0.0
    for i in range(len(rest) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + cpu(rest[i])
        f = maps.id_of[rest[i]].split("::")[0]
        if f not in seen:
            seen.add(f)
            acc += maps.file_fixed.get(f, FILE_OVERHEAD)
        fixed[i] = acc
    ks = list(range(0, min(len(rest), 40) + 1)) + [len(rest)]
    for k in sorted(set(ks) | set(range(max(0, len(rest) - min_run - 1), len(rest) + 1))):
        blind = forced + rest[:k]
        run = rest[k:]
        if len(run) < min_run and k != 0:
            continue
        _, bc = best_split(blind, secs)
        rc = (RUN_OVERHEAD + fixed[k] + suffix[k]) if run else 0.0
        if run and RUN_P_FAIL > 0:
            _, ac = best_split(blind + run, secs)
            rc += RUN_P_FAIL * (ac - bc)
        c = bc + rc
        if best is None or c < best[0] - 1e-9:
            best = (c, blind, run)
    if with_cost:
        return best[1], best[2], best[0]
    return best[1], best[2]


# --------------------------------------------------------------------------- running


def _single_thread_duckdb():
    """duckdb starts a thread per host core for every connection; one is plenty here."""
    import duckdb

    if getattr(duckdb, "_br_patched", False):
        return
    orig = duckdb.connect

    def connect(database=":memory:", read_only=False, config=None, **kw):
        config = dict(config or {})
        config.setdefault("threads", 1)
        return orig(database, read_only, config, **kw)

    duckdb.connect = connect
    duckdb._br_patched = True


GC_FIRST = {"tests/test_generator.py::TestGenerator::test_generator_memory"}
HEAVY_TEST_FILES = ("tests/test_executor.py", "tests/test_optimizer.py")


def prepare_child_env(heavy=True):
    """Import heavy third-party modules once in the parent so children fork warm."""
    mods = ["unittest", "unittest.mock", "decimal", "datetime", "json", "pickle", "concurrent.futures"]
    if heavy:
        mods += ["multiprocessing", "numpy", "pandas", "duckdb", "pandas.testing"]
    for m in mods:
        try:
            __import__(m)
        except Exception as e:  # pragma: no cover
            log("preimport failed", m, e)
    if "duckdb" in sys.modules:
        _single_thread_duckdb()
    sys.path.insert(0, ROOT)
    sys.path.insert(0, HERE)
    try:
        import sqlglot  # noqa: F401
        import sqlglot.optimizer  # noqa: F401
    except Exception as e:
        log("sqlglot preimport failed", e)
    import runner  # noqa: F401
    import syncpool  # noqa: F401


def spawn(job):
    """Fork a child that runs job['run'] against job's overrides. Returns (pid, read fd)."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.setpgid(0, 0)
            os.close(r)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            import runner

            stale = [k for k in sys.modules if k == "sqlglot" or k.startswith("sqlglot.") or k == "tests" or k.startswith("tests.")]
            if job["purge"]:
                for k in stale:
                    del sys.modules[k]
            if job.get("check"):
                # import check in its own process, so it does not change what tests see
                cpid = os.fork()
                if cpid == 0:
                    try:
                        runner.check_imports(ROOT, job["overrides"], job["redirects"], job["check"], w)
                    finally:
                        os._exit(0)
                os.waitpid(cpid, 0)
            tests = sorted(((n, job["ids"][n]) for n in job["run"]), key=lambda x: x[1])
            # tests that time gc.collect/tracemalloc run on their own, before the heap fills up
            # (and without changing what the other tests see, e.g. which dialects are loaded)
            first = [t for t in tests if t[1] in GC_FIRST]
            if first and len(first) < len(tests):
                tests = [t for t in tests if t[1] not in GC_FIRST]
                gpid = os.fork()
                if gpid == 0:
                    try:
                        runner.run(ROOT, job["overrides"], job["redirects"], first, w, job["limits"], job["lazy"], job["partial"])
                    finally:
                        os._exit(0)
                os.waitpid(gpid, 0)
            runner.run(ROOT, job["overrides"], job["redirects"], tests, w, job["limits"], job["lazy"], job["partial"])
            if os.environ.get("BR_DEBUG"):
                d = sys.modules.get("sqlglot.dialects")
                os.write(w, f"X purge={job['purge']} dialects={getattr(d, 'DIALECTS', None)} file={getattr(d, '__file__', None)}\n".encode())
            os.write(w, b"D\n")
        except BaseException:
            pass
        finally:
            os._exit(0)
    os.close(w)
    return pid, r


def main():
    tests = json.load(open(os.path.join(DATA, "tests.json")))["tests"]
    maps = Maps(tests)
    src = Source(ROOT)
    planner = Planner(maps, src)
    commits = sorted(f[:-6] for f in os.listdir(os.path.join(DATA, "commits")) if f.endswith(".patch"))
    n = len(commits)
    budget = min(max(3.9 * n, 100.0), 250.0)
    log(f"{n} commits, budget {budget:.0f}s, unmapped tests {len(maps.unmapped)}, loaded in {time.process_time():.2f}s cpu")

    jobs = []
    results = {}
    for cid in commits:
        job = {"cid": cid, "blind": set(), "run": [], "fail": set(), "done": set()}
        files = []
        jobs.append(job)
        try:
            text = open(os.path.join(DATA, "commits", cid + ".patch"), encoding="utf-8", errors="replace").read()
            pfiles = parse_patch(text)
            files = []
            overrides, redirects = {}, {}
            for pf in pfiles:
                old_rel, new_rel = pf["old"], pf["new"]
                rel = old_rel or new_rel
                old = src.read(old_rel) if old_rel else None
                if old_rel and old is None and not pf.get("created"):
                    # binary or missing file
                    old = ""
                new, changed, inserted = apply_hunks(old, pf["hunks"])
                if new_rel is None:
                    new = None
                files.append((old_rel, new_rel, old, new, changed, inserted))
                if rel.endswith(".py"):
                    if old_rel and new_rel and old_rel != new_rel:
                        overrides[old_rel] = None
                        overrides[new_rel] = new
                    else:
                        overrides[rel] = new
                else:
                    if old_rel:
                        redirects[os.path.join(ROOT, old_rel)] = new if new_rel == old_rel else None
                    if new_rel:
                        redirects[os.path.join(ROOT, new_rel)] = new
            cand, why = planner.plan(files)
            lazy = {k for k in LAZY_SETUP if k.split("::")[0] not in maps.touched_setup
                    and not any(o.startswith("tests/") for o in overrides)}
        except Exception as e:
            lazy = set()
            log(cid, "planning failed", repr(e))
            cand, why = ALL, ["error"]
            overrides, redirects = {}, {}
        pcfg = {}
        if cand == ALL:
            cand = set(range(maps.n))
        else:
            pcfg = getattr(planner, "partial_cfg", {})
        # Very expensive tests are only worth selecting when the change breaks something.
        # If enough cheaper candidates run and all pass, leave them out; if any fails, select them.
        gamble = {x for x in cand if maps.seconds[x] >= GAMBLE_SECONDS and maps.cpu.get(x, 99) >= GAMBLE_SECONDS}
        job["gamble"] = set()
        # only for changes to sqlglot itself: test helpers and fixtures feed the big tests directly
        only_lib = bool(files) and all((f[0] or f[1] or "").startswith("sqlglot/") for f in files)
        if gamble and only_lib:
            rest = cand - gamble
            runnable = [x for x in rest if x not in maps.no_run]
            if len(runnable) >= GAMBLE_EVIDENCE:
                _, _, c_no = choose(cand, maps, pcfg, with_cost=True)
                _, _, c_g = choose(rest, maps, pcfg, min_run=GAMBLE_EVIDENCE, with_cost=True)
                _, c_giant = best_split(sorted(cand), maps.seconds)
                if c_g + GAMBLE_P_FAIL * c_giant < c_no:
                    job["gamble"] = gamble
                    cand = rest
        blind, run = choose(cand, maps, pcfg, min_run=GAMBLE_EVIDENCE if job["gamble"] else 0)
        job["partial"] = {n: pcfg[n] for n in run if n in pcfg}
        rt_only = all(w.split(":", 2)[1].startswith(("rt", "lambda", "comment-only", "uncovered", "fixture", "other", "annotation"))
                      for w in why if w.count(":") >= 1) if why and why != ["error"] else False
        extra_imp = getattr(planner, "extra_imp", set()) if why != ["error"] else set()
        if cand != set(range(maps.n)) and (not rt_only or extra_imp):
            imp = planner.import_only_users(overrides)
            imp = imp if imp == ALL else (imp | extra_imp) - set(cand)
            if imp == ALL or imp:
                mods = planner.check_modules(overrides)
                if mods is not None:
                    mods = sorted(set(mods) | planner.extra_check)
                if imp == ALL or mods is None:
                    blind = list(blind) + sorted(set(range(maps.n)) - set(cand) if imp == ALL else imp)
                else:
                    job["imp_only"] = imp
                    job["check"] = mods
        job.update(blind=set(blind), run=run, overrides=overrides, redirects=redirects, why=why, lazy=lazy)
        log(cid, f"cand={len(cand)} blind={len(blind)} run={len(run)}", ";".join(why)[:300])

    # run phase
    log(f"planned in {time.process_time():.2f}s cpu")
    if any(j["run"] for j in jobs):
        heavy = sum(1 for j in jobs if any(maps.id_of[x].split("::")[0] in HEAVY_TEST_FILES for x in j["run"]))
        prepare_child_env(heavy >= 2)
    import gc

    gc.collect()
    gc.freeze()  # children's gc passes (some tests call gc.collect) skip the parent's objects
    log(f"prepared in {time.process_time():.2f}s cpu")
    loaded = set(k for k in sys.modules if k.startswith("sqlglot") or k.startswith("tests"))
    ids = {t["number"]: t["id"] for t in tests}
    queue = [j for j in jobs if j["run"] or j.get("check")]
    # most expensive first so the tail is short
    queue.sort(key=lambda j: -sum(maps.cpu.get(x, 0.1) for x in j["run"]))
    active = {}
    while queue or active:
        while queue and len(active) < PAR:
            j = queue.pop(0)
            if time.time() - T0 > budget:
                log(j["cid"], "out of time: selecting candidates blindly")
                j["blind"] |= set(j["run"])
                j["run"] = []
                continue
            mods = {k[:-3].replace("/", ".").removesuffix(".__init__") for k in j["overrides"]}
            j["purge"] = bool(mods & loaded)
            j["ids"] = ids
            j["limits"] = {x: max(2.0, 4 * maps.cpu.get(x, maps.seconds[x]) + 1.0) for x in j["run"]}
            j["deadline"] = time.time() + 5 + 2 * sum(j["limits"].values())
            j["buf"] = b""
            pid, fd = spawn(j)
            active[fd] = (pid, j)
        if not active:
            break
        timeout = max(0.05, min(j["deadline"] for _, j in active.values()) - time.time())
        ready, _, _ = _select.select(list(active), [], [], timeout)
        for fd in ready:
            pid, j = active[fd]
            data = os.read(fd, 65536)
            if data:
                j["buf"] += data
                maybe_stop(j, pid, maps)
                continue
            os.close(fd)
            _, _, ru = os.wait4(pid, 0)
            j["cpu"] = ru.ru_utime + ru.ru_stime
            del active[fd]
            finish(j, maps)
        now = time.time()
        for fd, (pid, j) in list(active.items()):
            if now > j["deadline"]:
                log(j["cid"], "child deadline hit")
                kill_group(pid)
                os.waitpid(pid, 0)
                os.close(fd)
                del active[fd]
                finish(j, maps)

    for j in jobs:
        sel = set(j["blind"]) | j["fail"] | (set(j["run"]) - j["done"])
        if j.get("imp_only"):
            # test files that import a patched module at module level: only if it fails to import
            if j.get("imp_bad") or not set(j.get("check", ())) <= j.get("imp_ok", set()):
                sel |= j["imp_only"]
                log(j["cid"], f"import check failed: {len(j['imp_only'])} importing tests selected")
        if j.get("gamble"):
            ran_ok = len(j["done"] - j["fail"])
            if j["fail"] or (set(j["run"]) - j["done"]) or ran_ok < GAMBLE_EVIDENCE:
                sel |= j["gamble"]
            else:
                log(j["cid"], f"left out {len(j['gamble'])} expensive candidates: {ran_ok} cheaper ones passed")
        workers, cost = best_split(sorted(sel), maps.seconds)
        log(j["cid"], f"select={len(sel)} run={len(j['run'])} fail_found={len(j['fail'])} cost={cost:.2f} cpu={j.get('cpu', 0):.2f}")
        sys.stdout.write(json.dumps({"commit": j["cid"], "workers": workers}) + "\n")
    sys.stdout.flush()
    import resource

    ch = resource.getrusage(resource.RUSAGE_CHILDREN)
    log(f"total wall {time.time() - T0:.1f}s cpu parent {time.process_time():.1f}s children {ch.ru_utime + ch.ru_stime:.1f}s")


STOP_CPU_FACTOR = float(os.environ.get("BR_STOP_CPU", "1.5"))   # grader cpu per recorded runner cpu
STOP_P_PASS = float(os.environ.get("BR_STOP_P_PASS", "0.35"))   # chance a later candidate passes once one failed


def kill_group(pid):
    for f in (lambda: os.killpg(pid, 9), lambda: os.kill(pid, 9)):
        try:
            f()
        except (ProcessLookupError, PermissionError, OSError):
            pass


def maybe_stop(j, pid, maps):
    """Once a candidate failed, the change is real and the rest likely fail too: stop running
    them when that is cheaper than the expected over-selection of selecting them unrun."""
    if j.get("stopped") or STOP_P_PASS >= 1:
        return
    done, fail = set(), set()
    for line in j["buf"].decode(errors="replace").splitlines():
        p = line.split()
        if len(p) >= 2 and p[0].isdigit():
            done.add(int(p[0]))
            if p[1] == "F":
                fail.add(int(p[0]))
        elif p and p[0] == "I":
            return  # import failure lines are handled by finish
    if not fail:
        return
    rest = [x for x in j["run"] if x not in done]
    if not rest:
        return
    cpu_rest = sum(maps.cpu.get(x, maps.seconds[x]) for x in rest) * STOP_CPU_FACTOR
    sel = sorted(set(j["blind"]) | fail)
    _, c0 = best_split(sel, maps.seconds)
    _, c1 = best_split(sorted(set(sel) | set(rest)), maps.seconds)
    if cpu_rest > STOP_P_PASS * (c1 - c0):
        j["stopped"] = True
        log(j["cid"], f"stop after a failure: {len(rest)} unrun candidates selected (cpu {cpu_rest:.2f} vs {c1 - c0:.2f} s)")
        kill_group(pid)


def finish(j, maps):
    done, fail = set(), set()
    complete = False
    for line in j["buf"].decode(errors="replace").splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "D":
            complete = True
        elif p[0] == "M":
            if len(p) >= 3:
                (j.setdefault("imp_ok", set()) if p[1] == "ok" else j.setdefault("imp_bad", set())).add(p[2])
        elif p[0] == "I":
            f = p[1]
            for x in j["run"]:
                if maps.id_of[x].split("::")[0] == f:
                    fail.add(x)
                    done.add(x)
        else:
            try:
                x = int(p[0])
            except ValueError:
                continue
            done.add(x)
            if p[1] == "F":
                fail.add(x)
            elif p[1] == "P" and len(p) > 2:
                # a test that got much slower than recorded may time out for the grader
                if float(p[2]) > 3 * maps.cpu.get(x, maps.seconds[x]) + 2.0:
                    fail.add(x)
    j["done"] = done
    j["fail"] = fail
    if os.environ.get("BR_DEBUG"):
        log(j["cid"], "results", j["buf"].decode(errors="replace").replace("\n", " | ")[:3000], "partial", sorted(j["partial"])[:20])
    if not complete and not j.get("stopped"):
        log(j["cid"], "child did not complete;", len(set(j["run"]) - done), "tests unknown")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--prepare"]:
        Source(ROOT).prepare(sys.argv[2])
        with gzip.open(MAP, "rt") as fh:
            m = json.load(fh)
        with open(MAP_PICKLE, "wb") as fh:
            pickle.dump(m, fh, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        main()
