"""Merge coverage dumps (cov-*.json from covplug) and runner CPU (measure.json) into data/map.json.gz.

Everything here is derived from running sqlglot's own test suite at the pinned commit.
"""
import glob, gzip, json, sys

outdir, tests_json, dest = sys.argv[1], sys.argv[2], sys.argv[3]
tests = json.load(open(tests_json))["tests"]
ids = [t["id"] for t in tests]
num = {t: i for i, t in enumerate(ids)}
own = {}        # (file, line) -> set(test idx)   (whole test: base + pooled jobs)
base = {}       # (file, line) -> set(test idx)   (test body outside pooled jobs)
ent = {}        # (file, line) -> {test idx: set(job k)}
jobs = {}       # test idx -> list of (fixture, pos) or None per job
yields = {}
modusers = {}
attrusers = {}  # "module:Class.attr" -> tests that read that class attribute at run time
attr_global = set()   # sqlglot module -> tests that instantiated a class defined there (via MRO)
setup = {}      # (file, line) -> set(test file)
imp = set()
fopen = {}      # path -> set(test idx)
fopen_setup = {}
SRCROOT = sys.argv[4] if len(sys.argv) > 4 else "/tmp/claude-0/work/sqlglot"


def test_importers():
    """test module file -> test files whose (static, transitive) imports include it."""
    import ast, os
    files = sorted(glob.glob(f"{SRCROOT}/tests/**/*.py", recursive=True))
    rel = [f[len(SRCROOT) + 1:] for f in files]
    bymod = {r[:-3].replace("/", "."): r for r in rel}
    for r in rel:
        if r.endswith("/__init__.py"):
            bymod[r[: -len("/__init__.py")].replace("/", ".")] = r
    deps = {}
    for f, r in zip(files, rel):
        pkg = r[:-3].replace("/", ".").rsplit(".", 1)[0]
        out = set()
        try:
            tree = ast.parse(open(f).read())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                base = n.module or ""
                if n.level:
                    parts = pkg.split(".")
                    parts = parts[: len(parts) - (n.level - 1)]
                    base = ".".join(parts + ([n.module] if n.module else []))
                mods = [base] + [f"{base}.{a.name}" for a in n.names]
            for m in mods:
                while m:
                    if m in bymod:
                        out.add(bymod[m])
                    m = m.rpartition(".")[0]
        parts = r.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            init = "/".join(parts[:i]) + "/__init__.py"
            if init in rel and init != r:
                out.add(init)
        deps[r] = out
    imp = {}
    for r in rel:
        seen, stack = {r}, [r]
        while stack:
            for d in deps.get(stack.pop(), ()):
                if d not in seen:
                    seen.add(d)
                    stack.append(d)
        for d in seen:
            imp.setdefault(d, set()).add(r)
    return imp


IMPORTERS = test_importers()
import_exec = {}   # (file, line) -> sqlglot modules whose import ran it
attr_import = {}   # class attribute key -> sqlglot modules whose import read it
dial_sites = {}    # dialect-machinery module -> "file:line" of import-time statements instantiating it
klook = {}        # dict class attribute -> normalized key -> tests that looked it up
attr_known = set()
kempty = set()     # dict class attributes whose emptiness is asked outside a test
kunsafe = set()    # dict class attributes walked as a whole, or looked up outside a test
attr_sites = {}    # expression attribute key -> "file:line" of import-time statements reading it
imp_users = {}   # sqlglot module -> test files that import it at module level
import_dialects = {}   # sqlglot module -> modules of dialect machinery instantiated while it imports


def expand_mod(d):
    """MOD:<test module> buckets (module-level code of a test module, run once per process by
    whichever test file imported it first) count for every test file that imports it."""
    for k in [k for k in d.get("hits", {}) if k.startswith("FX:")]:
        # function code run while sqlglot module <k> is imported
        for a, b in d["hits"].pop(k):
            import_exec.setdefault((a, b), set()).add(k[3:])
    for key in ("hits", "users", "attr_reads", "opens"):
        sub = d.get(key)
        if not sub:
            continue
        for k in [k for k in sub if k.startswith("MOD:")]:
            v = sub.pop(k)
            if not k.startswith("MOD:tests/"):
                # import-time work of a sqlglot module
                if key == "users":
                    import_dialects.setdefault(k[4:], set()).update(x for x in v if x.startswith("sqlglot."))
                if key == "attr_reads":
                    for a in v:
                        attr_import.setdefault(a, set()).add(k[4:])
                    v = [a for a in v if not a.startswith("sqlglot.expressions:")]
                sub.setdefault("IMPORT", []).extend(v)
                continue
            for g in IMPORTERS.get(k[4:], {k[4:]}):
                if not g.split("/")[-1].startswith("test_"):
                    continue
                if key == "users":
                    # sqlglot modules a test module imports (or whose classes it instantiates)
                    # at import: those test files only notice if that import fails
                    for m_ in v:
                        imp_users.setdefault(m_, set()).add(g)
                    continue
                sub.setdefault("SETUP:" + g, []).extend(v)
    return d


for f in sorted(glob.glob(f"{outdir}/cov-*.json")):
    d = expand_mod(json.load(open(f)))
    for k_, v_ in d.get("attr_sites", {}).items():
        attr_sites.setdefault(k_, set()).update(v_)
    kunsafe.update(d.get("key_star", ()))
    attr_known.update(d.get("attr_known", ()))
    for b, items in d.get("key_lookups", {}).items():
        tg = None
        if b in num:
            tg = {num[b]}
        elif b.startswith("E:"):
            tid = b[2:].rsplit("#", 1)[0]
            tg = {num[tid]} if tid in num else None
        elif b.startswith("SETUP:"):
            tg = {num[x] for x in num if x.startswith(b[6:] + "::")} or None
        for it in items:
            ak, nk = it.split("|", 1)
            if tg is None and nk == "?":
                kempty.add(ak)  # emptiness asked outside a test: any test may depend on it
            elif tg is None or nk == "*":
                kunsafe.add(ak)
            else:
                klook.setdefault(ak, {}).setdefault("*" if nk == "*t" else nk, set()).update(tg)
    for k_, v_ in d.get("dial_sites", {}).items():
        dial_sites.setdefault(k_, set()).update(v_)
    for k, v in d["hits"].items():
        if k in num:
            for a, b in v:
                own.setdefault((a, b), set()).add(num[k])
                base.setdefault((a, b), set()).add(num[k])
        elif k.startswith("E:"):
            tid, j = k[2:].rsplit("#", 1)
            if tid in num:
                for a, b in v:
                    own.setdefault((a, b), set()).add(num[tid])
                    ent.setdefault((a, b), {}).setdefault(num[tid], set()).add(int(j))
        elif k.startswith("SETUP:"):
            for a, b in v:
                setup.setdefault((a, b), set()).add(k[6:])
        elif k == "IMPORT":
            imp.update((a, b) for a, b in v)
    for k, v in d.get("users", {}).items():
        tgt = None
        if k in num:
            tgt = {num[k]}
        elif k.startswith("SETUP:"):
            tgt = {num[x] for x in num if x.startswith(k[6:] + "::")}
        if tgt:
            for mod in v:
                if mod.startswith("sqlglot"):
                    modusers.setdefault(mod, set()).update(tgt)
    for k, v in d.get("attr_reads", {}).items():
        tgt = None
        if k in num:
            tgt = {num[k]}
        elif k.startswith("E:"):
            tid = k[2:].rsplit("#", 1)[0]
            tgt = {num[tid]} if tid in num else None
        elif k.startswith("SETUP:"):
            tgt = {num[x] for x in num if x.startswith(k[6:] + "::")}
        else:
            attr_global.update(v)
        if tgt:
            for key in v:
                attrusers.setdefault(key, set()).update(tgt)
    for k, v in d.get("jobs", {}).items():
        if k in num:
            jobs[num[k]] = v
    for k, v in d.get("yields", {}).items():
        if k in num:
            yields[num[k]] = v
    for k, v in d["opens"].items():
        if k.startswith("E:"):
            k = k[2:].rsplit("#", 1)[0]
        for p in v:
            if k in num:
                fopen.setdefault(p, set()).add(num[k])
            elif k.startswith("SETUP:"):
                fopen_setup.setdefault(p, set()).add(k[6:])

sets, setidx = [], {}
def sid(s):
    key = tuple(sorted(s))
    if key not in setidx:
        setidx[key] = len(sets)
        sets.append(list(key))
    return setidx[key]

# tests whose pooled jobs map 1:1 onto fixture entries (or onto a class attribute list)
ATTR = {"tests/test_executor.py::TestExecutor::test_execute_tpch": "tpch_sqls",
        "tests/test_executor.py::TestExecutor::test_execute_tpcds": "tpcds_sqls"}
partial = {}
for t, js in jobs.items():
    if not js:
        continue
    if ids[t] in ATTR and all(j is None for j in js):
        partial[t] = {"attr": ATTR[ids[t]], "n": len(js)}
    elif all(j is not None for j in js) and [tuple(j) for j in js] == [tuple(y) for y in yields.get(t, [])]:
        partial[t] = {"jobs": js}
base_lines, ent_lines = {}, {}
for (f, l), s_ in base.items():
    s2 = s_ & set(partial)
    if s2:
        base_lines.setdefault(f, {})[str(l)] = sid(s2)
ent_sets, ent_idx = [], {}
for (f, l), d_ in ent.items():
    rows = tuple(tuple([t] + sorted(ks)) for t, ks in sorted(d_.items()) if t in partial)
    if rows:
        if rows not in ent_idx:
            ent_idx[rows] = len(ent_sets)
            ent_sets.append([list(r) for r in rows])
        ent_lines.setdefault(f, {})[str(l)] = ent_idx[rows]

lines = {}
for (f, l), s in own.items():
    lines.setdefault(f, {})[str(l)] = sid(s)
setup_lines = {}
for (f, l), s in setup.items():
    setup_lines.setdefault(f, {})[str(l)] = sorted(s)
impl = {}
for f, l in imp:
    impl.setdefault(f, []).append(l)
for f in impl:
    impl[f].sort()

cpu = [None] * len(ids)
iso = [None] * len(ids)
try:
    m = json.load(open(f"{outdir}/measure.json"))
    for k, v in m.items():
        cpu[int(k)] = round(sum(c for _, c in v), 4)
except FileNotFoundError:
    pass
try:
    m = json.load(open(f"{outdir}/iso.json"))
    for k, v in m.items():
        iso[int(k)] = v["iso"]
        if v.get("cpu") is not None:
            cpu[int(k)] = round(v["cpu"], 4)
except FileNotFoundError:
    pass

# lines that only run the first time something is used in a process (lazy class setup):
# their per-test attribution is order dependent, so the selector treats them as module-wide
warm = {}
for wf in glob.glob(f"{outdir}/warm-*.json"):
    for tid, ls in json.load(open(wf)).items():
        for a, b in ls:
            if (a, b) not in imp and a.startswith("sqlglot/"):
                warm.setdefault(a, set()).add(b)

ie_mods = sorted({m_ for v in import_exec.values() for m_ in v})
ie_idx = {m_: i for i, m_ in enumerate(ie_mods)}
ie_sets, ie_sid = [], {}
import_exec_out = {}
for (f, l), v in import_exec.items():
    key = tuple(sorted(ie_idx[m_] for m_ in v))
    if key not in ie_sid:
        ie_sid[key] = len(ie_sets)
        ie_sets.append(list(key))
    import_exec_out.setdefault(f, {})[str(l)] = ie_sid[key]

out = {
    "attr_import": {k: sorted(v) for k, v in attr_import.items()},
    "attr_sites": {k: sorted(v) for k, v in attr_sites.items()},
    "dial_sites": {k: sorted(v) for k, v in dial_sites.items()},
    "import_dialects": {k: sorted(v) for k, v in import_dialects.items()},
    "imp_users": {k: sorted(v) for k, v in imp_users.items()},
    "import_exec": {"mods": ie_mods, "sets": ie_sets, "lines": import_exec_out},
    "warm": {f: sorted(v) for f, v in warm.items()},
    "modusers": {m_: sid(v) for m_, v in modusers.items()},
    "attrusers": {k_: sid(v) for k_, v in attrusers.items()},
    "klook": {ak: {nk: sid(v) for nk, v in d_.items()} for ak, d_ in klook.items() if ak not in kunsafe},
    "kunsafe": sorted(kunsafe),
    "kempty": sorted(kempty),
    "attr_known": sorted(attr_known),
    "attr_global": sorted(attr_global),
    "ids": ids,
    "seconds": [t["seconds"] for t in tests],
    "cpu": cpu,
    "iso": iso,
    "sets": sets,
    "lines": lines,
    "setup_lines": setup_lines,
    "import_lines": impl,
    "fopen": {p: sid(s) for p, s in fopen.items()},
    "fopen_setup": {p: sorted(s) for p, s in fopen_setup.items()},
    "partial": {str(k): v for k, v in partial.items()},
    "base_lines": base_lines,
    "ent_lines": ent_lines,
    "ent_sets": ent_sets,
}
with gzip.open(dest, "wt", compresslevel=9) as fh:
    json.dump(out, fh, separators=(",", ":"))
print("sets", len(sets), "files", len(lines), "partial", len(partial), file=sys.stderr)
