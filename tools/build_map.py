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
modusers = {}   # sqlglot module -> tests that instantiated a class defined there (via MRO)
setup = {}      # (file, line) -> set(test file)
imp = set()
fopen = {}      # path -> set(test idx)
fopen_setup = {}
for f in sorted(glob.glob(f"{outdir}/cov-*.json")):
    d = json.load(open(f))
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

out = {
    "warm": {f: sorted(v) for f, v in warm.items()},
    "modusers": {m_: sid(v) for m_, v in modusers.items()},
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
