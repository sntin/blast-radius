import json, os, sys, time, cProfile, pstats
sys.path.insert(0, "/app/br")
import main as M
cid = sys.argv[1]
tests = json.load(open("/data/tests.json"))["tests"]
maps = M.Maps(tests); src = M.Source(M.ROOT); pl = M.Planner(maps, src)
pf = M.parse_patch(open(f"/data/commits/{cid}.patch").read())
files = []; overrides = {}; redirects = {}
for f in pf:
    old = src.read(f["old"]); new, ch, ins = M.apply_hunks(old, f["hunks"])
    files.append((f["old"], f["new"], old, new, ch, ins))
    if f["old"].endswith(".py"): overrides[f["old"]] = new
    else: redirects[os.path.join(M.ROOT, f["old"])] = new
c, why = pl.plan(files)
M.prepare_child_env()
import runner
purge = sys.argv[2] == "1"
if purge:
    for k in [k for k in sys.modules if k.startswith("sqlglot") or k.startswith("tests")]: del sys.modules[k]
r, w = os.pipe()
t = time.process_time()
cProfile.run("runner.run(M.ROOT, overrides, redirects, [(n, maps.id_of[n]) for n in sorted(c)], w, {})", "/tmp/p")
print("cpu", time.process_time() - t, [maps.id_of[n] for n in c])
print(os.read(r, 10000).decode())
pstats.Stats("/tmp/p").sort_stats("cumtime").print_stats(25)
