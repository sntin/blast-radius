"""Offline: plan every sample commit and compare candidates with answers (no running)."""
import json, os, sys, time
sys.path.insert(0, sys.argv[1] + "/br")
import main as M
S = sys.argv[2]
tests = json.load(open(S + "/tests.json"))["tests"]
ans = json.load(open(S + "/answers.json"))["commits"]
maps = M.Maps(tests); src = M.Source(M.ROOT); pl = M.Planner(maps, src)
for fn in sorted(os.listdir(S + "/commits")):
    cid = fn[:-6]
    t0 = time.process_time()
    pf = M.parse_patch(open(S + "/commits/" + fn).read())
    files = []
    for f in pf:
        old = src.read(f["old"]) if f["old"] else None
        new, ch, ins = M.apply_hunks(old, f["hunks"])
        files.append((f["old"], f["new"], old, new, ch, ins))
    c, why = pl.plan(files)
    if c == M.ALL: c = set(range(maps.n))
    fails = set(ans[cid]["failing"])
    miss = fails - c
    print(cid, ans[cid]["kind"], "fail", len(fails), "cand", len(c), round(sum(maps.seconds[x] for x in c), 1),
          "MISS" if miss else "ok", f"{time.process_time()-t0:.2f}s", ";".join(why)[:200])
