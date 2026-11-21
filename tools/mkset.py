"""Turn labelled synthetic patches into a sample-like directory (commits/, tests.json, answers.json)."""
import json, glob, os, shutil, sys
pdir, ldir, out, tests_json = sys.argv[1:5]
tests = json.load(open(tests_json))["tests"]
num = {t["id"]: t["number"] for t in tests}
byfile = {}
for t in tests: byfile.setdefault(t["file"], []).append(t["number"])
os.makedirs(out + "/commits", exist_ok=True)
shutil.copy(tests_json, out + "/tests.json")
ans = {}
for d in sorted(glob.glob(ldir + "/*.done")):
    cid = os.path.basename(d)[:-5]
    fails = set(); bad = False
    for f in glob.glob(f"{ldir}/{cid}.*"):
        if f.endswith((".done", ".log")): continue
        r = json.load(open(f))
        for nid in r["failed"]:
            nid = nid.split(" ")[0]
            if nid in num: fails.add(num[nid])
            else: print("unknown", cid, nid, file=sys.stderr)
        for c in r["colfail"]:
            for n in byfile.get(c, []): fails.add(n)
            if c not in byfile: print("colfail", cid, c, file=sys.stderr)
    log = open(f"{ldir}/{cid}.log").read()
    if "error" in log.splitlines()[-1] and "failed" not in log.splitlines()[-1]:
        print("check", cid, log.splitlines()[-1], file=sys.stderr)
    shutil.copy(f"{pdir}/{cid}.patch", f"{out}/commits/{cid}.patch")
    ans[cid] = {"failing": sorted(fails), "kind": "synthetic"}
json.dump({"commits": ans}, open(out + "/answers.json", "w"))
print(len(ans), "commits,", sum(1 for v in ans.values() if v["failing"]), "breaking")
