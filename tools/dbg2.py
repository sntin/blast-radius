import json, os, sys
sys.path.insert(0, "/app/br")
import main as M
tests = json.load(open("/data/tests.json"))["tests"]
maps = M.Maps(tests); src = M.Source(M.ROOT); pl = M.Planner(maps, src)
for cid in sys.argv[1:]:
    pf = M.parse_patch(open(f"/data/commits/{cid}.patch").read())
    files = []
    for f in pf:
        old = src.read(f["old"]); new, ch, ins = M.apply_hunks(old, f["hunks"])
        files.append((f["old"], f["new"], old, new, ch, ins))
    c, why = pl.plan(files)
    print(cid, len(c), why[:5])
    for t in sorted(c, key=lambda t: -maps.seconds[t])[:8]:
        cfg = pl.partial_cfg.get(t)
        print("  ", maps.id_of[t], maps.seconds[t], maps.cpu.get(t), maps.iso.get(t), "FULL" if t in pl.full else "", {k: (v if k != "fixtures" else {a: len(b) for a, b in v.items()}) for k, v in (cfg or {}).items() if k != "attr"}, (cfg or {}).get("attr", ("", []))[1][:10] if cfg else "")
    b, r = M.choose(c, maps, pl.partial_cfg)
    print("  blind", [maps.id_of[x].split("::")[-1] for x in b][:5], "run", len(r))
