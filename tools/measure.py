"""Run every test through br.runner on the unpatched tree, one forked child per test file; print cpu per test."""
import json, os, sys, time
sys.path.insert(0, "/app/br")
import runner
root = sys.argv[1]
tests = json.load(open(sys.argv[2]))["tests"]
sys.path.insert(0, root)
import pandas, numpy, duckdb  # noqa
byf = {}
for t in tests: byf.setdefault(t["file"], []).append((t["number"], t["id"]))
res = {}
t0=time.time()
for f, items in byf.items():
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        c0=time.process_time()
        try:
            runner.run(root, {}, {}, items, w, {})
        finally:
            os.write(w, f"X {time.process_time()-c0:.3f}\n".encode())
            os._exit(0)
    os.close(w)
    data = b""
    while True:
        b = os.read(r, 65536)
        if not b: break
        data += b
    os.waitpid(pid, 0)
    for line in data.decode().splitlines():
        p = line.split()
        if p[0] == "X": print(f, "child cpu", p[1], file=sys.stderr)
        elif p[0] == "I": print("IMPORTFAIL", f, file=sys.stderr)
        else: res.setdefault(p[0], []).append((p[1], float(p[2])))
json.dump(res, open("/out/measure.json", "w"))
print("wall", time.time()-t0, file=sys.stderr)
