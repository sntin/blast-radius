import json, os, sys
sys.path.insert(0, "/app/br")
import main as M
tests = json.load(open("/s/tests.json"))["tests"]
M.prepare_child_env()
import runner
want = sys.argv[1:]
for t in tests:
    if t["id"].split("::")[-1] not in want: continue
    r, w = os.pipe(); pid = os.fork()
    if pid == 0:
        os.close(r); runner.run(M.ROOT, {}, {}, [(t["number"], t["id"])], w, {t["number"]: 300.0}, M.LAZY_SETUP); os._exit(0)
    os.close(w); data = b""
    while True:
        b = os.read(r, 65536)
        if not b: break
        data += b
    _, _, ru = os.wait4(pid, 0)
    print(t["number"], t["id"], "iso", round(ru.ru_utime + ru.ru_stime, 3), data.decode().strip(), flush=True)
