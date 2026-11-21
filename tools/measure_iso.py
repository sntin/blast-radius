"""Marginal CPU of running each test alone in a forked child of a warm parent (as br/main.py does)."""
import json, os, sys, time
sys.path.insert(0, "/app/br")
os.environ["BR_SQLGLOT"] = "/opt/sqlglot"
import main as M
M.ROOT = "/opt/sqlglot"
tests = json.load(open(sys.argv[1]))["tests"]
M.prepare_child_env()
import runner
res = {}
for t in tests:
    if t["id"] in ("tests/test_executor.py::TestExecutor::test_execute_tpcds", "tests/test_parser.py::TestParser::test_max_nodes"):
        continue
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        runner.run("/opt/sqlglot", {}, {}, [(t["number"], t["id"])], w, {t["number"]: 60.0}, M.LAZY_SETUP)
        os._exit(0)
    os.close(w)
    data = b""
    while True:
        b = os.read(r, 65536)
        if not b: break
        data += b
    _, _, ru = os.wait4(pid, 0)
    p = data.decode().split()
    res[t["number"]] = {"iso": round(ru.ru_utime + ru.ru_stime, 4), "cpu": float(p[2]) if len(p) > 2 else None, "st": p[1] if len(p) > 1 else "?"}
json.dump(res, open("/out/iso.json", "w"))
