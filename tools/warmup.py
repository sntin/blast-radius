"""Find 'first execution only' lines: run each test twice in one process; lines seen only in the first run."""
import json, os, sys, unittest, importlib
sys.path.insert(0, "/app/br"); sys.path.insert(0, "/src")
os.environ["SKIP_INTEGRATION"] = "0"
import syncpool; syncpool.install()
mon = sys.monitoring; TOOL = 4; SRC = "/src/"
cur = set()
def _line(code, line):
    f = code.co_filename
    if f.startswith(SRC):
        cur.add((f[len(SRC):], line))
    return mon.DISABLE
mon.use_tool_id(TOOL, "warm"); mon.register_callback(TOOL, mon.events.LINE, _line); mon.set_events(TOOL, mon.events.LINE)
tests = json.load(open(sys.argv[1]))["tests"]
skip = {"tests/test_executor.py::TestExecutor::test_execute_tpcds", "tests/test_parser.py::TestParser::test_max_nodes",
        "tests/dialects/test_dialect.py::TestDialect::test_lazy_load", "tests/test_generator.py::TestGenerator::test_generator_memory"}
byf = {}
for t in tests: byf.setdefault(t["file"], []).append(t["id"])
warm = {}
for f, ids in byf.items():
    pid = os.fork()
    if pid == 0:
        out = {}
        mod = importlib.import_module(f[:-3].replace("/", "."))
        for tid in ids:
            if tid in skip: continue
            _, cls, meth = tid.split("::")
            k = getattr(mod, cls)
            res = unittest.TestResult()
            runs = []
            for rep in range(2):
                cur.clear(); mon.restart_events()
                s = unittest.TestSuite([k(meth)]); s.run(res)
                runs.append(set(cur))
            w = runs[0] - runs[1]
            if w: out[tid] = sorted(w)
        json.dump(out, open(f"/out/warm-{f.replace('/', '_')}.json", "w"))
        os._exit(0)
    os.waitpid(pid, 0)
    print("done", f, file=sys.stderr, flush=True)
