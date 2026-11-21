import json, os, sys, time
sys.path.insert(0, "/app/br")
import main as M
tests = json.load(open("/data/tests.json"))["tests"]
M.prepare_child_env(sys.argv[2] == "1")
import runner
ids = {t["id"].split("::")[-1]: (t["number"], t["id"]) for t in tests}
r, w = os.pipe(); pid = os.fork()
if pid == 0:
    runner.run(M.ROOT, {}, {}, [ids[sys.argv[1]]], w, {}, M.LAZY_SETUP); os._exit(0)
os.close(w); print(os.read(r, 1000)); _, _, ru = os.wait4(pid, 0); print("iso", ru.ru_utime + ru.ru_stime)
