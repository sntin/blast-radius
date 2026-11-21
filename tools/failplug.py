"""pytest plugin: write the node ids of every test that fails (incl. subtests, setup errors, import errors)."""
import json, os
failed = set(); colfail = set()
def pytest_runtest_logreport(report):
    if report.failed:
        failed.add(report.nodeid.split(" ")[0].split("[")[0] if "::" in report.nodeid else report.nodeid)
def pytest_collectreport(report):
    if report.failed:
        colfail.add(report.nodeid)
def pytest_sessionfinish(session):
    wid = os.environ.get("PYTEST_XDIST_WORKER", "main")
    with open(os.environ["FAILOUT"] + f".{wid}", "w") as f:
        json.dump({"failed": sorted(failed), "colfail": sorted(colfail)}, f)
