"""pytest plugin: per-test line coverage of /src via sys.monitoring.

Buckets:
  <nodeid>            lines a test runs outside any pooled job
  E:<nodeid>#<k>      lines run by the k-th job a test submits to its process pool (and the
                      subtest that consumes that job's result)
  SETUP:<file>        collection, class setup and teardown of a test file
  IMPORT              module import
Process pools run in-process (pickle round trip kept) and lazily, so a job's work is
attributed to it. The executor's shared query cache is disabled so every test pays for its
own transpilation.
"""
import sys, os, json, builtins, io, importlib, pkgutil, pickle, unittest, contextlib
import concurrent.futures as cf
import concurrent.futures.process as cfp
from collections import namedtuple
import pytest

mon = sys.monitoring
TOOL = 3
SRC = "/src/"
cur = "IMPORT"
test_id = None
hits = {"IMPORT": set()}
opens = {}
code_ok = {}
fixture_pos = {}   # nodeid -> list of (fixture, position) per submitted job
yields = {}        # nodeid -> list of (fixture, position) yielded by load_sql_fixture_pairs, in order
last_yield = [None]
users = {}         # bucket -> modules of instantiated dialect machinery


def _line(code, line):
    f = code.co_filename
    ok = code_ok.get(f)
    if ok is None:
        ok = code_ok[f] = f.startswith(SRC)
    if ok:
        hits[cur].add((f[len(SRC):], line))
    return mon.DISABLE


_orig_open = builtins.open


def _open(file, *a, **k):
    if isinstance(file, (str, bytes, os.PathLike)):
        p = os.path.abspath(os.fsdecode(file))
        if p.startswith(SRC):
            opens.setdefault(cur, set()).add(p[len(SRC):])
    return _orig_open(file, *a, **k)


builtins.open = _open
io.open = _open


def setcur(c):
    global cur
    cur = c
    hits.setdefault(c, set())
    mon.restart_events()


class LazyFuture(cf.Future):
    def __init__(self, fn, args, kwargs, key):
        super().__init__()
        self._job = (fn, args, kwargs)
        self._key = key
        self._done = False

    def _run(self):
        if self._done:
            return
        self._done = True
        fn, args, kwargs = self._job
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            r = fn(*args, **kwargs)
            self.set_result(pickle.loads(pickle.dumps(r)))
        except BaseException as e:
            try:
                e = pickle.loads(pickle.dumps(e))
            except Exception:
                pass
            self.set_exception(e)

    def result(self, timeout=None):
        if self._key is not None and test_id is not None:
            setcur(self._key)
        self._run()
        return super().result()

    def exception(self, timeout=None):
        self._run()
        return super().exception()


class SyncPool:
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        if initializer is not None:
            fn, args = pickle.loads(pickle.dumps((initializer, initargs)))
            fn(*args)

    def submit(self, fn, *args, **kwargs):
        key = None
        if test_id is not None:
            lst = fixture_pos.setdefault(test_id, [])
            key = f"E:{test_id}#{len(lst)}"
            lst.append(last_yield[0])
        return LazyFuture(fn, args, kwargs, key)

    def map(self, fn, *its, **kw):
        return [self.submit(fn, *a).result() for a in zip(*its)]

    def shutdown(self, wait=True, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _wait(fs, timeout=None, return_when=None):
    return namedtuple("DoneAndNotDone", "done not_done")(set(fs), set())


def _as_completed(fs, timeout=None):
    return iter(list(fs))


cf.ProcessPoolExecutor = SyncPool
cfp.ProcessPoolExecutor = SyncPool
cf.wait = _wait
cf.as_completed = _as_completed

_orig_subtest = unittest.TestCase.subTest


@contextlib.contextmanager
def _subtest(self, *a, **k):
    with _orig_subtest(self, *a, **k):
        try:
            yield
        finally:
            if test_id is not None and cur != test_id:
                setcur(test_id)


unittest.TestCase.subTest = _subtest

mon.use_tool_id(TOOL, "brcov")
mon.register_callback(TOOL, mon.events.LINE, _line)
mon.set_events(TOOL, mon.events.LINE)


attr_reads = {}   # bucket -> set of "module:qualname.attr"
FAMILY = ("sqlglot.dialects.", "sqlglot.parsers.", "sqlglot.generators.", "sqlglot.tokenizers.", "sqlglot.typing.")


class _Rec:
    """Stands in for a plain class attribute and records who reads it."""
    __slots__ = ("value", "key")

    def __init__(self, value, key):
        self.value = value
        self.key = key

    def __get__(self, obj, owner=None):
        attr_reads.setdefault(test_id or cur, set()).add(self.key)
        return self.value


def instrument_class_attrs():
    """Wrap data attributes of classes defined in dialect-family modules (after import, so
    class creation already consumed them)."""
    import types, enum
    n = 0
    for name, mod in list(sys.modules.items()):
        if not name.startswith(FAMILY) or mod is None:
            continue
        stack = [v for v in vars(mod).values() if isinstance(v, type) and v.__module__ == name]
        seen = set()
        while stack:
            cls = stack.pop()
            if id(cls) in seen:
                continue
            seen.add(id(cls))
            for k, v in list(cls.__dict__.items()):
                if isinstance(v, type):
                    if v.__module__ == name:
                        stack.append(v)
                    continue
                if k.startswith("__") or callable(v) or isinstance(v, (staticmethod, classmethod, property, types.MemberDescriptorType, types.GetSetDescriptorType)):
                    continue
                if isinstance(v, enum.Enum) and False:
                    continue
                try:
                    setattr(cls, k, _Rec(v, f"{name}:{cls.__qualname__}.{k}"))
                    n += 1
                except (TypeError, AttributeError):
                    pass
    print("instrumented", n, "class attributes", file=sys.stderr)


def pytest_sessionstart(session):
    import sqlglot
    for m in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        try:
            importlib.import_module(m.name)
        except Exception as e:
            print("import fail", m.name, e, file=sys.stderr)
    # record, per test, the modules of every Dialect/Tokenizer/Parser/Generator class instantiated
    from sqlglot.dialects.dialect import Dialect
    from sqlglot.tokens import Tokenizer
    from sqlglot.parser import Parser
    from sqlglot.generator import Generator

    def wrap(base):
        orig_init = base.__init__

        def __init__(self, *a, **k):
            key = test_id or cur
            ms = users.setdefault(key, set())
            for c in type(self).__mro__:
                ms.add(c.__module__)
            return orig_init(self, *a, **k)

        base.__init__ = __init__

    for b in (Dialect, Tokenizer, Parser, Generator):
        wrap(b)
    instrument_class_attrs()
    import tests.helpers as h
    orig_pairs = h.load_sql_fixture_pairs

    def pairs(filename):
        for i, item in enumerate(orig_pairs(filename)):
            last_yield[0] = (filename, i)
            if test_id is not None:
                yields.setdefault(test_id, []).append((filename, i))
            yield item
        last_yield[0] = None

    h.load_sql_fixture_pairs = pairs


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    if isinstance(collector, pytest.Module):
        setcur("SETUP:" + str(collector.path)[len(SRC):])
    yield
    setcur("IMPORT")
    # the executor shares query results between tests through a class-level cache: disable it
    mod = getattr(collector, "obj", None) if isinstance(collector, pytest.Module) else None


class NoCache(dict):
    def __contains__(self, k):
        return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    global test_id
    test_id = None
    setcur("SETUP:" + item.nodeid.split("::")[0])
    yield
    cls = getattr(item, "cls", None)
    if cls is not None and isinstance(getattr(cls, "cache", None), dict) and not isinstance(cls.cache, NoCache):
        cls.cache = NoCache()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    global test_id
    test_id = item.nodeid
    last_yield[0] = None
    setcur(item.nodeid)
    yield
    test_id = None
    setcur("SETUP:" + item.nodeid.split("::")[0])


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    setcur("SETUP:" + item.nodeid.split("::")[0])
    yield
    setcur("IMPORT")


def pytest_sessionfinish(session):
    wid = os.environ.get("PYTEST_XDIST_WORKER", "main")
    out = {"hits": {k: sorted(v) for k, v in hits.items()},
           "opens": {k: sorted(v) for k, v in opens.items()},
           "jobs": fixture_pos, "yields": yields, "users": {k: sorted(v) for k, v in users.items()},
           "attr_reads": {k: sorted(v) for k, v in attr_reads.items()}}
    with _orig_open(f"/out/cov-{wid}.json", "w") as f:
        json.dump(out, f)
