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
DISABLE = mon.DISABLE
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
        if cur.startswith("MOD:sqlglot") and code.co_flags & 3 == 3:
            # function code run while a sqlglot module is imported
            hits.setdefault("FX:" + cur[4:], set()).add((f[len(SRC):], line))
    return DISABLE


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

def _start(code, offset):
    # a one-line lambda gets no LINE event for its only line: count its start instead
    if code.co_name == "<lambda>":
        _line(code, code.co_firstlineno)
    return DISABLE


mon.use_tool_id(TOOL, "brcov")
mon.register_callback(TOOL, mon.events.LINE, _line)
mon.register_callback(TOOL, mon.events.PY_START, _start)
mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)


class _ModLoader:
    """Runs a test module's top-level code in its own MOD:<file> bucket."""

    def __init__(self, inner, rel):
        self.inner = inner
        self.rel = rel

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        prev = cur
        setcur("MOD:" + self.rel)
        try:
            return self.inner.exec_module(module)
        finally:
            setcur(prev)

    def __getattr__(self, k):
        return getattr(self.inner, k)


class _ModFinder:
    def find_spec(self, name, path=None, target=None):
        if not (name == "tests" or name.startswith(("tests.", "sqlglot"))):
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None:
                if spec.origin and spec.origin.startswith(SRC) and spec.loader is not None:
                    spec.loader = _ModLoader(spec.loader, spec.origin[len(SRC):])
                return spec
        return None


sys.meta_path.insert(0, _ModFinder())


attr_reads = {}   # bucket -> set of "module:qualname.attr"
dial_sites = {}   # dialect-machinery module -> "file:line" of import-time statements instantiating it
attr_sites = {}   # expression attr key -> "file:line" of import-time statements reading it
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


class _ORec:
    """Like _Rec, keyed by the class the attribute is looked up through (so a subclass that
    inherits the value is recorded under its own name)."""
    __slots__ = ("value", "attr", "cache")

    def __init__(self, value, attr):
        self.value = value
        self.attr = attr
        self.cache = {}

    def __get__(self, obj, owner=None):
        if owner is None:
            owner = type(obj)
        k = self.cache.get(owner)
        if k is None:
            k = self.cache[owner] = f"sqlglot.expressions:{owner.__qualname__}.{self.attr}"
        attr_reads.setdefault(test_id or cur, set()).add(k)
        if not test_id and cur.startswith("MOD:sqlglot"):
            # import-time read: note the module-level / class-body statement doing it
            f = sys._getframe(1)
            while f is not None:
                c = f.f_code
                if c.co_flags & 3 != 3 and c.co_filename.startswith(SRC):
                    attr_sites.setdefault(k, set()).add(f"{c.co_filename[len(SRC):]}:{f.f_lineno}")
                    break
                f = f.f_back
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


def instrument_expr_attrs():
    """Expression classes: the attributes that describe a node's arguments. Installed before
    the other sqlglot modules are imported, so reads made while they import are recorded
    under that module's MOD: bucket."""
    import sqlglot.expressions  # noqa: F401
    from sqlglot.expressions.core import Expr
    m = 0
    stack, seen = [Expr], set()
    while stack:
        cls = stack.pop()
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        stack.extend(cls.__subclasses__())
        if not str(cls.__module__).startswith("sqlglot.expressions"):
            continue
        for k in EXPR_ATTRS:
            if k in cls.__dict__ and not isinstance(cls.__dict__[k], (_Rec, _ORec)):
                try:
                    setattr(cls, k, _ORec(cls.__dict__[k], k))
                    m += 1
                except (TypeError, AttributeError):
                    pass
    print("instrumented", m, "expression attributes", file=sys.stderr)


EXPR_ATTRS = ("arg_types", "required_args", "is_var_len_args")


def install_import_recorder():
    """Record, per bucket, the sqlglot modules that test code imports (all are preloaded, so
    such an import is otherwise invisible; in CI it may be the first import)."""
    import types
    orig_import = builtins.__import__
    orig_import_module = importlib.import_module

    def note(mods):
        if cur == "IMPORT":
            return
        users.setdefault(test_id or cur, set()).update(mods)

    def _imp(name, globals=None, locals=None, fromlist=(), level=0):
        m = orig_import(name, globals, locals, fromlist, level)
        try:
            if level == 0 and name.startswith("sqlglot") and globals and str(globals.get("__name__", "")).startswith("tests"):
                mods = {name}
                for it in fromlist or ():
                    v = getattr(m, it, None)
                    if isinstance(v, types.ModuleType):
                        mods.add(v.__name__)
                    elif isinstance(v, type) and str(v.__module__).startswith("sqlglot"):
                        mods.add(v.__module__)
                note(mods)
        except Exception:
            pass
        return m

    def _imod(name, package=None):
        m = orig_import_module(name, package)
        try:
            if name.startswith("sqlglot") and str(sys._getframe(1).f_globals.get("__name__", "")).startswith("tests"):
                note({m.__name__})
        except Exception:
            pass
        return m

    builtins.__import__ = _imp
    importlib.import_module = _imod


def pytest_sessionstart(session):
    import sqlglot
    instrument_expr_attrs()
    # record, per bucket, the modules of every Dialect/Tokenizer/Parser/Generator class
    # instantiated (installed before the other modules import, so a module that parses SQL
    # while it imports is seen too)
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
            if not test_id and cur.startswith("MOD:sqlglot"):
                # instantiated while a module imports: note the statement doing it
                f = sys._getframe(1)
                while f is not None:
                    c = f.f_code
                    if c.co_flags & 3 != 3 and c.co_filename.startswith(SRC):
                        site = f"{c.co_filename[len(SRC):]}:{f.f_lineno}"
                        for m_ in {c_.__module__ for c_ in type(self).__mro__}:
                            if m_.startswith("sqlglot"):
                                dial_sites.setdefault(m_, set()).add(site)
                        break
                    f = f.f_back
            return orig_init(self, *a, **k)

        base.__init__ = __init__

    for b in (Dialect, Tokenizer, Parser, Generator):
        wrap(b)
    for m in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        try:
            importlib.import_module(m.name)
        except Exception as e:
            print("import fail", m.name, e, file=sys.stderr)
    instrument_class_attrs()
    install_import_recorder()
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
           "attr_reads": {k: sorted(v) for k, v in attr_reads.items()},
           "attr_sites": {k: sorted(v) for k, v in attr_sites.items()},
           "dial_sites": {k: sorted(v) for k, v in dial_sites.items()}}
    with _orig_open(f"/out/cov-{wid}.json", "w") as f:
        json.dump(out, f)
