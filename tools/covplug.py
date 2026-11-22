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


missing_args = {}  # bucket -> "Class|arg" for arg_types keys missing when an instance was validated
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


attr_known = set()  # class attributes wrapped by instrument_class_attrs (absent from reads = never read)
key_lookups = {}   # bucket -> set of "attrkey|normalized key" looked up in a dict class attribute
key_star = set()   # dict class attributes iterated, copied, sized or changed somewhere (not key-safe)
_kd_cache = {}     # id(original dict) -> _KDict standing in for it
_kquiet = [False]


def wrap_error_messages():
    """Record which arg_types keys were missing on each validated expression: a change to a
    key's required flag only changes the errors of instances missing that key."""
    from sqlglot.expressions.core import Expression as Expr
    orig = Expr.error_messages

    def raw_arg_types(cls):
        for c in cls.__mro__:
            if "arg_types" in c.__dict__:
                v = c.__dict__["arg_types"]
                return getattr(v, "value", v) if isinstance(v, (_Rec, _ORec)) else v
        return {}

    def error_messages(self, args=None):
        cls = type(self)
        s = missing_args.setdefault(test_id or cur, set())
        a = self.args
        for k in raw_arg_types(cls):
            v = a.get(k)
            if v is None or (isinstance(v, list) and not v):
                s.add(f"{cls.__name__}|{k}")
        return orig(self, args)

    Expr.error_messages = error_messages


def wrap_dispatch():
    """Generator dispatch tables are copies of TRANSFORMS cached per class: record lookups on
    them as lookups on the TRANSFORMS they were built from."""
    import sqlglot.generator as g
    orig = g._build_dispatch

    def build(cls):
        _kquiet[0] = True
        try:
            d = orig(cls)
        finally:
            _kquiet[0] = False
        keys = []
        for c in cls.__mro__:
            if "TRANSFORMS" in c.__dict__:
                keys.append(f"{c.__module__}:{c.__qualname__}.TRANSFORMS")
                break
        kd = _KDict(d)
        kd._bk = keys
        return kd

    g._build_dispatch = build
    g._DISPATCH_CACHE.clear()


def _nk(k):
    import enum
    if isinstance(k, str):
        return "s:" + k
    if isinstance(k, type):
        return "c:" + k.__name__
    if isinstance(k, enum.Enum):
        return f"e:{type(k).__name__}.{k.name}"
    try:
        return "r:" + repr(k)[:200]
    except Exception:
        return "r:?"


class _KDict(dict):
    """A dict class attribute that records which keys each test looks up (and whether anything
    walks the whole dict, which makes key-level selection unsafe for it)."""

    __slots__ = ("_bk",)

    def _rk(self, k):
        s = key_lookups.setdefault(test_id or cur, set())
        n = _nk(k)
        for b in self._bk:
            s.add(b + "|" + n)

    def _rq(self):
        s = key_lookups.setdefault(test_id or cur, set())
        for b in self._bk:
            s.add(b + "|?")

    def _star(self):
        if _kquiet[0]:
            return
        # walked by test code ("*t": that test sees every entry) or by library code ("*": it
        # may build a cache other tests use, so key-level selection is off for this dict)
        f = sys._getframe(2)
        tag = "*t" if f is not None and f.f_code.co_filename.startswith(SRC + "tests/") else "*"
        s = key_lookups.setdefault(test_id or cur, set())
        for b in self._bk:
            s.add(b + "|" + tag)

    def __getitem__(self, k):
        self._rk(k)
        return dict.__getitem__(self, k)

    def get(self, k, *d):
        self._rk(k)
        return dict.get(self, k, *d)

    def __contains__(self, k):
        self._rk(k)
        return dict.__contains__(self, k)

    def setdefault(self, k, *d):
        self._star()
        return dict.setdefault(self, k, *d)

    def pop(self, k, *d):
        self._star()
        return dict.pop(self, k, *d)

    def __iter__(self):
        self._star()
        return dict.__iter__(self)

    def keys(self):
        self._star()
        return dict.keys(self)

    def values(self):
        self._star()
        return dict.values(self)

    def items(self):
        self._star()
        return dict.items(self)

    def copy(self):
        self._star()
        return dict.copy(self)

    def __len__(self):
        self._star()
        return dict.__len__(self)

    def __bool__(self):
        # a truth test only asks whether it is empty
        self._rq()
        return dict.__len__(self) > 0

    def __eq__(self, o):
        self._star()
        return dict.__eq__(self, o)

    def __ne__(self, o):
        self._star()
        return dict.__ne__(self, o)

    def __or__(self, o):
        self._star()
        return dict.__or__(self, o)

    def __ror__(self, o):
        self._star()
        return dict.__ror__(self, o)

    def __setitem__(self, k, v):
        self._star()
        return dict.__setitem__(self, k, v)

    def __delitem__(self, k):
        self._star()
        return dict.__delitem__(self, k)

    def update(self, *a, **k):
        self._star()
        return dict.update(self, *a, **k)

    def popitem(self):
        self._star()
        return dict.popitem(self)

    def clear(self):
        self._star()
        return dict.clear(self)

    def __reversed__(self):
        self._star()
        return dict.__reversed__(self)

    def __reduce_ex__(self, p):
        self._star()
        return (dict, (dict(dict.items(self)),))

    __hash__ = None


def _kset_cls(base):
    class _KS(base):
        """A set class attribute that records which elements each test asks about."""

        __slots__ = ("_bk",)

        _rk = _KDict._rk
        _rq = _KDict._rq
        _star = _KDict._star

        def __bool__(self):
            self._rq()
            return base.__len__(self) > 0

        def __contains__(self, k):
            self._rk(k)
            return base.__contains__(self, k)

    def star_method(name):
        orig = getattr(base, name)

        def f(self, *a, **k):
            self._star()
            return orig(self, *a, **k)
        f.__name__ = name
        return f

    for name in ("__iter__", "__len__", "__eq__", "__ne__", "__or__", "__ror__", "__and__", "__rand__",
                 "__sub__", "__rsub__", "__xor__", "__rxor__", "__le__", "__lt__", "__ge__", "__gt__",
                 "union", "intersection", "difference", "symmetric_difference", "issubset", "issuperset",
                 "isdisjoint", "copy", "__reduce__", "__reduce_ex__"):
        if hasattr(base, name):
            setattr(_KS, name, star_method(name))
    if base is set:
        for name in ("add", "discard", "remove", "pop", "clear", "update", "__ior__", "__iand__", "__isub__", "__ixor__",
                     "intersection_update", "difference_update", "symmetric_difference_update"):
            setattr(_KS, name, star_method(name))
    _KS.__hash__ = base.__hash__
    return _KS


_KSet = _kset_cls(set)
_KFSet = _kset_cls(frozenset)


def _kdict(v, key):
    if type(v) in (set, frozenset):
        kd = _kd_cache.get(id(v))
        if kd is None:
            kd = (_KSet if type(v) is set else _KFSet)(v)
            kd._bk = [key]
            _kd_cache[id(v)] = kd
            _kd_cache[id(kd)] = kd
        elif key not in kd._bk:
            kd._bk.append(key)
        return kd
    kd = _kd_cache.get(id(v))
    if kd is None:
        kd = _KDict(v)
        kd._bk = [key]
        _kd_cache[id(v)] = kd
        _kd_cache[id(kd)] = kd
    elif key not in kd._bk:
        kd._bk.append(key)
    return kd


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
                    if type(v) in (dict, set, frozenset) and os.environ.get("BR_KEYS", "1") == "1":
                        v = _kdict(v, f"{name}:{cls.__qualname__}.{k}")
                    setattr(cls, k, _Rec(v, f"{name}:{cls.__qualname__}.{k}"))
                    attr_known.add(f"{name}:{cls.__qualname__}.{k}")
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
    if os.environ.get("BR_KEYS", "1") == "1":
        wrap_dispatch()
    wrap_error_messages()
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
           "key_lookups": {k: sorted(v) for k, v in key_lookups.items()}, "key_star": sorted(key_star),
           "attr_known": sorted(attr_known),
           "missing_args": {k: sorted(v) for k, v in missing_args.items()},
           "dial_sites": {k: sorted(v) for k, v in dial_sites.items()}}
    with _orig_open(f"/out/cov-{wid}.json", "w") as f:
        json.dump(out, f)
