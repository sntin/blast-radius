"""Runs a list of sqlglot unittest tests in this process against an in-memory patched tree.

Used in a forked child. Reports one line per finished test on the given fd:
  "<num> <status> <cpu>"  status in P (pass), F (fail), S (skip)
and "I <file>" when a test file cannot be imported.
"""
import builtins
import importlib
import importlib.abc
import importlib.util
import io
import os
import signal
import sys
import time
import unittest


class PatchedFinder(importlib.abc.MetaPathFinder):
    """Serves patched module sources from memory, keeping the original __file__."""

    def __init__(self, root, overrides):
        self.root = root
        self.mods = {}  # module name -> (path, source or None if deleted, is_pkg)
        for rel, src in overrides.items():
            if not rel.endswith(".py"):
                continue
            parts = rel[:-3].split("/")
            is_pkg = parts[-1] == "__init__"
            if is_pkg:
                parts = parts[:-1]
            self.mods[".".join(parts)] = (os.path.join(root, rel), src, is_pkg)

    def find_spec(self, fullname, path=None, target=None):
        hit = self.mods.get(fullname)
        if hit is None:
            return None
        path_, src, is_pkg = hit
        if src is None:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        loader = _MemLoader(fullname, path_, src)
        spec = importlib.util.spec_from_loader(fullname, loader, origin=path_, is_package=is_pkg)
        spec.has_location = True
        if is_pkg:
            spec.submodule_search_locations = [os.path.dirname(path_)]
        return spec


class _MemLoader(importlib.abc.Loader):
    def __init__(self, name, path, src):
        self.name, self.path, self.src = name, path, src

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = self.path
        code = compile(self.src, self.path, "exec", dont_inherit=True)
        exec(code, module.__dict__)

    def get_source(self, name):
        return self.src

    def is_package(self, name):
        return self.path.endswith("__init__.py")

    def get_filename(self, name):
        return self.path


def install_open_redirect(redirects):
    """redirects: abs original path -> replacement text (None = deleted)."""
    if not redirects:
        return
    orig_open = builtins.open

    def _open(file, mode="r", *a, **k):
        if isinstance(file, (str, bytes, os.PathLike)):
            p = os.path.abspath(os.fsdecode(file))
            if p in redirects:
                data = redirects[p]
                if data is None:
                    raise FileNotFoundError(2, "No such file or directory", p)
                if "b" in mode:
                    return io.BytesIO(data.encode("utf-8"))
                return io.StringIO(data)
        return orig_open(file, mode, *a, **k)

    builtins.open = _open
    io.open = _open


class _Timeout(BaseException):
    pass


PARTIAL_CUR = [None]


def install_partial(partial):
    """partial: test id -> {"fixtures": {fixture: [positions]}, "attr": (name, [positions]) or None}.

    Runs only the listed fixture entries of a test (the others cannot see the change)."""
    if not partial:
        return
    import tests.helpers as h

    orig = h.load_sql_fixture_pairs

    def pairs(filename):
        cfg = PARTIAL_CUR[0]
        keep = None
        if cfg is not None and cfg.get("fixtures") is not None:
            keep = set(cfg["fixtures"].get(filename, ()))
        for i, item in enumerate(orig(filename)):
            if keep is None or i in keep:
                yield item

    h.load_sql_fixture_pairs = pairs


class Result(unittest.TestResult):
    def __init__(self, nums, out, limits, partial=None):
        super().__init__()
        self.nums = nums
        self.out = out
        self.limits = limits
        self.bad = False
        self.t0 = 0.0
        self.partial = partial or {}

    def startTest(self, test):
        super().startTest(test)
        self.bad = False
        self.skipped_ = False
        self.t0 = time.process_time()
        cfg = self.partial.get(test.id())
        PARTIAL_CUR[0] = cfg
        if cfg is not None and cfg.get("attr"):
            name, keep = cfg["attr"]
            full = getattr(type(test), name)
            setattr(test, name, [full[k] for k in keep if k < len(full)])
        lim = self.limits.get(test.id(), 10.0)
        signal.setitimer(signal.ITIMER_PROF, lim)

    def stopTest(self, test):
        signal.setitimer(signal.ITIMER_PROF, 0)
        PARTIAL_CUR[0] = None
        super().stopTest(test)
        cpu = time.process_time() - self.t0
        n = self.nums.get(test.id())
        if n is not None:
            st = "F" if self.bad else ("S" if self.skipped_ else "P")
            os.write(self.out, f"{n} {st} {cpu:.4f}\n".encode())

    def addError(self, test, err):
        self.bad = True

    def addFailure(self, test, err):
        self.bad = True

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self.bad = True

    def addUnexpectedSuccess(self, test):
        self.bad = True

    def addSkip(self, test, reason):
        self.skipped_ = True


LAZY_SETUP = {"tests/test_executor.py::TestExecutor"}


def make_lazy_setup(klass):
    """Defer setUpClass until a test touches one of the attributes it assigns.

    Only used when the patch cannot change what setUpClass does (checked by the caller)."""
    import inspect
    import re

    func = klass.__dict__.get("setUpClass")
    if func is None:
        return
    real = func.__func__
    try:
        names = set(re.findall(r"cls\.(\w+)\s*=", inspect.getsource(real)))
    except (OSError, TypeError):
        return
    teardown = klass.__dict__.get("tearDownClass")
    state = {"loaded": False}

    def load(owner):
        if state["loaded"]:
            return
        state["loaded"] = True
        for n in names:
            if isinstance(klass.__dict__.get(n), _Lazy):
                delattr(klass, n)
        real(klass)

    class _Lazy:
        def __init__(self, name):
            self.name = name

        def __get__(self, obj, owner=None):
            load(klass)
            return getattr(klass, self.name)

    for n in names:
        setattr(klass, n, _Lazy(n))
    klass.setUpClass = classmethod(lambda cls: None)
    if teardown is not None:
        real_td = teardown.__func__
        klass.tearDownClass = classmethod(lambda cls: real_td(cls) if state["loaded"] else None)


def _alarm(signum, frame):
    raise _Timeout()


def check_imports(root, overrides, redirects, mods, out_fd):
    """Import the given modules against the patched tree; report 'M ok|fail <module>'."""
    sys.dont_write_bytecode = True
    install_open_redirect(redirects)
    sys.meta_path.insert(0, PatchedFinder(root, overrides))
    signal.signal(signal.SIGPROF, _alarm)
    for m in mods:
        try:
            signal.setitimer(signal.ITIMER_PROF, 20.0)
            importlib.import_module(m)
            signal.setitimer(signal.ITIMER_PROF, 0)
            os.write(out_fd, f"M ok {m}\n".encode())
        except BaseException:
            signal.setitimer(signal.ITIMER_PROF, 0)
            os.write(out_fd, f"M fail {m}\n".encode())


def run(root, overrides, redirects, tests, out_fd, limits, lazy=(), partial=None):
    """tests: list of (num, 'tests/x.py::Class::method')."""
    sys.dont_write_bytecode = True
    os.environ["SKIP_INTEGRATION"] = "0"
    install_open_redirect(redirects)
    sys.meta_path.insert(0, PatchedFinder(root, overrides))
    import syncpool

    syncpool.install()
    try:
        import main as _main

        _main._single_thread_duckdb()
    except Exception:
        pass
    signal.signal(signal.SIGPROF, _alarm)
    partial = partial or {}
    if partial:
        try:
            install_partial(partial)
        except BaseException:
            partial = {}

    by_file = {}
    for num, tid in tests:
        f, cls, meth = tid.split("::")
        by_file.setdefault(f, []).append((num, cls, meth, tid))

    for f, items in by_file.items():
        modname = f[:-3].replace("/", ".")
        try:
            signal.setitimer(signal.ITIMER_PROF, 20.0)
            mod = importlib.import_module(modname)
            signal.setitimer(signal.ITIMER_PROF, 0)
        except BaseException:
            signal.setitimer(signal.ITIMER_PROF, 0)
            os.write(out_fd, f"I {f}\n".encode())
            continue
        by_cls = {}
        for num, cls, meth, tid in items:
            by_cls.setdefault(cls, []).append((num, meth))
        for cls, ms in by_cls.items():
            suite = unittest.TestSuite()
            nums = {}
            lim = {}
            part = {}
            klass = getattr(mod, cls, None)
            if klass is not None and f"{f}::{cls}" in lazy:
                make_lazy_setup(klass)
            if klass is None:
                for num, meth in ms:
                    os.write(out_fd, f"{num} F 0\n".encode())
                continue
            for num, meth in ms:
                try:
                    tc = klass(meth)
                except BaseException:
                    os.write(out_fd, f"{num} F 0\n".encode())
                    continue
                suite.addTest(tc)
                nums[tc.id()] = num
                lim[tc.id()] = limits.get(num, 10.0)
                if num in partial:
                    part[tc.id()] = partial[num]
            res = Result(nums, out_fd, lim, part)
            try:
                suite.run(res)
            except _Timeout:
                pass
            # anything not reported (class setup failure etc.) is reported failing
            for tid_, num in nums.items():
                pass
            if res.errors and not res.testsRun:
                for num, meth in ms:
                    os.write(out_fd, f"{num} F 0\n".encode())
            # class-level setup errors: unittest reports them via addError with a _ErrorHolder
            for t_, _ in res.errors:
                if isinstance(t_, unittest.suite._ErrorHolder):
                    for num, meth in ms:
                        os.write(out_fd, f"{num} F 0\n".encode())
                    break
