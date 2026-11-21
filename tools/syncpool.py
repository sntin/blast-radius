"""Synchronous stand-in for ProcessPoolExecutor that keeps the pickle round trip."""
import pickle
from concurrent.futures import Future

class SyncPool:
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        if initializer is not None:
            fn, args = pickle.loads(pickle.dumps((initializer, initargs)))
            fn(*args)
    def submit(self, fn, *args, **kwargs):
        f = Future()
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            r = fn(*args, **kwargs)
            f.set_result(pickle.loads(pickle.dumps(r)))
        except BaseException as e:
            try:
                e = pickle.loads(pickle.dumps(e))
            except Exception:
                pass
            f.set_exception(e)
        return f
    def map(self, fn, *its, **kw):
        return [self.submit(fn, *a).result() for a in zip(*its)]
    def shutdown(self, wait=True, **kw):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False

def install():
    import concurrent.futures as cf, concurrent.futures.process as cfp
    cf.ProcessPoolExecutor = SyncPool
    cfp.ProcessPoolExecutor = SyncPool
