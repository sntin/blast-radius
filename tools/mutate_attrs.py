"""Synthetic edits of class-level specs: expression arg_types and dialect class attributes (local validation only)."""
import ast, difflib, os, random, sys

root, outdir, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rng = random.Random(seed)
os.makedirs(outdir, exist_ok=True)


def udiff(rel, a, b):
    d = difflib.unified_diff(a.splitlines(True), b.splitlines(True), f"a/{rel}", f"b/{rel}", n=3)
    return f"diff --git a/{rel} b/{rel}\nindex 0000000..1111111 100644\n" + "".join(d)


cands = []
for dp, dn, fn in os.walk(os.path.join(root, "sqlglot/expressions")):
    for f in sorted(fn):
        if not f.endswith(".py"):
            continue
        rel = os.path.relpath(os.path.join(dp, f), root)
        src = open(os.path.join(root, rel)).read()
        tree = ast.parse(src)
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef):
                continue
            for st in cls.body:
                if isinstance(st, (ast.Assign, ast.AnnAssign)):
                    tg = st.targets[0] if isinstance(st, ast.Assign) else st.target
                    if isinstance(tg, ast.Name) and tg.id == "arg_types" and isinstance(st.value, ast.Dict):
                        for k, v in zip(st.value.keys, st.value.values):
                            if isinstance(k, ast.Constant) and isinstance(v, ast.Constant) and k.lineno == v.lineno:
                                cands.append((rel, k, v))
rng.shuffle(cands)
done = 0
for rel, k, v in cands:
    if done >= n:
        break
    src = open(os.path.join(root, rel)).read()
    lines = src.split("\n")
    L = lines[k.lineno - 1]
    kind = rng.choice(["flip", "flip", "rename"])
    if kind == "flip":
        seg = L[v.col_offset:v.end_col_offset]
        if seg not in ("True", "False"):
            continue
        L2 = L[:v.col_offset] + ("False" if seg == "True" else "True") + L[v.end_col_offset:]
    else:
        seg = L[k.col_offset:k.end_col_offset]
        L2 = L[:k.col_offset] + seg[:-1] + "_x" + seg[-1] + L[k.end_col_offset:]
    lines[k.lineno - 1] = L2
    new = "\n".join(lines)
    cid = f"m9x{done:03d}"
    open(os.path.join(outdir, cid + ".patch"), "w").write(udiff(rel, src, new))
    done += 1
print(done, "patches")
