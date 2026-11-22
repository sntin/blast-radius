"""Synthetic edits of dialect-family class attributes (flags, settings, mapping entries); local validation only."""
import ast, difflib, os, random, sys

root, outdir, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rng = random.Random(seed)
os.makedirs(outdir, exist_ok=True)


def udiff(rel, a, b):
    d = difflib.unified_diff(a.splitlines(True), b.splitlines(True), f"a/{rel}", f"b/{rel}", n=3)
    return f"diff --git a/{rel} b/{rel}\nindex 0000000..1111111 100644\n" + "".join(d)


cands = []
for sub in ("dialects", "tokenizers", "parsers", "generators"):
    d = os.path.join(root, "sqlglot", sub)
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if not f.endswith(".py") or f in ("__init__.py", "dialect.py"):
            continue
        rel = f"sqlglot/{sub}/{f}"
        tree = ast.parse(open(os.path.join(root, rel)).read())
        stack = [c for c in tree.body if isinstance(c, ast.ClassDef)]
        while stack:
            cls = stack.pop()
            for st in cls.body:
                if isinstance(st, ast.ClassDef):
                    stack.append(st)
                elif isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) and st.targets[0].id.isupper():
                    for c in ast.walk(st.value):
                        if isinstance(c, ast.Constant) and c.lineno == c.end_lineno and (
                            (isinstance(c.value, int) and not isinstance(c.value, bool)) or (isinstance(c.value, str) and 1 <= len(c.value) <= 25)
                        ):
                            cands.append((rel, c))
                    if isinstance(st.value, (ast.Dict, ast.Set)):
                        for e in (st.value.keys if isinstance(st.value, ast.Dict) else st.value.elts):
                            if e is not None and e.lineno == e.end_lineno:
                                cands.append((rel, ("del", e)))
rng.shuffle(cands)
done, used = 0, set()
for rel, c in cands:
    if done >= n:
        break
    if not isinstance(c, tuple) and (rel, c.lineno) in used:
        continue
    src = open(os.path.join(root, rel)).read()
    lines = src.split("\n")
    if isinstance(c, tuple):
        e = c[1]
        if (rel, e.lineno) in used:
            continue
        L = lines[e.lineno - 1]
        if L.strip().endswith(",") and L.lstrip().startswith(L[e.col_offset:e.col_offset+1]) and L.count(":") <= 2:
            del lines[e.lineno - 1]
        else:
            continue
        cid = f"m5x{done:03d}"
        open(os.path.join(outdir, cid + ".patch"), "w").write(udiff(rel, src, "\n".join(lines)))
        used.add((rel, e.lineno))
        done += 1
        continue
    L = lines[c.lineno - 1]
    seg = L[c.col_offset:c.end_col_offset]
    if isinstance(c.value, int):
        if not seg.isdigit():
            continue
        rep = str(c.value + 1)
    else:
        if len(seg) < 3 or seg[0] not in "'\"" or seg[-1] != seg[0] or "\\" in seg:
            continue
        rep = seg[:-1] + "_X" + seg[-1]
    lines[c.lineno - 1] = L[:c.col_offset] + rep + L[c.end_col_offset:]
    cid = f"m5x{done:03d}"
    open(os.path.join(outdir, cid + ".patch"), "w").write(udiff(rel, src, "\n".join(lines)))
    used.add((rel, c.lineno))
    done += 1
print(done, "patches")
