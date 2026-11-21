"""Generate synthetic single-change patches against the pinned sqlglot tree (for local validation only)."""
import ast, difflib, os, random, re, sys

root, outdir, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rng = random.Random(seed)
os.makedirs(outdir, exist_ok=True)
pyfiles = []
for dp, dn, fn in os.walk(os.path.join(root, "sqlglot")):
    for f in fn:
        if f.endswith(".py"):
            pyfiles.append(os.path.relpath(os.path.join(dp, f), root))
fixtures = []
for dp, dn, fn in os.walk(os.path.join(root, "tests/fixtures")):
    for f in fn:
        if f.endswith(".sql"):
            fixtures.append(os.path.relpath(os.path.join(dp, f), root))
pyfiles.sort(); fixtures.sort()
COMMENTS = ["# Cheap check first", "# Kept for backwards compatibility", "# NOTE: order matters here", "# TODO: revisit"]

def udiff(rel, a, b):
    d = difflib.unified_diff(a.splitlines(True), b.splitlines(True), f"a/{rel}", f"b/{rel}", n=3)
    body = "".join(d)
    return f"diff --git a/{rel} b/{rel}\nindex 0000000..1111111 100644\n" + body

def mut_code(src):
    lines = src.split("\n")
    tree = ast.parse(src)
    cands = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and node.lineno == node.end_lineno:
            cands.append(("cmp", node))
        elif isinstance(node, ast.BoolOp):
            cands.append(("bool", node))
        elif isinstance(node, ast.Constant) and node.lineno == node.end_lineno and isinstance(node.value, (bool, int)) and not isinstance(node.value, str):
            cands.append(("const", node))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            cands.append(("not", node))
        elif isinstance(node, ast.Return) and node.value is not None and node.lineno == node.end_lineno:
            cands.append(("ret", node))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.lineno == node.end_lineno and 2 < len(node.value) < 30 and node.value.isupper():
            cands.append(("str", node))
    rng.shuffle(cands)
    for kind, node in cands:
        L = node.lineno - 1
        line = lines[L]
        new = None
        if kind == "cmp":
            for a, b in [(" == ", " != "), (" != ", " == "), (" is not ", " is "), (" is ", " is not "), (" < ", " <= "), (" > ", " >= "), (" not in ", " in "), (" in ", " not in ")]:
                if a in line:
                    new = line.replace(a, b, 1); break
        elif kind == "bool":
            if " and " in line: new = line.replace(" and ", " or ", 1)
            elif " or " in line: new = line.replace(" or ", " and ", 1)
        elif kind == "const":
            seg = line[node.col_offset:node.end_col_offset]
            if seg == "True": rep = "False"
            elif seg == "False": rep = "True"
            elif seg.isdigit(): rep = str(int(seg) + 1)
            else: continue
            new = line[:node.col_offset] + rep + line[node.end_col_offset:]
        elif kind == "not":
            seg = line[node.col_offset:node.col_offset + 4]
            if seg == "not ": new = line[:node.col_offset] + line[node.col_offset + 4:]
        elif kind == "ret":
            new = re.sub(r"return .*", "return None", line, count=1)
        elif kind == "str":
            seg = line[node.col_offset:node.end_col_offset]
            if seg[0] in "\"'":
                new = line[:node.col_offset] + seg[0] + seg[1:-1] + "_X" + seg[-1] + line[node.end_col_offset:]
        if new and new != line:
            lines[L] = new
            return "\n".join(lines), kind
    return None, None

def mut_comment(src):
    lines = src.split("\n")
    idx = [i for i, l in enumerate(lines) if l.strip() and not l.strip().startswith(("#", '"', "'", ")", "]", "}")) and not l.rstrip().endswith(("(", "[", "{", ",", "\\"))]
    i = rng.choice(idx)
    ind = re.match(r"\s*", lines[i]).group(0)
    lines.insert(i, ind + rng.choice(COMMENTS))
    return "\n".join(lines)

def mut_fixture(src):
    lines = src.split("\n")
    idx = [i for i, l in enumerate(lines) if l.strip() and not l.startswith(("#", "--"))]
    i = rng.choice(idx)
    words = lines[i].split(" ")
    j = rng.randrange(len(words))
    words.insert(j, rng.choice(["42", "x", "0", "NULL"]))
    lines[i] = " ".join(words)
    return "\n".join(lines)

made = 0
while made < n:
    r = rng.random()
    patches = []
    kind = None
    if r < 0.6:
        rel = rng.choice(pyfiles)
        src = open(os.path.join(root, rel)).read()
        new, k = mut_code(src)
        if new is None: continue
        patches.append(udiff(rel, src, new)); kind = "code-" + k
    elif r < 0.75:
        rel = rng.choice(fixtures)
        src = open(os.path.join(root, rel)).read()
        patches.append(udiff(rel, src, mut_fixture(src))); kind = "fixture"
    else:
        rel = rng.choice(pyfiles)
        src = open(os.path.join(root, rel)).read()
        patches.append(udiff(rel, src, mut_comment(src))); kind = "comment"
    cid = f"m{seed}x{made:03d}"
    open(os.path.join(outdir, cid + ".patch"), "w").write("".join(patches))
    print(cid, kind, rel)
    made += 1
