import sys, os
sys.path.insert(0, "/app/br")
import main as M
patch, root = sys.argv[1], sys.argv[2]
for f in M.parse_patch(open(patch).read()):
    old = open(os.path.join(root, f["old"])).read() if f["old"] else None
    new, _, _ = M.apply_hunks(old, f["hunks"])
    if f["new"] is None:
        os.remove(os.path.join(root, f["old"]))
    else:
        open(os.path.join(root, f["new"]), "w").write(new)
