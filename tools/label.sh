#!/bin/sh
# label.sh <patchdir> <outdir>: inside brdev container with /src (pristine) mounted ro
set -u
P=$1; O=$2
mkdir -p $O
for p in $P/*.patch; do
  id=$(basename $p .patch)
  [ -f $O/$id.done ] && continue
  rm -rf /tmp/w && cp -r /src /tmp/w && cd /tmp/w
  python /tools/apply.py $p /tmp/w || { echo "$id patchfail"; cd /; continue; }
  FAILOUT=$O/$id PYTHONPATH=/tmp/w:/tools SKIP_INTEGRATION=0 PYTHONDONTWRITEBYTECODE=1 timeout 900 python -m pytest -q -p no:cacheprovider -p failplug -n 4 --timeout 300 tests > $O/$id.log 2>&1
  tail -1 $O/$id.log
  touch $O/$id.done; cd /
done
