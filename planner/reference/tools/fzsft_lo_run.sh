#!/bin/bash
# Layout-only out-domain variant of one fzsft-9b checkpoint (plans from fzsft_lo_prepare.py): package, then
# balanced-1500-layoutonly and balanced-1500-qwenpe-multi-layoutonly, each on the workers its plan lists (hold_nodes), once the
# full-JSON balanced run of the checkpoint has published its HF export (restored, not re-exported) and those workers are idle;
# after both publish, indomain_fix.py (rule repair + 3 retry rounds in one run, on the balanced plan's first two workers)
# publishes <dataset>-repair-retry3/ under planner-9b-1025/fzsft-9b-4b3162283b0b/c<step>/, then fzsft_organize.py.
#   fzsft_lo_run.sh <step> <kube context> <hold>      (log tools/fzsft_lo_c<step>.log)
set -uo pipefail
cd "$(dirname "$0")"
st=$1 CTX=$2 H=$3
PFX=s3://core-cn-oss.canva.com/usr/grantzhang/runbooks/editparser-r3-9b-pe-rtx8-20261023/packages
G=s3://core-cn-oss.canva.com/usr/grantzhang
PY=/home/coder/work/zhangguiwei/miniconda3/bin/python
export NODE_RUN_S3=$PFX/node_run_v2.sh
K="kubectl --context $CTX -n b-core-cn-prod"
A="timeout 300 aws --profile corp_core-cn-readwrite s3"
say(){ echo "$(date -u +%FT%TZ) $*"; }
pod(){ timeout 60 $K get pods -o name 2>/dev/null | grep "^pod/$1-workers-$2-0-" | cut -d/ -f2; }
idle(){ local p; p=$(pod $1 $2); [ -n "$p" ] && timeout 60 $K exec $p -c hold -- true >/dev/null 2>&1 && ! timeout 60 $K exec $p -c hold -- pgrep -f "controller.py --node|rerun_subset.py|node_rerun.sh" >/dev/null 2>&1; }
published(){ $A ls ${1}SUMMARY.json >/dev/null 2>&1; }
BAL=$G/balanced-1500-r6-editparser-fzsft-9b-4b3162283b0b-c$st-trainprompt-greedy-rtx32-bs64-20261026/
LB=balanced-1500-layoutonly LQ=balanced-1500-qwenpe-multi-layoutonly
out(){ echo $G/$1-editparser-fzsft-9b-4b3162283b0b-c$st-trainprompt-greedy-rtx8-bs64-20261026/; }
pkg=.pkg_fzsft_lo_c$st-$H.txt
ws(){ $PY -c "import json,sys; print(*json.load(open(sys.argv[1]))['hold_nodes'])" ../pe/plans/plan-$st-$1-fzsft-9b.json; }
if [ ! -s $pkg ]; then
  ( cd .. && flock .git/fzsft_lo.lock git -c core.fsmonitor=false add pe/plans/plan-$st-$LB-fzsft-9b.json pe/plans/plan-$st-$LQ-fzsft-9b.json &&
    { git -c core.fsmonitor=false diff --cached --quiet || git -c core.fsmonitor=false commit -q -m "fzsft-9b c$st layout-only out-domain plans ($CTX $H; export of the full-JSON balanced run reused)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; } ) || { say "commit failed"; exit 1; }
  C=$(git -C .. rev-parse --short HEAD); T=/home/coder/work/zhangguiwei/r3-9b-pe-1023/.pkg-$C.tar.gz
  git -C .. -c core.fsmonitor=false archive --format=tar.gz --prefix=pe/ HEAD:pe -o $T || exit 1
  LIST=$(tar tzf $T)
  for f in pe/plans/plan-$st-$LB-fzsft-9b.json pe/plans/plan-$st-$LQ-fzsft-9b.json pe/datasets/$LB/expected-prompt-hashes.json pe/datasets/$LQ/expected-prompt-hashes.json pe/controller.py; do
    grep -qx $f <<<"$LIST" || { say "package lacks $f"; exit 1; }; done
  D=$(sha256sum $T | cut -d' ' -f1)
  $A cp $T $PFX/pe-$C-$D.tar.gz --only-show-errors && [ "$($A cp $PFX/pe-$C-$D.tar.gz - | sha256sum | cut -d' ' -f1)" = "$D" ] || { say "upload check failed"; exit 1; }
  echo "$C $D" > $pkg; say "c$st layout-only package $C"
fi
read C D < $pkg
launch(){  # <hold> <worker> <group node> <plan>
  local p o; p=$(pod $1 $2); [ -n "$p" ] || return 1
  o=$(./launch.sh $CTX $p $st $3 $PFX/pe-$C-$D.tar.gz $D 1 $4 2>&1); grep -E "started|cannot" <<<"$o"; grep -q "^started " <<<"$o"
}
for X in $LB $LQ; do
  until $A ls $(out $X)control/STATE-node0.json >/dev/null 2>&1; do
    ok=1; $A ls ${BAL}checkpoint-$st/EXPORTED.json >/dev/null 2>&1 || ok=0
    [ $ok = 1 ] && for w in $(ws $X); do idle $H $w || ok=0; done
    if [ $ok = 1 ]; then
      i=0; for w in $(ws $X); do launch $H $w $i plan-$st-$X-fzsft-9b; i=$((i + 1)); done; say "c$st $X launched on $H workers $(ws $X)"; sleep 120
    else sleep 60; fi
  done
done
until published $(out $LB) && published $(out $LQ); do sleep 120; done
say "c$st layout-only greedy published"; echo $st >> fzsft_lo_steps.txt; $PY fzsft_organize.py >> fzsft_organize.log 2>&1
w=indomain_fix-fzsft-lo-c$st; mkdir -p $w
FIX_LINEAGE=fzsft-9b FIX_MULTI=1 FIX_RETRY_DATE=20261026 FIX_DEST=$G/planner-9b-1025/fzsft-9b-4b3162283b0b/ FIX_STEPS=$st FIX_WORK=$w \
  FIX_DATASETS=$LB,$LQ FIX_CONTEXT=$CTX FIX_HOLD=$H FIX_WORKERS=$(ws $LB | tr ' ' ',') $PY -u indomain_fix.py run >> $w/run.log 2>&1
say "c$st layout-only repair / retry finished (status $?)"; $PY fzsft_organize.py >> fzsft_organize.log 2>&1
