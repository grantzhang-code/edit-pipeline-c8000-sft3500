#!/bin/bash
# Runs inside a hold pod: fetch the pinned package, preflight, then the controller for one group node.
# usage: node_run.sh <step> <group_node> <package_s3> <package_sha256> <jobset_name> <jobset_uid> [attempt] [plan_name]
# The task directory and the training lineage (r7 / r8) come from the step's plan inside the package;
# that lineage's overlay, args, identity and jobset yaml are placed at the task root before preflight.
set -euo pipefail
STEP=$1 NODE=$2 PKG=$3 PKG_SHA=$4 JOBSET_NAME=$5 JOBSET_UID_ARG=$6 ATTEMPT=${7:-1}
PLAN_FILE=plans/${8:-plan-$STEP}.json  # default plans/plan-<step>.json
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
export JOBSET_UID=$JOBSET_UID_ARG CANVA_JOBSET_NAME=$JOBSET_NAME
PY=/opt/mcore-venv/bin/python
mkdir -p /mnt/ephemeral/pkg
cd /mnt/ephemeral/pkg
# s5cmd (official v2.3.0 linux-amd64, copied into the pod by tools/launch.sh) for every download.
echo "672299fea8941281702bd52a4e51c330a4e39c1540f4bdfc3b4e737823ac2878  /mnt/ephemeral/s5cmd" | sha256sum -c -
/mnt/ephemeral/s5cmd cp --source-region us-east-1 "$PKG" pe-$PKG_SHA.tar.gz
echo "$PKG_SHA  pe-$PKG_SHA.tar.gz" | sha256sum -c -
X=/mnt/ephemeral/pkg/x-$PKG_SHA
mkdir -p $X && tar -xzf pe-$PKG_SHA.tar.gz -C $X --strip-components=1
TASK=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['task_id'])" $X/$PLAN_FILE)
LINEAGE=$($PY -c "import json,sys; print(json.load(open(sys.argv[1])).get('lineage','r8'))" $X/$PLAN_FILE)
P_ATTEMPT=$($PY -c "import json,sys; print(json.load(open(sys.argv[1])).get('attempt',1))" $X/$PLAN_FILE)
[ "$P_ATTEMPT" = "$ATTEMPT" ] || { echo "attempt mismatch: plan $P_ATTEMPT vs launch $ATTEMPT"; exit 1; }
ROOT=/mnt/ephemeral/$TASK
[ ! -e $ROOT/controller.lock ] || { echo "$ROOT already used by a controller; refusing to reuse it"; exit 1; }
mkdir -p $ROOT && cp -a $X/. $ROOT/
cd $ROOT
cp $PLAN_FILE plan.json
cp lineage/$LINEAGE/* $ROOT/
DATASET=$($PY -c "import json,sys; print(json.load(open(sys.argv[1])).get('dataset') or '')" plan.json)
[ -z "$DATASET" ] || cp datasets/$DATASET/cases.jsonl datasets/$DATASET/source-regions.json datasets/$DATASET/target-dimensions.json datasets/$DATASET/input-exclusions.json $ROOT/
PREFLIGHT=$($PY -c "import json,sys; print(json.load(open(sys.argv[1])).get('preflight','preflight.py'))" plan.json)
echo "task $TASK lineage $LINEAGE attempt $ATTEMPT"
mkdir -p training-runtime && tar -xzf training-overlay.tar.gz -C training-runtime
echo "dataset ${DATASET:-balanced-1500-r6} preflight $PREFLIGHT"
PE_TRAINING_ALIGNMENT=0 $PY $ROOT/$PREFLIGHT $ROOT > $ROOT/preflight.log 2>&1
$PY - <<PY
import hashlib,json
p=json.load(open('$ROOT/plan.json'))
assert hashlib.sha256(open('$ROOT/expected-prompt-hashes.json','rb').read()).hexdigest()==p['expected_prompt_hashes_sha256']
assert json.load(open('$ROOT/TRAINING_PROMPT_PREFLIGHT.json'))['status']=='PASS'
assert json.load(open('$ROOT/TRAINING_IDENTITY.json'))['run_id']==p['run_id'], 'lineage identity differs from plan'
print('preflight PASS', p['steps'], p['lineage'] if 'lineage' in p else 'r8', p['output_s3'])
PY
PE_TRAINING_ALIGNMENT=0 exec $PY -u $ROOT/controller.py --node $NODE > $ROOT/controller.log 2>&1
