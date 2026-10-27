#!/bin/bash
# usage: launch.sh <kube_context> <pod> <step> <group_node> <package_s3> <package_sha256> [attempt] [plan_name]
# Copies the pinned s5cmd into the hold pod, then starts node_run.sh detached (kubectl exec would otherwise block).
set -uo pipefail
CTX=$1 POD=$2 STEP=$3 NODE=$4 PKG=$5 SHA=$6 ATTEMPT=${7:-1} PLAN_NAME=${8:-}
RUN=${NODE_RUN_S3:-s3://core-cn-oss.canva.com/usr/grantzhang/runbooks/editparser-r8-pe-rtx64-20261022/packages/node_run.sh}
K="kubectl --context $CTX -n b-core-cn-prod"
JS=$($K get pod $POD -o jsonpath='{.metadata.labels.jobset\.sigs\.k8s\.io/jobset-name}')
UID_=$($K get jobset $JS -o jsonpath='{.metadata.uid}')
[ -n "$JS" ] && [ -n "$UID_" ] || { echo "cannot resolve jobset of $POD"; exit 1; }
S5=$(dirname "$0")/../pe/bin/s5cmd
timeout 60 $K cp "$S5" $POD:/mnt/ephemeral/s5cmd -c hold
timeout 60 $K exec $POD -- /bin/bash -c "
/mnt/ephemeral/s5cmd cp --source-region us-east-1 $RUN /mnt/ephemeral/node_run.sh >/dev/null && chmod +x /mnt/ephemeral/node_run.sh &&
setsid nohup /mnt/ephemeral/node_run.sh $STEP $NODE $PKG $SHA $JS $UID_ $ATTEMPT $PLAN_NAME > /mnt/ephemeral/node_run_c${STEP}_a${ATTEMPT}_n${NODE}${PLAN_NAME:+_$PLAN_NAME}.log 2>&1 < /dev/null &
sleep 2; echo started $CTX $POD c$STEP node$NODE"
