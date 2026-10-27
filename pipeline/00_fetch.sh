#!/bin/bash
# Fetch the three frozen dependencies into vendor/ (gitignored), sha256-checked against pins.json:
#   vendor/pe/                          Planner inference package (r3-9b-pe-1023 @ 30c6515; runs on the GPU hold, read here)
#   vendor/dit-render-cfg2-v3-tayo58k/  DiT render package (cfg2-v3 + SFT-v2 checkpoint allow-list), used by stages 3-4
#   vendor/prism-like/                  canvanauts/jimmyqiancode 10分-prism-like @ 5a9db85, used by stage 5
# usage: pipeline/00_fetch.sh            (AWS_PROFILE defaults to corp_core-cn-readwrite; gh / git need GitHub access)
set -euo pipefail
cd "$(dirname "$0")/.."
V=vendor; mkdir -p $V
P=${AWS_PROFILE:-corp_core-cn-readwrite}
pin(){ python3 -c "import json,sys; d=json.load(open('pins.json'))
for k in sys.argv[1].split('.'): d=d[k]
print(d)" "$1"; }

get_tar(){  # <uri> <sha256> <dest dir> <marker file inside>
  local uri=$1 want=$2 dest=$3 marker=$4 t
  [ -e "$dest/$marker" ] && { echo "have $dest"; return; }
  t=$(mktemp -d); aws --profile $P s3 cp "$uri" $t/pkg.tgz --only-show-errors
  echo "$want  $t/pkg.tgz" | sha256sum -c - >/dev/null
  tar -xzf $t/pkg.tgz -C $V; rm -rf $t; echo "fetched $dest"
}
get_tar "$(pin planner.inference_package.uri)" "$(pin planner.inference_package.sha256)" $V/pe controller.py
get_tar "$(pin dit.render_package.uri)" "$(pin dit.render_package.sha256)" $V/$(pin dit.render_package.dir) render.py

if [ ! -e $V/prism-like/SKILL.md ]; then
  t=$(mktemp -d)
  git clone -q --filter=blob:none --no-checkout https://github.com/canvanauts/jimmyqiancode.git $t/repo
  git -C $t/repo -c advice.detachedHead=false checkout -q "$(pin evaluation.commit)" -- "10分-prism-like"
  mv "$t/repo/10分-prism-like" $V/prism-like; rm -rf $t
  echo "fetched $V/prism-like @ $(pin evaluation.commit)"
fi
echo "vendor ready: $(ls $V | tr '\n' ' ')"
