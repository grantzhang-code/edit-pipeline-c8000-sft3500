"""Stage 5: render the DiT inputs with tayo SFT-v2 step=3500 (EMA) on one 8-GPU node (H100 / H200), in the training image.

Runs the frozen package's render.py (vendor/dit-render-cfg2-v3-tayo58k) exactly as the dit58k pod workers did:
  1. CPU admission: render.py --prepare-only (downloads ckpt / Qwen / VAE into --cache-dir, native token-budget audit);
     samples the audit rejects are dropped (EXCLUDED.json) and admission reruns in a fresh work dir (<= 8 attempts);
  2. one GPU run (torchrun, 8 ranks, FSDP shard 8), protocol euler50-cfg2-allsteps-v2, seed 42; a work dir is never
     relaunched once GPU_LAUNCH_STARTED.json exists.
Outputs: <run>/native-a<k>/results/step=000003500/ema/{images,records}/<sample_id>.{png,json}, VERIFIED.json, and
<run>/INDEX.json (sample_id -> original_sample_id / input_index / image / record) for stage 6.

  python pipeline/05_render_dit.py --inputs work/stage4/inputs --run /mnt/local/dit-c8000-sft3500 --cache-dir /mnt/local/dit-cache
Needs: image pins.dit.image (Transformer Engine / magi_attention / torch must match the frozen runtime), 8 visible GPUs,
S3 read for the checkpoint bucket (pod role, or --profile). ~155 GB ckpt + ~56 GB Qwen/VAE in the cache on first use.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import PINS, REPO

PKG = REPO / "vendor" / PINS["dit"]["render_package"]["dir"]


def admission_failures(work: Path) -> dict:
    failed = {}
    for f in (work / "cpu-preflight").glob("*/native-audit-*.json"):
        for x in json.loads(f.read_text()).get("failed", []):
            failed[x["sample_id"]] = x["error"][:500]
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True, help="stage-4 inputs/ (manifest.json + assets/)")
    ap.add_argument("--run", type=Path, required=True, help="new run directory")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", default=PINS["dit"]["checkpoint"])
    ap.add_argument("--weights", default=PINS["dit"]["weights"], choices=["ema", "live"])
    ap.add_argument("--profile", default=None, help="AWS profile; omit inside a pod with a workload role")
    a = ap.parse_args()

    step = int(a.checkpoint.rstrip("/").rsplit("step=", 1)[1].split(".")[0])
    inputs = a.run / "inputs"
    if not inputs.exists():
        shutil.copytree(a.inputs, inputs)
    rows = json.loads((inputs / "manifest.json").read_text())
    # images without EFA: the image's NCCL_NET_PLUGIN=ofi fails and NCCL then segfaults in UCX -> plain sockets
    env = {k: v for k, v in os.environ.items() if k not in ("NCCL_TUNER_PLUGIN", "FI_PROVIDER")}
    env.update(NCCL_NET_PLUGIN="none", NCCL_NET="Socket")

    def argv(work: Path, prepare_only: bool) -> list:
        return [sys.executable, "-u", str(PKG / "render.py"), "--inputs", str(inputs), "--work-dir", str(work),
                "--cache-dir", str(a.cache_dir), "--checkpoint", a.checkpoint, "--weights", a.weights,
                *(["--profile", a.profile] if a.profile else []), "--download-workers", "48",
                "--startup-workers", "8", "--audit-workers", "4", *(["--prepare-only"] if prepare_only else [])]

    excluded, log = [], (a.run / "render.log").open("ab")
    for attempt in range(8):
        work = a.run / f"native-a{attempt}"
        if (work / "GPU_LAUNCH_STARTED.json").exists():
            sys.exit(f"{work} already launched on GPU; use a new --run directory")
        if subprocess.run(argv(work, True), stdout=log, stderr=subprocess.STDOUT, cwd=a.run, env=env).returncode == 0:
            break
        failed = admission_failures(work)
        if not failed:
            sys.exit(f"prepare-only failed without per-sample audit failures; see {a.run / 'render.log'}")
        excluded += [dict(sample_id=k, error=v, attempt=attempt) for k, v in sorted(failed.items())]
        rows = [r for r in rows if r["sample_id"] not in failed]
        (inputs / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    else:
        sys.exit("CPU admission did not converge in 8 attempts")
    (a.run / "EXCLUDED.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=1) + "\n")
    print(f"admitted {len(rows)}, excluded {len(excluded)}; rendering in {work}", flush=True)

    rc = subprocess.run(argv(work, False), stdout=log, stderr=subprocess.STDOUT, cwd=a.run, env=env).returncode
    result = work / "results" / f"step={step:09d}" / a.weights
    verified = (work / "VERIFIED.json").exists() and json.loads((work / "VERIFIED.json").read_text()).get("passed")
    index = [dict(sample_id=r["sample_id"], original_sample_id=r["original_sample_id"], input_index=r["input_index"],
                  repair_method=r.get("repair_method"), image=str(result / "images" / f"{r['sample_id']}.png"),
                  record=str(result / "records" / f"{r['sample_id']}.json"),
                  status="generated" if (result / "records" / f"{r['sample_id']}.json").exists() else "missing") for r in rows]
    index += [dict(sample_id=x["sample_id"], status="excluded", reason=x["error"]) for x in excluded]
    (a.run / "INDEX.json").write_text(json.dumps(dict(checkpoint=a.checkpoint, weights=a.weights,
                                                      protocol=PINS["dit"]["protocol"]["id"], entries=index), indent=1) + "\n")
    n = sum(e["status"] == "generated" for e in index)
    print(f"render rc={rc} verified={bool(verified)} generated {n}/{len(rows)} -> {a.run / 'INDEX.json'}")
    sys.exit(0 if rc == 0 and verified and n == len(rows) else 1)


if __name__ == "__main__":
    main()
