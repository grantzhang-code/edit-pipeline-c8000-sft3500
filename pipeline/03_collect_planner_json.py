"""Stage 3: collect the final Planner JSON per case (repair-retry3 output) together with its render inputs.

Reads <planner dir>/captions.jsonl (+ SOURCE.json) of a <dataset>-repair-retry3 publication and the greedy base run's
responses.jsonl.gz (ordered Source images, per-frame Source regions, target size: identical inputs for every round).
Writes <out>/planner_json.jsonl, one row per inferred case:

  input_index, case_id, repair_method (greedy | rule_repair | retry_r1..r3), valid {json, schema, source_references},
  instruction (Luna), planner_json  (= captions.caption_grouped: grouped plan, bbox XYXY 0-1000, DiT input),
  planner_json_inline (= captions.caption: the model's own plan-only inline-edit output after repair),
  source_images [{uri, sha256}], source_regions {frame_i: {...}}, target_size [w, h] (GPT target)

  python pipeline/03_collect_planner_json.py --out work/stage3 [--planner-dir s3://.../c8000/balanced-1500-layoutonly-repair-retry3/]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from common import PINS, fetch, read_jsonl, write_jsonl


def main():
    out_pins = PINS["planner"]["outputs"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--planner-dir", default=out_pins["final_dir"], help="<dataset>-repair-retry3/ directory (s3:// or local)")
    a = ap.parse_args()
    d = a.planner_dir.rstrip("/") + "/"
    pinned = d == out_pins["final_dir"]
    source = json.loads(fetch(d + "SOURCE.json").read_text())
    captions = read_jsonl(fetch(d + "captions.jsonl", out_pins["final_captions"]["sha256"] if pinned else None))
    step = source["step"]
    responses = {r["input_index"]: r for r in read_jsonl(fetch(
        f"{source['base_run'].rstrip('/')}/checkpoint-{step}/responses.jsonl.gz",
        out_pins["greedy_responses"]["sha256"] if pinned else None))}

    rows, methods = [], collections.Counter()
    for c in sorted(captions, key=lambda r: r["input_index"]):
        r = responses[c["input_index"]]
        assert r["case_id"] == c["case_id"] and c["checkpoint_step"] == step
        grouped = json.loads(c["caption_grouped"]) if isinstance(c["caption_grouped"], str) else c["caption_grouped"]
        inline = json.loads(c["caption"]) if isinstance(c["caption"], str) else c["caption"]
        method = c.get("repair_method", "greedy")
        methods[method] += 1
        rows.append(dict(
            input_index=c["input_index"], case_id=c["case_id"], planner_step=step, repair_method=method,
            repair_rules=c.get("repair_rules", []), rerun_sampling=c.get("rerun_sampling"),
            valid=dict(json=c["json_valid"], schema=c["schema_valid"], source_references=c["source_references_valid"]),
            instruction=c["instruction"], instruction_variant=c["instruction_variant"],
            planner_json=grouped, planner_json_inline=inline,
            source_images=[dict(uri=u, sha256=h) for u, h in zip(r["source_image_uris"], r["source_image_sha256"])],
            source_regions=r["source_regions"], target_size=[c["target_width"], c["target_height"]]))
    write_jsonl(a.out / "planner_json.jsonl", rows)
    valid = sum(all(r["valid"].values()) for r in rows)
    print(f"{len(rows)} cases, {valid} valid (json+schema+refs), repair methods {dict(methods)} -> {a.out / 'planner_json.jsonl'}")
    if pinned:
        assert len(rows) == 1465 and valid == 1460, (len(rows), valid)
        by_outcome = collections.Counter()  # "retry_r1+rule_repair" = a retry output that also needed a rule repair
        for k, v in methods.items():
            by_outcome[k.split("+")[0]] += v
        by_outcome.pop("greedy")
        assert dict(by_outcome) == {k: v for k, v in source["outcome"].items() if k != "unrecovered"}, (by_outcome, source["outcome"])


if __name__ == "__main__":
    main()
