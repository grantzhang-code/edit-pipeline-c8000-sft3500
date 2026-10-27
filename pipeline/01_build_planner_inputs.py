"""Stage 1: build the Planner requests (system + user prompt + ordered Source images) for balanced-1500 layout-only.

Inputs (all pinned in pins.json, sha256-checked):
  * benchmark cases.luna.jsonl      - case order (input_index), ordered Source images, Luna instruction
  * layout_only source-captions.jsonl - per Source image (keyed by image sha256): compositional_deconstruction.elements
                                       [{label, parent, bbox xyxy_1000}] from ericdu layout-only-shortstart-c10000 (greedy)
  * data/balanced-1500-layoutonly/target-dimensions.json - target width/height per case (GPT Image 2.5 target)
  * data/balanced-1500-layoutonly/input-exclusions.json  - the 35 cases not inferred (missing Source JSON / target)

Output: <out>/planner_requests.jsonl, one row per eligible case (1465), plus <out>/source-regions.json in the format the
frozen Planner package reads ({image sha256: {"regions": {...}}}).

The prompt is the training contract given_ratio_plan_only_inline_edits (prompt spec eecfcbe5): SYSTEM =
prompts/planner_system.txt, user = '<image>\\nSource: frame_i\\n' per image + <source_regions> + <user_prompt> +
<aspect_ratio> (width/height, 4 decimals, ROUND_HALF_UP) + USER_OUTPUT_RULES. chat_prompt is the exact qwen3_5 chat
template text the engine receives (enable_thinking=False); prompt_sha256 = sha256(chat_prompt).

--verify <responses.jsonl.gz> checks every row byte-for-byte against the published c8000 greedy run (system, user,
source_regions, prompt_sha256) and data/.../expected-prompt-hashes.json; source-regions.json is checked against
DATASET.json.
  python pipeline/01_build_planner_inputs.py --out work/stage1 [--verify]
"""
from __future__ import annotations

import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from common import DATA, PINS, PROMPTS, fetch, read_jsonl, sha, write_jsonl

SYSTEM = (PROMPTS / "planner_system.txt").read_text()
CONTRACT = json.loads((PROMPTS / "planner_prompt_contract.json").read_text())
assert CONTRACT["SYSTEM"] == SYSTEM and sha(SYSTEM.encode()) == PINS["planner"]["system_prompt_sha256"]
RULES = CONTRACT["USER_OUTPUT_RULES"]
RESERVED = ("<image>", "<video>", "<audio>", "<|im_start|>", "<|im_end|>")


def user_prompt(instruction: str, count: int, size: tuple[int, int], source_regions: dict) -> str:
    """Same text as the frozen package's sourcejson_contract.user_prompt + INLINE.USER_OUTPUT_RULES."""
    assert instruction.strip() and not any(x in instruction for x in RESERVED)
    assert list(source_regions) == [f"frame_{i}" for i in range(count)]
    ratio = (Decimal(size[0]) / Decimal(size[1])).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    regions = ("<source_regions>\n" + json.dumps(source_regions, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
               + "\n</source_regions>\n")
    return ("".join(f"<image>\nSource: frame_{i}\n" for i in range(count)) + regions
            + f"<user_prompt>\n{instruction}\n</user_prompt>\n<aspect_ratio>\n{ratio}\n</aspect_ratio>" + RULES)


def chat_prompt(user: str) -> str:
    """qwen3_5 chat template, add_generation_prompt=True, enable_thinking=False (asserted equal in the worker)."""
    return ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n"
            + user.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def layout_regions(captions: Path) -> dict:
    """image sha256 -> {"compositional_deconstruction": {"elements": [{label, parent, bbox}]}} (schema-valid, untruncated)."""
    out = {}
    for r in read_jsonl(captions):
        if not r["schema_valid"] or r["finish_reason"] == "length":
            continue
        elements = json.loads(r["response"])["compositional_deconstruction"]["elements"]
        out[r["input_metadata"]["sha256"]] = {"compositional_deconstruction": {"elements": [
            {k: e[k] for k in ("label", "parent", "bbox")} for e in elements]}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verify", nargs="?", const=PINS["planner"]["outputs"]["greedy_responses"]["uri"],
                    help="responses.jsonl.gz to compare with (default: the pinned c8000 greedy run)")
    a = ap.parse_args()

    b = PINS["benchmark"]["cases_luna"]
    cases = read_jsonl(fetch(b["uri"], b["sha256"]))
    c = PINS["source_layout_json"]["captions"]
    layout = layout_regions(fetch(c["uri"], c["sha256"]))
    targets = json.loads((DATA / "target-dimensions.json").read_text())
    excluded = json.loads((DATA / "input-exclusions.json").read_text())

    rows, regions_file = [], {}
    for i, case in enumerate(cases):
        if str(i) in excluded:
            continue
        assert case["instruction"] == case["user_instruction"] == case["rewritten_instruction"]
        assert case["instruction_variant"] == PINS["luna_rewrite"]["variant"]
        keys = [x.get("source_key", x["sha256"]) for x in case["inputs"]]
        regions = {f"frame_{k}": layout[key] for k, key in enumerate(keys)}
        for key in keys:
            regions_file[key] = {"regions": layout[key]}
        t = targets[case["id"]]
        size = (t["width"], t["height"])
        user = user_prompt(case["instruction"], case["input_count"], size, regions)
        rows.append(dict(
            input_index=i, case_id=case["id"], input_count=case["input_count"],
            instruction=case["instruction"], instruction_variant=case["instruction_variant"],
            original_user_instruction=case.get("original_user_instruction"),
            source_images=[dict(uri=x["s3_uri"], sha256=x["sha256"], width=x["width"], height=x["height"]) for x in case["inputs"]],
            source_regions=regions, target_size=list(size), target_image_uri=t["s3_uri"],
            system_prompt=SYSTEM, user_prompt=user, prompt_sha256=sha(chat_prompt(user).encode())))
    a.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.out / "planner_requests.jsonl", rows)
    (a.out / "source-regions.json").write_text(json.dumps(regions_file, ensure_ascii=False) + "\n")
    print(f"{len(rows)} requests ({sum(r['input_count'] > 1 for r in rows)} multi-image), {len(excluded)} excluded "
          f"-> {a.out / 'planner_requests.jsonl'}")

    if a.verify:
        dataset = json.loads((DATA / "DATASET.json").read_text())
        assert sha((a.out / "source-regions.json").read_bytes()) == dataset["files"]["source-regions.json"], "source-regions.json differs"
        expected = json.loads((DATA / "expected-prompt-hashes.json").read_text())
        published = {r["input_index"]: r for r in read_jsonl(fetch(a.verify)) if r.get("status") == "inferred"}
        assert len(published) == len(rows) == len(expected), (len(published), len(rows), len(expected))
        for r in rows:
            p = published[r["input_index"]]
            assert r["case_id"] == p["case_id"] and r["prompt_sha256"] == p["prompt_sha256"] == expected[str(r["input_index"])]
            assert r["system_prompt"] == p["system_prompt"] and r["user_prompt"] == p["user_prompt"]
            assert r["source_regions"] == p["source_regions"] and r["target_size"] == p["target_size"]
        print(f"verify: {len(rows)}/{len(rows)} prompts byte-identical to {a.verify}; source-regions.json sha matches DATASET.json")


if __name__ == "__main__":
    main()
