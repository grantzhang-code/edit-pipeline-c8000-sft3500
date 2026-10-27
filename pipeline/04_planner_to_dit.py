"""Stage 4: Planner JSON -> native DiT render inputs (<out>/inputs/manifest.json + <out>/inputs/assets/*.png).

Conversion = the frozen DiT package's own code (vendor/dit-render-cfg2-v3-tayo58k, fetched by 00_fetch.sh):
  planner_to_native(planner_json, source region label->bbox maps, n)   XYXY -> YXYX, reference_label -> reference_bbox,
                                                                         edit actions moved onto their target element
  -> adapt_native_edit_annotation -> drop_color_palettes                 training-native annotation, no palettes
Target size = GPT target scaled to <= 1024^2 px with both sides multiples of 32; each Source image is composited on
white and resized the same way (BICUBIC, PNG, content-addressed by PNG sha256). Rows whose Planner output is still
invalid after repair-retry3, or that fail conversion, go to <out>/ledger.json. This is the same code path that built the
published version fzsft-balanced1500layoutonly-retry3-c8000-ditv2s3500-b39abcc3196c (dit58k prep1025.prepare).

  python pipeline/04_planner_to_dit.py --planner-json work/stage3/planner_json.jsonl --out work/stage4 [--limit N] [--verify]
--verify compares every produced row (sample_id, annotation, sizes, reference PNG sha256) with the pinned published
manifest.json of that version.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from common import PINS, REPO, fetch, read_jsonl, s3_bytes, sha

PKG = REPO / "vendor" / PINS["dit"]["render_package"]["dir"]
sys.path.insert(0, str(PKG))
from prepare_inputs import planner_to_native  # noqa: E402
from native_helpers import _patch_aligned_size, _rgb_on_white, adapt_native_edit_annotation, drop_color_palettes  # noqa: E402

ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")


def convert(planner_json: dict, source_regions: dict, n: int) -> dict:
    maps = [{e["label"]: e["bbox"] for e in source_regions.get(f"frame_{i}", {})
             .get("compositional_deconstruction", {}).get("elements", [])} for i in range(n)]
    native = adapt_native_edit_annotation(planner_to_native(planner_json, maps, n))
    return json.loads(drop_color_palettes(json.dumps(native, ensure_ascii=False)))


def resize_reference(raw: bytes) -> bytes:
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        rgb = _rgb_on_white(image)
        size = _patch_aligned_size(*image.size, max_pixels=1024**2, patch_size=32)
        if rgb.size != size:
            rgb = rgb.resize(size, Image.Resampling.BICUBIC)
        out = io.BytesIO()
        rgb.save(out, "PNG")
        return out.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample-prefix", default="fzsft-c{step}ball-", help="sample_id prefix (default = published naming)")
    ap.add_argument("--limit", type=int, default=0, help="first N eligible rows only (smoke test)")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    manifest, ledger = [], []
    for r in read_jsonl(a.planner_json):
        entry = dict(input_index=r["input_index"], case_id=r["case_id"], repair_method=r["repair_method"])
        if not all(r["valid"].values()):
            ledger.append(dict(entry, status="excluded", stage="planner", reason=f"planner output invalid after repair-retry3 {r['valid']}"))
            continue
        try:
            annotation = convert(r["planner_json"], r["source_regions"], len(r["source_images"]))
        except Exception as e:
            ledger.append(dict(entry, status="excluded", stage="conversion", reason=f"{type(e).__name__}: {e}"[:500]))
            continue
        base_id = r["case_id"] if ID_RE.fullmatch(r["case_id"]) and len(r["case_id"]) <= 140 else sha(r["case_id"].encode())
        w, h = _patch_aligned_size(r["target_size"][0], r["target_size"][1], max_pixels=1024**2, patch_size=32)
        manifest.append(dict(sample_id=a.sample_prefix.format(step=r["planner_step"]) + base_id, original_sample_id=r["case_id"],
                             input_index=r["input_index"], instruction=annotation, seed=PINS["dit"]["protocol"]["seed"],
                             source_sha256=[x["sha256"] for x in r["source_images"]], _uris=[x["uri"] for x in r["source_images"]],
                             target_width=w, target_height=h, gpt_target_width=r["target_size"][0],
                             gpt_target_height=r["target_size"][1], planner_checkpoint=r["planner_step"],
                             repair_method=r["repair_method"]))
        ledger.append(dict(entry, status="eligible", sample_id=manifest[-1]["sample_id"]))
    if a.limit:
        manifest = manifest[:a.limit]

    assets = a.out / "inputs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    todo = {h: u for m in manifest for h, u in zip(m["source_sha256"], m["_uris"])}

    def one(item):
        h, uri = item
        raw = s3_bytes(uri) if uri.startswith("s3://") else Path(uri).read_bytes()
        assert sha(raw) == h, f"source image sha mismatch {uri}"
        png = resize_reference(raw)
        (assets / f"{sha(png)}.png").write_bytes(png)
        return h, f"assets/{sha(png)}.png"

    with ThreadPoolExecutor(32) as ex:
        rel = dict(ex.map(one, sorted(todo.items())))
    for m in manifest:
        m["ref_relpaths"] = [rel[h] for h in m["source_sha256"]]
        m.pop("_uris")
    (a.out / "inputs" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    (a.out / "ledger.json").write_text(json.dumps(sorted(ledger, key=lambda x: x["input_index"]), ensure_ascii=False, indent=1) + "\n")
    print(f"{len(manifest)} DiT inputs, {sum(x['status'] == 'excluded' for x in ledger)} excluded, {len(todo)} reference images "
          f"-> {a.out / 'inputs'}")

    if a.verify:
        o = PINS["dit"]["outputs"]
        ref = {m["sample_id"]: m for m in json.loads(fetch(o["version_dir"] + "manifest.json").read_text())}
        if not a.limit:
            assert len(ref) == len(manifest), (len(ref), len(manifest))
        for m in manifest:
            p = ref[m["sample_id"]]
            for k in m:
                assert m[k] == p[k], (m["sample_id"], k)
        print(f"verify: {len(manifest)}/{len(manifest)} rows identical to {o['version_dir']}manifest.json "
              "(annotation, sizes, reference PNG sha256)")


if __name__ == "__main__":
    main()
