"""Stage 6: score DiT outputs with 10分-prism-like (canvanauts/jimmyqiancode @ 5a9db85, protocol prism-edit1500-v7-single).

Per image two independent gpt-6.1-sol (effort medium) judgments, one per axis, never re-rolled:
  alignment  - original user instruction (the skill's cases.jsonl, NOT the Luna rewrite) + all ordered inputs + candidate
               + task-category focus;  aesthetic - original instruction + candidate.  Score 1-10 + one-sentence evidence.
Report: mean per task category, then equal weight over the 20 categories; overall = mean of the two axes.

Modes
  official  the skill's own CLI per image (prepare -> run --execute -> collect -> report), Flex + background polling.
  fast      the skill's request builder / validator / aggregator, but service_tier "default" and synchronous requests
            (isolated variant prism10-v7-default-sync). This is how the reported c8000 SFT-3500 numbers were produced;
            do not mix its scores with official Flex scores.
Candidates: --index = stage-5 INDEX.json (local paths) or a published INDEX.json with image_uri / record_uri entries
(default: the pinned c8000 SFT-3500 renders). Nothing is sent without --execute (paid API calls).

  python pipeline/06_eval_prism.py --work work/prism --model-id "c8000 SFT-3500" [--index ...] [--mode fast|official]
         [--execute --key-file ~/.config/openai/api_key] [--concurrency 128] [--limit N]
Outputs in --work: results.jsonl (one skill result row per image) and REPORT.json (skill summarize()).
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import PINS, REPO, fetch, s3_bytes, sha

SKILL = REPO / "vendor" / "prism-like"
sys.path[:0] = [str(SKILL / "scripts"), str(SKILL / "assets" / "evaluator")]
from presentation import media  # noqa: E402
from request_builder import build  # noqa: E402
from prism_score import VERSION, aggregate as combine, validate  # noqa: E402
import evaluate as skill_eval  # noqa: E402

AXES = ("alignment", "aesthetic")
MODEL = "gpt-6.1-sol"
CASES = {c["id"]: c for c in map(json.loads, (SKILL / "assets" / "cases.jsonl").open())}
TASKS = json.loads((SKILL / "assets" / "task_map.json").read_text())
lock = threading.Lock()


def read(uri_or_path: str) -> bytes:
    return s3_bytes(uri_or_path) if uri_or_path.startswith("s3://") else Path(uri_or_path).read_bytes()


def candidates(index: str) -> list[dict]:
    d = json.loads(fetch(index).read_text())
    out = []
    for e in d["entries"]:
        if e.get("status") != "generated":
            continue
        out.append(dict(case_id=e["original_sample_id"], sample_id=e["sample_id"],
                        image=e.get("image_uri") or e["image"], record=e.get("record_uri") or e["record"]))
    return out


def originals(case: dict, media_dir: Path) -> dict:
    """input_id -> local path of each original input (sha-checked, shared cache)."""
    paths = {}
    for i in case["inputs"]:
        p = media_dir / (i["sha256"] + Path(i["s3_uri"]).suffix)
        if not p.exists():
            raw = s3_bytes(i["s3_uri"])
            assert sha(raw) == i["sha256"], i["s3_uri"]
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + f".{threading.get_ident()}")
            tmp.write_bytes(raw)
            tmp.replace(p)
        paths[i["input_id"]] = p
    return paths


def candidate_file(c: dict, media_dir: Path) -> tuple[Path, str]:
    png = read(c["image"])
    h = sha(png)
    assert h == json.loads(read(c["record"]))["png_sha256"], "candidate PNG differs from its render record"
    p = media_dir / f"{h}.png"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(png)
    return p, h


def score_official(c, a) -> dict | None:
    case_dir = a.work / "official" / sha(c["case_id"].encode())[:16]
    result = case_dir / "result.json"
    if result.exists():
        return json.loads(result.read_text())
    media_dir = a.work / "media"
    cand, _ = candidate_file(c, media_dir)
    inputs = case_dir / "inputs.json"
    case_dir.mkdir(parents=True, exist_ok=True)
    inputs.write_text(json.dumps({k: str(v) for k, v in originals(CASES[c["case_id"]], media_dir).items()}))
    ev = [sys.executable, str(SKILL / "scripts" / "evaluate.py")]  # its own dir on sys.path (io_support)
    work = case_dir / "work"
    if not work.exists():
        subprocess.run(ev + ["prepare", "--case-id", c["case_id"], "--model-id", a.model_id, "--candidate", str(cand),
                             "--inputs-json", str(inputs), "--work", str(work)], check=True, capture_output=True)
    run = ev + ["run", "--work", str(work)] + (["--execute", "--key-file", str(a.key_file)] if a.execute else [])
    subprocess.run(run, check=True, capture_output=True)
    if not a.execute:
        return None
    subprocess.run(ev + ["collect", "--work", str(work), "--out", str(result)], check=True, capture_output=True)
    return json.loads(result.read_text())


def post(body: bytes, key: str) -> dict:
    req = urllib.request.Request("https://api.openai.com/v1/responses", data=body, method="POST",
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def append(path: Path, row: dict):
    with lock, path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_fast(c, a, key, done) -> dict | None:
    """One synchronous default-tier judgment per missing axis; retries only requests that got no completed response."""
    case, task = CASES[c["case_id"]], TASKS[c["case_id"]]
    media_dir = a.work / "media"
    cand, h = candidate_file(c, media_dir)
    sources = {i["sha256"]: {"local_path": str(p)} for i, p in
               zip(case["inputs"], originals(case, media_dir).values())}
    for axis in AXES:
        if (c["case_id"], axis) in done or not a.execute:
            continue
        request, proof = build(axis, {"image_path": str(cand), "image_sha256": h}, media, case=case, sources=sources, task=task)
        request.update(service_tier="default", background=False)
        body = json.dumps(request).encode()
        for attempt in range(8):
            try:
                raw = post(body, key)
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 and e.code < 500:
                    append(a.work / "stopped.jsonl", dict(case_id=c["case_id"], axis=axis, error=f"HTTP {e.code}"))
                    return None
            except Exception:  # timeout / reset: no response id, re-POST
                pass
            time.sleep(min(300, 10 * 2 ** attempt) + random.uniform(0, 10))
        else:
            return None
        append(a.work / "raw.jsonl", dict(case_id=c["case_id"], axis=axis, candidate_sha256=h, response=raw))
        try:
            assert raw.get("status") == "completed" and str(raw.get("model", "")).startswith(MODEL)
            assert (raw.get("reasoning") or {}).get("effort") == "medium"
            text = "".join(x.get("text", "") for o in raw.get("output", []) for x in o.get("content", []) or []
                           if x.get("type") == "output_text")
            value = validate(json.loads(text), axis)
        except Exception as e:  # a completed-but-invalid judgment is kept and the axis stops (no reroll)
            append(a.work / "stopped.jsonl", dict(case_id=c["case_id"], axis=axis, response_id=raw.get("id"), error=str(e)[:300]))
            return None
        append(a.work / "judgments.jsonl", dict(case_id=c["case_id"], axis=axis, task=task, candidate_sha256=h,
                                                sample_id=c["sample_id"], response_id=raw["id"], judgment=value,
                                                presentation=proof))
    return None


def fast_results(a) -> list[dict]:
    j = {}
    p = a.work / "judgments.jsonl"
    for r in (map(json.loads, p.open()) if p.exists() else []):
        j.setdefault(r["case_id"], {}).setdefault(r["axis"], r)  # first completed judgment wins
    rows = []
    for cid, axes in j.items():
        if set(axes) != set(AXES):
            continue
        r = combine({x: [axes[x]["judgment"]] for x in AXES})
        r.update(case_id=cid, model_id=a.model_id, task=axes["alignment"]["task"],
                 candidate_sha256=axes["alignment"]["candidate_sha256"],
                 provenance=dict(response_ids=[axes[x]["response_id"] for x in AXES], variant="prism10-v7-default-sync"))
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--index", default=PINS["dit"]["outputs"]["images"] + "INDEX.json")
    ap.add_argument("--mode", choices=["fast", "official"], default="fast")
    ap.add_argument("--execute", action="store_true", help="send the paid judge requests")
    ap.add_argument("--key-file", type=Path, default=Path.home() / ".config/openai/api_key")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    cands = candidates(a.index)
    unknown = [c for c in cands if c["case_id"] not in CASES]
    assert not unknown, f"{len(unknown)} candidates are not balanced-1500 cases"
    cands = cands[:a.limit] if a.limit else cands
    print(f"{len(cands)} candidates, mode {a.mode}, {'EXECUTE' if a.execute else 'dry run (no API calls)'}", flush=True)

    if a.mode == "fast":
        key = a.key_file.read_text().strip() if a.execute else None
        p = a.work / "judgments.jsonl"
        done = {(r["case_id"], r["axis"]) for r in map(json.loads, p.open())} if p.exists() else set()
        with ThreadPoolExecutor(a.concurrency) as ex:
            list(ex.map(lambda c: score_fast(c, a, key, done), cands))
        rows = fast_results(a)
    else:
        with ThreadPoolExecutor(a.concurrency) as ex:
            rows = [r for r in ex.map(lambda c: score_official(c, a), cands) if r]
    with (a.work / "results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = dict(protocol_id=VERSION, mode=a.mode, judge=f"{MODEL} medium, 1 vote/axis",
                  skill=f"canvanauts/jimmyqiancode 10分-prism-like @{PINS['evaluation']['commit']}",
                  candidates=len(cands), scored=len(rows), by_model=skill_eval.summarize(rows) if rows else {})
    (a.work / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    s = report["by_model"].get(a.model_id, {}).get("scores")
    print(f"scored {len(rows)}/{len(cands)}; scores {s} -> {a.work / 'REPORT.json'}")


if __name__ == "__main__":
    main()
