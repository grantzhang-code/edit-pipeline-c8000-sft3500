"""Shared helpers: pinned URIs (pins.json), sha-checked S3 reads with a local cache."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PINS = json.loads((REPO / "pins.json").read_text())
DATA = REPO / "data" / "balanced-1500-layoutonly"
PROMPTS = REPO / "prompts"
CACHE = Path(os.environ.get("PIPELINE_CACHE", Path.home() / ".cache" / "edit-pipeline-c8000-sft3500"))
PROFILE = os.environ.get("AWS_PROFILE", "corp_core-cn-readwrite")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def s3_bytes(uri: str) -> bytes:
    r = subprocess.run(["aws", "--profile", PROFILE, "s3", "cp", uri, "-", "--only-show-errors"], capture_output=True)
    if r.returncode:
        raise RuntimeError(f"S3 read failed for {uri}: {r.stderr.decode(errors='replace')[-300:]}")
    return r.stdout


def fetch(uri: str, sha256: str | None = None) -> Path:
    """Local copy of an S3 object (or a local path), cached by URI; verified against sha256 when given."""
    if not uri.startswith("s3://"):
        p = Path(uri)
    else:
        p = CACHE / uri[5:]
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".part")
            tmp.write_bytes(s3_bytes(uri))
            tmp.replace(p)
    if sha256 and sha(p.read_bytes()) != sha256:
        raise ValueError(f"sha256 mismatch for {uri}")
    return p


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
