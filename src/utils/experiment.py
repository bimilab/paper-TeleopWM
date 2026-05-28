from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def sanitize_tag(tag: str | None) -> str:
    if not tag:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag.strip())
    cleaned = cleaned.strip("._-")
    return cleaned


def create_run_dir(root: str | Path, tag: str | None = None) -> Path:
    root = Path(root)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = sanitize_tag(tag)
    name = f"{stamp}_{suffix}" if suffix else stamp
    run_dir = root / name

    counter = 1
    while run_dir.exists():
        run_dir = root / f"{name}_{counter:02d}"
        counter += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def infer_run_dir_from_checkpoint(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint)
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
