from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_DATA_ROOT = Path("/path/to/mile_action_diverse/train/Town01")
ACTION_NAMES = ("throttle", "steer", "brake")


@dataclass(frozen=True)
class WindowIndex:
    run_key: str
    run_id: str
    town: str | None
    run_path: Path
    start: int
    end: int


@dataclass(frozen=True)
class RunInfo:
    run_key: str
    run_id: str
    town: str | None
    path: Path


class CarlaRolloutDataset(Dataset):
    """Sliding rollout windows for MILE/CARLA Town01 driving runs.

    Each sample is a contiguous slice:
    - past frames/actions: indices [start, start + past_len)
    - future frames/actions: indices [start + past_len, start + past_len + future_len)

    The MILE dataframe stores actions in one vector-valued `action` column. In
    this dataset the vector is treated as [throttle, steer, brake].
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_DATA_ROOT,
        split: str = "train",
        runs: Iterable[str] | None = None,
        past_len: int = 9,
        future_len: int = 8,
        image_size: tuple[int, int] = (160, 256),
        action_column: str = "action",
        speed_column: str = "speed",
        normalize_controls: bool = False,
        speed_scale: float = 20.0,
        include_metadata: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.past_len = past_len
        self.future_len = future_len
        self.window_len = past_len + future_len
        self.image_size = image_size
        self.action_column = action_column
        self.speed_column = speed_column
        self.normalize_controls = normalize_controls
        self.speed_scale = speed_scale
        self.include_metadata = include_metadata

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.layout = self._detect_layout(self.root)
        self.run_infos = self._discover_runs(runs)
        self.run_ids = [run.run_id for run in self.run_infos]
        self.frames_by_run: dict[str, list[Path]] = {}
        self.actions_by_run: dict[str, np.ndarray] = {}
        self.speed_by_run: dict[str, np.ndarray] = {}
        self.windows: list[WindowIndex] = []

        for run_info in self.run_infos:
            self._load_run(run_info)

    @staticmethod
    def _default_runs(split: str) -> list[str]:
        if split == "train":
            return ["0000", "0001"]
        if split in {"val", "validation", "test"}:
            return ["0002"]
        raise ValueError(f"Unknown split {split!r}. Use train, val, or test.")

    def _split_frame_range(self, n_frames: int) -> tuple[int, int]:
        if self.layout == "multi_town":
            return 0, n_frames
        if self.split == "train":
            return 0, n_frames
        midpoint = n_frames // 2
        if self.split in {"val", "validation"}:
            return 0, midpoint
        if self.split == "test":
            return midpoint, n_frames
        return 0, n_frames

    @staticmethod
    def _is_run_dir(path: Path) -> bool:
        return path.is_dir() and (path / "image").is_dir() and (path / "pd_dataframe.pkl").is_file()

    @staticmethod
    def _is_town_dir(path: Path) -> bool:
        return path.is_dir() and path.name.startswith("Town")

    def _detect_layout(self, root: Path) -> str:
        direct_runs = [path for path in sorted(root.iterdir()) if self._is_run_dir(path)]
        town_dirs = [path for path in sorted(root.iterdir()) if self._is_town_dir(path)]
        if direct_runs:
            return "single_town"
        if town_dirs:
            return "multi_town"
        return "unknown"

    def _discover_runs(self, runs: Iterable[str] | None) -> list[RunInfo]:
        if self.layout == "single_town":
            return self._discover_single_town_runs(runs)
        if self.layout == "multi_town":
            return self._discover_multi_town_runs(runs)
        raise FileNotFoundError(
            f"Could not find MILE run folders under {self.root}. Expected either "
            f"{self.root}/0000/image or {self.root}/TownXX/0000/image."
        )

    def _discover_single_town_runs(self, runs: Iterable[str] | None) -> list[RunInfo]:
        available = {path.name: path for path in sorted(self.root.iterdir()) if self._is_run_dir(path)}
        town = self.root.name if self.root.name.startswith("Town") else None
        if runs is None:
            requested = self._default_runs(split=self.split)
            if not all(run_id in available for run_id in requested):
                requested = sorted(available)
        else:
            requested = list(runs)

        run_infos = []
        for run_id in requested:
            run_path = available.get(run_id)
            if run_path is None:
                run_path = self.root / run_id
            run_key = f"{town}/{run_id}" if town else run_id
            run_infos.append(RunInfo(run_key=run_key, run_id=run_id, town=town, path=run_path))
        return run_infos

    def _discover_multi_town_runs(self, runs: Iterable[str] | None) -> list[RunInfo]:
        requested = set(runs) if runs is not None else None
        run_infos: list[RunInfo] = []
        for town_dir in sorted(path for path in self.root.iterdir() if self._is_town_dir(path)):
            for run_path in sorted(path for path in town_dir.iterdir() if self._is_run_dir(path)):
                run_id = run_path.name
                run_key = f"{town_dir.name}/{run_id}"
                if requested is not None and run_id not in requested and run_key not in requested:
                    continue
                run_infos.append(
                    RunInfo(run_key=run_key, run_id=run_id, town=town_dir.name, path=run_path)
                )
        if not run_infos:
            raise FileNotFoundError(f"No matching run folders found under multi-town root: {self.root}")
        return run_infos

    def _load_run(self, run_info: RunInfo) -> None:
        run_dir = run_info.path
        run_id = run_info.run_id
        frame_dir = run_dir / "image"
        df_path = run_dir / "pd_dataframe.pkl"

        if not frame_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {frame_dir}")
        if not df_path.exists():
            raise FileNotFoundError(f"Missing dataframe: {df_path}")

        df = pd.read_pickle(df_path)
        if self.action_column not in df.columns:
            raise KeyError(
                f"Missing action column {self.action_column!r} in {df_path}. "
                f"Available columns: {list(df.columns)}"
            )
        if self.speed_column not in df.columns:
            raise KeyError(
                f"Missing speed column {self.speed_column!r} in {df_path}. "
                f"Available columns: {list(df.columns)}"
            )

        frame_paths = self._frame_paths_from_dataframe(run_dir, df)
        actions = np.stack(
            df[self.action_column].map(lambda value: np.asarray(value, dtype=np.float32)).to_numpy()
        )
        speed = np.stack(
            df[self.speed_column].map(lambda value: np.asarray(value, dtype=np.float32)).to_numpy()
        )
        if actions.ndim != 2:
            raise ValueError(f"Expected 2D action array for run {run_id}, got {actions.shape}")
        if speed.ndim == 1:
            speed = speed[:, None]
        if speed.ndim != 2 or speed.shape[1] != 1:
            raise ValueError(f"Expected speed shape [N, 1] for run {run_id}, got {speed.shape}")
        if self.normalize_controls:
            if self.speed_scale <= 0:
                raise ValueError(f"speed_scale must be positive, got {self.speed_scale}")
            speed = speed / float(self.speed_scale)
        if len(frame_paths) != len(actions) or len(frame_paths) != len(speed):
            raise ValueError(
                f"Frame/action count mismatch for run {run_id}: "
                f"{len(frame_paths)} frames vs {len(actions)} actions vs {len(speed)} speed rows"
            )

        start_frame, end_frame = self._split_frame_range(len(frame_paths))
        last_start = end_frame - self.window_len
        if last_start < start_frame:
            raise ValueError(
                f"Run {run_id} split {self.split} is too short for window length {self.window_len}"
            )

        self.frames_by_run[run_info.run_key] = frame_paths
        self.actions_by_run[run_info.run_key] = actions
        self.speed_by_run[run_info.run_key] = speed
        for start in range(start_frame, last_start + 1):
            self.windows.append(
                WindowIndex(
                    run_key=run_info.run_key,
                    run_id=run_info.run_id,
                    town=run_info.town,
                    run_path=run_info.path,
                    start=start,
                    end=start + self.window_len,
                )
            )

    @staticmethod
    def _frame_paths_from_dataframe(run_dir: Path, df: pd.DataFrame) -> list[Path]:
        if "image_path" in df.columns:
            frame_paths = [run_dir / rel_path for rel_path in df["image_path"].tolist()]
        else:
            frame_paths = sorted((run_dir / "image").glob("*.png"))

        missing = [path for path in frame_paths[:10] if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing image file(s), first missing: {missing[0]}")
        return frame_paths

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict[str, object]]:
        window = self.windows[index]
        frame_paths = self.frames_by_run[window.run_key][window.start : window.end]
        actions = self.actions_by_run[window.run_key][window.start : window.end]
        speed = self.speed_by_run[window.run_key][window.start : window.end]

        frames = torch.stack([self._load_frame(path) for path in frame_paths], dim=0)
        actions_tensor = torch.from_numpy(actions.copy()).float()
        speed_tensor = torch.from_numpy(speed.copy()).float()

        past_end = self.past_len
        sample: dict[str, torch.Tensor | dict[str, object]] = {
            "past_frames": frames[:past_end],
            "future_frames": frames[past_end:],
            "past_actions": actions_tensor[:past_end],
            "future_actions": actions_tensor[past_end:],
            "past_speed": speed_tensor[:past_end],
            "future_speed": speed_tensor[past_end:],
        }

        if self.include_metadata:
            sample["metadata"] = {
                "town": window.town,
                "run_key": window.run_key,
                "run_id": window.run_id,
                "run_path": str(window.run_path),
                "start": window.start,
                "end": window.end - 1,
                "past_indices": list(range(window.start, window.start + self.past_len)),
                "future_indices": list(range(window.start + self.past_len, window.end)),
                "frame_paths": [str(path) for path in frame_paths],
                "action_names": ACTION_NAMES,
            }

        return sample

    def _load_frame(self, path: Path) -> torch.Tensor:
        height, width = self.image_size
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()
