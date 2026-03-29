"""SurvVAUDataset — PyTorch Dataset for Conan-R1 training and evaluation."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .types import StructuredSample, SEVERITY_LEVELS
from .video_utils import frames_from_array, sample_frames

logger = logging.getLogger(__name__)

VALID_SPLITS = {"sft_train", "rl_train", "val", "test"}


class SurvVAUDataset(Dataset):
    """Dataset for Surv-VAU structured samples.

    Expects a directory with:
        annotations.jsonl  — one JSON object per line (StructuredSample fields)
        splits.json        — {"video_id": "sft_train"|"rl_train"|"val"|"test"}
        videos/            — video files named {video_id}.mp4 (optional)
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        num_frames: int = 25,
        frame_size: int = 448,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'")

        self.data_dir = Path(data_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.samples: List[Dict] = []
        self._load()

    def _load(self) -> None:
        ann_path = self.data_dir / "annotations.jsonl"
        splits_path = self.data_dir / "splits.json"

        if not ann_path.exists():
            logger.warning("annotations.jsonl not found at %s", ann_path)
            return

        split_map: Dict[str, str] = {}
        if splits_path.exists():
            with open(splits_path) as f:
                split_map = json.load(f)

        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                vid = obj.get("video_id", "")
                assigned_split = split_map.get(vid, obj.get("split", ""))
                if assigned_split == self.split:
                    self.samples.append(obj)

        logger.info("Loaded %d samples for split '%s'.", len(self.samples), self.split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        obj = self.samples[idx]
        video_path = self.data_dir / "videos" / f"{obj['video_id']}.mp4"

        if video_path.exists():
            frames = sample_frames(
                str(video_path), n=self.num_frames, size=(self.frame_size, self.frame_size)
            )
        else:
            # Return placeholder frames if video file is missing
            frames = [
                np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
                for _ in range(self.num_frames)
            ]

        # Convert frames to tensor: (T, C, H, W) float32 in [0, 1]
        frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)

        return {
            "video_id": obj["video_id"],
            "frames": frames_tensor,
            "prompt": obj.get("prompt", ""),
            "degradation_profile": obj.get("degradation_profile", []),
            "difficulty_level": float(obj.get("difficulty_level", 0.0)),
            "gt_interval": obj.get("gt_interval", [0.0, 1.0]),
            "type_annotation": obj.get("type_annotation", ""),
            "influence_annotation": obj.get("influence_annotation", ""),
            "reasoning_annotation": obj.get("reasoning_annotation", ""),
            "conclusion_annotation": obj.get("conclusion_annotation", ""),
            "answer_annotation": obj.get("answer_annotation", ""),
            "split": obj.get("split", self.split),
        }
