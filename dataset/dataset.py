"""SurvVAUDataset — PyTorch Dataset for Conan-R1 training and evaluation."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .types import StructuredSample, SEVERITY_LEVELS
from .video_utils import sample_frames, uniform_sample_timestamps

logger = logging.getLogger(__name__)

VALID_SPLITS = {"sft_train", "rl_train", "val", "test"}
STRUCTURED_OUTPUT_INSTRUCTION = (
    "Return exactly these blocks in order: "
    "<TYPE>degradation_factor:severity entries<TYPE_END>"
    "<INFLUENCE>effect on observable evidence<INFLUENCE_END>"
    "<REASONING>evidence-grounded explanation<REASONING_END>"
    "<CONCLUSION>compact event judgment<CONCLUSION_END>"
    "<ANSWER>event_type: LABEL; interval: [start_sec, end_sec]; "
    "explanation: TEXT<ANSWER_END>."
)


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
        split: Union[str, Sequence[str]],
        num_frames: int = 25,
        frame_size: int = 224,
        require_videos: bool = True,
    ) -> None:
        requested_splits = [split] if isinstance(split, str) else list(split)
        if not requested_splits or not set(requested_splits).issubset(VALID_SPLITS):
            raise ValueError(
                f"split must contain only {sorted(VALID_SPLITS)}, got {split!r}"
            )

        self.data_dir = Path(data_dir)
        self.splits = tuple(requested_splits)
        self.split = "+".join(self.splits)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.require_videos = require_videos
        self.samples: List[Dict] = []
        self._load()

    def _load(self) -> None:
        ann_path = self.data_dir / "annotations.jsonl"
        splits_path = self.data_dir / "splits.json"

        if not ann_path.exists():
            raise FileNotFoundError(f"annotations.jsonl not found at {ann_path}")
        if not splits_path.exists():
            raise FileNotFoundError(f"splits.json not found at {splits_path}")

        with open(splits_path, encoding="utf-8") as handle:
            split_map: Dict[str, str] = json.load(handle)

        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                vid = obj.get("video_id", "")
                assigned_split = split_map.get(vid, obj.get("split", ""))
                if assigned_split in self.splits:
                    self.samples.append(obj)

        if self.require_videos:
            missing_videos = [
                obj["video_id"]
                for obj in self.samples
                if not (self.data_dir / "videos" / f"{obj['video_id']}.mp4").is_file()
            ]
            if missing_videos:
                preview = ", ".join(missing_videos[:3])
                suffix = " ..." if len(missing_videos) > 3 else ""
                raise FileNotFoundError(
                    f"Split '{self.split}' is missing {len(missing_videos)} video files "
                    f"(e.g. {preview}{suffix}). Refusing to substitute blank frames."
                )

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
            # Metadata-only callers may opt out of strict video validation.
            frames = [
                np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
                for _ in range(self.num_frames)
            ]

        # Convert frames to tensor: (T, C, H, W) float32 in [0, 1]
        frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)

        return {
            "video_id": obj["video_id"],
            "source_video_id": obj["source_video_id"],
            "source_dataset": obj["source_dataset"],
            "frames": frames_tensor,
            "prompt": self._temporal_prompt(obj),
            "degradation_profile": obj.get("degradation_profile", []),
            "degradation_level": float(
                obj.get("degradation_level", obj.get("difficulty_level", 0.0))
            ),
            # Deprecated compatibility key for early Surv-VAU manifests.
            "difficulty_level": float(
                obj.get("degradation_level", obj.get("difficulty_level", 0.0))
            ),
            "gt_interval": obj.get("gt_interval", [0.0, 1.0]),
            "event_type": obj["event_type"],
            "event_aliases": obj.get("event_aliases", []),
            "reasoning_target_length": int(obj["reasoning_target_length"]),
            "reasoning_target_source": obj["reasoning_target_source"],
            "duration_sec": float(obj["duration_sec"]),
            "fps": float(obj["fps"]),
            "num_source_frames": int(obj["num_source_frames"]),
            "degradation_domain": obj.get("degradation_domain", "synthetic_seen"),
            "degradation_combination": obj.get(
                "degradation_combination", "single_or_seen"
            ),
            "type_annotation": obj.get("type_annotation", ""),
            "influence_annotation": obj.get("influence_annotation", ""),
            "reasoning_annotation": obj.get("reasoning_annotation", ""),
            "conclusion_annotation": obj.get("conclusion_annotation", ""),
            "answer_annotation": obj.get("answer_annotation", ""),
            "answer_references": obj.get(
                "answer_references", [obj.get("answer_annotation", "")]
            ),
            "official_question_id": obj.get("official_question_id"),
            "split": obj.get("split", self.split),
        }

    def _temporal_prompt(self, obj: Dict) -> str:
        """Attach the exact seconds represented by the uniformly sampled frames."""
        duration = float(obj["duration_sec"])
        timestamps = uniform_sample_timestamps(
            int(obj["num_source_frames"]),
            float(obj["fps"]),
            self.num_frames,
        )
        timestamp_text = ", ".join(f"{value:.3f}" for value in timestamps)
        base_prompt = obj.get("prompt", "").strip()
        return (
            f"{base_prompt}\n"
            f"Video duration: {duration:.3f} seconds. The {self.num_frames} frames "
            f"are uniformly sampled at seconds [{timestamp_text}]. Report temporal "
            "boundaries in seconds using interval: [start_sec, end_sec]. "
            f"{STRUCTURED_OUTPUT_INSTRUCTION}"
        )
