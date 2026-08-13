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
from .video_utils import native_motion_pairs, sample_anchor_motion_frames

logger = logging.getLogger(__name__)

VALID_SPLITS = {"sft_train", "rl_train", "val", "test"}
BLOCK_DESCRIPTIONS = {
    "TYPE": "degradation_factor:severity entries",
    "INFLUENCE": "effect on observable evidence",
    "REASONING": "evidence-grounded explanation",
    "CONCLUSION": "compact event judgment",
    "ANSWER": "the active task fields only",
}
BLOCK_ORDER = tuple(BLOCK_DESCRIPTIONS)


def structured_output_instruction(enabled_blocks=BLOCK_ORDER) -> str:
    """Return the exact prompt contract for a full or structural-ablation run."""
    enabled = tuple(str(block).upper() for block in enabled_blocks)
    if not enabled or "ANSWER" not in enabled:
        raise ValueError("At least ANSWER must be enabled.")
    if any(block not in BLOCK_DESCRIPTIONS for block in enabled):
        raise ValueError("Unknown structured-output block.")
    if enabled != tuple(block for block in BLOCK_ORDER if block in enabled):
        raise ValueError("Enabled blocks must preserve the canonical order.")
    serialization = "".join(
        f"<{block}>{BLOCK_DESCRIPTIONS[block]}<{block}_END>"
        for block in enabled
    )
    return (
        f"Return exactly these blocks in order: {serialization}. "
        "If event_active is true, include exactly one event_type field. If "
        "temporal_active is true, include exactly one interval field."
    )


STRUCTURED_OUTPUT_INSTRUCTION = structured_output_instruction()


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
        enabled_blocks=BLOCK_ORDER,
        force_joint_task: bool = False,
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
        self.enabled_blocks = tuple(str(block).upper() for block in enabled_blocks)
        self.force_joint_task = bool(force_joint_task)
        # Validate once at construction rather than failing mid-epoch.
        structured_output_instruction(self.enabled_blocks)
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

        training_request = bool(set(self.splits) & {"sft_train", "rl_train"})
        if training_request:
            missing_pairs = [
                obj["video_id"]
                for obj in self.samples
                if not obj.get("source_video_file")
                or not (self.data_dir / obj["source_video_file"]).is_file()
            ]
            if missing_pairs:
                raise FileNotFoundError(
                    f"Training requires aligned source videos; missing {len(missing_pairs)} pairs."
                )

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
            frames, motion_frames, motion_pairs = sample_anchor_motion_frames(
                str(video_path), n=self.num_frames, size=(self.frame_size, self.frame_size)
            )
        else:
            # Metadata-only callers may opt out of strict video validation.
            frames = [
                np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
                for _ in range(self.num_frames)
            ]
            motion_frames = [frame.copy() for frame in frames]
            motion_pairs = [
                (index, min(index + 1, self.num_frames - 1))
                for index in range(self.num_frames)
            ]

        source_path = (
            self.data_dir / obj["source_video_file"]
            if obj.get("source_video_file")
            else None
        )
        if source_path is not None and source_path.is_file():
            source_frames, source_motion_frames, source_motion_pairs = (
                sample_anchor_motion_frames(
                    str(source_path),
                    n=self.num_frames,
                    size=(self.frame_size, self.frame_size),
                )
            )
        else:
            source_frames = [frame.copy() for frame in frames]
            source_motion_frames = [frame.copy() for frame in motion_frames]
            source_motion_pairs = list(motion_pairs)

        # Convert frames to tensor: (T, C, H, W) float32 in [0, 1]
        frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)
        motion_np = np.stack(motion_frames, axis=0).astype(np.float32) / 255.0
        motion_tensor = torch.from_numpy(motion_np).permute(0, 3, 1, 2)
        source_np = np.stack(source_frames, axis=0).astype(np.float32) / 255.0
        source_tensor = torch.from_numpy(source_np).permute(0, 3, 1, 2)
        source_motion_np = (
            np.stack(source_motion_frames, axis=0).astype(np.float32) / 255.0
        )
        source_motion_tensor = torch.from_numpy(source_motion_np).permute(0, 3, 1, 2)

        task_mask = obj.get("task_mask", {"event": True, "temporal": True})
        if self.force_joint_task:
            task_mask = {"event": True, "temporal": True}
        if not isinstance(task_mask, dict) or not {"event", "temporal"}.issubset(task_mask):
            raise ValueError(f"{obj['video_id']}: task_mask must define event and temporal")
        anchor_indices = [int(first) for first, _ in motion_pairs]
        timestamps = [index / float(obj["fps"]) for index in anchor_indices]
        return {
            "video_id": obj["video_id"],
            "source_video_id": obj["source_video_id"],
            "source_dataset": obj["source_dataset"],
            "scene_environment": obj["scene_environment"],
            "frames": frames_tensor,
            "motion_frames": motion_tensor,
            "source_frames": source_tensor,
            "source_motion_frames": source_motion_tensor,
            "anchor_indices": anchor_indices,
            "anchor_timestamps_sec": timestamps,
            "motion_pair_indices": [list(pair) for pair in motion_pairs],
            "motion_elapsed_sec": [
                (second - first) / float(obj["fps"])
                for first, second in motion_pairs
            ],
            "source_motion_elapsed_sec": [
                (second - first) / float(obj["fps"])
                for first, second in source_motion_pairs
            ],
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
            "task_mask": {
                "event": bool(task_mask["event"]),
                "temporal": bool(task_mask["temporal"]),
            },
            "source_video_file": obj.get("source_video_file"),
            "occlusion_token_mask": obj.get("occlusion_token_mask"),
            "influence_targets": obj.get("influence_targets"),
            "synthesis_metadata": obj.get("synthesis_metadata", {}),
            "duration_sec": float(obj["duration_sec"]),
            "fps": float(obj["fps"]),
            "num_source_frames": int(obj["num_source_frames"]),
            "degradation_domain": obj.get("degradation_domain", "synthetic_seen"),
            "degradation_combination": obj.get(
                "degradation_combination", "single_or_seen"
            ),
            "synthesis_applied": bool(obj.get("synthesis_applied", False)),
            "degradation_protocol": obj.get(
                "degradation_protocol", "source_observation"
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
        timestamps = [
            first / float(obj["fps"])
            for first, _ in native_motion_pairs(
                int(obj["num_source_frames"]), self.num_frames
            )
        ]
        timestamp_text = ", ".join(f"{value:.3f}" for value in timestamps)
        base_prompt = obj.get("prompt", "").strip()
        task_mask = obj.get("task_mask", {"event": True, "temporal": True})
        if self.force_joint_task:
            task_mask = {"event": True, "temporal": True}
        active_fields = []
        if task_mask.get("event", True):
            active_fields.append("event_type: LABEL")
        if task_mask.get("temporal", True):
            active_fields.append("interval: [start_sec, end_sec]")
        answer_contract = "; ".join(active_fields)
        return (
            f"{base_prompt}\n"
            f"Video duration: {duration:.3f} seconds. The {self.num_frames} frames "
            f"are uniformly sampled at seconds [{timestamp_text}]. "
            f"Active ANSWER fields: {answer_contract}. "
            f"{structured_output_instruction(self.enabled_blocks)}"
        )
