#!/usr/bin/env python3
"""Command-line inference script for Conan-R1.

Usage:
    python scripts/infer.py \\
        --video path/to/video.mp4 \\
        --checkpoint checkpoints/grpo_full \\
        [--prompt "Describe the traffic anomaly..."] \\
        [--output result.json]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Describe the traffic anomaly event shown in this surveillance video clip "
    "and identify its temporal boundaries [start_sec, end_sec]."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conan-R1 inference")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint directory")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Task prompt")
    parser.add_argument("--output", default=None, help="Optional path to save JSON output")
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate video path
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    # Lazy imports (avoid loading torch/transformers at import time)
    try:
        from dataset.video_utils import (
            probe_video,
            sample_anchor_motion_frames,
        )
        from dataset.dataset import structured_output_instruction
        from dataset.types import VideoLoadError
        from model.conan_r1 import ConanR1Model, LoRAConfig
        from scripts._common import load_core_protocol
        from model.parser import parse_answer_fields, parse_structured_output
        from model.reliability_pathway import ReliabilityPathwayConfig
        from scripts._common import resolve_device
    except ImportError as e:
        print(f"ERROR: Import failed — {e}", file=sys.stderr)
        sys.exit(1)

    core_protocol = load_core_protocol(Path(args.checkpoint))
    preprocessing = core_protocol["motion_preprocessing"]
    anchors = int(preprocessing["anchors"])
    frame_size = int(preprocessing["frame_size"][0])
    native_offset = int(preprocessing["native_frame_offset"])

    # Load frames with the checkpoint-bound motion preprocessing.
    try:
        fps, frame_count, duration_sec = probe_video(str(video_path))
        frames, motion_frames, motion_pairs = sample_anchor_motion_frames(
            str(video_path),
            n=anchors,
            size=(frame_size, frame_size),
            offset=native_offset,
        )
    except VideoLoadError as e:
        print(f"ERROR: Cannot read video — {e}", file=sys.stderr)
        sys.exit(1)

    # Load model
    logger.info("Loading model from checkpoint: %s", args.checkpoint)
    model = ConanR1Model(
        lora_config=LoRAConfig(),
        base_model=core_protocol["base_model"],
        base_model_revision=core_protocol["base_model_revision"],
        device=resolve_device(args.device),
        reliability_config=ReliabilityPathwayConfig(
            **core_protocol["reliability_config"]
        ),
        motion_v_max=float(core_protocol["motion_v_max"]),
        degradation_factor_names=core_protocol["degradation_factor_names"],
        motion_flow_parameters=core_protocol["motion_flow_parameters"],
        motion_frame_size=frame_size,
        motion_native_offset=native_offset,
    )
    model.load_core(args.checkpoint, is_trainable=False)

    # Generate
    logger.info("Generating structured output...")
    timestamps = [first / fps for first, _ in motion_pairs]
    temporal_prompt = (
        f"{args.prompt}\nVideo duration: {duration_sec:.3f} seconds. The {anchors} "
        "frames are uniformly sampled at seconds ["
        + ", ".join(f"{value:.3f}" for value in timestamps)
        + "]. Report temporal boundaries in seconds using interval: "
        f"[start_sec, end_sec]. {structured_output_instruction()}"
    )
    raw_output = model.generate(
        frames,
        temporal_prompt,
        max_new_tokens=args.max_new_tokens,
        motion_frames=motion_frames,
        elapsed_seconds=[(second - first) / fps for first, second in motion_pairs],
        timestamps=timestamps,
    )

    # Parse
    parsed = parse_structured_output(raw_output)
    if parsed is None:
        print("WARNING: Output format is invalid — could not parse five-block structure.")
        result = {"raw_output": raw_output, "parsed": None}
    else:
        answer_fields = parse_answer_fields(
            parsed.answer_block,
            event_active=True,
            temporal_active=True,
            duration_sec=duration_sec,
        )
        if answer_fields is None:
            print("WARNING: ANSWER does not satisfy the strict joint-task grammar.")
            result = {"raw_output": raw_output, "parsed": None}
            parsed = None
    if parsed is not None:
        interval = answer_fields.interval
        result = {
            "type": parsed.type_block,
            "influence": parsed.influence_block,
            "reasoning": parsed.reasoning_block,
            "conclusion": parsed.conclusion_block,
            "answer": parsed.answer_block,
            "temporal_interval": list(interval) if interval else None,
            "protocol": {
                "fps": fps,
                "num_source_frames": frame_count,
                "duration_sec": duration_sec,
                "sampled_timestamps_sec": timestamps,
                "native_motion_pairs": motion_pairs,
                "frame_size": [224, 224],
            },
        }
        print("\n=== Conan-R1 Output ===")
        for key, val in result.items():
            print(f"\n[{key.upper()}]\n{val}")

    # Save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("Output saved to %s", args.output)


if __name__ == "__main__":
    main()
