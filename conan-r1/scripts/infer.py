#!/usr/bin/env python3
"""Command-line inference script for Conan-R1.

Usage:
    python scripts/infer.py \\
        --video path/to/video.mp4 \\
        --checkpoint checkpoints/grpo \\
        [--prompt "Describe the traffic anomaly..."] \\
        [--output result.json]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

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
        from dataset.video_utils import sample_frames
        from dataset.types import VideoLoadError
        from model.conan_r1 import ConanR1Model, LoRAConfig
        from model.parser import parse_structured_output, extract_temporal_interval
    except ImportError as e:
        print(f"ERROR: Import failed — {e}", file=sys.stderr)
        sys.exit(1)

    # Load frames
    try:
        frames = sample_frames(str(video_path), n=25, size=(448, 448))
    except VideoLoadError as e:
        print(f"ERROR: Cannot read video — {e}", file=sys.stderr)
        sys.exit(1)

    # Load model
    logger.info("Loading model from checkpoint: %s", args.checkpoint)
    model = ConanR1Model(lora_config=LoRAConfig())
    model.load_lora(args.checkpoint)

    # Generate
    logger.info("Generating structured output...")
    raw_output = model.generate(frames, args.prompt, max_new_tokens=args.max_new_tokens)

    # Parse
    parsed = parse_structured_output(raw_output)
    if parsed is None:
        print("WARNING: Output format is invalid — could not parse five-block structure.")
        result = {"raw_output": raw_output, "parsed": None}
    else:
        interval = extract_temporal_interval(parsed.answer_block)
        result = {
            "type": parsed.type_block,
            "influence": parsed.influence_block,
            "reasoning": parsed.reasoning_block,
            "conclusion": parsed.conclusion_block,
            "answer": parsed.answer_block,
            "temporal_interval": list(interval) if interval else None,
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
