"""Structured output parser for Conan-R1 five-block format."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class StructuredOutput:
    """Parsed five-block output from Conan-R1."""
    type_block: str        # content of <TYPE>...</TYPE_END>
    influence_block: str   # content of <INFLUENCE>...</INFLUENCE_END>
    reasoning_block: str   # content of <REASONING>...</REASONING_END>
    conclusion_block: str  # content of <CONCLUSION>...</CONCLUSION_END>
    answer_block: str      # content of <ANSWER>...</ANSWER_END>
    raw_text: str


# ---------------------------------------------------------------------------
# Block definitions (ordered)
# ---------------------------------------------------------------------------

_BLOCKS = [
    ("type_block",       "TYPE"),
    ("influence_block",  "INFLUENCE"),
    ("reasoning_block",  "REASONING"),
    ("conclusion_block", "CONCLUSION"),
    ("answer_block",     "ANSWER"),
]


def _extract_block(text: str, tag: str) -> Optional[str]:
    """Extract content between <TAG> and <TAG_END>."""
    pattern = rf"<{tag}>(.*?)<{tag}_END>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_structured_output(text: str) -> Optional[StructuredOutput]:
    """Parse a model-generated string into a StructuredOutput.

    The output must contain all five blocks in the correct order:
    TYPE → INFLUENCE → REASONING → CONCLUSION → ANSWER.

    Args:
        text: Raw model output string.

    Returns:
        StructuredOutput if all five blocks are present and in order,
        None otherwise (reward will be set to 0.0 by the trainer).
    """
    extracted = {}
    for field_name, tag in _BLOCKS:
        content = _extract_block(text, tag)
        if content is None:
            return None
        extracted[field_name] = content

    # Verify ordering: each block's start position must be increasing
    positions = []
    for _, tag in _BLOCKS:
        pattern = rf"<{tag}>"
        match = re.search(pattern, text, re.IGNORECASE)
        if match is None:
            return None
        positions.append(match.start())

    if positions != sorted(positions):
        return None

    return StructuredOutput(
        type_block=extracted["type_block"],
        influence_block=extracted["influence_block"],
        reasoning_block=extracted["reasoning_block"],
        conclusion_block=extracted["conclusion_block"],
        answer_block=extracted["answer_block"],
        raw_text=text,
    )


# ---------------------------------------------------------------------------
# Temporal interval extraction
# ---------------------------------------------------------------------------

_INTERVAL_PATTERNS = [
    # [start, end] or [start_sec, end_sec]
    r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]",
    # start_sec: X, end_sec: Y
    r"start[_\s]sec[:\s]+(\d+(?:\.\d+)?)[^\d]+end[_\s]sec[:\s]+(\d+(?:\.\d+)?)",
    # from X to Y seconds
    r"from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s+sec",
    # X-Y s
    r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*s(?:ec)?",
]


def extract_temporal_interval(
    answer_text: str,
) -> Optional[Tuple[float, float]]:
    """Parse a temporal interval from the ANSWER block.

    Tries multiple regex patterns. Returns None if no valid interval found.

    Args:
        answer_text: Content of the <ANSWER> block.

    Returns:
        (start_sec, end_sec) tuple, or None if parsing fails.
    """
    for pattern in _INTERVAL_PATTERNS:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            start = float(match.group(1))
            end = float(match.group(2))
            if start < end and start >= 0.0:
                return (start, end)
    return None
