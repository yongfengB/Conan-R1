"""Structured output parser for Conan-R1 five-block format."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


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


def _extract_block(text: str, tag: str) -> Optional[Tuple[str, int, int]]:
    """Extract content and the complete span between `<TAG>` markers."""
    pattern = rf"<{tag}>(.*?)<{tag}_END>"
    matches = list(re.finditer(pattern, text, re.DOTALL | re.IGNORECASE))
    if len(matches) != 1:
        return None
    match = matches[0]
    return match.group(1).strip(), match.start(), match.end()


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_structured_output(
    text: str,
    optional_blocks: Iterable[str] = (),
) -> Optional[StructuredOutput]:
    """Parse a model-generated string into a StructuredOutput.

    The output must contain all five blocks in the correct order:
    TYPE → INFLUENCE → REASONING → CONCLUSION → ANSWER.

    Args:
        text: Raw model output string.

    Returns:
        StructuredOutput if all five blocks are present and in order,
        None otherwise (reward will be set to 0.0 by the trainer).
    """
    optional = {block.upper() for block in optional_blocks}
    extracted = {}
    spans = []
    for field_name, tag in _BLOCKS:
        match = _extract_block(text, tag)
        if match is None:
            has_marker = bool(
                re.search(rf"<{tag}(?:_END)?>", text, re.IGNORECASE)
            )
            if tag in optional and not has_marker:
                extracted[field_name] = ""
                continue
            return None
        content, start, end = match
        extracted[field_name] = content
        spans.append((start, end))

    # Blocks must be non-overlapping and appear in the declared order.
    if any(
        previous_end > current_start
        for (_, previous_end), (current_start, _) in zip(spans, spans[1:])
    ):
        return None

    return StructuredOutput(
        type_block=extracted["type_block"],
        influence_block=extracted["influence_block"],
        reasoning_block=extracted["reasoning_block"],
        conclusion_block=extracted["conclusion_block"],
        answer_block=extracted["answer_block"],
        raw_text=text,
    )


_EVENT_PATTERNS = [
    r"(?:event|anomaly)[_\s-]*type\s*[:=]\s*([^;\n,\[\]]+)",
    r"(?:category|label)\s*[:=]\s*([^;\n,\[\]]+)",
]


def extract_event_type(answer_text: str) -> Optional[str]:
    """Extract the explicit categorical event label from ``<ANSWER>``.

    The training manifest should use answers such as
    ``event_type: rear-end collision; interval: [5.0, 12.5]``.  We do not
    infer a category from arbitrary prose because doing so would introduce an
    unreported semantic judge into the reward.
    """
    for pattern in _EVENT_PATTERNS:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            label = match.group(1).strip()
            return label or None
    return None


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
    candidates = []
    for pattern in _INTERVAL_PATTERNS:
        for match in re.finditer(pattern, answer_text, re.IGNORECASE):
            start = float(match.group(1))
            end = float(match.group(2))
            if start < end and start >= 0.0:
                candidates.append((match.span(), (start, end)))
    unique = {
        (span, interval): interval for span, interval in candidates
    }
    if len(unique) != 1:
        return None
    return next(iter(unique.values()))
