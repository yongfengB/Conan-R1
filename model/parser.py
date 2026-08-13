"""Structured output parser for Conan-R1 five-block format."""
from __future__ import annotations
import re
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


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


@dataclass(frozen=True)
class AnswerFields:
    """Exactly the benchmark fields activated by a sample task mask."""

    event_type: Optional[str]
    interval: Optional[Tuple[float, float]]


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
        # An optional block may be absent, but an explicitly emitted empty
        # block is malformed.  Otherwise empty required fields can receive
        # task rewards through downstream defaults.
        if not content:
            return None
        extracted[field_name] = content
        spans.append((start, end))

    # Blocks must be non-overlapping and appear in the declared order.
    if any(
        previous_end > current_start
        for (_, previous_end), (current_start, _) in zip(spans, spans[1:])
    ):
        return None

    # Fixed serialization means that no free text, duplicated markers, or
    # prompt-injection residue may occur outside the declared blocks.
    cursor = 0
    for start, end in spans:
        if text[cursor:start].strip():
            return None
        cursor = end
    if text[cursor:].strip():
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
    matches = []
    for pattern in _EVENT_PATTERNS:
        matches.extend(re.finditer(pattern, answer_text, re.IGNORECASE))
    if len(matches) != 1:
        return None
    label = matches[0].group(1).strip()
    return label or None


def extract_degradation_profile(
    type_text: str,
) -> Optional[List[Tuple[str, float]]]:
    """Parse the deterministic ``<TYPE>`` profile grammar.

    Clean clips use the literal ``none``.  Non-clean clips use one or more
    ``factor_name:severity`` entries separated by semicolons or commas, with
    finite severities in ``[0, 1]``.  Unknown factor names remain structurally
    valid so that ``r_d`` can penalize them as false positives.
    """
    normalized = type_text.strip()
    if normalized.lower() == "none":
        return []
    if not normalized:
        return None
    entries = [
        entry.strip()
        for entry in normalized.replace("\n", ";").replace(",", ";").split(";")
        if entry.strip()
    ]
    if not entries or any(entry.lower() == "none" for entry in entries):
        return None
    profile: List[Tuple[str, float]] = []
    seen_factors = set()
    for entry in entries:
        if entry.count(":") != 1:
            return None
        name, severity_text = entry.split(":", 1)
        factor = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
        try:
            severity = float(severity_text.strip())
        except ValueError:
            return None
        if not factor or not math.isfinite(severity) or not 0.0 <= severity <= 1.0:
            return None
        if factor in seen_factors:
            return None
        seen_factors.add(factor)
        profile.append((factor, severity))
    return profile


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


_EVENT_FIELD = r"event_type\s*:\s*([^;\[\]\r\n]+?)"
_INTERVAL_FIELD = (
    r"interval\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*"
    r"(\d+(?:\.\d+)?)\s*\]"
)


def parse_answer_fields(
    answer_text: str,
    *,
    event_active: bool,
    temporal_active: bool,
    duration_sec: Optional[float] = None,
) -> Optional[AnswerFields]:
    """Parse the task-conditioned, closed ``<ANSWER>`` grammar.

    Only fields activated by the manifest task mask may be emitted.  This is
    deliberately stricter than the compatibility extractors above: it rejects
    prose, aliases for field names, repeated fields and answers that disclose
    an inactive target.  Temporal endpoints are expressed in clip seconds.
    """
    if not event_active and not temporal_active:
        raise ValueError("At least one ANSWER field must be active.")
    if duration_sec is not None and (
        not math.isfinite(float(duration_sec)) or float(duration_sec) <= 0.0
    ):
        raise ValueError("duration_sec must be finite and positive.")

    if event_active and temporal_active:
        pattern = rf"{_EVENT_FIELD}\s*;\s*{_INTERVAL_FIELD}"
    elif event_active:
        pattern = _EVENT_FIELD
    else:
        pattern = _INTERVAL_FIELD
    match = re.fullmatch(pattern, answer_text.strip(), re.IGNORECASE)
    if match is None:
        return None

    if event_active:
        label = " ".join(match.group(1).split())
        if not label:
            return None
        interval_groups = (2, 3)
    else:
        label = None
        interval_groups = (1, 2)

    interval = None
    if temporal_active:
        start = float(match.group(interval_groups[0]))
        end = float(match.group(interval_groups[1]))
        if not (math.isfinite(start) and math.isfinite(end) and 0.0 <= start < end):
            return None
        if duration_sec is not None and end > float(duration_sec) + 1e-6:
            return None
        interval = (start, end)
    return AnswerFields(event_type=label, interval=interval)
