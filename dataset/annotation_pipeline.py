"""Five-stage structured annotation pipeline for Surv-VAU."""
from __future__ import annotations
import logging
from typing import Any, List, Tuple

import numpy as np

from .types import DegradationProfile, DegradedClip

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_INFLUENCE_PROMPT = (
    "You are analyzing a traffic surveillance video clip. "
    "The following observation-degradation factors have been applied: {factors}. "
    "Describe how these factors affect the reliability of visual evidence "
    "and what compensatory cues should be considered during interpretation. "
    "Be concise and specific."
)

_REASONING_PROMPT = (
    "You are analyzing a traffic surveillance video clip. "
    "Observation degradation: {factors}. "
    "Influence on evidence: {influence}. "
    "Provide a step-by-step evidence-grounded explanation that is consistent with "
    "both the degraded visual evidence and the identified observation condition."
)

_CONCLUSION_PROMPT = (
    "Based on the following reasoning chain, provide a compact event-level "
    "judgment about the traffic anomaly:\n{reasoning}"
)

_COMPACTNESS_PROMPT = (
    "Rewrite the following reasoning chain to be {length_instruction}. "
    "Preserve all key evidence-to-judgment steps but {action}.\n\nReasoning:\n{reasoning}"
)


def _format_factors(profile: DegradationProfile) -> str:
    if not profile.factors:
        return "none"
    return ", ".join(f"{name} (severity={sev:.1f})" for name, sev in profile.factors)


# ---------------------------------------------------------------------------
# Annotation generation functions
# ---------------------------------------------------------------------------

def generate_influence(
    clip: DegradedClip,
    profile: DegradationProfile,
    model_q: Any,
) -> str:
    """Generate an <INFLUENCE> annotation using Qwen2.5-VL-Instruct.

    Args:
        clip: The degraded video clip.
        profile: Applied degradation profile.
        model_q: Qwen2.5-VL annotator exposing a ``generate`` method.

    Returns:
        Influence annotation string.
    """
    factors_str = _format_factors(profile)
    prompt = _INFLUENCE_PROMPT.format(factors=factors_str)
    return model_q.generate(clip.frames, prompt)


def generate_reasoning(
    clip: DegradedClip,
    profile: DegradationProfile,
    influence: str,
    model_q: Any,
) -> Tuple[str, str]:
    """Generate <REASONING> and <CONCLUSION> annotations.

    Args:
        clip: The degraded video clip.
        profile: Applied degradation profile.
        influence: Previously generated influence annotation.
        model_q: Annotator model.

    Returns:
        Tuple of (reasoning, conclusion) strings.
    """
    factors_str = _format_factors(profile)
    reasoning_prompt = _REASONING_PROMPT.format(
        factors=factors_str, influence=influence
    )
    reasoning = model_q.generate(clip.frames, reasoning_prompt)

    conclusion_prompt = _CONCLUSION_PROMPT.format(reasoning=reasoning)
    conclusion = model_q.generate(clip.frames, conclusion_prompt)

    return reasoning, conclusion


def compute_aggregated_severity(profile: DegradationProfile) -> float:
    """Compute mean severity across all active factors.

    s_bar = (1/K) * sum(s_k for k in 1..K)

    Returns 0.0 if no factors are active.
    """
    return profile.aggregated_score()


def compute_aggregated_difficulty(profile: DegradationProfile) -> float:
    """Backward-compatible alias for :func:`compute_aggregated_severity`."""
    return compute_aggregated_severity(profile)


def adjust_compactness(
    reasoning: str,
    s_bar: float,
    model_q: Any,
) -> str:
    """Adjust reasoning length based on aggregated degradation severity.

    Higher s_bar → more detailed reasoning.
    Lower s_bar  → more concise reasoning.

    Args:
        reasoning: Original reasoning annotation.
        s_bar: Aggregated degradation-severity score in [0, 1].
        model_q: Annotator model.

    Returns:
        Compactness-adjusted reasoning string.
    """
    if s_bar >= 0.6:
        length_instruction = "more detailed and thorough"
        action = "add compensatory reasoning steps for degraded evidence"
    elif s_bar >= 0.3:
        length_instruction = "moderately detailed"
        action = "keep all key evidence-to-judgment steps without excessive elaboration"
    else:
        length_instruction = "concise"
        action = "remove redundant steps and keep only the essential explanation"

    prompt = _COMPACTNESS_PROMPT.format(
        length_instruction=length_instruction,
        action=action,
        reasoning=reasoning,
    )
    return model_q.generate([], prompt)


def derive_reasoning_target_length(s_bar: float) -> int:
    """Map severity to an explicit, auditable effective-token target.

    This deterministic policy is independent of the annotator's realized
    sentence length.  It is a training target, not evidence that longer
    reasoning is intrinsically better.
    """
    bounded = max(0.0, min(1.0, float(s_bar)))
    return int(round(32 + 96 * bounded))
