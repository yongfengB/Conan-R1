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
    "The following observation-difficulty factors have been applied: {factors}. "
    "Describe how these factors affect the reliability of visual evidence "
    "and what compensatory cues should be considered during interpretation. "
    "Be concise and specific."
)

_REASONING_PROMPT = (
    "You are analyzing a traffic surveillance video clip. "
    "Observation difficulty: {factors}. "
    "Influence on evidence: {influence}. "
    "Provide a step-by-step causal reasoning chain that is consistent with "
    "both the degraded visual evidence and the identified observation condition."
)

_CONCLUSION_PROMPT = (
    "Based on the following reasoning chain, provide a compact event-level "
    "judgment about the traffic anomaly:\n{reasoning}"
)

_ANSWER_PROMPT = (
    "Based on the following conclusion about a traffic anomaly event, "
    "provide a benchmark-compatible answer that includes: "
    "(1) the anomaly type, (2) the temporal interval [start_sec, end_sec], "
    "and (3) a brief causal explanation.\nConclusion: {conclusion}"
)

_COMPACTNESS_PROMPT = (
    "Rewrite the following reasoning chain to be {length_instruction}. "
    "Preserve all key causal steps but {action}.\n\nReasoning:\n{reasoning}"
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
    """Generate <INFLUENCE> annotation using Qwen2.5-3B-Instruct.

    Args:
        clip: The degraded video clip.
        profile: Applied degradation profile.
        model_q: Annotator model (Qwen2.5-3B-Instruct) with a .generate() method.

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


def generate_answer(
    clip: DegradedClip,
    conclusion: str,
    model_q: Any,
) -> str:
    """Generate <ANSWER> annotation.

    Args:
        clip: The degraded video clip.
        conclusion: Previously generated conclusion annotation.
        model_q: Annotator model.

    Returns:
        Answer annotation string.
    """
    prompt = _ANSWER_PROMPT.format(conclusion=conclusion)
    return model_q.generate(clip.frames, prompt)


def compute_aggregated_difficulty(profile: DegradationProfile) -> float:
    """Compute mean severity across all active factors.

    s_bar = (1/K) * sum(s_k for k in 1..K)

    Returns 0.0 if no factors are active.
    """
    return profile.aggregated_score()


def adjust_compactness(
    reasoning: str,
    s_bar: float,
    model_q: Any,
) -> str:
    """Adjust reasoning length based on aggregated difficulty score.

    Higher s_bar → more detailed reasoning.
    Lower s_bar  → more concise reasoning.

    Args:
        reasoning: Original reasoning annotation.
        s_bar: Aggregated difficulty score in [0, 1].
        model_q: Annotator model.

    Returns:
        Compactness-adjusted reasoning string.
    """
    if s_bar >= 0.6:
        length_instruction = "more detailed and thorough"
        action = "add compensatory reasoning steps for degraded evidence"
    elif s_bar >= 0.3:
        length_instruction = "moderately detailed"
        action = "keep all key causal steps without excessive elaboration"
    else:
        length_instruction = "concise"
        action = "remove redundant steps and keep only the essential causal chain"

    prompt = _COMPACTNESS_PROMPT.format(
        length_instruction=length_instruction,
        action=action,
        reasoning=reasoning,
    )
    return model_q.generate([], prompt)
