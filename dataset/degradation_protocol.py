"""Load and sample the machine-readable Surv-VAU degradation protocol."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import yaml

from .types import DegradationProfile


DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / "configs" / "degradation_protocol.yaml"


def load_degradation_protocol(path: str | Path = DEFAULT_PROTOCOL) -> Dict:
    protocol = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    probabilities = protocol["factor_count_distribution"]
    if abs(sum(float(value) for value in probabilities.values()) - 1.0) > 1e-8:
        raise ValueError("factor_count_distribution must sum to one")
    combinations = protocol["combination_distribution"]
    if abs(sum(float(value) for value in combinations.values()) - 1.0) > 1e-8:
        raise ValueError("combination_distribution must sum to one")
    if sorted(int(key) for key in probabilities) != [1, 2, 3]:
        raise ValueError("the protocol must define K in {1, 2, 3}")
    return protocol


def _flatten_seen_operators(protocol: Dict) -> List[str]:
    return [
        operator
        for operators in protocol["seen_training_operators"].values()
        for operator in operators
    ]


def sample_degradation_profile(
    rng: np.random.Generator,
    severity: float,
    protocol: Dict,
    domain: str = "synthetic_seen",
) -> DegradationProfile:
    """Sample one profile according to the explicit K/combination contract."""
    if float(severity) not in {float(value) for value in protocol["severity_levels"]}:
        raise ValueError("severity is not part of the protocol")
    if domain not in {"synthetic_seen", "synthetic_unseen"}:
        raise ValueError("profiles are sampled only for synthetic domains")

    k_values = np.array(sorted(int(key) for key in protocol["factor_count_distribution"]))
    k_probabilities = np.array(
        [float(protocol["factor_count_distribution"][int(key)]) for key in k_values]
    )
    k = int(rng.choice(k_values, p=k_probabilities))

    categories = protocol["seen_training_operators"]
    if domain == "synthetic_unseen":
        held_out = list(protocol["synthetic_unseen_test_operators"])
        remaining = _flatten_seen_operators(protocol)
        factors = [str(rng.choice(held_out))]
        if k > 1:
            factors.extend(
                str(value)
                for value in rng.choice(remaining, size=k - 1, replace=False)
            )
    elif k == 1:
        factors = [str(rng.choice(_flatten_seen_operators(protocol)))]
    else:
        category_names = list(categories)
        selected_categories = list(
            rng.choice(category_names, size=min(k, len(category_names)), replace=False)
        )
        factors = [str(rng.choice(categories[name])) for name in selected_categories]

    return DegradationProfile(
        factors=[(factor, float(severity)) for factor in factors],
        difficulty_level=float(severity),
        domain=domain,
    )
