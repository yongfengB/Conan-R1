"""Load and sample the machine-readable Surv-VAU degradation protocol."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import yaml

from .types import DegradationProfile
from .types import VALID_FACTOR_NAMES


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
    expected_combinations = {
        "single_operator": float(probabilities[1]),
        "cross_category_pair": float(probabilities[2]),
        "one_per_seen_category": float(probabilities[3]),
    }
    if {
        key: float(combinations.get(key, -1.0)) for key in expected_combinations
    } != expected_combinations:
        raise ValueError(
            "combination_distribution must be identical to the declared K distribution"
        )
    operator_order = list(protocol["operator_order"])
    if len(operator_order) != len(set(operator_order)) or set(operator_order) != VALID_FACTOR_NAMES:
        raise ValueError("operator_order must list every operator exactly once")
    if set(protocol["operators"]) != VALID_FACTOR_NAMES:
        raise ValueError("operators must define every supported operator exactly once")
    seen = _flatten_seen_operators(protocol)
    held_out = list(protocol["synthetic_unseen_test_operators"])
    if len(seen) != len(set(seen)) or set(seen) & set(held_out):
        raise ValueError("seen and held-out operators must be disjoint and unique")
    if set(seen) | set(held_out) != VALID_FACTOR_NAMES:
        raise ValueError("seen plus held-out operators must cover the operator vocabulary")
    unseen_modes = protocol["synthetic_unseen_mode_distribution"]
    if set(unseen_modes) != {"held_out_operator", "held_out_combination"} or abs(
        sum(float(value) for value in unseen_modes.values()) - 1.0
    ) > 1e-8:
        raise ValueError("synthetic_unseen_mode_distribution is invalid")
    compatibility = protocol.get("operator_compatibility", {})
    if not set(compatibility).issubset(VALID_FACTOR_NAMES):
        raise ValueError("operator_compatibility contains an unknown operator")
    valid_environments = {"outdoor", "tunnel", "indoor"}
    if any(
        not values or not set(values).issubset(valid_environments)
        for values in compatibility.values()
    ):
        raise ValueError("operator_compatibility contains an invalid environment")

    training_combinations = set()
    categories = protocol["seen_training_operators"]
    category_names = list(categories)
    for factors in categories.values():
        training_combinations.update(frozenset([factor]) for factor in factors)
    for first_index, first in enumerate(category_names):
        for second in category_names[first_index + 1 :]:
            training_combinations.update(
                frozenset([left, right])
                for left in categories[first]
                for right in categories[second]
            )
    training_combinations.update(
        frozenset([first, second, third])
        for first in categories[category_names[0]]
        for second in categories[category_names[1]]
        for third in categories[category_names[2]]
    )
    for combination in protocol["synthetic_unseen_test_combinations"]:
        frozen = frozenset(combination)
        if len(frozen) != len(combination) or not frozen <= set(seen):
            raise ValueError("held-out combinations must contain unique seen operators")
        if frozen in training_combinations:
            raise ValueError(
                f"Declared held-out combination is reachable in training: {combination}"
            )
    return protocol


def validate_profile_compatibility(
    profile: DegradationProfile,
    scene_environment: str,
    protocol: Dict,
) -> None:
    """Reject environmental operators outside their declared scene domain."""
    compatibility = protocol.get("operator_compatibility", {})
    for factor, _ in profile.factors:
        allowed = compatibility.get(factor)
        if allowed is not None and scene_environment not in allowed:
            raise ValueError(
                f"{factor} is incompatible with {scene_environment}; "
                f"allowed environments are {allowed}."
            )


def degradation_combination_label(
    profile: DegradationProfile, protocol: Dict
) -> str:
    """Return the protocol-defined combination class for one exact profile."""
    if not profile.factors:
        return "none"
    factors = [name for name, _ in profile.factors]
    categories = protocol["seen_training_operators"]
    category_by_operator = {
        operator: category
        for category, operators in categories.items()
        for operator in operators
    }
    if profile.domain == "synthetic_unseen":
        if set(factors) & set(protocol["synthetic_unseen_test_operators"]):
            return "held_out_operator"
        if frozenset(factors) in {
            frozenset(values)
            for values in protocol["synthetic_unseen_test_combinations"]
        }:
            return "held_out_combination"
        raise ValueError("Synthetic-unseen profile is not declared by the protocol.")
    if profile.domain != "synthetic_seen":
        raise ValueError("Only synthetic profiles have a combination label.")
    if any(factor not in category_by_operator for factor in factors):
        raise ValueError("Synthetic-seen profile contains a held-out operator.")
    represented = [category_by_operator[factor] for factor in factors]
    if len(factors) == 1:
        return "single_operator"
    if len(factors) == 2 and len(set(represented)) == 2:
        return "cross_category_pair"
    if len(factors) == 3 and len(set(represented)) == 3:
        return "one_per_seen_category"
    raise ValueError("Synthetic-seen profile violates the K/category contract.")


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
    scene_environment: str = "outdoor",
) -> DegradationProfile:
    """Sample one profile according to the explicit K/combination contract."""
    if float(severity) not in {float(value) for value in protocol["severity_levels"]}:
        raise ValueError("severity is not part of the protocol")
    if domain not in {"synthetic_seen", "synthetic_unseen"}:
        raise ValueError("profiles are sampled only for synthetic domains")
    compatibility = protocol.get("operator_compatibility", {})

    def compatible(name: str) -> bool:
        return scene_environment in compatibility.get(
            name, ["outdoor", "tunnel", "indoor"]
        )

    k_values = np.array(sorted(int(key) for key in protocol["factor_count_distribution"]))
    k_probabilities = np.array(
        [float(protocol["factor_count_distribution"][int(key)]) for key in k_values]
    )
    k = int(rng.choice(k_values, p=k_probabilities))

    categories = protocol["seen_training_operators"]
    if domain == "synthetic_unseen":
        held_out = list(protocol["synthetic_unseen_test_operators"])
        remaining = [
            name for name in _flatten_seen_operators(protocol) if compatible(name)
        ]
        combinations = [
            list(values)
            for values in protocol["synthetic_unseen_test_combinations"]
            if len(values) == k and all(compatible(name) for name in values)
        ]
        modes = protocol["synthetic_unseen_mode_distribution"]
        use_combination = bool(combinations) and str(
            rng.choice(
                ["held_out_operator", "held_out_combination"],
                p=[
                    float(modes["held_out_operator"]),
                    float(modes["held_out_combination"]),
                ],
            )
        ) == "held_out_combination"
        if use_combination:
            factors = list(combinations[int(rng.integers(0, len(combinations)))])
        else:
            factors = [str(rng.choice(held_out))]
        if k > len(factors):
            factors.extend(
                str(value)
                for value in rng.choice(
                    [name for name in remaining if name not in factors],
                    size=k - len(factors),
                    replace=False,
                )
            )
    elif k == 1:
        choices = [
            name for name in _flatten_seen_operators(protocol) if compatible(name)
        ]
        factors = [str(rng.choice(choices))]
    else:
        category_names = list(categories)
        compatible_categories = {
            category: [name for name in names if compatible(name)]
            for category, names in categories.items()
        }
        compatible_categories = {
            category: names
            for category, names in compatible_categories.items()
            if names
        }
        selected_categories = list(
            rng.choice(
                list(compatible_categories),
                size=min(k, len(compatible_categories)),
                replace=False,
            )
        )
        factors = [
            str(rng.choice(compatible_categories[name]))
            for name in selected_categories
        ]

    if len(factors) != k:
        raise ValueError(
            f"Scene {scene_environment!r} cannot realize K={k} compatible operators."
        )

    profile = DegradationProfile(
        factors=[(factor, float(severity)) for factor in factors],
        difficulty_level=float(severity),
        domain=domain,
    )
    validate_profile_compatibility(profile, scene_environment, protocol)
    return profile
