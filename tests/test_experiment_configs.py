"""Audit that every advertised experiment variant is a complete runnable YAML."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.materialize_experiments import materialize


ROOT = Path(__file__).resolve().parents[1]


def test_materialized_reward_controls_are_complete_and_hashed(tmp_path):
    hashes = materialize(
        ROOT / "experiments" / "experiment_matrix.yaml",
        tmp_path,
        overwrite=True,
    )
    assert set(path.name for path in tmp_path.glob("*.yaml")) == {
        "conan_r1.yaml",
        "fixed_length_70.yaml",
        "without_rd.yaml",
        "without_re.yaml",
        "without_rl.yaml",
        "without_rt.yaml",
    }
    manifest = json.loads((tmp_path / "SHA256SUMS.json").read_text())
    assert manifest == hashes
    for path in tmp_path.glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        assert set(config) == {
            "model", "training", "auxiliary_losses", "reward",
            "ablation", "data", "output",
        }
        reward = config["reward"]
        assert sum(reward[key] for key in ("w_d", "w_e", "w_t", "w_l")) == pytest.approx(1.0)
        assert config["model"]["base_model_revision"]


def test_structured_sft_is_a_true_lora_only_baseline():
    config = yaml.safe_load(
        (ROOT / "configs" / "structured_sft_config.yaml").read_text()
    )
    assert config["model"]["policy_scope"] == "lora_only"
    assert set(config["training"]["enabled_blocks"]) == {
        "TYPE", "INFLUENCE", "REASONING", "CONCLUSION", "ANSWER"
    }
    assert all(value == 0.0 for value in config["auxiliary_losses"].values())
