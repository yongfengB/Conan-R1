"""Audit that every advertised experiment variant is a complete runnable YAML."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.materialize_experiments import materialize


ROOT = Path(__file__).resolve().parents[1]


def test_every_manuscript_control_materializes_to_complete_hashed_yaml(tmp_path):
    hashes = materialize(
        ROOT / "experiments" / "experiment_matrix.yaml",
        tmp_path,
        overwrite=True,
    )
    names = {path.stem for path in tmp_path.glob("*.yaml")}
    assert {
        "global_grpo",
        "grpo_only",
        "stage1_plus_motion",
        "stage1_plus_reliability_supervision",
        "stage1_plus_reliability_fusion",
        "stage1_plus_event_pooling",
        "stage1_plus_temporal_reliability",
        "uniform_grpo",
        "without_rd",
        "without_re",
        "without_rt",
        "without_rl",
        "conan_r1",
        "data_sft",
        "update_sft",
        "lora_grpo",
        "pathway_grpo",
        "weights_rd_040",
        "reliability_online_target_sft",
        "reliability_online_target_grpo",
        "slot_stage1_only",
    }.issubset(names)
    manifest = json.loads((tmp_path / "SHA256SUMS.json").read_text())
    assert manifest == hashes
    for path in tmp_path.glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        assert {"model", "training", "auxiliary_losses", "data", "output", "experiment"}.issubset(config)
        if "reward" in config:
            reward = config["reward"]
            assert sum(reward[key] for key in ("w_d", "w_e", "w_t", "w_l")) == pytest.approx(1.0)
        assert config["model"]["base_model_revision"]


def test_stage1_cumulative_configs_toggle_real_forward_switches(tmp_path):
    materialize(ROOT / "experiments" / "experiment_matrix.yaml", tmp_path, True)
    motion = yaml.safe_load((tmp_path / "stage1_plus_motion.yaml").read_text())
    supervision = yaml.safe_load(
        (tmp_path / "stage1_plus_reliability_supervision.yaml").read_text()
    )
    fusion = yaml.safe_load(
        (tmp_path / "stage1_plus_reliability_fusion.yaml").read_text()
    )
    event = yaml.safe_load(
        (tmp_path / "stage1_plus_event_pooling.yaml").read_text()
    )
    temporal = yaml.safe_load(
        (tmp_path / "stage1_plus_temporal_reliability.yaml").read_text()
    )
    assert motion["auxiliary_losses"]["lambda_q_sft"] == 0.0
    assert supervision["auxiliary_losses"]["lambda_q_sft"] == 1.0
    assert supervision["model"]["pathway"]["use_reliability_fusion"] is False
    assert fusion["model"]["pathway"]["use_reliability_fusion"] is True
    assert fusion["model"]["pathway"]["use_event_aware_pooling"] is False
    assert event["model"]["pathway"]["use_event_aware_pooling"] is True
    assert event["model"]["pathway"]["use_temporal_reliability"] is False
    assert "pathway" not in temporal["model"]


def test_matched_grpo_controls_separate_pathway_use_and_update_scope(tmp_path):
    materialize(ROOT / "experiments" / "experiment_matrix.yaml", tmp_path, True)
    global_grpo = yaml.safe_load((tmp_path / "global_grpo.yaml").read_text())
    lora_grpo = yaml.safe_load((tmp_path / "lora_grpo.yaml").read_text())
    pathway_grpo = yaml.safe_load((tmp_path / "pathway_grpo.yaml").read_text())
    assert global_grpo["model"]["pathway_mode"] == "none"
    assert global_grpo["model"]["policy_scope"] == "lora_only"
    assert lora_grpo["model"]["pathway_mode"] == "reliability"
    assert lora_grpo["model"]["policy_scope"] == "lora_only"
    assert pathway_grpo["model"]["policy_scope"] == "full"
    assert pathway_grpo["auxiliary_losses"]["preserve_during_grpo"] is False


def test_structured_sft_is_a_true_lora_only_baseline():
    config = yaml.safe_load(
        (ROOT / "configs" / "structured_sft_config.yaml").read_text()
    )
    assert config["model"]["policy_scope"] == "lora_only"
    assert set(config["training"]["enabled_blocks"]) == {
        "TYPE", "INFLUENCE", "REASONING", "CONCLUSION", "ANSWER"
    }
    assert all(value == 0.0 for value in config["auxiliary_losses"].values())
