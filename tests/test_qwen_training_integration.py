"""Opt-in fixed-revision Qwen tokenizer and one-step training integration.

Run with ``CONAN_RUN_QWEN_INTEGRATION=1 pytest -q
tests/test_qwen_training_integration.py`` in the locked environment.  The test
downloads only the frozen tokenizer; it uses a tiny trainable decoder so the
forward/backward contract can run on CPU without a 3B checkpoint.
"""
from pathlib import Path
from types import SimpleNamespace
from types import MethodType
import os

import pytest
import torch
import yaml


pytestmark = pytest.mark.skipif(
    os.environ.get("CONAN_RUN_QWEN_INTEGRATION") != "1",
    reason="set CONAN_RUN_QWEN_INTEGRATION=1 for the fixed-Qwen integration",
)


QWEN_REVISION = "c747f21f03e7d0792c30766310bd7d8de17eeeb3"
RESPONSE = (
    "<TYPE>motion_blur:0.4<TYPE_END>"
    "<INFLUENCE>motion evidence is weakened<INFLUENCE_END>"
    "<REASONING>Vehicle trajectories converge.<REASONING_END>"
    "<CONCLUSION>A collision occurs.<CONCLUSION_END>"
    "<ANSWER>event_type: collision; interval: [1.0, 2.0]<ANSWER_END>"
)


def _load_tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        revision=QWEN_REVISION,
        use_fast=True,
        trust_remote_code=True,
    )


def _attach(model, adapted, hidden, response, labels):
    from model.conan_r1 import ConanR1Model

    return ConanR1Model._attach_diagnostic_slots(
        model, adapted, hidden, response, labels, strict=True
    )


def test_fixed_qwen_response_slots_and_default_stage_steps():
    pytest.importorskip("peft")
    from model.conan_r1 import ConanR1Model
    from model.qwen_adapter import response_only_labels
    from training.grpo_math import clipped_grpo_loss
    from training.stage_objectives import (
        AuxiliaryLossWeights,
        stage1_loss,
        stage2_loss,
    )

    tokenizer = _load_tokenizer()
    prompt = "The prompt names <TYPE> and <INFLUENCE> as required fields.\n"
    eos = tokenizer.eos_token or ""
    full_encoding = tokenizer(
        prompt + RESPONSE + eos,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = full_encoding["input_ids"]
    labels = response_only_labels(
        tokenizer,
        RESPONSE,
        eos,
        input_ids,
        full_encoding.get("attention_mask"),
    )
    sequence = input_ids[0].tolist()
    isolated_influence = tokenizer(
        "<INFLUENCE>", add_special_tokens=False
    )["input_ids"]
    old_matches = [
        index
        for index in range(len(sequence) - len(isolated_influence) + 1)
        if sequence[index : index + len(isolated_influence)] == isolated_influence
    ]
    assert old_matches == []

    root = Path(__file__).resolve().parents[1]
    sft = yaml.safe_load((root / "configs/sft_config.yaml").read_text())
    grpo = yaml.safe_load((root / "configs/grpo_config.yaml").read_text())
    assert float(sft["auxiliary_losses"]["lambda_c_sft"]) == 0.1
    assert float(grpo["auxiliary_losses"]["lambda_c_rl"]) == 0.1

    width = 12

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(len(tokenizer), width)

        def forward(
            self, input_ids, labels=None, output_hidden_states=False, **kwargs
        ):
            hidden = self.embedding(input_ids)
            logits = hidden @ self.embedding.weight.transpose(0, 1)
            if labels is None:
                loss = hidden.sum() * 0.0
            else:
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )
            return SimpleNamespace(
                loss=loss,
                logits=logits,
                hidden_states=(hidden,) if output_hidden_states else None,
            )

    decoder = TinyPolicy()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3)
    model = ConanR1Model.__new__(ConanR1Model)
    model.model = decoder
    model.processor = SimpleNamespace(tokenizer=tokenizer)
    adapted = SimpleNamespace(
        input_ids=input_ids,
        type_decoder_slot=None,
        influence_decoder_slot=None,
    )

    def response_inputs(self, frames, prompt, response, **kwargs):
        return {"input_ids": input_ids, "labels": labels.clone()}, adapted

    model._response_inputs = MethodType(response_inputs, model)

    # Default Stage-I: one sample, forward -> composite loss -> backward -> step.
    lm, adapted = model.response_nll_with_state(
        [], "prompt", RESPONSE, require_diagnostic_slots=True
    )
    assert adapted.type_decoder_slot.shape == (1, width)
    assert adapted.influence_decoder_slot.shape == (1, width)
    deg = adapted.type_decoder_slot.mean().square()
    rel = adapted.influence_decoder_slot.std().square()
    cons = (
        adapted.type_decoder_slot.square().mean()
        + adapted.influence_decoder_slot.square().mean()
    )
    sft_total = stage1_loss(
        lm, deg, rel, cons, AuxiliaryLossWeights(1.0, 1.0, 0.1)
    ).total
    optimizer.zero_grad(set_to_none=True)
    sft_total.backward()
    optimizer.step()

    # Default Stage-II: repeat the response-state path and one clipped-GRPO step.
    adapted.type_decoder_slot = adapted.influence_decoder_slot = None
    current, adapted = model.response_token_log_probs_with_state(
        [],
        "prompt",
        RESPONSE,
        require_grad=True,
        require_diagnostic_slots=True,
    )
    old = current.detach().clone()
    reference = (current.detach() - 0.05).clone()
    policy, _ = clipped_grpo_loss(
        current,
        old,
        reference,
        advantage=torch.tensor(1.0),
        clip_eps=0.2,
        kl_coef=0.02,
    )
    cons = (
        adapted.type_decoder_slot.square().mean()
        + adapted.influence_decoder_slot.square().mean()
    )
    grpo_total = stage2_loss(
        policy,
        adapted.type_decoder_slot.mean().square(),
        adapted.influence_decoder_slot.std().square(),
        cons,
        AuxiliaryLossWeights(1.0, 1.0, 0.1),
    ).total
    optimizer.zero_grad(set_to_none=True)
    grpo_total.backward()
    optimizer.step()
    assert torch.isfinite(sft_total)
    assert torch.isfinite(grpo_total)

    # A malformed rollout does not abort Stage-II; its undefined consistency
    # term is masked and the format reward remains responsible for the error.
    malformed = "<TYPE>motion_blur:0.4<TYPE_END>"
    malformed_encoding = tokenizer(
        malformed + eos, add_special_tokens=False, return_tensors="pt"
    )
    malformed_labels = response_only_labels(
        tokenizer,
        malformed,
        eos,
        malformed_encoding["input_ids"],
        malformed_encoding.get("attention_mask"),
    )
    malformed_state = SimpleNamespace(
        input_ids=malformed_encoding["input_ids"],
        type_decoder_slot=None,
        influence_decoder_slot=None,
    )
    assert model._attach_diagnostic_slots(
        malformed_state,
        decoder.embedding(malformed_encoding["input_ids"]),
        malformed,
        malformed_labels,
        strict=False,
    ) is False
    assert malformed_state.type_decoder_slot is None
    assert malformed_state.influence_decoder_slot is None
