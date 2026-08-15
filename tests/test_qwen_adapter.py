import pytest
import torch

from model.qwen_adapter import (
    DiagnosticSlotError,
    image_token_runs,
    insert_sequence_values,
    pool_flow_to_token_grid,
    response_marker_hidden_positions,
    response_only_labels,
)


def test_pool_flow_matches_qwen_merged_grid_shape():
    dense = torch.randn(1, 5, 32, 32, 2)
    pooled = pool_flow_to_token_grid(dense, 3, 4)
    assert pooled.shape == (1, 5, 12, 2)


def test_pool_flow_rejects_non_xy_input():
    with pytest.raises(ValueError):
        pool_flow_to_token_grid(torch.randn(1, 5, 32, 32, 3), 3, 4)


def test_temporal_summary_is_inserted_after_each_image_block():
    input_ids = torch.tensor([[7, 9, 9, 9, 3, 7, 9, 9, 9, 4]])
    runs = image_token_runs(input_ids, image_token_id=9)
    assert runs == [(1, 4), (6, 9)]
    expanded = insert_sequence_values(
        input_ids,
        runs,
        [torch.tensor([[99]]), torch.tensor([[98]])],
    )
    assert expanded.tolist() == [[7, 9, 9, 9, 99, 3, 7, 9, 9, 9, 98, 4]]


def test_insert_sequence_values_expands_labels_with_ignore_positions():
    labels = torch.tensor([[-100, -100, 10, 11, 12]])
    expanded = insert_sequence_values(
        labels,
        [(1, 3)],
        [torch.tensor([[-100]])],
    )
    assert expanded.tolist() == [[-100, -100, 10, -100, 11, 12]]


class _OffsetTokenizer:
    def __init__(self, response, ids, offsets):
        self.response = response
        self.ids = ids
        self.offsets = offsets
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(text)
        assert text == self.response
        assert kwargs["return_offsets_mapping"] is True
        return {"input_ids": self.ids, "offset_mapping": self.offsets}


def test_diagnostic_slots_use_label_mask_and_full_response_offsets():
    response = "<TYPE>blur<TYPE_END><INFLUENCE>weak<INFLUENCE_END>"
    # Token 4 crosses the block boundary, emulating context-dependent BPE.
    ids = [31, 32, 33, 34, 35, 36, 37, 38]
    offsets = [
        (0, 6),
        (6, 10),
        (10, 20),
        (20, 22),
        (22, 33),
        (33, 37),
        (37, 52),
        (52, len(response)),
    ]
    tokenizer = _OffsetTokenizer(response, ids, offsets)
    # The prompt deliberately contains marker-like token ids. They are masked.
    input_ids = torch.tensor([[31, 99, 35, *ids, 2]])
    labels = input_ids.clone()
    labels[:, :3] = -100
    positions = response_marker_hidden_positions(
        tokenizer, response, input_ids, labels
    )
    assert positions["<TYPE>"] == 4
    assert positions["<INFLUENCE>"] == 7
    assert all(position >= 3 for position in positions.values())
    assert tokenizer.calls == [response]


def test_diagnostic_slots_reject_an_inexact_response_label_boundary():
    response = "<TYPE>x<TYPE_END><INFLUENCE>y<INFLUENCE_END>"
    tokenizer = _OffsetTokenizer(
        response,
        [1, 2, 3, 4],
        [(0, 6), (6, 17), (17, 28), (28, len(response))],
    )
    input_ids = torch.tensor([[9, 8, 1, 7, 3, 4]])
    labels = input_ids.clone()
    labels[:, :2] = -100
    with pytest.raises(DiagnosticSlotError, match="exact response-token boundary"):
        response_marker_hidden_positions(tokenizer, response, input_ids, labels)


def test_response_labels_are_built_from_the_exact_active_suffix():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "response<eos>"
            return {"input_ids": [7, 8, 2]}

    ids = torch.tensor([[90, 91, 7, 8, 2, 0, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
    labels = response_only_labels(
        Tokenizer(), "response", "<eos>", ids, attention
    )
    assert labels.tolist() == [[-100, -100, 7, 8, 2, -100, -100]]
