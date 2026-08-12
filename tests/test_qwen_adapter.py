import pytest
import torch

from model.qwen_adapter import (
    image_token_runs,
    insert_sequence_values,
    pool_flow_to_token_grid,
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
