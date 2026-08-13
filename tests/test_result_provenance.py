"""Result rows cannot be collected without raw outputs and exact identities."""
from pathlib import Path

import pytest

from scripts.collect_results import validate_result_artifact


def complete_payload():
    return {
        "protocol": {"checkpoint": "checkpoints/model"},
        "per_sample": [{"video_id": "v1", "raw_output": "<ANSWER>x<ANSWER_END>"}],
        "provenance": {
            "code_revision": "a" * 40,
            "annotations_sha256": "b" * 64,
            "splits_sha256": "c" * 64,
            "split_manifest_sha256": "d" * 64,
            "checkpoint_identity_sha256": "e" * 64,
        },
    }


def test_complete_result_artifact_is_accepted():
    validate_result_artifact(complete_payload(), Path("result.json"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("per_sample"),
        lambda payload: payload["provenance"].update(code_revision=None),
        lambda payload: payload["provenance"].pop("annotations_sha256"),
        lambda payload: payload["provenance"].pop("checkpoint_identity_sha256"),
    ],
)
def test_identity_free_or_aggregate_only_artifact_is_rejected(mutation):
    payload = complete_payload()
    mutation(payload)
    with pytest.raises(ValueError):
        validate_result_artifact(payload, Path("result.json"))
