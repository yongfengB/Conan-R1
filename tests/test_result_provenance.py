"""Result rows cannot be collected without raw outputs and exact identities."""
import copy
import json
from pathlib import Path

import pytest

from scripts.collect_results import validate_result_artifact
from scripts._common import prediction_rows_sha256
from scripts.verify_paper_results import validate_paper_manifest


def complete_payload():
    payload = {
        "protocol": {
            "checkpoint": "checkpoints/model",
            "artifact_role": "paper_evidence",
        },
        "per_sample": [{"video_id": "v1", "raw_output": "<ANSWER>x<ANSWER_END>"}],
        "provenance": {
            "code_revision": "a" * 40,
            "git_worktree_clean": True,
            "annotations_sha256": "b" * 64,
            "splits_sha256": "c" * 64,
            "split_manifest_sha256": "d" * 64,
            "checkpoint_identity_sha256": "e" * 64,
            "resolved_config_sha256": "f" * 64,
            "checkpoint_files_sha256": {"conan_core.pt": "1" * 64},
        },
    }
    payload["provenance"]["raw_predictions_sha256"] = prediction_rows_sha256(
        payload["per_sample"]
    )
    return payload


def test_complete_result_artifact_is_accepted():
    validate_result_artifact(complete_payload(), Path("result.json"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("per_sample"),
        lambda payload: payload["provenance"].update(code_revision=None),
        lambda payload: payload["provenance"].pop("annotations_sha256"),
        lambda payload: payload["provenance"].pop("checkpoint_identity_sha256"),
        lambda payload: payload["provenance"].pop("resolved_config_sha256"),
        lambda payload: payload["provenance"].update(raw_predictions_sha256="0" * 64),
        lambda payload: payload["provenance"].update(git_worktree_clean=False),
    ],
)
def test_identity_free_or_aggregate_only_artifact_is_rejected(mutation):
    payload = complete_payload()
    mutation(payload)
    with pytest.raises(ValueError):
        validate_result_artifact(payload, Path("result.json"))


def test_all_tables_must_bind_the_same_release_commit(tmp_path):
    tables = {}
    for index in range(1, 8):
        table = f"Table {index}"
        payload = copy.deepcopy(complete_payload())
        payload["protocol"]["table_id"] = table
        path = tmp_path / f"table_{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        tables[table] = [path.name]
    manifest = {
        "schema_version": 1,
        "release": {"tag": "v0.3.0", "commit": "a" * 40},
        "tables": tables,
    }
    manifest_path = tmp_path / "paper_results_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validate_paper_manifest(manifest, manifest_path, check_local_tag=False)
