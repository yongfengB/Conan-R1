"""Tests for agreement statistics used by the independent human protocol."""
import pytest

from scripts.summarize_human_evaluation import fleiss_kappa


def test_fleiss_kappa_perfect_agreement():
    assert fleiss_kappa(
        [["collision", "collision", "collision"], ["fire", "fire", "fire"]]
    ) == pytest.approx(1.0)


def test_fleiss_kappa_requires_equal_rater_counts():
    with pytest.raises(ValueError):
        fleiss_kappa([["a", "a", "a"], ["b", "b"]])
