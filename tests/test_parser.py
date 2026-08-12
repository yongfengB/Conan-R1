"""Unit tests for the structured output parser."""
import pytest
from model.parser import (
    extract_degradation_profile,
    extract_event_type,
    extract_temporal_interval,
    parse_structured_output,
)


def _make_valid_output(
    type_="motion_blur:0.4",
    influence="Evidence reliability reduced.",
    reasoning="Step 1: vehicle braked. Step 2: collision occurred.",
    conclusion="Rear-end collision caused by sudden braking.",
    answer="event_type: rear-end collision; interval: [2.5, 8.0]",
) -> str:
    return (
        f"<TYPE>{type_}<TYPE_END>"
        f"<INFLUENCE>{influence}<INFLUENCE_END>"
        f"<REASONING>{reasoning}<REASONING_END>"
        f"<CONCLUSION>{conclusion}<CONCLUSION_END>"
        f"<ANSWER>{answer}<ANSWER_END>"
    )


class TestParseStructuredOutput:
    def test_valid_output_parsed(self):
        text = _make_valid_output()
        result = parse_structured_output(text)
        assert result is not None
        assert result.type_block == "motion_blur:0.4"
        assert "collision" in result.conclusion_block

    def test_missing_block_returns_none(self):
        # Missing ANSWER block
        text = (
            "<TYPE>blur<TYPE_END>"
            "<INFLUENCE>reduced<INFLUENCE_END>"
            "<REASONING>step1<REASONING_END>"
            "<CONCLUSION>collision<CONCLUSION_END>"
        )
        assert parse_structured_output(text) is None

    def test_optional_type_influence_for_structural_ablation(self):
        text = (
            "<REASONING>step1<REASONING_END>"
            "<CONCLUSION>collision<CONCLUSION_END>"
            "<ANSWER>event_type: collision; interval: [1, 2]<ANSWER_END>"
        )
        parsed = parse_structured_output(
            text, optional_blocks=("TYPE", "INFLUENCE")
        )
        assert parsed is not None
        assert parsed.type_block == ""

    def test_wrong_order_returns_none(self):
        # ANSWER before TYPE
        text = (
            "<ANSWER>answer<ANSWER_END>"
            "<TYPE>blur<TYPE_END>"
            "<INFLUENCE>reduced<INFLUENCE_END>"
            "<REASONING>step1<REASONING_END>"
            "<CONCLUSION>collision<CONCLUSION_END>"
        )
        assert parse_structured_output(text) is None

    def test_duplicate_block_returns_none(self):
        text = _make_valid_output() + "<ANSWER>duplicate<ANSWER_END>"
        assert parse_structured_output(text) is None

    def test_empty_string_returns_none(self):
        assert parse_structured_output("") is None

    def test_empty_required_block_returns_none(self):
        assert parse_structured_output(_make_valid_output(reasoning="")) is None

    def test_free_text_outside_blocks_returns_none(self):
        assert parse_structured_output("analysis: " + _make_valid_output()) is None
        assert parse_structured_output(_make_valid_output() + " trailing") is None

    def test_raw_text_preserved(self):
        text = _make_valid_output()
        result = parse_structured_output(text)
        assert result is not None
        assert result.raw_text == text

    def test_whitespace_stripped(self):
        text = (
            "<TYPE>  blur  <TYPE_END>"
            "<INFLUENCE>  reduced  <INFLUENCE_END>"
            "<REASONING>  step1  <REASONING_END>"
            "<CONCLUSION>  collision  <CONCLUSION_END>"
            "<ANSWER>  answer  <ANSWER_END>"
        )
        result = parse_structured_output(text)
        assert result is not None
        assert result.type_block == "blur"


class TestExtractTemporalInterval:
    def test_bracket_format(self):
        interval = extract_temporal_interval("The event occurred at [2.5, 8.0] seconds.")
        assert interval == pytest.approx((2.5, 8.0))

    def test_from_to_format(self):
        interval = extract_temporal_interval("from 3.0 to 9.5 sec")
        assert interval == pytest.approx((3.0, 9.5))

    def test_no_interval_returns_none(self):
        assert extract_temporal_interval("No timestamps here.") is None

    def test_invalid_interval_returns_none(self):
        # start >= end
        assert extract_temporal_interval("[8.0, 2.0]") is None

    def test_integer_timestamps(self):
        interval = extract_temporal_interval("[3, 10]")
        assert interval == pytest.approx((3.0, 10.0))

    def test_multiple_intervals_are_rejected(self):
        assert extract_temporal_interval("[1, 2] or [3, 4]") is None


class TestExtractEventType:
    def test_explicit_event_field(self):
        assert extract_event_type(
            "event_type: rear-end collision; interval: [3, 10]"
        ) == "rear-end collision"

    def test_free_form_is_not_silently_judged(self):
        assert extract_event_type("A rear-end collision occurs.") is None

    def test_duplicate_event_fields_are_rejected(self):
        assert extract_event_type(
            "event_type: collision; event_type: collision; interval: [1, 2]"
        ) is None


class TestExtractDegradationProfile:
    def test_clean_literal(self):
        assert extract_degradation_profile("none") == []

    def test_compound_profile(self):
        assert extract_degradation_profile(
            "motion blur:0.4; sensor_noise:0.8"
        ) == [("motion_blur", 0.4), ("sensor_noise", 0.8)]

    @pytest.mark.parametrize(
        "value",
        ["blur", "blur:not-a-number", "blur:1.1", "blur:nan", "none;blur:0.2"],
    )
    def test_malformed_profile(self, value):
        assert extract_degradation_profile(value) is None
