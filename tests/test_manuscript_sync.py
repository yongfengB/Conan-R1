from pathlib import Path

from scripts.check_manuscript_sync import check_manuscript_sync


def test_paper_result_reference_matches_latex_tables():
    paper = Path(__file__).resolve().parents[3] / "sections" / "06_experiments.tex"
    summary = check_manuscript_sync(paper)
    assert summary == {"tables": 4, "rows": 26, "values": 140}
