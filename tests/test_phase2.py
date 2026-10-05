"""End-to-end smoke tests for Phase 2 — scraper cleaning, fallback & title suggestions.

Run with:  python tests/test_phase2.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

import scraper  # noqa: E402
from ats_scorer import suggest_job_titles  # noqa: E402
from scraper import (  # noqa: E402
    CRITICAL_COLUMNS,
    clean_jobs_dataframe,
    scrape_job_listings,
)


class _FakeJobType:  # mimics jobspy's JobType enum objects
    def __init__(self, value: str):
        self.value = value


def _raw_jobspy_style_df() -> pd.DataFrame:
    """Synthetic DataFrame using jobspy's real column schema (title, enum job_type…)."""
    return pd.DataFrame([
        {"site": "indeed", "title": "Data Scientist", "company": "Acme",
         "location": "Austin, TX", "job_type": [_FakeJobType("fulltime")],
         "job_url": "https://x.example/1", "description": "Python and SQL models."},
        {"site": "indeed", "title": "Data Scientist", "company": "Acme",
         "location": "Austin, TX", "job_type": None,
         "job_url": "https://x.example/1", "description": None},  # duplicate URL
        {"site": "linkedin", "title": "ML Engineer", "company": "Beta",
         "location": "Remote", "job_type": "fulltime",
         "job_url": "https://x.example/2", "description": "PyTorch, Docker, AWS."},
        {"site": "linkedin", "title": None, "company": "Ghost",
         "location": "Remote", "job_type": None,
         "job_url": "https://x.example/3", "description": "No title -> dropped."},
    ])


def test_clean_jobs_dataframe() -> None:
    cleaned = clean_jobs_dataframe(_raw_jobspy_style_df())
    assert set(CRITICAL_COLUMNS).issubset(cleaned.columns), "critical columns missing"
    assert "job_title" in cleaned.columns, "title was not renamed to job_title"
    assert len(cleaned) == 2, f"expected dedupe+drop -> 2 rows, got {len(cleaned)}"
    assert cleaned.loc[0, "job_type"] == "fulltime", "enum job_type not normalised"
    assert cleaned.loc[1, "job_type"] == "fulltime"
    assert cleaned["description"].isna().sum() == 0
    assert cleaned["job_url"].is_unique, "duplicate apply URLs survived cleaning"
    empty = clean_jobs_dataframe(pd.DataFrame())
    assert empty.empty and set(CRITICAL_COLUMNS).issubset(empty.columns)
    print("PASS  clean_jobs_dataframe (rename, normalise, dedupe, drop-empties)")


def test_live_path_via_monkeypatch() -> None:
    original = scraper.scrape_jobs
    scraper.scrape_jobs = lambda **kwargs: _raw_jobspy_style_df()
    try:
        result = scrape_job_listings(["indeed"], "Data Scientist", "Austin, TX")
    finally:
        scraper.scrape_jobs = original
    assert result.source == "live", result.message
    assert len(result.jobs) == 2
    print("PASS  scrape_job_listings live path (monkeypatched jobspy)")


def test_unavailable_on_exception() -> None:
    def _boom(**kwargs):
        raise RuntimeError("429 rate limited")

    original = scraper.scrape_jobs
    scraper.scrape_jobs = _boom
    try:
        result = scrape_job_listings(["indeed"], "Data Scientist", "Austin, TX",
                                     results_wanted=10)
    finally:
        scraper.scrape_jobs = original
    assert result.source == "unavailable", result.source
    assert result.jobs.empty, "no fabricated listings may be returned"
    assert set(CRITICAL_COLUMNS).issubset(result.jobs.columns)
    assert "temporarily unavailable" in result.message
    print("PASS  jobspy exception -> honest 'unavailable' status, zero rows")


def test_empty_result_and_job_type_filter() -> None:
    original = scraper.scrape_jobs
    scraper.scrape_jobs = lambda **kwargs: pd.DataFrame()  # boards return nothing
    try:
        empty = scrape_job_listings(["indeed"], "Data Scientist", "Austin, TX")
    finally:
        scraper.scrape_jobs = original
    assert empty.source == "empty", empty.source
    assert empty.jobs.empty and "No live listings matched" in empty.message

    scraper.scrape_jobs = lambda **kwargs: _raw_jobspy_style_df()
    try:
        fulltime = scrape_job_listings(["indeed"], "Data Scientist", "Austin, TX",
                                       job_type="fulltime")
        contract = scrape_job_listings(["indeed"], "Data Scientist", "Austin, TX",
                                       job_type="contract")
    finally:
        scraper.scrape_jobs = original
    assert fulltime.source == "live"
    assert (fulltime.jobs["job_type"] == "fulltime").all()
    # Zero matches for the requested type -> fall back to unfiltered results.
    assert contract.source == "live" and len(contract.jobs) == 2
    print("PASS  empty-result status + local job_type filter")


def test_suggest_job_titles() -> None:
    skills = [{"skill": "Python", "mentions": 5},
              {"skill": "Machine Learning", "mentions": 3},
              {"skill": "SQL", "mentions": 2}]
    titles = suggest_job_titles(skills)
    assert "Machine Learning Engineer" in titles and "Data Scientist" in titles
    assert len(titles) <= 3
    assert suggest_job_titles(None) == ["Software Engineer"]
    assert suggest_job_titles([{"skill": "Kafka", "mentions": 1}]) == ["Software Engineer"]
    print("PASS  suggest_job_titles (rules, cap, defaults)")


def main() -> None:
    test_clean_jobs_dataframe()
    test_live_path_via_monkeypatch()
    test_unavailable_on_exception()
    test_empty_result_and_job_type_filter()
    test_suggest_job_titles()
    print("\nAll Phase 2 smoke tests passed.")


if __name__ == "__main__":
    main()
