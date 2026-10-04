"""
scraper.py
==========
Live job aggregation engine — Phase 2.

Wraps :func:`jobspy.scrape_jobs` to aggregate listings from Indeed, LinkedIn,
Glassdoor and ZipRecruiter into a cleaned Pandas DataFrame, and degrades
gracefully to built-in sample listings when the live boards rate-limit,
return nothing, or the network is unavailable (common in CI/demo settings).
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

try:
    from jobspy import scrape_jobs

    _JOBSPY_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-specific
    scrape_jobs = None  # type: ignore[assignment]
    _JOBSPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
SUPPORTED_SITES: list[str] = ["indeed", "linkedin", "glassdoor", "zip_recruiter"]

CRITICAL_COLUMNS: list[str] = [
    "job_title", "company", "location", "job_type", "job_url", "description",
]

# UI label -> jobspy `job_type` filter value (None = no filter).
JOB_TYPE_OPTIONS: dict[str, str | None] = {
    "Any": None,
    "Full-time": "fulltime",
    "Part-time": "parttime",
    "Contract": "contract",
    "Internship": "internship",
}


class JobScrapingError(Exception):
    """Raised when live scraping fails and ``use_mock_fallback`` is disabled."""


@dataclass
class JobSearchResult:
    """Contract handed to the Streamlit UI after every search."""
    jobs: pd.DataFrame
    source: str            # "live" | "mock-fallback" | "mock-demo"
    message: str           # human-readable status for the UI banner
    search_term: str
    location: str


# ---------------------------------------------------------------------------
# DataFrame cleaning / normalisation
# ---------------------------------------------------------------------------
def _normalize_job_type(value: Any) -> str:
    """Normalise jobspy's job_type cell (enum | list[enum] | str | NaN) to a string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unspecified"
    if isinstance(value, (list, tuple)):
        if not value:
            return "unspecified"
        value = value[0]
    enum_value = getattr(value, "value", value)  # JobType enum -> its value
    text = str(enum_value).strip().lower()
    return text or "unspecified"


def clean_jobs_dataframe(raw_df: pd.DataFrame | None) -> pd.DataFrame:
    """Reduce a raw jobspy DataFrame to the critical, UI-ready columns.

    * Renames jobspy's ``title`` column to ``job_title``.
    * Normalises ``job_type`` to plain lowercase strings.
    * Drops rows without a title or apply-URL, de-duplicates by URL.
    * Keeps ``site`` / ``date_posted`` as extra display metadata when present.
    """
    keep = CRITICAL_COLUMNS + ["site", "date_posted"]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=keep)

    df = raw_df.copy()
    if "title" in df.columns and "job_title" not in df.columns:
        df = df.rename(columns={"title": "job_title"})
    for col in keep:
        if col not in df.columns:
            df[col] = None

    df["job_type"] = df["job_type"].map(_normalize_job_type)
    for col in ("job_title", "company", "location"):
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()

    df = df[(df["job_title"] != "") & (df["job_url"].notna())]
    df = df.drop_duplicates(subset=["job_url"], keep="first")
    df["job_url"] = df["job_url"].astype(str)

    ordered = CRITICAL_COLUMNS + [c for c in ("site", "date_posted") if c in df.columns]
    return df[ordered].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Sample listings (offline fallback / demo mode)
# ---------------------------------------------------------------------------
# Realistic, keyword-rich samples so Phase 3 semantic matching stays
# meaningful even when live boards are unreachable.
MOCK_JOBS: list[dict[str, str]] = [
    {
        "job_title": "Senior Machine Learning Engineer",
        "company": "NovaHealth AI",
        "location": "United States",
        "job_type": "fulltime",
        "job_url": "https://jobs.example.com/novahealth-senior-mle",
        "description": (
            "NovaHealth AI is hiring a Senior Machine Learning Engineer to productionize "
            "clinical NLP models. You will design training pipelines in Python with PyTorch, "
            "deploy services with Docker and AWS, and collaborate with data science and "
            "compliance teams. Requirements: 5+ years of machine learning experience, strong "
            "Python, deep learning, NLP, Docker, AWS; HIPAA familiarity is a plus."
        ),
        "site": "sample",
        "date_posted": "2026-09-28",
    },
    {
        "job_title": "Data Scientist",
        "company": "Quantium Labs",
        "location": "United States",
        "job_type": "fulltime",
        "job_url": "https://jobs.example.com/quantium-data-scientist",
        "description": (
            "Quantium Labs seeks a Data Scientist to own experimentation end-to-end. "
            "Responsibilities include A/B testing, building machine learning models with "
            "scikit-learn, and presenting insights with Tableau. Requirements: strong SQL, "
            "Python, statistics, pandas, and excellent communication with stakeholders."
        ),
        "site": "sample",
        "date_posted": "2026-09-30",
    },
    {
        "job_title": "Data Analyst",
        "company": "BrightCart",
        "location": "United States",
        "job_type": "fulltime",
        "job_url": "https://jobs.example.com/brightcart-data-analyst",
        "description": (
            "BrightCart is looking for a Data Analyst to power retail decision-making. You "
            "will write complex SQL, build Power BI dashboards, and automate weekly reporting "
            "in Excel. Requirements: data analysis, SQL, Excel, Power BI or Tableau; Python "
            "and ETL experience preferred."
        ),
        "site": "sample",
        "date_posted": "2026-10-01",
    },
    {
        "job_title": "Regulatory Affairs Specialist",
        "company": "MedDevice Corp",
        "location": "United States",
        "job_type": "fulltime",
        "job_url": "https://jobs.example.com/meddevice-regulatory-affairs",
        "description": (
            "MedDevice Corp seeks a Regulatory Affairs Specialist to own FDA submissions for "
            "Class II devices. You will prepare 510(k) documentation, maintain 21 CFR "
            "compliance records, and coordinate with quality teams. Requirements: FDA "
            "regulations, 21 CFR, risk analysis, technical writing, stakeholder management."
        ),
        "site": "sample",
        "date_posted": "2026-09-25",
    },
    {
        "job_title": "Frontend Developer (React)",
        "company": "PixelWorks",
        "location": "United States",
        "job_type": "contract",
        "job_url": "https://jobs.example.com/pixelworks-frontend-react",
        "description": (
            "PixelWorks needs a Frontend Developer to rebuild our design system. Daily work "
            "includes React with TypeScript, component testing, and shipping through CI/CD. "
            "Requirements: React, TypeScript, JavaScript, REST APIs, Git; agile experience "
            "preferred."
        ),
        "site": "sample",
        "date_posted": "2026-10-02",
    },
    {
        "job_title": "DevOps Engineer",
        "company": "CloudNine Systems",
        "location": "United States",
        "job_type": "fulltime",
        "job_url": "https://jobs.example.com/cloudnine-devops",
        "description": (
            "CloudNine Systems is hiring a DevOps Engineer to scale our Kubernetes platform. "
            "You will manage AWS infrastructure with Terraform, harden CI/CD pipelines, and "
            "containerize services with Docker. Requirements: AWS, Kubernetes, Terraform, "
            "CI/CD, Linux, and scripting in Python."
        ),
        "site": "sample",
        "date_posted": "2026-09-27",
    },
    {
        "job_title": "Backend Engineer (Python)",
        "company": "FinEdge",
        "location": "United States",
        "job_type": "fulltime",
        "job_url": "https://jobs.example.com/finedge-backend-python",
        "description": (
            "FinEdge is looking for a Backend Engineer to build payment APIs. You will design "
            "REST APIs with FastAPI, model data in PostgreSQL, and deploy with Docker on AWS. "
            "Requirements: Python, FastAPI or Django, SQL, REST API design, Git."
        ),
        "site": "sample",
        "date_posted": "2026-09-29",
    },
    {
        "job_title": "Junior Data Scientist (Internship)",
        "company": "Insight Partners",
        "location": "United States",
        "job_type": "internship",
        "job_url": "https://jobs.example.com/insight-junior-ds-intern",
        "description": (
            "Insight Partners offers a Junior Data Scientist internship for new graduates. You "
            "will clean data with pandas, prototype models in scikit-learn, and visualize "
            "results for client decks. Requirements: Python, pandas, NumPy, statistics, and "
            "strong communication skills."
        ),
        "site": "sample",
        "date_posted": "2026-10-03",
    },
]

# ---------------------------------------------------------------------------
# Public search API
# ---------------------------------------------------------------------------
def _mock_dataframe(
    location: str, job_type: str | None, results_wanted: int
) -> pd.DataFrame:
    """Build the sample-listings DataFrame, personalised to the query."""
    df = pd.DataFrame([dict(job) for job in MOCK_JOBS])
    if location:
        df["location"] = location
    if job_type:
        filtered = df[df["job_type"] == job_type]
        if not filtered.empty:  # never wipe out the whole demo set
            df = filtered
    return df.head(max(1, results_wanted)).reset_index(drop=True)


def scrape_job_listings(
    site_name: list[str] | None = None,
    search_term: str = "",
    location: str = "",
    results_wanted: int = 15,
    hours_old: int | None = None,
    job_type: str | None = None,
    country_indeed: str = "usa",
    use_mock_fallback: bool = True,
    force_mock: bool = False,
) -> JobSearchResult:
    """Aggregate live job listings into a cleaned DataFrame.

    Parameters mirror :func:`jobspy.scrape_jobs`. On any failure (rate limit,
    network error, zero results) a personalised sample set is returned with
    ``source="mock-fallback"`` so the UI workflow remains testable; pass
    ``use_mock_fallback=False`` to raise :class:`JobScrapingError` instead.
    ``force_mock=True`` skips the network entirely (offline demos).
    """
    sites = [s for s in (site_name or ["indeed"]) if s in SUPPORTED_SITES]
    if not sites:
        sites = ["indeed"]
    if not (search_term or "").strip():
        raise ValueError("search_term must be a non-empty string.")
    search_term = search_term.strip()
    location = (location or "").strip()

    def _mock_result(source: str, message: str) -> JobSearchResult:
        return JobSearchResult(
            jobs=_mock_dataframe(location, job_type, results_wanted),
            source=source,
            message=message,
            search_term=search_term,
            location=location or "Any",
        )

    if force_mock:
        return _mock_result(
            "mock-demo",
            "Demo mode: built-in sample listings (live scraping skipped).",
        )
    if not _JOBSPY_AVAILABLE:
        return _mock_result(
            "mock-fallback",
            "python-jobspy is not installed in this environment. Showing sample listings.",
        )

    try:
        # LinkedIn withholds the description body unless explicitly requested —
        # jobspy's `linkedin_fetch_description` defaults to False, which left
        # every LinkedIn listing with an empty JD.  Guarded by signature check
        # so older/newer jobspy releases without the parameter still work.
        extra_kwargs: dict = {}
        if "linkedin_fetch_description" in inspect.signature(scrape_jobs).parameters:
            extra_kwargs["linkedin_fetch_description"] = True

        raw_df = scrape_jobs(
            site_name=sites,
            search_term=search_term,
            location=location or None,
            results_wanted=int(results_wanted),
            hours_old=hours_old,
            job_type=job_type,
            country_indeed=country_indeed,
            description_format="markdown",
            verbose=0,
            **extra_kwargs,
        )
    except Exception as exc:  # rate limits, proxy blocks, TLS errors…
        logger.warning("jobspy scrape failed: %s", exc)
        if not use_mock_fallback:
            raise JobScrapingError(f"Live scrape failed: {exc}") from exc
        return _mock_result(
            "mock-fallback",
            f"Live scraping unavailable ({type(exc).__name__}: likely rate-limited or "
            "network-blocked). Showing sample listings so you can keep testing.",
        )

    cleaned = clean_jobs_dataframe(raw_df)
    if cleaned.empty:
        logger.warning("jobspy returned no usable rows for %r in %r.", search_term, location)
        if not use_mock_fallback:
            raise JobScrapingError("Live scrape returned zero usable listings.")
        return _mock_result(
            "mock-fallback",
            "Live boards returned no listings (rate limit or no matches). "
            "Showing sample listings so you can keep testing.",
        )

    # Defensive local filter — site support for the job_type filter varies.
    if job_type:
        filtered = cleaned[cleaned["job_type"] == job_type]
        if not filtered.empty:
            cleaned = filtered.reset_index(drop=True)

    return JobSearchResult(
        jobs=cleaned,
        source="live",
        message=f"Live scrape complete: {len(cleaned)} listings from {', '.join(sites)}.",
        search_term=search_term,
        location=location or "Any",
    )