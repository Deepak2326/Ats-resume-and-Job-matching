"""
scraper.py
==========
Live job aggregation engine — Phase 2.

Wraps :func:`jobspy.scrape_jobs` to aggregate listings from Indeed, LinkedIn,
Glassdoor and ZipRecruiter into a cleaned Pandas DataFrame. When the live
boards rate-limit, return nothing, or the network is unavailable, the search
reports an honest "unavailable"/"empty" status — the app never fabricates
sample listings.
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

_DISPLAY_COLUMNS = CRITICAL_COLUMNS + ["site", "date_posted"]


@dataclass
class JobSearchResult:
    """Contract handed to the Streamlit UI after every search."""
    jobs: pd.DataFrame
    source: str            # "live" | "empty" | "unavailable"
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
    keep = _DISPLAY_COLUMNS
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=keep)

    df = raw_df.copy()
    if "title" in df.columns and "job_title" not in df.columns:
        df = df.rename(columns={"title": "job_title"})
    for col in keep:
        if col not in df.columns:
            df[col] = None

    df["job_type"] = df["job_type"].map(_normalize_job_type)
    for col in ("job_title", "company", "location", "job_url", "site"):
        df[col] = df[col].astype("string").str.strip()

    df = df.dropna(subset=["job_title", "job_url"])
    df = df[(df["job_title"] != "") & (df["job_url"] != "")]
    df = df.drop_duplicates(subset="job_url", keep="first")

    return df[keep].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Live scraping
# ---------------------------------------------------------------------------
def scrape_job_listings(
    site_name: list[str] | None = None,
    search_term: str = "",
    location: str = "",
    results_wanted: int = 15,
    hours_old: int | None = None,
    job_type: str | None = None,
    country_indeed: str = "USA",
) -> JobSearchResult:
    """Aggregate live listings into a cleaned, UI-ready DataFrame.

    Live-only by design — no fabricated listings:
    * jobspy missing/broken/rate-limited -> ``source="unavailable"``
    * live boards returned nothing usable -> ``source="empty"``
    * success                             -> ``source="live"``
    """
    sites = [s for s in (site_name or ["indeed"]) if s in SUPPORTED_SITES]
    if not sites:
        sites = ["indeed"]
    if not (search_term or "").strip():
        raise ValueError("search_term must be a non-empty string.")
    search_term = search_term.strip()
    location = (location or "").strip()

    def _empty_result(source: str, message: str) -> JobSearchResult:
        return JobSearchResult(
            jobs=pd.DataFrame(columns=_DISPLAY_COLUMNS),
            source=source,
            message=message,
            search_term=search_term,
            location=location or "Any",
        )

    if not _JOBSPY_AVAILABLE:
        return _empty_result(
            "unavailable",
            "The live job-board connector (python-jobspy) is not installed on "
            "this server, so listings cannot be fetched right now.",
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
        return _empty_result(
            "unavailable",
            "Live job boards are temporarily unavailable "
            f"({type(exc).__name__} — usually rate limiting or a network "
            "block). Please try again in a few minutes.",
        )

    cleaned = clean_jobs_dataframe(raw_df)
    if cleaned.empty:
        logger.warning("jobspy returned no usable rows for %r in %r.",
                       search_term, location)
        return _empty_result(
            "empty",
            "No live listings matched this search — try a broader keyword, "
            "more job boards, or a longer posting window.",
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


    return df[keep].reset_index(drop=True)
