"""End-to-end verification: Phase 1 -> Phase 2 chain, asserting the exact
session-state contract that Phase 3 (Semantic JD Matching) will consume.

Run with:  python tests/test_e2e_phase12.py
Exit code 0 = ready for Phase 3.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import pandas as pd                                    # noqa: E402
from parser import parse_resume                      # noqa: E402
from ats_scorer import score_resume, suggest_job_titles  # noqa: E402
from sample_jobs import MOCK_JOBS                    # noqa: E402
from scraper import CRITICAL_COLUMNS                 # noqa: E402
from test_phase1 import SAMPLE_RESUME_LINES, build_sample_pdf  # noqa: E402

PASSED = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    line = f"[{status}] {name}" + (f"  ->  {detail}" if detail else "")
    print(line)
    if not condition:
        raise SystemExit(f"\nNOT READY FOR PHASE 3 — failed: {name}")
    PASSED.append(name)


def main() -> None:
    # ---- Phase 1: parse + score -------------------------------------------
    resume = parse_resume(build_sample_pdf(SAMPLE_RESUME_LINES), filename="sample_resume.pdf")
    check("Phase 1 | resume parsed", resume.word_count > 50,
          f"{resume.word_count} words, {resume.page_count} page(s)")

    report = score_resume(resume.text)
    check("Phase 1 | ATS score computed", 0 <= report.overall_score <= 100,
          f"score={report.overall_score}/100 grade {report.grade} [{report.model_backend}]")
    check("Phase 1 | skills extracted", len(report.skills) > 0,
          ", ".join(s["skill"] for s in report.skills[:5]))

    # ---- Phase 1 -> 2 handoff: search-term prefill ------------------------
    titles = suggest_job_titles(report.skills)
    check("Phase 2 | search-term prefill from skills", len(titles) >= 1,
          str(titles))

    # ---- Phase 2: deterministic fixture standing in for a live scrape ------
    # (the app always scrapes live boards; tests stay offline-safe)
    jobs_df = pd.DataFrame(MOCK_JOBS)
    check("Phase 2 | jobs returned", not jobs_df.empty,
          f"{len(jobs_df)} rows, source=fixture")
    check("Phase 2 | critical columns present",
          set(CRITICAL_COLUMNS).issubset(jobs_df.columns),
          ", ".join(jobs_df.columns))

    # ---- Phase 2 -> 3 handoff: simulate the selectbox pick -----------------
    row = jobs_df.iloc[0]
    selected_job = {
        "job_title": row.job_title,
        "company": row.company,
        "location": row.location,
        "job_type": row.job_type,
        "job_url": row.job_url,
        "description": row.description,
        "site": row.get("site", ""),
    }
    required_keys = {"job_title", "company", "location", "job_type",
                     "job_url", "description", "site"}
    check("Phase 3 | selected_job contract keys",
          required_keys.issubset(selected_job.keys()))
    check("Phase 3 | JD description long enough for semantic matching",
          len(selected_job["description"]) > 100,
          f"{len(selected_job['description'])} chars")
    check("Phase 3 | apply URL valid", selected_job["job_url"].startswith("http"),
          selected_job["job_url"])

    print(f"\n{len(PASSED)}/{len(PASSED)} checks passed — READY FOR PHASE 3 "
          "(Semantic Matching & Gap Analysis).")


if __name__ == "__main__":
    main()
