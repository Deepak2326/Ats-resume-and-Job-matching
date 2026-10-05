"""
test_phase3_ui.py
=================
Headless UI regression tests for the Phase 3 "JD Match" tab, driving the
real app through Streamlit's AppTest harness.

Covers the exact failure reported after Phase 3 went live:
  * a selected listing WITHOUT an extractable description dead-ended the tab
    (no JD shown, no matching). The tab must now surface a recovery path —
    a manual JD paste — and match against it;
  * the results view must include explicit score-improvement suggestions
    (missing skills + JD keywords + feedback tips);
  * the happy path (scraped description present) must agree with the
    deterministic engine-level expectations (TF-IDF fast mode).

Run (this machine): see run_tests.ps1 — uv base interpreter + venv PYTHONPATH.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from streamlit.testing.v1 import AppTest          # noqa: E402

from parser import parse_resume                   # noqa: E402
from sample_jobs import MOCK_JOBS                 # noqa: E402
from test_phase1 import SAMPLE_RESUME_LINES, build_sample_pdf  # noqa: E402

APP = os.path.join(ROOT, "app.py")
MOCK_JD = MOCK_JOBS[0]["description"]             # NovaHealth AI JD (deterministic)
JOB_URL = "https://example.com/jobs/novahealth-ml"
MANUAL_KEY = f"manual_jd::{JOB_URL}"

CHECKS: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    suffix = f"  ->  {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")
    CHECKS.append(label)
    if not condition:
        raise SystemExit(f"FAILED: {label} {detail}")


def _resume():
    return parse_resume(build_sample_pdf(SAMPLE_RESUME_LINES),
                        filename="sample_resume.pdf")


def _job(description: str) -> dict:
    return {
        "job_title": "Machine Learning Engineer",
        "company": "NovaHealth AI",
        "location": "Bengaluru",
        "job_type": "fulltime",
        "job_url": JOB_URL,
        "description": description,
        "site": "indeed",
    }


def _boot() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["authenticated"] = True
    at.session_state["username"] = "demo"
    at.session_state["resume"] = _resume()
    return at


def _find(seq, needle: str):
    return [el for el in seq if needle in str(getattr(el, "label", "") or "")]


def _analyze(at: AppTest) -> AppTest:
    """Force fast TF-IDF mode (hidden from the UI) and click Analyze Fit."""
    check("fast-mode checkbox hidden from the UI",
          len(_find(at.checkbox, "Fast mode")) == 0)
    at.session_state["fast_mode"] = True
    buttons = _find(at.button, "Analyze Fit")
    check("Analyze Fit button rendered", len(buttons) == 1)
    buttons[0].click()
    at.run()
    check("no script exception after Analyze", not at.exception)
    return at


def _assert_result(at: AppTest) -> None:
    mr = at.session_state.get("match_result")
    check("match_result stored in session", mr is not None)
    check("backend is tfidf-fallback (fast mode)",
          mr is not None and mr.backend == "tfidf-fallback", str(mr.backend))
    check("deterministic JD-match score 34.7%",
          mr is not None and round(mr.jd_match_score, 1) == 34.7,
          f"{mr.jd_match_score:.1f}")
    check("missing skills exact",
          mr is not None and mr.missing_skills == ["Deep Learning", "HIPAA", "PyTorch"])
    check("missing_skills handoff populated",
          at.session_state.get("missing_skills") == mr.missing_skills)
    score_metric = [m for m in at.metric if m.label == "JD-Match Score"]
    check("JD-Match Score metric rendered", len(score_metric) == 1,
          score_metric[0].value if score_metric else "-")
    check("score-improvement suggestions rendered",
          any("Suggestions to improve your match score" in str(m.value)
              for m in at.markdown))
    check("JD keyword suggestions rendered",
          any("JD keywords to work into your bullets" in str(m.value)
              for m in at.markdown))


# --- Scenario 1: listing WITHOUT a description (the reported bug) ----------
at = _boot()
at.session_state["selected_job"] = _job("")
at.run()
check("no script exception on empty-description listing", not at.exception)
check("clear 'no JD extracted' error shown",
      any("No job description was extracted" in str(getattr(e, "value", ""))
          for e in at.error))
check("manual JD paste box offered", len(at.text_area) >= 1)
check("no sample-data shortcuts in the UI",
      len(_find(at.button, "sample JD")) == 0)
check("paste-to-enable hint shown",
      any("Paste the job description above" in str(getattr(i, "value", ""))
          for i in at.info))
check("Analyze Fit gated until a JD exists",
      len(_find(at.button, "Analyze Fit")) == 0)

# --- Scenario 2: recovery via manual paste, then match ----------------------
at.session_state[MANUAL_KEY] = MOCK_JD
at.run()
check("no script exception after pasting JD", not at.exception)
check("JD preview expander shows the pasted text",
      any("Job description used for matching" in str(getattr(x, "label", ""))
          for x in at.expander))
_analyze(at)
_assert_result(at)
print("[INFO] empty-description recovery path fully verified")

# --- Scenario 3: happy path — scraped description present -------------------
at = _boot()
at.session_state["selected_job"] = _job(MOCK_JD)
at.run()
check("no script exception on happy path", not at.exception)
check("no spurious 'no JD' error on happy path",
      not any("No job description was extracted" in str(getattr(e, "value", ""))
              for e in at.error))
_analyze(at)
_assert_result(at)
print("[INFO] happy path verified")

print(f"\n{len(CHECKS)}/{len(CHECKS)} checks passed — "
      "PHASE 3 UI REGRESSIONS FIXED (JD recovery + suggestions).")
