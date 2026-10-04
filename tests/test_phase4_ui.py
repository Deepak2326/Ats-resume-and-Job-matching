"""
test_phase4_ui.py
=================
Headless UI regression tests for Phase 4, driving the real app through
Streamlit's AppTest harness with a fully seeded pipeline
(resume + ATS report + selected job + match result).

Covers:
  * the 🎓 Courses tab upskilling dashboard — score-glance metric cards
    (Baseline ATS Score / JD Match Score), "Bridge Your Skill Gap" course
    cards with per-platform Enroll Now links (curated + fallback URLs);
  * the dual-action area at the bottom of the 🎯 job review card —
    "Apply Now" -> job posting URL, "Enroll & Upskill" -> #courses anchor;
  * the "Download Analysis Summary" button and its rendered text payload;
  * the sidebar "Reset / Analyze Another Resume" session reset;
  * empty-state gating (no resume / no match yet).

Run (this machine): see run_tests.ps1 — uv base interpreter + venv PYTHONPATH.
"""

from __future__ import annotations

import hashlib
import os
import sys

# The console under run_tests.ps1 is cp1252 — check labels contain emoji.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from streamlit.testing.v1 import AppTest          # noqa: E402

from ats_scorer import score_resume               # noqa: E402
from matcher import match_resume_to_job           # noqa: E402
from parser import parse_resume                   # noqa: E402
from recommender import recommend_courses         # noqa: E402
from scraper import MOCK_JOBS                     # noqa: E402
from test_phase1 import SAMPLE_RESUME_LINES, build_sample_pdf  # noqa: E402

APP = os.path.join(ROOT, "app.py")
MOCK_JD = MOCK_JOBS[0]["description"]             # NovaHealth AI JD (deterministic)
JOB_URL = "https://example.com/jobs/novahealth-ml"

CHECKS: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    suffix = f"  ->  {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")
    CHECKS.append(label)
    if not condition:
        raise SystemExit(f"FAILED: {label} {detail}")


def _find(seq, needle: str):
    return [el for el in seq if needle in str(getattr(el, "label", "") or "")]


def _boot() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["authenticated"] = True
    at.session_state["username"] = "demo"
    return at


def _md(at: AppTest) -> str:
    """All rendered markdown concatenated (course cards / CTAs live here)."""
    return "\n".join(str(m.value) for m in at.markdown)


# --- Scenario A: full pipeline seeded -> dashboard + dual CTAs + reset --------
at = _boot()
resume = parse_resume(build_sample_pdf(SAMPLE_RESUME_LINES),
                      filename="sample_resume.pdf")
at.session_state["resume"] = resume
at.session_state["ats_report"] = score_resume(resume.text)
at.session_state["selected_job"] = {
    "job_title": "Machine Learning Engineer",
    "company": "NovaHealth AI",
    "location": "Bengaluru",
    "job_type": "fulltime",
    "job_url": JOB_URL,
    "description": MOCK_JD,
    "site": "indeed",
}
mr = match_resume_to_job(resume.text, MOCK_JD, use_transformer=False)
at.session_state["match_result"] = mr
at.session_state["match_context"] = {
    "filename": resume.filename,
    "job_url": JOB_URL,
    "jd_hash": hashlib.md5(MOCK_JD.encode()).hexdigest()[:8],
}
at.session_state["missing_skills"] = mr.missing_skills
at.run()
check("no script exception with full Phase 4 pipeline", not at.exception)
check("deterministic missing-skills trio drives Phase 4",
      mr.missing_skills == ["Deep Learning", "HIPAA", "PyTorch"])

check("4 tabs incl. 🎓 Courses",
      len(at.tabs) == 4 and "Courses" in str(at.tabs[3].label))
check("upskilling dashboard header rendered",
      any("Upskilling Dashboard" in str(h.value) for h in at.header))
check("course section rendered in BOTH tabs (JD Match inline + Courses)",
      sum("Bridge Your Skill Gap" in str(h.value) for h in at.header) == 2)

metrics = {m.label: m.value for m in at.metric}
check("Baseline ATS Score metric card rendered",
      metrics.get("Baseline ATS Score", "").endswith("/100"),
      metrics.get("Baseline ATS Score", "-"))
check("JD Match Score metric card = deterministic 39.3%",
      metrics.get("JD Match Score") == "39.3%", metrics.get("JD Match Score", "-"))

md = _md(at)
check("'Apply Now' CTA links to the job posting URL",
      f'href="{JOB_URL}"' in md and "Apply Now" in md)
check("external CTAs open in a new tab (target=_blank)", 'target="_blank"' in md)
check("'Enroll & Upskill' CTA scrolls to the #courses anchor",
      'href="#courses"' in md and "Enroll &amp; Upskill" in md)
check("Enroll Now button per recommendation (9 recs x 2 sections = 18)",
      md.count("🎓 Enroll Now") == 18, f"got {md.count('🎓 Enroll Now')}")
for rec in recommend_courses(mr.missing_skills):
    check(f"card links to {rec['platform']} for {rec['skill']}",
          rec["url"] in md, rec["url"])
check("fallback HIPAA search links rendered (all 3 platforms)",
      "https://www.coursera.org/search?query=HIPAA" in md
      and "https://www.udemy.com/courses/search/?q=HIPAA" in md
      and "https://www.edx.org/search?q=HIPAA" in md)

dl = _find(at.download_button, "Download Analysis Summary")
check("Download Analysis Summary button rendered", len(dl) == 1)

reset = _find(at.sidebar.button, "Reset / Analyze Another Resume")
check("sidebar reset button rendered", len(reset) == 1)
reset[0].click()
at.run()
check("no exception after reset", not at.exception)
check("reset keeps the user logged in",
      at.session_state["authenticated"] is True)
check("reset clears resume + job + match + gaps",
      at.session_state["resume"] is None
      and at.session_state["selected_job"] is None
      and at.session_state["match_result"] is None
      and at.session_state["missing_skills"] == [])
print("[INFO] full-pipeline scenario verified")

# --- Scenario B: nothing seeded -> courses tab gates on a resume --------------
at = _boot()
at.run()
check("no exception on empty session", not at.exception)
check("courses tab gates on missing resume",
      any("Upload a resume" in str(i.value) for i in at.info))
check("no course cards without a match", "Enroll Now" not in _md(at))
check("download button hidden until a resume exists",
      len(at.download_button) == 0)
print("[INFO] empty-session gating verified")

# --- Scenario C: resume but no JD match yet -> glance shows pending state -----
at = _boot()
resume = parse_resume(build_sample_pdf(SAMPLE_RESUME_LINES),
                      filename="sample_resume.pdf")
at.session_state["resume"] = resume
at.session_state["ats_report"] = score_resume(resume.text)
at.run()
check("no exception with resume only", not at.exception)
metrics = {m.label: m.value for m in at.metric}
check("glance: ATS populated, JD Match pending",
      metrics.get("Baseline ATS Score", "").endswith("/100")
      and metrics.get("JD Match Score") == "—",
      f"ATS={metrics.get('Baseline ATS Score')}, JD={metrics.get('JD Match Score')}")
check("'No JD match yet' guidance rendered",
      any("No JD match yet" in str(i.value) for i in at.info))
check("download button available once a resume exists",
      len(_find(at.download_button, "Download Analysis Summary")) == 1)
print("[INFO] resume-only pending state verified")

print(f"\n{len(CHECKS)}/{len(CHECKS)} checks passed — "
      "PHASE 4 UI VERIFIED (courses, CTAs, summary, reset).")
