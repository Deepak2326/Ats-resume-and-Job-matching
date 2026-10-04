"""
app.py
======
Streamlit entry point — ATS Resume Analyzer & Job Recommendation Platform.

Features: email/password accounts (auth.py, SQLite-backed), resume
upload (PDF/DOCX) with per-account persistence so returning users never
re-upload, an ATS score dashboard (heuristics + ML head), live job scraping
via JobSpy with demo-mode fallback, semantic resume-to-JD matching with
skill-gap analysis (sentence-transformers with TF-IDF fallback), and course
recommendations keyed off the identified skill gaps (recommender.py) with
dual Apply/Upskill actions, score cards and a downloadable analysis summary.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
from datetime import datetime

import streamlit as st

from ats_scorer import ATSReport, ATSScorer, suggest_job_titles
from auth import (
    authenticate,
    create_user,
    delete_resume,
    load_resume,
    save_resume,
)
from matcher import MatchResult, match_resume_to_job
from parser import ParsedResume, ResumeParserError, parse_resume
from recommender import PLATFORM_COLORS, recommend_courses
from scraper import (
    JOB_TYPE_OPTIONS,
    SUPPORTED_SITES,
    JobSearchResult,
    scrape_job_listings,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ATS Resume Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource(show_spinner="Warming up the ATS scoring engine…")
def load_scorer() -> ATSScorer:
    """Train/cache the mock RF head once per server process."""
    return ATSScorer()


def init_session_state() -> None:
    """Guarantee the session-state keys the app relies on."""
    defaults = {
        "authenticated": False,
        "username": None,
        "resume": None,            # ParsedResume
        "ats_report": None,        # ATSReport
        "uploaded_filename": None,
        "jobs_df": None,           # pd.DataFrame
        "jobs_result": None,       # JobSearchResult
        "selected_job": None,      # dict
        "match_result": None,      # MatchResult
        "match_context": None,     # {"filename", "job_url"} staleness check
        "missing_skills": [],      # list[str] — feeds the course engine
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ---------------------------------------------------------------------------
# Auth screens
# ---------------------------------------------------------------------------
def _restore_saved_resume(email: str) -> None:
    """Load the account's saved resume (if any) so returning users can skip
    the upload step entirely — resume + ATS report land straight in session."""
    saved = load_resume(email)
    if not saved:
        return
    try:
        resume = parse_resume(saved["file_bytes"], filename=saved["filename"])
    except ResumeParserError:
        return
    st.session_state.resume = resume
    st.session_state.ats_report = load_scorer().score(resume.text)
    st.session_state.uploaded_filename = saved["filename"]


def login_screen() -> None:
    """Single clean auth view — Sign In by default, Sign Up one click away.
    A successful sign-up drops the user straight back on the Sign In form."""
    st.title("ATS Resume Analyzer & Job Recommendation Platform")
    st.caption(
        "Sign in to analyze, match and improve your resume — it stays saved "
        "to your account, so you only ever upload it once."
    )

    _left, center, _right = st.columns([1, 1.4, 1])  # narrow centered card
    with center:
        if st.session_state.pop("just_signed_up", False):
            # Reset the switcher so the new user lands on the Sign In form.
            st.session_state.pop("auth_mode", None)
            st.success("Account created — sign in with your new credentials.")

        mode = st.radio(
            "Account access",
            ["Sign In", "Sign Up"],
            horizontal=True,
            label_visibility="collapsed",
            key="auth_mode",
        )

        if mode == "Sign In":
            with st.form("signin_form", clear_on_submit=False):
                email = st.text_input("Email", key="signin_email",
                                      placeholder="you@example.com")
                password = st.text_input("Password", type="password",
                                         key="signin_password",
                                         placeholder="Your password")
                submitted = st.form_submit_button(
                    "Sign In", type="primary", use_container_width=True)
            if submitted:
                ok, message = authenticate(email, password)
                if ok:
                    st.session_state.authenticated = True
                    st.session_state.username = email.strip().lower()
                    _restore_saved_resume(st.session_state.username)
                    st.rerun()
                else:
                    st.error(message)
        else:
            with st.form("signup_form", clear_on_submit=False):
                new_email = st.text_input("Email", key="signup_email",
                                          placeholder="you@example.com")
                pw1 = st.text_input("Password", type="password",
                                    key="signup_pw1",
                                    placeholder="Minimum 8 characters",
                                    help="Minimum 8 characters.")
                pw2 = st.text_input("Confirm password", type="password",
                                    key="signup_pw2",
                                    placeholder="Repeat your password")
                submitted = st.form_submit_button(
                    "Create Account", type="primary",
                    use_container_width=True)
            if submitted:
                if pw1 != pw2:
                    st.error("Passwords do not match.")
                else:
                    ok, message = create_user(new_email, pw1)
                    if not ok:
                        st.error(message)
                    else:
                        st.session_state.just_signed_up = True
                        st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Session")
        st.success(f"Logged in as **{st.session_state.username}**")
        if st.button("Log out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.resume = None
            st.session_state.ats_report = None
            st.session_state.uploaded_filename = None
            st.session_state.jobs_df = None
            st.session_state.jobs_result = None
            st.session_state.selected_job = None
            st.session_state.match_result = None
            st.session_state.match_context = None
            st.session_state.missing_skills = []
            st.rerun()
        st.divider()
        if st.session_state.resume is not None:
            if st.button("Remove Saved Resume", use_container_width=True):
                # Detach the resume from the account AND this session.
                delete_resume(st.session_state.username)
                st.session_state.resume = None
                st.session_state.ats_report = None
                st.session_state.uploaded_filename = None
                st.session_state.match_result = None
                st.session_state.match_context = None
                st.session_state.missing_skills = []
                st.rerun()
        if st.button("Reset / Analyze Another Resume", use_container_width=True):
            # Keep the login; clear every analysis artifact so the user can
            # restart the pipeline with a fresh resume.
            st.session_state.resume = None
            st.session_state.ats_report = None
            st.session_state.uploaded_filename = None
            st.session_state.jobs_df = None
            st.session_state.jobs_result = None
            st.session_state.selected_job = None
            st.session_state.match_result = None
            st.session_state.match_context = None
            st.session_state.missing_skills = []
            st.rerun()

# ---------------------------------------------------------------------------
# Upload + ATS score dashboard
# ---------------------------------------------------------------------------
def render_upload_and_score() -> None:
    st.header("Resume upload & ATS score")

    uploaded = st.file_uploader(
        "Upload your resume (PDF or DOCX)",
        type=["pdf", "docx"],
        help="Text-based PDFs parse best. Scanned-image PDFs are flagged — OCR support is on the way.",
    )

    # Re-parse only when a *new* file arrives — reruns stay cheap.
    if uploaded is not None and uploaded.name != st.session_state.uploaded_filename:
        try:
            with st.spinner("Extracting text from resume…"):
                resume = parse_resume(uploaded)
        except ResumeParserError as exc:
            st.error(f"Could not parse the file: {exc}")
            return
        if resume.is_empty:
            st.warning(
                "No machine-readable text found — this looks like a scanned image. "
                "OCR support is planned for a future update."
            )
            return
        with st.spinner("Scoring resume against ATS heuristics + ML head…"):
            report = load_scorer().score(resume.text)
        st.session_state.resume = resume
        st.session_state.ats_report = report
        st.session_state.uploaded_filename = uploaded.name
        # Persist to the account — signed-in users never have to re-upload.
        save_resume(st.session_state.username, uploaded.name,
                    uploaded.getvalue(), resume.text)
        # A new resume invalidates any match analysis computed on the old one.
        st.session_state.match_result = None
        st.session_state.match_context = None
        st.session_state.missing_skills = []

    resume: ParsedResume | None = st.session_state.resume
    report: ATSReport | None = st.session_state.ats_report
    if resume and report:
        st.caption(
            f"**{resume.filename}** is saved to your account — no need to "
            "re-upload. Upload a new file above to replace it."
        )
        for warning in resume.extraction_warnings:
            st.caption(f"{warning}")
        _render_report(resume, report)
    else:
        st.info("Upload a resume to see your ATS score, detected skills and improvement tips.")


def _skill_chips(skills: list[dict]) -> None:
    if not skills:
        st.markdown("_No recognised skills detected._")
        return
    chips = " ".join(
        "<span style='display:inline-block;background:#1f6feb22;border:1px solid #1f6feb55;"
        "color:#58a6ff;padding:4px 12px;margin:3px;border-radius:16px;font-size:0.85rem'>"
        f"{s['skill']} · {s['mentions']}</span>"
        for s in skills
    )
    st.markdown(chips, unsafe_allow_html=True)


def _label_chips(
    labels: list[str],
    *,
    bg: str,
    border: str,
    fg: str,
    empty: str = "_None._",
) -> None:
    """Render plain-string labels as colored pills (gap analysis)."""
    if not labels:
        st.markdown(empty)
        return
    chips = " ".join(
        f"<span style='display:inline-block;background:{bg};border:1px solid {border};"
        f"color:{fg};padding:4px 12px;margin:3px;border-radius:16px;font-size:0.85rem'>"
        f"{label}</span>"
        for label in labels
    )
    st.markdown(chips, unsafe_allow_html=True)


def _inject_css() -> None:
    """App-wide styling for the action buttons + smooth anchor scrolling."""
    st.markdown(
        """
        <style>
        html { scroll-behavior: smooth; }
        a.rml-btn {
            display: block; width: 100%; box-sizing: border-box;
            padding: 0.55rem 1rem; border-radius: 0.5rem;
            font-weight: 600; text-align: center;
            text-decoration: none !important;
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        a.rml-btn:hover { filter: brightness(1.15); text-decoration: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _link_button_html(label: str, url: str, *, bg: str, fg: str = "#ffffff",
                      new_tab: bool = True) -> str:
    """Button-styled anchor. ``new_tab=True`` opens a fresh browser tab
    (``target=_blank``); ``False`` stays in-page so ``#anchor`` links scroll
    (e.g. "Enroll & Upskill" -> the course section)."""
    target = ' target="_blank" rel="noopener noreferrer"' if new_tab else ""
    return (
        f'<a class="rml-btn" href="{html.escape(url, quote=True)}"{target} '
        f'style="background:{bg};color:{fg};">{html.escape(label)}</a>'
    )


def _render_report(resume: ParsedResume, report: ATSReport) -> None:
    st.subheader(f"Results for `{resume.filename}`")

    score_col, grade_col, words_col, pages_col = st.columns(4)
    score_col.metric("ATS Score", f"{report.overall_score}/100")
    grade_col.metric("Grade", report.grade)
    words_col.metric("Words", resume.word_count)
    pages_col.metric("Pages", resume.page_count or "—")
    st.progress(report.overall_score / 100)
    st.caption(
        f"Composite = 60% ML head ({report.ml_score:.1f}, `{report.model_backend}`) "
        f"+ 40% heuristic rubric ({report.heuristic_score:.1f})."
    )

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### Resume health checks")
        for label, ok in report.checks.items():
            dot = "#2ea043" if ok else "#cf222e"
            st.markdown(
                f"<span style='color:{dot};font-weight:700'>&#9679;</span>"
                f"&nbsp;&nbsp;{label}",
                unsafe_allow_html=True,
            )
    with right:
        st.markdown("#### Impact signals")
        f = report.features
        r1c1, r1c2 = st.columns(2)
        r1c1.metric("Action verbs", f["action_verb_count"])
        r1c2.metric("Quantified metrics (%, $, scale)", f["metric_count"])
        r2c1, r2c2 = st.columns(2)
        r2c1.metric("Bullet points", f["bullet_count"])
        r2c2.metric("Sections detected", f["sections_present"])

    st.markdown("#### Standard sections")
    found = ", ".join(sorted(report.sections_found)) or "none"
    missing = ", ".join(sorted(report.sections_missing)) or "none"
    st.markdown(f"**Found:** {found}")
    st.markdown(f"**Missing:** {missing}")

    st.markdown("#### Top extracted skills")
    _skill_chips(report.skills)

    st.markdown("#### Improvement suggestions")
    for tip in report.feedback:
        st.markdown(f"- {tip}")

    with st.expander("View extracted raw text"):
        preview = resume.text[:5000]
        st.text(preview + ("…" if len(resume.text) > 5000 else ""))


# ---------------------------------------------------------------------------
# Live job search
# ---------------------------------------------------------------------------
def render_job_search() -> None:
    """Search form + interactive results + single-job selection for matching."""
    st.header("Live Job Search")

    # Prefill the search term from the resume analysis (manual override OK).
    report: ATSReport | None = st.session_state.ats_report
    suggested = suggest_job_titles(report.skills if report else None)
    if report:
        st.caption(
            "Suggested from your resume skills: "
            + " · ".join(f"**{title}**" for title in suggested)
        )
    else:
        st.caption("Upload a resume in the first tab for skill-based search suggestions.")

    with st.form("job_search_form", clear_on_submit=False):
        col_term, col_loc = st.columns(2)
        search_term = col_term.text_input(
            "Search term",
            value=suggested[0],
            help="Prefilled from your resume analysis — edit freely.",
        )
        location = col_loc.text_input("Location", value="United States")

        col_sites, col_type, col_n = st.columns(3)
        sites = col_sites.multiselect(
            "Job boards", SUPPORTED_SITES, default=["indeed", "linkedin"]
        )
        job_type_label = col_type.selectbox(
            "Experience level / job type", list(JOB_TYPE_OPTIONS.keys())
        )
        results_wanted = col_n.slider("Number of results", 5, 50, 15, step=5)

        col_hours, col_demo = st.columns(2)
        hours_old = col_hours.number_input(
            "Posted within last N hours (0 = any time)", 0, 720, 0, step=24
        )
        force_mock = col_demo.checkbox(
            "Force demo data (offline testing)",
            value=False,
            help="Skip live scraping and use the built-in sample listings.",
        )

        submitted = st.form_submit_button("Search Jobs", use_container_width=True)

    if submitted:
        if not search_term.strip():
            st.error("Please enter a search term.")
            return
        if not sites and not force_mock:
            st.error("Please select at least one job board.")
            return
        with st.spinner("Scraping live job boards… this can take up to a minute."):
            result = scrape_job_listings(
                site_name=sites,
                search_term=search_term,
                location=location,
                results_wanted=results_wanted,
                hours_old=int(hours_old) or None,
                job_type=JOB_TYPE_OPTIONS[job_type_label],
                force_mock=force_mock,
            )
        # Session-state contract: jobs_df + selected_job.
        st.session_state.jobs_result = result
        st.session_state.jobs_df = result.jobs
        st.session_state.selected_job = None  # stale selection from a previous search

    result: JobSearchResult | None = st.session_state.jobs_result
    if result is None:
        st.info(
            "Run a search to aggregate live listings from Indeed, LinkedIn, "
            "Glassdoor and ZipRecruiter."
        )
        return
    _render_job_results(result)


def _render_job_results(result: JobSearchResult) -> None:
    """Interactive results table + selection card feeding the JD matcher."""
    if result.source == "live":
        st.success(result.message)
    else:
        st.warning(result.message)

    df = result.jobs
    st.markdown(f"#### {len(df)} listings for `{result.search_term}` — {result.location}")

    display_cols = [
        c for c in ("job_title", "company", "location", "job_type", "site",
                    "date_posted", "job_url")
        if c in df.columns
    ]
    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "job_title": st.column_config.TextColumn("Job Title", width="large"),
            "company": st.column_config.TextColumn("Company"),
            "location": st.column_config.TextColumn("Location"),
            "job_type": st.column_config.TextColumn("Type"),
            "site": st.column_config.TextColumn("Board"),
            "date_posted": st.column_config.TextColumn("Posted"),
            "job_url": st.column_config.LinkColumn("Apply", display_text="Open"),
        },
    )

    st.markdown("#### Select a job to match against your resume")
    labels = {
        i: f"{row.job_title} — {row.company} ({row.location})"
        for i, row in df.iterrows()
    }
    selected_idx = st.selectbox(
        "Pick one listing to load its full description into the JD-Match engine:",
        options=list(labels.keys()),
        format_func=lambda i: labels[i],
    )
    if selected_idx is None:
        return

    row = df.loc[selected_idx]
    # Boards (esp. LinkedIn) often withhold the description when detail
    # scraping is blocked/rate-limited — normalise None/NaN to "" instead of
    # crashing on len() below, and let the JD Match tab recover gracefully.
    description = row.description if isinstance(row.description, str) else ""
    st.session_state.selected_job = {
        "job_title": row.job_title,
        "company": row.company,
        "location": row.location,
        "job_type": row.job_type,
        "job_url": row.job_url,
        "description": description,
        "site": row.get("site", ""),
    }

    with st.container(border=True):
        st.markdown(f"### {row.job_title}")
        st.markdown(f"**{row.company}** · {row.location} · `{row.job_type}`")
        col_apply, col_meta = st.columns([1, 3])
        col_apply.link_button("Apply to this job", row.job_url,
                              use_container_width=True)
        if description.strip():
            col_meta.caption(
                f"Description: {len(description):,} characters — ready for "
                "semantic matching in the JD Match tab."
            )
        else:
            col_meta.warning(
                "This listing came back without description text — the "
                "JD Match tab will let you paste the JD manually."
            )
        with st.expander("Full job description", expanded=False):
            description = row.description or "_No description returned by this board._"
            st.markdown(description[:6000] + ("…" if len(description) > 6000 else ""))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    init_session_state()
    if not st.session_state.authenticated:
        login_screen()
        return

    render_sidebar()
    _inject_css()
    tab_resume, tab_jobs, tab_match, tab_courses = st.tabs(
        ["Resume & ATS Score", "Job Search", "JD Match", "Courses"]
    )
    with tab_resume:
        render_upload_and_score()
    with tab_jobs:
        render_job_search()
    with tab_match:
        render_jd_match()
    with tab_courses:
        render_courses_tab()


# ---------------------------------------------------------------------------
# Gap analysis & JD-match score
# ---------------------------------------------------------------------------
def _sample_jd() -> str:
    """Built-in demo JD for when the live board withholds description text."""
    try:
        demo = scrape_job_listings(
            "machine learning engineer", results_wanted=3, force_mock=True
        )
        description = demo.jobs.iloc[0]["description"]
        if isinstance(description, str) and description.strip():
            return description.strip()
    except Exception:  # pragma: no cover - ultra-defensive UI fallback
        pass
    return (
        "We are hiring a Machine Learning Engineer with strong Python, "
        "scikit-learn, TensorFlow or PyTorch experience. You will design "
        "training pipelines, deploy services with Docker on AWS, and "
        "collaborate with data science and product teams."
    )


def render_jd_match() -> None:
    """Semantic resume-to-JD matching UI — unlocked by a job selection."""
    st.header("Gap Analysis & Match Score")

    resume: ParsedResume | None = st.session_state.resume
    job: dict | None = st.session_state.selected_job
    if resume is None:
        st.info("Upload a resume in the **Resume & ATS Score** tab first.")
        return
    if job is None:
        st.info(
            "Pick a listing in the **Job Search** tab — its full description "
            "loads into the matching engine."
        )
        return

    st.caption(
        f"Matching **`{resume.filename}`** against "
        f"**{job['job_title']} — {job['company']}** ({job['location']}, `{job['site']}`)."
    )

    # --- Resolve the JD text ---------------------------------------------
    # Boards frequently return listings with no description (detail fetch
    # blocked / rate-limited). Recover with a manual paste or the built-in
    # sample JD instead of dead-ending the tab.
    raw_description = job.get("description")
    scraped_jd = raw_description.strip() if isinstance(raw_description, str) else ""
    jd_text = scraped_jd
    jd_source = f"scraped from {job.get('site') or 'the board'}"

    if not scraped_jd:
        st.error(
            f"**No job description was extracted from "
            f"{job.get('site') or 'the job board'}** — the listing withheld "
            "it (common when detail scraping is rate-limited or blocked)."
        )
        manual_key = f"manual_jd::{job.get('job_url', '')}"
        st.markdown(
            "**Two ways to fix it:** (1) open the listing via **Apply**, "
            "copy the full description and paste it below — or (2) load the "
            "built-in sample JD to demo the engine."
        )
        st.text_area(
            "Job description (paste here)",
            key=manual_key,
            height=220,
            placeholder="Paste the complete job description here…",
            label_visibility="collapsed",
        )
        if st.button("Use the built-in sample JD instead"):
            st.session_state[manual_key] = _sample_jd()
            st.rerun()
        jd_text = (st.session_state.get(manual_key) or "").strip()
        if not jd_text:
            st.info("Paste the JD above (or load the sample) to enable matching.")
            return
        jd_source = "manual paste"

    with st.expander(
        f"Job description used for matching — {jd_source} · "
        f"{len(jd_text):,} characters"
    ):
        preview = (jd_text if len(jd_text) <= 6000
                   else jd_text[:6000] + "\n\n… *(preview truncated)*")
        st.markdown(preview)

    if st.button("Analyze Fit", type="primary", use_container_width=True):
        with st.spinner(
            "Embedding resume & job description… "
            "(first transformer run downloads the ~90 MB model)"
        ):
            result = match_resume_to_job(
                resume.text, jd_text,
                # TF-IDF fast mode stays available for tests/headless runs via
                # session state, but is no longer exposed in the UI.
                use_transformer=not st.session_state.get("fast_mode", False),
            )
        st.session_state.match_result = result
        st.session_state.match_context = {
            "filename": resume.filename,
            "job_url": job["job_url"],
            "jd_hash": hashlib.md5(jd_text.encode()).hexdigest()[:8],
        }
        # Course-engine contract: recommendations key off this list.
        st.session_state.missing_skills = result.missing_skills

    result: MatchResult | None = st.session_state.match_result
    if result is None:
        st.info("Click **Analyze Fit** to compute the JD-Match Score and skill gaps.")
        return

    context = st.session_state.match_context or {}
    current_jd_hash = hashlib.md5(jd_text.encode()).hexdigest()[:8]
    if (context.get("job_url") != job["job_url"]
            or context.get("filename") != resume.filename
            or context.get("jd_hash") != current_jd_hash):
        st.warning(
            "Resume, selected job or JD text changed since the last analysis — "
            "click **Analyze Fit** again for fresh results."
        )
    _render_match_result(result, job)
    _render_action_area(job, result)
    _render_course_section(result.missing_skills, anchor="courses")


def _render_match_result(result: MatchResult, job: dict) -> None:
    """Verdict banner + score metrics + matched/missing skills + feedback."""
    score = result.jd_match_score
    if score >= 75:
        band, color = "Excellent fit", "#2e7d32"
    elif score >= 60:
        band, color = "Strong fit", "#43a047"
    elif score >= 40:
        band, color = "Moderate fit", "#ef6c00"
    else:
        band, color = "Low fit", "#c62828"

    # High-contrast verdict banner — the fit rating is impossible to miss.
    st.markdown(
        f"<div style='text-align:center;margin:4px 0 12px'>"
        f"<span style='display:inline-block;background:{color};color:#ffffff;"
        f"padding:8px 28px;border-radius:999px;font-size:1.1rem;"
        f"font-weight:700;letter-spacing:0.4px'>"
        f"{band} &mdash; {score:.0f}% match</span></div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("JD-Match Score", f"{score:.1f}%")
    c2.metric("Fit rating", band)
    c3.metric("Semantic similarity", f"{result.semantic_similarity:.1f}%")
    c4.metric(
        "Skill coverage",
        f"{result.skill_coverage:.0f}%",
        f"{len(result.matched_skills)}/{len(result.jd_skills)} JD skills",
    )
    st.progress(score / 100)
    st.caption(
        f"Score = 75% semantic cosine (`{result.backend}`, raw cos "
        f"{result.raw_cosine:.3f}) + 25% skill coverage, for "
        f"**{job['job_title']} @ {job['company']}**."
    )

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown(f"#### Matched skills ({len(result.matched_skills)})")
        _label_chips(
            result.matched_skills,
            bg="#23863633", border="#2ea04366", fg="#3fb950",
            empty="_None of the JD's curated skills detected in your resume._",
        )
    with right:
        st.markdown(f"#### Missing skills — add these ({len(result.missing_skills)})")
        _label_chips(
            result.missing_skills,
            bg="#da363322", border="#f8514966", fg="#ff7b72",
            empty="_None — full skill coverage!_",
        )
        if result.missing_keywords:
            st.markdown("**JD keywords to work into your bullets & summary:**")
            _label_chips(
                result.missing_keywords,
                bg="#bb800922", border="#d2992266", fg="#e3b341",
            )
            st.caption(
                "Mirroring these exact JD phrases — truthfully, inside "
                "quantified bullets — raises both the semantic and keyword "
                "component of your score."
            )

    st.markdown("#### Suggestions to improve your match score")
    for tip in result.feedback:
        st.markdown(f"- {tip}")
    st.caption(
        "Skill gaps saved — they drive the personalized course "
        "recommendations in the Courses tab."
    )


# ---------------------------------------------------------------------------
# Course recommendations, action links & final polish
# ---------------------------------------------------------------------------
def _score_glance() -> None:
    """Metric cards for the two headline scores (ATS + JD match)."""
    report: ATSReport | None = st.session_state.ats_report
    result: MatchResult | None = st.session_state.match_result
    col_ats, col_match = st.columns(2)
    with col_ats, st.container(border=True):
        if report is not None:
            st.metric("Baseline ATS Score", f"{report.overall_score}/100",
                      f"Grade {report.grade}")
        else:
            st.metric("Baseline ATS Score", "—",
                      "upload a resume in the tab")
    with col_match, st.container(border=True):
        if result is not None:
            st.metric("JD Match Score", f"{result.jd_match_score:.1f}%",
                      f"{len(result.matched_skills)}/{len(result.jd_skills)} "
                      "JD skills covered")
        else:
            st.metric("JD Match Score", "—",
                      "run Analyze Fit in the tab")


def _render_action_area(job: dict, result: MatchResult) -> None:
    """Dual CTA at the bottom of the job review card: apply or upskill."""
    st.divider()
    st.markdown("#### Ready? Take action")
    col_apply, col_upskill = st.columns(2)
    with col_apply:
        st.markdown(
            _link_button_html("Apply Now — open the job posting",
                              job["job_url"], bg="#238636"),
            unsafe_allow_html=True,
        )
        st.caption("Opens the original listing in a new browser tab.")
    with col_upskill:
        if result.missing_skills:
            st.markdown(
                _link_button_html(
                    f"Enroll & Upskill — bridge "
                    f"{len(result.missing_skills)} skill gap(s)",
                    "#courses", bg="#6f42c1", new_tab=False),
                unsafe_allow_html=True,
            )
            st.caption("Scrolls to your personalized course plan below "
                       "(also in the Courses tab).")
        else:
            st.success("No skill gaps — you're ready to apply!")


def _render_course_section(skills: list[str], *, anchor: str | None) -> None:
    """"Bridge Your Skill Gap" cards — one bordered row per missing skill,
    up to 3 platform recommendations each (curated picks first, then
    platform search-link fallbacks). Rendered in both the JD Match tab
    (``anchor="courses"``, the Enroll & Upskill scroll target) and the
    Courses tab (``anchor=None``)."""
    if not skills:
        return
    header = "Bridge Your Skill Gap: Recommended Courses"
    if anchor:
        st.header(header, anchor=anchor)
    else:
        st.header(header)
    st.caption(
        "Curated picks from Coursera, Udemy and edX for every skill the JD "
        "wants but your resume doesn't show — plus platform search links "
        "for skills without a curated pick."
    )
    grouped: dict[str, list[dict]] = {}
    for rec in recommend_courses(skills, max_per_skill=3):
        grouped.setdefault(rec["skill"], []).append(rec)
    for skill, courses in grouped.items():
        with st.container(border=True):
            st.markdown(f"#### {skill}")
            cols = st.columns(len(courses))
            for col, course in zip(cols, courses):
                with col:
                    color = PLATFORM_COLORS.get(course["platform"], "#6e7681")
                    _label_chips([course["platform"]],
                                 bg=color, border=color, fg="#ffffff")
                    st.markdown(f"**{course['course_title']}**")
                    st.markdown(
                        _link_button_html("Enroll Now", course["url"],
                                          bg=color),
                        unsafe_allow_html=True,
                    )


def _build_analysis_summary() -> str:
    """Lightweight plain-text snapshot for the download button."""
    resume: ParsedResume | None = st.session_state.resume
    report: ATSReport | None = st.session_state.ats_report
    result: MatchResult | None = st.session_state.match_result
    job: dict | None = st.session_state.selected_job

    lines = [
        "ATS RESUME ANALYZER — ANALYSIS SUMMARY",
        "=" * 46,
        f"Generated : {datetime.now():%Y-%m-%d %H:%M}",
        f"User      : {st.session_state.username or '—'}",
        f"Resume    : {resume.filename if resume else '—'}",
        "",
        "1) BASELINE ATS SCORE",
    ]
    if report is not None:
        lines.append(
            f"   Score  : {report.overall_score}/100 (grade {report.grade})")
        lines.append(
            f"   Engine : {report.model_backend} (60% ML head "
            f"{report.ml_score:.1f} + 40% heuristics {report.heuristic_score:.1f})")
        if report.skills:
            lines.append("   Skills : "
                         + ", ".join(s["skill"] for s in report.skills[:12]))
    else:
        lines.append("   Not computed — upload a resume in the Resume tab.")
    lines += ["", "2) JD MATCH"]
    if result is not None and job is not None:
        lines.append(f"   Role       : {job['job_title']} @ {job['company']}")
        lines.append(f"   Apply here : {job['job_url']}")
        lines.append(
            f"   JD Match   : {result.jd_match_score:.1f}% (semantic "
            f"{result.semantic_similarity:.1f}%, backend {result.backend})")
        lines.append(f"   Matched    : {', '.join(result.matched_skills) or '—'}")
        lines.append(f"   Missing    : {', '.join(result.missing_skills) or '—'}")
        lines.append(
            f"   JD keywords: {', '.join(result.missing_keywords) or '—'}")
    else:
        lines.append("   Not computed — run Analyze Fit in the JD Match tab.")
    lines += ["", "3) RECOMMENDED COURSES"]
    recommendations = recommend_courses(st.session_state.missing_skills or [])
    if recommendations:
        current_skill = None
        for rec in recommendations:
            if rec["skill"] != current_skill:
                current_skill = rec["skill"]
                lines.append(f"   • {current_skill}")
            lines.append(f"       - [{rec['platform']}] {rec['course_title']}")
            lines.append(f"         {rec['url']}")
    else:
        lines.append("   No skill gaps identified — nothing to recommend.")
    lines += [
        "", "—",
        "Generated by the ATS Resume Analyzer & Job Recommendation Platform.",
    ]
    return "\n".join(lines)


def render_courses_tab() -> None:
    """Upskilling dashboard: score glance + course plan + summary."""
    st.header("Upskilling Dashboard")
    if st.session_state.resume is None:
        st.info(
            "Upload a resume in the **Resume & ATS Score** tab, then match "
            "against a job in **JD Match** — your personalized course plan "
            "builds from the identified skill gaps."
        )
        return

    st.subheader("Your scores at a glance")
    _score_glance()
    st.divider()

    result: MatchResult | None = st.session_state.match_result
    if result is None:
        st.info(
            "No JD match yet — pick a job in **Job Search** and click "
            "**Analyze Fit** in **JD Match** to reveal your skill gaps."
        )
    elif not result.missing_skills:
        st.success(
            "Full skill coverage — no gaps to bridge for this job! "
            "Match against another listing to keep benchmarking yourself."
        )
    else:
        _render_course_section(result.missing_skills, anchor=None)

    st.divider()
    st.download_button(
        "Download Analysis Summary",
        data=_build_analysis_summary(),
        file_name="ats_analysis_summary.txt",
        mime="text/plain",
        help="Plain-text snapshot of both scores, skill gaps and course links.",
    )


if __name__ == "__main__":
    main()

