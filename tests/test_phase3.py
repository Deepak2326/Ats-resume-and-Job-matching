"""Phase 3 verification: semantic JD matching & skill-gap analysis (matcher.py).

Deterministic checks run on the TF-IDF backend (``use_transformer=False``);
a guarded smoke test exercises the real sentence-transformers backend when
the model can be loaded (skipped gracefully when offline/blocked).

Run with:  python tests/test_phase3.py
Exit code 0 = all checks passed.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import matcher  # noqa: E402
from matcher import MatchResult, match_resume_to_job, semantic_cosine  # noqa: E402
from sample_jobs import MOCK_JOBS  # noqa: E402
from test_phase1 import SAMPLE_RESUME_LINES, build_sample_pdf  # noqa: E402
from parser import parse_resume  # noqa: E402

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    line = f"[{status}] {name}" + (f"  ->  {detail}" if detail else "")
    print(line)
    if not condition:
        raise SystemExit(f"\nPHASE 3 TESTS FAILED — {name}")
    PASSED.append(name)


SAMPLE_RESUME_TEXT = "\n".join(SAMPLE_RESUME_LINES)
NOVAHEALTH_JD = MOCK_JOBS[0]["description"]  # Senior ML Engineer @ NovaHealth AI
UNRELATED_JD = (
    "Gourmet pastry chef wanted for an artisan bakery. You will knead dough, "
    "laminate croissants, manage oven schedules and design seasonal dessert "
    "menus. Requirements: 5 years of professional baking and food-safety "
    "certification."
)


def test_contract_and_empty_inputs() -> None:
    """Empty inputs degrade gracefully; the result exposes the Phase 4 contract."""
    result = match_resume_to_job("", NOVAHEALTH_JD, use_transformer=False)
    check("empty resume -> zero result", result.jd_match_score == 0.0
          and result.backend == "none" and result.matched_skills == []
          and result.missing_skills == [])
    result = match_resume_to_job(SAMPLE_RESUME_TEXT, "", use_transformer=False)
    check("empty JD -> zero result", result.jd_match_score == 0.0
          and result.backend == "none" and bool(result.feedback))

    result = match_resume_to_job(SAMPLE_RESUME_TEXT, NOVAHEALTH_JD,
                                 use_transformer=False)
    payload = result.to_dict()
    check("contract keys in to_dict()",
          {"jd_match_score", "matched_skills", "missing_skills"} <= set(payload))
    check("contract types",
          isinstance(payload["jd_match_score"], float)
          and all(isinstance(s, str) for s in payload["matched_skills"])
          and all(isinstance(s, str) for s in payload["missing_skills"]))
    check("result is a MatchResult dataclass", isinstance(result, MatchResult))


def test_skill_gap_vs_mock_jd() -> None:
    """Sample resume vs NovaHealth Senior-MLE JD (TF-IDF backend)."""
    result = match_resume_to_job(SAMPLE_RESUME_TEXT, NOVAHEALTH_JD,
                                 use_transformer=False)
    check("backend is tfidf-fallback", result.backend == "tfidf-fallback")
    check("matched skills exact",
          result.matched_skills == ["AWS", "Docker", "Machine Learning", "NLP", "Python"],
          ", ".join(result.matched_skills))
    check("missing skills exact + ranked",
          result.missing_skills == ["Deep Learning", "HIPAA", "PyTorch"],
          ", ".join(result.missing_skills))
    check("skill coverage = 5/8", result.skill_coverage == 62.5,
          f"{result.skill_coverage}%")
    check("JD skill inventory complete", len(result.jd_skills) == 8,
          ", ".join(result.jd_skills))
    check("resume skills inventoried",
          {"SQL", "Pandas", "Tableau", "Git"} <= set(result.resume_skills))
    check("blended score in plausible band", 30.0 <= result.jd_match_score <= 75.0,
          f"{result.jd_match_score}% (sem={result.semantic_similarity}%, "
          f"cos={result.raw_cosine})")


def test_semantic_ordering_and_ceiling() -> None:
    """Related texts outscore unrelated ones; identical texts hit the ceiling."""
    related = match_resume_to_job(SAMPLE_RESUME_TEXT, NOVAHEALTH_JD,
                                  use_transformer=False)
    unrelated = match_resume_to_job(SAMPLE_RESUME_TEXT, UNRELATED_JD,
                                    use_transformer=False)
    check("related JD scores clearly higher",
          related.jd_match_score > unrelated.jd_match_score + 15,
          f"{related.jd_match_score}% vs {unrelated.jd_match_score}%")
    check("unrelated JD has zero skill coverage", unrelated.skill_coverage == 0.0)

    identical = match_resume_to_job(SAMPLE_RESUME_TEXT, SAMPLE_RESUME_TEXT,
                                    use_transformer=False)
    check("identical texts -> cosine ~1.0", identical.raw_cosine >= 0.99,
          f"cos={identical.raw_cosine}")
    check("identical texts -> 100% score", identical.jd_match_score == 100.0
          and identical.semantic_similarity == 100.0
          and identical.skill_coverage == 100.0)
    check("identical texts -> no gaps", identical.missing_skills == [])


def test_missing_keywords_layer() -> None:
    """Free-keyword layer surfaces JD domain terms outside the skill DB."""
    result = match_resume_to_job(SAMPLE_RESUME_TEXT, NOVAHEALTH_JD,
                                 use_transformer=False)
    keywords = result.missing_keywords
    check("keywords capped at 8", 0 < len(keywords) <= 8, ", ".join(keywords))
    check("clinical domain term surfaced",
          any("clinical" in kw for kw in keywords), ", ".join(keywords))
    check("resume-covered skill not flagged", "docker" not in keywords)
    check("curated skill aliases never leak into keyword layer",
          not (set(keywords) & {"pytorch", "deep learning", "hipaa"}))
    check("no n-gram redundancy", not any(
        a != b and (a in b or b in a) for a in keywords for b in keywords))


def test_feedback_tips() -> None:
    """Feedback is actionable and references the top missing skill."""
    result = match_resume_to_job(SAMPLE_RESUME_TEXT, NOVAHEALTH_JD,
                                 use_transformer=False)
    joined = " ".join(result.feedback)
    check("feedback present", len(result.feedback) >= 2)
    check("feedback names top missing skill", "Deep Learning" in joined)
    check("feedback suggests Skills-section mirroring", "Skills section" in joined)

    perfect = match_resume_to_job(SAMPLE_RESUME_TEXT, SAMPLE_RESUME_TEXT,
                                  use_transformer=False)
    check("full-coverage feedback is encouraging",
          any("Excellent" in tip or "Strong" in tip for tip in perfect.feedback))



def test_transformer_backend_smoke() -> None:
    """Real all-MiniLM-L6-v2 path — skipped gracefully when unavailable."""
    try:
        cosine, backend = semantic_cosine(SAMPLE_RESUME_TEXT, NOVAHEALTH_JD,
                                          use_transformer=True)
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"[SKIP] transformer backend unavailable ({type(exc).__name__}: {exc})")
        return
    if backend != matcher.MODEL_NAME:
        print(f"[SKIP] transformer backend unavailable — fell back to {backend}")
        return
    cos_same, _ = semantic_cosine(SAMPLE_RESUME_TEXT, SAMPLE_RESUME_TEXT,
                                  use_transformer=True)
    cos_unrel, _ = semantic_cosine(SAMPLE_RESUME_TEXT, UNRELATED_JD,
                                   use_transformer=True)
    check("transformer: identical texts -> cosine ~1.0", cos_same >= 0.99,
          f"cos={cos_same}")
    check("transformer: related > unrelated ordering", cosine > cos_unrel,
          f"related={cosine:.3f} vs unrelated={cos_unrel:.3f}")


def test_e2e_phase123_chain() -> None:
    """Parse -> score -> scrape -> select -> match (the exact app flow)."""
    resume = parse_resume(build_sample_pdf(SAMPLE_RESUME_LINES),
                          filename="sample_resume.pdf")
    # Deterministic fixture stands in for the live scrape (tests stay
    # offline-safe; the app itself always scrapes live boards).
    row = MOCK_JOBS[0]
    selected_job = {"job_title": row["job_title"], "company": row["company"],
                    "location": row["location"], "job_type": row["job_type"],
                    "job_url": row["job_url"], "description": row["description"],
                    "site": row.get("site", "")}
    match = match_resume_to_job(resume.text, selected_job["description"],
                                use_transformer=False)
    check("e2e: match computed from parsed PDF + scraped JD",
          0.0 < match.jd_match_score <= 100.0, f"{match.jd_match_score}%")
    # The app mirrors missing_skills into session state for Phase 4.
    session_missing_skills = match.missing_skills
    check("e2e: missing_skills handoff is list[str]",
          isinstance(session_missing_skills, list)
          and all(isinstance(s, str) for s in session_missing_skills)
          and session_missing_skills == match.missing_skills,
          ", ".join(session_missing_skills))


def main() -> None:
    test_contract_and_empty_inputs()
    test_skill_gap_vs_mock_jd()
    test_semantic_ordering_and_ceiling()
    test_missing_keywords_layer()
    test_feedback_tips()
    test_transformer_backend_smoke()
    test_e2e_phase123_chain()
    print(f"\n{len(PASSED)}/{len(PASSED)} checks passed — PHASE 3 READY "
          "(semantic matching & gap analysis).")


if __name__ == "__main__":
    main()

