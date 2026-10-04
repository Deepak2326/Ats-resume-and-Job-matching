"""
matcher.py
==========
Semantic resume <-> job-description matching and skill-gap analysis — Phase 3.

Consumes the Phase 1 parsed resume text and the Phase 2 selected job
description (``st.session_state["selected_job"]["description"]``) and returns
a :class:`MatchResult` with:

* **JD-Match Score (0-100)** — cosine similarity between the two documents'
  embeddings. Primary backend: ``sentence-transformers/all-MiniLM-L6-v2``
  (mean-pooled embeddings of 180-word chunks, so long resumes/JDs never hit
  the model's 256-token window). Fallback backend: scikit-learn TF-IDF
  cosine — engaged automatically when sentence-transformers/torch is not
  installed, the first-run model download fails, DLLs are blocked by policy,
  encoding errors out, or the caller passes ``use_transformer=False``.
  The raw cosine is square-root stretched (realistic resume<->JD pairs land
  at 0.2-0.6 cosine; sqrt spreads them legibly over 0-100) and blended
  75% semantic / 25% skill coverage (purely semantic when the JD names no
  curated skills) so closing the skill gap visibly moves the number.
* **Matched / missing skills** — canonical skills from
  ``ats_scorer.SKILLS_DB`` detected in the JD, split by whether the resume
  mentions them (alias-boundary regex, same rules as Phase 1). Missing
  skills are ranked by JD emphasis (mention count, then alphabetical).
* **Missing keywords** — high-signal JD terms *outside* the curated skill
  database (JD TF-IDF top terms, bigrams preferred) the resume never uses —
  the "domain vocabulary" layer of the gap.
* **Feedback** — rule-based, actionable incorporation tips.

The module is deliberately Streamlit-agnostic: ``app.py`` wraps it with
spinners and session state, and the test-suite drives it directly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ats_scorer import SKILLS_DB, extract_skills

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# MiniLM-L6-v2 truncates at 256 word-pieces (~190 English words); chunking at
# 180 words keeps every chunk inside the window, and mean-pooling the chunk
# embeddings gives a stable whole-document vector for any resume/JD length.
_CHUNK_WORDS = 180

# Score blend: semantic similarity dominates; skill coverage keeps the score
# actionable (fixing a missing skill moves the number).
_SEMANTIC_WEIGHT = 0.75
_COVERAGE_WEIGHT = 0.25

# Alias set (lowercase) used to keep curated skills out of the free-keyword
# layer — they are already reported via matched/missing skills.
_SKILL_TERMS: frozenset[str] = frozenset(
    {canonical.lower() for canonical in SKILLS_DB}
    | {alias for aliases in SKILLS_DB.values() for alias in aliases}
)


# ---------------------------------------------------------------------------
# Result contract (Phase 3 -> app.py / Phase 4)
# ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    """Full gap-analysis contract consumed by the Streamlit UI.

    ``missing_skills`` is additionally mirrored into
    ``st.session_state["missing_skills"]`` for the Phase 4 course engine.
    """

    jd_match_score: float          # 0-100 blended headline score
    semantic_similarity: float     # 0-100 sqrt-stretched cosine
    raw_cosine: float              # 0-1 uncalibrated cosine (diagnostics)
    skill_coverage: float          # % of JD skills present in the resume
    matched_skills: list[str]      # canonical skills in both JD and resume
    missing_skills: list[str]      # JD skills absent from the resume (ranked)
    jd_skills: list[str]           # every curated skill detected in the JD
    resume_skills: list[str]       # every curated skill detected in the resume
    missing_keywords: list[str]    # JD domain terms outside the skill DB
    backend: str                   # MODEL_NAME | "tfidf-fallback" | "none"
    feedback: list[str]            # actionable incorporation tips

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view (handy for Phase 4 / persistence)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Embedding backend (lazy, cached, fully guarded)
# ---------------------------------------------------------------------------
_MODEL: Any = None
_MODEL_FAILED = False


def _load_model() -> Any:
    """Load the sentence-transformer once per process; None on any failure.

    Failures are expected and non-fatal: package not installed, first-run
    ~90 MB download blocked/offline, Application Control blocking torch DLLs,
    or OOM on small machines — all degrade to the TF-IDF fallback.
    """
    global _MODEL, _MODEL_FAILED
    if _MODEL is not None or _MODEL_FAILED:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(MODEL_NAME)
        logger.info("Loaded embedding model %s", MODEL_NAME)
    except Exception as exc:  # noqa: BLE001 — deliberate broad safety net
        logger.warning(
            "sentence-transformers unavailable (%s: %s) — using TF-IDF fallback.",
            type(exc).__name__, exc,
        )
        _MODEL_FAILED = True
        _MODEL = None
    return _MODEL


def _chunks(text: str, max_words: int = _CHUNK_WORDS) -> list[str]:
    """Split text into <=``max_words`` whitespace-delimited chunks."""
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _transformer_cosine(text_a: str, text_b: str, model: Any) -> float:
    """Cosine between mean-pooled, L2-normalised chunk embeddings."""
    emb_a = model.encode(_chunks(text_a), normalize_embeddings=True,
                         show_progress_bar=False)
    emb_b = model.encode(_chunks(text_b), normalize_embeddings=True,
                         show_progress_bar=False)
    vec_a = np.asarray(emb_a, dtype=float).mean(axis=0)
    vec_b = np.asarray(emb_b, dtype=float).mean(axis=0)
    vec_a /= np.linalg.norm(vec_a) + 1e-12
    vec_b /= np.linalg.norm(vec_b) + 1e-12
    return float(np.clip(vec_a @ vec_b, 0.0, 1.0))


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """Cosine between TF-IDF vectors (rows are L2-normalised by default)."""
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_features=4096,
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform([text_a, text_b])
    except ValueError:  # e.g. a document with no usable tokens
        return 0.0
    return float(np.clip((matrix[0] @ matrix[1].T).toarray()[0, 0], 0.0, 1.0))


def semantic_cosine(text_a: str, text_b: str, *, use_transformer: bool = True) -> tuple[float, str]:
    """Return ``(cosine, backend)`` for two documents, with automatic fallback."""
    if use_transformer:
        model = _load_model()
        if model is not None:
            try:
                return _transformer_cosine(text_a, text_b, model), MODEL_NAME
            except Exception as exc:  # noqa: BLE001 — encode-time safety net
                logger.warning(
                    "Transformer encoding failed (%s: %s) — TF-IDF fallback.",
                    type(exc).__name__, exc,
                )
    return _tfidf_cosine(text_a, text_b), "tfidf-fallback"



# ---------------------------------------------------------------------------
# Skill & keyword gap analysis
# ---------------------------------------------------------------------------
def _skill_present(text_lower: str, canonical: str) -> bool:
    """True when any alias of ``canonical`` appears with token boundaries."""
    for alias in SKILLS_DB.get(canonical, (canonical.lower(),)):
        pattern = r"(?<![\w+#])" + re.escape(alias) + r"(?![\w+#])"
        if re.search(pattern, text_lower):
            return True
    return False


def _missing_keywords(
    jd_text: str,
    resume_text: str,
    missing_skills: list[str],
    top_n: int = 8,
) -> list[str]:
    """Top JD terms outside the skill DB that the resume never mentions.

    Ranked by JD TF-IDF weight (ties: bigrams first — they carry more domain
    meaning — then alphabetically), de-duplicated against already-kept terms
    and against the missing-skill list.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_features=512,
        sublinear_tf=True,
    )
    try:
        row = vectorizer.fit_transform([jd_text])
    except ValueError:
        return []
    terms = vectorizer.get_feature_names_out()
    weights = row.toarray()[0]
    order = sorted(
        range(len(terms)),
        key=lambda i: (-weights[i], 0 if " " in terms[i] else 1, terms[i]),
    )

    resume_lower = resume_text.lower()
    # sklearn builds n-grams AFTER stop-word removal, which glues unrelated
    # tokens across removed words ("Docker and AWS" -> "docker aws"). Keep a
    # bigram only when it appears verbatim in the JD's lowercased text —
    # punctuation is PRESERVED here so comma-separated list items ("Docker,
    # AWS") don't create false adjacency.
    jd_verbatim = " " + re.sub(r"\s+", " ", jd_text.lower()) + " "
    blocked = {skill.lower() for skill in missing_skills}
    kept: list[str] = []
    for idx in order:
        term = terms[idx]
        if len(term) < 3 or term in _SKILL_TERMS:
            continue  # noise, or already covered by the skill-gap layer
        if " " in term and f" {term} " not in jd_verbatim:
            continue  # stop-word-removal glue artifact, not a real JD phrase
        if re.search(r"(?<![\w+#])" + re.escape(term) + r"(?![\w+#])", resume_lower):
            continue  # resume already uses the term
        if any(term in b or b in term for b in blocked):
            continue  # duplicates a missing skill
        if any(term in k or k in term for k in kept):
            continue  # overlapping n-gram already kept
        kept.append(term)
        if len(kept) >= top_n:
            break
    return kept



# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------
def _build_feedback(
    matched: list[str],
    missing: list[str],
    keywords: list[str],
    coverage: float,
    semantic_pct: float,
) -> list[str]:
    """Rule-based, actionable tips for raising the JD-Match Score."""
    tips: list[str] = []
    if not matched and not missing:
        tips.append(
            "No curated skill keywords detected in this JD — the score is purely "
            "semantic; mirror the posting's exact phrasing in your Summary."
        )
    elif coverage >= 90 and not missing:
        tips.append(
            "Excellent skill alignment — your resume already names every "
            "curated skill in this JD."
        )
    elif missing:
        top3 = ", ".join(f"'{skill}'" for skill in missing[:3])
        tips.append(
            f"Mirror the JD's exact wording: add {top3} to your Skills section — "
            "ATS keyword filters match terms literally."
        )
        tips.append(
            f"Weave '{missing[0]}' into a recent experience bullet with a quantified "
            f"outcome (e.g. 'Applied {missing[0]} to …, improving … by X%')."
        )
    if keywords:
        tips.append(
            "Sprinkle these JD domain phrases where truthful: "
            + ", ".join(f"'{kw}'" for kw in keywords[:4])
            + "."
        )
    if semantic_pct < 50 and coverage >= 60:
        tips.append(
            "Skills align but the phrasing diverges — rewrite your Summary/headline "
            "to echo the posting's own language."
        )
    elif semantic_pct < 50:
        tips.append(
            "Overall language is far from this JD — mirror its terminology in your "
            "Summary and most recent bullets."
        )
    if not tips:
        tips.append(
            "Strong match. Fine-tune by echoing the JD's exact phrasing in your "
            "Summary section and re-running the analysis."
        )
    return tips



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def match_resume_to_job(
    resume_text: str,
    job_description: str,
    *,
    use_transformer: bool = True,
    top_missing_keywords: int = 8,
) -> MatchResult:
    """Match a resume against a job description.

    Parameters
    ----------
    resume_text, job_description:
        Raw texts (Phase 1 parse output and Phase 2 selected JD).
    use_transformer:
        When False, skip sentence-transformers entirely (fast TF-IDF mode).
        When True (default), any load/encode failure still falls back to TF-IDF.
    top_missing_keywords:
        Cap on the free-keyword layer of the gap report.
    """
    resume_text = (resume_text or "").strip()
    job_description = (job_description or "").strip()
    if not resume_text or not job_description:
        return MatchResult(
            jd_match_score=0.0, semantic_similarity=0.0, raw_cosine=0.0,
            skill_coverage=0.0, matched_skills=[], missing_skills=[], jd_skills=[],
            resume_skills=[], missing_keywords=[], backend="none",
            feedback=[
                "Upload a resume and select a job with a non-empty description to "
                "compute a match."
            ],
        )

    # ---- Skill gap (curated DB, alias-boundary matching) -------------------
    jd_entries = extract_skills(job_description, top_n=len(SKILLS_DB))
    resume_lower = resume_text.lower()
    matched = [e["skill"] for e in jd_entries if _skill_present(resume_lower, e["skill"])]
    missing = [e["skill"] for e in jd_entries if not _skill_present(resume_lower, e["skill"])]
    jd_skills = [e["skill"] for e in jd_entries]
    resume_skills = [e["skill"] for e in extract_skills(resume_text, top_n=len(SKILLS_DB))]
    coverage = (100.0 * len(matched) / len(jd_skills)) if jd_skills else 0.0

    # ---- Semantic similarity (transformer w/ TF-IDF fallback) --------------
    cosine, backend = semantic_cosine(resume_text, job_description,
                                      use_transformer=use_transformer)
    semantic_pct = round(100.0 * float(np.sqrt(np.clip(cosine, 0.0, 1.0))), 1)
    if jd_skills:
        score = round(
            float(np.clip(_SEMANTIC_WEIGHT * semantic_pct + _COVERAGE_WEIGHT * coverage,
                          0.0, 100.0)),
            1,
        )
    else:
        # No curated skills named in the JD — nothing to "cover", so the
        # coverage weight folds back into the semantic signal.
        score = semantic_pct

    keywords = _missing_keywords(job_description, resume_text, missing,
                                 top_missing_keywords)
    feedback = _build_feedback(matched, missing, keywords, coverage, semantic_pct)

    return MatchResult(
        jd_match_score=score,
        semantic_similarity=semantic_pct,
        raw_cosine=round(cosine, 4),
        skill_coverage=round(coverage, 1),
        matched_skills=matched,
        missing_skills=missing,
        jd_skills=jd_skills,
        resume_skills=resume_skills,
        missing_keywords=keywords,
        backend=backend,
        feedback=feedback,
    )

