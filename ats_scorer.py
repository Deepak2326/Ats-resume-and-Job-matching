"""
ats_scorer.py
=============
ATS (Applicant Tracking System) scoring engine — Phase 1 baseline.

Two cooperating components:

1. **Heuristic feature extractor** — regex/rule-based signals that real ATS
   platforms are known to reward: contact information, standard section
   headers, action verbs, quantified achievements (%, $, scale numbers),
   bullet usage and document length.
2. **Predictive scoring head** — a ``RandomForestRegressor`` trained on a
   synthetic, fully deterministic dataset that maps
   ``[TF-IDF summary stats + heuristic counts] -> ATS score (0-100)``.

The ML head is intentionally *dynamically mocked*: it demonstrates the exact
feature contract a real, offline-trained model will consume in later phases
(swap ``ATSScorer._train_mock_model`` for a ``joblib.load`` of a persisted
pipeline once labelled recruiter data exists).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sklearn.ensemble import RandomForestRegressor

    _SKLEARN_ENSEMBLE_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-specific (e.g. blocked DLL policy)
    RandomForestRegressor = None  # type: ignore[assignment]
    _SKLEARN_ENSEMBLE_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex signal library
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Phone *candidate* pattern — deliberately permissive; find_phones() validates
# the digit count (10-15, per E.164) so date ranges ("2018-2020"), percentages
# and currency amounts never qualify. Handles US formats, Indian formats
# (+91 98765 43210, +91-9876543210, 098765 43210, bare 10-digit mobiles) and
# other international formats with space/dot/dash/paren grouping.
PHONE_RE = re.compile(
    r"(?<![\w@$%/:+.])"        # not inside an email, URL, currency or version string
    r"\+?\d[\d \t.\-()]{6,16}\d"
    r"(?!\d)"
)
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
# Profile links — protocol and "www." optional (bare domains are common in
# resumes, and DOCX/PDF extraction can lose the protocol prefix); locale
# subdomains like "in.linkedin.com" are covered.
LINKEDIN_RE = re.compile(
    r"(?<![\w.+-])(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/(?:in|pub|company)/[\w%\-]+",
    re.IGNORECASE,
)
GITHUB_RE = re.compile(
    r"(?<![\w.+-])(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9][\w\-]*",
    re.IGNORECASE,
)
PORTFOLIO_RE = re.compile(
    r"(?<![\w.+-])(?:https?://)?(?:www\.)?"
    r"(?:gitlab\.com|bitbucket\.org|kaggle\.com|medium\.com|behance\.net|"
    r"dribbble\.com|stackoverflow\.com|about\.me|linktr\.ee|[\w\-]+\.github\.io)"
    r"(?:/[\w\-/]*)?",
    re.IGNORECASE,
)
_URL_TRAILING_PUNCT = ".,;:!?)]}>'\""
PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%")
CURRENCY_RE = re.compile(r"[$€£]\s?\d[\d,]*(?:\.\d+)?\s?[kKmMbB]?")
MAGNITUDE_RE = re.compile(r"\b\d+(?:\.\d+)?[kKmMbB]\+\b")
# Bullet glyphs seen in real resumes: typographic bullets, en/em dashes, stars,
# numbered items, and the Wingdings/Symbol private-use range (\uf0b7 …) that
# Word-authored PDFs and DOCX conversions emit.
BULLET_RE = re.compile(
    r"^\s*(?:[•◦▪●○‣➢➤✓✔»·∙■□⚬*\-\u2013\u2014]|[\uf000-\uf0ff]|\d+[.)])\s+",
    re.MULTILINE,
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'+\-]*")


def find_phones(text: str) -> list[str]:
    """Return validated phone numbers (10-15 digits, E.164) from raw text."""
    phones: list[str] = []
    for match in PHONE_RE.finditer(text or ""):
        digits = re.sub(r"\D", "", match.group())
        if 10 <= len(digits) <= 15:
            phones.append(match.group().strip(" .-()"))
    return phones


def find_urls(text: str) -> list[str]:
    """Return URLs with trailing sentence punctuation stripped."""
    return [m.group().rstrip(_URL_TRAILING_PUNCT) for m in URL_RE.finditer(text or "")]

# ---------------------------------------------------------------------------
# Rule libraries
# ---------------------------------------------------------------------------
ACTION_VERBS: frozenset[str] = frozenset({
    "accelerated", "achieved", "advised", "analyzed", "analysed", "architected",
    "authored", "automated", "boosted", "built", "championed", "coached",
    "collaborated", "converted", "created", "cut", "decreased", "delivered",
    "deployed", "designed", "developed", "doubled", "drove", "eliminated",
    "engineered", "enhanced", "established", "exceeded", "executed", "expanded",
    "facilitated", "forecasted", "founded", "generated", "grew", "guided",
    "headed", "implemented", "improved", "increased", "initiated", "integrated",
    "introduced", "invented", "launched", "led", "managed", "maximized",
    "mentored", "migrated", "minimized", "modeled", "modernized", "negotiated",
    "optimized", "orchestrated", "outperformed", "overhauled", "owned",
    "pioneered", "planned", "presented", "produced", "programmed", "published",
    "rebuilt", "redesigned", "reduced", "refactored", "researched", "resolved",
    "revitalized", "saved", "scaled", "secured", "shipped", "simplified",
    "spearheaded", "standardized", "streamlined", "strengthened", "supervised",
    "tested", "trained", "transformed", "tripled", "validated", "won",
})

STANDARD_SECTIONS: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "objective", "profile", "about me"),
    "experience": (
        "experience", "employment", "work history",
        "professional experience", "work experience",
    ),
    "education": ("education", "academic background", "academics"),
    "skills": (
        "skills", "technical skills", "core competencies",
        "technologies", "competencies",
    ),
    "projects": ("projects", "personal projects", "key projects", "selected projects"),
    "certifications": ("certifications", "certificates", "licenses", "courses"),
}

# Canonical skill -> literal aliases (matched case-insensitively with
# alphanumeric boundaries, so "Java" never matches "JavaScript").
SKILLS_DB: dict[str, tuple[str, ...]] = {
    # Languages & core data stack
    "Python": ("python",),
    "SQL": ("sql", "mysql", "postgresql", "postgres", "sqlite"),
    "R": ("r",),
    "Java": ("java",),
    "JavaScript": ("javascript",),
    "TypeScript": ("typescript",),
    "C++": ("c++",),
    "C#": ("c#",),
    "Go": ("golang",),
    # Data science / ML
    "Machine Learning": ("machine learning",),
    "Deep Learning": ("deep learning",),
    "NLP": ("nlp", "natural language processing"),
    "Data Analysis": ("data analysis", "data analytics"),
    "Data Visualization": ("data visualization", "data visualisation"),
    "Statistics": ("statistics", "statistical analysis"),
    "A/B Testing": ("a/b testing", "ab testing"),
    "Pandas": ("pandas",),
    "NumPy": ("numpy",),
    "Scikit-learn": ("scikit-learn", "scikit learn", "sklearn"),
    "TensorFlow": ("tensorflow",),
    "PyTorch": ("pytorch",),
    "Keras": ("keras",),
    "XGBoost": ("xgboost",),
    "Spark": ("apache spark", "pyspark", "spark"),
    "Hadoop": ("hadoop",),
    "ETL": ("etl", "data pipeline", "data pipelines"),
    # BI & cloud / devops
    "Excel": ("excel",),
    "Tableau": ("tableau",),
    "Power BI": ("power bi", "powerbi"),
    "AWS": ("aws", "amazon web services"),
    "Azure": ("azure",),
    "Google Cloud": ("google cloud", "gcp"),
    "Docker": ("docker",),
    "Kubernetes": ("kubernetes", "k8s"),
    "Git": ("git", "github", "gitlab"),
    "Linux": ("linux",),
    "CI/CD": ("ci/cd", "continuous integration", "continuous delivery"),
    "Terraform": ("terraform",),
    # Web & product
    "React": ("react", "react.js", "reactjs"),
    "Node.js": ("node.js", "nodejs", "node"),
    "Django": ("django",),
    "Flask": ("flask",),
    "FastAPI": ("fastapi",),
    "REST APIs": ("rest api", "rest apis", "restful"),
    "Agile": ("agile", "scrum", "kanban"),
    "Jira": ("jira",),
    # Business, compliance & soft skills
    "Project Management": ("project management",),
    "Leadership": ("leadership",),
    "Communication": ("communication",),
    "Stakeholder Management": ("stakeholder management", "stakeholder engagement"),
    "FDA Regulations": ("fda regulations", "fda compliance", "21 cfr", "fda"),
    "HIPAA": ("hipaa",),
    "Financial Modeling": ("financial modeling", "financial modelling"),
    "Risk Analysis": ("risk analysis", "risk assessment"),
    "Supply Chain": ("supply chain",),
    "Salesforce": ("salesforce",),
    "SEO": ("seo",),
}


# ---------------------------------------------------------------------------
# Heuristic feature extractor
# ---------------------------------------------------------------------------
def _detect_sections(text: str) -> list[str]:
    """Find standard resume sections via short header-like lines."""
    found: list[str] = []
    lines = [ln.strip().lower().strip(":") for ln in text.splitlines()]
    for canonical, aliases in STANDARD_SECTIONS.items():
        for line in lines:
            # Long lines are prose, not headers — skip them.
            if not line or len(line) > 45:
                continue
            if any(alias in line for alias in aliases):
                found.append(canonical)
                break
    return found


def extract_heuristic_features(text: str) -> dict[str, Any]:
    """Compute the full heuristic feature dictionary for a resume text."""
    text = text or ""
    words = WORD_RE.findall(text)
    word_count = len(words)
    action_verb_count = sum(1 for w in words if w.lower() in ACTION_VERBS)
    metric_count = (
        len(PERCENT_RE.findall(text))
        + len(CURRENCY_RE.findall(text))
        + len(MAGNITUDE_RE.findall(text))
    )
    sections = _detect_sections(text)
    urls = find_urls(text)
    # A portfolio = any explicit URL that isn't a LinkedIn/GitHub profile
    # (e.g. a personal site), plus well-known portfolio platforms.
    has_portfolio = bool(PORTFOLIO_RE.search(text)) or any(
        not LINKEDIN_RE.search(u) and not GITHUB_RE.search(u) for u in urls
    )
    return {
        "word_count": word_count,
        "bullet_count": len(BULLET_RE.findall(text)),
        "action_verb_count": action_verb_count,
        "action_verb_density": round(action_verb_count / max(word_count, 1) * 100, 2),
        "metric_count": metric_count,
        "metric_density": round(metric_count / max(word_count, 1) * 100, 2),
        "has_email": bool(EMAIL_RE.search(text)),
        "has_phone": bool(find_phones(text)),
        "has_url": bool(urls),
        "has_linkedin": bool(LINKEDIN_RE.search(text)),
        "has_github": bool(GITHUB_RE.search(text)),
        "has_portfolio": has_portfolio,
        "sections": sections,
        "sections_present": len(sections),
    }


def extract_skills(text: str, top_n: int = 12) -> list[dict[str, Any]]:
    """Return the top-N detected skills as ``{'skill', 'mentions'}`` dicts,
    sorted by mention frequency (descending), then alphabetically."""
    text_lower = (text or "").lower()
    results: list[dict[str, Any]] = []
    for canonical, aliases in SKILLS_DB.items():
        mentions = 0
        for alias in aliases:
            pattern = r"(?<![\w+#])" + re.escape(alias) + r"(?![\w+#])"
            mentions += len(re.findall(pattern, text_lower))
        if mentions:
            results.append({"skill": canonical, "mentions": mentions})
    results.sort(key=lambda d: (-d["mentions"], d["skill"]))
    return results[:top_n]

# ---------------------------------------------------------------------------
# Predictive scoring head (mock RandomForest over TF-IDF + heuristics)
# ---------------------------------------------------------------------------
MODEL_FEATURES: list[str] = [
    "word_count", "bullet_count", "action_verb_density", "metric_density",
    "has_email", "has_phone", "has_linkedin", "sections_present",
    "tfidf_mean", "tfidf_max",
]

# Small fixed background corpus so TF-IDF statistics for a single resume are
# meaningful (IDF needs multiple documents). Replaced by a persisted, fitted
# vectorizer once a real training corpus exists.
_REFERENCE_CORPUS: list[str] = [
    "Experienced software engineer skilled in python java and cloud infrastructure.",
    "Data analyst with strong sql excel tableau and statistics background.",
    "Project manager leading agile teams, stakeholder communication and delivery.",
    "Marketing specialist focused on seo content strategy and brand growth.",
    "Machine learning engineer deploying models with tensorflow pytorch and docker.",
    "Financial analyst with expertise in risk analysis modeling and forecasting.",
    "Registered nurse with clinical experience patient care and hipaa compliance.",
    "Sales executive exceeding quota through negotiation and crm pipeline management.",
]


def _synthetic_training_set(n_samples: int = 800, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic dataset encoding a plausible scoring rubric.

    The ground-truth formula rewards contact info, section coverage, ideal
    length, bullets, action verbs, quantified impact and lexical richness —
    exactly the contract the production model will be trained against.
    """
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.integers(120, 1300, n_samples),    # word_count
        rng.integers(0, 45, n_samples),        # bullet_count
        rng.uniform(0.0, 8.0, n_samples),      # action_verb_density (per 100 words)
        rng.uniform(0.0, 6.0, n_samples),      # metric_density     (per 100 words)
        rng.binomial(1, 0.85, n_samples),      # has_email
        rng.binomial(1, 0.85, n_samples),      # has_phone
        rng.binomial(1, 0.60, n_samples),      # has_linkedin
        rng.integers(0, 7, n_samples),         # sections_present
        rng.uniform(0.0, 0.10, n_samples),     # tfidf_mean
        rng.uniform(0.05, 0.50, n_samples),    # tfidf_max
    ])
    y = (
        8 * X[:, 4] + 6 * X[:, 5] + 5 * X[:, 6]
        + 4 * X[:, 7]
        + 10 * np.minimum(X[:, 0] / 800.0, 1.0)
        + 8 * np.minimum(X[:, 1] / 12.0, 1.0)
        + 14 * np.minimum(X[:, 2] / 3.0, 1.0)
        + 15 * np.minimum(X[:, 3] / 2.0, 1.0)
        + 4 * np.minimum(X[:, 8] / 0.06, 1.0)
        + 5 * np.minimum(X[:, 9] / 0.30, 1.0)
    )
    y = np.clip(y + rng.normal(0, 2.5, n_samples), 0, 100)
    return X, y


class _NumpyRidgeHead:
    """Dependency-light fallback used only when sklearn's native ensemble
    extensions are unavailable (e.g. an Application Control policy blocking
    the compiled ``_sorting`` DLL).

    Fits a ridge regression in closed form over
    ``[bias, raw features, capped hinge transforms]`` — a basis that
    reconstructs the synthetic rubric almost exactly, so scores stay within
    a couple of points of the RandomForest head. Deterministic, NumPy-only.
    """

    def __init__(self, seed: int = 42) -> None:
        X, y = _synthetic_training_set(seed=seed)
        phi = self._basis(X)
        gram = phi.T @ phi + 1e-6 * np.eye(phi.shape[1])
        self._coef = np.linalg.solve(gram, phi.T @ y)

    @staticmethod
    def _basis(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        hinges = np.column_stack([
            np.minimum(X[:, 0] / 800.0, 1.0),
            np.minimum(X[:, 1] / 12.0, 1.0),
            np.minimum(X[:, 2] / 3.0, 1.0),
            np.minimum(X[:, 3] / 2.0, 1.0),
            np.minimum(X[:, 8] / 0.06, 1.0),
            np.minimum(X[:, 9] / 0.30, 1.0),
        ])
        return np.column_stack([np.ones(len(X)), X, hinges])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._basis(X) @ self._coef


def build_feature_vector(features: dict[str, Any], tfidf_mean: float, tfidf_max: float) -> np.ndarray:
    """Assemble the model input row in ``MODEL_FEATURES`` order."""
    return np.array([
        features["word_count"],
        features["bullet_count"],
        features["action_verb_density"],
        features["metric_density"],
        float(features["has_email"]),
        float(features["has_phone"]),
        float(features["has_linkedin"]),
        features["sections_present"],
        tfidf_mean,
        tfidf_max,
    ], dtype=float)


def _heuristic_subscore(f: dict[str, Any]) -> float:
    """Transparent 0-100 rubric used for the composite blend and UI explainability."""
    score = 0.0
    score += 8 if f["has_email"] else 0
    score += 6 if f["has_phone"] else 0
    score += 5 if f["has_linkedin"] else 0
    score += 3 if f["has_github"] or f["has_portfolio"] else 0
    score += min(f["sections_present"] / 5, 1.0) * 20
    score += min(f["action_verb_density"] / 3, 1.0) * 18
    score += min(f["metric_density"] / 2, 1.0) * 18
    wc = f["word_count"]
    if wc < 300:
        score += (wc / 300) * 15
    elif wc <= 900:
        score += 15
    else:
        score += max(0.0, 15 - (wc - 900) / 50)
    score += min(f["bullet_count"] / 10, 1.0) * 10
    return round(min(score, 100.0), 1)


def _grade(score: int) -> str:
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    return "D"

def _build_feedback(f: dict[str, Any]) -> list[str]:
    """Rule-based, actionable improvement tips derived from the heuristics."""
    tips: list[str] = []
    if not f["has_email"]:
        tips.append("Add a professional email address — ATS parsers reject resumes without contact info.")
    if not f["has_phone"]:
        tips.append("Include a phone number with country code (e.g. +91 98765 43210 or +1 415-555-0132).")
    if not f["has_linkedin"] and not f["has_github"] and not f["has_url"]:
        tips.append("Add a LinkedIn, GitHub or portfolio URL to boost recruiter trust signals.")
    elif not f["has_linkedin"]:
        tips.append("Add your LinkedIn profile URL — recruiters cross-check it most often.")
    if f["metric_count"] < 3:
        tips.append("Quantify impact: add at least 3 measurable achievements (%, $, users, time saved).")
    if f["action_verb_density"] < 1.5:
        tips.append("Start more bullets with strong action verbs (led, built, optimized, reduced…).")
    for section, label in (("summary", "Summary"), ("experience", "Experience"),
                           ("education", "Education"), ("skills", "Skills")):
        if section not in f["sections"]:
            tips.append(f"Add a clearly labelled '{label}' section header — ATS parsers segment resumes by headers.")
    if f["word_count"] < 300:
        tips.append("Resume looks thin (<300 words). Expand your experience with concrete outcomes.")
    elif f["word_count"] > 900:
        tips.append("Resume is long (>900 words). Trim to the most relevant 1–2 pages.")
    if f["bullet_count"] < 5:
        tips.append("Use bullet points for achievements — dense paragraphs are penalised by ATS tokenizers.")
    if not tips:
        tips.append("Strong fundamentals! Phase 3 will fine-tune this against a specific job description.")
    return tips


@dataclass
class ATSReport:
    """Full scoring contract consumed by the Streamlit dashboard."""
    overall_score: int
    grade: str
    ml_score: float
    heuristic_score: float
    model_backend: str
    features: dict[str, Any]
    skills: list[dict[str, Any]]
    sections_found: list[str]
    sections_missing: list[str]
    checks: dict[str, bool]
    feedback: list[str]


class ATSScorer:
    """End-to-end resume scorer: heuristics + TF-IDF + RandomForest head."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=512, ngram_range=(1, 2)
        )
        self.model = self._train_mock_model()

    def _train_mock_model(self) -> Any:
        """Train the deterministic mock scoring head on synthetic rubric data.

        Primary path: ``sklearn.ensemble.RandomForestRegressor``. Fallback
        path (sklearn native extensions unavailable): NumPy ridge head.
        Swap either for ``joblib.load("models/ats_rf.joblib")`` once a real
        recruiter-labelled dataset is available (Phase 2+ backlog).
        """
        if _SKLEARN_ENSEMBLE_AVAILABLE:
            X, y = _synthetic_training_set(seed=self.random_state)
            model = RandomForestRegressor(
                n_estimators=120, max_depth=12, random_state=self.random_state
            )
            model.fit(X, y)
            self.backend = "sklearn-random-forest"
            logger.info("Mock RandomForest ATS head trained on %d synthetic samples.", len(X))
        else:
            logger.warning(
                "sklearn.ensemble unavailable in this environment — "
                "using the deterministic NumPy ridge fallback head."
            )
            model = _NumpyRidgeHead(seed=self.random_state)
            self.backend = "numpy-ridge-fallback"
        return model

    def _tfidf_stats(self, text: str) -> tuple[float, float]:
        docs = _REFERENCE_CORPUS + [text.strip() or "empty document"]
        matrix = self.vectorizer.fit_transform(docs)
        resume_row = matrix[-1]
        return float(resume_row.mean()), float(resume_row.max())

    def score(self, text: str) -> ATSReport:
        """Score a resume and return the full :class:`ATSReport`."""
        text = text or ""
        feats = extract_heuristic_features(text)
        tfidf_mean, tfidf_max = self._tfidf_stats(text)
        vector = build_feature_vector(feats, tfidf_mean, tfidf_max).reshape(1, -1)
        ml_score = float(np.clip(self.model.predict(vector)[0], 0, 100))
        heuristic_score = _heuristic_subscore(feats)
        overall = int(round(np.clip(0.6 * ml_score + 0.4 * heuristic_score, 0, 100)))

        found = feats["sections"]
        checks = {
            "Email address detected": feats["has_email"],
            "Phone number detected": feats["has_phone"],
            "LinkedIn URL detected": feats["has_linkedin"],
            "GitHub / portfolio URL": feats["has_github"] or feats["has_portfolio"],
            "Summary / objective section": "summary" in found,
            "Experience section": "experience" in found,
            "Education section": "education" in found,
            "Skills section": "skills" in found,
            "Quantified achievements (≥3)": feats["metric_count"] >= 3,
            "Action-verb driven bullets (≥8)": feats["action_verb_count"] >= 8,
            "Adequate length (300–900 words)": 300 <= feats["word_count"] <= 900,
        }
        return ATSReport(
            overall_score=overall,
            grade=_grade(overall),
            ml_score=round(ml_score, 1),
            heuristic_score=heuristic_score,
            model_backend=self.backend,
            features=feats,
            skills=extract_skills(text),
            sections_found=found,
            sections_missing=[s for s in STANDARD_SECTIONS if s not in found],
            checks=checks,
            feedback=_build_feedback(feats),
        )


# Convenience singleton for non-Streamlit callers (app.py uses st.cache_resource).
_DEFAULT_SCORER: ATSScorer | None = None


def get_scorer() -> ATSScorer:
    global _DEFAULT_SCORER
    if _DEFAULT_SCORER is None:
        _DEFAULT_SCORER = ATSScorer()
    return _DEFAULT_SCORER


def score_resume(text: str) -> ATSReport:
    """One-call helper: score raw resume text with the shared scorer."""
    return get_scorer().score(text)


# ---------------------------------------------------------------------------
# Job-title suggestions (Phase 2 search prefill)
# ---------------------------------------------------------------------------
# Ordered by specificity: each rule fires when ANY of its skills is present.
_JOB_TITLE_RULES: list[tuple[tuple[str, ...], str]] = [
    (("Machine Learning", "Deep Learning", "NLP"), "Machine Learning Engineer"),
    (("Python", "Statistics", "Pandas"), "Data Scientist"),
    (("Data Analysis", "SQL", "Excel"), "Data Analyst"),
    (("React", "JavaScript", "TypeScript"), "Frontend Developer"),
    (("Node.js", "Django", "Flask", "FastAPI"), "Backend Developer"),
    (("Docker", "Kubernetes", "CI/CD", "Terraform"), "DevOps Engineer"),
    (("AWS", "Azure", "Google Cloud"), "Cloud Engineer"),
    (("FDA Regulations", "HIPAA"), "Regulatory Affairs Specialist"),
    (("Project Management", "Agile", "Jira"), "Project Manager"),
    (("Financial Modeling", "Risk Analysis"), "Financial Analyst"),
    (("Supply Chain",), "Supply Chain Analyst"),
    (("Salesforce",), "Salesforce Administrator"),
    (("SEO",), "Digital Marketing Specialist"),
]

_DEFAULT_TITLES: list[str] = ["Software Engineer"]


def suggest_job_titles(skills: list[dict[str, Any]] | None, max_titles: int = 3) -> list[str]:
    """Map extracted resume skills to likely target job titles.

    Used to prefill the Phase 2 job-search box. Returns up to ``max_titles``
    unique titles, falling back to a neutral default when no rule fires.
    """
    if not skills:
        return list(_DEFAULT_TITLES)
    names = {entry["skill"] for entry in skills}
    titles: list[str] = []
    for required, title in _JOB_TITLE_RULES:
        if title not in titles and any(skill in names for skill in required):
            titles.append(title)
    return titles[:max_titles] or list(_DEFAULT_TITLES)
