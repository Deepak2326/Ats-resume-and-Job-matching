"""
recommender.py
==============
Course recommendation engine — Phase 4.

Maps the skill gaps identified by the Phase 3 matcher
(``st.session_state["missing_skills"]``) to concrete, clickable course
recommendations:

* a curated map of popular skills -> specific, well-known courses on Coursera,
  Udemy and edX (direct course URLs), and
* a dynamic fallback generator producing platform *search* links for any skill
  outside the curated map, so every gap always yields recommendations.

Every recommendation is a plain dict::

    {"skill": str, "course_title": str, "platform": str, "url": str}

Pure and deterministic — no network calls, safe to unit-test and cheap enough
to call on every Streamlit rerun.
"""

from __future__ import annotations

from urllib.parse import quote_plus

# ---------------------------------------------------------------------------
# Platforms
# ---------------------------------------------------------------------------
#: Platform -> search URL template (``{q}`` = URL-encoded skill name).
PLATFORM_SEARCH_URLS: dict[str, str] = {
    "Coursera": "https://www.coursera.org/search?query={q}",
    "Udemy": "https://www.udemy.com/courses/search/?q={q}",
    "edX": "https://www.edx.org/search?q={q}",
}

#: Brand colors used by the Streamlit UI for platform badges/buttons.
PLATFORM_COLORS: dict[str, str] = {
    "Coursera": "#0056D2",
    "Udemy": "#A435F0",
    "edX": "#02262B",
}

#: Deterministic platform order for fallback generation and top-ups.
_PLATFORM_ORDER: tuple[str, ...] = ("Coursera", "Udemy", "edX")


def _course(platform: str, title: str, url: str) -> dict[str, str]:
    return {"course_title": title, "platform": platform, "url": url}


# ---------------------------------------------------------------------------
# Curated map: canonical skill (ats_scorer.SKILLS_DB names) -> known courses.
# Direct course URLs for popular, stable offerings; anything not listed here
# is handled by the dynamic fallback generator below.
# ---------------------------------------------------------------------------
CURATED_COURSES: dict[str, tuple[dict[str, str], ...]] = {
    # Languages & core data stack
    "Python": (_course(
        "Coursera", "Python for Everybody Specialization (University of Michigan)",
        "https://www.coursera.org/specializations/python"),),
    "SQL": (_course(
        "Coursera", "SQL for Data Science (UC Davis)",
        "https://www.coursera.org/learn/sql-for-data-science"),),
    "R": (_course(
        "Coursera", "Data Science: Foundations using R Specialization (JHU)",
        "https://www.coursera.org/specializations/data-science-foundations-r"),),
    "Java": (_course(
        "Coursera", "Java Programming & Software Engineering Fundamentals (Duke)",
        "https://www.coursera.org/specializations/java-programming"),),
    "JavaScript": (_course(
        "Coursera", "Programming with JavaScript (Meta)",
        "https://www.coursera.org/learn/programming-with-javascript"),),
    # Data science / ML
    "Machine Learning": (_course(
        "Coursera", "Machine Learning Specialization (DeepLearning.AI & Stanford)",
        "https://www.coursera.org/specializations/machine-learning-introduction"),),
    "Deep Learning": (_course(
        "Coursera", "Deep Learning Specialization (DeepLearning.AI)",
        "https://www.coursera.org/specializations/deep-learning"),),
    "NLP": (_course(
        "Coursera", "Natural Language Processing Specialization (DeepLearning.AI)",
        "https://www.coursera.org/specializations/natural-language-processing"),),
    "Data Analysis": (_course(
        "Coursera", "Google Data Analytics Professional Certificate",
        "https://www.coursera.org/professional-certificates/google-data-analytics"),),
    "Data Visualization": (_course(
        "Coursera", "Data Visualization with Tableau Specialization (UC Davis)",
        "https://www.coursera.org/specializations/data-visualization"),),
    "Statistics": (_course(
        "Coursera", "Introduction to Statistics (Stanford)",
        "https://www.coursera.org/learn/stanford-statistics"),),
    "Pandas": (_course(
        "Udemy", "Data Analysis with Pandas and Python",
        "https://www.udemy.com/course/data-analysis-with-pandas/"),),
    "Scikit-learn": (_course(
        "Coursera", "Applied Machine Learning in Python (University of Michigan)",
        "https://www.coursera.org/learn/python-machine-learning"),),
    "TensorFlow": (_course(
        "Coursera", "DeepLearning.AI TensorFlow Developer Professional Certificate",
        "https://www.coursera.org/professional-certificates/tensorflow-in-practice"),),
    "PyTorch": (_course(
        "Udemy", "PyTorch for Deep Learning with Python Bootcamp",
        "https://www.udemy.com/course/pytorch-for-deep-learning-with-python-bootcamp/"),),
    "Spark": (_course(
        "Coursera", "Introduction to Big Data with Spark and Hadoop (IBM)",
        "https://www.coursera.org/learn/introduction-to-big-data-with-spark-hadoop"),),
    # BI & cloud / devops
    "Excel": (_course(
        "Coursera", "Excel Skills for Business Specialization (Macquarie)",
        "https://www.coursera.org/specializations/excel"),),
    "Tableau": (_course(
        "Coursera", "Data Visualization with Tableau Specialization (UC Davis)",
        "https://www.coursera.org/specializations/data-visualization"),),
    "Power BI": (_course(
        "Udemy", "Microsoft Power BI Desktop for Business Intelligence (Maven Analytics)",
        "https://www.udemy.com/course/microsoft-power-bi-up-running-with-power-bi-desktop/"),),
    "AWS": (_course(
        "Coursera", "AWS Cloud Technical Essentials (Amazon Web Services)",
        "https://www.coursera.org/learn/aws-cloud-technical-essentials"),),
    "Azure": (_course(
        "Coursera", "Microsoft Azure Fundamentals AZ-900 Specialization",
        "https://www.coursera.org/specializations/microsoft-azure-fundamentals-az-900"),),
    "Google Cloud": (_course(
        "Coursera", "Google Cloud Fundamentals: Core Infrastructure",
        "https://www.coursera.org/learn/gcp-fundamentals"),),
    "Docker": (_course(
        "Udemy", "Docker Mastery: with Kubernetes + Swarm (Bret Fisher)",
        "https://www.udemy.com/course/docker-mastery/"),),
    "Kubernetes": (_course(
        "Udemy", "Kubernetes for the Absolute Beginners — Hands-on (KodeKloud)",
        "https://www.udemy.com/course/learn-kubernetes/"),),
    "Git": (_course(
        "Coursera", "Version Control with Git (Atlassian)",
        "https://www.coursera.org/learn/version-control-with-git"),),
    # Web & product
    "React": (_course(
        "Coursera", "React Basics (Meta)",
        "https://www.coursera.org/learn/react-basics"),),
    "Agile": (_course(
        "Coursera", "Agile with Atlassian Jira",
        "https://www.coursera.org/learn/agile-atlassian-jira"),),
    # Business, compliance & soft skills
    "Project Management": (_course(
        "Coursera", "Google Project Management: Professional Certificate",
        "https://www.coursera.org/professional-certificates/google-project-management"),),
    "Communication": (_course(
        "Coursera", "Improving Communication Skills (Wharton, UPenn)",
        "https://www.coursera.org/learn/wharton-communication-skills"),),
    "Financial Modeling": (_course(
        "Coursera", "Business and Financial Modeling Specialization (Wharton)",
        "https://www.coursera.org/specializations/wharton-business-financial-modeling"),),
    "Supply Chain": (_course(
        "Coursera", "Supply Chain Management Specialization (Rutgers)",
        "https://www.coursera.org/specializations/supply-chain-management"),),
    "SEO": (_course(
        "Coursera", "Search Engine Optimization (SEO) Specialization (UC Davis)",
        "https://www.coursera.org/specializations/seo"),),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _norm(skill: str) -> str:
    return (skill or "").strip().casefold()


# Case-insensitive lookup over the curated map.
_CURATED_LOOKUP: dict[str, tuple[dict[str, str], ...]] = {
    _norm(skill): courses for skill, courses in CURATED_COURSES.items()
}


def platform_search_url(platform: str, skill: str) -> str:
    """Platform search URL for an arbitrary skill (dynamic fallback link)."""
    template = PLATFORM_SEARCH_URLS[platform]
    return template.format(q=quote_plus(skill.strip()))


def _fallback_entry(skill: str, platform: str) -> dict[str, str]:
    return {
        "skill": skill,
        "course_title": f"Top-rated {skill} courses on {platform}",
        "platform": platform,
        "url": platform_search_url(platform, skill),
    }


def recommend_for_skill(skill: str, max_per_skill: int = 3) -> list[dict[str, str]]:
    """Up to ``max_per_skill`` course recommendations for a single skill.

    Curated entries come first; the list is then topped up with platform
    search links (for platforms not already covered) so every skill always
    yields ``max_per_skill`` options.  Returns ``[]`` for blank input.
    """
    skill = (skill or "").strip()
    if not skill or max_per_skill < 1:
        return []
    entries: list[dict[str, str]] = [
        {"skill": skill, **course}
        for course in _CURATED_LOOKUP.get(_norm(skill), ())
    ]
    used = {entry["platform"] for entry in entries}
    for platform in _PLATFORM_ORDER:
        if len(entries) >= max_per_skill:
            break
        if platform not in used:
            entries.append(_fallback_entry(skill, platform))
            used.add(platform)
    return entries[:max_per_skill]


def recommend_courses(skills, max_per_skill: int = 3) -> list[dict[str, str]]:
    """Flat recommendation list for a missing-skills list.

    * Accepts ``st.session_state["missing_skills"]`` (any iterable of str).
    * De-duplicates skills case-insensitively (order-preserving); blanks
      skipped; ``None``/empty input -> ``[]``.
    * Deterministic: identical input -> identical output, no network calls.
    """
    seen: set[str] = set()
    recommendations: list[dict[str, str]] = []
    for raw in skills or []:
        skill = str(raw).strip()
        key = skill.casefold()
        if not skill or key in seen:
            continue
        seen.add(key)
        recommendations.extend(recommend_for_skill(skill, max_per_skill))
    return recommendations
