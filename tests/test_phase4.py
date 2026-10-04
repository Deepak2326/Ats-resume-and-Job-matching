"""
test_phase4.py
==============
Unit tests for the Phase 4 course recommendation engine (recommender.py).

Covers: curated-map hits, the dynamic platform-search fallback generator
(Coursera / Udemy / edX), per-platform top-ups, max_per_skill capping,
URL-encoding, case-insensitive lookup + dedupe, determinism, and alignment
with the shared skill lexicon (ats_scorer.SKILLS_DB).

Run (this machine): see run_tests.ps1 — uv base interpreter + venv PYTHONPATH.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from ats_scorer import SKILLS_DB                      # noqa: E402
from recommender import (                             # noqa: E402
    CURATED_COURSES,
    PLATFORM_COLORS,
    PLATFORM_SEARCH_URLS,
    platform_search_url,
    recommend_courses,
    recommend_for_skill,
)

CHECKS: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    suffix = f"  ->  {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")
    CHECKS.append(label)
    if not condition:
        raise SystemExit(f"FAILED: {label} {detail}")


DOMAIN = {"Coursera": "coursera.org", "Udemy": "udemy.com", "edX": "edx.org"}

# --- Empty / degenerate input ------------------------------------------------
check("empty input -> []", recommend_courses([]) == [])
check("None input -> []", recommend_courses(None) == [])
check("blank entries skipped", recommend_courses(["", "   "]) == [])
check("recommend_for_skill('') -> []", recommend_for_skill("") == [])

# --- Structure & invariants on the deterministic Phase 3 trio ----------------
recs = recommend_courses(["Deep Learning", "HIPAA", "PyTorch"])
check("3 skills x 3 recs = 9 recommendations", len(recs) == 9, f"got {len(recs)}")
check("every rec has exactly the required keys",
      all(set(r) == {"skill", "course_title", "platform", "url"} for r in recs))
check("platforms limited to Coursera/Udemy/edX",
      all(r["platform"] in PLATFORM_SEARCH_URLS for r in recs))
check("urls are absolute https on the platform's own domain",
      all(r["url"].startswith("https://") and DOMAIN[r["platform"]] in r["url"]
          for r in recs))

# --- Curated hits ------------------------------------------------------------
dl = [r for r in recs if r["skill"] == "Deep Learning"]
check("Deep Learning -> curated DeepLearning.AI specialization first",
      dl[0]["platform"] == "Coursera"
      and dl[0]["url"] == "https://www.coursera.org/specializations/deep-learning",
      dl[0]["url"])
pt = [r for r in recs if r["skill"] == "PyTorch"]
check("PyTorch -> curated Udemy bootcamp first",
      pt[0]["platform"] == "Udemy"
      and "pytorch-for-deep-learning" in pt[0]["url"], pt[0]["url"])

# --- Dynamic fallback (HIPAA is deliberately NOT in the curated map) ---------
hipaa = [r for r in recs if r["skill"] == "HIPAA"]
check("HIPAA -> pure fallback (3 search links)", len(hipaa) == 3)
check("fallback platform order Coursera, Udemy, edX",
      [r["platform"] for r in hipaa] == ["Coursera", "Udemy", "edX"])
check("fallback URLs match the spec templates exactly",
      [r["url"] for r in hipaa] == [
          "https://www.coursera.org/search?query=HIPAA",
          "https://www.udemy.com/courses/search/?q=HIPAA",
          "https://www.edx.org/search?q=HIPAA",
      ])
check("fallback titles name the skill and platform",
      all(r["course_title"] == f"Top-rated HIPAA courses on {r['platform']}"
          for r in hipaa))

# --- Top-up: curated skill is padded to max_per_skill with unused platforms --
check("curated skill topped up to 3 options",
      len(pt) == 3 and {r["platform"] for r in pt} == {"Coursera", "Udemy", "edX"})
check("one recommendation per platform per skill",
      len({r["platform"] for r in pt}) == len(pt))

# --- max_per_skill capping ----------------------------------------------------
check("max_per_skill=1 -> curated entry only",
      recommend_courses(["PyTorch"], max_per_skill=1) == [{
          "skill": "PyTorch",
          "course_title": "PyTorch for Deep Learning with Python Bootcamp",
          "platform": "Udemy",
          "url": "https://www.udemy.com/course/pytorch-for-deep-learning-with-python-bootcamp/",
      }])
check("max_per_skill=2 on a fallback skill -> 2 recs",
      len(recommend_courses(["HIPAA"], max_per_skill=2)) == 2)

# --- URL encoding --------------------------------------------------------------
ab = recommend_courses(["A/B Testing"], max_per_skill=1)
check("special chars URL-encoded (A/B Testing)",
      ab[0]["url"] == "https://www.coursera.org/search?query=A%2FB+Testing",
      ab[0]["url"])
check("platform_search_url encodes C++ correctly",
      platform_search_url("Udemy", "C++")
      == "https://www.udemy.com/courses/search/?q=C%2B%2B")

# --- Case-insensitive lookup + order-preserving dedupe -------------------------
ci = recommend_courses(["python", "PYTHON ", "Python"])
check("case-insensitive dedupe -> one skill, 3 recs", len(ci) == 3)
check("curated found regardless of case",
      ci[0]["url"] == "https://www.coursera.org/specializations/python", ci[0]["url"])
check("first-seen casing preserved in output", ci[0]["skill"] == "python")

# --- Determinism + unknown skills ----------------------------------------------
check("deterministic output for identical input",
      recommend_courses(["Docker", "HIPAA"]) == recommend_courses(["Docker", "HIPAA"]))
unk = recommend_for_skill("Underwater Basket Weaving")
check("unknown skill -> 3 fallback links with encoded query",
      len(unk) == 3
      and all("Underwater+Basket+Weaving" in r["url"] for r in unk))

# --- Curated map sanity + lexicon alignment -------------------------------------
check("curated map covers >= 25 popular skills",
      len(CURATED_COURSES) >= 25, f"{len(CURATED_COURSES)} curated")
check("all curated entries well-formed (keys, https, known platform)",
      all(set(c) == {"course_title", "platform", "url"}
          and c["url"].startswith("https://")
          and c["platform"] in PLATFORM_COLORS
          for courses in CURATED_COURSES.values() for c in courses))
check("curated keys align with the shared skill lexicon (SKILLS_DB)",
      set(CURATED_COURSES) <= set(SKILLS_DB))
check("every lexicon skill yields exactly 3 recommendations",
      all(len(recommend_for_skill(s)) == 3 for s in SKILLS_DB))

print(f"\n{len(CHECKS)}/{len(CHECKS)} checks passed — "
      "PHASE 4 RECOMMENDER VERIFIED (curated + fallback).")
