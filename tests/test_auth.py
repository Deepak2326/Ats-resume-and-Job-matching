"""
test_auth.py
============
Tests for auth.py (email/password accounts, per-account resume persistence)
plus an AppTest end-to-end run of the auth screen:
sign-up -> sign-in -> saved-resume restore across sessions.

The test database is isolated in a temp dir (auth.DB_PATH is monkeypatched
BEFORE any auth call; AppTest runs in-process, so the patch also covers the
code paths app.py triggers).

Run (this machine): see run_tests.ps1 — uv base interpreter + venv PYTHONPATH.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import auth                                     # noqa: E402

_TMP = tempfile.mkdtemp(prefix="ats_auth_test_")
auth.DB_PATH = os.path.join(_TMP, "test_users.db")

from streamlit.testing.v1 import AppTest        # noqa: E402
from test_phase1 import SAMPLE_RESUME_LINES, build_sample_pdf  # noqa: E402

APP = os.path.join(ROOT, "app.py")
EMAIL = "Jane.Doe@Example.com"                  # mixed case on purpose
PASSWORD = "s3cure-password"

CHECKS: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    suffix = f"  ->  {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")
    CHECKS.append(label)
    if not condition:
        raise SystemExit(f"FAILED: {label} {detail}")


def _set(at: AppTest, key: str, value: str) -> None:
    els = [el for el in at.text_input if el.key == key]
    check(f"input '{key}' rendered", len(els) == 1)
    els[0].set_value(value)


def _click(at: AppTest, needle: str) -> None:
    btns = [b for b in at.button if needle in str(b.label)]
    check(f"button '{needle}' rendered", len(btns) == 1)
    btns[0].click()


# --- Engine level: accounts ---------------------------------------------------
ok, msg = auth.create_user(EMAIL, PASSWORD)
check("sign-up succeeds", ok and "created" in msg.lower())
check("duplicate email rejected (case-insensitive)",
      auth.create_user("jane.doe@example.com", PASSWORD)[0] is False)
check("invalid email rejected",
      auth.create_user("not-an-email", PASSWORD)[0] is False)
check("short password rejected (< 8 chars)",
      auth.create_user("fresh@example.com", "short")[0] is False)

check("sign-in works immediately after sign-up (no verification step)",
      auth.authenticate(EMAIL, PASSWORD)[0])
check("sign-in is case-insensitive on email",
      auth.authenticate("jane.doe@example.com", PASSWORD)[0])
check("wrong password rejected",
      auth.authenticate(EMAIL, "wrong-password")[0] is False)
ok, msg = auth.authenticate("nobody@example.com", PASSWORD)
check("unknown email rejected with a sign-up hint",
      not ok and "sign up" in msg.lower())

# Rows left over from the verification era must still sign in (auto-migrated).
_legacy_salt = auth.secrets.token_hex(16)
with sqlite3.connect(auth.DB_PATH) as conn:
    conn.execute(
        "INSERT INTO users (email, password_hash, salt, verified,"
        " verify_code, created_at) VALUES (?, ?, ?, 0, '123456', ?)",
        ("legacy@example.com",
         auth._hash_password("legacy-pass-1", _legacy_salt),
         _legacy_salt, "2024-01-01T00:00:00"),
    )
check("legacy unverified account can sign in (auto-migrated)",
      auth.authenticate("legacy@example.com", "legacy-pass-1")[0])
with sqlite3.connect(auth.DB_PATH) as conn:
    row = conn.execute(
        "SELECT verified, verify_code FROM users"
        " WHERE email = 'legacy@example.com'").fetchone()
check("migration activated the account and cleared the code",
      row == (1, None))

# Password storage
with sqlite3.connect(auth.DB_PATH) as conn:
    row = conn.execute(
        "SELECT password_hash, salt FROM users WHERE email = ?",
        (EMAIL.lower(),)).fetchone()
check("password stored as a hash, never plaintext",
      row is not None and row[0] != PASSWORD and PASSWORD not in row[0])
check("per-user salt stored alongside the hash",
      row is not None and len(row[1]) == 32)

# --- Engine level: per-account resume persistence -----------------------------
pdf = build_sample_pdf(SAMPLE_RESUME_LINES)
check("no resume saved yet", auth.load_resume(EMAIL) is None)
auth.save_resume(EMAIL, "sample_resume.pdf", pdf,
                 "\n".join(SAMPLE_RESUME_LINES))
saved = auth.load_resume(EMAIL)
check("resume round-trips through the DB",
      saved is not None and saved["filename"] == "sample_resume.pdf"
      and saved["file_bytes"] == pdf)
auth.save_resume(EMAIL, "sample_resume_v2.pdf", pdf, "v2 text")
saved = auth.load_resume(EMAIL)
check("saving again replaces it (one resume per account)",
      saved is not None and saved["filename"] == "sample_resume_v2.pdf")
auth.delete_resume(EMAIL)
check("delete removes the saved resume", auth.load_resume(EMAIL) is None)
auth.save_resume(EMAIL, "sample_resume.pdf", pdf,
                 "\n".join(SAMPLE_RESUME_LINES))
check("resume re-saved for the UI run", auth.load_resume(EMAIL) is not None)

# --- UI: auth screen ----------------------------------------------------------
at = AppTest.from_file(APP, default_timeout=60)
at.run()
check("no exception on first paint", not at.exception)
check("login view shows the Sign In/Sign Up switcher, no app tabs",
      len(at.radio) == 1 and len(at.tabs) == 0)
check("no verification UI anywhere on the auth screen",
      len([el for el in at.text_input if "verify" in str(el.key)]) == 0)

# Sign-up: validation first, then success -> straight back to Sign In.
at.radio[0].set_value("Sign Up")
at.run()
_set(at, "signup_email", "ui-user@example.com")
_set(at, "signup_pw1", "pass-one-123")
_set(at, "signup_pw2", "pass-two-456")
_click(at, "Create Account")
at.run()
check("mismatched passwords rejected",
      any("do not match" in str(e.value) for e in at.error))
_set(at, "signup_pw1", "pass-one-123")
_set(at, "signup_pw2", "pass-one-123")
_click(at, "Create Account")
at.run()
check("sign-up lands back on Sign In with a success banner",
      len([el for el in at.text_input if el.key == "signin_email"]) == 1
      and any("Account created" in str(s.value) for s in at.success))
check("switcher reset to Sign In", at.radio[0].value == "Sign In")
check("no verification step after sign-up",
      len([el for el in at.text_input if "verify" in str(el.key)]) == 0)
check("engine agrees the new account can sign in",
      auth.authenticate("ui-user@example.com", "pass-one-123")[0] is True)

# Sign-up validation: weak password.
at.radio[0].set_value("Sign Up")
at.run()
_set(at, "signup_email", "weak@example.com")
_set(at, "signup_pw1", "short")
_set(at, "signup_pw2", "short")
_click(at, "Create Account")
at.run()
check("weak password rejected at sign-up",
      any("8 characters" in str(e.value) for e in at.error))
at.radio[0].set_value("Sign In")
at.run()

# Sign-in: wrong password, then success.
_set(at, "signin_email", "ui-user@example.com")
_set(at, "signin_password", "wrong-pass-999")
_click(at, "Sign In")
at.run()
check("wrong password error, stays logged out",
      not at.session_state["authenticated"]
      and any("Incorrect password" in str(e.value) for e in at.error))
_set(at, "signin_email", "ui-user@example.com")
_set(at, "signin_password", "pass-one-123")
_click(at, "Sign In")
at.run()
check("sign-in succeeds right after sign-up",
      at.session_state["authenticated"] is True)

# Sign out, then sign in as the account that has a saved resume.
_logout = [b for b in at.sidebar.button if "Log out" in str(b.label)]
check("Log out button rendered", len(_logout) == 1)
_logout[0].click()
at.run()
check("logout returns to the clean auth screen",
      at.session_state["authenticated"] is False and len(at.tabs) == 0
      and len(at.radio) == 1)

# --- UI: sign-in restores the saved resume ------------------------------------
_set(at, "signin_email", EMAIL)
_set(at, "signin_password", PASSWORD)
_click(at, "Sign In")
at.run()
check("no exception on sign-in", not at.exception)
check("authenticated after sign-in",
      at.session_state["authenticated"] is True)
check("username normalised to lowercase email",
      at.session_state["username"] == EMAIL.lower())
check("4 app tabs after sign-in", len(at.tabs) == 4)
check("saved resume auto-restored on sign-in (no re-upload needed)",
      at.session_state["resume"] is not None
      and at.session_state["resume"].filename == "sample_resume.pdf")
check("ATS report restored alongside the resume",
      at.session_state["ats_report"] is not None)
check("saved-resume caption shown in the upload tab",
      any("saved to your account" in str(c.value) for c in at.caption))
check("roadmap expander gone from the sidebar",
      len(at.sidebar.expander) == 0)
check("'Remove Saved Resume' button offered when a resume is loaded",
      len([b for b in at.sidebar.button
           if "Remove Saved Resume" in str(b.label)]) == 1)

# --- UI: logout, then sign in again -> resume STILL there ----------------------
_click_sidebar = [b for b in at.sidebar.button if "Log out" in str(b.label)]
check("Log out button still rendered", len(_click_sidebar) == 1)
_click_sidebar[0].click()
at.run()
check("second logout returns to the auth screen",
      at.session_state["authenticated"] is False and len(at.tabs) == 0)

_set(at, "signin_email", EMAIL)
_set(at, "signin_password", PASSWORD)
_click(at, "Sign In")
at.run()
check("resume survives logout + fresh sign-in (account persistence)",
      at.session_state["authenticated"] is True
      and at.session_state["resume"] is not None
      and at.session_state["resume"].filename == "sample_resume.pdf")

print(f"\n{len(CHECKS)}/{len(CHECKS)} checks passed — "
      "AUTH VERIFIED (sign-up, sign-in, persistence).")
