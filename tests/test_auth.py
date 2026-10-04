"""
test_auth.py
============
Tests for auth.py (email/password accounts, verification codes, per-account
resume persistence) plus an AppTest end-to-end run of the auth screen:
sign-up -> verify -> sign-in -> saved-resume restore across sessions.

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


# --- Engine level: accounts & verification -----------------------------------
ok, msg, code = auth.create_user(EMAIL, PASSWORD)
check("sign-up succeeds and returns a code", ok and code is not None)
check("verification code is 6 digits",
      code is not None and len(code) == 6 and code.isdigit(), code)
check("duplicate email rejected (case-insensitive)",
      auth.create_user("jane.doe@example.com", PASSWORD)[0] is False)
check("invalid email rejected",
      auth.create_user("not-an-email", PASSWORD)[0] is False)
check("short password rejected (< 8 chars)",
      auth.create_user("fresh@example.com", "short")[0] is False)

ok, msg = auth.authenticate(EMAIL, PASSWORD)
check("sign-in blocked before verification", not ok and "not verified" in msg)
wrong = "000000" if code != "000000" else "111111"
check("wrong code rejected", auth.verify_user(EMAIL, wrong)[0] is False)
check("correct code verifies the account", auth.verify_user(EMAIL, code)[0])
check("re-verify is idempotent", auth.verify_user(EMAIL, code)[0])
check("sign-in succeeds after verification (mixed-case email)",
      auth.authenticate(EMAIL, PASSWORD)[0])
check("wrong password rejected",
      auth.authenticate(EMAIL, "wrong-password")[0] is False)
check("unknown email rejected",
      auth.authenticate("nobody@example.com", PASSWORD)[0] is False)

with sqlite3.connect(auth.DB_PATH) as conn:
    row = conn.execute(
        "SELECT password_hash, salt, verify_code FROM users WHERE email = ?",
        (EMAIL.lower(),)).fetchone()
check("plaintext password never stored", PASSWORD not in str(row))
check("verification code cleared after activation", row[2] is None)

ok, _, code1 = auth.create_user("second@example.com", "another-pass-1")
ok, _, code2 = auth.resend_code("second@example.com")
check("resend rotates the code", ok and code2 is not None and code2 != code1)
check("old code invalidated by resend",
      auth.verify_user("second@example.com", code1)[0] is False)
check("new code verifies", auth.verify_user("second@example.com", code2)[0])
check("resend rejected once verified",
      auth.resend_code("second@example.com")[0] is False)

# --- Engine level: resume persistence -----------------------------------------
pdf = build_sample_pdf(SAMPLE_RESUME_LINES)
payload = bytes(pdf) if isinstance(pdf, (bytes, bytearray)) else pdf.getvalue()
auth.save_resume(EMAIL, "sample_resume.pdf", payload, "parsed text")
saved = auth.load_resume(EMAIL)
check("resume saved + loads back byte-identical",
      saved is not None and saved["file_bytes"] == payload
      and saved["filename"] == "sample_resume.pdf")
auth.save_resume(EMAIL, "v2.pdf", b"new-bytes", "v2 text")
saved = auth.load_resume(EMAIL)
check("re-upload replaces the saved resume",
      saved is not None and saved["filename"] == "v2.pdf"
      and saved["file_bytes"] == b"new-bytes")
auth.delete_resume(EMAIL)
check("delete removes the saved resume", auth.load_resume(EMAIL) is None)
# restore the real sample for the UI flow below
auth.save_resume(EMAIL, "sample_resume.pdf", payload, "parsed text")
check("no SMTP configured -> dev mode (email not sent)",
      auth.send_verification_email(EMAIL, "123456") is False)
print("[INFO] auth engine verified")

# --- UI: auth screen renders ---------------------------------------------------
at = AppTest.from_file(APP, default_timeout=120)
at.run()
check("no exception on auth screen", not at.exception)
check("Sign In / Sign Up / Verify tabs rendered",
      len(at.tabs) == 3
      and "Sign In" in str(at.tabs[0].label)
      and "Sign Up" in str(at.tabs[1].label)
      and "Verify" in str(at.tabs[2].label))
check("no phase/roadmap language on the auth screen",
      not any("phase" in str(m.value).lower() or "roadmap" in str(m.value).lower()
              for m in at.markdown)
      and not any("phase" in str(c.value).lower() or "roadmap" in str(c.value).lower()
                  for c in at.caption))

# --- UI: sign-up via the form (dev mode reveals the code) ----------------------
_set(at, "signup_email", "ui-user@example.com")
_set(at, "signup_pw1", "pass-one-123")
_set(at, "signup_pw2", "pass-two-456")
_click(at, "Create Account")
at.run()
check("mismatched passwords rejected", not at.exception
      and any("do not match" in str(e.value) for e in at.error))

_set(at, "signup_pw1", "pass-one-123")
_set(at, "signup_pw2", "pass-one-123")
_click(at, "Create Account")
at.run()
check("no exception on sign-up", not at.exception)
check("dev-mode verification code surfaced",
      any("Dev mode" in str(i.value) for i in at.info))
with sqlite3.connect(auth.DB_PATH) as conn:
    row = conn.execute(
        "SELECT verify_code FROM users WHERE email = 'ui-user@example.com'"
    ).fetchone()
check("account persisted as unverified with a code", row is not None and row[0])
ui_code = row[0]

_set(at, "verify_email", "ui-user@example.com")
_set(at, "verify_code", ui_code)
_click(at, "Verify Account")
at.run()
check("account verified via the UI",
      any("You can sign in now" in str(s.value) for s in at.success))
check("engine agrees the account is verified",
      auth.authenticate("ui-user@example.com", "pass-one-123")[0] is True)

# --- UI: sign-in restores the saved resume --------------------------------------
_set(at, "signin_email", EMAIL)
_set(at, "signin_password", PASSWORD)
_click(at, "Sign In")
at.run()
check("no exception on sign-in", not at.exception)
check("authenticated after sign-in", at.session_state["authenticated"] is True)
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
      len([b for b in at.sidebar.button if "Remove Saved Resume" in str(b.label)]) == 1)

# --- UI: logout, then sign in again -> resume STILL there ------------------------
_click_sidebar = [b for b in at.sidebar.button if "Log out" in str(b.label)]
check("Log out button rendered", len(_click_sidebar) == 1)
_click_sidebar[0].click()
at.run()
check("logout returns to the auth screen",
      at.session_state["authenticated"] is False and len(at.tabs) == 3)

_set(at, "signin_email", EMAIL)
_set(at, "signin_password", PASSWORD)
_click(at, "Sign In")
at.run()
check("resume survives logout + fresh sign-in (account persistence)",
      at.session_state["authenticated"] is True
      and at.session_state["resume"] is not None
      and at.session_state["resume"].filename == "sample_resume.pdf")

print(f"\n{len(CHECKS)}/{len(CHECKS)} checks passed — "
      "AUTH VERIFIED (sign-up, verification, persistence).")
