"""
auth.py
=======
Account management for the ATS Resume Analyzer: email + password sign-up,
verification-code activation, sign-in, and per-account resume persistence.

Storage is a local SQLite database (``app_users.db`` next to app.py) — zero
extra dependencies. Passwords are stored as salted PBKDF2-SHA256 hashes;
plaintext passwords are never persisted.

Verification codes are 6 digits. They are delivered by email when SMTP is
configured (via Streamlit secrets ``[smtp]`` or ``SMTP_*`` environment
variables — see :func:`send_verification_email`); otherwise the caller
surfaces the code in the UI ("dev mode") so the flow stays usable locally.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import smtplib
import sqlite3
import ssl
from datetime import datetime
from email.message import EmailMessage

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "app_users.db")

_PBKDF2_ITERATIONS = 200_000
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    email          TEXT PRIMARY KEY,
    password_hash  TEXT NOT NULL,
    salt           TEXT NOT NULL,
    verified       INTEGER NOT NULL DEFAULT 0,
    verify_code    TEXT,
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS resumes (
    email        TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    file_bytes   BLOB NOT NULL,
    parsed_text  TEXT NOT NULL,
    uploaded_at  TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    return conn


def _normalise(email: str) -> str:
    return (email or "").strip().lower()


def _hash_password(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex),
        _PBKDF2_ITERATIONS,
    ).hex()


def _new_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------
def create_user(email: str, password: str) -> tuple[bool, str, str | None]:
    """Register ``email`` (unverified). Returns ``(ok, message, code)``."""
    email = _normalise(email)
    if not _EMAIL_RE.match(email):
        return False, "Please enter a valid email address.", None
    if len(password) < 8:
        return False, "Password must be at least 8 characters long.", None
    salt = secrets.token_hex(16)
    code = _new_code()
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (email, password_hash, salt, verified,"
                " verify_code, created_at) VALUES (?, ?, ?, 0, ?, ?)",
                (email, _hash_password(password, salt), salt, code,
                 datetime.now().isoformat(timespec="seconds")),
            )
    except sqlite3.IntegrityError:
        return (False,
                "An account with this email already exists — sign in instead.",
                None)
    return True, "Account created.", code


def verify_user(email: str, code: str) -> tuple[bool, str]:
    """Activate the account when the emailed 6-digit code matches."""
    email = _normalise(email)
    with _connect() as conn:
        row = conn.execute(
            "SELECT verify_code, verified FROM users WHERE email = ?",
            (email,)).fetchone()
        if row is None:
            return False, "No account found for that email — sign up first."
        if row[1]:
            return True, "Account already verified."
        if not secrets.compare_digest((code or "").strip(), row[0] or ""):
            return False, "Incorrect verification code — check it and retry."
        conn.execute(
            "UPDATE users SET verified = 1, verify_code = NULL"
            " WHERE email = ?", (email,))
    return True, "Account verified."


def resend_code(email: str) -> tuple[bool, str, str | None]:
    """Rotate the verification code for an unverified account."""
    email = _normalise(email)
    code = _new_code()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE users SET verify_code = ? WHERE email = ? AND verified = 0",
            (code, email))
        if cur.rowcount == 0:
            return False, "No unverified account found for that email.", None
    return True, "A new verification code was generated.", code


def authenticate(email: str, password: str) -> tuple[bool, str]:
    """Verify credentials AND that the account has been activated."""
    email = _normalise(email)
    with _connect() as conn:
        row = conn.execute(
            "SELECT password_hash, salt, verified FROM users WHERE email = ?",
            (email,)).fetchone()
    if row is None:
        return False, "No account found for that email — sign up first."
    if _hash_password(password, row[1]) != row[0]:
        return False, "Incorrect password."
    if not row[2]:
        return (False,
                "Account not verified yet — enter your code in the "
                "✅ Verify Account tab.")
    return True, "Signed in."


# ---------------------------------------------------------------------------
# Per-account resume persistence
# ---------------------------------------------------------------------------
def save_resume(email: str, filename: str, file_bytes: bytes,
                parsed_text: str) -> None:
    """Attach (or replace) the resume stored under this account."""
    email = _normalise(email)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO resumes (email, filename, file_bytes, parsed_text,"
            " uploaded_at) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(email) DO UPDATE SET"
            " filename = excluded.filename,"
            " file_bytes = excluded.file_bytes,"
            " parsed_text = excluded.parsed_text,"
            " uploaded_at = excluded.uploaded_at",
            (email, filename, file_bytes, parsed_text,
             datetime.now().isoformat(timespec="seconds")),
        )


def load_resume(email: str) -> dict | None:
    """Return the saved resume dict, or None when the account has none."""
    email = _normalise(email)
    with _connect() as conn:
        row = conn.execute(
            "SELECT filename, file_bytes, parsed_text, uploaded_at"
            " FROM resumes WHERE email = ?", (email,)).fetchone()
    if row is None:
        return None
    return {"filename": row[0], "file_bytes": bytes(row[1]),
            "parsed_text": row[2], "uploaded_at": row[3]}


def delete_resume(email: str) -> None:
    email = _normalise(email)
    with _connect() as conn:
        conn.execute("DELETE FROM resumes WHERE email = ?", (email,))


# ---------------------------------------------------------------------------
# Verification email delivery
# ---------------------------------------------------------------------------
def _smtp_config() -> dict | None:
    """SMTP settings from ``st.secrets['smtp']`` or ``SMTP_*`` env vars.

    Returns None when unconfigured — the caller then shows the code in the
    UI (dev mode) instead of sending email.
    """
    cfg: dict = {}
    try:
        import streamlit as st
        cfg = dict(st.secrets.get("smtp", {}))
    except Exception:
        cfg = {}
    for key, env in (("host", "SMTP_HOST"), ("port", "SMTP_PORT"),
                     ("username", "SMTP_USERNAME"), ("password", "SMTP_PASSWORD"),
                     ("sender", "SMTP_SENDER")):
        if not cfg.get(key) and os.environ.get(env):
            cfg[key] = os.environ[env]
    return cfg if cfg.get("host") and cfg.get("username") else None


def send_verification_email(email: str, code: str) -> bool:
    """Email the 6-digit verification code.

    Returns False when SMTP is not configured or delivery fails — callers
    should fall back to displaying the code (dev mode).
    """
    cfg = _smtp_config()
    if cfg is None:
        return False
    msg = EmailMessage()
    msg["Subject"] = "Your ATS Resume Analyzer verification code"
    msg["From"] = cfg.get("sender") or cfg["username"]
    msg["To"] = email
    msg.set_content(
        "Welcome to the ATS Resume Analyzer!\n\n"
        f"Your verification code is: {code}\n\n"
        "Enter it in the app's 'Verify Account' tab to activate your account.\n"
        "If you did not create this account, you can ignore this email.\n"
    )
    port = int(cfg.get("port", 465))
    try:
        if port == 587:  # STARTTLS
            with smtplib.SMTP(cfg["host"], port, timeout=15) as server:
                server.starttls(context=ssl.create_default_context())
                server.login(cfg["username"], cfg["password"])
                server.send_message(msg)
        else:  # implicit SSL (465)
            with smtplib.SMTP_SSL(cfg["host"], port, timeout=15,
                                  context=ssl.create_default_context()) as server:
                server.login(cfg["username"], cfg["password"])
                server.send_message(msg)
        return True
    except Exception:
        return False
