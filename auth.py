"""
auth.py
=======
Account management for the ATS Resume Analyzer: email + password sign-up,
sign-in, and per-account resume persistence.

Storage is a local SQLite database (``app_users.db`` next to app.py) — zero
extra dependencies. Passwords are stored as salted PBKDF2-SHA256 hashes;
plaintext passwords are never persisted. Accounts are active immediately
after sign-up — no verification step, no external services required.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "app_users.db")

_PBKDF2_ITERATIONS = 200_000
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    email          TEXT PRIMARY KEY,
    password_hash  TEXT NOT NULL,
    salt           TEXT NOT NULL,
    verified       INTEGER NOT NULL DEFAULT 1,
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

# One-time cleanup for databases created by an older build: accounts left
# unverified by the retired e-mail-code step are activated transparently.
_MIGRATE = "UPDATE users SET verified = 1, verify_code = NULL WHERE verified = 0"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    conn.execute(_MIGRATE)
    return conn


def _normalise(email: str) -> str:
    return (email or "").strip().lower()


def _hash_password(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex),
        _PBKDF2_ITERATIONS,
    ).hex()


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------
def create_user(email: str, password: str) -> tuple[bool, str]:
    """Register ``email`` — active immediately. Returns ``(ok, message)``."""
    email = _normalise(email)
    if not _EMAIL_RE.match(email):
        return False, "Please enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    salt = secrets.token_hex(16)
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (email, password_hash, salt, verified,"
                " verify_code, created_at) VALUES (?, ?, ?, 1, NULL, ?)",
                (email, _hash_password(password, salt), salt,
                 datetime.now().isoformat(timespec="seconds")),
            )
    except sqlite3.IntegrityError:
        return (False,
                "An account with this email already exists — sign in instead.")
    return True, "Account created."


def authenticate(email: str, password: str) -> tuple[bool, str]:
    """Validate credentials. Returns ``(ok, message)``."""
    email = _normalise(email)
    with _connect() as conn:
        row = conn.execute(
            "SELECT password_hash, salt FROM users WHERE email = ?",
            (email,)).fetchone()
    if row is None:
        return False, "No account found for this email — sign up first."
    if _hash_password(password, row[1]) != row[0]:
        return False, "Incorrect password."
    return True, "Signed in."


# ---------------------------------------------------------------------------
# Per-account resume persistence
# ---------------------------------------------------------------------------
def save_resume(email: str, filename: str, file_bytes: bytes,
                parsed_text: str) -> None:
    """Insert or replace the account's saved resume."""
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
    """Remove the account's saved resume (no-op when none exists)."""
    email = _normalise(email)
    with _connect() as conn:
        conn.execute("DELETE FROM resumes WHERE email = ?", (email,))
