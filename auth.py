"""
auth.py
=======
Account management for the ATS Resume Analyzer: email + password sign-up,
sign-in, and per-account resume persistence.

Storage is a local SQLite database (``app_users.db`` next to app.py) — zero
extra dependencies. Passwords are stored as salted PBKDF2-SHA256 hashes;
plaintext passwords are never persisted. Accounts are active immediately
after sign-up — no verification step, no external services required.

Multi-user by design: any number of users can create accounts; the only
constraint is one account per email address. Databases created by older
builds are migrated transparently on first connect.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime

logger = logging.getLogger(__name__)

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

# Columns every build relies on, with ALTER-friendly definitions — databases
# created by any older build gain missing columns automatically on connect.
_USERS_COLUMNS = {
    "password_hash": "TEXT NOT NULL DEFAULT ''",
    "salt": "TEXT NOT NULL DEFAULT ''",
    "verified": "INTEGER NOT NULL DEFAULT 1",
    "verify_code": "TEXT",
    "created_at": "TEXT NOT NULL DEFAULT ''",
}
_RESUMES_COLUMNS = {
    "filename": "TEXT NOT NULL DEFAULT ''",
    "file_bytes": "BLOB NOT NULL DEFAULT X''",
    "parsed_text": "TEXT NOT NULL DEFAULT ''",
    "uploaded_at": "TEXT NOT NULL DEFAULT ''",
}

# Cleanup for databases created by the retired e-mail-code build: accounts
# left unverified are activated transparently.
_MIGRATE = "UPDATE users SET verified = 1, verify_code = NULL WHERE verified = 0"

_UNAVAILABLE = ("The account service is temporarily unavailable — "
                "please try again in a moment.")


def _ensure_columns(conn: sqlite3.Connection, table: str,
                    wanted: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for column, definition in wanted.items():
        if column not in existing:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _connect() -> sqlite3.Connection:
    """Open the DB and bring the schema up to date.

    Retries briefly on transient locks (e.g. a concurrent write from another
    session or process) so a momentary "database is locked" never surfaces to
    the user as a failed sign-up/sign-in.
    """
    last_exc: sqlite3.Error | None = None
    for attempt in range(3):
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(DB_PATH, timeout=15)
            conn.executescript(_SCHEMA)
            _ensure_columns(conn, "users", _USERS_COLUMNS)
            _ensure_columns(conn, "resumes", _RESUMES_COLUMNS)
            conn.execute(_MIGRATE)
            conn.commit()
            return conn
        except sqlite3.Error as exc:
            last_exc = exc
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            if attempt < 2:
                time.sleep(0.3 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


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
    """Register ``email`` — active immediately. Returns ``(ok, message)``.

    Never raises for storage problems: any database failure is logged and
    reported as a friendly ``(False, message)`` so the UI stays clean.
    """
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
    except Exception:  # noqa: BLE001 - auth must never crash the UI
        logger.exception("create_user failed for %s", email)
        return False, _UNAVAILABLE
    return True, "Account created."


def authenticate(email: str, password: str) -> tuple[bool, str]:
    """Validate credentials. Returns ``(ok, message)`` — never raises."""
    email = _normalise(email)
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT password_hash, salt FROM users WHERE email = ?",
                (email,)).fetchone()
    except Exception:  # noqa: BLE001 - auth must never crash the UI
        logger.exception("authenticate failed for %s", email)
        return False, _UNAVAILABLE
    if row is None:
        return False, "No account found for this email — sign up first."
    try:
        candidate = _hash_password(password, row[1])
    except (ValueError, TypeError):
        # Row written by an incompatible older build (unusable salt) — fail
        # the login cleanly instead of crashing the screen.
        logger.warning("legacy user row with unusable salt for %s", email)
        return False, "Incorrect password."
    if candidate != row[0]:
        return False, "Incorrect password."
    return True, "Signed in."


# ---------------------------------------------------------------------------
# Per-account resume persistence
# ---------------------------------------------------------------------------
def save_resume(email: str, filename: str, file_bytes: bytes,
                parsed_text: str) -> None:
    """Insert or replace the account's saved resume (failures logged only)."""
    email = _normalise(email)
    try:
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
    except sqlite3.Error:
        logger.exception("save_resume failed for %s", email)


def load_resume(email: str) -> dict | None:
    """Return the saved resume dict, or None when unavailable/none exists."""
    email = _normalise(email)
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT filename, file_bytes, parsed_text, uploaded_at"
                " FROM resumes WHERE email = ?", (email,)).fetchone()
    except sqlite3.Error:
        logger.exception("load_resume failed for %s", email)
        return None
    if row is None:
        return None
    return {"filename": row[0], "file_bytes": bytes(row[1]),
            "parsed_text": row[2], "uploaded_at": row[3]}


def delete_resume(email: str) -> None:
    """Remove the account's saved resume (no-op when none exists)."""
    email = _normalise(email)
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM resumes WHERE email = ?", (email,))
    except sqlite3.Error:
        logger.exception("delete_resume failed for %s", email)
