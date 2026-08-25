"""
app_config.py
-------------

Shared config for the Streamlit UI and backend runner.

Why this exists:
    Your UI and backend must use the SAME:
        - MONGO_URI
        - MONGO_DB
        - user collection name
        - employee-key collection name
        - PAYROLL_ENC_KEY
        - output folders

Put this file in the repo root next to:
    tester8_admin_handyman.py
    payrollrunner_dbkeys_handyman.py
    crypto_utils.py
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import streamlit as st
except Exception:
    st = None


def get_secret_or_env(name: str, default: str = "") -> str:
    """
    Read Streamlit secrets first, then environment variables.
    Works both inside Streamlit and inside backend/Playwright code.
    """
    try:
        if st is not None:
            value = st.secrets.get(name, None)
            if value not in (None, ""):
                return str(value).strip()
    except Exception:
        pass

    value = os.getenv(name, "").strip()
    if value:
        return value

    return default


# Mongo
MONGO_URI = get_secret_or_env("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = get_secret_or_env("MONGO_DB", "payrollapp")
MONGO_USERS_COLL = get_secret_or_env("MONGO_USERS_COLL", "userInfo")
MONGO_LOGIN_EVENTS_COLL = get_secret_or_env("MONGO_LOGIN_EVENTS_COLL", "loginEvents")
MONGO_KEYS_COLL = get_secret_or_env("MONGO_KEYS_COLL", "employeeKeysByUser")
MONGO_PDF_GRIDFS_BUCKET = get_secret_or_env("MONGO_PDF_GRIDFS_BUCKET", "payroll_pdfs")

# Encryption
PAYROLL_ENC_KEY = get_secret_or_env("PAYROLL_ENC_KEY", "")

# Local run/output folders
RUN_OUTPUT_ROOT = get_secret_or_env("RUN_OUTPUT_ROOT", "payroll_runs")
PDF_OUTPUT_DIR = get_secret_or_env("PDF_OUTPUT_DIR", "payroll_pdfs")

KEEP_LOCAL_PDFS = (
    get_secret_or_env("KEEP_LOCAL_PDFS", "0").strip().lower()
    in ("1", "true", "yes", "y")
)

# App behavior
DEFAULT_ORG_ID = get_secret_or_env("DEFAULT_ORG_ID", "default")
SEED_DEFAULT_ADMIN = (
    get_secret_or_env("SEED_DEFAULT_ADMIN", "1").strip().lower()
    in ("1", "true", "yes", "y")
)

DEFAULT_ADMIN_USERNAME = get_secret_or_env("DEFAULT_ADMIN_USERNAME", "owner@example.com")
DEFAULT_ADMIN_PASSWORD = get_secret_or_env("DEFAULT_ADMIN_PASSWORD", "changeme")

ALLOW_SELF_SIGNUP = (
    get_secret_or_env("ALLOW_SELF_SIGNUP", "0").strip().lower()
    in ("1", "true", "yes", "y")
)


def ensure_output_dirs() -> None:
    """
    Creates local folders used by backend runs.
    Safe to call from UI or backend.
    """
    Path(RUN_OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    Path(PDF_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


ensure_output_dirs()
