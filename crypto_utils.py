"""
crypto_utils.py
---------------

Shared encryption for the Streamlit UI and backend runner.

Important:
    Do NOT hardcode Fernet keys in this file.
    The key must come from PAYROLL_ENC_KEY in Streamlit secrets/env.

Put this file in the repo root and replace the old crypto_utils.py.
"""

from __future__ import annotations

from cryptography.fernet import Fernet
from app_config import PAYROLL_ENC_KEY


def _get_fernet() -> Fernet:
    if not PAYROLL_ENC_KEY:
        raise RuntimeError(
            "PAYROLL_ENC_KEY is missing. Add it to Streamlit secrets or environment variables."
        )

    try:
        return Fernet(PAYROLL_ENC_KEY.encode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "PAYROLL_ENC_KEY is invalid. It must be a valid Fernet base64 key. "
            "Generate one with: from cryptography.fernet import Fernet; Fernet.generate_key()"
        ) from exc


_FERNET = _get_fernet()


def encrypt_str(value: str) -> str:
    """
    Encrypt a string to a Fernet token.
    """
    return _FERNET.encrypt((value or "").encode("utf-8")).decode("utf-8")


def decrypt_str(token: str) -> str:
    """
    Decrypt a Fernet token to a string.
    """
    return _FERNET.decrypt((token or "").encode("utf-8")).decode("utf-8")
