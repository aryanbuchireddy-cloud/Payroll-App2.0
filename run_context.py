"""
run_context.py
--------------

Per-user/per-run file handling.

Why this exists:
    A single Streamlit app with multiple users cannot save every run to:
        salondata_payroll.csv
    because users/runs can overwrite each other.

This module creates safe folders like:
    payroll_runs/quopayroll_gmail_com/20260428_223011_a1b2c3d4/salondata_payroll.csv

Use in payrollrunner_dbkeys_handyman.py:
    from run_context import make_run_context, save_download_as, cleanup_run_dir_if_allowed

    run_ctx = make_run_context(username)
    csv_path = run_ctx.path("salondata_payroll.csv")
    await download.save_as(csv_path)
"""

from __future__ import annotations

import os
import re
import time
import uuid
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from app_config import RUN_OUTPUT_ROOT, KEEP_LOCAL_PDFS


def safe_slug(value: str) -> str:
    """
    Makes usernames safe for folder names.
    """
    s = (value or "unknown").strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return s or "unknown"


def new_run_id() -> str:
    """
    Unique id per payroll run.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


@dataclass
class RunContext:
    username: str
    run_id: str
    run_dir: str

    def path(self, filename: str) -> str:
        """
        Return a safe path inside this run folder.
        """
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)
        clean_name = os.path.basename(filename or "file")
        return str(Path(self.run_dir) / clean_name)

    def exists(self, filename: str) -> bool:
        return Path(self.path(filename)).exists()


def make_run_context(username: str, run_id: Optional[str] = None) -> RunContext:
    rid = run_id or new_run_id()
    run_dir = Path(RUN_OUTPUT_ROOT) / safe_slug(username) / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(username=(username or "").strip().lower(), run_id=rid, run_dir=str(run_dir))


async def save_download_as(download, run_ctx: RunContext, filename: str) -> str:
    """
    Save a Playwright download into this run's folder.
    """
    path = run_ctx.path(filename)
    await download.save_as(path)
    return path


def cleanup_run_dir_if_allowed(run_ctx_or_path) -> None:
    """
    Deletes local temp files if KEEP_LOCAL_PDFS is false.

    During debugging, set KEEP_LOCAL_PDFS=1 so files stay visible.
    """
    if KEEP_LOCAL_PDFS:
        return

    path = run_ctx_or_path.run_dir if hasattr(run_ctx_or_path, "run_dir") else str(run_ctx_or_path or "")
    if not path:
        return

    try:
        p = Path(path)
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass
