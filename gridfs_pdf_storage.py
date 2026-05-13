"""
gridfs_pdf_storage.py
---------------------

Shared PDF storage helpers.

Use this when the backend generates a validation PDF:
    from gridfs_pdf_storage import store_and_log_pdf

    item = store_and_log_pdf(username, pdf_path, period_end)
"""

from __future__ import annotations

import os
import time
from typing import Optional, Dict, Any

from bson import ObjectId

from app_config import KEEP_LOCAL_PDFS
from mongo_helpers import get_users_collection, get_pdf_gridfs, norm_username


def _delete_old_pdf_for_period(username: str, period_end: str) -> None:
    uname = norm_username(username)
    period_end = (period_end or "").strip()
    if not period_end:
        return

    fs = get_pdf_gridfs()
    users = get_users_collection()
    doc = users.find_one({"username": uname}, {"pdf_history": 1}) or {}

    for h in (doc.get("pdf_history") or []):
        if isinstance(h, dict) and (h.get("period_end") or "").strip() == period_end:
            old_id = str(h.get("gridfs_id") or "").strip()
            if old_id:
                try:
                    fs.delete(ObjectId(old_id))
                except Exception:
                    pass
            break


def store_pdf_in_gridfs(username: str, pdf_path: str, period_end: str) -> Optional[str]:
    """
    Stores a PDF in GridFS and returns the gridfs id.
    """
    if not pdf_path or not os.path.exists(pdf_path):
        return None

    uname = norm_username(username)
    period_end = (period_end or "").strip()

    _delete_old_pdf_for_period(uname, period_end)

    fs = get_pdf_gridfs()
    filename = os.path.basename(pdf_path)

    with open(pdf_path, "rb") as f:
        data = f.read()

    grid_id = fs.put(
        data,
        filename=filename,
        contentType="application/pdf",
        metadata={"username": uname, "period_end": period_end, "ts": time.time()},
    )
    return str(grid_id)


def log_payroll_pdf_item(username: str, item: Dict[str, Any]) -> None:
    """
    Logs last_pdf and pdf_history on the user document.
    De-dupes by period_end.
    """
    uname = norm_username(username)
    period_end = (item.get("period_end") or "").strip()
    if not item.get("ts"):
        item["ts"] = time.time()

    users = get_users_collection()
    users.update_one({"username": uname}, {"$set": {"last_pdf": item}}, upsert=True)

    doc = users.find_one({"username": uname}, {"pdf_history": 1}) or {}
    hist = doc.get("pdf_history") or []
    if not isinstance(hist, list):
        hist = []

    new_hist = []
    replaced = False

    for old in hist:
        if isinstance(old, dict) and (old.get("period_end") or "").strip() == period_end and period_end:
            new_hist.append(item)
            replaced = True
        else:
            new_hist.append(old)

    if not replaced:
        new_hist.append(item)

    new_hist = sorted(new_hist, key=lambda x: float((x or {}).get("ts", 0) or 0), reverse=True)[:25]
    users.update_one({"username": uname}, {"$set": {"pdf_history": new_hist}}, upsert=True)


def store_and_log_pdf(username: str, pdf_path: str, period_end: str) -> Dict[str, Any]:
    """
    Store PDF in GridFS, log metadata to Mongo, and optionally delete local file.
    """
    gridfs_id = store_pdf_in_gridfs(username, pdf_path, period_end)

    item = {
        "path": pdf_path if KEEP_LOCAL_PDFS else "",
        "filename": os.path.basename(pdf_path) if pdf_path else "",
        "period_end": (period_end or "").strip(),
        "gridfs_id": gridfs_id,
        "storage": "gridfs" if gridfs_id else "local",
        "ts": time.time(),
    }

    log_payroll_pdf_item(username, item)

    if gridfs_id and not KEEP_LOCAL_PDFS:
        try:
            os.remove(pdf_path)
        except Exception:
            pass

    return item
