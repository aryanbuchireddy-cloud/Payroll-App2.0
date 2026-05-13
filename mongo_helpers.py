"""
mongo_helpers.py
----------------

Shared Mongo helpers for UI and backend.

This avoids the UI using one Mongo DB and the runner using another.
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any

from pymongo import MongoClient, ReturnDocument
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import gridfs

from app_config import (
    MONGO_URI,
    MONGO_DB,
    MONGO_USERS_COLL,
    MONGO_LOGIN_EVENTS_COLL,
    MONGO_KEYS_COLL,
    MONGO_PDF_GRIDFS_BUCKET,
)

_mongo_client_cache: Optional[MongoClient] = None


def norm_username(username: str) -> str:
    return (username or "").strip().lower()


def now_ts() -> float:
    return time.time()


def get_mongo_client() -> MongoClient:
    global _mongo_client_cache
    if _mongo_client_cache is None:
        _mongo_client_cache = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=3000,
            uuidRepresentation="standard",
            maxPoolSize=10,
        )
        _mongo_client_cache.admin.command("ping")
    return _mongo_client_cache


def get_db():
    return get_mongo_client()[MONGO_DB]


def get_users_collection():
    col = get_db()[MONGO_USERS_COLL]
    try:
        col.create_index("username", unique=True)
    except Exception:
        pass
    return col


def get_login_events_collection():
    return get_db()[MONGO_LOGIN_EVENTS_COLL]


def get_keys_collection():
    col = get_db()[MONGO_KEYS_COLL]
    try:
        col.create_index("username", unique=True)
    except Exception:
        pass
    return col


def get_pdf_gridfs():
    return gridfs.GridFS(get_db(), collection=MONGO_PDF_GRIDFS_BUCKET)


def get_user_doc(username: str, projection: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    return get_users_collection().find_one({"username": norm_username(username)}, projection or {"_id": 0}) or {}


def set_user_fields(username: str, fields: Dict[str, Any], *, upsert: bool = True) -> None:
    get_users_collection().update_one(
        {"username": norm_username(username)},
        {"$set": fields},
        upsert=upsert,
    )


def unset_user_fields(username: str, fields: list[str]) -> None:
    get_users_collection().update_one(
        {"username": norm_username(username)},
        {"$unset": {k: "" for k in fields}},
    )
