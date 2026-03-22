import os
import threading
import time
import hashlib
import pandas as pd

import streamlit as st
from cryptography.fernet import Fernet
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None
from pymongo import MongoClient, ReturnDocument
import gridfs
from bson import ObjectId
from pymongo.errors import DuplicateKeyError, ConnectionFailure

from crypto_utils import encrypt_str
from payrollrunner_dbkeys import run_payroll_for_user, check_payroll_ready_for_user
import subprocess, sys


# ── Playwright ────────────────────────────────────────────────────────────────
def ensure_chromium():
    if os.environ.get("PLAYWRIGHT_BROWSERS_INSTALLED") == "1":
        return
    try:
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        os.environ["PLAYWRIGHT_BROWSERS_INSTALLED"] = "1"
    except Exception as e:
        raise RuntimeError(f"Playwright browser install failed at runtime: {e}")

ensure_chromium()


# ── Encryption ────────────────────────────────────────────────────────────────
def _get_payroll_enc_key() -> str:
    try:
        k = st.secrets.get("PAYROLL_ENC_KEY", None)
        if k:
            return str(k).strip()
    except Exception:
        pass
    k = os.getenv("PAYROLL_ENC_KEY", "").strip()
    if k:
        return k
    raise RuntimeError(
        "PAYROLL_ENC_KEY is missing. Add it to Streamlit secrets or set it as an environment variable."
    )

PAYROLL_ENC_KEY = _get_payroll_enc_key()

try:
    Fernet(PAYROLL_ENC_KEY.encode("utf-8"))
except Exception:
    raise RuntimeError(
        "PAYROLL_ENC_KEY is invalid. Must be a valid Fernet base64 key "
        "(generate one with cryptography.fernet.Fernet.generate_key())."
    )

_fernet = Fernet(PAYROLL_ENC_KEY.encode("utf-8"))

def encrypt_str(s: str) -> str:
    return _fernet.encrypt((s or "").encode("utf-8")).decode("utf-8")

def decrypt_str(token: str) -> str:
    return _fernet.decrypt((token or "").encode("utf-8")).decode("utf-8")


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Payroll Portal", page_icon="💈", layout="centered")


# ── Settings ──────────────────────────────────────────────────────────────────
SEED_DEFAULT_ADMIN = True
DEFAULT_ADMIN      = {"username": "owner@example.com", "password": "changeme"}
ALLOW_SELF_SIGNUP  = False
DEFAULT_ORG_ID     = "default"


# ── MongoDB ───────────────────────────────────────────────────────────────────
def _get_mongo_uri() -> str:
    try:
        uri = st.secrets.get("MONGO_URI")
        if uri:
            return str(uri).strip()
    except Exception:
        pass
    uri = os.getenv("MONGO_URI", "").strip()
    return uri or "mongodb://localhost:27017"

MONGO_URI  = _get_mongo_uri()
MONGO_DB   = "payrollapp"
MONGO_USERS   = "userInfo"
LOGIN_EVENTS  = "loginEvents"

@st.cache_resource
def _get_mongo_client_cached():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    return client

@st.cache_resource
def get_mongo_client():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000, uuidRepresentation="standard")
    try:
        client.admin.command("ping")
    except ConnectionFailure as e:
        raise RuntimeError(f"MongoDB not reachable at {MONGO_URI}. Start MongoDB locally.") from e
    return client

def _get_mongo_client():
    return get_mongo_client()

_MONGO_DBNAME = MONGO_DB

mongo_client = _get_mongo_client_cached()
db = mongo_client[MONGO_DB]


# ── Session state ─────────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("auth_user",               None)
ss.setdefault("onboarding_mode",         False)
ss.setdefault("payroll_thread_started",  False)
ss.setdefault("payroll_thread",          None)
ss.setdefault("readiness_thread_started", False)
ss.setdefault("readiness_thread",        None)
ss.setdefault("last_login_ts",           0)     # guards against component replaying last value


# ── Date helpers ──────────────────────────────────────────────────────────────
from datetime import date, datetime, timedelta

def _is_friday(d: date) -> bool:
    return isinstance(d, date) and d.weekday() == 4

def _prev_friday(from_day: date) -> date:
    offset = (from_day.weekday() - 4) % 7
    return from_day - timedelta(days=offset)

def _default_payroll_friday(today: date | None = None) -> date:
    return _prev_friday(today or date.today())

def _now() -> float:
    return time.time()


# ── PDF name helpers ──────────────────────────────────────────────────────────
def _validation_display_name(period_end: str) -> str:
    """
    Converts any stored period_end string into the display label:
      '03/06/2026' or '03-06-2026' or '2026-03-06'  →  'Payroll Validation File-03/06/2026'
    """
    s = (period_end or "").strip()
    # Normalise separators to slashes
    s = s.replace("-", "/")
    parts = s.split("/")
    # Handle YYYY/MM/DD → MM/DD/YYYY
    if len(parts) == 3 and len(parts[0]) == 4:
        s = f"{parts[1]}/{parts[2]}/{parts[0]}"
    return f"Payroll Validation File-{s}" if s else "Payroll Validation File"


def _validation_download_name(period_end: str) -> str:
    """
    Filesystem-safe download filename:
      '03/06/2026' → 'Payroll Validation File-03-06-2026.pdf'
    """
    s = (period_end or "").strip().replace("/", "-")
    parts = s.split("-")
    # YYYY-MM-DD → MM-DD-YYYY
    if len(parts) == 3 and len(parts[0]) == 4:
        s = f"{parts[1]}-{parts[2]}-{parts[0]}"
    return f"Payroll Validation File-{s}.pdf" if s else "Payroll Validation File.pdf"


# ── Misc helpers ──────────────────────────────────────────────────────────────
def _norm_username(u: str) -> str:
    return (u or "").strip().lower()

def _hash_pw(pw: str) -> str:
    return hashlib.sha256((pw or "").encode("utf-8")).hexdigest()

def _fmt_ts(x):
    try:
        if x is None or x == "":
            return ""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(x)))
    except Exception:
        return ""

def _friendly_error(err: str | None) -> str:
    if not err:
        return "Something went wrong. Please try again."
    msg        = str(err).strip()
    first_line = msg.splitlines()[0].strip()
    low        = msg.lower()

    if "missing salondata credentials" in low or "missing salondata" in low:
        return "SalonData is not connected. Complete Setup with your SalonData username and password."
    if "missing heartland credentials" in low or "missing heartland" in low:
        return "Heartland is not connected. Complete Setup with your Heartland username and password."
    if "salondata" in low and ("invalid" in low or "incorrect" in low) and "password" in low:
        return "SalonData login failed — incorrect password. Update your SalonData password and try again."
    if "heartland" in low and ("invalid" in low or "incorrect" in low) and "password" in low:
        return "Heartland login failed — incorrect password. Update your Heartland password and try again."
    if "mfa" in low and ("code" in low or "otp" in low or "required" in low):
        return "Waiting for Heartland MFA. Enter the code below and click Submit MFA."
    if "mfa" in low and "timeout" in low:
        return "MFA timed out — no code was entered in time. Please try again."
    if "employee" in low and "report" in low:
        return "Could not download the Heartland Employee ID report. Contact admin to confirm the report selection."
    if "pdf" in low and ("not found" in low or "missing" in low or "load" in low):
        return "A required PDF/report could not be loaded. Try again or contact admin."
    if "timeout" in low or "timed out" in low:
        return "The request timed out — the site may be slow or your password may have changed. Please try again."
    if "network" in low or "connection" in low or "unreachable" in low:
        return "A network error occurred. Check your internet connection and try again."
    if "checkstatus" in low or "reportdata" in low or "viewpdf" in low:
        return "Could not retrieve the Heartland employee report. Please try again."
    if "missing keys" in low or "missing heartland keys" in low:
        return "Some employees are missing Heartland keys. Run Check Payroll Readiness to sync them."
    if "payroll is not ready" in low:
        return "Payroll is not ready. Run Check Payroll Readiness first."
    if "file successfully imported" in low:
        return "Payroll successfully uploaded to Heartland."
    if "time card import failed" in low:
        return "Heartland time card import failed. Please try again or contact admin."
    if "formatting returned none" in low:
        return "Could not format the payroll file. Please try again or contact admin."

    # Fallback: never show a raw Python traceback or exception class to the user
    # Strip common technical prefixes so the message is readable
    clean = first_line
    for prefix in ("RuntimeError:", "ValueError:", "Exception:", "Error:", "TimeoutError:", "PlaywrightTimeoutError:"):
        if clean.lower().startswith(prefix.lower()):
            clean = clean[len(prefix):].strip()
    # If what's left still looks like code (has colons, brackets, etc.), replace entirely
    if any(c in clean for c in ("Traceback", "File \"", "line ", "  at ", "assert ")):
        return "An unexpected error occurred. Please try again or contact admin."
    return clean[:220] if clean else "Something went wrong. Please try again."

def _safe_check_payroll_ready(username: str, *, dry_run: bool) -> dict:
    try:
        out = check_payroll_ready_for_user(username, dry_run=dry_run) or {}
        if out.get("error"):
            out["error"] = _friendly_error(out.get("error"))
        return out
    except Exception as e:
        return {"ready": False, "missing_keys": [], "csv_path": None, "needs_sync": None, "error": _friendly_error(str(e))}


# ── Mongo helpers ─────────────────────────────────────────────────────────────
def _get_collection(client, name: str):
    col = client[MONGO_DB][name]
    if name == MONGO_USERS:
        try:
            col.create_index("username", unique=True)
        except Exception:
            pass
    return col

def _mongo_get_user(col, username: str):
    return col.find_one({"username": _norm_username(username)})

def _mongo_upsert_username_only(col, username: str):
    u = _norm_username(username)
    return col.find_one_and_update(
        {"username": u},
        {"$setOnInsert": {
            "username": u,
            "created_at": _now(),
            "last_login_at": None,
            "profile_completed": False,
            "enabled": True,
            "role": "user",
            "org_id": DEFAULT_ORG_ID,
            "integrations": {"salondata": {}, "heartland": {}},
            "must_change_password": False,
        }},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )

def _mongo_set_password_if_empty(col, username: str, password: str):
    u   = _norm_username(username)
    res = col.update_one(
        {"username": u, "password_hash": {"$exists": False}},
        {"$set": {"password_hash": _hash_pw(password)}},
    )
    return res.modified_count == 1

def _mongo_verify_password(col, username: str, password: str):
    user = _mongo_get_user(col, username)
    if not user or "password_hash" not in user:
        return False, user
    return (user["password_hash"] == _hash_pw(password), user)

def _must_change_password(user_doc: dict | None) -> bool:
    return bool((user_doc or {}).get("must_change_password", False))

def _mongo_set_password_force(col, username: str, new_password: str):
    col.update_one(
        {"username": _norm_username(username)},
        {"$set": {
            "password_hash": _hash_pw(new_password),
            "must_change_password": False,
            "password_changed_at": _now(),
        }},
    )

def _is_disabled(user_doc: dict | None) -> bool:
    if not user_doc:
        return True
    if user_doc.get("deleted_at"):
        return True
    return not bool(user_doc.get("enabled", True))

def _is_admin(user_doc: dict | None) -> bool:
    return str((user_doc or {}).get("role") or "").lower().strip() == "admin"

def _mongo_admin_create_user(users_col, username: str, *, role: str = "user", enabled: bool = True, temp_password: str | None = None):
    u = _norm_username(username)
    if not u:
        raise ValueError("username is required")
    doc = {
        "username": u,
        "created_at": _now(),
        "last_login_at": None,
        "profile_completed": False,
        "enabled": bool(enabled),
        "role": (role or "user").lower().strip(),
        "org_id": DEFAULT_ORG_ID,
        "integrations": {"salondata": {}, "heartland": {}},
        "must_change_password": bool(temp_password),
        "temp_password_set_at": _now() if temp_password else None,
    }
    if temp_password:
        doc["password_hash"] = _hash_pw(temp_password)
    users_col.update_one({"username": u}, {"$setOnInsert": doc}, upsert=True)
    sets = {"enabled": bool(enabled), "role": (role or "user").lower().strip()}
    if temp_password:
        sets.update({"password_hash": _hash_pw(temp_password), "must_change_password": True, "temp_password_set_at": _now()})
    users_col.update_one({"username": u}, {"$set": sets})

def _mongo_admin_set_enabled(users_col, username: str, enabled: bool, reason: str | None = None):
    users_col.update_one(
        {"username": _norm_username(username)},
        {"$set": {
            "enabled": bool(enabled),
            "disabled_reason": (reason or "").strip() if not enabled else None,
            "disabled_at":  _now() if not enabled else None,
            "enabled_at":   _now() if enabled else None,
        }},
    )

def _mongo_admin_soft_delete(users_col, username: str):
    users_col.update_one(
        {"username": _norm_username(username)},
        {"$set": {"enabled": False, "deleted_at": _now()}},
    )

def _mongo_admin_reset_portal_password(users_col, username: str):
    users_col.update_one({"username": _norm_username(username)}, {"$unset": {"password_hash": ""}})

def _mongo_seed_admin_if_needed(col):
    if not SEED_DEFAULT_ADMIN:
        return
    if col.estimated_document_count() == 0:
        try:
            col.insert_one({
                "username":      _norm_username(DEFAULT_ADMIN["username"]),
                "password_hash": _hash_pw(DEFAULT_ADMIN["password"]),
                "created_at":    _now(),
                "last_login_at": None,
                "profile_completed": False,
                "enabled": True,
                "role": "admin",
                "org_id": DEFAULT_ORG_ID,
                "integrations": {"salondata": {}, "heartland": {}},
                "must_change_password": False,
            })
        except DuplicateKeyError:
            pass

def _mongo_log_login_attempt(login_events_col, username: str, password: str | None, success: bool):
    try:
        login_events_col.insert_one({
            "username":               _norm_username(username),
            "password_hash_provided": _hash_pw(password) if password else None,
            "success": bool(success),
            "ts":      _now(),
            "app":     "Payroll Portal",
        })
    except Exception:
        pass

def _has_integration_creds(u: dict, key: str) -> bool:
    integ = (u or {}).get("integrations", {}).get(key, {}) or {}
    return bool((integ.get("username") or "").strip()) and bool((integ.get("password_enc") or "").strip())

def _save_creds_once(col, username: str, provider: str, u: str, p: str) -> bool:
    uname = _norm_username(username)
    res   = col.update_one(
        {"username": uname, "$or": [
            {f"integrations.{provider}.username":     {"$in": [None, ""]}},
            {f"integrations.{provider}.password_enc": {"$in": [None, ""]}},
        ]},
        {"$set": {
            f"integrations.{provider}.username":     (u or "").strip(),
            f"integrations.{provider}.password_enc": encrypt_str(p or ""),
            f"integrations.{provider}.saved_at":     _now(),
        }},
    )
    return res.modified_count == 1

def _update_integration_password(col, username: str, provider: str, new_password: str) -> bool:
    uname = _norm_username(username)
    new_password = (new_password or "").strip()
    if not new_password:
        return False
    existing = col.find_one({"username": uname}, {f"integrations.{provider}.username": 1, "_id": 0}) or {}
    integ    = (existing.get("integrations") or {}).get(provider) or {}
    if not (integ.get("username") or "").strip():
        return False
    res = col.update_one(
        {"username": uname},
        {"$set": {
            f"integrations.{provider}.password_enc":        encrypt_str(new_password),
            f"integrations.{provider}.password_updated_at": _now(),
        }},
    )
    return res.modified_count == 1


# ── Readiness state ───────────────────────────────────────────────────────────
def _set_readiness_status(users_col, username: str, state: str, *, error=None, missing_keys=None, csv_path=None, needs_sync=None):
    users_col.update_one(
        {"username": _norm_username(username)},
        {"$set": {"readiness_status": {
            "state":        state,
            "error":        error,
            "missing_keys": missing_keys or [],
            "csv_path":     csv_path,
            "needs_sync":   needs_sync,
            "ts":           _now(),
        }}},
    )

def _clear_readiness_state(users_col=None, username: str | None = None):
    ss.readiness_thread_started = False
    ss.readiness_thread         = None
    if users_col is not None and username:
        users_col.update_one({"username": _norm_username(username)}, {"$unset": {"readiness_status": ""}})

def _start_readiness_thread(users_col, username: str, period_end_date):
    """
    Phase 1: dry_run=True (fast, no browser)
    Phase 2: if keys missing → dry_run=False (Heartland sync, waits for MFA in Mongo)
    """
    users_col.update_one({"username": _norm_username(username)}, {"$unset": {"mfa_code": ""}})

    def _worker():
        try:
            _set_readiness_status(users_col, username, "running")

            pre = check_payroll_ready_for_user(username, dry_run=True, period_end_date=period_end_date)

            if pre.get("ready") and not (pre.get("missing_keys") or []):
                _set_readiness_status(users_col, username, "ready",
                                      csv_path=pre.get("csv_path"), needs_sync=False)
                return

            missing = pre.get("missing_keys") or []
            _set_readiness_status(users_col, username, "syncing_keys",
                                  missing_keys=missing, csv_path=pre.get("csv_path"), needs_sync=True)

            users_col.update_one({"username": _norm_username(username)}, {"$unset": {"mfa_code": ""}})

            full = check_payroll_ready_for_user(username, dry_run=False, period_end_date=period_end_date)

            if full.get("ready") and not (full.get("missing_keys") or []):
                _set_readiness_status(users_col, username, "ready",
                                      csv_path=full.get("csv_path"), needs_sync=bool(full.get("needs_sync")))
                return

            missing2 = full.get("missing_keys") or []
            _set_readiness_status(users_col, username, "not_ready",
                                  error=_friendly_error(full.get("error") or "Payroll is not ready."),
                                  missing_keys=missing2, csv_path=full.get("csv_path"), needs_sync=True)

        except Exception as e:
            _set_readiness_status(users_col, username, "failed", error=_friendly_error(str(e)))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    ss.readiness_thread         = t
    ss.readiness_thread_started = True


# ── Connect Mongo ─────────────────────────────────────────────────────────────
try:
    client       = get_mongo_client()
    users        = _get_collection(client, MONGO_USERS)
    login_events = _get_collection(client, LOGIN_EVENTS)
    _mongo_seed_admin_if_needed(users)
except Exception as e:
    st.error(f"❌ Could not connect to MongoDB: {e}")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("<h2 style='margin-top:0'>💈 Payroll Portal</h2>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if ss.auth_user:
        st.write(f"Signed in as **{ss.auth_user}**")
        if st.button("Sign out", use_container_width=True, key="btn_signout"):
            _clear_readiness_state(users, ss.auth_user)
            ss.auth_user              = None
            ss.onboarding_mode        = False
            ss.payroll_thread_started = False
            ss.payroll_thread         = None
            st.rerun()
    else:
        st.caption("Sign in to continue.")


# ═════════════════════════════════════════════════════════════════════════════
#  LOGIN  — native HTML form so Chrome shows credential picker on username click
# ═════════════════════════════════════════════════════════════════════════════
if not ss.auth_user:
    import os as _os, tempfile as _tempfile
    import streamlit.components.v1 as _cv1

    # A real <form> parsed at page-load is the only way Chrome shows the
    # credential picker on the username field (matching Auris Payroll behaviour).
    # st.components.v1.declare_component renders a full iframe with its own DOM
    # so Chrome treats it as a proper login form.
    _COMP_DIR = _os.path.join(_tempfile.gettempdir(), "payroll_login_comp_v3")
    _os.makedirs(_COMP_DIR, exist_ok=True)
    _COMP_HTML_PATH = _os.path.join(_COMP_DIR, "index.html")

    _COMP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  /* ── Light mode (Streamlit default) ── */
  html, body {
    background: #ffffff;
    color: #31333f;
  }
  label {
    display: block;
    font-size: 0.875rem;
    color: #31333f;
    margin: 16px 0 6px;
    font-weight: 400;
  }
  input {
    width: 100%;
    padding: 8px 12px;
    background: #ffffff;
    border: 1px solid #d0d3da;
    border-radius: 6px;
    color: #31333f;
    font-size: 0.95rem;
    outline: none;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  input::placeholder { color: #adb5bd; }

  /* ── Dark mode ── */
  @media (prefers-color-scheme: dark) {
    html, body { background: #0e1117; color: #fafafa; }
    label { color: #fafafa; }
    input {
      background: transparent;
      border-color: rgba(250,250,250,0.2);
      color: #fafafa;
    }
    input::placeholder { color: rgba(250,250,250,0.35); }
  }

  body {
    font-family: "Source Sans Pro", -apple-system, sans-serif;
    padding: 4px 2px 8px;
  }
  label:first-of-type { margin-top: 0; }
  input:focus {
    border-color: #ff4b4b;
    box-shadow: 0 0 0 2px rgba(255,75,75,0.25);
  }
  .err {
    color: #ff4b4b;
    font-size: 0.82rem;
    margin-top: 10px;
    padding: 6px 10px;
    background: rgba(255,75,75,0.12);
    border-radius: 4px;
    display: none;
  }
  .err.visible { display: block; }
  button {
    margin-top: 18px;
    width: 100%;
    padding: 10px;
    background: #ff4b4b;
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
    letter-spacing: 0.01em;
  }
  button:hover:not(:disabled) { background: #e03c3c; }
  button:disabled { opacity: 0.55; cursor: default; }
</style>
</head>
<body>
<form id="f" method="post" action="#">
  <label for="u">Username</label>
  <input id="u" name="username" type="email"
         autocomplete="username"
         placeholder="you@example.com" required autofocus />
  <label for="p">Password</label>
  <input id="p" name="password" type="password"
         autocomplete="current-password"
         placeholder="Enter your portal password" required />
  <div class="err" id="err"></div>
  <button type="submit" id="btn">Continue</button>
</form>
<script>
  function post(msg) {
    window.parent.postMessage(Object.assign({isStreamlitMessage: true}, msg), "*");
  }

  function resize() {
    post({type: "streamlit:setFrameHeight", height: document.body.scrollHeight + 4});
  }

  post({type: "streamlit:componentReady", apiVersion: 1});
  resize();

  document.getElementById("f").addEventListener("submit", function(e) {
    e.preventDefault();
    var u = document.getElementById("u").value.trim();
    var p = document.getElementById("p").value;
    var errEl = document.getElementById("err");
    if (!u || !p) {
      errEl.textContent = "Please fill in both fields.";
      errEl.className = "err visible";
      resize();
      return;
    }
    errEl.className = "err";
    document.getElementById("btn").disabled = true;
    document.getElementById("btn").textContent = "Signing in…";
    post({type: "streamlit:setComponentValue", value: {username: u, password: p, ts: Date.now()}});
  });

  window.addEventListener("message", function(ev) {
    if (!ev.data || ev.data.type !== "streamlit:render") return;
    var args = ev.data.args || {};
    var errEl = document.getElementById("err");
    if (args.error) {
      errEl.textContent = args.error;
      errEl.className = "err visible";
      document.getElementById("btn").disabled = false;
      document.getElementById("btn").textContent = "Continue";
      resize();
    } else {
      errEl.className = "err";
    }
  });
</script>
</body>
</html>"""

    with open(_COMP_HTML_PATH, "w") as _f:
        _f.write(_COMP_HTML)

    _login_component = _cv1.declare_component("payroll_login_v3", path=_COMP_DIR)

    st.subheader("Sign in")
    ss.setdefault("login_error", "")
    result = _login_component(error=ss.login_error, key="login_comp", height=240)

    # The component persists its last submitted value across every rerun, so we
    # guard with a timestamp: only process a submission we haven't seen before.
    _result_ts = int((result or {}).get("ts") or 0)
    if result and result.get("username") and _result_ts > ss.last_login_ts:
        ss.last_login_ts = _result_ts   # mark as handled immediately — prevents replay loop

        username = result["username"].strip()
        password = result.get("password") or ""
        ss.login_error = ""

        user = _mongo_get_user(users, username)

        if not user:
            if not ALLOW_SELF_SIGNUP:
                _mongo_log_login_attempt(login_events, username, password, success=False)
                ss.login_error = "Account not found. Contact your admin to request access."
                st.rerun()
            user = _mongo_upsert_username_only(users, username)

        if _is_disabled(user):
            _mongo_log_login_attempt(login_events, username, password, success=False)
            ss.login_error = "Your account is disabled. Contact your admin."
            st.rerun()

        if not password:
            ss.login_error = "Please enter your password."
            st.rerun()

        ok, _ = _mongo_verify_password(users, username, password)
        if not ok:
            _mongo_log_login_attempt(login_events, username, password, success=False)
            ss.login_error = "Incorrect password."
            st.rerun()

        ss.auth_user = user["username"]
        ss.login_error = ""
        users.update_one({"username": ss.auth_user}, {"$set": {"last_login_at": _now()}})
        _clear_readiness_state(users, ss.auth_user)
        _mongo_log_login_attempt(login_events, username, password, success=True)
        st.rerun()

    st.stop()

# ── Load user record ──────────────────────────────────────────────────────────
user_rec = _mongo_get_user(users, ss.auth_user)
if not user_rec:
    st.error("Your account record could not be found. Please contact your admin.")
    st.stop()

if _is_disabled(user_rec):
    st.error("Your account is disabled. Please contact your admin.")
    st.stop()


# ── Navigation ────────────────────────────────────────────────────────────────
menu = "Payroll"
with st.sidebar:
    if _is_admin(user_rec):
        menu = st.radio("Menu", ["Payroll", "Admin"], index=0, key="nav_menu")


# ═════════════════════════════════════════════════════════════════════════════
#  ADMIN PAGE
# ═════════════════════════════════════════════════════════════════════════════
if menu == "Admin":
    st.subheader("🔐 Admin")

    # ── Account list ──────────────────────────────────────────────────────────
    st.markdown("### Accounts")
    docs = list(users.find({}, {"_id": 0, "password_hash": 0, "integrations": 0}))
    if docs:
        dfu = pd.DataFrame(docs)
        ts_cols = ["created_at", "last_login_at", "disabled_at", "enabled_at", "temp_password_set_at", "password_changed_at"]
        for c in ts_cols:
            if c in dfu.columns:
                dfu[c] = dfu[c].apply(_fmt_ts)

        q = st.text_input("Filter by username", placeholder="type to filter…", key="adm_list_search")
        if q.strip() and "username" in dfu.columns:
            dfu = dfu[dfu["username"].astype(str).str.contains(q.strip().lower(), case=False, na=False)]

        show_cols = [c for c in [
            "username", "enabled", "role", "profile_completed", "must_change_password",
            "disabled_reason", "created_at", "last_login_at",
        ] if c in dfu.columns]
        st.dataframe(dfu[show_cols] if show_cols else dfu, use_container_width=True, hide_index=True)
    else:
        st.info("No accounts found.")

    st.divider()

    # ── Create user ───────────────────────────────────────────────────────────
    st.markdown("### Create user")
    with st.form("admin_create"):
        new_u    = st.text_input("Username",          placeholder="user@example.com", key="adm_create_user")
        new_role = st.selectbox("Role", ["user", "admin"], index=0,                   key="adm_create_role")
        enabled  = st.checkbox("Enabled", value=True,                                 key="adm_create_enabled")
        temp_pw  = st.text_input("Temporary password (user must change on first login)",
                                 type="password",                                     key="adm_create_temp_pw")
        create   = st.form_submit_button("Create / Update", type="primary", use_container_width=True)

    if create:
        if not new_u.strip():
            st.error("Username is required.")
        elif not temp_pw.strip():
            st.error("A temporary password is required.")
        else:
            _mongo_admin_create_user(users, new_u, role=new_role, enabled=enabled, temp_password=temp_pw)
            st.success("User created. They'll be prompted to set a new password on first login.")
            st.rerun()

    st.divider()

    # ── Enable / Disable / Delete ─────────────────────────────────────────────
    st.markdown("### Manage account")
    usernames_all = sorted([d.get("username", "") for d in docs if (d.get("username") or "").strip()])
    find_user = st.text_input("Search", placeholder="type part of username…", key="adm_target_search")
    filtered  = [u for u in usernames_all if find_user.strip().lower() in u.lower()] if find_user.strip() else usernames_all

    target = st.selectbox("Select account", [""] + filtered, index=0, key="adm_target_select")
    reason = st.text_input("Disable reason (optional)", placeholder="e.g. Non-payment", key="adm_disable_reason")

    c1, c2 = st.columns(2)
    if c1.button("Enable",  use_container_width=True, key="adm_enable_btn"):
        if not target:
            st.error("Select an account first.")
        else:
            _mongo_admin_set_enabled(users, target, True)
            st.success(f"Enabled: {target}")
            st.rerun()

    if c2.button("Disable", use_container_width=True, key="adm_disable_btn"):
        if not target:
            st.error("Select an account first.")
        else:
            _mongo_admin_set_enabled(users, target, False, reason=reason)
            st.warning(f"Disabled: {target}")
            st.rerun()

    c3, c4 = st.columns(2)
    if c3.button("Soft delete", use_container_width=True, key="adm_soft_delete_btn"):
        if not target:
            st.error("Select an account first.")
        elif target == ss.auth_user:
            st.error("You cannot delete your own admin account.")
        else:
            _mongo_admin_soft_delete(users, target)
            st.success(f"Deleted: {target}")
            st.rerun()

    if c4.button("Reset portal password", use_container_width=True, key="adm_reset_pw_btn"):
        if not target:
            st.error("Select an account first.")
        else:
            _mongo_admin_reset_portal_password(users, target)
            st.success("Portal password cleared. User will set a new one on next login.")
            st.rerun()

    st.divider()
    st.info(
        "Employee keys are synced automatically from Heartland during the readiness check — "
        "no manual management needed."
    )
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  SETUP / ONBOARDING
# ═════════════════════════════════════════════════════════════════════════════
user_rec     = _mongo_get_user(users, ss.auth_user) or {}
sd_missing   = not _has_integration_creds(user_rec, "salondata")
hl_missing   = not _has_integration_creds(user_rec, "heartland")
must_change_pw = _must_change_password(user_rec)
needs_setup  = sd_missing or hl_missing or must_change_pw

if ss.onboarding_mode or not user_rec.get("profile_completed") or needs_setup:
    if must_change_pw:
        st.warning("⚠️ Your account was set up with a temporary password. Please choose a new password before continuing.")
    else:
        st.info("Connect your external accounts below. Credentials are saved once and never overwritten.")

    with st.form("setup"):
        new_pw1 = new_pw2 = ""
        if must_change_pw:
            st.subheader("Set new portal password")
            new_pw1 = st.text_input("New password",     type="password", key="setup_new_pw1")
            new_pw2 = st.text_input("Confirm password", type="password", key="setup_new_pw2")
            st.caption("Your admin will not be able to see this password.")
            st.divider()

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("SalonData")
            if sd_missing:
                sd_user = st.text_input("Username", key="setup_sd_user")
                sd_pass = st.text_input("Password", type="password", key="setup_sd_pass")
            else:
                st.success("Connected ✅")
                sd_user = sd_pass = ""

        with c2:
            st.subheader("Heartland")
            if hl_missing:
                hl_user = st.text_input("Username", key="setup_hl_user")
                hl_pass = st.text_input("Password", type="password", key="setup_hl_pass")
            else:
                st.success("Connected ✅")
                hl_user = hl_pass = ""

        agree = st.checkbox("I confirm the credentials above are correct.", value=False, key="setup_agree")
        done  = st.form_submit_button("Save & Continue", type="primary", use_container_width=True)

    if done:
        if must_change_pw:
            if not new_pw1.strip() or not new_pw2.strip():
                st.error("Please enter and confirm your new portal password.")
                st.stop()
            if new_pw1 != new_pw2:
                st.error("Passwords do not match.")
                st.stop()
            if len(new_pw1.strip()) < 8:
                st.error("Password must be at least 8 characters.")
                st.stop()

        creds_incomplete = (
            (sd_missing and (not sd_user.strip() or not sd_pass.strip())) or
            (hl_missing and (not hl_user.strip() or not hl_pass.strip()))
        )
        if creds_incomplete or ((sd_missing or hl_missing) and not agree):
            st.error("Please fill in all required fields and check the confirmation box.")
            st.stop()

        if must_change_pw:
            _mongo_set_password_force(users, ss.auth_user, new_pw1.strip())
        if sd_missing:
            _save_creds_once(users, ss.auth_user, "salondata", sd_user, sd_pass)
        if hl_missing:
            _save_creds_once(users, ss.auth_user, "heartland", hl_user, hl_pass)

        updated = _mongo_get_user(users, ss.auth_user) or {}
        if _has_integration_creds(updated, "salondata") and _has_integration_creds(updated, "heartland") \
                and not _must_change_password(updated) and not updated.get("profile_completed"):
            users.update_one({"username": ss.auth_user}, {"$set": {"profile_completed": True}})

        ss.onboarding_mode = False
        st.success("Setup complete!")
        st.rerun()

    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PAYROLL PAGE
# ═════════════════════════════════════════════════════════════════════════════
user_rec = _mongo_get_user(users, ss.auth_user) or {}

st.success(f"Welcome back, **{ss.auth_user}**! 🎉")
st.caption(
    f"SalonData: {'✅ connected' if _has_integration_creds(user_rec, 'salondata') else '❌ not connected'}  "
    f"·  Heartland: {'✅ connected' if _has_integration_creds(user_rec, 'heartland') else '❌ not connected'}"
)

st.divider()

# ── Extra session state flags ─────────────────────────────────────────────────
ss.setdefault("mfa_active",    False)
ss.setdefault("mfa_submitted", False)
ss.setdefault("payroll_done",  False)
ss.setdefault("mfa_thank_you", False)
ss.setdefault("notify_msg",    ("info", "Click **Check Payroll Readiness** first, then **Execute Payroll**.", ""))


# ═════════════════════════════════════════════════════════════════════════════
#  1 ▸ PERIOD END DATE
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("📅 Period End Date")

default_friday        = _default_payroll_friday(date.today())
selected_payroll_date = st.date_input(
    "Select a Friday",
    value=default_friday,
    key="selected_payroll_date",
)
is_valid_friday = _is_friday(selected_payroll_date)

if not is_valid_friday:
    st.warning("⚠️ Please select a Friday to enable payroll actions.")

st.divider()


# ═════════════════════════════════════════════════════════════════════════════
#  READ LIVE STATE FROM MONGO (single read, used everywhere below)
# ═════════════════════════════════════════════════════════════════════════════
latest_user      = _mongo_get_user(users, ss.auth_user) or {}
readiness_status = latest_user.get("readiness_status") or {}
r_state          = str(readiness_status.get("state") or "").strip().lower()
r_missing        = readiness_status.get("missing_keys") or []
r_error          = readiness_status.get("error")
r_csv            = readiness_status.get("csv_path")

payroll_doc = latest_user.get("payroll") or {}
p_state     = str(payroll_doc.get("state") or "").strip().lower()
p_err       = payroll_doc.get("error")

readiness_thread = ss.get("readiness_thread", None)
payroll_thread   = ss.get("payroll_thread",   None)
readiness_running = bool(readiness_thread and readiness_thread.is_alive())
payroll_running   = bool(payroll_thread   and payroll_thread.is_alive())

# ── Auto-clear stale "running" readiness when no live thread owns it ──────────
if r_state in ("running", "syncing_keys") and not readiness_running:
    try:
        _clear_readiness_state(users, ss.auth_user)
    except Exception:
        pass
    r_state = ""

# ── Detect payroll thread finishing this render cycle ─────────────────────────
payroll_just_finished = (
    ss.payroll_thread_started
    and not payroll_running
    and not payroll_thread  # thread object cleaned up below
)

# Clear stale payroll "running" in Mongo when thread is gone
if p_state == "running" and not payroll_running:
    try:
        users.update_one(
            {"username": ss.auth_user},
            {"$set": {"payroll.state": "idle", "payroll.error": None, "payroll.updated_at": _now()}},
        )
    except Exception:
        pass
    p_state = "idle"

# When the payroll thread finishes, lock the Execute button and reset MFA state
if ss.payroll_thread_started and not payroll_running:
    ss.payroll_thread_started = False
    ss.payroll_thread         = None
    ss.mfa_active             = False
    ss.mfa_submitted          = False
    ss.payroll_done           = True   # locks Execute Payroll for this session


# ═════════════════════════════════════════════════════════════════════════════
#  2 ▸ ACTIONS — Check Readiness | Execute Payroll
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("▶️ Actions")

# Button disable rules:
#   Check Readiness  → disabled only while readiness check is actively running
#   Execute Payroll  → disabled if: not ready, payroll running, OR already done this session
is_ready        = (r_state == "ready")
check_disabled  = (readiness_running or not is_valid_friday)
run_disabled    = (
    not is_ready
    or payroll_running
    or ss.payroll_done
    or not is_valid_friday
    or (readiness_running and r_state == "syncing_keys")
)

c1, c2 = st.columns(2)
check_clicked = c1.button(
    "🔍 Check Payroll Readiness",
    use_container_width=True,
    disabled=check_disabled,
    key="btn_check_ready",
)
run_clicked = c2.button(
    "▶️ Execute Payroll",
    use_container_width=True,
    disabled=run_disabled,
    key="btn_run_payroll",
)

# ── Notification: render immediately from session state (no blank-frame blink) ─
def _notify(kind: str, msg: str, caption: str = ""):
    """Store message — rendered at top of NEXT render cycle with no blank gap."""
    ss.notify_msg = (kind, msg, caption)

def _render_notify():
    _nm = ss.get("notify_msg") or ("info", "Click **Check Payroll Readiness** first, then **Execute Payroll**.", "")
    kind, msg, caption = _nm
    if kind == "info":
        st.info(msg)
    elif kind == "success":
        st.success(msg)
    elif kind == "warning":
        st.warning(msg)
    elif kind == "error":
        st.error(msg)
    if caption:
        st.caption(caption)

_render_notify()

# ── Handle Check Readiness click ──────────────────────────────────────────────
if check_clicked:
    ss.payroll_done  = False   # allow Execute again after a fresh readiness check
    ss.mfa_active    = False
    ss.mfa_submitted = False
    ss.mfa_thank_you = False
    _clear_readiness_state(users, ss.auth_user)
    users.update_one({"username": ss.auth_user}, {"$unset": {"mfa_code": ""}})
    _start_readiness_thread(users, ss.auth_user, selected_payroll_date)
    # Do NOT block or loop here — let the auto-refresh below pick up state changes

# ── Handle Execute Payroll click ──────────────────────────────────────────────
if run_clicked:
    users.update_one({"username": ss.auth_user}, {"$unset": {"mfa_code": ""}})
    users.update_one(
        {"username": ss.auth_user},
        {"$set": {"payroll.state": "running", "payroll.error": None, "payroll.updated_at": _now()}},
    )
    _clear_readiness_state(users, ss.auth_user)
    ss.mfa_active    = True
    ss.mfa_submitted = False

    def _payroll_worker(username: str, _period: date):
        try:
            run_payroll_for_user(username, period_end_date=_period)
        except Exception as e:
            print("Background payroll error:", repr(e))

    t = threading.Thread(target=_payroll_worker, args=(ss.auth_user, selected_payroll_date), daemon=True)
    t.start()
    ss.payroll_thread         = t
    ss.payroll_thread_started = True
    payroll_running           = True   # reflect immediately this render

# ── Render the single status message ─────────────────────────────────────────
# Re-read payroll state for accurate display after any click above.
# The backend can optionally write payroll.state = "awaiting_mfa" just before
# calling _wait_for_mfa_code(), and readiness_status.state = "awaiting_mfa"
# just before waiting during a key sync — this unlocks the specific "Please
# enter your MFA code now" message.  All other states degrade gracefully.
_latest  = _mongo_get_user(users, ss.auth_user) or {}
_pd      = _latest.get("payroll") or {}
_ps      = str(_pd.get("state") or "").strip().lower()
_pe      = _pd.get("error")
_rlatest = _latest.get("readiness_status") or {}
_rs_sub  = str(_rlatest.get("substate") or "").strip().lower()   # "awaiting_mfa" if backend writes it

# ── Payroll execution messages ────────────────────────────────────────────────
if payroll_running or _ps == "running":
    if ss.mfa_thank_you:
        # Flash "Thank you" for exactly one render cycle, then transition to finishing
        _notify("success", "✅ Thank you — MFA received. Finishing up…")
        ss.mfa_thank_you = False
    elif ss.mfa_submitted:
        _notify("info", "⚙️ Finishing execution — uploading to Heartland…")
    elif _ps == "awaiting_mfa":
        _notify("warning", "🔐 Please enter your MFA code now")
    else:
        _notify("info", "⏳ Executing payroll…")

# ── Payroll completed / failed ────────────────────────────────────────────────
elif ss.payroll_done:
    if _ps == "failed" or (_pe and _ps != "completed"):
        _notify("error", f"❌ Payroll failed: {_friendly_error(_pe)}")
    else:
        _notify("success", "✅ Payroll successfully executed.")

# ── Readiness check messages ──────────────────────────────────────────────────
elif r_state == "running":
    _notify("info", "🔍 Checking payroll readiness…")

elif r_state == "syncing_keys":
    n = len(r_missing or [])
    names_caption = ("New employees: " + ", ".join(r_missing)) if r_missing else ""
    if ss.mfa_thank_you:
        _notify("success", "✅ Thank you — MFA received. Finishing sync…",
                caption=names_caption)
        ss.mfa_thank_you = False
    elif ss.mfa_submitted:
        _notify("info", "⚙️ Finishing sync — importing keys from Heartland…",
                caption=names_caption)
    elif _rs_sub == "awaiting_mfa":
        _notify("warning", "🔐 Please enter your MFA code now",
                caption=names_caption)
    else:
        _notify(
            "warning",
            f"Found **{n}** new employee{'s' if n != 1 else ''} — heading to Heartland to fetch IDs…",
            caption=names_caption,
        )

# ── Readiness final states ────────────────────────────────────────────────────
elif r_state == "ready":
    _notify("success", "✅ Your payroll is ready to run.",
            caption=f"Payroll CSV: {r_csv}" if r_csv else "")
elif r_state == "not_ready":
    _notify("error", f"❌ {_friendly_error(r_error) or 'Payroll is not ready.'}",
            caption=("Missing keys: " + ", ".join(r_missing)) if r_missing else "")
elif r_state == "failed":
    _notify("error", f"❌ {_friendly_error(r_error) or 'Readiness check failed.'}")
else:
    _notify("info", "Click **Check Payroll Readiness** first, then **Execute Payroll**.")

st.divider()


# ═════════════════════════════════════════════════════════════════════════════
#  3 ▸ PAYROLL VALIDATION FILE
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("📄 Payroll Validation File")

mongo_client_pdf = _get_mongo_client()
pdf_history      = user_rec.get("pdf_history") or []
if not isinstance(pdf_history, list):
    pdf_history = []

options = []
seen    = set()
for it in pdf_history:
    if not isinstance(it, dict):
        continue
    if not (it.get("gridfs_id") or it.get("path")):
        continue
    period     = (it.get("period_end") or "").strip() or "Unknown date"
    disp_label = _validation_display_name(period)
    label      = disp_label
    if label in seen:
        label = f"{label} ({int(float(it.get('ts', 0) or 0))})"
    seen.add(label)
    options.append((label, it))

if options:
    choice   = st.selectbox("Select file", options, format_func=lambda x: x[0], key="pdf_selectbox")
    selected = choice[1] if choice else {}

    selected_gridfs_id = selected.get("gridfs_id")
    selected_path      = selected.get("path")
    period_end_raw     = (selected.get("period_end") or "").strip()
    download_name      = _validation_download_name(period_end_raw)
    sel_key            = str(selected_gridfs_id or selected_path or download_name or "")

    cache = st.session_state.get("__pdf_cache") or {}
    if cache.get("sel_key") != sel_key:
        with st.spinner("Loading file…"):
            pdf_bytes  = None
            source_msg = ""
            if selected_gridfs_id:
                try:
                    fs       = gridfs.GridFS(mongo_client_pdf[_MONGO_DBNAME], collection="payroll_pdfs")
                    grid_out = fs.get(ObjectId(str(selected_gridfs_id)))
                    pdf_bytes  = grid_out.read()
                    source_msg = f"GridFS ({selected_gridfs_id})"
                except Exception as e:
                    source_msg = f"GridFS error: {e}"
            elif selected_path and os.path.exists(selected_path):
                try:
                    with open(selected_path, "rb") as f:
                        pdf_bytes = f.read()
                    source_msg = f"Local file ({selected_path})"
                except Exception as e:
                    source_msg = f"File error: {e}"
            else:
                source_msg = "No source available."

        st.session_state["__pdf_cache"] = {
            "sel_key": sel_key, "bytes": pdf_bytes,
            "download_name": download_name, "source_msg": source_msg,
        }
        cache = st.session_state["__pdf_cache"]

    pdf_bytes = cache.get("bytes")
    if pdf_bytes:
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=download_name,
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.error(f"Could not load file. {cache.get('source_msg', '')}")
else:
    st.info("No Payroll Validation File yet. Run payroll above to generate one.")

st.divider()


# ═════════════════════════════════════════════════════════════════════════════
#  4 ▸ HEARTLAND MFA
#  Only active after Execute Payroll is clicked; locked after Submit MFA
#  until the payroll run ends.
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("🔐 Heartland MFA")

mfa_input_disabled  = not (ss.mfa_active or (r_state == "syncing_keys" and readiness_running))
mfa_submit_disabled = not (ss.mfa_active or (r_state == "syncing_keys" and readiness_running)) or ss.mfa_submitted

mfa_col1, mfa_col2 = st.columns([3, 1])
with mfa_col1:
    mfa_code_input = st.text_input(
        "MFA code", placeholder="Enter 6-digit code when prompted",
        label_visibility="collapsed", key="mfa_code_input",
        disabled=mfa_input_disabled,
    )
with mfa_col2:
    if st.button(
        "Submit MFA",
        use_container_width=True,
        key="btn_submit_mfa",
        disabled=mfa_submit_disabled,
    ):
        if mfa_code_input.strip():
            users.update_one({"username": ss.auth_user}, {"$set": {"mfa_code": mfa_code_input.strip()}})
            ss.mfa_submitted = True
            ss.mfa_thank_you = True
            ss["mfa_code_input"] = ""   # clear the input field
            st.rerun()
        else:
            st.error("Please enter the MFA code.")

st.divider()


# ═════════════════════════════════════════════════════════════════════════════
#  5 ▸ UPDATE PASSWORDS
# ═════════════════════════════════════════════════════════════════════════════
with st.expander("🔑 Update passwords", expanded=False):

    st.markdown("**Portal password**")
    st.caption("This is the password you use to log in to this portal.")
    pp_col1, pp_col2 = st.columns(2)
    with pp_col1:
        portal_cur = st.text_input("Current password",  type="password", key="pwupd_portal_cur")
        portal_pw1 = st.text_input("New password",      type="password", key="pwupd_portal_1")
        portal_pw2 = st.text_input("Confirm password",  type="password", key="pwupd_portal_2")
    with pp_col2:
        st.write("")
        st.write("")
        st.write("")
        if st.button("Update portal password", use_container_width=True, key="pwupd_portal_btn"):
            if not portal_cur.strip() or not portal_pw1.strip() or not portal_pw2.strip():
                st.error("Please fill in all three fields.")
            elif portal_pw1 != portal_pw2:
                st.error("New passwords do not match.")
            elif len(portal_pw1.strip()) < 8:
                st.error("New password must be at least 8 characters.")
            else:
                ok, _ = _mongo_verify_password(users, ss.auth_user, portal_cur)
                if not ok:
                    st.error("Current password is incorrect.")
                else:
                    _mongo_set_password_force(users, ss.auth_user, portal_pw1.strip())
                    st.success("✅ Portal password updated.")

    st.divider()

    st.caption("Use the fields below when SalonData or Heartland forces a password change. Usernames cannot be changed here.")
    c_sd, c_hl = st.columns(2)

    with c_sd:
        st.markdown("**SalonData**")
        sd_pw1 = st.text_input("New password",     type="password", key="pwupd_sd_1")
        sd_pw2 = st.text_input("Confirm password", type="password", key="pwupd_sd_2")
        if st.button("Update SalonData password", use_container_width=True, key="pwupd_sd_btn"):
            if not sd_pw1.strip() or not sd_pw2.strip():
                st.error("Enter and confirm the new password.")
            elif sd_pw1 != sd_pw2:
                st.error("Passwords do not match.")
            else:
                if _update_integration_password(users, ss.auth_user, "salondata", sd_pw1):
                    _clear_readiness_state(users, ss.auth_user)
                    users.update_one({"username": ss.auth_user}, {"$unset": {"mfa_code": ""}})
                    st.success("✅ SalonData password updated.")
                else:
                    st.error("Update failed — SalonData must be configured first.")

    with c_hl:
        st.markdown("**Heartland**")
        hl_pw1 = st.text_input("New password",     type="password", key="pwupd_hl_1")
        hl_pw2 = st.text_input("Confirm password", type="password", key="pwupd_hl_2")
        if st.button("Update Heartland password", use_container_width=True, key="pwupd_hl_btn"):
            if not hl_pw1.strip() or not hl_pw2.strip():
                st.error("Enter and confirm the new password.")
            elif hl_pw1 != hl_pw2:
                st.error("Passwords do not match.")
            else:
                if _update_integration_password(users, ss.auth_user, "heartland", hl_pw1):
                    _clear_readiness_state(users, ss.auth_user)
                    users.update_one({"username": ss.auth_user}, {"$unset": {"mfa_code": ""}})
                    st.success("✅ Heartland password updated.")
                else:
                    st.error("Update failed — Heartland must be configured first.")


# ═════════════════════════════════════════════════════════════════════════════
#  AUTO-REFRESH (background polling)
# ═════════════════════════════════════════════════════════════════════════════
_rt = ss.get("readiness_thread", None)
_pt = ss.get("payroll_thread",   None)
_bg_running = (
    (_rt is not None and _rt.is_alive())
    or (_pt is not None and _pt.is_alive())
)

_thread_just_died = ss.get("_last_bg_running", False) and not _bg_running
ss["_last_bg_running"] = _bg_running

if _thread_just_died:
    st.rerun()
elif _bg_running:
    if st_autorefresh is not None:
        # Render inside a zero-height container so the component
        # occupies no visible space and can't bleed into adjacent widgets
        _ar_slot = st.empty()
        with _ar_slot:
            st_autorefresh(interval=2500, limit=None, key="bg_autorefresh")
    else:
        time.sleep(2)
        st.rerun()
