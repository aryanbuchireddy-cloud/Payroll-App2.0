import os
import threading
import time
import hashlib
import pandas as pd

import streamlit as st
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

# Simulate an env var for local dev in Thonny (if needed)
os.environ.setdefault(
    "PAYROLL_ENC_KEY",
    os.environ.get("PAYROLL_ENC_KEY", "Y7-Dsht3fzSZ3b9RiuxpYgIqnPefA30nNB6s84iQCoA=")
)

st.set_page_config(page_title="Payroll Portal", page_icon="💈", layout="centered")

# ---------------- Settings ----------------
SEED_DEFAULT_ADMIN = True
DEFAULT_ADMIN = {"username": "owner@example.com", "password": "changeme"}

ALLOW_SELF_SIGNUP = False  # admin-controlled accounts only
DEFAULT_ORG_ID = "default"  # kept, but NOT used for employee key UI anymore

import os
from pymongo import MongoClient

def _get_mongo_uri() -> str:
    # Try Streamlit secrets if Streamlit exists (Cloud)
    try:
        import streamlit as st  # only if available
        uri = st.secrets.get("MONGO_URI")
        if uri:
            return str(uri).strip()
    except Exception:
        pass

    # Env var
    uri = os.getenv("MONGO_URI", "").strip()
    if uri:
        return uri

    # Local fallback
    return "mongodb://localhost:27017"

MONGO_URI = _get_mongo_uri()
MONGO_DB="payrollapp"

@st.cache_resource
def _get_mongo_client_cached():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")  # fail fast if URI/auth is wrong
    return client

mongo_client = _get_mongo_client_cached()
db = mongo_client[MONGO_DB]

MONGO_USERS = "userInfo"
LOGIN_EVENTS = "loginEvents"

# ---------------- Session ----------------
ss = st.session_state
ss.setdefault("auth_user", None)
ss.setdefault("onboarding_mode", False)

ss.setdefault("payroll_thread_started", False)
ss.setdefault("payroll_thread", None)

ss.setdefault("readiness_thread_started", False)
ss.setdefault("readiness_thread", None)


# ---------------- Helpers ----------------
def _now() -> float:
    return time.time()


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
    """Turn noisy/internal errors into client-friendly messages."""
    if not err:
        return "Something went wrong. Please try again."
    msg = str(err).strip()

    # If a traceback got embedded in a string, keep only the first line by default
    first_line = msg.splitlines()[0].strip()
    low = msg.lower()

    # Common cases
    if "missing salondata credentials" in low:
        return "SalonData is not connected yet. Please complete Setup (SalonData username + password)."
    if "missing heartland credentials" in low:
        return "Heartland is not connected yet. Please complete Setup (Heartland username + password)."

    if "salondata" in low and ("invalid" in low or "incorrect" in low) and "password" in low:
        return "SalonData login failed (incorrect password). Please update your SalonData password and try again."
    if "heartland" in low and ("invalid" in low or "incorrect" in low) and "password" in low:
        return "Heartland login failed (incorrect password). Please update your Heartland password and try again."

    if "mfa" in low and ("code" in low or "otp" in low):
        return "Waiting for Heartland MFA. Enter the code below and click Submit MFA."

    if "employee" in low and "report" in low:
        return "Could not download the Heartland Employee ID report. Please contact admin to confirm the report selection."

    if "pdf" in low and ("not found" in low or "missing" in low or "load" in low):
        return "A required PDF/report could not be loaded. Please try again or contact admin."

    # Default: keep it short
    return first_line[:220]


def _safe_check_payroll_ready(username: str, *, dry_run: bool) -> dict:
    """Call backend safely so Streamlit doesn't show a huge red traceback."""
    try:
        out = check_payroll_ready_for_user(username, dry_run=dry_run) or {}
        if out.get("error"):
            out["error"] = _friendly_error(out.get("error"))
        return out
    except Exception as e:
        return {
            "ready": False,
            "missing_keys": [],
            "csv_path": None,
            "needs_sync": None,
            "error": _friendly_error(str(e)),
        }


@st.cache_resource
def get_mongo_client():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000, uuidRepresentation="standard")
    try:
        client.admin.command("ping")
    except ConnectionFailure as e:
        raise RuntimeError(f"MongoDB not reachable at {MONGO_URI}. Start MongoDB locally.") from e
    return client



# Backwards-compatible alias used by some GridFS helpers
def _get_mongo_client():
    return get_mongo_client()

# Alias for older code paths
_MONGO_DBNAME = MONGO_DB

def _get_collection(client, name: str):
    db = client[MONGO_DB]
    col = db[name]
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
        {
            "$setOnInsert": {
                "username": u,
                "created_at": _now(),
                "last_login_at": None,
                "profile_completed": False,
                "enabled": True,
                "role": "user",
                "org_id": DEFAULT_ORG_ID,
                "integrations": {"salondata": {}, "heartland": {}},
                "must_change_password": False,
            }
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )


def _mongo_set_password_if_empty(col, username: str, password: str):
    u = _norm_username(username)
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
    """Force-set portal password and clear must_change_password."""
    u = _norm_username(username)
    col.update_one(
        {"username": u},
        {"$set": {
            "password_hash": _hash_pw(new_password),
            "must_change_password": False,
            "password_changed_at": _now(),
        }},
        upsert=False,
    )


def _is_disabled(user_doc: dict | None) -> bool:
    if not user_doc:
        return True
    if user_doc.get("deleted_at"):
        return True
    return not bool(user_doc.get("enabled", True))


def _is_admin(user_doc: dict | None) -> bool:
    if not user_doc:
        return False
    return str(user_doc.get("role") or "").lower().strip() == "admin"


def _mongo_admin_create_user(users_col, username: str, *, role: str = "user", enabled: bool = True, temp_password: str | None = None):
    """
    Admin creates user.
    If temp_password provided => must_change_password=True so user resets after login during setup.
    """
    u = _norm_username(username)
    if not u:
        raise ValueError("username is required")

    # Create if missing
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

    # If user exists, apply enable/role and (optionally) temp password / must_change_password
    sets = {"enabled": bool(enabled), "role": (role or "user").lower().strip()}
    if temp_password:
        sets.update({
            "password_hash": _hash_pw(temp_password),
            "must_change_password": True,
            "temp_password_set_at": _now(),
        })
    users_col.update_one({"username": u}, {"$set": sets}, upsert=False)


def _mongo_admin_set_enabled(users_col, username: str, enabled: bool, reason: str | None = None):
    u = _norm_username(username)
    users_col.update_one(
        {"username": u},
        {"$set": {
            "enabled": bool(enabled),
            "disabled_reason": (reason or "").strip() if not enabled else None,
            "disabled_at": _now() if not enabled else None,
            "enabled_at": _now() if enabled else None,
        }},
    )


def _mongo_admin_soft_delete(users_col, username: str):
    u = _norm_username(username)
    users_col.update_one(
        {"username": u},
        {"$set": {"enabled": False, "deleted_at": _now()}},
    )


def _mongo_admin_reset_portal_password(users_col, username: str):
    u = _norm_username(username)
    users_col.update_one({"username": u}, {"$unset": {"password_hash": ""}})


def _mongo_seed_admin_if_needed(col):
    if not SEED_DEFAULT_ADMIN:
        return
    if col.estimated_document_count() == 0:
        try:
            col.insert_one({
                "username": _norm_username(DEFAULT_ADMIN["username"]),
                "password_hash": _hash_pw(DEFAULT_ADMIN["password"]),
                "created_at": _now(),
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
    doc = {
        "username": _norm_username(username),
        "password_hash_provided": _hash_pw(password) if password else None,
        "success": bool(success),
        "ts": _now(),
        "app": "Payroll Portal",
    }
    try:
        login_events_col.insert_one(doc)
    except Exception:
        pass


def _has_integration_creds(u: dict, key: str) -> bool:
    integ = (u or {}).get("integrations", {}).get(key, {}) or {}
    return bool((integ.get("username") or "").strip()) and bool((integ.get("password_enc") or "").strip())


def _save_creds_once(col, username: str, provider: str, u: str, p: str) -> bool:
    uname = _norm_username(username)
    filter_doc = {
        "username": uname,
        "$or": [
            {f"integrations.{provider}.username": {"$in": [None, ""]}},
            {f"integrations.{provider}.password_enc": {"$in": [None, ""]}},
        ],
    }
    enc_pw = encrypt_str(p or "")
    update_doc = {
        "$set": {
            f"integrations.{provider}.username": (u or "").strip(),
            f"integrations.{provider}.password_enc": enc_pw,
            f"integrations.{provider}.saved_at": _now(),
        }
    }
    res = col.update_one(filter_doc, update_doc)
    return res.modified_count == 1


def _update_integration_password(col, username: str, provider: str, new_password: str) -> bool:
    """Update ONLY external password (does not touch external username)."""
    uname = _norm_username(username)
    new_password = (new_password or "").strip()
    if not new_password:
        return False

    existing = col.find_one({"username": uname}, {f"integrations.{provider}.username": 1, "_id": 0}) or {}
    integ = (existing.get("integrations") or {}).get(provider) or {}
    if not (integ.get("username") or "").strip():
        return False

    enc_pw = encrypt_str(new_password)
    res = col.update_one(
        {"username": uname},
        {"$set": {
            f"integrations.{provider}.password_enc": enc_pw,
            f"integrations.{provider}.password_updated_at": _now(),
        }},
    )
    return res.modified_count == 1


# ---------------- Readiness state ----------------
def _set_readiness_status(
    users_col,
    username: str,
    state: str,
    *,
    error: str | None = None,
    missing_keys: list[str] | None = None,
    csv_path: str | None = None,
    needs_sync: bool | None = None,
):
    users_col.update_one(
        {"username": _norm_username(username)},
        {"$set": {
            "readiness_status": {
                "state": state,  # running | syncing_keys | ready | not_ready | failed
                "error": error,
                "missing_keys": missing_keys or [],
                "csv_path": csv_path,
                "needs_sync": needs_sync,
                "ts": _now(),
            }
        }},
    )


def _clear_readiness_state(users_col=None, username: str | None = None):
    ss.readiness_thread_started = False
    ss.readiness_thread = None
    if users_col is not None and username:
        users_col.update_one({"username": _norm_username(username)}, {"$unset": {"readiness_status": ""}})


def _start_readiness_thread(users_col, username: str):
    """
    Background worker writes ONLY to Mongo.
    Phase 1: dry_run=True
    Phase 2: if missing -> dry_run=False (Heartland sync, waits for MFA in Mongo)
    """
    users_col.update_one({"username": _norm_username(username)}, {"$unset": {"mfa_code": ""}})

    def _worker():
        try:
            _set_readiness_status(users_col, username, "running", error=None, missing_keys=[], csv_path=None, needs_sync=None)

            pre = _safe_check_payroll_ready(username, dry_run=True)

            if pre.get("ready") and not (pre.get("mis							sing_keys") or []):
                _set_readiness_status(
                    users_col, username, "ready",
                    error=None, missing_keys=[],
                    csv_path=pre.get("csv_path"),
                    needs_sync=False,
                )
                return

            missing = pre.get("missing_keys") or []
            _set_readiness_status(
                users_col, username, "syncing_keys",
                error=None, missing_keys=missing,
                csv_path=pre.get("csv_path"),
                needs_sync=True,
            )

            # Clear MFA right before the real sync starts
            users_col.update_one({"username": _norm_username(username)}, {"$unset": {"mfa_code": ""}})

            full = _safe_check_payroll_ready(username, dry_run=False)

            if full.get("ready") and not (full.get("missing_keys") or []):
                _set_readiness_status(
                    users_col, username, "ready",
                    error=None, missing_keys=[],
                    csv_path=full.get("csv_path"),
                    needs_sync=bool(full.get("needs_sync")),
                )
                return

            missing2 = full.get("missing_keys") or []
            err = full.get("error") or "Payroll is not ready."
            _set_readiness_status(
                users_col, username, "not_ready",
                error=_friendly_error(err),
                missing_keys=missing2,
                csv_path=full.get("csv_path"),
                needs_sync=True,
            )

        except Exception as e:
            _set_readiness_status(users_col, username, "failed", error=_friendly_error(str(e)), missing_keys=[], csv_path=None, needs_sync=None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    ss.readiness_thread = t
    ss.readiness_thread_started = True


# ---------------- Connect Mongo ----------------
try:
    client = get_mongo_client()
    users = _get_collection(client, MONGO_USERS)
    login_events = _get_collection(client, LOGIN_EVENTS)
    _mongo_seed_admin_if_needed(users)
except Exception as e:
    st.error(f"❌ Could not connect to MongoDB: {e}")
    st.stop()

st.markdown("<h2 style='margin-top:0'>💈 Payroll Portal</h2>", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    if ss.auth_user:
        st.write(f"Signed in as **{ss.auth_user}**")
        if st.button("Sign out", use_container_width=True, key="btn_signout"):
            _clear_readiness_state(users, ss.auth_user)
            ss.auth_user = None
            ss.onboarding_mode = False
            ss.payroll_thread_started = False
            ss.payroll_thread = None
            st.rerun()
    else:
        st.caption("Sign in below to continue.")


# ---------------- Login ----------------
if not ss.auth_user:
    with st.form("login", clear_on_submit=False):
        st.subheader("Sign in")
        username = st.text_input("Username", placeholder="you@example.com", key="login_user")
        password = st.text_input("Password (portal password)", type="password", placeholder="Temp password (if admin gave you one)", key="login_pass")
        submit = st.form_submit_button("Continue", type="primary", use_container_width=True)

    if submit:
        if not username.strip():
            st.error("Enter a username.")
            st.stop()

        user = _mongo_get_user(users, username)

        if not user:
            if not ALLOW_SELF_SIGNUP:
                _mongo_log_login_attempt(login_events, username, password, success=False)
                st.error("Account not found. Please contact the admin to create your access.")
                st.stop()
            user = _mongo_upsert_username_only(users, username)

        if _is_disabled(user):
            _mongo_log_login_attempt(login_events, username, password, success=False)
            st.error("Your access is disabled. Please contact the admin.")
            st.stop()

        if not password:
            st.error("Enter your portal password.")
            st.stop()

        ok, _ = _mongo_verify_password(users, username, password)
        if not ok:
            _mongo_log_login_attempt(login_events, username, password, success=False)
            st.error("Invalid password.")
            st.stop()

        ss.auth_user = user["username"]
        users.update_one({"username": ss.auth_user}, {"$set": {"last_login_at": _now()}})
        _clear_readiness_state(users, ss.auth_user)
        _mongo_log_login_attempt(login_events, username, password, success=True)
        st.success("Signed in.")
        st.rerun()

    st.stop()


# ---------------- Load user ----------------
user_rec = _mongo_get_user(users, ss.auth_user)
if not user_rec:
    st.error("Your account record could not be found. Please contact the admin.")
    st.stop()

if _is_disabled(user_rec):
    st.error("Your access is disabled. Please contact the admin.")
    st.stop()


# ---------------- Menu ----------------
menu = "Payroll"
with st.sidebar:
    if _is_admin(user_rec):
        menu = st.radio("Menu", ["Payroll", "Admin"], index=0, key="nav_menu")


# ---------------- Admin Page ----------------
if menu == "Admin":
    st.subheader("🔐 Admin Controls")

    # ---- Accounts list ----
    st.markdown("### All Accounts")

    docs = list(users.find({}, {"_id": 0, "password_hash": 0, "integrations": 0}))
    if docs:
        dfu = pd.DataFrame(docs)
        for col in ["created_at", "last_login_at", "disabled_at", "enabled_at", "temp_password_set_at", "password_changed_at"]:
            if col in dfu.columns:
                dfu[col] = dfu[col].apply(_fmt_ts)

        q = st.text_input("Search username", placeholder="type part of email/username to filter", key="adm_list_search")
        if q.strip() and "username" in dfu.columns:
            dfu = dfu[dfu["username"].astype(str).str.contains(q.strip().lower(), case=False, na=False)]

        show_cols = [c for c in [
            "username", "enabled", "role", "profile_completed", "must_change_password",
            "disabled_reason", "created_at", "last_login_at"
        ] if c in dfu.columns]

        st.dataframe(dfu[show_cols] if show_cols else dfu, width="stretch", hide_index=True)
    else:
        st.info("No accounts found.")

    st.divider()

    # ---- Create user ----
    st.markdown("### Create user (TEMP password)")
    with st.form("admin_create"):
        new_u = st.text_input("New username", placeholder="user@example.com", key="adm_create_user")
        new_role = st.selectbox("Role", ["user", "admin"], index=0, key="adm_create_role")
        enabled = st.checkbox("Enabled", value=True, key="adm_create_enabled")
        temp_pw = st.text_input("Temp portal password (user will change it during setup)", type="password", key="adm_create_temp_pw")
        create = st.form_submit_button("Create / Update", type="primary",  use_container_width=True)

    if create:
        if not new_u.strip():
            st.error("Enter username.")
        elif not temp_pw.strip():
            st.error("Enter a temp password.")
        else:
            _mongo_admin_create_user(users, new_u, role=new_role, enabled=enabled, temp_password=temp_pw)
            st.success("User created/updated. They will be forced to set a new password during setup.")
            st.rerun()

    st.divider()

    # ---- Enable/Disable/Delete with dropdown target ----
    st.markdown("### Enable / Disable / Delete")

    usernames_all = sorted([d.get("username", "") for d in docs if (d.get("username") or "").strip()])
    find_user = st.text_input("Search target", placeholder="type part of username/email", key="adm_target_search")
    if find_user.strip():
        filtered = [u for u in usernames_all if find_user.strip().lower() in u.lower()]
    else:
        filtered = usernames_all

    target = st.selectbox("Target account", [""] + filtered, index=0, key="adm_target_select")
    reason = st.text_input("Disable reason (optional)", placeholder="Non-payment etc.", key="adm_disable_reason")

    c1, c2 = st.columns(2)
    if c1.button("Enable", width="stretch", disabled=False, key="adm_enable_btn"):
        if not target:
            st.error("Select an account first.")
        else:
            _mongo_admin_set_enabled(users, target, True)
            st.success(f"Enabled: {target}")
            st.rerun()

    if c2.button("Disable", width="stretch", disabled=False, key="adm_disable_btn"):
        if not target:
            st.error("Select an account first.")
        else:
            _mongo_admin_set_enabled(users, target, False, reason=reason)
            st.warning(f"Disabled: {target}")
            st.rerun()

    st.markdown("### Soft delete user")
    if st.button("Soft delete", use_container_width=True, key="adm_soft_delete_btn"):
        if not target:
            st.error("Select an account first.")
        elif target == ss.auth_user:
            st.error("You cannot delete your own logged-in admin account.")
        else:
            _mongo_admin_soft_delete(users, target)
            st.success(f"Soft deleted: {target}")
            st.rerun()

    st.markdown("### Reset portal password (unset)")
    if st.button("Reset portal password",use_container_width=True, key="adm_reset_pw_btn"):
        if not target:
            st.error("Select an account first.")
        else:
            _mongo_admin_reset_portal_password(users, target)
            st.success("Portal password reset. User will set a new one on next login.")
            st.rerun()

    st.divider()
    st.info(
        "✅ Employee keys are NOT managed here.\n\n"
        "Each portal profile auto-syncs employee IDs/keys from Heartland during the readiness/sync step "
        "(first time and whenever needed). This works across any computer that logs into the same profile."
    )

    st.stop()


# ---------------- Setup ----------------
user_rec = _mongo_get_user(users, ss.auth_user) or {}
sd_missing = not _has_integration_creds(user_rec, "salondata")
hl_missing = not _has_integration_creds(user_rec, "heartland")
must_change_pw = _must_change_password(user_rec)
needs_setup = sd_missing or hl_missing or must_change_pw

if ss.onboarding_mode or not user_rec.get("profile_completed") or needs_setup:
    if must_change_pw:
        st.warning("Your account was created with a TEMP portal password. Please set your own new portal password before continuing.")

    st.info("Connect your external accounts. Saved **once** and never overwritten.")

    with st.form("setup"):
        # ---- force portal password change (temp password flow) ----
        new_pw1, new_pw2 = "", ""
        if must_change_pw:
            st.subheader("Set your new portal password")
            new_pw1 = st.text_input("New portal password", type="password", key="setup_new_pw1")
            new_pw2 = st.text_input("Confirm new portal password", type="password", key="setup_new_pw2")
            st.caption("Admin will NOT know your new password.")
            st.divider()

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("SalonData")
            if sd_missing:
                sd_user = st.text_input("SalonData username", key="setup_sd_user")
                sd_pass = st.text_input("SalonData password", type="password", key="setup_sd_pass")
            else:
                st.success("SalonData credentials already saved ✅")
                sd_user, sd_pass = "", ""

        with c2:
            st.subheader("Heartland")
            if hl_missing:
                hl_user = st.text_input("Heartland username", key="setup_hl_user")
                hl_pass = st.text_input("Heartland password", type="password", key="setup_hl_pass")
            else:
                st.success("Heartland credentials already saved ✅")
                hl_user, hl_pass = "", ""

        agree = st.checkbox("I confirm the above credentials are correct.", value=False, key="setup_agree")
        done = st.form_submit_button("Save", type="primary", use_container_width=True)

    if done:
        if must_change_pw:
            if not new_pw1.strip() or not new_pw2.strip():
                st.error("Please set your new portal password to continue.")
                st.stop()
            if new_pw1 != new_pw2:
                st.error("New portal passwords do not match.")
                st.stop()
            if len(new_pw1.strip()) < 8:
                st.error("Password should be at least 8 characters.")
                st.stop()

        if (sd_missing and (not sd_user.strip() or not sd_pass.strip())) or \
           (hl_missing and (not hl_user.strip() or not hl_pass.strip())) or \
           ((sd_missing or hl_missing) and not agree):
            st.error("Please fill required fields and confirm.")
            st.stop()

        if must_change_pw:
            _mongo_set_password_force(users, ss.auth_user, new_pw1.strip())

        wrote_any = False
        if sd_missing:
            wrote_any |= _save_creds_once(users, ss.auth_user, "salondata", sd_user, sd_pass)
        if hl_missing:
            wrote_any |= _save_creds_once(users, ss.auth_user, "heartland", hl_user, hl_pass)

        updated = _mongo_get_user(users, ss.auth_user) or {}
        sd_ok = _has_integration_creds(updated, "salondata")
        hl_ok = _has_integration_creds(updated, "heartland")
        pw_ok = not _must_change_password(updated)

        if sd_ok and hl_ok and pw_ok and not updated.get("profile_completed"):
            users.update_one({"username": ss.auth_user}, {"$set": {"profile_completed": True}})

        st.success("Saved." if wrote_any or must_change_pw else "Already on file. Nothing changed.")

        # ensure we exit onboarding mode after successful save
        ss.onboarding_mode = False
        st.rerun()

    # stop here so the main page doesn't render under Setup
    st.stop()


# ---------------- Main Payroll Page ----------------
st.success(f"Welcome back, **{ss.auth_user}**! 🎉")

user_rec = _mongo_get_user(users, ss.auth_user) or {}
st.caption(
    f"SalonData creds: {'✅ saved' if _has_integration_creds(user_rec,'salondata') else '❌ missing'} | "
    f"Heartland creds: {'✅ saved' if _has_integration_creds(user_rec,'heartland') else '❌ missing'}"
)

st.divider()
st.subheader("📄 Payroll PDFs")
st.caption("Select a PDF to download")

# Use a cached Mongo client for GridFS downloads
mongo_client = _get_mongo_client()

# Pull PDF history for this user
pdf_history = user_rec.get("pdf_history") or []
if not isinstance(pdf_history, list):
    pdf_history = []

# Build dropdown options (prefer GridFS if available)
options = []
seen = set()
for it in pdf_history:
    if not isinstance(it, dict):
        continue

    has_grid = bool(str(it.get("gridfs_id") or "").strip())
    has_path = bool(str(it.get("path") or "").strip())
    if not (has_grid or has_path):
        continue

    period = (it.get("period_end") or "").strip() or "Unknown date"
    fname = (it.get("filename") or os.path.basename(it.get("path") or "") or "Payroll.pdf").strip()
    label = f"{period} — {fname}"

    # Ensure unique labels so Streamlit selectbox behaves predictably
    if label in seen:
        label = f"{label} ({int(float(it.get('ts', 0) or 0))})"
    seen.add(label)

    options.append((label, it))

if options:
    # Streamlit warns (and may error in the future) when a widget label is empty.
    # Keep the UI clean by collapsing the label, but still provide a non-empty value.
    choice = st.selectbox(
        "PDF history",
        options,
        format_func=lambda x: x[0],
        key="pdf_selectbox",
        label_visibility="collapsed",
    )
    selected = choice[1] if choice else {}

    selected_gridfs_id = selected.get("gridfs_id")
    selected_path = selected.get("path")
    download_name = selected.get("filename") or (os.path.basename(selected_path) if selected_path else "Payroll_Report.pdf")

    # --- Lazy load: only fetch bytes when user clicks ---
    def _fetch_pdf_bytes():
        pdf_bytes = None
        source = None

        # Prefer GridFS if present
        if selected_gridfs_id:
            try:
                mongo_client = _get_mongo_client()
                fs = gridfs.GridFS(mongo_client[_MONGO_DBNAME], collection="payroll_pdfs")
                grid_out = fs.get(ObjectId(str(selected_gridfs_id)))
                pdf_bytes = grid_out.read()
                source = f"GridFS ({selected_gridfs_id})"
                return pdf_bytes, source
            except Exception as e:
                # Don't block the whole page for PDF fetch errors
                return None, f"GridFS error: {e}"

        # Fallback to local path
        if selected_path and os.path.exists(selected_path):
            try:
                with open(selected_path, "rb") as f:
                    pdf_bytes = f.read()
                source = f"Local file ({selected_path})"
                return pdf_bytes, source
            except Exception as e:
                return None, f"Local file error: {e}"

        return None, "No source available (missing GridFS id and local path)"

    # Persist loaded bytes in session so we don't re-fetch on every rerun
    sel_key = str(selected_gridfs_id or selected_path or download_name or "")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        load_pdf_clicked = st.button("Load PDF", key="load_pdf_btn", use_container_width=True)

    if load_pdf_clicked:
        with st.spinner("Loading PDF..."):
            pdf_bytes, source_msg = _fetch_pdf_bytes()
        st.session_state["__pdf_cache"] = {
            "sel_key": sel_key,
            "bytes": pdf_bytes,
            "download_name": download_name,
            "source_msg": source_msg,
        }

    cache = st.session_state.get("__pdf_cache") or {}
    cache_bytes = cache.get("bytes") if cache.get("sel_key") == sel_key else None
    cache_source = cache.get("source_msg") if cache.get("sel_key") == sel_key else None

    if cache_source:
        # This tells you if we used GridFS or local path (or why it failed)
        st.caption(f"PDF source: {cache_source}")

    if cache_bytes:
        st.download_button(
            label="Download selected PDF",
            data=cache_bytes,
            file_name=download_name,
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("Select a PDF, then click **Load PDF** to enable download.")
else:
    st.info("No PDFs found yet. Run payroll to generate one.")


# ---- External Password Update ----
with st.expander("🔑 Update external passwords (SalonData / Heartland)", expanded=False):
    st.caption("Use this when SalonData/Heartland forces a password change. Usernames cannot be edited here.")

    c_sd, c_hl = st.columns(2)

    with c_sd:
        st.markdown("### SalonData password")
        sd_pw1 = st.text_input("New SalonData password", type="password", key="pwupd_sd_1")
        sd_pw2 = st.text_input("Confirm", type="password", key="pwupd_sd_2")
        if st.button("Update SalonData password",use_container_width=True, key="pwupd_sd_btn"):
            if not sd_pw1.strip() or not sd_pw2.strip():
                st.error("Enter and confirm the new SalonData password.")
            elif sd_pw1 != sd_pw2:
                st.error("Passwords do not match.")
            else:
                ok = _update_integration_password(users, ss.auth_user, "salondata", sd_pw1)
                if ok:
                    _clear_readiness_state(users, ss.auth_user)
                    users.update_one({"username": ss.auth_user}, {"$unset": {"mfa_code": ""}})
                    st.success("✅ SalonData password updated.")
                else:
                    st.error("Could not update SalonData password (SalonData must be configured first).")

    with c_hl:
        st.markdown("### Heartland password")
        hl_pw1 = st.text_input("New Heartland password", type="password", key="pwupd_hl_1")
        hl_pw2 = st.text_input("Confirm", type="password", key="pwupd_hl_2")
        if st.button("Update Heartland password",  use_container_width=True, key="pwupd_hl_btn"):
            if not hl_pw1.strip() or not hl_pw2.strip():
                st.error("Enter and confirm the new Heartland password.")
            elif hl_pw1 != hl_pw2:
                st.error("Passwords do not match.")
            else:
                ok = _update_integration_password(users, ss.auth_user, "heartland", hl_pw1)
                if ok:
                    _clear_readiness_state(users, ss.auth_user)
                    users.update_one({"username": ss.auth_user}, {"$unset": {"mfa_code": ""}})
                    st.success("✅ Heartland password updated.")
                else:
                    st.error("Could not update Heartland password (Heartland must be configured first).")

# ---- Actions ----
latest_user = _mongo_get_user(users, ss.auth_user) or {}
readiness_status = latest_user.get("readiness_status") or {}
r_state = str(readiness_status.get("state") or "").strip().lower()

with st.container():
    st.subheader("Actions")

    readiness_running = bool(ss.readiness_thread and ss.readiness_thread.is_alive())
    payroll_running = bool(ss.payroll_thread and ss.payroll_thread.is_alive())

    # Auto-clear stuck readiness state on page load if this tab has no active readiness thread
    # (prevents showing 'running' after reopening the tab).
    if r_state in ("running", "syncing_keys") and (not readiness_running):
        try:
            _clear_readiness_state(users, ss.auth_user)
        except Exception:
            pass
        readiness_status = {}
        r_state = None


    c1, c2 = st.columns(2)
    check_clicked = c1.button("Payroll readiness check",  use_container_width=True, disabled=readiness_running, key="btn_check_ready")

    run_disabled = payroll_running
    run_clicked = c2.button("Run payroll", width="stretch", disabled=run_disabled, key="btn_run_payroll")

    if check_clicked:
        _clear_readiness_state(users, ss.auth_user)
        users.update_one({"username": ss.auth_user}, {"$unset": {"mfa_code": ""}})
        _start_readiness_thread(users, ss.auth_user)
        st.toast("Checking payroll readiness…", icon="🔍")

        # Poll Mongo briefly so READY/NOT READY can appear without an auto-rerun (prevents UI flicker/duplication).
        _ph = st.empty()
        _deadline = time.time() + 20.0
        try:
            while time.time() < _deadline:
                _u = _mongo_get_user(users, ss.auth_user) or {}
                _rs = (_u.get("readiness_status") or {})
                _st = str(_rs.get("state") or "").strip().lower()

                # Stop polling once we reach a stable state or if the worker thread ended.
                _rt = ss.get("readiness_thread", None)
                _alive = bool(_rt is not None and _rt.is_alive())
                if _st in ("ready", "not_ready", "syncing_keys", "failed") or (not _alive and _st != "running"):
                    break

                _ph.info("🔍 Checking payroll readiness…")
                time.sleep(0.35)
        finally:
            _ph.empty()

    # Refresh readiness
    latest_user = _mongo_get_user(users, ss.auth_user) or {}
    readiness_status = latest_user.get("readiness_status") or {}
    r_state = str(readiness_status.get("state") or "").strip().lower()
    r_missing = readiness_status.get("missing_keys") or []
    r_error = readiness_status.get("error")
    r_csv = readiness_status.get("csv_path")

    if r_state == "running":
        st.info("🔍 Readiness check running…")
    elif r_state == "syncing_keys":
        N=len(r_missing or [])
        st.warning(f"I’ve detected {N} new employee(s) I need to fetch details from Heartland to complete setup. "
    f"When you receive the Heartland code, enter it below and click Submit MFA.")
        if r_missing:
            st.caption("Missing before sync: " + ", ".join(r_missing))
    elif r_state == "ready":
        st.success("✅ Payroll is ready to run.(No new employees found")
        if r_csv:
            st.caption(f"Payroll CSV: {r_csv}")
    elif r_state == "not_ready":
        st.error(f"❌ {_friendly_error(r_error) or 'Payroll is not ready.'}")
        if r_missing:
            st.write("Missing keys:")
            st.write(", ".join(r_missing))
    elif r_state == "failed":
        st.error(f"❌ {_friendly_error(r_error) or 'Readiness check failed.'}")

    if run_clicked:
        # Re-check readiness right now (button states can lag during the same rerun).
        latest_user_now = _mongo_get_user(users, ss.auth_user) or {}
        rs_now = latest_user_now.get("readiness_status") or {}
        state_now = str(rs_now.get("state") or "").strip().lower()

        if readiness_running or (state_now != "ready"):
            if state_now == "running":
                st.warning("Readiness check is still running. Wait until it says **Payroll is ready**.")
            elif state_now == "syncing_keys":
                st.warning("Keys are syncing. Enter the Heartland MFA code below and click **Submit MFA**.")
            else:
                st.warning("Please click **Check payroll ready** first (and wait until it says **Payroll is ready**).")
        else:
            st.toast("Starting payroll…", icon="⏳")
            # Start immediately so the UI updates right away (no blocking checks here).
            users.update_one({"username": ss.auth_user}, {"$unset": {"mfa_code": ""}})

            # Mark running in Mongo immediately (backend will update this too).
            users.update_one(
                {"username": ss.auth_user},
                {"$set": {"payroll.state": "running", "payroll.error": None, "payroll.updated_at": _now()}},
                upsert=False,
            )

            _clear_readiness_state(users, ss.auth_user)

            def _payroll_worker(username: str):
                try:
                    run_payroll_for_user(username)
                except Exception as e:
                    print("Background payroll error:", repr(e))

            t = threading.Thread(target=_payroll_worker, args=(ss.auth_user,), daemon=True)
            t.start()
            ss.payroll_thread = t
            ss.payroll_thread_started = True

            st.success("Payroll run started. Enter the Heartland MFA code below when you get it.")
    t = ss.get("payroll_thread", None)

    # Pull the latest status from Mongo (so failures show correctly).
    latest_user2 = _mongo_get_user(users, ss.auth_user) or {}
    payroll_doc = latest_user2.get("payroll") or {}
    p_state = str(payroll_doc.get("state") or "").strip().lower()
    p_err = payroll_doc.get("error")


    


    # If Mongo says payroll is running but this tab has no live worker thread (e.g., you closed the tab
    # and reopened it), clear it immediately on load so the UI doesn't stay stuck on "running".
    has_live_thread = bool(t is not None and t.is_alive())
    if (p_state == "running") and (not has_live_thread):
        try:
            users.update_one(
                {"username": ss.auth_user},
                {"$set": {"payroll.state": "idle", "payroll.error": None, "payroll.updated_at": _now()}},
            )
        except Exception:
            pass
        p_state = "idle"
        p_err = None
        ss.payroll_thread_started = False
        ss.payroll_thread = None
        t = None

    # Status display
    if (t is not None and t.is_alive()) or (p_state == "running"):
        st.info("⏳ Payroll is currently running…")
    elif ss.payroll_thread_started and (t is None or (not t.is_alive())):
        if p_state == "failed" or (p_err and p_state != "completed"):
            st.error(f"❌ Payroll failed: {_friendly_error(p_err)}")
        elif p_state == "completed":
            st.success("✅ Payroll completed.")
        else:
            st.warning("Payroll finished, but status is unknown. Please check logs.")
        ss.payroll_thread_started = False
        ss.payroll_thread = None
    else:
        st.write("🛈 Click **Check payroll ready** first, then **Run payroll**.")

    st.markdown("### Heartland MFA")
    mfa_code_input = st.text_input("Heartland MFA code", placeholder="6-digit code", key="mfa_code_input")
    if st.button("Submit MFA",  use_container_width=True, key="btn_submit_mfa"):
        if mfa_code_input.strip():
            users.update_one({"username": ss.auth_user}, {"$set": {"mfa_code": mfa_code_input.strip()}})
            st.success("MFA submitted. Automation will continue when it sees it.")
        else:
            st.error("Please enter the MFA code first.")

    # Auto-refresh while background work runs (no sleep => much less flicker)
    rt = ss.get("readiness_thread", None)
    pt = ss.get("payroll_thread", None)

    auto_refresh_needed = (
        (rt is not None and rt.is_alive()) or
        (pt is not None and pt.is_alive())
    )

    # Pause auto-refresh while waiting for MFA input (otherwise it interrupts typing)
    if auto_refresh_needed and ss.get("readiness_running") is True:
        # if you track MFA state differently, change this condition
        pass

    if auto_refresh_needed and ss.get("readiness_running") is not True:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=2000, key="auto_refresh_status")
        except Exception:
            # fallback if streamlit_autorefresh isn't installed
            st.rerun()




