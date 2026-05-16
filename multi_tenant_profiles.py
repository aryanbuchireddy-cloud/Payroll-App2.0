"""
multi_tenant_profiles.py
------------------------

Purpose:
    Adds a multi-client / multi-tenant profile layer for ONE Streamlit app.

Use case:
    One Streamlit deployment supports many portal users. Each user can have
    different Heartland selection screens, parser profile names, report picker
    preferences, and readiness rules without creating separate apps.

This file intentionally does NOT change:
    - SalonData CSV parser logic
    - Heartland CSV formatting logic
    - payroll math
    - employee-key matching logic
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    from playwright.async_api import Page
except Exception:
    Page = Any

from mongo_helpers import norm_username, get_users_collection


DEFAULT_TENANT_PROFILE: Dict[str, Any] = {
    "parser_profile": "standard",
    "heartland": {
        "profile_pick": {"match": "", "index": 0, "required": False},
        "client_pick": {"match": "", "index": 0, "required": False},
        "employeeid_report_pick": {"match": "Employee id", "index": 0, "required": False},
        "continue_button_text": "Continue",
        "max_selection_rounds": 5,
        "wait_after_click_ms": 1200,
    },
}


BUILTIN_USER_PROFILE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "quopayroll@gmail.com": {
        "parser_profile": "geoff",
        "heartland": {
            "profile_pick": {"match": "Partner User", "index": 1, "required": False},
            "client_pick": {"match": "Great Clips", "index": 0, "required": False},
            "employeeid_report_pick": {"match": "Employee id", "index": 1, "required": False},
            "continue_button_text": "Continue",
            "max_selection_rounds": 5,
            "wait_after_click_ms": 1200,
        },
    },
    "owner@example.com": {
        "parser_profile": "standard",
        "heartland": {
            "profile_pick": {"match": "Partner User", "index": 1, "required": False},
            "client_pick": {"match": "Great Clips", "index": 0, "required": False},
            "employeeid_report_pick": {"match": "Employee id", "index": 1, "required": False},
            "continue_button_text": "Continue",
            "max_selection_rounds": 5,
            "wait_after_click_ms": 1200,
        },
    },
}


def deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _clean_pick(pick: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pick = pick or {}
    return {
        "match": str(pick.get("match") or "").strip(),
        "index": int(pick.get("index") or 0),
        "required": bool(pick.get("required", False)),
    }


@dataclass
class TenantProfile:
    username: str
    parser_profile: str = "standard"
    heartland: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def profile_pick(self) -> Dict[str, Any]:
        return _clean_pick((self.heartland or {}).get("profile_pick"))

    @property
    def client_pick(self) -> Dict[str, Any]:
        return _clean_pick((self.heartland or {}).get("client_pick"))

    @property
    def employeeid_report_pick(self) -> Dict[str, Any]:
        return _clean_pick((self.heartland or {}).get("employeeid_report_pick"))

    @property
    def continue_button_text(self) -> str:
        return str((self.heartland or {}).get("continue_button_text") or "Continue").strip()

    @property
    def max_selection_rounds(self) -> int:
        try:
            return max(1, int((self.heartland or {}).get("max_selection_rounds") or 5))
        except Exception:
            return 5

    @property
    def wait_after_click_ms(self) -> int:
        try:
            return max(250, int((self.heartland or {}).get("wait_after_click_ms") or 1200))
        except Exception:
            return 1200


def profile_from_dict(username: str, data: Optional[Dict[str, Any]]) -> TenantProfile:
    merged = deep_merge(DEFAULT_TENANT_PROFILE, data or {})
    heartland = merged.get("heartland") if isinstance(merged.get("heartland"), dict) else {}
    return TenantProfile(
        username=norm_username(username),
        parser_profile=str(merged.get("parser_profile") or "standard").strip() or "standard",
        heartland=heartland,
        raw=merged,
    )


def get_tenant_profile(username: str, users_col=None) -> TenantProfile:
    uname = norm_username(username)
    if users_col is None:
        users_col = get_users_collection()

    mongo_profile: Dict[str, Any] = {}
    try:
        doc = users_col.find_one({"username": uname}, {"tenant_profile": 1, "_id": 0}) or {}
        tp = doc.get("tenant_profile") or {}
        if isinstance(tp, dict):
            mongo_profile = tp
    except Exception:
        mongo_profile = {}

    builtin = BUILTIN_USER_PROFILE_OVERRIDES.get(uname, {})
    merged = deep_merge(builtin, mongo_profile)
    return profile_from_dict(uname, merged)


def upsert_tenant_profile(
    users_col,
    username: str,
    *,
    parser_profile: str = "standard",
    profile_match: str = "",
    profile_index: int = 0,
    client_match: str = "",
    client_index: int = 0,
    employeeid_match: str = "Employee id",
    employeeid_index: int = 0,
    profile_required: bool = False,
    client_required: bool = False,
) -> None:
    uname = norm_username(username)
    profile = {
        "parser_profile": parser_profile or "standard",
        "heartland": {
            "profile_pick": {
                "match": profile_match or "",
                "index": int(profile_index or 0),
                "required": bool(profile_required),
            },
            "client_pick": {
                "match": client_match or "",
                "index": int(client_index or 0),
                "required": bool(client_required),
            },
            "employeeid_report_pick": {
                "match": employeeid_match or "Employee id",
                "index": int(employeeid_index or 0),
                "required": False,
            },
            "continue_button_text": "Continue",
            "max_selection_rounds": 5,
            "wait_after_click_ms": 1200,
        },
        "updated_at": time.time(),
    }

    users_col.update_one(
        {"username": uname},
        {"$set": {"tenant_profile": profile}},
        upsert=True,
    )


def get_parser_profile_name(username: str, users_col=None) -> str:
    return get_tenant_profile(username, users_col).parser_profile


def get_heartland_pick(username: str, kind: str, users_col=None) -> Dict[str, Any]:
    p = get_tenant_profile(username, users_col)
    k = (kind or "").strip().lower()

    if k in ("profile", "multiaccount", "account", "profile_pick"):
        return p.profile_pick
    if k in ("client", "multiclient", "company", "client_pick"):
        return p.client_pick
    if k in ("employeeid", "employeeid_report", "employee_report"):
        return p.employeeid_report_pick

    raise ValueError(f"Unknown Heartland pick kind: {kind}")


async def _visible_text(page: Page) -> str:
    try:
        txt = await page.locator("body").inner_text(timeout=3000)
        return re.sub(r"\s+", " ", txt or " ").strip()
    except Exception:
        return ""


def _short_text(s: str, limit: int = 900) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s[:limit] + ("..." if len(s) > limit else "")


async def _debug_heartland_page(page: Page, label: str) -> None:
    try:
        url = page.url or ""
    except Exception:
        url = "[url unavailable]"
    try:
        title = await page.title()
    except Exception:
        title = "[title unavailable]"
    try:
        body = await _visible_text(page)
    except Exception:
        body = "[body unavailable]"

    print("")
    print("=" * 80)
    print(f"🧪 HEARTLAND DEBUG: {label}")
    print(f"URL: {url}")
    print(f"TITLE: {title}")
    print(f"BODY: {_short_text(body)}")
    print("=" * 80)
    print("")


async def _click_continue(page: Page, text: str = "Continue") -> bool:
    candidates = [
        f'button:has-text("{text}")',
        f'text="{text}"',
        'button:has-text("Continue")',
        'input[type="submit"]',
    ]

    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0:
                await loc.wait_for(state="visible", timeout=1500)
                await loc.click()
                return True
        except Exception:
            continue

    return False


async def _click_nth_text_match(page: Page, match: str, index: int = 0) -> bool:
    match = (match or "").strip()
    if not match:
        return False

    idx = max(0, int(index or 0))

    try:
        loc = page.get_by_text(match, exact=False)
        count = await loc.count()
        if count > idx:
            target = loc.nth(idx)
            await target.wait_for(state="visible", timeout=2500)
            await target.scroll_into_view_if_needed()
            await target.click()
            return True
    except Exception:
        pass

    broad_selectors = [
        f'div:has-text("{match}")',
        f'li:has-text("{match}")',
        f'tr:has-text("{match}")',
        f'button:has-text("{match}")',
        f'a:has-text("{match}")',
        f'[role="button"]:has-text("{match}")',
    ]

    for sel in broad_selectors:
        try:
            loc = page.locator(sel)
            count = await loc.count()
            if count > idx:
                target = loc.nth(idx)
                await target.wait_for(state="visible", timeout=2000)
                await target.scroll_into_view_if_needed()
                await target.click()
                return True
        except Exception:
            continue

    return False


async def _smart_pick_with_optional_handyman(
    page: Page,
    *,
    pick: Dict[str, Any],
    task_name: str,
    handyman=None,
) -> bool:
    match = str((pick or {}).get("match") or "").strip()
    index = int((pick or {}).get("index") or 0)
    required = bool((pick or {}).get("required", False))

    if not match:
        return not required

    clicked = await _click_nth_text_match(page, match, index)
    if clicked:
        return True

    if handyman is not None:
        try:
            clicked = await handyman.smart_click(
                page,
                task=f"On Heartland, select {task_name}: text similar to '{match}', option index {index}.",
                max_steps=8,
            )
            if clicked:
                return True
        except Exception:
            pass

    if required:
        raise RuntimeError(f"Could not select required Heartland {task_name}: match='{match}', index={index}")

    return False


async def _screen_seems_like_selection(page: Page) -> bool:
    txt = (await _visible_text(page)).lower()
    if _screen_seems_like_mfa(txt):
        return False
    clues = ["continue", "select", "profile", "client", "company", "partner user", "great clips"]
    return any(c in txt for c in clues)


def _screen_seems_like_mfa(body_text: str) -> bool:
    low = (body_text or "").lower()
    return (
        "verify mfa" in low
        or "verification code" in low
        or "secondary factor" in low
        or "authenticator app" in low
        or "verifying" in low
    )


async def handle_heartland_post_login_selection_flow(
    page: Page,
    username: str,
    *,
    users_col=None,
    profile: Optional[TenantProfile] = None,
    handyman=None,
) -> Dict[str, Any]:
    """
    Handles Heartland post-login screens:
        - no selection screen
        - profile/multi-account only
        - client/multi-client only
        - both profile + client
        - screen reset/glitch after Continue
    """
    tenant = profile or get_tenant_profile(username, users_col)
    events = []

    async def log_event(action: str, detail: str = ""):
        events.append({"action": action, "detail": detail, "ts": time.time()})
        print(f"🏢 Heartland tenant flow | {action}: {detail}")

    await log_event("start", f"user={tenant.username}, parser_profile={tenant.parser_profile}")
    print(f"🧪 tenant.username={tenant.username}")
    print(f"🧪 tenant.parser_profile={tenant.parser_profile}")
    print(f"🧪 tenant.profile_pick={tenant.profile_pick}")
    print(f"🧪 tenant.client_pick={tenant.client_pick}")
    print(f"🧪 tenant.employeeid_report_pick={tenant.employeeid_report_pick}")
    await _debug_heartland_page(page, "after login before tenant selection")

    try:
        await page.wait_for_load_state("domcontentloaded", timeout=8000)
    except Exception:
        pass

    for round_num in range(1, tenant.max_selection_rounds + 1):
        body_text = await _visible_text(page)
        body_low = body_text.lower()
        await log_event("round", f"{round_num}/{tenant.max_selection_rounds}")
        print(f"🧪 round={round_num}, url={page.url}")
        print(f"🧪 body_preview={_short_text(body_text, 600)}")

        if _screen_seems_like_mfa(body_text):
            await _debug_heartland_page(page, "Heartland MFA unresolved before tenant selection")
            await log_event("failed", "Heartland MFA screen is still visible after code submission.")
            return {
                "ok": False,
                "reason": "heartland_mfa_unresolved",
                "events": events,
                "last_screen_text": body_text[:1000],
                "profile": tenant.raw,
            }

        did_something = False

        profile_pick = tenant.profile_pick
        if profile_pick.get("match"):
            if profile_pick["match"].lower() in body_low or "profile" in body_low or "partner" in body_low:
                ok = await _smart_pick_with_optional_handyman(
                    page,
                    pick=profile_pick,
                    task_name="profile / account type",
                    handyman=handyman,
                )
                if ok:
                    did_something = True
                    await log_event("selected_profile", str(profile_pick))
                    await page.wait_for_timeout(tenant.wait_after_click_ms)

        client_pick = tenant.client_pick
        body_text = await _visible_text(page)
        body_low = body_text.lower()
        if client_pick.get("match"):
            if client_pick["match"].lower() in body_low or "client" in body_low or "company" in body_low:
                ok = await _smart_pick_with_optional_handyman(
                    page,
                    pick=client_pick,
                    task_name="client / company",
                    handyman=handyman,
                )
                if ok:
                    did_something = True
                    await log_event("selected_client", str(client_pick))
                    await page.wait_for_timeout(tenant.wait_after_click_ms)

        clicked_continue = await _click_continue(page, tenant.continue_button_text)
        if clicked_continue:
            did_something = True
            await log_event("clicked_continue", tenant.continue_button_text)
            await page.wait_for_timeout(tenant.wait_after_click_ms)

        still_selection = await _screen_seems_like_selection(page)
        print(f"🧪 round={round_num}, did_something={did_something}, still_selection={still_selection}, url={page.url}")

        if not did_something and not still_selection:
            await log_event("done", "No more Heartland selection screens detected.")
            return {"ok": True, "reason": "completed", "events": events, "profile": tenant.raw}

        if did_something:
            continue

        if handyman is not None and still_selection:
            try:
                ok = await handyman.smart_click(
                    page,
                    task=(
                        "On Heartland, complete the visible profile/client selection screen. "
                        "Select the relevant profile/client based on the tenant profile, then continue."
                    ),
                    max_steps=6,
                )
                if ok:
                    await log_event("handyman_selection_attempt", "clicked on selection screen")
                    await page.wait_for_timeout(tenant.wait_after_click_ms)
                    continue
            except Exception:
                pass

        break

    body_final = await _visible_text(page)
    await _debug_heartland_page(page, "FAILED tenant selection final state")
    await log_event("failed", "Heartland selection did not complete cleanly.")
    return {
        "ok": False,
        "reason": "selection_loop_or_unknown_screen",
        "events": events,
        "last_screen_text": body_final[:1000],
        "profile": tenant.raw,
    }


def tenant_profile_preview(username: str, users_col=None) -> Dict[str, Any]:
    p = get_tenant_profile(username, users_col)
    return {"username": p.username, "parser_profile": p.parser_profile, "heartland": p.heartland}


def profile_to_legacy_maps(username: str, users_col=None) -> Dict[str, Any]:
    p = get_tenant_profile(username, users_col)
    return {
        "client_profile": p.parser_profile,
        "heartland_multiclient_pick": p.client_pick,
        "heartland_multiaccount_pick": p.profile_pick,
        "heartland_employeeid_report_pick": p.employeeid_report_pick,
    }
