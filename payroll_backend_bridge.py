"""
payroll_backend_bridge.py
-------------------------

Backend integration helpers for payrollrunner_dbkeys_handyman.py.

This file does NOT replace your runner.
It gives you small functions to import and call.

Add near top of payrollrunner_dbkeys_handyman.py:

    from payroll_backend_bridge import (
        get_runner_parser_profile,
        handle_heartland_selection_for_user,
        make_user_run_context,
        save_salondata_download_for_user,
    )

Use:
    run_ctx = make_user_run_context(username)
    csv_path = await save_salondata_download_for_user(download, run_ctx)

After Heartland login/MFA:
    result = await handle_heartland_selection_for_user(page, username, handyman=_get_handyman_agent())
    if not result["ok"]:
        raise RuntimeError(...)
"""

from __future__ import annotations

from typing import Any, Dict

from mongo_helpers import get_users_collection
from run_context import make_run_context, RunContext, save_download_as
from multi_tenant_profiles import (
    get_parser_profile_name,
    handle_heartland_post_login_selection_flow,
)


def get_runner_parser_profile(username: str) -> str:
    """
    Replaces CLIENT_PROFILE_BY_USER.get(username, "standard").
    """
    return get_parser_profile_name(username, get_users_collection())


def make_user_run_context(username: str) -> RunContext:
    """
    Creates a per-user/per-run folder.
    """
    return make_run_context(username)


async def save_salondata_download_for_user(download, run_ctx: RunContext) -> str:
    """
    Saves SalonData CSV safely, instead of overwriting salondata_payroll.csv.
    """
    return await save_download_as(download, run_ctx, "salondata_payroll.csv")


async def handle_heartland_selection_for_user(page, username: str, *, handyman=None) -> Dict[str, Any]:
    """
    Handles Heartland profile/client screens using the user's Mongo tenant profile.
    """
    return await handle_heartland_post_login_selection_flow(
        page,
        username,
        users_col=get_users_collection(),
        handyman=handyman,
    )
