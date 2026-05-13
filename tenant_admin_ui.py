"""
tenant_admin_ui.py
------------------

Streamlit Admin UI for editing per-user tenant profiles.

How to use in tester8_admin_handyman.py:

    from tenant_admin_ui import render_tenant_profile_admin

    # inside if menu == "Admin":
    render_tenant_profile_admin(users)

This lets one Streamlit app support many clients/users.
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import streamlit as st

from multi_tenant_profiles import (
    get_tenant_profile,
    upsert_tenant_profile,
    tenant_profile_preview,
)
from mongo_helpers import norm_username


def _list_usernames(users_col) -> list[str]:
    try:
        docs = list(users_col.find({}, {"username": 1, "_id": 0}).sort("username", 1))
        return [str(d.get("username") or "").strip().lower() for d in docs if d.get("username")]
    except Exception:
        return []


def render_tenant_profile_admin(users_col, *, key_prefix: str = "tenant_admin") -> None:
    """
    Render this inside the Admin page.
    """
    st.markdown("### Multi-Client / Tenant Profiles")
    st.caption(
        "Use this to configure how each portal user gets through Heartland profile/client selection "
        "inside the single Streamlit app."
    )

    usernames = _list_usernames(users_col)
    if not usernames:
        st.info("No users found yet.")
        return

    selected = st.selectbox(
        "Choose user to configure",
        usernames,
        key=f"{key_prefix}_selected_user",
    )

    current = get_tenant_profile(selected, users_col)

    with st.expander("Current profile JSON", expanded=False):
        st.json(tenant_profile_preview(selected, users_col))

    st.markdown("#### Profile Settings")

    parser_profile = st.selectbox(
        "Parser profile",
        ["standard", "geoff"],
        index=0 if current.parser_profile != "geoff" else 1,
        help="Do not change parser code here. This only selects which existing parser profile to use.",
        key=f"{key_prefix}_parser_profile",
    )

    st.markdown("#### Heartland Profile / Multi-Account Screen")
    st.caption("Use this when Heartland asks for something like Partner User / Company User / profile type.")

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        profile_match = st.text_input(
            "Profile match text",
            value=current.profile_pick.get("match", ""),
            placeholder="Example: Partner User",
            key=f"{key_prefix}_profile_match",
        )
    with c2:
        profile_index = st.number_input(
            "Profile index",
            min_value=0,
            value=int(current.profile_pick.get("index", 0)),
            step=1,
            key=f"{key_prefix}_profile_index",
        )
    with c3:
        profile_required = st.checkbox(
            "Required",
            value=bool(current.profile_pick.get("required", False)),
            key=f"{key_prefix}_profile_required",
        )

    st.markdown("#### Heartland Client / Multi-Client Screen")
    st.caption("Use this when Heartland asks which client/company to enter.")

    c4, c5, c6 = st.columns([3, 1, 1])
    with c4:
        client_match = st.text_input(
            "Client match text",
            value=current.client_pick.get("match", ""),
            placeholder="Example: Great Clips",
            key=f"{key_prefix}_client_match",
        )
    with c5:
        client_index = st.number_input(
            "Client index",
            min_value=0,
            value=int(current.client_pick.get("index", 0)),
            step=1,
            key=f"{key_prefix}_client_index",
        )
    with c6:
        client_required = st.checkbox(
            "Required ",
            value=bool(current.client_pick.get("required", False)),
            key=f"{key_prefix}_client_required",
        )

    st.markdown("#### Heartland Employee ID Report")
    c7, c8 = st.columns([3, 1])
    with c7:
        employeeid_match = st.text_input(
            "Employee ID report match text",
            value=current.employeeid_report_pick.get("match", "Employee id"),
            placeholder="Example: Employee id",
            key=f"{key_prefix}_employeeid_match",
        )
    with c8:
        employeeid_index = st.number_input(
            "Report index",
            min_value=0,
            value=int(current.employeeid_report_pick.get("index", 0)),
            step=1,
            key=f"{key_prefix}_employeeid_index",
        )

    if st.button("Save tenant profile", type="primary", key=f"{key_prefix}_save"):
        upsert_tenant_profile(
            users_col,
            selected,
            parser_profile=parser_profile,
            profile_match=profile_match,
            profile_index=int(profile_index),
            client_match=client_match,
            client_index=int(client_index),
            employeeid_match=employeeid_match,
            employeeid_index=int(employeeid_index),
            profile_required=bool(profile_required),
            client_required=bool(client_required),
        )
        st.success(f"Saved tenant profile for {selected}.")
        st.rerun()

    st.markdown("#### Quick setup examples")
    st.code(
        """# Both profile + client selection
Profile match: Partner User
Profile index: 1
Client match: Great Clips
Client index: 0

# Client selection only
Profile match: [blank]
Client match: Great Clips
Client index: 0

# Profile selection only
Profile match: Partner User
Profile index: 1
Client match: [blank]

# No Heartland selection screen
Profile match: [blank]
Client match: [blank]""",
        language="text",
    )
