import asyncio
import inspect
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PORTAL_SOURCE = (ROOT / "tester8_admin_handyman.py").read_text(encoding="utf-8")
RUNNER_SOURCE = (ROOT / "payrollrunner_dbkeys_handyman.py").read_text(encoding="utf-8")


def _function_source(source: str, name: str) -> str:
    match = re.search(rf"^def {name}\(.*?(?=^def |\Z)", source, re.M | re.S)
    assert match, f"Could not find function {name}"
    return match.group(0)


def _async_function_source(source: str, name: str) -> str:
    match = re.search(rf"^async def {name}\(.*?(?=^(?:async )?def |\Z)", source, re.M | re.S)
    assert match, f"Could not find async function {name}"
    return match.group(0)


def _portal_status_fn():
    ns = {}
    exec(_function_source(PORTAL_SOURCE, "_friendly_error"), ns)
    exec(_function_source(PORTAL_SOURCE, "derive_status_view"), ns)
    return ns["derive_status_view"]


def test_readiness_status_uses_updated_at_for_stale_timeout():
    """Regression: readiness used to write ts while the UI read updated_at."""
    fn = _function_source(PORTAL_SOURCE, "_set_readiness_status")

    assert '"updated_at"' in fn
    assert '"ts"' not in fn

    assert 'readiness_status.get("updated_at")' in PORTAL_SOURCE
    assert 'readiness_status.get("ts")' in PORTAL_SOURCE


def test_heartland_login_separates_readiness_and_payroll_state():
    """Readiness MFA must not poison payroll.state with awaiting_mfa/failed."""
    fn = _async_function_source(RUNNER_SOURCE, "_heartland_login")
    signature = fn.splitlines()[0]

    assert "flow" in signature
    assert 'flow="readiness"' in RUNNER_SOURCE or 'flow = "readiness"' in RUNNER_SOURCE
    assert 'flow="payroll"' in RUNNER_SOURCE or 'flow = "payroll"' in RUNNER_SOURCE

    readiness_branch = re.search(r'flow\s*==\s*["\']readiness["\'].*?(?:elif|else|$)', fn, re.S)
    assert readiness_branch, "readiness branch should update only readiness_status"
    assert 'substate="awaiting_mfa"' in readiness_branch.group(0)
    assert '"payroll.state"' not in readiness_branch.group(0)


def test_mfa_wait_is_scoped_to_run_id_or_session_id():
    """Regression: a global per-user mfa_code lets old workers consume new MFA codes."""
    import payrollrunner_dbkeys_handyman as runner

    signature = inspect.signature(runner._wait_for_mfa_code)

    assert "run_id" in signature.parameters
    assert "flow" in signature.parameters
    wait_source = _async_function_source(RUNNER_SOURCE, "_wait_for_mfa_code")
    assert 'doc.get("mfa_code")' not in wait_source
    assert 'doc.get("mfa")' in wait_source
    assert "mfa.run_id" in RUNNER_SOURCE or '"mfa_run_id"' in RUNNER_SOURCE


def test_backend_writes_are_guarded_by_active_run_id():
    """Old workers must not update visible state after logout or a newer run starts."""
    assert '"readiness_status.run_id"' in RUNNER_SOURCE
    assert '"payroll.run_id"' in RUNNER_SOURCE
    assert "_flow_run_filter" in RUNNER_SOURCE


def test_duplicate_runs_are_blocked_from_backend_state_not_only_thread_state():
    """Two tabs/processes should not start duplicate readiness/payroll runs."""
    actions_block = re.search(
        r"check_disabled\s*=.*?run_clicked\s*=.*?\)",
        PORTAL_SOURCE,
        re.S,
    )
    assert actions_block, "Could not find action button disable block"
    block = actions_block.group(0)

    assert "readiness_running" in block
    assert "payroll_running" in block
    assert "readiness_thread_alive or not is_valid_friday" not in block
    assert "payroll_thread_alive" not in block


def test_pdf_ready_and_dropdown_use_latest_generated_pdf_only():
    """UI should show the latest generated PDF slot, not old history entries."""
    from datetime import date, datetime

    ns = {"date": date, "datetime": datetime}
    exec(_function_source(PORTAL_SOURCE, "_period_key"), ns)
    exec(_function_source(PORTAL_SOURCE, "_latest_pdf_items"), ns)
    exec(_function_source(PORTAL_SOURCE, "_latest_pdf_matches_period"), ns)

    user_doc = {
        "last_pdf": {"period_end": "05/15/2026", "path": "latest.pdf", "ts": 3},
        "pdf_history": [
            {"period_end": "04/17/2026", "path": "old.pdf", "ts": 1},
            {"period_end": "05/08/2026", "path": "older.pdf", "ts": 2},
        ],
    }

    assert [item["path"] for item in ns["_latest_pdf_items"](user_doc)] == ["latest.pdf"]
    assert ns["_latest_pdf_items"]({"pdf_history": user_doc["pdf_history"]}) == []
    assert ns["_latest_pdf_matches_period"](user_doc, "05/15/2026") is True
    assert ns["_latest_pdf_matches_period"](user_doc, date(2026, 5, 29)) is False
    assert '("Current PDF", pdf_status)' in PORTAL_SOURCE
    assert "selected = latest_pdf_items[0] if latest_pdf_items else {}" in PORTAL_SOURCE
    assert 'st.selectbox("Select file"' not in PORTAL_SOURCE


def test_load_clean_biweekly_table_skips_empty_employee_blocks():
    import payrollrunner_dbkeys_handyman as runner

    sample = """Payroll Detail Report - Biweekly, #1234
John Doe, Position:
Total Tips, Charge Tips, 100
, Position:
Total Tips, Charge Tips, 50
"""
    csv_path = ROOT / ".pytest-tmp" / "biweekly_nan_employee.csv"
    csv_path.parent.mkdir(exist_ok=True)
    csv_path.write_text(sample, encoding="utf-8")

    df = runner.load_clean_biweekly_table(str(csv_path))

    assert df["Employee"].tolist() == ["John Doe"]
    assert df["Tips"].tolist() == ["100.00"]


def test_workflow_steps_are_hidden_until_payroll_execution_starts():
    """Readiness success should not render a full payroll progress row."""
    fn = _function_source(PORTAL_SOURCE, "_render_workflow_steps")

    assert "show_payroll_progress" in fn
    assert "if not show_payroll_progress:" in fn
    assert "return" in fn
    assert "or ss.mfa_active" in fn
    assert "mfa_step_active" in fn
    assert '("Readiness", r_state' not in fn
    assert '("Execute payroll",' not in fn
    assert "_render_compact_tracker" in fn
    assert "Readiness tracker" in fn
    assert "Payroll tracker" in fn
    assert 'upload_done = p_state in ("generating_pdf", "completed")' in fn
    assert 'pdf_active = p_state == "generating_pdf"' in fn
    assert '("Payroll upload", "active" if upload_active else "done" if upload_done else "pending")' in fn
    assert '("Current PDF", pdf_status)' in fn
    assert 'p_state in ("running", "completed") or r_state in ("ready", "not_ready")' not in fn


def test_payroll_upload_happens_before_validation_pdf_generation():
    """Heartland upload should start before the recoverable PDF work."""
    fn = _async_function_source(RUNNER_SOURCE, "_full_agentic_flow_inner")

    upload_idx = fn.index("await upload_to_heartland")
    pdf_idx = fn.index("_build_store_log_pdf_for_user")

    assert upload_idx < pdf_idx
    assert '"generating_pdf"' in fn


def test_pdf_failure_after_upload_is_recorded_without_failing_payroll():
    """A post-upload PDF error should be tracked separately from payroll failure."""
    flow_fn = _async_function_source(RUNNER_SOURCE, "_full_agentic_flow_inner")
    run_fn = _function_source(RUNNER_SOURCE, "run_payroll_for_user")

    assert "except Exception as e:" in flow_fn
    assert "_set_payroll_pdf_error(username, pdf_error, run_id=run_id)" in flow_fn
    assert '"pdf_error": pdf_error' in flow_fn
    assert '_update_payroll_status(username, "completed", run_id=run_id)' in run_fn
    assert 'result.get("pdf_error")' not in run_fn


def test_regenerate_pdf_never_uploads_to_heartland():
    """Regenerate PDF must rebuild the validation PDF only, with no duplicate upload."""
    fn = _function_source(RUNNER_SOURCE, "regenerate_validation_pdf_for_user")

    assert "_build_store_log_pdf_for_user" in fn
    assert "_set_payroll_pdf_error" in fn
    assert "upload_to_heartland" not in fn


def test_runner_exports_regenerate_pdf_function():
    """The Streamlit import path must expose the regenerate entry point."""
    import payrollrunner_dbkeys_handyman as runner

    assert callable(runner.regenerate_validation_pdf_for_user)


def test_ui_exposes_pdf_only_recovery_state():
    """The PDF section should make recovery explicit and disable it during work."""
    assert "Payroll uploaded. Creating validation PDF..." in PORTAL_SOURCE
    assert "Payroll was uploaded, but the validation PDF could not be created." in PORTAL_SOURCE
    assert "Creates the PDF only. Payroll will not be uploaded again." in PORTAL_SOURCE
    assert "Regenerate PDF" in PORTAL_SOURCE
    assert "regen_disabled = readiness_running or payroll_running" in PORTAL_SOURCE
    assert "_regenerate_validation_pdf_for_user" in PORTAL_SOURCE
    assert "importlib.reload" in PORTAL_SOURCE


def test_compact_tracker_uses_css_spinner_not_streamlit_status_boxes():
    tracker_fn = _function_source(PORTAL_SOURCE, "_render_compact_tracker")
    workflow_fn = _function_source(PORTAL_SOURCE, "_render_workflow_steps")

    assert "workflow-tracker" in PORTAL_SOURCE
    assert "workflow-spin" in PORTAL_SOURCE
    assert "workflow-step {safe_status}" in tracker_fn
    assert "unsafe_allow_html=True" in tracker_fn
    assert "st.info(f\"Step" not in workflow_fn
    assert "st.success(f\"Step" not in workflow_fn


def test_status_view_handles_real_readiness_and_payroll_state_combinations():
    derive_status_view = _portal_status_fn()

    assert derive_status_view(
        r_state="ready",
        mfa_flow="readiness",
        mfa_status="post_mfa_running",
    )[1] == "Your payroll is ready to run."

    assert derive_status_view(
        payroll_running=True,
        p_state="running",
        r_state="ready",
        mfa_flow="readiness",
        mfa_status="post_mfa_running",
    )[1] == "Executing payroll..."

    assert derive_status_view(
        payroll_running=True,
        p_state="awaiting_mfa",
        r_state="ready",
        mfa_flow="payroll",
        mfa_status="awaiting_mfa",
    )[1] == "Enter your Heartland MFA code."

    assert derive_status_view(
        readiness_running=True,
        r_state="syncing_keys",
        r_missing=["New Person"],
        mfa_flow="readiness",
        mfa_status="awaiting_mfa",
    ) == ("warning", "Enter your Heartland MFA code.", "New employees: New Person")

    assert derive_status_view(
        payroll_done=True,
        p_state="completed",
        pdf_error="PDF failed",
    )[1] == "Payroll uploaded successfully. Validation PDF needs attention."

    assert derive_status_view(
        payroll_running=True,
        p_state="generating_pdf",
        pdf_error="PDF failed",
    )[1] == "Payroll uploaded. Creating validation PDF..."


def test_status_ui_uses_single_derived_status_contract():
    derive_fn = _function_source(PORTAL_SOURCE, "derive_status_view")

    assert "status_kind, status_msg, status_caption = derive_status_view(" in PORTAL_SOURCE
    assert "_notify(status_kind, status_msg, status_caption)" in PORTAL_SOURCE
    assert 'mfa_flow == "payroll"' in derive_fn
    assert 'mfa_flow == "readiness"' in derive_fn


def test_readiness_terminal_states_clear_stale_mfa_state():
    fn = _function_source(PORTAL_SOURCE, "_start_readiness_thread")

    assert '"$unset": {"mfa": ""}' in fn
    assert fn.count('"$unset": {"mfa": ""}') >= 4


def test_vendor_credential_errors_are_friendly_and_specific():
    import payrollrunner_dbkeys_handyman as runner

    assert runner._friendly_error_message("SALONDATA_LOGIN_BAD_PASSWORD") == (
        "SalonData login failed. The saved password looks wrong. Update your SalonData password and try again."
    )
    assert runner._friendly_error_message("SALONDATA_LOGIN_BAD_USERNAME") == (
        "SalonData login failed. Ask an admin to confirm the SalonData username for this account."
    )
    assert runner._friendly_error_message("HEARTLAND_LOGIN_BAD_PASSWORD") == (
        "Heartland login failed. Update your Heartland password and try again."
    )
    assert runner._friendly_error_message("HEARTLAND_LOGIN_BAD_USERNAME") == (
        "Heartland login failed. Ask an admin to confirm the Heartland username for this account."
    )
    assert "MFA" in runner._friendly_error_message("Heartland MFA did not finish after the code was submitted.")


def test_vendor_login_text_classification_drives_recovery_copy():
    import payrollrunner_dbkeys_handyman as runner

    assert runner._vendor_login_error_from_text("SALONDATA", "Invalid login") == "SALONDATA_LOGIN_BAD_PASSWORD"
    assert runner._vendor_login_error_from_text("SALONDATA", "Account not found") == "SALONDATA_LOGIN_BAD_USERNAME"
    assert runner._vendor_login_error_from_text("HEARTLAND", "Email or password is incorrect") == "HEARTLAND_LOGIN_BAD_PASSWORD"
    assert runner._vendor_login_error_from_text("HEARTLAND", "Access denied") == "HEARTLAND_LOGIN_BAD_USERNAME"


def test_login_flows_detect_vendor_credential_failures_before_generic_automation_errors():
    salondata_fn = _async_function_source(RUNNER_SOURCE, "download_salondata_csv")
    heartland_fn = _async_function_source(RUNNER_SOURCE, "_heartland_login")

    assert "_vendor_login_error_from_text(\"SALONDATA\"" in salondata_fn
    assert "SALONDATA_LOGIN_BAD_PASSWORD" in salondata_fn
    assert "_vendor_login_error_from_text(\"HEARTLAND\"" in heartland_fn
    assert "HEARTLAND_LOGIN_FAILED" in heartland_fn
    assert "_set_mfa_status" in heartland_fn


def test_password_recovery_ui_is_hidden_by_default_and_username_read_only():
    assert "_integration_creds_for_display" in PORTAL_SOURCE
    assert "(u or {}).get(\"vendors\", {}).get(key, {})" in PORTAL_SOURCE
    assert "Reveal saved SalonData password" in PORTAL_SOURCE
    assert "Reveal saved Heartland password" in PORTAL_SOURCE
    assert 'value=False, key="pwupd_sd_reveal"' in PORTAL_SOURCE
    assert 'value=False, key="pwupd_hl_reveal"' in PORTAL_SOURCE
    assert "Only reveal this on a private screen." in PORTAL_SOURCE
    assert "Saved SalonData username" in PORTAL_SOURCE
    assert "Saved Heartland username" in PORTAL_SOURCE
    assert 'key="pwupd_sd_username")' in PORTAL_SOURCE
    assert 'key="pwupd_hl_username")' in PORTAL_SOURCE
    assert "Only admins can change vendor usernames." in PORTAL_SOURCE
    assert "Update SalonData password" in PORTAL_SOURCE
    assert "Update Heartland password" in PORTAL_SOURCE
    assert "disabled=sd_update_busy" in PORTAL_SOURCE
    assert "disabled=hl_update_busy" in PORTAL_SOURCE


def test_wait_for_mfa_code_rejects_cancelled_run(monkeypatch):
    import payrollrunner_dbkeys_handyman as runner

    monkeypatch.setattr(runner, "_set_mfa_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_get_user_doc",
        lambda username: {
            "username": username,
            "payroll": {"cancel_requested": True},
            "mfa": {"run_id": "run-1", "flow": "payroll", "code": "439194"},
        },
    )

    with pytest.raises(RuntimeError, match="cancelled"):
        asyncio.run(runner._wait_for_mfa_code("owner@example.com", run_id="run-1", timeout_sec=1, poll_sec=0))


def test_wait_for_mfa_code_times_out_with_user_action_message(monkeypatch):
    import payrollrunner_dbkeys_handyman as runner

    monkeypatch.setattr(runner, "_set_mfa_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_get_user_doc",
        lambda username: {"username": username, "payroll": {"run_id": "run-1"}, "mfa": {"run_id": "run-1", "code": ""}},
    )

    with pytest.raises(RuntimeError, match="Submit MFA"):
        asyncio.run(runner._wait_for_mfa_code("owner@example.com", run_id="run-1", timeout_sec=0, poll_sec=0))


def test_wait_for_mfa_code_normalizes_six_digit_code(monkeypatch):
    import payrollrunner_dbkeys_handyman as runner

    statuses = []
    monkeypatch.setattr(runner, "_set_mfa_status", lambda *args, **kwargs: statuses.append(kwargs.get("status")))
    monkeypatch.setattr(
        runner,
        "_get_user_doc",
        lambda username: {
            "username": username,
            "payroll": {"run_id": "run-1"},
            "mfa": {"run_id": "run-1", "flow": "payroll", "code": "code: 439194 extra"},
        },
    )

    code = asyncio.run(runner._wait_for_mfa_code("owner@example.com", run_id="run-1", timeout_sec=1, poll_sec=0))

    assert code == "439194"
    assert "submitted" in statuses


def test_heartland_employee_parser_preserves_dashed_employee_ids(monkeypatch):
    import pandas as pd
    import payrollrunner_dbkeys_handyman as runner

    raw = pd.DataFrame(
        [
            ["Number", "Employee Name", "First Name", "Last Name", "Status", "Code"],
            ["0001-3637", "Alicia G Brewer", "Alicia", "Brewer", "A", "9249"],
            ["524", "Haley G Holmes", "Haley", "Holmes", None, "5822"],
            ["0000-6025", "Kerri L Bramlett", "Kerri", "Bramlett", "T", "9249"],
        ],
        columns=["Employee", "Unnamed: 1", "Employee.1", "Employee.2", "Unnamed: 4", "Department"],
    )
    monkeypatch.setattr(pd, "read_excel", lambda path: raw)

    parsed = runner._parse_employee_excel_to_df("fake.xlsx")

    assert {"0001-3637", "524"} == set(parsed["Key"])
    assert "0000-6025" not in set(parsed["Key"])
    assert dict(zip(parsed["Employee"], parsed["Key"]))["Alicia Brewer"] == "0001-3637"


def test_employee_key_lookup_handles_heartland_name_variants():
    import pandas as pd
    import payrollrunner_dbkeys_handyman as runner

    keys = pd.DataFrame(
        [
            {"Employee": "Tami Reynolds-Ray", "Key": "518", "Department": "9249"},
            {"Employee": "Sue Ellen Mathews", "Key": "627", "Department": "5822"},
            {"Employee": "Hayley Arnold", "Key": "584", "Department": "1067"},
        ]
    )
    lookup = runner._employee_key_lookup(keys)

    assert runner._lookup_employee_key("Tami Reynolds", lookup) == "518"
    assert runner._lookup_employee_key("Sue Mathews", lookup) == "627"
    assert runner._lookup_employee_key("Haley Arnold", lookup) == "584"


def test_geoff_formatter_maps_training_to_breaks_paid_and_keeps_training_zero(monkeypatch):
    import pandas as pd
    import payrollrunner_dbkeys_handyman as runner

    payroll = pd.DataFrame(
        [
            {
                "Employee": "Haley Arnold",
                "Dept": "1067",
                "Pay Rate": "0.00",
                "FLOOR (Earn Hrs)": "0.00",
                "CLOSING (Earn Hrs)": "0.49",
                "BREAKS PAID (Earn Hrs)": "0.00",
                "ADMIN (Earn Hrs)": "0.00",
                "TRAINING (Earn Hrs)": "0.57",
                "OVERTIME (Earn Hrs)": "0.00",
                "BONUS (Earn $)": "0.00",
                "COMMISSION (Earn $)": "0.00",
                "CREDIT TIPS (Earn $)": "0.00",
                "RECEPTIONISTS (Earn Hrs)": "44.60",
            }
        ]
    )
    keys = pd.DataFrame([{"Employee": "Hayley Arnold", "Key": "584", "Department": "1067"}])

    monkeypatch.setattr(runner, "load_clean_biweekly_table_for_user", lambda csv_path, username: payroll)
    monkeypatch.setattr(runner, "load_employee_keys_df", lambda username: keys)

    out_path = ROOT / ".pytest-tmp" / "geoff_unit.csv"
    out_path.parent.mkdir(exist_ok=True)
    runner.format_csv_for_heartland_geoff("ignored.csv", "quopayroll@gmail.com", str(out_path))
    out = pd.read_csv(out_path, dtype=str).fillna("")

    assert out.loc[0, "Key"] == "584"
    assert out.loc[0, "E_Breaks Paid_Hours"] == "0.57"
    assert out.loc[0, "E_Training_Hours"] == "0.00"


def test_geoff_pdf_tables_restore_location_ot_total_hours_and_pay():
    import pandas as pd
    import payrollrunner_dbkeys_handyman as runner

    payroll = pd.DataFrame(
        [
            {
                "Employee": "Haley Arnold",
                "Dept": "1067",
                "Pay Rate": "0.00",
                "FLOOR (Earn Hrs)": "10.00",
                "CLOSING (Earn Hrs)": "1.50",
                "BREAKS PAID (Earn Hrs)": "2.00",
                "ADMIN (Earn Hrs)": "3.00",
                "TRAINING (Earn Hrs)": "4.00",
                "OVERTIME (Earn Hrs)": "50.00",
                "BONUS (Earn $)": "20.00",
                "COMMISSION (Earn $)": "30.00",
                "CREDIT TIPS (Earn $)": "40.00",
                "RECEPTIONISTS (Earn Hrs)": "5.00",
            }
        ]
    )

    salon_totals = pd.DataFrame(
        [
            {
                "SalonName": "Example Salon",
                "Dept": "1067",
                "OT Dollars": 50.0,
                "TOTAL Hours": 25.5,
                "Pay": 420.0,
            }
        ]
    )
    hours_agg, money_agg, _df_std, hours_cols, money_cols = runner._build_geoff_pdf_report_frames(payroll, salon_totals)

    assert "OVERTIME (Earn Hrs)" not in hours_cols
    assert "OVERTIME (Earn Hrs)" not in hours_agg.columns
    assert "OT Dollars" in hours_cols
    assert "TOTAL Hours" in hours_cols
    assert "Pay" in hours_cols
    assert "Gross Pay" not in money_cols

    location_hours = hours_agg.loc[hours_agg["Dept"].eq("1067")].iloc[0]

    assert location_hours["Breaks Paid Hrs"] == pytest.approx(6.0)
    assert location_hours["OT Dollars"] == pytest.approx(50.0)
    assert location_hours["TOTAL Hours"] == pytest.approx(25.5)
    assert location_hours["Pay"] == pytest.approx(420.0)


def test_geoff_salon_summary_parser_reads_ot_dollars_total_hours_and_pay():
    import payrollrunner_dbkeys_handyman as runner

    report = ROOT / ".pytest-tmp" / "geoff_summary_report.csv"
    report.parent.mkdir(exist_ok=True)
    report.write_text(
        "\n".join(
            [
                '"Payroll Detail Report - Biweekly","Example Salon #1067"',
                '"SALON TOTALS"',
                '"OT Hrs","2.00","","75.50"',
                '"TOTALS*","25.50","","420.00"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    totals = runner._parse_geoff_salon_summary_totals(str(report))
    location = totals.iloc[0]

    assert location["SalonName"] == "Example Salon"
    assert location["Dept"] == "1067"
    assert location["OT Dollars"] == pytest.approx(75.5)
    assert location["TOTAL Hours"] == pytest.approx(25.5)
    assert location["Pay"] == pytest.approx(420.0)


def test_employee_alias_keys_handles_float_nan_names():
    import payrollrunner_dbkeys_handyman as runner

    assert runner._employee_alias_keys(float("nan")) == set()
    assert runner._emp_key(float("nan")) == ""


def test_standard_formatter_handles_numeric_blank_employee_cells(monkeypatch):
    import pandas as pd
    import payrollrunner_dbkeys_handyman as runner

    payroll = pd.DataFrame(
        [
            {"Employee": "Jane Doe", "Dept": "1234", "E_Regular_Hours": "8.00"},
            {"Employee": float("nan"), "Dept": "1234", "E_Regular_Hours": "0.00"},
        ]
    )
    keys = pd.DataFrame([{"Employee": "Jane Doe", "Key": "0001-1234", "Department": "1234"}])

    monkeypatch.setattr(runner, "load_clean_biweekly_table_for_user", lambda csv_path, username: payroll)
    monkeypatch.setattr(runner, "load_employee_keys_df", lambda username: keys)

    out_path = ROOT / ".pytest-tmp" / "standard_float_name.csv"
    out_path.parent.mkdir(exist_ok=True)
    runner.format_csv_for_heartland("ignored.csv", "owner@example.com", str(out_path))
    out = pd.read_csv(out_path, dtype=str).fillna("")

    assert out["Key"].tolist() == ["0001-1234"]
