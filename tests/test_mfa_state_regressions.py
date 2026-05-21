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
