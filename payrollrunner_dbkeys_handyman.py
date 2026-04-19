import os
import time
import asyncio
import csv
import re
import traceback
import pandas as pd

from dotenv import load_dotenv
from typing import Optional, Tuple, List, Dict, Any

try:
    import streamlit as st
except Exception:
    st = None

from pymongo import MongoClient
from bson import ObjectId
import gridfs
from playwright.async_api import async_playwright, Page

from crypto_utils import decrypt_str
from vision_handyman_agent import BrowserHandymanAgent
PDF_OUTPUT_DIR = os.path.join(os.getcwd(), "payroll_pdfs")
os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)


def _get_secret_or_env(name: str, default: str = "") -> str:
    try:
        if st is not None:
            v = st.secrets.get(name, None)
            if v not in (None, ""):
                return str(v).strip()
    except Exception:
        pass
    v = os.getenv(name, "").strip()
    if v:
        return v
    return default

MONGO_URI = _get_secret_or_env("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = _get_secret_or_env("MONGO_DB", "payrollInfo")
MONGO_COLL = "userInfo"

# ---------- GridFS bucket for PDFs ----------
PDF_GRIDFS_BUCKET = _get_secret_or_env("MONGO_PDF_GRIDFS_BUCKET", "payroll_pdfs")
KEEP_LOCAL_PDFS = str(_get_secret_or_env("KEEP_LOCAL_PDFS", "0")).strip().lower() in ("1","true","yes")

# ---------- Per-user employee keys (stored in Mongo) ----------
KEYS_COLL = _get_secret_or_env("MONGO_KEYS_COLL", "employeeKeysByUser")

# ---------- Client-specific behavior (username-driven) ----------
CLIENT_PROFILE_BY_USER = {
    "quopayroll@gmail.com": "geoff",
}

HEARTLAND_MULTICLIENT_PICK = {
    "quopayroll@gmail.com": {"match": "Great Clips", "index": 0},
    "owner@example.com": {"match": "Great Clips", "index": 0},
}

HEARTLAND_MULTIACCOUNT_PICK = {
    "quopayroll@gmail.com": {"match": "Partner User", "index": 1},
    "owner@example.com": {"match": "Partner User", "index": 1},
}
HEARTLAND_EMPLOYEEID_REPORT_PICK={
    "quopayroll@gmail.com" : {"match": "Employee id", "index":1},
    "owner@example.com" : {"match": "Employee id", "index":0},
}

# ---------- General helpers / regex ----------
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_DEPT_HASH_RE = re.compile(r"#\s*(\d{3,6})\b", re.I)
from datetime import date, datetime, timedelta

def _is_friday(d: date) -> bool:
    return isinstance(d, date) and d.weekday() == 4

def _next_friday(from_day: date) -> date:
    offset = (4 - from_day.weekday()) % 7
    return from_day + timedelta(days=offset)

def _prev_friday(from_day: date) -> date:
    offset = (from_day.weekday() - 4) % 7
    return from_day - timedelta(days=offset)

def _default_payroll_friday(today: date | None = None) -> date:
    t = today or date.today()
    return _prev_friday(t)

def _coerce_date(value) -> date:
    if value is None:
        raise ValueError("period_end_date is required")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    return datetime.strptime(s, "%m/%d/%Y").date()


# ---------- Mongo helpers ----------
_mongo_client_cache: MongoClient | None = None

def _get_mongo_client() -> MongoClient:
    global _mongo_client_cache
    if _mongo_client_cache is None:
        _mongo_client_cache = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=2000,
            uuidRepresentation="standard",
            maxPoolSize=10,
        )
    return _mongo_client_cache

def _get_users_collection():
    return _get_mongo_client()[MONGO_DB][MONGO_COLL]

def _get_db():
    return _get_mongo_client()[MONGO_DB]

def _get_pdf_fs():
    return gridfs.GridFS(_get_db(), collection=PDF_GRIDFS_BUCKET)

def _get_keys_collection():
    col = _get_mongo_client()[MONGO_DB][KEYS_COLL]
    try:
        col.create_index("username", unique=True)
    except Exception:
        pass
    return col

_handyman_agent_cache = None

def _get_handyman_agent():
    global _handyman_agent_cache
    if _handyman_agent_cache is False:
        return None
    if _handyman_agent_cache is None:
        try:
            _handyman_agent_cache = BrowserHandymanAgent()
        except Exception as e:
            print(f"⚠️ Handyman agent unavailable: {e}")
            _handyman_agent_cache = False
    return None if _handyman_agent_cache is False else _handyman_agent_cache



async def _wait_for_mfa_code(username: str, *, timeout_sec: int = 600, poll_sec: float = 2.0) -> str:
    """
    Wait for Heartland MFA code to appear in Mongo under user.mfa_code.
    Lets the user type the code in the Streamlit portal while automation waits.
    """
    deadline = time.time() + timeout_sec
    last_log = 0.0

    while time.time() < deadline:
        doc = _get_user_doc(username) or {}
        raw = str(doc.get("mfa_code") or "").strip()
        code = re.sub(r"\D+", "", raw)

        if len(code) >= 6:
            return code[:6]

        if time.time() - last_log > 10:
            print("⏳ Waiting for Heartland MFA code in portal (Submit MFA)…")
            last_log = time.time()

        await asyncio.sleep(poll_sec)

    raise RuntimeError(
        "Heartland MFA code is required. Please enter the 6-digit code in the portal and click 'Submit MFA'."
    )



def load_employee_keys_df(username: str) -> pd.DataFrame:
    uname = (username or "").lower().strip()
    doc = _get_keys_collection().find_one({"username": uname}, {"_id": 0}) or {}
    rows = doc.get("rows") or []
    df = pd.DataFrame(rows)

    for c in ["Key", "Employee", "Department"]:
        if c not in df.columns:
            df[c] = ""

    df["Key"] = df["Key"].fillna("").astype(str).str.strip()
    df["Key"] = df["Key"].mask(df["Key"].str.lower().isin(["nan","none"]), "")
    df["Employee"] = df["Employee"].fillna("").astype(str).str.strip()
    df["Employee"] = df["Employee"].mask(df["Employee"].str.lower().isin(["nan","none"]), "")
    df["Department"] = df["Department"].fillna("").astype(str).str.strip()
    df["Department"] = df["Department"].mask(df["Department"].str.lower().isin(["nan","none"]), "")

    return df


def save_employee_keys_df(username: str, df: pd.DataFrame, source: str = "portal") -> None:
    uname = (username or "").lower().strip()
    df2 = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)

    for c in ["Key", "Employee", "Department"]:
        if c not in df2.columns:
            df2[c] = ""
        df2[c] = df2[c].fillna("").astype(str).str.strip()
        df2[c] = df2[c].mask(df2[c].str.lower().isin(["nan","none"]), "")

    rows = df2[["Key", "Employee", "Department"]].to_dict("records")
    _get_keys_collection().replace_one(
        {"username": uname},
        {"username": uname, "rows": rows, "updated_at": time.time(), "source": source},
        upsert=True,
    )


def _update_payroll_status(username: str, state: str, error: Optional[str] = None) -> None:
    col = _get_users_collection()
    doc = col.find_one({"username": username}, {"_id": 0}) or {}
    payroll = doc.get("payroll", {}) if isinstance(doc.get("payroll", {}), dict) else {}
    payroll.update({"state": state, "error": error, "updated_at": time.time()})
    col.update_one({"username": username}, {"$set": {"payroll": payroll}}, upsert=True)


def _log_payroll_pdf_item(username: str, item: dict) -> None:
    uname = (username or "").lower().strip()
    period_end = (item.get("period_end") or "").strip()
    if not item.get("ts"):
        item["ts"] = time.time()

    col = _get_users_collection()
    col.update_one({"username": uname}, {"$set": {"last_pdf": item}}, upsert=True)

    doc = col.find_one({"username": uname}, {"pdf_history": 1}) or {}
    hist = doc.get("pdf_history") or []
    if not isinstance(hist, list):
        hist = []

    replaced = False
    new_hist = []
    for h in hist:
        if isinstance(h, dict) and (h.get("period_end") or "").strip() == period_end and period_end:
            new_hist.append(item)
            replaced = True
        else:
            new_hist.append(h)

    if not replaced:
        new_hist.append(item)

    new_hist = sorted(new_hist, key=lambda x: float((x or {}).get("ts", 0) or 0), reverse=True)
    new_hist = new_hist[:25]
    col.update_one({"username": uname}, {"$set": {"pdf_history": new_hist}}, upsert=True)


def _log_payroll_pdf(username: str, pdf_path: str, period_end: str):
    item = {
        "path": pdf_path,
        "filename": os.path.basename(pdf_path) if pdf_path else "",
        "period_end": (period_end or "").strip(),
        "ts": time.time(),
    }
    _log_payroll_pdf_item(username, item)


def _store_pdf_in_gridfs(username: str, pdf_path: str, period_end: str) -> Optional[str]:
    uname = (username or "").lower().strip()
    period_end = (period_end or "").strip()
    if not pdf_path or not os.path.exists(pdf_path):
        return None

    fs = _get_pdf_fs()
    filename = os.path.basename(pdf_path)

    try:
        doc = _get_users_collection().find_one({"username": uname}, {"pdf_history": 1}) or {}
        for h in (doc.get("pdf_history") or []):
            if isinstance(h, dict) and (h.get("period_end") or "").strip() == period_end:
                old_id = str(h.get("gridfs_id") or "").strip()
                if old_id:
                    try:
                        fs.delete(ObjectId(old_id))
                    except Exception:
                        pass
                break
    except Exception:
        pass

    with open(pdf_path, "rb") as f:
        data = f.read()

    gid = fs.put(
        data,
        filename=filename,
        contentType="application/pdf",
        metadata={"username": uname, "period_end": period_end, "ts": time.time()},
    )
    return str(gid)

def _get_user_doc(username: str) -> dict:
    col = _get_users_collection()
    doc = col.find_one({"username": username}, {"_id": 0})
    return doc or {}


def _get_vendor_creds(username: str, vendor: str) -> Tuple[str, str]:
    doc = _get_user_doc(username) or {}

    integrations = doc.get("integrations", {}) if isinstance(doc.get("integrations", {}), dict) else {}
    integ = integrations.get(vendor, {}) if isinstance(integrations.get(vendor, {}), dict) else {}

    u_plain = (integ.get("username") or "").strip()
    u_enc = integ.get("username_enc")
    p_enc = integ.get("password_enc")

    if (u_plain or u_enc) and p_enc:
        u = u_plain if u_plain else decrypt_str(u_enc)
        p = decrypt_str(p_enc)
        return (u or "").strip(), (p or "").strip()

    vendors = doc.get("vendors", {}) if isinstance(doc.get("vendors", {}), dict) else {}
    v = vendors.get(vendor, {}) if isinstance(vendors.get(vendor, {}), dict) else {}

    u_enc2 = v.get("username_enc")
    p_enc2 = v.get("password_enc")

    if u_enc2 and p_enc2:
        u = decrypt_str(u_enc2)
        p = decrypt_str(p_enc2)
        return (u or "").strip(), (p or "").strip()

    raise RuntimeError(f"Missing {vendor} credentials for user '{username}'.")



def _name_norm(x) -> str:
    try:
        if x is None or pd.isna(x):
            return ""
    except Exception:
        pass

    s = str(x).strip().strip('"')
    if not s:
        return ""
    if s.lower() in {"nan", "none", "null", "na"}:
        return ""

    s = re.sub(r"\s+", " ", s).strip()

    if "," in s:
        last, first = [p.strip() for p in s.split(",", 1)]
        first = first.split()[0] if first else ""
        if first and last:
            s = f"{first} {last}"
        else:
            s = (first or last).strip()
    else:
        parts = s.split()
        if len(parts) >= 2:
            s = f"{parts[0]} {parts[-1]}"

    return " ".join(w[:1].upper() + w[1:].lower() if w else "" for w in s.split())

def _last_num(row) -> Optional[float]:
    nums = [m.group(0) for cell in row for m in _NUM_RE.finditer(str(cell))]
    return float(nums[-1]) if nums else None


def _friendly_error_message(err) -> str:
    s = str(err or "").strip()
    s_low = s.lower()

    if "missing salondata credentials" in s_low or "missing salondata" in s_low:
        return "SalonData is not connected for this portal account. Please go to Setup and save your SalonData username/password."
    if "missing heartland credentials" in s_low or "missing heartland" in s_low:
        return "Heartland is not connected for this portal account. Please go to Setup and save your Heartland username/password."
    if "timeout" in s_low:
        return "The website took too long to respond. This is usually a wrong password, a site outage, or a slow connection. Please try again (and update your password if it recently changed)."
    if "mfa" in s_low and ("missing" in s_low or "not found" in s_low):
        return "Heartland MFA code is required. Please enter the 6-digit code and try again."
    if "no view" in s_low and ("icon" in s_low or "icons" in s_low):
        return "Could not find the Employee ID report in Heartland. Please confirm the report name (or update the report picker)."
    if "could not find" in s_low and "employee" in s_low and "key" in s_low:
        return "Employee ID report format changed in Heartland (could not find the expected columns)."
    if ("pdf" in s_low and "not found" in s_low) or ("filenotfounderror" in s_low):
        return "Payroll PDF could not be generated or found. Please run again, and contact support if it keeps happening."

    return s


# ---------- SalonData download ----------
async def download_salondata_csv(
    page: Page,
    sd_user: str,
    sd_pass: str,
    period_end_date=None,
) -> Tuple[str, str]:
    d = _coerce_date(period_end_date) if period_end_date is not None else _default_payroll_friday()
    if not _is_friday(d):
        raise RuntimeError(
            f"SalonData biweekly payroll can only run for a Friday. You selected {d.strftime('%m/%d/%Y')}."
        )
    friday_str = d.strftime("%m/%d/%Y")

    print("🔐 Logging into SalonData...")
    await page.goto("https://reports.salondata.com/static/reports/index.html")

    await page.wait_for_selector('input[placeholder="email"]')
    await page.fill('input[placeholder="email"]', sd_user)
    await page.fill('input[placeholder="password"]', sd_pass)
    await page.click('button:has-text("Log In")')

    handyman = _get_handyman_agent()

    try:
        await page.click('#reportChooser')
    except Exception:
        if handyman:
            ok = await handyman.smart_click(
                page,
                task="On SalonData, open the report chooser dropdown or picker.",
                selector='#reportChooser',
            )
            if not ok:
                raise
        else:
            raise

    try:
        await page.locator('#reportCategoryList').get_by_text('Accounting & Payroll').click()
    except Exception:
        if handyman:
            ok = await handyman.smart_click(
                page,
                task="In SalonData, click the Accounting & Payroll report category.",
            )
            if not ok:
                raise
        else:
            raise

    try:
        await page.locator('#reportList').get_by_text('Payroll Detail - Biweekly').click()
    except Exception:
        if handyman:
            ok = await handyman.smart_click(
                page,
                task="In SalonData, click the report named Payroll Detail - Biweekly.",
            )
            if not ok:
                raise
        else:
            raise

    end_date_input = page.locator('input').nth(1)
    await end_date_input.wait_for(state="visible", timeout=30000)

    async def _set_date_value(loc, value: str):
        try:
            await loc.click()
            await loc.evaluate(
                """(el, value) => {
                    el.focus();
                    el.value = value;
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    el.blur();
                }""",
                value,
            )
            await page.wait_for_timeout(300)
            actual = (await loc.input_value()).strip()
            if actual == value:
                return
        except Exception:
            pass

        await loc.click()
        try:
            await loc.press("Control+A")
        except Exception:
            await page.keyboard.press("Control+A")
        await page.wait_for_timeout(100)
        try:
            await loc.press("Backspace")
        except Exception:
            await page.keyboard.press("Backspace")
        await page.wait_for_timeout(150)
        for ch in value:
            await loc.type(ch, delay=125)
            await page.wait_for_timeout(35)

        actual = (await loc.input_value()).strip()
        if actual != value:
            raise RuntimeError(
                f"Failed to set SalonData end date correctly. Wanted {value}, got {actual or '[blank]'}"
            )

    await _set_date_value(end_date_input, friday_str)

    try:
        await page.click('text=Select All')
    except Exception:
        pass

    try:
        await page.click('text=Run Report')
    except Exception:
        if handyman:
            ok = await handyman.smart_click(
                page,
                task="In SalonData, click the Run Report button.",
            )
            if not ok:
                raise
        else:
            raise

    try:
        await page.wait_for_selector('text=Download CSV', timeout=20000)
    except Exception:
        if handyman:
            await handyman.wait_until_ready(
                page,
                task="Wait until the Download CSV button appears on the SalonData report page.",
                timeout_sec=25.0,
            )

    async with page.expect_download() as download_info:
        clicked = False
        try:
            await page.click("text=Download CSV")
            clicked = True
        except Exception:
            if handyman:
                clicked = await handyman.smart_click(
                    page,
                    task="Click the Download CSV button for the SalonData payroll report.",
                )
        if not clicked:
            raise RuntimeError("Could not click Download CSV in SalonData.")
    download = await download_info.value

    csv_path = "salondata_payroll.csv"
    await download.save_as(csv_path)
    print(f"✅ Downloaded SalonData payroll CSV → {csv_path}")
    return csv_path, friday_str

def _norm_name_any(s: str) -> str:
    return _name_norm(s)


# ---------- Parsing biweekly payroll CSV (DEFAULT / ORIGINAL) ----------
def load_clean_biweekly_table(csv_path: str) -> pd.DataFrame:
    try:
        cash_tips_total = 0.0
        employees = []
        current = {}
        current_dept = ""

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)

            for row in reader:
                if not row:
                    continue

                if str(row[0]).strip().startswith("Payroll Detail Report - Biweekly"):
                    header_text = " ".join(str(c) for c in row)
                    m = _DEPT_HASH_RE.search(header_text)
                    if m:
                        current_dept = m.group(1).strip()
                    continue

                if any("Position:" in str(cell) for cell in row):
                    if current:
                        employees.append(current)

                    current = {}
                    current["Dept"] = str(current_dept)

                    full_name = str(row[0]).strip().strip('"')
                    current["Employee"] = _norm_name_any(full_name)

                    if "Base Wage:" in row:
                        idx = row.index("Base Wage:")
                        current["Pay Rate"] = row[idx + 1].strip().strip('"') if idx + 1 < len(row) else "0.00"

                    continue

                if any("Total Hours" in str(cell) for cell in row):
                    matches = [cell for cell in row if re.match(r"^\d+(\.\d+)?$", str(cell).strip())]
                    if len(matches) >= 2:
                        current["Total Hours"] = str(matches[-2]).strip()

                if any("OT Hrs" in str(cell) for cell in row):
                    matches = [cell for cell in row if re.match(r"^\d+(\.\d+)?$", str(cell).strip())]
                    if len(matches) >= 2:
                        current["OT Hours"] = str(matches[-2]).strip()
                    elif matches:
                        current["OT Hours"] = str(matches[-1]).strip()

                if any("Productivity" in str(cell) for cell in row):
                    matches = [cell for cell in row[::-1] if re.match(r"^\d+(\.\d+)?$", str(cell).strip())]
                    current["Productivity"] = str(matches[0]).strip() if matches else "0.00"

                if any("Product Sales" in str(cell) for cell in row):
                    matches = [cell for cell in row[::-1] if re.match(r"^\d+(\.\d+)?$", str(cell).strip())]
                    current["Prod Bonus"] = str(matches[0]).strip() if matches else "0.00"

                if any("Cash/Check Tips" in str(cell) for cell in row) and any(
                    keyword in " ".join(row) for keyword in ["Floor Hrs", "Floor Pay", "SALON TOTALS", "TOTALS*"]
                ):
                    print("⏭️ Skipping salon-level summary cash tip row.")
                    continue

                if any("Cash/Check Tips" in str(cell) for cell in row):
                    matches = [cell for cell in row if re.match(r"^\d+(\.\d+)?$", str(cell).strip())]
                    if matches:
                        val = float(matches[-1])
                        cash_tips_total += val
                        print(f"💵 Found cash tip: {val}, running total: {cash_tips_total}")

                if any("Total Tips" in str(cell) for cell in row) and any("Charge Tips" in str(cell) for cell in row):
                    matches = [cell for cell in row[::-1] if re.match(r"^\d+(\.\d+)?$", str(cell).strip())]
                    if matches:
                        total_tips = float(matches[0])
                        credit_tips = total_tips - cash_tips_total
                        print(f"🧾 {current.get('Employee','[Unknown]')} | Total: {total_tips}, Cash: {cash_tips_total}, Credit: {credit_tips}")
                        current["Tips"] = f"{credit_tips:.2f}"
                    else:
                        print(f"⚠️ No numeric match found in 'Total Tips' row: {row}")
                        current["Tips"] = "0.00"

                    employees.append(current)
                    current = {}
                    cash_tips_total = 0.0

        if current:
            employees.append(current)

        df = pd.DataFrame(employees)

        if "Dept" in df.columns:
            df["Dept"] = df["Dept"].astype(str).str.replace(".0", "", regex=False).str.strip()

        for col in ["Pay Rate", "Total Hours", "OT Hours", "Productivity", "Prod Bonus", "Tips"]:
            if col not in df.columns:
                df[col] = "0.00"
            else:
                df[col] = df[col].fillna("0.00")

        df["E_Regular_Hours"] = df.apply(
            lambda r: str(round(float(r["Total Hours"]) - float(r["OT Hours"]), 2)),
            axis=1
        )
        df["E_Overtime_Hours"] = df["OT Hours"]
        df.drop(columns=["OT Hours", "Total Hours"], inplace=True)

        return df

    except Exception as e:
        print(f"❌ Error parsing biweekly payroll data: {e}")
        return pd.DataFrame(columns=[
            "Employee", "Dept", "Pay Rate",
            "E_Regular_Hours", "E_Overtime_Hours",
            "Productivity", "Prod Bonus", "Tips"
        ])


def _canon_name(s: str) -> str:
    try:
        if s is None or pd.isna(s):
            return ""
    except Exception:
        pass

    s = str(s).strip().strip('"')
    if not s or s.lower() in {"nan", "none", "null", "na", "<na>"}:
        return ""

    s = re.sub(r"\s+", " ", s).strip()

    if "," in s:
        last, rest = s.split(",", 1)
        first = (rest.strip().split()[0] if rest.strip() else "")
        s = f"{first} {last}".strip()
    else:
        parts = s.split()
        if len(parts) >= 2:
            s = f"{parts[0]} {parts[-1]}"

    return " ".join(w[:1].upper() + w[1:].lower() if w else "" for w in s.split())


# ---------- Geoff-specific biweekly parser ----------
def load_clean_biweekly_table_geoff(csv_path: str) -> pd.DataFrame:
    def last_num(seq: List[str]) -> float:
        for c in reversed(seq):
            s = str(c).replace(",", "").replace("$", "").strip()
            if s.replace(".", "", 1).replace("-", "", 1).isdigit():
                try: return float(s)
                except: pass
        return 0.0

    def nums_in_row(row: List[str]) -> list:
        vals = []
        for c in row:
            s = str(c).replace(",", "").replace("$", "").strip()
            if s.replace(".", "", 1).replace("-", "", 1).isdigit():
                try: vals.append(float(s))
                except: pass
        return vals

    def next_num_after_token(row: List[str], token: str) -> float:
        token = token.lower().strip()
        for i, c in enumerate(row):
            if str(c).strip().lower() == token:
                for k in range(i + 1, len(row)):
                    s = str(row[k]).replace(",", "").replace("$", "").strip()
                    if s.replace(".", "", 1).replace("-", "", 1).isdigit():
                        return float(s)
                break
        return 0.0

    def is_emp_start(row: List[str]) -> bool:
        return ("Position:" in ",".join(map(str, row))) and ("," in str(row[0]))

    def is_boundary(row: List[str]) -> bool:
        if not row: return False
        first = str(row[0]).strip()
        j = " ".join(str(c).strip().lower() for c in row)
        return (
            first.upper() == "SALON TOTALS"
            or first.lower().startswith("salon")
            or "performance summary" in j
            or "payroll %" in j
            or "totals*" in j
        )

    def fm_money(x) -> str:
        try: return f"{float(str(x).replace('$','').replace(',','') or 0):.2f}"
        except: return "0.00"

    def fm_hours(x) -> str:
        try: return f"{float(x):.2f}"
        except: return "0.00"

    with open(csv_path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))

    blocks: List[Dict[str, Any]] = []
    cur_name, cur_block = None, []
    current_dept = ""

    def flush():
        nonlocal cur_name, cur_block
        if cur_name and cur_block:
            blocks.append({"name": cur_name, "rows": cur_block, "dept": current_dept})
        cur_name, cur_block = None, []

    for row in rows:
        if row and len(row) >= 2 and str(row[0]).strip().startswith("Payroll Detail Report - Biweekly"):
            m = _DEPT_HASH_RE.search(str(row[1]))
            if m:
                current_dept = m.group(1).strip()
            continue

        if is_emp_start(row):
            flush()
            cur_name = _canon_name(row[0])
            cur_block = [row]
            continue

        if cur_name:
            if is_boundary(row) or is_emp_start(row):
                flush()
                if is_emp_start(row):
                    cur_name = _canon_name(row[0])
                    cur_block = [row]
                continue
            cur_block.append(row)

    flush()

    out_rows: List[Dict[str, str]] = []
    RECEP_ALIASES = ("recept", "reception", "receptionist", "receptionists")

    for blk in blocks:
        name = blk["name"]; brws = blk["rows"]
        dept_code = str(blk.get("dept", "") or "").strip()

        pay_rate = "0.00"
        floor_h = close_h = breaks_h = admin_h = recep_h = training_h = 0.0
        overtime_h = 0.0
        bonus_d = comm_d = 0.0
        cash_tips_total = 0.0
        current_tips_value = "0.00"

        for r in brws:
            if "Base Wage:" in r:
                idx = r.index("Base Wage:")
                if idx + 1 < len(r):
                    pay_rate = r[idx + 1]
                break

        for r in brws:
            if (not r) or (not any(str(c).strip() for c in r)):
                continue

            j = " ".join(str(c).strip().lower() for c in r)

            if "floor" in j:
                v = next_num_after_token(r, "floor");       floor_h = v or floor_h
            if ("close" in j) or ("closing" in j):
                v = next_num_after_token(r, "close") or next_num_after_token(r, "closing"); close_h = v or close_h
            if "admin" in j:
                v = next_num_after_token(r, "admin");       admin_h = v or admin_h
            if "train" in j:
                v = next_num_after_token(r, "train")
                if v: training_h = v
            if any(alias in j for alias in RECEP_ALIASES):
                v = (next_num_after_token(r, "recept")
                     or next_num_after_token(r, "reception")
                     or next_num_after_token(r, "receptionist")
                     or next_num_after_token(r, "receptionists"))
                if v: recep_h = v or recep_h
            if "breaks paid" in j:
                v = next_num_after_token(r, "breaks paid"); breaks_h = v or breaks_h
            if "productivity" in j:
                bonus_d = last_num(r)
            if ("product sales" in j) or ("product commission" in j):
                comm_d = last_num(r)
            if any("OT Hrs" in str(c) for c in r):
                if any(str(c).strip().lower() == "ot" for c in r):
                    overtime_h = last_num(r)
                continue

            row_lc = [str(c).lower() for c in r]
            row_lc_join = " ".join(row_lc)

            if any("cash/check tips" in c for c in row_lc) and any(k in row_lc_join for k in ["floor hrs", "floor pay", "salon totals", "totals*"]):
                continue

            if any("cash/check tips" in c for c in row_lc):
                nums = [str(cell).replace(",", "").replace("$", "").strip() for cell in r]
                nums = [n for n in nums if re.match(r"^\d+(\.\d+)?$", n)]
                if nums:
                    val = float(nums[-1])
                    cash_tips_total += val

            if any("total tips" in c for c in row_lc):
                nums = [str(cell).replace(",", "").replace("$", "").strip() for cell in r[::-1]]
                nums = [n for n in nums if re.match(r"^\d+(\.\d+)?$", n)]
                if nums:
                    total_tips = float(nums[0])
                    credit_tips = total_tips - cash_tips_total
                    current_tips_value = f"{credit_tips:.2f}"
                else:
                    current_tips_value = "0.00"

        out_rows.append({
            "Employee": name,
            "Dept": dept_code,
            "Pay Rate": fm_money(pay_rate),
            "FLOOR (Earn Hrs)": fm_hours(floor_h),
            "CLOSING (Earn Hrs)": fm_hours(close_h),
            "BREAKS PAID (Earn Hrs)": fm_hours(breaks_h),
            "ADMIN (Earn Hrs)": fm_hours(admin_h),
            "TRAINING (Earn Hrs)": fm_hours(training_h),
            "OVERTIME (Earn Hrs)": fm_hours(overtime_h),
            "BONUS (Earn $)": fm_money(bonus_d),
            "COMMISSION (Earn $)": fm_money(comm_d),
            "CREDIT TIPS (Earn $)": current_tips_value,
            "RECEPTIONISTS (Earn Hrs)": fm_hours(recep_h),
        })

    df = pd.DataFrame(out_rows, columns=[
        "Employee","Dept","Pay Rate",
        "FLOOR (Earn Hrs)","CLOSING (Earn Hrs)","BREAKS PAID (Earn Hrs)","ADMIN (Earn Hrs)",
        "TRAINING (Earn Hrs)","OVERTIME (Earn Hrs)","BONUS (Earn $)","COMMISSION (Earn $)",
        "CREDIT TIPS (Earn $)","RECEPTIONISTS (Earn Hrs)"
    ]).fillna("0.00")

    df["Employee"] = df["Employee"].astype(str)
    df["Dept"] = df["Dept"].astype(str).str.strip()
    return df


def load_clean_biweekly_table_for_user(csv_path: str, username: str) -> pd.DataFrame:
    uname = (username or "").lower().strip()
    profile = (CLIENT_PROFILE_BY_USER.get(uname) or "default").lower().strip()
    if profile == "geoff":
        return load_clean_biweekly_table_geoff(csv_path)
    return load_clean_biweekly_table(csv_path)


# ---------- Heartland login + MFA ----------
async def _heartland_login(page: Page, hl_user: str, hl_pass: str, username: str) -> None:
    """
    Log into Heartland and complete MFA, ending on the Dashboard.
    Reused by employee Excel download and timecard upload.
    """
    print("🔐 Logging into Heartland...")
    await page.goto("https://www.heartlandpayroll.com", wait_until="load")

    await page.wait_for_selector('input[name="Email Address"]', timeout=60000)
    await page.fill('input[name="Email Address"]', hl_user)
    await page.fill('input[name="Password"]', hl_pass)
    await page.click('button[type="submit"]')
    print("✅ Submitted username and password; waiting for MFA screen...")

    await page.wait_for_selector("input", timeout=600000)
    mfa_input = page.locator("input").first

    # ── Signal portal that we are actively waiting for MFA ──────────────────
    # Both fields written so the correct message shows whether this is a
    # payroll run (portal checks payroll.state) or a key-sync
    # (portal checks readiness_status.substate).
    try:
        _get_users_collection().update_one(
            {"username": username},
            {"$set": {
                "payroll.state": "awaiting_mfa",
                "readiness_status.substate": "awaiting_mfa",
            }},
        )
    except Exception:
        pass

    mfa_code = await _wait_for_mfa_code(username, timeout_sec=600, poll_sec=2.0)

    # Clear the awaiting_mfa substate now that the code has been received
    try:
        _get_users_collection().update_one(
            {"username": username},
            {"$unset": {"readiness_status.substate": ""}},
        )
    except Exception:
        pass
    # ────────────────────────────────────────────────────────────────────────

    await mfa_input.click()
    await mfa_input.type(mfa_code, delay=100)

    try:
        await page.check('input[type="checkbox"]', timeout=3000)
    except Exception:
        pass

    await page.keyboard.press("Enter")

    try:
        await page.wait_for_load_state("networkidle", timeout=20000)
    except Exception:
        pass

    await _maybe_select_multi_account(page, username)
    await _maybe_select_multi_client(page, username)

    await page.wait_for_selector(r"text=/\b(?:Welcome|General)\b/i", timeout=300000)
    print("✅ Welcome page loaded.")


async def _maybe_select_multi_account(page: Page, username: str) -> None:
    uname = (username or "").lower().strip()
    cfg = HEARTLAND_MULTIACCOUNT_PICK.get(uname) or {}

    for _ in range(60):
        url = page.url or ""
        if "MultiAccountSelection" in url:
            break
        if "Dashboard" in url or "MultiClient" in url:
            return
        if "secure.globalpay.com" in url:
            await asyncio.sleep(0.5)
            continue

        try:
            if await page.locator("text=Select a Profile to Log Into").count() > 0:
                break
            if await page.locator("[id*='payroll-multiAccountSelection']").count() > 0:
                break
        except Exception:
            pass

        await asyncio.sleep(0.5)

    url = page.url or ""
    is_picker = False
    if "MultiAccountSelection" in url:
        is_picker = True
    else:
        try:
            is_picker = (await page.locator("text=Select a Profile to Log Into").count() > 0) or (await page.locator("[id*='payroll-multiAccountSelection']").count() > 0)
        except Exception:
            is_picker = False

    if not is_picker:
        return

    match = str(cfg.get("match") or "").strip()
    index = int(cfg.get("index", 1) or 1)

    print(f"🧭 MultiAccountSelection detected for {uname}. Selecting profile…")

    if match:
        try:
            row = page.locator("tr").filter(has_text=re.compile(re.escape(match), re.I)).first
            btn = row.locator("button:has-text('Select'), [id*='multiAccountSelection'][id*='select-btn'][id$='innerButton']").first
            if await btn.count() > 0:
                await btn.scroll_into_view_if_needed()
                await btn.click()
                try:
                    await page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    pass
                return
        except Exception as e:
            print(f"⚠️ MultiAccountSelection match-click failed ({match}): {e}. Falling back to index {index}.")

    try:
        btn = page.locator(f"#payroll-multiAccountSelection-payGroup-grid-select-btn-{index}-innerButton")
        await btn.click()
        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        return
    except Exception:
        pass

    try:
        btns = page.locator("button:has-text('Select')")
        cnt = await btns.count()
        if cnt == 0:
            return
        idx = min(max(index, 0), cnt - 1)
        await btns.nth(idx).scroll_into_view_if_needed()
        await btns.nth(idx).click()
        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
    except Exception as e:
        print(f"⚠️ MultiAccountSelection index-click failed: {e}")


async def _maybe_select_multi_client(page: Page, username: str) -> None:
    uname = (username or "").lower().strip()
    cfg = HEARTLAND_MULTICLIENT_PICK.get(uname) or {}
    if not cfg:
        return

    try:
        await page.wait_for_url("**/Dashboard/DashboardPartial/MultiClient**", timeout=8000)
    except Exception:
        return

    match = str(cfg.get("match") or "").strip()
    index = int(cfg.get("index") or 0)

    print(f"🧭 MultiClient picker detected for {uname}. Selecting client…")

    try:
        print("hi")
        btns = page.locator(f"#payroll-dashboard-multiClient-client-change-{index}")
        await btns.click()
        try:
            await page.wait_for_load_state("networkidle")
        except Exception:
            pass
    except Exception as e:
        print(f"⚠️ MultiClient index-click failed: {e}")


# ---------- Heartland "Employee id" custom report → Excel ----------
async def _open_employee_id_report_modal(page: Page, portal_username: str):
    await page.goto(
        "https://www.heartlandpayroll.com/Reports/CustomReports/CustomReportWriter",
        wait_until="domcontentloaded",
    )
    await page.wait_for_timeout(1200)

    uname = (portal_username or "").strip().lower()
    cfg = HEARTLAND_EMPLOYEEID_REPORT_PICK.get(uname) or {}
    match_text = (cfg.get("match") or "").strip() or "Employee id"
    pick_index = int(cfg.get("index", 0) or 0)
    handyman = _get_handyman_agent()

    try:
        if match_text:
            locator = page.locator(f"text={match_text}")
            if await locator.count() > 0:
                row = locator.first.locator("xpath=ancestor::tr[1]")
                if await row.count() == 0:
                    row = locator.first.locator("xpath=ancestor::div[1]")

                view_in_row = row.locator("fa-icon.view, i.fa-eye, i.fa.fa-eye, svg.fa-eye, [id*='view-action']")
                if await view_in_row.count() > 0:
                    await view_in_row.first.click()
                    return
    except Exception:
        pass

    icons = page.locator('fa-icon.view, i.fa-eye, i.fa.fa-eye, svg.fa-eye, [id*="view-action"]')
    try:
        n = await icons.count()
    except Exception:
        n = 0

    if n > 0:
        pick_index = min(max(pick_index, 0), n - 1)
        await icons.nth(pick_index).click()
        return

    if handyman:
        ok = await handyman.smart_click(
            page,
            task=f"On Heartland Custom Reports, click the view or eye icon for the report named {match_text}.",
            max_steps=12,
        )
        if ok:
            return

    raise RuntimeError('No view (eye) icons found for Heartland reports list.')


async def _download_employee_excel_from_heartland(page: Page, hl_user: str, hl_pass: str, username: str) -> str:
    async def _select_dropdown_report(label_text: str, option_text: str = None, option_index: int = 0):
        import re as _re
        field = page.locator("mat-form-field").filter(has_text=_re.compile(label_text, _re.I)).first
        overlay_options = page.locator(
            ".cdk-overlay-pane mat-option, .cdk-overlay-pane .mat-mdc-option, "
            ".cdk-overlay-container mat-option, .cdk-overlay-container .mat-mdc-option, [role='option']"
        )
        trigger = field.locator(".mat-mdc-select-trigger, .mat-select-trigger, mat-select").first
        opened = False
        for attempt in [lambda: trigger.click(force=True), lambda: field.click(force=True), lambda: trigger.press("Enter")]:
            try:
                await attempt()
                await page.wait_for_timeout(1200)
                if await overlay_options.count() > 0:
                    opened = True
                    break
            except Exception:
                pass
        if not opened:
            raise RuntimeError(f"Could not open dropdown for '{label_text}' on report modal.")
        if option_text:
            option = overlay_options.filter(has_text=_re.compile(_re.escape(option_text), _re.I)).first
            if await option.count() == 0:
                option = overlay_options.nth(option_index)
        else:
            option = overlay_options.nth(option_index)
        await option.wait_for(state="visible", timeout=20000)
        await option.scroll_into_view_if_needed()
        await option.click(force=True)
        await page.wait_for_timeout(800)

    await _heartland_login(page, hl_user, hl_pass, username)
    await _open_employee_id_report_modal(page, username)

    # Wait for the report modal form to fully render before touching dropdowns.
    # The modal can take anywhere from ~0.5s to 10s+ depending on Heartland load.
    print("⏳ Waiting for report modal to load…")
    try:
        await page.wait_for_selector("mat-form-field", state="visible", timeout=30000)
        await page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    await page.wait_for_timeout(800)   # small extra settle after networkidle

    try:
        print("🔽 Output Type: Excel")
        await _select_dropdown_report(r"Output\s*Type", "Excel")
    except Exception as e:
        print(f"⚠️ Failed to change Output Type to Excel: {e}. Assuming it is already Excel.")

    print("▶️ Running Employee id report…")
    excel_path = "heartland_employee_ids.xlsx"
    report_id_found: list[str] = []

    async def _intercept_checkstatus(response):
        if "checkstatus" in response.url.lower() and not report_id_found:
            try:
                data = await response.json()
                rid = str(data.get("reportId") or data.get("ReportId") or "").strip()
                if rid:
                    report_id_found.append(rid)
                    print(f"📋 Got reportId from checkstatus: {rid}")
            except Exception:
                pass  # first poll may return non-JSON — ignore

    async with page.context.expect_page() as popup_info:
        await page.get_by_role("button", name="Run Report").click()
    viewer = await popup_info.value
    viewer.on("response", _intercept_checkstatus)

    print("⏳ Waiting for checkstatus to return reportId (may take several minutes)…")
    for _ in range(150):   # up to 5 minutes
        await viewer.wait_for_timeout(2000)
        if report_id_found:
            break
    else:
        raise RuntimeError("checkstatus never returned a reportId after 5 minutes.")

    report_id = report_id_found[0]
    download_url = f"https://www.heartlandpayroll.com/Reports/Viewer/viewpdf?reportId={report_id}&download=true"
    print(f"⬇️ Fetching {download_url}")

    cookies = await page.context.cookies()
    cookie_jar = {c["name"]: c["value"] for c in cookies}

    import httpx
    async with httpx.AsyncClient(cookies=cookie_jar, timeout=120, follow_redirects=True) as client:
        r = await client.get(download_url)

    if r.status_code != 200 or len(r.content) < 512:
        raise RuntimeError(f"viewpdf returned status {r.status_code}, size {len(r.content)} — not a valid Excel file.")

    with open(excel_path, "wb") as f:
        f.write(r.content)
    print(f"✅ Downloaded Employee ID Excel → {excel_path} ({len(r.content)} bytes)")
    return excel_path



def _parse_employee_excel_to_df(excel_path: str) -> pd.DataFrame:
    try:
        df_raw = pd.read_excel(excel_path)
        print("Employee Excel columns:", list(df_raw.columns))

        if df_raw.empty:
            print("⚠️ Employee Excel is empty.")
            return pd.DataFrame(columns=["Key", "Employee", "Department"])

        lower_cols = [str(c).lower() for c in df_raw.columns]
        first_row = df_raw.iloc[0]

        if not any("number" in c for c in lower_cols) and any(
            "number" in str(v).lower() for v in first_row
        ):
            header_row = df_raw.iloc[0]
            df = df_raw.iloc[1:].copy()
            df.columns = header_row
        else:
            df = df_raw.copy()

        status_col = next(
            (c for c in df.columns if "status" in str(c).lower()),
            None,
        )
        if status_col is not None:
            df = df[
                df[status_col]
                .astype(str)
                .str.strip()
                .str.upper()
                .eq("A")
            ]

        emp_no_col = next(
            c for c in df.columns
            if "number" in str(c).lower()
            or ("employee" in str(c).lower() and "#" in str(c).lower())
        )
        first_col = next(c for c in df.columns if "first" in str(c).lower())
        last_col = next(c for c in df.columns if "last" in str(c).lower())
        dept_col = next(
            (c for c in df.columns
             if "department" in str(c).lower()
             or "dept" in str(c).lower()
             or "code" in str(c).lower()),
            None,
        )

        import pandas as _pd

        def _clean_num_series(s: pd.Series) -> pd.Series:
            nums = _pd.to_numeric(s, errors="coerce")
            return nums.map(lambda x: "" if _pd.isna(x) else str(int(x)))

        key_series = _clean_num_series(df[emp_no_col])

        if dept_col is not None:
            dept_raw = df[dept_col].astype(str)
            dept_digits = dept_raw.str.extract(r"(\d{3,6})", expand=False)
            dept_series = dept_digits.fillna("")
            empty_mask = dept_series.eq("")
            if empty_mask.any():
                dept_series.loc[empty_mask] = _clean_num_series(dept_raw[empty_mask])
        else:
            dept_series = pd.Series([""] * len(df))

        out = pd.DataFrame(
            {
                "Key": key_series.str.strip(),
                "Employee": (
                    df[first_col].fillna("").astype(str).str.strip()
                    + " "
                    + df[last_col].fillna("").astype(str).str.strip()
                ).str.strip(),
                "Department": dept_series.astype(str).str.strip(),
            }
        )

        out = out[out["Key"].str.len() > 0]
        out = out[out["Employee"].str.len() > 0]
        out = out.reset_index(drop=True)

        print("Parsed ACTIVE employee keys from Excel (with Department):")
        print(out.head())
        return out

    except Exception as e:
        print(f"❌ Error parsing Heartland employee Excel: {e}")
        return pd.DataFrame(columns=["Key", "Employee", "Department"])


# ---------- Heartland employee keys refresh (Mongo-based) ----------
def refresh_employee_keys_from_heartland(username: str) -> dict:
    try:
        hl_user, hl_pass = _get_vendor_creds(username, "heartland")

        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        async def _inner():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, slow_mo=50)
                context = await browser.new_context(accept_downloads=True)
                page = await context.new_page()
                excel_path = await _download_employee_excel_from_heartland(page, hl_user, hl_pass, username)
                await browser.close()
                return _parse_employee_excel_to_df(excel_path)

        df_excel = asyncio.run(_inner())
        if df_excel.empty:
            return {"ok": False, "error": "Employee Excel parse returned no rows.", "count": 0}

        keys_df = load_employee_keys_df(username)
        if keys_df.empty:
            keys_df = pd.DataFrame(columns=["Key", "Employee", "Department"])
        if "Department" not in keys_df.columns:
            keys_df["Department"] = ""

        df_excel["EmployeeNorm"] = df_excel["Employee"].map(_name_norm)
        keys_df["EmployeeNorm"] = keys_df["Employee"].map(_name_norm)

        merge_cols = ["EmployeeNorm", "Key", "Employee", "Department"]
        merged = keys_df.merge(
            df_excel[merge_cols],
            on="EmployeeNorm",
            how="outer",
            suffixes=("_old", "_new"),
        )

        def _pick_key(row):
            new = str(row.get("Key_new") or "").strip()
            old = str(row.get("Key_old") or "").strip()
            return new if new else old

        def _pick_emp(row):
            new = str(row.get("Employee_new") or "").strip()
            old = str(row.get("Employee_old") or "").strip()
            return new if new else old

        def _pick_dept(row):
            new = str(row.get("Department_new") or "").strip()
            old = str(row.get("Department_old") or "").strip()
            return new if new else old

        updated_keys = pd.DataFrame(
            {
                "Key": merged.apply(_pick_key, axis=1),
                "Employee": merged.apply(_pick_emp, axis=1),
                "Department": merged.apply(_pick_dept, axis=1),
            }
        )

        updated_keys["Employee"] = updated_keys["Employee"].astype(str).str.strip()
        updated_keys["Key"] = updated_keys["Key"].astype(str).str.strip()
        updated_keys["Department"] = updated_keys["Department"].astype(str).str.strip()

        save_employee_keys_df(username, updated_keys, source="heartland_sync")
        print(f"🔄 Keys refreshed in Mongo for {username} with {len(updated_keys)} employees.")
        return {"ok": True, "error": None, "count": len(updated_keys)}

    except Exception as e:
        print("❌ Error refreshing employee keys from Heartland (Excel):", repr(e))
        traceback.print_exc()
        return {"ok": False, "error": _friendly_error_message(e), "count": 0}


# ---------- PDF combined report ----------
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, LongTable, PageBreak
from reportlab.lib.styles import getSampleStyleSheet


def _name_norm(s: str) -> str:
    s = (s or "").strip()
    m = re.match(r"^([^,]+),\s*(.+)$", s)
    if m:
        last, first = m.group(1), m.group(2).split()[0]
        s = f"{first} {last}"
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def _clean_dept(v) -> str:
    return re.sub(r"\D", "", str(v)) or ""


def compute_cross_department_frames_from_df(df: pd.DataFrame, keys_csv_path: str):
    df = df.copy()
    df["Employee"] = df["Employee"].astype(str).map(_name_norm)
    if "Dept" not in df.columns:
        df["Dept"] = ""
    df["Dept"] = df["Dept"].map(_clean_dept)

    for c in ["Pay Rate", "E_Regular_Hours", "E_Overtime_Hours", "Productivity", "Prod Bonus", "Tips"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0)
    df["Regular Hrs"] = df["E_Regular_Hours"]
    df["OT Hrs"] = df["E_Overtime_Hours"]
    df["Total Hrs"] = df["Regular Hrs"] + df["OT Hrs"]

    df["Prod Bonus $"] = pd.to_numeric(df.get("Productivity", 0), errors="coerce").fillna(0.0)
    df["Prod Comm $"] = pd.to_numeric(df.get("Prod Bonus", 0), errors="coerce").fillna(0.0)
    df["Total Bonus $"] = df["Prod Bonus $"] + df["Prod Comm $"]
    df["Tips $"] = pd.to_numeric(df.get("Tips", 0), errors="coerce").fillna(0.0)

    maybe_hours: List[str] = []
    for raw, nice in [
        ("FLOOR (Earn Hrs)", "Floor"),
        ("ADMIN (Earn Hrs)", "Admin"),
        ("CLOSING (Earn Hrs)", "Close"),
        ("RECEPTIONISTS (Earn Hrs)", "Recept"),
        ("TRAINING (Earn Hrs)", "Train"),
    ]:
        if raw in df.columns:
            df[nice] = pd.to_numeric(df[raw], errors="coerce").fillna(0.0)
            maybe_hours.append(nice)

    username = (keys_csv_path or "").strip().lower()
    keys = load_employee_keys_df(username).copy()

    for col in ["Employee", "Key", "Department"]:
        if col not in keys.columns:
            keys[col] = ""
    keys[col] = keys[col].fillna("").astype(str).str.strip()
    keys[col] = keys[col].mask(keys[col].str.lower().isin(["nan", "none", "<na>"]), "")

    keys["Employee"] = keys["Employee"].astype(str).map(_name_norm)
    keys["Home"] = keys.get("Department", "").astype(str).map(_clean_dept)

    merged = df.merge(keys[["Employee", "Home"]], on="Employee", how="left")
    merged["Worked"] = merged["Dept"].map(_clean_dept)
    merged["Home"] = merged["Home"].map(_clean_dept)

    cross = merged[
        (merged["Worked"] != "")
        & (merged["Home"] != "")
        & (merged["Worked"] != merged["Home"])
    ].copy()

    disp_cols = ["Employee", "Worked", "Home"] + maybe_hours + [
        "Regular Hrs", "OT Hrs", "Total Hrs",
        "Prod Bonus $", "Prod Comm $", "Total Bonus $", "Tips $",
    ]
    if cross.empty:
        empty = pd.DataFrame(columns=disp_cols)
        return empty, empty, "No cross-department entries found"

    add_df = cross[disp_cols].copy()
    sub_df = cross[disp_cols].copy()

    for c in [c for c in disp_cols if c not in ("Employee", "Worked", "Home")]:
        add_df[c] = pd.to_numeric(add_df[c], errors="coerce").fillna(0.0)
        sub_df[c] = -pd.to_numeric(sub_df[c], errors="coerce").fillna(0.0)

    for df_ in (add_df, sub_df):
        for c in [c for c in disp_cols if c not in ("Employee", "Worked", "Home")]:
            df_[c] = df_[c].map(lambda x: f"{float(x):.2f}")

    return add_df, sub_df, None


def build_combined_pdf(input_csv_path: str, keys_csv_path: str, output_pdf_path: str) -> str:
    df = load_clean_biweekly_table(input_csv_path).copy()
    df["Dept"] = df.get("Dept", "").astype(str).str.replace(".0", "", regex=False).str.strip()

    for c in ["Pay Rate", "E_Regular_Hours", "E_Overtime_Hours", "Productivity", "Prod Bonus", "Tips"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0)

    df["Regular Hrs"] = df["E_Regular_Hours"]
    df["OT Hrs"] = df["E_Overtime_Hours"]
    df["Total Hrs"] = df["Regular Hrs"] + df["OT Hrs"]
    df["Prod Bonus $"] = pd.to_numeric(df.get("Productivity", 0), errors="coerce").fillna(0.0)
    df["Prod Comm $"] = pd.to_numeric(df.get("Prod Bonus", 0), errors="coerce").fillna(0.0)
    df["Total Bonus $"] = df["Prod Bonus $"] + df["Prod Comm $"]
    df["Tips $"] = pd.to_numeric(df.get("Tips", 0), errors="coerce").fillna(0.0)

    hours_agg = df.groupby(["Dept"], dropna=False)[["Regular Hrs", "OT Hrs", "Total Hrs"]].sum().reset_index()
    hours_agg.insert(0, "SalonName", "")
    hours_tot = {"SalonName": "ALL SALONS", "Dept": "", "Regular Hrs": hours_agg["Regular Hrs"].sum(), "OT Hrs": hours_agg["OT Hrs"].sum(), "Total Hrs": hours_agg["Total Hrs"].sum()}
    hours_agg = pd.concat([hours_agg, pd.DataFrame([hours_tot])], ignore_index=True)

    money_agg = df.groupby(["Dept"], dropna=False)[["Prod Bonus $", "Prod Comm $", "Total Bonus $", "Tips $"]].sum().reset_index()
    money_agg.insert(0, "SalonName", "")
    money_tot = {"SalonName": "ALL SALONS", "Dept": "", "Prod Bonus $": money_agg["Prod Bonus $"].sum(), "Prod Comm $": money_agg["Prod Comm $"].sum(), "Total Bonus $": money_agg["Total Bonus $"].sum(), "Tips $": money_agg["Tips $"].sum()}
    money_agg = pd.concat([money_agg, pd.DataFrame([money_tot])], ignore_index=True)

    add_df, sub_df, cross_msg = compute_cross_department_frames_from_df(df, keys_csv_path)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_pdf_path, pagesize=landscape(LETTER), leftMargin=18, rightMargin=18, topMargin=20, bottomMargin=20)

    def _fmt2(x):
        try: return f"{float(x):.2f}"
        except: return x

    for df_, cols in [(hours_agg, ["Regular Hrs", "OT Hrs", "Total Hrs"]), (money_agg, ["Prod Bonus $", "Prod Comm $", "Total Bonus $", "Tips $"])]:
        for c in cols:
            if c in df_.columns:
                df_[c] = df_[c].map(_fmt2)

    def _longtable(df_in, title):
        df_local = df_in.copy()
        if df_local.empty:
            df_local = pd.DataFrame(columns=["No data"])
        data = [list(df_local.columns)] + df_local.astype(str).values.tolist()
        total_width = landscape(LETTER)[0] - doc.leftMargin - doc.rightMargin
        sample = df_local.head(25).astype(str)
        max_lens = []
        for c in df_local.columns:
            try: mx = int(sample[c].map(len).max()) if not sample.empty else 0
            except: mx = 0
            mx = max(mx, len(str(c)))
            max_lens.append(mx)
        denom = sum(max_lens) or 1
        raw_widths = [max(42.0, total_width * (m / denom)) for m in max_lens]
        scale = total_width / sum(raw_widths) if sum(raw_widths) else 1.0
        col_widths = [w * scale for w in raw_widths]
        tbl = LongTable(data, repeatRows=1, colWidths=col_widths)
        tbl.hAlign = "LEFT"
        tbl.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
            ("FONT", (0, 1), (-1, -1), "Helvetica", 7),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("ALIGN", (0, 1), (0, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ]))
        for r in range(1, len(data)):
            if r % 2 == 0:
                tbl.setStyle(TableStyle([("BACKGROUND", (0, r), (-1, r), colors.Color(0.97, 0.97, 0.97))]))
        return [Paragraph(f"<b>{title}</b>", styles["Heading3"]), Spacer(1, 6), tbl, Spacer(1, 12)]

    story = [
        Paragraph("<b>Payroll Combined Report</b>", styles["Title"]),
        Spacer(1, 8),
        *_longtable(hours_agg, "Salon Totals — Hours"),
        *_longtable(money_agg, "Salon Totals — Bonuses & Tips"),
        PageBreak(),
        *_longtable(add_df, "ADD (credit worked department) — Hours, Bonuses & Tips"),
        *_longtable(sub_df, "SUBTRACT (debit home department) — Hours, Bonuses & Tips"),
    ]
    if cross_msg:
        story.insert(2, Paragraph(f"<i>Note:</i> {cross_msg}", styles["Normal"]))

    doc.build(story)
    print(f"📄 Wrote combined-style PDF → {output_pdf_path}")
    return output_pdf_path


def build_combined_pdf_geoff(input_csv_path: str, keys_csv_path: str, output_pdf_path: str) -> str:
    df = load_clean_biweekly_table_geoff(input_csv_path).copy()
    if df.empty:
        return build_combined_pdf(input_csv_path, keys_csv_path, output_pdf_path)

    df["Dept"] = df.get("Dept", "").astype(str).str.replace(".0", "", regex=False).str.strip()

    hours_cols = [c for c in ["FLOOR (Earn Hrs)","CLOSING (Earn Hrs)","BREAKS PAID (Earn Hrs)","ADMIN (Earn Hrs)","TRAINING (Earn Hrs)","RECEPTIONISTS (Earn Hrs)","OVERTIME (Earn Hrs)"] if c in df.columns]
    money_cols = [c for c in ["BONUS (Earn $)","COMMISSION (Earn $)","CREDIT TIPS (Earn $)"] if c in df.columns]

    for c in hours_cols + money_cols:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0)

    hours_agg = df.groupby(["Dept"], dropna=False)[hours_cols].sum().reset_index() if hours_cols else pd.DataFrame(columns=["Dept"])
    hours_agg.insert(0, "SalonName", "")
    if hours_cols and not hours_agg.empty:
        hours_tot = {"SalonName": "ALL SALONS", "Dept": ""}
        for c in hours_cols: hours_tot[c] = float(hours_agg[c].sum())
        hours_agg = pd.concat([hours_agg, pd.DataFrame([hours_tot])], ignore_index=True)

    money_agg = df.groupby(["Dept"], dropna=False)[money_cols].sum().reset_index() if money_cols else pd.DataFrame(columns=["Dept"])
    money_agg.insert(0, "SalonName", "")
    if money_cols and not money_agg.empty:
        money_tot = {"SalonName": "ALL SALONS", "Dept": ""}
        for c in money_cols: money_tot[c] = float(money_agg[c].sum())
        money_agg = pd.concat([money_agg, pd.DataFrame([money_tot])], ignore_index=True)

    df_std = df.copy()
    def _colnum(series_name: str) -> pd.Series:
        return pd.to_numeric(df_std.get(series_name, 0), errors="coerce").fillna(0.0)

    reg = (_colnum("FLOOR (Earn Hrs)") + _colnum("CLOSING (Earn Hrs)") + _colnum("BREAKS PAID (Earn Hrs)") + _colnum("ADMIN (Earn Hrs)") + _colnum("TRAINING (Earn Hrs)") + _colnum("RECEPTIONISTS (Earn Hrs)"))
    df_std["E_Regular_Hours"] = reg
    df_std["E_Overtime_Hours"] = _colnum("OVERTIME (Earn Hrs)")
    df_std["Productivity"] = _colnum("BONUS (Earn $)")
    df_std["Prod Bonus"] = _colnum("COMMISSION (Earn $)")
    df_std["Tips"] = _colnum("CREDIT TIPS (Earn $)")
    df_std["Pay Rate"] = _colnum("Pay Rate")

    add_df, sub_df, cross_msg = compute_cross_department_frames_from_df(df_std, keys_csv_path)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_pdf_path, pagesize=landscape(LETTER), leftMargin=18, rightMargin=18, topMargin=20, bottomMargin=20)

    def _fmt2(x):
        try: return f"{float(x):.2f}"
        except: return x

    for df_, cols in [(hours_agg, hours_cols), (money_agg, money_cols)]:
        for c in cols:
            if c in df_.columns:
                df_[c] = df_[c].map(_fmt2)

    def _longtable(df_in, title):
        df_local = df_in.copy()
        if df_local.empty:
            df_local = pd.DataFrame(columns=["No data"])
        data = [list(df_local.columns)] + df_local.astype(str).values.tolist()
        total_width = landscape(LETTER)[0] - doc.leftMargin - doc.rightMargin
        sample = df_local.head(25).astype(str)
        max_lens = []
        for c in df_local.columns:
            try: mx = int(sample[c].map(len).max()) if not sample.empty else 0
            except: mx = 0
            mx = max(mx, len(str(c)))
            max_lens.append(mx)
        denom = sum(max_lens) or 1
        raw_widths = [max(42.0, total_width * (m / denom)) for m in max_lens]
        scale = total_width / sum(raw_widths) if sum(raw_widths) else 1.0
        col_widths = [w * scale for w in raw_widths]
        tbl = LongTable(data, repeatRows=1, colWidths=col_widths)
        tbl.hAlign = "LEFT"
        tbl.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
            ("FONT", (0, 1), (-1, -1), "Helvetica", 7),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("ALIGN", (0, 1), (0, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ]))
        for r in range(1, len(data)):
            if r % 2 == 0:
                tbl.setStyle(TableStyle([("BACKGROUND", (0, r), (-1, r), colors.Color(0.97, 0.97, 0.97))]))
        return [Paragraph(f"<b>{title}</b>", styles["Heading3"]), Spacer(1, 6), tbl, Spacer(1, 12)]

    story = [
        Paragraph("<b>Payroll Combined Report</b>", styles["Title"]),
        Spacer(1, 8),
        *_longtable(hours_agg, "Salon Totals — Hours"),
        *_longtable(money_agg, "Salon Totals — Bonuses & Tips"),
        PageBreak(),
        *_longtable(add_df, "ADD (credit worked department) — Hours, Bonuses & Tips"),
        *_longtable(sub_df, "SUBTRACT (debit home department) — Hours, Bonuses & Tips"),
    ]
    if cross_msg:
        story.insert(2, Paragraph(f"<i>Note:</i> {cross_msg}", styles["Normal"]))

    doc.build(story)
    print(f"📄 Wrote combined-style PDF → {output_pdf_path}")
    return output_pdf_path


def build_combined_pdf_for_user(input_csv_path: str, keys_csv_path: str, output_pdf_path: str) -> str:
    uname = (keys_csv_path or "").lower().strip()
    profile = (CLIENT_PROFILE_BY_USER.get(uname) or "default").lower().strip()
    if profile == "geoff":
        return build_combined_pdf_geoff(input_csv_path, keys_csv_path, output_pdf_path)
    return build_combined_pdf(input_csv_path, keys_csv_path, output_pdf_path)


# ---------- Formatting CSV for Heartland ----------
from typing import Optional
import pandas as pd

def _clean_keys_df(keys_df: pd.DataFrame) -> pd.DataFrame:
    keys_df = keys_df.copy()
    for col in ["Employee", "Key", "Department"]:
        if col not in keys_df.columns:
            keys_df[col] = ""
        keys_df[col] = keys_df[col].fillna("").astype(str).str.strip()
        keys_df[col] = keys_df[col].mask(keys_df[col].str.lower().isin(["nan", "none", "<na>"]), "")
    return keys_df

import re

def _emp_key(name: str) -> str:
    s = (name or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-zA-Z0-9 ]+", "", s)
    return s.lower().strip()


def format_csv_for_heartland(input_csv_path: str, key_file_path: str, output_path: str = "Cleaned_Heartland_Ready_Payroll.csv") -> Optional[str]:
    try:
        username = (key_file_path or "").strip().lower()
        payroll_df = load_clean_biweekly_table_for_user(input_csv_path, username)
        print("\n✅ Parsed Biweekly Table:")
        print(payroll_df)

        keys_df = load_employee_keys_df(username)
        keys_df = _clean_keys_df(keys_df)

        payroll_df["EmployeeKey"] = payroll_df["Employee"].astype(str).map(_emp_key)
        keys_df["EmployeeKey"] = keys_df["Employee"].astype(str).map(_emp_key)

        merged = payroll_df.merge(keys_df.drop(columns=["Employee"], errors="ignore"), on="EmployeeKey", how="left", indicator=True)

        missing = merged[merged["_merge"] == "left_only"].copy()
        if not missing.empty:
            missing["Employee"] = missing["Employee"].fillna("").astype(str).str.strip()
            missing = missing[~missing["Employee"].str.lower().isin(["", "nan", "none", "<na>"])]
            if not missing.empty:
                print("\n❌ Missing Key for these employees:")
                print(missing[["Employee"]])

        final_df = merged.rename(columns={
            "Dept": "LaborValue2",
            "Productivity": "E_Productivity Bonus_Dollars",
            "Prod Bonus": "E_Product Comm._Dollars",
            "Tips": "E_Pd- Credit Tips_Dollars",
        })

        keep_cols = ["Key", "Employee", "E_Regular_Hours", "E_Overtime_Hours", "E_Productivity Bonus_Dollars", "E_Product Comm._Dollars", "E_Pd- Credit Tips_Dollars", "LaborValue2"]
        for c in keep_cols:
            if c not in final_df.columns:
                final_df[c] = ""
        final_df = final_df[keep_cols].copy()

        final_df["Key"] = final_df["Key"].fillna("").astype(str).str.strip()
        final_df["Key"] = final_df["Key"].mask(final_df["Key"].str.lower().isin(["nan", "none", "<na>"]), "")
        final_df = final_df[final_df["Key"].ne("")].copy()
        final_df["Key"] = pd.to_numeric(final_df["Key"], errors="coerce").astype("Int64").astype(str)
        final_df = final_df[~final_df["Key"].str.lower().isin(["<na>", "nan", "none"])].copy()

        print("\n✅ Final Heartland Upload Table:")
        print(final_df)
        final_df.to_csv(output_path, index=False)
        return output_path

    except Exception as e:
        print(f"\n❌ Error during formatting: {e}")
        return None


def format_csv_for_heartland_geoff(input_csv_path: str, username: str, output_path: str = "Cleaned_Heartland_Ready_Payroll.csv"):
    import re
    from pathlib import Path
    import pandas as pd

    def _canon_name(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        if "," in s:
            last, rest = s.split(",", 1)
            first = (rest.strip().split()[0] if rest.strip() else "")
            s = f"{first} {last.strip()}".strip()
        else:
            parts = s.split()
            if len(parts) >= 2:
                s = f"{parts[0]} {parts[-1]}"
        return s.title()

    def _fmt2(x) -> str:
        try: return f"{float(str(x).replace(',', '').replace('$','').strip() or 0):.2f}"
        except: return "0.00"

    uname = (username or "").strip().lower()
    payroll_df = load_clean_biweekly_table_for_user(input_csv_path, uname)
    if payroll_df is None:
        return None
    payroll_df = payroll_df.copy()

    if "Employee" not in payroll_df.columns:
        payroll_df["Employee"] = ""
    payroll_df["Employee"] = payroll_df["Employee"].astype(str).map(_canon_name)

    keys_df = load_employee_keys_df(uname)
    keys_df = _clean_keys_df(keys_df)
    keys_df["Employee"] = keys_df["Employee"].astype(str).map(_canon_name)
    key_map = dict(zip(keys_df["Employee"], keys_df["Key"].astype(str)))

    out_cols = ["Key","Employee","Pay Rate","E_Floor_Hours","E_Closing_Hours","E_Breaks Paid_Hours","E_Admin_Hours","E_Overtime_Dollars","E_Bonus_Dollars","E_Commission_Dollars","E_Credit Tips_Dollars","E_Receptionists_Hours","LaborValue2"]
    out = pd.DataFrame(index=payroll_df.index, columns=out_cols)

    out["Employee"] = payroll_df["Employee"].astype(str)
    out["Pay Rate"] = payroll_df.get("Pay Rate", "0.00").astype(str).map(_fmt2)
    out["Key"] = out["Employee"].map(lambda n: str(key_map.get(n, "")).strip())

    def _col_or_zero(colname: str):
        if colname in payroll_df.columns:
            return payroll_df[colname].astype(str).map(_fmt2)
        return pd.Series(["0.00"] * len(payroll_df), index=payroll_df.index)

    out["E_Floor_Hours"] = _col_or_zero("FLOOR (Earn Hrs)")
    out["E_Closing_Hours"] = _col_or_zero("CLOSING (Earn Hrs)")

    if "BREAKS PAID (Earn Hrs)" in payroll_df.columns:
        out["E_Breaks Paid_Hours"] = _col_or_zero("BREAKS PAID (Earn Hrs)")
    elif "TRAINING (Earn Hrs)" in payroll_df.columns:
        out["E_Breaks Paid_Hours"] = _col_or_zero("TRAINING (Earn Hrs)")
    else:
        out["E_Breaks Paid_Hours"] = "0.00"

    out["E_Admin_Hours"] = _col_or_zero("ADMIN (Earn Hrs)")
    out["E_Overtime_Dollars"] = _col_or_zero("OVERTIME (Earn Hrs)")
    out["E_Bonus_Dollars"] = _col_or_zero("BONUS (Earn $)")
    out["E_Commission_Dollars"] = _col_or_zero("COMMISSION (Earn $)")
    out["E_Credit Tips_Dollars"] = _col_or_zero("CREDIT TIPS (Earn $)")
    out["E_Receptionists_Hours"] = _col_or_zero("RECEPTIONISTS (Earn Hrs)")

    dept_src = "Dept" if "Dept" in payroll_df.columns else ("Department" if "Department" in payroll_df.columns else None)
    out["LaborValue2"] = payroll_df[dept_src].astype(str).str.strip() if dept_src else ""

    key_str = out["Key"].fillna("").astype(str).str.strip()
    missing = out[key_str.eq("") | key_str.str.lower().isin(["nan","none","<na>"])].copy()
    if not missing.empty:
        print("\n=== MISSING KEYS ===")
        for name in missing["Employee"].astype(str).tolist():
            if name.strip():
                print(f"- {name}: missing key")
        print("====================\n")

    out["Key"] = out["Key"].fillna("").astype(str).str.strip()
    out = out[~out["Key"].str.lower().isin(["", "nan", "none", "<na>"])].copy()
    out["Key"] = pd.to_numeric(out["Key"], errors="coerce").astype("Int64").astype(str)
    out = out[~out["Key"].str.lower().isin(["<na>", "nan", "none"])].copy()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Done → {output_path} | Rows: {len(out)}")
    print(out)
    return output_path


def format_csv_for_heartland_for_user(csv_path: str, username: str) -> pd.DataFrame:
    uname = (username or "").lower().strip()
    profile = (CLIENT_PROFILE_BY_USER.get(uname) or "default").lower().strip()
    if profile == "geoff":
        return format_csv_for_heartland_geoff(csv_path, username, "Cleaned_Heartland_Ready_Payroll.csv")
    return format_csv_for_heartland(csv_path, username, "Cleaned_Heartland_Ready_Payroll.csv")


# ---------- Heartland upload ----------
async def upload_to_heartland(page: Page, file_path: str, hl_user: str, hl_pass: str, username: str) -> None:
    import re

    handyman = _get_handyman_agent()

    async def select_dropdown(label_text: str, option_text: str = None, option_index: int = 0):
        field = page.locator("mat-form-field").filter(has_text=re.compile(label_text, re.I)).first
        await field.wait_for(state="visible", timeout=60000)
        trigger = field.locator(".mat-mdc-select-trigger, .mat-select-trigger, mat-select").first
        await trigger.scroll_into_view_if_needed()
        await page.wait_for_timeout(500)
        overlay_options = page.locator(".cdk-overlay-pane mat-option, .cdk-overlay-pane .mat-mdc-option, .cdk-overlay-container mat-option, .cdk-overlay-container .mat-mdc-option, [role='option']")
        opened = False
        open_attempts = [lambda: trigger.click(force=True), lambda: field.click(force=True), lambda: trigger.press("Enter"), lambda: trigger.press(" ")]
        last_err = None
        for attempt in open_attempts:
            try:
                await attempt()
                await page.wait_for_timeout(1200)
                if await overlay_options.count() > 0:
                    opened = True
                    break
            except Exception as e:
                last_err = e

        if not opened and handyman:
            opened = await handyman.smart_click(
                page,
                task=f"Open the dropdown labeled {label_text} on the Heartland Time Card Import page.",
                max_steps=8,
            )
            if opened:
                await page.wait_for_timeout(1200)

        if not opened:
            raise RuntimeError(f"Could not open dropdown for '{label_text}'. URL={page.url}. Last error={last_err}")

        if option_text:
            option = overlay_options.filter(has_text=re.compile(rf"^\s*{re.escape(option_text)}\s*$", re.I)).first
            if await option.count() == 0:
                option = overlay_options.filter(has_text=re.compile(re.escape(option_text), re.I)).first
            if await option.count() == 0:
                option = page.locator(".cdk-overlay-pane mat-option:not([aria-disabled='true']), .cdk-overlay-pane .mat-mdc-option[aria-disabled='false'], [role='option']").nth(option_index)
        else:
            option = page.locator(".cdk-overlay-pane mat-option:not([aria-disabled='true']), .cdk-overlay-pane .mat-mdc-option[aria-disabled='false'], [role='option']").nth(option_index)

        try:
            await option.wait_for(state="visible", timeout=20000)
            await option.scroll_into_view_if_needed()
            await option.click(force=True)
            await page.wait_for_timeout(800)
            return
        except Exception:
            if handyman and option_text:
                ok = await handyman.smart_click(
                    page,
                    task=f"Choose the option {option_text} from the currently open dropdown on Heartland.",
                    max_steps=8,
                )
                if ok:
                    await page.wait_for_timeout(800)
                    return
            raise

    await _heartland_login(page, hl_user, hl_pass, username)

    await page.goto("https://www.heartlandpayroll.com/Payroll/PayrollTimeCardImport/NewTimeCardImport", wait_until="domcontentloaded")
    try:
        await page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    await page.wait_for_timeout(2000)

    try:
        await page.wait_for_selector("text=New Time Card Import", timeout=60000)
    except Exception:
        await page.wait_for_selector("text=Time Card Import", timeout=60000)

    await page.wait_for_selector("mat-form-field", timeout=60000)
    print("✅ Time Card Import page loaded.")

    try:
        print("🔽 Import Type: Timecard")
        await select_dropdown(r"Import\s*Type", "Timecard")
        print("🔽 Template: Default")
        await select_dropdown(r"Template", "Default")
        print("🔽 File Format: Comma Delimited File")
        await select_dropdown(r"File\s*Format", "Comma Delimited File")
        print("🔽 Default Pay Group")
        await select_dropdown(r"Default\s*Pay\s*Group", option_text=None, option_index=0)
        print("🔽 Import Key")
        await select_dropdown(r"Import\s*Key", option_text=None, option_index=0)
        print("📁 Uploading CSV file...")
        await page.set_input_files('input[type="file"]', file_path)
        await page.wait_for_timeout(2500)
        print("📝 Submitting form with Validate...")
        try:
            await page.get_by_role("button", name=re.compile(r"^Validate$", re.I)).click()
        except Exception:
            if handyman:
                ok = await handyman.smart_click(page, task="Click the Validate button on the Heartland Time Card Import page.")
                if not ok:
                    raise
            else:
                raise
        print("⏳ Waiting for Import button...")
        import_btn = page.get_by_role("button", name=re.compile(r"^Import$", re.I)).first
        try:
            await import_btn.wait_for(state="visible", timeout=60000)
            print("🚀 Clicking Import...")
            await import_btn.click()
        except Exception:
            if handyman:
                ok = await handyman.smart_click(page, task="Click the Import button on the Heartland Time Card Import page.")
                if not ok:
                    raise
            else:
                raise
        print("⏳ Waiting for confirmation of successful import...")
        success_msg = page.get_by_text("File Successfully Imported!", exact=False).first
        await success_msg.wait_for(state="visible", timeout=60000)
        print("✅ File import confirmed as successful.")
    except Exception as e:
        raise RuntimeError(f"Time Card Import failed: {e}")


# ---------- Readiness check ----------
def check_payroll_ready_for_user(username: str, dry_run: bool = False, period_end_date=None) -> dict:
    try:
        sd_user, sd_pass = _get_vendor_creds(username, "salondata")

        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        async def _inner_salondata():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, slow_mo=50)
                context = await browser.new_context(accept_downloads=True)
                page = await context.new_page()
                csv_path, _ = await download_salondata_csv(page, sd_user, sd_pass, period_end_date)
                await browser.close()
                return csv_path

        csv_path = asyncio.run(_inner_salondata())

        def _missing_from_keys() -> List[str]:
            payroll_df = load_clean_biweekly_table_for_user(csv_path, username)
            keys_df = load_employee_keys_df(username)
            payroll_df["Employee"] = payroll_df["Employee"].fillna("").astype(str).map(_name_norm)
            keys_df["Employee"] = keys_df["Employee"].fillna("").astype(str).map(_name_norm)
            payroll_df = payroll_df[payroll_df["Employee"].astype(str).str.strip() != ""]
            keys_df = keys_df[keys_df["Employee"].astype(str).str.strip() != ""]
            merged = payroll_df.merge(keys_df, on="Employee", how="left", indicator=True)
            key_series = merged.get("Key")
            key_str = key_series.fillna("").astype(str).str.strip()
            key_empty = key_str.eq("") | key_str.str.lower().isin(["nan","none"])
            missing_mask = (merged["_merge"] == "left_only") | key_empty
            missing_df = merged[missing_mask]
            return sorted(missing_df["Employee"].dropna().unique().tolist())

        missing_names = _missing_from_keys()

        if not missing_names:
            result = {"ready": True, "csv_path": csv_path, "missing_keys": [], "error": None, "needs_sync": False}
            print("check_payroll_ready_for_user (dry_run =", dry_run, ") →", result)
            return result

        if dry_run:
            onboard_msg = "Missing Heartland employee Keys for: " + ", ".join(missing_names)
            result = {"ready": False, "csv_path": csv_path, "missing_keys": missing_names, "error": onboard_msg, "needs_sync": True}
            print("check_payroll_ready_for_user (dry_run =", dry_run, ") →", result)
            return result

        sync = refresh_employee_keys_from_heartland(username)
        if not sync.get("ok"):
            result = {"ready": False, "csv_path": csv_path, "missing_keys": missing_names, "error": sync.get("error") or "Heartland sync failed.", "needs_sync": True}
            print("check_payroll_ready_for_user (dry_run =", dry_run, ") →", result)
            return result

        missing_after = _missing_from_keys()
        if missing_after:
            onboard_msg = "Still missing Heartland Keys for: " + ", ".join(missing_after)
            result = {"ready": False, "csv_path": csv_path, "missing_keys": missing_after, "error": onboard_msg, "needs_sync": True}
            print("check_payroll_ready_for_user (dry_run =", dry_run, ") →", result)
            return result

        result = {"ready": True, "csv_path": csv_path, "missing_keys": [], "error": None, "needs_sync": False}
        print("check_payroll_ready_for_user (dry_run =", dry_run, ") →", result)
        return result

    except Exception as e:
        print("check_payroll_ready_for_user ERROR →", repr(e))
        traceback.print_exc()
        return {"ready": False, "csv_path": None, "missing_keys": [], "error": _friendly_error_message(e), "needs_sync": False}


# ---------- Orchestration for Streamlit ----------
async def _full_agentic_flow_inner(sd_user, sd_pass, hl_user, hl_pass, username, period_end_date=None, csv_path_prefetched=None) -> dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, slow_mo=50)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        if csv_path_prefetched and os.path.exists(csv_path_prefetched):
            print(f"♻️ Reusing prefetched SalonData CSV → {csv_path_prefetched}")
            csv_path = csv_path_prefetched
            # Derive period_end from the date we already know
            d = _coerce_date(period_end_date) if period_end_date else _default_payroll_friday()
            period_end = d.strftime("%m/%d/%Y")
        else:
            csv_path, period_end = await download_salondata_csv(page, sd_user, sd_pass, period_end_date)

        safe_period = re.sub(r"[^0-9A-Za-z]+", "-", (period_end or "").strip()).strip("-")
        if not safe_period:
            safe_period = "unknown_period"

        safe_user = re.sub(r"[^0-9A-Za-z]+", "_", (username or "").strip()).strip("_")
        if not safe_user:
            safe_user = "unknown_user"

        pdf_filename = f"Payroll_PDF_{safe_period}_{safe_user}.pdf"
        pdf_path = os.path.join(PDF_OUTPUT_DIR, pdf_filename)
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

        build_combined_pdf_for_user(csv_path, username, pdf_path)

        pdf_gridfs_id = None
        try:
            pdf_gridfs_id = _store_pdf_in_gridfs(username, pdf_path, period_end)
        except Exception as e:
            print(f"⚠️ GridFS upload failed; falling back to local PDF path. Reason: {e}")

        pdf_item = {
            "period_end": (period_end or "").strip(),
            "ts": time.time(),
            "filename": pdf_filename,
        }
        if pdf_gridfs_id:
            pdf_item["gridfs_id"] = pdf_gridfs_id
        pdf_item["path"] = pdf_path

        _log_payroll_pdf_item(username, pdf_item)

        if pdf_gridfs_id and not KEEP_LOCAL_PDFS:
            try:
                os.remove(pdf_path)
            except Exception:
                pass

        formatted = format_csv_for_heartland_for_user(csv_path, username)
        if not formatted:
            raise RuntimeError("Formatting returned None. Check console for formatting errors.")

        await upload_to_heartland(page, formatted, hl_user, hl_pass, username)

        await browser.close()
        return {
            "salondata_csv": csv_path,
            "heartland_csv": formatted,
            "combined_pdf": pdf_path,
            "combined_pdf_gridfs_id": pdf_gridfs_id,
            "combined_pdf_filename": pdf_filename,
            "period_end": period_end,
        }


def run_payroll_for_user(username: str, period_end_date=None) -> dict:
    try:
        _update_payroll_status(username, "running")
        sd_user, sd_pass = _get_vendor_creds(username, "salondata")
        hl_user, hl_pass = _get_vendor_creds(username, "heartland")

        ready = check_payroll_ready_for_user(username, dry_run=False, period_end_date=period_end_date)
        if not ready.get("ready"):
            raise RuntimeError(ready.get("error") or "Payroll is not ready. Missing employee keys.")

        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        result = asyncio.run(_full_agentic_flow_inner(
            sd_user, sd_pass, hl_user, hl_pass, username,
            period_end_date=period_end_date,
            csv_path_prefetched=ready.get("csv_path"),
        ))
        if isinstance(result, dict) and (result.get("error") or (result.get("ok") is False)):
            raise RuntimeError(result.get("error") or "Payroll failed.")
        _update_payroll_status(username, "completed")
        return result

    except Exception as e:
        msg = str(e).strip() or f"{type(e).__name__} (no message). See console for traceback."
        _update_payroll_status(username, "failed", error=msg)
        print("❌ run_payroll_for_user failed:", repr(e))
        traceback.print_exc()
        raise RuntimeError(_friendly_error_message(msg)) from e


if __name__ == "__main__":
    u = input("Portal username to run payroll for: ").strip()
    info = run_payroll_for_user(u)
    print("Run complete:", info)
