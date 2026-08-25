# Payroll App Architecture

## Simple System Diagram

If you paste this into Mermaid Live Editor, use `ARCHITECTURE.mmd` or copy only the lines between the code fences. Do not include the ```mermaid and ``` lines.

```mermaid
flowchart TD
    User["User in Browser"] --> UI["Streamlit App<br/>tester8_admin_handyman.py"]

    UI --> Auth["Login + Admin UI<br/>users, roles, setup"]
    UI --> Mongo["MongoDB<br/>userInfo, employeeKeysByUser,<br/>loginEvents, PDF metadata"]
    UI --> Secrets["Streamlit Secrets / Env Vars<br/>Mongo URI, encryption key,<br/>Gemini key"]

    Auth --> Crypto["crypto_utils.py<br/>encrypt/decrypt saved vendor passwords"]
    Crypto --> Mongo

    UI --> Readiness["Check Payroll Readiness<br/>background thread"]
    UI --> Execute["Execute Payroll<br/>background thread"]
    UI --> MFA["MFA Code Box<br/>writes mfa_code to Mongo"]

    Readiness --> Runner["Payroll Runner<br/>payrollrunner_dbkeys_handyman.py"]
    Execute --> Runner
    MFA --> Mongo

    Runner --> RunCtx["Per-Run Folder<br/>run_context.py<br/>payroll_runs/user/run_id"]
    Runner --> SalonData["SalonData Website<br/>Playwright downloads payroll CSV"]
    Runner --> Heartland["Heartland Website<br/>Playwright login, MFA,<br/>tenant/client selection, upload"]

    Runner --> Profiles["Tenant Profiles<br/>multi_tenant_profiles.py<br/>parser + Heartland selections"]
    Profiles --> Mongo

    Runner --> Gemini["Optional Gemini Vision Helper<br/>vision_handyman_agent.py<br/>fallback clicks for flaky pages"]

    SalonData --> CSV["SalonData Payroll CSV"]
    Heartland --> EmployeeReport["Heartland Employee ID Report"]
    CSV --> Keys["Employee Key Matching<br/>employeeKeysByUser"]
    EmployeeReport --> Keys
    Keys --> Mongo

    Runner --> PDF["Combined Payroll PDF<br/>ReportLab"]
    Runner --> UploadCSV["Formatted Heartland CSV"]

    UploadCSV --> Heartland
    PDF --> GridFS["Mongo GridFS<br/>stored payroll PDFs"]
    GridFS --> UI
    UI --> Download["Download PDF Button"]
    Download --> User
```

## Workflow In Plain English

1. The user opens the Streamlit app in the browser.
2. The app reads settings from `.streamlit/secrets.toml` or environment variables.
3. The user logs in. User records, roles, encrypted vendor credentials, MFA codes, and payroll status live in MongoDB.
4. The user can run **Check Payroll Readiness**.
   - First it downloads SalonData payroll data.
   - It compares employees against saved employee keys.
   - If keys are missing, it logs into Heartland to sync/download the Employee ID report.
   - If Heartland asks for MFA, the UI waits for the user to type the code.
5. The user can run **Execute Payroll**.
   - The backend starts a background thread.
   - Playwright opens SalonData and downloads the payroll CSV.
   - Playwright logs into Heartland, handles MFA, chooses the correct profile/client, and reaches the payroll area.
   - The runner builds a payroll PDF and formats a Heartland upload CSV.
   - The upload CSV is sent into Heartland.
6. The generated PDF is stored in Mongo GridFS and logged on the user document.
7. The Streamlit UI refreshes status and shows the latest PDF download button.

## Main Files

- `tester8_admin_handyman.py`: Streamlit app, login, admin page, readiness button, execute button, MFA input, PDF download.
- `payrollrunner_dbkeys_handyman.py`: Main backend automation and payroll processing.
- `multi_tenant_profiles.py`: Per-user Heartland profile/client choices and parser profile settings.
- `payroll_backend_bridge.py`: Small adapter connecting the runner to tenant profiles and run folders.
- `run_context.py`: Creates safe per-user, per-run folders.
- `crypto_utils.py`: Encrypts and decrypts saved vendor credentials.
- `mongo_helpers.py`: Shared MongoDB helpers.
- `gridfs_pdf_storage.py`: Stores generated PDFs in Mongo GridFS.
- `vision_handyman_agent.py`: Optional Gemini screenshot-based helper for flaky browser steps.
- `app_config.py`: Shared configuration from Streamlit secrets or environment variables.

## Data Stores

- `userInfo`: users, roles, encrypted credentials, payroll status, latest PDF metadata, MFA code.
- `employeeKeysByUser`: saved employee key mappings by user.
- `loginEvents`: login attempt history.
- `payroll_pdfs` GridFS bucket: generated payroll PDFs.

## External Systems

- **SalonData**: source payroll report CSV.
- **Heartland**: MFA, profile/client selection, Employee ID report, final payroll CSV upload.
- **Gemini API**: optional visual fallback when page selectors fail.
