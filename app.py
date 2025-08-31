import streamlit as st
import pandas as pd
import yaml
import base64
import io
import os
import datetime
from rapidfuzz import fuzz
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from openpyxl import Workbook
import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------
LOGO_PATH = "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png"
HISTORY_FILE = "history/audit_history.csv"
RULES_FILE = "rules_example.yaml"
RULES_PASSWORD = "vanB3lkum21"
ENTRY_PASSWORD = "Seker123"

DEFAULT_UI_VISIBILITY_HOURS = 24
EXPORT_FOLDER = "daily_exports"
os.makedirs("history", exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)

# ------------------ HELPERS ------------------
def draw_logo_top_left():
    try:
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
              .seker-logo {{
                position: fixed;
                top: 8px;
                left: 12px;
                width: 180px;
                max-width: 35vw;
                z-index: 9999;
                pointer-events: none;
              }}
              .block-container {{
                padding-top: 90px !important;
              }}
            </style>
            <img class="seker-logo" src="data:image/png;base64,{b64}" />
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

def load_rules(path=RULES_FILE):
    if not os.path.exists(path):
        return {"checklist": []}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {"checklist": []}

def save_rules(data, path=RULES_FILE):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)

def text_from_pdf(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_pages = [page.extract_text() or "" for page in reader.pages]
    return text_pages

def spelling_checks(pages, allowlist):
    findings = []
    for pi, text in enumerate(pages):
        words = text.split()
        for w in words:
            if w.isalpha() and w.lower() not in allowlist:
                findings.append({
                    "page": pi+1,
                    "issue": f"Potential spelling error: {w}",
                    "severity": "minor",
                    "suggestion": None
                })
    return findings

def apply_rules(pages, rules):
    findings = []
    for rule in rules.get("checklist", []):
        name = rule.get("name", "")
        severity = rule.get("severity", "minor")
        must = rule.get("must_contain", [])
        reject = rule.get("reject_if_present", [])
        for pi, text in enumerate(pages):
            for m in must:
                if m not in text:
                    findings.append({
                        "page": pi+1,
                        "issue": f"Missing required text '{m}' for rule '{name}'",
                        "severity": severity,
                        "suggestion": f"Add '{m}'"
                    })
            for r in reject:
                if r in text:
                    findings.append({
                        "page": pi+1,
                        "issue": f"Forbidden text '{r}' found for rule '{name}'",
                        "severity": severity,
                        "suggestion": f"Remove '{r}'"
                    })
    return findings

def make_excel(findings, meta, fname, status):
    wb = Workbook()
    ws = wb.active
    ws.title = "Audit Report"
    ws.append(["File", "Status", "Supplier", "Client", "Project", "Vendor", "Cabinet", "Radio", "Address", "Issue", "Page", "Severity", "Suggestion"])
    for f in findings:
        ws.append([
            fname, status,
            meta["Supplier"], meta["Client"], meta["Project"], meta["Proposed Vendor"],
            meta["Proposed Cabinet Location"], meta["Proposed Radio Location"], meta["Site Address"],
            f["issue"], f["page"], f["severity"], f.get("suggestion", "")
        ])
    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()

def append_history(findings, meta, fname, status):
    now = datetime.datetime.utcnow().isoformat()
    rows = []
    for f in findings:
        rows.append({
            "timestamp_utc": now,
            "file": fname,
            "status": status,
            "issue": f["issue"],
            "severity": f["severity"],
            "supplier": meta["Supplier"],
            "client": meta["Client"],
            "project": meta["Project"],
            "vendor": meta["Proposed Vendor"],
            "radio": meta["Proposed Radio Location"],
            "address": meta["Site Address"],
            "exclude": False
        })
    df = pd.DataFrame(rows)
    if os.path.exists(HISTORY_FILE):
        dfh = pd.read_csv(HISTORY_FILE)
        dfh = pd.concat([dfh, df], ignore_index=True)
    else:
        dfh = df
    dfh.to_csv(HISTORY_FILE, index=False)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def filter_history_for_ui(df, hours=DEFAULT_UI_VISIBILITY_HOURS):
    if "timestamp_utc" not in df:
        return df
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)
    return df[df["timestamp_utc"] >= cutoff]

# ------------------ UI ------------------
def gate_with_password():
    pw = st.text_input("Enter access password", type="password")
    if pw != ENTRY_PASSWORD:
        st.stop()

def main():
    gate_with_password()
    draw_logo_top_left()
    st.title("AI Design Quality Auditor")

    tabs = st.tabs(["Audit", "History & Analytics", "Settings"])

    # ---------------- Audit ----------------
    with tabs[0]:
        st.header("Audit Metadata (all required)")

        suppliers = ["CEG","CTIL","Emfyser","Innov8","Invict","KTL Team (Internal)","Trylon"]
        clients = ["BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"]
        projects = ["RAN","Power Resilience","East Unwind","Beacon 4"]
        site_types = ["Greenfield","Rooftop","Streetworks"]
        vendors = ["Ericsson","Nokia"]
        cabinets = ["Indoor","Outdoor"]
        radios = ["Low Level","High Level","Unique Coverage","Midway"]
        drawing_types = ["General Arrangement","Detailed Design"]

        meta = {}
        col1,col2,col3,col4 = st.columns(4)
        with col1: meta["Supplier"] = st.selectbox("Supplier", suppliers)
        with col2: meta["Drawing Type"] = st.selectbox("Drawing Type", drawing_types)
        with col3: meta["Client"] = st.selectbox("Client", clients)
        with col4: meta["Project"] = st.selectbox("Project", projects)

        col5,col6,col7,col8 = st.columns(4)
        with col5: meta["Site Type"] = st.selectbox("Site Type", site_types)
        with col6: meta["Proposed Vendor"] = st.selectbox("Proposed Vendor", vendors)
        with col7: meta["Proposed Cabinet Location"] = st.selectbox("Proposed Cabinet Location", cabinets)
        with col8: meta["Proposed Radio Location"] = st.selectbox("Proposed Radio Location", radios)

        meta["Quantity of Sectors"] = st.selectbox("Quantity of Sectors", [1,2,3,4,5,6])
        meta["Site Address"] = st.text_input("Site Address")

        st.subheader("Proposed MIMO Config")
        use_all = st.checkbox("Use S1 for all sectors", value=True)
        mimo_opts = [
            "18 @2x2","18 @2x2; 26 @4x4","18 @2x2; 70\\80 @2x2","18 @2x2; 80 @2x2",
            "18\\21 @2x2","18\\21 @2x2; 26 @4x4","18\\21 @2x2; 3500 @32x32",
            "18\\21 @2x2; 70\\80 @2x2","18\\21 @2x2; 80 @2x2","18\\21 @4x4",
            "18\\21 @4x4; 3500 @32x32","18\\21 @4x4; 70 @2x4","18\\21 @4x4; 70\\80 @2x2",
            "18\\21 @4x4; 70\\80 @2x2; 3500 @32x32","18\\21 @4x4; 70\\80 @2x4",
            "18\\21 @4x4; 70\\80 @2x4; 3500 @32x32","18\\21 @4x4; 70\\80 @2x4; 3500 @8x8",
            "18\\21@4x4; 70\\80 @2x2","18\\21@4x4; 70\\80 @2x4","18\\21\\26 @2x2",
            "18\\21\\26 @2x2; 3500 @32x32","18\\21\\26 @2x2; 3500 @8X8",
            "18\\21\\26 @2x2; 70\\80 @2x2","18\\21\\26 @2x2; 70\\80 @2x2; 3500 @32x32",
            "18\\21\\26 @2x2; 70\\80 @2x2; 3500 @8x8","18\\21\\26 @2x2; 70\\80 @2x4; 3500 @32x32",
            "18\\21\\26 @2x2; 80 @2x2","18\\21\\26 @2x2; 80 @2x2; 3500 @8x8","18\\21\\26 @4x4",
            "18\\21\\26 @4x4; 3500 @32x32","18\\21\\26 @4x4; 3500 @8x8",
            "18\\21\\26 @4x4; 70 @2x4; 3500 @8x8","18\\21\\26 @4x4; 70\\80 @2x2",
            "18\\21\\26 @4x4; 70\\80 @2x2; 3500 @32x32","18\\21\\26 @4x4; 70\\80 @2x2; 3500 @8x8",
            "18\\21\\26 @4x4; 70\\80 @2x4","18\\21\\26 @4x4; 70\\80 @2x4; 3500 @32x32",
            "18\\21\\26 @4x4; 70\\80 @2x4; 3500 @8x8","18\\21\\26 @4x4; 80 @2x2",
            "18\\21\\26 @4x4; 80 @2x2; 3500 @32x32","18\\21\\26 @4x4; 80 @2x4",
            "18\\21\\26 @4x4; 80 @2x4; 3500 @8x8","18\\26 @2x2","18\\26 @4x4; 21 @2x2; 80 @2x2"
        ]
        if use_all:
            meta["MIMO"] = {f"S{i+1}": st.selectbox("MIMO S1", mimo_opts) for i in range(meta["Quantity of Sectors"])}
        else:
            meta["MIMO"] = {f"S{i+1}": st.selectbox(f"MIMO S{i+1}", mimo_opts) for i in range(meta["Quantity of Sectors"])}

        up = st.file_uploader("Upload PDF Design", type=["pdf"])
        if up and st.button("Run Audit"):
            raw_pdf = up.read()
            pages = text_from_pdf(raw_pdf)
            rules = load_rules(RULES_FILE)
            findings = []
            findings += spelling_checks(pages, allowlist=[])
            findings += apply_rules(pages, rules)

            status = "Rejected" if any(f["severity"]=="major" for f in findings) else "Pass"

            excel_bytes = make_excel(findings, meta, up.name, status)
            append_history(findings, meta, up.name, status)

            st.download_button("Download Excel Report", excel_bytes, file_name=f"{up.name}_{status}.xlsx")

    # ---------------- History & Analytics ----------------
    with tabs[1]:
        st.header("History & Analytics")
        dfh = load_history()
        if not dfh.empty:
            dfh = filter_history_for_ui(dfh)
            st.dataframe(dfh.tail(50))

            fig, ax = plt.subplots()
            dfh["severity"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)
        else:
            st.info("No history yet.")

    # ---------------- Settings ----------------
    with tabs[2]:
        st.header("Edit rules (YAML)")
        pw = st.text_input("Rules password", type="password")
        if pw == RULES_PASSWORD:
            rules_txt = st.text_area("rules_example.yaml", value=open(RULES_FILE).read(), height=300)
            if st.button("Save rules"):
                with open(RULES_FILE, "w") as f:
                    f.write(rules_txt)
                st.success("Rules updated.")

if __name__ == "__main__":
    main()
