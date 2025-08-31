import streamlit as st
import pandas as pd
import numpy as np
import yaml
import io
import base64
import os
import datetime
from spellchecker import SpellChecker
import fitz  # PyMuPDF
from typing import List, Dict, Any

# ------------------------------
# CONFIG
# ------------------------------
APP_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"
RULES_FILE = "rules_example.yaml"
HISTORY_FILE = "history.csv"
LOGO_FILE = "logo.png"

SUPPLIERS = ["CEG", "CTIL", "Emfyser", "Innov8", "Invict", "KTL Team (Internal)", "Trylon"]
CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
CAB_LOCATIONS = ["Indoor", "Outdoor"]
RADIO_LOCATIONS = ["Low Level", "Midway", "High Level", "Unique Coverage"]

MIMO_OPTIONS = [
    "18 @2x2","18 @2x2; 26 @4x4","18 @2x2; 70\\80 @2x2","18 @2x2; 80 @2x2",
    "18\\21 @2x2","18\\21 @2x2; 26 @4x4","18\\21 @2x2; 3500 @32x32",
    "18\\21 @2x2; 70\\80 @2x2","18\\21 @2x2; 80 @2x2","18\\21 @4x4",
    "18\\21 @4x4; 3500 @32x32","18\\21 @4x4; 70 @2x4",
    "18\\21 @4x4; 70\\80 @2x2","18\\21 @4x4; 70\\80 @2x2; 3500 @32x32",
    "18\\21 @4x4; 70\\80 @2x4","18\\21 @4x4; 70\\80 @2x4; 3500 @32x32",
    "18\\21 @4x4; 70\\80 @2x4; 3500 @8x8","18\\21@4x4; 70\\80 @2x2",
    "18\\21@4x4; 70\\80 @2x4","18\\21\\26 @2x2","18\\21\\26 @2x2; 3500 @32x32",
    "18\\21\\26 @2x2; 3500 @8X8","18\\21\\26 @2x2; 70\\80 @2x2",
    "18\\21\\26 @2x2; 70\\80 @2x2; 3500 @32x32","18\\21\\26 @2x2; 70\\80 @2x2; 3500 @8x8",
    "18\\21\\26 @2x2; 70\\80 @2x4; 3500 @32x32","18\\21\\26 @2x2; 80 @2x2",
    "18\\21\\26 @2x2; 80 @2x2; 3500 @8x8","18\\21\\26 @4x4",
    "18\\21\\26 @4x4; 3500 @32x32","18\\21\\26 @4x4; 3500 @8x8",
    "18\\21\\26 @4x4; 70 @2x4; 3500 @8x8","18\\21\\26 @4x4; 70\\80 @2x2",
    "18\\21\\26 @4x4; 70\\80 @2x2; 3500 @32x32","18\\21\\26 @4x4; 70\\80 @2x2; 3500 @8x8",
    "18\\21\\26 @4x4; 70\\80 @2x4","18\\21\\26 @4x4; 70\\80 @2x4; 3500 @32x32",
    "18\\21\\26 @4x4; 70\\80 @2x4; 3500 @8x8","18\\21\\26 @4x4; 80 @2x2",
    "18\\21\\26 @4x4; 80 @2x2; 3500 @32x32","18\\21\\26 @4x4; 80 @2x4",
    "18\\21\\26 @4x4; 80 @2x4; 3500 @8x8","18\\26 @2x2",
    "18\\26 @4x4; 21 @2x2; 80 @2x2",""
]

# ------------------------------
# HELPERS
# ------------------------------
def now_utc_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat()

def load_rules(path: str) -> dict:
    if not os.path.exists(path):
        return {"checklist": []}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {"checklist": []}

def save_rules(path: str, data: dict):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)

def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_history(df: pd.DataFrame):
    df.to_csv(HISTORY_FILE, index=False)

def make_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], src_pdf: str, status: str) -> bytes:
    df = pd.DataFrame(findings)
    with io.BytesIO() as mem:
        with pd.ExcelWriter(mem, engine="xlsxwriter") as xw:
            # Summary
            s = pd.DataFrame([{
                **meta,
                "source_pdf": src_pdf,
                "status": status,
                "timestamp_utc": now_utc_iso(),
                "total_findings": len(findings),
                "majors": int((df.severity == "major").sum()) if not df.empty else 0,
                "minors": int((df.severity == "minor").sum()) if not df.empty else 0,
            }])
            s.to_excel(xw, index=False, sheet_name="Summary")

            # Findings
            (
                df if not df.empty 
                else pd.DataFrame(columns=["page","rule","severity","detail"])
            ).to_excel(xw, index=False, sheet_name="Findings")

        return mem.getvalue()

def annotate_pdf(pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    doc = fitz.open("pdf", pdf_bytes)
    for f in findings:
        pg = f.get("page", 0)
        if pg < len(doc):
            page = doc[pg]
            text = f.get("detail", "")
            rect = fitz.Rect(50, 50, 400, 100)  # fixed placeholder
            page.add_highlight_annot(rect)
            page.insert_text((55, 65), f"{f['severity'].upper()}: {text}", fontsize=8, color=(1,0,0))
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

def spelling_findings(pages: List[str], allow: List[str]) -> List[Dict[str, Any]]:
    sp = SpellChecker()
    results = []
    for i, text in enumerate(pages):
        words = text.split()
        for w in words:
            wl = w.strip().lower()
            if wl and wl not in allow and wl in sp.unknown([wl]):
                sug = next(iter(sp.candidates(wl)), None)
                results.append({"page": i, "rule": "Spelling", "severity": "minor",
                                "detail": f"Word '{w}' may be misspelled. Suggest: {sug}"})
    return results

def run_checks(pages: List[str], meta: Dict[str, Any], rules: dict, do_spell: bool, allow: List[str]) -> List[Dict[str, Any]]:
    findings = []
    for r in rules.get("checklist", []):
        for i, text in enumerate(pages):
            must = r.get("must_contain", [])
            rej = r.get("reject_if_present", [])
            for m in must:
                if m not in text:
                    findings.append({"page": i, "rule": r["name"], "severity": r["severity"], "detail": f"Missing {m}"})
            for bad in rej:
                if bad in text:
                    findings.append({"page": i, "rule": r["name"], "severity": r["severity"], "detail": f"Forbidden {bad}"})
    if do_spell:
        findings.extend(spelling_findings(pages, allow))
    return findings

# ------------------------------
# UI
# ------------------------------
def gate():
    if "auth" not in st.session_state:
        st.session_state.auth = False
    if not st.session_state.auth:
        pw = st.text_input("Enter access password", type="password")
        if pw == APP_PASSWORD:
            st.session_state.auth = True
            st.experimental_rerun()
        else:
            st.stop()

def main():
    st.set_page_config("AI Design Quality Auditor", layout="wide")
    gate()

    # Logo
    if os.path.exists(LOGO_FILE):
        logo = open(LOGO_FILE, "rb").read()
        st.image(logo, width=150)

    tabs = st.tabs(["Audit", "Training", "Analytics", "Settings"])

    # ---------------- Audit ----------------
    with tabs[0]:
        st.subheader("Audit Metadata (all required)")
        col1, col2, col3, col4 = st.columns(4)
        supplier = col1.selectbox("Supplier", SUPPLIERS)
        drawing_type = col2.selectbox("Drawing Type", ["General Arrangement", "Detailed Design"])
        client = col3.selectbox("Client", CLIENTS)
        project = col4.selectbox("Project", PROJECTS)
        site_type = col1.selectbox("Site Type", SITE_TYPES)
        vendor = col2.selectbox("Proposed Vendor", VENDORS)
        cab = col3.selectbox("Proposed Cabinet Location", CAB_LOCATIONS)
        radio = col4.selectbox("Proposed Radio Location", RADIO_LOCATIONS)
        sectors = col1.selectbox("Quantity of Sectors", [1,2,3,4,5,6])
        addr = col2.text_input("Site Address")

        st.markdown("### Proposed MIMO Config")
        same = st.checkbox("Use S1 for all sectors", value=True)
        mimo = {}
        for s in range(1, sectors+1):
            if s == 1 or not same:
                mimo[f"S{s}"] = st.selectbox(f"MIMO S{s}", MIMO_OPTIONS, key=f"mimo{s}")
            else:
                mimo[f"S{s}"] = mimo["S1"]

        up = st.file_uploader("Upload PDF Design", type=["pdf"])
        if up and st.button("Run Audit"):
            raw = up.read()
            doc = fitz.open("pdf", raw)
            pages = [p.get_text() for p in doc]
            rules = load_rules(RULES_FILE)
            findings = run_checks(pages, {}, rules, True, [])
            status = "Rejected" if any(f["severity"]=="major" for f in findings) else "Pass"
            excel = make_excel(findings, {"supplier":supplier,"client":client,"project":project}, up.name, status)
            annotated = annotate_pdf(raw, findings)

            st.download_button("⬇️ Download Excel", excel, file_name=f"{up.name}_audit.xlsx")
            st.download_button("⬇️ Download Annotated PDF", annotated, file_name=f"{up.name}_audit.pdf")
            st.dataframe(pd.DataFrame(findings))

    # ---------------- Training ----------------
    with tabs[1]:
        st.subheader("Training")
        train = st.file_uploader("Upload audited report (Excel/JSON)")
        decision = st.selectbox("This audit decision is…", ["Valid","Not Valid"])
        if st.button("Ingest training record"):
            st.success("Training record ingested (placeholder)")

        st.markdown("### Add a quick rule")
        rn = st.text_input("Rule name")
        sev = st.selectbox("Severity", ["major","minor"])
        must = st.text_input("Must contain (comma-separated)")
        rej = st.text_input("Reject if present (comma-separated)")
        pw = st.text_input("Rules password", type="password")
        if st.button("Append rule"):
            if pw == RULES_PASSWORD:
                rules = load_rules(RULES_FILE)
                rules["checklist"].append({
                    "name": rn,
                    "severity": sev,
                    "must_contain": [m.strip() for m in must.split(",") if m.strip()],
                    "reject_if_present": [r.strip() for r in rej.split(",") if r.strip()]
                })
                save_rules(RULES_FILE, rules)
                st.success("Rule appended")
            else:
                st.error("Bad password")

    # ---------------- Analytics ----------------
    with tabs[2]:
        st.subheader("Analytics")
        dfh = load_history()
        if dfh.empty:
            st.info("No history yet")
        else:
            c1, c2, c3 = st.columns(3)
            sf = c1.selectbox("Supplier", ["All"]+SUPPLIERS)
            cf = c2.selectbox("Client", ["All"]+CLIENTS)
            pf = c3.selectbox("Project", ["All"]+PROJECTS)
            show = dfh.copy()
            if sf!="All": show = show[show.supplier==sf]
            if cf!="All": show = show[show.client==cf]
            if pf!="All": show = show[show.project==pf]
            st.dataframe(show)

    # ---------------- Settings ----------------
    with tabs[3]:
        st.subheader("Edit rules (YAML)")
        pw = st.text_input("Rules password", type="password")
        if pw == RULES_PASSWORD:
            txt = st.text_area("rules_example.yaml", value=open(RULES_FILE).read(), height=300)
            if st.button("Save rules"):
                with open(RULES_FILE,"w") as f: f.write(txt)
                st.success("Saved")
        else:
            st.warning("Enter rules password to edit")

# ------------------------------
if __name__ == "__main__":
    main()
