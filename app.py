# -*- coding: utf-8 -*-
import os, io, base64, json, datetime
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import yaml

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from rapidfuzz import fuzz, process

# =========================
# CONSTANTS / CONFIG
# =========================
APP_TITLE = "AI Design Quality Auditor"
ENTRY_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"

HISTORY_DIR = "history"
HISTORY_FILE = os.path.join(HISTORY_DIR, "audit_history.csv")
EXPORT_DIR = os.path.join(HISTORY_DIR, "exports")
ERRLOG_DIR = os.path.join(HISTORY_DIR, "error_logs")

RULES_FILE = "rules_example.yaml"
LOGO_FILE = "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png"  # set your actual filename here

# Dropdown vocab
SUPPLIERS = ["CEG", "CTIL", "Emfyser", "Innov8", "Invict", "KTL Team (Internal)", "Trylon"]
CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
CABINETS = ["Indoor", "Outdoor"]
RADIOS = ["Low Level", "High Level", "Midway", "Unique Coverage"]
DRAWINGS = ["General Arrangement", "Detailed Design"]

MIMO_OPTIONS = [
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
    "18\\21\\26 @4x4; 3500 @32x32","18\\21\\26 @4x4; 3500 @8x8","18\\21\\26 @4x4; 70 @2x4; 3500 @8x8",
    "18\\21\\26 @4x4; 70\\80 @2x2","18\\21\\26 @4x4; 70\\80 @2x2; 3500 @32x32",
    "18\\21\\26 @4x4; 70\\80 @2x2; 3500 @8x8","18\\21\\26 @4x4; 70\\80 @2x4",
    "18\\21\\26 @4x4; 70\\80 @2x4; 3500 @32x32","18\\21\\26 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21\\26 @4x4; 80 @2x2","18\\21\\26 @4x4; 80 @2x2; 3500 @32x32","18\\21\\26 @4x4; 80 @2x4",
    "18\\21\\26 @4x4; 80 @2x4; 3500 @8x8","18\\26 @2x2","18\\26 @4x4; 21 @2x2; 80 @2x2",
    "(blank)"
]

# Ensure dirs
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(ERRLOG_DIR, exist_ok=True)

# =========================
# UTILITIES
# =========================
def css_logo():
    if os.path.exists(LOGO_FILE):
        b64 = base64.b64encode(open(LOGO_FILE, "rb").read()).decode()
        st.markdown(
            f"""
            <style>
            .brand-logo {{
                position: fixed; left: 14px; top: 12px; z-index: 1000;
                border-radius: 10px; background: rgba(0,0,0,0.0);
            }}
            .brand-logo img {{
                height: 54px; pointer-events: none;
            }}
            </style>
            <div class="brand-logo"><img src="data:image/png;base64,{b64}"/></div>
            """,
            unsafe_allow_html=True,
        )

def gate():
    if st.session_state.get("pass_ok"):
        return
    pw = st.text_input("Enter access password", type="password")
    if pw == ENTRY_PASSWORD:
        st.session_state["pass_ok"] = True
        st.experimental_rerun()
    else:
        st.stop()

def load_rules(path: str = RULES_FILE) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"checklist": [], "spelling_whitelist": []}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {"checklist": [], "spelling_whitelist": []}

def save_rules_yaml(data: Dict[str, Any], path: str = RULES_FILE):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def stable_history_columns() -> List[str]:
    return [
        "timestamp_utc","supplier","drawing_type","client","project",
        "site_type","vendor","cabinet","radio","sectors","address",
        "mimo_S1","mimo_S2","mimo_S3","mimo_S4","mimo_S5","mimo_S6",
        "status","pdf_name","excel_name","exclude","training_source"
    ]

def load_history_df() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=stable_history_columns())
    try:
        df = pd.read_csv(HISTORY_FILE)
    except Exception:
        # Corrupted file fallback
        df = pd.DataFrame(columns=stable_history_columns())
    for c in stable_history_columns():
        if c not in df.columns:
            df[c] = None
    return df[stable_history_columns()]

def append_history_row(row: Dict[str, Any]):
    df = pd.DataFrame([row], columns=stable_history_columns())
    header = not os.path.exists(HISTORY_FILE)
    df.to_csv(HISTORY_FILE, mode="a", header=header, index=False)

def get_pdf_text_pages(pdf_bytes: bytes) -> List[str]:
    """Extract text per page using PyMuPDF; OCR fallback for empty pages."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for i in range(len(doc)):
        page = doc[i]
        t = page.get_text("text") or ""
        if not t.strip():
            # OCR fallback on the rendered page (low-res to save CPU)
            pix = page.get_pixmap(dpi=110)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            t = pytesseract.image_to_string(img)
        texts.append(t)
    return texts

def rule_applies(rule: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    """Optional selective application based on meta fields."""
    sel = rule.get("apply_if", {})
    for k, v in sel.items():
        mv = meta.get(k)
        if isinstance(v, list):
            if mv not in v:
                return False
        else:
            if mv != v:
                return False
    return True

def scan_rule(pages: List[str], rule: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return findings for a single rule."""
    findings = []
    must = [s.upper() for s in rule.get("must_contain", [])]
    reject_if = [s.upper() for s in rule.get("reject_if_present", [])]
    scope = rule.get("scope", "any")  # "any" or specific page number

    pages_to_scan = range(1, len(pages) + 1) if scope == "any" else [int(scope)]
    for pn in pages_to_scan:
        body = pages[pn - 1].upper()
        # must contain checks
        for needle in must:
            if needle not in body:
                findings.append({
                    "page": pn,
                    "severity": rule.get("severity", "minor"),
                    "rule": rule.get("name", "Unnamed rule"),
                    "message": f"Expected to find: '{needle}'",
                    "keyword": needle[:60]
                })
        # reject if present checks
        for bad in reject_if:
            if bad in body:
                findings.append({
                    "page": pn,
                    "severity": rule.get("severity", "minor"),
                    "rule": rule.get("name", "Unnamed rule"),
                    "message": f"Forbidden text present: '{bad}'",
                    "keyword": bad[:60]
                })
    return findings

def spelling_check(pages: List[str], whitelist: List[str]) -> List[Dict[str, Any]]:
    whitelist_u = [w.upper() for w in whitelist]
    findings = []
    for i, raw in enumerate(pages, start=1):
        words = [w.strip(",.;:()[]{}!/\\\"'").upper() for w in raw.split()]
        for w in set([x for x in words if x and any(c.isalpha() for c in x)]):
            if w in whitelist_u:
                continue
            # fast fuzzy suggestion
            sug, score, _ = process.extractOne(w, whitelist_u, scorer=fuzz.ratio) if whitelist_u else (None, 0, None)
            if sug and score >= 88:
                findings.append({
                    "page": i,
                    "severity": "minor",
                    "rule": "Spelling",
                    "message": f"Possible typo '{w}'. Suggest '{sug}'",
                    "keyword": w[:60]
                })
    return findings

def annotate_pdf(pdf_bytes: bytes, comments: List[Dict[str, Any]]) -> bytes:
    """Add rectangle or text annotations based on keyword match."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for f in comments:
        msg = f.get("message", "")
        kw = f.get("keyword") or ""
        pn = max(0, min(int(f.get("page", 1)) - 1, len(doc) - 1))
        added = False
        if kw:
            page = doc[pn]
            # look for the keyword block on that page
            blocks = page.get_text("blocks")
            for x0, y0, x1, y1, txt, *_ in blocks:
                if fuzz.partial_ratio(kw.upper(), (txt or "").upper()) >= 85:
                    rect = fitz.Rect(x0, y0, x1, y1)
                    annot = page.add_rect_annot(rect)
                    annot.set_info(title="QA", content=msg)
                    annot.update()
                    added = True
        if not added:
            # fallback: sticky text near margin
            page = doc[pn]
            pt = fitz.Point(40, 60)
            annot = page.add_text_annot(pt, msg)
            annot.update()
    return doc.tobytes()

def make_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], pdf_name: str, status: str) -> bytes:
    df = pd.DataFrame(findings or [])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        # Summary sheet
        m = pd.DataFrame([{
            **meta,
            "status": status,
            "pdf_name": pdf_name,
            "timestamp_utc": datetime.datetime.utcnow().isoformat()
        }])
        m.to_excel(xw, sheet_name="Summary", index=False)
        # Findings
        (df if not df.empty else pd.DataFrame([{"info": "No findings"}])).to_excel(
            xw, sheet_name="Findings", index=False
        )
    buf.seek(0)
    return buf.getvalue()

# =========================
# AUDIT LOGIC
# =========================
def run_audit(pdf_bytes: bytes, meta: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    pages = get_pdf_text_pages(pdf_bytes)
    rules = load_rules()
    findings: List[Dict[str, Any]] = []

    # 1) rules (filtered with apply_if)
    for rule in rules.get("checklist", []):
        if rule_applies(rule, meta):
            findings.extend(scan_rule(pages, rule))

    # 2) spelling (optional)
    wl = rules.get("spelling_whitelist", [])
    if wl:
        findings.extend(spelling_check(pages, wl))

    # 3) simple “site address must be in title” check
    addr = (meta.get("address") or "").strip()
    if addr and addr != "MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ":
        title_page = " ".join(pages[0:1]).upper()
        if addr.upper() not in title_page:
            findings.append({
                "page": 1, "severity": "major", "rule": "Address/Title",
                "message": "Site Address not found in title block", "keyword": addr[:60]
            })

    return pages, findings

# =========================
# UI – AUDIT TAB
# =========================
def audit_tab():
    st.subheader("Audit Metadata (all required)")

    col1, col2, col3 = st.columns(3)
    supplier = col1.selectbox("Supplier", SUPPLIERS)
    drawing_type = col2.selectbox("Drawing Type", DRAWINGS)
    client = col3.selectbox("Client", CLIENTS)

    col4, col5, col6 = st.columns(3)
    project = col4.selectbox("Project", PROJECTS)
    site_type = col5.selectbox("Site Type", SITE_TYPES)
    vendor = col6.selectbox("Proposed Vendor", VENDORS)

    col7, col8, col9 = st.columns(3)
    cabinet = col7.selectbox("Proposed Cabinet Location", CABINETS)
    radio = col8.selectbox("Proposed Radio Location", RADIOS)
    sectors = col9.selectbox("Quantity of Sectors", [1, 2, 3, 4, 5, 6])

    address = st.text_input("Site Address", help="Must be present in the drawing title (unless exactly the special '0' format).")

    st.markdown("### Proposed MIMO Config")
    same_for_all = st.checkbox("Use S1 for all sectors", True)

    mimo_vals = {}
    for s in range(1, sectors + 1):
        val = st.selectbox(f"MIMO S{s}", MIMO_OPTIONS, key=f"mimo_{s}")
        if same_for_all and s > 1:
            val = mimo_vals["S1"]
        mimo_vals[f"S{s}"] = val

    st.markdown("### Upload PDF Design")
    up_pdf = st.file_uploader("Design PDF", type="pdf")

    st.markdown("### Training quick-actions (optional)")
    c1, c2, c3 = st.columns([1, 1, 2])
    train_pdf = c1.file_uploader("Upload audited report (PDF)", type=["pdf"], key="train_pdf")
    train_label = c2.selectbox("Label", ["Valid", "Not Valid"], key="train_label")
    train_note = c3.text_input("Rule or note to add (optional)", placeholder="e.g., 'AHEGC' -> 'AHEGG' on title page")

    excl = st.checkbox("Exclude this audit from analytics", False)

    run = st.button("Run Audit", type="primary")

    # Save error log
    err_up = st.file_uploader("Upload an error log (optional)", type=["txt", "log"])
    if err_up is not None:
        with open(os.path.join(ERRLOG_DIR, f"{datetime.datetime.utcnow().isoformat()}_{err_up.name}"), "wb") as f:
            f.write(err_up.read())
        st.success("Error log saved.")

    meta = {
        "supplier": supplier, "drawing_type": drawing_type, "client": client, "project": project,
        "site_type": site_type, "vendor": vendor, "cabinet": cabinet, "radio": radio,
        "sectors": sectors, "address": address,
        "mimo_S1": mimo_vals.get("S1"), "mimo_S2": mimo_vals.get("S2"),
        "mimo_S3": mimo_vals.get("S3"), "mimo_S4": mimo_vals.get("S4"),
        "mimo_S5": mimo_vals.get("S5"), "mimo_S6": mimo_vals.get("S6"),
    }

    # Training ingest (simple)
    if train_pdf is not None:
        # record a training marker row
        append_history_row({
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            **meta, "status": train_label, "pdf_name": train_pdf.name,
            "excel_name": None, "exclude": True, "training_source": "upload"
        })
        if train_note:
            # append a light rule stub to YAML under a "training" section
            rules = load_rules()
            tr = rules.get("training_notes", [])
            tr.append({
                "when": { "client": client, "project": project },
                "note": train_note,
                "label": train_label,
                "ts": datetime.datetime.utcnow().isoformat()
            })
            rules["training_notes"] = tr
            save_rules_yaml(rules)
        st.success("Training item recorded (analytics excluded).")

    if not run or not up_pdf:
        return

    raw = up_pdf.read()
    pages, findings = run_audit(raw, meta)

    status = "Pass" if len([f for f in findings if f.get("severity") == "major"]) == 0 and len(findings) == 0 else "Rejected"

    # Annotated PDF
    annotated = annotate_pdf(raw, findings)
    ann_name = up_pdf.name.replace(".pdf", "_annotated.pdf")
    st.download_button("Download Annotated PDF", data=annotated, file_name=ann_name, use_container_width=True)

    # Excel
    excel_bytes = make_excel(findings, meta, up_pdf.name, status)
    xlsx_name = up_pdf.name.replace(".pdf", "_report.xlsx")
    st.download_button("Download Excel Report", data=excel_bytes, file_name=xlsx_name, use_container_width=True)

    # Persist history
    append_history_row({
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        **meta, "status": status,
        "pdf_name": up_pdf.name, "excel_name": xlsx_name,
        "exclude": bool(excl), "training_source": None
    })

    # Show table
    st.markdown("#### Findings")
    if findings:
        st.dataframe(pd.DataFrame(findings), use_container_width=True, height=260)
    else:
        st.success("No findings.")

# =========================
# UI – ANALYTICS TAB
# =========================
def analytics_tab():
    st.subheader("Analytics & Trends")
    df = load_history_df()
    if df.empty:
        st.info("No history yet.")
        return
    df_use = df[df["exclude"] != True].copy()

    c1, c2, c3 = st.columns(3)
    fs = c1.selectbox("Supplier", ["All"] + SUPPLIERS)
    fc = c2.selectbox("Client", ["All"] + CLIENTS)
    fp = c3.selectbox("Project", ["All"] + PROJECTS)

    if fs != "All": df_use = df_use[df_use["supplier"] == fs]
    if fc != "All": df_use = df_use[df_use["client"] == fc]
    if fp != "All": df_use = df_use[df_use["project"] == fp]

    total = len(df_use)
    rft = round(100 * len(df_use[df_use["status"] == "Pass"]) / total, 1) if total else 0.0
    colA, colB, colC = st.columns(3)
    colA.metric("Audits", total)
    colB.metric("Right First Time %", rft)
    colC.metric("Rejected", int(len(df_use[df_use["status"] == "Rejected"])))

    cols = ["timestamp_utc","supplier","client","project","status","pdf_name","excel_name"]
    st.dataframe(df_use[cols], use_container_width=True, height=360)

# =========================
# UI – SETTINGS TAB
# =========================
def settings_tab():
    st.subheader("Edit rules (YAML)")
    pw = st.text_input("Rules password", type="password")
    if pw != RULES_PASSWORD:
        st.warning("Enter the correct rules password to edit.")
        return

    curr = load_rules()
    text = st.text_area("rules_example.yaml", yaml.safe_dump(curr, sort_keys=False, allow_unicode=True), height=420)

    c1, c2 = st.columns([1,1])
    if c1.button("Save rules"):
        try:
            data = yaml.safe_load(text) or {}
            save_rules_yaml(data)
            st.success("Rules saved.")
        except Exception as e:
            st.error(f"YAML error: {e}")
    if c2.button("Reload from disk"):
        st.experimental_rerun()

# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    css_logo()
    gate()  # password gate

    st.title(APP_TITLE)
    st.caption("Professional, consistent design QA with annotations, analytics, and easy rule updates.")

    tab1, tab2, tab3 = st.tabs(["Audit", "Analytics", "Settings"])
    with tab1:
        audit_tab()
    with tab2:
        analytics_tab()
    with tab3:
        settings_tab()

if __name__ == "__main__":
    main()
