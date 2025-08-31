# app.py
# AI Design Quality Auditor – compact, production-ready Streamlit app
# ---------------------------------------------------------------
# Tabs: Audit | Analytics | Settings
# - Password gate (Seker123)
# - YAML rules editing (vanB3lkum21)
# - Dynamic MIMO per sector with "Use S1 for all sectors"
# - PDF checks + annotation (PyMuPDF)
# - Excel report export
# - History with exclude flag + Analytics filters
# ---------------------------------------------------------------

import io
import os
import re
import json
import time
import base64
import zipfile
import textwrap
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Any

import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
import yaml

APP_TITLE = "AI Design Quality Auditor"
REPO_ROOT = "."
HISTORY_DIR = os.path.join(REPO_ROOT, "history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.csv")
RULES_FILE_DEFAULT = os.path.join(REPO_ROOT, "rules_example.yaml")

ENTRY_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"

# ------------ UI constants (dropdowns) ----------------
SUPPLIERS = [
    "CEG", "CTIL", "Emfyser", "Innov8", "Invict", "KTL Team (Internal)", "Trylon"
]
CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
CAB_LOC = ["Indoor", "Outdoor"]
RADIO_LOC = ["Low Level", "High Level", "Unique Coverage", "Midway"]
SECTORS = ["1", "2", "3", "4", "5", "6"]

MIMO_OPTIONS = [
"18 @2x2",
"18 @2x2; 26 @4x4",
"18 @2x2; 70\\80 @2x2",
"18 @2x2; 80 @2x2",
"18\\21 @2x2",
"18\\21 @2x2; 26 @4x4",
"18\\21 @2x2; 3500 @32x32",
"18\\21 @2x2; 70\\80 @2x2",
"18\\21 @2x2; 80 @2x2",
"18\\21 @4x4",
"18\\21 @4x4; 3500 @32x32",
"18\\21 @4x4; 70 @2x4",
"18\\21 @4x4; 70\\80 @2x2",
"18\\21 @4x4; 70\\80 @2x2; 3500 @32x32",
"18\\21 @4x4; 70\\80 @2x4",
"18\\21 @4x4; 70\\80 @2x4; 3500 @32x32",
"18\\21 @4x4; 70\\80 @2x4; 3500 @8x8",
"18\\21@4x4; 70\\80 @2x2",
"18\\21@4x4; 70\\80 @2x4",
"18\\21\\26 @2x2",
"18\\21\\26 @2x2; 3500 @32x32",
"18\\21\\26 @2x2; 3500 @8X8",
"18\\21\\26 @2x2; 70\\80 @2x2",
"18\\21\\26 @2x2; 70\\80 @2x2; 3500 @32x32",
"18\\21\\26 @2x2; 70\\80 @2x2; 3500 @8x8",
"18\\21\\26 @2x2; 70\\80 @2x4; 3500 @32x32",
"18\\21\\26 @2x2; 80 @2x2",
"18\\21\\26 @2x2; 80 @2x2; 3500 @8x8",
"18\\21\\26 @4x4",
"18\\21\\26 @4x4; 3500 @32x32",
"18\\21\\26 @4x4; 3500 @8x8",
"18\\21\\26 @4x4; 70 @2x4; 3500 @8x8",
"18\\21\\26 @4x4; 70\\80 @2x2",
"18\\21\\26 @4x4; 70\\80 @2x2; 3500 @32x32",
"18\\21\\26 @4x4; 70\\80 @2x2; 3500 @8x8",
"18\\21\\26 @4x4; 70\\80 @2x4",
"18\\21\\26 @4x4; 70\\80 @2x4; 3500 @32x32",
"18\\21\\26 @4x4; 70\\80 @2x4; 3500 @8x8",
"18\\21\\26 @4x4; 80 @2x2",
"18\\21\\26 @4x4; 80 @2x2; 3500 @32x32",
"18\\21\\26 @4x4; 80 @2x4",
"18\\21\\26 @4x4; 80 @2x4; 3500 @8x8",
"18\\26 @2x2",
"18\\26 @4x4; 21 @2x2; 80 @2x2",
"(blank)"
]

# --------- utility -----------
def ensure_dirs():
    os.makedirs(HISTORY_DIR, exist_ok=True)

def now_utc_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

# --------- Logo -------------
def load_logo_html():
    # Try to find any reasonable logo in repo root (png/svg/jpg)
    candidates = [
        "logo.png", "logo.svg", "logo.jpg", "logo.jpeg",
        # also support the user's uploaded GUID name
        "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png",
        "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png",
    ]
    for name in candidates:
        p = os.path.join(REPO_ROOT, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            mime = "image/png" if name.lower().endswith("png") else (
                "image/svg+xml" if name.lower().endswith("svg") else "image/jpeg"
            )
            return f"""
            <div class="brand-logo">
              <img src="data:{mime};base64,{b64}" />
            </div>
            <style>
            .brand-logo {{
              position: fixed;
              left: 16px;
              top: 12px;
              z-index: 1000;
              pointer-events: none; /* don't block clicks */
            }}
            .brand-logo img {{
              height: 56px;
              opacity: 0.95;
              filter: drop-shadow(0 2px 2px rgba(0,0,0,0.35));
            }}
            </style>
            """
    return ""  # silent if missing

# -------- Rules I/O ----------
def load_rules(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"checklist": []}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # normalize
    if "checklist" not in data or data["checklist"] is None:
        data["checklist"] = []
    return data

def save_rules(path: str, rules: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(rules, f, sort_keys=False, allow_unicode=True)

# ------- PDF text + index ----
def extract_all_text(pdf_bytes: bytes) -> List[str]:
    pages: List[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i in range(len(doc)):
            pages.append(doc[i].get_text("text"))
    return pages

class PdfIndex:
    def __init__(self, pdf_bytes: bytes):
        self.doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        self.n_pages = len(self.doc)

    def search_exact(self, needle: str) -> List[Tuple[int, fitz.Rect]]:
        """Case-insensitive substring locate via search_for (no flags needed)."""
        out: List[Tuple[int, fitz.Rect]] = []
        pattern = needle.strip()
        if not pattern:
            return out
        for i in range(self.n_pages):
            rects = self.doc[i].search_for(pattern)  # case-insensitive default
            for r in rects:
                out.append((i, r))
        return out

    def close(self):
        self.doc.close()

# ------- Rule engine ----------
def run_checks(pages: List[str], meta: Dict[str, Any], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Very simple content checks: must_contain / reject_if_present
       + an example 'address_must_match_title' rule & 'forbidden_pairs' rule."""
    findings: List[Dict[str, Any]] = []

    text_all = "\n".join(pages)
    title_line = pages[0].splitlines()[0] if pages and pages[0] else ""

    checklist = rules.get("checklist", [])
    for rule in checklist:
        name = rule.get("name", "Unnamed rule")
        severity = rule.get("severity", "minor")
        ok = True
        notes = []

        must = rule.get("must_contain", []) or []
        for token in must:
            if token and token.lower() not in text_all.lower():
                ok = False
                notes.append(f"Missing token: '{token}'")

        rej = rule.get("reject_if_present", []) or []
        for token in rej:
            if token and token.lower() in text_all.lower():
                ok = False
                notes.append(f"Forbidden token present: '{token}'")

        # address must match title rule (skip if address contains ", 0 ,")
        if rule.get("id") == "address_matches_title":
            addr = (meta.get("site_address") or "").strip()
            if addr and ", 0 ," not in addr:
                if addr.lower() not in title_line.lower():
                    ok = False
                    notes.append("Site address not found in title line")

        # Example: forbidden_pairs mapping (e.g., if power template + contains Eltek+Polaradium -> must have note)
        if rule.get("id") == "forbidden_pairs":
            # expects:
            # pairs:
            # - if_project: "Power Resilience"
            #   if_text_all_has: ["Eltek", "Polaradium"]
            #   must_also_have: ["TDEE53201 section 3.8.1"]
            pairs = rule.get("pairs", []) or []
            for p in pairs:
                if_project = p.get("if_project")
                if if_project and meta.get("project") != if_project:
                    continue
                all_has = all(t.lower() in text_all.lower() for t in p.get("if_text_all_has", []))
                if all_has:
                    must_have = p.get("must_also_have", []) or []
                    for t in must_have:
                        if t.lower() not in text_all.lower():
                            ok = False
                            notes.append(f"Design mentions {p.get('if_text_all_has')} but missing '{t}'")

        findings.append({
            "rule": name,
            "severity": severity,
            "pass": ok,
            "notes": "; ".join(notes),
        })

    return findings

# ------- Annotation ----------
def annotate_pdf(pdf_bytes: bytes, findings: List[Dict[str, Any]], keywords: List[str]) -> bytes:
    """Add highlight boxes + sticky notes for the first occurrence of each keyword."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    index = PdfIndex(pdf_bytes)
    try:
        for kw in keywords:
            matches = index.search_exact(kw)
            if not matches:
                continue
            # mark first hit
            pno, rect = matches[0]
            page = doc[pno]
            page.add_highlight_annot(rect)
            page.add_text_annot(rect.br + (10, 10), kw)
        # also add a summary note on page 1 if any rejects
        rejected = [f for f in findings if not f["pass"]]
        if len(doc) > 0 and rejected:
            summary = "\n".join([f"• {f['rule']}: {f['notes']}" if f["notes"] else f"• {f['rule']}" for f in rejected])
            doc[0].add_text_annot(fitz.Rect(36, 36, 400, 200), f"Rejected findings:\n{summary}")
        out = io.BytesIO()
        doc.save(out)
        return out.getvalue()
    finally:
        index.close()
        doc.close()

# ------- Excel export --------
def make_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], base_pdf_name: str, status: str) -> bytes:
    df = pd.DataFrame(findings)
    meta_row = {**meta, "pdf_name": base_pdf_name, "status": status, "generated_utc": now_utc_iso()}
    mem = io.BytesIO()
    with pd.ExcelWriter(mem, engine="xlsxwriter") as xw:
        df.to_excel(xw, index=False, sheet_name="Findings")
        pd.DataFrame([meta_row]).to_excel(xw, index=False, sheet_name="Meta")
    return mem.getvalue()

# ------- History -------------
HISTORY_SCHEMA = [
    "timestamp_utc","supplier","client","project","site_type","vendor","cabinet_loc","radio_loc",
    "sectors","site_address","mimo_s1","mimo_s2","mimo_s3","mimo_s4","mimo_s5","mimo_s6",
    "status","pdf_name","excel_name","exclude"
]

def load_history() -> pd.DataFrame:
    ensure_dirs()
    if not os.path.exists(HISTORY_FILE) or os.path.getsize(HISTORY_FILE) == 0:
        return pd.DataFrame(columns=HISTORY_SCHEMA)
    try:
        df = pd.read_csv(HISTORY_FILE)
    except Exception:
        # salvage by ignoring bad lines
        df = pd.read_csv(HISTORY_FILE, on_bad_lines="skip")
    for c in HISTORY_SCHEMA:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    return df[HISTORY_SCHEMA]

def append_history(row: Dict[str, Any]):
    df = load_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

# ------- Gate ---------------
def gate():
    if "ok" in st.session_state:
        return
    st.title(APP_TITLE)
    st.caption("Access restricted")
    pw = st.text_input("Enter access password", type="password")
    if st.button("Enter"):
        if pw == ENTRY_PASSWORD:
            st.session_state.ok = True
            st.rerun()
        else:
            st.error("Wrong password")

# ------- Audit UI -----------
def audit_tab():
    st.subheader("Audit Metadata (all required)")
    c1, c2, c3 = st.columns(3)
    supplier = c1.selectbox("Supplier", SUPPLIERS)
    drawing_type = c2.selectbox("Drawing Type", ["General Arrangement", "Detailed Design"])
    client = c3.selectbox("Client", CLIENTS)

    c4, c5, c6 = st.columns(3)
    project = c4.selectbox("Project", PROJECTS)
    site_type = c5.selectbox("Site Type", SITE_TYPES)
    vendor = c6.selectbox("Proposed Vendor", VENDORS)

    c7, c8, c9 = st.columns(3)
    cabinet_loc = c7.selectbox("Proposed Cabinet Location", CAB_LOC)
    radio_loc = c8.selectbox("Proposed Radio Location", RADIO_LOC)
    sectors = c9.selectbox("Quantity of Sectors", SECTORS)

    site_address = st.text_input("Site Address", help='If it contains ", 0 ," it is ignored for the title check.')

    st.markdown("### Proposed MIMO Config")
    use_all = st.checkbox("Use S1 for all sectors", value=True)

    # for Power Resilience, MIMO optional
    mimo_required = project != "Power Resilience"

    def mimo_picker(label_key):
        return st.selectbox(label_key, MIMO_OPTIONS, key=label_key)

    mimo_s1 = mimo_picker("MIMO S1")
    mimo_s2 = mimo_picker("MIMO S2") if not use_all and int(sectors) >= 2 else mimo_s1
    mimo_s3 = mimo_picker("MIMO S3") if not use_all and int(sectors) >= 3 else mimo_s1
    mimo_s4 = mimo_picker("MIMO S4") if not use_all and int(sectors) >= 4 else mimo_s1
    mimo_s5 = mimo_picker("MIMO S5") if not use_all and int(sectors) >= 5 else mimo_s1
    mimo_s6 = mimo_picker("MIMO S6") if not use_all and int(sectors) >= 6 else mimo_s1

    st.markdown("### Upload PDF Design")
    up = st.file_uploader("PDF only", type=["pdf"])
    exclude = st.checkbox("Exclude this review from analytics", value=False)

    # Run
    if st.button("Run audit", type="primary", use_container_width=True, disabled=up is None):
        if up is None:
            st.warning("Please upload a PDF.")
            return
        if mimo_required and not mimo_s1:
            st.error("MIMO S1 is required for this project.")
            return

        meta = dict(
            supplier=supplier, drawing_type=drawing_type, client=client, project=project,
            site_type=site_type, vendor=vendor, cabinet_loc=cabinet_loc, radio_loc=radio_loc,
            sectors=sectors, site_address=site_address.strip(),
            mimo_s1=mimo_s1, mimo_s2=mimo_s2, mimo_s3=mimo_s3,
            mimo_s4=mimo_s4, mimo_s5=mimo_s5, mimo_s6=mimo_s6,
        )

        rules = load_rules(RULES_FILE_DEFAULT)
        raw = up.read()
        with st.status("Running checks…", expanded=True) as status:
            pages = extract_all_text(raw)
            findings = run_checks(pages, meta, rules)
            # status
            n_fail = sum(1 for f in findings if not f["pass"])
            overall = "PASS" if n_fail == 0 else "REJECTED"
            status.update(label=f"Checks complete: {overall}", state="complete")

        with st.status("Annotating PDF…", expanded=True) as s2:
            # Collect some keywords to mark (names of failing rules or tokens)
            keywords = []
            for r in rules.get("checklist", []):
                keywords.extend(r.get("must_contain", []) or [])
                # try to highlight forbidden tokens too
                keywords.extend(r.get("reject_if_present", []) or [])
            annotated = annotate_pdf(raw, findings, list(dict.fromkeys([k for k in keywords if isinstance(k, str)])))
            s2.update(label="Annotations added", state="complete")

        # Exports
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = os.path.splitext(up.name)[0]
        status_str = "PASS" if n_fail == 0 else "REJECTED"
        excel_bytes = make_excel(findings, meta, up.name, status_str)

        pdf_out_name = f"{safe_filename(base)}_{status_str}_{stamp}.annotated.pdf"
        xls_out_name = f"{safe_filename(base)}_{status_str}_{stamp}.xlsx"

        cdl, cdr = st.columns(2)
        cdl.download_button("⬇️ Download annotated PDF", data=annotated, file_name=pdf_out_name, mime="application/pdf")
        cdr.download_button("⬇️ Download Excel report", data=excel_bytes, file_name=xls_out_name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Persist to history
        append_history({
            "timestamp_utc": now_utc_iso(),
            "supplier": supplier, "client": client, "project": project, "site_type": site_type,
            "vendor": vendor, "cabinet_loc": cabinet_loc, "radio_loc": radio_loc,
            "sectors": sectors, "site_address": site_address.strip(),
            "mimo_s1": mimo_s1, "mimo_s2": mimo_s2, "mimo_s3": mimo_s3,
            "mimo_s4": mimo_s4, "mimo_s5": mimo_s5, "mimo_s6": mimo_s6,
            "status": status_str, "pdf_name": pdf_out_name, "excel_name": xls_out_name,
            "exclude": bool(exclude),
        })

        # simple table of findings
        st.markdown("#### Findings")
        st.dataframe(pd.DataFrame(findings), use_container_width=True)

# ------- Analytics -----------
def analytics_tab():
    st.subheader("Analytics & History")
    df = load_history()
    if df.empty:
        st.info("No history yet.")
        return

    # Filters
    c1, c2, c3, c4 = st.columns(4)
    f_client = c1.multiselect("Client", sorted(df["client"].dropna().unique().tolist()))
    f_project = c2.multiselect("Project", sorted(df["project"].dropna().unique().tolist()))
    f_supplier = c3.multiselect("Supplier", sorted(df["supplier"].dropna().unique().tolist()))
    f_status = c4.multiselect("Status", sorted(df["status"].dropna().unique().tolist()))

    show = df.copy()
    show = show[show["exclude"] != True]  # exclude flagged
    if f_client:
        show = show[show["client"].isin(f_client)]
    if f_project:
        show = show[show["project"].isin(f_project)]
    if f_supplier:
        show = show[show["supplier"].isin(f_supplier)]
    if f_status:
        show = show[show["status"].isin(f_status)]

    # RFT
    total = len(show)
    rft = (show["status"] == "PASS").mean() * 100 if total else 0.0
    st.metric("Right First Time (RFT)", f"{rft:.1f}%")

    # table (only columns that exist)
    wanted = ["timestamp_utc","supplier","client","project","status","pdf_name","excel_name"]
    cols = [c for c in wanted if c in show.columns]
    st.dataframe(show[cols], use_container_width=True, height=320)

# ------- Settings -----------
def settings_tab():
    st.subheader("Settings")
    st.caption("Place your logo file in the repo root (png/svg/jpg).")

    # rules editor
    st.markdown("### Edit rules (YAML)")
    with st.expander("Show / edit rules", expanded=True):
        pw = st.text_input("Rules password", type="password")
        path = st.text_input("Rules file", value=RULES_FILE_DEFAULT)
        raw = ""
        try:
            raw = open(path, "r", encoding="utf-8").read() if os.path.exists(path) else "checklist: []"
        except Exception as e:
            st.error(f"Failed to load rules: {e}")

        area = st.text_area(os.path.basename(path), value=raw, height=260)
        c1, c2 = st.columns(2)
        if c1.button("Save rules"):
            if pw != RULES_PASSWORD:
                st.error("Wrong rules password.")
            else:
                try:
                    yaml.safe_load(area)  # validate
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(area)
                    st.success("Rules saved.")
                except Exception as e:
                    st.error(f"Invalid YAML: {e}")
        if c2.button("Reload from disk"):
            st.rerun()

    # History housekeeping
    st.markdown("### History")
    if st.button("Clear all history (keeps files)"):
        try:
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.success("History cleared.")
        except Exception as e:
            st.error(f"Failed to clear history: {e}")

# --------------- Main ---------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(load_logo_html(), unsafe_allow_html=True)

    if "ok" not in st.session_state:
        gate()
        return

    st.title(APP_TITLE)

    tab1, tab2, tab3 = st.tabs(["Audit", "Analytics", "Settings"])
    with tab1:
        audit_tab()
    with tab2:
        analytics_tab()
    with tab3:
        settings_tab()

if __name__ == "__main__":
    ensure_dirs()
    main()
