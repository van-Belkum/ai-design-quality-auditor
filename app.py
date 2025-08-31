# app.py
import io
import os
import re
import zipfile
import base64
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import yaml
from rapidfuzz import fuzz, process
from pdf2image import convert_from_bytes
import pytesseract
import fitz  # PyMuPDF (for annotation)

# ============================ CONFIG ============================
APP_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"

REPO_LOGO = "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png"  # put in repo root
HISTORY_DIR = "history"
HISTORY_CSV = os.path.join(HISTORY_DIR, "history.csv")
TRAINING_CSV = os.path.join(HISTORY_DIR, "training.csv")
RULES_FILE = "rules_example.yaml"
RULES_DELTA_FILE = "rules_delta.yaml"  # incremental adds from training

# dropdowns
SUPPLIERS = ["CEG","CTIL","Emfyser","Innov8","Invict","KTL Team (Internal)","Trylon"]
DRAWING_TYPES = ["General Arrangement","Detailed Design"]
CLIENTS = ["BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"]
PROJECTS = ["RAN","Power Resilience","East Unwind","Beacon 4"]
SITE_TYPES = ["Greenfield","Rooftop","Streetworks"]
VENDORS = ["Ericsson","Nokia"]
CAB_LOCS = ["Indoor","Outdoor"]
RADIO_LOCS = ["Low Level","Midway","High Level","Unique Coverage"]
SECTOR_RANGE = [1,2,3,4,5,6]

MIMO_OPTIONS = [
    "18 @2x2","18 @2x2; 26 @4x4","18 @2x2; 70\\80 @2x2","18 @2x2; 80 @2x2",
    "18\\21 @2x2","18\\21 @2x2; 26 @4x4","18\\21 @2x2; 3500 @32x32","18\\21 @2x2; 70\\80 @2x2",
    "18\\21 @2x2; 80 @2x2","18\\21 @4x4","18\\21 @4x4; 3500 @32x32","18\\21 @4x4; 70 @2x4",
    "18\\21 @4x4; 70\\80 @2x2","18\\21 @4x4; 70\\80 @2x2; 3500 @32x32","18\\21 @4x4; 70\\80 @2x4",
    "18\\21 @4x4; 70\\80 @2x4; 3500 @32x32","18\\21 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21@4x4; 70\\80 @2x2","18\\21@4x4; 70\\80 @2x4","18\\21\\26 @2x2",
    "18\\21\\26 @2x2; 3500 @32x32","18\\21\\26 @2x2; 3500 @8X8","18\\21\\26 @2x2; 70\\80 @2x2",
    "18\\21\\26 @2x2; 70\\80 @2x2; 3500 @32x32","18\\21\\26 @2x2; 70\\80 @2x2; 3500 @8x8",
    "18\\21\\26 @2x2; 70\\80 @2x4; 3500 @32x32","18\\21\\26 @2x2; 80 @2x2",
    "18\\21\\26 @2x2; 80 @2x2; 3500 @8x8","18\\21\\26 @4x4","18\\21\\26 @4x4; 3500 @32x32",
    "18\\21\\26 @4x4; 3500 @8x8","18\\21\\26 @4x4; 70 @2x4; 3500 @8x8",
    "18\\21\\26 @4x4; 70\\80 @2x2","18\\21\\26 @4x4; 70\\80 @2x2; 3500 @32x32",
    "18\\21\\26 @4x4; 70\\80 @2x2; 3500 @8x8","18\\21\\26 @4x4; 70\\80 @2x4",
    "18\\21\\26 @4x4; 70\\80 @2x4; 3500 @32x32","18\\21\\26 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21\\26 @4x4; 80 @2x2","18\\21\\26 @4x4; 80 @2x2; 3500 @32x32",
    "18\\21\\26 @4x4; 80 @2x4","18\\21\\26 @4x4; 80 @2x4; 3500 @8x8",
    "18\\26 @2x2","18\\26 @4x4; 21 @2x2; 80 @2x2","(blank)"
]

ALLOW_WORDS = {"the","and","to","of","for","on","at","by","level","radio","cabinet","site","address","mimo","sector"}

# ======================= HELPERS / IO ==========================
def utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()

def ensure_dirs():
    os.makedirs(HISTORY_DIR, exist_ok=True)

def read_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception:
        return {"checklist": []}

def write_yaml(path: str, data: Dict[str, Any]):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        # last resort
        return pd.DataFrame()

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def logo_html() -> str:
    if os.path.exists(REPO_LOGO):
        with open(REPO_LOGO, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"""
        <div class="sticky-logo"><img src="data:image/png;base64,{b64}" style="height:44px;"></div>
        <style>
            .sticky-logo{{position:fixed;left:18px;top:12px;z-index:1000;}}
            header[data-testid="stHeader"] {{ background: transparent; }}
        </style>
        """
    return ""

# ======================= AUDIT FUNCTIONS =======================
def text_from_pdf(pdf_bytes: bytes) -> List[str]:
    # OCR every page for robust search
    images = convert_from_bytes(pdf_bytes)
    pages = []
    for img in images:
        txt = pytesseract.image_to_string(img)
        pages.append(txt)
    return pages

def check_site_address_against_filename(site_address: str, filename: str) -> Optional[Dict[str, Any]]:
    # Ignore ', 0 ,' fragments like "MANBY ROAD , 0 , IMMINGHAM ..."
    cleaned = re.sub(r"\s*,\s*0\s*,\s*", ", ", site_address, flags=re.I).strip()
    base = os.path.splitext(os.path.basename(filename))[0]
    score = fuzz.token_set_ratio(cleaned.upper(), base.upper())
    if score < 80:
        return {
            "rule_id": "SiteAddressTitleMatch",
            "severity": "major",
            "message": f"Site address '{cleaned}' does not match drawing name '{base}' (score {score}).",
            "page_num": 0
        }
    return None

def run_yaml_rules(pages: List[str], meta: Dict[str, Any], rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    # simple scoping: project/vendor/site_type can be attached to a rule under 'scope'
    for rule in rules:
        scope = rule.get("scope", {})
        # filter by scope if provided
        ok = True
        for k,v in scope.items():
            if str(meta.get(k, "")).lower() != str(v).lower():
                ok = False
                break
        if not ok:  # rule not applicable
            continue

        must = rule.get("must_contain", []) or []
        reject_if = rule.get("reject_if_present", []) or []
        name = rule.get("name","unnamed")
        sev = rule.get("severity","minor")

        # evaluate each page
        for i, page in enumerate(pages):
            page_up = page.upper()
            # missing required text
            for m in must:
                if m and str(m).upper() not in page_up:
                    findings.append({"rule_id": name, "severity": sev, "message": f"Missing required '{m}'", "page_num": i})
            # forbidden text
            for bad in reject_if:
                if bad and str(bad).upper() in page_up:
                    findings.append({"rule_id": name, "severity": sev, "message": f"Forbidden text '{bad}' present", "page_num": i})
    return findings

def spelling_check(pages: List[str]) -> List[Dict[str, Any]]:
    out = []
    for i, page in enumerate(pages):
        for raw in re.findall(r"[A-Za-z]{3,}", page):
            w = raw.lower()
            if w in ALLOW_WORDS:
                continue
            # very light suggestion using fuzzy to common allow list
            suggestion = process.extractOne(w, list(ALLOW_WORDS))
            msg = f"Possible misspelling: '{raw}'"
            if suggestion:
                msg += f" (did you mean '{suggestion[0]}')"
            out.append({"rule_id":"Spelling","severity":"minor","message":msg,"page_num":i})
    return out

def annotate_pdf(pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    doc = fitz.open("pdf", pdf_bytes)
    for f in findings:
        pg = f.get("page_num", 0)
        msg = f.get("message", "")
        if 0 <= pg < len(doc):
            page = doc[pg]
            # try to find a short keyword from message for highlight
            key = None
            m = re.search(r"'([^']+)'", msg)
            if m:
                key = m.group(1)
            elif len(msg) > 6:
                key = msg[:12]
            added = False
            if key:
                for rect in page.search_for(key)[:5]:
                    annot = page.add_rect_annot(rect)
                    annot.set_colors({"stroke": (1,0,0)})
                    annot.set_border(width=1)
                    annot.update()
                    added = True
            if not added:
                # add a page note near top-left
                page.add_text_annot(fitz.Point(36, 36), msg)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

def make_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], original_name: str, status: str) -> bytes:
    df = pd.DataFrame(findings) if findings else pd.DataFrame(columns=["rule_id","severity","message","page_num"])
    meta_df = pd.DataFrame([{**meta,"original_file":original_name,"status":status,"generated_utc":utc_iso()}])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Findings")
        meta_df.to_excel(w, index=False, sheet_name="Metadata")
    buf.seek(0)
    return buf.getvalue()

def make_zip(excel_bytes: bytes, pdf_bytes: bytes, base_name: str) -> bytes:
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base_name}.xlsx", excel_bytes)
        zf.writestr(f"{base_name}.pdf", pdf_bytes)
    zbuf.seek(0)
    return zbuf.getvalue()

# ============================ UI ===============================
st.set_page_config(page_title="AI Design Quality Auditor", layout="wide")
ensure_dirs()
st.markdown(logo_html(), unsafe_allow_html=True)

# Gate
if "auth" not in st.session_state:
    st.session_state["auth"] = False
if not st.session_state["auth"]:
    pw = st.text_input("Enter password", type="password")
    if st.button("Login"):
        st.session_state["auth"] = (pw == APP_PASSWORD)
        if not st.session_state["auth"]:
            st.error("Wrong password")
    st.stop()

st.title("AI Design Quality Auditor")
st.caption("Professional, consistent design QA with audit trail, annotations, analytics, and rapid rule updates.")

tab_audit, tab_history, tab_analytics, tab_training, tab_settings = st.tabs(
    ["Audit","History","Analytics","Training","Settings"]
)

# --------------------------- AUDIT -----------------------------
with tab_audit:
    st.subheader("Audit Metadata (required)")
    c1,c2,c3,c4 = st.columns(4)
    supplier = c1.selectbox("Supplier", SUPPLIERS)
    drawing_type = c2.selectbox("Drawing Type", DRAWING_TYPES)
    client = c3.selectbox("Client", CLIENTS)
    project = c4.selectbox("Project", PROJECTS)

    c5,c6,c7,c8 = st.columns(4)
    site_type = c5.selectbox("Site Type", SITE_TYPES)
    vendor = c6.selectbox("Proposed Vendor", VENDORS)
    cab_loc = c7.selectbox("Proposed Cabinet Location", CAB_LOCS)
    radio_loc = c8.selectbox("Proposed Radio Location", RADIO_LOCS)

    c9,c10 = st.columns(2)
    qty_sectors = c9.selectbox("Quantity of Sectors", SECTOR_RANGE)
    site_address = c10.text_input("Site Address", placeholder="e.g. MANBY ROAD , 0 , IMMINGHAM , DN40 2LQ")

    # MIMO group (optional if Power Resilience)
    mimo_optional = (project == "Power Resilience")
    with st.expander("Proposed MIMO Config (optional for Power Resilience only)" if mimo_optional else "Proposed MIMO Config", expanded=not mimo_optional):
        same_for_all = st.checkbox("Use S1 for all sectors", value=True, key="mimo_same")
        sector_cfg: Dict[str,str] = {}
        for i in range(1, qty_sectors+1):
            if same_for_all and i>1:
                sector_cfg[f"S{i}"] = sector_cfg["S1"]
                st.text_input(f"MIMO S{i}", value=sector_cfg["S1"], key=f"mimo_copy_{i}", disabled=True)
            else:
                sector_cfg[f"S{i}"] = st.selectbox(f"MIMO S{i}", MIMO_OPTIONS, key=f"mimo_sel_{i}")

    up = st.file_uploader("Upload PDF design", type=["pdf"])
    exclude_now = st.checkbox("Exclude this audit from analytics", value=False)

    if up and st.button("Run Audit"):
        raw_pdf = up.read()
        pages = text_from_pdf(raw_pdf)

        # baseline checks
        findings: List[Dict[str,Any]] = []
        addr_mismatch = check_site_address_against_filename(site_address, up.name)
        if addr_mismatch:
            findings.append(addr_mismatch)

        # YAML rules
        full_rules = read_yaml(RULES_FILE)
        rules = full_rules.get("checklist", [])
        meta_scope = {
            "client": client,
            "project": project,
            "site_type": site_type,
            "vendor": vendor,
            "drawing_type": drawing_type,
        }
        findings += run_yaml_rules(pages, meta_scope, rules)

        # Spelling
        findings += spelling_check(pages)

        status = "Rejected" if len(findings) else "Pass"

        # artifacts
        annotated_pdf = annotate_pdf(raw_pdf, findings)
        excel_bytes = make_excel(findings, {
            "supplier": supplier, "drawing_type": drawing_type, "client": client,
            "project": project, "site_type": site_type, "vendor": vendor,
            "cab_loc": cab_loc, "radio_loc": radio_loc, "qty_sectors": qty_sectors,
            "site_address": site_address, **{k:v for k,v in sector_cfg.items()}
        }, up.name, status)

        base_out = f"{os.path.splitext(up.name)[0]}_{status}_{datetime.now().strftime('%Y-%m-%d')}"
        st.download_button("⬇️ Download Excel", excel_bytes, file_name=f"{base_out}.xlsx")
        st.download_button("⬇️ Download Annotated PDF", annotated_pdf, file_name=f"{base_out}.pdf")
        st.download_button("⬇️ Download ZIP (both)", make_zip(excel_bytes, annotated_pdf, base_out), file_name=f"{base_out}.zip")

        # persist history
        dfh = load_csv_safe(HISTORY_CSV)
        row = {
            "timestamp_utc": utc_iso(),
            "supplier": supplier,
            "drawing_type": drawing_type,
            "client": client,
            "project": project,
            "site_type": site_type,
            "vendor": vendor,
            "cab_loc": cab_loc,
            "radio_loc": radio_loc,
            "qty_sectors": qty_sectors,
            "site_address": site_address,
            "status": status,
            "findings_count": len(findings),
            "exclude": exclude_now,
            "original_file": up.name,
            **{k:v for k,v in sector_cfg.items()},
        }
        dfh = pd.concat([dfh, pd.DataFrame([row])], ignore_index=True)
        save_csv(dfh, HISTORY_CSV)

        st.success(f"Audit complete — {status}. Saved to history.")

# -------------------------- HISTORY ----------------------------
with tab_history:
    st.subheader("Audit history")
    dfh = load_csv_safe(HISTORY_CSV)
    if dfh.empty:
        st.info("No history yet.")
    else:
        st.dataframe(dfh, use_container_width=True)
        idx = st.number_input("Row index", min_value=0, max_value=len(dfh)-1, value=0, step=1)
        colA, colB = st.columns(2)
        if colA.button("Toggle exclude/include"):
            dfh.loc[idx, "exclude"] = not bool(dfh.loc[idx, "exclude"])
            save_csv(dfh, HISTORY_CSV)
            st.experimental_rerun()
        if colB.button("Clear ALL history (keeps files)"):
            save_csv(pd.DataFrame(), HISTORY_CSV)
            st.experimental_rerun()

# ------------------------- ANALYTICS ---------------------------
with tab_analytics:
    st.subheader("Analytics")
    df = load_csv_safe(HISTORY_CSV)
    if df.empty:
        st.info("No data to analyze yet.")
    else:
        df = df[df.get("exclude", False) != True].copy()
        supplier_filter = st.selectbox("Filter by supplier", ["All"] + SUPPLIERS)
        if supplier_filter != "All":
            df = df[df["supplier"] == supplier_filter]
        st.metric("Total audits", len(df))
        if len(df) > 0:
            pass_rate = (df["status"] == "Pass").mean() * 100
            st.metric("Pass rate", f"{pass_rate:.1f}%")
            # RFT = first time pass ratio (proxy: Pass status)
            st.metric("Right First Time (RFT)", f"{pass_rate:.1f}%")

        c1, c2 = st.columns(2)
        if len(df) > 0:
            by_status = df["status"].value_counts().to_frame("count")
            c1.bar_chart(by_status)

            by_supplier = df.groupby("supplier")["status"].apply(lambda s: (s=="Pass").mean()*100).sort_values(ascending=False)
            c2.bar_chart(by_supplier)

# -------------------------- TRAINING ---------------------------
with tab_training:
    st.subheader("Reviewer training & rapid rule growth")
    st.caption("Upload a previous audit export (Excel) and/or annotated PDF. Mark it Valid/Not Valid and optionally add a quick new rule.")
    colT1, colT2 = st.columns(2)
    excel_up = colT1.file_uploader("Upload audit Excel (optional)", type=["xlsx"])
    pdf_up = colT2.file_uploader("Upload annotated PDF (optional)", type=["pdf"])
    verdict = st.selectbox("Verdict for this audit", ["Valid (correct decision)","Not Valid (false positive/negative)"])
    note = st.text_area("Reviewer notes (why / what to learn)")

    # quick rule add
    st.markdown("##### Add a quick rule from this example")
    q1,q2,q3,q4,q5 = st.columns(5)
    r_name = q1.text_input("Rule name", value="New field must exist")
    r_sev = q2.selectbox("Severity", ["minor","major"], index=0)
    r_scope_proj = q3.selectbox("Scope: Project (optional)", ["", *PROJECTS])
    r_scope_vend = q4.selectbox("Scope: Vendor (optional)", ["", *VENDORS])
    r_scope_type = q5.selectbox("Scope: Site Type (optional)", ["", *SITE_TYPES])
    must_txt = st.text_input("Must contain (comma-sep)", value="")
    forbid_txt = st.text_input("Reject if present (comma-sep)", value="")
    add_rule_now = st.checkbox("Stage this rule to rules_delta.yaml")

    if st.button("Save training record"):
        tdf = load_csv_safe(TRAINING_CSV)
        trow = {
            "timestamp_utc": utc_iso(),
            "verdict": verdict,
            "note": note,
        }
        tdf = pd.concat([tdf, pd.DataFrame([trow])], ignore_index=True)
        save_csv(tdf, TRAINING_CSV)
        st.success("Training item saved.")

        if add_rule_now:
            delta = read_yaml(RULES_DELTA_FILE)
            if "checklist" not in delta: delta["checklist"] = []
            scope = {}
            if r_scope_proj: scope["project"] = r_scope_proj
            if r_scope_vend: scope["vendor"] = r_scope_vend
            if r_scope_type: scope["site_type"] = r_scope_type
            delta["checklist"].append({
                "name": r_name,
                "severity": r_sev,
                "scope": scope,
                "must_contain": [s.strip() for s in must_txt.split(",") if s.strip()],
                "reject_if_present": [s.strip() for s in forbid_txt.split(",") if s.strip()],
            })
            write_yaml(RULES_DELTA_FILE, delta)
            st.success("Rule staged in rules_delta.yaml. Promote it in Settings when ready.")

# -------------------------- SETTINGS ---------------------------
with tab_settings:
    st.subheader("Rules & housekeeping")
    pw = st.text_input("Rules password", type="password")
    if pw == RULES_PASSWORD:
        st.success("Rules editor unlocked.")
        # merged preview info
        base_rules = read_yaml(RULES_FILE)
        delta_rules = read_yaml(RULES_DELTA_FILE)
        st.caption("Below is the contents of your main rules file.")
        main_content = st.text_area("rules_example.yaml", value=yaml.safe_dump(base_rules, sort_keys=False), height=280)
        if st.button("Save rules_example.yaml"):
            try:
                new_obj = yaml.safe_load(main_content) or {"checklist":[]}
                write_yaml(RULES_FILE, new_obj)
                st.success("Main rules saved.")
            except Exception as e:
                st.error(f"YAML error: {e}")

        st.caption("Staged rules (rules_delta.yaml) – these can be promoted into main rules.")
        staged_content = st.text_area("rules_delta.yaml", value=yaml.safe_dump(delta_rules, sort_keys=False), height=240)
        cA, cB = st.columns(2)
        if cA.button("Save staged file"):
            try:
                new_obj = yaml.safe_load(staged_content) or {"checklist":[]}
                write_yaml(RULES_DELTA_FILE, new_obj)
                st.success("Staged rules saved.")
            except Exception as e:
                st.error(f"YAML error: {e}")
        if cB.button("Promote staged → main (append)"):
            main = read_yaml(RULES_FILE)
            staged = read_yaml(RULES_DELTA_FILE)
            mlist = main.get("checklist", [])
            mlist.extend(staged.get("checklist", []))
            main["checklist"] = mlist
            write_yaml(RULES_FILE, main)
            write_yaml(RULES_DELTA_FILE, {"checklist":[]})
            st.success("Staged rules appended to main and cleared.")

    if st.button("Clear all history (keeps yaml & files)"):
        save_csv(pd.DataFrame(), HISTORY_CSV)
        st.success("History cleared.")
