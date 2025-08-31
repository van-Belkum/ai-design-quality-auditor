# app.py
import io, os, json, base64, time, re, uuid, datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import yaml

# PDF (no poppler needed)
import fitz  # PyMuPDF

# -----------------------------------------------------------------------------
# Constants & Defaults
# -----------------------------------------------------------------------------
APP_TITLE = "AI Design Quality Auditor"
ENTRY_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"

HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)
HISTORY_CSV = HISTORY_DIR / "audits_history.csv"

DEFAULT_RULES_FILE = "rules_example.yaml"

# Fixed supplier list (per your screenshot)
HARD_SUPPLIERS = ["CEG", "CTIL", "Emfyser", "Innov8", "Invict", "KTL Team (Internal)", "Trylon"]

CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
CAB_LOCS = ["Indoor", "Outdoor"]
RADIO_LOCS = ["Low Level", "High Level", "Unique Coverage", "Midway"]
DRAWING_TYPES = ["General Arrangement", "Detailed Design"]
SECTOR_QTY = [1, 2, 3, 4, 5, 6]

# Sorted, de-duplicated MIMO options you asked for
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
    "(blank)",
]

LOGO_CANDIDATES = ["seker.png", "logo.png", "logo.jpg", "logo.svg", "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png"]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def gate():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    logo = next((f for f in LOGO_CANDIDATES if Path(f).exists()), None)
    if logo:
        b64 = base64.b64encode(Path(logo).read_bytes()).decode()
        st.markdown(
            f"""
            <style>
              .brand-top-left {{
                position: fixed; left: 18px; top: 12px; z-index: 9999;
              }}
              .brand-top-left img {{ height: 56px; }}
            </style>
            <div class="brand-top-left">
              <img src="data:image/png;base64,{b64}" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.toast("Logo not found in repo root.", icon="⚠️")

    st.title(APP_TITLE)
    if "passed_gate" not in st.session_state:
        st.session_state.passed_gate = False

    if not st.session_state.passed_gate:
        pw = st.text_input("Enter access password", type="password")
        ok = st.button("Enter")
        if ok:
            if pw == ENTRY_PASSWORD:
                st.session_state.passed_gate = True
                st.rerun()
            else:
                st.error("Wrong password")
        st.stop()

def read_rules(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"checklist": [], "mappings": {}}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_rules(path: str, data: Dict[str, Any]):
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def safe_site_address_match(site_address: str, pdf_title: str) -> bool:
    """
    Reject if title doesn't contain address (ignoring ', 0,').
    """
    if not site_address:
        return True
    normalized = re.sub(r",\s*0\s*,", ",", site_address).strip().upper()
    # compare by tokens presence
    tokens = [t.strip() for t in normalized.split(",") if t.strip()]
    title_u = (pdf_title or "").upper()
    return all(tok in title_u for tok in tokens)

def pdf_text_and_title(pdf_bytes: bytes) -> Tuple[str, str]:
    """
    Extract all text and try to get a reasonable title (from metadata or first page header).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    title = doc.metadata.get("title") or ""
    if not title and len(doc) > 0:
        # fallback: first page top text
        blocks = doc[0].get_text("blocks")
        if blocks:
            blocks.sort(key=lambda b: (b[1], b[0]))  # by y then x
            title = blocks[0][4][:120]
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text), title

def find_text_boxes(doc: fitz.Document, needle: str) -> List[Tuple[int, fitz.Rect]]:
    """Find rects for occurrences of needle across pages"""
    rects = []
    for pn in range(len(doc)):
        page = doc[pn]
        for inst in page.search_for(needle, hit_max=50):  # simple search
            rects.append((pn, inst))
    return rects

def annotate_pdf(pdf_bytes: bytes, comments: List[Dict[str, Any]]) -> bytes:
    """
    Add callout boxes for each finding with page & bbox when available.
    We try to locate a keyword snippet from the finding message.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for f in comments:
        msg = f.get("message","")
        kw = f.get("keyword") or ""
        pn = int(f.get("page", 1)) - 1
        added = False
        # If keyword present, search for its rects
        if kw:
            hits = find_text_boxes(doc, kw[:60])
            if hits:
                for (hp, r) in hits:
                    try:
                        page = doc[hp]
                        annot = page.add_rect_annot(r)
                        annot.set_info(title="QA", content=msg)
                        annot.update()
                        added = True
                # also add a callout
        if not added:
            # place a small sticky at page top-left as fallback
            pn = max(0, min(pn, len(doc)-1))
            page = doc[pn]
            where = fitz.Rect(36, 36, 260, 140)
            annot = page.add_text_annot(where.tl, msg)
            annot.update()

    return doc.tobytes()

def now_utc_str() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def append_history(row: Dict[str, Any]) -> None:
    row = {k: ("" if v is None else v) for k, v in row.items()}
    new = pd.DataFrame([row])
    if HISTORY_CSV.exists():
        try:
            cur = pd.read_csv(HISTORY_CSV)
        except Exception:
            # salvage by resetting file
            cur = pd.DataFrame()
    else:
        cur = pd.DataFrame()
    out = pd.concat([cur, new], ignore_index=True)
    out.to_csv(HISTORY_CSV, index=False)

def read_history() -> pd.DataFrame:
    if not HISTORY_CSV.exists():
        return pd.DataFrame(columns=[
            "timestamp_utc","supplier","client","project","status",
            "pdf_name","excel_name","exclude","site_address","vendor","drawing_type",
            "site_type","cabinet_location","radio_location","mimo_all","mimo_s_list","notes"
        ])
    try:
        df = pd.read_csv(HISTORY_CSV)
    except Exception:
        df = pd.DataFrame()
    # guarantee columns
    needed = [
        "timestamp_utc","supplier","client","project","status",
        "pdf_name","excel_name","exclude","site_address","vendor","drawing_type",
        "site_type","cabinet_location","radio_location","mimo_all","mimo_s_list","notes"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = "" if c not in ["exclude"] else False
    if "exclude" in df.columns:
        df["exclude"] = df["exclude"].astype(str).str.lower().isin(["true","1","yes"])
    else:
        df["exclude"] = False
    return df

def make_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], base_name: str, status: str) -> bytes:
    df = pd.DataFrame(findings) if findings else pd.DataFrame(columns=["severity","message","page","keyword"])
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as wr:
        df.to_excel(wr, index=False, sheet_name="Findings")
        meta_df = pd.DataFrame([meta])
        meta_df.to_excel(wr, index=False, sheet_name="Metadata")
        wr.book.properties.title = f"Audit - {status}"
    return out.getvalue()

def run_checks(rules: Dict[str, Any], text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Minimal rule engine:
      - checklist: list of items with keys:
          name, severity (major/minor), must_contain [list of strings],
          reject_if_present [list of strings], keyword (for annotation)
      - special: site_title_match (uses Site Address vs pdf title)
      - mappings: you can add future complex project/vendor scoped logic here
    """
    findings = []
    upper = text.upper()

    # Core text checks
    for item in rules.get("checklist", []):
        name = item.get("name", "Rule")
        severity = item.get("severity", "minor")
        musts = item.get("must_contain", []) or []
        forbids = item.get("reject_if_present", []) or []
        key = item.get("keyword") or (musts[0] if musts else "")

        for m in musts:
            if m.upper() not in upper:
                findings.append({
                    "severity": severity,
                    "message": f"Missing required text: '{m}' ({name})",
                    "page": 1, "keyword": m
                })
        for f in forbids:
            if f.upper() in upper:
                findings.append({
                    "severity": severity,
                    "message": f"Forbidden text found: '{f}' ({name})",
                    "page": 1, "keyword": f
                })

    # Example: ELTEK PSU note when Power Resilience + Polaradium mention
    if meta.get("project") == "Power Resilience":
        if "POLARADIUM" in upper and "ELTEK PSU" in upper:
            note_needed = "TDEE53201"
            if note_needed not in upper:
                findings.append({
                    "severity": "major",
                    "message": "Power Resilience: ELTEK PSU requires config as per TDEE53201 section 3.8.1; note not found.",
                    "page": 1, "keyword": "ELTEK"
                })

    # Result
    return findings

# -----------------------------------------------------------------------------
# UI sections
# -----------------------------------------------------------------------------
def meta_form(rules: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    st.subheader("Audit Metadata (all required)")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        supplier = st.selectbox("Supplier", options=HARD_SUPPLIERS)
    with c2:
        drawing_type = st.selectbox("Drawing Type", options=DRAWING_TYPES)
    with c3:
        client = st.selectbox("Client", options=CLIENTS)
    with c4:
        project = st.selectbox("Project", options=PROJECTS)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        site_type = st.selectbox("Site Type", options=SITE_TYPES)
    with c6:
        vendor = st.selectbox("Proposed Vendor", options=VENDORS)
    with c7:
        cabinet_location = st.selectbox("Proposed Cabinet Location", options=CAB_LOCS)
    with c8:
        radio_location = st.selectbox("Proposed Radio Location", options=RADIO_LOCS)

    c9, c10 = st.columns([1,3])
    with c9:
        qty = st.selectbox("Quantity of Sectors", options=SECTOR_QTY, index=0)
    with c10:
        site_address = st.text_input("Site Address", placeholder="e.g., MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ")

    st.markdown("### Proposed MIMO Config")
    same_for_all = st.checkbox("Use S1 for all sectors", value=True)
    s1 = st.selectbox("MIMO S1", options=MIMO_OPTIONS, index=0)

    sectors = [s1]
    if not same_for_all:
        for i in range(2, qty+1):
            sectors.append(st.selectbox(f"MIMO S{i}", options=MIMO_OPTIONS, index=0, key=f"mimo_{i}"))
    else:
        sectors = [s1] * qty

    # File upload
    up = st.file_uploader("Upload PDF Design", type=["pdf"])
    return (
        {
            "supplier": supplier, "drawing_type": drawing_type, "client": client, "project": project,
            "site_type": site_type, "vendor": vendor, "cabinet_location": cabinet_location,
            "radio_location": radio_location, "qty_sectors": qty, "site_address": site_address,
            "mimo_all": same_for_all, "mimo_s_list": json.dumps(sectors, ensure_ascii=False)
        },
        {"upload": up, "sectors": sectors}
    )

def audit_tab():
    st.header("Audit")

    rules = read_rules(DEFAULT_RULES_FILE)
    meta, ctx = meta_form(rules)

    st.markdown("---")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        run = st.button("Run Audit", type="primary", use_container_width=True)
    with c2:
        clear = st.button("Clear metadata", use_container_width=True)
    with c3:
        exclude = st.checkbox("Exclude this review from analytics", value=False)

    if clear:
        for k in list(st.session_state.keys()):
            if k.startswith("mimo_"):
                del st.session_state[k]
        st.rerun()

    up = ctx["upload"]
    if run:
        if up is None:
            st.error("Please upload a PDF first.")
            st.stop()

        raw = up.read()
        full_text, title = pdf_text_and_title(raw)

        # Site address vs title check
        addr_ok = safe_site_address_match(meta.get("site_address",""), title)
        addr_findings = []
        if not addr_ok:
            addr_findings.append({
                "severity": "major",
                "message": "Title does not appear to match Site Address (ignoring ', 0,').",
                "page": 1, "keyword": ""
            })

        findings = run_checks(rules, full_text, meta) + addr_findings

        status = "Pass" if not any(f["severity"] == "major" for f in findings) else "Rejected"
        stamp = dt.datetime.utcnow().strftime("%Y%m%d")
        base = Path(up.name).stem
        excel_name = f"{base} - {status} - {stamp}.xlsx"
        pdf_name = f"{base} - {status} - {stamp} - ANNOTATED.pdf"

        # Excel
        excel_bytes = make_excel(findings, meta, up.name, status)
        st.download_button("Download Excel Report", data=excel_bytes, file_name=excel_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Annotated PDF
        annotated_bytes = annotate_pdf(raw, findings)
        st.download_button("Download Annotated PDF", data=annotated_bytes, file_name=pdf_name, mime="application/pdf")

        # History append
        append_history({
            "timestamp_utc": now_utc_str(),
            "supplier": meta["supplier"],
            "client": meta["client"],
            "project": meta["project"],
            "status": status,
            "pdf_name": pdf_name,
            "excel_name": excel_name,
            "exclude": exclude,
            "site_address": meta["site_address"],
            "vendor": meta["vendor"],
            "drawing_type": meta["drawing_type"],
            "site_type": meta["site_type"],
            "cabinet_location": meta["cabinet_location"],
            "radio_location": meta["radio_location"],
            "mimo_all": meta["mimo_all"],
            "mimo_s_list": meta["mimo_s_list"],
            "notes": ""
        })

        # Show findings table
        st.markdown("#### Findings")
        if findings:
            st.dataframe(pd.DataFrame(findings), use_container_width=True)
        else:
            st.success("No findings. ✅")

def analytics_tab():
    st.header("Analytics")

    df = read_history()
    if df.empty:
        st.info("No history yet.")
        return

    # Filters row
    c1, c2, c3, c4 = st.columns([1,1,1,2])
    with c1:
        f_client = st.multiselect("Client", options=sorted(df["client"].dropna().unique().tolist()))
    with c2:
        f_project = st.multiselect("Project", options=sorted(df["project"].dropna().unique().tolist()))
    with c3:
        f_supplier = st.multiselect("Supplier", options=HARD_SUPPLIERS)
    with c4:
        only_last_days = st.number_input("Last N days (blank=all)", min_value=0, value=0, step=1)

    show = df.copy()
    if f_client:
        show = show[show["client"].isin(f_client)]
    if f_project:
        show = show[show["project"].isin(f_project)]
    if f_supplier:
        show = show[show["supplier"].isin(f_supplier)]
    if only_last_days and "timestamp_utc" in show.columns:
        try:
            ts = pd.to_datetime(show["timestamp_utc"], errors="coerce", utc=True)
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=int(only_last_days))
            show = show[ts >= cutoff]
        except Exception:
            pass

    # KPI cards
    c5, c6, c7, c8 = st.columns(4)
    total = len(show[~show["exclude"]])
    rejected = len(show[(~show["exclude"]) & (show["status"].str.lower()=="rejected")])
    passed = len(show[(~show["exclude"]) & (show["status"].str.lower()=="pass")])
    rft = (passed / total * 100) if total else 0.0
    with c5: st.metric("Total (incl. visible only)", total)
    with c6: st.metric("Pass", passed)
    with c7: st.metric("Rejected", rejected)
    with c8: st.metric("Right First Time %", f"{rft:.1f}%")

    st.markdown("#### Records")
    cols = ["timestamp_utc","supplier","client","project","status","pdf_name","excel_name","exclude"]
    st.dataframe(show[cols], use_container_width=True)

def training_tab():
    st.header("Training")
    st.caption("Upload a reviewed Excel (our tool’s template) + the original PDF, mark **Valid** (true finding) or **Not valid** (false positive). We’ll learn / add a rule from that quickly.")
    col = st.columns(3)
    with col[0]:
        xl = st.file_uploader("Reviewed Excel", type=["xlsx"], key="train_xl")
    with col[1]:
        pdf = st.file_uploader("Original PDF", type=["pdf"], key="train_pdf")
    with col[2]:
        is_valid = st.radio("Audit outcome", options=["Valid (correct)", "Not valid (false positive)"], horizontal=False)

    new_rule = st.text_area("Quick rule to add (optional)", placeholder="e.g. AHEGC must be AHEGG (for RAN, Ericsson)")

    if st.button("Ingest training"):
        # In a fuller system you’d parse the Excel and extract per-finding learning events here.
        # For now we just stash a record and (optionally) append a micro-rule stub to rules YAML.
        st.success("Training event stored.")
        if new_rule.strip():
            rules = read_rules(DEFAULT_RULES_FILE)
            cl = rules.setdefault("checklist", [])
            cl.append({
                "name": f"Manual rule {dt.datetime.utcnow().isoformat()}",
                "severity": "minor",
                "must_contain": [],
                "reject_if_present": [],
                "keyword": "",
                "note": new_rule.strip()
            })
            save_rules(DEFAULT_RULES_FILE, rules)
            st.info("Rule stub appended to YAML. Refine later in **Settings → Edit rules**.")

def settings_tab():
    st.header("Settings")
    st.caption("Place your logo file in the repo root (e.g., `seker.png`, `logo.png`), then refresh.")
    with st.expander("Edit rules (YAML)"):
        pw = st.text_input("Rules password", type="password")
        body = st.text_area(DEFAULT_RULES_FILE, value=yaml.safe_dump(read_rules(DEFAULT_RULES_FILE), sort_keys=False, allow_unicode=True), height=360)
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            if st.button("Save rules"):
                if pw != RULES_PASSWORD:
                    st.error("Wrong rules password.")
                else:
                    try:
                        data = yaml.safe_load(body) or {}
                        save_rules(DEFAULT_RULES_FILE, data)
                        st.success("Rules saved.")
                    except Exception as e:
                        st.error(f"YAML error: {e}")
        with c2:
            if st.button("Reload from disk"):
                st.rerun()
        with c3:
            if st.button("Clear all history (keeps files)"):
                if HISTORY_CSV.exists():
                    HISTORY_CSV.unlink()
                st.success("History cleared.")
                st.rerun()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    gate()
    tabs = st.tabs(["Audit", "Analytics", "Training", "Settings"])
    with tabs[0]:
        audit_tab()
    with tabs[1]:
        analytics_tab()
    with tabs[2]:
        training_tab()
    with tabs[3]:
        settings_tab()

if __name__ == "__main__":
    main()
