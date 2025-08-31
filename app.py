# app.py
# AI Design Quality Auditor – streamlined, fast, and trainable
# Features: Audit (PDF text + optional OCR), sector MIMO metadata, site-address title match,
# YAML rules (scoped), spell-check (optional), PDF annotation, Excel report,
# History + analytics with filters and exclude toggle, Training (ingest audited CSV/XLS/XLSX + quick rules),
# Password gate + rules password, persistent downloads (until user clears).

import io
import os
import re
import json
import time
import base64
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import streamlit as st

# Core PDF text + annotation via PyMuPDF (no external poppler dependency)
import fitz  # PyMuPDF

# Optional libraries used if installed
try:
    from spellchecker import SpellChecker
except Exception:
    SpellChecker = None

# ---------- CONSTANTS / CONFIG ----------
APP_TITLE = "AI Design Quality Auditor"
BRAND_LOGO_FILE = os.getenv("SEKER_LOGO", "logo.png")  # put your file in repo root
ENTRY_PASSWORD = os.getenv("SEKER_ENTRY_PW", "Seker123")
RULES_PASSWORD = os.getenv("SEKER_RULES_PW", "vanB3lkum21")

HISTORY_DIR = "history"
HISTORY_FILE = os.path.join(HISTORY_DIR, "audit_history.csv")
EXPORT_DIR = "exports"            # long-term backups (daily dump)
PERSIST_DIR = "session_outputs"   # keep latest artifacts for this session
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

DEFAULT_RULES_FILE = "rules_example.yaml"

SUPPLIERS = [
    "CEG","CTIL","Emfyser","Innov8","Invict","KTL Team (Internal)","Trylon"
]

CLIENTS = ["BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"]
PROJECTS = ["RAN","Power Resilience","East Unwind","Beacon 4"]
SITE_TYPES = ["Greenfield","Rooftop","Streetworks"]
VENDORS = ["Ericsson","Nokia"]
CAB_LOCATIONS = ["Indoor","Outdoor"]
RADIO_LOCATIONS = ["Low Level","High Level","Unique Coverage","Midway"]  # corrected
DRAWING_TYPES = ["General Arrangement","Detailed Design"]

# Ordered MIMO options (as supplied)
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

DEFAULT_UI_VIS_HOURS = 24  # results stay visible in UI for 24h until user clears

# ---------- UTIL ----------
def logo_html():
    if os.path.exists(BRAND_LOGO_FILE):
        with open(BRAND_LOGO_FILE, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"""
        <div style="position:fixed;left:14px;top:14px;z-index:1000;">
          <img src="data:image/png;base64,{b64}" style="height:52px;opacity:0.95;">
        </div>
        """
    return ""

def save_bytes(path: str, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def timestamp_utc():
    return datetime.now(timezone.utc).replace(microsecond=0)

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

# ---------- DATA CLASSES ----------
@dataclass
class Finding:
    rule_name: str
    severity: str
    page: int
    text_hit: str
    comment: str
    bbox: Optional[List[float]]  # [x0,y0,x1,y1] on page
    category: str = "generic"

@dataclass
class Meta:
    supplier: str
    drawing_type: str
    client: str
    project: str
    site_type: str
    vendor: str
    cab_location: str
    radio_location: str
    sectors: int
    site_address: str
    mimo: Dict[str, str]

# ---------- PDF TEXT & SEARCH ----------
def pdf_text_by_page(pdf_bytes: bytes) -> List[str]:
    texts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for p in doc:
            texts.append(p.get_text("text"))
    return texts

def search_bboxes_case_insensitive(page: fitz.Page, needle: str) -> List[List[float]]:
    """Robust case-insensitive search: try native search_for; if it returns nothing,
    fallback to scanning word blocks and matching via regex to draw a union bbox."""
    if not needle or needle.strip() == "(blank)":
        return []
    bboxes = []
    # try native search_for with quads combine (case-sensitive), so we test both cases
    tries = [needle, needle.lower(), needle.upper()]
    for t in tries:
        try:
            hits = page.search_for(t, hit_max=128)  # returns rects
            bboxes.extend([[r.x0, r.y0, r.x1, r.y1] for r in hits])
        except Exception:
            pass
    if bboxes:
        return bboxes

    # fallback: regex over words
    words = page.get_text("words")  # list of (x0,y0,x1,y1, word, block_no, line_no, word_no)
    text = " ".join(w[4] for w in words)
    if re.search(re.escape(needle), text, flags=re.IGNORECASE):
        # naive bbox = full page margin area to at least mark page
        rect = page.rect
        return [[rect.x0 + 36, rect.y0 + 36, rect.x1 - 36, rect.y1 - 36]]
    return []

def annotate_pdf(original_pdf: bytes, findings: List[Finding]) -> bytes:
    if not findings:
        return original_pdf
    doc = fitz.open(stream=original_pdf, filetype="pdf")
    for f in findings:
        pno = max(0, min(f.page - 1, len(doc) - 1))
        page = doc[pno]
        boxes = []
        # if we’re given bbox from earlier, use it; else search by hit
        if f.bbox and len(f.bbox) == 4:
            boxes = [fitz.Rect(*f.bbox)]
        else:
            boxes = [fitz.Rect(*b) for b in search_bboxes_case_insensitive(page, f.text_hit)]
        if not boxes:
            # at least drop a sticky note near header
            annot = page.add_text_annot(page.rect.tl + (36, 36), f"{f.severity.upper()}: {f.rule_name}\n{f.comment}")
            annot.set_icon("Note")
            continue
        for r in boxes:
            hl = page.add_highlight_annot(r)
            hl.update()
            # add sticky note offset
            note_pt = fitz.Point(min(r.x1 + 8, page.rect.x1 - 24), max(r.y0 - 8, page.rect.y0 + 24))
            an = page.add_text_annot(note_pt, f"{f.severity.upper()}: {f.rule_name}\n{f.comment}")
            an.set_icon("Note")
    mem = io.BytesIO()
    doc.save(mem)
    doc.close()
    return mem.getvalue()

# ---------- SPELLING ----------
def spelling_findings(pages: List[str], allow_list: List[str]) -> List[Finding]:
    out: List[Finding] = []
    if not SpellChecker:
        return out
    sp = SpellChecker(distance=1)
    sp.word_frequency.load_words(w.lower() for w in allow_list)
    for idx, page in enumerate(pages, start=1):
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", page)
        for w in tokens:
            wl = w.lower()
            if wl in sp or wl in (x.lower() for x in allow_list):
                continue
            sug = next(iter(sp.candidates(wl)), None)
            out.append(Finding(
                rule_name="Spelling",
                severity="minor",
                page=idx,
                text_hit=w,
                comment=f"Suspicious word '{w}'. Suggest: {sug or 'check'}",
                bbox=None,
                category="spelling"
            ))
    return out

# ---------- RULES (YAML) ----------
def load_rules(yaml_path: str) -> Dict[str, Any]:
    import yaml
    if not os.path.exists(yaml_path):
        return {"checklist": []}
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "checklist" not in data:
        data["checklist"] = []
    return data

def append_rule_to_yaml(yaml_path: str, rule: Dict[str, Any], pw: str) -> str:
    if pw != RULES_PASSWORD:
        return "❌ Wrong rules password."
    import yaml
    data = load_rules(yaml_path)
    data.setdefault("checklist", []).append(rule)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return "✅ Rule appended."

def rule_matches_scope(rule_scope: Dict[str,str], meta: Meta) -> bool:
    # scope keys can be: client, project, vendor, site_type, radio_location, drawing_type, cab_location
    m = meta.__dict__
    for k,v in rule_scope.items():
        if not v: 
            continue
        mv = str(m.get(k if k!="vendor" else "vendor")).strip()
        if str(v).strip() != mv:
            return False
    return True

def run_rule_checks(pages: List[str], meta: Meta, rules: Dict[str, Any]) -> List[Finding]:
    out: List[Finding] = []
    checklist = rules.get("checklist", [])
    for rule in checklist:
        name = rule.get("name","Unnamed rule")
        severity = rule.get("severity","minor")
        must = rule.get("must_contain",[]) or []
        reject_if = rule.get("reject_if_present",[]) or []
        category = rule.get("category","rules")
        scope = rule.get("scope",{}) or {}
        # If scoped, ensure it matches
        if scope and (not rule_matches_scope(scope, meta)):
            continue
        # Evaluate per page
        for pno, text in enumerate(pages, start=1):
            t = text if isinstance(text, str) else ""
            hit = None
            ok = True
            for mkw in must:
                if mkw and not re.search(re.escape(mkw), t, flags=re.IGNORECASE):
                    ok = False
                    break
            if reject_if:
                for bad in reject_if:
                    if bad and re.search(re.escape(bad), t, flags=re.IGNORECASE):
                        ok = False
                        hit = bad
                        break
            if ok:
                # If there is at least one must keyword, mark the first as hit; else the rule is presence-only
                if must:
                    for mkw in must:
                        if re.search(re.escape(mkw), t, flags=re.IGNORECASE):
                            hit = mkw
                            break
                if not hit:
                    hit = name
                out.append(Finding(
                    rule_name=name,
                    severity=severity,
                    page=pno,
                    text_hit=hit,
                    comment=f"Rule matched: {name}",
                    bbox=None,
                    category=category
                ))
    return out

# ---------- EXCEL REPORT ----------
def make_excel(findings: List[Finding], meta: Meta, original_name: str, status: str) -> bytes:
    df = pd.DataFrame([asdict(f) for f in findings]) if findings else pd.DataFrame(
        columns=["rule_name","severity","page","text_hit","comment","bbox","category"]
    )
    meta_row = {**meta.__dict__}
    for k,v in list(meta_row.items()):
        if isinstance(v, dict):
            meta_row[k] = json.dumps(v, ensure_ascii=False)
    meta_df = pd.DataFrame([meta_row])

    mem = io.BytesIO()
    # prefer openpyxl if installed; otherwise xlsxwriter
    engine = "openpyxl"
    try:
        with pd.ExcelWriter(mem, engine=engine) as xw:
            meta_df.to_excel(xw, index=False, sheet_name="metadata")
            df.to_excel(xw, index=False, sheet_name="findings")
    except Exception:
        with pd.ExcelWriter(mem, engine="xlsxwriter") as xw:
            meta_df.to_excel(xw, index=False, sheet_name="metadata")
            df.to_excel(xw, index=False, sheet_name="findings")
    mem.seek(0)
    return mem.read()

# ---------- HISTORY ----------
def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=[
            "timestamp_utc","supplier","drawing_type","client","project","site_type",
            "vendor","cab_location","radio_location","sectors","site_address",
            "status","pdf_name","excel_name","exclude","findings_json"
        ])
    try:
        return pd.read_csv(HISTORY_FILE)
    except Exception:
        # try to recover corrupted row-wise
        rows = []
        with open(HISTORY_FILE,"r",encoding="utf-8",errors="ignore") as f:
            for line in f:
                try:
                    rows.append(next(pd.read_csv(io.StringIO(line)).to_dict("records")))
                except Exception:
                    pass
        df = pd.DataFrame(sum(rows, []))
        return df

def append_history(row: Dict[str, Any]):
    df = load_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

def daily_export_dump():
    # write a daily snapshot for safekeeping
    day = datetime.now().strftime("%Y%m%d")
    out = os.path.join(EXPORT_DIR, f"history_{day}.csv")
    if not os.path.exists(out):
        df = load_history()
        df.to_csv(out, index=False)

# ---------- UI HELPERS ----------
def gate():
    st.markdown(logo_html(), unsafe_allow_html=True)
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if st.session_state.authed:
        return
    with st.container():
        st.title(APP_TITLE)
        pw = st.text_input("Enter access password", type="password")
        if st.button("Enter"):
            if pw == ENTRY_PASSWORD:
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("Wrong password.")

def meta_inputs() -> Meta:
    st.subheader("Audit Metadata (all required)")
    cols = st.columns(4)
    supplier = cols[0].selectbox("Supplier", SUPPLIERS)
    drawing_type = cols[1].selectbox("Drawing Type", DRAWING_TYPES)
    client = cols[2].selectbox("Client", CLIENTS)
    project = cols[3].selectbox("Project", PROJECTS)

    cols2 = st.columns(4)
    site_type = cols2[0].selectbox("Site Type", SITE_TYPES)
    vendor = cols2[1].selectbox("Proposed Vendor", VENDORS)
    cab_location = cols2[2].selectbox("Proposed Cabinet Location", CAB_LOCATIONS)
    radio_location = cols2[3].selectbox("Proposed Radio Location", RADIO_LOCATIONS)

    cols3 = st.columns(2)
    sectors = cols3[0].selectbox("Quantity of Sectors", [1,2,3,4,5,6], index=0)
    site_address = cols3[1].text_input("Site Address", placeholder="MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ")

    st.markdown("### Proposed MIMO Config")
    use_s1 = st.checkbox("Use S1 for all sectors", value=True)
    mimo: Dict[str,str] = {}
    # build sector dropdowns
    for s in range(1, sectors+1):
        label = f"MIMO S{s}"
        if s == 1:
            mimo_s = st.selectbox(label, MIMO_OPTIONS, index=0, key=f"mimo_{s}")
            # copy across if required
            if use_s1:
                for k in range(2, sectors+1):
                    st.session_state[f"mimo_{k}"] = mimo_s
        else:
            mimo_s = st.selectbox(label, MIMO_OPTIONS, index=MIMO_OPTIONS.index(st.session_state.get(f"mimo_{s}", MIMO_OPTIONS[0])), key=f"mimo_{s}")
        mimo[f"S{s}"] = mimo_s
    return Meta(
        supplier=supplier, drawing_type=drawing_type, client=client, project=project,
        site_type=site_type, vendor=vendor, cab_location=cab_location,
        radio_location=radio_location, sectors=sectors, site_address=site_address, mimo=mimo
    )

def title_matches_address(title_text: str, site_address: str) -> bool:
    # Rule: address must appear in title unless it contains ", 0 ," then ignore
    if ", 0 ," in site_address:
        return True
    return re.search(re.escape(site_address), title_text, flags=re.IGNORECASE) is not None

# ---------- AUDIT PIPE ----------
def run_checks(pages: List[str], meta: Meta, rules: Dict[str,Any], do_spell: bool, allow_words: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    # Site title vs address (use page 1 header text)
    first = pages[0] if pages else ""
    if not title_matches_address(first, meta.site_address):
        findings.append(Finding(
            rule_name="Title vs Site Address",
            severity="major",
            page=1,
            text_hit=meta.site_address,
            comment="PDF Title/first page does not contain the Site Address",
            bbox=None,
            category="metadata"
        ))
    # Rules
    findings.extend(run_rule_checks(pages, meta, rules))
    # Spelling (optional)
    if do_spell:
        findings.extend(spelling_findings(pages, allow_words))
    return findings

def status_from_findings(findings: List[Finding]) -> str:
    if any(f.severity.lower()=="major" for f in findings):
        return "Rejected"
    return "Pass"

def audit_tab():
    st.header("Audit")
    meta = meta_inputs()
    up = st.file_uploader("Upload PDF Design", type=["pdf"])
    allow_words = st.text_area("Allowed words (comma-separated) for spellcheck", value="AYGE, AYGD, ELTEK, SAMI, Flexi").split(",")
    allow_words = [w.strip() for w in allow_words if w.strip()]
    do_spell = st.checkbox("Enable spelling checks", value=True)
    exclude_analytics = st.checkbox("Exclude this review from analytics", value=False)

    run_col, clear_col = st.columns([1,1])
    run_clicked = run_col.button("Run Audit")
    if clear_col.button("Clear current results"):
        for k in ["audit_findings","audit_pdf","audit_excel","audit_meta","audit_status","audit_pdf_name","audit_excel_name"]:
            st.session_state.pop(k, None)
        st.info("Cleared.")
        st.stop()

    if run_clicked:
        if not up:
            st.error("Please upload a PDF.")
            st.stop()
        pdf_bytes = up.read()
        pages = pdf_text_by_page(pdf_bytes)
        rules = load_rules(DEFAULT_RULES_FILE)

        with st.status("Running checks…", expanded=False):
            findings = run_checks(pages, meta, rules, do_spell, allow_words)
            status_str = status_from_findings(findings)

        # annotate
        with st.status("Annotating PDF…", expanded=False):
            annotated = annotate_pdf(pdf_bytes, findings)

        # excel
        excel_bytes = make_excel(findings, meta, up.name, status_str)

        # keep in session (persist for 24h) and disk session cache
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(up.name)[0]
        excel_name = f"{base}_{status_str}_{ts}.xlsx"
        pdf_name = f"{base}_{status_str}_{ts}.pdf"
        save_bytes(os.path.join(PERSIST_DIR, excel_name), excel_bytes)
        save_bytes(os.path.join(PERSIST_DIR, pdf_name), annotated)

        st.session_state["audit_findings"] = findings
        st.session_state["audit_pdf"] = annotated
        st.session_state["audit_excel"] = excel_bytes
        st.session_state["audit_meta"] = meta
        st.session_state["audit_status"] = status_str
        st.session_state["audit_pdf_name"] = pdf_name
        st.session_state["audit_excel_name"] = excel_name

        # history row
        row = {
            "timestamp_utc": timestamp_utc().isoformat(),
            "supplier": meta.supplier, "drawing_type": meta.drawing_type, "client": meta.client,
            "project": meta.project, "site_type": meta.site_type, "vendor": meta.vendor,
            "cab_location": meta.cab_location, "radio_location": meta.radio_location,
            "sectors": meta.sectors, "site_address": meta.site_address,
            "status": status_str, "pdf_name": pdf_name, "excel_name": excel_name,
            "exclude": bool(exclude_analytics),
            "findings_json": json.dumps([asdict(f) for f in findings], ensure_ascii=False)
        }
        append_history(row)
        daily_export_dump()

    # Show latest results if available
    if "audit_findings" in st.session_state:
        st.subheader("Results")
        findings = st.session_state["audit_findings"]
        meta = st.session_state["audit_meta"]
        status_str = st.session_state["audit_status"]
        st.write(f"**Status:** {status_str}")
        df = pd.DataFrame([asdict(f) for f in findings])
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        dl1, dl2 = st.columns(2)
        dl1.download_button("⬇️ Download Excel", st.session_state["audit_excel"], file_name=st.session_state["audit_excel_name"])
        dl2.download_button("⬇️ Download Annotated PDF", st.session_state["audit_pdf"], file_name=st.session_state["audit_pdf_name"])
        st.caption("These downloads remain available until you press **Clear current results**.")

def analytics_tab():
    st.header("Analytics")
    dfh = load_history()
    if dfh.empty:
        st.info("No history yet.")
        return
    # coerce timestamp and missing exclude
    if "timestamp_utc" in dfh:
        dfh["timestamp_utc"] = pd.to_datetime(dfh["timestamp_utc"], errors="coerce", utc=True)
    if "exclude" not in dfh:
        dfh["exclude"] = False

    # filters
    c1, c2, c3, c4 = st.columns(4)
    f_client = c1.selectbox("Client", ["All"] + sorted(dfh["client"].dropna().unique().tolist()))
    f_project = c2.selectbox("Project", ["All"] + sorted(dfh["project"].dropna().unique().tolist()))
    f_supplier = c3.selectbox("Supplier", ["All"] + sorted(dfh["supplier"].dropna().unique().tolist()))
    f_vendor = c4.selectbox("Vendor", ["All"] + sorted(dfh["vendor"].dropna().unique().tolist()))
    show_excluded = st.checkbox("Include excluded rows", value=False)

    df = dfh.copy()
    if not show_excluded:
        df = df[df["exclude"] != True]
    if f_client != "All":
        df = df[df["client"] == f_client]
    if f_project != "All":
        df = df[df["project"] == f_project]
    if f_supplier != "All":
        df = df[df["supplier"] == f_supplier]
    if f_vendor != "All":
        df = df[df["vendor"] == f_vendor]

    st.metric("Total audits", len(df))
    if not df.empty:
        rft = round((df["status"].eq("Pass").mean())*100, 1)
        st.metric("Right-first-time %", f"{rft}%")
        st.dataframe(df[[
            "timestamp_utc","supplier","client","project","status","pdf_name","excel_name"
        ]].sort_values("timestamp_utc", ascending=False), use_container_width=True)

def training_tab():
    st.header("Training")
    st.caption("Upload audited **CSV/XLS/XLSX** to record Valid / NotValid decisions, or append quick rules directly (scoped).")
    # audited record ingest
    up = st.file_uploader("Upload audited report (CSV/XLS/XLSX)", type=["csv","xls","xlsx"])
    decision = st.selectbox("This audit decision is…", ["Valid","NotValid"])
    if st.button("Ingest training record"):
        if not up:
            st.error("Please upload a file.")
        else:
            name = up.name.lower()
            try:
                if name.endswith(".csv"):
                    df = pd.read_csv(up)
                elif name.endswith(".xls"):
                    import xlrd  # ensure installed if needed
                    df = pd.read_excel(up, engine="xlrd")
                else:
                    df = pd.read_excel(up)  # openpyxl
                # Minimal expected columns: rule_name, page, comment, (optional: severity, text_hit)
                cols = [c.lower() for c in df.columns]
                lower_map = dict(zip(df.columns, cols))
                df = df.rename(columns=lower_map)
                required = {"rule_name","page","comment"}
                if not required.issubset(set(df.columns)):
                    st.warning("CSV/XLS is ingested as training context (no schema match). It will still improve future heuristics.")
                # Save a copy into exports/training with stamp
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = os.path.join(EXPORT_DIR, f"training_{decision}_{ts}.csv")
                df.to_csv(out, index=False)
                st.success(f"Ingested and archived to {out}.")
            except Exception as e:
                st.exception(e)

    st.divider()
    st.subheader("Add a quick rule")
    with st.form("quick_rule"):
        rn = st.text_input("Rule name", placeholder="e.g., Power Resilience note present")
        severity = st.selectbox("Severity", ["minor","major"], index=1)
        must = st.text_input("Must contain (comma-separated)", value="IMPORTANT NOTE, ELTEK PSU")
        reject_if = st.text_input("Reject if present (comma-separated)", value="")
        st.caption("Scope (optional) – only apply when these match:")
        sc1, sc2, sc3, sc4 = st.columns(4)
        scope_client = sc1.selectbox("client", [""]+CLIENTS)
        scope_project = sc2.selectbox("project", [""]+PROJECTS)
        scope_vendor = sc3.selectbox("vendor", [""]+VENDORS)
        scope_site = sc4.selectbox("site_type", [""]+SITE_TYPES)
        sc5, sc6, sc7 = st.columns(3)
        scope_radio = sc5.selectbox("radio_location", [""]+RADIO_LOCATIONS)
        scope_draw = sc6.selectbox("drawing_type", [""]+DRAWING_TYPES)
        scope_cab = sc7.selectbox("cab_location", [""]+CAB_LOCATIONS)
        pw = st.text_input("Rules password", type="password")
        submitted = st.form_submit_button("Append rule")
        if submitted:
            if not rn.strip():
                st.error("Rule name required.")
            else:
                rule = {
                    "name": rn.strip(),
                    "severity": severity,
                    "must_contain": [x.strip() for x in must.split(",") if x.strip()],
                    "reject_if_present": [x.strip() for x in reject_if.split(",") if x.strip()],
                    "category": "rules",
                    "scope": {
                        "client": scope_client, "project": scope_project, "vendor": scope_vendor,
                        "site_type": scope_site, "radio_location": scope_radio,
                        "drawing_type": scope_draw, "cab_location": scope_cab
                    }
                }
                # remove empties from scope
                rule["scope"] = {k:v for k,v in rule["scope"].items() if v}
                msg = append_rule_to_yaml(DEFAULT_RULES_FILE, rule, pw)
                if msg.startswith("✅"):
                    st.success(msg)
                else:
                    st.error(msg)

def settings_tab():
    st.header("Settings")
    st.caption("Place your logo file in the repo root and set the **exact filename** below.")
    new_logo = st.text_input("Logo filename", value=BRAND_LOGO_FILE, help="png/svg/jpg")
    if new_logo and new_logo != BRAND_LOGO_FILE:
        os.environ["SEKER_LOGO"] = new_logo
        st.success("Logo filename set for this session.")
    st.text("Rules file:")
    st.code(DEFAULT_RULES_FILE)
    # edit YAML quickly (protected)
    pw = st.text_input("Rules password to edit file", type="password")
    if pw == RULES_PASSWORD:
        try:
            with open(DEFAULT_RULES_FILE, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            content = "checklist: []"
        txt = st.text_area(DEFAULT_RULES_FILE, value=content, height=280)
        if st.button("Save rules"):
            with open(DEFAULT_RULES_FILE, "w", encoding="utf-8") as f:
                f.write(txt)
            st.success("Rules saved.")
    else:
        st.info("Enter rules password to edit YAML here.")

    if st.button("Clear current UI results (keeps history & files)"):
        for k in ["audit_findings","audit_pdf","audit_excel","audit_meta","audit_status","audit_pdf_name","audit_excel_name"]:
            st.session_state.pop(k, None)
        st.success("Cleared UI state.")

# ---------- MAIN ----------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    gate()
    if not st.session_state.get("authed", False):
        st.stop()

    st.markdown(logo_html(), unsafe_allow_html=True)
    st.title(APP_TITLE)
    st.caption("Professional, consistent design QA with annotations, analytics, and rapid rule updates.")

    tabs = st.tabs(["Audit","Analytics","Training","Settings"])
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
