# app.py
# AI Design Quality Auditor — Streamlit
# Keeps full feature set: metadata, per-sector MIMO, spelling, rules YAML, training,
# annotated PDF, Excel report, history with analytics & exclude flag, supplier filters,
# logo, passwords, and PyMuPDF-based extraction (no poppler needed).

from __future__ import annotations
import io, os, re, json, base64, uuid, shutil, zipfile, textwrap
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import yaml
from rapidfuzz import process, fuzz
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# -------------------- Constants & Config --------------------

APP_TITLE = "AI Design Quality Auditor"
HISTORY_DIR = "history"
RUNS_DIR = os.path.join(HISTORY_DIR, "runs")
HISTORY_CSV = os.path.join(HISTORY_DIR, "audit_history.csv")
FEEDBACK_CSV = os.path.join(HISTORY_DIR, "feedback.csv")
DEFAULT_RULES_PATH = "rules_example.yaml"

ENTRY_PASSWORD = "Seker123"
ADMIN_RULES_PASSWORD = "vanB3lkum21"

DEFAULT_UI_VISIBILITY_HOURS = 24  # how far back to show in UI (analytics uses filters anyway)
ANNOTATION_COLOR = (1, 0, 0)  # red rectangles

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# Dropdown lists
CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
CAB_LOC = ["Indoor", "Outdoor"]
RADIO_LOC = ["Low Level", "Midway", "High Level", "Unique Coverage"]
SUPPLIERS = [
    # keep editable — shows in metadata & analytics
    "Circet", "WHP", "Daisy", "BT", "Telent", "Morrison", "Babcock",
    "Euro Communications", "CTIL", "Cellnex", "MNO", "Other"
]
QTY_SECTORS = [1,2,3,4,5,6]

# MIMO options (deduplicated & sorted-ish; you can edit freely)
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
]

# -------------------- Utility --------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s[:120] if len(s) > 120 else s

def ensure_history_csv_exists():
    if not os.path.exists(HISTORY_CSV):
        cols = [
            "run_id","timestamp_utc","user","pdf_name","status","exclude",
            "client","project","site_type","vendor","cabinet_loc","radio_loc",
            "supplier","qty_sectors","mimo_s1","mimo_s2","mimo_s3","mimo_s4","mimo_s5","mimo_s6",
            "site_address","findings_count","minor_count","major_count","pass_pct","rft"
        ]
        pd.DataFrame(columns=cols).to_csv(HISTORY_CSV, index=False)

def load_history() -> pd.DataFrame:
    ensure_history_csv_exists()
    try:
        df = pd.read_csv(HISTORY_CSV)
    except Exception:
        # fallback if file partially written: try python engine
        df = pd.read_csv(HISTORY_CSV, engine="python", on_bad_lines="skip")
    # Coerce types
    if "exclude" in df.columns:
        df["exclude"] = df["exclude"].astype(bool)
    if "timestamp_utc" in df.columns:
        # leave as string for portability; analytics converts as needed
        df["timestamp_utc"] = df["timestamp_utc"].astype(str)
    for c in ["findings_count","minor_count","major_count","pass_pct","rft","qty_sectors"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def append_history(row: Dict[str, Any]):
    df = load_history()
    # fill missing columns
    for col in set(df.columns).difference(row.keys()):
        row[col] = None
    # also include any new keys
    for col in set(row.keys()).difference(df.columns):
        df[col] = None
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_CSV, index=False)

def filter_history_hours(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    if "timestamp_utc" not in df.columns:
        return df
    try:
        dfc = df.copy()
        dfc["timestamp_dt"] = pd.to_datetime(dfc["timestamp_utc"], errors="coerce", utc=True)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return dfc[dfc["timestamp_dt"] >= cutoff].drop(columns=["timestamp_dt"])
    except Exception:
        return df

# -------------------- Rules I/O --------------------

def load_rules(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            # normalize structure
            data.setdefault("allowlist", [])
            data.setdefault("forbidden_terms", [])
            data.setdefault("patterns", [])
            data.setdefault("cross_checks", [])
            data.setdefault("project_overrides", {})
            return data
    except FileNotFoundError:
        # create a minimal file
        data = {"allowlist": [], "forbidden_terms": [], "patterns": [], "cross_checks": [], "project_overrides": {}}
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        return data
    except yaml.YAMLError as e:
        # surface line to help fixes
        st.error(f"YAML parse error in {path}: {e}")
        return {"allowlist": [], "forbidden_terms": [], "patterns": [], "cross_checks": [], "project_overrides": {}}

def save_rules(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

# -------------------- PDF Extraction --------------------

def text_from_pdf(pdf_bytes: bytes) -> List[str]:
    """Extract per-page text. If a page has no selectable text, OCR that page only."""
    pages: List[str] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i in range(len(doc)):
        page = doc[i]
        txt = page.get_text("text")
        if txt and txt.strip():
            pages.append(txt)
            continue
        # OCR fallback
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(img)
            pages.append(ocr_text or "")
        except Exception:
            pages.append("")
    return pages

def find_bbox_for_term(doc: fitz.Document, page_index: int, term: str) -> Optional[Tuple[float,float,float,float]]:
    try:
        page = doc[page_index]
        rects = page.search_for(term, quads=False)
        if rects:
            r = rects[0]
            return (r.x0, r.y0, r.x1, r.y1)
    except Exception:
        pass
    return None

# -------------------- Checks --------------------

def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']+", text)

def spelling_findings(pages: List[str], allowlist: List[str]) -> List[Dict[str, Any]]:
    """Simple spell-like checker using fuzzy suggestions; allowlist is lowercase words to skip."""
    allow = set([w.lower() for w in allowlist])
    vocab = set()
    for p in pages:
        for t in tokenize(p):
            vocab.add(t.lower())

    # Split words into 'suspect' if rare casing or short; we keep it simple:
    findings = []
    words = sorted(vocab)
    # Prepare a suggestion corpus (common english-ish + allowlist); for simplicity, use allowlist only
    corpus = list(allow) if allow else []
    for w in words:
        if len(w) < 3:  # ignore very short
            continue
        if w in allow:
            continue
        # heuristic: if contains digits or all caps short codes -> skip here; enforce via patterns instead
        if re.search(r"\d", w):
            continue
        if w.isupper() and len(w) <= 5:
            continue

        suggestion = None
        if corpus:
            sug, score, _ = process.extractOne(w, corpus, scorer=fuzz.ratio)
            # only suggest if close
            if score >= 85:
                suggestion = sug

        findings.append({
            "type": "Spelling",
            "severity": "Minor",
            "rule_id": "SPELLING",
            "message": f"Suspicious token: '{w}'" + (f" → consider '{suggestion}'" if suggestion else ""),
            "page": None,     # we’ll try to locate on annotate
            "term": w,
            "context": w,
            "valid": True,
        })
    return findings

def forbidden_terms_findings(pages: List[str], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    terms = rules.get("forbidden_terms", [])
    for pi, text in enumerate(pages):
        for t in terms:
            if not t:
                continue
            if re.search(rf"\b{re.escape(t)}\b", text, flags=re.IGNORECASE):
                out.append({
                    "type": "ForbiddenTerm",
                    "severity": "Major",
                    "rule_id": f"TERM_{t}",
                    "message": f"Forbidden term found: '{t}'",
                    "page": pi,
                    "term": t,
                    "context": t,
                    "valid": True,
                })
    return out

def patterns_findings(pages: List[str], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for pat in rules.get("patterns", []):
        rid = pat.get("id") or f"PAT_{abs(hash(pat.get('pattern','')))%10_000}"
        pattern = pat.get("pattern", "")
        severity = pat.get("severity", "Minor")
        desc = pat.get("description", "Pattern match")
        flags = re.IGNORECASE if pat.get("ignore_case", True) else 0
        try:
            cre = re.compile(pattern, flags)
        except re.error:
            # skip bad regex, surface in UI later
            continue
        for pi, text in enumerate(pages):
            for m in cre.finditer(text):
                snippet = text[max(0, m.start()-40): m.end()+40]
                out.append({
                    "type": "Pattern",
                    "severity": severity,
                    "rule_id": rid,
                    "message": f"{desc}: '{m.group(0)}'",
                    "page": pi,
                    "term": m.group(0),
                    "context": snippet,
                    "valid": True,
                })
    return out

def metadata_cross_checks(meta: Dict[str, Any], pages: List[str], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    # Example: Brush vs Generator Power (user earlier example: if "Brush" then "Generator Power" shouldn't be used)
    cross = rules.get("cross_checks", [])
    text_all = "\n".join(pages)
    for c in cross:
        rid = c.get("id") or f"X_{abs(hash(json.dumps(c, sort_keys=True)))%10_000}"
        desc = c.get("description", "Cross check")
        severity = c.get("severity", "Major")
        # types: "if_term_then_forbid_term", "address_must_match_title"
        ctype = c.get("type")
        if ctype == "if_term_then_forbid_term":
            if re.search(rf"\b{re.escape(c.get('if_term',''))}\b", text_all, flags=re.IGNORECASE):
                if re.search(rf"\b{re.escape(c.get('forbid_term',''))}\b", text_all, flags=re.IGNORECASE):
                    out.append({
                        "type": "CrossCheck",
                        "severity": severity,
                        "rule_id": rid,
                        "message": desc,
                        "page": None,
                        "term": c.get('forbid_term',''),
                        "context": desc,
                        "valid": True,
                    })
        elif ctype == "address_must_match_title":
            # If site_address provided, ensure it's in first page (title area). Ignore if contains ", 0 ,"
            sa = (meta.get("site_address") or "").strip()
            if sa and ", 0 ," not in sa:
                title_text = pages[0] if pages else ""
                if sa not in title_text:
                    out.append({
                        "type": "CrossCheck",
                        "severity": severity,
                        "rule_id": rid,
                        "message": f"Site Address mismatch: '{sa}' not found on title page.",
                        "page": 0 if pages else None,
                        "term": sa,
                        "context": title_text[:400],
                        "valid": True,
                    })
    # Project overrides example: hide MIMO checks for Power Resilience
    if (meta.get("project") == "Power Resilience"):
        # mark any MIMO-specific pattern finding as valid-but-ignored (or drop). Here we drop nothing, but we could downgrade severity.
        pass
    return out

# -------------------- Annotation & Reports --------------------

def annotate_pdf(pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    """Draw boxes and callouts for findings. We try to locate bbox via searching term."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for f in findings:
        if not f.get("valid", True):
            continue  # don't mark invalidated
        pi = f.get("page")
        term = f.get("term") or ""
        if pi is None or pi<0 or pi>=len(doc) or not term:
            # try to locate anywhere
            located = False
            for page_index in range(len(doc)):
                bbox = find_bbox_for_term(doc, page_index, term)
                if bbox:
                    pi = page_index
                    located = True
                    break
            if not located:
                continue
        bbox = find_bbox_for_term(doc, pi, term)
        if not bbox:
            continue
        page = doc[pi]
        rect = fitz.Rect(*bbox)
        # draw rectangle
        shape = page.new_shape()
        shape.draw_rect(rect)
        shape.finish(color=ANNOTATION_COLOR, fill=None, width=1)
        shape.commit()
        # add text note nearby
        note = f"[{f.get('severity','')}] {f.get('message','')}"
        # put a small textbox above
        tr = fitz.Rect(rect.x0, max(0, rect.y0-18), min(page.rect.width, rect.x0+220), rect.y0)
        page.insert_textbox(tr, note, fontsize=7, color=(0,0,0))
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

def make_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], pdf_name: str, status: str) -> bytes:
    df = pd.DataFrame(findings)
    # ensure expected cols
    for c in ["type","severity","rule_id","message","page","term","context","valid"]:
        if c not in df.columns: df[c] = None
    meta_block = {f"meta_{k}": [v] for k,v in meta.items()}
    meta_df = pd.DataFrame(meta_block)
    with pd.ExcelWriter(io.BytesIO(), engine="openpyxl") as xw:
        meta_df.to_excel(xw, index=False, sheet_name="Meta")
        df.to_excel(xw, index=False, sheet_name="Findings")
        ws = xw.book.create_sheet("Summary")
        ws["A1"] = "PDF"
        ws["B1"] = pdf_name
        ws["A2"] = "Status"
        ws["B2"] = status
        # return bytes
        bio = xw.book.properties  # touch to avoid lint
        data = xw._writer.fp.getvalue()  # type: ignore
    return data

# -------------------- UI Helpers --------------------

def gate_with_password():
    if "passed_gate" not in st.session_state:
        st.session_state.passed_gate = False
    if st.session_state.passed_gate:
        return True
    with st.form("entry_form", clear_on_submit=False):
        st.password_input("Enter access password", key="entry_pw", help="Ask admin if you don't have it.")
        ok = st.form_submit_button("Enter")
    if ok:
        if st.session_state.get("entry_pw") == ENTRY_PASSWORD:
            st.session_state.passed_gate = True
        else:
            st.error("Wrong password.")
    return st.session_state.passed_gate

def logo_block():
    st.sidebar.subheader("Branding")
    lgf = st.sidebar.file_uploader("Upload logo (png/jpg/svg)", type=["png","jpg","jpeg","svg"], key="logo_up")
    if lgf:
        st.session_state["logo_bytes"] = lgf.read()
        st.session_state["logo_name"] = lgf.name
    logo_name_text = st.sidebar.text_input("Or type logo filename in repo root", value=st.session_state.get("logo_name_text",""))
    st.session_state["logo_name_text"] = logo_name_text
    size = st.sidebar.slider("Logo width (px)", 60, 420, 180)
    # render in header (top-left)
    col_logo, col_title = st.columns([1,6])
    with col_logo:
        img_bytes = None
        if st.session_state.get("logo_bytes"):
            img_bytes = st.session_state["logo_bytes"]
        elif logo_name_text and os.path.exists(logo_name_text):
            img_bytes = open(logo_name_text, "rb").read()
        elif os.path.exists("logo.png"):
            img_bytes = open("logo.png","rb").read()
        if img_bytes:
            st.image(img_bytes, width=size, caption=None)
        else:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with col_title:
        st.markdown(f"## {APP_TITLE}")

def meta_form() -> Dict[str, Any]:
    st.subheader("Audit Metadata")
    with st.form("meta_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        client = c1.selectbox("Client", CLIENTS, index=None, placeholder="Select client")
        project = c2.selectbox("Project", PROJECTS, index=None, placeholder="Select project")
        site_type = c3.selectbox("Site Type", SITE_TYPES, index=None, placeholder="Select site type")

        c4, c5, c6 = st.columns(3)
        vendor = c4.selectbox("Proposed Vendor", VENDORS, index=None, placeholder="Select vendor")
        cabinet = c5.selectbox("Proposed Cabinet Location", CAB_LOC, index=None, placeholder="Select")
        radio = c6.selectbox("Proposed Radio Location", RADIO_LOC, index=None, placeholder="Select")

        c7, c8, c9 = st.columns(3)
        supplier = c7.selectbox("Supplier", SUPPLIERS, index=None, placeholder="Select supplier")
        qty = c8.selectbox("Quantity of Sectors", QTY_SECTORS, index=0)
        site_address = c9.text_input("Site Address", placeholder="e.g., MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ")

        # MIMO per sector (hidden if Power Resilience)
        mimo_section_visible = (project != "Power Resilience")
        copy_all = False
        mimos = {"S1": None, "S2": None, "S3": None, "S4": None, "S5": None, "S6": None}
        if mimo_section_visible:
            st.markdown("#### Proposed MIMO Config (hidden for Power Resilience)")
            copy_all = st.checkbox("Use S1 for all sectors", value=False)
            mimos["S1"] = st.selectbox("Proposed MIMO S1", MIMO_OPTIONS, index=None, placeholder="Select MIMO S1")
            for s in range(2, qty+1):
                label = f"Proposed MIMO S{s}"
                if copy_all and mimos["S1"]:
                    mimos[f"S{s}"] = mimos["S1"]
                    st.text_input(label, value=mimos["S1"], disabled=True)
                else:
                    mimos[f"S{s}"] = st.selectbox(label, MIMO_OPTIONS, index=None, placeholder=f"Select MIMO S{s}")
        else:
            st.info("MIMO config not required for Power Resilience.")

        exclude = st.checkbox("Exclude this review from analytics", value=False)

        submitted = st.form_submit_button("Save Metadata")
    # validate required
    missing = []
    for key, val in {
        "client": client, "project": project, "site_type": site_type,
        "vendor": vendor, "cabinet_loc": cabinet, "radio_loc": radio,
        "supplier": supplier, "qty_sectors": qty, "site_address": site_address
    }.items():
        if val in (None, "", []):
            missing.append(key)
    if missing:
        st.warning("Please complete all metadata fields.")
    meta = {
        "client": client, "project": project, "site_type": site_type,
        "vendor": vendor, "cabinet_loc": cabinet, "radio_loc": radio,
        "supplier": supplier, "qty_sectors": qty, "site_address": site_address,
        "mimo": mimos, "exclude": exclude
    }
    st.session_state["meta"] = meta
    return meta

# -------------------- Training --------------------

def upsert_feedback_from_excel(excel_bytes: bytes):
    """Ingest a previously downloaded report to learn Valid/Not Valid + new rules."""
    try:
        x = pd.ExcelFile(io.BytesIO(excel_bytes))
        if "Findings" not in x.sheet_names:
            st.error("Uploaded workbook has no 'Findings' sheet.")
            return
        df = pd.read_excel(x, sheet_name="Findings")
        # expecting a 'valid' column; if user changed cells to flag not valid, capture it
        if "valid" not in df.columns:
            st.warning("No 'valid' column found; nothing to learn.")
            return
        # Append to feedback log
        df2 = df[["type","severity","rule_id","message","page","term","context","valid"]].copy()
        df2["timestamp_utc"] = now_utc_iso()
        if os.path.exists(FEEDBACK_CSV):
            old = pd.read_csv(FEEDBACK_CSV)
            df2 = pd.concat([old, df2], ignore_index=True)
        df2.to_csv(FEEDBACK_CSV, index=False)
        st.success("Feedback ingested. Future runs will respect your 'valid' flags when possible.")
    except Exception as e:
        st.error(f"Failed to ingest feedback: {e}")

def apply_feedback(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """If a term/rule previously marked invalid, apply that preference."""
    if not os.path.exists(FEEDBACK_CSV):
        return findings
    try:
        fb = pd.read_csv(FEEDBACK_CSV)
        # build quick lookups
        inv_terms = set(
            fb.loc[(fb.get("valid")==False) & fb["term"].notna(), "term"].str.lower().tolist()
        )
        inv_rules = set(
            fb.loc[(fb.get("valid")==False) & fb["rule_id"].notna(), "rule_id"].tolist()
        )
        for f in findings:
            t = (f.get("term") or "").lower()
            rid = f.get("rule_id")
            if t in inv_terms or rid in inv_rules:
                f["valid"] = False
        return findings
    except Exception:
        return findings

# -------------------- Main --------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    if not gate_with_password():
        st.stop()

    logo_block()

    # Sidebar navigation
    page = st.sidebar.radio("Navigation", ["Audit", "Training", "Analytics", "Settings"], index=0)

    # Rules loader
    rules_file = st.sidebar.text_input("Rules file path", value=DEFAULT_RULES_PATH)
    rules = load_rules(rules_file)

    # Audit Page
    if page == "Audit":
        meta = meta_form()

        st.subheader("Run Audit")
        up = st.file_uploader("Upload PDF (preferred) or DWG (as reference)", type=["pdf","dwg"])
        run_btn = st.button("Run Audit")

        if run_btn:
            # metadata completeness
            m = st.session_state.get("meta", {})
            required = ["client","project","site_type","vendor","cabinet_loc","radio_loc","supplier","qty_sectors","site_address"]
            if any((m.get(k) in (None,"",[])) for k in required):
                st.error("Please complete metadata ( Save Metadata ) before running.")
                st.stop()

            if not up:
                st.error("Please upload a PDF.")
                st.stop()

            raw = up.read()
            pages = text_from_pdf(raw)

            # Checks
            findings: List[Dict[str, Any]] = []
            findings += spelling_findings(pages, rules.get("allowlist", []))
            findings += forbidden_terms_findings(pages, rules)
            findings += patterns_findings(pages, rules)
            findings += metadata_cross_checks(m, pages, rules)

            # Apply previous feedback (user-set Valid/Not Valid)
            findings = apply_feedback(findings)

            # Status rollup
            use_findings = [f for f in findings if f.get("valid", True)]
            major = sum(1 for x in use_findings if x.get("severity","").lower()=="major")
            minor = sum(1 for x in use_findings if x.get("severity","").lower()=="minor")
            status = "PASS" if (major==0 and minor==0) else "REJECTED"

            st.markdown(f"**Status:** {status}  |  **Major:** {major}  **Minor:** {minor}  **Total Considered:** {len(use_findings)}")

            # Findings table with toggles to mark valid/not valid
            st.markdown("### Findings (toggle Valid to correct false positives)")
            df = pd.DataFrame(findings)
            if not df.empty:
                # Editable 'valid' column
                df["valid"] = df["valid"].fillna(True).astype(bool)
                edited = st.data_editor(
                    df[["type","severity","rule_id","message","page","term","context","valid"]],
                    use_container_width=True,
                    num_rows="fixed",
                    hide_index=True
                )
                # Replace findings with edited 'valid'
                findings = edited.to_dict("records")

            # Generate files
            # Annotate after trying to locate terms
            try:
                annotated = annotate_pdf(raw, [f for f in findings if f.get("valid", True)])
            except Exception:
                annotated = raw  # fallback if annotation fails

            excel_bytes = make_excel(findings, m, up.name, status)

            # Persist run
            run_id = uuid.uuid4().hex[:12]
            run_dir = os.path.join(RUNS_DIR, run_id)
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "input.pdf"), "wb") as f:
                f.write(raw)
            with open(os.path.join(run_dir, "annotated.pdf"), "wb") as f:
                f.write(annotated)
            with open(os.path.join(run_dir, "report.xlsx"), "wb") as f:
                f.write(excel_bytes)
            with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2)
            with open(os.path.join(run_dir, "findings.json"), "w", encoding="utf-8") as f:
                json.dump(findings, f, indent=2)

            # Append history
            rft = 1.0 if status == "PASS" else 0.0
            append_history({
                "run_id": run_id,
                "timestamp_utc": now_utc_iso(),
                "user": "streamlit_user",
                "pdf_name": up.name,
                "status": status,
                "exclude": bool(m.get("exclude", False)),
                "client": m.get("client"),
                "project": m.get("project"),
                "site_type": m.get("site_type"),
                "vendor": m.get("vendor"),
                "cabinet_loc": m.get("cabinet_loc"),
                "radio_loc": m.get("radio_loc"),
                "supplier": m.get("supplier"),
                "qty_sectors": m.get("qty_sectors"),
                "mimo_s1": m["mimo"].get("S1"),
                "mimo_s2": m["mimo"].get("S2"),
                "mimo_s3": m["mimo"].get("S3"),
                "mimo_s4": m["mimo"].get("S4"),
                "mimo_s5": m["mimo"].get("S5"),
                "mimo_s6": m["mimo"].get("S6"),
                "site_address": m.get("site_address"),
                "findings_count": len([f for f in findings if f.get("valid", True)]),
                "minor_count": minor,
                "major_count": major,
                "pass_pct": 100.0 if status=="PASS" else 0.0,
                "rft": rft
            })

            # Prepare file names with status + date stamp
            date_stamp = datetime.now().strftime("%Y-%m-%d")
            base = os.path.splitext(safe_filename(up.name))[0]
            excel_name = f"{base}__{status}__{date_stamp}.xlsx"
            pdf_name = f"{base}__{status}__{date_stamp}.pdf"

            cdl1, cdl2 = st.columns(2)
            with cdl1:
                st.download_button("⬇️ Download Excel Report", excel_bytes, file_name=excel_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with cdl2:
                st.download_button("⬇️ Download Annotated PDF", annotated, file_name=pdf_name, mime="application/pdf")

            st.success(f"Run saved in {run_dir}. This keeps your files available after download.")

    # Training Page
    if page == "Training":
        st.subheader("Rapid Training — Learn from my decisions")
        st.caption("Upload a previously downloaded Excel report. We’ll ingest your 'Valid' (true/false) flags and remember them.")

        up_x = st.file_uploader("Upload Excel report (Findings sheet required)", type=["xlsx"], key="train_xlsx")
        if up_x and st.button("Ingest Feedback"):
            upsert_feedback_from_excel(up_x.read())

        st.markdown("---")
        st.subheader("Add a single rule quickly")
        with st.form("quick_rule"):
            rtype = st.selectbox("Rule type", ["Forbidden Term", "Regex Pattern", "Cross Check: if term then forbid term", "Cross Check: Site Address must match title"])
            desc = st.text_input("Description (optional)")
            sev = st.selectbox("Severity", ["Minor","Major"], index=1)
            submitted = st.form_submit_button("Add Rule")
            if submitted:
                cur = load_rules(rules_file)
                if rtype == "Forbidden Term":
                    term = st.session_state.get("qt_forbid") or st.text_input  # trick to access below
                # Render below in one go
        # Slightly awkward in a single pass; provide individual widgets outside form:
        fr_col1, fr_col2 = st.columns(2)
        with fr_col1:
            forbid_term = st.text_input("Forbidden Term (for the 'Forbidden Term' rule type)", key="qt_forbid")
            regex_pat = st.text_input("Regex Pattern (for 'Regex Pattern' rule type)", key="qt_regex")
        with fr_col2:
            if_term = st.text_input("IF term (for cross check)", key="qt_if")
            then_forbid = st.text_input("THEN forbid term (for cross check)", key="qt_then")

        if st.button("✅ Save Quick Rule"):
            cur = load_rules(rules_file)
            sev = "Major"
            desc = "Quick add"
            if forbid_term:
                cur.setdefault("forbidden_terms", []).append(forbid_term)
                save_rules(rules_file, cur)
                st.success(f"Added forbidden term: {forbid_term}")
            elif regex_pat:
                cur.setdefault("patterns", []).append({"id": f"PAT_{uuid.uuid4().hex[:6]}", "pattern": regex_pat, "description":"Quick regex", "severity":"Major", "ignore_case": True})
                save_rules(rules_file, cur)
                st.success(f"Added regex pattern.")
            elif if_term and then_forbid:
                cur.setdefault("cross_checks", []).append({
                    "id": f"X_{uuid.uuid4().hex[:6]}",
                    "type": "if_term_then_forbid_term",
                    "if_term": if_term,
                    "forbid_term": then_forbid,
                    "description": f"If '{if_term}' appears, '{then_forbid}' is not allowed.",
                    "severity": "Major"
                })
                save_rules(rules_file, cur)
                st.success("Added cross-check rule.")
            else:
                st.info("Fill at least one of the quick-rule fields before saving.")

    # Analytics Page
    if page == "Analytics":
        st.subheader("Analytics & Right-First-Time")
        dfh = load_history()
        if dfh.empty:
            st.info("No runs yet.")
        else:
            colf1, colf2, colf3 = st.columns(3)
            with colf1:
                supplier_f = st.selectbox("Supplier filter", ["(All)"] + SUPPLIERS, index=0)
            with colf2:
                client_f = st.selectbox("Client filter", ["(All)"] + CLIENTS, index=0)
            with colf3:
                project_f = st.selectbox("Project filter", ["(All)"] + PROJECTS, index=0)
            # exclude flag
            dfu = dfh[dfh.get("exclude", False) != True].copy() if "exclude" in dfh.columns else dfh.copy()
            if supplier_f != "(All)":
                dfu = dfu[dfu.get("supplier")==supplier_f]
            if client_f != "(All)":
                dfu = dfu[dfu.get("client")==client_f]
            if project_f != "(All)":
                dfu = dfu[dfu.get("project")==project_f]

            # basic KPIs
            total = len(dfu)
            passes = (dfu.get("status","")=="PASS").sum() if "status" in dfu.columns else 0
            rft = (passes/total*100.0) if total else 0.0
            st.metric("Audits (included in analytics)", total)
            st.metric("Right-First-Time %", f"{rft:.1f}%")
            # table
            show_cols = ["timestamp_utc","pdf_name","status","client","project","site_type","vendor","cabinet_loc","radio_loc","supplier","qty_sectors","minor_count","major_count","rft"]
            show_cols = [c for c in show_cols if c in dfu.columns]
            st.dataframe(dfu[show_cols].sort_values("timestamp_utc", ascending=False), use_container_width=True)

    # Settings Page (YAML editor)
    if page == "Settings":
        st.subheader("Rules (YAML) — Admin")
        pw = st.text_input("Admin password", type="password", help="Required to edit YAML")
        txt = st.text_area("rules_example.yaml", value=yaml.safe_dump(rules, sort_keys=False, allow_unicode=True), height=400)
        col_s1, col_s2 = st.columns([1,1])
        with col_s1:
            if st.button("💾 Save Rules"):
                if pw == ADMIN_RULES_PASSWORD:
                    try:
                        parsed = yaml.safe_load(txt) or {}
                        save_rules(rules_file, parsed)
                        st.success("Saved.")
                    except yaml.YAMLError as e:
                        st.error(f"YAML error: {e}")
                else:
                    st.error("Wrong admin password.")
        with col_s2:
            if st.button("⤵️ Download Rules YAML"):
                st.download_button("Download now", data=txt.encode("utf-8"), file_name=os.path.basename(rules_file), mime="text/yaml", use_container_width=True)

        # History housekeeping
        st.markdown("---")
        st.subheader("History & Exports")
        dfh = load_history()
        if not dfh.empty:
            if st.button("⬇️ Export full history CSV"):
                st.download_button("Download history.csv", data=open(HISTORY_CSV,"rb").read(), file_name="audit_history.csv", mime="text/csv", use_container_width=True)
            if st.button("⬇️ Export all run folders (zip)"):
                zip_path = os.path.join(HISTORY_DIR, f"runs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
                with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for root, _, files in os.walk(RUNS_DIR):
                        for fn in files:
                            full = os.path.join(root, fn)
                            arc = os.path.relpath(full, HISTORY_DIR)
                            zf.write(full, arcname=arc)
                st.download_button("Download runs zip", data=open(zip_path,"rb").read(), file_name=os.path.basename(zip_path), mime="application/zip", use_container_width=True)

if __name__ == "__main__":
    main()
