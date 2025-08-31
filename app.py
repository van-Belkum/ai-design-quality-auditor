# app.py
# AI Design Quality Auditor – Stable Baseline
# Tabs: Audit | Training | Analytics
# Storage: storage/history/history.csv, storage/reports/*
# Annotations: PyMuPDF only (no Poppler). Excel: openpyxl.
# Entry password: Seker123 ; YAML/settings password: vanB3lkum21

import io, os, re, json, base64, zipfile, datetime as dt
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import yaml

try:
    from spellchecker import SpellChecker
except Exception:
    SpellChecker = None

# --------------------------- CONFIG / CONSTANTS ---------------------------

ENTRY_PASSWORD = "Seker123"       # gate to use the tool
YAML_PASSWORD  = "vanB3lkum21"    # to apply rules/settings updates

APP_DIR = os.getcwd()
STORAGE_DIR = os.path.join(APP_DIR, "storage")
REPORTS_DIR = os.path.join(STORAGE_DIR, "reports")
HISTORY_DIR = os.path.join(STORAGE_DIR, "history")
HISTORY_CSV = os.path.join(HISTORY_DIR, "history.csv")
RULES_FILE  = os.path.join(APP_DIR, "rules_example.yaml")

DEFAULT_UI_KEEP_DAYS = 14  # keep visible in tool
DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Default supplier list lives in YAML (metadata_options.suppliers).
# If missing, we fall back to this conservative set (you can change later in Settings):
FALLBACK_SUPPLIERS = [
    "KTL","BT","BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex","Ericsson","Nokia","Other"
]

PROJECTS = ["RAN","Power Resilience","East Unwind","Beacon 4"]
SITE_TYPES = ["Greenfield","Rooftop","Streetworks"]
VENDORS = ["Ericsson","Nokia"]
CAB_LOCATIONS = ["Indoor","Outdoor"]
RADIO_LOCATIONS = ["Low Level","High Level","Unique Coverage","Midway"]  # per your request
SECTOR_QTY = [1,2,3,4,5,6]
DRAWING_TYPES = ["General Arrangement","Detailed Design"]

# Your full MIMO list (order lightly normalised). You can edit in Settings.
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

# --------------------------- UTILITIES ---------------------------

def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

def now_utc() -> dt.datetime:
    return dt.datetime.utcnow()

def ts_str(ts: Optional[dt.datetime]=None) -> str:
    return (ts or now_utc()).strftime(DATE_FMT)

def clean_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def load_rules(path: str = RULES_FILE) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {
            "allowed_words": ["EE","ADC","RRH","PCC","BAC","NGR","kVA","AHPMDB","AHEGC","ARHA","AZQJ","RZZHHTTS4-65B-R7","CV65BSX-2X2"],
            "metadata_options": {
                "suppliers": FALLBACK_SUPPLIERS,
                "mimo_options": MIMO_OPTIONS
            },
            "atomic_rules": [
                # Example template rule
                # {"name":"Eltek PSU note present","when":{"project":["Power Resilience"]},
                #  "must_include":{"pages":[300],"patterns":["To support the power resilience configure settings the Eltek PSU"]},
                #  "severity":"major"}
            ]
        }
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("allowed_words", [])
    data.setdefault("metadata_options", {})
    data["metadata_options"].setdefault("suppliers", FALLBACK_SUPPLIERS)
    data["metadata_options"].setdefault("mimo_options", MIMO_OPTIONS)
    data.setdefault("atomic_rules", [])
    return data

def save_rules(data: Dict[str, Any], path: str = RULES_FILE):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def read_pdf_text_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Return list of pages with 'text' and a PyMuPDF page object reference (for later search)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        pg = doc.load_page(i)
        pages.append({"index": i, "text": pg.get_text("text"), "page": pg})
    return pages, doc

def address_from_pdf(pages: List[Dict[str, Any]]) -> Optional[str]:
    # Heuristic: look for lines starting with "Address:" or prominent block near top pages
    for p in pages[:3]:
        for line in p["text"].splitlines():
            if re.search(r"^\s*Address\s*:", line, flags=re.I):
                val = line.split(":",1)[-1].strip()
                if val:
                    return val
    return None

def normalize_addr(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s*,\s*0\s*,", ",", s)  # ignore ", 0 ,"
    s = re.sub(r"\s+", " ", s)
    return s.upper()

def safe_spell_candidates(sp, token: str) -> Optional[str]:
    """Guarded suggestion; returns best suggestion or None."""
    try:
        # spell.correction is faster and returns a single suggestion
        best = sp.correction(token)
        if best and best.lower() != token.lower():
            return best
        return None
    except Exception:
        return None

def is_acronym_or_code(w: str) -> bool:
    if len(w) <= 2:
        return True
    if w.isupper():
        return True
    if any(ch.isdigit() for ch in w):
        return True
    return False

def annotate_pdf(pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    """Annotate PDF with squares around found snippets and sticky notes. Returns new PDF bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for f in findings:
        page_no = f.get("page", 1) - 1
        if page_no < 0 or page_no >= len(doc):
            continue
        page = doc.load_page(page_no)
        note_text = f"[{f.get('severity','minor').upper()}] {f.get('message','Finding')}"
        snippet = f.get("snippet") or f.get("pattern") or ""
        placed = False
        rects = []
        if snippet:
            # Use a small search window (first ~50 chars)
            key = snippet.strip()
            if len(key) > 80:
                key = key[:80]
            try:
                rects = page.search_for(key, quads=False)
            except Exception:
                rects = []
        if rects:
            for r in rects[:3]:  # cap to avoid noise
                page.add_rect_annot(r, color=(1,0,0), fill=None)
                page.add_text_annot(r.br, note_text)
            placed = True
        if not placed:
            # fallback top-left sticky
            page.add_text_annot(fitz.Point(36, 36), note_text)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

def make_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], original_pdf_name: str, status: str) -> Tuple[bytes, str]:
    """Create Excel in memory with 'Findings' + 'Meta'."""
    df = pd.DataFrame(findings) if findings else pd.DataFrame(columns=["type","severity","page","message","rule_name","suggestion","snippet"])
    meta_df = pd.DataFrame([meta])
    mem = io.BytesIO()
    with pd.ExcelWriter(mem, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Findings")
        meta_df.to_excel(xw, index=False, sheet_name="Metadata")
        # simple summary
        summary = {
            "original_pdf": original_pdf_name,
            "status": status,
            "minor": int((df["severity"]=="minor").sum()) if "severity" in df.columns else 0,
            "major": int((df["severity"]=="major").sum()) if "severity" in df.columns else 0,
            "total": int(len(df))
        }
        pd.DataFrame([summary]).to_excel(xw, index=False, sheet_name="Summary")
    mem.seek(0)
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(original_pdf_name)[0]
    fname = f"{clean_filename(base)}_{status}_{ts}.xlsx"
    return mem.getvalue(), fname

def append_history(row: Dict[str, Any]):
    ensure_dirs()
    exists = os.path.exists(HISTORY_CSV)
    df = pd.DataFrame([row])
    if exists:
        df.to_csv(HISTORY_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_CSV, mode="w", header=True, index=False)

def load_history_df() -> pd.DataFrame:
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame(columns=[
            "timestamp_utc","supplier","client","project","status","pdf_name","excel_name",
            "exclude_from_analytics","minor","major","total","rft"
        ])
    try:
        df = pd.read_csv(HISTORY_CSV)
        # best-effort dtypes
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
        for c in ["exclude_from_analytics","rft"]:
            if c in df.columns:
                df[c] = df[c].astype(bool)
        return df
    except Exception:
        # corrupted file fallback
        return pd.DataFrame(columns=[
            "timestamp_utc","supplier","client","project","status","pdf_name","excel_name",
            "exclude_from_analytics","minor","major","total","rft"
        ])

def keep_file_for_days(path: str, days: int = DEFAULT_UI_KEEP_DAYS):
    """Delete files older than N days in folder."""
    if not os.path.exists(path):
        return
    cutoff = now_utc() - dt.timedelta(days=days)
    for fn in os.listdir(path):
        fp = os.path.join(path, fn)
        try:
            mtime = dt.datetime.utcfromtimestamp(os.path.getmtime(fp))
            if mtime < cutoff:
                os.remove(fp)
        except Exception:
            pass

# --------------------------- CORE CHECKS ---------------------------

def spelling_findings(pages: List[Dict[str, Any]], allow_words: List[str], limit_flags: int = 150) -> List[Dict[str, Any]]:
    findings = []
    if not SpellChecker:
        return findings
    sp = SpellChecker()
    allow = set(w.lower() for w in allow_words)
    flagged = 0
    token_re = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
    for pg in pages:
        page_no = pg["index"] + 1
        text = pg["text"]
        for m in token_re.finditer(text):
            w = m.group(0)
            wl = w.lower()
            if wl in allow or is_acronym_or_code(w):
                continue
            sug = safe_spell_candidates(sp, wl)
            if sug:
                findings.append({
                    "type":"spelling",
                    "severity":"minor",
                    "page":page_no,
                    "message": f"Possible typo: '{w}' → '{sug}'",
                    "rule_name":"spelling",
                    "suggestion":sug,
                    "snippet": w
                })
                flagged += 1
                if flagged >= limit_flags:
                    return findings
    return findings

def address_match_check(pages: List[Dict[str, Any]], site_address: str) -> Optional[Dict[str, Any]]:
    if not site_address:
        return None
    pdf_addr = address_from_pdf(pages)
    if not pdf_addr:
        return None
    a = normalize_addr(site_address)
    b = normalize_addr(pdf_addr)
    if a == b:
        return None
    return {
        "type":"metadata",
        "severity":"major",
        "page":1,
        "message": f"Site Address mismatch. Metadata='{site_address}' vs PDF='{pdf_addr}'.",
        "rule_name":"address_match",
        "snippet": pdf_addr
    }

def apply_atomic_rules(pages: List[Dict[str, Any]], meta: Dict[str, Any], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    atomics = rules.get("atomic_rules", [])
    for r in atomics:
        name = r.get("name","unnamed_rule")
        when = r.get("when", {})
        severity = r.get("severity","major")
        # Match metadata constraints (all provided keys must match one of values)
        ok = True
        for k,v in when.items():
            mv = str(meta.get(k,""))
            vals = [str(x) for x in (v if isinstance(v, list) else [v])]
            if mv not in vals:
                ok = False
                break
        if not ok:
            continue
        must_inc = r.get("must_include", {})
        pages_list = must_inc.get("pages", []) or list(range(1, len(pages)+1))
        patterns = must_inc.get("patterns", [])
        for pno in pages_list:
            page = pages[pno-1]["page"] if 1<=pno<=len(pages) else None
            text = pages[pno-1]["text"] if 1<=pno<=len(pages) else ""
            for pat in patterns:
                if pat and (pat not in text):
                    out.append({
                        "type":"rule",
                        "severity": severity,
                        "page": pno,
                        "message": f"Required text not found: '{pat}' (Rule: {name})",
                        "rule_name": name,
                        "snippet": pat
                    })
                elif pat and page:
                    # If present, also record as a pass pinpoint (optional)
                    pass
    return out

def run_checks(pages: List[Dict[str, Any]], meta: Dict[str, Any], rules: Dict[str, Any], do_spell: bool, allow_words: List[str]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    # Site Address rule
    addr_f = address_match_check(pages, meta.get("site_address","").strip())
    if addr_f:
        findings.append(addr_f)

    # Atomic rules from YAML
    findings.extend(apply_atomic_rules(pages, meta, rules))

    # Spelling (optional)
    if do_spell:
        findings.extend(spelling_findings(pages, allow_words))

    return findings

# --------------------------- UI HELPERS ---------------------------

def logo_css(logo_path: Optional[str]) -> str:
    if not logo_path or not os.path.exists(logo_path):
        return ""
    with open(logo_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    # top-left, responsive height
    css = f"""
    <style>
    .top-left-logo {{
        position: fixed;
        top: 14px;
        left: 16px;
        z-index: 9999;
        height: 42px;
        width: auto;
    }}
    .block-container {{ padding-top: 70px; }}
    </style>
    <img src="data:image/png;base64,{b64}" class="top-left-logo" />
    """
    return css

def gate() -> bool:
    st.title("AI Design Quality Auditor")
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if st.session_state.authed:
        return True
    pw = st.text_input("Enter access password", type="password")
    if st.button("Unlock"):
        if pw == ENTRY_PASSWORD:
            st.session_state.authed = True
            return True
        else:
            st.error("Incorrect password.")
    st.stop()  # end exec for unauthenticated users

def meta_form(rules: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    suppliers = rules.get("metadata_options", {}).get("suppliers", FALLBACK_SUPPLIERS)
    mimo_opts = rules.get("metadata_options", {}).get("mimo_options", MIMO_OPTIONS)

    with st.expander("Audit Metadata (required)", expanded=True):
        c1,c2,c3 = st.columns(3)
        with c1:
            meta["supplier"] = st.selectbox("Supplier", options=suppliers, index=0)
            meta["client"]   = st.selectbox("Client", options=["BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"])
            meta["site_type"]= st.selectbox("Site Type", options=SITE_TYPES)
        with c2:
            meta["project"]  = st.selectbox("Project", options=PROJECTS)
            meta["vendor"]   = st.selectbox("Proposed Vendor", options=VENDORS)
            meta["cabinet_location"] = st.selectbox("Proposed Cabinet Location", options=CAB_LOCATIONS)
        with c3:
            meta["radio_location"] = st.selectbox("Proposed Radio Location", options=RADIO_LOCATIONS)
            meta["drawing_type"]   = st.selectbox("Drawing Type", options=DRAWING_TYPES)
            meta["sectors_qty"]    = st.selectbox("Quantity of Sectors", options=SECTOR_QTY, index=2)

        meta["site_address"] = st.text_input("Site Address", placeholder="e.g. MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ").strip()

        # MIMO per sector (hide when Power Resilience)
        if meta["project"] != "Power Resilience":
            st.markdown("**MIMO Config per Sector**")
            c4,c5 = st.columns([1,1])
            with c4:
                copy_all = st.checkbox("Use S1 config for all sectors?")
            # build sector configs
            sectors = {}
            s1_val = st.selectbox("S1 MIMO Config", options=mimo_opts, index=0, key="mimo_s1")
            sectors["S1"] = s1_val
            for i in range(2, meta["sectors_qty"]+1):
                if copy_all:
                    sectors[f"S{i}"] = s1_val
                else:
                    sectors[f"S{i}"] = st.selectbox(f"S{i} MIMO Config", options=mimo_opts, index=0, key=f"mimo_s{i}")
            meta["mimo_by_sector"] = sectors
        else:
            meta["mimo_by_sector"] = {}
            st.caption("Proposed MIMO Config is optional for Power Resilience and is hidden.")

    # validation
    required_keys = ["supplier","client","project","site_type","vendor","cabinet_location","radio_location","drawing_type","sectors_qty","site_address"]
    missing = [k for k in required_keys if not meta.get(k)]
    if missing:
        st.error(f"Please fill in all required fields: {', '.join(missing)}")
    return meta

# --------------------------- TABS ---------------------------

def audit_tab():
    rules = load_rules()
    with st.sidebar:
        st.subheader("Settings")
        st.toggle("Enable spelling check (slower)", key="spell_on", value=False)
        st.number_input("Keep results visible (days)", min_value=1, max_value=60, value=DEFAULT_UI_KEEP_DAYS, key="keep_days")
        logo_path = st.text_input("Logo file path (optional)", value=st.session_state.get("logo_path",""))
        if st.button("Apply logo"):
            st.session_state["logo_path"] = logo_path
        st.text_input("Rules password (for Training tab)", type="password", key="yaml_pw")

    # Logo render
    if st.session_state.get("logo_path"):
        st.markdown(logo_css(st.session_state["logo_path"]), unsafe_allow_html=True)

    st.header("Audit")
    meta = meta_form(rules)

    up = st.file_uploader("Upload PDF drawing", type=["pdf"])
    c1,c2,c3,c4 = st.columns([1,1,1,1])
    run = c1.button("Run Audit", type="primary", use_container_width=True, disabled=up is None)
    clear_meta = c2.button("Clear Metadata", use_container_width=True)
    exclude_flag = c3.checkbox("Exclude this run from analytics", value=False)
    # We'll set after results
    spell_on = st.session_state.get("spell_on", False)

    if clear_meta:
        for k in list(st.session_state.keys()):
            if k.startswith("mimo_s"):
                del st.session_state[k]
        st.session_state.pop("last_results", None)
        st.session_state.pop("last_files", None)
        st.info("Cleared metadata of MIMO selections and last results.")

    if run and up:
        ensure_dirs()
        raw = up.read()
        pages, _doc = read_pdf_text_pages(raw)
        allow = rules.get("allowed_words", [])
        findings = run_checks(pages, meta, rules, do_spell=spell_on, allow_words=allow)
        # Status & RFT
        major = sum(1 for f in findings if f.get("severity")=="major")
        minor = sum(1 for f in findings if f.get("severity")=="minor")
        status = "Rejected" if major>0 else "Pass"
        rft = (len(findings)==0)

        # outputs
        excel_bytes, excel_name = make_excel(findings, meta, up.name, status)
        pdf_annot = annotate_pdf(raw, findings) if findings else raw
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(up.name)[0]
        ann_name = f"{clean_filename(base)}_{status}_{ts}_ANNOTATED.pdf"

        # persist to disk
        keep_file_for_days(REPORTS_DIR, st.session_state.get("keep_days", DEFAULT_UI_KEEP_DAYS))
        excel_path = os.path.join(REPORTS_DIR, excel_name)
        pdf_path = os.path.join(REPORTS_DIR, ann_name)
        with open(excel_path, "wb") as f: f.write(excel_bytes)
        with open(pdf_path, "wb") as f: f.write(pdf_annot)

        # history row
        row = {
            "timestamp_utc": ts_str(),
            "supplier": meta.get("supplier",""),
            "client": meta.get("client",""),
            "project": meta.get("project",""),
            "status": status,
            "pdf_name": ann_name,
            "excel_name": excel_name,
            "exclude_from_analytics": bool(exclude_flag),
            "minor": minor,
            "major": major,
            "total": minor+major,
            "rft": rft,
        }
        append_history(row)

        # store in session for further downloads
        st.session_state["last_results"] = {
            "findings": findings,
            "meta": meta,
            "status": status,
            "summary": {"minor":minor,"major":major,"total":minor+major,"rft":rft}
        }
        st.session_state["last_files"] = {
            "excel_bytes": excel_bytes,
            "excel_name": excel_name,
            "pdf_bytes": pdf_annot,
            "pdf_name": ann_name
        }

    # Show results panel (persisting across downloads)
    if "last_results" in st.session_state and "last_files" in st.session_state:
        res = st.session_state["last_results"]
        files = st.session_state["last_files"]
        st.subheader(f"Results — **{res['status']}**")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Major", res["summary"]["major"])
        m2.metric("Minor", res["summary"]["minor"])
        m3.metric("Total", res["summary"]["total"])
        m4.metric("RFT", "Yes" if res["summary"]["rft"] else "No")
        st.dataframe(pd.DataFrame(res["findings"]), use_container_width=True, height=300)

        cdl1, cdl2, cdl3 = st.columns([1,1,1])
        cdl1.download_button("Download Excel", data=files["excel_bytes"], file_name=files["excel_name"], mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        cdl2.download_button("Download Annotated PDF", data=files["pdf_bytes"], file_name=files["pdf_name"], mime="application/pdf", use_container_width=True)

        # Zip both
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(files["excel_name"], files["excel_bytes"])
            zf.writestr(files["pdf_name"], files["pdf_bytes"])
        zbuf.seek(0)
        cdl3.download_button("Download Both (.zip)", data=zbuf.getvalue(), file_name=f"Audit_Package_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip", use_container_width=True)

        st.caption("Results are also saved under `storage/reports/` and kept for the number of days set in Settings.")

def parse_training_table(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Expected columns (any subset): rule_name, decision (Valid/Not Valid), pattern, page, comment,
    scope_client, scope_project, scope_vendor, scope_site_type, severity
    """
    out = {"allow_words": set(), "atomic_rules": []}
    for _, row in df.iterrows():
        decision = str(row.get("decision","")).strip().lower()
        pat = str(row.get("pattern","")).strip()
        rule_name = str(row.get("rule_name","AutoRule")).strip() or "AutoRule"
        scope = {}
        for k_csv, k_meta in [
            ("scope_client","client"),
            ("scope_project","project"),
            ("scope_vendor","vendor"),
            ("scope_site_type","site_type"),
        ]:
            v = str(row.get(k_csv,"")).strip()
            if v:
                scope[k_meta] = [x.strip() for x in v.split("|") if x.strip()]
        severity = row.get("severity","major")
        if decision in ["valid","allow","whitelist"] and pat:
            # treat as allowed word (spelling context) or a no-op? We'll push to allowed words.
            out["allow_words"].add(pat)
        elif decision in ["not valid","reject","must include"] and pat:
            out["atomic_rules"].append({
                "name": rule_name,
                "when": scope,
                "must_include": {"patterns":[pat]},
                "severity": str(severity or "major")
            })
    return out

def training_tab():
    st.header("Training")
    st.write("Upload a **previous audit CSV/Excel** or paste notes to teach the tool quickly. Apply changes with the YAML password.")

    uptrain = st.file_uploader("Upload training table (CSV/XLSX)", type=["csv","xlsx"])
    pasted = st.text_area("Paste rejection lines (optional)", placeholder="e.g.\n233 - GPS cable type does not match schedule on 245\n...")
    parsed = None

    if uptrain:
        try:
            if uptrain.name.lower().endswith(".csv"):
                df = pd.read_csv(uptrain)
            else:
                df = pd.read_excel(uptrain)
            st.success(f"Loaded table with {len(df)} rows.")
            st.dataframe(df, use_container_width=True, height=300)
            parsed = parse_training_table(df)
        except Exception as e:
            st.error(f"Could not read training file: {e}")

    elif pasted.strip():
        # Very simple parser: "<num> - <text>"
        rows = []
        for line in pasted.strip().splitlines():
            m = re.match(r"(\d+)\s*[-–]\s*(.+)", line.strip())
            if m:
                rows.append({"decision":"not valid","pattern": m.group(2),"rule_name": f"Note_{m.group(1)}"})
        df = pd.DataFrame(rows)
        if not df.empty:
            st.dataframe(df, use_container_width=True, height=300)
            parsed = parse_training_table(df)
        else:
            st.warning("No parseable lines found.")

    if parsed:
        st.info(f"Parsed: {len(parsed['atomic_rules'])} rule(s) and {len(parsed['allow_words'])} allow word(s).")
        if st.text_input("Confirm YAML/Settings password to apply", type="password", key="yaml_pw_apply") == YAML_PASSWORD:
            if st.button("Apply to rules.yaml", type="primary"):
                rules = load_rules()
                # merge allow words
                aw = set(rules.get("allowed_words", []))
                aw |= parsed["allow_words"]
                rules["allowed_words"] = sorted(aw)
                # extend atomic rules
                rules.setdefault("atomic_rules", [])
                rules["atomic_rules"].extend(parsed["atomic_rules"])
                save_rules(rules)
                st.success(f"Applied. Rules file updated with {len(parsed['atomic_rules'])} rule(s) and {len(parsed['allow_words'])} allow word(s).")
        else:
            st.caption("Enter the YAML password to enable the apply button.")

def analytics_tab():
    st.header("Analytics")
    dfh = load_history_df()
    if dfh.empty:
        st.info("No history yet.")
        return

    # Filters
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        dt_from = st.date_input("From", value=(dt.datetime.utcnow()-dt.timedelta(days=30)).date())
    with c2:
        dt_to   = st.date_input("To", value=dt.datetime.utcnow().date())
    with c3:
        client_f = st.multiselect("Client", sorted(dfh["client"].dropna().unique().tolist()))
    with c4:
        project_f = st.multiselect("Project", sorted(dfh["project"].dropna().unique().tolist()))
    supplier_f = st.multiselect("Supplier", sorted(dfh["supplier"].dropna().unique().tolist()))

    show = dfh.copy()
    show = show[~show["exclude_from_analytics"]]
    if dt_from:
        show = show[show["timestamp_utc"] >= pd.to_datetime(str(dt_from))]
    if dt_to:
        show = show[show["timestamp_utc"] <= pd.to_datetime(str(dt_to) + " 23:59:59")]
    if client_f:
        show = show[show["client"].isin(client_f)]
    if project_f:
        show = show[show["project"].isin(project_f)]
    if supplier_f:
        show = show[show["supplier"].isin(supplier_f)]

    # KPIs
    total = len(show)
    passes = (show["status"]=="Pass").sum() if "status" in show.columns else 0
    rft_pct = (passes/total*100.0) if total>0 else 0.0
    majors = show["major"].sum() if "major" in show.columns else 0
    minors = show["minor"].sum() if "minor" in show.columns else 0

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Audits", total)
    m2.metric("RFT %", f"{rft_pct:.1f}%")
    m3.metric("Majors", int(majors))
    m4.metric("Minors", int(minors))

    # Trend (daily RFT)
    if not show.empty and "timestamp_utc" in show.columns:
        trend = show.copy()
        trend["date"] = pd.to_datetime(trend["timestamp_utc"]).dt.date
        grp = trend.groupby("date").apply(lambda d: (d["status"]=="Pass").mean()*100.0).reset_index(name="RFT%")
        st.line_chart(grp.set_index("date")["RFT%"])

    # Table
    cols = [c for c in ["timestamp_utc","supplier","client","project","status","pdf_name","excel_name"] if c in show.columns]
    if cols:
        st.dataframe(show[cols].sort_values("timestamp_utc", ascending=False), use_container_width=True, height=300)

    # Quick links to recent report files
    if os.path.exists(REPORTS_DIR):
        st.subheader("Recent files (local storage)")
        files = sorted([f for f in os.listdir(REPORTS_DIR) if os.path.isfile(os.path.join(REPORTS_DIR, f))], reverse=True)[:30]
        for f in files:
            fp = os.path.join(REPORTS_DIR, f)
            with open(fp, "rb") as fh:
                data = fh.read()
            st.download_button(f"Download {f}", data=data, file_name=f, key=f"dl_{f}")

def settings_note():
    with st.sidebar.expander("Advanced (YAML & Options)", expanded=False):
        st.caption("Supplier & MIMO lists come from `rules_example.yaml > metadata_options`.")
        if st.text_input("YAML password", type="password", key="yaml_pw2") == YAML_PASSWORD:
            if st.button("Reload rules from file"):
                st.session_state["rules_cache"] = load_rules()
                st.success("Reloaded rules.")
            if st.button("Open rules file as text"):
                try:
                    with open(RULES_FILE,"r",encoding="utf-8") as f:
                        st.code(f.read(), language="yaml")
                except Exception as e:
                    st.error(f"Cannot read rules file: {e}")
        else:
            st.caption("Enter YAML password to view/edit rules file.")

# --------------------------- MAIN ---------------------------

def main():
    ensure_dirs()
    if not gate():
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔎 Audit","🧠 Training","📊 Analytics"])

    with tab1:
        audit_tab()
    with tab2:
        training_tab()
    with tab3:
        analytics_tab()

    settings_note()

if __name__ == "__main__":
    st.set_page_config(page_title="AI Design Quality Auditor", layout="wide")
    main()
