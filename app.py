# app.py
from __future__ import annotations
import os, io, re, json, base64, textwrap, hashlib, time, shutil
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import yaml

# Heavy deps guarded (fitz may not be available locally)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from rapidfuzz import process, fuzz

# ========== CONSTANTS / PATHS ==========
APP_TITLE = "AI Design Quality Auditor"
PASSWORD = "vanB3lkum21"

VAULT_DIR = "vault"           # where runs (xlsx + annotated pdf) are persisted
EXPORT_DIR = "exports"        # daily exported CSV snapshots
HISTORY_CSV = os.path.join(VAULT_DIR, "history.csv")

DEFAULT_UI_VISIBILITY_HOURS = 24  # how long runs stay visible in the UI list
DATA_RETENTION_DAYS = 14          # hard retention in storage
AUTO_EXPORT_ONCE_PER_DAY = True   # pseudo-cron: export at first run after midnight

# ========== UTIL FS ==========
os.makedirs(VAULT_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def ts_iso(dt: datetime | None = None) -> str:
    return (dt or utc_now()).isoformat()

def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s.strip())
    return s[:180] or "unnamed"

# ========== AUTH ==========
def password_gate():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if st.session_state.auth_ok:
        return True

    st.title(APP_TITLE)
    st.caption("🔒 Secure access required")
    pwd = st.text_input("Enter password", type="password")
    if st.button("Unlock"):
        if pwd == PASSWORD:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# ========== LOGO ==========
def top_right_logo(logo_path: str):
    if not logo_path or not os.path.exists(logo_path):
        return
    try:
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .top-right-logo {{
                position: fixed;
                top: 12px;
                right: 16px;
                width: 140px;
                height: auto;
                z-index: 9999;
                opacity: 0.95;
            }}
            @media (max-width: 768px) {{
              .top-right-logo {{ width: 100px; }}
            }}
            </style>
            <img src="data:image/png;base64,{b64}" class="top-right-logo">
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

# ========== RULES LOADER ==========
def load_rules(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        data = f.read()
    # Normalize Windows line endings and escape stray backslashes if needed
    txt = data.decode(errors="ignore").replace("\r\n", "\n")
    try:
        obj = yaml.safe_load(txt) or {}
    except yaml.YAMLError:
        # attempt to sanitize unknown escapes in quoted strings
        txt2 = re.sub(r'\\(?![nrt"\\])', r'\\\\', txt)
        obj = yaml.safe_load(txt2) or {}
    if not isinstance(obj, dict):
        obj = {}
    # Ensure expected containers
    obj.setdefault("allowlist", [])
    obj.setdefault("rules", [])
    obj.setdefault("relationships", [])
    obj.setdefault("checklist", [])
    return obj

# ========== PDF TEXT & BBOX ==========
def extract_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Returns list of pages with text and word-level bbox.
    [{page_num:int, text:str, words:[(w, x0,y0,x1,y1)]}]
    """
    pages = []
    if not fitz:
        # Minimal fallback
        return [{"page_num": 1, "text": "", "words": []}]
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        words = page.get_text("words")  # list of tuples: (x0,y0,x1,y1, word, block_no, line_no, word_no)
        words_norm = []
        txt_parts = []
        for w in words:
            x0, y0, x1, y1, word, *_ = w
            words_norm.append((word, float(x0), float(y0), float(x1), float(y1)))
            txt_parts.append(word)
        pages.append({
            "page_num": i+1,
            "text": " ".join(txt_parts),
            "words": words_norm
        })
    doc.close()
    return pages

# ========== SPELLING (robust, allowlist-aware) ==========
def build_vocab(pages: List[Dict[str, Any]], allow: set[str]) -> List[str]:
    # naive vocab from doc + allowlist to help suggestions
    words = []
    for p in pages:
        for w in re.findall(r"[A-Za-z][A-Za-z\-']{1,}", p.get("text","")):
            words.append(w.lower())
    base = list(set(words) | allow)
    return base or list(allow)

def spelling_findings(pages: List[Dict[str, Any]], allowlist: List[str]) -> List[Dict[str, Any]]:
    allow = set([w.strip().lower() for w in allowlist if w])
    vocab = build_vocab(pages, allow)
    findings = []
    for p in pages:
        page_num = p["page_num"]
        for (w, x0,y0,x1,y1) in p.get("words", []):
            wl = w.lower()
            if not re.match(r"^[A-Za-z][A-Za-z\-']*$", w):
                continue
            if wl in allow:
                continue
            # Very simple heuristic: suspicious if mixedcase weirdly or uncommon short edits
            if len(w) <= 2:
                continue
            # Suggest from vocab (which includes allowlist) to avoid crashy .candidates
            suggestion = None
            try:
                best = process.extractOne(wl, vocab, scorer=fuzz.WRatio, score_cutoff=87)
                if best:
                    suggestion = best[0]
            except Exception:
                suggestion = None
            findings.append({
                "category": "Spelling",
                "severity": "Minor",
                "message": f"Possible misspelling: '{w}'",
                "suggestion": suggestion,
                "page": page_num,
                "bbox": [x0,y0,x1,y1],
                "rule_id": "spelling_generic"
            })
    return findings

# ========== RULE ENGINE ==========
def normalize_site_address(addr: str) -> str:
    if not addr:
        return ""
    # drop literal ", 0 ," tokens & extra spaces
    parts = [p.strip() for p in addr.split(",")]
    parts = [p for p in parts if p != "0"]
    return ", ".join([p for p in parts if p])

def title_matches_address(pdf_title: str, site_addr: str) -> bool:
    # loose compare: all address tokens must be found in title (order-agnostic)
    A = normalize_site_address(site_addr).upper()
    if not A:
        return True  # nothing to enforce
    tokens = [t for t in re.split(r"[\s,]+", A) if t and t not in {"ROAD","STREET","AVENUE","LANE"}]
    T = re.sub(r"\s+", " ", pdf_title.upper())
    return all(tok in T for tok in tokens)

def project_hides_mimo(project: str) -> bool:
    return (project or "").strip().lower() in {"power resilience", "power res", "power resillience"}

def run_rules(
    pages: List[Dict[str, Any]],
    rules_blob: Dict[str, Any],
    metadata: Dict[str, Any],
    pdf_title: str
) -> List[Dict[str, Any]]:
    findings = []

    # 1) Site address vs title
    site_addr = metadata.get("site_address","")
    if site_addr and pdf_title:
        if not title_matches_address(pdf_title, site_addr):
            findings.append({
                "category": "Metadata",
                "severity": "Major",
                "message": "Site Address does not match the PDF title.",
                "suggestion": f"Ensure title contains: {normalize_site_address(site_addr)}",
                "page": 1,
                "bbox": None,
                "rule_id": "meta_address_vs_title"
            })

    # 2) Forbidden relationships (e.g., Brush vs Generator Power) from rules YAML
    for rel in rules_blob.get("relationships", []):
        # each rel: {"if_contains":"Brush", "forbid":"Generator Power", "severity":"Major"}
        ic = (rel.get("if_contains") or "").strip()
        fb = (rel.get("forbid") or "").strip()
        if not ic or not fb:
            continue
        text_all = " ".join([p.get("text","") for p in pages]).upper()
        if ic.upper() in text_all and fb.upper() in text_all:
            findings.append({
                "category": "Relationship",
                "severity": rel.get("severity","Major"),
                "message": f"'{fb}' is not allowed when '{ic}' is present.",
                "suggestion": "Remove the forbidden pairing or adjust design.",
                "page": 1,
                "bbox": None,
                "rule_id": f"rel_{ic}_{fb}"
            })

    # 3) Regex/content rules from YAML
    for rule in rules_blob.get("rules", []):
        pattern = rule.get("pattern")
        if not pattern:
            continue
        sev = rule.get("severity","Minor")
        msg = rule.get("message","Rule match")
        rx = re.compile(pattern, re.IGNORECASE)
        for p in pages:
            text = p.get("text","")
            if rx.search(text):
                # try to get bbox for the first matching word (best effort)
                bbox = None
                try:
                    word = rx.search(text).group(0)
                    for (w,x0,y0,x1,y1) in p.get("words", []):
                        if w.lower() == word.lower():
                            bbox = [x0,y0,x1,y1]; break
                except Exception:
                    pass
                findings.append({
                    "category": rule.get("category","Rule"),
                    "severity": sev,
                    "message": msg,
                    "suggestion": rule.get("suggestion"),
                    "page": p["page_num"],
                    "bbox": bbox,
                    "rule_id": rule.get("id", f"rule_{pattern}")
                })

    return findings

# ========== ANNOTATION ==========
def annotate_pdf(pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    if not fitz:
        return pdf_bytes
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for f in findings:
        page_idx = max(0, int(f.get("page",1)) - 1)
        if page_idx >= len(doc):
            continue
        page = doc[page_idx]
        note = f"{f.get('category','')}: {f.get('message','')}"
        bbox = f.get("bbox")
        if bbox and len(bbox)==4:
            rect = fitz.Rect(*bbox)
            try:
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=(1,0,0))
                annot.set_border(width=0.8, dashes=[2,2])
                annot.update()
                page.add_text_annot(rect.br, note)
            except Exception:
                # fallback to a sticky note
                page.add_text_annot(rect.tl, note)
        else:
            # place a sticky note at top-left margin
            page.add_text_annot(fitz.Point(36, 36), note)
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()

# ========== REPORT ==========
def build_report_df(findings: List[Dict[str, Any]], metadata: Dict[str, Any], file_name: str) -> pd.DataFrame:
    rows = []
    for f in findings:
        rows.append({
            "file_name": file_name,
            "category": f.get("category"),
            "severity": f.get("severity"),
            "message": f.get("message"),
            "suggestion": f.get("suggestion"),
            "page": f.get("page"),
            "bbox": json.dumps(f.get("bbox")),
            "rule_id": f.get("rule_id"),
            # metadata echoes
            **{f"meta_{k}": v for k,v in metadata.items()}
        })
    df = pd.DataFrame(rows)
    if df.empty:
        # still output a “pass” sheet downstream
        df = pd.DataFrame(columns=[
            "file_name","category","severity","message","suggestion","page","bbox","rule_id"
        ] + [f"meta_{k}" for k in metadata.keys()])
    return df

def save_report_and_annotation(df: pd.DataFrame, annotated_pdf: bytes | None, file_name: str, passed: bool) -> Tuple[str, Optional[str]]:
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    base = os.path.splitext(safe_filename(file_name))[0]
    verdict = "PASS" if passed else "REJECTED"
    xlsx_path = os.path.join(VAULT_DIR, f"{base}__{verdict}__{stamp}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Findings")
    pdf_path = None
    if annotated_pdf:
        pdf_path = os.path.join(VAULT_DIR, f"{base}__{verdict}__{stamp}.annotated.pdf")
        with open(pdf_path, "wb") as f:
            f.write(annotated_pdf)
    return xlsx_path, pdf_path

# ========== HISTORY ==========
def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_CSV)
    except Exception:
        return pd.DataFrame()
    # timestamp_utc as tz-aware
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    return df

def push_history(record: Dict[str, Any]):
    df = load_history()
    record = record.copy()
    record["timestamp_utc"] = pd.to_datetime(record.get("timestamp_utc") or ts_iso(), utc=True)
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(HISTORY_CSV, index=False)

def filter_history_for_ui(df: pd.DataFrame, hours: int | None) -> pd.DataFrame:
    if df.empty or not hours:
        return df
    if "timestamp_utc" not in df.columns:
        return df
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp_utc"])
    cutoff_utc = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    return df[df["timestamp_utc"] >= cutoff_utc]

def enforce_retention():
    # purge files older than DATA_RETENTION_DAYS
    cutoff = utc_now() - timedelta(days=DATA_RETENTION_DAYS)
    # history thinning
    df = load_history()
    if not df.empty:
        df_keep = df[pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True) >= cutoff]
        df_keep.to_csv(HISTORY_CSV, index=False)
    # vault files
    for root in (VAULT_DIR, EXPORT_DIR):
        for f in os.listdir(root):
            p = os.path.join(root, f)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(p), tz=timezone.utc)
                if mtime < cutoff:
                    os.remove(p)
            except Exception:
                pass

def maybe_daily_export():
    if not AUTO_EXPORT_ONCE_PER_DAY:
        return
    flag_path = os.path.join(EXPORT_DIR, "last_export_date.txt")
    today = utc_now().date().isoformat()
    last = None
    if os.path.exists(flag_path):
        with open(flag_path,"r") as f:
            last = f.read().strip()
    if last == today:
        return
    # perform export
    df = load_history()
    if not df.empty:
        out = os.path.join(EXPORT_DIR, f"audit_history_{today}.csv")
        df.to_csv(out, index=False)
    with open(flag_path, "w") as f:
        f.write(today)

# ==========\ STREAMLIT UI ==========
password_gate()

st.set_page_config(page_title=APP_TITLE, layout="wide")
top_right_logo("88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png")

enforce_retention()
maybe_daily_export()

with st.sidebar:
    st.header("Settings")
    rules_file = st.file_uploader("Rules YAML (rules_example.yaml)", type=["yaml", "yml"])
    allow_edit = st.checkbox("Show raw rules JSON preview", value=False)
    ui_hours = st.number_input("UI Visibility Window (hours)", 1, 336, value=DEFAULT_UI_VISIBILITY_HOURS)
    st.session_state["ui_visibility_hours"] = int(ui_hours)

st.title(APP_TITLE)
st.caption("Professional, fast, and auditable design checks — with learning feedback loops.")

# ---------- Metadata (all mandatory unless noted) ----------
colA, colB, colC, colD = st.columns(4)

with colA:
    supplier = st.selectbox("Supplier (mandatory)", ["— Select —","Cellnex","Cornerstone","Vodafone","MBNL","BTEE","H3G","Other"])
with colB:
    drawing_type = st.selectbox("Drawing Type (mandatory)", ["— Select —","General Arrangement","Detailed Design"])
with colC:
    client = st.selectbox("Client (mandatory)", ["— Select —","BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"])
with colD:
    project = st.selectbox("Project (mandatory)", ["— Select —","RAN","Power Resilience","East Unwind","Beacon 4"])

colE, colF, colG, colH = st.columns(4)
with colE:
    site_type = st.selectbox("Site Type (mandatory)", ["— Select —","Greenfield","Rooftop","Streetworks"])
with colF:
    vendor = st.selectbox("Proposed Vendor (mandatory)", ["— Select —","Ericsson","Nokia"])
with colG:
    cabinet_loc = st.selectbox("Proposed Cabinet Location (mandatory)", ["— Select —","Indoor","Outdoor"])
with colH:
    radio_loc = st.selectbox("Proposed Radio Location (mandatory)", ["— Select —","High Level","Low Level","Indoor","Door"])

colI, colJ = st.columns(2)
with colI:
    sectors = st.selectbox("Quantity of Sectors (mandatory)", ["— Select —","1","2","3","4","5","6"])
with colJ:
    site_address = st.text_input("Site Address (mandatory)", placeholder="e.g., MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ")

# MIMO block (hidden for Power Resilience)
show_mimo = not project_hides_mimo(project)
if show_mimo:
    st.markdown("#### Proposed MIMO Config (optional unless required by client)")
    same_mimo = st.checkbox("Use same MIMO config for all sectors", value=True)
    mimo_options = [
        "18\\21\\26 @4x4; 70\\80 @2x2",
        "18\\21 @2x2",
        "18\\21\\26 @4x4; 3500 @8x8",
        "18\\21\\26 @4x4",
    ]
    colM1, colM2, colM3 = st.columns(3)
    mimo_s1 = colM1.selectbox("Proposed Mimo S1", mimo_options, index=0)
    mimo_s2 = colM2.selectbox("Proposed Mimo S2", mimo_options, index=0 if same_mimo else 1, disabled=same_mimo)
    mimo_s3 = colM3.selectbox("Proposed Mimo S3", mimo_options, index=0 if same_mimo else 2, disabled=same_mimo)
else:
    mimo_s1 = mimo_s2 = mimo_s3 = None

# File upload
st.markdown("---")
pdf_file = st.file_uploader("Upload a PDF to audit", type=["pdf"])

# Controls
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1,1,1,2])
with col_btn1:
    audit_clicked = st.button("▶️ Run Audit", use_container_width=True)
with col_btn2:
    clear_meta = st.button("🧹 Clear Metadata", use_container_width=True)
with col_btn3:
    export_now = st.button("📤 Export History Now", use_container_width=True)
with col_btn4:
    exclude_analytics = st.checkbox("Exclude this run from analytics (while training rules)")

if clear_meta:
    for k in ["supplier","drawing_type","client","project","site_type","vendor","cabinet_loc","radio_loc","sectors","site_address"]:
        st.session_state.pop(k, None)
    st.rerun()

if export_now:
    dfh = load_history()
    if not dfh.empty:
        out = os.path.join(EXPORT_DIR, f"audit_history_{utc_now().date().isoformat()}_manual.csv")
        dfh.to_csv(out, index=False)
        st.success(f"History exported: {out}")
    else:
        st.info("No history yet to export.")

# Validate rules
rules_blob = {}
if rules_file is not None:
    # write to temp and load
    tmp_path = os.path.join(VAULT_DIR, f"rules_{int(time.time())}.yaml")
    with open(tmp_path, "wb") as f:
        f.write(rules_file.read())
    rules_blob = load_rules(tmp_path)
else:
    # allow empty but keep structure
    rules_blob = {"allowlist": [], "rules": [], "relationships": [], "checklist": []}

if allow_edit:
    with st.expander("Rules Preview (read-only)", expanded=False):
        st.json(rules_blob)

# Mandatory checks
def all_meta_filled() -> Tuple[bool, str]:
    if supplier == "— Select —": return False, "Supplier is required."
    if drawing_type == "— Select —": return False, "Drawing Type is required."
    if client == "— Select —": return False, "Client is required."
    if project == "— Select —": return False, "Project is required."
    if site_type == "— Select —": return False, "Site Type is required."
    if vendor == "— Select —": return False, "Proposed Vendor is required."
    if cabinet_loc == "— Select —": return False, "Proposed Cabinet Location is required."
    if radio_loc == "— Select —": return False, "Proposed Radio Location is required."
    if sectors == "— Select —": return False, "Quantity of Sectors is required."
    if not site_address.strip(): return False, "Site Address is required."
    return True, ""

# ========== AUDIT ==========
if audit_clicked:
    ok, msg = all_meta_filled()
    if not ok:
        st.error(msg)
        st.stop()
    if not pdf_file:
        st.error("Please upload a PDF to audit.")
        st.stop()

    original_bytes = pdf_file.read()
    file_name = pdf_file.name
    pdf_title_guess = os.path.splitext(file_name)[0]

    # Extract pages / text / bbox
    pages = extract_pages(original_bytes)

    # Findings
    findings = []
    findings += spelling_findings(pages, rules_blob.get("allowlist", []))
    findings += run_rules(
        pages,
        rules_blob,
        {
            "supplier": supplier,
            "drawing_type": drawing_type,
            "client": client,
            "project": project,
            "site_type": site_type,
            "vendor": vendor,
            "cabinet_location": cabinet_loc,
            "radio_location": radio_loc,
            "sectors": sectors,
            "site_address": site_address,
            "mimo_s1": mimo_s1,
            "mimo_s2": mimo_s2,
            "mimo_s3": mimo_s3,
        },
        pdf_title_guess
    )

    # Checklist (YAML) — add any hard fails if items missing
    for item in rules_blob.get("checklist", []):
        # {"id":"title_block_present", "must_contain":"TITLE BLOCK", "severity":"Major", "message":"Title block missing"}
        must = (item.get("must_contain") or "").strip()
        if must:
            text_all = " ".join([p.get("text","") for p in pages]).upper()
            if must.upper() not in text_all:
                findings.append({
                    "category": "Checklist",
                    "severity": item.get("severity","Major"),
                    "message": item.get("message","Checklist requirement not met"),
                    "suggestion": f"Add '{must}'",
                    "page": 1,
                    "bbox": None,
                    "rule_id": item.get("id","check_missing")
                })

    # Verdict
    major_cnt = sum(1 for f in findings if str(f.get("severity","")).lower()=="major")
    passed = (major_cnt == 0)

    # Annotate
    annotated_pdf = annotate_pdf(original_bytes, findings)
    # Build report
    meta_dict = {
        "supplier": supplier,
        "drawing_type": drawing_type,
        "client": client,
        "project": project,
        "site_type": site_type,
        "vendor": vendor,
        "cabinet_location": cabinet_loc,
        "radio_location": radio_loc,
        "sectors": sectors,
        "site_address": site_address,
        "mimo_s1": mimo_s1,
        "mimo_s2": mimo_s2,
        "mimo_s3": mimo_s3,
        "verdict": "PASS" if passed else "REJECTED",
    }
    df = build_report_df(findings, meta_dict, file_name)
    xlsx_path, pdf_path = save_report_and_annotation(df, annotated_pdf, file_name, passed)

    # Save to history
    rec = {
        "timestamp_utc": ts_iso(),
        "file_name": file_name,
        "supplier": supplier,
        "drawing_type": drawing_type,
        "client": client,
        "project": project,
        "site_type": site_type,
        "vendor": vendor,
        "cabinet_location": cabinet_loc,
        "radio_location": radio_loc,
        "sectors": sectors,
        "site_address": site_address,
        "mimo_s1": mimo_s1,
        "mimo_s2": mimo_s2,
        "mimo_s3": mimo_s3,
        "verdict": "PASS" if passed else "REJECTED",
        "findings_count": len(findings),
        "major_count": major_cnt,
        "exclude": bool(exclude_analytics),
        "report_path": xlsx_path,
        "annotated_pdf_path": pdf_path,
    }
    push_history(rec)

    # UI summary
    st.success("Audit complete.")
    st.subheader("Summary")
    colS1,colS2,colS3,colS4 = st.columns(4)
    colS1.metric("Findings", len(findings))
    colS2.metric("Major", major_cnt)
    colS3.metric("Verdict", "PASS" if passed else "REJECTED")
    colS4.metric("Supplier", supplier)

    st.subheader("Downloads")
    st.write(f"**Excel Report:** `{os.path.basename(xlsx_path)}`")
    with open(xlsx_path, "rb") as f:
        st.download_button("Download Excel Report", f, file_name=os.path.basename(xlsx_path), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.write(f"**Annotated PDF:** `{os.path.basename(pdf_path)}`")
    with open(pdf_path, "rb") as f:
        st.download_button("Download Annotated PDF", f, file_name=os.path.basename(pdf_path), mime="application/pdf")

# ---------- HISTORY / ANALYTICS ----------
st.markdown("---")
st.subheader("Recent Runs")
dfh = load_history()
if dfh.empty:
    st.info("No runs yet.")
else:
    # Visibility window
    df_vis = filter_history_for_ui(dfh.copy(), st.session_state.get("ui_visibility_hours", DEFAULT_UI_VISIBILITY_HOURS))
    # Respect exclude flag in analytics visuals (but still list files)
    df_analytics = df_vis[df_vis.get("exclude", False) != True]

    # Trendline / KPIs
    colA1, colA2, colA3, colA4 = st.columns(4)
    if not df_analytics.empty:
        rft = (df_analytics["verdict"] == "PASS").mean() * 100
        colA1.metric("Right First Time (%)", f"{rft:.1f}")
        colA2.metric("Major Findings", int(df_analytics["major_count"].sum()))
        colA3.metric("Total Findings", int(df_analytics["findings_count"].sum()))
        colA4.metric("Runs", int(len(df_vis)))
        # Simple trend by day & supplier
        df_analytics["day"] = pd.to_datetime(df_analytics["timestamp_utc"], utc=True).dt.tz_convert("UTC").dt.date
        daily = df_analytics.groupby(["day","supplier"])["major_count"].sum().reset_index()
        st.line_chart(daily.pivot(index="day", columns="supplier", values="major_count").fillna(0))
    else:
        st.caption("Analytics excluded or empty for the current window.")

    # Show table of runs (including excluded)
    show_cols = ["timestamp_utc","file_name","supplier","drawing_type","client","project","verdict","major_count","findings_count","exclude"]
    for c in show_cols:
        if c not in df_vis.columns:
            df_vis[c] = ""
    st.dataframe(df_vis[show_cols].sort_values("timestamp_utc", ascending=False), use_container_width=True)
