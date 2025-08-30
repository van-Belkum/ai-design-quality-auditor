# app.py
# AI Design Quality Auditor – full refreshed build
# Features:
# - Persistent outputs (/outputs), 14-day retention (configurable)
# - Midnight UK daily backup (/backups) with history CSV + outputs
# - UI visibility window (default 24h)
# - History with downloads, exclude toggle
# - Supplier & Drawing Type mandatory; Power Resilience hides MIMO field
# - Address/title match validation (ignores ", 0 ,")
# - Logo fixed top-right
# - PDF annotations + Excel report
# - Simple rule text area (password gated) – saves to rules_example.yaml

import os, io, re, json, zipfile
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd
import numpy as np
import yaml

import fitz  # PyMuPDF

from rapidfuzz import fuzz

# -----------------------------
# Constants & paths
# -----------------------------
LONDON_TZ = ZoneInfo("Europe/London")

ROOT = Path(".")
HISTORY_DIR = ROOT / "history"
HISTORY_DIR.mkdir(exist_ok=True)
HISTORY_PATH = HISTORY_DIR / "audit_history.csv"

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BACKUP_DIR = ROOT / "backups"
BACKUP_DIR.mkdir(exist_ok=True)

RULES_PATH = ROOT / "rules_example.yaml"

DEFAULT_RETENTION_DAYS = 14
MAX_OUTPUT_DIR_BYTES = 2 * 1024**3  # 2 GB
DEFAULT_UI_VISIBILITY_HOURS = 24
BACKUP_RETENTION_DAYS = 60

APP_PASSWORD = "vanB3lkum21"

# -----------------------------
# Small utilities
# -----------------------------
def now_uk():
    return datetime.now(tz=LONDON_TZ)

def utc_iso():
    return datetime.utcnow().isoformat()

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", (name or "").strip())
    return name[:180] if name else "file"

def ensure_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "timestamp_utc","file","client","project","site_type","vendor",
        "cabinet_loc","radio_loc","sectors","mimo_config","site_address",
        "supplier","drawing_type","used_ocr","pages",
        "minor_findings","major_findings","total_findings",
        "outcome","rft_percent","exclude",
        "excel_path","annotated_pdf_path"
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = "" if c not in ("pages","minor_findings","major_findings","total_findings","rft_percent","exclude") else (0 if c!="exclude" else False)
    return df

def ensure_history_csv():
    if not HISTORY_PATH.exists():
        df = pd.DataFrame(columns=[
            "timestamp_utc","file","client","project","site_type","vendor",
            "cabinet_loc","radio_loc","sectors","mimo_config","site_address",
            "supplier","drawing_type","used_ocr","pages",
            "minor_findings","major_findings","total_findings",
            "outcome","rft_percent","exclude",
            "excel_path","annotated_pdf_path"
        ])
        df.to_csv(HISTORY_PATH, index=False)

def save_bytes_to_outputs(filename: str, blob: bytes) -> str:
    filename = sanitize_filename(filename)
    path = OUTPUT_DIR / filename
    with open(path, "wb") as f:
        f.write(blob)
    return str(path)

def file_age_days(path: Path) -> float:
    try:
        mtime = path.stat().st_mtime
        return (datetime.utcnow() - datetime.utcfromtimestamp(mtime)).days
    except Exception:
        return 0.0

def outputs_size_bytes() -> int:
    total = 0
    for p in OUTPUT_DIR.glob("*"):
        try:
            if p.is_file(): total += p.stat().st_size
        except: pass
    return total

def size_human(nbytes: int) -> str:
    val = float(nbytes)
    for unit in ["B","KB","MB","GB","TB"]:
        if val < 1024.0:
            return f"{val:,.1f} {unit}"
        val /= 1024.0
    return f"{val:,.1f} PB"

def clean_outputs(retention_days: int = DEFAULT_RETENTION_DAYS,
                  max_bytes: int = MAX_OUTPUT_DIR_BYTES) -> dict:
    deleted_by_age, deleted_by_size = [], []
    for p in OUTPUT_DIR.glob("*"):
        if p.is_file() and file_age_days(p) > retention_days:
            try: p.unlink(); deleted_by_age.append(p.name)
            except: pass
    # size cap
    files = [p for p in OUTPUT_DIR.glob("*") if p.is_file()]
    files_sorted = sorted(files, key=lambda p: p.stat().st_mtime)
    total = outputs_size_bytes()
    i = 0
    while total > max_bytes and i < len(files_sorted):
        p = files_sorted[i]
        try:
            sz = p.stat().st_size
            p.unlink()
            deleted_by_size.append(p.name)
            total -= sz
        except: pass
        i += 1
    return {"deleted_by_age": deleted_by_age, "deleted_by_size": deleted_by_size}

def zipdir(ziph: zipfile.ZipFile, base_dir: Path, prefix: str):
    for p in base_dir.rglob("*"):
        if p.is_file():
            arcname = f"{prefix}/{p.relative_to(base_dir)}"
            try: ziph.write(p, arcname)
            except: pass

def export_daily_backup() -> str:
    ensure_history_csv()
    ts = now_uk().strftime("%Y%m%d")
    zip_path = BACKUP_DIR / f"backup_{ts}.zip"
    if zip_path.exists():
        return str(zip_path)
    meta = {"generated_at_uk": now_uk().isoformat(), "note": "Daily export of history & outputs"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if HISTORY_PATH.exists():
            z.write(HISTORY_PATH, arcname="history/audit_history.csv")
        zipdir(z, OUTPUT_DIR, "outputs")
        z.writestr("meta.json", json.dumps(meta, indent=2))
    return str(zip_path)

def clean_old_backups(retention_days: int = BACKUP_RETENTION_DAYS) -> dict:
    cutoff = now_uk() - timedelta(days=retention_days)
    deleted = []
    for p in BACKUP_DIR.glob("backup_*.zip"):
        try:
            m = re.search(r"backup_(\d{8})\.zip$", p.name)
            dt = datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=LONDON_TZ) if m else datetime.fromtimestamp(p.stat().st_mtime, tz=LONDON_TZ)
            if dt < cutoff:
                p.unlink(); deleted.append(p.name)
        except: pass
    return {"deleted": deleted}

def filter_history_for_ui(df: pd.DataFrame, hours: int | None) -> pd.DataFrame:
    if df.empty or not hours:
        return df
    if "timestamp_utc" not in df.columns: return df
    if not np.issubdtype(df["timestamp_utc"].dtype, np.datetime64):
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    cutoff_utc = datetime.utcnow() - timedelta(hours=hours)
    return df[df["timestamp_utc"] >= cutoff_utc]

# -----------------------------
# Rules I/O
# -----------------------------
def load_rules() -> dict:
    if not RULES_PATH.exists():
        # Create a minimal default
        base = {
            "allowlist": ["MBNL","Vodafone","Ericsson","Nokia","BTEE","Cellnex","H3G","Cornerstone"],
            "checks": {
                "spelling": {"enabled": True, "severity": "minor"},
                "address_title_match": {"enabled": True, "severity": "major"}
            }
        }
        RULES_PATH.write_text(yaml.safe_dump(base, sort_keys=False))
        return base
    try:
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception as e:
        st.error(f"YAML load error: {e}")
        return {}

def save_rules(yaml_text: str) -> tuple[bool,str]:
    try:
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            return False, "Top-level YAML must be a mapping (dict)."
        RULES_PATH.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        return True, "Rules saved."
    except Exception as e:
        return False, f"Parse error: {e}"

# -----------------------------
# PDF utilities
# -----------------------------
def extract_pdf_text_pages(pdf_bytes: bytes) -> list[str]:
    pages = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for p in doc:
                pages.append(p.get_text("text"))
    except Exception:
        pass
    return pages or [""]

def annotate_pdf(pdf_bytes: bytes, marks: list[dict]) -> bytes:
    """
    marks: [{page:int, bbox:(x0,y0,x1,y1)|None, note:str}]
    If no bbox, drop a sticky note at top-left margin.
    """
    bio = io.BytesIO()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for m in marks:
            page_idx = int(m.get("page",0))
            note = str(m.get("note","Finding"))
            bbox = m.get("bbox", None)
            if page_idx < 0 or page_idx >= len(doc):
                page_idx = 0
            page = doc[page_idx]
            try:
                if bbox and isinstance(bbox,(list,tuple)) and len(bbox)==4:
                    rect = fitz.Rect(*[float(v) for v in bbox])
                    try:
                        page.add_highlight_annot(rect)
                    except Exception:
                        pass
                    # Add a text annotation near bbox
                    pin = fitz.Point(rect.x0, max(10, rect.y0-8))
                    try:
                        page.add_text_annot(pin, note)
                    except Exception:
                        pass
                else:
                    # fallback top-left margin
                    pin = fitz.Point(36, 36)
                    try:
                        page.add_text_annot(pin, note)
                    except Exception:
                        pass
            except Exception:
                # never crash annotation
                pass
        doc.save(bio, deflate=True, garbage=4)
    return bio.getvalue()

# -----------------------------
# Auditing
# -----------------------------
def normalize_tokens(text: str) -> list[str]:
    # keep letter/digit words; strip punctuation
    return re.findall(r"[A-Za-z][A-Za-z0-9\-']{2,}", text)

def remove_address_zero_tokens(s: str) -> str:
    # remove lone ", 0 ," token pattern (with surrounding commas/space)
    s2 = re.sub(r"\s*,\s*0\s*,\s*", ", ", s)
    return re.sub(r"\s{2,}", " ", s2).strip()

def spelling_checks(pages: list[str], allowlist: set[str], severity="minor") -> list[dict]:
    """
    Conservative spelling checker:
    - Flags tokens with letters that aren't entirely uppercase acronyms
    - Skips allowlist items (case-insensitive)
    - No external dictionary to avoid brittleness; uses simple heuristics
    """
    findings = []
    for i, page in enumerate(pages):
        tokens = normalize_tokens(page)
        for t in tokens:
            low = t.lower()
            if low in allowlist:  # allowlisted (already lowered)
                continue
            if t.isupper() and len(t) <= 5:
                # short acronym – ignore
                continue
            # heuristic: suspicious if contains 3+ consecutive consonants and not hyphenated brandy term
            if re.search(r"[bcdfghjklmnpqrstvwxz]{4,}", low):
                findings.append({
                    "page": i,
                    "category": "Spelling",
                    "rule": "Heuristic misspelling",
                    "severity": severity,
                    "message": f"Possible misspelling: '{t}'",
                    "bbox": None
                })
    return findings

def title_address_check(title_text: str, site_address: str, severity="major") -> list[dict]:
    """
    Ensure address (minus ', 0 ,') appears in title or first page text.
    """
    if not site_address:
        return []
    addr_norm = remove_address_zero_tokens(site_address).upper()
    addr_norm = re.sub(r"\s+", " ", addr_norm)
    title_norm = re.sub(r"\s+", " ", (title_text or "")).upper()
    # fuzzy contains – require good ratio for a significant chunk
    ratio = fuzz.partial_ratio(addr_norm, title_norm)
    if ratio >= 90:
        return []
    return [{
        "page": 0,
        "category": "Metadata",
        "rule": "Address must match title",
        "severity": severity,
        "message": f"Site address not found in title (ratio {ratio}). Expected fragment: {addr_norm[:50]}...",
        "bbox": None
    }]

def audit_pdf(pdf_bytes: bytes, meta: dict, rules: dict) -> list[dict]:
    pages = extract_pdf_text_pages(pdf_bytes)
    first_page_text = pages[0] if pages else ""
    findings: list[dict] = []

    checks = (rules.get("checks") or {})
    allow = set([w.lower() for w in (rules.get("allowlist") or [])])

    # Spelling
    sc = checks.get("spelling", {})
    if sc.get("enabled", True):
        findings += spelling_checks(pages, allow, severity=sc.get("severity","minor"))

    # Address/Title match
    ac = checks.get("address_title_match", {})
    if ac.get("enabled", True):
        findings += title_address_check(first_page_text, meta.get("site_address",""), severity=ac.get("severity","major"))

    # Example rule: disallow pair "Brush" with "Generator Power" (user asked earlier)
    xr = checks.get("incompatible_terms", {"enabled": False})
    if xr.get("enabled", False):
        bad_pairs = xr.get("pairs", [["BRUSH","GENERATOR POWER"]])
        corpus = (" ".join(pages)).upper()
        for a,b in bad_pairs:
            if a.upper() in corpus and b.upper() in corpus:
                findings.append({
                    "page": 0,
                    "category": "Content",
                    "rule": "Incompatible terms",
                    "severity": xr.get("severity","major"),
                    "message": f"Incompatible terms found together: {a} and {b}",
                    "bbox": None
                })

    # Mandatory metadata
    required = ["client","project","site_type","vendor","cabinet_loc","radio_loc","sectors","supplier","drawing_type","site_address"]
    missing = [k for k in required if not str(meta.get(k,"")).strip()]
    if missing:
        findings.append({
            "page": 0,
            "category": "Metadata",
            "rule": "Missing required metadata",
            "severity": "major",
            "message": f"Missing: {', '.join(missing)}",
            "bbox": None
        })

    return findings

# -----------------------------
# Reporting (Excel + PDF)
# -----------------------------
def build_excel_and_pdf(file_name: str, pdf_bytes: bytes, findings: list[dict], meta: dict) -> tuple[bytes, bytes, str, str, str]:
    rows = []
    majors = 0; minors = 0
    for f in findings:
        if (f.get("severity") or "").lower() == "major": majors += 1
        else: minors += 1
        rows.append({
            "file": file_name,
            "category": f.get("category",""),
            "rule": f.get("rule",""),
            "severity": f.get("severity",""),
            "message": f.get("message",""),
            "page": f.get("page",0)
        })
    df = pd.DataFrame(rows) if rows else pd.DataFrame([{
        "file": file_name, "category":"", "rule":"", "severity":"", "message":"No findings", "page":""
    }])
    outcome = "PASS" if len(findings)==0 else "REJECTED"
    rft = 100.0 if len(findings)==0 else 0.0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(file_name).stem

    xlsx_name = f"{base}_{outcome}_{stamp}.xlsx"
    pdf_name  = f"{base}_ANNOTATED_{outcome}_{stamp}.pdf"

    # Excel bytes
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        meta_df = pd.DataFrame([meta])
        meta_df.to_excel(xw, index=False, sheet_name="Audit Meta")
        df.to_excel(xw, index=False, sheet_name="Findings")
        summary = pd.DataFrame([{
            "Outcome": outcome,
            "Minor Findings": minors,
            "Major Findings": majors,
            "Total Findings": len(findings),
            "RFT %": rft
        }])
        summary.to_excel(xw, index=False, sheet_name="Summary")
    excel_bytes = buf.getvalue()

    # Annotation
    marks = [{"page": f.get("page",0), "bbox": f.get("bbox"), "note": f.get("message","Finding")} for f in findings]
    annotated_pdf = annotate_pdf(pdf_bytes, marks)

    # Persist (respect retention elsewhere)
    excel_path = save_bytes_to_outputs(xlsx_name, excel_bytes)
    pdf_path   = save_bytes_to_outputs(pdf_name, annotated_pdf)

    return excel_bytes, annotated_pdf, outcome, excel_path, pdf_path

def push_history(file_name: str, findings: list[dict], outcome: str, pages: int, meta: dict, excel_path: str, pdf_path: str):
    ensure_history_csv()
    try:
        dfh = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp_utc"])
    except Exception:
        dfh = pd.DataFrame()
    dfh = ensure_history_columns(dfh)

    minor = sum(1 for f in findings if (f.get("severity") or "").lower()=="minor")
    major = sum(1 for f in findings if (f.get("severity") or "").lower()=="major")
    new_row = {
        "timestamp_utc": utc_iso(),
        "file": file_name,
        "client": meta.get("client",""),
        "project": meta.get("project",""),
        "site_type": meta.get("site_type",""),
        "vendor": meta.get("vendor",""),
        "cabinet_loc": meta.get("cabinet_loc",""),
        "radio_loc": meta.get("radio_loc",""),
        "sectors": meta.get("sectors",""),
        "mimo_config": meta.get("mimo_config",""),
        "site_address": meta.get("site_address",""),
        "supplier": meta.get("supplier",""),
        "drawing_type": meta.get("drawing_type",""),
        "used_ocr": "False",
        "pages": pages,
        "minor_findings": minor,
        "major_findings": major,
        "total_findings": minor+major,
        "outcome": outcome,
        "rft_percent": 100.0 if (minor+major)==0 else 0.0,
        "exclude": False,
        "excel_path": excel_path,
        "annotated_pdf_path": pdf_path
    }
    dfh = pd.concat([dfh, pd.DataFrame([new_row])], ignore_index=True)
    dfh.to_csv(HISTORY_PATH, index=False)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Design Quality Auditor", layout="wide")

# Sticky logo (top-right)
def draw_logo():
    logo_file = st.session_state.get("logo_path","")
    st.markdown("""
        <style>
        .top-right-logo{
            position:fixed; top:16px; right:20px; z-index:1000;
            width: 140px; height:auto; opacity:0.95;
        }
        </style>
    """, unsafe_allow_html=True)
    if logo_file and Path(logo_file).exists():
        import base64
        b64 = base64.b64encode(open(logo_file, "rb").read()).decode()
        st.markdown(f'<img src="data:image/png;base64,{b64}" class="top-right-logo">', unsafe_allow_html=True)

if "logo_path" not in st.session_state:
    st.session_state["logo_path"] = ""

# Daily backup (lazy cron) & visibility setup
if "last_daily_backup_date" not in st.session_state:
    st.session_state["last_daily_backup_date"] = ""
today_uk = now_uk().strftime("%Y-%m-%d")
if st.session_state["last_daily_backup_date"] != today_uk:
    try:
        export_daily_backup()
        clean_old_backups(BACKUP_RETENTION_DAYS)
    except Exception as e:
        st.warning(f"Daily backup failed: {e}")
    st.session_state["last_daily_backup_date"] = today_uk

if "ui_visibility_hours" not in st.session_state:
    st.session_state["ui_visibility_hours"] = DEFAULT_UI_VISIBILITY_HOURS

# Retention cleaning (tighten to 1 day if visibility = 24h)
visibility_hours = st.session_state["ui_visibility_hours"]
outputs_retention_days = 1 if visibility_hours == 24 else DEFAULT_RETENTION_DAYS
clean_outputs(outputs_retention_days)

# Sidebar – branding & quick settings
with st.sidebar:
    st.header("Branding / Quick Settings")
    st.text_input("Logo file path (in repo root)", key="logo_path", placeholder="e.g. 88F3...png")
    st.caption("Logo stays fixed in the top-right if the file exists.")
    draw_logo()

    st.divider()
    st.caption(f"Outputs usage: {size_human(outputs_size_bytes())}")
    st.write(f"Visible window: **{visibility_hours}h**")
    st.caption("Daily backup runs at first open after midnight (UK).")

# Tabs
tab_audit, tab_history, tab_settings = st.tabs(["🔎 Audit", "📈 History & Analytics", "⚙️ Settings"])

# -----------------------------
# AUDIT TAB
# -----------------------------
with tab_audit:
    st.subheader("Run an Audit")

    # Metadata form
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        client = st.selectbox("Client*", ["","BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"])
    with c2:
        project = st.selectbox("Project*", ["","RAN","Power Resilience","East Unwind","Beacon 4"])
    with c3:
        site_type = st.selectbox("Site Type*", ["","Greenfield","Rooftop","Streetworks"])
    with c4:
        supplier = st.selectbox("Supplier*", ["","Balfour Beatty","Circet","Morrison Data Services","N/A"])

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        vendor = st.selectbox("Proposed Vendor*", ["","Ericsson","Nokia"])
    with c6:
        cabinet_loc = st.selectbox("Proposed Cabinet Location*", ["","Indoor","Outdoor"])
    with c7:
        radio_loc = st.selectbox("Proposed Radio Location*", ["","High Level","Low Level","Indoor","Door"])
    with c8:
        drawing_type = st.selectbox("Drawing Type*", ["","General Arrangement","Detailed Design"])

    c9, c10, c11 = st.columns([1,2,3])
    with c9:
        sectors = st.selectbox("Quantity of Sectors*", ["","1","2","3","4","5","6"])
    with c10:
        # Hide MIMO field when Power Resilience
        show_mimo = (project != "Power Resilience")
        mimo_config = st.selectbox("Proposed MIMO Config (hidden for Power Resilience)" if show_mimo else "Proposed MIMO Config (not required for Power Resilience)",
                                   [""] + [
                                       "18\\21\\26 @4x4; 70\\80 @2x2",
                                       "18\\21 @2x2",
                                       "18\\21\\26 @4x4; 3500 @8x8",
                                       "18\\21\\26 @4x4",
                                   ],
                                   disabled=not show_mimo)
    with c11:
        site_address = st.text_input("Site Address*", placeholder="MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ")

    meta = {
        "client": client, "project": project, "site_type": site_type,
        "vendor": vendor, "cabinet_loc": cabinet_loc, "radio_loc": radio_loc,
        "sectors": sectors, "mimo_config": mimo_config if show_mimo else "",
        "site_address": site_address, "supplier": supplier, "drawing_type": drawing_type
    }

    st.divider()

    # File upload (single audit at a time)
    pdf_file = st.file_uploader("Upload a PDF to audit", type=["pdf"])

    cols = st.columns([1,1,6])
    with cols[0]:
        do_audit = st.button("🚀 Audit", type="primary", use_container_width=True)
    with cols[1]:
        if st.button("🧹 Clear Metadata", use_container_width=True):
            for k in ["client","project","site_type","vendor","cabinet_loc","radio_loc","sectors","mimo_config","site_address","supplier","drawing_type"]:
                st.session_state.pop(k, None)
            st.experimental_rerun()

    # Show current rules summary
    with st.expander("Current Rules (read-only summary)"):
        rules = load_rules()
        st.json(rules)

    # Run audit
    if do_audit:
        # Validate mandatory metadata
        required = ["client","project","site_type","vendor","cabinet_loc","radio_loc","sectors","supplier","drawing_type","site_address"]
        missing = [k for k in required if not str(meta.get(k,"")).strip()]
        if not pdf_file:
            st.error("Please upload a PDF.")
        elif missing:
            st.error("Please complete all required metadata: " + ", ".join(missing))
        else:
            pdf_bytes = pdf_file.read()
            findings = audit_pdf(pdf_bytes, meta, rules)
            pages = extract_pdf_text_pages(pdf_bytes)
            excel_bytes, annotated_pdf, outcome, excel_path, pdf_path = build_excel_and_pdf(pdf_file.name, pdf_bytes, findings, meta)
            push_history(pdf_file.name, findings, outcome, pages=len(pages), meta=meta, excel_path=excel_path, pdf_path=pdf_path)

            st.success(f"Audit complete → **{outcome}**")
            cdl1, cdl2 = st.columns(2)
            with cdl1:
                st.download_button("⬇️ Excel Report", data=excel_bytes,
                                   file_name=os.path.basename(excel_path),
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with cdl2:
                st.download_button("⬇️ Annotated PDF", data=annotated_pdf,
                                   file_name=os.path.basename(pdf_path),
                                   mime="application/pdf")

            if findings:
                st.markdown("### Findings")
                st.dataframe(pd.DataFrame([{
                    "Category": f.get("category",""),
                    "Rule": f.get("rule",""),
                    "Severity": f.get("severity",""),
                    "Message": f.get("message",""),
                    "Page": f.get("page",0)
                } for f in findings]), use_container_width=True)
            else:
                st.info("No findings 🎉")

# -----------------------------
# HISTORY & ANALYTICS TAB
# -----------------------------
with tab_history:
    st.subheader("History & Analytics")

    try:
        dfh = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp_utc"])
    except Exception:
        dfh = pd.DataFrame()

    dfh = ensure_history_columns(dfh)
    df_vis = filter_history_for_ui(dfh.copy(), st.session_state.get("ui_visibility_hours", DEFAULT_UI_VISIBILITY_HOURS))

    st.markdown(f"Showing last **{st.session_state['ui_visibility_hours']} hours**")
    if df_vis.empty:
        st.info("No runs in the current visibility window.")
    else:
        # Recent runs
        st.markdown("### Recent Runs")
        show_cols = ["timestamp_utc","file","client","project","supplier","drawing_type","outcome","total_findings","exclude"]
        st.dataframe(df_vis[show_cols].sort_values("timestamp_utc", ascending=False), use_container_width=True)

        # Saved outputs quick downloads
        st.markdown("### Saved Outputs")
        for _, row in df_vis.sort_values("timestamp_utc", ascending=False).head(50).iterrows():
            c = st.columns([3,1,1,2,2,1])
            with c[0]:
                st.caption(str(row["timestamp_utc"]))
                st.write(f'**{row["file"]}** — {row["outcome"]}')
            with c[1]:
                xp = row.get("excel_path","")
                if xp and Path(xp).exists():
                    st.download_button("Excel", data=open(xp,"rb").read(),
                                       file_name=os.path.basename(xp),
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       key=f"x_{row.name}")
                else:
                    st.button("Excel", disabled=True, key=f"xd_{row.name}")
            with c[2]:
                pp = row.get("annotated_pdf_path","")
                if pp and Path(pp).exists():
                    st.download_button("PDF", data=open(pp,"rb").read(),
                                       file_name=os.path.basename(pp),
                                       mime="application/pdf",
                                       key=f"p_{row.name}")
                else:
                    st.button("PDF", disabled=True, key=f"pd_{row.name}")
            with c[3]:
                st.caption("Supplier")
                st.write(str(row.get("supplier","")))
            with c[4]:
                st.caption("Drawing Type")
                st.write(str(row.get("drawing_type","")))
            with c[5]:
                # Exclude toggle
                excl_key = f"excl_{row.name}"
                cur = bool(row.get("exclude", False))
                if st.checkbox("Exclude", value=cur, key=excl_key):
                    new_val = True
                else:
                    new_val = False
                # Write back on change
                if new_val != cur:
                    dfh.loc[row.name, "exclude"] = new_val
                    dfh.to_csv(HISTORY_PATH, index=False)
                    st.experimental_rerun()

        # Simple analytics (excluding excluded)
        st.markdown("### Analytics (excluding excluded)")
        use = dfh[dfh.get("exclude", False) != True].copy()
        if not use.empty:
            total = len(use)
            pass_count = (use["outcome"]=="PASS").sum()
            rft = 100.0 * pass_count / total
            cA, cB, cC = st.columns(3)
            cA.metric("Runs", f"{total}")
            cB.metric("PASS", f"{pass_count}")
            cC.metric("Right First Time", f"{rft:.1f}%")

            by_supplier = use.groupby("supplier")["outcome"].apply(lambda s: (s=="REJECTED").sum()).reset_index(name="Rejected")
            st.bar_chart(by_supplier.set_index("supplier"))
        else:
            st.info("Nothing to analyse (all runs excluded).")

# -----------------------------
# SETTINGS TAB (password gated)
# -----------------------------
with tab_settings:
    st.subheader("Settings")
    pw = st.text_input("Admin password", type="password")
    if pw != APP_PASSWORD:
        st.warning("Enter the admin password to edit settings.")
        st.stop()

    # Visibility horizon
    st.markdown("#### Visibility")
    st.session_state["ui_visibility_hours"] = st.select_slider(
        "How long should runs stay VISIBLE in the app?",
        options=[6,12,24,48,72,168],  # 168 = 7 days
        value=st.session_state.get("ui_visibility_hours", DEFAULT_UI_VISIBILITY_HOURS),
        help="This only affects the UI listing. Backups remain safely stored."
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Run cleanup now"):
            res = clean_outputs(outputs_retention_days)
            st.success(f"Deleted {len(res['deleted_by_age'])} old files + {len(res['deleted_by_size'])} for size cap.")
    with c2:
        st.caption(f"/outputs usage: {size_human(outputs_size_bytes())}  •  Cap: {size_human(MAX_OUTPUT_DIR_BYTES)}")

    st.divider()

    # Backups
    st.markdown("#### Backups")
    d1, d2, d3 = st.columns([1,1,2])
    with d1:
        if st.button("Run daily export now"):
            try:
                zp = export_daily_backup()
                st.success(f"Backup created: {Path(zp).name}")
            except Exception as e:
                st.error(f"Backup failed: {e}")
    with d2:
        if st.button("Purge backups older than 60 days"):
            res = clean_old_backups(BACKUP_RETENTION_DAYS)
            st.success(f"Deleted {len(res['deleted'])} old backups.")
    with d3:
        st.caption("Backups include: history CSV snapshot + all current outputs + meta.json")

    st.markdown("**Available Backups**")
    bks = sorted(BACKUP_DIR.glob("backup_*.zip"), key=lambda p: p.name, reverse=True)
    if not bks:
        st.info("No backups yet. One will be created automatically at midnight (UK) or via the button.")
    else:
        for p in bks[:20]:
            with open(p, "rb") as fh:
                st.download_button(
                    label=f"Download {p.name}",
                    data=fh.read(),
                    file_name=p.name,
                    mime="application/zip",
                    key=f"bk_{p.name}"
                )

    st.divider()

    # Rules editor (simple)
    st.markdown("#### Rules (YAML)")
    current = RULES_PATH.read_text(encoding="utf-8") if RULES_PATH.exists() else yaml.safe_dump(load_rules(), sort_keys=False)
    updated = st.text_area("rules_example.yaml", value=current, height=300)
    c3, c4 = st.columns([1,3])
    with c3:
        if st.button("Save Rules"):
            ok, msg = save_rules(updated)
            if ok: st.success(msg)
            else: st.error(msg)
    with c4:
        st.caption("Tip: Example structure:")
        st.code("""checks:
  spelling:
    enabled: true
    severity: minor
  address_title_match:
    enabled: true
    severity: major
  incompatible_terms:
    enabled: true
    severity: major
    pairs:
      - ["Brush", "Generator Power"]
allowlist:
  - MBNL
  - Ericsson
""", language="yaml")
