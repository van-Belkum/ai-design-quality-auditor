# app.py — Professional layout + Analytics + Settings + top-left logo
from __future__ import annotations
import os, io, re, json, base64
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import yaml
from rapidfuzz import process, fuzz

# Optional PDF annotation
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# ────────────────────────────── APP CONFIG ──────────────────────────────
APP_TITLE = "AI Design Quality Auditor"
ENTRY_PASSWORD = "Seker123"          # ← entry gate
RULES_PASSWORD = "vanB3lkum21"       # ← required only to overwrite rules in-app

VAULT_DIR   = "vault"     # reports & annotated PDFs
EXPORT_DIR  = "exports"   # daily CSV snapshots
HISTORY_CSV = os.path.join(VAULT_DIR, "history.csv")
RULES_STORE = os.path.join(VAULT_DIR, "rules_latest.yaml")

DEFAULT_UI_VISIBILITY_HOURS = 24     # analytics window (hours)
DATA_RETENTION_DAYS = 14             # vault/history cleanup
AUTO_EXPORT_DAILY = True             # export once per day (first run after midnight UTC)

# Put your logo file in the repo root; adjust path if needed
LOGO_PATH = "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png"

# ────────────────────────────── FS SETUP ───────────────────────────────
os.makedirs(VAULT_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# ────────────────────────────── UTILITIES ──────────────────────────────
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def ts_iso(dt: Optional[datetime] = None) -> str:
    return (dt or utc_now()).isoformat()

def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s.strip())
    return s[:180] or "file"

def pin_logo_top_left(path: str):
    """Pins a logo image to the top-left of the app."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
              .pinned-logo {{
                position: fixed;
                top: 12px; left: 16px;
                width: 170px; height: auto;
                z-index: 9999; opacity: .98;
              }}
              @media (max-width: 768px) {{ .pinned-logo {{ width: 130px; }} }}
              /* give main app some left padding so logo doesn't overlap */
              .block-container {{ padding-top: 80px; }}
            </style>
            <img class="pinned-logo" src="data:image/png;base64,{b64}">
            """,
            unsafe_allow_html=True
        )
    except Exception:
        pass

def entry_gate():
    st.title(APP_TITLE)
    pin_logo_top_left(LOGO_PATH)
    if st.session_state.get("auth_ok"):
        return
    pw = st.text_input("Enter password to continue", type="password")
    if st.button("Unlock"):
        if pw == ENTRY_PASSWORD:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

def load_rules_text(txt: str) -> Dict[str, Any]:
    txt = txt.replace("\r\n", "\n")
    try:
        obj = yaml.safe_load(txt) or {}
    except yaml.YAMLError:
        # escape stray backslashes inside quoted scalars
        txt2 = re.sub(r'\\(?![nrt"\\])', r'\\\\', txt)
        obj = yaml.safe_load(txt2) or {}
    if not isinstance(obj, dict):
        obj = {}
    obj.setdefault("allowlist", [])
    obj.setdefault("rules", [])
    obj.setdefault("relationships", [])
    obj.setdefault("checklist", [])
    return obj

def load_rules_file(file_bytes: bytes) -> Dict[str, Any]:
    try:
        return load_rules_text(file_bytes.decode(errors="ignore"))
    except Exception:
        return {"allowlist": [], "rules": [], "relationships": [], "checklist": []}

def extract_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Return list of pages with flattened text + word bboxes."""
    if not fitz:
        return [{"page_num": 1, "text": "", "words": []}]
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        words = page.get_text("words")
        words_n = []
        txt_parts = []
        for w in words:
            x0, y0, x1, y1, word, *rest = w
            words_n.append((word, float(x0), float(y0), float(x1), float(y1)))
            txt_parts.append(word)
        pages.append({"page_num": i + 1, "text": " ".join(txt_parts), "words": words_n})
    doc.close()
    return pages

def build_vocab(pages: List[Dict[str, Any]], allow: set[str]) -> List[str]:
    vocab = set(allow)
    for p in pages:
        for w in re.findall(r"[A-Za-z][A-Za-z\-']{1,}", p.get("text", "")):
            vocab.add(w.lower())
    return list(vocab)

def spelling_findings(pages: List[Dict[str, Any]], allowlist: List[str]) -> List[Dict[str, Any]]:
    allow = set([w.strip().lower() for w in allowlist if w])
    vocab = build_vocab(pages, allow)
    out = []
    for p in pages:
        for (w, x0, y0, x1, y1) in p.get("words", []):
            wl = w.lower()
            if wl in allow: 
                continue
            if not re.match(r"^[A-Za-z][A-Za-z\-']*$", w):
                continue
            if len(w) <= 2:
                continue
            suggestion = None
            try:
                best = process.extractOne(wl, vocab, scorer=fuzz.WRatio, score_cutoff=87)
                if best:
                    suggestion = best[0]
            except Exception:
                pass
            out.append({
                "category": "Spelling",
                "severity": "Minor",
                "message": f"Possible misspelling: '{w}'",
                "suggestion": suggestion,
                "page": p["page_num"],
                "bbox": [x0, y0, x1, y1],
                "rule_id": "spelling_generic"
            })
    return out

def normalize_site_address(addr: str) -> str:
    if not addr:
        return ""
    parts = [p.strip() for p in addr.split(",")]
    parts = [p for p in parts if p != "0"]  # ignore literal '0' placeholders
    return ", ".join([p for p in parts if p])

def title_matches_address(file_title: str, site_addr: str) -> bool:
    A = normalize_site_address(site_addr).upper()
    if not A:
        return True
    tokens = [t for t in re.split(r"[\s,]+", A) if t]
    T = re.sub(r"\s+", " ", file_title.upper())
    return all(tok in T for tok in tokens)

def project_hides_mimo(project: str) -> bool:
    return (project or "").strip().lower() in {"power resilience", "power res", "power resillience"}

def run_rules(pages: List[Dict[str, Any]], rules_blob: Dict[str, Any],
              metadata: Dict[str, Any], pdf_title: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    # Address vs title
    addr = metadata.get("site_address", "")
    if addr and pdf_title and not title_matches_address(pdf_title, addr):
        findings.append({
            "category": "Metadata",
            "severity": "Major",
            "message": "Site Address does not match the PDF title.",
            "suggestion": f"Ensure title contains: {normalize_site_address(addr)}",
            "page": 1,
            "bbox": None,
            "rule_id": "meta_address_vs_title"
        })

    # Relationship rules (if_contains + forbid)
    text_all = " ".join([p.get("text", "") for p in pages]).upper()
    for rel in rules_blob.get("relationships", []):
        ic = (rel.get("if_contains") or "").strip()
        fb = (rel.get("forbid") or "").strip()
        if not ic or not fb:
            continue
        if ic.upper() in text_all and fb.upper() in text_all:
            findings.append({
                "category": "Relationship",
                "severity": rel.get("severity", "Major"),
                "message": f"'{fb}' is not allowed when '{ic}' is present.",
                "suggestion": "Remove forbidden pairing or adjust design.",
                "page": 1,
                "bbox": None,
                "rule_id": f"rel_{ic}_{fb}"
            })

    # Pattern rules
    for rule in rules_blob.get("rules", []):
        pat = rule.get("pattern")
        if not pat:
            continue
        rx = re.compile(pat, re.IGNORECASE)
        for p in pages:
            m = rx.search(p.get("text", ""))
            if not m:
                continue
            hit = m.group(0)
            bbox = None
            for (w, x0, y0, x1, y1) in p.get("words", []):
                if w.lower() == hit.lower():
                    bbox = [x0, y0, x1, y1]
                    break
            findings.append({
                "category": rule.get("category", "Rule"),
                "severity": rule.get("severity", "Minor"),
                "message": rule.get("message", "Rule match"),
                "suggestion": rule.get("suggestion"),
                "page": p["page_num"],
                "bbox": bbox,
                "rule_id": rule.get("id", f"rule_{pat}")
            })

    # Checklist must_appear items
    for item in rules_blob.get("checklist", []):
        must = (item.get("must_contain") or "").strip()
        if not must:
            continue
        if must.upper() not in text_all:
            findings.append({
                "category": "Checklist",
                "severity": item.get("severity", "Major"),
                "message": item.get("message", "Checklist requirement not met"),
                "suggestion": f"Add '{must}'",
                "page": 1,
                "bbox": None,
                "rule_id": item.get("id", "check_missing")
            })

    return findings

def annotate_pdf(pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    if not fitz:
        return pdf_bytes
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for f in findings:
        pidx = max(0, int(f.get("page", 1)) - 1)
        if pidx >= len(doc):
            continue
        page = doc[pidx]
        note = f"{f.get('category','')}: {f.get('message','')}"
        bbox = f.get("bbox")
        if bbox and len(bbox) == 4:
            rect = fitz.Rect(*bbox)
            try:
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=(1, 0, 0))
                annot.set_border(width=0.8, dashes=[2, 2])
                annot.update()
                page.add_text_annot(rect.br, note)
            except Exception:
                page.add_text_annot(rect.tl, note)
        else:
            page.add_text_annot(fitz.Point(36, 36), note)
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()

def build_report_df(findings: List[Dict[str, Any]], meta: Dict[str, Any], file_name: str) -> pd.DataFrame:
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
            **{f"meta_{k}": v for k, v in meta.items()}
        })
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "file_name","category","severity","message","suggestion","page","bbox","rule_id",
            *[f"meta_{k}" for k in meta.keys()]
        ])
    return df

def save_report_and_pdf(df: pd.DataFrame, annotated_pdf: bytes, file_name: str, passed: bool) -> Tuple[str, str]:
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    base = os.path.splitext(safe_filename(file_name))[0]
    verdict = "PASS" if passed else "REJECTED"
    xlsx = os.path.join(VAULT_DIR, f"{base}__{verdict}__{stamp}.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Findings")
    pdfp = os.path.join(VAULT_DIR, f"{base}__{verdict}__{stamp}.annotated.pdf")
    with open(pdfp, "wb") as f:
        f.write(annotated_pdf)
    return xlsx, pdfp

def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_CSV)
    except Exception:
        return pd.DataFrame()
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    return df

def push_history(rec: Dict[str, Any]):
    df = load_history()
    rec = rec.copy()
    rec["timestamp_utc"] = pd.to_datetime(rec.get("timestamp_utc") or ts_iso(), utc=True)
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    df.to_csv(HISTORY_CSV, index=False)

def filter_history_for_ui(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    if df.empty:
        return df
    if "timestamp_utc" not in df.columns:
        return df
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp_utc"])
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=int(hours))
    return df[df["timestamp_utc"] >= cutoff]

def enforce_retention():
    cutoff = utc_now() - timedelta(days=DATA_RETENTION_DAYS)
    df = load_history()
    if not df.empty:
        keep = df[pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True) >= cutoff]
        keep.to_csv(HISTORY_CSV, index=False)
    for root in (VAULT_DIR, EXPORT_DIR):
        for f in os.listdir(root):
            p = os.path.join(root, f)
            try:
                m = datetime.fromtimestamp(os.path.getmtime(p), tz=timezone.utc)
                if m < cutoff:
                    os.remove(p)
            except Exception:
                pass

def maybe_daily_export():
    if not AUTO_EXPORT_DAILY:
        return
    flag = os.path.join(EXPORT_DIR, "last_export_date.txt")
    today = utc_now().date().isoformat()
    last = None
    if os.path.exists(flag):
        with open(flag, "r") as f:
            last = f.read().strip()
    if last == today:
        return
    df = load_history()
    if not df.empty:
        out = os.path.join(EXPORT_DIR, f"audit_history_{today}.csv")
        df.to_csv(out, index=False)
    with open(flag, "w") as f:
        f.write(today)

# ───────────────────────────── STREAMLIT APP ─────────────────────────────
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
entry_gate()
pin_logo_top_left(LOGO_PATH)
enforce_retention()
maybe_daily_export()

# Sidebar: Upload & Settings
with st.sidebar:
    st.header("Upload")
    pdf_file = st.file_uploader("PDF to audit", type=["pdf"])

    st.markdown("---")
    st.header("Settings")
    ui_hours = st.number_input("Analytics window (hours)", 1, 336, value=st.session_state.get("ui_visibility_hours", DEFAULT_UI_VISIBILITY_HOURS))
    st.session_state["ui_visibility_hours"] = int(ui_hours)

    with st.expander("Rules management"):
        rules_src = st.radio("Load rules", ["Upload YAML", "Use last saved"], horizontal=True)
        rules_file = None
        if rules_src == "Upload YAML":
            rules_file = st.file_uploader("rules_example.yaml", type=["yaml", "yml"], key="rules_upload")
        rule_pw = st.text_input("Rules edit password (to save/overwrite)", type="password")
        save_rules_clicked = st.button("Save rules in app (overwrite)")
        if save_rules_clicked:
            if rule_pw != RULES_PASSWORD:
                st.error("Wrong rules password.")
            else:
                if rules_src == "Upload YAML" and rules_file is not None:
                    with open(RULES_STORE, "wb") as f:
                        f.write(rules_file.getvalue())
                    st.success("Rules saved.")
                else:
                    # Save current blob if any loaded
                    try:
                        with open(RULES_STORE, "w", encoding="utf-8") as f:
                            yaml.safe_dump(st.session_state.get("rules_blob", {}), f, sort_keys=False, allow_unicode=True)
                        st.success("Current rules saved.")
                    except Exception as e:
                        st.error(f"Save failed: {e}")

# Load rules
if os.path.exists(RULES_STORE) and (st.session_state.get("rules_blob") is None):
    try:
        with open(RULES_STORE, "rb") as f:
            st.session_state["rules_blob"] = load_rules_file(f.read())
    except Exception:
        st.session_state["rules_blob"] = {"allowlist": [], "rules": [], "relationships": [], "checklist": []}

rules_blob: Dict[str, Any] = {"allowlist": [], "rules": [], "relationships": [], "checklist": []}
if st.session_state.get("rules_blob"):
    rules_blob = st.session_state["rules_blob"]

if 'rules_upload' in st.session_state and st.session_state['rules_upload'] is not None:
    try:
        rules_blob = load_rules_file(st.session_state['rules_upload'].getvalue())
        st.session_state["rules_blob"] = rules_blob
    except Exception:
        pass

# Title
st.title(APP_TITLE)
st.caption("Professional, consistent design QA with audit trail, annotations, analytics, and easy rule updates.")

# ────────────── METADATA (Required) ──────────────
st.subheader("Audit Metadata (required)")
c1, c2, c3, c4 = st.columns(4)
with c1:
    supplier = st.selectbox("Supplier", ["— Select —","Cellnex","Cornerstone","Vodafone","MBNL","BTEE","H3G","Other"])
with c2:
    drawing_type = st.selectbox("Drawing Type", ["— Select —","General Arrangement","Detailed Design"])
with c3:
    client = st.selectbox("Client", ["— Select —","BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"])
with c4:
    project = st.selectbox("Project", ["— Select —","RAN","Power Resilience","East Unwind","Beacon 4"])

c5, c6, c7, c8 = st.columns(4)
with c5:
    site_type = st.selectbox("Site Type", ["— Select —","Greenfield","Rooftop","Streetworks"])
with c6:
    vendor = st.selectbox("Proposed Vendor", ["— Select —","Ericsson","Nokia"])
with c7:
    cabinet_loc = st.selectbox("Proposed Cabinet Location", ["— Select —","Indoor","Outdoor"])
with c8:
    radio_loc = st.selectbox("Proposed Radio Location", ["— Select —","High Level","Low Level","Indoor","Door"])

c9, c10 = st.columns(2)
with c9:
    sectors = st.selectbox("Quantity of Sectors", ["— Select —","1","2","3","4","5","6"])
with c10:
    site_address = st.text_input("Site Address", placeholder="MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ")

# MIMO (hidden for Power Resilience)
show_mimo = not project_hides_mimo(project)
if show_mimo:
    st.markdown("**Proposed MIMO Config (optional unless required by client)**")
    same_mimo = st.checkbox("Use same config for all", value=True)
    # keep a trimmed options list here; you can expand in rules if needed
    mimo_options = [
        "18\\21\\26 @4x4; 70\\80 @2x2",
        "18\\21 @2x2",
        "18\\21\\26 @4x4; 3500 @8x8",
        "18\\21\\26 @4x4",
    ]
    m1, m2, m3 = st.columns(3)
    mimo_s1 = m1.selectbox("MIMO S1", mimo_options, index=0)
    mimo_s2 = m2.selectbox("MIMO S2", mimo_options, index=0, disabled=same_mimo)
    mimo_s3 = m3.selectbox("MIMO S3", mimo_options, index=0, disabled=same_mimo)
else:
    mimo_s1 = mimo_s2 = mimo_s3 = None

st.markdown("---")
b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
with b1:
    run_audit = st.button("▶️ Run Audit", use_container_width=True)
with b2:
    clear_meta = st.button("🧹 Clear Metadata", use_container_width=True)
with b3:
    export_now = st.button("📤 Export History", use_container_width=True)
with b4:
    exclude_flag = st.checkbox("Exclude this run from analytics (training mode)")

if clear_meta:
    for k in [
        "supplier","drawing_type","client","project","site_type",
        "vendor","cabinet_loc","radio_loc","sectors","site_address"
    ]:
        st.session_state.pop(k, None)
    st.rerun()

if export_now:
    dfh = load_history()
    if dfh.empty:
        st.info("No history to export yet.")
    else:
        out = os.path.join(EXPORT_DIR, f"audit_history_{utc_now().date().isoformat()}_manual.csv")
        dfh.to_csv(out, index=False)
        st.success(f"Exported: {out}")

def meta_ok() -> Tuple[bool, str]:
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

# ───────────────────────────── RUN AUDIT ─────────────────────────────
if run_audit:
    ok, msg = meta_ok()
    if not ok:
        st.error(msg)
        st.stop()
    if not pdf_file:
        st.error("Please upload a PDF first.")
        st.stop()

    raw = pdf_file.read()
    file_name = pdf_file.name
    title_guess = os.path.splitext(file_name)[0]

    pages = extract_pages(raw)

    findings: List[Dict[str, Any]] = []
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
            "mimo_s3": mimo_s3
        },
        title_guess
    )

    major = sum(1 for f in findings if str(f.get("severity", "")).lower() == "major")
    passed = (major == 0)

    annotated = annotate_pdf(raw, findings)
    meta = {
        "supplier": supplier, "drawing_type": drawing_type, "client": client,
        "project": project, "site_type": site_type, "vendor": vendor,
        "cabinet_location": cabinet_loc, "radio_location": radio_loc,
        "sectors": sectors, "site_address": site_address,
        "mimo_s1": mimo_s1, "mimo_s2": mimo_s2, "mimo_s3": mimo_s3,
        "verdict": "PASS" if passed else "REJECTED",
    }
    df = build_report_df(findings, meta, file_name)
    xlsx_path, pdf_path = save_report_and_pdf(df, annotated, file_name, passed)

    push_history({
        "timestamp_utc": ts_iso(),
        "file_name": file_name,
        **meta,
        "findings_count": len(findings),
        "major_count": major,
        "exclude": bool(exclude_flag),
        "report_path": xlsx_path,
        "annotated_pdf_path": pdf_path
    })

    st.success("Audit complete.")
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Findings", len(findings))
    cB.metric("Major", major)
    cC.metric("Verdict", "PASS" if passed else "REJECTED")
    cD.metric("Supplier", supplier)

    st.subheader("Downloads")
    with open(xlsx_path, "rb") as f:
        st.download_button(
            "Download Excel Report", f,
            file_name=os.path.basename(xlsx_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with open(pdf_path, "rb") as f:
        st.download_button(
            "Download Annotated PDF", f,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf"
        )

# ───────────────────────────── ANALYTICS ─────────────────────────────
st.markdown("---")
st.subheader("Analytics & History")

dfh = load_history()
if dfh.empty:
    st.info("No runs yet.")
else:
    # visibility window
    df_vis = filter_history_for_ui(dfh.copy(), st.session_state.get("ui_visibility_hours", DEFAULT_UI_VISIBILITY_HOURS))

    # KPIs (ignore excluded)
    exclude_series = df_vis["exclude"] if "exclude" in df_vis.columns else False
    df_analytics = df_vis[~exclude_series.astype(bool)] if isinstance(exclude_series, pd.Series) else df_vis

    if not df_analytics.empty and "verdict" in df_analytics.columns:
        rft = (df_analytics["verdict"] == "PASS").mean() * 100
    else:
        rft = 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Right First Time (%)", f"{rft:.1f}")
    k2.metric("Major Findings", int(df_analytics["major_count"].sum() if "major_count" in df_analytics.columns else 0))
    k3.metric("Total Findings", int(df_analytics["findings_count"].sum() if "findings_count" in df_analytics.columns else 0))
    k4.metric("Runs (visible window)", int(len(df_vis)))

    # Trends (by day & supplier)
    if not df_analytics.empty and "timestamp_utc" in df_analytics.columns:
        tmp = df_analytics.copy()
        tmp["day"] = pd.to_datetime(tmp["timestamp_utc"], errors="coerce", utc=True).dt.date
        if "supplier" in tmp.columns and "major_count" in tmp.columns:
            daily = tmp.groupby(["day", "supplier"])["major_count"].sum().reset_index()
            if not daily.empty:
                st.line_chart(daily.pivot(index="day", columns="supplier", values="major_count").fillna(0))

    # History table (show most useful cols; fill if missing)
    show_cols = [
        "timestamp_utc","file_name","supplier","drawing_type","client","project",
        "verdict","major_count","findings_count","exclude","report_path","annotated_pdf_path"
    ]
    for c in show_cols:
        if c not in df_vis.columns:
            df_vis[c] = ""
    st.dataframe(df_vis[show_cols].sort_values("timestamp_utc", ascending=False), use_container_width=True)
