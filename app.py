# app.py
# AI Design Quality Auditor – full app
# Streamlit 1.49+, Python 3.11+
# ---------------------------------------------------------------

import base64
import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import yaml
from PIL import Image
import streamlit as st

# ----------- Constants & Options ----------------------------------------

APP_TITLE = "AI Design Quality Auditor"

CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
CAB_LOC = ["Indoor", "Outdoor"]
RADIO_LOC = ["High Level", "Low Level", "Indoor and Door"]
SECTOR_QTY = [1, 2, 3, 4, 5, 6]

# New
SUPPLIER_OPTIONS = [
    "CEG", "CTIL", "Emfyser", "Innov8", "Invict",
    "KTL Team (Internal)", "Trylon", "Other"
]
DRAWING_TYPES = ["General Arrangement", "Detailed Design"]

HISTORY_PATH = Path("history/audit_history.csv")

# ----------- Utility: Logo injection ------------------------------------

def _resolve_logo_bytes() -> Optional[bytes]:
    """
    Resolve logo file bytes either from st.secrets["logo_path"] or
    by scanning repo root for a likely image file.
    """
    # 1) secrets override
    logo_path = st.secrets.get("logo_path", None)
    cand_paths: List[Path] = []
    if logo_path:
        cand_paths.append(Path(logo_path))

    # 2) common names in repo root
    for name in [
        "logo.png", "logo.svg", "logo.jpg", "logo.jpeg",
        "seker.png", "Seker.png",
        # user-provided id file:
        "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png",
    ]:
        cand_paths.append(Path(name))

    for p in cand_paths:
        if p.exists() and p.is_file():
            try:
                return p.read_bytes()
            except Exception:
                pass
    return None


def _inject_top_right_logo() -> None:
    """Stick a logo on the top-right using a small CSS helper."""
    data = _resolve_logo_bytes()
    st.markdown(
        """
        <style>
        .top-right-logo {
            position: fixed;
            top: 10px;
            right: 16px;
            z-index: 99999;
            height: 42px;
            opacity: 0.95;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if data:
        b64 = base64.b64encode(data).decode()
        st.markdown(
            f'<img class="top-right-logo" src="data:image/png;base64,{b64}"/>',
            unsafe_allow_html=True,
        )
    else:
        st.info("⚠️ Logo file not found in repo root (png/svg/jpg) or `st.secrets['logo_path']`.")

# ----------- Rules loading ----------------------------------------------

def load_rules(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
            return data
        except yaml.YAMLError as e:
            st.error(f"YAML parse error in {path.name}: {e}")
            return {}

# ----------- PDF text extraction ----------------------------------------

def extract_text_pymupdf(pdf_bytes: bytes) -> Tuple[List[Dict[str, Any]], int]:
    """
    Return list of pages -> {page_no, text, words: [ {text,bbox} ]}.
    """
    pages: List[Dict[str, Any]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            # raw text
            text = page.get_text("text")
            words = []
            # words with bbox: (x0, y0, x1, y1, word, block_no, line_no, word_no)
            for w in page.get_text("words"):
                x0, y0, x1, y1, word, *_ = w
                words.append({"text": word, "bbox": (x0, y0, x1, y1)})
            pages.append({"page_no": i + 1, "text": text, "words": words})
        return pages, len(doc)


# ----------- Spelling check (allow list) ---------------------------------

def spelling_checks(pages: List[Dict[str, Any]], allow: List[str]) -> List[Dict[str, Any]]:
    """
    Very light spell check: flags words that look like mixed alpha and not in allow list.
    Uses page words with bbox for annotation.
    """
    allow_set = set([w.lower() for w in allow])
    findings: List[Dict[str, Any]] = []
    for p in pages:
        for w in p["words"]:
            token = w["text"]
            # short tokens or numeric/units ignored
            if len(token) < 3:
                continue
            if not any(c.isalpha() for c in token):
                continue
            token_l = token.lower()
            if token_l in allow_set:
                continue
            # heuristic: words with weird punctuation or obvious typos (no dict)
            bad = False
            if any(ch in token for ch in ["/", "\\", "…", "—", "–", "~"]):
                bad = True
            if not bad and token_l not in allow_set:
                # mark as suspicious spelling
                bad = True
            if bad:
                findings.append({
                    "page": p["page_no"],
                    "kind": "Spelling",
                    "severity": "minor",
                    "message": f"Suspicious word: “{token}”",
                    "context": token,
                    "bbox": w.get("bbox"),
                })
    return findings

# ----------- Rule checks -------------------------------------------------

def check_metadata_rules(meta: Dict[str, Any], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply simple allow/required checks based on `metadata_rules` in YAML.
    """
    out: List[Dict[str, Any]] = []
    mr = rules.get("metadata_rules", {})
    for key, cfg in mr.items():
        if not cfg or not cfg.get("enabled", True):
            continue
        label = key.replace("_", " ").title()
        if cfg.get("required", False):
            if not meta.get(key) and meta.get(key) != 0:
                out.append({
                    "page": 1,
                    "kind": "Metadata",
                    "severity": "major",
                    "message": f"{label} is required.",
                    "context": key,
                    "bbox": None
                })
        allowed = cfg.get("allowed")
        if allowed:
            val = meta.get(key)
            if val and val not in allowed:
                out.append({
                    "page": 1, "kind": "Metadata", "severity": "major",
                    "message": f"{label} “{val}” not in allowed list.",
                    "context": key, "bbox": None
                })

        # Special: mimo required unless project == "Power Resilience"
        if key == "mimo_config" and cfg.get("required_unless_project_is"):
            unless = cfg["required_unless_project_is"]
            if str(meta.get("project")) != unless and not meta.get("mimo_config"):
                out.append({
                    "page": 1, "kind": "Metadata", "severity": "major",
                    "message": f"Proposed Mimo Config is required for project “{meta.get('project')}”.",
                    "context": key, "bbox": None
                })
    return out


# ----------- PDF annotation ---------------------------------------------

def annotate_pdf(original_pdf: bytes, findings: List[Dict[str, Any]]) -> bytes:
    """
    Draw highlight boxes + sticky notes for findings with bbox.
    Robust to missing/invalid bbox.
    """
    doc = fitz.open(stream=original_pdf, filetype="pdf")
    try:
        for f in findings:
            page_idx = max(0, int(f.get("page", 1)) - 1)
            if page_idx >= len(doc):
                continue
            page = doc[page_idx]
            bbox = f.get("bbox")
            note = f"{f.get('kind','')}: {f.get('message','')}"
            # Always add a small text annotation in the page margin
            page.add_text_annot(page.rect.top_right, note)

            # Optional highlight at bbox
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    rect = fitz.Rect(*bbox)
                    # a transparent highlight box
                    annot = page.add_rect_annot(rect)
                    annot.set_colors(stroke=(1, 0, 0))  # red border
                    annot.set_opacity(0.6)
                    annot.update()
                except Exception:
                    # ignore bbox drawing errors
                    pass
        out = io.BytesIO()
        doc.save(out)
        return out.getvalue()
    finally:
        doc.close()

# ----------- History helpers --------------------------------------------

def _ensure_history():
    if not HISTORY_PATH.parent.exists():
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        import csv
        with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp_utc","file","client","project","site_type","vendor",
                "cabinet_loc","radio_loc","sectors","mimo_config","site_address",
                "supplier","drawing_type","used_ocr","pages",
                "minor_findings","major_findings","total_findings",
                "outcome","rft_percent"
            ])


def _classify_sev(kind: str, current: Optional[str]) -> str:
    if current:
        return current
    return "minor" if str(kind).lower() == "spelling" else "major"


def summarise_findings(findings: List[Dict[str, Any]]) -> Tuple[int, int]:
    minor = 0
    major = 0
    for f in findings:
        sev = _classify_sev(f.get("kind"), f.get("severity"))
        if sev == "minor":
            minor += 1
        else:
            major += 1
    return minor, major


def append_history(row: Dict[str, Any]):
    _ensure_history()
    import csv
    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            row["timestamp_utc"], row["file"], row["client"], row["project"], row["site_type"], row["vendor"],
            row["cabinet_loc"], row["radio_loc"], row["sectors"], row["mimo_config"], row["site_address"],
            row["supplier"], row["drawing_type"], row["used_ocr"], row["pages"],
            row["minor_findings"], row["major_findings"], row["total_findings"],
            row["outcome"], row["rft_percent"]
        ])

# ----------- Excel export ------------------------------------------------

def build_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], original_filename: str) -> bytes:
    """
    Create an Excel workbook in-memory with Metadata + Findings sheets.
    """
    meta_rows = [{"Field": k.replace("_"," ").title(), "Value": v} for k, v in meta.items()]
    df_meta = pd.DataFrame(meta_rows)

    cols = ["page","kind","severity","message","context"]
    df_find = pd.DataFrame(findings)[cols] if findings else pd.DataFrame(columns=cols)

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_meta.to_excel(writer, sheet_name="Metadata", index=False)
        df_find.to_excel(writer, sheet_name="Findings", index=False)
    return out.getvalue()

# ----------- Page Layout -------------------------------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
_inject_top_right_logo()

st.title(APP_TITLE)

# Sidebar: YAML rules & spell allow list info
st.sidebar.header("Configuration")
rules_file = st.sidebar.text_input("Rules file", "rules_example.yaml")
with st.sidebar.expander("Spell Allow-list (from YAML)", expanded=False):
    st.write("Words listed under `spelling.allow` are not flagged as typos.")

# Load rules
rules = load_rules(Path(rules_file))
sp_allow = rules.get("spelling", {}).get("allow", [])

# ----------- Metadata Controls ------------------------------------------

# Upper meta row
c1, c2, c3, c4 = st.columns(4)
with c1:
    client = st.selectbox("Client *", CLIENTS, index=None, placeholder="Select client")
with c2:
    project = st.selectbox("Project *", PROJECTS, index=None, placeholder="Select project")
with c3:
    site_type = st.selectbox("Site Type *", SITE_TYPES, index=None, placeholder="Select site type")
with c4:
    vendor = st.selectbox("Proposed Vendor *", VENDORS, index=None, placeholder="Select vendor")

# Row 2
c5, c6, c7 = st.columns(3)
with c5:
    cabinet_loc = st.selectbox("Proposed Cabinet Location *", CAB_LOC, index=None, placeholder="Select cabinet location")
with c6:
    radio_loc = st.selectbox("Proposed Radio Location *", RADIO_LOC, index=None, placeholder="Select radio location")
with c7:
    sectors = st.selectbox("Quantity of Sectors *", SECTOR_QTY, index=None, placeholder="Select sectors")

# Row 3: Supplier / Drawing Type
c8, c9 = st.columns(2)
with c8:
    supplier = st.selectbox("Supplier *", SUPPLIER_OPTIONS, index=None, placeholder="Select supplier")
with c9:
    drawing_type = st.selectbox("Drawing Type *", DRAWING_TYPES, index=None, placeholder="Select drawing type")

# MIMO (hide for Power Resilience)
show_mimo = (project is not None and project != "Power Resilience")
mimo_config = None
if show_mimo:
    mimo_config = st.text_input("Proposed Mimo Config *", placeholder="e.g., 18\\21\\26 @4x4; 70\\80 @2x2")

# Address
site_address = st.text_input("Site Address *", placeholder="Street, City, Postcode")

# Buttons
b1, b2 = st.columns([1, 1])
with b1:
    audit_now = st.button("▶️ Audit", type="primary")
with b2:
    clear_meta = st.button("🗑️ Clear all metadata")

if clear_meta:
    for key in list(st.session_state.keys()):
        if key in [
            "Client *", "Project *", "Site Type *", "Proposed Vendor *",
            "Proposed Cabinet Location *", "Proposed Radio Location *",
            "Quantity of Sectors *", "Supplier *", "Drawing Type *"
        ]:
            del st.session_state[key]
    st.rerun()

# File upload
uploaded = st.file_uploader("Upload PDF drawing", type=["pdf"])

# OCR toggle
use_ocr = st.checkbox("Use OCR fallback if text extraction is weak", value=False)

# ----------- Validate & Run ---------------------------------------------

def meta_dict() -> Dict[str, Any]:
    return {
        "client": client,
        "project": project,
        "site_type": site_type,
        "vendor": vendor,
        "cabinet_location": cabinet_loc,
        "radio_location": radio_loc,
        "sectors": sectors,
        "mimo_config": (mimo_config if show_mimo else ""),
        "site_address": site_address,
        "supplier": supplier,
        "drawing_type": drawing_type,
    }

def missing_fields(meta: Dict[str, Any]) -> List[str]:
    req = ["client","project","site_type","vendor","cabinet_location",
           "radio_location","sectors","site_address","supplier","drawing_type"]
    miss = [k for k in req if not meta.get(k) and meta.get(k) != 0]
    if show_mimo and not meta.get("mimo_config"):
        miss.append("mimo_config")
    return miss

# Results placeholders
findings: List[Dict[str, Any]] = []
annotated_pdf_bytes: Optional[bytes] = None
excel_bytes: Optional[bytes] = None
outcome = "Pass"
original_filename = ""
page_count = 0

if audit_now:
    if not uploaded:
        st.warning("Please upload a PDF before auditing.")
        st.stop()

    meta = meta_dict()
    miss = missing_fields(meta)
    if miss:
        human = ", ".join([m.replace("_"," ").title() for m in miss])
        st.warning(f"Please complete required fields: {human}")
        st.stop()

    original_filename = uploaded.name

    pdf_bytes = uploaded.read()
    pages, page_count = extract_text_pymupdf(pdf_bytes)

    # Heuristic: use OCR if no text found and toggle is on
    used_ocr = False
    if use_ocr and all((not p["text"].strip()) for p in pages):
        used_ocr = True
        # Keep this simple: still rely on PyMuPDF text (OCR pipeline omitted to avoid extra deps in cloud)
        # In future you can add pdf2image + pytesseract here if desired.
        pass

    # Apply rules
    findings.extend(check_metadata_rules(meta, rules))

    # Spelling
    findings.extend(spelling_checks(pages, sp_allow))

    minor_cnt, major_cnt = summarise_findings(findings)
    total_cnt = minor_cnt + major_cnt
    outcome = "Pass" if total_cnt == 0 else "Rejected"
    rft_percent = 100.0 if outcome == "Pass" else 0.0

    # Annotated PDF & Excel
    try:
        annotated_pdf_bytes = annotate_pdf(pdf_bytes, findings)
    except Exception as e:
        st.warning(f"Could not annotate PDF: {e}")

    excel_bytes = build_excel(findings, meta, original_filename)

    # History row
    append_history({
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "file": original_filename,
        "client": client, "project": project, "site_type": site_type, "vendor": vendor,
        "cabinet_loc": cabinet_loc, "radio_loc": radio_loc, "sectors": sectors,
        "mimo_config": (mimo_config if show_mimo else ""),
        "site_address": site_address,
        "supplier": supplier, "drawing_type": drawing_type,
        "used_ocr": used_ocr, "pages": page_count,
        "minor_findings": minor_cnt, "major_findings": major_cnt, "total_findings": total_cnt,
        "outcome": outcome, "rft_percent": round(rft_percent, 1)
    })

    # Output
    st.success(f"Audit complete — **{outcome}** (Minor: {minor_cnt}, Major: {major_cnt}).")

    # Export filenames keep original + outcome + datestamp
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    base, _ = os.path.splitext(original_filename)
    excel_name = f"{base}_{outcome}_{stamp}.xlsx"
    pdf_name = f"{base}_ANNOTATED_{outcome}_{stamp}.pdf"

    if excel_bytes:
        st.download_button("⬇️ Download Excel report", data=excel_bytes, file_name=excel_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if annotated_pdf_bytes:
        st.download_button("⬇️ Download annotated PDF", data=annotated_pdf_bytes, file_name=pdf_name, mime="application/pdf")

    # Show findings table
    if findings:
        df_find = pd.DataFrame(findings)[["page","kind","severity","message","context"]]
        st.dataframe(df_find, use_container_width=True)
    else:
        st.info("No findings. Nice work!")

# ----------- Analytics ---------------------------------------------------

st.markdown("---")
st.subheader("📈 Audit Analytics")

def render_analytics() -> None:
    _ensure_history()
    if not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0:
        st.info("No history yet. Run a few audits and this dashboard will populate.")
        return
    dfh = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp_utc"])
    if dfh.empty:
        st.info("No history yet.")
        return

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        supplier_f = st.multiselect("Supplier", sorted(dfh["supplier"].dropna().unique().tolist()))
    with f2:
        project_f = st.multiselect("Project", sorted(dfh["project"].dropna().unique().tolist()))
    with f3:
        dtype_f = st.multiselect("Drawing Type", sorted(dfh["drawing_type"].dropna().unique().tolist()))

    mask = pd.Series(True, index=dfh.index)
    if supplier_f: mask &= dfh["supplier"].isin(supplier_f)
    if project_f:  mask &= dfh["project"].isin(project_f)
    if dtype_f:    mask &= dfh["drawing_type"].isin(dtype_f)
    dfv = dfh.loc[mask].sort_values("timestamp_utc")

    if dfv.empty:
        st.warning("No records match the current filters.")
        return

    # Trend RFT with moving average
    tt = dfv[["timestamp_utc","rft_percent"]].copy()
    tt["rft_ma7"] = tt["rft_percent"].rolling(window=7, min_periods=1).mean()
    st.markdown("**Right-First-Time (%) over time**")
    st.line_chart(tt.set_index("timestamp_utc")[["rft_percent","rft_ma7"]])

    # Minor/Major by supplier (stacked)
    st.markdown("**Minor/Major findings by supplier**")
    agg = dfv.groupby("supplier")[["minor_findings","major_findings","total_findings"]].sum().sort_values("total_findings", ascending=False)
    st.bar_chart(agg[["minor_findings","major_findings"]])

    # Average RFT% by supplier
    st.markdown("**Average RFT% by supplier**")
    rft = dfv.groupby("supplier")["rft_percent"].mean().sort_values(ascending=False).to_frame("rft_percent")
    st.bar_chart(rft)

render_analytics()
