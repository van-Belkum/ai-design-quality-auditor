# app.py
# AI Design Quality Auditor — streamlined UI with sector-aware MIMO, fixed supplier & radio lists
# Entry password: Seker123 | Rules password: vanB3lkum21

import os
import io
import re
import json
import base64
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd

# Optional heavy deps (guarded import so app boots even if some are missing)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
except Exception:
    rf_process, rf_fuzz = None, None

try:
    from PIL import Image
except Exception:
    Image = None

# ----------------------------- CONSTANTS / CHOICES -----------------------------

ENTRY_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"

# Your logo file (put it in repo root). You can change this name without touching code elsewhere.
LOGO_CANDIDATES = [
    "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png",
    "Seker.png",
    "logo.png",
    "seker-logo.png",
]

CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
DRAWING_TYPES = ["General Arrangement", "Detailed Design"]
CABINET_LOCATIONS = ["Indoor", "Outdoor"]

# ✅ As requested: suppliers live only in app.py and do NOT change rules logic
SUPPLIERS = [
    "CEG",
    "CTIL",
    "Emfyser",
    "Innov8",
    "Invict",
    "KTL Team (Internal)",
    "Trylon",
]

# ✅ Correct Radio Location options
RADIO_LOCATIONS = ["Low Level", "High Level", "Unique Coverage", "Midway"]

# ✅ Full MIMO options (exact list you supplied; order preserved)
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

# History / retention
HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)
DEFAULT_UI_VISIBILITY_HOURS = 24
DAILY_EXPORT_HOUR = 0  # local container time; Streamlit Community often UTC

# ----------------------------- HELPERS -----------------------------

def sticky_logo_html() -> str:
    """Render a bigger, always-visible top-left logo if a candidate exists."""
    logo_path = next((p for p in LOGO_CANDIDATES if os.path.exists(p)), None)
    if not logo_path:
        return """
        <style>.logo-badge{position:fixed;top:8px;left:12px;z-index:9999;
        color:#fff;background:rgba(0,0,0,0.0);padding:0;border-radius:8px;
        font-weight:600;font-size:12px}</style>
        <div class="logo-badge">⚠️ Add a logo (png/jpg/svg) to repo root</div>
        """
    with open(logo_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(logo_path)[1].lower().replace(".", "")
    return f"""
    <style>
      .sticky-logo {{
        position: fixed; top: 14px; left: 16px; z-index: 10000;
        width: 180px; height: auto; opacity: 0.95;
      }}
      @media (max-width: 640px) {{
        .sticky-logo {{ width: 130px; }}
      }}
    </style>
    <img class="sticky-logo" src="data:image/{ext};base64,{b64}" alt="logo"/>
    """

def ensure_dt(s: Any) -> datetime:
    if isinstance(s, datetime):
        return s
    try:
        return pd.to_datetime(s, utc=True).to_pydatetime()
    except Exception:
        return datetime.now(timezone.utc)

def save_history_row(row: Dict[str, Any]) -> None:
    fn = os.path.join(HISTORY_DIR, "audits.csv")
    df = pd.DataFrame([row])
    if os.path.exists(fn):
        df.to_csv(fn, mode="a", header=False, index=False)
    else:
        df.to_csv(fn, index=False)

def load_history() -> pd.DataFrame:
    fn = os.path.join(HISTORY_DIR, "audits.csv")
    if not os.path.exists(fn):
        return pd.DataFrame()
    df = pd.read_csv(fn)
    # Normalize
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df

def filter_history_for_ui(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    if df.empty or "timestamp_utc" not in df.columns:
        return df
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return df[df["timestamp_utc"] >= cutoff].copy()

def annotate_pdf(original_pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    if not fitz:
        return original_pdf_bytes  # graceful fallback
    try:
        doc = fitz.open(stream=original_pdf_bytes, filetype="pdf")
        for f in findings:
            pno = int(f.get("page", 1)) - 1
            if 0 <= pno < len(doc):
                page = doc[pno]
                # draw a small marker near the top-right
                page.insert_text((36, 36 + 14 * (pno % 30)),
                                 f"[{f.get('severity','minor')}] {f.get('rule','')}: {f.get('message','')}",
                                 fontsize=8, color=(1, 0.3, 0.3))
        out = io.BytesIO()
        doc.save(out)
        doc.close()
        return out.getvalue()
    except Exception:
        return original_pdf_bytes

def match_title_to_address(pdf_name: str, site_address: str) -> Tuple[bool, str]:
    """Address must appear in title unless address contains ', 0 ,'."""
    if ", 0 ," in site_address.replace(",0,", ", 0 ,"):
        return True, "Address contains ', 0 ,' so title check is ignored."
    base = os.path.splitext(os.path.basename(pdf_name or ""))[0]
    # Simple containment check after cleaning
    a = re.sub(r"[^A-Z0-9]", "", site_address.upper())
    b = re.sub(r"[^A-Z0-9]", "", base.upper())
    ok = a in b or b in a
    return ok, "OK" if ok else f"Title '{base}' does not match Site Address."

# ----------------------------- RULES IO (YAML TEXT BOX) -----------------------------
# We keep rules in a YAML file on disk, but suppliers are not in rules.

RULES_FILE = "rules_example.yaml"

DEFAULT_RULES_YAML = """\
checklist:
  - name: Title block present
    severity: major
    must_contain: ["TITLE"]
    reject_if_present: []
  - name: Scale must be shown
    severity: minor
    must_contain: ["SCALE"]
    reject_if_present: []
metadata_rules:
  - name: Address matches drawing title (unless ', 0 ,' present)
    field: site_address
    severity: major
    when:
      project_in: ["RAN", "East Unwind", "Beacon 4", "Power Resilience"]
    expr: title_matches_address
"""

def load_rules_text() -> str:
    if not os.path.exists(RULES_FILE):
        with open(RULES_FILE, "w", encoding="utf-8") as f:
            f.write(DEFAULT_RULES_YAML)
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        return f.read()

def save_rules_text(text: str) -> None:
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        f.write(text)

# ----------------------------- AUDIT ENGINE (LIGHTWEIGHT) -----------------------------

def run_audit(pdf_name: str,
              pdf_bytes: bytes,
              meta: Dict[str, Any]) -> Tuple[pd.DataFrame, bytes]:
    """
    Returns (findings_df, annotated_pdf_bytes).
    We enforce: site address vs title; and basic 'must contain' strings if we can read text (guarded).
    """
    findings: List[Dict[str, Any]] = []

    # Rule: address vs title
    ok, msg = match_title_to_address(pdf_name, meta.get("site_address", ""))
    if not ok:
        findings.append({
            "rule": "Address vs Title",
            "severity": "major",
            "page": 1,
            "message": msg
        })

    # Optionally parse pdf text (if PyMuPDF available)
    text = ""
    if fitz and pdf_bytes:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for i, page in enumerate(doc):
                t = page.get_text("text") or ""
                text += f"\n---PAGE {i+1}---\n{t}"
            doc.close()
        except Exception:
            text = ""

    # Minimal 'must contain' checks from rules file (parse quickly via regex for keys)
    rules_raw = load_rules_text()
    # naive parse of 'must_contain' values to avoid full yaml load here (keeps resilience)
    musts = re.findall(r"must_contain:\s*\[(.*?)\]", rules_raw)
    tokens = []
    for m in musts:
        parts = [p.strip().strip('"').strip("'") for p in m.split(",") if p.strip()]
        tokens.extend(parts)
    for token in tokens:
        if token and token.upper() not in (text.upper() if text else ""):
            findings.append({
                "rule": f"Must contain: {token}",
                "severity": "minor",
                "page": 1,
                "message": f"'{token}' not detected in PDF text."
            })

    annotated = annotate_pdf(pdf_bytes, findings) if pdf_bytes else b""
    df = pd.DataFrame(findings) if findings else pd.DataFrame(columns=["rule","severity","page","message"])
    return df, annotated

# ----------------------------- UI -----------------------------

st.set_page_config(page_title="AI Design Quality Auditor", layout="wide", page_icon="✅")
st.markdown(sticky_logo_html(), unsafe_allow_html=True)

# Password gate
if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    st.title("AI Design Quality Auditor")
    pw = st.text_input("Enter access password", type="password")
    if st.button("Enter"):
        if pw == ENTRY_PASSWORD:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

st.title("AI Design Quality Auditor")
st.caption("Professional, consistent design QA with audit trail, analytics, annotations, and easy rule updates.")

tab_audit, tab_analytics, tab_settings = st.tabs(["🔎 Audit", "📈 Analytics", "⚙️ Settings"])

with tab_audit:
    st.subheader("Audit Metadata (required)")

    c1, c2, c3, c4 = st.columns(4)
    supplier = c1.selectbox("Supplier", SUPPLIERS, index=None, placeholder="— Select —")
    drawing_type = c2.selectbox("Drawing Type", DRAWING_TYPES, index=None, placeholder="— Select —")
    client = c3.selectbox("Client", CLIENTS, index=None, placeholder="— Select —")
    project = c4.selectbox("Project", PROJECTS, index=None, placeholder="— Select —")

    c5, c6, c7, c8 = st.columns(4)
    site_type = c5.selectbox("Site Type", SITE_TYPES, index=None, placeholder="— Select —")
    vendor = c6.selectbox("Proposed Vendor", VENDORS, index=None, placeholder="— Select —")
    cabinet_loc = c7.selectbox("Proposed Cabinet Location", CABINET_LOCATIONS, index=None, placeholder="— Select —")
    radio_loc = c8.selectbox("Proposed Radio Location", RADIO_LOCATIONS, index=None, placeholder="— Select —")

    c9, = st.columns(1)
    qty_sectors = c9.selectbox("Quantity of Sectors", [1,2,3,4,5,6], index=None, placeholder="— Select —")

    site_address = st.text_input("Site Address", placeholder="e.g. MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ")

    # MIMO area (optional when Power Resilience)
    mimo_optional = (project == "Power Resilience")
    st.markdown(f"**Proposed MIMO Config ({'optional for Power Resilience' if mimo_optional else 'required'})**")

    use_same = st.checkbox("Use same config for all sectors", value=True, help="Tick to copy S1 to all sectors.")
    mimo_selected = {}

    if qty_sectors:
        cols = st.columns(min(qty_sectors, 3))
        # Create rows of up to 3 columns
        for idx in range(1, qty_sectors + 1):
            col = cols[(idx-1) % len(cols)]
            mimo_selected[f"S{idx}"] = col.selectbox(
                f"MIMO S{idx}",
                MIMO_OPTIONS,
                index=None,
                placeholder="— Select —"
            )
            if idx % 3 == 0 and idx < qty_sectors:
                cols = st.columns(min(qty_sectors - idx, 3))

        if use_same and mimo_selected.get("S1"):
            # copy S1 to others visually – user can still change if they want
            base = mimo_selected["S1"]
            for i in range(2, qty_sectors+1):
                if not mimo_selected.get(f"S{i}"):
                    mimo_selected[f"S{i}"] = base

    st.divider()
    pdf_file = st.file_uploader("Upload design PDF", type=["pdf"])

    # Buttons
    colb1, colb2, colb3 = st.columns([1,1,6])
    run_clicked = colb1.button("▶️ Run Audit", type="primary")
    clear_clicked = colb2.button("🧹 Clear all metadata")

    if clear_clicked:
        for k in ["supplier","drawing_type","client","project","site_type","vendor","cabinet_loc","radio_loc","qty_sectors","site_address"]:
            if k in st.session_state: del st.session_state[k]
        st.rerun()

    # Validate
    def meta_ok() -> Tuple[bool, str]:
        req = [supplier, drawing_type, client, project, site_type, vendor, cabinet_loc, radio_loc, qty_sectors, site_address]
        if any(x in [None, "", 0] for x in req):
            return False, "Please complete all metadata fields."
        # MIMO required unless Power Resilience
        if not mimo_optional:
            if qty_sectors and (any(not mimo_selected.get(f"S{i}") for i in range(1, qty_sectors+1))):
                return False, "Please select MIMO configuration for all sectors."
        return True, "OK"

    if run_clicked:
        ok, msg = meta_ok()
        if not ok:
            st.error(msg)
        elif not pdf_file:
            st.error("Please upload a PDF.")
        else:
            meta = dict(
                supplier=supplier,
                drawing_type=drawing_type,
                client=client,
                project=project,
                site_type=site_type,
                vendor=vendor,
                cabinet_location=cabinet_loc,
                radio_location=radio_loc,
                qty_sectors=qty_sectors,
                site_address=site_address,
                mimo={k:v for k,v in mimo_selected.items() if v}
            )
            pdf_bytes = pdf_file.read()
            findings_df, ann_bytes = run_audit(pdf_file.name, pdf_bytes, meta)

            # Status
            status = "Pass" if findings_df.empty else "Rejected"
            st.success(f"Audit complete — **{status}**")

            # Downloads
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            base = os.path.splitext(pdf_file.name)[0]
            report_name = f"{base} - {status} - {ts}.xlsx"

            if not findings_df.empty:
                st.dataframe(findings_df, use_container_width=True)
            else:
                st.info("No findings.")

            # Excel
            xbuf = io.BytesIO()
            with pd.ExcelWriter(xbuf, engine="openpyxl") as xl:
                (findings_df if not findings_df.empty else pd.DataFrame([{"note":"No findings."}])).to_excel(xl, index=False, sheet_name="Findings")
                pd.DataFrame([meta]).to_excel(xl, index=False, sheet_name="Metadata")
            st.download_button("⬇️ Download report (Excel)", xbuf.getvalue(), file_name=report_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # Annotated PDF
            if ann_bytes:
                st.download_button("⬇️ Download annotated PDF", ann_bytes, file_name=f"{base} - annotations.pdf", mime="application/pdf")

            # Save to history
            save_history_row({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "file": pdf_file.name,
                "client": client,
                "project": project,
                "site_type": site_type,
                "vendor": vendor,
                "radio_location": radio_loc,
                "qty_sectors": qty_sectors,
                "drawing_type": drawing_type,
                "supplier": supplier,
                "findings_count": len(findings_df.index),
                "mimo_json": json.dumps(meta.get("mimo", {})),
            })

with tab_analytics:
    st.subheader("Performance & Trends (last 24h by default)")
    vis_hours = st.slider("Show records from last (hours)", 1, 24*14, value=DEFAULT_UI_VISIBILITY_HOURS, step=1)
    dfh = load_history()
    if not dfh.empty and "timestamp_utc" in dfh.columns:
        dfh["timestamp_utc"] = pd.to_datetime(dfh["timestamp_utc"], utc=True, errors="coerce")
    dfv = filter_history_for_ui(dfh, vis_hours)

    if dfv.empty:
        st.info("No records yet.")
    else:
        st.metric("Total audits", len(dfv))
        st.metric("Right-first-time %", round((dfv["status"].eq("Pass").mean() * 100.0), 1))
        by_vendor = dfv.groupby(["vendor","status"]).size().unstack(fill_value=0)
        st.write("By vendor (counts):")
        st.dataframe(by_vendor)

        by_project = dfv.groupby(["project","status"]).size().unstack(fill_value=0)
        st.write("By project (counts):")
        st.dataframe(by_project)

        st.write("Raw (filtered) history:")
        st.dataframe(dfv.sort_values("timestamp_utc", ascending=False), use_container_width=True)

        # Download filtered
        csv = dfv.to_csv(index=False).encode()
        st.download_button("⬇️ Download filtered CSV", csv, file_name="audits_filtered.csv", mime="text/csv")

with tab_settings:
    st.subheader("Edit rules (YAML)")
    st.caption("Suppliers are fixed in code; rules should key off project/vendor/config, not supplier.")
    pw = st.text_input("Rules password", type="password", help="Required to save changes.")
    text = st.text_area(RULES_FILE, value=load_rules_text(), height=380)

    csa, csb, csc = st.columns([1,1,4])
    if csa.button("Save rules"):
        if pw == RULES_PASSWORD:
            try:
                save_rules_text(text)
                st.success("Rules saved.")
            except Exception as e:
                st.error(f"Failed to save: {e}")
        else:
            st.error("Incorrect rules password.")

    if csb.button("Reload from disk"):
        st.experimental_rerun()

    st.divider()
    if st.button("Clear all history (keeps files)"):
        fn = os.path.join(HISTORY_DIR, "audits.csv")
        if os.path.exists(fn):
            os.remove(fn)
        st.success("History cleared.")
