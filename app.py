# app.py
from __future__ import annotations

import base64
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st
import yaml
from rapidfuzz import fuzz, process

# Optional heavy deps (only used when present)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# ---------- Constants & paths ----------
APP_ROOT = Path(__file__).parent
HISTORY_PATH = APP_ROOT / "history" / "audit_history.csv"
RULES_DEFAULT = APP_ROOT / "rules_example.yaml"

LOGO_ENV = os.getenv("LOGO_FILE") or st.secrets.get("logo_file", None) if hasattr(st, "secrets") else None
LOGO_CANDIDATES = [
    LOGO_ENV,
    str(APP_ROOT / "logo.png"),
    str(APP_ROOT / "logo.jpg"),
    str(APP_ROOT / "logo.jpeg"),
    str(APP_ROOT / "logo.svg"),
    # specific user file (rename is recommended, but we still try it)
    str(APP_ROOT / "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png"),
]

MANDATORY_FIELDS = [
    "client", "project", "site_type", "vendor", "cabinet_loc", "radio_loc",
    "sectors", "supplier", "drawing_type", "site_address"
]

CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
CAB_LOCS = ["Indoor", "Outdoor"]
RADIO_LOCS = ["High Level", "Low Level", "Indoor", "Door"]
SECTORS = ["1", "2", "3", "4", "5", "6"]

SUPPLIERS = ["CEG", "CTIL", "Emfyser", "Innov8", "Invict", "KTL Team (Internal)", "Trylon"]
DRAWING_TYPES = ["General Arrangement", "Detailed Design"]

# -------------- Utilities --------------
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
                "outcome","rft_percent",
                "exclude"
            ])

def append_history(row: Dict[str, Any]):
    _ensure_history()
    import csv
    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            row["timestamp_utc"], row["file"], row["client"], row["project"], row["site_type"], row["vendor"],
            row["cabinet_loc"], row["radio_loc"], row["sectors"], row.get("mimo_config",""), row["site_address"],
            row["supplier"], row["drawing_type"], row.get("used_ocr", False), row.get("pages", 0),
            row.get("minor_findings",0), row.get("major_findings",0), row.get("total_findings",0),
            row["outcome"], row.get("rft_percent",0.0),
            row.get("exclude", False)
        ])

def load_rules(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def embed_logo_top_right():
    logo_path = None
    for cand in LOGO_CANDIDATES:
        if cand and Path(cand).exists():
            logo_path = cand
            break
    if not logo_path:
        st.warning("⚠️ Logo file not found in repo root (png/svg/jpg).")
        return
    try:
        ext = Path(logo_path).suffix.lower()
        data = Path(logo_path).read_bytes()
        if ext == ".svg":
            b64 = base64.b64encode(data).decode()
            tag = f'<img src="data:image/svg+xml;base64,{b64}" class="top-right-logo" />'
        else:
            b64 = base64.b64encode(data).decode()
            tag = f'<img src="data:image/png;base64,{b64}" class="top-right-logo" />'
        st.markdown(
            """
            <style>
            .top-right-logo{
                position:fixed; top:14px; right:18px; height:40px; z-index:1000; opacity:0.92;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(tag, unsafe_allow_html=True)
    except Exception:
        st.warning("⚠️ Could not render logo.")

def normalize_text(s: str) -> str:
    return " ".join("".join(ch for ch in s.upper() if ch.isalnum() or ch.isspace()).split())

def site_address_check(address: str, filename: str, ignore_pattern: str = ", 0 ,") -> Optional[Dict[str, Any]]:
    """
    If address contains ', 0 ,' ignore. Otherwise require a decent fuzzy match with filename/title.
    """
    if ignore_pattern in address:
        return None

    addr_norm = normalize_text(address)
    name_norm = normalize_text(Path(filename).stem)

    score = fuzz.token_set_ratio(addr_norm, name_norm)
    if score < 85:
        return {
            "kind": "Metadata",
            "message": f"Site Address mismatch with filename/title (score {score}).",
            "page": None,
            "severity": "major",
            "bbox": None
        }
    return None

# ---- Simple OCR/Text Extractor (placeholder for your real logic) ----
def extract_pdf_texts(uploaded_file) -> Tuple[List[str], int]:
    """
    Returns a list of per-page text (very light), and page count.
    If PyMuPDF not available, we fallback to reading bytes and return a single "page" text.
    """
    if fitz is None:
        b = uploaded_file.getvalue()
        try:
            text = b.decode(errors="ignore")
        except Exception:
            text = ""
        return [text], 1
    else:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        pages = []
        for p in doc:
            pages.append(p.get_text("text") or "")
        n = len(doc)
        doc.close()
        return pages, n

# ------ Spelling checks (defensive) ------
def spelling_checks(pages: List[str], allowlist: List[str]) -> List[Dict[str, Any]]:
    """
    Very lightweight spelling check using fuzzy candidates from the allowlist.
    Produces only "minor" notes when a suspicious word is far from any allowed term.
    """
    findings: List[Dict[str, Any]] = []
    if not allowlist:
        return findings

    tokens_by_page = []
    for i, txt in enumerate(pages, start=1):
        tokens = [t for t in normalize_text(txt).split() if len(t) > 3]
        tokens_by_page.append((i, tokens))

    for page_no, tokens in tokens_by_page:
        for t in tokens:
            # find best candidate, but guard against empty iterables
            matches = process.extract(t, allowlist, scorer=fuzz.WRatio, limit=1)
            best = matches[0] if matches else None
            if not best:
                continue
            candidate, score, _ = best
            # If token doesn't resemble any allowlisted word, flag
            if score < 60:
                findings.append({
                    "kind": "Spelling",
                    "message": f"Suspicious token '{t}' (best ref '{candidate}', score {score}).",
                    "page": page_no,
                    "severity": "minor",
                    "bbox": None
                })
    return findings

# ------ Rules engine (very compact) ------
def run_rules(pages: List[str], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Minimal rules: supports 'must_contain' phrases and 'reject_if_present' phrases.
    """
    findings: List[Dict[str, Any]] = []
    full_text = "\n".join(pages).upper()

    for rule in rules.get("checklist", []):
        name = rule.get("name", "unnamed")
        need = [s.upper() for s in rule.get("must_contain", [])]
        bad = [s.upper() for s in rule.get("reject_if_present", [])]
        sev = rule.get("severity", "major")

        for req in need:
            if req and req not in full_text:
                findings.append({
                    "kind": "Checklist",
                    "message": f"Missing required text: '{req}'",
                    "page": None,
                    "severity": sev,
                    "bbox": None
                })
        for blk in bad:
            if blk and blk in full_text:
                findings.append({
                    "kind": "Checklist",
                    "message": f"Forbidden text present: '{blk}'",
                    "page": None,
                    "severity": sev,
                    "bbox": None
                })
    return findings

# ------ PDF annotation ------
def annotate_pdf(original_bytes: bytes, findings: List[Dict[str, Any]]) -> Optional[bytes]:
    if fitz is None:
        return None
    try:
        doc = fitz.open(stream=original_bytes, filetype="pdf")
        red = (1, 0, 0)
        for f in findings:
            pg = f.get("page")
            bbox = f.get("bbox")
            if pg is None or pg < 1 or pg > len(doc):
                continue
            page = doc[pg - 1]
            # Draw a simple callout rectangle if bbox is a proper list of 4 numbers
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(x,(int,float)) for x in bbox):
                rect = fitz.Rect(*bbox)
                page.draw_rect(rect, color=red, width=1.2)
            # Always add a sticky note with the message
            msg = f'{f.get("kind","")}: {f.get("message","")}'
            page.add_text_annot(page.rect.tl + (20, 40), msg)
        out = io.BytesIO()
        doc.save(out)
        doc.close()
        return out.getvalue()
    except Exception:
        return None

# ------ Excel export ------
def build_excel(findings: List[Dict[str, Any]]) -> bytes:
    df = pd.DataFrame(findings or [])
    with io.BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="openpyxl") as xw:
            (df if not df.empty else pd.DataFrame([{"note":"No findings"}])).to_excel(
                xw, index=False, sheet_name="findings"
            )
        return buffer.getvalue()

# ------ Analytics ------
def render_analytics():
    _ensure_history()
    try:
        dfh = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp_utc"])
    except Exception:
        st.info("No history yet.")
        return

    if "exclude" not in dfh.columns:
        dfh["exclude"] = False
    dfh_use = dfh[dfh["exclude"] != True].copy()

    st.subheader("📈 Trends (excluded runs hidden)")
    if dfh_use.empty:
        st.info("No included runs to chart yet.")
    else:
        # RFT over time
        t = dfh_use.sort_values("timestamp_utc")
        st.line_chart(t.set_index("timestamp_utc")["rft_percent"])

        # Minor/Major by Supplier
        agg = dfh_use.groupby("supplier")[["minor_findings","major_findings"]].sum().sort_values("major_findings", ascending=False)
        st.bar_chart(agg)

    with st.expander("🧰 History Manager (toggle exclude on past runs)"):
        view = dfh.sort_values("timestamp_utc", ascending=False).head(25)
        edited = st.data_editor(
            view[["timestamp_utc","file","client","project","supplier","drawing_type","outcome","total_findings","exclude"]],
            num_rows="fixed",
            use_container_width=True
        )
        if st.button("Save changes to history"):
            merge_cols = ["timestamp_utc","file"]
            dfh = dfh.merge(
                edited[merge_cols + ["exclude"]],
                on=merge_cols, how="left", suffixes=("","_new")
            )
            dfh["exclude"] = dfh["exclude_new"].fillna(dfh["exclude"])
            dfh.drop(columns=["exclude_new"], inplace=True)
            dfh.to_csv(HISTORY_PATH, index=False)
            st.success("History updated. Refresh analytics above.")

# ------ UI + App ------
st.set_page_config(page_title="AI Design QA", layout="wide")
embed_logo_top_right()

st.title("AI Design QA")

# Sidebar: rule file + guidance
with st.sidebar:
    st.header("Config")
    rules_file = st.text_input("Rules YAML path", str(RULES_DEFAULT))
    allowlist_csv = st.text_area(
        "Spelling allowlist (comma-separated)",
        value="DIMENSION,DIMENSIONS,MM,STEEL,FOUNDATION,DRAWING,LAYOUT"
    )
    allow = [t.strip() for t in allowlist_csv.split(",") if t.strip()]
    st.caption("Tip: keep this list short and specific.")

    st.markdown("---")
    st.header("Analytics")
    render_analytics()

# Metadata state
if "meta" not in st.session_state:
    st.session_state.meta = {
        "client": CLIENTS[0],
        "project": PROJECTS[0],
        "site_type": SITE_TYPES[0],
        "vendor": VENDORS[0],
        "cabinet_loc": CAB_LOCS[0],
        "radio_loc": RADIO_LOCS[0],
        "sectors": SECTORS[0],
        "mimo_config": "",
        "site_address": "",
        "supplier": SUPPLIERS[0],
        "drawing_type": DRAWING_TYPES[0],
    }

colL, colR = st.columns([1.2, 1.8], gap="large")

with colL:
    st.subheader("Audit Metadata")
    st.session_state.meta["client"] = st.selectbox("Client", CLIENTS, index=CLIENTS.index(st.session_state.meta["client"]))
    st.session_state.meta["project"] = st.selectbox("Project", PROJECTS, index=PROJECTS.index(st.session_state.meta["project"]))
    st.session_state.meta["site_type"] = st.selectbox("Site Type", SITE_TYPES, index=SITE_TYPES.index(st.session_state.meta["site_type"]))
    st.session_state.meta["vendor"] = st.selectbox("Proposed Vendor", VENDORS, index=VENDORS.index(st.session_state.meta["vendor"]))
    st.session_state.meta["cabinet_loc"] = st.selectbox("Proposed Cabinet Location", CAB_LOCS, index=CAB_LOCS.index(st.session_state.meta["cabinet_loc"]))
    st.session_state.meta["radio_loc"] = st.selectbox("Proposed Radio Location", RADIO_LOCS, index=RADIO_LOCS.index(st.session_state.meta["radio_loc"]))
    st.session_state.meta["sectors"] = st.selectbox("Quantity of Sectors", SECTORS, index=SECTORS.index(st.session_state.meta["sectors"]))

    # Hide MIMO when Power Resilience
    if st.session_state.meta["project"] != "Power Resilience":
        st.session_state.meta["mimo_config"] = st.text_input("Proposed MIMO Config (S1/S2/S3 or single)", st.session_state.meta["mimo_config"])
    else:
        st.info("Project is Power Resilience → Proposed MIMO Config not required.")
        st.session_state.meta["mimo_config"] = ""

    st.session_state.meta["supplier"] = st.selectbox("Supplier", SUPPLIERS, index=SUPPLIERS.index(st.session_state.meta["supplier"]))
    st.session_state.meta["drawing_type"] = st.selectbox("Drawing Type", DRAWING_TYPES, index=DRAWING_TYPES.index(st.session_state.meta["drawing_type"]))

    st.session_state.meta["site_address"] = st.text_input("Site Address", st.session_state.meta["site_address"], help="Must match the PDF file title; ignored if the address contains ', 0 ,'.")

    exclude_from_analytics = st.checkbox(
        "Exclude this audit from analytics",
        value=False,
        help="Useful while iterating rules."
    )

    meta_ok = all(str(st.session_state.meta.get(k,"")).strip() for k in MANDATORY_FIELDS)

    cols = st.columns(2)
    with cols[0]:
        run_audit = st.button("🔎 Run Audit", type="primary", disabled=not meta_ok)
        if not meta_ok:
            st.caption("Complete all metadata fields to enable auditing.")
    with cols[1]:
        clear_meta = st.button("🧹 Clear metadata")
        if clear_meta:
            st.session_state.meta.update({
                "client": CLIENTS[0],
                "project": PROJECTS[0],
                "site_type": SITE_TYPES[0],
                "vendor": VENDORS[0],
                "cabinet_loc": CAB_LOCS[0],
                "radio_loc": RADIO_LOCS[0],
                "sectors": SECTORS[0],
                "mimo_config": "",
                "site_address": "",
                "supplier": SUPPLIERS[0],
                "drawing_type": DRAWING_TYPES[0],
            })
            st.experimental_rerun()

with colR:
    st.subheader("Upload PDF")
    up = st.file_uploader("Design PDF", type=["pdf"], accept_multiple_files=False)

    if run_audit:
        if not up:
            st.error("Please upload a PDF.")
        else:
            rules = load_rules(rules_file)
            # Extract text
            pages, page_count = extract_pdf_texts(up)

            findings: List[Dict[str, Any]] = []
            # Address rule
            addr_issue = site_address_check(st.session_state.meta["site_address"], up.name)
            if addr_issue:
                findings.append(addr_issue)

            # Rules
            findings += run_rules(pages, rules)

            # Spelling (guarded)
            try:
                findings += spelling_checks(pages, allow)
            except Exception:
                st.warning("Spelling check skipped due to an internal error.")

            # Outcome calc
            minors = sum(1 for f in findings if f.get("severity","minor") == "minor")
            majors = sum(1 for f in findings if f.get("severity","major") == "major")
            total = minors + majors
            outcome = "Pass" if majors == 0 and total == 0 else "Rejected"
            rft_percent = 100.0 if outcome == "Pass" else max(0.0, 100.0 - (majors*10 + minors*2))

            # Show table
            df = pd.DataFrame(findings or [])
            st.success(f"Outcome: {outcome} — M:{majors} m:{minors} (RFT {rft_percent:.1f}%)")
            st.dataframe(df, use_container_width=True)

            # Exports
            base = Path(up.name).stem
            date_str = datetime.now().strftime("%Y-%m-%d")
            excel_name = f"{base}__{outcome}__{date_str}.xlsx"
            pdf_name = f"{base}__{outcome}__{date_str}.pdf"

            xls_bytes = build_excel(findings)
            st.download_button("⬇️ Download Excel report", data=xls_bytes, file_name=excel_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            ann_bytes = None
            orig_bytes = up.getvalue()
            ann_bytes = annotate_pdf(orig_bytes, findings) if findings else None
            if ann_bytes:
                st.download_button("⬇️ Download annotated PDF", data=ann_bytes, file_name=pdf_name, mime="application/pdf")
            else:
                st.caption("No annotated PDF generated (no findings or PyMuPDF unavailable).")

            # History
            append_history({
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "file": up.name,
                "client": st.session_state.meta["client"],
                "project": st.session_state.meta["project"],
                "site_type": st.session_state.meta["site_type"],
                "vendor": st.session_state.meta["vendor"],
                "cabinet_loc": st.session_state.meta["cabinet_loc"],
                "radio_loc": st.session_state.meta["radio_loc"],
                "sectors": st.session_state.meta["sectors"],
                "mimo_config": st.session_state.meta["mimo_config"],
                "site_address": st.session_state.meta["site_address"],
                "supplier": st.session_state.meta["supplier"],
                "drawing_type": st.session_state.meta["drawing_type"],
                "used_ocr": False,
                "pages": page_count,
                "minor_findings": minors,
                "major_findings": majors,
                "total_findings": total,
                "outcome": outcome,
                "rft_percent": round(rft_percent, 1),
                "exclude": bool(exclude_from_analytics)
            })
            st.toast("Run saved to history.", icon="💾")
