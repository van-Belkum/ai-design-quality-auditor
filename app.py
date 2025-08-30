# app.py
from __future__ import annotations

import base64
import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st
import yaml
from rapidfuzz import fuzz, process

# Try to import PyMuPDF for annotation (optional)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# --------------------------- Paths & constants ---------------------------
APP_ROOT = Path(__file__).parent
HISTORY_PATH = APP_ROOT / "history" / "audit_history.csv"
RULES_DEFAULT = APP_ROOT / "rules_example.yaml"

LOGO_ENV = os.getenv("LOGO_FILE") or (st.secrets.get("logo_file") if hasattr(st, "secrets") else None)
LOGO_CANDIDATES = [
    LOGO_ENV,
    str(APP_ROOT / "logo.png"),
    str(APP_ROOT / "logo.jpg"),
    str(APP_ROOT / "logo.jpeg"),
    str(APP_ROOT / "logo.svg"),
    # user-provided specific name
    str(APP_ROOT / "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png"),
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

MANDATORY_FIELDS = [
    "client","project","site_type","vendor","cabinet_loc","radio_loc",
    "sectors","supplier","drawing_type","site_address"
]

# --------------------------- Helpers ---------------------------
def _ensure_history():
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        pd.DataFrame(columns=[
            "timestamp_utc","file","client","project","site_type","vendor",
            "cabinet_loc","radio_loc","sectors","mimo_config","site_address",
            "supplier","drawing_type","used_ocr","pages",
            "minor_findings","major_findings","total_findings",
            "outcome","rft_percent","exclude"
        ]).to_csv(HISTORY_PATH, index=False)

def append_history(row: Dict[str, Any]):
    _ensure_history()
    df = pd.read_csv(HISTORY_PATH)
    df.loc[len(df)] = row
    df.to_csv(HISTORY_PATH, index=False)

def load_rules(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def normalize_text(s: str) -> str:
    return " ".join("".join(ch for ch in str(s).upper() if ch.isalnum() or ch.isspace()).split())

def site_address_check(address: str, filename: str, ignore_pattern: str = ", 0 ,") -> Optional[Dict[str, Any]]:
    """Skip if ', 0 ,' present. Else fuzzy-match address vs filename stem."""
    if ignore_pattern in address:
        return None
    addr_norm = normalize_text(address)
    name_norm = normalize_text(Path(filename).stem)
    score = fuzz.token_set_ratio(addr_norm, name_norm)
    if score < 85:
        return {
            "kind": "Metadata",
            "message": f"Site Address ≠ Filename/Title (similarity {score}).",
            "page": None,
            "severity": "major",
            "bbox": None,
        }
    return None

def extract_pdf_texts(uploaded_file) -> Tuple[List[str], int]:
    if fitz is None:
        b = uploaded_file.getvalue()
        try:
            text = b.decode(errors="ignore")
        except Exception:
            text = ""
        return [text], 1
    else:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        pages = [p.get_text("text") or "" for p in doc]
        n = len(doc)
        doc.close()
        return pages, n

def spelling_checks(pages: List[str], allowlist: List[str]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    if not allowlist:
        return findings
    for i, txt in enumerate(pages, start=1):
        tokens = [t for t in normalize_text(txt).split() if len(t) > 3]
        for t in tokens:
            matches = process.extract(t, allowlist, scorer=fuzz.WRatio, limit=1)
            best = matches[0] if matches else None
            if not best:
                continue
            cand, score, _ = best
            if score < 60:
                findings.append({
                    "kind":"Spelling","message":f"Suspicious token '{t}' (closest '{cand}', {score}).",
                    "page":i,"severity":"minor","bbox":None
                })
    return findings

def run_rules(pages: List[str], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    full = "\n".join(pages).upper()
    for rule in rules.get("checklist", []):
        sev = rule.get("severity","major")
        for req in [s.upper() for s in rule.get("must_contain", [])]:
            if req and req not in full:
                findings.append({"kind":"Checklist","message":f"Missing: '{req}'","page":None,"severity":sev,"bbox":None})
        for bad in [s.upper() for s in rule.get("reject_if_present", [])]:
            if bad and bad in full:
                findings.append({"kind":"Checklist","message":f"Forbidden: '{bad}'","page":None,"severity":sev,"bbox":None})
    return findings

def annotate_pdf(original_bytes: bytes, findings: List[Dict[str, Any]]) -> Optional[bytes]:
    if fitz is None:
        return None
    try:
        doc = fitz.open(stream=original_bytes, filetype="pdf")
        red = (1,0,0)
        for f in findings:
            pg = f.get("page")
            if pg is None or pg < 1 or pg > len(doc):
                continue
            page = doc[pg-1]
            msg = f'{f.get("kind","")}: {f.get("message","")}'
            page.add_text_annot(page.rect.tl + (20, 40), msg)
        out = io.BytesIO()
        doc.save(out)
        doc.close()
        return out.getvalue()
    except Exception:
        return None

def build_excel(findings: List[Dict[str, Any]]) -> bytes:
    df = pd.DataFrame(findings or [])
    with io.BytesIO() as buf:
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            (df if not df.empty else pd.DataFrame([{"note":"No findings"}])).to_excel(xw, index=False, sheet_name="findings")
        return buf.getvalue()

def embed_logo_top_right():
    logo_path = None
    for cand in LOGO_CANDIDATES:
        if cand and Path(cand).exists():
            logo_path = cand; break
    if not logo_path:
        return
    try:
        b64 = base64.b64encode(Path(logo_path).read_bytes()).decode()
        mime = "image/svg+xml" if str(logo_path).lower().endswith(".svg") else "image/png"
        tag = f'<img src="data:{mime};base64,{b64}" class="top-right-logo" />'
        st.markdown("""
        <style>
        .top-nav {display:flex; align-items:center; justify-content:space-between;
                  padding:8px 4px 0 4px;}
        .brand {font-weight:600; font-size:20px; opacity:.9;}
        .top-right-logo{height:36px; opacity:.95;}
        .card {background: rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
               border-radius:10px; padding:14px;}
        .muted {opacity:.7}
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="top-nav"><div class="brand">AI Design QA</div>{tag}</div>', unsafe_allow_html=True)
    except Exception:
        pass

# --------------------------- App UI ---------------------------
st.set_page_config(page_title="AI Design QA", layout="wide")
embed_logo_top_right()

tab_audit, tab_history, tab_settings = st.tabs(["🔎 Audit", "📈 History & Analytics", "⚙️ Settings"])

# ---------- Settings tab ----------
with tab_settings:
    st.subheader("Rule & Spell Settings")
    rules_path = st.text_input("Rules YAML path", str(RULES_DEFAULT))
    allowlist_csv = st.text_area(
        "Spelling allowlist (comma separated)",
        "DIMENSION,DIMENSIONS,MM,STEEL,FOUNDATION,DRAWING,LAYOUT"
    )
    st.caption("Keep this list brief and relevant for best results.")
    st.divider()
    st.subheader("Logo")
    st.write("Place `logo.png`/`logo.jpg`/`logo.svg` in the repo root, or set **LOGO_FILE** env/secret.")
    st.code("logo_file = \"88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png\"  # in Streamlit secrets")

# ---------- History tab ----------
with tab_history:
    _ensure_history()
    try:
        dfh = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp_utc"])
    except Exception:
        dfh = pd.DataFrame()
    if dfh.empty:
        st.info("No history yet.")
    else:
        st.subheader("Recent Runs")
        st.dataframe(dfh.sort_values("timestamp_utc", ascending=False).head(50), use_container_width=True)
        st.markdown("### Trends (excluded runs hidden)")
        df_use = dfh[dfh.get("exclude", False) != True].copy()
        if df_use.empty:
            st.info("Nothing to chart (all runs excluded).")
        else:
            st.line_chart(df_use.set_index("timestamp_utc")["rft_percent"])
            agg = df_use.groupby("supplier")[["minor_findings","major_findings"]].sum().sort_values("major_findings", ascending=False)
            st.bar_chart(agg)

        st.markdown("### Manage Exclusions")
        view = dfh.sort_values("timestamp_utc", ascending=False).head(25)
        edited = st.data_editor(
            view[["timestamp_utc","file","client","project","supplier","drawing_type","outcome","total_findings","exclude"]],
            num_rows="fixed", use_container_width=True
        )
        if st.button("Save history changes"):
            key = ["timestamp_utc","file"]
            merged = dfh.merge(edited[key + ["exclude"]], on=key, how="left", suffixes=("","_new"))
            merged["exclude"] = merged["exclude_new"].fillna(merged["exclude"])
            merged.drop(columns=["exclude_new"], inplace=True)
            merged.to_csv(HISTORY_PATH, index=False)
            st.success("Saved. Refresh to see updated analytics.")

# ---------- Audit tab ----------
with tab_audit:
    st.subheader("Upload & Audit")
    up = st.file_uploader("Design PDF", type=["pdf"], accept_multiple_files=False)

    # Metadata defaults in session
    if "meta" not in st.session_state:
        st.session_state.meta = {
            "client": CLIENTS[0], "project": PROJECTS[0], "site_type": SITE_TYPES[0],
            "vendor": VENDORS[0], "cabinet_loc": CAB_LOCS[0], "radio_loc": RADIO_LOCS[0],
            "sectors": SECTORS[0], "mimo_config": "", "site_address": "",
            "supplier": SUPPLIERS[0], "drawing_type": DRAWING_TYPES[0],
        }

    allow = [t.strip() for t in (allowlist_csv if 'allowlist_csv' in locals() else "").split(",") if t.strip()]
    rules_path_eff = rules_path if 'rules_path' in locals() else str(RULES_DEFAULT)

    with st.form("audit_form", clear_on_submit=False):
        st.markdown("#### Audit Metadata")
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.meta["client"] = st.selectbox("Client", CLIENTS, index=CLIENTS.index(st.session_state.meta["client"]))
            st.session_state.meta["project"] = st.selectbox("Project", PROJECTS, index=PROJECTS.index(st.session_state.meta["project"]))
            st.session_state.meta["site_type"] = st.selectbox("Site Type", SITE_TYPES, index=SITE_TYPES.index(st.session_state.meta["site_type"]))
            st.session_state.meta["vendor"] = st.selectbox("Proposed Vendor", VENDORS, index=VENDORS.index(st.session_state.meta["vendor"]))
            st.session_state.meta["supplier"] = st.selectbox("Supplier", SUPPLIERS, index=SUPPLIERS.index(st.session_state.meta["supplier"]))
        with c2:
            st.session_state.meta["cabinet_loc"] = st.selectbox("Proposed Cabinet Location", CAB_LOCS, index=CAB_LOCS.index(st.session_state.meta["cabinet_loc"]))
            st.session_state.meta["radio_loc"] = st.selectbox("Proposed Radio Location", RADIO_LOCS, index=RADIO_LOCS.index(st.session_state.meta["radio_loc"]))
            st.session_state.meta["sectors"] = st.selectbox("Quantity of Sectors", SECTORS, index=SECTORS.index(st.session_state.meta["sectors"]))

            if st.session_state.meta["project"] != "Power Resilience":
                st.session_state.meta["mimo_config"] = st.text_input("Proposed MIMO Config (optional)", st.session_state.meta["mimo_config"])
            else:
                st.info("Project is Power Resilience → MIMO not required.")
                st.session_state.meta["mimo_config"] = ""

            st.session_state.meta["drawing_type"] = st.selectbox("Drawing Type", DRAWING_TYPES, index=DRAWING_TYPES.index(st.session_state.meta["drawing_type"]))

        st.session_state.meta["site_address"] = st.text_input("Site Address", st.session_state.meta["site_address"],
                                                             help="Must match the design filename/title. Ignored if ', 0 ,' present.")

        excl = st.checkbox("Exclude this audit from analytics", value=False)

        action_col1, action_col2 = st.columns([1,1])
        with action_col1:
            submitted = st.form_submit_button("Run Audit", type="primary")
        with action_col2:
            clear = st.form_submit_button("Clear Metadata")

    if clear:
        st.session_state.meta.update({
            "client": CLIENTS[0], "project": PROJECTS[0], "site_type": SITE_TYPES[0],
            "vendor": VENDORS[0], "cabinet_loc": CAB_LOCS[0], "radio_loc": RADIO_LOCS[0],
            "sectors": SECTORS[0], "mimo_config": "", "site_address": "",
            "supplier": SUPPLIERS[0], "drawing_type": DRAWING_TYPES[0],
        })
        st.experimental_rerun()

    if submitted:
        # Validate required fields
        missing = [k for k in MANDATORY_FIELDS if not str(st.session_state.meta.get(k,"")).strip()]
        if not up:
            st.error("Please upload a PDF to audit.")
        elif missing:
            st.error(f"Please complete all required fields: {', '.join(missing)}.")
        else:
            # Load rules
            rules = load_rules(rules_path_eff)

            # Extract text
            pages, page_count = extract_pdf_texts(up)

            # Findings
            findings: List[Dict[str, Any]] = []
            addr_issue = site_address_check(st.session_state.meta["site_address"], up.name)
            if addr_issue:
                findings.append(addr_issue)

            findings += run_rules(pages, rules)
            try:
                findings += spelling_checks(pages, allow)
            except Exception:
                st.warning("Spelling check skipped due to an internal error.")

            # Outcome
            minors = sum(1 for f in findings if f.get("severity","minor") == "minor")
            majors = sum(1 for f in findings if f.get("severity","major") == "major")
            total = minors + majors
            outcome = "Pass" if total == 0 else "Rejected"
            rft_percent = 100.0 if outcome == "Pass" else max(0.0, 100.0 - (majors*10 + minors*2))

            # Results card
            st.markdown("#### Results")
            with st.container():
                st.write(f"**Outcome:** {outcome} — **Major:** {majors}  •  **Minor:** {minors}  •  **RFT:** {rft_percent:.1f}%")

            df = pd.DataFrame(findings or [])
            st.dataframe(df, use_container_width=True)

            # Exports
            base = Path(up.name).stem
            date_str = datetime.now().strftime("%Y-%m-%d")
            excel_name = f"{base}__{outcome}__{date_str}.xlsx"
            pdf_name = f"{base}__{outcome}__{date_str}.pdf"

            st.download_button(
                "Download Excel report",
                data=build_excel(findings),
                file_name=excel_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            ann_bytes = annotate_pdf(up.getvalue(), findings) if findings else None
            if ann_bytes:
                st.download_button("Download annotated PDF", data=ann_bytes, file_name=pdf_name, mime="application/pdf")
            else:
                st.caption("No annotated PDF generated (no findings or PyMuPDF unavailable).")

            # Save history
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
                "exclude": bool(excl)
            })
            st.toast("Audit saved to history.", icon="💾")
