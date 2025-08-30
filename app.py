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

# Optional: PyMuPDF (annotations). If not present, annotated PDF is skipped gracefully.
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# --------------------------- Paths & constants ---------------------------
APP_ROOT = Path(__file__).parent
HISTORY_PATH = APP_ROOT / "history" / "audit_history.csv"
RULES_DEFAULT = APP_ROOT / "rules_example.yaml"
ALLOWLIST_FILE = APP_ROOT / "allowlist.txt"  # persisted allowlist (one line, comma-separated)

LOGO_ENV = os.getenv("LOGO_FILE") or (st.secrets.get("logo_file") if hasattr(st, "secrets") else None)
LOGO_CANDIDATES = [
    LOGO_ENV,
    str(APP_ROOT / "logo.png"),
    str(APP_ROOT / "logo.jpg"),
    str(APP_ROOT / "logo.jpeg"),
    str(APP_ROOT / "logo.svg"),
    # your uploaded file name (kept here for convenience)
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

# MIMO is validated dynamically (required unless Power Resilience)
MANDATORY_FIELDS_BASE = [
    "client","project","site_type","vendor","cabinet_loc","radio_loc",
    "sectors","supplier","drawing_type","site_address"
]

# --------------------------- Small helpers ---------------------------
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
        return {"checklist": []}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "checklist" not in data or not isinstance(data["checklist"], list):
        data["checklist"] = []
    return data

def save_rules(path: Union[str, Path], data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def load_allowlist() -> List[str]:
    if ALLOWLIST_FILE.exists():
        txt = ALLOWLIST_FILE.read_text(encoding="utf-8")
        return [t.strip() for t in txt.split(",") if t.strip()]
    # default bootstrap terms
    return ["DIMENSION","DIMENSIONS","MM","STEEL","FOUNDATION","DRAWING","LAYOUT"]

def save_allowlist(tokens: List[str]) -> None:
    ALLOWLIST_FILE.write_text(",".join(sorted(set([t.strip() for t in tokens if t.strip()]))), encoding="utf-8")

def normalize_text(s: str) -> str:
    return " ".join("".join(ch for ch in str(s).upper() if ch.isalnum() or ch.isspace()).split())

def site_address_check(address: str, filename: str, ignore_pattern: str = ", 0 ,") -> Optional[Dict[str, Any]]:
    """Skip if literal ', 0 ,' present in address. Else fuzzy-match address vs file stem."""
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
        fbytes = uploaded_file.read()
        doc = fitz.open(stream=fbytes, filetype="pdf")
        pages = [p.get_text("text") or "" for p in doc]
        n = len(doc)
        doc.close()
        return pages, n

def spelling_checks(pages: List[str], allowlist: List[str]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    if not allowlist:
        return findings
    allow_upper = [a.upper() for a in allowlist]
    for i, txt in enumerate(pages, start=1):
        tokens = [t for t in normalize_text(txt).split() if len(t) > 3]
        for t in tokens:
            # skip if token looks like an allowed word exactly
            if t in allow_upper:
                continue
            # find best approximate match in allowlist; if it's too far, flag.
            best = process.extractOne(t, allow_upper, scorer=fuzz.WRatio)
            if not best:
                continue
            _, score = best[0], best[1]
            if score < 60:
                findings.append({
                    "kind":"Spelling",
                    "message":f"Suspicious token '{t}' (closest allowlist score {score}).",
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
        for f in findings:
            pg = f.get("page")
            if pg is None or not (1 <= pg <= len(doc)):
                continue
            page = doc[pg-1]
            msg = f'{f.get("kind","")}: {f.get("message","")}'
            # Add a simple text annotation near top-left margin
            page.add_text_annot(page.rect.tl + (24, 48), msg)
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
    # style first
    st.markdown("""
    <style>
      .top-nav {position: sticky; top: 0; z-index: 9999;
                display:flex; align-items:center; justify-content:space-between;
                background: var(--background-color, rgba(0,0,0,0));
                padding: 6px 4px 2px 4px;}
      .brand {font-weight:600; font-size:20px; opacity:.9;}
      .top-right-logo{height:64px; opacity:.97; } /* <— bigger logo */
      .card {background: rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
             border-radius:10px; padding:14px;}
      .muted {opacity:.7}
    </style>
    """, unsafe_allow_html=True)
    tag = ""
    if logo_path:
        try:
            b64 = base64.b64encode(Path(logo_path).read_bytes()).decode()
            mime = "image/svg+xml" if str(logo_path).lower().endswith(".svg") else "image/png"
            tag = f'<img src="data:{mime};base64,{b64}" class="top-right-logo" />'
        except Exception:
            pass
    st.markdown(f'<div class="top-nav"><div class="brand">AI Design QA</div>{tag}</div>', unsafe_allow_html=True)

# --------------------------- App UI ---------------------------
st.set_page_config(page_title="AI Design QA", layout="wide")
embed_logo_top_right()

tab_audit, tab_history, tab_settings = st.tabs(["🔎 Audit", "📈 History & Analytics", "⚙️ Settings"])

# ---------- Settings tab ----------
with tab_settings:
    st.subheader("Rules & Spelling — Simple Management")

    # RULES: load current
    rules_path_input = st.text_input("Rules YAML path", str(RULES_DEFAULT))
    current_rules = load_rules(rules_path_input)

    st.markdown("##### Quick Rule Builder")
    with st.form("quick_rule_form"):
        qr_name = st.text_input("Rule name (for your reference)", "")
        qr_sev = st.selectbox("Severity", ["major","minor"], index=0)
        qr_must = st.text_input("Must contain (comma-separated)", "")
        qr_reject = st.text_input("Reject if present (comma-separated)", "")
        add_rule = st.form_submit_button("Add Rule to YAML")
        if add_rule:
            new_rule = {
                "name": qr_name or f"Rule {len(current_rules.get('checklist',[]))+1}",
                "severity": qr_sev,
                "must_contain": [t.strip() for t in qr_must.split(",") if t.strip()],
                "reject_if_present": [t.strip() for t in qr_reject.split(",") if t.strip()],
            }
            current_rules.setdefault("checklist", []).append(new_rule)
            try:
                save_rules(rules_path_input, current_rules)
                st.success(f"Rule added and saved to {rules_path_input}.")
            except Exception as e:
                st.error(f"Failed to save rule: {e}")

    st.markdown("##### Full YAML Editor")
    yaml_text = st.text_area(
        "Edit your full rules YAML below and click Validate & Save.",
        value=yaml.safe_dump(current_rules, sort_keys=False, allow_unicode=True),
        height=260
    )
    colY1, colY2 = st.columns([1,1])
    with colY1:
        if st.button("Validate & Save YAML", type="primary"):
            try:
                parsed = yaml.safe_load(yaml_text) or {}
                if "checklist" in parsed and not isinstance(parsed["checklist"], list):
                    raise ValueError("'checklist' must be a list")
                save_rules(rules_path_input, parsed)
                st.success("YAML validated and saved.")
            except Exception as e:
                st.error(f"YAML error: {e}")

    with colY2:
        st.download_button("Download current YAML", data=yaml_text.encode("utf-8"),
                           file_name="rules_example.yaml", mime="text/yaml")

    st.divider()
    st.subheader("Spelling Allowlist")
    existing_allow = load_allowlist()
    allow_text = st.text_area(
        "Comma-separated allowlist",
        value=",".join(existing_allow),
        height=100
    )
    if st.button("Save Allowlist"):
        save_allowlist([t.strip() for t in allow_text.split(",")])
        st.success("Allowlist saved.")

    st.divider()
    st.subheader("Logo")
    st.write("Place `logo.png`/`logo.jpg`/`logo.svg` in repo root, or set **LOGO_FILE** env/secret.")
    st.code('logo_file = "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png"  # in Streamlit secrets')

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

    # Session defaults
    if "meta" not in st.session_state:
        st.session_state.meta = {
            "client": CLIENTS[0], "project": PROJECTS[0], "site_type": SITE_TYPES[0],
            "vendor": VENDORS[0], "cabinet_loc": CAB_LOCS[0], "radio_loc": RADIO_LOCS[0],
            "sectors": SECTORS[0], "mimo_config": "", "site_address": "",
            "supplier": SUPPLIERS[0], "drawing_type": DRAWING_TYPES[0],
        }

    # Load allowlist from file each render (so Settings save takes effect)
    allow = load_allowlist()

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

            # MIMO: optional for Power Resilience only; required otherwise
            if st.session_state.meta["project"] == "Power Resilience":
                st.info("Project is Power Resilience → MIMO not required.")
                st.session_state.meta["mimo_config"] = ""
            else:
                st.session_state.meta["mimo_config"] = st.text_input("Proposed MIMO Config (optional for Power Resilience only)", st.session_state.meta["mimo_config"])

            st.session_state.meta["drawing_type"] = st.selectbox("Drawing Type", DRAWING_TYPES, index=DRAWING_TYPES.index(st.session_state.meta["drawing_type"]))

        st.session_state.meta["site_address"] = st.text_input(
            "Site Address",
            st.session_state.meta["site_address"],
            help="Must match the design filename/title. Ignored if the literal ', 0 ,' appears."
        )

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
        # Dynamic mandatory fields (MIMO required unless Power Resilience)
        required = MANDATORY_FIELDS_BASE.copy()
        if st.session_state.meta["project"] != "Power Resilience":
            # MIMO required for non-Power Resilience projects
            if not str(st.session_state.meta.get("mimo_config","")).strip():
                required = required + ["mimo_config"]

        missing = [k for k in required if not str(st.session_state.meta.get(k,"")).strip()]

        if not up:
            st.error("Please upload a PDF to audit.")
        elif missing:
            nice = {
                "client":"Client","project":"Project","site_type":"Site Type","vendor":"Proposed Vendor",
                "cabinet_loc":"Proposed Cabinet Location","radio_loc":"Proposed Radio Location","sectors":"Quantity of Sectors",
                "supplier":"Supplier","drawing_type":"Drawing Type","site_address":"Site Address","mimo_config":"Proposed MIMO Config"
            }
            st.error("Please complete all required fields: " + ", ".join(nice.get(m, m) for m in missing) + ".")
        else:
            # Load rules from disk (reflect Settings changes)
            rules = load_rules(rules_path_input)

            # Get PDF text
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

            st.markdown("#### Results")
            st.write(f"**Outcome:** {outcome} — **Major:** {majors}  •  **Minor:** {minors}  •  **RFT:** {rft_percent:.1f}%")

            df = pd.DataFrame(findings or [])
            st.dataframe(df, use_container_width=True)

            # Exports with stamped filenames
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
