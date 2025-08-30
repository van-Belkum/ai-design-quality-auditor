# app.py
# AI Design Quality Auditor – streamlined professional UI
# - Logo top-left (auto-sizes)
# - Audit tab: single-run with complete metadata
# - Per-sector MIMO dropdowns always visible (optional for Power Resilience)
# - "Use S1 for all sectors" copies S1 across
# - History & Analytics with exclude toggle
# - Settings with live YAML editor and allow/deny learning controls
# ---------------------------------------------------------------

import io
import os
import re
import json
import base64
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import yaml

# Light deps for spell/ocr/pdf – keep your existing requirements.txt in place
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
from rapidfuzz.distance import Levenshtein
import fitz  # PyMuPDF

# --------- CONFIG ---------
APP_TITLE = "AI Design Quality Auditor"
HISTORY_CSV = "audit_history.csv"
DEFAULT_RULES_FILE = "rules_example.yaml"
EXPORT_DIR = "exports"    # nightly dumps etc.
ANNOTATIONS_DIR = "annotations"
LOGO_FILE_CANDIDATES = ["logo.png", "logo.jpg", "logo.jpeg", "logo.svg"]  # default search
# You can also set st.secrets["LOGO_FILE"] to force a name

# Password gates
ENTRY_PASSWORD = "Seker123"        # entering app
RULES_PASSWORD = "vanB3lkum21"     # editing YAML in Settings

# --------- UTILS ---------
def ensure_dirs():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

def load_rules(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return {}
            return data
        except yaml.YAMLError:
            # fall back empty if malformed
            return {}

def save_rules(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def now_utc_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def parse_address_from_title(text: str) -> Optional[str]:
    # Very light extractor example (customize for your title patterns)
    # Look for a line in ALL CAPS with commas (typical title block)
    lines = [ln.strip() for ln in text.splitlines()]
    caps = [l for l in lines if l.isupper() and "," in l]
    return caps[0] if caps else None

def site_address_matches(title_addr: str, input_addr: str) -> bool:
    if not title_addr or not input_addr:
        return True  # don't hard-fail if missing – other checks can handle missing metadata
    # Ignore ", 0 ," segment
    cleaned = re.sub(r"\s*,\s*0\s*,\s*", ", ", input_addr, flags=re.IGNORECASE)
    # Normalize: collapse spaces, uppercase
    norm_in = re.sub(r"\s+", " ", cleaned).strip().upper()
    norm_tt = re.sub(r"\s+", " ", title_addr).strip().upper()
    # allow fuzzy match threshold
    score = rf_fuzz.token_set_ratio(norm_in, norm_tt)
    return score >= 90

def to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()

def load_logo_b64() -> Optional[str]:
    # priority: explicit session setting -> secrets -> file candidates
    explicit = st.session_state.get("logo_path") or st.secrets.get("LOGO_FILE", None)
    cand = []
    if explicit:
        cand.append(explicit)
    cand += LOGO_FILE_CANDIDATES
    for name in cand:
        if os.path.exists(name):
            try:
                with open(name, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            except Exception:
                continue
    return None

def annotate_pdf(pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    # Draw simple boxes and notes if bbox present (page, x0, y0, x1, y1)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for f in findings:
        bbox = f.get("bbox")
        page_idx = f.get("page_index", 0)
        note = f.get("finding", "Issue")
        sev = f.get("severity", "minor")
        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                page = doc[page_idx]
                rect = fitz.Rect(*bbox)
                page.add_rect_annot(rect)
                # Add a callout as a text annotation near the box
                text_rect = fitz.Rect(rect.x0, max(0, rect.y0-30), rect.x1+200, rect.y0)
                page.add_freetext_annot(text_rect, f"{sev.upper()}: {note}",
                                        fontsize=8, rotate=0, fill_opacity=0.0)
            except Exception:
                continue
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()

def find_text_boxes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    # very light text extraction with bbox; returns list of dicts per fragment
    out = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for pno in range(len(doc)):
        page = doc[pno]
        blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no, ...)
        for b in blocks:
            if len(b) >= 5 and b[4].strip():
                out.append({
                    "page_index": pno,
                    "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "text": b[4]
                })
    doc.close()
    return out

# --------- SPELL + SIMPLE RULE CHECKS (lightweight scaffolding) ---------
def spelling_findings(text_fragments: List[Dict[str, Any]], allow: set) -> List[Dict[str, Any]]:
    # Dummy example – leverage denylist/allowlist in rules for real usage
    # Just look for common “obvious” typos provided via rules["spelling"]["denylist"]
    findings = []
    for frag in text_fragments:
        t = frag["text"]
        words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", t)
        for w in words:
            lw = w.lower()
            if lw in allow:
                continue
            # here you could run a dictionary check; we keep it minimal (rules-driven)
    return findings

def keyword_rules_findings(text_frags: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings = []
    denylist = [d for d in rules.get("spelling", {}).get("denylist", []) if isinstance(d, str)]
    allowlist = set([a.lower() for a in rules.get("allowlist", []) if isinstance(a, str)])

    # naive scan: mark denylist tokens that are not allowlisted variants
    for frag in text_frags:
        text = frag["text"]
        up = text.upper()
        for bad in denylist:
            bad_up = bad.upper()
            if bad_up in up and bad.lower() not in allowlist:
                findings.append({
                    "page_index": frag["page_index"],
                    "bbox": frag["bbox"],
                    "finding": f"Found disallowed term: '{bad}'",
                    "severity": "minor",
                    "rule": "denylist"
                })
    # cross-field example rule: Brush vs Generator Power (example user rule)
    cross = rules.get("cross_rules", [])
    for c in cross:
        a = c.get("requires_present")
        b = c.get("forbids_with")
        if not a or not b:
            continue
        present_a = any(a.upper() in frag["text"].upper() for frag in text_frags)
        present_b = any(b.upper() in frag["text"].upper() for frag in text_frags)
        if present_a and present_b:
            findings.append({
                "page_index": 0,
                "bbox": None,
                "finding": f"'{a}' must not appear with '{b}'",
                "severity": "major",
                "rule": "cross_rules"
            })
    return findings

# --------- HISTORY ----------

HISTORY_COLS = [
    "timestamp_utc","user","client","project","site_type","vendor","cabinet_location","radio_location",
    "sectors_qty","mimo_s1","mimo_s2","mimo_s3","mimo_s4","mimo_s5","mimo_s6","mimo_same_all",
    "site_address","supplier","drawing_type","file_name","status","minor_count","major_count",
    "exclude"
]

def read_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame(columns=HISTORY_COLS)
    df = pd.read_csv(HISTORY_CSV)
    for c in HISTORY_COLS:
        if c not in df.columns:
            df[c] = None
    # ensure exclude exists and is bool
    if "exclude" in df.columns:
        df["exclude"] = df["exclude"].fillna(False).astype(bool)
    else:
        df["exclude"] = False
    return df[HISTORY_CSV and HISTORY_COLS]

def append_history(row: Dict[str, Any]):
    df = read_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_CSV, index=False)

# --------- MIMO OPTIONS (ordered + includes "(blank)") ----------

MIMO_OPTIONS = [
    "(blank)",
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

CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
CABINET_LOCS = ["Indoor", "Outdoor"]
RADIO_LOCS = ["Low Level", "High Level", "Unique Coverage", "Midway"]
SUPPLIERS = ["— Select —", "Circet", "Lynch", "IQA", "Kelly", "KN Circet", "Other"]
DRAWING_TYPES = ["— Select —", "General Arrangement", "Detailed Design"]

# --------- UI HELPERS ---------

def gate_entry():
    st.markdown("### Enter password to access the tool")
    pw = st.text_input("Password", type="password", placeholder="Enter access password")
    ok = st.button("Enter")
    if ok:
        if pw == ENTRY_PASSWORD:
            st.session_state["gate_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

def header_with_logo():
    col_logo, col_title = st.columns([1, 9], vertical_alignment="center")
    with col_logo:
        b64 = load_logo_b64()
        if b64:
            # keep sane min size
            st.markdown(
                f"""<img src="data:image/png;base64,{b64}" style="max-height:64px; max-width:140px; object-fit:contain;"/>""",
                unsafe_allow_html=True
            )
        else:
            st.caption("Upload logo as `logo.png` in repo root to display.")
    with col_title:
        st.markdown(f"## {APP_TITLE}")

def metadata_form() -> Dict[str, Any]:
    with st.container():
        st.markdown("### Audit Metadata")
        c1, c2, c3 = st.columns(3)
        client = c1.selectbox("Client", CLIENTS, index=0)
        project = c2.selectbox("Project", PROJECTS, index=0)
        site_type = c3.selectbox("Site Type", SITE_TYPES, index=0)

        c4, c5, c6 = st.columns(3)
        vendor = c4.selectbox("Proposed Vendor", VENDORS, index=0)
        cabinet_loc = c5.selectbox("Proposed Cabinet Location", CABINET_LOCS, index=0)
        radio_loc = c6.selectbox("Proposed Radio Location", RADIO_LOCS, index=0)

        c7, c8 = st.columns(2)
        supplier = c7.selectbox("Supplier", SUPPLIERS, index=0)
        drawing_type = c8.selectbox("Drawing Type", DRAWING_TYPES, index=0)

        c9, c10 = st.columns(2)
        sectors_qty = c9.selectbox("Quantity of Sectors", [1,2,3,4,5,6], index=2)
        site_address = c10.text_input("Site Address (must match title; ', 0 ,' ignored)", "")

        # MIMO – always visible; optional for Power Resilience
        st.divider()
        st.markdown("#### Proposed MIMO Config (optional for Power Resilience)")
        same_all = st.checkbox("Use S1 for all sectors", value=True, help="If checked, S1 selection is copied to all sectors.")
        mimo_vals = {}
        for i in range(1, sectors_qty+1):
            lbl = f"MIMO S{i}"
            if i == 1:
                mimo_vals[f"mimo_s{i}"] = st.selectbox(lbl, MIMO_OPTIONS, index=0, key=f"mimo_{i}")
            else:
                if same_all:
                    mimo_vals[f"mimo_s{i}"] = st.selectbox(lbl, MIMO_OPTIONS, index=MIMO_OPTIONS.index(st.session_state.get("mimo_1", "(blank)")), key=f"mimo_{i}")
                else:
                    mimo_vals[f"mimo_s{i}"] = st.selectbox(lbl, MIMO_OPTIONS, index=0, key=f"mimo_{i}")

        # Fill missing S* up to 6 with "(blank)" to keep history consistent
        for i in range(sectors_qty+1, 7):
            mimo_vals[f"mimo_s{i}"] = "(blank)"

        return {
            "client": client,
            "project": project,
            "site_type": site_type,
            "vendor": vendor,
            "cabinet_location": cabinet_loc,
            "radio_location": radio_loc,
            "sectors_qty": sectors_qty,
            "mimo_same_all": same_all,
            **mimo_vals,
            "site_address": site_address,
            "supplier": supplier,
            "drawing_type": drawing_type
        }

def validate_metadata(meta: Dict[str, Any]) -> Optional[str]:
    # Make everything mandatory except MIMO when project == Power Resilience
    missing = []
    req_fields = ["client","project","site_type","vendor","cabinet_location","radio_location",
                  "sectors_qty","site_address","supplier","drawing_type"]
    for f in req_fields:
        v = meta.get(f)
        if v in (None, "", "— Select —"):
            missing.append(f)
    # MIMO required unless Power Resilience
    if meta.get("project") != "Power Resilience":
        for i in range(1, meta.get("sectors_qty", 1)+1):
            if meta.get(f"mimo_s{i}", "(blank)") == "(blank)":
                missing.append(f"mimo_s{i}")
    if missing:
        return "Please complete required fields: " + ", ".join(missing)
    return None

# --------- APP ---------

def main():
    ensure_dirs()
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    if not st.session_state.get("gate_ok"):
        gate_entry()

    header_with_logo()

    tabs = st.tabs(["Audit", "History & Analytics", "Settings"])

    # ---------- AUDIT TAB ----------
    with tabs[0]:
        left, right = st.columns([2, 1])
        with left:
            meta = metadata_form()
            uploaded = st.file_uploader("Upload PDF to audit", type=["pdf"])
            run = st.button("Run Audit", type="primary", use_container_width=True)
            clear = st.button("Clear Metadata", use_container_width=True)

            if clear:
                for k in list(st.session_state.keys()):
                    if k.startswith("mimo_"):
                        del st.session_state[k]
                st.rerun()

            if run:
                err = validate_metadata(meta)
                if err:
                    st.error(err)
                elif not uploaded:
                    st.error("Please upload a PDF.")
                else:
                    rules_file = st.session_state.get("rules_file_path", DEFAULT_RULES_FILE)
                    rules = load_rules(rules_file)
                    pdf_bytes = uploaded.read()
                    # Extract text fragments with bbox
                    frags = find_text_boxes(pdf_bytes)

                    # Address/title match gate
                    title_addr = parse_address_from_title("\n".join([f["text"] for f in frags]))
                    addr_ok = site_address_matches(title_addr, meta.get("site_address", ""))

                    # Findings
                    findings = []
                    findings += keyword_rules_findings(frags, rules)
                    # (Keep spelling scanner hook for future; currently rules-driven)
                    allow = set([a.lower() for a in rules.get("allowlist", [])])
                    findings += spelling_findings(frags, allow)

                    # If address mismatch, add major finding
                    if not addr_ok:
                        findings.append({
                            "page_index": 0, "bbox": None,
                            "finding": "Site Address does not match document title.",
                            "severity": "major", "rule": "address_match"
                        })

                    # Summaries
                    minor = sum(1 for f in findings if f.get("severity") == "minor")
                    major = sum(1 for f in findings if f.get("severity") == "major")
                    status = "Pass" if (minor == 0 and major == 0) else "Rejected"

                    st.markdown("### Results")
                    st.metric("Status", status)
                    cA, cB = st.columns(2)
                    cA.metric("Minor", minor)
                    cB.metric("Major", major)

                    df = pd.DataFrame(findings) if findings else pd.DataFrame(columns=["page_index","bbox","finding","severity","rule"])
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Annotated PDF + Excel
                    annotated = annotate_pdf(pdf_bytes, findings) if not df.empty else pdf_bytes
                    # Save permanently to keep record visible
                    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    base = os.path.splitext(uploaded.name)[0]
                    ann_name = f"{base}_{status}_{ts}.pdf"
                    xlsx_name = f"{base}_{status}_{ts}.xlsx"
                    ann_path = os.path.join(ANNOTATIONS_DIR, ann_name)
                    xlsx_path = os.path.join(EXPORT_DIR, xlsx_name)

                    with open(ann_path, "wb") as f:
                        f.write(annotated)

                    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
                        (df if not df.empty else pd.DataFrame([{"finding":"No issues"}])).to_excel(w, index=False, sheet_name="Findings")
                        pd.DataFrame([meta]).to_excel(w, index=False, sheet_name="Metadata")

                    st.download_button("Download annotated PDF", data=annotated, file_name=ann_name, mime="application/pdf")
                    with open(xlsx_path, "rb") as f:
                        st.download_button("Download Excel report", data=f.read(), file_name=xlsx_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                    # History row
                    hist_row = {
                        "timestamp_utc": now_utc_iso(),
                        "user": "web",
                        "client": meta["client"],
                        "project": meta["project"],
                        "site_type": meta["site_type"],
                        "vendor": meta["vendor"],
                        "cabinet_location": meta["cabinet_location"],
                        "radio_location": meta["radio_location"],
                        "sectors_qty": meta["sectors_qty"],
                        "mimo_s1": meta.get("mimo_s1","(blank)"),
                        "mimo_s2": meta.get("mimo_s2","(blank)"),
                        "mimo_s3": meta.get("mimo_s3","(blank)"),
                        "mimo_s4": meta.get("mimo_s4","(blank)"),
                        "mimo_s5": meta.get("mimo_s5","(blank)"),
                        "mimo_s6": meta.get("mimo_s6","(blank)"),
                        "mimo_same_all": meta["mimo_same_all"],
                        "site_address": meta["site_address"],
                        "supplier": meta["supplier"],
                        "drawing_type": meta["drawing_type"],
                        "file_name": uploaded.name,
                        "status": status,
                        "minor_count": minor,
                        "major_count": major,
                        "exclude": False,
                    }
                    append_history(hist_row)

        with right:
            st.markdown("### Quick Help")
            st.info(
                "• MIMO selections are always visible. For **Power Resilience**, they’re optional, not required.\n"
                "• Title-to-address match is enforced (`, 0 ,` segment is ignored).\n"
                "• Mark finds Valid/Not Valid later via **History & Analytics → Learn**.\n"
                "• Use Settings to edit YAML rules (password protected)."
            )

    # ---------- HISTORY & ANALYTICS ----------
    with tabs[1]:
        st.markdown("### History & Analytics")
        dfh = read_history()
        if dfh.empty:
            st.info("No audits recorded yet.")
        else:
            # Controls
            c1, c2, c3, c4 = st.columns(4)
            client_f = c1.multiselect("Client", sorted(dfh["client"].dropna().unique().tolist()), default=None)
            supplier_f = c2.multiselect("Supplier", sorted(dfh["supplier"].dropna().unique().tolist()), default=None)
            status_f = c3.multiselect("Status", sorted(dfh["status"].dropna().unique().tolist()), default=None)
            include_ex = c4.checkbox("Include excluded runs", value=False)

            dff = dfh.copy()
            if client_f:
                dff = dff[dff["client"].isin(client_f)]
            if supplier_f:
                dff = dff[dff["supplier"].isin(supplier_f)]
            if status_f:
                dff = dff[dff["status"].isin(status_f)]
            if not include_ex:
                dff = dff[dff["exclude"] == False]

            st.dataframe(dff, use_container_width=True)

            # simple KPIs
            total = len(dff)
            passes = (dff["status"] == "Pass").sum() if total else 0
            rft = round(100.0 * passes / total, 1) if total else 0.0
            cA, cB, cC = st.columns(3)
            cA.metric("Runs", total)
            cB.metric("Passes", passes)
            cC.metric("Right First Time %", f"{rft}%")

            st.markdown("#### Exclude / Include selected runs")
            idxs = st.multiselect("Select rows to toggle exclude", dff.index.tolist(), [])
            if st.button("Toggle exclude on selected"):
                dfh.loc[idxs, "exclude"] = ~dfh.loc[idxs, "exclude"]
                dfh.to_csv(HISTORY_CSV, index=False)
                st.success("Updated.")
                st.rerun()

            st.markdown("#### Learn: mark findings as Valid / Not Valid")
            st.caption("Use the Settings → Rules editor for bulk changes. This section is kept simple to adjust allow/deny quickly.")
            # This is a placeholder for a per-token learner if you present raw findings here.

    # ---------- SETTINGS ----------
    with tabs[2]:
        st.markdown("### Settings")
        c1, c2 = st.columns([2,1])
        with c1:
            st.text_input("Rules file path", value=st.session_state.get("rules_file_path", DEFAULT_RULES_FILE), key="rules_file_path")
            st.text_input("Optional custom logo filename (e.g. `logo.png`)", value=st.session_state.get("logo_path",""), key="logo_path")
            st.caption("Place your logo file in the repo root, then set the exact name here.")

            st.markdown("#### Edit rules (YAML)")
            pw = st.text_input("Rules password", type="password", placeholder="Enter to unlock editing")
            rules_file = st.session_state.get("rules_file_path", DEFAULT_RULES_FILE)
            curr = load_rules(rules_file)
            yaml_txt = st.text_area("rules_example.yaml", value=yaml.safe_dump(curr, sort_keys=False, allow_unicode=True), height=300, disabled=(pw != RULES_PASSWORD))
            cols = st.columns(2)
            if cols[0].button("Save rules", disabled=(pw != RULES_PASSWORD)):
                try:
                    parsed = yaml.safe_load(yaml_txt) or {}
                    if not isinstance(parsed, dict):
                        st.error("Top-level YAML must be a mapping (dict).")
                    else:
                        save_rules(rules_file, parsed)
                        st.success(f"Saved rules to {rules_file}")
                except yaml.YAMLError as e:
                    st.error(f"YAML error: {e}")

            if cols[1].button("Reload from disk"):
                st.rerun()

        with c2:
            st.markdown("#### Export / Backups")
            if st.button("Export full history CSV"):
                dfh = read_history()
                csv_bytes = dfh.to_csv(index=False).encode()
                st.download_button("Download now", data=csv_bytes, file_name=f"audit_history_{dt.datetime.utcnow():%Y%m%d_%H%M%S}.csv", mime="text/csv")

            st.markdown("#### Maintenance")
            if st.button("Clear all history (keeps files)", type="secondary"):
                if os.path.exists(HISTORY_CSV):
                    pd.DataFrame(columns=HISTORY_COLS).to_csv(HISTORY_CSV, index=False)
                st.success("History cleared.")

if __name__ == "__main__":
    main()
