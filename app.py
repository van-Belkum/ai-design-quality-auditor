# app.py
import io, os, json, base64, datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import yaml
from rapidfuzz import fuzz, process
from spellchecker import SpellChecker
import fitz  # PyMuPDF

# --------------------------------------------------------------------------------------
# Constants / Defaults (YAML can override picklists)
# --------------------------------------------------------------------------------------

APP_TITLE = "AI Design Quality Auditor"
ENTRY_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"

HISTORY_DIR = "history"
HISTORY_FILE = os.path.join(HISTORY_DIR, "audits.csv")
EXPORT_DIR = "exports"      # where we persist downloadable artifacts
RULES_FILE_DEFAULT = "rules_example.yaml"

DEFAULT_PICKLISTS = {
    "suppliers": [
        "CEG",
        "CTIL",
        "Emfyser",
        "Innov8",
        "Invict",
        "KTL Team (Internal)",
        "Trylon",
    ],
    "clients": ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"],
    "projects": ["RAN", "Power Resilience", "East Unwind", "Beacon 4"],
    "drawing_types": ["General Arrangement", "Detailed Design"],
    "site_types": ["Greenfield", "Rooftop", "Streetworks"],
    "vendors": ["Ericsson", "Nokia"],
    "cabinet_locations": ["Indoor", "Outdoor"],
    "radio_locations": ["Low Level", "Midway", "High Level", "Unique Coverage"],
}

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

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def ensure_dirs():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)

def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def load_rules(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
        except Exception:
            return {}
    return data

def save_rules(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def picklists_from_rules(rules: Dict[str, Any]) -> Dict[str, List[str]]:
    ui = rules.get("ui", {})
    merged = DEFAULT_PICKLISTS.copy()
    for key, default in DEFAULT_PICKLISTS.items():
        merged[key] = ui.get(f"{key[:-1]}_options", ui.get(f"{key}_options", default))
    return merged

def read_pdf_text(pdf_bytes: bytes) -> List[str]:
    """Return text per page using PyMuPDF only (fast, no poppler required)."""
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for p in doc:
            txt = p.get_text("text")
            pages.append(txt if txt else "")
    return pages

def site_address_matches_title(site_addr: str, filename: str) -> bool:
    """Match site address words to filename title; ignore ', 0 ,' pattern."""
    if not site_addr:
        return True
    cleaned = site_addr.replace(", 0 ,", ",").replace(",0,", ",")
    # use only letters/digits
    norm = " ".join("".join(ch for ch in token if ch.isalnum()) for token in cleaned.split()).strip()
    if not norm:
        return True
    # score against filename (without extension)
    name = os.path.splitext(os.path.basename(filename))[0]
    score = fuzz.token_set_ratio(norm.lower(), name.lower())
    return score >= 70  # threshold

def status_label(findings: List[Dict[str, Any]]) -> str:
    if not findings:
        return "PASS"
    return "REJECT" if any(f.get("severity") == "major" for f in findings) else "PASS"

def make_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], src_pdf: str, status: str) -> bytes:
    df = pd.DataFrame(findings)
    with io.BytesIO() as mem:
        with pd.ExcelWriter(mem, engine="xlsxwriter") as xw:
            # Summary
            s = pd.DataFrame([{
                **meta,
                "source_pdf": src_pdf,
                "status": status,
                "timestamp_utc": now_utc_iso(),
                "total_findings": len(findings),
                "majors": int((df.severity == "major").sum()) if not df.empty else 0,
                "minors": int((df.severity == "minor").sum()) if not df.empty else 0,
            }])
            s.to_excel(xw, index=False, sheet_name="Summary")
            # Findings
            (df if not df.empty else pd.DataFrame(columns=["page","rule","severity","detail"]))
            .to_excel(xw, index=False, sheet_name="Findings")
        return mem.getvalue()

def annotate_pdf(pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    """Overlay simple red boxes and notes where text is found; fall back to sticky notes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for f in findings:
        page_idx = max(0, int(f.get("page", 1)) - 1)
        if page_idx >= len(doc):
            continue
        page = doc[page_idx]
        kw = f.get("keyword") or f.get("detail") or f.get("rule", "")
        note = f"[{f.get('severity','minor').upper()}] {f.get('rule','')}: {f.get('detail','')}"
        added = False
        if kw:
            areas = page.search_for(kw, quads=False)
            for rect in areas[:3]:
                shape = page.new_shape()
                shape.draw_rect(rect)
                shape.finish(color=(1, 0, 0), width=1)
                shape.commit()
                page.insert_textbox(
                    fitz.Rect(rect.x0, max(rect.y0-12, 0), rect.x1, rect.y0),
                    note[:150], fontsize=7, color=(1,0,0)
                )
                added = True
        if not added:
            page.add_note(page.rect.tl + (20, 20), contents=note)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

def append_history_row(row: Dict[str, Any]):
    ensure_dirs()
    header_needed = not os.path.exists(HISTORY_FILE)
    df = pd.DataFrame([row])
    df.to_csv(HISTORY_FILE, mode="a", index=False, header=header_needed)

def load_history_df() -> pd.DataFrame:
    ensure_dirs()
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_FILE, dtype=str, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

# --------------------------------------------------------------------------------------
# Rules execution
# --------------------------------------------------------------------------------------

def apply_checklist(pages: List[str], checklist: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings = []
    for rule in checklist:
        name = rule.get("name","")
        sev = str(rule.get("severity","minor")).lower()
        must = rule.get("must_contain", []) or []
        reject_if = rule.get("reject_if_present", []) or []
        page_hits = []
        for idx, txt in enumerate(pages, start=1):
            t = txt.upper()
            if any(w.upper() in t for w in reject_if):
                findings.append({"page": idx, "rule": name, "severity": sev, "detail": f"Forbidden text found: {', '.join(reject_if)}", "keyword": reject_if[0] if reject_if else ""})
            if must and not any(w.upper() in t for w in must):
                page_hits.append(idx)
        if must and page_hits:
            findings.append({"page": page_hits[0], "rule": name, "severity": sev, "detail": f"Missing required text: {', '.join(must)}", "keyword": must[0]})
    return findings

def spelling_findings(pages: List[str], allow: List[str]) -> List[Dict[str, Any]]:
    sp = SpellChecker(distance=1)
    for w in allow:
        try:
            sp.word_frequency.load_words([w.lower()])
        except Exception:
            pass
    out=[]
    for i, txt in enumerate(pages, start=1):
        words = [w.strip(".,:;()[]{}").lower() for w in txt.split()]
        miss = sp.unknown([w for w in words if len(w) > 3 and w not in set(map(str.lower, allow))])
        for w in list(miss)[:50]:
            try:
                sug = next(iter(sp.candidates(w)), None)
            except Exception:
                sug = None
            out.append({"page": i, "rule": "Spelling", "severity": "minor", "detail": f"Possible misspelling: '{w}' → '{sug}'" if sug else f"Possible misspelling: '{w}'"})
    return out

def run_checks(pages: List[str], meta: Dict[str, Any], rules: Dict[str, Any], do_spell: bool, allow_words: List[str]) -> List[Dict[str, Any]]:
    findings = []
    # site address vs file title
    if not site_address_matches_title(meta.get("site_address",""), meta.get("src_name","")):
        findings.append({
            "page": 1, "rule": "Site address vs title", "severity": "major",
            "detail": "Site address does not appear to match file title/name.",
            "keyword": ""
        })
    # checklist
    findings.extend(apply_checklist(pages, rules.get("checklist", [])))
    # spelling
    if do_spell:
        findings.extend(spelling_findings(pages, allow_words))
    # unique & clean
    for f in findings:
        f["page"] = int(f.get("page",1))
        f["severity"] = "major" if str(f.get("severity","minor")).lower() == "major" else "minor"
        f["rule"] = f.get("rule","").strip()
        f["detail"] = f.get("detail","").strip()
    return findings

# --------------------------------------------------------------------------------------
# UI Helpers
# --------------------------------------------------------------------------------------

def css_logo_top_left(logo_name: Optional[str]):
    if not logo_name:
        return
    path = os.path.join(".", logo_name)
    if not os.path.exists(path):
        st.info("⚠️ Logo file not found in repo root (png/svg/jpg). Set it in Settings → Branding.")
        return
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .brand-logo {{
            position: fixed; left: 20px; top: 14px; z-index: 9999;
        }}
        </style>
        <img class="brand-logo" src="data:image/*;base64,{b64}" width="140"/>
        """,
        unsafe_allow_html=True,
    )

def gate() -> bool:
    """Simple entry password gate with rerun-free behavior."""
    st.session_state.setdefault("gate_ok", False)
    if st.session_state["gate_ok"]:
        return True
    st.title(APP_TITLE)
    pw = st.text_input("Enter access password", type="password")
    if st.button("Enter"):
        if pw == ENTRY_PASSWORD:
            st.session_state["gate_ok"] = True
            st.success("Access granted.")
        else:
            st.error("Incorrect password.")
    return st.session_state["gate_ok"]

# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------

def audit_tab(pick: Dict[str, List[str]], rules: Dict[str, Any]):
    st.subheader("Audit Metadata (all required)")
    cols = st.columns(4)
    supplier = cols[0].selectbox("Supplier", pick["suppliers"])
    drawing = cols[1].selectbox("Drawing Type", pick["drawing_types"])
    client   = cols[2].selectbox("Client", pick["clients"])
    project  = cols[3].selectbox("Project", pick["projects"])

    cols2 = st.columns(4)
    site_type = cols2[0].selectbox("Site Type", pick["site_types"])
    vendor    = cols2[1].selectbox("Proposed Vendor", pick["vendors"])
    cab_loc   = cols2[2].selectbox("Proposed Cabinet Location", pick["cabinet_locations"])
    radio_loc = cols2[3].selectbox("Proposed Radio Location", pick["radio_locations"])

    cols3 = st.columns(2)
    n_sectors = cols3[0].selectbox("Quantity of Sectors", list(map(str, range(1,7))))
    site_addr = cols3[1].text_input("Site Address", placeholder="MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ")

    st.markdown("### Proposed MIMO Config")
    use_all = st.checkbox("Use S1 for all sectors", value=True)
    mimo_values = {}
    for i in range(1, int(n_sectors)+1):
        lbl = f"MIMO S{i}"
        if i == 1 or not use_all:
            mimo_values[lbl] = st.selectbox(lbl, MIMO_OPTIONS, index=0, key=f"mimo_{i}")
        else:
            mimo_values[lbl] = st.session_state.get("mimo_1", MIMO_OPTIONS[0])
            st.selectbox(lbl, MIMO_OPTIONS, index=MIMO_OPTIONS.index(mimo_values[lbl]), key=f"mimo_{i}", disabled=True)

    st.markdown("### Upload PDF Design")
    up = st.file_uploader("Upload PDF", type=["pdf"])
    do_spell = st.toggle("Enable spelling scan", value=False,
                         help="Runs a light dictionary check and suggests replacements. Uses a small allow-list from YAML.")
    exclude = st.checkbox("Exclude this review from analytics", value=False)

    if st.button("Run Audit", type="primary", use_container_width=True, disabled=(up is None)):
        if up is None:
            st.warning("Please upload a PDF.")
            return
        meta = {
            "supplier": supplier,
            "drawing_type": drawing,
            "client": client,
            "project": project,
            "site_type": site_type,
            "vendor": vendor,
            "cabinet_location": cab_loc,
            "radio_location": radio_loc,
            "sectors": int(n_sectors),
            "site_address": site_addr,
            "mimo": mimo_values,
            "src_name": up.name,
            "exclude": bool(exclude),
        }
        with st.status("Running checks…", expanded=True) as s:
            raw = up.read()
            pages = read_pdf_text(raw)
            allow_words = rules.get("spelling_allow", []) or []
            findings = run_checks(pages, meta, rules, do_spell, allow_words)
            status = status_label(findings)
            st.write(f"Findings: {len(findings)} | Status: **{status}**")
            s.update(label="Annotating PDF…")
            pdf_annot = annotate_pdf(raw, findings)
            # Name outputs
            base = os.path.splitext(up.name)[0]
            stamp = dt.datetime.utcnow().strftime("%Y%m%d")
            pdf_name = f"{base} - {status} - {stamp}.annotated.pdf"
            xls_name = f"{base} - {status} - {stamp}.xlsx"
            # Persist to disk for later
            ensure_dirs()
            with open(os.path.join(EXPORT_DIR, pdf_name), "wb") as f:
                f.write(pdf_annot)
            excel_bytes = make_excel(findings, meta, up.name, status)
            with open(os.path.join(EXPORT_DIR, xls_name), "wb") as f:
                f.write(excel_bytes)
            # History row
            append_history_row({
                **meta,
                "status": status,
                "timestamp_utc": now_utc_iso(),
                "pdf_name": pdf_name,
                "excel_name": xls_name,
            })
            s.update(label="Done", state="complete")

        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download annotated PDF", data=pdf_annot, file_name=pdf_name, mime="application/pdf", use_container_width=True)
        c2.download_button("⬇️ Download Excel report", data=excel_bytes, file_name=xls_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        st.success("Files also saved to `/exports` and this audit added to Analytics.")

def analytics_tab():
    st.subheader("Analytics")
    df = load_history_df()
    if df.empty:
        st.info("No audits yet.")
        return
    # Filters
    cols = st.columns(4)
    supplier = cols[0].multiselect("Supplier", sorted(df.get("supplier", pd.Series(dtype=str)).dropna().unique().tolist()))
    client   = cols[1].multiselect("Client", sorted(df.get("client", pd.Series(dtype=str)).dropna().unique().tolist()))
    project  = cols[2].multiselect("Project", sorted(df.get("project", pd.Series(dtype=str)).dropna().unique().tolist()))
    status   = cols[3].multiselect("Status", sorted(df.get("status", pd.Series(dtype=str)).dropna().unique().tolist()))

    if supplier: df = df[df["supplier"].isin(supplier)]
    if client:   df = df[df["client"].isin(client)]
    if project:  df = df[df["project"].isin(project)]
    if status:   df = df[df["status"].isin(status)]
    # Drop excluded
    if "exclude" in df.columns:
        df = df[df["exclude"].astype(str).str.lower() != "true"]

    st.metric("Total audits", len(df))
    st.metric("Right first time (PASS %)", f"{(df['status'].eq('PASS').mean()*100 if len(df) else 0):.0f}%")
    show_cols = [c for c in ["timestamp_utc","supplier","client","project","status","pdf_name","excel_name"] if c in df.columns]
    st.dataframe(df[show_cols].sort_values("timestamp_utc", ascending=False), use_container_width=True)

def training_tab(rules_path: str, rules: Dict[str, Any]):
    st.subheader("Training")
    st.caption("Upload **audited** report (Excel/JSON) to record *Valid/Not-Valid* decisions. Also add quick rules to YAML.")
    up = st.file_uploader("Upload Excel/JSON training record", type=["xlsx","xls","json"])
    decision = st.selectbox("This audit decision is…", ["Valid","Not Valid"])
    if st.button("Ingest training record", disabled=up is None):
        if up is None:
            st.warning("No file.")
        else:
            # Store as raw artifact for traceability
            ensure_dirs()
            stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(EXPORT_DIR, f"training_{stamp}_{up.name}")
            with open(path, "wb") as f:
                f.write(up.read())
            append_history_row({
                "timestamp_utc": now_utc_iso(),
                "training_artifact": os.path.basename(path),
                "training_decision": decision,
                "exclude": "true",  # training records are excluded from analytics
            })
            st.success("Training artifact saved and recorded (excluded from analytics).")

    st.markdown("---")
    st.markdown("#### Add a quick rule (appends to YAML instantly)")
    rule_name = st.text_input("Rule name", placeholder="e.g., Power Resilience note present")
    sev = st.selectbox("Severity", ["major","minor"])
    must = st.text_input("Must contain (comma-separated)", placeholder="IMPORTANT NOTE, ELTEK PSU")
    reject = st.text_input("Reject if present (comma-separated)")
    pw = st.text_input("Rules password", type="password", value="", placeholder="Enter rules password")
    if st.button("Append rule", disabled=not rule_name):
        if pw != RULES_PASSWORD:
            st.error("Wrong password.")
        else:
            data = load_rules(rules_path)
            cl = data.setdefault("checklist", [])
            cl.append({
                "name": rule_name.strip(),
                "severity": sev,
                "must_contain": [x.strip() for x in must.split(",") if x.strip()],
                "reject_if_present": [x.strip() for x in reject.split(",") if x.strip()],
            })
            save_rules(rules_path, data)
            st.success("Rule appended to YAML.")

def settings_tab(rules_path: str, rules: Dict[str, Any]):
    st.subheader("Settings")
    st.caption("Edit YAML and branding. Changes persist to repo files.")

    # Branding
    st.markdown("##### Branding")
    st.text_input("Logo file name in repo root (png/svg/jpg)", key="logo_name", value=st.session_state.get("logo_name",""))
    if st.button("Save branding"):
        st.success("Branding saved in session (logo shown top-left).")

    st.markdown("---")
    # Rules editor (password protected)
    pwd = st.text_input("Rules password", type="password")
    txt = st.text_area(os.path.basename(rules_path), value=yaml.safe_dump(rules, sort_keys=False, allow_unicode=True), height=280)
    c1, c2, c3 = st.columns([1,1,1])
    if c1.button("Save rules", type="primary"):
        if pwd != RULES_PASSWORD:
            st.error("Wrong password.")
        else:
            try:
                new = yaml.safe_load(txt) or {}
                save_rules(rules_path, new)
                st.success("Rules saved.")
            except Exception as e:
                st.error(f"YAML error: {e}")
    if c2.button("Reload from disk"):
        st.experimental_rerun()
    if c3.button("Clear all history (keeps files)"):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.success("History cleared.")

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    ensure_dirs()

    # Top-left logo
    css_logo_top_left(st.session_state.get("logo_name"))

    if not gate():
        return

    # Load rules
    rules_path = st.session_state.get("rules_path", RULES_FILE_DEFAULT)
    rules = load_rules(rules_path)

    # Picklists merged with defaults
    pick = picklists_from_rules(rules)

    st.title(APP_TITLE)

    tabs = st.tabs(["🧪 Audit", "📊 Analytics", "🧠 Training", "⚙️ Settings"])
    with tabs[0]:
        audit_tab(pick, rules)
    with tabs[1]:
        analytics_tab()
    with tabs[2]:
        training_tab(rules_path, rules)
    with tabs[3]:
        settings_tab(rules_path, rules)

if __name__ == "__main__":
    main()
