# app.py — AI Design Quality Auditor (stable build with fixes)
import io, os, re, json, base64, datetime as dt
from typing import List, Dict, Any, Optional

import streamlit as st
import yaml
import pandas as pd
from spellchecker import SpellChecker
import fitz  # PyMuPDF

# -----------------------
# Config
# -----------------------
APP_TITLE = "AI Design Quality Auditor"
ENTRY_PASSWORD = "Seker123"
YAML_ADMIN_PASSWORD = "vanB3lkum21"

HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)
HISTORY_CSV = os.path.join(HISTORY_DIR, "audit_history.csv")
DEFAULT_UI_VISIBILITY_HOURS = 24 * 14  # 14 days

DEFAULT_LOGO = None  # e.g. "88F3...png"

# Canonical dropdowns
CLIENTS   = ["BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"]
PROJECTS  = ["RAN","Power Resilience","East Unwind","Beacon 4"]
SITE_TYPE = ["Greenfield","Rooftop","Streetworks"]
VENDORS   = ["Ericsson","Nokia"]
CAB_LOCS  = ["Indoor","Outdoor"]
RAD_LOCS  = ["Low Level","Midway","High Level","Unique Coverage"]
SECTORS_QTY = [1,2,3,4,5,6]

# *** YOUR EXACT SUPPLIER LIST (as requested) ***
SUPPLIERS = [
    "CEG",
    "CTIL",
    "Emfyser",
    "Innov8",
    "Invict",
    "KTL Team (Internal)",
    "Trylon",
]

# MIMO options (dedup + ordered)
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
]
MIMO_OPTIONS = list(dict.fromkeys(MIMO_OPTIONS))

# -----------------------
# Helpers
# -----------------------
def b64_of_file(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

def gate() -> bool:
    st.session_state.setdefault("gate_ok", False)
    if st.session_state["gate_ok"]:
        return True
    st.title(APP_TITLE)
    pw = st.text_input("Enter access password", type="password")
    if st.button("Enter"):
        if pw == ENTRY_PASSWORD:
            st.session_state["gate_ok"] = True
        else:
            st.error("Incorrect password")
    return st.session_state["gate_ok"]

def load_rules(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_rules(path: str, data: Dict[str, Any]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    except Exception as e:
        st.error(f"Failed to save rules: {e}")

def text_by_page(pdf_bytes: bytes) -> List[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return [doc.load_page(i).get_text("text") for i in range(len(doc))]

def normalize_words(s: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']+", s)

def spelling_findings(pages: List[str], allow_words: set) -> List[Dict[str, Any]]:
    sp = SpellChecker()
    finds: List[Dict[str, Any]] = []
    for idx, txt in enumerate(pages, start=1):
        for w in normalize_words(txt):
            wl = w.lower()
            if wl in allow_words or len(wl) < 3:
                continue
            if wl in sp:
                continue
            try:
                sug = next(iter(sp.candidates(wl)), None)
            except Exception:
                sug = None
            finds.append({
                "page": idx,
                "pattern": w,
                "message": f"Possible misspelling: '{w}'" + (f" -> '{sug}'" if sug else ""),
                "severity": "minor",
                "rule_id": "SPELLING",
            })
    return finds

def pair_rules_findings(pages: List[str], pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    finds: List[Dict[str, Any]] = []
    for i, page_txt in enumerate(pages, start=1):
        low = page_txt.lower()
        for pr in pairs or []:
            a = (pr.get("a") or "").lower()
            b = (pr.get("b") or "").lower()
            mode = pr.get("mode","forbid")
            msg = pr.get("message") or ""
            if not a or not b:
                continue
            if mode == "forbid":
                if a in low and b in low:
                    finds.append({
                        "page": i, "pattern": f"{a} & {b}",
                        "message": msg or f"'{a}' must not appear with '{b}'.",
                        "severity": pr.get("severity","major"),
                        "rule_id": pr.get("id","PAIR_FORBID")
                    })
            else:
                if a in low and b not in low:
                    finds.append({
                        "page": i, "pattern": a,
                        "message": msg or f"'{a}' requires '{b}' also to be present.",
                        "severity": pr.get("severity","major"),
                        "rule_id": pr.get("id","PAIR_REQUIRE")
                    })
    return finds

def metadata_rules_findings(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    address = (meta.get("site_address") or "").strip()
    title    = (meta.get("pdf_name") or "").strip()
    if address and ", 0 ," not in address and title:
        if address.replace(" ","").lower() not in title.replace(" ","").lower():
            out.append({
                "page": 1,
                "pattern": address,
                "message": "Site Address does not appear in the PDF filename/title.",
                "severity": "major",
                "rule_id": "ADDR_FILENAME_MATCH"
            })
    if meta.get("project") == "Power Resilience":
        if any(meta.get(f"mimo_s{i}") for i in range(1,7)):
            out.append({
                "page": 1,
                "pattern": "MIMO",
                "message": "MIMO Config is not applicable for Power Resilience projects.",
                "severity": "minor",
                "rule_id": "MIMO_NOT_APPLICABLE"
            })
    return out

def annotate_pdf(pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for f in findings:
        page_idx = max(0, int((f.get("page") or 1) - 1))
        if page_idx >= len(doc):
            continue
        page = doc.load_page(page_idx)
        msg  = f"[{(f.get('severity') or 'minor').upper()}] {f.get('message','Finding')}"
        placed = False

        bbox = f.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                r = fitz.Rect(*bbox)
                ra = page.add_rect_annot(r)
                ra.set_colors(stroke=(1,0,0)); ra.set_border(width=1); ra.update()
                page.add_text_annot(r.br + (3,3), msg)
                placed = True
            except Exception:
                placed = False

        if not placed:
            token = (f.get("snippet") or f.get("pattern") or "").strip()
            if token:
                try:
                    for r in page.search_for(token)[:3]:
                        ra = page.add_rect_annot(r)
                        ra.set_colors(stroke=(1,0,0)); ra.set_border(width=1); ra.update()
                        page.add_text_annot(r.br + (3,3), msg)
                        placed = True
                except Exception:
                    placed = False

        if not placed:
            try:
                page.add_text_annot(fitz.Point(36, 36), msg)
            except Exception:
                pass
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

def make_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], original_name: str, status: str) -> bytes:
    df = pd.DataFrame(findings or [])
    meta_row = {**meta, "status": status, "pdf_name": original_name}
    mem = io.BytesIO()
    with pd.ExcelWriter(mem, engine="openpyxl") as xw:
        cols = ["page","rule_id","severity","pattern","message"]
        if not df.empty:
            df = df[[c for c in cols if c in df.columns]]
        else:
            df = pd.DataFrame(columns=cols)
        df.to_excel(xw, index=False, sheet_name="Findings")
        pd.DataFrame([meta_row]).to_excel(xw, index=False, sheet_name="Metadata")
    mem.seek(0)
    return mem.read()

def persist_history(record: Dict[str, Any]) -> None:
    record["timestamp_utc"] = dt.datetime.utcnow().isoformat()
    record["exclude"] = bool(record.get("exclude", False))
    try:
        if os.path.exists(HISTORY_CSV):
            df = pd.read_csv(HISTORY_CSV)
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        else:
            df = pd.DataFrame([record])
        df.to_csv(HISTORY_CSV, index=False)
    except Exception as e:
        st.warning(f"Could not write history: {e}")

def load_history() -> pd.DataFrame:
    cols = ["timestamp_utc","supplier","client","project","status","pdf_name","excel_name","exclude"]
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(HISTORY_CSV)
        # Ensure all expected columns exist
        for c in cols:
            if c not in df.columns:
                df[c] = None if c != "exclude" else False
        # Coerce types
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        if "exclude" in df.columns:
            df["exclude"] = df["exclude"].fillna(False).astype(bool)
        return df[cols]
    except Exception as e:
        st.warning(f"History read issue: {e}")
        return pd.DataFrame(columns=cols)

def run_checks(pages: List[str], meta: Dict[str, Any], rules: Dict[str, Any], do_spell: bool, allow_words: set) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    for pidx, txt in enumerate(pages, start=1):
        low = txt.lower()
        for r in (rules.get("required_phrases") or []):
            phrase = (r.get("text") or "").lower().strip()
            if phrase and phrase not in low:
                findings.append({
                    "page": pidx, "pattern": phrase,
                    "message": f"Missing required phrase: {phrase}",
                    "severity": r.get("severity","major"),
                    "rule_id": r.get("id","REQUIRED")
                })
        for r in (rules.get("forbidden_phrases") or []):
            phrase = (r.get("text") or "").lower().strip()
            if phrase and phrase in low:
                findings.append({
                    "page": pidx, "pattern": phrase,
                    "message": f"Forbidden phrase found: {phrase}",
                    "severity": r.get("severity","major"),
                    "rule_id": r.get("id","FORBIDDEN")
                })

    findings.extend(pair_rules_findings(pages, rules.get("pair_rules") or []))
    findings.extend(metadata_rules_findings(meta))

    if do_spell:
        findings.extend(spelling_findings(pages, allow_words))

    return findings

# -----------------------
# UI helpers
# -----------------------
def top_logo():
    logo_path = st.session_state.get("logo_path") or DEFAULT_LOGO
    b64 = b64_of_file(logo_path) if logo_path else None
    if b64:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                <img src="data:image/png;base64,{b64}" style="height:46px;border-radius:6px;" />
                <div style="font-weight:600;font-size:18px;">{APP_TITLE}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"### {APP_TITLE}")

def audit_tab():
    st.subheader("Audit")
    with st.container():
        colm = st.columns([1,1,1,1])
        supplier = colm[0].selectbox("Supplier (for analytics)", SUPPLIERS, index=0)
        client   = colm[1].selectbox("Client", CLIENTS, index=0)
        project  = colm[2].selectbox("Project", PROJECTS, index=0)
        site_tp  = colm[3].selectbox("Site Type", SITE_TYPE, index=0)

        col2 = st.columns([1,1,1,1])
        vendor   = col2[0].selectbox("Proposed Vendor", VENDORS, index=0)
        cab_loc  = col2[1].selectbox("Proposed Cabinet Location", CAB_LOCS, index=0)
        rad_loc  = col2[2].selectbox("Proposed Radio Location", RAD_LOCS, index=0)
        n_sect   = col2[3].selectbox("Quantity of Sectors", SECTORS_QTY, index=2)

        address  = st.text_input("Site Address (if contains ', 0 ,' we ignore filename check)")
        mimo_hide = (project == "Power Resilience")
        if not mimo_hide:
            st.markdown("**Proposed MIMO Config per Sector**")
            same_all = st.checkbox("Use S1 MIMO for all sectors", value=True)
            mimo_s1 = st.selectbox("S1 MIMO", MIMO_OPTIONS, index=0, key="mimo_s1")
            mimo_vals = {"mimo_s1": mimo_s1}
            for i in range(2, n_sect+1):
                if same_all:
                    mimo_vals[f"mimo_s{i}"] = mimo_s1
                else:
                    mimo_vals[f"mimo_s{i}"] = st.selectbox(f"S{i} MIMO", MIMO_OPTIONS, index=0, key=f"mimo_s{i}")
        else:
            mimo_vals = {}

        up = st.file_uploader("Upload PDF", type=["pdf"])
        do_spell = st.checkbox("Run spelling checks", value=True)
        exclude_hist = st.checkbox("Exclude this review from analytics")
        rules_file = st.text_input("Rules file (YAML)", value="rules_example.yaml")

        if st.button("Run Audit", type="primary", use_container_width=True) and up:
            raw = up.read()
            pages = text_by_page(raw)
            rules = load_rules(rules_file)
            allow = set([w.lower() for w in (rules.get("allowlist") or [])])

            meta = {
                "supplier": supplier, "client": client, "project": project,
                "site_type": site_tp, "vendor": vendor, "cabinet_location": cab_loc,
                "radio_location": rad_loc, "sectors": n_sect, "site_address": address,
                "pdf_name": up.name,
            }
            meta.update(mimo_vals)

            findings = run_checks(pages, meta, rules, do_spell, allow)
            status = "Pass" if not any(f.get("severity","minor") in ("major","critical") for f in findings) else "Rejected"

            excel_bytes = make_excel(findings, meta, up.name, status)
            pdf_annot = annotate_pdf(raw, findings) if findings else raw

            st.session_state["last_audit"] = {
                "meta": meta, "findings": findings, "status": status,
                "annot_pdf_bytes": pdf_annot, "excel_bytes": excel_bytes,
                "pdf_name": up.name,
                "excel_name": f"{os.path.splitext(up.name)[0]} - {status} - {dt.datetime.utcnow().strftime('%Y%m%d')}.xlsx",
                "exclude": bool(exclude_hist),
            }

            persist_history({
                "supplier": supplier, "client": client, "project": project,
                "status": status, "pdf_name": up.name,
                "excel_name": f"{os.path.splitext(up.name)[0]} - {status}.xlsx",
                "exclude": bool(exclude_hist),
            })

        last = st.session_state.get("last_audit")
        if last:
            st.success(f"Status: **{last['status']}**")
            df = pd.DataFrame(last["findings"] or [])
            st.dataframe(df if not df.empty else pd.DataFrame({"info":["No findings"]}),
                         use_container_width=True, height=260)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button("Download Excel Report",
                                   data=last["excel_bytes"],
                                   file_name=last["excel_name"],
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with c2:
                st.download_button("Download Annotated PDF",
                                   data=last["annot_pdf_bytes"],
                                   file_name=f"{os.path.splitext(last['pdf_name'])[0]} - ANNOTATED.pdf",
                                   mime="application/pdf")
            with c3:
                if st.button("Clear last audit"):
                    st.session_state.pop("last_audit", None)

def training_tab():
    st.subheader("Training (Rapid Learn)")
    rules_file = st.text_input("Rules file to update", value="rules_example.yaml", key="train_rules_file")
    up = st.file_uploader("Upload Excel exported by this tool (add a 'valid' column in Findings)", type=["xlsx"], key="train_upl")
    add_rule_text = st.text_input("Add a single forbidden phrase quickly")

    if up and st.button("Ingest Corrections"):
        rules = load_rules(rules_file)
        try:
            xls = pd.ExcelFile(up)
            fdf = pd.read_excel(xls, "Findings")
        except Exception as e:
            st.error(f"Could not read Excel: {e}")
            return

        if "valid" not in fdf.columns:
            st.info("Add a 'valid' column in the Findings sheet (Valid / Not Valid). No changes applied.")
            return

        allow = set([w.lower() for w in (rules.get("allowlist") or [])])
        forb = rules.get("forbidden_phrases") or []
        added_f = 0; added_a = 0
        for _, row in fdf.iterrows():
            pattern = str(row.get("pattern") or "").strip()
            valid = str(row.get("valid") or "").strip().lower()
            if not pattern:
                continue
            if valid == "valid":
                if pattern.lower() not in allow:
                    allow.add(pattern.lower()); added_a += 1
            elif valid in ("not valid","invalid","reject"):
                if not any((d.get("text") or "").lower() == pattern.lower() for d in forb):
                    forb.append({"id": f"FORBID_{len(forb)+1}", "text": pattern, "severity":"major"}); added_f += 1
        rules["allowlist"] = sorted(list(allow))
        rules["forbidden_phrases"] = forb
        save_rules(rules_file, rules)
        st.success(f"Updated rules: +{added_f} forbidden, +{added_a} allowlist")

    if add_rule_text and st.button("Add forbidden phrase"):
        rules = load_rules(rules_file)
        forb = rules.get("forbidden_phrases") or []
        if not any((d.get("text") or "").lower() == add_rule_text.lower() for d in forb):
            forb.append({"id": f"FORBID_{len(forb)+1}", "text": add_rule_text, "severity":"major"})
            rules["forbidden_phrases"] = forb
            save_rules(rules_file, rules)
            st.success(f"Added forbidden phrase: {add_rule_text}")
        else:
            st.info("Already present.")

def analytics_tab():
    st.subheader("Analytics")
    df = load_history()
    if df.empty:
        st.info("No history yet.")
        return

    col = st.columns(3)
    sel_client = col[0].multiselect("Client", sorted(df["client"].dropna().unique().tolist()))
    sel_project = col[1].multiselect("Project", sorted(df["project"].dropna().unique().tolist()))
    sel_supplier = col[2].multiselect("Supplier", sorted(df["supplier"].dropna().unique().tolist()))

    show = df.copy()
    if sel_client:  show = show[show["client"].isin(sel_client)]
    if sel_project: show = show[show["project"].isin(sel_project)]
    if sel_supplier:show = show[show["supplier"].isin(sel_supplier)]

    # *** FIX: ensure boolean then invert ***
    if "exclude" in show.columns:
        mask = ~show["exclude"].fillna(False).astype(bool)
        show = show[mask]

    total = len(show)
    rejected = int((show["status"] == "Rejected").sum()) if "status" in show.columns else 0
    passed = total - rejected
    rft = (passed / total * 100.0) if total else 0.0

    c = st.columns(3)
    c[0].metric("Total Audits", total)
    c[1].metric("Rejected", rejected)
    c[2].metric("Right First Time %", f"{rft:.1f}%")

    cols = [c for c in ["timestamp_utc","supplier","client","project","status","pdf_name","excel_name"] if c in show.columns]
    st.dataframe(show[cols].sort_values("timestamp_utc", ascending=False), use_container_width=True, height=320)

def settings_tab():
    st.subheader("Settings")
    # Logo
    col1, col2 = st.columns([1,1])
    with col1:
        st.caption("Logo")
        logo_up = st.file_uploader("Upload logo (png/jpg/svg)", type=["png","jpg","jpeg","svg"], key="logo_upl")
        if logo_up and st.button("Set as logo"):
            path = os.path.join(".", f"logo_{int(dt.datetime.utcnow().timestamp())}.png")
            with open(path, "wb") as f:
                f.write(logo_up.read())
            st.session_state["logo_path"] = path
            st.success("Logo set.")
    with col2:
        st.caption("UI retention")
        hrs = st.number_input("Keep results visible in UI (hours)", min_value=1, max_value=24*30, value=DEFAULT_UI_VISIBILITY_HOURS)
        st.session_state["ui_visibility_hours"] = int(hrs)

    st.divider()
    st.caption("Rules YAML editor (admin)")
    admin_pw = st.text_input("Admin password", type="password")
    rules_path = st.text_input("Rules file", value="rules_example.yaml", key="settings_rules")
    if admin_pw == YAML_ADMIN_PASSWORD:
        curr = load_rules(rules_path)
        raw = st.text_area("rules_example.yaml", value=yaml.safe_dump(curr, sort_keys=False, allow_unicode=True), height=280)
        if st.button("Save rules"):
            try:
                new_obj = yaml.safe_load(raw) or {}
                save_rules(rules_path, new_obj)
                st.success("Rules saved.")
            except Exception as e:
                st.error(f"Invalid YAML: {e}")
    else:
        st.info("Enter admin password to edit rules.")

# -----------------------
# Main
# -----------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")
    if not gate():
        st.stop()

    # header with logo
    logo_path = st.session_state.get("logo_path") or DEFAULT_LOGO
    if logo_path and b64_of_file(logo_path):
        top_logo()
    else:
        st.markdown(f"### {APP_TITLE}")

    tab_a, tab_t, tab_an, tab_s = st.tabs(["🔎 Audit", "🧠 Training", "📊 Analytics", "⚙️ Settings"])
    with tab_a:
        audit_tab()
    with tab_t:
        training_tab()
    with tab_an:
        analytics_tab()
    with tab_s:
        settings_tab()

if __name__ == "__main__":
    main()
