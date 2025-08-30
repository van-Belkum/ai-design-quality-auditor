# app.py
# AI Design Quality Auditor – High-volume workflow + rapid learning
# Tabs: Audit | History & Analytics | Settings
# Features:
# - Logo top-left
# - Full metadata (Supplier, Drawing Type included)
# - Per-sector MIMO (always visible; optional if Power Resilience)
# - One-click ZIP download (PDF+Excel)
# - Results: mark False Positives -> scoped allowlist
# - Quick Rule Builder (forbid/require) with scoping (Global or current metadata)
# - Rules YAML supports "global" + "scopes" merged per audit
# - History: bulk ZIP export for filtered runs; exclude toggle

import io, os, re, json, base64, zipfile
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import yaml
from rapidfuzz import fuzz as rf_fuzz
import fitz  # PyMuPDF

# ----------------- CONFIG -----------------
APP_TITLE = "AI Design Quality Auditor"
HISTORY_CSV = "audit_history.csv"
DEFAULT_RULES_FILE = "rules_example.yaml"
EXPORT_DIR = "exports"        # Excel reports
ANNOTATIONS_DIR = "annotations"  # Annotated PDFs
LOGO_FILE_CANDIDATES = ["logo.png","logo.jpg","logo.jpeg","logo.svg"]

# Passwords
ENTRY_PASSWORD = "Seker123"        # gate to open the app
RULES_PASSWORD = "vanB3lkum21"     # edit YAML in Settings

# ----------------- OPTIONS -----------------
CLIENTS = ["BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"]
PROJECTS = ["RAN","Power Resilience","East Unwind","Beacon 4"]
SITE_TYPES = ["Greenfield","Rooftop","Streetworks"]
VENDORS = ["Ericsson","Nokia"]
CABINET_LOCS = ["Indoor","Outdoor"]
RADIO_LOCS = ["Low Level","High Level","Unique Coverage","Midway"]
SUPPLIERS = ["CEG","CTIL","Emfyser","Innov8","Invict","KTL Team (Internal)","Trylon"]
DRAWING_TYPES = ["General Arrangement","Detailed Design"]

# Ordered + "(blank)"
MIMO_OPTIONS = [
    "(blank)",
    "18 @2x2","18 @2x2; 26 @4x4","18 @2x2; 70\\80 @2x2","18 @2x2; 80 @2x2",
    "18\\21 @2x2","18\\21 @2x2; 26 @4x4","18\\21 @2x2; 3500 @32x32","18\\21 @2x2; 70\\80 @2x2","18\\21 @2x2; 80 @2x2",
    "18\\21 @4x4","18\\21 @4x4; 3500 @32x32","18\\21 @4x4; 70 @2x4","18\\21 @4x4; 70\\80 @2x2","18\\21 @4x4; 70\\80 @2x2; 3500 @32x32",
    "18\\21 @4x4; 70\\80 @2x4","18\\21 @4x4; 70\\80 @2x4; 3500 @32x32","18\\21 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21@4x4; 70\\80 @2x2","18\\21@4x4; 70\\80 @2x4",
    "18\\21\\26 @2x2","18\\21\\26 @2x2; 3500 @32x32","18\\21\\26 @2x2; 3500 @8X8","18\\21\\26 @2x2; 70\\80 @2x2",
    "18\\21\\26 @2x2; 70\\80 @2x2; 3500 @32x32","18\\21\\26 @2x2; 70\\80 @2x2; 3500 @8x8","18\\21\\26 @2x2; 70\\80 @2x4; 3500 @32x32",
    "18\\21\\26 @2x2; 80 @2x2","18\\21\\26 @2x2; 80 @2x2; 3500 @8x8",
    "18\\21\\26 @4x4","18\\21\\26 @4x4; 3500 @32x32","18\\21\\26 @4x4; 3500 @8x8","18\\21\\26 @4x4; 70 @2x4; 3500 @8x8",
    "18\\21\\26 @4x4; 70\\80 @2x2","18\\21\\26 @4x4; 70\\80 @2x2; 3500 @32x32","18\\21\\26 @4x4; 70\\80 @2x2; 3500 @8x8",
    "18\\21\\26 @4x4; 70\\80 @2x4","18\\21\\26 @4x4; 70\\80 @2x4; 3500 @32x32","18\\21\\26 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21\\26 @4x4; 80 @2x2","18\\21\\26 @4x4; 80 @2x2; 3500 @32x32","18\\21\\26 @4x4; 80 @2x4","18\\21\\26 @4x4; 80 @2x4; 3500 @8x8",
    "18\\26 @2x2","18\\26 @4x4; 21 @2x2; 80 @2x2"
]

# ----------------- FILES & RULES -----------------
def ensure_dirs():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

def load_rules(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"allowlist": [], "spelling": {"denylist": []}, "checklist": [], "scopes": {"*":{}}}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # normalize keys we use
    data.setdefault("allowlist", [])
    data.setdefault("spelling", {}).setdefault("denylist", [])
    data.setdefault("checklist", [])
    data.setdefault("scopes", {"*": {}})
    if "*" not in data["scopes"]:
        data["scopes"]["*"] = {}
    for k in ["allowlist","denylist","require"]:
        data["scopes"]["*"].setdefault(k, [])
    return data

def save_rules(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def scope_key(meta: Dict[str, Any]) -> str:
    return "|".join([
        meta.get("client",""),
        meta.get("project",""),
        meta.get("vendor",""),
        meta.get("supplier",""),
        meta.get("drawing_type",""),
        meta.get("site_type",""),
    ])

def merged_scope_rules(rules: Dict[str,Any], meta: Dict[str,Any]) -> Dict[str,List[str]]:
    out = {"allowlist": [], "denylist": [], "require": []}
    g = rules.get("scopes",{}).get("*", {})
    out["allowlist"] += g.get("allowlist", [])
    out["denylist"]  += g.get("denylist", [])
    out["require"]   += g.get("require", [])
    s = rules.get("scopes",{}).get(scope_key(meta), {})
    out["allowlist"] += s.get("allowlist", [])
    out["denylist"]  += s.get("denylist", [])
    out["require"]   += s.get("require", [])
    # also include global top-level allowlist + spelling.denylist
    out["allowlist"] += [a for a in rules.get("allowlist", []) if isinstance(a,str)]
    out["denylist"]  += [d for d in rules.get("spelling",{}).get("denylist", []) if isinstance(d,str)]
    return out

# ----------------- PDF / TEXT -----------------
def to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()

def load_logo_b64() -> Optional[str]:
    name = st.session_state.get("logo_path") or st.secrets.get("LOGO_FILE")
    cand = []
    if name: cand.append(name)
    cand += LOGO_FILE_CANDIDATES
    for fn in cand:
        if os.path.exists(fn):
            try:
                return to_b64(open(fn,"rb").read())
            except Exception:
                pass
    return None

def find_text_boxes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    frags = []
    for pno in range(len(doc)):
        page = doc[pno]
        for (x0,y0,x1,y1,txt, *_rest) in page.get_text("blocks"):
            if isinstance(txt, str) and txt.strip():
                frags.append({"page_index": pno,"bbox":[float(x0),float(y0),float(x1),float(y1)],"text":txt})
    doc.close()
    return frags

def annotate_pdf(pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for f in findings:
        bbox = f.get("bbox")
        if bbox and len(bbox)==4:
            try:
                page = doc[f.get("page_index",0)]
                rect = fitz.Rect(*bbox)
                page.add_rect_annot(rect)
                note = f"{f.get('severity','minor').upper()}: {f.get('finding','Issue')}"
                text_rect = fitz.Rect(rect.x0, max(0, rect.y0-28), rect.x1+220, rect.y0)
                page.add_freetext_annot(text_rect, note, fontsize=8, fill_opacity=0.0)
            except Exception:
                pass
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()

# ----------------- CHECKS -----------------
def parse_title_address(frags: List[Dict[str,Any]]) -> Optional[str]:
    lines = []
    for f in frags:
        lines += f["text"].splitlines()
    caps = [ln.strip() for ln in lines if ln.strip().isupper() and "," in ln]
    return caps[0] if caps else None

def site_address_matches(title_addr: str, input_addr: str) -> bool:
    if not title_addr or not input_addr:
        return True
    cleaned = re.sub(r"\s*,\s*0\s*,\s*", ", ", input_addr, flags=re.IGNORECASE)
    norm_in = re.sub(r"\s+", " ", cleaned).strip().upper()
    norm_tt = re.sub(r"\s+", " ", title_addr).strip().upper()
    score = rf_fuzz.token_set_ratio(norm_in, norm_tt)
    return score >= 90

def run_checks(frags: List[Dict[str,Any]], rules: Dict[str,Any], meta: Dict[str,Any]) -> List[Dict[str,Any]]:
    scope = merged_scope_rules(rules, meta)
    allow = set(a.lower() for a in scope["allowlist"])
    deny  = [d for d in scope["denylist"] if isinstance(d,str)]
    reqs  = [r for r in scope["require"] if isinstance(r,str)]

    findings: List[Dict[str,Any]] = []

    # Require tokens present somewhere
    full_text = "\n".join([f["text"] for f in frags]).upper()
    for token in reqs:
        if token.upper() not in full_text:
            findings.append({"page_index":0,"bbox":None,"finding":f"Missing required token: {token}","severity":"major","rule":"require"})

    # Deny tokens if not allowlisted (basic)
    for frag in frags:
        up = frag["text"].upper()
        for bad in deny:
            if bad.upper() in up and bad.lower() not in allow:
                findings.append({
                    "page_index": frag["page_index"], "bbox": frag["bbox"],
                    "finding": f"Found disallowed term: {bad}", "severity":"minor","rule":"denylist"
                })

    # Address match
    title_addr = parse_title_address(frags)
    if not site_address_matches(title_addr, meta.get("site_address","")):
        findings.append({"page_index":0,"bbox":None,"finding":"Site Address does not match document title.","severity":"major","rule":"address_match"})

    return findings

# ----------------- HISTORY -----------------
HISTORY_COLS = [
    "timestamp_utc","user","client","project","site_type","vendor","cabinet_location","radio_location",
    "supplier","drawing_type","sectors_qty","mimo_same_all","mimo_s1","mimo_s2","mimo_s3","mimo_s4","mimo_s5","mimo_s6",
    "site_address","file_name","status","minor_count","major_count","exclude",
    "excel_path","pdf_path"
]

def now_utc_iso(): return dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z"

def read_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame(columns=HISTORY_COLS)
    df = pd.read_csv(HISTORY_CSV)
    for c in HISTORY_COLS:
        if c not in df.columns: df[c] = None
    if "exclude" in df.columns:
        df["exclude"] = df["exclude"].fillna(False).astype(bool)
    else:
        df["exclude"] = False
    return df[HISTORY_COLS]

def append_history(row: Dict[str,Any]):
    df = read_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_CSV, index=False)

# ----------------- UI HELPERS -----------------
def gate_entry():
    st.markdown("### Enter password to access the tool")
    pw = st.text_input("Password", type="password", placeholder="Enter access password")
    if st.button("Enter"):
        if pw == ENTRY_PASSWORD:
            st.session_state["gate_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

def header_with_logo():
    c1, c2 = st.columns([1,9], vertical_alignment="center")
    with c1:
        b64 = load_logo_b64()
        if b64:
            st.markdown(f'<img src="data:image/png;base64,{b64}" style="max-height:64px;max-width:160px;object-fit:contain;">', unsafe_allow_html=True)
        else:
            st.caption("Add logo as `logo.png` (or set in Settings).")
    with c2:
        st.markdown(f"## {APP_TITLE}")
        st.caption("Professional QA with audit trail, annotations, analytics, and ultra-fast rule updates.")

def metadata_form() -> Dict[str,Any]:
    st.markdown("### Audit Metadata (required)")
    a,b,c,d,e,f = st.columns(6)
    supplier      = a.selectbox("Supplier", SUPPLIERS)
    drawing_type  = b.selectbox("Drawing Type", DRAWING_TYPES)
    client        = c.selectbox("Client", CLIENTS)
    project       = d.selectbox("Project", PROJECTS)
    site_type     = e.selectbox("Site Type", SITE_TYPES)
    vendor        = f.selectbox("Proposed Vendor", VENDORS)

    g,h,i = st.columns(3)
    cabinet_loc   = g.selectbox("Proposed Cabinet Location", CABINET_LOCS)
    radio_loc     = h.selectbox("Proposed Radio Location", RADIO_LOCS)
    sectors_qty   = i.selectbox("Quantity of Sectors", [1,2,3,4,5,6], index=2)

    site_address  = st.text_input("Site Address (must match title; ', 0 ,' ignored)", "")

    st.markdown("#### Proposed MIMO Config (optional for Power Resilience)")
    same_all = st.checkbox("Use S1 for all sectors", value=True)
    mimo_vals = {}
    for sn in range(1, sectors_qty+1):
        key = f"mimo_s{sn}"
        if sn == 1:
            mimo_vals[key] = st.selectbox(f"MIMO S{sn}", MIMO_OPTIONS, key=key)
        else:
            idx = MIMO_OPTIONS.index(st.session_state.get("mimo_s1","(blank)")) if same_all else 0
            mimo_vals[key] = st.selectbox(f"MIMO S{sn}", MIMO_OPTIONS, index=idx, key=key)

    for sn in range(sectors_qty+1, 7):
        mimo_vals[f"mimo_s{sn}"] = "(blank)"

    return {
        "supplier": supplier, "drawing_type": drawing_type,
        "client": client, "project": project, "site_type": site_type, "vendor": vendor,
        "cabinet_location": cabinet_loc, "radio_location": radio_loc,
        "sectors_qty": sectors_qty, "mimo_same_all": same_all, **mimo_vals,
        "site_address": site_address
    }

def validate_metadata(meta: Dict[str,Any]) -> Optional[str]:
    req = ["supplier","drawing_type","client","project","site_type","vendor","cabinet_location","radio_location","sectors_qty","site_address"]
    missing = [k for k in req if not meta.get(k)]
    if meta.get("project") != "Power Resilience":
        for i in range(1, meta.get("sectors_qty",1)+1):
            if meta.get(f"mimo_s{i}","(blank)") == "(blank)":
                missing.append(f"mimo_s{i}")
    return f"Please complete: {', '.join(missing)}" if missing else None

def rules_quick_add(rules: Dict[str,Any], meta: Dict[str,Any], forbid_tokens: List[str], require_tokens: List[str], scope_choice: str, severity: str="minor") -> Dict[str,Any]:
    # severity is not used in this simple deny/require storage, but kept for future expansion
    rules = dict(rules)  # shallow copy
    rules.setdefault("scopes",{"*":{}})
    for key in ["allowlist","denylist","require"]:
        rules["scopes"].setdefault("*",{}).setdefault(key, [])
    if scope_choice == "Global":
        target = rules["scopes"]["*"]
    else:
        sk = scope_key(meta)
        rules["scopes"].setdefault(sk, {"allowlist":[], "denylist":[], "require":[]})
        target = rules["scopes"][sk]
    if forbid_tokens:
        target["denylist"] = sorted(set(target.get("denylist", []) + [t.strip() for t in forbid_tokens if t.strip()]))
    if require_tokens:
        target["require"] = sorted(set(target.get("require", []) + [t.strip() for t in require_tokens if t.strip()]))
    return rules

def mark_false_positives_to_allow(rules: Dict[str,Any], meta: Dict[str,Any], selected_findings: pd.DataFrame, scope_choice: str) -> Dict[str,Any]:
    # Parse tokens from "Found disallowed term: XXX"
    tokens = []
    for _, r in selected_findings.iterrows():
        m = re.search(r"Found disallowed term:\s*(.+)$", str(r.get("finding","")).strip())
        if m:
            tokens.append(m.group(1).strip())
    if not tokens:
        return rules
    rules = dict(rules)
    rules.setdefault("scopes",{"*":{}})
    for key in ["allowlist","denylist","require"]:
        rules["scopes"].setdefault("*",{}).setdefault(key, [])
    if scope_choice == "Global":
        target = rules["scopes"]["*"]
    else:
        sk = scope_key(meta)
        rules["scopes"].setdefault(sk, {"allowlist":[], "denylist":[], "require":[]})
        target = rules["scopes"][sk]
    target["allowlist"] = sorted(set(target.get("allowlist", []) + tokens))
    return rules

# ----------------- MAIN -----------------
def main():
    ensure_dirs()
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    if not st.session_state.get("gate_ok"): gate_entry()

    header_with_logo()
    tabs = st.tabs(["Audit","History & Analytics","Settings"])

    # ---------------- AUDIT ----------------
    with tabs[0]:
        meta = metadata_form()
        c1, c2 = st.columns([3,1])
        with c1:
            file = st.file_uploader("Upload PDF", type=["pdf"])
        with c2:
            run = st.button("Run Audit", type="primary", use_container_width=True)
            if st.button("Clear Metadata", use_container_width=True):
                for k in list(st.session_state.keys()):
                    if k.startswith("mimo_"):
                        st.session_state.pop(k)
                st.rerun()

        if run:
            err = validate_metadata(meta)
            if err:
                st.error(err)
            elif not file:
                st.error("Please upload a PDF.")
            else:
                rules_file = st.session_state.get("rules_file_path", DEFAULT_RULES_FILE)
                rules = load_rules(rules_file)
                pdf_bytes = file.read()
                frags = find_text_boxes(pdf_bytes)
                findings = run_checks(frags, rules, meta)

                minor = sum(1 for f in findings if f.get("severity")=="minor")
                major = sum(1 for f in findings if f.get("severity")=="major")
                status = "Pass" if (minor==0 and major==0) else "Rejected"

                st.subheader("Results")
                a,b,c = st.columns(3)
                a.metric("Status", status)
                b.metric("Minor", minor)
                c.metric("Major", major)

                df = pd.DataFrame(findings) if findings else pd.DataFrame(columns=["page_index","bbox","finding","severity","rule"])
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Save outputs first (so always on disk), then offer downloads + ZIP
                ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                base = os.path.splitext(file.name)[0]
                pdf_name = f"{base}_{status}_{ts}.pdf"
                xls_name = f"{base}_{status}_{ts}.xlsx"
                ann_path = os.path.join(ANNOTATIONS_DIR, pdf_name)
                xls_path = os.path.join(EXPORT_DIR, xls_name)

                annotated = annotate_pdf(pdf_bytes, findings) if not df.empty else pdf_bytes
                open(ann_path,"wb").write(annotated)

                with pd.ExcelWriter(xls_path, engine="openpyxl") as w:
                    (df if not df.empty else pd.DataFrame([{"finding":"No issues"}])).to_excel(w, index=False, sheet_name="Findings")
                    pd.DataFrame([meta]).to_excel(w, index=False, sheet_name="Metadata")

                st.download_button("Download annotated PDF", data=annotated, file_name=pdf_name, mime="application/pdf")
                st.download_button("Download Excel report", data=open(xls_path,"rb").read(),
                                   file_name=xls_name,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                # One-click bundle
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    z.writestr(pdf_name, annotated)
                    z.write(xls_path, arcname=xls_name)
                st.download_button("Download bundle (ZIP)", data=buf.getvalue(), file_name=f"{base}_{status}_{ts}.zip", mime="application/zip")

                # History row (persist)
                append_history({
                    "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds")+"Z",
                    "user":"web",
                    "client":meta["client"],"project":meta["project"],"site_type":meta["site_type"],
                    "vendor":meta["vendor"],"cabinet_location":meta["cabinet_location"],"radio_location":meta["radio_location"],
                    "supplier":meta["supplier"],"drawing_type":meta["drawing_type"],
                    "sectors_qty":meta["sectors_qty"],"mimo_same_all":meta["mimo_same_all"],
                    "mimo_s1":meta["mimo_s1"],"mimo_s2":meta["mimo_s2"],"mimo_s3":meta["mimo_s3"],
                    "mimo_s4":meta["mimo_s4"],"mimo_s5":meta["mimo_s5"],"mimo_s6":meta["mimo_s6"],
                    "site_address":meta["site_address"],"file_name":file.name,"status":status,
                    "minor_count":minor,"major_count":major,"exclude":False,
                    "excel_path":xls_path,"pdf_path":ann_path
                })

                # ---------- Rapid Learning Controls ----------
                st.markdown("### Rapid Learning")
                st.caption("Mark false positives to allow them next time (scoped), or add new forbid/require rules quickly.")
                scope_choice = st.radio("Rule scope", ["Current Metadata","Global"], horizontal=True)
                scope_label = scope_key(meta) if scope_choice=="Current Metadata" else "*"
                st.write(f"Scope: **{scope_label}**")

                if not df.empty:
                    # selection by index
                    idxs = st.multiselect("Select findings that are False Positives", df.index.tolist(), [])
                    if st.button("Mark selected as False Positive (add to allowlist)"):
                        sel = df.loc[idxs] if len(idxs) else pd.DataFrame()
                        new_rules = mark_false_positives_to_allow(rules, meta, sel, "Global" if scope_choice=="Global" else "Current")
                        save_rules(rules_file, new_rules)
                        st.success("Allowed selected tokens. Rules updated.")
                st.divider()

                st.markdown("#### Quick Rule Builder")
                colx, coly = st.columns(2)
                forbid_in = colx.text_area("Forbid tokens (comma-separated)", placeholder="AHEGC, WRONG_LABEL")
                require_in = coly.text_area("Require tokens (comma-separated)", placeholder="TITLE, SCALE")
                sev = st.selectbox("Severity (informational for now)", ["minor","major"], index=0)
                if st.button("Add rule(s)"):
                    forbid = [t.strip() for t in forbid_in.split(",")] if forbid_in else []
                    require = [t.strip() for t in require_in.split(",")] if require_in else []
                    updated = rules_quick_add(rules, meta, forbid, require, "Global" if scope_choice=="Global" else "Current", sev)
                    save_rules(rules_file, updated)
                    st.success("Rules saved.")

    # ---------------- HISTORY & ANALYTICS ----------------
    with tabs[1]:
        st.markdown("### History & Analytics")
        dfh = read_history()
        if dfh.empty:
            st.info("No audits yet.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            f_client  = c1.multiselect("Client", sorted(dfh["client"].dropna().unique().tolist()))
            f_supplier= c2.multiselect("Supplier", sorted(dfh["supplier"].dropna().unique().tolist()))
            f_status  = c3.multiselect("Status", sorted(dfh["status"].dropna().unique().tolist()))
            show_exc  = c4.checkbox("Include excluded", value=False)

            dff = dfh.copy()
            if f_client:   dff = dff[dff["client"].isin(f_client)]
            if f_supplier: dff = dff[dff["supplier"].isin(f_supplier)]
            if f_status:   dff = dff[dff["status"].isin(f_status)]
            if not show_exc: dff = dff[dff["exclude"]==False]

            st.dataframe(dff, use_container_width=True)
            total = len(dff)
            passes = int((dff["status"]=="Pass").sum()) if total else 0
            rft = round(100*passes/total,1) if total else 0.0
            k1,k2,k3 = st.columns(3)
            k1.metric("Runs", total); k2.metric("Passes", passes); k3.metric("Right First Time %", f"{rft}%")

            st.markdown("#### Exclude / Include")
            idxs = st.multiselect("Select rows to toggle exclude", dff.index.tolist(), [])
            if st.button("Toggle exclude"):
                dfh.loc[idxs,"exclude"] = ~dfh.loc[idxs,"exclude"]
                dfh.to_csv(HISTORY_CSV, index=False)
                st.success("Updated.")
                st.rerun()

            st.markdown("#### Bulk export filtered runs (ZIP)")
            if st.button("Create ZIP of filtered annotated PDFs + Excels"):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for _, r in dff.iterrows():
                        p = str(r.get("pdf_path","")); x = str(r.get("excel_path",""))
                        if p and os.path.exists(p): z.write(p, arcname=os.path.basename(p))
                        if x and os.path.exists(x): z.write(x, arcname=os.path.basename(x))
                st.download_button("Download filtered bundle", data=buf.getvalue(),
                                   file_name=f"audits_bundle_{dt.datetime.utcnow():%Y%m%d_%H%M%S}.zip",
                                   mime="application/zip")

    # ---------------- SETTINGS ----------------
    with tabs[2]:
        st.markdown("### Settings")
        c1, c2 = st.columns([2,1])
        with c1:
            st.text_input("Rules file path", value=st.session_state.get("rules_file_path", DEFAULT_RULES_FILE), key="rules_file_path")
            st.text_input("Custom logo file name (optional)", value=st.session_state.get("logo_path",""), key="logo_path")
            st.caption("Place your logo in the repo root and set its exact file name here if not using logo.png.")

            st.markdown("#### Edit rules (YAML)")
            pw = st.text_input("Rules password", type="password", placeholder="Enter to unlock")
            rules_file = st.session_state.get("rules_file_path", DEFAULT_RULES_FILE)
            curr = load_rules(rules_file)
            ta = st.text_area("rules_example.yaml", value=yaml.safe_dump(curr, sort_keys=False, allow_unicode=True),
                              height=320, disabled=(pw != RULES_PASSWORD))
            d0, d1 = st.columns(2)
            if d0.button("Save rules", disabled=(pw != RULES_PASSWORD)):
                try:
                    parsed = yaml.safe_load(ta) or {}
                    if not isinstance(parsed, dict):
                        st.error("Top-level YAML must be a mapping.")
                    else:
                        save_rules(rules_file, parsed)
                        st.success("Rules saved.")
                except yaml.YAMLError as e:
                    st.error(f"YAML error: {e}")
            if d1.button("Reload from disk"):
                st.rerun()
        with c2:
            st.markdown("#### Data")
            if st.button("Clear all history (keeps files)"):
                pd.DataFrame(columns=HISTORY_COLS).to_csv(HISTORY_CSV, index=False)
                st.success("History cleared.")

if __name__ == "__main__":
    main()
