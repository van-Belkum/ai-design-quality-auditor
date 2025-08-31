import streamlit as st
import pandas as pd
import yaml, base64, io, os, re, zipfile, datetime as dt
from typing import List, Dict, Any

# Prefer PyMuPDF for text + annotation (works on Streamlit Cloud)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # We'll degrade gracefully

# ─────────────────────────────── App Config ───────────────────────────────
ENTRY_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"
RULES_FILE = "rules_example.yaml"
HISTORY_FILE = "history/audit_history.csv"
EXPORT_DIR = "daily_exports"
LOGO_PATH = "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png"  # change if needed
os.makedirs("history", exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

SUPPLIERS = ["CEG","CTIL","Emfyser","Innov8","Invict","KTL Team (Internal)","Trylon"]
CLIENTS   = ["BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"]
PROJECTS  = ["RAN","Power Resilience","East Unwind","Beacon 4"]
SITETYPES = ["Greenfield","Rooftop","Streetworks"]
VENDORS   = ["Ericsson","Nokia"]
CABINETS  = ["Indoor","Outdoor"]
RADIOS    = ["Low Level","High Level","Unique Coverage","Midway"]
DRAWINGS  = ["General Arrangement","Detailed Design"]

MIMO_OPTIONS = [
    "18 @2x2","18 @2x2; 26 @4x4","18 @2x2; 70\\80 @2x2","18 @2x2; 80 @2x2",
    "18\\21 @2x2","18\\21 @2x2; 26 @4x4","18\\21 @2x2; 3500 @32x32",
    "18\\21 @2x2; 70\\80 @2x2","18\\21 @2x2; 80 @2x2","18\\21 @4x4",
    "18\\21 @4x4; 3500 @32x32","18\\21 @4x4; 70 @2x4","18\\21 @4x4; 70\\80 @2x2",
    "18\\21 @4x4; 70\\80 @2x2; 3500 @32x32","18\\21 @4x4; 70\\80 @2x4",
    "18\\21 @4x4; 70\\80 @2x4; 3500 @32x32","18\\21 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21@4x4; 70\\80 @2x2","18\\21@4x4; 70\\80 @2x4","18\\21\\26 @2x2",
    "18\\21\\26 @2x2; 3500 @32x32","18\\21\\26 @2x2; 3500 @8X8",
    "18\\21\\26 @2x2; 70\\80 @2x2","18\\21\\26 @2x2; 70\\80 @2x2; 3500 @32x32",
    "18\\21\\26 @2x2; 70\\80 @2x2; 3500 @8x8","18\\21\\26 @2x2; 70\\80 @2x4; 3500 @32x32",
    "18\\21\\26 @2x2; 80 @2x2","18\\21\\26 @2x2; 80 @2x2; 3500 @8x8","18\\21\\26 @4x4",
    "18\\21\\26 @4x4; 3500 @32x32","18\\21\\26 @4x4; 3500 @8x8",
    "18\\21\\26 @4x4; 70 @2x4; 3500 @8x8","18\\21\\26 @4x4; 70\\80 @2x2",
    "18\\21\\26 @4x4; 70\\80 @2x2; 3500 @32x32","18\\21\\26 @4x4; 70\\80 @2x2; 3500 @8x8",
    "18\\21\\26 @4x4; 70\\80 @2x4","18\\21\\26 @4x4; 70\\80 @2x4; 3500 @32x32",
    "18\\21\\26 @4x4; 70\\80 @2x4; 3500 @8x8","18\\21\\26 @4x4; 80 @2x2",
    "18\\21\\26 @4x4; 80 @2x2; 3500 @32x32","18\\21\\26 @4x4; 80 @2x4",
    "18\\21\\26 @4x4; 80 @2x4; 3500 @8x8","18\\26 @2x2","18\\26 @4x4; 21 @2x2; 80 @2x2",
]

# ─────────────────────────────── Utilities ───────────────────────────────
def logo_css():
    try:
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
              .seker-logo {{
                position: fixed; top: 8px; left: 12px;
                width: 170px; z-index: 9999; pointer-events:none;
              }}
              .block-container {{ padding-top: 84px !important; }}
            </style>
            <img class="seker-logo" src="data:image/png;base64,{b64}">
            """,
            unsafe_allow_html=True
        )
    except Exception:
        pass

def gate():
    pw = st.text_input("Enter access password", type="password", help="Ask admin if you don't have it.")
    if pw != ENTRY_PASSWORD:
        st.stop()

def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"checklist": [], "spelling_allow": []}
    with open(path, "r") as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            st.error(f"YAML error: {e}")
            data = {}
    data.setdefault("checklist", [])
    data.setdefault("spelling_allow", [])
    return data

def save_yaml(path: str, data: Dict[str, Any]):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def safe_read_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_FILE, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

def append_history(rows: List[Dict[str, Any]]):
    df_new = pd.DataFrame(rows)
    df_old = safe_read_history()
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all.to_csv(HISTORY_FILE, index=False)

def export_history_zip() -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        if os.path.exists(HISTORY_FILE):
            z.write(HISTORY_FILE, arcname=os.path.basename(HISTORY_FILE))
    return bio.getvalue()

def extract_text_with_fitz(pdf_bytes: bytes) -> List[str]:
    if not fitz:
        return [""]
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for p in doc:
        pages.append(p.get_text("text") or "")
    doc.close()
    return pages

def annotate_pdf_with_fitz(pdf_bytes: bytes, comments: List[Dict[str, Any]]) -> bytes:
    """Drop a sticky note near the top-left of each page for each finding."""
    if not fitz:
        return pdf_bytes
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    grouped: Dict[int, List[str]] = {}
    for f in comments:
        p = max(1, int(f.get("page", 1)))
        grouped.setdefault(p, []).append(f"{f.get('severity','minor').upper()}: {f.get('issue','')}")
    for pno, msgs in grouped.items():
        if pno-1 < 0 or pno-1 >= len(doc): 
            continue
        page = doc[pno-1]
        text = "\n".join(msgs)[:1024]
        # add a small rectangle and text
        rect = fitz.Rect(36, 36, min(page.rect.width-36, 360), 36+14*min(8, len(msgs)))
        page.draw_rect(rect, color=(1,0,0), width=0.6)
        page.insert_text((rect.x0+4, rect.y0+10), text, fontsize=8)
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()

def make_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], fname: str, status: str) -> bytes:
    df = pd.DataFrame([{
        "file": fname, "status": status,
        "supplier": meta["Supplier"], "drawing_type": meta["Drawing Type"],
        "client": meta["Client"], "project": meta["Project"],
        "site_type": meta["Site Type"], "vendor": meta["Proposed Vendor"],
        "cabinet": meta["Proposed Cabinet Location"], "radio": meta["Proposed Radio Location"],
        "sectors": meta["Quantity of Sectors"], "address": meta["Site Address"],
        "mimo": "; ".join([f"{k}:{v}" for k,v in meta["MIMO"].items()]),
        "page": f.get("page", ""), "severity": f.get("severity",""),
        "issue": f.get("issue",""), "suggestion": f.get("suggestion","")
    } for f in (findings or [{}])])
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Audit", index=False)
    return bio.getvalue()

def zip_reports(excel_bytes: bytes, pdf_bytes: bytes, base: str) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{base}.xlsx", excel_bytes)
        z.writestr(f"{base}.pdf", pdf_bytes)
    return bio.getvalue()

def status_from_findings(findings: List[Dict[str, Any]]) -> str:
    if any(f.get("severity","minor").lower()=="major" for f in findings):
        return "Rejected"
    return "Pass"

def spelling_findings(pages: List[str], allow: List[str]) -> List[Dict[str,Any]]:
    allow_set = {a.lower() for a in allow}
    out = []
    word_re = re.compile(r"[A-Za-z]{3,}")
    for i, t in enumerate(pages, start=1):
        for w in word_re.findall(t or ""):
            if w.lower() not in allow_set:
                out.append({"page": i, "issue": f"Possible spelling error: {w}", "severity": "minor"})
    return out

def apply_rules(pages: List[str], rules: Dict[str, Any], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Simple content rules; scope filtering by meta (client/project/vendor/etc.)."""
    out = []
    cl = rules.get("checklist", [])
    for r in cl:
        scope = r.get("scope", {})  # e.g. {project: ["RAN"], client: ["BTEE"]}
        def in_scope(key):
            vals = scope.get(key)
            return not vals or meta.get(key.replace("_"," ").title()) in vals
        if not all(in_scope(k) for k in ["client","project","vendor","site_type","drawing_type"]):
            continue
        name = r.get("name","Rule")
        sev = r.get("severity","minor")
        must = r.get("must_contain",[])
        reject = r.get("reject_if_present",[])
        for i, text in enumerate(pages, start=1):
            for token in must:
                if token and token not in (text or ""):
                    out.append({"page": i, "issue": f"Missing '{token}' ({name})", "severity": sev, "suggestion": f"Add '{token}'"})
            for token in reject:
                if token and token in (text or ""):
                    out.append({"page": i, "issue": f"Forbidden '{token}' ({name})", "severity": sev, "suggestion": f"Remove '{token}'"})
    return out

# ─────────────────────────────── App ───────────────────────────────
st.set_page_config(page_title="AI Design Quality Auditor", layout="wide")
gate()
logo_css()
st.title("AI Design Quality Auditor")

tab_audit, tab_train, tab_hist, tab_settings = st.tabs(["Audit", "Training", "History & Analytics", "Settings"])

# ───────────── Audit
with tab_audit:
    st.subheader("Audit Metadata (all required)")
    c1,c2,c3,c4 = st.columns(4)
    with c1: supplier = st.selectbox("Supplier", SUPPLIERS)
    with c2: drawing = st.selectbox("Drawing Type", DRAWINGS)
    with c3: client   = st.selectbox("Client", CLIENTS)
    with c4: project  = st.selectbox("Project", PROJECTS)

    c5,c6,c7,c8 = st.columns(4)
    with c5: site_type = st.selectbox("Site Type", SITETYPES)
    with c6: vendor    = st.selectbox("Proposed Vendor", VENDORS)
    with c7: cabinet   = st.selectbox("Proposed Cabinet Location", CABINETS)
    with c8: radio     = st.selectbox("Proposed Radio Location", RADIOS)

    qty = st.selectbox("Quantity of Sectors", [1,2,3,4,5,6])
    address = st.text_input("Site Address", placeholder="MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ")

    st.markdown("**Proposed MIMO Config (optional for Power Resilience only)**")
    same_for_all = st.checkbox("Use S1 for all sectors", value=True)
    mimo = {}
    if same_for_all:
        s1 = st.selectbox("MIMO S1", MIMO_OPTIONS, key="mimo_s1")
        for i in range(1, qty+1):
            mimo[f"S{i}"] = s1
    else:
        cols = st.columns(min(3, qty))
        for i in range(1, qty+1):
            col = cols[(i-1) % len(cols)]
            with col:
                mimo[f"S{i}"] = st.selectbox(f"MIMO S{i}", MIMO_OPTIONS, key=f"mimo_{i}")

    up_pdf = st.file_uploader("Upload PDF Design", type=["pdf"])
    exclude_from_stats = st.checkbox("Exclude this run from analytics", value=False)

    if up_pdf and st.button("Run Audit"):
        pdf_bytes = up_pdf.read()
        # 1) Extract text
        pages = extract_text_with_fitz(pdf_bytes)

        # 2) Load rules
        rules = load_yaml(RULES_FILE)

        # 3) Run checks
        meta = {
            "Supplier": supplier, "Drawing Type": drawing,
            "Client": client, "Project": project, "Site Type": site_type,
            "Proposed Vendor": vendor, "Proposed Cabinet Location": cabinet,
            "Proposed Radio Location": radio, "Quantity of Sectors": qty,
            "Site Address": address, "MIMO": mimo
        }

        # Title/site address rule: must appear in PDF title text; ignore ", 0 ," form per your requirement
        cleaned_addr = re.sub(r",\s*0\s*,", ",", address).strip()
        address_issue = []
        if cleaned_addr and not any(cleaned_addr in (p or "") for p in pages):
            address_issue.append({"page": 1, "issue": "Site address in metadata does not match PDF text", "severity": "major",
                                  "suggestion": "Ensure the title block matches the address exactly (ignore ', 0 ,')."})

        f_spelling = spelling_findings(pages, rules.get("spelling_allow", []))
        f_rules    = apply_rules(pages, rules, {
            "client": client, "project": project, "vendor": vendor,
            "site_type": site_type, "drawing_type": drawing
        })
        findings = address_issue + f_rules + f_spelling

        status = status_from_findings(findings)

        # 4) Reports
        excel_bytes = make_excel(findings, meta, up_pdf.name, status)
        annotated   = annotate_pdf_with_fitz(pdf_bytes, findings)

        # 5) Save to history
        ts = dt.datetime.utcnow().isoformat()
        rows = [{
            "timestamp_utc": ts, "file": up_pdf.name, "status": status,
            "issue": f.get("issue",""), "severity": f.get("severity",""),
            "supplier": supplier, "drawing_type": drawing, "client": client, "project": project,
            "site_type": site_type, "vendor": vendor, "cabinet": cabinet, "radio": radio,
            "address": address, "exclude": bool(exclude_from_stats)
        } for f in (findings or [{}])]
        append_history(rows)

        base = f"{os.path.splitext(up_pdf.name)[0]}_{status}_{dt.datetime.utcnow().strftime('%Y%m%d')}"
        zip_bytes = zip_reports(excel_bytes, annotated, base)

        st.success(f"Audit complete — **{status}**")
        st.download_button("Download Excel", excel_bytes, file_name=f"{base}.xlsx")
        st.download_button("Download Annotated PDF", annotated, file_name=f"{base}.pdf")
        st.download_button("Download Both (ZIP)", zip_bytes, file_name=f"{base}.zip")

# ───────────── Training
with tab_train:
    st.subheader("Bulk train from validation spreadsheet")
    st.caption("Upload an Excel with columns: issue, valid (TRUE/FALSE), add_rule_text (optional), severity (minor/major), scope_client, scope_project, scope_vendor, scope_site_type, scope_drawing_type")
    upload = st.file_uploader("Upload validation Excel", type=["xlsx"], key="train_xlsx")
    if upload and st.button("Process training file"):
        dfv = pd.read_excel(upload)
        rules = load_yaml(RULES_FILE)
        added = 0
        allowed = 0
        for _, r in dfv.iterrows():
            valid = str(r.get("valid","")).strip().lower() in ["true","yes","1"]
            issue = str(r.get("issue","")).strip()
            add_text = str(r.get("add_rule_text","")).strip()
            severity = str(r.get("severity","minor")).strip().lower() or "minor"
            scope = {
                "client": r.get("scope_client"),
                "project": r.get("scope_project"),
                "vendor": r.get("scope_vendor"),
                "site_type": r.get("scope_site_type"),
                "drawing_type": r.get("scope_drawing_type"),
            }
            scope = {k:[v] for k,v in scope.items() if isinstance(v,str) and v}
            if valid and add_text:
                rules["checklist"].append({
                    "name": add_text[:60],
                    "severity": "major" if severity=="major" else "minor",
                    "must_contain": [add_text],
                    "reject_if_present": [],
                    "scope": scope
                })
                added += 1
            if not valid and issue:
                # treat as false-positive word -> allowlist
                rules.setdefault("spelling_allow", [])
                if issue not in rules["spelling_allow"]:
                    rules["spelling_allow"].append(issue)
                    allowed += 1
        save_yaml(RULES_FILE, rules)
        st.success(f"Training applied. Rules added: {added}, spelling allowlist additions: {allowed}")

    st.divider()
    st.subheader("Add a single rule quickly")
    with st.form("quick_rule"):
        qr_name = st.text_input("Rule name")
        qr_sev  = st.selectbox("Severity", ["minor","major"])
        qr_must = st.text_input("Must contain (comma-separated tokens)")
        qr_rej  = st.text_input("Reject if present (comma-separated tokens)")
        qs1,qs2,qs3,qs4,qs5 = st.columns(5)
        with qs1: sc_client  = st.multiselect("Scope: Client", CLIENTS)
        with qs2: sc_proj    = st.multiselect("Scope: Project", PROJECTS)
        with qs3: sc_vendor  = st.multiselect("Scope: Vendor", VENDORS)
        with qs4: sc_site    = st.multiselect("Scope: Site Type", SITETYPES)
        with qs5: sc_draw    = st.multiselect("Scope: Drawing", DRAWINGS)
        submitted = st.form_submit_button("Add rule")
        if submitted:
            rules = load_yaml(RULES_FILE)
            rules["checklist"].append({
                "name": qr_name or "Rule",
                "severity": qr_sev,
                "must_contain": [s.strip() for s in qr_must.split(",") if s.strip()],
                "reject_if_present": [s.strip() for s in qr_rej.split(",") if s.strip()],
                "scope": {
                    "client": sc_client or None,
                    "project": sc_proj or None,
                    "vendor": sc_vendor or None,
                    "site_type": sc_site or None,
                    "drawing_type": sc_draw or None,
                }
            })
            # prune Nones
            for r in rules["checklist"]:
                r["scope"] = {k:v for k,v in (r.get("scope") or {}).items() if v}
            save_yaml(RULES_FILE, rules)
            st.success("Rule added.")

# ───────────── History & Analytics
with tab_hist:
    st.subheader("History")
    dfh = safe_read_history()
    if dfh.empty:
        st.info("No history yet.")
    else:
        # filters
        f1,f2,f3,f4 = st.columns(4)
        with f1: f_supplier = st.selectbox("Supplier filter", ["(all)"]+SUPPLIERS)
        with f2: f_client   = st.selectbox("Client filter", ["(all)"]+CLIENTS)
        with f3: f_project  = st.selectbox("Project filter", ["(all)"]+PROJECTS)
        with f4: f_status   = st.selectbox("Status filter", ["(all)","Pass","Rejected"])
        mask = pd.Series(True, index=dfh.index)
        if f_supplier != "(all)": mask &= (dfh["supplier"]==f_supplier)
        if f_client   != "(all)": mask &= (dfh["client"]==f_client)
        if f_project  != "(all)": mask &= (dfh["project"]==f_project)
        if f_status   != "(all)": mask &= (dfh["status"]==f_status)
        dfv = dfh[mask].copy()

        # toggle exclude in-line
        st.caption("Tip: toggle ‘exclude’ to remove/include rows from analytics, then click Save changes.")
        df_edit = st.data_editor(dfv, key="hist_editor", use_container_width=True, num_rows="dynamic")
        if st.button("Save changes"):
            # merge back to original by index
            dfh.loc[df_edit.index, "exclude"] = df_edit["exclude"].values
            dfh.to_csv(HISTORY_FILE, index=False)
            st.success("History updated.")

        st.divider()
        st.subheader("Analytics")
        df_use = dfh[dfh.get("exclude", False) != True].copy() if "exclude" in dfh else dfh.copy()
        if df_use.empty:
            st.info("No rows included in analytics.")
        else:
            # Right First Time %
            total = df_use["file"].nunique()
            rft   = df_use.groupby("file")["status"].max().eq("Pass").sum()
            pct   = (rft/total*100) if total else 0
            c1,c2,c3 = st.columns(3)
            c1.metric("Audits (unique files)", total)
            c2.metric("Right First Time", f"{pct:.1f}%")
            c3.metric("Rejected files", df_use.groupby("file")["status"].max().eq("Rejected").sum())

            # supplier breakdown
            st.bar_chart(df_use.groupby("supplier")["severity"].count())

        st.divider()
        st.subheader("Export history")
        st.download_button("Download history ZIP", export_history_zip(),
                           file_name=f"history_{dt.datetime.utcnow().strftime('%Y%m%d')}.zip")

# ───────────── Settings
with tab_settings:
    st.subheader("Edit rules (YAML)")
    pw = st.text_input("Rules password", type="password")
    if pw == RULES_PASSWORD:
        current = open(RULES_FILE).read() if os.path.exists(RULES_FILE) else "checklist: []\nspelling_allow: []\n"
        txt = st.text_area("rules_example.yaml", value=current, height=320)
        if st.button("Save rules"):
            try:
                yaml.safe_load(txt)  # validate
                with open(RULES_FILE,"w") as f:
                    f.write(txt)
                st.success("Rules saved.")
            except yaml.YAMLError as e:
                st.error(f"YAML error: {e}")
    else:
        st.info("Enter the rules password to edit the YAML.")
