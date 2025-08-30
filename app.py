# app.py — AI Design Quality Auditor (exclude + dual download ZIP)
# Entry password: Seker123 | Rules password: vanB3lkum21

import os, io, re, json, base64, zipfile
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple

import streamlit as st
import pandas as pd

# Optional deps (graceful fallbacks)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# ----------------------------- CONSTANTS -----------------------------
ENTRY_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"

LOGO_CANDIDATES = [
    "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png",
    "Seker.png", "logo.png", "seker-logo.png"
]

CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
DRAWING_TYPES = ["General Arrangement", "Detailed Design"]
CABINET_LOCATIONS = ["Indoor", "Outdoor"]
SUPPLIERS = ["CEG","CTIL","Emfyser","Innov8","Invict","KTL Team (Internal)","Trylon"]  # fixed; rules ignore
RADIO_LOCATIONS = ["Low Level", "High Level", "Unique Coverage", "Midway"]

MIMO_OPTIONS = [
    "18 @2x2","18 @2x2; 26 @4x4","18 @2x2; 70\\80 @2x2","18 @2x2; 80 @2x2",
    "18\\21 @2x2","18\\21 @2x2; 26 @4x4","18\\21 @2x2; 3500 @32x32","18\\21 @2x2; 70\\80 @2x2","18\\21 @2x2; 80 @2x2",
    "18\\21 @4x4","18\\21 @4x4; 3500 @32x32","18\\21 @4x4; 70 @2x4","18\\21 @4x4; 70\\80 @2x2",
    "18\\21 @4x4; 70\\80 @2x2; 3500 @32x32","18\\21 @4x4; 70\\80 @2x4",
    "18\\21 @4x4; 70\\80 @2x4; 3500 @32x32","18\\21 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21@4x4; 70\\80 @2x2","18\\21@4x4; 70\\80 @2x4",
    "18\\21\\26 @2x2","18\\21\\26 @2x2; 3500 @32x32","18\\21\\26 @2x2; 3500 @8X8",
    "18\\21\\26 @2x2; 70\\80 @2x2","18\\21\\26 @2x2; 70\\80 @2x2; 3500 @32x32","18\\21\\26 @2x2; 70\\80 @2x2; 3500 @8x8",
    "18\\21\\26 @2x2; 70\\80 @2x4; 3500 @32x32","18\\21\\26 @2x2; 80 @2x2","18\\21\\26 @2x2; 80 @2x2; 3500 @8x8",
    "18\\21\\26 @4x4","18\\21\\26 @4x4; 3500 @32x32","18\\21\\26 @4x4; 3500 @8x8",
    "18\\21\\26 @4x4; 70 @2x4; 3500 @8x8","18\\21\\26 @4x4; 70\\80 @2x2",
    "18\\21\\26 @4x4; 70\\80 @2x2; 3500 @32x32","18\\21\\26 @4x4; 70\\80 @2x2; 3500 @8x8",
    "18\\21\\26 @4x4; 70\\80 @2x4","18\\21\\26 @4x4; 70\\80 @2x4; 3500 @32x32","18\\21\\26 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21\\26 @4x4; 80 @2x2","18\\21\\26 @4x4; 80 @2x2; 3500 @32x32","18\\21\\26 @4x4; 80 @2x4","18\\21\\26 @4x4; 80 @2x4; 3500 @8x8",
    "18\\26 @2x2","18\\26 @4x4; 21 @2x2; 80 @2x2","(blank)"
]

HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)
RULES_FILE = "rules_example.yaml"
CORRECTIONS_FILE = "corrections.json"   # {"AHEGC":"AHEGG", ...}
EXCEPTIONS_FILE = "exceptions.json"     # [{"rule":"...", "message_regex":"..."}]

DEFAULT_UI_VISIBILITY_HOURS = 24

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
    when: {project_in: ["RAN","East Unwind","Beacon 4","Power Resilience"]}
    expr: title_matches_address
"""

# ----------------------------- UTILS -----------------------------
def sticky_logo_html() -> str:
    p = next((x for x in LOGO_CANDIDATES if os.path.exists(x)), None)
    if not p:
        return """<style>.logo-badge{position:fixed;top:8px;left:12px;z-index:9999;
        color:#fff;background:rgba(0,0,0,0.0);font-size:12px;font-weight:600}</style>
        <div class="logo-badge">⚠️ Add logo file to repo root</div>"""
    with open(p, "rb") as f: b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(p)[1].lower().strip(".")
    return f"""<style>.sticky-logo{{position:fixed;top:14px;left:16px;z-index:10000;width:180px}}</style>
    <img class="sticky-logo" src="data:image/{ext};base64,{b64}"/>"""

def ensure_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f: json.dump(default, f, indent=2)
        return default
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception:
        return default

def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)

def load_rules_text() -> str:
    if not os.path.exists(RULES_FILE):
        with open(RULES_FILE, "w", encoding="utf-8") as f: f.write(DEFAULT_RULES_YAML)
    return open(RULES_FILE, "r", encoding="utf-8").read()

def save_rules_text(txt: str) -> None:
    with open(RULES_FILE, "w", encoding="utf-8") as f: f.write(txt)

def save_history_row(row: Dict[str,Any]) -> None:
    fn = os.path.join(HISTORY_DIR, "audits.csv")
    df = pd.DataFrame([row])
    if os.path.exists(fn): df.to_csv(fn, mode="a", header=False, index=False)
    else: df.to_csv(fn, index=False)

def load_history() -> pd.DataFrame:
    fn = os.path.join(HISTORY_DIR, "audits.csv")
    if not os.path.exists(fn): return pd.DataFrame()
    df = pd.read_csv(fn)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    if "exclude" not in df.columns:
        df["exclude"] = False
    return df

def filter_history_for_ui(df: pd.DataFrame, hours:int, include_excluded:bool=False) -> pd.DataFrame:
    if df.empty or "timestamp_utc" not in df.columns: return df
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    df2 = df[df["timestamp_utc"] >= cutoff].copy()
    if not include_excluded and "exclude" in df2.columns:
        df2 = df2[df2["exclude"] != True]
    return df2

def annotate_pdf(original: bytes, findings: List[Dict[str,Any]]) -> bytes:
    if not fitz: return original
    try:
        doc = fitz.open(stream=original, filetype="pdf")
        y = 36
        for f in findings:
            p = max(int(f.get("page",1))-1, 0)
            p = min(p, len(doc)-1)
            page = doc[p]
            page.insert_text((36, y),
                             f"[{f.get('severity','minor')}] {f.get('rule','')}: {f.get('message','')}",
                             fontsize=8, color=(1,0.3,0.3))
            y += 10
            if y > 780: y = 36
        out = io.BytesIO(); doc.save(out); doc.close()
        return out.getvalue()
    except Exception:
        return original

def make_dual_zip(xlsx_bytes: bytes, pdf_bytes: bytes, base: str) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{base}.xlsx", xlsx_bytes)
        if pdf_bytes:
            z.writestr(f"{base} - annotations.pdf", pdf_bytes)
    return mem.getvalue()

def match_title_to_address(pdf_name: str, site_address: str) -> Tuple[bool,str]:
    if ", 0 ," in site_address.replace(",0,",", 0 ,"): return True, "Ignored by ', 0 ,' rule."
    base = os.path.splitext(os.path.basename(pdf_name or ""))[0]
    a = re.sub(r"[^A-Z0-9]","", site_address.upper())
    b = re.sub(r"[^A-Z0-9]","", base.upper())
    ok = a in b or b in a
    return ok, "OK" if ok else f"Title '{base}' does not match Site Address."

def normalize_text(s: str, corrections: Dict[str,str]) -> str:
    if not s or not corrections: return s
    out = s
    for bad, good in corrections.items():
        if bad: out = re.sub(re.escape(bad), good, out, flags=re.IGNORECASE)
    return out

def is_exception(find: Dict[str,Any], exc_rules: List[Dict[str,Any]]) -> bool:
    r = find.get("rule",""); msg = find.get("message","")
    for e in exc_rules:
        if e.get("rule") == r:
            rx = e.get("message_regex")
            if not rx or re.search(rx, msg, flags=re.IGNORECASE): return True
    return False

# ----------------------------- AUDIT ENGINE -----------------------------
def run_audit(pdf_name: str, pdf_bytes: bytes, meta: Dict[str,Any]) -> Tuple[pd.DataFrame, bytes]:
    findings: List[Dict[str,Any]] = []

    corrections = ensure_json(CORRECTIONS_FILE, {})
    exceptions = ensure_json(EXCEPTIONS_FILE, [])

    ok, msg = match_title_to_address(pdf_name, meta.get("site_address",""))
    if not ok:
        findings.append({"rule":"Address vs Title","severity":"major","page":1,"message":msg})

    text = ""
    if fitz and pdf_bytes:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for i, page in enumerate(doc):
                text += f"\n---PAGE {i+1}---\n{page.get_text('text') or ''}"
            doc.close()
        except Exception:
            pass
    text_norm = normalize_text(text, corrections)

    rules_raw = load_rules_text()
    musts = re.findall(r"must_contain:\s*\[(.*?)\]", rules_raw)
    tokens = []
    for m in musts:
        parts = [p.strip().strip('"').strip("'") for p in m.split(",") if p.strip()]
        tokens.extend(parts)
    for token in tokens:
        if token and token.upper() not in (text_norm.upper() if text_norm else ""):
            f = {"rule":f"Must contain: {token}","severity":"minor","page":1,
                 "message":f"'{token}' not detected in PDF text."}
            if not is_exception(f, exceptions):
                findings.append(f)

    annotated = annotate_pdf(pdf_bytes, findings) if pdf_bytes else b""
    df = pd.DataFrame(findings) if findings else pd.DataFrame(columns=["rule","severity","page","message"])
    return df, annotated

# ----------------------------- UI -----------------------------
st.set_page_config(page_title="AI Design Quality Auditor", layout="wide", page_icon="✅")
st.markdown(sticky_logo_html(), unsafe_allow_html=True)

if "authed" not in st.session_state: st.session_state.authed = False
if not st.session_state.authed:
    st.title("AI Design Quality Auditor")
    p = st.text_input("Enter access password", type="password")
    if st.button("Enter"):
        if p == ENTRY_PASSWORD:
            st.session_state.authed = True; st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

st.title("AI Design Quality Auditor")
st.caption("QA with audit trail, annotations, analytics — and rapid, guided training.")

tab_audit, tab_train, tab_analytics, tab_settings = st.tabs(["🔎 Audit","🏋️ Training","📈 Analytics","⚙️ Settings"])

# ---------- AUDIT ----------
with tab_audit:
    st.subheader("Audit Metadata (required)")
    c1,c2,c3,c4 = st.columns(4)
    supplier = c1.selectbox("Supplier", SUPPLIERS, index=None, placeholder="— Select —")
    drawing_type = c2.selectbox("Drawing Type", DRAWING_TYPES, index=None, placeholder="— Select —")
    client = c3.selectbox("Client", CLIENTS, index=None, placeholder="— Select —")
    project = c4.selectbox("Project", PROJECTS, index=None, placeholder="— Select —")

    c5,c6,c7,c8 = st.columns(4)
    site_type = c5.selectbox("Site Type", SITE_TYPES, index=None, placeholder="— Select —")
    vendor = c6.selectbox("Proposed Vendor", VENDORS, index=None, placeholder="— Select —")
    cabinet_loc = c7.selectbox("Proposed Cabinet Location", CABINET_LOCATIONS, index=None, placeholder="— Select —")
    radio_loc = c8.selectbox("Proposed Radio Location", RADIO_LOCATIONS, index=None, placeholder="— Select —")

    qty = st.selectbox("Quantity of Sectors", [1,2,3,4,5,6], index=None, placeholder="— Select —")
    site_address = st.text_input("Site Address", placeholder="e.g. MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ")

    mimo_optional = (project == "Power Resilience")
    st.markdown(f"**Proposed MIMO Config ({'optional for Power Resilience' if mimo_optional else 'required'})**")
    use_same = st.checkbox("Use same config for all sectors", value=True)
    mimo_selected = {}
    if qty:
        cols = st.columns(min(qty,3))
        for i in range(1, qty+1):
            col = cols[(i-1)%len(cols)]
            mimo_selected[f"S{i}"] = col.selectbox(f"MIMO S{i}", MIMO_OPTIONS, index=None, placeholder="— Select —")
            if i%3==0 and i<qty: cols = st.columns(min(qty-i,3))
        if use_same and mimo_selected.get("S1"):
            for i in range(2, qty+1):
                if not mimo_selected.get(f"S{i}"):
                    mimo_selected[f"S{i}"] = mimo_selected["S1"]

    st.divider()
    pdf_file = st.file_uploader("Upload design PDF", type=["pdf"])
    cbtn1, cbtn2, cbtn3, _ = st.columns([1.2,1.4,2,4])
    run_clicked = cbtn1.button("▶️ Run Audit", type="primary")
    if cbtn2.button("🧹 Clear all metadata"): st.session_state.clear(); st.rerun()
    exclude_flag = cbtn3.checkbox("Exclude this review from analytics", value=False,
                                  help="When checked, this audit is saved but hidden from analytics by default.")

    def meta_ok()->Tuple[bool,str]:
        req = [supplier,drawing_type,client,project,site_type,vendor,cabinet_loc,radio_loc,qty,site_address]
        if any(x in [None,"",0] for x in req): return False, "Please complete all metadata fields."
        if not mimo_optional and qty and any(not mimo_selected.get(f"S{i}") for i in range(1,qty+1)):
            return False, "Please select MIMO for all sectors."
        return True, "OK"

    if run_clicked:
        ok,msg = meta_ok()
        if not ok: st.error(msg)
        elif not pdf_file: st.error("Please upload a PDF.")
        else:
            meta = dict(supplier=supplier,drawing_type=drawing_type,client=client,project=project,
                        site_type=site_type,vendor=vendor,cabinet_location=cabinet_loc,radio_location=radio_loc,
                        qty_sectors=qty,site_address=site_address,mimo={k:v for k,v in mimo_selected.items() if v})
            pdf_bytes = pdf_file.read()
            df, ann = run_audit(pdf_file.name, pdf_bytes, meta)
            status = "Pass" if df.empty else "Rejected"
            if exclude_flag:
                st.info("This review is **excluded** from analytics.")

            if df.empty: st.success("Audit complete — **Pass**"); st.info("No findings.")
            else: st.warning(f"Audit complete — **Rejected**"); st.dataframe(df, use_container_width=True)

            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            base = f"{os.path.splitext(pdf_file.name)[0]} - {status} - {ts}"

            # Build Excel
            xbuf = io.BytesIO()
            with pd.ExcelWriter(xbuf, engine="openpyxl") as xl:
                (df if not df.empty else pd.DataFrame([{"note":"No findings."}])).to_excel(xl, index=False, sheet_name="Findings")
                pd.DataFrame([meta | {"status": status, "excluded": exclude_flag}]).to_excel(xl, index=False, sheet_name="Metadata")
            xbytes = xbuf.getvalue()

            # Downloads
            st.download_button("⬇️ Download report (Excel)", xbytes,
                               file_name=f"{base}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            if ann:
                st.download_button("⬇️ Download annotated PDF", ann,
                                   file_name=f"{base} - annotations.pdf", mime="application/pdf")

            zbytes = make_dual_zip(xbytes, ann, base)
            st.download_button("🗜️ Download BOTH (ZIP)", zbytes, file_name=f"{base}.zip", mime="application/zip")

            # Persist history
            save_history_row({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": status, "file": pdf_file.name, "client": client, "project": project,
                "site_type": site_type, "vendor": vendor, "radio_location": radio_loc, "qty_sectors": qty,
                "drawing_type": drawing_type, "supplier": supplier, "findings_count": len(df.index),
                "mimo_json": json.dumps(meta.get("mimo", {})), "exclude": bool(exclude_flag)
            })

# ---------- TRAINING ----------
with tab_train:
    st.subheader("Rapid Trainer")
    st.caption("Upload reviewed findings to mark false-positives, append scoped rules, and define text corrections.")

    # A) Reviewed findings => exceptions
    st.markdown("**A. Upload reviewed findings (CSV/XLSX)**")
    st.write("Columns: `rule,severity,page,message,decision,comment` — rows with `decision=invalid` are ignored next time.")
    f_up = st.file_uploader("Reviewed findings file", type=["csv","xlsx"], key="train_findings")
    if f_up:
        try:
            df = pd.read_excel(f_up) if f_up.name.endswith(".xlsx") else pd.read_csv(f_up)
            lookup = {c.lower(): c for c in df.columns}
            required = {"rule","message","decision"}
            if not required.issubset(set(lookup.keys())):
                st.error("Missing required columns: rule, message, decision")
            else:
                exceptions = ensure_json(EXCEPTIONS_FILE, [])
                added = 0
                for _, r in df.iterrows():
                    if str(r[lookup["decision"]]).strip().lower() == "invalid":
                        rule = str(r[lookup["rule"]]).strip()
                        msg = str(r[lookup["message"]]).strip()
                        exceptions.append({"rule": rule, "message_regex": re.escape(msg)})
                        added += 1
                save_json(EXCEPTIONS_FILE, exceptions)
                st.success(f"Added {added} exception(s).")
        except Exception as e:
            st.error(f"Failed to process file: {e}")

    st.divider()

    # B) Add a rule via UI
    st.markdown("**B. Add a new rule (scoped)**")
    c1,c2,c3,c4,c5 = st.columns(5)
    r_name = c1.text_input("Rule name", placeholder="e.g., MIMO tag present")
    r_sev = c2.selectbox("Severity", ["minor","major"], index=0)
    r_project = c3.multiselect("Project scope", PROJECTS)
    r_vendor = c4.multiselect("Vendor scope", VENDORS)
    r_site = c5.multiselect("Site type scope", SITE_TYPES)
    c6,c7,c8 = st.columns(3)
    r_draw = c6.multiselect("Drawing type scope", DRAWING_TYPES)
    r_must = c7.text_input("Must contain (comma-separated)")
    r_reject = c8.text_input("Reject if present (comma-separated)")
    if st.button("➕ Append rule to YAML"):
        if not r_name:
            st.error("Please give the rule a name.")
        else:
            txt = load_rules_text()
            block = "\n- name: " + r_name + f"\n  severity: {r_sev}\n"
            if r_must:
                arr = ", ".join([f'"{x.strip()}"' for x in r_must.split(",") if x.strip()])
                block += f"  must_contain: [{arr}]\n"
            else:
                block += "  must_contain: []\n"
            if r_reject:
                arr = ", ".join([f'"{x.strip()}"' for x in r_reject.split(",") if x.strip()])
                block += f"  reject_if_present: [{arr}]\n"
            else:
                block += "  reject_if_present: []\n"
            scopes = []
            if r_project: scopes.append(f'project_in: [{", ".join([f"""\"{x}\"""" for x in r_project])}]')
            if r_vendor:  scopes.append(f'vendor_in: [{", ".join([f"""\"{x}\"""" for x in r_vendor])}]')
            if r_site:    scopes.append(f'site_type_in: [{", ".join([f"""\"{x}\"""" for x in r_site])}]')
            if r_draw:    scopes.append(f'drawing_type_in: [{", ".join([f"""\"{x}\"""" for x in r_draw])}]')
            if scopes:
                block = block.rstrip() + "\n  when: {" + ", ".join(scopes) + "}\n"
            if "checklist:" not in txt:
                txt = "checklist:\n" + block
            else:
                txt = txt.replace("checklist:\n", "checklist:\n" + block)
            save_rules_text(txt)
            st.success("Rule appended to YAML.")

    st.divider()

    # C) Text corrections
    st.markdown("**C. Text corrections**")
    corrections = ensure_json(CORRECTIONS_FILE, {})
    colx, coly = st.columns(2)
    corr_from = colx.text_input("Find (case-insensitive)")
    corr_to = coly.text_input("Replace with")
    if st.button("➕ Add / Update correction"):
        if corr_from:
            corrections[corr_from] = corr_to
            save_json(CORRECTIONS_FILE, corrections)
            st.success(f"Added/updated: {corr_from} → {corr_to}")
    if corrections:
        st.write(pd.DataFrame([{"from":k,"to":v} for k,v in corrections.items()]))

# ---------- ANALYTICS ----------
with tab_analytics:
    st.subheader("Performance & Trends")
    c1,c2 = st.columns([2,1])
    vis = c1.slider("Show records from last (hours)", 1, 24*14, value=DEFAULT_UI_VISIBILITY_HOURS)
    include_ex = c2.checkbox("Include excluded reviews", value=False)
    dfh = load_history()
    dfv = filter_history_for_ui(dfh, vis, include_excluded=include_ex)
    if dfv.empty:
        st.info("No records yet.")
    else:
        st.metric("Total audits", len(dfv))
        st.metric("Right-first-time %", round((dfv["status"].eq("Pass").mean()*100),1))
        st.write("By vendor:"); st.dataframe(dfv.groupby(["vendor","status"]).size().unstack(fill_value=0))
        st.write("By project:"); st.dataframe(dfv.groupby(["project","status"]).size().unstack(fill_value=0))
        st.write("History (latest first):")
        st.dataframe(dfv.sort_values("timestamp_utc", ascending=False), use_container_width=True)
        st.download_button("⬇️ Download filtered CSV", dfv.to_csv(index=False).encode(),
                           file_name="audits_filtered.csv", mime="text/csv")

# ---------- SETTINGS ----------
with tab_settings:
    st.subheader("Edit rules (YAML)")
    st.caption("Scope rules by project/vendor/site type/drawing type. Suppliers do not affect rules.")
    pw = st.text_input("Rules password", type="password", help="Required to save changes.")
    text = st.text_area(RULES_FILE, value=load_rules_text(), height=380)
    csa, csb, csc = st.columns([1,1,4])
    if csa.button("Save rules"):
        if pw == RULES_PASSWORD:
            try: save_rules_text(text); st.success("Rules saved.")
            except Exception as e: st.error(f"Save failed: {e}")
        else:
            st.error("Incorrect rules password.")
    if csb.button("Reload from disk"): st.experimental_rerun()
    st.divider()
    if st.button("Clear all history (keeps files)"):
        fn = os.path.join(HISTORY_DIR, "audits.csv")
        if os.path.exists(fn): os.remove(fn)
        st.success("History cleared.")
