# app.py
# AI Design Quality Auditor – compact stable build
# Tabs: Audit | Analytics | Settings | Training
# Gate password: Seker123  | Rules password (for YAML editor): vanB3lkum21

import io, os, re, json, base64, textwrap, datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import yaml

# PDF handling
import fitz  # PyMuPDF

# Optional OCR path
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Spelling
try:
    from spellchecker import SpellChecker
    SPELL_OK = True
except Exception:
    SPELL_OK = False

APP_NAME = "AI Design Quality Auditor"
HISTORY_DIR = "history"
HISTORY_CSV = os.path.join(HISTORY_DIR, "audit_history.csv")
EXPORT_DIR = "exports"            # daily export dumps if you enable them
LOGO_GLOB = ["logo.png", "logo.jpg", "logo.jpeg", "logo.svg",
             "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png"]

ENTRY_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"
DEFAULT_RULES_FILE = "rules_example.yaml"

# Master lists (stable)
SUPPLIERS = ["CEG", "CTIL", "Emfyser", "Innov8", "Invict",
             "KTL Team (Internal)", "Trylon"]

DRAWING_TYPES = ["General Arrangement", "Detailed Design"]
CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
CAB_LOC = ["Indoor", "Outdoor"]
RADIO_LOC = ["Low Level", "High Level", "Unique Coverage", "Midway"]

SECTOR_QTY = [1, 2, 3, 4, 5, 6]

# MIMO options (as provided)
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
 "(blank)"
]

# ==== utilities =============================================================

def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def ensure_dirs():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)

def load_history() -> pd.DataFrame:
    ensure_dirs()
    if not os.path.exists(HISTORY_CSV):
        cols = ["timestamp_utc","supplier","drawing_type","client","project",
                "site_type","vendor","cab_loc","radio_loc",
                "sectors","site_address","mimo_s1","mimo_s2","mimo_s3",
                "mimo_s4","mimo_s5","mimo_s6","status",
                "pdf_name","excel_name","annotated_pdf","exclude"]
        return pd.DataFrame(columns=cols)
    try:
        return pd.read_csv(HISTORY_CSV)
    except Exception:
        # corrupted row – keep what we can
        return pd.read_csv(HISTORY_CSV, on_bad_lines="skip")

def save_history_row(row: Dict[str, Any]) -> None:
    df = load_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_CSV, index=False)

def load_rules(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"checklist": []}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"checklist": []}

def save_rules(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def site_address_ok(address: str, pdf_name: str) -> bool:
    if not address.strip():
        return False
    # If ", 0 ," present, ignore that token
    cleaned = re.sub(r"\s*,\s*0\s*,\s*", ", ", address.strip(), flags=re.I)
    # crude match: every alpha token from address must appear in pdf title
    title = os.path.splitext(os.path.basename(pdf_name))[0]
    a_tokens = [t for t in re.split(r"[^A-Za-z0-9]+", cleaned) if t]
    title_low = title.lower()
    return all(t.lower() in title_low for t in a_tokens if len(t) > 2)

def get_logo_path() -> Optional[str]:
    for name in LOGO_GLOB:
        if os.path.exists(name):
            return name
    return None

def inject_logo_top_left():
    p = get_logo_path()
    if not p:
        return
    ext = p.split(".")[-1].lower()
    if ext == "svg":
        with open(p, "r", encoding="utf-8") as f:
            svg = f.read()
        b64 = base64.b64encode(svg.encode()).decode()
        tag = f'<img src="data:image/svg+xml;base64,{b64}" class="logo">'
    else:
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        tag = f'<img src="data:image/{ext};base64,{b64}" class="logo">'
    st.markdown("""
    <style>
      .logo {position: fixed; left: 18px; top: 12px; height: 56px; z-index: 9999;}
      header {padding-top: 60px;}
    </style>
    """ + tag, unsafe_allow_html=True)

# ==== PDF text + search/annotate ============================================

def extract_text_with_fitz(pdf_bytes: bytes) -> List[str]:
    pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for p in doc:
        pages.append(p.get_text("text"))
    doc.close()
    return pages

def extract_text_with_ocr(pdf_bytes: bytes) -> List[str]:
    if not OCR_AVAILABLE:
        return []
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception:
        return []
    text_pages = []
    for im in imgs:
        try:
            text_pages.append(pytesseract.image_to_string(im))
        except Exception:
            text_pages.append("")
    return text_pages

def keyword_boxes(pdf_bytes: bytes, term: str) -> List[Tuple[int, fitz.Rect]]:
    """Locate approximate bounding boxes for a term (case-insensitive).
       Robust to environments where TEXT_IGNORECASE flag may not exist."""
    out = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        q = term.strip()
        if not q:
            return out
        for pno, page in enumerate(doc):
            # safest path: run both default search and a manual scan of words
            try:
                hits = page.search_for(q)  # may be case-sensitive depending on build
                out += [(pno, r) for r in hits]
            except Exception:
                pass
            # fallback manual scan
            words = page.get_text("words")  # list: x0,y0,x1,y1, word, block, line, word_no
            for x0,y0,x1,y1,w, *_ in words:
                if w.lower() == q.lower():
                    out.append((pno, fitz.Rect(x0,y0,x1,y1)))
        doc.close()
    except Exception:
        pass
    return out

def annotate_pdf(pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    """Add simple rectangles and sticky notes for each finding."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return pdf_bytes

    for f in findings:
        txt = f.get("message", "")
        key = f.get("matched", "") or f.get("rule", "")
        page_hint = int(f.get("page", 1)) - 1
        boxes = []
        if key:
            boxes = keyword_boxes(pdf_bytes, key)
        if not boxes and page_hint >= 0 and page_hint < len(doc):
            # fallback: drop a note on the hinted page
            p = doc[page_hint]
            p.add_text_annot(p.rect.tl + fitz.Point(50, 60), txt)
            continue
        for pno, rect in boxes[:5]:
            p = doc[pno]
            try:
                p.add_rect_annot(rect)
                p.add_text_annot(rect.tl + fitz.Point(0, -8), txt[:200])
            except Exception:
                # last-resort: just add a note near top-left
                p.add_text_annot(p.rect.tl + fitz.Point(50, 60), txt[:200])

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()

# ==== Rules + checks ========================================================

def run_checks(pages: List[str], meta: Dict[str, Any], rules: Dict[str, Any], do_spell: bool) -> List[Dict[str, Any]]:
    """Evaluate simple YAML-driven checks + optional spelling."""
    findings: List[Dict[str, Any]] = []
    text_all = "\n".join(pages)

    # Site Address vs title check (always)
    if meta.get("pdf_name") and meta.get("site_address"):
        if not site_address_ok(meta["site_address"], meta["pdf_name"]):
            findings.append({
                "severity": "major",
                "rule": "Site address must match drawing title",
                "message": f"Site address '{meta['site_address']}' not found in '{meta['pdf_name']}'.",
                "matched": meta["site_address"],
                "page": 1,
                "category": "metadata"
            })

    for chk in rules.get("checklist", []):
        name = chk.get("name", "rule")
        sev  = chk.get("severity", "minor")
        must = [m for m in chk.get("must_contain", []) if isinstance(m, str)]
        reject_if = [m for m in chk.get("reject_if_present", []) if isinstance(m, str)]
        scope = chk.get("scope", "any")  # future: page/sheet
        matched = None

        if must:
            ok = all(m.lower() in text_all.lower() for m in must)
            if not ok:
                missing = [m for m in must if m.lower() not in text_all.lower()]
                findings.append({
                    "severity": sev, "rule": name,
                    "message": f"Missing required text: {', '.join(missing)}",
                    "matched": (missing[0] if missing else ""),
                    "page": 1, "category": "rule"
                })
        for bad in reject_if:
            if bad and bad.lower() in text_all.lower():
                findings.append({
                    "severity": "major", "rule": name,
                    "message": f"Forbidden text present: {bad}",
                    "matched": bad, "page": 1, "category": "rule"
                })

    if do_spell and SPELL_OK:
        # simple spell pass on long words; whitelist common RF tokens
        wl = set(["ericsson","nokia","mimo","sector","cabinet","outdoor","indoor",
                  "mbnl","cellnex","cornerstone","vodafone","btee"])
        sp = SpellChecker()
        words = re.findall(r"[A-Za-z]{4,}", text_all)
        miss = [w for w in words if w.lower() not in wl and w.lower() not in sp]
        for w in sorted(set(miss))[:50]:
            sug = next(iter(sp.candidates(w)), None)
            findings.append({
                "severity": "minor",
                "rule": "Spelling",
                "message": f"Possible typo: '{w}'" + (f" → '{sug}'" if sug else ""),
                "matched": w, "page": 1, "category": "spelling"
            })

    return findings

# ==== Excel / downloads =====================================================

def make_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], base_pdf_name: str, status: str) -> bytes:
    df = pd.DataFrame(findings)
    meta_row = {**meta, "pdf_name": base_pdf_name, "status": status, "generated_utc": now_utc_iso()}
    mem = io.BytesIO()
    with pd.ExcelWriter(mem, engine="openpyxl") as xw:
        (df if not df.empty else pd.DataFrame(columns=["rule","severity","message","matched","page","category"])) \
            .to_excel(xw, index=False, sheet_name="Findings")
        pd.DataFrame([meta_row]).to_excel(xw, index=False, sheet_name="Meta")
    return mem.getvalue()

def status_from_findings(findings: List[Dict[str, Any]]) -> str:
    if any(f.get("severity") == "major" for f in findings):
        return "Rejected"
    return "Pass"

# ==== UI helpers ============================================================

def gate() -> bool:
    st.session_state.setdefault("gate_ok", False)
    if st.session_state["gate_ok"]:
        return True
    col1, col2 = st.columns([1,3])
    with col1:
        inject_logo_top_left()
    with col2:
        st.title(APP_NAME)
        pw = st.text_input("Enter access password", type="password")
        if st.button("Unlock"):
            st.session_state["gate_ok"] = (pw == ENTRY_PASSWORD)
            if not st.session_state["gate_ok"]:
                st.error("Wrong password.")
        st.info("Hint: ask your admin for access.")
    return st.session_state["gate_ok"]

def meta_form() -> Dict[str, Any]:
    st.subheader("Audit Metadata (all required)")
    c1,c2,c3,c4 = st.columns(4)
    supplier = c1.selectbox("Supplier", SUPPLIERS, index=None, placeholder="— Select —")
    drawing_type = c2.selectbox("Drawing Type", DRAWING_TYPES, index=None, placeholder="— Select —")
    client = c3.selectbox("Client", CLIENTS, index=None, placeholder="— Select —")
    project = c4.selectbox("Project", PROJECTS, index=None, placeholder="— Select —")

    c5,c6,c7,c8 = st.columns(4)
    site_type = c5.selectbox("Site Type", SITE_TYPES, index=None, placeholder="— Select —")
    vendor = c6.selectbox("Proposed Vendor", VENDORS, index=None, placeholder="— Select —")
    cab_loc = c7.selectbox("Proposed Cabinet Location", CAB_LOC, index=None, placeholder="— Select —")
    radio_loc = c8.selectbox("Proposed Radio Location", RADIO_LOC, index=None, placeholder="— Select —")

    c9,c10 = st.columns([1,3])
    sectors = c9.selectbox("Quantity of Sectors", SECTOR_QTY, index=0)
    site_address = c10.text_input("Site Address", placeholder="e.g., MANBY ROAD , 0 , IMMINGHAM , DN40 2LQ")

    st.markdown("### Proposed MIMO Config")
    same_all = st.checkbox("Use S1 for all sectors", value=True)
    mimo1 = st.selectbox("MIMO S1", MIMO_OPTIONS, index=0)
    # build S2..S6 conditionally
    mimo_map = {"mimo_s1": mimo1}
    for i in range(2, sectors+1):
        mimo_map[f"mimo_s{i}"] = st.selectbox(f"MIMO S{i}",
                                              MIMO_OPTIONS,
                                              index=MIMO_OPTIONS.index(mimo1) if same_all else 0,
                                              key=f"mimo_{i}",
                                              disabled=same_all)
    for j in range(sectors+1, 7):
        mimo_map[f"mimo_s{j}"] = ""

    meta = dict(supplier=supplier, drawing_type=drawing_type, client=client, project=project,
                site_type=site_type, vendor=vendor, cab_loc=cab_loc, radio_loc=radio_loc,
                sectors=sectors, site_address=site_address)
    meta.update(mimo_map)
    return meta

def assert_all_meta(meta: Dict[str, Any]) -> Optional[str]:
    for k in ["supplier","drawing_type","client","project","site_type","vendor","cab_loc","radio_loc","site_address","sectors"]:
        if not meta.get(k):
            return f"Missing required metadata: {k}"
    return None

# ==== Tabs ==================================================================

def audit_tab():
    st.header("Audit")
    meta = meta_form()

    st.markdown("#### Upload PDF Design")
    up = st.file_uploader("Drop PDF here", type=["pdf"], label_visibility="collapsed")

    st.markdown("---")
    left, mid, right = st.columns([1,1,1])
    do_spell = left.checkbox("Enable spelling check", value=True)
    want_ocr = mid.checkbox("Enable OCR fallback (slower)", value=False and OCR_AVAILABLE,
                            help="Uses pdf2image + Tesseract if available.")
    exclude = right.checkbox("Exclude this result from Analytics", value=False)

    if st.button("Run Audit", type="primary", use_container_width=True, disabled=(up is None)):
        err = assert_all_meta(meta)
        if err:
            st.error(err)
            return
        meta["pdf_name"] = up.name

        pdf_bytes = up.read()
        pages = extract_text_with_fitz(pdf_bytes)
        if want_ocr and not any(p.strip() for p in pages):
            pages = extract_text_with_ocr(pdf_bytes)

        rules = load_rules(DEFAULT_RULES_FILE)
        with st.status("Running checks…", expanded=True) as s:
            findings = run_checks(pages, meta, rules, do_spell)
            st.write(f"{len(findings)} findings")
            s.update(label="Annotating PDF…")
            annotated_pdf = annotate_pdf(pdf_bytes, findings)
            s.update(label="Building Excel…")
            status = status_from_findings(findings)
            xlsx = make_excel(findings, meta, up.name, status)

        # Save a history row
        stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        excel_name = f"{os.path.splitext(up.name)[0]}_{status}_{stamp}.xlsx"
        pdf_name = f"{os.path.splitext(up.name)[0]}_{status}_{stamp}.pdf"
        history_row = {
            "timestamp_utc": now_utc_iso(), **meta,
            "status": status, "pdf_name": pdf_name, "excel_name": excel_name,
            "annotated_pdf": pdf_name, "exclude": bool(exclude)
        }
        save_history_row(history_row)

        c1,c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download Excel report", data=xlsx, file_name=excel_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with c2:
            st.download_button("⬇️ Download annotated PDF", data=annotated_pdf, file_name=pdf_name, mime="application/pdf", use_container_width=True)

        st.success(f"Audit complete: **{status}**")

def analytics_tab():
    st.header("Analytics")
    df = load_history()
    if df.empty:
        st.info("No history yet.")
        return

    # Keep records not excluded
    df_use = df.copy()
    if "exclude" in df_use.columns:
        df_use = df_use[df_use["exclude"] != True]  # noqa: E712

    # filters
    f1,f2,f3 = st.columns(3)
    sel_client = f1.multiselect("Client", sorted(df_use["client"].dropna().unique().tolist()))
    sel_project = f2.multiselect("Project", sorted(df_use["project"].dropna().unique().tolist()))
    sel_supplier = f3.multiselect("Supplier", sorted(df_use["supplier"].dropna().unique().tolist()))

    mask = pd.Series(True, index=df_use.index)
    if sel_client:  mask &= df_use["client"].isin(sel_client)
    if sel_project: mask &= df_use["project"].isin(sel_project)
    if sel_supplier: mask &= df_use["supplier"].isin(sel_supplier)
    show = df_use[mask].copy()

    # KPIs
    total = len(show)
    passes = int((show["status"] == "Pass").sum())
    rejects = int((show["status"] == "Rejected").sum())
    rft = (passes / total * 100.0) if total else 0.0
    k1,k2,k3 = st.columns(3)
    k1.metric("Audits", total)
    k2.metric("Right-First-Time %", f"{rft:.1f}%")
    k3.metric("Rejected", rejects)

    # table
    cols = [c for c in ["timestamp_utc","supplier","client","project","status","pdf_name","excel_name"] if c in show.columns]
    st.dataframe(show[cols], use_container_width=True, height=320)

def settings_tab():
    st.header("Settings")
    inject_logo_top_left()
    st.write("Place your logo file in repo root (e.g., `logo.png`).")
    # Rules editor – gated
    pw = st.text_input("Rules password", type="password")
    rules_file = DEFAULT_RULES_FILE
    st.caption(rules_file)
    rules_text = None
    if os.path.exists(rules_file):
        rules_text = open(rules_file, "r", encoding="utf-8").read()
    else:
        rules_text = "checklist:\n  - name: Title block present\n    severity: major\n    must_contain:\n      - TITLE\n    reject_if_present: []\n"

    if st.text_input("Confirm file name to edit", value=rules_file, disabled=True):
        area = st.text_area("Edit YAML rules", value=rules_text, height=280)
        c1,c2 = st.columns(2)
        if c1.button("Save rules", type="primary", disabled=(pw != RULES_PASSWORD)):
            try:
                data = yaml.safe_load(area) or {}
                save_rules(rules_file, data)
                st.success("Rules saved.")
            except Exception as e:
                st.error(f"YAML error: {e}")
        if c2.button("Reload from disk"):
            st.experimental_rerun()

    st.divider()
    st.write("Daily export dumps (CSV of history).")
    if st.button("Export now"):
        df = load_history()
        if df.empty:
            st.info("Nothing to export.")
        else:
            ensure_dirs()
            fn = os.path.join(EXPORT_DIR, f"history_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(fn, index=False)
            st.success(f"Exported to {fn}")

def training_tab():
    st.header("Training")
    st.write("Upload **audited report** (Excel or JSON) to record **Valid / Not-Valid** decisions and add quick rules.")
    colA, colB = st.columns(2)
    with colA:
        up = st.file_uploader("Upload Excel/JSON training record", type=["xlsx","json"])
        label = st.selectbox("This audit decision is…", ["Valid", "Not Valid"])
        if st.button("Ingest training record", disabled=(up is None)):
            try:
                if up.name.lower().endswith(".xlsx"):
                    df = pd.read_excel(up)
                    added = False
                    if not df.empty:
                        # Store a slim snapshot in history for transparency
                        row = {"timestamp_utc": now_utc_iso(), "supplier": "", "drawing_type":"",
                               "client":"", "project":"", "site_type":"", "vendor":"", "cab_loc":"",
                               "radio_loc":"", "sectors":"", "site_address":"",
                               "mimo_s1":"", "mimo_s2":"", "mimo_s3":"", "mimo_s4":"",
                               "mimo_s5":"", "mimo_s6":"", "status": label,
                               "pdf_name": f"TRAIN_{up.name}", "excel_name": up.name,
                               "annotated_pdf":"", "exclude": True}
                        save_history_row(row); added = True
                    if added:
                        st.success("Training snapshot saved to history (excluded from analytics).")
                else:
                    data = json.loads(up.read().decode("utf-8"))
                    row = {"timestamp_utc": now_utc_iso(), "status": label, "pdf_name":"TRAIN_JSON",
                           "excel_name":"TRAIN_JSON", "exclude": True}
                    save_history_row(row)
                    st.success("JSON training ingested (excluded from analytics).")
            except Exception as e:
                st.error(f"Could not ingest: {e}")
    with colB:
        st.write("Add a **quick rule** (appends to YAML instantly).")
        rule_name = st.text_input("Rule name", placeholder="e.g., Power Resilience note present")
        sev = st.selectbox("Severity", ["minor","major"], index=1)
        must = st.text_input("Must contain (comma-separated)", placeholder="IMPORTANT NOTE, ELTEK PSU")
        rej = st.text_input("Reject if present (comma-separated)", placeholder="")
        pw = st.text_input("Rules password", type="password", placeholder="vanB3lkum21")
        if st.button("Append rule", type="primary", disabled=(pw != RULES_PASSWORD or not rule_name.strip())):
            try:
                data = load_rules(DEFAULT_RULES_FILE)
                data.setdefault("checklist", []).append({
                    "name": rule_name.strip(),
                    "severity": sev,
                    "must_contain": [x.strip() for x in must.split(",") if x.strip()],
                    "reject_if_present": [x.strip() for x in rej.split(",") if x.strip()]
                })
                save_rules(DEFAULT_RULES_FILE, data)
                st.success("Rule appended.")
            except Exception as e:
                st.error(f"Failed to append: {e}")

# ==== main ==================================================================

def main():
    st.set_page_config(page_title=APP_NAME, page_icon="🛠️", layout="wide")
    inject_logo_top_left()
    if not gate():
        return

    tabs = st.tabs(["Audit", "Analytics", "Settings", "Training"])
    with tabs[0]:
        audit_tab()
    with tabs[1]:
        analytics_tab()
    with tabs[2]:
        settings_tab()
    with tabs[3]:
        training_tab()

if __name__ == "__main__":
    main()
