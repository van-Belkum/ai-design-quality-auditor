# app.py  — compact professional build
import os, io, re, json, base64, textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import yaml

# Optional libs (soft deps)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# -----------------------
# Constants & Paths
# -----------------------
APP_TITLE = "AI Design Quality Auditor"
RULES_PATH = "rules_example.yaml"
HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)
HISTORY_PATH = HISTORY_DIR / "audit_history.csv"
ALLOWLIST_PATH = "allowlist.txt"  # optional per-user additions
ADMIN_PASS = os.getenv("ADMIN_PASS") or (getattr(st, "secrets", {}).get("admin_pass") if hasattr(st, "secrets") else None) or "vanB3lkum21"

SUPPLIERS = ["CEG","CTIL","Emfyser","Innov8","Invict","KTL Team (Internal)","Trylon"]
DRAWING_TYPES = ["General Arrangement","Detailed Design"]

CLIENTS = ["BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"]
PROJECTS = ["RAN","Power Resilience","East Unwind","Beacon 4"]
SITE_TYPES = ["Greenfield","Rooftop","Streetworks"]
VENDORS = ["Ericsson","Nokia"]
CABINET_LOCS = ["Indoor","Outdoor"]
RADIO_LOCS = ["High Level","Low Level","Indoor","Door"]
SECTORS = ["1","2","3","4","5","6"]

# -----------------------
# Utilities
# -----------------------
def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def _ensure_history():
    if not HISTORY_PATH.exists():
        pd.DataFrame().to_csv(HISTORY_PATH, index=False)

def ensure_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "timestamp_utc","file","client","project","site_type","vendor",
        "cabinet_loc","radio_loc","sectors","mimo_config","site_address",
        "supplier","drawing_type","used_ocr","pages",
        "minor_findings","major_findings","total_findings",
        "outcome","rft_percent","exclude"
    ]
    if df.empty:
        for c in expected: df[c] = []
        return df
    for col in expected:
        if col not in df.columns:
            if col == "exclude":
                df[col] = False
            elif col in {"minor_findings","major_findings","total_findings","pages"}:
                df[col] = 0
            elif col == "rft_percent":
                df[col] = 0.0
            else:
                df[col] = ""
    df["exclude"] = df["exclude"].fillna(False).astype(bool)
    return df

def is_admin_unlocked() -> bool:
    return st.session_state.get("admin_ok", False)

def load_rules(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"checklist":{}, "clients":{}, "projects":{}, "site_types":{}, "vendors":{},
                "cabinet_locations":{}, "radio_locations":{}, "sectors":{},
                "suppliers":{}, "drawing_types":{}}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data

def save_rules(path: str, text: str):
    # validate parse before saving
    _ = yaml.safe_load(text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def load_allowlist() -> set:
    if not os.path.exists(ALLOWLIST_PATH):
        return set()
    with open(ALLOWLIST_PATH, "r", encoding="utf-8") as f:
        return set([ln.strip().lower() for ln in f if ln.strip()])

def save_allowlist(words: List[str]):
    with open(ALLOWLIST_PATH, "w", encoding="utf-8") as f:
        for w in sorted(set([w.strip().lower() for w in words if w.strip()])):
            f.write(w + "\n")

def read_logo_bytes(logo_path: str) -> bytes:
    try:
        with open(logo_path, "rb") as f:
            return f.read()
    except Exception:
        return b""

def top_right_logo_css():
    st.markdown("""
        <style>
        .top-right-logo {
            position: fixed; 
            top: 10px; 
            right: 18px; 
            width: 140px; 
            z-index: 1000; 
        }
        /* widen the page a bit */
        .block-container {padding-top: 70px;}
        </style>
    """, unsafe_allow_html=True)

def show_logo(logo_bytes: bytes):
    if not logo_bytes:
        return
    b64 = base64.b64encode(logo_bytes).decode("ascii")
    st.markdown(f"""<img class="top-right-logo" src="data:image/png;base64,{b64}" />""", unsafe_allow_html=True)

# -----------------------
# PDF text & bbox
# -----------------------
def extract_pages_and_index(pdf_bytes: bytes) -> Tuple[List[str], List[Dict]]:
    """Return (pages_text, word_index) 
       word_index = list per page: [{"text":word,"bbox":(x0,y0,x1,y1)}...]"""
    if not fitz:
        return [], []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []
    pages_words = []
    for p in doc:
        text = p.get_text("text") or ""
        pages_text.append(text)
        words = []
        for w in p.get_text("words") or []:
            # w: (x0, y0, x1, y1, "text", block_no, line_no, word_no)
            words.append({"text": w[4], "bbox": (w[0], w[1], w[2], w[3])})
        pages_words.append(words)
    return pages_text, pages_words

def find_first_bbox(pages_words: List[List[Dict]], token: str):
    token_norm = token.strip().lower()
    for pi, words in enumerate(pages_words):
        for w in words:
            if w["text"].strip().lower() == token_norm:
                return pi, w["bbox"]
    # fallback: search inside longer words
    for pi, words in enumerate(pages_words):
        for w in words:
            if token_norm in w["text"].strip().lower():
                return pi, w["bbox"]
    return None, None

def annotate_pdf(pdf_bytes: bytes, marks: List[Dict[str, Any]]) -> bytes:
    """marks: [{'page':int,'bbox':(x0,y0,x1,y1) or None,'note':str}]"""
    if not fitz:
        return pdf_bytes
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for m in marks:
        page_i = m.get("page", 0)
        if page_i < 0 or page_i >= len(doc):
            page_i = 0
        page = doc[page_i]
        bbox = m.get("bbox")
        note = m.get("note", "")
        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                rect = fitz.Rect(*bbox)
                page.add_rect_annot(rect)
                page.add_freetext_annot(rect, note or "Finding", rotate=0, fontsize=8)
            except Exception:
                # fallback to a sticky note at top-left
                page.add_text_annot(page.rect.tl, note or "Finding")
        else:
            # no bbox known, drop a note near top-left margin
            page.add_text_annot(page.rect.tl, note or "Finding")
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

# -----------------------
# Finding helpers
# -----------------------
def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip().lower()

def token_present(pages_text: List[str], token: str) -> Tuple[bool, int]:
    tok = normalize_text(token)
    for i, t in enumerate(pages_text):
        if tok and tok in normalize_text(t):
            return True, i
    return False, -1

def apply_rule_sets(pages_text: List[str], rules: Dict[str, Any], meta: Dict[str, str]) -> List[Dict[str, Any]]:
    findings = []
    # Generic checklist
    for item in rules.get("checklist", []) or []:
        name = item.get("name") or "Checklist"
        sev = (item.get("severity") or "minor").lower()
        must = item.get("must_contain", []) or []
        rej = item.get("reject_if_present", []) or []
        # must_contain: each token must appear at least once
        for tok in must:
            present, page_i = token_present(pages_text, tok)
            if not present:
                findings.append({"category":"checklist","rule":name,"severity":sev,
                                 "message":f"Missing required token: '{tok}'","page":max(0,page_i)})
        # reject_if_present: any token -> major
        for tok in rej:
            present, page_i = token_present(pages_text, tok)
            if present:
                findings.append({"category":"checklist","rule":name,"severity":"major",
                                 "message":f"Forbidden token present: '{tok}'","page":max(0,page_i)})

    # Category dicts and selected values
    mapping = [
        ("clients","client"),
        ("projects","project"),
        ("site_types","site_type"),
        ("vendors","vendor"),
        ("cabinet_locations","cabinet_loc"),
        ("radio_locations","radio_loc"),
        ("sectors","sectors"),
        ("suppliers","supplier"),
        ("drawing_types","drawing_type"),
    ]
    for rule_key, meta_key in mapping:
        selected = meta.get(meta_key, "")
        if not selected:
            continue
        bucket = rules.get(rule_key, {}) or {}
        node = bucket.get(selected, {}) or {}
        must = node.get("must_contain", []) or []
        rej = node.get("reject_if_present", []) or []
        for tok in must:
            present, page_i = token_present(pages_text, tok)
            if not present:
                findings.append({"category":rule_key,"rule":selected,"severity":"minor",
                                 "message":f"Missing required token for {meta_key}: '{tok}'","page":max(0,page_i)})
        for tok in rej:
            present, page_i = token_present(pages_text, tok)
            if present:
                findings.append({"category":rule_key,"rule":selected,"severity":"major",
                                 "message":f"Forbidden token for {meta_key}: '{tok}'","page":max(0,page_i)})

    # Site address vs title/text check (ignore if contains ", 0,")
    site_addr = (meta.get("site_address") or "").strip()
    if site_addr and ", 0," not in site_addr.replace(" ,", ","):
        present, page_i = token_present(pages_text, site_addr)
        if not present:
            findings.append({"category":"address","rule":"address-title-match","severity":"major",
                             "message":"Site address not found in drawing text/title.","page":max(0,page_i)})

    # Power Resilience: hide MIMO; otherwise if provided, lightly validate form
    if meta.get("project") != "Power Resilience":
        mimo = (meta.get("mimo_config") or "").strip()
        if mimo and not re.search(r"\d{2,4}\s*@\s*\d+x\d+", mimo):
            findings.append({"category":"mimo","rule":"format","severity":"minor",
                             "message":"Proposed MIMO Config format looks unusual (expected like '3500 @32x32').","page":0})

    return findings

def attach_bboxes_to_findings(findings: List[Dict[str,Any]], pages_words: List[List[Dict]], pages_text: List[str]):
    enriched = []
    for f in findings:
        note_token = None
        # Try to derive a token we can search for a bbox
        m = re.search(r"'([^']+)'", f.get("message",""))
        if m:
            note_token = m.group(1).strip()
        page = int(f.get("page",0)) if isinstance(f.get("page",0), int) else 0
        bbox = None
        if note_token:
            pi, bb = find_first_bbox(pages_words, note_token)
            if bb: 
                page = pi if pi is not None else page
                bbox = bb
        enriched.append({**f, "page": page, "bbox": bbox})
    return enriched

def build_excel_and_pdf(file_name: str, pdf_bytes: bytes, findings: List[Dict[str,Any]], meta: Dict[str,str]) -> Tuple[bytes, bytes, str]:
    # dataframe
    rows = []
    majors = 0; minors = 0
    for f in findings:
        if f.get("severity","minor") == "major": majors += 1
        else: minors += 1
        rows.append({
            "file": file_name,
            "category": f.get("category",""),
            "rule": f.get("rule",""),
            "severity": f.get("severity",""),
            "message": f.get("message",""),
            "page": f.get("page",0)
        })
    df = pd.DataFrame(rows or [{"file":file_name,"category":"","rule":"","severity":"","message":"No findings","page":""}])
    outcome = "PASS" if len(findings)==0 else "REJECTED"
    rft = 100.0 if len(findings)==0 else 0.0
    stamp = datetime.now().strftime("%Y%m%d")
    base = Path(file_name).stem
    xlsx_name = f"{base}_{outcome}_{stamp}.xlsx"
    pdf_name = f"{base}_ANNOTATED_{outcome}_{stamp}.pdf"

    # Excel bytes
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as xw:
        meta_df = pd.DataFrame([meta])
        meta_df.to_excel(xw, index=False, sheet_name="Audit Meta")
        df.to_excel(xw, index=False, sheet_name="Findings")
        summary = pd.DataFrame([{
            "Outcome": outcome,
            "Minor Findings": minors,
            "Major Findings": majors,
            "Total Findings": len(findings),
            "RFT %": rft
        }])
        summary.to_excel(xw, index=False, sheet_name="Summary")
    excel_bytes = excel_buf.getvalue()

    # Annotated PDF
    marks = [{"page": f.get("page",0), "bbox": f.get("bbox"), "note": f.get("message","Finding")} for f in findings]
    annotated_pdf = annotate_pdf(pdf_bytes, marks)

    return excel_bytes, annotated_pdf, outcome

def push_history(file_name: str, findings: List[Dict[str,Any]], outcome: str, pages: int, meta: Dict[str,str]):
    _ensure_history()
    try:
        dfh = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp_utc"])
    except Exception:
        dfh = pd.DataFrame()

    dfh = ensure_history_columns(dfh)
    minor = sum(1 for f in findings if f.get("severity")=="minor")
    major = sum(1 for f in findings if f.get("severity")=="major")
    new_row = {
        "timestamp_utc": _utc_now_iso(),
        "file": file_name,
        "client": meta.get("client",""),
        "project": meta.get("project",""),
        "site_type": meta.get("site_type",""),
        "vendor": meta.get("vendor",""),
        "cabinet_loc": meta.get("cabinet_loc",""),
        "radio_loc": meta.get("radio_loc",""),
        "sectors": meta.get("sectors",""),
        "mimo_config": meta.get("mimo_config",""),
        "site_address": meta.get("site_address",""),
        "supplier": meta.get("supplier",""),
        "drawing_type": meta.get("drawing_type",""),
        "used_ocr": "False",
        "pages": pages,
        "minor_findings": minor,
        "major_findings": major,
        "total_findings": minor+major,
        "outcome": outcome,
        "rft_percent": 100.0 if (minor+major)==0 else 0.0,
        "exclude": False
    }
    dfh = pd.concat([dfh, pd.DataFrame([new_row])], ignore_index=True)
    dfh.to_csv(HISTORY_PATH, index=False)

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
top_right_logo_css()

# Settings state defaults
if "logo_path" not in st.session_state:
    st.session_state["logo_path"] = ""
logo_bytes = read_logo_bytes(st.session_state.get("logo_path",""))
show_logo(logo_bytes)

st.title(APP_TITLE)

tabs = st.tabs(["🧪 Audit", "📈 History & Analytics", "⚙️ Settings"])

# -----------------------
# TAB 1 — Audit
# -----------------------
with tabs[0]:
    st.subheader("Audit Metadata (all required)")

    c1, c2, c3 = st.columns(3)
    with c1:
        client = st.selectbox("Client", CLIENTS, index=None, placeholder="Select client")
        project = st.selectbox("Project", PROJECTS, index=None, placeholder="Select project")
        site_type = st.selectbox("Site Type", SITE_TYPES, index=None, placeholder="Select site type")
    with c2:
        vendor = st.selectbox("Proposed Vendor", VENDORS, index=None, placeholder="Select vendor")
        cabinet_loc = st.selectbox("Proposed Cabinet Location", CABINET_LOCS, index=None, placeholder="Select cabinet location")
        radio_loc = st.selectbox("Proposed Radio Location", RADIO_LOCS, index=None, placeholder="Select radio location")
    with c3:
        sectors = st.selectbox("Quantity of Sectors", SECTORS, index=None, placeholder="Select sectors")
        supplier = st.selectbox("Supplier", SUPPLIERS, index=None, placeholder="Select supplier")
        drawing_type = st.selectbox("Drawing Type", DRAWING_TYPES, index=None, placeholder="Select drawing type")

    # MIMO hidden for Power Resilience
    if project != "Power Resilience":
        mimo_config = st.text_input("Proposed MIMO Config (e.g., '3500 @32x32')", value="")
    else:
        mimo_config = ""

    site_address = st.text_input("Site Address (exact string expected in drawing; ignored if contains ', 0,')", value="")

    st.divider()

    pdf_file = st.file_uploader("Upload PDF drawing(s)", type=["pdf"])
    cta1, cta2 = st.columns([1,1])
    with cta1:
        run = st.button("▶️ Run Audit", type="primary")
    with cta2:
        if st.button("🧹 Clear Metadata"):
            for k in ["client","project","site_type","vendor","cabinet_loc","radio_loc","sectors","supplier","drawing_type"]:
                st.session_state.pop(k, None)
            st.rerun()

    rules = load_rules(RULES_PATH)
    allow = load_allowlist()

    if run:
        # Validate required metadata
        required = {
            "Client": client, "Project": project, "Site Type": site_type, "Vendor": vendor,
            "Cabinet Location": cabinet_loc, "Radio Location": radio_loc, "Sectors": sectors,
            "Supplier": supplier, "Drawing Type": drawing_type
        }
        missing = [k for k,v in required.items() if not v]
        if missing:
            st.error("Please complete all required metadata: " + ", ".join(missing))
            st.stop()

        if not pdf_file:
            st.error("Please upload a PDF to audit.")
            st.stop()

        pdf_bytes = pdf_file.read()
        pages_text, pages_words = extract_pages_and_index(pdf_bytes)
        if not pages_text:
            st.error("Could not read PDF (PyMuPDF missing or empty).")
            st.stop()

        meta = {
            "client": client, "project": project, "site_type": site_type, "vendor": vendor,
            "cabinet_loc": cabinet_loc, "radio_loc": radio_loc, "sectors": sectors,
            "mimo_config": mimo_config, "site_address": site_address,
            "supplier": supplier, "drawing_type": drawing_type
        }

        findings = apply_rule_sets(pages_text, rules, meta)
        findings = attach_bboxes_to_findings(findings, pages_words, pages_text)

        # Build outputs
        excel_bytes, annotated_pdf, outcome = build_excel_and_pdf(pdf_file.name, pdf_bytes, findings, meta)

        # Save history
        push_history(pdf_file.name, findings, outcome, pages=len(pages_text), meta=meta)

        # Show summary
        st.success(f"Audit complete: **{outcome}** — {len(findings)} findings")
        df = pd.DataFrame(findings or [{"category":"","rule":"","severity":"","message":"","page":""}])
        st.dataframe(df, use_container_width=True)

        # Downloads
        st.download_button("⬇️ Download Excel Report", data=excel_bytes, file_name=f"{Path(pdf_file.name).stem}_{outcome}_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("⬇️ Download Annotated PDF", data=annotated_pdf, file_name=f"{Path(pdf_file.name).stem}_ANNOTATED_{outcome}_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf")

# -----------------------
# TAB 2 — History & Analytics
# -----------------------
with tabs[1]:
    st.subheader("History & Analytics")
    _ensure_history()
    try:
        dfh = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp_utc"])
    except Exception:
        dfh = pd.DataFrame()
    dfh = ensure_history_columns(dfh)

    if dfh.empty:
        st.info("No history yet.")
    else:
        st.dataframe(dfh.sort_values("timestamp_utc", ascending=False).head(50), use_container_width=True)

        st.markdown("### Trends (excluded runs hidden)")
        df_use = dfh.loc[~dfh["exclude"].astype(bool)].copy()
        if df_use.empty:
            st.info("Nothing to chart (all runs excluded).")
        else:
            # RFT% line
            st.line_chart(df_use.set_index("timestamp_utc")["rft_percent"])
            # Minor/Major by supplier
            agg = df_use.groupby("supplier")[["minor_findings","major_findings"]].sum().sort_values("major_findings", ascending=False)
            st.bar_chart(agg)

        st.markdown("### Manage Exclusions")
        view = dfh.sort_values("timestamp_utc", ascending=False).head(25).copy()
        view = ensure_history_columns(view)
        edited = st.data_editor(
            view[["timestamp_utc","file","client","project","supplier","drawing_type","outcome","total_findings","exclude"]],
            num_rows="fixed", use_container_width=True
        )
        if st.button("Save history changes"):
            key = ["timestamp_utc","file"]
            merged = dfh.merge(edited[key + ["exclude"]], on=key, how="left", suffixes=("","_new"))
            merged["exclude"] = merged["exclude_new"].fillna(merged["exclude"]).astype(bool)
            merged.drop(columns=["exclude_new"], inplace=True)
            merged.to_csv(HISTORY_PATH, index=False)
            st.success("Saved. Refresh to see updated analytics.")

# -----------------------
# TAB 3 — Settings (Admin)
# -----------------------
with tabs[2]:
    st.subheader("Admin Access")
    pwd = st.text_input("Enter admin password to edit rules / allowlist / logo", type="password", placeholder="••••••••")
    if st.button("Unlock"):
        if pwd == ADMIN_PASS:
            st.session_state["admin_ok"] = True
            st.success("Settings unlocked.")
        else:
            st.session_state["admin_ok"] = False
            st.error("Incorrect password.")

    if not is_admin_unlocked():
        st.info("Settings are locked. Enter the admin password to proceed.")
        st.stop()

    st.markdown("### Logo")
    st.caption("Provide a file path relative to the repo root (e.g., `88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png`).")
    lp = st.text_input("Logo Path", value=st.session_state.get("logo_path",""))
    if st.button("Apply Logo"):
        st.session_state["logo_path"] = lp
        if read_logo_bytes(lp):
            st.success("Logo loaded.")
        else:
            st.warning("Logo file not found at that path.")
        st.rerun()

    st.divider()
    st.markdown("### Rules (rules_example.yaml)")
    current = ""
    if os.path.exists(RULES_PATH):
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            current = f.read()
    text = st.text_area("Edit YAML rules", value=current, height=380)
    if st.button("Save Rules"):
        try:
            save_rules(RULES_PATH, text)
            st.success("Rules saved.")
        except Exception as e:
            st.error(f"YAML error: {e}")

    st.markdown("### Quick Rule Builder")
    with st.form("quick_rule"):
        qr_scope = st.selectbox("Scope", ["checklist","clients","projects","site_types","vendors","cabinet_locations","radio_locations","sectors","suppliers","drawing_types"])
        qr_key = st.text_input("Key (ignored for checklist; for others use the exact value e.g. 'Vodafone', 'Indoor')")
        qr_name = st.text_input("Rule name (checklist only)", value="")
        qr_sev = st.selectbox("Severity (checklist only)", ["minor","major"])
        qr_must = st.text_input("must_contain (comma-separated)")
        qr_rej = st.text_input("reject_if_present (comma-separated)")
        submitted = st.form_submit_button("Add / Update Rule")
        if submitted:
            try:
                data = load_rules(RULES_PATH)
                must = [x.strip() for x in qr_must.split(",") if x.strip()]
                rej = [x.strip() for x in qr_rej.split(",") if x.strip()]
                if qr_scope == "checklist":
                    lst = data.get("checklist", []) or []
                    lst.append({"name": qr_name or "Checklist", "severity": qr_sev, "must_contain": must, "reject_if_present": rej})
                    data["checklist"] = lst
                else:
                    bucket = data.get(qr_scope, {}) or {}
                    node = bucket.get(qr_key, {}) or {}
                    node["must_contain"] = list(sorted(set((node.get("must_contain") or []) + must)))
                    node["reject_if_present"] = list(sorted(set((node.get("reject_if_present") or []) + rej)))
                    bucket[qr_key] = node
                    data[qr_scope] = bucket
                with open(RULES_PATH, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
                st.success("Rule added/updated.")
            except Exception as e:
                st.error(f"Could not update: {e}")

    st.divider()
    st.markdown("### Allowlist (spelling exceptions)")
    words = sorted(list(load_allowlist()))
    edit = st.text_area("One word per line", value="\n".join(words), height=160)
    if st.button("Save Allowlist"):
        save_allowlist(edit.splitlines())
        st.success("Allowlist saved.")
