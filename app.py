# app.py
import os, io, re, json, uuid, base64, textwrap
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import pandas as pd
import yaml
import fitz  # PyMuPDF
from rapidfuzz import process, fuzz

# --------------------------------------------------------------------------------------
# CONSTANTS / PATHS
# --------------------------------------------------------------------------------------
APP_TITLE = "AI Design Quality Auditor"
HISTORY_CSV = "history.csv"
EXPORT_DIR = "exports"
DEFAULT_RULES_FILE = "rules_example.yaml"

ENTRY_PASSWORD = "Seker123"        # Gate to enter app
ADMIN_PASSWORD = "vanB3lkum21"     # Unlock Settings → Rules editing

# Suppliers (used for metadata + analytics filters; extend anytime)
DEFAULT_SUPPLIERS = [
    "WFS", "Circet", "Telent", "AlanDick", "Cellnex", "Cornerstone",
    "BT", "Vodafone", "MBNL", "H3G", "BTEE"
]

# Clients / Projects / etc. (can be extended in Settings)
DEFAULT_CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
DEFAULT_PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
DEFAULT_SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
DEFAULT_VENDORS = ["Ericsson", "Nokia"]
DEFAULT_RADIO_LOCATIONS = ["Low Level", "Midway", "High Level", "Unique Coverage", "Indoor"]
DEFAULT_CABINET_LOCATIONS = ["Indoor", "Outdoor"]
DEFAULT_DRAWING_TYPES = ["General Arrangement", "Detailed Design"]
DEFAULT_SECTOR_QTY = [1, 2, 3, 4, 5, 6]

# Compact, curated MIMO options (you can paste bigger list if needed)
DEFAULT_MIMO_OPTIONS = sorted(list({
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
    "18\\21 @4x4; 70\\80 @2x2",
    "18\\21 @4x4; 70\\80 @2x4",
    "18\\21 @4x4; 70\\80 @2x4; 3500 @32x32",
    "18\\21\\26 @2x2",
    "18\\21\\26 @2x2; 70\\80 @2x2",
    "18\\21\\26 @4x4",
    "18\\21\\26 @4x4; 70\\80 @2x4",
    "18\\21\\26 @4x4; 70\\80 @2x4; 3500 @32x32",
    "18\\21\\26 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21\\26 @4x4; 3500 @32x32",
    "18\\21\\26 @4x4; 3500 @8x8",
    "18\\26 @2x2",
    "18\\26 @4x4; 21 @2x2; 80 @2x2",
    "18\\21 @4x4; 70 @2x4",
    "18\\21 @4x4; 3500 @32x32",
}))

DEFAULT_UI_VISIBILITY_HOURS = 24  # How long results are pinned in UI (analytics uses history anyway)

# --------------------------------------------------------------------------------------
# UTILS / I/O
# --------------------------------------------------------------------------------------
def ensure_dirs():
    os.makedirs(EXPORT_DIR, exist_ok=True)

def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def write_text_file(path: str, data: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def b64_of_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

def uuid_str() -> str:
    return str(uuid.uuid4())

# --------------------------------------------------------------------------------------
# RULES LOADING / SAVING
# --------------------------------------------------------------------------------------
def safe_yaml_load(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return {}
            return data
    except Exception:
        return {}

def safe_yaml_dump(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp, path)

def load_rules(path: str) -> dict:
    data = safe_yaml_load(path)
    # Ensure base structure
    data.setdefault("allowlist", [])     # words that should never be flagged as spelling errors
    data.setdefault("denylist", [])      # patterns that are always errors
    data.setdefault("phrase_rules", [])  # list of {id, when, message, severity, patterns: [ {must:any_of/all_of, not:any_of}, scope:"text|title|..."} ]
    data.setdefault("mappings", {})      # quick key->value corrections (e.g., "AHEGC"->"AHEGG")
    data.setdefault("metadata_rules", {})# per-metadata rules (client/project/vendor/site_type/…)
    data.setdefault("address_match_title", True)  # enforces Site Address must appear in title
    data.setdefault("spelling", {"enabled": True})
    return data

def save_rules(path: str, data: dict):
    safe_yaml_dump(path, data)

def get_vocab_from_rules(rules: dict) -> List[str]:
    vocab = set()
    for w in rules.get("allowlist", []):
        vocab.add(str(w).strip())
    # You can extend here with project/client/vendor specific words later
    return sorted([v for v in vocab if v])

# --------------------------------------------------------------------------------------
# HISTORY (ROBUST)
# --------------------------------------------------------------------------------------
def load_history() -> pd.DataFrame:
    """Load history CSV safely and normalize schema."""
    cols = [
        "timestamp_utc","reviewer","supplier","client","project","site_type",
        "vendor","radio_location","cabinet_location","sectors_qty",
        "mimo_s1","mimo_s2","mimo_s3","mimo_s4","mimo_s5","mimo_s6",
        "site_address","drawing_type","status","pdf_name","excel_name",
        "annot_pdf_name","exclude","right_first_time","notes","run_id"
    ]
    try:
        if os.path.exists(HISTORY_CSV):
            try:
                df = pd.read_csv(HISTORY_CSV, dtype=str)
            except Exception:
                df = pd.read_csv(HISTORY_CSV, dtype=str, engine="python", on_bad_lines="skip")
        else:
            df = pd.DataFrame(columns=cols)
    except Exception:
        df = pd.DataFrame(columns=cols)

    for c in cols:
        if c not in df.columns:
            df[c] = ""
    # Types
    df["exclude"] = df["exclude"].astype(str).str.lower().isin(["true","1","yes"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    df["right_first_time"] = pd.to_numeric(df["right_first_time"], errors="coerce")
    return df[cols]

def append_history_row(meta: dict, status: str, pdf_name: str, excel_name: str,
                       annot_pdf_name: str = "", notes: str = "", rft: float | None = None,
                       run_id: str = ""):
    df = load_history()
    row = {
        "timestamp_utc": pd.Timestamp.utcnow(),
        "reviewer": meta.get("reviewer",""),
        "supplier": meta.get("supplier",""),
        "client": meta.get("client",""),
        "project": meta.get("project",""),
        "site_type": meta.get("site_type",""),
        "vendor": meta.get("vendor",""),
        "radio_location": meta.get("radio_location",""),
        "cabinet_location": meta.get("cabinet_location",""),
        "sectors_qty": str(meta.get("sectors_qty","")),
        "mimo_s1": meta.get("mimo_s1",""),
        "mimo_s2": meta.get("mimo_s2",""),
        "mimo_s3": meta.get("mimo_s3",""),
        "mimo_s4": meta.get("mimo_s4",""),
        "mimo_s5": meta.get("mimo_s5",""),
        "mimo_s6": meta.get("mimo_s6",""),
        "site_address": meta.get("site_address",""),
        "drawing_type": meta.get("drawing_type",""),
        "status": status,
        "pdf_name": pdf_name,
        "excel_name": excel_name,
        "annot_pdf_name": annot_pdf_name,
        "exclude": bool(meta.get("exclude_from_analytics", False)),
        "right_first_time": rft if rft is not None else "",
        "notes": notes,
        "run_id": run_id,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    tmp = HISTORY_CSV + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, HISTORY_CSV)

def safe_select(df: pd.DataFrame, wanted: List[str]) -> pd.DataFrame:
    have = [c for c in wanted if c in df.columns]
    for c in wanted:
        if c not in df.columns:
            df[c] = ""
    return df[wanted]

def filter_history_for_ui(df: pd.DataFrame, hours: int = DEFAULT_UI_VISIBILITY_HOURS) -> pd.DataFrame:
    cutoff = now_utc() - timedelta(hours=hours)
    # guard if timestamp is NaT
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    return df[df["timestamp_utc"].fillna(pd.Timestamp.utcnow()).ge(cutoff)]

# --------------------------------------------------------------------------------------
# PDF INGEST / ANNOTATE (PyMuPDF-only)
# --------------------------------------------------------------------------------------
def extract_text_and_words(pdf_bytes: bytes) -> Tuple[List[str], List[List[dict]]]:
    """
    Returns:
      pages_text: list of page plain text strings
      pages_words: per page list of words as dicts: {text, bbox(x0,y0,x1,y1), page}
    """
    pages_text = []
    pages_words = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for pno in range(len(doc)):
        page = doc[pno]
        pages_text.append(page.get_text("text"))
        words = []
        for w in page.get_text("words"):  # (x0, y0, x1, y1, "text", block_no, line_no, word_no)
            x0, y0, x1, y1, wtext, *_ = w
            words.append({"text": wtext, "bbox": (x0, y0, x1, y1), "page": pno+1})
        pages_words.append(words)
    doc.close()
    return pages_text, pages_words

def annotate_pdf(original_pdf: bytes, findings: List[dict]) -> bytes:
    """
    findings: list of dict with keys: page(int), bbox(tuple|None), message(str), severity(str)
    """
    doc = fitz.open(stream=original_pdf, filetype="pdf")
    for f in findings:
        page_no = f.get("page")
        if not page_no or page_no < 1 or page_no > len(doc):
            continue
        page = doc[page_no-1]
        bbox = f.get("bbox")
        msg = f.get("message","")
        if bbox and isinstance(bbox, (list,tuple)) and len(bbox)==4:
            rect = fitz.Rect(*bbox)
            page.add_rect_annot(rect)
            # Add a callout note nearby
            note = page.add_note(rect.tl, text=msg)
            note.set_info(info={"title": f.get("rule_id","Finding")})
        else:
            # place a sticky note in the top-left margin
            note = page.add_note(page.rect.tl + (20, 20), text=msg)
            note.set_info(info={"title": f.get("rule_id","Finding")})
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()

# --------------------------------------------------------------------------------------
# SPELLING & GENERIC FINDINGS
# --------------------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']{1,}", text)

def spelling_checks(pages_text: List[str], rules: dict) -> List[dict]:
    findings = []
    cfg = rules.get("spelling", {"enabled": True})
    if not cfg.get("enabled", True):
        return findings

    allow = set([w.lower() for w in rules.get("allowlist", [])])
    # build a vocabulary from allowlist (used for suggestions)
    vocab = get_vocab_from_rules(rules)
    vocab_lower = [v.lower() for v in vocab]

    # very light heuristic: flag words with digits+letters mixed or very long sequences
    for pidx, t in enumerate(pages_text, start=1):
        tokens = tokenize(t)
        for w in tokens:
            wl = w.lower()
            # skip if in allow
            if wl in allow:
                continue
            # skip if typical unit-like tokens, small words, etc.
            if len(wl) <= 2:
                continue
            if re.search(r"\d", wl):
                continue
            # basic suggestion from vocab if present
            suggestion = None
            if vocab_lower:
                m = process.extractOne(wl, vocab_lower, scorer=fuzz.WRatio)
                if m and m[1] >= 92:
                    suggestion = vocab[vocab_lower.index(m[0])]
            findings.append({
                "page": pidx,
                "bbox": None,  # we annotate generic notes unless mapped via word boxes (later)
                "rule_id": "SPELLING",
                "message": f"Possible spelling issue: '{w}'" + (f" → '{suggestion}'" if suggestion else ""),
                "severity": "Minor",
                "category": "Spelling",
            })
    return findings

def apply_mappings(text: str, rules: dict) -> List[dict]:
    """Mappings like AHEGC -> AHEGG flagged when found."""
    findings = []
    maps = rules.get("mappings", {})
    if not maps:
        return findings
    for wrong, right in maps.items():
        if not wrong:
            continue
        for m in re.finditer(re.escape(wrong), text, flags=re.IGNORECASE):
            findings.append({
                "page": 1,  # refined later if per-page search is needed; kept simple
                "bbox": None,
                "rule_id": "MAP:" + wrong,
                "message": f"Use '{right}' instead of '{wrong}'.",
                "severity": "Minor",
                "category": "Mappings",
            })
    return findings

def find_phrase_rules(pages_text: List[str], rules: dict, meta: dict) -> List[dict]:
    """Evaluate phrase_rules with optional metadata filters."""
    out = []
    phrase_rules = rules.get("phrase_rules", [])
    for pr in phrase_rules:
        when = pr.get("when", {})
        # check metadata gating
        ok = True
        for k, v in when.items():
            if not str(meta.get(k,"")).strip():
                ok = False
                break
            if str(meta.get(k,"")).strip() != str(v).strip():
                ok = False
                break
        if not ok:
            continue

        message = pr.get("message", "Rule triggered")
        severity = pr.get("severity","Minor")
        scope = pr.get("scope","text")  # "text" or "title"
        pats: List[dict] = pr.get("patterns", [])

        # Build search source
        source_pages = pages_text
        if scope == "title":
            # Use first page text as title area
            source_pages = [pages_text[0] if pages_text else ""]

        for pidx, page_text in enumerate(source_pages, start=1):
            page_ok = True
            for pat in pats:
                must_any = pat.get("any_of", [])
                all_of = pat.get("all_of", [])
                not_any = pat.get("not_any", [])

                if must_any:
                    if not any(re.search(x, page_text, re.IGNORECASE) for x in must_any):
                        page_ok = False
                        break
                if all_of:
                    if not all(re.search(x, page_text, re.IGNORECASE) for x in all_of):
                        page_ok = False
                        break
                if not_any:
                    if any(re.search(x, page_text, re.IGNORECASE) for x in not_any):
                        page_ok = False
                        break
            if page_ok:
                out.append({
                    "page": pidx,
                    "bbox": None,
                    "rule_id": pr.get("id","PHRASE_RULE"),
                    "message": message,
                    "severity": severity,
                    "category": "Phrase",
                })
    return out

def site_address_vs_title(pages_text: List[str], site_address: str, enabled: bool) -> List[dict]:
    if not enabled or not pages_text:
        return []
    title_text = pages_text[0]
    # The user asked: if address contains ", 0 ," ignore the zero
    addr_norm = re.sub(r",\s*0\s*,", ",", site_address, flags=re.IGNORECASE).strip()
    # Check phrase-by-phrase presence (loose contains)
    ok = addr_norm and (addr_norm.lower() in title_text.lower())
    if ok:
        return []
    return [{
        "page": 1,
        "bbox": None,
        "rule_id": "ADDRESS_TITLE_MISMATCH",
        "message": f"Site Address not found in title: '{addr_norm}'.",
        "severity": "Major",
        "category": "Metadata",
    }]

# --------------------------------------------------------------------------------------
# AUDIT PIPELINE
# --------------------------------------------------------------------------------------
def run_audit(pdf_bytes: bytes, meta: dict, rules: dict) -> List[dict]:
    pages_text, pages_words = extract_text_and_words(pdf_bytes)

    findings: List[dict] = []
    # 1) basic mappings (like AHEGC -> AHEGG)
    for pidx, t in enumerate(pages_text, start=1):
        for f in apply_mappings(t, rules):
            f["page"] = pidx
            findings.append(f)

    # 2) spelling checks (lightweight)
    findings += spelling_checks(pages_text, rules)

    # 3) phrase rules
    findings += find_phrase_rules(pages_text, rules, meta)

    # 4) address in title check
    findings += site_address_vs_title(pages_text, meta.get("site_address",""), rules.get("address_match_title", True))

    # You can append more domain rules here (mimo consistency across sectors, vendor combos, etc.)

    # Attach rough bbox if we can match exact words (best-effort)
    # Map each finding message first found token back to a word bbox on the page
    for f in findings:
        if f.get("bbox"):
            continue
        page = f.get("page", 1)
        msg = f.get("message","")
        m = re.findall(r"'([^']+)'", msg)
        if m and 1 <= page <= len(pages_words):
            target = m[0]
            for wd in pages_words[page-1]:
                if wd["text"].lower() == target.lower():
                    f["bbox"] = wd["bbox"]
                    break
    return findings

# --------------------------------------------------------------------------------------
# REPORTS (Excel + Annotated PDF)
# --------------------------------------------------------------------------------------
def make_excel(findings: List[dict], meta: dict, original_pdf_name: str, status: str) -> bytes:
    df = pd.DataFrame(findings if findings else [])
    if df.empty:
        df = pd.DataFrame(columns=["page","rule_id","severity","category","message","x0","y0","x1","y1"])
    else:
        # split bbox into columns
        df["x0"] = df["bbox"].apply(lambda b: b[0] if isinstance(b, (list,tuple)) and len(b)==4 else "")
        df["y0"] = df["bbox"].apply(lambda b: b[1] if isinstance(b, (list,tuple)) and len(b)==4 else "")
        df["x1"] = df["bbox"].apply(lambda b: b[2] if isinstance(b, (list,tuple)) and len(b)==4 else "")
        df["y1"] = df["bbox"].apply(lambda b: b[3] if isinstance(b, (list,tuple)) and len(b)==4 else "")

    meta_df = pd.DataFrame([meta])

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        meta_df.to_excel(xw, index=False, sheet_name="Audit_Metadata")
        df.to_excel(xw, index=False, sheet_name="Findings")
        summary = pd.DataFrame([{
            "status": status,
            "findings_count": len(findings),
            "timestamp_utc": datetime.utcnow().isoformat()
        }])
        summary.to_excel(xw, index=False, sheet_name="Summary")
    return out.getvalue()

def persist_export_files(base_name_no_ext: str, excel_bytes: bytes, annot_pdf_bytes: bytes) -> Tuple[str,str]:
    ensure_dirs()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    excel_name = f"{base_name_no_ext}__{ts}.xlsx"
    pdf_name   = f"{base_name_no_ext}__{ts}.pdf"
    excel_path = os.path.join(EXPORT_DIR, excel_name)
    pdf_path   = os.path.join(EXPORT_DIR, pdf_name)
    with open(excel_path, "wb") as f:
        f.write(excel_bytes)
    with open(pdf_path, "wb") as f:
        f.write(annot_pdf_bytes)
    return excel_name, pdf_name

# --------------------------------------------------------------------------------------
# UI HELPERS
# --------------------------------------------------------------------------------------
def gate_with_password() -> bool:
    c1, c2 = st.columns([1,3])
    with c1:
        show_logo()
    with c2:
        st.title(APP_TITLE)
        st.caption("Secure Access")
        pwd = st.text_input("Enter access password", type="password", key="entry_pw")
        ok = st.button("Enter", use_container_width=True)
        if ok:
            if pwd.strip() == ENTRY_PASSWORD:
                st.session_state["authed"] = True
                return True
            else:
                st.error("Incorrect password.")
    return st.session_state.get("authed", False)

def show_logo():
    logo_path = st.session_state.get("logo_path", "")
    b64 = b64_of_file(logo_path)
    if b64:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:8px;">
                <img src="data:image/png;base64,{b64}" style="height:50px;object-fit:contain;border-radius:6px;" />
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.caption("🔖 Add logo in Settings → Branding")

def meta_form() -> dict:
    st.subheader("Audit Metadata")
    c0, c1, c2, c3 = st.columns(4)
    with c0:
        reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer",""))
    with c1:
        supplier = st.selectbox("Supplier", options=sorted(set(DEFAULT_SUPPLIERS + list(load_history()["supplier"].dropna()))), index=0 if DEFAULT_SUPPLIERS else None)
    with c2:
        client = st.selectbox("Client", options=DEFAULT_CLIENTS)
    with c3:
        project = st.selectbox("Project", options=DEFAULT_PROJECTS)

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        site_type = st.selectbox("Site Type", options=DEFAULT_SITE_TYPES)
    with c5:
        vendor = st.selectbox("Vendor", options=DEFAULT_VENDORS)
    with c6:
        radio_location = st.selectbox("Radio Location", options=DEFAULT_RADIO_LOCATIONS)
    with c7:
        cabinet_location = st.selectbox("Cabinet Location", options=DEFAULT_CABINET_LOCATIONS)

    c8, c9 = st.columns([1,3])
    with c8:
        sectors_qty = st.selectbox("Quantity of Sectors", options=DEFAULT_SECTOR_QTY, index=2)  # default 3
    with c9:
        site_address = st.text_input("Site Address (must match title unless '0' placeholder present)")

    # Drawings
    drawing_type = st.selectbox("Drawing Type", options=DEFAULT_DRAWING_TYPES)

    st.markdown("---")
    st.markdown("#### Proposed MIMO Config (per sector)")
    hide_mimo = (project.strip().lower() == "power resilience")
    if hide_mimo:
        st.info("Proposed MIMO Config is optional for Power Resilience.", icon="ℹ️")

    use_s1_for_all = st.checkbox("Use S1 for all sectors", value=True)

    mimo_s1 = st.selectbox("MIMO S1", options=DEFAULT_MIMO_OPTIONS)
    def mimo_sel(label, default):
        return st.selectbox(label, options=DEFAULT_MIMO_OPTIONS, index=DEFAULT_MIMO_OPTIONS.index(default) if default in DEFAULT_MIMO_OPTIONS else 0, disabled=use_s1_for_all)

    mimo_values = {"mimo_s1": mimo_s1}
    for i in range(2, sectors_qty+1):
        default = mimo_s1 if use_s1_for_all else DEFAULT_MIMO_OPTIONS[0]
        mimo_values[f"mimo_s{i}"] = mimo_sel(f"MIMO S{i}", default)

    exclude_from_analytics = st.checkbox("Exclude this run from analytics", value=False)

    meta = {
        "reviewer": reviewer,
        "supplier": supplier,
        "client": client,
        "project": project,
        "site_type": site_type,
        "vendor": vendor,
        "radio_location": radio_location,
        "cabinet_location": cabinet_location,
        "sectors_qty": sectors_qty,
        "site_address": site_address,
        "drawing_type": drawing_type,
        "exclude_from_analytics": exclude_from_analytics,
    }
    meta.update(mimo_values)
    return meta

def training_panel(rules_path: str, rules: dict, findings: List[dict]):
    st.subheader("Training / Rapid Learning")
    st.caption("Mark false positives as **Not Valid** (will append to allowlist / phrase exceptions), add quick mappings, or bulk-apply via uploaded review CSV.")

    if findings:
        df = pd.DataFrame(findings)
        edit_df = st.data_editor(
            df[["page","rule_id","severity","category","message"]],
            hide_index=True,
            use_container_width=True,
            key="training_editor"
        )
        col_a, col_b, col_c = st.columns([1,1,2])
        with col_a:
            add_to_allow = st.text_input("Add word to allowlist (never flag)", placeholder="e.g., AHEGG")
            if st.button("➕ Add to Allowlist"):
                if add_to_allow:
                    rules.setdefault("allowlist", [])
                    if add_to_allow not in rules["allowlist"]:
                        rules["allowlist"].append(add_to_allow)
                        save_rules(rules_path, rules)
                        st.success(f"Added '{add_to_allow}' to allowlist.")
        with col_b:
            wrong = st.text_input("Mapping: wrong", placeholder="AHEGC")
            right = st.text_input("Mapping: correct", placeholder="AHEGG")
            if st.button("➕ Add Mapping"):
                if wrong and right:
                    rules.setdefault("mappings", {})
                    rules["mappings"][wrong] = right
                    save_rules(rules_path, rules)
                    st.success(f"Mapping added: {wrong} → {right}")
        with col_c:
            st.caption("Bulk: Upload review CSV with columns like rule_id, decision(Valid/Not Valid), note")
            up = st.file_uploader("Upload Review CSV", type=["csv"], key="train_csv")
            if up is not None:
                try:
                    tdf = pd.read_csv(up)
                    # For any Not Valid - try to pull quoted tokens and add to allowlist
                    rules.setdefault("allowlist", [])
                    for _, r in tdf.iterrows():
                        dec = str(r.get("decision","")).strip().lower()
                        if dec == "not valid":
                            msg = str(r.get("message",""))
                            quoted = re.findall(r"'([^']+)'", msg)
                            for q in quoted:
                                if q and q not in rules["allowlist"]:
                                    rules["allowlist"].append(q)
                    save_rules(rules_path, rules)
                    st.success("Bulk review ingested and allowlist updated.")
                except Exception as e:
                    st.error(f"Could not parse CSV: {e}")
    else:
        st.info("Run an audit to enable training on its findings.")

def analytics_tab():
    st.subheader("📊 Analytics")
    dfh = load_history()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        supplier_filter = st.selectbox("Supplier (filter)", ["All"] + sorted([s for s in dfh["supplier"].dropna().unique() if s!=""]))
    with c2:
        client_filter = st.selectbox("Client (filter)", ["All"] + sorted([s for s in dfh["client"].dropna().unique() if s!=""]))
    with c3:
        project_filter = st.selectbox("Project (filter)", ["All"] + sorted([s for s in dfh["project"].dropna().unique() if s!=""]))
    with c4:
        include_excluded = st.toggle("Include excluded from analytics", value=False)

    show = dfh.copy()
    if not include_excluded:
        show = show[~show["exclude"].fillna(False)]

    if supplier_filter != "All":
        show = show[show["supplier"] == supplier_filter]
    if client_filter != "All":
        show = show[show["client"] == client_filter]
    if project_filter != "All":
        show = show[show["project"] == project_filter]

    show = show.sort_values("timestamp_utc", ascending=False)

    total = len(show)
    rejected = int((show["status"].str.upper() == "REJECTED").sum())
    passed = int((show["status"].str.upper() == "PASS").sum())
    rft = show["right_first_time"].dropna()
    rft_avg = float(rft.mean()) if not rft.empty else None

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Audits", total)
    k2.metric("Pass / Rejected", f"{passed} / {rejected}")
    k3.metric("Avg Right-First-Time %", f"{rft_avg:.1f}%" if rft_avg is not None else "—")

    st.dataframe(
        safe_select(show, ["timestamp_utc","supplier","client","project","status","pdf_name","excel_name","annot_pdf_name"]),
        use_container_width=True,
        hide_index=True
    )

def settings_tab(rules_path: str, rules: dict):
    st.subheader("Settings")
    st.caption("Admin password required to edit rules or branding.")
    apw = st.text_input("Admin password", type="password", key="admin_pw")
    locked = apw.strip() != ADMIN_PASSWORD

    with st.expander("Branding & UI", expanded=True):
        if locked:
            st.info("Enter admin password to modify.", icon="🔒")
        logo_path = st.text_input("Logo file path (repo root)", value=st.session_state.get("logo_path",""), disabled=locked)
        if not locked and st.button("Save Branding"):
            st.session_state["logo_path"] = logo_path
            st.success("Branding updated.")

        st.number_input("Keep results visible in UI (hours)", min_value=1, max_value=336,
                        value=st.session_state.get("ui_visibility_hours", DEFAULT_UI_VISIBILITY_HOURS),
                        key="ui_visibility_hours", disabled=locked)

    with st.expander("Rules: Global Flags", expanded=True):
        address_match = st.checkbox("Require Site Address to appear in title", value=rules.get("address_match_title", True), disabled=locked)
        spelling_en = st.checkbox("Enable Spelling Check", value=rules.get("spelling",{}).get("enabled", True), disabled=locked)
        if not locked and st.button("Save Global Flags"):
            rules["address_match_title"] = bool(address_match)
            rules.setdefault("spelling", {})["enabled"] = bool(spelling_en)
            save_rules(rules_path, rules)
            st.success("Global flags saved.")

    with st.expander("Rules: Allowlist & Mappings", expanded=False):
        if locked:
            st.info("Enter admin password to modify.", icon="🔒")
        allowlist_txt = "\n".join(rules.get("allowlist", []))
        mappings_txt = "\n".join([f"{k}=>{v}" for k,v in rules.get("mappings", {}).items()])
        new_allow = st.text_area("Allowlist (one per line)", value=allowlist_txt, height=160, disabled=locked)
        new_maps = st.text_area("Mappings (wrong=>right, one per line)", value=mappings_txt, height=160, disabled=locked)
        if not locked and st.button("Save Allowlist & Mappings"):
            rules["allowlist"] = [x.strip() for x in new_allow.splitlines() if x.strip()]
            maps = {}
            for line in new_maps.splitlines():
                if "=>" in line:
                    a,b = line.split("=>",1)
                    maps[a.strip()] = b.strip()
            rules["mappings"] = maps
            save_rules(rules_path, rules)
            st.success("Allowlist & Mappings saved.")

    with st.expander("Rules: Phrase Rules (YAML raw)", expanded=False):
        st.caption("Advanced users can edit the raw YAML rules.")
        raw = yaml.safe_dump(rules, sort_keys=False, allow_unicode=True)
        new_raw = st.text_area("rules_example.yaml (full)", value=raw, height=400, disabled=locked, help="Be careful with YAML syntax.")
        if not locked and st.button("Save YAML"):
            try:
                parsed = yaml.safe_load(new_raw) or {}
                save_rules(rules_path, parsed)
                st.success("Rules YAML saved.")
            except Exception as e:
                st.error(f"YAML error: {e}")

# --------------------------------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")
    ensure_dirs()

    # Gate
    if not gate_with_password():
        st.stop()

    # Header row with logo
    h1, h2 = st.columns([1,6], vertical_alignment="center")
    with h1:
        show_logo()
    with h2:
        st.title(APP_TITLE)

    # Sidebar: audit controls
    with st.sidebar:
        st.header("Controls")
        clear_meta = st.button("Clear Metadata", use_container_width=True)
        st.session_state.setdefault("reviewer","")
        if clear_meta:
            # reset only metadata keys
            for k in list(st.session_state.keys()):
                if k.startswith("mimo_") or k in ("reviewer","entry_pw"):
                    st.session_state[k] = ""

    rules_path = DEFAULT_RULES_FILE
    if not os.path.exists(rules_path):
        # seed a minimal file
        seed = {
            "allowlist": [],
            "mappings": {},
            "phrase_rules": [],
            "metadata_rules": {},
            "address_match_title": True,
            "spelling": {"enabled": True}
        }
        save_rules(rules_path, seed)

    rules = load_rules(rules_path)

    # Tabs
    tab_audit, tab_training, tab_analytics, tab_settings = st.tabs(["🔍 Audit", "🧠 Training", "📊 Analytics", "⚙️ Settings"])

    with tab_audit:
        meta = meta_form()
        st.markdown("---")
        up = st.file_uploader("Upload PDF (unlimited size; DWG optional later)", type=["pdf"], accept_multiple_files=False)
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            audit_clicked = st.button("▶️ Run Audit", type="primary", use_container_width=True, disabled=(up is None))
        with c2:
            pass

        if up and audit_clicked:
            pdf_bytes = up.read()
            # Guard: require all metadata filled
            missing = [k for k,v in meta.items() if v in ("", None) and k not in ("mimo_s2","mimo_s3","mimo_s4","mimo_s5","mimo_s6")]
            if missing:
                st.error(f"Please complete all metadata fields before auditing. Missing: {', '.join(missing)}")
                st.stop()

            findings = run_audit(pdf_bytes, meta, rules)
            status = "PASS" if len(findings) == 0 else "REJECTED"

            # Build outputs
            base_no_ext = os.path.splitext(up.name)[0] + f"__{status}"
            excel_bytes = make_excel(findings, meta, up.name, status)
            annot_pdf_bytes = annotate_pdf(pdf_bytes, findings)

            # Persist (these remain on disk; UI shows last 24h by default)
            excel_name, pdf_name = persist_export_files(base_no_ext, excel_bytes, annot_pdf_bytes)

            # History
            append_history_row(meta, status, pdf_name=pdf_name, excel_name=excel_name, annot_pdf_name=pdf_name, run_id=uuid_str())

            st.success(f"Audit complete: **{status}**")
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("⬇️ Download Excel Report", data=excel_bytes, file_name=f"{base_no_ext}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with dl2:
                st.download_button("⬇️ Download Annotated PDF", data=annot_pdf_bytes, file_name=f"{base_no_ext}.pdf", mime="application/pdf", use_container_width=True)

            st.markdown("#### Findings")
            if findings:
                st.dataframe(pd.DataFrame(findings)[["page","rule_id","severity","category","message"]], use_container_width=True, hide_index=True)
            else:
                st.info("No issues found. ✅")

    with tab_training:
        # Allow training from the last run in session? Instead, let user upload last report's Excel to ingest decisions
        st.info("Tip: **Run an audit**, then use this panel to add allowlist items, mappings or bulk-apply decisions. You can also ingest a previously downloaded Excel report.")
        # If they just ran an audit, we can reuse (not persisted in session here), so ask to upload
        xup = st.file_uploader("Upload a prior Excel Report (Findings sheet) to train from", type=["xlsx"])
        findings_for_training: List[dict] = []
        if xup is not None:
            try:
                xls = pd.ExcelFile(xup)
                fd = pd.read_excel(xls, "Findings")
                # Convert to findings-like dicts
                for _, r in fd.iterrows():
                    findings_for_training.append({
                        "page": int(r.get("page", 1)) if pd.notna(r.get("page")) else 1,
                        "rule_id": str(r.get("rule_id","")),
                        "severity": str(r.get("severity","")),
                        "category": str(r.get("category","")),
                        "message": str(r.get("message","")),
                        "bbox": None
                    })
                st.success(f"Loaded {len(findings_for_training)} findings from report.")
            except Exception as e:
                st.error(f"Could not read report: {e}")
        training_panel(rules_path, rules, findings_for_training)

    with tab_analytics:
        analytics_tab()

    with tab_settings:
        settings_tab(rules_path, rules)

if __name__ == "__main__":
    main()
