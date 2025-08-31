# app.py
import os, io, re, base64, json, datetime
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import pandas as pd
import yaml
from rapidfuzz import fuzz
import fitz  # PyMuPDF

# --------------------------------------------------------------------------------------
# CONSTANTS / CONFIG
# --------------------------------------------------------------------------------------
ENTRY_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"

HISTORY_DIR = "history"
OUTPUT_DIR = "outputs"
HISTORY_CSV = os.path.join(HISTORY_DIR, "audit_history.csv")
RULES_FILE = "rules_example.yaml"

# Try to find a logo automatically
PREFERRED_LOGOS = ["logo.png", "logo.jpg", "logo.jpeg", "logo.svg"]

SUPPLIERS = [
    "CEG", "CTIL", "Emfyser", "Innov8", "Invict", "KTL Team (Internal)", "Trylon"
]
CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
CABINETS = ["Indoor", "Outdoor"]
RADIOS = ["Low Level", "High Level", "Midway", "Unique Coverage"]
DRAWING_TYPES = ["General Arrangement", "Detailed Design"]

# Curated + deduped MIMO list (as provided; backslashes escaped)
MIMO_CONFIGS = [
    "18 @2x2", "18 @2x2; 26 @4x4", "18 @2x2; 70\\80 @2x2", "18 @2x2; 80 @2x2",
    "18\\21 @2x2", "18\\21 @2x2; 26 @4x4", "18\\21 @2x2; 3500 @32x32",
    "18\\21 @2x2; 70\\80 @2x2", "18\\21 @2x2; 80 @2x2",
    "18\\21 @4x4", "18\\21 @4x4; 3500 @32x32", "18\\21 @4x4; 70 @2x4",
    "18\\21 @4x4; 70\\80 @2x2", "18\\21 @4x4; 70\\80 @2x2; 3500 @32x32",
    "18\\21 @4x4; 70\\80 @2x4", "18\\21 @4x4; 70\\80 @2x4; 3500 @32x32",
    "18\\21 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21@4x4; 70\\80 @2x2", "18\\21@4x4; 70\\80 @2x4",
    "18\\21\\26 @2x2", "18\\21\\26 @2x2; 3500 @32x32", "18\\21\\26 @2x2; 3500 @8X8",
    "18\\21\\26 @2x2; 70\\80 @2x2",
    "18\\21\\26 @2x2; 70\\80 @2x2; 3500 @32x32",
    "18\\21\\26 @2x2; 70\\80 @2x2; 3500 @8x8",
    "18\\21\\26 @2x2; 70\\80 @2x4; 3500 @32x32",
    "18\\21\\26 @2x2; 80 @2x2",
    "18\\21\\26 @2x2; 80 @2x2; 3500 @8x8",
    "18\\21\\26 @4x4", "18\\21\\26 @4x4; 3500 @32x32",
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
    "18\\26 @2x2", "18\\26 @4x4; 21 @2x2; 80 @2x2",
    "(blank)"
]

# --------------------------------------------------------------------------------------
# UTIL / IO
# --------------------------------------------------------------------------------------
def ensure_dirs():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def now_utc_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat()

def date_tag() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d")

def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

def find_logo_path(custom: Optional[str] = None) -> Optional[str]:
    if custom and os.path.exists(custom): return custom
    for p in PREFERRED_LOGOS:
        if os.path.exists(p): return p
    return None

# --------------------------------------------------------------------------------------
# RULES
# --------------------------------------------------------------------------------------
DEFAULT_RULES = {
    "schema_version": 2,
    "allowlist": [
        "EE", "BTS", "MBNL", "RAN", "LTE", "NR", "DC", "AC", "kVA",
        "AHEGG", "Polaradium", "Eltek", "PSU"
    ],
    "forbid_pairs": [
        # Example: if "Brush" is present, "Generator Power" must not be used (and vice versa)
        {"if_any": ["Brush"], "forbid_any": ["Generator Power"]},
        {"if_any": ["Generator Power"], "forbid_any": ["Brush"]}
    ],
    "checklist": [
        {
            "id": "power-res-eltek-note",
            "when": {"project": ["Power Resilience"]},
            "must_include": {
                "text": "IMPORTANT NOTE: To support the power resilience configure settings the Eltek PSU will need to be configured as per TDEE53201 section 3.8.1",
                "page": 300  # if page > pages, we search everywhere
            }
        },
        {
            "id": "address-in-title",
            "when": {},
            "address_in_title": True
        }
    ]
}

def load_rules() -> Dict[str, Any]:
    if not os.path.exists(RULES_FILE):
        return DEFAULT_RULES.copy()
    try:
        with open(RULES_FILE, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        data = {}
    # Merge with defaults to ensure keys exist
    data.setdefault("schema_version", 2)
    data.setdefault("allowlist", [])
    data.setdefault("forbid_pairs", [])
    data.setdefault("checklist", [])
    return data

def save_rules(rules: Dict[str, Any]) -> None:
    with open(RULES_FILE, "w") as f:
        yaml.safe_dump(rules, f, sort_keys=False, allow_unicode=True)

# --------------------------------------------------------------------------------------
# PDF TEXT & ANNOTATION
# --------------------------------------------------------------------------------------
def extract_pdf_text_pages(pdf_bytes: bytes) -> List[str]:
    """Use PyMuPDF to get per-page text."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        pages.append(doc[i].get_text("text"))
    doc.close()
    return pages

def find_text_boxes(doc: fitz.Document, needle: str, thresh: int = 80) -> List[Tuple[int, fitz.Rect]]:
    """Return list of (page_index, rect) for blocks similar to needle."""
    hits = []
    needle_up = needle.upper().strip()
    if not needle_up: return hits
    for pi in range(len(doc)):
        page = doc[pi]
        blocks = page.get_text("blocks")
        for b in blocks:
            # b: (x0,y0,x1,y1,"text", block_no, block_type, ...)
            txt = (b[4] or "").strip()
            if not txt: continue
            if fuzz.partial_ratio(needle_up, txt.upper()) >= thresh:
                hits.append((pi, fitz.Rect(b[:4])))
    return hits

def annotate_pdf(pdf_bytes: bytes, comments: List[Dict[str, Any]]) -> bytes:
    """Annotate the PDF with rectangles on matched keywords or text sticky when not found."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for c in comments:
        message = c.get("message", "")
        keyword = c.get("keyword", "")
        page_hint = int(c.get("page", 1)) - 1
        added = False

        if keyword:
            # try block-match first
            boxes = find_text_boxes(doc, keyword, 80)
            for (pi, rect) in boxes:
                annot = doc[pi].add_rect_annot(rect)
                annot.set_info(title="QA", content=message)
                annot.update()
                added = True

        if not added:
            # fallback sticky text near top-left of intended page (or page 1)
            pi = max(0, min(page_hint, len(doc) - 1))
            where = fitz.Point(40, 60)
            annot = doc[pi].add_text_annot(where, message)
            annot.update()

    out = doc.tobytes()
    doc.close()
    return out

# --------------------------------------------------------------------------------------
# FINDINGS ENGINE
# --------------------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9\-_/]+", text)

def spelling_findings(pages: List[str], allow: set) -> List[Dict[str, Any]]:
    """Very gentle spelling flagger: only flags rare singletons not in allowlist or common acronyms."""
    findings = []
    # A tiny built-in whitelist for common terms; extend with rules.allowlist
    built_in_ok = set([
        "PDF", "LTE", "NR", "5G", "PSU", "kVA", "RF", "BTS", "RAN", "AHEGG", "ELTEK",
        "ERICSSON", "NOKIA", "MIMO", "GA", "DD", "MBNL", "BTEE", "CTIL"
    ])
    seen = {}
    for pi, text in enumerate(pages, start=1):
        for tok in tokenize(text):
            up = tok.upper()
            seen[up] = seen.get(up, 0) + 1

    for pi, text in enumerate(pages, start=1):
        for tok in tokenize(text):
            up = tok.upper()
            if up in built_in_ok or up in (s.upper() for s in allow):
                continue
            if len(up) <= 2:  # super short tokens are noisy
                continue
            if seen.get(up, 0) < 3 and not re.search(r"\d", up):  # rare & not codes
                findings.append({
                    "type": "Spelling",
                    "page": pi,
                    "keyword": tok,
                    "message": f"Possible typo: '{tok}'",
                    "rule_id": "spelling-basic"
                })
    return findings

def address_title_check(pages: List[str], site_address: str) -> Optional[Dict[str, Any]]:
    if not site_address: return None
    # Normalize address: ignore ", 0 ," pattern
    addr_norm = re.sub(r"\s*,\s*0\s*,\s*", ", ", site_address).strip()
    title_text = " ".join(pages[:1]).upper()  # first page only as "title"
    if addr_norm and addr_norm.upper() not in title_text:
        return {
            "type": "Address",
            "page": 1,
            "keyword": addr_norm,
            "message": "Site Address not found in the title page.",
            "rule_id": "address-in-title"
        }
    return None

def apply_forbid_pairs(pages: List[str], pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    text_all = "\n".join(pages).upper()
    out = []
    for p in pairs:
        if_any = [x.upper() for x in p.get("if_any", [])]
        forbid_any = [x.upper() for x in p.get("forbid_any", [])]
        if any(k in text_all for k in if_any):
            bad = [b for b in forbid_any if b in text_all]
            if bad:
                out.append({
                    "type": "PairRule",
                    "page": 1,
                    "keyword": bad[0],
                    "message": f"Forbidden with {'/'.join(if_any)}: found {bad[0]}",
                    "rule_id": "forbid-pair"
                })
    return out

def must_include_check(pages: List[str], rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    mi = rule.get("must_include", {})
    text = (mi.get("text") or "").strip()
    if not text: return None
    page = mi.get("page", None)
    # If page beyond document length, search all pages
    if page and 1 <= page <= len(pages):
        scope = pages[page-1]
        pi = page
    else:
        scope = "\n".join(pages)
        pi = 1
    if text.upper() not in scope.upper():
        return {
            "type": "MustInclude",
            "page": pi,
            "keyword": text[:60],
            "message": f"Required text missing: '{text[:100]}...'",
            "rule_id": rule.get("id", "must-include")
        }
    return None

def when_matches(meta: Dict[str, Any], when: Dict[str, Any]) -> bool:
    """Return True if all provided keys match (set membership)."""
    for k, vals in when.items():
        v = meta.get(k)
        if v is None: return False
        if isinstance(vals, list):
            if v not in vals: return False
        else:
            if v != vals: return False
    return True

def rules_findings(pages: List[str], meta: Dict[str, Any], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for rule in rules.get("checklist", []):
        if not when_matches(meta, rule.get("when", {})):
            continue
        if rule.get("address_in_title"):
            f = address_title_check(pages, meta.get("address", ""))
            if f: out.append(f)
        if rule.get("must_include"):
            f = must_include_check(pages, rule)
            if f: out.append(f)
    # Pairs
    out += apply_forbid_pairs(pages, rules.get("forbid_pairs", []))
    return out

# --------------------------------------------------------------------------------------
# HISTORY
# --------------------------------------------------------------------------------------
def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_CSV)
    except Exception:
        # Broken CSV (e.g. during debugging). Recover minimally.
        try:
            df = pd.read_csv(HISTORY_CSV, on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()
    # Ensure columns
    base_cols = [
        "timestamp_utc","supplier","drawing_type","client","project","site_type",
        "vendor","cabinet","radio","sectors","address","mimo_json","status",
        "pdf_name","excel_name","exclude"
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = None
    if "exclude" in df.columns:
        df["exclude"] = df["exclude"].fillna(False).astype(bool)
    return df

def append_history(meta: Dict[str, Any], status: str, pdf_name: str, excel_name: str) -> None:
    ensure_dirs()
    row = {
        "timestamp_utc": now_utc_iso(),
        "supplier": meta["supplier"],
        "drawing_type": meta["drawing_type"],
        "client": meta["client"],
        "project": meta["project"],
        "site_type": meta["site_type"],
        "vendor": meta["vendor"],
        "cabinet": meta["cabinet"],
        "radio": meta["radio"],
        "sectors": meta["sectors"],
        "address": meta["address"],
        "mimo_json": json.dumps(meta["mimo"], ensure_ascii=False),
        "status": status,
        "pdf_name": pdf_name,
        "excel_name": excel_name,
        "exclude": bool(meta.get("exclude_from_analytics", False))
    }
    header = not os.path.exists(HISTORY_CSV)
    pd.DataFrame([row]).to_csv(HISTORY_CSV, mode="a", header=header, index=False)

# --------------------------------------------------------------------------------------
# TRAINING (“LEARNING”) HELPERS
# --------------------------------------------------------------------------------------
def add_allow_terms(terms: List[str]) -> None:
    rules = load_rules()
    allow = set(rules.get("allowlist", []))
    for t in terms:
        t = (t or "").strip()
        if t: allow.add(t)
    rules["allowlist"] = sorted(allow)
    save_rules(rules)

def add_forbid_pair(trigger_terms: List[str], forbidden_terms: List[str]) -> None:
    rules = load_rules()
    fp = rules.get("forbid_pairs", [])
    fp.append({"if_any": trigger_terms, "forbid_any": forbidden_terms})
    rules["forbid_pairs"] = fp
    save_rules(rules)

def add_must_include_rule(rule_id: str, when_ctx: Dict[str, Any], text: str, page: Optional[int]) -> None:
    rules = load_rules()
    cl = rules.get("checklist", [])
    r = {
        "id": rule_id,
        "when": when_ctx or {},
        "must_include": {
            "text": text
        }
    }
    if page:
        r["must_include"]["page"] = int(page)
    cl.append(r)
    rules["checklist"] = cl
    save_rules(rules)

# --------------------------------------------------------------------------------------
# UI: PASSWORD GATE
# --------------------------------------------------------------------------------------
def password_gate():
    if st.session_state.get("entry_ok"):
        return
    st.title("AI Design Quality Auditor")
    st.caption("Secure access")
    pw = st.text_input("Enter access password", type="password")
    if st.button("Enter"):
        if pw == ENTRY_PASSWORD:
            st.session_state["entry_ok"] = True
            try:
                st.rerun()
            except Exception:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass
        else:
            st.error("Wrong password.")
    st.stop()

# --------------------------------------------------------------------------------------
# UI: AUDIT TAB
# --------------------------------------------------------------------------------------
def run_audit_ui():
    # Header row with logo (top-left, not overlay)
    cols = st.columns([0.15, 0.85])
    with cols[0]:
        logo_path = st.text_input("Logo file (optional)", value=find_logo_path() or "")
        if logo_path and os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
    with cols[1]:
        st.subheader("Audit")

    # Metadata (all required)
    meta = {}
    c1, c2, c3 = st.columns(3)
    meta["supplier"] = c1.selectbox("Supplier", SUPPLIERS)
    meta["drawing_type"] = c2.selectbox("Drawing Type", DRAWING_TYPES)
    meta["client"] = c3.selectbox("Client", CLIENTS)

    c4, c5, c6 = st.columns(3)
    meta["project"] = c4.selectbox("Project", PROJECTS)
    meta["site_type"] = c5.selectbox("Site Type", SITE_TYPES)
    meta["vendor"] = c6.selectbox("Proposed Vendor", VENDORS)

    c7, c8, c9 = st.columns(3)
    meta["cabinet"] = c7.selectbox("Proposed Cabinet Location", CABINETS)
    meta["radio"] = c8.selectbox("Proposed Radio Location", RADIOS)
    meta["sectors"] = c9.selectbox("Quantity of Sectors", [1,2,3,4,5,6])

    meta["address"] = st.text_input("Site Address (required)")

    # MIMO section (hidden if Project = Power Resilience)
    show_mimo = meta["project"] != "Power Resilience"
    mimo = {}
    if show_mimo:
        st.markdown("### Proposed MIMO Config")
        same = st.checkbox("Use S1 for all sectors", value=True)
        for s in range(1, meta["sectors"] + 1):
            if s == 1 or not same:
                mimo[f"S{s}"] = st.selectbox(f"MIMO S{s}", MIMO_CONFIGS, key=f"mimo_s{s}")
            else:
                mimo[f"S{s}"] = mimo["S1"]
    meta["mimo"] = mimo

    st.divider()

    # PDF upload
    up = st.file_uploader("Upload PDF design", type=["pdf"])
    meta["exclude_from_analytics"] = st.checkbox("Exclude this review from Analytics (training / dry-run)", value=False)

    # Trainer expander
    with st.expander("Trainer (teach the tool quickly)"):
        st.caption("Use these controls after you run an audit, to mark false positives or add new rules fast.")
        colT1, colT2 = st.columns(2)
        with colT1:
            st.markdown("**Add to allowlist (spelling / terms)**")
            allow_terms = st.text_input("Terms (comma-separated)", key="trainer_allow")
            if st.button("Add terms to allowlist"):
                add_allow_terms([t.strip() for t in allow_terms.split(",") if t.strip()])
                st.success("Added to allowlist.")
        with colT2:
            st.markdown("**Add a forbid-pair rule (X present ⇒ Y must not appear)**")
            trig = st.text_input("Trigger terms (comma-separated)", key="trainer_trig")
            forb = st.text_input("Forbidden terms (comma-separated)", key="trainer_forb")
            if st.button("Add forbid pair rule"):
                add_forbid_pair(
                    [t.strip() for t in trig.split(",") if t.strip()],
                    [t.strip() for t in forb.split(",") if t.strip()],
                )
                st.success("Forbid pair rule added.")

        st.markdown("**Add a must-include rule (X must be present, optional page)**")
        rule_id = st.text_input("Rule ID", value=f"rule-{date_tag()}")
        must_text = st.text_area("Required text")
        page_opt = st.text_input("Page (optional integer)", value="")
        ctx_cols = st.columns(4)
        ctx = {}
        ctx["project"] = ctx_cols[0].selectbox("Ctx: Project (optional)", ["(any)"] + PROJECTS)
        ctx["client"] = ctx_cols[1].selectbox("Ctx: Client (optional)", ["(any)"] + CLIENTS)
        ctx["vendor"] = ctx_cols[2].selectbox("Ctx: Vendor (optional)", ["(any)"] + VENDORS)
        ctx["drawing_type"] = ctx_cols[3].selectbox("Ctx: Drawing Type (optional)", ["(any)"] + DRAWING_TYPES)

        def _ctx_clean(_ctx: Dict[str, Any]) -> Dict[str, Any]:
            return {k:v for k,v in _ctx.items() if v and v != "(any)"}

        if st.button("Add must-include rule"):
            try:
                add_must_include_rule(
                    rule_id.strip(),
                    _ctx_clean(ctx),
                    must_text.strip(),
                    int(page_opt) if page_opt.strip().isdigit() else None
                )
                st.success("Must-include rule added.")
            except Exception as e:
                st.error(f"Failed to add rule: {e}")

        st.markdown("**Bulk training upload** (CSV or XLSX)")
        st.caption("Columns supported: action [allow|forbid|must], terms / trigger / forbidden / text / page / project / client / vendor / drawing_type")
        fb = st.file_uploader("Upload training file", type=["csv","xlsx"], key="trainer_upload")
        if fb is not None:
            try:
                if fb.name.lower().endswith(".csv"):
                    df_fb = pd.read_csv(fb)
                else:
                    df_fb = pd.read_excel(fb)
                applied = 0
                for _, r in df_fb.iterrows():
                    action = str(r.get("action","")).strip().lower()
                    if action == "allow":
                        terms = str(r.get("terms","")).split(",")
                        add_allow_terms([t.strip() for t in terms if t.strip()])
                        applied += 1
                    elif action == "forbid":
                        trig = str(r.get("trigger","")).split(",")
                        forb = str(r.get("forbidden","")).split(",")
                        add_forbid_pair([t.strip() for t in trig if t.strip()],
                                        [t.strip() for t in forb if t.strip()])
                        applied += 1
                    elif action == "must":
                        when_ctx = {}
                        for k in ["project","client","vendor","drawing_type"]:
                            v = r.get(k, None)
                            if isinstance(v, str) and v.strip():
                                when_ctx[k] = v.strip()
                        text_req = str(r.get("text","")).strip()
                        pg = r.get("page", None)
                        pgint = int(pg) if pd.notna(pg) and str(pg).isdigit() else None
                        rid = str(r.get("rule_id", f"rule-{date_tag()}")).strip()
                        add_must_include_rule(rid, when_ctx, text_req, pgint)
                        applied += 1
                st.success(f"Applied {applied} training changes.")
            except Exception as e:
                st.error(f"Training upload error: {e}")

    st.divider()

    # Audit execution
    run = st.button("Run Audit", use_container_width=True)
    if run:
        # Validate required metadata
        missing = [k for k in ["supplier","drawing_type","client","project","site_type","vendor","cabinet","radio","sectors","address"] if not meta.get(k)]
        if missing:
            st.error(f"Please complete all metadata: {', '.join(missing)}")
            return
        if show_mimo:
            if not meta["mimo"] or not meta["mimo"].get("S1"):
                st.error("Please select at least MIMO S1 (or hide by selecting Project = Power Resilience).")
                return
        if up is None:
            st.error("Please upload a PDF.")
            return

        pdf_bytes = up.read()
        try:
            pages = extract_pdf_text_pages(pdf_bytes)
        except Exception as e:
            st.error(f"PDF text extraction failed: {e}")
            return

        rules = load_rules()
        allow = set(rules.get("allowlist", []))

        findings = []
        # Spelling (gentle)
        findings += spelling_findings(pages, allow)
        # Rules
        findings += rules_findings(pages, meta, rules)

        # Status
        status = "Pass" if len(findings) == 0 else "Rejected"

        # Build annotated PDF and Excel
        # Prepare messages for annotation
        anno = []
        for f in findings:
            anno.append({
                "page": f.get("page", 1),
                "keyword": f.get("keyword", ""),
                "message": f"[{f.get('type','Rule')}] {f.get('message','')}"
            })
        annotated_bytes = annotate_pdf(pdf_bytes, anno)

        # Excel
        df = pd.DataFrame(findings) if findings else pd.DataFrame(columns=["type","page","keyword","message","rule_id"])
        tag = f"{status}_{date_tag()}"
        base = os.path.splitext(up.name)[0]
        excel_name = safe_filename(f"{base}_{tag}.xlsx")
        pdf_ann_name = safe_filename(f"{base}_{tag}_annotated.pdf")

        ensure_dirs()
        # Save outputs for persistence
        excel_path = os.path.join(OUTPUT_DIR, excel_name)
        pdf_ann_path = os.path.join(OUTPUT_DIR, pdf_ann_name)
        with pd.ExcelWriter(excel_path, engine="openpyxl") as xw:
            df.to_excel(xw, index=False, sheet_name="Findings")
            # Add metadata sheet
            pd.DataFrame([meta]).to_excel(xw, index=False, sheet_name="Metadata")
        with open(pdf_ann_path, "wb") as f:
            f.write(annotated_bytes)

        # Download buttons
        cdl1, cdl2 = st.columns(2)
        with cdl1:
            with open(pdf_ann_path, "rb") as f:
                st.download_button("Download Annotated PDF", data=f.read(), file_name=pdf_ann_name, mime="application/pdf")
        with cdl2:
            with open(excel_path, "rb") as f:
                st.download_button("Download Excel Report", data=f.read(), file_name=excel_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Show table
        st.success(f"Audit complete: **{status}**")
        st.dataframe(df, use_container_width=True)

        # Append to history
        append_history(meta, status, up.name, excel_name)

# --------------------------------------------------------------------------------------
# UI: ANALYTICS TAB
# --------------------------------------------------------------------------------------
def analytics_ui():
    st.subheader("Analytics & Trends")
    df = load_history()
    if df.empty:
        st.info("No audits yet.")
        return

    # Filters
    c1, c2, c3 = st.columns(3)
    f_supplier = c1.selectbox("Supplier", ["All"] + SUPPLIERS)
    f_client = c2.selectbox("Client", ["All"] + CLIENTS)
    f_project = c3.selectbox("Project", ["All"] + PROJECTS)

    show = df.copy()
    # Respect exclude flag
    show = show[show.get("exclude", False) == False] if "exclude" in show.columns else show

    if f_supplier != "All":
        show = show[show["supplier"] == f_supplier]
    if f_client != "All":
        show = show[show["client"] == f_client]
    if f_project != "All":
        show = show[show["project"] == f_project]

    total = len(show)
    rft = 0.0
    if total > 0:
        rft = round(100 * len(show[show["status"] == "Pass"]) / total, 1)

    m1, m2 = st.columns(2)
    m1.metric("Total Audits", total)
    m2.metric("Right First Time %", rft)

    # Simple trend line over time
    if total > 0 and "timestamp_utc" in show.columns:
        try:
            show["_date"] = pd.to_datetime(show["timestamp_utc"]).dt.date
            trend = show.groupby("_date")["status"].apply(lambda s: (s == "Pass").mean() * 100).reset_index(name="RFT%")
            st.line_chart(trend.set_index("_date"))
        except Exception:
            pass

    # Table (guard missing columns)
    cols = [c for c in ["timestamp_utc","supplier","client","project","status","pdf_name","excel_name"] if c in show.columns]
    st.dataframe(show[cols], use_container_width=True)

    st.divider()
    st.markdown("### Manage history records")
    st.caption("Exclude/Include rows from analytics.")
    if "exclude" in df.columns:
        edit = df.copy()
        edit["exclude"] = edit["exclude"].astype(bool)
        st.dataframe(edit[cols + ["exclude"]] if "exclude" in edit.columns else edit[cols], use_container_width=True)
        if st.button("Export all outputs to ZIP"):
            # On Streamlit Cloud we can’t run zip reliably without shell; we can at least show where files are.
            st.info(f"Outputs are saved in ./{OUTPUT_DIR}. You can download individual files from the Audit runs.")
    else:
        st.info("No exclude column yet—run at least one audit.")

# --------------------------------------------------------------------------------------
# UI: SETTINGS TAB
# --------------------------------------------------------------------------------------
def settings_ui():
    st.subheader("Settings")
    st.markdown("**Rules editor (YAML)**")
    pw = st.text_input("Rules password", type="password")
    if pw != RULES_PASSWORD:
        st.warning("Enter rules password to edit the rules.")
        return

    rules = load_rules()
    raw = yaml.safe_dump(rules, sort_keys=False, allow_unicode=True)
    text = st.text_area("rules_example.yaml", value=raw, height=420)
    colS1, colS2 = st.columns(2)
    if colS1.button("Save rules"):
        try:
            new_rules = yaml.safe_load(text) or {}
            # Minimal validation
            new_rules.setdefault("schema_version", 2)
            new_rules.setdefault("allowlist", [])
            new_rules.setdefault("forbid_pairs", [])
            new_rules.setdefault("checklist", [])
            save_rules(new_rules)
            st.success("Rules saved.")
        except Exception as e:
            st.error(f"YAML error: {e}")
    if colS2.button("Restore defaults"):
        save_rules(DEFAULT_RULES.copy())
        st.success("Defaults restored.")

# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    ensure_dirs()
    password_gate()

    st.set_page_config(page_title="AI Design Quality Auditor", layout="wide")
    st.title("AI Design Quality Auditor")

    tabs = st.tabs(["Audit", "Analytics", "Settings"])
    with tabs[0]:
        run_audit_ui()
    with tabs[1]:
        analytics_ui()
    with tabs[2]:
        settings_ui()

if __name__ == "__main__":
    main()
