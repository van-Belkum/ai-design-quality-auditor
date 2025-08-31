# app.py
import os, io, re, json, base64, time, datetime
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import pandas as pd
import yaml
from rapidfuzz import fuzz, process
import fitz  # PyMuPDF

# ======================================================================================
# TUNABLES (speed vs depth)
# ======================================================================================
ENTRY_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"

FAST_MODE = True               # Fast scans on by default
MAX_SPELL_FLAGS = 120          # Upper limit of spelling findings
MIN_TOKEN_LEN = 3              # Ignore tiny tokens for spelling
RARE_TOKEN_THRESHOLD = 3       # Token must appear < 3 times to be flagged
FUZZ_THRESHOLD = 88            # Fuzzy match threshold for locating boxes
MAX_PAGES_INDEX = 400          # Hard cap to protect huge PDFs
ANNOTATE_MAX_FINDINGS = 200    # Don’t try to annotate thousands of notes
SEARCH_EXACT_FIRST = True      # Try exact locate (fast) before fuzzy
PROGRESS_UPDATE_EVERY = 10     # Update progress bars every N pages

HISTORY_DIR = "history"
OUTPUT_DIR = "outputs"
HISTORY_CSV = os.path.join(HISTORY_DIR, "audit_history.csv")
RULES_FILE = "rules_example.yaml"

PREFERRED_LOGOS = ["logo.png","logo.jpg","logo.jpeg","logo.svg"]

# ======================================================================================
# STATIC OPTIONS
# ======================================================================================
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

# ======================================================================================
# DEFAULT RULES (safe baseline)
# ======================================================================================
DEFAULT_RULES = {
    "schema_version": 2,
    "allowlist": [
        "EE", "BTS", "MBNL", "RAN", "LTE", "NR", "DC", "AC", "kVA", "AHEGG",
        "Polaradium", "Eltek", "PSU", "GA", "DD", "CTIL"
    ],
    "forbid_pairs": [
        {"if_any": ["Brush"], "forbid_any": ["Generator Power"]},
        {"if_any": ["Generator Power"], "forbid_any": ["Brush"]}
    ],
    "checklist": [
        {
            "id": "power-res-eltek-note",
            "when": {"project": ["Power Resilience"]},
            "must_include": {
                "text": "IMPORTANT NOTE: To support the power resilience configure settings the Eltek PSU will need to be configured as per TDEE53201 section 3.8.1",
                "page": 300
            }
        },
        {"id": "address-in-title", "when": {}, "address_in_title": True}
    ]
}

# ======================================================================================
# UTILITIES
# ======================================================================================
def ensure_dirs():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def now_utc_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat()

def date_tag() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d")

def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

def find_logo_path(custom: Optional[str]=None) -> Optional[str]:
    if custom and os.path.exists(custom): return custom
    for p in PREFERRED_LOGOS:
        if os.path.exists(p): return p
    return None

# ======================================================================================
# RULES IO
# ======================================================================================
def load_rules() -> Dict[str, Any]:
    if not os.path.exists(RULES_FILE):
        return DEFAULT_RULES.copy()
    try:
        with open(RULES_FILE, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        data = {}
    data.setdefault("schema_version", 2)
    data.setdefault("allowlist", [])
    data.setdefault("forbid_pairs", [])
    data.setdefault("checklist", [])
    return data

def save_rules(rules: Dict[str, Any]) -> None:
    with open(RULES_FILE, "w") as f:
        yaml.safe_dump(rules, f, sort_keys=False, allow_unicode=True)

# ======================================================================================
# INDEX PDF (single pass, reused)
# ======================================================================================
class PdfIndex:
    def __init__(self, pdf_bytes: bytes):
        self.doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        self.n_pages = min(len(self.doc), MAX_PAGES_INDEX)
        self.pages_text: List[str] = []
        self.pages_text_lc: List[str] = []
        self.blocks_per_page: List[list] = []

    def build(self, progress=None):
        for i in range(self.n_pages):
            if progress and (i % PROGRESS_UPDATE_EVERY == 0 or i == self.n_pages-1):
                progress.progress((i+1)/self.n_pages)
            page = self.doc[i]
            txt = page.get_text("text", flags=fitz.TEXT_DEHYPHENATE)
            self.pages_text.append(txt)
            self.pages_text_lc.append(txt.lower())
            self.blocks_per_page.append(page.get_text("blocks"))

    def search_exact(self, needle: str) -> List[Tuple[int, fitz.Rect]]:
        """Fast exact substring locate using search_for (case-insensitive)."""
        out = []
        pattern = needle.strip()
        if not pattern: return out
        flags = fitz.TEXT_IGNORECASE
        for i in range(self.n_pages):
            rects = self.doc[i].search_for(pattern, flags=flags)
            for r in rects:
                out.append((i, r))
        return out

    def search_fuzzy_blocks(self, needle: str, thresh: int = FUZZ_THRESHOLD) -> List[Tuple[int, fitz.Rect]]:
        """Slower: fuzzy match against block texts (bounded by thresh)."""
        out = []
        nu = needle.strip().upper()
        if not nu:
            return out
        for i in range(self.n_pages):
            for b in self.blocks_per_page[i]:
                txt = (b[4] or "").strip()
                if not txt: continue
                if fuzz.partial_ratio(nu, txt.upper()) >= thresh:
                    out.append((i, fitz.Rect(b[:4])))
        return out

    def close(self):
        self.doc.close()

# ======================================================================================
# FINDINGS
# ======================================================================================
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/]{2,}")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)

def spelling_findings(index: PdfIndex, allow: set) -> List[Dict[str, Any]]:
    builtin_ok = set(["PDF","LTE","NR","5G","PSU","KVA","RF","BTS","RAN","AHEGG","ELTEK","ERICSSON","NOKIA","MIMO","GA","DD","MBNL","BTEE","CTIL"])
    counts = {}
    for pi, txt in enumerate(index.pages_text, start=1):
        for tok in tokenize(txt):
            u = tok.upper()
            counts[u] = counts.get(u, 0) + 1

    findings = []
    limit = MAX_SPELL_FLAGS if FAST_MODE else 999999
    for pi, txt in enumerate(index.pages_text, start=1):
        if len(findings) >= limit: break
        for tok in tokenize(txt):
            if len(findings) >= limit: break
            u = tok.upper()
            if len(u) < MIN_TOKEN_LEN: continue
            if u in builtin_ok or u in (s.upper() for s in allow): continue
            if any(c.isdigit() for c in u): continue
            if counts.get(u,0) >= RARE_TOKEN_THRESHOLD: continue
            findings.append({
                "type":"Spelling","page":pi,"keyword":tok,
                "message":f"Possible typo: '{tok}'","rule_id":"spelling-basic"
            })
    return findings

def normalize_address(addr: str) -> str:
    if not addr: return ""
    return re.sub(r"\s*,\s*0\s*,\s*", ", ", addr).strip()

def address_title_check(index: PdfIndex, site_address: str) -> Optional[Dict[str, Any]]:
    addr = normalize_address(site_address)
    if not addr: return None
    title = index.pages_text[0] if index.n_pages>0 else ""
    if addr.upper() not in title.upper():
        return {"type":"Address","page":1,"keyword":addr,"message":"Site Address not found on title page.","rule_id":"address-in-title"}
    return None

def apply_forbid_pairs(index: PdfIndex, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    big = "\n".join(index.pages_text).upper()
    out = []
    for p in pairs:
        if_any = [x.upper() for x in p.get("if_any",[])]
        forbid_any = [x.upper() for x in p.get("forbid_any",[])]
        if any(k in big for k in if_any):
            bad = [b for b in forbid_any if b in big]
            if bad:
                out.append({"type":"PairRule","page":1,"keyword":bad[0],
                            "message":f"Forbidden with {'/'.join(if_any)}: found {bad[0]}",
                            "rule_id":"forbid-pair"})
    return out

def must_include_check(index: PdfIndex, rule: Dict[str,Any]) -> Optional[Dict[str,Any]]:
    mi = rule.get("must_include", {})
    text = (mi.get("text") or "").strip()
    if not text: return None
    page = mi.get("page", None)
    if page and 1 <= page <= index.n_pages:
        scope = index.pages_text[page-1]
        pi = page
    else:
        scope = "\n".join(index.pages_text)
        pi = 1
    if text.upper() not in scope.upper():
        return {"type":"MustInclude","page":pi,"keyword":text[:60],
                "message":f"Required text missing: '{text[:100]}...'",
                "rule_id": rule.get("id","must-include")}
    return None

def when_matches(meta: Dict[str,Any], when: Dict[str,Any]) -> bool:
    for k,vals in when.items():
        v = meta.get(k)
        if v is None: return False
        if isinstance(vals, list):
            if v not in vals: return False
        else:
            if v != vals: return False
    return True

def rules_findings(index: PdfIndex, meta: Dict[str,Any], rules: Dict[str,Any]) -> List[Dict[str,Any]]:
    out = []
    for rule in rules.get("checklist", []):
        if not when_matches(meta, rule.get("when", {})): continue
        if rule.get("address_in_title"):
            f = address_title_check(index, meta.get("address",""))
            if f: out.append(f)
        if rule.get("must_include"):
            f = must_include_check(index, rule)
            if f: out.append(f)
    out += apply_forbid_pairs(index, rules.get("forbid_pairs", []))
    return out

# ======================================================================================
# ANNOTATION (uses index; exact-first, fuzzy fallback; bounded)
# ======================================================================================
def annotate_pdf(index: PdfIndex, findings: List[Dict[str,Any]]) -> bytes:
    maxn = min(len(findings), ANNOTATE_MAX_FINDINGS)
    for i, f in enumerate(findings[:maxn]):
        msg = f"[{f.get('type','Rule')}] {f.get('message','')}"
        kw = f.get("keyword","").strip()
        page_hint = max(1, int(f.get("page",1))) - 1
        placed = False

        if kw:
            if SEARCH_EXACT_FIRST:
                boxes = index.search_exact(kw)
                if boxes:
                    for (pi, rect) in boxes[:1]:
                        a = index.doc[pi].add_rect_annot(rect)
                        a.set_info(title="QA", content=msg); a.update()
                        placed = True
            if not placed:
                fb = index.search_fuzzy_blocks(kw, FUZZ_THRESHOLD)
                if fb:
                    (pi, rect) = fb[0]
                    a = index.doc[pi].add_rect_annot(rect)
                    a.set_info(title="QA", content=msg); a.update()
                    placed = True

        if not placed:
            pi = max(0, min(page_hint, index.n_pages-1))
            pt = fitz.Point(40, 60)
            a = index.doc[pi].add_text_annot(pt, msg); a.update()

    out = index.doc.tobytes()
    return out

# ======================================================================================
# HISTORY
# ======================================================================================
def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_CSV)
    except Exception:
        try:
            df = pd.read_csv(HISTORY_CSV, on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()
    base_cols = [
        "timestamp_utc","supplier","drawing_type","client","project","site_type",
        "vendor","cabinet","radio","sectors","address","mimo_json","status",
        "pdf_name","excel_name","exclude"
    ]
    for c in base_cols:
        if c not in df.columns: df[c] = None
    if "exclude" in df.columns:
        df["exclude"] = df["exclude"].fillna(False).astype(bool)
    return df

def append_history(meta: Dict[str,Any], status: str, pdf_name: str, excel_name: str):
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
        "mimo_json": json.dumps(meta.get("mimo",{}), ensure_ascii=False),
        "status": status,
        "pdf_name": pdf_name,
        "excel_name": excel_name,
        "exclude": bool(meta.get("exclude_from_analytics", False))
    }
    header = not os.path.exists(HISTORY_CSV)
    pd.DataFrame([row]).to_csv(HISTORY_CSV, mode="a", header=header, index=False)

# ======================================================================================
# TRAINER API
# ======================================================================================
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

def add_must_include_rule(rule_id: str, when_ctx: Dict[str,Any], text: str, page: Optional[int]) -> None:
    rules = load_rules()
    cl = rules.get("checklist", [])
    r = {"id": rule_id, "when": when_ctx or {}, "must_include": {"text": text}}
    if page: r["must_include"]["page"] = int(page)
    cl.append(r)
    rules["checklist"] = cl
    save_rules(rules)

# ======================================================================================
# UI HELPERS
# ======================================================================================
def password_gate():
    if st.session_state.get("entry_ok"): return
    st.title("AI Design Quality Auditor")
    st.caption("Secure access")
    pw = st.text_input("Enter access password", type="password")
    if st.button("Enter"):
        if pw == ENTRY_PASSWORD:
            st.session_state["entry_ok"] = True
            try: st.rerun()
            except Exception:
                try: st.experimental_rerun()
                except Exception: pass
        else:
            st.error("Wrong password.")
    st.stop()

def top_header():
    c = st.columns([0.12, 0.88])
    with c[0]:
        lp = st.text_input("Logo path (optional)", value=find_logo_path() or "", key="logo_path")
        if lp and os.path.exists(lp):
            st.image(lp, use_container_width=True)
    with c[1]:
        st.subheader("Audit")

# ======================================================================================
# AUDIT TAB
# ======================================================================================
def audit_tab():
    top_header()

    meta = {}
    c1,c2,c3 = st.columns(3)
    meta["supplier"] = c1.selectbox("Supplier", SUPPLIERS)
    meta["drawing_type"] = c2.selectbox("Drawing Type", DRAWING_TYPES)
    meta["client"] = c3.selectbox("Client", CLIENTS)

    c4,c5,c6 = st.columns(3)
    meta["project"] = c4.selectbox("Project", PROJECTS)
    meta["site_type"] = c5.selectbox("Site Type", SITE_TYPES)
    meta["vendor"] = c6.selectbox("Proposed Vendor", VENDORS)

    c7,c8,c9 = st.columns(3)
    meta["cabinet"] = c7.selectbox("Proposed Cabinet Location", CABINETS)
    meta["radio"] = c8.selectbox("Proposed Radio Location", RADIOS)
    meta["sectors"] = c9.selectbox("Quantity of Sectors", [1,2,3,4,5,6])

    meta["address"] = st.text_input("Site Address (required)")

    show_mimo = meta["project"] != "Power Resilience"
    mimo = {}
    if show_mimo:
        st.markdown("### Proposed MIMO Config")
        same = st.checkbox("Use S1 for all sectors", value=True)
        for s in range(1, meta["sectors"]+1):
            if s == 1 or not same:
                mimo[f"S{s}"] = st.selectbox(f"MIMO S{s}", MIMO_CONFIGS, key=f"mimo_s{s}")
            else:
                mimo[f"S{s}"] = mimo["S1"]
    meta["mimo"] = mimo

    st.divider()
    up = st.file_uploader("Upload PDF design", type=["pdf"])
    meta["exclude_from_analytics"] = st.checkbox("Exclude this review from Analytics (training / dry-run)", value=False)

    with st.expander("Trainer (teach fast)"):
        st.caption("Mark false positives or add new rules quickly.")
        colT1, colT2 = st.columns(2)
        with colT1:
            allow_terms = st.text_input("Add to allowlist (comma-separated)")
            if st.button("Add allow terms"):
                add_allow_terms([t.strip() for t in allow_terms.split(",") if t.strip()])
                st.success("Allowlist updated.")
        with colT2:
            trig = st.text_input("Trigger terms (comma-separated)")
            forb = st.text_input("Forbidden terms (comma-separated)")
            if st.button("Add forbid-pair"):
                add_forbid_pair(
                    [t.strip() for t in trig.split(",") if t.strip()],
                    [t.strip() for t in forb.split(",") if t.strip()]
                )
                st.success("Forbid rule added.")

        st.markdown("**Add must-include rule**")
        rid = st.text_input("Rule ID", value=f"rule-{date_tag()}")
        must_text = st.text_area("Required text")
        ctx_cols = st.columns(4)
        ctx = {}
        ctx["project"] = ctx_cols[0].selectbox("Ctx Project", ["(any)"] + PROJECTS)
        ctx["client"] = ctx_cols[1].selectbox("Ctx Client", ["(any)"] + CLIENTS)
        ctx["vendor"] = ctx_cols[2].selectbox("Ctx Vendor", ["(any)"] + VENDORS)
        ctx["drawing_type"] = ctx_cols[3].selectbox("Ctx Drawing Type", ["(any)"] + DRAWING_TYPES)
        page_raw = st.text_input("Page (optional integer)")
        def _ctx(c): return {k:v for k,v in c.items() if v and v!="(any)"}
        if st.button("Add must-include"):
            add_must_include_rule(rid.strip(), _ctx(ctx), must_text.strip(),
                                  int(page_raw) if page_raw.strip().isdigit() else None)
            st.success("Must-include added.")

        st.markdown("**Bulk trainer upload** (CSV/XLSX)")
        fb = st.file_uploader("Upload training file", type=["csv","xlsx"], key="trainer_up")
        if fb is not None:
            try:
                dfb = pd.read_csv(fb) if fb.name.lower().endswith(".csv") else pd.read_excel(fb)
                applied = 0
                for _, r in dfb.iterrows():
                    action = str(r.get("action","")).strip().lower()
                    if action == "allow":
                        terms = [t.strip() for t in str(r.get("terms","")).split(",") if t.strip()]
                        add_allow_terms(terms); applied += 1
                    elif action == "forbid":
                        trig = [t.strip() for t in str(r.get("trigger","")).split(",") if t.strip()]
                        forb = [t.strip() for t in str(r.get("forbidden","")).split(",") if t.strip()]
                        add_forbid_pair(trig, forb); applied += 1
                    elif action == "must":
                        when = {}
                        for k in ["project","client","vendor","drawing_type"]:
                            v = r.get(k, None)
                            if isinstance(v, str) and v.strip(): when[k] = v.strip()
                        text_req = str(r.get("text","")).strip()
                        pg = r.get("page", None)
                        pgint = int(pg) if pd.notna(pg) and str(pg).isdigit() else None
                        rid2 = str(r.get("rule_id", f"rule-{date_tag()}")).strip()
                        add_must_include_rule(rid2, when, text_req, pgint); applied += 1
                st.success(f"Applied {applied} training changes.")
            except Exception as e:
                st.error(f"Training upload error: {e}")

    st.divider()
    run = st.button("Run Audit", type="primary", use_container_width=True)

    if run:
        # Validate
        needed = ["supplier","drawing_type","client","project","site_type","vendor","cabinet","radio","sectors","address"]
        miss = [k for k in needed if not meta.get(k)]
        if miss:
            st.error(f"Please complete metadata: {', '.join(miss)}"); return
        if (meta["project"] != "Power Resilience") and not meta["mimo"].get("S1"):
            st.error("Select at least MIMO S1 (or choose Project = Power Resilience)."); return
        if up is None:
            st.error("Upload a PDF to audit."); return

        pdf_bytes = up.read()

        st.info("Indexing PDF…")
        t0 = time.time()
        prog = st.progress(0.0)
        index = PdfIndex(pdf_bytes)
        index.build(progress=prog)
        prog.progress(1.0)
        t1 = time.time()

        st.info("Running checks…")
        rules = load_rules()
        allow = set(rules.get("allowlist", []))

        findings: List[Dict[str,Any]] = []

        # Spelling (fast & bounded)
        findings += spelling_findings(index, allow)

        # Rules
        findings += rules_findings(index, meta, rules)

        status = "Pass" if len(findings)==0 else "Rejected"
        t2 = time.time()

        # Annotate quickly (bounded)
        st.info("Annotating PDF…")
        annotated_bytes = annotate_pdf(index, findings)
        t3 = time.time()

        # Build Excel + persist
        ensure_dirs()
        df = pd.DataFrame(findings) if findings else pd.DataFrame(columns=["type","page","keyword","message","rule_id"])
        tag = f"{status}_{date_tag()}"
        base = os.path.splitext(up.name)[0]
        excel_name = safe_filename(f"{base}_{tag}.xlsx")
        pdf_ann_name = safe_filename(f"{base}_{tag}_annotated.pdf")
        excel_path = os.path.join(OUTPUT_DIR, excel_name)
        pdf_ann_path = os.path.join(OUTPUT_DIR, pdf_ann_name)

        with pd.ExcelWriter(excel_path, engine="openpyxl") as xw:
            df.to_excel(xw, index=False, sheet_name="Findings")
            pd.DataFrame([meta]).to_excel(xw, index=False, sheet_name="Metadata")

        with open(pdf_ann_path, "wb") as f:
            f.write(annotated_bytes)

        # History
        append_history(meta, status, up.name, excel_name)

        # Results
        st.success(f"Audit complete: **{status}**  ·  Index {t1-t0:.1f}s  ·  Checks {t2-t1:.1f}s  ·  Annot {t3-t2:.1f}s")

        cdl1, cdl2 = st.columns(2)
        with cdl1:
            with open(pdf_ann_path, "rb") as f:
                st.download_button("Download Annotated PDF", f.read(), file_name=pdf_ann_name, mime="application/pdf")
        with cdl2:
            with open(excel_path, "rb") as f:
                st.download_button("Download Excel Report", f.read(), file_name=excel_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.dataframe(df, use_container_width=True)

        # Close doc handle
        index.close()

# ======================================================================================
# ANALYTICS TAB
# ======================================================================================
def analytics_tab():
    st.subheader("Analytics & Trends")
    df = load_history()
    if df.empty:
        st.info("No audits yet.")
        return

    c1,c2,c3 = st.columns(3)
    f_supplier = c1.selectbox("Supplier", ["All"] + SUPPLIERS)
    f_client = c2.selectbox("Client", ["All"] + CLIENTS)
    f_project = c3.selectbox("Project", ["All"] + PROJECTS)

    show = df.copy()
    show = show[show.get("exclude", False) == False] if "exclude" in show.columns else show
    if f_supplier != "All": show = show[show["supplier"] == f_supplier]
    if f_client != "All": show = show[show["client"] == f_client]
    if f_project != "All": show = show[show["project"] == f_project]

    total = len(show)
    rft = round(100*len(show[show["status"]=="Pass"])/total,1) if total else 0.0

    m1,m2 = st.columns(2)
    m1.metric("Total Audits", total)
    m2.metric("Right First Time %", rft)

    if total and "timestamp_utc" in show.columns:
        try:
            show["_date"] = pd.to_datetime(show["timestamp_utc"]).dt.date
            trend = show.groupby("_date")["status"].apply(lambda s: (s=="Pass").mean()*100).reset_index(name="RFT%")
            st.line_chart(trend.set_index("_date"))
        except Exception:
            pass

    cols = [c for c in ["timestamp_utc","supplier","client","project","status","pdf_name","excel_name"] if c in show.columns]
    st.dataframe(show[cols], use_container_width=True)

# ======================================================================================
# SETTINGS TAB
# ======================================================================================
def settings_tab():
    st.subheader("Settings")
    st.markdown("**Rules editor (YAML)**")
    pw = st.text_input("Rules password", type="password")
    if pw != RULES_PASSWORD:
        st.warning("Enter rules password to edit.")
        return
    rules = load_rules()
    raw = yaml.safe_dump(rules, sort_keys=False, allow_unicode=True)
    text = st.text_area("rules_example.yaml", value=raw, height=420)
    c1,c2 = st.columns(2)
    if c1.button("Save rules"):
        try:
            new_rules = yaml.safe_load(text) or {}
            new_rules.setdefault("schema_version", 2)
            new_rules.setdefault("allowlist", [])
            new_rules.setdefault("forbid_pairs", [])
            new_rules.setdefault("checklist", [])
            save_rules(new_rules)
            st.success("Rules saved.")
        except Exception as e:
            st.error(f"YAML error: {e}")
    if c2.button("Restore defaults"):
        save_rules(DEFAULT_RULES.copy())
        st.success("Defaults restored.")

# ======================================================================================
# MAIN
# ======================================================================================
def main():
    st.set_page_config(page_title="AI Design Quality Auditor", layout="wide")
    ensure_dirs()
    password_gate()

    st.title("AI Design Quality Auditor")

    tabs = st.tabs(["Audit","Analytics","Settings"])
    with tabs[0]:
        audit_tab()
    with tabs[1]:
        analytics_tab()
    with tabs[2]:
        settings_tab()

if __name__ == "__main__":
    main()
