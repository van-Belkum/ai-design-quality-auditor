
import streamlit as st
import pandas as pd
import pdfplumber
import fitz  # PyMuPDF
import yaml
from spellchecker import SpellChecker
import io, re, hashlib, os, json
from datetime import datetime, timezone

APP_VERSION = "v7"
HISTORY_DIR = "history"
IGNORE_FILE = os.path.join(HISTORY_DIR, "ignore_rules.csv")
AUDIT_LOG = os.path.join(HISTORY_DIR, "audit_log.csv")

# ----------------------------------------
# Utilities
# ----------------------------------------

def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def ensure_history_files():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    if not os.path.exists(IGNORE_FILE):
        pd.DataFrame(columns=["client","kind","pattern","created_utc"]).to_csv(IGNORE_FILE, index=False)
    if not os.path.exists(AUDIT_LOG):
        pd.DataFrame(columns=[
            "timestamp_utc","user","client","project","site_type","vendor",
            "cabinet_location","radio_location","mimo_s1","mimo_s2","mimo_s3",
            "file","pages","outcome","total_findings","app_version"
        ]).to_csv(AUDIT_LOG, index=False)

def load_ignore_rules(client: str):
    try:
        df = pd.read_csv(IGNORE_FILE)
    except Exception:
        return []
    rules = []
    for _, row in df.iterrows():
        if row.get("client") in (client, "ANY", None) and isinstance(row.get("pattern"), str):
            rules.append(row["pattern"])
    return rules

def save_ignore_rules(new_rows: pd.DataFrame):
    ensure_history_files()
    existing = pd.read_csv(IGNORE_FILE) if os.path.exists(IGNORE_FILE) else pd.DataFrame(columns=new_rows.columns)
    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged.drop_duplicates(subset=["client","kind","pattern"], inplace=True)
    merged.to_csv(IGNORE_FILE, index=False)

def log_audit(row: dict):
    ensure_history_files()
    df = pd.read_csv(AUDIT_LOG)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(AUDIT_LOG, index=False)

def dedupe(seq):
    seen = set()
    out = []
    for x in seq:
        x = x.strip()
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def make_downloadable_excel(base_name, summary_df, findings_df, outcome):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    safe_base = os.path.splitext(os.path.basename(base_name))[0]
    fname = f"{safe_base}_QA_{outcome}_{ts}.xlsx"
    path = os.path.join("history", fname)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        findings_df.to_excel(writer, sheet_name="Findings", index=False)
    return path, fname

def annotate_pdf(pdf_bytes: bytes, findings_df: pd.DataFrame):
    # Using PyMuPDF to draw rectangles/comments; we assume 'boxes' column = list of [x0,y0,x1,y1] in PDF coords per page (0-based page index in 'page').
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for _, row in findings_df.iterrows():
        page_no = int(row.get("page", 1)) - 1
        if 0 <= page_no < len(doc):
            page = doc[page_no]
            try:
                # parse boxes if present
                boxes = row.get("boxes", None)
                if isinstance(boxes, str):
                    try:
                        boxes = json.loads(boxes)
                    except Exception:
                        boxes = None
                if not boxes:
                    # fallback: draw a small callout at top-left
                    rect = fitz.Rect(36, 36, 36+300, 36+20)
                    page.add_rect_annot(rect)
                    page.add_text_annot(rect.br, row.get("message","Finding"))
                else:
                    for b in boxes:
                        x0,y0,x1,y1 = b
                        rect = fitz.Rect(x0,y0,x1,y1)
                        page.add_rect_annot(rect)
                        page.add_text_annot(rect.br, row.get("message","Finding"))
            except Exception:
                pass
    out_bytes = io.BytesIO()
    doc.save(out_bytes)
    doc.close()
    return out_bytes.getvalue()

# ----------------------------------------
# YAML rules structure
# ----------------------------------------
DEFAULT_RULES = {
    "allowlist": ["fenceline","hybrid","utilised","flexi","polarium","Kathrein","Mafi","solarium"],
    "required_text": [
        {"text": "ALL DIMENSIONS IN MM", "category":"Checklist"},
        {"text": "NORTH", "category":"Checklist"}
    ],
    "forbidden_pairs": [
        {"if_contains": "Brush", "must_not_contain": "Generator Power", "message": "If 'Brush' present, 'Generator Power' must not be used."}
    ],
    "client_overrides": {
        "BTEE": {"allowlist": ["EE","BT"]},
        "Vodafone": {"allowlist": ["VF"]},
        "MBNL": {"allowlist": ["MBNL"]},
        "H3G": {"allowlist": ["Three","H3G"]},
        "Cornerstone": {"allowlist": ["CST"]},
        "Cellnex": {"allowlist": ["Cellnex"]}
    }
}

def load_rules(file) -> dict:
    if file is None:
        return DEFAULT_RULES
    try:
        blob = yaml.safe_load(file)
        if not isinstance(blob, dict):
            return DEFAULT_RULES
        return blob
    except Exception:
        return DEFAULT_RULES

# ----------------------------------------
# Core auditing
# ----------------------------------------

def extract_text_pages(pdf_bytes: bytes):
    pages = []
    boxes_by_page = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            pages.append(txt)
            # get words as boxes
            words = p.extract_words() or []
            boxes = [{"text": w.get("text",""), "box":[w["x0"], w["top"], w["x1"], w["bottom"]]} for w in words]
            boxes_by_page.append(boxes)
    return pages, boxes_by_page

def find_boxes_for_phrase(boxes, phrase):
    # simple: return the union box for tokens that match in sequence
    toks = phrase.split()
    if not toks:
        return []
    out = []
    # build list of word texts with indices
    words = [(i,b["text"]) for i,b in enumerate(boxes)]
    lower = [w[1].lower() for w in words]
    tok_lower = [t.lower() for t in toks]
    # sliding window
    for i in range(len(lower) - len(tok_lower) + 1):
        if lower[i:i+len(tok_lower)] == tok_lower:
            # merge boxes from i..i+len-1
            x0 = min(boxes[i]["box"][0], boxes[i+len(tok_lower)-1]["box"][0])
            y0 = min(b["box"][1] for b in boxes[i:i+len(tok_lower)])
            x1 = max(b["box"][2] for b in boxes[i:i+len(tok_lower)])
            y1 = max(b["box"][3] for b in boxes[i:i+len(tok_lower)])
            out.append([x0,y0,x1,y1])
    return out

def spell_findings(text_pages, allowlist_words, client_allowlist):
    sc = SpellChecker(distance=1)
    findings = []
    allow = set([w.lower() for w in allowlist_words] + [w.lower() for w in client_allowlist])
    word_re = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
    for idx, txt in enumerate(text_pages, start=1):
        words = word_re.findall(txt or "")
        misspelled = sc.unknown([w.lower() for w in words if w.lower() not in allow])
        for m in sorted(misspelled):
            suggestion = next(iter(sc.candidates(m)), None)
            msg = f"Possible typo: '{m}' -> '{suggestion}'" if suggestion else f"Possible typo: '{m}'"
            findings.append({"page": idx, "kind":"Spelling", "message": msg, "boxes": []})
    return findings

def checklist_findings(text_pages, boxes_by_page, rules):
    findings = []
    required = rules.get("required_text", [])
    for idx, txt in enumerate(text_pages, start=1):
        low = (txt or "").lower()
        for item in required:
            req = item.get("text","").strip()
            if not req:
                continue
            if req.lower() not in low:
                findings.append({"page": idx, "kind": item.get("category","Checklist"), "message": f"Missing required text: '{req}'", "boxes": []})
            else:
                # highlight location(s)
                boxes = find_boxes_for_phrase(boxes_by_page[idx-1], req)
                # Note: no finding when present; we could add "Present" informational entries if wanted.
                _ = boxes
    # forbidden pairs evaluated per page
    for idx, txt in enumerate(text_pages, start=1):
        low = (txt or "").lower()
        for rule in rules.get("forbidden_pairs", []):
            a = rule.get("if_contains","").lower()
            b = rule.get("must_not_contain","").lower()
            if a and b and (a in low) and (b in low):
                findings.append({"page": idx, "kind":"Checklist", "message": rule.get("message") or f"Forbidden combination: '{a}' with '{b}'", "boxes": []})
    return findings

def apply_ignore(findings_df: pd.DataFrame, ignore_patterns):
    if findings_df.empty or not ignore_patterns:
        return findings_df
    mask = pd.Series([True]*len(findings_df))
    for pat in ignore_patterns:
        # simple substring ignore (case-insensitive); treat pattern like literal
        lower_pat = str(pat).lower().strip()
        if not lower_pat:
            continue
        mask = mask & ~findings_df["message"].str.lower().str.contains(re.escape(lower_pat), na=False)
    return findings_df[mask].reset_index(drop=True)

def audit_pdf(pdf_bytes: bytes, metadata: dict, rules_blob: dict):
    text_pages, boxes_by_page = extract_text_pages(pdf_bytes)
    # Allowlist merge
    base_allow = rules_blob.get("allowlist", [])
    client = metadata.get("client","")
    client_allow = rules_blob.get("client_overrides", {}).get(client, {}).get("allowlist", [])
    findings = []
    findings += spell_findings(text_pages, base_allow, client_allow)
    findings += checklist_findings(text_pages, boxes_by_page, rules_blob)

    df = pd.DataFrame(findings, columns=["page","kind","message","boxes"])
    if "boxes" in df.columns:
        df["boxes"] = df["boxes"].apply(lambda x: json.dumps(x) if isinstance(x, list) else "[]")
    return df, len(text_pages)

# ----------------------------------------
# UI
# ----------------------------------------
st.set_page_config(page_title="AI Design QA", layout="wide")
st.title("AI Design QA " + APP_VERSION)

with st.expander("About", expanded=False):
    st.write("Batch-quality auditor for telecom design PDFs. This build includes a learning loop, audit history, annotated PDFs, and Excel exports.")

# Setup panel
st.subheader("Audit Setup")
colA, colB, colC = st.columns(3)
clients = ["BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"]
projects = ["RAN","Power Resilience","East Unwind","Beacon 4"]
site_types = ["Greenfield","Rooftop","Streetworks"]
vendors = ["Ericsson","Nokia"]
cabinet_locs = ["Indoor","Outdoor"]
radio_locs = ["High Level","Low Level","Indoor and Door"]

with colA:
    reviewer = st.text_input("Your name/initials (for history)")
    client = st.selectbox("Client", clients, index=0)
    project = st.selectbox("Project", projects, index=0)
with colB:
    site_type = st.selectbox("Site Type", site_types, index=0)
    vendor = st.selectbox("Proposed Vendor", vendors, index=0)
    cabinet = st.selectbox("Proposed Cabinet Location", cabinet_locs, index=0)
with colC:
    radio_loc = st.selectbox("Proposed Radio Location", radio_locs, index=0)

# MIMO lists
_MIMO_RAW = """
18\\21\\26 @4x4; 70\\80 @2x2
18\\21 @2x2
18\\21\\26 @4x4; 3500 @8x8
18\\21\\26 @2x2
18\\21\\26 @4x4
18\\21 @4x4; 70\\80 @2x4
18\\21\\26 @4x4; 70\\80 @2x4; 3500 @32x32
18\\21\\26 @4x4; 70\\80 @2x2; 3500 @8x8
18\\21\\26 @2x2; 70\\80 @2x2; 3500 @8x8
18\\21 @2x2; 3500 @32x32
18 @2x2
3500 @8x8
3500 @32x32
"""
mimo_options = dedupe([ln.strip() for ln in _MIMO_RAW.strip().splitlines() if ln.strip()])
same_mimo = st.checkbox("Use the same Proposed MIMO config for S1/S2/S3", value=True)
colM1, colM2, colM3 = st.columns(3)
with colM1:
    mimo_s1 = st.selectbox("Proposed Mimo S1", mimo_options, index=0)
with colM2:
    mimo_s2 = st.selectbox("Proposed Mimo S2", mimo_options, index=0 if same_mimo else min(1,len(mimo_options)-1))
with colM3:
    mimo_s3 = st.selectbox("Proposed Mimo S3", mimo_options, index=0 if same_mimo else min(2,len(mimo_options)-1))
if same_mimo:
    mimo_s2, mimo_s3 = mimo_s1, mimo_s1

# File inputs
st.subheader("Inputs")
pdf_file = st.file_uploader("Upload a single PDF to audit", type=["pdf"])
rules_upload = st.file_uploader("Optional: Rules pack (YAML)", type=["yaml","yml"])
feedback_csv = st.file_uploader("Optional: Learning feedback CSV (file,page,kind,message,disposition)", type=["csv"])

# Load rules
rules_blob = load_rules(rules_upload)

# Apply learning CSV to ignore list
if feedback_csv is not None:
    try:
        fb = pd.read_csv(feedback_csv)
        if not fb.empty and "disposition" in fb.columns:
            not_valid = fb[fb["disposition"].astype(str).str.lower() == "not valid"]
            if not not_valid.empty:
                rows = []
                for _, r in not_valid.iterrows():
                    patt = str(r.get("message","")).strip()
                    if patt:
                        rows.append({"client": client, "kind": r.get("kind",""), "pattern": patt, "created_utc": datetime.now(timezone.utc).isoformat()})
                if rows:
                    save_ignore_rules(pd.DataFrame(rows))
                    st.success(f"Added {len(rows)} ignore rule(s) from feedback.")
    except Exception as e:
        st.warning(f"Could not process feedback CSV: {e}")

# Run audit
if st.button("Run Audit", type="primary", disabled=pdf_file is None):
    ensure_history_files()
    if pdf_file is None:
        st.stop()
    pdf_bytes = pdf_file.read()

    meta = {
        "reviewer": reviewer,
        "client": client,
        "project": project,
        "site_type": site_type,
        "vendor": vendor,
        "cabinet_location": cabinet,
        "radio_location": radio_loc,
        "mimo_s1": mimo_s1, "mimo_s2": mimo_s2, "mimo_s3": mimo_s3
    }

    with st.spinner("Auditing..."):
        findings_df, num_pages = audit_pdf(pdf_bytes, meta, rules_blob)
        # Apply ignore rules (learning loop)
        ignore_patterns = load_ignore_rules(client)
        filtered = apply_ignore(findings_df, ignore_patterns)

        # Outcome
        outcome = "Pass" if filtered.empty else "Rejected"

        # Summary table
        summary = {
            "file": [pdf_file.name],
            "status": [outcome.upper()],
            "Spelling": [int((filtered["kind"]=="Spelling").sum() if not filtered.empty else 0)],
            "Checklist": [int((filtered["kind"]=="Checklist").sum() if not filtered.empty else 0)],
            "pages": [num_pages]
        }
        summary_df = pd.DataFrame(summary)
        st.subheader("Results")
        st.dataframe(summary_df, use_container_width=True)

        # Findings table
        if filtered.empty:
            st.success("✅ QA PASS — please continue with Second Check.")
        else:
            st.error("❌ REJECTED — findings below.")
            st.dataframe(filtered, use_container_width=True)

        # Annotated PDF
        annotated = annotate_pdf(pdf_bytes, filtered if not filtered.empty else findings_df.head(0))
        st.download_button("Download annotated PDF", data=annotated, file_name=os.path.splitext(pdf_file.name)[0] + "_annotated.pdf", mime="application/pdf")

        # Excel report
        excel_path, excel_name = make_downloadable_excel(pdf_file.name, summary_df, filtered if not filtered.empty else pd.DataFrame(columns=["page","kind","message","boxes"]), outcome.upper())
        with open(excel_path, "rb") as fh:
            st.download_button("Download Excel Report", data=fh.read(), file_name=excel_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Log audit
        log_audit({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "user": reviewer,
            "client": client,
            "project": project,
            "site_type": site_type,
            "vendor": vendor,
            "cabinet_location": cabinet,
            "radio_location": radio_loc,
            "mimo_s1": mimo_s1, "mimo_s2": mimo_s2, "mimo_s3": mimo_s3,
            "file": pdf_file.name,
            "pages": num_pages,
            "outcome": outcome.upper(),
            "total_findings": len(filtered) if not filtered.empty else 0,
            "app_version": APP_VERSION
        })

st.subheader("Audit history (latest 200)")
ensure_history_files()
try:
    hist = pd.read_csv(AUDIT_LOG)
    hist = hist.sort_values("timestamp_utc", ascending=False).head(200).reset_index(drop=True)
    st.dataframe(hist, use_container_width=True)
except Exception as e:
    st.info("No history yet.")
