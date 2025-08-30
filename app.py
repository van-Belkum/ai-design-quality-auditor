import os
import io
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path
import yaml

# --- Safe runtime defaults ---
HISTORY_DIR = Path(os.environ.get("HISTORY_DIR", "/tmp/qa_history"))
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

try:
    import pytesseract  # noqa: F401
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

try:
    import pdf2image  # noqa: F401
    HAS_PDF2IMAGE = True
except Exception:
    HAS_PDF2IMAGE = False

try:
    import cv2  # noqa: F401
    HAS_OPENCV = True
except Exception:
    HAS_OPENCV = False

# --- Load rules ---
def load_rules(yaml_file):
    try:
        return yaml.safe_load(yaml_file) or {}
    except Exception as e:
        st.error(f"Failed to load rules file: {e}")
        return {}

# --- Extract text ---
def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page in doc:
        text = page.get_text("text")
        pages.append((page.number+1, text))
    return pages

# --- Core checks ---
def spelling_check(pages, allowlist):
    findings = []
    for page_num, text in pages:
        words = text.split()
        for w in words:
            if w.isalpha() and w.lower() not in allowlist:
                findings.append({
                    "page": page_num,
                    "kind": "Spelling",
                    "message": f"Possible typo: {w}"
                })
    return findings

def checklist_check(pages, checklist_rules):
    findings = []
    for page_num, text in pages:
        for required_text in checklist_rules:
            if required_text.lower() not in text.lower():
                findings.append({
                    "page": page_num,
                    "kind": "Checklist",
                    "message": f"Missing required text: '{required_text}'"
                })
    return findings

# --- Run audit ---
def audit_pdf(file, rules):
    file_bytes = file.read()
    pages = extract_text_from_pdf(file_bytes)

    allowlist = set([w.lower() for w in rules.get("allowlist", [])])
    checklist_rules = rules.get("checklist", [])

    findings = []
    findings += spelling_check(pages, allowlist)
    findings += checklist_check(pages, checklist_rules)

    df = pd.DataFrame(findings)
    df.insert(0, "file", file.name)
    return df

# --- Save results ---
def save_audit_results(df, user, status):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_stem = Path(df["file"].iloc[0]).stem
    out_name = f"{file_stem}_{status}_{timestamp}.xlsx"

    out_path = HISTORY_DIR / out_name
    df.to_excel(out_path, index=False)

    history_file = HISTORY_DIR / "audit_history.csv"
    meta = pd.DataFrame([{
        "timestamp_utc": timestamp,
        "user": user,
        "file": file_stem,
        "status": status,
        "total_findings": len(df)
    }])
    if history_file.exists():
        meta.to_csv(history_file, mode="a", index=False, header=False)
    else:
        meta.to_csv(history_file, index=False)

    return out_path

# --- Streamlit UI ---
st.set_page_config(page_title="AI Design QA v7 Fixed", layout="wide")
st.title("AI Design QA v7 Fixed")

st.sidebar.header("Run Audit")

rules_file = st.sidebar.file_uploader("Upload Rules (YAML)", type=["yaml", "yml"])
pdf_file = st.sidebar.file_uploader("Upload Design PDF", type=["pdf"])

user = st.sidebar.text_input("Your Name / Initials", "QA_User")

if st.sidebar.button("Run Audit"):
    if not pdf_file:
        st.error("Please upload a PDF.")
    else:
        rules = load_rules(rules_file) if rules_file else {}
        df = audit_pdf(pdf_file, rules)

        status = "PASS" if df.empty else "REJECTED"

        st.subheader("Findings")
        if df.empty:
            st.success("QA Pass – please continue with Second Check ✅")
        else:
            st.error("QA Rejected – issues found ❌")
            st.dataframe(df)

        out_path = save_audit_results(df, user, status)
        with open(out_path, "rb") as f:
            st.download_button("Download Excel Report", f, file_name=out_path.name)

st.sidebar.header("Audit History")
history_file = HISTORY_DIR / "audit_history.csv"
if history_file.exists():
    hist = pd.read_csv(history_file).tail(20)
    st.sidebar.dataframe(hist)
