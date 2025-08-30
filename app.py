import streamlit as st
import pandas as pd
import yaml
import base64
import os
import io
import datetime
import fitz  # PyMuPDF
from rapidfuzz import process, fuzz

# ----------------------------
# Utility: load rules from YAML
# ----------------------------
def load_rules(path="rules_example.yaml"):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

# ----------------------------
# Utility: save annotated PDF
# ----------------------------
def annotate_pdf(file_bytes, findings):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for f in findings:
        page_num = f.get("page", None)
        msg = f.get("message", "")
        bbox = f.get("boxes", None)
        if page_num is not None and 0 <= page_num < len(doc):
            page = doc[page_num]
            if bbox and len(bbox) == 4:
                rect = fitz.Rect(bbox)
                highlight = page.addHighlightAnnot(rect)
                highlight.setInfo(title="AI QA", content=msg)
            else:
                page.insert_text((72, 72), f"[QA] {msg}", fontsize=8, color=(1, 0, 0))
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

# ----------------------------
# Spelling check (very simple fuzzy)
# ----------------------------
def spelling_checks(pages, allowlist):
    findings = []
    dictionary = ["approve", "hybrid", "utilize", "really", "flex", "singlemode", "swan"]
    for page_num, text in enumerate(pages):
        words = text.split()
        for w in words:
            wl = w.lower().strip(",.;:!?()[]{}")
            if wl in allowlist or wl in dictionary:
                continue
            match, score, _ = process.extractOne(wl, dictionary, scorer=fuzz.ratio)
            if score > 80:
                findings.append({
                    "page": page_num,
                    "kind": "Spelling",
                    "message": f"Possible typo: '{wl}' → '{match}'",
                    "boxes": None
                })
    return findings

# ----------------------------
# Checklist checks
# ----------------------------
def checklist_checks(pages, rules):
    findings = []
    required = rules.get("required_text", [])
    for req in required:
        found = any(req.lower() in page.lower() for page in pages)
        if not found:
            findings.append({
                "page": 0,
                "kind": "Checklist",
                "message": f"Missing required text: '{req}'",
                "boxes": None
            })
    return findings

# ----------------------------
# Run audit
# ----------------------------
def run_audit(pdf_file, rules, metadata):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages = [page.get_text("text") for page in doc]

    findings = []
    findings += spelling_checks(pages, set(rules.get("allowlist", [])))
    findings += checklist_checks(pages, rules)

    # Status
    status = "PASS" if len(findings) == 0 else "REJECTED"

    # Save Excel report
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_name = metadata["file_name"].replace(".pdf", "")
    excel_name = f"{base_name}_{status}_{timestamp}.xlsx"

    df = pd.DataFrame(findings)
    df.to_excel(excel_name, index=False)

    # Annotated PDF
    annotated_pdf = annotate_pdf(pdf_file.getvalue(), findings)

    return status, df, excel_name, annotated_pdf

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Design QA", layout="wide")

# Logo rendering
logo_file = None
for candidate in ["logo.png", "logo.jpg", "logo.svg"]:
    if os.path.exists(candidate):
        logo_file = candidate
        break

if logo_file:
    logo_bytes = open(logo_file, "rb").read()
    logo_b64 = base64.b64encode(logo_bytes).decode()
    st.markdown(
        f"""
        <style>
        .top-right-logo {{
            position: absolute;
            top: 20px;
            right: 20px;
            width: 120px;
        }}
        </style>
        <img src="data:image/png;base64,{logo_b64}" class="top-right-logo">
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning("⚠️ Logo file not found in repo root (png/jpg/svg).")

st.title("AI Design QA Tool v8")

# ----------------------------
# Metadata form
# ----------------------------
st.subheader("Audit Metadata (All Required)")
with st.form("metadata_form"):
    client = st.selectbox("Client", ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"])
    project = st.selectbox("Project", ["RAN", "Power Resilience", "East Unwind", "Beacon 4"])
    site_type = st.selectbox("Site Type", ["Greenfield", "Rooftop", "Streetworks"])
    vendor = st.selectbox("Proposed Vendor", ["Ericsson", "Nokia"])
    cabinet = st.selectbox("Proposed Cabinet Location", ["Indoor", "Outdoor"])
    radio = st.selectbox("Proposed Radio Location", ["High Level", "Low Level", "Indoor and Door"])
    sectors = st.selectbox("Quantity of Sectors", [1, 2, 3, 4, 5, 6])

    # Hide MIMO config if Power Resilience
    mimo = None
    if project != "Power Resilience":
        mimo = st.text_input("Proposed MIMO Config")

    site_address = st.text_input("Site Address")

    pdf_file = st.file_uploader("Upload PDF for Audit", type=["pdf"])

    submitted = st.form_submit_button("Run Audit")

if submitted:
    if not pdf_file:
        st.error("❌ Please upload a PDF.")
    elif not site_address:
        st.error("❌ Please enter Site Address.")
    else:
        metadata = {
            "client": client,
            "project": project,
            "site_type": site_type,
            "vendor": vendor,
            "cabinet": cabinet,
            "radio": radio,
            "sectors": sectors,
            "mimo": mimo,
            "site_address": site_address,
            "file_name": pdf_file.name,
        }
        rules = load_rules()
        status, df, excel_name, annotated_pdf = run_audit(pdf_file, rules, metadata)

        st.subheader("Audit Result")
        st.write(f"**Status:** {status}")

        st.download_button("⬇️ Download Excel Report", open(excel_name, "rb"), file_name=excel_name)
        st.download_button("⬇️ Download Annotated PDF", annotated_pdf, file_name=f"annotated_{pdf_file.name}")

        st.dataframe(df)

# Clear Metadata button
if st.button("Clear All Metadata"):
    st.experimental_rerun()
