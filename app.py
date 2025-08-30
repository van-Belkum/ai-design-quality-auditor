import streamlit as st
import pandas as pd
import yaml
import fitz  # PyMuPDF
import io
import base64
from datetime import datetime

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Design QA", layout="wide")

# ---------- LOGO (always visible top-right) ----------
logo_path = "212BAAC2-5CB6-46A5-A53D-06497B78CF23.png"
st.markdown(
    f"""
    <style>
        .top-right-logo {{
            position: fixed;
            top: 10px;
            right: 20px;
            width: 120px;
            z-index: 1000;
        }}
    </style>
    <img src="data:image/png;base64,{base64.b64encode(open(logo_path,"rb").read()).decode()}" class="top-right-logo">
    """,
    unsafe_allow_html=True
)

# ---------- LOAD RULES ----------
def load_rules(path="rules_example.yaml"):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

# ---------- SPELLING CHECK (placeholder) ----------
def spell_findings(pages, allowlist):
    findings = []
    for i, text in enumerate(pages, start=1):
        if "ehybrid" in text.lower():
            findings.append({
                "page": i,
                "kind": "Spelling",
                "message": "Possible typo: 'ehybrid' → 'hybrid'",
                "bbox": None
            })
    return findings

# ---------- PDF ANNOTATION ----------
def annotate_pdf(pdf_bytes, findings):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for f in findings:
        bbox = f.get("bbox")
        if bbox and len(bbox) == 4:
            page = doc[f["page"] - 1]
            rect = fitz.Rect(bbox)
            highlight = page.add_rect_annot(rect)
            highlight.set_colors(stroke=(1, 0, 0))
            highlight.update()
            page.add_text_annot(rect.tl, f["message"])
    output = io.BytesIO()
    doc.save(output)
    return output.getvalue()

# ---------- EXCEL EXPORT ----------
def export_excel(findings, original_filename, outcome):
    df = pd.DataFrame(findings)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{original_filename}_{outcome}_{timestamp}.xlsx"
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue(), filename

# ---------- UI ----------
st.title("AI Design QA v8")

# ----------- METADATA FIELDS -----------
with st.form("metadata_form"):
    st.subheader("Audit Metadata (all fields required)")
    client = st.selectbox("Client", ["", "BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"])
    project = st.selectbox("Project", ["", "RAN", "Power Resilience", "East Unwind", "Beacon 4"])
    site_type = st.selectbox("Site Type", ["", "Greenfield", "Rooftop", "Streetworks"])
    vendor = st.selectbox("Proposed Vendor", ["", "Ericsson", "Nokia"])
    cab_loc = st.selectbox("Proposed Cabinet Location", ["", "Indoor", "Outdoor"])
    radio_loc = st.selectbox("Proposed Radio Location", ["", "High Level", "Low Level", "Indoor", "Door"])
    sectors = st.selectbox("Quantity of Sectors", ["", "1", "2", "3", "4", "5", "6"])
    mimo_cfg = st.text_input("Proposed MIMO Config")
    site_addr = st.text_input("Site Address")

    uploaded_file = st.file_uploader("Upload PDF to Audit", type=["pdf"])

    audit_btn = st.form_submit_button("Run Audit")
    clear_btn = st.form_submit_button("Clear Metadata")

# ---------- CLEAR METADATA ----------
if clear_btn:
    st.experimental_rerun()

# ---------- AUDIT LOGIC ----------
if audit_btn:
    # Enforce required metadata
    missing = []
    for label, val in {
        "Client": client, "Project": project, "Site Type": site_type,
        "Vendor": vendor, "Cabinet": cab_loc, "Radio": radio_loc,
        "Sectors": sectors, "MIMO": mimo_cfg, "Site Address": site_addr
    }.items():
        if not val:
            missing.append(label)

    if missing:
        st.error(f"Missing required metadata: {', '.join(missing)}")
    elif not uploaded_file:
        st.error("Please upload a PDF before running the audit.")
    else:
        pdf_bytes = uploaded_file.read()
        # Extract text (simplified)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]

        # Run rules
        rules = load_rules()
        allow = set(rules.get("allowlist", []))
        findings = spell_findings(pages, allow)

        # Decide outcome
        outcome = "REJECTED" if findings else "PASS"

        # Annotated PDF
        annotated_pdf = annotate_pdf(pdf_bytes, findings)

        # Export Excel
        excel_bytes, excel_filename = export_excel(findings, uploaded_file.name.replace(".pdf", ""), outcome)

        # Display summary
        st.write(f"**Audit Outcome:** {outcome}")
        st.download_button("Download Annotated PDF", annotated_pdf, file_name=f"{uploaded_file.name}_annotated.pdf")
        st.download_button("Download Findings Excel", excel_bytes, file_name=excel_filename)

        # Show findings in table
        if findings:
            st.dataframe(pd.DataFrame(findings))
