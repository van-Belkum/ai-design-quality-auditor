import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import io, os, base64, datetime
from pathlib import Path
from rapidfuzz import fuzz, process

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="AI Design QA", layout="wide")

# -----------------------
# Logo auto-detect
# -----------------------
def _encode_image(fp: Path) -> str:
    ext = fp.suffix.lower()
    if ext == ".svg":
        return f'data:image/svg+xml;base64,{base64.b64encode(fp.read_bytes()).decode()}'
    mime = "image/png" if ext == ".png" else ("image/jpeg" if ext in [".jpg", ".jpeg"] else "image/octet-stream")
    return f'data:{mime};base64,{base64.b64encode(fp.read_bytes()).decode()}'

def render_top_right_logo():
    preferred = [
        "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png",
        "212BAAC2-5CB6-46A5-A53D-06497B78CF23.png",
        "logo.png", "seker.png", "seker.svg"
    ]
    root = Path(__file__).parent
    for name in preferred:
        p = root / name
        if p.exists():
            src = _encode_image(p)
            st.markdown(f"""
                <style>
                  .top-right-logo {{
                    position: fixed; top: 12px; right: 16px; z-index: 9999;
                    width: 132px; height: auto; opacity: .95;
                  }}
                  @media (max-width: 900px) {{
                    .top-right-logo {{ width: 96px; }}
                  }}
                </style>
                <img src="{src}" class="top-right-logo" />
                """, unsafe_allow_html=True)
            return
    st.info("⚠️ Logo file not found in repo root (png/svg/jpg).", icon="⚠️")

render_top_right_logo()

# -----------------------
# Metadata state
# -----------------------
if "metadata" not in st.session_state:
    st.session_state["metadata"] = {
        "Client": "", "Project": "", "Site Type": "",
        "Proposed Vendor": "", "Cabinet Location": "",
        "Radio Location": "", "Sectors": "", "MIMO Config": "",
        "Site Address": ""
    }

def clear_metadata():
    for k in st.session_state["metadata"]:
        st.session_state["metadata"][k] = ""

# -----------------------
# Metadata inputs
# -----------------------
st.sidebar.header("Audit Metadata")

clients = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
projects = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
site_types = ["Greenfield", "Rooftop", "Streetworks"]
vendors = ["Ericsson", "Nokia"]
cab_locs = ["Indoor", "Outdoor"]
radio_locs = ["High Level", "Low Level", "Indoor", "Door"]
sectors = ["1","2","3","4","5","6"]

st.session_state["metadata"]["Client"] = st.sidebar.selectbox("Client", clients, index=0)
st.session_state["metadata"]["Project"] = st.sidebar.selectbox("Project", projects, index=0)
st.session_state["metadata"]["Site Type"] = st.sidebar.selectbox("Site Type", site_types, index=0)
st.session_state["metadata"]["Proposed Vendor"] = st.sidebar.selectbox("Proposed Vendor", vendors, index=0)
st.session_state["metadata"]["Cabinet Location"] = st.sidebar.selectbox("Proposed Cabinet Location", cab_locs, index=0)
st.session_state["metadata"]["Radio Location"] = st.sidebar.selectbox("Proposed Radio Location", radio_locs, index=0)
st.session_state["metadata"]["Sectors"] = st.sidebar.selectbox("Quantity of Sectors", sectors, index=0)

# Hide MIMO if Project == Power Resilience
if st.session_state["metadata"]["Project"] != "Power Resilience":
    st.session_state["metadata"]["MIMO Config"] = st.sidebar.text_input("Proposed MIMO Config", value="")
else:
    st.session_state["metadata"]["MIMO Config"] = "N/A"

st.session_state["metadata"]["Site Address"] = st.sidebar.text_area("Site Address", value="")

colb1, colb2 = st.sidebar.columns(2)
run_audit = colb1.button("▶ Run Audit")
colb2.button("🗑 Clear Metadata", on_click=clear_metadata)

# -----------------------
# Helpers
# -----------------------
def extract_text_from_pdf(file_bytes):
    text_pages, boxes = [], []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for i in range(len(doc)):
        page = doc[i]
        text_pages.append(page.get_text("text"))
        boxes.append(page.get_text("blocks"))
    return text_pages, boxes

def spelling_checks(pages, allow):
    findings = []
    for pi, page in enumerate(pages, start=1):
        words = page.split()
        for w in words:
            wl = w.strip().lower()
            if wl and wl not in allow:
                sug = process.extractOne(wl, allow, scorer=fuzz.ratio)
                if sug and sug[1] > 80:  # close match
                    findings.append({
                        "page": pi,
                        "kind": "Spelling",
                        "message": f"Possible typo: '{w}' → '{sug[0]}'",
                        "boxes": None
                    })
    return findings

def annotate_pdf(file_bytes, findings):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for f in findings:
        if f.get("boxes"):
            page = doc[f["page"] - 1]
            bbox = f["boxes"]
            if bbox and len(bbox) == 4:
                rect = fitz.Rect(bbox)
                page.add_rect_annot(rect).set_colors(stroke=(1, 0, 0))
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

# -----------------------
# Main Audit
# -----------------------
st.title("📑 AI Design QA")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if run_audit and uploaded:
    # Check metadata required
    missing = [k for k,v in st.session_state["metadata"].items() if v.strip()==""]
    if missing:
        st.error(f"Please fill all metadata before running audit: {', '.join(missing)}")
    else:
        bytes_in = uploaded.read()
        pages, boxes = extract_text_from_pdf(bytes_in)

        allow = set(["the","and","site","cabinet","antenna"])  # demo allowlist
        findings = []
        findings += spelling_checks(pages, allow)

        df = pd.DataFrame(findings)
        st.subheader("Findings")
        if df.empty:
            st.success("✅ No issues found, QA Pass. Please continue with Second Check.")
            status = "Pass"
        else:
            st.error("❌ Issues found, QA Rejected.")
            st.dataframe(df)
            status = "Rejected"

        # Save Excel
        fname = uploaded.name.replace(".pdf","")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        xname = f"{fname}_{status}_{ts}.xlsx"
        df.to_excel(xname, index=False)
        with open(xname,"rb") as f:
            st.download_button("⬇ Download Excel Report", f, file_name=xname)

        # Save annotated PDF
        pdf_bytes = annotate_pdf(bytes_in, findings)
        pname = f"{fname}_{status}_{ts}_annotated.pdf"
        st.download_button("⬇ Download Annotated PDF", pdf_bytes, file_name=pname)
