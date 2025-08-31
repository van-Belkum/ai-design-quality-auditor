import streamlit as st
import pandas as pd
import yaml
import io
import base64
from datetime import datetime, timedelta
import fitz  # PyMuPDF
from rapidfuzz import fuzz
from typing import List, Dict, Any
import pytesseract
from pdf2image import convert_from_bytes

# ---------- CONFIG ----------
APP_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"
HISTORY_FILE = "history/history.csv"
RULES_FILE = "rules_example.yaml"
LOGO_PATH = "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png"

SUPPLIERS = [
    "CEG","CTIL","Emfyser","Innov8","Invict","KTL Team (Internal)","Trylon"
]
DRAWING_TYPES = ["General Arrangement", "Detailed Design"]
CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
CAB_LOCATIONS = ["Indoor", "Outdoor"]
RADIO_LOCATIONS = ["Low Level", "Midway", "High Level", "Unique Coverage"]
MIMO_CONFIGS = [
    "18 @2x2","18 @2x2; 26 @4x4","18 @2x2; 70\\80 @2x2","18 @2x2; 80 @2x2",
    "18\\21 @2x2","18\\21 @2x2; 26 @4x4","18\\21 @2x2; 3500 @32x32","18\\21 @2x2; 70\\80 @2x2",
    "18\\21 @2x2; 80 @2x2","18\\21 @4x4","18\\21 @4x4; 3500 @32x32","18\\21 @4x4; 70 @2x4",
    "18\\21 @4x4; 70\\80 @2x2","18\\21 @4x4; 70\\80 @2x2; 3500 @32x32","18\\21 @4x4; 70\\80 @2x4",
    "18\\21 @4x4; 70\\80 @2x4; 3500 @32x32","18\\21 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21@4x4; 70\\80 @2x2","18\\21@4x4; 70\\80 @2x4","18\\21\\26 @2x2",
    "18\\21\\26 @2x2; 3500 @32x32","18\\21\\26 @2x2; 3500 @8X8","18\\21\\26 @2x2; 70\\80 @2x2",
    "18\\21\\26 @2x2; 70\\80 @2x2; 3500 @32x32","18\\21\\26 @2x2; 70\\80 @2x2; 3500 @8x8",
    "18\\21\\26 @2x2; 70\\80 @2x4; 3500 @32x32","18\\21\\26 @2x2; 80 @2x2",
    "18\\21\\26 @2x2; 80 @2x2; 3500 @8x8","18\\21\\26 @4x4","18\\21\\26 @4x4; 3500 @32x32",
    "18\\21\\26 @4x4; 3500 @8x8","18\\21\\26 @4x4; 70 @2x4; 3500 @8x8",
    "18\\21\\26 @4x4; 70\\80 @2x2","18\\21\\26 @4x4; 70\\80 @2x2; 3500 @32x32",
    "18\\21\\26 @4x4; 70\\80 @2x2; 3500 @8x8","18\\21\\26 @4x4; 70\\80 @2x4",
    "18\\21\\26 @4x4; 70\\80 @2x4; 3500 @32x32","18\\21\\26 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21\\26 @4x4; 80 @2x2","18\\21\\26 @4x4; 80 @2x2; 3500 @32x32",
    "18\\21\\26 @4x4; 80 @2x4","18\\21\\26 @4x4; 80 @2x4; 3500 @8x8",
    "18\\26 @2x2","18\\26 @4x4; 21 @2x2; 80 @2x2","(blank)"
]

# ---------- UTILS ----------
def now_utc_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat()

def load_rules(path: str):
    try:
        with open(path,"r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {"checklist":[]}

def save_rules(path: str, data: Dict[str,Any]):
    with open(path,"w") as f:
        yaml.safe_dump(data,f,sort_keys=False)

def load_history() -> pd.DataFrame:
    try:
        return pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame()

def save_history(df: pd.DataFrame):
    df.to_csv(HISTORY_FILE,index=False)

def annotate_pdf(original_bytes: bytes, findings: List[Dict[str,Any]]) -> bytes:
    doc = fitz.open("pdf", original_bytes)
    for f in findings:
        page_num = f.get("page_num")
        if page_num is not None and 0 <= page_num < len(doc):
            page = doc[page_num]
            text_instances = page.search_for(f.get("message","")[:15])
            for inst in text_instances:
                page.add_rect_annot(inst).set_colors({"stroke":(1,0,0)}).update()
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

def spelling_checks(pages: List[str], allow: List[str]):
    findings = []
    for i,p in enumerate(pages):
        words = p.split()
        for w in words:
            if w.isalpha() and w.lower() not in allow:
                findings.append({"rule_id":"spelling","severity":"minor","message":f"Unknown word: {w}","page_num":i})
    return findings

def make_excel(findings: List[Dict[str,Any]], meta: Dict[str,Any], original_name: str, status: str) -> bytes:
    df = pd.DataFrame(findings) if findings else pd.DataFrame(columns=["rule_id","severity","message","page_num"])
    meta_row = pd.DataFrame([{
        **meta,
        "original_file": original_name,
        "status": status,
        "generated_utc": now_utc_iso(),
    }])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Findings")
        meta_row.to_excel(writer, index=False, sheet_name="Metadata")
    buf.seek(0)
    return buf.getvalue()

# ---------- APP ----------
def main():
    st.set_page_config("AI Design Quality Auditor", layout="wide")

    # Logo
    try:
        with open(LOGO_PATH,"rb") as f:
            img64 = base64.b64encode(f.read()).decode()
        st.sidebar.markdown(f"<img src='data:image/png;base64,{img64}' style='max-width:180px;'>", unsafe_allow_html=True)
    except:
        st.sidebar.warning("⚠️ Logo not found")

    # Login
    if "logged_in" not in st.session_state:
        pwd = st.text_input("Enter password to continue", type="password")
        if pwd == APP_PASSWORD:
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.stop()

    tab_audit, tab_history, tab_analytics, tab_settings = st.tabs(["Audit","History","Analytics","Settings"])

    with tab_audit:
        st.header("Audit Metadata (required)")
        col1,col2,col3,col4 = st.columns(4)
        supplier = col1.selectbox("Supplier", SUPPLIERS)
        drawing_type = col2.selectbox("Drawing Type", DRAWING_TYPES)
        client = col3.selectbox("Client", CLIENTS)
        project = col4.selectbox("Project", PROJECTS)
        site_type = col1.selectbox("Site Type", SITE_TYPES)
        vendor = col2.selectbox("Proposed Vendor", VENDORS)
        cab_loc = col3.selectbox("Proposed Cabinet Location", CAB_LOCATIONS)
        radio_loc = col4.selectbox("Proposed Radio Location", RADIO_LOCATIONS)
        qty = col1.selectbox("Quantity of Sectors",[1,2,3,4,5,6])
        site_address = col2.text_input("Site Address")

        mimo_same = st.checkbox("Use same config for all")
        sector_cfg = {}
        for i in range(1, qty+1):
            if mimo_same and i>1:
                sector_cfg[f"S{i}"] = sector_cfg["S1"]
            else:
                sector_cfg[f"S{i}"] = st.selectbox(f"MIMO S{i}", MIMO_CONFIGS, key=f"mimo_{i}")

        file = st.file_uploader("Upload PDF design", type="pdf")
        if file and st.button("Run Audit"):
            pages = [pytesseract.image_to_string(img) for img in convert_from_bytes(file.read())]
            rules = load_rules(RULES_FILE).get("checklist",[])
            findings = []
            for r in rules:
                for i,p in enumerate(pages):
                    for must in r.get("must_contain",[]):
                        if must not in p:
                            findings.append({"rule_id":r["name"],"severity":r["severity"],"message":f"Missing {must}","page_num":i})
            findings += spelling_checks(pages,["the","and","to"]) # example allowlist
            status = "Rejected" if findings else "Pass"
            meta = dict(supplier=supplier,drawing_type=drawing_type,client=client,project=project,
                        site_type=site_type,vendor=vendor,cab_loc=cab_loc,radio_loc=radio_loc,
                        qty=qty,site_address=site_address,sector_cfg=str(sector_cfg))
            pdf_bytes = annotate_pdf(file.getvalue(), findings)
            excel_bytes = make_excel(findings, meta, file.name, status)

            st.download_button("⬇️ Download Excel", excel_bytes, file_name=f"{file.name}_{status}_{datetime.now().date()}.xlsx")
            st.download_button("⬇️ Download Annotated PDF", pdf_bytes, file_name=f"{file.name}_{status}_{datetime.now().date()}.pdf")

            row = {**meta,"status":status,"findings_count":len(findings),"timestamp_utc":now_utc_iso(),"exclude":False}
            dfh = load_history()
            dfh = pd.concat([dfh,pd.DataFrame([row])],ignore_index=True)
            save_history(dfh)

    with tab_history:
        st.subheader("History")
        dfh = load_history()
        if not dfh.empty:
            st.dataframe(dfh)
            idx = st.number_input("Index to exclude/include",0,len(dfh)-1,0)
            if st.button("Toggle Exclude"):
                dfh.loc[idx,"exclude"] = not dfh.loc[idx,"exclude"]
                save_history(dfh)
                st.experimental_rerun()

    with tab_analytics:
        st.subheader("Analytics")
        dfh = load_history()
        if not dfh.empty:
            dfu = dfh[dfh["exclude"]!=True]
            supplier_filter = st.selectbox("Filter by Supplier",["All"]+SUPPLIERS)
            if supplier_filter!="All":
                dfu = dfu[dfu["supplier"]==supplier_filter]
            st.metric("Total Audits",len(dfu))
            st.metric("Pass Rate",f"{(dfu['status']=='Pass').mean()*100:.1f}%")

    with tab_settings:
        st.subheader("Edit rules (YAML)")
        pwd = st.text_input("Rules password", type="password")
        if pwd == RULES_PASSWORD:
            content = st.text_area("rules_example.yaml", value=open(RULES_FILE).read(), height=300)
            if st.button("Save rules"):
                try:
                    data = yaml.safe_load(content)
                    save_rules(RULES_FILE,data)
                    st.success("Rules updated")
                except Exception as e:
                    st.error(str(e))
        if st.button("Clear all history (keeps files)"):
            save_history(pd.DataFrame())
            st.success("History cleared")

if __name__=="__main__":
    main()
