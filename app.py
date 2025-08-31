import streamlit as st
import pandas as pd
import numpy as np
import yaml
import fitz  # PyMuPDF
import io, os, base64, datetime
from spellchecker import SpellChecker
from typing import List, Dict, Any

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
APP_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"

HISTORY_FILE = "audit_history.csv"
RULES_FILE = "rules_example.yaml"
LOGO_FILE = "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png"

DEFAULT_SUPPLIERS = [
    "CEG","CTIL","Emfyser","Innov8","Invict","KTL Team (Internal)","Trylon"
]

DEFAULT_PICKLISTS = {
    "suppliers": DEFAULT_SUPPLIERS,
    "clients": ["BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"],
    "projects": ["RAN","Power Resilience","East Unwind","Beacon 4"],
    "vendors": ["Ericsson","Nokia"],
    "drawing_types": ["General Arrangement","Detailed Design"],
    "site_types": ["Greenfield","Rooftop","Streetworks"],
    "cabinet_locations": ["Indoor","Outdoor"],
    "radio_locations": ["Low Level","Midway","High Level","Unique Coverage"],
    "mimo_configs": [
        "18 @2x2","18 @2x2; 26 @4x4","18\\21 @4x4; 70\\80 @2x2","18\\21 @4x4; 70\\80 @2x2; 3500 @32x32"
    ]
}

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat()

def load_rules(path: str = RULES_FILE) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def save_rules(data: dict, path: str = RULES_FILE):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)

def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame()

def save_history(df: pd.DataFrame):
    df.to_csv(HISTORY_FILE, index=False)

def annotate_pdf(pdf_bytes: bytes, findings: List[Dict[str,Any]]) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for f in findings:
        page_num = f.get("page",1)-1
        detail = f.get("detail","")
        try:
            page = doc[page_num]
            rect = fitz.Rect(50, 50, 400, 100)
            page.add_highlight_annot(rect)
            page.insert_textbox(rect, detail, fontsize=8, color=(1,0,0))
        except Exception:
            continue
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

def make_excel(findings: List[Dict[str,Any]], meta: Dict[str,Any], src_pdf: str, status: str) -> bytes:
    df = pd.DataFrame(findings)
    with io.BytesIO() as mem:
        with pd.ExcelWriter(mem, engine="xlsxwriter") as xw:
            summary = pd.DataFrame([{
                **meta,
                "source_pdf": src_pdf,
                "status": status,
                "timestamp_utc": now_iso(),
                "total_findings": len(findings),
                "majors": int((df.severity=="major").sum()) if not df.empty else 0,
                "minors": int((df.severity=="minor").sum()) if not df.empty else 0
            }])
            summary.to_excel(xw, index=False, sheet_name="Summary")
            (
                df if not df.empty else pd.DataFrame(columns=["page","rule","severity","detail"])
            ).to_excel(xw, index=False, sheet_name="Findings")
        return mem.getvalue()

def spelling_findings(pages: List[str], allow: set) -> List[Dict[str,Any]]:
    sp = SpellChecker()
    finds=[]
    for i, text in enumerate(pages, start=1):
        for w in text.split():
            wl = w.strip(",.()").lower()
            if wl and wl not in allow and wl not in sp:
                sug = next(iter(sp.candidates(wl)), None)
                finds.append({"page":i,"rule":"spelling","severity":"minor","detail":f"Unknown word '{wl}', suggestion '{sug}'"})
    return finds

# -------------------------------------------------------------------
# UI GATE
# -------------------------------------------------------------------
def gate():
    st.session_state.setdefault("auth", False)
    if not st.session_state["auth"]:
        st.title("AI Design Quality Auditor")
        pw = st.text_input("Enter access password", type="password", key="entry_pw")
        if st.button("Enter"):
            if pw == APP_PASSWORD:
                st.session_state["auth"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
                st.stop()
        else:
            st.stop()

# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
def main():
    gate()

    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Go to", ["Audit","Training","Analytics","Settings"])

    rules = load_rules()
    history = load_history()

    # ---------------- Audit ----------------
    if tab=="Audit":
        st.header("Run Audit")
        with st.form("audit_form"):
            meta = {}
            meta["supplier"] = st.selectbox("Supplier", DEFAULT_PICKLISTS["suppliers"])
            meta["client"] = st.selectbox("Client", DEFAULT_PICKLISTS["clients"])
            meta["project"] = st.selectbox("Project", DEFAULT_PICKLISTS["projects"])
            meta["drawing_type"] = st.selectbox("Drawing Type", DEFAULT_PICKLISTS["drawing_types"])
            meta["vendor"] = st.selectbox("Vendor", DEFAULT_PICKLISTS["vendors"])
            meta["site_type"] = st.selectbox("Site Type", DEFAULT_PICKLISTS["site_types"])
            meta["cabinet_location"] = st.selectbox("Cabinet Location", DEFAULT_PICKLISTS["cabinet_locations"])
            meta["radio_location"] = st.selectbox("Radio Location", DEFAULT_PICKLISTS["radio_locations"])
            meta["sectors"] = st.selectbox("Qty Sectors",[1,2,3,4,5,6])
            meta["mimo_config"] = st.selectbox("MIMO Config", DEFAULT_PICKLISTS["mimo_configs"])
            meta["site_address"] = st.text_input("Site Address")

            up = st.file_uploader("Upload PDF", type="pdf")
            submitted = st.form_submit_button("Run Audit")

        if submitted and up:
            pdf_bytes = up.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages=[p.get_text("text") for p in doc]

            allow=set(rules.get("allowlist",[]))
            findings=spelling_findings(pages,allow)

            status="Pass" if not findings else "Rejected"
            st.write(f"### Result: {status}")

            excel_bytes = make_excel(findings, meta, up.name, status)
            pdf_ann = annotate_pdf(pdf_bytes, findings)

            col1,col2 = st.columns(2)
            col1.download_button("⬇️ Download Excel", excel_bytes, file_name=f"{up.name}_{status}_{now_iso()}.xlsx")
            col2.download_button("⬇️ Download PDF", pdf_ann, file_name=f"{up.name}_{status}_{now_iso()}.pdf")

            rec = {
                "timestamp_utc": now_iso(),
                **meta,
                "status": status,
                "pdf_name": up.name
            }
            history = pd.concat([history, pd.DataFrame([rec])], ignore_index=True)
            save_history(history)

    # ---------------- Training ----------------
    elif tab=="Training":
        st.header("Training")
        st.info("Upload an audit Excel with Valid/Not-Valid or add single rule below.")

        up = st.file_uploader("Upload Training CSV/XLSX", type=["csv","xlsx"])
        if up:
            try:
                df = pd.read_excel(up) if up.name.endswith("xlsx") else pd.read_csv(up)
                st.dataframe(df.head())
                if "Valid" in df.columns:
                    st.success("Found 'Valid' column — training data loaded.")
            except Exception as e:
                st.error(f"Failed to read: {e}")

        with st.form("manual_rule"):
            txt = st.text_input("New rule (text match or condition)")
            cat = st.selectbox("Category",["spelling","content","format"])
            submit = st.form_submit_button("Add Rule")
            if submit and txt:
                rules.setdefault("rules",[]).append({"rule":txt,"category":cat})
                save_rules(rules)
                st.success("Rule added.")

    # ---------------- Analytics ----------------
    elif tab=="Analytics":
        st.header("Analytics")
        if history.empty:
            st.info("No history yet.")
        else:
            filt_client = st.selectbox("Filter by Client",["All"]+DEFAULT_PICKLISTS["clients"])
            filt_project = st.selectbox("Filter by Project",["All"]+DEFAULT_PICKLISTS["projects"])
            filt_supplier = st.selectbox("Filter by Supplier",["All"]+DEFAULT_PICKLISTS["suppliers"])

            show=history.copy()
            if filt_client!="All":
                show=show[show.client==filt_client]
            if filt_project!="All":
                show=show[show.project==filt_project]
            if filt_supplier!="All":
                show=show[show.supplier==filt_supplier]

            st.dataframe(show,use_container_width=True)

    # ---------------- Settings ----------------
    elif tab=="Settings":
        st.header("Settings")
        pw = st.text_input("Enter rules password", type="password")
        if pw==RULES_PASSWORD:
            st.success("Access granted")
            st.write("Picklist Management")
            for k,v in DEFAULT_PICKLISTS.items():
                new = st.text_area(k, "\n".join(v))
                if st.button(f"Save {k}"):
                    DEFAULT_PICKLISTS[k] = [x.strip() for x in new.splitlines() if x.strip()]
                    st.success(f"{k} updated")
        else:
            st.warning("Password required for rules management.")

# -------------------------------------------------------------------
if __name__=="__main__":
    main()
