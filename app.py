import re
import io
import os
import ast
import json
import base64
import yaml
import fitz  # PyMuPDF
import pandas as pd
from datetime import datetime, timezone
from spellchecker import SpellChecker
import streamlit as st

APP_VERSION = "rules-engine-v5-full"

# ---------------- LOGO (top-right, always visible) ----------------
logo_path = "212BAAC2-5CB6-46A5-A53D-06497B78CF23.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
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
        <img src="data:image/png;base64,{logo_base64}" class="top-right-logo">
        """,
        unsafe_allow_html=True,
    )

# ---------------- Persistence ----------------
def ensure_history():
    os.makedirs("history", exist_ok=True)
    sup_path = os.path.join("history", "suppressions.json")
    if not os.path.exists(sup_path):
        with open(sup_path, "w", encoding="utf-8") as f:
            json.dump({"messages": [], "patterns": []}, f, indent=2)
    log_path = os.path.join("history", "audit_log.csv")
    if not os.path.exists(log_path):
        pd.DataFrame(columns=["timestamp","file","status","findings","meta"]).to_csv(log_path, index=False)
    return sup_path, log_path

def read_suppressions():
    sup_path, _ = ensure_history()
    with open(sup_path, "r", encoding="utf-8") as f:
        return json.load(f)

def update_suppressions(new_msgs):
    sup_path, _ = ensure_history()
    blob = read_suppressions()
    msgs = set(blob.get("messages", []))
    msgs.update(new_msgs)
    blob["messages"] = sorted(msgs)
    with open(sup_path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)

def log_audit(file, status, findings, meta: dict):
    _, log_path = ensure_history()
    row = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "file": file,
        "status": status,
        "findings": findings,
        "meta": json.dumps(meta, ensure_ascii=False),
    }])
    try:
        old = pd.read_csv(log_path)
        new = pd.concat([row, old], axis=0, ignore_index=True).head(300)
        new.to_csv(log_path, index=False)
    except Exception:
        row.to_csv(log_path, index=False)

# ---------------- Rules Engine ----------------
def load_rules(upload) -> dict:
    try:
        if upload is None:
            with open("rules_example.yaml","r",encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        else:
            return yaml.safe_load(upload) or {}
    except Exception as e:
        st.error(f"Rules YAML load error: {e}")
        return {}

def when_matches(when: dict, meta: dict) -> bool:
    if not when: return True
    for k, v in when.items():
        if meta.get(k) != v: return False
    return True

def run_rules(rules: dict, pages: list, doc, meta: dict) -> list:
    findings = []
    all_text = " ".join(pages)

    for chk in rules.get("checks", []):
        if not when_matches(chk.get("when", {}), meta):
            continue
        cid = chk.get("id", "RULE")
        sev = chk.get("severity","low")
        typ = chk.get("type")

        if typ == "include_text":
            txt = chk.get("text")
            if txt and txt.lower() not in all_text.lower():
                findings.append({"page":0,"kind":"Checklist","rule_id":cid,
                                 "message":f"Missing required text: '{txt}'",
                                 "severity":sev,"term":txt})

        elif typ == "forbid_together":
            a, b = chk.get("a",""), chk.get("b","")
            if a.lower() in all_text.lower() and b.lower() in all_text.lower():
                findings.append({"page":0,"kind":"Consistency","rule_id":cid,
                                 "message":f"Forbidden together: {a}+{b}",
                                 "severity":sev,"term":a})

        elif typ == "regex_forbid":
            rgx = chk.get("regex")
            hint = chk.get("hint","")
            if rgx and re.search(rgx, all_text, re.I|re.M):
                msg = f"Forbidden pattern: /{rgx}/ {hint}"
                findings.append({"page":0,"kind":"Consistency","rule_id":cid,
                                 "message":msg,"severity":sev,"term":rgx})

    return findings

# ---------------- Checks ----------------
def spelling_checks(pages, allowlist:set):
    sp = SpellChecker(distance=1)
    out=[]
    for i, page in enumerate(pages, start=1):
        for w in re.findall(r"[A-Za-z][A-Za-z'\-]{1,}", page):
            wl=w.lower()
            if wl in allowlist or wl in sp: continue
            sug = next(iter(sp.candidates(wl)), None)
            if sug and sug != wl:
                out.append({"page":i,"kind":"Spelling","rule_id":"SPELL",
                            "message":f"Possible typo: '{w}'→'{sug}'",
                            "severity":"low","term":w})
    return out

# ---------------- Annotation ----------------
def annotate_pdf(pdf_bytes:bytes, findings:list)->bytes:
    if not findings: return pdf_bytes
    buf_in=io.BytesIO(pdf_bytes)
    doc=fitz.open(stream=buf_in,filetype="pdf")

    for f in findings:
        pno=f.get("page",0)
        if not pno or pno>len(doc): continue
        page=doc[pno-1]
        term=f.get("term")
        msg=f"[{f.get('kind')}] {f.get('message')}"
        if term:
            quads=page.search_for(term,quads=True)
            if quads:
                rect=fitz.Rect(quads[0].rect)
                page.add_text_annot(rect.tl,msg)
    out=io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()

# ---------------- UI ----------------
st.title("AI Design Quality Auditor")
st.caption(f"Build {APP_VERSION}")

# ---- Metadata ----
client=st.sidebar.selectbox("Client",["","BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"])
project=st.sidebar.selectbox("Project",["","RAN","Power Resilience","East Unwind","Beacon 4"])
site=st.sidebar.selectbox("Site Type",["","Greenfield","Rooftop","Streetworks"])
vendor=st.sidebar.selectbox("Vendor",["","Ericsson","Nokia"])
cab=st.sidebar.selectbox("Cabinet",["","Indoor","Outdoor"])
radio=st.sidebar.selectbox("Radio",["","High Level","Low Level","Indoor","Door"])
sectors=st.sidebar.selectbox("Sectors",["","1","2","3","4","5","6"])
mimo="" if project=="Power Resilience" else st.sidebar.text_input("MIMO Config")
site_addr=st.sidebar.text_area("Site Address",height=60)

if st.sidebar.button("Clear Metadata"):
    st.session_state.clear()
    st.rerun()

rules_file=st.sidebar.file_uploader("Rules YAML",type=["yaml","yml"])
feedback_csv=st.sidebar.file_uploader("Feedback CSV",type=["csv"])
if feedback_csv is not None:
    try:
        df_fb=pd.read_csv(feedback_csv)
        bad=df_fb.loc[df_fb["decision"].str.lower().str.contains("not valid"),"message"].tolist()
        update_suppressions(bad)
        st.sidebar.success(f"Stored {len(bad)} suppressions")
    except Exception as e:
        st.sidebar.error(str(e))

rules=load_rules(rules_file)

# ---- File ----
pdf=st.file_uploader("Upload PDF",type=["pdf"])

meta_complete=all([client,project,site,vendor,cab,radio,sectors,site_addr or mimo!=""])

if st.button("Run Audit",disabled=not(pdf and meta_complete)):
    if not meta_complete:
        st.error("Fill ALL metadata fields")
    elif not pdf:
        st.error("Upload a PDF")
    else:
        data=pdf.read()
        doc=fitz.open(stream=data,filetype="pdf")
        pages=[p.get_text("text") for p in doc]

        allow=set(rules.get("allowlist",[]))
        findings=[]
        findings+=spelling_checks(pages,allow)
        findings+=run_rules(rules,pages,doc,{
            "client":client,"project":project,"site":site,"vendor":vendor,
            "cab":cab,"radio":radio,"sectors":sectors,"mimo":mimo,"site_addr":site_addr
        })

        sup=read_suppressions()
        df=pd.DataFrame(findings)
        if not df.empty:
            df=df[~df["message"].isin(sup.get("messages",[]))]

        status="PASS" if df.empty else "REJECTED"
        st.subheader("Summary")
        st.write({"File":pdf.name,"Status":status,"Pages":len(pages),"Findings":len(df)})

        st.subheader("Findings")
        st.dataframe(df)

        today=datetime.now().strftime("%Y%m%d")
        base=os.path.splitext(pdf.name)[0]

        # Excel
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine="openpyxl") as xw:
            df.to_excel(xw,sheet_name="Findings",index=False)
            pd.DataFrame([{
                "client":client,"project":project,"site":site,"vendor":vendor,
                "cab":cab,"radio":radio,"sectors":sectors,"mimo":mimo,"site_addr":site_addr
            }]).to_excel(xw,sheet_name="Metadata",index=False)
        st.download_button("Download Excel",buf.getvalue(),file_name=f"{base}_{status}_{today}.xlsx")

        # Annotated PDF
        ann=annotate_pdf(data,df.to_dict("records"))
        st.download_button("Download Annotated PDF",ann,file_name=f"{base}_{status}_{today}.pdf")

        log_audit(pdf.name,status,len(df),{
            "client":client,"project":project,"site":site,"vendor":vendor,
            "cab":cab,"radio":radio,"sectors":sectors,"mimo":mimo,"site_addr":site_addr
        })

# ---- History ----
st.divider()
st.subheader("History")
try:
    hist=pd.read_csv(os.path.join("history","audit_log.csv"))
    st.dataframe(hist)
except: st.info("No history yet")
