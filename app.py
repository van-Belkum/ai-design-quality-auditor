import os, io, re, sys, json, tempfile, subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# ---------- Safe imports ----------
def safe_import(name, pip=None):
    try:
        return __import__(name)
    except Exception:
        if pip:
            st.warning(f"Optional dependency '{name}' not installed (pip install {pip})")
        else:
            st.warning(f"Optional dependency '{name}' not installed")
        return None

fitz = safe_import("fitz", "PyMuPDF")
cv2 = safe_import("cv2", "opencv-python-headless")
SpellChecker = None
try:
    from spellchecker import SpellChecker as _SC
    SpellChecker = _SC
except Exception:
    st.warning("Optional dependency 'pyspellchecker' not installed")
pdfminer = safe_import("pdfminer", "pdfminer.six")

# ---------- Data structures ----------
@dataclass
class Rule:
    id: str
    when_page_contains: Optional[str] = None
    must_include_regex: List[str] = field(default_factory=list)
    forbids: List[Dict[str, str]] = field(default_factory=list)
    hint: Optional[str] = None

@dataclass
class ClientPack:
    name: str
    global_includes: List[str] = field(default_factory=list)
    forbids: List[Dict[str, str]] = field(default_factory=list)
    page_rules: List[Rule] = field(default_factory=list)

@dataclass
class Finding:
    file: str
    page: int
    kind: str
    message: str
    context: Optional[str] = None

# ---------- Utils ----------
def load_rules(file: Optional[io.BytesIO]) -> Dict[str, Any]:
    default_path = os.path.join(os.path.dirname(__file__), "rules_example.yaml")
    import yaml
    if file is None:
        with open(default_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        return yaml.safe_load(file.getvalue().decode("utf-8", errors="ignore"))

def tokenise(text: str) -> List[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z\-']{1,}|[A-Za-z]{2,}\d+|\d+[A-Za-z]+", text)
    return [t.strip("'").strip("-") for t in raw]

def extract_text_pymupdf(doc) -> List[str]:
    return [doc[i].get_text("text") for i in range(len(doc))]

def extract_text_pdfminer(buf: bytes) -> List[str]:
    try:
        from pdfminer.high_level import extract_text
        t = extract_text(io.BytesIO(buf)) or ""
        return [t]
    except Exception:
        return [""]

def detect_misalignment(doc):
    if cv2 is None:
        return []
    res = []
    for i in range(len(doc)):
        try:
            pix = doc[i].get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
            img = np.frombuffer(pix.tobytes(), dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            if coords.size == 0: continue
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45: angle = -(90 + angle)
            else: angle = -angle
            if abs(angle) > 1.5:
                res.append((i+1, angle))
        except Exception:
            pass
    return res

def compile_client_packs(blob: Dict[str, Any]):
    packs = {}
    allowlist = set([w.lower() for w in blob.get("allowlist", [])])
    for c in blob.get("clients", []):
        page_rules = []
        for pr in c.get("page_rules", []):
            page_rules.append(Rule(
                id=pr.get("id", ""),
                when_page_contains=pr.get("when_page_contains"),
                must_include_regex=pr.get("must_include_regex", []),
                forbids=pr.get("forbids", []),
                hint=pr.get("hint")
            ))
        packs[c["name"].upper()] = ClientPack(
            name=c["name"].upper(),
            global_includes=c.get("global_includes", []),
            forbids=c.get("forbids", []),
            page_rules=page_rules
        )
    return packs, allowlist

# ---------- Checks ----------
def spell_findings(text_pages, allowlist, client_allow):
    finds = []
    if SpellChecker is None: return finds
    try: sc = SpellChecker(language="en")
    except: return finds
    for w in allowlist | client_allow:
        sc.word_frequency.add(w.lower())
    for pno, text in enumerate(text_pages, start=1):
        tokens = tokenise(text)
        check = [t.lower() for t in tokens if t.isalpha() and len(t) >= 3]
        try: miss = sc.unknown(check)
        except: miss = set()
        for m in sorted(miss):
            if m in allowlist or m in client_allow: continue
            suggestion = None
            try:
                corr = sc.correction(m)
                if corr and corr != m: suggestion = corr
            except: pass
            msg = f"Possible typo: '{m}'" + (f" → '{suggestion}'" if suggestion else "")
            finds.append(Finding("", pno, "Spelling", msg))
    return finds

def checklist_findings(text_pages, pack: ClientPack, rules_blob: Dict[str, Any]):
    finds = []
    for pno, text in enumerate(text_pages, start=1):
        for s in pack.global_includes:
            if s and s not in text:
                finds.append(Finding("", pno, "Checklist", f"Missing required text: '{s}'"))
        for rule in pack.page_rules:
            if rule.when_page_contains and rule.when_page_contains not in text:
                continue
            for rx in rule.must_include_regex:
                if not re.search(rx, text, flags=re.IGNORECASE):
                    finds.append(Finding("", pno, "Checklist", f"[{rule.id}] Missing pattern: {rx}"))
            for forb in rule.forbids:
                pat = forb.get("pattern","")
                if pat and re.search(pat, text, flags=re.IGNORECASE):
                    finds.append(Finding("", pno, "Checklist", f"[{rule.id}] Forbidden phrase present: '{pat}'"))
        for forb in pack.forbids:
            pat = forb.get("pattern","")
            if pat and re.search(pat, text, flags=re.IGNORECASE):
                finds.append(Finding("", pno, "Checklist", f"Forbidden phrase present: '{pat}'"))
        # Custom cross-rules
        for pair in rules_blob.get("mutual_exclusive", []):
            a, b = pair.get("a"), pair.get("b")
            if a and b and a in text and b in text:
                finds.append(Finding("", pno, "Checklist", f"Mutual exclusion violated: '{a}' AND '{b}' both present"))
    return finds

def consistency_findings(text_pages):
    from rapidfuzz.distance import Levenshtein
    toks = [t for t in tokenise("\n".join(text_pages)) if len(t)>=6 and any(ch.isdigit() for ch in t)]
    toks = list(set(toks))
    finds=[]
    for i,a in enumerate(toks):
        for b in toks[i+1:]:
            d=Levenshtein.distance(a,b)
            if d<=2: finds.append(Finding("",0,"Consistency",f"Similar model codes: {a} vs {b}"))
    return finds

# ---------- Audit ----------
def audit_pdf_bytes(buf, client, rules_blob):
    packs, allowlist = compile_client_packs(rules_blob)
    client_pack = packs.get(client.upper(), ClientPack(name=client.upper()))
    if fitz: 
        try:
            with fitz.open(stream=buf, filetype="pdf") as doc:
                text_pages = extract_text_pymupdf(doc)
                mis = detect_misalignment(doc)
        except:
            text_pages = extract_text_pdfminer(buf); mis=[]
    else:
        text_pages = extract_text_pdfminer(buf); mis=[]
    capitals={t for t in tokenise("\n".join(text_pages)) if t.isupper()}
    client_allow={w.lower() for w in capitals}
    findings=[]
    findings+=spell_findings(text_pages, allowlist, client_allow)
    findings+=checklist_findings(text_pages, client_pack, rules_blob)
    findings+=consistency_findings(text_pages)
    for pno,angle in mis:
        findings.append(Finding("",pno,"Layout",f"Page skew {angle:.2f}°"))
    summary={"pages":max(1,len(text_pages)),"counts":{k:sum(1 for f in findings if f.kind==k) for k in ["Spelling","Checklist","Consistency","Layout"]}}
    rows=[{"page":f.page,"kind":f.kind,"message":f.message} for f in findings]
    return {"summary":summary,"findings":rows}

def write_excel_report(results,out_path):
    summaries,details=[],[]
    for fn,res in results.items():
        summaries.append({"file":fn,**res["summary"]["counts"],"pages":res["summary"]["pages"]})
        for row in res["findings"]: details.append({"file":fn,**row})
    with pd.ExcelWriter(out_path,engine="openpyxl") as xw:
        pd.DataFrame(summaries).to_excel(xw,"Summary",index=False)
        pd.DataFrame(details).to_excel(xw,"Findings",index=False)

# ---------- Feedback learner ----------
def learn_from_feedback(fb_df: pd.DataFrame, rules_blob: Dict[str,Any]) -> Dict[str,Any]:
    label_col=None
    for c in fb_df.columns:
        if fb_df[c].astype(str).str.contains("Valid",case=False,na=False).any():
            label_col=c; break
    if not label_col: return rules_blob
    # Spelling allowlist
    mask=(fb_df["kind"]=="Spelling") & (fb_df[label_col].str.lower()=="not valid")
    for msg in fb_df.loc[mask,"message"].dropna():
        m=re.search(r"Possible typo: '([^']+)'",msg)
        if m: rules_blob.setdefault("allowlist",[]).append(m.group(1))
    # TODO extend for checklist/layout
    rules_blob["allowlist"]=sorted(set(rules_blob["allowlist"]))
    return rules_blob

# ---------- UI ----------
st.set_page_config(page_title="AI Design Quality Auditor", layout="wide")
st.title("AI Design Quality Auditor")

with st.sidebar:
    client=st.selectbox("Client",["EE","H3G","MBNL","VODAFONE","CUSTOM"])
    rules_file=st.file_uploader("Rules pack (YAML)",type=["yaml","yml"])
    run_btn=st.button("Run Audit",type="primary")
    st.divider(); st.subheader("Learn from feedback")
    fb_csv=st.file_uploader("Upload labeled CSV",type=["csv"],key="fb")
    learn_btn=st.button("Apply feedback")
    st.divider(); st.subheader("Add mutual exclusion")
    new_a=st.text_input("Word A"); new_b=st.text_input("Word B")
    add_pair=st.button("Add exclusion rule")

uploads=st.file_uploader("Upload PDFs/DWGs",type=["pdf","dwg"],accept_multiple_files=True)

# Load rules
rules_blob=load_rules(rules_file)

# Feedback loop
if learn_btn and fb_csv:
    fb_df=pd.read_csv(fb_csv)
    rules_blob=learn_from_feedback(fb_df,rules_blob)
    import yaml
    buf=io.StringIO(); yaml.safe_dump(rules_blob,buf,sort_keys=False,allow_unicode=True)
    st.download_button("Download updated rules.yaml",data=buf.getvalue().encode(),file_name="rules_learned.yaml")
    st.success("Rules updated from feedback")

# Add cross-rule
if add_pair and new_a and new_b:
    rules_blob.setdefault("mutual_exclusive",[]).append({"a":new_a,"b":new_b})
    st.success(f"Added exclusion: {new_a} vs {new_b}")

if run_btn and uploads:
    results={}
    for upl in uploads:
        buf=upl.read()
        results[upl.name]=audit_pdf_bytes(buf,client,rules_blob)
    out_dir=tempfile.mkdtemp(); excel=os.path.join(out_dir,"report.xlsx")
    write_excel_report(results,excel)
    st.download_button("Download Excel report",data=open(excel,"rb").read(),file_name="report.xlsx")
    st.dataframe(pd.DataFrame([{"file":fn,**res["summary"]["counts"]} for fn,res in results.items()]))
