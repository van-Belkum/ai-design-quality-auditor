import os, io, re, json, tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process

# ---------------- optional imports (safe) ----------------
def safe_import(name):
    try:
        return __import__(name)
    except Exception:
        return None

fitz = safe_import("fitz")            # PyMuPDF (PDF annotate + text/words)
pdfminer = safe_import("pdfminer")     # pdfminer.six fallback text
SpellChecker = None
try:
    from spellchecker import SpellChecker as _SC
    SpellChecker = _SC
except Exception:
    pass

pytesseract = None
pdf2image = None
try:
    import pytesseract as _pt; pytesseract = _pt
    from pdf2image import convert_from_bytes as _c; pdf2image = _c
except Exception:
    pass

# ---------------- history log ----------------
LOG_DIR = "history"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_CSV = os.path.join(LOG_DIR, "audit_log.csv")

# ---------------- data models ----------------
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
    boxes: List[Tuple[float,float,float,float]] = field(default_factory=list)
    context: Optional[str] = None

# ---------------- utils ----------------
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

def _ocr_image(image):
    if pytesseract is None:
        return {"text": "", "words": []}
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words=[]; toks=[]
        for i in range(len(data["text"])):
            word = data["text"][i]
            if not word or str(word).strip()=="": continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            words.append((float(x), float(y), float(x+w), float(y+h), str(word)))
            toks.append(str(word))
        return {"text": " ".join(toks), "words": words}
    except Exception:
        return {"text": "", "words": []}

def extract_pages_with_ocr(pdf_bytes):
    pages=[]; used_ocr=False
    if fitz is not None:
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for i in range(len(doc)):
                    text = doc[i].get_text("text") or ""
                    words = [(float(w[0]),float(w[1]),float(w[2]),float(w[3]),str(w[4])) for w in doc[i].get_text("words")]
                    if not words and pdf2image is not None and pytesseract is not None:
                        imgs = pdf2image(pdf_bytes, first_page=i+1, last_page=i+1, dpi=200)
                        if imgs:
                            data = _ocr_image(imgs[0])
                            text = data["text"] or text
                            words = data["words"] or words
                            used_ocr=True
                    pages.append({"text": text, "words": words})
        except Exception:
            pass
    if not pages:
        if pdf2image is not None and pytesseract is not None:
            imgs = pdf2image(pdf_bytes, dpi=200)
            for im in imgs:
                data = _ocr_image(im)
                pages.append({"text": data["text"], "words": data["words"]})
            used_ocr=True
        else:
            try:
                from pdfminer.high_level import extract_text
                t = extract_text(io.BytesIO(pdf_bytes)) or ""
            except Exception:
                t = ""
            pages=[{"text": t, "words": []}]
    return pages, used_ocr

def find_phrase_boxes(words, phrase):
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9\-']+", str(phrase)) if t]
    if not tokens: return []
    lower_words = [(w[4].lower(), w) for w in words]
    boxes=[]; n=len(tokens)
    for i in range(0, len(lower_words)-n+1):
        seq = [lower_words[i+k][0] for k in range(n)]
        if seq == tokens:
            rects=[lower_words[i+k][1] for k in range(n)]
            x0=min(r[0] for r in rects); y0=min(r[1] for r in rects)
            x1=max(r[2] for r in rects); y1=max(r[3] for r in rects)
            boxes.append((x0,y0,x1,y1))
    return boxes

# ---------------- checks ----------------
def spell_findings(pages, allowlist, client_allow):
    finds=[]
    if SpellChecker is None: return finds
    try: sc = SpellChecker(language="en")
    except Exception: return finds
    for w in allowlist | client_allow:
        if re.match(r"^[A-Za-z][A-Za-z0-9\-]{1,}$", str(w)):
            try: sc.word_frequency.add(str(w).lower())
            except: pass
    for pno, pg in enumerate(pages, start=1):
        tokens = tokenise(pg["text"])
        check = [t.lower() for t in tokens if t.isalpha() and len(t) >= 3]
        try: miss = sc.unknown(check)
        except: miss=set()
        for m in sorted(miss):
            if m in allowlist or m in client_allow: continue
            msg=f"Possible typo: '{m}'"
            boxes = find_phrase_boxes(pg["words"], m)
            finds.append(Finding("", pno, "Spelling", msg, boxes=boxes))
    return finds

def checklist_findings(pages, pack: Optional[ClientPack], rules_blob: Dict[str, Any]):
    if not pack: return []
    finds=[]
    for pno, pg in enumerate(pages, start=1):
        text=pg["text"]
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
                    boxes=[]
                    if re.fullmatch(r"[A-Za-z0-9\s\-']+", pat, flags=re.IGNORECASE):
                        boxes=find_phrase_boxes(pg["words"], pat)
                    finds.append(Finding("", pno, "Checklist", f"[{rule.id}] Forbidden phrase present: '{pat}'", boxes=boxes))
    return finds

def roster_findings(filename, pages, roster_df):
    finds=[]
    if roster_df is None or roster_df.empty: return finds
    base=os.path.splitext(os.path.basename(filename))[0]
    full_text="\n".join(p["text"] for p in pages)
    row=roster_df.iloc[0].fillna("")
    # very lightweight starters
    if row.get("site_name") and row["site_name"].lower() not in full_text.lower():
        finds.append(Finding(filename,0,"Data",f"site_name mismatch: expected '{row['site_name']}'"))
    if row.get("postcode"):
        if row["postcode"].replace(" ","").lower() not in full_text.replace(" ","").lower():
            finds.append(Finding(filename,0,"Data",f"Postcode differs: expected '{row['postcode']}'"))
    return finds

# ---------------- annotate ----------------
def annotate_pdf(pdf_bytes: bytes, findings_rows):
    if fitz is None: return pdf_bytes
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    by_page={}
    for r in findings_rows:
        p=int(r.get("page") or 1)
        p=max(1, min(p, len(doc)))
        by_page.setdefault(p, []).append(r)
    for pno, rows in by_page.items():
        page=doc[pno-1]
        y=36
        for r in rows:
            boxes=r.get("boxes") or []
            if boxes:
                for (x0,y0,x1,y1) in boxes[:5]:
                    try:
                        ann=page.add_highlight_annot(fitz.Rect(x0,y0,x1,y1))
                        ann.set_info(info={"content": r.get("message","")})
                    except Exception: pass
            else:
                try:
                    page.add_text_annot((36,y), r.get("message","")); y+=14
                except Exception: pass
    out=io.BytesIO(); doc.save(out); return out.getvalue()

# ---------------- history helpers ----------------
def _status_from_counts(counts: dict) -> str:
    total = sum(counts.values())
    return "PASS" if total == 0 else "REJECTED"

def record_history(user: str, stage: str, client: str, results: dict) -> pd.DataFrame:
    rows = []
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    for fn, res in results.items():
        counts = res["summary"]["counts"]
        rows.append({
            "timestamp_utc": ts,
            "user": user or "unknown",
            "stage": stage,
            "client": client,
            "file": fn,
            "pages": res["summary"].get("pages", 0),
            "used_ocr": bool(res["summary"].get("used_ocr", False)),
            "total_findings": sum(counts.values()),
            **{f"n_{k.lower()}": v for k, v in counts.items()},
            "outcome": _status_from_counts(counts),
        })
    df_new = pd.DataFrame(rows)
    try:
        if os.path.exists(LOG_CSV):
            df_old = pd.read_csv(LOG_CSV)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(LOG_CSV, index=False)
    except Exception as e:
        st.warning(f"Could not write audit log: {e}")
        df = df_new
    return df

# ---------------- UI ----------------
st.set_page_config(page_title="AI Design QA — Learn v5.1", layout="wide")
st.title("AI Design Quality Auditor — Learn v5.1")
st.caption("OCR, PDF markups, Excel, history logging, PASS/REJECTED banner.")

with st.sidebar:
    st.header("Configuration")
    client = st.selectbox("Client", ["EE","H3G","MBNL","VODAFONE","CUSTOM"], index=0)
    rules_file = st.file_uploader("Rules pack (YAML)", type=["yaml","yml"])

    st.divider(); st.subheader("Reference data")
    roster_csv = st.file_uploader("Site roster CSV", type=["csv"])

    st.divider(); st.subheader("Outputs")
    gen_markups = st.checkbox("Generate marked-up PDFs (highlights + notes)", value=True)

    st.divider(); st.subheader("QA Operator")
    qa_user  = st.text_input("Your name / initials", value="")
    qa_stage = st.selectbox("Stage", ["First Check","Second Check","Final Sign-off"], index=0)

    run_btn = st.button("Run Audit", type="primary")

# load rules
rules_blob = load_rules(rules_file)

def _load_csv(u):
    if u is None: return None
    try: return pd.read_csv(u)
    except Exception:
        u.seek(0); return pd.read_excel(u)

roster_df = None

uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if run_btn and uploads:
    roster_df = _load_csv(roster_csv)
    packs, allowlist = compile_client_packs(rules_blob)
    client_pack = packs.get(client.upper())

    results={}
    pdf_bytes={}

    for upl in uploads:
        data=upl.read(); pdf_bytes[upl.name]=data
        pages, used_ocr = extract_pages_with_ocr(data)

        findings=[]
        capitals={t for t in tokenise("\n".join(p["text"] for p in pages)) if t.isupper()}
        client_allow={w.lower() for w in capitals}

        findings += spell_findings(pages, set([w.lower() for w in rules_blob.get("allowlist", [])]), client_allow)
        findings += checklist_findings(pages, client_pack, rules_blob)
        findings += roster_findings(upl.name, pages, roster_df)

        rows = [{"page": f.page, "kind": f.kind, "message": f.message, "boxes": f.boxes} for f in findings]
        kinds = ["Spelling","Checklist","Consistency","Data","Structural","Electrical","Cooling"]
        counts = {k: sum(1 for r in rows if r["kind"]==k) for k in kinds}
        results[upl.name] = {"summary": {"pages": len(pages), "used_ocr": used_ocr, "counts": counts}, "findings": rows}

    # ---- Excel + PASS/REJECT + tables ----
    tmp=tempfile.mkdtemp(prefix="audit_")
    excel_path=os.path.join(tmp,"report.xlsx")
    sum_rows=[]; det_rows=[]
    for fn,res in results.items():
        counts=res["summary"]["counts"]
        status=_status_from_counts(counts)
        sum_rows.append({"file": fn, "status": status, **counts, "pages": res["summary"]["pages"], "used_ocr": res["summary"]["used_ocr"]})
        for row in res["findings"]:
            det_rows.append({"file": fn, **row})

    with pd.ExcelWriter(excel_path, engine="openpyxl") as xw:
        pd.DataFrame(sum_rows).to_excel(xw,"Summary",index=False)
        pd.DataFrame(det_rows).to_excel(xw,"Findings",index=False)
    st.success("Audit complete.")
    st.download_button("Download Excel report", data=open(excel_path,"rb").read(), file_name="report.xlsx")

    overall_total = sum(sum_row.get("Spelling",0)+sum_row.get("Checklist",0)+sum_row.get("Consistency",0)
                        +sum_row.get("Data",0)+sum_row.get("Structural",0)+sum_row.get("Electrical",0)
                        +sum_row.get("Cooling",0) for sum_row in sum_rows)
    if overall_total==0:
        st.success("✅ **QA PASS** — please continue with **Second Check**.")
    else:
        st.error("❌ **REJECTED** — findings detected. Please fix and re-run.")

    st.write("#### Summary"); st.dataframe(pd.DataFrame(sum_rows))
    st.write("#### Findings"); st.dataframe(pd.DataFrame(det_rows))

    # ---- History log ----
    log_df = record_history(qa_user, qa_stage, client, results)
    st.write("### Audit history (latest 200)")
    st.dataframe(log_df.tail(200))
    try:
        st.download_button("Download full audit history CSV", data=open(LOG_CSV,"rb").read(), file_name="audit_history.csv")
    except Exception:
        pass

else:
    st.info("Upload rules/roster (optional), then add PDFs and click Run Audit.")
