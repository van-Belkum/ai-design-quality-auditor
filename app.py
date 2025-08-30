import os, io, re, json, tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process

# Optional deps
def safe_import(name):
    try:
        return __import__(name)
    except Exception:
        return None
fitz = safe_import("fitz")  # PyMuPDF
cv2 = safe_import("cv2")
pdfminer = safe_import("pdfminer")
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

# ---------------- Models ----------------
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

# ---------------- Utils ----------------
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

def explode_aliases(s):
    return [a.strip() for a in str(s or "").split(";") if a.strip()]

def best_match(token, choices, score_cutoff=85):
    if not token or not choices: return None, 0
    res = process.extractOne(token, choices, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
    if not res: return None, 0
    match, score, _ = res
    return match, score

UK_POSTCODE_RX = re.compile(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s*\d[A-Z]{2}\b", re.I)
SHEET_RXES = [re.compile(r"Sheet\s+(\d+)\s+of\s+(\d+)", re.I), re.compile(r"Page\s+(\d+)\s*/\s*(\d+)", re.I)]

# ---------------- OCR ----------------
def ocr_page(image):
    # Return (text, words[(x0,y0,x1,y1,word),...]) via Tesseract.
    if pytesseract is None: return "", []
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = []
        for i in range(len(data["text"])):
            word = data["text"][i]
            if not word or str(word).strip()=="":
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            words.append((float(x), float(y), float(x+w), float(y+h), str(word)))
        text = " ".join([w[4] for w in words])
        return text, words
    except Exception:
        return "", []

def extract_pages_with_ocr(pdf_bytes):
    # Try PyMuPDF text; if no words, fall back to OCR.
    pages=[]
    used_ocr=False
    if fitz is not None:
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for i in range(len(doc)):
                    text = doc[i].get_text("text") or ""
                    words = [(float(w[0]),float(w[1]),float(w[2]),float(w[3]),str(w[4])) for w in doc[i].get_text("words")]
                    if not words and pdf2image is not None and pytesseract is not None:
                        # OCR fallback
                        imgs = pdf2image(pdf_bytes, first_page=i+1, last_page=i+1, dpi=200)
                        if imgs:
                            t, wds = ocr_page(imgs[0])
                            text = t or text
                            words = wds or words
                            used_ocr=True
                    pages.append({"text": text, "words": words})
        except Exception:
            pass
    if not pages:
        # pure OCR of all pages
        if pdf2image is not None and pytesseract is not None:
            imgs = pdf2image(pdf_bytes, dpi=200)
            for im in imgs:
                t, wds = ocr_page(im)
                pages.append({"text": t, "words": wds})
            used_ocr=True
        else:
            # last resort: pdfminer text only
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
    boxes=[]
    n=len(tokens)
    for i in range(0, len(lower_words)-n+1):
        seq = [lower_words[i+k][0] for k in range(n)]
        if seq == tokens:
            rects=[lower_words[i+k][1] for k in range(n)]
            x0=min(r[0] for r in rects); y0=min(r[1] for r in rects)
            x1=max(r[2] for r in rects); y1=max(r[3] for r in rects)
            boxes.append((x0,y0,x1,y1))
    return boxes

# ---------------- Checks ----------------
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
            suggestion=None
            try:
                corr=sc.correction(m)
                if corr and corr!=m: suggestion=corr
            except: pass
            msg=f"Possible typo: '{m}'"+(f" → '{suggestion}'" if suggestion else "")
            boxes = find_phrase_boxes(pg["words"], m)
            finds.append(Finding("", pno, "Spelling", msg, boxes=boxes))
    return finds

def checklist_findings(pages, pack: ClientPack, rules_blob: Dict[str, Any]):
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
        for forb in pack.forbids:
            pat=forb.get("pattern","")
            if pat and re.search(pat, text, flags=re.IGNORECASE):
                boxes=[]
                if re.fullmatch(r"[A-Za-z0-9\s\-']+", pat, flags=re.IGNORECASE):
                    boxes=find_phrase_boxes(pg["words"], pat)
                finds.append(Finding("", pno, "Checklist", f"Forbidden phrase present: '{pat}'", boxes=boxes))
        for pair in rules_blob.get("mutual_exclusive", []):
            a,b=pair.get("a"),pair.get("b")
            if a and b and a in text and b in text:
                boxes_a=find_phrase_boxes(pg["words"], a)
                boxes_b=find_phrase_boxes(pg["words"], b)
                finds.append(Finding("", pno, "Checklist", f"Mutual exclusion violated: '{a}' AND '{b}' both present", boxes=boxes_a+boxes_b))
    return finds

def consistency_findings(pages, rules_blob=None):
    from rapidfuzz.distance import Levenshtein
    toks=[t for t in tokenise("\n".join(p["text"] for p in pages)) if len(t)>=6 and any(ch.isdigit() for ch in t)]
    toks=list(set(toks))
    ignore=set()
    if rules_blob:
        for p in rules_blob.get("consistency_ignore_pairs", []):
            ignore.add(tuple(sorted([p["a"], p["b"]])))
    finds=[]
    for i,a in enumerate(toks):
        for b in toks[i+1:]:
            key=tuple(sorted([a,b]))
            if key in ignore: continue
            d=Levenshtein.distance(a,b)
            if d<=2:
                finds.append(Finding("",0,"Consistency",f"Similar model codes: '{a}' vs '{b}'"))
    return finds

def roster_findings(filename, pages, roster_df):
    finds=[]
    if roster_df is None or roster_df.empty: return finds
    base=os.path.splitext(os.path.basename(filename))[0]
    tokens=re.split(r"[_\-\s]+", base)
    file_key="_".join(tokens[:2]) if len(tokens)>=2 else tokens[0]
    row=roster_df[roster_df["file_key"].astype(str).str.contains(re.escape(file_key), case=False, na=False)]
    if row.empty: return finds
    row=row.iloc[0].fillna("")
    full_text="\n".join(p["text"] for p in pages)
    for label, expected in [("site_name", row.get("site_name","")), ("address_line1", row.get("address_line1","")), ("city", row.get("city",""))]:
        if expected:
            score=fuzz.partial_ratio(expected.lower(), full_text.lower())
            if score<80:
                finds.append(Finding(filename,0,"Data",f"{label} mismatch: expected '{expected}' (match {score}%)"))
    expected_pc=str(row.get("postcode","")).strip()
    if expected_pc:
        if not re.search(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b", full_text, flags=re.I):
            finds.append(Finding(filename,0,"Data",f"Postcode missing; expected like '{expected_pc}'"))
        elif expected_pc.replace(" ","").lower() not in full_text.replace(" ","").lower():
            finds.append(Finding(filename,0,"Data",f"Postcode differs: expected '{expected_pc}'"))
    expected_w3w=str(row.get("what3words","")).strip()
    if expected_w3w:
        if not re.search(r"\b[a-z]+\.[a-z]+\.[a-z]+\b", full_text):
            finds.append(Finding(filename,0,"Data",f"W3W missing; expected like '{expected_w3w}'"))
        elif expected_w3w.lower() not in full_text.lower():
            finds.append(Finding(filename,0,"Data",f"W3W differs: expected '{expected_w3w}'"))
    total_claimed=0
    for rx in SHEET_RXES:
        for p in pages:
            m=rx.search(p["text"])
            if m: total_claimed=max(total_claimed, int(m.group(2)))
    roster_total=int(row.get("sheet_count") or 0)
    if roster_total and total_claimed and roster_total!=total_claimed:
        finds.append(Finding(filename,0,"Data",f"Sheet count mismatch: roster {roster_total} vs drawing {total_claimed}"))
    return finds

# ---- First-pass Structural/Electrical/Cooling checks ----
def structural_findings(pages, structural_df: Optional[pd.DataFrame]):
    finds=[]
    if structural_df is None or structural_df.empty: return finds
    text="\n".join(p["text"] for p in pages)
    antennas=len(re.findall(r"\bantenna\b", text, flags=re.I))
    rr_units=len(re.findall(r"\bRRU\b|\bRRH\b", text, flags=re.I))
    wind_area=antennas*0.25 + rr_units*0.05  # m2 (rough default)
    candidates=structural_df["element"].astype(str).tolist()
    mention=None
    for c in candidates:
        if c and c in text:
            mention=c; break
    if mention:
        row=structural_df[structural_df["element"]==mention].iloc[0]
        cap=row.get("max_wind_area_m2") or 0
        if wind_area>cap:
            finds.append(Finding("",0,"Structural",f"Wind area {wind_area:.2f} m² exceeds {mention} capacity {cap:.2f} m²"))
    return finds

def electrical_findings(pages, electrical_df: Optional[pd.DataFrame]):
    finds=[]
    if electrical_df is None or electrical_df.empty: return finds
    text="\n".join(p["text"] for p in pages)
    total_watts=0
    for _,r in electrical_df.iterrows():
        code=str(r["equip_code"])
        count=len(re.findall(rf"\b{re.escape(code)}\b", text))
        total_watts+=count*float(r.get("dc_watts") or 0)
    dcdus = electrical_df[electrical_df["equip_code"].str.upper()=="DCDU"]
    if not dcdus.empty:
        cap_A= float(dcdus.iloc[0]["breaker_A"])
        approx_V= 48.0
        cap_W= cap_A * approx_V
        if total_watts> cap_W*0.8:
            finds.append(Finding("",0,"Electrical",f"Estimated DC load {total_watts:.0f} W near/exceeds DCDU capacity (~{cap_W:.0f} W)"))
    return finds

def cooling_findings(pages, cooling_df: Optional[pd.DataFrame]):
    finds=[]
    if cooling_df is None or cooling_df.empty: return finds
    text="\n".join(p["text"] for p in pages)
    total_btu=0
    for _,r in cooling_df.dropna(subset=["equip_code"]).iterrows():
        code=str(r["equip_code"])
        count=len(re.findall(rf"\b{re.escape(code)}\b", text))
        total_btu += count*float(r.get("heat_btu_per_hr") or 0)
    cab_rows = cooling_df.dropna(subset=["cabinet_model"])
    cap_btu=0
    for _,r in cab_rows.iterrows():
        if str(r["cabinet_model"]) in text:
            cap_btu = float(r.get("cooling_capacity_btu_hr") or 0)
            break
    if cap_btu and total_btu>cap_btu:
        finds.append(Finding("",0,"Cooling",f"Heat load {total_btu:.0f} BTU/hr exceeds cabinet capacity {cap_btu:.0f} BTU/hr"))
    return finds

# ---------------- Annotate ----------------
def annotate_pdf(pdf_bytes: bytes, findings_rows: List[Dict[str,Any]]) -> bytes:
    if fitz is None: return pdf_bytes
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    by_page={}
    for r in findings_rows:
        p=int(r.get("page") or 0)
        if p<1 or p>len(doc): p=1
        by_page.setdefault(p, []).append(r)
    for pno, rows in by_page.items():
        page=doc[pno-1]
        y_offset=36
        for r in rows:
            boxes=r.get("boxes") or []
            if boxes:
                for (x0,y0,x1,y1) in boxes[:5]:
                    try:
                        annot=page.add_highlight_annot(fitz.Rect(x0,y0,x1,y1))
                        annot.set_info(info={"content": r.get("message","")})
                    except Exception:
                        pass
            else:
                try:
                    page.add_text_annot((36,y_offset), r.get("message",""))
                    y_offset += 14
                except Exception: pass
    out=io.BytesIO(); doc.save(out); return out.getvalue()

# ---------------- Audit ----------------
def audit_pdf_bytes(buf, client, rules_blob, filename,
                    roster_df=None, equip_df=None, cab_df=None,
                    structural_df=None, electrical_df=None, cooling_df=None):
    packs, allowlist = compile_client_packs(rules_blob)
    client_pack = packs.get(client.upper(), ClientPack(name=client.upper()))
    pages, used_ocr = extract_pages_with_ocr(buf)
    capitals={t for t in tokenise("\n".join(p["text"] for p in pages)) if t.isupper()}
    client_allow={w.lower() for w in capitals}
    findings=[]
    findings += spell_findings(pages, set([w.lower() for w in rules_blob.get("allowlist", [])]), client_allow)
    findings += checklist_findings(pages, client_pack, rules_blob)
    findings += consistency_findings(pages, rules_blob)
    findings += roster_findings(filename, pages, roster_df)
    findings += structural_findings(pages, structural_df)
    findings += electrical_findings(pages, electrical_df)
    findings += cooling_findings(pages, cooling_df)
    counts={k: sum(1 for f in findings if f.kind==k) for k in ["Spelling","Checklist","Consistency","Data","Structural","Electrical","Cooling"]}
    summary={"pages": max(1,len(pages)), "used_ocr": used_ocr, "counts": counts}
    rows=[{"page": f.page, "kind": f.kind, "message": f.message, "boxes": f.boxes} for f in findings]
    return {"summary": summary, "findings": rows}

# ---------------- Learning & QA ----------------
def learn_from_feedback(fb_df: pd.DataFrame, rules_blob: Dict[str,Any], immediate_disable=True):
    label_col=None
    for c in fb_df.columns:
        if fb_df[c].astype(str).str.contains("Valid",case=False,na=False).any():
            label_col=c; break
    if not label_col: return rules_blob
    mask=(fb_df["kind"].astype(str)=="Spelling") & (fb_df[label_col].astype(str).str.lower()=="not valid")
    to_allow=set()
    for msg in fb_df.loc[mask,"message"].dropna().astype(str):
        m=re.search(r"Possible typo: '([^']+)'",msg)
        if m and len(m.group(1))>=3: to_allow.add(m.group(1).lower())
    rules_blob.setdefault("allowlist", [])
    rules_blob["allowlist"]=sorted(set([w.lower() for w in rules_blob["allowlist"]])|to_allow)
    nv=fb_df[(fb_df["kind"]=="Checklist") & (fb_df[label_col].str.lower()=="not valid")].copy()
    nv["rule_id"]=nv["message"].str.extract(r"\[([A-Za-z0-9\-_]+)\]")
    counts=nv["rule_id"].value_counts()
    for rid,cnt in counts.items():
        if not rid: continue
        if immediate_disable or cnt>=3:
            for client in rules_blob.get("clients", []):
                client["page_rules"]=[r for r in client.get("page_rules", []) if r.get("id")!=rid]
    return rules_blob

def apply_manual_qa(manual_df: pd.DataFrame, rules_blob: Dict[str,Any]):
    if manual_df is None or manual_df.empty: return rules_blob
    rules_blob.setdefault("clients", [])
    name_index={c["name"].upper(): i for i,c in enumerate(rules_blob["clients"]) if "name" in c}
    def ensure_client(name: str):
        if name not in name_index:
            rules_blob["clients"].append({"name": name, "global_includes": [], "forbids": [], "page_rules": []})
            name_index[name]=len(rules_blob["clients"])-1
        return rules_blob["clients"][name_index[name]]
    for _, r in manual_df.fillna("").iterrows():
        client=r.get("client","GLOBAL").strip().upper()
        rtype=r.get("rule_type","").strip().lower()
        pattern=str(r.get("pattern",""))
        rule_id=str(r.get("rule_id","")).strip() or f"MANUAL-{abs(hash(pattern))%100000}"
        wpc=str(r.get("when_page_contains","")).strip() or None
        hint=str(r.get("hint","")).strip()
        if rtype=="mutual_exclusion":
            a,b=r.get("a",""),r.get("b","")
            if a and b:
                rules_blob.setdefault("mutual_exclusive", []).append({"a":a,"b":b})
            continue
        if rtype=="allowlist":
            rules_blob.setdefault("allowlist", []).append(pattern.lower())
            rules_blob["allowlist"]=sorted(set(rules_blob["allowlist"]))
            continue
        if rtype=="disable_rule":
            targets=rules_blob["clients"] if client=="GLOBAL" else [ensure_client(client)]
            for t in targets:
                t["page_rules"]=[pr for pr in t.get("page_rules", []) if pr.get("id")!=rule_id]
            continue
        targets=rules_blob["clients"] if client=="GLOBAL" else [ensure_client(client)]
        if rtype=="forbid":
            for t in targets:
                t.setdefault("forbids", []).append({"pattern": pattern, "hint": hint})
        elif rtype in ("require_regex","forbid_regex"):
            for t in targets:
                pr_list=t.setdefault("page_rules", [])
                pr=None
                for x in pr_list:
                    if x.get("id")==rule_id and x.get("when_page_contains")==wpc: pr=x; break
                if pr is None:
                    pr={"id": rule_id, "when_page_contains": wpc, "must_include_regex": [], "forbids": []}
                    pr_list.append(pr)
                if rtype=="require_regex":
                    pr.setdefault("must_include_regex", []).append(pattern)
                else:
                    pr.setdefault("forbids", []).append({"pattern": pattern, "hint": hint})
    return rules_blob

# ---------------- UI ----------------
st.set_page_config(page_title="AI Design QA — Learn v5", layout="wide")
st.title("AI Design Quality Auditor — Learn v5")
st.caption("OCR fallback, deeper reference checks, PDF markups, Excel reporting, and quick rule promotion.")

with st.sidebar:
    st.header("Configuration")
    client = st.selectbox("Client", ["EE","H3G","MBNL","VODAFONE","CUSTOM"], index=0)
    rules_file = st.file_uploader("Rules pack (YAML)", type=["yaml","yml"])

    st.divider(); st.subheader("Reference data")
    roster_csv     = st.file_uploader("Site roster CSV", type=["csv"])
    equipment_csv  = st.file_uploader("Equipment catalog CSV", type=["csv"])
    cabinets_csv   = st.file_uploader("Cabinet catalog CSV", type=["csv"])
    structural_csv = st.file_uploader("Structural ref CSV", type=["csv"])
    electrical_csv = st.file_uploader("Electrical ref CSV", type=["csv"])
    cooling_csv    = st.file_uploader("Cooling ref CSV", type=["csv"])

    st.divider(); st.subheader("Manual QA → Rules")
    manual_csv = st.file_uploader("Manual QA CSV", type=["csv"])
    apply_manual = st.button("Apply Manual QA to rules")

    st.divider(); st.subheader("Learn from findings")
    fb_csv = st.file_uploader("Labeled findings CSV", type=["csv"])
    immediate_disable = st.checkbox("Disable checklist rules immediately when Not Valid", value=True)
    learn_btn = st.button("Apply feedback to rules")

    st.divider(); st.subheader("Outputs")
    gen_markups = st.checkbox("Generate marked-up PDFs (highlights + notes)", value=True)

    run_btn = st.button("Run Audit", type="primary")

# Load rules
rules_blob = load_rules(rules_file)

# Apply Manual QA
if apply_manual and manual_csv is not None:
    mdf=pd.read_csv(manual_csv)
    rules_blob=apply_manual_qa(mdf, rules_blob)
    import yaml
    buf=io.StringIO(); yaml.safe_dump(rules_blob, buf, sort_keys=False, allow_unicode=True)
    st.download_button("Download rules (after Manual QA)", data=buf.getvalue().encode("utf-8"), file_name="rules_manual.yaml")
    st.success("Manual QA applied to rules.")

# Learning
if learn_btn and fb_csv is not None:
    fb_df=pd.read_csv(fb_csv)
    rules_blob=learn_from_feedback(fb_df, rules_blob, immediate_disable=immediate_disable)
    import yaml
    buf=io.StringIO(); yaml.safe_dump(rules_blob, buf, sort_keys=False, allow_unicode=True)
    st.download_button("Download rules (after Learning)", data=buf.getvalue().encode("utf-8"), file_name="rules_learned.yaml")
    st.success("Learning applied.")

# Load reference CSVs
def _load_csv(u):
    if u is None: return None
    try: return pd.read_csv(u)
    except Exception:
        u.seek(0); return pd.read_excel(u)
roster_df     = _load_csv(roster_csv)
equip_df      = _load_csv(equipment_csv)
cabs_df       = _load_csv(cabinets_csv)
structural_df = _load_csv(structural_csv)
electrical_df = _load_csv(electrical_csv)
cooling_df    = _load_csv(cooling_csv)

# Upload PDFs
st.write("### Upload PDFs")
uploads = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True)

results={}
pdf_bytes={}

if run_btn and uploads:
    for upl in uploads:
        data=upl.read(); pdf_bytes[upl.name]=data
    # Run
    for fn, data in pdf_bytes.items():
        res = audit_pdf_bytes(data, client, rules_blob, filename=fn,
                              roster_df=roster_df, equip_df=equip_df, cab_df=cabs_df,
                              structural_df=structural_df, electrical_df=electrical_df, cooling_df=cooling_df)
        results[fn]=res

    # Excel
    tmp=tempfile.mkdtemp(prefix="audit_")
    excel_path=os.path.join(tmp,"report.xlsx")
    sum_rows=[]; det_rows=[]
    for fn,res in results.items():
        sum_rows.append({"file": fn, **res["summary"]["counts"], "pages": res["summary"]["pages"], "used_ocr": res["summary"]["used_ocr"]})
        for row in res["findings"]:
            det_rows.append({"file": fn, **row})
    with pd.ExcelWriter(excel_path, engine="openpyxl") as xw:
        pd.DataFrame(sum_rows).to_excel(xw,"Summary",index=False)
        pd.DataFrame(det_rows).to_excel(xw,"Findings",index=False)
    st.success("Audit complete.")
    st.download_button("Download Excel report", data=open(excel_path,"rb").read(), file_name="report.xlsx")

    # Marked-up PDFs
    if gen_markups and fitz is not None:
        zpath=os.path.join(tmp,"annotated_pdfs.zip")
        import zipfile
        with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
            for fn,res in results.items():
                annotated=annotate_pdf(pdf_bytes[fn], res["findings"])
                outp=os.path.join(tmp,f"{os.path.splitext(fn)[0]}_annotated.pdf")
                open(outp,"wb").write(annotated)
                z.write(outp, arcname=os.path.basename(outp))
        st.download_button("Download annotated PDFs (zip)", data=open(zpath,"rb").read(), file_name="annotated_pdfs.zip")

    # Tables
    st.write("#### Summary"); st.dataframe(pd.DataFrame(sum_rows))
    st.write("#### Findings"); df=pd.DataFrame(det_rows); st.dataframe(df)

    # Row actions: promote finding to rule
    st.write("### Promote findings to rules (row actions)")
    st.caption("Select rows and choose what rule to create.")
    choices = df["message"].dropna().unique().tolist()
    pick = st.multiselect("Select finding messages", choices)
    rule_type = st.selectbox("Rule type", ["allowlist (spelling)", "forbid (phrase)", "require_regex (enter regex below)", "mutual_exclusion"])
    extra = st.text_input("Extra (regex pattern or 'A,B' for exclusion)")
    if st.button("Create rules from selection"):
        import yaml
        blob=rules_blob.copy()
        blob.setdefault("clients", [])
        if rule_type=="allowlist (spelling)":
            blob.setdefault("allowlist", [])
            for m in pick:
                m2=re.search(r"Possible typo: '([^']+)'", m)
                if m2: blob["allowlist"].append(m2.group(1).lower())
            blob["allowlist"]=sorted(set(blob["allowlist"]))
        elif rule_type=="forbid (phrase)":
            phrase=extra.strip() if extra else None
            if phrase:
                for c in blob.get("clients", []):
                    c.setdefault("forbids", []).append({"pattern": phrase, "hint": ""})
        elif rule_type=="require_regex (enter regex below)":
            rx=extra.strip() if extra else None
            if rx:
                for c in blob.get("clients", []):
                    pr={"id": "WB-RULE", "when_page_contains": None, "must_include_regex": [rx], "forbids": []}
                    c.setdefault("page_rules", []).append(pr)
        elif rule_type=="mutual_exclusion":
            parts=[p.strip() for p in (extra or "").split(",")]
            if len(parts)==2:
                blob.setdefault("mutual_exclusive", []).append({"a": parts[0], "b": parts[1]})
        buf=io.StringIO(); yaml.safe_dump(blob, buf, sort_keys=False, allow_unicode=True)
        st.download_button("Download rules from selection", data=buf.getvalue().encode("utf-8"), file_name="rules_from_rows.yaml")

else:
    st.info("Upload rules and refs, then add PDFs and click Run Audit.")
