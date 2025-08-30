import os, io, re, json, tempfile, zipfile
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

# fuzzy (keep available)
from rapidfuzz import fuzz, process

# ---------- safe optional imports ----------
def safe_import(name):
    try:
        return __import__(name)
    except Exception:
        return None

fitz = safe_import("fitz")            # PyMuPDF
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

# ---------- history ----------
LOG_DIR = "history"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_CSV = os.path.join(LOG_DIR, "audit_log.csv")

# ---------- models ----------
from typing import NamedTuple
class Finding(NamedTuple):
    file: str
    page: int
    kind: str       # category
    message: str
    boxes: list     # [(x0,y0,x1,y1), ...]
    context: str | None = None

# ---------- utils ----------
def load_rules(file: Optional[io.BytesIO]) -> Dict[str, Any]:
    default_path = os.path.join(os.path.dirname(__file__), "rules_example.yaml")
    import yaml
    if file is None:
        with open(default_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        return yaml.safe_load(file.getvalue().decode("utf-8", errors="ignore"))

def tokenise(text: str) -> list[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z\-']{1,}|[A-Za-z]{2,}\d+|\d+[A-Za-z]+", text)
    return [t.strip("'").strip("-") for t in raw]

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

def extract_pages_with_ocr(pdf_bytes: bytes):
    pages=[]; used_ocr=False
    # Try PyMuPDF first (text + words); if page has no words, fall back to OCR
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
        # OCR whole doc
        if pdf2image is not None and pytesseract is not None:
            imgs = pdf2image(pdf_bytes, dpi=200)
            for im in imgs:
                data = _ocr_image(im)
                pages.append({"text": data["text"], "words": data["words"]})
            used_ocr=True
        else:
            # as last resort, single concatenated text (no boxes)
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

# ---------- categories ----------
CATEGORIES = ["Checklist","Data","Structural","Electrical","Cooling","Consistency","Spelling"]

def _status_from_counts(counts: dict) -> str:
    total = sum(counts.values())
    return "PASS" if total == 0 else "REJECTED"

def record_history(user: str, stage: str, meta: dict, results: dict) -> pd.DataFrame:
    rows = []
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    for fn, res in results.items():
        counts = res["summary"]["counts"]
        rows.append({
            "timestamp_utc": ts,
            "user": user or "unknown",
            "stage": stage,
            # metadata
            "client": meta.get("client",""),
            "project": meta.get("project",""),
            "site_type": meta.get("site_type",""),
            "vendor": meta.get("vendor",""),
            "cabinet_loc": meta.get("cabinet_loc",""),
            "radio_loc": meta.get("radio_loc",""),
            "config": meta.get("config",""),
            "mimo_config": meta.get("mimo_config",""),
            "site_address": meta.get("site_address",""),
            # file stats
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

# ---------- conditioned rules (v2) ----------
def _ci_eq(a: str, b: str) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()

def rule_applies(rule_when: dict, meta: dict) -> bool:
    if not rule_when:
        return True
    for key, want in rule_when.items():
        have = meta.get(key)
        if isinstance(want, list):
            if not any(_ci_eq(have, w) for w in want):
                return False
        else:
            if not _ci_eq(have, want):
                return False
    return True

def run_conditioned_rules(pages, rules_blob: dict, meta: dict):
    finds = []
    rules = rules_blob.get("rules", [])
    for pno, pg in enumerate(pages, start=1):
        text = pg["text"]

        for r in rules:
            when = r.get("when", {}) or {}
            if not rule_applies(when, meta):
                continue

            checks = r.get("checks", {}) or {}
            category = r.get("category") or "Checklist"
            if category not in CATEGORIES:
                category = "Checklist"

            # Optional page gate
            on_page_contains = checks.get("on_page_contains")
            if on_page_contains and (on_page_contains not in text):
                continue

            # require_text
            for s in checks.get("require_text", []) or []:
                if s not in text:
                    finds.append(Finding("", pno, category, f"[{r.get('id','RULE')}] Missing required text: '{s}'", []))

            # require_regex
            for rx in checks.get("require_regex", []) or []:
                if not re.search(rx, text, flags=re.IGNORECASE):
                    finds.append(Finding("", pno, category, f"[{r.get('id','RULE')}] Missing pattern: {rx}", []))

            # forbid
            for forb in checks.get("forbid", []) or []:
                pat = forb.get("pattern", "")
                if pat and re.search(pat, text, flags=re.IGNORECASE):
                    boxes = find_phrase_boxes(pg["words"], pat) if re.fullmatch(r"[A-Za-z0-9\s\-']+", pat, flags=re.IGNORECASE) else []
                    hint = forb.get("hint", "")
                    msg = f"[{r.get('id','RULE')}] Forbidden phrase present: '{pat}'"
                    if hint: msg += f" ({hint})"
                    finds.append(Finding("", pno, category, msg, boxes))
    return finds

# ---------- spelling & roster ----------
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
            finds.append(Finding("", pno, "Spelling", msg, boxes))
    return finds

def roster_findings(filename, pages, roster_df):
    finds=[]
    if roster_df is None or roster_df.empty: return finds
    full_text="\n".join(p["text"] for p in pages)
    row=roster_df.iloc[0].fillna("")
    if row.get("site_name") and row["site_name"].lower() not in full_text.lower():
        finds.append(Finding(filename,0,"Data",f"site_name mismatch: expected '{row['site_name']}'", []))
    if row.get("postcode"):
        if row["postcode"].replace(" ","").lower() not in full_text.replace(" ","").lower():
            finds.append(Finding(filename,0,"Data",f"Postcode differs: expected '{row['postcode']}'", []))
    return finds

# ---------- annotate ----------
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

# ---------- learning ----------
def learn_from_feedback(
    fb_df: pd.DataFrame,
    rules_blob: dict,
    immediate_disable: bool = True,
    target_meta: Optional[dict] = None,
    default_category: str = "Checklist",
):
    """
    Learn from a Findings CSV labeled with 'Valid'/'Not Valid'.
      - Spelling + Not Valid -> add word to allowlist.
      - Checklist/Data/etc + Not Valid:
          * if message has [RULE_ID] -> disable/remove that rule
          * if "Missing required text: 'X'" -> remove 'X' from any matching rule's require_text
          * if "Forbidden phrase present: 'X'" -> remove 'X' from forbid lists
      - 'Valid' on missing/forbid -> ADD a rule (category = default_category) with this requirement/forbid
        conditioned by target_meta (client/project/site_type/...).
    """
    import yaml

    # find label col
    label_col = None
    for c in fb_df.columns:
        if "valid" in str(c).lower() or fb_df[c].astype(str).str.contains("Valid", case=False, na=False).any():
            label_col = c; break
    if not label_col:
        st.warning("No 'Valid/Not Valid' column found — nothing learned.")
        return rules_blob

    lab = fb_df[label_col].astype(str).str.strip().str.lower()
    rules_blob.setdefault("rules", [])
    rules_list = rules_blob["rules"]

    # ---- Not Valid: Spelling -> allowlist
    mask_spell_nv = (fb_df["kind"].astype(str) == "Spelling") & (lab == "not valid")
    to_allow=set()
    for msg in fb_df.loc[mask_spell_nv,"message"].dropna().astype(str):
        m=re.search(r"Possible typo:\s*'([^']+)'", msg)
        if m and len(m.group(1))>=3:
            to_allow.add(m.group(1).lower())
    if to_allow:
        rules_blob.setdefault("allowlist", [])
        rules_blob["allowlist"]=sorted(set(rules_blob["allowlist"])|to_allow)

    # helpers to remove items from existing rules
    def _remove_from_rules(field: str, token: str):
        changed=False
        for r in rules_list:
            arr=(r.get("checks",{}) or {}).get(field, [])
            if isinstance(arr, list):
                if field=="forbid":
                    new=[x for x in arr if x.get("pattern")!=token]
                else:
                    new=[x for x in arr if x!=token]
                if len(new)!=len(arr):
                    r.setdefault("checks",{})[field]=new
                    changed=True
        return changed

    # ---- Not Valid: delete offending constraints
    mask_nv = (lab == "not valid") & (fb_df["kind"].astype(str) != "Spelling")
    if mask_nv.any():
        df_nv = fb_df.loc[mask_nv].copy()
        # disable by [RULE_ID]
        has_rule = df_nv["message"].astype(str).str.contains(r"\[[A-Za-z0-9\-_]+\]")
        if has_rule.any():
            df2=df_nv.loc[has_rule].copy()
            df2["rule_id"]=df2["message"].astype(str).str.extract(r"\[([A-Za-z0-9\-_]+)\]")
            counts=df2["rule_id"].value_counts()
            for rid, n in counts.items():
                if rid:
                    if immediate_disable or n>=3:
                        rules_blob["rules"]=[x for x in rules_list if x.get("id")!=rid]

        # remove require_text
        for msg in df_nv["message"].astype(str):
            m=re.search(r"Missing required text:\s*'([^']+)'", msg)
            if m:
                _remove_from_rules("require_text", m.group(1))
        # remove forbid phrase
        for msg in df_nv["message"].astype(str):
            m=re.search(r"Forbidden phrase present:\s*'([^']+)'", msg)
            if m:
                _remove_from_rules("forbid", m.group(1))

    # ---- Valid: add new rules based on messages
    mask_v = (lab == "valid") & (fb_df["kind"].astype(str) != "Spelling")
    if mask_v.any():
        cond = {}
        if target_meta:
            # turn meta into v2 'when' filter
            for k in ["client","project","site_type","vendor","cabinet_loc","radio_loc","config","mimo_config"]:
                v=str(target_meta.get(k,"")).strip()
                if v: cond[k]=[v]

        for _, row in fb_df.loc[mask_v].iterrows():
            msg=str(row.get("message",""))
            # Missing required text
            m_req=re.search(r"Missing required text:\s*'([^']+)'", msg)
            if m_req:
                rid=f"AUTO-REQ-{abs(hash(m_req.group(1)))%100000}"
                rules_list.append({
                    "id": rid,
                    "category": default_category,
                    "when": cond,
                    "checks": {"require_text":[m_req.group(1)]}
                })
                continue
            # Missing pattern (regex)
            m_rx=re.search(r"Missing pattern:\s*(.+)$", msg)
            if m_rx:
                rid=f"AUTO-RX-{abs(hash(m_rx.group(1)))%100000}"
                rules_list.append({
                    "id": rid,
                    "category": default_category,
                    "when": cond,
                    "checks": {"require_regex":[m_rx.group(1)]}
                })
                continue
            # Forbidden phrase present
            m_forb=re.search(r"Forbidden phrase present:\s*'([^']+)'", msg)
            if m_forb:
                rid=f"AUTO-FORB-{abs(hash(m_forb.group(1)))%100000}"
                rules_list.append({
                    "id": rid,
                    "category": default_category,
                    "when": cond,
                    "checks": {"forbid":[{"pattern": m_forb.group(1), "hint": ""}]}
                })

    return rules_blob

def apply_manual_qa(manual_df: pd.DataFrame, rules_blob: dict):
    """
    CSV columns (any can be blank):
      rule_id, rule_type {allowlist, forbid, require_text, require_regex, mutual_exclusion, disable_rule},
      pattern, regex, on_page_contains,
      client,project,site_type,vendor,cabinet_loc,radio_loc,config,mimo_config,
      category, hint, a, b
    -> Writes into rules_blob['rules'] (v2 schema) and/or allowlist.
    """
    rules_blob.setdefault("rules", [])
    rules_list = rules_blob["rules"]

    def _cond_from_row(r):
        cond = {}
        for k in ["client","project","site_type","vendor","cabinet_loc","radio_loc","config","mimo_config"]:
            v = str(r.get(k,"")).strip()
            if v:
                if ";" in v or "," in v:
                    cond[k] = [x.strip() for x in re.split(r"[;,]", v) if x.strip()]
                else:
                    cond[k] = [v]
        return cond

    for _, r in manual_df.fillna("").iterrows():
        rtype = str(r.get("rule_type","")).lower()
        rid = str(r.get("rule_id","") or f"MANUAL-{abs(hash(str(r)))%100000}")
        hint = str(r.get("hint",""))
        on_page_contains = str(r.get("on_page_contains","") or None)
        category = str(r.get("category","") or "Checklist")
        if category not in CATEGORIES: category = "Checklist"
        cond = _cond_from_row(r)

        if rtype == "allowlist":
            rules_blob.setdefault("allowlist", [])
            pat = str(r.get("pattern","")).strip().lower()
            if pat:
                rules_blob["allowlist"] = sorted(set(rules_blob["allowlist"]) | {pat})
            continue

        if rtype == "mutual_exclusion":
            a = str(r.get("a","")); b = str(r.get("b",""))
            if a and b:
                rules_blob.setdefault("mutual_exclusive", []).append({"a": a, "b": b})
            continue

        if rtype == "disable_rule":
            rules_blob["rules"] = [x for x in rules_list if x.get("id") != rid]
            continue

        entry = {"id": rid, "category": category, "when": cond, "checks": {}}
        if on_page_contains:
            entry["checks"]["on_page_contains"] = on_page_contains

        if rtype == "require_text":
            pt = str(r.get("pattern","")).strip()
            entry["checks"]["require_text"] = [pt] if pt else []
        elif rtype == "require_regex":
            rx = str(r.get("regex","")).strip()
            entry["checks"]["require_regex"] = [rx] if rx else []
        elif rtype == "forbid":
            pt = str(r.get("pattern","")).strip()
            entry["checks"]["forbid"] = [{"pattern": pt, "hint": hint}] if pt else []

        # upsert
        idx = next((i for i,x in enumerate(rules_list) if x.get("id")==rid), None)
        if idx is None:
            rules_list.append(entry)
        else:
            ex = rules_list[idx]
            ex["category"] = entry["category"] or ex.get("category") or "Checklist"
            ex["when"] = entry["when"] or ex.get("when",{})
            ex_checks = ex.setdefault("checks",{})
            for k,v in entry["checks"].items():
                ex_list = ex_checks.setdefault(k,[])
                if isinstance(v, list):
                    ex_checks[k] = list({*ex_list, *v})
                elif isinstance(v, dict):
                    ex_checks[k] = ex_checks.get(k,[])
                    ex_checks[k].append(v)

    return rules_blob

# ---------- UI ----------
st.set_page_config(page_title="AI Design QA — v6 (single audit + categories)", layout="wide")
st.title("AI Design Quality Auditor — v6")
st.caption("Single-file audits, metadata-aware rules, fast learning to the category you choose.")

with st.sidebar:
    st.header("Audit Metadata (single file)")
    meta = {
        "client": st.selectbox("Client", ["BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"]),
        "project": st.selectbox("Project", ["RAN","Power Resilience","East Unwind","Beacon 4"]),
        "site_type": st.selectbox("Site Type", ["Greenfield","Rooftop","Streetworks"]),
        "vendor": st.selectbox("Proposed Vendor", ["Ericsson","Nokia"]),
        "cabinet_loc": st.selectbox("Proposed Cabinet Location", ["Indoor","Outdoor"]),
        "radio_loc": st.selectbox("Proposed Radio Location", ["High Level","Low Level","Indoor and Door"]),
        "config": st.text_input("Proposed Config"),
        "mimo_config": st.text_input("Proposed MIMO Config"),
        "site_address": st.text_area("Site Address"),
    }

    st.divider(); st.subheader("Rules & Learning")
    rules_file = st.file_uploader("Rules pack (YAML)", type=["yaml","yml"])
    default_rule_category = st.selectbox("New rule category (for learning & quick-add)", CATEGORIES, index=0)

    st.divider(); st.subheader("Manual QA → Rules")
    manual_csv = st.file_uploader("Manual / Quick Rules CSV", type=["csv"])
    apply_manual_btn = st.button("Apply Manual/Quick Rules")

    st.divider(); st.subheader("Learn from labeled findings")
    fb_csv = st.file_uploader("Findings CSV (with Valid/Not Valid)", type=["csv"])
    immediate_disable = st.checkbox("Disable rules immediately when Not Valid", value=True)
    learn_btn = st.button("Apply learning to rules")

    st.divider(); st.subheader("QA Operator")
    qa_user  = st.text_input("Your name / initials", value="")
    qa_stage = st.selectbox("Stage", ["First Check","Second Check","Final Sign-off"], index=0)

# load rules blob (v2)
rules_blob = load_rules(rules_file)

# manual rules apply
if apply_manual_btn and manual_csv is not None:
    try:
        mdf = pd.read_csv(manual_csv)
    except Exception:
        manual_csv.seek(0); mdf = pd.read_excel(manual_csv)
    rules_blob = apply_manual_qa(mdf, rules_blob)
    import yaml
    buf = io.StringIO()
    yaml.safe_dump(rules_blob, buf, sort_keys=False, allow_unicode=True)
    st.success("Manual / Quick Rules applied. Download updated rules below.")
    st.download_button("Download updated rules", data=buf.getvalue().encode("utf-8"), file_name="rules_updated.yaml")

# learning apply
if learn_btn and fb_csv is not None:
    try:
        fdf = pd.read_csv(fb_csv)
    except Exception:
        fb_csv.seek(0); fdf = pd.read_excel(fb_csv)
    rules_blob = learn_from_feedback(
        fdf, rules_blob,
        immediate_disable=immediate_disable,
        target_meta=meta,
        default_category=default_rule_category
    )
    import yaml
    buf = io.StringIO()
    yaml.safe_dump(rules_blob, buf, sort_keys=False, allow_unicode=True)
    st.success("Learning applied. Download updated rules below.")
    st.download_button("Download learned rules", data=buf.getvalue().encode("utf-8"), file_name="rules_learned.yaml")

# single file upload & run
upload = st.file_uploader("Upload a single PDF", type=["pdf"], accept_multiple_files=False)
run_btn = st.button("Run Audit", type="primary")

def _load_csv(u):
    if u is None: return None
    try: return pd.read_csv(u)
    except Exception:
        u.seek(0); return pd.read_excel(u)

if run_btn and upload is not None:
    pdf_bytes = upload.read()
    pages, used_ocr = extract_pages_with_ocr(pdf_bytes)

    findings=[]
    capitals={t for t in tokenise("\n".join(p["text"] for p in pages)) if t.isupper()}
    client_allow={w.lower() for w in capitals}
    allowlist=set([w.lower() for w in rules_blob.get("allowlist", [])])

    # Spelling
    findings += spell_findings(pages, allowlist, client_allow)
    # Conditioned rules
    findings += run_conditioned_rules(pages, rules_blob, meta)
    # (Optional) roster/other checks can be added here

    rows = [{"page": f.page, "kind": f.kind, "message": f.message, "boxes": f.boxes} for f in findings]
    counts = {k: sum(1 for r in rows if r["kind"]==k) for k in CATEGORIES}
    results = {
        upload.name: {"summary": {"pages": len(pages), "used_ocr": used_ocr, "counts": counts}, "findings": rows}
    }

    # ---- Excel + status ----
    tmp=tempfile.mkdtemp(prefix="audit_")
    excel_path=os.path.join(tmp,"report.xlsx")
    sum_rows=[{"file": upload.name, "status": _status_from_counts(counts), **counts, "pages": len(pages), "used_ocr": used_ocr}]
    det_rows=[{"file": upload.name, **r} for r in rows]

    with pd.ExcelWriter(excel_path, engine="openpyxl") as xw:
        pd.DataFrame(sum_rows).to_excel(xw,"Summary",index=False)
        pd.DataFrame(det_rows).to_excel(xw,"Findings",index=False)

    overall_total = sum(counts.values())
    overall_status = "PASS" if overall_total==0 else "REJECTED"
    stamp = datetime.utcnow().strftime("%Y-%m-%d")
    base = os.path.splitext(upload.name)[0]
    excel_filename = f"{base}__{overall_status}__{stamp}.xlsx"

    st.success("Audit complete.")
    st.download_button("Download Excel report", data=open(excel_path,"rb").read(), file_name=excel_filename)

    # Optional annotated PDF
    if fitz is not None:
        annotated = annotate_pdf(pdf_bytes, rows)
        st.download_button("Download annotated PDF", data=annotated, file_name=f"{base}__annotated.pdf")

    # PASS/REJECTED banner
    if overall_status=="PASS":
        st.success("✅ **QA PASS** — please continue with **Second Check**.")
    else:
        st.error("❌ **REJECTED** — findings detected. Please fix and re-run.")

    # tables
    st.write("#### Summary"); st.dataframe(pd.DataFrame(sum_rows))
    st.write("#### Findings"); st.dataframe(pd.DataFrame(det_rows))

    # history with metadata
    log_df = record_history(qa_user, qa_stage, meta, results)
    st.write("### Audit history (latest 200)")
    st.dataframe(log_df.tail(200))
    try:
        st.download_button("Download full audit history CSV", data=open(LOG_CSV,"rb").read(), file_name="audit_history.csv")
    except Exception:
        pass

else:
    st.info("Fill in metadata, upload one PDF, then click **Run Audit**.")
