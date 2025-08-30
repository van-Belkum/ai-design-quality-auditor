import os, io, re, json, tempfile, zipfile
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
    full_text="\n".join(p["text"] for p in pages)
    row=roster_df.iloc[0].fillna("")
    # lightweight starters
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

# ---------------- Learning / Manual QA ----------------
def learn_from_feedback(
    fb_df: pd.DataFrame,
    rules_blob: dict,
    immediate_disable: bool = True,
    target_client_name: Optional[str] = None,
):
    """
    Learn from a Findings CSV labeled with 'Valid'/'Not Valid'.
      - Spelling + Not Valid -> add word to allowlist.
      - Checklist + Not Valid with [RULE_ID] -> disable that page rule.
      - 'Missing required text: "X"' + Not Valid -> remove X from target client's global_includes.
      - "Forbidden phrase present: 'X'" + Not Valid -> remove 'X' from target client's forbids.
    If target_client_name is None, we fall back to the current sidebar client.
    """

    # pick a label column
    label_col = None
    for c in fb_df.columns:
        if "valid" in str(c).lower() or fb_df[c].astype(str).str.contains("Valid", case=False, na=False).any():
            label_col = c
            break
    if not label_col:
        st.warning("No 'Valid/Not Valid' column found — nothing learned.")
        return rules_blob

    labels = fb_df[label_col].astype(str).str.strip().str.lower()

    # which client to apply to?
    if not target_client_name:
        # try sidebar state; safe default "EE"
        target_client_name = st.session_state.get("client", "EE")
    target_client_name = str(target_client_name).upper()

    # Map target(s)
    rules_blob.setdefault("clients", [])
    name_to_client = {c.get("name","").upper(): c for c in rules_blob["clients"] if "name" in c}

    def _targets():
        has_client_col = any(col.lower() == "client" for col in fb_df.columns)
        if has_client_col:
            grouping = {}
            for idx, val in fb_df["client"].fillna("").astype(str).items():
                name = val.strip().upper() or target_client_name
                grouping.setdefault(name, []).append(idx)
            return grouping
        return {target_client_name: fb_df.index.tolist()}

    # 1) Spelling → allowlist
    mask_spell = (fb_df["kind"].astype(str) == "Spelling") & (labels == "not valid")
    to_allow = set()
    for msg in fb_df.loc[mask_spell, "message"].dropna().astype(str):
        m = re.search(r"Possible typo:\s*'([^']+)'", msg)
        if m and len(m.group(1)) >= 3:
            to_allow.add(m.group(1).lower())
    if to_allow:
        rules_blob.setdefault("allowlist", [])
        rules_blob["allowlist"] = sorted(set(w.lower() for w in rules_blob["allowlist"]) | to_allow)

    # 2) Checklist with [RULE_ID] → disable page rule
    mask_chk_rule = (fb_df["kind"].astype(str) == "Checklist") & (labels == "not valid") \
                    & fb_df["message"].astype(str).str.contains(r"\[[A-Za-z0-9\-_]+\]")
    if mask_chk_rule.any():
        df2 = fb_df.loc[mask_chk_rule].copy()
        df2["rule_id"] = df2["message"].astype(str).str.extract(r"\[([A-Za-z0-9\-_]+)\]")
        counts = df2["rule_id"].value_counts()
        for rid, n in counts.items():
            if not rid:
                continue
            if immediate_disable or n >= 3:
                for client in rules_blob.get("clients", []):
                    client["page_rules"] = [r for r in client.get("page_rules", []) if r.get("id") != rid]

    # 3) Global includes → remove string
    mask_missing = (fb_df["kind"].astype(str) == "Checklist") & (labels == "not valid") \
                   & fb_df["message"].astype(str).str.contains(r"Missing required text:\s*'")
    if mask_missing.any():
        for client_name, row_idxs in _targets().items():
            client_pack = name_to_client.get(client_name)
            if not client_pack:
                client_pack = {"name": client_name, "global_includes": [], "forbids": [], "page_rules": []}
                rules_blob["clients"].append(client_pack)
                name_to_client[client_name] = client_pack
            gi = client_pack.setdefault("global_includes", [])
            remove_set = set()
            for msg in fb_df.loc[row_idxs, "message"].dropna().astype(str):
                m = re.search(r"Missing required text:\s*'([^']+)'", msg)
                if m:
                    remove_set.add(m.group(1))
            if remove_set:
                client_pack["global_includes"] = [s for s in gi if s not in remove_set]

    # 4) Client forbids → remove phrase
    mask_forbid = (fb_df["kind"].astype(str) == "Checklist") & (labels == "not valid") \
                  & fb_df["message"].astype(str).str.contains(r"Forbidden phrase present:\s*'")
    if mask_forbid.any():
        for client_name, row_idxs in _targets().items():
            client_pack = name_to_client.get(client_name)
            if not client_pack:
                continue
            forb = client_pack.setdefault("forbids", [])
            remove_set = set()
            for msg in fb_df.loc[row_idxs, "message"].dropna().astype(str):
                m = re.search(r"Forbidden phrase present:\s*'([^']+)'", msg)
                if m:
                    remove_set.add(m.group(1))
            if remove_set:
                client_pack["forbids"] = [f for f in forb if f.get("pattern") not in remove_set]

    return rules_blob

def apply_manual_qa(manual_df: pd.DataFrame, rules_blob: dict):
    """
    template_manual_qa.csv columns:
      client, rule_type, pattern, rule_id, when_page_contains, hint, a, b
    rule_type in {allowlist, forbid, require_regex, forbid_regex, mutual_exclusion, disable_rule}
    """
    rules_blob.setdefault("clients", [])
    idx = {c["name"].upper(): i for i, c in enumerate(rules_blob["clients"]) if "name" in c}

    def get_client(name: str):
        name = (name or "GLOBAL").upper()
        if name not in idx:
            rules_blob["clients"].append({"name": name, "global_includes": [], "forbids": [], "page_rules": []})
            idx[name] = len(rules_blob["clients"]) - 1
        return rules_blob["clients"][idx[name]]

    for _, r in manual_df.fillna("").iterrows():
        client = r.get("client", "GLOBAL")
        rtype = str(r.get("rule_type", "")).lower()
        pattern = str(r.get("pattern", ""))
        rule_id = str(r.get("rule_id", "") or f"MANUAL-{abs(hash(pattern))%100000}")
        wpc = str(r.get("when_page_contains", "") or None)
        hint = str(r.get("hint", ""))
        a = str(r.get("a", "")); b = str(r.get("b", ""))

        if rtype == "allowlist":
            rules_blob.setdefault("allowlist", [])
            rules_blob["allowlist"] = sorted(set(rules_blob["allowlist"]) | {pattern.lower()})
            continue

        if rtype == "mutual_exclusion":
            if a and b:
                rules_blob.setdefault("mutual_exclusive", []).append({"a": a, "b": b})
            continue

        targets = rules_blob["clients"] if client.upper() == "GLOBAL" else [get_client(client)]

        if rtype == "disable_rule":
            for t in targets:
                t["page_rules"] = [pr for pr in t.get("page_rules", []) if pr.get("id") != rule_id]
            continue

        if rtype == "forbid":
            for t in targets:
                t.setdefault("forbids", []).append({"pattern": pattern, "hint": hint})
            continue

        if rtype in ("require_regex", "forbid_regex"):
            for t in targets:
                pr_list = t.setdefault("page_rules", [])
                # upsert page rule
                pr = None
                for x in pr_list:
                    if x.get("id") == rule_id and x.get("when_page_contains") == wpc:
                        pr = x; break
                if pr is None:
                    pr = {"id": rule_id, "when_page_contains": wpc, "must_include_regex": [], "forbids": []}
                    pr_list.append(pr)
                if rtype == "require_regex":
                    pr.setdefault("must_include_regex", []).append(pattern)
                else:
                    pr.setdefault("forbids", []).append({"pattern": pattern, "hint": hint})
            continue

    return rules_blob

# ---------------- UI ----------------
st.set_page_config(page_title="AI Design QA — Learn v5.2", layout="wide")
st.title("AI Design Quality Auditor — Learn v5.2")
st.caption("OCR, PDF markups, Excel, history, PASS/REJECTED banner, learning & manual QA (incl. global_includes).")

with st.sidebar:
    st.header("Configuration")
    client = st.selectbox("Client", ["EE","H3G","MBNL","VODAFONE","CUSTOM"], index=0)
    st.session_state["client"] = client  # used by learner fallback
    rules_file = st.file_uploader("Rules pack (YAML)", type=["yaml","yml"])

    st.divider(); st.subheader("Reference data")
    roster_csv = st.file_uploader("Site roster CSV (optional)", type=["csv"])

    st.divider(); st.subheader("Outputs")
    gen_markups = st.checkbox("Generate marked-up PDFs (highlights + notes)", value=True)

    st.divider(); st.subheader("QA Operator")
    qa_user  = st.text_input("Your name / initials", value="")
    qa_stage = st.selectbox("Stage", ["First Check","Second Check","Final Sign-off"], index=0)

    st.divider(); st.subheader("Manual QA → Rules")
    manual_csv = st.file_uploader("Manual QA CSV", type=["csv"])
    apply_manual_btn = st.button("Apply Manual QA to rules")

    st.divider(); st.subheader("Learn from labeled findings")
    fb_csv = st.file_uploader("Findings CSV (with Valid/Not Valid)", type=["csv"])
    learning_client = st.selectbox(
        "Apply learning to client",
        ["(current sidebar client)", "EE", "H3G", "MBNL", "VODAFONE", "CUSTOM"],
        index=0
    )
    immediate_disable = st.checkbox("Disable checklist rules immediately when Not Valid", value=True)
    learn_btn = st.button("Apply learning to rules")

    run_btn = st.button("Run Audit", type="primary")

# load rules
rules_blob = load_rules(rules_file)

# --- Apply Manual QA CSV to rules ---
if apply_manual_btn and manual_csv is not None:
    try:
        mdf = pd.read_csv(manual_csv)
    except Exception:
        manual_csv.seek(0); mdf = pd.read_excel(manual_csv)
    rules_blob = apply_manual_qa(mdf, rules_blob)
    import yaml
    buf = io.StringIO()
    yaml.safe_dump(rules_blob, buf, sort_keys=False, allow_unicode=True)
    st.success("Manual QA applied. Download updated rules below.")
    st.download_button("Download updated rules (Manual QA)", data=buf.getvalue().encode("utf-8"),
                       file_name="rules_manual.yaml")

# --- Learn from labeled findings (Valid / Not Valid) ---
if learn_btn and fb_csv is not None:
    try:
        fdf = pd.read_csv(fb_csv)
    except Exception:
        fb_csv.seek(0); fdf = pd.read_excel(fb_csv)

    _target = None if learning_client == "(current sidebar client)" else learning_client
    rules_blob = learn_from_feedback(
        fdf, rules_blob,
        immediate_disable=immediate_disable,
        target_client_name=_target or client
    )

    import yaml
    buf = io.StringIO()
    yaml.safe_dump(rules_blob, buf, sort_keys=False, allow_unicode=True)
    st.success("Learning applied. Download updated rules below.")
    st.download_button("Download updated rules (Learning)", data=buf.getvalue().encode("utf-8"),
                       file_name="rules_learned.yaml")

def _load_csv(u):
    if u is None: return None
    try: return pd.read_csv(u)
    except Exception:
        u.seek(0); return pd.read_excel(u)

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

    # Optional annotated PDFs
    if gen_markups and fitz is not None:
        zip_path = os.path.join(tmp, "annotated_pdfs.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for fn,res in results.items():
                annotated = annotate_pdf(pdf_bytes[fn], res["findings"])
                outp = os.path.join(tmp, f"{os.path.splitext(fn)[0]}_annotated.pdf")
                open(outp,"wb").write(annotated)
                z.write(outp, arcname=os.path.basename(outp))
        st.download_button("Download annotated PDFs (zip)", data=open(zip_path,"rb").read(), file_name="annotated_pdfs.zip")

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

    # ---- Promote findings to rules (workbench) ----
    st.write("### Promote selected findings → rules")
    if det_rows:
        df_find = pd.DataFrame(det_rows)
        choices = df_find["message"].dropna().unique().tolist()
        pick = st.multiselect("Pick finding messages", choices)
        rule_type = st.selectbox("Rule type", ["allowlist (spelling)", "forbid (phrase)", "require_regex (enter regex below)", "mutual_exclusion"])
        extra = st.text_input("Extra (regex pattern or 'A,B' for exclusion)")
        if st.button("Create rules from selection"):
            import yaml
            blob = rules_blob.copy()
            blob.setdefault("clients", [])

            if rule_type == "allowlist (spelling)":
                blob.setdefault("allowlist", [])
                for m in pick:
                    m2 = re.search(r"Possible typo:\s*'([^']+)'", m)
                    if m2:
                        blob["allowlist"].append(m2.group(1).lower())
                blob["allowlist"] = sorted(set(blob["allowlist"]))

            elif rule_type == "forbid (phrase)":
                phrase = extra.strip()
                if phrase:
                    for c in blob.get("clients", []):
                        c.setdefault("forbids", []).append({"pattern": phrase, "hint": ""})

            elif rule_type == "require_regex (enter regex below)":
                rx = extra.strip()
                if rx:
                    for c in blob.get("clients", []):
                        c.setdefault("page_rules", []).append(
                            {"id": "WB-RULE", "when_page_contains": None, "must_include_regex": [rx], "forbids": []}
                        )

            elif rule_type == "mutual_exclusion":
                parts = [p.strip() for p in (extra or "").split(",")]
                if len(parts) == 2:
                    blob.setdefault("mutual_exclusive", []).append({"a": parts[0], "b": parts[1]})

            out = io.StringIO()
            yaml.safe_dump(blob, out, sort_keys=False, allow_unicode=True)
            st.download_button("Download rules from selection", data=out.getvalue().encode("utf-8"),
                               file_name="rules_from_rows.yaml")
    else:
        st.caption("Run an audit to enable promoting findings to rules.")

else:
    st.info("Upload rules/roster (optional), then add PDFs and click Run Audit.")
