
import os, io, re, sys, json, tempfile, subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# ---- Safe imports (soft fail) ----
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
    st.warning("Optional dependency 'pyspellchecker' not installed (pip install pyspellchecker)")
pdfminer = safe_import("pdfminer", "pdfminer.six")

# ---- Data structures ----
from dataclasses import dataclass, field

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

# ---- Utils ----
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
    pages = []
    for i in range(len(doc)):
        try:
            pages.append(doc[i].get_text("text"))
        except Exception:
            pages.append("")
    return pages

def extract_text_pdfminer(buf: bytes) -> List[str]:
    try:
        from pdfminer.high_level import extract_text
        t = extract_text(io.BytesIO(buf)) or ""
        return [t]
    except Exception:
        return [""]

def page_skew_angle_from_pix(pix):
    if cv2 is None:
        return None
    try:
        img = np.frombuffer(pix.tobytes(), dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if coords.size == 0:
            return None
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        return float(angle)
    except Exception:
        return None

def detect_misalignment(doc):
    res = []
    if fitz is None:
        return res
    for i in range(len(doc)):
        try:
            pix = doc[i].get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
            ang = page_skew_angle_from_pix(pix)
            if ang is not None:
                res.append((i+1, ang))
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

def spell_findings(text_pages: List[str], allowlist: set, client_allow: set) -> List[Finding]:
    finds = []
    if SpellChecker is None:
        return finds
    sc = SpellChecker(language='en')
    for w in list(allowlist | client_allow):
        if re.match(r"^[A-Za-z][A-Za-z0-9\-]{1,}$", w):
            sc.word_frequency.add(w.lower())
    for pno, text in enumerate(text_pages, start=1):
        tokens = tokenise(text)
        check = [t.lower() for t in tokens if t.islower() and len(t) >= 3]
        miss = sc.unknown(check)
        for m in sorted(miss):
            if m in allowlist or m in client_allow:
                continue
            suggestion = next(iter(sc.candidates(m)), None)
            msg = f"Possible typo: '{m}'" + (f" → '{suggestion}'" if suggestion else "")
            finds.append(Finding(file="", page=pno, kind="Spelling", message=msg))
    return finds

def checklist_findings(text_pages: List[str], pack: ClientPack) -> List[Finding]:
    finds = []
    for pno, text in enumerate(text_pages, start=1):
        for s in pack.global_includes:
            if s and s not in text:
                finds.append(Finding(file="", page=pno, kind="Checklist", message=f"Missing required text: '{s}'"))
        for rule in pack.page_rules:
            if rule.when_page_contains and rule.when_page_contains not in text:
                continue
            for rx in rule.must_include_regex:
                if not re.search(rx, text, flags=re.IGNORECASE):
                    hint = rule.hint or ""
                    finds.append(Finding(file="", page=pno, kind="Checklist", message=f"[{rule.id}] Missing pattern: {rx}. {hint}".strip()))
            for forb in rule.forbids:
                pat = forb.get("pattern","")
                if pat and re.search(pat, text, flags=re.IGNORECASE):
                    finds.append(Finding(file="", page=pno, kind="Checklist", message=f"[{rule.id}] Forbidden phrase present: '{pat}'. Hint: {forb.get('hint','')}".strip()))
        for forb in pack.forbids:
            pat = forb.get("pattern","")
            if pat and re.search(pat, text, flags=re.IGNORECASE):
                finds.append(Finding(file="", page=pno, kind="Checklist", message=f"Forbidden phrase present: '{pat}'. Hint: {forb.get('hint','')}".strip()))
        if "W3W:" in text and not re.search(r"W3W:\s*[a-z]+\.[a-z]+\.[a-z]+", text):
            finds.append(Finding(file="", page=pno, kind="Data", message="W3W field present but empty/invalid"))
    return finds

def consistency_findings(text_pages: List[str]) -> List[Finding]:
    from rapidfuzz.distance import Levenshtein
    toks = set()
    for t in tokenise("\\n".join(text_pages)):
        if len(t) >= 6 and any(ch.isdigit() for ch in t):
            toks.add(t)
    toks = list(toks)
    finds = []
    for i, a in enumerate(toks):
        for b in toks[i+1:]:
            if a[0].isalpha() and b[0].isalpha():
                d = Levenshtein.distance(a, b)
                if d == 1 or (d == 2 and min(len(a), len(b)) >= 8):
                    finds.append(Finding(file="", page=0, kind="Consistency", message=f"Similar model codes found: '{a}' vs '{b}'"))
    return finds

def misalignment_findings(doc) -> List[Finding]:
    res = []
    for pno, angle in detect_misalignment(doc):
        if abs(angle) > 1.5:
            res.append(Finding(file="", page=pno, kind="Layout", message=f"Page skew angle ≈ {angle:.2f}° (threshold 1.5°)"))
    return res

def audit_pdf_bytes(buf: bytes, client: str, rules_blob: Dict[str, Any]) -> Dict[str, Any]:
    packs, allowlist = compile_client_packs(rules_blob)
    client_pack = packs.get(client.upper(), ClientPack(name=client.upper()))
    text_pages = []
    if fitz is not None:
        try:
            with fitz.open(stream=buf, filetype="pdf") as doc:
                text_pages = extract_text_pymupdf(doc)
        except Exception:
            text_pages = extract_text_pdfminer(buf)
    else:
        text_pages = extract_text_pdfminer(buf)

    capitals = {t for t in tokenise("\\n".join(text_pages)) if t.isupper()}
    client_allow = set([w.lower() for w in capitals])

    findings = []
    findings += spell_findings(text_pages, set([w.lower() for w in rules_blob.get("allowlist", [])]), client_allow)
    findings += checklist_findings(text_pages, client_pack)
    findings += consistency_findings(text_pages)

    if fitz is not None:
        try:
            with fitz.open(stream=buf, filetype="pdf") as doc:
                findings += misalignment_findings(doc)
        except Exception:
            pass

    pages = max(1, len(text_pages))
    summary = {
        "pages": pages,
        "counts": {
            "Spelling": sum(1 for f in findings if f.kind == "Spelling"),
            "Checklist": sum(2 if "Missing pattern" in f.message else 1 for f in findings if f.kind == "Checklist"),
            "Consistency": sum(1 for f in findings if f.kind == "Consistency"),
            "Layout": sum(1 for f in findings if f.kind == "Layout"),
            "Data": sum(1 for f in findings if f.kind == "Data"),
        }
    }
    rows = [{"page": f.page, "kind": f.kind, "message": f.message} for f in findings]
    return {"summary": summary, "findings": rows}

def write_excel_report(all_results: Dict[str, Dict[str, Any]], out_path: str):
    summaries, failures, typos, perpage = [], [], [], []
    for fname, result in all_results.items():
        s = result["summary"]; cnts = s["counts"]
        summaries.append({"file": fname, **cnts, "pages": s["pages"]})
        for row in result["findings"]:
            failures.append({"file": fname, **row})
            if row["kind"] == "Spelling":
                typos.append({"file": fname, **row})
            perpage.append({"file": fname, **row})
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        pd.DataFrame(summaries).to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame(failures).to_excel(xw, sheet_name="Failures", index=False)
        pd.DataFrame(typos).to_excel(xw, sheet_name="Typos", index=False)
        pd.DataFrame(perpage).to_excel(xw, sheet_name="PerPage", index=False)

def handle_dwg(file_name: str, data: bytes, workdir: str) -> Optional[str]:
    oda = os.environ.get("ODA_CONVERTER")
    if not oda:
        return None
    in_path = os.path.join(workdir, os.path.basename(file_name))
    with open(in_path, "wb") as f:
        f.write(data)
    out_dir = os.path.join(workdir, "dwg_out"); os.makedirs(out_dir, exist_ok=True)
    try:
        cmd = [oda, in_path, out_dir, "PDF", "true", "true"]
        subprocess.run(cmd, check=True)
        for f in os.listdir(out_dir):
            if f.lower().endswith(".pdf"):
                return os.path.join(out_dir, f)
    except Exception as e:
        st.warning(f"DWG conversion failed: {e}")
    return None

# ---- Streamlit UI ----
st.set_page_config(page_title="AI Design Quality Auditor", layout="wide")
st.title("AI Design Quality Auditor")
st.caption("Batch audit PDFs with spelling, layout, and per-client checklist rules.")

with st.sidebar:
    st.header("Configuration")
    client = st.selectbox("Client", ["EE","H3G","MBNL","VODAFONE","CUSTOM"], index=0)
    rules_file = st.file_uploader("Rules pack (YAML)", type=["yaml","yml"], accept_multiple_files=False)
    run_btn = st.button("Run Audit", type="primary")

st.write("### Upload files")
uploads = st.file_uploader("Drop PDFs (and optional DWG/DXF) here", type=["pdf","dwg","dxf"], accept_multiple_files=True)

if run_btn and uploads:
    rules_blob = load_rules(rules_file)
    results = {}
    tmp = tempfile.mkdtemp(prefix="audit_")
    try:
        for upl in uploads:
            fname = upl.name; data = upl.read()
            pdf_path = None
            if fname.lower().endswith(".pdf"):
                pdf_path = os.path.join(tmp, fname)
                with open(pdf_path, "wb") as f:
                    f.write(data)
            elif fname.lower().endswith(".dwg"):
                st.info(f"Converting DWG → PDF for {fname}...")
                converted = handle_dwg(fname, data, tmp)
                if converted:
                    pdf_path = converted
                else:
                    st.warning(f"Skipping {fname}: DWG conversion not available. Convert to PDF first.")
                    continue
            else:
                st.warning(f"Skipping {fname}: unsupported format.")
                continue

            with open(pdf_path, "rb") as f:
                buf = f.read()
            result = audit_pdf_bytes(buf, client=client, rules_blob=rules_blob)
            results[fname] = result

        out_dir = os.path.join(tmp, "reports"); os.makedirs(out_dir, exist_ok=True)
        excel_path = os.path.join(out_dir, "report.xlsx")
        write_excel_report(results, excel_path)

        rej_rows = []
        for fn, res in results.items():
            for r in res["findings"]:
                rej_rows.append({"file": fn, **r})
        import pandas as pd
        rej_df = pd.DataFrame(rej_rows)
        rej_csv = os.path.join(out_dir, "rejections.csv")
        rej_df.to_csv(rej_csv, index=False)

        st.success("Audit complete.")
        st.download_button("Download Excel report", data=open(excel_path, "rb").read(), file_name="report.xlsx")
        st.download_button("Download rejection list (CSV)", data=open(rej_csv, "rb").read(), file_name="rejections.csv")

        st.write("#### Summary")
        sum_rows = []
        for fn, res in results.items():
            c = res["summary"]["counts"]
            sum_rows.append({"file": fn, **c, "pages": res["summary"]["pages"]})
        st.dataframe(pd.DataFrame(sum_rows))

        st.write("#### Findings")
        st.dataframe(pd.DataFrame(rej_rows))
    finally:
        pass
elif run_btn and not uploads:
    st.warning("Please upload at least one file.")
else:
    st.info("Load your rules (or use the example), upload PDFs, then click **Run Audit**.")
