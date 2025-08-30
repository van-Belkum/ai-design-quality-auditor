import re
import io
import os
import ast
import json
from datetime import datetime, timezone
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import yaml
import fitz  # PyMuPDF
from spellchecker import SpellChecker

APP_VERSION = "rules-engine-v4-annotate-ui"

# ============================ Persistence ============================

def ensure_history():
    os.makedirs("history", exist_ok=True)
    sup_path = os.path.join("history", "suppressions.json")
    if not os.path.exists(sup_path):
        with open(sup_path, "w", encoding="utf-8") as f:
            json.dump({"messages": [], "patterns": []}, f, indent=2)
    log_path = os.path.join("history", "audit_log.csv")
    if not os.path.exists(log_path):
        pd.DataFrame(columns=[
            "timestamp_utc","user","file","status","total_findings","pages","meta"
        ]).to_csv(log_path, index=False)
    return sup_path, log_path

def read_suppressions():
    sup_path, _ = ensure_history()
    with open(sup_path, "r", encoding="utf-8") as f:
        return json.load(f)

def update_suppressions(new_msgs, new_patterns):
    sup_path, _ = ensure_history()
    blob = read_suppressions()
    msg = set(blob.get("messages", []))
    pat = set(blob.get("patterns", []))
    msg.update([m for m in new_msgs if isinstance(m, str) and m.strip()])
    pat.update([p for p in new_patterns if isinstance(p, str) and p.strip()])
    blob["messages"] = sorted(msg)
    blob["patterns"] = sorted(pat)
    with open(sup_path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)

def log_audit(file, status, total, pages, meta: dict):
    _, log_path = ensure_history()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([{
        "timestamp_utc": ts,
        "user": "web",
        "file": file,
        "status": status,
        "total_findings": total,
        "pages": pages,
        "meta": json.dumps(meta, ensure_ascii=False)
    }])
    try:
        old = pd.read_csv(log_path)
        new = pd.concat([row, old], axis=0, ignore_index=True).head(300)
        new.to_csv(log_path, index=False)
    except Exception:
        row.to_csv(log_path, index=False)

# ============================ Helpers ============================

def load_rules(upload) -> dict:
    """Load YAML rules either from uploaded file or local rules_example.yaml."""
    try:
        if upload is None:
            with open("rules_example.yaml","r",encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        else:
            return yaml.safe_load(upload) or {}
    except Exception as e:
        st.error(f"Rules YAML load error: {e}")
        return {}

def apply_suppressions(df: pd.DataFrame, sup: dict) -> pd.DataFrame:
    if df.empty:
        return df
    msgs = set(sup.get("messages", []))
    pats = [re.compile(p, re.I) for p in sup.get("patterns", []) if p]
    keep_idx = []
    for i, r in df.iterrows():
        m = r.get("message","")
        if isinstance(m, str) and (m in msgs or any(p.search(m) for p in pats)):
            continue
        keep_idx.append(i)
    return df.loc[keep_idx].reset_index(drop=True)

# ============================ Text Extraction ============================

def pdf_text_pages(file_bytes: bytes) -> List[str]:
    texts = []
    with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
        for p in doc:
            texts.append(p.get_text("text") or "")
    return texts

# ============================ Base Checks ============================

def spelling_checks(pages: List[str], allowlist: set) -> List[dict]:
    """Return findings with page+term for annotation."""
    sp = SpellChecker(distance=1)
    findings = []
    for i, page in enumerate(pages, start=1):
        words = re.findall(r"[A-Za-z][A-Za-z'\-]{1,}", page)
        for w in words:
            wl = w.lower()
            if wl in allowlist:
                continue
            if wl in sp:
                continue
            try:
                cand = sp.candidates(wl) or []
            except Exception:
                cand = []
            sug = next(iter(cand), None)
            if sug and sug != wl:
                findings.append({
                    "page": i,
                    "kind": "Spelling",
                    "message": f"Possible typo: '{w}' -> '{sug}'",
                    "severity": "low",
                    "rule_id": "SPELLING",
                    "term": w
                })
    return findings

# ============================ Rules Engine ============================

def when_matches(when: dict, meta: dict) -> bool:
    if not when: return True
    for key, val in when.items():
        mv = meta.get(key)
        if isinstance(val, list):
            if mv not in val: return False
        else:
            if mv != val: return False
    return True

def run_check(check: dict, pages: List[str], doc, meta: dict) -> List[dict]:
    ctype = check.get("type")
    severity = check.get("severity","low")
    cid = check.get("id", ctype.upper())
    out = []
    all_text = " ".join(pages)

    if ctype == "include_text":
        txt = check.get("text","")
        if txt and txt.lower() not in all_text.lower():
            out.append({
                "page": 0, "kind": "Checklist",
                "message": f"Missing required text: '{txt}'",
                "severity": severity, "rule_id": cid, "term": None
            })

    elif ctype == "forbid_together":
        a = check.get("a",""); b = check.get("b","")
        if a and b and a.lower() in all_text.lower() and b.lower() in all_text.lower():
            for pno, page in enumerate(doc, start=1):
                for needle in [a, b]:
                    quads = page.search_for(needle, quads=True) or page.search_for(needle.lower(), quads=True)
                    for q in quads:
                        rect = fitz.Rect(q.rect)
                        out.append({
                            "page": pno, "kind": "Consistency",
                            "message": f"Forbidden together: '{a}' with '{b}'",
                            "severity": severity, "rule_id": cid,
                            "term": needle, "bbox": [rect.x0, rect.y0, rect.x1, rect.y1]
                        })

    elif ctype == "regex_must":
        rgx = check.get("regex","")
        hint = check.get("hint","")
        if rgx and not re.search(rgx, all_text, re.I|re.M):
            msg = f"Required pattern missing: /{rgx}/"
            if hint: msg += f" – {hint}"
            out.append({
                "page": 0, "kind": "Checklist", "message": msg,
                "severity": severity, "rule_id": cid, "term": None
            })

    elif ctype == "regex_forbid":
        rgx = check.get("regex","")
        hint = check.get("hint","")
        if rgx:
            comp = re.compile(rgx, re.I|re.M)
            for pno, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                for m in comp.finditer(text):
                    matched = m.group(0)
                    quads = page.search_for(matched, quads=True)
                    if not quads and matched.strip():
                        quads = page.search_for(matched.strip(), quads=True)
                    if quads:
                        for q in quads:
                            rect = fitz.Rect(q.rect)
                            msg = f"Forbidden pattern present: /{rgx}/"
                            if hint: msg += f" – {hint}"
                            out.append({
                                "page": pno, "kind": "Consistency",
                                "message": msg, "severity": severity,
                                "rule_id": cid, "term": matched,
                                "bbox": [rect.x0, rect.y0, rect.x1, rect.y1]
                            })
                    else:
                        msg = f"Forbidden pattern present: /{rgx}/"
                        if hint: msg += f" – {hint}"
                        out.append({
                            "page": pno, "kind": "Consistency",
                            "message": msg, "severity": severity,
                            "rule_id": cid, "term": matched
                        })

    elif ctype == "numeric_range":
        field = check.get("field")
        mn = check.get("min")
        mx = check.get("max")
        try:
            val = float(meta.get(field)) if meta.get(field) not in (None, "") else None
        except Exception:
            val = None
        if val is None or (mn is not None and val < mn) or (mx is not None and val > mx):
            out.append({
                "page": 0, "kind": "Data",
                "message": f"Field '{field}' out of range [{mn},{mx}] (got {meta.get(field)!r})",
                "severity": severity, "rule_id": cid, "term": None
            })

    return out

def run_rules_engine(rules: dict, pages: List[str], doc, meta: dict) -> List[dict]:
    findings = []
    for chk in rules.get("checks", []):
        if when_matches(chk.get("when", {}), meta):
            findings.extend(run_check(chk, pages, doc, meta))
    return findings

# ============================ PDF Annotation ============================

def annotate_pdf(original_bytes: bytes, findings: List[dict]) -> bytes:
    """
    Adds sticky note annotations at the bbox if provided.
    If only 'term' and 'page' are available, attempts to locate with search_for.
    Handles bbox stored as a string (e.g. '[x0,y0,x1,y1]') safely.
    """
    if not findings:
        return original_bytes

    buf_in = io.BytesIO(original_bytes)
    doc = fitz.open(stream=buf_in, filetype="pdf")

    def add_note(page, rect: fitz.Rect, contents: str):
        note = page.add_text_annot(rect.tl, contents)
        try:
            note.set_icon("Comment")
        except Exception:
            pass
        note.update()

    for f in findings:
        pno = f.get("page", 0)
        if not isinstance(pno, int):
            try:
                pno = int(pno)
            except Exception:
                pno = 0
        if pno <= 0 or pno > len(doc):
            continue

        page = doc[pno-1]
        contents = f"[{f.get('kind','')}] {f.get('message','')}"
        bbox = f.get("bbox")
        term = f.get("term")

        # Normalize bbox if it is a string
        if isinstance(bbox, str):
            try:
                bbox = ast.literal_eval(bbox)
            except Exception:
                bbox = None

        targets = []
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                targets.append(fitz.Rect(*bbox))
            except Exception:
                pass
        elif isinstance(term, str) and term.strip():
            quads = page.search_for(term, quads=True) or page.search_for(term.strip(), quads=True)
            for q in quads:
                targets.append(fitz.Rect(q.rect))

        if not targets:
            continue

        for idx, r in enumerate(targets):
            anchor = fitz.Rect(r.x0, max(r.y0 - 8, 0), r.x0 + 16, max(r.y0 + 8, 16))
            add_note(page, anchor, contents)
            if idx >= 4:  # avoid clutter
                break

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()

# ============================ UI (Streamlit) ============================

st.set_page_config(page_title="AI Design QA — Rules Engine", layout="wide")
st.title("AI Design Quality Auditor")
st.caption(f"Build {APP_VERSION}")

# ---------- Session state for metadata ----------
DEFAULTS = {
    "client": "",
    "project": "",
    "site_type": "",
    "vendor": "",
    "cabinet_loc": "",
    "radio_loc": "",
    "sectors": "",
    "mimo_config": "",
    "site_address": "",
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def clear_metadata():
    for k in DEFAULTS:
        st.session_state[k] = DEFAULTS[k]
    st.session_state["rules_file"] = None

with st.sidebar:
    st.subheader("Audit metadata")

    st.session_state["client"] = st.selectbox(
        "Client", ["", "BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"],
        index=(["", "BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"].index(st.session_state["client"])
               if st.session_state["client"] in ["", "BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"] else 0)
    )

    st.session_state["project"] = st.selectbox(
        "Project", ["", "RAN","Power Resilience","East Unwind","Beacon 4"],
        index=(["", "RAN","Power Resilience","East Unwind","Beacon 4"].index(st.session_state["project"])
               if st.session_state["project"] in ["", "RAN","Power Resilience","East Unwind","Beacon 4"] else 0)
    )

    st.session_state["site_type"] = st.selectbox(
        "Site Type", ["", "Greenfield","Rooftop","Streetworks"],
        index=(["", "Greenfield","Rooftop","Streetworks"].index(st.session_state["site_type"])
               if st.session_state["site_type"] in ["", "Greenfield","Rooftop","Streetworks"] else 0)
    )

    st.session_state["vendor"] = st.selectbox(
        "Proposed Vendor", ["", "Ericsson","Nokia"],
        index=(["", "Ericsson","Nokia"].index(st.session_state["vendor"])
               if st.session_state["vendor"] in ["", "Ericsson","Nokia"] else 0)
    )

    st.session_state["cabinet_loc"] = st.selectbox(
        "Proposed Cabinet Location", ["", "Indoor","Outdoor"],
        index=(["", "Indoor","Outdoor"].index(st.session_state["cabinet_loc"])
               if st.session_state["cabinet_loc"] in ["", "Indoor","Outdoor"] else 0)
    )

    st.session_state["radio_loc"] = st.selectbox(
        "Proposed Radio Location", ["", "High Level","Low Level","Indoor","Door"],
        index=(["", "High Level","Low Level","Indoor","Door"].index(st.session_state["radio_loc"])
               if st.session_state["radio_loc"] in ["", "High Level","Low Level","Indoor","Door"] else 0)
    )

    st.session_state["sectors"] = st.selectbox(
        "Quantity of Sectors", ["", 1,2,3,4,5,6],
        index=(["", 1,2,3,4,5,6].index(st.session_state["sectors"])
               if st.session_state["sectors"] in ["", 1,2,3,4,5,6] else 0)
    )

    # Hide MIMO field for Power Resilience
    if st.session_state["project"] != "Power Resilience":
        st.session_state["mimo_config"] = st.text_input(
            "Proposed MIMO Config", value=st.session_state["mimo_config"]
        )
    else:
        st.session_state["mimo_config"] = ""

    st.session_state["site_address"] = st.text_area(
        "Site Address", height=80, value=st.session_state["site_address"]
    )

    rules_file = st.file_uploader("Rules YAML (or use rules_example.yaml)", type=["yaml","yml"])

    quick_csv = st.file_uploader("Feedback CSV (message, decision)", type=["csv"])
    if quick_csv is not None:
        try:
            qdf = pd.read_csv(quick_csv)
            not_valid = qdf.loc[
                qdf["decision"].astype(str).str.lower().str.contains("not valid", na=False),
                "message"
            ].dropna().astype(str).tolist()
            update_suppressions(not_valid, [])
            st.success(f"Stored {len(not_valid)} suppression(s) from feedback.")
        except Exception as e:
            st.error(f"Feedback CSV error: {e}")

    st.button("Clear metadata", on_click=clear_metadata)

# ---------- Load rules ----------
rules = load_rules(rules_file)

# ---------- Main Panel ----------
st.subheader("1) Upload PDF")
pdf = st.file_uploader("PDF", type=["pdf"])

# Metadata completeness check
meta_complete = all([
    st.session_state["client"],
    st.session_state["project"],
    st.session_state["site_type"],
    st.session_state["vendor"],
    st.session_state["cabinet_loc"],
    st.session_state["radio_loc"],
    st.session_state["sectors"] != "",
    st.session_state["site_address"],
])

# Audit button
audit_disabled_reason = None
if pdf is None:
    audit_disabled_reason = "Upload a PDF"
elif not meta_complete:
    audit_disabled_reason = "Complete all metadata"

run_audit = st.button(
    "Run Audit",
    type="primary",
    disabled=audit_disabled_reason is not None,
    help=audit_disabled_reason
)

if pdf is not None and not meta_complete:
    st.error("⚠️ Please complete all metadata fields before running an audit.")

if run_audit and pdf is not None and meta_complete:
    original_bytes = pdf.read()

    # Extract text + an open doc for search/annotate
    pages_text = []
    doc_for_rules = fitz.open(stream=io.BytesIO(original_bytes), filetype="pdf")
    for p in doc_for_rules:
        pages_text.append(p.get_text("text") or "")

    meta = {
        "client": st.session_state["client"],
        "project": st.session_state["project"],
        "site_type": st.session_state["site_type"],
        "vendor": st.session_state["vendor"],
        "cabinet_loc": st.session_state["cabinet_loc"],
        "radio_loc": st.session_state["radio_loc"],
        "sectors": st.session_state["sectors"],
        "mimo_config": st.session_state["mimo_config"],
        "site_address": st.session_state["site_address"],
    }

    # Run checks
    allow = set(w.lower() for w in (rules.get("allowlist") or []))
    findings = []
    findings += spelling_checks(pages_text, allow)
    findings += run_rules_engine(rules, pages_text, doc_for_rules, meta)

    # Build DF (keep rule_id/term/bbox for annotation and report)
    cols = ["page","kind","rule_id","message","severity","term","bbox"]
    df = pd.DataFrame(findings, columns=cols)
    if df.empty:
        df = pd.DataFrame(columns=cols)

    # Apply suppressions
    df = apply_suppressions(df, read_suppressions())

    # Summary + status
    total = len(df)
    status = "PASS" if total == 0 else "REJECTED"

    st.subheader("Summary")
    st.write({
        "file": pdf.name, "status": status, "pages": len(pages_text),
        "Spelling": int((df["kind"]=="Spelling").sum()) if not df.empty else 0,
        "Checklist": int((df["kind"]=="Checklist").sum()) if not df.empty else 0,
        "Consistency": int((df["kind"]=="Consistency").sum()) if not df.empty else 0,
        "Data": int((df["kind"]=="Data").sum()) if not df.empty else 0,
    })

    st.subheader("Findings")
    st.dataframe(df, use_container_width=True)

    # Excel export
    today = datetime.now().strftime("%Y%m%d")
    base = os.path.splitext(pdf.name)[0]
    out_xlsx = f"{base} - {status} - {today}.xlsx"
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as xw:
        df_out = df.copy()
        meta_df = pd.DataFrame([meta])
        df_out.to_excel(xw, sheet_name="Findings", index=False)
        meta_df.to_excel(xw, sheet_name="AuditMeta", index=False)
    st.download_button("Download report (Excel)", data=xbuf.getvalue(),
                       file_name=out_xlsx,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Annotated PDF export
    annotated_pdf_bytes = annotate_pdf(original_bytes, df.to_dict("records"))
    out_pdf = f"{base} - ANNOTATED - {today}.pdf"
    st.download_button("Download annotated PDF", data=annotated_pdf_bytes,
                       file_name=out_pdf, mime="application/pdf")

    # Close doc handle
    try:
        doc_for_rules.close()
    except Exception:
        pass

    # Log
    log_audit(pdf.name, status, total, len(pages_text), meta)

st.divider()
st.subheader("Audit history (latest 300)")
try:
    hist = pd.read_csv(os.path.join("history","audit_log.csv"))
    st.dataframe(hist, use_container_width=True)
except Exception:
    st.info("No history yet.")

st.divider()
st.subheader("Quick rules template")
# Provide a simple in-memory CSV in case the file isn't present locally
tpl = "message,decision\nExample finding text,Not Valid\nAnother finding,Valid\n"
st.download_button(
    "Download quick_rules_template.csv",
    data=tpl.encode("utf-8"),
    file_name="quick_rules_template.csv"
)

st.subheader("Current suppressions")
st.code(json.dumps(read_suppressions(), indent=2))
