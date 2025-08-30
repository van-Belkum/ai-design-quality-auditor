import re
import io
import os
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import fitz  # PyMuPDF
from spellchecker import SpellChecker

APP_VERSION = "v7-pre-mimo-stable-patch"

# ---------------------------- Utils ----------------------------

def load_yaml(path_or_buffer) -> dict:
    try:
        return yaml.safe_load(path_or_buffer)
    except Exception as e:
        st.error(f"YAML load error: {e}")
        return {}

def ensure_history():
    os.makedirs("history", exist_ok=True)
    sup_path = os.path.join("history", "suppressions.json")
    if not os.path.exists(sup_path):
        with open(sup_path, "w") as f:
            json.dump({"messages": [], "patterns": []}, f, indent=2)
    log_path = os.path.join("history", "audit_log.csv")
    if not os.path.exists(log_path):
        pd.DataFrame(columns=["timestamp_utc","user","file","outcome","total_findings","used_ocr","pages"]).to_csv(log_path, index=False)
    return sup_path, log_path

def read_suppressions():
    sup_path, _ = ensure_history()
    with open(sup_path) as f:
        return json.load(f)

def update_suppressions(new_msgs: List[str], new_patterns: List[str]):
    sup_path, _ = ensure_history()
    blob = read_suppressions()
    msg_set = set(blob.get("messages", []))
    pat_set = set(blob.get("patterns", []))
    msg_set.update(m.strip() for m in new_msgs if m)
    pat_set.update(p.strip() for p in new_patterns if p)
    blob["messages"] = sorted(msg_set)
    blob["patterns"] = sorted(pat_set)
    with open(sup_path, "w") as f:
        json.dump(blob, f, indent=2)

def log_audit(user: str, file: str, outcome: str, total_findings: int, pages: int, used_ocr: bool=False):
    _, log_path = ensure_history()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([{
        "timestamp_utc": ts, "user": user or "anonymous",
        "file": file, "outcome": outcome,
        "total_findings": total_findings, "used_ocr": int(used_ocr), "pages": pages
    }])
    try:
        old = pd.read_csv(log_path)
        new = pd.concat([row, old], axis=0, ignore_index=True).head(200)
        new.to_csv(log_path, index=False)
    except Exception:
        row.to_csv(log_path, index=False)

# ---------------------------- Rules ----------------------------

def load_rules(rules_file) -> dict:
    if rules_file is None:
        with open("rules_example.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        return yaml.safe_load(rules_file)

# ---------------------------- Checkers ----------------------------

def text_from_pdf(file_bytes: bytes) -> List[str]:
    pages = []
    with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
        for p in doc:
            pages.append(p.get_text("text"))
    return pages

def spell_findings(pages: List[str], allowlist: set) -> List[Dict[str, Any]]:
    sp = SpellChecker(distance=1)
    findings = []
    for i, page in enumerate(pages, start=1):
        words = re.findall(r"[A-Za-z][A-Za-z'\\-]{1,}", page)
        for w in words:
            wl = w.lower()
            if wl in allowlist:
                continue
            if wl in sp:
                continue
            try:
                candidates = sp.candidates(wl) or []
            except Exception:
                candidates = []
            suggestion = next(iter(candidates), None)
            if suggestion and suggestion != wl:
                findings.append({
                    "page": i,
                    "kind": "Spelling",
                    "message": f"Possible typo: '{wl}' -> '{suggestion}'"
                })
    return findings

def required_text_findings(pages: List[str], required_strs: List[str]) -> List[Dict[str, Any]]:
    findings = []
    for i, page in enumerate(pages, start=1):
        lc = page.lower()
        for r in required_strs:
            if r.lower() not in lc:
                findings.append({
                    "page": i,
                    "kind": "Checklist",
                    "message": f"Missing required text: '{r}'"
                })
    return findings

def mutual_exclusive_findings(pages: List[str], pairs: List[Dict[str,str]]) -> List[Dict[str, Any]]:
    all_text = " ".join(pages).lower()
    fin = []
    for pair in pairs or []:
        a = pair.get("a","").lower()
        b = pair.get("b","").lower()
        if a and b and (a in all_text) and (b in all_text):
            fin.append({
                "page": 0,
                "kind": "Consistency",
                "message": f"Mutually exclusive terms both present: '{pair.get('a')}' and '{pair.get('b')}'"
            })
    return fin

def apply_suppressions(df: pd.DataFrame, sup: dict) -> pd.DataFrame:
    msgs = set(sup.get("messages", []))
    pats = [re.compile(p, re.I) for p in sup.get("patterns", [])]
    keep = []
    for _, r in df.iterrows():
        m = r["message"]
        if m in msgs:
            continue
        if any(p.search(m) for p in pats):
            continue
        keep.append(True)
    return df[keep].reset_index(drop=True)

# ---------------------------- UI ----------------------------

st.set_page_config(page_title="AI Design QA (stable)", layout="wide")
st.title("AI Design Quality Auditor")
st.caption(f"Stable build {APP_VERSION} — pre-MIMO changes")

col_left, col_right = st.columns([2,1])

with col_left:
    pdf = st.file_uploader("Upload a single PDF", type=["pdf"])
with col_right:
    rules_file = st.file_uploader("Optional rules YAML (or use rules_example.yaml)", type=["yaml","yml"])
    quick_csv = st.file_uploader("Upload 'quick_rules' feedback CSV (message, decision)", type=["csv"])

# Feedback loop (learn: Not Valid -> suppression)
if quick_csv is not None:
    try:
        qdf = pd.read_csv(quick_csv)
        not_valid = qdf.loc[qdf["decision"].str.lower().str.contains("not valid", na=False), "message"].dropna().tolist()
        update_suppressions(not_valid, [])
        st.success(f"Stored {len(not_valid)} suppression(s) from feedback CSV.")
    except Exception as e:
        st.error(f"Could not read feedback CSV: {e}")

rules = load_rules(rules_file)
sup_blob = read_suppressions()

client_names = [c.get("name") for c in rules.get("clients", [])]
client = st.selectbox("Client", client_names) if client_names else None

if pdf is not None:
    bytes_data = pdf.read()
    pages = text_from_pdf(bytes_data)

    allow = set(w.lower() for w in rules.get("allowlist", []))
    findings = []

    # Spell
    findings += spell_findings(pages, allow)

    # Client checks
    if client:
        cblob = next((c for c in rules["clients"] if c.get("name")==client), {})
        required = cblob.get("global_includes", [])
        findings += required_text_findings(pages, required)

    # Mutual exclusion
    findings += mutual_exclusive_findings(pages, rules.get("mutual_exclusive", []))

    df = pd.DataFrame(findings)
    if df.empty:
        df = pd.DataFrame(columns=["page","kind","message"])

    # Apply learned suppressions
    df = apply_suppressions(df, sup_blob)

    # Results / pass-fail
    total = len(df)
    outcome = "PASS" if total == 0 else "REJECTED"
    st.subheader("Summary")
    st.dataframe(pd.DataFrame([{
        "file": pdf.name, "status": outcome,
        "Spelling": int((df["kind"]=="Spelling").sum()) if not df.empty else 0,
        "Checklist": int((df["kind"]=="Checklist").sum()) if not df.empty else 0,
        "Consistency": int((df["kind"]=="Consistency").sum()) if not df.empty else 0,
        "pages": len(pages),
    }]))

    st.subheader("Findings")
    st.dataframe(df, use_container_width=True)

    # Save history
    log_audit(user=st.session_state.get("user",""), file=pdf.name, outcome=outcome, total_findings=total, pages=len(pages))

    # Excel export with file name + outcome + date
    today = datetime.now().strftime("%Y%m%d")
    base = os.path.splitext(pdf.name)[0]
    out_name = f"{base} - {outcome} - {today}.xlsx"

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df_out = df.copy()
        df_out.insert(0, "file", pdf.name)
        df_out.to_excel(xw, sheet_name="Findings", index=False)
    st.download_button("Download report (Excel)", data=buf.getvalue(), file_name=out_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.divider()
st.subheader("Audit history (latest 200)")
try:
    hist = pd.read_csv(os.path.join("history","audit_log.csv"))
    st.dataframe(hist.head(200), use_container_width=True)
except Exception:
    st.info("No history yet.")

st.divider()
st.subheader("Quick rules template")
st.download_button("Download quick_rules_template.csv", data=open("quick_rules_template.csv","rb").read(), file_name="quick_rules_template.csv")

st.subheader("Current suppressions")
st.code(json.dumps(read_suppressions(), indent=2))
