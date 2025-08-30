
import io
import os
import re
import json
import time
import fitz  # PyMuPDF
import yaml
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

APP_TITLE = "AI Design QA v6 (Revert - no MIMO)"
HISTORY_PATH = "history/history.csv"
DEFAULT_RULES_FILE = "rules_example.yaml"

# --------------------------- Rules I/O ---------------------------
def load_rules(path: str) -> dict:
    if not os.path.exists(path):
        return {
            "required_phrases": ["ALL DIMENSIONS IN MM"],
            "forbidden_pairs": [
                {"if_contains": "Brush", "must_not_contain": "Generator Power"},
                {"if_contains": "Generator Power", "must_not_contain": "Brush"},
            ],
            "allowlist": [],
            "suppress_patterns": [],
        }
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Ensure keys exist
    data.setdefault("required_phrases", [])
    data.setdefault("forbidden_pairs", [])
    data.setdefault("allowlist", [])
    data.setdefault("suppress_patterns", [])
    return data

def save_rules(path: str, rules: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(rules, f, sort_keys=False, allow_unicode=True)

# --------------------------- PDF helpers ---------------------------
def pdf_to_pages(file_bytes: bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        pages.append({"index": i+1, "text": text})
    doc.close()
    return pages

# --------------------------- Checks ---------------------------
def check_required_phrases(pages, required_phrases):
    findings = []
    for phrase in required_phrases:
        found = False
        for p in pages:
            if phrase.lower() in p["text"].lower():
                found = True
                break
        if not found:
            findings.append({
                "file": "",
                "page": None,
                "kind": "Checklist",
                "message": f"Missing required text: '{phrase}'",
                "boxes": ""
            })
    return findings

def check_forbidden_pairs(pages, pairs):
    findings = []
    for rule in pairs:
        a = rule.get("if_contains","")
        b = rule.get("must_not_contain","")
        if not a or not b:
            continue
        pages_with_a = [p for p in pages if a.lower() in p["text"].lower()]
        pages_with_b = [p for p in pages if b.lower() in p["text"].lower()]
        if pages_with_a and pages_with_b:
            pages_hit = sorted({p["index"] for p in pages_with_a + pages_with_b})
            for idx in pages_hit:
                findings.append({
                    "file": "",
                    "page": idx,
                    "kind": "Consistency",
                    "message": f"Forbidden together: '{a}' with '{b}'",
                    "boxes": ""
                })
    return findings

def check_spelling_like(pages, allowlist):
    # Very light-weight: flag words longer than 4 that are low similarity
    # to their own lowercase (simulating a typo scan) but allowlist disables.
    # Placeholder for your previous heavier spell logic.
    findings = []
    allow = set([w.lower() for w in allowlist or []])
    word_re = re.compile(r"[A-Za-z]{5,}")
    for p in pages:
        words = word_re.findall(p["text"])
        for w in set(words):
            lw = w.lower()
            if lw in allow:
                continue
            # Heuristic: weird case or repeated letters
            if re.search(r"(.)\1\1", w) or (w != w.lower() and w != w.upper() and fuzz.ratio(w, w.lower()) < 90):
                findings.append({
                    "file": "",
                    "page": p["index"],
                    "kind": "Spelling",
                    "message": f"Suspicious token: '{w}'",
                    "boxes": ""
                })
    return findings

def apply_suppression(findings, suppress_patterns):
    if not suppress_patterns:
        return findings
    kept = []
    for f in findings:
        msg = f.get("message","").lower()
        if any(pat.lower() in msg for pat in suppress_patterns):
            continue
        kept.append(f)
    return kept

# --------------------------- History ---------------------------
def append_history(row: dict):
    cols = ["timestamp_utc","user","stage","client","file","pages","used_ocr","total_findings","outcome"]
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    if not os.path.exists(HISTORY_PATH):
        pd.DataFrame(columns=cols).to_csv(HISTORY_PATH, index=False)
    df = pd.read_csv(HISTORY_PATH)
    # align columns
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df.loc[len(df)] = [row.get(c) for c in cols]
    df.to_csv(HISTORY_PATH, index=False)

def load_history(n=200):
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame(columns=["timestamp_utc","user","stage","client","file","pages","used_ocr","total_findings","outcome"])
    df = pd.read_csv(HISTORY_PATH)
    return df.tail(n).iloc[::-1]

# --------------------------- Excel export ---------------------------
def export_findings_xlsx(findings_df, original_name, status):
    date = time.strftime("%Y%m%d")
    base = os.path.splitext(original_name)[0]
    out_name = f"{base}__{status}__{date}.xlsx"
    with pd.ExcelWriter(out_name, engine="openpyxl") as writer:
        findings_df.to_excel(writer, index=False, sheet_name="Findings")
    return out_name

# --------------------------- UI ---------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Audit details")
    rules_file = st.text_input("Rules file", value=DEFAULT_RULES_FILE)
    user_name = st.text_input("Your name", value="qa_user")
    # Metadata (no MIMO here)
    client = st.selectbox("Client", ["BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"], index=0)
    project = st.text_input("Project")
    site_type = st.selectbox("Site Type", ["Greenfield","Rooftop","Streetworks"])
    vendor = st.selectbox("Proposed Vendor", ["Ericsson","Nokia"])
    cab_loc = st.selectbox("Proposed Cabinet Location", ["Indoor","Outdoor"])
    radio_loc = st.selectbox("Proposed Radio Location", ["High Level","Low Level","Indoor","Door"])
    site_address = st.text_area("Site Address", height=80)

    st.divider()
    st.caption("Rules editor")
    rules = load_rules(rules_file)
    if st.checkbox("Show / edit YAML", value=False):
        yaml_text = st.text_area("rules yaml", value=yaml.safe_dump(rules, sort_keys=False, allow_unicode=True), height=220)
        if st.button("Save YAML"):
            try:
                new_rules = yaml.safe_load(yaml_text) or {}
                save_rules(rules_file, new_rules)
                st.success("Saved rules.")
                rules = new_rules
            except Exception as e:
                st.error(f"YAML error: {e}")

st.subheader("1) Upload a single PDF")
pdf_file = st.file_uploader("PDF", type=["pdf"])

if pdf_file is not None:
    file_bytes = pdf_file.read()
    pages = pdf_to_pages(file_bytes)

    st.subheader("2) Run audit")
    if st.button("Audit now", use_container_width=True):
        with st.spinner("Auditing…"):
            all_findings = []
            # Checks
            all_findings += check_required_phrases(pages, rules.get("required_phrases", []))
            all_findings += check_forbidden_pairs(pages, rules.get("forbidden_pairs", []))
            all_findings += check_spelling_like(pages, rules.get("allowlist", []))
            # Attach file name
            for f in all_findings:
                f["file"] = pdf_file.name
            # Apply suppression from learning
            all_findings = apply_suppression(all_findings, rules.get("suppress_patterns", []))

        # Display summary
        counts = pd.Series([f["kind"] for f in all_findings]).value_counts().to_dict()
        cols = st.columns(6)
        cols[0].metric("Spelling", counts.get("Spelling", 0))
        cols[1].metric("Checklist", counts.get("Checklist", 0))
        cols[2].metric("Consistency", counts.get("Consistency", 0))
        cols[3].metric("Data", 0)
        cols[4].metric("Structural", 0)
        cols[5].metric("Electrical", 0)

        status = "Pass" if len(all_findings) == 0 else "Rejected"
        st.success("QA PASS – please continue with second check.") if status=="Pass" else st.error("REJECTED – errors found.")

        # Findings table with learning
        if all_findings:
            df = pd.DataFrame(all_findings, columns=["file","page","kind","message","boxes"])
            df["Not valid? (ignore next time)"] = False
            st.subheader("Findings")
            st.dataframe(df, use_container_width=True)

            # Apply learning
            if st.button("Apply learning (suppress selected messages)"):
                to_suppress = df.loc[df["Not valid? (ignore next time)"]==True, "message"].tolist()
                if to_suppress:
                    rules = load_rules(rules_file)
                    sup = rules.get("suppress_patterns", [])
                    for m in to_suppress:
                        frag = m.strip()
                        if frag and frag not in sup:
                            sup.append(frag)
                    rules["suppress_patterns"] = sup
                    save_rules(rules_file, rules)
                    st.success(f"Added {len(to_suppress)} patterns to suppress list in {rules_file}. Re-run audit to see effect.")
                else:
                    st.info("No rows selected.")

            # Export
            out_name = export_findings_xlsx(df.drop(columns=["Not valid? (ignore next time)"]), pdf_file.name, status)
            st.download_button("Download Excel report", data=open(out_name,"rb").read(), file_name=out_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # History log
        append_history({
            "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "user": user_name,
            "stage": "audit",
            "client": client,
            "file": pdf_file.name,
            "pages": len(pages),
            "used_ocr": 0,
            "total_findings": len(all_findings),
            "outcome": status.upper()
        })

st.subheader("Audit history (latest 200)")
st.dataframe(load_history(200), use_container_width=True)
