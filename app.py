
import io
import os
import re
import fitz  # PyMuPDF
import yaml
import base64
import pandas as pd
from datetime import datetime
import streamlit as st

APP_TITLE = "AI Design QA v7"
HISTORY_DIR = "history"
HISTORY_CSV = os.path.join(HISTORY_DIR, "history.csv")

DEFAULT_RULES_FILE = "rules_example.yaml"

# -------------------------- Utils --------------------------

def load_rules(file_or_bytes: bytes | None):
    """Load YAML rules either from uploaded file or from default file on disk."""
    if file_or_bytes is None:
        path = DEFAULT_RULES_FILE if os.path.exists(DEFAULT_RULES_FILE) else None
        if path is None:
            return {"required_phrases": [], "forbidden_pairs": [], "suppress_patterns": [], "meta": {}}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    else:
        return yaml.safe_load(file_or_bytes) or {}

def save_rules(rules: dict, dest_path: str = DEFAULT_RULES_FILE):
    with open(dest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(rules, f, sort_keys=False, allow_unicode=True)

def extract_text_from_pdf(file_bytes: bytes) -> list[str]:
    """Return a list of page texts using PyMuPDF."""
    texts = []
    with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
        for page in doc:
            texts.append(page.get_text("text") or "")
    return texts

def pass_fail_text(success: bool) -> str:
    return "PASS" if success else "REJECTED"

def append_history(row: dict):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    if os.path.exists(HISTORY_CSV):
        df = pd.read_csv(HISTORY_CSV)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(HISTORY_CSV, index=False)

def make_download_link(buffer: bytes, filename: str, label: str = "Download"):
    b64 = base64.b64encode(buffer).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'

def normalize(s: str) -> str:
    return (s or "").strip()

# -------------------------- Rule Checks --------------------------

def check_required_phrases(text_pages: list[str], rules: dict) -> list[dict]:
    findings = []
    required = rules.get("required_phrases", [])
    for item in required:
        if isinstance(item, dict):
            phrase = normalize(item.get("text"))
            where = item.get("where", "anywhere")
        else:
            phrase = normalize(str(item))
            where = "anywhere"

        if not phrase:
            continue

        present = False
        pages_hit = []
        for i, page_text in enumerate(text_pages, start=1):
            if phrase.lower() in page_text.lower():
                present = True
                pages_hit.append(i)
        if not present:
            findings.append({
                "kind": "Checklist",
                "page": "",
                "message": f"Missing required text: '{phrase}'"
            })
    return findings

def check_forbidden_pairs(text_pages: list[str], rules: dict) -> list[dict]:
    findings = []
    pairs = rules.get("forbidden_pairs", [])
    doc_text = "\n".join(text_pages).lower()

    for pair in pairs:
        left = [s.lower() for s in pair.get("left", [])]
        right = [s.lower() for s in pair.get("right", [])]
        if not left or not right:
            continue

        left_hit = any(w in doc_text for w in left)
        right_hit = any(w in doc_text for w in right)

        if left_hit and right_hit:
            findings.append({
                "kind": "Consistency",
                "page": "",
                "message": f"Forbidden combination detected: left={pair.get('left')} right={pair.get('right')}"
            })
    return findings

def apply_suppressions(findings: list[dict], rules: dict) -> list[dict]:
    """Remove findings that match any substring in suppress_patterns."""
    suppress = [s.lower() for s in rules.get("suppress_patterns", [])]
    if not suppress:
        return findings
    kept = []
    for f in findings:
        msg = f.get("message", "").lower()
        if any(p in msg for p in suppress):
            continue
        kept.append(f)
    return kept

# -------------------------- UI --------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

left, right = st.columns([1,2])

with left:
    st.subheader("Audit setup")

    # Load rules
    rules_file = st.file_uploader("Load rules YAML (optional)", type=["yaml", "yml"])
    rules = load_rules(rules_file.read() if rules_file else None)
    if "suppress_patterns" not in rules:
        rules["suppress_patterns"] = []

    # Meta picklists
    meta = rules.get("meta", {})
    clients = meta.get("clients", [])
    projects = meta.get("projects", [])
    site_types = meta.get("site_types", [])
    vendors = meta.get("vendors", [])
    cabinet_locations = meta.get("cabinet_locations", [])
    radio_locations = meta.get("radio_locations", [])

    client = st.selectbox("Client", options=clients or ["--"], index=0)
    project = st.selectbox("Project", options=projects or ["--"], index=0)
    site_type = st.selectbox("Site Type", options=site_types or ["--"], index=0)
    vendor = st.selectbox("Proposed Vendor", options=vendors or ["--"], index=0)
    cabinet_loc = st.selectbox("Proposed Cabinet Location", options=cabinet_locations or ["--"], index=0)
    radio_loc = st.selectbox("Proposed Radio Location", options=radio_locations or ["--"], index=0)

    st.markdown("**Proposed MIMO**")
    same_across = st.checkbox("Same across S1/S2/S3", value=True)
    mimo_options = [
        "18\\21 @2x2",
        "18\\21\\26 @4x4",
        "18\\21\\26 @4x4; 3500 @8x8",
        "18\\21\\26 @4x4; 70\\80 @2x4; 3500 @32x32",
        "18 @2x2",
    ]
    s1 = st.selectbox("Proposed Mimo S1", options=mimo_options, index=0)
    if same_across:
        s2 = s1
        s3 = s1
        st.caption(f"S2 and S3 set to S1 ({s1})")
    else:
        s2 = st.selectbox("Proposed Mimo S2", options=mimo_options, index=0)
        s3 = st.selectbox("Proposed Mimo S3", options=mimo_options, index=0)

    site_address = st.text_input("Site Address", "")

    pdf_file = st.file_uploader("Upload a single PDF", type=["pdf"])

    run = st.button("Run Audit", type="primary")

with right:
    st.subheader("Results")

    if run and pdf_file is not None:
        pdf_bytes = pdf_file.read()
        text_pages = extract_text_from_pdf(pdf_bytes)

        # Rule checks
        findings = []
        findings += check_required_phrases(text_pages, rules)
        findings += check_forbidden_pairs(text_pages, rules)

        # Apply suppressions (learned "not valid")
        findings = apply_suppressions(findings, rules)

        # Build results table
        df = pd.DataFrame(findings) if findings else pd.DataFrame(columns=["kind", "page", "message"])
        total = len(df)
        status = pass_fail_text(total == 0)

        st.metric("Outcome", status)
        st.metric("Findings", total)

        # Findings table with "Not valid?" selection
        if total > 0:
            df_show = df.copy()
            df_show["Not valid? (ignore next time)"] = False
            edited = st.data_editor(df_show, num_rows="dynamic", use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Apply learning (add selected to suppressions)"):
                    selected = edited[edited["Not valid? (ignore next time)"] == True]
                    if not selected.empty:
                        new_patterns = list({m for m in selected["message"].tolist()})
                        rules["suppress_patterns"] = list({*rules.get("suppress_patterns", []), *new_patterns})
                        save_rules(rules)  # write back to rules_example.yaml
                        st.success(f"Added {len(new_patterns)} message pattern(s) to suppressions and saved rules.")
                    else:
                        st.info("No rows were marked as Not valid.")

            with col2:
                # Export Excel
                today = datetime.utcnow().strftime("%Y%m%d")
                base_name = os.path.splitext(pdf_file.name)[0]
                out_name = f"{base_name}__{status}__{today}.xlsx"
                with pd.ExcelWriter(out_name, engine="openpyxl") as xw:
                    # Summary sheet
                    summary = pd.DataFrame([{
                        "file": pdf_file.name,
                        "status": status,
                        "client": client,
                        "project": project,
                        "site_type": site_type,
                        "vendor": vendor,
                        "cabinet_location": cabinet_loc,
                        "radio_location": radio_loc,
                        "mimo_s1": s1, "mimo_s2": s2, "mimo_s3": s3,
                        "address": site_address,
                        "total_findings": total,
                        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds")
                    }])
                    summary.to_excel(xw, index=False, sheet_name="Summary")
                    df.to_excel(xw, index=False, sheet_name="Findings")
                with open(out_name, "rb") as f:
                    st.markdown(make_download_link(f.read(), out_name, "Download Excel report"), unsafe_allow_html=True)

        else:
            st.success("No findings. Document PASSED. You can still export an empty report below.")
            # Even with no findings, allow export
            today = datetime.utcnow().strftime("%Y%m%d")
            base_name = os.path.splitext(pdf_file.name)[0]
            out_name = f"{base_name}__{status}__{today}.xlsx"
            with pd.ExcelWriter(out_name, engine="openpyxl") as xw:
                summary = pd.DataFrame([{
                    "file": pdf_file.name,
                    "status": status,
                    "client": client,
                    "project": project,
                    "site_type": site_type,
                    "vendor": vendor,
                    "cabinet_location": cabinet_loc,
                    "radio_location": radio_loc,
                    "mimo_s1": s1, "mimo_s2": s2, "mimo_s3": s3,
                    "address": site_address,
                    "total_findings": 0,
                    "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds")
                }])
                empty = pd.DataFrame(columns=["kind", "page", "message"])
                summary.to_excel(xw, index=False, sheet_name="Summary")
                empty.to_excel(xw, index=False, sheet_name="Findings")
            with open(out_name, "rb") as f:
                st.markdown(make_download_link(f.read(), out_name, "Download Excel report"), unsafe_allow_html=True)

        # Append to history
        append_history({
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "user": st.experimental_user.email if hasattr(st.experimental_user, "email") else "",
            "file": pdf_file.name,
            "client": client,
            "project": project,
            "site_type": site_type,
            "vendor": vendor,
            "mimo_s1": s1, "mimo_s2": s2, "mimo_s3": s3,
            "status": status,
            "total_findings": total,
        })

    with st.expander("Rules (view / edit raw YAML)"):
        st.caption("These are the currently loaded rules. When you click **Apply learning** above, "
                   "messages you marked as 'Not valid' will be added to `suppress_patterns` below, and saved.")
        rules_text = st.text_area("YAML", value=yaml.safe_dump(rules, sort_keys=False, allow_unicode=True), height=240)
        if st.button("Save YAML"):
            try:
                new_rules = yaml.safe_load(rules_text) or {}
                save_rules(new_rules)
                st.success("Rules saved to rules_example.yaml")
            except Exception as e:
                st.error(f"Failed to parse YAML: {e}")

    with st.expander("History (latest 200)"):
        if os.path.exists(HISTORY_CSV):
            dfh = pd.read_csv(HISTORY_CSV)
            dfh = dfh.sort_values("timestamp_utc", ascending=False).head(200)
            st.dataframe(dfh, use_container_width=True, height=300)
        else:
            st.info("No history yet.")
