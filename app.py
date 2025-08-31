# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import fitz  # PyMuPDF
import io, os, base64, datetime, re
from spellchecker import SpellChecker
from typing import Dict, Any, List

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="AI Design Quality Auditor", layout="wide")

APP_PASSWORD   = "Seker123"      # entry password
RULES_PASSWORD = "vanB3lkum21"   # settings/training password

HISTORY_FILE = "audit_history.csv"
RULES_FILE   = "rules_example.yaml"
LOGO_FILE    = "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png"  # update if needed

SUPPLIERS = ["CEG","CTIL","Emfyser","Innov8","Invict","KTL Team (Internal)","Trylon"]
CLIENTS   = ["BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"]
PROJECTS  = ["RAN","Power Resilience","East Unwind","Beacon 4"]
VENDORS   = ["Ericsson","Nokia"]
DRAWING_TYPES = ["General Arrangement","Detailed Design"]
SITE_TYPES    = ["Greenfield","Rooftop","Streetworks"]
CABINET_LOCS  = ["Indoor","Outdoor"]
RADIO_LOCS    = ["Low Level","Midway","High Level","Unique Coverage"]

# a short, sane MIMO list to start (you can expand in Settings)
MIMO_DEFAULTS = [
    "18 @2x2",
    "18 @2x2; 26 @4x4",
    "18\\21 @2x2",
    "18\\21 @4x4; 70\\80 @2x2",
    "18\\21 @4x4; 70\\80 @2x2; 3500 @32x32"
]

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat()

def load_rules(path: str = RULES_FILE) -> dict:
    if not os.path.exists(path):
        return {"allowlist": [], "rules": []}
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
            if "allowlist" not in data: data["allowlist"] = []
            if "rules" not in data: data["rules"] = []
            return data
    except Exception:
        # corrupt YAML — don’t crash app
        return {"allowlist": [], "rules": []}

def save_rules(data: dict, path: str = RULES_FILE):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame()

def save_history(df: pd.DataFrame):
    df.to_csv(HISTORY_FILE, index=False)

def inject_logo(filepath: str):
    if os.path.exists(filepath):
        try:
            b64 = base64.b64encode(open(filepath, "rb").read()).decode()
            st.markdown(
                f"""
                <style>
                .app-logo {{
                    position: fixed;
                    top: 10px; left: 16px;
                    z-index: 9999;
                }}
                .app-logo img {{
                    height: 48px; width: auto;
                }}
                /* push content a bit so logo doesn't overlap */
                .block-container {{
                    padding-top: 70px !important;
                }}
                </style>
                <div class="app-logo"><img src="data:image/png;base64,{b64}"></div>
                """,
                unsafe_allow_html=True
            )
        except Exception:
            pass

def text_from_pdf(pdf_bytes: bytes) -> List[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return [doc[i].get_text("text") for i in range(len(doc))]

def annotate_pdf(pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    """
    Simple annotation: a red text box at a consistent position per page that has findings.
    (More accurate per-word marking can be added if needed.)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    by_page: Dict[int, List[str]] = {}
    for f in findings:
        p = max(1, int(f.get("page", 1)))
        by_page.setdefault(p, []).append(f.get("detail", ""))
    for p, rows in by_page.items():
        try:
            page = doc[p-1]
            txt = "\n".join(f"• {r}" for r in rows)
            rect = fitz.Rect(36, 36, page.rect.width - 36, 140)
            page.add_rect_annot(rect)
            page.insert_textbox(rect, txt, fontsize=8, color=(1, 0, 0))
        except Exception:
            continue
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

def make_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], src_pdf: str, status: str) -> bytes:
    df = pd.DataFrame(findings)
    with io.BytesIO() as buf:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            summary = pd.DataFrame([{
                **meta,
                "source_pdf": src_pdf,
                "status": status,
                "timestamp_utc": now_iso(),
                "total_findings": len(findings),
                "majors": int((df.severity == "major").sum()) if not df.empty else 0,
                "minors": int((df.severity == "minor").sum()) if not df.empty else 0,
            }])
            summary.to_excel(xw, sheet_name="Summary", index=False)
            (df if not df.empty else pd.DataFrame(columns=["page","rule","severity","detail"]))\
                .to_excel(xw, sheet_name="Findings", index=False)
        return buf.getvalue()

# ------------------------------------------------------------
# CHECKS
# ------------------------------------------------------------
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")

def normalize_tokens(text: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(text)]

def spelling_findings(pages: List[str], allow_words: set) -> List[Dict[str, Any]]:
    """Robust + fast spelling: use unknown() and safe correction lookup."""
    sp = SpellChecker()
    findings: List[Dict[str, Any]] = []
    for i, text in enumerate(pages, start=1):
        toks = normalize_tokens(text)
        if not toks:
            continue
        # remove allowlist and short tokens
        toks = [t for t in toks if len(t) >= 3 and t not in allow_words]
        try:
            unknown = sp.unknown(toks)
        except Exception:
            unknown = set()
        for bad in sorted(unknown):
            suggestion = None
            try:
                suggestion = sp.correction(bad)
            except Exception:
                suggestion = None
            findings.append({
                "page": i,
                "rule": "spelling",
                "severity": "minor",
                "detail": f"Unknown word '{bad}'" + (f", suggestion '{suggestion}'" if suggestion else "")
            })
    return findings

def run_checks(pages: List[str], meta: Dict[str, Any], rules: Dict[str, Any], do_spelling: bool, allow_words: set) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    # Spelling
    if do_spelling:
        results.extend(spelling_findings(pages, allow_words))
    # Simple site address consistency (reject if title doesn't contain address unless ", 0 ," present)
    title_text = pages[0] if pages else ""
    addr = (meta.get("site_address") or "").strip()
    if addr and ", 0 ," not in addr:
        if addr.lower() not in title_text.lower():
            results.append({
                "page": 1,
                "rule": "site_address_title_match",
                "severity": "major",
                "detail": f"Site address not found in PDF title/page 1: '{addr}'"
            })
    # Hide MIMO when Power Resilience (no failure, just skip)
    # (your deep content rules can be added here)
    return results

# ------------------------------------------------------------
# GATE
# ------------------------------------------------------------
def gate():
    st.session_state.setdefault("auth_ok", False)
    if not st.session_state["auth_ok"]:
        st.title("AI Design Quality Auditor")
        pw = st.text_input("Enter access password", type="password")
        if st.button("Enter"):
            if pw == APP_PASSWORD:
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("Incorrect password")
                st.stop()
        else:
            st.stop()

# ------------------------------------------------------------
# MAIN UI
# ------------------------------------------------------------
def main():
    inject_logo(LOGO_FILE)
    gate()

    # in-memory session persistence so downloads don't clear results
    st.session_state.setdefault("last_findings", None)
    st.session_state.setdefault("last_excel_bytes", None)
    st.session_state.setdefault("last_pdf_bytes", None)
    st.session_state.setdefault("last_meta", None)
    st.session_state.setdefault("last_status", None)
    st.session_state.setdefault("last_src_pdf", None)

    rules = load_rules()
    allow_words = set([w.lower() for w in rules.get("allowlist", [])])
    history = load_history()

    tabs = st.tabs(["Audit", "Training", "Analytics", "Settings"])

    # ---------------------- AUDIT ----------------------
    with tabs[0]:
        st.subheader("Run Audit")

        colA, colB, colC = st.columns([1.2,1.2,1])
        with colA:
            supplier = st.selectbox("Supplier", SUPPLIERS, key="supplier")
            client   = st.selectbox("Client", CLIENTS, key="client")
            project  = st.selectbox("Project", PROJECTS, key="project")
            drawing  = st.selectbox("Drawing Type", DRAWING_TYPES, key="drawing_type")
        with colB:
            vendor   = st.selectbox("Vendor", VENDORS, key="vendor")
            site_tp  = st.selectbox("Site Type", SITE_TYPES, key="site_type")
            cab_loc  = st.selectbox("Cabinet Location", CABINET_LOCS, key="cab_loc")
            radio_loc= st.selectbox("Radio Location", RADIO_LOCS, key="radio_loc")
        with colC:
            sectors  = st.selectbox("Qty of Sectors", [1,2,3,4,5,6], key="sectors")
            mimo_all_same = st.checkbox("Use same MIMO for all sectors", value=True)
            mimo_list = st.session_state.get("mimo_list", MIMO_DEFAULTS)
            if project == "Power Resilience":
                st.caption("MIMO not required for Power Resilience.")
                mimo_configs = []
            else:
                if mimo_all_same:
                    mimo_s1 = st.selectbox("MIMO (S1 = all)", mimo_list, key="mimo_s1_all")
                    mimo_configs = [mimo_s1] * sectors
                else:
                    mimo_configs = []
                    for s in range(1, sectors+1):
                        mimo_configs.append(st.selectbox(f"MIMO S{s}", mimo_list, key=f"mimo_s{s}"))

        site_address = st.text_input("Site Address (title must contain this, unless contains ', 0 ,')", key="site_address")
        do_spelling  = st.checkbox("Enable spelling check", value=True, help="Uncheck for performance on huge PDFs")

        pdf = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")

        c1, c2, c3 = st.columns([1,1,1])
        run = c1.button("Run Audit", type="primary")
        clear = c2.button("Clear Results")
        exclude_analytics = c3.checkbox("Exclude this run from analytics", value=False)

        if clear:
            st.session_state.update({
                "last_findings": None,
                "last_excel_bytes": None,
                "last_pdf_bytes": None,
                "last_meta": None,
                "last_status": None,
                "last_src_pdf": None
            })
            st.success("Cleared.")

        if run and pdf:
            raw = pdf.read()
            pages = text_from_pdf(raw)
            meta = {
                "supplier": supplier,
                "client": client,
                "project": project,
                "drawing_type": drawing,
                "vendor": vendor,
                "site_type": site_tp,
                "cabinet_location": cab_loc,
                "radio_location": radio_loc,
                "sectors": sectors,
                "mimo_configs": mimo_configs,
                "site_address": site_address
            }
            findings = run_checks(pages, meta, rules, do_spelling, allow_words)
            status = "Pass" if len(findings) == 0 else "Rejected"

            excel_bytes = make_excel(findings, meta, pdf.name, status)
            pdf_bytes   = annotate_pdf(raw, findings)

            # persist
            st.session_state["last_findings"] = findings
            st.session_state["last_excel_bytes"] = excel_bytes
            st.session_state["last_pdf_bytes"] = pdf_bytes
            st.session_state["last_meta"] = meta
            st.session_state["last_status"] = status
            st.session_state["last_src_pdf"] = pdf.name

            # history
            rec = {
                "timestamp_utc": now_iso(),
                **meta,
                "status": status,
                "pdf_name": pdf.name,
            }
            if not exclude_analytics:
                history = pd.concat([history, pd.DataFrame([rec])], ignore_index=True)
                save_history(history)

        # show persisted results (don’t disappear after a download)
        if st.session_state["last_findings"] is not None:
            st.markdown(f"### Result: **{st.session_state['last_status']}**")
            st.dataframe(pd.DataFrame(st.session_state["last_findings"]), use_container_width=True)
            d1, d2 = st.columns(2)
            d1.download_button(
                "⬇️ Download Excel",
                st.session_state["last_excel_bytes"],
                file_name=f"{st.session_state['last_src_pdf']}_{st.session_state['last_status']}_{now_iso()}.xlsx"
            )
            d2.download_button(
                "⬇️ Download Annotated PDF",
                st.session_state["last_pdf_bytes"],
                file_name=f"{st.session_state['last_src_pdf']}_{st.session_state['last_status']}_{now_iso()}.pdf"
            )

    # ---------------------- TRAINING ----------------------
    with tabs[1]:
        st.subheader("Training (learn from you)")
        pw = st.text_input("Rules password", type="password", key="train_pw")
        if pw != RULES_PASSWORD:
            st.info("Enter the rules password to add allowlist words or rules.")
        else:
            st.success("Rules access granted")

            st.markdown("**Bulk training file (CSV/XLSX)** — include columns like `word`, `valid` or `rule_text`, `category`, `valid`.")
            up_train = st.file_uploader("Upload training file", type=["csv","xlsx"], key="train_upload")
            if up_train:
                try:
                    df_t = pd.read_excel(up_train) if up_train.name.endswith(".xlsx") else pd.read_csv(up_train)
                    st.dataframe(df_t.head(), use_container_width=True)
                    if "word" in df_t.columns and "valid" in df_t.columns:
                        added = 0
                        for _, r in df_t.iterrows():
                            if str(r["valid"]).strip().lower() == "true":
                                w = str(r["word"]).strip().lower()
                                if w and w not in rules["allowlist"]:
                                    rules["allowlist"].append(w)
                                    added += 1
                        save_rules(rules)
                        st.success(f"Added {added} allowed words from file.")
                    if "rule_text" in df_t.columns and "valid" in df_t.columns:
                        added = 0
                        for _, r in df_t.iterrows():
                            if str(r["valid"]).strip().lower() == "true":
                                txt = str(r["rule_text"]).strip()
                                cat = str(r.get("category", "content")).strip()
                                rules["rules"].append({"rule": txt, "category": cat})
                                added += 1
                        save_rules(rules)
                        st.success(f"Added {added} rules from file.")
                except Exception as e:
                    st.error(f"Could not process training file: {e}")

            st.markdown("---")
            st.markdown("**Quick adds**")
            c1, c2 = st.columns(2)
            with c1:
                add_word = st.text_input("Add to spelling allowlist (word)")
                if st.button("Add allowed word"):
                    if pw == RULES_PASSWORD and add_word:
                        w = add_word.strip().lower()
                        if w and w not in rules["allowlist"]:
                            rules["allowlist"].append(w)
                            save_rules(rules)
                            st.success(f"Added '{w}' to allowlist")
            with c2:
                add_rule = st.text_input("Add simple rule text")
                cat = st.selectbox("Category", ["content","spelling","format"])
                if st.button("Add rule"):
                    if pw == RULES_PASSWORD and add_rule:
                        rules["rules"].append({"rule": add_rule.strip(), "category": cat})
                        save_rules(rules)
                        st.success("Rule added")

    # ---------------------- ANALYTICS ----------------------
    with tabs[2]:
        st.subheader("Analytics")
        dfh = load_history()
        if dfh.empty:
            st.info("No history yet.")
        else:
            c1, c2, c3 = st.columns(3)
            f_client = c1.selectbox("Client", ["All"] + CLIENTS)
            f_project= c2.selectbox("Project", ["All"] + PROJECTS)
            f_supplier=c3.selectbox("Supplier", ["All"] + SUPPLIERS)

            show = dfh.copy()
            if f_client != "All":
                show = show[show["client"] == f_client]
            if f_project != "All":
                show = show[show["project"] == f_project]
            if f_supplier != "All":
                show = show[show["supplier"] == f_supplier]

            # basic KPIs
            total = len(show)
            passes = int((show["status"]=="Pass").sum()) if "status" in show.columns else 0
            rej    = total - passes
            rft    = (passes/total*100.0) if total else 0.0

            k1,k2,k3 = st.columns(3)
            k1.metric("Total Audits", total)
            k2.metric("Passes", passes)
            k3.metric("Right First Time %", f"{rft:.1f}%")

            st.dataframe(show, use_container_width=True)

    # ---------------------- SETTINGS ----------------------
    with tabs[3]:
        st.subheader("Settings")
        pw2 = st.text_input("Rules password", type="password", key="settings_pw")
        if pw2 != RULES_PASSWORD:
            st.info("Enter the rules password to edit lists.")
        else:
            st.success("Rules access granted")

            # manage MIMO picklist used in Audit tab
            mimo_text = st.text_area("MIMO picklist (one per line)", "\n".join(st.session_state.get("mimo_list", MIMO_DEFAULTS)), height=160)
            if st.button("Save MIMO list"):
                st.session_state["mimo_list"] = [x.strip() for x in mimo_text.splitlines() if x.strip()]
                st.success("MIMO list updated.")

            # show/edit YAML (raw)
            st.markdown("**Raw YAML rules**")
            raw = yaml.safe_dump(load_rules(), sort_keys=False)
            raw_edit = st.text_area("rules_example.yaml", raw, height=240)
            if st.button("Save YAML"):
                try:
                    data = yaml.safe_load(raw_edit) or {}
                    if "allowlist" not in data: data["allowlist"] = []
                    if "rules" not in data: data["rules"] = []
                    save_rules(data)
                    st.success("YAML saved.")
                except Exception as e:
                    st.error(f"Invalid YAML: {e}")

if __name__ == "__main__":
    main()
