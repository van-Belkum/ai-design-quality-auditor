import streamlit as st
import pandas as pd
import yaml
import io
import os
import base64
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from rapidfuzz import fuzz

# ==============================
# CONFIG / CONSTANTS
# ==============================
ENTRY_PASSWORD = "Seker123"
ADMIN_RULES_PASSWORD = "vanB3lkum21"

RULES_FILE = "rules_example.yaml"
HISTORY_DIR = "history"
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.csv")
ARTIFACTS_DIR = os.path.join(HISTORY_DIR, "artifacts")

LOGO_PATH = "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png"

# Dropdowns (Supplier used only for analytics, not for rule logic)
SUPPLIERS = ["CEG","CTIL","Emfyser","Innov8","Invict","KTL Team (Internal)","Trylon"]
DRAWING_TYPES = ["General Arrangement", "Detailed Design"]
CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
CAB_LOCATIONS = ["Indoor", "Outdoor"]
RADIO_LOCATIONS = ["Low Level", "Midway", "High Level", "Unique Coverage"]

# Ordered, deduped list of MIMO configs (as provided)
_MIMO_RAW = [
    "18 @2x2","18 @2x2; 26 @4x4","18 @2x2; 70\\80 @2x2","18 @2x2; 80 @2x2",
    "18\\21 @2x2","18\\21 @2x2; 26 @4x4","18\\21 @2x2; 3500 @32x32",
    "18\\21 @2x2; 70\\80 @2x2","18\\21 @2x2; 80 @2x2",
    "18\\21 @4x4","18\\21 @4x4; 3500 @32x32","18\\21 @4x4; 70 @2x4",
    "18\\21 @4x4; 70\\80 @2x2","18\\21 @4x4; 70\\80 @2x2; 3500 @32x32",
    "18\\21 @4x4; 70\\80 @2x4","18\\21 @4x4; 70\\80 @2x4; 3500 @32x32",
    "18\\21 @4x4; 70\\80 @2x4; 3500 @8x8","18\\21@4x4; 70\\80 @2x2",
    "18\\21@4x4; 70\\80 @2x4","18\\21\\26 @2x2","18\\21\\26 @2x2; 3500 @32x32",
    "18\\21\\26 @2x2; 3500 @8X8","18\\21\\26 @2x2; 70\\80 @2x2",
    "18\\21\\26 @2x2; 70\\80 @2x2; 3500 @32x32",
    "18\\21\\26 @2x2; 70\\80 @2x2; 3500 @8x8",
    "18\\21\\26 @2x2; 70\\80 @2x4; 3500 @32x32","18\\21\\26 @2x2; 80 @2x2",
    "18\\21\\26 @2x2; 80 @2x2; 3500 @8x8","18\\21\\26 @4x4",
    "18\\21\\26 @4x4; 3500 @32x32","18\\21\\26 @4x4; 3500 @8x8",
    "18\\21\\26 @4x4; 70 @2x4; 3500 @8x8","18\\21\\26 @4x4; 70\\80 @2x2",
    "18\\21\\26 @4x4; 70\\80 @2x2; 3500 @32x32",
    "18\\21\\26 @4x4; 70\\80 @2x2; 3500 @8x8","18\\21\\26 @4x4; 70\\80 @2x4",
    "18\\21\\26 @4x4; 70\\80 @2x4; 3500 @32x32",
    "18\\21\\26 @4x4; 70\\80 @2x4; 3500 @8x8","18\\21\\26 @4x4; 80 @2x2",
    "18\\21\\26 @4x4; 80 @2x2; 3500 @32x32","18\\21\\26 @4x4; 80 @2x4",
    "18\\21\\26 @4x4; 80 @2x4; 3500 @8x8","18\\26 @2x2",
    "18\\26 @4x4; 21 @2x2; 80 @2x2","(blank)"
]
MIMO_CONFIGS = list(dict.fromkeys(_MIMO_RAW))  # dedupe preserve order

DEFAULT_ALLOWLIST = ["the","and","to","for","with","of","in","on","by","from"]

# ==============================
# FILE / STATE UTILS
# ==============================
def ensure_dirs():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()

def safe_read_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception:
        # minimal valid structure
        return {"allowlist": DEFAULT_ALLOWLIST, "checklist": [], "ignore_rules": []}

def safe_write_yaml(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def load_history() -> pd.DataFrame:
    ensure_dirs()
    if not os.path.exists(HISTORY_FILE) or os.path.getsize(HISTORY_FILE) == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_FILE)
    except Exception:
        # corrupted or partially written file—back it up and start fresh
        backup = HISTORY_FILE + f".backup_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        try:
            os.replace(HISTORY_FILE, backup)
        except Exception:
            pass
        return pd.DataFrame()

    # Normalize columns we'd expect
    if "timestamp_utc" in df.columns:
        try:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        except Exception:
            df["timestamp_utc"] = pd.NaT
    if "exclude" not in df.columns:
        df["exclude"] = False
    if "artifacts_zip" not in df.columns:
        df["artifacts_zip"] = ""
    return df

def save_history(df: pd.DataFrame) -> None:
    ensure_dirs()
    # Convert timestamp back to ISO string for storage
    if "timestamp_utc" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp_utc"]):
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S")
    df.to_csv(HISTORY_FILE, index=False)

# ==============================
# TEXT / PDF / ANNOTATION
# ==============================
def extract_pages(pdf_bytes: bytes) -> List[str]:
    """Try text; if blank, OCR image of page."""
    texts: List[str] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i in range(len(doc)):
        page = doc[i]
        t = page.get_text("text")
        if t and t.strip():
            texts.append(t)
            continue
        # OCR fallback
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr = pytesseract.image_to_string(img)
            texts.append(ocr or "")
        except Exception:
            texts.append("")
    return texts

def annotate_pdf(original_pdf: bytes, hits: List[Dict[str, Any]]) -> bytes:
    """
    Draws red boxes around best-effort matched strings. 
    If finding has 'needle' we search that; else we derive a short needle from its message.
    """
    doc = fitz.open("pdf", original_pdf)
    for f in hits:
        p = f.get("page_num")
        if p is None or p < 0 or p >= len(doc):
            continue
        page = doc[p]
        needle = (f.get("needle") or "").strip()
        if not needle:
            # try to guess a short term to search
            msg = f.get("message","")
            # take stuff between quotes or last token
            if '"' in msg:
                parts = [x for x in msg.split('"') if x.strip()]
                needle = parts[0][:40] if parts else msg[:30]
            else:
                needle = msg.split()[-1][:30] if msg else ""

        if needle:
            rects = page.search_for(needle, quads=False)
            for r in rects:
                annot = page.add_rect_annot(r)
                annot.set_colors(stroke=(1,0,0))
                annot.set_border(width=1)
                annot.update()

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

# ==============================
# RULES / CHECKS
# ==============================
def rules_applicable(rule: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    """Match by metadata filters; missing filter = wildcard."""
    filt = rule.get("filters", {})
    def ok(field, value):
        want = filt.get(field)
        return (not want) or (str(value).strip() in (want if isinstance(want, list) else [want]))
    return all([
        ok("client", meta["client"]),
        ok("project", meta["project"]),
        ok("vendor", meta["vendor"]),
        ok("site_type", meta["site_type"]),
        ok("drawing_type", meta["drawing_type"]),
        # supplier intentionally NOT used for rule logic
    ])

def run_rules(pages: List[str], rules: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for rule in rules:
        if rule.get("name") in (meta.get("ignore_rules", []) or []):
            continue
        if not rules_applicable(rule, meta):
            continue

        rname = rule.get("name","unnamed")
        sev = rule.get("severity","minor")
        musts: List[str] = rule.get("must_contain", []) or []
        must_not: List[str] = rule.get("must_not_contain", []) or []
        pairs: List[Dict[str, str]] = rule.get("forbidden_pairs", []) or []

        # must contain
        for i, txt in enumerate(pages):
            for token in musts:
                if token and token not in txt:
                    findings.append({
                        "rule_id": rname, "severity": sev,
                        "message": f'Missing required text "{token}"',
                        "page_num": i, "needle": token
                    })

        # must not contain
        for i, txt in enumerate(pages):
            for token in must_not:
                if token and token in txt:
                    findings.append({
                        "rule_id": rname, "severity": sev,
                        "message": f'Forbidden text present "{token}"',
                        "page_num": i, "needle": token
                    })

        # forbidden pairs: if A appears, B must not
        for i, txt in enumerate(pages):
            for pair in pairs:
                a = (pair.get("if_contains") or "").strip()
                b = (pair.get("must_not_include") or "").strip()
                if a and b and (a in txt) and (b in txt):
                    findings.append({
                        "rule_id": rname, "severity": sev,
                        "message": f'Pair violation: has "{a}" and "{b}"',
                        "page_num": i, "needle": b
                    })

    return findings

def spelling_checks(pages: List[str], allowlist: List[str]) -> List[Dict[str, Any]]:
    """Very light spell-like check: words not in allowlist and >2 alpha chars."""
    allow = {w.lower() for w in allowlist}
    out: List[Dict[str, Any]] = []
    for i, txt in enumerate(pages):
        for w in txt.split():
            ww = "".join(ch for ch in w if ch.isalpha())
            if len(ww) > 2 and ww.lower() not in allow:
                out.append({
                    "rule_id": "spelling",
                    "severity": "minor",
                    "message": f"Unknown word: {ww}",
                    "page_num": i,
                    "needle": ww
                })
    return out

# ==============================
# REPORTS
# ==============================
def make_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], original_name: str, status: str) -> bytes:
    df_find = pd.DataFrame(findings) if findings else pd.DataFrame(columns=["rule_id","severity","message","page_num","needle"])
    df_meta = pd.DataFrame([{
        **meta,
        "original_file": original_name,
        "status": status,
        "generated_utc": now_iso(),
    }])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df_find.to_excel(xw, index=False, sheet_name="Findings")
        df_meta.to_excel(xw, index=False, sheet_name="Metadata")
    buf.seek(0)
    return buf.read()

def make_zip(excel_bytes: bytes, pdf_bytes: bytes, base_name: str) -> bytes:
    import zipfile
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{base_name}.xlsx", excel_bytes)
        z.writestr(f"{base_name}.pdf", pdf_bytes)
    bio.seek(0)
    return bio.read()

# ==============================
# AUTH HELPERS
# ==============================
def gate_password() -> bool:
    st.session_state.setdefault("entry_ok", False)
    if st.session_state["entry_ok"]:
        return True
    pw = st.text_input("Enter access password", type="password")
    if st.button("Unlock"):
        if (pw or "").strip() == ENTRY_PASSWORD:
            st.session_state["entry_ok"] = True
            st.success("Unlocked.")
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

def rules_admin_unlocked() -> bool:
    st.session_state.setdefault("rules_ok", False)
    if st.session_state["rules_ok"]:
        return True
    with st.expander("🔐 Admin unlock (Rules Editor)", expanded=False):
        pw = st.text_input("Admin password", type="password", key="rules_pw")
        if st.button("Unlock rules"):
            if (pw or "").strip() == ADMIN_RULES_PASSWORD:
                st.session_state["rules_ok"] = True
                st.success("Rules editor unlocked.")
                st.rerun()
            else:
                st.error("Incorrect admin password.")
    return False

# ==============================
# UI
# ==============================
def draw_logo_top_left():
    try:
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style="position:relative;height:0;">
              <img src="data:image/png;base64,{b64}" style="position:absolute;top:10px;left:12px;width:220px;max-width:30%;z-index:999;border-radius:6px;" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

def metadata_block() -> Dict[str, Any]:
    st.subheader("Audit Metadata (all required)")
    c1, c2, c3, c4 = st.columns(4)
    supplier = c1.selectbox("Supplier", SUPPLIERS)
    drawing_type = c2.selectbox("Drawing Type", DRAWING_TYPES)
    client = c3.selectbox("Client", CLIENTS)
    project = c4.selectbox("Project", PROJECTS)
    site_type = c1.selectbox("Site Type", SITE_TYPES)
    vendor = c2.selectbox("Proposed Vendor", VENDORS)
    cab_loc = c3.selectbox("Proposed Cabinet Location", CAB_LOCATIONS)
    radio_loc = c4.selectbox("Proposed Radio Location", RADIO_LOCATIONS)
    qty = c1.selectbox("Quantity of Sectors", [1,2,3,4,5,6])
    site_address = c2.text_input("Site Address")

    mimo_block_needed = (project != "Power Resilience")
    sector_cfg = {}
    if mimo_block_needed:
        st.markdown("#### Proposed MIMO Config")
        same_all = st.checkbox("Use S1 for all sectors", value=True)
        for i in range(1, qty+1):
            if same_all and i > 1:
                sector_cfg[f"S{i}"] = sector_cfg["S1"]
                st.selectbox(f"MIMO S{i}", MIMO_CONFIGS, key=f"mimo_{i}", index=MIMO_CONFIGS.index(sector_cfg["S1"]), disabled=True)
            else:
                sector_cfg[f"S{i}"] = st.selectbox(f"MIMO S{i}", MIMO_CONFIGS, key=f"mimo_{i}")
    else:
        st.info("Project is Power Resilience → MIMO config optional/hidden.")
        sector_cfg = {f"S{i}": "(blank)" for i in range(1, qty+1)}

    meta = dict(
        supplier=supplier, drawing_type=drawing_type, client=client, project=project,
        site_type=site_type, vendor=vendor, cab_loc=cab_loc, radio_loc=radio_loc,
        qty=qty, site_address=site_address, sector_cfg=sector_cfg
    )
    return meta

def validate_meta(meta: Dict[str, Any]) -> Optional[str]:
    required_keys = ["supplier","drawing_type","client","project","site_type",
                     "vendor","cab_loc","radio_loc","qty","site_address"]
    for k in required_keys:
        v = meta.get(k, "")
        if (isinstance(v, str) and not v.strip()) or (v is None):
            return f"Missing required field: {k}"
    return None

# ==============================
# TRAINING APPLY
# ==============================
def apply_feedback_to_rules(feedback_df: pd.DataFrame, rules_blob: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expect columns like: rule_id, decision (Valid/Not Valid), token(optional), scope fields (client, project, vendor, site_type, drawing_type)
    - Not Valid on 'spelling' with token -> add token to allowlist
    - Not Valid on named rule with token-> add token to a 'must_not_contain' ignore? Here we push into 'ignore_rules' by id if repeated.
    - Valid on named rule with token -> ensure that token is in must_contain for a scoped synthetic rule
    """
    rules = rules_blob.get("checklist", [])
    allow = set((rules_blob.get("allowlist") or DEFAULT_ALLOWLIST))
    ignore_rules = set(rules_blob.get("ignore_rules") or [])

    # Aggregate by action
    for _, row in feedback_df.iterrows():
        rid = str(row.get("rule_id","")).strip()
        decision = str(row.get("decision","")).strip().lower()
        token = str(row.get("token","")).strip()
        scope = {
            "client": str(row.get("client","")).strip() or None,
            "project": str(row.get("project","")).strip() or None,
            "vendor": str(row.get("vendor","")).strip() or None,
            "site_type": str(row.get("site_type","")).strip() or None,
            "drawing_type": str(row.get("drawing_type","")).strip() or None,
        }

        if rid == "spelling" and decision == "not valid" and token:
            allow.add(token)

        elif decision == "not valid" and rid:
            # user says this rule shouldn't flag → add to ignore_rules
            ignore_rules.add(rid)

        elif decision == "valid" and token:
            # create or strengthen a scoped rule that requires token
            synthetic_name = f"require:{token}"
            # find existing
            found = None
            for r in rules:
                if r.get("name") == synthetic_name:
                    found = r
                    break
            if not found:
                found = {"name": synthetic_name, "severity": "major", "filters": {}, "must_contain": [token]}
                rules.append(found)
            # update filters by scope (only keep non-None)
            filt = found.setdefault("filters", {})
            for k,v in scope.items():
                if v:
                    vv = filt.get(k)
                    if vv is None:
                        filt[k] = [v]
                    elif isinstance(vv, list) and v not in vv:
                        vv.append(v)

    rules_blob["allowlist"] = sorted(allow)
    rules_blob["ignore_rules"] = sorted(ignore_rules)
    rules_blob["checklist"] = rules
    return rules_blob

# ==============================
# MAIN APP
# ==============================
def main():
    st.set_page_config("AI Design Quality Auditor", layout="wide")
    draw_logo_top_left()

    # Sidebar title
    st.sidebar.title("AI Design Quality Auditor")

    # Gate
    if not gate_password():
        st.stop()

    # Tabs
    t_audit, t_history, t_analytics, t_training, t_settings = st.tabs(
        ["Audit", "History", "Analytics", "Training", "Settings"]
    )

    # ---------------- AUDIT ----------------
    with t_audit:
        meta = metadata_block()
        err = validate_meta(meta)
        up = st.file_uploader("Upload PDF Design", type="pdf")

        c1, c2 = st.columns(2)
        run_clicked = c1.button("Run Audit", use_container_width=True)
        clear_clicked = c2.button("Clear Metadata", use_container_width=True)

        if clear_clicked:
            st.experimental_rerun()

        if up and run_clicked:
            pdf_bytes = up.read()

            # Load rules
            blob = safe_read_yaml(RULES_FILE)
            allowlist = blob.get("allowlist") or DEFAULT_ALLOWLIST
            checklist = blob.get("checklist") or []
            ignored = set(blob.get("ignore_rules") or [])

            if err:
                st.error(err)
                st.stop()

            # Extract text
            pages = extract_pages(pdf_bytes)

            # Site address check: if site_address contains ", 0 ," pattern we ignore match; else must appear in document title/text
            site_ok = True
            if ", 0 ," not in meta["site_address"]:
                # check exact-ish presence with fuzzy match threshold
                full_text = "\n".join(pages)
                sim = fuzz.partial_ratio(meta["site_address"], full_text)
                site_ok = sim >= 85

            findings: List[Dict[str, Any]] = []
            if not site_ok:
                findings.append({
                    "rule_id": "metadata:site_address_mismatch",
                    "severity": "major",
                    "message": f'Site address not found/matched: "{meta["site_address"]}"',
                    "page_num": 0,
                    "needle": meta["site_address"]
                })

            # Rules (skip any ignored)
            meta_for_rules = {
                "client": meta["client"], "project": meta["project"], "vendor": meta["vendor"],
                "site_type": meta["site_type"], "drawing_type": meta["drawing_type"],
                "ignore_rules": list(ignored)
            }
            findings += run_rules(pages, checklist, meta_for_rules)

            # Spelling
            findings += spelling_checks(pages, allowlist)

            status = "Rejected" if findings else "Pass"
            date_tag = datetime.utcnow().strftime("%Y-%m-%d")
            base_export_name = f"{up.name}_{status}_{date_tag}"

            # Annotated PDF + Excel + ZIP
            annotated_pdf = annotate_pdf(pdf_bytes, findings)
            excel = make_excel(findings, meta, up.name, status)
            zbytes = make_zip(excel, annotated_pdf, base_export_name)

            st.success(f"Audit complete: **{status}** — {len(findings)} findings")

            colz1, colz2, colz3 = st.columns(3)
            colz1.download_button("⬇️ Excel report", data=excel, file_name=f"{base_export_name}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            colz2.download_button("⬇️ Annotated PDF", data=annotated_pdf, file_name=f"{base_export_name}.pdf", mime="application/pdf")
            colz3.download_button("⬇️ ZIP (Excel + PDF)", data=zbytes, file_name=f"{base_export_name}.zip", mime="application/zip")

            # Persist to history and save artifacts so they remain available
            ensure_dirs()
            zip_path = os.path.join(ARTIFACTS_DIR, f"{base_export_name}.zip")
            with open(zip_path, "wb") as f:
                f.write(zbytes)

            row = {
                "timestamp_utc": now_iso(),
                "supplier": meta["supplier"],
                "drawing_type": meta["drawing_type"],
                "client": meta["client"],
                "project": meta["project"],
                "site_type": meta["site_type"],
                "vendor": meta["vendor"],
                "cab_loc": meta["cab_loc"],
                "radio_loc": meta["radio_loc"],
                "qty": meta["qty"],
                "site_address": meta["site_address"],
                "sector_cfg": str(meta["sector_cfg"]),
                "status": status,
                "findings_count": len(findings),
                "original_file": up.name,
                "artifacts_zip": zip_path,
                "exclude": False,
            }
            dfh = load_history()
            dfh = pd.concat([dfh, pd.DataFrame([row])], ignore_index=True)
            # Convert ts back to datetime for consistent saving
            if "timestamp_utc" in dfh.columns:
                dfh["timestamp_utc"] = pd.to_datetime(dfh["timestamp_utc"], utc=True, errors="coerce")
            save_history(dfh)

    # ---------------- HISTORY ----------------
    with t_history:
        st.subheader("Audit History")
        dfh = load_history()
        if dfh.empty:
            st.info("No history yet.")
        else:
            # Display table
            st.dataframe(dfh, use_container_width=True, height=400)
            # Toggle exclude
            idx = st.number_input("Row index to toggle Exclude from Analytics", min_value=0, max_value=len(dfh)-1, value=0, step=1)
            cc1, cc2, cc3 = st.columns(3)
            if cc1.button("Toggle Exclude"):
                dfh.loc[idx, "exclude"] = not bool(dfh.loc[idx, "exclude"])
                save_history(dfh)
                st.success(f"Row {idx} exclude set to {dfh.loc[idx,'exclude']}")
                st.experimental_rerun()

            # Re-download artifacts for selected row
            if cc2.button("Download ZIP for selected row"):
                path = str(dfh.loc[idx, "artifacts_zip"])
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button("Click to download ZIP", data=f.read(), file_name=os.path.basename(path), mime="application/zip", use_container_width=True)
                else:
                    st.error("Artifacts file not found on disk.")

            # Clean up very old artifacts (manual)
            days = cc3.number_input("Delete artifacts older than (days)", 1, 365, 60)
            if st.button("Delete old artifacts from disk"):
                cutoff = datetime.utcnow() - timedelta(days=int(days))
                removed = 0
                for _, r in dfh.iterrows():
                    p = r.get("artifacts_zip", "")
                    ts = r.get("timestamp_utc")
                    if p and isinstance(ts, str):
                        try:
                            tsdt = datetime.fromisoformat(ts.replace("Z",""))
                        except Exception:
                            tsdt = None
                        if tsdt and tsdt < cutoff and os.path.exists(p):
                            try:
                                os.remove(p); removed += 1
                            except Exception:
                                pass
                st.success(f"Removed ~{removed} old artifact files (where possible).")

    # ---------------- ANALYTICS ----------------
    with t_analytics:
        st.subheader("Analytics")
        dfh = load_history()
        if dfh.empty:
            st.info("No history yet.")
        else:
            dfh["timestamp_utc"] = pd.to_datetime(dfh["timestamp_utc"], utc=True, errors="coerce")
            df = dfh[dfh["exclude"] != True].copy()
            # Filters
            colA, colB = st.columns(2)
            sel_supplier = colA.selectbox("Filter by Supplier", ["All"] + SUPPLIERS)
            if sel_supplier != "All":
                df = df[df["supplier"] == sel_supplier]

            # Metrics
            total = len(df)
            pass_rate = (df["status"] == "Pass").mean() * 100 if total else 0.0
            rft = (df["findings_count"] == 0).mean() * 100 if total else 0.0
            m1,m2,m3 = st.columns(3)
            m1.metric("Total Audits", total)
            m2.metric("Pass Rate", f"{pass_rate:.1f}%")
            m3.metric("Right First Time", f"{rft:.1f}%")

            # Trend (daily)
            if not df.empty and "timestamp_utc" in df.columns:
                df["day"] = df["timestamp_utc"].dt.tz_convert("UTC").dt.date
                trend = df.groupby(["day","status"]).size().unstack(fill_value=0)
                st.line_chart(trend)

    # ---------------- TRAINING ----------------
    with t_training:
        st.subheader("Trainer – Teach the tool (fast)")

        st.markdown(
            "Upload a feedback CSV/Excel with columns like: "
            "`rule_id, decision(Valid/Not Valid), token(optional), client, project, vendor, site_type, drawing_type`."
        )
        f_up = st.file_uploader("Upload feedback (CSV/XLSX)", type=["csv","xlsx"])
        if f_up and st.button("Apply Feedback to Rules"):
            try:
                if f_up.name.lower().endswith(".xlsx"):
                    fb = pd.read_excel(f_up)
                else:
                    fb = pd.read_csv(f_up)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                fb = None
            if fb is not None:
                blob = safe_read_yaml(RULES_FILE)
                new_blob = apply_feedback_to_rules(fb, blob)
                safe_write_yaml(RULES_FILE, new_blob)
                st.success("Feedback applied to rules.yaml")

        st.markdown("---")
        st.markdown("### Quick add single rule")
        with st.form("quick_rule"):
            r_name = st.text_input("Rule name (unique)", "")
            r_sev = st.selectbox("Severity", ["minor","major","critical"])
            # scope
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            f_client = sc1.multiselect("Client (scope)", CLIENTS)
            f_project = sc2.multiselect("Project (scope)", PROJECTS)
            f_vendor = sc3.multiselect("Vendor (scope)", VENDORS)
            f_site = sc4.multiselect("Site Type (scope)", SITE_TYPES)
            f_draw = sc5.multiselect("Drawing Type (scope)", DRAWING_TYPES)

            must = st.text_area("Must contain (one per line)", "")
            mustnot = st.text_area("Must NOT contain (one per line)", "")
            pair_note = st.text_area('Forbidden pairs (format: IF => NOT, one per line, e.g. `Brush => Generator Power`)', "")

            submitted = st.form_submit_button("Add rule")
            if submitted:
                if not r_name.strip():
                    st.error("Rule name required.")
                else:
                    blob = safe_read_yaml(RULES_FILE)
                    checklist = blob.get("checklist") or []
                    # create rule
                    rule = {
                        "name": r_name.strip(),
                        "severity": r_sev,
                        "filters": {
                            "client": f_client or None,
                            "project": f_project or None,
                            "vendor": f_vendor or None,
                            "site_type": f_site or None,
                            "drawing_type": f_draw or None,
                        },
                        "must_contain": [x.strip() for x in must.splitlines() if x.strip()],
                        "must_not_contain": [x.strip() for x in mustnot.splitlines() if x.strip()],
                        "forbidden_pairs": []
                    }
                    for line in pair_note.splitlines():
                        if "=>" in line:
                            a,b = line.split("=>",1)
                            rule["forbidden_pairs"].append({"if_contains": a.strip(), "must_not_include": b.strip()})
                    checklist.append(rule)
                    blob["checklist"] = checklist
                    safe_write_yaml(RULES_FILE, blob)
                    st.success("Rule added to rules.yaml")

    # ---------------- SETTINGS ----------------
    with t_settings:
        st.subheader("Settings")
        # Show / edit rules.yaml (admin gated)
        if rules_admin_unlocked():
            try:
                current_text = open(RULES_FILE,"r",encoding="utf-8").read()
            except Exception:
                current_text = yaml.safe_dump({"allowlist": DEFAULT_ALLOWLIST, "checklist": [], "ignore_rules": []}, sort_keys=False)
            new_text = st.text_area("rules_example.yaml", current_text, height=360)
            if st.button("Save YAML"):
                try:
                    test = yaml.safe_load(new_text) or {}
                    safe_write_yaml(RULES_FILE, test)
                    st.success("Saved rules_example.yaml")
                except Exception as e:
                    st.error(f"YAML error: {e}")
        else:
            st.info("Unlock with admin password to edit YAML.")

        if st.button("Clear ALL history (keeps artifacts on disk)"):
            save_history(pd.DataFrame(columns=[
                "timestamp_utc","supplier","drawing_type","client","project","site_type",
                "vendor","cab_loc","radio_loc","qty","site_address","sector_cfg",
                "status","findings_count","original_file","artifacts_zip","exclude"
            ]))
            st.success("History cleared (CSV). Artifacts left intact.")


if __name__ == "__main__":
    main()
