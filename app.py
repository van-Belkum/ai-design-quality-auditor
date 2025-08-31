import os
import io
import re
import json
import base64
import hashlib
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import yaml

# PDF + OCR stack
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import pytesseract

# Fuzzy/spell
from rapidfuzz import fuzz, process
from spellchecker import SpellChecker

# -----------------------------
# CONSTANTS & DEFAULTS
# -----------------------------
ENTRY_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"

APP_TITLE = "AI Design Quality Auditor"

EXPORT_DIR = "exports"
HISTORY_DIR = "history"
HISTORY_CSV = os.path.join(HISTORY_DIR, "audit_history.csv")

DEFAULT_CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
DEFAULT_PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
DEFAULT_SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
DEFAULT_VENDORS = ["Ericsson", "Nokia"]
DEFAULT_CABINET_LOCS = ["Indoor", "Outdoor"]
DEFAULT_RADIO_LOCS = ["Low Level", "High Level", "Unique Coverage", "Midway"]
DEFAULT_SUPPLIERS = [
    "TXM", "Cobra", "Mott MacDonald", "SSE", "BT", "Telent", "Daly",
    "DMS", "Naturally Wild", "F&L", "Mono", "Gowrings", "Other"
]  # You can overwrite in Settings

# A tidy, deduped and sorted MIMO options list (you can extend via Settings)
DEFAULT_MIMO_OPTIONS = sorted(list({
    "18 @2x2",
    "18 @2x2; 26 @4x4",
    "18 @2x2; 70\\80 @2x2",
    "18 @2x2; 80 @2x2",
    "18\\21 @2x2",
    "18\\21 @2x2; 26 @4x4",
    "18\\21 @2x2; 3500 @32x32",
    "18\\21 @2x2; 70\\80 @2x2",
    "18\\21 @2x2; 80 @2x2",
    "18\\21 @4x4",
    "18\\21 @4x4; 3500 @32x32",
    "18\\21 @4x4; 70 @2x4",
    "18\\21 @4x4; 70\\80 @2x2",
    "18\\21 @4x4; 70\\80 @2x2; 3500 @32x32",
    "18\\21 @4x4; 70\\80 @2x4",
    "18\\21 @4x4; 70\\80 @2x4; 3500 @32x32",
    "18\\21 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21@4x4; 70\\80 @2x2",
    "18\\21@4x4; 70\\80 @2x4",
    "18\\21\\26 @2x2",
    "18\\21\\26 @2x2; 3500 @32x32",
    "18\\21\\26 @2x2; 3500 @8X8",
    "18\\21\\26 @2x2; 70\\80 @2x2",
    "18\\21\\26 @2x2; 70\\80 @2x2; 3500 @32x32",
    "18\\21\\26 @2x2; 70\\80 @2x2; 3500 @8x8",
    "18\\21\\26 @2x2; 70\\80 @2x4; 3500 @32x32",
    "18\\21\\26 @2x2; 80 @2x2",
    "18\\21\\26 @2x2; 80 @2x2; 3500 @8x8",
    "18\\21\\26 @4x4",
    "18\\21\\26 @4x4; 3500 @32x32",
    "18\\21\\26 @4x4; 3500 @8x8",
    "18\\21\\26 @4x4; 70 @2x4; 3500 @8x8",
    "18\\21\\26 @4x4; 70\\80 @2x2",
    "18\\21\\26 @4x4; 70\\80 @2x2; 3500 @32x32",
    "18\\21\\26 @4x4; 70\\80 @2x2; 3500 @8x8",
    "18\\21\\26 @4x4; 70\\80 @2x4",
    "18\\21\\26 @4x4; 70\\80 @2x4; 3500 @32x32",
    "18\\21\\26 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21\\26 @4x4; 80 @2x2",
    "18\\21\\26 @4x4; 80 @2x2; 3500 @32x32",
    "18\\21\\26 @4x4; 80 @2x4",
    "18\\21\\26 @4x4; 80 @2x4; 3500 @8x8",
    "18\\26 @2x2",
    "18\\26 @4x4; 21 @2x2; 80 @2x2",
    "(blank)"
}))

DEFAULT_UI_VISIBILITY_HOURS = 24  # how long to show runs in-app by default

# -----------------------------
# UTILITIES
# -----------------------------
def ensure_dirs():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

def safe_yaml_load(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return {}
            return data
    except Exception:
        return {}

def safe_yaml_dump(path: str, data: Dict[str, Any]) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        return True
    except Exception:
        return False

def load_history() -> pd.DataFrame:
    ensure_dirs()
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_CSV)
    except Exception:
        # create a backup and start fresh if corrupt
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.rename(HISTORY_CSV, os.path.join(HISTORY_DIR, f"audit_history_corrupt_{ts}.csv"))
        return pd.DataFrame()
    return df

def append_history(row: Dict[str, Any]) -> None:
    ensure_dirs()
    df = load_history()
    new = pd.DataFrame([row])
    df_out = pd.concat([df, new], ignore_index=True)
    df_out.to_csv(HISTORY_CSV, index=False)

def b64_logo(logo_path: str) -> Optional[str]:
    try:
        if logo_path and os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception:
        pass
    return None

def today_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d")

def stamp_filename(base: str, status: str) -> str:
    name, ext = os.path.splitext(base)
    return f"{name} - {status.upper()} - {today_stamp()}{ext}"

def naive_text_from_pdf(pdf_bytes: bytes) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns a list of page texts and a list of text spans with bbox for annotation:
    spans = [{"page": i, "text": "...", "bbox": (x0,y0,x1,y1)}...]
    """
    spans: List[Dict[str, Any]] = []
    texts: List[str] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            texts.append(page.get_text())
            # Keep span-level positions for later matching
            for b in page.get_text("blocks"):
                # each block: (x0, y0, x1, y1, "text", block_no, block_type)
                if len(b) >= 5 and isinstance(b[4], str):
                    spans.append({
                        "page": i,
                        "text": b[4],
                        "bbox": (b[0], b[1], b[2], b[3])
                    })
    except Exception:
        # Fallback via OCR (slower)
        try:
            images = convert_from_bytes(pdf_bytes)
            for i, im in enumerate(images):
                t = pytesseract.image_to_string(im)
                texts.append(t)
            # No bbox available in OCR fallback
        except Exception:
            texts = []
    return texts, spans

def annotate_pdf(original_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    """
    Draw rectangles & notes for findings that have bbox or page, otherwise skip.
    We try to place a sticky note near matched bbox; if no bbox, we add a page-level note in the margin.
    """
    try:
        doc = fitz.open(stream=original_bytes, filetype="pdf")
        for f in findings:
            page_idx = f.get("page_index", 0)
            if page_idx < 0 or page_idx >= len(doc):
                continue
            page = doc[page_idx]
            comment = f'{f.get("rule_id","RULE")}: {f.get("message","Issue")}'
            bbox = f.get("bbox")
            if bbox and isinstance(bbox, (tuple, list)) and len(bbox) == 4:
                rect = fitz.Rect(*bbox)
                try:
                    highlight = page.add_rect_annot(rect)
                    highlight.set_colors(stroke=(1, 0, 0))
                    highlight.set_border(width=1)
                    highlight.update()
                    page.add_text_annot(rect.br, comment)
                except Exception:
                    page.add_text_annot(page.rect.br, comment)
            else:
                page.add_text_annot(page.rect.br, comment)
        pdf_out = doc.tobytes()
        doc.close()
        return pdf_out
    except Exception:
        return original_bytes

def make_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], original_name: str, status: str) -> bytes:
    """
    Create an .xlsx with Findings + Metadata sheets using openpyxl engine (available via pandas).
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df_f = pd.DataFrame(findings) if findings else pd.DataFrame(
            columns=["rule_id","severity","message","page_index","bbox","evidence","validated"])
        df_m = pd.DataFrame([meta])
        df_f.to_excel(xw, sheet_name="Findings", index=False)
        df_m.to_excel(xw, sheet_name="Metadata", index=False)
    buf.seek(0)
    return buf.read()

def normalized_words(s: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']+", s)

def spelling_findings(pages: List[str], allow_words: set) -> List[Dict[str, Any]]:
    sp = SpellChecker(distance=1)
    res: List[Dict[str, Any]] = []
    for pi, txt in enumerate(pages):
        for w in normalized_words(txt):
            wl = w.lower()
            if wl in allow_words:
                continue
            # ignore short tokens and mixed stuff
            if len(wl) <= 2:
                continue
            candidates = set()
            try:
                candidates = set(sp.candidates(wl))
            except Exception:
                # safety guard for rare token bugs
                candidates = set()
            if wl in sp:
                continue
            suggestion = None
            try:
                suggestion = next(iter(candidates), None)
            except Exception:
                suggestion = None
            res.append({
                "rule_id": "SPELLING",
                "severity": "Minor",
                "message": f"Possible misspelling: '{w}'",
                "page_index": pi,
                "bbox": None,
                "evidence": suggestion,
                "validated": None
            })
    return res

def rule_eval_findings(pages: List[str], meta: Dict[str, Any], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Very lightweight rules engine:
      - regex rules (with optional severity, must_contain, must_not_contain)
      - address/title matching rule (reject if mismatch unless contains ', 0 ,')
    """
    findings: List[Dict[str, Any]] = []
    text_all = "\n".join(pages)

    # 1) Address vs Title check (simple version)
    site_addr = (meta.get("site_address") or "").strip()
    pdf_title = (meta.get("pdf_title") or "").strip()
    if site_addr and pdf_title:
        if ", 0 ," in site_addr:
            pass  # ignore per your rule
        else:
            norm = lambda s: re.sub(r"[^A-Za-z0-9]", "", s).lower()
            if norm(site_addr) not in norm(pdf_title):
                findings.append({
                    "rule_id": "ADDR_TITLE",
                    "severity": "Major",
                    "message": "Site Address does not match PDF title.",
                    "page_index": 0,
                    "bbox": None,
                    "evidence": f"title='{pdf_title}' vs site='{site_addr}'",
                    "validated": None
                })

    # 2) Hide MIMO check for Power Resilience handled in UI; we still record metadata.
    # 3) Regex rules from YAML
    yaml_rules = (rules.get("rules") or [])
    for r in yaml_rules:
        try:
            # Optional filters by metadata (client, project, vendor, site_type, etc.)
            # If any filter present, enforce match.
            meta_filters = r.get("when", {})
            if meta_filters:
                miss = False
                for k, v in meta_filters.items():
                    if str(meta.get(k, "")).strip() != str(v).strip():
                        miss = True
                        break
                if miss:
                    continue

            pattern = r.get("pattern")
            if not pattern:
                continue
            regex = re.compile(pattern, flags=re.IGNORECASE|re.MULTILINE)
            matches = list(regex.finditer(text_all))
            should_exist = r.get("should_exist", True)
            severity = r.get("severity", "Major")
            rid = r.get("id", f"RULE_{hashlib.md5(pattern.encode()).hexdigest()[:6]}")
            msg = r.get("message", "Rule check")

            if should_exist:
                if not matches:
                    findings.append({
                        "rule_id": rid,
                        "severity": severity,
                        "message": msg if msg else f"Missing required pattern: {pattern}",
                        "page_index": 0,
                        "bbox": None,
                        "evidence": None,
                        "validated": None
                    })
                else:
                    # If requires also must_contain or must_not_contain
                    must_contain = r.get("must_contain")
                    must_not = r.get("must_not_contain")
                    if must_contain and must_contain.lower() not in text_all.lower():
                        findings.append({
                            "rule_id": rid,
                            "severity": severity,
                            "message": f"Document missing required phrase: {must_contain}",
                            "page_index": 0,
                            "bbox": None,
                            "evidence": None,
                            "validated": None
                        })
                    if must_not and must_not.lower() in text_all.lower():
                        findings.append({
                            "rule_id": rid,
                            "severity": severity,
                            "message": f"Document contains forbidden phrase: {must_not}",
                            "page_index": 0,
                            "bbox": None,
                            "evidence": None,
                            "validated": None
                        })
            else:
                if matches:
                    findings.append({
                        "rule_id": rid,
                        "severity": severity,
                        "message": msg if msg else f"Forbidden pattern present: {pattern}",
                        "page_index": 0,
                        "bbox": None,
                        "evidence": matches[0].group(0)[:200],
                        "validated": None
                    })
        except Exception:
            continue

    # 4) Example cross-term rule via config: "if 'Brush' then 'Generator Power' should not be used"
    forbid_pairs = (rules.get("forbid_pairs") or [])
    for fp in forbid_pairs:
        a = fp.get("contains")
        not_with = fp.get("not_with")
        if a and not_with:
            if a.lower() in text_all.lower() and not_with.lower() in text_all.lower():
                findings.append({
                    "rule_id": "PAIR_FORBID",
                    "severity": "Major",
                    "message": f"'{a}' must not appear with '{not_with}'.",
                    "page_index": 0,
                    "bbox": None,
                    "evidence": None,
                    "validated": None
                })

    return findings

def run_checks(pages: List[str], meta: Dict[str, Any], rules: Dict[str, Any],
               do_spell: bool, allow_words: set) -> List[Dict[str, Any]]:
    out = []
    out.extend(rule_eval_findings(pages, meta, rules))
    if do_spell:
        out.extend(spelling_findings(pages, allow_words))
    return out

def filter_history_for_ui(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    if df.empty or "timestamp_utc" not in df.columns:
        return df
    try:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return df[df["timestamp_utc"] >= cutoff]
    except Exception:
        return df

def gate() -> bool:
    st.title(APP_TITLE)
    if "entry_ok" not in st.session_state:
        st.session_state.entry_ok = False

    if not st.session_state.entry_ok:
        with st.form("entry_gate"):
            pw = st.text_input("Enter access password", type="password")
            ok = st.form_submit_button("Enter")
        if ok:
            if pw == ENTRY_PASSWORD:
                st.session_state.entry_ok = True
                st.experimental_rerun()
            else:
                st.error("Incorrect password.")
        return False
    return True

def rules_password_ok() -> bool:
    if "rules_ok" not in st.session_state:
        st.session_state.rules_ok = False
    if not st.session_state.rules_ok:
        with st.form("rules_pw"):
            pw = st.text_input("Rules/Settings password", type="password")
            ok = st.form_submit_button("Unlock")
        if ok:
            if pw == RULES_PASSWORD:
                st.session_state.rules_ok = True
            else:
                st.error("Incorrect password.")
    return st.session_state.rules_ok

def get_ui_lists(rules: Dict[str, Any]) -> Dict[str, List[str]]:
    ui = rules.get("ui") or {}
    def pick(key, default):
        v = ui.get(key)
        if isinstance(v, list) and v:
            return [str(x) for x in v]
        return default
    return {
        "suppliers": pick("supplier_options", DEFAULT_SUPPLIERS),
        "clients": pick("client_options", DEFAULT_CLIENTS),
        "projects": pick("project_options", DEFAULT_PROJECTS),
        "site_types": pick("site_type_options", DEFAULT_SITE_TYPES),
        "vendors": pick("vendor_options", DEFAULT_VENDORS),
        "cabinet_locs": pick("cabinet_loc_options", DEFAULT_CABINET_LOCS),
        "radio_locs": pick("radio_loc_options", DEFAULT_RADIO_LOCS),
        "mimo_options": pick("mimo_options", DEFAULT_MIMO_OPTIONS),
    }

# -----------------------------
# UI AREAS
# -----------------------------
def header_with_logo(logo_path: Optional[str]):
    if logo_path and os.path.exists(logo_path):
        b64 = b64_logo(logo_path)
        if b64:
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;justify-content:space-between;">
                  <h2 style="margin:0;padding:0">{APP_TITLE}</h2>
                  <img src="data:image/png;base64,{b64}" style="height:52px;object-fit:contain;margin-left:12px;" />
                </div>
                <hr style="margin-top:8px;"/>
                """,
                unsafe_allow_html=True
            )
            return
    st.subheader(APP_TITLE)

def audit_tab(rules_path: str):
    rules = safe_yaml_load(rules_path)
    lists = get_ui_lists(rules)

    with st.form("audit_form"):
        c1, c2, c3 = st.columns(3)
        supplier = c1.selectbox("Supplier", lists["suppliers"])
        client = c2.selectbox("Client", lists["clients"])
        project = c3.selectbox("Project", lists["projects"])

        c4, c5, c6 = st.columns(3)
        site_type = c4.selectbox("Site Type", lists["site_types"])
        vendor = c5.selectbox("Proposed Vendor", lists["vendors"])
        cabinet_loc = c6.selectbox("Proposed Cabinet Location", lists["cabinet_locs"])

        c7, c8, c9 = st.columns(3)
        radio_loc = c7.selectbox("Proposed Radio Location", lists["radio_locs"])
        qty_sectors = c8.selectbox("Quantity of Sectors", [1,2,3,4,5,6], index=2)
        do_spell = c9.checkbox("Run spelling check", value=True)

        # Sector MIMO configs (hide for Power Resilience)
        sector_mimos = {}
        same_all = False
        if project != "Power Resilience":
            st.markdown("**Proposed MIMO Config (per sector)**")
            same_all = st.checkbox("Use S1 MIMO for all sectors", value=False)
            s1 = st.selectbox("S1 MIMO", lists["mimo_options"], key="m_s1")
            sector_mimos["S1"] = s1
            if not same_all:
                for s in range(2, qty_sectors+1):
                    sector_mimos[f"S{s}"] = st.selectbox(f"S{s} MIMO", lists["mimo_options"], key=f"m_s{s}")
            else:
                for s in range(2, qty_sectors+1):
                    sector_mimos[f"S{s}"] = s1

        site_address = st.text_input("Site Address (exact; include ', 0 ,' to bypass title match rule)")
        pdf_file = st.file_uploader("Upload PDF to Audit", type=["pdf"])

        exclude_from_analytics = st.checkbox("Exclude this review from analytics", value=False)
        audit_btn = st.form_submit_button("🔍 Run Audit")

    if not audit_btn:
        return

    if not pdf_file:
        st.error("Please upload a PDF.")
        return

    # Pull PDF text & spans
    raw = pdf_file.read()
    pages, spans = naive_text_from_pdf(raw)

    # Allowlist (from rules)
    allow = set([w.lower() for w in (rules.get("allowlist") or [])])

    # Metadata
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "supplier": supplier,
        "client": client,
        "project": project,
        "site_type": site_type,
        "vendor": vendor,
        "cabinet_loc": cabinet_loc,
        "radio_loc": radio_loc,
        "qty_sectors": qty_sectors,
        "mimo": sector_mimos,
        "site_address": site_address,
        "pdf_title": pages[0].splitlines()[0] if pages else "",
        "exclude": bool(exclude_from_analytics),
    }

    findings = run_checks(pages, meta, rules, do_spell, allow)

    # Attempt to attach bbox/page to findings by fuzzy locating evidence text in spans
    # (best-effort; if not found, annotations will still add a page note)
    try:
        flat = "\n".join(pages).lower()
        for f in findings:
            snippet = (f.get("evidence") or f.get("message") or "")[:80].lower()
            if not snippet or not spans:
                f.setdefault("page_index", 0)
                continue
            # find page with best fuzz ratio on block text
            best = (-1, None, None)  # score, page, bbox
            for s in spans:
                score = fuzz.partial_ratio(snippet, (s.get("text") or "").lower())
                if score > best[0]:
                    best = (score, s.get("page"), s.get("bbox"))
            if best[1] is not None:
                f["page_index"] = int(best[1])
                f["bbox"] = best[2]
            else:
                f.setdefault("page_index", 0)
    except Exception:
        for f in findings:
            f.setdefault("page_index", 0)

    status = "Pass" if not any(x.get("severity","").lower()=="major" for x in findings) else "Rejected"

    # Build files
    excel_bytes = make_excel(findings, meta, pdf_file.name, status)
    annotated_bytes = annotate_pdf(raw, findings)

    # Persist both to exports with stamped names
    ensure_dirs()
    stamped_xlsx = stamp_filename(os.path.splitext(pdf_file.name)[0] + ".xlsx", status)
    stamped_pdf = stamp_filename(pdf_file.name.replace(".pdf"," - ANNOTATED.pdf"), status)

    xlsx_path = os.path.join(EXPORT_DIR, stamped_xlsx)
    with open(xlsx_path, "wb") as f:
        f.write(excel_bytes)

    pdf_path = os.path.join(EXPORT_DIR, stamped_pdf)
    with open(pdf_path, "wb") as f:
        f.write(annotated_bytes)

    # History row
    hist_row = {
        "timestamp_utc": meta["timestamp_utc"],
        "supplier": supplier,
        "client": client,
        "project": project,
        "site_type": site_type,
        "vendor": vendor,
        "cabinet_loc": cabinet_loc,
        "radio_loc": radio_loc,
        "qty_sectors": qty_sectors,
        "status": status,
        "pdf_name": stamped_pdf,
        "excel_name": stamped_xlsx,
        "exclude": bool(exclude_from_analytics)
    }
    append_history(hist_row)

    st.success(f"Audit complete: **{status}**")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download Findings (.xlsx)", data=excel_bytes,
                           file_name=stamped_xlsx, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.caption(f"Saved: {xlsx_path}")
    with c2:
        st.download_button("⬇️ Download Annotated PDF", data=annotated_bytes,
                           file_name=stamped_pdf, mime="application/pdf")
        st.caption(f"Saved: {pdf_path}")

    # Show findings table for quick triage
    if findings:
        st.markdown("### Findings")
        st.dataframe(pd.DataFrame(findings), use_container_width=True)
    else:
        st.info("No issues detected.")

def training_tab(rules_path: str):
    st.markdown("### Training / Learning")
    st.caption("Upload a prior report (Excel) you’ve marked Valid/Not Valid OR add simple rules interactively.")

    up = st.file_uploader("Upload Findings Excel (with a 'validated' column containing Valid/Not Valid)", type=["xlsx"])
    if up:
        try:
            df = pd.read_excel(up, sheet_name="Findings")
            st.dataframe(df, use_container_width=True)
            # Update allowlist/rules from validations
            valid_tokens = set()
            new_forbid_pairs = []
            for _, r in df.iterrows():
                val = str(r.get("validated","")).strip().lower()
                msg = str(r.get("message",""))
                ev = str(r.get("evidence",""))
                if val in ("valid","true","yes","y"):
                    # treat evidence token (if simple) as allow-word
                    for w in normalized_words(ev or msg):
                        if len(w) > 2:
                            valid_tokens.add(w.lower())
                elif val in ("not valid","false","no","n"):
                    # user says this is a false positive rejection — remove from allowlist if present
                    pass
            # Load and update YAML
            rules = safe_yaml_load(rules_path)
            allow = set([w.lower() for w in (rules.get("allowlist") or [])])
            allow |= valid_tokens
            rules["allowlist"] = sorted(list(allow))
            if safe_yaml_dump(rules_path, rules):
                st.success(f"Allowlist updated with {len(valid_tokens)} tokens.")
            else:
                st.error("Failed to write rules YAML.")
        except Exception as e:
            st.error(f"Could not parse Excel: {e}")

    st.divider()
    st.markdown("#### Quick add: simple rule (contains / not-with)")
    with st.form("quick_rule"):
        col1, col2 = st.columns(2)
        contains = col1.text_input("If text contains… (e.g., Brush)")
        not_with = col2.text_input("…it must NOT appear with (e.g., Generator Power)")
        rid = st.text_input("Rule ID (optional)")
        sev = st.selectbox("Severity", ["Major","Minor"], index=0)
        add = st.form_submit_button("Add forbid pair")
    if add:
        rules = safe_yaml_load(rules_path)
        lst = rules.get("forbid_pairs") or []
        lst.append({"contains": contains, "not_with": not_with, "severity": sev, "id": rid or None})
        rules["forbid_pairs"] = lst
        if safe_yaml_dump(rules_path, rules):
            st.success("Forbid-pair rule added.")
        else:
            st.error("Failed to update rules YAML.")

def analytics_tab():
    st.markdown("### Analytics")
    dfh = load_history()
    if dfh.empty:
        st.info("No runs yet.")
        return
    # Filters
    cols = st.columns(3)
    sup = cols[0].multiselect("Supplier", sorted(dfh["supplier"].dropna().unique().tolist()))
    cli = cols[1].multiselect("Client", sorted(dfh["client"].dropna().unique().tolist()))
    pro = cols[2].multiselect("Project", sorted(dfh["project"].dropna().unique().tolist()))

    df = dfh.copy()
    if sup: df = df[df["supplier"].isin(sup)]
    if cli: df = df[df["client"].isin(cli)]
    if pro: df = df[df["project"].isin(pro)]

    st.metric("Total Audits", len(df))
    if "status" in df.columns:
        st.metric("Right First Time (%)", round(100* (df["status"].str.lower()=="pass").mean(), 1))

    show_cols = ["timestamp_utc","supplier","client","project","status","pdf_name","excel_name"]
    exist_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[exist_cols], use_container_width=True)

def settings_tab(rules_path: str):
    st.markdown("### Settings / Pick-lists / YAML")
    if not rules_password_ok():
        return
    rules = safe_yaml_load(rules_path)
    lists = get_ui_lists(rules)

    st.markdown("#### Branding")
    lp = st.text_input("Logo file path (relative)", value=st.session_state.get("logo_path",""))
    if st.button("Save logo path"):
        st.session_state["logo_path"] = lp
        st.success("Logo path saved for this session.")

    st.divider()
    st.markdown("#### Pick-list Management")
    col1, col2 = st.columns(2)
    suppliers_text = col1.text_area("Supplier options (one per line)", value="\n".join(lists["suppliers"]), height=160)
    clients_text = col1.text_area("Client options (one per line)", value="\n".join(lists["clients"]), height=160)
    projects_text = col1.text_area("Project options (one per line)", value="\n".join(lists["projects"]), height=160)

    site_text = col2.text_area("Site type options", value="\n".join(lists["site_types"]), height=160)
    vendor_text = col2.text_area("Vendor options", value="\n".join(lists["vendors"]), height=160)
    radio_text = col2.text_area("Radio location options", value="\n".join(lists["radio_locs"]), height=160)

    mimo_text = st.text_area("MIMO options", value="\n".join(lists["mimo_options"]), height=180)

    if st.button("Save all pick-lists"):
        ui = rules.get("ui") or {}
        ui["supplier_options"] = [s.strip() for s in suppliers_text.splitlines() if s.strip()]
        ui["client_options"] = [s.strip() for s in clients_text.splitlines() if s.strip()]
        ui["project_options"] = [s.strip() for s in projects_text.splitlines() if s.strip()]
        ui["site_type_options"] = [s.strip() for s in site_text.splitlines() if s.strip()]
        ui["vendor_options"] = [s.strip() for s in vendor_text.splitlines() if s.strip()]
        ui["radio_loc_options"] = [s.strip() for s in radio_text.splitlines() if s.strip()]
        ui["mimo_options"] = [s.strip() for s in mimo_text.splitlines() if s.strip()]
        rules["ui"] = ui
        if safe_yaml_dump(rules_path, rules):
            st.success("Saved pick-lists.")
        else:
            st.error("Failed writing YAML.")

    st.divider()
    st.markdown("#### Raw YAML editor (advanced)")
    raw = yaml.safe_dump(rules, sort_keys=False, allow_unicode=True)
    raw_edit = st.text_area("rules_example.yaml", value=raw, height=300)
    if st.button("Save YAML"):
        try:
            data = yaml.safe_load(raw_edit) or {}
            if safe_yaml_dump(rules_path, data):
                st.success("YAML saved.")
            else:
                st.error("Failed to save YAML.")
        except yaml.YAMLError as e:
            st.error(f"YAML error: {e}")

# -----------------------------
# MAIN
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    ensure_dirs()

    # locate rules yaml in working dir; allow user to swap via sidebar
    rules_file = st.sidebar.text_input("Rules YAML path", value="rules_example.yaml")
    if not os.path.exists(rules_file):
        # create a minimal seed file
        seed = {
            "ui": {
                "supplier_options": DEFAULT_SUPPLIERS,
                "client_options": DEFAULT_CLIENTS,
                "project_options": DEFAULT_PROJECTS,
                "site_type_options": DEFAULT_SITE_TYPES,
                "vendor_options": DEFAULT_VENDORS,
                "radio_loc_options": DEFAULT_RADIO_LOCS,
                "mimo_options": DEFAULT_MIMO_OPTIONS,
            },
            "allowlist": [],
            "rules": [],
            "forbid_pairs": []
        }
        safe_yaml_dump(rules_file, seed)

    if not gate():
        return

    header_with_logo(st.session_state.get("logo_path"))

    tab_audit, tab_train, tab_analytics, tab_settings = st.tabs(
        ["🔍 Audit", "🎓 Training", "📈 Analytics", "⚙️ Settings"]
    )

    with tab_audit:
        audit_tab(rules_file)
    with tab_train:
        training_tab(rules_file)
    with tab_analytics:
        analytics_tab()
    with tab_settings:
        settings_tab(rules_file)

if __name__ == "__main__":
    main()
