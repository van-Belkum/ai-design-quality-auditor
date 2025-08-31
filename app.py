# app.py
# AI Design Quality Auditor - Stable Build (fixes, learning, analytics)
# Requires: streamlit, pandas, numpy, rapidfuzz, pyyaml, openpyxl, PyMuPDF, pytesseract (optional), pyspellchecker (optional)

from __future__ import annotations

import base64
import io
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yaml

# PDF
import fitz  # PyMuPDF

# Optional deps
try:
    from spellchecker import SpellChecker  # pyspellchecker
except Exception:
    SpellChecker = None  # type: ignore

# ----------------------------
# CONSTANTS & PATHS
# ----------------------------
ENTRY_PASSWORD = "Seker123"
SETTINGS_PASSWORD = "vanB3lkum21"

RULES_PATH = "rules_example.yaml"
HISTORY_DIR = "history"
HISTORY_CSV = os.path.join(HISTORY_DIR, "audits.csv")
EXPORTS_DIR = "exports"
SESSION_DIR = "session_outputs"
GUIDES_DIR = "guides"

DEFAULT_SUPPLIERS = [
    # Maintain here, but can be edited in Settings
    "Cornerstone",
    "MBNL",
    "BTEE",
    "Vodafone",
    "H3G",
    "Cellnex",
    "WHP Telecoms",
    "Wireless Infrastructure Group",
    "On Tower",
    "BT",
    "Openreach",
]

DEFAULT_CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
DEFAULT_PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
DEFAULT_SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
DEFAULT_VENDORS = ["Ericsson", "Nokia"]
DEFAULT_CAB_LOC = ["Indoor", "Outdoor"]
DEFAULT_RADIO_LOC = ["Low Level", "High Level", "Unique Coverage", "Midway"]
DEFAULT_DRAWING_TYPES = ["General Arrangement", "Detailed Design"]

DEFAULT_MIMO_OPTIONS = [
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
]

DEFAULT_UI_VISIBILITY_HOURS = 24  # history shown in Analytics for last N hours by default

# ----------------------------
# UTILITIES
# ----------------------------
def ensure_dirs():
    for d in [HISTORY_DIR, EXPORTS_DIR, SESSION_DIR, GUIDES_DIR]:
        os.makedirs(d, exist_ok=True)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ts_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def load_rules(path: str = RULES_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
        except Exception:
            # YAML error friendly fallback
            return {}
    return data


def save_rules(data: Dict[str, Any], path: str = RULES_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_history() -> pd.DataFrame:
    ensure_dirs()
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "supplier",
                "client",
                "project",
                "site_type",
                "vendor",
                "cabinet_location",
                "radio_location",
                "drawing_type",
                "sectors",
                "mimo_s1",
                "mimo_s2",
                "mimo_s3",
                "mimo_s4",
                "mimo_s5",
                "mimo_s6",
                "site_address",
                "status",
                "minor_count",
                "major_count",
                "pdf_name",
                "excel_name",
                "exclude",
            ]
        )
    try:
        df = pd.read_csv(HISTORY_CSV)
    except Exception:
        # corrupted file fallback
        backup = os.path.join(HISTORY_DIR, f"audits_corrupt_{int(datetime.now().timestamp())}.csv")
        try:
            os.rename(HISTORY_CSV, backup)
        except Exception:
            pass
        df = pd.DataFrame(
            columns=[
                "timestamp_utc",
                "supplier",
                "client",
                "project",
                "site_type",
                "vendor",
                "cabinet_location",
                "radio_location",
                "drawing_type",
                "sectors",
                "mimo_s1",
                "mimo_s2",
                "mimo_s3",
                "mimo_s4",
                "mimo_s5",
                "mimo_s6",
                "site_address",
                "status",
                "minor_count",
                "major_count",
                "pdf_name",
                "excel_name",
                "exclude",
            ]
        )
    # Strong typing
    if "exclude" not in df.columns:
        df["exclude"] = False
    # Coerce timestamp to datetime
    if "timestamp_utc" in df.columns:
        try:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        except Exception:
            pass
    return df


def save_history_row(row: Dict[str, Any]) -> None:
    df = load_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    # Ensure dtypes
    if "exclude" in df.columns:
        df["exclude"] = df["exclude"].astype(bool)
    df.to_csv(HISTORY_CSV, index=False)


def filter_history_for_ui(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = now_utc() - timedelta(hours=max(1, hours))
    if df["timestamp_utc"].dtype == "O":
        # coerce if needed
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df[df["timestamp_utc"] >= cutoff]


def read_pdf_text_and_pages(pdf_bytes: bytes) -> Tuple[List[str], fitz.Document]:
    """Return per-page text (list) and an open fitz document (keep open for annotation)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        texts.append(page.get_text("text") or "")
    return texts, doc


# ----------------------------
# SPELLING (SAFE)
# ----------------------------
def spelling_findings(pages: List[str], allow_list: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not SpellChecker:
        return out
    try:
        sp = SpellChecker(distance=1)
    except Exception:
        return out

    allow = set(w.strip().lower() for w in allow_list if w and isinstance(w, str))
    if allow:
        try:
            sp.word_frequency.load_words(allow)
        except Exception:
            pass

    for idx, page in enumerate(pages, start=1):
        if not page:
            continue
        if len(page) > 20000:
            # avoid heavy work on huge pages
            continue
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", page)
        tokens = tokens[:800]  # cap
        for w in tokens:
            wl = w.lower()
            if wl in allow:
                continue
            try:
                cand = sp.candidates(wl) or set()
                sug = next(iter(cand)) if cand else None
            except Exception:
                sug = None
            out.append(
                dict(
                    rule_name="Spelling",
                    severity="minor",
                    page=idx,
                    text_hit=w,
                    comment=f"Suspicious word '{w}'. Suggest: {sug or 'check'}",
                    bbox=None,
                    category="spelling",
                )
            )
    return out


# ----------------------------
# RULE CHECKS (text / regex)
# ----------------------------
def applies(scope: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    """Scope matching with AND semantics."""
    if not scope:
        return True
    for k, v in scope.items():
        mv = (meta.get(k) or "").strip().lower()
        vv = (v or "").strip().lower()
        if vv and mv != vv:
            return False
    return True


def text_hit_positions(page: fitz.Page, needle: str) -> List[fitz.Rect]:
    rects: List[fitz.Rect] = []
    try:
        inst = page.search_for(needle, quads=False, hit_max=50)
        rects.extend(inst or [])
    except Exception:
        pass
    return rects


def run_rule_checks(pages: List[str], doc: fitz.Document, meta: Dict[str, Any], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    pack = rules.get("rules", [])
    for r in pack:
        rname = r.get("name", "Unnamed")
        severity = r.get("severity", "minor")
        scope = r.get("scope", {})
        evidence = r.get("evidence", [])
        reject_if = r.get("reject_if", [])
        category = r.get("category", "rule")

        if not applies(scope, meta):
            continue

        # Basic text/regex evidence engine
        ok_hits = []
        for ev in evidence:
            etype = ev.get("type", "text")
            page_hint = ev.get("page_hint")
            if etype == "text":
                q_any = [q for q in ev.get("query_any", []) if q]
                found = False
                pages_to_scan = range(len(pages))
                if isinstance(page_hint, int) and 1 <= page_hint <= len(pages):
                    pages_to_scan = [page_hint - 1]
                for pidx in pages_to_scan:
                    page_txt = pages[pidx] or ""
                    for q in q_any:
                        if q and q.lower() in page_txt.lower():
                            ok_hits.append((pidx + 1, q))
                            found = True
                            break
                    if found:
                        break
                if not found:
                    findings.append(
                        dict(
                            rule_name=rname,
                            severity=severity,
                            page=None,
                            text_hit=None,
                            comment=f"Required text not found: {q_any}",
                            bbox=None,
                            category=category,
                        )
                    )
            elif etype == "regex":
                pattern = ev.get("pattern")
                flags = re.IGNORECASE | re.MULTILINE
                pages_to_scan = range(len(pages))
                if isinstance(page_hint, int) and 1 <= page_hint <= len(pages):
                    pages_to_scan = [page_hint - 1]
                compiled = None
                try:
                    compiled = re.compile(pattern, flags)
                except Exception:
                    findings.append(
                        dict(
                            rule_name=rname,
                            severity="minor",
                            page=None,
                            text_hit=None,
                            comment=f"Invalid regex in rule '{rname}': {pattern}",
                            bbox=None,
                            category=category,
                        )
                    )
                if compiled:
                    found = False
                    for pidx in pages_to_scan:
                        m = compiled.search(pages[pidx] or "")
                        if m:
                            ok_hits.append((pidx + 1, m.group(0)))
                            found = True
                            break
                    if not found:
                        findings.append(
                            dict(
                                rule_name=rname,
                                severity=severity,
                                page=None,
                                text_hit=None,
                                comment=f"Pattern not found: {pattern}",
                                bbox=None,
                                category=category,
                            )
                        )

        # Anti-patterns
        for rej in reject_if:
            etype = rej.get("type", "text")
            page_hint = rej.get("page_hint")
            if etype == "text":
                q_any = [q for q in rej.get("query_any", []) if q]
                pages_to_scan = range(len(pages))
                if isinstance(page_hint, int) and 1 <= page_hint <= len(pages):
                    pages_to_scan = [page_hint - 1]
                for pidx in pages_to_scan:
                    page_txt = pages[pidx] or ""
                    for q in q_any:
                        if q and q.lower() in page_txt.lower():
                            # Found an explicit rejection hit
                            findings.append(
                                dict(
                                    rule_name=rname,
                                    severity="major",
                                    page=pidx + 1,
                                    text_hit=q,
                                    comment=f"Prohibited text found: '{q}'",
                                    bbox=None,
                                    category=category,
                                )
                            )
            elif etype == "regex":
                pattern = rej.get("pattern")
                flags = re.IGNORECASE | re.MULTILINE
                try:
                    compiled = re.compile(pattern, flags)
                except Exception:
                    continue
                pages_to_scan = range(len(pages))
                if isinstance(page_hint, int) and 1 <= page_hint <= len(pages):
                    pages_to_scan = [page_hint - 1]
                for pidx in pages_to_scan:
                    m = compiled.search(pages[pidx] or "")
                    if m:
                        findings.append(
                            dict(
                                rule_name=rname,
                                severity="major",
                                page=pidx + 1,
                                text_hit=m.group(0),
                                comment=f"Prohibited pattern matched: {pattern}",
                                bbox=None,
                                category=category,
                            )
                        )

        # If evidence had explicit ok_hits, add informational trace so you see where it was satisfied
        for pno, hit in ok_hits[:6]:  # cap noise
            findings.append(
                dict(
                    rule_name=rname,
                    severity="info",
                    page=pno,
                    text_hit=hit,
                    comment="Evidence satisfied",
                    bbox=None,
                    category=category,
                )
            )

    return findings


def run_checks(
    pages: List[str],
    doc: fitz.Document,
    meta: Dict[str, Any],
    rules: Dict[str, Any],
    do_spelling: bool,
    allow_words: List[str],
) -> List[Dict[str, Any]]:
    f: List[Dict[str, Any]] = []
    f.extend(run_rule_checks(pages, doc, meta, rules))
    if do_spelling:
        f.extend(spelling_findings(pages, allow_words))
    return f


# ----------------------------
# ANNOTATION & EXPORTS
# ----------------------------
def annotate_pdf(original_doc: fitz.Document, findings: List[Dict[str, Any]]) -> bytes:
    # Work on a copy of the open doc
    doc = fitz.open()  # new
    doc.insert_pdf(original_doc)
    for row in findings:
        pno = row.get("page")
        if not pno or not isinstance(pno, int):
            continue
        if pno < 1 or pno > doc.page_count:
            continue
        page = doc.load_page(pno - 1)
        text_hit = row.get("text_hit") or ""
        comment = f"[{row.get('severity','')}] {row.get('rule_name','Rule')}: {row.get('comment','')}"
        used_mark = False
        if text_hit:
            rects = text_hit_positions(page, str(text_hit))
            for r in rects[:1]:  # mark first occurrence
                try:
                    annot = page.add_highlight_annot(r)
                    annot.set_info(title="Audit", content=comment)
                    used_mark = True
                    break
                except Exception:
                    pass
        if not used_mark:
            # Drop a sticky note near the top-left
            try:
                xy = fitz.Point(36, 36)
                note = page.add_text_annot(xy, comment, icon="Comment")
                note.set_info(title="Audit")
            except Exception:
                pass
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def make_excel(findings: List[Dict[str, Any]], meta: Dict[str, Any], pdf_name: str, status: str) -> bytes:
    df = pd.DataFrame(findings or [])
    if df.empty:
        df = pd.DataFrame(columns=["rule_name", "severity", "page", "text_hit", "comment", "category"])
    meta_row = pd.DataFrame([meta])
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Findings", index=False)
        meta_row.to_excel(xw, sheet_name="Meta", index=False)
        pd.DataFrame([{"status": status, "pdf_name": pdf_name}]).to_excel(xw, sheet_name="Summary", index=False)
    return out.getvalue()


# ----------------------------
# UI HELPERS
# ----------------------------
def inject_logo(logo_path: str, width_px: int = 140):
    if not logo_path or not os.path.exists(logo_path):
        return
    try:
        b64 = base64.b64encode(open(logo_path, "rb").read()).decode("utf-8")
        st.markdown(
            f"""
<style>
.top-left-logo {{
    position: fixed;
    top: 12px;
    left: 12px;
    z-index: 9999;
}}
</style>
<div class="top-left-logo">
    <img src="data:image;base64,{b64}" width="{width_px}">
</div>
""",
            unsafe_allow_html=True,
        )
    except Exception:
        pass


def gate():
    st.title("AI Design Quality Auditor")
    st.caption("Secure access")
    pw = st.text_input("Enter access password", type="password")
    if st.button("Enter"):
        if pw == ENTRY_PASSWORD:
            st.session_state["gate_ok"] = True
        else:
            st.error("Wrong password.")


def picklist_from_rules_or_default(rules: Dict[str, Any], key: str, default: List[str]) -> List[str]:
    ui = rules.get("ui", {})
    vals = ui.get(key)
    if isinstance(vals, list) and vals:
        return vals
    return default


def status_from_findings(findings: List[Dict[str, Any]]) -> Tuple[str, int, int]:
    minor = sum(1 for r in findings if r.get("severity") == "minor")
    major = sum(1 for r in findings if r.get("severity") == "major")
    if major > 0:
        return "REJECTED", minor, major
    # Let you configure stricter pass logic if desired
    return "PASS", minor, major


# ----------------------------
# TABS
# ----------------------------
def audit_tab(rules: Dict[str, Any]):
    st.header("Audit")

    # Load UI picklists
    suppliers = picklist_from_rules_or_default(rules, "supplier_options", DEFAULT_SUPPLIERS)
    clients = picklist_from_rules_or_default(rules, "client_options", DEFAULT_CLIENTS)
    projects = picklist_from_rules_or_default(rules, "project_options", DEFAULT_PROJECTS)
    site_types = picklist_from_rules_or_default(rules, "site_type_options", DEFAULT_SITE_TYPES)
    vendors = picklist_from_rules_or_default(rules, "vendor_options", DEFAULT_VENDORS)
    cab_locs = picklist_from_rules_or_default(rules, "cabinet_location_options", DEFAULT_CAB_LOC)
    radio_locs = picklist_from_rules_or_default(rules, "radio_location_options", DEFAULT_RADIO_LOC)
    drawing_types = picklist_from_rules_or_default(rules, "drawing_type_options", DEFAULT_DRAWING_TYPES)
    mimo_opts = picklist_from_rules_or_default(rules, "mimo_options", DEFAULT_MIMO_OPTIONS)

    with st.form("audit_meta_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            supplier = st.selectbox("Supplier", suppliers, index=0)
            client = st.selectbox("Client", clients, index=0)
            project = st.selectbox("Project", projects, index=0)
            drawing_type = st.selectbox("Drawing Type", drawing_types, index=1)
        with c2:
            site_type = st.selectbox("Site Type", site_types, index=0)
            vendor = st.selectbox("Vendor", vendors, index=0)
            cabinet_location = st.selectbox("Proposed Cabinet Location", cab_locs, index=0)
            radio_location = st.selectbox("Proposed Radio Location", radio_locs, index=0)
        with c3:
            sectors = st.number_input("Quantity of Sectors", min_value=1, max_value=6, step=1, value=3)
            site_address = st.text_input("Site Address (title must match unless contains ', 0 ,')", "")
            do_spell = st.checkbox("Enable spelling checks (slower)", value=False)
            exclude_from_analytics = st.checkbox("Exclude this run from analytics", value=False)

        # Hide MIMO when Power Resilience
        show_mimo = (project.strip().lower() != "power resilience")

        if show_mimo:
            st.subheader("Proposed MIMO Config")
            same_all = st.checkbox("Use S1 config for all sectors", value=True)
            s1 = st.selectbox("MIMO S1", mimo_opts, index=0)
            s_values = {"mimo_s1": s1}
            if not same_all:
                # Individual per sector
                for i in range(2, sectors + 1):
                    s_values[f"mimo_s{i}"] = st.selectbox(f"MIMO S{i}", mimo_opts, index=0, key=f"mimo_{i}")
                # Blank rest
                for i in range(sectors + 1, 7):
                    s_values[f"mimo_s{i}"] = ""
            else:
                # Copy S1
                for i in range(2, 7):
                    s_values[f"mimo_s{i}"] = s1 if i <= sectors else ""
        else:
            # Not used for Power Resilience
            s_values = {f"mimo_s{i}": "" for i in range(1, 7)}

        st.divider()
        pdf_file = st.file_uploader("Upload PDF design", type=["pdf"])
        allowlist = rules.get("allowlist", []) or []
        allow_words = [w for w in allowlist if isinstance(w, str)]

        submitted = st.form_submit_button("Run Audit", type="primary", use_container_width=True)

    if not submitted:
        return

    # Validate metadata completeness
    required_meta = {
        "supplier": supplier,
        "client": client,
        "project": project,
        "site_type": site_type,
        "vendor": vendor,
        "cabinet_location": cabinet_location,
        "radio_location": radio_location,
        "drawing_type": drawing_type,
        "site_address": site_address,
    }
    missing = [k for k, v in required_meta.items() if not str(v).strip()]
    if missing:
        st.error(f"Please complete all metadata fields: {', '.join(missing)}")
        return
    if not pdf_file:
        st.error("Please upload a PDF to audit.")
        return

    raw_pdf = pdf_file.read()
    try:
        pages, open_doc = read_pdf_text_and_pages(raw_pdf)
    except Exception as e:
        st.error("Failed to read PDF. Ensure the file is a valid PDF.")
        return

    # Site Address rule: must match title unless contains ', 0 ,'
    if ", 0 ," not in site_address:
        # Title guess = first non-empty line on page1
        title_guess = ""
        for line in (pages[0] or "").splitlines():
            if line.strip():
                title_guess = line.strip()
                break
        if title_guess and site_address.strip().lower() not in title_guess.lower():
            st.warning("Site Address does not appear to match the PDF title text on page 1. This will be flagged.")

    meta = dict(
        supplier=supplier,
        client=client,
        project=project,
        site_type=site_type,
        vendor=vendor,
        cabinet_location=cabinet_location,
        radio_location=radio_location,
        drawing_type=drawing_type,
        sectors=int(sectors),
        site_address=site_address,
        **s_values,
    )

    with st.spinner("Running checks..."):
        findings = run_checks(pages, open_doc, meta, rules, do_spell, allow_words)

        # Add the site-address check as a rule-like finding
        if ", 0 ," not in site_address:
            title_text = pages[0] or ""
            if site_address.strip() and site_address.strip().lower() not in title_text.lower():
                findings.append(
                    dict(
                        rule_name="Title vs Site Address",
                        severity="major",
                        page=1,
                        text_hit=site_address.strip(),
                        comment="Site Address does not match title text on page 1.",
                        bbox=None,
                        category="metadata",
                    )
                )

        status, minor_count, major_count = status_from_findings(findings)

        excel_bytes = make_excel(findings, meta, pdf_file.name, status)
        pdf_annot_bytes = annotate_pdf(open_doc, findings)

    # Keep results in session (don’t disappear on download)
    st.session_state["last_findings_df"] = pd.DataFrame(findings or [])
    st.session_state["last_excel"] = excel_bytes
    st.session_state["last_pdf"] = pdf_annot_bytes
    st.session_state["last_meta"] = meta
    st.session_state["last_pdf_name"] = pdf_file.name
    st.session_state["last_status"] = status
    st.session_state["last_minor"] = minor_count
    st.session_state["last_major"] = major_count

    st.success(f"Audit complete: **{status}**  |  Minor: {minor_count}  Major: {major_count}")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download Excel Report",
            data=excel_bytes,
            file_name=f"{os.path.splitext(pdf_file.name)[0]}_{status}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download Annotated PDF",
            data=pdf_annot_bytes,
            file_name=f"{os.path.splitext(pdf_file.name)[0]}_{status}_{datetime.now().strftime('%Y%m%d')}_ANNOTATED.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # Save to history
    row = dict(
        timestamp_utc=now_utc().isoformat(),
        status=status,
        minor_count=minor_count,
        major_count=major_count,
        pdf_name=os.path.splitext(pdf_file.name)[0] + f"_{status}.pdf",
        excel_name=os.path.splitext(pdf_file.name)[0] + f"_{status}.xlsx",
        exclude=bool(exclude_from_analytics),
        **meta,
    )
    try:
        save_history_row(row)
    except Exception:
        st.warning("Could not append to history (file busy or corrupted).")


def training_tab(rules: Dict[str, Any]):
    st.header("Training")
    st.caption("Confirm VALID / NOT VALID, add quick rules, and extend the allowlist.")

    # Password for changing rules
    with st.expander("Unlock rule updates"):
        pw = st.text_input("Rules password", type="password", placeholder="Enter to enable saving")
        can_write = pw == SETTINGS_PASSWORD
        if can_write:
            st.success("Rule editing unlocked.")
        else:
            st.info("Enter the rules password to enable saving changes.")

    # Upload a previously downloaded Excel report to bulk-mark VALID/NOT VALID (optional)
    st.subheader("Re-upload a Report to Train")
    up = st.file_uploader("Upload Excel audit report (.xlsx) with your markings", type=["xlsx"], key="train_up")
    if up:
        try:
            xls = pd.ExcelFile(up)
            df_find = xls.parse("Findings")
            st.dataframe(df_find, use_container_width=True)
            st.info("Use this as reference to add rules below. (Auto-learning from Excel will come in V2.)")
        except Exception:
            st.error("Could not read the Excel report. Make sure it has a 'Findings' sheet.")

    st.subheader("Quick Allowlist")
    new_allow = st.text_input("Add allowable words (comma-separated)", "")
    if st.button("Append to allowlist"):
        if new_allow.strip():
            allows = rules.get("allowlist", []) or []
            for w in [x.strip() for x in new_allow.split(",") if x.strip()]:
                if w not in allows:
                    allows.append(w)
            rules["allowlist"] = allows
            save_rules(rules)
            st.success("Allowlist updated.")
        else:
            st.warning("Nothing to add.")

    st.subheader("Add a Quick Rule")
    with st.form("add_rule"):
        name = st.text_input("Rule name *")
        severity = st.selectbox("Severity", ["minor", "major"], index=1)
        col1, col2, col3 = st.columns(3)
        with col1:
            project = st.text_input("Scope: project", "")
            client = st.text_input("Scope: client", "")
        with col2:
            vendor = st.text_input("Scope: vendor", "")
            site_type = st.text_input("Scope: site_type", "")
        with col3:
            drawing_type = st.text_input("Scope: drawing_type", "")
            radio_location = st.text_input("Scope: radio_location", "")
        query_any = st.text_area("Text must contain ANY of (comma-separated)", "")
        page_hint = st.number_input("Preferred page (0 for any)", min_value=0, max_value=9999, value=0)
        reject_any = st.text_area("Prohibited text (comma-separated)", "")

        submitted = st.form_submit_button("Add Rule")
        if submitted:
            if not can_write:
                st.error("Enter rules password above to save.")
            elif not name.strip():
                st.error("Rule name is required.")
            else:
                newr = dict(
                    name=name.strip(),
                    severity=severity,
                    scope={
                        **({"project": project.strip()} if project.strip() else {}),
                        **({"client": client.strip()} if client.strip() else {}),
                        **({"vendor": vendor.strip()} if vendor.strip() else {}),
                        **({"site_type": site_type.strip()} if site_type.strip() else {}),
                        **({"drawing_type": drawing_type.strip()} if drawing_type.strip() else {}),
                        **({"radio_location": radio_location.strip()} if radio_location.strip() else {}),
                    },
                    evidence=[
                        {
                            "type": "text",
                            "query_any": [x.strip() for x in query_any.split(",") if x.strip()],
                            **({"page_hint": int(page_hint)} if page_hint > 0 else {}),
                        }
                    ],
                    reject_if=[
                        {"type": "text", "query_any": [x.strip() for x in reject_any.split(",") if x.strip()]}
                    ]
                    if reject_any.strip()
                    else [],
                    category="custom",
                )
                rlist = rules.get("rules", []) or []
                rlist.append(newr)
                rules["rules"] = rlist
                save_rules(rules)
                st.success("Rule added.")


def analytics_tab():
    st.header("Analytics")

    dfh = load_history()
    if dfh.empty:
        st.info("No history yet.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        hours = st.number_input("Show last N hours", min_value=1, max_value=24 * 60, value=DEFAULT_UI_VISIBILITY_HOURS)
    with c2:
        f_client = st.multiselect("Client filter", sorted(dfh["client"].dropna().unique().tolist()))
    with c3:
        f_project = st.multiselect("Project filter", sorted(dfh["project"].dropna().unique().tolist()))
    with c4:
        f_supplier = st.multiselect("Supplier filter", sorted(dfh["supplier"].dropna().unique().tolist()))

    show = filter_history_for_ui(dfh.copy(), int(hours))
    show = show[show.get("exclude", False) != True]

    if f_client:
        show = show[show["client"].isin(f_client)]
    if f_project:
        show = show[show["project"].isin(f_project)]
    if f_supplier:
        show = show[show["supplier"].isin(f_supplier)]

    # Summary KPIs
    total = len(show)
    passes = int((show["status"] == "PASS").sum()) if total else 0
    rejects = int((show["status"] == "REJECTED").sum()) if total else 0
    rft = (passes / total * 100.0) if total else 0.0

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Audits", total)
    k2.metric("Right First Time %", f"{rft:.1f}%")
    k3.metric("Rejected", rejects)

    # Table
    cols = [
        "timestamp_utc",
        "supplier",
        "client",
        "project",
        "status",
        "pdf_name",
        "excel_name",
        "minor_count",
        "major_count",
    ]
    for c in cols:
        if c not in show.columns:
            show[c] = ""
    st.dataframe(show[cols].sort_values("timestamp_utc", ascending=False), use_container_width=True, height=380)


def settings_tab(rules: Dict[str, Any]):
    st.header("Settings")
    with st.expander("UI"):
        logo_path = st.text_input(
            "Logo path (file in repo root)",
            rules.get("ui", {}).get("logo_path", "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png"),
        )
        width = st.number_input("Logo width (px)", min_value=60, max_value=360, value=140, step=10)
        st.session_state["logo_path"] = logo_path
        st.session_state["logo_width"] = width

    with st.expander("Pick-list Management (requires rules password)"):
        pw = st.text_input("Rules password", type="password", key="settings_pw")
        can_write = pw == SETTINGS_PASSWORD
        if not can_write:
            st.info("Enter rules password to save any changes below.")

        ui = rules.get("ui", {})
        def _edit_list(label: str, key: str, default: List[str]) -> None:
            current = ui.get(key, default)
            txt = st.text_area(label, "\n".join(current))
            if st.button(f"Save {label}"):
                if can_write:
                    new_vals = [x.strip() for x in txt.splitlines() if x.strip()]
                    if "ui" not in rules:
                        rules["ui"] = {}
                    rules["ui"][key] = new_vals
                    save_rules(rules)
                    st.success(f"{label} saved.")
                else:
                    st.error("Rules password required.")

        _edit_list("Supplier options", "supplier_options", DEFAULT_SUPPLIERS)
        _edit_list("Client options", "client_options", DEFAULT_CLIENTS)
        _edit_list("Project options", "project_options", DEFAULT_PROJECTS)
        _edit_list("Site type options", "site_type_options", DEFAULT_SITE_TYPES)
        _edit_list("Vendor options", "vendor_options", DEFAULT_VENDORS)
        _edit_list("Cabinet location options", "cabinet_location_options", DEFAULT_CAB_LOC)
        _edit_list("Radio location options", "radio_location_options", DEFAULT_RADIO_LOC)
        _edit_list("Drawing type options", "drawing_type_options", DEFAULT_DRAWING_TYPES)
        _edit_list("MIMO options", "mimo_options", DEFAULT_MIMO_OPTIONS)

    with st.expander("Allowlist (words ignored by spelling)"):
        allows = rules.get("allowlist", []) or []
        txt = st.text_area("Allowlist words (one per line)", "\n".join(allows))
        if st.button("Save allowlist"):
            rules["allowlist"] = [x.strip() for x in txt.splitlines() if x.strip()]
            save_rules(rules)
            st.success("Allowlist saved.")

    st.caption("Settings changes write to rules_example.yaml (under the 'ui' and 'allowlist' sections).")


# ----------------------------
# MAIN
# ----------------------------
def main():
    st.set_page_config(page_title="AI Design Quality Auditor", layout="wide")
    ensure_dirs()

    # Inject logo
    rules = load_rules()
    logo_path = st.session_state.get("logo_path") or rules.get("ui", {}).get("logo_path") or ""
    inject_logo(logo_path, width_px=int(st.session_state.get("logo_width", 140)))

    if not st.session_state.get("gate_ok"):
        gate()
        return

    tabs = st.tabs(["🔎 Audit", "🧠 Training", "📈 Analytics", "⚙️ Settings"])
    with tabs[0]:
        audit_tab(rules)
        # If a report is in session, show a compact preview table under the buttons
        df_last = st.session_state.get("last_findings_df")
        if isinstance(df_last, pd.DataFrame) and not df_last.empty:
            st.subheader("Latest Findings")
            st.dataframe(df_last[["rule_name", "severity", "page", "text_hit", "comment"]], use_container_width=True, height=300)
    with tabs[1]:
        training_tab(rules)
    with tabs[2]:
        analytics_tab()
    with tabs[3]:
        settings_tab(rules)


if __name__ == "__main__":
    main()
