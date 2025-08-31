# app.py
# AI Design Quality Auditor — full version
# Streamlit app with metadata-gated rules, drawing/page-aware checks, training loop,
# PDF annotation, Excel export, history, analytics, and settings.

import io
import os
import re
import json
import base64
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import yaml

# Text search / similarity
from rapidfuzz import process, fuzz

# PDF (no poppler needed)
import fitz  # PyMuPDF

# -------------------------
# CONSTANTS / CONFIG
# -------------------------
APP_TITLE = "AI Design Quality Auditor"
DEFAULT_RULES_FILE = "rules_example.yaml"
HISTORY_FILE = "audit_history.csv"
DAILY_EXPORT_DIR = "daily_exports"  # optional dump location (if you decide to cron outside)
ENTRY_PASSWORD = "Seker123"         # gate to enter app
RULES_PASSWORD = "vanB3lkum21"      # gate to edit/upload YAML rules
LOGO_FILE_FALLBACK = ""             # leave blank by default; set in Settings
MAX_SECTORS = 6

DEFAULT_CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
DEFAULT_PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
DEFAULT_SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
DEFAULT_VENDORS = ["Ericsson", "Nokia"]
DEFAULT_CAB_LOC = ["Indoor", "Outdoor"]
DEFAULT_RADIO_LOCATIONS = ["Low Level", "Midway", "High Level", "Unique Coverage"]
DEFAULT_SUPPLIERS_FALLBACK = ["WFS", "Circet", "Telent", "AlanDick", "Cellnex", "Cornerstone", "BT", "Vodafone", "MBNL", "H3G", "BTEE"]

# MIMO options (you can also drive these from YAML later if you prefer)
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
    "(blank)"
]

SEVERITY_ORDER = {"Major": 2, "Minor": 1}

# -------------------------
# UTILITIES
# -------------------------
def load_rules(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # normalize keys used elsewhere
    data.setdefault("allowlist", [])
    data.setdefault("mappings", {})
    data.setdefault("phrase_rules", [])
    data.setdefault("drawing_rules", [])
    data.setdefault("suppliers", [])
    return data

def save_rules(path: str, rules: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(rules, f, sort_keys=False, allow_unicode=True)

def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE):
        cols = [
            "timestamp_utc","supplier","client","project","site_type","vendor",
            "cabinet_location","radio_location","sectors","mimo_s1","mimo_s2","mimo_s3",
            "mimo_s4","mimo_s5","mimo_s6","use_s1_for_all","site_address","status",
            "pdf_name","excel_name","exclude","rft_pass","errors_major","errors_minor"
        ]
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(HISTORY_FILE)
        return df
    except Exception:
        # if corrupted, back up and start fresh
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.rename(HISTORY_FILE, f"{HISTORY_FILE}.broken_{ts}")
        return load_history()

def append_history(row: dict) -> None:
    df = load_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

def logo_html(logo_path: str) -> str:
    if not logo_path or not os.path.exists(logo_path):
        return ""
    # scale to a tidy height and pin top-left
    with open(logo_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"""
    <style>
      .top-left-logo {{
        position: fixed;
        top: 8px;
        left: 8px;
        z-index: 9999;
        height: 42px;
      }}
      /* tighten main padding so logo doesn't overlap content */
      section.main > div:first-child {{ padding-top: 56px; }}
    </style>
    <img src="data:image/png;base64,{b64}" class="top-left-logo" />
    """

def suppliers_from_rules_or_history(rules: dict) -> List[str]:
    dfh = load_history()
    seen = set()
    if "supplier" in dfh.columns:
        seen |= set([s for s in dfh["supplier"].dropna().astype(str).tolist() if s.strip()])
    seen |= set(rules.get("suppliers", []))
    if not seen:
        seen = set(DEFAULT_SUPPLIERS_FALLBACK)
    return sorted(seen)

# -------------------------
# PDF / TEXT
# -------------------------
def pdf_pages_text_and_words(pdf_bytes: bytes):
    """
    Returns:
      pages_text: List[str]
      pages_words: List[List[dict]] where dict has keys text, bbox=(x0,y0,x1,y1)
    """
    pages_text = []
    pages_words = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text("text")
            pages_text.append(text or "")
            # words list tuples: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            raw_words = page.get_text("words") or []
            words = []
            for w in raw_words:
                x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
                words.append({"text": word, "bbox": (x0, y0, x1, y1)})
            pages_words.append(words)
    return pages_text, pages_words

def search_bboxes_on_page(page: fitz.Page, needle: str) -> List[tuple]:
    """Find rectangles for occurrences of 'needle' on page (case-insensitive)."""
    rects = []
    if not needle or not isinstance(needle, str):
        return rects
    try:
        rects = page.search_for(needle, flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_IGNORECASE)
    except Exception:
        # fallback simple
        rects = page.search_for(needle)
    return [ (r.x0, r.y0, r.x1, r.y1) for r in rects ]

# -------------------------
# AUDIT CHECKS
# -------------------------
def normalize_word(w: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-_/\.]", "", (w or "")).strip()

def spelling_checks(pages_text: List[str], rules: dict) -> List[dict]:
    allow = set([normalize_word(a).lower() for a in rules.get("allowlist", []) if isinstance(a, str)])
    mappings: Dict[str, str] = {str(k).lower(): str(v) for k, v in rules.get("mappings", {}).items()}
    findings = []

    # naive tokenization per page
    for pno, txt in enumerate(pages_text, start=1):
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-/_.]{1,}", txt)
        for t in tokens:
            low = t.lower()
            if low in allow:
                continue
            # mapping (auto-correct)
            if low in mappings:
                findings.append({
                    "page": pno, "bbox": None,
                    "rule_id": "SPELL_MAPPING",
                    "message": f"'{t}' should be '{mappings[low]}'",
                    "severity": "Minor", "category": "Spelling"
                })
                continue
            # optional suggestion using rapidfuzz against allowlist (only if allowlist non-empty)
            if allow:
                sug, score, _ = process.extractOne(low, list(allow), scorer=fuzz.ratio)  # type: ignore
                if score >= 92:
                    continue  # close enough to allowed
                elif score >= 70:
                    findings.append({
                        "page": pno, "bbox": None,
                        "rule_id": "SPELL_SUGGEST",
                        "message": f"Possible misspelling: '{t}'. Did you mean '{sug}'?",
                        "severity": "Minor", "category": "Spelling"
                    })
    return findings

def phrase_rules_checks(pages_text: List[str], rules: dict, meta: dict) -> List[dict]:
    """
    phrase_rules:
      - id: REPLACE_XY
        when: {project: "RAN", vendor: "Ericsson"}  # all must match
        any_page:
          require_any:
            - "Brush"
          forbid_any:
            - "Generator Power"
        message: "Brush cannot appear with Generator Power."
        severity: "Major"
    """
    out = []
    prules = rules.get("phrase_rules", [])
    for r in prules:
        rid = r.get("id", "PHRASE_RULE")
        when = r.get("when", {})
        msg = r.get("message", "Phrase rule violation.")
        sev = r.get("severity", "Major")

        # metadata gate
        gated = True
        for k, v in when.items():
            if str(meta.get(k, "")).strip() != str(v).strip():
                gated = False
                break
        if not gated:
            continue

        block = r.get("any_page", {})
        req_any = [re.compile(rx, re.I) for rx in block.get("require_any", [])]
        forbid_any = [re.compile(rx, re.I) for rx in block.get("forbid_any", [])]

        # evaluate per page
        for pno, txt in enumerate(pages_text, start=1):
            ok_req = True if not req_any else any(rx.search(txt or "") for rx in req_any)
            bad_forbid = any(rx.search(txt or "") for rx in forbid_any)
            if ok_req and bad_forbid:
                out.append({
                    "page": pno, "bbox": None, "rule_id": rid,
                    "message": msg, "severity": sev, "category": "Phrase"
                })
    return out

def find_pages_matching(hint_regex: str, pages_text: List[str]) -> List[int]:
    out = []
    pat = re.compile(hint_regex, re.IGNORECASE)
    for i, txt in enumerate(pages_text, start=1):
        if pat.search(txt or ""):
            out.append(i)
    return out

def drawing_rules_checks(pages_text: List[str], rules: dict, meta: dict) -> List[dict]:
    """
    drawing_rules:
      - id: POWERRES_ELTEK_POLARADIUM_NOTE_300
        when: { project: "Power Resilience" }
        page_hint: "Drawing\\s*300|DRG\\s*300|D300\\b"
        require_all:
          - "(?i)IMPORTANT NOTE:"
          - "(?i)Eltek\\s*PSU"
          - "(?i)TDEE53201\\s*section\\s*3\\.8\\.1"
          - "(?i)Polaradium"
        message: "Power Resilience: mandatory IMPORTANT NOTE not present on Drawing 300."
        severity: "Major"
    """
    out = []
    drules = rules.get("drawing_rules", [])
    for r in drules:
        rid = r.get("id", "DRAWING_RULE")
        when = r.get("when", {})
        msg = r.get("message", "Drawing rule not satisfied.")
        sev = r.get("severity", "Major")
        page_hint = r.get("page_hint", "")
        req_all = [re.compile(rx, re.I) for rx in r.get("require_all", [])]

        # gate on metadata
        gated = True
        for k, v in when.items():
            if str(meta.get(k, "")).strip() != str(v).strip():
                gated = False
                break
        if not gated:
            continue

        cand_pages = list(range(1, len(pages_text) + 1))
        if page_hint:
            hinted = find_pages_matching(page_hint, pages_text)
            if hinted:
                cand_pages = hinted

        satisfied = False
        for pno in cand_pages:
            txt = pages_text[pno - 1] if 1 <= pno <= len(pages_text) else ""
            if all(rx.search(txt or "") for rx in req_all):
                satisfied = True
                break

        if not satisfied:
            firstp = cand_pages[0] if cand_pages else 1
            out.append({
                "page": firstp, "bbox": None, "rule_id": rid,
                "message": msg, "severity": sev, "category": "Drawing"
            })

    return out

def address_title_match_check(pages_text: List[str], meta: dict) -> List[dict]:
    """
    Rule: Site Address must appear in the PDF when not '..., 0, ...'.
    If address contains ', 0 ,' pattern, we ignore the check (as requested).
    """
    addr = (meta.get("site_address") or "").strip()
    if not addr:
        return []
    if re.search(r",\s*0\s*,", addr):
        return []  # ignore special case

    # naive check: must appear at least once (case-insensitive, ignore multiple spaces/commas)
    norm = re.sub(r"\s+", " ", addr).strip().lower()
    norm = norm.replace(",", " ").replace("  ", " ")
    rx = re.compile(re.escape(norm), re.I)

    pages_norm = []
    for txt in pages_text:
        t = re.sub(r"\s+", " ", (txt or "").lower())
        t = t.replace(",", " ").replace("  ", " ")
        pages_norm.append(t)

    for pno, norm_txt in enumerate(pages_norm, start=1):
        if rx.search(norm_txt):
            return []
    return [{
        "page": 1,
        "bbox": None,
        "rule_id": "ADDR_TITLE_MISMATCH",
        "message": "Site Address not found in PDF text (strict match).",
        "severity": "Major",
        "category": "Metadata"
    }]

def run_audit(pdf_bytes: bytes, meta: dict, rules: dict) -> List[dict]:
    pages_text, pages_words = pdf_pages_text_and_words(pdf_bytes)
    findings: List[dict] = []

    # 1) spelling / mappings
    findings += spelling_checks(pages_text, rules)

    # 2) phrase rules (Brush v Generator Power, etc.)
    findings += phrase_rules_checks(pages_text, rules, meta)

    # 3) address/title consistency (skip when ", 0 ," present)
    findings += address_title_match_check(pages_text, meta)

    # 4) drawing/page-aware rules
    findings += drawing_rules_checks(pages_text, rules, meta)

    # enrich with bbox where we can (search the page for key tokens from message)
    findings = attach_bboxes(pdf_bytes, findings)

    # sort: Major first, then Minor
    findings.sort(key=lambda r: (-SEVERITY_ORDER.get(r.get("severity", "Minor"), 0), r.get("page", 0)))
    return findings

def attach_bboxes(pdf_bytes: bytes, findings: List[dict]) -> List[dict]:
    # simple heuristic: extract quoted tokens from message and search
    token_rx = re.compile(r"'([^']+)'|\"([^\"]+)\"")
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for f in findings:
            pno = int(f.get("page", 1))
            if not (1 <= pno <= len(doc)):
                continue
            page = doc[pno - 1]
            msg = str(f.get("message", ""))

            needles = []
            for m in token_rx.finditer(msg):
                grp = m.group(1) or m.group(2)
                if grp:
                    needles.append(grp)

            rects = []
            for n in needles:
                rects += search_bboxes_on_page(page, n)

            f["bbox"] = rects[0] if rects else None
            f["bboxes"] = rects  # keep all rects for annotation
    return findings

# -------------------------
# OUTPUTS
# -------------------------
def annotate_pdf(pdf_bytes: bytes, findings: List[dict]) -> bytes:
    """Draw rectangles & sticky notes at finding locations."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for f in findings:
        pno = int(f.get("page", 1))
        if not (1 <= pno <= len(doc)):
            continue
        page = doc[pno - 1]
        note = f"[{f.get('severity','Minor')}] {f.get('message','')}"
        bboxes = f.get("bboxes") or []
        if bboxes:
            for (x0, y0, x1, y1) in bboxes:
                rect = fitz.Rect(x0, y0, x1, y1)
                page.add_rect_annot(rect)
                page.add_text_annot(rect.tl, note)
        else:
            # fallback: place a note near top-left
            page.add_text_annot(fitz.Point(36, 36), note)
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()

def make_excel(findings: List[dict], meta: dict, original_pdf_name: str, status: str) -> bytes:
    rows = []
    for f in findings:
        rows.append({
            "page": f.get("page"),
            "severity": f.get("severity"),
            "category": f.get("category"),
            "rule_id": f.get("rule_id"),
            "message": f.get("message"),
        })
    df = pd.DataFrame(rows)
    meta_df = pd.DataFrame([meta])

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        meta_df.to_excel(xw, sheet_name="Meta", index=False)
        (df if not df.empty else pd.DataFrame(columns=["page","severity","category","rule_id","message"])) \
            .to_excel(xw, sheet_name="Findings", index=False)
        # summary
        summ = {
            "pdf_name": original_pdf_name,
            "status": status,
            "errors_major": int(sum(1 for r in findings if r.get("severity") == "Major")),
            "errors_minor": int(sum(1 for r in findings if r.get("severity") == "Minor")),
            "generated_utc": ts
        }
        pd.DataFrame([summ]).to_excel(xw, sheet_name="Summary", index=False)
    return out.getvalue()

# -------------------------
# UI HELPERS
# -------------------------
def gate_with_password() -> bool:
    st.info("Enter access password to use the tool.")
    pwd = st.text_input("Access Password", type="password", key="entry_pw")
    return (pwd == ENTRY_PASSWORD)

def rules_edit_allowed() -> bool:
    pwd = st.text_input("Rules Update Password", type="password", key="rules_pw")
    return (pwd == RULES_PASSWORD)

def meta_form(rules: dict) -> dict:
    st.subheader("Audit Metadata (required)")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c7, c8 = st.columns(2)

    with c1:
        supplier_opts = suppliers_from_rules_or_history(rules)
        supplier = st.selectbox("Supplier", options=supplier_opts)

    with c2:
        client = st.selectbox("Client", options=rules.get("clients", DEFAULT_CLIENTS))

    with c3:
        project = st.selectbox("Project", options=rules.get("projects", DEFAULT_PROJECTS))

    with c4:
        site_type = st.selectbox("Site Type", options=rules.get("site_types", DEFAULT_SITE_TYPES))

    with c5:
        vendor = st.selectbox("Proposed Vendor", options=rules.get("vendors", DEFAULT_VENDORS))

    with c6:
        cabinet_location = st.selectbox("Proposed Cabinet Location", options=rules.get("cabinet_location", DEFAULT_CAB_LOC))

    with c7:
        radio_location = st.selectbox("Proposed Radio Location", options=rules.get("radio_locations", DEFAULT_RADIO_LOCATIONS))

    with c8:
        sectors = st.selectbox("Quantity of Sectors", options=list(range(1, MAX_SECTORS + 1)), index=2)

    # MIMO per sector (hide when Power Resilience)
    mimo_data = {}
    use_s1_for_all = False
    if project != "Power Resilience":
        st.markdown("### Proposed MIMO Config (by sector)")
        c_m1, c_m2 = st.columns([2, 1])
        with c_m2:
            use_s1_for_all = st.checkbox("Use S1 for all sectors", value=False)
        mimo_opts = rules.get("mimo_options", DEFAULT_MIMO_OPTIONS)
        # sector 1
        mimo_data["mimo_s1"] = st.selectbox("MIMO S1", mimo_opts, key="mimo_s1")
        for i in range(2, sectors + 1):
            if use_s1_for_all:
                st.text_input(f"MIMO S{i}", mimo_data["mimo_s1"], key=f"mimo_s{i}", disabled=True)
                mimo_data[f"mimo_s{i}"] = mimo_data["mimo_s1"]
            else:
                mimo_data[f"mimo_s{i}"] = st.selectbox(f"MIMO S{i}", mimo_opts, key=f"mimo_s{i}")
        # pad remaining to 6
        for i in range(sectors + 1, MAX_SECTORS + 1):
            mimo_data[f"mimo_s{i}"] = ""
    else:
        # PR optional MIMO (hide controls)
        for i in range(1, MAX_SECTORS + 1):
            mimo_data[f"mimo_s{i}"] = ""
        st.caption("Proposed MIMO Config (optional for Power Resilience)")

    site_address = st.text_input("Site Address (exact text should appear in PDF; ignored if contains ', 0 ,')")

    meta = {
        "supplier": supplier, "client": client, "project": project, "site_type": site_type,
        "vendor": vendor, "cabinet_location": cabinet_location, "radio_location": radio_location,
        "sectors": int(sectors), "use_s1_for_all": bool(use_s1_for_all),
        "site_address": site_address
    }
    meta.update(mimo_data)
    return meta

def meta_complete(meta: dict, require_mimo: bool) -> bool:
    needed = ["supplier","client","project","site_type","vendor","cabinet_location","radio_location","site_address"]
    for k in needed:
        if not str(meta.get(k,"")).strip():
            return False
    if require_mimo:
        if not str(meta.get("mimo_s1","")).strip():
            return False
    return True

# -------------------------
# TRAINING
# -------------------------
def training_tab(rules: dict):
    st.subheader("Rapid Training (Not Valid / Valid & Rule Growth)")
    st.caption("Use this area to push allowlist/mappings fast and to add structured rules without editing code.")

    st.markdown("#### Upload Findings Excel (from a previous audit) to mark Valid/Not-Valid")
    fup = st.file_uploader("Findings Excel (.xlsx)", type=["xlsx"], key="train_findings")
    if fup is not None:
        try:
            xls = pd.ExcelFile(fup)
            df_find = pd.read_excel(xls, "Findings")
            st.dataframe(df_find, use_container_width=True)
            st.info("Select rows below and click buttons to update allowlist or mappings.")
            # Simple helpers (expects columns 'message' or 'rule_id')
        except Exception as e:
            st.error(f"Could not read Findings Excel: {e}")

    st.markdown("---")
    st.subheader("Upload Error Log (CSV/XLSX)")
    elog = st.file_uploader("Error Log", type=["csv","xlsx"], key="elog")
    if elog is not None:
        try:
            edf = pd.read_csv(elog) if elog.name.lower().endswith(".csv") else pd.read_excel(elog)
            st.dataframe(edf.head(200), use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("➕ Add allowlist from 'allow' column"):
                    rules.setdefault("allowlist", [])
                    seen = set([a.lower() for a in rules["allowlist"]])
                    for v in edf.get("allow", []):
                        if pd.notna(v):
                            val = str(v).strip()
                            if val and val.lower() not in seen:
                                rules["allowlist"].append(val)
                                seen.add(val.lower())
                    save_rules(DEFAULT_RULES_FILE, rules)
                    st.success("Allowlist updated.")
            with c2:
                if st.button("➕ Add mappings from 'wrong' → 'right'"):
                    rules.setdefault("mappings", {})
                    wrongs = edf.get("wrong", [])
                    rights = edf.get("right", [])
                    for w, r in zip(wrongs, rights):
                        if pd.notna(w) and pd.notna(r):
                            rules["mappings"][str(w).strip().lower()] = str(r).strip()
                    save_rules(DEFAULT_RULES_FILE, rules)
                    st.success("Mappings updated.")
        except Exception as e:
            st.error(f"Could not read Error Log: {e}")

    st.markdown("---")
    st.subheader("Add Phrase Rule (X cannot appear with Y, etc.)")
    with st.form("add_phrase_rule"):
        rid = st.text_input("Rule ID", "CUSTOM_PHRASE_RULE_1")
        when_project = st.text_input("Gate: project (leave blank to ignore)")
        when_vendor = st.text_input("Gate: vendor (leave blank to ignore)")
        req_any = st.text_area("Regex list: require ANY (one per line)", "Brush")
        forbid_any = st.text_area("Regex list: FORBID ANY (one per line)", "Generator Power")
        msg = st.text_input("Message", "Brush cannot appear with Generator Power.")
        sev = st.selectbox("Severity", ["Major","Minor"], index=0)
        submitted = st.form_submit_button("Add Rule")
        if submitted:
            rules.setdefault("phrase_rules", [])
            when = {}
            if when_project.strip(): when["project"] = when_project.strip()
            if when_vendor.strip(): when["vendor"] = when_vendor.strip()
            rules["phrase_rules"].append({
                "id": rid,
                "when": when,
                "any_page": {
                    "require_any": [l.strip() for l in req_any.splitlines() if l.strip()],
                    "forbid_any": [l.strip() for l in forbid_any.splitlines() if l.strip()]
                },
                "message": msg, "severity": sev
            })
            save_rules(DEFAULT_RULES_FILE, rules)
            st.success(f"Rule {rid} added.")

    st.markdown("---")
    st.subheader("Add Drawing/Page-Aware Rule")
    with st.form("add_drawing_rule"):
        rid = st.text_input("Rule ID", "CUSTOM_DRAWING_RULE_1")
        when_project = st.text_input("Gate: project (leave blank to ignore)", "Power Resilience")
        page_hint = st.text_input("Page hint (regex)", "Drawing\\s*300|DRG\\s*300|D300\\b")
        req_all = st.text_area("Require ALL regex (one per line)", "IMPORTANT NOTE:\nEltek\\s*PSU\nTDEE53201\\s*section\\s*3\\.8\\.1\nPolaradium")
        msg = st.text_input("Message", "Mandatory note not present on Drawing 300.")
        sev = st.selectbox("Severity ", ["Major","Minor"], index=0, key="draw_sev")
        submitted = st.form_submit_button("Add Drawing Rule")
        if submitted:
            rules.setdefault("drawing_rules", [])
            when = {}
            if when_project.strip(): when["project"] = when_project.strip()
            rules["drawing_rules"].append({
                "id": rid,
                "when": when,
                "page_hint": page_hint.strip(),
                "require_all": [l.strip() for l in req_all.splitlines() if l.strip()],
                "message": msg, "severity": sev
            })
            save_rules(DEFAULT_RULES_FILE, rules)
            st.success(f"Rule {rid} added.")

# -------------------------
# ANALYTICS
# -------------------------
def rft_status(errors_major: int, errors_minor: int) -> bool:
    return (int(errors_major) == 0 and int(errors_minor) == 0)

def analytics_tab():
    dfh = load_history()
    st.subheader("Analytics")
    if dfh.empty:
        st.info("No history yet.")
        return

    # filters
    c1, c2, c3 = st.columns(3)
    with c1:
        sup_opts = ["(All)"] + sorted([s for s in dfh["supplier"].dropna().unique().tolist() if s != ""])
        sup = st.selectbox("Supplier", sup_opts)
    with c2:
        proj_opts = ["(All)"] + sorted([s for s in dfh["project"].dropna().unique().tolist() if s != ""])
        proj = st.selectbox("Project", proj_opts)
    with c3:
        only_included = st.checkbox("Only Included (exclude==False)", value=True)

    show = dfh.copy()
    if sup != "(All)":
        show = show[show["supplier"] == sup]
    if proj != "(All)":
        show = show[show["project"] == proj]
    if "exclude" in show.columns and only_included:
        show = show[~show["exclude"].fillna(False)]

    # compute RFT
    if "errors_major" in show.columns and "errors_minor" in show.columns:
        show["rft_pass"] = (show["errors_major"].fillna(0).astype(int) == 0) & (show["errors_minor"].fillna(0).astype(int) == 0)
    else:
        show["rft_pass"] = False

    # KPI
    total = len(show)
    rft = int(show["rft_pass"].sum())
    maj = int(show["errors_major"].fillna(0).sum())
    minr = int(show["errors_minor"].fillna(0).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Audits", total)
    c2.metric("Right First Time", f"{(rft/total*100):.1f}%" if total else "—", f"{rft}/{total}")
    c3.metric("Major", maj)
    c4.metric("Minor", minr)

    # trend (by day)
    if "timestamp_utc" in show.columns:
        tmp = show.copy()
        tmp["date"] = pd.to_datetime(tmp["timestamp_utc"]).dt.date
        trend = tmp.groupby("date").agg(
            audits=("status","count"),
            major=("errors_major","sum"),
            minor=("errors_minor","sum"),
            rft=("rft_pass","sum")
        ).reset_index()
        st.line_chart(trend.set_index("date")[["audits","major","minor","rft"]])

    st.markdown("#### Audit Records")
    cols = [c for c in ["timestamp_utc","supplier","client","project","status","pdf_name","excel_name","exclude"] if c in show.columns]
    st.dataframe(show[cols], use_container_width=True)

# -------------------------
# SETTINGS
# -------------------------
def settings_tab(rules: dict):
    st.subheader("Settings")
    st.caption("Upload/Download rules and set the logo path. Password required to update rules.")

    # Logo path (persists in session only; you can wire it to a file if preferred)
    st.text_input("Logo file name (relative to repo root)", value=st.session_state.get("logo_path", LOGO_FILE_FALLBACK), key="logo_path")

    st.markdown("---")
    st.markdown("#### Rules file")
    if rules_edit_allowed():
        up = st.file_uploader("Upload rules_example.yaml", type=["yaml","yml"])
        if up is not None:
            try:
                data = yaml.safe_load(up.read().decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("Top-level YAML must be a mapping/dict.")
                save_rules(DEFAULT_RULES_FILE, data)
                st.success("Rules file replaced.")
            except Exception as e:
                st.error(f"YAML error: {e}")

        if st.button("Download current rules"):
            with open(DEFAULT_RULES_FILE, "rb") as f:
                st.download_button("Download rules_example.yaml", f.read(), file_name="rules_example.yaml", mime="text/yaml")
    else:
        st.info("Enter Rules Update Password to modify rules.")

# -------------------------
# MAIN
# -------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="✅")

    # Logo top-left (if present)
    logo_path = st.session_state.get("logo_path", LOGO_FILE_FALLBACK)
    if logo_path and os.path.exists(logo_path):
        st.markdown(logo_html(logo_path), unsafe_allow_html=True)

    st.title(APP_TITLE)

    # Entry gate
    if not gate_with_password():
        st.stop()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔎 Audit", "🎓 Training", "📊 Analytics", "⚙️ Settings"])

    rules = load_rules(DEFAULT_RULES_FILE)

    with tab1:
        st.subheader("Single Audit")
        meta = meta_form(rules)
        require_mimo = (meta.get("project") != "Power Resilience")

        st.markdown("---")
        up = st.file_uploader("Upload PDF Design", type=["pdf"])
        exclude_from_analytics = st.checkbox("Exclude this audit from analytics", value=False)
        ready = up is not None and meta_complete(meta, require_mimo)

        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            run = st.button("Run Audit", use_container_width=True, disabled=not ready)
        with c2:
            st.button("Clear Metadata", on_click=lambda: st.session_state.clear(), use_container_width=True)

        if not ready:
            st.warning("Fill all required metadata and upload a PDF to enable 'Run Audit'.")

        if run and up is not None:
            pdf_bytes = up.read()
            findings = run_audit(pdf_bytes, meta, rules)
            n_major = sum(1 for f in findings if f.get("severity") == "Major")
            n_minor = sum(1 for f in findings if f.get("severity") == "Minor")
            status = "PASS" if (n_major == 0 and n_minor == 0) else "REJECT"

            st.success(f"Audit complete: {status}. Major={n_major}, Minor={n_minor}")

            # Excel & Annotated PDF
            excel_bytes = make_excel(findings, meta, up.name, status)
            ann_pdf_bytes = annotate_pdf(pdf_bytes, findings)

            ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
            excel_name = f"{os.path.splitext(up.name)[0]}__{status}__{ts}.xlsx"
            pdf_name = f"{os.path.splitext(up.name)[0]}__ANNOTATED__{ts}.pdf"

            cdl1, cdl2 = st.columns(2)
            with cdl1:
                st.download_button("⬇️ Download Findings Excel", data=excel_bytes, file_name=excel_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with cdl2:
                st.download_button("⬇️ Download Annotated PDF", data=ann_pdf_bytes, file_name=pdf_name, mime="application/pdf", use_container_width=True)

            # Persist in history (keeps record even after download)
            row = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "supplier": meta.get("supplier",""),
                "client": meta.get("client",""),
                "project": meta.get("project",""),
                "site_type": meta.get("site_type",""),
                "vendor": meta.get("vendor",""),
                "cabinet_location": meta.get("cabinet_location",""),
                "radio_location": meta.get("radio_location",""),
                "sectors": meta.get("sectors",1),
                "mimo_s1": meta.get("mimo_s1",""),
                "mimo_s2": meta.get("mimo_s2",""),
                "mimo_s3": meta.get("mimo_s3",""),
                "mimo_s4": meta.get("mimo_s4",""),
                "mimo_s5": meta.get("mimo_s5",""),
                "mimo_s6": meta.get("mimo_s6",""),
                "use_s1_for_all": meta.get("use_s1_for_all", False),
                "site_address": meta.get("site_address",""),
                "status": status,
                "pdf_name": pdf_name,
                "excel_name": excel_name,
                "exclude": bool(exclude_from_analytics),
                "rft_pass": (status == "PASS"),
                "errors_major": n_major,
                "errors_minor": n_minor
            }
            append_history(row)

            # Show table of findings
            st.markdown("#### Findings")
            if findings:
                show = pd.DataFrame([{
                    "page": f.get("page"),
                    "severity": f.get("severity"),
                    "category": f.get("category"),
                    "rule_id": f.get("rule_id"),
                    "message": f.get("message")
                } for f in findings])
                st.dataframe(show, use_container_width=True)
            else:
                st.info("No findings. Quality gate passed.")

    with tab2:
        training_tab(rules)

    with tab3:
        analytics_tab()

    with tab4:
        settings_tab(rules)

# -------------------------
if __name__ == "__main__":
    main()
