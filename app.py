# AI Design QA — single-PDF auditor
# Streamlit app with metadata, easy rule editing, history, exports, and top-right logo

import os
import io
import re
import sys
import json
import time
import base64
import shutil
import zipfile
import datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import yaml
import fitz  # PyMuPDF
from rapidfuzz import process, fuzz

import streamlit as st

APP_TITLE = "AI Design QA"
HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)
HISTORY_FILE = HISTORY_DIR / "audit_log.csv"
DEFAULT_RULES_FILE = Path("rules_example.yaml")

# ─────────────────────────────────────────────
# Streamlit page setup
# ─────────────────────────────────────────────
st.set_page_config(page_title=APP_TITLE, page_icon="✅", layout="wide")


# ─────────────────────────────────────────────
# Top-right logo (tolerant file finder + secrets/env override)
# ─────────────────────────────────────────────
def _find_logo_path() -> Optional[str]:
    # 1) optional explicit configuration
    candidate = None
    try:
        candidate = st.secrets.get("logo_path", None)
    except Exception:
        pass
    candidate = candidate or os.environ.get("LOGO_PATH")
    if candidate and Path(candidate).exists():
        return candidate

    # 2) search repo root for something that looks like a logo
    exts = ("png", "jpg", "jpeg", "svg", "webp")
    root = Path(".")
    patterns = []
    for kw in ("logo", "seker"):
        for ext in exts:
            patterns.append(f"*{kw}*.{ext}")
            patterns.append(f"*{kw}*.{ext}.*")  # tolerate double extensions like .png.png
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if p.is_file():
                return str(p)
    return None


def _img_mime(p: str) -> str:
    pl = p.lower()
    if ".svg" in pl:
        return "image/svg+xml"
    if ".webp" in pl:
        return "image/webp"
    if ".jpg" in pl or ".jpeg" in pl:
        return "image/jpeg"
    return "image/png"


def render_logo_top_right():
    logo_path = _find_logo_path()
    if not logo_path:
        st.info(
            "⚠️ Logo file not found in repo root (try `logo.png` or a filename containing `seker`). "
            "You can also set `logo_path` in `.streamlit/secrets.toml` or set `LOGO_PATH` env var."
        )
        return
    try:
        b64 = base64.b64encode(Path(logo_path).read_bytes()).decode()
        mime = _img_mime(logo_path)
        st.markdown(
            f"""
            <style>
              .top-right-logo {{
                position: fixed; top: 16px; right: 16px; width: 120px; z-index: 9999;
              }}
              @media (max-width: 900px) {{
                .top-right-logo {{ width: 90px; }}
              }}
            </style>
            <img src="data:{mime};base64,{b64}" class="top-right-logo"/>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"⚠️ Could not load logo (`{e}`)")


render_logo_top_right()


# ─────────────────────────────────────────────
# Utilities: rules load/save, history, filenames
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_rules(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data
    except Exception as e:
        st.error(f"Could not read rules file: {e}")
        return {}


def append_history(row: Dict[str, Any]):
    df = pd.DataFrame([row])
    if HISTORY_FILE.exists():
        old = pd.read_csv(HISTORY_FILE)
        out = pd.concat([old, df], ignore_index=True)
    else:
        out = df
    out.tail(5000).to_csv(HISTORY_FILE, index=False)  # keep last 5k


def timestamp_utc() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def stamped_filename(base: str, outcome: str, ext: str) -> str:
    stem = Path(base).stem
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    return f"{stem}__{outcome.upper()}__{stamp}.{ext.lstrip('.')}"


# ─────────────────────────────────────────────
# PDF helpers
# ─────────────────────────────────────────────
def extract_pages_text(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            pages.append((i + 1, page.get_text("text") or ""))
    return pages


def annotate_pdf(pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    """
    Minimal annotation: draws small boxes and adds sticky notes at the top-left of each page
    listing messages detected for that page. If no coordinates are known, we still add a note.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    by_page: Dict[int, List[str]] = {}
    for f in findings:
        pg = int(f.get("page") or 1)
        msg = f.get("message", "Finding")
        by_page.setdefault(pg, []).append(msg)

        bbox = f.get("boxes")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                r = fitz.Rect(*[float(x) for x in bbox])
                page = doc[pg - 1]
                page.draw_rect(r, color=(1, 0, 0), width=1)
            except Exception:
                pass  # best effort

    for pg, msgs in by_page.items():
        try:
            page = doc[pg - 1]
            note_text = "Findings:\n- " + "\n- ".join(msgs[:12])
            # place near top-left margin
            rect = fitz.Rect(36, 36, 300, 200)
            page.add_text_annot(rect.tl, note_text)
        except Exception:
            pass

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


# ─────────────────────────────────────────────
# Rule evaluation
# ─────────────────────────────────────────────
def text_matches_rule(text: str, rule: Dict[str, Any]) -> bool:
    # Support any of: keywords (all must be present), any_keywords (at least one),
    # regex (single or list), not_keywords (block if present)
    kw_all = rule.get("keywords") or []
    kw_any = rule.get("any_keywords") or []
    kw_not = rule.get("not_keywords") or []
    regs = rule.get("regex") or []

    t = text.lower()

    # block first
    for w in kw_not:
        if w.lower() in t:
            return False

    for w in kw_all:
        if w.lower() not in t:
            return False

    if kw_any:
        ok = any(w.lower() in t for w in kw_any)
        if not ok:
            return False

    if regs:
        if isinstance(regs, str):
            regs = [regs]
        for r in regs:
            if not re.search(r, text, flags=re.IGNORECASE | re.MULTILINE):
                return False

    return True


def scope_allows(meta: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    """
    Optional scoping in each rule:
      scope:
        client: ["BTEE", "Vodafone"]
        project: ["RAN", "Power Resilience"]
        vendor: ["Ericsson"]
        site_type: [...]
        cabinet: [...]
        radio: [...]
        sectors: [1,2,3,4,5,6]
    If a scope key exists, value must match one of the allowed values.
    """
    scope = rule.get("scope") or {}
    for k, allowed in scope.items():
        if allowed is None:
            continue
        v = meta.get(k)
        if isinstance(allowed, list):
            if v not in allowed:
                return False
        else:
            if v != allowed:
                return False
    return True


def run_audit(pdf_bytes: bytes, rules: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    pages = extract_pages_text(pdf_bytes)

    findings: List[Dict[str, Any]] = []

    # 1) spelling (optional allow list)
    spell_section = rules.get("spelling", {})
    allow_words = set((spell_section.get("allow") or []))
    # build dynamic lexicon candidates (per page tokens)
    for page_num, txt in pages:
        words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", txt)
        # detect "obvious" misspellings: words with no vowels and longer than 4, or
        # words similar to a common near miss sample
        # This is intentionally conservative to avoid noise.
        for w in words:
            wl = w.lower()
            if wl in allow_words:
                continue
            if not re.search(r"[aeiou]", wl) and len(wl) > 5:
                sug = None
                findings.append(
                    {
                        "file": meta.get("file_name", "document.pdf"),
                        "page": page_num,
                        "kind": "Spelling",
                        "message": f"Suspicious token: '{w}' (no vowels)",
                        "boxes": None,
                    }
                )
            # optional fuzzy heuristic against allow list (only if allow list is populated)
            elif allow_words:
                cand, score, _ = process.extractOne(wl, allow_words, scorer=fuzz.ratio)
                if score < 60 and wl not in allow_words:
                    findings.append(
                        {
                            "file": meta.get("file_name", "document.pdf"),
                            "page": page_num,
                            "kind": "Spelling",
                            "message": f"Possible misspelling: '{w}'",
                            "boxes": None,
                        }
                    )

    # 2) checklist/regex rules
    for section_name, section in (rules.get("checklist") or {}).items():
        enabled = section.get("enabled", True)
        if not enabled:
            continue
        if not scope_allows(meta, section):
            continue

        message = section.get("message") or f"Failed rule: {section_name}"
        kind = section.get("kind", "Checklist")

        for page_num, txt in pages:
            if not text_matches_rule(txt, section):
                findings.append(
                    {
                        "file": meta.get("file_name", "document.pdf"),
                        "page": page_num,
                        "kind": kind,
                        "message": message,
                        "boxes": None,
                    }
                )

    df = pd.DataFrame(findings, columns=["file", "page", "kind", "message", "boxes"])
    outcome = "PASS" if df.empty else "REJECTED"
    return df, outcome


# ─────────────────────────────────────────────
# UI — header
# ─────────────────────────────────────────────
st.title(APP_TITLE)
st.caption("Single-file QA with easy rules, full metadata, history, and exports")


# ─────────────────────────────────────────────
# Metadata form (all fields required)
# ─────────────────────────────────────────────
with st.form("metadata_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        client = st.selectbox("Client*", ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"])
        project = st.selectbox("Project*", ["RAN", "Power Resilience", "East Unwind", "Beacon 4"])
        site_type = st.selectbox("Site Type*", ["Greenfield", "Rooftop", "Streetworks"])
    with col2:
        vendor = st.selectbox("Proposed Vendor*", ["Ericsson", "Nokia"])
        cabinet = st.selectbox("Proposed Cabinet Location*", ["Indoor", "Outdoor"])
        radio = st.selectbox("Proposed Radio Location*", ["High Level", "Low Level", "Indoor and Door"])
    with col3:
        sectors = st.selectbox("Quantity of Sectors*", [1, 2, 3, 4, 5, 6])
        site_address = st.text_input("Site Address*")

    # robust rule: hide MIMO for any "power res…" spelling
    hide_mimo = project.strip().lower().startswith("power res")
    if hide_mimo:
        mimo = "(n/a)"
        st.caption("Proposed MIMO Config not required for Power Resilience projects.")
    else:
        mimo = st.text_input("Proposed MIMO Config*", placeholder="e.g. 18\\21\\26 @4x4; 3500 @8x8")

    file = st.file_uploader("Upload a single PDF*", type=["pdf"])

    c1, c2, c3 = st.columns([1, 1, 6])
    with c1:
        run = st.form_submit_button("Run Audit", use_container_width=True, type="primary")
    with c2:
        clear = st.form_submit_button("Clear Metadata", use_container_width=True)

if clear:
    st.experimental_rerun()

# Require all fields before running
def _all_filled() -> bool:
    req_text_ok = bool(site_address.strip()) and (hide_mimo or bool(mimo.strip()))
    return all(
        [
            client,
            project,
            site_type,
            vendor,
            cabinet,
            radio,
            sectors,
            req_text_ok,
            file is not None,
        ]
    )


# ─────────────────────────────────────────────
# Run audit
# ─────────────────────────────────────────────
if run:
    if not _all_filled():
        st.error("Please complete **all** metadata fields and upload a PDF before running.")
        st.stop()

    pdf_bytes = file.read()
    rules_path = DEFAULT_RULES_FILE
    rules = load_rules(rules_path)

    meta = {
        "client": client,
        "project": project,
        "site_type": site_type,
        "vendor": vendor,
        "cabinet": cabinet,
        "radio": radio,
        "sectors": sectors,
        "mimo": mimo,
        "site_address": site_address,
        "file_name": file.name,
        "timestamp_utc": timestamp_utc(),
        "user": os.environ.get("STREAMLIT_SHARE_USER", "anonymous"),
    }

    with st.spinner("Running audit…"):
        findings_df, outcome = run_audit(pdf_bytes, rules, meta)

    # Top status table
    st.subheader("Audit Summary")
    with st.container(border=True):
        summary = {
            "file": [file.name],
            "status": [outcome],
            "Spelling": [int((findings_df.kind == "Spelling").sum())],
            "Checklist": [int((findings_df.kind == "Checklist").sum())],
            "pages": [len(extract_pages_text(pdf_bytes))],
        }
        st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)

    # Decision banner
    if outcome == "PASS":
        st.success("✅ **QA PASS** – please continue with Second Check.")
    else:
        st.error("❌ **REJECTED** – findings listed below.")

    # Findings table
    st.subheader("Findings")
    if findings_df.empty:
        st.info("No findings. 🎉")
    else:
        st.dataframe(findings_df, use_container_width=True, hide_index=True)

    # Annotated PDF + Excel exports
    st.subheader("Exports")
    # Annotated PDF (best effort even if no boxes)
    annotated_pdf = annotate_pdf(pdf_bytes, findings_df.to_dict("records"))
    # Excel
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as xw:
        findings_df.to_excel(xw, index=False, sheet_name="findings")
        # Also include metadata sheet for traceability
        pd.DataFrame([meta]).to_excel(xw, index=False, sheet_name="metadata")
    excel_bytes = excel_buf.getvalue()

    excel_name = stamped_filename(file.name, outcome if outcome == "PASS" else "REJECTED", "xlsx")
    pdf_name = stamped_filename(file.name, outcome if outcome == "PASS" else "REJECTED", "pdf")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download Excel report", data=excel_bytes, file_name=excel_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with c2:
        st.download_button("⬇️ Download annotated PDF", data=annotated_pdf, file_name=pdf_name, mime="application/pdf", use_container_width=True)

    # Append history
    append_history(
        {
            "timestamp_utc": meta["timestamp_utc"],
            "user": meta["user"],
            "client": client,
            "project": project,
            "site_type": site_type,
            "vendor": vendor,
            "cabinet": cabinet,
            "radio": radio,
            "sectors": sectors,
            "mimo": mimo,
            "file": file.name,
            "outcome": outcome,
            "total_findings": len(findings_df),
        }
    )

# ─────────────────────────────────────────────
# History (latest 200)
# ─────────────────────────────────────────────
st.subheader("Audit history (latest 200)")
if HISTORY_FILE.exists():
    hist = pd.read_csv(HISTORY_FILE).tail(200)
    st.dataframe(hist, use_container_width=True, hide_index=True)
else:
    st.caption("No history yet. Run your first audit to see entries here.")
