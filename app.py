# app.py
import os, io, re, json, base64, csv, zipfile, hashlib, textwrap
from datetime import datetime, timezone, timedelta
from tempfile import NamedTemporaryFile
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import yaml

# PDF + OCR
import fitz  # PyMuPDF

# Fuzzy/string utils
from rapidfuzz import fuzz, process

# --- Optional Spelling (graceful fallback) ---
try:
    from spellchecker import SpellChecker
    _HAS_SPELL = True
except Exception:
    _HAS_SPELL = False

# ----------------------------
# CONSTANTS / CONFIG DEFAULTS
# ----------------------------
ENTRY_PASSWORD = "Seker123"         # Gate to enter the app
SETTINGS_PASSWORD = "vanB3lkum21"   # To edit YAML directly in Settings

HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(HISTORY_DIR, "audits.csv")

RULES_FILE = "rules_example.yaml"

# How long results stay visible in the UI (analytics table), files always on disk
DEFAULT_UI_VISIBILITY_HOURS = 24
DEFAULT_EXPORT_TIME_UTC = "00:00"  # for nightly manual export button hint

# MIMO OPTIONS (cleaned + sorted)
MIMO_OPTIONS = [
    "18 @2x2",
    "18 @2x2; 26 @4x4",
    "18 @2x2; 70\\80 @2x2",
    "18 @2x2; 80 @2x2",
    "18\\21 @2x2",
    "18\\21 @2x2; 26 @4x4",
    "18\\21 @2x2; 70\\80 @2x2",
    "18\\21 @2x2; 80 @2x2",
    "18\\21 @2x2; 3500 @32x32",
    "18\\21 @4x4",
    "18\\21 @4x4; 70 @2x4",
    "18\\21 @4x4; 70\\80 @2x2",
    "18\\21 @4x4; 70\\80 @2x2; 3500 @32x32",
    "18\\21 @4x4; 70\\80 @2x4",
    "18\\21 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21 @4x4; 70\\80 @2x4; 3500 @32x32",
    "18\\21@4x4; 70\\80 @2x2",
    "18\\21@4x4; 70\\80 @2x4",
    "18\\21\\26 @2x2",
    "18\\21\\26 @2x2; 70\\80 @2x2",
    "18\\21\\26 @2x2; 3500 @32x32",
    "18\\21\\26 @2x2; 3500 @8X8",
    "18\\21\\26 @2x2; 70\\80 @2x2; 3500 @32x32",
    "18\\21\\26 @2x2; 70\\80 @2x2; 3500 @8x8",
    "18\\21\\26 @2x2; 70\\80 @2x4; 3500 @32x32",
    "18\\21\\26 @2x2; 80 @2x2",
    "18\\21\\26 @2x2; 80 @2x2; 3500 @8x8",
    "18\\21\\26 @4x4",
    "18\\21\\26 @4x4; 70\\80 @2x2",
    "18\\21\\26 @4x4; 3500 @8x8",
    "18\\21\\26 @4x4; 3500 @32x32",
    "18\\21\\26 @4x4; 70\\80 @2x2; 3500 @8x8",
    "18\\21\\26 @4x4; 70\\80 @2x2; 3500 @32x32",
    "18\\21\\26 @4x4; 70\\80 @2x4",
    "18\\21\\26 @4x4; 70\\80 @2x4; 3500 @8x8",
    "18\\21\\26 @4x4; 70\\80 @2x4; 3500 @32x32",
    "18\\21\\26 @4x4; 80 @2x2",
    "18\\21\\26 @4x4; 80 @2x2; 3500 @32x32",
    "18\\21\\26 @4x4; 80 @2x4",
    "18\\21\\26 @4x4; 80 @2x4; 3500 @8x8",
    "18\\26 @2x2",
    "18\\26 @4x4; 21 @2x2; 80 @2x2",
]

CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
RADIO_LOCATIONS = ["Low Level", "Midway", "High Level", "Unique Coverage"]
DRAWING_TYPES = ["General Arrangement", "Detailed Design"]

# Analytics columns to persist
HISTORY_COLUMNS = [
    "timestamp_utc","status","file","client","project","site_type","vendor",
    "radio_location","qty_sectors","drawing_type","supplier",
    "findings_count","mimo_json","exclude"
]

# ----------------------------
# UTILS: logo, hashing, etc.
# ----------------------------
def _read_logo_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

def render_logo_top_left():
    logo_path = st.session_state.get("logo_path", "logo.png")
    b = _read_logo_bytes(logo_path)
    if not b:
        st.markdown(
            "<div style='position:fixed;top:8px;left:10px;font-weight:600;color:#555'>No logo (set in Settings → UI)</div>",
            unsafe_allow_html=True,
        )
        return
    b64 = base64.b64encode(b).decode()
    st.markdown(
        f"""
        <style>
          .brand-logo {{
            position: fixed;
            top: 8px;
            left: 12px;
            z-index: 9999;
            max-height: 56px;
            width: auto;
            opacity: 0.95;
          }}
          .block-container {{ padding-top: 80px; }}
        </style>
        <img class="brand-logo" src="data:image/png;base64,{b64}" />
        """,
        unsafe_allow_html=True,
    )

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_utc_str():
    return datetime.now(timezone.utc).isoformat()

def sha1_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

# ----------------------------
# RULES: load / save / merge
# ----------------------------
def load_rules(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {
            "allowlist": [],
            "regex_rules": [],
            "cross_rules": [],
            "site_address_rule": {"enabled": True, "ignore_if_contains_zero_segment": True},
            "learned_overrides": {},
        }
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
        except Exception as e:
            st.error(f"YAML parse error in {path}: {e}")
            raise
    # normalize
    data.setdefault("allowlist", [])
    data.setdefault("regex_rules", [])
    data.setdefault("cross_rules", [])
    data.setdefault("site_address_rule", {"enabled": True, "ignore_if_contains_zero_segment": True})
    data.setdefault("learned_overrides", {})
    return data

def save_rules(path: str, data: Dict[str, Any]):
    tmp = NamedTemporaryFile("w", delete=False, encoding="utf-8", newline="")
    tmp_name = tmp.name
    tmp.close()
    with open(tmp_name, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp_name, path)

def add_learned_override(rules: Dict[str, Any], rule_key: str, decision: str, ctx: Dict[str, Any]):
    """
    Store override keyed by (rule_key + normalized context),
    decision in {"valid","not_valid"}.
    """
    inventory = rules.setdefault("learned_overrides", {})
    ctx_key = json.dumps({k: ctx.get(k) for k in ["client","project","site_type","vendor","drawing_type","qty_sectors"]}, sort_keys=True)
    bucket = inventory.setdefault(rule_key, {})
    bucket[ctx_key] = decision

def get_learned_override(rules: Dict[str, Any], rule_key: str, ctx: Dict[str, Any]) -> Optional[str]:
    inventory = rules.get("learned_overrides", {})
    b = inventory.get(rule_key, {})
    if not b: return None
    ctx_key = json.dumps({k: ctx.get(k) for k in ["client","project","site_type","vendor","drawing_type","qty_sectors"]}, sort_keys=True)
    return b.get(ctx_key)

# ----------------------------
# HISTORY: robust save/load
# ----------------------------
def save_history_row(row: Dict[str, Any]) -> None:
    safe = {k: row.get(k, "") for k in HISTORY_COLUMNS}
    safe["exclude"] = bool(safe.get("exclude", False))
    for k, v in safe.items():
        if isinstance(v, (dict, list)):
            safe[k] = json.dumps(v, ensure_ascii=False)
        elif v is None:
            safe[k] = ""
    df = pd.DataFrame([safe], columns=HISTORY_COLUMNS)

    if not os.path.exists(HISTORY_FILE):
        tmp = NamedTemporaryFile("w", delete=False, newline="", encoding="utf-8")
        n = tmp.name; tmp.close()
        df.to_csv(n, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        os.replace(n, HISTORY_FILE)
    else:
        tmp = NamedTemporaryFile("w", delete=False, newline="", encoding="utf-8")
        n = tmp.name; tmp.close()
        df.to_csv(n, index=False, header=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        with open(HISTORY_FILE, "a", encoding="utf-8", newline="") as main, open(n, "r", encoding="utf-8") as part:
            for line in part:
                main.write(line)
        os.remove(n)

def _try_read_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    # tolerant read
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", dtype=str, encoding="utf-8")
    except Exception:
        pass
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", dtype=str, encoding="utf-8-sig")
    except Exception:
        pass
    # brutal repair
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        if not lines: return pd.DataFrame(columns=HISTORY_COLUMNS)
        header = lines[0]; expected = header.count(",")
        good = [header] + [ln for ln in lines[1:] if ln.strip() and ln.count(",")==expected]
        repaired = os.path.join(HISTORY_DIR, "_repaired_audits.csv")
        with open(repaired, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(good) + "\n")
        return pd.read_csv(repaired, dtype=str, encoding="utf-8")
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

def load_history() -> pd.DataFrame:
    df = _try_read_history(HISTORY_FILE)
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col!="exclude" else False
    df["exclude"] = df["exclude"].astype(str).str.lower().isin(["true","1","yes"])
    df["findings_count"] = pd.to_numeric(df["findings_count"], errors="coerce").fillna(0).astype(int)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df[HISTORY_COLUMNS]

def filter_history_for_ui(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    if df.empty: return df
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    if "timestamp_utc" in df.columns:
        return df[df["timestamp_utc"] >= cutoff]
    return df

# ----------------------------
# PDF helpers
# ----------------------------
def extract_pdf_text_and_boxes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        spans = page.get_text("blocks")  # coarse blocks
        pages.append({"index": i, "text": text, "blocks": spans, "width": page.rect.width, "height": page.rect.height})
    doc.close()
    return pages

def find_text_instances(page, needle: str) -> List[Tuple[float,float,float,float]]:
    # returns list of bbox for each appearance of needle (case-insensitive)
    rects = []
    if not needle.strip():
        return rects
    areas = page.search_for(needle, hit_max=1000, quads=False)
    for r in areas:
        rects.append((r.x0, r.y0, r.x1, r.y1))
    return rects

def annotate_pdf(original_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    doc = fitz.open(stream=original_bytes, filetype="pdf")
    for f in findings:
        pidx = int(f.get("page_index", 0))
        if pidx < 0 or pidx >= len(doc):
            continue
        page = doc[pidx]
        note = f"[{f.get('severity','MINOR')}] {f.get('rule_key','misc')}: {f.get('message','')}"
        bbox = f.get("bbox")
        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                r = fitz.Rect(*bbox)
                page.add_rect_annot(r)
                page.add_freetext_annot(r, note, fontsize=8, rotate=0, fill_color=None, text_color=(1,0,0))
            except Exception:
                page.insert_text((36, 36), note)
        else:
            page.insert_text((36, 36), note)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()

# ----------------------------
# CHECKS (rules + spelling)
# ----------------------------
def tokenize_words(text: str) -> List[str]:
    # very simple tokenizer
    return re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text)

def spelling_checks(pages: List[Dict[str, Any]], allowlist: set) -> List[Dict[str, Any]]:
    findings = []
    if not _HAS_SPELL:
        st.warning("Spelling: `pyspellchecker` not installed — skipping spelling step.")
        return findings

    sp = SpellChecker(distance=1)
    # seed allowlist words to dictionary (will be treated as correct)
    for w in allowlist:
        sp.word_frequency.add(w.lower())

    for p in pages:
        words = tokenize_words(p["text"])
        for w in words:
            wl = w.lower()
            if wl in allowlist or wl.isnumeric():
                continue
            if sp.correction(wl) != wl:
                # candidate suggestion (safe)
                try:
                    cands = sp.candidates(wl)
                    sug = next(iter(cands)) if cands else None
                except Exception:
                    sug = None
                findings.append({
                    "rule_key": "SPELLING",
                    "severity": "MINOR",
                    "page_index": p["index"],
                    "message": f"Possibly misspelled: '{w}'" + (f" → suggestion: {sug}" if sug else ""),
                    "bbox": None,
                })
    return findings

def regex_checks(pages: List[Dict[str, Any]], regex_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings = []
    for rule in regex_rules:
        key = rule.get("key","REGEX")
        pattern = rule.get("pattern","")
        severity = rule.get("severity","MINOR")
        msg = rule.get("message", f"Matched rule: {key}")
        if not pattern: continue
        try:
            rx = re.compile(pattern, flags=re.IGNORECASE|re.MULTILINE)
        except re.error:
            continue
        for p in pages:
            for m in rx.finditer(p["text"]):
                findings.append({
                    "rule_key": key,
                    "severity": severity,
                    "page_index": p["index"],
                    "message": msg,
                    "bbox": None,
                })
    return findings

def cross_checks(ctx: Dict[str, Any], cross_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rules like: if text X appears, Y must not/should appear, vendor/project/site dependent, etc.
    Each cross_rule:
      - key, message, severity
      - when: {client/project/site_type/vendor/drawing_type/qty_sectors optional exact matches}
      - require_any: ["Brush"]             -> flag if none present
      - forbid_any: ["Generator Power"]    -> flag if any present
    """
    findings = []
    textset = ctx.get("all_text_lower", "")
    for r in cross_rules:
        key = r.get("key","CROSS")
        severity = r.get("severity","MAJOR")
        message = r.get("message", key)
        cond = r.get("when", {})
        # match context
        ok = True
        for k,v in cond.items():
            if str(ctx.get(k,"")).strip() != str(v).strip():
                ok = False; break
        if not ok: continue

        need = [s.lower() for s in r.get("require_any", [])]
        bad  = [s.lower() for s in r.get("forbid_any", [])]

        if need:
            if not any(s in textset for s in need):
                findings.append({
                    "rule_key": key, "severity": severity, "page_index": 0,
                    "message": f"Expected any of {need} not found. {message}", "bbox": None
                })
        if bad:
            if any(s in textset for s in bad):
                findings.append({
                    "rule_key": key, "severity": severity, "page_index": 0,
                    "message": f"Found forbidden term from {bad}. {message}", "bbox": None
                })
    return findings

def site_address_check(site_address: str, pages: List[Dict[str, Any]], enabled: bool, ignore_zero_segment: bool) -> List[Dict[str, Any]]:
    """
    If enabled, the 'site_address' provided must be found on page 1 title/text (fuzzy),
    except if it contains a ', 0 ,' segment and ignore_zero_segment is True.
    """
    if not enabled or not site_address.strip():
        return []
    if ignore_zero_segment and re.search(r",\s*0\s*,", site_address):
        # explicit bypass
        return []
    # We try to find 2+ tokens match on first two pages
    needles = [t.strip() for t in re.split(r"[,\n]+", site_address) if t.strip()]
    if not needles:
        return []
    search_space = " ".join([p["text"] for p in pages[:2]]).lower()
    hits = 0
    for n in needles:
        if len(n) < 3: 
            continue
        if n.lower() in search_space:
            hits += 1
    # threshold: at least 2 tokens must match
    if hits >= 2:
        return []
    return [{
        "rule_key": "SITE_ADDRESS",
        "severity": "MAJOR",
        "page_index": 0,
        "message": f"Site Address from metadata not found in title/top pages: '{site_address}'.",
        "bbox": None,
    }]

# ----------------------------
# UI HELPERS
# ----------------------------
def gated_entry():
    st.title("AI Design Quality Auditor")
    pw = st.text_input("Enter password to continue", type="password")
    if st.button("Unlock"):
        if pw == ENTRY_PASSWORD:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

def render_metadata_form():
    st.subheader("Audit Metadata")
    col1, col2 = st.columns(2)
    with col1:
        client = st.selectbox("Client", CLIENTS, index=None, placeholder="Select a client")
        project = st.selectbox("Project", PROJECTS, index=None, placeholder="Select a project")
        site_type = st.selectbox("Site Type", SITE_TYPES, index=None, placeholder="Select a site type")
        vendor = st.selectbox("Proposed Vendor", VENDORS, index=None, placeholder="Select a vendor")
        radio_location = st.selectbox("Proposed Radio Location", RADIO_LOCATIONS, index=None, placeholder="Select radio location")
    with col2:
        drawing_type = st.selectbox("Drawing Type", DRAWING_TYPES, index=None, placeholder="Select drawing type")
        supplier = st.text_input("Supplier (for analytics only)", value=st.session_state.get("supplier",""))
        qty_sectors = st.number_input("Quantity of Sectors", min_value=1, max_value=6, value=3, step=1)
        site_address = st.text_input("Site Address", placeholder="e.g., MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ")

    # MIMO per sector (hide when Power Resilience)
    hide_mimo = (project == "Power Resilience")
    use_same = st.checkbox("Use S1 MIMO config for all sectors", value=True, disabled=hide_mimo)
    mimo_values = {}
    if not hide_mimo:
        st.markdown("**Proposed MIMO Config (by sector)**")
        for s in range(1, qty_sectors+1):
            if s == 1:
                mimo_values["S1"] = st.selectbox("S1", MIMO_OPTIONS, index=None, placeholder="Select MIMO")
                if use_same:
                    # copy S1 to others
                    for t in range(2, qty_sectors+1):
                        mimo_values[f"S{t}"] = mimo_values["S1"]
                    break
            else:
                mimo_values[f"S{s}"] = st.selectbox(f"S{s}", MIMO_OPTIONS, index=None, placeholder="Select MIMO")
    else:
        mimo_values = {}

    # completeness check
    mandatory = [("Client", client), ("Project", project), ("Site Type", site_type),
                 ("Vendor", vendor), ("Radio Location", radio_location),
                 ("Drawing Type", drawing_type), ("Site Address", site_address)]
    missing = [k for k,v in mandatory if (v is None or str(v).strip()=="")]
    if missing:
        st.warning("Please complete all metadata fields before auditing: " + ", ".join(missing))

    return {
        "client": client or "",
        "project": project or "",
        "site_type": site_type or "",
        "vendor": vendor or "",
        "radio_location": radio_location or "",
        "drawing_type": drawing_type or "",
        "supplier": supplier or "",
        "qty_sectors": int(qty_sectors),
        "site_address": site_address or "",
        "mimo_values": mimo_values,
        "mimo_hidden": hide_mimo,
        "metadata_complete": len(missing)==0
    }

# ----------------------------
# TRAINING FROM REPORT
# ----------------------------
def train_from_report_xlsx(rules: Dict[str,Any], file: io.BytesIO, ctx: Dict[str,Any]) -> Tuple[int,int]:
    """
    Expects the Excel you downloaded from this tool. Columns must include:
      rule_key, decision (Valid/Not Valid), message (optional), severity (optional)
    We record learned_overrides per context.
    """
    df = pd.read_excel(file)
    ok, bad = 0, 0
    for _, r in df.iterrows():
        rk = str(r.get("rule_key","")).strip()
        dec = str(r.get("decision","")).strip().lower()
        if not rk or dec not in ("valid","not valid","not_valid"):
            bad += 1
            continue
        norm_dec = "valid" if dec=="valid" else "not_valid"
        add_learned_override(rules, rk, norm_dec, ctx)
        ok += 1
    return ok, bad

def apply_learned_overrides(findings: List[Dict[str,Any]], rules: Dict[str,Any], ctx: Dict[str,Any]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    """
    Split findings into (kept, overridden) based on learned_overrides.
    """
    kept, overridden = [], []
    for f in findings:
        rk = f.get("rule_key","")
        dec = get_learned_override(rules, rk, ctx)
        if dec == "valid":
            # user said this rule is not an error in this context -> drop
            overridden.append({**f, "override":"valid"})
        elif dec == "not_valid":
            # explicitly confirmed as true error -> keep as is
            kept.append(f)
        else:
            kept.append(f)
    return kept, overridden

# ----------------------------
# EXPORTS
# ----------------------------
def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Audit")
    return out.getvalue()

def package_zip(files: Dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return out.getvalue()

# ----------------------------
# MAIN
# ----------------------------
def main():
    st.set_page_config(page_title="AI Design Quality Auditor", layout="wide")
    render_logo_top_left()

    if "authed" not in st.session_state or not st.session_state["authed"]:
        return gated_entry()

    # --- Tabs ---
    tab_audit, tab_train, tab_analytics, tab_settings = st.tabs(["🔍 Audit", "🎓 Train", "📈 Analytics", "⚙️ Settings"])

    # ---------------- AUDIT ----------------
    with tab_audit:
        st.header("Single Audit")
        rules = load_rules(RULES_FILE)
        colL, colR = st.columns([2,1])

        with colL:
            meta = render_metadata_form()
            uploaded = st.file_uploader("Upload PDF (one at a time)", type=["pdf"])
            exclude_this = st.checkbox("Exclude this review from analytics", value=False)

            run = st.button("🚦 Run Audit", type="primary", disabled=(uploaded is None or not meta["metadata_complete"]))
            if uploaded and run:
                original_bytes = uploaded.read()
                pages = extract_pdf_text_and_boxes(original_bytes)

                # Build context for learned overrides
                ctx = {
                    "client": meta["client"], "project": meta["project"], "site_type": meta["site_type"],
                    "vendor": meta["vendor"], "drawing_type": meta["drawing_type"], "qty_sectors": meta["qty_sectors"],
                    "all_text_lower": " ".join([p["text"] for p in pages]).lower()
                }

                # 1) Spelling
                allow = set([w.lower() for w in rules.get("allowlist", [])])
                findings = []
                findings += spelling_checks(pages, allow)

                # 2) Regex
                findings += regex_checks(pages, rules.get("regex_rules", []))

                # 3) Cross field/text checks
                findings += site_address_check(meta["site_address"], pages,
                                               enabled=rules.get("site_address_rule",{}).get("enabled", True),
                                               ignore_zero_segment=rules.get("site_address_rule",{}).get("ignore_if_contains_zero_segment", True))
                findings += cross_checks(ctx, rules.get("cross_rules", []))

                # 4) Apply learned overrides
                kept, overridden = apply_learned_overrides(findings, rules, ctx)

                df = pd.DataFrame(kept + overridden)
                if df.empty:
                    df = pd.DataFrame([{"rule_key":"INFO","severity":"INFO","page_index":0,"message":"No findings."}])

                # Decide PASS/REJECT
                has_error = any(sev in ("MAJOR","MINOR") for sev in df.get("severity","MINOR"))
                status = "PASS" if not has_error else "REJECTED"

                # Annotate PDF at coarse level
                annotated = annotate_pdf(original_bytes, kept)

                # Prepare Excel
                # Add decision column placeholder for training export later
                if "decision" not in df.columns:
                    df["decision"] = ""
                excel_bytes = dataframe_to_excel_bytes(df)

                # Save artifacts under history/
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(uploaded.name)[0]
                stamped = f"{base_name}_{status}_{ts}"
                ensure_dir(HISTORY_DIR)
                pdf_name = os.path.join(HISTORY_DIR, f"{stamped}.annotated.pdf")
                xls_name = os.path.join(HISTORY_DIR, f"{stamped}.xlsx")
                with open(pdf_name, "wb") as f: f.write(annotated)
                with open(xls_name, "wb") as f: f.write(excel_bytes)

                # Persist history row
                save_history_row({
                    "timestamp_utc": now_utc_str(), "status": status, "file": uploaded.name,
                    "client": meta["client"], "project": meta["project"], "site_type": meta["site_type"],
                    "vendor": meta["vendor"], "radio_location": meta["radio_location"],
                    "qty_sectors": str(meta["qty_sectors"]), "drawing_type": meta["drawing_type"],
                    "supplier": meta["supplier"], "findings_count": len(kept),
                    "mimo_json": json.dumps(meta["mimo_values"], ensure_ascii=False),
                    "exclude": exclude_this
                })

                st.success(f"Audit completed: **{status}** — {len(kept)} actionable / {len(overridden)} auto-ignored by learning.")

                # Downloads (all three)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.download_button("⬇️ Excel Report", data=excel_bytes, file_name=f"{stamped}.xlsx")
                with c2:
                    st.download_button("⬇️ Annotated PDF", data=annotated, file_name=f"{stamped}.annotated.pdf")
                with c3:
                    z = package_zip({f"{stamped}.xlsx":excel_bytes, f"{stamped}.annotated.pdf":annotated})
                    st.download_button("⬇️ Zip (Excel+PDF)", data=z, file_name=f"{stamped}.zip")

                st.divider()
                st.markdown("#### Findings")
                st.dataframe(df, use_container_width=True)

        with colR:
            st.markdown("### Tips")
            st.info("• MIMO configs are hidden when **Project = Power Resilience**.\n"
                    "• Use **Training** tab to upload the Excel you downloaded and teach the tool what's Valid / Not-Valid.\n"
                    "• Toggle **Exclude from analytics** if you’re trialing rules.")

    # ---------------- TRAIN ----------------
    with tab_train:
        st.header("Training & Rules")

        st.markdown("##### 1) Bulk learning from a downloaded Excel report")
        st.caption("Upload a report previously downloaded from this tool, add your ‘decision’ per row (Valid / Not Valid), then upload it here to teach the model.")
        ctx_col1, ctx_col2 = st.columns(2)
        with ctx_col1:
            t_client = st.selectbox("Client (context for learning)", CLIENTS, index=None)
            t_project = st.selectbox("Project (context)", PROJECTS, index=None)
            t_site_type = st.selectbox("Site Type (context)", SITE_TYPES, index=None)
        with ctx_col2:
            t_vendor = st.selectbox("Vendor (context)", VENDORS, index=None)
            t_drawing = st.selectbox("Drawing Type (context)", DRAWING_TYPES, index=None)
            t_qty = st.number_input("Qty Sectors (context)", min_value=1, max_value=6, value=3, step=1)

        train_file = st.file_uploader("Upload Excel (the report you downloaded) to apply your decisions", type=["xlsx"], key="train_x")
        if st.button("Apply learning from Excel"):
            if not all([t_client, t_project, t_site_type, t_vendor, t_drawing]):
                st.error("Please provide all context fields for training.")
            elif not train_file:
                st.error("Upload the Excel report first.")
            else:
                rules = load_rules(RULES_FILE)
                ctx = {"client":t_client,"project":t_project,"site_type":t_site_type,"vendor":t_vendor,
                       "drawing_type":t_drawing,"qty_sectors":int(t_qty)}
                ok, bad = train_from_report_xlsx(rules, train_file, ctx)
                save_rules(RULES_FILE, rules)
                st.success(f"Training applied. Learned decisions for {ok} rules. {bad} rows skipped.")

        st.divider()
        st.markdown("##### 2) Quick add one rule")
        colA, colB = st.columns(2)
        with colA:
            r_key = st.text_input("Rule Key (e.g., AHEGC_should_be_AHEGG)")
            r_pattern = st.text_input("Regex Pattern to flag (optional)")
            r_message = st.text_input("Message", value="Rule matched.")
            r_severity = st.selectbox("Severity", ["MINOR","MAJOR"], index=1)
            r_allow_word = st.text_input("Add a word to spelling allowlist (optional)")
        with colB:
            st.write("**Cross rule (optional)**")
            cr_when_client = st.selectbox("When: client", [""]+CLIENTS, index=0)
            cr_when_project = st.selectbox("When: project", [""]+PROJECTS, index=0)
            cr_need = st.text_input("Require any of (comma separated)")
            cr_forbid = st.text_input("Forbid any of (comma separated)")
        if st.button("➕ Add / Update Rule"):
            rules = load_rules(RULES_FILE)
            if r_allow_word.strip():
                if r_allow_word.lower() not in [w.lower() for w in rules["allowlist"]]:
                    rules["allowlist"].append(r_allow_word.strip())
            if r_pattern.strip():
                # upsert regex rule
                existing = next((r for r in rules["regex_rules"] if r.get("key")==r_key), None)
                newr = {"key": r_key, "pattern": r_pattern, "message": r_message, "severity": r_severity}
                if existing:
                    existing.update(newr)
                else:
                    rules["regex_rules"].append(newr)
            if cr_need.strip() or cr_forbid.strip():
                cond = {}
                if cr_when_client: cond["client"] = cr_when_client
                if cr_when_project: cond["project"] = cr_when_project
                rules["cross_rules"].append({
                    "key": r_key or "CROSS",
                    "when": cond,
                    "require_any": [s.strip() for s in cr_need.split(",") if s.strip()],
                    "forbid_any": [s.strip() for s in cr_forbid.split(",") if s.strip()],
                    "message": r_message, "severity": r_severity
                })
            save_rules(RULES_FILE, rules)
            st.success("Rule(s) saved.")

        st.divider()
        st.markdown("##### 3) Edit YAML directly (advanced)")
        pw = st.text_input("Settings password", type="password")
        if pw == SETTINGS_PASSWORD:
            rules_text = ""
            try:
                with open(RULES_FILE, "r", encoding="utf-8") as f:
                    rules_text = f.read()
            except Exception:
                rules_text = yaml.safe_dump(load_rules(RULES_FILE), sort_keys=False, allow_unicode=True)
            edited = st.text_area("rules_example.yaml", value=rules_text, height=280)
            if st.button("Save YAML"):
                try:
                    data = yaml.safe_load(edited) or {}
                    save_rules(RULES_FILE, data)
                    st.success("YAML saved.")
                except Exception as e:
                    st.error(f"YAML error: {e}")
        else:
            st.info("Enter Settings password to edit YAML.")

    # ---------------- ANALYTICS ----------------
    with tab_analytics:
        st.header("Analytics")
        dfh = load_history()
        st.caption("Only rows not marked ‘exclude’ are counted in the charts.")
        vis_hours = st.number_input("Show last N hours (UI only)", min_value=1, max_value=24*14, value=DEFAULT_UI_VISIBILITY_HOURS)
        st.session_state["ui_visibility_hours"] = int(vis_hours)
        df_vis = filter_history_for_ui(dfh.copy(), st.session_state["ui_visibility_hours"])
        df_use = df_vis[df_vis["exclude"] != True].copy()

        if df_use.empty:
            st.info("No analytics yet.")
        else:
            # KPI tiles
            col1, col2, col3, col4 = st.columns(4)
            total = len(df_use)
            rejected = (df_use["status"]=="REJECTED").sum()
            rft = (df_use["status"]=="PASS").sum() / total * 100.0
            col1.metric("Total audits", f"{total}")
            col2.metric("Rejected", f"{rejected}")
            col3.metric("Right-First-Time %", f"{rft:.1f}%")
            col4.metric("Avg findings", f"{df_use['findings_count'].mean():.2f}")

            st.markdown("#### Recent audits")
            st.dataframe(df_vis.sort_values("timestamp_utc", ascending=False), use_container_width=True)

            # Download raw history csv (clean)
            out_clean = io.BytesIO()
            dfh.to_csv(out_clean, index=False)
            st.download_button("⬇️ Download history CSV", data=out_clean.getvalue(), file_name="audits_history.csv")

    # ---------------- SETTINGS ----------------
    with tab_settings:
        st.header("Settings")

        st.markdown("##### UI")
        lp = st.text_input("Logo file path (relative)", value=st.session_state.get("logo_path","logo.png"))
        if st.button("Save UI"):
            st.session_state["logo_path"] = lp
            st.success("Saved. Refresh the page if needed.")

        st.divider()
        st.markdown("##### History maintenance")
        if st.button("🩺 Repair history file"):
            before = sum(1 for _ in open(HISTORY_FILE, "r", encoding="utf-8")) if os.path.exists(HISTORY_FILE) else 0
            df = _try_read_history(HISTORY_FILE)
            if df.empty:
                st.warning("No valid rows found (or file missing). Nothing to repair.")
            else:
                tmp = NamedTemporaryFile("w", delete=False, newline="", encoding="utf-8")
                n = tmp.name; tmp.close()
                df.to_csv(n, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                os.replace(n, HISTORY_FILE)
                after = len(df.index) + 1
                st.success(f"Repaired history. Lines before: {before:,} → after: {after:,}.")

        st.divider()
        st.markdown("##### Nightly export (manual)")
        st.caption("Click to generate a full zipped dump of last 24h reports under history/.")
        if st.button("Generate 24h export (.zip)"):
            dfh = load_history()
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            recent = dfh[dfh["timestamp_utc"] >= cutoff]
            if recent.empty:
                st.info("No records in last 24h.")
            else:
                zname = f"history_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.zip"
                mem = io.BytesIO()
                with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
                    # include CSV snapshot
                    csvb = recent.to_csv(index=False).encode("utf-8")
                    z.writestr("recent_history.csv", csvb)
                    # include any matching files
                    for fn in os.listdir(HISTORY_DIR):
                        if any(fn.endswith(ext) for ext in (".xlsx",".annotated.pdf",".zip",".csv")):
                            fp = os.path.join(HISTORY_DIR, fn)
                            try:
                                with open(fp, "rb") as f:
                                    z.writestr(f"files/{fn}", f.read())
                            except Exception:
                                pass
                st.download_button("⬇️ Download 24h export", data=mem.getvalue(), file_name=zname)

if __name__ == "__main__":
    main()
