# app.py
# AI Design Quality Auditor – full refresh with learning-from-report import
# ---------------------------------------------------------------
# Features:
# - Auth (Seker123) + YAML edit password (vanB3lkum21)
# - Metadata-first single-audit flow
# - MIMO per sector (hide if Project=Power Resilience)
# - PDF text extraction (PyMuPDF), bbox, annotation
# - Excel + ZIP export (report.xlsx, annotated.pdf, metadata.json, patch_suggestions.json)
# - History + Analytics
# - Training: Import audit ZIP/XLSX -> append feedback store -> build YAML patch -> merge into rules_current.yaml
# - Rules: must_contain, must_not_contain, regex, must_match_any, must_not_match_any,
#          if_then_require, if_then_forbid, title_contains_site_address_normalized
# ---------------------------------------------------------------

import io, os, re, json, base64, zipfile
from pathlib import Path
from datetime import datetime, timezone
import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import yaml

# ------------ Paths / constants ------------
ROOT = Path(".")
HISTORY_DIR = ROOT / "history"
RULES_FILE = ROOT / "rules_example.yaml"
MERGED_RULES_FILE = ROOT / "rules_current.yaml"
PATCH_DIR = ROOT / "rules" / "patches"
EXPORTS_DIR = ROOT / "exports"  # keep artifacts on disk until you clean manually

HISTORY_DIR.mkdir(parents=True, exist_ok=True)
PATCH_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_PASSWORD = "Seker123"
YAML_EDIT_PASSWORD = "vanB3lkum21"

DEFAULT_CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
DEFAULT_PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
DEFAULT_SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
DEFAULT_VENDORS = ["Ericsson", "Nokia"]
DEFAULT_SUPPLIERS = [
    "Adapt", "Telecom Plus", "COCO", "Monroe", "CKL", "Obelisk", "Wifibre", "MTI", "Circet"
]
DEFAULT_DRAWING_TYPES = ["General Arrangement", "Detailed Design"]
DEFAULT_RADIO_LOC = ["Low Level", "Midway", "High Level", "Unique Coverage"]

# MIMO options (deduped + sorted lightly). You can extend in rules yaml too.
MIMO_OPTIONS = [
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

# ------------ Utility ------------
def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def normalize_address_for_title(addr: str) -> str:
    if not addr:
        return ""
    # Upper, remove double spaces, remove ", 0 ," segments, strip punctuation commas
    s = addr.upper()
    # remove ", 0 ," or ",0," patterns (zero-only segment)
    s = re.sub(r"\s*,\s*0\s*,\s*", ", ", s)
    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip(" ,")
    return s

def read_pdf_text_and_blocks(pdf_bytes: bytes):
    """Return (pages_text: list[str], pages_blocks: list[list[block]]) where block=(x0,y0,x1,y1,text,page_index)."""
    pages_text, pages_blocks = [], []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        pages_text.append(page.get_text("text"))
        blocks = []
        for x0, y0, x1, y1, txt, *_ in page.get_text("blocks"):
            if txt and txt.strip():
                blocks.append((float(x0), float(y0), float(x1), float(y1), txt, i))
        pages_blocks.append(blocks)
    title = ""
    try:
        title = doc.metadata.get("title") or ""
    except Exception:
        pass
    doc.close()
    return pages_text, pages_blocks, title

def find_token_bbox(pblocks, token):
    """Very simple bbox: find first block containing token (case-insensitive)."""
    pat = re.escape(token.strip())
    if not pat:
        return None
    cre = re.compile(pat, re.IGNORECASE)
    for (x0,y0,x1,y1,txt,pidx) in pblocks:
        if cre.search(txt):
            return pidx, [x0,y0,x1,y1]
    return None

def annotate_pdf(pdf_bytes: bytes, findings: list):
    """Draw boxes for findings with bbox. Returns bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    red = (1, 0, 0)
    for f in findings:
        page_index = f.get("page")
        bbox = f.get("bbox")
        label = f'{f.get("severity","info").upper()} - {f.get("rule_id","") or f.get("id","")}'
        if page_index is None or bbox is None or len(bbox) != 4:
            continue
        try:
            page = doc[page_index]
            rect = fitz.Rect(*bbox)
            page.draw_rect(rect, color=red, width=1)
            page.insert_text((rect.x0, max(0, rect.y0-6)), label, fontsize=7, color=red)
        except Exception:
            continue
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    out.seek(0)
    return out.read()

def export_report_zip(findings_df: pd.DataFrame, annotated_pdf_bytes: bytes, metadata: dict, filename_stem: str):
    # Excel
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        findings_df.to_excel(writer, index=False, sheet_name="findings")
    excel_buf.seek(0)

    # Suggestions placeholder (can be smarter later)
    auto_patch = {
        "version": now_utc_iso(),
        "patch": {"groups":[{"id":"SUGGESTED","name":"Suggested (review)","rules":[]}]}
    }

    # Zip
    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.xlsx", excel_buf.getvalue())
        z.writestr("annotated.pdf", annotated_pdf_bytes)
        z.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        z.writestr("patch_suggestions.json", json.dumps(auto_patch, ensure_ascii=False, indent=2))
    zip_bytes.seek(0)

    # Persist a local copy for records
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_zip_path = EXPORTS_DIR / f"{filename_stem}_{stamp}.zip"
    with open(out_zip_path, "wb") as f:
        f.write(zip_bytes.getvalue())

    return zip_bytes, out_zip_path

# ------------ Rules management ------------
def load_rules(path: Path = None):
    """Load current rules preferring merged; fall back to example; safe defaults."""
    candidate = path or (MERGED_RULES_FILE if MERGED_RULES_FILE.exists() else RULES_FILE)
    if candidate.exists():
        with open(candidate, "r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f) or {}
            except Exception as e:
                st.error(f"YAML load error in {candidate.name}: {e}")
                data = {}
    else:
        data = {}
    # Ensure minimal structure
    data.setdefault("groups", [])
    data.setdefault("allowlist", [])
    return data

def save_merged_rules(merged: dict):
    with open(MERGED_RULES_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False, allow_unicode=True)

def apply_yaml_patch(patch: dict):
    base = load_rules(RULES_FILE if RULES_FILE.exists() else MERGED_RULES_FILE)
    merged = json.loads(json.dumps(base))  # deep copy
    if "patch" in patch and "groups" in patch["patch"]:
        merged.setdefault("groups", [])
        merged["groups"].extend(patch["patch"]["groups"])
    pid = patch.get("version", now_utc_iso()).replace(":","").replace("-","")
    with open(PATCH_DIR / f"auto_patch_{pid}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(patch, f, sort_keys=False, allow_unicode=True)
    save_merged_rules(merged)
    return merged

# ------------ Rules engine ------------
def rule_conditions_match(conditions: dict, meta: dict) -> bool:
    if not conditions:
        return True
    # Simple AND of equality checks
    # format: {"all":[{"field":"vendor","eq":"Ericsson"}, ...]}
    allc = conditions.get("all", [])
    for c in allc:
        field = c.get("field")
        eq = c.get("eq")
        if field and eq is not None:
            if str(meta.get(field, "")).strip() != str(eq).strip():
                return False
    return True

def normalize_for_title_compare(s: str) -> str:
    if not s:
        return ""
    s = s.upper()
    s = re.sub(r"\s*,\s*0\s*(,|$)", ",", s)  # ignore standalone zero segments
    s = re.sub(r"[\s,]+", " ", s).strip()
    return s

def audit_pdf(pdf_bytes: bytes, meta: dict, rules: dict):
    pages_text, pages_blocks, pdf_title = read_pdf_text_and_blocks(pdf_bytes)
    full_text = "\n".join(pages_text)

    findings = []
    allowlist = set([str(w).lower() for w in rules.get("allowlist", [])])

    def add_finding(rule_id, name, severity, page, bbox, text, check_type):
        token_norm = (text or "").lower().strip()
        if token_norm and token_norm in allowlist:
            return  # ignore allowed items
        findings.append({
            "id": f"F{len(findings)+1:05d}",
            "rule_id": rule_id,
            "rule_name": name,
            "severity": severity,
            "page": page,
            "bbox": bbox,
            "text": text,
            "check_type": check_type
        })

    # global checks for site address vs title
    # allow a special rule type or default enforcement:
    for g in rules.get("groups", []):
        for r in g.get("rules", []):
            if not r.get("enabled", True):
                continue
            severity = r.get("severity","minor")
            if not rule_conditions_match(r.get("conditions"), meta):
                continue
            r_id = r.get("id") or f"RULE_{g.get('id','G')}_{len(findings)}"
            name = r.get("name","Unnamed")

            # Allowlist local to rule
            local_allow = set([str(w).lower() for w in r.get("allowlist", [])])

            for chk in r.get("checks", []):
                ctype = chk.get("type")
                if ctype == "must_contain":
                    token = chk.get("token","").strip()
                    if token and re.search(re.escape(token), full_text, re.IGNORECASE) is None:
                        # missing required token
                        add_finding(r_id, name, severity, None, None, f"Missing required token: {token}", "must_contain")
                    else:
                        # present -> pinpoint first block
                        if token:
                            m = find_first_match_block(pages_blocks, token)
                            if m:
                                pidx, bbox = m
                                # Not a failure; we usually report violations, so skip
                                pass

                elif ctype == "must_not_contain":
                    token = chk.get("token","").strip()
                    if token and re.search(re.escape(token), full_text, re.IGNORECASE):
                        loc = find_first_match_block(pages_blocks, token)
                        pidx, bbox = (loc or (None, None))
                        tnorm = token.lower()
                        if tnorm in local_allow or tnorm in allowlist:
                            continue
                        add_finding(r_id, name, severity, pidx, bbox, f"Forbidden token present: {token}", "must_not_contain")

                elif ctype == "regex":
                    pattern = chk.get("pattern","").strip()
                    if not pattern:
                        continue
                    comp = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    for m in comp.finditer(full_text):
                        snippet = m.group(0)
                        pidx, bbox = backmap_to_block(pages_blocks, snippet)
                        add_finding(r_id, name, severity, pidx, bbox, f"Regex matched: {snippet}", "regex")

                elif ctype in ("must_match_any","must_not_match_any"):
                    patterns = chk.get("patterns", [])
                    field = chk.get("field") or "text"  # "mimo", "title", etc.
                    hay = field_value_for(field, meta, pdf_title, full_text)
                    matched = any(re.search(p, hay, re.IGNORECASE) for p in patterns if p)
                    if ctype == "must_match_any" and not matched:
                        add_finding(r_id, name, severity, None, None, f"{field} must match one of patterns", ctype)
                    if ctype == "must_not_match_any" and matched:
                        add_finding(r_id, name, severity, None, None, f"{field} matched a forbidden pattern", ctype)

                elif ctype in ("if_then_require","if_then_forbid"):
                    left = [t for t in chk.get("if_any", []) if t]
                    right_req = [t for t in chk.get("require_any", []) if t]
                    right_forbid = [t for t in chk.get("forbid_any", []) if t]
                    if any(re.search(re.escape(t), full_text, re.IGNORECASE) for t in left):
                        if ctype == "if_then_require":
                            if not any(re.search(re.escape(t), full_text, re.IGNORECASE) for t in right_req):
                                add_finding(r_id, name, severity, None, None,
                                            f"If any({left}) then require any({right_req}) not found", ctype)
                        else:
                            if any(re.search(re.escape(t), full_text, re.IGNORECASE) for t in right_forbid):
                                add_finding(r_id, name, severity, None, None,
                                            f"If any({left}) then forbid any({right_forbid}) violated", ctype)

                elif ctype == "title_contains_site_address_normalized":
                    # Compare normalized title vs normalized site address from meta
                    addr = normalize_for_title_compare(meta.get("site_address",""))
                    title_norm = normalize_for_title_compare(pdf_title or "")
                    if addr and addr != "(blank)" and addr != "0":
                        if addr not in title_norm:
                            add_finding(r_id, name, severity, 0, None,
                                        "Title does not contain normalized Site Address", ctype)

    # Generic spelling-ish: use rules allowlist only; no heavy dictionary to avoid runtime issues
    # (If you need a real spellcheck, we can wire a dictionary with rapidfuzz later)
    # -> Skipped here intentionally to be stable.

    # Attach quick bbox for obvious tokens inside text-based findings
    for f in findings:
        if f.get("bbox") is None and f.get("text"):
            # try to locate snippet token portion
            token = extract_token_from_text(f["text"])
            if token:
                m = find_first_match_block(pages_blocks, token)
                if m:
                    pidx, bbox = m
                    f["page"], f["bbox"] = pidx, bbox

    return findings, pdf_title

def find_first_match_block(pages_blocks, token):
    pat = re.escape(token)
    cre = re.compile(pat, re.IGNORECASE)
    for blocks in pages_blocks:
        for (x0,y0,x1,y1,txt,pidx) in blocks:
            if cre.search(txt):
                return pidx, [x0,y0,x1,y1]
    return None

def backmap_to_block(pages_blocks, snippet):
    # Best-effort mapping: find any block containing snippet
    return find_first_match_block(pages_blocks, snippet)

def extract_token_from_text(txt):
    # from strings like "Forbidden token present: Brush"
    m = re.search(r":\s*(.+)$", txt)
    if m:
        return m.group(1).strip()
    # or "Regex matched: XYZ"
    m = re.search(r"matched:\s*(.+)$", txt, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

def field_value_for(field, meta, title, full_text):
    field = field.lower().strip()
    if field in ("mimo","mimo_config","mimo_s1_s2_s3"):
        s1 = meta.get("mimo_s1") or ""
        s2 = meta.get("mimo_s2") or ""
        s3 = meta.get("mimo_s3") or ""
        return " | ".join([s1,s2,s3])
    if field == "title":
        return title or ""
    if field == "text":
        return full_text or ""
    # allow metadata generic
    return str(meta.get(field,""))

# ------------ Learning: import reports ------------
def load_feedback_store():
    fp = HISTORY_DIR / "feedback.csv"
    if fp.exists():
        try:
            return pd.read_csv(fp)
        except Exception:
            return pd.DataFrame(columns=[
                "timestamp_utc","finding_id","rule_id","text","decision","scope",
                "client","project","site_type","vendor","supplier","drawing_type","notes","severity"
            ])
    return pd.DataFrame(columns=[
        "timestamp_utc","finding_id","rule_id","text","decision","scope",
        "client","project","site_type","vendor","supplier","drawing_type","notes","severity"
    ])

def save_feedback_store(df):
    df.to_csv(HISTORY_DIR / "feedback.csv", index=False)

def append_feedback_from_df(df_find, meta):
    store = load_feedback_store()
    df = df_find.copy()
    for c, default in [
        ("finding_id",""),("rule_id",""),("text",""),("decision",""),
        ("scope","Global"),("notes",""),("severity","minor")
    ]:
        if c not in df.columns: df[c] = default
    # attach metadata
    for k in ["client","project","site_type","vendor","supplier","drawing_type"]:
        df[k] = meta.get(k,"")
    df["timestamp_utc"] = now_utc_iso()
    store = pd.concat([store, df[store.columns]], ignore_index=True)
    save_feedback_store(store)
    return store

def scope_conditions(scope, meta):
    s = (scope or "Global").lower()
    if s == "global": return None
    mapping = {
        "client": ("client", meta.get("client")),
        "project": ("project", meta.get("project")),
        "sitetype": ("site_type", meta.get("site_type")),
        "vendor": ("vendor", meta.get("vendor")),
        "supplier": ("supplier", meta.get("supplier")),
        "drawingtype": ("drawing_type", meta.get("drawing_type")),
    }
    if s in mapping:
        k, v = mapping[s]
        if v:
            return {"all":[{"field": k, "eq": str(v)}]}
    return None

def row_to_rule(row, meta):
    rtype = str(row.get("new_rule_type") or "").strip()
    token = str(row.get("new_rule_token_or_pattern") or "").strip()
    if not rtype or not token:
        return None
    rid = f"AUTO_{rtype.upper()}_{abs(hash(token))%10**8}"
    rule = {
        "id": rid,
        "name": row.get("notes") or f"Auto rule from finding {row.get('finding_id','')}",
        "severity": (row.get("severity") or "minor").lower(),
        "enabled": True,
        "checks": []
    }
    if rtype in ("must_contain","must_not_contain"):
        rule["checks"].append({"type": rtype, "token": token})
    elif rtype == "regex":
        rule["checks"].append({"type":"regex","pattern": token})
    elif rtype in ("must_match_any","must_not_match_any"):
        rule["checks"].append({"type":rtype,"patterns":[token],"field":"text"})
    elif rtype in ("if_then_require","if_then_forbid"):
        m = re.match(r"(?i)IF:(.+)\s+THEN:(.+)", token)
        if not m: return None
        left = [s.strip() for s in m.group(1).split("|") if s.strip()]
        right = [s.strip() for s in m.group(2).split("|") if s.strip()]
        if rtype == "if_then_require":
            rule["checks"].append({"type":rtype,"if_any":left,"require_any":right})
        else:
            rule["checks"].append({"type":rtype,"if_any":left,"forbid_any":right})
    elif rtype == "title_contains_site_address_normalized":
        rule["checks"].append({"type":rtype,"normalizers":["strip_commas","squash_spaces","upper","ignore_zero_segments"]})
    else:
        return None
    cond = scope_conditions(row.get("scope"), meta)
    if cond:
        rule["conditions"] = cond
    return rule

def build_patch_from_feedback(df, meta):
    patch = {"version": now_utc_iso(), "patch":{"groups":[]}}
    bucket = {"id":"AUTO_LEARNING","name":"Auto learned rules","rules":[]}

    for _, row in df.iterrows():
        decision = str(row.get("decision") or "").strip().lower()
        # new rule?
        nr = row_to_rule(row, meta)
        if nr:
            bucket["rules"].append(nr)
            continue

        # Not Valid -> add to allowlist rule in scope
        if decision == "not valid":
            token = str(row.get("text") or "").strip()
            if token:
                rid = f"AUTO_ALLOW_{abs(hash(token))%10**8}"
                rule = {
                    "id": rid, "name": f"Allow token: {token[:32]}",
                    "severity": "info", "enabled": True,
                    "checks":[{"type":"must_not_contain","token":"<<<no-op>>>"}],
                    "allowlist":[token]
                }
                cond = scope_conditions(row.get("scope"), meta)
                if cond: rule["conditions"] = cond
                bucket["rules"].append(rule)
        elif decision == "valid":
            # keep rule enabled; could tighten here if needed
            pass

    if bucket["rules"]:
        patch["patch"]["groups"].append(bucket)
    return patch

# ------------ UI helpers ------------
def set_logo_css(logo_filename: str):
    if not logo_filename:
        return
    if not Path(logo_filename).exists():
        st.info("Logo file not found in repo root.")
        return
    with open(logo_filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .top-left-logo {{
            position: fixed;
            top: 12px;
            left: 16px;
            z-index: 9999;
            width: 160px;
            height: auto;
        }}
        </style>
        <img class="top-left-logo" src="data:image/png;base64,{b64}">
        """,
        unsafe_allow_html=True
    )

def auth_gate():
    st.session_state.setdefault("authed", False)
    if st.session_state["authed"]:
        return True
    st.title("AI Design Quality Auditor")
    pw = st.text_input("Enter access password", type="password")
    if st.button("Enter"):
        if pw == ENTRY_PASSWORD:
            st.session_state["authed"] = True
            st.experimental_rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# ------------ Analytics ------------
def append_history_row(meta, findings_df, status_label, file_name):
    fp = HISTORY_DIR / "history.csv"
    row = {
        "timestamp_utc": now_utc_iso(),
        "client": meta.get("client",""),
        "project": meta.get("project",""),
        "site_type": meta.get("site_type",""),
        "vendor": meta.get("vendor",""),
        "supplier": meta.get("supplier",""),
        "drawing_type": meta.get("drawing_type",""),
        "sectors": meta.get("sectors",""),
        "status": status_label,  # Pass / Rejected
        "minor": int((findings_df["severity"]=="minor").sum()) if not findings_df.empty else 0,
        "major": int((findings_df["severity"]=="major").sum()) if not findings_df.empty else 0,
        "file_name": file_name
    }
    df = pd.read_csv(fp) if fp.exists() else pd.DataFrame(columns=row.keys())
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(fp, index=False)

def analytics_panel():
    st.subheader("Analytics overview")
    fp = HISTORY_DIR / "history.csv"
    if not fp.exists():
        st.info("No runs yet.")
        return
    df = pd.read_csv(fp)
    if df.empty:
        st.info("No runs yet.")
        return
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    rejected = (df["status"]=="Rejected").sum()
    passed = (df["status"]=="Pass").sum()
    rft = (passed/total*100.0) if total else 0.0
    col1.metric("Total audits", total)
    col2.metric("Pass", int(passed))
    col3.metric("Rejected", int(rejected))
    col4.metric("Right First Time %", f"{rft:.1f}%")

    st.markdown("#### By supplier")
    st.bar_chart(df.groupby("supplier")["status"].apply(lambda s: (s=="Rejected").mean()*100).fillna(0))

    st.markdown("#### By project (rejection %)")
    st.bar_chart(df.groupby("project")["status"].apply(lambda s: (s=="Rejected").mean()*100).fillna(0))

# ------------ Main app ------------
def main():
    st.set_page_config(page_title="AI Design QA", layout="wide")
    auth_gate()

    # Persistent settings in session
    st.session_state.setdefault("logo_path", "")

    with st.sidebar:
        st.header("Settings (Quick)")
        st.session_state["logo_path"] = st.text_input("Logo file in repo", value=st.session_state["logo_path"])
        set_logo_css(st.session_state["logo_path"])
        st.markdown("---")
        st.caption("YAML Edit is protected")
        yaml_pw = st.text_input("YAML edit password", type="password")
        can_edit_yaml = (yaml_pw == YAML_EDIT_PASSWORD)

    tabs = st.tabs(["🧪 Audit", "🔁 Training", "⚙️ Settings", "📊 Analytics"])

    # -------- Audit tab --------
    with tabs[0]:
        st.subheader("Run a single audit")

        # Metadata
        with st.form("meta_form"):
            c1, c2, c3 = st.columns(3)
            client = c1.selectbox("Client", DEFAULT_CLIENTS, index=0)
            project = c2.selectbox("Project", DEFAULT_PROJECTS, index=0)
            site_type = c3.selectbox("Site Type", DEFAULT_SITE_TYPES, index=0)

            v1, v2, v3 = st.columns(3)
            vendor = v1.selectbox("Proposed Vendor", DEFAULT_VENDORS, index=0)
            supplier = v2.selectbox("Supplier", DEFAULT_SUPPLIERS, index=0)
            drawing_type = v3.selectbox("Drawing Type", DEFAULT_DRAWING_TYPES, index=0)

            r1, r2 = st.columns(2)
            radio_loc = r1.selectbox("Proposed Radio Location", DEFAULT_RADIO_LOC, index=0)
            sectors = r2.selectbox("Quantity of Sectors", [1,2,3,4,5,6], index=2)

            addr = st.text_input("Site Address (normalized, ignore ', 0 ,')", "")

            hide_mimo = (project == "Power Resilience")
            st.markdown("**MIMO Configs per Sector**" + (" (hidden for Power Resilience)" if hide_mimo else ""))

            mimo_same_all = False
            mimo_s1 = mimo_s2 = mimo_s3 = ""
            if not hide_mimo:
                mimo_same_all = st.checkbox("Use S1 MIMO for all sectors", value=True)
                mimo_s1 = st.selectbox("Proposed MIMO Config S1", MIMO_OPTIONS, index=0, key="mimo_s1")
                if mimo_same_all:
                    mimo_s2, mimo_s3 = mimo_s1, mimo_s1
                else:
                    mimo_s2 = st.selectbox("Proposed MIMO Config S2", MIMO_OPTIONS, index=0, key="mimo_s2")
                    mimo_s3 = st.selectbox("Proposed MIMO Config S3", MIMO_OPTIONS, index=0, key="mimo_s3")

            uploaded = st.file_uploader("Upload a single PDF to audit", type=["pdf"])
            run = st.form_submit_button("Run Audit", use_container_width=True)

        if run:
            # Validate required metadata
            missing = []
            for k,v in {
                "Client": client, "Project": project, "Site Type": site_type,
                "Vendor": vendor, "Supplier": supplier, "Drawing Type": drawing_type,
                "Radio Location": radio_loc, "Sectors": sectors, "Site Address": addr or "(blank)"
            }.items():
                if v in ("", None):
                    missing.append(k)
            if uploaded is None:
                st.error("Please upload a PDF.")
            elif missing:
                st.error("Missing required metadata: " + ", ".join(missing))
            else:
                pdf_bytes = uploaded.read()
                rules = load_rules()
                meta = {
                    "client": client, "project": project, "site_type": site_type,
                    "vendor": vendor, "supplier": supplier, "drawing_type": drawing_type,
                    "radio_location": radio_loc, "sectors": sectors,
                    "mimo_s1": mimo_s1 if not hide_mimo else "",
                    "mimo_s2": mimo_s2 if not hide_mimo else "",
                    "mimo_s3": mimo_s3 if not hide_mimo else "",
                    "site_address": addr
                }
                findings, pdf_title = audit_pdf(pdf_bytes, meta, rules)
                df = pd.DataFrame(findings)

                # Pass / Reject
                major = int((df["severity"]=="major").sum()) if not df.empty else 0
                label = "Pass" if df.empty or major==0 else "Rejected"

                st.success(f"Audit complete: **{label}**")
                st.caption(f"PDF Title: {pdf_title}")

                st.dataframe(df if not df.empty else pd.DataFrame([{"message":"No findings"}]))

                # Annotate + Export
                annotated_bytes = annotate_pdf(pdf_bytes, findings)
                file_stem = Path(uploaded.name).stem + ("_Pass" if label=="Pass" else "_Rejected") + "_" + datetime.now().strftime("%Y%m%d")
                zip_buf, saved_zip_path = export_report_zip(
                    df,
                    annotated_bytes,
                    {**meta, "pdf_title": pdf_title},
                    file_stem
                )
                st.download_button("⬇️ Download Audit Package (ZIP)",
                                   data=zip_buf.getvalue(),
                                   file_name=f"{file_stem}.zip",
                                   mime="application/zip")

                append_history_row(meta, df, label, uploaded.name)

    # -------- Training tab --------
    with tabs[1]:
        st.subheader("Learn from your reviewed report")
        st.caption("Re-upload the **ZIP** (preferred) or **report.xlsx** you downloaded after an audit.")
        up = st.file_uploader("Drop ZIP or report.xlsx", type=["zip","xlsx"], key="train_upl")
        auto_apply = st.checkbox("Auto-apply patch to live rules (merge into rules_current.yaml)", value=True)
        if st.button("Import & Learn", use_container_width=True, disabled=up is None):
            try:
                meta = {}
                df = None
                if up.name.lower().endswith(".zip"):
                    zf = zipfile.ZipFile(io.BytesIO(up.read()))
                    if "metadata.json" in zf.namelist():
                        meta = json.loads(zf.read("metadata.json").decode("utf-8"))
                    # first xlsx
                    xlsx_names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
                    if xlsx_names:
                        with zf.open(xlsx_names[0]) as f:
                            df = pd.read_excel(f)
                else:
                    df = pd.read_excel(up)

                if df is None:
                    st.error("No report.xlsx found / provided.")
                else:
                    # Append feedback
                    store = append_feedback_from_df(df, meta)
                    # Build and show patch
                    patch = build_patch_from_feedback(df, meta)
                    st.code(yaml.safe_dump(patch, sort_keys=False, allow_unicode=True), language="yaml")
                    if auto_apply:
                        merged = apply_yaml_patch(patch)
                        st.success("Patch applied to live rules (rules_current.yaml).")
                    st.info(f"Feedback rows total: {len(store)}")
            except Exception as e:
                st.exception(e)

    # -------- Settings tab --------
    with tabs[2]:
        st.subheader("Rulebook & configuration")
        st.caption("View and (optionally) edit rules YAML. Editing requires YAML password in the sidebar.")
        cur = load_rules()
        st.markdown("##### Active rules (merged view)")
        st.code(yaml.safe_dump(cur, sort_keys=False, allow_unicode=True), language="yaml")

        if can_edit_yaml:
            st.markdown("##### Edit and save `rules_current.yaml`")
            text = st.text_area("YAML content", value=yaml.safe_dump(cur, sort_keys=False, allow_unicode=True), height=300)
            if st.button("Save YAML to rules_current.yaml", type="primary"):
                try:
                    test_obj = yaml.safe_load(text) or {}
                    if "groups" not in test_obj:
                        test_obj["groups"] = []
                    if "allowlist" not in test_obj:
                        test_obj["allowlist"] = []
                    save_merged_rules(test_obj)
                    st.success("Saved to rules_current.yaml")
                except Exception as e:
                    st.error(f"YAML error: {e}")
        else:
            st.info("Enter YAML edit password in the sidebar to enable saving.")

        st.markdown("---")
        st.markdown("##### Download current history & exports pointers")
        if (HISTORY_DIR / "history.csv").exists():
            st.download_button("⬇️ Download history.csv", data=open(HISTORY_DIR / "history.csv","rb").read(),
                               file_name="history.csv", mime="text/csv")

    # -------- Analytics tab --------
    with tabs[3]:
        analytics_panel()

def run_once_banner():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 70px; }
        </style>
        """,
        unsafe_allow_html=True
    )

def find_logo_early():
    # No-op; we inject via sidebar setting
    pass

# Helper used in audit rules
def find_first_page_title_text(pages_text):
    for t in pages_text:
        if t.strip():
            return t.splitlines()[0][:120]
    return ""

if __name__ == "__main__":
    run_once_banner()
    main()
