# ----------------------------- AI Design Quality Auditor -----------------------------
# Full app with precise PDF annotations (PyMuPDF), training, analytics, and YAML editor
# ------------------------------------------------------------------------------------

import os, io, re, json, zipfile, base64, datetime as dt
from pathlib import Path

import streamlit as st
import pandas as pd
import yaml
from rapidfuzz import process, fuzz
import fitz  # PyMuPDF

# ============================== CONSTANTS & SETUP ====================================

APP_TITLE = "AI Design Quality Auditor"
REPO_ROOT = Path(__file__).parent
HISTORY_DIR = REPO_ROOT / "history"
HISTORY_DIR.mkdir(exist_ok=True)
HISTORY_CSV = HISTORY_DIR / "audit_history.csv"

ENTRY_PASSWORD = "Seker123"          # App access
RULES_EDIT_PASSWORD = "vanB3lkum21"  # Rules editor guard
RULES_FILE_DEFAULT = "rules_example.yaml"

SUPPLIERS = ["CEG", "CTIL", "Emfyser", "Innov8", "Invict", "KTL Team (Internal)", "Trylon"]

CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN", "Power Resilience", "East Unwind", "Beacon 4"]
SITE_TYPES = ["Greenfield", "Rooftop", "Streetworks"]
VENDORS = ["Ericsson", "Nokia"]
CAB_LOCS = ["Indoor", "Outdoor"]
RADIO_LOCS = ["Low Level", "High Level", "Unique Coverage", "Midway"]
DRAWING_TYPES = ["General Arrangement", "Detailed Design"]

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
    "(blank)",
]

ANNOTATION = {
    "pin_radius": 7.0,
    "pin_color": (1, 0.35, 0.0),
    "pin_outline": (1, 1, 1),
    "callout_color": (1, 0.9, 0.8),
    "callout_text": (0.1, 0.1, 0.1),
    "callout_line": (1, 0.35, 0.0),
    "font_size": 9.5,
    "margin_box_w": 260,
    "margin_box_pad": 8,
    "margin_box_fill": (0.04, 0.04, 0.04),
    "margin_box_stroke": (0.6, 0.6, 0.6),
    "footer_text_color": (0.7, 0.7, 0.7),
}

st.set_page_config(page_title=APP_TITLE, layout="wide")

# ----------------------------------- UTIL --------------------------------------------

def logo_header():
    # Try any of these names in repo root, or use session override
    cands = [
        st.session_state.get("logo_file") or "",
        "logo.png", "logo.jpg", "logo.jpeg", "logo.svg",
        "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png",
        "88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png",
    ]
    hit = None
    for nm in cands:
        p = REPO_ROOT / nm
        if nm and p.exists():
            hit = p
            break

    c1, c2 = st.columns([0.22, 0.78])
    if hit:
        c1.image(str(hit), use_container_width=True)
    else:
        c1.markdown("### Seker")
    c2.markdown(f"## {APP_TITLE}")

def load_yaml_safe(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return {}
            return data
    except Exception:
        return {}

def save_yaml_safe(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def now_utc_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def parse_ts_col(s):
    try:
        return pd.to_datetime(s, utc=True, errors="coerce")
    except Exception:
        return pd.NaT

def ensure_history_headers():
    if not HISTORY_CSV.exists():
        pd.DataFrame(columns=[
            "timestamp_utc","status","client","project","site_type","vendor",
            "cabinet_location","radio_location","drawing_type","supplier",
            "qty_sectors","mimo_s1","mimo_s2","mimo_s3","mimo_s4","mimo_s5","mimo_s6",
            "site_address","file_name","excel_name","pdf_name","zip_name",
            "exclude"
        ]).to_csv(HISTORY_CSV, index=False)

def load_history() -> pd.DataFrame:
    ensure_history_headers()
    try:
        df = pd.read_csv(HISTORY_CSV, on_bad_lines="skip")
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        return df
    # normalize types
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    if "exclude" not in df.columns:
        df["exclude"] = False
    return df

def append_history(row: dict):
    ensure_history_headers()
    df = load_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_CSV, index=False)

def safe_file_name(stem: str, suffix: str) -> str:
    stem = re.sub(r"[^\w\-.]+","_", stem)
    return f"{stem}{suffix}"

# ---------------------------- PDF EXTRACT & ANNOTATE ---------------------------------

def extract_pages_text(pdf_bytes: bytes) -> list[str]:
    """Return list of page strings using PyMuPDF only."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(doc.page_count):
        p = doc.load_page(i)
        pages.append(p.get_text("text"))
    doc.close()
    return pages

def _page_text_blocks(page: fitz.Page):
    blocks = page.get_text("blocks")
    out = []
    for b in blocks:
        x0,y0,x1,y1,txt = b[0],b[1],b[2],b[3],b[4]
        if txt and txt.strip():
            out.append({"bbox": fitz.Rect(x0,y0,x1,y1), "text": txt})
    return out

def _find_text_bbox(page: fitz.Page, needle: str):
    if not needle:
        return None
    # exact
    hits = page.search_for(needle, quads=False, flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_IGNORECASE)
    if hits:
        return hits[0]
    # regex relaxed
    safe = re.escape(needle)
    patt = re.sub(r"\\\s+", r"\\s+", safe)
    try:
        areas = page.search_for(patt, quads=False, hit_max=16, flags=fitz.TEXT_REGEX|fitz.TEXT_IGNORECASE)
        if areas:
            return areas[0]
    except Exception:
        pass
    # fuzzy to blocks
    blocks = _page_text_blocks(page)
    nrm = re.sub(r"\W+","", needle.lower())
    best, score = None, 0
    for blk in blocks:
        hay = re.sub(r"\W+","", blk["text"].lower())
        if not hay:
            continue
        overlap = len(set(nrm) & set(hay)) / max(1, len(nrm))
        if overlap > score:
            best, score = blk, overlap
    if best and score > 0.35:
        return best["bbox"]
    return None

def _draw_pin(page: fitz.Page, idx: int, rect: fitz.Rect):
    c = ANNOTATION
    center = rect.tl + (rect.width/2, rect.height/2)
    sh = page.new_shape()
    sh.draw_circle(center, c["pin_radius"])
    sh.finish(color=c["pin_color"], fill=c["pin_color"])
    sh.draw_circle(center, c["pin_radius"]+1)
    sh.finish(width=1, color=c["pin_outline"])
    sh.commit()
    page.insert_text(center - (c["pin_radius"]/1.2, c["pin_radius"]/1.3),
                     str(idx), fontsize=c["font_size"]+1,
                     color=(1,1,1), fontname="helv", render_mode=3)

def _ensure_margin_box(page: fitz.Page) -> fitz.Rect:
    c = ANNOTATION
    r = page.rect
    box = fitz.Rect(r.width - c["margin_box_w"] + 6, 36, r.width - 6, r.height - 36)
    page.draw_rect(box, color=c["margin_box_stroke"], width=0.7, fill=c["margin_box_fill"])
    page.insert_text((box.x0 + c["margin_box_pad"], box.y0 + c["margin_box_pad"]),
                     "Comments", fontsize=c["font_size"]+1, color=(1,1,1))
    return box

def _add_comment(page: fitz.Page, box: fitz.Rect, idx: int, text: str, y: float) -> float:
    c = ANNOTATION
    start = fitz.Point(box.x0 + c["margin_box_pad"], y)
    badge = 7
    page.draw_circle(start + (badge, badge), badge, color=(1,1,1), fill=(1,1,1))
    page.insert_text(start + (badge-3.5, badge-4),
                     str(idx), fontsize=c["font_size"], color=(0,0,0))
    page.insert_textbox(
        fitz.Rect(start.x + 2*badge + 6, start.y, box.x1 - c["margin_box_pad"], start.y + 60),
        text, fontsize=c["font_size"], color=c["callout_text"]
    )
    return start.y + 50

def annotate_pdf(original_pdf_bytes: bytes, findings: list[dict], meta: dict, status: str) -> bytes:
    doc = fitz.open(stream=original_pdf_bytes, filetype="pdf")
    # group by page
    byp = {}
    for i,f in enumerate(findings, start=1):
        p = int(f.get("page", 1))
        p = max(1, min(doc.page_count, p))
        byp.setdefault(p, []).append((i,f))

    for pno, items in byp.items():
        page = doc.load_page(pno-1)
        box = _ensure_margin_box(page)
        y = box.y0 + 28
        for idx, f in items:
            msg = f.get("message","").strip() or "(no detail)"
            b = f.get("bbox")
            rect = fitz.Rect(*b) if (b and len(b)==4) else _find_text_bbox(page, f.get("anchor") or f.get("snippet") or "")
            if rect is None:
                rect = fitz.Rect(box.x0 - 40, y, box.x0 - 20, y+20)
            _draw_pin(page, idx, rect)
            y = _add_comment(page, box, idx, msg, y)
        footer = f"Audit: {status} • {meta.get('Client','')} • {meta.get('Project','')} • {meta.get('Site Address','')} • Generated by Seker"
        page.insert_text((36, page.rect.height - 24), footer, fontsize=8.5, color=ANNOTATION["footer_text_color"])
    out = io.BytesIO()
    doc.save(out, deflate=True, garbage=3)
    doc.close()
    return out.getvalue()

# -------------------------------- CHECKS / RULES ------------------------------------

def load_rules(path: Path) -> dict:
    return load_yaml_safe(path)

def apply_rules(pages_text: list[str], meta: dict, rules: dict) -> list[dict]:
    """
    Extremely simple rules engine that supports:
    rules['checklist']: list of {name, severity, must_contain: [str], reject_if_present: [str]}
    """
    findings = []
    checklist = rules.get("checklist", []) if isinstance(rules, dict) else []
    for rule in checklist:
        name = rule.get("name","Rule")
        sev = rule.get("severity","minor")
        must = rule.get("must_contain", []) or []
        rej = rule.get("reject_if_present", []) or []

        for i, txt in enumerate(pages_text, start=1):
            hit_all = all(s.lower() in txt.lower() for s in must)
            hit_rej = any(s.lower() in txt.lower() for s in rej)
            if must and not hit_all:
                findings.append({"page": i, "message": f"{name}: missing required content {must}", "severity": sev, "anchor": must[0] if must else ""})
            if rej and hit_rej:
                findings.append({"page": i, "message": f"{name}: forbidden content present {list(set([s for s in rej if s.lower() in txt.lower()]))}", "severity": sev, "anchor": list(set([s for s in rej if s]))[0]})
    return findings

def spelling_checks(pages_text: list[str], allow: set[str]) -> list[dict]:
    """
    Very light spelling-ish: finds tokens that look like 5+ letters not in allow-list
    and tries a suggestion via fuzzy matching within the allow-list.
    """
    findings = []
    words = set()
    for t in pages_text:
        words.update(re.findall(r"[A-Za-z]{5,}", t))
    bad = [w for w in sorted(words) if w.lower() not in allow]
    allow_list = list(allow)
    for w in bad:
        cand = process.extractOne(w, allow_list, scorer=fuzz.WRatio)
        sug = cand[0] if cand and cand[1] >= 90 else None
        msg = f"Possible misspelling: '{w}'" + (f" → did you mean '{sug}'?" if sug else "")
        findings.append({"page": 1, "message": msg, "severity": "minor", "anchor": w})
    return findings

def address_vs_title_check(pages_text: list[str], meta: dict) -> list[dict]:
    addr = (meta.get("Site Address","") or "").strip()
    if not addr:
        return []
    if ", 0 ," in addr:
        return []
    first = pages_text[0] if pages_text else ""
    if addr.upper() not in first.upper():
        return [{"page": 1, "message": "Site Address in metadata doesn’t match the title block.", "severity": "major", "anchor": "TITLE"}]
    return []

# ------------------------------ REPORT GENERATION -----------------------------------

def make_excel(findings: list[dict], meta: dict, src_name: str, status: str) -> bytes:
    rows = []
    for f in findings:
        rows.append({
            "Page": f.get("page",1), "Severity": f.get("severity","minor"),
            "Comment": f.get("message",""), "Anchor": f.get("anchor","")
        })
    df = pd.DataFrame(rows or [{"Page":"","Severity":"","Comment":"","Anchor":""}])

    # Meta sheet as one row
    mrow = {k: meta.get(k,"") for k in [
        "Supplier","Drawing Type","Client","Project","Site Type","Proposed Vendor",
        "Proposed Cabinet Location","Proposed Radio Location","Quantity of Sectors",
        "MIMO S1","MIMO S2","MIMO S3","MIMO S4","MIMO S5","MIMO S6","Site Address"
    ]}
    meta_df = pd.DataFrame([mrow])

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Findings", index=False)
        meta_df.to_excel(xw, sheet_name="Metadata", index=False)
        ws = xw.book.create_sheet("Summary", 0)
        ws["A1"] = "Source file"; ws["B1"] = src_name
        ws["A2"] = "Status"; ws["B2"] = status
        ws["A3"] = "Generated"; ws["B3"] = now_utc_iso()
    return buf.getvalue()

def bundle_zip(excel_bytes: bytes, pdf_bytes: bytes, base_stem: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(base_stem + ".xlsx", excel_bytes)
        z.writestr(base_stem + ".pdf", pdf_bytes)
    return buf.getvalue()

# ------------------------------------ UI GATES --------------------------------------

def gate_with_password():
    if "entry_ok" not in st.session_state:
        st.session_state.entry_ok = False
    if st.session_state.entry_ok:
        return True
    pw = st.text_input("Enter access password", type="password")
    if st.button("Unlock"):
        st.session_state.entry_ok = (pw == ENTRY_PASSWORD)
        if not st.session_state.entry_ok:
            st.error("Wrong password.")
    return st.session_state.entry_ok

# =============================== MAIN APP ============================================

def audit_tab():
    st.markdown("### Audit Metadata (all required)")
    col1,col2,col3,col4 = st.columns(4)
    supplier = col1.selectbox("Supplier", SUPPLIERS, index=0)
    drawing_type = col2.selectbox("Drawing Type", DRAWING_TYPES)
    client = col3.selectbox("Client", CLIENTS)
    project = col4.selectbox("Project", PROJECTS)

    col1,col2,col3,col4 = st.columns(4)
    site_type = col1.selectbox("Site Type", SITE_TYPES)
    vendor = col2.selectbox("Proposed Vendor", VENDORS)
    cab_loc = col3.selectbox("Proposed Cabinet Location", CAB_LOCS)
    radio_loc = col4.selectbox("Proposed Radio Location", RADIO_LOCS)

    col1,col2 = st.columns([1,3])
    qty = col1.selectbox("Quantity of Sectors", [1,2,3,4,5,6], index=0)
    site_addr = col2.text_input("Site Address")

    st.markdown("### Proposed MIMO Config")
    use_all = st.checkbox("Use S1 for all sectors", value=True)
    sector_mimos = {}
    for s in range(1, qty+1):
        if s == 1:
            sector_mimos[f"MIMO S{s}"] = st.selectbox("MIMO S1", MIMO_OPTIONS, index=0, key=f"mimo_{s}")
        else:
            label = f"MIMO S{s}"
            if use_all:
                sector_mimos[label] = sector_mimos["MIMO S1"]
                st.selectbox(label, MIMO_OPTIONS, index=MIMO_OPTIONS.index(sector_mimos["MIMO S1"]), key=f"mimo_{s}", disabled=True)
            else:
                sector_mimos[label] = st.selectbox(label, MIMO_OPTIONS, index=0, key=f"mimo_{s}")

    up = st.file_uploader("Upload PDF Design", type=["pdf"])
    exclude = st.checkbox("Exclude this review from analytics", value=False)
    do_spell = st.checkbox("Run spelling pass", value=True)

    if st.button("Run Audit", type="primary", disabled=(up is None)):
        if up is None:
            st.warning("Please upload a PDF design.")
            return

        original_bytes = up.read()
        pages = extract_pages_text(original_bytes)

        meta = {
            "Supplier": supplier,
            "Drawing Type": drawing_type,
            "Client": client,
            "Project": project,
            "Site Type": site_type,
            "Proposed Vendor": vendor,
            "Proposed Cabinet Location": cab_loc,
            "Proposed Radio Location": radio_loc,
            "Quantity of Sectors": qty,
            "Site Address": site_addr,
        }
        for k,v in sector_mimos.items():
            meta[k] = v

        # Load rules
        rules_file = REPO_ROOT / st.session_state.get("rules_file_name", RULES_FILE_DEFAULT)
        rules = load_rules(rules_file)

        findings = []
        findings += apply_rules(pages, meta, rules)
        findings += address_vs_title_check(pages, meta)

        if do_spell:
            # allow-list seeds from rules
            allow = set(w.lower() for w in rules.get("dictionary", []))
            findings += spelling_checks(pages, allow)

        # Decide PASS/REJECT
        status = "REJECTED" if any(f.get("severity","minor") == "major" for f in findings) else "PASS"

        # Outputs
        stem = Path(up.name).stem + f" - {status} - {dt.datetime.utcnow().strftime('%Y%m%d')}"
        excel_bytes = make_excel(findings, meta, up.name, status)
        pdf_bytes = annotate_pdf(original_bytes, findings, meta, status)
        zip_bytes = bundle_zip(excel_bytes, pdf_bytes, stem)

        # Persist files
        excel_name = safe_file_name(stem, ".xlsx")
        pdf_name = safe_file_name(stem, ".pdf")
        zip_name = safe_file_name(stem, ".zip")
        (HISTORY_DIR / excel_name).write_bytes(excel_bytes)
        (HISTORY_DIR / pdf_name).write_bytes(pdf_bytes)
        (HISTORY_DIR / zip_name).write_bytes(zip_bytes)

        # History row
        row = {
            "timestamp_utc": now_utc_iso(),
            "status": status,
            "client": client,
            "project": project,
            "site_type": site_type,
            "vendor": vendor,
            "cabinet_location": cab_loc,
            "radio_location": radio_loc,
            "drawing_type": drawing_type,
            "supplier": supplier,
            "qty_sectors": qty,
            "mimo_s1": meta.get("MIMO S1",""),
            "mimo_s2": meta.get("MIMO S2",""),
            "mimo_s3": meta.get("MIMO S3",""),
            "mimo_s4": meta.get("MIMO S4",""),
            "mimo_s5": meta.get("MIMO S5",""),
            "mimo_s6": meta.get("MIMO S6",""),
            "site_address": site_addr,
            "file_name": up.name,
            "excel_name": excel_name,
            "pdf_name": pdf_name,
            "zip_name": zip_name,
            "exclude": bool(exclude),
        }
        append_history(row)

        st.success(f"Audit complete → {status}")
        c1,c2,c3 = st.columns(3)
        c1.download_button("⬇️ Excel report", excel_bytes, file_name=excel_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        c2.download_button("⬇️ Annotated PDF", pdf_bytes, file_name=pdf_name, mime="application/pdf")
        c3.download_button("⬇️ Both (ZIP)", zip_bytes, file_name=zip_name, mime="application/zip")

def analytics_tab():
    st.markdown("### Analytics")
    df = load_history()
    if df.empty:
        st.info("No history yet.")
        return

    # Filter UI
    c1, c2, c3 = st.columns(3)
    supplier_f = c1.selectbox("Supplier filter", ["(All)"] + SUPPLIERS, index=0)
    project_f = c2.selectbox("Project filter", ["(All)"] + PROJECTS, index=0)
    client_f  = c3.selectbox("Client filter", ["(All)"] + CLIENTS, index=0)
    dfv = df[df["exclude"] != True].copy()
    if supplier_f != "(All)":
        dfv = dfv[dfv["supplier"] == supplier_f]
    if project_f != "(All)":
        dfv = dfv[dfv["project"] == project_f]
    if client_f != "(All)":
        dfv = dfv[dfv["client"] == client_f]

    total = len(dfv)
    rft = int((dfv["status"] == "PASS").sum())
    rft_pct = (rft/total*100) if total else 0
    rej = (dfv["status"] == "REJECTED").sum()

    k1,k2,k3 = st.columns(3)
    k1.metric("Total audits", total)
    k2.metric("Right First Time", f"{rft_pct:.1f}%")
    k3.metric("Rejected", rej)

    st.markdown("#### Recent audits")
    show = dfv.sort_values("timestamp_utc", ascending=False).head(50)
    st.dataframe(show[[
        "timestamp_utc","supplier","client","project","status","pdf_name","excel_name"
    ]], use_container_width=True)

def training_tab():
    st.markdown("### Training / Feedback")
    st.write("Upload a previous result to **teach** the tool. Mark as **Valid** if the tool was right, or **Not Valid** if it was wrong and enter the correction (which can become a rule).")

    uploaded = st.file_uploader("Upload Excel report (Findings) or JSON export", type=["xlsx","json"], key="train_upl")
    valid = st.radio("Verdict", ["Valid","Not Valid"], horizontal=True)
    corr_text = st.text_area("Correction (new rule idea or mapping)", placeholder="e.g., AHEGC should be AHEGG; or 'Title block must include Drawing Number'")

    if st.button("Record feedback"):
        rec = {
            "timestamp_utc": now_utc_iso(),
            "type": "feedback",
            "verdict": valid,
            "note": corr_text.strip(),
            "attachment_name": uploaded.name if uploaded else "",
        }
        # Store quick feedback as JSON alongside history for ops review
        fb_name = safe_file_name(f"feedback_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}", ".json")
        (HISTORY_DIR / fb_name).write_text(json.dumps(rec, indent=2))
        st.success("Feedback captured. You can later convert Not Valid items into new rules from Settings.")

def settings_tab():
    st.markdown("### Settings")
    # Logo file name override
    st.text_input("Logo file in repo root (optional, e.g., `88F3...png`)", key="logo_file", help="Just the file name, placed in the repository root.")

    # Choose rules file
    st.text_input("Rules file name", key="rules_file_name", value=st.session_state.get("rules_file_name", RULES_FILE_DEFAULT))

    # Rules editor gate
    pw = st.text_input("Rules password", type="password")
    rules_file = REPO_ROOT / (st.session_state.get("rules_file_name") or RULES_FILE_DEFAULT)
    curr = load_rules(rules_file)

    txt = st.text_area(rules_file.name, value=yaml.safe_dump(curr, sort_keys=False, allow_unicode=True), height=380)
    c1,c2 = st.columns([1,1])
    if c1.button("Save rules"):
        if pw != RULES_EDIT_PASSWORD:
            st.error("Wrong rules password.")
        else:
            try:
                new_data = yaml.safe_load(txt) or {}
                if not isinstance(new_data, dict):
                    raise ValueError("Rules must be a mapping at top level.")
                save_yaml_safe(rules_file, new_data)
                st.success("Rules saved.")
            except Exception as e:
                st.error(f"YAML error: {e}")

    if c2.button("Reload from disk"):
        st.experimental_rerun()

    st.markdown("---")
    if st.button("Clear all history (keeps files)"):
        ensure_history_headers()
        pd.DataFrame(columns=pd.read_csv(HISTORY_CSV).columns).to_csv(HISTORY_CSV, index=False)
        st.success("History cleared (files remain in /history).")

# -------------------------------------------------------------------------------------

def main():
    if not gate_with_password():
        st.stop()

    logo_header()
    st.write("Professional, consistent design QA with audit trail, annotations, analytics, and easy rule updates.")

    tab1, tab2, tab3, tab4 = st.tabs(["Audit", "Analytics", "Training", "Settings"])

    with tab1:
        audit_tab()
    with tab2:
        analytics_tab()
    with tab3:
        training_tab()
    with tab4:
        settings_tab()

if __name__ == "__main__":
    main()
