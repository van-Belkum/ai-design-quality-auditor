# app.py
import io, os, re, json, base64, zipfile, datetime as dt
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import yaml
from rapidfuzz import process, fuzz
import fitz  # PyMuPDF

APP_TITLE = "AI Design Quality Auditor"
ENTRY_PASSWORD = "Seker123"
RULES_PASSWORD = "vanB3lkum21"

HISTORY_DIR = "history"
DAILY_EXPORT_DIR = "daily_exports"
RULES_FILE_DEFAULT = "rules_example.yaml"
LOGO_CANDIDATES = ["logo.png","logo.svg","logo.jpg","logo.jpeg","seker.png","Seker.png","Seker.svg"]

# ---- Metadata options ----
CLIENTS = ["BTEE", "Vodafone", "MBNL", "H3G", "Cornerstone", "Cellnex"]
PROJECTS = ["RAN","Power Resilience","East Unwind","Beacon 4"]
SITE_TYPES = ["Greenfield","Rooftop","Streetworks"]
VENDORS = ["Ericsson","Nokia"]
CAB_LOC = ["Indoor","Outdoor"]
RADIO_LOC = ["Low Level","High Level","Unique Coverage","Midway"]
SECTOR_QTY = [1,2,3,4,5,6]

# Supplier list exactly as provided
SUPPLIERS = ["CEG","CTIL","Emfyser","Innov8","Invict","KTL Team (Internal)","Trylon"]

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

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(DAILY_EXPORT_DIR, exist_ok=True)

# ---------------- Utility ----------------
def now_utc_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def detect_logo_path() -> Optional[str]:
    for name in LOGO_CANDIDATES:
        if os.path.exists(name): return name
    # also allow any lone image at repo root by extension
    for f in os.listdir("."):
        if re.search(r"\.(png|svg|jpg|jpeg)$", f, re.I): return f
    return None

def render_logo_top_left():
    path = detect_logo_path()
    if not path:
        return
    try:
        if path.lower().endswith(".svg"):
            with open(path,"r",encoding="utf-8") as f:
                svg = f.read()
            st.markdown(
                f"""
                <div style="position:fixed;left:16px;top:10px;z-index:1000;opacity:.95">{svg}</div>
                """,
                unsafe_allow_html=True
            )
        else:
            b64 = base64.b64encode(open(path,"rb").read()).decode()
            st.markdown(
                f"""
                <style>.brand-logo{{height:46px;border-radius:6px;}}</style>
                <img class="brand-logo" src="data:image/png;base64,{b64}" />
                """, unsafe_allow_html=True
            )
    except Exception:
        pass

def read_rules(path: str) -> Dict[str, Any]:
    if not os.path.exists(path): return {"checklist":[],"spelling":{"allow":[]}}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"checklist":[],"spelling":{"allow":[]}}

def save_rules(path: str, content: str):
    yaml.safe_load(content)  # validate
    with open(path,"w",encoding="utf-8") as f:
        f.write(content)

def _safe_filename(s: str) -> str:
    return re.sub(r"[^\w\-.]+","_", s).strip("_") or "file"

def load_history() -> pd.DataFrame:
    files = sorted([os.path.join(HISTORY_DIR,f) for f in os.listdir(HISTORY_DIR) if f.endswith(".csv")])
    if not files:
        cols = ["timestamp_utc","file_name","client","project","site_type","vendor",
                "cabinet_loc","radio_loc","sectors","mimo_s1","mimo_s2","mimo_s3","mimo_s4","mimo_s5","mimo_s6",
                "status","reject_count","minor_count","exclude_from_analytics","supplier",
                "site_address","notes","excel_path","annotated_pdf_path"]
        return pd.DataFrame(columns=cols)
    frames=[]
    for fn in files:
        try:
            frames.append(pd.read_csv(fn))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    return df

def append_history_row(row: Dict[str, Any]) -> None:
    day = dt.datetime.utcnow().strftime("%Y-%m-%d")
    path = os.path.join(HISTORY_DIR, f"history_{day}.csv")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)

# ---------- PDF precise annotation ----------
def _find_text_bbox(page: fitz.Page, needle: str) -> Optional[fitz.Rect]:
    if not needle or not needle.strip(): return None
    rects = page.search_for(needle, hit_max=1)
    if rects: return rects[0]
    tokens = sorted(re.findall(r"[A-Za-z0-9\-/@x]+", needle), key=len, reverse=True)
    for tok in tokens[:3]:
        r = page.search_for(tok, hit_max=1)
        if r: return r[0]
    return None

def attach_precise_bbox(f: Dict[str,Any], doc: fitz.Document) -> Dict[str,Any]:
    g = dict(f)
    p = g.get("page_num")
    text = g.get("context") or g.get("needle") or g.get("message")
    if isinstance(p,int) and 0<=p<len(doc) and text:
        try:
            rect = _find_text_bbox(doc[p], text)
            if rect: g["bbox"] = [rect.x0,rect.y0,rect.x1,rect.y1]
        except Exception: pass
    return g

def annotate_pdf_exact(original_pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> bytes:
    doc = fitz.open(stream=original_pdf_bytes, filetype="pdf")
    callout = 1
    for f in findings:
        p = f.get("page_num")
        if not isinstance(p,int) or not (0<=p<len(doc)): continue
        bbox = f.get("bbox")
        if not (isinstance(bbox,(list,tuple)) and len(bbox)==4):
            f = attach_precise_bbox(f, doc)
            bbox = f.get("bbox")
            if not bbox: continue
        rect = fitz.Rect(*bbox)
        page = doc[p]
        shape = page.new_shape()
        shape.draw_rect(rect)
        shape.finish(width=1.2,color=(1,0,0))
        shape.commit()
        label = f"[{callout}]"
        callout += 1
        tag = fitz.Rect(rect.x0, rect.y0-12, rect.x0+16, rect.y0-2)
        page.draw_rect(tag,color=(1,0,0),fill=(1,0.92,0.92),width=0.8)
        page.insert_textbox(tag,label,fontsize=8,color=(0.2,0,0))
        note = f.get("comment") or f.get("message") or ""
        if note:
            note_rect = fitz.Rect(page.rect.x1-190, rect.y0, page.rect.x1-10, rect.y0+60)
            if note_rect.x0 < rect.x1+12: note_rect = fitz.Rect(10, rect.y0, 190, rect.y0+60)
            page.draw_rect(note_rect,color=(0.6,0,0),fill=(1,0.97,0.97),width=0.7)
            page.insert_textbox(note_rect, f"{label} {note}", fontsize=8.5, color=(0.1,0.1,0.1))
            cx = note_rect.x0 if note_rect.x0>rect.x1 else note_rect.x1
            page.draw_line((rect.x1,rect.y0),(cx,note_rect.y0),color=(1,0,0),width=0.6)
    out = doc.tobytes()
    doc.close()
    return out

# ---------- Spelling ----------
def spelling_checks(pages: List[str], allow: List[str]) -> List[Dict[str,Any]]:
    wl = set()
    for p in pages:
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", p):
            wl.add(tok)
    wl = sorted(wl)
    findings=[]
    allow_set = {w.lower() for w in allow}
    for i, word in enumerate(wl):
        if word.lower() in allow_set: continue
        candidates = process.extract(word, allow, scorer=fuzz.ratio, limit=3)
        sug = candidates[0][0] if (candidates and candidates[0][1]>=90) else None
        if sug:
            findings.append({
                "rule_id":"SPELLING",
                "severity":"minor",
                "message": f"“{word}” looks like a typo → {sug}",
                "page_num": 0,
                "context": word
            })
    return findings

# ---------- Checklist rules ----------
def run_checklist(pages: List[str], rules: Dict[str,Any], meta: Dict[str,Any]) -> List[Dict[str,Any]]:
    checks = rules.get("checklist", [])
    findings=[]
    full_text = "\n".join(pages)
    for idx, rule in enumerate(checks, start=1):
        name = rule.get("name","")
        must = [str(x) for x in rule.get("must_contain",[])]
        forbid = [str(x) for x in rule.get("reject_if_present",[])]
        severity = rule.get("severity","minor")
        # condition on metadata
        cond = rule.get("when",{})
        ok = True
        for k,v in cond.items():
            if str(meta.get(k)) != str(v): ok=False; break
        if not ok: continue

        miss=[]
        for m in must:
            if m and (m not in full_text):
                miss.append(m)
        hit=[]
        for f in forbid:
            if f and (f in full_text):
                hit.append(f)
        if miss or hit:
            msg=[]
            if miss: msg.append(f"Missing: {', '.join(miss)}")
            if hit: msg.append(f"Forbidden present: {', '.join(hit)}")
            findings.append({
                "rule_id": f"RULE_{idx}",
                "severity": severity,
                "message": f"{name}: " + " | ".join(msg),
                "page_num": 0,
                "context": (miss+hit)[0] if (miss or hit) else name
            })
    return findings

# ---------- Exports ----------
def make_excel(findings: List[Dict[str,Any]], meta: Dict[str,Any], original_name: str, status: str) -> bytes:
    df = pd.DataFrame(findings) if findings else pd.DataFrame(columns=["rule_id","severity","message","page_num"])
    meta_row = pd.DataFrame([meta])
    with pd.ExcelWriter(io.BytesIO(), engine="openpyxl") as xls:
        df.to_excel(xls, index=False, sheet_name="Findings")
        meta_row.to_excel(xls, index=False, sheet_name="Metadata")
        xls_io = xls.book._writer.fp  # type: ignore
        xls_bytes = xls_io.getvalue()
    return xls_bytes

# ---------- UI helpers ----------
def require_entry_password():
    if "authed" not in st.session_state:
        st.session_state["authed"] = False
    if st.session_state["authed"]: return True
    pw = st.text_input("Enter access password", type="password")
    if st.button("Enter"):
        st.session_state["authed"] = (pw == ENTRY_PASSWORD)
        if not st.session_state["authed"]:
            st.error("Incorrect password")
    st.stop()

def audit_form():
    st.subheader("Audit Metadata (required)")
    cols1 = st.columns(4)
    supplier = cols1[0].selectbox("Supplier", SUPPLIERS, index=0)
    drawing_type = cols1[1].selectbox("Drawing Type", ["General Arrangement","Detailed Design"])
    client = cols1[2].selectbox("Client", CLIENTS)
    project = cols1[3].selectbox("Project", PROJECTS)

    cols2 = st.columns(4)
    site_type = cols2[0].selectbox("Site Type", SITE_TYPES)
    vendor = cols2[1].selectbox("Proposed Vendor", VENDORS)
    cab = cols2[2].selectbox("Proposed Cabinet Location", CAB_LOC)
    radio = cols2[3].selectbox("Proposed Radio Location", RADIO_LOC)

    cols3 = st.columns(2)
    sectors = cols3[0].selectbox("Quantity of Sectors", SECTOR_QTY, index=2)
    address = cols3[1].text_input("Site Address", "MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ")

    st.markdown("**Proposed MIMO Config** (optional unless required by client)")
    same_all = st.checkbox("Use same config for all sectors", value=True)
    mimo_s1 = st.selectbox("MIMO S1", MIMO_OPTIONS, index=0, key="mimo_s1")
    mimo_values = {"mimo_s1":mimo_s1}
    for i in range(2, sectors+1):
        key = f"mimo_s{i}"
        if same_all:
            st.selectbox(f"MIMO S{i}", MIMO_OPTIONS, index=MIMO_OPTIONS.index(mimo_s1), key=key, disabled=True)
            mimo_values[key]=mimo_s1
        else:
            mimo_values[key] = st.selectbox(f"MIMO S{i}", MIMO_OPTIONS, index=0, key=key)

    for i in range(sectors+1, 7):
        st.session_state.pop(f"mimo_s{i}", None)

    return {
        "supplier": supplier,
        "drawing_type": drawing_type,
        "client": client,
        "project": project,
        "site_type": site_type,
        "vendor": vendor,
        "cabinet_loc": cab,
        "radio_loc": radio,
        "sectors": sectors,
        "site_address": address,
        **mimo_values
    }

# ---------- Main ----------
def main():
    st.set_page_config(APP_TITLE, layout="wide")
    render_logo_top_left()
    st.title(APP_TITLE)
    require_entry_password()

    tabs = st.tabs(["Audit","Training","Analytics","Settings"])

    # ---------------- Audit ----------------
    with tabs[0]:
        meta = audit_form()
        up = st.file_uploader("Upload a multi-page PDF (design for audit)", type=["pdf"])
        rules_file = st.text_input("Rules file name", RULES_FILE_DEFAULT)
        exclude = st.checkbox("Exclude this review from analytics", value=False)
        run = st.button("Run audit")
        if run and up:
            rules = read_rules(rules_file)
            pdf_bytes = up.read()
            # naive text extraction per page
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            for i in range(len(doc)):
                try:
                    pages.append(doc[i].get_text("text"))
                except Exception:
                    pages.append("")
            doc.close()

            # Address/title check (reject if not matching unless has , 0 ,)
            fn_title = os.path.splitext(up.name)[0]
            addr_ok = (", 0 ," in meta["site_address"]) or (fn_title.upper().replace(" ","") in meta["site_address"].upper().replace(" ",""))

            findings = []
            findings += run_checklist(pages, rules, meta)
            findings += spelling_checks(pages, rules.get("spelling",{}).get("allow",[]))

            if not addr_ok:
                findings.append({
                    "rule_id":"ADDR",
                    "severity":"major",
                    "message":"Site address does not match PDF title (ignored if contains ', 0 ,').",
                    "page_num":0,
                    "context": fn_title
                })

            reject_count = sum(1 for f in findings if f.get("severity","minor").lower()=="major")
            status = "Rejected" if reject_count>0 else "Pass"

            # Exports
            excel_bytes = make_excel(findings, meta, up.name, status)
            annotated_pdf = annotate_pdf_exact(pdf_bytes, findings)

            stamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            base = _safe_filename(os.path.splitext(up.name)[0])
            excel_name = f"{base}_{status}_{stamp}.xlsx"
            pdf_name = f"{base}_{status}_{stamp}.annotated.pdf"

            st.download_button("⬇️ Download Findings (Excel)", data=excel_bytes, file_name=excel_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button("⬇️ Download Annotated PDF", data=annotated_pdf, file_name=pdf_name, mime="application/pdf")

            # Persist files to history for later access
            excel_path = os.path.join(HISTORY_DIR, excel_name)
            pdf_path = os.path.join(HISTORY_DIR, pdf_name)
            with open(excel_path,"wb") as f: f.write(excel_bytes)
            with open(pdf_path,"wb") as f: f.write(annotated_pdf)

            row = {
                "timestamp_utc": now_utc_iso(),
                "file_name": up.name,
                "client": meta["client"],
                "project": meta["project"],
                "site_type": meta["site_type"],
                "vendor": meta["vendor"],
                "cabinet_loc": meta["cabinet_loc"],
                "radio_loc": meta["radio_loc"],
                "sectors": meta["sectors"],
                "mimo_s1": meta.get("mimo_s1",""),
                "mimo_s2": meta.get("mimo_s2",""),
                "mimo_s3": meta.get("mimo_s3",""),
                "mimo_s4": meta.get("mimo_s4",""),
                "mimo_s5": meta.get("mimo_s5",""),
                "mimo_s6": meta.get("mimo_s6",""),
                "status": status,
                "reject_count": reject_count,
                "minor_count": sum(1 for f in findings if f.get("severity")=="minor"),
                "exclude_from_analytics": exclude,
                "supplier": meta["supplier"],
                "site_address": meta["site_address"],
                "notes": "",
                "excel_path": excel_path,
                "annotated_pdf_path": pdf_path
            }
            append_history_row(row)
            st.success(f"Audit complete: **{status}**. Saved to history.")

    # ---------------- Training ----------------
    with tabs[1]:
        st.subheader("Upload a reviewed Excel to teach Valid / Not-valid")
        upx = st.file_uploader("Upload a **Findings Excel** that you've marked as Valid/Not-valid", type=["xlsx"], key="trainup")
        if upx:
            try:
                dfx = pd.read_excel(upx, sheet_name="Findings")
                # Expect optional columns user might add: valid (True/False), comment, new_rule
                valid_cnt = int(dfx.get("valid", pd.Series([True]*len(dfx))).sum())
                total = len(dfx)
                st.success(f"Loaded {total} findings; {valid_cnt} marked valid.")
                # Show and allow quick add to rules
                st.dataframe(dfx)
                st.info("Use Settings → Rules editor to paste rules for persistent updates.")
            except Exception as e:
                st.error("Could not read training Excel. Ensure it has a 'Findings' sheet.")

    # ---------------- Analytics ----------------
    with tabs[2]:
        st.subheader("Analytics")
        dfh = load_history()
        if dfh.empty:
            st.info("No history yet.")
        else:
            # filters
            cols = st.columns(4)
            supplier_filter = cols[0].selectbox("Supplier", ["All"] + SUPPLIERS, index=0)
            proj_filter = cols[1].selectbox("Project", ["All"] + PROJECTS, index=0)
            client_filter = cols[2].selectbox("Client", ["All"] + CLIENTS, index=0)
            include_excl = cols[3].checkbox("Include excluded", value=False)

            dfv = dfh.copy()
            if not include_excl:
                dfv = dfv[dfv["exclude_from_analytics"]!=True]
            if supplier_filter!="All":
                dfv = dfv[dfv["supplier"]==supplier_filter]
            if proj_filter!="All":
                dfv = dfv[dfv["project"]==proj_filter]
            if client_filter!="All":
                dfv = dfv[dfv["client"]==client_filter]

            total = len(dfv)
            passed = int((dfv["status"]=="Pass").sum())
            rft = (passed/total*100.0) if total else 0.0

            c1,c2,c3 = st.columns(3)
            c1.metric("Audits", f"{total:,}")
            c2.metric("Pass", f"{passed:,}")
            c3.metric("RFT %", f"{rft:0.1f}%")

            st.dataframe(dfv.sort_values("timestamp_utc", ascending=False)[
                ["timestamp_utc","file_name","supplier","client","project","status","reject_count","excel_path","annotated_pdf_path"]
            ])

            # quick downloads from history table
            sel = st.text_input("Enter exact file_name to re-download exports")
            if sel:
                row = dfv[dfv["file_name"]==sel].tail(1)
                if not row.empty:
                    ep = row.iloc[0]["excel_path"]
                    pp = row.iloc[0]["annotated_pdf_path"]
                    if os.path.exists(ep):
                        st.download_button("Download Excel (history)", open(ep,"rb").read(), file_name=os.path.basename(ep))
                    if os.path.exists(pp):
                        st.download_button("Download Annotated PDF (history)", open(pp,"rb").read(), file_name=os.path.basename(pp))

    # ---------------- Settings ----------------
    with tabs[3]:
        st.subheader("Rules & housekeeping")
        pw = st.text_input("Rules password", type="password")
        rules_file = st.text_input("Rules file name", RULES_FILE_DEFAULT, key="rulesfile_edit")
        content = st.text_area(rules_file, value=open(rules_file,"r",encoding="utf-8").read() if os.path.exists(rules_file) else "checklist: []\nspelling:\n  allow: []\n", height=300)
        c1,c2 = st.columns(2)
        if c1.button("Save rules"):
            if pw != RULES_PASSWORD:
                st.error("Wrong password.")
            else:
                try:
                    save_rules(rules_file, content)
                    st.success("Rules saved.")
                except Exception as e:
                    st.error(f"YAML error: {e}")
        if c2.button("Reload from disk"):
            st.experimental_rerun()

        st.divider()
        if st.button("Clear all history (keeps files)"):
            for f in os.listdir(HISTORY_DIR):
                if f.endswith(".csv"):
                    os.remove(os.path.join(HISTORY_DIR,f))
            st.success("Cleared CSV history (export files kept).")

if __name__ == "__main__":
    main()
