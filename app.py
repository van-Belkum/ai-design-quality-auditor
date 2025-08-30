# app.py
import os, io, base64, re, json, textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st
import pandas as pd
import yaml

# Optional PDF libs (only used if a PDF is uploaded)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

APP_TITLE = "AI Design Quality Auditor"
ENTRY_PASSWORD = "Seker123"
RULES_EDIT_PASSWORD = "vanB3lkum21"

HISTORY_DIR = Path("history")
EXPORT_DIR = HISTORY_DIR / "exports"
HISTORY_CSV = HISTORY_DIR / "audit_history.csv"
RULES_FILE = Path("rules_example.yaml")

# ------------------------ Utilities ------------------------

def ensure_dirs():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def load_rules(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data

def save_rules(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def get_logo_b64() -> str | None:
    # Priority: secrets LOGO_FILE -> first matching in repo root
    cand = st.secrets.get("LOGO_FILE", "").strip() if hasattr(st, "secrets") else ""
    root = Path(".")
    picks = []
    if cand:
        p = root / cand
        if p.exists():
            picks.append(p)
    if not picks:
        for ext in (".png", ".jpg", ".jpeg", ".svg"):
            for p in root.glob(f"*{ext}"):
                picks.append(p)
        # de-dupe
        seen = set()
        picks = [p for p in picks if not (p.name in seen or seen.add(p.name))]
    if not picks:
        return None
    p0 = picks[0]
    try:
        data = p0.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return None

def inject_logo_css():
    b64 = get_logo_b64()
    if not b64:
        st.warning("⚠️ Logo file not found in repo root (png/svg/jpg). Set `LOGO_FILE` in Secrets or add an image.")
        return
    is_svg = b64.strip().startswith("PHN2Zy")  # base64 of "<svg"
    mime = "image/svg+xml" if is_svg else "image/png"
    st.markdown(
        f"""
        <style>
        .app-logo {{
            position: fixed;
            top: 10px;
            left: 18px;
            z-index: 9999;
            opacity: 0.95;
        }}
        .app-logo img {{
            height: 42px;      /* adjust */
        }}
        @media (max-width: 600px){{
            .app-logo img {{ height: 34px; }}
        }}
        </style>
        <div class="app-logo">
            <img src="data:{mime};base64,{b64}" alt="logo"/>
        </div>
        """,
        unsafe_allow_html=True,
    )

def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def address_matches_title(site_address: str, pdf_title: str) -> bool:
    # Ignore if ", 0 ," present
    if ", 0 ," in site_address.replace(",0,", ", 0 ,"):
        return True
    sa = normalize(site_address)
    pt = normalize(pdf_title or "")
    # Require that the address tokens appear in the title string
    # Relaxed contains check
    return sa and sa[:12] in pt  # small heuristic: at least first 12 normalized chars

def safe_read_pdf_title(pdf_bytes: bytes) -> str:
    if not fitz:
        return ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        title = (doc.metadata or {}).get("title") or ""
        if not title:
            # fallback: take first line of first page text
            if doc.page_count > 0:
                t = doc.load_page(0).get_text("text")
                first_line = (t or "").strip().splitlines()[0] if t else ""
                title = first_line
        doc.close()
        return title
    except Exception:
        return ""

def annotate_pdf_simple(pdf_bytes: bytes, find_rows: list[dict]) -> bytes | None:
    """Adds a top-of-page note with issues per page. Avoids bbox complexity."""
    if not fitz:
        return None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        by_page: dict[int, list[str]] = {}
        for r in find_rows:
            p = int(r.get("page", 1))
            msg = r.get("message", "")
            sev = r.get("severity", "Info")
            by_page.setdefault(p, []).append(f"[{sev}] {msg}")

        for pno, msgs in by_page.items():
            i = min(max(pno - 1, 0), doc.page_count - 1)
            page = doc.load_page(i)
            text = "Audit notes:\n" + "\n".join(f"• {m}" for m in msgs[:12])
            rect = fitz.Rect(36, 36, page.rect.width - 36, 160)
            page.draw_rect(rect, color=(0.95, 0.6, 0.1), fill=(0.98, 0.98, 0.88), width=0.6, opacity=0.9)
            page.insert_textbox(rect, text, fontsize=9, color=(0.1, 0.1, 0.1))
        out = io.BytesIO()
        doc.save(out)
        doc.close()
        return out.getvalue()
    except Exception:
        return None

def write_excel_report(findings_df: pd.DataFrame, meta: dict, out_path: Path):
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        meta_df = pd.DataFrame([meta])
        findings_df.to_excel(xw, index=False, sheet_name="Findings")
        meta_df.to_excel(xw, index=False, sheet_name="Metadata")

def init_history():
    ensure_dirs()
    if not HISTORY_CSV.exists():
        cols = [
            "timestamp_utc","supplier","drawing_type","client","project","site_type","vendor",
            "cabinet_location","radio_location","sectors","mimo_config","site_address",
            "original_filename","excel_report","annotated_pdf","findings_total",
            "findings_major","findings_minor","rft_percent","exclude"
        ]
        pd.DataFrame(columns=cols).to_csv(HISTORY_CSV, index=False)

def append_history(row: dict):
    init_history()
    df = pd.read_csv(HISTORY_CSV)
    # ensure columns
    for k in row.keys():
        if k not in df.columns:
            df[k] = None
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_CSV, index=False)

def read_history() -> pd.DataFrame:
    init_history()
    df = pd.read_csv(HISTORY_CSV)
    # type fix
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    if "exclude" in df.columns:
        df["exclude"] = df["exclude"].fillna(False).astype(bool)
    return df

# ------------------------ Static choices ------------------------

SUPPLIERS = [
    "— Select —","CEG","CTIL","Emfyser","Innov8","Invict","KTL Team (Internal)","Trylon"
]

DRAWING_TYPES = ["— Select —","General Arrangement","Detailed Design"]

CLIENTS = ["— Select —","BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"]

PROJECTS = ["— Select —","RAN","Power Resilience","East Unwind","Beacon 4"]

SITE_TYPES = ["— Select —","Greenfield","Rooftop","Streetworks"]

VENDORS = ["— Select —","Ericsson","Nokia"]

CABINET_LOCS = ["— Select —","Indoor","Outdoor"]

RADIO_LOCS = ["— Select —","High Level","Low Level","Indoor","Door"]

SECTORS = ["— Select —","1","2","3","4","5","6"]

MIMO_OPTIONS = [
    "— Select —",
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
    "18\\21\\26 @2x2; 3500 @8x8",
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

# ------------------------ Streamlit App ------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
inject_logo_css()
ensure_dirs()
init_history()

# --- Login ---
if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    st.title(APP_TITLE)
    pw = st.text_input("Enter password to continue", type="password", help="Provided to your team.")
    if st.button("Unlock"):
        if pw == ENTRY_PASSWORD:
            st.session_state.authed = True
            st.experimental_rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Audit","Analytics","Settings"], index=0)

# Common header
st.title(APP_TITLE)
st.caption("Professional, consistent design QA with audit trail, annotations, analytics, and simple rule updates.")

# ------------- AUDIT PAGE -------------
if page == "Audit":
    with st.form("audit_form", clear_on_submit=False):
        st.subheader("Audit Metadata (required)")
        c1,c2,c3,c4 = st.columns(4)
        supplier = c1.selectbox("Supplier", SUPPLIERS, index=0)
        drawing_type = c2.selectbox("Drawing Type", DRAWING_TYPES, index=0)
        client = c3.selectbox("Client", CLIENTS, index=0)
        project = c4.selectbox("Project", PROJECTS, index=0)

        c5,c6,c7,c8 = st.columns(4)
        site_type = c5.selectbox("Site Type", SITE_TYPES, index=0)
        vendor = c6.selectbox("Proposed Vendor", VENDORS, index=0)
        cabinet_location = c7.selectbox("Proposed Cabinet Location", CABINET_LOCS, index=0)
        radio_location = c8.selectbox("Proposed Radio Location", RADIO_LOCS, index=0)

        c9,c10 = st.columns(2)
        sectors = c9.selectbox("Quantity of Sectors", SECTORS, index=0)
        site_address = c10.text_input("Site Address", placeholder="e.g. MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ")

        # MIMO: Mandatory unless Power Resilience
        is_power_res = project.strip().lower() == "power resilience"
        mimo_label = "Proposed MIMO Config (optional for Power Resilience)" if is_power_res else "Proposed MIMO Config"
        mimo = st.selectbox(mimo_label, MIMO_OPTIONS, index=0, help="Choose a standard configuration")

        st.divider()
        st.subheader("Design Documents")
        pdf_file = st.file_uploader("Upload design PDF", type=["pdf"])
        st.caption("Tip: The Site Address must match the PDF Title (ignored if the address contains \", 0 ,\").")

        cA,cB,cC = st.columns([1,1,2])
        exclude_from_analytics = cA.checkbox("Exclude this run from analytics", value=False)
        run_btn = cB.form_submit_button("Run Audit", use_container_width=True, type="primary")
        clear_btn = cC.form_submit_button("Clear Metadata")

    if clear_btn:
        for k in ("supplier","drawing_type","client","project","site_type","vendor",
                  "cabinet_location","radio_location","sectors","mimo","site_address"):
            st.session_state.pop(k, None)
        st.experimental_rerun()

    if run_btn:
        # Validate metadata
        errors = []
        req_pairs = {
            "Supplier": supplier, "Drawing Type": drawing_type, "Client": client, "Project": project,
            "Site Type": site_type, "Proposed Vendor": vendor, "Proposed Cabinet Location": cabinet_location,
            "Proposed Radio Location": radio_location, "Quantity of Sectors": sectors, "Site Address": site_address.strip()
        }
        for label, val in req_pairs.items():
            if (isinstance(val,str) and (val.startswith("— ") or val == "")) or val is None:
                errors.append(f"{label} is required.")
        if not is_power_res and (mimo.startswith("— ") or mimo == "(blank)"):
            errors.append("Proposed MIMO Config is required for non–Power Resilience projects.")
        if not pdf_file:
            errors.append("Please upload a PDF.")

        if errors:
            st.error("Please fix the following:\n\n" + "\n".join(f"• {e}" for e in errors))
        else:
            pdf_bytes = pdf_file.read()
            pdf_title = safe_read_pdf_title(pdf_bytes)
            addr_ok = address_matches_title(site_address, pdf_title)

            rules = load_rules(RULES_FILE)

            # Build findings
            findings = []
            if not addr_ok:
                findings.append({
                    "severity":"Major",
                    "category":"Metadata",
                    "message":"Site Address does not appear to match PDF title.",
                    "page":1
                })

            # Placeholder: you can iterate your YAML rules here and append real findings.

            # Score
            majors = sum(1 for f in findings if f["severity"].lower()=="major")
            minors = sum(1 for f in findings if f["severity"].lower()=="minor")
            total = len(findings)
            rft = 0 if total>0 else 100

            df_find = pd.DataFrame(findings if findings else [{"severity":"Info","category":"General","message":"No issues found.","page":1}])

            # File outputs
            ts_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
            base = Path(pdf_file.name).stem
            report_name = f"{base}_{'Rejected' if total>0 else 'Pass'}_{ts_tag}.xlsx"
            pdf_name = f"{base}_annotated_{ts_tag}.pdf"

            excel_path = EXPORT_DIR / report_name
            pdf_out_path = EXPORT_DIR / pdf_name

            meta = {
                "timestamp_utc": now_utc_iso(),
                "supplier": supplier, "drawing_type": drawing_type, "client": client, "project": project,
                "site_type": site_type, "vendor": vendor, "cabinet_location": cabinet_location,
                "radio_location": radio_location, "sectors": sectors, "mimo_config": mimo, "site_address": site_address,
                "original_filename": pdf_file.name, "pdf_title": pdf_title
            }
            write_excel_report(df_find, meta, excel_path)

            pdf_annot_bytes = annotate_pdf_simple(pdf_bytes, df_find.to_dict("records")) or pdf_bytes
            pdf_out_path.write_bytes(pdf_annot_bytes)

            # Show links
            st.success(f"Audit complete. Findings: {total} (Major {majors} / Minor {minors}) — RFT {rft}%")
            st.download_button("⬇️ Download Excel Report", excel_path.read_bytes(), file_name=report_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button("⬇️ Download Annotated PDF", pdf_out_path.read_bytes(), file_name=pdf_name, mime="application/pdf")

            # History row
            append_history({
                "timestamp_utc": meta["timestamp_utc"],
                "supplier": supplier, "drawing_type": drawing_type, "client": client, "project": project,
                "site_type": site_type, "vendor": vendor, "cabinet_location": cabinet_location,
                "radio_location": radio_location, "sectors": sectors, "mimo_config": mimo, "site_address": site_address,
                "original_filename": pdf_file.name, "excel_report": str(excel_path), "annotated_pdf": str(pdf_out_path),
                "findings_total": total, "findings_major": majors, "findings_minor": minors, "rft_percent": rft,
                "exclude": bool(exclude_from_analytics),
            })

# ------------- ANALYTICS PAGE -------------
elif page == "Analytics":
    st.subheader("Audit History & Analytics")
    df = read_history()
    if df.empty:
        st.info("No audits yet.")
    else:
        # Visibility: show all rows (we keep files for long-term)
        show_df = df.copy()
        show_df_view = show_df[[
            "timestamp_utc","supplier","client","project","drawing_type","site_type",
            "vendor","mimo_config","findings_total","findings_major","findings_minor","rft_percent","exclude","excel_report","annotated_pdf"
        ]].sort_values("timestamp_utc", ascending=False)
        st.dataframe(show_df_view, use_container_width=True, hide_index=True)

        # Simple KPIs (excluding excluded)
        work = show_df[~show_df["exclude"].fillna(False)]
        c1,c2,c3,c4 = st.columns(4)
        total_runs = len(work)
        major = int(work["findings_major"].sum()) if total_runs else 0
        minor = int(work["findings_minor"].sum()) if total_runs else 0
        rft = round((work["rft_percent"].mean() if total_runs else 0), 1)
        c1.metric("Audits (incl. included only)", total_runs)
        c2.metric("Major findings", major)
        c3.metric("Minor findings", minor)
        c4.metric("Avg RFT %", rft)

        # Trend by supplier
        st.markdown("#### Findings per Supplier (included only)")
        by_sup = work.groupby("supplier")[["findings_major","findings_minor"]].sum().sort_values("findings_major", ascending=False)
        st.bar_chart(by_sup, use_container_width=True)

        # Toggle exclude inline (quick manage)
        with st.expander("Manage 'Exclude from analytics' flags"):
            idx_map = {i: r for i,r in work.reset_index().iterrows()}
            for i, row in idx_map.items():
                cols = st.columns([2,5,2,2])
                cols[0].write(f"**{row['timestamp_utc']}**")
                cols[1].write(f"{row['supplier']} • {row['client']} • {row['project']}")
                key = f"exc_{i}"
                new_val = cols[2].checkbox("Exclude", value=bool(row["exclude"]), key=key)
                if cols[3].button("Save", key=f"s_{i}"):
                    # write back to csv
                    full = read_history()
                    mask = (full["timestamp_utc"] == row["timestamp_utc"])
                    full.loc[mask, "exclude"] = bool(new_val)
                    full.to_csv(HISTORY_CSV, index=False)
                    st.success("Saved.")
                    st.experimental_rerun()

# ------------- SETTINGS PAGE -------------
else:
    st.subheader("Rules & App Settings")
    st.caption("Update YAML rules safely. Changes are written to `rules_example.yaml`.")
    pass1 = st.text_input("Enter rules update password to proceed", type="password")
    if pass1 != RULES_EDIT_PASSWORD:
        st.info("Enter the update password to edit rules.")
    else:
        current = load_rules(RULES_FILE)
        text = st.text_area("rules_example.yaml", value=yaml.safe_dump(current, sort_keys=False, allow_unicode=True), height=360)
        c1,c2 = st.columns(2)
        if c1.button("Validate YAML"):
            try:
                data = yaml.safe_load(text) or {}
                st.success("YAML is valid.")
            except Exception as e:
                st.error(f"YAML error: {e}")
        if c2.button("Save YAML"):
            try:
                data = yaml.safe_load(text) or {}
                save_rules(RULES_FILE, data)
                st.success("Saved rules_example.yaml")
            except Exception as e:
                st.error(f"Failed to save: {e}")

    st.divider()
    st.markdown("**Logo**")
    st.caption("App tries to auto-detect the first image in repo root. Or set a specific name in *Secrets* as `LOGO_FILE`.")
    st.code("LOGO_FILE = \"88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png\"")
