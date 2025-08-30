# app.py
import os, io, base64, re
from pathlib import Path
from datetime import datetime, timezone
import streamlit as st
import pandas as pd
import yaml

# Optional PDF annotation
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

APP_TITLE = "AI Design Quality Auditor"

# Passwords
ENTRY_PASSWORD = "Seker123"
RULES_EDIT_PASSWORD = "vanB3lkum21"

# Paths
HISTORY_DIR = Path("history")
EXPORT_DIR = HISTORY_DIR / "exports"
HISTORY_CSV = HISTORY_DIR / "audit_history.csv"
RULES_FILE = Path("rules_example.yaml")

# ------------------------ Helpers ------------------------
def ensure_dirs():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def now_iso_utc():
    return datetime.now(timezone.utc).isoformat()

def load_rules(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_rules(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def get_logo_b64() -> str | None:
    # Order: Settings override -> Secrets LOGO_FILE -> first image in repo root
    override = st.session_state.get("logo_file_override", "").strip()
    cand = override or (st.secrets.get("LOGO_FILE", "").strip() if hasattr(st, "secrets") else "")
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
                break
    if not picks:
        return None
    try:
        data = picks[0].read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return None

def inject_logo_top_left():
    b64 = get_logo_b64()
    if not b64:
        st.warning("⚠️ Logo file not found. Set **Settings → Logo override** or add a png/jpg/svg to repo root.")
        return
    # Heuristic: if base64 begins with "<svg" encoded ("PHN2Zy") treat as svg
    is_svg = b64.startswith("PHN2Zy")
    mime = "image/svg+xml" if is_svg else "image/png"
    st.markdown(
        f"""
        <style>
        .brand-logo {{
          position: fixed; top: 10px; left: 18px; z-index: 9999; opacity: .98;
        }}
        .brand-logo img {{ height: 44px; }}
        @media (max-width: 640px) {{
          .brand-logo img {{ height: 36px; }}
        }}
        </style>
        <div class="brand-logo">
          <img src="data:{mime};base64,{b64}" alt="logo"/>
        </div>
        """,
        unsafe_allow_html=True
    )

def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def address_matches_title(site_address: str, pdf_title: str) -> bool:
    # Ignore check if address includes ", 0 ,"
    if ", 0 ," in site_address.replace(",0,", ", 0 ,"):
        return True
    a = normalize(site_address)
    t = normalize(pdf_title)
    return bool(a) and (a[:12] in t)

def safe_pdf_title(pdf_bytes: bytes) -> str:
    if not fitz:
        return ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        title = (doc.metadata or {}).get("title") or ""
        if not title and doc.page_count > 0:
            txt = doc.load_page(0).get_text("text") or ""
            title = (txt.strip().splitlines() or [""])[0]
        doc.close()
        return title
    except Exception:
        return ""

def annotate_pdf_simple(pdf_bytes: bytes, rows: list[dict]) -> bytes | None:
    if not fitz:
        return None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        notes = {}
        for r in rows:
            p = int(r.get("page", 1))
            msg = r.get("message", "")
            sev = r.get("severity", "Info")
            notes.setdefault(p, []).append(f"[{sev}] {msg}")
        for pno, msgs in notes.items():
            i = max(0, min(pno - 1, doc.page_count - 1))
            page = doc.load_page(i)
            rect = fitz.Rect(36, 36, page.rect.width - 36, 160)
            page.draw_rect(rect, fill=(0.98, 0.98, 0.90), color=(0.9, 0.6, 0.2), width=0.6)
            page.insert_textbox(rect, "Audit notes:\n" + "\n".join("• " + m for m in msgs[:12]), fontsize=9)
        out = io.BytesIO()
        doc.save(out)
        doc.close()
        return out.getvalue()
    except Exception:
        return None

def write_excel(find_df: pd.DataFrame, meta: dict, out_path: Path):
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        find_df.to_excel(xw, index=False, sheet_name="Findings")
        pd.DataFrame([meta]).to_excel(xw, index=False, sheet_name="Metadata")

def init_history_file():
    ensure_dirs()
    if not HISTORY_CSV.exists():
        cols = [
            "timestamp_utc","supplier","drawing_type","client","project","site_type","vendor",
            "cabinet_location","radio_location","sectors","mimo_config","site_address",
            "original_filename","excel_report","annotated_pdf",
            "findings_total","findings_major","findings_minor","rft_percent","exclude"
        ]
        pd.DataFrame(columns=cols).to_csv(HISTORY_CSV, index=False)

def read_history() -> pd.DataFrame:
    init_history_file()
    df = pd.read_csv(HISTORY_CSV)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    if "exclude" in df.columns:
        df["exclude"] = df["exclude"].fillna(False).astype(bool)
    return df

def append_history(row: dict):
    init_history_file()
    df = pd.read_csv(HISTORY_CSV)
    for k in row.keys():
        if k not in df.columns:
            df[k] = None
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_CSV, index=False)

# ------------------------ Static Choices ------------------------
SUPPLIERS = ["— Select —","CEG","CTIL","Emfyser","Innov8","Invict","KTL Team (Internal)","Trylon"]
DRAWING_TYPES = ["— Select —","General Arrangement","Detailed Design"]
CLIENTS = ["— Select —","BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"]
PROJECTS = ["— Select —","RAN","Power Resilience","East Unwind","Beacon 4"]
SITE_TYPES = ["— Select —","Greenfield","Rooftop","Streetworks"]
VENDORS = ["— Select —","Ericsson","Nokia"]
CABINET_LOCS = ["— Select —","Indoor","Outdoor"]
# Updated per request:
RADIO_LOCS = ["— Select —","Low Level","High Level","Unique Coverage","Midway"]
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
    "(blank)"
]

# ------------------------ App ------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
inject_logo_top_left()
ensure_dirs()
init_history_file()

# Login gate
if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    st.title(APP_TITLE)
    pw = st.text_input("Enter password", type="password")
    if st.button("Unlock"):
        if pw == ENTRY_PASSWORD:
            st.session_state.authed = True
            st.experimental_rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Audit","Analytics","Settings"], index=0)

st.title(APP_TITLE)
st.caption("Professional design QA with audit trail, annotations, analytics, and simple rule updates.")

# ------------------------ AUDIT ------------------------
if page == "Audit":
    with st.form("audit_form"):
        st.subheader("Audit Metadata (required)")
        c1,c2,c3,c4 = st.columns(4)
        supplier = c1.selectbox("Supplier", SUPPLIERS)
        drawing_type = c2.selectbox("Drawing Type", DRAWING_TYPES)
        client = c3.selectbox("Client", CLIENTS)
        project = c4.selectbox("Project", PROJECTS)

        c5,c6,c7,c8 = st.columns(4)
        site_type = c5.selectbox("Site Type", SITE_TYPES)
        vendor = c6.selectbox("Proposed Vendor", VENDORS)
        cabinet_location = c7.selectbox("Proposed Cabinet Location", CABINET_LOCS)
        radio_location = c8.selectbox("Proposed Radio Location", RADIO_LOCS)

        c9,c10 = st.columns(2)
        sectors_choice = c9.selectbox("Quantity of Sectors", SECTORS)
        site_address = c10.text_input("Site Address", placeholder="e.g. MANBY ROAD , 0 , IMMINGHAM , IMMINGHAM , DN40 2LQ")

        is_power_res = project.lower() == "power resilience"
        st.markdown("##### Proposed MIMO Config{}".format(" (optional for Power Resilience)" if is_power_res else ""))

        same_all = st.checkbox("Use same config for all sectors", value=True, disabled=is_power_res and False)
        mimo_all = None
        sector_count = int(sectors_choice) if sectors_choice.isdigit() else 0

        if same_all:
            mimo_all = st.selectbox("MIMO (all sectors)", MIMO_OPTIONS, key="mimo_all")
            mimo_per_sector = [mimo_all] * max(1, sector_count)
        else:
            cols = st.columns(min(3, max(1, sector_count)))
            mimo_per_sector = []
            for i in range(sector_count):
                col = cols[i % len(cols)]
                mimo_per_sector.append(col.selectbox(f"MIMO S{i+1}", MIMO_OPTIONS, key=f"mimo_s{i+1}"))

        st.divider()
        st.subheader("Design Documents")
        pdf_file = st.file_uploader("Upload design PDF", type=["pdf"])
        st.caption("Address must match the PDF title (ignored if it contains \", 0 ,\").")

        cA,cB,cC = st.columns([1,1,2])
        exclude_analytics = cA.checkbox("Exclude this run from analytics", value=False)
        run_btn = cB.form_submit_button("Run Audit", use_container_width=True, type="primary")
        clear_btn = cC.form_submit_button("Clear Metadata")

    if clear_btn:
        for k in list(st.session_state.keys()):
            if k.startswith("mimo_s") or k in ("mimo_all","logo_file_override"):
                st.session_state.pop(k, None)
        st.experimental_rerun()

    if run_btn:
        errs = []
        # Required fields
        req = {
            "Supplier": supplier, "Drawing Type": drawing_type, "Client": client, "Project": project,
            "Site Type": site_type, "Proposed Vendor": vendor, "Proposed Cabinet Location": cabinet_location,
            "Proposed Radio Location": radio_location, "Quantity of Sectors": sectors_choice, "Site Address": site_address.strip()
        }
        for label, val in req.items():
            if (isinstance(val, str) and (val == "" or val.startswith("— "))) or val is None:
                errs.append(f"{label} is required.")
        if not is_power_res:
            # validate all MIMO selections
            if sector_count < 1:
                errs.append("Quantity of Sectors must be 1–6.")
            for i, mv in enumerate(mimo_per_sector or []):
                if (not mv) or mv.startswith("— ") or mv == "(blank)":
                    errs.append(f"MIMO for Sector {i+1} is required.")
        if not pdf_file:
            errs.append("Please upload a PDF.")
        if errs:
            st.error("Please fix the following:\n\n" + "\n".join(f"• {e}" for e in errs))
        else:
            pdf_bytes = pdf_file.read()
            pdf_title = safe_pdf_title(pdf_bytes)
            addr_ok = address_matches_title(site_address, pdf_title)

            rules = load_rules(RULES_FILE)  # (not used yet—hook for your rule engine)

            findings = []
            if not addr_ok:
                findings.append({"severity":"Major","category":"Metadata","message":"Site Address does not appear to match PDF title.","page":1})

            # (Hook) Add future rule checks here using `rules`.

            majors = sum(1 for f in findings if f["severity"].lower()=="major")
            minors = sum(1 for f in findings if f["severity"].lower()=="minor")
            total = len(findings)
            rft = 0 if total>0 else 100
            df_find = pd.DataFrame(findings or [{"severity":"Info","category":"General","message":"No issues found.","page":1}])

            # Output filenames
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            base = Path(pdf_file.name).stem
            report_name = f"{base}_{'Rejected' if total>0 else 'Pass'}_{ts}.xlsx"
            pdf_out_name = f"{base}_annotated_{ts}.pdf"
            excel_path = EXPORT_DIR / report_name
            pdf_out_path = EXPORT_DIR / pdf_out_name

            # MIMO string summary
            if sector_count <= 1:
                mimo_string = mimo_per_sector[0] if mimo_per_sector else ""
            else:
                mimo_string = " | ".join(f"S{i+1}: {m or ''}" for i, m in enumerate(mimo_per_sector))

            meta = {
                "timestamp_utc": now_iso_utc(),
                "supplier": supplier, "drawing_type": drawing_type, "client": client, "project": project,
                "site_type": site_type, "vendor": vendor, "cabinet_location": cabinet_location,
                "radio_location": radio_location, "sectors": sector_count, "mimo_config": mimo_string,
                "site_address": site_address, "original_filename": pdf_file.name, "pdf_title": pdf_title
            }

            write_excel(df_find, meta, excel_path)
            annotated = annotate_pdf_simple(pdf_bytes, df_find.to_dict("records")) or pdf_bytes
            pdf_out_path.write_bytes(annotated)

            st.success(f"Audit complete — Findings: {total} (Major {majors} / Minor {minors}), RFT {rft}%")
            st.download_button("⬇️ Download Excel Report", excel_path.read_bytes(), file_name=report_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button("⬇️ Download Annotated PDF", pdf_out_path.read_bytes(), file_name=pdf_out_name, mime="application/pdf")

            append_history({
                **{k: meta[k] for k in ["timestamp_utc","supplier","drawing_type","client","project","site_type","vendor","cabinet_location","radio_location","sectors","mimo_config","site_address","original_filename"]},
                "excel_report": str(excel_path), "annotated_pdf": str(pdf_out_path),
                "findings_total": total, "findings_major": majors, "findings_minor": minors, "rft_percent": rft,
                "exclude": bool(exclude_analytics),
            })

# ------------------------ ANALYTICS ------------------------
elif page == "Analytics":
    st.subheader("Audit History & Analytics")
    df = read_history()
    if df.empty:
        st.info("No audits yet.")
    else:
        view = df.copy().sort_values("timestamp_utc", ascending=False)
        st.dataframe(
            view[["timestamp_utc","supplier","client","project","drawing_type","site_type","vendor","radio_location","sectors","mimo_config","findings_total","findings_major","findings_minor","rft_percent","exclude","excel_report","annotated_pdf"]],
            use_container_width=True, hide_index=True
        )
        included = view[~view["exclude"]]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Audits (included)", len(included))
        c2.metric("Major findings", int(included["findings_major"].sum()) if len(included) else 0)
        c3.metric("Minor findings", int(included["findings_minor"].sum()) if len(included) else 0)
        c4.metric("Avg RFT %", round(included["rft_percent"].mean(), 1) if len(included) else 0.0)

        st.markdown("#### Findings per Supplier (included)")
        by_sup = included.groupby("supplier")[["findings_major","findings_minor"]].sum().sort_values("findings_major", ascending=False)
        if not by_sup.empty:
            st.bar_chart(by_sup, use_container_width=True)
        else:
            st.info("Nothing to chart yet.")

        with st.expander("Manage 'Exclude from analytics'"):
            full = read_history().sort_values("timestamp_utc", ascending=False).reset_index(drop=True)
            for idx, row in full.iterrows():
                cols = st.columns([3,5,2,2])
                cols[0].write(f"**{row['timestamp_utc']}**")
                cols[1].write(f"{row['supplier']} • {row['client']} • {row['project']}")
                new_val = cols[2].checkbox("Exclude", value=bool(row["exclude"]), key=f"exc_{idx}")
                if cols[3].button("Save", key=f"save_{idx}"):
                    all_df = read_history()
                    mask = (all_df["timestamp_utc"] == row["timestamp_utc"])
                    all_df.loc[mask, "exclude"] = bool(new_val)
                    all_df.to_csv(HISTORY_CSV, index=False)
                    st.success("Saved.")
                    st.experimental_rerun()

# ------------------------ SETTINGS ------------------------
else:
    st.subheader("Rules & App Settings")
    with st.expander("Logo override", expanded=True):
        st.caption("If the logo doesn’t show, enter the file name exactly as it appears in the repo root.")
        st.text_input("Logo filename (e.g. 88F3AB03-9D27-435B-AE39-7427F9A17FFC.png.png)", key="logo_file_override")
    st.divider()

    pwd = st.text_input("Enter rules update password", type="password")
    if pwd != RULES_EDIT_PASSWORD:
        st.info("Enter the update password to edit rules.")
    else:
        current = load_rules(RULES_FILE)
        text = st.text_area("rules_example.yaml", value=yaml.safe_dump(current, sort_keys=False, allow_unicode=True), height=380)
        c1,c2 = st.columns(2)
        if c1.button("Validate YAML"):
            try:
                yaml.safe_load(text)
                st.success("YAML is valid.")
            except Exception as e:
                st.error(f"YAML error: {e}")
        if c2.button("Save YAML"):
            try:
                data = yaml.safe_load(text) or {}
                save_rules(RULES_FILE, data)
                st.success("Saved rules_example.yaml")
            except Exception as e:
                st.error(f"Save failed: {e}")
