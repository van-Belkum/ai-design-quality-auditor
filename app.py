# app.py
import os, io, re, json, base64
from pathlib import Path
from datetime import datetime, timezone
import streamlit as st
import pandas as pd
import yaml

# Optional PDF annotation / text search
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
RECORDS_DIR = HISTORY_DIR / "records"
HISTORY_CSV = HISTORY_DIR / "audit_history.csv"
RULES_FILE = Path("rules_example.yaml")

# ------------------------ Utilities ------------------------
def ensure_dirs():
    for d in (HISTORY_DIR, EXPORT_DIR, RECORDS_DIR):
        d.mkdir(parents=True, exist_ok=True)

def now_iso_utc():
    return datetime.now(timezone.utc).isoformat()

def load_rules(path: Path) -> dict:
    if not path.exists():
        # seed with expected keys
        data = {
            "normalizations": {},        # e.g. { "AHEGC": "AHEGG" }
            "ignore_patterns": [],       # e.g. ["Optional check X", "Minor whitespace .*"]
        }
        save_rules(path, data)
        return data
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("normalizations", {})
    data.setdefault("ignore_patterns", [])
    return data

def save_rules(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def get_logo_b64() -> str | None:
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
        st.warning("⚠️ Logo file not found. Set **Settings → Logo override** or add an image to the repo root.")
        return
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

def normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def address_matches_title(site_address: str, pdf_title: str) -> bool:
    if ", 0 ," in site_address.replace(",0,", ", 0 ,"):
        return True
    a = normalize_text(site_address)
    t = normalize_text(pdf_title)
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

def extract_pdf_text_first_pages(pdf_bytes: bytes, pages: int = 2) -> str:
    if not fitz:
        return ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = min(pages, doc.page_count)
        chunks = []
        for i in range(pages):
            chunks.append(doc.load_page(i).get_text("text") or "")
        doc.close()
        return "\n".join(chunks)
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
            "run_id","timestamp_utc","supplier","drawing_type","client","project","site_type","vendor",
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

def save_run_records(run_id: str, metadata: dict, findings: list[dict]):
    RECORDS_DIR.mkdir(exist_ok=True, parents=True)
    with (RECORDS_DIR / f"{run_id}.json").open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "findings": findings}, f, ensure_ascii=False, indent=2)

def load_run_records(run_id: str) -> dict | None:
    p = RECORDS_DIR / f"{run_id}.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

# ------------------------ Static Choices ------------------------
SUPPLIERS = ["— Select —","CEG","CTIL","Emfyser","Innov8","Invict","KTL Team (Internal)","Trylon"]
DRAWING_TYPES = ["— Select —","General Arrangement","Detailed Design"]
CLIENTS = ["— Select —","BTEE","Vodafone","MBNL","H3G","Cornerstone","Cellnex"]
PROJECTS = ["— Select —","RAN","Power Resilience","East Unwind","Beacon 4"]
SITE_TYPES = ["— Select —","Greenfield","Rooftop","Streetworks"]
VENDORS = ["— Select —","Ericsson","Nokia"]
CABINET_LOCS = ["— Select —","Indoor","Outdoor"]
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
page = st.sidebar.radio("Go to:", ["Audit","Review & Train","Analytics","Settings"], index=0)

st.title(APP_TITLE)
st.caption("Audit → Review/Train → Improve rules. YAML updates are password-protected.")

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

        same_all = st.checkbox("Use same config for all sectors", value=True, disabled=False)
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
            if k.startswith("mimo_s") or k in ("mimo_all",):
                st.session_state.pop(k, None)
        st.experimental_rerun()

    if run_btn:
        errs = []
        req = {
            "Supplier": supplier, "Drawing Type": drawing_type, "Client": client, "Project": project,
            "Site Type": site_type, "Proposed Vendor": vendor, "Proposed Cabinet Location": cabinet_location,
            "Proposed Radio Location": radio_location, "Quantity of Sectors": sectors_choice, "Site Address": site_address.strip()
        }
        for label, val in req.items():
            if (isinstance(val, str) and (val == "" or val.startswith("— "))) or val is None:
                errs.append(f"{label} is required.")
        if not is_power_res:
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
            rules = load_rules(RULES_FILE)

            findings = []

            # Address vs title
            if not address_matches_title(site_address, pdf_title):
                findings.append({"severity":"Major","category":"Metadata","message":"Site Address does not appear to match PDF title.","page":1, "valid": None})

            # Normalization (learning) checks: scan first 2 pages for bad tokens
            pdf_text = extract_pdf_text_first_pages(pdf_bytes, pages=2)
            for bad, good in (rules.get("normalizations") or {}).items():
                if bad and (bad in pdf_text) and (good not in pdf_text):
                    findings.append({
                        "severity": "Minor",
                        "category": "Spelling/Tag",
                        "message": f"Found '{bad}'. Consider normalizing to '{good}'.",
                        "page": 1,
                        "valid": None
                    })

            # Suppress by ignore_patterns
            ignore_res = [re.compile(pat) for pat in (rules.get("ignore_patterns") or []) if pat]
            def suppressed(msg: str) -> bool:
                return any(rx.search(msg or "") for rx in ignore_res)
            findings = [f for f in findings if not suppressed(f["message"])]

            majors = sum(1 for f in findings if f["severity"].lower()=="major")
            minors = sum(1 for f in findings if f["severity"].lower()=="minor")
            total  = len(findings)
            rft = 0 if total>0 else 100
            df_find = pd.DataFrame(findings or [{"severity":"Info","category":"General","message":"No issues found.","page":1, "valid": True}])

            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_id = ts  # simple run id
            base = Path(pdf_file.name).stem
            report_name = f"{base}_{'Rejected' if total>0 else 'Pass'}_{ts}.xlsx"
            pdf_out_name = f"{base}_annotated_{ts}.pdf"
            excel_path = EXPORT_DIR / report_name
            pdf_out_path = EXPORT_DIR / pdf_out_name

            # MIMO string
            if (int(sector_count or 0)) <= 1:
                mimo_string = (mimo_per_sector[0] if mimo_per_sector else "") or ""
            else:
                mimo_string = " | ".join(f"S{i+1}: {m or ''}" for i, m in enumerate(mimo_per_sector))

            meta = {
                "run_id": run_id,
                "timestamp_utc": now_iso_utc(),
                "supplier": supplier, "drawing_type": drawing_type, "client": client, "project": project,
                "site_type": site_type, "vendor": vendor, "cabinet_location": cabinet_location,
                "radio_location": radio_location, "sectors": int(sector_count or 0), "mimo_config": mimo_string,
                "site_address": site_address, "original_filename": pdf_file.name, "pdf_title": pdf_title
            }

            write_excel(df_find.drop(columns=["valid"], errors="ignore"), meta, excel_path)
            annotated = annotate_pdf_simple(pdf_bytes, df_find.to_dict("records")) or pdf_bytes
            pdf_out_path.write_bytes(annotated)

            st.success(f"Audit complete — Findings: {total} (Major {majors} / Minor {minors}), RFT {rft}%")
            st.download_button("⬇️ Download Excel Report", excel_path.read_bytes(), file_name=report_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button("⬇️ Download Annotated PDF", pdf_out_path.read_bytes(), file_name=pdf_out_name, mime="application/pdf")

            append_history({
                **{k: meta[k] for k in ["run_id","timestamp_utc","supplier","drawing_type","client","project","site_type","vendor","cabinet_location","radio_location","sectors","mimo_config","site_address","original_filename"]},
                "excel_report": str(excel_path), "annotated_pdf": str(pdf_out_path),
                "findings_total": total, "findings_major": majors, "findings_minor": minors, "rft_percent": rft,
                "exclude": bool(exclude_analytics),
            })
            save_run_records(run_id, meta, findings)

# ------------------------ REVIEW & TRAIN ------------------------
elif page == "Review & Train":
    st.subheader("Validate findings and teach the tool")
    hist = read_history().sort_values("timestamp_utc", ascending=False)
    if hist.empty:
        st.info("No audit runs yet.")
    else:
        left, right = st.columns([2,3])
        run_opt = left.selectbox(
            "Select a run",
            options=[(r.run_id, f"{r.timestamp_utc} • {r.supplier} • {r.client} • {r.project}") for _, r in hist.iterrows()],
            format_func=lambda t: t[1]
        )
        if run_opt:
            run_id = run_opt[0]
            rec = load_run_records(run_id)
            if not rec:
                st.warning("No record file found for this run.")
            else:
                meta = rec["metadata"]
                findings = rec["findings"]
                st.write(f"**File:** {meta.get('original_filename','')}  |  **Findings:** {len(findings)}")

                if not findings:
                    st.success("No findings for this run.")
                else:
                    st.markdown("##### Label each finding")
                    rows = []
                    for i, f in enumerate(findings):
                        cols = st.columns([4,1,2,2])
                        cols[0].write(f"**[{f.get('severity','')}] {f.get('category','')}** — {f.get('message','')}")
                        valid = cols[1].selectbox("Valid?", ["Unreviewed","Valid","Not valid"], index=0, key=f"lab_{run_id}_{i}")
                        corr = cols[2].text_input("Normalization (bad→good)", key=f"corr_{run_id}_{i}", placeholder="e.g. AHEGC→AHEGG")
                        ignr = cols[3].text_input("Ignore pattern (regex)", key=f"ign_{run_id}_{i}", placeholder=r"e.g. Optional.*")
                        rows.append({"valid": valid, "corr": corr, "ign": ignr, "orig": f})

                    apply = st.button("Apply feedback to rules & save labels", type="primary")
                    if apply:
                        # Update record with labels
                        for i, r in enumerate(rows):
                            findings[i]["valid"] = None if r["valid"]=="Unreviewed" else (r["valid"]=="Valid")
                        save_run_records(run_id, meta, findings)

                        # Update rules
                        rules = load_rules(RULES_FILE)
                        changed = False

                        # Normalizations
                        for r in rows:
                            if "→" in r["corr"]:
                                bad, good = [x.strip() for x in r["corr"].split("→",1)]
                                if bad and good:
                                    rules["normalizations"][bad] = good
                                    changed = True

                        # Ignore patterns
                        for r in rows:
                            pat = r["ign"].strip()
                            if pat:
                                if pat not in rules["ignore_patterns"]:
                                    rules["ignore_patterns"].append(pat)
                                    changed = True

                        if changed:
                            save_rules(RULES_FILE, rules)
                            st.success("Rules updated.")
                        else:
                            st.info("No rule changes to save.")

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
