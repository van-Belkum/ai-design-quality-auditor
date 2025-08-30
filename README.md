# AI Design Quality Auditor (PDF & DWG-ready)

A fast, rule-based auditor for telecoms (and similar) design packs. Upload unlimited PDFs, run spelling and layout checks, and enforce per‑client checklists. Exports Excel reports with reject pages.

## Features
- **Batch audit** of any number of PDFs.
- **Spell check (UK/technical aware)** with allowlist & per‑client dictionaries.
- **Checklist engine** (YAML/Excel) enforcing `must include`, `must not include`, and regex rules.
- **Misalignment / skew** detection (page-level) using computer vision.
- **Blank/placeholder detection** (e.g., missing W3W).
- **Consistency checks** (e.g., model codes vary slightly across pages).
- **DWG support (convert-first)** — optionally auto-convert DWG→PDF using ODA File Converter if available.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit shows. Drag in PDFs (and DXF/DWG if you set the converter) and your rules pack.

## Rules pack
Rules live in YAML (see `rules_example.yaml`) or an Excel with columns:
`client, scope, when_page_contains, must_include, forbid, regex, hint`

## DWG / DXF
- Preferred: export **PDF** from CAD and audit normally.
- Optional: set environment variable `ODA_CONVERTER` to the ODA File Converter binary and the app will auto-convert `.dwg` to PDF before auditing.
- Or convert DWG→DXF and upload the DXF (the app will rasterise via CAD-viewer if present, else skip with a warning).

## Output
- `report.xlsx` — Summary, Failures, Typos, PerPage.
- `rejections.csv` — file,page,issue list.
- `json/` per-file machine-readable results.

## Notes
- “Unlimited” is bounded by your machine/VM resources. The app streams pages and releases memory to scale to very large jobs.
- OCR requires Tesseract if a page is image-only; otherwise the app uses native PDF text.
