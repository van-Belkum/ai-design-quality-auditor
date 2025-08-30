# AI Design Quality Auditor (pre‑MIMO stable build)

This is a minimal, known-good build that restores the rule learning and PDF audit from earlier working versions.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud
- Keep **only**: `app.py`, `requirements.txt`, `rules_example.yaml` and `history/` (auto-created).
- In App settings, "Main module file" = `app.py`.
- No `packages.txt` is required.
- Ensure `requirements.txt` contains *LF* line endings (copy the file from this repo to avoid Windows backslashes).

## What’s included
- Spellcheck with allowlist
- Required text checks (per client)
- Simple mutual exclusion (e.g., **Brush** vs **Generator Power**)
- Feedback CSV loop (`quick_rules_template.csv`) to mark **Valid** / **Not Valid** so the app learns
- Excel report + pass/fail naming

