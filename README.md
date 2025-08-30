# AI Design Quality Auditor (v7)

This is a slim, **working** baseline of the Streamlit app we iterated.
It includes:

- Single-audit flow with left-hand metadata (Client, Project, Site Type, Vendor, MIMO S1/S2/S3 etc, with "Same across" toggle)
- Rule engine with:
  - **Required phrases**
  - **Forbidden pairs** (e.g., *Brush* vs *Generator Power*)
  - **Learning loop**: mark findings as **Not valid** → app stores into `suppress_patterns` in rules and persists
- **History log** in `history/history.csv` (user, timestamp, outcome, counters)
- **Excel export** that names the file as `ORIGINALFILENAME__Pass|Rejected__YYYYMMDD.xlsx`

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a PDF and the current `rules_example.yaml` (or let the default be used).
