# AI Design QA — Rules Engine v1

**What’s new**
- Single-audit flow
- Sidebar prompts for: Client, Project, Site Type, Vendor, Cabinet Location, Radio Location, Quantity of Sectors, Proposed MIMO Config, Site Address
- Metadata-aware rules engine with simple 'when' filters
- Rule types: include_text, forbid_together, regex_must, regex_forbid, numeric_range
- Learning: upload a CSV to mark messages "Not Valid" → suppression
- Excel export naming: `<original> - PASS|REJECTED - YYYYMMDD.xlsx`

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
