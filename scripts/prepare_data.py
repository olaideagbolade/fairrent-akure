"""Step 1: clean and anonymise the raw survey export.

    python scripts/prepare_data.py path/to/survey_raw.csv

Writes:
    data/tenants_clean.csv    anonymised tenant responses
    data/landlords_clean.csv  anonymised landlord responses
    data/rent_reference.json  typical rents by house type and area
    data/survey_summary.json  headline findings

The raw export contains phone numbers and emails. Keep it out of the repo
(it is listed in .gitignore).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fairrent.cleaning import clean_survey  # noqa: E402
from fairrent.engine import FeeEngine, summary_stats  # noqa: E402


def main(raw_path: str) -> None:
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    tenants, landlords = clean_survey(raw_path)
    tenants.to_csv(data / "tenants_clean.csv", index=False)
    landlords.to_csv(data / "landlords_clean.csv", index=False)

    engine = FeeEngine(tenants)
    reference = {}
    for house_type in sorted(tenants["house_type"].dropna().unique()):
        for area in [None] + sorted(tenants["area"].dropna().unique()):
            ref = engine.rent_reference(house_type, area)
            if ref and ref.group not in reference:
                reference[ref.group] = ref.__dict__
    (data / "rent_reference.json").write_text(json.dumps(reference, indent=2))

    summary = summary_stats(tenants)
    summary["landlords"] = len(landlords)
    summary["rows_with_quality_flags"] = int((tenants["quality_flags"].fillna("") != "").sum())
    (data / "survey_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Tenants: {len(tenants)}  Landlords: {len(landlords)}")
    print(json.dumps(summary, indent=2))
    print(f"Rent reference groups: {list(reference)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "raw" / "survey_raw.csv"))
