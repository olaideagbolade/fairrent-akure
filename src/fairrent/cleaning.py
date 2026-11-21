"""Clean and anonymise the FairRent Akure survey export.

The raw Google Form export contains contact details (phone numbers and
emails). This module drops them, standardises areas and house types, parses
the money columns, and flags rows whose numbers do not add up.

Usage:
    from fairrent.cleaning import clean_survey
    tenants, landlords = clean_survey("survey_raw.csv")
"""
from __future__ import annotations

import re

import pandas as pd

# Column positions in the Google Form export (see the form builder script).
TENANT_COLS = {
    3: "age_group",
    4: "situation",
    5: "area",
    6: "house_type",
    7: "start_year",
    8: "rent",
    9: "agency_fee",
    10: "agreement_fee",
    11: "caution_fee",
    12: "inspection_total",
    13: "other_fee_text",
    14: "total_paid",
    15: "houses_inspected",
    16: "how_found",
    17: "bait_inspection",
    18: "found_rent_inflated",
    19: "receipt_given",
    20: "fairness_rating",
    21: "agent_experience",
    22: "would_use_assistant",
    23: "useful_features",
    44: "preferred_languages",
    45: "preferred_channels",
    46: "comments",
}
LANDLORD_COLS = {
    3: "age_group",
    24: "area",
    25: "house_types",
    26: "units",
    27: "rent_common_unit",
    28: "agent_use",
    29: "why_agents",
    30: "agent_asked_increase",
    31: "increase_reason",
    32: "fees_known",
    33: "would_list_directly",
    34: "listing_motivators",
    44: "preferred_languages",
    45: "preferred_channels",
    46: "comments",
}
# Columns that identify people. They are never written to the cleaned data.
PII_COLS = [0, 47, 48]  # timestamp, opt-in to testing, phone/email

MONEY_COLS = ["rent", "agency_fee", "agreement_fee", "caution_fee",
              "inspection_total", "total_paid"]

AREA_MAP = {
    "futa west gate": "FUTA West Gate",
    "it's a crime not to put westgate here": "FUTA West Gate",
    "westgate": "FUTA West Gate",
    "ologede quarters": "Ologede",
    "wesco": "Wesco",
}
HOUSE_TYPE_MAP = {
    "single room (face-me-i-face-you)": "Single room",
    "self-contain": "Self-contain",
    "mini flat (1 bedroom)": "Mini flat",
    "2-bedroom flat": "2-bedroom flat",
    "3-bedroom flat": "3-bedroom flat",
    "bungalow or duplex": "Bungalow or duplex",
    "pg hostel": "PG hostel",
}
# Words that mark a recurring utility charge rather than a move-in fee.
UTILITY_WORDS = ("light", "electric", "meter", "nepa", "water")
EMPTY_WORDS = {"", "0", "none", "nil", "no", "n/a", "na", "0000", "000"}


def to_naira(value) -> float | None:
    """Turn a form answer like '250000', '000' or ' 30,000 ' into a number."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().replace(",", "").replace("₦", "")
    if text == "":
        return None
    if re.fullmatch(r"0+", text):
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def standardise_area(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    key = value.strip().lower()
    return AREA_MAP.get(key, value.strip())


def standardise_house_type(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return HOUSE_TYPE_MAP.get(value.strip().lower(), value.strip())


def parse_other_fee(text) -> tuple[str | None, float | None, bool]:
    """Split '"Caretaker fee: 50000"' into ('Caretaker fee', 50000, False).

    Returns (name, amount, is_utility). Empty answers return (None, None, False).
    """
    if not isinstance(text, str) or text.strip().lower() in EMPTY_WORDS:
        return None, None, False
    numbers = re.findall(r"\d[\d,]*", text)
    amount = float(numbers[-1].replace(",", "")) if numbers else None
    name = re.sub(r"[-:]?\s*\d[\d,]*.*$", "", text).strip(" -:(") or text.strip()
    is_utility = any(w in text.lower() for w in UTILITY_WORDS)
    return name, amount, is_utility


def quality_flags(row: pd.Series) -> list[str]:
    """Rules that mark numbers which do not add up. Rows are kept, not edited."""
    flags = []
    rent = row["rent"]
    if rent is None or pd.isna(rent) or rent < 20000:
        flags.append("rent_missing_or_implausible")
        return flags
    for col in ["agency_fee", "agreement_fee", "caution_fee"]:
        v = row[col]
        if v is not None and not pd.isna(v) and v > rent:
            flags.append(f"{col}_greater_than_rent")
    insp = row["inspection_total"]
    if insp is not None and not pd.isna(insp) and insp >= rent:
        flags.append("inspection_greater_than_rent")
    total = row["total_paid"]
    if total is not None and not pd.isna(total):
        if total < rent:
            flags.append("total_less_than_rent")
        components = rent + sum(
            (row[c] or 0) for c in ["agency_fee", "agreement_fee", "caution_fee"]
            if row[c] is not None and not pd.isna(row[c])
        )
        if total > 2 * components:
            flags.append("total_much_higher_than_fees")
    return flags


def clean_survey(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    consent = raw.iloc[:, 1].str.startswith("Yes")
    # Respondents under 18 are excluded: the survey did not collect parental consent.
    adult = raw.iloc[:, 3].str.strip() != "Under 18"
    consent = consent & adult
    role = raw.iloc[:, 2]
    raw = raw[consent]
    role = role[consent]

    # ---- tenants ----
    t = raw[role.str.startswith("Tenant")].iloc[:, list(TENANT_COLS)].copy()
    t.columns = list(TENANT_COLS.values())
    for col in MONEY_COLS:
        t[col] = t[col].map(to_naira)
    t["houses_inspected"] = t["houses_inspected"].map(to_naira)
    t["fairness_rating"] = pd.to_numeric(t["fairness_rating"], errors="coerce")
    t["area"] = t["area"].map(standardise_area)
    t["house_type"] = t["house_type"].map(standardise_house_type)
    t["situation"] = t["situation"].str.strip()
    parsed = t["other_fee_text"].map(parse_other_fee)
    t["other_fee_name"] = parsed.map(lambda p: p[0])
    t["other_fee_amount"] = parsed.map(lambda p: p[1])
    t["other_fee_is_utility"] = parsed.map(lambda p: p[2])
    t = t.drop(columns=["other_fee_text"])
    t["quality_flags"] = t.apply(lambda r: ";".join(quality_flags(r)), axis=1)
    t.insert(0, "respondent_id", [f"T{i:03d}" for i in range(1, len(t) + 1)])

    # ---- landlords ----
    l = raw[role.str.startswith("Landlord")].iloc[:, list(LANDLORD_COLS)].copy()
    l.columns = list(LANDLORD_COLS.values())
    l["rent_common_unit"] = l["rent_common_unit"].map(to_naira)
    l["units"] = l["units"].map(to_naira)
    l["area"] = l["area"].map(standardise_area)
    l.insert(0, "respondent_id", [f"L{i:03d}" for i in range(1, len(l) + 1)])

    return t.reset_index(drop=True), l.reset_index(drop=True)
