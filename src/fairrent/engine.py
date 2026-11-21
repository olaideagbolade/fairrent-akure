"""FairRent fee fairness engine.

Every number FairRent shows comes from here, never from the language model.
The engine compares a tenant's fees with what other Akure tenants reported in
the survey and gives each fee one of four verdicts:

    normal          within the usual range
    above_typical   higher than most tenants paid; worth negotiating
    red_flag        far above the usual range, or a fee few tenants report
    not_enough_data the survey has too few similar reports to judge

Thresholds (fees as a share of yearly rent) were set from the survey's
quartiles and rounded to numbers tenants can remember. See DATA_CARD.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

MIN_REPORTS = 5  # fewer similar reports than this -> widen the comparison group

# (normal_max, above_typical_max) as a fraction of yearly rent.
# Survey quartiles among tenants who paid each fee:
#   agency    median 12.8%, upper quartile 20.6%
#   agreement median  6.3%, upper quartile 11.8%
#   caution   median 10.8%, upper quartile 13.7%
FEE_RULES = {
    "agency": (0.10, 0.20),
    "agreement": (0.10, 0.20),
    "caution": (0.15, 0.25),
}
# Inspection fee per house, in naira. Survey median about 2,100; 11 of 12
# reports were 5,000 or less.
INSPECTION_RULE = (5000, 10000)
RENT_ABOVE_FACTOR = 1.25  # up to 25% above the upper quartile = above_typical

FEE_LABELS = {
    "rent": "Rent",
    "agency": "Agency fee",
    "agreement": "Agreement / legal fee",
    "caution": "Caution fee",
    "inspection": "Inspection fee",
}
UTILITY_WORDS = ("light", "electric", "meter", "nepa", "water")


@dataclass
class RentReference:
    group: str
    n: int
    p25: float
    median: float
    p75: float


@dataclass
class FeeEngine:
    tenants: pd.DataFrame
    references: dict = field(default_factory=dict)

    @classmethod
    def from_csv(cls, path: str) -> "FeeEngine":
        return cls(pd.read_csv(path, keep_default_na=False, na_values=[""]))

    # ---------- reference data ----------
    def _valid_rents(self) -> pd.DataFrame:
        t = self.tenants
        flags = t["quality_flags"].fillna("").astype(str)
        return t[~flags.str.contains("rent_missing")].dropna(subset=["rent"])

    def rent_reference(self, house_type: str | None, area: str | None) -> RentReference | None:
        """Typical rent for the closest group with enough reports."""
        rents = self._valid_rents()
        candidates = []
        if house_type and area:
            candidates.append((f"{house_type} in {area}",
                               rents[(rents.house_type == house_type) & (rents.area == area)]))
        if house_type:
            candidates.append((f"{house_type} across Akure", rents[rents.house_type == house_type]))
        for name, group in candidates:
            if len(group) >= MIN_REPORTS:
                q = group["rent"].quantile([0.25, 0.5, 0.75])
                return RentReference(name, len(group), float(q[0.25]), float(q[0.5]), float(q[0.75]))
        return None

    # ---------- verdicts ----------
    @staticmethod
    def _ratio_verdict(ratio: float, rule: tuple[float, float]) -> str:
        if ratio <= rule[0]:
            return "normal"
        if ratio <= rule[1]:
            return "above_typical"
        return "red_flag"

    def check(self, extraction: dict) -> dict:
        """Run the fee check on the model's structured extraction.

        `extraction` looks like:
            {"intent": "fee_check", "area": "FUTA South Gate",
             "house_type": "Self-contain",
             "fees": {"rent": 280000, "agency": 56000, "agreement": None,
                      "caution": 50000, "inspection": 5000,
                      "other": [{"name": "Development fee", "amount": 40000}]},
             "language": "pcm"}
        """
        intent = extraction.get("intent", "fee_check")
        area = extraction.get("area")
        house_type = extraction.get("house_type")
        language = extraction.get("language", "en")
        base = {"intent": intent, "area": area, "house_type": house_type, "language": language}

        if intent == "other":
            return {**base, "status": "out_of_scope"}

        ref = self.rent_reference(house_type, area)
        ref_dict = None if ref is None else {
            "group": ref.group, "reports": ref.n,
            "p25": round(ref.p25), "median": round(ref.median), "p75": round(ref.p75)}

        if intent == "rent_info":
            if not house_type:
                return {**base, "status": "needs_info", "missing": ["house_type"]}
            return {**base, "status": "ok" if ref else "not_enough_data", "reference": ref_dict}

        fees = extraction.get("fees") or {}
        rent = fees.get("rent")
        if not rent:
            return {**base, "status": "needs_info", "missing": ["rent"], "reference": ref_dict}

        items = []
        # Rent
        if ref is None:
            items.append(self._item("rent", rent, None, "not_enough_data",
                                    "Too few similar reports to compare the rent."))
        elif rent <= ref.p75:
            items.append(self._item("rent", rent, None, "normal",
                                    f"Within the usual range for {ref.group}."))
        elif rent <= ref.p75 * RENT_ABOVE_FACTOR:
            items.append(self._item("rent", rent, None, "above_typical",
                                    f"Higher than most reports for {ref.group}."))
        else:
            items.append(self._item("rent", rent, None, "red_flag",
                                    f"Far above most reports for {ref.group}. Confirm the rent with the landlord."))
        # Percentage fees
        for key, rule in FEE_RULES.items():
            amount = fees.get(key)
            if amount is None:
                continue
            ratio = amount / rent
            verdict = self._ratio_verdict(ratio, rule)
            reason = {
                "normal": f"{ratio:.0%} of rent, within the usual {rule[0]:.0%} or less.",
                "above_typical": f"{ratio:.0%} of rent; most tenants pay {rule[0]:.0%} or less.",
                "red_flag": f"{ratio:.0%} of rent, far above the usual {rule[0]:.0%}.",
            }[verdict]
            if key == "caution":
                reason += " It should be refundable, so get that in writing."
            item = self._item(key, amount, round(ratio, 3), verdict, reason)
            item["usual_max_share"] = rule[0]
            items.append(item)
        # Inspection (per house)
        insp = fees.get("inspection")
        if insp is not None:
            v = "normal" if insp <= INSPECTION_RULE[0] else (
                "above_typical" if insp <= INSPECTION_RULE[1] else "red_flag")
            reason = {
                "normal": "Within the usual inspection fee per house.",
                "above_typical": f"Most tenants paid {INSPECTION_RULE[0]:,} or less per house.",
                "red_flag": "Far above the usual inspection fee. Ask to see photos or a video first.",
            }[v]
            item = self._item("inspection", insp, None, v, reason)
            item["usual_max_amount"] = INSPECTION_RULE[0]
            items.append(item)
        # Other named fees
        for other in fees.get("other") or []:
            name, amount = other.get("name") or "Other fee", other.get("amount")
            if any(w in name.lower() for w in UTILITY_WORDS):
                items.append(self._item(name, amount, None, "normal",
                                        "A utility charge, not a move-in fee. Confirm how often it is paid."))
            else:
                items.append(self._item(name, amount, None, "red_flag",
                                        "Few tenants report this fee. Ask what it covers and who receives it."))

        total = sum(i["amount"] for i in items if i["amount"])
        return {**base, "status": "ok", "reference": ref_dict, "items": items,
                "total": total,
                "flagged": [i["fee"] for i in items if i["verdict"] in ("above_typical", "red_flag")]}

    @staticmethod
    def _item(key, amount, ratio, verdict, reason):
        return {"fee": FEE_LABELS.get(key, key), "amount": amount, "share_of_rent": ratio,
                "verdict": verdict, "reason": reason}


def summary_stats(tenants: pd.DataFrame) -> dict:
    """Headline survey findings, used in the README and data card."""
    t = tenants
    n = len(t)

    def share(col, prefix):
        return round(float(t[col].fillna("").astype(str).str.startswith(prefix).mean()), 3)

    return {
        "tenants": n,
        "found_rent_inflated_yes": share("found_rent_inflated", "Yes"),
        "bait_inspection_yes": share("bait_inspection", "Yes"),
        "no_receipt": share("receipt_given", "No"),
        "would_use_yes_or_maybe": round(float(t["would_use_assistant"].isin(["Yes", "Maybe"]).mean()), 3),
        "mean_fairness_rating": round(float(t["fairness_rating"].mean()), 2),
    }


if __name__ == "__main__":  # quick manual check
    import sys
    engine = FeeEngine.from_csv(sys.argv[1])
    example = {"intent": "fee_check", "area": "FUTA South Gate", "house_type": "Self-contain",
               "fees": {"rent": 280000, "agency": 56000, "agreement": 56000, "caution": 50000,
                        "inspection": 5000, "other": [{"name": "Development fee", "amount": 40000}]},
               "language": "pcm"}
    print(json.dumps(engine.check(example), indent=2))
