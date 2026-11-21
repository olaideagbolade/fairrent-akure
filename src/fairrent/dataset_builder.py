"""Build the FairRent conversation dataset from the cleaned survey.

Each conversation has:
  - a user message (English, Nigerian Pidgin, or Yoruba mixed with English)
  - the gold extraction (structured JSON the model should produce)
  - the gold engine result and gold reply

Sources:
  - "survey":     fees exactly as reported by a real respondent
  - "variant":    a real respondent's record with fees changed to cover
                  inflated packages and unusual fees (labelled synthetic)
  - "rent_info", "missing_info", "out_of_scope": short scripted questions

Leakage control: respondents are split into train and test, and the test set
uses its own message templates, so test phrasings are never seen in training.
"""
from __future__ import annotations

import json
import random

import pandas as pd

from .composer import compose_reply
from .engine import FeeEngine

# How users refer to each area / house type (first entry = canonical name).
AREA_SAYINGS = {
    "FUTA South Gate": ["FUTA South Gate", "South Gate", "southgate", "FUTA southgate"],
    "FUTA North Gate": ["FUTA North Gate", "North Gate", "northgate"],
    "FUTA West Gate": ["FUTA West Gate", "West Gate", "westgate"],
    "Apatapiti": ["Apatapiti", "Apata"],
    "Aule": ["Aule", "Aule road"],
    "Oba-Ile": ["Oba-Ile", "Oba Ile"],
    "Ologede": ["Ologede"],
    "Wesco": ["Wesco"],
}
TYPE_SAYINGS = {
    "Self-contain": ["self-contain", "self contain", "selfcon", "self-con"],
    "Single room": ["single room", "room", "face-me-I-face-you"],
    "Mini flat": ["mini flat", "one bedroom flat"],
    "2-bedroom flat": ["2-bedroom flat", "two bedroom"],
    "3-bedroom flat": ["3-bedroom flat", "three bedroom flat"],
    "PG hostel": ["PG hostel", "hostel"],
    "Bungalow or duplex": ["bungalow", "duplex"],
}
UNUSUAL_FEES = ["Development fee", "Caretaker fee", "Service charge", "Agent transport",
                "Documentation fee", "Cleaning fee"]

# ---------- message templates ----------
# {area} {htype} {fees} are filled in; {fees} is built per language.
TRAIN_TEMPLATES = {
    "en": [
        "I want to rent a {htype} at {area}. {fees} Is this fair?",
        "Please check this for me. {htype}, {area}. {fees}",
        "Agent quoted me for a {htype} in {area}: {fees} Am I being cheated?",
        "{fees} This is for a {htype} around {area}. What do you think?",
    ],
    "pcm": [
        "Abeg check am for me. {htype} for {area}. {fees} E make sense?",
        "Agent wan collect money for {htype} for {area}. {fees} Dem dey cheat me?",
        "{htype} for {area}: {fees} Abeg e fair?",
        "I see {htype} for {area}. {fees} Wetin you think?",
    ],
    "yo": [
        "Ẹ n lẹ́. Mo fẹ́ gba {htype} ní {area}. {fees} Ṣé ó dára?",
        "Ẹ jọ̀ọ́, ẹ bá mi wò ó. {htype} ní {area}. {fees}",
    ],
}
TEST_TEMPLATES = {
    "en": [
        "Good evening. Found a {htype} near {area}. {fees} Does this look okay?",
        "Quick question: {htype} at {area}. {fees} Too much?",
    ],
    "pcm": [
        "Boss, {htype} for {area} o. {fees} Dem no dey overcharge me so?",
        "My guy, check this {htype} for {area}. {fees}",
    ],
    "yo": [
        "Ẹ káàsán. {htype} ní {area}. {fees} Ṣé owó yìí pọ̀ jù?",
    ],
}
FEE_WORDS = {
    "en": {"rent": ["rent", "yearly rent"], "agency": ["agency fee", "agent fee"],
           "agreement": ["agreement fee", "legal fee"], "caution": ["caution fee", "caution deposit"],
           "inspection": ["inspection fee", "inspection"]},
    "pcm": {"rent": ["rent", "house money"], "agency": ["agency", "agent money"],
            "agreement": ["agreement", "lawyer money"], "caution": ["caution", "caution money"],
            "inspection": ["inspection", "inspection money"]},
    "yo": {"rent": ["rent", "owó ilé"], "agency": ["agency fee", "owó agent"],
           "agreement": ["agreement fee", "agreement"], "caution": ["caution fee", "caution"],
           "inspection": ["inspection fee", "inspection"]},
}
JOINERS = {"en": [" is ", ": "], "pcm": [" na ", " "], "yo": [" jẹ́ ", ": "]}

RENT_INFO = {
    "train": {
        "en": ["How much is a {htype} in {area}?", "What is the normal rent for a {htype} at {area}?"],
        "pcm": ["How much dem dey rent {htype} for {area}?", "Wetin be normal price for {htype} for {area}?"],
        "yo": ["Èló ni {htype} ní {area}?"],
    },
    "test": {
        "en": ["What should I expect to pay for a {htype} around {area}?"],
        "pcm": ["Abeg how much {htype} dey cost for {area} now?"],
        "yo": ["Ẹ jọ̀ọ́, èló ni wọ́n ń gba fún {htype} ní {area}?"],
    },
}
MISSING_INFO = {
    "train": {
        "en": ["The agent wants {agency} agency fee and {caution} caution for a {htype}. Is that okay?"],
        "pcm": ["Agent talk say agency na {agency} and caution na {caution}. E normal?"],
    },
    "test": {
        "en": ["They are asking for {agency} as agent fee. Is that normal?"],
        "pcm": ["Dem say make I pay {agency} for agent. E correct?"],
    },
}
OUT_OF_SCOPE = {
    "train": [("en", "Who won the football match yesterday?"),
              ("pcm", "Abeg wetin I go cook tonight?"),
              ("en", "Can you help me write my CV?"),
              ("pcm", "Which phone better pass for 100k?"),
              ("en", "What is the capital of Ondo State?"),
              ("yo", "Ṣé òjò máa rọ̀ lónìí?")],
    "test": [("en", "Tell me a joke about lecturers."),
             ("pcm", "How I fit make money online?"),
             ("en", "What time does FUTA library close?"),
             ("yo", "Kí ni orúkọ ọba Àkúrẹ́?"),
             ("pcm", "Abeg help me do my assignment."),
             ("en", "Is it going to rain in Akure tomorrow?")],
}


def money_text(rng: random.Random, amount: int) -> str:
    """Write an amount the way people type it: 250k, 250,000, ₦250,000, 250000."""
    style = rng.choice(["k", "comma", "naira", "plain"])
    if style == "k" and amount % 1000 == 0:
        return f"{amount // 1000}k"
    if style == "comma":
        return f"{amount:,}"
    if style == "naira":
        return f"₦{amount:,}"
    return str(amount)


def fees_text(rng, lang, fees: dict) -> str:
    words, joiners = FEE_WORDS[lang], JOINERS[lang]
    parts = []
    for key in ["rent", "agency", "agreement", "caution", "inspection"]:
        if fees.get(key) is not None:
            parts.append(f"{rng.choice(words[key])}{rng.choice(joiners)}{money_text(rng, fees[key])}")
    for other in fees.get("other", []):
        parts.append(f"{other['name'].lower()}{rng.choice(joiners)}{money_text(rng, other['amount'])}")
    sep = ", " if rng.random() < 0.7 else ". "
    text = sep.join(parts)
    return text[0].upper() + text[1:] + "."


def record_fees(row: pd.Series) -> dict:
    """Fees from a survey row. Zero fees are treated as 'not charged' (left out)."""
    fees = {"rent": int(row["rent"])}
    for col, key in [("agency_fee", "agency"), ("agreement_fee", "agreement"), ("caution_fee", "caution")]:
        v = row[col]
        if pd.notna(v) and v > 0:
            fees[key] = int(v)
    insp, houses = row["inspection_total"], row["houses_inspected"]
    if pd.notna(insp) and insp > 0 and pd.notna(houses) and houses > 0:
        per_house = int(round(insp / houses / 500) * 500) or 500
        fees["inspection"] = per_house
    if isinstance(row.get("other_fee_name"), str) and pd.notna(row.get("other_fee_amount")):
        fees["other"] = [{"name": row["other_fee_name"].strip().capitalize(),
                          "amount": int(row["other_fee_amount"])}]
    return fees


def variant_fees(rng, fees: dict) -> dict:
    """Make an inflated or unusual package from a real record."""
    rent = fees["rent"]
    out = {"rent": rent}
    kind = rng.choice(["inflated_agency", "unusual_fee", "both", "fair"])
    out["agency"] = int(round(rent * (rng.choice([0.2, 0.25, 0.3]) if kind in ("inflated_agency", "both")
                                       else rng.choice([0.05, 0.1])) / 1000) * 1000)
    if rng.random() < 0.6:
        out["agreement"] = int(round(rent * rng.choice([0.05, 0.1, 0.15]) / 1000) * 1000)
    if rng.random() < 0.5:
        out["caution"] = int(round(rent * rng.choice([0.1, 0.2]) / 1000) * 1000)
    if rng.random() < 0.5:
        out["inspection"] = rng.choice([1000, 2000, 3000, 5000, 10000])
    if kind in ("unusual_fee", "both"):
        out["other"] = [{"name": rng.choice(UNUSUAL_FEES),
                         "amount": rng.choice([10000, 20000, 30000, 40000, 50000])}]
    return out


def gold_extraction(intent, area, htype, fees, lang) -> dict:
    full = {"rent": None, "agency": None, "agreement": None, "caution": None,
            "inspection": None, "other": []}
    full.update(fees or {})
    return {"intent": intent, "area": area, "house_type": htype, "fees": full, "language": lang}


def say(rng, mapping, canonical):
    return rng.choice(mapping.get(canonical, [canonical]))


def make_conversation(cid, split, source, message, extraction, engine) -> dict:
    result = engine.check(extraction)
    return {"id": cid, "split": split, "source": source, "message": message,
            "extraction": extraction, "result": result, "reply": compose_reply(result)}


def build(tenants: pd.DataFrame, seed: int = 42, test_share: float = 0.3) -> list[dict]:
    rng = random.Random(seed)
    engine = FeeEngine(tenants)
    usable = tenants[(tenants["quality_flags"].fillna("") == "")
                     & tenants["area"].isin(AREA_SAYINGS)
                     & tenants["house_type"].isin(TYPE_SAYINGS)].copy()
    ids = sorted(usable["respondent_id"])
    rng.shuffle(ids)
    test_ids = set(ids[: int(len(ids) * test_share)])

    convs = []
    langs = ["en"] * 5 + ["pcm"] * 4 + ["yo"] * 2

    for split in ["train", "test"]:
        templates = TRAIN_TEMPLATES if split == "train" else TEST_TEMPLATES
        rows = usable[usable["respondent_id"].isin(test_ids) == (split == "test")]
        n_variants = 3 if split == "train" else 2
        for _, row in rows.iterrows():
            base = record_fees(row)
            packages = [("survey", base)] + [("variant", variant_fees(rng, base)) for _ in range(n_variants)]
            for source, fees in packages:
                lang = rng.choice(langs)
                msg = rng.choice(templates[lang]).format(
                    htype=say(rng, TYPE_SAYINGS, row["house_type"]),
                    area=say(rng, AREA_SAYINGS, row["area"]),
                    fees=fees_text(rng, lang, fees))
                ext = gold_extraction("fee_check", row["area"], row["house_type"], fees, lang)
                convs.append(make_conversation(f"{split}-{len(convs):03d}", split,
                                               f"{source}:{row['respondent_id']}", msg, ext, engine))

        # Rent questions
        pairs = [("Self-contain", "FUTA South Gate"), ("Self-contain", "FUTA North Gate"),
                 ("Single room", "FUTA South Gate"), ("Self-contain", "Apatapiti"),
                 ("Mini flat", "Aule"), ("Self-contain", "FUTA West Gate"),
                 ("Single room", "Oba-Ile"), ("2-bedroom flat", "FUTA South Gate")]
        for htype, area in (pairs if split == "train" else pairs[:5]):
            lang = rng.choice(["en", "pcm", "yo"])
            msg = rng.choice(RENT_INFO[split][lang]).format(
                htype=say(rng, TYPE_SAYINGS, htype), area=say(rng, AREA_SAYINGS, area))
            ext = gold_extraction("rent_info", area, htype, None, lang)
            convs.append(make_conversation(f"{split}-{len(convs):03d}", split, "rent_info", msg, ext, engine))

        # Missing rent
        for _ in range(6 if split == "train" else 4):
            lang = rng.choice(["en", "pcm"])
            agency, caution = rng.choice([20000, 30000, 50000]), rng.choice([10000, 20000])
            htype = rng.choice(["Self-contain", "Single room"])
            template = rng.choice(MISSING_INFO[split][lang])
            msg = template.format(agency=money_text(rng, agency), caution=money_text(rng, caution),
                                  htype=say(rng, TYPE_SAYINGS, htype))
            fees = {"agency": agency}
            if "{caution}" in template:
                fees["caution"] = caution
            ext = gold_extraction("fee_check", None, htype if "{htype}" in template else None,
                                  fees, lang)
            convs.append(make_conversation(f"{split}-{len(convs):03d}", split, "missing_info", msg, ext, engine))

        # Out of scope
        for lang, msg in OUT_OF_SCOPE[split]:
            ext = gold_extraction("other", None, None, None, lang)
            ext["fees"] = None
            convs.append(make_conversation(f"{split}-{len(convs):03d}", split, "out_of_scope", msg, ext, engine))
    return convs


def to_sft_examples(convs: list[dict]) -> list[dict]:
    """Two supervised examples per conversation: extraction and reply."""
    from .prompts import extract_messages, reply_messages
    out = []
    for c in convs:
        out.append({"id": c["id"] + "-extract", "task": "extract",
                    "messages": extract_messages(c["message"], c["extraction"])})
        out.append({"id": c["id"] + "-reply", "task": "reply",
                    "messages": reply_messages(c["result"], c["reply"])})
    return out


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
