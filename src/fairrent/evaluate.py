"""Metrics for FairRent.

Extraction (model reads the message):
    json_valid        reply parses as JSON
    intent_acc        intent matches the gold intent
    field_acc         share of fields (area, house type, language, 5 fees) that match
    fees_exact        all fee amounts (incl. other fees) exactly right
    exact_match       the whole extraction is right
End-to-end (predicted extraction -> fee engine):
    verdict_acc       share of gold fee items that get the same amount and verdict
Reply (model writes the answer from the engine result):
    grounded          every number in the reply appears in the engine result
    coverage          every fee in the result is mentioned with its amount
    language_ok       reply is in the expected language (English or Pidgin)
"""
from __future__ import annotations

import json
import re

FEE_KEYS = ["rent", "agency", "agreement", "caution", "inspection"]
PIDGIN_MARKERS = (" dey ", " na ", " wey ", "wetin", "e be like", "no be ", "abeg")


# ---------- parsing ----------
def parse_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply (handles ```json fences)."""
    if not isinstance(text, str):
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _str_or_none(v):
    return v if isinstance(v, str) and v.strip() else None


def _to_int(v):
    try:
        return None if v in (None, "", "null") else int(float(str(v).replace(",", "")))
    except (ValueError, TypeError, OverflowError):
        return None


def normalise_extraction(ext: dict | None) -> dict | None:
    """Fill missing keys and coerce types, so badly formed model output never crashes scoring."""
    if not isinstance(ext, dict):
        return None
    fees = ext.get("fees") or {}
    if not isinstance(fees, dict):
        fees = {}
    clean = {k: _to_int(fees.get(k)) for k in FEE_KEYS}
    others = []
    raw_other = fees.get("other")
    for o in (raw_other if isinstance(raw_other, list) else []):
        if isinstance(o, dict):
            amount = _to_int(o.get("amount"))
            if amount is not None:
                others.append({"name": str(o.get("name") or "Other fee"), "amount": amount})
    clean["other"] = others
    intent = ext.get("intent")
    language = ext.get("language")
    return {"intent": intent if intent in ("fee_check", "rent_info", "other") else "fee_check",
            "area": _str_or_none(ext.get("area")), "house_type": _str_or_none(ext.get("house_type")),
            "fees": clean, "language": language if language in ("en", "pcm", "yo") else "en"}


# ---------- extraction ----------
def score_extraction(pred: dict | None, gold: dict) -> dict:
    gold = normalise_extraction(gold)
    p = normalise_extraction(pred)
    if p is None:
        return {"json_valid": 0, "intent_acc": 0, "field_acc": 0.0, "fees_exact": 0, "exact_match": 0}
    fields = ["area", "house_type", "language"]
    hits = [p[f] == gold[f] for f in fields] + [p["fees"][k] == gold["fees"][k] for k in FEE_KEYS]
    other_ok = sorted(o["amount"] for o in p["fees"]["other"]) == sorted(o["amount"] for o in gold["fees"]["other"])
    fees_exact = all(p["fees"][k] == gold["fees"][k] for k in FEE_KEYS) and other_ok
    return {
        "json_valid": 1,
        "intent_acc": int(p["intent"] == gold["intent"]),
        "field_acc": sum(hits) / len(hits),
        "fees_exact": int(fees_exact),
        "exact_match": int(fees_exact and p["intent"] == gold["intent"] and all(hits)),
    }


def score_verdicts(pred_result: dict | None, gold_result: dict) -> float:
    """Share of gold items reproduced with the same amount and verdict."""
    if gold_result.get("status") != "ok" or "items" not in gold_result:
        return float(pred_result is not None and pred_result.get("status") == gold_result.get("status"))
    if not pred_result or "items" not in pred_result:
        return 0.0
    got = {(i["amount"], i["verdict"]) for i in pred_result["items"]}
    gold_items = gold_result["items"]
    return sum((i["amount"], i["verdict"]) in got for i in gold_items) / len(gold_items)


# ---------- reply ----------
def _numbers_in(obj, out: set):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(round(float(obj), 2))
        if 0 < obj <= 1:
            out.add(float(round(obj * 100)))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers_in(v, out)
    elif isinstance(obj, str):  # numbers inside names, e.g. "2-bedroom flat"
        for m in re.findall(r"\d+", obj):
            out.add(float(m))


def reply_numbers(text: str) -> list[float]:
    nums = re.findall(r"₦?\d[\d,]*(?:\.\d+)?", text)
    return [float(n.replace("₦", "").replace(",", "")) for n in nums]


def score_reply(reply: str, result: dict) -> dict:
    allowed = {1.0, 2.0, 3.0, 4.0}  # numbered next steps
    _numbers_in(result, allowed)
    nums = reply_numbers(reply or "")
    ungrounded = [n for n in nums if round(n, 2) not in allowed]
    items = result.get("items", [])
    covered = [f"{int(i['amount']):,}" in (reply or "") for i in items]
    lang = result.get("language", "en")
    padded = f" {(reply or '').lower()} "
    has_pidgin = any(m in padded for m in PIDGIN_MARKERS)
    language_ok = has_pidgin if lang == "pcm" else not has_pidgin
    return {
        "grounded": int(not ungrounded),
        "ungrounded_numbers": ungrounded,
        "coverage": sum(covered) / len(covered) if covered else 1.0,
        "language_ok": int(language_ok),
    }


def summarise(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in keys}
