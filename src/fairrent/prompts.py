"""Prompts shared by training, evaluation and the demo.

FairRent uses one model for two tasks:
  1. EXTRACT: read the tenant's message and return structured JSON.
  2. REPLY:   turn the fee engine's result into a friendly answer.
The fee engine (engine.py) sits between them, so every number in a reply
comes from survey data rather than the model's memory.
"""
import json

AREAS = [
    "FUTA South Gate", "FUTA North Gate", "FUTA West Gate", "Apatapiti", "Aule",
    "Oba-Ile", "Ologede", "Wesco", "Alagbaka", "Ijapo Estate", "Oke-Ijebu",
    "Shagari Village", "Isinkan", "Arakale", "Oda Road", "Ondo Road", "Ijoka", "Roadblock",
]
HOUSE_TYPES = ["Single room", "Self-contain", "Mini flat", "2-bedroom flat",
               "3-bedroom flat", "Bungalow or duplex", "PG hostel"]

EXTRACT_SYSTEM = (
    "You are the reader for FairRent Akure, a rent and fee checker for Akure, Nigeria. "
    "Read the user's message (English, Nigerian Pidgin or Yoruba) and reply with ONLY a JSON object:\n"
    '{"intent": "fee_check" | "rent_info" | "other", "area": <area or null>, '
    '"house_type": <house type or null>, "fees": {"rent": <int or null>, "agency": <int or null>, '
    '"agreement": <int or null>, "caution": <int or null>, "inspection": <int or null>, '
    '"other": [{"name": <str>, "amount": <int>}]}, "language": "en" | "pcm" | "yo"}\n'
    f"Areas: {', '.join(AREAS)}.\n"
    f"House types: {', '.join(HOUSE_TYPES)}.\n"
    "Write amounts as whole naira (250k = 250000). Use null for anything not in the message. "
    "Never guess an amount. Use intent \"other\" for questions that are not about rent or house fees."
)

REPLY_SYSTEM = (
    "You are FairRent Akure, a friendly assistant that helps tenants in Akure, Nigeria check "
    "whether their rent and house fees are fair. You receive the CHECK RESULT from the FairRent "
    "fee engine. Write a short reply in the user's language (en = English, pcm = Nigerian Pidgin; "
    "for yo reply in simple English with a Yoruba greeting). Use ONLY facts and numbers in the "
    "CHECK RESULT. List each fee with its amount and verdict, mention the typical range and "
    "number of reports, then give practical next steps. Never invent numbers. You give guidance, "
    "not legal advice."
)


def compact_result(result: dict) -> dict:
    """Engine result without long reason strings and empty values, to keep prompts short."""
    out = {k: v for k, v in result.items() if v not in (None, [], {})}
    if "items" in out:
        out["items"] = [
            {k: v for k, v in item.items() if k != "reason" and v is not None}
            for item in out["items"]
        ]
    return out


def extract_messages(user_text: str, answer: dict | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": user_text}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
    return msgs


def reply_messages(result: dict, answer: str | None = None) -> list[dict]:
    payload = json.dumps(compact_result(result), ensure_ascii=False)
    msgs = [{"role": "system", "content": REPLY_SYSTEM},
            {"role": "user", "content": f"LANGUAGE: {result.get('language', 'en')}\nCHECK RESULT: {payload}"}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs
