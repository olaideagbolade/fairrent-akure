"""Reference replies written from the engine result.

These templates produce the target ("gold") replies used to fine-tune the
model, and a safe fallback if the model's reply fails the groundedness check.
English (en) and Nigerian Pidgin (pcm) are supported. Yoruba (yo) messages get
an English reply with a Yoruba greeting until native-speaker reply data exists.
"""
from __future__ import annotations


STANDARD_FEES = ("Agency fee", "Agreement / legal fee", "Caution fee", "Inspection fee")


def naira(x) -> str:
    return f"₦{int(round(x)):,}"


VERDICT = {
    "en": {"normal": "✅ Normal", "above_typical": "⚠️ Above typical",
           "red_flag": "🚩 Red flag", "not_enough_data": "❔ Not enough data"},
    "pcm": {"normal": "✅ E normal", "above_typical": "⚠️ E pass normal",
            "red_flag": "🚩 Red flag", "not_enough_data": "❔ Data never reach"},
}

TEXT = {
    "en": {
        "intro": "Here is your fee check{where}:",
        "rent_ref": "Most reports for {group} are {p25}–{p75} a year ({n} reports).",
        "no_ref": "I don't have enough reports yet to compare this rent.",
        "pct": "{share}% of rent; usual is {usual}% or less",
        "insp": "usual is {usual} or less per house",
        "other_red": "few tenants report this fee",
        "other_util": "a utility charge, not a move-in fee",
        "total": "Total you were asked to pay: {total}.",
        "attention": "Needs attention: {items}.",
        "all_fair": "Everything here is within the usual range.",
        "partly_fair": "No fee is above the usual range, but I could not compare the rent.",
        "steps": "What you can do:",
        "receipt": "Ask for a written receipt for every fee.",
        "landlord": "Confirm the rent directly with the landlord before paying.",
        "negotiate": "Ask the agent to reduce the fees marked above typical.",
        "ask_other": "Ask what the {name} covers and who receives it.",
        "caution": "Get it in writing that the caution fee is refundable.",
        "fair_step": "It looks fair. Still collect a receipt for every payment.",
        "footer": "Based on Akure tenant survey reports. This is guidance, not legal advice.",
        "need_rent": "To check your fees I need the yearly rent. How much is the rent, and which other fees did the agent ask for?",
        "need_type": "Which type of house is it (single room, self-contain, mini flat...) and in which area?",
        "rent_info": "For {group}, most tenants reported {p25}–{p75} a year, with a middle value of {median} ({n} reports).",
        "rent_info_none": "I don't have enough reports for that type of house{where} yet. Tell me what you were quoted and I will check the fees.",
        "rent_info_tip": "Send me the full fee breakdown and I will check it.",
        "out_of_scope": "Sorry, I can only help with house rent and fees in Akure. Send me your rent and fee breakdown and I will check it.",
    },
    "pcm": {
        "intro": "See how your money take be{where}:",
        "rent_ref": "Most people wey report {group} dey pay {p25}–{p75} for one year ({n} reports).",
        "no_ref": "Reports never reach to compare this rent.",
        "pct": "{share}% of rent; normal na {usual}% or less",
        "insp": "normal na {usual} or less for one house",
        "other_red": "no be many people dey pay this one",
        "other_util": "na light/utility money, no be fee to pack enter",
        "total": "Total wey dem ask you to pay: {total}.",
        "attention": "Check these ones well: {items}.",
        "all_fair": "Everything dey inside normal range.",
        "partly_fair": "No fee pass normal, but I no fit compare the rent.",
        "steps": "Wetin you fit do:",
        "receipt": "Collect receipt for every money wey you pay.",
        "landlord": "Confirm the rent with the landlord yourself before you pay.",
        "negotiate": "Beg the agent make e reduce the fees wey pass normal.",
        "ask_other": "Ask wetin {name} dey cover and who dey collect am.",
        "caution": "Make dem write am down say caution fee go return.",
        "fair_step": "E be like say e fair. But still collect receipt for every payment.",
        "footer": "Na from Akure tenant survey we get these numbers. Na advice, no be legal advice.",
        "need_rent": "To check your fees, I need the rent for one year. How much be the rent, and which other fees the agent ask for?",
        "need_type": "Which kind house be am (single room, self-contain, mini flat...) and for which area?",
        "rent_info": "For {group}, most people dey pay {p25}–{p75} for one year, middle price na {median} ({n} reports).",
        "rent_info_none": "Reports never reach for that kind house{where}. Tell me wetin dem price you, make I check the fees.",
        "rent_info_tip": "Send me the full breakdown of the fees make I check am.",
        "out_of_scope": "Sorry, na only house rent and fees for Akure I fit help with. Send me your rent and fees make I check am.",
    },
}


def _lang(result: dict) -> str:
    return "pcm" if result.get("language") == "pcm" else "en"


def _where(result: dict) -> str:
    parts = [p for p in (result.get("house_type"), result.get("area")) if p]
    if not parts:
        return ""
    if len(parts) == 2:
        return f" ({parts[0]}, {parts[1]})"
    return f" ({parts[0]})"


def compose_reply(result: dict) -> str:
    lang = _lang(result)
    T, V = TEXT[lang], VERDICT[lang]
    greeting = "Ẹ n lẹ́! " if result.get("language") == "yo" else ""
    status = result.get("status")

    if status == "out_of_scope":
        return greeting + T["out_of_scope"]
    if status == "needs_info":
        key = "need_type" if "house_type" in result.get("missing", []) else "need_rent"
        return greeting + T[key]

    ref = result.get("reference")
    if result.get("intent") == "rent_info":
        if not ref:
            where = f" in {result['area']}" if result.get("area") else ""
            return greeting + T["rent_info_none"].format(where=where)
        return greeting + T["rent_info"].format(
            group=ref["group"], p25=naira(ref["p25"]), p75=naira(ref["p75"]),
            median=naira(ref["median"]), n=ref["reports"]) + " " + T["rent_info_tip"]

    lines = [greeting + T["intro"].format(where=_where(result)), ""]
    for item in result["items"]:
        detail = ""
        if item.get("share_of_rent") is not None and "usual_max_share" in item:
            detail = T["pct"].format(share=round(item["share_of_rent"] * 100),
                                     usual=round(item["usual_max_share"] * 100))
        elif "usual_max_amount" in item and item["verdict"] != "normal":
            detail = T["insp"].format(usual=naira(item["usual_max_amount"]))
        elif item["fee"] not in ("Rent", "Inspection fee", "Agency fee",
                                 "Agreement / legal fee", "Caution fee"):
            detail = T["other_util"] if item["verdict"] == "normal" else T["other_red"]
        line = f"- {item['fee']} {naira(item['amount'])}: {V[item['verdict']]}"
        if detail:
            line += f" ({detail})"
        lines.append(line)
    lines.append("")
    lines.append(T["rent_ref"].format(group=ref["group"], p25=naira(ref["p25"]),
                                      p75=naira(ref["p75"]), n=ref["reports"]) if ref else T["no_ref"])
    lines.append(T["total"].format(total=naira(result["total"])))
    flagged = result.get("flagged", [])
    unknown = any(i["verdict"] == "not_enough_data" for i in result["items"])
    if flagged:
        lines.append(T["attention"].format(items=", ".join(flagged)))
    else:
        lines.append(T["partly_fair"] if unknown else T["all_fair"])

    steps = []
    if flagged:
        steps.append(T["receipt"])
        if "Rent" in flagged:
            steps.append(T["landlord"])
        if any(i["verdict"] in ("above_typical", "red_flag") and i["fee"] in STANDARD_FEES
               for i in result["items"]):
            steps.append(T["negotiate"])
        for i in result["items"]:
            if i["verdict"] == "red_flag" and i["fee"] not in ("Rent", "Agency fee",
                                                               "Agreement / legal fee",
                                                               "Caution fee", "Inspection fee"):
                steps.append(T["ask_other"].format(name=i["fee"].lower()))
    elif unknown:
        steps.append(T["receipt"])
        steps.append(T["landlord"])
    else:
        steps.append(T["fair_step"])
    if any(i["fee"] == "Caution fee" for i in result["items"]):
        steps.append(T["caution"])
    lines.append("")
    lines.append(T["steps"])
    lines += [f"{n}. {s}" for n, s in enumerate(steps[:4], 1)]
    lines.append("")
    lines.append(T["footer"])
    return "\n".join(lines)
