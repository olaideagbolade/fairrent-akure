"""Run with:  python -m pytest -q"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fairrent.cleaning import parse_other_fee, to_naira  # noqa: E402
from fairrent.composer import compose_reply  # noqa: E402
from fairrent.engine import FeeEngine  # noqa: E402
from fairrent.evaluate import parse_json, score_extraction, score_reply, score_verdicts  # noqa: E402

ENGINE = FeeEngine(pd.read_csv(ROOT / "data" / "tenants_clean.csv", keep_default_na=False, na_values=[""]))
TEST = [json.loads(l) for l in open(ROOT / "data" / "conversations_test.jsonl", encoding="utf-8")]


def test_money_parsing():
    assert to_naira("250000") == 250000
    assert to_naira("000") == 0
    assert to_naira(" 30,000 ") == 30000
    assert to_naira("") is None


def test_other_fee_parsing():
    assert parse_other_fee("Caretaker fee: 50000") == ("Caretaker fee", 50000.0, False)
    assert parse_other_fee("Light 3500 a month")[2] is True
    assert parse_other_fee("None") == (None, None, False)


def test_agency_thresholds():
    base = {"intent": "fee_check", "area": "FUTA South Gate", "house_type": "Self-contain", "language": "en"}
    verdict = lambda agency: ENGINE.check({**base, "fees": {"rent": 200000, "agency": agency}})["items"][1]["verdict"]
    assert verdict(20000) == "normal"         # 10%
    assert verdict(30000) == "above_typical"  # 15%
    assert verdict(60000) == "red_flag"       # 30%


def test_unknown_fee_is_red_flag_and_utility_is_normal():
    r = ENGINE.check({"intent": "fee_check", "house_type": "Self-contain", "language": "en",
                      "fees": {"rent": 200000, "other": [{"name": "Development fee", "amount": 40000},
                                                         {"name": "Light bill", "amount": 3000}]}})
    verdicts = {i["fee"]: i["verdict"] for i in r["items"]}
    assert verdicts["Development fee"] == "red_flag"
    assert verdicts["Light bill"] == "normal"


def test_missing_rent_asks_for_it():
    r = ENGINE.check({"intent": "fee_check", "language": "en", "fees": {"agency": 30000}})
    assert r["status"] == "needs_info"
    assert "rent" in compose_reply(r).lower()


def test_gold_predictions_score_perfectly():
    for c in TEST:
        assert score_extraction(c["extraction"], c["extraction"])["exact_match"] == 1
        assert score_verdicts(ENGINE.check(c["extraction"]), c["result"]) == 1.0
        s = score_reply(c["reply"], c["result"])
        assert s["grounded"] == 1, (c["id"], s)
        assert s["coverage"] == 1.0
        assert s["language_ok"] == 1, (c["id"], c["reply"])


def test_invented_number_is_caught():
    c = next(c for c in TEST if c["result"].get("items"))
    assert score_reply(c["reply"] + " Most people pay ₦123,456.", c["result"])["grounded"] == 0


def test_parse_json_handles_fences():
    assert parse_json('```json\n{"intent": "other"}\n```') == {"intent": "other"}
    assert parse_json("no json here") is None
