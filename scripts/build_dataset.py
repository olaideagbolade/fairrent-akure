"""Step 2: build the conversation dataset from the cleaned survey.

    python scripts/build_dataset.py

Writes:
    data/conversations_train.jsonl / conversations_test.jsonl  (message, gold extraction, gold reply)
    data/sft_train.jsonl / sft_test.jsonl                       (chat-format examples for fine-tuning)
"""
import collections
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fairrent.dataset_builder import build, to_sft_examples, write_jsonl  # noqa: E402

tenants = pd.read_csv(ROOT / "data" / "tenants_clean.csv", keep_default_na=False, na_values=[""])
convs = build(tenants)
for split in ["train", "test"]:
    rows = [c for c in convs if c["split"] == split]
    write_jsonl(rows, ROOT / "data" / f"conversations_{split}.jsonl")
    write_jsonl(to_sft_examples(rows), ROOT / "data" / f"sft_{split}.jsonl")
    sources = collections.Counter(c["source"].split(":")[0] for c in rows)
    langs = collections.Counter(c["extraction"]["language"] for c in rows)
    print(f"{split}: {len(rows)} conversations ({2 * len(rows)} training examples)  "
          f"sources={dict(sources)}  languages={dict(langs)}")
