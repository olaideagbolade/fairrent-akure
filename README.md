# FairRent Akure

**A Gemma-powered conversational assistant that helps tenants in Akure, Nigeria check whether their rent and house fees are fair.**

In Akure, informal and uncertified house agents control access to rental housing. A house listed at ₦300,000 often becomes a ₦500,000 "total package" after agency, agreement and unexplained fees. Our survey of 46 Akure tenants found:

| Finding | Share of tenants |
|---|---|
| Later found the landlord's real rent was **lower than the agent said** | **52%** |
| Believe an agent showed them a house **just to collect an inspection fee** | **37%** |
| Got **no receipt** for the fees they paid | **52%** |
| **Would use** (or might use) a fee-checking assistant | **98%** |

FairRent lets a tenant type their fee breakdown in **English, Nigerian Pidgin or Yoruba** and get a clear answer: which fees are normal, which look inflated, what other tenants paid, and what to do next.

> **Tenant:** *Agent wan collect money for selfcon for southgate. Rent na 280k, agency 56k, caution 50k, development fee 40k. E make sense?*
>
> **FairRent:**
> - Rent ₦280,000: ⚠️ E pass normal
> - Agency fee ₦56,000: ⚠️ E pass normal (20% of rent; normal na 10% or less)
> - Caution fee ₦50,000: ⚠️ E pass normal (18% of rent; normal na 15% or less)
> - Development fee ₦40,000: 🚩 Red flag (no be many people dey pay this one)
>
> Most people wey report Self-contain in FUTA South Gate dey pay ₦172,500–₦250,000 for one year (16 reports). …

## How it works

```
Tenant message ──► Gemma 3 1B + LoRA (extract) ──► JSON: area, house type, fees, language
                                                          │
                                                          ▼
                              Fee engine: compares each fee with survey statistics
                                                          │
                                                          ▼
                 Gemma 3 1B + LoRA (reply) ──► safeguard: every number must come from the engine
                                                          │
                                                          ▼
                                                    Answer to the tenant
```

- **The model reads and explains. It never decides.** Verdicts come from a transparent rule engine built on survey statistics, so every number can be traced to data. ([`engine.py`](src/fairrent/engine.py))
- **Groundedness safeguard:** if a model reply contains a number that is not in the engine result, the system shows a template reply instead. ([`llm.py`](src/fairrent/llm.py))
- **Honest about gaps:** for house types or areas with fewer than 5 reports, it says "not enough data".

## Repository structure

```
fairrent-akure/
├── README.md               this file
├── DATA_CARD.md            dataset documentation
├── MODEL_CARD.md           model documentation and evaluation results
├── requirements.txt
├── data/                   anonymised survey, rent reference, conversation dataset
├── notebooks/
│   ├── 01_data_and_dataset.ipynb      explore survey, build dataset (CPU)
│   ├── 02_finetune_gemma_lora.ipynb   fine-tune Gemma 3 1B with LoRA (T4 GPU)
│   ├── 03_evaluate_models.ipynb       base Gemma vs fine-tuned vs Gemini (T4 GPU)
│   └── 04_demo_chat.ipynb             full pipeline + Gradio chat demo (T4 GPU)
├── scripts/
│   ├── prepare_data.py     raw survey export → anonymised CSVs and statistics
│   └── build_dataset.py    clean survey → train/test conversations
├── src/fairrent/
│   ├── cleaning.py         anonymise and clean survey answers
│   ├── engine.py           fee fairness engine (all numbers come from here)
│   ├── composer.py         reference replies in English and Pidgin
│   ├── dataset_builder.py  conversation dataset with leakage-safe split
│   ├── prompts.py          system prompts for both tasks
│   ├── evaluate.py         metrics
│   └── llm.py              model loading, generation, full pipeline
├── tests/test_fairrent.py  unit tests
└── results/                training log, evaluation tables, predictions
```

## Reproduce

**1. Data (CPU, about 1 minute).** Open `notebooks/01_data_and_dataset.ipynb` in Colab, or run locally:

```bash
pip install pandas pytest
python scripts/build_dataset.py   # rebuilds data/conversations_*.jsonl and data/sft_*.jsonl
python -m pytest -q               # 8 tests
```

(`scripts/prepare_data.py` needs the raw survey export, which is private because it contains contact details. The anonymised output is already in `data/`.)

**2. Fine-tune (Colab T4 GPU, about 15–30 minutes).**
- Accept the Gemma licence on [Hugging Face](https://huggingface.co/google/gemma-3-1b-it).
- Add `HF_TOKEN` to Colab Secrets.
- Run `notebooks/02_finetune_gemma_lora.ipynb`. The adapter is saved to Google Drive.

**3. Evaluate.** Run `notebooks/03_evaluate_models.ipynb`. It is optional to add `GEMINI_API_KEY` to Colab Secrets for the Gemini baseline. The notebook writes results to `results/`.

**4. Demo.** Run `notebooks/04_demo_chat.ipynb` to get a shareable Gradio chat link.

In each notebook, set `REPO_URL` in the first code cell to this repository's URL.

## Results

See [MODEL_CARD.md](MODEL_CARD.md#evaluation-results) and `results/eval_table.md`. Everything is evaluated on 48 held-out conversations from unseen respondents and unseen message templates.

## Responsible innovation

- **Stakeholders:** tenants (low power), landlords (high), house agents (high, informal), and Ondo State regulators (high, formal). The full stakeholder map, governance blueprint and environmental plan are in the project proposal.
- **Privacy:** no names are collected. Phone numbers and emails are dropped before any file is written. Under-18s are excluded. The project follows the Nigeria Data Protection Act 2023.
- **Never allowed:** legal advice, invented figures, naming or rating individual agents, or collecting payments.
- **Sustainability:** we use the smallest Gemma 3 model, LoRA instead of full fine-tuning, and about 100 training steps on one T4. The data is updated without retraining.

## Limitations

The dataset is small and mostly from FUTA students. Messages are template-based. Yoruba needs native-speaker review. The thresholds are guidance, not law. See the [data card](DATA_CARD.md) and [model card](MODEL_CARD.md).

## Licence

Code: MIT. Anonymised data: CC BY 4.0.
