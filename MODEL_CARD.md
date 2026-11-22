# Model Card: FairRent Akure (Gemma 3 1B + LoRA)

## Model details

| | |
|---|---|
| **Base model** | [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it): 1 billion parameters, instruction-tuned, decoder-only transformer |
| **Fine-tuning** | LoRA: rank 16, alpha 32, dropout 0.05, applied to all attention and MLP projections (q, k, v, o, gate, up, down). About 1% of the weights are trained. |
| **Training** | 3 epochs, learning rate 2e-4 with cosine schedule and 5 warm-up steps, batch size 1 with 8 gradient-accumulation steps (effective batch 8), max length 1,024 tokens. The loss is computed on assistant tokens only. |
| **Hardware** | Google Colab, one T4 GPU (float32) |
| **Code** | `notebooks/02_finetune_gemma_lora.ipynb` |
| **System type** | Conversational system: LLM, then rule-based fee engine, then LLM |
| **Version** | v1, November 2026 |

### How the system works

```
Tenant message ──► Gemma + LoRA (extract) ──► JSON: area, house type, fees, language
                                                      │
                                                      ▼
                            Fee engine (survey statistics, transparent rules)
                                                      │  verdict for each fee
                                                      ▼
                 Gemma + LoRA (reply) ──► safeguard: every number must come from the engine
                                                      │  (if not, use the template reply)
                                                      ▼
                                              Answer to the tenant
```

The model never decides whether a fee is fair. The engine does that, using survey data. The model only **reads** messages and **explains** the engine's result. This keeps answers grounded and lets the data be updated without retraining.

## Intended use

- **Intended users:** tenants in Akure, especially students around FUTA, checking a rent and fee package **before paying**. It can also be used by student unions and tenant groups.
- **Intended uses:** checking whether fees look normal, finding typical rents for common house types, and getting practical next steps (ask for receipts, confirm the rent with the landlord).
- **Out of scope:** legal advice, rent payments, judging individual agents, property sales or commercial property, and areas or house types with too few survey reports. In those cases the system says "not enough data".

## Training data

Based on 124 training conversations (248 examples) built from 46 anonymised tenant responses to the FairRent Akure survey (see [DATA_CARD.md](DATA_CARD.md)). Of these, 26 conversations use real reported packages, 78 are labelled variants of real records with inflated or unusual fees, and 20 are rent questions, messages with the rent missing, and off-topic messages. Messages are in English, Pidgin and Yoruba.

## Evaluation

**Test set:** 48 held-out conversations. They come from respondents who are not in training, and they use message templates that are not in training: 20 English, 19 Pidgin and 9 Yoruba.

**Baselines:** the same Gemma 3 1B without fine-tuning (zero-shot, same instructions), and Gemini Flash (zero-shot, same instructions; optional).

| Metric | Definition |
|---|---|
| JSON valid | The extraction parses as JSON |
| Intent acc. | The type of request is correct (fee check, rent question, off-topic) |
| Field acc. | Area, house type, language and the 5 fee amounts are correct |
| Fees exact | Every fee amount, including other fees, is exactly right |
| Exact match | The whole extraction is correct |
| Verdict acc. | End to end: after the engine runs on the model's extraction, each fee gets the right amount and verdict |
| Grounded replies | Every number in the reply appears in the engine result |
| Fee coverage | Every fee is mentioned with its amount |
| Language match | Pidgin users get Pidgin; others get English |

> ### Evaluation results

Results on the 48 held-out test conversations (unseen respondents and unseen message templates). All values are percentages. The Gemini baseline was not run.

| Model | JSON valid | Intent acc. | Field acc. | Fees exact | Exact match | Verdict acc. | Grounded replies | Fee coverage | Language match |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Gemma 3 1B (base) | 27.1 | 25.0 | 20.6 | 10.4 | 2.1 | 19.9 | 20.8 | 85.2 | 60.4 |
| **Gemma 3 1B + LoRA (ours)** | **100** | **91.7** | **94.5** | **91.7** | **56.2** | **89.6** | **83.3** | **100** | **95.8** |

The training loss curve is in `results/training_loss.png`. Per-example outputs are in `results/test_predictions.csv`.

**What the results show**

- **Fine-tuning made the model usable.** Without fine-tuning, Gemma 3 1B returned valid JSON only 27% of the time and got almost no extraction fully right (2.1%). After LoRA fine-tuning on 124 conversations, every output was valid JSON. Field accuracy rose from 20.6% to 94.5%.
- **Targets met for fees and verdicts.** Fee amounts were exactly right in 91.7% of conversations (target: 90%). After the fee engine ran on the model's own extraction, 89.6% of fees got the correct amount and verdict (target: 85%).
- **Exact match is lower (56.2%)** because it requires every field to be right at once: intent, area, house type, language and all fees. Most misses are small, such as the area or the language label, and they don't change the fee verdicts. That is why verdict accuracy stays at 89.6%.
- **Grounded replies: 83.3% before the safeguard.** In 8 of 48 replies, the model wrote at least one number that is not in the engine result. The groundedness safeguard catches these and replaces them with the template reply, so **no tenant sees an invented number**. Fee coverage is 100%: every fee was mentioned with its amount.
- **Language match rose from 60.4% to 95.8%.** Pidgin users get Pidgin replies and others get English.

**Against the project's success targets:** fee extraction 91.7% (target ≥ 90%) ✅ · verdict accuracy 89.6% (target ≥ 85%) ✅ · grounded answers shown to users 100% with the safeguard (target 100%) ✅. Model replies before the safeguard (83.3%) are the main area to improve, with more reply training examples.

## Limitations

- **Small, student-heavy data.** Typical rents exist only for self-contains (FUTA South Gate, FUTA North Gate, and across Akure) and single rooms. Other house types and areas get "not enough data".
- **Template-based messages.** Real users write more varied and messier messages, so accuracy on real messages will be lower than on the test set.
- **Yoruba.** The Yoruba messages were written by the team, not checked by native speakers, and replies to Yoruba messages are in English.
- **Thresholds are guidance, not law.** The 10% and 20% rules come from what tenants reported. They are not legal limits.
- **Thin evidence for "red flag" fees.** Unusual fees, such as development or caretaker fees, are flagged because few tenants report them, not because they are always illegitimate.
- **A 1B model can still make mistakes** in reading amounts, for example confusing 25k with 250k. The tenant should check the amounts FairRent repeats back.

## Risks and mitigations

| Risk | Who is affected | Mitigation |
|---|---|---|
| Wrong verdict leads a tenant to dispute a fair fee, or miss an overcharge | Tenants, landlords | Verdicts come from transparent rules. Each answer shows the number of reports. "Not enough data" is used instead of guessing. Replies advise confirming with the landlord. |
| Invented numbers (hallucination) | Tenants | A groundedness safeguard replaces any reply containing a number that is not in the engine result. |
| Harm to agents' reputation | Agents | No agent names are collected or shown. The tool judges fees, not people. |
| Privacy | Respondents | No names. Contacts are dropped before files are written. Under-18s are excluded. Only aggregate statistics are shown. Data is handled in line with the Nigeria Data Protection Act 2023. |
| Exclusion | People without smartphones, Yoruba-only speakers | Planned: a WhatsApp interface, native-speaker Yoruba data, and in-person survey collection beyond FUTA. |

## Environmental impact

The model is the smallest Gemma 3 instruction model, trained with LoRA instead of full fine-tuning, for 3 short epochs (about 100 optimisation steps on one T4 GPU). The training time is recorded in `results/training_log.json`. The rent data is updated in a CSV file, not by retraining.

## Responsible use

FairRent gives **guidance based on community survey data, not legal advice**. It should be used alongside receipts, written agreements, and direct contact with landlords.
