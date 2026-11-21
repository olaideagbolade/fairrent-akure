# Data Card: FairRent Akure Housing Survey and Conversation Dataset

## Summary

| | |
|---|---|
| **Name** | FairRent Akure Housing Survey (v1, November 2026) |
| **Purpose** | Ground a conversational assistant that tells Akure tenants whether their rent and house fees are fair |
| **Collected by** | The FairRent Akure team, through a consented Google Form survey |
| **Collection period** | 2 to 12 November 2026 |
| **Respondents** | 52 responses: 46 tenants and 4 landlords used. 1 person declined consent and 1 was under 18; both were removed. |
| **Location** | Akure, Ondo State, Nigeria. Mostly around the Federal University of Technology Akure (FUTA). |
| **Derived dataset** | 172 conversations (124 train, 48 test), giving 344 supervised examples |
| **Languages** | English, Nigerian Pidgin, Yoruba (mixed with English) |
| **Licence** | CC BY 4.0 for the anonymised data |

## Files

| File | Contents |
|---|---|
| `data/tenants_clean.csv` | 46 anonymised tenant responses: area, house type, rent, each fee, how they found the house, experiences with agents |
| `data/landlords_clean.csv` | 4 anonymised landlord responses |
| `data/rent_reference.json` | Typical rents (25th percentile, median, 75th percentile) for groups with at least 5 reports |
| `data/survey_summary.json` | Headline findings |
| `data/conversations_{train,test}.jsonl` | Each conversation: user message, gold extraction, fee engine result, gold reply |
| `data/sft_{train,test}.jsonl` | The same conversations in chat format for fine-tuning: one *extract* and one *reply* example each |

The raw Google Form export is **not** included because it contains phone numbers and email addresses.

## Key findings from the survey (46 tenants)

- **52%** (24 tenants) later found that the landlord's real rent was lower than the agent had said.
- **37%** (17) believe an agent showed them a house just to collect an inspection fee.
- **52%** (24) got no receipt or written breakdown of fees.
- **98%** would use (Yes) or might use (Maybe) a fee-checking assistant.
- The average fairness rating of the total cost was **3.4 out of 5**.
- Among tenants who paid each fee, the median agency fee was **12.8%** of rent (upper quartile 20.6%), the median agreement fee **6.3%** (11.8%), and the median caution fee **10.8%** (13.7%). The median inspection fee was about **₦2,100 per house**.
- The typical yearly rent for a self-contain was **₦180,000 to ₦250,000** (median ₦200,000, 33 reports). For a single room it was **₦58,750 to ₦102,500** (8 reports).

## Data card questions

| Section | Question | Response |
|---|---|---|
| **Input versus output** | What have users inputted? What outputs or labels did the model predict or generate? | **Inputs:** survey answers (multiple choice, amounts in ₦, short text). At runtime, the input is a tenant's typed message in English, Pidgin or Yoruba describing a rent and fee package. **Outputs:** (1) a structured JSON extraction of intent, area, house type, language and each fee; (2) a short reply giving each fee's verdict, the typical range, and next steps. |
| **Classification problem** | What was the classification task, if any? | FairRent is a conversational system with two classification steps. **Intent:** fee check, rent question or out of scope. **Fee verdict:** normal, above typical, red flag or not enough data. The verdict comes from a transparent rule engine using survey statistics, not from the language model, so every judgement can be traced to data. |
| **Labeling** | Where did labels come from? Did you check consistency? | Fee categories come from respondents themselves, who typed amounts into named fee fields. Free-text "other fees" were parsed by code and reviewed by hand. Verdict thresholds come from survey quartiles (for example, agency fee 10% normal and 20% upper limit), rounded to numbers people remember. Gold extractions are built from the structured survey fields, so they are exact. Consistency checks flag rows whose totals don't add up. Unit tests confirm that gold labels score 100% on every metric. |
| **Data creation** | How did you collect the data? | **Direct source:** a Google Form shared through FUTA student networks and community contacts, with consent on the first page. **Conversation data:** user messages were generated from each real respondent's fees, using templates written in English, Pidgin and Yoruba. In addition, 100 "variant" packages were made from real records by raising agency fees or adding unusual fees, so the model sees overcharging cases. Variants are labelled `variant:` in the `source` field and are never used for rent statistics. |
| **Web collection ethics** | Did you source data from the web? | No. Nothing was scraped. All data comes from the consented survey. No names were collected. Phone numbers and emails (given only by people who volunteered for testing) are dropped before any file is written. Exact timestamps are removed. |
| **Bias and representation** | Who might be left out? | 43 of 46 tenants are FUTA students, aged 18 to 34. Most live around FUTA South Gate (23) and North Gate (12). Working families, older residents, people without smartphones, and other parts of Akure (Alagbaka, Ijapo, Oba-Ile) are barely represented. There are only 4 landlords and **no agents** responded. So rent references exist only for self-contains (FUTA South Gate, North Gate, and across Akure) and single rooms. For other house types and areas the engine says **"not enough data"** instead of guessing. Yoruba messages were written by the project team and **need native-speaker review**. Replies to Yoruba messages are in English. |
| **Values and community alignment** | Does the design reflect the project's values? How will the community benefit? | It reflects **fairness, transparency, accountability and inclusion**. Responses are anonymous, and only aggregate statistics are shown to users. Each answer states how many reports it is based on. Thresholds are published in code and in this card, so anyone can question them. Results will be shared back with respondents and the FUTA Students' Union, and the rent reference will be updated as more reports come in. |

## Cleaning steps (`src/fairrent/cleaning.py`)

1. Keep only consenting adults. Split respondents into tenants and landlords by role.
2. Drop identifying columns: timestamp, testing opt-in, and phone/email.
3. Parse amounts ("000" becomes 0, and commas and ₦ signs are removed).
4. Standardise area names (for example, "Westgate" becomes FUTA West Gate) and house types.
5. Split "other fee" text into name and amount, and mark utility charges such as light or meter.
6. Add `quality_flags` instead of editing numbers. 9 rows are flagged, for example where the total paid is lower than the rent, the inspection total is greater than or equal to the rent, or the rent is implausible (₦18,000, probably a typo). Flagged rows are excluded from the conversation dataset. Only rows with an implausible rent are excluded from rent statistics.

## Train/test split

Respondents with clean data, and an area and house type the templates cover, were split 70/30 at random (seed 42) **before** any messages were generated. The test set also uses **its own message templates**, so test phrasings never appear in training.

| Split | Survey | Variant | Rent question | Missing rent | Out of scope | Total |
|---|---|---|---|---|---|---|
| Train | 26 | 78 | 8 | 6 | 6 | **124** |
| Test | 11 | 22 | 5 | 4 | 6 | **48** |

## Limitations

- The sample is small (46 tenants) and student-heavy, so rent ranges are rough guides, not market prices.
- Self-reported amounts may be misremembered.
- No agent perspective is included.
- Messages are template-based. Real users will write more varied and messier messages, so the model must be tested with real users before launch.
