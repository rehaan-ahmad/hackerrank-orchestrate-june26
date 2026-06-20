# ClaimVerify AI — Multi-Modal Insurance Claim Verification

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Gemini](https://img.shields.io/badge/Gemini-2.0--Flash-4285F4?logo=google&logoColor=white)
![HackerRank](https://img.shields.io/badge/HackerRank-Orchestrate-00EA64?logo=hackerrank&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Automated damage claim verification pipeline for **cars**, **laptops**, and **packages**. Takes a claim conversation, submitted images, and user history — returns a structured verdict with evidence assessment, risk flags, and severity in under 5 seconds per claim.

Built for [HackerRank Orchestrate](https://hackerrank.com) (24-hour multi-modal AI challenge).

---

## How It Works

Each claim goes through a five-stage reasoning chain:

```
Chat transcript + Images + User history + Evidence requirements
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  1. CLAIM EXTRACTION   — parse intent from conversation     │
│  2. EVIDENCE CHECK     — match images against requirements  │
│  3. VISUAL ANALYSIS    — Gemini 2.0 Flash inspects images  │
│  4. RISK ASSESSMENT    — history flags + image authenticity │
│  5. VERDICT            — supported / contradicted / NEI     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
Structured row in output.csv (14 fields, schema-enforced)
```

Gemini's `response_schema` enforces output structure at the API level — no regex parsing, no markdown stripping.

---

## Architecture

```
code/
├── main.py              # Orchestrator with checkpointing + retry
├── agent.py             # Gemini 2.0 Flash call, response_schema, validation
├── data_loader.py       # CSV joins (claims + history + evidence requirements)
├── prompt_builder.py    # Strategy A (zero-shot) / Strategy B (CoT) templates
├── risk_engine.py       # Pre-VLM rule-based risk flag layer
├── constants.py         # All allowed enum values from problem spec
├── prompts/
│   ├── strategy_a.txt   # Baseline: zero-shot
│   └── strategy_b.txt   # Final: 10-step chain-of-thought
└── evaluation/
    ├── main.py          # Runs both strategies on sample_claims.csv
    ├── metrics.py       # Per-field accuracy + weighted F1
    └── evaluation_report.md
```

---

## Output Schema

Each input claim maps to one output row:

| Field | Type | Description |
|---|---|---|
| `evidence_standard_met` | bool | Images sufficient to evaluate the claim |
| `evidence_standard_met_reason` | str | Why the standard was or wasn't met |
| `risk_flags` | str | Semicolon-separated flags (see allowed values) |
| `issue_type` | enum | Visible damage type |
| `object_part` | enum | Affected part of the object |
| `claim_status` | enum | `supported` / `contradicted` / `not_enough_information` |
| `claim_status_justification` | str | Image-grounded explanation with image IDs |
| `supporting_image_ids` | str | Which images support the decision |
| `valid_image` | bool | Whether images are usable for automated review |
| `severity` | enum | `none` / `low` / `medium` / `high` / `unknown` |

Detectable risk flags: `blurry_image`, `wrong_object`, `claim_mismatch`, `possible_manipulation`, `non_original_image`, `text_instruction_present`, `user_history_risk`, `manual_review_required`, and more.

---

## Evaluation

Two prompting strategies compared on `sample_claims.csv` (20 labeled claims):

| Strategy | Description | claim_status acc | severity acc |
|---|---|---|---|
| A | Zero-shot | — | — |
| B | 10-step CoT | — | — |

*Fill after running `python code/evaluation/main.py`*

**Dataset distribution (sample set):** 8 car / 6 laptop / 6 package claims. 65% supported, 25% contradicted, 10% not enough information.

---

## Setup

```bash
git clone https://github.com/your-username/hackerrank-orchestrate-june26.git
cd hackerrank-orchestrate-june26
pip install -r code/requirements.txt
```

Create `.env` in the repo root:

```env
GEMINI_API_KEY=your_key_here
```

Get a key at [aistudio.google.com](https://aistudio.google.com). Free tier works (15 RPM); the full test run of 44 claims completes in ~4 minutes.

---

## Usage

```bash
# Run on test set → output.csv
python code/main.py

# Run on sample set (development)
python code/main.py --split sample --strategy b --output output_sample.csv

# Compare strategies (runs both on sample set, prints accuracy table)
python code/evaluation/main.py
```

Progress is checkpointed per row — safe to interrupt and resume.

---

## Operational Notes

| Metric | Value |
|---|---|
| Model | `gemini-2.0-flash` |
| Test claims | 44 rows, avg 1.86 images/claim |
| Model calls | 1 per claim (all images in one call) |
| Approx cost (test set) | < $0.05 |
| Avg latency/claim | ~2–4s |
| Rate limit strategy | `ResourceExhausted` → exponential backoff (4s / 8s / 16s) |

---

## License

MIT
