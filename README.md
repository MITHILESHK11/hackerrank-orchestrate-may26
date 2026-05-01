# Support Triage CLI

Installable CLI project for multi-domain support ticket triage across HackerRank, Claude, and Visa.

## Approach Overview

The agent processes each ticket with a deterministic safety-first pipeline:

1. Normalize input: clean issue, subject, and company fields.
2. Rule-based classification: detect company, request type, and out-of-scope cases.
3. Escalation detection: apply hard and soft escalation rules before LLM generation.
4. Retrieval grounding: retrieve relevant corpus passages from `data/` using BM25 with optional semantic reranking.
5. Response generation:
   - Uses Gemini JSON output when `GEMINI_API_KEY` is present.
   - Falls back to grounded non-LLM response generation when Gemini is unavailable.
6. Output writing: write strict CSV schema in deterministic column order.

This keeps risky tickets safe, avoids unsupported claims, and still runs fully offline except optional Gemini calls.

## Project Structure

```text
.
|-- code/                  # Python source modules
|-- data/                  # Local support corpus
|-- support_tickets/       # Input/output CSVs
|-- pyproject.toml         # Packaging + CLI entrypoints
|-- .env.example
`-- README.md
```

## Setup Instructions

1. Clone the repository and move to repo root.
2. Create a virtual environment.
3. Activate the environment.
4. Install the project in editable mode.

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Python version: `3.10+`.

## Environment Variables

Optional `.env` file at repo root:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
```

If `GEMINI_API_KEY` is missing, the CLI still works using retrieval-grounded fallback mode.

## CLI Usage

Primary commands installed by `pip install -e .`:

- `triage-cli`
- `triage-eval`

If command aliases are not on `PATH`, use module form:

- `python -m main`
- `python -m evaluator`

### Interactive CLI

```bash
triage-cli --interactive
```

Available in-app commands:

- `triage` process a single ticket
- `sample` run sample CSV
- `full` run full CSV
- `custom` run custom input/output paths
- `eval` evaluate sample predictions
- `help` show command help
- `exit` quit

### Batch Runs

Run sample dataset:

```bash
triage-cli --sample --verbose
```

Run full dataset:

```bash
triage-cli
```

Default output files:

- full run -> `support_tickets/output.csv`
- sample run -> `support_tickets/output.sample.csv`

Custom I/O:

```bash
triage-cli --input support_tickets/support_tickets.csv --output support_tickets/output.csv
```

## Evaluation

Recommended:

```bash
triage-eval --sample support_tickets/sample_support_tickets.csv --output support_tickets/output.sample.csv
```

Module fallback:

```bash
python -m evaluator --sample support_tickets/sample_support_tickets.csv --output support_tickets/output.sample.csv
```

If evaluation reports `Matched output rows: 0`, run sample prediction first using `sample` or `--sample`.

## Output Contract

Generated CSV columns are always:

```text
issue,subject,company,response,product_area,status,request_type,justification
```
