# Support Triage CLI

Production-style CLI project for multi-domain support ticket triage across:

- HackerRank
- Claude
- Visa

The system classifies each ticket, detects escalation risk, retrieves grounding passages from the local support corpus, and produces compliant output rows for `support_tickets/output.csv`.

## Project Layout

```text
.
├── code/                  # Python source modules
├── data/                  # Local support corpus (required)
├── support_tickets/       # Input/output CSV files
├── pyproject.toml         # Installable package metadata + CLI entrypoints
└── README.md
```

## Requirements

- Python 3.10+
- `pip`

## Install (Recommended)

From repository root:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install as editable package:

```bash
python -m pip install -e .
```

This installs two commands:

- `triage-cli`
- `triage-eval`

If your shell does not detect those commands (common on Windows when user script path is not on `PATH`), use:

- `python -m main`
- `python -m evaluator`

## Environment

Optional `.env` at repo root:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
```

If `GEMINI_API_KEY` is absent, the project still runs using grounded retrieval fallback responses.

## Real CLI Interface

Launch interactive mode:

```bash
triage-cli --interactive
```

Fallback:

```bash
python -m main --interactive
```

Commands inside the CLI:

- `triage` one ticket interactively
- `sample` run sample CSV
- `full` run full CSV
- `custom` run custom input/output paths
- `eval` evaluate current output
- `help` show command help
- `exit` quit

## Batch Usage

Run on sample file:

```bash
triage-cli --sample --verbose
```

Fallback:

```bash
python -m main --sample --verbose
```

Run on full file:

```bash
triage-cli
```

By default:

- full run writes `support_tickets/output.csv`
- sample run writes `support_tickets/output.sample.csv`

Custom paths:

```bash
triage-cli --input support_tickets/support_tickets.csv --output support_tickets/output.csv
```

## Evaluate Output

```bash
triage-eval --sample support_tickets/sample_support_tickets.csv --output support_tickets/output.csv
```

Recommended for sample evaluation:

```bash
triage-eval --sample support_tickets/sample_support_tickets.csv --output support_tickets/output.sample.csv
```

Fallback:

```bash
python -m evaluator --sample support_tickets/sample_support_tickets.csv --output support_tickets/output.csv
```

## Output Schema

Output columns are always written in this exact order:

```text
issue,subject,company,response,product_area,status,request_type,justification
```
