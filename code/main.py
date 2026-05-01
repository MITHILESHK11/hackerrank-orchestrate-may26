from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from agent import SupportTriageAgent
from corpus_loader import load_corpus
from llm_client import GeminiClient
from retriever import HybridRetriever


OUTPUT_COLUMNS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]


def _resolve_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "data").exists() and (cwd / "support_tickets").exists():
        return cwd
    module_root = Path(__file__).resolve().parents[1]
    if (module_root / "data").exists() and (module_root / "support_tickets").exists():
        return module_root
    return cwd


def _parse_args() -> argparse.Namespace:
    project_root = _resolve_project_root()
    default_input = project_root / "support_tickets" / "support_tickets.csv"
    default_output = project_root / "support_tickets" / "output.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(default_input),
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
    )
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--data-dir",
        default=str(project_root / "data"),
    )
    return parser.parse_args()


def _load_input(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    required = {"issue", "subject", "company"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame


def _atomic_write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_csv(
        tmp_path,
        index=False,
        columns=OUTPUT_COLUMNS,
        quoting=csv.QUOTE_ALL,
        encoding="utf-8",
    )
    os.replace(tmp_path, output_path)


def _create_agent(data_dir: Path) -> SupportTriageAgent:
    chunks = load_corpus(str(data_dir))
    retriever = HybridRetriever()
    retriever.build_index(chunks)
    llm_client = GeminiClient()
    return SupportTriageAgent(retriever=retriever, llm_client=llm_client)


def _run_batch(
    agent: SupportTriageAgent,
    input_path: Path,
    output_path: Path,
    limit: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    source_df = _load_input(input_path)
    if limit is not None and limit > 0:
        source_df = source_df.head(limit).copy()

    results: list[dict[str, str]] = []
    for _, row in tqdm(source_df.iterrows(), total=len(source_df), desc="Processing"):
        issue = row.get("issue", "")
        subject = row.get("subject", "")
        company = row.get("company", "")
        result = agent.process_ticket(issue=issue, subject=subject, company=company)
        item = {
            "issue": str(issue),
            "subject": str(subject),
            "company": str(company),
            "response": result["response"],
            "product_area": result["product_area"],
            "status": result["status"],
            "request_type": result["request_type"],
            "justification": result["justification"],
        }
        results.append(item)
        if verbose:
            print(
                f"[{item['status']}] {item['company'] or 'None'} | "
                f"{item['request_type']} | {item['product_area']}"
            )

    output_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    _atomic_write_csv(output_df, output_path)
    return output_df


def _print_result(ticket_result: dict[str, str]) -> None:
    print("\n" + "=" * 72)
    print(f"status       : {ticket_result['status']}")
    print(f"request_type : {ticket_result['request_type']}")
    print(f"product_area : {ticket_result['product_area']}")
    print("-" * 72)
    print("response:")
    print(ticket_result["response"])
    print("-" * 72)
    print("justification:")
    print(ticket_result["justification"])
    print("=" * 72 + "\n")


def _interactive_mode(
    agent: SupportTriageAgent,
    default_input: Path,
    default_output: Path,
) -> None:
    sample_output = default_output.parent / "output.sample.csv"
    print("\nSupport Triage CLI")
    print("Type one of: triage, sample, full, custom, eval, help, exit")

    while True:
        cmd = input("triage-cli> ").strip().lower()
        if cmd in {"exit", "quit", "q"}:
            print("Exiting CLI.")
            return
        if cmd in {"help", "h"}:
            print("triage : run one interactive ticket")
            print(f"sample : run sample_support_tickets.csv -> {sample_output.name}")
            print(f"full   : run support_tickets.csv -> {default_output.name}")
            print("custom : batch run custom input/output paths")
            print("eval   : evaluate sample_support_tickets.csv against sample output file")
            print("exit   : quit")
            continue
        if cmd == "triage":
            issue = input("Issue: ").strip()
            subject = input("Subject (optional): ").strip()
            company = input("Company (HackerRank/Claude/Visa/None): ").strip()
            result = agent.process_ticket(issue=issue, subject=subject, company=company)
            _print_result(result)
            continue
        if cmd == "sample":
            output_df = _run_batch(
                agent=agent,
                input_path=default_input.parent / "sample_support_tickets.csv",
                output_path=sample_output,
                verbose=True,
            )
            print(
                f"Done. Wrote {len(output_df)} rows to {sample_output}"
            )
            continue
        if cmd == "full":
            output_df = _run_batch(
                agent=agent,
                input_path=default_input,
                output_path=default_output,
                verbose=True,
            )
            print(
                f"Done. Wrote {len(output_df)} rows to {default_output}"
            )
            continue
        if cmd == "custom":
            in_path = Path(input("Input CSV path: ").strip().strip('"'))
            out_path = Path(input("Output CSV path: ").strip().strip('"'))
            limit_raw = input("Limit rows (empty for all): ").strip()
            limit = int(limit_raw) if limit_raw else None
            output_df = _run_batch(
                agent=agent,
                input_path=in_path,
                output_path=out_path,
                limit=limit,
                verbose=True,
            )
            print(f"Done. Wrote {len(output_df)} rows to {out_path}")
            continue
        if cmd == "eval":
            sample_path = default_input.parent / "sample_support_tickets.csv"
            eval_output = sample_output if sample_output.exists() else default_output
            cmdline = [
                sys.executable,
                "-m",
                "evaluator",
                "--sample",
                str(sample_path),
                "--output",
                str(eval_output),
            ]
            subprocess.run(cmdline, check=False)
            continue
        if cmd:
            print("Unknown command. Type help for options.")


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    load_dotenv()
    args = _parse_args()

    input_path = Path(args.input)
    if args.sample:
        input_path = input_path.parent / "sample_support_tickets.csv"
    output_path = Path(args.output)
    if args.sample:
        default_full_output = _resolve_project_root() / "support_tickets" / "output.csv"
        if output_path.resolve() == default_full_output.resolve():
            output_path = output_path.parent / "output.sample.csv"
    data_dir = Path(args.data_dir)
    agent = _create_agent(data_dir)

    if args.interactive:
        _interactive_mode(
            agent=agent,
            default_input=Path(args.input),
            default_output=output_path,
        )
        return

    output_df = _run_batch(
        agent=agent,
        input_path=input_path,
        output_path=output_path,
        limit=args.limit,
        verbose=args.verbose,
    )

    replied = int((output_df["status"] == "replied").sum()) if not output_df.empty else 0
    escalated = int((output_df["status"] == "escalated").sum()) if not output_df.empty else 0
    invalid = int((output_df["request_type"] == "invalid").sum()) if not output_df.empty else 0
    print(
        f"Processed {len(output_df)} rows: "
        f"{replied} replied, {escalated} escalated, {invalid} invalid"
    )


if __name__ == "__main__":
    main()
