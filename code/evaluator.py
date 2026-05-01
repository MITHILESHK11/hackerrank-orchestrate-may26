from __future__ import annotations

import argparse
import difflib
from pathlib import Path

import pandas as pd

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        default=str(Path(__file__).resolve().parents[1] / "support_tickets" / "sample_support_tickets.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "support_tickets" / "output.csv"),
    )
    return parser.parse_args()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _match_rows(sample: pd.DataFrame, output: pd.DataFrame) -> pd.DataFrame:
    keys = ["issue", "subject", "company"]
    for key in keys:
        if key not in sample.columns or key not in output.columns:
            raise ValueError(f"Missing key column: {key}")
    for frame in (sample, output):
        for key in keys:
            frame[key] = (
                frame[key]
                .astype(str)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .str.lower()
            )
    merged = sample.merge(
        output,
        how="left",
        on=keys,
        suffixes=("_exp", "_pred"),
    )
    return merged


def _accuracy(series_expected: pd.Series, series_predicted: pd.Series) -> float:
    if len(series_expected) == 0:
        return 0.0
    return float((series_expected.fillna("") == series_predicted.fillna("")).mean())


def _confusion_matrix(expected: pd.Series, predicted: pd.Series, labels: list[str]) -> list[list[int]]:
    matrix = []
    for gold in labels:
        row = []
        for pred in labels:
            row.append(int(((expected == gold) & (predicted == pred)).sum()))
        matrix.append(row)
    return matrix


def _print_confusion(title: str, labels: list[str], matrix: list[list[int]]) -> None:
    print(title)
    header = "gold\\pred".ljust(18) + "".join(label.ljust(18) for label in labels)
    print(header)
    for idx, gold in enumerate(labels):
        line = gold.ljust(18) + "".join(str(matrix[idx][j]).ljust(18) for j in range(len(labels)))
        print(line)


def main() -> None:
    args = _parse_args()
    sample = _normalize_columns(pd.read_csv(args.sample, dtype=str, keep_default_na=False))
    output = _normalize_columns(pd.read_csv(args.output, dtype=str, keep_default_na=False))
    joined = _match_rows(sample, output)
    for col in ["status_exp", "request_type_exp"]:
        if col in joined.columns:
            joined[col] = joined[col].astype(str).str.strip().str.lower()
    for col in ["status_pred", "request_type_pred"]:
        if col in joined.columns:
            joined[col] = joined[col].apply(
                lambda x: str(x).strip().lower() if pd.notna(x) and str(x).strip() else pd.NA
            )
    for col in ["product_area_exp", "product_area_pred", "response_exp", "response_pred"]:
        if col in joined.columns:
            if col.endswith("_exp"):
                joined[col] = joined[col].astype(str).str.strip()
            else:
                joined[col] = joined[col].apply(
                    lambda x: str(x).strip() if pd.notna(x) and str(x).strip() else pd.NA
                )

    matched = joined["status_pred"].notna() if "status_pred" in joined.columns else pd.Series([False] * len(joined))
    matched_count = int(matched.sum())
    print(f"Expected sample rows      : {len(joined)}")
    print(f"Matched output rows       : {matched_count}")
    if matched_count == 0:
        print("No matching rows found between sample and output CSV.")
        print("Run sample mode first: `sample` in interactive CLI or `python -m main --sample`.")
        return
    eval_df = joined[matched].copy()

    status_acc = _accuracy(eval_df["status_exp"], eval_df["status_pred"])
    request_type_acc = _accuracy(eval_df["request_type_exp"], eval_df["request_type_pred"])
    product_exact = _accuracy(eval_df["product_area_exp"], eval_df["product_area_pred"])
    if fuzz is not None:
        product_fuzzy = float(
            (
                eval_df.apply(
                    lambda r: fuzz.token_sort_ratio(
                        str(r.get("product_area_exp", "")),
                        str(r.get("product_area_pred", "")),
                    )
                    >= 80,
                    axis=1,
                ).mean()
            )
        )
    else:
        product_fuzzy = float(
            (
                eval_df.apply(
                    lambda r: difflib.SequenceMatcher(
                        None,
                        str(r.get("product_area_exp", "")).lower(),
                        str(r.get("product_area_pred", "")).lower(),
                    ).ratio()
                    >= 0.8,
                    axis=1,
                ).mean()
            )
        )

    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores: list[float] = []
        for _, row in eval_df.iterrows():
            exp = str(row.get("response_exp", ""))
            pred = str(row.get("response_pred", ""))
            rouge_scores.append(float(scorer.score(exp, pred)["rougeL"].fmeasure))
        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    else:
        scores = []
        for _, row in eval_df.iterrows():
            exp = str(row.get("response_exp", "")).lower()
            pred = str(row.get("response_pred", "")).lower()
            scores.append(difflib.SequenceMatcher(None, exp, pred).ratio())
        avg_rouge = sum(scores) / len(scores) if scores else 0.0

    print(f"Rows evaluated            : {len(eval_df)}")
    print(f"Status accuracy           : {status_acc:.2%}")
    print(f"Request type accuracy     : {request_type_acc:.2%}")
    print(f"Product area exact        : {product_exact:.2%}")
    print(f"Product area fuzzy >= 80  : {product_fuzzy:.2%}")
    print(f"Average ROUGE-L           : {avg_rouge:.4f}")

    status_labels = ["replied", "escalated"]
    status_matrix = _confusion_matrix(eval_df["status_exp"], eval_df["status_pred"], status_labels)
    _print_confusion("Status confusion matrix", status_labels, status_matrix)

    req_labels = ["product_issue", "feature_request", "bug", "invalid"]
    req_matrix = _confusion_matrix(eval_df["request_type_exp"], eval_df["request_type_pred"], req_labels)
    _print_confusion("Request type confusion matrix", req_labels, req_matrix)


if __name__ == "__main__":
    main()
