"""
================================================================================
TRIPWIRE ENTITY RESOLUTION EVALUATOR MODULE
================================================================================

Evaluates current EntityResolver algorithm against ground-truth labeled entity pairs.
Computes Precision, Recall, F1, Confusion Matrix, Method Rates, Confidence Stats,
and sweeps fuzzy matching thresholds to recommend optimal F1 configuration.

Usage:
  python -m evaluation.resolution.evaluator [--dataset PATH] [--output PATH]
================================================================================
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from rapidfuzz import fuzz
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.resolution.entity_resolver import EntityResolver, normalize_entity_name

from evaluation.resolution.dataset import load_resolution_dataset
from evaluation.resolution.metrics import (
    compute_resolution_evaluation_metrics,
    run_threshold_sweep,
)

logger = structlog.get_logger(__name__)
console = Console()


class EntityResolutionEvaluator:
    """Evaluator harness testing EntityResolver accuracy and threshold performance."""

    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset = load_resolution_dataset(dataset_path)
        self.resolver = EntityResolver()

    def evaluate(self, fuzzy_threshold: float = 85.0) -> Dict[str, Any]:
        """
        Evaluate current EntityResolver algorithm on dataset pairs.
        Returns full evaluation metrics and threshold curve analysis.
        """
        predictions: List[Dict[str, Any]] = []
        pair_scores: List[Dict[str, Any]] = []

        for pair in self.dataset:
            a_norm = normalize_entity_name(pair.entity_a)
            b_norm = normalize_entity_name(pair.entity_b)

            # 1. Check exact / normalized string match
            is_exact = pair.entity_a.strip().lower() == pair.entity_b.strip().lower()
            is_norm = a_norm == b_norm and len(a_norm) > 0

            # 2. Check pairwise RapidFuzz token_sort_ratio
            raw_fuzzy_score = float(fuzz.token_sort_ratio(a_norm, b_norm))
            conf_score = raw_fuzzy_score / 100.0

            # 3. Determine resolution prediction and method
            if is_exact:
                method = "exact"
                pred_match = True
                conf_score = 1.0
            elif is_norm:
                method = "normalized"
                pred_match = True
                conf_score = 0.95
            elif raw_fuzzy_score >= fuzzy_threshold:
                method = "fuzzy"
                pred_match = True
            else:
                method = "unresolved"
                pred_match = False

            # Check if canonical seed resolution maps them to same canonical entity
            canon_a, m_a, c_a = self.resolver.resolve(pair.entity_a)
            canon_b, m_b, c_b = self.resolver.resolve(pair.entity_b)
            if canon_a and canon_b and canon_a == canon_b:
                pred_match = True
                method = m_a if m_a in ("exact", "normalized") else method
                conf_score = max(conf_score, c_a)

            prediction_entry = {
                "id": pair.id,
                "entity_a": pair.entity_a,
                "entity_b": pair.entity_b,
                "expected_match": pair.expected_match,
                "predicted_match": pred_match,
                "case_category": pair.case_category,
                "method_used": method,
                "confidence_score": conf_score,
            }
            predictions.append(prediction_entry)

            pair_scores.append({
                "id": pair.id,
                "expected_match": pair.expected_match,
                "raw_fuzzy_score": raw_fuzzy_score,
                "is_exact_or_norm": is_exact or is_norm,
            })

        # Calculate primary evaluation metrics
        metrics = compute_resolution_evaluation_metrics(predictions)

        # Run threshold analysis curve (50.0 to 95.0)
        threshold_analysis = run_threshold_sweep(pair_scores)

        # Category breakdown
        category_breakdown = self._compute_category_breakdown(predictions)

        # Compute empirical decision model breakdown
        decision_counts = {"MATCH": 0, "REVIEW": 0, "NEW": 0}
        for p in predictions:
            method = p["method_used"]
            conf = p["confidence_score"]
            if method in ("exact", "normalized") or conf >= (fuzzy_threshold / 100.0):
                dec = "MATCH"
            elif conf >= 0.65:
                dec = "REVIEW"
            else:
                dec = "NEW"
            p["decision"] = dec
            decision_counts[dec] += 1

        total_pairs = len(self.dataset) or 1
        decision_model = {
            "counts": decision_counts,
            "match_threshold": fuzzy_threshold,
            "review_threshold": 65.0,
            "match_rate": round(decision_counts["MATCH"] / total_pairs, 4),
            "review_rate": round(decision_counts["REVIEW"] / total_pairs, 4),
            "new_rate": round(decision_counts["NEW"] / total_pairs, 4),
        }

        report = {
            "evaluator": "EntityResolutionEvaluator",
            "active_fuzzy_threshold": fuzzy_threshold,
            "total_pairs_evaluated": len(self.dataset),
            "metrics": metrics,
            "decision_model": decision_model,
            "threshold_analysis": threshold_analysis,
            "category_breakdown": category_breakdown,
            "per_pair_predictions": predictions,
        }
        return report

    def _compute_category_breakdown(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute accuracy per edge-case category."""
        categories: Dict[str, Dict[str, int]] = {}
        for p in predictions:
            cat = p["case_category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "correct": 0}
            categories[cat]["total"] += 1
            if p["expected_match"] == p["predicted_match"]:
                categories[cat]["correct"] += 1

        breakdown = {}
        for cat, counts in categories.items():
            acc = round(counts["correct"] / counts["total"], 4) if counts["total"] > 0 else 0.0
            breakdown[cat] = {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": acc,
            }
        return breakdown


def print_resolution_evaluation_summary(report: Dict[str, Any]) -> None:
    """Print Rich terminal summary tables of entity resolution evaluation."""
    m = report["metrics"]
    cls = m["classification"]
    cm = m["confusion_matrix"]
    rates = m["method_rates"]
    ta = report["threshold_analysis"]

    title = f"[bold cyan]ENTITY RESOLUTION EVALUATION REPORT[/bold cyan] (Active Threshold: [yellow]{report['active_fuzzy_threshold']}[/yellow])"
    console.print("\n")
    console.print(
        Panel(
            f"Evaluated Pairs: [bold white]{report['total_pairs_evaluated']}[/bold white] | "
            f"Precision: [bold green]{round(cls['precision']*100, 2)}%[/bold green] | "
            f"Recall: [bold green]{round(cls['recall']*100, 2)}%[/bold green] | "
            f"F1 Score: [bold magenta]{round(cls['f1_score']*100, 2)}%[/bold magenta]",
            title=title,
            border_style="cyan",
        )
    )

    # 1. Classification & Confusion Matrix Table
    table_cls = Table(title="1. Resolution Accuracy & Confusion Matrix", show_header=True, header_style="bold magenta")
    table_cls.add_column("Metric Name", style="cyan", width=28)
    table_cls.add_column("Value / Count", style="bold green", justify="right")

    table_cls.add_row("Precision", f"{round(cls['precision'] * 100, 2)}%")
    table_cls.add_row("Recall", f"{round(cls['recall'] * 100, 2)}%")
    table_cls.add_row("F1 Score", f"{round(cls['f1_score'] * 100, 2)}%")
    table_cls.add_row("Accuracy", f"{round(cls['accuracy'] * 100, 2)}%")
    table_cls.add_row("True Positives (TP)", str(cm["true_positives"]))
    table_cls.add_row("False Positives (FP)", str(cm["false_positives"]))
    table_cls.add_row("False Negatives (FN)", str(cm["false_negatives"]))
    table_cls.add_row("True Negatives (TN)", str(cm["true_negatives"]))
    console.print(table_cls)

    # 2. Method Match Rates Table
    table_rates = Table(title="2. Match Method Breakdown", show_header=True, header_style="bold cyan")
    table_rates.add_column("Match Method", style="white")
    table_rates.add_column("Rate", style="bold yellow", justify="right")
    table_rates.add_row("Exact Match Rate", f"{round(rates['exact_match_rate'] * 100, 2)}%")
    table_rates.add_row("Normalized Match Rate", f"{round(rates['normalized_match_rate'] * 100, 2)}%")
    table_rates.add_row("Fuzzy Match Rate", f"{round(rates['fuzzy_match_rate'] * 100, 2)}%")
    table_rates.add_row("Unresolved Rate", f"{round(rates['unresolved_rate'] * 100, 2)}%")
    console.print(table_rates)

    # 3. Threshold Analysis Curve Table
    table_thresh = Table(title="3. Fuzzy Matching Threshold Sweep Analysis", show_header=True, header_style="bold yellow")
    table_thresh.add_column("Threshold", style="cyan", justify="right")
    table_thresh.add_column("Precision", justify="right")
    table_thresh.add_column("Recall", justify="right")
    table_thresh.add_column("F1 Score", justify="right")
    table_thresh.add_column("TP / FP / FN", justify="right")

    for t_entry in ta["threshold_curve"]:
        t_val = t_entry["threshold"]
        is_rec = t_val == ta["recommended_threshold"]
        style = "bold green" if is_rec else "white"
        rec_tag = " (Recommended Max F1)" if is_rec else ""
        table_thresh.add_row(
            f"[{style}]{t_val}{rec_tag}[/{style}]",
            f"{round(t_entry['precision'] * 100, 1)}%",
            f"{round(t_entry['recall'] * 100, 1)}%",
            f"[{style}]{round(t_entry['f1_score'] * 100, 1)}%[/{style}]",
            f"{t_entry['tp']} / {t_entry['fp']} / {t_entry['fn']}",
        )
    console.print(table_thresh)
    console.print(f"[bold green]Optimal Recommended Threshold based on Measured Max F1 Score:[/bold green] [yellow]{ta['recommended_threshold']}[/yellow] (F1 = {round(ta['best_f1_score']*100, 2)}%)\n")

    # 4. Confidence Decision Model Breakdown Table
    if "decision_model" in report:
        dm = report["decision_model"]
        table_dm = Table(title="4. Confidence-Based Decision Routing (Empirical Policy)", show_header=True, header_style="bold green")
        table_dm.add_column("Decision Class", style="cyan", width=24)
        table_dm.add_column("Condition / Rule", style="white")
        table_dm.add_column("Count", style="bold yellow", justify="right")
        table_dm.add_column("Share", style="bold green", justify="right")

        table_dm.add_row(
            "[bold green]MATCH[/bold green]",
            f"Exact/Norm or Fuzzy >= {dm['match_threshold']}% (High Precision Auto-merge)",
            str(dm["counts"]["MATCH"]),
            f"{round(dm['match_rate'] * 100, 1)}%",
        )
        table_dm.add_row(
            "[bold yellow]REVIEW[/bold yellow]",
            f"Fuzzy [{dm['review_threshold']}%, {dm['match_threshold']}%) (Ambiguous -> Human Queue)",
            str(dm["counts"]["REVIEW"]),
            f"{round(dm['review_rate'] * 100, 1)}%",
        )
        table_dm.add_row(
            "[bold blue]NEW[/bold blue]",
            f"Fuzzy < {dm['review_threshold']}% (Distinct Entity Creation)",
            str(dm["counts"]["NEW"]),
            f"{round(dm['new_rate'] * 100, 1)}%",
        )
        console.print(table_dm)
        console.print("\n")


def main():
    """CLI entry point for python -m evaluation.resolution.evaluator."""
    parser = argparse.ArgumentParser(
        description="Tripwire Entity Resolution Engine Evaluation Harness"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional path to custom ground-truth resolution pairs JSON dataset",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=85.0,
        help="Fuzzy matching threshold to evaluate (default: 85.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/reports/resolution_eval.json",
        help="Target report JSON destination path",
    )

    args = parser.parse_args()

    evaluator = EntityResolutionEvaluator(dataset_path=Path(args.dataset) if args.dataset else None)
    report = evaluator.evaluate(fuzzy_threshold=args.threshold)

    print_resolution_evaluation_summary(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved entity resolution evaluation report JSON", path=str(out_path))


if __name__ == "__main__":
    main()
