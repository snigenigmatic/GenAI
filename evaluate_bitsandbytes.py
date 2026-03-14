"""
Evaluation Script for Epistemic Uncertainty RAG Research
Using BitsAndBytes quantization (FP16 vs INT4)
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

from rag_bitsandbytes import RAGSystem, BitsAndBytesLLM, Retriever, UncertaintyEstimator


class ExperimentRunner:

    def __init__(
        self,
        dataset_path: str,
        documents: List[str],
        model_name: str = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        quantization: str = "fp16",
    ):
        self.documents     = documents
        self.model_name    = model_name
        self.quantization  = quantization

        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)

        print(f"Loaded {len(self.dataset)} questions from {dataset_path}")

        # Build RAG system ONCE — shared across all experiments
        print("\nInitialising shared RAG system …")
        self.rag = RAGSystem(
            documents=documents,
            model_name=model_name,
            quantization=quantization,
        )

    def run_experiment(
        self,
        exp_id: str,
        use_retrieval: bool,
        method: str,
        max_questions: Optional[int] = None,
    ) -> List[Dict]:
        dataset = self.dataset[:max_questions] if max_questions else self.dataset

        print(f"\n{'='*70}")
        print(f"Experiment {exp_id}  |  retrieval={use_retrieval}  |  method={method}  |  n={len(dataset)}")
        print(f"{'='*70}\n")

        results = []

        for item in tqdm(dataset, desc=f"Exp {exp_id}"):
            question     = item["question"]
            ground_truth = item.get("answer", "")
            category     = item.get("category", "unknown")

            # weak_evidence rows have "[" as a placeholder — treat as no ground truth
            if ground_truth.strip() == "[":
                ground_truth = ""

            try:
                result = self.rag.query(
                    question,
                    k=5,
                    use_retrieval=use_retrieval,
                    return_details=False,
                )

                answer = result["answer"]

                if method == "Entropy":
                    uncertainty = min(1.0, result["uncertainty"]["token_entropy"] / 2.0)
                elif method == "SemanticVar":
                    uncertainty = result["uncertainty"]["semantic_variance"]
                else:
                    uncertainty = result["uncertainty"]["self_eval"]

                decision      = result["decision"]
                is_correct    = self._check_correctness(answer, ground_truth)
                should_abstain = category in ("weak_evidence", "conflicting_evidence")

                results.append({
                    "exp_id":           exp_id,
                    "question_id":      item.get("id", -1),
                    "question":         question,
                    "ground_truth":     ground_truth,
                    "answer":           answer,
                    "category":         category,
                    "is_correct":       is_correct,
                    "uncertainty":      uncertainty,
                    "token_entropy":    result["uncertainty"]["token_entropy"],
                    "semantic_variance": result["uncertainty"]["semantic_variance"],
                    "self_eval":        result["uncertainty"]["self_eval"],
                    "combined":         result["uncertainty"]["combined"],
                    "decision":         decision,
                    "should_abstain":   should_abstain,
                    "quantization":     self.quantization,
                    "retrieval":        use_retrieval,
                    "method":           method,
                })

            except Exception as e:
                print(f"  Error on Q{item.get('id', '?')}: {e}")
                continue

        return results

    @staticmethod
    def _check_correctness(answer: str, ground_truth: str) -> bool:
        if not ground_truth:
            return False

        a  = answer.lower().strip()
        gt = ground_truth.lower().strip()

        if a == gt:
            return True
        if gt in a or a in gt:
            return True

        a_tok  = set(a.split())
        gt_tok = set(gt.split())
        if a_tok and gt_tok:
            overlap = len(a_tok & gt_tok) / min(len(a_tok), len(gt_tok))
            if overlap >= 0.5:
                return True

        return False

    def compute_metrics(self, results: List[Dict]) -> Dict:
        if not results:
            return {}

        correct            = sum(1 for r in results if r["is_correct"])
        accuracy           = correct / len(results)

        overconfident      = sum(
            1 for r in results if not r["is_correct"] and r["uncertainty"] < 0.4
        )
        overconfidence_rate = overconfident / len(results)

        abstained = [r for r in results if r["decision"] == "abstain"]
        if abstained:
            correct_abstentions = sum(1 for r in abstained if r["should_abstain"])
            abstention_precision = correct_abstentions / len(abstained)
        else:
            abstention_precision = 0.0

        ece = self._compute_ece(results)

        try:
            y_true  = [1 if r["is_correct"] else 0 for r in results]
            y_score = [1 - r["uncertainty"] for r in results]
            auroc   = roc_auc_score(y_true, y_score)
        except Exception:
            auroc = 0.5

        danger_zone = sum(
            1 for r in results
            if r["token_entropy"] < 0.35 and r["semantic_variance"] > 0.4
        )

        return {
            "accuracy":            accuracy,
            "overconfidence_rate": overconfidence_rate,
            "abstention_precision": abstention_precision,
            "ece":                 ece,
            "auroc":               auroc,
            "danger_zone_pct":     danger_zone / len(results),
            "num_samples":         len(results),
            "num_correct":         correct,
            "num_abstained":       len(abstained),
            "num_danger_zone":     danger_zone,
        }

    @staticmethod
    def _compute_ece(results: List[Dict], n_bins: int = 5) -> float:
        confidences = [1 - r["uncertainty"] for r in results]
        correctness = [1 if r["is_correct"] else 0 for r in results]

        bins        = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bins[:-1]) - 1

        ece   = 0.0
        total = len(results)

        for b in range(n_bins):
            in_bin = [i for i, idx in enumerate(bin_indices) if idx == b]
            if in_bin:
                bin_conf = np.mean([confidences[i] for i in in_bin])
                bin_acc  = np.mean([correctness[i] for i in in_bin])
                ece     += abs(bin_conf - bin_acc) * (len(in_bin) / total)

        return ece

    def run_all_experiments(
        self,
        quick: bool = False,
    ):
        """
        Full 12-experiment grid.
        quick=True runs E1/E2/E4 on 10 questions each — for local smoke-testing.
        """
        all_experiments = [
            ("E1",  True,  "Entropy"),
            ("E2",  True,  "SemanticVar"),
            ("E3",  True,  "SelfEval"),
            ("E4",  False, "Entropy"),
            ("E5",  False, "SemanticVar"),
            ("E6",  False, "SelfEval"),
        ]

        # In quick mode run only the two meaningful methods with retrieval on
        if quick:
            experiments   = [e for e in all_experiments if e[0] in ("E1", "E2")]
            max_questions = 10
            print("QUICK MODE — 2 experiments × 10 questions")
        else:
            experiments   = all_experiments
            max_questions = None

        all_results     = []
        summary_metrics = []

        for exp_id, retrieval, method in experiments:
            exp_results = self.run_experiment(
                exp_id, retrieval, method, max_questions=max_questions
            )
            metrics             = self.compute_metrics(exp_results)
            metrics["exp_id"]   = exp_id
            metrics["quantization"] = self.quantization
            metrics["retrieval"] = "On" if retrieval else "Off"
            metrics["method"]   = method

            summary_metrics.append(metrics)
            all_results.extend(exp_results)
            self._save_results(exp_id, exp_results, metrics)

        summary_df      = pd.DataFrame(summary_metrics)
        all_results_df  = pd.DataFrame(all_results)
        return summary_df, all_results_df

    def _save_results(self, exp_id: str, results: List[Dict], metrics: Dict):
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / f"{exp_id}_results.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(output_dir / f"{exp_id}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n✓ Saved {exp_id}")

    def generate_plots(self, summary_df: pd.DataFrame):
        output_dir = Path("results/plots")
        output_dir.mkdir(exist_ok=True, parents=True)
        sns.set_style("whitegrid")

        for metric, title, ylabel in [
            ("accuracy",        "Answer Accuracy by Method",          "Accuracy"),
            ("ece",             "Expected Calibration Error",          "ECE (lower is better)"),
            ("danger_zone_pct", "Danger Zone %",                      "Fraction of samples"),
            ("auroc",           "AUROC (uncertainty vs correctness)",  "AUROC"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            summary_df.plot(x="method", y=metric, kind="bar", ax=ax)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Uncertainty method")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(output_dir / f"{metric}.png", dpi=300)
            plt.close()

        print(f"✓ Plots saved to {output_dir}")

    def generate_latex_table(self, summary_df: pd.DataFrame) -> str:
        latex  = "\\begin{table}[h]\n\\centering\n"
        latex += "\\caption{Experimental Results}\n\\label{tab:results}\n"
        latex += "\\begin{tabular}{lcccccc}\n\\hline\n"
        latex += "Exp & Retr & Method & Acc & ECE & AUROC & AbsPrec \\\\\n\\hline\n"

        for _, row in summary_df.iterrows():
            latex += (
                f"{row['exp_id']} & {row['retrieval']} & {row['method']} & "
                f"{row['accuracy']:.3f} & {row['ece']:.3f} & "
                f"{row['auroc']:.3f} & {row['abstention_precision']:.3f} \\\\\n"
            )

        latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
        return latex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="benchmark_60_hf.json")
    parser.add_argument("--model",    default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    parser.add_argument("--quant",    default="fp16", choices=["fp16", "int4"])
    parser.add_argument("--documents", default=None)
    parser.add_argument("--quick",    action="store_true",
                        help="Run 2 experiments × 10 questions for local testing")
    args = parser.parse_args()

    if args.documents:
        with open(args.documents, encoding="utf-8") as f:
            documents = [l.strip() for l in f if l.strip()]
    else:
        documents = [
            "The Eiffel Tower is located in Paris, France.",
            "Paris is the capital city of France.",
            "France is a country in Western Europe.",
            "Machine learning is a subfield of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Python is a popular programming language for AI.",
            "The Earth revolves around the Sun.",
            "Water boils at 100 degrees Celsius at sea level.",
            "DNA carries genetic information.",
            "Photosynthesis uses sunlight, carbon dioxide, and water.",
        ]

    runner = ExperimentRunner(
        dataset_path=args.dataset,
        documents=documents,
        model_name=args.model,
        quantization=args.quant,
    )

    summary_df, all_results_df = runner.run_all_experiments(quick=args.quick)

    output_dir = Path("results")
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    all_results_df.to_csv(output_dir / "all_results.csv", index=False)

    runner.generate_plots(summary_df)

    latex_table = runner.generate_latex_table(summary_df)
    with open(output_dir / "results_table.tex", "w") as f:
        f.write(latex_table)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    print(f"\n✓ Results saved to {output_dir}/")


if __name__ == "__main__":
    main()