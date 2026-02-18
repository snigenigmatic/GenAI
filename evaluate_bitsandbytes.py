"""
Evaluation Script for Epistemic Uncertainty RAG Research
Using BitsAndBytes quantization (FP16 vs INT4)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

from rag_bitsandbytes import RAGSystem


class ExperimentRunner:
    """Runs experiments across different configurations"""
    
    def __init__(
        self,
        dataset_path: str,
        documents: List[str],
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    ):
        """
        Initialize experiment runner
        
        Args:
            dataset_path: Path to benchmark_60.json
            documents: Knowledge base documents
            model_name: HuggingFace model name
        """
        self.dataset_path = dataset_path
        self.documents = documents
        self.model_name = model_name
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        
        print(f"Loaded {len(self.dataset)} questions")
    
    def run_experiment(
        self,
        exp_id: str,
        quantization: str,  # "fp16" or "int4"
        use_retrieval: bool,
        method: str  # "Entropy", "SemanticVar", or "SelfEval"
    ) -> List[Dict]:
        """
        Run a single experiment configuration
        
        Args:
            exp_id: Experiment ID (E1-E12)
            quantization: "fp16" or "int4"
            use_retrieval: Whether to use RAG
            method: Which uncertainty method to use for decision
        
        Returns:
            List of results per question
        """
        print(f"\n{'='*70}")
        print(f"Running Experiment {exp_id}")
        print(f"Quantization: {quantization}, Retrieval: {use_retrieval}, Method: {method}")
        print(f"{'='*70}\n")
        
        # Initialize RAG system
        rag = RAGSystem(
            documents=self.documents,
            model_name=self.model_name,
            quantization=quantization
        )
        
        results = []
        
        # Run on each question
        for item in tqdm(self.dataset, desc=f"Exp {exp_id}"):
            question = item['question']
            ground_truth = item.get('answer', '')
            category = item.get('category', 'unknown')
            
            try:
                # Query system
                result = rag.query(
                    question,
                    k=5,
                    use_retrieval=use_retrieval,
                    return_details=False
                )
                
                answer = result['answer']
                
                # Extract uncertainty based on method
                if method == "Entropy":
                    uncertainty = result['uncertainty']['token_entropy']
                    # Normalize to 0-1 range
                    uncertainty = min(1.0, uncertainty / 2.0)
                elif method == "SemanticVar":
                    uncertainty = result['uncertainty']['semantic_variance']
                else:  # SelfEval
                    uncertainty = result['uncertainty']['self_eval']
                
                decision = result['decision']
                
                # Check correctness (simple string matching)
                is_correct = self._check_correctness(answer, ground_truth)
                
                # Determine if should abstain
                should_abstain = (category in ['weak_evidence', 'conflicting_evidence'])
                
                # Record result
                results.append({
                    'exp_id': exp_id,
                    'question_id': item.get('id', -1),
                    'question': question,
                    'ground_truth': ground_truth,
                    'answer': answer,
                    'category': category,
                    'is_correct': is_correct,
                    'uncertainty': uncertainty,
                    'token_entropy': result['uncertainty']['token_entropy'],
                    'semantic_variance': result['uncertainty']['semantic_variance'],
                    'self_eval': result['uncertainty']['self_eval'],
                    'combined': result['uncertainty']['combined'],
                    'decision': decision,
                    'should_abstain': should_abstain,
                    'quantization': quantization,
                    'retrieval': use_retrieval,
                    'method': method
                })
                
            except Exception as e:
                print(f"Error on question {item.get('id', '?')}: {e}")
                continue
        
        return results
    
    def _check_correctness(self, answer: str, ground_truth: str) -> bool:
        """Check if answer is correct"""
        if not ground_truth:
            return False  # No ground truth to compare
        
        answer_clean = answer.lower().strip()
        gt_clean = ground_truth.lower().strip()
        
        # Exact match
        if answer_clean == gt_clean:
            return True
        
        # Substring match
        if gt_clean in answer_clean or answer_clean in gt_clean:
            return True
        
        # Token overlap
        answer_tokens = set(answer_clean.split())
        gt_tokens = set(gt_clean.split())
        
        if len(answer_tokens) > 0 and len(gt_tokens) > 0:
            overlap = len(answer_tokens & gt_tokens)
            overlap_ratio = overlap / min(len(answer_tokens), len(gt_tokens))
            if overlap_ratio >= 0.5:
                return True
        
        return False
    
    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute evaluation metrics"""
        if not results:
            return {}
        
        # Accuracy
        correct = sum(1 for r in results if r['is_correct'])
        accuracy = correct / len(results)
        
        # Overconfidence rate
        overconfident = sum(
            1 for r in results
            if not r['is_correct'] and r['uncertainty'] < 0.4
        )
        overconfidence_rate = overconfident / len(results)
        
        # Abstention metrics
        abstained = [r for r in results if r['decision'] == 'abstain']
        if abstained:
            correct_abstentions = sum(1 for r in abstained if r['should_abstain'])
            abstention_precision = correct_abstentions / len(abstained)
        else:
            abstention_precision = 0.0
        
        # ECE (Expected Calibration Error)
        ece = self._compute_ece(results)
        
        # AUROC
        try:
            y_true = [1 if r['is_correct'] else 0 for r in results]
            y_score = [1 - r['uncertainty'] for r in results]
            auroc = roc_auc_score(y_true, y_score)
        except:
            auroc = 0.5
        
        # Additional: Danger zone percentage
        danger_zone = sum(
            1 for r in results
            if r['token_entropy'] < 0.35 and r['semantic_variance'] > 0.4
        )
        danger_zone_pct = danger_zone / len(results)
        
        return {
            'accuracy': accuracy,
            'overconfidence_rate': overconfidence_rate,
            'abstention_precision': abstention_precision,
            'ece': ece,
            'auroc': auroc,
            'danger_zone_pct': danger_zone_pct,
            'num_samples': len(results),
            'num_correct': correct,
            'num_abstained': len(abstained),
            'num_danger_zone': danger_zone
        }
    
    def _compute_ece(self, results: List[Dict], n_bins: int = 5) -> float:
        """Compute Expected Calibration Error"""
        confidences = [1 - r['uncertainty'] for r in results]
        correctness = [1 if r['is_correct'] else 0 for r in results]
        
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bins[:-1]) - 1
        
        ece = 0.0
        total = len(results)
        
        for b in range(n_bins):
            in_bin = [i for i, idx in enumerate(bin_indices) if idx == b]
            
            if in_bin:
                bin_confidence = np.mean([confidences[i] for i in in_bin])
                bin_accuracy = np.mean([correctness[i] for i in in_bin])
                bin_weight = len(in_bin) / total
                
                ece += abs(bin_confidence - bin_accuracy) * bin_weight
        
        return ece
    
    def run_all_experiments(self) -> pd.DataFrame:
        """
        Run all 12 experiments
        
        Grid:
        - 2 quantizations (FP16, INT4)
        - 2 retrieval modes (On, Off)
        - 3 uncertainty methods (Entropy, SemanticVar, SelfEval)
        
        Returns:
            Summary DataFrame
        """
        experiments = [
            # FP16 + Retrieval On
            ('E1', 'fp16', True, 'Entropy'),
            ('E2', 'fp16', True, 'SemanticVar'),
            ('E3', 'fp16', True, 'SelfEval'),
            # INT4 + Retrieval On
            ('E4', 'int4', True, 'Entropy'),
            ('E5', 'int4', True, 'SemanticVar'),
            ('E6', 'int4', True, 'SelfEval'),
            # FP16 + Retrieval Off
            ('E7', 'fp16', False, 'Entropy'),
            ('E8', 'fp16', False, 'SemanticVar'),
            ('E9', 'fp16', False, 'SelfEval'),
            # INT4 + Retrieval Off
            ('E10', 'int4', False, 'Entropy'),
            ('E11', 'int4', False, 'SemanticVar'),
            ('E12', 'int4', False, 'SelfEval'),
        ]
        
        all_results = []
        summary_metrics = []
        
        for exp_id, quant, retrieval, method in experiments:
            # Run experiment
            exp_results = self.run_experiment(exp_id, quant, retrieval, method)
            
            # Compute metrics
            metrics = self.compute_metrics(exp_results)
            metrics['exp_id'] = exp_id
            metrics['quantization'] = quant
            metrics['retrieval'] = 'On' if retrieval else 'Off'
            metrics['method'] = method
            
            summary_metrics.append(metrics)
            all_results.extend(exp_results)
            
            # Save intermediate results
            self._save_results(exp_id, exp_results, metrics)
        
        # Convert to DataFrames
        summary_df = pd.DataFrame(summary_metrics)
        all_results_df = pd.DataFrame(all_results)
        
        return summary_df, all_results_df
    
    def _save_results(self, exp_id: str, results: List[Dict], metrics: Dict):
        """Save results for single experiment"""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        with open(output_dir / f"{exp_id}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save metrics
        with open(output_dir / f"{exp_id}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Saved results for {exp_id}")
    
    def generate_plots(self, summary_df: pd.DataFrame):
        """Generate plots for paper"""
        output_dir = Path("results/plots")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        sns.set_style("whitegrid")
        
        # Plot 1: Accuracy by Quantization
        fig, ax = plt.subplots(figsize=(10, 6))
        summary_df.pivot(
            index='method',
            columns='quantization',
            values='accuracy'
        ).plot(kind='bar', ax=ax)
        ax.set_title('Answer Accuracy: FP16 vs INT4')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Uncertainty Method')
        ax.legend(title='Quantization')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_by_quantization.png', dpi=300)
        plt.close()
        
        # Plot 2: ECE Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        summary_df.pivot(
            index='method',
            columns='quantization',
            values='ece'
        ).plot(kind='bar', ax=ax)
        ax.set_title('Expected Calibration Error: FP16 vs INT4')
        ax.set_ylabel('ECE (lower is better)')
        ax.set_xlabel('Uncertainty Method')
        ax.legend(title='Quantization')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'ece_comparison.png', dpi=300)
        plt.close()
        
        # Plot 3: Danger Zone Percentage
        fig, ax = plt.subplots(figsize=(10, 6))
        summary_df.pivot(
            index='method',
            columns='quantization',
            values='danger_zone_pct'
        ).plot(kind='bar', ax=ax)
        ax.set_title('Overconfidence Rate (Danger Zone)')
        ax.set_ylabel('Percentage of Samples')
        ax.set_xlabel('Uncertainty Method')
        ax.legend(title='Quantization')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'danger_zone.png', dpi=300)
        plt.close()
        
        print(f"✓ Plots saved to {output_dir}")
    
    def generate_latex_table(self, summary_df: pd.DataFrame) -> str:
        """Generate LaTeX table"""
        latex = "\\begin{table}[h]\n"
        latex += "\\centering\n"
        latex += "\\caption{Experimental Results: FP16 vs INT4 Quantization}\n"
        latex += "\\label{tab:results}\n"
        latex += "\\begin{tabular}{lcccccc}\n"
        latex += "\\hline\n"
        latex += "Exp & Quant & Retr & Method & Acc & ECE & AUROC \\\\\n"
        latex += "\\hline\n"
        
        for _, row in summary_df.iterrows():
            latex += f"{row['exp_id']} & "
            latex += f"{row['quantization']} & "
            latex += f"{row['retrieval']} & "
            latex += f"{row['method']} & "
            latex += f"{row['accuracy']:.3f} & "
            latex += f"{row['ece']:.3f} & "
            latex += f"{row['auroc']:.3f} \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmark_60.json",
        help="Path to benchmark dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--documents",
        type=str,
        default=None,
        help="Path to knowledge base documents"
    )
    
    args = parser.parse_args()
    
    # Load documents
    if args.documents:
        with open(args.documents, 'r') as f:
            documents = [line.strip() for line in f if line.strip()]
    else:
        # Default documents
        documents = [
            "The Eiffel Tower is located in Paris, France.",
            "Paris is the capital city of France.",
            "France is a country located in Western Europe.",
            "Machine learning is a subfield of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Python is a popular programming language for AI.",
            "The Earth revolves around the Sun.",
            "Water boils at 100 degrees Celsius at sea level.",
            "DNA carries genetic information.",
            "Photosynthesis uses sunlight, carbon dioxide, and water.",
        ]
    
    # Initialize runner
    runner = ExperimentRunner(
        dataset_path=args.dataset,
        documents=documents,
        model_name=args.model
    )
    
    # Run experiments
    print("\n" + "="*70)
    print("Starting Full Experimental Grid (12 experiments)")
    print("="*70 + "\n")
    
    summary_df, all_results_df = runner.run_all_experiments()
    
    # Save results
    output_dir = Path("results")
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    all_results_df.to_csv(output_dir / "all_results.csv", index=False)
    
    print("\n" + "="*70)
    print("All Experiments Complete!")
    print("="*70)
    
    # Generate plots
    runner.generate_plots(summary_df)
    
    # Generate LaTeX table
    latex_table = runner.generate_latex_table(summary_df)
    with open(output_dir / "results_table.tex", 'w') as f:
        f.write(latex_table)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY METRICS")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    print(f"\n✓ Results saved to {output_dir}/")
    print("✓ LaTeX table: results/results_table.tex")
    print("✓ Plots: results/plots/")


if __name__ == "__main__":
    main()