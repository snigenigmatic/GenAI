"""
Quick Start with SciFact Dataset
Uses the same dataset from your notebook
"""

from datasets import load_dataset
from rag_bitsandbytes import RAGSystem
import pandas as pd
import json


def prepare_scifact_for_evaluation(n_samples: int = 50):
    """
    Load SciFact and prepare for evaluation
    This is what you used in your notebook!
    
    Args:
        n_samples: Number of samples to use
    
    Returns:
        List of question dictionaries
    """
    print(f"Loading SciFact dataset ({n_samples} samples)...")
    
    # Load SciFact
    dataset = load_dataset("mteb/scifact", split="test")
    
    questions = []
    for i, item in enumerate(dataset):
        if len(questions) >= n_samples:
            break
        
        claim_id = item.get('_id', str(i))
        claim_text = item.get('text', '')
        
        if not claim_text:
            continue
        
        # SciFact is fact verification, so these are all "answerable"
        # The model should verify if claim is supported by evidence
        questions.append({
            'id': len(questions),
            'question': claim_text,
            'answer': 'Supported' if item.get('label', 0) == 1 else 'Refuted',
            'category': 'answerable',  # All SciFact claims are answerable
            'source': 'scifact',
            'claim_id': claim_id
        })
    
    print(f"Loaded {len(questions)} questions from SciFact")
    return questions


def get_scifact_corpus():
    """
    Get the SciFact corpus for retrieval
    These are the documents to search through
    
    Returns:
        List of document strings
    """
    print("Loading SciFact corpus...")
    
    try:
        # Load the corpus
        corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
        
        documents = []
        for item in corpus:
            text = item.get('text', '')
            title = item.get('title', '')
            
            # Combine title and text
            if title and text:
                doc = f"{title}. {text}"
            else:
                doc = text or title
            
            if doc:
                documents.append(doc)
        
        print(f"Loaded {len(documents)} documents")
        return documents
        
    except Exception as e:
        print(f"Error loading corpus: {e}")
        # Fallback to generic scientific documents
        return [
            "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "DNA stores genetic information in all living organisms.",
            "The mitochondria are the powerhouse of the cell, producing ATP.",
            "Water molecules consist of two hydrogen atoms and one oxygen atom.",
            "The human brain contains approximately 86 billion neurons.",
            "Gravity is one of the four fundamental forces of nature.",
            "The Earth orbits the Sun once every 365.25 days.",
            "Vaccines work by training the immune system to recognize pathogens.",
            "Evolution occurs through the process of natural selection.",
            "The speed of light in a vacuum is 299,792,458 meters per second.",
        ]


def run_quick_scifact_experiment():
    """
    Run a quick experiment on SciFact
    Same as your notebook but with full system
    """
    print("\n" + "="*70)
    print("Quick SciFact Experiment")
    print("="*70 + "\n")
    
    # Load data
    questions = prepare_scifact_for_evaluation(n_samples=20)  # Start with 20
    documents = get_scifact_corpus()
    
    # Initialize RAG with INT4 (what you have working)
    print("\nInitializing RAG system with INT4...")
    rag = RAGSystem(
        documents=documents,
        quantization="int4"
    )
    
    # Run on each question
    print("\nProcessing questions...")
    results = []
    
    for item in questions:
        print(f"\nQuestion {item['id']}/{len(questions)}: {item['question'][:60]}...")
        
        result = rag.query(
            item['question'],
            k=5,
            use_retrieval=True,
            return_details=False
        )
        
        results.append({
            'id': item['id'],
            'question': item['question'][:100],
            'answer': result['answer'][:100],
            'ground_truth': item['answer'],
            'decision': result['decision'],
            'token_entropy': result['uncertainty']['token_entropy'],
            'semantic_variance': result['uncertainty']['semantic_variance'],
            'self_eval': result['uncertainty']['self_eval'],
            'combined_uncertainty': result['uncertainty']['combined'],
            'category': item['category']
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("scifact_results.csv", index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    
    print(f"\nTotal questions: {len(results)}")
    print(f"\nDecision breakdown:")
    for decision in ['answer', 'hedge', 'abstain']:
        count = sum(1 for r in results if r['decision'] == decision)
        print(f"  {decision.capitalize()}: {count} ({count/len(results)*100:.1f}%)")
    
    print(f"\nUncertainty Statistics:")
    print(f"  Token Entropy:")
    print(f"    Mean: {df['token_entropy'].mean():.3f}")
    print(f"    Range: {df['token_entropy'].min():.3f} - {df['token_entropy'].max():.3f}")
    
    print(f"  Semantic Variance:")
    print(f"    Mean: {df['semantic_variance'].mean():.3f}")
    print(f"    Range: {df['semantic_variance'].min():.3f} - {df['semantic_variance'].max():.3f}")
    
    print(f"  Self-Evaluation:")
    print(f"    Mean: {df['self_eval'].mean():.3f}")
    print(f"    Range: {df['self_eval'].min():.3f} - {df['self_eval'].max():.3f}")
    
    # Danger zone analysis (from your notebook!)
    danger_zone = df[
        (df['token_entropy'] < 0.35) & 
        (df['semantic_variance'] > 0.4)
    ]
    
    print(f"\n⚠️  Danger Zone (Overconfident):")
    print(f"  Count: {len(danger_zone)} ({len(danger_zone)/len(df)*100:.1f}%)")
    
    if len(danger_zone) > 0:
        print(f"\n  Examples:")
        for i, row in danger_zone.head(3).iterrows():
            print(f"    - Q: {row['question'][:60]}...")
            print(f"      Entropy: {row['token_entropy']:.3f}, Semantic: {row['semantic_variance']:.3f}")
    
    print(f"\n✓ Results saved to scifact_results.csv")


if __name__ == "__main__":
    run_quick_scifact_experiment()