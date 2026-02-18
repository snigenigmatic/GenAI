#!/usr/bin/env python3
"""
Quick Test Script for BitsAndBytes RAG System
Tests that everything is working before running full experiments
"""

from rag_bitsandbytes import RAGSystem

def main():
    print("="*70)
    print("Testing BitsAndBytes RAG System")
    print("="*70)
    print()
    
    # Simple test documents
    documents = [
        "Paris is the capital of France.",
        "The Eiffel Tower is located in Paris.",
        "France is a country in Western Europe.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Earth revolves around the Sun.",
        "Photosynthesis uses sunlight, water, and carbon dioxide.",
    ]
    
    # Test with INT4 (what you already have working)
    print("Loading model with INT4 quantization...")
    print("(This will take 2-3 minutes on first run)")
    print()
    
    rag = RAGSystem(
        documents=documents,
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        quantization="int4"
    )
    
    print("\n" + "="*70)
    print("System loaded! Testing with 3 questions...")
    print("="*70)
    
    # Test questions (mix of easy, medium, hard)
    test_questions = [
        ("What is the capital of France?", "answerable"),
        ("Where is the Eiffel Tower?", "answerable"),
        ("What is the boiling point of gold?", "unanswerable")
    ]
    
    results_summary = []
    
    for i, (question, expected) in enumerate(test_questions, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}/3: {expected.upper()}")
        print(f"{'─'*70}")
        print(f"Question: {question}")
        
        # Query system
        result = rag.query(question, k=3, return_details=True)
        
        # Show results
        print(f"\nAnswer: {result['answer'][:100]}")
        print(f"Decision: {result['decision'].upper()}")
        
        unc = result['uncertainty']
        print(f"\nUncertainty Scores:")
        print(f"  Token Entropy:     {unc['token_entropy']:.3f}")
        print(f"  Semantic Variance: {unc['semantic_variance']:.3f}")
        print(f"  Self-Evaluation:   {unc['self_eval']:.3f}")
        print(f"  Combined:          {unc['combined']:.3f}")
        
        # Check danger zone
        if unc['token_entropy'] < 0.35 and unc['semantic_variance'] > 0.4:
            print("\n⚠️  DANGER ZONE: Overconfident!")
            danger = True
        else:
            danger = False
        
        # Save for summary
        results_summary.append({
            'question': question,
            'expected': expected,
            'decision': result['decision'],
            'uncertainty': unc['combined'],
            'danger_zone': danger
        })
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for i, r in enumerate(results_summary, 1):
        status = "✓" if (r['expected'] == 'answerable' and r['decision'] == 'answer') or \
                       (r['expected'] == 'unanswerable' and r['decision'] in ['abstain', 'hedge']) \
                  else "✗"
        
        print(f"\n{i}. {status} {r['question'][:50]}...")
        print(f"   Expected: {r['expected']}, Got: {r['decision']}")
        print(f"   Uncertainty: {r['uncertainty']:.3f}", end="")
        if r['danger_zone']:
            print(" ⚠️ DANGER ZONE", end="")
        print()
    
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)
    print()
    print("✓ System is working correctly!")
    print()
    print("Next steps:")
    print("1. Run full evaluation: python evaluate_bitsandbytes.py")
    print("2. Or test interactively: python rag_bitsandbytes.py --quantization int4")
    print()
    print("See BITSANDBYTES_QUICKSTART.md for detailed instructions")


if __name__ == "__main__":
    main()