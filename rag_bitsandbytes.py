"""
Complete RAG System with Uncertainty Estimation using BitsAndBytes
Builds on your working implementation
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss


class Decision(Enum):
    ANSWER = "answer"
    HEDGE = "hedge"
    ABSTAIN = "abstain"


@dataclass
class UncertaintyEstimate:
    """Container for all uncertainty scores"""
    token_entropy: float
    semantic_variance: float
    self_eval_uncertainty: float
    combined_uncertainty: float
    decision: Decision
    confidence: float


class BitsAndBytesLLM:
    """
    Wrapper for BitsAndBytes quantized models
    Supports both FP16 and INT4
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        quantization: str = "int4",  # "fp16" or "int4"
        device: str = "cuda"
    ):
        """
        Initialize model with BitsAndBytes quantization
        
        Args:
            model_name: HuggingFace model name
            quantization: "fp16" or "int4"
            device: "cuda" or "cpu"
        """
        self.model_name = model_name
        self.quantization = quantization
        self.device = device
        
        print(f"\nLoading {model_name}")
        print(f"Quantization: {quantization}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization
        if quantization == "int4":
            print("Using INT4 quantization (BitsAndBytes)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:  # fp16
            print("Using FP16 (no quantization)")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
        
        self.model.eval()
        print("Model loaded successfully\n")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 100,
        return_logprobs: bool = False
    ) -> Dict:
        """
        Generate text with optional logprobs
        
        Returns:
            Dict with 'text' and optionally 'logprobs'
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if return_logprobs or temperature == 0.0:
                # Greedy decoding with logprobs
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else 1.0,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                if return_logprobs:
                    # Extract logprobs from scores
                    logprobs = self._extract_logprobs(outputs.scores)
                    return {'text': text, 'logprobs': logprobs}
                else:
                    return {'text': text}
            else:
                # Simple sampling without logprobs (faster)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated_ids = outputs[0][inputs.input_ids.shape[1]:]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                return {'text': text}
    
    def _extract_logprobs(self, scores: Tuple[torch.Tensor]) -> List[Dict]:
        """
        Extract top-k logprobs from generation scores
        
        Args:
            scores: Tuple of (vocab_size,) tensors, one per generated token
        
        Returns:
            List of dicts mapping tokens to logprobs
        """
        logprobs_list = []
        
        for score in scores:
            # score shape: (batch_size, vocab_size)
            score = score[0]  # Remove batch dimension
            
            # Get top-5 tokens
            top_logprobs, top_indices = torch.topk(
                torch.log_softmax(score, dim=-1),
                k=min(5, score.shape[-1])
            )
            
            # Convert to dict
            logprobs_dict = {}
            for logprob, idx in zip(top_logprobs, top_indices):
                token = self.tokenizer.decode([idx])
                logprobs_dict[token] = logprob.item()
            
            logprobs_list.append(logprobs_dict)
        
        return logprobs_list
    
    @staticmethod
    def compute_token_entropy(logprobs_list: List[Dict]) -> float:
        """
        Compute mean token entropy from logprobs
        
        Args:
            logprobs_list: List of dicts {token: logprob}
        
        Returns:
            Mean entropy across tokens
        """
        entropies = []
        
        for token_logprobs in logprobs_list:
            if not token_logprobs:
                continue
            
            # Extract logprobs and convert to probabilities
            lp = np.array(list(token_logprobs.values()))
            p = np.exp(lp)
            p = p / p.sum()  # Ensure normalization
            
            # Compute entropy: H = -sum(p * log(p))
            H = -np.sum(p * np.log(p + 1e-10))
            entropies.append(H)
        
        return float(np.mean(entropies)) if entropies else 0.0


class Retriever:
    """Sentence-transformers retriever with FAISS"""
    
    def __init__(
        self,
        documents: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initialize retriever with document corpus"""
        self.documents = documents
        print(f"Loading retriever: {model_name}")
        self.embedder = SentenceTransformer(model_name, device="cuda")
        
        print(f"Indexing {len(documents)} documents...")
        embeddings = self.embedder.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype('float32'))
        print(f"Index built: {self.index.ntotal} documents\n")
    
    def retrieve(
        self,
        query: str,
        k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve top-k documents
        
        Returns:
            Tuple of (documents, scores)
        """
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        scores, idx = self.index.search(q_emb.astype('float32'), k)
        
        docs = [self.documents[i] for i in idx[0]]
        scores = scores[0].tolist()
        
        return docs, scores


class UncertaintyEstimator:
    """Estimates epistemic uncertainty using 3 methods"""
    
    def __init__(
        self,
        llm: BitsAndBytesLLM,
        embedder: SentenceTransformer,
        n_samples: int = 5,
        temperature: float = 0.9
    ):
        """
        Initialize uncertainty estimator
        
        Args:
            llm: Language model
            embedder: Sentence transformer for semantic variance
            n_samples: Number of samples for semantic variance
            temperature: Sampling temperature
        """
        self.llm = llm
        self.embedder = embedder
        self.n_samples = n_samples
        self.temperature = temperature
    
    def estimate_token_entropy(self, prompt: str) -> Tuple[str, float]:
        """
        Method 1: Token-level entropy
        
        Returns:
            (answer, entropy)
        """
        output = self.llm.generate(prompt, temperature=0.0, return_logprobs=True)
        answer = output['text'].strip()
        entropy = self.llm.compute_token_entropy(output['logprobs'])
        
        return answer, entropy
    
    def estimate_semantic_variance(self, prompt: str) -> Tuple[str, float, List[str]]:
        """
        Method 2: Semantic variance across samples
        
        Returns:
            (primary_answer, variance, all_answers)
        """
        answers = []
        
        # Generate multiple samples
        for _ in range(self.n_samples):
            output = self.llm.generate(
                prompt,
                temperature=self.temperature,
                max_new_tokens=100
            )
            answers.append(output['text'].strip())
        
        # Embed all answers
        embeddings = self.embedder.encode(
            answers,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Compute pairwise distances
        n = len(embeddings)
        if n > 1:
            distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    sim = np.dot(embeddings[i], embeddings[j])
                    distances.append(1 - sim)
            variance = float(np.mean(distances))
        else:
            variance = 0.0
        
        # Most common answer as primary
        primary = max(set(answers), key=answers.count)
        
        return primary, variance, answers
    
    def estimate_self_evaluation(
        self,
        question: str,
        context: str,
        answer: str
    ) -> float:
        """
        Method 3: Self-evaluation confidence
        
        Returns:
            Uncertainty score (0-1)
        """
        prompt = f"""[INST] Given the context and question below, rate your confidence that the answer is correct on a scale from 0 to 10.

Context: {context[:500]}

Question: {question}

Answer: {answer}

Rate your confidence from 0 (no confidence) to 10 (absolute confidence).
Confidence rating: [/INST]"""
        
        output = self.llm.generate(prompt, temperature=0.0, max_new_tokens=5)
        confidence_text = output['text'].strip()
        
        # Parse confidence
        try:
            import re
            numbers = re.findall(r'\d+', confidence_text)
            if numbers:
                confidence = min(10, max(0, int(numbers[0])))
            else:
                confidence = 5  # Default
        except:
            confidence = 5
        
        # Convert to uncertainty
        uncertainty = (10 - confidence) / 10
        return uncertainty
    
    def estimate_all(
        self,
        prompt: str,
        question: str,
        context: str
    ) -> UncertaintyEstimate:
        """
        Estimate using all 3 methods
        
        Returns:
            UncertaintyEstimate object
        """
        print("  Computing token entropy...", end=" ", flush=True)
        answer, token_entropy = self.estimate_token_entropy(prompt)
        print(f"✓ ({token_entropy:.3f})")
        
        print("  Computing semantic variance...", end=" ", flush=True)
        _, semantic_var, _ = self.estimate_semantic_variance(prompt)
        print(f"✓ ({semantic_var:.3f})")
        
        print("  Computing self-evaluation...", end=" ", flush=True)
        self_eval = self.estimate_self_evaluation(question, context, answer)
        print(f"✓ ({self_eval:.3f})")
        
        # Combined uncertainty (weighted average)
        combined = (
            0.2 * min(1.0, token_entropy / 2.0) +  # Normalize entropy
            0.5 * semantic_var +  # Semantic is most reliable
            0.3 * self_eval
        )
        
        # Make decision
        decision = self._make_decision(answer, combined, token_entropy, semantic_var)
        
        return UncertaintyEstimate(
            token_entropy=token_entropy,
            semantic_variance=semantic_var,
            self_eval_uncertainty=self_eval,
            combined_uncertainty=combined,
            decision=decision,
            confidence=1.0 - combined
        )
    
    def _make_decision(
        self,
        answer: str,
        combined: float,
        entropy: float,
        semantic: float
    ) -> Decision:
        """Decision policy based on uncertainty thresholds"""
        # Check for explicit abstention
        if any(phrase in answer.lower() for phrase in ["i don't know", "cannot answer", "not sure"]):
            return Decision.ABSTAIN
        
        # Danger zone: low entropy but high semantic (overconfident)
        if entropy < 0.35 and semantic > 0.4:
            return Decision.ABSTAIN
        
        # High combined uncertainty
        if combined >= 0.6:
            return Decision.ABSTAIN
        elif combined >= 0.4:
            return Decision.HEDGE
        else:
            return Decision.ANSWER


class RAGSystem:
    """Complete RAG system with uncertainty estimation"""
    
    def __init__(
        self,
        documents: List[str],
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        quantization: str = "int4",
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG system
        
        Args:
            documents: Knowledge base
            model_name: HuggingFace model name
            quantization: "fp16" or "int4"
            retriever_model: Sentence transformer model
        """
        print("="*70)
        print("Initializing RAG System with BitsAndBytes")
        print("="*70)
        
        self.retriever = Retriever(documents, retriever_model)
        self.llm = BitsAndBytesLLM(model_name, quantization)
        self.uncertainty = UncertaintyEstimator(
            self.llm,
            self.retriever.embedder
        )
        
        print("="*70)
        print("RAG System Ready!")
        print("="*70)
        print()
    
    def build_prompt(self, question: str, contexts: List[str]) -> str:
        """Build RAG prompt"""
        context_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(contexts))
        
        return f"""[INST] You are a helpful assistant. Answer the question using ONLY the context provided below. If the answer is not in the context, say "I don't know".

Context:
{context_block}

Question: {question}

Answer: [/INST]"""
    
    def query(
        self,
        question: str,
        k: int = 5,
        use_retrieval: bool = True,
        return_details: bool = False
    ) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: User question
            k: Number of documents to retrieve
            use_retrieval: Whether to use RAG (or direct prompting)
            return_details: Return detailed info
        
        Returns:
            Dictionary with answer and uncertainty info
        """
        if use_retrieval:
            # RAG mode
            contexts, scores = self.retriever.retrieve(question, k=k)
            prompt = self.build_prompt(question, contexts)
            context_str = "\n".join(contexts)
        else:
            # Direct prompting (no RAG)
            prompt = f"[INST] Answer the following question concisely:\n\nQuestion: {question}\n\nAnswer: [/INST]"
            contexts = []
            scores = []
            context_str = ""
        
        # Estimate uncertainty
        uncertainty = self.uncertainty.estimate_all(prompt, question, context_str)
        
        # Get primary answer
        answer, _ = self.uncertainty.estimate_token_entropy(prompt)
        
        # Format response
        if uncertainty.decision == Decision.ANSWER:
            response = answer
        elif uncertainty.decision == Decision.HEDGE:
            response = f"Based on available information, {answer.lower() if answer else 'the answer is unclear'}"
        else:
            response = "I don't have sufficient confidence to answer this question."
        
        result = {
            'question': question,
            'answer': answer,
            'response': response,
            'decision': uncertainty.decision.value,
            'uncertainty': {
                'combined': uncertainty.combined_uncertainty,
                'confidence': uncertainty.confidence,
                'token_entropy': uncertainty.token_entropy,
                'semantic_variance': uncertainty.semantic_variance,
                'self_eval': uncertainty.self_eval_uncertainty
            },
            'use_retrieval': use_retrieval
        }
        
        if return_details:
            result['contexts'] = contexts
            result['retrieval_scores'] = scores
        
        return result
    
    def print_result(self, result: Dict):
        """Pretty print result"""
        print("\n" + "─"*70)
        print(f"Question: {result['question']}")
        print("─"*70)
        
        if 'contexts' in result and result['use_retrieval']:
            print("\nRetrieved Contexts:")
            for i, ctx in enumerate(result['contexts'], 1):
                score = result['retrieval_scores'][i-1]
                print(f"  {i}. [{score:.3f}] {ctx[:80]}...")
        
        print(f"\nAnswer: {result['answer']}")
        print(f"Response: {result['response']}")
        print(f"\n🎯 Decision: {result['decision'].upper()}")
        
        unc = result['uncertainty']
        print(f"\n📊 Uncertainty Scores:")
        print(f"  Combined:          {unc['combined']:.3f} (Confidence: {unc['confidence']:.3f})")
        print(f"  Token Entropy:     {unc['token_entropy']:.3f}")
        print(f"  Semantic Variance: {unc['semantic_variance']:.3f}")
        print(f"  Self-Evaluation:   {unc['self_eval']:.3f}")
        print("─"*70)


def main():
    """Interactive demo"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--quantization", type=str, choices=["fp16", "int4"], default="int4")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    
    # Example documents
    documents = [
        "The Eiffel Tower is located in Paris, France.",
        "Paris is the capital city of France.",
        "Machine learning is a subfield of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Python is a popular programming language for AI.",
        "The Earth revolves around the Sun.",
        "Water boils at 100 degrees Celsius at sea level.",
        "DNA carries genetic information.",
        "Photosynthesis uses sunlight, carbon dioxide, and water.",
        "The human heart pumps blood throughout the body.",
    ]
    
    # Initialize system
    rag = RAGSystem(
        documents=documents,
        model_name=args.model,
        quantization=args.quantization
    )
    
    # Interactive loop
    print("\nInteractive RAG System (type 'exit' to quit)")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ['exit', 'quit']:
            break
        if not question:
            continue
        
        result = rag.query(question, k=args.k, return_details=True)
        rag.print_result(result)


if __name__ == "__main__":
    main()