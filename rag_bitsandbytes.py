"""
RAG System with Epistemic Uncertainty Estimation
BitsAndBytes quantization (FP16 / INT4)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────


class Decision(Enum):
    ANSWER = "answer"
    HEDGE = "hedge"
    ABSTAIN = "abstain"


@dataclass
class UncertaintyEstimate:
    answer: str
    token_entropy: float
    semantic_variance: float
    self_eval_uncertainty: float
    combined: float
    decision: Decision
    confidence: float


# ─────────────────────────────────────────────
# LLM wrapper
# ─────────────────────────────────────────────


class BitsAndBytesLLM:
    """
    Wraps a HuggingFace causal LM with BitsAndBytes quantisation.
    Uses apply_chat_template so Mistral-Instruct actually follows instructions.
    """

    def __init__(
        self,
        model_name: str = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        quantization: str = "int4",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.device = device

        print(f"\nLoading {model_name}  [{quantization}]")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if quantization == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )

        self.model.eval()
        print("Model ready\n")

    # ── helpers ──────────────────────────────

    def _apply_template(self, messages: List[Dict]) -> Dict[str, torch.Tensor]:
        """Apply the model's chat template and move to device."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

    @staticmethod
    def _entropy_from_scores(scores: Tuple[torch.Tensor]) -> float:
        """
        Mean per-token Shannon entropy over the FULL vocabulary distribution.
        H = -sum(p * log p)
        This matches what the notebook computes.
        """
        entropies = []
        for step_logits in scores:
            logits = step_logits[0]  # (vocab,)
            probs = torch.softmax(logits, dim=-1)
            log_p = torch.log_softmax(logits, dim=-1)
            H = -(probs * log_p).sum().item()
            entropies.append(H)
        return float(np.mean(entropies)) if entropies else 0.0

    # ── generation ───────────────────────────

    def greedy(
        self,
        messages: List[Dict],
        max_new_tokens: int = 100,
    ) -> Tuple[str, float]:
        """
        Greedy decode; returns (answer_text, mean_token_entropy).
        Single forward pass — no redundant re-generation.
        """
        inputs = self._apply_template(messages)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(
            out.sequences[0][input_len:], skip_special_tokens=True
        ).strip()
        entropy = self._entropy_from_scores(out.scores)
        return text, entropy

    def sample(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_new_tokens: int = 100,
    ) -> str:
        """Single sampled generation for semantic variance."""
        inputs = self._apply_template(messages)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(
            out[0][input_len:], skip_special_tokens=True
        ).strip()


# ─────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────


class Retriever:
    """FAISS dense retriever backed by sentence-transformers."""

    def __init__(
        self,
        documents: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.documents = documents

        print(f"Loading retriever: {model_name}")
        self.embedder = SentenceTransformer(model_name, device="cpu")

        print(f"Indexing {len(documents)} documents …")
        embeddings = self.embedder.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"Index ready: {self.index.ntotal} docs\n")

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[str], List[float]]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype(
            "float32"
        )
        scores, indices = self.index.search(q_emb, k)
        docs = [self.documents[i] for i in indices[0]]
        return docs, scores[0].tolist()


# ─────────────────────────────────────────────
# Uncertainty estimator
# ─────────────────────────────────────────────


class UncertaintyEstimator:
    """
    Three uncertainty signals:
      1. Token entropy   — how uncertain the model is token-by-token
      2. Semantic variance — disagreement across stochastic samples
      3. Self-evaluation  — model's own confidence rating
    """

    def __init__(
        self,
        llm: BitsAndBytesLLM,
        embedder: SentenceTransformer,
        n_samples: int = 3,
        temperature: float = 0.7,
    ):
        self.llm = llm
        self.embedder = embedder
        self.n_samples = n_samples
        self.temperature = temperature

    # ── individual signals ───────────────────

    def semantic_variance(self, messages: List[Dict]) -> float:
        """Mean pairwise cosine distance across n sampled answers."""
        answers = [
            self.llm.sample(messages, temperature=self.temperature)
            for _ in range(self.n_samples)
        ]
        print(
            f"   Sampled answers for semantic variance:\n      "
            + "\n      ".join(answers)
        )
        embeddings = self.embedder.encode(answers, normalize_embeddings=True)
        n = len(embeddings)
        if n < 2:
            return 0.0
        dists = [
            1 - float(np.dot(embeddings[i], embeddings[j]))
            for i in range(n)
            for j in range(i + 1, n)
        ]
        return float(np.mean(dists))

    def self_eval(self, question: str, context: str, answer: str) -> float:
        """Ask the model to rate its own confidence → convert to uncertainty."""
        messages = [
            {
                "role": "user",
                "content": (
                    "Given the context and question, rate your confidence that the answer "
                    "is correct on a scale from 0 (no confidence) to 10 (certain).\n\n"
                    f"Context: {context[:500]}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer: {answer}\n\n"
                    "Confidence rating (0-10, single integer):"
                ),
            }
        ]
        raw, _ = self.llm.greedy(messages, max_new_tokens=5)

        import re

        nums = re.findall(r"\d+", raw)
        score = min(10, max(0, int(nums[0]))) if nums else 5
        return (10 - score) / 10.0

    # ── combined ─────────────────────────────

    def estimate_all(
        self,
        messages: List[Dict],
        question: str,
        context: str,
    ) -> UncertaintyEstimate:
        """
        Run all three signals.
        Returns an UncertaintyEstimate that includes the greedy answer text —
        so callers never need to re-run generation.
        """
        print("  [1/3] token entropy …", end=" ", flush=True)
        answer, t_entropy = self.llm.greedy(messages)
        print(f"{t_entropy:.3f}")

        print("  [2/3] semantic variance …", end=" ", flush=True)
        sem_var = self.semantic_variance(messages)
        print(f"{sem_var:.3f}")

        print("  [3/3] self-evaluation …", end=" ", flush=True)
        s_eval = self.self_eval(question, context, answer)
        print(f"{s_eval:.3f}")

        # Weighted combination
        combined = (
            0.2 * min(1.0, t_entropy / 2.0)  # normalise entropy to [0,1]
            + 0.5 * sem_var
            + 0.3 * s_eval
        )

        decision = self._decide(answer, combined, t_entropy, sem_var)

        return UncertaintyEstimate(
            answer=answer,
            token_entropy=t_entropy,
            semantic_variance=sem_var,
            self_eval_uncertainty=s_eval,
            combined=combined,
            decision=decision,
            confidence=1.0 - combined,
        )

    @staticmethod
    def _decide(
        answer: str, combined: float, entropy: float, semantic: float
    ) -> Decision:
        if any(
            p in answer.lower()
            for p in ["i don't know", "cannot answer", "not sure", "i cannot"]
        ):
            return Decision.ABSTAIN
        if (
            entropy < 0.35 and semantic > 0.4
        ):  # danger zone: low entropy, high disagreement
            return Decision.ABSTAIN
        if combined >= 0.6:
            return Decision.ABSTAIN
        if combined >= 0.4:
            return Decision.HEDGE
        return Decision.ANSWER


# ─────────────────────────────────────────────
# RAG system
# ─────────────────────────────────────────────


class RAGSystem:
    """End-to-end RAG with uncertainty-aware answering."""

    def __init__(
        self,
        documents: List[str],
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        quantization: str = "int4",
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        print("=" * 70)
        print("Initialising RAG System")
        print("=" * 70)

        self.retriever = Retriever(documents, retriever_model)
        self.llm = BitsAndBytesLLM(model_name, quantization)
        self.uncertainty = UncertaintyEstimator(self.llm, self.retriever.embedder)

        print("=" * 70)
        print("System ready")
        print("=" * 70 + "\n")

    # ── prompt builder ───────────────────────

    @staticmethod
    def _build_messages(question: str, contexts: List[str]) -> List[Dict]:
        context_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(contexts))
        return [
            {
                "role": "user",
                "content": (
                    "Answer the question using ONLY the context below. "
                    'If the answer is not present, say "I don\'t know".\n\n'
                    f"Context:\n{context_block}\n\n"
                    f"Question: {question}"
                ),
            }
        ]

    @staticmethod
    def _build_direct_messages(question: str) -> List[Dict]:
        return [{"role": "user", "content": f"Answer concisely:\n{question}"}]

    # ── query ────────────────────────────────

    def query(
        self,
        question: str,
        k: int = 5,
        use_retrieval: bool = True,
        return_details: bool = False,
    ) -> Dict:
        if use_retrieval:
            contexts, scores = self.retriever.retrieve(question, k=k)
            messages = self._build_messages(question, contexts)
            context_str = "\n".join(contexts)
        else:
            contexts = []
            scores = []
            context_str = ""
            messages = self._build_direct_messages(question)

        unc = self.uncertainty.estimate_all(messages, question, context_str)

        # Format the public response based on decision
        if unc.decision == Decision.ANSWER:
            response = unc.answer
        elif unc.decision == Decision.HEDGE:
            lower = unc.answer.lower() if unc.answer else "the answer is unclear"
            response = f"Based on available information, {lower}"
        else:
            response = "I don't have sufficient confidence to answer this question."

        result = {
            "question": question,
            "answer": unc.answer,
            "response": response,
            "decision": unc.decision.value,
            "uncertainty": {
                "combined": unc.combined,
                "confidence": unc.confidence,
                "token_entropy": unc.token_entropy,
                "semantic_variance": unc.semantic_variance,
                "self_eval": unc.self_eval_uncertainty,
            },
            "use_retrieval": use_retrieval,
        }

        if return_details:
            result["contexts"] = contexts
            result["retrieval_scores"] = scores

        return result

    # ── pretty print ─────────────────────────

    @staticmethod
    def print_result(result: Dict) -> None:
        print("\n" + "─" * 70)
        print(f"Q: {result['question']}")
        if result.get("contexts"):
            print("\nContexts:")
            for i, (c, s) in enumerate(
                zip(result["contexts"], result["retrieval_scores"]), 1
            ):
                print(f"  {i}. [{s:.3f}] {c[:80]}…")
        print(f"\nAnswer:   {result['answer'][:120]}")
        print(f"Response: {result['response'][:120]}")
        print(f"\nDecision: {result['decision'].upper()}")
        u = result["uncertainty"]
        print(f"\nUncertainty:")
        print(
            f"  Combined:          {u['combined']:.3f}  (confidence: {u['confidence']:.3f})"
        )
        print(f"  Token entropy:     {u['token_entropy']:.3f}")
        print(f"  Semantic variance: {u['semantic_variance']:.3f}")
        print(f"  Self-eval:         {u['self_eval']:.3f}")
        print("─" * 70)


# ─────────────────────────────────────────────
# Quick smoke-test  (python rag_bitsandbytes.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    docs = [
        "Paris is the capital of France.",
        "The Eiffel Tower is located in Paris.",
        "Water boils at 100 °C at sea level.",
        "DNA stores genetic information.",
        "The Earth orbits the Sun.",
    ]

    rag = RAGSystem(
        docs,
        model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        quantization="fp16",
    )

    for q in [
        "What is the capital of France?",
        "Where is the Eiffel Tower?",
        "What is the boiling point of gold?",
    ]:
        print(f"\nProcessing: {q}")
        r = rag.query(q, k=3, return_details=True)
        rag.print_result(r)
