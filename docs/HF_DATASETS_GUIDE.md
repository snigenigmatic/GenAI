# Using HuggingFace Hub Datasets - Quick Start

## 🎯 Three Ways to Use Your Data

### Option 1: Use SciFact (What You Already Have!) ⭐ EASIEST

```bash
# This uses the exact dataset from your notebook
python run_scifact.py
```

**What it does:**
- Loads SciFact from HuggingFace Hub (same as your notebook)
- Loads the SciFact corpus for retrieval
- Runs 20 questions through the system
- Shows all 3 uncertainty metrics
- Identifies danger zone cases
- Saves results to `scifact_results.csv`

**Time**: ~15-20 minutes for 20 questions

### Option 2: Build 60-Question Benchmark from HF Hub

```bash
# Creates balanced dataset from multiple HF datasets
python dataset_from_hf.py

# This creates: benchmark_60_hf.json
```

**Dataset composition:**
- 10 from SciFact (scientific claims)
- 10 from Natural Questions (factual QA)
- 20 from PopQA (long-tail entities)
- 20 manually created (conflicting evidence)

**Time**: ~2-3 minutes to download and prepare

Then run evaluation:
```bash
python evaluate_bitsandbytes.py --dataset benchmark_60_hf.json
```

### Option 3: Use Your Own HF Dataset

```python
from datasets import load_dataset
from rag_bitsandbytes import RAGSystem

# Load your dataset
dataset = load_dataset("your-dataset-name", split="test")

# Prepare questions
questions = []
for item in dataset:
    questions.append({
        'id': len(questions),
        'question': item['question'],  # Adjust field name
        'answer': item['answer'],      # Adjust field name
        'category': 'answerable'       # or label appropriately
    })

# Save for evaluation
import json
with open('my_dataset.json', 'w') as f:
    json.dump(questions, f, indent=2)

# Run evaluation
python evaluate_bitsandbytes.py --dataset my_dataset.json
```

## 📊 Available HF Datasets for RAG Research

| Dataset | HF Name | Questions | Good For |
|---------|---------|-----------|----------|
| **SciFact** | `mteb/scifact` | 1,109 | Scientific fact verification |
| **Natural Questions** | `nq_open` | 3,610 | Factual QA (Google queries) |
| **TriviaQA** | `trivia_qa` | 95k | Trivia questions |
| **PopQA** | `akariasai/PopQA` | 14k | Long-tail entities |
| **AmbigQA** | `ambig_qa` | 14k | Ambiguous questions |
| **HotpotQA** | `hotpot_qa` | 113k | Multi-hop reasoning |
| **MS MARCO** | `ms_marco` | 1M | Passage ranking |

## 🚀 Quick Start Commands

### 1. Test with SciFact (Fastest)

```bash
# Just run this - uses your working setup!
python run_scifact.py
```

Expected output:
```
Loading SciFact dataset (20 samples)...
Loading SciFact corpus...
Initializing RAG system with INT4...
Processing questions...

Question 0/20: Memory deficits in middle-aged Tg2576 mice are caused by...
  Computing token entropy... ✓ (0.342)
  Computing semantic variance... ✓ (0.156)
  Computing self-evaluation... ✓ (0.300)

[... processes all 20 questions ...]

Results Summary
======================================================================

Total questions: 20

Decision breakdown:
  Answer: 14 (70.0%)
  Hedge: 4 (20.0%)
  Abstain: 2 (10.0%)

Uncertainty Statistics:
  Token Entropy:
    Mean: 0.341
    Range: 0.184 - 0.571

  Semantic Variance:
    Mean: 0.288
    Range: 0.017 - 0.757

⚠️  Danger Zone (Overconfident):
  Count: 3 (15.0%)

✓ Results saved to scifact_results.csv
```

### 2. Build Complete Benchmark

```bash
# Download and prepare 60 questions
python dataset_from_hf.py

# Output: benchmark_60_hf.json
```

### 3. Run Full Experiments

```bash
# Use SciFact
python evaluate_bitsandbytes.py \
    --dataset scifact \
    --n-samples 50

# Or use prepared benchmark
python evaluate_bitsandbytes.py \
    --dataset benchmark_60_hf.json
```

## 📝 Custom Dataset Format

If you want to use your own questions, create a JSON file:

```json
[
  {
    "id": 0,
    "question": "Your question here?",
    "answer": "The correct answer",
    "category": "answerable",
    "source": "your_source"
  },
  {
    "id": 1,
    "question": "Another question?",
    "answer": "Another answer",
    "category": "weak_evidence",
    "source": "your_source"
  }
]
```

Required fields:
- `id`: Unique integer
- `question`: The question text
- `answer`: Ground truth answer (for correctness checking)
- `category`: One of `answerable`, `weak_evidence`, `conflicting_evidence`

Optional fields:
- `source`: Where the question came from
- `should_abstain`: Boolean (auto-computed if not provided)
- Any other metadata you want

## 🔍 Using Different Corpuses

### SciFact Corpus (Scientific)

```python
from datasets import load_dataset

corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
documents = [f"{item['title']}. {item['text']}" for item in corpus]
```

### Wikipedia (General Knowledge)

```python
from datasets import load_dataset

wiki = load_dataset("wikipedia", "20220301.en", split="train[:10000]")
documents = [item['text'][:500] for item in wiki]  # First 500 chars
```

### Your Own Documents

```python
# From text file (one document per line)
with open('my_documents.txt', 'r') as f:
    documents = [line.strip() for line in f if line.strip()]

# From list
documents = [
    "Document 1 content here.",
    "Document 2 content here.",
    # ... more documents
]
```

## 💡 Tips for Dataset Selection

### For Your Research (Uncertainty under Quantization)

**Best choice**: Use SciFact
- ✅ You already validated it works
- ✅ ~1000 questions available
- ✅ Scientific domain (good for uncertainty)
- ✅ Has corpus for retrieval
- ✅ Clear ground truth labels

**Alternative**: Mix of datasets
- SciFact (scientific)
- Natural Questions (factual)
- PopQA (long-tail)
- Manual conflicting (edge cases)

### Sample Sizes

| Purpose | N Questions | Time (INT4) |
|---------|-------------|-------------|
| Quick test | 10-20 | 10-20 min |
| Development | 50 | 45-60 min |
| Full experiment | 60 | 60-90 min |
| Publication | 100-200 | 2-4 hours |

## 🎯 Recommended Workflow

**Week 1: Validation**
```bash
# Day 1: Test with small sample
python run_scifact.py  # 20 questions

# Day 2-3: Run INT4 experiments
python evaluate_bitsandbytes.py --dataset scifact --n-samples 50

# Day 4-5: Analyze results, validate danger zones
```

**Week 2: Full Experiments**
```bash
# Build benchmark
python dataset_from_hf.py

# Run all 12 experiments
python evaluate_bitsandbytes.py --dataset benchmark_60_hf.json
```

**Week 3: Paper Writing**
- Analyze results
- Generate plots
- Write paper

## 🔧 Troubleshooting

### Dataset Download Fails

```bash
# Use HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com

# Or download manually and cache
huggingface-cli download mteb/scifact
```

### Dataset Not Found

```python
# Check available datasets
from datasets import list_datasets
datasets = list_datasets()
print([d for d in datasets if 'scifact' in d.lower()])
```

### Wrong Dataset Format

```python
# Inspect dataset structure
from datasets import load_dataset
dataset = load_dataset("your-dataset", split="test")
print(dataset[0])  # Show first item
print(dataset.column_names)  # Show available fields
```

## 📦 File Structure

After running:

```
.
├── run_scifact.py              # Quick SciFact experiment
├── dataset_from_hf.py          # Build 60-question benchmark
├── evaluate_bitsandbytes.py    # Full evaluation
├── scifact_results.csv         # SciFact results
├── benchmark_60_hf.json        # Prepared benchmark (if created)
└── results/                    # Full experiment results
    ├── E1_results.json
    ├── E1_metrics.json
    ├── ...
    └── plots/
```

## 🎓 Examples from Your Notebook

### What You Had (Notebook)

```python
# Your code
dataset = load_dataset("mteb/scifact", split="test")

for item in dataset:
    claim = item['text']
    # ... process claim ...
    
    entropy = compute_entropy(...)
    semantic_unc = compute_semantic(...)
```

### What You Have Now

```python
# New: Complete system
from run_scifact import run_quick_scifact_experiment

# Runs everything automatically
run_quick_scifact_experiment()

# Or use programmatically
from rag_bitsandbytes import RAGSystem
from datasets import load_dataset

dataset = load_dataset("mteb/scifact", split="test")
rag = RAGSystem(documents=corpus, quantization="int4")

for item in dataset[:10]:
    result = rag.query(item['text'])
    print(f"Uncertainty: {result['uncertainty']['combined']:.3f}")
```

## ✅ Summary

**Easiest path**: 
1. Run `python run_scifact.py`
2. Check `scifact_results.csv`
3. Validate it matches your notebook findings

**Full research path**:
1. Run `python dataset_from_hf.py`
2. Run `python evaluate_bitsandbytes.py --dataset benchmark_60_hf.json`
3. Analyze results in `results/`

**Both use HuggingFace Hub - no manual JSON files needed!** 🚀

