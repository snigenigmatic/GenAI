"""
Fetch Wikipedia passages for benchmark questions.
Produces docs.txt — one passage per line — ready for --documents flag.

Usage:
    uv run fetch_wiki_docs.py                          # uses benchmark_60_hf.json
    uv run fetch_wiki_docs.py --dataset my.json --out docs.txt
"""

import json
import re
import time
import argparse
from pathlib import Path
from typing import List, Optional

try:
    import wikipedia
except ImportError:
    raise SystemExit("Run: uv pip install wikipedia")


# ── helpers ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, max_words: int = 80, overlap: int = 20) -> List[str]:
    """Split article text into overlapping word-window chunks."""
    words  = text.split()
    chunks = []
    step   = max_words - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + max_words])
        if len(chunk.split()) >= 15:          # skip tiny tail chunks
            chunks.append(chunk)
    return chunks


def clean(text: str) -> str:
    """Remove Wikipedia markup noise."""
    text = re.sub(r"==+[^=]+=+", " ", text)   # section headers
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def question_to_queries(question: str, answer: str) -> List[str]:
    """
    Build 2-3 search queries per item:
      1. Cleaned question (strip wh-words that confuse Wikipedia search)
      2. Answer entity directly (most reliable for named entities)
      3. Combined short form
    """
    q = question.strip().rstrip("?")

    # Strip leading question words — Wikipedia search hates them
    q_clean = re.sub(
        r"^(who (was|is|were|has)|what (is|are|was|were)|"
        r"where (is|does|did)|when (was|did|do)|"
        r"how (many|much|do|does|did)|in what|which)\s+",
        "", q, flags=re.IGNORECASE
    ).strip()

    queries = [q_clean]

    # Add answer entity as a direct search target if it's a real value
    if answer and answer.strip() not in ("[", "", "unknown"):
        queries.append(answer.strip())

    return queries


def fetch_page(title: str) -> List[str]:
    """Fetch a single Wikipedia page by title, return chunks. Handles disambiguation."""
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return chunk_text(clean(page.content))
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return chunk_text(clean(page.content))
        except Exception:
            return []
    except Exception as e:
        print(f"    page error ({title}): {e}")
        return []


def fetch_for_question(question: str, answer: str = "", n_results: int = 3) -> List[str]:
    wikipedia.set_lang("en")
    wikipedia.set_rate_limiting(True)

    queries  = question_to_queries(question, answer)
    seen     = set()
    passages = []

    for query in queries:
        try:
            titles = wikipedia.search(query, results=n_results)
        except Exception as e:
            print(f"    search error ({query!r}): {e}")
            continue

        for title in titles:
            if title in seen:
                continue
            seen.add(title)
            chunks = fetch_page(title)
            passages.extend(chunks)
            time.sleep(0.2)

    return passages


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="benchmark_60_hf.json")
    parser.add_argument("--out",        default="docs.txt")
    parser.add_argument("--n_results",  type=int, default=3,
                        help="Wikipedia articles to fetch per question")
    parser.add_argument("--only_answerable", action="store_true",
                        help="Only fetch docs for answerable questions (faster)")
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    if args.only_answerable:
        dataset = [q for q in dataset if q.get("category") == "answerable"]
        print(f"Fetching docs for {len(dataset)} answerable questions only")
    else:
        print(f"Fetching docs for {len(dataset)} questions")

    all_passages = set()      # dedup via set

    for i, item in enumerate(dataset):
        q      = item["question"]
        answer = item.get("answer", "")
        print(f"[{i+1}/{len(dataset)}] {q[:70]}")
        passages = fetch_for_question(q, answer=answer, n_results=args.n_results)
        all_passages.update(passages)
        print(f"    +{len(passages)} passages  (total unique: {len(all_passages)})")

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in sorted(all_passages):       # sorted for reproducibility
            f.write(p.replace("\n", " ") + "\n")

    print(f"\n✓ Wrote {len(all_passages)} passages to {out_path}")
    print(f"  Pass to eval with:  --documents {out_path}")


if __name__ == "__main__":
    main()