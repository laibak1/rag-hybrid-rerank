"""
Hybrid RAG with Re-ranking
==========================
Pipeline:
  1. Load documents      -> read .txt files from ./data
  2. Chunk documents     -> split into sentence-aware chunks
  3. Index (two ways)    -> a) BM25 keyword index   b) dense embedding index
  4. Hybrid retrieval    -> BM25 top-N + dense top-N, merged with Reciprocal Rank Fusion (RRF)
  5. Re-ranking          -> cross-encoder scores every (question, chunk) pair, keep best k
  6. Generate            -> LLM (Groq, Llama) answers using only the re-ranked context

Why hybrid? BM25 is strong on exact terms (e.g. "SPF", "3-2-1", "RDP"),
dense search is strong on meaning/synonyms. The cross-encoder then reads
question and chunk together, which is more accurate than either score alone.

Usage:
  python rag_hybrid.py --question "What is the 3-2-1 backup rule?"
  python rag_hybrid.py            # interactive mode
"""

import argparse
import glob
import os
import re

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

DATA_DIR = "data"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_CHUNK_WORDS = 90
CANDIDATES = 8   # chunks taken from each retriever before fusion
RRF_K = 60       # standard constant for Reciprocal Rank Fusion


# ---------------------------------------------------------------- 1. Load
def load_documents(data_dir=DATA_DIR):
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})
    print(f"[1] Loaded {len(docs)} documents from '{data_dir}/'")
    return docs


# ---------------------------------------------------------------- 2. Chunk (sentence-aware)
def chunk_text(text, max_words=MAX_CHUNK_WORDS):
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    chunks, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence]).split()) > max_words:
            chunks.append(" ".join(current))
            current = current[-1:]   # carry last sentence over as overlap
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_documents(docs):
    chunks = []
    for doc in docs:
        for i, piece in enumerate(chunk_text(doc["text"])):
            chunks.append({"id": f"{doc['source']}#{i}", "source": doc["source"], "text": piece})
    print(f"[2] Split into {len(chunks)} sentence-aware chunks (max {MAX_CHUNK_WORDS} words)")
    return chunks


# ---------------------------------------------------------------- 3. Indexes
def tokenize(text):
    return re.findall(r"[a-z0-9\-]+", text.lower())


class HybridIndex:
    def __init__(self, chunks, embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self.vectors = np.asarray(
            embedder.encode([c["text"] for c in chunks], normalize_embeddings=True,
                            show_progress_bar=False), dtype="float32")
        print(f"[3] Built BM25 keyword index + dense index ({self.vectors.shape[1]}-dim vectors)")

    def bm25_search(self, question, n):
        scores = self.bm25.get_scores(tokenize(question))
        return list(np.argsort(scores)[::-1][:n])

    def dense_search(self, question, n):
        q = self.embedder.encode([question], normalize_embeddings=True)[0]
        scores = self.vectors @ q
        return list(np.argsort(scores)[::-1][:n])


# ---------------------------------------------------------------- 4. Hybrid retrieval (RRF)
def hybrid_retrieve(question, index, n=CANDIDATES):
    bm25_ids = index.bm25_search(question, n)
    dense_ids = index.dense_search(question, n)

    fused = {}
    for ranking in (bm25_ids, dense_ids):
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
    ordered = sorted(fused, key=fused.get, reverse=True)

    print(f"\n[4] Hybrid retrieval")
    print(f"    BM25 top-3 : {[index.chunks[i]['id'] for i in bm25_ids[:3]]}")
    print(f"    Dense top-3: {[index.chunks[i]['id'] for i in dense_ids[:3]]}")
    print(f"    RRF merged -> {len(ordered)} unique candidates")
    return [index.chunks[i] for i in ordered]


# ---------------------------------------------------------------- 5. Re-rank
def rerank(question, candidates, reranker, top_k=3):
    scores = reranker.predict([(question, c["text"]) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
    print(f"\n[5] Cross-encoder re-ranked top-{top_k}:")
    for c, s in ranked:
        print(f"    {s:7.3f}  {c['id']}")
    return [{**c, "score": float(s)} for c, s in ranked]


# ---------------------------------------------------------------- 6. Generate
def generate(question, contexts):
    context_text = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in contexts)
    prompt = (
        "You are a cybersecurity assistant. Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know. "
        "Cite the source file names in square brackets.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "(GROQ_API_KEY not set - showing re-ranked context only)\n\n" + context_text

    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG with re-ranking")
    parser.add_argument("--question", "-q", help="question to ask (omit for interactive mode)")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    index = HybridIndex(chunk_documents(load_documents()), embedder)

    def ask(question):
        candidates = hybrid_retrieve(question, index)
        contexts = rerank(question, candidates, reranker, args.top_k)
        print(f"\n[6] Answer ({LLM_MODEL}):\n")
        print(generate(question, contexts))

    if args.question:
        ask(args.question)
    else:
        print("\nInteractive mode - type 'exit' to quit.")
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in {"exit", "quit", ""}:
                break
            ask(question)


if __name__ == "__main__":
    main()
