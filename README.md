# Hybrid RAG with Re-ranking – Cybersecurity Knowledge Assistant

An **Advanced Retrieval-Augmented Generation** system that combines **keyword search (BM25)** and **semantic search (dense embeddings)**, merges them with **Reciprocal Rank Fusion (RRF)**, and then **re-ranks** the candidates with a **cross-encoder** before an LLM (**openai/gpt-oss-120b** on **Groq**) writes the answer.

## Approach

```
                         ┌──► BM25 keyword search ──┐
 Question ──► Hybrid ────┤                          ├──► RRF fusion ──► Cross-encoder ──► Top-k ──► LLM ──► Answer
              retrieval  └──► Dense vector search ──┘     (merge)        re-ranking        chunks   (Groq)   [sources]
```

| Step | What happens | Tool |
|---|---|---|
| 1. Load | Reads every `.txt` file in `data/` | Python |
| 2. Chunk | **Sentence-aware** chunks (max 90 words), never cutting a sentence; last sentence carried over as overlap | Python / regex |
| 3. Index | Two indexes: a **BM25** keyword index and a **dense** embedding matrix | `rank-bm25`, `all-MiniLM-L6-v2` |
| 4. Hybrid retrieval | Top-8 from BM25 and top-8 from dense search are merged with RRF: `score = Σ 1 / (60 + rank)` | NumPy |
| 5. Re-rank | A cross-encoder reads each (question, chunk) pair together and scores relevance; best 3 are kept | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| 6. Generate | Re-ranked chunks go into a prompt that tells the LLM to answer **only** from the context and cite sources | Groq API (`openai/gpt-oss-120b`) |

### Why this is different from basic vector RAG

| | Basic Vector RAG | Hybrid RAG + Re-ranking |
|---|---|---|
| Retrieval | Dense vectors only | BM25 **and** dense vectors |
| Exact terms / acronyms (SPF, RDP, 3-2-1) | Can miss them | Caught by BM25 |
| Ranking | Cosine similarity | RRF fusion + cross-encoder |
| Chunking | Fixed word windows | Sentence-aware |
| Speed | Faster | Slightly slower (re-ranking step) |
| Accuracy | Good | Higher precision |

## Document collection

`data/` contains 7 short cybersecurity documents written for this project: phishing, malware, ransomware, DDoS, SQL injection, brute force attacks and incident response.

## Project structure

```
rag-hybrid-rerank/
├── data/               # document collection (.txt)
├── rag_hybrid.py       # full RAG pipeline
├── requirements.txt    # dependencies
├── .env.example        # template for the API key
├── .gitignore
└── README.md
```

## Dependencies

- Python 3.9+
- `sentence-transformers`, `rank-bm25`, `numpy`, `groq`, `python-dotenv`

## Setup

1. Clone the repository
   ```bash
   git clone https://github.com/laibak1/rag-hybrid-rerank.git
   cd rag-hybrid-rerank
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free API key from https://console.groq.com/keys, then copy `.env.example` to `.env` and paste the key:
   ```
   GROQ_API_KEY=your_key_here
   ```
   Without a key the program still runs and shows the re-ranked chunks.

## Execution

Ask a single question:
```bash
python rag_hybrid.py --question "What is the 3-2-1 backup rule?"
```

Interactive mode:
```bash
python rag_hybrid.py
```

Options:

| Flag | Meaning |
|---|---|
| `--question`, `-q` | Question to ask |
| `--top_k` | Number of chunks kept after re-ranking (default 3) |

### Run in Google Colab

```python
!git clone https://github.com/laibak1/rag-hybrid-rerank.git
%cd rag-hybrid-rerank
!pip install -q -r requirements.txt

import os
from google.colab import userdata          # add GROQ_API_KEY in the Colab "Secrets" (key icon) panel
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

!python rag_hybrid.py -q "What is the 3-2-1 backup rule?"
```

## Example output

```
[1] Loaded 7 documents from 'data/'
[2] Split into 36 sentence-aware chunks (max 90 words)
[3] Built BM25 keyword index + dense index (384-dim vectors)

[4] Hybrid retrieval
    BM25 top-3 : ['ransomware.txt#2', 'incident_response.txt#3', 'malware.txt#4']
    Dense top-3: ['ransomware.txt#2', 'ransomware.txt#3', 'incident_response.txt#3']
    RRF merged -> 13 unique candidates

[5] Cross-encoder re-ranked top-3:
      6.793  ransomware.txt#2
     -4.978  ransomware.txt#3
     -7.703  incident_response.txt#3

[6] Answer (openai/gpt-oss-120b):
The 3-2-1 rule advises keeping three copies of data, stored on two different types of storage, with one of those copies kept offline or immutable so that ransomware cannot reach it. [ransomware.txt]
```
