# AI Customer Support Copilot — ZENDS Communications

An end-to-end AI customer support system for a virtual telecom company: detects customer intent and sentiment, retrieves relevant policy/product information (RAG), and generates a suggested reply for agent review — all through an interactive Streamlit dashboard.

**Domain:** Telecom Industry / Customer Service AI Automation
**Skills:** Python, NLP, Synthetic Data Generation, Intent Classification, Sentiment Analysis, RAG, Vector DB, Streamlit

---

## ⚠️ Read This First: Two Important Deviations from the Brief

This project has two significant, deliberate deviations from the original brief. Both are documented in detail below rather than glossed over, because they materially affect what this project can honestly claim.

### 1. Dataset: 96.3% of the supplied data was exact duplicates

The supplied `zends_customer_queries.csv` (10,000 rows) contains only **368 unique text values** — everything else is an exact repeat. Root cause: the generation notebook (`zends_Communications_Synthetic_Generation_.ipynb`) selects templates/entities using `i % len(list)` over short lists (3 templates, 8 products, 4 countries, 5 amounts), which cycles through the same combinations well before reaching 1,000 rows per intent, and the intended paraphrasing step didn't add enough variety to compensate.

**Decision made (per project owner's choice):** deduplicate down to the honest 368 unique examples rather than fine-tune on a dataset that's 96% duplicated (which would let a model memorize instead of generalize, and would leak identical examples across any train/test split). See `scripts/01_dedupe_data.py`.

**Consequence:** 368 examples across 5 intents (imbalanced: 176 billing vs. 48 each for the rest) and 3 sentiments (imbalanced: 288 neutral, 48 angry, 32 happy) is a small, imbalanced dataset. This directly shaped the modeling choices below — see "A Note on Dataset Size."

### 2. Environment: no access to huggingface.co in this build environment

This project was built in a sandboxed environment whose network access is restricted to PyPI/npm/GitHub — it cannot reach huggingface.co, so it's not possible to download or test-run the brief-specified HuggingFace transformer models (DistilBERT/BERT/RoBERTa fine-tuning, sentence-transformer embeddings, or open-source LLMs like Mistral-7B/Falcon-7B/LLaMA-2-7B).

**What was built instead — a fully working, fully tested "lite" pipeline** using the same architecture but lighter-weight, no-download components:

| Brief spec | Lite version (built & tested here) | Brief-spec script (written, untested here) |
|---|---|---|
| Fine-tuned DistilBERT/BERT/RoBERTa for intent | TF-IDF + Logistic Regression | `02b_train_intent_classifier_huggingface.py` |
| Pretrained HF sentiment model | VADER (rule-based, no download) | `03b_sentiment_analysis_huggingface.py` |
| Sentence-Transformers embeddings | TF-IDF vectors | swap-in noted in `04_build_vector_store.py` |
| Mistral-7B/Falcon-7B/LLaMA-2-7B generation | Intent+sentiment-aware template, grounded in retrieved context | swap-in noted in `05_rag_engine.py` |
| Vector DB (FAISS/Pinecone/Chroma) | **FAISS** (installs from PyPI, no download needed) — same as brief spec | — |

The `*b_*.py` scripts are written correctly to the brief's exact specification and are meant to be run on your own machine or Google Colab where huggingface.co is reachable (ideally with a GPU for the fine-tuning script). Each is clearly commented with this caveat.

---

## Project Structure

```
.
├── data/
│   ├── zends_queries_raw.csv       # supplied dataset (10,000 rows, 368 unique)
│   └── cleaned_queries.csv         # deduplicated (368 rows)
├── knowledge_base/
│   └── company_docs.md             # ZENDS company/product/policy reference doc
├── models/
│   ├── intent_vectorizer.pkl / intent_classifier.pkl / intent_labels.pkl / intent_confusion_matrix.npy
│   ├── sentiment_vectorizer.pkl / sentiment_classifier.pkl / ...   (illustrative only — see caveat below)
│   ├── knowledge_base.index        # FAISS index over knowledge base chunks
│   ├── kb_vectorizer.pkl / kb_chunks.pkl
├── scripts/
│   ├── 01_dedupe_data.py
│   ├── 02_train_intent_classifier.py            # LITE — tested
│   ├── 02b_train_intent_classifier_huggingface.py  # BRIEF SPEC — untested here
│   ├── 03_sentiment_analysis.py                 # LITE — tested (VADER)
│   ├── 03b_sentiment_analysis_huggingface.py    # BRIEF SPEC — untested here
│   ├── 04_build_vector_store.py                 # TF-IDF + FAISS
│   └── 05_rag_engine.py                         # retrieval + response generation
├── streamlit_app/
│   └── app.py
├── requirements.txt
└── README.md
```

## A Note on Dataset Size

368 examples for a 5-class + 3-class classification problem is small by deep learning standards. Two consequences worth being upfront about:

1. **Both the intent and sentiment classifiers score a perfect 1.0 F1** in cross-validation. This is *not* a sign of a highly robust model — it reflects that each intent has a distinctive, non-overlapping vocabulary by construction (e.g. "refund" queries always contain the word "refund"), and that the sentiment labels were themselves generated by VADER from thresholded compound scores, so a bag-of-words model trivially re-learns VADER's own lexicon triggers. A perfect score here means "this small, templated dataset is easy to fit," not "this system handles messy real-world phrasing perfectly." See the caveats directly in `02_train_intent_classifier.py` and `03_sentiment_analysis.py`.
2. Given this, TF-IDF + Logistic Regression (with `class_weight='balanced'` for the imbalance) is the *appropriate* choice for this data size — fine-tuning a full transformer on 368 examples would very likely overfit even harder with no way to validate it's actually generalizing.

## Data Flow / Architecture

```
Customer Message (Streamlit input)
        │
        ▼
Preprocessing (implicit in TF-IDF vectorization)
        │
        ▼
Intent Classification (TF-IDF + Logistic Regression) ──┐
        │                                                │ (intent used to bias retrieval)
        ▼                                                │
Sentiment Analysis (VADER)                               │
        │                                                │
        ▼                                                ▼
Knowledge Base Retrieval (TF-IDF + FAISS, intent-boosted reranking)
        │
        ▼
Response Generation (intent+sentiment-aware template, grounded in retrieved context)
        │
        ▼
Streamlit Dashboard (shows intent, sentiment, retrieved sources, suggested reply)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Workflow / Execution

```bash
cd scripts

# 1. Deduplicate the raw synthetic dataset
python 01_dedupe_data.py --input ../data/zends_queries_raw.csv --output ../data/cleaned_queries.csv

# 2. Train the intent classifier
python 02_train_intent_classifier.py --input ../data/cleaned_queries.csv --output-dir ../models --target intent

# 3. Run sentiment analysis (VADER, no training needed)
python 03_sentiment_analysis.py --input ../data/cleaned_queries.csv

# 4. Build the RAG knowledge base (chunk + TF-IDF + FAISS index)
python 04_build_vector_store.py --input ../knowledge_base/company_docs.md --output-dir ../models

# 5. Smoke-test the RAG retrieval + generation engine
python 05_rag_engine.py
```

Then launch the app:

```bash
cd ../streamlit_app
streamlit run app.py
```

### Running the brief-spec HuggingFace scripts

On a machine or Colab notebook with huggingface.co access:

```bash
pip install transformers datasets torch sentence-transformers

python 02b_train_intent_classifier_huggingface.py --input ../data/cleaned_queries.csv --output-dir ../models/distilbert_intent --epochs 3
python 03b_sentiment_analysis_huggingface.py --input ../data/cleaned_queries.csv
```

## RAG Design Decisions

- **Chunking:** the company knowledge base is split by section (`##`/`###` markdown headers) into 14 chunks — one per product group, and one per individual policy (billing, refund, contract, SLA, privacy, fair usage, support tiers, discount). This granularity means a refund question retrieves just the refund policy, not all eight policies at once.
- **Intent-aware retrieval boosting:** short customer messages can coincidentally share a word with an unrelated chunk (e.g. "Does Prepaid Basic **support** 5G data?" matching the "**Support** Tiers" chunk on the word "support" alone). The already-reliable intent classifier's prediction is used to give a small score boost to chunks known to be relevant for that intent (e.g. `billing` boosts the Billing Policy chunk), which fixed several retrieval mismatches found during testing. See `INTENT_CHUNK_BOOST` in `05_rag_engine.py`.
- **Sentence-level extraction:** rather than quoting a matched chunk's opening lines, `extract_most_relevant_sentence()` finds the specific sentence within the top-matched chunk with the most keyword overlap with the query — so a question about "Postpaid Gold" pricing surfaces the Postpaid Gold line specifically, not whichever plan happens to be listed first in that product group.

## Setup Instructions for Real HuggingFace/LLM Components

If you want to run this with the actual brief-specified stack:
1. Run on a machine with internet access to huggingface.co (this sandbox cannot reach it).
2. `pip install transformers sentence-transformers torch`
3. Swap `TfidfVectorizer` in `04_build_vector_store.py` for `SentenceTransformer("all-MiniLM-L6-v2").encode(...)` — no other code changes needed, FAISS indexing/retrieval logic is identical.
4. Swap the templating logic in `generate_response()` (`05_rag_engine.py`) for an actual LLM call (see the docstring in that file for an example prompt structure using a HuggingFace text-generation pipeline).
5. Run `02b_*.py` and `03b_*.py` for the fine-tuned/pretrained transformer models.

## Key Results & Insights

- Traced and documented a real data-generation bug (modulo-based template cycling) that would otherwise have silently undermined model training.
- 368 unique examples after deduplication; both classifiers hit perfect CV scores, appropriately caveated as a reflection of the data's templated nature rather than genuine robustness.
- RAG retrieval improved substantially (in testing) by combining TF-IDF similarity with intent-aware reranking and sentence-level extraction — both documented as concrete before/after fixes in `05_rag_engine.py`.
- Full pipeline (intent → sentiment → retrieval → generation) runs end-to-end in the Streamlit app with real, verifiable output — not a mockup.

## Tech Stack

- **Languages:** Python 3
- **Libraries (lite pipeline):** pandas, scikit-learn (TfidfVectorizer, LogisticRegression), vaderSentiment, FAISS
- **Libraries (brief-spec scripts, for use elsewhere):** transformers, datasets, torch, sentence-transformers
- **App:** Streamlit

## Coding Standards

- PEP 8 compliant, modular scripts (one responsibility per script)
- Every deviation from the brief is documented in-code (module docstrings) and here in the README — nothing is silently substituted
- All pipeline scripts are CLI-driven with `argparse` for reproducibility
- `random_state=42` throughout for reproducible splits/folds
