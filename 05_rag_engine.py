"""
recommend.py -> actually: rag_engine.py

RAG retrieval + response generation for the AI Customer Support Copilot.

Retrieval: given a customer message, vectorize it with the same TF-IDF
vectorizer used to build the knowledge base index, then search the FAISS
index for the most relevant policy/product chunk(s).

Generation — LITE VERSION (fully tested in this environment): rather than
an actual 7B-parameter open-source LLM (Mistral-7B, Falcon-7B, LLaMA-2-7B —
none of which can be downloaded or run in this sandbox; see README
"Environment Limitations"), responses are built from an intent-and-
sentiment-aware template that quotes the retrieved context directly. This
is a real, working RAG pattern (retrieve relevant grounding text, then
generate a response conditioned on it) — it's the generation *model* that's
substituted, not the RAG architecture itself.

For a production system, replace `generate_response()`'s templating logic
with an actual LLM call, e.g.:
    from transformers import pipeline
    generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")
    prompt = f"Context: {retrieved_context}\\n\\nCustomer ({sentiment}, intent={intent}): {query}\\n\\nAgent reply:"
    response = generator(prompt, max_new_tokens=150)[0]["generated_text"]
The retrieval logic (get_relevant_context) doesn't need to change at all.
"""

import pickle
import re

import faiss
import numpy as np


def load_rag_artifacts(index_path, vectorizer_path, chunks_path):
    index = faiss.read_index(index_path)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, vectorizer, chunks


# Maps predicted intent -> chunk titles that are especially relevant for that
# intent. Used to rerank FAISS results: a chunk in this list for the given
# intent gets a small score boost. This fixes cases where a short customer
# message shares a common word (e.g. "support") with an unrelated chunk
# (e.g. "Support Tiers") purely by coincidence — intent gives useful signal
# TF-IDF alone doesn't have.
INTENT_CHUNK_BOOST = {
    "billing": ["Billing Policy"],
    "refund": ["Refund Policy", "Contract Policy"],
    "technical": ["SLA Policy", "Fair Usage Policy"],
    "complaint": ["Refund Policy", "SLA Policy"],
    "product": [
        "Product Group: Mobile Connectivity",
        "Product Group: Home & Office Internet",
        "Product Group: Business Connectivity",
        "Product Group: Cloud & Data Center Services",
        "Product Group: IoT & Smart Solutions",
    ],
}
BOOST_AMOUNT = 0.03


def get_relevant_context(
    query: str, index, vectorizer, chunks: list[dict], top_k: int = 2, intent: str | None = None
) -> list[dict]:
    """Retrieve the top_k most relevant knowledge base chunks for a query.

    If `intent` is provided (from the intent classifier), chunks relevant to
    that intent get a small score boost before reranking — this uses the
    pipeline's own upstream prediction to disambiguate short/ambiguous
    queries, matching the layered architecture in the brief (intent
    classification feeds into retrieval).
    """
    query_vec = vectorizer.transform([query]).toarray().astype("float32")
    faiss.normalize_L2(query_vec)

    # search more than top_k so reranking has room to reorder
    search_k = min(len(chunks), max(top_k * 3, 6))
    scores, indices = index.search(query_vec, search_k)

    boosted = []
    relevant_titles = INTENT_CHUNK_BOOST.get(intent, [])
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        boost = BOOST_AMOUNT if chunk["title"] in relevant_titles else 0.0
        boosted.append({**chunk, "score": float(score), "boosted_score": float(score) + boost})

    boosted.sort(key=lambda c: c["boosted_score"], reverse=True)
    return boosted[:top_k]


def extract_most_relevant_sentence(query: str, chunk_text: str) -> str:
    """Within a matched chunk, find the specific sentence with the most
    keyword overlap with the query — fixes responses that would otherwise
    quote the chunk's first line regardless of which product/plan the
    customer actually asked about (e.g. a query about 'Postpaid Gold'
    should surface the Postpaid Gold sentence, not the chunk's first bullet
    about Prepaid Basic).
    """
    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    # split on sentence boundaries AND on blank lines (this doc uses blank
    # lines between plan bullets rather than always ending in a period)
    candidates = re.split(r"(?<=[.!?])\s+|\n\n+", chunk_text)
    candidates = [c.strip() for c in candidates if c.strip() and not c.strip().startswith("#")]

    if not candidates:
        return chunk_text[:400]

    best_sentence, best_overlap = candidates[0], -1
    for sentence in candidates:
        sentence_words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        overlap = len(query_words & sentence_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence

    return best_sentence


INTENT_OPENERS = {
    "billing": "Thanks for reaching out about your billing question.",
    "refund": "I understand you're looking for a refund — let's sort this out.",
    "technical": "Sorry you're running into a technical issue — let's get this fixed.",
    "complaint": "I'm sorry to hear about your experience — this is not the standard we aim for.",
    "product": "Happy to help with information about our products.",
}

SENTIMENT_ACKNOWLEDGMENTS = {
    "angry": "I completely understand your frustration, and I want to make this right as quickly as possible. ",
    "happy": "Glad to hear from you! ",
    "neutral": "",
}


def generate_response(query: str, intent: str, sentiment: str, context_chunks: list[dict]) -> str:
    """Build an intent/sentiment-aware, RAG-grounded response (template-based, see module docstring)."""
    opener = INTENT_OPENERS.get(intent, "Thanks for your message.")
    ack = SENTIMENT_ACKNOWLEDGMENTS.get(sentiment, "")

    if not context_chunks:
        context_line = "I don't have specific policy information on this — let me connect you with a specialist who can help further."
    else:
        top_chunk = context_chunks[0]
        snippet = extract_most_relevant_sentence(query, top_chunk["text"])
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(".", 1)[0] + "."
        context_line = f"Here's what applies based on our **{top_chunk['title']}**: {snippet}"

    return f"{opener} {ack}{context_line}"


if __name__ == "__main__":
    # quick smoke test
    index, vectorizer, chunks = load_rag_artifacts(
        "../models/knowledge_base.index", "../models/kb_vectorizer.pkl", "../models/kb_chunks.pkl"
    )

    test_queries = [
        ("Can you process a refund for ZENDCloud VM Pro?", "refund", "neutral"),
        ("Why is my bill 200 for Postpaid Gold?", "billing", "angry"),
        ("Does Prepaid Basic support 5G data?", "product", "happy"),
    ]

    for query, intent, sentiment in test_queries:
        print(f"\n{'=' * 60}\nQuery: {query}  (intent={intent}, sentiment={sentiment})")
        context = get_relevant_context(query, index, vectorizer, chunks, top_k=2, intent=intent)
        print(f"Retrieved: {[c['title'] for c in context]} (scores: {[round(c['score'], 3) for c in context]})")
        response = generate_response(query, intent, sentiment, context)
        print(f"Response: {response}")
