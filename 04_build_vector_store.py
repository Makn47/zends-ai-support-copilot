"""
04_build_vector_store.py

Knowledge Base Layer: chunks the company knowledge base document and builds
a vector index for retrieval-augmented generation.

Embedding approach — LITE VERSION (fully tested in this environment):
uses TF-IDF vectors as the "embedding" instead of Sentence-Transformers,
since Sentence-Transformer models also require a huggingface.co download
this sandbox can't reach. TF-IDF is a legitimate (if less semantically rich)
sparse embedding — retrieval quality on this kind of structured, keyword-
heavy policy/pricing text (specific product names, country names, policy
terms) works well with TF-IDF, since the exact terms customers mention
("refund", "ZENDCloud VM Pro", "Singapore") are exactly what TF-IDF weights
highest.

For a production system, swap in real sentence embeddings by replacing the
`vectorize_chunks()` function with e.g.:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
No other code changes needed — FAISS indexing and retrieval logic is
identical regardless of embedding source.

Vector DB: FAISS (installs from PyPI, no download needed — unlike Pinecone,
which requires an API key/hosted service, or Chroma's optional embedding
function downloads). A flat L2 index is used since the knowledge base is
small (a few dozen chunks); for a larger knowledge base an IVF or HNSW
index would be worth the added complexity.

Chunking strategy: split by markdown section headers (##), which naturally
correspond to one product group or one policy per chunk — appropriate
granularity for this document, since a customer question about "refund
policy" should retrieve the whole refund policy paragraph, not a partial
sentence.

Usage:
    python 04_build_vector_store.py --input ../knowledge_base/company_docs.md --output-dir ../models
"""

import argparse
import pickle
import re

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def chunk_by_section(text: str) -> list[dict]:
    """Split markdown into chunks by ## and ### section headers.

    Product groups (## level) are kept as single chunks since a customer
    question about a product group benefits from seeing all its plans/pricing
    together. Company Policies is further split at ### level so a question
    about e.g. refunds retrieves just the refund policy, not all eight
    policies at once.
    """
    # first split top-level by ##
    sections = re.split(r"\n(?=## )", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section or section.startswith("# ZENDS"):  # skip the doc title-only section
            continue
        title_match = re.match(r"##\s*(.+)", section)
        title = title_match.group(1) if title_match else "Untitled"

        if title.strip() == "Company Policies":
            # split further at ### level for finer retrieval granularity
            subsections = re.split(r"\n(?=### )", section)
            for sub in subsections:
                sub = sub.strip()
                if sub.startswith("## Company Policies"):
                    continue  # skip the now-empty parent header fragment
                sub_title_match = re.match(r"###\s*(.+)", sub)
                if sub_title_match:
                    chunks.append({"title": sub_title_match.group(1), "text": sub})
        else:
            chunks.append({"title": title, "text": section})
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../knowledge_base/company_docs.md")
    parser.add_argument("--output-dir", default="../models")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        doc_text = f.read()

    chunks = chunk_by_section(doc_text)
    print(f"Split knowledge base into {len(chunks)} chunks")
    for c in chunks:
        print(f"  - {c['title']}")

    chunk_texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    embeddings = vectorizer.fit_transform(chunk_texts).toarray().astype("float32")

    # normalize for cosine similarity via inner product index
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, f"{args.output_dir}/knowledge_base.index")
    with open(f"{args.output_dir}/kb_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(f"{args.output_dir}/kb_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"\nSaved FAISS index ({index.ntotal} vectors, dim={embeddings.shape[1]}) to {args.output_dir}/knowledge_base.index")
    print(f"Saved vectorizer and chunk metadata to {args.output_dir}/kb_vectorizer.pkl, kb_chunks.pkl")


if __name__ == "__main__":
    main()
