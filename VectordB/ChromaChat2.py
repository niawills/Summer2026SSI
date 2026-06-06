"""
Simple CLI chat that retrieves top-k documents from Pinecone and answers with inline citations.

Usage:
  python VectordB/ChromaChatTest.py

Features:
- Uses Pinecone cloud vector storage for retrieval
- Uses same embedding model (text-embedding-3-small with 1536 dimensions)
- Markdown-formatted source list with titles and links
- Uses restaurant review context only, without external web search
- Avoids adding new external sources during chat
- Explicit guardrails to avoid hallucinating beyond sources
- Retries on transient API errors
"""
import pathlib
import sys

# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
env_path = pathlib.Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

import os
import time
import datetime
import json
from typing import List, Dict, Tuple


try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from pinecone import Pinecone
from openai import OpenAI
from uuid import uuid4
import numpy as np
import hashlib

# === Configuration ===

# Access API keys from Streamlit secrets or .env fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "restaurant-bots")
ID_STRATEGY = os.getenv("PINECONE_ID_STRATEGY", "url")  # 'url' (default) or 'content'

# Override with Streamlit secrets if available (for cloud deployment)
if STREAMLIT_AVAILABLE:
    try:
        if hasattr(st, 'secrets') and st.secrets:
            OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", OPENAI_API_KEY)
            PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", PINECONE_API_KEY)
            PINECONE_INDEX = st.secrets.get("PINECONE_INDEX", PINECONE_INDEX)
    except:
        pass  # Use .env values

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSIONS = 1536  # Using 1536 dimensions for Pinecone compatibility
SIMILARITY_TOP_K = 5
MAX_RESPONSE_TOKENS = 500
FALLBACK_TEXT = "No relevant restaurant reviews were found for that question."

# Ingestion-like params for saving external sources
MIN_ARTICLE_LENGTH = 200
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# === Clients ===

# Initialize Pinecone
if not PINECONE_API_KEY:
    print("❌ PINECONE_API_KEY not set in .env file")
    sys.exit(1)

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX)

def get_openai_client():
    """Initialize OpenAI client with API key from Streamlit secrets or .env"""
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set in Streamlit secrets or .env file")
        sys.exit(1)
    return OpenAI(api_key=OPENAI_API_KEY)

openai_client = get_openai_client()

# === Embedding & Retrieval ===

def embed_text(text: str) -> List[float]:
    resp = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
        dimensions=EMBED_DIMENSIONS
    )
    return resp.data[0].embedding

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch embed helper to reduce API calls."""
    if not texts:
        return []
    resp = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIMENSIONS
    )
    return [r.embedding for r in resp.data]

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

DATA_FILES = [
    (pathlib.Path(__file__).resolve().parent / "crimson_coward.json", "crimson_coward"),
    (pathlib.Path(__file__).resolve().parent / "vocelli_pizza.json", "vocelli_pizza"),
]


def load_reviews_from_file(file_path: pathlib.Path) -> List[Dict]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of review objects in {file_path}")
    return data


def prepare_review_vectors(reviews: List[Dict], namespace: str) -> List[Dict]:
    vectors: List[Dict] = []
    texts: List[str] = []
    metadata_list: List[Dict] = []

    for idx, review in enumerate(reviews):
        text = str(review.get("text", "")).strip()
        if not text:
            continue

        metadata = {
            "text": text,
            "source": namespace,
            "author": review.get("author", ""),
            "rating": review.get("rating"),
            "date": review.get("date", ""),
            "sentiment": review.get("sentiment", ""),
            "review_index": idx,
        }
        texts.append(text)
        metadata_list.append(metadata)

    if not texts:
        return []

    embeddings = embed_texts(texts)
    for idx, (emb, meta) in enumerate(zip(embeddings, metadata_list)):
        vector_id = f"{namespace}_{idx:08d}"
        vectors.append({
            "id": vector_id,
            "values": emb,
            "metadata": meta,
        })

    return vectors


def upsert_vectors(vectors: List[Dict], namespace: str) -> int:
    if not vectors:
        return 0

    batch_size = 100
    upserted_total = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        resp = pinecone_index.upsert(vectors=batch, namespace=namespace)
        if isinstance(resp, dict):
            upserted_total += int(resp.get("upserted_count", 0))
        else:
            upserted_total += int(getattr(resp, "upserted_count", 0))
    return upserted_total


def ingest_datasets() -> None:
    print(f"🔁 Ingesting review datasets into Pinecone index '{PINECONE_INDEX}'")
    for file_path, namespace in DATA_FILES:
        if not file_path.exists():
            print(f"⚠️ File not found: {file_path}")
            continue

        reviews = load_reviews_from_file(file_path)
        print(f"⏳ Processing {len(reviews)} reviews from '{file_path.name}' into namespace '{namespace}'")
        vectors = prepare_review_vectors(reviews, namespace)
        upserted = upsert_vectors(vectors, namespace)
        print(f"✅ Upserted {upserted} vectors into namespace '{namespace}'")

    print("🎉 Ingestion complete.")


def retrieve_relevant_chunks(query: str, namespace: str = None, top_k: int = SIMILARITY_TOP_K) -> List[Dict]:
    """Retrieve relevant chunks from Pinecone using the specified namespace."""
    q_emb = embed_text(query)
    
    try:
        results = pinecone_index.query(
            vector=q_emb,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        
        chunks = []
        matches = []
        if isinstance(results, dict):
            matches = results.get('matches', [])
        else:
            matches = getattr(results, 'matches', []) or []
        for match in matches:
            metadata = match.get('metadata', {})
            text = metadata.get('text', '')
            
            if text:
                chunks.append({
                    "document": text,
                    "metadata": metadata
                })
        
        return chunks
    except Exception as e:
        print(f"⚠️ Pinecone query error: {e}")
        return []

def generate_doc_id(url: str, chunk_index: int, chunk_text: str | None = None) -> str:
    """Generate an ID for a document chunk.
    Strategies:
      - url (default): stable per URL and chunk index -> overwrites on repeat runs
      - content: based on the chunk content -> treats new content as a new vector
    """
    try:
        if ID_STRATEGY.lower() == "content" and chunk_text:
            ch = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            return f"webc_{ch}"
        # Fallback/default: URL-based stable IDs
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"web_{url_hash}_chunk_{chunk_index:03d}"
    except Exception:
        # Last resort: random-like but deterministic per input
        base = (url or "") + "|" + str(chunk_index) + "|" + (chunk_text or "")
        h = hashlib.md5(base.encode()).hexdigest()[:12]
        return f"web_{h}"

# === Prompt Construction ===

def build_prompt(query: str, embedded_chunks: List[Dict]) -> str:
    system_instructions = (
        "You are a customer review insights assistant.\n"
        "Use ONLY the provided restaurant review chunks as evidence.\n"
        "If the retrieved reviews do not support a claim, answer: 'Not enough evidence in retrieved reviews.'\n\n"
        "Guidelines:\n"
        "- Be concise, factual, and grounded in the review content.\n"
        "- Summarize customer sentiment, recurring issues, and strengths.\n"
        "- Provide review citations when relevant.\n"
        "- If the question asks for recommendations, keep them short and actionable.\n"
    )
    
    parts = []
    for chunk in embedded_chunks:
        meta = chunk["metadata"]
        title = meta.get("title", "Review Chunk")
        source = meta.get("source", "")
        parts.append(
            f"Title: {title}" + (f" (Source: {source})" if source else "") + f"\n{chunk['document']}"
        )

    context_text = "\n---\n".join(parts)
    
    return (
        f"{system_instructions}\n\n"
        f"REVIEW CONTEXT:\n{context_text}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )


def parse_sources(answer: str) -> Tuple[str, List[Tuple[str, str]]]:
    marker = "\n📚 Sources:"
    if marker in answer:
        ans_part, src_part = answer.split(marker, 1)
        sources = []
        for line in src_part.strip().splitlines():
            if line.startswith("- [") and "](" in line:
                try:
                    title = line.split("[", 1)[1].split("]")[0]
                    url = line.split("(", 1)[1].split(")")[0]
                    sources.append((title, url))
                except Exception:
                    continue
        return ans_part.strip(), sources
    return answer.strip(), []

# === Chat Loop ===

def chat():
    print("Chat Assistant - Powered by Pinecone (type 'exit' or Ctrl-C to quit)")
    try:
        stats = pinecone_index.describe_index_stats()
        initial_count = stats.total_vector_count
        print(f"🔢 Pinecone total embeddings at start: {initial_count}")
    except Exception as e:
        print(f"⚠️ Could not retrieve initial Pinecone count: {e}")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            embedded_chunks = retrieve_relevant_chunks(user_input)
            if not embedded_chunks:
                print(f"Assistant: {FALLBACK_TEXT}")
                continue

            prompt = build_prompt(user_input, embedded_chunks)

            response = None
            for attempt in range(3):
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": prompt}],
                        max_tokens=MAX_RESPONSE_TOKENS,
                        temperature=0.3,
                    )
                    break
                except Exception as e:
                    print(f"API error (attempt {attempt+1}): {e}")
                    time.sleep(1)

            if not response:
                print("Assistant: Sorry, I couldn't get a response.")
                continue

            full_answer = response.choices[0].message.content.strip()
            ans_text, sources = parse_sources(full_answer)

            print(f"\nAssistant: {ans_text}\n")
            print("📚 Sources:" if sources else "📚 Sources: None cited.")
            for title, url in sources:
                print(f"- [{title}]({url})")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "chat":
        chat()
    else:
        ingest_datasets()