import os
from typing import List, Dict

import chromadb
import shutil
from sentence_transformers import SentenceTransformer

from agents.deals import ScrapedDeal, feeds

VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "data/vectorstore")
VECTORSTORE_NAME = os.getenv("VECTORSTORE_NAME", "products")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(VECTORSTORE_PATH):
    shutil.rmtree(VECTORSTORE_PATH)

def to_doc(s: ScrapedDeal) -> Dict:
    desc = f"{s.title}. {s.summary} {s.details} {s.features}".strip()
    return {
        "title": s.title,
        "description": desc,
        "url": s.url,
        "price": 0.0,  
    }

def seed_vectorstore(limit_per_feed: int = 3) -> None:
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)

    scraped: List[ScrapedDeal] = []
    try:
        scraped = ScrapedDeal.fetch(show_progress=False, limit_per_feed=limit_per_feed, fetch_page=False)
    except TypeError:
        scraped = ScrapedDeal.fetch(show_progress=False)

    docs = [to_doc(s) for s in scraped][: 5 * limit_per_feed]  # ~15 rekordów max
    if not docs:
        raise RuntimeError("No RSS deals fetched – vectorstore seed aborted.")

    # Embeddingi
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [d["description"] for d in docs]
    embeddings = encoder.encode(texts, normalize_embeddings=True).tolist()

    # Chroma 
    client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    try:
        client.delete_collection(VECTORSTORE_NAME)
    except Exception:
        pass
    col = client.create_collection(VECTORSTORE_NAME)

    ids = [f"deal-{i}" for i in range(len(docs))]
    metadatas = [{"price": d["price"], "url": d["url"], "title": d["title"]} for d in docs]

    col.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"[seed_vectorstore] Seeded {len(docs)} items into Chroma '{VECTORSTORE_NAME}' at '{VECTORSTORE_PATH}'")

if __name__ == "__main__":
    seed_vectorstore()
