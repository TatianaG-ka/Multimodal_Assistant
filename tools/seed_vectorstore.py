import os
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from agents.deals import ScrapedDeal, feeds  

DEF_PERSIST = os.getenv("PERSIST_DIRECTORY", "/tmp/chroma")
DEF_NAME = os.getenv("VECTORSTORE_NAME", "products")
DEF_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def to_doc(s: ScrapedDeal) -> Dict:
    desc = f"{s.title}. {s.summary} {s.details} {s.features}".strip()
    return {"title": s.title, "description": desc, "url": s.url, "price": 0.0}

def seed_vectorstore(
    limit_per_feed: int = 3,
    path: str | None = None,
    name: str | None = None,
    reset: bool = False,
) -> None:
    path = path or os.getenv("VECTORSTORE_PATH", DEF_PERSIST)
    name = name or os.getenv("VECTORSTORE_NAME", DEF_NAME)

    os.makedirs(path, exist_ok=True)

    try:
        scraped: List[ScrapedDeal] = ScrapedDeal.fetch(
            show_progress=False, limit_per_feed=limit_per_feed, fetch_page=False
        )
    except TypeError:
        scraped = ScrapedDeal.fetch(show_progress=False)

    docs = [to_doc(s) for s in scraped][: 5 * limit_per_feed] 
    if not docs:
        raise RuntimeError("No RSS deals fetched – vectorstore seed aborted.")

    encoder = SentenceTransformer(DEF_MODEL)
    texts = [d["description"] for d in docs]
    embeddings = encoder.encode(texts, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=path)

    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    col = client.get_or_create_collection(name=name)

    ids = [f"deal-{i}" for i in range(len(docs))]
    metadatas = [{"price": d["price"], "url": d["url"], "title": d["title"]} for d in docs]

    col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    print(f"[seed_vectorstore] Seeded {len(docs)} items into Chroma '{name}' at '{path}'")

if __name__ == "__main__":
    seed_vectorstore(reset=True)
