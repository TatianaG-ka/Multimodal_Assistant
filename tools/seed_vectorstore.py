import os
import chromadb
from sentence_transformers import SentenceTransformer

VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "data/vectorstore")
VECTORSTORE_NAME = os.getenv("VECTORSTORE_NAME", "products")

client = chromadb.PersistentClient(path=VECTORSTORE_PATH)

try:
    col = client.get_collection(VECTORSTORE_NAME)
except Exception:
    col = client.create_collection(VECTORSTORE_NAME)

docs = [
    "Apple AirPods Pro 2nd gen wireless earbuds with ANC",
    "Samsung 27-inch 144Hz gaming monitor 1080p",
    "Roomba robot vacuum cleaner with mapping",
    "Sony WH-1000XM4 noise cancelling headphones",
    "Logitech MX Master 3S wireless mouse"
]
prices = [199.0, 179.0, 249.0, 349.0, 99.0]

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embs = encoder.encode(docs).astype(float).tolist()

ids = [f"p{i}" for i in range(len(docs))]
metas = [{"price": float(p)} for p in prices]

col.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)

print(f"OK: collection '{VECTORSTORE_NAME}' sown in '{VECTORSTORE_PATH}'. "
      f"Number of documents: {len(docs)}")
