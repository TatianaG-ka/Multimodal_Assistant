import os, chromadb

path = os.getenv("VECTORSTORE_PATH", "data/vectorstore")
name = os.getenv("VECTORSTORE_NAME", "products")

client = chromadb.PersistentClient(path=path)
col = client.get_collection(name)

res = col.query(query_texts=["test"], n_results=3)
print("OK: collection available. Keys:", list(res.keys()))
print("n_results returned:", len(res.get("ids", [[]])[0]))
