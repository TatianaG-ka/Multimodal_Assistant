
import os, chromadb
path = os.getenv("VECTORSTORE_PATH", os.getenv("PERSIST_DIRECTORY", "/tmp/chroma"))
name = os.getenv("VECTORSTORE_NAME", "products")
client = chromadb.PersistentClient(path=path)
try:
    col = client.get_collection(name)
    print("Collection:", name, "count:", col.count())
except Exception as e:
    print("Missing or empty collection:", e)
