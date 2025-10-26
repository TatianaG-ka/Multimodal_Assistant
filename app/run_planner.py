try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs): return False
load_dotenv()

import os
import logging
import chromadb
from agents.planning_agent import PlanningAgent
from tools.seed_vectorstore import seed_vectorstore

def ensure_collection():
    path = os.getenv("VECTORSTORE_PATH", "data/vectorstore")
    name = os.getenv("VECTORSTORE_NAME", "products")
    client = chromadb.PersistentClient(path=path)
    try:
        col = client.get_collection(name)
        if col.count() > 0:
            return col
    except Exception:
        pass

    seed_vectorstore(limit_per_feed=3)

    client = chromadb.PersistentClient(path=path)
    return client.get_collection(name)

if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
    collection = ensure_collection()
    planner = PlanningAgent(collection)
    best = planner.plan(memory=[])
    if best:
        print("\n=== BEST OPPORTUNITY ===")
        print("desc:", best.deal.product_description)
        print("price:", best.deal.price)
        print("estimate:", best.estimate)
        print("discount:", best.discount)
    else:
        print("\nNo opportunity found (pipeline ran).")
