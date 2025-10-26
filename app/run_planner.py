from dotenv import load_dotenv
load_dotenv()  

import os
import logging
import chromadb
from agents.planning_agent import PlanningAgent

def get_collection():
    path = os.getenv("VECTORSTORE_PATH", "data/vectorstore")
    name = os.getenv("VECTORSTORE_NAME", "products")
    client = chromadb.PersistentClient(path=path)
    try:
        col = client.get_collection(name)
    except Exception:
        col = client.create_collection(name)
    return col

if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
    collection = get_collection()
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
