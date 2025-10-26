try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs): return False
load_dotenv()

import os
import pandas as pd
import gradio as gr
import chromadb
from agents.planning_agent import PlanningAgent

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

    # brak / pusta kolekcja → seed z RSS
    from tools.seed_vectorstore import seed_vectorstore
    seed_vectorstore(limit_per_feed=3)

    client = chromadb.PersistentClient(path=path)
    return client.get_collection(name)

collection = ensure_collection()
planner = PlanningAgent(collection)

def scan():
    opp = planner.plan(memory=[])
    status = "OK" if opp else "No deals"
    rows = []
    if opp:
        rows.append({
            "product_description": opp.deal.product_description,
            "price": opp.deal.price,
            "url": opp.deal.url,
            "estimate": opp.estimate,
            "discount": opp.discount
        })
    df = pd.DataFrame(rows, columns=["product_description","price","url","estimate","discount"])
    return status, df

with gr.Blocks() as demo:
    gr.Markdown("### Multimodal Agent — scan → rank → notify (minimal UI)")
    with gr.Row():
        scan_btn = gr.Button("Scan", scale=1)
        notify_btn = gr.Button("Notify", scale=1, visible=False)  
    status = gr.Textbox(label="Status")
    table = gr.Dataframe(headers=["product_description","price","url","estimate","discount"], interactive=False)
    scan_btn.click(fn=scan, outputs=[status, table])

demo.launch()

def main():
    gr.Interface(...).launch()