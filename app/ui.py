try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs): return False
load_dotenv()

import os
import pathlib
import pandas as pd
import gradio as gr
import chromadb
from agents.planning_agent import PlanningAgent
from tools.seed_vectorstore import seed_vectorstore

DEF_PERSIST = os.getenv("PERSIST_DIRECTORY", "/tmp/chroma")
DEF_COLL = os.getenv("VECTORSTORE_NAME", "products")

def ensure_collection():
    path = os.getenv("VECTORSTORE_PATH", DEF_PERSIST)
    name = os.getenv("VECTORSTORE_NAME", DEF_COLL)

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=path)
    try:
        col = client.get_collection(name)
        try:
            if col.count() > 0:
                return col
        except Exception:
            pass
    except Exception:
        pass
    seed_vectorstore(limit_per_feed=3)

    client = chromadb.PersistentClient(path=path)
    return client.get_collection(name)

_collection = None
_planner = None

def get_planner():
    global _collection, _planner
    if _planner is None:
        _collection = ensure_collection()
        _planner = PlanningAgent(_collection)
    return _planner

def scan(limit: int):
 
    try:
        planner = get_planner()
        selection = planner.scanner.scan(memory=[])
        if not selection or not selection.deals:
            empty = pd.DataFrame(columns=["product_description", "price", "url", "estimate", "discount"])
            return "No deals", empty

        n = max(1, int(limit))
        opportunities = [planner.run(deal) for deal in selection.deals[:n]]
        opportunities.sort(key=lambda opp: opp.discount, reverse=True)

        rows = [{
            "product_description": opp.deal.product_description,
            "price": opp.deal.price,
            "url": opp.deal.url,
            "estimate": opp.estimate,
            "discount": opp.discount
        } for opp in opportunities]

        df = pd.DataFrame(rows, columns=["product_description", "price", "url", "estimate", "discount"])
        return "OK", df

    except Exception as e:
        msg = f"Error: {type(e).__name__}: {e}"
        empty = pd.DataFrame(columns=["product_description", "price", "url", "estimate", "discount"])
        return msg, empty


def build_app():
    with gr.Blocks() as demo:
        gr.Markdown("### Multimodal Agent — scan → rank → notify")
        with gr.Row():
            scan_btn = gr.Button("Scan", scale=1)
            limit_slider = gr.Slider(
                label="Number of deals to process",
                minimum=1,
                maximum=20,
                step=1,
                value=5,
                interactive=True
            )

        status = gr.Textbox(label="Status", interactive=False)
        table = gr.Dataframe(
            headers=["product_description", "price", "url", "estimate", "discount"],
            interactive=False
        )

        scan_btn.click(fn=scan, inputs=[limit_slider], outputs=[status, table])
    return demo

def main():
    demo = build_app()
    demo.launch()

if __name__ == "__main__":
    main()
