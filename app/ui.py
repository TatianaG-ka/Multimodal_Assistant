import os
import pathlib
import shutil

import chromadb
import gradio as gr
import pandas as pd

from agents.planning_agent import PlanningAgent
from tools.seed_vectorstore import seed_vectorstore

DEF_PERSIST = os.getenv("PERSIST_DIRECTORY", "/tmp/chroma")
DEF_COLL = os.getenv("VECTORSTORE_NAME", "products")


def ensure_collection():
    path = os.getenv("VECTORSTORE_PATH", DEF_PERSIST)
    name = os.getenv("VECTORSTORE_NAME", DEF_COLL)

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    try:
        client = chromadb.PersistentClient(path=path)
        col = client.get_or_create_collection(name=name)
        if col.count() == 0:
            seed_vectorstore(limit_per_feed=3, path=path, name=name, reset=False)
            col = client.get_or_create_collection(name=name)
        return col
    except KeyError as e:
        # Corrupted Chroma JSON ("_type") - reset directory and reseed demo data.
        if "_type" in str(e):
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path, exist_ok=True)
            seed_vectorstore(limit_per_feed=3, path=path, name=name, reset=True)
            client = chromadb.PersistentClient(path=path)
            return client.get_or_create_collection(name=name)
        seed_vectorstore(limit_per_feed=3, path=path, name=name, reset=True)
        client = chromadb.PersistentClient(path=path)
        return client.get_or_create_collection(name=name)


# Build a single planner eagerly - module-level reload tricks were replaced with
# mode propagation via os.environ read inside the agents themselves.
_collection = ensure_collection()
_planner = PlanningAgent(_collection)


def _resolve_mode(mode: str) -> str:
    mode = (mode or "").strip().lower()
    if mode not in ("offline", "online"):
        mode = os.getenv("APP_MODE", "offline")
    return mode


def scan(mode: str, limit: int, only_positive: bool = False):
    resolved = _resolve_mode(mode)
    # PlanningAgent was built once at module load; changing mode at runtime
    # without process restart only affects env-reading code paths.
    os.environ["APP_MODE"] = resolved

    try:
        selection = _planner.scanner.scan(memory=[])
        if not selection or not selection.deals:
            empty = pd.DataFrame(columns=["product_description", "price", "url", "estimate", "discount"])
            return f"Mode={resolved}: No deals", empty

        n = max(1, int(limit))
        opportunities = [_planner.run(deal) for deal in selection.deals[:n]]

        if only_positive:
            opportunities = [opp for opp in opportunities if opp.discount > 0]

        try:
            threshold = float(os.getenv("DEAL_THRESHOLD", str(PlanningAgent.DEAL_THRESHOLD)))
        except ValueError:
            threshold = float(PlanningAgent.DEAL_THRESHOLD)
        opportunities = [opp for opp in opportunities if opp.discount > threshold]

        opportunities.sort(key=lambda opp: opp.discount, reverse=True)

        rows = [
            {
                "product_description": opp.deal.product_description,
                "price": opp.deal.price,
                "url": opp.deal.url,
                "estimate": opp.estimate,
                "discount": opp.discount,
            }
            for opp in opportunities
        ]

        df = pd.DataFrame(rows, columns=["product_description", "price", "url", "estimate", "discount"])
        return f"Mode={resolved}: {len(rows)} rows", df

    except (ValueError, RuntimeError, KeyError) as e:
        msg = f"Error: {type(e).__name__}: {e}"
        empty = pd.DataFrame(columns=["product_description", "price", "url", "estimate", "discount"])
        return msg, empty


def build_app():
    with gr.Blocks() as demo:
        gr.Markdown("### Multimodal Agent - scan -> rank -> notify")
        with gr.Row():
            mode_dd = gr.Dropdown(
                choices=["offline", "online"],
                value=os.getenv("APP_MODE", "offline"),
                label="Mode",
                interactive=True,
            )
            scan_btn = gr.Button("Scan", scale=1)
            limit_slider = gr.Slider(
                label="Number of deals to process",
                minimum=1,
                maximum=20,
                step=1,
                value=5,
                interactive=True,
            )
        only_positive = gr.Checkbox(label="Show only positive discounts", value=False)

        status = gr.Textbox(label="Status", interactive=False)
        table = gr.Dataframe(
            headers=["product_description", "price", "url", "estimate", "discount"],
            interactive=False,
        )

        scan_btn.click(
            fn=scan, inputs=[mode_dd, limit_slider, only_positive], outputs=[status, table]
        )
    return demo
