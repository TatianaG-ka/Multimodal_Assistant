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

# GPT-4o-mini public pricing (USD per 1M tokens). Numbers are deliberately
# approximations: Scanner sees variable RSS payloads, Frontier output is
# capped at ~5 tokens by max_tokens. Treat the displayed cost as a per-scan
# estimate ("am I burning $0.001 or $0.10 per scan?"), not a billing figure.
_PRICE_INPUT_PER_1M = 0.15
_PRICE_OUTPUT_PER_1M = 0.60

# Avg (input, output) tokens per LLM call type.
_TOKENS_SCANNER = (4000, 700)    # 1x per scan, batches ~5 RSS deals
_TOKENS_FRONTIER = (500, 5)      # 1x per processed deal, max_tokens=5
_TOKENS_SPECIALIST = (300, 8)    # 1x per processed deal, temp=0


def _estimate_cost(mode: str, n_processed: int) -> float:
    if mode != "online" or n_processed <= 0:
        return 0.0
    def call_cost(t: tuple[int, int]) -> float:
        return t[0] * _PRICE_INPUT_PER_1M / 1e6 + t[1] * _PRICE_OUTPUT_PER_1M / 1e6
    return call_cost(_TOKENS_SCANNER) + n_processed * (
        call_cost(_TOKENS_FRONTIER) + call_cost(_TOKENS_SPECIALIST)
    )


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


_DEAL_COLS = ["product_description", "price", "url", "estimate", "discount"]
_ALERT_COLS = ["product_description", "price", "estimate", "discount"]


def scan(mode: str, limit: int, only_positive: bool = False):
    resolved = _resolve_mode(mode)
    # PlanningAgent was built once at module load; changing mode at runtime
    # without process restart only affects env-reading code paths.
    os.environ["APP_MODE"] = resolved

    empty_deals = pd.DataFrame(columns=_DEAL_COLS)
    empty_alerts = pd.DataFrame(columns=_ALERT_COLS)

    try:
        selection = _planner.scanner.scan(memory=[])
        if not selection or not selection.deals:
            return f"Mode={resolved}: No deals", empty_deals, empty_alerts

        n = max(1, int(limit))
        processed = selection.deals[:n]
        opportunities = [_planner.run(deal) for deal in processed]

        if only_positive:
            opportunities = [opp for opp in opportunities if opp.discount > 0]

        opportunities.sort(key=lambda opp: opp.discount, reverse=True)

        try:
            threshold = float(os.getenv("DEAL_THRESHOLD", str(PlanningAgent.DEAL_THRESHOLD)))
        except ValueError:
            threshold = float(PlanningAgent.DEAL_THRESHOLD)

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
        alert_rows = [
            {k: r[k] for k in _ALERT_COLS}
            for r in rows
            if r["discount"] > threshold
        ]

        deals_df = pd.DataFrame(rows, columns=_DEAL_COLS)
        alerts_df = pd.DataFrame(alert_rows, columns=_ALERT_COLS)

        cost = _estimate_cost(resolved, len(processed))
        if cost > 0:
            n_calls = 1 + 2 * len(processed)
            cost_str = f" | est. ~${cost:.4f} ({n_calls} LLM calls @ gpt-4o-mini)"
        else:
            cost_str = " | $0 (heuristic, no LLM)"

        funnel = f"{len(rows)} evaluated → {len(alert_rows)} alert{'s' if len(alert_rows) != 1 else ''}"
        return f"Mode={resolved}: {funnel}{cost_str}", deals_df, alerts_df

    except (ValueError, RuntimeError, KeyError) as e:
        msg = f"Error: {type(e).__name__}: {e}"
        return msg, empty_deals, empty_alerts


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

        gr.Markdown("#### All evaluated deals (sorted by discount)")
        deals_table = gr.Dataframe(headers=_DEAL_COLS, interactive=False)

        gr.Markdown(
            "#### 🔔 Alerts (discount > $50)  \n"
            "*In production this same list fires Pushover push + Twilio SMS via `MessagingAgent`. "
            "Rendered in-UI here so the channel-agnostic decision logic is visible without external auth.*"
        )
        alerts_table = gr.Dataframe(headers=_ALERT_COLS, interactive=False)

        scan_btn.click(
            fn=scan,
            inputs=[mode_dd, limit_slider, only_positive],
            outputs=[status, deals_table, alerts_table],
        )
    return demo
