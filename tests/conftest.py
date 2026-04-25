"""Shared fixtures for multimodal-assistant tests.

Two non-obvious choices live here:

1. SentenceTransformer is patched at module-import time for FrontierAgent.
   The real class downloads ~90MB of weights on first instantiation and
   needs network — both unacceptable in unit tests. The stub returns a
   deterministic 1-d vector so cosine math still works downstream.

2. ScannerAgent has no constructor injection for its OpenAI client, so
   tests build the agent and then assign `.openai` directly. This is
   intentional: production code uses module-level `get_llm_clients()`
   rather than DI, and we test what's actually shipped.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from agents.deals import Deal, DealSelection, Opportunity, ScrapedDeal


@pytest.fixture
def fake_collection():
    """Mock Chroma collection. .query() returns configurable docs+prices."""
    coll = MagicMock(name="ChromaCollection")
    coll.query.return_value = {
        "documents": [["similar item 1", "similar item 2", "similar item 3"]],
        "metadatas": [[{"price": 100.0}, {"price": 120.0}, {"price": 80.0}]],
    }
    return coll


@pytest.fixture
def empty_collection():
    """Chroma collection that returns no matches (cold-start scenario)."""
    coll = MagicMock(name="EmptyChromaCollection")
    coll.query.return_value = {"documents": [[]], "metadatas": [[]]}
    return coll


@pytest.fixture
def make_scraped_deal():
    """Factory for ScrapedDeal objects without hitting the RSS network.

    ScrapedDeal.__init__ accepts a feedparser-style entry dict. We synthesize
    just enough of one to satisfy the parser without `fetch_page=True`
    triggering a real HTTP call.
    """

    def _make(
        title: str = "Test Deal",
        url: str = "https://example.com/deal",
        summary: str = "Great product for $99",
    ) -> ScrapedDeal:
        entry: Dict[str, Any] = {
            "title": title,
            "summary": summary,
            "links": [{"href": url}],
        }
        return ScrapedDeal(entry, fetch_page=False)

    return _make


@pytest.fixture
def make_deal():
    """Factory for Deal pydantic models."""

    def _make(
        product_description: str = "A nice product",
        price: float = 50.0,
        url: str = "https://example.com/x",
    ) -> Deal:
        return Deal(product_description=product_description, price=price, url=url)

    return _make


@pytest.fixture
def make_opportunity(make_deal):
    """Factory for Opportunity (used to seed scanner-dedup memory)."""

    def _make(url: str = "https://example.com/seen", **deal_kwargs) -> Opportunity:
        deal = make_deal(url=url, **deal_kwargs)
        return Opportunity(deal=deal, estimate=200.0, discount=150.0)

    return _make


@pytest.fixture
def fake_openai_client():
    """OpenAI client stub. Configure response via .return_value attributes.

    Default behavior: chat.completions.create returns "100"; structured
    parse returns DealSelection with two deals.
    """
    client = MagicMock(name="OpenAIClient")
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="100"))]
    )
    client.beta.chat.completions.parse.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=DealSelection(
                        deals=[
                            Deal(
                                product_description="Parsed deal",
                                price=120.0,
                                url="https://example.com/parsed",
                            )
                        ]
                    )
                )
            )
        ]
    )
    return client


class _StubSentenceTransformer:
    """Drop-in for sentence_transformers.SentenceTransformer.

    Returns deterministic 384-dim zero vectors as a numpy array. The real
    encoder returns np.ndarray and FrontierAgent calls .astype on the
    result, so a plain list breaks the call chain silently. Tests don't
    exercise vector quality — they exercise orchestration around the encoder.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.zeros((len(texts), 384), dtype=float)


@pytest.fixture(autouse=True)
def _patch_sentence_transformer(monkeypatch):
    """Autouse: replace SentenceTransformer everywhere it's imported.

    FrontierAgent imports the real class at module load time. Without this
    patch, every FrontierAgent() in tests downloads weights on first run.
    """
    monkeypatch.setattr(
        "agents.frontier_agent.SentenceTransformer", _StubSentenceTransformer
    )


@pytest.fixture
def env_clean(monkeypatch):
    """Wipe env vars that bleed between tests (DEAL_THRESHOLD, DO_PUSH, ...)."""
    for var in [
        "DEAL_THRESHOLD",
        "DO_PUSH",
        "DO_TEXT",
        "PUSHOVER_TOKEN",
        "PUSHOVER_USER",
        "TWILIO_ACCOUNT_SID",
    ]:
        monkeypatch.delenv(var, raising=False)
