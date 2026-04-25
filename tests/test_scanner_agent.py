"""ScannerAgent: deduplication + heuristic fallback paths."""

from unittest.mock import MagicMock

import pytest
from openai import APIError

from agents.deals import Deal, ScrapedDeal
from agents.scanner_agent import ScannerAgent


@pytest.fixture
def scanner(monkeypatch):
    """Build a ScannerAgent without triggering real OpenAI client init.

    The agent's __init__ touches get_llm_clients() which can return None
    in the test env (no OPENAI_API_KEY); we override use_llm explicitly
    per test to keep behavior deterministic.
    """
    agent = ScannerAgent.__new__(ScannerAgent)
    agent.use_llm = False
    agent.openai = None
    return agent


def test_fetch_deals_filters_already_seen_urls(
    scanner, monkeypatch, make_scraped_deal, make_opportunity
):
    """Re-running the scanner must skip URLs already alerted on.

    Without this, the planning loop would re-alert the same deal every
    cycle and the user would unsubscribe from notifications within a day.
    """
    seen_url = "https://example.com/already-seen"
    new_url = "https://example.com/fresh"
    seen_deal = make_scraped_deal(url=seen_url, title="Seen deal")
    new_deal = make_scraped_deal(url=new_url, title="Fresh deal")

    monkeypatch.setattr(
        ScrapedDeal, "fetch", classmethod(lambda cls, **kw: [seen_deal, new_deal])
    )

    memory = [make_opportunity(url=seen_url)]
    result = scanner.fetch_deals(memory)

    assert len(result) == 1
    assert result[0].url == new_url


def test_scan_returns_heuristic_deals_when_llm_disabled(
    scanner, monkeypatch, make_scraped_deal
):
    """USE_LLM=false must still produce a DealSelection from the heuristic."""
    fake_deals = [
        make_scraped_deal(
            title="Headphones $199",
            url=f"https://example.com/{i}",
            summary=f"Item {i} on sale",
        )
        for i in range(3)
    ]
    monkeypatch.setattr(
        ScrapedDeal, "fetch", classmethod(lambda cls, **kw: fake_deals)
    )

    result = scanner.scan(memory=[])

    assert result is not None
    assert len(result.deals) == 3
    # Heuristic should pick 199 from "$199" in title.
    assert all(isinstance(d, Deal) for d in result.deals)
    assert result.deals[0].price == 199.0


def test_scan_falls_back_to_heuristic_when_openai_raises(
    monkeypatch, make_scraped_deal
):
    """An OpenAI APIError must NOT crash the run — heuristic deals are returned.

    This is the entire point of having APIError in _LLM_ERRORS: the user's
    cron run shouldn't die on transient OpenAI 5xx.
    """
    agent = ScannerAgent.__new__(ScannerAgent)
    agent.use_llm = True
    agent.openai = MagicMock()
    agent.openai.beta.chat.completions.parse.side_effect = APIError(
        message="boom", request=MagicMock(), body=None
    )

    fake_deals = [
        make_scraped_deal(title=f"Deal $50", url=f"https://example.com/{i}")
        for i in range(2)
    ]
    monkeypatch.setattr(
        ScrapedDeal, "fetch", classmethod(lambda cls, **kw: fake_deals)
    )

    result = agent.scan(memory=[])

    assert result is not None
    assert len(result.deals) == 2
    assert all(d.price > 0 for d in result.deals)
