"""FrontierAgent: similars cache + LLM-disabled fallback to RAG average."""

from agents.frontier_agent import FrontierAgent


def test_find_similars_uses_cache_on_repeat_call(fake_collection):
    """Calling find_similars twice with the same description must hit the
    Chroma collection only once. Without the cache, EnsembleAgent would
    pay for two embeddings + two vector searches per pricing call.
    """
    agent = FrontierAgent(fake_collection)
    description = "Some product"

    docs1, prices1 = agent.find_similars(description)
    docs2, prices2 = agent.find_similars(description)

    assert docs1 == docs2
    assert prices1 == prices2
    fake_collection.query.assert_called_once()


def test_price_falls_back_to_average_when_llm_disabled(fake_collection):
    """USE_LLM=false: returned price is the mean of similar-item prices.

    Locks down the contract in ensemble_agent.py:55 — when the LLM is off,
    Frontier degrades to a mean-of-RAG estimator and EnsembleAgent uses it
    instead of dropping to zero.
    """
    agent = FrontierAgent(fake_collection)
    agent.use_llm = False

    result = agent.price("Some product")

    # fake_collection metadatas: [100, 120, 80] -> mean = 100.
    assert result == 100.0
