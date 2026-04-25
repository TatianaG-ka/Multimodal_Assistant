"""EnsembleAgent: weighted fusion + degradation when one estimator fails."""

from unittest.mock import MagicMock

import pytest
from openai import APIError

from agents.ensemble_agent import EnsembleAgent, WEIGHT_FRONTIER, WEIGHT_SPECIALIST


@pytest.fixture
def ensemble(fake_collection):
    """Ensemble with both estimators mocked. find_similars stub avoids
    triggering the real Frontier RAG path inside the test."""
    specialist = MagicMock(name="Specialist")
    frontier = MagicMock(name="Frontier")
    frontier.find_similars.return_value = (["doc1"], [100.0])
    return EnsembleAgent(fake_collection, specialist=specialist, frontier=frontier)


def test_price_weighted_fusion_when_both_estimators_succeed(ensemble):
    """Locks down the 0.6/0.4 fusion math.

    Constants live in ensemble_agent.py and have business meaning (RAG-
    augmented Frontier > context-less Specialist). If a future refactor
    changes the weights, this test forces the author to update both the
    constants AND the lock-down — i.e. acknowledge the change.
    """
    ensemble.frontier.price.return_value = 100.0
    ensemble.specialist.estimate.return_value = 50.0

    result = ensemble.price("anything")

    expected = WEIGHT_FRONTIER * 100.0 + WEIGHT_SPECIALIST * 50.0  # 80.0
    assert result == pytest.approx(expected)


def test_price_falls_back_to_frontier_when_specialist_raises(ensemble):
    """Specialist outage must NOT zero out the ensemble — Frontier alone wins."""
    ensemble.specialist.estimate.side_effect = APIError(
        message="boom", request=MagicMock(), body=None
    )
    ensemble.frontier.price.return_value = 200.0

    result = ensemble.price("anything")

    # Specialist failed -> specialist=0, frontier=200 -> ensemble returns 200.
    assert result == 200.0


def test_price_returns_zero_when_both_estimators_fail(ensemble):
    """If everything is down we must return 0.0, not raise. The planner
    treats 0.0 as a negative signal (no positive discount possible) and
    silently skips the alert — which is the correct behavior in an outage.
    """
    ensemble.specialist.estimate.side_effect = APIError(
        message="s_boom", request=MagicMock(), body=None
    )
    ensemble.frontier.price.side_effect = APIError(
        message="f_boom", request=MagicMock(), body=None
    )

    result = ensemble.price("anything")

    assert result == 0.0
