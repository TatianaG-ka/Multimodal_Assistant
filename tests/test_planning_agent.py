"""PlanningAgent: orchestration + threshold gating."""

from unittest.mock import MagicMock

import pytest

from agents.deals import DealSelection
from agents.planning_agent import PlanningAgent


@pytest.fixture
def planner(fake_collection, env_clean):
    """Planner with all collaborators mocked. env_clean ensures DEAL_THRESHOLD
    inherits the class default (50.0) instead of leaking from prior tests.
    """
    scanner = MagicMock(name="Scanner")
    ensemble = MagicMock(name="Ensemble")
    messenger = MagicMock(name="Messenger")
    return PlanningAgent(
        fake_collection,
        scanner=scanner,
        ensemble=ensemble,
        messenger=messenger,
    )


def test_plan_returns_none_when_scanner_finds_nothing(planner):
    """Empty selection must short-circuit before pricing — no point burning
    LLM calls on zero deals."""
    planner.scanner.scan.return_value = DealSelection(deals=[])

    result = planner.plan(memory=[])

    assert result is None
    planner.ensemble.price.assert_not_called()
    planner.messenger.alert.assert_not_called()


def test_plan_does_not_alert_when_discount_below_threshold(planner, make_deal):
    """A 'good deal' below threshold (default $50) must NOT trigger an alert.

    Past incidents in this code: the planner once compared to threshold
    inclusively and fired alerts on $0.01 margins. The condition is
    strictly `> threshold` and this test locks that semantic in.
    """
    deal = make_deal(price=100.0)
    planner.scanner.scan.return_value = DealSelection(deals=[deal])
    planner.ensemble.price.return_value = 130.0  # discount = 30 < 50

    result = planner.plan(memory=[])

    assert result is not None
    assert result.discount == 30.0
    planner.messenger.alert.assert_not_called()


def test_plan_alerts_when_discount_clears_threshold(planner, make_deal):
    """Discount > threshold must call messenger.alert exactly once with the best opp."""
    deal = make_deal(price=100.0)
    planner.scanner.scan.return_value = DealSelection(deals=[deal])
    planner.ensemble.price.return_value = 250.0  # discount = 150 > 50

    result = planner.plan(memory=[])

    assert result is not None
    assert result.discount == 150.0
    planner.messenger.alert.assert_called_once_with(result)
