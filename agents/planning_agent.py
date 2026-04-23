import os
from typing import List, Optional

from .agent import Agent
from .deals import Deal, Opportunity
from .ensemble_agent import EnsembleAgent
from .messaging_agent import MessagingAgent
from .scanner_agent import ScannerAgent


class PlanningAgent(Agent):
    name = "Planning Agent"
    color = Agent.GREEN

    # Minimum dollar discount required to fire an alert.
    # Keeping it above $0 avoids alerting on trivial (sub-dollar) margins
    # that the LLM estimators will produce by rounding noise.
    DEAL_THRESHOLD = 50.0

    def __init__(
        self,
        collection,
        scanner: Optional[ScannerAgent] = None,
        ensemble: Optional[EnsembleAgent] = None,
        messenger: Optional[MessagingAgent] = None,
    ):
        self.log("Planning Agent is initializing")
        self.scanner = scanner or ScannerAgent()
        self.ensemble = ensemble or EnsembleAgent(collection)
        self.messenger = messenger or MessagingAgent()
        self.log("Planning Agent is ready")

    def run(self, deal: Deal) -> Opportunity:
        self.log("Planning Agent is pricing up a potential deal")
        estimate = self.ensemble.price(deal.product_description)
        discount = max(0.0, estimate - deal.price)
        self.log(f"Planning Agent has processed a deal with discount ${discount:.2f}")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: Optional[List[Opportunity]] = None) -> Optional[Opportunity]:
        """Run one planning cycle. Pass prior opportunities as `memory` to dedupe."""
        memory = memory if memory is not None else []

        self.log("Planning Agent is kicking off a run")
        selection = self.scanner.scan(memory=memory)
        if not selection or not selection.deals:
            self.log("No offers to evaluate - run ended.")
            return None

        opportunities = [self.run(deal) for deal in selection.deals]
        opportunities.sort(key=lambda opp: opp.discount, reverse=True)
        best = opportunities[0]

        try:
            threshold = float(os.getenv("DEAL_THRESHOLD", str(self.DEAL_THRESHOLD)))
        except ValueError:
            self.log("Invalid DEAL_THRESHOLD env var; falling back to default")
            threshold = float(self.DEAL_THRESHOLD)

        self.log(f"The best opportunity: discount ${best.discount:.2f} (threshold=${threshold:.2f})")
        if best.discount > threshold:
            self.messenger.alert(best)
            self.log("Alert sent.")

        self.log("Planning Agent has finished its run (returning best).")
        return best
