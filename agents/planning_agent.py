import os
from typing import Optional, List
from agents.agent import Agent
from agents.deals import Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent

class PlanningAgent(Agent):
    name = "Planning Agent"
    color = Agent.GREEN
    DEAL_THRESHOLD = 0  # demo

    def __init__(self, collection):
        self.log("Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.log("Planning Agent is ready")

    def run(self, deal: Deal) -> Opportunity:
        self.log("Planning Agent is pricing up a potential deal")
        estimate = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price
        self.log(f"Planning Agent has processed a deal with discount ${discount:.2f}")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: List[str] = []) -> Optional[Opportunity]:
        self.log("Planning Agent is kicking off a run")
        selection = self.scanner.scan(memory=memory)
        if not selection or not selection.deals:
            self.log("No offers to evaluate - run ended.")
            return None

        opportunities = [self.run(deal) for deal in selection.deals[:5]]
        opportunities.sort(key=lambda opp: opp.discount, reverse=True)
        best = opportunities[0]

        try:
            threshold = float(os.getenv("DEAL_THRESHOLD", str(self.DEAL_THRESHOLD)))
        except Exception:
            threshold = float(self.DEAL_THRESHOLD)

        self.log(f"The best opportunity: discount ${best.discount:.2f} (threshold=${threshold:.2f})")
        if best.discount > threshold:
            self.messenger.alert(best)
            self.log("Alert sent.")

        self.log("Planning Agent has finished its run (returning best).")
        return best
