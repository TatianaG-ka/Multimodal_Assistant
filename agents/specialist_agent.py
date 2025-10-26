import os
from agents.agent import Agent

class SpecialistAgent(Agent):
    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        self.enabled = os.getenv("USE_SPECIALIST", "false").lower() == "true"
        if not self.enabled:
            self.pricer = None
            self.log("Specialist disabled (USE_SPECIALIST=false)")
            return
        
        try:
            import modal
            Pricer = modal.Cls.from_name("pricer-service", "Pricer")
            self.pricer = Pricer()
            self.log("Specialist ready (Modal)")
        except Exception as e:
            self.enabled = False
            self.pricer = None
            self.log(f"Specialist disabled (Modal error: {e})")

    def price(self, description: str) -> float:
        if not self.enabled or not self.pricer:
            return 0.0
        result = self.pricer.price.remote(description)
        return float(result)
