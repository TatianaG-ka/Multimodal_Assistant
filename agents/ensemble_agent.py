"""Weighted fusion of two pricing agents (LLM-only + RAG-augmented).

Previous revisions stacked a RandomForest + meta-learner on top of the LLM
outputs, but those models were trained on synthetic noise and added no real
signal. They were removed in favor of an honest 2-agent ensemble.
ML-based meta-learner is tracked as v0.2 roadmap.
"""

from typing import Optional
from openai import APIError, APIConnectionError, APITimeoutError

from .agent import Agent
from .specialist_agent import SpecialistAgent
from .frontier_agent import FrontierAgent


# RAG-augmented Frontier beats context-less Specialist on average,
# so it carries more weight in the fused estimate.
WEIGHT_FRONTIER = 0.6
WEIGHT_SPECIALIST = 0.4

_LLM_ERRORS = (APIError, APIConnectionError, APITimeoutError, ValueError, TimeoutError)


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(
        self,
        collection,
        specialist: Optional[SpecialistAgent] = None,
        frontier: Optional[FrontierAgent] = None,
    ):
        self.log("Ensemble Agent is initializing")
        self.specialist = specialist or SpecialistAgent()
        self.frontier = frontier or FrontierAgent(collection)
        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        """Fused estimate: weighted average of Specialist + Frontier."""
        self.log("Running Ensemble Agent (Specialist + Frontier fusion)")

        try:
            similars, _ = self.frontier.find_similars(description)
        except _LLM_ERRORS as e:
            self.log(f"Frontier similarity lookup failed -> {e!r}")
            similars = []

        try:
            specialist = float(self.specialist.estimate(description, similars))
        except _LLM_ERRORS as e:
            self.log(f"Specialist failed -> {e!r}; dropping from ensemble")
            specialist = 0.0

        try:
            frontier = float(self.frontier.price(description))
        except _LLM_ERRORS as e:
            self.log(f"Frontier failed -> {e!r}; dropping from ensemble")
            frontier = 0.0

        if specialist > 0 and frontier > 0:
            y = WEIGHT_FRONTIER * frontier + WEIGHT_SPECIALIST * specialist
        elif frontier > 0:
            y = frontier
        elif specialist > 0:
            y = specialist
        else:
            y = 0.0

        self.log(f"Ensemble returning ${y:.2f} (specialist={specialist:.2f}, frontier={frontier:.2f})")
        return y
