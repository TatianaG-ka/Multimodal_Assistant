import os
import pandas as pd
import joblib
from agents.agent import Agent
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.random_forest_agent import RandomForestAgent

class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection):
        self.log("Ensemble Agent is initializing")
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.random_forest = RandomForestAgent()
        path = os.getenv("ENSEMBLE_MODEL_PATH", "data/models/ensemble_model.pkl")
        try:
            self.model = joblib.load(path)
            self.log(f"Ensemble model loaded from {path}")
        except Exception as e:
            self.model = None
            self.log(f"Ensemble model missing ({e}) — will fallback to average")
        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        self.log("Running Ensemble Agent - collaborating with specialist, frontier and random forest agents")
        specialist = self.specialist.price(description)
        frontier = self.frontier.price(description)
        random_forest = self.random_forest.price(description)

        if self.model:
            X = pd.DataFrame({
                'Specialist': [specialist],
                'Frontier': [frontier],
                'RandomForest': [random_forest],
                'Min': [min(specialist, frontier, random_forest)],
                'Max': [max(specialist, frontier, random_forest)],
            })
            y = max(0.0, float(self.model.predict(X)[0]))
            self.log(f"Ensemble Agent complete - returning ${y:.2f}")
            return y

        vals = [v for v in [specialist, frontier, random_forest] if v > 0]
        y = float(sum(vals) / len(vals)) if vals else 0.0
        self.log(f"Ensemble Agent (fallback avg) - returning ${y:.2f}")
        return y
