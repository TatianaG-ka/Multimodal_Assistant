import os
import joblib
from agents.agent import Agent

class RandomForestAgent(Agent):
    name = "RandomForest Agent"
    color = Agent.YELLOW

    def __init__(self):
        self.log("RandomForest Agent is initializing")
        path = os.getenv("RF_MODEL_PATH", "data/models/random_forest_model.pkl")
        try:
            self.model = joblib.load(path)
            self.log(f"RandomForest model loaded from {path}")
        except Exception as e:
            self.model = None
            self.log(f"RandomForest model missing ({e}) — will return 0.0")
        self.log("RandomForest Agent is ready")

    def price(self, description: str) -> float:
        if self.model is None:
            return 0.0
        return 0.0
