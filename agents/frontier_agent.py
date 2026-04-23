import re
from typing import Dict, List, Tuple

from openai import APIError, APIConnectionError, APITimeoutError
from sentence_transformers import SentenceTransformer

from .agent import Agent
from .config import FRONTIER_USE_LLM, LLM_MODEL, LLM_PROVIDER, get_llm_clients


_LLM_ERRORS = (APIError, APIConnectionError, APITimeoutError, ValueError)


class FrontierAgent(Agent):
    name = "Frontier Agent"
    color = Agent.BLUE

    def __init__(self, collection):
        self.log("Initializing Frontier Agent")
        self.collection = collection
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.use_llm = FRONTIER_USE_LLM
        self.clients = get_llm_clients()
        # Per-call cache so EnsembleAgent can call find_similars() and price()
        # back-to-back without re-embedding the same description twice.
        self._similars_cache: Tuple[str, List[str], List[float]] | None = None
        self.log(f"Frontier Agent is ready (USE_LLM={self.use_llm}, provider={LLM_PROVIDER})")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        msg = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            msg += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return msg

    def messages_for(
        self, description: str, similars: List[str], prices: List[float]
    ) -> List[Dict[str, str]]:
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\nHow much does this cost?\n\n" + description
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"},
        ]

    def find_similars(self, description: str) -> Tuple[List[str], List[float]]:
        if self._similars_cache and self._similars_cache[0] == description:
            _, docs, prices = self._similars_cache
            return docs, prices

        self.log("Frontier Agent is performing a RAG search of the Chroma datastore")
        vector = self.model.encode([description])
        try:
            results = self.collection.query(
                query_embeddings=[vector[0].astype(float).tolist()],
                n_results=5,
            )
            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            prices = [m.get("price", 0.0) for m in metadatas]
        except (AttributeError, KeyError, ValueError) as e:
            self.log(f"Frontier RAG error: {e!r}")
            documents, prices = [], []

        self._similars_cache = (description, documents, prices)
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, s: str) -> float:
        s = s.replace("$", "").replace(",", "")
        m = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(m.group()) if m else 0.0

    def _llm_price(
        self, description: str, documents: List[str], prices: List[float]
    ) -> float:
        messages = self.messages_for(description, documents, prices)

        if LLM_PROVIDER == "openai" and self.clients.openai:
            self.log(f"Frontier Agent is calling OpenAI ({LLM_MODEL}) with context")
            response = self.clients.openai.chat.completions.create(
                model=LLM_MODEL, messages=messages, seed=42, max_tokens=5
            )
            return self.get_price(response.choices[0].message.content or "")

        if LLM_PROVIDER == "deepseek" and self.clients.deepseek:
            self.log("Frontier Agent is calling DeepSeek with context")
            response = self.clients.deepseek.chat.completions.create(
                model="deepseek-chat", messages=messages, seed=42, max_tokens=5
            )
            return self.get_price(response.choices[0].message.content or "")

        self.log("No usable LLM client available; falling back to average of similar prices")
        return float(sum(prices) / len(prices)) if prices else 0.0

    def price(self, description: str) -> float:
        documents, prices = self.find_similars(description)

        if not self.use_llm:
            return float(sum(prices) / len(prices)) if prices else 0.0

        try:
            result = self._llm_price(description, documents, prices)
        except _LLM_ERRORS as e:
            self.log(f"Frontier LLM error: {e!r}; falling back to average of similars")
            return float(sum(prices) / len(prices)) if prices else 0.0

        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
