import os
import re
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from agents.agent import Agent

class FrontierAgent(Agent):
    name = "Frontier Agent"
    color = Agent.BLUE
    MODEL = "gpt-4o-mini"

    def __init__(self, collection):
        self.log("Initializing Frontier Agent")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_api_key:
            self.client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
            self.MODEL = "deepseek-chat"
            self.log("Frontier Agent is set up with DeepSeek")
        else:
            self.client = OpenAI()
            self.MODEL = "gpt-4o-mini"
            self.log("Frontier Agent is setting up with OpenAI")
        self.collection = collection
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.use_llm = os.getenv("FRONTIER_USE_LLM", "true").lower() == "true"
        self.log(f"Frontier Agent is ready (USE_LLM={self.use_llm})")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        msg = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            msg += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return msg

    def messages_for(self, description: str, similars: List[str], prices: List[float]) -> List[Dict[str, str]]:
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\nHow much does this cost?\n\n" + description
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"},
        ]

    def find_similars(self, description: str):
        self.log("Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products")
        vector = self.model.encode([description])
        try:
            results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
            documents = (results.get('documents') or [[]])[0]
            metadatas = (results.get('metadatas') or [[]])[0]
            prices = [m.get('price', 0.0) for m in metadatas]
        except Exception as e:
            self.log(f"Frontier RAG error: {e}")
            documents, prices = [], []
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, s: str) -> float:
        s = s.replace('$', '').replace(',', '')
        m = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(m.group()) if m else 0.0

    def price(self, description: str) -> float:
        documents, prices = self.find_similars(description)

        if not self.use_llm:
            return float(sum(prices) / len(prices)) if prices else 0.0

        self.log(f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products")
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=self.messages_for(description, documents, prices),
            seed=42,
            max_tokens=5
        )
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
