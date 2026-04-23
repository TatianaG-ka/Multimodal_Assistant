import re
from openai import APIError, APIConnectionError, APITimeoutError

from .agent import Agent
from .config import (
    FRONTIER_USE_LLM,
    LLM_MODEL,
    LLM_PROVIDER,
    SCANNER_USE_LLM,
    get_llm_clients,
)

_LLM_ERRORS = (APIError, APIConnectionError, APITimeoutError, ValueError, TimeoutError)


class SpecialistAgent(Agent):
    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        self.log("Specialist Agent is initializing")
        self.use_llm = SCANNER_USE_LLM or FRONTIER_USE_LLM
        self.provider = LLM_PROVIDER
        self.model = LLM_MODEL
        self.clients = get_llm_clients()
        self.log(f"Specialist Agent is ready (USE_LLM={self.use_llm}, provider={self.provider})")

    def estimate(self, description: str, similars: list[str]) -> float:
        """Context-less price estimate via LLM, with heuristic fallback."""
        if not self.use_llm:
            return self._heuristic(description, similars)

        try:
            price_str = self._call_llm(self._build_prompt(description, similars))
            return float(price_str)
        except _LLM_ERRORS as e:
            self.log(f"LLM estimate failed ({e!r}); falling back to heuristic")
            return self._heuristic(description, similars)

    def _heuristic(self, description: str, similars: list[str]) -> float:
        """Fallback used when the LLM is unavailable.

        Strategy: average numeric prices from similar items if we have any,
        otherwise parse the first dollar amount out of the description.
        """
        similar_prices: list[float] = []
        for s in similars or []:
            m = re.search(r"\$\s*([0-9]+(?:\.[0-9]{1,2})?)", s)
            if m:
                similar_prices.append(float(m.group(1)))
        if similar_prices:
            return sum(similar_prices) / len(similar_prices)

        m = re.search(r"\$\s*([0-9]+(?:\.[0-9]{1,2})?)", description)
        return float(m.group(1)) if m else 0.0

    def _build_prompt(self, description: str, similars: list[str]) -> str:
        ctx = "\n".join(f"- {s}" for s in similars[:3])
        return (
            "You are pricing expert. Based on the item and similar items below, "
            "return a single number - a fair market price in USD (no text).\n\n"
            f"Item: {description}\nSimilar:\n{ctx}\n\nPrice:"
        )

    def _call_llm(self, prompt: str) -> str:
        if self.provider == "openai" and self.clients.openai:
            resp = self.clients.openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return (resp.choices[0].message.content or "").strip()
        if self.provider == "deepseek" and self.clients.deepseek:
            resp = self.clients.deepseek.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return (resp.choices[0].message.content or "").strip()
        raise ValueError("No LLM client available")
