import os
import re
from typing import Optional, List
from openai import OpenAI
from agents.deals import ScrapedDeal, DealSelection, Deal
from agents.agent import Agent

class ScannerAgent(Agent):
    MODEL = "gpt-4o-mini"

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 

{"deals": [
    {
        "product_description": "Your clearly expressed summary of the product in 4-5 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a paragpraph of text for each item you choose.",
        "price": 99.99,
        "url": "the url as provided"
    },
    ...
]}"""

    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
Respond strictly in JSON, and only JSON. You should rephrase the description to be a summary of the product itself, not the terms of the deal.
Remember to respond with a paragraph of text in the product_description field for each of the 5 items that you select.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 

Deals:

"""

    USER_PROMPT_SUFFIX = "\n\nStrictly respond in JSON and include exactly 5 deals, no more."

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self):
        self.log("Scanner Agent is initializing")
        self.openai = OpenAI()
        self.use_llm = os.getenv("SCANNER_USE_LLM", "true").lower() == "true"
        self.log(f"Scanner Agent is ready (USE_LLM={self.use_llm})")

    def fetch_deals(self, memory) -> List[ScrapedDeal]:
        self.log("Scanner Agent is about to fetch deals from RSS feed")
        seen_urls = {opp.deal.url for opp in memory}
        scraped = ScrapedDeal.fetch(limit_per_feed=3, fetch_page=False)
        result = [s for s in scraped if s.url not in seen_urls]
        self.log(f"Scanner Agent received {len(result)} deals not already scraped")
        return result

    def make_user_prompt(self, scraped) -> str:
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += '\n\n'.join([s.describe() for s in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    def _heuristic_price(self, s: ScrapedDeal) -> float:
        text = f"{s.title} {s.summary} {s.details} {s.features}".replace(',', '')
        m = re.search(r"\$?\s*([0-9]+(?:\.[0-9]{1,2})?)", text)
        if m:
            return float(m.group(1))
        return 99.0  # awaryjny fallback

    def scan(self, memory: List[str] = []) -> Optional[DealSelection]:
        scraped = self.fetch_deals(memory)
        if not scraped:
            return None

        if not self.use_llm:
            deals = [Deal(product_description=s.title, price=self._heuristic_price(s), url=s.url)
                     for s in scraped[:5]]
            return DealSelection(deals=deals)

        try:
            user_prompt = self.make_user_prompt(scraped)
            self.log("Scanner Agent is calling OpenAI using Structured Output")
            result = self.openai.beta.chat.completions.parse(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=DealSelection
            )
            parsed = result.choices[0].message.parsed
            parsed.deals = [d for d in parsed.deals if d.price > 0]
            self.log(f"Scanner Agent received {len(parsed.deals)} deals with price>0 from OpenAI")
            return parsed
        except Exception as e:
            self.log(f"Scanner LLM error -> {e}. Falling back to heuristic selection.")
            deals = [Deal(product_description=s.title, price=self._heuristic_price(s), url=s.url)
                     for s in scraped[:5]]
            return DealSelection(deals=deals)
