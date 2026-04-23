import re
from typing import Any, Dict, List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from tqdm import tqdm
from typing_extensions import Self


feeds = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/c238/Automotive/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
    "https://www.dealnews.com/c196/Home-Garden/?rss=1",
]


def extract(html_snippet: str) -> str:
    soup = BeautifulSoup(html_snippet, "html.parser")
    snippet_div = soup.find("div", class_="snippet summary")
    if snippet_div:
        description = snippet_div.get_text(strip=True)
        description = BeautifulSoup(description, "html.parser").get_text()
        description = re.sub("<[^<]+?>", "", description)
        result = description.strip()
    else:
        result = html_snippet
    return result.replace("\n", " ")


def _entry_url(entry: Dict[str, Any]) -> Optional[str]:
    links = entry.get("links") or []
    if links and isinstance(links, list):
        href = links[0].get("href") if isinstance(links[0], dict) else None
        if href:
            return href
    return entry.get("link")


class ScrapedDeal:
    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict[str, Any], fetch_page: bool = True) -> None:
        self.title = entry.get("title", "").strip()
        self.summary = extract(entry.get("summary", "") or "")
        url = _entry_url(entry)
        if not url:
            raise ValueError("RSS entry is missing a usable URL")
        self.url = url

        if not fetch_page:
            self.details = self.summary
            self.features = ""
            return

        try:
            stuff = requests.get(self.url, timeout=10).content
            soup = BeautifulSoup(stuff, "html.parser")
            node = soup.find("div", class_="content-section")
            content = node.get_text() if node else self.summary
            content = content.replace("\nmore", "").replace("\n", " ")
            if "Features" in content:
                self.details, self.features = content.split("Features", 1)
            else:
                self.details = content
                self.features = ""
        except (requests.RequestException, ValueError, AttributeError):
            self.details = self.summary
            self.features = ""

    def __repr__(self) -> str:
        return f"<{self.title}>"

    def describe(self) -> str:
        return (
            f"Title: {self.title}\n"
            f"Details: {self.details.strip()}\n"
            f"Features: {self.features.strip()}\n"
            f"URL: {self.url}"
        )

    @classmethod
    def fetch(
        cls,
        show_progress: bool = False,
        limit_per_feed: int = 3,
        fetch_page: bool = False,
    ) -> List[Self]:
        deals: List[ScrapedDeal] = []
        feed_iter = tqdm(feeds) if show_progress else feeds
        for feed_url in feed_iter:
            try:
                feed = feedparser.parse(feed_url)
            except Exception:
                # feedparser already swallows most errors, but guard anyway.
                continue
            for entry in feed.entries[:limit_per_feed]:
                try:
                    deals.append(cls(entry, fetch_page=fetch_page))
                except (KeyError, ValueError, AttributeError):
                    # Skip malformed entries instead of crashing the whole feed.
                    continue
        return deals


class Deal(BaseModel):
    product_description: str
    price: float = Field(ge=0.0)
    url: str


class DealSelection(BaseModel):
    deals: List[Deal]


class Opportunity(BaseModel):
    deal: Deal
    estimate: float = Field(ge=0.0)
    discount: float = Field(ge=0.0)
