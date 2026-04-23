
import os
from dataclasses import dataclass
from typing import Optional

# Central mode switch (offline/online)
APP_MODE = os.getenv("APP_MODE", os.getenv("APP_ENV", "offline")).strip().lower()
USE_LLM = APP_MODE == "online"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()  # openai|deepseek
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Per-agent toggles default to APP_MODE
SCANNER_USE_LLM = os.getenv("SCANNER_USE_LLM", "true" if USE_LLM else "false").lower() == "true"
FRONTIER_USE_LLM = os.getenv("FRONTIER_USE_LLM", "true" if USE_LLM else "false").lower() == "true"

@dataclass
class LLMClients:
    openai: Optional[object] = None
    deepseek: Optional[object] = None

_clients: Optional[LLMClients] = None

def get_llm_clients() -> LLMClients:
    global _clients
    if _clients is not None:
        return _clients
    clients = LLMClients()
    if USE_LLM:
        if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
            try:
                from openai import OpenAI
                clients.openai = OpenAI(api_key=OPENAI_API_KEY)
            except Exception:
                clients.openai = None
        elif LLM_PROVIDER == "deepseek" and DEEPSEEK_API_KEY:
            try:
                from openai import OpenAI as DSClient
                clients.deepseek = DSClient(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            except Exception:
                clients.deepseek = None
    _clients = clients
    return clients
