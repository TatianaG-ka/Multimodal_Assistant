import os, pathlib

# move all caches & local data to ephemeral storage 
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf/hub")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/hf/sentencetransformers")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg")
os.environ.pop("TRANSFORMERS_CACHE", None) 

os.environ.setdefault("PERSIST_DIRECTORY", "/tmp/chroma")
os.environ.setdefault("MODELS_LOCAL_DIR", "/tmp/models")

for p in [
    os.environ["HF_HOME"],
    os.environ["HF_HUB_CACHE"],
    os.environ["SENTENCE_TRANSFORMERS_HOME"],
    os.environ["XDG_CACHE_HOME"],
    os.environ["PERSIST_DIRECTORY"],
    os.environ["MODELS_LOCAL_DIR"],
]:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

print("[BOOT] APP_MODE:", os.getenv("APP_MODE", "offline"))
print("[BOOT] LLM_PROVIDER:", os.getenv("LLM_PROVIDER", "openai"))
print("[BOOT] HF_HOME:", os.environ["HF_HOME"])
print("[BOOT] PERSIST_DIRECTORY:", os.environ["PERSIST_DIRECTORY"])
print("[BOOT] MODELS_LOCAL_DIR:", os.environ["MODELS_LOCAL_DIR"])


from dotenv import load_dotenv; load_dotenv()

from app.ui import ensure_collection, build_app

ensure_collection()
demo = build_app()

if __name__ == "__main__":
    is_hf = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID"))
    host = "0.0.0.0" if is_hf else "127.0.0.1"
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name=host, server_port=port)
