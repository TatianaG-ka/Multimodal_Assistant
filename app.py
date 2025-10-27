from dotenv import load_dotenv; load_dotenv()
import os
import tools.seed_models
from app.ui import ensure_collection, build_app

ensure_collection()
demo = build_app()

if __name__ == "__main__":
    is_hf = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID"))
    host = "0.0.0.0" if is_hf else "127.0.0.1"
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name=host, server_port=port)
