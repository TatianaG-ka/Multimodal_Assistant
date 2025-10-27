from dotenv import load_dotenv; load_dotenv()
import os
from tools.seed_models import seed_models
from app.ui import ensure_collection, build_app

seed_models()
ensure_collection()
demo = build_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
