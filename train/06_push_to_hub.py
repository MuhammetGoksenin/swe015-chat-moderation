"""Step 6 - upload models/ to the Hugging Face Hub so the repository stays free of weight files (R7).

Usage:  HF_REPO_ID=<your-hf-username>/swe015-chat-moderation-models  python train/06_push_to_hub.py
Needs a write token: `huggingface-cli login` (or the HF_TOKEN environment variable / Colab secret).
"""
import _common  # noqa: F401
from huggingface_hub import HfApi

from src import config

if not config.HF_REPO_ID:
    raise SystemExit("Set HF_REPO_ID, e.g. HF_REPO_ID=myname/swe015-chat-moderation-models")
api = HfApi()
api.create_repo(config.HF_REPO_ID, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=str(config.MODELS_DIR), repo_id=config.HF_REPO_ID, repo_type="model",
                  commit_message="Upload trained models")
print(f"Uploaded {config.MODELS_DIR} -> https://huggingface.co/{config.HF_REPO_ID}")
