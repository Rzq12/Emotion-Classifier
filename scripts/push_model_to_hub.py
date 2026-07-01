"""Upload the fine-tuned IndoBERT model to a Hugging Face Hub model repo.

The Space then loads it by setting ``MODEL_DIR=<username>/<repo>``.

Usage:
    huggingface-cli login            # atau set env HF_TOKEN
    python -m scripts.push_model_to_hub --repo username/indo-emotion-indobert
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Push model folder to HF Hub.")
    parser.add_argument("--repo", required=True, help="Target repo id, mis. user/indo-emotion-indobert")
    parser.add_argument("--model-dir", default="artifacts/indobert", help="Folder model lokal.")
    parser.add_argument("--private", action="store_true", help="Buat repo privat.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"Model tidak ditemukan di '{model_dir}'. Jalankan training dulu.")

    from huggingface_hub import HfApi

    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(model_dir), repo_id=args.repo, repo_type="model")
    print(f"[push_model_to_hub] uploaded '{model_dir}' -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
