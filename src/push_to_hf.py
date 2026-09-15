#!/usr/bin/env python3
"""Upload this package to the Hugging Face Hub. Requires HF_TOKEN."""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, login

REPO_ID = "Midwest-Stratospheric/uogw-aerostratospheric-forecast"
ROOT = Path(__file__).resolve().parents[1]


def main():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "No HF_TOKEN. Create a write token at https://huggingface.co/settings/tokens "
            "and export HF_TOKEN=hf_..."
        )
    login(token=token)
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True, token=token)
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=REPO_ID,
        repo_type="model",
        token=token,
        ignore_patterns=[".git/*", "__pycache__/*", "*.pyc"],
    )
    print(f"https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
