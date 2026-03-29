"""
Repository-relative defaults for MultiModalQA assets and logs.

Override with environment variables (no secrets here).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def mmqa_dataset_dir() -> Path:
    return Path(
        os.environ.get(
            "MMQA_DATASET_DIR",
            REPO_ROOT / "data" / "multimodalqa" / "dataset",
        )
    )


def mmqa_file(filename: str) -> str:
    return str(mmqa_dataset_dir() / filename)


def mmqa_images_dir() -> str:
    return str(
        Path(
            os.environ.get(
                "MMQA_IMAGES_DIR",
                REPO_ROOT / "data" / "multimodalqa" / "final_dataset_images",
            )
        )
    )


def default_log_dir() -> Path:
    return Path(os.environ.get("MMGRAPHRAG_LOG_DIR", REPO_ROOT / "logs"))


def default_colembed_model_ref() -> str:
    return (
        os.getenv("COLEMBED_MODEL_PATH", "").strip()
        or os.getenv("COLEMBED_MODEL_ID", "").strip()
        or str(REPO_ROOT / "models" / "retriever" / "llama-nemotron-colembed-vl-3b-v2")
    )


def mmqa_embedding_dir() -> str:
    return str(
        Path(
            os.environ.get(
                "MMQA_EMBEDDING_DIR",
                REPO_ROOT / "data" / "multimodalqa" / "embedding",
            )
        )
    )


def mmqa_image_description_dir() -> str:
    """Optional per-image caption .txt files (one file per image id)."""
    return str(
        Path(
            os.environ.get(
                "MMQA_IMAGE_DESCRIPTION_DIR",
                REPO_ROOT / "data" / "multimodalqa" / "image_description",
            )
        )
    )
