#!/usr/bin/env python3
"""
Download required Hugging Face models for offline use.
Run this script once when you have internet connection.
"""

from sentence_transformers import SentenceTransformer
import os

print("Downloading models to cache...")
print("This may take a while depending on your connection speed.\n")

models = [
    "BAAI/bge-large-en-v1.5",
    "Qwen/Qwen3-Embedding-0.6B", 
    "google/embeddinggemma-300m",
    "sentence-transformers/all-MiniLM-L6-v2"
]

for model_name in models:
    try:
        print(f"Downloading {model_name}...")
        if "google/embeddinggemma" in model_name:
            # Requires HF token
            model = SentenceTransformer(model_name, device="cpu", token=True)
        else:
            model = SentenceTransformer(model_name, device="cpu")
        print(f"✓ {model_name} downloaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}\n")

print("\nDownload complete! Models are cached at:")
print(f"{os.path.expanduser('~/.cache/huggingface/hub/')}")
