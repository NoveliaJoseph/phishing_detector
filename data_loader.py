"""
data_loader.py
--------------
Handles all data loading and preprocessing for the phishing detector.
Tries the SetFit/enron_spam Hugging Face dataset first; falls back to
the local dataset.csv if the network is unavailable.

Dataset schema (SetFit/enron_spam):
  - subject   : Email subject line (str)
  - message   : Email body (str)
  - label     : 0 = ham (legitimate), 1 = spam (phishing)
  - label_text: "ham" or "spam" (str, not used for training)

Label mapping (consistent with api.py):
  0 → "Legitimate"
  1 → "Phishing"

HF Authentication (optional but recommended for higher rate limits):
  Set the environment variable HF_TOKEN before running:
    Windows PowerShell : $env:HF_TOKEN = "hf_..."
    Windows CMD        : set HF_TOKEN=hf_...
    Linux/macOS        : export HF_TOKEN="hf_..."
  Get your token at: https://huggingface.co/settings/tokens
"""

import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------
# Fallback PyTorch Dataset for local CSV
# ---------------------------------------------------------------
class LocalEmailDataset(Dataset):
    """Simple dataset backed by a local CSV file."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _load_from_huggingface(tokenizer, batch_size, max_samples, max_length, retries=3):
    """
    Attempt to download and tokenize SetFit/enron_spam from Hugging Face.
    Retries up to `retries` times with exponential backoff on network errors.
    """
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN", None)
    if not token:
        print("  [TIP] Set the HF_TOKEN env variable to avoid rate limits:")
        print("        $env:HF_TOKEN = 'hf_...'  (PowerShell)")

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            print(f"  Attempt {attempt}/{retries}: downloading SetFit/enron_spam...")
            dataset = load_dataset("SetFit/enron_spam", token=token)
            break
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  Download failed: {type(e).__name__}: {e}")
            if attempt < retries:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    else:
        raise RuntimeError(
            f"All {retries} download attempts failed. "
            "Check your internet connection or set HF_TOKEN. "
            f"Last error: {last_err}"
        )

    # Optional subset
    if max_samples is not None:
        print(f"  Limiting to {max_samples} samples per split.")
        for split in dataset.keys():
            n = min(max_samples, len(dataset[split]))
            dataset[split] = dataset[split].select(range(n))

    # Combine subject + message for richer signal
    def combine_fields(examples):
        if isinstance(examples["subject"], list):
            combined = [
                (s or "") + "\n" + (m or "")
                for s, m in zip(examples["subject"], examples["message"])
            ]
        else:
            combined = (examples["subject"] or "") + "\n" + (examples["message"] or "")
        return {"combined_text": combined}

    print("  Combining subject + message fields...")
    dataset = dataset.map(combine_fields, batched=True)

    def tokenize_function(examples):
        return tokenizer(
            examples["combined_text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    print("  Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    train_loader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size)
    val_loader   = DataLoader(tokenized_datasets["test"],  shuffle=False, batch_size=batch_size)

    print(f"  HF dataset ready — Train: {len(tokenized_datasets['train'])} | "
          f"Val: {len(tokenized_datasets['test'])} samples")
    return train_loader, val_loader


def _load_from_local_csv(tokenizer, batch_size, max_samples, max_length,
                          csv_path="dataset.csv"):
    """
    Fallback: load from the local dataset.csv (columns: email_text, label).
    Performs an 80/20 train/val split.
    """
    print(f"  Loading fallback dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["email_text", "label"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    if max_samples is not None:
        df = df.head(max_samples)

    split_idx  = int(len(df) * 0.8)
    train_df   = df[:split_idx].reset_index(drop=True)
    val_df     = df[split_idx:].reset_index(drop=True)

    def encode(texts):
        return tokenizer(
            texts.tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    train_enc = encode(train_df["email_text"])
    val_enc   = encode(val_df["email_text"])

    train_dataset = LocalEmailDataset(train_enc, train_df["label"].tolist())
    val_dataset   = LocalEmailDataset(val_enc,   val_df["label"].tolist())

    train_loader = DataLoader(train_dataset, shuffle=True,  batch_size=batch_size)
    val_loader   = DataLoader(val_dataset,   shuffle=False, batch_size=batch_size)

    print(f"  Local dataset ready — Train: {len(train_dataset)} | Val: {len(val_dataset)} samples")
    return train_loader, val_loader


def get_dataloaders(tokenizer, batch_size: int = 8, max_samples: int = None,
                    max_length: int = 128, force_local: bool = False):
    """
    Return (train_loader, val_loader).

    Tries Hugging Face first; falls back to local dataset.csv on any
    network/connectivity failure.

    Args:
        tokenizer   : A HuggingFace tokenizer (e.g. BertTokenizer).
        batch_size  : Samples per batch.
        max_samples : Cap samples per split (None = use all data).
        max_length  : Max token length (default 128).
        force_local : Skip HF and go straight to local CSV.
    """
    if not force_local:
        print("Downloading dataset from Hugging Face (SetFit/enron_spam)...")
        try:
            return _load_from_huggingface(tokenizer, batch_size, max_samples, max_length)
        except RuntimeError as e:
            print(f"\n[WARNING] HF download failed: {e}")
            print("[FALLBACK] Switching to local dataset.csv...\n")

    return _load_from_local_csv(tokenizer, batch_size, max_samples, max_length)


# ---------------------------------------------------------------
# Quick smoke-test: python data_loader.py
# ---------------------------------------------------------------
if __name__ == "__main__":
    from transformers import BertTokenizer

    print("=" * 60)
    print("Running smoke-test with max_samples=100 ...")
    print("=" * 60)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_loader, val_loader = get_dataloaders(tokenizer, batch_size=4, max_samples=100)

    batch = next(iter(train_loader))
    print("\nSample batch keys     :", list(batch.keys()))
    print("input_ids shape       :", batch["input_ids"].shape)
    print("attention_mask shape  :", batch["attention_mask"].shape)
    print("label values          :", batch["label"])
    print("\nSmoke-test PASSED -- all checks OK!")
