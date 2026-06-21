"""
dataset.py

PyTorch Dataset for the 3-class ticket classifier (Sales Inquiry / Complaint / Spam),
plus a small LabelEncoder helper to keep label<->index mapping consistent between
training and inference.
"""

import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import Vocabulary


class LabelEncoder:
    """Maps class name strings (e.g. 'Spam') to integer ids and back.
    The mapping order is fixed explicitly (not inferred from data) so it stays
    identical across training, evaluation, and inference."""

    CLASSES = ["Sales Inquiry", "Complaint", "Spam"]

    def __init__(self):
        self.label_to_idx = {label: i for i, label in enumerate(self.CLASSES)}
        self.idx_to_label = {i: label for i, label in enumerate(self.CLASSES)}

    def encode(self, label: str) -> int:
        return self.label_to_idx[label]

    def decode(self, idx: int) -> str:
        return self.idx_to_label[idx]

    def num_classes(self) -> int:
        return len(self.CLASSES)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"classes": self.CLASSES}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "LabelEncoder":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        enc = cls()
        enc.CLASSES = data["classes"]
        enc.label_to_idx = {label: i for i, label in enumerate(enc.CLASSES)}
        enc.idx_to_label = {i: label for i, label in enumerate(enc.CLASSES)}
        return enc


class TicketDataset(Dataset):
    """Wraps a processed CSV (columns: text, category, label) into
    (token_id_tensor, label_id) pairs, padded/truncated to a fixed length."""

    def __init__(
        self,
        csv_path: str,
        vocab: Vocabulary,
        label_encoder: LabelEncoder,
        max_len: int = 32,
    ):
        self.df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.label_encoder = label_encoder
        self.max_len = max_len

        self.encoded_texts: List[List[int]] = [
            self._encode_and_pad(text) for text in self.df["text"].tolist()
        ]
        self.encoded_labels: List[int] = [
            self.label_encoder.encode(label) for label in self.df["label"].tolist()
        ]

    def _encode_and_pad(self, text: str) -> List[int]:
        ids = self.vocab.encode(text)
        pad_idx = self.vocab.token_to_idx["<PAD>"]
        if len(ids) >= self.max_len:
            return ids[: self.max_len]
        return ids + [pad_idx] * (self.max_len - len(ids))

    def __len__(self) -> int:
        return len(self.encoded_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.encoded_texts[idx], dtype=torch.long)
        y = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        return x, y


if __name__ == "__main__":
    # Quick manual sanity check using the actual processed train split.
    train_df = pd.read_csv("data/processed/train.csv")
    vocab = Vocabulary(max_vocab_size=10000, min_freq=2).build(train_df["text"].tolist())
    label_enc = LabelEncoder()

    ds = TicketDataset("data/processed/train.csv", vocab, label_enc, max_len=32)
    print(f"Dataset size: {len(ds)}")
    x, y = ds[0]
    print(f"Sample x shape: {x.shape}, dtype: {x.dtype}")
    print(f"Sample y: {y.item()} -> {label_enc.decode(y.item())}")
    print(f"Decoded tokens: {vocab.decode(x.tolist())}")
