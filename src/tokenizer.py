"""
tokenizer.py

A minimal, from-scratch tokenizer + vocabulary for the text classifier.

We deliberately use simple whitespace tokenization (lowercased, punctuation
stripped) rather than a pretrained tokenizer (e.g. BPE/WordPiece) — the
assignment asks for the neural network and its tokenization/embedding layer
to be built from scratch, and our vocabulary is small/domain-specific enough
that a simple word-level tokenizer works well and stays interpretable.

Special tokens:
    <PAD> - padding, used to make batches the same length. Always index 0.
    <UNK> - unknown word, used for words not seen often enough during training
            (or never seen at all, e.g. at inference time). Always index 1.
"""

import re
import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Iterable

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def simple_tokenize(text: str) -> List[str]:
    """Lowercase + extract word-like tokens (letters, digits, apostrophes).
    Punctuation like '.', ',', '!' is dropped as it carries little signal for
    this classification task and keeps the vocabulary smaller."""
    return _TOKEN_RE.findall(text.lower())


class Vocabulary:
    """Builds a word -> index mapping from a corpus of texts, and encodes new
    texts into lists of integer token ids."""

    def __init__(self, max_vocab_size: int = 10000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.token_to_idx: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx_to_token: Dict[int, str] = {0: PAD_TOKEN, 1: UNK_TOKEN}

    def build(self, texts: Iterable[str]) -> "Vocabulary":
        counter = Counter()
        for text in texts:
            counter.update(simple_tokenize(text))

        # Keep the most common tokens up to max_vocab_size (minus the 2
        # special tokens already reserved), dropping anything below min_freq.
        most_common = [
            (tok, freq) for tok, freq in counter.most_common()
            if freq >= self.min_freq
        ]
        most_common = most_common[: self.max_vocab_size - len(self.token_to_idx)]

        for tok, _freq in most_common:
            idx = len(self.token_to_idx)
            self.token_to_idx[tok] = idx
            self.idx_to_token[idx] = tok

        return self

    def encode(self, text: str) -> List[int]:
        tokens = simple_tokenize(text)
        unk_idx = self.token_to_idx[UNK_TOKEN]
        return [self.token_to_idx.get(tok, unk_idx) for tok in tokens]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.idx_to_token.get(i, UNK_TOKEN) for i in ids)

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "max_vocab_size": self.max_vocab_size,
                    "min_freq": self.min_freq,
                    "token_to_idx": self.token_to_idx,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(max_vocab_size=data["max_vocab_size"], min_freq=data["min_freq"])
        vocab.token_to_idx = {k: int(v) for k, v in data["token_to_idx"].items()}
        vocab.idx_to_token = {int(v): k for k, v in vocab.token_to_idx.items()}
        return vocab


if __name__ == "__main__":
    # Quick manual sanity check.
    sample_texts = [
        "I would like to request a refund for my last payment.",
        "Congratulations! You've won a free gift, click here to claim it now!",
        "The app crashes every time I try to open the settings page.",
    ]
    vocab = Vocabulary(max_vocab_size=50).build(sample_texts)
    print(f"Vocab size: {len(vocab)}")
    encoded = vocab.encode("I want a refund for a totally unseen product")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {vocab.decode(encoded)}")
