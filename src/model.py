"""
model.py

A lightweight, from-scratch text classifier built with PyTorch:

    token ids --> Embedding --> masked mean pooling --> Dropout --> Linear --> logits

Design rationale:
  - Embedding: learns a dense vector representation for each vocabulary word
    (random-initialized, trained from scratch alongside the classifier - no
    pretrained word vectors are used, per the assignment's "from scratch" requirement).
  - Masked mean pooling: averages the embeddings of all REAL tokens in a sentence
    into a single fixed-size vector, ignoring <PAD> positions (otherwise padding
    would dilute the average, especially for short sentences in a fixed max_len).
    This is a deliberately simple alternative to an RNN/LSTM - it is fast to train,
    has very few parameters, and works well when the classes are separated by
    word choice (vocabulary) rather than by word order, which fits this dataset.
  - Linear classifier head: a small Dropout + Linear layer maps the pooled sentence
    vector to class logits (3 classes: Sales Inquiry / Complaint / Spam).
"""

import torch
import torch.nn as nn


class MeanPoolingTextClassifier(nn.Module):
    """Embedding -> masked mean pooling -> Dropout -> Linear classifier."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_classes: int = 3,
        pad_idx: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,  # gradient for the <PAD> row is always zeroed
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: LongTensor of shape (batch_size, seq_len), containing token ids
           (already padded with self.pad_idx where needed).
        returns: FloatTensor of shape (batch_size, num_classes) - raw logits
                 (no softmax here; loss function will handle that).
        """
        # mask: 1.0 for real tokens, 0.0 for <PAD> positions. Shape: (batch, seq_len)
        mask = (x != self.pad_idx).float()

        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Zero out padding embeddings before summing, then divide by the
        # number of REAL tokens per sentence (not seq_len) to get a true mean.
        mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        summed = (embedded * mask_expanded).sum(dim=1)  # (batch, embed_dim)
        token_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # avoid div-by-0
        pooled = summed / token_counts  # (batch, embed_dim)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (batch, num_classes)
        return logits


if __name__ == "__main__":
    # Quick manual sanity check with random/fake data.
    batch_size, seq_len, vocab_size, num_classes = 4, 10, 100, 3

    model = MeanPoolingTextClassifier(
        vocab_size=vocab_size, embed_dim=64, num_classes=num_classes
    )

    fake_input = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    # simulate some padding at the end of each sequence
    fake_input[:, 7:] = 0  # pad_idx = 0

    logits = model(fake_input)
    print(f"Input shape: {fake_input.shape}")
    print(f"Output logits shape: {logits.shape}")  # expect (4, 3)
    print(f"Sample logits:\n{logits}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {num_params:,}")