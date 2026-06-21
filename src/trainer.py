"""
trainer.py

Trainer class that owns the training loop: runs epochs over the training set,
evaluates on the validation set after each epoch, tracks loss/accuracy history,
and saves the best-performing model checkpoint (by validation accuracy).
"""

import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device = None,
        learning_rate: float = 1e-3,
        checkpoint_path: str = "checkpoints/best_model.pt",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.checkpoint_path = checkpoint_path
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.best_val_acc = 0.0

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple:
        """Runs one pass over `loader`. If train=True, updates model weights.
        Returns (average_loss, accuracy) for this pass."""
        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                logits = self.model(x)
                loss = self.criterion(logits, y)

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self, num_epochs: int = 10, verbose: bool = True) -> Dict[str, List[float]]:
        for epoch in range(1, num_epochs + 1):
            start = time.time()

            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc)

            elapsed = time.time() - start
            if verbose:
                print(
                    f"Epoch {epoch:2d}/{num_epochs} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                    f"{elapsed:.1f}s"
                )

        if verbose:
            print(f"\nBest validation accuracy: {self.best_val_acc:.4f} "
                  f"(checkpoint saved to {self.checkpoint_path})")

        return self.history

    def _save_checkpoint(self, epoch: int, val_acc: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "val_acc": val_acc,
            },
            self.checkpoint_path,
        )


if __name__ == "__main__":
    # Quick end-to-end sanity check using the real processed dataset, but only
    # 2 epochs on a small slice, just to confirm the training loop runs without
    # errors before committing to a full training run via scripts/train.py.
    import sys
    sys.path.insert(0, ".")
    from src.tokenizer import Vocabulary
    from src.dataset import TicketDataset, LabelEncoder
    from src.model import MeanPoolingTextClassifier
    import pandas as pd

    train_df = pd.read_csv("data/processed/train.csv")
    vocab = Vocabulary(max_vocab_size=10000, min_freq=2).build(train_df["text"].tolist())
    label_enc = LabelEncoder()

    train_ds = TicketDataset("data/processed/train.csv", vocab, label_enc, max_len=32)
    val_ds = TicketDataset("data/processed/val.csv", vocab, label_enc, max_len=32)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = MeanPoolingTextClassifier(vocab_size=len(vocab), num_classes=label_enc.num_classes())

    trainer = Trainer(model, train_loader, val_loader, checkpoint_path="checkpoints/sanity_check.pt")
    trainer.train(num_epochs=2)