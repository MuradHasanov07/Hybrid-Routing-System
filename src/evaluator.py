"""
evaluator.py

Evaluator class that runs a trained model over a test DataLoader and computes
the metrics required by the assignment: accuracy, precision, recall, F1 score
(per-class and averaged), plus a confusion matrix - saved both as a PNG plot
and a plain-text report.
"""

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, just save PNGs
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: List[str],
        device: torch.device = None,
    ):
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _collect_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                logits = self.model(x)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        return np.array(all_labels), np.array(all_preds)

    def evaluate(self, save_dir: str = "outputs") -> dict:
        y_true, y_pred = self._collect_predictions()
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(self.class_names)), zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        report_text = classification_report(
            y_true, y_pred, target_names=self.class_names, zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.class_names)))
        self._plot_confusion_matrix(cm, save_path / "confusion_matrix.png")
        self._save_text_report(
            accuracy, macro_p, macro_r, macro_f1, report_text, save_path / "evaluation_report.txt"
        )

        results = {
            "accuracy": accuracy,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "per_class_precision": dict(zip(self.class_names, precision)),
            "per_class_recall": dict(zip(self.class_names, recall)),
            "per_class_f1": dict(zip(self.class_names, f1)),
            "confusion_matrix": cm,
        }
        return results

    def _plot_confusion_matrix(self, cm: np.ndarray, save_to: Path) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")

        ax.set_xticks(range(len(self.class_names)))
        ax.set_yticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.class_names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix")

        # annotate each cell with its count
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.close(fig)

    def _save_text_report(
        self, accuracy, macro_p, macro_r, macro_f1, sklearn_report, save_to: Path
    ) -> None:
        with open(save_to, "w", encoding="utf-8") as f:
            f.write("EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n")
            f.write(f"Macro Precision:  {macro_p:.4f}\n")
            f.write(f"Macro Recall:     {macro_r:.4f}\n")
            f.write(f"Macro F1 Score:   {macro_f1:.4f}\n\n")
            f.write("Per-class breakdown (sklearn classification_report):\n")
            f.write("-" * 50 + "\n")
            f.write(sklearn_report)
            f.write("\n")


def plot_training_history(history: dict, save_to: str = "outputs/training_history.png") -> None:
    """Plots train/val loss and train/val accuracy curves side by side from a
    Trainer.history dict (keys: train_loss, train_acc, val_loss, val_acc)."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # End-to-end sanity check: trains briefly, then evaluates on the test set.
    import sys
    sys.path.insert(0, ".")
    from src.tokenizer import Vocabulary
    from src.dataset import TicketDataset, LabelEncoder
    from src.model import MeanPoolingTextClassifier
    from src.trainer import Trainer
    import pandas as pd

    train_df = pd.read_csv("data/processed/train.csv")
    vocab = Vocabulary(max_vocab_size=10000, min_freq=2).build(train_df["text"].tolist())
    label_enc = LabelEncoder()

    train_ds = TicketDataset("data/processed/train.csv", vocab, label_enc, max_len=32)
    val_ds = TicketDataset("data/processed/val.csv", vocab, label_enc, max_len=32)
    test_ds = TicketDataset("data/processed/test.csv", vocab, label_enc, max_len=32)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = MeanPoolingTextClassifier(vocab_size=len(vocab), num_classes=label_enc.num_classes())
    trainer = Trainer(model, train_loader, val_loader, checkpoint_path="checkpoints/sanity_check.pt")
    trainer.train(num_epochs=3, verbose=True)

    evaluator = Evaluator(model, test_loader, class_names=label_enc.CLASSES)
    results = evaluator.evaluate(save_dir="outputs")

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print("Saved confusion_matrix.png and evaluation_report.txt to outputs/")