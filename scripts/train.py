"""
train.py

Main training entry point. Builds the vocabulary from the training set, trains
the classifier, evaluates it on the test set, and saves everything needed for
later inference (model checkpoint, vocabulary, label encoder) into checkpoints/.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 15 --batch_size 128 --embed_dim 64
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokenizer import Vocabulary
from dataset import TicketDataset, LabelEncoder
from model import MeanPoolingTextClassifier
from trainer import Trainer
from evaluator import Evaluator, plot_training_history


def main():
    parser = argparse.ArgumentParser(description="Train the hybrid routing classifier.")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--max_vocab_size", type=int, default=10000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv(data_dir / "train.csv")

    print("Building vocabulary from training data...")
    vocab = Vocabulary(max_vocab_size=args.max_vocab_size, min_freq=args.min_freq)
    vocab.build(train_df["text"].tolist())
    print(f"Vocabulary size: {len(vocab)}")

    label_encoder = LabelEncoder()

    train_ds = TicketDataset(data_dir / "train.csv", vocab, label_encoder, max_len=args.max_len)
    val_ds = TicketDataset(data_dir / "val.csv", vocab, label_encoder, max_len=args.max_len)
    test_ds = TicketDataset(data_dir / "test.csv", vocab, label_encoder, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"\nTrain size: {len(train_ds)} | Val size: {len(val_ds)} | Test size: {len(test_ds)}")

    model = MeanPoolingTextClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_classes=label_encoder.num_classes(),
        dropout=args.dropout,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}\n")

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        learning_rate=args.learning_rate,
        checkpoint_path=str(checkpoint_dir / "best_model.pt"),
    )

    print("Starting training...\n")
    history = trainer.train(num_epochs=args.epochs)

    plot_training_history(history, save_to=str(Path(args.output_dir) / "training_history.png"))
    print(f"Saved training_history.png to {args.output_dir}/")

    # Reload best checkpoint (not necessarily the last epoch) before final evaluation.
    best_checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    print(f"\nLoaded best checkpoint from epoch {best_checkpoint['epoch']} "
          f"(val_acc={best_checkpoint['val_acc']:.4f}) for final test evaluation.")

    print("\nEvaluating on test set...")
    evaluator = Evaluator(model, test_loader, class_names=label_encoder.CLASSES)
    results = evaluator.evaluate(save_dir=args.output_dir)

    print(f"\nFinal Test Accuracy: {results['accuracy']:.4f}")
    print(f"Final Test Macro F1:  {results['macro_f1']:.4f}")

    # Save vocab + label encoder so pipeline.py can load everything for inference.
    vocab.save(str(checkpoint_dir / "vocab.json"))
    label_encoder.save(str(checkpoint_dir / "label_encoder.json"))
    print(f"\nSaved vocab.json and label_encoder.json to {checkpoint_dir}/")
    print(f"Training complete. Model checkpoint, vocab, and label encoder are ready "
          f"for use in src/pipeline.py.")


if __name__ == "__main__":
    main()