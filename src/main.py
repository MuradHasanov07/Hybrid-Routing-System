"""
main.py

Command-line entry point for the hybrid routing system. This is the script an
end user (or a grader/demo) actually runs to classify and respond to a
real, freely-typed customer message - as opposed to pipeline.py's fixed
demo messages, which only exist as a quick internal sanity check.

Usage:
    # Interactive mode - keeps the model loaded and lets you type as many
    # messages as you want, one per line.
    python src/main.py

    # Single-shot mode - classify one message and exit immediately.
    python src/main.py --message "Can you tell me the price and mileage of the 2019 model?"

Both modes load the trained model, vocabulary, and label encoder ONCE at
startup (loading is the slow part - classifying a message afterwards takes
milliseconds), so interactive mode stays fast even after many messages.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import RoutingPipeline


def build_pipeline() -> RoutingPipeline:
    project_root = Path(__file__).resolve().parent.parent
    checkpoint_path = project_root / "checkpoints" / "best_model.pt"
    vocab_path = project_root / "checkpoints" / "vocab.json"
    label_encoder_path = project_root / "checkpoints" / "label_encoder.json"

    if not checkpoint_path.exists():
        print("ERROR: no trained checkpoint found at "
              f"{checkpoint_path}.\nRun 'python scripts/train.py' first to "
              "train a model.")
        sys.exit(1)

    return RoutingPipeline(
        model_checkpoint=str(checkpoint_path),
        vocab_path=str(vocab_path),
        label_encoder_path=str(label_encoder_path),
    )


def print_result(result: dict) -> None:
    print(f"\nPredicted class: {result['predicted_label']}")
    print(f"Confidence:      {result['confidence']:.2f}")
    print(f"Source:          {result['source']}")
    print(f"Response:        {result['response']}\n")


def run_single_message(pipeline: RoutingPipeline, message: str) -> None:
    result = pipeline.predict(message)
    print_result(result)


def run_interactive(pipeline: RoutingPipeline) -> None:
    print("Hybrid Routing System - interactive mode")
    print("Type a customer message and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            message = input("Enter customer message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if message.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        if not message:
            continue

        result = pipeline.predict(message)
        print_result(result)


def main():
    parser = argparse.ArgumentParser(
        description="Classify a customer message and route it to a template "
                     "reply (Spam) or a mock-GPT response (Sales Inquiry / Complaint)."
    )
    parser.add_argument(
        "--message", type=str, default=None,
        help="A single message to classify. If omitted, starts an interactive session."
    )
    args = parser.parse_args()

    pipeline = build_pipeline()

    if args.message:
        run_single_message(pipeline, args.message)
    else:
        run_interactive(pipeline)


if __name__ == "__main__":
    main()