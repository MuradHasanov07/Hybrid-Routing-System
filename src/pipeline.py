"""
pipeline.py

RoutingPipeline ties everything together: loads a trained model + vocabulary +
label encoder, classifies an incoming message, and routes it to either the
TemplateResponder (Spam) or the GPTRouter (Sales Inquiry / Complaint).

This is the single entry point the rest of the system (or a demo script) is
expected to call: pipeline.predict("some customer message") -> result dict.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tokenizer import Vocabulary
from dataset import LabelEncoder
from model import MeanPoolingTextClassifier
from router import TemplateResponder, GPTRouter


class RoutingPipeline:
    def __init__(
        self,
        model_checkpoint: str,
        vocab_path: str,
        label_encoder_path: str,
        max_len: int = 32,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len

        self.vocab = Vocabulary.load(vocab_path)
        self.label_encoder = LabelEncoder.load(label_encoder_path)

        self.model = MeanPoolingTextClassifier(
            vocab_size=len(self.vocab),
            num_classes=self.label_encoder.num_classes(),
        )
        checkpoint = torch.load(model_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.template_responder = TemplateResponder()
        self.gpt_router = GPTRouter()

    def _encode_message(self, message: str) -> torch.Tensor:
        ids = self.vocab.encode(message)
        pad_idx = self.vocab.token_to_idx["<PAD>"]
        if len(ids) >= self.max_len:
            ids = ids[: self.max_len]
        else:
            ids = ids + [pad_idx] * (self.max_len - len(ids))
        return torch.tensor([ids], dtype=torch.long)  # shape (1, max_len)

    def classify(self, message: str) -> Dict:
        """Runs only the classifier step, without routing to a responder.
        Returns the predicted label and the model's confidence."""
        x = self._encode_message(message).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(dim=1)

        predicted_label = self.label_encoder.decode(pred_idx.item())
        return {
            "predicted_label": predicted_label,
            "confidence": confidence.item(),
        }

    def predict(self, message: str) -> Dict:
        """Full pipeline: classify the message, then route it to the
        appropriate responder (template for Spam, mock-GPT otherwise)."""
        classification = self.classify(message)
        predicted_label = classification["predicted_label"]

        if predicted_label == "Spam":
            response = self.template_responder.respond(message)
            source = "template"
        else:
            response = self.gpt_router.generate_response(message, predicted_label)
            source = "gpt (mock)"

        return {
            "message": message,
            "predicted_label": predicted_label,
            "confidence": classification["confidence"],
            "response": response,
            "source": source,
        }


if __name__ == "__main__":
    # End-to-end demo: trains a fresh model (if no checkpoint exists yet) and
    # runs a handful of example messages through the full pipeline.
    import sys
    sys.path.insert(0, "..")
    from pathlib import Path as _Path

    project_root = _Path(__file__).resolve().parent.parent
    checkpoint_path = project_root / "checkpoints" / "best_model.pt"
    vocab_path = project_root / "checkpoints" / "vocab.json"
    label_encoder_path = project_root / "checkpoints" / "label_encoder.json"

    if not checkpoint_path.exists():
        print("No trained checkpoint found yet. Run scripts/train.py first to "
              "train a model and save the checkpoint/vocab/label_encoder files.")
        sys.exit(1)

    pipeline = RoutingPipeline(
        model_checkpoint=str(checkpoint_path),
        vocab_path=str(vocab_path),
        label_encoder_path=str(label_encoder_path),
    )

    demo_messages = [
        "Can you tell me the price and mileage of the 2019 model?",
        "Congratulations, you've won a free reward, click here to claim it!",
        "The app crashes every time I try to log in, please help.",
        "I'd like to cancel my subscription before the next billing cycle.",
    ]

    for msg in demo_messages:
        result = pipeline.predict(msg)
        print(f"\nMessage:    {result['message']}")
        print(f"Predicted:  {result['predicted_label']} (confidence: {result['confidence']:.2f})")
        print(f"Source:     {result['source']}")
        print(f"Response:   {result['response']}")