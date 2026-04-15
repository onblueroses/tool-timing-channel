"""HuggingFace model wrapper for token-level steganography.

Provides access to raw logits/probability distributions over the
full vocabulary, which ollama doesn't expose.
"""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class StegoModel:
    """Wraps a HuggingFace causal LM for steganographic sampling."""

    def __init__(self, model_name: str, dtype: torch.dtype | None = None) -> None:
        self.model_name = model_name
        self._dtype = dtype
        # typed as Any because transformers' Auto classes lack proper stubs
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        """Load the model and tokenizer. Call before any other method."""
        kwargs: dict = {"torch_dtype": self._dtype} if self._dtype else {}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("Call load() before using the model")
        return self._model

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("Call load() before using the tokenizer")
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text into token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, ids: list[int]) -> str:
        """Convert token IDs back to text."""
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    @torch.no_grad()
    def get_distribution(self, input_ids: list[int]) -> list[float]:
        """Get the probability distribution over next tokens.

        Args:
            input_ids: Token IDs for the context so far.

        Returns:
            List of probabilities, one per vocabulary entry, summing to 1.0.
        """
        ids_tensor = torch.tensor([input_ids], dtype=torch.long)
        if hasattr(self.model, "device"):
            ids_tensor = ids_tensor.to(self.model.device)

        outputs = self.model(ids_tensor)
        logits = outputs.logits[0, -1, :]  # last position
        probs = torch.softmax(logits, dim=0)
        return probs.cpu().tolist()
