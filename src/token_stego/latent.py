"""Latent-space instrumentation for steganography research.

Captures model internals (KV cache, hidden states, attention patterns)
during inference via PyTorch hooks. Enables measurement of information
content beyond what the output text alone conveys.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .model import StegoModel


@dataclass
class LatentSnapshot:
    """Captured model internals from a forward pass."""

    hidden_states: dict[int, Tensor] = field(default_factory=dict)
    # layer -> (key, value) tensors
    kv_cache: dict[int, tuple[Tensor, Tensor]] = field(default_factory=dict)
    model_name: str = ""
    seq_len: int = 0
    num_layers: int = 0

    def total_elements(self) -> int:
        """Total number of float values captured."""
        total = 0
        for t in self.hidden_states.values():
            total += t.numel()
        for k, v in self.kv_cache.values():
            total += k.numel() + v.numel()
        return total


class LatentCapture:
    """Context manager that captures model internals during inference.

    Usage:
        with LatentCapture(model) as cap:
            model.get_distribution(input_ids)
            snap = cap.snapshot()
    """

    def __init__(
        self, model: StegoModel, capture_layers: list[int] | None = None
    ) -> None:
        self._model = model
        self._capture_layers = capture_layers
        self._hooks: list = []
        self._hidden_states: dict[int, Tensor] = {}
        self._kv_cache: dict[int, tuple[Tensor, Tensor]] = {}

    def __enter__(self) -> "LatentCapture":
        self._register_hooks()
        return self

    def __exit__(self, *args: object) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _register_hooks(self) -> None:
        """Register forward hooks on model layers."""
        model = self._model.model

        # Find attention/transformer layers
        layers = self._find_layers(model)
        num_layers = len(layers)

        target_layers = self._capture_layers
        if target_layers is None:
            # Capture every 4th layer + first + last to manage memory
            target_layers = [0, num_layers - 1]
            target_layers.extend(range(0, num_layers, max(1, num_layers // 8)))
            target_layers = sorted(set(target_layers))

        for layer_idx in target_layers:
            if layer_idx >= num_layers:
                continue
            layer = layers[layer_idx]
            hook = layer.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def _find_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        """Find transformer layers in the model."""
        # Try common attribute names for transformer layer containers
        for attr in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            obj: object = model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return [m for m in obj]  # type: ignore[union-attr]
            except (AttributeError, TypeError):
                continue
        # Fallback: search for repeated module containers
        for _name, module in model.named_modules():
            children = list(module.children())
            if len(children) > 10:
                return children
        return []

    def _make_hook(self, layer_idx: int):  # noqa: ANN202
        """Create a forward hook for a specific layer."""

        def hook_fn(module: torch.nn.Module, input: tuple, output: object) -> None:
            # Output varies by architecture; try to extract hidden states
            if isinstance(output, tuple) and len(output) > 0:
                hidden = output[0]
                if isinstance(hidden, Tensor):
                    # Detach and move to CPU to avoid GPU memory buildup
                    self._hidden_states[layer_idx] = hidden.detach().cpu()

                # Some architectures return (hidden, present_kv, ...)
                if len(output) > 1 and output[1] is not None:
                    kv = output[1]
                    if isinstance(kv, tuple) and len(kv) == 2:
                        k, v = kv
                        if isinstance(k, Tensor) and isinstance(v, Tensor):
                            self._kv_cache[layer_idx] = (
                                k.detach().cpu(),
                                v.detach().cpu(),
                            )
            elif isinstance(output, Tensor):
                self._hidden_states[layer_idx] = output.detach().cpu()

        return hook_fn

    def snapshot(self) -> LatentSnapshot:
        """Create a snapshot of currently captured states."""
        seq_len = 0
        if self._hidden_states:
            first = next(iter(self._hidden_states.values()))
            seq_len = first.shape[1] if first.dim() >= 2 else first.shape[0]

        return LatentSnapshot(
            hidden_states=dict(self._hidden_states),
            kv_cache=dict(self._kv_cache),
            model_name=self._model.model_name,
            seq_len=seq_len,
            num_layers=len(self._hidden_states),
        )

    def clear(self) -> None:
        """Clear captured states for the next forward pass."""
        self._hidden_states.clear()
        self._kv_cache.clear()


def compare_snapshots(snap_a: LatentSnapshot, snap_b: LatentSnapshot) -> dict:
    """Compare two latent snapshots, computing per-layer distances.

    Returns dict with per-layer cosine similarity and L2 distance
    for hidden states and KV cache.
    """
    results: dict = {"hidden_states": {}, "kv_cache": {}}

    # Compare hidden states
    common_layers = set(snap_a.hidden_states) & set(snap_b.hidden_states)
    for layer in sorted(common_layers):
        a = snap_a.hidden_states[layer].float().flatten()
        b = snap_b.hidden_states[layer].float().flatten()
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        cosine = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)
        ).item()
        l2 = torch.norm(a - b).item()
        results["hidden_states"][layer] = {"cosine_sim": cosine, "l2_distance": l2}

    # Compare KV cache
    common_kv = set(snap_a.kv_cache) & set(snap_b.kv_cache)
    for layer in sorted(common_kv):
        ka, va = snap_a.kv_cache[layer]
        kb, vb = snap_b.kv_cache[layer]
        ka, kb = ka.float().flatten(), kb.float().flatten()
        va, vb = va.float().flatten(), vb.float().flatten()

        min_k = min(len(ka), len(kb))
        min_v = min(len(va), len(vb))
        k_cos = torch.nn.functional.cosine_similarity(
            ka[:min_k].unsqueeze(0), kb[:min_k].unsqueeze(0)
        ).item()
        v_cos = torch.nn.functional.cosine_similarity(
            va[:min_v].unsqueeze(0), vb[:min_v].unsqueeze(0)
        ).item()
        k_l2 = torch.norm(ka[:min_k] - kb[:min_k]).item()
        v_l2 = torch.norm(va[:min_v] - vb[:min_v]).item()

        results["kv_cache"][layer] = {
            "key_cosine": k_cos,
            "key_l2": k_l2,
            "value_cosine": v_cos,
            "value_l2": v_l2,
        }

    return results
