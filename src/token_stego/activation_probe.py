"""Activation probe for detecting token-likelihood steganography.

Trains a simple MLP to classify "normal inference" vs "stego inference"
from captured hidden states. If the probe achieves high AUC, latent-space
stego is detectable at the activation level.
"""

import torch
from torch import Tensor, nn

from .latent import LatentSnapshot


class ActivationProbe(nn.Module):
    """2-layer MLP probe for stego vs normal classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def snapshot_to_features(snap: LatentSnapshot, layer: int | None = None) -> Tensor:
    """Extract a fixed-size feature vector from a LatentSnapshot.

    When layer is specified, returns mean-pooled hidden states from that
    single layer. When layer is None, concatenates mean-pooled vectors
    from ALL captured layers (sorted by layer index), producing a vector
    of size num_layers * hidden_dim.
    """
    if not snap.hidden_states:
        raise ValueError("Snapshot has no hidden states")

    if layer is not None:
        hidden = snap.hidden_states[layer].float()
        if hidden.dim() == 3:
            hidden = hidden[0]
        return hidden.mean(dim=0)

    # Concatenate mean-pooled features from all captured layers
    layer_features = []
    for layer_idx in sorted(snap.hidden_states.keys()):
        hidden = snap.hidden_states[layer_idx].float()
        if hidden.dim() == 3:
            hidden = hidden[0]
        layer_features.append(hidden.mean(dim=0))
    return torch.cat(layer_features, dim=0)


def train_probe(
    normal_snaps: list[LatentSnapshot],
    stego_snaps: list[LatentSnapshot],
    layer: int | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
) -> ActivationProbe:
    """Train a probe to distinguish normal from stego inference.

    Args:
        normal_snaps: Snapshots from normal (non-stego) inference.
        stego_snaps: Snapshots from stego inference.
        layer: Which layer's hidden states to use. None = last captured.
        epochs: Training epochs.
        lr: Learning rate.

    Returns:
        Trained ActivationProbe.
    """
    # Build feature matrix and labels
    features = []
    labels = []
    for snap in normal_snaps:
        features.append(snapshot_to_features(snap, layer))
        labels.append(0.0)
    for snap in stego_snaps:
        features.append(snapshot_to_features(snap, layer))
        labels.append(1.0)

    x = torch.stack(features)
    y = torch.tensor(labels)

    probe = ActivationProbe(input_dim=x.shape[1])
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    probe.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = probe(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    probe.eval()
    return probe


def evaluate_probe(
    probe: ActivationProbe,
    snapshots: list[LatentSnapshot],
    labels: list[int],
    layer: int | None = None,
) -> dict:
    """Evaluate probe on test data. Returns AUC and accuracy."""
    features = [snapshot_to_features(snap, layer) for snap in snapshots]
    x = torch.stack(features)

    with torch.no_grad():
        logits = probe(x)
        scores = torch.sigmoid(logits).tolist()

    # Compute AUC
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    tp = fp = 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0 or total_neg == 0:
        auc = 0.5
    else:
        auc = 0.0
        prev_fpr = prev_tpr = 0.0
        i = 0
        while i < len(pairs):
            j = i
            while j < len(pairs) and pairs[j][0] == pairs[i][0]:
                if pairs[j][1] == 1:
                    tp += 1
                else:
                    fp += 1
                j += 1
            tpr = tp / total_pos
            fpr = fp / total_neg
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
            prev_fpr, prev_tpr = fpr, tpr
            i = j

    # Accuracy at threshold 0.5
    preds = [1 if s > 0.5 else 0 for s in scores]
    accuracy = sum(p == lab for p, lab in zip(preds, labels)) / len(labels)

    return {"auc": auc, "accuracy": accuracy, "scores": scores}
