"""Training / evaluation helpers shared by scripts."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    np.random.seed(seed); torch.manual_seed(seed)


def train_torch(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int, lr: float,
                batch: int = 128, seed: int = 0) -> list[float]:
    g = torch.Generator().manual_seed(seed)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    pos = max(float(y.mean()), 1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, (1 - pos) / pos]).clamp(max=20))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for _ in range(epochs):
        model.train(); perm = torch.randperm(len(Xt), generator=g); tot = 0.0
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad(); loss = loss_fn(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        hist.append(tot / len(Xt))
    return hist


@torch.no_grad()
def predict_torch(model: nn.Module, X: np.ndarray, batch: int = 512) -> tuple[np.ndarray, list[dict]]:
    model.eval(); probs, stats = [], []
    for i in range(0, len(X), batch):
        out = model(torch.from_numpy(X[i:i + batch]))
        probs.append(torch.softmax(out, -1)[:, 1].numpy())
        if getattr(model, "last_stats", None):
            stats.append(dict(model.last_stats))
    return np.concatenate(probs), stats


def merge_stats(stats: list[dict]) -> dict:
    keys = ["in_spikes", "hidden_spikes", "synops", "neuron_steps_hidden", "input_steps", "windows"]
    out = {k: sum(s[k] for s in stats) for k in keys}
    out["T"] = stats[0]["T"]
    return out


def to_alarm(prob: np.ndarray, w: int, n_steps: int, thr: float) -> np.ndarray:
    """Window predictions -> per-step alarm series aligned to the split (first w-1 steps silent)."""
    a = np.zeros(n_steps, dtype=np.int64)
    a[w - 1:w - 1 + len(prob)] = (prob >= thr).astype(np.int64)
    return a


def pick_threshold(prob: np.ndarray, y: np.ndarray, grid=None) -> float:
    """Choose the decision threshold on the *validation* split by point-F1 (never on test)."""
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 19)
    best, bt = -1.0, 0.5
    for t in grid:
        p = prob >= t
        tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn = (~p & (y == 1)).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best:
            best, bt = f1, float(t)
    return bt
