"""Baselines: CUSUM change-point, gradient-boosted trees, TCN, LSTM, small transformer."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn


# ---------------------------------------------------------------- change-point
class CUSUM:
    """Two-sided CUSUM per channel on z-scored data; alarm when any channel exceeds h.
    Swap for BOCPD or `ruptures` (Truong et al., arXiv:1801.00826) in the full study."""

    def __init__(self, k: float = 0.5, h: float = 5.0):
        self.k, self.h = k, h

    def score(self, x: np.ndarray) -> np.ndarray:
        n, c = x.shape
        gp, gn = np.zeros(c), np.zeros(c)
        out = np.zeros(n, dtype=np.float32)
        for t in range(n):
            gp = np.maximum(0, gp + x[t] - self.k)
            gn = np.maximum(0, gn - x[t] - self.k)
            out[t] = max(gp.max(), gn.max())
        return out / self.h          # >1 means alarm

    def ops_per_step(self, c: int) -> int:
        return 6 * c                  # adds/compares per step, no multiplies


# ---------------------------------------------------------------- GBT
def window_features(X: np.ndarray) -> np.ndarray:
    """[N, W, C] -> hand features per channel: last, mean, std, min, max, last-first, last-mean."""
    last, first = X[:, -1], X[:, 0]
    mean = X.mean(1)
    return np.concatenate([last, mean, X.std(1), X.min(1), X.max(1), last - first, last - mean], axis=1)


class GBTDetector:
    def __init__(self, max_iter: int = 200, max_depth: int = 4, seed: int = 0):
        self.clf = HistGradientBoostingClassifier(max_iter=max_iter, max_depth=max_depth,
                                                  class_weight="balanced", random_state=seed)
        self.max_iter, self.max_depth = max_iter, max_depth

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBTDetector":
        self.clf.fit(window_features(X), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(window_features(X))[:, 1]

    def ops_per_window(self, w: int, c: int) -> int:
        feat = 7 * w * c                                   # feature extraction (adds/compares)
        trees = self.clf.n_iter_ * self.max_depth           # one compare per tree level
        return feat + trees


# ---------------------------------------------------------------- deep baselines
class TCN(nn.Module):
    """Small causal dilated TCN (Bai, Kolter & Koltun, arXiv:1803.01271)."""

    def __init__(self, c_in: int, ch: int = 32, levels: int = 3, k: int = 3):
        super().__init__()
        layers, c = [], c_in
        for i in range(levels):
            d = 2 ** i
            layers += [nn.ConstantPad1d(((k - 1) * d, 0), 0.0), nn.Conv1d(c, ch, k, dilation=d), nn.ReLU()]
            c = ch
        self.net, self.head = nn.Sequential(*layers), nn.Linear(ch, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x [B, W, C]
        return self.head(self.net(x.transpose(1, 2))[:, :, -1])


class LSTMDetector(nn.Module):
    def __init__(self, c_in: int, hidden: int = 32):
        super().__init__()
        self.lstm, self.head = nn.LSTM(c_in, hidden, batch_first=True), nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.lstm(x)
        return self.head(o[:, -1])


class TinyTransformer(nn.Module):
    def __init__(self, c_in: int, d: int = 32, heads: int = 2, layers: int = 1, w: int = 64):
        super().__init__()
        self.inp = nn.Linear(c_in, d)
        self.pos = nn.Parameter(torch.zeros(1, w, d))
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, batch_first=True)
        self.enc, self.head = nn.TransformerEncoder(enc, layers), nn.Linear(d, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x) + self.pos[:, : x.shape[1]]
        return self.head(self.enc(h)[:, -1])


def count_macs(model: nn.Module, x: torch.Tensor) -> int:
    """MACs for one forward pass on x (per batch item), via hooks on Linear/Conv1d/LSTM.
    Attention score/value matmuls are added analytically for TinyTransformer. Dense ANN cost:
    every MAC happens regardless of whether the input changed."""
    total = [0]

    def lin(m, i, o):
        total[0] += i[0].numel() // i[0].shape[-1] * m.in_features * m.out_features

    def conv(m, i, o):
        total[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0]

    def lstm(m, i, o):
        B, T, C = i[0].shape
        H = m.hidden_size
        total[0] += B * T * 4 * H * (C + H)

    hooks = []
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin))
        elif isinstance(mod, nn.Conv1d):
            hooks.append(mod.register_forward_hook(conv))
        elif isinstance(mod, nn.LSTM):
            hooks.append(mod.register_forward_hook(lstm))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    if isinstance(model, TinyTransformer):
        B, W, _ = x.shape
        d = model.inp.out_features
        total[0] += B * len(model.enc.layers) * 2 * W * W * d
        # MultiheadAttention calls F.linear for q,k,v and out projections (hooks do not fire) -> add 4*W*d*d
        total[0] += B * len(model.enc.layers) * 4 * W * d * d
    return total[0] // x.shape[0]
