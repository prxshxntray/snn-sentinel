"""Send-on-delta (sigma-delta) spike encoding.

Runs causally over the whole stream, so in deployment each sensor only emits an
event when its value has moved by more than `threshold` since its last event.
Two output channels per sensor (up / down). This is what makes the detector
event-driven: a quiet data hall produces almost no spikes and almost no SynOps.
"""
from __future__ import annotations

import numpy as np


def sigma_delta_encode(x: np.ndarray, threshold: float) -> np.ndarray:
    """x: [n_steps, C] (z-scored) -> spikes [n_steps, 2C] in {0,1}. At most one event per
    channel per step; the reference moves by one threshold per event (so large jumps
    are spread over consecutive steps, like a rate code of the change)."""
    n, c = x.shape
    ref = x[0].copy()
    out = np.zeros((n, 2 * c), dtype=np.float32)
    for t in range(1, n):
        d = x[t] - ref
        up, dn = d > threshold, d < -threshold
        out[t, :c][up] = 1.0
        out[t, c:][dn] = 1.0
        ref = ref + threshold * up - threshold * dn
    return out
