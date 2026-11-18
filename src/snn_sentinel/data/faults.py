"""Fault injection on top of a baseline trace.

Each event has an onset (when the fault starts), a critical time (when an operator
would have to act: e.g. inlet temperature crosses the ASHRAE-style limit or the
spike reaches peak) and an end. Labels are 1 on [onset, end). Detection lead time is
measured against the critical time, so early detection during the precursor counts.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

N_CH = 5  # inlet, outlet, util, fan, power


@dataclass
class Event:
    kind: str
    rack: int
    onset: int
    critical: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)


def inject(x: np.ndarray, n_racks: int, n_events: int, kinds: list[str],
           rng: np.random.Generator, margin: int = 64) -> tuple[np.ndarray, np.ndarray, list[Event]]:
    x = x.copy()
    n = len(x)
    y = np.zeros(n, dtype=np.int64)
    events: list[Event] = []
    slots = np.linspace(margin, n - margin - 200, n_events).astype(int)
    for s in slots:
        onset = int(s + rng.integers(0, 60))
        rack = int(rng.integers(0, n_racks))
        kind = str(rng.choice(kinds))
        b = rack * N_CH
        if kind == "workload_spike":          # sudden job burst: util/power jump, temps follow
            dur, ramp = int(rng.integers(30, 90)), int(rng.integers(3, 10))
            prof = np.minimum(np.arange(dur) / ramp, 1.0)
            sl = slice(onset, onset + dur)
            x[sl, b + 2] = np.clip(x[sl, b + 2] + 0.4 * prof, 0, 1.2)
            x[sl, b + 4] += 2.4 * prof
            x[sl, b + 1] += 3.0 * prof
            x[sl, b + 0] += 0.6 * prof
            crit = onset + ramp
        elif kind == "cooling_failure":       # CRAC/fan degradation: slow thermal ramp, fan saturates
            dur = int(rng.integers(80, 160))
            prof = np.arange(dur) / dur
            sl = slice(onset, onset + dur)
            x[sl, b + 0] += 6.0 * prof
            x[sl, b + 1] += 8.0 * prof
            x[sl, b + 3] = np.minimum(x[sl, b + 3] + 0.5 * prof, 1.0)
            crit = onset + int(0.6 * dur)      # when inlet would cross the limit
        else:                                 # hot_spot: local recirculation, inlet only
            dur = int(rng.integers(40, 100))
            prof = np.sin(np.pi * np.arange(dur) / dur)
            sl = slice(onset, onset + dur)
            x[sl, b + 0] += 3.5 * prof
            x[sl, b + 1] += 2.0 * prof
            crit = onset + dur // 2
        end = onset + dur
        y[onset:end] = 1
        events.append(Event(kind, rack, onset, int(crit), end))
    return x, y, events


def drop_sensors(x: np.ndarray, frac: float, rng: np.random.Generator, fill: float = 0.0) -> np.ndarray:
    """Simulate missing sensors: whole channels go dark (filled with `fill` after z-scoring)."""
    if frac <= 0:
        return x
    x = x.copy()
    k = int(round(frac * x.shape[1]))
    idx = rng.choice(x.shape[1], size=k, replace=False)
    x[:, idx] = fill
    return x
