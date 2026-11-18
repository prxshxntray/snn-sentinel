"""Synthetic rack telemetry with a diurnal workload cycle.

This is a stand-in for the Wattr Modelica/FMU digital twin: same channel layout,
crude first-order thermal physics. Replace `simulate_baseline` with FMU rollouts
(e.g. via fmpy) once the twin is exported; fault injection in `faults.py` is
model-agnostic and works on either.
"""
from __future__ import annotations

import numpy as np

CHANNELS = ["inlet_temp", "outlet_temp", "cpu_util", "fan_speed", "power"]


def simulate_baseline(n_steps: int, n_racks: int, rng: np.random.Generator) -> np.ndarray:
    """Return array [n_steps, n_racks * 5] of normal operation (1 step = 1 minute)."""
    t = np.arange(n_steps)
    out = []
    for _ in range(n_racks):
        phase = rng.uniform(0, 2 * np.pi)
        util = 0.45 + 0.25 * np.sin(2 * np.pi * t / 1440 + phase)          # diurnal load
        util += 0.05 * np.sin(2 * np.pi * t / 97 + rng.uniform(0, 6))       # batch-job ripple
        util = np.clip(util + rng.normal(0, 0.03, n_steps), 0, 1)
        power = 4.0 + 6.0 * util + rng.normal(0, 0.1, n_steps)              # kW per rack
        fan = 0.3 + 0.6 * util + rng.normal(0, 0.02, n_steps)
        inlet = np.empty(n_steps)
        inlet[0] = 22.0
        for i in range(1, n_steps):                                         # first-order lag to setpoint
            target = 22.0 + 1.5 * (util[i] - 0.45)
            inlet[i] = inlet[i - 1] + 0.05 * (target - inlet[i - 1]) + rng.normal(0, 0.05)
        outlet = inlet + 2.0 * power / (fan + 0.2) + rng.normal(0, 0.1, n_steps)
        out.append(np.stack([inlet, outlet, util, fan, power], axis=1))
    return np.concatenate(out, axis=1).astype(np.float32)


def channel_names(n_racks: int) -> list[str]:
    return [f"r{r}_{c}" for r in range(n_racks) for c in CHANNELS]
