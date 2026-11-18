"""Dataset loaders + windowing.

Real datasets (download separately, see README):
  - NAB (Numenta Anomaly Benchmark): https://github.com/numenta/NAB
      e.g. data/realKnownCause/machine_temperature_system_failure.csv, labels in labels/combined_windows.json
  - M100 anomaly-detection subset (CINECA Marconi100, 15-min aggregates + Nagios labels):
      https://zenodo.org/records/7541722  (needs `pip install pyarrow`)
  - Cluster workload traces for realistic load profiles: https://github.com/alibaba/clusterdata,
      https://github.com/google/cluster-data, https://github.com/Azure/AzurePublicDataset
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import faults, synthetic


@dataclass
class Split:
    x: np.ndarray      # [n_steps, n_channels], z-scored with train stats
    y: np.ndarray      # [n_steps]
    events: list       # list[faults.Event] falling in this split (indices relative to split start)


def zscore_splits(x: np.ndarray, bounds: list[tuple[int, int]]) -> list[np.ndarray]:
    a, b = bounds[0]
    mu, sd = x[a:b].mean(0), x[a:b].std(0) + 1e-6
    return [((x[s:e] - mu) / sd).astype(np.float32) for s, e in bounds]


def load_synthetic(cfg: dict, seed: int) -> dict[str, Split]:
    d = cfg["data"]
    rng = np.random.default_rng(seed)
    base = synthetic.simulate_baseline(d["n_steps"], d["n_racks"], rng)
    x, y, events = faults.inject(base, d["n_racks"], d["n_events"], d["event_types"], rng)
    n = len(x)
    t1, t2 = int(d["train_frac"] * n), int((d["train_frac"] + d["val_frac"]) * n)
    bounds = [(0, t1), (t1, t2), (t2, n)]
    xs = zscore_splits(x, bounds)
    out = {}
    for name, (s, e), xi in zip(["train", "val", "test"], bounds, xs):
        ev = [faults.Event(v.kind, v.rack, v.onset - s, v.critical - s, v.end - s)
              for v in events if s <= v.onset and v.end <= e]
        out[name] = Split(xi, y[s:e].copy(), ev)
    return out


def load_nab_series(nab_root: str | Path, rel_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one NAB series + its labelled anomaly windows. Returns (values[n,1], labels[n])."""
    nab_root = Path(nab_root)
    df = pd.read_csv(nab_root / "data" / rel_path, parse_dates=["timestamp"])
    windows = json.loads((nab_root / "labels" / "combined_windows.json").read_text())[rel_path]
    y = np.zeros(len(df), dtype=np.int64)
    for a, b in windows:
        y[(df.timestamp >= pd.Timestamp(a)) & (df.timestamp <= pd.Timestamp(b))] = 1
    return df[["value"]].to_numpy(np.float32), y


def load_m100_node(parquet_path: str | Path, label_col: str = "value") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one M100 node file (15-min rows). Label column name may differ by release; check the
    companion repo https://gitlab.com/ecs-lab/exadata and adjust `label_col`."""
    df = pd.read_parquet(parquet_path)
    feats = [c for c in df.columns if c.endswith("_avg")]
    x = df[feats].ffill().fillna(0).to_numpy(np.float32)
    y = (df[label_col].fillna(0).to_numpy() != 0).astype(np.int64)
    return x, y, feats


def make_windows(x: np.ndarray, y: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal sliding windows: X[i] = x[i:i+w], label = y[i+w-1] (label of the newest step)."""
    idx = np.arange(len(x) - w + 1)
    X = np.stack([x[i:i + w] for i in idx]).astype(np.float32)
    return X, y[idx + w - 1]
