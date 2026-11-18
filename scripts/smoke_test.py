#!/usr/bin/env python
"""End-to-end smoke test on synthetic telemetry.

Trains a tiny spiking sentinel (snnTorch) and a gradient-boosted-tree baseline, runs a
CUSUM change-point detector, then prints detection metrics (lead time, detection rate,
false alarms/day), missing-sensor robustness, and SynOps/MAC-based energy estimates.

    python scripts/smoke_test.py --config configs/smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snn_sentinel.data import faults, loaders  # noqa: E402
from snn_sentinel.encoding import sigma_delta_encode  # noqa: E402
from snn_sentinel.metrics.detection import event_metrics  # noqa: E402
from snn_sentinel.metrics.energy import snn_cost, ops_cost  # noqa: E402
from snn_sentinel.models.baselines import CUSUM, GBTDetector  # noqa: E402
from snn_sentinel.models.snn import SpikingSentinel  # noqa: E402
from snn_sentinel.train import (merge_stats, pick_threshold, predict_torch, set_seed,  # noqa: E402
                                to_alarm, train_torch)


def fmt(d: dict) -> str:
    return "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def run(cfg: dict) -> dict:
    t0 = time.time()
    seed = cfg["seed"]; set_seed(seed)
    w, ev_cfg, en = cfg["window"], cfg["eval"], cfg["energy"]
    S = loaders.load_synthetic(cfg, seed)
    C = S["train"].x.shape[1]
    print(f"data: synthetic  channels={C}  steps train/val/test="
          f"{len(S['train'].x)}/{len(S['val'].x)}/{len(S['test'].x)}  "
          f"events train/val/test={len(S['train'].events)}/{len(S['val'].events)}/{len(S['test'].events)}")

    th = cfg["model"]["snn"]["delta_threshold"]
    def spike_windows(x, y):
        return loaders.make_windows(sigma_delta_encode(x, th), y, w)

    def raw_windows(x, y):
        return loaders.make_windows(x, y, w)

    results: dict = {}

    # ---------------- SNN
    sc = cfg["model"]["snn"]
    Xtr, ytr = spike_windows(S["train"].x, S["train"].y)
    Xva, yva = spike_windows(S["val"].x, S["val"].y)
    snn_model = SpikingSentinel(2 * C, hidden=sc["hidden"], beta=sc["beta"])
    hist = train_torch(snn_model, Xtr, ytr, sc["epochs"], sc["lr"], seed=seed)
    print(f"snn train loss by epoch: {[round(h, 4) for h in hist]}")
    pva, _ = predict_torch(snn_model, Xva)
    thr_snn = pick_threshold(pva, yva)

    # ---------------- GBT
    gc = cfg["model"]["gbt"]
    Rtr, rtr = raw_windows(S["train"].x, S["train"].y)
    Rva, rva = raw_windows(S["val"].x, S["val"].y)
    gbt = GBTDetector(gc["max_iter"], gc["max_depth"], seed).fit(Rtr, rtr)
    thr_gbt = pick_threshold(gbt.predict_proba(Rva), rva)

    cus = CUSUM(cfg["model"]["cusum"]["k"], cfg["model"]["cusum"]["h"])

    rng = np.random.default_rng(seed + 1)
    for frac in cfg["missing_sensor_eval"]:
        xt = faults.drop_sensors(S["test"].x, frac, rng)
        yt, evs, n = S["test"].y, S["test"].events, len(S["test"].x)
        key = f"missing={frac:.0%}"
        Xte, _ = spike_windows(xt, yt)
        pte, stats = predict_torch(snn_model, Xte)
        m_snn = event_metrics(to_alarm(pte, w, n, thr_snn), evs, ev_cfg["steps_per_day"], ev_cfg["min_alarm_gap"], yt)
        Rte, _ = raw_windows(xt, yt)
        m_gbt = event_metrics(to_alarm(gbt.predict_proba(Rte), w, n, thr_gbt), evs,
                              ev_cfg["steps_per_day"], ev_cfg["min_alarm_gap"], yt)
        m_cus = event_metrics((cus.score(xt) > 1).astype(np.int64), evs,
                              ev_cfg["steps_per_day"], ev_cfg["min_alarm_gap"], yt)
        results[key] = {"snn": m_snn, "gbt": m_gbt, "cusum": m_cus}
        if frac == 0.0:
            ms = merge_stats(stats)
            results["cost"] = {
                "snn": {**snn_cost(ms, en, "step"), **snn_cost(ms, en, "window")},
                "gbt": ops_cost(gbt.ops_per_window(w, C), en, "window"),
                "cusum": ops_cost(cus.ops_per_step(C), en, "step"),
            }

    print("\n=== detection on held-out test split (lead time in steps; 1 step = 1 min) ===")
    for key, block in results.items():
        if key == "cost":
            continue
        print(f"[{key}]")
        for name, m in block.items():
            print(f"  {name:6s} {fmt(m)}")
    print("\n=== inference cost (estimates; 45 nm per-op energies, no memory/data-movement) ===")
    for name, m in results["cost"].items():
        print(f"  {name:6s} {fmt(m)}")
    print("  note: SNN per-step = streaming cost per new sample; GBT recomputes its whole window every step.")
    print(f"  note: Yan et al. threshold ({en['spike_rate_threshold']:.1%}) is derived for T in [5,10]; "
          f"this run uses T={w}, so the comparison is indicative only.")
    results["thresholds"] = {"snn": thr_snn, "gbt": thr_gbt}
    results["elapsed_s"] = round(time.time() - t0, 1)
    print(f"\nelapsed: {results['elapsed_s']} s")
    return results


def check(results: dict) -> None:
    """Sanity gates for CI: pipeline ran, metrics are finite, SNN learned something."""
    base = results["missing=0%"]
    assert base["snn"]["events"] > 0, "no test events"
    for name in ("snn", "gbt", "cusum"):
        assert 0.0 <= base[name]["detection_rate"] <= 1.0
    assert base["snn"]["detection_rate"] >= 0.5, f"SNN detected too few events: {base['snn']}"
    assert results["cost"]["snn"]["synops_per_step"] > 0
    assert 0.0 < results["cost"]["snn"]["hidden_spike_rate"] < 1.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"))
    ap.add_argument("--out", default=None, help="optional path to write results JSON")
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    res = run(yaml.safe_load(open(a.config)))
    check(res)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print("SMOKE TEST PASSED")
