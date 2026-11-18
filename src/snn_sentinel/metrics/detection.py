"""Event-level detection metrics: lead time, detection rate, false alarms per day."""
from __future__ import annotations

import numpy as np


def alarm_segments(alarm: np.ndarray, min_gap: int = 0) -> list[tuple[int, int]]:
    """Contiguous runs of alarm==1 as [start, end), merging runs separated by < min_gap steps."""
    a = np.asarray(alarm).astype(bool)
    edges = np.flatnonzero(np.diff(np.r_[0, a.astype(int), 0]))
    segs = list(zip(edges[::2], edges[1::2]))
    merged: list[list[int]] = []
    for s, e in segs:
        if merged and s - merged[-1][1] < min_gap:
            merged[-1][1] = e
        else:
            merged.append([int(s), int(e)])
    return [(s, e) for s, e in merged]


def event_metrics(alarm: np.ndarray, events: list, steps_per_day: int = 1440, min_gap: int = 10,
                  y: np.ndarray | None = None) -> dict:
    """alarm: per-step 0/1 over the split. events: objects with onset/critical/end (split-relative).
    Lead time (steps) = critical - first alarm inside [onset, end); positive = warned before critical."""
    segs = alarm_segments(alarm, min_gap)
    leads, detected, early = [], 0, 0
    for ev in events:
        hit = np.flatnonzero(alarm[ev.onset:ev.end])
        if len(hit):
            detected += 1
            lead = ev.critical - (ev.onset + int(hit[0]))
            leads.append(lead)
            early += int(lead > 0)
    fa = sum(1 for s, e in segs if not any(s < ev.end and e > ev.onset for ev in events))
    days = len(alarm) / steps_per_day
    out = {
        "events": len(events),
        "detection_rate": detected / max(len(events), 1),
        "early_rate": early / max(len(events), 1),
        "mean_lead_steps": float(np.mean(leads)) if leads else float("nan"),
        "median_lead_steps": float(np.median(leads)) if leads else float("nan"),
        "false_alarms": fa,
        "false_alarms_per_day": fa / max(days, 1e-9),
    }
    if y is not None:
        tp = int(((alarm == 1) & (y == 1)).sum()); fp = int(((alarm == 1) & (y == 0)).sum())
        fn = int(((alarm == 0) & (y == 1)).sum())
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        out["point_f1"] = 2 * p * r / max(p + r, 1e-9)   # reference only; event metrics are primary
    return out
