"""Inference-cost accounting in the NeuroBench spirit (arXiv:2304.04640 / Nat Comms 2025):
report hardware-independent counts first (SynOps / MACs / activation sparsity), then
*estimated* energy from per-op constants. Per-op constants ignore memory and data movement,
which Yan et al. (arXiv:2409.08290) show can dominate; treat estimates as optimistic."""
from __future__ import annotations


def snn_cost(stats: dict, cfg_energy: dict, per: str = "step") -> dict:
    """stats from SpikingSentinel.last_stats (summed over a batch of windows)."""
    T, W = stats["T"], stats["windows"]
    denom = W * T if per == "step" else W
    synops = stats["synops"] / denom
    hidden_rate = stats["hidden_spikes"] / max(stats["neuron_steps_hidden"], 1)
    input_rate = stats["in_spikes"] / max(stats["input_steps"], 1)
    out = {
        "synops_per_" + per: synops,
        "hidden_spike_rate": hidden_rate,
        "input_event_rate": input_rate,
        "est_energy_pj_per_" + per + "_45nm_ac": synops * cfg_energy["e_ac_pj"],
        "below_yan_threshold": hidden_rate < cfg_energy["spike_rate_threshold"],
    }
    if cfg_energy.get("e_synop_loihi2_pj"):
        out["est_energy_pj_per_" + per + "_loihi2"] = synops * cfg_energy["e_synop_loihi2_pj"]
    return out


def ann_cost(macs: float, cfg_energy: dict, per: str = "window") -> dict:
    return {"macs_per_" + per: macs, "est_energy_pj_per_" + per + "_45nm_mac": macs * cfg_energy["e_mac_pj"]}


def ops_cost(ops: float, cfg_energy: dict, per: str = "window") -> dict:
    """Add/compare-only models (CUSUM, tree ensembles): price each op as an accumulate."""
    return {"ops_per_" + per: ops, "est_energy_pj_per_" + per + "_45nm_ac": ops * cfg_energy["e_ac_pj"]}
