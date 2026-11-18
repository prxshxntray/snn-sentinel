# snn-sentinel

An event-driven spiking "sentinel" that watches data-centre telemetry (inlet and outlet temperatures, CPU use, fan speed, rack power) and raises early alarms for sudden workload and thermal events. It is benchmarked against change-point, gradient-boosted-tree, TCN, LSTM and small-transformer baselines on:

1. **Detection lead time**: minutes of warning before the critical point.
2. **False alarms per day**.
3. **Robustness to missing sensors**: whole channels dropped at test time.
4. **Inference cost**: SynOps, spike rate and MACs, with *estimated* energy following NeuroBench conventions. It is checked against the spike-rate break-even in Yan et al., "Reconsidering the Energy Efficiency of SNNs" (arXiv:2409.08290).

The research question is energy per unit of useful detection performance. Can an event-driven detector buy the same lead time and false-alarm rate for far fewer operations when the data hall is quiet?

Part of the Wattr stack: Modelica/FMU digital twin → **SNN sentinel** → RL cooling controller → safety shield.

## Quick start (CPU only)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"
python scripts/smoke_test.py            # about 10 s; prints metrics, ends with SMOKE TEST PASSED
pytest -q                               # 4 tests
```

## Layout

```
configs/            smoke.yaml (tiny), default.yaml (full synthetic run)
src/snn_sentinel/
  data/synthetic.py   rack telemetry with diurnal load (stand-in for the FMU digital twin)
  data/faults.py      fault injection (workload_spike, cooling_failure, hot_spot), missing-sensor dropout
  data/loaders.py     synthetic splits, NAB and M100 loaders, causal sliding windows
  encoding.py         send-on-delta (sigma-delta) spike encoder, streaming and causal
  models/snn.py       snnTorch LIF sentinel; logs input events, hidden spikes and SynOps
  models/baselines.py CUSUM, HistGradientBoosting, TCN, LSTM, TinyTransformer, MAC counter
  metrics/detection.py event-level lead time, detection/early rate, false alarms per day, point-F1
  metrics/energy.py   SynOps/MAC/op accounting -> estimated pJ, spike-rate threshold check
  train.py            training loop, threshold picked on the validation split only
scripts/smoke_test.py end-to-end run: tiny SNN + GBT + CUSUM, 0/20/40% missing sensors
tests/                unit tests + smoke test
```

## Evaluation protocol (current)

- Splits are chronological (60/20/20), with z-scoring fitted on train only.
- Labels are 1 from fault onset to fault end. **Lead time = critical time - first alarm** inside the event window, so a positive value means a warning came before the critical point.
- The decision threshold is chosen on validation by point-F1 and never tuned on test. (TODO: also report a fixed false-alarm budget, such as ≤1 per day.)
- Cost is reported per new sample. The SNN streams, so it costs SynOps per step. Windowed models (GBT, TCN, transformer) redo their whole window every step.
- The energy numbers use 45 nm per-op constants (Horowitz, ISSCC 2014: 4.6 pJ fp32 MAC, 0.9 pJ fp32 add). They ignore memory and data movement, so they are **optimistic estimates, not measurements**. The Loihi 2 per-SynOp energy is left `null` in the configs until it is sourced.

## Real data (download separately; not committed)

| Dataset | Use | Link |
|---|---|---|
| NAB: Numenta Anomaly Benchmark | univariate sanity check (`realKnownCause/machine_temperature_system_failure.csv`, `realAWSCloudwatch/ec2_cpu_utilization_*`) | https://github.com/numenta/NAB |
| M100 anomaly-detection subset (CINECA Marconi100, 15-min IPMI aggregates + Nagios labels, 24.8 GB, CC-BY) | main real-world multivariate benchmark | https://zenodo.org/records/7541722 |
| HazardNet (Marconi A2 inlet and outlet temperatures + power) | thermal-hazard lead-time benchmark | https://github.com/MSKazemi/HazardNet |
| Alibaba / Google / Azure cluster traces | realistic workload profiles to drive the twin | https://github.com/alibaba/clusterdata, https://github.com/google/cluster-data, https://github.com/Azure/AzurePublicDataset |

Caveat: Wu & Keogh (arXiv:2009.13807) show that many TSAD benchmarks, NAB included, have trivial or mislabelled anomalies. Use NAB only as a sanity check, and report a trivial baseline (a threshold on the one-step difference) next to every model.

## Roadmap

- [ ] TCN, LSTM and transformer runs (the code is in `models/baselines.py`; wire them into a `scripts/run_benchmark.py`)
- [ ] BOCPD / `ruptures` change-point baseline
- [ ] M100 + HazardNet pipelines
- [ ] FMU digital-twin fault injection (fmpy) in place of `data/synthetic.py`
- [ ] NeuroBench harness metrics (activation sparsity, SynOps), with a Norse cross-check
- [ ] Lava / Loihi 2 deployment estimate (needs Intel INRC access for real hardware)
- [ ] Sweep encoder threshold and T against the Yan et al. spike-rate break-even

Local only. No remote is configured yet.
