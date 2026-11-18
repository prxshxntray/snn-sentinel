import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from snn_sentinel.encoding import sigma_delta_encode  # noqa: E402
from snn_sentinel.metrics.detection import event_metrics  # noqa: E402
from snn_sentinel.data.faults import Event  # noqa: E402
from snn_sentinel.models.baselines import TCN, LSTMDetector, TinyTransformer, count_macs  # noqa: E402


def test_encoder_is_sparse_on_flat_signal():
    x = np.zeros((100, 3), dtype=np.float32)
    assert sigma_delta_encode(x, 0.1).sum() == 0


def test_lead_time_and_false_alarms():
    alarm = np.zeros(1440, dtype=int); alarm[105:120] = 1; alarm[800:805] = 1
    m = event_metrics(alarm, [Event("x", 0, 100, 110, 150)])
    assert m["detection_rate"] == 1.0 and m["mean_lead_steps"] == 5 and m["false_alarms"] == 1


def test_deep_baselines_forward_and_macs():
    x = torch.randn(2, 32, 10)
    for m in (TCN(10), LSTMDetector(10), TinyTransformer(10, w=32)):
        assert m(x).shape == (2, 2) and count_macs(m, x) > 0


def test_smoke_end_to_end():
    import smoke_test
    cfg = yaml.safe_load(open(ROOT / "configs" / "smoke.yaml"))
    smoke_test.check(smoke_test.run(cfg))
