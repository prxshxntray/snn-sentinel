"""Spiking sentinel in snnTorch: sigma-delta events -> LIF hidden layer -> leaky-integrator readout."""
from __future__ import annotations

import snntorch as snn
import torch
from snntorch import surrogate
from torch import nn


class SpikingSentinel(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, beta: float = 0.9, n_out: int = 2):
        super().__init__()
        sg = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(n_in, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=sg)
        self.fc2 = nn.Linear(hidden, n_out)
        self.readout = snn.Leaky(beta=beta, spike_grad=sg, reset_mechanism="none")  # integrator, no spikes used
        self.hidden, self.n_in, self.n_out = hidden, n_in, n_out
        self.last_stats: dict = {}

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: [B, T, n_in] spikes. Returns logits [B, n_out] = readout membrane at the last step."""
        B, T, _ = s.shape
        m1, m2 = self.lif1.init_leaky(), self.readout.init_leaky()
        hid_spikes = torch.zeros((), device=s.device)
        for t in range(T):
            spk1, m1 = self.lif1(self.fc1(s[:, t]), m1)
            _, m2 = self.readout(self.fc2(spk1), m2)
            hid_spikes = hid_spikes + spk1.detach().sum()
        in_spikes = s.detach().sum()
        # Event-driven cost: each input event fans out to `hidden` synapses, each hidden spike to n_out.
        self.last_stats = {
            "in_spikes": float(in_spikes),
            "hidden_spikes": float(hid_spikes),
            "synops": float(in_spikes * self.hidden + hid_spikes * self.n_out),
            "neuron_steps_hidden": B * T * self.hidden,
            "input_steps": B * T * self.n_in,
            "windows": B,
            "T": T,
        }
        return m2
