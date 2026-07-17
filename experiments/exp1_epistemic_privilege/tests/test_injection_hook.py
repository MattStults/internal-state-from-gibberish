"""CPU unit test for the concept-injection hook: legacy all-position vs generation-only.
No model, no GPU, no inference. Verifies the generation-only hook injects ONLY generated
positions (>= prompt_len) and never the prompt prefix -- the open-introspection post-02
convention (all-position injection contaminates the prompt/probe).

Run:  .venv/bin/python tests/test_injection_hook.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
from common import _injection_hook  # noqa: E402

D = 4
V = torch.ones(D)
ALPHA = 3.0


def layer_out(L):
    """Mimic a decoder-layer forward output: a tuple whose [0] is (batch, seq, hidden)."""
    return (torch.zeros(1, L, D),)


checks = []

# --- legacy all-position (prompt_len=None): every position gets alpha*v ---
hk = _injection_hook(V, ALPHA)
hs = hk(None, None, layer_out(5))[0]
checks.append(("legacy: injects all positions", torch.allclose(hs, torch.full((1, 5, D), ALPHA))))

# --- generation-only, full forward [prompt + generated]: prompt untouched, generated injected ---
P = 3
hk = _injection_hook(V, ALPHA, prompt_len=P)
hs = hk(None, None, layer_out(5))[0]
checks.append(("gen-only: prompt positions NOT injected", torch.allclose(hs[:, :P, :], torch.zeros(1, P, D))))
checks.append(("gen-only: generated positions injected", torch.allclose(hs[:, P:, :], torch.full((1, 5 - P, D), ALPHA))))

# --- generation-only, pure prompt prefill (L == prompt_len): nothing injected ---
hs = hk(None, None, layer_out(P))[0]
checks.append(("gen-only: pure prompt prefill untouched", torch.allclose(hs, torch.zeros(1, P, D))))

# --- generation-only, KV-cached decode step (L == 1): injected ---
hs = hk(None, None, layer_out(1))[0]
checks.append(("gen-only: cached decode step injected", torch.allclose(hs, torch.full((1, 1, D), ALPHA))))

# --- non-tuple output form is handled too ---
hk = _injection_hook(V, ALPHA, prompt_len=P)
out = hk(None, None, torch.zeros(1, 5, D))
checks.append(("gen-only: bare-tensor output supported", torch.allclose(out[:, :P, :], torch.zeros(1, P, D))
               and torch.allclose(out[:, P:, :], torch.full((1, 5 - P, D), ALPHA))))

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print(f"\n{'ALL PASS' if all(c for _, c in checks) else 'FAILURES PRESENT'} ({sum(c for _, c in checks)}/{len(checks)})")
sys.exit(0 if all(c for _, c in checks) else 1)
