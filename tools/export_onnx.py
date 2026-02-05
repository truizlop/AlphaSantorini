#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from safetensors import safe_open
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: safetensors. Install with: pip install safetensors") from exc


class SantoriniNetV1(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.policyHead = nn.Linear(hidden_dim, 153)
        self.valueHead = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        policy = torch.softmax(self.policyHead(x), dim=-1)
        value = torch.tanh(self.valueHead(x))
        return policy, value


class SantoriniNetV2(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.policyHead1 = nn.Linear(hidden_dim, hidden_dim)
        self.policyHead2 = nn.Linear(hidden_dim, 153)
        self.valueHead1 = nn.Linear(hidden_dim, hidden_dim)
        self.valueHead2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        policy = self.policyHead2(torch.relu(self.policyHead1(x)))
        policy = torch.softmax(policy, dim=-1)
        value = torch.tanh(self.valueHead2(torch.relu(self.valueHead1(x))))
        return policy, value


def normalize_key(key: str) -> str:
    return key.replace("/", ".")


def load_tensors(path: Path) -> dict:
    tensors = {}
    with safe_open(path, framework="numpy") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def find_key(keys, pattern: str) -> str:
    for key in keys:
        if normalize_key(key) == pattern:
            return key
    return ""


def assign_linear(linear: nn.Linear, weight: np.ndarray, bias: np.ndarray):
    w = torch.from_numpy(weight)
    b = torch.from_numpy(bias)
    out_dim, in_dim = linear.weight.shape
    if w.shape == (out_dim, in_dim):
        linear.weight.copy_(w)
    elif w.shape == (in_dim, out_dim):
        linear.weight.copy_(w.t())
    else:
        raise ValueError(f"Unexpected weight shape {w.shape} for expected {(out_dim, in_dim)}")
    if b.shape != (out_dim,):
        raise ValueError(f"Unexpected bias shape {b.shape} for expected {(out_dim,)}")
    linear.bias.copy_(b)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Santorini MLX safetensors to ONNX")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-dim", type=int, default=200)
    args = parser.parse_args()

    tensors = load_tensors(args.checkpoint)
    keys = list(tensors.keys())

    def key_for(name: str) -> tuple:
        w = find_key(keys, f"{name}.weight")
        b = find_key(keys, f"{name}.bias")
        if not w or not b:
            raise KeyError(f"Missing weights for {name} (found: {keys})")
        return w, b

    w1_key, b1_key = key_for("layer1")
    w1 = tensors[w1_key]

    input_dim = args.input_dim
    if w1.shape[0] == input_dim:
        hidden_dim = w1.shape[1]
    elif w1.shape[1] == input_dim:
        hidden_dim = w1.shape[0]
    else:
        hidden_dim = w1.shape[0]
        input_dim = w1.shape[1]

    use_v2 = bool(find_key(keys, "policyHead1.weight")) and bool(find_key(keys, "valueHead1.weight"))
    if use_v2:
        model = SantoriniNetV2(input_dim=input_dim, hidden_dim=hidden_dim)
        layers = ["layer1", "layer2", "layer3", "policyHead1", "policyHead2", "valueHead1", "valueHead2"]
    else:
        model = SantoriniNetV1(input_dim=input_dim, hidden_dim=hidden_dim)
        layers = ["layer1", "layer2", "layer3", "policyHead", "valueHead"]

    model.eval()

    with torch.no_grad():
        for layer_name in layers:
            w_key, b_key = key_for(layer_name)
            assign_linear(getattr(model, layer_name), tensors[w_key], tensors[b_key])

    dummy = torch.zeros(1, input_dim, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={"input": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
    )

    arch = "v2" if use_v2 else "v1"
    print(f"Exported ONNX model to {args.output} (input_dim={input_dim}, hidden_dim={hidden_dim}, arch={arch})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
