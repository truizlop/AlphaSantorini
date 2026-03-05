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


# ---------------------------------------------------------------------------
# V1 / V2 fully-connected architectures (legacy)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Convolutional residual architecture (matches SantoriniNet.swift)
# ---------------------------------------------------------------------------

class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(residual + out)


class SantoriniNetConv(nn.Module):
    """Mirrors the MLX SantoriniNet convolutional architecture.

    Input:  [batch, 5, 5, 9]  (NHWC — matches WASM encodeState output)
    Output: policy [batch, 153], value [batch, 1]
    """

    def __init__(self, filters: int = 256, residual_blocks: int = 8):
        super().__init__()
        # Input block
        self.input_conv = nn.Conv2d(9, filters, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(filters)
        # Residual tower
        self.res_tower = nn.ModuleList(
            [ConvResidualBlock(filters) for _ in range(residual_blocks)]
        )
        # Policy head
        self.policy_conv = nn.Conv2d(filters, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_linear = nn.Linear(2 * 5 * 5, 153)
        # Value head
        self.value_conv = nn.Conv2d(filters, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_linear1 = nn.Linear(5 * 5, 64)
        self.value_linear2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [batch, 5, 5, 9] NHWC
        x = x.permute(0, 3, 1, 2)  # -> [batch, 9, 5, 5] NCHW
        x = torch.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_tower:
            x = block(x)
        # Policy head — permute back to NHWC before flatten so linear weights
        # match the HWC flattening order used during MLX training.
        p = torch.relu(self.policy_bn(self.policy_conv(x)))  # [batch, 2, 5, 5]
        p = p.permute(0, 2, 3, 1).reshape(-1, 50)            # NHWC flatten
        p = torch.softmax(self.policy_linear(p), dim=-1)      # [batch, 153]
        # Value head
        v = torch.relu(self.value_bn(self.value_conv(x)))     # [batch, 1, 5, 5]
        v = v.permute(0, 2, 3, 1).reshape(-1, 25)             # NHWC flatten
        v = torch.tanh(self.value_linear2(torch.relu(self.value_linear1(v))))
        return p, v


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------

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


def assign_conv(conv: nn.Conv2d, weight: np.ndarray, bias: np.ndarray):
    """Load MLX conv weights [O, kH, kW, I] into PyTorch conv [O, I, kH, kW]."""
    w = torch.from_numpy(np.ascontiguousarray(np.transpose(weight, (0, 3, 1, 2))))
    b = torch.from_numpy(bias)
    conv.weight.copy_(w)
    conv.bias.copy_(b)


def assign_bn(bn: nn.BatchNorm2d, tensors: dict, prefix: str, keys: list):
    """Load BatchNorm parameters including running stats."""
    w_key = find_key(keys, f"{prefix}.weight")
    b_key = find_key(keys, f"{prefix}.bias")
    rm_key = find_key(keys, f"{prefix}.running_mean")
    rv_key = find_key(keys, f"{prefix}.running_var")

    if w_key:
        bn.weight.copy_(torch.from_numpy(tensors[w_key]))
    if b_key:
        bn.bias.copy_(torch.from_numpy(tensors[b_key]))
    if rm_key:
        bn.running_mean.copy_(torch.from_numpy(tensors[rm_key]))
    if rv_key:
        bn.running_var.copy_(torch.from_numpy(tensors[rv_key]))


def is_conv_architecture(keys: list) -> bool:
    return bool(find_key(keys, "inputBlock.conv.weight"))


# ---------------------------------------------------------------------------
# Conv model loading
# ---------------------------------------------------------------------------

def load_conv_model(tensors: dict, keys: list) -> SantoriniNetConv:
    """Build and load weights for the convolutional architecture."""
    # Detect number of residual blocks
    num_blocks = 0
    while find_key(keys, f"resTower.{num_blocks}.conv1.weight"):
        num_blocks += 1

    # Detect filter count from input conv
    input_conv_w = tensors[find_key(keys, "inputBlock.conv.weight")]
    filters = input_conv_w.shape[0]

    print(f"Detected conv architecture: {filters} filters, {num_blocks} residual blocks")
    model = SantoriniNetConv(filters=filters, residual_blocks=num_blocks)
    model.eval()

    with torch.no_grad():
        # Input block
        assign_conv(
            model.input_conv,
            tensors[find_key(keys, "inputBlock.conv.weight")],
            tensors[find_key(keys, "inputBlock.conv.bias")],
        )
        assign_bn(model.input_bn, tensors, "inputBlock.norm", keys)

        # Residual tower
        for i in range(num_blocks):
            block = model.res_tower[i]
            assign_conv(
                block.conv1,
                tensors[find_key(keys, f"resTower.{i}.conv1.weight")],
                tensors[find_key(keys, f"resTower.{i}.conv1.bias")],
            )
            assign_bn(block.bn1, tensors, f"resTower.{i}.norm1", keys)
            assign_conv(
                block.conv2,
                tensors[find_key(keys, f"resTower.{i}.conv2.weight")],
                tensors[find_key(keys, f"resTower.{i}.conv2.bias")],
            )
            assign_bn(block.bn2, tensors, f"resTower.{i}.norm2", keys)

        # Policy head
        assign_conv(
            model.policy_conv,
            tensors[find_key(keys, "policyHead.conv.weight")],
            tensors[find_key(keys, "policyHead.conv.bias")],
        )
        assign_bn(model.policy_bn, tensors, "policyHead.norm", keys)
        assign_linear(
            model.policy_linear,
            tensors[find_key(keys, "policyHead.linear.weight")],
            tensors[find_key(keys, "policyHead.linear.bias")],
        )

        # Value head
        assign_conv(
            model.value_conv,
            tensors[find_key(keys, "valueHead.conv.weight")],
            tensors[find_key(keys, "valueHead.conv.bias")],
        )
        assign_bn(model.value_bn, tensors, "valueHead.norm", keys)
        assign_linear(
            model.value_linear1,
            tensors[find_key(keys, "valueHead.linear1.weight")],
            tensors[find_key(keys, "valueHead.linear1.bias")],
        )
        assign_linear(
            model.value_linear2,
            tensors[find_key(keys, "valueHead.linear2.weight")],
            tensors[find_key(keys, "valueHead.linear2.bias")],
        )

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Export Santorini MLX safetensors to ONNX")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-dim", type=int, default=200,
                        help="Input dimension for legacy FC architectures")
    args = parser.parse_args()

    tensors = load_tensors(args.checkpoint)
    keys = list(tensors.keys())

    if is_conv_architecture(keys):
        model = load_conv_model(tensors, keys)
        dummy = torch.zeros(1, 5, 5, 9, dtype=torch.float32)
        input_desc = "NHWC [batch, 5, 5, 9]"
        arch = "conv"
    else:
        # Legacy FC architectures
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
            arch = "v2"
        else:
            model = SantoriniNetV1(input_dim=input_dim, hidden_dim=hidden_dim)
            layers = ["layer1", "layer2", "layer3", "policyHead", "valueHead"]
            arch = "v1"

        model.eval()
        with torch.no_grad():
            for layer_name in layers:
                w_key, b_key = key_for(layer_name)
                assign_linear(getattr(model, layer_name), tensors[w_key], tensors[b_key])

        dummy = torch.zeros(1, input_dim, dtype=torch.float32)
        input_desc = f"[batch, {input_dim}]"

    # Quick sanity check
    with torch.no_grad():
        policy, value = model(dummy)
        print(f"Sanity check — policy sum: {policy.sum().item():.4f}, value: {value.item():.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Export to a temporary path first, then convert to a single self-contained
    # file (the dynamo exporter may split weights into an external .data file
    # which onnxruntime-web cannot fetch automatically).
    import tempfile, onnx
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "model.onnx"
        torch.onnx.export(
            model,
            dummy,
            tmp_path,
            input_names=["input"],
            output_names=["policy", "value"],
            dynamic_axes={"input": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
            opset_version=17,
        )
        # Reload and save as single file with all data embedded
        onnx_model = onnx.load(str(tmp_path))
        ext_data = tmp_path.with_suffix(".onnx.data")
        if ext_data.exists():
            onnx.load_external_data_for_model(onnx_model, tmpdir)
            print("Converted external data model to single self-contained file")
        onnx.save(onnx_model, str(args.output))

    print(f"Exported ONNX model to {args.output} (arch={arch}, input={input_desc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
