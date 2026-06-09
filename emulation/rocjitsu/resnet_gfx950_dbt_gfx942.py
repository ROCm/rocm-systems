#!/usr/bin/env python3
"""ResNet50 numerics helper for gfx950->gfx942 DBT experiments.

This script was tried with:

  uv pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
      "torch[device-gfx942,device-gfx950]==2.12.0+rocm7.14.0a20260528" \
      "torchvision[device-gfx942,device-gfx950]"

Assumption: this is run on a gfx942 host.

The script does not set DBT environment variables. Run it under the environment
you want to test. For the gfx950 guest -> gfx942 host DBT path, use flags like:

  export LD_PRELOAD=/path/to/librocjitsu_hip_intercept_hooks.so
  export RJ_HIP_DEVICE_GFX_OVERRIDE=gfx950
  export HSA_TOOLS_ROCPROFILER_V1_TOOLS=1
  export HSA_TOOLS_REPORT_LOAD_FAILURE=1
  export HSA_TOOLS_LIB=/path/to/librocjitsu_hooks.so
  export RJ_DBT_SOURCE_ISA="gfx950:xnack-"
  export RJ_DBT_TARGET_ISA=gfx942
  export RJ_DBT_LOG=1
  export RJ_DBT_DUMP_SOURCE_DIR=/tmp/rj-dbt-resnet-gfx950-dumps

Three useful ways to run it:

  1. CPU FP32 reference:
       ./resnet_gfx950_dbt_gfx942.py cpu --out /tmp/resnet_cpu.pt

  2. Native GPU FP16:
       ./resnet_gfx950_dbt_gfx942.py gpu --out /tmp/resnet_native.pt

     If native gfx942 device libraries are absent, this is expected to fail
     with an invalid kernel image.

  3. DBT GPU FP16:
       env <DBT flags above> ./resnet_gfx950_dbt_gfx942.py gpu --out /tmp/resnet_dbt.pt

     With native gfx942 device libraries absent, this should still run because
     ROCm selects gfx950 kernels and the HSA hook translates them to gfx942.

Then compare saved outputs:

  ./resnet_gfx950_dbt_gfx942.py compare --ref /tmp/resnet_cpu.pt --other /tmp/resnet_dbt.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ARTIFACT = Path("/tmp/resnet_real_weights_artifact.pt")


def ensure_artifact(path: Path) -> None:
    """Create the shared real-weights/input artifact if it does not exist."""
    if path.exists():
        return

    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    torch.manual_seed(1234)
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).eval()
    x = torch.randn(1, 3, 224, 224)

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "input": x}, path)
    print(f"created artifact {path} with weights {weights}")


def load_model_and_input(artifact: Path):
    import torch
    from torchvision.models import resnet50

    ensure_artifact(artifact)
    data = torch.load(artifact, map_location="cpu")
    model = resnet50(weights=None).eval()
    model.load_state_dict(data["state"])
    return model, data["input"]


def run_cpu(args: argparse.Namespace) -> int:
    import torch

    model, x = load_model_and_input(args.artifact)
    with torch.no_grad():
        y = model(x)

    save_output(y, args.out)
    return 0


def run_gpu(args: argparse.Namespace) -> int:
    import torch

    props = torch.cuda.get_device_properties(0)
    if hasattr(props, "gcnArchName"):
        print(f"gcn {props.gcnArchName}")

    model, x = load_model_and_input(args.artifact)
    torch.backends.cudnn.enabled = True
    model = model.cuda().half()
    x = x.cuda().half()

    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize()

    save_output(y, args.out)
    return 0


def save_output(y, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    y_cpu = y.float().cpu()
    torch.save(y_cpu, path)
    print(
        "saved",
        path,
        tuple(y.shape),
        y.dtype,
        bool(torch.isfinite(y_cpu).all().item()),
        float(y_cpu.sum()),
    )


def compare(args: argparse.Namespace) -> int:
    import torch

    ref = torch.load(args.ref, map_location="cpu")
    other = torch.load(args.other, map_location="cpu")

    d = (ref - other).abs().flatten()
    rel = d / ref.abs().flatten().clamp_min(1e-6)
    print(
        "ref",
        tuple(ref.shape),
        bool(torch.isfinite(ref).all().item()),
        float(ref.sum()),
    )
    print(
        "other",
        tuple(other.shape),
        bool(torch.isfinite(other).all().item()),
        float(other.sum()),
    )
    print(f"max_abs {float(d.max())}")
    print(f"mean_abs {float(d.mean())}")
    print(f"p50 {float(d.quantile(0.5))}")
    print(f"p95 {float(d.quantile(0.95))}")
    print(f"p99 {float(d.quantile(0.99))}")
    print(f"max_rel {float(rel.max())}")
    print(f"mean_rel {float(rel.mean())}")
    print(f"top1 {int(ref.argmax())} {int(other.argmax())}")
    print(
        "top5_equal",
        set(torch.topk(ref.flatten(), 5).indices.tolist())
        == set(torch.topk(other.flatten(), 5).indices.tolist()),
    )
    print(
        "allclose atol=0.01 rtol=0.01",
        bool(torch.allclose(ref, other, atol=1e-2, rtol=1e-2)),
    )
    print(
        "allclose atol=0.03125 rtol=0",
        bool(torch.allclose(ref, other, atol=3.125e-2, rtol=0)),
    )
    print(
        "allclose atol=0.0625 rtol=0",
        bool(torch.allclose(ref, other, atol=6.25e-2, rtol=0)),
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cpu = subparsers.add_parser("cpu", help="Run CPU FP32 reference")
    cpu.add_argument("--out", type=Path, required=True)

    gpu = subparsers.add_parser("gpu", help="Run GPU FP16 under current environment")
    gpu.add_argument("--out", type=Path, required=True)

    comp = subparsers.add_parser("compare", help="Compare two saved output tensors")
    comp.add_argument("--ref", type=Path, required=True)
    comp.add_argument("--other", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "cpu":
        return run_cpu(args)
    if args.command == "gpu":
        return run_gpu(args)
    if args.command == "compare":
        return compare(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
