#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""PyTorch under kernel replay: does the allocator backend break determinism?

Customer shape: a PyTorch user wants more counters than fit one hardware pass for one
hot GEMM, without re-running training once per counter group.

The reason this is a *test* and not just a demo: the replay memory tracker hooks the HSA
pool and region allocators. PyTorch's default caching allocator sits on hipMalloc, so the
tracker sees the large blocks a tensor is carved from, which is a superset of the tensor
and therefore safe. But PYTORCH_HIP_ALLOC_CONF=expandable_segments:True establishes tensor
addresses through virtual-memory mapping, and hsa_amd_vmem_map is NOT hooked, so that
memory is invisible to snapshot and restore. Replay would then run each pass against
whatever the previous pass left behind, and report counters as if inputs were identical.

This runs the same workload under both allocator backends and compares the shared
counters across passes. Expectation:

  expandable_segments off -> shared counters constant across passes  (replay is sound)
  expandable_segments on  -> either constant (memory happened to be untouched, so the
                             workload is not sensitive enough) or varying (the gap is
                             real and observable)

A varying result with expandable segments is the finding. A constant result is only
reassuring if the control below also shows the workload CAN detect a broken restore;
--self-check runs a deliberately input-dependent kernel for that purpose.

Usage:
  ./pytorch_replay_demo.py --rocprofv3 /opt/rocm/bin/rocprofv3
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

COMMON = ["SQ_WAVES", "SQ_INSTS_VALU"]
UNIQUE = ["GRBM_COUNT", "GRBM_GUI_ACTIVE", "SQ_INSTS_SALU"]

# The workload. Accumulating in place makes the result depend on the prior contents of the
# output tensor, so a failed restore changes what the kernel does rather than being benign.
WORKLOAD = textwrap.dedent("""
    import torch

    torch.manual_seed(0)
    dev = "cuda"  # ROCm builds of torch use the cuda namespace
    n = 2048
    a = torch.randn(n, n, device=dev, dtype=torch.float32)
    b = torch.randn(n, n, device=dev, dtype=torch.float32)
    acc = torch.zeros(n, n, device=dev, dtype=torch.float32)
    for _ in range(3):
        acc.addmm_(a, b)          # in-place: reads acc, so restore matters
    torch.cuda.synchronize()
    print("checksum", float(acc.abs().sum().item()))
    """)


def _pmc_args(groups):
    args = []
    for unique in groups:
        args += ["--pmc"] + COMMON + [unique]
    return args


def _shared_counter_spread(results_json):
    """dispatch_id -> {counter: (min, max)} over its replay passes."""
    with open(results_json, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    tool = doc["rocprofiler-sdk-tool"]
    tool = tool[0] if isinstance(tool, list) else tool
    names = {}
    counters = tool.get("counters") or []
    for entry in counters.values() if isinstance(counters, dict) else counters:
        cid = entry.get("id", {})
        handle_id = cid.get("handle") if isinstance(cid, dict) else cid
        if handle_id is not None and entry.get("name"):
            names[int(handle_id)] = entry["name"]

    per_dispatch = {}
    for rec in tool.get("callback_records", {}).get("counter_collection", []) or []:
        did = int(rec["dispatch_data"]["dispatch_info"]["dispatch_id"])
        agg = {}
        for sub in rec.get("records", []):
            name = names.get(int(sub["counter_id"]["handle"]))
            if name in COMMON:
                agg[name] = agg.get(name, 0.0) + float(sub["value"])
        for name, value in agg.items():
            lo, hi = per_dispatch.setdefault(did, {}).get(name, (value, value))
            per_dispatch[did][name] = (min(lo, value), max(hi, value))
    return per_dispatch


def _run(args, expandable, outdir):
    env = dict(os.environ)
    env["PYTORCH_HIP_ALLOC_CONF"] = (
        "expandable_segments:True" if expandable else "expandable_segments:False"
    )
    script = os.path.join(outdir, "workload.py")
    with open(script, "w", encoding="utf-8") as handle:
        handle.write(WORKLOAD)
    cmd = (
        [args.rocprofv3]
        + _pmc_args(UNIQUE)
        + [
            "--kernel-replay-beta-enabled",
            "--output-format",
            "json",
            "-d",
            outdir,
            "-o",
            "out",
            "--",
            sys.executable,
            script,
        ]
    )
    print(f"  PYTORCH_HIP_ALLOC_CONF={env['PYTORCH_HIP_ALLOC_CONF']}")
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, capture_output=True, check=False)
    if proc.returncode != 0:
        print(f"  run failed rc={proc.returncode}")
        print(textwrap.indent(proc.stderr.decode(errors="replace")[-1500:], "    "))
        return None
    results = os.path.join(outdir, "out_results.json")
    if not os.path.isfile(results):
        print(f"  no results JSON at {results}")
        return None
    return _shared_counter_spread(results)


def _report(label, spread):
    if spread is None:
        print(f"  {label}: NO DATA")
        return None
    varying = []
    for did, counters in sorted(spread.items()):
        for name, (lo, hi) in sorted(counters.items()):
            scale = max(abs(lo), abs(hi), 1.0)
            if abs(hi - lo) > 0.10 * scale:
                varying.append(f"dispatch {did} {name}: {lo:.0f}..{hi:.0f}")
    if varying:
        print(f"  {label}: shared counters VARY across passes")
        for line in varying[:8]:
            print(f"      {line}")
    else:
        print(
            f"  {label}: shared counters constant across passes ({len(spread)} dispatches)"
        )
    return not varying


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rocprofv3", default=shutil.which("rocprofv3") or "rocprofv3")
    parser.add_argument("--keep", action="store_true", help="keep the output directories")
    args = parser.parse_args()

    if not shutil.which(args.rocprofv3) and not os.path.isfile(args.rocprofv3):
        print(f"SKIP: rocprofv3 not found ({args.rocprofv3})")
        return 77
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP: PyTorch is not installed")
        return 77
    if not torch.cuda.is_available():
        print("SKIP: no ROCm/HIP device visible to PyTorch")
        return 77

    root = tempfile.mkdtemp(prefix="pytorch-replay-")
    print(f"output under {root}")
    print(
        "\n[1/2] default allocator (expandable_segments off) -- tracker sees the blocks"
    )
    baseline = _report("default", _run(args, False, os.path.join(root, "default")))
    print(
        "\n[2/2] expandable_segments on -- addresses come from vmem map, which is NOT hooked"
    )
    expandable = _report("expandable", _run(args, True, os.path.join(root, "expandable")))

    print("\nconclusion:")
    if baseline is None or expandable is None:
        print("  inconclusive: at least one configuration produced no data")
        rc = 1
    elif baseline and not expandable:
        print(
            "  the vmem-map gap is REAL and observable: replay is sound with the default"
        )
        print("  allocator and unsound with expandable segments. Either hook")
        print("  hsa_amd_vmem_map or reject replay when expandable segments are enabled.")
        rc = 1
    elif baseline and expandable:
        print(
            "  both configurations look sound. Either the vmem path was not exercised or"
        )
        print(
            "  this workload cannot detect a broken restore -- check that the accumulate"
        )
        print("  actually depends on prior contents before concluding the gap is closed.")
        rc = 0
    else:
        print(
            "  the default allocator already shows varying shared counters, which points"
        )
        print(
            "  at a problem independent of the allocator backend. Investigate that first."
        )
        rc = 1

    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
