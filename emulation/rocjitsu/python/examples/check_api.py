# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run with --mode worker, global, or hardware; see ../README.md."""

import argparse
import json
import os
from pathlib import Path

import rocjitsu


def report(result, expected_arch, simulated):
    assert result["arch"].split(":")[0] == expected_arch, result
    assert result["simulated"] == simulated, result
    print(
        json.dumps({key: value for key, value in result.items() if key != "output"}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("worker", "global", "hardware"), required=True
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--hardware-index", type=int)
    args = parser.parse_args()
    config = None
    if args.config:
        config = json.loads(args.config.read_text())
        config["cpu_thread_budget"] = 2
        config["vm"]["gpu"]["device"]["marketing_name"] = (
            "rocjitsu Python simulated " + args.arch
        )
    if args.mode == "global":
        before = os.getpid()
        rocjitsu.enable(config)
        assert os.getpid() == before

    # Global activation precedes framework imports and all device discovery.
    import torch
    from toy_kernels import check, gpu_tensor

    def hardware():
        for framework in ("pytorch", "triton"):
            result = check(framework, device=args.hardware_index)
            assert result["pid"] == os.getpid()
            report(result, "gfx1201", False)

    print(
        json.dumps({"torch": torch.__version__, "hip": torch.version.hip}), flush=True
    )
    if args.mode == "hardware":
        hardware()
    elif args.mode == "global":
        for framework in ("pytorch", "triton"):
            result = check(framework)
            assert result["pid"] == before
            report(result, args.arch, True)
        rocjitsu.enable(config)
        print("global activation retained interpreter PID", flush=True)
    else:
        if args.hardware_index is not None:
            hardware()
            try:
                rocjitsu.enable(config)
            except RuntimeError as error:
                assert "initialization" in str(error)
                print("late global activation rejected: " + str(error), flush=True)
            else:
                raise AssertionError("late global activation accepted")
        for framework in ("pytorch", "triton"):
            result = rocjitsu.run(check, framework, config=config, timeout=120)
            assert result["pid"] != os.getpid()
            report(result, args.arch, True)
        with rocjitsu.Session(config) as session:
            for framework in ("pytorch", "triton", "pytorch", "triton"):
                result = session.run(check, framework, timeout=120)
                assert result["pid"] == session.pid
                report(result, args.arch, True)
                if args.hardware_index is not None:
                    hardware()
            # GPU allocations cannot escape the interpreter that owns them.
            try:
                session.run(gpu_tensor, timeout=120)
            except rocjitsu.RemoteError as error:
                assert "GPU tensors" in str(error)
            else:
                raise AssertionError("GPU tensor escaped the worker")
            cpu = torch.arange(8, dtype=torch.float32)
            returned = session.run(lambda x: (x.cuda() + 1).cpu(), cpu, timeout=120)
            torch.testing.assert_close(returned, cpu + 1, rtol=0, atol=0)
        if args.hardware_index is not None:
            hardware()


if __name__ == "__main__":
    main()
