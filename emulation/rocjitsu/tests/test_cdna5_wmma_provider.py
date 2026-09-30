# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compare actual shared-image WMMA callbacks with a fresh-process scalar oracle."""

import os
import re
import subprocess
import sys


def probe(driver, scenario, backend, force_scalar):
    env = dict(
        os.environ,
        RJ_CDNA5_WMMA_BACKEND=backend,
        RJ_MATH_BACKEND="auto",
        RJ_FORCE_SCALAR=str(force_scalar),
    )
    expected = "main" if force_scalar or backend == "v3" else "provider"
    forms = ("0", "2", "4") if scenario in (5, 6, 7) else ("all",)
    results = {}
    for form in forms:
        process = subprocess.run(
            [driver, form, "0", str(scenario), expected, "--dump"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode == 77 and backend == "v4" and not force_scalar:
            return None
        if process.returncode:
            raise AssertionError(
                f"{backend=} {force_scalar=} {scenario=} {form=}: "
                f"{process.stdout}{process.stderr}"
            )
        parsed = re.findall(
            r"form=(\w+) scenario=(\d+) module=(\w+)[^\n]*\nwords=([0-9a-f]+)\n",
            process.stdout,
        )
        if len(parsed) != (5 if form == "all" else 1):
            raise AssertionError(f"Incomplete probe output: {process.stdout}")
        for name, observed_scenario, module, words in parsed:
            count = 256 if name in ("f32_f16", "f32_bf16") else 128
            if (
                int(observed_scenario) != scenario
                or module != expected
                or len(words) != count * 8
            ):
                raise AssertionError(f"Invalid probe output: {process.stdout}")
            results[name] = tuple(
                int(words[i : i + 8], 16) for i in range(0, len(words), 8)
            )
    return results


def main():
    if len(sys.argv) != 2:
        raise ValueError("usage: test_cdna5_wmma_provider.py <probe-driver>")
    driver = sys.argv[1]
    comparisons = 0
    for scenario in range(15):
        # An explicit v4 request must fail on loader/ABI errors; the driver
        # returns 77 only for unsupported CPU/OS AVX-512 state.
        v4 = probe(driver, scenario, "v4", 0)
        if v4 is None:
            print("Skipped: x86-v4 provider requires CPU and OS AVX-512 support")
            return 77
        scalar = probe(driver, scenario, "v3", 1)
        for backend, actual in (
            ("v4", v4),
            ("auto", probe(driver, scenario, "auto", 0)),
            ("v3", probe(driver, scenario, "v3", 0)),
        ):
            if actual != scalar:
                for name in scalar:
                    if actual[name] != scalar[name]:
                        lane = next(
                            i
                            for i, (a, b) in enumerate(zip(actual[name], scalar[name]))
                            if a != b
                        )
                        raise AssertionError(
                            f"{backend} {name} scenario={scenario} word={lane}: "
                            f"actual=0x{actual[name][lane]:08x} scalar=0x{scalar[name][lane]:08x}"
                        )
            comparisons += len(scalar)
    print(
        f"Passed: {comparisons} WMMA form/scenario/backend comparisons against scalar"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
