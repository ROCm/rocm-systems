# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compare actual shared-image WMMA callbacks with a fresh-process scalar oracle."""

import os
import re
import subprocess
import sys


def parse_accesses(text):
    accesses = {}
    for access in text.split(",") if text else ():
        reg, lanes, byte_mask = access.split(":")
        reg = int(reg)
        if reg in accesses:
            raise AssertionError(f"Duplicate observed VGPR: {text}")
        accesses[reg] = (int(lanes, 16), int(byte_mask, 16))
    return accesses


def probe(driver, scenario, backend, force_scalar, *, observe=False):
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
            [
                driver,
                form,
                "0",
                str(scenario),
                expected,
                "--observe" if observe else "--dump",
            ],
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
        pattern = r"form=(\w+) scenario=(\d+) module=(\w+)[^\n]*\nwords=([0-9a-f]+)\n"
        if observe:
            pattern += r"reads=([^\n]*)\nwrites=([^\n]*)\n"
        parsed = re.findall(pattern, process.stdout)
        if len(parsed) != (5 if form == "all" else 1):
            raise AssertionError(f"Incomplete probe output: {process.stdout}")
        for match in parsed:
            name, observed_scenario, module, words = match[:4]
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
            if observe:
                results[name] = (
                    results[name],
                    parse_accesses(match[4]),
                    parse_accesses(match[5]),
                )
    return results


def check_observations(driver):
    comparisons = 0
    for scenario in (0, 3, 8):
        scalar = probe(driver, scenario, "v3", 1, observe=True)
        scalar_values = probe(driver, scenario, "v3", 1)
        for name, (words, reads, writes) in scalar.items():
            if words != scalar_values[name]:
                raise AssertionError(
                    f"scalar {name} scenario={scenario}: observation or D poison changed output"
                )
            output_regs = 8 if name in ("f32_f16", "f32_bf16") else 4
            acc_regs = 4 if name in ("f16_f16", "bf16_bf16") else 8
            sources = set(range(8)) | set(range(32, 40))
            if scenario != 8:
                sources |= set(range(96, 96 + acc_regs))
            dst = 96 if scenario == 3 else 64
            expected_reads = {reg: (0xFFFFFFFF, 0xF) for reg in sources}
            expected_writes = {
                reg: (0xFFFFFFFF, 0xF) for reg in range(dst, dst + output_regs)
            }
            if reads != expected_reads or writes != expected_writes:
                raise AssertionError(
                    f"scalar {name} scenario={scenario}: "
                    f"{reads=} {expected_reads=} {writes=} {expected_writes=}"
                )
        for backend in ("v3", "v4", "auto"):
            observed = probe(driver, scenario, backend, 0, observe=True)
            if observed is None:
                return None
            if observed != scalar:
                for name in scalar:
                    if observed[name] != scalar[name]:
                        raise AssertionError(
                            f"{backend} {name} observer scenario={scenario}: "
                            f"value_match={observed[name][0] == scalar[name][0]} "
                            f"actual_reads={observed[name][1]} scalar_reads={scalar[name][1]} "
                            f"actual_writes={observed[name][2]} scalar_writes={scalar[name][2]}"
                        )
            comparisons += len(scalar)
    return comparisons


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
    observations = check_observations(driver)
    if observations is None:
        print("Skipped: x86-v4 provider requires CPU and OS AVX-512 support")
        return 77
    print(
        f"Passed: {comparisons} WMMA form/scenario/backend value comparisons "
        f"and {observations} observer comparisons against scalar"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
