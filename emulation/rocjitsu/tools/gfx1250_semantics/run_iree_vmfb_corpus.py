#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


EXPORT_RE = re.compile(r"\s*\[\s*\d+\]\s+([^(]+)\(")


@dataclass
class CaseResult:
    name: str
    status: str
    detail: str
    checks: int = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Translate and run the IREE gfx1250 VMFB corpus through "
            "gfx1250 -> RDNA4 DBT."
        )
    )
    parser.add_argument(
        "--validation-dir",
        required=True,
        type=Path,
        help="iree-gfx1250-validation directory with cases/ and extra/",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for translated RDNA4 VMFB outputs",
    )
    parser.add_argument(
        "--translator",
        required=True,
        type=Path,
        help="rocjitsu-translate-vmfb binary",
    )
    parser.add_argument(
        "--iree-run-module",
        required=True,
        type=Path,
        help="IREE iree-run-module binary",
    )
    parser.add_argument(
        "--iree-dump-module",
        required=True,
        type=Path,
        help="IREE iree-dump-module binary",
    )
    parser.add_argument(
        "--iree-matmul-test",
        required=True,
        type=Path,
        help="IREE tools/testing/e2e/iree-e2e-matmul-test binary",
    )
    parser.add_argument(
        "--device",
        default="hip",
        help="IREE HAL device used for runtime checks",
    )
    parser.add_argument(
        "--guest-arch",
        default="gfx1250",
        help="Guest ISA architecture passed to rocjitsu-translate-vmfb",
    )
    parser.add_argument(
        "--host-arch",
        default="rdna4",
        help="Host ISA architecture passed to rocjitsu-translate-vmfb",
    )
    parser.add_argument(
        "--host-machine",
        default="gfx1201",
        help="Host machine passed to rocjitsu-translate-vmfb",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help=(
            "Run only the named case. Matmul cases use the cases/ directory "
            "name; extra cases use extra/<name>. May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-command timeout in seconds",
    )
    return parser.parse_args()


def run_command(cmd, timeout):
    try:
        return subprocess.run(
            [str(part) for part in cmd],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return e


def summarize_output(stdout, stderr, max_lines=20):
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    combined = "\n".join(part for part in (stdout, stderr) if part)
    lines = [line.rstrip() for line in combined.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:]) if lines else ""


def fail_from_process(name, action, proc):
    if isinstance(proc, subprocess.TimeoutExpired):
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        tail = summarize_output(stdout, stderr)
        detail = f"{action} timed out"
        if tail:
            detail += f"\n{tail}"
        return CaseResult(name, "FAIL", detail)
    tail = summarize_output(proc.stdout, proc.stderr)
    detail = f"{action} exited {proc.returncode}"
    if tail:
        detail += f"\n{tail}"
    return CaseResult(name, "FAIL", detail)


def translate_vmfb(args, name, input_vmfb, output_vmfb):
    output_vmfb.parent.mkdir(parents=True, exist_ok=True)
    proc = run_command(
        [
            args.translator,
            input_vmfb,
            output_vmfb,
            args.guest_arch,
            args.host_arch,
            args.host_machine,
            "--fail-on-warnings",
        ],
        args.timeout,
    )
    if isinstance(proc, subprocess.TimeoutExpired) or proc.returncode != 0:
        return output_vmfb, fail_from_process(name, "translate", proc)
    return output_vmfb, None


def parse_exported_functions(args, vmfb):
    proc = run_command(
        [args.iree_dump_module, "--output=metadata", vmfb],
        args.timeout,
    )
    if isinstance(proc, subprocess.TimeoutExpired) or proc.returncode != 0:
        raise RuntimeError(f"iree-dump-module failed for {vmfb}")

    exports = []
    in_exports = False
    for line in proc.stdout.splitlines():
        if line.startswith("Exported Functions:"):
            in_exports = True
            continue
        if in_exports and line.startswith("//==="):
            break
        if not in_exports:
            continue
        match = EXPORT_RE.match(line)
        if match:
            name = match.group(1).strip()
            if not name.startswith("__"):
                exports.append(name)
    return exports


def matmul_uses_inexact_fp(case_name):
    return "_f16" in case_name or "_f8" in case_name


def discover_matmul_cases(validation_dir):
    cases_dir = validation_dir / "cases"
    for case_dir in sorted(cases_dir.glob("*")):
        if not case_dir.is_dir():
            continue
        matmul_vmfbs = sorted(case_dir.glob("*_matmul.vmfb"))
        calls_vmfbs = sorted(case_dir.glob("*_calls.vmfb"))
        if len(matmul_vmfbs) != 1 or len(calls_vmfbs) != 1:
            raise RuntimeError(f"expected one matmul/calls pair in {case_dir}")
        yield case_dir.name, matmul_vmfbs[0], calls_vmfbs[0]


def discover_extra_cases(validation_dir):
    extra_dir = validation_dir / "extra"
    for case_dir in sorted(extra_dir.glob("*")):
        if not case_dir.is_dir():
            continue
        vmfbs = sorted(case_dir.glob("*.vmfb"))
        if len(vmfbs) != 1:
            raise RuntimeError(f"expected one VMFB in {case_dir}")
        yield f"extra/{case_dir.name}", vmfbs[0]


def select_cases(discovered, selected):
    selected_set = set(selected or [])
    return [case for case in discovered if not selected_set or case[0] in selected_set]


def check_selected_cases(discovered, selected):
    selected_set = set(selected or [])
    if not selected_set:
        return
    found = {case[0] for case in discovered}
    missing = selected_set - found
    if missing:
        raise RuntimeError("unknown case(s): " + ", ".join(sorted(missing)))


def run_matmul_case(args, case_name, matmul_vmfb, calls_vmfb):
    output_vmfb = args.output_dir / "cases" / case_name / f"{case_name}_matmul.rdna4.vmfb"
    translated_vmfb, failure = translate_vmfb(args, case_name, matmul_vmfb, output_vmfb)
    if failure:
        return failure

    cmd = [
        args.iree_matmul_test,
        f"--device={args.device}",
        f"--module={translated_vmfb}",
        f"--module={calls_vmfb}",
    ]
    if matmul_uses_inexact_fp(case_name):
        cmd.extend(["--require_exact_results=false", "--acceptable_fp_delta=1e-04"])

    proc = run_command(cmd, args.timeout)
    if isinstance(proc, subprocess.TimeoutExpired) or proc.returncode != 0:
        return fail_from_process(case_name, "matmul runtime", proc)

    check_count = proc.stdout.count("--- TEST[") + proc.stderr.count("--- TEST[")
    return CaseResult(case_name, "PASS", f"{check_count} generated matmul checks", check_count)


def run_extra_case(args, case_name, vmfb):
    output_name = case_name.split("/", 1)[1]
    output_vmfb = args.output_dir / "extra" / output_name / f"{output_name}.rdna4.vmfb"
    translated_vmfb, failure = translate_vmfb(args, case_name, vmfb, output_vmfb)
    if failure:
        return failure

    try:
        exports = parse_exported_functions(args, translated_vmfb)
    except RuntimeError as e:
        return CaseResult(case_name, "FAIL", str(e))
    if not exports:
        return CaseResult(case_name, "FAIL", "no exported functions")

    for function in exports:
        proc = run_command(
            [
                args.iree_run_module,
                f"--device={args.device}",
                f"--module={translated_vmfb}",
                f"--function={function}",
            ],
            args.timeout,
        )
        if isinstance(proc, subprocess.TimeoutExpired) or proc.returncode != 0:
            result = fail_from_process(case_name, f"runtime {function}", proc)
            result.checks = 0
            return result

    return CaseResult(case_name, "PASS", f"{len(exports)} exported function checks", len(exports))


def validate_paths(args):
    paths = [
        args.validation_dir,
        args.translator,
        args.iree_run_module,
        args.iree_dump_module,
        args.iree_matmul_test,
    ]
    for path in paths:
        if not path.exists():
            raise RuntimeError(f"required path not found: {path}")


def main():
    args = parse_args()
    validate_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    discovered_matmul_cases = list(discover_matmul_cases(args.validation_dir))
    discovered_extra_cases = list(discover_extra_cases(args.validation_dir))
    check_selected_cases(
        discovered_matmul_cases + discovered_extra_cases,
        args.cases,
    )
    matmul_cases = select_cases(discovered_matmul_cases, args.cases)
    extra_cases = select_cases(discovered_extra_cases, args.cases)
    cases_found = len(matmul_cases) + len(extra_cases)
    if not cases_found:
        raise RuntimeError("no cases selected")

    results = []
    for case_name, matmul_vmfb, calls_vmfb in matmul_cases:
        result = run_matmul_case(args, case_name, matmul_vmfb, calls_vmfb)
        results.append(result)
        print(f"{result.name}: {result.status} {result.detail}", flush=True)

    for case_name, vmfb in extra_cases:
        result = run_extra_case(args, case_name, vmfb)
        results.append(result)
        print(f"{result.name}: {result.status} {result.detail}", flush=True)

    pass_count = sum(1 for result in results if result.status == "PASS")
    fail_count = sum(1 for result in results if result.status == "FAIL")
    check_count = sum(result.checks for result in results)
    print(
        "SUMMARY "
        f"pass={pass_count} fail={fail_count} total={len(results)} checks={check_count}"
    )
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
