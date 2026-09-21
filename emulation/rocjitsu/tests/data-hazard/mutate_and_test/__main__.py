# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
"""
Hazard Mutation Testing for AMDGPU Shaders

Compiles HIP shaders to GCN assembly, systematically removes s_wait_*
instructions one at a time, rebuilds from modified assembly, runs each
mutant, and produces a matrix of which wait removals cause failures.

Build pipeline (follows ROCm assembly_to_executable example):
  1. hipcc -S --cuda-device-only  →  device assembly (.s)
  2. hipcc -c --cuda-host-only    →  host object (.o)
  3. clang -target amdgcn-amd-amdhsa -mcpu=<arch>  →  device object (.o)
  4. clang-offload-bundler  →  offload bundle (.hipfb)
  5. llvm-mc (embed via .incbin)  →  fatbin object (.o)
  6. hipcc (link host + fatbin)   →  executable

Reference: https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/assembly_to_executable

Only the command line lives here; the pipeline it drives is in
:mod:`mutate_and_test.pipeline`, so importing the package does not load this
module.
"""

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

from .benchmark import Benchmarker, PerfCollector
from .kernel_processor import KernelBuilder, KernelBuilderConfig
from .models import ShaderReport
from .pipeline import (
    DEFAULT_ARCH,
    DEFAULT_EXCLUDED_WAITS,
    DEFAULT_TIMEOUT,
    MUTATION_KIND_MEMORY,
    MUTATION_KIND_WAIT,
    discover_shaders,
    find_tool,
    process_shader,
    select_rocm_path,
    shader_mutation_kinds,
    shader_required_archs,
    shader_supports_arch,
)
from .report_writer import (
    compute_summary,
    write_benchmark_report,
    write_csv_report,
    write_json_report,
    write_markdown_report,
    write_terminal_report,
)
from .rocjitsu_runner import RocjitsuRunner, build_rocjitsu_runner


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mutate_and_test",
        description="Mutation testing: remove s_wait_* instructions from "
        "AMDGPU assembly and test for data hazards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "shaders",
        nargs="*",
        type=Path,
        help="HIP .hip shader files to test. If omitted, all .hip files in "
        "the script's directory are used.",
    )
    parser.add_argument(
        "--arch",
        default=DEFAULT_ARCH,
        help=f"GPU target architecture (default: {DEFAULT_ARCH})",
    )
    parser.add_argument(
        "--rocm-path",
        default=None,
        help="ROCm installation path (auto-detected from rocm-sdk or $ROCM_PATH)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for build artifacts and reports. "
        "If omitted, a temporary directory is used.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-execution timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Do not delete the build directory on exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed progress.",
    )
    parser.add_argument(
        "--hazard-detection",
        action="store_true",
        help="Run every kernel under rocJitsu with the data_hazard plugin "
        "enabled and collect its report. Without this flag the kernels "
        "run natively and no hazards are recorded.",
    )
    parser.add_argument(
        "--rocjitsu-launcher",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to the rocjitsu launcher binary. If omitted, uses "
        "$ROCJITSU_LAUNCHER, or <build-dir>/rocjitsu.",
    )
    parser.add_argument(
        "--rocjitsu-config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Simulator config to run with. If omitted, uses $ROCJITSU_CONFIG, "
        "or the config under configs/ that simulates --arch. Whichever is "
        "used has to match --arch.",
    )
    parser.add_argument(
        "--rocjitsu-build-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="rocJitsu build directory used to locate the launcher "
        "(default: $ROCJITSU_BUILD_DIR, or <rocjitsu>/build).",
    )
    parser.add_argument(
        "--target-features",
        default=None,
        help="Comma-separated AMDGPU target features to enable "
        "(e.g. wmma-256b-insts,wavefrontsize32).",
    )
    parser.add_argument(
        "--cxxflags",
        default=None,
        help="Extra compiler flags passed to hipcc during assembly compilation "
        "(e.g. -I/path/to/include). Space-separated.",
    )
    parser.add_argument(
        "--include-xcnt",
        action="store_true",
        help="Include s_wait_xcnt instructions as mutation targets. "
        "Excluded by default because XCNT tracks address translation "
        "(XNACK replay), not data completion, so removing it does not "
        "produce data hazards detectable by this plugin.",
    )
    parser.add_argument(
        "--mutate-memory",
        action="store_true",
        help="Force sub-dword load->store mutation on for every shader, "
        "regardless of its '// mutate:' annotation. Each byte/short load is "
        "flipped to a store to the same address (read->write), which turns an "
        "all-reads kernel racy and exercises the global shadow's sub-dword "
        "tracking. Without this flag, a shader runs whichever paths its "
        "'// mutate:' comment names (default: wait-stripping only). Most "
        "useful with --hazard-detection.",
    )
    parser.add_argument(
        "--subprocess-output",
        action="store_true",
        help="Print stdout and stderr from every subprocess invocation "
        "(kernel runs and build steps). Always printed on non-zero exit. "
        "Use this flag in CI to diagnose crashes such as exit code -11 (SIGSEGV).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Record and report wall-clock execution time for each kernel run.",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Profile each kernel run with perf record and generate a cumulative flamegraph SVG. "
        "Most useful with --hazard-detection to capture plugin call stacks. "
        "Requires perf and FlameGraph scripts.",
    )
    parser.add_argument(
        "--flamegraph-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory containing flamegraph.pl and stackcollapse-perf.pl "
        "(default: auto-discovered from $FLAMEGRAPH_DIR, /opt/flamegraph, /usr/share/flamegraph).",
    )
    args = parser.parse_args()

    try:
        # --- discover shaders ---
        if args.shaders:
            shaders = [Path(s).resolve() for s in args.shaders]
        else:
            shaders = discover_shaders(
                Path(__file__).resolve().parent.parent / "shaders"
            )

        if not shaders:
            print("No shader files found.", file=sys.stderr)
            return 1

        skipped = [s for s in shaders if not shader_supports_arch(s, args.arch)]
        shaders = [s for s in shaders if shader_supports_arch(s, args.arch)]
        if not shaders:
            print(f"No shader files target {args.arch}.", file=sys.stderr)
            return 1

        print(f"Shaders: {[s.name for s in shaders]}")
        if skipped:
            print(
                f"Skipped (not for {args.arch}): "
                + ", ".join(
                    f"{s.name} [{', '.join(shader_required_archs(s))}]" for s in skipped
                )
            )
        print(f"Architecture: {args.arch}")

        # --- find tools ---
        rocm_path = select_rocm_path(args.rocm_path)
        print(f"ROCm path: {rocm_path}")
        try:
            tools = {
                "hipcc": find_tool("hipcc", rocm_path),
                "clang": find_tool("clang", rocm_path),
                "bundler": find_tool("clang-offload-bundler", rocm_path),
                "llvm_mc": find_tool("llvm-mc", rocm_path),
                "llvm_nm": find_tool("llvm-nm", rocm_path),
            }
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        if args.verbose:
            for k, v in tools.items():
                print(f"  {k}: {v}")

        # --- set up workdir ---
        if args.output_dir:
            workdir = args.output_dir.resolve()
            workdir.mkdir(parents=True, exist_ok=True)
            tmp_dir = None
        else:
            tmp_dir = tempfile.mkdtemp(prefix="hazard_mutate_")
            workdir = Path(tmp_dir)
            print(f"Working directory: {workdir}")

        # --- set up hazard detection ---
        runner: Optional[RocjitsuRunner] = None
        if args.hazard_detection:
            try:
                runner = build_rocjitsu_runner(
                    workdir,
                    arch=args.arch,
                    launcher=args.rocjitsu_launcher,
                    config=args.rocjitsu_config,
                    build_dir=args.rocjitsu_build_dir,
                    verbose=args.verbose,
                )
            except FileNotFoundError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 1
            print(f"Hazard detection via {runner.launcher} ({runner.base_config.name})")

        # --- parse target features ---
        target_features = (
            args.target_features.split(",") if args.target_features else None
        )

        # --- parse extra cxxflags ---
        extra_cxxflags = args.cxxflags.split() if args.cxxflags else None

        # --- build wait exclusion set ---
        excluded_waits = None if args.include_xcnt else DEFAULT_EXCLUDED_WAITS

        # --- benchmarking ---
        benchmarker = Benchmarker()
        benchmark_enabled: bool = args.benchmark

        # --- perf / flamegraph ---
        perf_collector: Optional[PerfCollector] = None
        if args.perf:
            if runner is None:
                print(
                    "WARNING: --perf without --hazard-detection — flamegraph will "
                    "not capture data hazard plugin call stacks.",
                    file=sys.stderr,
                )
            perf_collector = PerfCollector.discover(
                workdir=workdir,
                flamegraph_dir=args.flamegraph_dir,
            )

        # --- process each shader ---
        reports: List[ShaderReport] = []
        for shader in shaders:
            print(f"\n{'─' * 60}")
            print(f"Processing: {shader.name}")
            print(f"{'─' * 60}")

            # A shader's '// mutate:' comment selects which paths run; the
            # --mutate-memory flag forces the memory path on everywhere.
            kinds = set(shader_mutation_kinds(shader))
            if args.mutate_memory:
                kinds.add(MUTATION_KIND_MEMORY)
            mutate_waits = MUTATION_KIND_WAIT in kinds
            mutate_memory = MUTATION_KIND_MEMORY in kinds
            print(f"Mutation paths: {', '.join(sorted(kinds)) or '(none)'}")

            config = KernelBuilderConfig(
                tools=tools,
                shader_cpp=shader,
                asm_file=workdir / f"{shader.stem}.s",
                arch=args.arch,
                workdir=workdir,
                tag=shader.name,
                rocm_path=rocm_path,
                target_features=target_features,
            )
            kernel_builder = KernelBuilder(config)

            rpt = process_shader(
                kernel_builder,
                extra_cxxflags=extra_cxxflags,
                verbose=args.verbose,
                runner=runner,
                timeout=args.timeout,
                excluded_waits=excluded_waits,
                subprocess_output=args.subprocess_output,
                benchmarker=benchmarker,
                benchmark_enabled=benchmark_enabled,
                perf_collector=perf_collector,
                mutate_waits=mutate_waits,
                mutate_memory=mutate_memory,
            )
            reports.append(rpt)

        # --- output ---
        write_terminal_report(reports, benchmarker=benchmarker)
        write_json_report(reports, workdir / "mutation_report.json", args.arch)
        write_csv_report(reports, workdir / "mutation_report.csv")
        write_markdown_report(reports, workdir / "mutation_report.md", args.arch)
        if benchmarker.enabled:
            write_benchmark_report(benchmarker, workdir / "benchmark_report.json")
        if perf_collector is not None:
            perf_collector.generate_flamegraph(workdir / "flamegraph.svg")

        # --- copy reports to cwd ---
        cwd = Path.cwd()
        if workdir != cwd:
            for name in (
                "mutation_report.json",
                "mutation_report.csv",
                "mutation_report.md",
                "benchmark_report.json",
                "flamegraph.svg",
            ):
                src = workdir / name
                if src.is_file():
                    shutil.copy2(str(src), str(cwd / name))
            if runner is not None:
                # Find `_reports` dirs in workdir and copy them to cwd
                for reports_dir in workdir.glob("*_reports"):
                    for file in reports_dir.glob("*.json"):
                        shutil.copy2(str(file), str(cwd / file.name))
            print(f"Reports copied to {cwd}")

        # --- cleanup ---
        if tmp_dir and not args.keep_artifacts:
            print(f"\nCleaning up {workdir} (use --keep-artifacts to keep)")
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif workdir:
            print(f"\nArtifacts preserved in {workdir}")

        summary = compute_summary(reports)
        if summary["total_mutants"] and not summary["scored_mutants"]:
            return 1

        return 0

    except Exception as e:
        print(f"ERROR: {traceback.format_exc()}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
