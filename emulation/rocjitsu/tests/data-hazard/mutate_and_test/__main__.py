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
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import traceback
from typing import Dict, List, Optional, Tuple

from .benchmark import Benchmarker, PerfCollector
from .kernel_processor import KernelBuilder, KernelBuilderConfig
from .models import MutantResult, ShaderReport, WaitInstruction
from .report_writer import (
    compute_summary,
    write_benchmark_report,
    write_csv_report,
    write_json_report,
    write_markdown_report,
    write_terminal_report,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ARCH = os.environ.get("TARGET_ARCH", "gfx1250")
DEFAULT_TIMEOUT = 240  # 4 minutes

WAIT_RE = re.compile(r"^\s+(s_wait\w+)\s+", re.MULTILINE)

# s_wait_xcnt tracks address translation (XNACK replay), not data completion.
# Mutating it does not produce data hazards detectable by this plugin, so it
# is excluded by default.  Use --include-xcnt to re-enable for future work.
DEFAULT_EXCLUDED_WAITS = frozenset({"s_wait_xcnt"})


def _rocjitsu_root() -> Path:
    """Return the rocjitsu source root (``tests/data-hazard/mutate_and_test`` → root)."""
    return Path(__file__).resolve().parent.parent.parent.parent


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------


def _detect_rocm_path() -> str:
    """Try to auto-detect ROCM_PATH from environment or rocm-sdk."""
    for var in ("ROCM_PATH", "ROCM_HOME"):
        val = os.environ.get(var)
        if val and os.path.isdir(val):
            return val
    try:
        r = subprocess.run(
            ["rocm-sdk", "path", "--root"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "/opt/rocm"


def _select_rocm_path(requested_rocm_path: Optional[str]) -> str:
    """Select a ROCm install. The launcher sets up the simulated runtime itself."""
    return requested_rocm_path or _detect_rocm_path()


def find_tool(name: str, rocm_path: str) -> str:
    """Locate a ROCm / LLVM tool binary."""
    candidates = [
        os.path.join(rocm_path, "bin", name),
        os.path.join(rocm_path, "llvm", "bin", name),
    ]
    which = shutil.which(name)
    if which:
        candidates.append(which)
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise FileNotFoundError(
        f"Cannot find '{name}'. Searched: {candidates}. "
        f"Set ROCM_PATH or ensure {name} is in PATH."
    )


# ---------------------------------------------------------------------------
# Assembly analysis / mutation
# ---------------------------------------------------------------------------


def find_wait_instructions(
    asm_path: Path,
    excluded_waits: Optional[frozenset] = None,
) -> List[WaitInstruction]:
    """Parse an assembly file and return all s_wait_* instructions.

    Args:
        excluded_waits: Set of instruction mnemonics to skip (e.g. ``{"s_wait_xcnt"}``).
                        Pass ``None`` or empty set to include everything.
    """
    exclude = excluded_waits or frozenset()
    waits: List[WaitInstruction] = []
    with open(asm_path) as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if stripped.startswith(";") or stripped.startswith("."):
                continue
            m = re.match(r"(s_wait\w+)\s+(\S+)", stripped)
            if m:
                mnemonic = m.group(1)
                if mnemonic in exclude:
                    continue
                waits.append(
                    WaitInstruction(
                        line_number=idx,
                        instruction=mnemonic,
                        operand=m.group(2),
                        full_line=line.rstrip(),
                    )
                )
    return waits


def create_mutant_asm(asm_path: Path, wait: WaitInstruction, output: Path) -> None:
    """
    Create a mutant assembly file with one s_wait replaced by s_nop.
    The original instruction is preserved as a comment for traceability.
    """
    with open(asm_path) as f:
        lines = f.readlines()

    original = lines[wait.line_number].rstrip()
    lines[wait.line_number] = f"\ts_nop 0  ; MUTANT: removed {original.strip()}\n"

    with open(output, "w") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_command(
    argv: List[str], timeout: int, env: Optional[Dict[str, str]] = None
) -> Tuple[int, str, str]:
    """Run a command; returns (exit_code, stdout, stderr)."""
    try:
        run_env = None
        if env:
            run_env = {**os.environ, **env}
        r = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except OSError as e:
        return -2, "", str(e)


def perf_record_command(
    argv: List[str],
    timeout: int,
    perf_collector: PerfCollector,
    label: str,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    """Run *argv* under ``perf record``; returns (exit_code, stdout, stderr)."""
    perf_args, _ = perf_collector.record_args(label)
    return run_command(perf_args + argv, timeout, env=env)


def _run_command_for(
    runner: Optional["RocjitsuRunner"], exe: Path, report_path: Path
) -> Tuple[List[str], Dict[str, str]]:
    """Argv and env for one kernel run, with or without hazard detection."""
    if runner is None:
        return [str(exe)], {}
    return runner.command(exe, report_path)


def read_hazard_report(path: Path) -> Tuple[int, str]:
    """Read a data_hazard plugin JSON report. Returns (count, raw_content)."""
    if not path.is_file():
        return 0, ""
    try:
        raw = path.read_text()
        hazards = json.loads(raw)
        if isinstance(hazards, list):
            return len(hazards), raw
        return 0, raw
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return 0, ""


# ---------------------------------------------------------------------------
# Per-shader processing
# ---------------------------------------------------------------------------


def _compile_and_analyse(
    report: ShaderReport,
    kernel_builder: KernelBuilder,
    extra_cxxflags: Optional[List[str]] = None,
    verbose: bool = False,
    excluded_waits: Optional[frozenset] = None,
) -> Optional[Path]:
    """Compile shader to assembly, read it, and find wait instructions.

    Populates *report* fields and returns the assembly path on success,
    or ``None`` if compilation failed.
    """
    config = kernel_builder.config
    shader_cpp = config.shader_cpp
    asm_file = config.asm_file
    ok, err = kernel_builder.compile_to_assembly(
        extra_cxxflags=extra_cxxflags,
    )
    if not ok:
        report.error = err
        print(f"  [SKIP] Cannot compile {shader_cpp.name}: {err}", file=sys.stderr)
        return None
    report.compile_ok = True
    report.asm_file = str(asm_file)

    try:
        report.asm_content = asm_file.read_text()
    except OSError:
        report.asm_content = ""

    waits = find_wait_instructions(asm_file, excluded_waits=excluded_waits)
    report.waits_found = waits
    if verbose:
        print(f"  Found {len(waits)} s_wait instructions")

    return asm_file


def _run_baseline(
    report: ShaderReport,
    kernel_builder: KernelBuilder,
    runner: Optional["RocjitsuRunner"] = None,
    verbose: bool = False,
    timeout: int = 10,
    subprocess_output: bool = False,
    benchmarker: Optional[Benchmarker] = None,
    benchmark_enabled: bool = False,
    benchmark_label: str = "",
    perf_collector: Optional[PerfCollector] = None,
) -> bool:
    """Build and run the unmodified (baseline) shader.

    Returns ``True`` if the baseline built and ran successfully enough
    to proceed with mutation testing.
    """
    config = kernel_builder.config
    baseline_exe, err = kernel_builder.build_from_asm()

    hazard_dir = config.baseline_hazard_report_path.parent
    if not hazard_dir.exists():
        hazard_dir.mkdir(parents=True, exist_ok=True)

    if baseline_exe is None:
        report.error = err
        print(f"  [SKIP] Baseline build failed: {err}", file=sys.stderr)
        return False
    report.build_ok = True

    hazard_report_path = config.baseline_hazard_report_path
    hazard_report_path.unlink(missing_ok=True)
    argv, baseline_env = _run_command_for(runner, baseline_exe, hazard_report_path)

    _bm = benchmarker or Benchmarker()
    with _bm.benchmark(benchmark_label, benchmark_enabled=benchmark_enabled):
        if perf_collector is not None:
            ec, stdout, stderr = perf_record_command(
                argv, timeout, perf_collector, benchmark_label, env=baseline_env
            )
        else:
            ec, stdout, stderr = run_command(argv, timeout, env=baseline_env)

    report.baseline_ok = ec == 0
    report.baseline_stdout = stdout
    report.baseline_exit_code = ec

    if subprocess_output or (not report.baseline_ok):
        label = "baseline"
        if stdout:
            print(f"  [{label} stdout]\n{stdout}", flush=True)
        if stderr:
            print(f"  [{label} stderr]\n{stderr}", flush=True)

    count, raw = read_hazard_report(hazard_report_path)
    report.baseline_hazard_count = count
    report.baseline_hazard_report_raw = raw
    report.baseline_hazard_report_path = str(hazard_report_path)
    if verbose:
        print(f"  Baseline hazards detected: {count}")

    if verbose:
        print(f"  Baseline: exit={ec}  {'PASS' if ec == 0 else 'FAIL'}")

    return True


def _run_single_mutant(
    kernel_builder: KernelBuilder,
    i: int,
    wait: WaitInstruction,
    baseline_stdout: str,
    runner: Optional["RocjitsuRunner"] = None,
    verbose: bool = False,
    timeout: int = 10,
    subprocess_output: bool = False,
    benchmarker: Optional[Benchmarker] = None,
    benchmark_enabled: bool = False,
    benchmark_label: str = "",
    perf_collector: Optional[PerfCollector] = None,
) -> MutantResult:
    """Build, run, and evaluate a single wait-removal mutant."""
    tag = f"mut{i}"
    config = kernel_builder.config
    shader_cpp = config.shader_cpp
    mr = MutantResult(
        shader=shader_cpp.stem,
        wait_index=i,
        wait_instruction=f"{wait.instruction} {wait.operand}",
        wait_line=wait.line_number,
    )

    mut_asm = config.workdir / f"{shader_cpp.stem}_{tag}.s"
    create_mutant_asm(config.asm_file, wait, mut_asm)
    # Point the builder at the mutated assembly for this build
    original_asm = config.asm_file
    config.asm_file = mut_asm
    try:
        exe, err = kernel_builder.build_from_asm()
    finally:
        config.asm_file = original_asm
    if exe is None:
        mr.error = err
        if verbose:
            print(
                f"  Mutant {i} ({wait.instruction} L{wait.line_number}): "
                f"BUILD FAIL — {err}"
            )
        return mr
    mr.compiled = True
    mr.linked = True

    mutant_hazard_report_path = config.mutant_hazard_report_path(tag=tag)
    mutant_hazard_report_path.unlink(missing_ok=True)
    argv, mutant_env = _run_command_for(runner, exe, mutant_hazard_report_path)

    _bm = benchmarker or Benchmarker()
    with _bm.benchmark(benchmark_label, benchmark_enabled=benchmark_enabled):
        if perf_collector is not None:
            ec, stdout, stderr = perf_record_command(
                argv, timeout, perf_collector, benchmark_label, env=mutant_env
            )
        else:
            ec, stdout, stderr = run_command(argv, timeout, env=mutant_env)

    mr.exit_code = ec
    mr.stdout = stdout
    mr.stderr = stderr

    if subprocess_output or ec not in (0, -1):
        label = f"mut{i} ({wait.instruction} L{wait.line_number})"
        if stdout:
            print(f"  [{label} stdout]\n{stdout}", flush=True)
        if stderr:
            print(f"  [{label} stderr]\n{stderr}", flush=True)

    if ec == -1:
        mr.error = "TIMEOUT"
    elif ec < 0:
        mr.error = stderr
    else:
        mr.ran = True
        mr.stdout_match = stdout == baseline_stdout
        mr.correct = ec == 0 and mr.stdout_match

    status = (
        "MATCH"
        if mr.correct
        else (
            "TIMEOUT"
            if ec == -1
            else (
                "WRONG"
                if mr.ran and not mr.stdout_match
                else "CRASH" if mr.ran and ec != 0 else "FAIL"
            )
        )
    )

    count, raw = read_hazard_report(mutant_hazard_report_path)
    mr.hazard_count = count
    mr.hazard_report_raw = raw
    mr.hazard_report_path = str(mutant_hazard_report_path)

    if verbose:
        print(
            f"  Mutant {i} ({wait.instruction} L{wait.line_number}): "
            f"{status}  exit={ec}  hazards={mr.hazard_count}"
        )

    return mr


def process_shader(
    kernel_builder: KernelBuilder,
    extra_cxxflags: Optional[List[str]] = None,
    verbose: bool = False,
    runner: Optional["RocjitsuRunner"] = None,
    timeout: int = 10,
    excluded_waits: Optional[frozenset] = None,
    subprocess_output: bool = False,
    benchmarker: Optional[Benchmarker] = None,
    benchmark_enabled: bool = False,
    perf_collector: Optional[PerfCollector] = None,
) -> ShaderReport:
    """Run the full mutation pipeline for one shader."""
    config = kernel_builder.config
    report = ShaderReport(shader=config.shader_cpp.stem, asm_file="")

    asm_file = _compile_and_analyse(
        report,
        kernel_builder,
        extra_cxxflags=extra_cxxflags,
        verbose=verbose,
        excluded_waits=excluded_waits,
    )
    if asm_file is None:
        return report

    if not _run_baseline(
        report,
        kernel_builder,
        runner=runner,
        verbose=verbose,
        timeout=timeout,
        subprocess_output=subprocess_output,
        benchmarker=benchmarker,
        benchmark_enabled=benchmark_enabled,
        benchmark_label=report.shader,
        perf_collector=perf_collector,
    ):
        return report

    for i, wait in enumerate(report.waits_found):
        mr = _run_single_mutant(
            kernel_builder,
            i,
            wait,
            report.baseline_stdout,
            runner=runner,
            verbose=verbose,
            timeout=timeout,
            subprocess_output=subprocess_output,
            benchmarker=benchmarker,
            benchmark_enabled=benchmark_enabled,
            benchmark_label=f"{report.shader}_mut{i}",
            perf_collector=perf_collector,
        )
        report.mutants.append(mr)

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def discover_shaders(shader_dir: Path) -> List[Path]:
    """Find all .hip shader files in a directory."""
    excluded = {"make_test_shaders.py"}
    shaders = sorted(p for p in shader_dir.glob("*.hip") if p.name not in excluded)
    return shaders


class RocjitsuRunner:
    """
    Runs kernels under rocJitsu with the ``data_hazard`` plugin enabled.

    Plugin selection and the report destination are part of the simulator
    configuration rather than the environment, so a fresh config file is written
    for every run and handed to the launcher as
    ``rocjitsu --config <file> -- <exe>``.
    """

    def __init__(
        self,
        launcher: Path,
        base_config: Path,
        workdir: Path,
        *,
        verbose: bool = False,
    ) -> None:
        self.launcher = launcher.resolve()
        self.base_config = base_config.resolve()
        self.config_dir = workdir / "rocjitsu_configs"
        self.verbose = verbose

        if not self.launcher.is_file() or not os.access(self.launcher, os.X_OK):
            raise FileNotFoundError(f"rocjitsu launcher not found at {self.launcher}")
        if not self.base_config.is_file():
            raise FileNotFoundError(f"rocjitsu config not found at {self.base_config}")
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def command(self, exe: Path, report_path: Path) -> Tuple[List[str], Dict[str, str]]:
        """Build the launcher argv that runs *exe* reporting into *report_path*."""
        config_path = self.config_dir / f"{report_path.stem}.json"
        config = json.loads(self.base_config.read_text())
        plugins = config.setdefault("plugins", {})
        plugins.setdefault("data_hazard", {})["report_path"] = str(report_path)
        config_path.write_text(json.dumps(config, indent=2) + "\n")

        argv = [str(self.launcher), "--config", str(config_path), "--", str(exe)]
        if self.verbose:
            print(f"  rocjitsu: {' '.join(argv)}")
        return argv, {}


def gfx_target_version(arch: str) -> Optional[int]:
    """Encode a gfx target name the way the configs do: ``gfx1250`` -> 120500."""
    m = re.fullmatch(r"gfx(\d+)([0-9a-f])([0-9a-f])", arch)
    if not m:
        return None
    return int(m.group(1)) * 10000 + int(m.group(2), 16) * 100 + int(m.group(3), 16)


def default_config_for_arch(arch: str) -> Optional[Path]:
    """Pick the plain (non-kmd, non-guest) config that simulates *arch*."""
    config_dir = _rocjitsu_root() / "configs"
    wanted = gfx_target_version(arch)
    if wanted is None:
        return None
    candidates = [
        p
        for p in sorted(config_dir.glob("*.json"))
        if "kmd" not in p.stem and not p.stem.startswith("guest_")
    ]
    for path in candidates:
        try:
            device = json.loads(path.read_text())["vm"]["gpu"]["device"]
        except (OSError, ValueError, KeyError):
            continue
        if device.get("gfx_target_version") == wanted:
            return path
    return None


def build_rocjitsu_runner(
    workdir: Path,
    *,
    arch: str,
    launcher: Optional[Path] = None,
    config: Optional[Path] = None,
    build_dir: Optional[Path] = None,
    verbose: bool = False,
) -> RocjitsuRunner:
    """
    Construct a :class:`RocjitsuRunner` from explicit paths or the environment.

    Resolution order for each path is the explicit argument, then the matching
    environment variable (``ROCJITSU_LAUNCHER`` / ``ROCJITSU_CONFIG`` /
    ``ROCJITSU_BUILD_DIR``), then the default location inside the build tree.
    The config must simulate the same target the kernels are compiled for: a
    kernel that the simulated device cannot run reports no hazards at all,
    which otherwise looks like a clean run rather than a misconfiguration.
    """
    build = (
        build_dir
        or Path(os.environ.get("ROCJITSU_BUILD_DIR", _rocjitsu_root() / "build"))
    ).resolve()

    resolved_launcher = launcher or Path(
        os.environ.get(
            "ROCJITSU_LAUNCHER", str(build / "tools" / "rocjitsu" / "rocjitsu")
        )
    )
    env_config = os.environ.get("ROCJITSU_CONFIG")
    resolved_config = config or (
        Path(env_config) if env_config else default_config_for_arch(arch)
    )
    if resolved_config is None:
        raise FileNotFoundError(
            f"no config in {_rocjitsu_root() / 'configs'} simulates {arch}; "
            "pass --rocjitsu-config"
        )

    wanted = gfx_target_version(arch)
    try:
        device = json.loads(Path(resolved_config).read_text())["vm"]["gpu"]["device"]
        found = device.get("gfx_target_version")
    except (OSError, ValueError, KeyError):
        found = None
    if wanted is not None and found is not None and found != wanted:
        raise FileNotFoundError(
            f"config {resolved_config} simulates gfx_target_version {found}, but kernels "
            f"are compiled for {arch} ({wanted}); the mismatch reports zero hazards"
        )

    return RocjitsuRunner(resolved_launcher, resolved_config, workdir, verbose=verbose)


def main() -> int:
    parser = argparse.ArgumentParser(
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

        print(f"Shaders: {[s.name for s in shaders]}")
        print(f"Architecture: {args.arch}")

        # --- find tools ---
        rocm_path = _select_rocm_path(args.rocm_path)
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
