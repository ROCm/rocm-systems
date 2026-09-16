# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Mutation pipeline: compile a shader, remove one ``s_wait_*``, rebuild, run.

Kept apart from the command line entry point in ``__main__`` so that importing
the package does not load the executable module. Drivers outside this tree (the
FFM harness runs the same shader corpus against a different simulator) build
their own campaign loop out of these pieces.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

from .benchmark import Benchmarker, PerfCollector
from .kernel_processor import KernelBuilder
from .models import MemoryMutation, MutantResult, ShaderReport, WaitInstruction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ARCH = os.environ.get("TARGET_ARCH", "gfx1250")
DEFAULT_TIMEOUT = 240  # 4 minutes

# Shaders using instructions that only assemble for some targets declare them
# with a `// requires: <arch>` comment; they are skipped on every other
# architecture.
REQUIRES_RE = re.compile(r"^\s*//\s*requires:\s*(.+)$", re.MULTILINE)

# A shader picks which mutation paths run against it with a `// mutate: <kinds>`
# comment, where kinds is a comma-separated subset of {wait, memory}. The
# annotation is exclusive: it fully selects the paths, so a memory-mutation
# kernel writes `// mutate: memory` and no waits are stripped from it. A shader
# with no such comment defaults to wait-stripping only, which is how the whole
# existing corpus behaves.
MUTATE_RE = re.compile(r"^\s*//\s*mutate:\s*(.+)$", re.MULTILINE)

MUTATION_KIND_WAIT = "wait"
MUTATION_KIND_MEMORY = "memory"
VALID_MUTATION_KINDS = frozenset({MUTATION_KIND_WAIT, MUTATION_KIND_MEMORY})
DEFAULT_MUTATION_KINDS = frozenset({MUTATION_KIND_WAIT})

# s_wait_xcnt tracks address translation (XNACK replay), not data completion.
# Mutating it does not produce data hazards detectable by this plugin, so it
# is excluded by default.  Use --include-xcnt to re-enable for future work.
DEFAULT_EXCLUDED_WAITS = frozenset({"s_wait_xcnt"})


class KernelRunner(Protocol):
    """
    How one kernel run is started.

    The pipeline itself never launches a simulator: it asks the runner for the
    argv and environment that run *exe* with its hazard report going to
    *report_path*. :class:`~mutate_and_test.rocjitsu_runner.RocjitsuRunner`
    implements this for rocJitsu.
    """

    def command(
        self, exe: Path, report_path: Path
    ) -> Tuple[List[str], Dict[str, str]]: ...


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


def select_rocm_path(requested_rocm_path: Optional[str] = None) -> str:
    """
    ROCm install providing hipcc and the LLVM tools shaders are built with.

    Without an explicit request the install is detected from ``ROCM_PATH`` /
    ``ROCM_HOME``, then ``rocm-sdk``, then ``/opt/rocm``. Only the build side
    needs it; the launcher sets up the simulated runtime itself.
    """
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
# Memory mutation: sub-dword load -> store
# ---------------------------------------------------------------------------

# Sub-dword (byte / short) load mnemonics mapped to the store that writes the
# same operand width. A byte read turned into a byte write to the same address
# is the single edit that makes an all-reads kernel racy: the new writer must
# race the byte's foreign readers. Full-dword loads are deliberately absent —
# they cannot expose the sub-dword eviction gap. Legacy and _b8/_b16 spellings
# both appear because the compiler emits either depending on the pointer type;
# the store spelling is matched to the load's family (flat vs global) and era
# (legacy byte/short vs bN). d16 half-register loads are omitted: their store
# side writes only part of the data register and needs different operands.
SUBDWORD_LOAD_TO_STORE = {
    # byte, legacy spelling
    "flat_load_ubyte": "flat_store_byte",
    "flat_load_sbyte": "flat_store_byte",
    "global_load_ubyte": "global_store_byte",
    "global_load_sbyte": "global_store_byte",
    # byte, bN spelling
    "flat_load_u8": "flat_store_b8",
    "flat_load_i8": "flat_store_b8",
    "global_load_u8": "global_store_b8",
    "global_load_i8": "global_store_b8",
    # short, legacy spelling
    "flat_load_ushort": "flat_store_short",
    "flat_load_sshort": "flat_store_short",
    "global_load_ushort": "global_store_short",
    "global_load_sshort": "global_store_short",
    # short, bN spelling
    "flat_load_u16": "flat_store_b16",
    "flat_load_i16": "flat_store_b16",
    "global_load_u16": "global_store_b16",
    "global_load_i16": "global_store_b16",
}

_SUBDWORD_LOAD_RE = re.compile(
    r"^\s*(" + "|".join(re.escape(m) for m in SUBDWORD_LOAD_TO_STORE) + r")\b\s+(.*)$"
)


def _rewrite_load_to_store(store_mnemonic: str, operands: str) -> str:
    """Rewrite a sub-dword load's operands for its store counterpart.

    A load reads into its first operand from the address in the following
    operand(s); the store writes the same register to the same address, so the
    first two operands swap and everything after (a scalar base, ``offset:``,
    cache flags) is preserved verbatim:

        flat_load_ubyte  v0, v[0:1] offset:1 sc0 sc1
            -> flat_store_byte v[0:1], v0 offset:1 sc0 sc1
        global_load_ubyte v0, v2, s[0:1] offset:1
            -> global_store_byte v2, v0, s[0:1] offset:1

    The load's destination register is reused as the store's data source; its
    contents do not matter to the shadow, which records only that the byte was
    written.
    """
    parts = [p.strip() for p in operands.split(",")]
    # Trailing modifiers (offset:, sc0, nt, ...) hang off the last operand with
    # no comma, so split them away from that register token.
    last_tokens = parts[-1].split()
    regs = parts[:-1] + [last_tokens[0]]
    modifiers = " ".join(last_tokens[1:])

    vdst, vaddr, *rest = regs
    new_operands = ", ".join([vaddr, vdst, *rest])
    if modifiers:
        new_operands += " " + modifiers
    return f"{store_mnemonic} {new_operands}"


def find_sub_dword_loads(asm_path: Path) -> List[MemoryMutation]:
    """Parse an assembly file and return every sub-dword load it can flip to a store."""
    mutations: List[MemoryMutation] = []
    with open(asm_path) as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if stripped.startswith(";") or stripped.startswith("."):
                continue
            m = _SUBDWORD_LOAD_RE.match(stripped)
            if not m:
                continue
            load_mnemonic = m.group(1)
            store_mnemonic = SUBDWORD_LOAD_TO_STORE[load_mnemonic]
            rewritten = _rewrite_load_to_store(store_mnemonic, m.group(2))
            mutations.append(
                MemoryMutation(
                    line_number=idx,
                    load_mnemonic=load_mnemonic,
                    store_mnemonic=store_mnemonic,
                    full_line=line.rstrip(),
                    rewritten_line=rewritten,
                )
            )
    return mutations


def create_memory_mutant_asm(asm_path: Path, mem: MemoryMutation, output: Path) -> None:
    """
    Create a mutant assembly file with one sub-dword load rewritten as a store.
    The original instruction is preserved as a comment for traceability.
    """
    with open(asm_path) as f:
        lines = f.readlines()

    # Preserve the original line's leading indentation.
    original = lines[mem.line_number]
    indent = original[: len(original) - len(original.lstrip())]
    lines[mem.line_number] = (
        f"{indent}{mem.rewritten_line}  ; MUTANT: load->store {mem.full_line.strip()}\n"
    )

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
    runner: Optional[KernelRunner], exe: Path, report_path: Path
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
    runner: Optional[KernelRunner] = None,
    verbose: bool = False,
    timeout: int = 10,
    subprocess_output: bool = False,
    benchmarker: Optional[Benchmarker] = None,
    benchmark_enabled: bool = False,
    benchmark_label: str = "",
    perf_collector: Optional[PerfCollector] = None,
) -> bool:
    """Build and run the unmodified (baseline) shader.

    Returns ``True`` only when the baseline built and then ran to a clean exit,
    which is the reference every mutant is compared against.
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

    if not report.baseline_ok:
        # Every mutant is judged against the baseline's output and hazard count,
        # so a baseline that timed out or failed leaves nothing to judge them by.
        reason = (
            "TIMEOUT"
            if ec == -1
            else (stderr or "could not be started") if ec < 0 else f"exit {ec}"
        )
        report.error = f"baseline run failed: {reason}"
        print(f"  [SKIP] {report.error}", file=sys.stderr)
        return False

    return True


def _run_single_mutant(
    kernel_builder: KernelBuilder,
    i: int,
    wait: WaitInstruction,
    baseline_stdout: str,
    runner: Optional[KernelRunner] = None,
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
        kind="wait",
    )

    mut_asm = config.workdir / f"{shader_cpp.stem}_{tag}.s"
    create_mutant_asm(config.asm_file, wait, mut_asm)
    return _build_run_and_evaluate_mutant(
        kernel_builder,
        mr,
        i,
        tag,
        mut_asm,
        baseline_stdout,
        runner=runner,
        verbose=verbose,
        timeout=timeout,
        subprocess_output=subprocess_output,
        benchmarker=benchmarker,
        benchmark_enabled=benchmark_enabled,
        benchmark_label=benchmark_label,
        perf_collector=perf_collector,
    )


def _run_single_memory_mutant(
    kernel_builder: KernelBuilder,
    i: int,
    mem: MemoryMutation,
    baseline_stdout: str,
    runner: Optional[KernelRunner] = None,
    verbose: bool = False,
    timeout: int = 10,
    subprocess_output: bool = False,
    benchmarker: Optional[Benchmarker] = None,
    benchmark_enabled: bool = False,
    benchmark_label: str = "",
    perf_collector: Optional[PerfCollector] = None,
) -> MutantResult:
    """Build, run, and evaluate a single load->store memory mutant."""
    tag = f"mut{i}"
    config = kernel_builder.config
    shader_cpp = config.shader_cpp
    mr = MutantResult(
        shader=shader_cpp.stem,
        wait_index=i,
        wait_instruction=f"{mem.load_mnemonic}→{mem.store_mnemonic}",
        wait_line=mem.line_number,
        kind="memory",
    )

    mut_asm = config.workdir / f"{shader_cpp.stem}_{tag}.s"
    create_memory_mutant_asm(config.asm_file, mem, mut_asm)
    return _build_run_and_evaluate_mutant(
        kernel_builder,
        mr,
        i,
        tag,
        mut_asm,
        baseline_stdout,
        runner=runner,
        verbose=verbose,
        timeout=timeout,
        subprocess_output=subprocess_output,
        benchmarker=benchmarker,
        benchmark_enabled=benchmark_enabled,
        benchmark_label=benchmark_label,
        perf_collector=perf_collector,
    )


def _build_run_and_evaluate_mutant(
    kernel_builder: KernelBuilder,
    mr: MutantResult,
    i: int,
    tag: str,
    mut_asm: Path,
    baseline_stdout: str,
    runner: Optional[KernelRunner] = None,
    verbose: bool = False,
    timeout: int = 10,
    subprocess_output: bool = False,
    benchmarker: Optional[Benchmarker] = None,
    benchmark_enabled: bool = False,
    benchmark_label: str = "",
    perf_collector: Optional[PerfCollector] = None,
) -> MutantResult:
    """Build one already-written mutant assembly, run it, and score it.

    Shared by the wait-removal and load->store mutant paths, which differ only
    in how they produce *mut_asm*.
    """
    config = kernel_builder.config
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
                f"  Mutant {i} ({mr.wait_instruction} L{mr.wait_line}): "
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
        label = f"mut{i} ({mr.wait_instruction} L{mr.wait_line})"
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
            f"  Mutant {i} ({mr.wait_instruction} L{mr.wait_line}): "
            f"{status}  exit={ec}  hazards={mr.hazard_count}"
        )

    return mr


def process_shader(
    kernel_builder: KernelBuilder,
    extra_cxxflags: Optional[List[str]] = None,
    verbose: bool = False,
    runner: Optional[KernelRunner] = None,
    timeout: int = 10,
    excluded_waits: Optional[frozenset] = None,
    subprocess_output: bool = False,
    benchmarker: Optional[Benchmarker] = None,
    benchmark_enabled: bool = False,
    perf_collector: Optional[PerfCollector] = None,
    mutate_waits: bool = True,
    mutate_memory: bool = False,
) -> ShaderReport:
    """Run the full mutation pipeline for one shader.

    ``mutate_waits`` and ``mutate_memory`` select which mutation paths run;
    callers resolve them from the shader's ``// mutate:`` annotation (see
    :func:`shader_mutation_kinds`). The defaults reproduce the historical
    behavior — wait-stripping only — for drivers that do not pass them.
    """
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

    if mutate_waits:
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

    if mutate_memory:
        # Memory mutants continue the same index sequence so each row's # stays
        # unique across both mutation kinds.
        mem_mutations = find_sub_dword_loads(asm_file)
        if verbose:
            print(f"  Found {len(mem_mutations)} sub-dword loads to flip to stores")
        base = len(report.mutants)
        for j, mem in enumerate(mem_mutations):
            i = base + j
            mr = _run_single_memory_mutant(
                kernel_builder,
                i,
                mem,
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
# Shader discovery
# ---------------------------------------------------------------------------


def discover_shaders(shader_dir: Path) -> List[Path]:
    """Find all .hip shader files in a directory."""
    excluded = {"make_test_shaders.py"}
    shaders = sorted(p for p in shader_dir.glob("*.hip") if p.name not in excluded)
    return shaders


def shader_required_archs(shader: Path) -> List[str]:
    """
    Architectures a shader is restricted to, from a ``// requires: <arch>[, ...]`` line.

    An empty list means the shader is portable across every target.
    """
    try:
        text = shader.read_text(errors="replace")
    except OSError:
        return []
    archs: List[str] = []
    for match in REQUIRES_RE.finditer(text):
        archs.extend(a.strip() for a in match.group(1).split(",") if a.strip())
    return archs


def shader_supports_arch(shader: Path, arch: str) -> bool:
    """True when ``shader`` can be built for ``arch``."""
    required = shader_required_archs(shader)
    return not required or arch in required


def shader_mutation_kinds(shader: Path) -> "frozenset[str]":
    """
    Mutation paths a shader opts into, from its ``// mutate: <kinds>[, ...]`` line.

    The annotation is exclusive: whatever it lists is exactly what runs, drawn
    from ``{wait, memory}``. A shader with no annotation defaults to
    :data:`DEFAULT_MUTATION_KINDS` (wait-stripping only), preserving the
    behavior of the existing corpus. Unknown kinds are ignored with a warning
    so a typo never silently disables every path.
    """
    try:
        text = shader.read_text(errors="replace")
    except OSError:
        return DEFAULT_MUTATION_KINDS

    matches = list(MUTATE_RE.finditer(text))
    if not matches:
        return DEFAULT_MUTATION_KINDS

    kinds = set()
    for match in matches:
        for raw in match.group(1).split(","):
            kind = raw.strip().lower()
            if not kind:
                continue
            if kind in VALID_MUTATION_KINDS:
                kinds.add(kind)
            else:
                print(
                    f"  [WARN] {shader.name}: unknown mutate kind '{kind}' "
                    f"(known: {', '.join(sorted(VALID_MUTATION_KINDS))})",
                    file=sys.stderr,
                )
    # An annotation that named only unknown kinds still means the author chose
    # to override the default; fall back rather than run nothing silently.
    return frozenset(kinds) if kinds else DEFAULT_MUTATION_KINDS
