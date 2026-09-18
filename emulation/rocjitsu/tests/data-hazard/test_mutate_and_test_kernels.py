# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Pytest: one test per ``.hip`` kernel through the mutation pipeline.

Run from ``tests/data-hazard`` (so ``mutate_and_test`` imports)::

    cd tests/data-hazard && pytest test_mutate_and_test_kernels.py -v

Every kernel runs under rocJitsu with the ``data_hazard`` plugin enabled.
``ROCJITSU_LAUNCHER``, ``ROCJITSU_CONFIG`` and ``ROCJITSU_BUILD_DIR`` select
the launcher and simulator config; see :func:`build_rocjitsu_runner` for the
defaults. The suite skips when the launcher cannot be found.

Each test writes ``mutation_report.json`` / ``mutation_report.csv`` under pytest ``tmp_path``
(the ``--output-dir`` equivalent).

At session end, aggregated ``mutation_report.json`` and ``mutation_report.md`` are written to
the pytest rootdir; the ``## Summary`` table is printed (GitHub Actions ``::group::``) for CI
visibility. CI uploads those two files as job artifacts.

A kernel selects its mutation path with a ``// mutate: <kinds>`` line (``wait``, ``memory``,
or both); without one it defaults to wait-stripping. The two paths differ in what a killed
mutant looks like:

- **Wait mutants** remove an ``s_wait*`` the compiler emitted. The deterministic simulator
  issues instructions in a fixed order and returns the expected result with the wait gone,
  so the output is *not* the oracle: a killed wait mutant still exits 0 with matching output
  (``m.correct`` stays true). A non-zero exit there is a real crash to diagnose.
- **Memory mutants** rewrite a sub-dword load as a store to the same address, which genuinely
  changes the computed result. The kernel's own host-side check then fails and the process
  exits non-zero — that divergence *is* the kill, not a crash.

**Detection rules** (each mutant vs the baseline run for that shader):

- **Baseline FP:** unmodified shader must report ``baseline_hazard_count == 0``. Being the
  known-good program, it is where a false positive shows up.
- **Missed hazard (FN):** a killed mutant whose introduced hazard the plugin did not catch,
  i.e. ``m.hazard_count <= baseline_hazard_count``. For both paths the hazard count — not the
  exit code or output — is the detection signal.
- **Crash (wait path only):** a wait mutant that exited non-zero is judged separately — its
  hazard count says how far it got, not how well hazards are detected. A memory mutant's
  non-zero exit is expected and is not treated as a crash.

ROCm tool discovery uses ``ROCM_PATH`` / ``ROCM_HOME`` when set; architecture uses
``TARGET_ARCH`` when set. A kernel that declares ``// requires: <arch>`` is skipped on
every other architecture.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_TEST_DIR = Path(__file__).resolve().parent
_SHADERS_DIR = _TEST_DIR / "shaders"
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from mutate_and_test import (  # noqa: E402
    DEFAULT_ARCH,
    MUTATION_KIND_MEMORY,
    MUTATION_KIND_WAIT,
    KernelBuilder,
    KernelBuilderConfig,
    MutantResult,
    RocjitsuRunner,
    ShaderReport,
    build_rocjitsu_runner,
    discover_shaders,
    find_tool,
    process_shader,
    select_rocm_path,
    shader_mutation_kinds,
    shader_required_archs,
    shader_supports_arch,
    write_csv_report,
    write_json_report,
)


def _all_kernel_paths() -> list[Path]:
    return discover_shaders(_SHADERS_DIR)


@pytest.fixture(scope="session")
def target_arch() -> str:
    return os.environ.get("TARGET_ARCH", DEFAULT_ARCH)


@pytest.fixture(scope="session")
def rocm_path() -> str:
    return select_rocm_path()


@pytest.fixture(scope="session")
def rocm_tools(rocm_path: str) -> Dict[str, str]:
    try:
        return {
            "hipcc": find_tool("hipcc", rocm_path),
            "clang": find_tool("clang", rocm_path),
            "bundler": find_tool("clang-offload-bundler", rocm_path),
            "llvm_mc": find_tool("llvm-mc", rocm_path),
            "llvm_nm": find_tool("llvm-nm", rocm_path),
        }
    except FileNotFoundError as e:
        pytest.skip(f"ROCm tools not available: {e}")


@pytest.fixture(scope="session")
def hazard_runner(
    tmp_path_factory: pytest.TempPathFactory, target_arch: str
) -> RocjitsuRunner:
    """Launcher and config used to run every kernel with hazard detection on."""
    try:
        return build_rocjitsu_runner(
            tmp_path_factory.mktemp("rocjitsu"), arch=target_arch
        )
    except FileNotFoundError as e:
        pytest.skip(
            "rocJitsu is not built; set ROCJITSU_LAUNCHER / ROCJITSU_BUILD_DIR "
            f"or build the launcher first: {e}"
        )


@pytest.fixture
def output_workdir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def kernel_builder_for_shader(
    shader_path: Path,
    rocm_tools: Dict[str, str],
    target_arch: str,
    rocm_path: str,
    output_workdir: Path,
) -> KernelBuilder:
    cfg = KernelBuilderConfig(
        tools=rocm_tools,
        shader_cpp=shader_path,
        asm_file=output_workdir / f"{shader_path.stem}.s",
        arch=target_arch,
        workdir=output_workdir,
        tag=shader_path.name,
        rocm_path=rocm_path,
    )
    return KernelBuilder(cfg)


def _write_reports(report: ShaderReport, arch: str, out_dir: Path) -> None:
    write_json_report([report], out_dir / "mutation_report.json", arch=arch)
    write_csv_report([report], out_dir / "mutation_report.csv")


# Mutants proven passing from redundant wait conditions
_KNOWN_PASSING_MUTANTS: dict[str, dict[int, str]] = {
    # This epilogue wait is already satisfied by an earlier s_wait_kmcnt on the
    # K > 0 path the test runs, so removing it has no effect.
    "wmma_rocwmma": {3: "s_wait_kmcnt 0x0"},
}


def _assert_mutation_ground_truth(report: ShaderReport, artifact_dir: Path) -> None:
    """
    Ground truth vs plugin using only the written JSON payload (see module docstring).

    Rules are implemented here in the test module — not via ``report_writer`` — so
    pass/fail matches what you can inspect in ``mutation_report.json`` on disk.
    """
    baseline_hazard_count = report.baseline_hazard_count
    assert baseline_hazard_count == 0, (
        "baseline false positive: expected 0 hazards on the unmodified shader "
        f"for {report.shader!r}, got baseline_hazard_count={baseline_hazard_count}. "
        f"See {artifact_dir / 'mutation_report.json'}"
    )
    known_passing = _KNOWN_PASSING_MUTANTS.get(report.shader, {})
    crashed: list[str] = []
    missed: list[str] = []
    for i, mutant in enumerate(report.mutants):
        assert mutant.ran, f"mutant {i} ({mutant.wait_instruction!r}) did not run"
        if i in known_passing:
            expected = known_passing[i]
            assert mutant.wait_instruction == expected, (
                f"{report.shader!r} mutant [{i}] is exempt for {expected!r} but the "
                f"assembly now has {mutant.wait_instruction!r}; re-verify the "
                "passing condition and update _KNOWN_PASSING_MUTANTS."
            )
            continue
        detail = (
            f"  [{i}] {mutant.wait_instruction!r}: "
            f"hazard_count={mutant.hazard_count}, baseline={baseline_hazard_count}"
        )
        if mutant.kind == MUTATION_KIND_MEMORY:
            # A load->store mutation genuinely changes the computed result, so
            # the kernel's own host-side check fails and the process exits
            # non-zero. Unlike a stripped wait -- where the deterministic
            # simulator returns the expected result and exit 0 -- that
            # divergence IS the kill, not a crash to diagnose. The detection
            # signal is the plugin's hazard count: it must exceed the baseline.
            if mutant.hazard_count <= baseline_hazard_count:
                outcome = "output matched" if mutant.stdout_match else "output diverged"
                missed.append(f"{detail} ({outcome})")
        elif mutant.exit_code != 0:
            crashed.append(f"{detail}, exit_code={mutant.exit_code!r}")
        elif mutant.hazard_count <= baseline_hazard_count:
            outcome = "output matched" if mutant.stdout_match else "output diverged"
            missed.append(f"{detail} ({outcome})")

    report_path = artifact_dir / "mutation_report.json"
    assert not crashed, (
        "these mutants crashed instead of finishing, so their hazard counts say how "
        "far they got, not what was detected. Diagnose the crash first.\n"
        + "\n".join(crashed)
        + f"\nReport: {report_path}"
    )
    assert not missed, (
        "these mutants ran to completion with a required wait removed, so the plugin "
        "must report more hazards than the baseline and did not. Matching output does "
        "not clear the mutation: the simulators return the same result either way.\n"
        + "\n".join(missed)
        + f"\nReport: {report_path}"
    )


@pytest.mark.mutation_pipeline
@pytest.mark.parametrize("shader_path", _all_kernel_paths(), ids=lambda p: p.stem)
def test_kernel_mutation_pipeline_runs(
    shader_path: Path,
    kernel_builder_for_shader: KernelBuilder,
    output_workdir: Path,
    hazard_runner: RocjitsuRunner,
    target_arch: str,
    mutation_session_reports: list,
) -> None:
    from mutate_and_test import DEFAULT_EXCLUDED_WAITS

    if not shader_supports_arch(shader_path, target_arch):
        pytest.skip(
            f"{shader_path.name} is written for "
            f"{', '.join(shader_required_archs(shader_path))}, not {target_arch}"
        )

    # A kernel declares which mutation path exercises it with a ``// mutate:``
    # line; without one it defaults to wait-stripping. Resolve it here so the
    # pytest driver runs the same path the CLI (__main__) does, rather than
    # forcing every kernel through wait-stripping regardless of annotation.
    kinds = shader_mutation_kinds(shader_path)
    mutate_waits = MUTATION_KIND_WAIT in kinds
    mutate_memory = MUTATION_KIND_MEMORY in kinds

    report = process_shader(
        kernel_builder_for_shader,
        extra_cxxflags=None,
        verbose=False,
        runner=hazard_runner,
        timeout=120,
        excluded_waits=DEFAULT_EXCLUDED_WAITS,
        subprocess_output=False,
        mutate_waits=mutate_waits,
        mutate_memory=mutate_memory,
    )
    mutation_session_reports.append(report)

    _write_reports(report, target_arch, output_workdir)

    assert report.shader == shader_path.stem
    assert report.compile_ok, f"compile_to_assembly failed: {report.error}"
    assert report.build_ok, f"baseline build failed: {report.error}"
    assert report.baseline_ok, (
        f"baseline run failed exit={report.baseline_exit_code!r}; "
        "GPU/runtime may be required for some shaders."
    )
    # Each enabled path contributes its own mutants; a disabled one contributes
    # none. Count by kind so a memory-annotated kernel is not held to the
    # wait-count invariant (and vice versa).
    wait_mutants = [m for m in report.mutants if m.kind == MUTATION_KIND_WAIT]
    memory_mutants = [m for m in report.mutants if m.kind == MUTATION_KIND_MEMORY]
    assert len(wait_mutants) == (len(report.waits_found) if mutate_waits else 0)
    if not mutate_memory:
        assert not memory_mutants
    assert report.mutants, (
        f"{shader_path.stem!r} produced no mutants for kinds "
        f"{sorted(kinds)}; check its // mutate: annotation and the assembly."
    )
    for m in report.mutants:
        assert m.shader == shader_path.stem

    json_path = output_workdir / "mutation_report.json"
    _assert_mutation_ground_truth(report, output_workdir)
