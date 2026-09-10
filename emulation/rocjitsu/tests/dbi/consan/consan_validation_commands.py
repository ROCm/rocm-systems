"""ConSan validation paths, environments, commands, and provenance."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

from consan_tensile_support import (
    TensileValidationPaths,
    resolve_tensile_validation_paths,
    tensile_python_environment,
)
from consan_validation_catalog import (
    CONTROLLED_ENV_PREFIX,
    HIP_MOI_GPU_BENCHMARK_ITERATIONS_ENV,
    HIP_MOI_GPU_BENCHMARK_WARMUP_ITERATIONS_ENV,
    HSA_TOOL_ENVIRONMENT,
    LLAMA_BUILD_DIR_ENV,
    ORDINARY_FORBIDDEN_ENVIRONMENT,
    ORDINARY_MOI_RUNTIME_DEFAULTS,
    PROFILE_IDS,
    PROFILES,
    PROVENANCE_SCHEMA_VERSION,
    PYTORCH_PYTHON_ENV,
    QWEN_BUILD_MANIFEST_SCHEMA_VERSION,
    QWEN_COMPILE_OPTIONS,
    QWEN_OVERHEAD_REPETITIONS,
    RDNA4_MATMUL_DIR_ENV,
    RECORD_REPLAY_STANDARD_RUNTIME_DEFAULTS,
    SAMPLED_STANDARD_RUNTIME_DEFAULTS,
    SCHEMA_VERSION,
    SETTING_CATEGORIES,
    SHARKTANK_PYTHON_ENV,
    SINGLE_REPETITION_TARGETS,
    SOFTWARE_MODEL_ENVIRONMENT,
    TARGET_ENV,
    TENSILE_PYTHON_ENV,
    TENSILE_SHARD_TIMING_CANARY_MS,
    TIMEOUT_SECONDS,
    TOOLS,
    ValidationError,
    Workload,
    WORKLOAD_BY_ID,
    WORKSPACE_ENV,
    _resolved_workload,
    _target_fault_families,
    _workload_for_target,
    _workloads_for_target,
)
from consan_validation_diagnostics import _llvm_readelf
from consan_validation_support import atomic_write_json, git_identity, sha256_file




def _workspace_from_environment() -> Path:
    value = os.environ.get(WORKSPACE_ENV)
    if not value:
        raise ValidationError(f"{WORKSPACE_ENV} is required")
    workspace = Path(value).expanduser().resolve()
    if not workspace.is_dir():
        raise ValidationError(f"{WORKSPACE_ENV} is not a directory: {workspace}")
    return workspace


def _corpus_root(workspace: Path, corpus: str) -> Path:
    """Resolve a validation corpus in both standalone and TheRock layouts."""
    canonical = workspace / corpus
    if canonical.exists() or corpus != "rocm-systems":
        return canonical
    therock = workspace / "TheRock" / "rocm-systems"
    return therock if therock.is_dir() else canonical


def _target(args: argparse.Namespace) -> str:
    value = args.target or os.environ.get(TARGET_ENV)
    if not value or re.fullmatch(r"gfx[0-9a-z]+", value) is None:
        raise ValidationError(
            f"set --target or {TARGET_ENV} to a gfx architecture name"
        )
    return value


@dataclass(frozen=True)
class WorkloadSelection:
    target: str
    workload_id: str
    workload: Workload | None

    @property
    def is_all(self) -> bool:
        return self.workload_id == "all"

    def require_workload(self) -> Workload:
        if self.workload is None:
            raise ValidationError("command requires one concrete workload")
        return self.workload

    def selected_ids(self) -> tuple[str, ...] | None:
        return None if self.is_all else (self.workload_id,)


def _resolve_workload_selection(
    args: argparse.Namespace,
    *,
    allow_all: bool,
) -> WorkloadSelection:
    target = _target(args)
    workload_id = getattr(args, "workload", "all")
    if workload_id == "all":
        if not allow_all:
            command = getattr(args, "command", "this command")
            raise ValidationError(f"{command} requires one concrete workload")
        return WorkloadSelection(target=target, workload_id="all", workload=None)
    return WorkloadSelection(
        target=target,
        workload_id=workload_id,
        workload=_workload_for_target(target, workload_id),
    )


def _command_json(value: str) -> list[str]:
    try:
        command = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("command must be valid JSON") from error
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise argparse.ArgumentTypeError(
            "command must be a non-empty JSON array of non-empty strings"
        )
    return command


def _hook_path(workspace: Path) -> Path:
    suffix = Path("lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so")
    candidates = (
        workspace / "rocjitsu-build" / suffix,
        workspace / "rocjitsu-main-gpu-build" / suffix,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _rdna4_matmul_root(workspace: Path) -> Path:
    configured = os.environ.get(RDNA4_MATMUL_DIR_ENV)
    if configured:
        return Path(os.path.abspath(Path(configured).expanduser()))
    return workspace / "rdna4_matmul"


def _required_paths(
    workspace: Path, workloads: tuple[Workload, ...]
) -> dict[str, Path]:
    hook = _hook_path(workspace)
    paths = {
        "rocjitsu-build": hook.parents[5],
        "hook": hook,
    }
    if any(workload.corpus == "iree-test-suites" for workload in workloads):
        paths["iree-test-suites"] = workspace / "iree-test-suites"
    if any(workload.kind == "qwen" for workload in workloads):
        paths["iree-test-suites-build"] = workspace / "iree-test-suites-build"
    if any(workload.corpus == "hip-moi" for workload in workloads):
        paths["hip-moi"] = workspace / "hip-moi"
    if any(workload.corpus == "rocjitsu-test-corpus" for workload in workloads):
        paths["rocjitsu-test-corpus"] = workspace / "rocjitsu-test-corpus"
    if any(workload.corpus == "rocm-systems" for workload in workloads):
        paths["rocm-systems"] = _corpus_root(workspace, "rocm-systems")
    if any(workload.corpus == "rdna4-matmul" for workload in workloads):
        paths["rdna4-matmul"] = _rdna4_matmul_root(workspace)
    if any(workload.kind == "tensile" for workload in workloads):
        tensile_paths = resolve_tensile_validation_paths(workspace)
        paths["tensilelite"] = tensile_paths.tensilelite
        paths["rocm"] = tensile_paths.rocm
    return paths


def _pytorch_python(workspace: Path | None = None) -> Path:
    # Preserve a virtual environment's interpreter path. Resolving its python
    # symlink would silently bypass that environment and lose torch/triton.
    configured = os.environ.get(PYTORCH_PYTHON_ENV)
    if configured:
        return Path(os.path.abspath(Path(configured).expanduser()))
    if workspace is not None:
        workspace_interpreter = workspace / "consan-pytorch-venv" / "bin" / "python"
        if workspace_interpreter.is_file():
            return workspace_interpreter
    return Path(os.path.abspath(Path(sys.executable).expanduser()))


def _sharktank_python() -> Path:
    return Path(
        os.path.abspath(
            Path(os.environ.get(SHARKTANK_PYTHON_ENV, sys.executable)).expanduser()
        )
    )


def _pytorch_runtime_probe(
    python: Path,
    hook: Path,
    target: str,
    workload: Workload,
    workspace: Path,
    launcher: list[str] | None = None,
) -> dict:
    """Proves that PyTorch can dispatch and that its HSA runtime loads ConSan."""
    probe_source = """
import json
import os
import pathlib
import sys

import torch
import triton

value = torch.ones(1, device="cuda")
torch.cuda.synchronize()
properties = torch.cuda.get_device_properties(0)
maps = pathlib.Path("/proc/self/maps").read_text(encoding="utf-8")
print(json.dumps({
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "triton": triton.__version__,
    "device": torch.cuda.get_device_name(0),
    "arch": getattr(properties, "gcnArchName", None),
    "numeric_oracle": value.item() == 1.0,
    "hook_loaded": sys.argv[1] in maps,
}), flush=True)
# Large precompiled operator libraries can spend longer tearing down a
# software target than executing this linkage canary.  The workload clients
# finalize ConSan explicitly; this doctor probe only needs the flushed result
# and process mappings, so skip unrelated runtime shutdown.
os._exit(0)
"""
    environment = _clean_environment("supercollider", workload, hook, target, workspace)
    # This is a runtime-linkage canary, not a coverage row.  Avoid spending
    # preflight time on PyTorch's large bundled kernel object; the real rows
    # run without this filter and enforce complete coverage independently.
    environment.update(
        {
            "RJ_CONSAN_LOG": "0",
            "RJ_CONSAN_REQUIRE_PATCH": "0",
            "RJ_CONSAN_TEST_KERNEL_FILTER": (
                "__consan_pytorch_runtime_probe_never_matches__"
            ),
        }
    )
    try:
        command = [str(python), "-c", probe_source, str(hook.resolve())]
        probe = subprocess.run(
            _with_launcher(launcher or [], command),
            check=False,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "ok": False,
            "python": str(python),
            "detail": str(error),
        }
    try:
        payload = json.loads(probe.stdout.strip())
    except json.JSONDecodeError:
        payload = None
    reasons = []
    if probe.returncode != 0:
        reasons.append(f"probe exited with status {probe.returncode}")
    if not isinstance(payload, dict):
        reasons.append("probe did not emit its JSON result")
    else:
        arch = payload.get("arch")
        if not isinstance(arch, str) or not arch.startswith(target):
            reasons.append(f"device architecture is {arch!r}, expected {target!r}")
        if not payload.get("numeric_oracle"):
            reasons.append("GPU numeric oracle failed")
        if not payload.get("hook_loaded"):
            reasons.append("PyTorch HSA runtime did not load the ConSan hook")
    if reasons and probe.stderr.strip():
        reasons.append(probe.stderr.strip())
    return {
        "ok": not reasons,
        "python": str(python),
        "detail": payload if payload is not None else probe.stderr.strip(),
        "reasons": reasons,
    }


def _tensile_python() -> Path:
    return Path(
        os.path.abspath(
            Path(os.environ.get(TENSILE_PYTHON_ENV, sys.executable)).expanduser()
        )
    )


def _tensile_runtime_probe(
    python: Path, paths: TensileValidationPaths
) -> dict:
    """Proves that the Tensile driver and its native client are loadable."""
    environment = tensile_python_environment(paths)
    try:
        import_probe = subprocess.run(
            [str(python), "-P", "-c", "from Tensile import Tensile"],
            check=False,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "python": str(python), "detail": str(error)}
    ldd = shutil.which("ldd")
    if ldd is None:
        return {
            "ok": False,
            "python": str(python),
            "detail": "ldd is missing",
            "reasons": ["cannot verify Tensile client runtime dependencies"],
        }
    try:
        client_probe = subprocess.run(
            [ldd, str(paths.client)],
            check=False,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "ok": False,
            "python": str(python),
            "detail": str(error),
            "reasons": ["cannot inspect Tensile client runtime dependencies"],
        }
    import_detail = (import_probe.stderr or import_probe.stdout).strip()
    client_detail = (client_probe.stdout or client_probe.stderr).strip()
    missing_libraries = [
        line.strip()
        for line in client_detail.splitlines()
        if re.search(r"\s=>\s+not found\s*$", line)
    ]
    reasons = []
    if import_probe.returncode != 0:
        reasons.append(f"Tensile import exited with status {import_probe.returncode}")
    if client_probe.returncode != 0:
        reasons.append(f"Tensile client ldd exited with status {client_probe.returncode}")
    if missing_libraries:
        reasons.append(
            "Tensile client has missing runtime libraries: "
            + ", ".join(missing_libraries)
        )
    linkage_detail = "\n".join(missing_libraries)
    if client_probe.returncode != 0 and not linkage_detail:
        linkage_detail = client_detail
    detail = "\n".join(
        part for part in (import_detail, linkage_detail) if part
    ) or "Tensile import and client runtime closure passed"
    return {
        "ok": not reasons,
        "python": str(python),
        "detail": detail,
        "reasons": reasons,
    }


@dataclass(frozen=True)
class LlamaRuntime:
    build_root: Path
    executable: Path
    libraries: dict[str, Path]


def _llama_runtime_files(build_root: Path) -> dict[str, Path]:
    source_root = build_root / "third_party" / "llama.cpp" / "ggml" / "src"
    return {
        "ggml": source_root / "libggml.so",
        "ggml-base": source_root / "libggml-base.so",
        "ggml-cpu": source_root / "libggml-cpu.so",
        "ggml-hip": source_root / "ggml-hip" / "libggml-hip.so",
    }


def _llama_runtime(workspace: Path, target: str, name: str) -> LlamaRuntime:
    configured = os.environ.get(LLAMA_BUILD_DIR_ENV)
    if configured:
        build_roots = (Path(os.path.abspath(Path(configured).expanduser())),)
    else:
        build_roots = (
            workspace / "rocjitsu-test-corpus-build" / "kernels" / target,
            workspace
            / "rocjitsu-test-corpus"
            / ".pytest-artifacts-rdna4-llama-baseline"
            / "_suite_shards"
            / "kernels_shard_0"
            / "kernels"
            / target
            / "build",
        )

    failures = []
    for build_root in build_roots:
        executable = build_root / "cases" / "llama.cpp" / name
        libraries = _llama_runtime_files(build_root)
        missing = [
            path for path in (executable, *libraries.values()) if not path.is_file()
        ]
        if not missing:
            return LlamaRuntime(build_root, executable, libraries)
        failures.append((build_root, missing))

    details = "; ".join(
        f"{build_root}: missing {', '.join(str(path) for path in missing)}"
        for build_root, missing in failures
    )
    if configured:
        raise ValidationError(
            f"{LLAMA_BUILD_DIR_ENV} must name a complete llama build root; {details}"
        )
    raise ValidationError(f"cannot locate a complete llama runtime; checked {details}")


def _llama_executable(workspace: Path, target: str, name: str) -> Path:
    return _llama_runtime(workspace, target, name).executable


def _llama_library_environment(
    runtime: LlamaRuntime, environment: dict[str, str]
) -> None:
    library_directories = list(
        dict.fromkeys(str(path.parent) for path in runtime.libraries.values())
    )
    existing = environment.get("LD_LIBRARY_PATH")
    if existing:
        library_directories.append(existing)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_directories)


def _input_files(workspace: Path, target: str, workload: Workload) -> dict[str, Path]:
    workload = _resolved_workload(target, workload)
    if workload.kind == "pytorch":
        return {
            "python": _pytorch_python(workspace),
            "workload-source": Path(__file__).with_name(workload.relative_path),
        }
    if workload.kind == "tensile":
        paths = resolve_tensile_validation_paths(workspace, target)
        return {
            "python": _tensile_python(),
            "workload-source": Path(__file__).with_name("consan_tensile_validation.py"),
            "support-source": Path(__file__).with_name("consan_tensile_support.py"),
            "config": _corpus_root(workspace, workload.corpus)
            / workload.relative_path,
            "client": paths.client,
            "wrapper": paths.wrapper,
            "rocjitsu": paths.rocjitsu,
            "rocjitsu-config": paths.rocjitsu_config,
            "llvm-readelf": paths.llvm_readelf,
            "amdclang++": paths.rocm / "bin" / "amdclang++",
        }
    if workload.kind == "llama":
        case = (
            "mul_mat_vec_q"
            if workload.id == "llama-rdna4-mul-mat-vec-q"
            else "rms_norm"
        )
        runtime = _llama_runtime(workspace, target, workload.relative_path)
        return {
            "python": Path(os.path.abspath(Path(sys.executable).expanduser())),
            "workload-source": Path(__file__).with_name("consan_llama_validation.py"),
            "case": workspace
            / "rocjitsu-test-corpus"
            / "corpus"
            / "kernels"
            / "cases"
            / "llama.cpp"
            / case
            / "case.json",
            "executable": runtime.executable,
            **runtime.libraries,
        }
    if workload.kind == "rdna4-matmul":
        root = _rdna4_matmul_root(workspace)
        return {
            "workload-source": Path(__file__).with_name(
                "consan_rdna4_matmul_validation.py"
            ),
            "project-source": root / "rdna4_matmul.hip",
            "project-build-script": root / "build_and_test.sh",
            "executable": root / workload.relative_path,
        }
    if workload.kind == "qwen":
        root = workspace / workload.relative_path
        data = root / "hf" / "qwen3-600m"
        return {
            "vmfb": root / target / "qwen3-600m.vmfb",
            "build-manifest": root / target / "qwen3-600m.consan-build.json",
            "source": workspace
            / "iree-test-suites/torch_models/qwen3-600m/model.mlir",
            "parameters": data / "real_weights.irpa",
            "input": data / "inference_input.0.bin",
            "expected": data / "inference_output.0.bin",
        }
    if workload.kind == "gtest":
        return {"executable": workspace / workload.relative_path}
    if workload.kind == "native-executable":
        return {"executable": workspace / workload.relative_path}
    source = workspace / workload.relative_path
    if workload.sharktank_workload in {"tp1", "tp2"}:
        assets = source.parent / "assets"
        names = (
            ("toy_llama.mlir", "toy_llama.irpa")
            if workload.sharktank_workload == "tp1"
            else (
                "toy_llama_tp2.mlir",
                "toy_llama_tp2.irpa",
                "toy_llama_tp2.rank0.irpa",
                "toy_llama_tp2.rank1.irpa",
            )
        )
        return {"workload-source": source, **{name: assets / name for name in names}}
    assets = source.parent / "assets" / "text_model" / "toy"
    return {
        "workload-source": source,
        "bf16.mlir": assets / "bf16.mlir",
        "bf16_parameters.irpa": assets / "bf16_parameters.irpa",
        "input": assets / "forward_bs4_arg0_input_ids.irpa",
        "expected": assets / "forward_bs4_expected_result0_last_hidden_state_f32.irpa",
    }


def _qwen_compile_options(target: str, encoder_output: Path) -> tuple[str, ...]:
    return (
        f"--iree-rocm-target={target}",
        *QWEN_COMPILE_OPTIONS,
        f"--iree-parameter-encoder-output-file={encoder_output}",
    )


def _qwen_build_manifest(
    target: str,
    source: Path,
    vmfb: Path,
    compiler: Path,
    encoder_output: Path,
) -> dict[str, object]:
    return {
        "schema_version": QWEN_BUILD_MANIFEST_SCHEMA_VERSION,
        "workload": "qwen-prefill",
        "target": target,
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "compiler": {"path": str(compiler), "sha256": sha256_file(compiler)},
        "compile_options": list(_qwen_compile_options(target, encoder_output)),
        "vmfb": {"path": str(vmfb), "sha256": sha256_file(vmfb)},
    }


def _qwen_build_check(workspace: Path, target: str) -> dict[str, object]:
    inputs = _input_files(workspace, target, WORKLOAD_BY_ID["qwen-prefill"])
    manifest_path = inputs["build-manifest"]
    result: dict[str, object] = {
        "ok": False,
        "manifest": str(manifest_path),
        "reasons": [],
    }
    reasons = result["reasons"]
    assert isinstance(reasons, list)
    if not manifest_path.is_file():
        reasons.append("missing canonical Qwen build manifest; run prepare")
        return result
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        reasons.append(f"cannot read canonical Qwen build manifest: {error}")
        return result
    if not isinstance(document, dict):
        reasons.append("canonical Qwen build manifest is not a JSON object")
        return result
    expected_options = list(
        _qwen_compile_options(
            target, manifest_path.with_name("qwen3-600m.parameter-encoder.mlir")
        )
    )
    scalar_expectations = {
        "schema_version": QWEN_BUILD_MANIFEST_SCHEMA_VERSION,
        "workload": "qwen-prefill",
        "target": target,
        "compile_options": expected_options,
    }
    for field, expected in scalar_expectations.items():
        if document.get(field) != expected:
            reasons.append(f"Qwen build manifest has stale {field}")
    for field, path in (("source", inputs["source"]), ("vmfb", inputs["vmfb"])):
        record = document.get(field)
        if (
            not path.is_file()
            or not isinstance(record, dict)
            or record.get("sha256") != sha256_file(path)
        ):
            reasons.append(f"Qwen build manifest {field} hash does not match")
    result["ok"] = not reasons
    return result


def _prepare_qwen(workspace: Path, target: str) -> dict[str, object]:
    workload = _workload_for_target(target, "qwen-prefill")
    inputs = _input_files(workspace, target, workload)
    source = inputs["source"]
    if not source.is_file():
        raise ValidationError(f"missing Qwen source MLIR: {source}")
    compiler_name = shutil.which("iree-compile")
    if compiler_name is None:
        raise ValidationError("iree-compile is required to prepare Qwen")
    compiler = Path(compiler_name).resolve()
    vmfb = inputs["vmfb"]
    vmfb.parent.mkdir(parents=True, exist_ok=True)
    candidate = vmfb.with_name(f"qwen3-600m.{os.getpid()}.next.vmfb")
    encoder_output = vmfb.with_name("qwen3-600m.parameter-encoder.mlir")
    command = [
        str(compiler),
        str(source),
        *_qwen_compile_options(target, encoder_output),
        "-o",
        str(candidate),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise ValidationError(
            f"Qwen compilation failed with exit code {completed.returncode}"
        )
    if not candidate.is_file() or candidate.stat().st_size == 0:
        raise ValidationError("Qwen compilation produced no VMFB")
    os.replace(candidate, vmfb)
    document = _qwen_build_manifest(
        target, source, vmfb, compiler, encoder_output
    )
    atomic_write_json(inputs["build-manifest"], document)
    return document


def _doctor(
    workspace: Path,
    target: str,
    workload_ids: tuple[str, ...] | None = None,
    launcher: list[str] | None = None,
) -> dict:
    selected_ids = (
        tuple(workload.id for workload in _workloads_for_target(target))
        if workload_ids is None
        else workload_ids
    )
    workloads = tuple(
        _workload_for_target(target, workload_id) for workload_id in selected_ids
    )
    paths = _required_paths(workspace, workloads)
    path_checks = {
        label: {
            "path": str(path),
            "present": path.is_file() if label == "hook" else path.is_dir(),
        }
        for label, path in paths.items()
    }
    # PyTorch's runtime probe performs a numeric dispatch and reports the
    # device architecture from the same process that loads the ConSan hook.
    # Requiring a separately installed rocminfo for a PyTorch-only row adds no
    # target assurance and can reject an otherwise complete wheel-based setup.
    required_tools = (
        ("rocminfo",)
        if any(workload.kind != "pytorch" for workload in workloads)
        else ()
    )
    if any(workload.kind == "qwen" for workload in workloads):
        required_tools = TOOLS
    tools = {tool: shutil.which(tool) for tool in required_tools}
    for workload in workloads:
        for label, path in _input_files(workspace, target, workload).items():
            executable = (
                workload.kind == "native-executable" and label == "executable"
            ) or (
                workload.kind == "tensile"
                and label
                in {
                    "python",
                    "client",
                    "wrapper",
                    "rocjitsu",
                    "llvm-readelf",
                    "amdclang++",
                }
            )
            path_checks[f"workload:{workload.id}:{label}"] = {
                "path": str(path),
                "present": path.is_file()
                and (not executable or os.access(path, os.X_OK)),
            }
    runtimes = {}
    pytorch_workloads = tuple(
        workload for workload in workloads if workload.kind == "pytorch"
    )
    if pytorch_workloads:
        python = _pytorch_python(workspace)
        if python.is_file():
            runtimes["pytorch"] = _pytorch_runtime_probe(
                python,
                _hook_path(workspace),
                target,
                pytorch_workloads[0],
                workspace,
                launcher,
            )
        else:
            runtimes["pytorch"] = {
                "ok": False,
                "python": str(python),
                "detail": "interpreter is missing",
            }
    tensile_workloads = tuple(
        workload for workload in workloads if workload.kind == "tensile"
    )
    if tensile_workloads:
        python = _tensile_python()
        tensile_paths = resolve_tensile_validation_paths(workspace, target)
        if (
            python.is_file()
            and os.access(python, os.X_OK)
            and tensile_paths.tensilelite.is_dir()
        ):
            runtimes["tensile"] = _tensile_runtime_probe(python, tensile_paths)
        else:
            runtimes["tensile"] = {
                "ok": False,
                "python": str(python),
                "detail": "interpreter or TensileLite package is missing",
            }
    artifacts = {}
    if any(workload.kind == "qwen" for workload in workloads):
        artifacts["qwen-prefill"] = _qwen_build_check(workspace, target)
    ok = (
        all(item["present"] for item in path_checks.values())
        and all(tools.values())
        and all(item["ok"] for item in runtimes.values())
        and all(item["ok"] for item in artifacts.values())
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "workspace": str(workspace),
        "target": target,
        "workloads": list(selected_ids),
        "paths": path_checks,
        "runtimes": runtimes,
        "artifacts": artifacts,
        "tools": tools,
    }


def _manifest(target: str) -> dict:
    def manifest_workload(workload: Workload) -> dict:
        # The manifest is the executable target contract, not the union declared
        # by the target-independent Workload row.
        return asdict(_effective_workload(target, workload))

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": "consan-real-workload-validation-v1",
        "workspace_environment": WORKSPACE_ENV,
        "target": target,
        "tools_from_path": list(TOOLS),
        "profiles": [asdict(PROFILES[profile]) for profile in PROFILE_IDS],
        "workloads": [
            manifest_workload(workload) for workload in _workloads_for_target(target)
        ],
        "ordinary_forbidden_environment": list(ORDINARY_FORBIDDEN_ENVIRONMENT),
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_gpu_parallelism": 4,
    }


def _clean_environment(
    profile: str | None,
    workload: Workload,
    hook: Path | None,
    target: str | None,
    workspace: Path,
) -> dict[str, str]:
    if target is not None:
        workload = _resolved_workload(target, workload)
    ignored_environment = HSA_TOOL_ENVIRONMENT | {
        "HIP_TARGET",
        HIP_MOI_GPU_BENCHMARK_ITERATIONS_ENV,
        HIP_MOI_GPU_BENCHMARK_WARMUP_ITERATIONS_ENV,
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(CONTROLLED_ENV_PREFIX)
        and key not in ignored_environment
        and key not in SOFTWARE_MODEL_ENVIRONMENT
    }
    environment.update(dict(workload.command_environment))
    if target is not None:
        environment["HIP_TARGET"] = target
    if workload.kind == "llama":
        if target is None:
            raise ValidationError("llama runtime environment requires a target")
        runtime = _llama_runtime(workspace, target, workload.relative_path)
        _llama_library_environment(runtime, environment)
    if target == "gfx1250":
        # ROCr loads the gfx1250 A0 HotSwap helper by soname when it is not
        # installed beside libhsa-runtime64. Standalone RocJITsu builds keep
        # that companion DSO beside the ConSan hook, so make the complete hook
        # bundle visible to validation payloads and health probes.
        companion_dir = str((hook or _hook_path(workspace)).parent)
        existing = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            (companion_dir, existing) if existing else (companion_dir,)
        )
    if profile is None:
        return environment
    if hook is None:
        raise ValidationError("instrumented runtime environment requires a hook")
    config = PROFILES[profile]
    environment.update(config.environment)
    environment.update(
        {
            "HSA_TOOLS_LIB": str(hook),
            "RJ_CONSAN_LOG": "1",
        }
    )
    if (
        profile == "record-replay"
        and workload.record_replay_runtime_sample_stride is not None
    ):
        stride = workload.record_replay_runtime_sample_stride
        if stride <= 0 or stride & (stride - 1):
            raise ValidationError(
                f"{workload.id} has a non-power-of-two Record/Replay sampling stride: {stride}"
            )
        environment["RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE"] = str(stride)
    if workload.kind in {"pytorch", "llama", "rdna4-matmul"}:
        # These clients use a modern HSA runtime which returns after successful
        # rocprofiler registration unless legacy environment tools are
        # explicitly requested. ConSan is currently such a tool.
        environment["HSA_TOOLS_ROCPROFILER_V1_TOOLS"] = "1"
    if not workload.moi_record_evidence_expected:
        environment["RJ_CONSAN_MOI_REQUIRE_RECORDS"] = "0"
    if workload.id == "qwen-prefill" and profile == "sampled":
        environment["RJ_CONSAN_MOI_REQUIRE_RECORDS"] = "1"
    return environment


def _run_environment(
    profile: str | None,
    workload: Workload,
    hook: Path,
    target: str,
    phase: str,
    workspace: Path,
) -> dict[str, str]:
    if phase not in {"clean", "overhead"}:
        raise ValidationError(f"unsupported validation phase: {phase}")
    return _clean_environment(profile, workload, hook, target, workspace)


def _controlled_environment(environment: dict[str, str]) -> dict[str, str]:
    runtime_names = {
        "HSA_TOOLS_LIB",
        "HSA_TOOLS_ROCPROFILER_V1_TOOLS",
        "CTEST_PARALLEL_LEVEL",
        "HIP_PATH",
        "HIP_TARGET",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONPATH",
        "ROCM_PATH",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HSA_OVERRIDE_GFX_VERSION",
        HIP_MOI_GPU_BENCHMARK_ITERATIONS_ENV,
        HIP_MOI_GPU_BENCHMARK_WARMUP_ITERATIONS_ENV,
    }
    names = {
        key
        for key in environment
        if key.startswith(CONTROLLED_ENV_PREFIX) or key in runtime_names
    }
    return {key: environment[key] for key in sorted(names)}


def _setting_metadata(name: str) -> dict:
    if name in HSA_TOOL_ENVIRONMENT | {"HIP_TARGET"}:
        category = "runtime-plumbing"
    elif name == "CTEST_PARALLEL_LEVEL":
        category = "fault-containment"
    elif name.startswith("RJ_CONSAN_FAULT_"):
        category = "fault-injection"
    elif name in {
        "RJ_CONSAN_MODE",
        "RJ_CONSAN_POLICY",
        "RJ_CONSAN_MOI_TRACK_BARRIERS",
        "RJ_CONSAN_MOI_TRACK_ATOMICS",
    }:
        category = "instrumentation-selection"
    elif name in ORDINARY_FORBIDDEN_ENVIRONMENT or name in {
        "RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE",
        "RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET",
        "RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES",
        "RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT",
        "RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES",
        "RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES",
        "RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES",
    }:
        category = "workload-tuning"
    elif name == "RJ_CONSAN_LOG" or "_REQUIRE_" in name or "_FORBID_" in name:
        category = "acceptance-assertion"
    else:
        raise ValidationError(f"unclassified validation setting: {name}")
    result = {
        "category": category,
        "category_description": SETTING_CATEGORIES[category],
        "usability_exception": category == "workload-tuning",
    }
    if name in {
        "RJ_CONSAN_MOI_TRACK_BARRIERS",
        "RJ_CONSAN_MOI_TRACK_ATOMICS",
    }:
        result["usability_note"] = (
            "Ordinary MOI enables this event family by default; an explicit "
            "value is an expert compatibility override."
        )
    elif category == "workload-tuning":
        result["usability_note"] = (
            "This is a workload-specific non-default operating point."
        )
    return result


def _audited_settings(environment: dict[str, str]) -> list[dict]:
    names = sorted(
        name
        for name in environment
        if name.startswith(CONTROLLED_ENV_PREFIX)
        or name in HSA_TOOL_ENVIRONMENT | {"HIP_TARGET", "CTEST_PARALLEL_LEVEL"}
    )
    return [
        {"name": name, "value": environment[name], **_setting_metadata(name)}
        for name in names
    ]


def _audited_unsets(names: list[str]) -> list[dict]:
    return [
        {
            "name": name,
            "operation": "unset",
            **_setting_metadata(name),
            "usability_note": (
                "Fault-only policy relaxes this clean-run acceptance assertion."
            ),
        }
        for name in sorted(names)
    ]


def _profile_runtime_defaults(
    profile: str, explicit_environment: dict[str, str] | None = None
) -> list[dict]:
    if PROFILES[profile].flavor != "moi":
        return []
    defaults = dict(ORDINARY_MOI_RUNTIME_DEFAULTS)
    if profile == "record-replay":
        defaults.update(RECORD_REPLAY_STANDARD_RUNTIME_DEFAULTS)
    elif profile == "sampled":
        defaults.update(SAMPLED_STANDARD_RUNTIME_DEFAULTS)
    explicit_names = set(explicit_environment or {})
    settings = _audited_settings(
        {name: value for name, value in defaults.items() if name not in explicit_names}
    )
    return [
        {
            **setting,
            "source": "standard-profile-runtime-default",
            "usability_exception": False,
            "usability_note": "The standard profile selects this automatically.",
        }
        for setting in settings
    ]


def _qwen_command(
    workspace: Path,
    target: str,
    overhead: bool,
    output: Path,
    repetitions_override: int | None = None,
) -> list[str]:
    root = workspace / "iree-test-suites-build" / "torch_models" / "qwen3-600m"
    data = root / "hf" / "qwen3-600m"
    command = [
        "iree-benchmark-module" if overhead else "iree-run-module",
        "--device=hip",
        f"--module={root / target / 'qwen3-600m.vmfb'}",
        f"--parameters=model={data / 'real_weights.irpa'}",
        "--function=main",
        f"--input=1x5xi64=@{data / 'inference_input.0.bin'}",
    ]
    if overhead:
        repetitions = repetitions_override or QWEN_OVERHEAD_REPETITIONS.get(target, 10)
        command.extend(
            [
                f"--benchmark_repetitions={repetitions}",
                "--benchmark_min_time=0s",
                f"--benchmark_out={output}",
                "--benchmark_out_format=json",
            ]
        )
    else:
        command.extend(
            [
                f"--expected_output=1x5x151936xf32=@{data / 'inference_output.0.bin'}",
                "--expected_f32_threshold=0.05",
            ]
        )
    return command


def _health_smoke_command(
    workspace: Path, target: str, workload: Workload, output: Path
) -> list[str]:
    # Qwen is a useful universal smoke on hardware targets, but merely having
    # its files does not make it a bounded software-emulator health probe.  The
    # gfx1250 campaign already qualifies the single D128-pressure exact oracle
    # as its independent, sub-30-second target-dispatch denominator.
    if target == "gfx1250":
        smoke_workload = WORKLOAD_BY_ID["d128-pressure"]
        if all(
            path.is_file()
            for path in _input_files(workspace, target, smoke_workload).values()
        ):
            return _workload_command(
                workspace, target, smoke_workload, "overhead", output
            )
    qwen = WORKLOAD_BY_ID["qwen-prefill"]
    qwen_command = _qwen_command(workspace, target, False, output)
    if shutil.which(qwen_command[0]) and all(
        path.is_file() for path in _input_files(workspace, target, qwen).values()
    ):
        return qwen_command
    # A workload-scoped doctor permits an independently ready row to proceed
    # when unrelated Qwen artifacts are absent. Its destructive health gate
    # must honor the same contract instead of manufacturing an unhealthy GPU
    # result from a missing universal smoke file.
    return _workload_command(workspace, target, workload, "clean", output)


def _inner_repetitions(target: str, phase: str, workload: Workload) -> int:
    if phase != "overhead" or target in SINGLE_REPETITION_TARGETS:
        return 1
    # A workload either collects warm in-process samples or declares multiple
    # isolated outer processes. Do not multiply the two repetition axes.
    return 1 if workload.overhead_processes > 1 else 10


def _workload_command(
    workspace: Path,
    target: str,
    workload: Workload,
    phase: str,
    output: Path,
    inner_repetitions_override: int | None = None,
    *,
    tensile_exact_problem_sizes: tuple[tuple[int, ...], ...] | None = None,
    tensile_expected_numeric_rows: int | None = None,
    tensile_expected_client_passes: int | None = None,
) -> list[str]:
    workload = _resolved_workload(target, workload)
    overhead = phase == "overhead"
    if workload.kind == "qwen":
        return _qwen_command(
            workspace,
            target,
            overhead,
            output,
            inner_repetitions_override,
        )
    if workload.kind == "sharktank":
        # The active architecture campaigns use one end-to-end repetition.
        # Keep both the outer process count and this inner suite count at one.
        repetitions = inner_repetitions_override or _inner_repetitions(
            target, phase, workload
        )
        command = [
            str(_sharktank_python()),
            str(Path(__file__).with_name("consan_sharktank_validation.py")),
            "--suite-root",
            str(workspace / "iree-test-suites"),
            "--workload",
            str(workload.sharktank_workload),
            "--mode",
            str(workload.sharktank_mode),
            "--repetitions",
            str(repetitions),
            "--label",
            f"{workload.id}-{phase}",
        ]
        if workload.sharktank_skip_warmup:
            command.append("--skip-warmup")
        return command
    if workload.kind == "pytorch":
        # Large rows may declare isolated outer processes because repeated
        # instrumented dispatches accumulate bounded report state. Small rows
        # retain warm in-process timing, and simulator targets stay single-shot.
        repetitions = inner_repetitions_override or _inner_repetitions(
            target, phase, workload
        )
        return [
            str(_pytorch_python(workspace)),
            str(Path(__file__).with_name(workload.relative_path)),
            "--workload",
            workload.id.removeprefix("pytorch-"),
            "--repetitions",
            str(repetitions),
            "--label",
            f"{workload.id}-{phase}",
        ]
    if workload.kind == "tensile":
        minimum_timed_ms = workload.tensile_minimum_timed_ms
        if tensile_exact_problem_sizes is not None:
            # Each leaf proves a positive timing canary. The parent result
            # aggregates all shards and enforces the workload-level minimum.
            minimum_timed_ms = min(
                minimum_timed_ms, TENSILE_SHARD_TIMING_CANARY_MS
            )
        command = [
            str(_tensile_python()),
            str(Path(__file__).with_name("consan_tensile_validation.py")),
            "--workspace",
            str(workspace),
            "--config",
            str(_corpus_root(workspace, workload.corpus) / workload.relative_path),
            "--gpu-target",
            target,
            "--output-dir",
            str(output.parent / "tensile-work"),
            "--repetitions",
            "1",
            "--minimum-timed-ms",
            str(minimum_timed_ms),
            "--label",
            f"{workload.id}-{phase}",
        ]
        if workload.tensile_inner_timeout_seconds is not None:
            command.extend(
                (
                    "--timeout-seconds",
                    str(workload.tensile_inner_timeout_seconds),
                )
            )
        expected_numeric_rows = (
            tensile_expected_numeric_rows
            if tensile_expected_numeric_rows is not None
            else workload.tensile_expected_numeric_rows
        )
        if expected_numeric_rows is not None:
            command.extend(
                (
                    "--expect-numeric-rows",
                    str(expected_numeric_rows),
                )
            )
        expected_client_passes = (
            tensile_expected_client_passes
            if tensile_expected_client_passes is not None
            else workload.tensile_expected_client_passes
        )
        if expected_client_passes is not None:
            command.extend(
                (
                    "--expect-client-passes",
                    str(expected_client_passes),
                )
            )
        if tensile_exact_problem_sizes is not None:
            source_blocks = (
                workload.tensile_expected_source_exact_problem_size_blocks
            )
            source_problem_sizes = tuple(
                dict.fromkeys(
                    size
                    for shard in workload.tensile_exact_problem_size_shards
                    for size in shard
                )
            )
            if not source_problem_sizes:
                raise ValidationError(
                    f"{workload.id} command selected a problem-size shard "
                    "without a source inventory"
                )
            command.extend(
                (
                    "--exact-problem-sizes-json",
                    json.dumps(tensile_exact_problem_sizes, separators=(",", ":")),
                )
            )
            if source_blocks:
                command.extend(
                    (
                        "--expect-source-exact-problem-size-blocks-json",
                        json.dumps(source_blocks, separators=(",", ":")),
                    )
                )
            else:
                command.extend(
                    (
                        "--expect-source-exact-problem-sizes-json",
                        json.dumps(source_problem_sizes, separators=(",", ":")),
                    )
                )
        if workload.tensile_streamk_fixed_grid is not None:
            command.extend(
                (
                    "--streamk-fixed-grid",
                    str(workload.tensile_streamk_fixed_grid),
                )
            )
        if workload.tensile_streamk_mode is not None:
            command.extend(
                (
                    "--require-streamk-mode",
                    str(workload.tensile_streamk_mode),
                )
            )
        return command
    if workload.kind == "llama":
        llama_workload = (
            "mul-mat-vec-q"
            if workload.id == "llama-rdna4-mul-mat-vec-q"
            else "rms-norm"
        )
        command = [
            sys.executable,
            str(Path(__file__).with_name("consan_llama_validation.py")),
            "--executable",
            str(_llama_executable(workspace, target, workload.relative_path)),
            "--workload",
            llama_workload,
            "--output-dir",
            str(output.parent / f"{output.stem}-llama-work"),
        ]
        if overhead and workload.id == "llama-rdna4-mul-mat-vec-q":
            command.extend(
                [
                    "--n-embd",
                    "1024",
                    "--benchmark-iterations",
                    str(inner_repetitions_override or 40_000),
                    "--benchmark-warmup-iterations",
                    "5",
                    "--minimum-timed-ms",
                    str(workload.self_timed_device_minimum_ms),
                ]
            )
        return command
    if workload.kind == "rdna4-matmul":
        command = [
            sys.executable,
            str(Path(__file__).with_name("consan_rdna4_matmul_validation.py")),
            "--executable",
            str(_rdna4_matmul_root(workspace) / workload.relative_path),
            "--workload",
            workload.id.removeprefix("rdna4-matmul-"),
            "--phase",
            "clean" if phase in {"clean", "fault"} else "warm",
            "--repetitions",
            "1",
            "--minimum-timed-ms",
            str(workload.self_timed_device_minimum_ms),
            "--label",
            f"{workload.id}-{phase}",
        ]
        if inner_repetitions_override is not None:
            command.extend(["--fixed-iterations", str(inner_repetitions_override)])
        return command
    if workload.kind == "native-executable":
        return [str(workspace / workload.relative_path), *workload.command_arguments]
    executable = workspace / workload.relative_path
    selected_filter = (
        workload.fault_filter or workload.clean_filter
        if phase == "fault"
        else workload.overhead_filter if overhead else workload.clean_filter
    )
    return [str(executable), f"--gtest_filter={selected_filter}"]


def _workload_commands(
    workspace: Path,
    target: str,
    workload: Workload,
    phase: str,
    output: Path,
    inner_repetitions_override: int | None = None,
) -> list[list[str]]:
    workload = _resolved_workload(target, workload)
    shards = workload.tensile_exact_problem_size_shards
    if not shards:
        return [
            _workload_command(
                workspace,
                target,
                workload,
                phase,
                output,
                inner_repetitions_override,
            )
        ]
    expected_rows = workload.tensile_expected_numeric_rows_per_shard or (
        None,
    ) * len(shards)
    expected_clients = workload.tensile_expected_client_passes_per_shard or (
        None,
    ) * len(shards)
    return [
        _workload_command(
            workspace,
            target,
            workload,
            phase,
            output,
            inner_repetitions_override,
            tensile_exact_problem_sizes=shard,
            tensile_expected_numeric_rows=shard_expected_rows,
            tensile_expected_client_passes=shard_expected_clients,
        )
        for shard, shard_expected_rows, shard_expected_clients in zip(
            shards,
            expected_rows,
            expected_clients,
            strict=True,
        )
    ]


def _fault_workload_command(
    workspace: Path,
    target: str,
    workload: Workload,
    output: Path,
) -> list[str]:
    """Build one bounded command for inventory and exact-one fault execution."""
    workload = _resolved_workload(target, workload)
    index = workload.tensile_fault_shard_index
    if index is None:
        return _workload_command(workspace, target, workload, "fault", output)
    return _workload_command(
        workspace,
        target,
        workload,
        "fault",
        output,
        tensile_exact_problem_sizes=workload.tensile_exact_problem_size_shards[index],
        tensile_expected_numeric_rows=(
            workload.tensile_expected_numeric_rows_per_shard[index]
            if workload.tensile_expected_numeric_rows_per_shard
            else None
        ),
        tensile_expected_client_passes=(
            workload.tensile_expected_client_passes_per_shard[index]
            if workload.tensile_expected_client_passes_per_shard
            else None
        ),
    )


def _write_provenance(
    workspace: Path,
    target: str,
    workload: Workload,
    workload_root: Path,
    launcher: list[str] | None = None,
) -> Path:
    workload_root.mkdir(parents=True, exist_ok=True)
    path = workload_root / "provenance.json"
    hook = _hook_path(workspace)
    files = {"hook": hook, **_input_files(workspace, target, workload)}
    llvm_readelf = _llvm_readelf()
    if llvm_readelf is not None and llvm_readelf.is_file():
        files["llvm-readelf"] = llvm_readelf
    document = {
        "schema_version": SCHEMA_VERSION,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "target": target,
        "workload": workload.id,
        "files": {
            label: {
                "path": str(file),
                "size": file.stat().st_size,
                "sha256": sha256_file(file),
            }
            for label, file in files.items()
        },
        "sources": _source_identities(workspace, workload),
        "manifest": _manifest(target),
        "environment_selectors": {
            name: {"present": name in os.environ, "value": os.environ.get(name)}
            for name in (
                "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL",
                "HSA_OVERRIDE_GFX_VERSION",
            )
        },
        "machine": _machine_identity(target),
        "runtime_tools": _runtime_tool_identities(llvm_readelf, hook),
        "workload_runtime": _workload_runtime_identity(
            workspace, target, workload, launcher
        ),
        "observations": _empirical_observation_snapshot(),
    }
    normalized_document = json.loads(json.dumps(document))
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValidationError(
                f"cannot read existing provenance {path}: {error}"
            ) from error
        existing_schema = existing.get("provenance_schema_version")
        if existing_schema != PROVENANCE_SCHEMA_VERSION:
            raise ValidationError(
                "provenance schema changed "
                f"from {existing_schema!r} to {PROVENANCE_SCHEMA_VERSION}; "
                f"use a new artifact root instead of resuming {path}"
            )
        stable_existing = {
            key: value for key, value in existing.items() if key != "observations"
        }
        stable_document = {
            key: value
            for key, value in normalized_document.items()
            if key != "observations"
        }
        if stable_existing != stable_document:
            raise ValidationError(
                f"provenance conflicts with existing artifact: {path}"
            )
        return path
    atomic_write_json(path, document)
    return path


def _read_identity_file(path: Path) -> dict[str, object]:
    try:
        value = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as error:
        return {"available": False, "reason": str(error)}
    return {"available": True, "value": value}


def _command_identity(
    command: list[str],
    timeout: int = 10,
    environment: dict[str, str] | None = None,
    output_normalizer: Callable[[str], str] | None = None,
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "command": command, "reason": str(error)}
    output = completed.stdout or ""
    if output_normalizer is not None:
        output = output_normalizer(output)
    limit = 65536
    return {
        "available": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "output": output[:limit],
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "output_truncated": len(output) > limit,
    }


_DYNAMIC_LOADER_ADDRESS = re.compile(
    r"[ \t]+\(0x[0-9a-fA-F]+\)[ \t]*(?=\r?$)", re.MULTILINE
)


def _normalize_dynamic_loader_output(output: str) -> str:
    """Remove per-process load addresses while retaining the loader closure."""
    return _DYNAMIC_LOADER_ADDRESS.sub("", output)


def _runtime_library_records(paths: dict[str, Path]) -> dict[str, object]:
    records = {}
    for label, path in paths.items():
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ValidationError(
                f"cannot resolve loaded {label} runtime {path}: {error}"
            ) from error
        records[label] = {
            "path": str(path),
            "resolved_path": str(resolved),
            "size": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    return records


def _runtime_libraries_from_ldd_output(output: str) -> dict[str, Path]:
    prefixes = {
        "hip-runtime": "libamdhip64.so",
        "hsa-runtime": "libhsa-runtime64.so",
    }
    matches: dict[str, set[Path]] = {label: set() for label in prefixes}
    for line in output.splitlines():
        match = re.match(r"^\s*(\S+)\s+=>\s+(\S+)", line)
        if match is None or match.group(2) == "not":
            continue
        soname, loaded = match.groups()
        for label, prefix in prefixes.items():
            if soname.startswith(prefix):
                matches[label].add(Path(loaded))
    ambiguous = {label: paths for label, paths in matches.items() if len(paths) > 1}
    if ambiguous:
        raise ValidationError(
            "dynamic loader reported multiple runtime libraries: "
            + "; ".join(
                f"{label}={','.join(sorted(str(path) for path in paths))}"
                for label, paths in sorted(ambiguous.items())
            )
        )
    return {label: next(iter(paths)) for label, paths in matches.items() if paths}


def _native_runtime_identity(
    executable: Path, environment: dict[str, str]
) -> dict[str, object]:
    ldd = shutil.which("ldd")
    if ldd is None:
        raise ValidationError("cannot verify native runtime closure: ldd is missing")
    identity = _command_identity(
        [ldd, str(executable)],
        timeout=TIMEOUT_SECONDS,
        environment=environment,
        output_normalizer=_normalize_dynamic_loader_output,
    )
    if not identity.get("available"):
        raise ValidationError(
            "cannot verify native runtime closure: "
            + json.dumps(identity, sort_keys=True)
        )
    libraries = _runtime_libraries_from_ldd_output(str(identity.get("output", "")))
    missing = {"hip-runtime", "hsa-runtime"} - set(libraries)
    if missing:
        raise ValidationError(
            "native runtime closure is missing " + ", ".join(sorted(missing))
        )
    return {
        "loader": identity,
        "loaded_runtime_libraries": _runtime_library_records(libraries),
    }


def _llama_runtime_identity(
    workspace: Path, target: str, workload: Workload
) -> dict[str, object]:
    runtime = _llama_runtime(workspace, target, workload.relative_path)
    ldd = shutil.which("ldd")
    if ldd is None:
        raise ValidationError("cannot verify llama runtime closure: ldd is missing")
    environment = _clean_environment(None, workload, None, target, workspace)
    try:
        completed = subprocess.run(
            [ldd, str(runtime.executable)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValidationError(
            f"cannot verify llama runtime closure with ldd: {error}"
        ) from error
    output = completed.stdout or ""
    if completed.returncode != 0:
        raise ValidationError(
            "cannot verify llama runtime closure: "
            f"ldd exited with status {completed.returncode}: {output.strip()}"
        )

    loaded_paths = []
    missing_names = []
    for line in output.splitlines():
        match = re.match(r"^\s*(\S+)\s+=>\s+(\S+)", line)
        if match is None or not match.group(1).startswith("libggml"):
            continue
        if match.group(2) == "not":
            missing_names.append(match.group(1))
            continue
        loaded_paths.append(Path(match.group(2)))
    if missing_names:
        raise ValidationError(
            "llama runtime closure has unresolved libraries: "
            + ", ".join(sorted(missing_names))
        )

    libraries = {}
    unmatched = list(loaded_paths)
    for label, expected in runtime.libraries.items():
        expected_real = expected.resolve(strict=True)
        match = next(
            (
                candidate
                for candidate in unmatched
                if candidate.resolve(strict=True) == expected_real
            ),
            None,
        )
        if match is None:
            raise ValidationError(
                "llama loader did not resolve the recorded runtime library "
                f"{label} from {expected}"
            )
        unmatched.remove(match)
        libraries[label] = {
            "recorded_path": str(expected),
            "loader_path": str(match),
            "resolved_path": str(expected_real),
        }
    if unmatched:
        raise ValidationError(
            "llama loader resolved unrecorded ggml libraries: "
            + ", ".join(str(path) for path in unmatched)
        )
    return {
        "kind": workload.kind,
        "identity_source": "validated dynamic-loader closure",
        "executable": str(runtime.executable),
        "libraries": libraries,
        "loaded_runtime_libraries": _runtime_library_records(
            _runtime_libraries_from_ldd_output(output)
        ),
    }


def _workload_runtime_identity(
    workspace: Path,
    target: str,
    workload: Workload,
    launcher: list[str] | None = None,
) -> dict[str, object]:
    if workload.kind == "llama":
        return _llama_runtime_identity(workspace, target, workload)
    if workload.kind in {"gtest", "native-executable", "rdna4-matmul"}:
        executable = _input_files(workspace, target, workload)["executable"]
        environment = _clean_environment(None, workload, None, target, workspace)
        return {
            "kind": workload.kind,
            "identity_source": "validated dynamic-loader closure",
            "executable": str(executable),
            **_native_runtime_identity(executable, environment),
        }
    if workload.kind != "pytorch":
        return {
            "kind": workload.kind,
            "identity_source": "hashed provenance files",
        }
    python = _pytorch_python(workspace)
    identity_prefix = "CONSAN_PYTORCH_RUNTIME_IDENTITY="
    script = """
import json
import pathlib

import torch
import triton

torch.ones(1, device="cuda")
torch.cuda.synchronize()
mapped = sorted({
    fields[-1]
    for line in pathlib.Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    if len(fields := line.split()) >= 6 and fields[-1].startswith("/")
})
prefixes = {
    "hip-runtime": "libamdhip64.so",
    "hsa-runtime": "libhsa-runtime64.so",
}
runtime_libraries = {
    label: [path for path in mapped if pathlib.Path(path).name.startswith(prefix)]
    for label, prefix in prefixes.items()
}
print("CONSAN_PYTORCH_RUNTIME_IDENTITY=" + json.dumps({
    "torch_version": torch.__version__,
    "torch_hip_version": torch.version.hip,
    "torch_file": torch.__file__,
    "triton_version": getattr(triton, "__version__", None),
    "triton_file": triton.__file__,
    "runtime_libraries": runtime_libraries,
}, sort_keys=True))
"""
    packages = _command_identity(
        _with_launcher(launcher or [], [str(python), "-c", script]),
        timeout=TIMEOUT_SECONDS,
        environment=_clean_environment(None, workload, None, target, workspace),
    )
    if not packages.get("available"):
        raise ValidationError(
            "cannot record required PyTorch/Triton runtime identity: "
            + json.dumps(packages, sort_keys=True)
        )
    identity_documents = [
        line.removeprefix(identity_prefix)
        for line in str(packages.get("output", "")).splitlines()
        if line.startswith(identity_prefix)
    ]
    if len(identity_documents) != 1:
        raise ValidationError(
            "PyTorch/Triton runtime identity did not emit exactly one "
            "prefixed JSON document"
        )
    try:
        package_document = json.loads(identity_documents[0])
    except json.JSONDecodeError as error:
        raise ValidationError(
            "PyTorch/Triton runtime identity did not emit valid JSON"
        ) from error
    required = {
        "torch_version",
        "torch_hip_version",
        "torch_file",
        "triton_version",
        "triton_file",
        "runtime_libraries",
    }
    if not isinstance(package_document, dict) or any(
        not package_document.get(name) for name in required
    ):
        raise ValidationError(
            "PyTorch/Triton runtime identity is missing required fields: "
            + ", ".join(
                sorted(
                    required
                    - set(
                        package_document if isinstance(package_document, dict) else ()
                    )
                )
            )
        )
    runtime_candidates = package_document["runtime_libraries"]
    if not isinstance(runtime_candidates, dict):
        raise ValidationError("PyTorch runtime identity has invalid library closure")
    runtime_paths = {}
    for label in ("hip-runtime", "hsa-runtime"):
        candidates = runtime_candidates.get(label)
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValidationError(
                f"PyTorch runtime identity needs exactly one loaded {label}: "
                f"{candidates!r}"
            )
        runtime_paths[label] = Path(candidates[0])
    return {
        "kind": workload.kind,
        "identity_source": "framework probe and loaded process mappings",
        "python_packages": packages,
        "package_document": package_document,
        "loaded_runtime_libraries": _runtime_library_records(runtime_paths),
    }


def _gfx_target_version(target: str) -> int | None:
    match = re.fullmatch(r"gfx([0-9]{2})([0-9])([0-9])", target)
    if match is None:
        return None
    major, minor, stepping = (int(value) for value in match.groups())
    return major * 10000 + minor * 100 + stepping


def _machine_identity(target: str) -> dict[str, object]:
    topology = {}
    topology_root = Path("/sys/class/kfd/kfd/topology/nodes")
    if topology_root.is_dir():
        for node in sorted(topology_root.iterdir(), key=lambda path: path.name):
            if not node.is_dir():
                continue
            topology[node.name] = {
                name: _read_identity_file(node / name)
                for name in ("gpu_id", "name", "properties")
            }
    pci_devices = {}
    for device in sorted(Path("/sys/bus/pci/devices").glob("*")):
        vendor = _read_identity_file(device / "vendor")
        if vendor.get("value") != "0x1002":
            continue
        pci_devices[device.name] = {
            name: _read_identity_file(device / name)
            for name in (
                "vendor",
                "device",
                "subsystem_vendor",
                "subsystem_device",
                "revision",
            )
        }
        driver = device / "driver"
        try:
            pci_devices[device.name]["driver"] = driver.resolve().name
        except OSError:
            pci_devices[device.name]["driver"] = None
    target_version = _gfx_target_version(target)
    selected_kfd_nodes = []
    if target_version is not None:
        marker = f"gfx_target_version {target_version}"
        selected_kfd_nodes = [
            node
            for node, identity in topology.items()
            if marker in str(identity["properties"].get("value", "")).splitlines()
        ]
    return {
        "uname": platform.uname()._asdict(),
        "os_release": _read_identity_file(Path("/etc/os-release")),
        "kfd_topology": topology,
        "selected_kfd_nodes": selected_kfd_nodes,
        "amd_pci_devices": pci_devices,
        "amdgpu_module_version": _read_identity_file(
            Path("/sys/module/amdgpu/version")
        ),
        "amdgpu_module_source_version": _read_identity_file(
            Path("/sys/module/amdgpu/srcversion")
        ),
    }


def _unavailable_command_identity(name: str) -> dict[str, object]:
    return {"available": False, "command": [name], "reason": "tool is unavailable"}


def _empirical_observation_snapshot() -> dict[str, object]:
    rocm_smi = shutil.which("rocm-smi")
    amd_smi = shutil.which("amd-smi")
    return {
        "schema_version": 1,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clocks_temperature_and_utilization": (
            _command_identity(
                [
                    rocm_smi,
                    "--showtemp",
                    "--showclocks",
                    "--showuse",
                    "--json",
                ]
            )
            if rocm_smi
            else _unavailable_command_identity("rocm-smi")
        ),
        "competing_gpu_processes": (
            _command_identity([amd_smi, "process", "--json"])
            if amd_smi
            else _unavailable_command_identity("amd-smi")
        ),
        "firmware": (
            _command_identity([amd_smi, "firmware", "--json"])
            if amd_smi
            else _unavailable_command_identity("amd-smi")
        ),
        "amd_smi_metrics": (
            _command_identity([amd_smi, "metric", "--json"])
            if amd_smi
            else _unavailable_command_identity("amd-smi")
        ),
    }


def _runtime_tool_identities(
    llvm_readelf: Path | None, hook: Path
) -> dict[str, object]:
    commands = {
        "python": [sys.executable, "--version"],
        "rocminfo": [shutil.which("rocminfo") or "rocminfo"],
    }
    if llvm_readelf is not None:
        commands["llvm-readelf"] = [str(llvm_readelf), "--version"]
    rocm_sdk = shutil.which("rocm-sdk")
    if rocm_sdk:
        commands["rocm-sdk"] = [rocm_sdk, "path", "--root"]
    amdclang = shutil.which("amdclang++")
    if amdclang:
        commands["amdclang++"] = [amdclang, "--version"]
    identities = {
        name: _command_identity(command) for name, command in commands.items()
    }
    identities["hook-linkage"] = _command_identity(
        [shutil.which("ldd") or "ldd", str(hook)],
        output_normalizer=_normalize_dynamic_loader_output,
    )
    return identities


def _source_identities(workspace: Path, workload: Workload) -> list[dict | None]:
    roots = [
        workspace / "iree-test-suites",
        workspace / "hip-moi",
        Path(__file__).resolve().parents[5],
    ]
    if workload.corpus == "rocjitsu-test-corpus":
        roots.append(workspace / "rocjitsu-test-corpus")
    if workload.kind == "tensile":
        paths = resolve_tensile_validation_paths(workspace)
        roots.append(paths.tensilelite)
    if workload.kind == "llama":
        roots.append(workspace / "rocjitsu-test-corpus")
    if workload.kind == "rdna4-matmul":
        roots.append(_rdna4_matmul_root(workspace))
    if workload.kind == "pytorch":
        roots.append(workspace / "pytorch")
    return [git_identity(root) for root in roots]




def _launcher_from_json(value: str | None) -> list[str]:
    if value is None:
        return []
    try:
        launcher = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValidationError(f"invalid --launcher-json: {error}") from error
    if (
        not isinstance(launcher, list)
        or not launcher
        or any(not isinstance(item, str) or not item for item in launcher)
    ):
        raise ValidationError(
            "--launcher-json must be a nonempty JSON array of nonempty strings"
        )
    return launcher


def _launcher_argument(value: str) -> list[str]:
    try:
        return _launcher_from_json(value)
    except ValidationError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _with_launcher(launcher: list[str], command: list[str]) -> list[str]:
    return [*launcher, *command]


def _target_outer_repetitions(target: str, phase: str, workload: Workload) -> int:
    if target in SINGLE_REPETITION_TARGETS or phase != "overhead":
        return 1
    return workload.overhead_processes


def _outer_repetitions(target: str, phase: str, workload: Workload) -> int:
    return _target_outer_repetitions(
        target, phase, _resolved_workload(target, workload)
    )


def _effective_workload(target: str, workload: Workload) -> Workload:
    resolved = _resolved_workload(target, workload)
    return replace(
        resolved,
        fault_families=_target_fault_families(target, resolved),
        overhead_processes=_target_outer_repetitions(target, "overhead", resolved),
    )


def _workload_provenance_path(artifact_root: Path, workload: Workload) -> Path:
    return artifact_root / workload.id / "provenance.json"
