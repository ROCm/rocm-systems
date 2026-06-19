###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################
"""perfxpert init - first-run wizard."""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from perfxpert.analysis.payload import scan_tier0_sources
from perfxpert.config._cli import _config_path as _default_config_path
from perfxpert.config._config import PerfXpertConfig
from perfxpert.tools.arch import lookup_peaks
from perfxpert.tools.gpu_discovery import first_runtime_gpu_specs


__all__ = ["add_args", "run_init"]


def add_args(parser: argparse.ArgumentParser) -> None:
    """Register flags for ``perfxpert init``."""
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Source tree to scan for framework detection (default: current directory).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["anthropic", "openai", "ollama", "private", "opencode"],
        help="Preselect the LLM provider. Default: detect from environment.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        metavar="GFXID",
        help="Override GPU architecture id (for example: gfx942).",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts and use detected defaults (CI / scripting).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Override config file location (default: ~/.config/perfxpert/config.yaml).",
    )


def _detect_gpu(override: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Resolve a local GPU and look up runtime-enriched peak specs."""
    if override:
        gfx_id = override
    else:
        detected = first_runtime_gpu_specs()
        gfx_id = detected.get("gfx_id") if detected else None
    if not gfx_id:
        return None
    try:
        peaks = lookup_peaks(gfx_id, prefer_runtime=True)
    except KeyError:
        peaks = None
    return {"gfx_id": gfx_id, "peaks": peaks}


_PY_FRAMEWORKS = ("torch", "tensorflow", "jax", "cupy")


def _detect_python_framework() -> Optional[str]:
    """Return the first Python ML framework importable in this env."""
    for name in _PY_FRAMEWORKS:
        try:
            spec = importlib.util.find_spec(name)
        except (ModuleNotFoundError, ValueError):
            spec = None
        if spec is None:
            continue
        if name == "torch":
            return "PyTorch"
        if name == "tensorflow":
            return "TensorFlow"
        if name == "jax":
            return "JAX"
        if name == "cupy":
            return "CuPy"
    return None


def _detect_framework(source_dir: str) -> Dict[str, Any]:
    """Combine source-scan programming model with Python env probe."""
    tier0 = scan_tier0_sources(source_dir) or {}
    return {
        "source_dir": source_dir,
        "python_framework": _detect_python_framework(),
        "programming_model": tier0.get("programming_model", "Unknown"),
        "kernel_count": int(tier0.get("kernel_count", 0) or 0),
        "file_count": int(tier0.get("files_scanned", 0) or 0),
        "suggested_first_command": tier0.get("suggested_first_command") or "",
        "suggested_counters": list(tier0.get("suggested_counters") or []),
    }


_PROVIDER_ENV_ORDER = [
    ("anthropic", ("ANTHROPIC_API_KEY", "PERFXPERT_LLM_ANTHROPIC_KEY")),
    ("openai", ("OPENAI_API_KEY", "PERFXPERT_LLM_OPENAI_KEY")),
    ("private", ("PERFXPERT_LLM_PRIVATE_URL", "PRIVATE_LLM_ENDPOINT")),
    ("ollama", ("PERFXPERT_LLM_LOCAL_URL", "OLLAMA_HOST")),
]


def _detect_configured_provider() -> str:
    """Return the first provider with credentials, else opencode."""
    for name, env_keys in _PROVIDER_ENV_ORDER:
        for key in env_keys:
            if os.environ.get(key):
                return name
    return "opencode"


def _build_config(provider: str) -> Dict[str, Any]:
    """Produce a config dict matching the perfxpert config schema."""
    valid = {"anthropic", "openai", "ollama", "private", "opencode"}
    if provider not in valid:
        provider = "opencode"

    cfg = PerfXpertConfig(provider=provider)  # type: ignore[arg-type]
    data = cfg.model_dump()
    if data.get("model") is None:
        data.pop("model", None)
    return data


def _resolve_config_path(override: Optional[str]) -> Path:
    if override:
        return Path(override).expanduser()
    path = _default_config_path()
    if path is None:
        raise RuntimeError("cannot resolve perfxpert config path")
    return path


def _write_config(
    config: Dict[str, Any],
    target: Path,
    *,
    non_interactive: bool,
    stream=sys.stdout,
) -> bool:
    """Atomically write config to target and return True when changed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    new_text = yaml.safe_dump(config, sort_keys=True)

    if target.exists():
        existing = target.read_text()
        if existing.strip() == new_text.strip():
            print(f"  (no change - {target} already matches)", file=stream)
            return False
        if not non_interactive:
            diff = difflib.unified_diff(
                existing.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=str(target),
                tofile="<new>",
            )
            print("  existing config differs; diff:", file=stream)
            for line in diff:
                print("    " + line.rstrip(), file=stream)
            try:
                response = input("  overwrite? [y/N] ").strip().lower()
            except EOFError:
                response = ""
            if response not in ("y", "yes"):
                print("  keeping existing config.", file=stream)
                return False

    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(new_text)
    os.replace(tmp, target)
    return True


def _suggest_first_command(framework_info: Dict[str, Any]) -> List[str]:
    """Return one or two suggested rocprofv3 commands."""
    commands: List[str] = []
    python_framework = framework_info.get("python_framework")
    target = "./your_app"
    if python_framework in {"PyTorch", "TensorFlow", "JAX"}:
        target = "python train.py"
    elif python_framework == "CuPy":
        target = "python run.py"

    commands.append(f"rocprofv3 --sys-trace -d ./profile_out -- {target}")

    if framework_info.get("kernel_count", 0) > 0:
        counters = framework_info.get("suggested_counters") or [
            "SQ_WAVES",
            "GRBM_COUNT",
            "GRBM_GUI_ACTIVE",
        ]
        commands.append("rocprofv3 --pmc " + " ".join(counters) + f" -d ./profile_out_pmc -- {target}")
    return commands


def _print_header(title: str, stream=sys.stdout) -> None:
    print(f"== {title} ==", file=stream)
    print(file=stream)


def _print_step(n: int, total: int, title: str, body: str, stream=sys.stdout) -> None:
    print(f"Step {n}/{total} - {title}", file=stream)
    for line in body.splitlines():
        print(f"  {line}", file=stream)
    print(file=stream)


def _format_gpu_info(gpu_info: Optional[Dict[str, Any]]) -> str:
    if gpu_info is None:
        return (
            "could not detect GPU via rocminfo, rocm-smi, or amd-smi.\n"
            "pass `--arch gfx942` (or similar) to proceed manually."
        )
    gfx_id = gpu_info.get("gfx_id", "unknown")
    peaks = gpu_info.get("peaks") or {}
    if not peaks:
        return f"detected: {gfx_id} (no peak spec available)"
    name = peaks.get("name", "?")
    cu_count = peaks.get("cu_count", "?")
    bandwidth = peaks.get("memory_bandwidth_tbs", "?")
    fp32 = peaks.get("peak_fp32_tflops", "?")
    return (
        f"detected: {gfx_id} ({name}, {cu_count} CU)\n" f"  peak FP32: {fp32} TFLOPS\n" f"  peak HBM : {bandwidth} TB/s"
    )


def _format_framework_info(info: Dict[str, Any]) -> str:
    lines = [f"source dir: {info.get('source_dir') or '.'}"]
    python_framework = info.get("python_framework")
    if python_framework:
        lines.append(f"python deps: {python_framework} importable")
    else:
        lines.append("python deps: no ML framework importable")
    lines.append(
        "source scan: "
        f"{info.get('file_count', 0)} file(s), "
        f"{info.get('kernel_count', 0)} kernel(s) => programming model: "
        f"{info.get('programming_model') or 'Unknown'}"
    )
    if python_framework and info.get("programming_model") in {"HIP", "OpenCL"}:
        lines.append("(mixed python + kernel workload detected)")
    return "\n".join(lines)


def _format_config(cfg: Dict[str, Any], path: Path) -> str:
    return (
        f"provider  : {cfg.get('provider', '?')}\n"
        f"airgap    : {str(cfg.get('airgap', False)).lower()}\n"
        f"max_tokens: {cfg.get('max_tokens', '?')}\n"
        f"target    : {path}"
    )


def _format_suggested_cmds(commands: List[str]) -> str:
    lines = []
    for index, command in enumerate(commands):
        prefix = "primary : " if index == 0 else "extra   : "
        lines.append(prefix + command)
    lines.append("(--pc-sampling / --att are second-tier - run after Tier-1 identifies hot kernels)")
    return "\n".join(lines)


def run_init(args) -> int:
    """Execute the first-run wizard."""
    stream = sys.stdout
    _print_header("perfxpert init - first-run wizard", stream=stream)

    gpu_info = _detect_gpu(override=getattr(args, "arch", None))
    _print_step(1, 4, "GPU detection", _format_gpu_info(gpu_info), stream=stream)

    source_dir = getattr(args, "source_dir", None) or "."
    framework_info = _detect_framework(source_dir)
    _print_step(2, 4, "Framework detection", _format_framework_info(framework_info), stream=stream)

    provider = getattr(args, "provider", None) or _detect_configured_provider()
    try:
        config = _build_config(provider)
        target = _resolve_config_path(getattr(args, "config_path", None))
    except Exception as exc:
        print(f"error: could not prepare config path: {exc}", file=sys.stderr)
        return 1

    try:
        _write_config(
            config,
            target,
            non_interactive=bool(getattr(args, "non_interactive", False)),
            stream=stream,
        )
    except OSError as exc:
        print(f"error: could not write config to {target}: {exc}", file=sys.stderr)
        return 1

    _print_step(3, 4, "Config generation", _format_config(config, target), stream=stream)

    commands = _suggest_first_command(framework_info)
    _print_step(4, 4, "Suggested first profiling command", _format_suggested_cmds(commands), stream=stream)

    print("Then:", file=stream)
    print(
        "  perfxpert analyze -i ./profile_out/*.db --source-dir . --format webview -o report",
        file=stream,
    )
    print(file=stream)
    print("Wizard complete. Your setup is ready.", file=stream)
    return 0
