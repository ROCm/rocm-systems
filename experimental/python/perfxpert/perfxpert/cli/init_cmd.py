###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################
"""perfxpert init — first-run wizard.

Chains four pre-existing building blocks into a single guided flow:

    1. GPU detection    (rocm-smi --json / rocminfo fallback / --arch override)
    2. Framework detect (Tier-0 source scan + Python env probe)
    3. Config generation (~/.config/perfxpert/config.yaml via perfxpert.config)
    4. Suggested first rocprofv3 command (cost-ordered ladder)

No new analysis logic; this module just composes existing primitives:

* ``perfxpert.tools.arch.lookup_peaks``
* ``perfxpert.analysis.payload.scan_tier0_sources``
* ``perfxpert.tools.profiling`` (cost ladder, consulted for messaging)
* ``perfxpert.config`` (YAML read/merge/write + Pydantic validation)

Public entry point: :func:`run_init`.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from perfxpert.analysis.payload import scan_tier0_sources
from perfxpert.config._cli import _config_path as _default_config_path
from perfxpert.config._config import PerfXpertConfig
from perfxpert.tools.arch import lookup_peaks


__all__ = ["add_args", "run_init"]


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def add_args(parser: argparse.ArgumentParser) -> None:
    """Register flags for `perfxpert init` on ``parser``."""
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
        help="Preselect LLM provider. Default: probe env vars and pick the first configured.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        metavar="GFXID",
        help="Override GPU architecture id (e.g. gfx942) when rocm-smi/rocminfo unavailable.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts; use detected defaults (CI / scripting).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Override config file location (default: ~/.config/perfxpert/config.yaml).",
    )


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

_GFX_RE = re.compile(r"\b(gfx\d{3,4}[a-z]?)\b")


def _detect_gpu_via_rocm_smi() -> Optional[str]:
    """Run rocm-smi and return the first gfx arch string, or None."""
    if shutil.which("rocm-smi") is None:
        return None
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    # rocm-smi --json shape varies across ROCm versions; probe for gfx token
    text = proc.stdout or ""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        m = _GFX_RE.search(text)
        return m.group(1) if m else None
    # Walk the JSON tree looking for any value that matches a gfx string.
    def _walk(obj: Any) -> Optional[str]:
        if isinstance(obj, str):
            m = _GFX_RE.search(obj)
            return m.group(1) if m else None
        if isinstance(obj, dict):
            for v in obj.values():
                hit = _walk(v)
                if hit:
                    return hit
        if isinstance(obj, list):
            for v in obj:
                hit = _walk(v)
                if hit:
                    return hit
        return None
    return _walk(data)


def _detect_gpu_via_rocminfo() -> Optional[str]:
    """Fallback: parse rocminfo text output for a gfx arch."""
    if shutil.which("rocminfo") is None:
        return None
    try:
        proc = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    m = _GFX_RE.search(proc.stdout or "")
    return m.group(1) if m else None


def _detect_gpu(override: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Resolve a gfx id (override > rocm-smi > rocminfo) and look up peaks.

    Returns a dict with ``{gfx_id, peaks}`` on success, or ``None`` when no
    source could produce a gfx id.
    """
    gfx = override or _detect_gpu_via_rocm_smi() or _detect_gpu_via_rocminfo()
    if not gfx:
        return None
    try:
        peaks = lookup_peaks(gfx)
    except KeyError:
        return {"gfx_id": gfx, "peaks": None}
    return {"gfx_id": gfx, "peaks": peaks}


# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------

_PY_FRAMEWORKS = ("torch", "tensorflow", "jax", "cupy")


def _detect_python_framework() -> Optional[str]:
    """Return the first Python ML framework importable in this env, or None."""
    for name in _PY_FRAMEWORKS:
        try:
            spec = importlib.util.find_spec(name)
        except (ValueError, ModuleNotFoundError):
            spec = None
        if spec is not None:
            # Normalize names for display
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
    """Combine source-scan programming model with Python env probe.

    Returns a dict ``{source_dir, python_framework, programming_model,
    kernel_count, file_count, suggested_first_command, suggested_counters}``.
    """
    # Tier-0 source scan — reuses its own .git/node_modules filters.
    tier0 = scan_tier0_sources(source_dir) or {}
    programming_model = tier0.get("programming_model", "Unknown")
    kernel_count = int(tier0.get("kernel_count", 0) or 0)
    file_count = int(tier0.get("files_scanned", 0) or 0)
    suggested_cmd = tier0.get("suggested_first_command") or ""
    suggested_counters = list(tier0.get("suggested_counters") or [])

    py_fw = _detect_python_framework()

    return {
        "source_dir": source_dir,
        "python_framework": py_fw,
        "programming_model": programming_model,
        "kernel_count": kernel_count,
        "file_count": file_count,
        "suggested_first_command": suggested_cmd,
        "suggested_counters": suggested_counters,
    }


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

_PROVIDER_ENV_ORDER = [
    ("anthropic", ("ANTHROPIC_API_KEY", "PERFXPERT_LLM_ANTHROPIC_KEY")),
    ("openai", ("OPENAI_API_KEY", "PERFXPERT_LLM_OPENAI_KEY")),
    ("private", ("PERFXPERT_LLM_PRIVATE_URL", "PRIVATE_LLM_ENDPOINT")),
    ("ollama", ("OLLAMA_HOST",)),
]


def _detect_configured_provider() -> str:
    """Return the first provider with a populated env var, else 'opencode'."""
    for name, env_keys in _PROVIDER_ENV_ORDER:
        for key in env_keys:
            if os.environ.get(key):
                return name
    return "opencode"


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def _build_config(provider: str) -> Dict[str, Any]:
    """Produce a config dict matching the perfxpert.config schema.

    Uses the Pydantic model's defaults for every field except ``provider``
    and the airgap implication. The model field is left unset so the
    provider's default applies at resolve time (see `perfxpert doctor`).
    """
    # Validate the chosen provider is a real enum value.
    valid = {"anthropic", "openai", "ollama", "private", "opencode"}
    if provider not in valid:
        provider = "opencode"

    # Round-trip through the Pydantic model so defaults are canonical.
    cfg = PerfXpertConfig(provider=provider)  # type: ignore[arg-type]
    data = cfg.model_dump()
    # Strip None model so YAML stays compact and provider default applies.
    if data.get("model") is None:
        data.pop("model", None)
    return data


def _resolve_config_path(override: Optional[str]) -> Path:
    if override:
        return Path(override).expanduser()
    return _default_config_path()


def _write_config(
    config: Dict[str, Any],
    target: Path,
    *,
    non_interactive: bool,
    stream=sys.stdout,
) -> bool:
    """Atomically write ``config`` to ``target``. Returns True if written.

    When the file exists and the session is interactive, prints a unified
    diff and asks y/N before overwriting. Non-interactive always overwrites.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    new_text = yaml.safe_dump(config, sort_keys=True)

    if target.exists():
        existing = target.read_text()
        if existing.strip() == new_text.strip():
            print(f"  (no change — {target} already matches)", file=stream)
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
                resp = input("  overwrite? [y/N] ").strip().lower()
            except EOFError:
                resp = ""
            if resp not in ("y", "yes"):
                print("  keeping existing config.", file=stream)
                return False

    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(new_text)
    os.replace(tmp, target)
    return True


# ---------------------------------------------------------------------------
# Suggested-first-command composition
# ---------------------------------------------------------------------------

def _suggest_first_command(framework_info: Dict[str, Any]) -> List[str]:
    """Return one or two commands (cost-ladder lowest rung: --sys-trace).

    * Default: ``rocprofv3 --sys-trace -d ./profile_out -- ./your_app``
    * PyTorch / TF / JAX detected: run against ``python train.py``
    * HIP-kernels in source: add an auxiliary --pmc line
    """
    cmds: List[str] = []
    py_fw = framework_info.get("python_framework")
    target = "./your_app"
    if py_fw == "PyTorch":
        target = "python train.py"
    elif py_fw == "TensorFlow":
        target = "python train.py"
    elif py_fw == "JAX":
        target = "python train.py"
    elif py_fw == "CuPy":
        target = "python run.py"

    cmds.append(f"rocprofv3 --sys-trace -d ./profile_out -- {target}")

    if framework_info.get("kernel_count", 0) > 0:
        counters = framework_info.get("suggested_counters") or [
            "SQ_WAVES",
            "GRBM_COUNT",
            "GRBM_GUI_ACTIVE",
        ]
        cmds.append(
            "rocprofv3 --pmc "
            + " ".join(counters)
            + f" -d ./profile_out_pmc -- {target}"
        )
    return cmds


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _print_header(title: str, stream=sys.stdout) -> None:
    print(f"== {title} ==", file=stream)
    print(file=stream)


def _print_step(n: int, total: int, title: str, body: str, stream=sys.stdout) -> None:
    print(f"Step {n}/{total} — {title}", file=stream)
    for line in body.splitlines():
        print(f"  {line}", file=stream)
    print(file=stream)


def _format_gpu_info(gpu_info: Optional[Dict[str, Any]]) -> str:
    if gpu_info is None:
        return (
            "could not detect GPU via rocm-smi or rocminfo.\n"
            "pass `--arch gfx942` (or similar) to proceed manually."
        )
    gfx = gpu_info.get("gfx_id", "unknown")
    peaks = gpu_info.get("peaks") or {}
    if not peaks:
        return f"detected: {gfx} (no peak spec available — check perfxpert/knowledge/gpu_specs.yaml)"
    name = peaks.get("name", "?")
    cu = peaks.get("cu_count", "?")
    bw = peaks.get("memory_bandwidth_tbs", "?")
    fp32 = peaks.get("peak_fp32_tflops", "?")
    return (
        f"detected: {gfx} ({name}, {cu} CU)\n"
        f"  peak FP32: {fp32} TFLOPS\n"
        f"  peak HBM : {bw} TB/s"
    )


def _format_framework_info(info: Dict[str, Any]) -> str:
    src = info.get("source_dir") or "."
    py_fw = info.get("python_framework")
    pm = info.get("programming_model") or "Unknown"
    kc = info.get("kernel_count", 0)
    fc = info.get("file_count", 0)

    lines = [f"source dir: {src}"]
    if py_fw:
        lines.append(f"python deps: {py_fw} importable ⇒ framework: {py_fw}")
    else:
        lines.append("python deps: no ML framework importable")
    lines.append(
        f"source scan: {fc} file(s), {kc} kernel(s) ⇒ programming model: {pm}"
    )
    if py_fw and pm in ("HIP", "OpenCL"):
        lines.append("(both detected ⇒ mixed python+kernel workload)")
    return "\n".join(lines)


def _format_config(cfg: Dict[str, Any], path: Path) -> str:
    provider = cfg.get("provider", "?")
    airgap = cfg.get("airgap", False)
    return (
        f"provider  : {provider}\n"
        f"airgap    : {str(airgap).lower()}\n"
        f"max_tokens: {cfg.get('max_tokens', '?')}\n"
        f"target    : {path}"
    )


def _format_suggested_cmds(cmds: List[str]) -> str:
    lines = []
    for i, c in enumerate(cmds):
        prefix = "primary : " if i == 0 else "extra   : "
        lines.append(prefix + c)
    lines.append(
        "(--pc-sampling / --att are second-tier — run after Tier-1 "
        "identifies hot kernels)"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_init(args) -> int:
    """Execute the first-run wizard. Returns 0 on success, non-zero on failure."""
    stream = sys.stdout
    _print_header("perfxpert init — first-run wizard", stream=stream)

    # --- Step 1 -------------------------------------------------------------
    gpu_info = _detect_gpu(override=getattr(args, "arch", None))
    _print_step(1, 4, "GPU detection", _format_gpu_info(gpu_info), stream=stream)

    # --- Step 2 -------------------------------------------------------------
    source_dir = getattr(args, "source_dir", None) or "."
    framework_info = _detect_framework(source_dir)
    _print_step(2, 4, "Framework detection", _format_framework_info(framework_info), stream=stream)

    # --- Step 3 -------------------------------------------------------------
    provider = getattr(args, "provider", None) or _detect_configured_provider()
    try:
        config = _build_config(provider)
    except Exception as e:  # pragma: no cover - defensive
        print(f"✗ config build failed: {e}", file=sys.stderr)
        return 1

    target = _resolve_config_path(getattr(args, "config_path", None))
    non_interactive = bool(getattr(args, "non_interactive", False))
    try:
        _write_config(config, target, non_interactive=non_interactive, stream=stream)
    except OSError as e:
        print(f"✗ could not write config to {target}: {e}", file=sys.stderr)
        return 1
    _print_step(3, 4, "Config generation", _format_config(config, target), stream=stream)

    # --- Step 4 -------------------------------------------------------------
    cmds = _suggest_first_command(framework_info)
    _print_step(4, 4, "Suggested first profiling command",
                _format_suggested_cmds(cmds), stream=stream)

    # Next-step breadcrumb.
    print("Then:", file=stream)
    print("  perfxpert analyze -i ./profile_out/*.db --source-dir . "
          "--format webview -o report", file=stream)
    print(file=stream)
    print("Wizard complete. Your setup is ready.", file=stream)
    return 0
