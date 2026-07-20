#!/usr/bin/env python3
"""Preflight replay compatibility using HRR manifest metadata (PR #8680)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CaptureMetadata:
    schema_version: int | None = None
    hip_runtime_version: str | None = None
    comgr_version: str | None = None
    device_count: int | None = None
    captured_device_count: int | None = None
    devices: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] | None = None


@dataclass
class ReplayEnvironment:
    visible_gpus: int | None = None
    gpu_archs: list[str] = field(default_factory=list)
    gpu_names: list[str] = field(default_factory=list)
    hip_runtime_version: str | None = None
    comgr_version: str | None = None
    source: str = "host"


@dataclass
class CompatReport:
    ok: bool = True
    blocks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    capture: CaptureMetadata | None = None
    replay: ReplayEnvironment | None = None


def _nested_get(obj: dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def load_capture_metadata(archive: Path) -> CaptureMetadata | None:
    manifest = archive / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        return None

    runtime = meta.get("runtime")
    hip_ver = None
    comgr_ver = None
    if isinstance(runtime, dict):
        hip_ver = runtime.get("hip_runtime_version")
        comgr_ver = runtime.get("comgr_version")
    if hip_ver is None:
        hip_ver = meta.get("hip_runtime_version")
    if comgr_ver is None:
        comgr_ver = meta.get("comgr_version")

    devices: list[dict[str, Any]] = []
    raw_devices = meta.get("devices")
    if isinstance(raw_devices, list):
        for item in raw_devices:
            if isinstance(item, dict):
                devices.append(item)

    return CaptureMetadata(
        schema_version=meta.get("schema_version"),
        hip_runtime_version=str(hip_ver) if hip_ver else None,
        comgr_version=str(comgr_ver) if comgr_ver else None,
        device_count=_as_int(meta.get("device_count")),
        captured_device_count=_as_int(meta.get("captured_device_count")),
        devices=devices,
        raw=meta,
    )


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _normalize_arch(name: str | None) -> str | None:
    if not name:
        return None
    base = name.split(":", 1)[0].strip().lower()
    return base or None


def _run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_rocm_smi(text: str) -> tuple[int, list[str], list[str]]:
    gpu_ids = sorted({int(m.group(1)) for m in re.finditer(r"GPU\[(\d+)\]", text)})
    archs: list[str] = []
    names: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Card series:") or line.startswith("Card Series:"):
            names.append(line.split(":", 1)[1].strip())
        if "gfx" in line.lower():
            m = re.search(r"(gfx[0-9a-zA-Z]+)", line)
            if m:
                archs.append(m.group(1).lower())
    count = len(gpu_ids) if gpu_ids else max(len(names), len(archs))
    return count, archs, names


def probe_host_replay_env() -> ReplayEnvironment:
    env = ReplayEnvironment(source="host")
    if shutil.which("rocm-smi"):
        proc = _run(["rocm-smi", "--showid", "--showproductname"])
        out = proc.stdout + proc.stderr
        count, archs, names = _parse_rocm_smi(out)
        if count > 0:
            env.visible_gpus = count
            env.gpu_archs = archs
            env.gpu_names = names
    if env.visible_gpus is None and Path("/dev/kfd").exists():
        render_nodes = sorted(Path("/dev/dri").glob("renderD*"))
        if render_nodes:
            env.visible_gpus = len(render_nodes)
    if shutil.which("hipconfig"):
        proc = _run(["hipconfig", "--version"])
        for line in (proc.stdout + proc.stderr).splitlines():
            m = re.search(r"HIP version\s*:\s*([0-9.]+)", line, re.I)
            if m:
                env.hip_runtime_version = m.group(1)
                break
    return env


def probe_docker_replay_env(image: str) -> ReplayEnvironment:
    env = ReplayEnvironment(source=f"docker:{image}")
    if not shutil.which("docker"):
        env.source = "docker:unavailable"
        return env
    cmd = [
        "docker",
        "run",
        "--rm",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        image,
        "bash",
        "-lc",
        "rocm-smi --showid --showproductname 2>/dev/null || true; "
        "hipconfig --version 2>/dev/null || true",
    ]
    proc = _run(cmd, timeout=120)
    out = proc.stdout + proc.stderr
    count, archs, names = _parse_rocm_smi(out)
    if count > 0:
        env.visible_gpus = count
        env.gpu_archs = archs
        env.gpu_names = names
    for line in out.splitlines():
        m = re.search(r"HIP version\s*:\s*([0-9.]+)", line, re.I)
        if m:
            env.hip_runtime_version = m.group(1)
            break
    return env


def evaluate_compat(
    capture: CaptureMetadata,
    replay: ReplayEnvironment,
    *,
    gpu: int = 0,
    strict_version: bool = False,
    strict_arch: bool = False,
) -> CompatReport:
    report = CompatReport(capture=capture, replay=replay)

    if capture.device_count is not None and replay.visible_gpus is not None:
        if capture.device_count > replay.visible_gpus:
            report.blocks.append(
                f"capture saw {capture.device_count} GPU(s) but replay environment "
                f"only exposes {replay.visible_gpus}"
            )
        if gpu >= replay.visible_gpus:
            report.blocks.append(
                f"requested replay GPU {gpu} but replay environment only has "
                f"{replay.visible_gpus} GPU(s) (0..{replay.visible_gpus - 1})"
            )

    capture_archs = []
    for dev in capture.devices:
        props = dev.get("properties")
        if isinstance(props, dict):
            arch = _normalize_arch(str(props.get("gcn_arch_name", "")))
        else:
            arch = _normalize_arch(str(dev.get("gcn_arch_name", "")))
        if arch:
            capture_archs.append(arch)

    if capture_archs and replay.gpu_archs:
        replay_arch = replay.gpu_archs[min(gpu, len(replay.gpu_archs) - 1)]
        expected = capture_archs[min(gpu, len(capture_archs) - 1)]
        if expected and replay_arch and expected != replay_arch:
            msg = (
                f"GPU architecture mismatch: capture GPU {gpu} is {expected}, "
                f"replay GPU {gpu} looks like {replay_arch}"
            )
            if strict_arch:
                report.blocks.append(msg)
            else:
                report.warnings.append(msg)

    if capture.hip_runtime_version and replay.hip_runtime_version:
        if capture.hip_runtime_version != replay.hip_runtime_version:
            msg = (
                "HIP runtime version mismatch: capture "
                f"{capture.hip_runtime_version} vs replay {replay.hip_runtime_version}"
            )
            if strict_version:
                report.blocks.append(msg)
            else:
                report.warnings.append(msg)

    if capture.comgr_version and replay.comgr_version:
        if capture.comgr_version != replay.comgr_version:
            report.warnings.append(
                "comgr version mismatch: capture "
                f"{capture.comgr_version} vs replay {replay.comgr_version}"
            )

    report.ok = not report.blocks
    return report


def render_report(report: CompatReport) -> str:
    lines = ["# HRR replay compatibility"]
    if report.capture:
        c = report.capture
        lines.extend(
            [
                "",
                "## Capture metadata",
                f"- schema_version: {c.schema_version or 'n/a'}",
                f"- hip_runtime_version: {c.hip_runtime_version or 'n/a'}",
                f"- comgr_version: {c.comgr_version or 'n/a'}",
                f"- device_count: {c.device_count if c.device_count is not None else 'n/a'}",
                f"- captured_device_count: "
                f"{c.captured_device_count if c.captured_device_count is not None else 'n/a'}",
            ]
        )
    if report.replay:
        r = report.replay
        lines.extend(
            [
                "",
                "## Replay environment",
                f"- source: {r.source}",
                f"- visible_gpus: {r.visible_gpus if r.visible_gpus is not None else 'unknown'}",
                f"- hip_runtime_version: {r.hip_runtime_version or 'n/a'}",
            ]
        )
        if r.gpu_archs:
            lines.append(f"- gpu_archs: {', '.join(r.gpu_archs)}")
    if report.blocks:
        lines.extend(["", "## Blocking issues"])
        lines.extend(f"- {b}" for b in report.blocks)
    if report.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {w}" for w in report.warnings)
    if report.ok and not report.warnings:
        lines.extend(["", "Replay preflight: OK"])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, help="pid-* archive directory")
    ap.add_argument("--mode", choices=("host", "docker"), default="host")
    ap.add_argument("--docker-image", help="Capture/replay Docker image")
    ap.add_argument("--gpu", type=int, default=0, help="Replay GPU ordinal")
    ap.add_argument("--strict-version", action="store_true")
    ap.add_argument("--strict-arch", action="store_true")
    args = ap.parse_args()

    archive = Path(args.archive)
    capture = load_capture_metadata(archive)
    if capture is None:
        print(
            "[compat] no manifest metadata (legacy capture); skipping preflight",
            file=sys.stderr,
        )
        return 0

    if args.mode == "docker":
        if not args.docker_image:
            print(
                "error: --docker-image required for docker preflight", file=sys.stderr
            )
            return 2
        replay = probe_docker_replay_env(args.docker_image)
    else:
        replay = probe_host_replay_env()

    if replay.visible_gpus is None:
        print(
            "[compat] could not determine replay GPU count; continuing with warnings",
            file=sys.stderr,
        )

    report = evaluate_compat(
        capture,
        replay,
        gpu=args.gpu,
        strict_version=args.strict_version,
        strict_arch=args.strict_arch,
    )
    print(render_report(report), end="")
    return 1 if not report.ok else 0


if __name__ == "__main__":
    sys.exit(main())
