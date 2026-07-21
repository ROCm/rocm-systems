#!/usr/bin/env python3
"""Preflight replay compatibility using HRR manifest metadata (PR #8680)."""

from __future__ import annotations

import argparse
import json
import os
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
    prompts: list[str] = field(default_factory=list)
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


def capture_device_arch_names(capture: CaptureMetadata) -> list[str]:
    names: list[str] = []
    for dev in capture.devices:
        props = dev.get("properties")
        if isinstance(props, dict):
            arch = props.get("gcn_arch_name")
        else:
            arch = dev.get("gcn_arch_name")
        if arch:
            names.append(str(arch))
    return names


def _run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _docker_cmd(*args: str) -> list[str]:
    """Match replay_docker.sh: use passwordless sudo when available."""
    if shutil.which("docker") is None:
        return ["docker", *args]
    sudo_check = _run(["sudo", "-n", "true"], timeout=5)
    if sudo_check.returncode == 0:
        return ["sudo", "-n", "docker", *args]
    return ["docker", *args]


def resolve_clr_lib_dir(
    *,
    clr_build: str | None = None,
    hrr_playback: str | None = None,
    clr_lib: str | None = None,
) -> Path | None:
    """Resolve the CLR hipamd lib dir used for docker replay mounts."""
    if clr_lib:
        path = Path(clr_lib)
        if path.is_dir() and any(path.glob("libamdhip64.so*")):
            return path.resolve()
    if clr_build:
        path = Path(clr_build) / "hipamd" / "lib"
        if path.is_dir() and any(path.glob("libamdhip64.so*")):
            return path.resolve()
    if hrr_playback:
        play = Path(hrr_playback)
        if play.is_file():
            path = (play.parent / "../../../lib").resolve()
            if path.is_dir() and any(path.glob("libamdhip64.so*")):
                return path
    return None


def _hip_version_from_soname(filename: str) -> str | None:
    match = re.search(r"libamdhip64\.so\.(\d+\.\d+\.\d+)", filename)
    if match:
        return match.group(1)
    return None


def hip_version_from_clr_lib(clr_lib: Path) -> str | None:
    """Parse HIP runtime version from the mounted libamdhip64 SONAME."""
    lib_dir = clr_lib.resolve()
    resolved: list[Path] = []
    for name in ("libamdhip64.so", "libamdhip64.so.7"):
        link = lib_dir / name
        if not link.exists():
            continue
        try:
            target = link.resolve()
        except OSError:
            continue
        if target.is_file():
            resolved.append(target)
    if not resolved:
        versioned = sorted(
            lib_dir.glob("libamdhip64.so.*.*.*"),
            key=lambda path: path.name,
        )
        if versioned:
            resolved.append(versioned[-1].resolve())
    for path in resolved:
        version = _hip_version_from_soname(path.name)
        if version:
            return version
    return None


def default_docker_extra_ld(image: str) -> str | None:
    if image.startswith("rocm/vllm:"):
        return "/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib"
    return None


def build_docker_ld_inside(
    *,
    clr_lib: Path | None,
    rocr_lib: Path | None,
    extra_ld: str | None,
) -> str:
    """Mirror replay_docker.sh LD_LIBRARY_PATH inside the container."""
    if clr_lib is None:
        parts: list[str] = []
        if extra_ld:
            parts.append(extra_ld)
        parts.append("/opt/rocm/lib")
        return ":".join(parts)
    parts = ["/opt/hrr/lib"]
    if rocr_lib is not None:
        parts.append("/opt/hrr/rocr")
    if extra_ld:
        parts.extend([extra_ld, "/opt/rocm/lib"])
    else:
        parts.append("/opt/rocm/lib")
    return ":".join(parts)


def docker_mount_clr_enabled() -> bool:
    return os.environ.get("HRR_DOCKER_MOUNT_CLR", "0") == "1"


def resolve_replay_mounts(
    *,
    docker_image: str | None = None,
    clr_build: str | None = None,
    hrr_playback: str | None = None,
    clr_lib: str | None = None,
    rocr_lib: str | None = None,
    extra_ld: str | None = None,
    mount_clr: bool = False,
) -> tuple[Path | None, Path | None, str | None]:
    extra = extra_ld
    if extra is None and docker_image:
        extra = default_docker_extra_ld(docker_image)
    if not mount_clr:
        return None, None, extra
    clr = resolve_clr_lib_dir(
        clr_build=clr_build,
        hrr_playback=hrr_playback,
        clr_lib=clr_lib,
    )
    rocr: Path | None = None
    if rocr_lib:
        candidate = Path(rocr_lib)
        if candidate.is_dir() and (candidate / "libhsa-runtime64.so.1").is_file():
            rocr = candidate.resolve()
    return clr, rocr, extra


def _parse_hip_version(text: str) -> str | None:
    for line in text.splitlines():
        m = re.search(r"HIP version\s*:\s*(\S+)", line, re.I)
        if m:
            return m.group(1)
        stripped = line.strip()
        if re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:-[0-9a-f]+)?", stripped):
            return stripped
    return None


_PROBE_COMGR_PY = r"""
import ctypes
import glob
import os
import sys

candidates: list[str] = []
for root in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    if root:
        candidates.extend(glob.glob(os.path.join(root, "libamd_comgr.so*")))
candidates.extend(
    [
        "/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib/libamd_comgr.so.3",
        "/opt/python/lib/python3.13/site-packages/_rocm_sdk_devel/lib/libamd_comgr.so.3",
        "/opt/rocm/lib/libamd_comgr.so.3",
        "/opt/rocm/lib/libamd_comgr.so",
    ]
)
seen: set[str] = set()
for path in candidates:
    if not path or path in seen:
        continue
    seen.add(path)
    try:
        lib = ctypes.CDLL(path)
        major = ctypes.c_size_t()
        minor = ctypes.c_size_t()
        lib.amd_comgr_get_version(ctypes.byref(major), ctypes.byref(minor))
        print(f"{major.value}.{minor.value}")
        sys.exit(0)
    except OSError:
        continue
sys.exit(1)
"""


def probe_comgr_version() -> str | None:
    proc = _run(["python3", "-c", _PROBE_COMGR_PY], timeout=15)
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()
    if not line:
        return None
    version = line[-1].strip()
    return version or None


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
        env.hip_runtime_version = _parse_hip_version(proc.stdout + proc.stderr)
    env.comgr_version = probe_comgr_version()
    return env


def probe_docker_replay_env(
    image: str,
    *,
    clr_lib: Path | None = None,
    rocr_lib: Path | None = None,
    extra_ld: str | None = None,
) -> ReplayEnvironment:
    env = ReplayEnvironment(source=f"docker:{image}")
    if not shutil.which("docker"):
        env.source = "docker:unavailable"
        return env

    ld_inside = build_docker_ld_inside(
        clr_lib=clr_lib,
        rocr_lib=rocr_lib,
        extra_ld=extra_ld,
    )
    if clr_lib is not None:
        env.source = f"docker:{image}:overlay:{clr_lib}"

    probe_shell = (
        f"export LD_LIBRARY_PATH={ld_inside}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}; "
        "rocm-smi --showid --showproductname 2>/dev/null || true; "
        "hipconfig --version 2>/dev/null || true"
    )
    docker_args = [
        "run",
        "--rm",
        "--device=/dev/kfd",
        "--device=/dev/dri",
    ]
    if clr_lib is not None:
        docker_args.extend(["-v", f"{clr_lib}:/opt/hrr/lib:ro"])
    if rocr_lib is not None:
        docker_args.extend(["-v", f"{rocr_lib}:/opt/hrr/rocr:ro"])
    docker_args.extend([image, "bash", "-lc", probe_shell])

    proc = _run(_docker_cmd(*docker_args), timeout=120)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0 and "permission denied" in out.lower():
        env.source = "docker:permission-denied"
        return env
    count, archs, names = _parse_rocm_smi(out)
    if count > 0:
        env.visible_gpus = count
        env.gpu_archs = archs
        env.gpu_names = names
    if clr_lib is not None:
        env.hip_runtime_version = hip_version_from_clr_lib(clr_lib)
    if env.hip_runtime_version is None:
        env.hip_runtime_version = _parse_hip_version(out)
    comgr_proc = _run(
        _docker_cmd(
            "run",
            "--rm",
            *(
                ["-v", f"{clr_lib}:/opt/hrr/lib:ro"]
                if clr_lib is not None
                else []
            ),
            "-e",
            f"LD_LIBRARY_PATH={ld_inside}",
            image,
            "python3",
            "-c",
            _PROBE_COMGR_PY,
        ),
        timeout=60,
    )
    if comgr_proc.returncode == 0:
        lines = comgr_proc.stdout.strip().splitlines()
        if lines:
            env.comgr_version = lines[-1].strip()
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
                report.prompts.append(msg)

    if capture.comgr_version and replay.comgr_version:
        if capture.comgr_version != replay.comgr_version:
            msg = (
                "comgr version mismatch: capture "
                f"{capture.comgr_version} vs replay {replay.comgr_version}"
            )
            if strict_version:
                report.blocks.append(msg)
            else:
                report.prompts.append(msg)

    report.ok = not report.blocks
    return report


def render_report(report: CompatReport) -> str:
    lines = ["# HRR replay compatibility"]
    if report.capture:
        c = report.capture
        arch_names = capture_device_arch_names(c)
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
                f"- gcn_arch_name: {arch_names[0] if arch_names else 'n/a'}",
            ]
        )
        if len(arch_names) > 1:
            lines.append(f"- gcn_arch_names: {', '.join(arch_names)}")
    if report.replay:
        r = report.replay
        lines.extend(
            [
                "",
                "## Replay environment",
                f"- source: {r.source}",
                f"- visible_gpus: {r.visible_gpus if r.visible_gpus is not None else 'unknown'}",
                f"- hip_runtime_version: {r.hip_runtime_version or 'n/a'}",
                f"- comgr_version: {r.comgr_version or 'n/a'}",
            ]
        )
        if r.gpu_archs:
            lines.append(f"- gpu_archs: {', '.join(r.gpu_archs)}")
    if report.blocks:
        lines.extend(["", "## Blocking issues"])
        lines.extend(f"- {b}" for b in report.blocks)
    if report.prompts:
        lines.extend(["", "## Confirmation required"])
        lines.extend(f"- {p}" for p in report.prompts)
        lines.extend(
            [
                "",
                "Replay stack versions differ from capture. Do you want to continue?",
                "Set HRR_CONTINUE=1 to proceed without an interactive prompt.",
            ]
        )
    if report.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {w}" for w in report.warnings)
    if report.ok and not report.warnings and not report.prompts:
        lines.extend(["", "Replay preflight: OK"])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, help="pid-* archive directory")
    ap.add_argument("--mode", choices=("host", "docker"), default="host")
    ap.add_argument("--docker-image", help="Capture/replay Docker image")
    ap.add_argument(
        "--clr-lib",
        help="CLR hipamd lib dir mounted for replay (default: CLR_BUILD/HRR_PLAYBACK)",
    )
    ap.add_argument(
        "--rocr-lib",
        help="In-tree ROCR lib dir mounted for replay (default: ROCR_LIB env)",
    )
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

    clr_lib, rocr_lib, extra_ld = resolve_replay_mounts(
        docker_image=args.docker_image,
        clr_build=os.environ.get("CLR_BUILD"),
        hrr_playback=os.environ.get("HRR_PLAYBACK"),
        clr_lib=args.clr_lib,
        rocr_lib=args.rocr_lib or os.environ.get("ROCR_LIB"),
        extra_ld=os.environ.get("HRR_DOCKER_EXTRA_LD"),
        mount_clr=args.mode == "docker" and docker_mount_clr_enabled(),
    )
    if clr_lib is not None:
        print(
            f"[compat] dev overlay: probing mounted CLR lib at {clr_lib}",
            file=sys.stderr,
        )
    elif args.mode == "docker":
        print(
            "[compat] probing Docker image HRR stack (set HRR_DOCKER_MOUNT_CLR=1 to overlay host build)",
            file=sys.stderr,
        )

    if args.mode == "docker":
        if not args.docker_image:
            print(
                "error: --docker-image required for docker preflight", file=sys.stderr
            )
            return 2
        replay = probe_docker_replay_env(
            args.docker_image,
            clr_lib=clr_lib,
            rocr_lib=rocr_lib,
            extra_ld=extra_ld,
        )
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
    if not report.ok:
        return 1
    if report.prompts:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
