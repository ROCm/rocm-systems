#!/usr/bin/env python3
"""Preflight replay compatibility using HRR manifest metadata.

Cross-platform: works on Linux (--mode host|docker) and Windows (--mode host only).
Docker replay on Windows is not supported; the script exits with an informative error.
"""

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

_IS_WIN = sys.platform == "win32"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nested_get(obj: dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


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


# ---------------------------------------------------------------------------
# Manifest loader (cross-platform)
# ---------------------------------------------------------------------------

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
    hip_ver = comgr_ver = None
    if isinstance(runtime, dict):
        hip_ver = runtime.get("hip_runtime_version")
        comgr_ver = runtime.get("comgr_version")
    if hip_ver is None:
        hip_ver = meta.get("hip_runtime_version")
    if comgr_ver is None:
        comgr_ver = meta.get("comgr_version")

    devices: list[dict[str, Any]] = []
    for item in meta.get("devices", []):
        if isinstance(item, dict):
            devices.append(item)

    return CaptureMetadata(
        schema_version=_as_int(meta.get("schema_version")),
        hip_runtime_version=str(hip_ver) if hip_ver else None,
        comgr_version=str(comgr_ver) if comgr_ver else None,
        device_count=_as_int(meta.get("device_count")),
        captured_device_count=_as_int(meta.get("captured_device_count")),
        devices=devices,
        raw=meta,
    )


def capture_device_arch_names(capture: CaptureMetadata) -> list[str]:
    names: list[str] = []
    for dev in capture.devices:
        props = dev.get("properties")
        arch = (props or dev).get("gcn_arch_name") if isinstance(props, dict) else dev.get("gcn_arch_name")
        if arch:
            names.append(str(arch))
    return names


# ---------------------------------------------------------------------------
# HIP version parsing
# ---------------------------------------------------------------------------

def _parse_hip_version(text: str) -> str | None:
    for line in text.splitlines():
        m = re.search(r"HIP version\s*:\s*(\S+)", line, re.I)
        if m:
            return m.group(1)
        stripped = line.strip()
        if re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:-[0-9a-f]+)?", stripped):
            return stripped
    return None


def _find_hip_exe(name: str) -> str | None:
    """Find a HIP SDK executable on Linux or Windows."""
    hip_root = os.environ.get("HIP_PATH") or os.environ.get("ROCM_PATH", "")
    suffixes = ([".exe", ""] if _IS_WIN else [""])
    dirs = []
    if hip_root:
        for sub in (("bin", "hip\\bin", "tools\\bin") if _IS_WIN else ("bin",)):
            dirs.append(os.path.join(hip_root, sub))
    # PATH
    for p in os.environ.get("PATH", "").split(os.pathsep):
        dirs.append(p)
    for d in dirs:
        for sfx in suffixes:
            full = os.path.join(d, name + sfx)
            if os.path.isfile(full):
                return full
    return shutil.which(name + (".exe" if _IS_WIN else "")) or shutil.which(name)


# ---------------------------------------------------------------------------
# comgr version probe (cross-platform)
# ---------------------------------------------------------------------------

# On Linux we dlopen libamd_comgr.so; on Windows we LoadLibrary amd_comgr.dll.
_PROBE_COMGR_PY_LINUX = r"""
import ctypes, glob, os, sys

candidates = []
for root in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    if root:
        candidates.extend(glob.glob(os.path.join(root, "libamd_comgr.so*")))
candidates += [
    "/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib/libamd_comgr.so.3",
    "/opt/rocm/lib/libamd_comgr.so.3",
    "/opt/rocm/lib/libamd_comgr.so",
]
seen = set()
for path in candidates:
    if not path or path in seen: continue
    seen.add(path)
    try:
        lib = ctypes.CDLL(path)
        major, minor = ctypes.c_size_t(), ctypes.c_size_t()
        lib.amd_comgr_get_version(ctypes.byref(major), ctypes.byref(minor))
        print(f"{major.value}.{minor.value}")
        sys.exit(0)
    except OSError:
        continue
sys.exit(1)
"""

_PROBE_COMGR_PY_WIN = r"""
import ctypes, os, sys
from pathlib import Path

hip_root = os.environ.get("HIP_PATH") or os.environ.get("ROCM_PATH", "")
candidates = []
if hip_root:
    for sub in ("bin", "lib", r"hip\bin", r"hip\lib"):
        candidates.append(str(Path(hip_root) / sub / "amd_comgr.dll"))
for p in os.environ.get("PATH", "").split(os.pathsep):
    candidates.append(os.path.join(p, "amd_comgr.dll"))

seen = set()
for path in candidates:
    if not path or path in seen: continue
    seen.add(path)
    if not Path(path).is_file(): continue
    try:
        lib = ctypes.CDLL(path)
        major, minor = ctypes.c_size_t(), ctypes.c_size_t()
        lib.amd_comgr_get_version(ctypes.byref(major), ctypes.byref(minor))
        print(f"{major.value}.{minor.value}")
        sys.exit(0)
    except OSError:
        continue
sys.exit(1)
"""


def probe_comgr_version() -> str | None:
    script = _PROBE_COMGR_PY_WIN if _IS_WIN else _PROBE_COMGR_PY_LINUX
    py = shutil.which("python") or shutil.which("python3") or "python"
    proc = _run([py, "-c", script], timeout=15)
    if proc.returncode != 0:
        return None
    lines = proc.stdout.strip().splitlines()
    return lines[-1].strip() if lines else None


# ---------------------------------------------------------------------------
# Linux: rocm-smi based GPU enumeration
# ---------------------------------------------------------------------------

def _parse_rocm_smi(text: str) -> tuple[int, list[str], list[str]]:
    gpu_ids = sorted({int(m.group(1)) for m in re.finditer(r"GPU\[(\d+)\]", text)})
    archs: list[str] = []
    names: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if "GFX Version" in line or "GPU Family" in line or "Card series" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                archs.append(parts[-1].strip().lower())
        if "Card model" in line or "Card name" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                names.append(parts[-1].strip())
    return len(gpu_ids), archs, names


# ---------------------------------------------------------------------------
# Windows: amd-smi / hipDeviceGetProperties based GPU enumeration
# ---------------------------------------------------------------------------

_PROBE_GPU_WIN_PY = r"""
import ctypes, os, sys
from pathlib import Path

hip_root = os.environ.get("HIP_PATH") or os.environ.get("ROCM_PATH", "")
dll_candidates = []
if hip_root:
    for sub in ("bin", r"hip\bin"):
        dll_candidates.append(str(Path(hip_root) / sub / "amdhip64.dll"))
for p in os.environ.get("PATH", "").split(os.pathsep):
    dll_candidates.append(os.path.join(p, "amdhip64.dll"))

lib = None
for c in dll_candidates:
    if not Path(c).is_file(): continue
    try:
        lib = ctypes.CDLL(c)
        break
    except OSError:
        continue

if lib is None:
    sys.exit(1)

# hipGetDeviceCount
count = ctypes.c_int(0)
rc = lib.hipGetDeviceCount(ctypes.byref(count))
if rc != 0:
    sys.exit(1)

n = count.value
print(f"gpu_count={n}")

# hipDeviceGetAttribute for arch
hipDeviceAttributeGcnArch = 57   # HIP attribute enum value
for i in range(n):
    val = ctypes.c_int(0)
    rc = lib.hipDeviceGetAttribute(ctypes.byref(val), hipDeviceAttributeGcnArch, i)
    if rc == 0:
        print(f"gpu_{i}_gcn={val.value}")

# hipGetDeviceProperties for name
class hipDevicePropFields(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("_pad", ctypes.c_uint8 * (4096 - 256)),
    ]

for i in range(n):
    props = hipDevicePropFields()
    rc = lib.hipGetDeviceProperties(ctypes.byref(props), i)
    if rc == 0:
        print(f"gpu_{i}_name={props.name.decode(errors='replace')}")
"""

_PROBE_HIP_VERSION_WIN_PY = r"""
import ctypes, os, sys
from pathlib import Path

hip_root = os.environ.get("HIP_PATH") or os.environ.get("ROCM_PATH", "")
dll_candidates = []
if hip_root:
    for sub in ("bin", r"hip\bin"):
        dll_candidates.append(str(Path(hip_root) / sub / "amdhip64.dll"))
for p in os.environ.get("PATH", "").split(os.pathsep):
    dll_candidates.append(os.path.join(p, "amdhip64.dll"))

lib = None
for c in dll_candidates:
    if not Path(c).is_file(): continue
    try:
        lib = ctypes.CDLL(c)
        break
    except OSError:
        continue

if lib is None:
    sys.exit(1)

major = ctypes.c_int(0)
minor = ctypes.c_int(0)
patch = ctypes.c_int(0)
lib.hipRuntimeGetVersion.restype = ctypes.c_int
lib.hipRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
flat = ctypes.c_int(0)
rc = lib.hipRuntimeGetVersion(ctypes.byref(flat))
if rc == 0:
    v = flat.value
    maj = v // 10000000
    min_ = (v // 100000) % 100
    pat = (v // 100) % 1000
    print(f"{maj}.{min_}.{pat:05d}")
    sys.exit(0)
sys.exit(1)
"""


def _parse_win_gpu_probe(text: str) -> tuple[int, list[str], list[str]]:
    """Parse the output of _PROBE_GPU_WIN_PY."""
    count = 0
    archs: list[str] = []
    names: list[str] = []
    for line in text.splitlines():
        m = re.match(r"gpu_count=(\d+)", line)
        if m:
            count = int(m.group(1))
        m = re.match(r"gpu_(\d+)_gcn=(\d+)", line)
        if m:
            gcn = int(m.group(2))
            # Map integer GCN arch to string (e.g. 0x942 → gfx942)
            archs.append(f"gfx{gcn:x}")
        m = re.match(r"gpu_(\d+)_name=(.+)", line)
        if m:
            names.append(m.group(2).strip())
    return count, archs, names


# ---------------------------------------------------------------------------
# Host environment probe — Linux
# ---------------------------------------------------------------------------

def probe_host_replay_env_linux() -> ReplayEnvironment:
    env = ReplayEnvironment(source="host:linux")
    smi = shutil.which("rocm-smi") or shutil.which("amd-smi")
    if smi:
        proc = _run([smi, "--showid", "--showproductname"], timeout=30)
        out = proc.stdout + proc.stderr
        count, archs, names = _parse_rocm_smi(out)
        if count > 0:
            env.visible_gpus = count
            env.gpu_archs = archs
            env.gpu_names = names

    hip_config = shutil.which("hipconfig")
    if hip_config:
        proc = _run([hip_config, "--version"], timeout=15)
        env.hip_runtime_version = _parse_hip_version(proc.stdout + proc.stderr)

    env.comgr_version = probe_comgr_version()
    return env


# ---------------------------------------------------------------------------
# Host environment probe — Windows
# ---------------------------------------------------------------------------

def probe_host_replay_env_windows() -> ReplayEnvironment:
    env = ReplayEnvironment(source="host:windows")
    py = shutil.which("python") or shutil.which("python3") or "python"

    # GPU count + arch via HIP runtime DLL
    proc = _run([py, "-c", _PROBE_GPU_WIN_PY], timeout=30)
    if proc.returncode == 0:
        count, archs, names = _parse_win_gpu_probe(proc.stdout)
        env.visible_gpus = count
        env.gpu_archs = archs
        env.gpu_names = names

    # HIP version via amd-smi > hipconfig.bat > amdhip64.dll
    amd_smi = _find_hip_exe("amd-smi")
    if amd_smi:
        proc2 = _run([amd_smi, "--version"], timeout=15)
        v = _parse_hip_version(proc2.stdout + proc2.stderr)
        if v:
            env.hip_runtime_version = v

    if not env.hip_runtime_version:
        hip_config = _find_hip_exe("hipconfig")
        if hip_config:
            proc3 = _run([hip_config, "--version"], timeout=15)
            env.hip_runtime_version = _parse_hip_version(proc3.stdout + proc3.stderr)

    if not env.hip_runtime_version:
        proc4 = _run([py, "-c", _PROBE_HIP_VERSION_WIN_PY], timeout=15)
        if proc4.returncode == 0:
            v = proc4.stdout.strip().splitlines()
            if v:
                env.hip_runtime_version = v[-1].strip()

    env.comgr_version = probe_comgr_version()
    return env


def probe_host_replay_env() -> ReplayEnvironment:
    return probe_host_replay_env_windows() if _IS_WIN else probe_host_replay_env_linux()


# ---------------------------------------------------------------------------
# Docker environment probe — Linux only
# ---------------------------------------------------------------------------

def _docker_cmd(*args: str) -> list[str]:
    if _IS_WIN:
        # On Windows, Docker Desktop does not require sudo
        return ["docker", *args]
    if shutil.which("docker") is None:
        return ["docker", *args]
    sudo_check = _run(["sudo", "-n", "true"], timeout=5)
    if sudo_check.returncode == 0:
        return ["sudo", "-n", "docker", *args]
    return ["docker", *args]


def default_docker_extra_ld(image: str) -> str | None:
    if image.startswith("rocm/vllm:"):
        return "/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib"
    return None


def docker_mount_clr_enabled() -> bool:
    return os.environ.get("HRR_DOCKER_MOUNT_CLR", "0") == "1"


def resolve_clr_lib_dir(
    *,
    clr_build: str | None = None,
    hrr_playback: str | None = None,
    clr_lib: str | None = None,
) -> Path | None:
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


def build_docker_ld_inside(
    *,
    clr_lib: Path | None,
    rocr_lib: Path | None,
    extra_ld: str | None,
) -> str:
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


_PROBE_COMGR_PY = _PROBE_COMGR_PY_LINUX  # used by docker probe (always Linux inside container)


def probe_docker_replay_env(
    image: str,
    *,
    clr_lib: Path | None = None,
    rocr_lib: Path | None = None,
    extra_ld: str | None = None,
) -> ReplayEnvironment:
    if _IS_WIN:
        env = ReplayEnvironment(source="docker:unsupported-on-windows")
        return env

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
        "run", "--rm",
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
    # comgr inside container
    comgr_args = _docker_cmd("run", "--rm")
    if clr_lib is not None:
        comgr_args.extend(["-v", f"{clr_lib}:/opt/hrr/lib:ro"])
    comgr_args.extend(["-e", f"LD_LIBRARY_PATH={ld_inside}", image, "python3", "-c", _PROBE_COMGR_PY])
    cp = _run(comgr_args, timeout=60)
    if cp.returncode == 0:
        lines = cp.stdout.strip().splitlines()
        if lines:
            env.comgr_version = lines[-1].strip()
    return env


# ---------------------------------------------------------------------------
# Compatibility evaluation
# ---------------------------------------------------------------------------

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

    capture_archs: list[str] = []
    for dev in capture.devices:
        props = dev.get("properties")
        arch = (props if isinstance(props, dict) else dev).get("gcn_arch_name", "")
        a = _normalize_arch(str(arch))
        if a:
            capture_archs.append(a)

    if capture_archs and replay.gpu_archs:
        replay_arch = replay.gpu_archs[min(gpu, len(replay.gpu_archs) - 1)]
        expected = capture_archs[min(gpu, len(capture_archs) - 1)]
        if expected and replay_arch and expected != replay_arch:
            msg = (
                f"GPU architecture mismatch: capture GPU {gpu} is {expected}, "
                f"replay GPU {gpu} looks like {replay_arch}"
            )
            (report.blocks if strict_arch else report.warnings).append(msg)

    if capture.hip_runtime_version and replay.hip_runtime_version:
        if capture.hip_runtime_version != replay.hip_runtime_version:
            msg = (
                "HIP runtime version mismatch: capture "
                f"{capture.hip_runtime_version} vs replay {replay.hip_runtime_version}"
            )
            (report.blocks if strict_version else report.prompts).append(msg)

    if capture.comgr_version and replay.comgr_version:
        if capture.comgr_version != replay.comgr_version:
            msg = (
                "comgr version mismatch: capture "
                f"{capture.comgr_version} vs replay {replay.comgr_version}"
            )
            (report.blocks if strict_version else report.prompts).append(msg)

    report.ok = not report.blocks
    return report


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_report(report: CompatReport) -> str:
    lines = ["# HRR replay compatibility"]
    if report.capture:
        c = report.capture
        arch_names = capture_device_arch_names(c)
        lines += [
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
        if len(arch_names) > 1:
            lines.append(f"- gcn_arch_names: {', '.join(arch_names)}")
    if report.replay:
        r = report.replay
        lines += [
            "",
            "## Replay environment",
            f"- source: {r.source}",
            f"- visible_gpus: {r.visible_gpus if r.visible_gpus is not None else 'unknown'}",
            f"- hip_runtime_version: {r.hip_runtime_version or 'n/a'}",
            f"- comgr_version: {r.comgr_version or 'n/a'}",
        ]
        if r.gpu_archs:
            lines.append(f"- gpu_archs: {', '.join(r.gpu_archs)}")
    if report.blocks:
        lines += ["", "## Blocking issues"]
        lines += [f"- {b}" for b in report.blocks]
    if report.prompts:
        lines += ["", "## Confirmation required"]
        lines += [f"- {p}" for p in report.prompts]
        lines += [
            "",
            "Replay stack versions differ from capture. Do you want to continue?",
            "Set HRR_CONTINUE=1 to proceed without an interactive prompt.",
        ]
    if report.warnings:
        lines += ["", "## Warnings"]
        lines += [f"- {w}" for w in report.warnings]
    if report.ok and not report.warnings and not report.prompts:
        lines += ["", "Replay preflight: OK"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, help="pid-* archive directory")
    ap.add_argument("--mode", choices=("host", "docker"), default="host")
    ap.add_argument("--docker-image", help="Capture/replay Docker image (Linux only)")
    ap.add_argument("--clr-lib", help="CLR hipamd lib dir for docker overlay")
    ap.add_argument("--rocr-lib", help="In-tree ROCR lib dir for docker overlay")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--strict-version", action="store_true")
    ap.add_argument("--strict-arch", action="store_true")
    args = ap.parse_args()

    if args.mode == "docker" and _IS_WIN:
        print(
            "error: --mode docker is not supported on Windows.\n"
            "       For Docker replay use WSL2 and run replay_docker.sh from there.",
            file=sys.stderr,
        )
        return 1

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

    if args.mode == "docker":
        if not args.docker_image:
            print("error: --docker-image required for docker preflight", file=sys.stderr)
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
        capture, replay,
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
