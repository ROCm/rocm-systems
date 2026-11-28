"""GPU-free tests for the packaging inputs in setup.py and CMakeLists.txt.

Outside tests/ because tests/conftest.py initializes rocSHMEM. The wheel test
needs nanobind, cmake, readelf, patchelf and a rocSHMEM install that setup.py
would find; it is skipped without them, or fails if
ROCSHMEM4PY_REQUIRE_WHEEL_BUILD_TESTS=1.

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

try:
    from packaging.requirements import Requirement
except ImportError:  # setuptools < 71
    from setuptools.extern.packaging.requirements import Requirement

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUTS = (
    "ROCSHMEM4PY_VERSION_SUFFIX",
    "ROCSHMEM4PY_EXTRA_INSTALL_REQUIREMENTS",
    "ROCSHMEM4PY_INSTALL_RPATH",
)
REQUIREMENT = "rocm-sdk-core==10.1.0.dev0+0123abcdef"
RPATH = "$ORIGIN/_rocm_sdk_core/lib:$ORIGIN/_rocm_sdk_core/lib/llvm/lib"
BASE_VERSION = re.search(
    r'BASE_VERSION = "([^"]+)"', (PROJECT_DIR / "setup.py").read_text()
).group(1)


def _copy_project(tmp_path):
    dst = tmp_path / "src"
    shutil.copytree(
        PROJECT_DIR,
        dst,
        ignore=shutil.ignore_patterns(
            "build", "dist", "*.egg-info", "tests", "build_tests"
        ),
    )
    return dst


def _env(**inputs):
    env = {k: v for k, v in os.environ.items() if k not in INPUTS}
    env.update(inputs)
    return env


def _headers(metadata, name):
    header = metadata.split("\n\n", 1)[0]
    return [
        line.split(":", 1)[1].strip()
        for line in header.splitlines()
        if line.startswith(name + ":")
    ]


def _requires(metadata):
    # Older wheel versions write "name (>=1)"; compare the normalized form.
    return [str(Requirement(r)) for r in _headers(metadata, "Requires-Dist")]


def _metadata(tmp_path, **inputs):
    """Run the PEP 517 metadata hook, which evaluates setup.py without building."""
    out = tmp_path / "metadata"
    out.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from setuptools import build_meta as b; b.prepare_metadata_for_build_wheel(sys.argv[1])",
            str(out),
        ],
        cwd=_copy_project(tmp_path),
        env=_env(**inputs),
        capture_output=True,
        text=True,
        check=False,
    )
    files = list(out.glob("*.dist-info/METADATA"))
    return proc, files[0].read_text() if files else ""


@pytest.mark.parametrize(
    "suffix, version",
    [
        ("+rocm7.10.0", f"{BASE_VERSION}+rocm7.10.0"),
        # TheRock's dev-build suffix, passed through unchanged; pip normalizes it.
        (
            "+devrocm10.1.0.dev0-0123abcdef",
            f"{BASE_VERSION}+devrocm10.1.0.dev0.0123abcdef",
        ),
    ],
)
def test_version_suffix(tmp_path, suffix, version):
    proc, metadata = _metadata(tmp_path, ROCSHMEM4PY_VERSION_SUFFIX=suffix)
    assert proc.returncode == 0, proc.stderr
    assert _headers(metadata, "Version") == [version]


def test_extra_install_requirements(tmp_path):
    proc, metadata = _metadata(
        tmp_path, ROCSHMEM4PY_EXTRA_INSTALL_REQUIREMENTS=f"{REQUIREMENT}|packaging>=20"
    )
    assert proc.returncode == 0, proc.stderr
    requires = _requires(metadata)
    assert REQUIREMENT in requires and "packaging>=20" in requires
    assert any("extra ==" in req for req in requires)


@pytest.mark.parametrize(
    "var, value",
    [
        ("ROCSHMEM4PY_VERSION_SUFFIX", "rocm7.10.0"),
        ("ROCSHMEM4PY_VERSION_SUFFIX", "+rocm!7"),
        ("ROCSHMEM4PY_EXTRA_INSTALL_REQUIREMENTS", "not a requirement!!"),
    ],
)
def test_malformed_input_fails(tmp_path, var, value):
    proc, _ = _metadata(tmp_path, **{var: value})
    assert proc.returncode != 0
    assert var in proc.stderr


def _find_rocshmem():
    """(prefix, targets, problem) for the first install in setup.py's search order."""
    min_version = re.search(
        r"set\(ROCSHMEM_MIN_VERSION ([0-9.]+)\)",
        (PROJECT_DIR / "CMakeLists.txt").read_text(),
    ).group(1)
    prefixes = os.environ.get("CMAKE_PREFIX_PATH", "").split(os.pathsep)
    prefixes += [
        os.environ.get("ROCSHMEM_HOME", ""),
        os.environ.get("ROCM_PATH", ""),
        "/opt/rocm",
    ]
    for prefix in filter(None, prefixes):
        header = Path(prefix) / "include" / "rocshmem" / "rocshmem_config.h"
        if not header.is_file():
            continue
        text = header.read_text()
        version = re.search(r'ROCSHMEM_VERSION\s+"([^"]+)"', text)
        targets = re.search(r'ROCSHMEM_OFFLOAD_TARGETS\s+"([^"]+)"', text)
        as_tuple = lambda v: tuple(
            int(p) for p in re.findall(r"\d+", v)[:3]
        )  # noqa: E731
        if not version or as_tuple(version.group(1)) < as_tuple(min_version):
            return None, [], f"rocSHMEM at {prefix} is older than {min_version}"
        if not targets:
            return None, [], f"rocSHMEM at {prefix} records no ROCSHMEM_OFFLOAD_TARGETS"
        return prefix, targets.group(1).split(), None
    return None, [], "no rocSHMEM install found"


def _wheel_build_problem():
    try:
        import nanobind  # noqa: F401
    except ImportError:
        return "nanobind is not installed"
    missing = [
        tool for tool in ("cmake", "readelf", "patchelf") if shutil.which(tool) is None
    ]
    return f"{', '.join(missing)} not on PATH" if missing else _find_rocshmem()[2]


def _build_wheel(src, out, **inputs):
    targets = _find_rocshmem()[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "-w",
            str(out),
            str(src),
        ],
        env=_env(ROCSHMEM_GPU_TARGETS=targets[0], **inputs),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    (wheel,) = out.glob("*.whl")
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(out / "x")
    (metadata,) = (out / "x").glob("*.dist-info/METADATA")
    (extension,) = (out / "x").glob("_rocshmem4py*.so")
    dynamic = subprocess.run(
        ["readelf", "-dW", str(extension)], capture_output=True, text=True, check=True
    ).stdout
    paths = dict(
        re.findall(r"\((RPATH|RUNPATH)\)\s+Library \w+: \[([^\]]*)\]", dynamic)
    )
    return wheel.name, metadata.read_text(), paths


def test_wheel_inputs_then_rebuild_without_them(tmp_path):
    problem = _wheel_build_problem()
    if problem:
        if os.environ.get("ROCSHMEM4PY_REQUIRE_WHEEL_BUILD_TESTS") == "1":
            pytest.fail(problem)
        pytest.skip(problem)
    src = _copy_project(tmp_path)

    name, metadata, paths = _build_wheel(
        src,
        tmp_path / "with",
        ROCSHMEM4PY_VERSION_SUFFIX="+rocm7.10.0",
        ROCSHMEM4PY_EXTRA_INSTALL_REQUIREMENTS=REQUIREMENT,
        ROCSHMEM4PY_INSTALL_RPATH=RPATH,
    )
    assert name.startswith(f"rocshmem4py-{BASE_VERSION}+rocm7.10.0-")
    assert REQUIREMENT in _requires(metadata)
    # Exactly the requested entries: no build-host ROCm path in a packaged wheel.
    assert paths == {"RPATH": RPATH}

    # Same build tree, as repeated `pip wheel .` runs reuse: no input may stick.
    name, metadata, paths = _build_wheel(src, tmp_path / "without")
    assert name.startswith(f"rocshmem4py-{BASE_VERSION}+rocshmem")
    assert all("extra ==" in req for req in _requires(metadata))
    assert list(paths) == ["RUNPATH"] and "$ORIGIN" not in paths["RUNPATH"]
