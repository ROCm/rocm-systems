#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Minimal unit tests for the nightly tarball runner.

These cover the pure helpers (KFD decode, tarball names, env pinning, and the
offline/force-sync guard). They do not download a tarball, clone source, or
need a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from rocprof_sys_nightly.cli import main  # noqa: E402
from rocprof_sys_nightly.constants import MULTIARCH_VARIANT  # noqa: E402
from rocprof_sys_nightly.environment import make_rocm_env  # noqa: E402
from rocprof_sys_nightly.gpu import (  # noqa: E402
    _gfx_from_kfd_version,
    _kfd_gpu_archs,
    resolve_variant,
)
from rocprof_sys_nightly.tarball import (  # noqa: E402
    _rocm_version_matches,
    index_dist_variants,
    parse_dist_tarball,
    redirect_variant,
)


@pytest.mark.parametrize(
    "version,arch",
    [
        (90402, "gfx942"),
        (90010, "gfx90a"),
        (110000, "gfx1100"),
        (110501, "gfx1151"),
        (100300, "gfx1030"),
        (0, None),
        (-1, None),
    ],
)
def test_gfx_from_kfd_version(version, arch):
    assert _gfx_from_kfd_version(version) == arch


def test_kfd_gpu_archs_reads_topology(tmp_path, monkeypatch):
    nodes = tmp_path / "nodes"
    samples = {
        "0": "gfx_target_version 0\n",
        "1": "gfx_target_version 90402\n",
        "2": "gfx_target_version 90402\n",
        "3": "gfx_target_version 110501\n",
    }
    for name, text in samples.items():
        node = nodes / name
        node.mkdir(parents=True)
        (node / "properties").write_text(text)

    real_path = Path

    def _path(value, *args, **kwargs):
        if value == "/sys/class/kfd/kfd/topology/nodes":
            return nodes
        return real_path(value, *args, **kwargs)

    monkeypatch.setattr("rocprof_sys_nightly.gpu.Path", _path)
    assert _kfd_gpu_archs() == ["gfx942", "gfx1151"]


def test_parse_dist_tarball_splits_variant_and_version():
    parsed = parse_dist_tarball("therock-dist-linux-gfx94X-dcgpu-7.15.0a20260717.tar.gz")
    assert parsed == ("gfx94X-dcgpu", "7.15.0a20260717")
    assert parse_dist_tarball("not-a-tarball.tar.gz") is None


def test_parse_dist_tarball_keeps_tests_suffix_in_the_variant():
    parsed = parse_dist_tarball(
        "therock-dist-linux-gfx94X-dcgpu-tests-7.15.0a20260717.tar.gz"
    )
    assert parsed == ("gfx94X-dcgpu-tests", "7.15.0a20260717")


def test_index_dist_variants_skips_tests_tarballs():
    html = """
    therock-dist-linux-gfx94X-dcgpu-7.15.0a20260717.tar.gz
    therock-dist-linux-gfx94X-dcgpu-tests-7.15.0a20260717.tar.gz
    therock-dist-linux-multiarch-7.15.0a20260717.tar.gz
    """
    assert index_dist_variants(html) == ["gfx94X-dcgpu", "multiarch"]


def test_redirect_variant_maps_arch_onto_family_tarball():
    available = ["gfx94X-dcgpu", "multiarch"]
    assert redirect_variant("gfx942", available) == "gfx94X-dcgpu"
    assert redirect_variant("gfx94X-dcgpu", available) == "gfx94X-dcgpu"


@pytest.mark.parametrize(
    "requested,full,date,expected",
    [
        ("7.15.0a20260717", "7.15.0a20260717", 20260717, True),
        ("20260717", "7.15.0a20260717", 20260717, True),
        ("7.15.0", "7.15.0a20260717", 20260717, True),
        ("10.1", "7.15.0a20260717", 20260717, False),
        ("0", "7.15.0a20260717", 20260717, False),
        ("7.15", "7.15.0a20260717", 20260717, False),
    ],
)
def test_rocm_version_matches_exact_selectors_only(requested, full, date, expected):
    assert _rocm_version_matches(requested, full, date) is expected


def test_resolve_variant_auto_uses_first_arch_or_multiarch():
    assert resolve_variant("gfx94X-dcgpu", []) == "gfx94X-dcgpu"
    assert resolve_variant("auto", ["gfx942", "gfx90a"]) == "gfx942"
    assert resolve_variant("auto", []) == MULTIARCH_VARIANT


def test_make_rocm_env_drops_leftover_therock(tmp_path):
    rocm = tmp_path / "extracted"
    (rocm / "bin").mkdir(parents=True)
    host = tmp_path / "therock-tarball"
    (host / "bin").mkdir(parents=True)
    stale = "/no/such/therock-tarball/bin"
    env = make_rocm_env(
        {
            "PATH": ":".join([str(host / "bin"), stale, "/usr/bin"]),
            "HIP_PATH": str(host),
            "CMAKE_PREFIX_PATH": f"{host};/opt/other",
            "LD_LIBRARY_PATH": f"{host / 'lib'}:/usr/lib",
        },
        rocm,
    )
    assert env["ROCM_PATH"] == str(rocm)
    assert env["HIP_PATH"] == str(rocm)
    assert env["HSA_PATH"] == str(rocm)
    assert env["CMAKE_PREFIX_PATH"] == str(rocm)
    assert "HIP_CLANG_PATH" not in env
    path = env["PATH"].split(":")
    assert path[0] == str(rocm / "bin")
    assert str(host) not in env["PATH"]
    assert "therock" not in env["PATH"]
    assert "/usr/bin" in path


def test_offline_cannot_combine_with_force_sync():
    with pytest.raises(SystemExit) as exc:
        main(["--offline", "--force-sync"])
    assert exc.value.code == 1
