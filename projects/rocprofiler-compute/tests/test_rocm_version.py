# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path
from unittest import mock

import pytest

import utils.specs as specs
from utils.specs import (
    _rocm_ver_from_core_dirs,
    _rocm_ver_from_sibling_paths,
    _rocm_ver_from_versioned_path,
    get_rocm_ver,
)


def test_rocm_ver_from_versioned_path():
    assert _rocm_ver_from_versioned_path(Path("/opt/rocm-10.1.0")) == "10.1.0"
    assert _rocm_ver_from_versioned_path(Path("/opt/rocm")) is None


def test_rocm_ver_from_core_dirs(tmp_path: Path):
    (tmp_path / "core-10").mkdir()
    (tmp_path / "core-10.1").mkdir()
    assert _rocm_ver_from_core_dirs(tmp_path) == "10.1"


def test_rocm_ver_from_sibling_paths(tmp_path: Path):
    (tmp_path / "rocm").mkdir()
    (tmp_path / "rocm-10.1.0").mkdir()
    assert _rocm_ver_from_sibling_paths(tmp_path / "rocm") == "10.1.0"


def test_get_rocm_ver_uses_core_dir_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rocm_path = tmp_path / "rocm"
    rocm_path.mkdir()
    (rocm_path / "core-10.1.0").mkdir()

    monkeypatch.setenv("ROCM_PATH", str(rocm_path))
    monkeypatch.delenv("ROCM_VER", raising=False)

    with mock.patch.object(specs, "_rocm_ver_from_amdsmi", return_value=None), mock.patch.object(
        specs, "_rocm_ver_from_packages", return_value=None
    ):
        assert get_rocm_ver() == "10.1.0"


def test_get_rocm_ver_short_circuits_before_subprocess_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rocm_path = tmp_path / "rocm"
    rocm_path.mkdir()
    (rocm_path / "core-10.1.0").mkdir()

    monkeypatch.setenv("ROCM_PATH", str(rocm_path))
    monkeypatch.delenv("ROCM_VER", raising=False)

    with mock.patch.object(specs, "_rocm_ver_from_amdsmi") as mock_amdsmi, mock.patch.object(
        specs, "_rocm_ver_from_packages"
    ) as mock_packages:
        assert get_rocm_ver() == "10.1.0"
        mock_amdsmi.assert_not_called()
        mock_packages.assert_not_called()


def test_get_rocm_ver_uses_rocm_ver_when_no_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rocm_path = tmp_path / "rocm"
    rocm_path.mkdir()

    monkeypatch.setenv("ROCM_PATH", str(rocm_path))
    monkeypatch.setenv("ROCM_VER", "10.1.0")

    with mock.patch.object(specs, "_rocm_ver_from_amdsmi", return_value=None), mock.patch.object(
        specs, "_rocm_ver_from_packages", return_value=None
    ):
        assert get_rocm_ver() == "10.1.0"
