# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for the P2P-DMA / AIS capability checks."""

# Test functions are self-documenting by name; docstrings add noise.
# pylint: disable=missing-function-docstring

import io
from collections import namedtuple

import pytest

_UNAME = namedtuple("uname_result", "sysname nodename release version machine")


def _fake_uname(release="6.6.0-test"):
    return _UNAME("Linux", "host", release, "#1 SMP", "x86_64")


def _install_fake_openers(monkeypatch, ais_check, files):
    """
    Redirect both ``open`` (builtins) and ``gzip.open`` so that only the paths
    present in ``files`` are readable; everything else raises OSError, matching
    a missing config file.
    """

    def opener(path, *_args, **_kwargs):
        if path in files:
            return io.StringIO(files[path])
        raise OSError(f"no such file: {path}")

    monkeypatch.setattr(ais_check, "open", opener, raising=False)
    monkeypatch.setattr(ais_check.gzip, "open", opener)


# ---------------------------------------------------------------------------
# config_supports_p2pdma(): the kernel-config fallback path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_path",
    [
        "/boot/config-6.6.0-test",
        "/lib/modules/6.6.0-test/build/.config",
        "/proc/config.gz",
    ],
)
def test_supported_from_each_source(monkeypatch, ais_check, config_path):
    monkeypatch.setattr(ais_check.os, "uname", _fake_uname)
    _install_fake_openers(
        monkeypatch,
        ais_check,
        {config_path: "CONFIG_FOO=y\nCONFIG_PCI_P2PDMA=y\nCONFIG_BAR=m\n"},
    )

    assert ais_check.config_supports_p2pdma() is True


def test_config_present_but_option_absent(monkeypatch, ais_check):
    monkeypatch.setattr(ais_check.os, "uname", _fake_uname)
    _install_fake_openers(
        monkeypatch,
        ais_check,
        {"/boot/config-6.6.0-test": "CONFIG_FOO=y\nCONFIG_BAR=m\n"},
    )

    assert ais_check.config_supports_p2pdma() is False


def test_option_as_module_does_not_count(monkeypatch, ais_check):
    monkeypatch.setattr(ais_check.os, "uname", _fake_uname)
    _install_fake_openers(
        monkeypatch,
        ais_check,
        {"/boot/config-6.6.0-test": "CONFIG_PCI_P2PDMA=m\n"},
    )

    assert ais_check.config_supports_p2pdma() is False


def test_option_commented_out_does_not_count(monkeypatch, ais_check):
    monkeypatch.setattr(ais_check.os, "uname", _fake_uname)
    _install_fake_openers(
        monkeypatch,
        ais_check,
        {"/boot/config-6.6.0-test": "# CONFIG_PCI_P2PDMA is not set\n"},
    )

    assert ais_check.config_supports_p2pdma() is False


def test_no_configs_found_warns_on_stderr(monkeypatch, capsys, ais_check):
    monkeypatch.setattr(ais_check.os, "uname", _fake_uname)
    _install_fake_openers(monkeypatch, ais_check, {})

    assert ais_check.config_supports_p2pdma() is False
    assert "No kernel config files found!" in capsys.readouterr().err


def test_first_matching_source_wins(monkeypatch, ais_check):
    """
    A readable /boot config without the option followed by /proc/config.gz with
    it should still report supported (the loop must consult all sources, not
    bail after the first readable one).
    """
    monkeypatch.setattr(ais_check.os, "uname", _fake_uname)
    _install_fake_openers(
        monkeypatch,
        ais_check,
        {
            "/boot/config-6.6.0-test": "CONFIG_FOO=y\n",
            "/proc/config.gz": "CONFIG_PCI_P2PDMA=y\n",
        },
    )

    assert ais_check.config_supports_p2pdma() is True


# ---------------------------------------------------------------------------
# amdgpu_dkms_version(): parse the build number out of `dkms status`.
# ---------------------------------------------------------------------------

_Completed = namedtuple("CompletedProcess", "stdout")


def _patch_dkms(monkeypatch, ais_check, *, stdout=None, exc=None, release="6.8.0-101"):
    monkeypatch.setattr(ais_check.os, "uname", lambda: _fake_uname(release))

    def fake_run(*_args, **_kwargs):
        if exc is not None:
            raise exc
        return _Completed(stdout)

    monkeypatch.setattr(ais_check.subprocess, "run", fake_run)


def test_dkms_version_new_format(monkeypatch, ais_check):
    _patch_dkms(
        monkeypatch,
        ais_check,
        stdout=(
            "amdgpu/6.16.13-2278356.24.04, 6.8.0-101, x86_64: installed\n"
            "nvidia/555.42.06, 6.8.0-101, x86_64: installed\n"
        ),
    )

    assert ais_check.amdgpu_dkms_version() == 2278356


def test_dkms_version_old_format(monkeypatch, ais_check):
    _patch_dkms(
        monkeypatch,
        ais_check,
        stdout="amdgpu, 6.16.13-2327937.24.04, 6.8.0-101, x86_64: installed\n",
    )

    assert ais_check.amdgpu_dkms_version() == 2327937


def test_dkms_version_prefers_running_kernel(monkeypatch, ais_check):
    _patch_dkms(
        monkeypatch,
        ais_check,
        release="6.8.0-101",
        stdout=(
            "amdgpu/6.16.13-1111111.24.04, 6.8.0-100, x86_64: installed\n"
            "amdgpu/6.16.13-2278356.24.04, 6.8.0-101, x86_64: installed\n"
        ),
    )

    assert ais_check.amdgpu_dkms_version() == 2278356


def test_dkms_version_falls_back_to_max_when_no_running_match(monkeypatch, ais_check):
    _patch_dkms(
        monkeypatch,
        ais_check,
        release="6.8.0-999",
        stdout=(
            "amdgpu/6.16.13-1111111.24.04, 6.8.0-100, x86_64: installed\n"
            "amdgpu/6.16.13-2278356.24.04, 6.8.0-101, x86_64: installed\n"
        ),
    )

    assert ais_check.amdgpu_dkms_version() == 2278356


def test_dkms_version_no_amdgpu_entry(monkeypatch, ais_check):
    _patch_dkms(
        monkeypatch,
        ais_check,
        stdout="nvidia/555.42.06, 6.8.0-101, x86_64: installed\n",
    )

    assert ais_check.amdgpu_dkms_version() is None


def test_dkms_version_missing_binary(monkeypatch, ais_check):
    _patch_dkms(monkeypatch, ais_check, exc=FileNotFoundError())

    assert ais_check.amdgpu_dkms_version() is None


# ---------------------------------------------------------------------------
# topology_reports_ais(): read the KFD topology "capability" bit.
# ---------------------------------------------------------------------------


def _patch_topology(monkeypatch, ais_check, nodes):
    """
    ``nodes`` maps node name -> properties file contents (or None to make the
    properties file unreadable).
    """
    monkeypatch.setattr(ais_check.os, "listdir", lambda _p: list(nodes))

    def opener(path, *_args, **_kwargs):
        for name, contents in nodes.items():
            if path.endswith(f"/{name}/properties"):
                if contents is None:
                    raise OSError("unreadable")
                return io.StringIO(contents)
        raise OSError(f"no such file: {path}")

    monkeypatch.setattr(ais_check, "open", opener, raising=False)


def test_topology_bit_set(monkeypatch, ais_check):
    # 0x40 set -> AIS initialized.
    _patch_topology(
        monkeypatch,
        ais_check,
        {
            "0": "cpu_cores_count 48\n",
            "1": "simd_count 144\ncapability 64\n",
        },
    )

    assert ais_check.topology_reports_ais() is True


def test_topology_bit_clear(monkeypatch, ais_check):
    # 675521152 (0x28410000) does not have 0x40 set.
    _patch_topology(
        monkeypatch,
        ais_check,
        {"1": "simd_count 144\ncapability 675521152\n"},
    )

    assert ais_check.topology_reports_ais() is False


def test_topology_any_gpu_node_with_bit_counts(monkeypatch, ais_check):
    _patch_topology(
        monkeypatch,
        ais_check,
        {
            "1": "simd_count 144\ncapability 0\n",
            "2": "simd_count 144\ncapability 64\n",
        },
    )

    assert ais_check.topology_reports_ais() is True


def test_topology_bit_set_on_non_gpu_node_ignored(monkeypatch, ais_check):
    # capability bit is set, but simd_count is 0 -> not a GPU, so skipped.
    _patch_topology(
        monkeypatch,
        ais_check,
        {
            "0": "cpu_cores_count 48\nsimd_count 0\ncapability 64\n",
        },
    )

    assert ais_check.topology_reports_ais() is False


def test_topology_bit_set_but_no_simd_count(monkeypatch, ais_check):
    # capability bit set but simd_count absent -> not treated as a GPU.
    _patch_topology(
        monkeypatch,
        ais_check,
        {"0": "capability 64\n"},
    )

    assert ais_check.topology_reports_ais() is False


def test_topology_missing_capability_field(monkeypatch, ais_check):
    _patch_topology(monkeypatch, ais_check, {"0": "cpu_cores_count 48\n"})

    assert ais_check.topology_reports_ais() is False


def test_topology_no_nodes_dir(monkeypatch, ais_check):
    def raise_oserror(_path):
        raise OSError("no topology")

    monkeypatch.setattr(ais_check.os, "listdir", raise_oserror)

    assert ais_check.topology_reports_ais() is False


# ---------------------------------------------------------------------------
# kernel_supports_p2pdma(): the version-gated dispatcher.
# ---------------------------------------------------------------------------


def _patch_dispatch(monkeypatch, ais_check, *, version, topology, config):
    monkeypatch.setattr(ais_check, "amdgpu_dkms_version", lambda: version)
    monkeypatch.setattr(ais_check, "topology_reports_ais", lambda: topology)
    monkeypatch.setattr(ais_check, "config_supports_p2pdma", lambda: config)


def test_new_driver_uses_topology_true(monkeypatch, ais_check):
    _patch_dispatch(
        monkeypatch,
        ais_check,
        version=ais_check.KFD_AIS_CAPABILITY_MIN_VERSION + 1,
        topology=True,
        config=False,
    )

    assert ais_check.kernel_supports_p2pdma() is True


def test_new_driver_bit_unset_reports_unsupported(monkeypatch, ais_check):
    # New enough driver but topology bit clear -> unsupported, config ignored.
    _patch_dispatch(
        monkeypatch,
        ais_check,
        version=ais_check.KFD_AIS_CAPABILITY_MIN_VERSION + 1,
        topology=False,
        config=True,
    )

    assert ais_check.kernel_supports_p2pdma() is False


def test_threshold_is_exclusive(monkeypatch, ais_check):
    # Exactly at the threshold is not "higher", so it uses the config path.
    _patch_dispatch(
        monkeypatch,
        ais_check,
        version=ais_check.KFD_AIS_CAPABILITY_MIN_VERSION,
        topology=False,
        config=True,
    )

    assert ais_check.kernel_supports_p2pdma() is True


def test_old_driver_uses_config(monkeypatch, ais_check):
    _patch_dispatch(
        monkeypatch,
        ais_check,
        version=ais_check.KFD_AIS_CAPABILITY_MIN_VERSION - 1,
        topology=False,
        config=True,
    )

    assert ais_check.kernel_supports_p2pdma() is True


def test_unknown_version_falls_back_to_config(monkeypatch, ais_check):
    _patch_dispatch(
        monkeypatch,
        ais_check,
        version=None,
        topology=True,
        config=False,
    )

    assert ais_check.kernel_supports_p2pdma() is False
