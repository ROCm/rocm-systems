# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for the SR-IOV virtual function detection / warning."""

# Test functions are self-documenting by name; docstrings add noise.
# pylint: disable=missing-function-docstring

import enum
import sys
import types


class _FakeAmdSmiException(Exception):
    """Stand-in for amdsmi.AmdSmiException."""


class _FakeVirtMode(enum.IntEnum):
    UNKNOWN = 0
    BAREMETAL = 1
    HOST = 2
    GUEST = 3
    PASSTHROUGH = 4


# Sentinel used by _fake_amdsmi to make a per-GPU query raise instead of return.
_RAISE = object()


def _install_fake_amdsmi(
    monkeypatch,
    gpus,
    *,
    init_exc=None,
    handles_exc=None,
):
    """Inject a stand-in ``amdsmi`` module into sys.modules.

    ``gpus`` is a list of dicts, one per GPU, with keys:
      mode   -- an _FakeVirtMode member, or _RAISE to make the mode query throw
      bdf    -- BDF string, or _RAISE to make the BDF query throw
      market -- market name string (optional, defaults to "")
    """
    module = types.ModuleType("amdsmi")
    module.AmdSmiException = _FakeAmdSmiException
    module.AmdSmiVirtualizationMode = _FakeVirtMode

    def amdsmi_init():
        if init_exc is not None:
            raise init_exc

    def amdsmi_shut_down():
        pass

    def amdsmi_get_processor_handles():
        if handles_exc is not None:
            raise handles_exc
        return list(range(len(gpus)))

    def amdsmi_get_gpu_virtualization_mode(handle):
        mode = gpus[handle]["mode"]
        if mode is _RAISE:
            raise _FakeAmdSmiException("mode unsupported")
        return {"mode": mode}

    def amdsmi_get_gpu_device_bdf(handle):
        bdf = gpus[handle].get("bdf")
        if bdf is _RAISE or bdf is None:
            raise _FakeAmdSmiException("no bdf")
        return bdf

    def amdsmi_get_gpu_asic_info(handle):
        return {"market_name": gpus[handle].get("market", "")}

    module.amdsmi_init = amdsmi_init
    module.amdsmi_shut_down = amdsmi_shut_down
    module.amdsmi_get_processor_handles = amdsmi_get_processor_handles
    module.amdsmi_get_gpu_virtualization_mode = amdsmi_get_gpu_virtualization_mode
    module.amdsmi_get_gpu_device_bdf = amdsmi_get_gpu_device_bdf
    module.amdsmi_get_gpu_asic_info = amdsmi_get_gpu_asic_info

    monkeypatch.setitem(sys.modules, "amdsmi", module)
    return module


def _vf(bdf, market="AMD Instinct MI300X VF"):
    return {"mode": _FakeVirtMode.GUEST, "bdf": bdf, "market": market}


def _pf(bdf, mode=_FakeVirtMode.BAREMETAL, market="AMD Instinct MI300X"):
    return {"mode": mode, "bdf": bdf, "market": market}


# ---------------------------------------------------------------------------
# sriov_vf_gpus()
# ---------------------------------------------------------------------------


def test_single_vf_detected(monkeypatch, ais_check):
    _install_fake_amdsmi(monkeypatch, [_vf("0000:05:00.0")])

    assert ais_check.sriov_vf_gpus() == [("0000:05:00.0", "AMD Instinct MI300X VF")]


def test_physical_function_not_flagged(monkeypatch, ais_check):
    _install_fake_amdsmi(
        monkeypatch, [_pf("0000:23:00.0", market="AMD Radeon RX 6800 XT")]
    )

    assert ais_check.sriov_vf_gpus() == []


def test_non_guest_modes_not_flagged(monkeypatch, ais_check):
    # Only GUEST is a VF; HOST and PASSTHROUGH keep the fast path.
    _install_fake_amdsmi(
        monkeypatch,
        [
            _pf("0000:05:00.0", mode=_FakeVirtMode.HOST),
            _pf("0000:06:00.0", mode=_FakeVirtMode.PASSTHROUGH),
        ],
    )

    assert ais_check.sriov_vf_gpus() == []


def test_mixed_pf_and_vf(monkeypatch, ais_check):
    _install_fake_amdsmi(
        monkeypatch,
        [
            _pf("0000:05:00.0"),
            _vf("0000:06:00.0"),
            _vf("0000:07:00.0"),
        ],
    )

    assert ais_check.sriov_vf_gpus() == [
        ("0000:06:00.0", "AMD Instinct MI300X VF"),
        ("0000:07:00.0", "AMD Instinct MI300X VF"),
    ]


def test_missing_bdf_falls_back_to_gpu_index(monkeypatch, ais_check):
    _install_fake_amdsmi(
        monkeypatch,
        [{"mode": _FakeVirtMode.GUEST, "bdf": _RAISE, "market": "MI300X VF"}],
    )

    assert ais_check.sriov_vf_gpus() == [("gpu 0", "MI300X VF")]


def test_amdsmi_import_missing(monkeypatch, ais_check):
    # Setting the module to None makes `import amdsmi` raise ImportError.
    monkeypatch.setitem(sys.modules, "amdsmi", None)

    assert ais_check.sriov_vf_gpus() == []


def test_init_failure(monkeypatch, ais_check):
    _install_fake_amdsmi(
        monkeypatch, [_vf("0000:05:00.0")], init_exc=_FakeAmdSmiException("init failed")
    )

    assert ais_check.sriov_vf_gpus() == []


def test_handles_query_failure(monkeypatch, ais_check):
    _install_fake_amdsmi(
        monkeypatch,
        [_vf("0000:05:00.0")],
        handles_exc=_FakeAmdSmiException("no handles"),
    )

    assert ais_check.sriov_vf_gpus() == []


def test_per_gpu_mode_query_failure_is_skipped(monkeypatch, ais_check):
    # A GPU whose mode query throws is treated as not-a-VF, and doesn't abort the
    # scan of the remaining GPUs.
    _install_fake_amdsmi(
        monkeypatch,
        [
            {"mode": _RAISE, "bdf": "0000:05:00.0"},
            _vf("0000:06:00.0"),
        ],
    )

    assert ais_check.sriov_vf_gpus() == [("0000:06:00.0", "AMD Instinct MI300X VF")]


def test_no_gpus(monkeypatch, ais_check):
    _install_fake_amdsmi(monkeypatch, [])

    assert ais_check.sriov_vf_gpus() == []


# ---------------------------------------------------------------------------
# print_sriov_warning()
# ---------------------------------------------------------------------------


def test_warning_printed_for_vf(capsys, ais_check):
    ais_check.print_sriov_warning([("0000:05:00.0", "AMD Instinct MI300X VF")])

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "0000:05:00.0" in captured.err
    assert "AMD Instinct MI300X VF" in captured.err
    # The warning must go to stderr, not stdout.
    assert captured.out == ""


def test_no_warning_when_empty(capsys, ais_check):
    ais_check.print_sriov_warning([])

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""
