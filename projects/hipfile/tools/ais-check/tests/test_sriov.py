# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for the SR-IOV virtual function detection / warning."""

# Test functions are self-documenting by name; docstrings add noise.
# pylint: disable=missing-function-docstring

import json
from collections import namedtuple

_Completed = namedtuple("CompletedProcess", "returncode stdout")


def _patch_amd_smi(monkeypatch, ais_check, *, stdout="", returncode=0, exc=None):
    def fake_run(*_args, **_kwargs):
        if exc is not None:
            raise exc
        return _Completed(returncode, stdout)

    monkeypatch.setattr(ais_check.subprocess, "run", fake_run)


def _gpu(index, market_name, bdf):
    return {
        "gpu": index,
        "asic": {"market_name": market_name},
        "bus": {"bdf": bdf},
    }


def _payload(*gpus):
    return json.dumps({"gpu_data": list(gpus)})


# ---------------------------------------------------------------------------
# sriov_vf_gpus()
# ---------------------------------------------------------------------------


def test_single_vf_detected(monkeypatch, ais_check):
    _patch_amd_smi(
        monkeypatch,
        ais_check,
        stdout=_payload(_gpu(0, "AMD Instinct MI300X VF", "0000:05:00.0")),
    )

    assert ais_check.sriov_vf_gpus() == [("0000:05:00.0", "AMD Instinct MI300X VF")]


def test_physical_function_not_flagged(monkeypatch, ais_check):
    _patch_amd_smi(
        monkeypatch,
        ais_check,
        stdout=_payload(_gpu(0, "AMD Radeon RX 6800 XT", "0000:23:00.0")),
    )

    assert ais_check.sriov_vf_gpus() == []


def test_mixed_pf_and_vf(monkeypatch, ais_check):
    _patch_amd_smi(
        monkeypatch,
        ais_check,
        stdout=_payload(
            _gpu(0, "AMD Instinct MI300X", "0000:05:00.0"),
            _gpu(1, "AMD Instinct MI300X VF", "0000:06:00.0"),
            _gpu(2, "AMD Instinct MI300X VF", "0000:07:00.0"),
        ),
    )

    assert ais_check.sriov_vf_gpus() == [
        ("0000:06:00.0", "AMD Instinct MI300X VF"),
        ("0000:07:00.0", "AMD Instinct MI300X VF"),
    ]


def test_vf_suffix_requires_word_boundary(monkeypatch, ais_check):
    # A name that merely ends in the letters "VF" without a separating space is
    # not treated as a virtual function.
    _patch_amd_smi(
        monkeypatch,
        ais_check,
        stdout=_payload(_gpu(0, "AMD SomethingVF", "0000:05:00.0")),
    )

    assert ais_check.sriov_vf_gpus() == []


def test_missing_bdf_falls_back_to_gpu_index(monkeypatch, ais_check):
    _patch_amd_smi(
        monkeypatch,
        ais_check,
        stdout=json.dumps(
            {
                "gpu_data": [
                    {"gpu": 3, "asic": {"market_name": "AMD Instinct MI300X VF"}}
                ]
            }
        ),
    )

    assert ais_check.sriov_vf_gpus() == [("gpu 3", "AMD Instinct MI300X VF")]


def test_amd_smi_missing(monkeypatch, ais_check):
    _patch_amd_smi(monkeypatch, ais_check, exc=FileNotFoundError())

    assert ais_check.sriov_vf_gpus() == []


def test_non_zero_returncode(monkeypatch, ais_check):
    _patch_amd_smi(
        monkeypatch,
        ais_check,
        stdout=_payload(_gpu(0, "AMD Instinct MI300X VF", "0000:05:00.0")),
        returncode=1,
    )

    assert ais_check.sriov_vf_gpus() == []


def test_malformed_json(monkeypatch, ais_check):
    _patch_amd_smi(monkeypatch, ais_check, stdout="not json at all")

    assert ais_check.sriov_vf_gpus() == []


def test_empty_gpu_data(monkeypatch, ais_check):
    _patch_amd_smi(monkeypatch, ais_check, stdout=json.dumps({"gpu_data": []}))

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
