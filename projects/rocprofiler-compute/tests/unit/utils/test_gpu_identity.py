# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Physical GPU identity normalization and matching without hardware."""

import importlib.util
import sys
from dataclasses import FrozenInstanceError

import pytest
from conftest import ProfileModeImportGuard

from utils import gpu_identity
from utils.gpu_identity import GpuIdentity, identity_from_specs, match_gpu_identity


@pytest.mark.parametrize(
    "missing",
    [None, "", "N/A", "none", float("nan"), -1, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF],
)
def test_missing_integer_metadata_is_unavailable(missing):
    identity = GpuIdentity(node_id=missing, bdf_id=missing)

    assert identity.node_id is None
    assert identity.bdf_id is None
    assert identity.spec_fields() == {}


@pytest.mark.parametrize("missing", [None, "", "GPU-XX", "N/A", "GPU-0000000000000000"])
def test_missing_uuid_metadata_is_unavailable(missing):
    assert GpuIdentity(uuid=missing).uuid is None


def test_normalized_identity_retains_zero_and_matches_metadata_types():
    identity = identity_from_specs({
        "gpu_arch": "gfx1151",
        "gpu_chip_id": 5510,
        "gpu_node_id": 0.0,
        "gpu_bdf_id": (2 << 32) | 512,
        "gpu_uuid": "gpu-ABCDEF0123456789",
    })

    assert identity == GpuIdentity("gfx1151", "5510", 0, 512, "GPU-abcdef0123456789")
    assert identity.spec_fields() == {
        "gpu_node_id": 0,
        "gpu_bdf_id": 512,
        "gpu_uuid": "GPU-abcdef0123456789",
    }
    with pytest.raises(FrozenInstanceError):
        identity.node_id = 2


@pytest.mark.parametrize(
    "selected, candidates, expected",
    [
        pytest.param(
            GpuIdentity(node_id=3, bdf_id=512),
            [("sibling", GpuIdentity(node_id=1)), ("selected", GpuIdentity(node_id=3))],
            "selected",
            id="node-match-with-missing-secondary-metadata",
        ),
        pytest.param(
            GpuIdentity(node_id=3, bdf_id=512),
            [
                ("sibling", GpuIdentity(node_id=1)),
                ("selected", GpuIdentity(bdf_id=512)),
            ],
            "selected",
            id="missing-candidate-node-permits-bdf-fallback",
        ),
        pytest.param(
            GpuIdentity(node_id=3, bdf_id=512),
            [
                ("sibling", GpuIdentity(bdf_id=256)),
                ("selected", GpuIdentity(bdf_id=512)),
            ],
            "selected",
            id="missing-adapter-node-permits-bdf-fallback",
        ),
        pytest.param(
            GpuIdentity(node_id=3, bdf_id=512, uuid="GPU-abcdef0123456789"),
            [("selected", GpuIdentity(uuid="gpu-ABCDEF0123456789"))],
            "selected",
            id="missing-adapter-topology-permits-uuid-fallback",
        ),
        pytest.param(
            GpuIdentity(),
            [("only-device", GpuIdentity())],
            "only-device",
            id="metadata-free-single-device",
        ),
    ],
)
def test_physical_identity_matches_unique_metadata(selected, candidates, expected):
    assert match_gpu_identity(selected, candidates, "test adapter") == expected


@pytest.mark.parametrize(
    "selected, candidates, diagnostic",
    [
        pytest.param(
            GpuIdentity(node_id=3, bdf_id=512),
            [("sibling", GpuIdentity(node_id=1, bdf_id=512))],
            "node_id",
            id="conflicting-node-cannot-use-matching-bdf",
        ),
        pytest.param(
            GpuIdentity(node_id=3, bdf_id=512),
            [("conflict", GpuIdentity(node_id=3, bdf_id=256))],
            "conflicting bdf_id",
            id="matching-node-cannot-hide-secondary-conflict",
        ),
        pytest.param(
            GpuIdentity(node_id=3, bdf_id=512),
            [("first", GpuIdentity(node_id=3)), ("second", GpuIdentity(node_id=3))],
            "ambiguous node_id",
            id="duplicate-node-is-ambiguous",
        ),
        pytest.param(
            GpuIdentity(node_id=3, bdf_id=512),
            [("first", GpuIdentity(bdf_id=512)), ("second", GpuIdentity(bdf_id=512))],
            "ambiguous bdf_id",
            id="duplicate-fallback-is-ambiguous",
        ),
        pytest.param(
            GpuIdentity(arch="gfx1151", node_id=3),
            [("other-arch", GpuIdentity(arch="gfx950", node_id=3))],
            "conflicting arch",
            id="architecture-conflict",
        ),
        pytest.param(
            GpuIdentity(chip_id="5510", node_id=3),
            [("other-chip", GpuIdentity(chip_id="30112", node_id=3))],
            "conflicting chip_id",
            id="chip-conflict",
        ),
        pytest.param(
            GpuIdentity(node_id=3),
            [("missing", GpuIdentity())],
            "lacks a unique",
            id="unverifiable-physical-identity",
        ),
        pytest.param(
            GpuIdentity(),
            [("first", GpuIdentity()), ("second", GpuIdentity())],
            "lacks a unique",
            id="metadata-free-multiple-devices",
        ),
    ],
)
def test_physical_identity_rejects_conflicts_and_ambiguity(
    selected, candidates, diagnostic
):
    with pytest.raises(ValueError, match=diagnostic):
        match_gpu_identity(selected, candidates, "test adapter")


def test_identity_module_imports_with_profile_dependency_guard(monkeypatch):
    module_spec = importlib.util.spec_from_file_location(
        "utils.gpu_identity_import_probe", gpu_identity.__file__
    )
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, module_spec.name, module)

    with ProfileModeImportGuard():
        module_spec.loader.exec_module(module)
        identity = module.identity_from_specs({"gpu_node_id": 3})

    assert identity.node_id == 3
