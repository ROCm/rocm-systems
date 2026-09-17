# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Full strict identity survives database classification, splitting and archives."""

from pathlib import Path

import pytest

from rocm_kpack.artifact_splitter import ArtifactSplitter, base_arch
from rocm_kpack.artifact_utils import (
    extract_architecture_from_target,
    write_artifact_manifest,
)
from rocm_kpack.binutils import Toolchain
from rocm_kpack.database_handlers import HipBLASLtHandler, MIOpenHandler, RocBLASHandler
from rocm_kpack.kpack import PackedKernelArchive
from rocm_kpack.tools.verify_artifacts import ArtifactVerifier


@pytest.mark.parametrize("target", ["gfx1250", "gfx1250-strict"])
@pytest.mark.parametrize(
    "handler,relative",
    [
        (RocBLASHandler(), "lib/rocblas/library/TensileLibrary_{target}.dat"),
        (HipBLASLtHandler(), "lib/hipblaslt/library/{target}/TensileManifest.txt"),
        (MIOpenHandler(), "share/miopen/db/{target}_256.db.txt"),
        (MIOpenHandler(), "share/miopen/db/{target}256.HIP.fdb.txt"),
        (MIOpenHandler(), "lib/libMIOpenCKGroupedConv_{target}.so"),
    ],
)
def test_database_identity(tmp_path, target, handler, relative):
    path = tmp_path / relative.format(target=target)
    path.parent.mkdir(parents=True)
    path.write_text("payload")
    assert handler.detect(path, tmp_path) == target


@pytest.mark.parametrize(
    "selected", [["gfx1250"], ["gfx1250-strict"], ["gfx1250", "gfx1250-strict"]]
)
def test_split_preserves_only_selected_database_payloads(tmp_path, selected):
    # A family-named build artifact can supply canonical per-target split files.
    source = tmp_path / "miopen_lib_gfx125X-all"
    prefix = "ml-libs/MIOpen/stage"
    stage = source / prefix
    db = stage / "share/miopen/db"
    db.mkdir(parents=True)
    for target in ["gfx1250", "gfx1250-strict"]:
        (db / f"{target}_256.db.txt").write_text(target)
    write_artifact_manifest(source, [prefix])
    dest = tmp_path / "split"
    splitter = ArtifactSplitter(
        artifact_prefix="miopen_lib",
        toolchain=Toolchain(),
        database_handlers=[MIOpenHandler()],
        gpu_targets=selected,
    )
    splitter.split(source, dest)
    assert {p.name for p in dest.iterdir() if p.is_dir()} == {"miopen_lib_generic"} | {
        f"miopen_lib_{t}" for t in selected
    }
    for target in selected:
        output = dest / f"miopen_lib_{target}" / prefix / "share/miopen/db"
        assert [p.name for p in output.iterdir()] == [f"{target}_256.db.txt"]
        assert (output / f"{target}_256.db.txt").read_text() == target
    assert not list((dest / "miopen_lib_generic").rglob("*.db.txt"))


def test_archive_and_bundler_identity(tmp_path):
    target = "gfx1250-strict"
    assert (
        extract_architecture_from_target(f"hipv4-amdgcn-amd-amdhsa--{target}") == target
    )
    assert base_arch(target + "-xnack-") == target
    archive = PackedKernelArchive("blas", target, [target])
    archive.add_kernel(
        archive.prepare_kernel("lib/kernel", target, b"synthetic code object")
    )
    path = tmp_path / f"blas_{target}.kpack"
    archive.finalize_archive()
    archive.write(path)
    assert path.stat().st_size > 0
    restored = PackedKernelArchive.read(path)
    assert restored.gfx_arch_family == target
    assert restored.gfx_arches == [target]
    assert set(restored.toc["lib/kernel"]) == {target}


@pytest.mark.parametrize(
    "artifact_target,payload_target",
    [("gfx1250", "gfx1250-strict"), ("gfx1250-strict", "gfx1250")],
)
def test_verifier_detects_base_strict_contamination(
    tmp_path, artifact_target, payload_target
):
    artifact = tmp_path / f"blas_lib_{artifact_target}"
    artifact.mkdir()
    (artifact / f"TensileLibrary_{payload_target}.dat").write_text("data")
    verifier = ArtifactVerifier(tmp_path, Toolchain())
    verifier._check_architecture_separation([artifact])
    assert not verifier.results[-1].passed


@pytest.mark.parametrize("target", ["gfx1250", "gfx1250-strict"])
def test_verifier_accepts_database_cu_count(tmp_path, target):
    artifact = tmp_path / f"miopen_lib_{target}"
    db = artifact / "stage/share/miopen/db"
    db.mkdir(parents=True)
    (db / f"{target}256.db.txt").write_text("data")
    verifier = ArtifactVerifier(tmp_path, Toolchain())
    verifier._check_architecture_separation([artifact])
    assert verifier.results[-1].passed


def test_verifier_detects_target_directory_contamination(tmp_path):
    artifact = tmp_path / "blas_lib_gfx1250"
    directory = artifact / "stage/lib/rocblas/library/gfx1250-strict"
    directory.mkdir(parents=True)
    (directory / "TensileManifest.txt").write_text("data")
    verifier = ArtifactVerifier(tmp_path, Toolchain())
    verifier._check_architecture_separation([artifact])
    assert not verifier.results[-1].passed


@pytest.mark.parametrize("valid", [True, False])
def test_verifier_checks_archives_under_component_stage(tmp_path, valid):
    target = "gfx1250-strict"
    artifact = tmp_path / f"rand_lib_{target}"
    directory = artifact / "math-libs/rocRAND/stage/.kpack"
    directory.mkdir(parents=True)
    path = directory / f"rand_lib_{target}.kpack"
    if valid:
        archive = PackedKernelArchive("rand_lib", target, [target])
        archive.add_kernel(archive.prepare_kernel("lib/kernel", target, b"payload"))
        archive.finalize_archive()
        archive.write(path)
    else:
        path.write_bytes(b"invalid archive")
    verifier = ArtifactVerifier(tmp_path, Toolchain())
    verifier._check_kpack_archives([artifact])
    result = verifier.results[-1]
    assert result.passed == valid
    assert any(path.name in detail for detail in result.details)


def test_verifier_allows_generic_manifest_only_directory(tmp_path):
    artifact = tmp_path / "rand_lib_generic"
    directory = artifact / "math-libs/rocRAND/stage/.kpack"
    directory.mkdir(parents=True)
    (directory / "rand_lib.kpm").write_bytes(b"host manifest")
    verifier = ArtifactVerifier(tmp_path, Toolchain())
    verifier._check_kpack_archives([artifact])
    assert verifier.results[-1].passed


@pytest.mark.parametrize(
    "artifact_target,payload_target,expected_pass",
    [
        ("gfx942-xnack+", "gfx942-xnack+", True),
        ("gfx942-xnack-", "gfx942-xnack-", True),
        ("gfx942", "gfx942-xnack+", True),
        ("gfx1250-strict-xnack+", "gfx1250-strict-xnack+", True),
        ("gfx1250-strict-xnack-", "gfx1250-strict-xnack-", True),
        ("gfx1250-xnack+", "gfx1250-strict-xnack+", False),
        ("gfx1250-strict-xnack+", "gfx1250-xnack+", False),
        ("gfx942-xnack+", "gfx90a-xnack+", False),
    ],
)
def test_verifier_normalizes_ck_features_without_losing_strict_identity(
    tmp_path, artifact_target, payload_target, expected_pass
):
    artifact = tmp_path / f"miopen_lib_{artifact_target}"
    library = artifact / "stage/lib" / f"libMIOpenCKGroupedConv_{payload_target}.so"
    library.parent.mkdir(parents=True)
    library.write_text("CK payload")
    verifier = ArtifactVerifier(tmp_path, Toolchain())
    verifier._check_architecture_separation([artifact])
    assert verifier.results[-1].passed == expected_pass
