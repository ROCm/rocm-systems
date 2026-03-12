"""
Tests for the wheel splitter.

Unit tests verify metadata parsing and generation without requiring a toolchain.
Integration tests use real fat binaries from test_assets to verify the full
split pipeline.
"""

import hashlib
import os
import shutil
from pathlib import Path

import pytest

from rocm_kpack.wheel_splitter import (
    FatBinaryInfo,
    InvalidWheelError,
    NoFatBinariesError,
    SplitResult,
    WheelIdentity,
    WheelSplitter,
    WheelSplitError,
    _arch_to_bundle_key,
    _bundle_key_to_dist_name,
    compute_dist_info_name,
    compute_wheel_filename,
    generate_device_metadata,
    generate_record,
    generate_wheel_file,
    parse_wheel_identity,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_wheel(tmp_path: Path):
    """Create a minimal mock wheel directory structure (no fat binaries)."""
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()

    # Create torch/ overlay root with some files
    torch_dir = wheel_dir / "torch"
    torch_dir.mkdir()
    (torch_dir / "__init__.py").write_text("# torch\n")
    lib_dir = torch_dir / "lib"
    lib_dir.mkdir()
    (lib_dir / "libtorch_cpu.so").write_bytes(b"\x00" * 100)

    # Create dist-info
    dist_info = wheel_dir / "torch-2.10.0+rocm7.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.4\n"
        "Name: torch\n"
        "Version: 2.10.0+rocm7.1\n"
        "Summary: Tensors and Dynamic neural networks\n"
    )
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\n"
        "Generator: setuptools 80.9.0\n"
        "Root-Is-Purelib: false\n"
        "Tag: cp313-cp313-manylinux_2_28_x86_64\n"
    )
    (dist_info / "RECORD").write_text("")
    (dist_info / "top_level.txt").write_text("torch\n")

    return wheel_dir


@pytest.fixture
def fat_binary_wheel(tmp_path: Path, test_assets_dir: Path):
    """Create a mock wheel directory with a real fat binary from test assets."""
    wheel_dir = tmp_path / "fat_wheel"
    wheel_dir.mkdir()

    # Create torch/ overlay root
    torch_dir = wheel_dir / "torch"
    torch_dir.mkdir()
    (torch_dir / "__init__.py").write_text("# torch\n")
    lib_dir = torch_dir / "lib"
    lib_dir.mkdir()

    # Copy a real fat binary as a simulated torch library
    source_binary = (
        test_assets_dir / "bundled_binaries/linux/cov5/libtest_kernel_single.so"
    )
    if not source_binary.exists():
        pytest.skip("Test assets not available")
    shutil.copy2(source_binary, lib_dir / "libfat_test.so")

    # Also add a non-fat binary
    (lib_dir / "libplain.so").write_bytes(b"\x00" * 100)

    # Create dist-info
    dist_info = wheel_dir / "torch-2.10.0+rocm7.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.4\n"
        "Name: torch\n"
        "Version: 2.10.0+rocm7.1\n"
        "Summary: Tensors and Dynamic neural networks\n"
    )
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\n"
        "Generator: setuptools 80.9.0\n"
        "Root-Is-Purelib: false\n"
        "Tag: cp313-cp313-manylinux_2_28_x86_64\n"
    )
    (dist_info / "RECORD").write_text("")
    (dist_info / "top_level.txt").write_text("torch\n")

    return wheel_dir


# =============================================================================
# Unit tests — no toolchain needed
# =============================================================================


class TestParseWheelIdentity:
    def test_basic(self, mock_wheel: Path):
        identity = parse_wheel_identity(mock_wheel)
        assert identity.name == "torch"
        assert identity.version == "2.10.0+rocm7.1"
        assert identity.python_tag == "cp313"
        assert identity.abi_tag == "cp313"
        assert identity.platform_tag == "manylinux_2_28_x86_64"
        assert identity.dist_info_name == "torch-2.10.0+rocm7.1.dist-info"

    def test_no_dist_info(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(InvalidWheelError, match="Expected exactly one .dist-info"):
            parse_wheel_identity(empty_dir)

    def test_multiple_dist_info(self, tmp_path: Path):
        wheel_dir = tmp_path / "multi"
        wheel_dir.mkdir()
        (wheel_dir / "a-1.0.dist-info").mkdir()
        (wheel_dir / "b-2.0.dist-info").mkdir()
        with pytest.raises(InvalidWheelError, match="Expected exactly one .dist-info"):
            parse_wheel_identity(wheel_dir)

    def test_missing_metadata(self, tmp_path: Path):
        wheel_dir = tmp_path / "no_meta"
        wheel_dir.mkdir()
        dist_info = wheel_dir / "pkg-1.0.dist-info"
        dist_info.mkdir()
        (dist_info / "WHEEL").write_text("Tag: cp313-cp313-linux_x86_64\n")
        with pytest.raises(InvalidWheelError, match="No METADATA"):
            parse_wheel_identity(wheel_dir)

    def test_missing_wheel_file(self, tmp_path: Path):
        wheel_dir = tmp_path / "no_wheel"
        wheel_dir.mkdir()
        dist_info = wheel_dir / "pkg-1.0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text("Name: pkg\nVersion: 1.0\n")
        with pytest.raises(InvalidWheelError, match="No WHEEL"):
            parse_wheel_identity(wheel_dir)


class TestComputeWheelFilename:
    def test_basic(self):
        identity = WheelIdentity(
            name="torch",
            version="2.10.0+rocm7.1",
            python_tag="cp313",
            abi_tag="cp313",
            platform_tag="manylinux_2_28_x86_64",
            dist_info_name="torch-2.10.0+rocm7.1.dist-info",
        )
        filename = compute_wheel_filename(
            "amd-torch-device-gfx942", "2.10.0+rocm7.1", identity
        )
        assert filename == (
            "amd_torch_device_gfx942-2.10.0+rocm7.1"
            "-cp313-cp313-manylinux_2_28_x86_64.whl"
        )

    def test_underscore_normalization(self):
        identity = WheelIdentity(
            name="pkg",
            version="1.0",
            python_tag="py3",
            abi_tag="none",
            platform_tag="any",
            dist_info_name="pkg-1.0.dist-info",
        )
        filename = compute_wheel_filename("my-package", "1.0", identity)
        assert filename == "my_package-1.0-py3-none-any.whl"


class TestComputeDistInfoName:
    def test_basic(self):
        assert (
            compute_dist_info_name("amd-torch-device-gfx942", "2.10.0+rocm7.1")
            == "amd_torch_device_gfx942-2.10.0+rocm7.1.dist-info"
        )


class TestGenerateDeviceMetadata:
    def test_target_level(self):
        identity = WheelIdentity(
            name="torch",
            version="2.10.0+rocm7.1",
            python_tag="cp313",
            abi_tag="cp313",
            platform_tag="manylinux_2_28_x86_64",
            dist_info_name="torch-2.10.0+rocm7.1.dist-info",
        )
        metadata = generate_device_metadata(identity, "gfx942", "amd-torch-device", [])
        assert "Name: amd-torch-device-gfx942" in metadata
        assert "Version: 2.10.0+rocm7.1" in metadata
        assert "Requires-Dist: torch == 2.10.0+rocm7.1" in metadata

    def test_family_level(self):
        identity = WheelIdentity(
            name="torch",
            version="2.10.0+rocm7.1",
            python_tag="cp313",
            abi_tag="cp313",
            platform_tag="manylinux_2_28_x86_64",
            dist_info_name="torch-2.10.0+rocm7.1.dist-info",
        )
        metadata = generate_device_metadata(identity, "gfx11", "amd-torch-device", [])
        assert "Name: amd-torch-device-gfx11" in metadata

    def test_sub_family_level(self):
        identity = WheelIdentity(
            name="torch",
            version="2.10.0+rocm7.1",
            python_tag="cp313",
            abi_tag="cp313",
            platform_tag="manylinux_2_28_x86_64",
            dist_info_name="torch-2.10.0+rocm7.1.dist-info",
        )
        metadata = generate_device_metadata(identity, "gfx12_0", "amd-torch-device", [])
        # gfx12_0 becomes gfx12-0 in dist name (underscore to hyphen)
        assert "Name: amd-torch-device-gfx12-0" in metadata

    def test_with_device_requires_dist(self):
        identity = WheelIdentity(
            name="torch",
            version="2.10.0+rocm7.1",
            python_tag="cp313",
            abi_tag="cp313",
            platform_tag="manylinux_2_28_x86_64",
            dist_info_name="torch-2.10.0+rocm7.1.dist-info",
        )
        metadata = generate_device_metadata(
            identity,
            "gfx942",
            "amd-torch-device",
            ["rocm-sdk-device-@GFXARCH@ == 7.1"],
        )
        assert "Requires-Dist: rocm-sdk-device-gfx942 == 7.1" in metadata

    def test_gfxarch_placeholder_expansion(self):
        identity = WheelIdentity(
            name="torch",
            version="1.0",
            python_tag="py3",
            abi_tag="none",
            platform_tag="any",
            dist_info_name="torch-1.0.dist-info",
        )
        metadata = generate_device_metadata(
            identity,
            "gfx1100",
            "amd-torch-device",
            ["some-dep-@GFXARCH@ >= 2.0"],
        )
        assert "Requires-Dist: some-dep-gfx1100 >= 2.0" in metadata


class TestArchToBundleKey:
    def test_bare_arch(self):
        assert _arch_to_bundle_key("gfx942") == "gfx942"

    def test_xnack_plus(self):
        assert _arch_to_bundle_key("gfx942:xnack+") == "gfx942"

    def test_xnack_minus(self):
        assert _arch_to_bundle_key("gfx90a:xnack-") == "gfx90a"

    def test_no_xnack(self):
        assert _arch_to_bundle_key("gfx1100") == "gfx1100"

    def test_family_key(self):
        assert _arch_to_bundle_key("gfx11") == "gfx11"

    def test_sub_family_key(self):
        assert _arch_to_bundle_key("gfx12_0") == "gfx12_0"


class TestBundleKeyToDistName:
    def test_target_level(self):
        assert (
            _bundle_key_to_dist_name("amd-torch-device", "gfx942")
            == "amd-torch-device-gfx942"
        )

    def test_family_level(self):
        assert (
            _bundle_key_to_dist_name("amd-torch-device", "gfx11")
            == "amd-torch-device-gfx11"
        )

    def test_sub_family_level(self):
        # Underscore in bundle key becomes hyphen in dist name
        assert (
            _bundle_key_to_dist_name("amd-torch-device", "gfx12_0")
            == "amd-torch-device-gfx12-0"
        )

    def test_various_targets(self):
        cases = [
            ("gfx900", "amd-torch-device-gfx900"),
            ("gfx90a", "amd-torch-device-gfx90a"),
            ("gfx1100", "amd-torch-device-gfx1100"),
            ("gfx1151", "amd-torch-device-gfx1151"),
            ("gfx9", "amd-torch-device-gfx9"),
            ("gfx9_4", "amd-torch-device-gfx9-4"),
        ]
        for bundle_key, expected in cases:
            assert (
                _bundle_key_to_dist_name("amd-torch-device", bundle_key) == expected
            ), f"Failed for {bundle_key}"


class TestRewriteHostMetadata:
    """Test _rewrite_host_metadata via WheelSplitter."""

    def _make_host_staging(self, tmp_path: Path) -> tuple[Path, WheelIdentity]:
        """Create a minimal host staging directory with METADATA."""
        staging = tmp_path / "host_staging"
        staging.mkdir()
        dist_info = staging / "torch-2.10.0+rocm7.1.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.4\n"
            "Name: torch\n"
            "Version: 2.10.0+rocm7.1\n"
            "Summary: Tensors and Dynamic neural networks\n"
            "\n"
            "A description body that comes after the header section.\n"
        )
        identity = WheelIdentity(
            name="torch",
            version="2.10.0+rocm7.1",
            python_tag="cp313",
            abi_tag="cp313",
            platform_tag="manylinux_2_28_x86_64",
            dist_info_name="torch-2.10.0+rocm7.1.dist-info",
        )
        return staging, identity

    def _make_splitter(self) -> WheelSplitter:
        from rocm_kpack.binutils import Toolchain

        return WheelSplitter(
            device_package_prefix="amd-torch-device",
            overlay_root="torch/",
            toolchain=Toolchain(),
        )

    def test_rocm_bootstrap_dep(self, tmp_path: Path):
        staging, identity = self._make_host_staging(tmp_path)
        splitter = self._make_splitter()
        splitter._rewrite_host_metadata(staging, identity, {"gfx942"})

        metadata = (staging / identity.dist_info_name / "METADATA").read_text()
        assert "Requires-Dist: rocm-bootstrap" in metadata

    def test_extras_for_single_target(self, tmp_path: Path):
        staging, identity = self._make_host_staging(tmp_path)
        splitter = self._make_splitter()
        # gfx942 packaging chain: gfx942 -> gfx9_4 -> gfx9
        # Only gfx942 has a device wheel (is in bundle_keys)
        splitter._rewrite_host_metadata(staging, identity, {"gfx942"})

        metadata = (staging / identity.dist_info_name / "METADATA").read_text()
        assert "Provides-Extra: gfx942" in metadata
        assert (
            'Requires-Dist: amd-torch-device-gfx942 == 2.10.0+rocm7.1; extra == "gfx942"'
            in metadata
        )
        assert "Provides-Extra: all" in metadata

    def test_classic_has_no_variant_markers(self, tmp_path: Path):
        staging, identity = self._make_host_staging(tmp_path)
        splitter = self._make_splitter()
        splitter._rewrite_host_metadata(staging, identity, {"gfx942"})

        metadata = (staging / identity.dist_info_name / "METADATA").read_text()
        assert "variant_properties" not in metadata

    def test_variant_markers_when_enabled(self, tmp_path: Path):
        staging, identity = self._make_host_staging(tmp_path)
        splitter = self._make_splitter()
        splitter._rewrite_host_metadata(
            staging, identity, {"gfx942"}, include_variant_markers=True
        )

        metadata = (staging / identity.dist_info_name / "METADATA").read_text()
        assert (
            "Requires-Dist: amd-torch-device-gfx942 == 2.10.0+rocm7.1; "
            '"amd :: gfx_arch :: gfx942" in variant_properties'
        ) in metadata

    def test_chain_with_family_wheel(self, tmp_path: Path):
        staging, identity = self._make_host_staging(tmp_path)
        splitter = self._make_splitter()
        # gfx1100 chain: gfx1100 -> gfx11_0 -> gfx11
        # If gfx11 and gfx1100 both have device wheels:
        splitter._rewrite_host_metadata(staging, identity, {"gfx1100", "gfx11"})

        metadata = (staging / identity.dist_info_name / "METADATA").read_text()
        # Both should appear as deps under the gfx1100 extra
        assert (
            'Requires-Dist: amd-torch-device-gfx1100 == 2.10.0+rocm7.1; extra == "gfx1100"'
            in metadata
        )
        assert (
            'Requires-Dist: amd-torch-device-gfx11 == 2.10.0+rocm7.1; extra == "gfx1100"'
            in metadata
        )
        # "all" should include both
        assert (
            'Requires-Dist: amd-torch-device-gfx11 == 2.10.0+rocm7.1; extra == "all"'
            in metadata
        )
        assert (
            'Requires-Dist: amd-torch-device-gfx1100 == 2.10.0+rocm7.1; extra == "all"'
            in metadata
        )

    def test_family_only_not_target_extra(self, tmp_path: Path):
        """Family-level bundle keys don't get their own Provides-Extra."""
        staging, identity = self._make_host_staging(tmp_path)
        splitter = self._make_splitter()
        # gfx11 is a family, not a target - no Provides-Extra for it directly
        splitter._rewrite_host_metadata(staging, identity, {"gfx11"})

        metadata = (staging / identity.dist_info_name / "METADATA").read_text()
        # No target-level extras since gfx11 is a family
        assert "Provides-Extra: gfx11" not in metadata
        # But "all" should still include it
        assert "Provides-Extra: all" in metadata
        assert (
            'Requires-Dist: amd-torch-device-gfx11 == 2.10.0+rocm7.1; extra == "all"'
            in metadata
        )

    def test_preserves_existing_metadata(self, tmp_path: Path):
        staging, identity = self._make_host_staging(tmp_path)
        splitter = self._make_splitter()
        splitter._rewrite_host_metadata(staging, identity, {"gfx942"})

        metadata = (staging / identity.dist_info_name / "METADATA").read_text()
        # Original fields should still be there
        assert "Name: torch" in metadata
        assert "Version: 2.10.0+rocm7.1" in metadata
        assert "Summary: Tensors and Dynamic neural networks" in metadata

    def test_headers_before_body(self, tmp_path: Path):
        staging, identity = self._make_host_staging(tmp_path)
        splitter = self._make_splitter()
        splitter._rewrite_host_metadata(staging, identity, {"gfx942"})

        metadata = (staging / identity.dist_info_name / "METADATA").read_text()
        # Split into header and body at the first blank line
        header_body = metadata.split("\n\n", 1)
        assert len(header_body) == 2, "METADATA should have header and body sections"
        header_section, body_section = header_body
        # All injected headers must be in the header section, not the body
        assert "Requires-Dist: rocm-bootstrap" in header_section
        assert "Provides-Extra: gfx942" in header_section
        assert "Provides-Extra: all" in header_section
        # Body should still contain the description
        assert "description body" in body_section


class TestVariantWheel:
    """Tests for PEP 817 variant wheel generation."""

    def _make_host_staging(self, tmp_path: Path) -> tuple[Path, WheelIdentity]:
        staging = tmp_path / "host_staging"
        staging.mkdir()
        dist_info = staging / "torch-2.10.0+rocm7.1.dist-info"
        dist_info.mkdir()
        # Write a classic-rewritten METADATA (extras only, no variant markers)
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.4\n"
            "Name: torch\n"
            "Version: 2.10.0+rocm7.1\n"
            "Requires-Dist: rocm-bootstrap\n"
            "Provides-Extra: gfx942\n"
            'Requires-Dist: amd-torch-device-gfx942 == 2.10.0+rocm7.1; extra == "gfx942"\n'
            "Provides-Extra: all\n"
            'Requires-Dist: amd-torch-device-gfx942 == 2.10.0+rocm7.1; extra == "all"\n'
            "\n"
            "Description body.\n"
        )
        (dist_info / "WHEEL").write_text(
            "Wheel-Version: 1.0\nTag: cp313-cp313-manylinux_2_28_x86_64\n"
        )
        (dist_info / "RECORD").write_text("")
        # Add a dummy file so the wheel isn't empty
        (staging / "torch").mkdir()
        (staging / "torch" / "__init__.py").write_text("")
        identity = WheelIdentity(
            name="torch",
            version="2.10.0+rocm7.1",
            python_tag="cp313",
            abi_tag="cp313",
            platform_tag="manylinux_2_28_x86_64",
            dist_info_name="torch-2.10.0+rocm7.1.dist-info",
        )
        return staging, identity

    def _make_splitter(self) -> WheelSplitter:
        from rocm_kpack.binutils import Toolchain

        return WheelSplitter(
            device_package_prefix="amd-torch-device",
            overlay_root="torch/",
            toolchain=Toolchain(),
            generate_variant_wheel=True,
        )

    def test_add_variant_markers(self, tmp_path: Path):
        staging, identity = self._make_host_staging(tmp_path)
        splitter = self._make_splitter()
        splitter._add_variant_markers_to_metadata(staging, identity, {"gfx942"})

        metadata = (staging / identity.dist_info_name / "METADATA").read_text()
        # Should have the extras line AND the variant marker line
        assert 'extra == "gfx942"' in metadata
        assert ('"amd :: gfx_arch :: gfx942" in variant_properties') in metadata
        # "all" extra should NOT get variant markers
        assert '"amd :: gfx_arch :: all"' not in metadata

    def test_variant_json(self, tmp_path: Path):
        import json

        staging, identity = self._make_host_staging(tmp_path)
        splitter = self._make_splitter()
        splitter._write_variant_json(staging, identity, {"gfx942", "gfx11"})

        variant_path = staging / identity.dist_info_name / "variant.json"
        assert variant_path.exists()
        data = json.loads(variant_path.read_text())
        assert "providers" in data
        assert "amd" in data["providers"]
        assert "variants" in data
        # gfx942 is a target, gfx11 is a family — only targets get variants
        assert "gfx942" in data["variants"]
        assert "gfx11" not in data["variants"]

    def test_create_variant_wheel_directory(self, tmp_path: Path):
        import json

        staging, identity = self._make_host_staging(tmp_path)
        splitter = self._make_splitter()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        variant_path = splitter._create_variant_wheel(
            staging, identity, {"gfx942"}, output_dir, "directory"
        )

        assert variant_path.exists()
        assert "variant" in variant_path.name

        # Check METADATA has variant markers
        metadata = (variant_path / identity.dist_info_name / "METADATA").read_text()
        assert "variant_properties" in metadata
        assert 'extra == "gfx942"' in metadata

        # Check variant.json exists
        variant_json = variant_path / identity.dist_info_name / "variant.json"
        assert variant_json.exists()
        data = json.loads(variant_json.read_text())
        assert "gfx942" in data["variants"]


class TestGenerateWheelFile:
    def test_basic(self):
        identity = WheelIdentity(
            name="torch",
            version="2.10.0+rocm7.1",
            python_tag="cp313",
            abi_tag="cp313",
            platform_tag="manylinux_2_28_x86_64",
            dist_info_name="torch-2.10.0+rocm7.1.dist-info",
        )
        content = generate_wheel_file(identity)
        assert "Wheel-Version: 1.0" in content
        assert "Tag: cp313-cp313-manylinux_2_28_x86_64" in content
        assert "Root-Is-Purelib: false" in content
        assert "Generator: rocm-kpack-split-python-wheels" in content


class TestGenerateRecord:
    def test_basic(self, tmp_path: Path):
        wheel_dir = tmp_path / "wheel"
        wheel_dir.mkdir()
        dist_info = wheel_dir / "pkg-1.0.dist-info"
        dist_info.mkdir()

        # Create a file with known content
        test_content = b"hello world"
        (wheel_dir / "test.py").write_bytes(test_content)
        expected_hash = hashlib.sha256(test_content).hexdigest()

        # Create empty RECORD
        (dist_info / "RECORD").write_text("")

        record = generate_record(wheel_dir, "pkg-1.0.dist-info")
        lines = record.strip().split("\n")

        # Should have 2 entries: test.py and RECORD itself
        assert len(lines) == 2

        # RECORD entry has no hash
        record_line = [l for l in lines if "RECORD" in l][0]
        assert record_line == "pkg-1.0.dist-info/RECORD,,"

        # test.py has hash
        test_line = [l for l in lines if "test.py" in l][0]
        assert f"sha256={expected_hash}" in test_line
        assert f",{len(test_content)}" in test_line


class TestKpackSearchPattern:
    """Test the search pattern computation logic."""

    def _compute(self, overlay_relative_path: str, group_name: str) -> str:
        """Helper to call the private method via a throwaway splitter."""
        from rocm_kpack.binutils import Toolchain

        splitter = WheelSplitter(
            device_package_prefix="test",
            overlay_root="torch/",
            toolchain=Toolchain(),
        )
        return splitter._compute_kpack_search_pattern(overlay_relative_path, group_name)

    def test_binary_in_subdir(self):
        # lib/libtorch_hip.so -> depth=1 -> ../.kpack/torch_@GFXARCH@.kpack
        pattern = self._compute("lib/libtorch_hip.so", "torch")
        assert pattern == "../.kpack/torch_@GFXARCH@.kpack"

    def test_binary_at_root(self):
        # libtorch_hip.so -> depth=0 -> .kpack/torch_@GFXARCH@.kpack
        pattern = self._compute("libtorch_hip.so", "torch")
        assert pattern == ".kpack/torch_@GFXARCH@.kpack"

    def test_binary_deeply_nested(self):
        # a/b/c/binary.so -> depth=3 -> ../../../.kpack/torch_@GFXARCH@.kpack
        pattern = self._compute("a/b/c/binary.so", "torch")
        assert pattern == "../../../.kpack/torch_@GFXARCH@.kpack"


# =============================================================================
# Integration tests — require toolchain and test assets
# =============================================================================


class TestWheelSplitterIntegration:
    """Integration tests that use real fat binaries from test_assets."""

    def test_split_directory_to_directory(
        self, fat_binary_wheel: Path, toolchain, tmp_path: Path
    ):
        """Test splitting an exploded wheel to exploded output directories."""
        output_dir = tmp_path / "output"

        splitter = WheelSplitter(
            device_package_prefix="amd-torch-device",
            overlay_root="torch/",
            toolchain=toolchain,
            verbose=True,
        )

        result = splitter.split(fat_binary_wheel, output_dir, output_format="directory")

        # Should have found architectures
        assert len(result.architectures_found) > 0
        assert result.fat_binaries_processed == 1

        # Host wheel directory should exist
        assert result.host_wheel_path.exists()
        assert result.host_wheel_path.is_dir()

        # Host should have the original files (minus device code)
        host_init = result.host_wheel_path / "torch" / "__init__.py"
        assert host_init.exists()

        # Host fat binary should still exist but be transformed
        host_fat = result.host_wheel_path / "torch" / "lib" / "libfat_test.so"
        assert host_fat.exists()

        # Host RECORD should exist and be non-empty
        host_record = (
            result.host_wheel_path / "torch-2.10.0+rocm7.1.dist-info" / "RECORD"
        )
        assert host_record.exists()
        assert host_record.stat().st_size > 0

        # Device wheels should exist for each bundle key found
        for bundle_key, device_path in result.device_wheel_paths.items():
            assert device_path.exists()
            assert device_path.is_dir()

            # Device wheel should have at least one kpack file
            kpack_dir = device_path / "torch" / ".kpack"
            kpack_files = list(kpack_dir.glob("*.kpack"))
            assert len(kpack_files) > 0
            for kf in kpack_files:
                assert kf.stat().st_size > 0

            # Device wheel should have dist-info with correct naming
            device_name = _bundle_key_to_dist_name("amd-torch-device", bundle_key)
            device_dist_info_name = (
                f"{device_name.replace('-', '_')}-2.10.0+rocm7.1.dist-info"
            )
            device_dist_info = device_path / device_dist_info_name
            assert device_dist_info.exists()

            # Check METADATA
            metadata = (device_dist_info / "METADATA").read_text()
            assert f"Name: {device_name}" in metadata
            assert "Requires-Dist: torch == 2.10.0+rocm7.1" in metadata

            # Check WHEEL
            wheel_content = (device_dist_info / "WHEEL").read_text()
            assert "Tag: cp313-cp313-manylinux_2_28_x86_64" in wheel_content

            # Check RECORD
            record = (device_dist_info / "RECORD").read_text()
            assert ".kpack/" in record

    def test_split_no_fat_binaries(self, mock_wheel: Path, toolchain, tmp_path: Path):
        """Test that splitting a wheel with no fat binaries raises an error."""
        output_dir = tmp_path / "output"

        splitter = WheelSplitter(
            device_package_prefix="amd-torch-device",
            overlay_root="torch/",
            toolchain=toolchain,
        )

        with pytest.raises(NoFatBinariesError):
            splitter.split(mock_wheel, output_dir)

    def test_split_invalid_overlay_root(
        self, mock_wheel: Path, toolchain, tmp_path: Path
    ):
        """Test that an invalid overlay root raises an error."""
        output_dir = tmp_path / "output"

        splitter = WheelSplitter(
            device_package_prefix="amd-torch-device",
            overlay_root="nonexistent/",
            toolchain=toolchain,
        )

        with pytest.raises(InvalidWheelError, match="Overlay root"):
            splitter.split(mock_wheel, output_dir)

    def test_host_record_integrity(
        self, fat_binary_wheel: Path, toolchain, tmp_path: Path
    ):
        """Verify that the host wheel's RECORD has valid hashes."""
        output_dir = tmp_path / "output"

        splitter = WheelSplitter(
            device_package_prefix="amd-torch-device",
            overlay_root="torch/",
            toolchain=toolchain,
        )

        result = splitter.split(fat_binary_wheel, output_dir, output_format="directory")

        host_dir = result.host_wheel_path
        dist_info_name = "torch-2.10.0+rocm7.1.dist-info"
        record_path = host_dir / dist_info_name / "RECORD"
        record_text = record_path.read_text()

        for line in record_text.strip().split("\n"):
            if not line:
                continue
            parts = line.split(",")
            rel_path = parts[0]

            if parts[1] == "":
                # RECORD itself — no hash
                assert rel_path.endswith("RECORD")
                continue

            # Verify hash
            hash_part = parts[1]
            size_part = parts[2]
            assert hash_part.startswith("sha256=")
            expected_hash = hash_part.removeprefix("sha256=")

            file_path = host_dir / rel_path
            assert file_path.exists(), f"RECORD references missing file: {rel_path}"
            actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}"
            assert int(size_part) == file_path.stat().st_size

    def test_device_requires_dist(
        self, fat_binary_wheel: Path, toolchain, tmp_path: Path
    ):
        """Test that --device-requires-dist with @GFXARCH@ expands correctly."""
        output_dir = tmp_path / "output"

        splitter = WheelSplitter(
            device_package_prefix="amd-torch-device",
            overlay_root="torch/",
            toolchain=toolchain,
            device_requires_dist=["rocm-sdk-device-@GFXARCH@ == 7.1"],
        )

        result = splitter.split(fat_binary_wheel, output_dir, output_format="directory")

        for bundle_key, device_path in result.device_wheel_paths.items():
            device_name = _bundle_key_to_dist_name("amd-torch-device", bundle_key)
            dist_info_name = f"{device_name.replace('-', '_')}-2.10.0+rocm7.1.dist-info"
            metadata = (device_path / dist_info_name / "METADATA").read_text()
            assert f"Requires-Dist: rocm-sdk-device-{bundle_key} == 7.1" in metadata

    def test_split_parallel(self, fat_binary_wheel: Path, toolchain, tmp_path: Path):
        """Test that parallel splitting (jobs=2) produces correct results."""
        output_dir = tmp_path / "output"

        splitter = WheelSplitter(
            device_package_prefix="amd-torch-device",
            overlay_root="torch/",
            toolchain=toolchain,
            verbose=True,
            jobs=2,
        )

        result = splitter.split(fat_binary_wheel, output_dir, output_format="directory")

        # Same assertions as test_split_directory_to_directory
        assert len(result.architectures_found) > 0
        assert result.fat_binaries_processed == 1
        assert result.host_wheel_path.exists()
        assert result.host_wheel_path.is_dir()

        host_fat = result.host_wheel_path / "torch" / "lib" / "libfat_test.so"
        assert host_fat.exists()

        host_record = (
            result.host_wheel_path / "torch-2.10.0+rocm7.1.dist-info" / "RECORD"
        )
        assert host_record.exists()
        assert host_record.stat().st_size > 0

        for bundle_key, device_path in result.device_wheel_paths.items():
            assert device_path.exists()
            kpack_dir = device_path / "torch" / ".kpack"
            kpack_files = list(kpack_dir.glob("*.kpack"))
            assert len(kpack_files) > 0
            for kf in kpack_files:
                assert kf.stat().st_size > 0
