"""Tests for rocm_bootstrap.naming — the naming contract.

These tests lock down the dist-safe and module-safe naming for every bundle
at every level of the hierarchy. If a test here fails, something has changed
that could break downstream packaging.
"""

import pytest

from rocm_bootstrap.naming import (
    PackageNames,
    bundle_names,
    device_dist_name,
    device_module_name,
    is_valid_dist_name,
    is_valid_module_name,
)
from rocm_bootstrap.targets import (
    ALL_FAMILIES,
    ALL_SUB_FAMILIES,
    ALL_TARGETS,
    PackagingLevel,
    all_bundles,
    lookup_bundle,
)


# ---------------------------------------------------------------------------
# Explicit naming expectations for each bundle level
# ---------------------------------------------------------------------------


class TestFamilyNames:
    @pytest.mark.parametrize(
        "key,expected_dist,expected_module",
        [
            ("gfx9", "gfx9", "gfx9"),
            ("gfx10", "gfx10", "gfx10"),
            ("gfx11", "gfx11", "gfx11"),
            ("gfx12", "gfx12", "gfx12"),
        ],
    )
    def test_family_names(self, key, expected_dist, expected_module):
        b = lookup_bundle(key)
        names = bundle_names(b)
        assert names.dist_name == expected_dist
        assert names.module_name == expected_module


class TestSubFamilyNames:
    @pytest.mark.parametrize(
        "key,expected_dist,expected_module",
        [
            ("gfx9_0", "gfx9-0", "gfx9_0"),
            ("gfx9_4", "gfx9-4", "gfx9_4"),
            ("gfx9_5", "gfx9-5", "gfx9_5"),
            ("gfx10_1", "gfx10-1", "gfx10_1"),
            ("gfx10_3", "gfx10-3", "gfx10_3"),
            ("gfx11_0", "gfx11-0", "gfx11_0"),
            ("gfx11_5", "gfx11-5", "gfx11_5"),
            ("gfx12_0", "gfx12-0", "gfx12_0"),
            ("gfx12_5", "gfx12-5", "gfx12_5"),
        ],
    )
    def test_sub_family_names(self, key, expected_dist, expected_module):
        b = lookup_bundle(key)
        names = bundle_names(b)
        assert names.dist_name == expected_dist
        assert names.module_name == expected_module


class TestTargetNames:
    @pytest.mark.parametrize(
        "name,expected_dist,expected_module",
        [
            ("gfx900", "gfx900", "gfx900"),
            ("gfx90a", "gfx90a", "gfx90a"),
            ("gfx90c", "gfx90c", "gfx90c"),
            ("gfx908", "gfx908", "gfx908"),
            ("gfx942", "gfx942", "gfx942"),
            ("gfx950", "gfx950", "gfx950"),
            ("gfx1010", "gfx1010", "gfx1010"),
            ("gfx1030", "gfx1030", "gfx1030"),
            ("gfx1100", "gfx1100", "gfx1100"),
            ("gfx1151", "gfx1151", "gfx1151"),
            ("gfx1201", "gfx1201", "gfx1201"),
            ("gfx1250", "gfx1250", "gfx1250"),
        ],
    )
    def test_target_names(self, name, expected_dist, expected_module):
        b = lookup_bundle(name)
        names = bundle_names(b)
        assert names.dist_name == expected_dist
        assert names.module_name == expected_module


# ---------------------------------------------------------------------------
# Exhaustive validation: EVERY bundle at EVERY level
# ---------------------------------------------------------------------------


class TestAllBundleNamesValid:
    """Every generated name must pass validation."""

    @pytest.mark.parametrize(
        "bundle",
        list(all_bundles()),
        ids=lambda b: f"{b.level.value}:{b.key}",
    )
    def test_dist_name_valid(self, bundle):
        names = bundle_names(bundle)
        assert is_valid_dist_name(
            names.dist_name
        ), f"Invalid dist name {names.dist_name!r} for bundle {bundle.key}"

    @pytest.mark.parametrize(
        "bundle",
        list(all_bundles()),
        ids=lambda b: f"{b.level.value}:{b.key}",
    )
    def test_module_name_valid(self, bundle):
        names = bundle_names(bundle)
        assert is_valid_module_name(
            names.module_name
        ), f"Invalid module name {names.module_name!r} for bundle {bundle.key}"

    @pytest.mark.parametrize(
        "bundle",
        list(all_bundles()),
        ids=lambda b: f"{b.level.value}:{b.key}",
    )
    def test_module_name_is_python_identifier(self, bundle):
        names = bundle_names(bundle)
        assert (
            names.module_name.isidentifier()
        ), f"{names.module_name!r} is not a valid Python identifier"


# ---------------------------------------------------------------------------
# Device package name composition
# ---------------------------------------------------------------------------


class TestDeviceDistName:
    @pytest.mark.parametrize(
        "prefix,key,expected",
        [
            ("rocm-sdk-device", "gfx11", "rocm-sdk-device-gfx11"),
            ("rocm-sdk-device", "gfx11_5", "rocm-sdk-device-gfx11-5"),
            ("rocm-sdk-device", "gfx1151", "rocm-sdk-device-gfx1151"),
            ("rocm-sdk-device", "gfx9_4", "rocm-sdk-device-gfx9-4"),
            ("rocm-sdk-device", "gfx942", "rocm-sdk-device-gfx942"),
            ("rocm-sdk-device", "gfx90a", "rocm-sdk-device-gfx90a"),
            ("amd-torch-device", "gfx1100", "amd-torch-device-gfx1100"),
            ("amd-torch-device", "gfx12_0", "amd-torch-device-gfx12-0"),
        ],
    )
    def test_device_dist_name(self, prefix, key, expected):
        b = lookup_bundle(key)
        assert device_dist_name(prefix, b) == expected

    @pytest.mark.parametrize(
        "bundle",
        list(all_bundles()),
        ids=lambda b: f"{b.level.value}:{b.key}",
    )
    def test_all_device_dist_names_valid(self, bundle):
        name = device_dist_name("rocm-sdk-device", bundle)
        assert is_valid_dist_name(name), f"Invalid device dist name: {name!r}"


class TestDeviceModuleName:
    @pytest.mark.parametrize(
        "prefix,key,expected",
        [
            ("rocm_sdk_device", "gfx11", "rocm_sdk_device_gfx11"),
            ("rocm_sdk_device", "gfx11_5", "rocm_sdk_device_gfx11_5"),
            ("rocm_sdk_device", "gfx1151", "rocm_sdk_device_gfx1151"),
            ("rocm_sdk_device", "gfx9_4", "rocm_sdk_device_gfx9_4"),
            ("rocm_sdk_device", "gfx90a", "rocm_sdk_device_gfx90a"),
        ],
    )
    def test_device_module_name(self, prefix, key, expected):
        b = lookup_bundle(key)
        assert device_module_name(prefix, b) == expected

    @pytest.mark.parametrize(
        "bundle",
        list(all_bundles()),
        ids=lambda b: f"{b.level.value}:{b.key}",
    )
    def test_all_device_module_names_valid(self, bundle):
        name = device_module_name("rocm_sdk_device", bundle)
        assert is_valid_module_name(name), f"Invalid device module name: {name!r}"


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


class TestIsValidDistName:
    @pytest.mark.parametrize(
        "name",
        ["gfx11", "gfx9-4", "rocm-sdk-device-gfx11-5", "my.package", "a"],
    )
    def test_valid(self, name):
        assert is_valid_dist_name(name)

    @pytest.mark.parametrize(
        "name",
        ["", "-gfx11", "GFX11", "gfx 11", "gfx11-"],
    )
    def test_invalid(self, name):
        assert not is_valid_dist_name(name)


class TestIsValidModuleName:
    @pytest.mark.parametrize(
        "name",
        ["gfx11", "gfx9_4", "rocm_sdk_device_gfx11_5", "_private", "a"],
    )
    def test_valid(self, name):
        assert is_valid_module_name(name)

    @pytest.mark.parametrize(
        "name",
        ["", "gfx-11", "9gfx", "gfx 11", "gfx.11"],
    )
    def test_invalid(self, name):
        assert not is_valid_module_name(name)
