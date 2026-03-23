"""Tests for rocm_bootstrap.variant_provider — WheelNext plugin protocol."""

import pytest

from rocm_bootstrap.targets import ALL_TARGETS, lookup_target
from rocm_bootstrap.tests.conftest import FakePlatform
from rocm_bootstrap.variant_provider import AMDVariantPlugin, VariantFeatureConfig


class TestPluginMetadata:
    def test_namespace(self):
        assert AMDVariantPlugin.namespace == "amd"

    def test_not_aot(self):
        assert AMDVariantPlugin.is_aot_plugin is False


class TestGetAllConfigs:
    def test_returns_gfx_arch_feature(self):
        configs = AMDVariantPlugin.get_all_configs()
        assert len(configs) == 1
        assert configs[0].name == "gfx_arch"

    def test_multi_value(self):
        configs = AMDVariantPlugin.get_all_configs()
        assert configs[0].multi_value is True

    def test_includes_all_targets(self):
        configs = AMDVariantPlugin.get_all_configs()
        values = set(configs[0].values)
        for target in ALL_TARGETS:
            assert (
                target.name in values
            ), f"Target {target.name} missing from get_all_configs()"

    def test_value_count_matches_target_count(self):
        configs = AMDVariantPlugin.get_all_configs()
        assert len(configs[0].values) == len(ALL_TARGETS)

    def test_constant_regardless_of_platform(self, fake_platform: FakePlatform):
        """get_all_configs is static — same result with or without GPUs."""
        gfx1100 = lookup_target("gfx1100")
        fake_platform.add_gpu_node(1, gfx1100)
        configs_with_gpu = AMDVariantPlugin.get_all_configs()

        # get_all_configs doesn't use detection, so clearing doesn't matter
        assert len(configs_with_gpu) == 1
        assert len(configs_with_gpu[0].values) == len(ALL_TARGETS)


class TestGetSupportedConfigs:
    def test_empty_when_no_gpus(self, fake_platform: FakePlatform):
        configs = AMDVariantPlugin.get_supported_configs()
        assert configs == []

    def test_empty_when_detection_disabled(self, fake_platform: FakePlatform):
        gfx1100 = lookup_target("gfx1100")
        fake_platform.add_gpu_node(1, gfx1100)
        fake_platform.set_env("ROCM_BOOTSTRAP_DISABLE_DETECTION", "1")
        configs = AMDVariantPlugin.get_supported_configs()
        assert configs == []

    def test_single_gpu(self, fake_platform: FakePlatform):
        gfx942 = lookup_target("gfx942")
        fake_platform.add_cpu_node(0)
        fake_platform.add_gpu_node(1, gfx942)
        configs = AMDVariantPlugin.get_supported_configs()
        assert len(configs) == 1
        assert configs[0].name == "gfx_arch"
        assert configs[0].values == ["gfx942"]
        assert configs[0].multi_value is True

    def test_multiple_gpus(self, fake_platform: FakePlatform):
        gfx1100 = lookup_target("gfx1100")
        gfx1201 = lookup_target("gfx1201")
        fake_platform.add_cpu_node(0)
        fake_platform.add_gpu_node(1, gfx1100)
        fake_platform.add_gpu_node(2, gfx1201)
        configs = AMDVariantPlugin.get_supported_configs()
        assert len(configs) == 1
        assert configs[0].values == ["gfx1100", "gfx1201"]

    def test_duplicate_gpus_deduplicated(self, fake_platform: FakePlatform):
        gfx942 = lookup_target("gfx942")
        fake_platform.add_cpu_node(0)
        fake_platform.add_gpu_node(1, gfx942)
        fake_platform.add_gpu_node(2, gfx942)
        configs = AMDVariantPlugin.get_supported_configs()
        assert configs[0].values == ["gfx942"]

    def test_forced_arch(self, fake_platform: FakePlatform):
        fake_platform.set_env("ROCM_BOOTSTRAP_FORCE_GFX_ARCH", "gfx950")
        configs = AMDVariantPlugin.get_supported_configs()
        assert len(configs) == 1
        assert configs[0].values == ["gfx950"]

    def test_supported_values_are_subset_of_all(self, fake_platform: FakePlatform):
        """Protocol invariant: supported values must be a subset of all values."""
        gfx1100 = lookup_target("gfx1100")
        fake_platform.add_gpu_node(1, gfx1100)
        supported = AMDVariantPlugin.get_supported_configs()
        all_configs = AMDVariantPlugin.get_all_configs()

        all_values = set(all_configs[0].values)
        for val in supported[0].values:
            assert val in all_values


class TestVariantFeatureConfig:
    def test_frozen(self):
        cfg = VariantFeatureConfig(name="test", values=["a"], multi_value=False)
        with pytest.raises(AttributeError):
            cfg.name = "other"  # type: ignore[misc]

    def test_fields(self):
        cfg = VariantFeatureConfig(
            name="gfx_arch", values=["gfx942", "gfx1100"], multi_value=True
        )
        assert cfg.name == "gfx_arch"
        assert cfg.values == ["gfx942", "gfx1100"]
        assert cfg.multi_value is True

    def test_multi_value_default(self):
        cfg = VariantFeatureConfig(name="test", values=["a"])
        assert cfg.multi_value is False
