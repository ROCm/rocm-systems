"""Tests for rocm_bootstrap.detect — GPU detection via faked _platform I/O.

Every test patches _platform functions through the FakePlatform fixture,
injecting realistic sysfs content. No real filesystem access occurs.
"""

import pytest

from rocm_bootstrap.detect import (
    DetectedGpu,
    _parse_kfd_properties,
    detect_gpus,
    detect_gfx_targets,
    main,
)
from rocm_bootstrap.targets import (
    ALL_TARGETS,
    GfxTarget,
    lookup_target,
    parse_gfx_target_version,
)
from rocm_bootstrap.tests.conftest import FakePlatform


# ---------------------------------------------------------------------------
# KFD properties parser unit tests
# ---------------------------------------------------------------------------


class TestParseKfdProperties:
    def test_basic_parsing(self):
        text = "cpu_cores_count 192\nsimd_count 128\ngfx_target_version 120001\n"
        props = _parse_kfd_properties(text)
        assert props["cpu_cores_count"] == 192
        assert props["simd_count"] == 128
        assert props["gfx_target_version"] == 120001

    def test_skips_non_integer_values(self):
        text = "name foo\ncount 42\n"
        props = _parse_kfd_properties(text)
        assert "name" not in props
        assert props["count"] == 42

    def test_empty_input(self):
        assert _parse_kfd_properties("") == {}

    def test_skips_malformed_lines(self):
        text = "good 1\nbad\nalso_good 2\nthree words here\n"
        props = _parse_kfd_properties(text)
        assert props == {"good": 1, "also_good": 2}


# ---------------------------------------------------------------------------
# Per-target KFD detection (parametrized over ALL targets)
# ---------------------------------------------------------------------------


class TestKfdDetectionPerTarget:
    """Every known target is correctly detected from realistic KFD properties."""

    @pytest.mark.parametrize("target", ALL_TARGETS, ids=lambda t: t.name)
    def test_single_gpu_detected(self, target: GfxTarget, fake_platform: FakePlatform):
        """A single GPU node with this target's gfx_target_version is detected."""
        fake_platform.add_cpu_node(0)
        fake_platform.add_gpu_node(1, target, device_id=29000 + target.stepping)
        # Re-apply after adding nodes
        gpus = detect_gpus()
        assert len(gpus) == 1
        gpu = gpus[0]
        assert gpu.target is target
        assert gpu.node_id == 1
        assert gpu.pci_id is not None

    @pytest.mark.parametrize("target", ALL_TARGETS, ids=lambda t: t.name)
    def test_gfx_target_version_round_trip(
        self, target: GfxTarget, fake_platform: FakePlatform
    ):
        """The gfx_target_version in sysfs round-trips to the correct target."""
        gtv = target.gfx_target_version
        fake_platform.add_gpu_node(0, target)
        gpus = detect_gpus()
        assert len(gpus) == 1
        assert gpus[0].target.gfx_target_version == gtv
        assert gpus[0].target.name == target.name


# ---------------------------------------------------------------------------
# Multi-GPU scenarios
# ---------------------------------------------------------------------------


class TestMultiGpu:
    def test_two_different_gpus(self, fake_platform: FakePlatform):
        gfx1201 = lookup_target("gfx1201")
        gfx1100 = lookup_target("gfx1100")
        fake_platform.add_cpu_node(0)
        fake_platform.add_gpu_node(1, gfx1201, device_id=30032)
        fake_platform.add_gpu_node(2, gfx1100, device_id=29768)

        gpus = detect_gpus()
        assert len(gpus) == 2
        assert gpus[0].target is gfx1201
        assert gpus[0].node_id == 1
        assert gpus[1].target is gfx1100
        assert gpus[1].node_id == 2

    def test_two_identical_gpus(self, fake_platform: FakePlatform):
        gfx942 = lookup_target("gfx942")
        fake_platform.add_cpu_node(0)
        fake_platform.add_gpu_node(1, gfx942)
        fake_platform.add_gpu_node(2, gfx942)

        gpus = detect_gpus()
        assert len(gpus) == 2
        assert all(g.target is gfx942 for g in gpus)

    def test_detect_gfx_targets_deduplicates(self, fake_platform: FakePlatform):
        gfx942 = lookup_target("gfx942")
        fake_platform.add_cpu_node(0)
        fake_platform.add_gpu_node(1, gfx942)
        fake_platform.add_gpu_node(2, gfx942)

        targets = detect_gfx_targets()
        assert len(targets) == 1
        assert targets[0] is gfx942

    def test_mixed_gpus_dedup_preserves_order(self, fake_platform: FakePlatform):
        gfx1100 = lookup_target("gfx1100")
        gfx942 = lookup_target("gfx942")
        fake_platform.add_cpu_node(0)
        fake_platform.add_gpu_node(1, gfx1100)
        fake_platform.add_gpu_node(2, gfx942)
        fake_platform.add_gpu_node(3, gfx1100)

        targets = detect_gfx_targets()
        assert len(targets) == 2
        assert targets[0] is gfx1100
        assert targets[1] is gfx942


# ---------------------------------------------------------------------------
# CPU-only and empty scenarios
# ---------------------------------------------------------------------------


class TestNoGpu:
    def test_cpu_only_node_skipped(self, fake_platform: FakePlatform):
        """CPU nodes (simd_count=0) produce no results."""
        fake_platform.add_cpu_node(0)
        assert detect_gpus() == []

    def test_no_kfd(self, fake_platform: FakePlatform):
        """Empty platform → empty list."""
        assert detect_gpus() == []


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvOverrides:
    def test_disable_detection(self, fake_platform: FakePlatform):
        gfx1100 = lookup_target("gfx1100")
        fake_platform.add_gpu_node(1, gfx1100)
        fake_platform.set_env("ROCM_BOOTSTRAP_DISABLE_DETECTION", "1")
        assert detect_gpus() == []

    def test_disable_not_set(self, fake_platform: FakePlatform):
        gfx1100 = lookup_target("gfx1100")
        fake_platform.add_cpu_node(0)
        fake_platform.add_gpu_node(1, gfx1100)
        # ROCM_BOOTSTRAP_DISABLE_DETECTION not set → detection works
        gpus = detect_gpus()
        assert len(gpus) == 1

    def test_force_single_arch(self, fake_platform: FakePlatform):
        fake_platform.set_env("ROCM_BOOTSTRAP_FORCE_GFX_ARCH", "gfx942")
        gpus = detect_gpus()
        assert len(gpus) == 1
        assert gpus[0].target.name == "gfx942"
        assert gpus[0].node_id == 0

    def test_force_multiple_archs(self, fake_platform: FakePlatform):
        fake_platform.set_env("ROCM_BOOTSTRAP_FORCE_GFX_ARCH", "gfx942,gfx942,gfx1100")
        gpus = detect_gpus()
        assert len(gpus) == 3
        assert gpus[0].target.name == "gfx942"
        assert gpus[1].target.name == "gfx942"
        assert gpus[2].target.name == "gfx1100"

    def test_force_overrides_real_detection(self, fake_platform: FakePlatform):
        """Forced arch takes precedence over actual GPU nodes."""
        gfx1201 = lookup_target("gfx1201")
        fake_platform.add_gpu_node(1, gfx1201)
        fake_platform.set_env("ROCM_BOOTSTRAP_FORCE_GFX_ARCH", "gfx950")
        gpus = detect_gpus()
        assert len(gpus) == 1
        assert gpus[0].target.name == "gfx950"

    def test_force_unknown_target_raises(self, fake_platform: FakePlatform):
        fake_platform.set_env("ROCM_BOOTSTRAP_FORCE_GFX_ARCH", "gfx9999")
        with pytest.raises(ValueError, match="Unknown GFX target"):
            detect_gpus()

    def test_disable_takes_precedence_over_force(self, fake_platform: FakePlatform):
        fake_platform.set_env("ROCM_BOOTSTRAP_DISABLE_DETECTION", "1")
        fake_platform.set_env("ROCM_BOOTSTRAP_FORCE_GFX_ARCH", "gfx942")
        assert detect_gpus() == []


# ---------------------------------------------------------------------------
# DetectedGpu dataclass
# ---------------------------------------------------------------------------


class TestDetectedGpu:
    def test_frozen(self):
        gpu = DetectedGpu(target=lookup_target("gfx1100"), node_id=0)
        with pytest.raises(AttributeError):
            gpu.node_id = 1  # type: ignore[misc]

    def test_defaults(self):
        gpu = DetectedGpu(target=lookup_target("gfx1100"), node_id=0)
        assert gpu.gpu_id == 0
        assert gpu.pci_id is None

    def test_pci_id_from_kfd(self, fake_platform: FakePlatform):
        """device_id from KFD properties is converted to hex pci_id."""
        gfx1201 = lookup_target("gfx1201")
        fake_platform.add_cpu_node(0)
        fake_platform.add_gpu_node(1, gfx1201, device_id=0x7550)
        gpus = detect_gpus()
        assert gpus[0].pci_id == "0x7550"


# ---------------------------------------------------------------------------
# CLI output
# ---------------------------------------------------------------------------


class TestCliUnique:
    def test_single_gpu(self, fake_platform: FakePlatform, capsys):
        gfx942 = lookup_target("gfx942")
        fake_platform.add_cpu_node(0)
        fake_platform.add_gpu_node(1, gfx942)
        main(["--unique"])
        assert capsys.readouterr().out == "gfx942\n"

    def test_short_flag(self, fake_platform: FakePlatform, capsys):
        fake_platform.add_gpu_node(1, lookup_target("gfx942"))
        main(["-u"])
        assert capsys.readouterr().out == "gfx942\n"

    def test_no_gpus(self, fake_platform: FakePlatform, capsys):
        main(["--unique"])
        assert capsys.readouterr().out == ""

    def test_deduplicates(self, fake_platform: FakePlatform, capsys):
        gfx942 = lookup_target("gfx942")
        fake_platform.add_gpu_node(1, gfx942)
        fake_platform.add_gpu_node(2, gfx942)
        main(["--unique"])
        assert capsys.readouterr().out == "gfx942\n"

    def test_multiple_targets(self, fake_platform: FakePlatform, capsys):
        fake_platform.add_cpu_node(0)
        fake_platform.add_gpu_node(1, lookup_target("gfx1201"))
        fake_platform.add_gpu_node(2, lookup_target("gfx1100"))
        main(["--unique"])
        out = capsys.readouterr().out
        assert out == "gfx1201\ngfx1100\n"


class TestCliHierarchy:
    def test_single_gpu(self, fake_platform: FakePlatform, capsys):
        fake_platform.add_gpu_node(1, lookup_target("gfx1151"))
        main(["--hierarchy"])
        assert capsys.readouterr().out == "gfx1151 gfx11_5 gfx11\n"

    def test_no_gpus(self, fake_platform: FakePlatform, capsys):
        main(["--hierarchy"])
        assert capsys.readouterr().out == ""

    def test_deduplicates(self, fake_platform: FakePlatform, capsys):
        gfx942 = lookup_target("gfx942")
        fake_platform.add_gpu_node(1, gfx942)
        fake_platform.add_gpu_node(2, gfx942)
        main(["--hierarchy"])
        assert capsys.readouterr().out == "gfx942 gfx9_4 gfx9\n"

    def test_multiple_targets(self, fake_platform: FakePlatform, capsys):
        fake_platform.add_gpu_node(1, lookup_target("gfx1201"))
        fake_platform.add_gpu_node(2, lookup_target("gfx942"))
        main(["--hierarchy"])
        out = capsys.readouterr().out
        assert out == "gfx1201 gfx12_0 gfx12\ngfx942 gfx9_4 gfx9\n"


class TestCliVerbose:
    def test_no_gpus(self, fake_platform: FakePlatform, capsys):
        main(["--verbose"])
        assert "No AMD GPUs detected" in capsys.readouterr().out

    def test_shows_hierarchy(self, fake_platform: FakePlatform, capsys):
        fake_platform.add_cpu_node(0)
        fake_platform.add_gpu_node(1, lookup_target("gfx1151"), device_id=0xABCD)
        main(["-v"])
        out = capsys.readouterr().out
        assert "gfx1151" in out
        assert "gfx11_5" in out
        assert "gfx11" in out
        assert "sub-family" in out
        assert "family" in out

    def test_shows_generic_isa(self, fake_platform: FakePlatform, capsys):
        fake_platform.add_gpu_node(1, lookup_target("gfx1201"))
        main(["--verbose"])
        out = capsys.readouterr().out
        assert "gfx12-generic" in out

    def test_shows_pci_id(self, fake_platform: FakePlatform, capsys):
        fake_platform.add_gpu_node(1, lookup_target("gfx942"), device_id=0x74A0)
        main(["--verbose"])
        out = capsys.readouterr().out
        assert "0x74a0" in out


class TestCliMutualExclusion:
    def test_no_args_defaults_to_unique(self, fake_platform: FakePlatform, capsys):
        fake_platform.add_gpu_node(1, lookup_target("gfx942"))
        main([])
        assert capsys.readouterr().out == "gfx942\n"

    def test_verbose_and_hierarchy_conflict(self, fake_platform: FakePlatform):
        with pytest.raises(SystemExit):
            main(["--verbose", "--hierarchy"])

    def test_unique_and_verbose_conflict(self, fake_platform: FakePlatform):
        with pytest.raises(SystemExit):
            main(["--unique", "--verbose"])
