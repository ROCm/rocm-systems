"""Tests for rocm_bootstrap.targets — hierarchy, lookups, and data integrity."""

import pytest

from rocm_bootstrap.targets import (
    ALL_FAMILIES,
    ALL_SUB_FAMILIES,
    ALL_TARGETS,
    GfxTarget,
    PackagingLevel,
    TargetBundle,
    XnackMode,
    _FAMILY_TO_SUB_FAMILIES,
    all_bundles,
    bundle_for_target,
    lookup_bundle,
    lookup_target,
    packaging_chain,
    parse_gfx_target_version,
)


# ---------------------------------------------------------------------------
# Hierarchy structure invariants
# ---------------------------------------------------------------------------


class TestHierarchyStructure:
    """Verify the 3-level hierarchy is consistent."""

    def test_every_target_in_exactly_one_sub_family(self):
        """Each target appears in exactly one sub-family."""
        seen: dict[str, str] = {}
        for sf in ALL_SUB_FAMILIES:
            for t in sf.members:
                assert (
                    t.name not in seen
                ), f"Target {t.name} in both {seen[t.name]} and {sf.key}"
                seen[t.name] = sf.key
        # Every target in ALL_TARGETS is accounted for
        for t in ALL_TARGETS:
            assert t.name in seen, f"Target {t.name} not in any sub-family"

    def test_every_sub_family_in_exactly_one_family(self):
        """Each sub-family belongs to exactly one family."""
        seen: dict[str, str] = {}
        for fam_key, sfs in _FAMILY_TO_SUB_FAMILIES.items():
            for sf in sfs:
                assert (
                    sf.key not in seen
                ), f"Sub-family {sf.key} in both {seen[sf.key]} and {fam_key}"
                seen[sf.key] = fam_key
        for sf in ALL_SUB_FAMILIES:
            assert sf.key in seen, f"Sub-family {sf.key} not in any family"

    def test_family_members_equal_union_of_sub_family_members(self):
        """A family's members == union of its sub-families' members."""
        for fam_key, sfs in _FAMILY_TO_SUB_FAMILIES.items():
            fam = lookup_bundle(fam_key)
            sf_members = set()
            for sf in sfs:
                sf_members.update(sf.members)
            assert (
                set(fam.members) == sf_members
            ), f"Family {fam_key} members don't match sub-family union"

    def test_bundle_levels_correct(self):
        """Each bundle has the correct PackagingLevel."""
        for fam in ALL_FAMILIES:
            assert fam.level == PackagingLevel.FAMILY
        for sf in ALL_SUB_FAMILIES:
            assert sf.level == PackagingLevel.SUB_FAMILY
        for t in ALL_TARGETS:
            b = lookup_bundle(t.name)
            assert b.level == PackagingLevel.TARGET

    def test_target_bundles_are_singletons(self):
        """Each target-level bundle has exactly one member."""
        for t in ALL_TARGETS:
            b = lookup_bundle(t.name)
            assert len(b.members) == 1
            assert b.members[0] is t

    def test_all_targets_count(self):
        """Sanity check: we have a reasonable number of targets."""
        # 8 (gfx9.0) + 1 (gfx9.4) + 1 (gfx9.5)
        # + 4 (gfx10.1) + 7 (gfx10.3)
        # + 4 (gfx11.0) + 4 (gfx11.5)
        # + 2 (gfx12.0) + 2 (gfx12.5)
        # = 33
        assert len(ALL_TARGETS) == 33


# ---------------------------------------------------------------------------
# GfxTarget dataclass
# ---------------------------------------------------------------------------


class TestGfxTarget:
    def test_frozen(self):
        t = lookup_target("gfx1100")
        with pytest.raises(AttributeError):
            t.name = "gfx9999"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "name,expected_major,expected_minor,expected_stepping",
        [
            ("gfx900", 9, 0, 0),
            ("gfx90a", 9, 0, 10),
            ("gfx90c", 9, 0, 12),
            ("gfx942", 9, 4, 2),
            ("gfx950", 9, 5, 0),
            ("gfx1010", 10, 1, 0),
            ("gfx1100", 11, 0, 0),
            ("gfx1151", 11, 5, 1),
            ("gfx1201", 12, 0, 1),
            ("gfx1250", 12, 5, 0),
        ],
    )
    def test_version_components(
        self, name, expected_major, expected_minor, expected_stepping
    ):
        t = lookup_target(name)
        assert t.major == expected_major
        assert t.minor == expected_minor
        assert t.stepping == expected_stepping


# ---------------------------------------------------------------------------
# gfx_target_version encoding round-trip
# ---------------------------------------------------------------------------


class TestGfxTargetVersion:
    @pytest.mark.parametrize("target", ALL_TARGETS, ids=lambda t: t.name)
    def test_round_trip(self, target: GfxTarget):
        """parse_gfx_target_version(target.gfx_target_version) == target."""
        gtv = target.gfx_target_version
        assert parse_gfx_target_version(gtv) is target

    @pytest.mark.parametrize(
        "gtv,expected_name",
        [
            (120001, "gfx1201"),
            (110000, "gfx1100"),
            (90402, "gfx942"),
            (110001, "gfx1101"),  # major=11, minor=0, stepping=1
            (110501, "gfx1151"),  # major=11, minor=5, stepping=1
            (90000, "gfx900"),
            (90010, "gfx90a"),  # stepping 10 = 0xa
            (90012, "gfx90c"),  # stepping 12 = 0xc
        ],
    )
    def test_documented_examples(self, gtv, expected_name):
        """Test the specific examples from the kpack-wheelnext doc."""
        t = parse_gfx_target_version(gtv)
        assert t.name == expected_name

    def test_unknown_gtv_raises(self):
        with pytest.raises(ValueError, match="Unknown gfx_target_version"):
            parse_gfx_target_version(999999)

    @pytest.mark.parametrize("target", ALL_TARGETS, ids=lambda t: t.name)
    def test_no_gtv_collisions(self, target: GfxTarget):
        """Every target has a unique gfx_target_version."""
        gtvs = [t.gfx_target_version for t in ALL_TARGETS]
        assert gtvs.count(target.gfx_target_version) == 1


# ---------------------------------------------------------------------------
# xnack classifications
# ---------------------------------------------------------------------------


class TestXnackClassification:
    @pytest.mark.parametrize(
        "name",
        [
            "gfx900",
            "gfx902",
            "gfx904",
            "gfx906",
            "gfx908",
            "gfx909",
            "gfx90a",
            "gfx90c",
            "gfx942",
            "gfx950",
        ],
    )
    def test_gfx9_targets_support_xnack(self, name):
        assert lookup_target(name).xnack == XnackMode.SUPPORTED

    @pytest.mark.parametrize("name", ["gfx1010", "gfx1011", "gfx1012", "gfx1013"])
    def test_gfx10_1_targets_support_xnack(self, name):
        assert lookup_target(name).xnack == XnackMode.SUPPORTED

    @pytest.mark.parametrize(
        "name",
        ["gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036"],
    )
    def test_gfx10_3_targets_no_xnack(self, name):
        assert lookup_target(name).xnack == XnackMode.UNSUPPORTED

    @pytest.mark.parametrize(
        "name",
        [
            "gfx1100",
            "gfx1101",
            "gfx1102",
            "gfx1103",
            "gfx1150",
            "gfx1151",
            "gfx1152",
            "gfx1153",
        ],
    )
    def test_gfx11_targets_no_xnack(self, name):
        assert lookup_target(name).xnack == XnackMode.UNSUPPORTED

    @pytest.mark.parametrize("name", ["gfx1200", "gfx1201"])
    def test_gfx12_0_targets_no_xnack(self, name):
        assert lookup_target(name).xnack == XnackMode.UNSUPPORTED

    @pytest.mark.parametrize("name", ["gfx1250", "gfx1251"])
    def test_gfx12_5_targets_xnack_default_on(self, name):
        assert lookup_target(name).xnack == XnackMode.DEFAULT_ON


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


class TestLookupTarget:
    def test_known_target(self):
        t = lookup_target("gfx1100")
        assert t.name == "gfx1100"
        assert t.major == 11

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Unknown GFX target"):
            lookup_target("gfx9999")

    @pytest.mark.parametrize("target", ALL_TARGETS, ids=lambda t: t.name)
    def test_all_targets_lookup_roundtrip(self, target: GfxTarget):
        assert lookup_target(target.name) is target


class TestLookupBundle:
    def test_family(self):
        b = lookup_bundle("gfx11")
        assert b.level == PackagingLevel.FAMILY

    def test_sub_family(self):
        b = lookup_bundle("gfx11_5")
        assert b.level == PackagingLevel.SUB_FAMILY

    def test_target(self):
        b = lookup_bundle("gfx1151")
        assert b.level == PackagingLevel.TARGET

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown bundle key"):
            lookup_bundle("gfx999")


# ---------------------------------------------------------------------------
# Packaging chain
# ---------------------------------------------------------------------------


class TestPackagingChain:
    @pytest.mark.parametrize(
        "target_name,expected_sf_key,expected_fam_key",
        [
            ("gfx1151", "gfx11_5", "gfx11"),
            ("gfx1100", "gfx11_0", "gfx11"),
            ("gfx942", "gfx9_4", "gfx9"),
            ("gfx950", "gfx9_5", "gfx9"),
            ("gfx90a", "gfx9_0", "gfx9"),
            ("gfx908", "gfx9_0", "gfx9"),
            ("gfx1201", "gfx12_0", "gfx12"),
            ("gfx1250", "gfx12_5", "gfx12"),
            ("gfx1030", "gfx10_3", "gfx10"),
            ("gfx1010", "gfx10_1", "gfx10"),
        ],
    )
    def test_chain_structure(self, target_name, expected_sf_key, expected_fam_key):
        target_b, sf_b, fam_b = packaging_chain(target_name)
        assert target_b.key == target_name
        assert target_b.level == PackagingLevel.TARGET
        assert sf_b.key == expected_sf_key
        assert sf_b.level == PackagingLevel.SUB_FAMILY
        assert fam_b.key == expected_fam_key
        assert fam_b.level == PackagingLevel.FAMILY

    def test_accepts_gfx_target_instance(self):
        from rocm_bootstrap.targets import GFX1151

        target_b, sf_b, fam_b = packaging_chain(GFX1151)
        assert target_b.key == "gfx1151"

    @pytest.mark.parametrize("target", ALL_TARGETS, ids=lambda t: t.name)
    def test_every_target_has_chain(self, target: GfxTarget):
        """Every target produces a valid 3-element chain."""
        chain = packaging_chain(target)
        assert len(chain) == 3
        assert chain[0].level == PackagingLevel.TARGET
        assert chain[1].level == PackagingLevel.SUB_FAMILY
        assert chain[2].level == PackagingLevel.FAMILY


class TestBundleForTarget:
    def test_target_level(self):
        b = bundle_for_target("gfx1151", PackagingLevel.TARGET)
        assert b.key == "gfx1151"

    def test_sub_family_level(self):
        b = bundle_for_target("gfx1151", PackagingLevel.SUB_FAMILY)
        assert b.key == "gfx11_5"

    def test_family_level(self):
        b = bundle_for_target("gfx1151", PackagingLevel.FAMILY)
        assert b.key == "gfx11"


# ---------------------------------------------------------------------------
# all_bundles
# ---------------------------------------------------------------------------


class TestAllBundles:
    def test_unfiltered_has_all_levels(self):
        bundles = all_bundles()
        levels = {b.level for b in bundles}
        assert PackagingLevel.FAMILY in levels
        assert PackagingLevel.SUB_FAMILY in levels
        assert PackagingLevel.TARGET in levels

    def test_filter_family(self):
        bundles = all_bundles(PackagingLevel.FAMILY)
        assert all(b.level == PackagingLevel.FAMILY for b in bundles)
        assert len(bundles) == 4  # gfx9, gfx10, gfx11, gfx12

    def test_filter_sub_family(self):
        bundles = all_bundles(PackagingLevel.SUB_FAMILY)
        assert all(b.level == PackagingLevel.SUB_FAMILY for b in bundles)
        assert len(bundles) == 9

    def test_filter_target(self):
        bundles = all_bundles(PackagingLevel.TARGET)
        assert all(b.level == PackagingLevel.TARGET for b in bundles)
        assert len(bundles) == len(ALL_TARGETS)


# ---------------------------------------------------------------------------
# LLVM generic assignments
# ---------------------------------------------------------------------------


class TestLLVMGenericAssignments:
    """Verify llvm_generic metadata matches GCNProcessors.td comments."""

    def test_gfx9_0_has_gfx9_generic(self):
        assert lookup_bundle("gfx9_0").llvm_generic == "gfx9-generic"

    def test_gfx9_4_has_gfx9_4_generic(self):
        assert lookup_bundle("gfx9_4").llvm_generic == "gfx9-4-generic"

    def test_gfx9_5_no_generic(self):
        assert lookup_bundle("gfx9_5").llvm_generic is None

    def test_gfx10_1_has_gfx10_1_generic(self):
        assert lookup_bundle("gfx10_1").llvm_generic == "gfx10-1-generic"

    def test_gfx10_3_has_gfx10_3_generic(self):
        assert lookup_bundle("gfx10_3").llvm_generic == "gfx10-3-generic"

    def test_gfx11_family_has_gfx11_generic(self):
        assert lookup_bundle("gfx11").llvm_generic == "gfx11-generic"

    def test_gfx11_0_no_generic(self):
        """gfx11-generic is at family level, not sub-family."""
        assert lookup_bundle("gfx11_0").llvm_generic is None

    def test_gfx11_5_no_generic(self):
        assert lookup_bundle("gfx11_5").llvm_generic is None

    def test_gfx12_0_has_gfx12_generic(self):
        assert lookup_bundle("gfx12_0").llvm_generic == "gfx12-generic"

    def test_gfx12_5_no_generic(self):
        assert lookup_bundle("gfx12_5").llvm_generic is None

    def test_gfx9_family_no_generic(self):
        """No gfx9-generic at the family level (it's at sub-family gfx9_0)."""
        assert lookup_bundle("gfx9").llvm_generic is None

    def test_gfx12_family_no_generic(self):
        """No gfx12-generic at family (it's at sub-family gfx12_0)."""
        assert lookup_bundle("gfx12").llvm_generic is None
