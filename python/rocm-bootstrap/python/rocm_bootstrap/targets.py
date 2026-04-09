"""GFX target definitions, hierarchy, and lookups.

This module is the source of truth for all AMD GFX target knowledge in
the ROCm Python packaging ecosystem. Target data is encoded as Python
literals derived from LLVM's canonical definitions:

  Source files (in amd-llvm):
    llvm/lib/Target/AMDGPU/GCNProcessors.td  -- processor definitions + generic groupings
    llvm/lib/Target/AMDGPU/AMDGPU.td         -- feature sets + xnack support

Every GFX target maps to a 3-level packaging hierarchy:
  target     -> sub-family -> family
  gfx1151    -> gfx11_5    -> gfx11

Packages can exist at any level. A complete installation uses packages
from all applicable levels (some may be empty/omitted).
"""

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class XnackMode(Enum):
    """XNACK hardware support level.

    Used for build configuration metadata. Does NOT affect package naming —
    xnack+/- kernels go inside the arch package. ASAN is a parallel universe
    at the index/version level.
    """

    UNSUPPORTED = "unsupported"
    SUPPORTED = "supported"  # Hardware supports, off by default
    DEFAULT_ON = "default_on"  # Hardware has xnack on by default (gfx1250+)


@dataclass(frozen=True)
class GfxTarget:
    """A specific GFX target ISA (e.g., gfx1151).

    Instances are module-level constants. Use :func:`lookup_target` to get
    the canonical instance by name.

    Attributes:
        name: Canonical LLVM target name (e.g., ``"gfx1151"``).
        major: ISA major version (e.g., 11).
        minor: ISA minor version (e.g., 5).
        stepping: ISA stepping (e.g., 1).
        xnack: XNACK hardware support level.
    """

    name: str
    major: int
    minor: int
    stepping: int
    xnack: XnackMode

    @property
    def gfx_target_version(self) -> int:
        """Sysfs encoding: ``major * 10000 + minor * 100 + stepping``.

        This matches the ``gfx_target_version`` field in
        ``/sys/class/kfd/kfd/topology/nodes/*/properties``.
        """
        return self.major * 10000 + self.minor * 100 + self.stepping


class PackagingLevel(Enum):
    """Level in the 3-tier packaging hierarchy."""

    FAMILY = "family"  # gfx11   (major only)
    SUB_FAMILY = "sub_family"  # gfx11_5 (major + minor)
    TARGET = "target"  # gfx1151 (specific)


@dataclass(frozen=True)
class TargetBundle:
    """A named group of targets at one level of the packaging hierarchy.

    Bundles are structural — derived from the GFX version number encoding:

    * **Family**: all targets sharing a major version.
    * **Sub-family**: all targets sharing major + minor.
    * **Target**: one specific target.

    Attributes:
        key: Python module-safe identifier (e.g., ``"gfx11_5"``).
            Uses underscores. The corresponding dist-safe name replaces
            ``_`` with ``-`` (e.g., ``"gfx11-5"``).
        level: Which tier of the hierarchy this bundle represents.
        display_name: Human-readable label (e.g., ``"GFX11.5 (RDNA 3.5)"``).
        llvm_generic: LLVM generic target name if one covers this level
            (e.g., ``"gfx11-generic"``), or ``None``.
        members: Specific :class:`GfxTarget` instances at this level,
            in the order they appear in ``GCNProcessors.td``.
    """

    key: str
    level: PackagingLevel
    display_name: str
    llvm_generic: str | None
    members: tuple[GfxTarget, ...]


# ---------------------------------------------------------------------------
# Target constants — from GCNProcessors.td
#
# xnack derived from AMDGPU.td feature sets:
#   FeatureGFX9 (line 1595): includes FeatureSupportsXNACK → all GFX9 SUPPORTED
#   FeatureISAVersion10_1_Common (line 1944): FeatureSupportsXNACK → GFX10.1 SUPPORTED
#   FeatureISAVersion10_3_0 (line 1988): absent → GFX10.3 UNSUPPORTED
#   FeatureISAVersion11 feature sets: absent → GFX11 UNSUPPORTED
#   FeatureISAVersion12 (line 2091): absent → GFX12.0 UNSUPPORTED
#   FeatureISAVersion12_50_Common (lines 2211-2212): both → GFX12.5 DEFAULT_ON
# ---------------------------------------------------------------------------

_S = XnackMode.SUPPORTED
_U = XnackMode.UNSUPPORTED
_D = XnackMode.DEFAULT_ON

# -- GFX9.0 (Vega / consumer + MI100/MI200) --
GFX900 = GfxTarget(name="gfx900", major=9, minor=0, stepping=0, xnack=_S)
GFX902 = GfxTarget(name="gfx902", major=9, minor=0, stepping=2, xnack=_S)
GFX904 = GfxTarget(name="gfx904", major=9, minor=0, stepping=4, xnack=_S)
GFX906 = GfxTarget(name="gfx906", major=9, minor=0, stepping=6, xnack=_S)
GFX908 = GfxTarget(name="gfx908", major=9, minor=0, stepping=8, xnack=_S)
GFX909 = GfxTarget(name="gfx909", major=9, minor=0, stepping=9, xnack=_S)
GFX90A = GfxTarget(name="gfx90a", major=9, minor=0, stepping=10, xnack=_S)
GFX90C = GfxTarget(name="gfx90c", major=9, minor=0, stepping=12, xnack=_S)

# -- GFX9.4 (Instinct MI300) --
GFX942 = GfxTarget(name="gfx942", major=9, minor=4, stepping=2, xnack=_S)

# -- GFX9.5 (Instinct MI350) --
GFX950 = GfxTarget(name="gfx950", major=9, minor=5, stepping=0, xnack=_S)

# -- GFX10.1 (RDNA 1) --
GFX1010 = GfxTarget(name="gfx1010", major=10, minor=1, stepping=0, xnack=_S)
GFX1011 = GfxTarget(name="gfx1011", major=10, minor=1, stepping=1, xnack=_S)
GFX1012 = GfxTarget(name="gfx1012", major=10, minor=1, stepping=2, xnack=_S)
GFX1013 = GfxTarget(name="gfx1013", major=10, minor=1, stepping=3, xnack=_S)

# -- GFX10.3 (RDNA 2) --
GFX1030 = GfxTarget(name="gfx1030", major=10, minor=3, stepping=0, xnack=_U)
GFX1031 = GfxTarget(name="gfx1031", major=10, minor=3, stepping=1, xnack=_U)
GFX1032 = GfxTarget(name="gfx1032", major=10, minor=3, stepping=2, xnack=_U)
GFX1033 = GfxTarget(name="gfx1033", major=10, minor=3, stepping=3, xnack=_U)
GFX1034 = GfxTarget(name="gfx1034", major=10, minor=3, stepping=4, xnack=_U)
GFX1035 = GfxTarget(name="gfx1035", major=10, minor=3, stepping=5, xnack=_U)
GFX1036 = GfxTarget(name="gfx1036", major=10, minor=3, stepping=6, xnack=_U)

# -- GFX11.0 (RDNA 3 desktop) --
GFX1100 = GfxTarget(name="gfx1100", major=11, minor=0, stepping=0, xnack=_U)
GFX1101 = GfxTarget(name="gfx1101", major=11, minor=0, stepping=1, xnack=_U)
GFX1102 = GfxTarget(name="gfx1102", major=11, minor=0, stepping=2, xnack=_U)
GFX1103 = GfxTarget(name="gfx1103", major=11, minor=0, stepping=3, xnack=_U)

# -- GFX11.5 (RDNA 3.5 APU — Strix/Phoenix) --
GFX1150 = GfxTarget(name="gfx1150", major=11, minor=5, stepping=0, xnack=_U)
GFX1151 = GfxTarget(name="gfx1151", major=11, minor=5, stepping=1, xnack=_U)
GFX1152 = GfxTarget(name="gfx1152", major=11, minor=5, stepping=2, xnack=_U)
GFX1153 = GfxTarget(name="gfx1153", major=11, minor=5, stepping=3, xnack=_U)

# -- GFX12.0 (RDNA 4) --
GFX1200 = GfxTarget(name="gfx1200", major=12, minor=0, stepping=0, xnack=_U)
GFX1201 = GfxTarget(name="gfx1201", major=12, minor=0, stepping=1, xnack=_U)

# -- GFX12.5 --
GFX1250 = GfxTarget(name="gfx1250", major=12, minor=5, stepping=0, xnack=_D)
GFX1251 = GfxTarget(name="gfx1251", major=12, minor=5, stepping=1, xnack=_D)


# ---------------------------------------------------------------------------
# Bundles — target level (one per specific target)
# ---------------------------------------------------------------------------

_TARGET = PackagingLevel.TARGET

BUNDLE_GFX900 = TargetBundle(
    key="gfx900",
    level=_TARGET,
    display_name="gfx900",
    llvm_generic=None,
    members=(GFX900,),
)
BUNDLE_GFX902 = TargetBundle(
    key="gfx902",
    level=_TARGET,
    display_name="gfx902",
    llvm_generic=None,
    members=(GFX902,),
)
BUNDLE_GFX904 = TargetBundle(
    key="gfx904",
    level=_TARGET,
    display_name="gfx904",
    llvm_generic=None,
    members=(GFX904,),
)
BUNDLE_GFX906 = TargetBundle(
    key="gfx906",
    level=_TARGET,
    display_name="gfx906",
    llvm_generic=None,
    members=(GFX906,),
)
BUNDLE_GFX908 = TargetBundle(
    key="gfx908",
    level=_TARGET,
    display_name="gfx908 (MI100)",
    llvm_generic=None,
    members=(GFX908,),
)
BUNDLE_GFX909 = TargetBundle(
    key="gfx909",
    level=_TARGET,
    display_name="gfx909",
    llvm_generic=None,
    members=(GFX909,),
)
BUNDLE_GFX90A = TargetBundle(
    key="gfx90a",
    level=_TARGET,
    display_name="gfx90a (MI200)",
    llvm_generic=None,
    members=(GFX90A,),
)
BUNDLE_GFX90C = TargetBundle(
    key="gfx90c",
    level=_TARGET,
    display_name="gfx90c",
    llvm_generic=None,
    members=(GFX90C,),
)
BUNDLE_GFX942 = TargetBundle(
    key="gfx942",
    level=_TARGET,
    display_name="gfx942 (MI300)",
    llvm_generic=None,
    members=(GFX942,),
)
BUNDLE_GFX950 = TargetBundle(
    key="gfx950",
    level=_TARGET,
    display_name="gfx950 (MI350)",
    llvm_generic=None,
    members=(GFX950,),
)
BUNDLE_GFX1010 = TargetBundle(
    key="gfx1010",
    level=_TARGET,
    display_name="gfx1010",
    llvm_generic=None,
    members=(GFX1010,),
)
BUNDLE_GFX1011 = TargetBundle(
    key="gfx1011",
    level=_TARGET,
    display_name="gfx1011",
    llvm_generic=None,
    members=(GFX1011,),
)
BUNDLE_GFX1012 = TargetBundle(
    key="gfx1012",
    level=_TARGET,
    display_name="gfx1012",
    llvm_generic=None,
    members=(GFX1012,),
)
BUNDLE_GFX1013 = TargetBundle(
    key="gfx1013",
    level=_TARGET,
    display_name="gfx1013",
    llvm_generic=None,
    members=(GFX1013,),
)
BUNDLE_GFX1030 = TargetBundle(
    key="gfx1030",
    level=_TARGET,
    display_name="gfx1030",
    llvm_generic=None,
    members=(GFX1030,),
)
BUNDLE_GFX1031 = TargetBundle(
    key="gfx1031",
    level=_TARGET,
    display_name="gfx1031",
    llvm_generic=None,
    members=(GFX1031,),
)
BUNDLE_GFX1032 = TargetBundle(
    key="gfx1032",
    level=_TARGET,
    display_name="gfx1032",
    llvm_generic=None,
    members=(GFX1032,),
)
BUNDLE_GFX1033 = TargetBundle(
    key="gfx1033",
    level=_TARGET,
    display_name="gfx1033",
    llvm_generic=None,
    members=(GFX1033,),
)
BUNDLE_GFX1034 = TargetBundle(
    key="gfx1034",
    level=_TARGET,
    display_name="gfx1034",
    llvm_generic=None,
    members=(GFX1034,),
)
BUNDLE_GFX1035 = TargetBundle(
    key="gfx1035",
    level=_TARGET,
    display_name="gfx1035",
    llvm_generic=None,
    members=(GFX1035,),
)
BUNDLE_GFX1036 = TargetBundle(
    key="gfx1036",
    level=_TARGET,
    display_name="gfx1036",
    llvm_generic=None,
    members=(GFX1036,),
)
BUNDLE_GFX1100 = TargetBundle(
    key="gfx1100",
    level=_TARGET,
    display_name="gfx1100",
    llvm_generic=None,
    members=(GFX1100,),
)
BUNDLE_GFX1101 = TargetBundle(
    key="gfx1101",
    level=_TARGET,
    display_name="gfx1101",
    llvm_generic=None,
    members=(GFX1101,),
)
BUNDLE_GFX1102 = TargetBundle(
    key="gfx1102",
    level=_TARGET,
    display_name="gfx1102",
    llvm_generic=None,
    members=(GFX1102,),
)
BUNDLE_GFX1103 = TargetBundle(
    key="gfx1103",
    level=_TARGET,
    display_name="gfx1103",
    llvm_generic=None,
    members=(GFX1103,),
)
BUNDLE_GFX1150 = TargetBundle(
    key="gfx1150",
    level=_TARGET,
    display_name="gfx1150 (Strix Point)",
    llvm_generic=None,
    members=(GFX1150,),
)
BUNDLE_GFX1151 = TargetBundle(
    key="gfx1151",
    level=_TARGET,
    display_name="gfx1151 (Strix Halo)",
    llvm_generic=None,
    members=(GFX1151,),
)
BUNDLE_GFX1152 = TargetBundle(
    key="gfx1152",
    level=_TARGET,
    display_name="gfx1152",
    llvm_generic=None,
    members=(GFX1152,),
)
BUNDLE_GFX1153 = TargetBundle(
    key="gfx1153",
    level=_TARGET,
    display_name="gfx1153",
    llvm_generic=None,
    members=(GFX1153,),
)
BUNDLE_GFX1200 = TargetBundle(
    key="gfx1200",
    level=_TARGET,
    display_name="gfx1200",
    llvm_generic=None,
    members=(GFX1200,),
)
BUNDLE_GFX1201 = TargetBundle(
    key="gfx1201",
    level=_TARGET,
    display_name="gfx1201",
    llvm_generic=None,
    members=(GFX1201,),
)
BUNDLE_GFX1250 = TargetBundle(
    key="gfx1250",
    level=_TARGET,
    display_name="gfx1250",
    llvm_generic=None,
    members=(GFX1250,),
)
BUNDLE_GFX1251 = TargetBundle(
    key="gfx1251",
    level=_TARGET,
    display_name="gfx1251",
    llvm_generic=None,
    members=(GFX1251,),
)


# ---------------------------------------------------------------------------
# Bundles — sub-family level (major + minor)
# ---------------------------------------------------------------------------

_SF = PackagingLevel.SUB_FAMILY

# gfx9-generic covers gfx900, 902, 904, 906, 909, 90c (NOT gfx908, gfx90a).
# The sub-family includes all structurally; the generic is partial coverage.
BUNDLE_GFX9_0 = TargetBundle(
    key="gfx9_0",
    level=_SF,
    display_name="Vega",
    llvm_generic="gfx9-generic",
    members=(GFX900, GFX902, GFX904, GFX906, GFX908, GFX909, GFX90A, GFX90C),
)

BUNDLE_GFX9_4 = TargetBundle(
    key="gfx9_4",
    level=_SF,
    display_name="MI300",
    llvm_generic="gfx9-4-generic",
    members=(GFX942,),
)

BUNDLE_GFX9_5 = TargetBundle(
    key="gfx9_5",
    level=_SF,
    display_name="MI350",
    llvm_generic=None,
    members=(GFX950,),
)

BUNDLE_GFX10_1 = TargetBundle(
    key="gfx10_1",
    level=_SF,
    display_name="RDNA1",
    llvm_generic="gfx10-1-generic",
    members=(GFX1010, GFX1011, GFX1012, GFX1013),
)

BUNDLE_GFX10_3 = TargetBundle(
    key="gfx10_3",
    level=_SF,
    display_name="RDNA2",
    llvm_generic="gfx10-3-generic",
    members=(GFX1030, GFX1031, GFX1032, GFX1033, GFX1034, GFX1035, GFX1036),
)

BUNDLE_GFX11_0 = TargetBundle(
    key="gfx11_0",
    level=_SF,
    display_name="RDNA3",
    llvm_generic=None,
    members=(GFX1100, GFX1101, GFX1102, GFX1103),
)

BUNDLE_GFX11_5 = TargetBundle(
    key="gfx11_5",
    level=_SF,
    display_name="RDNA3.5",
    llvm_generic=None,
    members=(GFX1150, GFX1151, GFX1152, GFX1153),
)

# gfx12-generic covers gfx1200, gfx1201 only (not gfx1250+).
BUNDLE_GFX12_0 = TargetBundle(
    key="gfx12_0",
    level=_SF,
    display_name="RDNA4",
    llvm_generic="gfx12-generic",
    members=(GFX1200, GFX1201),
)

BUNDLE_GFX12_5 = TargetBundle(
    key="gfx12_5",
    level=_SF,
    display_name="GFX12.5",
    llvm_generic=None,
    members=(GFX1250, GFX1251),
)


# ---------------------------------------------------------------------------
# Bundles — family level (major only)
# ---------------------------------------------------------------------------

_F = PackagingLevel.FAMILY

BUNDLE_GFX9 = TargetBundle(
    key="gfx9",
    level=_F,
    display_name="GFX9",
    llvm_generic=None,
    members=(
        GFX900,
        GFX902,
        GFX904,
        GFX906,
        GFX908,
        GFX909,
        GFX90A,
        GFX90C,
        GFX942,
        GFX950,
    ),
)

BUNDLE_GFX10 = TargetBundle(
    key="gfx10",
    level=_F,
    display_name="GFX10",
    llvm_generic=None,
    members=(
        GFX1010,
        GFX1011,
        GFX1012,
        GFX1013,
        GFX1030,
        GFX1031,
        GFX1032,
        GFX1033,
        GFX1034,
        GFX1035,
        GFX1036,
    ),
)

# gfx11-generic covers all of GFX11 (both 11.0 and 11.5).
BUNDLE_GFX11 = TargetBundle(
    key="gfx11",
    level=_F,
    display_name="RDNA3",
    llvm_generic="gfx11-generic",
    members=(GFX1100, GFX1101, GFX1102, GFX1103, GFX1150, GFX1151, GFX1152, GFX1153),
)

BUNDLE_GFX12 = TargetBundle(
    key="gfx12",
    level=_F,
    display_name="RDNA4",
    llvm_generic=None,
    members=(GFX1200, GFX1201, GFX1250, GFX1251),
)


# ---------------------------------------------------------------------------
# Canonical ordered collections
# ---------------------------------------------------------------------------

ALL_TARGETS: tuple[GfxTarget, ...] = (
    GFX900,
    GFX902,
    GFX904,
    GFX906,
    GFX908,
    GFX909,
    GFX90A,
    GFX90C,
    GFX942,
    GFX950,
    GFX1010,
    GFX1011,
    GFX1012,
    GFX1013,
    GFX1030,
    GFX1031,
    GFX1032,
    GFX1033,
    GFX1034,
    GFX1035,
    GFX1036,
    GFX1100,
    GFX1101,
    GFX1102,
    GFX1103,
    GFX1150,
    GFX1151,
    GFX1152,
    GFX1153,
    GFX1200,
    GFX1201,
    GFX1250,
    GFX1251,
)

ALL_FAMILIES: tuple[TargetBundle, ...] = (
    BUNDLE_GFX9,
    BUNDLE_GFX10,
    BUNDLE_GFX11,
    BUNDLE_GFX12,
)

ALL_SUB_FAMILIES: tuple[TargetBundle, ...] = (
    BUNDLE_GFX9_0,
    BUNDLE_GFX9_4,
    BUNDLE_GFX9_5,
    BUNDLE_GFX10_1,
    BUNDLE_GFX10_3,
    BUNDLE_GFX11_0,
    BUNDLE_GFX11_5,
    BUNDLE_GFX12_0,
    BUNDLE_GFX12_5,
)

# Sub-families grouped by their parent family, for hierarchy validation.
_FAMILY_TO_SUB_FAMILIES: dict[str, tuple[TargetBundle, ...]] = {
    "gfx9": (BUNDLE_GFX9_0, BUNDLE_GFX9_4, BUNDLE_GFX9_5),
    "gfx10": (BUNDLE_GFX10_1, BUNDLE_GFX10_3),
    "gfx11": (BUNDLE_GFX11_0, BUNDLE_GFX11_5),
    "gfx12": (BUNDLE_GFX12_0, BUNDLE_GFX12_5),
}


# ---------------------------------------------------------------------------
# Internal lookup tables — built once at import time
# ---------------------------------------------------------------------------


def _build_target_bundle_map() -> dict[str, TargetBundle]:
    """Build name -> target-level bundle map."""
    result: dict[str, TargetBundle] = {}
    for sf in ALL_SUB_FAMILIES:
        for t in sf.members:
            # Find the target-level bundle constant
            bundle_var_name = f"BUNDLE_{t.name.upper().replace('GFX', 'GFX')}"
            # Instead of getattr games, build from the target directly
            key = t.name
            if key in result:
                raise RuntimeError(f"Duplicate target name: {key}")
            result[key] = TargetBundle(
                key=key,
                level=PackagingLevel.TARGET,
                display_name=t.name,
                llvm_generic=None,
                members=(t,),
            )
    return result


# Map: target name -> GfxTarget
_TARGET_BY_NAME: dict[str, GfxTarget] = {t.name: t for t in ALL_TARGETS}

# Map: gfx_target_version int -> GfxTarget
_TARGET_BY_GTV: dict[int, GfxTarget] = {t.gfx_target_version: t for t in ALL_TARGETS}

# Map: bundle key -> TargetBundle (all levels)
_BUNDLE_BY_KEY: dict[str, TargetBundle] = {}

# Map: target name -> sub-family bundle
_SUB_FAMILY_BY_TARGET: dict[str, TargetBundle] = {}

# Map: sub-family key -> family bundle
_FAMILY_BY_SUB_FAMILY: dict[str, TargetBundle] = {}


def _build_lookups() -> None:
    """Populate all lookup tables. Called once at module import."""
    # Register family bundles
    for fam in ALL_FAMILIES:
        _BUNDLE_BY_KEY[fam.key] = fam

    # Register sub-family bundles and target->sub-family mapping
    for sf in ALL_SUB_FAMILIES:
        _BUNDLE_BY_KEY[sf.key] = sf
        for t in sf.members:
            if t.name in _SUB_FAMILY_BY_TARGET:
                raise RuntimeError(
                    f"Target {t.name} appears in multiple sub-families: "
                    f"{_SUB_FAMILY_BY_TARGET[t.name].key} and {sf.key}"
                )
            _SUB_FAMILY_BY_TARGET[t.name] = sf

    # Register target-level bundles
    for t in ALL_TARGETS:
        bundle = TargetBundle(
            key=t.name,
            level=PackagingLevel.TARGET,
            display_name=t.name,
            llvm_generic=None,
            members=(t,),
        )
        _BUNDLE_BY_KEY[t.name] = bundle

    # Build sub-family -> family mapping
    for fam_key, sfs in _FAMILY_TO_SUB_FAMILIES.items():
        fam = _BUNDLE_BY_KEY[fam_key]
        for sf in sfs:
            _FAMILY_BY_SUB_FAMILY[sf.key] = fam


_build_lookups()


# ---------------------------------------------------------------------------
# Public lookup functions
# ---------------------------------------------------------------------------


def lookup_target(name: str) -> GfxTarget:
    """Look up a :class:`GfxTarget` by canonical name.

    Args:
        name: GFX target name (e.g., ``"gfx1100"``).

    Returns:
        The canonical :class:`GfxTarget` instance.

    Raises:
        ValueError: If the name is not a known target.
    """
    try:
        return _TARGET_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"Unknown GFX target: {name!r}. "
            f"Known targets: {', '.join(sorted(_TARGET_BY_NAME))}"
        ) from None


def lookup_bundle(key: str) -> TargetBundle:
    """Look up a :class:`TargetBundle` by key.

    Works for all levels (family, sub-family, target).

    Args:
        key: Bundle key (e.g., ``"gfx11"``, ``"gfx11_5"``, ``"gfx1151"``).

    Returns:
        The :class:`TargetBundle` instance.

    Raises:
        ValueError: If the key is not a known bundle.
    """
    try:
        return _BUNDLE_BY_KEY[key]
    except KeyError:
        raise ValueError(
            f"Unknown bundle key: {key!r}. "
            f"Known bundles: {', '.join(sorted(_BUNDLE_BY_KEY))}"
        ) from None


def packaging_chain(
    target: GfxTarget | str,
) -> tuple[TargetBundle, TargetBundle, TargetBundle]:
    """Get the 3-level packaging chain for a target.

    Returns bundles from most specific to most general:
    ``(target_bundle, sub_family_bundle, family_bundle)``.

    Example::

        packaging_chain("gfx1151") == (
            TargetBundle("gfx1151", TARGET, ...),
            TargetBundle("gfx11_5", SUB_FAMILY, ...),
            TargetBundle("gfx11", FAMILY, ...),
        )

    Args:
        target: A :class:`GfxTarget` or target name string.

    Returns:
        3-tuple of :class:`TargetBundle` (target, sub-family, family).

    Raises:
        ValueError: If the target is not known.
    """
    if isinstance(target, str):
        target = lookup_target(target)
    name = target.name

    target_bundle = _BUNDLE_BY_KEY[name]
    sf_bundle = _SUB_FAMILY_BY_TARGET[name]
    fam_bundle = _FAMILY_BY_SUB_FAMILY[sf_bundle.key]

    return (target_bundle, sf_bundle, fam_bundle)


def bundle_for_target(target: GfxTarget | str, level: PackagingLevel) -> TargetBundle:
    """Get the bundle at a specific level for a target.

    Args:
        target: A :class:`GfxTarget` or target name string.
        level: Which level to return.

    Returns:
        The :class:`TargetBundle` at the requested level.

    Raises:
        ValueError: If the target is not known.
    """
    chain = packaging_chain(target)
    idx = {
        PackagingLevel.TARGET: 0,
        PackagingLevel.SUB_FAMILY: 1,
        PackagingLevel.FAMILY: 2,
    }
    return chain[idx[level]]


def all_bundles(level: PackagingLevel | None = None) -> tuple[TargetBundle, ...]:
    """Get all bundles, optionally filtered by level.

    Args:
        level: If given, only return bundles at this level.
            If ``None``, return all bundles at all levels.

    Returns:
        Tuple of :class:`TargetBundle` instances.
    """
    if level is None:
        return tuple(_BUNDLE_BY_KEY.values())
    if level == PackagingLevel.FAMILY:
        return ALL_FAMILIES
    if level == PackagingLevel.SUB_FAMILY:
        return ALL_SUB_FAMILIES
    # TARGET level
    return tuple(b for b in _BUNDLE_BY_KEY.values() if b.level == PackagingLevel.TARGET)


def parse_gfx_target_version(gtv: int) -> GfxTarget:
    """Parse a ``gfx_target_version`` integer from sysfs into a :class:`GfxTarget`.

    The encoding is: ``major * 10000 + minor * 100 + stepping``.

    Examples:
        - ``120001`` → gfx1201
        - ``110000`` → gfx1100
        - ``90402``  → gfx942
        - ``110100`` → gfx1101
        - ``115100`` → gfx1151

    Args:
        gtv: The ``gfx_target_version`` integer.

    Returns:
        The matching :class:`GfxTarget`.

    Raises:
        ValueError: If no known target matches the decoded version.
    """
    try:
        return _TARGET_BY_GTV[gtv]
    except KeyError:
        major = gtv // 10000
        minor = (gtv % 10000) // 100
        stepping = gtv % 100
        raise ValueError(
            f"Unknown gfx_target_version: {gtv} "
            f"(decoded: major={major}, minor={minor}, stepping={stepping})"
        ) from None
