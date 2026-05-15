"""rocm-bootstrap: GPU detection and target groupings for the ROCm Python ecosystem.

This package provides:
    - Canonical GFX target hierarchy (family/sub-family/target)
    - Pure-Python GPU detection via Linux sysfs
    - Python dist-safe and module-safe naming APIs
    - WheelNext variant plugin for AMD GPU detection
"""

from rocm_bootstrap.detect import DetectedGpu, detect_gfx_targets, detect_gpus
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
    GfxTarget,
    PackagingLevel,
    TargetBundle,
    XnackMode,
    all_bundles,
    bundle_for_target,
    lookup_bundle,
    lookup_target,
    packaging_chain,
    parse_gfx_target_version,
)

__all__ = [
    # targets
    "GfxTarget",
    "TargetBundle",
    "PackagingLevel",
    "XnackMode",
    "ALL_FAMILIES",
    "ALL_SUB_FAMILIES",
    "ALL_TARGETS",
    "all_bundles",
    "bundle_for_target",
    "lookup_bundle",
    "lookup_target",
    "packaging_chain",
    "parse_gfx_target_version",
    # naming
    "PackageNames",
    "bundle_names",
    "device_dist_name",
    "device_module_name",
    "is_valid_dist_name",
    "is_valid_module_name",
    # detect
    "DetectedGpu",
    "detect_gpus",
    "detect_gfx_targets",
]
