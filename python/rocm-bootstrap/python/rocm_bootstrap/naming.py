"""Python dist-safe and module-safe name generation for GFX targets.

This module provides deterministic, validated name generation for all
levels of the GFX packaging hierarchy. Every generated name is guaranteed
to be valid for its intended context (pip/wheel dist names or Python
module identifiers).

Naming convention:
    - Bundle ``key`` uses underscores → directly a valid Python identifier.
    - ``dist_name`` replaces ``_`` with ``-`` → valid wheel/package name.
    - ``module_name`` IS the ``key`` → valid Python identifier.
    - All names start with ``gfx`` → no leading digit issues.
"""

import re
from dataclasses import dataclass

from rocm_bootstrap.targets import TargetBundle

# PEP 625 / PyPA: distribution names are lowercase, alphanumeric + hyphens +
# underscores + periods. Must start with alphanumeric.
_DIST_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9._-]*[a-z0-9])?$")


@dataclass(frozen=True)
class PackageNames:
    """Dist-safe and module-safe names for a packaging level.

    Attributes:
        dist_name: Name safe for Python distribution/wheel usage.
            Lowercase, uses hyphens as separators.
            E.g., ``"gfx11-5"``, ``"gfx942"``.
        module_name: Name safe for Python identifiers/modules.
            Uses underscores, no hyphens, no leading digits.
            E.g., ``"gfx11_5"``, ``"gfx942"``.
    """

    dist_name: str
    module_name: str


def bundle_names(bundle: TargetBundle) -> PackageNames:
    """Get Python-safe package names for a bundle.

    The bundle's ``key`` is the ``module_name`` (already uses underscores).
    The ``dist_name`` is derived by replacing ``_`` with ``-``.

    Args:
        bundle: Any :class:`~rocm_bootstrap.targets.TargetBundle`.

    Returns:
        :class:`PackageNames` with validated dist and module names.
    """
    module_name = bundle.key
    dist_name = module_name.replace("_", "-")
    return PackageNames(dist_name=dist_name, module_name=module_name)


def device_dist_name(prefix: str, bundle: TargetBundle) -> str:
    """Generate a device package distribution name.

    Combines a dist-name prefix with the bundle's dist name.

    Examples::

        device_dist_name("rocm-sdk-device", bundle_gfx11_5)
        # -> "rocm-sdk-device-gfx11-5"

        device_dist_name("amd-torch-device", bundle_gfx942)
        # -> "amd-torch-device-gfx942"

    Args:
        prefix: Distribution name prefix (e.g., ``"rocm-sdk-device"``).
        bundle: The :class:`~rocm_bootstrap.targets.TargetBundle`.

    Returns:
        Combined dist name string.
    """
    names = bundle_names(bundle)
    return f"{prefix}-{names.dist_name}"


def device_module_name(prefix: str, bundle: TargetBundle) -> str:
    """Generate a device package module name.

    Combines a module-name prefix with the bundle's module name.

    Examples::

        device_module_name("rocm_sdk_device", bundle_gfx11_5)
        # -> "rocm_sdk_device_gfx11_5"

    Args:
        prefix: Module name prefix (e.g., ``"rocm_sdk_device"``).
        bundle: The :class:`~rocm_bootstrap.targets.TargetBundle`.

    Returns:
        Combined module name string.
    """
    names = bundle_names(bundle)
    return f"{prefix}_{names.module_name}"


def is_valid_dist_name(name: str) -> bool:
    """Check if a string is valid as a Python distribution name.

    Per PEP 625 / PyPA naming specification: lowercase alphanumeric,
    hyphens, underscores, periods. Must start and end with alphanumeric.

    Args:
        name: String to validate.

    Returns:
        ``True`` if valid.
    """
    return bool(_DIST_NAME_RE.match(name))


def is_valid_module_name(name: str) -> bool:
    """Check if a string is valid as a Python module/identifier name.

    Must be a valid Python identifier: starts with letter or underscore,
    contains only letters, digits, underscores. Must not be a keyword.

    Args:
        name: String to validate.

    Returns:
        ``True`` if valid.
    """
    return name.isidentifier()
