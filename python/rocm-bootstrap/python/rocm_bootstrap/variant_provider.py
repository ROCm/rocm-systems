"""WheelNext variant plugin for AMD GPU detection.

Implements the ``variant_plugins`` entry point protocol defined by
`variantlib <https://github.com/wheelnext/variantlib>`_. Reports
detected GFX architectures as variant features so that uv/pip can
select the correct device-specific wheel.

This plugin is duck-typed against ``variantlib.protocols.PluginType``
and does NOT depend on variantlib at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocm_bootstrap.detect import detect_gfx_targets
from rocm_bootstrap.targets import ALL_TARGETS


@dataclass(frozen=True)
class VariantFeatureConfig:
    """A variant feature configuration (duck-typed variantlib protocol).

    Attributes:
        name: Feature name (e.g., ``"gfx_arch"``).
        values: Valid values in priority order (most preferred first).
        multi_value: Whether a single installation can carry multiple
            values for this feature simultaneously.
    """

    name: str
    values: list[str]
    multi_value: bool = False


class AMDVariantPlugin:
    """WheelNext variant plugin for AMD GPUs.

    Detects AMD GPUs on the current system and reports their GFX
    architecture as a variant feature. This allows package installers
    to select the correct device-specific wheel.

    Registered via ``[project.entry-points.variant_plugins]`` in
    pyproject.toml.
    """

    namespace = "amd"
    is_aot_plugin = False

    @classmethod
    def get_all_configs(cls) -> list[VariantFeatureConfig]:
        """All possible GFX architecture values.

        Built dynamically from the target registry — no hardcoded lists.
        Returns every known GFX target as a possible value.
        """
        return [
            VariantFeatureConfig(
                name="gfx_arch",
                values=[t.name for t in ALL_TARGETS],
                multi_value=True,
            ),
        ]

    @classmethod
    def get_supported_configs(cls) -> list[VariantFeatureConfig]:
        """GFX architectures detected on the current system.

        Returns an empty list if no AMD GPUs are detected (e.g., on
        Intel/NVIDIA systems or when detection is disabled).
        """
        targets = detect_gfx_targets()
        if not targets:
            return []
        return [
            VariantFeatureConfig(
                name="gfx_arch",
                values=[t.name for t in targets],
                multi_value=True,
            ),
        ]
