"""Shared fixtures for rocm_bootstrap tests.

Provides monkeypatched _platform functions with realistic sysfs content
for every known GFX target.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rocm_bootstrap import _platform
from rocm_bootstrap.targets import ALL_TARGETS, GfxTarget

# ---------------------------------------------------------------------------
# Realistic KFD properties templates
# ---------------------------------------------------------------------------

# Template based on a real gfx1201 KFD node. The gfx_target_version and
# device_id fields are filled per target.
_KFD_GPU_TEMPLATE = """\
cpu_cores_count 0
simd_count 128
mem_banks_count 1
caches_count 138
io_links_count 1
p2p_links_count 1
cpu_core_id_base 0
simd_id_base 2147487744
max_waves_per_simd 16
lds_size_in_kb 64
gds_size_in_kb 0
num_gws 0
wave_front_size 32
array_count 8
simd_arrays_per_engine 2
cu_per_simd_array 8
simd_per_cu 2
max_slots_scratch_cu 32
gfx_target_version {gtv}
vendor_id 4098
device_id {device_id}
location_id 33792
domain 0
drm_render_minor 128
hive_id 0
num_sdma_engines 2
num_sdma_xgmi_engines 0
num_sdma_queues_per_engine 6
num_cp_queues 4
max_engine_clk_fcompute 2460
"""

_KFD_CPU_PROPERTIES = """\
cpu_cores_count 192
simd_count 0
mem_banks_count 1
caches_count 0
io_links_count 2
p2p_links_count 0
cpu_core_id_base 0
simd_id_base 0
max_waves_per_simd 0
lds_size_in_kb 0
gds_size_in_kb 0
num_gws 0
wave_front_size 0
array_count 0
simd_arrays_per_engine 0
cu_per_simd_array 0
simd_per_cu 0
max_slots_scratch_cu 0
gfx_target_version 0
vendor_id 0
device_id 0
location_id 0
domain 0
drm_render_minor 0
hive_id 0
num_sdma_engines 0
num_sdma_xgmi_engines 0
num_sdma_queues_per_engine 0
num_cp_queues 0
max_engine_clk_ccompute 5391
"""


def kfd_gpu_properties(target: GfxTarget, device_id: int = 30000) -> str:
    """Generate realistic KFD properties content for a GPU target."""
    return _KFD_GPU_TEMPLATE.format(
        gtv=target.gfx_target_version,
        device_id=device_id,
    )


# ---------------------------------------------------------------------------
# Platform patching helpers
# ---------------------------------------------------------------------------


class FakePlatform:
    """Configurable fake for _platform module functions.

    Usage::

        fake = FakePlatform()
        fake.add_cpu_node(0)
        fake.add_gpu_node(1, some_gfx_target)
        fake.apply(monkeypatch)
    """

    def __init__(self) -> None:
        self.kfd_nodes: dict[int, str] = {}  # node_id -> properties text
        self.env: dict[str, str] = {}

    def add_cpu_node(self, node_id: int) -> None:
        """Add a CPU-only KFD node."""
        self.kfd_nodes[node_id] = _KFD_CPU_PROPERTIES

    def add_gpu_node(
        self,
        node_id: int,
        target: GfxTarget,
        device_id: int = 30000,
    ) -> None:
        """Add a GPU KFD node."""
        self.kfd_nodes[node_id] = kfd_gpu_properties(target, device_id)

    def set_env(self, key: str, value: str) -> None:
        """Set a fake environment variable."""
        self.env[key] = value

    def apply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Monkeypatch _platform functions with this fake's state."""
        kfd_nodes = self.kfd_nodes
        env = self.env

        def fake_list_kfd_nodes() -> list[Path]:
            return [Path(f"/fake/kfd/nodes/{n}") for n in sorted(kfd_nodes)]

        def fake_read_kfd_properties(node_path: Path) -> str:
            node_id = int(node_path.name)
            if node_id not in kfd_nodes:
                raise FileNotFoundError(f"No fake node {node_id}")
            return kfd_nodes[node_id]

        def fake_get_env(key: str) -> str | None:
            return env.get(key)

        monkeypatch.setattr(_platform, "list_kfd_nodes", fake_list_kfd_nodes)
        monkeypatch.setattr(_platform, "read_kfd_properties", fake_read_kfd_properties)
        monkeypatch.setattr(_platform, "get_env", fake_get_env)


@pytest.fixture()
def fake_platform(monkeypatch: pytest.MonkeyPatch) -> FakePlatform:
    """A FakePlatform with empty state, already applied to monkeypatch.

    Tests can add nodes/cards/env vars and the patching is already active.
    """
    fake = FakePlatform()
    fake.apply(monkeypatch)
    return fake
