# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import ctypes
import os
import sys
from types import SimpleNamespace

import pytest

from utils import rocprofv3_avail_interface as avail_interface

LIBRARY_NAME = avail_interface.LIBRARY_NAME
BETA_GATE = "ROCPROFILER_PC_SAMPLING_BETA_ENABLED"


@pytest.fixture(autouse=True)
def reset_library_cache(monkeypatch):
    """Give each test an unloaded interface, since the handle is cached."""
    monkeypatch.setattr(avail_interface, "_library", None)
    monkeypatch.setattr(avail_interface, "_library_path", None)


@pytest.fixture
def rocm_tree(tmp_path):
    """A ROCm install holding the rocprofv3 tool but neither library yet.

    Both directories the interface probes exist, so tests opt into a layout by
    creating the library file in one of them.
    """
    tool_path = tmp_path / "lib" / "rocprofiler-sdk" / "rocprofv3"
    tool_path.parent.mkdir(parents=True)
    tool_path.touch()
    libexec_dir = tmp_path / "libexec" / "rocprofiler-sdk"
    libexec_dir.mkdir(parents=True)
    return SimpleNamespace(
        tool_path=str(tool_path),
        lib_layout=tool_path.parent / LIBRARY_NAME,
        libexec_layout=libexec_dir / LIBRARY_NAME,
    )


def fake_avail_library(configs_by_agent, gate_per_call=None):
    """Stand-in for the CDLL, serving canned per-agent configurations.

    Only the entry points the interface calls are defined, so a wrong C symbol
    name raises here instead of quietly succeeding. Plain functions are used
    because the interface assigns restype/argtypes onto them.
    """
    handles = sorted(configs_by_agent)

    def get_number_of_agents():
        if gate_per_call is not None:
            gate_per_call.append(os.environ.get(BETA_GATE))
        return len(handles)

    def agent_handles(buffer, count):
        buffer[0:count] = handles

    def get_number_of_pc_sample_configs(agent_handle):
        return len(configs_by_agent[agent_handle])

    def pc_sample_config(agent_handle, config_index, *fields):
        values = configs_by_agent[agent_handle][config_index]
        for field, value in zip(fields, values):
            field._obj.value = value

    return SimpleNamespace(
        get_number_of_agents=get_number_of_agents,
        agent_handles=agent_handles,
        get_number_of_pc_sample_configs=get_number_of_pc_sample_configs,
        pc_sample_config=pc_sample_config,
    )


def patch_library_load(monkeypatch, library):
    """Serve `library` from every dlopen and record the paths asked for."""
    loaded_paths = []

    def load(library_path):
        loaded_paths.append(library_path)
        return library

    monkeypatch.setattr(ctypes, "CDLL", load)
    return loaded_paths


def agent_metadata(node_id, location_id):
    """Model SDK metadata for two GPUs with the same architecture and chip."""
    return {
        "name": "gfx1201",
        "device_id": 30032,
        "node_id": node_id,
        "location_id": location_id,
    }


# ---------------------------------------------------------------------------
# library path resolution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "layouts, expected_layout",
    [
        pytest.param(["lib_layout"], "lib_layout", id="rocm_7_1_lib_layout"),
        pytest.param(
            ["libexec_layout"], "libexec_layout", id="rocm_7_0_libexec_layout"
        ),
        pytest.param(
            ["lib_layout", "libexec_layout"],
            "lib_layout",
            id="lib_layout_wins_when_both_exist",
        ),
        pytest.param([], None, id="neither_layout_present"),
    ],
)
def test_resolve_library_path_across_rocm_layouts(rocm_tree, layouts, expected_layout):
    """The library moved to libexec on ROCm 7.0.x, so both are probed."""
    for layout in layouts:
        getattr(rocm_tree, layout).touch()

    resolved = avail_interface.resolve_library_path(rocm_tree.tool_path)

    expected = expected_layout and str(getattr(rocm_tree, expected_layout))
    assert resolved == expected


def test_resolve_library_path_without_tool_path():
    """No sdk tool path means there is nothing to resolve against."""
    assert avail_interface.resolve_library_path(None) is None


# ---------------------------------------------------------------------------
# beta gate handling
# ---------------------------------------------------------------------------
def test_beta_gate_is_set_for_warmup_call_and_restored(monkeypatch, rocm_tree):
    """The gate covers the warm-up call and is gone once loading finishes.

    The library enumerates PC sampling configurations on the first call into
    it, so a gate that only covers the dlopen has no effect.
    """
    monkeypatch.delenv(BETA_GATE, raising=False)
    rocm_tree.lib_layout.touch()
    gate_per_call = []
    patch_library_load(monkeypatch, fake_avail_library({}, gate_per_call))

    avail_interface.get_pc_sample_configs(rocm_tree.tool_path)

    # Warm-up call sees the gate, the later query call no longer needs it.
    assert gate_per_call == ["ON", None]
    assert BETA_GATE not in os.environ


def test_beta_gate_restores_preexisting_value(monkeypatch, rocm_tree):
    """A gate the user already set is put back, not clobbered."""
    monkeypatch.setenv(BETA_GATE, "off")
    rocm_tree.lib_layout.touch()
    patch_library_load(monkeypatch, fake_avail_library({}))

    avail_interface.get_pc_sample_configs(rocm_tree.tool_path)

    assert os.environ[BETA_GATE] == "off"


def test_beta_gate_restored_when_load_fails(monkeypatch, rocm_tree):
    """A library that will not load still leaves the environment clean."""
    monkeypatch.delenv(BETA_GATE, raising=False)
    rocm_tree.lib_layout.touch()

    def fail_to_load(_library_path):
        raise OSError("file too short")

    monkeypatch.setattr(ctypes, "CDLL", fail_to_load)

    assert avail_interface.get_pc_sample_configs(rocm_tree.tool_path) is None
    assert BETA_GATE not in os.environ


# ---------------------------------------------------------------------------
# configuration marshalling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "configs_by_agent, expected",
    [
        pytest.param(
            {0: [(1, 2, 256, 1048576, 1)]},
            [(1, 2, 256, 1048576, 1)],
            id="single_agent_single_config",
        ),
        pytest.param(
            {0: [(1, 2, 256, 1048576, 1), (2, 3, 1, 1048576, 0)]},
            [(1, 2, 256, 1048576, 1), (2, 3, 1, 1048576, 0)],
            id="single_agent_both_methods",
        ),
        pytest.param(
            {0: [(1, 2, 512, 65536, 1)], 1: [(1, 2, 256, 1048576, 0)]},
            [(1, 2, 512, 65536, 1), (1, 2, 256, 1048576, 0)],
            id="configs_flattened_across_agents",
        ),
        pytest.param({0: []}, [], id="agent_reports_no_configs"),
    ],
)
def test_get_pc_sample_configs(monkeypatch, rocm_tree, configs_by_agent, expected):
    """Out-parameters are marshalled into plain tuples for every agent."""
    rocm_tree.lib_layout.touch()
    patch_library_load(monkeypatch, fake_avail_library(configs_by_agent))

    configs = avail_interface.get_pc_sample_configs(rocm_tree.tool_path)

    assert configs == expected


def test_get_pc_sample_configs_without_library(rocm_tree):
    """None distinguishes an unreachable library from a silent agent."""
    assert avail_interface.get_pc_sample_configs(rocm_tree.tool_path) is None


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------
def test_library_is_loaded_once(monkeypatch, rocm_tree):
    """Repeated queries reuse the handle instead of dlopening again.

    Reloading cannot undo an ungated first load, so the gate and the handle
    are settled once per process.
    """
    rocm_tree.lib_layout.touch()
    library = fake_avail_library({0: [(1, 2, 256, 1048576, 1)]})
    loaded_paths = patch_library_load(monkeypatch, library)

    avail_interface.get_pc_sample_configs(rocm_tree.tool_path)
    avail_interface.get_pc_sample_configs(rocm_tree.tool_path)

    assert loaded_paths == [str(rocm_tree.lib_layout)]


# ---------------------------------------------------------------------------
# counter listing
# ---------------------------------------------------------------------------
def test_get_counters_points_avail_module_at_resolved_library(monkeypatch, rocm_tree):
    """The avail module is aimed at the same file the gated load used."""
    rocm_tree.lib_layout.touch()
    patch_library_load(monkeypatch, fake_avail_library({}))
    avail_module = SimpleNamespace(
        loadLibrary=SimpleNamespace(libname=None),
        get_counters=lambda: {"agent-0": ["SQ_WAVES"]},
    )
    monkeypatch.setattr(
        avail_interface, "_import_avail_module", lambda _path: avail_module
    )

    counters = avail_interface.get_counters(rocm_tree.tool_path)

    assert avail_module.loadLibrary.libname == str(rocm_tree.lib_layout)
    assert counters == {"agent-0": ["SQ_WAVES"]}


def test_get_counters_errors_without_library(rocm_tree):
    """A missing library is fatal here, unlike the PC sampling query."""
    with pytest.raises(SystemExit):
        avail_interface.get_counters(rocm_tree.tool_path)


def test_query_agent_info_map_uses_the_counter_listing_library(monkeypatch, rocm_tree):
    """Metadata handles come from the same SDK library as counter handles."""
    rocm_tree.lib_layout.touch()
    patch_library_load(monkeypatch, fake_avail_library({}))
    agent_info = {31133: {"name": "gfx1201", "device_id": 30032}}
    avail_module = SimpleNamespace(
        loadLibrary=SimpleNamespace(libname=None),
        get_agent_info_map=lambda: agent_info,
    )
    monkeypatch.setattr(
        avail_interface, "_import_avail_module", lambda _path: avail_module
    )

    assert avail_interface.query_agent_info_map(rocm_tree.tool_path) == agent_info
    assert avail_module.loadLibrary.libname == str(rocm_tree.lib_layout)


def test_query_agent_info_map_errors_without_library(rocm_tree):
    with pytest.raises(SystemExit):
        avail_interface.query_agent_info_map(rocm_tree.tool_path)


@pytest.mark.parametrize(
    "node_id, bdf_id, agents, expected",
    [
        pytest.param(
            2,
            256,
            {91: agent_metadata(1, 256), 42: agent_metadata(2, 512)},
            None,
            id="node-match-cannot-hide-conflicting-bdf",
        ),
        pytest.param(
            9,
            512,
            {91: agent_metadata(1, 256), 42: agent_metadata(2, 512)},
            None,
            id="node-mismatch-does-not-fall-back-to-bdf",
        ),
        pytest.param(
            2,
            512,
            {91: agent_metadata(None, 256), 42: agent_metadata(None, 512)},
            {"COMMON", "SELECTED_GPU_ONLY"},
            id="missing-node-metadata-permits-unique-bdf-fallback",
        ),
        pytest.param(
            None,
            512,
            {91: agent_metadata(1, 256), 42: agent_metadata(2, 512)},
            {"COMMON", "SELECTED_GPU_ONLY"},
            id="bdf-fallback-without-selected-node",
        ),
        pytest.param(
            None,
            512,
            {91: agent_metadata(1, 512), 42: agent_metadata(2, 512)},
            None,
            id="ambiguous-bdf-is-unknown",
        ),
        pytest.param(
            None,
            None,
            {91: agent_metadata(1, 256), 42: agent_metadata(2, 512)},
            {"COMMON"},
            id="legacy-specs-intersect-matching-gpus",
        ),
    ],
)
def test_supported_counter_names_select_physical_gpu(node_id, bdf_id, agents, expected):
    counters = {
        91: [SimpleNamespace(name=name, block="SQ") for name in ("COMMON", "OTHER")],
        42: [
            SimpleNamespace(name=name, block="SQ")
            for name in ("COMMON", "SELECTED_GPU_ONLY")
        ],
    }
    assert (
        avail_interface.supported_counter_names_for_gpu(
            counters, agents, "gfx1201", "30032", node_id, bdf_id
        )
        == expected
    )


def test_supported_counter_names_normalize_raw_derived_and_constant_records():
    counters = {
        91: [
            SimpleNamespace(name="SQ_WAVES", block="SQ"),
            SimpleNamespace(name="CUSTOM_SUM", expression="reduce(SQ_WAVES,sum)"),
            SimpleNamespace(name="CU_COUNT"),
        ]
    }
    assert avail_interface.supported_counter_names_for_gpu(
        counters, {91: agent_metadata(1, 256)}, "gfx1201", "30032", 1
    ) == {"SQ_WAVES", "CUSTOM_SUM"}


def test_supported_counter_names_preserve_empty_selected_gpu_listing():
    assert (
        avail_interface.supported_counter_names_for_gpu(
            {91: [], 42: [SimpleNamespace(name="SQ_WAVES", block="SQ")]},
            {91: agent_metadata(1, 256), 42: agent_metadata(2, 512)},
            "gfx1201",
            "30032",
            1,
        )
        == set()
    )


def test_query_supported_counter_names_uses_selected_agent(monkeypatch, rocm_tree):
    rocm_tree.lib_layout.touch()
    patch_library_load(monkeypatch, fake_avail_library({}))
    avail_module = SimpleNamespace(
        loadLibrary=SimpleNamespace(libname=None),
        get_counters=lambda: {
            91: [SimpleNamespace(name="OTHER", block="SQ")],
            42: [SimpleNamespace(name="SQ_WAVES", block="SQ")],
        },
        get_agent_info_map=lambda: {
            91: agent_metadata(1, 256),
            42: agent_metadata(2, 512),
        },
    )
    monkeypatch.setattr(avail_interface, "_import_avail_module", lambda _: avail_module)
    assert avail_interface.query_supported_counter_names(
        rocm_tree.tool_path, "gfx1201", "30032", gpu_node_id=2
    ) == {"SQ_WAVES"}
    assert avail_module.loadLibrary.libname == str(rocm_tree.lib_layout)


def test_older_sdk_without_agent_metadata_keeps_counter_listing(monkeypatch, rocm_tree):
    rocm_tree.lib_layout.touch()
    patch_library_load(monkeypatch, fake_avail_library({}))
    counters = {91: [SimpleNamespace(name="SQ_WAVES", block="SQ")]}
    avail_module = SimpleNamespace(
        loadLibrary=SimpleNamespace(libname=None),
        get_counters=lambda: counters,
    )
    monkeypatch.setattr(avail_interface, "_import_avail_module", lambda _: avail_module)

    assert avail_interface.get_counters(rocm_tree.tool_path) == counters
    assert avail_interface.query_agent_info_map(rocm_tree.tool_path) == {}
    assert (
        avail_interface.query_supported_counter_names(
            rocm_tree.tool_path, "gfx1201", "30032", gpu_node_id=2
        )
        is None
    )


def test_support_query_imports_avail_module_once(monkeypatch, rocm_tree):
    rocm_tree.lib_layout.touch()
    patch_library_load(monkeypatch, fake_avail_library({}))
    imports = []
    avail_module = SimpleNamespace(
        loadLibrary=SimpleNamespace(libname=None),
        get_counters=lambda: {},
        get_agent_info_map=lambda: {},
    )

    def import_avail_module(sdk_path):
        imports.append(sdk_path)
        return avail_module

    monkeypatch.setattr(avail_interface, "_import_avail_module", import_avail_module)

    assert (
        avail_interface.query_supported_counter_names(
            rocm_tree.tool_path, "gfx1201", "30032"
        )
        is None
    )
    assert imports == [rocm_tree.tool_path]


def test_repeated_avail_import_does_not_duplicate_search_path(monkeypatch, rocm_tree):
    avail_module = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "rocprofv3", SimpleNamespace(avail=avail_module))
    monkeypatch.setattr(sys, "path", list(sys.path))
    new_path = str(rocm_tree.lib_layout.parents[1] / "python3/site-packages")

    assert avail_interface._import_avail_module(rocm_tree.tool_path) is avail_module
    assert avail_interface._import_avail_module(rocm_tree.tool_path) is avail_module
    assert sys.path.count(new_path) == 1


def test_legacy_avail_import_preserves_existing_search_paths(monkeypatch, rocm_tree):
    avail_module = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "rocprofv3", None)
    monkeypatch.setitem(
        sys.modules, "rocprofv3_avail_module", SimpleNamespace(avail=avail_module)
    )
    new_path = str(rocm_tree.lib_layout.parents[1] / "python3/site-packages")
    old_path = str(rocm_tree.lib_layout.parents[2] / "bin")
    monkeypatch.setattr(sys, "path", [*sys.path, new_path, old_path])

    assert avail_interface._import_avail_module(rocm_tree.tool_path) is avail_module
    assert avail_interface._import_avail_module(rocm_tree.tool_path) is avail_module
    assert sys.path.count(new_path) == 1
    assert sys.path.count(old_path) == 1


def test_supported_counter_names_use_uuid_when_topology_metadata_unavailable():
    assert avail_interface.supported_counter_names_for_gpu(
        {91: [SimpleNamespace(name="SQ_WAVES", block="SQ")]},
        {91: {**agent_metadata("N/A", "N/A"), "uuid": "gpu-ABCDEF0123456789"}},
        "gfx1201",
        "30032",
        3,
        512,
        "GPU-abcdef0123456789",
    ) == {"SQ_WAVES"}
