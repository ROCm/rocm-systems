# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/specs.py."""

import sys
from contextlib import nullcontext
from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import utils.specs as specs
from utils.specs import (
    MachineSpecs,
    MachineSpecsCDNA,
    MachineSpecsRDNA,
    MachineSpecsRDNA35,
    generate_machine_specs,
    spec_family_for_arch,
)


def rocminfo_gpu_agent(agent_number, arch, chip_id, node_id, bdf_id, uuid_body):
    return (
        f"Agent {agent_number}\n"
        f"  Name: {arch}\n"
        f"  Uuid: GPU-{uuid_body}\n"
        f"  Chip ID: {chip_id}\n"
        f"  BDFID: {bdf_id}\n"
        f"  Internal Node ID: {node_id}\n"
        "  Device Type: GPU\n"
        "  Compute Unit: 64\n"
    )


@pytest.fixture
def mixed_gpu_rocminfo(monkeypatch):
    for variable in (
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(variable, raising=False)
    output = "Agent 1\n  Name: AMD Ryzen\n  Device Type: CPU\n" + (
        rocminfo_gpu_agent(2, "gfx1201", 30032, 1, 256, "1111111111111111")
        + rocminfo_gpu_agent(3, "gfx950", 30112, 3, 512, "2222222222222222")
    )
    monkeypatch.setattr(
        specs.subprocess,
        "run",
        lambda *_args, **_kwargs: CompletedProcess(
            args=["rocminfo"], returncode=0, stdout=output, stderr=""
        ),
    )
    return output


@pytest.mark.misc
@pytest.mark.parametrize(
    "visibility, rocr_order, expected_arch, expected_chip, expected_node",
    [
        ({"HIP_VISIBLE_DEVICES": "1"}, [0, 1], "gfx950", "30112", 3),
        ({"ROCR_VISIBLE_DEVICES": "1"}, [1], "gfx950", "30112", 3),
        (
            {"ROCR_VISIBLE_DEVICES": "1,0", "HIP_VISIBLE_DEVICES": "1"},
            [1, 0],
            "gfx1201",
            "30032",
            1,
        ),
        (
            {"ROCR_VISIBLE_DEVICES": "GPU-2222222222222222"},
            [1],
            "gfx950",
            "30112",
            3,
        ),
        ({"HIP_VISIBLE_DEVICES": "GPU-2222"}, [0, 1], "gfx950", "30112", 3),
        ({"CUDA_VISIBLE_DEVICES": "1"}, [0, 1], "gfx950", "30112", 3),
        (
            {"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "1"},
            [0, 1],
            "gfx1201",
            "30032",
            1,
        ),
        ({}, [0, 1], "gfx1201", "30032", 1),
    ],
)
def test_extract_soc_info_selects_one_workload_visible_gpu(
    monkeypatch,
    mixed_gpu_rocminfo,
    visibility,
    rocr_order,
    expected_arch,
    expected_chip,
    expected_node,
):
    for variable, value in visibility.items():
        monkeypatch.setenv(variable, value)
    agents = specs._parse_rocminfo_gpu_agents(mixed_gpu_rocminfo)
    output = "".join(
        f"Agent {index + 1}\n  Name: {agents[index]['gpu_arch']}\n"
        + "\n".join(agents[index]["rocminfo_lines"])
        + "\n"
        for index in rocr_order
    )
    monkeypatch.setattr(specs, "_run_command", lambda *_args, **_kwargs: output)

    soc_info = specs._extract_soc_info()

    assert soc_info["gpu_arch"] == expected_arch
    assert soc_info["gpu_chip_id"] == expected_chip
    assert soc_info["gpu_node_id"] == expected_node
    assert not any(line.startswith("Agent ") for line in soc_info["rocminfo_lines"])


@pytest.mark.misc
def test_extract_soc_info_chip_id_is_bounded_to_selected_agent(monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    output = rocminfo_gpu_agent(1, "gfx1030", 30032, 1, 256, "1111111111111111")
    output += rocminfo_gpu_agent(2, "gfx950", 30112, 3, 512, "2222222222222222")
    monkeypatch.setattr(specs, "_run_command", lambda *_args, **_kwargs: output)

    soc_info = specs._extract_soc_info()

    assert soc_info["gpu_arch"] == "gfx950"
    assert soc_info["gpu_chip_id"] == "30112"


@pytest.mark.misc
@pytest.mark.parametrize(
    "visibility",
    [
        {"HIP_VISIBLE_DEVICES": "2"},
        {"HIP_VISIBLE_DEVICES": "-1"},
        {"HIP_VISIBLE_DEVICES": "01"},
        {"HIP_VISIBLE_DEVICES": " 1"},
        {"HIP_VISIBLE_DEVICES": "GPU-9999"},
        {"HIP_VISIBLE_DEVICES": ""},
        {"HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "1"},
        {"CUDA_VISIBLE_DEVICES": ""},
        {"ROCR_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": ""},
    ],
)
def test_extract_soc_info_unmatched_selection_exits(
    monkeypatch, mixed_gpu_rocminfo, visibility
):
    for variable, value in visibility.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.setattr(specs, "console_error", sys.exit)

    with pytest.raises(SystemExit, match="does not select a matching GPU"):
        specs._extract_soc_info()


@pytest.mark.misc
def test_extract_soc_info_requested_unsupported_gpu_does_not_select_sibling(
    monkeypatch, mixed_gpu_rocminfo
):
    output = mixed_gpu_rocminfo.replace("Name: gfx1201", "Name: gfx1030")
    monkeypatch.setattr(specs, "_run_command", lambda *_args, **_kwargs: output)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(specs, "console_error", sys.exit)

    with pytest.raises(SystemExit, match="selected GPU architecture gfx1030"):
        specs._extract_soc_info()


@pytest.mark.misc
@pytest.mark.parametrize(
    "variable, mask, expected_indices",
    [
        ("HIP_VISIBLE_DEVICES", "1,1,0", [1, 0]),
        ("HIP_VISIBLE_DEVICES", "1,2,0", [1]),
        ("CUDA_VISIBLE_DEVICES", "1,1,0", [1, 0]),
        ("HIP_VISIBLE_DEVICES", "1,", [1]),
        ("HIP_VISIBLE_DEVICES", "1,,0", [1]),
    ],
)
def test_visibility_masks_use_runtime_termination_order(
    mixed_gpu_rocminfo, variable, mask, expected_indices
):
    gpu_agents = specs._parse_rocminfo_gpu_agents(mixed_gpu_rocminfo)
    visible_agents = specs._apply_gpu_visibility(gpu_agents, mask, variable)
    assert [gpu_agents.index(agent) for agent in visible_agents] == expected_indices


@pytest.mark.misc
def test_extract_soc_info_uuid_selection_uses_first_substring_match(
    monkeypatch, mixed_gpu_rocminfo
):
    output = mixed_gpu_rocminfo.replace("GPU-2222222222222222", "GPU-1111222222222222")
    monkeypatch.setattr(specs, "_run_command", lambda *_args, **_kwargs: output)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "GPU-1111")
    assert specs._extract_soc_info()["gpu_node_id"] == 1


@pytest.mark.misc
@pytest.mark.parametrize("uuid_body", ["ABCDEF0123456789", "XX"])
def test_extract_soc_info_uuid_visibility_uses_exact_runtime_string(
    monkeypatch, mixed_gpu_rocminfo, uuid_body
):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", f"GPU-{uuid_body}")
    rocr_output = mixed_gpu_rocminfo.replace("1111111111111111", uuid_body)
    monkeypatch.setattr(specs, "_run_command", lambda *_args, **_kwargs: rocr_output)

    soc_info = specs._extract_soc_info()

    assert soc_info["gpu_node_id"] == 1
    assert soc_info["gpu_uuid"] == (
        None if uuid_body == "XX" else f"GPU-{uuid_body.lower()}"
    )


@pytest.mark.misc
def test_extract_soc_info_passes_rocr_visibility_to_rocminfo(
    monkeypatch, mixed_gpu_rocminfo
):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("ROCM_PATH", "/selected-rocm")
    with patch.object(specs.subprocess, "run") as run_command:
        run_command.return_value = CompletedProcess(
            args=["rocminfo"], returncode=0, stdout=mixed_gpu_rocminfo, stderr=""
        )
        specs._extract_soc_info()

    discovery_environment = run_command.call_args.kwargs["env"]
    assert discovery_environment["ROCM_PATH"] == "/selected-rocm"
    assert discovery_environment["ROCR_VISIBLE_DEVICES"] == "1,0"
    assert not any(
        variable in discovery_environment
        for variable in (
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        )
    )
    assert specs.os.environ["ROCR_VISIBLE_DEVICES"] == "1,0"


@pytest.mark.misc
@pytest.mark.parametrize("rocr_mask", ["", "1,1,0", "gpu-ABCDEF0123456789", " 1"])
def test_extract_soc_info_leaves_rocr_grammar_to_runtime(
    monkeypatch, mixed_gpu_rocminfo, rocr_mask
):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", rocr_mask)
    rocr_output = rocminfo_gpu_agent(2, "gfx950", 30112, 3, 512, "2222222222222222")
    environments = []

    def run_rocminfo(_command, *, env):
        environments.append(env)
        return rocr_output

    monkeypatch.setattr(specs, "_run_command", run_rocminfo)

    assert specs._extract_soc_info()["gpu_node_id"] == 3
    assert environments[0]["ROCR_VISIBLE_DEVICES"] == rocr_mask


@pytest.mark.misc
def test_extract_soc_info_hip_ordinal_uses_rocr_filtered_list(
    monkeypatch, mixed_gpu_rocminfo
):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    rocr_output = rocminfo_gpu_agent(2, "gfx950", 30112, 3, 512, "2222222222222222")
    monkeypatch.setattr(specs, "_run_command", lambda *_args, **_kwargs: rocr_output)
    monkeypatch.setattr(specs, "console_error", sys.exit)

    with pytest.raises(SystemExit, match="HIP_VISIBLE_DEVICES=.*matching GPU"):
        specs._extract_soc_info()


@pytest.mark.misc
def test_extract_soc_info_reports_rocr_with_no_visible_agents(
    monkeypatch, mixed_gpu_rocminfo
):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-9999")
    monkeypatch.setattr(specs, "_run_command", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(specs, "console_error", sys.exit)

    with pytest.raises(SystemExit, match="ROCr visibility.*matching GPU"):
        specs._extract_soc_info()


@pytest.mark.misc
@pytest.mark.parametrize("hip_mask, expected_warnings", [("1", []), ("0", ["99999"])])
def test_extract_soc_info_warns_unknown_chip_only_for_selected_agent(
    monkeypatch, mixed_gpu_rocminfo, hip_mask, expected_warnings
):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", hip_mask)
    rocr_output = mixed_gpu_rocminfo.replace("Chip ID: 30032", "Chip ID: 99999")
    monkeypatch.setattr(specs, "_run_command", lambda *_args, **_kwargs: rocr_output)
    warnings = []
    monkeypatch.setattr(specs, "console_warning", warnings.append)

    specs._extract_soc_info()

    assert warnings == [
        f"Unknown Chip ID(s) detected: {chip}" for chip in expected_warnings
    ]


@pytest.mark.misc
def test_match_amdsmi_identical_sku_uses_topology_identity():
    selected = {"gpu_arch": "gfx1201", "gpu_chip_id": "30032", "gpu_node_id": 3}
    device_identities = [
        ("first-navi48", {"gpu_node_id": 1}),
        ("second-navi48", {"gpu_node_id": 3}),
    ]

    assert specs._match_amdsmi_device(selected, device_identities) == "second-navi48"


@pytest.mark.misc
@pytest.mark.parametrize(
    "device_identities",
    [
        [("sibling", {"gpu_node_id": 1, "gpu_bdf_id": 512})],
        [("first", {"gpu_node_id": 3}), ("duplicate", {"gpu_node_id": 3})],
        [],
    ],
)
def test_match_amdsmi_unmatched_node_does_not_fall_back(device_identities):
    with pytest.raises(ValueError, match="Cannot match the selected GPU"):
        specs._match_amdsmi_device(
            {"gpu_node_id": 3, "gpu_bdf_id": 512}, device_identities
        )


@pytest.mark.misc
def test_match_amdsmi_bdf_fallback_requires_unique_device():
    identities = [("first", {"gpu_bdf_id": 256}), ("second", {"gpu_bdf_id": 512})]
    assert specs._match_amdsmi_device({"gpu_bdf_id": 512}, identities) == "second"
    assert (
        specs._match_amdsmi_device({"gpu_node_id": 3, "gpu_bdf_id": 512}, identities)
        == "second"
    )
    assert specs._match_amdsmi_device({}, identities[:1]) == "first"
    with pytest.raises(ValueError, match="lacks a unique"):
        specs._match_amdsmi_device({}, identities)


@pytest.mark.misc
@pytest.mark.parametrize("registered_model", ["MI350X", None])
def test_probe_live_specs_queries_the_same_selected_device(
    monkeypatch, mixed_gpu_rocminfo, registered_model
):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(specs, "get_version", lambda _home: {"version": "3.0.0"})
    monkeypatch.setattr(specs, "get_rocm_ver", lambda: "7.2.0")
    monkeypatch.setattr(
        specs,
        "_extract_machine_info",
        lambda: dict.fromkeys([
            "cpu_model",
            "sbios",
            "linux_kernel_version",
            "cpu_memory",
            "linux_distro",
        ]),
    )
    monkeypatch.setattr(
        specs,
        "_load_soc_module",
        lambda _args, machine_specs: setattr(
            machine_specs, "gpu_model", "SMI_FALLBACK"
        ),
    )
    monkeypatch.setattr(
        specs.mi_gpu_specs, "get_gpu_model", lambda *_args: registered_model
    )
    monkeypatch.setattr(MachineSpecsCDNA, "finalize_soc_fields", lambda *_args: None)
    monkeypatch.setattr(specs.amdsmi_interface, "amdsmi_ctx", nullcontext)
    monkeypatch.setattr(
        specs.amdsmi_interface, "get_device_handles", lambda: ["second", "first"]
    )
    monkeypatch.setattr(
        specs.amdsmi_interface,
        "get_device_identity",
        lambda device: {"gpu_node_id": 3 if device == "second" else 1},
    )
    amdsmi = SimpleNamespace(
        amdsmi_get_gpu_vbios_info=lambda device: {
            "part_number": f"vbios-{device}",
            "name": f"model-{device}",
        },
        amdsmi_get_gpu_board_info=lambda device: {"product_name": f"product-{device}"},
        amdsmi_get_gpu_driver_info=lambda device: {
            "driver_version": f"driver-{device}"
        },
        amdsmi_get_gpu_vram_info=lambda device: {
            "vram_size": 8192 if device == "second" else 16
        },
        amdsmi_get_gpu_asic_info=lambda device: {
            "num_compute_units": 304,
            "market_name": f"market-{device}",
        },
        amdsmi_get_gpu_cache_info=lambda _device: {},
        amdsmi_get_gpu_compute_partition=lambda _device: "SPX",
        amdsmi_get_gpu_memory_partition=lambda _device: "NPS1",
        amdsmi_get_clock_info=lambda device, _clock_type: {
            "max_clk": 100 if device == "second" else 1
        },
        AmdSmiClkType=SimpleNamespace(MEM="memory"),
    )
    monkeypatch.setattr(specs.amdsmi_interface, "import_amdsmi_module", lambda: amdsmi)

    machine_specs = specs._probe_live_machine_specs(None)

    assert machine_specs.gpu_arch == "gfx950"
    assert machine_specs.gpu_chip_id == "30112"
    assert machine_specs.gpu_node_id == 3
    assert machine_specs.gpu_bdf_id == 512
    assert machine_specs.gpu_memory == str(8192 * 1024)
    assert machine_specs.vbios == "vbios-second"
    assert machine_specs.amd_gpu_kernel_version == "driver-second"
    assert machine_specs.max_mclk == "100"
    assert machine_specs.gpu_model_names == (
        "product-second",
        "market-second",
        "model-second",
    )
    assert machine_specs.gpu_model == (registered_model or "SMI_FALLBACK")


@pytest.mark.misc
@pytest.mark.parametrize(
    "mock_kwargs",
    [
        {"side_effect": FileNotFoundError("missing")},
        {
            "return_value": CompletedProcess(
                args=["rocminfo"], returncode=1, stdout="", stderr="boom"
            )
        },
    ],
    ids=["missing_binary", "nonzero_exit"],
)
def test_run_fails_fast(mock_kwargs):
    with patch.object(specs.subprocess, "run", **mock_kwargs), pytest.raises(
        SystemExit
    ):
        specs._run_command(["rocminfo"])


@pytest.mark.misc
def test_set_cache_sizes_cdna_uses_l1():
    """CDNA models (e.g. mi300x_a1) should key vL1D as 'L1'."""
    cache_info = {
        "cache": [
            {
                "cache_level": 1,
                "cache_properties": ["DATA_CACHE"],
                "cache_size": 16,
                "num_cache_instance": 304,
            }
        ]
    }
    result = specs.set_cache_sizes(
        "mi300x_a1", 304, cache_info, num_dies=4, num_se=8, num_sa_se=2
    )
    assert "L1" in result
    assert "L0" not in result
    assert result["L1"] == 16 * 1024


@pytest.mark.misc
def test_set_cache_sizes_rdna_uses_l0_and_l1():
    """RDNA models (e.g. rdna35_halo) populate both L0 (per-CU GL0) and L1 (per-SA GL1).
    Both are reported as level-1 DATA_CACHE by amd-smi; L0 has higher instance count.
    """
    cache_info = {
        "cache": [
            {
                "cache_level": 1,
                "cache_properties": ["DATA_CACHE"],
                "cache_size": 32,
                "num_cache_instance": 40,  # GL0: one per CU (40 CUs total)
            },
            {
                "cache_level": 1,
                "cache_properties": ["DATA_CACHE"],
                "cache_size": 128,
                "num_cache_instance": 8,  # GL1: one per SA (num_se * num_sa_se = 4*2)
            },
        ]
    }
    result = specs.set_cache_sizes(
        "rdna35_halo", 40, cache_info, num_dies=1, num_se=4, num_sa_se=2
    )
    assert result["L0"] == 32 * 1024
    assert result["L1"] == 128 * 1024


@pytest.mark.misc
@pytest.mark.parametrize(
    "num_cu, gl1_groups, expected_gl1, expected_warnings",
    [
        (56, [(256, 4), (256, 4)], 256 * 1024, []),
        (64, [(256, 8)], 256 * 1024, []),
        (56, [(256, 3), (256, 4)], None, []),
        (
            56,
            [(128, 4), (128, 4), (256, 4), (256, 4)],
            None,
            ["Ambiguous GL1 cache size reported by AMD-SMI."],
        ),
    ],
    ids=["split_groups", "single_group", "incomplete_groups", "ambiguous_sizes"],
)
def test_set_cache_sizes_navi48(
    monkeypatch, num_cu, gl1_groups, expected_gl1, expected_warnings
):
    """RX 9070 cache groups preserve GL0, GL1, L2, and Infinity Cache capacity."""
    warnings = []
    monkeypatch.setattr(specs, "console_warning", warnings.append)
    cache_info = {
        "cache": [
            {
                "cache_level": level,
                "cache_properties": ["DATA_CACHE", "SIMD_CACHE"],
                "cache_size": size,
                "num_cache_instance": instances,
            }
            for level, size, instances in [
                (1, 32, num_cu),
                (1, 16, num_cu // 2),
                *((1, size, count) for size, count in gl1_groups),
                (2, 8192, 1),
                (3, 65536, 1),
            ]
        ]
    }
    cache_sizes = specs.set_cache_sizes(
        "rdna4_navi48", num_cu, cache_info, num_dies=1, num_se=4, num_sa_se=2
    )
    expected = {
        "L0": 32 * 1024,
        "L2": 8 * 1024 * 1024,
        "MALL": 64 * 1024 * 1024,
    }
    if expected_gl1 is not None:
        expected["L1"] = expected_gl1
    assert cache_sizes == expected
    assert warnings == expected_warnings


@pytest.mark.misc
def test_set_cache_sizes_l2_and_mall():
    """L2 and MALL (L3) cache entries are correctly parsed and scaled by num_dies."""
    cache_info = {
        "cache": [
            {
                "cache_level": 2,
                "cache_size": 512,
                "cache_properties": [],
                "num_cache_instance": 1,
            },
            {
                "cache_level": 3,
                "cache_size": 256,
                "cache_properties": [],
                "num_cache_instance": 1,
            },
        ]
    }
    result = specs.set_cache_sizes(
        "mi300x_a1", 304, cache_info, num_dies=4, num_se=8, num_sa_se=2
    )
    assert result["L2"] == 512 * 1024
    assert result["MALL"] == 256 * 1024 // 4


@pytest.mark.misc
def test_set_cache_sizes_selects_vl1d_by_max_instance_count():
    """Harvested GPU: num_cu reported by rocminfo may be less than the max
    num_cache_instance in amd-smi. set_cache_sizes must select vL1D by the
    highest num_cache_instance, not by exact match to num_cu.
    """
    cache_info = {
        "cache": [
            {
                "cache_level": 1,
                "cache_properties": ["DATA_CACHE"],
                "cache_size": 16,
                "num_cache_instance": 228,  # max instances (some CUs harvested)
            },
            {
                "cache_level": 1,
                "cache_properties": ["DATA_CACHE"],
                "cache_size": 16,
                "num_cache_instance": 190,  # lower instance count entry
            },
        ]
    }
    # num_cu=224 simulates harvested GPU (fewer than 228 active CUs)
    result = specs.set_cache_sizes(
        "mi300x_a1", 224, cache_info, num_dies=4, num_se=8, num_sa_se=2
    )
    assert "L1" in result
    assert result["L1"] == 16 * 1024


@pytest.mark.misc
def test_kw_only_rejects_positional_arguments():
    """MachineSpecs construction must be keyword-only (regression for the
    _kw_only decorator that previously accepted positional args silently).
    """
    with pytest.raises(TypeError):
        MachineSpecs("gfx942")

    spec = MachineSpecs(gpu_arch="gfx942")
    assert spec.gpu_arch == "gfx942"


@pytest.mark.misc
@pytest.mark.parametrize(
    "series, expected_cls",
    [
        ("RDNA3.5", MachineSpecsRDNA),
        ("RDNA4", MachineSpecsRDNA),
        ("mi300", MachineSpecsCDNA),
        (None, MachineSpecsCDNA),
    ],
    ids=["rdna35", "rdna4", "cdna", "unknown"],
)
def test_spec_family_for_arch_dispatch(monkeypatch, series, expected_cls):
    monkeypatch.setattr(specs.mi_gpu_specs, "get_gpu_series", lambda _arch: series)
    assert spec_family_for_arch("gfx_test") is expected_cls


@pytest.mark.misc
@pytest.mark.parametrize(
    "se_per_gpu, sa_per_se, vram_bit_width, expected_gl1c, expected_channels",
    [
        (4, 2, 256, "8", "8"),
        (4, 2, None, "8", "32"),
        (None, 2, 256, None, "8"),
    ],
    ids=["bit_width", "fallback_total_l2_chan", "missing_se"],
)
def test_rdna35_finalize_soc_fields(
    monkeypatch, se_per_gpu, sa_per_se, vram_bit_width, expected_gl1c, expected_channels
):
    """RDNA 3.5 derives num_gl1c from SE*SA and num_memory_channels from the
    VRAM bus width, falling back to total_l2_chan when width is unavailable.
    """
    spec = MachineSpecsRDNA35(
        gpu_arch="gfx1151",
        gpu_model="rdna35_halo",
        se_per_gpu=se_per_gpu,
        sa_per_se=sa_per_se,
        l2_banks="8",
        total_l2_chan="32",
    )
    gpu_info = {
        "num_compute_units": 0,
        "gpu_cache_info": {},
        "vram_bit_width": vram_bit_width,
        "perf_level": "AUTO",
    }
    monkeypatch.setattr(specs, "set_cache_sizes", lambda *_args: {})
    monkeypatch.setattr(specs.mi_gpu_specs, "get_num_dies", lambda *_args: 1)
    monkeypatch.setattr(specs, "totall2_banks", lambda *_args: "32")
    spec.finalize_soc_fields(gpu_info)

    assert spec.num_gl1c == expected_gl1c
    assert spec.num_memory_channels == expected_channels
    assert spec.perf_level == "AUTO"


@pytest.mark.misc
@pytest.mark.parametrize(
    "memory_partition, expected_channels",
    [
        ("NPS1", "128"),
        ("nps2", "64"),
        ("NPS4", "32"),
    ],
    ids=["nps1", "nps2", "nps4"],
)
def test_cdna_hbm_channels_nps_divisors(memory_partition, expected_channels):
    """NPS memory partitions divide HBM channel count by the NPS denominator.

    Uses whole-chip (SPX) XCD count, independent of the active compute partition.
    """
    spec = MachineSpecsCDNA(
        gpu_arch="gfx942",
        gpu_model="mi300x_a1",
        l2_banks="16",
        total_l2_chan="64",
        memory_partition=memory_partition,
        compute_partition="CPX",
    )
    with patch.object(
        specs.mi_gpu_specs, "get_num_xcds", return_value=8
    ) as get_num_xcds_mock:
        assert spec._get_hbm_channels() == expected_channels
        get_num_xcds_mock.assert_called_once_with("gfx942", "mi300x_a1", "SPX")


@pytest.mark.misc
@pytest.mark.parametrize("gpu_arch", ["gfx1151", "gfx1201"])
def test_reconstruct_specs_from_sysinfo_round_trip(monkeypatch, gpu_arch):
    """generate_machine_specs reconstructs the arch-specific subclass from a
    saved sysinfo dict.
    """
    sysinfo = {
        "version": "3",
        "gpu_arch": gpu_arch,
        "num_gl1c": "8",
        "num_memory_channels": "8",
        "perf_level": "STABLE_STD",
    }
    monkeypatch.setattr(specs, "get_version", lambda _home: {"version": "3.0.0"})
    spec = generate_machine_specs(None, sysinfo)

    assert isinstance(spec, MachineSpecsRDNA35)
    assert spec.num_gl1c == "8"
    assert spec.num_memory_channels == "8"
    assert spec.perf_level == "STABLE_STD"
    assert spec.gpu_node_id is None


@pytest.mark.misc
@pytest.mark.parametrize("node_id", [3, "3", 3.0])
def test_reconstruct_specs_preserves_saved_physical_identity(monkeypatch, node_id):
    monkeypatch.setattr(specs, "get_version", lambda _home: {"version": "3.0.0"})
    sysinfo = {
        "version": "3",
        "gpu_arch": "gfx1201",
        "gpu_node_id": node_id,
        "gpu_bdf_id": "512",
        "gpu_device_index": "1",
        "gpu_uuid": "GPU-2222222222222222",
    }

    machine_specs = generate_machine_specs(None, sysinfo)
    saved_specs = machine_specs.get_class_members()
    reconstructed = generate_machine_specs(None, saved_specs)

    assert reconstructed.gpu_node_id == 3
    assert reconstructed.gpu_bdf_id == 512
    assert reconstructed.gpu_uuid == "GPU-2222222222222222"
    assert sysinfo["gpu_bdf_id"] == "512"


@pytest.mark.misc
@pytest.mark.parametrize("missing_value", [None, "", float("nan")])
def test_reconstruct_specs_empty_physical_identity_defaults(monkeypatch, missing_value):
    monkeypatch.setattr(specs, "get_version", lambda _home: {"version": "3.0.0"})
    machine_specs = generate_machine_specs(
        None,
        {
            "version": "3",
            "gpu_arch": "gfx1201",
            "gpu_node_id": missing_value,
            "gpu_bdf_id": missing_value,
            "gpu_device_index": missing_value,
            "gpu_uuid": missing_value,
        },
    )

    assert machine_specs.gpu_node_id is None
    assert machine_specs.gpu_bdf_id is None
    assert machine_specs.gpu_uuid is None


@pytest.mark.misc
def test_reconstruct_specs_from_sysinfo_missing_version_raises():
    """A sysinfo dict without a version key aborts via console_error/KeyError."""
    with pytest.raises((KeyError, SystemExit)):
        generate_machine_specs(None, {"gpu_arch": "gfx1151"})


@pytest.mark.misc
def test_reconstruct_specs_from_sysinfo_unsupported_arch_exits(monkeypatch):
    """A workload from an arch without a spec entry stops with an error naming
    the arch, not a traceback."""
    monkeypatch.setattr(specs, "console_error", sys.exit)
    monkeypatch.setattr(specs, "get_version", lambda _home: {"version": "3.0.0"})
    with pytest.raises(SystemExit, match="profiled on gfx1030, which is not supported"):
        generate_machine_specs(None, {"version": "3", "gpu_arch": "gfx1030"})


@pytest.mark.misc
@pytest.mark.parametrize(
    "arch_name, expected_arch",
    [
        ("gfx1250-strict", "gfx1250"),
        ("gfx1201", "gfx1201"),
        ("GFX1201", "gfx1201"),
        ("GfX1201", "gfx1201"),
    ],
)
def test_extract_soc_info_recognizes_registered_architecture(
    monkeypatch, mixed_gpu_rocminfo, arch_name, expected_arch
):
    rocminfo_output = (
        "Agent 1\n  Name: AMD Ryzen\n  Device Type: CPU\n"
        f"Agent 2\n  Name: {arch_name}\n"
    )
    monkeypatch.setattr(
        specs,
        "_run_command",
        lambda *_args, **_kwargs: rocminfo_output,
    )

    soc_info = specs._extract_soc_info()

    assert soc_info["gpu_arch"] == expected_arch


@pytest.mark.misc
def test_extract_soc_info_name_only_output_keeps_selected_identity(
    monkeypatch, mixed_gpu_rocminfo
):
    name_only_output = "\n".join(
        line
        for line in mixed_gpu_rocminfo.splitlines()
        if not line.startswith("Agent ")
    )
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(
        specs, "_run_command", lambda *_args, **_kwargs: name_only_output
    )

    soc_info = specs._extract_soc_info()

    assert soc_info["gpu_arch"] == "gfx950"
    assert soc_info["gpu_chip_id"] == "30112"
    assert soc_info["gpu_node_id"] == 3
    assert soc_info["gpu_bdf_id"] == 512
    assert soc_info["gpu_uuid"] == "GPU-2222222222222222"


@pytest.mark.misc
def test_get_rocm_ver_env_override(monkeypatch, tmp_path):
    """ROCM_VER overrides detection when no .info/ directory is present."""
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    monkeypatch.setenv("ROCM_VER", "6.9.9")
    assert specs.get_rocm_ver() == "6.9.9"


@pytest.mark.misc
def test_get_rocm_ver_undetectable_errors(monkeypatch, tmp_path):
    """Missing .info/ and unset ROCM_VER terminates via console_error."""
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    monkeypatch.delenv("ROCM_VER", raising=False)
    with patch.object(specs, "console_error") as console_error_mock:
        specs.get_rocm_ver()
    console_error_mock.assert_called_once()
    assert (
        "Unable to detect a complete local ROCm installation"
        in (console_error_mock.call_args.args[0])
    )
