# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/amdsmi_interface.py."""

import functools
from types import SimpleNamespace
from unittest import mock

import pytest

import utils.amdsmi_interface as amdsmi_interface
from utils.amdsmi_interface import _per_device_query

# =============================================================================
# TESTS FOR AMDSMI INTERFACE
# =============================================================================


def gpu_vram_info_for_device(device):
    if device == "selected":
        raise RuntimeError("selected device query failed")
    return {"vram_size": 999}


def raise_smi_error():
    raise RuntimeError("AMD-SMI unavailable")


def test_explicit_device_query_does_not_retry_sibling(monkeypatch):
    amdsmi = SimpleNamespace(amdsmi_get_gpu_vram_info=gpu_vram_info_for_device)
    monkeypatch.setattr(amdsmi_interface, "import_amdsmi_module", lambda: amdsmi)
    monkeypatch.setattr(amdsmi_interface, "get_device_handles", lambda: ["sibling"])

    assert amdsmi_interface.get_gpu_vram_size(device="selected") == "0"


def test_device_identity_retains_kfd_node_and_rocminfo_bdf(monkeypatch):
    amdsmi = SimpleNamespace(
        AmdSmiException=RuntimeError,
        amdsmi_get_gpu_kfd_info=lambda _device: {"node_id": 3},
        amdsmi_get_gpu_bdf_id=lambda _device: (2 << 32) | 512,
        amdsmi_get_gpu_enumeration_info=lambda _device: {
            "hip_uuid": "GPU-2222222222222222"
        },
    )
    monkeypatch.setattr(amdsmi_interface, "import_amdsmi_module", lambda: amdsmi)

    assert amdsmi_interface.get_device_identity("selected") == {
        "gpu_node_id": 3,
        "gpu_bdf_id": 512,
        "gpu_uuid": "GPU-2222222222222222",
    }


@pytest.mark.parametrize("uuid_value", ["", "GPU-XX"])
def test_device_identity_missing_apis_and_unavailable_uuid(monkeypatch, uuid_value):
    amdsmi = SimpleNamespace(
        AmdSmiException=RuntimeError,
        amdsmi_get_gpu_enumeration_info=lambda _device: {"hip_uuid": uuid_value},
    )
    monkeypatch.setattr(amdsmi_interface, "import_amdsmi_module", lambda: amdsmi)

    assert amdsmi_interface.get_device_identity("selected") == {}


@pytest.mark.parametrize(
    "node_metadata",
    [{}, {"node_id": "N/A"}, {"node_id": None}, {"node_id": 0xFFFFFFFF}],
)
def test_device_identity_unavailable_node_retains_bdf(monkeypatch, node_metadata):
    amdsmi = SimpleNamespace(
        AmdSmiException=RuntimeError,
        amdsmi_get_gpu_kfd_info=lambda _device: node_metadata,
        amdsmi_get_gpu_bdf_id=lambda _device: 512,
    )
    monkeypatch.setattr(amdsmi_interface, "import_amdsmi_module", lambda: amdsmi)

    assert amdsmi_interface.get_device_identity("selected") == {"gpu_bdf_id": 512}


@pytest.mark.parametrize("missing_bdf", [None, "N/A", 0xFFFFFFFFFFFFFFFF])
def test_device_identity_unavailable_bdf_retains_uuid(monkeypatch, missing_bdf):
    amdsmi = SimpleNamespace(
        AmdSmiException=RuntimeError,
        amdsmi_get_gpu_bdf_id=lambda _device: missing_bdf,
        amdsmi_get_gpu_enumeration_info=lambda _device: {
            "hip_uuid": "gpu-ABCDEF0123456789"
        },
    )
    monkeypatch.setattr(amdsmi_interface, "import_amdsmi_module", lambda: amdsmi)

    assert amdsmi_interface.get_device_identity("selected") == {
        "gpu_uuid": "GPU-abcdef0123456789"
    }


def test_amdsmi_context_preserves_probe_error_and_shuts_down(monkeypatch):
    events = []
    amdsmi = SimpleNamespace(
        AmdSmiException=RuntimeError,
        amdsmi_init=lambda: events.append("init"),
        amdsmi_shut_down=lambda: events.append("shutdown"),
    )
    monkeypatch.setattr(amdsmi_interface, "import_amdsmi_module", lambda: amdsmi)

    with pytest.raises(ValueError, match="identity conflict"):
        with amdsmi_interface.amdsmi_ctx():
            raise ValueError("identity conflict")

    assert events == ["init", "shutdown"]


def test_amdsmi_context_preserves_initialization_failure(monkeypatch):
    events = []
    amdsmi = SimpleNamespace(
        AmdSmiException=RuntimeError,
        amdsmi_init=raise_smi_error,
        amdsmi_shut_down=lambda: events.append("shutdown"),
    )
    monkeypatch.setattr(amdsmi_interface, "import_amdsmi_module", lambda: amdsmi)

    with pytest.raises(RuntimeError, match="AMD-SMI unavailable"):
        with amdsmi_interface.amdsmi_ctx():
            events.append("body")

    assert events == []


def test_amdsmi_context_shutdown_failure_preserves_probe_error(monkeypatch):
    amdsmi = SimpleNamespace(
        AmdSmiException=RuntimeError,
        amdsmi_init=lambda: None,
        amdsmi_shut_down=raise_smi_error,
    )
    monkeypatch.setattr(amdsmi_interface, "import_amdsmi_module", lambda: amdsmi)

    with pytest.raises(ValueError, match="identity conflict"):
        with amdsmi_interface.amdsmi_ctx():
            raise ValueError("identity conflict")


def test_amdsmi_ctx():
    from utils.amdsmi_interface import amdsmi_ctx, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("amdsmi.amdsmi_init") as amdsmi_init_mock:
        with mock.patch("amdsmi.amdsmi_shut_down") as amdsmi_shutdown_mock:
            with amdsmi_ctx():
                amdsmi_init_mock.assert_called_once()
            amdsmi_shutdown_mock.assert_called_once()


def test_amdsmi_get_device_handles():
    from utils.amdsmi_interface import get_device_handles, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("amdsmi.amdsmi_get_processor_handles") as device_handles_mock:
        device_handles_mock.return_value = [12345]
        handles = get_device_handles()
        assert handles[0] == 12345
        device_handles_mock.assert_called_once()

    with mock.patch(
        "amdsmi.amdsmi_get_processor_handles", side_effect=Exception("Mock exception")
    ) as device_handles_mock:
        handle = get_device_handles()
        assert len(handle) == 0


def test_amdsmi_get_mem_max_clock():
    from utils.amdsmi_interface import get_mem_max_clock, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("utils.amdsmi_interface.get_device_handles") as device_handles_mock:
        device_handles_mock.return_value = [0, 4567]
        with mock.patch("amdsmi.amdsmi_get_clock_info") as mem_max_clock_mock:

            def side_effect(handle, *args, **kwargs):
                if handle == 0:
                    raise Exception("Invalid handle: 0")
                return {"max_clk": 100}

            mem_max_clock_mock.side_effect = side_effect
            clk = get_mem_max_clock()
            assert mem_max_clock_mock.call_count == 2
            assert clk == 100


def test_amdsmi_get_gpu_model():
    from utils.amdsmi_interface import get_gpu_model, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("utils.amdsmi_interface.get_device_handles") as device_handles_mock:
        device_handles_mock.return_value = [12345]
        with mock.patch("amdsmi.amdsmi_get_gpu_board_info") as device_name_mock:
            with mock.patch("amdsmi.amdsmi_get_gpu_asic_info") as asic_name_mock:
                with mock.patch("amdsmi.amdsmi_get_gpu_vbios_info") as vbios_name_mock:
                    device_name_mock.return_value = {"product_name": "AMD MIXXX"}
                    asic_name_mock.return_value = {"market_name": "MIXXX"}
                    vbios_name_mock.return_value = {"name": "mixxx"}
                    model = get_gpu_model()
                    device_name_mock.assert_called_once()
                    assert model == ("AMD MIXXX", "MIXXX", "mixxx")

        with mock.patch(
            "amdsmi.amdsmi_get_gpu_board_info", side_effect=Exception("Mock exception")
        ):
            model = get_gpu_model()
            assert model == ("N/A", "N/A", "N/A")


def test_amdsmi_get_gpu_vbios_part_number():
    from utils.amdsmi_interface import get_gpu_vbios_part_number, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("utils.amdsmi_interface.get_device_handles") as device_handles_mock:
        device_handles_mock.return_value = [12345]
        with mock.patch("amdsmi.amdsmi_get_gpu_vbios_info") as vbios_part_number_mock:
            vbios_part_number_mock.return_value = {
                "part_number": "12345-67890",
            }
            part_number = get_gpu_vbios_part_number()
            vbios_part_number_mock.assert_called_once()
            assert part_number == "12345-67890"

        with mock.patch(
            "amdsmi.amdsmi_get_gpu_vbios_info", side_effect=Exception("Mock exception")
        ):
            part_number = get_gpu_vbios_part_number()
            assert part_number == "N/A"


def test_amdsmi_get_gpu_compute_partition():
    from utils.amdsmi_interface import get_gpu_compute_partition, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("utils.amdsmi_interface.get_device_handles") as device_handles_mock:
        device_handles_mock.return_value = [12345]
        with mock.patch(
            "amdsmi.amdsmi_get_gpu_compute_partition"
        ) as compute_partition_mock:
            compute_partition_mock.return_value = "Mock Partition"
            partition = get_gpu_compute_partition()
            compute_partition_mock.assert_called_once()
            assert partition == "Mock Partition"

        with mock.patch(
            "amdsmi.amdsmi_get_gpu_compute_partition",
            side_effect=Exception("Mock exception"),
        ):
            partition = get_gpu_compute_partition()
            assert partition == "N/A"


def test_amdsmi_get_gpu_memory_partition():
    from utils.amdsmi_interface import get_gpu_memory_partition, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("utils.amdsmi_interface.get_device_handles") as device_handles_mock:
        device_handles_mock.return_value = [12345]
        with mock.patch(
            "amdsmi.amdsmi_get_gpu_memory_partition"
        ) as memory_partition_mock:
            memory_partition_mock.return_value = "Mock Memory Partition"
            partition = get_gpu_memory_partition()
            memory_partition_mock.assert_called_once()
            assert partition == "Mock Memory Partition"

        with mock.patch(
            "amdsmi.amdsmi_get_gpu_memory_partition",
            side_effect=Exception("Mock exception"),
        ):
            partition = get_gpu_memory_partition()
            assert partition == "N/A"


def test_amdsmi_get_gpu_cache_size():
    from utils.amdsmi_interface import get_gpu_cache_info, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("utils.amdsmi_interface.get_device_handles") as device_handles_mock:
        device_handles_mock.return_value = [12345]
        with mock.patch("amdsmi.amdsmi_get_gpu_cache_info") as cache_info_mock:
            cache_info_mock.return_value = {"cache": "Mock Cache Info"}
            cache_info = get_gpu_cache_info()
            cache_info_mock.assert_called_once()
            assert cache_info == {"cache": "Mock Cache Info"}

        with mock.patch(
            "amdsmi.amdsmi_get_gpu_cache_info",
            side_effect=Exception("Mock exception"),
        ):
            cache_info = get_gpu_cache_info()
            assert cache_info is None


def test_amdsmi_get_gpu_num_compute_units():
    from utils.amdsmi_interface import get_gpu_num_compute_units, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("utils.amdsmi_interface.get_device_handles") as device_handles_mock:
        device_handles_mock.return_value = [12345]
        with mock.patch("amdsmi.amdsmi_get_gpu_asic_info") as cu_mock:
            cu_mock.return_value = {"num_compute_units": 10}
            cu_count = get_gpu_num_compute_units()
            cu_mock.assert_called_once()
            assert cu_count == 10

        with mock.patch(
            "amdsmi.amdsmi_get_gpu_asic_info",
            side_effect=Exception("Mock exception"),
        ):
            cu_count = get_gpu_num_compute_units()
            assert cu_count == 0


def test_amdsmi_get_gpu_perf_level():
    from utils.amdsmi_interface import get_gpu_perf_level, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("utils.amdsmi_interface.get_device_handles") as device_handles_mock:
        device_handles_mock.return_value = [12345]
        with mock.patch("amdsmi.amdsmi_get_gpu_perf_level") as perf_level_mock:
            perf_level_mock.return_value = "AUTO"
            perf_level = get_gpu_perf_level()
            perf_level_mock.assert_called_once()
            assert perf_level == "AUTO"

        with mock.patch(
            "amdsmi.amdsmi_get_gpu_perf_level",
            side_effect=Exception("Mock exception"),
        ):
            perf_level = get_gpu_perf_level()
            assert perf_level is None


def test_per_device_query_returns_default_and_logs_last_error_on_all_failure():
    """When every device raises, return the default and warn with the last error."""

    @functools.partial(
        _per_device_query, default_return="DEFAULT", warning_label="test label"
    )
    def fn(device, amdsmi):
        raise RuntimeError(f"boom-{device}")

    with mock.patch("utils.amdsmi_interface.get_device_handles") as handles_mock:
        handles_mock.return_value = ["d1", "d2", "d3"]
        with mock.patch("utils.amdsmi_interface.import_amdsmi_module"):
            with mock.patch("utils.amdsmi_interface.console_warning") as warn_mock:
                result = fn()
                assert result == "DEFAULT"
                warn_mock.assert_called_once()
                warning_message = warn_mock.call_args[0][0]
                assert "test label" in warning_message
                assert "boom-d3" in warning_message
