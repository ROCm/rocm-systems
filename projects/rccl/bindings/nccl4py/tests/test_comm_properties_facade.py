# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nccl.core`` exposure of the NCCL 2.31 communicator properties, config
fields and enum values (upstream nccl4py commits 3cc8ed73 and cd31c6d8).

No GPU or librccl is needed: the library calls behind
``Communicator.properties`` are replaced by stand-ins reporting a
``CommProperties`` filled here (or an older NCCL's struct), and each config
facade is checked through the low-level struct it materializes into. The
``test/nccl4py`` build-smoke harness runs it from ``CPU_SMOKE_TESTS``.
"""

from types import SimpleNamespace

import numpy as np


def _comm_returning(monkeypatch, pod):
    """A Communicator whose ncclCommQueryProperties returns ``pod``."""
    from nccl.bindings import nccl as lowpp
    from nccl.core import Communicator

    monkeypatch.setattr(lowpp, "comm_query_properties", lambda comm: pod)
    # Communicator.properties first checks that the loaded library is no older
    # than the bindings (require_compatible_nccl).
    monkeypatch.setattr(lowpp, "get_version", lambda: lowpp.__version_code__)
    # A failing assertion prints the communicator, and the real repr asks
    # librccl for the rank behind this fake handle.
    monkeypatch.setattr(Communicator, "__repr__", lambda self: "<stand-in Communicator>")
    return Communicator(0x1000)  # never passed to librccl


def _comm_reporting(monkeypatch, gin_support=(), **fields):
    """A Communicator whose ncclCommQueryProperties reports ``fields``."""
    from nccl.bindings import nccl as lowpp

    data = np.zeros(1, dtype=lowpp.comm_properties_dtype)
    for name, value in fields.items():
        data[name] = value
    for gin_type in gin_support:
        data["gin_support"][0, gin_type] = 1
    return _comm_returning(monkeypatch, lowpp.CommProperties.from_data(data))


def test_control_existing_properties_and_fields(monkeypatch):
    import nccl.core as nccl
    from nccl.bindings import nccl as lowpp

    comm = _comm_reporting(
        monkeypatch, cuda_dev=5, gin_type=lowpp.GinType.PROXY, railed_gin_type=lowpp.GinType.GDAKI
    )
    assert comm.cuda_dev == 5
    assert comm.gin_type is nccl.NcclGinType.PROXY
    assert comm.railed_gin_type is nccl.NcclGinType.GDAKI
    cfg = nccl.NCCLConfig(blocking=False, max_p2p_peers=4)._to_lowpp()
    assert (cfg.blocking, cfg.max_p2p_peers) == (0, 4)
    rail = nccl.NcclGinConnectionType.RAIL
    reqs = nccl.NCCLDevCommRequirements(gin_connection_type=rail)._to_lowpp()
    assert reqs.gin_connection_type == lowpp.GinConnectionType.RAIL


def test_properties_reports_new_fields(monkeypatch):
    import nccl.core as nccl
    from nccl.bindings import nccl as lowpp

    comm = _comm_reporting(
        monkeypatch,
        # NONE and the last of the 64 slots are not transports.
        gin_support=(lowpp.GinType.NONE, lowpp.GinType.PROXY, lowpp.GinType.GDAKI, 63),
        rank=1,
        comm_hash=0xFEDCBA9876543210,  # bit 63 set: commHash is a uint64_t
        gin_min_stride=4,
        gin_connection_type=lowpp.GinConnectionType.CUSTOM_STRIDE,
        dev_comm_runtime_version_size=4096,
    )
    props = comm.properties
    assert isinstance(props, nccl.NCCLCommProperties)
    assert (props.rank, props.comm_hash, props.gin_min_stride) == (1, 0xFEDCBA9876543210, 4)
    assert props.gin_connection_type is nccl.NcclGinConnectionType.CUSTOM_STRIDE
    assert props.available_gin_types == {nccl.NcclGinType.PROXY, nccl.NcclGinType.GDAKI}
    assert isinstance(props.available_gin_types, frozenset)
    assert props.dev_comm_runtime_version_size == 4096


def test_properties_new_fields_none_before_nccl_2_31(monkeypatch):
    from nccl.bindings import nccl as lowpp

    # The members ncclCommProperties had before NCCL 2.31.
    older = SimpleNamespace(
        rank=1,
        n_ranks=2,
        cuda_dev=0,
        nvml_dev=0,
        device_api_support=1,
        multimem_support=0,
        gin_type=lowpp.GinType.NONE,
        n_lsa_teams=1,
        host_rma_support=0,
        railed_gin_type=lowpp.GinType.NONE,
    )
    props = _comm_returning(monkeypatch, older).properties
    assert (props.comm_hash, props.gin_min_stride, props.gin_connection_type) == (None,) * 3
    assert (props.available_gin_types, props.dev_comm_runtime_version_size) == (None,) * 2


def test_new_config_and_requirement_fields_reach_lowpp():
    import nccl.core as nccl
    from nccl.bindings import nccl as lowpp

    def new_config_fields(**fields):
        cfg = nccl.NCCLConfig(**fields)._to_lowpp()
        return cfg.launch_order_implicit, cfg.num_rma_sig, cfg.rma_eager_init

    on = new_config_fields(launch_order_implicit=True, num_rma_sig=3, rma_eager_init=True)
    off = new_config_fields(launch_order_implicit=False, num_rma_sig=0, rma_eager_init=False)
    assert on == (1, 3, 1)
    # Only None is skipped: False and 0 reach the struct, and unset fields keep
    # NCCL_CONFIG_UNDEF_INT.
    assert off == (0, 0, 0)
    assert new_config_fields() == (-2147483648,) * 3
    stride = nccl.NcclGinConnectionType.CUSTOM_STRIDE
    assert nccl.NcclGinType.EFA_GDA == lowpp.GinType.EFA_GDA
    assert stride == lowpp.GinConnectionType.CUSTOM_STRIDE
    reqs = nccl.NCCLDevCommRequirements(
        gin_connection_type=stride, gin_custom_stride=8, gin_type=nccl.NcclGinType.PROXY
    )._to_lowpp()
    expected = (lowpp.GinConnectionType.CUSTOM_STRIDE, 8, lowpp.GinType.PROXY)
    assert (reqs.gin_connection_type, reqs.gin_custom_stride, reqs.gin_type) == expected
