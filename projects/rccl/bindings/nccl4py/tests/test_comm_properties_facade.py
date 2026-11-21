# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nccl.core`` exposure of the NCCL 2.31 communicator properties, config
fields and enum values (upstream nccl4py commits 3cc8ed73 and cd31c6d8).

No GPU is needed: ``comm_query_properties`` is replaced by one returning a
``CommProperties`` filled here, and each config facade is checked through the
low-level struct it materializes into. Run with ``run_inplace.sh``.
"""

import numpy as np


def _comm_reporting(monkeypatch, gin_support=(), **fields):
    """A Communicator whose ncclCommQueryProperties reports ``fields``."""
    from nccl.bindings import nccl as lowpp
    from nccl.core import Communicator

    data = np.zeros(1, dtype=lowpp.comm_properties_dtype)
    for name, value in fields.items():
        data[name] = value
    for gin_type in gin_support:
        data["gin_support"][0, gin_type] = 1
    pod = lowpp.CommProperties.from_data(data)
    monkeypatch.setattr(lowpp, "comm_query_properties", lambda comm: pod)
    return Communicator(0x1000)  # never passed to librccl


def test_control_existing_properties_and_fields(monkeypatch):
    import nccl.core as nccl

    comm = _comm_reporting(monkeypatch, cuda_dev=5, gin_type=2, railed_gin_type=3)
    assert comm.cuda_dev == 5
    assert comm.gin_type is nccl.NcclGinType.PROXY
    assert comm.railed_gin_type is nccl.NcclGinType.GDAKI
    cfg = nccl.NCCLConfig(blocking=False, max_p2p_peers=4)._to_lowpp()
    assert (cfg.blocking, cfg.max_p2p_peers) == (0, 4)
    rail = nccl.NcclGinConnectionType.RAIL
    reqs = nccl.NCCLDevCommRequirements(gin_connection_type=rail)._to_lowpp()
    assert reqs.gin_connection_type == 2


def test_properties_reports_new_fields(monkeypatch):
    import nccl.core as nccl

    comm = _comm_reporting(
        monkeypatch,
        gin_support=(2, 3),
        rank=1,
        comm_hash=0x0123456789ABCDEF,
        gin_min_stride=4,
        gin_connection_type=3,
        dev_comm_runtime_version_size=4096,
    )
    props = comm.properties
    assert isinstance(props, nccl.NCCLCommProperties)
    assert (props.rank, props.comm_hash, props.gin_min_stride) == (1, 0x0123456789ABCDEF, 4)
    assert props.gin_connection_type is nccl.NcclGinConnectionType.CUSTOM_STRIDE
    assert props.available_gin_types == {nccl.NcclGinType.PROXY, nccl.NcclGinType.GDAKI}
    assert isinstance(props.available_gin_types, frozenset)
    assert props.dev_comm_runtime_version_size == 4096


def test_new_config_and_requirement_fields_reach_lowpp():
    import nccl.core as nccl

    cfg = nccl.NCCLConfig(launch_order_implicit=True, num_rma_sig=3, rma_eager_init=True)
    cfg = cfg._to_lowpp()
    assert (cfg.launch_order_implicit, cfg.num_rma_sig, cfg.rma_eager_init) == (1, 3, 1)
    stride = nccl.NcclGinConnectionType.CUSTOM_STRIDE
    assert (int(nccl.NcclGinType.EFA_GDA), int(stride)) == (5, 3)
    reqs = nccl.NCCLDevCommRequirements(
        gin_connection_type=stride, gin_custom_stride=8, gin_type=nccl.NcclGinType.PROXY
    )._to_lowpp()
    assert (reqs.gin_connection_type, reqs.gin_custom_stride, reqs.gin_type) == (3, 8, 2)
