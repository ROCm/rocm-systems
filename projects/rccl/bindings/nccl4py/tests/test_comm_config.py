# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``NCCLConfig`` fields reach RCCL's communicator configuration checks.

RCCL's ``parseCommConfig`` rejects a negative ``numRmaCtx`` (NCCL 2.29) or
``numRmaSig`` (NCCL 2.31) with ``ncclInvalidArgument``, and resets any field
newer than the ``version`` the bindings stamp into ``ncclConfig_t``. A
one-rank ``Communicator.init`` that accepts a valid value but rejects a
negative one therefore shows that the field travels from ``NCCLConfig``
through the low-level ``Config`` into ``librccl.so``. The rejection comes
before bootstrap, so no peer is needed. Run with::

    NCCL_LIBRARY=/path/to/librccl.so bash run_inplace.sh -q test_comm_config.py
"""

import pytest


def _init(**fields):
    import nccl.core as nccl

    config = nccl.NCCLConfig(**fields)
    return nccl.Communicator.init(nranks=1, rank=0, unique_id=nccl.get_unique_id(), config=config)


def _check_reaches_rccl(field, valid):
    from nccl.bindings import NCCLError, Result

    _init(**{field: valid}).destroy()
    with pytest.raises(NCCLError) as err:
        _init(**{field: -1})
    assert err.value.status == Result.InvalidArgument


def test_control_num_rma_ctx_reaches_rccl():
    _check_reaches_rccl("num_rma_ctx", 1)


def test_num_rma_sig_reaches_rccl():
    _check_reaches_rccl("num_rma_sig", 2)
