# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``NCCLConfig`` fields reach RCCL's communicator configuration checks.

RCCL's ``parseCommConfig`` rejects a negative ``numRmaCtx`` (NCCL 2.29) or
``numRmaSig`` (NCCL 2.31) with ``ncclInvalidArgument``, and resets any field
newer than the ``version`` the bindings stamp into ``ncclConfig_t``. A
one-rank ``Communicator.init`` that accepts a valid value but rejects a
negative one therefore shows that the field travels from ``NCCLConfig``
through the low-level ``Config`` into ``librccl.so``. The rejection comes
before bootstrap, so no peer is needed. One GPU; the module skips when
hip-python is missing or no HIP device is visible. The ``test/nccl4py``
build-smoke harness runs it from ``GPU_SMOKE_TESTS``.
"""

import pytest

pytest.importorskip("hip", reason="hip-python is required for the HIP shim")

import nccl.core as nccl  # noqa: E402  (also registers the cuda.core shim)
from cuda.core import system  # noqa: E402
from hip import hip  # noqa: E402


def _num_devices():
    try:
        return system.get_num_devices()
    except RuntimeError as err:
        # On a host with ROCm but no GPU, hipGetDeviceCount fails with hipErrorNoDevice
        # ("no ROCm-capable device is detected") instead of reporting zero devices.
        if getattr(err, "status", None) == hip.hipError_t.hipErrorNoDevice:
            return 0
        raise


if _num_devices() == 0:  # pragma: no cover - host without GPUs
    pytest.skip("no HIP devices visible", allow_module_level=True)


def _init(**fields):
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
