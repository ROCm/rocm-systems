# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-collective configuration: ``config=NCCLCollConfig(...)`` on every collective.

Each collective runs on a one-rank communicator, so one GPU is enough.
RCCL resolves a call's ``algSelection`` before its single-rank shortcut: a
selection that names no algorithm is ``ncclInvalidArgument`` while
``forceAlgSelection`` is set (the default) and falls back to automatic
selection once it is cleared. Seeing both outcomes shows that the
configuration reached the collective's ``nccl*Config`` entry point.

hip-python is optional: without it, the few device queries the cuda.core
shim makes go through ctypes to the HIP runtime librccl links. Run with::

    NCCL_LIBRARY=/path/to/librccl.so bash run_inplace.sh -q -rs test_coll_config.py
"""

import ctypes
import os

import pytest

import nccl.core as nccl
from nccl._hip_compat.cuda_core_shim._hip import hip as shim_hip
from nccl.bindings import NCCLError, Result

# Method name and the positional arguments after (sendbuf, recvbuf).
_COLLECTIVES = [
    ("allreduce", (nccl.SUM,)),
    ("broadcast", (0,)),
    ("reduce", (nccl.SUM, 0)),
    ("allgather", ()),
    ("reduce_scatter", (nccl.SUM,)),
    ("alltoall", ()),
    ("gather", (0,)),
    ("scatter", (0,)),
]
_NBYTES = 64


def _hip_int_out(fn):
    """hip-python's ``(status, value)`` convention for ``hipError_t f(int*)``."""

    def call():
        value = ctypes.c_int()
        return fn(ctypes.byref(value)), value.value

    return call


@pytest.fixture(scope="module")
def one_rank_comm():
    with pytest.MonkeyPatch.context() as mp:
        if not hasattr(shim_hip, "hipGetDeviceCount"):  # run_inplace.sh's empty hip stub
            # dlsym on librccl's handle also searches the HIP runtime it links.
            hip_rt = ctypes.CDLL(os.environ.get("NCCL_LIBRARY", "librccl.so"))
            for name in ("hipGetDeviceCount", "hipGetDevice"):
                mp.setattr(shim_hip, name, _hip_int_out(getattr(hip_rt, name)), raising=False)
            mp.setattr(shim_hip, "hipSetDevice", hip_rt.hipSetDevice, raising=False)
        status, ndev = shim_hip.hipGetDeviceCount()
        if int(status) != 0 or ndev == 0:
            pytest.skip("no GPU visible")
        comm = nccl.Communicator.init(nranks=1, rank=0, unique_id=nccl.get_unique_id())
        yield comm
        comm.destroy()


@pytest.fixture(scope="module")
def buffers(one_rank_comm):
    return nccl.mem_alloc(_NBYTES), nccl.mem_alloc(_NBYTES)


@pytest.mark.parametrize("method, args", _COLLECTIVES)
def test_control_collective_without_config(one_rank_comm, buffers, method, args):
    getattr(one_rank_comm, method)(*buffers, *args)


@pytest.mark.parametrize("method, args", _COLLECTIVES)
def test_collective_config_reaches_rccl(one_rank_comm, buffers, method, args):
    coll_config = getattr(nccl, "NCCLCollConfig", None)
    assert coll_config is not None, "nccl.core has no NCCLCollConfig"
    call = getattr(one_rank_comm, method)
    send, recv = buffers

    # RCCL ignores vendor options; they ride along to show the chain is accepted.
    vendor = (nccl.VendorOption(vendor_id=0x5CC1, option_id=1, int_value=1),)
    fallback = coll_config(
        alg_selection="no_such_alg", force_alg_selection=False, vendor_options=vendor
    )
    call(send, recv, *args, config=fallback)
    with pytest.raises(NCCLError) as err:
        call(send, recv, *args, config=coll_config(alg_selection="no_such_alg"))
    assert err.value.status == Result.InvalidArgument
