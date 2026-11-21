# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The nccl4py bindings track the NCCL release RCCL implements (AICOMRCCL-3197).

The bindings stamp ``version`` in every versioned struct with the NCCL version
they were generated for, and RCCL gates fields on it: ``ncclCommQueryProperties``
fills the 2.31 ``ncclCommProperties_t`` fields (``commHash`` ..
``devCommRuntimeVersionSize``) only for ``version >= 2.31.0``. Before the sync
RCCL implemented NCCL 2.31.2 while the bindings were generated for 2.30.7 and
stamped 23007.

Run through ``bindings_pytest.sh``; the query tests need one GPU.
"""

import ctypes
import os

import pytest

from nccl.bindings import nccl as bindings


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


@pytest.fixture(scope="module")
def queried_properties():
    """The library, and a CommProperties built by the bindings and filled by RCCL."""
    lib = ctypes.CDLL(os.environ["NCCL_LIBRARY"])  # the library the bindings load
    lib.ncclCommInitRank.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        _UniqueId,
        ctypes.c_int,
    ]
    lib.ncclCommQueryProperties.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    uid, comm = _UniqueId(), ctypes.c_void_p()
    assert lib.ncclGetUniqueId(ctypes.byref(uid)) == 0
    assert lib.ncclCommInitRank(ctypes.byref(comm), 1, uid, 0) == 0
    try:
        props = bindings.CommProperties()  # size, magic and version set by the bindings
        assert lib.ncclCommQueryProperties(comm, props.ptr) == 0
    finally:
        lib.ncclCommDestroy(comm)
    return lib, props


def test_control_query_fills_rank(queried_properties):
    lib, props = queried_properties
    version = ctypes.c_int()
    assert lib.ncclGetVersion(ctypes.byref(version)) == 0
    assert bindings.get_version() == version.value  # the bindings load the library under test
    assert (props.rank, props.n_ranks) == (0, 1)


def test_pod_version_stamp_matches_library_release():
    lib = bindings.get_version()
    for cls in (
        bindings.Config,
        bindings.SimInfo,
        bindings.CommProperties,
        bindings.DevCommRequirements,
    ):
        stamp = cls().version
        assert f"{stamp // 10000}.{stamp // 100 % 100}.{stamp % 100}" == bindings.__version__
        # RCCL gates a release's fields on NCCL_VERSION(X, Y, 0) and rejects newer stamps.
        assert lib - lib % 100 <= stamp <= lib, (
            f"{cls.__name__}.version is {stamp}, RCCL implements {lib}"
        )


def test_query_fills_nccl_2_31_fields(queried_properties):
    # RCCL reports sizeof(ncclDevComm_t) only to callers stamped >= 2.31.0, and
    # the regenerated DevComm mirror must have the same size.
    _, props = queried_properties
    assert props.dev_comm_runtime_version_size == bindings.dev_comm_dtype.itemsize
