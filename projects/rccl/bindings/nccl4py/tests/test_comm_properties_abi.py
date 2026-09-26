# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for AICOMRCCL-1530 (upstream NCCL 2.30.3 fixed issue):

    "Fixed ncclGinType_t uint8_t enum compatibility issue in nccl4py."

``ncclGinType_t`` (see ``src/include/nccl_device/core.h``) is a plain C
enum with no explicit underlying-type annotation::

    typedef enum {
      NCCL_GIN_TYPE_NONE = 0,
      NCCL_GIN_TYPE_PROXY = 2,
      NCCL_GIN_TYPE_GDAKI = 3,
    } ncclGinType_t;

so the compiler lays it out as a full, 4-byte-aligned ``int`` on every
ABI RCCL supports -- *not* as a ``uint8_t``. ``ncclCommProperties_t``
embeds two such fields (``ginType`` and ``railedGinType``) interleaved
with genuinely single-byte fields (``deviceApiSupport``,
``multimemSupport``, ``hostRmaSupport``, declared ``bool`` in the real
C struct and ``uint8_t`` in the Cython binding).

NVIDIA's public history has no diffable "before" commit for this fix
(the whole GIN feature landed in one post-fix squash commit, and RCCL
ported nccl4py from an already-fixed NVIDIA release), so the bug class
described below is reconstructed from the release-note wording rather
than quoted from a real historical diff: a plausible pre-fix
``cynccl.pxd`` would have mistakenly typed ``ginType``/``railedGinType``
as ``uint8_t`` too (by analogy with their single-byte neighbours). That
would silently shrink and misalign ``ncclCommProperties_t`` relative to
the real, compiler-generated ``ncclCommProperties`` struct that
``ncclCommQueryProperties()`` populates: every field at or after
``ginType`` would be read from the wrong byte offset and the struct's
total size would shrink (48 vs. the correct 56 bytes on x86_64), so
callers would silently get garbage back from ``Communicator.gin_type``,
``Communicator.n_lsa_teams``, ``Communicator.host_rma_support``, and
``Communicator.railed_gin_type``.

This test builds a reference ``ctypes.Structure`` that mirrors the
*real* upstream C layout (bools as 1 byte, GIN enums as native
``int``) and uses it two ways, without requiring a GPU, MPI, or a real
communicator:

1. It cross-checks that ``comm_properties_dtype`` -- derived by Cython
   purely from ``cynccl.pxd`` at build time, independent of any
   dlsym'd library call -- agrees with the reference layout on total
   size and on every field's byte offset.
2. It round-trips a buffer built from the reference layout through the
   public ``CommProperties.from_buffer()`` API (the same code path
   ``Communicator.gin_type``/``railed_gin_type`` use) and checks that
   every field, especially the ones sandwiching the GIN fields,
   decodes to the value that was actually written.

Declaring ``ginType``/``railedGinType`` as ``uint8_t`` in ``cynccl.pxd``
(simulating the regression class described above) makes every test in
this module fail immediately: either via a buffer-size ``ValueError``
from ``from_buffer()``, or via wrong decoded field values. This was
confirmed manually while investigating AICOMRCCL-1530 by making that
change locally and rebuilding the extension, which shifted
``comm_properties_dtype.itemsize`` from 56 to 48 bytes and every offset
from ``gin_type`` onward -- then reverted before committing, since it
does not reflect any real NVIDIA or RCCL commit.
"""

import ctypes

import pytest

from nccl.bindings import CommProperties, GinType, comm_properties_dtype


class _NcclCommPropertiesRef(ctypes.Structure):
    """Reference (known-correct) layout of ``ncclCommProperties_t``.

    Mirrors ``struct ncclCommProperties`` in
    ``src/include/nccl_device/core.h``::

        size_t size; unsigned int magic; unsigned int version;
        int rank, nRanks, cudaDev, nvmlDev;
        bool deviceApiSupport, multimemSupport;
        ncclGinType_t ginType;
        int nLsaTeams;
        bool hostRmaSupport;
        ncclGinType_t railedGinType;
    """

    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magic", ctypes.c_uint),
        ("version", ctypes.c_uint),
        ("rank", ctypes.c_int),
        ("nRanks", ctypes.c_int),
        ("cudaDev", ctypes.c_int),
        ("nvmlDev", ctypes.c_int),
        ("deviceApiSupport", ctypes.c_uint8),
        ("multimemSupport", ctypes.c_uint8),
        ("ginType", ctypes.c_int),  # ncclGinType_t: plain C enum -> `int`
        ("nLsaTeams", ctypes.c_int),
        ("hostRmaSupport", ctypes.c_uint8),
        ("railedGinType", ctypes.c_int),  # ncclGinType_t
    ]


class _NcclCommPropertiesBuggyRef(ctypes.Structure):
    """Reference layout of the regression this test guards against.

    Identical to :class:`_NcclCommPropertiesRef` except ``ginType`` and
    ``railedGinType`` are mistakenly declared as ``uint8_t`` instead of the
    real ``ncclGinType_t`` (4-byte enum), reproducing the pre-fix
    ``cynccl.pxd`` bug class described in the module docstring. Used only to
    compute the buggy struct's size explicitly, so the negative test below
    stays a faithful regression guard even if fields are added/reordered.
    """

    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magic", ctypes.c_uint),
        ("version", ctypes.c_uint),
        ("rank", ctypes.c_int),
        ("nRanks", ctypes.c_int),
        ("cudaDev", ctypes.c_int),
        ("nvmlDev", ctypes.c_int),
        ("deviceApiSupport", ctypes.c_uint8),
        ("multimemSupport", ctypes.c_uint8),
        ("ginType", ctypes.c_uint8),  # bug: should be ncclGinType_t (int)
        ("nLsaTeams", ctypes.c_int),
        ("hostRmaSupport", ctypes.c_uint8),
        ("railedGinType", ctypes.c_uint8),  # bug: should be ncclGinType_t
    ]


# Field-name mapping: reference (C) name -> nccl4py Python property name.
_FIELD_MAP = [
    ("size", "size_"),
    ("magic", "magic"),
    ("version", "version"),
    ("rank", "rank"),
    ("nRanks", "n_ranks"),
    ("cudaDev", "cuda_dev"),
    ("nvmlDev", "nvml_dev"),
    ("deviceApiSupport", "device_api_support"),
    ("multimemSupport", "multimem_support"),
    ("ginType", "gin_type"),
    ("nLsaTeams", "n_lsa_teams"),
    ("hostRmaSupport", "host_rma_support"),
    ("railedGinType", "railed_gin_type"),
]


def test_comm_properties_dtype_matches_reference_c_abi():
    """`comm_properties_dtype`'s size/offsets must match the real C ABI."""
    ref_size = ctypes.sizeof(_NcclCommPropertiesRef)
    assert comm_properties_dtype.itemsize == ref_size, (
        "ncclCommProperties_t ABI mismatch: nccl4py computed itemsize="
        f"{comm_properties_dtype.itemsize}, but the reference NCCL/RCCL C "
        f"layout is {ref_size} bytes. This is the AICOMRCCL-1530 / NCCL "
        "2.30.3 'ncclGinType_t uint8_t enum compatibility' regression -- "
        "check that ginType/railedGinType are declared as `ncclGinType_t`, "
        "not `uint8_t`, in cynccl.pxd."
    )

    for c_name, py_name in _FIELD_MAP:
        expected_offset = getattr(_NcclCommPropertiesRef, c_name).offset
        actual_offset = comm_properties_dtype.fields[py_name][1]
        assert actual_offset == expected_offset, (
            f"ncclCommProperties_t.{c_name} ('{py_name}') offset mismatch: "
            f"nccl4py={actual_offset}, reference C ABI={expected_offset}"
        )


@pytest.mark.parametrize(
    "gin_type,railed_gin_type",
    [
        (GinType.NONE, GinType.NONE),
        (GinType.PROXY, GinType.GDAKI),
        (GinType.GDAKI, GinType.PROXY),
    ],
)
def test_comm_properties_from_buffer_round_trip(gin_type, railed_gin_type):
    """`CommProperties.from_buffer()` must decode every field at its real
    ABI offset, especially the fields sandwiching the two GIN enum fields.
    """
    ref = _NcclCommPropertiesRef(
        size=ctypes.sizeof(_NcclCommPropertiesRef),
        magic=0xDEADBEEF,
        version=123456,
        rank=3,
        nRanks=8,
        cudaDev=5,
        nvmlDev=9,
        deviceApiSupport=1,
        multimemSupport=0,
        ginType=int(gin_type),
        nLsaTeams=42,
        hostRmaSupport=1,
        railedGinType=int(railed_gin_type),
    )

    props = CommProperties.from_buffer(bytes(ref))

    assert props.magic == 0xDEADBEEF
    assert props.version == 123456
    assert props.rank == 3
    assert props.n_ranks == 8
    assert props.cuda_dev == 5
    assert props.nvml_dev == 9
    assert bool(props.device_api_support) is True
    assert bool(props.multimem_support) is False
    assert GinType(props.gin_type) == gin_type
    assert props.n_lsa_teams == 42
    assert bool(props.host_rma_support) is True
    assert GinType(props.railed_gin_type) == railed_gin_type


def test_comm_properties_from_buffer_rejects_undersized_buffer():
    """A buffer sized for the buggy uint8_t-packed struct must be rejected
    rather than silently misinterpreted.
    """
    undersized = bytes(ctypes.sizeof(_NcclCommPropertiesBuggyRef))
    assert len(undersized) < ctypes.sizeof(_NcclCommPropertiesRef)
    with pytest.raises(ValueError, match="buffer length must be"):
        CommProperties.from_buffer(undersized)
