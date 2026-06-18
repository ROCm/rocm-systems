"""Full host AMO matrix tests.

Binding coverage: 69 host AMO symbols are generated in ``rocshmem4py.cc`` and
re-exported via ``_import_amo_symbols()``. ``test_amo_matrix_completeness``
checks the same 69 (type, op) pairs — there is no drift vs the C++ macro block.

rocSHMEM also declares ``longlong`` fetch/cas/swap/etc. in the header, but these
bindings intentionally expose only ``longlong`` set/add/inc (see comment in
``rocshmem4py.cc``).

Runtime limitation: these are host AMO bindings.  IPC no-MPI runtime support
for host AMOs is not implemented yet, so behavioral AMO tests are opt-in via
``ROCSHMEM4PY_RUN_HOST_AMO_IPC=1`` until the runtime grows that support.
"""

import os

import pytest

import rocshmem4py
from rocshmem4py import (
    rocshmem_my_pe,
    rocshmem_n_pes,
    rocshmem_barrier_all,
)
from conftest import requires_torch, requires_multi_pe

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


requires_host_amo_runtime = pytest.mark.skipif(
    os.environ.get("ROCSHMEM4PY_RUN_HOST_AMO_IPC") != "1",
    reason=(
        "IPC no-MPI host AMOs are not supported in rocSHMEM runtime yet; "
        "set ROCSHMEM4PY_RUN_HOST_AMO_IPC=1 only when validating a runtime fix"
    ),
)


# ---------------------------------------------------------------------------
# Coverage matrix — every (type, op) we expect the pybind layer to expose.
# ---------------------------------------------------------------------------

INTEGER_TYPES = ["int", "long", "uint32", "uint64", "size", "ptrdiff"]
FLOAT_TYPES = ["float", "double"]
ALL_NUMERIC_TYPES = INTEGER_TYPES + FLOAT_TYPES

_EXPECTED_OPS = {
    "fetch": ALL_NUMERIC_TYPES,
    "set": ALL_NUMERIC_TYPES + ["longlong"],
    "swap": ALL_NUMERIC_TYPES,
    "compare_swap": INTEGER_TYPES,  # CAS not provided on float/double.
    "fetch_add": INTEGER_TYPES,
    "fetch_inc": INTEGER_TYPES,
    "add": INTEGER_TYPES + ["longlong"],
    "inc": INTEGER_TYPES + ["longlong"],
    "fetch_and": ["uint32", "uint64"],
    "fetch_or": ["uint32", "uint64"],
    "fetch_xor": ["uint32", "uint64"],
    "and": ["uint32", "uint64"],
    "or": ["uint32", "uint64"],
    "xor": ["uint32", "uint64"],
}


def test_amo_matrix_completeness():
    """Every (type, op) in the expected matrix is bound in rocshmem4py."""
    missing = []
    for op, types in _EXPECTED_OPS.items():
        for t in types:
            name = f"rocshmem_{t}_atomic_{op}"
            if not hasattr(rocshmem4py, name):
                missing.append(name)
    assert not missing, f"Missing AMO bindings: {missing}"

    from rocshmem4py import _AMO_NAMES

    expected_count = sum(len(types) for types in _EXPECTED_OPS.values())
    assert expected_count == len(_AMO_NAMES), (
        f"test matrix ({expected_count}) vs imported AMOs ({len(_AMO_NAMES)})"
    )
    # longlong: only set/add/inc per rocshMEM header subset in rocshmem4py.cc.
    longlong_ops = {
        n.removeprefix("rocshmem_longlong_atomic_")
        for n in _AMO_NAMES
        if "longlong" in n
    }
    assert longlong_ops == {"set", "add", "inc"}


@requires_torch
@requires_multi_pe
@requires_host_amo_runtime
def test_int_atomic_fetch_add_against_peer0():
    """Every PE atomically adds 1 to peer 0's counter; peer 0 sees n_pes."""
    n = rocshmem_n_pes()
    me = rocshmem_my_pe()

    # Symmetric int counter.
    nbytes = 4
    buf = rocshmem4py.SymmetricBuffer(nbytes)
    try:
        t = torch.as_tensor(buf, device="cuda").view(torch.int32).view([1])
        t.zero_()
        rocshmem_barrier_all()
        torch.cuda.synchronize()

        rocshmem4py.rocshmem_int_atomic_fetch_add(buf.ptr, 1, 0)
        rocshmem_barrier_all()
        torch.cuda.synchronize()

        if me == 0:
            assert int(t.item()) == n
    finally:
        buf.free()


@requires_torch
@requires_multi_pe
@requires_host_amo_runtime
def test_int_atomic_compare_swap_lock_pattern():
    """Use CAS to acquire a lock from peer 0; only one PE wins."""
    me = rocshmem_my_pe()
    nbytes = 4
    buf = rocshmem4py.SymmetricBuffer(nbytes)
    try:
        t = torch.as_tensor(buf, device="cuda").view(torch.int32).view([1])
        t.zero_()
        rocshmem_barrier_all()
        torch.cuda.synchronize()

        # Each PE attempts CAS(0 -> me+1) on peer 0's counter.
        rocshmem4py.rocshmem_int_atomic_compare_swap(buf.ptr, 0, me + 1, 0)
        rocshmem_barrier_all()
        torch.cuda.synchronize()

        if me == 0:
            assert 1 <= int(t.item()) <= rocshmem_n_pes()
    finally:
        buf.free()


@requires_torch
@requires_multi_pe
@requires_host_amo_runtime
def test_uint64_atomic_or_against_peer0():
    """Every PE ORs (1 << me) into peer 0's mask; peer 0 sees full bitmask."""
    n = rocshmem_n_pes()
    me = rocshmem_my_pe()

    nbytes = 8
    buf = rocshmem4py.SymmetricBuffer(nbytes)
    try:
        # uint64 = int64 view here; we set/check via ints.
        t = torch.as_tensor(buf, device="cuda").view(torch.int64).view([1])
        t.zero_()
        rocshmem_barrier_all()
        torch.cuda.synchronize()

        rocshmem4py.rocshmem_uint64_atomic_or(buf.ptr, 1 << me, 0)
        rocshmem_barrier_all()
        torch.cuda.synchronize()

        if me == 0:
            # Use unsigned interpretation: shift mask is fine as int up to 63 PEs.
            expected = (1 << n) - 1 if n < 64 else -1
            assert int(t.item()) == expected
    finally:
        buf.free()
