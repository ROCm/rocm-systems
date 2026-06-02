"""rocshmem4py: Python bindings for rocSHMEM.

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License. See LICENSE for details.

Module layout
-------------
``rocshmem4py``
    Core host API — framework-agnostic.  Works without any ML framework.
    Symmetric memory is represented by :class:`SymmetricBuffer`, which
    exposes a raw HIP device pointer via ``.ptr`` and implements
    ``__cuda_array_interface__`` for zero-copy interop with PyTorch.

``rocshmem4py.interop.torch``
    PyTorch-specific helpers (requires ``torch``).
"""

import os
import ctypes
from typing import Sequence, Optional, Any

__version__ = "0.1.0"
__author__ = "Advanced Micro Devices, Inc."

try:
    # Keep this import list aligned with src/rocshmem4py.cc host bindings.
    from _rocshmem4py import (
        hip_device_synchronize,
        rocshmem_init,
        rocshmem_finalize,
        rocshmem_hipmodule_init,
        rocshmem_my_pe,
        rocshmem_n_pes,
        rocshmem_team_my_pe,
        rocshmem_team_n_pes,
        rocshmem_malloc,
        rocshmem_free,
        rocshmem_ptr,
        rocshmem_barrier_all,
        rocshmem_barrier_all_on_stream,
        rocshmem_fence,
        rocshmem_quiet,
        rocshmem_get_uniqueid,
        rocshmem_init_attr,
        rocshmem_putmem,
        rocshmem_getmem,
        rocshmem_putmem_nbi,
        rocshmem_getmem_nbi,
        rocshmem_putmem_on_stream,
        rocshmem_getmem_on_stream,
        rocshmem_putmem_signal_on_stream,
        rocshmem_signal_wait_until_on_stream,
        TeamConfig,
        rocshmem_sync_all,
        rocshmem_sync_all_on_stream,
        # TODO: rocshmem_ctx_{create,destroy,fence,quiet} to be added later
        # due to IPC no-MPI HostInterface aborting on non-MPI WindowInfo
        # rocshmem_ctx_create,
        # rocshmem_ctx_destroy,
        # rocshmem_ctx_fence,
        # rocshmem_ctx_quiet,
        rocshmem_alltoallmem_on_stream,
        rocshmem_broadcastmem_on_stream,
        rocshmem_query_thread,
        rocshmem_global_exit,
        rocshmem_dump_stats,
        rocshmem_reset_stats,
        rocshmem_get_device_ctx,
        # Constants
        ROCSHMEM_SUCCESS,
        ROCSHMEM_SIGNAL_SET,
        ROCSHMEM_SIGNAL_ADD,
        ROCSHMEM_CMP_EQ,
        ROCSHMEM_CMP_NE,
        ROCSHMEM_CMP_GT,
        ROCSHMEM_CMP_GE,
        ROCSHMEM_CMP_LT,
        ROCSHMEM_CMP_LE,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import _rocshmem4py. "
        "Ensure rocSHMEM is installed and LD_LIBRARY_PATH includes "
        "the rocSHMEM library path."
    ) from e


# Distinct Python sentinels for the two special team handles.  Both are
# translated by the C binding's resolve_team_handle():
#   ROCSHMEM_TEAM_WORLD   ==  0  -> host::ROCSHMEM_TEAM_WORLD (runtime ptr)
#   ROCSHMEM_TEAM_INVALID == -1  -> rocshmem_team_t(nullptr) (rocSHMEM ABI)
ROCSHMEM_TEAM_WORLD = 0
ROCSHMEM_TEAM_INVALID = -1
_SPECIAL_TEAM_HANDLES = {ROCSHMEM_TEAM_INVALID, ROCSHMEM_TEAM_WORLD}


# ---------------------------------------------------------------------------
# Team APIs (registry + tracked split/destroy)
# ---------------------------------------------------------------------------

# Live-team registry: every successful split_strided adds the new handle
# here so finalize_with_torch can drain leaks before rocshmem_finalize.
_live_teams: set = set()


def _team_split_strided_tracked(parent, start, stride, size,
                                config=None, mask=0):
    """Tracked variant of rocshmem_team_split_strided.

    Records every successfully created team handle in ``_live_teams`` so
    finalize_with_torch can destroy any leaked teams before
    rocshmem_finalize.  Returns ``(status, team_handle)`` matching the
    raw pybind signature.  Callers that intentionally bypass this wrapper
    via ``_rocshmem4py.rocshmem_team_split_strided`` own the returned team
    handle and must destroy it themselves.

    Translates the rocSHMEM ABI INVALID return (C ``nullptr`` ==
    Python ``0``) to the Python sentinel ``ROCSHMEM_TEAM_INVALID``
    (``-1``) so callers can distinguish a non-member return from
    ``ROCSHMEM_TEAM_WORLD``.  Without this translation, a non-member of
    a child split and a successful WORLD-equivalent call would both
    look like ``0`` to Python.
    """
    from _rocshmem4py import rocshmem_team_split_strided as _raw
    status, team = _raw(parent, start, stride, size, config, mask)
    if team == 0:
        return status, ROCSHMEM_TEAM_INVALID
    if status == ROCSHMEM_SUCCESS and team not in _SPECIAL_TEAM_HANDLES:
        _live_teams.add(team)
    return status, team


def _team_destroy_tracked(team):
    """Tracked variant of rocshmem_team_destroy.

    Removes the handle from the live-team registry on destroy.  Calling
    on an unknown handle is a no-op (matches rocshmem_team_destroy
    behavior for ROCSHMEM_TEAM_{INVALID,WORLD,SHARED}).
    """
    if team in _SPECIAL_TEAM_HANDLES:
        return
    from _rocshmem4py import rocshmem_team_destroy as _raw
    _raw(team)
    _live_teams.discard(team)


# Public names always go through the tracked variants.  Direct callers of
# _rocshmem4py.rocshmem_team_split_strided bypass the registry by design
# (e.g. tests that intentionally exercise the raw API); those callers own
# any returned team handle and must destroy it themselves.
rocshmem_team_split_strided = _team_split_strided_tracked
rocshmem_team_destroy = _team_destroy_tracked

from _rocshmem4py import rocshmem_team_translate_pe  # noqa: E402


# ---------------------------------------------------------------------------
# Full host AMO matrix (re-export by symbol-name discovery)
# ---------------------------------------------------------------------------
#
# The pybind layer generates host AMO bindings via macros.  Hand-listing
# them in the import block is fragile — instead, walk the extension module
# and re-export every name matching the AMO naming convention.  These are
# runtime-backed host APIs; IPC no-MPI support for host AMOs needs to be 
# added to the rocSHMEM runtime.

_AMO_TYPE_PREFIXES = (
    "int_", "long_", "longlong_",
    "uint32_", "uint64_", "size_", "ptrdiff_",
    "float_", "double_",
)


def _is_amo_name(name: str) -> bool:
    if not name.startswith("rocshmem_"):
        return False
    if "_atomic_" not in name:
        return False
    suffix = name[len("rocshmem_"):]
    return any(suffix.startswith(p) for p in _AMO_TYPE_PREFIXES)


def _import_amo_symbols() -> list:
    import _rocshmem4py as _ext
    names = []
    for name in dir(_ext):
        if _is_amo_name(name):
            globals()[name] = getattr(_ext, name)
            names.append(name)
    return names


_AMO_NAMES = tuple(sorted(_import_amo_symbols()))


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

_HOST_API_BINDINGS = tuple(
    sorted(
        name
        for name in globals()
        if (name.startswith("rocshmem_") or name.startswith("ROCSHMEM_"))
        and not name.startswith("_")
    )
)

__all__ = [
    '__version__',
    *_HOST_API_BINDINGS,
    'ROCSHMEM_TEAM_INVALID',
    'ROCSHMEM_TEAM_WORLD',
    'TeamConfig',
    # Core framework-agnostic symmetric memory
    'SymmetricBuffer',
    'rocshmem_create_buffer',
    'rocshmem_get_peer_buffer',
    # init / finalize
    'init_rocshmem_by_uniqueid',
    'init_with_mpi',
    'init_with_torch',
    'finalize_with_mpi',
    'finalize_with_torch',
    'set_hip_device_from_env',
]


def set_hip_device_from_env():
    """Pin the HIP device from LOCAL_RANK / OMPI_COMM_WORLD_LOCAL_RANK.

    Call before ``rocshmem_init()`` on the raw (non-torch, non-mpi4py) path so
    each PE owns a distinct GPU.  ``init_with_torch()`` and ``init_with_mpi()``
    already do this internally.
    """
    local_rank = os.environ.get("LOCAL_RANK") or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
    if local_rank is not None:
        try:
            hip = ctypes.CDLL("libamdhip64.so")
            hip.hipSetDevice(int(local_rank))
        except OSError:
            pass


_set_hip_device_from_env = set_hip_device_from_env


class SymmetricBuffer:
    """RAII wrapper around ``rocshmem_malloc`` that exposes the
    ``__cuda_array_interface__`` protocol for zero-copy interop with
    consumers that implement that protocol.

    Framework-agnostic: works without any ML framework installed.
    """

    def __init__(self, size: int, *, ptr: Optional[int] = None,
                 dtype=None, own_data: bool = True):
        if ptr is not None:
            self.ptr = ptr
            self.nbytes = size
        else:
            # rocshmem_malloc spawns kernels on every PE that can race with
            # concurrent GPU module setup unless the device is quiesced first.
            hip_device_synchronize()
            self.ptr = rocshmem_malloc(size)
            self.nbytes = size
        self.size = size
        self.own_data = own_data
        self._freed = False

        try:
            import torch
            self.dtype = dtype if dtype is not None else torch.uint8
        except ImportError:
            self.dtype = dtype

        try:
            self._hip = ctypes.CDLL("libamdhip64.so")
            device = ctypes.c_int()
            self._hip.hipGetDevice(ctypes.byref(device))
            self._device = device.value
        except OSError:
            self._hip = None
            self._device = 0

        # prevent dangling reference during interpreter shutdown
        self._rocshmem_free = rocshmem_free

        self.__cuda_array_interface__ = {
            "data": (self.ptr, False),
            "shape": (self.nbytes,),
            "typestr": "<i1",
            "strides": None,
            "version": 3,
        }

    def __del__(self):
        if self.own_data and not self._freed:
            self.free()

    def free(self):
        """Free the symmetric memory with device save/restore and sync."""
        if self._freed:
            return
        if not self.own_data:
            self._freed = True
            return

        if self._hip is not None:
            device = ctypes.c_int()
            self._hip.hipGetDevice(ctypes.byref(device))
            prev_device = device.value

            if prev_device != self._device:
                self._hip.hipSetDevice(self._device)

            self._hip.hipDeviceSynchronize()
            self._rocshmem_free(self.ptr)
            self._hip.hipDeviceSynchronize()

            if prev_device != self._device:
                self._hip.hipSetDevice(prev_device)
        else:
            self._rocshmem_free(self.ptr)

        self._freed = True

    def get_remote_ptr(self, pe: int) -> int:
        return rocshmem_ptr(self.ptr, pe)

    def put(self, source_ptr: int, nelems: int, pe: int):
        rocshmem_putmem(self.ptr, source_ptr, nelems, pe)

    def get(self, source_ptr: int, nelems: int, pe: int):
        rocshmem_getmem(self.ptr, source_ptr, nelems, pe)

    def __int__(self) -> int:
        return self.ptr

    def __repr__(self) -> str:
        status = "freed" if self._freed else f"size={self.size}"
        return f"SymmetricBuffer(ptr=0x{self.ptr:x}, {status})"


# ---------------------------------------------------------------------------
# Framework-agnostic symmetric memory allocation
# ---------------------------------------------------------------------------

def rocshmem_create_buffer(nbytes: int) -> SymmetricBuffer:
    """Collectively allocate ``nbytes`` of symmetric memory.

    Returns a :class:`SymmetricBuffer` that exposes
    ``__cuda_array_interface__`` for zero-copy integration with consumers
    that understand the protocol.

    **This is a collective operation** — all PEs must call this function
    with the same ``nbytes`` argument, matching the SHMEM symmetric heap
    contract.

    For the PyTorch convenience wrapper see :mod:`rocshmem4py.interop.torch`.
    """
    return SymmetricBuffer(nbytes)


# ---------------------------------------------------------------------------
# Heap-base introspection helpers for device-side pointer translation
# ---------------------------------------------------------------------------

def rocshmem_get_heap_bases(ptr: int) -> list:
    """Return the per-PE base pointer for a symmetric allocation.

    For PE ``i``, returns ``rocshmem_ptr(ptr, i)`` — the address of the
    same allocation in PE ``i``'s symmetric heap as visible from the
    local PE.  Entries are 0 when ``rocshmem_ptr`` returns NULL (e.g. on
    the RO backend, where direct remote dereference is unavailable).

    This helper constructs the ``heap_bases`` array used by device kernels
    that translate a local symmetric pointer to a peer's pointer via
    ``heap_bases[peer] + (local_ptr - heap_bases[my_pe])``.
    """
    n = rocshmem_n_pes()
    return [rocshmem_ptr(ptr, pe) for pe in range(n)]


def rocshmem_get_peer_buffer(buf: SymmetricBuffer, peer: int) -> SymmetricBuffer:
    """Return a non-owning :class:`SymmetricBuffer` view of *peer*'s buffer.

    Wraps ``rocshmem_ptr``.  Returns a zero-copy view that maps directly
    into the peer's symmetric heap — no data is copied.  The caller must
    not call ``free()`` on the returned buffer; the peer's original buffer
    must remain alive while the view is in use.

    Available only on backends that support direct remote memory access
    (IPC).  On other backends ``rocshmem_ptr`` returns NULL and this
    function raises :class:`RuntimeError`.

    For backends without direct remote access, use ``rocshmem_getmem``
    (or the ``*_on_stream`` variants) to explicitly copy peer data into a
    local symmetric buffer.
    """
    if peer == rocshmem_my_pe():
        return buf
    ptr = rocshmem_ptr(buf.ptr, peer)
    if ptr == 0:
        raise RuntimeError(
            f"rocshmem_ptr returned NULL for peer {peer} — remote direct "
            "memory access not supported by the current backend. "
            "Use rocshmem_getmem to copy peer data instead."
        )
    return SymmetricBuffer(buf.nbytes, ptr=ptr, own_data=False)


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------


def _drain_live_teams() -> None:
    """Destroy any teams left in the registry before rocshmem_finalize.

    Team destruction is collective: every PE must call ``rocshmem_team_destroy``
    on the same teams in the same order, or rocshmem_finalize will segfault.
    Iterating the sorted set guarantees identical order across PEs.
    """
    if not _live_teams:
        return
    from _rocshmem4py import rocshmem_team_destroy as _raw_destroy

    for team in sorted(_live_teams):
        _raw_destroy(team)
    _live_teams.clear()


def init_rocshmem_by_uniqueid(group: 'torch.distributed.ProcessGroup'):
    """Broadcast unique ID from rank 0 and call ``rocshmem_init_attr``."""
    import torch
    import torch.distributed as dist

    rank = dist.get_rank(group)
    nranks = dist.get_world_size(group)

    bcast_obj = [rocshmem_get_uniqueid() if rank == 0 else None]
    dist.broadcast_object_list(bcast_obj, src=0, group=group)
    dist.barrier(group=group)

    rocshmem_init_attr(rank, nranks, bcast_obj[0])

    dist.barrier(group=group)
    torch.cuda.synchronize()


def init_with_mpi(mpi_comm: Optional[Any] = None):
    """Initialize rocSHMEM using mpi4py to broadcast the unique ID.

    The caller must launch with ``mpirun`` (or another MPI launcher) so that
    ``MPI.COMM_WORLD`` actually spans all PEs.  Under ``torchrun`` (or any
    non-MPI launcher) ``MPI.COMM_WORLD`` is a singleton per process and this
    function raises -- use ``init_with_torch()`` in that case.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        raise ImportError("mpi4py is required for init_with_mpi().")

    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD

    _set_hip_device_from_env()

    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    # Sanity check: if a process launcher set WORLD_SIZE/OMPI_COMM_WORLD_SIZE
    # but MPI.COMM_WORLD doesn't match, we are not running under MPI and the
    # init below would silently create N independent size-1 worlds.
    launcher_size_str = (
        os.environ.get("WORLD_SIZE") or os.environ.get("OMPI_COMM_WORLD_SIZE")
    )
    if launcher_size_str is not None and int(launcher_size_str) != size:
        raise RuntimeError(
            f"MPI.COMM_WORLD has size {size} but the launcher reports "
            f"WORLD_SIZE={launcher_size_str}. This usually means the job was "
            "launched with torchrun (or another non-MPI launcher) where "
            "MPI.COMM_WORLD is a singleton per process. Use init_with_torch() "
            "for torchrun, or launch with mpirun for init_with_mpi()."
        )

    unique_id = rocshmem_get_uniqueid() if rank == 0 else None
    unique_id = mpi_comm.bcast(unique_id, root=0)

    rocshmem_init_attr(rank, size, unique_id)
    mpi_comm.Barrier()

    global _rocshmem_initialized
    _rocshmem_initialized = True


def init_with_torch(group: Optional[Any] = None,
                    backend: str = 'cpu:gloo,cuda:nccl',
                    init_method: Optional[str] = None):
    """Initialize rocSHMEM using ``torch.distributed`` for coordination.

    Works under both ``torchrun`` and ``mpirun`` for IPC / GDA backends.
    The RO backend additionally requires an ``mpirun`` launch so that the
    OpenMPI launcher exports the ``OMPI_COMM_WORLD_*`` env vars rocSHMEM
    expects; ``torchrun`` alone is not sufficient for RO. ``OMPI_COMM_WORLD_*``
    env vars (when present) are auto-mapped to the ``RANK`` / ``WORLD_SIZE``
    / ``LOCAL_RANK`` variables ``torch.distributed`` reads.

    For the launcher x backend matrix and the structural reason RO requires
    ``mpirun``, see the project README:
    https://github.com/ROCm/rocm-systems/blob/develop/projects/rocshmem/python/README.md#troubleshooting
    """
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        raise ImportError("PyTorch is required for init_with_torch().")

    _populate_torch_env_from_mpi()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        if init_method is None:
            init_method = 'env://'
        try:
            import datetime
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                timeout=datetime.timedelta(seconds=1800),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize torch.distributed: {e}\n"
                "Ensure RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT are set."
            )

    if group is None:
        group = dist.group.WORLD

    init_rocshmem_by_uniqueid(group)

    global _rocshmem_initialized
    _rocshmem_initialized = True


def _populate_torch_env_from_mpi():
    """Map OMPI_COMM_WORLD_* env vars to RANK/LOCAL_RANK/WORLD_SIZE.

    Rendezvous defaults can be overridden via ROCSHMEM_MASTER_ADDR /
    ROCSHMEM_MASTER_PORT, or (equivalently) MASTER_ADDR / MASTER_PORT. Users
    running concurrent jobs should set one of these to avoid port collisions.
    """
    mapping = {
        "RANK": "OMPI_COMM_WORLD_RANK",
        "LOCAL_RANK": "OMPI_COMM_WORLD_LOCAL_RANK",
        "WORLD_SIZE": "OMPI_COMM_WORLD_SIZE",
        "LOCAL_WORLD_SIZE": "OMPI_COMM_WORLD_LOCAL_SIZE",
    }
    for torch_var, mpi_var in mapping.items():
        if torch_var not in os.environ and mpi_var in os.environ:
            os.environ[torch_var] = os.environ[mpi_var]

    os.environ.setdefault("MASTER_ADDR", os.environ.get("ROCSHMEM_MASTER_ADDR", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", os.environ.get("ROCSHMEM_MASTER_PORT", "29500"))


_rocshmem_initialized = False


def finalize_with_mpi():
    """Synchronized teardown for sessions initialized via ``init_with_mpi()``.

    Drains any teams the caller forgot to destroy (collective op, sorted for
    deterministic order across PEs), quiesces the device, then barriers and
    finalizes rocSHMEM.  MPI itself is not finalized here -- the launcher /
    application owns ``MPI_Finalize``.
    """
    global _rocshmem_initialized
    if not _rocshmem_initialized:
        return

    hip_device_synchronize()
    _drain_live_teams()
    rocshmem_barrier_all()
    rocshmem_finalize()
    _rocshmem_initialized = False


def finalize_with_torch():
    """Synchronized teardown for sessions initialized via ``init_with_torch()``.

    Order is critical:
      1. ``torch.cuda.synchronize()`` so no rank enters finalize with pending GPU work
      2. drain leaked teams (collective; deterministic order via sorted set)
      3. ``rocshmem_barrier_all`` + ``dist.barrier`` -- both world barriers, paired
         so neither library tears down while the other is still communicating
      4. ``rocshmem_finalize`` then ``destroy_process_group``
    """
    global _rocshmem_initialized
    if not _rocshmem_initialized:
        return

    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        return

    torch.cuda.synchronize()
    _drain_live_teams()
    rocshmem_barrier_all()
    if dist.is_initialized():
        dist.barrier()
    rocshmem_finalize()
    _rocshmem_initialized = False
    if dist.is_initialized():
        dist.destroy_process_group()
