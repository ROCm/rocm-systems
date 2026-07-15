"""PyTorch interoperability for rocshmem4py symmetric memory.

All functions in this module require PyTorch to be installed.  They raise
``ImportError`` with a clear message if it is not.

Memory lifecycle
----------------
``create_tensor`` returns a tensor whose backing symmetric memory is tracked
via a ``_symm_buf`` attribute on the tensor.  As long as the tensor is alive,
the :class:`~rocshmem4py.SymmetricBuffer` is alive and the symmetric heap
allocation is valid.

Call :func:`free_tensor` explicitly before letting the tensor go out of
scope if you need deterministic, collective-safe teardown (all PEs free at
the same logical point).  If ``free_tensor`` is not called, the memory is
freed when the last reference to the tensor is dropped — which may differ
across PEs and cause hangs during finalization.
"""

from typing import TYPE_CHECKING, Optional, Sequence

import rocshmem4py

if TYPE_CHECKING:
    import torch


def create_tensor(shape: Sequence[int], dtype: 'torch.dtype') -> 'torch.Tensor':
    """Collectively allocate symmetric memory and return it as a PyTorch tensor.

    **Collective operation** — all PEs must call this function with identical
    arguments.

    Parameters
    ----------
    shape:
        Tensor shape (sequence of ints).
    dtype:
        ``torch.dtype`` for the returned tensor.

    Returns
    -------
    torch.Tensor
        A CUDA tensor backed by rocSHMEM symmetric memory.
        ``tensor.__symm_tensor__`` is set to ``True`` to mark it as a
        symmetric tensor eligible for :func:`get_peer_tensor`.

    See Also
    --------
    free_tensor : Explicit collective-safe deallocation.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for rocshmem4py.interop.torch.")

    nbytes = torch.Size(shape).numel() * dtype.itemsize
    torch.cuda.synchronize()
    buf = rocshmem4py.rocshmem_create_buffer(nbytes)
    # torch.as_tensor consumes __cuda_array_interface__; the resulting
    # view is int8 — reinterpret bytes to the requested dtype and shape.
    t_base = torch.as_tensor(buf, device="cuda")
    t = t_base.view(dtype).view(list(shape))
    # Keep buf alive: torch does not hold a reference to the source object
    # after consuming __cuda_array_interface__.
    t._symm_buf = buf
    setattr(t, "__symm_tensor__", True)
    return t


def get_peer_tensor(tensor: 'torch.Tensor', peer: int) -> 'torch.Tensor':
    """Return a zero-copy tensor view of *peer*'s symmetric buffer.

    Wraps ``rocshmem_ptr``.  Returns a CUDA tensor that maps directly into
    the peer's symmetric heap — no data is copied.

    Available only on backends that support direct remote memory access
    (IPC).  On other backends ``rocshmem_ptr`` returns NULL and this
    function raises :class:`RuntimeError`.

    For backends without direct remote access, use
    ``rocshmem4py.rocshmem_getmem_on_stream`` (or
    ``rocshmem4py.rocshmem_getmem``) to explicitly copy peer data into a
    local symmetric tensor.

    Parameters
    ----------
    tensor:
        A tensor created by :func:`create_tensor`.
    peer:
        Target PE index.

    Returns
    -------
    torch.Tensor
        A CUDA tensor view of the peer's symmetric memory region.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for rocshmem4py.interop.torch.")

    if not getattr(tensor, "__symm_tensor__", False):
        raise ValueError("tensor is not a symmetric tensor (missing __symm_tensor__ attribute). "
                         "Use create_tensor() to allocate it.")
    if not tensor.is_cuda:
        raise ValueError("tensor must be on a CUDA device.")

    if peer == rocshmem4py.rocshmem_my_pe():
        return tensor

    ptr = rocshmem4py.rocshmem_ptr(tensor.data_ptr(), peer)
    if ptr == 0:
        raise RuntimeError(
            f"rocshmem_ptr returned NULL for peer {peer} — remote direct "
            "memory access not supported by the current backend. "
            "Use rocshmem_getmem_on_stream to copy peer data instead."
        )
    buf = rocshmem4py.SymmetricBuffer(tensor.nbytes, ptr=ptr, own_data=False)
    t = torch.as_tensor(buf, device="cuda").view(tensor.dtype).view(tensor.shape)
    t._symm_buf = buf
    return t



def free_tensor(tensor: 'torch.Tensor') -> None:
    """Explicitly free the symmetric memory backing *tensor*.

    Call this on all PEs at the same logical program point to ensure
    collective-safe deallocation.  After this call *tensor* is invalid
    and must not be used.

    Parameters
    ----------
    tensor:
        A tensor created by :func:`create_tensor`.
    """
    buf = getattr(tensor, '_symm_buf', None)
    if buf is None:
        raise ValueError("tensor is not backed by rocSHMEM symmetric memory. "
                         "Was it allocated with create_tensor()?")
    buf.free()
    tensor._symm_buf = None


# ---------------------------------------------------------------------------
# Tensor-aware RMA wrappers (portable across every rocSHMEM backend)
# ---------------------------------------------------------------------------

def _stream_handle(stream: Optional['torch.cuda.Stream'] = None) -> int:
    """Return the raw hipStream_t handle for a torch.cuda.Stream.

    Defaults to ``torch.cuda.current_stream()`` when *stream* is None.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for rocshmem4py.interop.torch.")

    if stream is None:
        stream = torch.cuda.current_stream()
    return stream.cuda_stream


def put(dst: 'torch.Tensor', src: 'torch.Tensor', peer: int,
        stream: Optional['torch.cuda.Stream'] = None) -> None:
    """Stream-ordered put: copy *src* (local) -> *dst* on *peer*.

    Tensor-aware wrapper over ``rocshmem_putmem_on_stream`` that works on
    every rocSHMEM backend (IPC, RO, RDMA).  Both tensors must be symmetric
    (allocated by :func:`create_tensor`) and have identical nbytes.

    Parameters
    ----------
    dst:
        Destination tensor on *peer* (symmetric).
    src:
        Source tensor on the local PE (symmetric).
    peer:
        Target PE for the put.
    stream:
        ``torch.cuda.Stream`` to enqueue the operation on.
        Defaults to ``torch.cuda.current_stream()``.
    """
    if dst.nbytes != src.nbytes:
        raise ValueError(
            f"dst.nbytes ({dst.nbytes}) must match src.nbytes ({src.nbytes})"
        )
    rocshmem4py.rocshmem_putmem_on_stream(
        dst.data_ptr(), src.data_ptr(), src.nbytes, peer,
        _stream_handle(stream),
    )


def get(dst: 'torch.Tensor', src: 'torch.Tensor', peer: int,
        stream: Optional['torch.cuda.Stream'] = None) -> None:
    """Stream-ordered get: copy *src* from *peer* -> *dst* (local).

    Tensor-aware wrapper over ``rocshmem_getmem_on_stream`` that works on
    every rocSHMEM backend (IPC, RO, RDMA).  Both tensors must be symmetric
    (allocated by :func:`create_tensor`) and have identical nbytes.

    Parameters
    ----------
    dst:
        Destination tensor on the local PE (symmetric).
    src:
        Source tensor on *peer* (symmetric).
    peer:
        Source PE for the get.
    stream:
        ``torch.cuda.Stream`` to enqueue the operation on.
        Defaults to ``torch.cuda.current_stream()``.
    """
    if dst.nbytes != src.nbytes:
        raise ValueError(
            f"dst.nbytes ({dst.nbytes}) must match src.nbytes ({src.nbytes})"
        )
    rocshmem4py.rocshmem_getmem_on_stream(
        dst.data_ptr(), src.data_ptr(), dst.nbytes, peer,
        _stream_handle(stream),
    )


def barrier_all(stream: Optional['torch.cuda.Stream'] = None) -> None:
    """Stream-ordered collective barrier across all PEs.

    Thin wrapper over ``rocshmem_barrier_all_on_stream``.

    Parameters
    ----------
    stream:
        ``torch.cuda.Stream`` to enqueue the barrier on.
        Defaults to ``torch.cuda.current_stream()``.
    """
    rocshmem4py.rocshmem_barrier_all_on_stream(_stream_handle(stream))


# ---------------------------------------------------------------------------
# Heap-base helper for device kernels
# ---------------------------------------------------------------------------

def get_heap_bases(tensor: 'torch.Tensor') -> 'torch.Tensor':
    """Return an int64 CUDA tensor of per-PE base addresses for a symmetric tensor.

    The returned tensor has shape ``(n_pes,)``, dtype ``torch.int64``, on
    the current CUDA device.  Device kernels can use this array to translate
    a local symmetric pointer to a peer's pointer via
    ``heap_bases[peer] + (local_ptr - heap_bases[my_pe])``.

    Entries are 0 for PEs whose memory is not directly accessible from
    the local PE (e.g. on the RO backend).  Device kernels using this
    array must check for the zero entry or use the rocSHMEM device-side
    bitcode wrappers (``rocshmem_putmem_wrapper`` etc.) instead of
    arithmetic translation when targeting RO.

    Parameters
    ----------
    tensor:
        A symmetric tensor allocated by :func:`create_tensor` (must have
        ``__symm_tensor__`` set).
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for rocshmem4py.interop.torch.")

    if not getattr(tensor, "__symm_tensor__", False):
        raise ValueError(
            "tensor is not a symmetric tensor (missing __symm_tensor__ "
            "attribute). Use create_tensor() to allocate it."
        )

    bases = rocshmem4py.rocshmem_get_heap_bases(tensor.data_ptr())
    return torch.tensor(bases, dtype=torch.int64, device="cuda")


# ---------------------------------------------------------------------------
# Tensor-aware team-scoped collectives
# ---------------------------------------------------------------------------

def alltoall(team: int, dst: 'torch.Tensor', src: 'torch.Tensor',
             stream: Optional['torch.cuda.Stream'] = None) -> None:
    """Stream-ordered all-to-all over a team.

    Each PE in *team* sends ``src.nbytes / team_size`` bytes to every
    other PE; the layout is concatenation of per-PE chunks in PE-order.

    Parameters
    ----------
    team:
        rocshmem team handle (intptr_t).  Use ``ROCSHMEM_TEAM_WORLD``
        for the world team.
    dst:
        Destination tensor on the local PE (symmetric); must hold
        ``team_size`` chunks.
    src:
        Source tensor on the local PE (symmetric); must match
        ``dst.nbytes``.
    stream:
        Defaults to ``torch.cuda.current_stream()``.
    """
    if dst.nbytes != src.nbytes:
        raise ValueError(
            f"dst.nbytes ({dst.nbytes}) must match src.nbytes ({src.nbytes})"
        )
    team_size = rocshmem4py.rocshmem_team_n_pes(team)
    if team_size <= 0:
        raise ValueError(f"team has invalid size: {team_size}")
    if src.nbytes % team_size != 0:
        raise ValueError(
            f"src.nbytes ({src.nbytes}) must be divisible by team size "
            f"({team_size})"
        )
    rocshmem4py.rocshmem_alltoallmem_on_stream(
        team, dst.data_ptr(), src.data_ptr(), src.nbytes // team_size,
        _stream_handle(stream),
    )


def broadcast(team: int, dst: 'torch.Tensor', src: 'torch.Tensor',
              pe_root: int, stream: Optional['torch.cuda.Stream'] = None) -> None:
    """Stream-ordered broadcast over a team.

    On the root PE the contents of *src* are copied to *dst* on every
    other PE in *team*.  ``pe_root`` is in the team's PE space, not the
    world PE space.

    Parameters
    ----------
    team:
        rocshmem team handle (intptr_t).
    dst:
        Destination tensor on the local PE (symmetric).
    src:
        Source tensor on the local PE (symmetric); only its contents on
        the root PE matter.
    pe_root:
        Root PE index in *team*'s PE space.
    stream:
        Defaults to ``torch.cuda.current_stream()``.
    """
    if dst.nbytes != src.nbytes:
        raise ValueError(
            f"dst.nbytes ({dst.nbytes}) must match src.nbytes ({src.nbytes})"
        )
    rocshmem4py.rocshmem_broadcastmem_on_stream(
        team, dst.data_ptr(), src.data_ptr(), src.nbytes, pe_root,
        _stream_handle(stream),
    )
