"""rocshmem4py: Python bindings for rocSHMEM.

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License. See LICENSE for details.
"""

import os
import ctypes
from typing import Sequence, Optional, Any

__version__ = "0.1.0"
__author__ = "Advanced Micro Devices, Inc."

try:
    from _rocshmem4py import (
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
        rocshmem_int_atomic_fetch_add,
        rocshmem_long_atomic_fetch_add,
        rocshmem_int_atomic_compare_swap,
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

ROCSHMEM_TEAM_INVALID = -1
ROCSHMEM_TEAM_WORLD = 0

__all__ = [
    '__version__',
    'rocshmem_init',
    'rocshmem_finalize',
    'rocshmem_hipmodule_init',
    'rocshmem_my_pe',
    'rocshmem_n_pes',
    'rocshmem_team_my_pe',
    'rocshmem_team_n_pes',
    'rocshmem_malloc',
    'rocshmem_free',
    'rocshmem_ptr',
    'rocshmem_barrier_all',
    'rocshmem_barrier_all_on_stream',
    'rocshmem_fence',
    'rocshmem_quiet',
    'rocshmem_get_uniqueid',
    'rocshmem_init_attr',
    'rocshmem_putmem',
    'rocshmem_getmem',
    'rocshmem_putmem_nbi',
    'rocshmem_getmem_nbi',
    'rocshmem_putmem_on_stream',
    'rocshmem_getmem_on_stream',
    'rocshmem_putmem_signal_on_stream',
    'rocshmem_signal_wait_until_on_stream',
    'rocshmem_int_atomic_fetch_add',
    'rocshmem_long_atomic_fetch_add',
    'rocshmem_int_atomic_compare_swap',
    'ROCSHMEM_SUCCESS',
    'ROCSHMEM_SIGNAL_SET',
    'ROCSHMEM_SIGNAL_ADD',
    'ROCSHMEM_CMP_EQ',
    'ROCSHMEM_CMP_NE',
    'ROCSHMEM_CMP_GT',
    'ROCSHMEM_CMP_GE',
    'ROCSHMEM_CMP_LT',
    'ROCSHMEM_CMP_LE',
    'ROCSHMEM_TEAM_INVALID',
    'ROCSHMEM_TEAM_WORLD',
    'SymmetricBuffer',
    'rocshmem_create_tensor',
    'symm_rocshmem_tensor',
    'rocshmem_create_tensor_list_intra_node',
    'init_rocshmem_by_uniqueid',
    'init_with_mpi',
    'init_with_torch',
    'finalize_with_torch',
]


def _set_hip_device_from_env():
    """Set HIP device based on LOCAL_RANK or OMPI_COMM_WORLD_LOCAL_RANK."""
    local_rank = os.environ.get("LOCAL_RANK") or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
    if local_rank is not None:
        try:
            hip = ctypes.CDLL("libamdhip64.so")
            hip.hipSetDevice(int(local_rank))
        except OSError:
            pass


class SymmetricBuffer:
    """RAII wrapper for rocshmem_malloc with ``__cuda_array_interface__``
    so the buffer can be used with PyTorch tensors via ``torch.as_tensor``.

    Matches Triton-distributed's ``SymmRocShmemBuffer`` API.
    """

    def __init__(self, size: int, *, ptr: Optional[int] = None,
                 dtype=None, own_data: bool = True):
        if ptr is not None:
            self.ptr = ptr
            self.nbytes = size
        else:
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


def rocshmem_create_tensor(shape: Sequence[int], dtype) -> 'torch.Tensor':
    """Allocate symmetric memory and return it as a PyTorch tensor.

    **IMPORTANT**: This is a collective operation - all PEs must call this                                                                                                                 
    function with the same arguments, or the program will hang.

    Sets ``__symm_tensor__ = True`` on the returned tensor.
    """
    import torch

    nbytes = torch.Size(shape).numel() * dtype.itemsize
    torch.cuda.synchronize()
    ptr = rocshmem_malloc(nbytes)
    buf = SymmetricBuffer(nbytes, ptr=ptr, dtype=dtype, own_data=True)
    t = torch.as_tensor(buf, device="cuda").view(dtype).view(shape)
    setattr(t, "__symm_tensor__", True)
    return t


def symm_rocshmem_tensor(tensor: 'torch.Tensor', peer: int) -> 'torch.Tensor':
    """Return a tensor viewing the symmetric buffer on *peer*."""
    import torch

    if not getattr(tensor, "__symm_tensor__", False):
        raise ValueError("tensor is not a symm_tensor")
    if not tensor.is_cuda:
        raise ValueError("tensor must be on a CUDA device")

    if peer == rocshmem_my_pe():
        return tensor

    ptr = rocshmem_ptr(tensor.data_ptr(), peer)
    if ptr == 0:
        raise RuntimeError(
            f"rocshmem_ptr returned NULL for peer {peer} — remote direct access "
            "not supported by the current backend"
        )
    buf = SymmetricBuffer(tensor.nbytes, ptr=ptr, dtype=tensor.dtype,
                          own_data=False)
    return torch.as_tensor(buf, device="cuda").view(tensor.dtype).view(tensor.shape)


def rocshmem_create_tensor_list_intra_node(
        shape: Sequence[int], dtype) -> 'list[torch.Tensor]':
    """Allocate a symmetric tensor and return views for every PE on the node."""
    t = rocshmem_create_tensor(shape, dtype)
    return [symm_rocshmem_tensor(t, i) for i in range(rocshmem_n_pes())]


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
    """Initialize rocSHMEM using mpi4py for coordination."""
    try:
        from mpi4py import MPI
    except ImportError:
        raise ImportError("mpi4py is required for init_with_mpi().")

    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD

    _set_hip_device_from_env()

    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    unique_id = rocshmem_get_uniqueid() if rank == 0 else None
    unique_id = mpi_comm.bcast(unique_id, root=0)

    rocshmem_init_attr(rank, size, unique_id)
    mpi_comm.Barrier()

    global _rocshmem_initialized
    _rocshmem_initialized = True


def init_with_torch(group: Optional[Any] = None,
                    backend: str = 'cpu:gloo,cuda:nccl',
                    init_method: Optional[str] = None):
    """Initialize rocSHMEM using torch.distributed for coordination.

    The RO backend requires MPI-launched processes (``mpirun``).
    OpenMPI env vars are auto-mapped to torch.distributed equivalents.
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
    """Map OMPI_COMM_WORLD_* env vars to RANK/LOCAL_RANK/WORLD_SIZE."""
    mapping = {
        "RANK": "OMPI_COMM_WORLD_RANK",
        "LOCAL_RANK": "OMPI_COMM_WORLD_LOCAL_RANK",
        "WORLD_SIZE": "OMPI_COMM_WORLD_SIZE",
        "LOCAL_WORLD_SIZE": "OMPI_COMM_WORLD_LOCAL_SIZE",
    }
    for torch_var, mpi_var in mapping.items():
        if torch_var not in os.environ and mpi_var in os.environ:
            os.environ[torch_var] = os.environ[mpi_var]

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")


_rocshmem_initialized = False


def finalize_with_torch():
    """Synchronized teardown: sync -> barrier_all -> dist.barrier -> finalize.
    The barriers ensure all ranks have completed their work before any
    rank enters ``rocshmem_finalize()``, preventing the segfault that
    occurs when one rank tears down while another is still communicating.
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
    rocshmem_barrier_all()
    if dist.is_initialized():
        dist.barrier()
    rocshmem_finalize()
    _rocshmem_initialized = False
    if dist.is_initialized():
        dist.destroy_process_group()
