# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Communicator-owned resource management.

This module provides resource classes for NCCL communicator-owned objects including
registered buffers for zero-copy communication, registered windows for RMA operations,
and custom reduction operators. All resources are automatically cleaned up when the
owning communicator is destroyed or aborted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nccl.bindings import nccl as _nccl_bindings

from nccl.core._binding_helpers import LowppView
from nccl.core.constants import WindowFlag
from nccl.core.team import NCCLTeam
from nccl.core.typing import NcclDataType, NcclInvalid

_PointerBox = _nccl_bindings.PointerBox

__all__ = [
    "MultimemHandle",
    "LsaBarrierHandle",
    "GinBarrierHandle",
    "LLA2AHandle",
    "RegisteredBufferHandle",
    "RegisteredWindowHandle",
    "CustomRedOp",
    "DevCommResource",
]


class MultimemHandle(LowppView, lowpp_cls=_nccl_bindings.MultimemHandle):
    """Multimem handle, returned by
    :py:meth:`~nccl.core.DevCommResource.multimem_handle` for a team requested
    with ``multimem=True``. Pass it to device-side multimem operations.
    """

    @property
    def mc_base_pointer(self) -> int:
        """Multicast base pointer written by NCCL; ``0`` means none was provided."""
        return self._lowpp.mc_base_ptr


class LsaBarrierHandle(LowppView, lowpp_cls=_nccl_bindings.LsaBarrierHandle):
    """LSA barrier handle, returned by
    :py:attr:`~nccl.core.DevCommResource.resource_handles` for each
    :py:class:`~nccl.core.LsaBarrierRequirement`. Pass it to device-side
    barrier sessions.
    """

    @property
    def buf_handle(self) -> int:
        """Buffer resource offset assigned by NCCL during device comm creation."""
        return self._lowpp.buf_handle

    @property
    def n_barriers(self) -> int:
        """Number of barriers this handle provides."""
        return self._lowpp.n_barriers


class GinBarrierHandle(LowppView, lowpp_cls=_nccl_bindings.GinBarrierHandle):
    """GIN barrier handle, returned by
    :py:attr:`~nccl.core.DevCommResource.resource_handles` for each
    :py:class:`~nccl.core.GinBarrierRequirement`. Pass it to device-side
    barrier sessions.
    """

    @property
    def signal0(self) -> int:
        """Starting GIN signal index assigned by NCCL during device comm creation."""
        return self._lowpp.signal0


class LLA2AHandle(LowppView, lowpp_cls=_nccl_bindings.LLA2AHandle):
    """Low-latency all-to-all handle, returned by
    :py:attr:`~nccl.core.DevCommResource.resource_handles` for each
    :py:class:`~nccl.core.LLA2ARequirement`. Pass it to device-side all-to-all
    sessions.
    """

    @property
    def buf_handle(self) -> int:
        """Buffer resource offset assigned by NCCL during device comm creation."""
        return self._lowpp.buf_handle

    @property
    def n_slots(self) -> int:
        """Number of slots this handle provides."""
        return self._lowpp.n_slots


class CommResource(ABC):
    """Abstract base class for NCCL communicator-owned resources.

    Resources are tied to a specific communicator. They can be released
    explicitly via :py:meth:`close`, and are released automatically when the
    owning communicator is destroyed or aborted.
    """

    def __init__(self, comm_ptr: int):
        """Initializes the resource with a communicator pointer.

        Args:
            comm_ptr: Raw NCCL communicator pointer.

        Raises:
            NcclInvalid: If comm_ptr is 0 (invalid communicator).
        """
        if comm_ptr == 0:
            raise NcclInvalid(
                "Invalid communicator: cannot create resource with communicator ptr=0"
            )
        self._comm_ptr = comm_ptr
        self._closed = False

    @abstractmethod
    def _allocate(self) -> None:
        """Allocates the underlying NCCL resource.

        Called during initialization; must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _deallocate(self) -> None:
        """Deallocates the underlying NCCL resource.

        Called during :py:meth:`close`; must be implemented by subclasses.
        """
        pass

    def close(self) -> None:
        """Explicitly deallocates the resource.

        Idempotent: safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True
        self._deallocate()

    def _check_valid(self) -> None:
        """Raises if the resource has been closed.

        Raises:
            RuntimeError: If the resource has been closed.
        """
        if self._closed:
            raise RuntimeError(f"{self.__class__.__name__} has been closed and is no longer valid")

    @property
    def is_valid(self) -> bool:
        """Whether the resource has been initialized and is still valid (not closed)."""
        return not self._closed


class RegisteredBufferHandle(CommResource):
    """NCCL registered buffer handle for zero-copy optimized communication.

    Registers a user buffer with the communicator to enable performance
    optimizations in NCCL operations. Created by
    :py:meth:`Communicator.register_buffer`. The registration handle can be
    released explicitly via :py:meth:`close`, or automatically when the
    owning communicator is destroyed or aborted.
    """

    def __init__(self, comm_ptr: int, buffer_ptr: int, size: int):
        """Creates and registers a buffer with NCCL.

        Args:
            comm_ptr: NCCL communicator raw pointer.
            buffer_ptr: Device pointer to the buffer.
            size: Size of the buffer in bytes.

        Raises:
            NcclInvalid: If comm_ptr is 0 (invalid communicator).
        """
        self._buffer_ptr = buffer_ptr
        self._size = size
        self._handle: int | None = None
        super().__init__(comm_ptr)
        self._allocate()

    def _allocate(self) -> None:
        """Registers the buffer with NCCL for zero-copy communication."""
        self._handle = _nccl_bindings.comm_register(self._comm_ptr, self._buffer_ptr, self._size)

    def _deallocate(self) -> None:
        """Deregisters the buffer from NCCL."""
        if self._handle is not None:
            _nccl_bindings.comm_deregister(self._comm_ptr, self._handle)
            self._handle = None

    @property
    def handle(self) -> int:
        """Registration handle for NCCL operations.

        Raises:
            RuntimeError: If the buffer has been deregistered or the handle
                is invalid.
        """
        self._check_valid()
        if self._handle is None:
            raise RuntimeError("Buffer registration handle is invalid")
        return self._handle

    @property
    def size(self) -> int:
        """Size of the registered buffer in bytes."""
        return self._size

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<RegisteredBufferHandle: closed>"
        return f"<RegisteredBufferHandle: size={self._size}, handle={self._handle:#x}>"


class RegisteredWindowHandle(CommResource):
    """NCCL registered window handle for Remote Memory Access (RMA) operations.

    Registers a memory window with the communicator for one-sided communication
    patterns. Created by :py:meth:`Communicator.register_window`. Registration
    is collective: all ranks must call
    :py:meth:`Communicator.register_window` with equal buffer sizes by
    default. Deregistration is local. The window handle can be released
    explicitly via :py:meth:`close`, or automatically when the owning
    communicator is destroyed or aborted.
    """

    def __init__(self, comm_ptr: int, buffer_ptr: int, size: int, flags: WindowFlag | None = None):
        """Creates and registers a memory window with NCCL.

        Args:
            comm_ptr: NCCL communicator raw pointer.
            buffer_ptr: Device pointer to the buffer.
            size: Size of the window in bytes.
            flags: Window registration flags. Defaults to ``None``
                (:py:attr:`~nccl.core.WindowFlag.DEFAULT`).

        Raises:
            NcclInvalid: If comm_ptr is 0 (invalid communicator).
        """
        self._buffer_ptr = buffer_ptr
        self._size = size
        self._flags = flags if flags is not None else WindowFlag.DEFAULT
        self._handle = _PointerBox()
        self._closed = True
        self._comm_ptr = comm_ptr
        self._allocate()
        super().__init__(comm_ptr)

    def _allocate(self) -> None:
        """Collectively registers the window with NCCL."""
        _nccl_bindings.comm_window_register(
            self._comm_ptr, self._buffer_ptr, self._size, self._handle.address, self._flags.value
        )

    def _deallocate(self) -> None:
        """Deregisters the window from NCCL.

        Deregistration is local to the rank. The caller must ensure the
        buffer is not being accessed by any NCCL operation.
        """
        if self.handle:
            _nccl_bindings.comm_window_deregister(self._comm_ptr, self.handle)
            self._handle.ptr = 0
        self._closed = True

    @property
    def is_valid(self) -> bool:
        """Whether the window is still registered (not closed, handle non-null)."""
        return not self._closed and bool(self._handle.ptr)

    @property
    def handle(self) -> int:
        """Window handle for NCCL operations, or ``0`` once deregistered."""
        return int(self._handle.ptr)

    @property
    def size(self) -> int:
        """Size of the registered window in bytes."""
        return self._size

    @property
    def user_ptr(self) -> int:
        """Original user buffer pointer registered with this window.

        Raises:
            RuntimeError: If the window has been deregistered.
        """
        self._check_valid()
        return self._buffer_ptr

    def get_lsa_multimem_device_pointer(self, offset: int = 0) -> int | None:
        """Returns the LSA multicast device pointer for this window.

        Returns a device pointer suitable for multicast operations over the
        LSA (Load/Store Accessible) team. The pointer is valid as long as
        the window and communicator remain alive.

        Args:
            offset: Byte offset within the window buffer. Defaults to 0.

        Returns:
            Device pointer as int, or ``None`` if multimem is not supported.

        Raises:
            RuntimeError: If the window has been closed.
        """
        self._check_valid()
        ptr = _nccl_bindings.get_lsa_multimem_device_pointer(self.handle, offset)
        return ptr if ptr != 0 else None

    def get_multimem_device_pointer(self, multimem: MultimemHandle, offset: int = 0) -> int | None:
        """Returns the multicast device pointer for this window and ``multimem``.

        Unlike :meth:`get_lsa_multimem_device_pointer` (which uses the LSA
        team's multimem), this resolves the pointer for an explicit multimem
        handle produced during device communicator creation.

        Args:
            multimem: A :py:class:`~nccl.core.MultimemHandle` returned by
                :py:meth:`DevCommResource.multimem_handle`.
            offset: Byte offset within the window buffer. Defaults to 0.

        Returns:
            Device pointer as int, or ``None`` if multimem is not supported.

        Raises:
            RuntimeError: If the window has been closed.
        """
        self._check_valid()
        ptr = _nccl_bindings.get_multimem_device_pointer(self.handle, offset, multimem._lowpp.ptr)
        return ptr if ptr != 0 else None

    def get_lsa_device_pointer(self, lsa_rank: int, offset: int = 0) -> int:
        """Returns the LSA device pointer for a peer within the LSA team.

        Returns a device pointer to the peer's window buffer addressable
        from the local GPU via LSA (Load/Store Accessible) mapping.

        Args:
            lsa_rank: Rank within the LSA team (0 to lsa_size - 1).
            offset: Byte offset within the window buffer. Defaults to 0.

        Returns:
            Device pointer as int.

        Raises:
            RuntimeError: If the window has been closed.
        """
        self._check_valid()
        return _nccl_bindings.get_lsa_device_pointer(self.handle, offset, lsa_rank)

    def get_peer_device_pointer(self, peer: int, offset: int = 0) -> int | None:
        """Returns a device pointer to a peer's window buffer by world rank.

        If the peer is not reachable via LSA, returns ``None``.

        Args:
            peer: World rank of the peer (0 to nranks - 1).
            offset: Byte offset within the window buffer. Defaults to 0.

        Returns:
            Device pointer as int, or ``None`` if the peer is not reachable
            via LSA.

        Raises:
            RuntimeError: If the window has been closed.
        """
        self._check_valid()
        ptr = _nccl_bindings.get_peer_device_pointer(self.handle, offset, peer)
        return ptr if ptr != 0 else None

    def __repr__(self) -> str:
        return f"<RegisteredWindowHandle: size={self._size}, handle={self.handle:#x}, flags={self._flags}>"


class CustomRedOp(CommResource):
    """NCCL user-defined custom reduction operator.

    Created by :py:meth:`Communicator.create_pre_mul_sum`. The PreMulSum
    operator performs ``output = scalar * sum(inputs)``, useful for averaging
    or weighted reductions. The operator can be released explicitly via
    :py:meth:`close`, or automatically when the owning communicator is
    destroyed or aborted.
    """

    def __init__(
        self,
        comm_ptr: int,
        scalar_ptr: int,
        datatype: NcclDataType,
        residence: _nccl_bindings.ScalarResidence,
    ):
        """Creates a custom reduction operator.

        Args:
            comm_ptr: NCCL communicator raw pointer.
            scalar_ptr: Pointer to the scalar value (host or device memory).
            datatype: NCCL data type of the scalar and reduction.
            residence: Indicates scalar memory location (HostImmediate or
                Device).

        Raises:
            NcclInvalid: If comm_ptr is 0 (invalid communicator).
        """
        self._scalar_ptr = scalar_ptr
        self._datatype = datatype
        self._residence = residence
        self._op: int | None = None

        super().__init__(comm_ptr)
        self._allocate()

    def _allocate(self) -> None:
        """Creates the custom reduction operator in NCCL."""
        self._op = _nccl_bindings.red_op_create_pre_mul_sum(
            self._scalar_ptr, int(self._datatype), int(self._residence), self._comm_ptr
        )

    def _deallocate(self) -> None:
        """Destroys the custom reduction operator in NCCL."""
        if self._op is not None:
            _nccl_bindings.red_op_destroy(self._op, self._comm_ptr)
            self._op = None

    @property
    def op(self) -> int:
        """Operator handle for use in reduction operations.

        Raises:
            RuntimeError: If the operator has been destroyed or is invalid.
        """
        self._check_valid()
        if self._op is None:
            raise RuntimeError("RedOp is invalid")
        return self._op

    def __int__(self) -> int:
        """Returns the operator handle for direct use in collective operations.

        Raises:
            RuntimeError: If the operator has been destroyed or is invalid.
        """
        return self.op

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<CustomRedOp: closed>"
        return f"<CustomRedOp: type=PreMulSum, dtype={self._datatype}, residence={self._residence.name}, op={self._op}>"


class DevCommResource(CommResource):
    """NCCL device communicator resource for device-side operations.

    Wraps ``ncclDevComm_t`` and manages its lifecycle. Created by
    :py:meth:`Communicator.create_dev_comm`. The device communicator is
    automatically destroyed when the parent communicator is destroyed or
    aborted.
    """

    def __init__(
        self,
        comm_ptr: int,
        reqs_lowpp: _nccl_bindings.DevCommRequirements,
        team_multimem_lowpp: dict[NCCLTeam, _nccl_bindings.MultimemHandle] | None = None,
        resource_handle_lowpps: tuple[
            _nccl_bindings.LsaBarrierHandle
            | _nccl_bindings.GinBarrierHandle
            | _nccl_bindings.LLA2AHandle,
            ...,
        ]
        | None = None,
    ):
        """Creates a device communicator from an existing host communicator.

        Args:
            comm_ptr: NCCL communicator raw pointer.
            reqs_lowpp: Per-create low-level requirements root. Consumed during
                construction and released once ncclDevCommCreate has copied it;
                the caller keeps its linked-list nodes alive until this
                constructor returns.
            team_multimem_lowpp: Per-team multimem output storage. The resource
                retains these objects because NCCL writes through their
                pointers and returned handle facades reference them.
            resource_handle_lowpps: Per-resource output handle lowpps in
                requirement order. The resource retains the handle storage
                because NCCL writes offsets into it during creation and device
                kernels read it afterwards.

        Raises:
            NcclInvalid: If comm_ptr is 0 (invalid communicator).
        """
        self._reqs_lowpp = reqs_lowpp
        self._team_multimem_lowpp = team_multimem_lowpp or {}
        self._resource_handle_lowpps = resource_handle_lowpps or []
        self._dev_comm: _nccl_bindings.DevComm | None = None
        super().__init__(comm_ptr)
        # reqs_lowpp is only needed for the create call; drop it afterwards so we
        # don't retain a struct whose linked-list pointers dangle once the
        # caller's requirement nodes are freed.
        try:
            self._allocate()
        finally:
            del self._reqs_lowpp

    def _allocate(self) -> None:
        """Creates the device communicator via ncclDevCommCreate."""
        self._dev_comm = _nccl_bindings.dev_comm_create(self._comm_ptr, self._reqs_lowpp.ptr)

    def _deallocate(self) -> None:
        """Destroys the device communicator via ncclDevCommDestroy."""
        if self._dev_comm is not None:
            _nccl_bindings.dev_comm_destroy(self._comm_ptr, self._dev_comm.ptr)
            self._dev_comm = None

    @property
    def dev_comm(self) -> _nccl_bindings.DevComm:
        """DevComm object wrapping :c:type:`ncclDevComm_t <ncclDevComm>`.

        Raises:
            RuntimeError: If the device communicator has been destroyed.
        """
        self._check_valid()
        if self._dev_comm is None:
            raise RuntimeError("DevComm is invalid")
        return self._dev_comm

    @property
    def ptr(self) -> int:
        """Raw pointer to the underlying :c:type:`ncclDevComm_t <ncclDevComm>` structure.

        Raises:
            RuntimeError: If the device communicator has been destroyed.
        """
        return self.dev_comm.ptr

    def multimem_handle(self, team: NCCLTeam) -> MultimemHandle:
        """Returns the multimem handle requested for ``team``.

        The returned facade wraps this dev comm's per-create output storage;
        each lookup creates a new facade over the same storage. It remains
        backed by the resource until the device communicator is closed.

        Args:
            team: The team the handle was requested for, as an entry of the
                ``teams`` requirement used to create this device communicator.

        Returns:
            The :py:class:`~nccl.core.MultimemHandle` NCCL filled in for ``team``.

        Raises:
            RuntimeError: If the device communicator has been closed.
            KeyError: If ``team`` was not requested with ``multimem=True`` in
                the requirements used to create this device communicator.
        """
        self._check_valid()
        try:
            lowpp = self._team_multimem_lowpp[team]
        except KeyError:
            raise KeyError(
                f"no multimem handle for {team!r}; it was not requested with "
                "multimem=True in this device communicator's requirements"
            ) from None
        return MultimemHandle._from_lowpp(lowpp)

    @property
    def resource_handles(
        self,
    ) -> tuple[LsaBarrierHandle | GinBarrierHandle | LLA2AHandle, ...]:
        """Finalized resource handles, in the order of
        :py:attr:`~nccl.core.NCCLDevCommRequirements.resources`.

        ``resource_handles[i]`` corresponds to ``requirements.resources[i]`` and
        is an :py:class:`~nccl.core.LsaBarrierHandle`,
        :py:class:`~nccl.core.GinBarrierHandle`, or
        :py:class:`~nccl.core.LLA2AHandle` depending on the requirement. Each
        remains backed by this resource until the device communicator is closed.
        """
        self._check_valid()
        handles = []
        for lowpp in self._resource_handle_lowpps:
            if isinstance(lowpp, _nccl_bindings.LsaBarrierHandle):
                handles.append(LsaBarrierHandle._from_lowpp(lowpp))
            elif isinstance(lowpp, _nccl_bindings.GinBarrierHandle):
                handles.append(GinBarrierHandle._from_lowpp(lowpp))
            elif isinstance(lowpp, _nccl_bindings.LLA2AHandle):
                handles.append(LLA2AHandle._from_lowpp(lowpp))
            else:
                raise NcclInvalid(f"unexpected resource handle lowpp: {type(lowpp).__name__}")

        return tuple(handles)

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<DevCommResource: closed>"
        return f"<DevCommResource: ptr={self.ptr:#x}>"
