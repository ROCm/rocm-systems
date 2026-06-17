# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cuda.core.Stream`` shim, ported from cuda-core 0.3.2 ``_stream.pyx``.

Implements the subset :class:`Device` needs to be functional::

    Stream.from_handle(int) / .handle / .__cuda_stream__()
    Stream._init(obj=None, options=None, device_id=None)
    Stream.close() / .__del__
    default_stream()

Stream creation via priority/options (``cuStreamCreateWithPriority``)
and ``Stream.sync()`` / ``wait_event()`` etc. are not implemented; the
:class:`Buffer` / :class:`MemoryResource` paths used by ``nccl/core``
do not require them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ._hip import check_hip, hip
from .typing import IsStreamT


@dataclass
class StreamOptions:
    """Customizable options for :class:`Stream` creation.

    Mirrors cuda-core 0.3.2's surface but only ``nonblocking`` and
    ``priority`` are honored on HIP.
    """

    nonblocking: bool = True
    priority: Optional[int] = None


def _try_to_get_stream_ptr(obj: IsStreamT) -> int:
    """Extract the raw (cudaStream_t-equivalent) pointer from an object
    that implements the __cuda_stream__ protocol."""
    try:
        attr = obj.__cuda_stream__
    except AttributeError:
        raise TypeError(
            f"{type(obj).__name__} object does not have a '__cuda_stream__' attribute"
        ) from None

    info = attr() if callable(attr) else attr
    try:
        n = len(info)
    except TypeError as e:
        raise RuntimeError(
            f"obj.__cuda_stream__ must return a sequence with 2 elements, got {type(info).__name__}"
        ) from e
    if n != 2:
        raise RuntimeError(f"obj.__cuda_stream__ must return a sequence with 2 elements, got {n}")
    if int(info[0]) != 0:
        raise RuntimeError(f"protocol version of __cuda_stream__ must be 0, got {info[0]!r}")
    return int(info[1])


class Stream:
    """HIP queue equivalent of cuda.core.Stream.

    Direct construction is forbidden (matches upstream); use
    :meth:`Stream.from_handle`, :meth:`Device.create_stream`, or
    :func:`default_stream`.
    """

    __slots__ = ("_handle", "_owner", "_builtin", "_device_id")

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Stream objects cannot be instantiated directly. "
            "Please use Device APIs (create_stream) or other Stream APIs (from_handle)."
        )

    @classmethod
    def _legacy_default(cls) -> "Stream":
        # HIP's "legacy default" stream is the NULL stream (handle == 0).
        self = object.__new__(cls)
        self._handle = 0
        self._owner = None
        self._builtin = True
        self._device_id = None
        return self

    @classmethod
    def _init(
        cls,
        obj: Optional[IsStreamT] = None,
        options: Optional[StreamOptions] = None,
        device_id: Optional[int] = None,
    ) -> "Stream":
        if obj is not None and options is not None:
            raise ValueError("obj and options cannot be both specified")

        self = object.__new__(cls)
        self._owner = None
        self._builtin = False
        self._device_id = device_id

        if obj is not None:
            self._handle = _try_to_get_stream_ptr(obj)
            self._owner = obj
            return self

        # No obj, no options: create a non-blocking stream on the current device.
        opts = options if options is not None else StreamOptions()
        flags = int(hip.hipStreamNonBlocking) if opts.nonblocking else int(hip.hipStreamDefault)

        if opts.priority is not None:
            self._handle = int(
                check_hip(
                    hip.hipStreamCreateWithPriority(flags, int(opts.priority)),
                    "hipStreamCreateWithPriority",
                )
            )
        else:
            self._handle = int(
                check_hip(
                    hip.hipStreamCreateWithFlags(flags),
                    "hipStreamCreateWithFlags",
                )
            )
        return self

    @classmethod
    def from_handle(cls, handle: int) -> "Stream":
        """Wrap an existing raw stream handle (no ownership transferred)."""
        self = object.__new__(cls)
        self._handle = int(handle)
        self._owner = "external"
        self._builtin = False
        self._device_id = None
        return self

    @property
    def handle(self) -> int:
        """Raw HIP stream pointer as a Python int."""
        return self._handle

    def __cuda_stream__(self) -> Tuple[int, int]:
        """Protocol used by cuda-core consumers: ``(version=0, ptr)``."""
        return (0, int(self._handle))

    def close(self) -> None:
        """Destroy the stream if we own it; no-op for borrowed/builtin streams."""
        if self._owner is None and not self._builtin and self._handle:
            try:
                check_hip(hip.hipStreamDestroy(self._handle), "hipStreamDestroy")
            except Exception:
                # Avoid raising during finalization
                pass
            self._handle = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"<Stream handle={self._handle:#x} (HIP shim)>"


_DEFAULT_STREAM: Optional[Stream] = None


def default_stream() -> Stream:
    """Singleton wrapper around the HIP NULL (legacy default) stream."""
    global _DEFAULT_STREAM
    if _DEFAULT_STREAM is None:
        _DEFAULT_STREAM = Stream._legacy_default()
    return _DEFAULT_STREAM
