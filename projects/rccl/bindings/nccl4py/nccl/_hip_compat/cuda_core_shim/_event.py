# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cuda.core.Event`` shim: only the surface ``nccl/core`` uses.

``launch_completion_event`` only needs a handle; NCCL records the event.
``record()``, ``sync()``, ``query`` and timing are not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ._hip import check_hip, hip


@dataclass
class EventOptions:
    """Options for :class:`Event` creation; only ``enable_timing`` is honored."""

    enable_timing: bool = False


class Event:
    """HIP event; create via :meth:`Device.create_event` or :meth:`Event.from_handle`."""

    __slots__ = ("_handle", "_owner")

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Event objects cannot be instantiated directly. "
            "Please use Device APIs (create_event) or other Event APIs (from_handle)."
        )

    @classmethod
    def _init(cls, options: Optional[EventOptions] = None) -> "Event":
        opts = options if options is not None else EventOptions()
        flags = int(hip.hipEventDefault) if opts.enable_timing else int(hip.hipEventDisableTiming)

        self = object.__new__(cls)
        self._owner = None
        self._handle = int(
            check_hip(
                hip.hipEventCreateWithFlags(flags),
                "hipEventCreateWithFlags",
            )
        )
        return self

    @classmethod
    def from_handle(cls, handle: int) -> "Event":
        """Wrap an existing raw event handle (no ownership transferred)."""
        self = object.__new__(cls)
        self._handle = int(handle)
        self._owner = "external"
        return self

    @property
    def handle(self) -> int:
        """Raw HIP event pointer as a Python int."""
        return self._handle

    def close(self) -> None:
        """Destroy the event if we own it; no-op for borrowed events."""
        if self._owner is None and self._handle:
            try:
                check_hip(hip.hipEventDestroy(self._handle), "hipEventDestroy")
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
        return f"<Event handle={self._handle:#x} (HIP shim)>"
