# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Shared fixtures for the hipFile Python binding tests.

The whole ``hipfile`` package is unimportable without its compiled Cython
extension ``hipfile._hipfile``: ``hipfile/__init__.py`` imports names from it at
import time, and it in turn needs the hipFile C library plus the HIP runtime. To
keep the suite hermetic -- no GPU, no AIS-capable storage, no build step -- we
register a pure-Python fake ``hipfile._hipfile`` in ``sys.modules`` *before* any
test module imports ``hipfile``. Because pytest imports this file during
collection, the injection below runs first.

Why a fake (real ints + lambdas) rather than ``Mock``/``MagicMock`` for this
module-level substrate:

* **A correctly-behaving mock is just a decorated fake.** The code needs real
  ints from the enums (``IntEnum`` collapses members that share a value) and
  correctly-shaped tuples from the callables (``err[0] != 0``,
  ``ver, err = hipFileGetVersion()``). Getting those out of mocks means
  hand-configuring ``__int__`` on ~40 members and ``return_value`` on ~12
  callables -- reconstructing this fake with more ceremony and nothing for
  ``autospec`` to check.
* **A fake carries no cross-test state.** This object lives in ``sys.modules``
  for the whole session. A ``Mock`` would accumulate call history there and
  retain any per-test ``return_value``/``side_effect``, so every test would have
  to patch it just to get a clean baseline. The fake records nothing, so tests
  that don't care about a call need no setup, and tests that do use
  ``unittest.mock.patch.object`` locally (auto-reverted at block exit).
"""

import sys
import types
from pathlib import Path

import pytest

# The pure-Python ``hipfile`` package lives one directory up (``python/``).
# Put it on sys.path so ``import hipfile`` resolves without an editable install
# (which would require building the Cython extension we are deliberately faking).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- Fake extension construction ------------------------------------------

# Fixed, arbitrary version so tests can assert a known value.
_FAKE_VERSION = (1, 2, 3)

# Every hipFileOpError_t name re-exported by hipfile/enums.py, in C order.
# enums.py builds an IntEnum from these, so each MUST get a UNIQUE int -- a
# collision would make the second name an IntEnum *alias* rather than a distinct
# member, silently changing len(OpError) and membership behavior. Success is 0
# because the high-level API tests ``err[0] != 0`` for failure.
_OP_ERROR_NAMES = (
    "hipFileSuccess",
    "hipFileDriverNotInitialized",
    "hipFileDriverInvalidProps",
    "hipFileDriverUnsupportedLimit",
    "hipFileDriverVersionMismatch",
    "hipFileDriverVersionReadError",
    "hipFileDriverClosing",
    "hipFilePlatformNotSupported",
    "hipFileIONotSupported",
    "hipFileDeviceNotSupported",
    "hipFileDriverError",
    "hipFileHipDriverError",
    "hipFileHipPointerInvalid",
    "hipFileHipMemoryTypeInvalid",
    "hipFileHipPointerRangeError",
    "hipFileHipContextMismatch",
    "hipFileInvalidMappingSize",
    "hipFileInvalidMappingRange",
    "hipFileInvalidFileType",
    "hipFileInvalidFileOpenFlag",
    "hipFileDIONotSet",
    "hipFileInvalidValue",
    "hipFileMemoryAlreadyRegistered",
    "hipFileMemoryNotRegistered",
    "hipFilePermissionDenied",
    "hipFileDriverAlreadyOpen",
    "hipFileHandleNotRegistered",
    "hipFileHandleAlreadyRegistered",
    "hipFileDeviceNotFound",
    "hipFileInternalError",
    "hipFileGetNewFDFailed",
    "hipFileDriverSetupError",
    "hipFileIODisabled",
    "hipFileBatchSubmitFailed",
    "hipFileGPUMemoryPinningFailed",
    "hipFileBatchFull",
    "hipFileAsyncNotSupported",
    "hipFileIOMaxError",
)

# hipFileFileHandleType_t names re-exported by enums.py -- also need unique ints.
_HANDLE_TYPE_NAMES = (
    "hipFileHandleTypeOpaqueFD",
    "hipFileHandleTypeOpaqueWin32",
    "hipFileHandleTypeUserspaceFS",
)

# Sentinel returned by hipFileHandleRegister so tests can assert the FileHandle
# stored exactly what the extension handed back.
_FAKE_HANDLE = 0xF11E

_FAKE_PROPS = {
    "nvfs_major_version": 1,
    "nvfs_minor_version": 0,
    "nvfs_poll_thresh_size": 0,
    "nvfs_max_direct_io_size": 0,
    "nvfs_driver_status_flags": 0,
    "nvfs_driver_control_flags": 0,
    "feature_flags": 0,
    "max_device_cache_size": 0,
    "per_buffer_cache_size": 0,
    "max_device_pinned_mem_size": 0,
    "max_batch_io_count": 0,
    "max_batch_io_timeout_msecs": 0,
}


def _build_fake_hipfile():
    """Create a fake ``hipfile._hipfile`` module.

    Callable signatures mirror the real Cython wrappers so that
    ``patch.object(..., autospec=True)`` in tests enforces call arity. Defaults
    are all success-shaped; error-path tests replace individual callables.
    """
    mod = types.ModuleType("hipfile._hipfile")

    # Version constants.
    mod.VERSION_MAJOR, mod.VERSION_MINOR, mod.VERSION_PATCH = _FAKE_VERSION

    # Enum values -- unique ints, Success == 0.
    for value, name in enumerate(_OP_ERROR_NAMES):
        setattr(mod, name, value)
    for value, name in enumerate(_HANDLE_TYPE_NAMES):
        setattr(mod, name, value)

    # hipFileSuccess is set dynamically via setattr above, so pylint can't see it.
    _success = (mod.hipFileSuccess, 0)  # pylint: disable=no-member

    # Driver lifecycle.
    mod.hipFileDriverOpen = lambda: _success
    mod.hipFileDriverClose = lambda: _success
    mod.hipFileUseCount = lambda: 0

    # Version / properties.
    mod.hipFileGetVersion = lambda: (_FAKE_VERSION, _success)
    mod.hipFileDriverGetProperties = lambda: (dict(_FAKE_PROPS), _success)

    # File handles.
    mod.hipFileHandleRegister = lambda handle_value, handle_type: (
        _FAKE_HANDLE,
        _success,
    )
    mod.hipFileHandleDeregister = lambda handle: None

    # Buffer registration.
    mod.hipFileBufRegister = lambda buffer_base, length, flags=0: _success
    mod.hipFileBufDeregister = lambda buffer_base: _success

    # Synchronous I/O -- default: full transfer, no error.
    mod.hipFileRead = lambda handle, buffer_base, size, file_offset, buffer_offset: (
        size,
        0,
    )
    mod.hipFileWrite = lambda handle, buffer_base, size, file_offset, buffer_offset: (
        size,
        0,
    )

    # Error strings.
    mod.hipFileGetOpErrorString = lambda status: f"fake-op-error-{status}"

    return mod


# Inject BEFORE any test imports hipfile. The ``not in sys.modules`` guard only
# avoids clobbering the module if something imported it first.
if "hipfile._hipfile" not in sys.modules:
    sys.modules["hipfile._hipfile"] = _build_fake_hipfile()


# --- Fixtures --------------------------------------------------------------


@pytest.fixture
def fake_hipfile():
    """The injected fake ``hipfile._hipfile`` module."""
    return sys.modules["hipfile._hipfile"]


@pytest.fixture
def fake_handle():
    """The opaque handle value returned by the fake ``hipFileHandleRegister``."""
    return _FAKE_HANDLE


@pytest.fixture
def fake_version():
    """The ``(major, minor, patch)`` tuple the fake reports."""
    return _FAKE_VERSION


class _FakeVoidP:  # pylint: disable=too-few-public-methods
    """Minimal stand-in for ``ctypes.c_void_p`` (only ``.value`` is used)."""

    def __init__(self, value):
        self.value = value


@pytest.fixture
def fake_void_p():
    """Factory for ``ctypes.c_void_p``-like objects (Buffer.from_ctypes_void_p)."""
    return _FakeVoidP


class _FakeBuffer:  # pylint: disable=too-few-public-methods
    """Minimal Buffer stand-in exposing ``.ptr`` for FileHandle read/write."""

    def __init__(self, ptr=0x1000):
        self.ptr = ptr


@pytest.fixture
def fake_buffer():
    """A Buffer-like object with a ``.ptr`` attribute."""
    return _FakeBuffer()
