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
test module imports ``hipfile``. ``pytest_configure`` below is that window: it
runs after option parsing and before any test module is imported.

The suite therefore has two mutually exclusive modes, selected by ``--system``:

* **hermetic (default)** -- resolve ``hipfile`` from the source tree one
  directory up and fake the extension. Runs anywhere; every ``system``-marked
  test is skipped.
* **``--system``** -- inject nothing and touch nothing, so ``import hipfile``
  finds the *installed* package with its real compiled extension. Only
  ``system``-marked tests run; they need a GPU and an AIS-capable filesystem
  (see ``--ais-capable-dir``).

The modes cannot share a process: the fake lives in ``sys.modules`` for the
whole session, so a mixed run would be collection-order dependent. That is why
the flag deselects the other half of the suite rather than merely adding to it.

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


class _FakeAsyncIOHandle:  # pylint: disable=too-few-public-methods
    """Stand-in for the real ``AsyncIOHandle`` cdef class.

    The real class owns C storage the driver fills in when the async op
    completes; here we just mirror its constructor and attribute surface
    (``bytes_done`` read-only, ``size`` / ``file_offset`` / ``buffer_offset``
    read-write) so ``file.py``'s ``read_async`` / ``write_async`` construct and
    inspect it without a compiled extension.
    """

    def __init__(self, size, file_offset, buffer_offset):
        self.size = size
        self.file_offset = file_offset
        self.buffer_offset = buffer_offset
        self.bytes_done = 0


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

    # Asynchronous I/O -- default: full transfer, no error.
    mod.AsyncIOHandle = _FakeAsyncIOHandle
    mod.hipFileReadAsync = lambda handle, buffer_base, io, stream_handle: (0, 0)
    mod.hipFileWriteAsync = lambda handle, buffer_base, io, stream_handle: (0, 0)
    mod.hipFileStreamRegister = lambda stream_handle, flags=0: (0, 0)
    mod.hipFileStreamDeregister = lambda stream_handle: (0, 0)
    mod.supports_async = lambda: True

    # Error strings.
    mod.hipFileGetOpErrorString = lambda status: f"fake-op-error-{status}"

    return mod


# --- Mode selection --------------------------------------------------------
#
# Flag names and defaults mirror the GTest system suite so the two suites are
# driven the same way: see test/common/test-options.h (--ais-capable-dir,
# default /tmp, fed from the CMake AIS_CAPABLE_DIR cache variable) and
# test/system/amd/io.cpp (--allow-skip-fastpath).


def pytest_addoption(parser):
    """Register the system-test options."""
    group = parser.getgroup("hipfile")
    group.addoption(
        "--system",
        action="store_true",
        default=False,
        help=(
            "Run the system tests against the real installed hipfile extension "
            "instead of the hermetic fake. Requires a GPU and an AIS-capable "
            "filesystem. Deselects the hermetic unit tests."
        ),
    )
    group.addoption(
        "--ais-capable-dir",
        default="/tmp",
        help=(
            "Directory on an AIS-capable filesystem to do system-test IO in "
            "(default: /tmp). Matches the GTest suite's --ais-capable-dir."
        ),
    )
    group.addoption(
        "--allow-skip-fastpath",
        action="store_true",
        default=False,
        help=(
            "Skip rather than fail the system tests when the AIS fastpath is "
            "unavailable. Matches HIPFILE_ALLOW_SKIP_FASTPATH_TESTS."
        ),
    )


def pytest_configure(config):
    """Install the fake extension unless we were asked for the real one.

    Runs after option parsing and before any test module is imported, which is
    the only window in which ``sys.modules`` can still be primed.
    """
    config.addinivalue_line(
        "markers",
        "system: needs a real GPU and an AIS-capable filesystem; run with --system",
    )

    if config.getoption("--system"):
        # Leave sys.path and sys.modules alone so ``import hipfile`` finds the
        # installed package and its compiled extension -- but check that it is
        # really there, and really compiled, before collection starts. A bare
        # ModuleNotFoundError from every module is a poor error message, and a
        # fake reaching a hardware test would pass silently, which is the exact
        # failure this mode exists to rule out.
        try:
            from hipfile import _hipfile as extension  # pylint: disable=C0415
        except ImportError as exc:
            raise pytest.UsageError(
                f"--system needs the hipfile package installed with its "
                f"compiled extension, but importing it failed: {exc}"
            ) from exc
        origin = getattr(extension, "__file__", None) or "<none>"
        if not origin.endswith((".so", ".pyd")):
            raise pytest.UsageError(
                f"--system loaded hipfile._hipfile from {origin}, which is not "
                f"a compiled extension. Refusing to run hardware tests against "
                f"a stub."
            )
        return

    # The pure-Python ``hipfile`` package lives one directory up (``python/``).
    # Put it on sys.path so ``import hipfile`` resolves without an editable
    # install (which would require building the extension we are faking).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    # setdefault, not assignment: don't clobber the module if something
    # imported it first.
    sys.modules.setdefault("hipfile._hipfile", _build_fake_hipfile())


def pytest_ignore_collect(collection_path, config):
    """Under ``--system``, don't even import the hermetic ``test_*`` modules.

    Deselecting them after collection is too late: collection *imports* them,
    and in this mode ``import hipfile`` resolves to the installed package, so
    six modules would be imported against the real extension purely to be
    skipped. The file-name prefix ``test_system_`` is the selector.

    The opposite direction is deliberately not handled here -- the system
    modules are safe to import without hardware, so letting them collect in the
    default mode means their tests report as SKIPPED rather than vanishing.
    """
    if not config.getoption("--system"):
        return None
    if collection_path.suffix == ".py" and collection_path.name.startswith("test_"):
        return not collection_path.name.startswith("test_system_")
    return None


def pytest_collection_modifyitems(config, items):
    """Skip the hardware tests unless ``--system`` was given."""
    if config.getoption("--system"):
        return
    skip_system = pytest.mark.skip(reason="needs hardware; run with --system")
    for item in items:
        if item.get_closest_marker("system") is not None:
            item.add_marker(skip_system)


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
