# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
End-to-end exercise of the *real* hipFile Python bindings.

Every other module in this suite is hermetic: it runs against the fake
``hipfile._hipfile`` that ``conftest.py`` injects, so it verifies the Python
wrapper's contract and nothing below it. This module is the other half -- it
needs the compiled extension, a GPU, and an AIS-capable filesystem, and it
copies real bytes through device memory.

Run it with::

    python3 -m pytest --system --ais-capable-dir /mnt/ais/ext4

Without ``--system`` these tests are skipped (see ``conftest.py``); the flag
also selects the *installed* ``hipfile`` package rather than the source tree,
so what is under test is the extension the sdist actually built.
"""

import errno
import hashlib
import os
import shutil
from pathlib import Path

import pytest

from hipfile import (
    Buffer,
    Driver,
    FileHandle,
    FileHandleType,
    HipFileException,
    OpError,
    __version__,
    get_version,
)

pytestmark = pytest.mark.system

# A multiple of every plausible O_DIRECT block size, so no transfer length here
# needs rounding up. Small enough to stay well under the kernel's 2 GiB - 4 KiB
# single-transaction ceiling.
TRANSFER_SIZE = 2 * 1024 * 1024

# Read granularity for hashing; unrelated to the GPU transfer.
CHUNK_SIZE = 1024 * 1024

# errnos a filesystem raises when it cannot honour O_DIRECT at all.
_NO_O_DIRECT = (errno.EINVAL, errno.ENODEV, errno.EOPNOTSUPP, errno.ENOTSUP)

# hipFile codes meaning "this machine cannot do AIS", as opposed to a genuine
# bug in the bindings. Only these route through the skip/fail gate; anything
# else propagates as a real failure.
_UNSUPPORTED = frozenset(
    {
        OpError.DRIVER_NOT_INITIALIZED,
        OpError.PLATFORM_NOT_SUPPORTED,
        OpError.IO_NOT_SUPPORTED,
        OpError.DEVICE_NOT_SUPPORTED,
        OpError.DEVICE_NOT_FOUND,
        OpError.DRIVER_SETUP_ERROR,
        OpError.IO_DISABLED,
    }
)


def _sha256(path):
    """Return the hex SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unusable(config, reason):
    """Skip or fail out of an environment that cannot run these tests.

    Mirrors ``enforceFastpathGate()`` in ``test/system/amd/io.cpp``: an
    unusable environment is a *failure* by default and only degrades to a skip
    under ``--allow-skip-fastpath``. A silent skip is the wrong default -- in
    CI it is indistinguishable from a passing run, which is exactly the hole
    this suite exists to close. Never returns.
    """
    message = f"fastpath not available in this environment: {reason}"
    if config.getoption("--allow-skip-fastpath"):
        pytest.skip(message)
    pytest.fail(
        f"{message}\n"
        "Point --ais-capable-dir at a local NVMe-backed ext4 (mounted "
        "data=ordered) or xfs filesystem on a ROCm machine, or pass "
        "--allow-skip-fastpath to skip these tests instead of failing."
    )


def _probe_directory(directory, config):
    """Cheap precheck on the ``--ais-capable-dir`` path.

    Necessary but deliberately *not* sufficient: current tmpfs accepts
    ``O_DIRECT`` opens too, and accepting the flag says nothing about whether
    the backing device can do peer-to-peer DMA. ``_probe_driver`` is the
    authoritative check; this one just turns the two common misconfigurations
    -- a path that does not exist, and a filesystem that refuses ``O_DIRECT``
    outright -- into a message that names the actual problem.
    """
    if not directory.is_dir():
        _unusable(config, f"{directory} is not a directory")

    probe = directory / f"hipfile-pytest-probe-{os.getpid()}"
    try:
        file_descriptor = os.open(probe, os.O_RDWR | os.O_CREAT | os.O_DIRECT, 0o600)
    except OSError as exc:
        if exc.errno not in _NO_O_DIRECT:
            raise
        _unusable(config, f"{directory} does not support O_DIRECT ({exc.strerror})")
    else:
        os.close(file_descriptor)
    finally:
        probe.unlink(missing_ok=True)


def _probe_driver(config):
    """Authoritative check: the hipFile driver opens on this machine."""
    try:
        with Driver():
            pass
    except HipFileException as exc:
        if exc.hipfile_err in _UNSUPPORTED:
            _unusable(config, f"hipFile driver open failed: {exc}")
        raise
    except OSError as exc:
        _unusable(config, f"hipFile driver open failed: {exc}")


@pytest.fixture(scope="session")
def ais_driver(request):
    """Gate every system test on the hipFile driver being usable here.

    Separate from ``ais_dir`` so the tests that need a driver but no filesystem
    are gated too -- otherwise ``--allow-skip-fastpath`` would not cover them
    and they would fail with a raw exception on a machine with no GPU.
    """
    _probe_driver(request.config)


@pytest.fixture(scope="session")
def ais_dir(request, ais_driver):  # pylint: disable=W0613  # gate, not a value
    """A scratch directory on the AIS-capable filesystem, removed at teardown."""
    base = Path(request.config.getoption("--ais-capable-dir"))
    _probe_directory(base, request.config)

    scratch = base / f"hipfile-pytest-{os.getpid()}"
    scratch.mkdir(parents=True)
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


@pytest.fixture
def device_buffer(request):
    """Factory for hipMalloc'd device memory, freed even if the test fails.

    ``hipfile.hipMalloc`` does ``ctypes.CDLL("libamdhip64.so")`` at *import*
    time and is deliberately not pulled in by ``hipfile/__init__.py``. Import
    it here rather than at module scope so this file stays collectable -- and
    its tests visibly SKIPPED -- on a machine with no HIP runtime.
    """
    try:
        from hipfile.hipMalloc import (  # pylint: disable=C0415
            hipFree,
            hipMalloc,
        )
    except OSError as exc:  # HIP runtime not on the loader path
        _unusable(request.config, f"cannot load the HIP runtime: {exc}")

    allocations = []

    def allocate(size):
        buffer = hipMalloc(size)
        allocations.append(buffer)
        return buffer

    yield allocate

    for buffer in allocations:
        hipFree(buffer)


@pytest.fixture
def random_input(ais_dir):
    """A TRANSFER_SIZE file of random bytes on the AIS-capable filesystem."""
    path = ais_dir / "random_2MiB.bin"
    path.write_bytes(os.urandom(TRANSFER_SIZE))
    return path


def test_roundtrip_preserves_bytes(ais_dir, random_input, device_buffer):
    """A file copied through GPU memory comes back byte for byte."""
    output = ais_dir / "output.bin"
    buffer = device_buffer(TRANSFER_SIZE)

    with Driver():
        with Buffer.from_ctypes_void_p(buffer, TRANSFER_SIZE, 0) as registered:
            with FileHandle(
                random_input,
                os.O_RDONLY | os.O_DIRECT,
                handle_type=FileHandleType.OPAQUE_FD,
            ) as source:
                assert source.read(registered, TRANSFER_SIZE, 0, 0) == TRANSFER_SIZE
            with FileHandle(
                output, os.O_RDWR | os.O_DIRECT | os.O_CREAT | os.O_TRUNC
            ) as destination:
                written = destination.write(registered, TRANSFER_SIZE, 0, 0)
                assert written == TRANSFER_SIZE

    # The transfer length is block-aligned, so the write does not pad past EOF
    # and no ftruncate is needed to get the logical size right.
    assert output.stat().st_size == TRANSFER_SIZE
    assert _sha256(output) == _sha256(random_input)


def test_runtime_version_matches_build(ais_driver):  # pylint: disable=W0613
    """The loaded runtime library agrees with the headers we compiled against.

    ``get_version()`` asks the running driver; ``__version__`` is built from the
    ``VERSION_*`` constants baked into the extension at build time. CI installs
    the runtime and development packages separately, so a mismatch between them
    is a real and otherwise-quiet failure mode.
    """
    built = tuple(int(part) for part in __version__.split("."))
    runtime = get_version()
    assert runtime == built, (
        f"hipFile runtime reports {runtime} but the extension was compiled "
        f"against headers for {built}. The runtime and development packages are "
        f"out of step; check which libhipfile.so is on the loader path."
    )


def test_driver_use_count_balances(ais_driver):  # pylint: disable=W0613
    """The context manager leaves the driver refcount where it found it."""
    before = Driver.use_count()
    with Driver() as driver:
        assert driver.use_count() == before + 1
    assert Driver.use_count() == before
