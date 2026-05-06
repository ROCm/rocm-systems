"""One-time rocSHMEM setup and teardown for the pytest session.

Shared helpers exported to all test modules:
  - ``_world_size()``    : reads WORLD_SIZE / OMPI_COMM_WORLD_SIZE at collection time.
  - ``requires_torch``   : skip marker for tests that need PyTorch.
  - ``requires_multi_pe``: skip marker for tests that need at least 2 PEs.

Init priority
  1. torch   → ``init_with_torch()``  (torch.distributed rendezvous)
  2. mpi4py  → ``init_with_mpi()``    (Python MPI wrapper)
  3. neither → ``rocshmem_init()``    (rocSHMEM calls MPI_Init internally)
"""

import os
import ctypes
import pytest


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _mpi4py_available() -> bool:
    try:
        from mpi4py import MPI  # noqa: F401
        return True
    except ImportError:
        return False


def _use_torch_init() -> bool:
    """Use torch init if torch is available AND not explicitly disabled.

    ROCSHMEM_USE_TORCH_INIT=0 forces the mpi4py/raw path regardless of
    torch presence.  When torch is absent the function always returns False.
    """
    if not _torch_available():
        return False
    return os.environ.get("ROCSHMEM_USE_TORCH_INIT", "1") == "1"


def _world_size() -> int:
    return int(
        os.environ.get("WORLD_SIZE")
        or os.environ.get("OMPI_COMM_WORLD_SIZE")
        or 1
    )


requires_torch = pytest.mark.skipif(
    not _torch_available(), reason="torch not installed"
)

requires_multi_pe = pytest.mark.skipif(
    _world_size() < 2, reason="Requires at least 2 PEs"
)


# Tracks whether the session-scoped fixture successfully initialized rocSHMEM.
# pytest_sessionfinish must not call into the library if init never ran —
# an unguarded barrier_all / finalize on an uninitialized context segfaults
# and masks the real collection error in CI logs.
_rocshmem_initialized = False


@pytest.fixture(scope="session", autouse=True)
def rocshmem_session():
    import rocshmem4py

    global _rocshmem_initialized
    if _use_torch_init():
        rocshmem4py.init_with_torch()
    elif _mpi4py_available():
        rocshmem4py.init_with_mpi()
    else:
        # Neither torch nor mpi4py available (e.g. bare CI image).
        # rocshmem_init() calls MPI_Init internally — works under mpirun
        # with no Python MPI wrapper required.
        local_rank = (
            os.environ.get("LOCAL_RANK")
            or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        )
        if local_rank is not None:
            try:
                hip = ctypes.CDLL("libamdhip64.so")
                hip.hipSetDevice(int(local_rank))
            except OSError:
                pass
        rocshmem4py.rocshmem_init()
    _rocshmem_initialized = True

    yield


def pytest_sessionfinish(session, exitstatus):
    if not _rocshmem_initialized:
        return

    import rocshmem4py

    if _use_torch_init():
        rocshmem4py.finalize_with_torch()
    else:
        # Sync device before the collective barrier so no rank enters
        # finalize while another is still executing GPU work.
        try:
            hip = ctypes.CDLL("libamdhip64.so")
            hip.hipDeviceSynchronize()
        except OSError:
            pass
        rocshmem4py.rocshmem_barrier_all()
        rocshmem4py.rocshmem_finalize()
