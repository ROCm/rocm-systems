"""One-time rocSHMEM setup and teardown for the pytest session.

Shared helpers exported to all test modules:
  - ``_world_size()``    : reads WORLD_SIZE / OMPI_COMM_WORLD_SIZE at collection time.
  - ``requires_torch``   : skip marker for tests that need PyTorch.
  - ``requires_multi_pe``: skip marker for tests that need at least 2 PEs.

Init selection (first match wins):
  1. torch installed (unless explicitly opted out under mpirun) -> ``init_with_torch()``
  2. mpirun + mpi4py available                                  -> ``init_with_mpi()``
  3. mpirun without mpi4py                                      -> ``rocshmem_init()``

Set ``ROCSHMEM_USE_TORCH_INIT=0`` under mpirun to force the mpi4py path
(useful for exercising the MPI bootstrap when torch is also installed).
Outside mpirun the env var is ignored, since ``init_with_mpi()`` requires
``MPI.COMM_WORLD`` to span all PEs and raises otherwise.
"""

import os
import pytest


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


_HAS_TORCH = _try_import("torch")
_HAS_MPI4PY = _try_import("mpi4py")
_UNDER_MPIRUN = "OMPI_COMM_WORLD_SIZE" in os.environ
_USE_TORCH = _HAS_TORCH and (
    os.environ.get("ROCSHMEM_USE_TORCH_INIT", "1") == "1" or not _UNDER_MPIRUN
)
_USE_MPI = not _USE_TORCH and _UNDER_MPIRUN and _HAS_MPI4PY


def _world_size() -> int:
    return int(
        os.environ.get("WORLD_SIZE")
        or os.environ.get("OMPI_COMM_WORLD_SIZE")
        or 1
    )


requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
requires_multi_pe = pytest.mark.skipif(
    _world_size() < 2, reason="Requires at least 2 PEs"
)


# Tracks whether the session-scoped fixture successfully initialized rocSHMEM.
# pytest_sessionfinish must not call into the library if init never ran --
# an unguarded barrier_all / finalize on an uninitialized context segfaults
# and masks the real collection error in CI logs.
_rocshmem_initialized = False


@pytest.fixture(scope="session", autouse=True)
def rocshmem_session():
    import rocshmem4py

    global _rocshmem_initialized
    if _USE_TORCH:
        rocshmem4py.init_with_torch()
    elif _USE_MPI:
        rocshmem4py.init_with_mpi()
    else:
        # Bare CI image or mpirun without mpi4py.  Pin the HIP device from
        # LOCAL_RANK before rocshmem_init() so each PE owns a distinct GPU --
        # the torch / mpi4py helpers already do this internally.
        rocshmem4py.set_hip_device_from_env()
        rocshmem4py.rocshmem_init()
    _rocshmem_initialized = True
    yield


def pytest_sessionfinish(session, exitstatus):
    if not _rocshmem_initialized:
        return

    import rocshmem4py

    if _USE_TORCH:
        rocshmem4py.finalize_with_torch()
    elif _USE_MPI:
        rocshmem4py.finalize_with_mpi()
    else:
        # Sync device before the collective barrier so no rank enters
        # finalize while another is still executing GPU work.
        rocshmem4py.hip_device_synchronize()
        rocshmem4py.rocshmem_barrier_all()
        rocshmem4py.rocshmem_finalize()
