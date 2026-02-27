"""One-time rocSHMEM setup and teardown for the pytest session."""

import os
import ctypes
import pytest


def _use_torch_init():
    return os.environ.get("ROCSHMEM_USE_TORCH_INIT", "1") == "1"


@pytest.fixture(scope="session", autouse=True)
def rocshmem_session():
    import rocshmem4py

    if _use_torch_init():
        rocshmem4py.init_with_torch()
    else:
        rocshmem4py.init_with_mpi()

    yield


def pytest_sessionfinish(session, exitstatus):
    import rocshmem4py

    if _use_torch_init():
        rocshmem4py.finalize_with_torch()
    else:
        try:
            hip = ctypes.CDLL("libamdhip64.so")
            hip.hipDeviceSynchronize()
        except OSError:
            pass
        rocshmem4py.rocshmem_barrier_all()
        rocshmem4py.rocshmem_finalize()
