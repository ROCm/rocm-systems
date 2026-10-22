# Installed as conftest.py next to the fake-GPU suite; only the stress plumbing lives here.
from fake_gpu_stress import (  # noqa: F401
    pytest_configure,
    pytest_runtest_logreport,
    pytest_runtest_setup,
    pytest_sessionfinish,
)
