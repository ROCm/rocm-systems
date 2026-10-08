# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Windows registered inside ``nccl.core.group()`` keep their handle (nccl4py 0.5.0).

Inside a group, ``ncclCommWindowRegister`` fills the window handle only when the
group ends. nccl4py 0.4.1 saw the still-null handle and ``register_window()``
returned ``None``. Two ranks, one process and one GPU each; the module skips when
hip-python is missing or fewer than two HIP devices are visible::

    NCCL_LIBRARY=<build>/librccl.so pytest tests/test_group_window.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import pytest

pytest.importorskip("hip", reason="hip-python is required for the HIP shim")

import nccl.core as nccl  # noqa: E402  (also registers the cuda.core shim)
from cuda.core import Device, system  # noqa: E402

_NRANKS = 2
_NBYTES = 1 << 21
# Per call, so both tests together stay inside the 300 s the nccl4py build-smoke harness gives this module.
_TIMEOUT_S = 120
# The unique id goes through the environment: a child's argv is world-readable in /proc, its environment is not.
_UID_ENV = "NCCL4PY_TEST_GROUP_WINDOW_UID"

if system.get_num_devices() < _NRANKS:  # pragma: no cover - host with fewer GPUs
    pytest.skip(f"requires >= {_NRANKS} HIP devices", allow_module_level=True)


def _rank_main(rank: int, uid_hex: str, grouped: bool, out_path: str) -> None:
    Device(rank).set_current()
    uid = nccl.UniqueId.from_bytes(bytes.fromhex(uid_hex))
    comm = nccl.Communicator.init(nranks=_NRANKS, rank=rank, unique_id=uid)
    buf = nccl.mem_alloc(_NBYTES, device=rank)
    try:
        if grouped:
            with nccl.group():
                win = comm.register_window(buf)
        else:
            win = comm.register_window(buf)
        # A file, not stdout: RCCL's C-level logging shares stdout and can split lines.
        with open(out_path, "w") as f:
            json.dump({"handle": None if win is None else win.handle}, f)
    finally:
        comm.destroy()


def _register_windows(grouped: bool) -> list[int | None]:
    env = dict(os.environ, **{_UID_ENV: bytes(nccl.get_unique_id()).hex()})
    with tempfile.TemporaryDirectory() as tmp:
        outs = [os.path.join(tmp, f"rank{r}.json") for r in range(_NRANKS)]
        procs = [subprocess.Popen([sys.executable, __file__, str(r), str(int(grouped)), outs[r]], env=env)
                 for r in range(_NRANKS)]
        deadline = time.monotonic() + _TIMEOUT_S
        try:
            for p in procs:
                p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pytest.fail(f"ranks still running after {_TIMEOUT_S} s; exit codes so far: {[p.poll() for p in procs]}")
        finally:
            for p in procs:
                if p.poll() is None:
                    p.kill()
        for r, p in enumerate(procs):
            assert p.returncode == 0, f"rank {r} exited with {p.returncode}"
        handles = []
        for out in outs:
            with open(out) as f:
                handles.append(json.load(f)["handle"])
    return handles


def test_window_registered_outside_group_has_handle():
    handles = _register_windows(grouped=False)
    assert all(h not in (None, 0) for h in handles), f"window handles {handles}"


def test_window_registered_inside_group_has_handle_after_group():
    handles = _register_windows(grouped=True)
    assert None not in handles, f"register_window() inside group() returned None: {handles}"
    assert all(h != 0 for h in handles), f"window handles still null after group(): {handles}"


if __name__ == "__main__":
    _rank_main(int(sys.argv[1]), os.environ[_UID_ENV], sys.argv[2] == "1", sys.argv[3])
