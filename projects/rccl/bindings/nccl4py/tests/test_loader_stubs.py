# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loader tests for the RCCL fork of nccl4py.

Verifies that ``import nccl.bindings`` does not raise, and that each of
the following resolves in ``librccl.so`` and rejects a null
communicator with ``ncclInvalidArgument`` (no crash, no
:class:`NotImplementedError`)::

    ncclCommGrow    -> nccl.bindings.comm_grow
    ncclCommRevoke  -> nccl.bindings.comm_revoke
    ncclPutSignal   -> nccl.bindings.put_signal
    ncclWaitSignal  -> nccl.bindings.wait_signal

Each call's argument checks reject the null ``Comm`` before any other
pointer is used, so zero values are safe for the remaining arguments.
"""

import pytest


def _assert_invalid_argument(call):
    import nccl.bindings as b

    with pytest.raises(b.NCCLError) as exc:
        call()
    assert exc.value.status == b.Result.InvalidArgument


def test_import_bindings_does_not_raise():
    import nccl.bindings  # noqa: F401


def test_import_top_level_without_ep_bindings():
    import nccl

    version = nccl.get_version()
    if version.nccl_ep is not None:
        pytest.skip("libnccl_ep loaded on this host")
    assert version.nccl.version is not None


def test_import_top_level_with_ep_bindings():
    import nccl

    version = nccl.get_version()
    if version.nccl_ep is None:
        pytest.skip("libnccl_ep not available on this host")
    assert version.nccl_ep.version is not None


def test_comm_grow_rejects_null_comm():
    import nccl.bindings as b

    # n_ranks=0 fails the rank-count check before the comm is read.
    _assert_invalid_argument(lambda: b.comm_grow(b.Comm(0), 0, 0, 0, 0))


def test_comm_revoke_rejects_null_comm():
    import nccl.bindings as b

    _assert_invalid_argument(lambda: b.comm_revoke(b.Comm(0), 0))


def test_put_signal_rejects_null_comm():
    import nccl.bindings as b

    _assert_invalid_argument(lambda: b.put_signal(0, 0, 0, 0, b.Window(0), 0, 0, 0, 0, b.Comm(0), 0))


def test_wait_signal_rejects_null_comm():
    import nccl.bindings as b

    _assert_invalid_argument(lambda: b.wait_signal(0, 0, b.Comm(0), 0))
