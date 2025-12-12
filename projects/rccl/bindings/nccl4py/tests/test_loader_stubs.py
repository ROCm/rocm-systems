# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loader / stub tests for the RCCL fork of nccl4py.

Verifies that:

* ``import nccl.bindings`` does not raise even though several
  NCCL >= 2.29 host-side symbols are absent from ``librccl.so``.
* Calling each of the following at runtime raises
  :class:`NotImplementedError` and the C symbol name appears in the
  message::

    ncclCommGrow    -> nccl.bindings.comm_grow
    ncclCommRevoke  -> nccl.bindings.comm_revoke
    ncclPutSignal   -> nccl.bindings.put_signal
    ncclWaitSignal  -> nccl.bindings.wait_signal

The Python wrappers check the ``dlsym`` slot before dereferencing any
pointer argument, so dummy zero values are sufficient for all opaque
handles.
"""

import pytest


def test_import_bindings_does_not_raise():
    import nccl.bindings  # noqa: F401


def test_comm_grow_raises_not_implemented():
    import nccl.bindings as b

    with pytest.raises(NotImplementedError, match="ncclCommGrow"):
        b.comm_grow(0, 0, 0, 0, 0)


def test_comm_revoke_raises_not_implemented():
    import nccl.bindings as b

    with pytest.raises(NotImplementedError, match="ncclCommRevoke"):
        b.comm_revoke(0, 0)


def test_put_signal_raises_not_implemented():
    import nccl.bindings as b

    with pytest.raises(NotImplementedError, match="ncclPutSignal"):
        b.put_signal(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def test_wait_signal_raises_not_implemented():
    import nccl.bindings as b

    # Signature: wait_signal(int n_desc, intptr_t signal_descs,
    #                        intptr_t comm, intptr_t stream).
    # n_desc=0 + signal_descs=0 is safe: the dlsym slot is checked before
    # the pointer is dereferenced, so the call never reaches librccl.
    with pytest.raises(NotImplementedError, match="ncclWaitSignal"):
        b.wait_signal(0, 0, 0, 0)
