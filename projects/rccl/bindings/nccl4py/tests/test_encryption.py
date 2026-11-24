# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nccl.core.set_encryption`` configures RCCL's socket encryption (NCCL 2.32.6).

The public ``nccl.core`` API (``NCCLEncryptionConfig``, ``NcclEncryptionMode``,
``set_encryption``) materializes an ``ncclEncryptionConfig_t`` and passes it to
``ncclSetEncryption`` in ``librccl.so``. RCCL validates the config there: a PSK
shorter than 32 bytes, or PSK mode without a key, is ``ncclInvalidArgument``, so
those errors prove the facade's fields reached the library. No GPU is needed::

    NCCL_LIBRARY=<build>/librccl.so pytest tests/test_encryption.py

Imports stay inside the tests so the low-level control still runs where
``nccl.core`` has no encryption API.
"""

import sys
import types

import pytest

try:
    import hip  # noqa: F401
except Exception:
    # ``import nccl`` loads the cuda.core shim, which only binds ``hip.hip`` at
    # import time; ncclSetEncryption needs no HIP call, so a stub is enough.
    _hip = types.ModuleType("hip")
    _hip.hip = types.ModuleType("hip.hip")
    sys.modules["hip"] = _hip
    sys.modules["hip.hip"] = _hip.hip

_PSK = "k" * 32  # ncclPskMinBytes


def _assert_invalid_argument(call):
    import nccl.bindings as b

    with pytest.raises(b.NCCLError) as exc:
        call()
    assert exc.value.status == b.Result.InvalidArgument


def test_bindings_set_encryption_reaches_rccl():
    """Control: the low-level binding reaches ncclSetEncryption."""
    import nccl.bindings as b

    cfg = b.EncryptionConfig()
    cfg.mode = b.EncryptionMode.PSK
    cfg.psk = "short"
    _assert_invalid_argument(lambda: b.set_encryption(cfg))
    cfg.psk = _PSK
    b.set_encryption(cfg)
    b.set_encryption(0)


def test_core_set_encryption():
    import nccl.bindings as b
    import nccl.core as nccl
    from nccl.core import NCCLEncryptionConfig, NcclEncryptionMode, set_encryption

    for name in ("NCCLEncryptionConfig", "NcclEncryptionMode", "set_encryption"):
        assert name in nccl.__all__, name
    assert int(NcclEncryptionMode.NONE) == int(b.EncryptionMode.NONE)
    assert int(NcclEncryptionMode.PSK) == int(b.EncryptionMode.PSK)

    cfg = NCCLEncryptionConfig(mode=NcclEncryptionMode.PSK, psk=_PSK)
    lowpp = cfg._to_lowpp()
    assert lowpp.mode == NcclEncryptionMode.PSK
    assert lowpp.psk == _PSK

    try:
        assert set_encryption(cfg) is None
        # RCCL rejects these, so mode and psk must have reached ncclSetEncryption.
        _assert_invalid_argument(
            lambda: set_encryption(NCCLEncryptionConfig(mode=NcclEncryptionMode.PSK, psk="short"))
        )
        _assert_invalid_argument(
            lambda: set_encryption(NCCLEncryptionConfig(mode=NcclEncryptionMode.PSK))
        )
        set_encryption(NCCLEncryptionConfig())
    finally:
        set_encryption()
