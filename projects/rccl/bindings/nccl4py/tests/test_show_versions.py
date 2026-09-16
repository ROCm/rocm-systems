# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version reporting when the optional nccl_ep bindings are absent.

``nccl.bindings.nccl_ep`` is not built on ROCm, so ``_show_versions``
imports it optionally and has to report ``libnccl_ep.so`` as missing
rather than raise.
"""

from nccl import _show_versions as sv


def test_ep_library_info_is_none_without_bindings(monkeypatch):
    # _nccl_ep_importable is the documented patch seam, so the guard cannot
    # live there: patching it would otherwise reach get_version() on None.
    monkeypatch.setattr(sv, "_ep_bindings", None)
    monkeypatch.setattr(sv, "_nccl_ep_importable", lambda: True)

    assert sv._nccl_ep_library_info() is None
