# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""NCCL parameter access.

:py:data:`params` — read-only mapping of NCCL parameter names to their values.
:py:func:`dump_params` — print parameters to stdout.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from nccl.bindings import nccl as _nccl_bindings

__all__ = ["params", "dump_params"]


class _NcclParams(Mapping[str, str]):
    """Read-only mapping of NCCL parameter names to their current values."""

    def __init__(self) -> None:
        self._keys: list[str] | None = None

    def _get_keys(self) -> list[str]:
        if self._keys is None:
            self._keys = _nccl_bindings.param_get_all_keys()
        return self._keys

    def __getitem__(self, key: str) -> str:
        return _nccl_bindings.param_get_parameter(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._get_keys())

    def __len__(self) -> int:
        return len(self._get_keys())


params: _NcclParams = _NcclParams()


def dump_params() -> None:
    """Print NCCL parameters to stdout. Set ``NCCL_PARAM_DUMP_ALL=1`` to include internal parameters."""
    _nccl_bindings.param_dump_all()
