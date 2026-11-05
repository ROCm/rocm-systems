# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

"""Constants mirrored from ``nccl_m2n.h``."""

from __future__ import annotations

from nccl._extensions.bindings import nccl_m2n as _m2n_bindings

MAX_TENSOR_DIMS = _m2n_bindings.MAX_TENSOR_DIMS
MESH_NDIMS = _m2n_bindings.MESH_NDIMS
REPLICATE = _m2n_bindings.REPLICATE


def shard(dim: int) -> int:
    """Return the placement integer for ``NCCL_RESHARD_SHARD(dim)``."""

    if dim < 0:
        raise ValueError(f"shard dimension must be non-negative, got {dim}")
    return dim


__all__ = [
    "MAX_TENSOR_DIMS",
    "MESH_NDIMS",
    "shard",
]
