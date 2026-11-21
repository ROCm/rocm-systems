# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

"""Placement helpers for ``ncclDistTensor_t.placements``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from nccl.m2n.constants import MESH_NDIMS, REPLICATE, shard


@dataclass(frozen=True)
class Replicate:
    """Replicate the tensor slice along a mesh axis."""


@dataclass(frozen=True)
class Shard:
    """Shard the tensor along ``dim`` on a mesh axis."""

    dim: int

    def __post_init__(self) -> None:
        if self.dim < 0:
            raise ValueError(f"Shard dim must be non-negative, got {self.dim}")


def _placement_to_int(placement: object) -> int:
    if isinstance(placement, Replicate):
        return REPLICATE
    if isinstance(placement, Shard):
        return shard(placement.dim)
    if isinstance(placement, bool):
        raise TypeError("boolean placement is invalid; use Replicate() or Shard(dim)")
    if isinstance(placement, int):
        if placement < 0:
            raise ValueError(
                f"negative placement integer {placement} is ambiguous; "
                "use Replicate() for replication"
            )
        return shard(placement)

    cls_name = placement.__class__.__name__
    if cls_name == "Replicate":
        return REPLICATE
    if cls_name == "Shard":
        dim = int(placement.dim)
        if dim < 0:
            raise ValueError(f"Shard dim must be non-negative, got {dim}")
        return shard(dim)
    raise TypeError(
        "placements must contain nccl.m2n Replicate/Shard, PyTorch "
        f"Replicate/Shard, or placement integers; got {cls_name}"
    )


def normalize_placements(placements: Iterable[object]) -> tuple[int, ...]:
    values = tuple(_placement_to_int(p) for p in placements)
    if not 1 <= len(values) <= MESH_NDIMS:
        raise ValueError(f"expected 1..{MESH_NDIMS} placements, got {len(values)}")
    return values + (REPLICATE,) * (MESH_NDIMS - len(values))


__all__ = ["Replicate", "Shard", "normalize_placements"]
