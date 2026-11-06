# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

"""Configuration wrapper for ``ncclM2nConfig_t``."""

from __future__ import annotations

from dataclasses import dataclass

from nccl._extensions.bindings import nccl_m2n as _m2n_bindings


@dataclass(frozen=True)
class Config:
    """Pythonic configuration for :meth:`nccl.m2n.Handle.create`.

    The low-level ``nccl._extensions.bindings.nccl_m2n.Config`` class remains the raw
    C ABI wrapper.  This high-level dataclass intentionally exposes Python
    naming and ``None`` defaults while converting to the binding object at the
    call site. ``max_cta=None`` forwards ``NCCL_M2N_CONFIG_UNDEF_INT`` so the
    library default or ``NCCL_RESHARD_NUM_CTAS`` environment override applies.
    """

    max_cta: int | None = None

    def __post_init__(self) -> None:
        if self.max_cta is not None and self.max_cta <= 0:
            raise ValueError(f"max_cta must be positive or None, got {self.max_cta}")

    def to_binding(self) -> _m2n_bindings.Config:
        config = _m2n_bindings.Config()
        if self.max_cta is not None:
            config.maxCta = int(self.max_cta)
        return config


__all__ = ["Config"]
