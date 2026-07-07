# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Kernel selection filtering for analyze mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd

from utils.logger import console_error


@dataclass(frozen=True)
class KernelSelectionRequest:
    """A raw, unresolved kernel selection.

    ``mode`` disambiguates the values so no downstream code has to guess:
    ``"indices"`` are integer row positions in the Top Stats table (CLI ``-k``);
    ``"names"`` are kernel-name strings (GUI or operator selection).
    """

    mode: Literal["indices", "names"]
    values: list[int] | list[str]


@dataclass(frozen=True)
class KernelFilter:
    """A resolved kernel selection, canonicalized to stripped kernel names."""

    names: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_active(self) -> bool:
        return bool(self.names)


def resolve_kernel_filter(
    request: Optional[KernelSelectionRequest],
    kernel_top_df: Optional[pd.DataFrame],
) -> KernelFilter:
    """Resolve a selection request into a :class:`KernelFilter`.

    Integer indices are validated against and mapped through ``kernel_top_df``.
    Returns an inactive filter when ``request`` is empty.
    """
    if request is None or not request.values:
        return KernelFilter()

    if request.mode == "names":
        return KernelFilter(frozenset(str(name).strip() for name in request.values))

    if kernel_top_df is None:
        console_error(
            "Kernel top stats table not loaded. Ensure create_df_kernel_top_stats() "
            "is called before resolving an index-based kernel filter."
        )

    num_kernels = len(kernel_top_df["Kernel_Name"])
    for index in request.values:
        if not 0 <= index < num_kernels:
            console_error(
                f"{index} is an invalid kernel id. "
                f"Please enter an id between 0-{num_kernels - 1}"
            )

    return KernelFilter(
        frozenset(
            str(kernel_top_df.loc[index, "Kernel_Name"]).strip()
            for index in request.values
        )
    )


def mark_selected_kernels(
    kernel_top_df: pd.DataFrame,
    kernel_filter: KernelFilter,
) -> None:
    """Set a ``Selected`` column ("*") on rows selected by ``kernel_filter``."""
    stripped_names = kernel_top_df["Kernel_Name"].apply(
        lambda name: name.strip() if isinstance(name, str) else name
    )
    kernel_top_df["Selected"] = stripped_names.isin(kernel_filter.names).map({
        True: "*",
        False: "",
    })
