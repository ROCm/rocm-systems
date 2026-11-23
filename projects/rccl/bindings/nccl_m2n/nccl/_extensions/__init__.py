# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Internals shared by every nccl-extensions library facade (``nccl.ep``,
``nccl.m2n``, ...).

This package exists so the pieces those facades have in common -- the Cython
bindings, ``binding_dataclass``, the distribution version -- are built and
shipped once rather than duplicated per library. It also keeps every path this
distribution owns clear of nccl4py's, which owns ``nccl/__init__.py``,
``nccl/_version.py``, ``nccl/core/`` and ``nccl/bindings/`` in the same
namespace.

Nothing here is public API.
"""
