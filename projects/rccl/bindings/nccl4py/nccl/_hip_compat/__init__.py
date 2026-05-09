# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal compatibility layer for ROCm. Not a public API.

Hosts the cuda.core HIP shim used by nccl.core when running on ROCm,
so that ``from cuda.core import Device, Stream, Buffer, ...`` resolves
to a local minimal port of cuda-core 0.3.2 instead of pulling the
upstream NVIDIA package.
"""
