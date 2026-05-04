# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HIP-backed ``cuda`` namespace shipped by nccl4py for ROCm.

Hosts the :mod:`cuda.core` shim that ``nccl/core/*.py`` imports. A future
ROCm-native ``hip.core`` will replace this layer, at which point this
directory can be deleted from the wheel.

Importing :mod:`cuda` alone is lightweight; only ``import cuda.core``
triggers the eager hip-python load.
"""
