# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .nccl import *

# Hand-written wrappers for RCCL-only collectives
# (ncclAllReduceWithBias, ncclAllToAllv) that have no equivalent in the
# upstream NVIDIA nccl4py autogen we vendor. See rocm_extensions.pyx.
from .rocm_extensions import all_reduce_with_bias, all_to_all_v
