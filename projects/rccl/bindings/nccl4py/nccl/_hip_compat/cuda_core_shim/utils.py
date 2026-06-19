# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cuda.core.utils`` shim re-exports.

Mirrors ``cuda.core.experimental.utils`` from cuda-core 0.3.2: just
re-exports the :class:`StridedMemoryView` class and the
:func:`args_viewable_as_strided_memory` decorator from the local
``_memoryview`` submodule.
"""

from ._memoryview import (  # noqa: F401
    StridedMemoryView,
    args_viewable_as_strided_memory,
)
