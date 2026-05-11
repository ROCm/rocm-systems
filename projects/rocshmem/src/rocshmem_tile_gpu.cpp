/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

/**
 * @file rocshmem_tile_gpu.cpp
 * @brief Wiring layer for rocSHMEM tile API functions.
 *
 * This file provides the implementation of the public rocshmem_tile_* API
 * functions by delegating to the internal Context layer.
 *
 * NOTE: The actual template implementations are now header-only in
 * rocshmem_TILE_impl.hpp. This file exists to:
 * 1. Provide a compilation unit that includes both the public API and internal headers
 * 2. Allow the template implementations to access internal context types
 * 3. Keep the build system structure consistent
 */

#include <hip/hip_runtime.h>

#include "context_incl.hpp"
#include "rocshmem/rocshmem.hpp"

// Now include the template implementations - they need context types to be defined first
#include "rocshmem/rocshmem_TILE_impl.hpp"
