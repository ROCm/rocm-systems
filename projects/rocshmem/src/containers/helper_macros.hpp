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

#ifndef LIBRARY_SRC_CONTAINERS_HELPER_MACROS_HPP_
#define LIBRARY_SRC_CONTAINERS_HELPER_MACROS_HPP_

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "rocshmem/rocshmem.hpp"

#define BARRIER() rocshmem::rocshmem_wg_barrier_all()
#define RANK rocshmem::rocshmem_my_pe()
#define NPES rocshmem::rocshmem_n_pes()

#define PE_BITS ((uint64_t)ceil(log(NPES) / log(2)))
#define PE_OF(X) ((X) >> (64 - PE_BITS))

#define _printf \
  if (RANK == 0) printf
#define _cout \
  if (RANK == 0) std::cout
#define _cerr \
  if (RANK == 0) std::cerr

#define GIBI 1073741824L
#define MEBI 1048576

#endif  // LIBRARY_SRC_CONTAINERS_HELPER_MACROS_HPP_
