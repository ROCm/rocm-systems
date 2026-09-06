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

#ifndef LIBRARY_SRC_GDA_QUEUE_PAIR_PROVIDER_HPP_
#define LIBRARY_SRC_GDA_QUEUE_PAIR_PROVIDER_HPP_

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)

#if   defined(GDA_MUX)
#include "queue_pair_mux.hpp"
#elif defined(GDA_IONIC)
#include "gda/ionic/queue_pair_ionic.hpp"
#elif defined(GDA_BNXT)
#include "gda/bnxt/queue_pair_bnxt.hpp"
#elif defined(GDA_MLX5)
#include "gda/mlx5/queue_pair_mlx5.hpp"
#endif

namespace rocshmem {

#if   defined(GDA_MUX)
using QueuePair = QueuePairMux;
#elif defined(GDA_IONIC)
using QueuePair = QueuePairIONIC;
#elif defined(GDA_BNXT)
using QueuePair = QueuePairBNXT;
#elif defined(GDA_MLX5)
using QueuePair = QueuePairMLX5;
#endif

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_QUEUE_PAIR_PROVIDER_HPP_
