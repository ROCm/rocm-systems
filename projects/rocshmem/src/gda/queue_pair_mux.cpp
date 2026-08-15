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

#include <new>
#include <utility>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)

#include "log.hpp"
#include "queue_pair_mux.hpp"

#if defined(GDA_IONIC)
#include "gda/ionic/queue_pair_ionic.hpp"
#endif
#if defined(GDA_BNXT)
#include "gda/bnxt/queue_pair_bnxt.hpp"
#endif
#if defined(GDA_MLX5)
#include "gda/mlx5/queue_pair_mlx5.hpp"
#endif

namespace rocshmem {

#if defined(GDA_IONIC)
__host__ QueuePairMux::QueuePairMux(QueuePairIONIC&& ionic)
  : qp{std::move(ionic)} {
  provider = GDAProvider::IONIC;
}

__host__ QueuePairMux::QueuePairUnion::QueuePairUnion(QueuePairIONIC&& ionic)
  : ionic{std::move(ionic)} { }
#endif

#if defined(GDA_BNXT)
__host__ QueuePairMux::QueuePairMux(QueuePairBNXT&& bnxt)
  : qp{std::move(bnxt)} {
  provider = GDAProvider::BNXT;
}

__host__ QueuePairMux::QueuePairUnion::QueuePairUnion(QueuePairBNXT&& bnxt)
  : bnxt{std::move(bnxt)} { }
#endif

#if defined(GDA_MLX5)
__host__ QueuePairMux::QueuePairMux(QueuePairMLX5&& mlx5)
  : qp{std::move(mlx5)} {
  provider = GDAProvider::MLX5;
}

__host__ QueuePairMux::QueuePairUnion::QueuePairUnion(QueuePairMLX5&& mlx5)
  : mlx5{std::move(mlx5)} { }
#endif

__host__ QueuePairMux::QueuePairUnion QueuePairMux::QueuePairUnion::construct(
    QueuePairUnion&& other, GDAProvider provider) {
  switch (provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return {std::move(other.ionic)};
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return {std::move(other.bnxt)};
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return {std::move(other.mlx5)};
#endif
  default:
    static_assert(std::is_same_v<std::underlying_type_t<GDAProvider>, int>);
    LOG_ERROR_ABORT("Invalid GDAProvider (%d)", static_cast<int>(provider));
  }
}

__host__ QueuePairMux::QueuePairUnion& QueuePairMux::QueuePairUnion::assign(
    QueuePairUnion&& other, GDAProvider provider) {
  switch (provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    ionic = std::move(other.ionic);
    break;
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    bnxt = std::move(other.bnxt);
    break;
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    mlx5 = std::move(other.mlx5);
    break;
#endif
  default:
    static_assert(std::is_same_v<std::underlying_type_t<GDAProvider>, int>);
    LOG_ERROR_ABORT("Invalid GDAProvider (%d)", static_cast<int>(provider));
  }
  return *this;
}

__host__ void QueuePairMux::QueuePairUnion::destruct(GDAProvider provider) {
  /* Call destructor of active subobject, based on provider */
  switch (provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    ionic.~QueuePairIONIC();
    break;
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    bnxt.~QueuePairBNXT();
    break;
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    mlx5.~QueuePairMLX5();
    break;
#endif
  default:
    static_assert(std::is_same_v<std::underlying_type_t<GDAProvider>, int>);
    LOG_ERROR_ABORT("Invalid GDAProvider (%d)", static_cast<int>(provider));
  }
}

/* Note:
 * qp{QueuePairUnion::construct(std::move(other.qp), get_provider())}
 * only works in C++17 or later: the "guaranteed copy elision" prvalue semantics
 * mean that the qp subobject does not need to have an accessible copy or move constructor */
__host__ QueuePairMux::QueuePairMux(QueuePairMux&& other)
  : qp{QueuePairUnion::construct(std::move(other.qp), get_provider())} { }

__host__ QueuePairMux& QueuePairMux::operator=(QueuePairMux&& other) {
  /* All QueuePairMux have the same underlying provider */
  qp.assign(std::move(other.qp), get_provider());
  return *this;
}

__host__ QueuePairMux::~QueuePairMux() {
  qp.destruct(get_provider());
}

__host__ int QueuePairMux::buffer_register(void *addr, size_t length) {
  switch (get_provider()) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return qp.ionic.buffer_register(addr, length);
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return qp.bnxt.buffer_register(addr, length);
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return qp.mlx5.buffer_register(addr, length);
#endif
  default:
    static_assert(std::is_same_v<std::underlying_type_t<GDAProvider>, int>);
    LOG_ERROR_ABORT("Invalid GDAProvider (%d)", static_cast<int>(get_provider()));
  }
}

__host__ int QueuePairMux::buffer_unregister(void *addr) {
  switch (get_provider()) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return qp.ionic.buffer_unregister(addr);
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return qp.bnxt.buffer_unregister(addr);
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return qp.mlx5.buffer_unregister(addr);
#endif
  default:
    static_assert(std::is_same_v<std::underlying_type_t<GDAProvider>, int>);
    LOG_ERROR_ABORT("Invalid GDAProvider (%d)", static_cast<int>(get_provider()));
  }
}

__host__ int QueuePairMux::buffer_unregister_all() {
  switch (get_provider()) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return qp.ionic.buffer_unregister_all();
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return qp.bnxt.buffer_unregister_all();
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return qp.mlx5.buffer_unregister_all();
#endif
  default:
    static_assert(std::is_same_v<std::underlying_type_t<GDAProvider>, int>);
    LOG_ERROR_ABORT("Invalid GDAProvider (%d)", static_cast<int>(get_provider()));
  }
}

}  // namespace rocshmem
