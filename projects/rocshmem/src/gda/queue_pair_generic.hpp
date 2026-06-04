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

#ifndef LIBRARY_SRC_GDA_QUEUE_PAIR_GENERIC_HPP_
#define LIBRARY_SRC_GDA_QUEUE_PAIR_GENERIC_HPP_

#include <hip/hip_runtime.h>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)

#include "queue_pair.hpp"
#include "util.hpp"

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

class QueuePairGeneric;

template <> struct QueuePairTraits<QueuePairGeneric> {
  /**
   * @brief Generic OpCode enumeration
   */
  enum class OpCode {
    RDMA_WRITE,
    RDMA_READ,
    ATOMIC_FA,
    ATOMIC_CS,
  };
};

class QueuePairGeneric : public QueuePairSHMEM<QueuePairGeneric> {
public:
#if defined(GDA_IONIC)
  __host__ QueuePairGeneric(QueuePairIONIC&& ionic);
#endif
#if defined(GDA_BNXT)
  __host__ QueuePairGeneric(QueuePairBNXT&& bnxt);
#endif
#if defined(GDA_MLX5)
  __host__ QueuePairGeneric(QueuePairMLX5&& mlx5);
#endif

  __host__ QueuePairGeneric(const QueuePairGeneric& other) = delete;
  __host__ QueuePairGeneric& operator=(const QueuePairGeneric& other) = delete;
  __host__ QueuePairGeneric& operator=(QueuePairGeneric&& other);
  __host__ QueuePairGeneric(QueuePairGeneric&& other);
  __host__ ~QueuePairGeneric();

public:
  template <OpCode Op,
            bool RingDB = true, bool ThreadSafe = true, bool CheckCQ = true>
  __device__ void post_wqe_rma(uintptr_t laddr, uintptr_t raddr, size_t size,
                               const ActiveWFInfo& wf_info);

  template <OpCode Op,
            bool RingDB = true, bool ThreadSafe = true, bool CheckCQ = true>
  __device__ void post_wqe_rma_single(uintptr_t laddr, uintptr_t raddr, size_t size);

  template <OpCode Op, AMOFetchType Fetch,
            bool RingDB = true, bool ThreadSafe = true, bool CheckCQ = true>
  __device__ amo_ret_t<Fetch> post_wqe_amo(uintptr_t raddr, uint64_t value, uint64_t cond,
                                                 const ActiveWFInfo& wf_info);

  template <OpCode Op, AMOFetchType Fetch,
            bool RingDB = true, bool ThreadSafe = true, bool CheckCQ = true>
  __device__ amo_ret_t<Fetch> post_wqe_amo_single(uintptr_t raddr, uint64_t value, uint64_t cond);

  __device__ void quiet_single();

  /**
   * @brief Register buffer for use as local address in rocSHMEM routines
   *
   * @param[in] addr Base address of buffer.
   * @param[in] length Length of buffer.
   *
   * @retval ROCSHMEM_SUCCESS Buffer registered successfully.
   * @retval ROCSHMEM_ERROR Buffer could not be registered.
   */
  __host__ int buffer_register(void *addr, size_t length);

  /**
   * @brief Unregister buffer.
   * Buffer must have previously been registered with buffer_register(void *, size_t).
   *
   * @param[in] addr Base address of buffer.
   *
   * @retval ROCSHMEM_SUCCESS Buffer unregistered successfully.
   */
  __host__ int buffer_unregister(void *addr);

private:
  union QueuePairProvider {
#if defined(GDA_IONIC)
    QueuePairIONIC ionic;
    __host__ QueuePairProvider(QueuePairIONIC&& ionic);
    __host__ QueuePairProvider& operator=(QueuePairIONIC&& ionic);
#endif
#if defined(GDA_BNXT)
    QueuePairBNXT  bnxt;
    __host__ QueuePairProvider(QueuePairBNXT&& bnxt);
    __host__ QueuePairProvider& operator=(QueuePairBNXT&& bnxt);
#endif
#if defined(GDA_MLX5)
    QueuePairMLX5  mlx5;
    __host__ QueuePairProvider(QueuePairMLX5&& mlx5);
    __host__ QueuePairProvider& operator=(QueuePairMLX5&& mlx5);
#endif

    /* Copy and move constructors and assignment operators are deleted,
     * since they can't know which subobject to construct or assign */
    __host__ QueuePairProvider(const QueuePairProvider& other)            = delete;
    __host__ QueuePairProvider(QueuePairProvider&& other)                 = delete;
    __host__ QueuePairProvider& operator=(const QueuePairProvider& other) = delete;
    __host__ QueuePairProvider& operator=(QueuePairProvider&& other)      = delete;

    /*
     * @brief Empty destructor. Call QueuePairProvider::destruct to destroy active subobject.
     *
     * Since union members have non-trivial destructors,
     * the implicitely-declared or explicitly-defaulted destructor is defined as deleted.
     * To allow any definition of QueuePairGeneric::~QueuePairGeneric(),
     * we must provide some definition of ~QueuePairProvider(), as otherwise it cannot be called
     * as part of the destructor sequence of ~QueuePairGeneric().
     *
     * ~QueuePairProvider() is defined as empty;
     * destruction of the active subobject is delegated to QueuePairProvider::destruct.
     * Once provider is in __constant__ memory,
     * destruction can be done directly by ~QueuePairProvider().
     */
    __host__ ~QueuePairProvider() { }


    /*
     * @brief Construct new QueuePairProvider, moving the active subobject from other.
     *
     * @param[in,out] other QueuePairProvider object to move from.
     * @param[in] provider Type of active subobject of other.
     *
     * @return QueuePairProvider with an active subobject moved from other.
     */
    static __host__ QueuePairProvider construct(QueuePairProvider&& other, GDAProvider provider);

    /*
     * @brief Call destructor of active subobject.
     *
     * @param[in] provider Type of active subobject.
     */
    __host__ void destruct(GDAProvider provider);
  } qp;
  GDAProvider provider;

  /*
   * @brief Convert from QueuePairGeneric::OpCode to the equivalent Provider::OpCode
   */
  template <OpCode Op, typename Provider>
  static __host__ __device__ constexpr typename Provider::OpCode provider_op();
};

template <QueuePairGeneric::OpCode Op, typename Provider>
__host__ __device__ constexpr typename Provider::OpCode QueuePairGeneric::provider_op() {
  if constexpr (Op == OpCode::RDMA_WRITE) {
    return Provider::OpCode::RDMA_WRITE;
  } else if constexpr (Op == OpCode::RDMA_READ) {
    return Provider::OpCode::RDMA_READ;
  } else if constexpr (Op == OpCode::ATOMIC_FA) {
    return Provider::OpCode::ATOMIC_FA;
  } else if constexpr (Op == OpCode::ATOMIC_CS) {
    return Provider::OpCode::ATOMIC_CS;
  }
}

template <QueuePairGeneric::OpCode Op, bool RingDB, bool ThreadSafe, bool CheckCQ>
__device__ void QueuePairGeneric::post_wqe_rma(
    uintptr_t laddr, uintptr_t raddr, size_t size, const ActiveWFInfo& wf_info) {
  switch (provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return qp.ionic.post_wqe_rma<provider_op<Op, QueuePairIONIC>(),
                                 RingDB, ThreadSafe, CheckCQ>(laddr, raddr, size, wf_info);
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return qp.bnxt.post_wqe_rma<provider_op<Op, QueuePairBNXT>(),
                                RingDB, ThreadSafe, CheckCQ>(laddr, raddr, size, wf_info);
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return qp.mlx5.post_wqe_rma<provider_op<Op, QueuePairMLX5>(),
                                RingDB, ThreadSafe, CheckCQ>(laddr, raddr, size, wf_info);
#endif
  default:
    assert(false /* invalid GDAProvider */);
    __builtin_unreachable();
  }
}

template <QueuePairGeneric::OpCode Op, bool RingDB, bool ThreadSafe, bool CheckCQ>
__device__ void QueuePairGeneric::post_wqe_rma_single(
    uintptr_t laddr, uintptr_t raddr, size_t size) {
  switch (provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return qp.ionic.post_wqe_rma_single<provider_op<Op, QueuePairIONIC>(),
                                        RingDB, ThreadSafe, CheckCQ>(laddr, raddr, size);
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return qp.bnxt.post_wqe_rma_single<provider_op<Op, QueuePairBNXT>(),
                                       RingDB, ThreadSafe, CheckCQ>(laddr, raddr, size);
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return qp.mlx5.post_wqe_rma_single<provider_op<Op, QueuePairMLX5>(),
                                       RingDB, ThreadSafe, CheckCQ>(laddr, raddr, size);
#endif
  default:
    assert(false /* invalid GDAProvider */);
    __builtin_unreachable();
  }
}

template <QueuePairGeneric::OpCode Op, AMOFetchType Fetch,
          bool RingDB, bool ThreadSafe, bool CheckCQ>
__device__ QueuePairGeneric::amo_ret_t<Fetch> QueuePairGeneric::post_wqe_amo(
    uintptr_t raddr, uint64_t value, uint64_t cond, const ActiveWFInfo& wf_info) {
  static_assert(Fetch != AMOFetchType::NonBlocking);
  switch (provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return qp.ionic.post_wqe_amo<provider_op<Op, QueuePairIONIC>(), Fetch,
                                 RingDB, ThreadSafe, CheckCQ>(raddr, value, cond, wf_info);
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return qp.bnxt.post_wqe_amo<provider_op<Op, QueuePairBNXT>(), Fetch,
                                RingDB, ThreadSafe, CheckCQ>(raddr, value, cond, wf_info);
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return qp.mlx5.post_wqe_amo<provider_op<Op, QueuePairMLX5>(), Fetch,
                                RingDB, ThreadSafe, CheckCQ>(raddr, value, cond, wf_info);
#endif
  default:
    assert(false /* invalid GDAProvider */);
    __builtin_unreachable();
  }
}

template <QueuePairGeneric::OpCode Op, AMOFetchType Fetch,
          bool RingDB, bool ThreadSafe, bool CheckCQ>
__device__ QueuePairGeneric::amo_ret_t<Fetch> QueuePairGeneric::post_wqe_amo_single(
    uintptr_t raddr, uint64_t value, uint64_t cond) {
  static_assert(Fetch != AMOFetchType::NonBlocking);
  switch (provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC:
    return qp.ionic.post_wqe_amo_single<provider_op<Op, QueuePairIONIC>(), Fetch,
                                        RingDB, ThreadSafe, CheckCQ>(raddr, value, cond);
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT:
    return qp.bnxt.post_wqe_amo_single<provider_op<Op, QueuePairBNXT>(), Fetch,
                                       RingDB, ThreadSafe, CheckCQ>(raddr, value, cond);
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5:
    return qp.mlx5.post_wqe_amo_single<provider_op<Op, QueuePairMLX5>(), Fetch,
                                       RingDB, ThreadSafe, CheckCQ>(raddr, value, cond);
#endif
  default:
    assert(false /* invalid GDAProvider */);
    __builtin_unreachable();
  }
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_QUEUE_PAIR_GENERIC_HPP_
