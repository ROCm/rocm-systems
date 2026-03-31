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

#ifndef _ROCSHMEM_API_ADAPTER_HPP_
#define _ROCSHMEM_API_ADAPTER_HPP_

/**
 * ROCSHMEM API Adapter
 *
 * This header provides a unified interface for both rocSHMEM and Mori backends.
 * Use -DTEST_WITH_MORI to compile with Mori instead of rocSHMEM.
 */

#ifdef TEST_WITH_MORI
  // Mori backend
  #define MORI_WITH_MPI
  #include <mpi.h>
  #include <mori/shmem/shmem.hpp>
  #undef warpSize

  #define SHMEM_LIBRARY_NAME "Mori"

  namespace rocshmem {

#if 0
    // No context type for Mori (contexts are implicit)
    struct rocshmem_ctx_handle {
      int qp_id{0};  // Queue pair ID for Mori
    };
#endif
    using rocshmem_ctx_t = int; // rocshmem_ctx_handle;
    using rocshmem_uniqueid_t = mori::shmem::mori_shmem_uniqueid_t;
    using rocshmem_init_attr_t = mori::shmem::mori_shmem_init_attr_t;
    constexpr int ROCSHMEM_CTX_WG_PRIVATE = 4;
    constexpr int ROCSHMEM_SUCCESS = 0;
    constexpr unsigned int ROCSHMEM_INIT_WITH_MPI_COMM = mori::shmem::MORI_SHMEM_INIT_WITH_MPI_COMM;
    constexpr unsigned int ROCSHMEM_INIT_WITH_UNIQUEID = mori::shmem::MORI_SHMEM_INIT_WITH_UNIQUEID;

    // Host-side initialization
    inline int rocshmem_init() {
      MPI_Init(NULL, NULL);
      return mori::shmem::ShmemMpiInit(MPI_COMM_WORLD);
    }

    inline int rocshmem_get_uniqueid(rocshmem_uniqueid_t* uid) {
      return mori::shmem::ShmemGetUniqueId(uid);
    }

    inline int rocshmem_set_attr_uniqueid_args(int rank, int nranks, rocshmem_uniqueid_t* uid, rocshmem_init_attr_t* attr) {
      return mori::shmem::ShmemSetAttrUniqueIdArgs(rank, nranks, uid, attr);
    }

    inline int rocshmem_init_attr(unsigned int flags, rocshmem_init_attr_t* attr) {
      return mori::shmem::ShmemInitAttr(flags, attr);
    }

    inline void rocshmem_finalize() {
      mori::shmem::ShmemFinalize();
    }

    inline void* rocshmem_malloc(size_t size) {
      return mori::shmem::ShmemMalloc(size);
    }

    inline void rocshmem_free(void* ptr) {
      mori::shmem::ShmemFree(ptr);
    }

    __host__ __device__ inline int rocshmem_my_pe() {
      return mori::shmem::ShmemMyPe();
    }

    __host__ __device__ inline int rocshmem_n_pes() {
      return mori::shmem::ShmemNPes();
    }

    inline void rocshmem_barrier_all() {
      mori::shmem::ShmemBarrierAll();
    }

    // Device-side context management (no-op for Mori)
    __device__ inline void rocshmem_wg_ctx_create(int ctx_type, rocshmem_ctx_t* ctx) {
      *ctx = 0;  // Default queue pair
    }

    __device__ inline void rocshmem_wg_ctx_destroy(rocshmem_ctx_t* ctx) {
      // No-op for Mori
    }

    // Device-side PE info
    __host__ __device__ inline int rocshmem_ctx_my_pe(rocshmem_ctx_t ctx = 0) {
      return mori::shmem::ShmemMyPe();
    }

    __host__ __device__ inline int rocshmem_ctx_n_pes(rocshmem_ctx_t ctx = 0) {
      return mori::shmem::ShmemNPes();
    }

    // RMA operations - Thread scope
    __device__ inline void rocshmem_putmem(void* dest, const void* source,
                                        size_t nelems, int pe) {
      mori::shmem::ShmemPutMemNbiThread(dest, source, nelems, pe);
      mori::shmem::ShmemQuietThread(pe);
    }

    __device__ inline void rocshmem_putmem_nbi(void* dest, const void* source,
                                           size_t nelems, int pe) {
      mori::shmem::ShmemPutMemNbiThread(dest, source, nelems, pe);
    }

    __device__ inline void rocshmem_getmem(void* dest, const void* source,
                                       size_t nelems, int pe) {
      mori::shmem::ShmemGetMemNbiThread(dest, source, nelems, pe);
      mori::shmem::ShmemQuietThread(pe);
    }

    __device__ inline void rocshmem_getmem_nbi(void* dest, const void* source,
                                           size_t nelems, int pe) {
      mori::shmem::ShmemGetMemNbiThread(dest, source, nelems, pe);
    }

    __device__ inline void rocshmem_ctx_putmem(rocshmem_ctx_t ctx, void* dest, const void* source,
                                        size_t nelems, int pe) {
      mori::shmem::ShmemPutMemNbiThread(dest, source, nelems, pe, ctx);
      mori::shmem::ShmemQuietThread(pe, ctx);
    }

    __device__ inline void rocshmem_ctx_putmem_nbi(rocshmem_ctx_t ctx, void* dest, const void* source,
                                           size_t nelems, int pe) {
      mori::shmem::ShmemPutMemNbiThread(dest, source, nelems, pe, ctx);
    }

    __device__ inline void rocshmem_ctx_getmem(rocshmem_ctx_t ctx, void* dest, const void* source,
                                       size_t nelems, int pe) {
      mori::shmem::ShmemGetMemNbiThread(dest, source, nelems, pe, ctx);
      mori::shmem::ShmemQuietThread(pe, ctx);
    }

    __device__ inline void rocshmem_ctx_getmem_nbi(rocshmem_ctx_t ctx, void* dest, const void* source,
                                           size_t nelems, int pe) {
      mori::shmem::ShmemGetMemNbiThread(dest, source, nelems, pe, ctx);
    }

    // RMA operations - Wavefront scope
    __device__ inline void rocshmem_ctx_putmem_wave(rocshmem_ctx_t ctx, void* dest, const void* source,
                                            size_t nelems, int pe) {
      mori::shmem::ShmemPutMemNbiWarp(dest, source, nelems, pe, ctx);
      if (threadIdx.x % warpSize == 0) {
        mori::shmem::ShmemQuietThread(pe, ctx);
      }
    }

    __device__ inline void rocshmem_ctx_putmem_nbi_wave(rocshmem_ctx_t ctx, void* dest, const void* source,
                                                 size_t nelems, int pe) {
      mori::shmem::ShmemPutMemNbiWarp(dest, source, nelems, pe, ctx);
    }

    __device__ inline void rocshmem_ctx_getmem_wave(rocshmem_ctx_t ctx, void* dest, const void* source,
                                            size_t nelems, int pe) {
      mori::shmem::ShmemGetMemNbiWarp(dest, source, nelems, pe, ctx);
      if (threadIdx.x % warpSize == 0) {
        mori::shmem::ShmemQuietThread(pe, ctx);
      }
    }

    __device__ inline void rocshmem_ctx_getmem_nbi_wave(rocshmem_ctx_t ctx, void* dest, const void* source,
                                                 size_t nelems, int pe) {
      mori::shmem::ShmemGetMemNbiWarp(dest, source, nelems, pe, ctx);
    }

    // RMA operations - Work-group scope
    __device__ inline void rocshmem_ctx_putmem_wg(rocshmem_ctx_t ctx, void* dest, const void* source,
                                          size_t nelems, int pe) {
      mori::shmem::ShmemPutMemNbiBlock(dest, source, nelems, pe, ctx);
      if (threadIdx.x == 0) {
        mori::shmem::ShmemQuietThread(pe, ctx);
      }
    }

    __device__ inline void rocshmem_ctx_putmem_nbi_wg(rocshmem_ctx_t ctx, void* dest, const void* source,
                                              size_t nelems, int pe) {
      mori::shmem::ShmemPutMemNbiBlock(dest, source, nelems, pe, ctx);
    }

    __device__ inline void rocshmem_ctx_getmem_wg(rocshmem_ctx_t ctx, void* dest, const void* source,
                                          size_t nelems, int pe) {
      mori::shmem::ShmemGetMemNbiBlock(dest, source, nelems, pe, ctx);
      if (threadIdx.x == 0) {
        mori::shmem::ShmemQuietThread(pe, ctx);
      }
    }

    __device__ inline void rocshmem_ctx_getmem_nbi_wg(rocshmem_ctx_t ctx, void* dest, const void* source,
                                              size_t nelems, int pe) {
      mori::shmem::ShmemGetMemNbiBlock(dest, source, nelems, pe, ctx);
    }

    // Point-to-point operations
    __device__ inline void rocshmem_char_p(char* dest, char value, int pe) {
      mori::shmem::ShmemPutSizeImmNbiThread(dest, &value, sizeof(char), pe);
      mori::shmem::ShmemQuietThread(pe);
    }

    __device__ inline char rocshmem_char_g(const char* source, int pe) {
      char result;
      mori::shmem::ShmemGetMemNbiThread(&result, source, sizeof(char), pe);
      mori::shmem::ShmemQuietThread(pe);
      return result;
    }

    __device__ inline void rocshmem_ctx_char_p(rocshmem_ctx_t ctx, char* dest, char value, int pe) {
      mori::shmem::ShmemPutSizeImmNbiThread(dest, &value, sizeof(char), pe, ctx);
      mori::shmem::ShmemQuietThread(pe, ctx);
    }

    __device__ inline char rocshmem_ctx_char_g(rocshmem_ctx_t ctx, const char* source, int pe) {
      char result;
      mori::shmem::ShmemGetMemNbiThread(&result, source, sizeof(char), pe, ctx);
      mori::shmem::ShmemQuietThread(pe, ctx);
      return result;
    }

    __device__ inline void rocshmem_ctx_int_p(rocshmem_ctx_t ctx, int* dest, int value, int pe) {
      mori::shmem::ShmemPutSizeImmNbiThread(dest, &value, sizeof(int), pe, ctx);
      mori::shmem::ShmemQuietThread(pe, ctx);
    }

    __device__ inline void rocshmem_ctx_ulong_p(rocshmem_ctx_t ctx, unsigned long* dest,
                                        unsigned long value, int pe) {
      mori::shmem::ShmemPutSizeImmNbiThread(dest, &value, sizeof(unsigned long), pe, ctx);
      mori::shmem::ShmemQuietThread(pe, ctx);
    }

    __device__ inline unsigned long rocshmem_ctx_ulong_g(rocshmem_ctx_t ctx,
                                                  const unsigned long* source, int pe) {
      unsigned long result;
      mori::shmem::ShmemGetMemNbiThread(&result, source, sizeof(unsigned long), pe, ctx);
      mori::shmem::ShmemQuietThread(pe, ctx);
      return result;
    }

    // Synchronization
    __device__ inline void rocshmem_ctx_quiet(rocshmem_ctx_t ctx = 0) {
      mori::shmem::ShmemQuietThread();
    }

    __device__ inline void rocshmem_quiet() {
      mori::shmem::ShmemQuietThread();
    }

    __device__ inline void rocshmem_ctx_fence(rocshmem_ctx_t ctx = 0) {
      mori::shmem::ShmemFenceThread();
    }

    __device__ inline void rocshmem_fence() {
      mori::shmem::ShmemFenceThread();
    }

    // Wait operations
    enum rocshmem_cmp_constants {
      ROCSHMEM_CMP_EQ = 0,
      ROCSHMEM_CMP_NE = 1,
      ROCSHMEM_CMP_GT = 2,
      ROCSHMEM_CMP_GE = 3,
      ROCSHMEM_CMP_LT = 4,
      ROCSHMEM_CMP_LE = 5
    };

    __device__ inline void rocshmem_int_wait_until(int* ptr, int cmp, int value) {
      switch (cmp) {
        case ROCSHMEM_CMP_EQ:
          mori::shmem::ShmemInt32WaitUntilEquals(ptr, value);
          break;
        case ROCSHMEM_CMP_GT:
          mori::shmem::ShmemInt32WaitUntilGreaterThan(ptr, value);
          break;
        default:
          // Other comparisons not directly supported, busy wait
          while (true) {
            int current = __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
            bool done = false;
            switch (cmp) {
              case ROCSHMEM_CMP_NE: done = (current != value); break;
              case ROCSHMEM_CMP_GE: done = (current >= value); break;
              case ROCSHMEM_CMP_LT: done = (current < value); break;
              case ROCSHMEM_CMP_LE: done = (current <= value); break;
            }
            if (done) break;
          }
      }
    }

    // Atomic operations
    __device__ inline int rocshmem_ctx_int_atomic_add(rocshmem_ctx_t ctx, int* dest, int value, int pe) {
      mori::shmem::ShmemInt32AtomicAddThread(dest, value, pe, ctx);
      return 0;  // Non-fetch version
    }

    __device__ inline int rocshmem_ctx_int_atomic_fetch_add(rocshmem_ctx_t ctx, int* dest,
                                                     int value, int pe) {
      return mori::shmem::ShmemInt32AtomicFetchAddThread(dest, value, pe, ctx);
    }

    __device__ inline long rocshmem_ctx_long_atomic_add(rocshmem_ctx_t ctx,
                                                                    long* dest,
                                                                    long value,
                                                                    int pe) {
      mori::shmem::ShmemLongAtomicAddThread(dest, value, pe, ctx);
      return 0;
    }

    __device__ inline long rocshmem_ctx_long_atomic_fetch_add(rocshmem_ctx_t ctx,
                                                                          long* dest,
                                                                          long value,
                                                                          int pe) {
      return mori::shmem::ShmemLongAtomicFetchAddThread(dest, value, pe, ctx);
    }

    __device__ inline long rocshmem_ctx_long_atomic_fetch_inc(rocshmem_ctx_t ctx,
                                                                          long* dest,
                                                                          int pe) {
      return mori::shmem::ShmemLongAtomicFetchAddThread(dest, 1, pe, ctx);
    }

    __device__ inline long rocshmem_ctx_long_atomic_inc(rocshmem_ctx_t ctx,
                                                                    long* dest,
                                                                    int pe) {
      mori::shmem::ShmemLongAtomicAddThread(dest, 1, pe, ctx);
      return 0;
    }

    __device__ inline long rocshmem_ctx_long_atomic_compare_swap(rocshmem_ctx_t ctx,
                                                                             long* dest,
                                                                             long cond,
                                                                             long value,
                                                                             int pe) {
      abort();
#if 0
      return mori::shmem::ShmemAtomicTypeFetchThread<long>(dest, value, cond,
                                                  atomicType::AMO_CAS, pe, ctx);
#endif
      return 0;
    }


    __host__ __device__ inline void rocshmem_global_exit(int code) { abort(); }

  } // namespace rocshmem_adapter

#else
  // rocSHMEM backend (default)
  #include <rocshmem/rocshmem.hpp>

  #define SHMEM_LIBRARY_NAME "rocSHMEM"
#endif // TEST_WITH_MORI

#endif /* _ROCSHMEM_API_ADAPTER_HPP_ */
