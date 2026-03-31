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

#ifndef _SHMEM_API_ADAPTER_HPP_
#define _SHMEM_API_ADAPTER_HPP_

/**
 * SHMEM API Adapter
 *
 * This header provides a unified interface for both rocSHMEM and Mori backends.
 * Use -DUSE_MORI_BACKEND to compile with Mori instead of rocSHMEM.
 */

#ifdef USE_MORI_BACKEND
  // Mori backend
  #define MORI_WITH_MPI
  #include <mpi.h>
  #include <mori/shmem/shmem.hpp>
  #undef warpSize

  #define SHMEM_BACKEND_NAME "Mori"

  namespace shmem_adapter {
    using namespace mori::shmem;
    using namespace mori::core;

#if 0
    // No context type for Mori (contexts are implicit)
    struct shmem_ctx_handle {
      int qp_id{0};  // Queue pair ID for Mori
    };
#endif
    using shmem_ctx_t = int;
   // shmem_ctx_handle;

    // Host-side initialization
    inline int shmem_init() {
      MPI_Init(NULL, NULL);
      return mori::shmem::ShmemMpiInit(MPI_COMM_WORLD);
    }

    inline void shmem_finalize() {
      mori::shmem::ShmemFinalize();
    }

    inline void* shmem_malloc(size_t size) {
      return mori::shmem::ShmemMalloc(size);
    }

    inline void shmem_free(void* ptr) {
      mori::shmem::ShmemFree(ptr);
    }

    inline int shmem_my_pe() {
      return mori::shmem::ShmemMyPe();
    }

    inline int shmem_n_pes() {
      return mori::shmem::ShmemNPes();
    }

    inline void shmem_barrier_all() {
  ///    mori::shmem::ShmemBarrierAll();
    }

    // Device-side context management (no-op for Mori)
    __device__ inline void shmem_wg_ctx_create(int ctx_type, shmem_ctx_t* ctx) {
      *ctx = 0;  // Default queue pair
    }

    __device__ inline void shmem_wg_ctx_destroy(shmem_ctx_t* ctx) {
      // No-op for Mori
    }

    // Device-side PE info
    __host__ __device__ inline int shmem_my_pe(shmem_ctx_t ctx = 0) {
      return ShmemMyPe();
    }

    __host__ __device__ inline int shmem_n_pes(shmem_ctx_t ctx = 0) {
      return ShmemNPes();
    }

    // RMA operations - Thread scope
    __device__ inline void shmem_putmem(shmem_ctx_t ctx, void* dest, const void* source,
                                        size_t nelems, int pe) {
      ShmemPutMemNbiThread(dest, source, nelems, pe, ctx);
      ShmemQuietThread(pe, ctx);
    }

    __device__ inline void shmem_putmem_nbi(shmem_ctx_t ctx, void* dest, const void* source,
                                           size_t nelems, int pe) {
      ShmemPutMemNbiThread(dest, source, nelems, pe, ctx);
    }

    __device__ inline void shmem_getmem(shmem_ctx_t ctx, void* dest, const void* source,
                                       size_t nelems, int pe) {
      ShmemGetMemNbiThread(dest, source, nelems, pe, ctx);
      ShmemQuietThread(pe, ctx);
    }

    __device__ inline void shmem_getmem_nbi(shmem_ctx_t ctx, void* dest, const void* source,
                                           size_t nelems, int pe) {
      ShmemGetMemNbiThread(dest, source, nelems, pe, ctx);
    }

    // RMA operations - Wavefront scope
    __device__ inline void shmem_putmem_wave(shmem_ctx_t ctx, void* dest, const void* source,
                                            size_t nelems, int pe) {
      ShmemPutMemNbiWarp(dest, source, nelems, pe, ctx);
      if (threadIdx.x % warpSize == 0) {
        ShmemQuietThread(pe, ctx);
      }
    }

    __device__ inline void shmem_putmem_nbi_wave(shmem_ctx_t ctx, void* dest, const void* source,
                                                 size_t nelems, int pe) {
      ShmemPutMemNbiWarp(dest, source, nelems, pe, ctx);
    }

    __device__ inline void shmem_getmem_wave(shmem_ctx_t ctx, void* dest, const void* source,
                                            size_t nelems, int pe) {
      ShmemGetMemNbiWarp(dest, source, nelems, pe, ctx);
      if (threadIdx.x % warpSize == 0) {
        ShmemQuietThread(pe, ctx);
      }
    }

    __device__ inline void shmem_getmem_nbi_wave(shmem_ctx_t ctx, void* dest, const void* source,
                                                 size_t nelems, int pe) {
      ShmemGetMemNbiWarp(dest, source, nelems, pe, ctx);
    }

    // RMA operations - Work-group scope
    __device__ inline void shmem_putmem_wg(shmem_ctx_t ctx, void* dest, const void* source,
                                          size_t nelems, int pe) {
      ShmemPutMemNbiBlock(dest, source, nelems, pe, ctx);
      if (threadIdx.x == 0) {
        ShmemQuietThread(pe, ctx);
      }
    }

    __device__ inline void shmem_putmem_nbi_wg(shmem_ctx_t ctx, void* dest, const void* source,
                                              size_t nelems, int pe) {
      ShmemPutMemNbiBlock(dest, source, nelems, pe, ctx);
    }

    __device__ inline void shmem_getmem_wg(shmem_ctx_t ctx, void* dest, const void* source,
                                          size_t nelems, int pe) {
      ShmemGetMemNbiBlock(dest, source, nelems, pe, ctx);
      if (threadIdx.x == 0) {
        ShmemQuietThread(pe, ctx);
      }
    }

    __device__ inline void shmem_getmem_nbi_wg(shmem_ctx_t ctx, void* dest, const void* source,
                                              size_t nelems, int pe) {
      ShmemGetMemNbiBlock(dest, source, nelems, pe, ctx);
    }

    // Point-to-point operations
    __device__ inline void shmem_char_p(shmem_ctx_t ctx, char* dest, char value, int pe) {
      ShmemPutSizeImmNbiThread(dest, &value, sizeof(char), pe, ctx);
      ShmemQuietThread(pe, ctx);
    }

    __device__ inline char shmem_char_g(shmem_ctx_t ctx, const char* source, int pe) {
      char result;
      ShmemGetMemNbiThread(&result, source, sizeof(char), pe, ctx);
      ShmemQuietThread(pe, ctx);
      return result;
    }

    __device__ inline void shmem_int_p(shmem_ctx_t ctx, int* dest, int value, int pe) {
      ShmemPutSizeImmNbiThread(dest, &value, sizeof(int), pe, ctx);
      ShmemQuietThread(pe, ctx);
    }

    __device__ inline void shmem_ulong_p(shmem_ctx_t ctx, unsigned long* dest,
                                        unsigned long value, int pe) {
      ShmemPutSizeImmNbiThread(dest, &value, sizeof(unsigned long), pe, ctx);
      ShmemQuietThread(pe, ctx);
    }

    __device__ inline unsigned long shmem_ulong_g(shmem_ctx_t ctx,
                                                  const unsigned long* source, int pe) {
      unsigned long result;
      ShmemGetMemNbiThread(&result, source, sizeof(unsigned long), pe, ctx);
      ShmemQuietThread(pe, ctx);
      return result;
    }

    // Synchronization
    __device__ inline void shmem_quiet(shmem_ctx_t ctx) {
      ShmemQuietThread();
    }

    __device__ inline void shmem_fence(shmem_ctx_t ctx) {
      ShmemFenceThread();
    }

    // Wait operations
    enum shmem_cmp_constants {
      SHMEM_CMP_EQ = 0,
      SHMEM_CMP_NE = 1,
      SHMEM_CMP_GT = 2,
      SHMEM_CMP_GE = 3,
      SHMEM_CMP_LT = 4,
      SHMEM_CMP_LE = 5
    };

    __device__ inline void shmem_int_wait_until(int* ptr, int cmp, int value) {
      switch (cmp) {
        case SHMEM_CMP_EQ:
          ShmemInt32WaitUntilEquals(ptr, value);
          break;
        case SHMEM_CMP_GT:
          ShmemInt32WaitUntilGreaterThan(ptr, value);
          break;
        default:
          // Other comparisons not directly supported, busy wait
          while (true) {
            int current = __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
            bool done = false;
            switch (cmp) {
              case SHMEM_CMP_NE: done = (current != value); break;
              case SHMEM_CMP_GE: done = (current >= value); break;
              case SHMEM_CMP_LT: done = (current < value); break;
              case SHMEM_CMP_LE: done = (current <= value); break;
            }
            if (done) break;
          }
      }
    }

    // Atomic operations
    __device__ inline int shmem_int_atomic_add(shmem_ctx_t ctx, int* dest, int value, int pe) {
      ShmemInt32AtomicAddThread(dest, value, pe, ctx);
      return 0;  // Non-fetch version
    }

    __device__ inline int shmem_int_atomic_fetch_add(shmem_ctx_t ctx, int* dest,
                                                     int value, int pe) {
      return ShmemInt32AtomicFetchAddThread(dest, value, pe, ctx);
    }

    __device__ inline long shmem_long_atomic_add(shmem_ctx_t ctx,
                                                                    long* dest,
                                                                    long value,
                                                                    int pe) {
      ShmemLongAtomicAddThread(dest, value, pe, ctx);
      return 0;
    }

    __device__ inline long shmem_long_atomic_fetch_add(shmem_ctx_t ctx,
                                                                          long* dest,
                                                                          long value,
                                                                          int pe) {
      return ShmemLongAtomicFetchAddThread(dest, value, pe, ctx);
    }

    __device__ inline long shmem_long_atomic_fetch_inc(shmem_ctx_t ctx,
                                                                          long* dest,
                                                                          int pe) {
      return ShmemLongAtomicFetchAddThread(dest, 1, pe, ctx);
    }

    __device__ inline long shmem_long_atomic_inc(shmem_ctx_t ctx,
                                                                    long* dest,
                                                                    int pe) {
      ShmemLongAtomicAddThread(dest, 1, pe, ctx);
      return 0;
    }

    __device__ inline long shmem_long_atomic_compare_swap(shmem_ctx_t ctx,
                                                                             long* dest,
                                                                             long cond,
                                                                             long value,
                                                                             int pe) {
      abort();
#if 0
      return ShmemAtomicTypeFetchThread<long>(dest, value, cond,
                                                  atomicType::AMO_CAS, pe, ctx);
#endif
      return 0;
    }


    __host__ __device__ inline void shmem_global_exit(int code) { abort(); }

  } // namespace shmem_adapter

#else
  // rocSHMEM backend (default)
  #include <rocshmem/rocshmem.hpp>

  #define SHMEM_BACKEND_NAME "rocSHMEM"

  namespace shmem_adapter {
    using namespace rocshmem;

    // Use rocSHMEM's context type directly
    using shmem_ctx_t = rocshmem_ctx_t;

    // Host-side initialization
    inline int shmem_init() {
      rocshmem_init();
      return 0;
    }

    inline void shmem_finalize() {
      rocshmem_finalize();
    }

    inline void* shmem_malloc(size_t size) {
      return rocshmem_malloc(size);
    }

    inline void shmem_free(void* ptr) {
      rocshmem_free(ptr);
    }

    inline int shmem_my_pe() {
      return rocshmem_my_pe();
    }

    inline int shmem_n_pes() {
      return rocshmem_n_pes();
    }

    inline void shmem_barrier_all() {
      rocshmem_barrier_all();
    }

    // Device-side context management
    __device__ inline void shmem_wg_ctx_create(int ctx_type, shmem_ctx_t* ctx) {
      rocshmem_wg_ctx_create(ctx_type, ctx);
    }

    __device__ inline void shmem_wg_ctx_destroy(shmem_ctx_t* ctx) {
      rocshmem_wg_ctx_destroy(ctx);
    }

    // Device-side PE info
    __device__ inline int shmem_my_pe(shmem_ctx_t ctx = 0) {
      return rocshmem_ctx_my_pe(ctx);
    }

    __device__ inline int shmem_n_pes(shmem_ctx_t ctx = 0) {
      return rocshmem_ctx_n_pes(ctx);
    }

    // RMA operations - Thread scope
    __device__ inline void shmem_putmem(shmem_ctx_t ctx, void* dest, const void* source,
                                       size_t nelems, int pe) {
      rocshmem_ctx_putmem(ctx, dest, source, nelems, pe);
    }

    __device__ inline void shmem_putmem_nbi(shmem_ctx_t ctx, void* dest, const void* source,
                                           size_t nelems, int pe) {
      rocshmem_ctx_putmem_nbi(ctx, dest, source, nelems, pe);
    }

    __device__ inline void shmem_getmem(shmem_ctx_t ctx, void* dest, const void* source,
                                       size_t nelems, int pe) {
      rocshmem_ctx_getmem(ctx, dest, source, nelems, pe);
    }

    __device__ inline void shmem_getmem_nbi(shmem_ctx_t ctx, void* dest, const void* source,
                                           size_t nelems, int pe) {
      rocshmem_ctx_getmem_nbi(ctx, dest, source, nelems, pe);
    }

    // RMA operations - Wavefront scope
    __device__ inline void shmem_putmem_wave(shmem_ctx_t ctx, void* dest, const void* source,
                                            size_t nelems, int pe) {
      rocshmem_ctx_putmem_wave(ctx, dest, source, nelems, pe);
    }

    __device__ inline void shmem_putmem_nbi_wave(shmem_ctx_t ctx, void* dest, const void* source,
                                                 size_t nelems, int pe) {
      rocshmem_ctx_putmem_nbi_wave(ctx, dest, source, nelems, pe);
    }

    __device__ inline void shmem_getmem_wave(shmem_ctx_t ctx, void* dest, const void* source,
                                            size_t nelems, int pe) {
      rocshmem_ctx_getmem_wave(ctx, dest, source, nelems, pe);
    }

    __device__ inline void shmem_getmem_nbi_wave(shmem_ctx_t ctx, void* dest, const void* source,
                                                 size_t nelems, int pe) {
      rocshmem_ctx_getmem_nbi_wave(ctx, dest, source, nelems, pe);
    }

    // RMA operations - Work-group scope
    __device__ inline void shmem_putmem_wg(shmem_ctx_t ctx, void* dest, const void* source,
                                          size_t nelems, int pe) {
      rocshmem_ctx_putmem_wg(ctx, dest, source, nelems, pe);
    }

    __device__ inline void shmem_putmem_nbi_wg(shmem_ctx_t ctx, void* dest, const void* source,
                                              size_t nelems, int pe) {
      rocshmem_ctx_putmem_nbi_wg(ctx, dest, source, nelems, pe);
    }

    __device__ inline void shmem_getmem_wg(shmem_ctx_t ctx, void* dest, const void* source,
                                          size_t nelems, int pe) {
      rocshmem_ctx_getmem_wg(ctx, dest, source, nelems, pe);
    }

    __device__ inline void shmem_getmem_nbi_wg(shmem_ctx_t ctx, void* dest, const void* source,
                                              size_t nelems, int pe) {
      rocshmem_ctx_getmem_nbi_wg(ctx, dest, source, nelems, pe);
    }

    // Point-to-point operations
    __device__ inline void shmem_char_p(shmem_ctx_t ctx, char* dest, char value, int pe) {
      rocshmem_ctx_char_p(ctx, dest, value, pe);
    }

    __device__ inline char shmem_char_g(shmem_ctx_t ctx, const char* source, int pe) {
      return rocshmem_ctx_char_g(ctx, source, pe);
    }

    __device__ inline void shmem_int_p(shmem_ctx_t ctx, int* dest, int value, int pe) {
      rocshmem_ctx_int_p(ctx, dest, value, pe);
    }

    __device__ inline void shmem_ulong_p(shmem_ctx_t ctx, unsigned long* dest,
                                        unsigned long value, int pe) {
      rocshmem_ctx_ulong_p(ctx, dest, value, pe);
    }

    __device__ inline unsigned long shmem_ulong_g(shmem_ctx_t ctx,
                                                  const unsigned long* source, int pe) {
      return rocshmem_ctx_ulong_g(ctx, source, pe);
    }

    // Synchronization
    __device__ inline void shmem_quiet(shmem_ctx_t ctx) {
      rocshmem_ctx_quiet(ctx);
    }

    __device__ inline void shmem_fence(shmem_ctx_t ctx) {
      rocshmem_ctx_fence(ctx);
    }

    // Wait operations
    using shmem_cmp_constants = int;
    static constexpr int SHMEM_CMP_EQ = ROCSHMEM_CMP_EQ;
    static constexpr int SHMEM_CMP_NE = ROCSHMEM_CMP_NE;
    static constexpr int SHMEM_CMP_GT = ROCSHMEM_CMP_GT;
    static constexpr int SHMEM_CMP_GE = ROCSHMEM_CMP_GE;
    static constexpr int SHMEM_CMP_LT = ROCSHMEM_CMP_LT;
    static constexpr int SHMEM_CMP_LE = ROCSHMEM_CMP_LE;

    __device__ inline void shmem_int_wait_until(int* ptr, int cmp, int value) {
      rocshmem_int_wait_until(ptr, cmp, value);
    }

    // Atomic operations
    __device__ inline int shmem_int_atomic_add(shmem_ctx_t ctx, int* dest, int value, int pe) {
      rocshmem_ctx_int_atomic_add(ctx, dest, value, pe);
      return 0;
    }

    __device__ inline int shmem_int_atomic_fetch_add(shmem_ctx_t ctx, int* dest,
                                                     int value, int pe) {
      return rocshmem_ctx_int_atomic_fetch_add(ctx, dest, value, pe);
    }

    __device__ inline uint64_t shmem_long_atomic_add(shmem_ctx_t ctx,
                                                                    uint64_t* dest,
                                                                    uint64_t value,
                                                                    int pe) {
      rocshmem_ctx_long_atomic_add(ctx, dest, value, pe);
      return 0;
    }

    __device__ inline uint64_t shmem_long_atomic_fetch_add(shmem_ctx_t ctx,
                                                                          uint64_t* dest,
                                                                          uint64_t value,
                                                                          int pe) {
      return rocshmem_ctx_long_atomic_fetch_add(ctx, dest, value, pe);
    }

    __device__ inline uint64_t shmem_long_atomic_fetch_inc(shmem_ctx_t ctx,
                                                                          uint64_t* dest,
                                                                          int pe) {
      return rocshmem_ctx_long_atomic_fetch_inc(ctx, dest, pe);
    }

    __device__ inline uint64_t shmem_long_atomic_inc(shmem_ctx_t ctx,
                                                                    uint64_t* dest,
                                                                    int pe) {
      rocshmem_ctx_long_atomic_inc(ctx, dest, pe);
      return 0;
    }

    __device__ inline uint64_t shmem_long_atomic_compare_swap(shmem_ctx_t ctx,
                                                                             uint64_t* dest,
                                                                             uint64_t cond,
                                                                             uint64_t value,
                                                                             int pe) {
      return rocshmem_ctx_long_atomic_compare_swap(ctx, dest, cond, value, pe);
    }

  } // namespace shmem_adapter

#endif // USE_MORI_BACKEND

#endif /* _SHMEM_API_ADAPTER_HPP_ */
