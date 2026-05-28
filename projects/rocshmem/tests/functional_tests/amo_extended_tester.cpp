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

#include "amo_extended_tester.hpp"

#include <iostream>
#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/* Declare the global kernel template with a generic implementation */
template <typename T>
__global__ void AMOExtendedTest(int loop, int skip, long long int *start_time,
                                long long int *end_time, T *dest, T *ret_val,
                                AddrMode addr_mode, TestType type,
                                ShmemContextType ctx_type) {
  return;
}

template <class T>
__device__ inline T* compute_target_ptr(T* base_ptr, AddrMode addr_mode,
                                        int wg_idx, int itr, int n_wgs) {
  // PerBlock: element = wg_idx, with n_wgs elements per loop
  // PerGrid : single element shared by the whole grid per loop
  if (addr_mode == AddrMode::PerBlock) {
    size_t offset = wg_idx + itr * n_wgs;
    return base_ptr + offset;
  } else { // PerGrid
    return base_ptr + itr;
  }
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
template <typename T>
AMOExtendedTester<T>::AMOExtendedTester(TesterArguments args) : Tester(args) {
  n_out   = (args.addr_mode == AddrMode::PerBlock) ? args.num_wgs : 1;
  n_in    = args.num_wgs * args.wg_size;
  n_loops = args.loop + args.skip;

  // One return per *thread* per loop
  CHECK_HIP(hipMalloc((void **)&ret_val, args.max_msg_size * n_in * n_loops));

  dest = (T *)rocshmem_malloc(args.max_msg_size * n_out * n_loops);
  if (dest == nullptr) {
    std::cerr << "Error allocating memory from symmetric heap" << std::endl;
    std::cerr << "dest: " << (void*)dest << std::endl;
  }
}

template <typename T>
AMOExtendedTester<T>::~AMOExtendedTester() {
  CHECK_HIP(hipFree(ret_val));
  rocshmem_free(dest);
}

template <typename T>
void AMOExtendedTester<T>::resetBuffers(size_t size) {
  memset(ret_val, 0, args.max_msg_size * n_in  * n_loops);
  memset(dest,    0, args.max_msg_size * n_out * n_loops);
}

template <typename T>
void AMOExtendedTester<T>::launchKernel(dim3 gridsize, dim3 blocksize, int loop,
                                        size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(AMOExtendedTest, gridsize, blocksize, shared_bytes, stream,
                     args.loop, args.skip, start_time, end_time, dest,
                     ret_val, args.addr_mode, _type, _shmem_context);

  num_msgs       = n_loops   * gridsize.x * blocksize.x;
  num_timed_msgs = args.loop * gridsize.x * blocksize.x;
}

template <typename G>
void fail_eq(const G& got, const G& exp) {
  std::cerr << "data validation error\n"
            << "got " << got << ", expected " << exp << std::endl;
  std::exit(-1);
}

// Map (loop, elem_idx) -> dest[] index for current address mode.
template <typename T>
int AMOExtendedTester<T>::destIndex(int l, int elem_idx) const {
  return (args.addr_mode == AddrMode::PerBlock)
           ? l * static_cast<int>(args.num_wgs) + elem_idx
           : l; // PerGrid has a single element per loop
}

// Number of output elements to check per loop for current address mode.
template <typename T>
int AMOExtendedTester<T>::numElems() const {
  return (args.addr_mode == AddrMode::PerBlock)
           ? static_cast<int>(args.num_wgs)
           : 1; // PerGrid
}

// Return pointer to the start of the ret_val “chunk” for (loop, elem_idx)
// plus the chunk length for this address mode.
template <typename T>
std::pair<T*, int> AMOExtendedTester<T>::retChunk(int l, int elem_idx) const {
  if (args.addr_mode == AddrMode::PerBlock) {
    // One chunk per element (workgroup): wg_size returns
    T*  p  = ret_val + l * n_in + elem_idx * args.wg_size;
    int sz = static_cast<int>(args.wg_size);
    return {p, sz};
  }
  // PerGrid: one big chunk per loop (all threads)
  T*  p  = ret_val + l * n_in;
  int sz = static_cast<int>(n_in);
  return {p, sz};
}

template <typename T>
void AMOExtendedTester<T>::verifyDestValues() {
  const int loops   = static_cast<int>(n_loops);
  const int n_elems = numElems();

  auto check_equal_all = [&](T expected) {
    for (int l = 0; l < loops; ++l) {
      for (int elem = 0; elem < n_elems; ++elem) {
        const int idx = destIndex(l, elem);
        if (dest[idx] != expected) fail_eq(dest[idx], expected);
      }
    }
  };

  auto check_nonzero_all = [&]() {
    for (int l = 0; l < loops; ++l) {
      for (int elem = 0; elem < n_elems; ++elem) {
        const int idx = destIndex(l, elem);
        if (dest[idx] == T{0}) fail_eq(dest[idx], T{1});
      }
    }
  };

  switch (_type) {
    case AMO_FetchTestType:
      // fetch does not modify dest -> stays 0
      check_equal_all(T{0});
      break;

    case AMO_SetTestType:
      // set writes a constant (17)
      check_equal_all(static_cast<T>(17));
      break;

    case AMO_SwapTestType:
      // swap writes non-zero values -> final must be non-zero
      check_nonzero_all();
      break;

    default:
      break;
  }
}

template <typename T>
void AMOExtendedTester<T>::verifyReturnValues() {
  // Only fetch/swap produce return values to validate
  if (_type == AMO_SetTestType) return;

  const int loops   = static_cast<int>(n_loops);
  const int n_elems = numElems();

  for (int l = 0; l < loops; ++l) {
    for (int elem = 0; elem < n_elems; ++elem) {
      auto [p, cnt] = retChunk(l, elem);

      int zeros = 0;
      for (int i = 0; i < cnt; ++i) {
        zeros += (p[i] == T{0});
      }

      if (_type == AMO_FetchTestType) {
        // fetch returns the current value (initially 0), dest unchanged
        if (zeros != cnt) fail_eq(zeros, cnt);
      } else { // AMO_SwapTestType
        // For a single element per (loop,elem), exactly one atomic_swap
        // observes old==0 (the first arriving swap). The rest see non-zero.
        if (zeros != 1) fail_eq(zeros, 1);
      }
    }
  }
}

template <typename T>
void AMOExtendedTester<T>::verifyResults(size_t /*size*/) {
  // PE 0 checks returns; target PE checks dest.
  if (args.myid) {
    verifyDestValues();
  } else {
    verifyReturnValues();
  }
}

#define AMO_EXTENDED_DEF_GEN(T, TNAME)                                        \
  template <>                                                                 \
  __global__ void AMOExtendedTest<T>(                                         \
      int loop, int skip, long long int *start_time,                          \
      long long int *end_time, T *dest, T *ret_val,                           \
      AddrMode addr_mode, TestType type, ShmemContextType ctx_type) {         \
    __shared__ rocshmem_ctx_t ctx;                                            \
    int wg_id     = get_flat_grid_id();                                       \
    int global_id = get_flat_id();                                            \
    int t_id      = get_flat_block_id();                                      \
    int n_threads = get_flat_grid_size();                                     \
    int n_wgs     = get_grid_num_blocks();                                    \
    rocshmem_wg_ctx_create(ctx_type, &ctx);                                   \
    for (int i = 0; i < loop + skip; i++) {                                   \
    T *ptr = compute_target_ptr<T>(dest, addr_mode, wg_id, i, n_wgs);         \
    T ret = 0;                                                                \
      if (i == skip) {                                                        \
        start_time[wg_id] = wall_clock64();                                   \
      }                                                                       \
      switch (type) {                                                         \
        case AMO_FetchTestType:                                               \
          ret = rocshmem_ctx_##TNAME##_atomic_fetch(ctx, ptr, 1);             \
          break;                                                              \
        case AMO_SetTestType:                                                 \
          rocshmem_ctx_##TNAME##_atomic_set(ctx, ptr, (T)17, 1);              \
          break;                                                              \
        case AMO_SwapTestType:                                                \
          ret = rocshmem_ctx_##TNAME##_atomic_swap(ctx, ptr, (T)(t_id + 1),1);\
          break;                                                              \
        default:                                                              \
          break;                                                              \
      }                                                                       \
    ret_val[global_id + i * n_threads] = ret;                                 \
    }                                                                         \
    rocshmem_ctx_quiet(ctx);                                                  \
    end_time[wg_id] = wall_clock64();                                         \
    __syncthreads();                                                          \
    rocshmem_wg_ctx_destroy(&ctx);                                            \
  }                                                                           \
  template class AMOExtendedTester<T>;

AMO_EXTENDED_DEF_GEN(float, float)
AMO_EXTENDED_DEF_GEN(double, double)
AMO_EXTENDED_DEF_GEN(int, int)
AMO_EXTENDED_DEF_GEN(long, long)
AMO_EXTENDED_DEF_GEN(long long, longlong)
AMO_EXTENDED_DEF_GEN(unsigned int, uint)
AMO_EXTENDED_DEF_GEN(unsigned long, ulong)
AMO_EXTENDED_DEF_GEN(unsigned long long, ulonglong)
