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

using namespace rocshmem;

/******************************************************************************
 * DEVICE-SIDE TYPED DISPATCH HELPERS
 *
 * One template specialization per (type, operation) pair.  The generic base
 * returns without doing anything so that the compiler is happy for test types
 * that do not apply to a given instantiation.
 *****************************************************************************/

/* --- put --- */
template <typename T>
__device__ void typed_put([[maybe_unused]] rocshmem_ctx_t ctx,
                          [[maybe_unused]] T *dest,
                          [[maybe_unused]] const T *source,
                          [[maybe_unused]] size_t nelems,
                          [[maybe_unused]] int pe) {}

#define TYPED_PUT_DEF(T, TNAME)                                               \
  template <>                                                                 \
  __device__ void typed_put<T>(rocshmem_ctx_t ctx, T *dest,                   \
                                const T *source, size_t nelems, int pe) {     \
    rocshmem_ctx_##TNAME##_put(ctx, dest, source, nelems, pe);                \
  }

TYPED_PUT_DEF(__half,          half)
TYPED_PUT_DEF(__hip_bfloat16,  bfloat16)

/* --- put_nbi --- */
template <typename T>
__device__ void typed_put_nbi([[maybe_unused]] rocshmem_ctx_t ctx,
                              [[maybe_unused]] T *dest,
                              [[maybe_unused]] const T *source,
                              [[maybe_unused]] size_t nelems,
                              [[maybe_unused]] int pe) {}

#define TYPED_PUT_NBI_DEF(T, TNAME)                                           \
  template <>                                                                 \
  __device__ void typed_put_nbi<T>(rocshmem_ctx_t ctx, T *dest,               \
                                    const T *source, size_t nelems, int pe) { \
    rocshmem_ctx_##TNAME##_put_nbi(ctx, dest, source, nelems, pe);            \
  }

TYPED_PUT_NBI_DEF(__half,         half)
TYPED_PUT_NBI_DEF(__hip_bfloat16, bfloat16)

/* --- get --- */
template <typename T>
__device__ void typed_get([[maybe_unused]] rocshmem_ctx_t ctx,
                          [[maybe_unused]] T *dest,
                          [[maybe_unused]] const T *source,
                          [[maybe_unused]] size_t nelems,
                          [[maybe_unused]] int pe) {}

#define TYPED_GET_DEF(T, TNAME)                                               \
  template <>                                                                 \
  __device__ void typed_get<T>(rocshmem_ctx_t ctx, T *dest,                   \
                                const T *source, size_t nelems, int pe) {     \
    rocshmem_ctx_##TNAME##_get(ctx, dest, source, nelems, pe);                \
  }

TYPED_GET_DEF(__half,          half)
TYPED_GET_DEF(__hip_bfloat16,  bfloat16)

/* --- get_nbi --- */
template <typename T>
__device__ void typed_get_nbi([[maybe_unused]] rocshmem_ctx_t ctx,
                              [[maybe_unused]] T *dest,
                              [[maybe_unused]] const T *source,
                              [[maybe_unused]] size_t nelems,
                              [[maybe_unused]] int pe) {}

#define TYPED_GET_NBI_DEF(T, TNAME)                                           \
  template <>                                                                 \
  __device__ void typed_get_nbi<T>(rocshmem_ctx_t ctx, T *dest,               \
                                    const T *source, size_t nelems, int pe) { \
    rocshmem_ctx_##TNAME##_get_nbi(ctx, dest, source, nelems, pe);            \
  }

TYPED_GET_NBI_DEF(__half,         half)
TYPED_GET_NBI_DEF(__hip_bfloat16, bfloat16)

/* --- p (scalar put) --- */
template <typename T>
__device__ void typed_p([[maybe_unused]] rocshmem_ctx_t ctx,
                        [[maybe_unused]] T *dest,
                        [[maybe_unused]] T value,
                        [[maybe_unused]] int pe) {}

#define TYPED_P_DEF(T, TNAME)                                                 \
  template <>                                                                 \
  __device__ void typed_p<T>(rocshmem_ctx_t ctx, T *dest,                     \
                              T value, int pe) {                               \
    rocshmem_ctx_##TNAME##_p(ctx, dest, value, pe);                           \
  }

TYPED_P_DEF(__half,         half)
TYPED_P_DEF(__hip_bfloat16, bfloat16)

/* --- g (scalar get) --- */
template <typename T>
__device__ T typed_g([[maybe_unused]] rocshmem_ctx_t ctx,
                     [[maybe_unused]] const T *source,
                     [[maybe_unused]] int pe) { return T{}; }

#define TYPED_G_DEF(T, TNAME)                                                 \
  template <>                                                                 \
  __device__ T typed_g<T>(rocshmem_ctx_t ctx, const T *source, int pe) {      \
    return rocshmem_ctx_##TNAME##_g(ctx, source, pe);                         \
  }

TYPED_G_DEF(__half,         half)
TYPED_G_DEF(__hip_bfloat16, bfloat16)

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
template <TestType Type, typename T>
__global__ void TypedRMATest(int loop, int skip, long long int *start_time,
                             long long int *end_time, T *source, T *dest,
                             size_t nelems, ShmemContextType ctx_type,
                             int wf_size, int batch, int *grid_psync) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  int t_id  = get_flat_block_id();
  int wf_id = t_id / wf_size;
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  __shared__ long long int wf_start_time[32];

  // Each thread owns `batch` contiguous slots of `nelems` elements.
  source += nelems * batch * get_flat_id();
  dest   += nelems * batch * get_flat_id();

  int start_slot = (batch - (skip % batch)) % batch;

  for (int i = 0; i < loop + skip; i++) {
    size_t offset = ((start_slot + i) % batch) * nelems;

    if (offset == 0) {
      __syncthreads();
      if (is_thread_zero_in_block()) {
        rocshmem_ctx_quiet(ctx);
      }
      __syncthreads();
      if (i == skip) {
        grid_barrier(grid_psync, gridDim.x);
        wf_start_time[wf_id] = wall_clock64();
      }
    }

    if constexpr (Type == PutTestType) {
      typed_put<T>(ctx, dest + offset, source + offset, nelems, 1);
    } else if constexpr (Type == PutNBITestType) {
      typed_put_nbi<T>(ctx, dest + offset, source + offset, nelems, 1);
    } else if constexpr (Type == GetTestType) {
      typed_get<T>(ctx, dest + offset, source + offset, nelems, 1);
    } else if constexpr (Type == GetNBITestType) {
      typed_get_nbi<T>(ctx, dest + offset, source + offset, nelems, 1);
    } else if constexpr (Type == PTestType) {
      for (size_t s = 0; s < nelems; s++) {
        typed_p<T>(ctx, dest + offset + s, source[offset + s], 1);
      }
    } else if constexpr (Type == GTestType) {
      for (size_t s = 0; s < nelems; s++) {
        dest[offset + s] = typed_g<T>(ctx, source + offset + s, 1);
      }
    }
  }

  __syncthreads();
  if (is_thread_zero_in_block()) {
    rocshmem_ctx_quiet(ctx);
  }

  end_time[wg_id] = wall_clock64();

  int num_wfs = (get_flat_block_size() - 1) / wf_size + 1;
  for (int i = num_wfs / 2; i > 0; i >>= 1) {
    if (t_id < i) {
      wf_start_time[t_id] = min(wf_start_time[t_id], wf_start_time[t_id + i]);
    }
  }
  __syncthreads();

  if (t_id == 0) {
    start_time[wg_id] = wf_start_time[0];
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
template <typename T>
TypedRMATester<T>::TypedRMATester(TesterArguments args) : Tester(args) {
  // Clamp to at least 1 element per slot so alloc_test_buffer never receives 0.
  // When size < sizeof(T) the framework iterates over message sizes starting
  // from 1 byte; launchKernel will skip those sizes via the nelems == 0 guard.
  size_t elems_per_slot = std::max(max_msg_size / sizeof(T), (size_t)1);
  size_t elem_count = elems_per_slot * batch_size * args.wg_size * args.num_wgs;
  T *local  = (T *)alloc_test_buffer(elem_count * sizeof(T),
                                     args.local_buf_type);
  T *remote = (T *)alloc_test_buffer(elem_count * sizeof(T));
  CHECK_HIP(hipMalloc(&grid_psync, sizeof(int)));

  int max_co_resident = 0;
  CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_co_resident, TypedRMATest<GetTestType, T>, args.wg_size, 0));
  const int max_sustainable = max_co_resident * deviceProps.multiProcessorCount;
  if (args.num_wgs > static_cast<unsigned>(max_sustainable)) {
    std::cerr << "Error: Requested work-groups (" << args.num_wgs
              << ") exceeds max co-resident work-groups (" << max_sustainable
              << "). Reduce -w to avoid grid_barrier deadlock." << std::endl;
    exit(-1);
  }

  switch (_type) {
    case PutTestType:
    case PutNBITestType:
    case PTestType:
      source = local;
      dest   = remote;
      break;
    default:
      dest   = local;
      source = remote;
      break;
  }

  T init_val = static_cast<T>(3.14f);
  for (size_t i = 0; i < elem_count; i++) source[i] = init_val;
}

template <typename T>
TypedRMATester<T>::~TypedRMATester() {
  T *local  = nullptr;
  T *remote = nullptr;

  switch (_type) {
    case PutTestType:
    case PutNBITestType:
    case PTestType:
      local  = source;
      remote = dest;
      break;
    default:
      local  = dest;
      remote = source;
      break;
  }

  free_test_buffer(local, args.local_buf_type);
  free_test_buffer(remote);
  CHECK_HIP(hipFree(grid_psync));
}

template <typename T>
void TypedRMATester<T>::resetBuffers(size_t size) {
  size_t nelems = size / sizeof(T);
  if (nelems > 0) {
    size_t elem_count = nelems * batch_size * args.wg_size * args.num_wgs;
    T init_val = static_cast<T>(1.0f);
    for (size_t i = 0; i < elem_count; i++) dest[i] = init_val;
  }
  CHECK_HIP(hipMemsetAsync(grid_psync, 0, sizeof(int), stream));
  CHECK_HIP(hipStreamSynchronize(stream));
}

template <typename T>
void TypedRMATester<T>::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                     size_t size) {
  size_t nelems = size / sizeof(T);
  if (nelems == 0) {
    num_msgs = num_timed_msgs = 0;
    return;
  }

  size_t shared_bytes = 0;

#define LAUNCH_TYPED_RMA(TYPE)                                                \
  hipLaunchKernelGGL((TypedRMATest<TYPE, T>), gridSize, blockSize,           \
                     shared_bytes, stream, loop, args.skip, start_time,       \
                     end_time, source, dest, nelems, _shmem_context,          \
                     wf_size, batch_size, grid_psync)

  switch (_type) {
    case PutTestType:      LAUNCH_TYPED_RMA(PutTestType);    break;
    case PutNBITestType:   LAUNCH_TYPED_RMA(PutNBITestType); break;
    case GetTestType:      LAUNCH_TYPED_RMA(GetTestType);    break;
    case GetNBITestType:   LAUNCH_TYPED_RMA(GetNBITestType); break;
    case PTestType:        LAUNCH_TYPED_RMA(PTestType);      break;
    case GTestType:        LAUNCH_TYPED_RMA(GTestType);      break;
    default:
      std::cerr << "TypedRMATester: unhandled TestType " << _type << std::endl;
      exit(-1);
  }

#undef LAUNCH_TYPED_RMA

  num_msgs       = (loop + args.skip) * gridSize.x * blockSize.x;
  num_timed_msgs = loop * gridSize.x * blockSize.x;
}

template <typename T>
void TypedRMATester<T>::verifyResults(size_t size) {
  size_t nelems = size / sizeof(T);
  if (nelems == 0) return;

  bool is_get = (_type == GetTestType || _type == GetNBITestType ||
                 _type == GTestType);
  if (args.myid != (is_get ? 0 : 1)) return;
  int start_slot   = (batch_size - (args.skip % batch_size)) % batch_size;
  int verify_iters = std::min(batch_size, num_loops + args.skip);
  size_t concurrency = args.wg_size * args.num_wgs;
  T expected = static_cast<T>(3.14f);

  for (size_t b = 0; b < concurrency; b++) {
    for (int iter = 0; iter < verify_iters; iter++) {
      int slot = (start_slot + iter) % batch_size;
      for (size_t i = 0; i < nelems; i++) {
        size_t idx = b * nelems * batch_size + slot * nelems + i;
        if (static_cast<float>(dest[idx]) != static_cast<float>(expected)) {
          std::cerr << "TypedRMA data validation error at buffer " << b
                    << " slot " << slot << " idx " << i << std::endl;
          std::cerr << " Got "      << static_cast<float>(dest[idx])
                    << ", Expected " << static_cast<float>(expected)
                    << std::endl;
          exit(-1);
        }
      }
    }
  }
}

// Explicit instantiations for the types used in tester.cpp
template class TypedRMATester<__half>;
template class TypedRMATester<__hip_bfloat16>;
