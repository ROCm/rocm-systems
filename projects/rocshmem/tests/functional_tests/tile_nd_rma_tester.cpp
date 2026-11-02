/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

/**
 * @file tile_nd_rma_tester.cpp
 * @brief Functional tests for 3D and 4D tile RMA (put/get) operations.
 *
 * Tests tile put and get for rank-3 and rank-4 tensors using fixed problem
 * sizes (no message-size sweep). Thread-level, wavefront-collective and
 * workgroup-collective variants are covered.
 *
 * Fixed shapes:
 *   3D: (D0=8, D1=16, D2=4)  — 512 elements per tile
 *   4D: (D0=2, D1=8, D2=8, D3=4) — 512 elements per tile
 *
 * Layouts:
 *   Contiguous: densely packed tensor, tile covers the whole tensor.
 *   Strided 3D: innermost dimension packed, rows padded to 2 * D2 elements
 *               and planes padded to 2 * D1 rows.
 *   Strided 4D: innermost stride of 2 elements (no contiguous runs) and
 *               dimension 1 padded to twice its packed stride.
 *
 * Verification: the source is filled with value = flat_index % 256 and the
 * destination is zeroed. Since source and destination share the same layout,
 * every element covered by the tile must match the source and every padding
 * element must remain zero.
 */

#include "tile_nd_rma_tester.hpp"

#include <rocshmem/rocshmem.hpp>

// Include internal context types before the tile API implementations
#include "../../src/context_incl.hpp"

// Include tile API template implementations
#include <rocshmem/rocshmem_TILE_impl.hpp>

using namespace rocshmem;

/******************************************************************************
 * TEST TYPE TABLE
 *****************************************************************************/

enum class TileNDScope { Thread, Wave, WG };

struct TileNDInfo {
  int ndim;
  bool is_put;
  TileNDScope scope;
  bool strided;
};

// X(name, ndim, is_put, scope, strided)
#define TILE_ND_FOREACH_TEST_TYPE(X)                          \
  X(TilePut3D,             3, true,  Thread, false)           \
  X(TileGet3D,             3, false, Thread, false)           \
  X(TilePutWG3D,           3, true,  WG,     false)           \
  X(TileGetWG3D,           3, false, WG,     false)           \
  X(TilePutWave3D,         3, true,  Wave,   false)           \
  X(TileGetWave3D,         3, false, Wave,   false)           \
  X(TilePut4D,             4, true,  Thread, false)           \
  X(TileGet4D,             4, false, Thread, false)           \
  X(TilePutWG4D,           4, true,  WG,     false)           \
  X(TileGetWG4D,           4, false, WG,     false)           \
  X(TilePutWave4D,         4, true,  Wave,   false)           \
  X(TileGetWave4D,         4, false, Wave,   false)           \
  X(TilePut3DStrided,      3, true,  Thread, true)            \
  X(TileGet3DStrided,      3, false, Thread, true)            \
  X(TilePutWG3DStrided,    3, true,  WG,     true)            \
  X(TileGetWG3DStrided,    3, false, WG,     true)            \
  X(TilePutWave3DStrided,  3, true,  Wave,   true)            \
  X(TileGetWave3DStrided,  3, false, Wave,   true)            \
  X(TilePut4DStrided,      4, true,  Thread, true)            \
  X(TileGet4DStrided,      4, false, Thread, true)            \
  X(TilePutWG4DStrided,    4, true,  WG,     true)            \
  X(TileGetWG4DStrided,    4, false, WG,     true)            \
  X(TilePutWave4DStrided,  4, true,  Wave,   true)            \
  X(TileGetWave4DStrided,  4, false, Wave,   true)

__host__ __device__ constexpr TileNDInfo tile_nd_info(TestType type) {
#define TILE_ND_INFO_CASE(NAME, NDIM, IS_PUT, SCOPE, STRIDED) \
  case NAME##TestType:                                        \
    return TileNDInfo{NDIM, IS_PUT, TileNDScope::SCOPE, STRIDED};
  switch (type) {
    TILE_ND_FOREACH_TEST_TYPE(TILE_ND_INFO_CASE)
    default:
      return TileNDInfo{0, false, TileNDScope::Thread, false};
  }
#undef TILE_ND_INFO_CASE
}

// Tensor extents and element strides for the given rank and layout.
__host__ __device__ inline void tile_nd_shape(int ndim, bool strided,
                                              int *dims, int *strides) {
  if (ndim == 3) {
    dims[0] = TileNDRMATester::D0_3D;
    dims[1] = TileNDRMATester::D1_3D;
    dims[2] = TileNDRMATester::D2_3D;
    strides[2] = 1;
    strides[1] = strided ? 2 * dims[2] : dims[2];
    strides[0] = (strided ? 2 : 1) * dims[1] * strides[1];
  } else {
    dims[0] = TileNDRMATester::D0_4D;
    dims[1] = TileNDRMATester::D1_4D;
    dims[2] = TileNDRMATester::D2_4D;
    dims[3] = TileNDRMATester::D3_4D;
    strides[3] = strided ? 2 : 1;
    strides[2] = dims[3] * strides[3];
    strides[1] = (strided ? 2 : 1) * dims[2] * strides[2];
    strides[0] = dims[1] * strides[1];
  }
}

__host__ __device__ constexpr int tile_nd_slot_elems(bool strided) {
  return strided ? TileNDRMATester::SLOT_ELEMS_STRIDED
                 : TileNDRMATester::TILE_ELEMS;
}

/******************************************************************************
 * TENSOR / TUPLE TYPES
 *****************************************************************************/

template <typename T, int N>
struct TensorND {
  using element_type = T;
  static constexpr int ndim = N;

  T* data;
  int s[N];

  __device__ TensorND(T* data_, const int* strides) : data(data_) {
    for (int d = 0; d < N; d++) s[d] = strides[d];
  }

  __device__ T* data_handle() const { return data; }
  __device__ int stride(int dim) const { return s[dim]; }
};

template <int N>
struct TupleND {
  int v[N];

  __device__ explicit TupleND(int fill) {
    for (int d = 0; d < N; d++) v[d] = fill;
  }
  __device__ explicit TupleND(const int* vals) {
    for (int d = 0; d < N; d++) v[d] = vals[d];
  }
  __device__ int get(int dim) const { return v[dim]; }
};

/******************************************************************************
 * DEVICE KERNELS
 *****************************************************************************/

template <TestType Type>
__global__ void TileNDRMATest(long long int* start_time, long long int* end_time,
                               float* source, float* dest,
                               int loop, int skip,
                               ShmemContextType ctx_type) {
  constexpr TileNDInfo info = tile_nd_info(Type);
  constexpr int N = info.ndim;
  static_assert(N == 3 || N == 4, "unhandled TestType");

  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  const int wg_id = get_flat_grid_id();
  const int slot_offset = wg_id * tile_nd_slot_elems(info.strided);

  int dims[N];
  int strides[N];
  tile_nd_shape(N, info.strided, dims, strides);

  TensorND<float, N> src(source + slot_offset, strides);
  TensorND<float, N> dst(dest + slot_offset, strides);
  TupleND<N> start(0);
  TupleND<N> boundary(dims);

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      // Drain all warm-up operations before starting the timer.
      __syncthreads();
      if (threadIdx.x == 0) {
        rocshmem_ctx_quiet(ctx);
        start_time[wg_id] = wall_clock64();
      }
      // All threads wait until lane 0 has recorded the start timestamp before
      // entering the timed tile operation — mirrors the existing tile RMA tester.
      __syncthreads();
    }

    if constexpr (info.is_put) {
      if constexpr (info.scope == TileNDScope::Thread) {
        rocshmem_ctx_tile_put(ctx, dst, src, start, boundary, 1, 0);
      } else if constexpr (info.scope == TileNDScope::Wave) {
        rocshmem_ctx_tile_put_wave(ctx, dst, src, start, boundary, 1, 0);
      } else {
        rocshmem_ctx_tile_put_wg(ctx, dst, src, start, boundary, 1, 0);
      }
    } else {
      if constexpr (info.scope == TileNDScope::Thread) {
        rocshmem_ctx_tile_get(ctx, dst, src, start, boundary, 1, 0);
      } else if constexpr (info.scope == TileNDScope::Wave) {
        rocshmem_ctx_tile_get_wave(ctx, dst, src, start, boundary, 1, 0);
      } else {
        rocshmem_ctx_tile_get_wg(ctx, dst, src, start, boundary, 1, 0);
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    rocshmem_ctx_quiet(ctx);
    end_time[wg_id] = wall_clock64();
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS
 *****************************************************************************/

TileNDRMATester::TileNDRMATester(TesterArguments args) : Tester(args) {
  const TileNDInfo info = tile_nd_info(_type);
  if (info.ndim == 0) {
    std::cerr << "TileNDRMATester: unhandled TestType " << _type << "\n";
    exit(EXIT_FAILURE);
  }

  // Fixed tile size — no message sweep.
  const size_t total_threads = args.num_wgs * args.num_threads;
  num_elements = total_threads * tile_nd_slot_elems(info.strided);

  source_buf = static_cast<float*>(rocshmem_malloc(num_elements * sizeof(float)));
  dest_buf   = static_cast<float*>(rocshmem_malloc(num_elements * sizeof(float)));
  if (!source_buf || !dest_buf) {
    std::cerr << "TileNDRMATester: rocshmem_malloc failed\n";
    exit(EXIT_FAILURE);
  }

  // Put: local=source, remote=dest. Get: remote=source, local=dest.
  if (info.is_put) {
    source = source_buf;
    dest   = dest_buf;
  } else {
    source = dest_buf;   // remote source
    dest   = source_buf; // local dest
  }

  resetBuffers(0);

  // Report a single fixed message size for bandwidth calculation.
  max_msg_size = static_cast<size_t>(TILE_ELEMS) * sizeof(float);
  this->args.min_msg_size = max_msg_size;
}

TileNDRMATester::~TileNDRMATester() {
  rocshmem_free(source_buf);
  rocshmem_free(dest_buf);
}

void TileNDRMATester::resetBuffers(uint64_t /*size*/) {
  // Zero dest; re-fill source with the flat linear pattern.
  CHECK_HIP(hipMemset(dest, 0, num_elements * sizeof(float)));

  std::vector<float> h_src(num_elements);
  for (size_t i = 0; i < num_elements; i++)
    h_src[i] = static_cast<float>(i % 256);
  CHECK_HIP(hipMemcpy(source, h_src.data(), num_elements * sizeof(float),
                      hipMemcpyHostToDevice));
}

void TileNDRMATester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                   uint64_t /*size*/) {
#define TILE_ND_LAUNCH_CASE(NAME, NDIM, IS_PUT, SCOPE, STRIDED)              \
  case NAME##TestType:                                                       \
    hipLaunchKernelGGL(TileNDRMATest<NAME##TestType>, gridSize, blockSize,   \
                       0, stream, start_time, end_time, source, dest, loop,  \
                       args.skip, _shmem_context);                           \
    break;

  switch (_type) {
    TILE_ND_FOREACH_TEST_TYPE(TILE_ND_LAUNCH_CASE)
    default:
      std::cerr << "TileNDRMATester: unhandled TestType " << _type << "\n";
      exit(-1);
  }
#undef TILE_ND_LAUNCH_CASE

  num_msgs       = (loop + args.skip) * args.num_wgs;
  num_timed_msgs = loop * args.num_wgs;
}

void TileNDRMATester::verifyResults(uint64_t /*size*/) {
  const TileNDInfo info = tile_nd_info(_type);

  // For put: PE 1 receives into dest — verify on PE 1 (check_id=1).
  // For get: PE 0 receives into dest — verify on PE 0 (check_id=0).
  const int check_id = info.is_put ? 1 : 0;
  if (args.myid != check_id) return;

  // One tile is transferred per workgroup.
  const size_t num_tiles  = args.num_wgs;
  const size_t slot_elems = tile_nd_slot_elems(info.strided);

  int dims[4];
  int strides[4];
  tile_nd_shape(info.ndim, info.strided, dims, strides);

  std::vector<bool> in_tile(slot_elems, false);
  for (int idx = 0; idx < TILE_ELEMS; idx++) {
    size_t off = 0;
    int rem = idx;
    for (int d = info.ndim - 1; d >= 0; d--) {
      off += static_cast<size_t>(rem % dims[d]) * strides[d];
      rem /= dims[d];
    }
    in_tile[off] = true;
  }

  std::vector<float> h_dest(num_tiles * slot_elems);
  CHECK_HIP(hipMemcpy(h_dest.data(), dest, h_dest.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (size_t tile = 0; tile < num_tiles; tile++) {
    for (size_t i = 0; i < slot_elems; i++) {
      const size_t flat = tile * slot_elems + i;
      const float expected = in_tile[i] ? static_cast<float>(flat % 256) : 0.0f;
      const float actual   = h_dest[flat];
      if (actual != expected) {
        std::cerr << "TileNDRMATester: verification failed"
                  << " tile=" << tile << " idx=" << i
                  << (in_tile[i] ? "" : " (padding)")
                  << " got=" << actual << " expected=" << expected << "\n";
        exit(-1);
      }
    }
  }
}
