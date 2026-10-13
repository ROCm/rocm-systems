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
 * sizes (no message-size sweep). Both thread-level and workgroup-collective
 * variants are covered.
 *
 * Fixed shapes:
 *   3D: (D0=8, D1=16, D2=4)  — 512 elements per tile
 *   4D: (D0=2, D1=8, D2=8, D3=4) — 512 elements per tile
 *
 * Verification: flat linear fill (value = index % 256), flat linear check.
 */

#include "tile_nd_rma_tester.hpp"

#include <rocshmem/rocshmem.hpp>

// Include internal context types before the tile API implementations
#include "../../src/context_incl.hpp"

// Include tile API template implementations
#include <rocshmem/rocshmem_TILE_impl.hpp>

using namespace rocshmem;

/******************************************************************************
 * TENSOR / TUPLE TYPES
 *****************************************************************************/

template <typename T>
struct Tensor3D {
  using element_type = T;
  static constexpr int ndim = 3;

  T* data;
  int d0, d1, d2;

  __device__ Tensor3D(T* data_, int d0_, int d1_, int d2_)
      : data(data_), d0(d0_), d1(d1_), d2(d2_) {}

  __device__ T* data_handle() const { return data; }
  __device__ int stride(int dim) const {
    if (dim == 0) return d1 * d2;
    if (dim == 1) return d2;
    return 1;
  }
};

template <typename T>
struct Tensor4D {
  using element_type = T;
  static constexpr int ndim = 4;

  T* data;
  int d0, d1, d2, d3;

  __device__ Tensor4D(T* data_, int d0_, int d1_, int d2_, int d3_)
      : data(data_), d0(d0_), d1(d1_), d2(d2_), d3(d3_) {}

  __device__ T* data_handle() const { return data; }
  __device__ int stride(int dim) const {
    if (dim == 0) return d1 * d2 * d3;
    if (dim == 1) return d2 * d3;
    if (dim == 2) return d3;
    return 1;
  }
};

struct Tuple3D {
  int x, y, z;
  __device__ Tuple3D(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
  __device__ int get(int dim) const {
    if (dim == 0) return x;
    if (dim == 1) return y;
    return z;
  }
};

struct Tuple4D {
  int x, y, z, w;
  __device__ Tuple4D(int x_, int y_, int z_, int w_)
      : x(x_), y(y_), z(z_), w(w_) {}
  __device__ int get(int dim) const {
    if (dim == 0) return x;
    if (dim == 1) return y;
    if (dim == 2) return z;
    return w;
  }
};

/******************************************************************************
 * DEVICE KERNELS
 *****************************************************************************/

template <TestType Type>
__global__ void TileNDRMATest(long long int* start_time, long long int* end_time,
                               float* source, float* dest,
                               int loop, int skip,
                               ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  const int wg_id = get_flat_grid_id();

  // Each workgroup operates on its own slot in the buffer.
  // All test types use a fixed tile of TILE_ELEMS elements per slot.
  constexpr int D0_3D = TileNDRMATester::D0_3D;
  constexpr int D1_3D = TileNDRMATester::D1_3D;
  constexpr int D2_3D = TileNDRMATester::D2_3D;
  constexpr int D0_4D = TileNDRMATester::D0_4D;
  constexpr int D1_4D = TileNDRMATester::D1_4D;
  constexpr int D2_4D = TileNDRMATester::D2_4D;
  constexpr int D3_4D = TileNDRMATester::D3_4D;
  constexpr int TILE_ELEMS = TileNDRMATester::TILE_ELEMS;

  const int slot_offset = wg_id * TILE_ELEMS;

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && threadIdx.x == 0) {
      rocshmem_ctx_quiet(ctx);
      start_time[wg_id] = wall_clock64();
    }

    if constexpr (Type == TilePut3DTestType) {
      Tensor3D<float> src(source + slot_offset, D0_3D, D1_3D, D2_3D);
      Tensor3D<float> dst(dest   + slot_offset, D0_3D, D1_3D, D2_3D);
      Tuple3D start(0, 0, 0);
      Tuple3D boundary(D0_3D, D1_3D, D2_3D);
      rocshmem_ctx_tile_put(ctx, dst, src, start, boundary, 1, 0);
    } else if constexpr (Type == TileGet3DTestType) {
      Tensor3D<float> src(source + slot_offset, D0_3D, D1_3D, D2_3D);
      Tensor3D<float> dst(dest   + slot_offset, D0_3D, D1_3D, D2_3D);
      Tuple3D start(0, 0, 0);
      Tuple3D boundary(D0_3D, D1_3D, D2_3D);
      rocshmem_ctx_tile_get(ctx, dst, src, start, boundary, 1, 0);
    } else if constexpr (Type == TilePutWG3DTestType) {
      Tensor3D<float> src(source + slot_offset, D0_3D, D1_3D, D2_3D);
      Tensor3D<float> dst(dest   + slot_offset, D0_3D, D1_3D, D2_3D);
      Tuple3D start(0, 0, 0);
      Tuple3D boundary(D0_3D, D1_3D, D2_3D);
      rocshmem_ctx_tile_put_wg(ctx, dst, src, start, boundary, 1, 0);
    } else if constexpr (Type == TileGetWG3DTestType) {
      Tensor3D<float> src(source + slot_offset, D0_3D, D1_3D, D2_3D);
      Tensor3D<float> dst(dest   + slot_offset, D0_3D, D1_3D, D2_3D);
      Tuple3D start(0, 0, 0);
      Tuple3D boundary(D0_3D, D1_3D, D2_3D);
      rocshmem_ctx_tile_get_wg(ctx, dst, src, start, boundary, 1, 0);
    } else if constexpr (Type == TilePut4DTestType) {
      Tensor4D<float> src(source + slot_offset, D0_4D, D1_4D, D2_4D, D3_4D);
      Tensor4D<float> dst(dest   + slot_offset, D0_4D, D1_4D, D2_4D, D3_4D);
      Tuple4D start(0, 0, 0, 0);
      Tuple4D boundary(D0_4D, D1_4D, D2_4D, D3_4D);
      rocshmem_ctx_tile_put(ctx, dst, src, start, boundary, 1, 0);
    } else if constexpr (Type == TileGet4DTestType) {
      Tensor4D<float> src(source + slot_offset, D0_4D, D1_4D, D2_4D, D3_4D);
      Tensor4D<float> dst(dest   + slot_offset, D0_4D, D1_4D, D2_4D, D3_4D);
      Tuple4D start(0, 0, 0, 0);
      Tuple4D boundary(D0_4D, D1_4D, D2_4D, D3_4D);
      rocshmem_ctx_tile_get(ctx, dst, src, start, boundary, 1, 0);
    } else if constexpr (Type == TilePutWG4DTestType) {
      Tensor4D<float> src(source + slot_offset, D0_4D, D1_4D, D2_4D, D3_4D);
      Tensor4D<float> dst(dest   + slot_offset, D0_4D, D1_4D, D2_4D, D3_4D);
      Tuple4D start(0, 0, 0, 0);
      Tuple4D boundary(D0_4D, D1_4D, D2_4D, D3_4D);
      rocshmem_ctx_tile_put_wg(ctx, dst, src, start, boundary, 1, 0);
    } else if constexpr (Type == TileGetWG4DTestType) {
      Tensor4D<float> src(source + slot_offset, D0_4D, D1_4D, D2_4D, D3_4D);
      Tensor4D<float> dst(dest   + slot_offset, D0_4D, D1_4D, D2_4D, D3_4D);
      Tuple4D start(0, 0, 0, 0);
      Tuple4D boundary(D0_4D, D1_4D, D2_4D, D3_4D);
      rocshmem_ctx_tile_get_wg(ctx, dst, src, start, boundary, 1, 0);
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
  // Fixed tile size — no message sweep.
  const size_t total_threads = args.num_wgs * args.num_threads;
  const size_t num_elements  = static_cast<size_t>(total_threads) * TILE_ELEMS;

  // Put: local=source, remote=dest. Get: remote=source, local=dest.
  const bool is_put = (_type == TilePut3DTestType  || _type == TilePutWG3DTestType ||
                       _type == TilePut4DTestType  || _type == TilePutWG4DTestType);

  source_buf = static_cast<float*>(rocshmem_malloc(num_elements * sizeof(float)));
  dest_buf   = static_cast<float*>(rocshmem_malloc(num_elements * sizeof(float)));
  if (!source_buf || !dest_buf) {
    std::cerr << "TileNDRMATester: rocshmem_malloc failed\n";
    exit(EXIT_FAILURE);
  }

  if (is_put) {
    source = source_buf;
    dest   = dest_buf;
  } else {
    source = dest_buf;   // remote source
    dest   = source_buf; // local dest
  }

  // Flat linear fill: value[i] = i % 256
  std::vector<float> h_src(num_elements);
  for (size_t i = 0; i < num_elements; i++)
    h_src[i] = static_cast<float>(i % 256);
  CHECK_HIP(hipMemcpy(source, h_src.data(), num_elements * sizeof(float),
                      hipMemcpyHostToDevice));

  // Report a single fixed message size for bandwidth calculation.
  max_msg_size = static_cast<size_t>(TILE_ELEMS) * sizeof(float);
  this->args.min_msg_size = max_msg_size;
}

TileNDRMATester::~TileNDRMATester() {
  rocshmem_free(source_buf);
  rocshmem_free(dest_buf);
}

void TileNDRMATester::resetBuffers(uint64_t /*size*/) {
  const size_t total_threads = args.num_wgs * args.num_threads;
  const size_t num_elements  = static_cast<size_t>(total_threads) * TILE_ELEMS;

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
#define LAUNCH(TYPE)                                                       \
  hipLaunchKernelGGL(TileNDRMATest<TYPE##TestType>, gridSize, blockSize,  \
                     0, stream, start_time, end_time, source, dest,       \
                     loop, args.skip, _shmem_context)

  switch (_type) {
    case TilePut3DTestType:    LAUNCH(TilePut3D);    break;
    case TileGet3DTestType:    LAUNCH(TileGet3D);    break;
    case TilePutWG3DTestType:  LAUNCH(TilePutWG3D);  break;
    case TileGetWG3DTestType:  LAUNCH(TileGetWG3D);  break;
    case TilePut4DTestType:    LAUNCH(TilePut4D);    break;
    case TileGet4DTestType:    LAUNCH(TileGet4D);    break;
    case TilePutWG4DTestType:  LAUNCH(TilePutWG4D);  break;
    case TileGetWG4DTestType:  LAUNCH(TileGetWG4D);  break;
    default:
      std::cerr << "TileNDRMATester: unhandled TestType " << _type << "\n";
      exit(-1);
  }
#undef LAUNCH

  num_msgs       = (loop + args.skip) * args.num_wgs;
  num_timed_msgs = loop * args.num_wgs;
}

void TileNDRMATester::verifyResults(uint64_t /*size*/) {
  // For put: PE 1 receives into dest — verify on PE 1 (check_id=1).
  // For get: PE 0 receives into dest — verify on PE 0 (check_id=0).
  const bool is_put = (_type == TilePut3DTestType  || _type == TilePutWG3DTestType ||
                       _type == TilePut4DTestType  || _type == TilePutWG4DTestType);
  const int check_id = is_put ? 1 : 0;

  if (args.myid != check_id) return;

  const size_t num_wgs     = args.num_wgs;
  const size_t num_threads = args.num_threads;
  // For WG-collective ops each WG transfers one tile; for thread ops each
  // thread transfers one — but we launch one thread per WG for the nd tests.
  const size_t num_tiles = num_wgs;

  std::vector<float> h_dest(num_tiles * TILE_ELEMS);
  CHECK_HIP(hipMemcpy(h_dest.data(), dest,
                      num_tiles * TILE_ELEMS * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (size_t tile = 0; tile < num_tiles; tile++) {
    for (size_t i = 0; i < static_cast<size_t>(TILE_ELEMS); i++) {
      float expected = static_cast<float>(i % 256);
      float actual   = h_dest[tile * TILE_ELEMS + i];
      if (actual != expected) {
        std::cerr << "TileNDRMATester: verification failed"
                  << " tile=" << tile << " idx=" << i
                  << " got=" << actual << " expected=" << expected << "\n";
        exit(-1);
      }
    }
  }
}
