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

#include "tile_rma_tester.hpp"

#include <rocshmem/rocshmem.hpp>

// Include internal context types before the tile API implementations
#include "../../src/context_incl.hpp"

// Include tile API template implementations
#include <rocshmem/rocshmem_TILE_impl.hpp>

using namespace rocshmem;

/******************************************************************************
 * ROCSHMEM ALLOCATION WRAPPER
 *****************************************************************************/

template <typename T>
class rocshmemAllocation {
public:
    rocshmemAllocation() = default;
    ~rocshmemAllocation() { dealloc(); }

    void reset(size_t capacity) {
        dealloc();
        alloc(capacity * sizeof(T));
        _capacity = capacity;
    }

    T* get() { return _data; }
    size_t size() { return _capacity; }
    void free() { dealloc(); }

private:
    void dealloc() {
        if (_capacity) {
            rocshmem_free((void*)_data);
        }
        _capacity = 0;
    }

    void alloc(size_t size) {
        _data = (T*)rocshmem_malloc(size);
        if (!_data) {
            std::cerr << "rocshmem_malloc failed for " << size << " bytes\n";
            exit(EXIT_FAILURE);
        }
    }

    T* _data = nullptr;
    size_t _capacity = 0;
};

/******************************************************************************
 * TENSOR HELPERS
 *****************************************************************************/

// Simple 2D tensor implementation for testing
template <typename T>
struct Tensor2D {
  using element_type = T;
  static constexpr int ndim = 2;

  T* data;
  int stride_0;
  int stride_1;

  __device__ Tensor2D(T* data_, int stride_0_, int stride_1_)
      : data(data_), stride_0(stride_0_), stride_1(stride_1_) {}

  __device__ T* data_handle() const { return data; }
  __device__ int stride(int dim) const {
    return (dim == 0) ? stride_0 : stride_1;
  }
};

// Simple 1D tensor implementation for testing
template <typename T>
struct Tensor1D {
  using element_type = T;
  static constexpr int ndim = 1;

  T* data;
  int stride_0;

  __device__ Tensor1D(T* data_, int stride_0_)
      : data(data_), stride_0(stride_0_) {}

  __device__ T* data_handle() const { return data; }
  __device__ int stride(int dim) const { return stride_0; }
};

// Simple tuple for coordinates
struct Tuple2D {
  int x, y;
  __device__ Tuple2D(int x_, int y_) : x(x_), y(y_) {}
  __device__ int get(int dim) const { return (dim == 0) ? x : y; }
};

struct Tuple1D {
  int x;
  __device__ Tuple1D(int x_) : x(x_) {}
  __device__ int get(int dim) const { return x; }
};

/******************************************************************************
 * TEST KERNELS
 *****************************************************************************/

// Test type values come from TestType enum in tester.hpp

__global__ void TileRMATest(int loop, int skip, long long int *start_time,
                            long long int *end_time, float *source,
                            float *dest, int tile_extent_0, int tile_extent_1,
                            TestType test_type,
                            ShmemContextType ctx_type, int wf_size) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  int t_id = get_flat_block_id();
  int wf_id = t_id / wf_size;
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  __shared__ long long int wf_start_time[32];

  // Calculate base offset for this thread's data region
  int matrix_size = tile_extent_0 * tile_extent_1;
  int offset = matrix_size * get_flat_id();

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      __syncthreads();
      if (is_thread_zero_in_block()) {
        rocshmem_ctx_quiet(ctx);
      }
      __syncthreads();
      wf_start_time[wf_id] = wall_clock64();
    }

    switch (test_type) {
      case TilePutContiguousTestType: {
        // Fully contiguous: stride = [tile_extent_1, 1]
        Tensor2D<float> src_tensor(source + offset, tile_extent_1, 1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_1, 1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put(ctx, src_tensor, dst_tensor, start, boundary, 1, 0);
        break;
      }
      case TilePutRowMajorTestType: {
        // Row-major with gaps: stride = [2*tile_extent_1, 1]
        Tensor2D<float> src_tensor(source + offset, 2 * tile_extent_1, 1);
        Tensor2D<float> dst_tensor(dest + offset, 2 * tile_extent_1, 1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put(ctx, src_tensor, dst_tensor, start, boundary, 1, 0);
        break;
      }
      case TilePutColumnMajorTestType: {
        // Column-major: stride = [1, tile_extent_0]
        Tensor2D<float> src_tensor(source + offset, 1, tile_extent_0);
        Tensor2D<float> dst_tensor(dest + offset, 1, tile_extent_0);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put(ctx, src_tensor, dst_tensor, start, boundary, 1, 0);
        break;
      }
      case TilePutArbitraryTestType: {
        // Arbitrary strides: stride = [257, 3]
        Tensor2D<float> src_tensor(source + offset, 257, 3);
        Tensor2D<float> dst_tensor(dest + offset, 257, 3);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put(ctx, src_tensor, dst_tensor, start, boundary, 1, 0);
        break;
      }
      case TilePutWaveContiguousTestType: {
        // Wave-collective with contiguous layout
        Tensor2D<float> src_tensor(source + offset, tile_extent_1, 1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_1, 1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put_wave(ctx, src_tensor, dst_tensor, start, boundary, 1, 0);
        break;
      }
      case TilePutWGContiguousTestType: {
        // Workgroup-collective with contiguous layout
        Tensor2D<float> src_tensor(source + offset, tile_extent_1, 1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_1, 1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put_wg(ctx, src_tensor, dst_tensor, start, boundary, 1, 0);
        break;
      }
      case TileGetContiguousTestType: {
        // Thread-level get with contiguous layout
        Tensor2D<float> src_tensor(source + offset, tile_extent_1, 1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_1, 1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_get(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
        break;
      }
      case TileGetWGContiguousTestType: {
        // Workgroup-collective get with contiguous layout
        Tensor2D<float> src_tensor(source + offset, tile_extent_1, 1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_1, 1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_get_wg(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
        break;
      }
      case TileGetWaveContiguousTestType: {
        // Wave-collective get with contiguous layout
        Tensor2D<float> src_tensor(source + offset, tile_extent_1, 1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_1, 1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_get_wave(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
        break;
      }
      case TilePut1DTestType: {
        // 1D tensor put
        Tensor1D<float> src_tensor(source + offset, 1);
        Tensor1D<float> dst_tensor(dest + offset, 1);
        Tuple1D start(0);
        Tuple1D boundary(matrix_size);
        rocshmem_ctx_tile_put(ctx, src_tensor, dst_tensor, start, boundary, 1, 0);
        break;
      }
      case TileGet1DTestType: {
        // 1D tensor get
        Tensor1D<float> src_tensor(source + offset, 1);
        Tensor1D<float> dst_tensor(dest + offset, 1);
        Tuple1D start(0);
        Tuple1D boundary(matrix_size);
        rocshmem_ctx_tile_get(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
        break;
      }
      default:
        break;
    }
  }

  __syncthreads();
  if (is_thread_zero_in_block()) {
    rocshmem_ctx_quiet(ctx);
  }

  end_time[wg_id] = wall_clock64();

  // Find the earliest start time
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

TileRMATester::TileRMATester(TesterArguments args) : Tester(args) {
  // Allocate buffers for 64x64 tile (default test size)
  size_t tile_size = 64 * 64;
  size_t num_elements = tile_size * args.wg_size * args.num_wgs;

  // Allocate using rocshmem symmetric heap
  local_alloc = new rocshmemAllocation<float>();
  remote_alloc = new rocshmemAllocation<float>();

  local_alloc->reset(num_elements);
  remote_alloc->reset(num_elements);

  float *local = local_alloc->get();
  float *remote = remote_alloc->get();

  // For put operations, local is source, remote is dest
  // For get operations, remote is source, local is dest
  switch (_type) {
    case TilePutContiguousTestType:
    case TilePutRowMajorTestType:
    case TilePutColumnMajorTestType:
    case TilePutArbitraryTestType:
    case TilePutWaveContiguousTestType:
    case TilePutWGContiguousTestType:
    case TilePut1DTestType:
      source = local;
      dest = remote;
      break;
    case TileGetContiguousTestType:
    case TileGetWGContiguousTestType:
    case TileGetWaveContiguousTestType:
    case TileGet1DTestType:
    default:
      dest = local;
      source = remote;
      break;
  }

  // Initialize source buffer with pattern
  for (size_t i = 0; i < num_elements; i++) {
    source[i] = static_cast<float>(i % 256);
  }
}

TileRMATester::~TileRMATester() {
  if (local_alloc) {
    delete local_alloc;
    local_alloc = nullptr;
  }
  if (remote_alloc) {
    delete remote_alloc;
    remote_alloc = nullptr;
  }
}

void TileRMATester::resetBuffers(size_t size) {
  size_t tile_size = 64 * 64;
  size_t buff_size = tile_size * args.wg_size * args.num_wgs * sizeof(float);
  memset(dest, 0, buff_size);
}

void TileRMATester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                 size_t size) {
  size_t shared_bytes = 0;

  // Default to 64x64 tiles
  int tile_extent_0 = 64;
  int tile_extent_1 = 64;

  TestType test_type = _type;

  hipLaunchKernelGGL(TileRMATest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, source, dest,
                     tile_extent_0, tile_extent_1, test_type, _shmem_context,
                     wf_size);

  num_msgs = (loop + args.skip) * gridSize.x * blockSize.x;
  num_timed_msgs = loop * gridSize.x * blockSize.x;
}

void TileRMATester::verifyResults(size_t size) {
  int check_id;
  switch (_type) {
    case TileGetContiguousTestType:
    case TileGetWGContiguousTestType:
    case TileGetWaveContiguousTestType:
    case TileGet1DTestType:
      check_id = 0;
      break;
    default:
      check_id = 1;
      break;
  }

  if (args.myid == check_id) {
    size_t tile_size = 64 * 64;
    size_t total_elements = tile_size * args.wg_size * args.num_wgs;

    // Manual verification on host
    for (size_t i = 0; i < total_elements; i++) {
      float expected = static_cast<float>(i % 256);
      if (dest[i] != expected) {
        std::cerr << "Data validation error at idx " << i << std::endl;
        std::cerr << " Got " << dest[i] << ", Expected " << expected
                  << std::endl;
        exit(-1);
      }
    }
  }
}
