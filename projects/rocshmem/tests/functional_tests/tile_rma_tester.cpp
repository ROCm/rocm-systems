/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
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
class SymmetricTensorBuffer {
public:
    SymmetricTensorBuffer() = default;
    ~SymmetricTensorBuffer() { dealloc(); }

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
  int rows;
  int cols;
  int row_stride;
  int col_stride;

  __device__ Tensor2D(T* data_, int rows_, int cols_,
                      int row_stride_ = -1, int col_stride_ = 1)
      : data(data_), rows(rows_), cols(cols_), col_stride(col_stride_) {
    // Default row_stride is cols (contiguous row-major layout)
    row_stride = (row_stride_ == -1) ? cols : row_stride_;
  }

  __device__ T* data_handle() const { return data; }
  __device__ int stride(int dim) const {
    return (dim == 0) ? row_stride : col_stride;
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

template <TestType Type>
__global__ void TileRMATest(int loop, int skip, long long int *start_time,
                            long long int *end_time, float *source,
                            float *dest, int tile_extent_0, int tile_extent_1,
                            ShmemContextType ctx_type, int wf_size) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  int t_id = get_flat_block_id();
  int wf_id = t_id / wf_size;
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  __shared__ long long int wf_start_time[32];

  // Calculate base offset for this thread/wave/wg's data region.
  // The slot stride must match the host buffer allocation per slot:
  // - Contiguous / col-major / wave: matrix_size = tile_extent_0 * tile_extent_1
  // - Row-major (row_stride = 2 * tile_extent_1): each slot spans
  //   tile_extent_0 * (2 * tile_extent_1) elements
  int matrix_size = tile_extent_0 * tile_extent_1;
  int slot_stride;
  if constexpr (Type == TilePutRowMajorTestType    ||
                Type == TileGetRowMajorTestType     ||
                Type == TilePutWaveRowMajorTestType ||
                Type == TileGetWaveRowMajorTestType ||
                Type == TilePutWGRowMajorTestType   ||
                Type == TileGetWGRowMajorTestType) {
    slot_stride = tile_extent_0 * (2 * tile_extent_1);
  } else {
    slot_stride = matrix_size;
  }
  int offset;

  // For collective operations, all threads in the collective share the same tile
  // For thread-level operations, each thread has its own tile
  if constexpr (Type == TilePutWaveContiguousTestType ||
                Type == TileGetWaveContiguousTestType ||
                Type == TilePutWaveRowMajorTestType   ||
                Type == TilePutWaveColumnMajorTestType ||
                Type == TileGetWaveRowMajorTestType   ||
                Type == TileGetWaveColumnMajorTestType) {
    // Wave-collective: all threads in wave use same offset (wave ID)
    offset = slot_stride * (get_flat_id() / wf_size);
  } else if constexpr (Type == TilePutWGContiguousTestType ||
                       Type == TileGetWGContiguousTestType ||
                       Type == TilePutWGRowMajorTestType   ||
                       Type == TilePutWGColumnMajorTestType ||
                       Type == TileGetWGRowMajorTestType   ||
                       Type == TileGetWGColumnMajorTestType) {
    // Workgroup-collective: all threads in wg use same offset (workgroup ID)
    offset = slot_stride * get_flat_grid_id();
  } else {
    // Thread-level: each thread has its own offset
    offset = slot_stride * get_flat_id();
  }

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      __syncthreads();
      if (is_thread_zero_in_block()) {
        rocshmem_ctx_quiet(ctx);
      }
      __syncthreads();
      wf_start_time[wf_id] = wall_clock64();
    }

    if constexpr (Type == TilePutContiguousTestType) {
        // Fully contiguous: rows=tile_extent_0, cols=tile_extent_1, row_stride=tile_extent_1
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TilePutRowMajorTestType) {
        // Row-major with gaps: rows=tile_extent_0, cols=tile_extent_1, row_stride=2*tile_extent_1
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1, 2 * tile_extent_1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1, 2 * tile_extent_1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TilePutColumnMajorTestType) {
        // Column-major: rows=tile_extent_0, cols=tile_extent_1, row_stride=1, col_stride=tile_extent_0
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1, 1, tile_extent_0);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1, 1, tile_extent_0);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TilePutArbitraryTestType) {
        // Arbitrary strides: rows=tile_extent_0, cols=tile_extent_1, row_stride=257, col_stride=3
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1, 257, 3);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1, 257, 3);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TilePutWaveContiguousTestType) {
        // Wave-collective with contiguous layout
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put_wave(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TilePutWGContiguousTestType) {
        // Workgroup-collective with contiguous layout
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put_wg(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TilePutWGRowMajorTestType) {
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1, 2 * tile_extent_1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1, 2 * tile_extent_1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put_wg(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TilePutWGColumnMajorTestType) {
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1, 1, tile_extent_0);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1, 1, tile_extent_0);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put_wg(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TileGetContiguousTestType) {
        // Thread-level get with contiguous layout
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_get(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TileGetWGContiguousTestType) {
        // Workgroup-collective get with contiguous layout
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_get_wg(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TileGetWGRowMajorTestType) {
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1, 2 * tile_extent_1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1, 2 * tile_extent_1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_get_wg(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TileGetWGColumnMajorTestType) {
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1, 1, tile_extent_0);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1, 1, tile_extent_0);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_get_wg(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TileGetWaveContiguousTestType) {
        // Wave-collective get with contiguous layout
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_get_wave(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TilePutWaveRowMajorTestType) {
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1, 2 * tile_extent_1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1, 2 * tile_extent_1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put_wave(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TilePutWaveColumnMajorTestType) {
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1, 1, tile_extent_0);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1, 1, tile_extent_0);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_put_wave(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TileGetWaveRowMajorTestType) {
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1, 2 * tile_extent_1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1, 2 * tile_extent_1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_get_wave(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TileGetWaveColumnMajorTestType) {
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1, 1, tile_extent_0);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1, 1, tile_extent_0);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_get_wave(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TileGetRowMajorTestType) {
        // Row-major get with gaps: rows=tile_extent_0, cols=tile_extent_1, row_stride=2*tile_extent_1
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1, 2 * tile_extent_1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1, 2 * tile_extent_1);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_get(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TileGetColumnMajorTestType) {
        // Column-major get: rows=tile_extent_0, cols=tile_extent_1, row_stride=1, col_stride=tile_extent_0
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1, 1, tile_extent_0);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1, 1, tile_extent_0);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_get(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TileGetArbitraryTestType) {
        // Arbitrary strides get: rows=tile_extent_0, cols=tile_extent_1, row_stride=257, col_stride=3
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1, 257, 3);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1, 257, 3);
        Tuple2D start(0, 0);
        Tuple2D boundary(tile_extent_0, tile_extent_1);
        rocshmem_ctx_tile_get(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TilePut1DTestType) {
        // 1D tensor put
        Tensor1D<float> src_tensor(source + offset, 1);
        Tensor1D<float> dst_tensor(dest + offset, 1);
        Tuple1D start(0);
        Tuple1D boundary(matrix_size);
        rocshmem_ctx_tile_put(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
      } else if constexpr (Type == TileGet1DTestType) {
        // 1D tensor get
        Tensor1D<float> src_tensor(source + offset, 1);
        Tensor1D<float> dst_tensor(dest + offset, 1);
        Tuple1D start(0);
        Tuple1D boundary(matrix_size);
        rocshmem_ctx_tile_get(ctx, dst_tensor, src_tensor, start, boundary, 1, 0);
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
  // Derive tile dimensions from max_msg_size if provided.
  // max_msg_size is the total tile payload in bytes: tile_extent_0 * tile_extent_1 * sizeof(float).
  // tile_extent_0 (rows) is fixed at 64; tile_extent_1 (cols) scales with message size.
  tile_extent_0 = 64;
  tile_extent_1 = 64;
  if (args.max_msg_size_set) {
    int derived = static_cast<int>(args.max_msg_size / (tile_extent_0 * sizeof(float)));
    if (derived >= 1) {
      tile_extent_1 = derived;
    }
  }
  // For the arbitrary-stride layout (row_stride=257, col_stride=3), tile_extent_1
  // columns must fit within the row stride: tile_extent_1 * col_stride <= row_stride.
  if ((_type == TilePutArbitraryTestType || _type == TileGetArbitraryTestType) &&
      tile_extent_1 > 85) {
    tile_extent_1 = 85;  // floor(257 / 3)
  }

  // Set the message size sweep bounds.
  // For contiguous/wave/wg/1D layouts, each doubling step is a valid tile
  // (tile_extent_0 rows × increasing columns), so sweep from one row up to
  // the full tile. For strided layouts the buffer footprint depends on fixed
  // stride constants, so pin to a single point (the full tile size) for now.
  size_t tile_size_bytes =
      static_cast<size_t>(tile_extent_0) * tile_extent_1 * sizeof(float);
  size_t row_bytes = static_cast<size_t>(tile_extent_0) * sizeof(float);

  // Sweep from one column to the full tile for all layouts.
  // Strided layouts (row-major: stride=2*t1, col-major: col_stride=t0,
  // arbitrary: row_stride=257 fixed with t1 clamped to 85) are all safe
  // because the buffer is allocated for the maximum tile_extent_1 and
  // smaller ke_tile_extent_1 values only access a subset of it.
  this->args.min_msg_size = row_bytes;
  max_msg_size            = tile_size_bytes;

  // Buffer footprint per tile slot — accounts for strided layouts that need
  // more address space than a contiguous tile_extent_0 * tile_extent_1 region.
  size_t buffer_elements_per_thread;
  switch (_type) {
    case TilePutRowMajorTestType:
    case TileGetRowMajorTestType:
    case TilePutWaveRowMajorTestType:
    case TileGetWaveRowMajorTestType:
    case TilePutWGRowMajorTestType:
    case TileGetWGRowMajorTestType:
      // row_stride = 2 * tile_extent_1
      buffer_elements_per_thread = static_cast<size_t>(tile_extent_0) * (2 * tile_extent_1);
      break;
    case TilePutColumnMajorTestType:
    case TileGetColumnMajorTestType:
    case TilePutWaveColumnMajorTestType:
    case TileGetWaveColumnMajorTestType:
    case TilePutWGColumnMajorTestType:
    case TileGetWGColumnMajorTestType:
      // col_stride = tile_extent_0; contiguous in col direction
      buffer_elements_per_thread = static_cast<size_t>(tile_extent_0) * tile_extent_1;
      break;
    case TilePutArbitraryTestType:
    case TileGetArbitraryTestType:
      // row_stride = 257 (fixed constant), col_stride = 3
      buffer_elements_per_thread = static_cast<size_t>(tile_extent_0) * 257;
      break;
    default:
      // Contiguous, wave, wg, 1D: tile_extent_0 * tile_extent_1
      buffer_elements_per_thread = static_cast<size_t>(tile_extent_0) * tile_extent_1;
      break;
  }

  // For now, allocate one buffer per thread to keep it simple
  // TODO: Optimize to allocate only num_tiles (wave/wg count) for collective ops
  size_t total_threads = args.num_wgs * args.num_threads;
  size_t num_elements = buffer_elements_per_thread * total_threads;

  // Allocate using rocshmem symmetric heap
  local_alloc = new SymmetricTensorBuffer<float>();
  remote_alloc = new SymmetricTensorBuffer<float>();

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
    case TilePutWaveRowMajorTestType:
    case TilePutWaveColumnMajorTestType:
    case TilePutWGRowMajorTestType:
    case TilePutWGColumnMajorTestType:
      source = local;
      dest = remote;
      break;
    case TileGetContiguousTestType:
    case TileGetRowMajorTestType:
    case TileGetColumnMajorTestType:
    case TileGetArbitraryTestType:
    case TileGetWGContiguousTestType:
    case TileGetWaveContiguousTestType:
    case TileGet1DTestType:
    case TileGetWaveRowMajorTestType:
    case TileGetWaveColumnMajorTestType:
    case TileGetWGRowMajorTestType:
    case TileGetWGColumnMajorTestType:
    default:
      dest = local;
      source = remote;
      break;
  }

  // Initialize source buffer with pattern
  int row_stride, col_stride;
  switch (_type) {
    case TilePutRowMajorTestType:
    case TileGetRowMajorTestType:
      row_stride = 2 * tile_extent_1;
      col_stride = 1;
      break;
    case TilePutColumnMajorTestType:
    case TileGetColumnMajorTestType:
      row_stride = 1;
      col_stride = tile_extent_0;
      break;
    case TilePutArbitraryTestType:
    case TileGetArbitraryTestType:
      row_stride = 257;
      col_stride = 3;
      break;
    default:
      row_stride = tile_extent_1;
      col_stride = 1;
      break;
  }

  for (size_t tile_id = 0; tile_id < total_threads; tile_id++) {
    size_t base_offset = tile_id * buffer_elements_per_thread;
    for (int row = 0; row < tile_extent_0; row++) {
      for (int col = 0; col < tile_extent_1; col++) {
        size_t tile_linear_idx = row * tile_extent_1 + col;
        size_t buffer_idx = base_offset + row * row_stride + col * col_stride;
        source[buffer_idx] = static_cast<float>(tile_linear_idx % 256);
      }
    }
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

void TileRMATester::resetBuffers(uint64_t size) {
  size_t buffer_elements_per_thread;
  switch (_type) {
    case TilePutRowMajorTestType:
    case TileGetRowMajorTestType:
      buffer_elements_per_thread = static_cast<size_t>(tile_extent_0) * (2 * tile_extent_1);
      break;
    case TilePutArbitraryTestType:
    case TileGetArbitraryTestType:
      buffer_elements_per_thread = static_cast<size_t>(tile_extent_0) * 257;
      break;
    default:
      buffer_elements_per_thread = static_cast<size_t>(tile_extent_0) * tile_extent_1;
      break;
  }

  size_t total_threads = args.num_wgs * args.num_threads;
  size_t buff_size = buffer_elements_per_thread * total_threads * sizeof(float);
  memset(dest, 0, buff_size);

  // Re-initialize source using the current iteration's tile dimensions so that
  // tile_linear_idx and buffer offsets match exactly what the kernel expects.
  // The per-slot stride in the source must use the same ke_tile_extent_1 that
  // the kernel uses for its offset calculations.
  int t1 = tile_extent_1;
  if (size > 0) {
    int derived = static_cast<int>(size / (tile_extent_0 * sizeof(float)));
    if (derived >= 1 && derived <= tile_extent_1) t1 = derived;
  }
  if ((_type == TilePutArbitraryTestType || _type == TileGetArbitraryTestType) &&
      t1 > 85) {
    t1 = 85;
  }

  int src_row_stride, src_col_stride;
  size_t src_elements_per_slot;
  switch (_type) {
    case TilePutRowMajorTestType:
    case TileGetRowMajorTestType:
    case TilePutWaveRowMajorTestType:
    case TileGetWaveRowMajorTestType:
    case TilePutWGRowMajorTestType:
    case TileGetWGRowMajorTestType:
      src_row_stride = 2 * t1;
      src_col_stride = 1;
      src_elements_per_slot = static_cast<size_t>(tile_extent_0) * (2 * t1);
      break;
    case TilePutColumnMajorTestType:
    case TileGetColumnMajorTestType:
    case TilePutWaveColumnMajorTestType:
    case TileGetWaveColumnMajorTestType:
    case TilePutWGColumnMajorTestType:
    case TileGetWGColumnMajorTestType:
      src_row_stride = 1;
      src_col_stride = tile_extent_0;
      src_elements_per_slot = static_cast<size_t>(tile_extent_0) * t1;
      break;
    case TilePutArbitraryTestType:
    case TileGetArbitraryTestType:
      src_row_stride = 257;
      src_col_stride = 3;
      src_elements_per_slot = static_cast<size_t>(tile_extent_0) * 257;
      break;
    default:
      src_row_stride = t1;
      src_col_stride = 1;
      src_elements_per_slot = static_cast<size_t>(tile_extent_0) * t1;
      break;
  }

  for (size_t tile_id = 0; tile_id < total_threads; tile_id++) {
    size_t base_offset = tile_id * src_elements_per_slot;
    for (int row = 0; row < tile_extent_0; row++) {
      for (int col = 0; col < t1; col++) {
        size_t tile_linear_idx = row * t1 + col;
        size_t buffer_idx = base_offset + row * src_row_stride + col * src_col_stride;
        source[buffer_idx] = static_cast<float>(tile_linear_idx % 256);
      }
    }
  }
}

void TileRMATester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                 uint64_t size) {
  size_t shared_bytes = 0;

  // Derive tile dimensions for this iteration from the current message size so
  // that the reported bandwidth matches the actual bytes transferred.
  // Clamp to the member tile_extent_1 (the max allocated in the constructor)
  // to ensure we never address beyond the allocated buffer.
  int ke_tile_extent_0 = tile_extent_0;
  int ke_tile_extent_1 = tile_extent_1;
  if (size > 0) {
    int derived = static_cast<int>(size / (ke_tile_extent_0 * sizeof(float)));
    if (derived >= 1 && derived <= tile_extent_1) {
      ke_tile_extent_1 = derived;
    }
  }
  // Apply the same arbitrary-stride column clamp
  if ((_type == TilePutArbitraryTestType || _type == TileGetArbitraryTestType) &&
      ke_tile_extent_1 > 85) {
    ke_tile_extent_1 = 85;
  }

#define LAUNCH_TILE_RMA_TEST(SPECIFIC_TYPE)                                    \
  hipLaunchKernelGGL(TileRMATest<SPECIFIC_TYPE>, gridSize, blockSize,         \
                     shared_bytes, stream, loop, args.skip, start_time,        \
                     end_time, source, dest, ke_tile_extent_0, ke_tile_extent_1, \
                     _shmem_context, wf_size)

  switch (_type) {
    case TilePutContiguousTestType:
      LAUNCH_TILE_RMA_TEST(TilePutContiguousTestType);
      break;
    case TilePutRowMajorTestType:
      LAUNCH_TILE_RMA_TEST(TilePutRowMajorTestType);
      break;
    case TilePutColumnMajorTestType:
      LAUNCH_TILE_RMA_TEST(TilePutColumnMajorTestType);
      break;
    case TilePutArbitraryTestType:
      LAUNCH_TILE_RMA_TEST(TilePutArbitraryTestType);
      break;
    case TilePutWaveContiguousTestType:
      LAUNCH_TILE_RMA_TEST(TilePutWaveContiguousTestType);
      break;
    case TilePutWGContiguousTestType:
      LAUNCH_TILE_RMA_TEST(TilePutWGContiguousTestType);
      break;
    case TileGetContiguousTestType:
      LAUNCH_TILE_RMA_TEST(TileGetContiguousTestType);
      break;
    case TileGetWGContiguousTestType:
      LAUNCH_TILE_RMA_TEST(TileGetWGContiguousTestType);
      break;
    case TileGetWaveContiguousTestType:
      LAUNCH_TILE_RMA_TEST(TileGetWaveContiguousTestType);
      break;
    case TileGetRowMajorTestType:
      LAUNCH_TILE_RMA_TEST(TileGetRowMajorTestType);
      break;
    case TileGetColumnMajorTestType:
      LAUNCH_TILE_RMA_TEST(TileGetColumnMajorTestType);
      break;
    case TileGetArbitraryTestType:
      LAUNCH_TILE_RMA_TEST(TileGetArbitraryTestType);
      break;
    case TilePut1DTestType:
      LAUNCH_TILE_RMA_TEST(TilePut1DTestType);
      break;
    case TileGet1DTestType:
      LAUNCH_TILE_RMA_TEST(TileGet1DTestType);
      break;
    case TilePutWaveRowMajorTestType:
      LAUNCH_TILE_RMA_TEST(TilePutWaveRowMajorTestType);
      break;
    case TilePutWaveColumnMajorTestType:
      LAUNCH_TILE_RMA_TEST(TilePutWaveColumnMajorTestType);
      break;
    case TileGetWaveRowMajorTestType:
      LAUNCH_TILE_RMA_TEST(TileGetWaveRowMajorTestType);
      break;
    case TileGetWaveColumnMajorTestType:
      LAUNCH_TILE_RMA_TEST(TileGetWaveColumnMajorTestType);
      break;
    case TilePutWGRowMajorTestType:
      LAUNCH_TILE_RMA_TEST(TilePutWGRowMajorTestType);
      break;
    case TilePutWGColumnMajorTestType:
      LAUNCH_TILE_RMA_TEST(TilePutWGColumnMajorTestType);
      break;
    case TileGetWGRowMajorTestType:
      LAUNCH_TILE_RMA_TEST(TileGetWGRowMajorTestType);
      break;
    case TileGetWGColumnMajorTestType:
      LAUNCH_TILE_RMA_TEST(TileGetWGColumnMajorTestType);
      break;
    default:
      std::cerr << "Invalid Test: unhandled TestType " << _type
                << " in TileRMATester::launchKernel" << std::endl;
      exit(-1);
  }

#undef LAUNCH_TILE_RMA_TEST

  // Count tiles transferred, not threads: wave transfers one tile per wave,
  // wg transfers one tile per workgroup, thread transfers one tile per thread.
  size_t tiles_per_loop;
  switch (_type) {
    case TilePutWaveContiguousTestType:
    case TileGetWaveContiguousTestType:
    case TilePutWaveRowMajorTestType:
    case TilePutWaveColumnMajorTestType:
    case TileGetWaveRowMajorTestType:
    case TileGetWaveColumnMajorTestType:
      tiles_per_loop = gridSize.x * num_warps;
      break;
    case TilePutWGContiguousTestType:
    case TileGetWGContiguousTestType:
    case TilePutWGRowMajorTestType:
    case TilePutWGColumnMajorTestType:
    case TileGetWGRowMajorTestType:
    case TileGetWGColumnMajorTestType:
      tiles_per_loop = gridSize.x;
      break;
    default:
      tiles_per_loop = gridSize.x * blockSize.x;
      break;
  }
  num_msgs = (loop + args.skip) * tiles_per_loop;
  num_timed_msgs = loop * tiles_per_loop;

}

void TileRMATester::verifyResults(uint64_t size) {
  int check_id;
  switch (_type) {
    case TileGetContiguousTestType:
    case TileGetRowMajorTestType:
    case TileGetColumnMajorTestType:
    case TileGetArbitraryTestType:
    case TileGetWGContiguousTestType:
    case TileGetWaveContiguousTestType:
    case TileGet1DTestType:
    case TileGetWaveRowMajorTestType:
    case TileGetWaveColumnMajorTestType:
    case TileGetWGRowMajorTestType:
    case TileGetWGColumnMajorTestType:
      check_id = 0;
      break;
    default:
      check_id = 1;
      break;
  }

  if (args.myid == check_id) {
    // Re-derive tile dimensions from size using the same logic as launchKernel
    // so that we only verify the elements the kernel actually transferred.
    int t0 = tile_extent_0;
    int t1 = tile_extent_1;
    if (size > 0) {
      int derived = static_cast<int>(size / (t0 * sizeof(float)));
      if (derived >= 1 && derived <= tile_extent_1) {
        t1 = derived;
      }
    }
    if ((_type == TilePutArbitraryTestType || _type == TileGetArbitraryTestType) &&
        t1 > 85) {
      t1 = 85;
    }

    int row_stride, col_stride;
    switch (_type) {
      case TilePutRowMajorTestType:
      case TileGetRowMajorTestType:
      case TilePutWaveRowMajorTestType:
      case TileGetWaveRowMajorTestType:
      case TilePutWGRowMajorTestType:
      case TileGetWGRowMajorTestType:
        row_stride = 2 * t1;
        col_stride = 1;
        break;
      case TilePutColumnMajorTestType:
      case TileGetColumnMajorTestType:
      case TilePutWaveColumnMajorTestType:
      case TileGetWaveColumnMajorTestType:
      case TilePutWGColumnMajorTestType:
      case TileGetWGColumnMajorTestType:
        row_stride = 1;
        col_stride = t0;
        break;
      case TilePutArbitraryTestType:
      case TileGetArbitraryTestType:
        row_stride = 257;
        col_stride = 3;
        break;
      default:
        row_stride = t1;
        col_stride = 1;
        break;
    }

    size_t num_tiles_transferred;
    switch (_type) {
      case TilePutWaveContiguousTestType:
      case TileGetWaveContiguousTestType:
      case TilePutWaveRowMajorTestType:
      case TilePutWaveColumnMajorTestType:
      case TileGetWaveRowMajorTestType:
      case TileGetWaveColumnMajorTestType:
        num_tiles_transferred = (args.num_wgs * args.num_threads) / wf_size;
        break;
      case TilePutWGContiguousTestType:
      case TileGetWGContiguousTestType:
      case TilePutWGRowMajorTestType:
      case TilePutWGColumnMajorTestType:
      case TileGetWGRowMajorTestType:
      case TileGetWGColumnMajorTestType:
        num_tiles_transferred = args.num_wgs;
        break;
      default:
        num_tiles_transferred = args.num_wgs * args.num_threads;
        break;
    }

    // buffer_elements_per_thread must match the kernel's per-slot stride,
    // which is based on the MAX tile_extent_1 (the allocation granularity),
    // not the current iteration's t1. The kernel uses matrix_size * wg_id
    // where matrix_size = ke_tile_extent_0 * ke_tile_extent_1 = t0 * t1,
    // so we use t0 * t1 here to match.
    size_t buffer_elements_per_thread;
    switch (_type) {
      case TilePutRowMajorTestType:
      case TileGetRowMajorTestType:
      case TilePutWaveRowMajorTestType:
      case TileGetWaveRowMajorTestType:
      case TilePutWGRowMajorTestType:
      case TileGetWGRowMajorTestType:
        buffer_elements_per_thread = static_cast<size_t>(t0) * (2 * t1);
        break;
      case TilePutArbitraryTestType:
      case TileGetArbitraryTestType:
        buffer_elements_per_thread = static_cast<size_t>(t0) * 257;
        break;
      default:
        buffer_elements_per_thread = static_cast<size_t>(t0) * t1;
        break;
    }

    for (size_t tile_id = 0; tile_id < num_tiles_transferred; tile_id++) {
      size_t base_offset = tile_id * buffer_elements_per_thread;
      for (int row = 0; row < t0; row++) {
        for (int col = 0; col < t1; col++) {
          size_t tile_linear_idx = row * t1 + col;
          size_t buffer_idx = base_offset + row * row_stride + col * col_stride;
          float expected = static_cast<float>(tile_linear_idx % 256);
          if (dest[buffer_idx] != expected) {
            std::cerr << "Data validation error at buffer idx " << buffer_idx
                      << " (tile pos [" << row << "," << col << "], tile_id=" << tile_id << ")"
                      << std::endl;
            std::cerr << " Got " << dest[buffer_idx] << ", Expected " << expected << std::endl;
            exit(-1);
          }
        }
      }
    }
  }
}
