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

  // Calculate base offset for this thread/wave/wg's data region
  int matrix_size = tile_extent_0 * tile_extent_1;
  int offset;

  // For collective operations, all threads in the collective share the same tile
  // For thread-level operations, each thread has its own tile
  if constexpr (Type == TilePutWaveContiguousTestType ||
                Type == TileGetWaveContiguousTestType) {
    // Wave-collective: all threads in wave use same offset (wave ID)
    offset = matrix_size * (get_flat_id() / wf_size);
  } else if constexpr (Type == TilePutWGContiguousTestType ||
                       Type == TileGetWGContiguousTestType) {
    // Workgroup-collective: all threads in wg use same offset (workgroup ID)
    offset = matrix_size * get_flat_grid_id();
  } else {
    // Thread-level: each thread has its own offset
    offset = matrix_size * get_flat_id();
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
      } else if constexpr (Type == TileGetWaveContiguousTestType) {
        // Wave-collective get with contiguous layout
        Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1);
        Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1);
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
  // Allocate buffers for 64x64 tile (default test size)
  // For strided layouts, we need more space than just 64×64 elements
  size_t tile_size = 64 * 64;
  size_t buffer_elements_per_thread;

  switch (_type) {
    case TilePutRowMajorTestType:
    case TileGetRowMajorTestType:
      // Row stride = 2*64 = 128, need 64 rows * 128 stride
      buffer_elements_per_thread = 64 * 128;
      break;
    case TilePutColumnMajorTestType:
    case TileGetColumnMajorTestType:
      // Column-major: row_stride=1, col_stride=64
      // Need 64 cols * 64 col_stride = 4096 (same as contiguous)
      buffer_elements_per_thread = tile_size;
      break;
    case TilePutArbitraryTestType:
    case TileGetArbitraryTestType:
      // Arbitrary strides: row_stride=257, col_stride=3
      // Need 64 rows * 257 row_stride
      buffer_elements_per_thread = 64 * 257;
      break;
    default:
      // Contiguous and other cases: just 64×64
      buffer_elements_per_thread = tile_size;
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
    default:
      dest = local;
      source = remote;
      break;
  }

  // Initialize source buffer with pattern
  // For strided layouts, we need to initialize the tile elements correctly
  int tile_rows = 64;
  int tile_cols = 64;
  int row_stride, col_stride;

  switch (_type) {
    case TilePutRowMajorTestType:
    case TileGetRowMajorTestType:
      row_stride = 2 * 64;  // 128
      col_stride = 1;
      break;
    case TilePutColumnMajorTestType:
    case TileGetColumnMajorTestType:
      row_stride = 1;
      col_stride = 64;
      break;
    case TilePutArbitraryTestType:
    case TileGetArbitraryTestType:
      row_stride = 257;
      col_stride = 3;
      break;
    default:
      // Contiguous: row_stride = cols
      row_stride = tile_cols;
      col_stride = 1;
      break;
  }

  // Initialize with strided pattern for all threads
  for (size_t tile_id = 0; tile_id < total_threads; tile_id++) {
    size_t base_offset = tile_id * buffer_elements_per_thread;

    // Initialize each element in the 64×64 tile using stride pattern
    for (int row = 0; row < tile_rows; row++) {
      for (int col = 0; col < tile_cols; col++) {
        size_t tile_linear_idx = row * tile_cols + col;
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
  // Use the same buffer size calculation as constructor
  size_t tile_size = 64 * 64;
  size_t buffer_elements_per_thread;

  switch (_type) {
    case TilePutRowMajorTestType:
    case TileGetRowMajorTestType:
      buffer_elements_per_thread = 64 * 128;
      break;
    case TilePutColumnMajorTestType:
    case TileGetColumnMajorTestType:
      buffer_elements_per_thread = tile_size;
      break;
    case TilePutArbitraryTestType:
    case TileGetArbitraryTestType:
      buffer_elements_per_thread = 64 * 257;
      break;
    default:
      buffer_elements_per_thread = tile_size;
      break;
  }

  size_t total_threads = args.num_wgs * args.num_threads;
  size_t buff_size = buffer_elements_per_thread * total_threads * sizeof(float);
  memset(dest, 0, buff_size);
}

void TileRMATester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                 uint64_t size) {
  size_t shared_bytes = 0;

  // Default to 64x64 tiles
  int tile_extent_0 = 64;
  int tile_extent_1 = 64;

#define LAUNCH_TILE_RMA_TEST(SPECIFIC_TYPE)                                  \
  hipLaunchKernelGGL(TileRMATest<SPECIFIC_TYPE>, gridSize, blockSize,   \
                     shared_bytes, stream, loop, args.skip, start_time,      \
                     end_time, source, dest, tile_extent_0, tile_extent_1,   \
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
    default:
      std::cerr << "Invalid Test: unhandled TestType " << _type
                << " in TileRMATester::launchKernel" << std::endl;
      exit(-1);
  }

#undef LAUNCH_TILE_RMA_TEST

  num_msgs = (loop + args.skip) * gridSize.x * blockSize.x;
  num_timed_msgs = loop * gridSize.x * blockSize.x;
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
      check_id = 0;
      break;
    default:
      check_id = 1;
      break;
  }

  if (args.myid == check_id) {
    int tile_rows = 64;
    int tile_cols = 64;
    int row_stride, col_stride;

    // Determine strides based on test type
    switch (_type) {
      case TilePutRowMajorTestType:
      case TileGetRowMajorTestType:
        row_stride = 2 * 64;  // 128
        col_stride = 1;
        break;
      case TilePutColumnMajorTestType:
      case TileGetColumnMajorTestType:
        row_stride = 1;
        col_stride = 64;
        break;
      case TilePutArbitraryTestType:
      case TileGetArbitraryTestType:
        row_stride = 257;
        col_stride = 3;
        break;
      default:
        // Contiguous: row_stride = cols
        row_stride = tile_cols;
        col_stride = 1;
        break;
    }

    // Verify all tiles
    // For collective operations, the number of tiles transferred depends on the collective granularity
    size_t num_tiles_transferred;
    switch (_type) {
      case TilePutWaveContiguousTestType:
      case TileGetWaveContiguousTestType:
        // Wave-collective: one tile per wave
        num_tiles_transferred = (args.num_wgs * args.num_threads) / wf_size;
        break;
      case TilePutWGContiguousTestType:
      case TileGetWGContiguousTestType:
        // Workgroup-collective: one tile per workgroup
        num_tiles_transferred = args.num_wgs;
        break;
      default:
        // Thread-level: one tile per thread
        num_tiles_transferred = args.num_wgs * args.num_threads;
        break;
    }

    for (size_t tile_id = 0; tile_id < num_tiles_transferred; tile_id++) {
      // Calculate buffer offset for this thread
      size_t buffer_elements_per_thread;
      switch (_type) {
        case TilePutRowMajorTestType:
        case TileGetRowMajorTestType:
          buffer_elements_per_thread = 64 * 128;
          break;
        case TilePutColumnMajorTestType:
        case TileGetColumnMajorTestType:
          buffer_elements_per_thread = 64 * 64;
          break;
        case TilePutArbitraryTestType:
        case TileGetArbitraryTestType:
          buffer_elements_per_thread = 64 * 257;
          break;
        default:
          buffer_elements_per_thread = 64 * 64;
          break;
      }

      size_t base_offset = tile_id * buffer_elements_per_thread;

      // Verify each element in the 64×64 tile using stride pattern
      for (int row = 0; row < tile_rows; row++) {
        for (int col = 0; col < tile_cols; col++) {
          size_t tile_linear_idx = row * tile_cols + col;
          size_t buffer_idx = base_offset + row * row_stride + col * col_stride;

          // Expected value based on source initialization
          float expected = static_cast<float>(tile_linear_idx % 256);

          if (dest[buffer_idx] != expected) {
            std::cerr << "Data validation error at buffer idx " << buffer_idx
                      << " (tile pos [" << row << "," << col << "], tile_id=" << tile_id << ")" << std::endl;
            std::cerr << " Got " << dest[buffer_idx] << ", Expected " << expected << std::endl;
            exit(-1);
          }
        }
      }
    }
  }
}
