/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

/**
 * Regression test for symmetric start_coord / boundary application.
 *
 * The intended semantics (matching NVSHMEM) are:
 *   start_coord offsets BOTH src and dst into the same subregion of their
 *   respective tensors.
 *
 * Current rocshmem behaviour offsets only the remote tensor (dst for put,
 * src for get), leaving the local tensor anchored at element 0.
 *
 * Test design — put variant:
 *   Both src and dst are large matrices (ROWS x (2*TILE_COLS)) constructed at
 *   their global base.  The tile to transfer occupies columns [TILE_COLS,
 *   2*TILE_COLS), i.e. start_coord = (0, TILE_COLS), boundary = (ROWS, 2*TILE_COLS).
 *   The source matrix has a known pattern only in the right half; the left half
 *   is zeros.  With symmetric semantics the kernel reads src[start_coord..] and
 *   writes dst[start_coord..], so dst's right half matches src's right half.
 *   With the current asymmetric implementation the kernel reads src from element 0
 *   (the zero-filled left half) and writes that into dst's right half — producing
 *   all zeros instead of the expected pattern, which causes verifyResults to fail.
 *
 * Run with exactly 2 PEs:
 *   mpirun -n 2 ./tile_rma_start_coord_test
 */

#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>
#ifdef __HIP_DEVICE_COMPILE__
#include <rocshmem/rocshmem_device.hpp>
#endif

#include <iostream>
#include <cstring>
#include <vector>

using namespace rocshmem;

/******************************************************************************
 * Tensor / tuple types (same layout as existing examples)
 *****************************************************************************/

template <typename T>
struct WTensor2D {
  using element_type = T;
  static constexpr int ndim = 2;

  T* data;
  int rows;
  int cols;
  int row_stride;
  int col_stride;

  __host__ __device__ WTensor2D(T* data_, int rows_, int cols_,
                                int row_stride_ = -1, int col_stride_ = 1)
      : data(data_), rows(rows_), cols(cols_), col_stride(col_stride_) {
    row_stride = (row_stride_ == -1) ? cols : row_stride_;
  }

  __host__ __device__ T* data_handle() const { return data; }
  __host__ __device__ int stride(int dim) const {
    return (dim == 0) ? row_stride : col_stride;
  }
};

struct WCoord2D {
  int r, c;
  __host__ __device__ WCoord2D(int r_, int c_) : r(r_), c(c_) {}
  __host__ __device__ int get(int dim) const { return (dim == 0) ? r : c; }
};

/******************************************************************************
 * Constants
 *****************************************************************************/

static constexpr int ROWS      = 8;
static constexpr int TILE_COLS = 4;   // tile width; full matrix is 2*TILE_COLS wide
static constexpr int FULL_COLS = 2 * TILE_COLS;

/******************************************************************************
 * Kernel: thread-level tile_put with non-zero start_coord
 *
 * src and dst are both (ROWS x FULL_COLS) matrices passed at their global base.
 * We transfer the subregion [0..ROWS) x [TILE_COLS..FULL_COLS) using
 * start_coord=(0,TILE_COLS), boundary=(ROWS,FULL_COLS).
 *****************************************************************************/

__global__ void tile_put_start_coord_kernel(float* src, float* dst, int dst_pe) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_ctx_create(ROCSHMEM_CTX_WG_PRIVATE, &ctx);

  if (threadIdx.x == 0) {
    // Both tensors at global base with the full matrix stride
    WTensor2D<float> src_tensor(src, ROWS, FULL_COLS);
    WTensor2D<float> dst_tensor(dst, ROWS, FULL_COLS);

    WCoord2D start(0, TILE_COLS);
    WCoord2D boundary(ROWS, FULL_COLS);

    rocshmem_ctx_tile_put(ctx, dst_tensor, src_tensor, start, boundary, dst_pe, 0);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    rocshmem_ctx_quiet(ctx);
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

__global__ void tile_get_start_coord_kernel(float* dst, float* src, int src_pe) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_ctx_create(ROCSHMEM_CTX_WG_PRIVATE, &ctx);

  if (threadIdx.x == 0) {
    WTensor2D<float> dst_tensor(dst, ROWS, FULL_COLS);
    WTensor2D<float> src_tensor(src, ROWS, FULL_COLS);

    WCoord2D start(0, TILE_COLS);
    WCoord2D boundary(ROWS, FULL_COLS);

    rocshmem_ctx_tile_get(ctx, dst_tensor, src_tensor, start, boundary, src_pe, 0);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    rocshmem_ctx_quiet(ctx);
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * Helpers
 *****************************************************************************/

// Expected value for element (row, col) in the right-half tile.
// col here is the logical column index within the full matrix (TILE_COLS <= col < FULL_COLS).
static float expected_val(int row, int col) {
  return static_cast<float>(row * TILE_COLS + (col - TILE_COLS) + 1);
}

// Fill the right half of a host buffer (ROWS x FULL_COLS) with the pattern;
// leave the left half as zero.
static void fill_src(float* buf) {
  std::memset(buf, 0, ROWS * FULL_COLS * sizeof(float));
  for (int r = 0; r < ROWS; r++) {
    for (int c = TILE_COLS; c < FULL_COLS; c++) {
      buf[r * FULL_COLS + c] = expected_val(r, c);
    }
  }
}

// Verify that only the right half of dst carries the expected pattern,
// and the left half is still zero.
static bool verify(const float* buf, const char* label) {
  bool ok = true;
  for (int r = 0; r < ROWS; r++) {
    // Left half must stay zero (was not part of the transfer)
    for (int c = 0; c < TILE_COLS; c++) {
      if (buf[r * FULL_COLS + c] != 0.0f) {
        std::cerr << "[" << label << "] left-half contamination at (" << r << "," << c
                  << "): got " << buf[r * FULL_COLS + c] << ", expected 0\n";
        ok = false;
      }
    }
    // Right half must carry the pattern
    for (int c = TILE_COLS; c < FULL_COLS; c++) {
      float exp = expected_val(r, c);
      if (buf[r * FULL_COLS + c] != exp) {
        std::cerr << "[" << label << "] mismatch at (" << r << "," << c
                  << "): got " << buf[r * FULL_COLS + c] << ", expected " << exp << "\n";
        ok = false;
      }
    }
  }
  return ok;
}

/******************************************************************************
 * Main
 *****************************************************************************/

int main() {
  rocshmem_init();

  int my_pe = rocshmem_my_pe();
  int n_pes = rocshmem_n_pes();

  if (n_pes != 2) {
    if (my_pe == 0) {
      std::cerr << "This test requires exactly 2 PEs (got " << n_pes << ")\n";
    }
    rocshmem_finalize();
    return 1;
  }

  const size_t nelems = ROWS * FULL_COLS;
  const size_t nbytes = nelems * sizeof(float);

  // Symmetric buffers
  float* sym_buf_a = static_cast<float*>(rocshmem_malloc(nbytes));  // "A" buffer
  float* sym_buf_b = static_cast<float*>(rocshmem_malloc(nbytes));  // "B" buffer

  // Host staging
  std::vector<float> h_src(nelems), h_dst(nelems);

  bool all_ok = true;

  //---------------------------------------------------------------------------
  // TEST 1: tile_put  (PE 0 puts into PE 1's buffer)
  //---------------------------------------------------------------------------
  rocshmem_barrier_all();

  if (my_pe == 0) {
    // Fill sym_buf_a (source) with pattern in right half, zeros in left half
    fill_src(h_src.data());
    hipMemcpy(sym_buf_a, h_src.data(), nbytes, hipMemcpyHostToDevice);
    // Zero the destination on PE 0 side (not used for this put, but clean state)
    hipMemset(sym_buf_b, 0, nbytes);
  } else {
    // PE 1: zero both buffers — sym_buf_b will receive the put
    hipMemset(sym_buf_a, 0, nbytes);
    hipMemset(sym_buf_b, 0, nbytes);
  }

  rocshmem_barrier_all();

  if (my_pe == 0) {
    // PE 0 puts sym_buf_a[start_coord..boundary] -> PE 1's sym_buf_b[start_coord..boundary]
    tile_put_start_coord_kernel<<<1, 64>>>(sym_buf_a, sym_buf_b, /*dst_pe=*/1);
    hipDeviceSynchronize();
  }

  rocshmem_barrier_all();

  if (my_pe == 1) {
    // Verify PE 1's sym_buf_b
    hipMemcpy(h_dst.data(), sym_buf_b, nbytes, hipMemcpyDeviceToHost);
    if (!verify(h_dst.data(), "tile_put")) {
      std::cerr << "FAIL: tile_put with non-zero start_coord\n";
      all_ok = false;
    } else {
      std::cout << "PASS: tile_put with non-zero start_coord\n";
    }
  }

  //---------------------------------------------------------------------------
  // TEST 2: tile_get  (PE 0 gets from PE 1's buffer)
  //---------------------------------------------------------------------------
  rocshmem_barrier_all();

  if (my_pe == 1) {
    // Fill sym_buf_a (the source PE 0 will get from) with the pattern
    fill_src(h_src.data());
    hipMemcpy(sym_buf_a, h_src.data(), nbytes, hipMemcpyHostToDevice);
  }
  if (my_pe == 0) {
    // Zero PE 0's destination buffer
    hipMemset(sym_buf_b, 0, nbytes);
  }

  rocshmem_barrier_all();

  if (my_pe == 0) {
    // PE 0 gets PE 1's sym_buf_a[start_coord..boundary] -> local sym_buf_b[start_coord..boundary]
    tile_get_start_coord_kernel<<<1, 64>>>(sym_buf_b, sym_buf_a, /*src_pe=*/1);
    hipDeviceSynchronize();

    hipMemcpy(h_dst.data(), sym_buf_b, nbytes, hipMemcpyDeviceToHost);
    if (!verify(h_dst.data(), "tile_get")) {
      std::cerr << "FAIL: tile_get with non-zero start_coord\n";
      all_ok = false;
    } else {
      std::cout << "PASS: tile_get with non-zero start_coord\n";
    }
  }

  rocshmem_barrier_all();

  rocshmem_free(sym_buf_a);
  rocshmem_free(sym_buf_b);
  rocshmem_finalize();

  return all_ok ? 0 : 1;
}
