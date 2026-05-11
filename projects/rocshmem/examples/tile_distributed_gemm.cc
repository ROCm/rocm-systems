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

/**
 * @file tile_distributed_gemm.cc
 * @brief Example demonstrating distributed GEMM using rocSHMEM Tile API
 *
 * This example shows how to use the rocSHMEM Tile API with tensor-aware
 * data structures for distributed matrix multiplication. The example uses
 * a simple tensor wrapper compatible with AMD Composable Kernels (CK) design.
 *
 * Algorithm: 1D block row decomposition
 * - Matrix A (M×K) is distributed row-wise across PEs
 * - Matrix B (K×N) is replicated on all PEs
 * - Matrix C (M×N) is distributed row-wise matching A
 * - Each PE computes its local block: C_local = A_local × B
 *
 * Communication pattern:
 * - Uses tile_get to fetch remote matrix tiles during computation
 * - Demonstrates tensor-aware RMA with stride handling
 */

#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>

// For device code using Tile API
#ifdef __HIP_DEVICE_COMPILE__
#include "../../src/context_incl.hpp"
#include <rocshmem/rocshmem_TILE_impl.hpp>
#endif

#include <iostream>
#include <vector>
#include <cmath>

#include "util.h"  // For get_launcher_local_rank()

using namespace rocshmem;

/******************************************************************************
 * TENSOR WRAPPER - Compatible with AMD Composable Kernels design patterns
 *****************************************************************************/

/**
 * @brief Simple 2D tensor descriptor for use with rocSHMEM Tile API
 *
 * This tensor type satisfies the interface required by the Tile API:
 * - element_type: type alias for the element type
 * - ndim: static constexpr for dimensionality
 * - data_handle(): returns pointer to data
 * - stride(int): returns stride for a given dimension
 *
 * This design is compatible with AMD Composable Kernels (CK) tensor patterns
 * and can be easily adapted to use CK's StaticTensor or other tensor libraries.
 */
template <typename T>
struct Tensor2D {
  using element_type = T;
  static constexpr int ndim = 2;

  T* data;
  int rows;
  int cols;
  int row_stride;  // Stride in elements (typically cols for row-major)
  int col_stride;  // Stride in elements (typically 1 for row-major)

  /**
   * @brief Construct a row-major 2D tensor
   * @param data_ Pointer to data buffer
   * @param rows_ Number of rows
   * @param cols_ Number of columns
   * @param row_stride_ Stride between rows (default: cols for contiguous)
   */
  __host__ __device__ Tensor2D(T* data_, int rows_, int cols_,
                                int row_stride_ = -1, int col_stride_ = 1)
      : data(data_), rows(rows_), cols(cols_), col_stride(col_stride_) {
    row_stride = (row_stride_ == -1) ? cols : row_stride_;
  }

  __host__ __device__ T* data_handle() const { return data; }
  __host__ __device__ int stride(int dim) const {
    return (dim == 0) ? row_stride : col_stride;
  }

  __host__ __device__ T& operator()(int i, int j) {
    return data[i * row_stride + j * col_stride];
  }

  __host__ __device__ const T& operator()(int i, int j) const {
    return data[i * row_stride + j * col_stride];
  }
};

/**
 * @brief Coordinate tuple for 2D indexing
 */
struct Coord2D {
  int row, col;
  __host__ __device__ Coord2D(int r, int c) : row(r), col(c) {}
  __host__ __device__ int get(int dim) const { return (dim == 0) ? row : col; }
};

/******************************************************************************
 * GPU KERNELS
 *****************************************************************************/

/**
 * @brief GPU kernel for local matrix multiplication using tensor objects
 *
 * Each thread block computes a tile of the output matrix C.
 * Demonstrates how tensor descriptors work with GPU computation.
 *
 * @param A_tensor Local portion of matrix A as Tensor2D
 * @param B_tensor Full matrix B as Tensor2D
 * @param C_tensor Local portion of matrix C as Tensor2D
 */
__global__ void gemm_kernel(Tensor2D<int> A_tensor,
                           Tensor2D<int> B_tensor,
                           Tensor2D<int> C_tensor) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < A_tensor.rows && col < B_tensor.cols) {
    int sum = 0;
    for (int k = 0; k < A_tensor.cols; k++) {
      sum += A_tensor(row, k) * B_tensor(k, col);
    }
    C_tensor(row, col) = sum;
  }
}

/**
 * @brief GPU kernel demonstrating tile-based data exchange
 *
 * This kernel uses the rocSHMEM Tile API to send a tile to a remote PE.
 * It demonstrates tensor-aware RMA with proper stride handling.
 *
 * @param local_tensor Source tensor on local PE
 * @param remote_buffer Remote buffer to receive the tile
 * @param tile_row Starting row of the tile
 * @param tile_col Starting column of the tile
 * @param tile_rows Number of rows in the tile
 * @param tile_cols Number of columns in the tile
 * @param remote_pe PE ID to send to
 */
__global__ void tile_exchange_kernel(Tensor2D<int> local_tensor,
                                    Tensor2D<int> remote_buffer,
                                    int tile_row, int tile_col,
                                    int tile_rows, int tile_cols,
                                    int remote_pe) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Define source tile on local PE
    Coord2D start(tile_row, tile_col);
    Coord2D boundary(tile_row + tile_rows, tile_col + tile_cols);

    // Send tile using Tile API
    // Note: tile_put uses (src, dst) parameter order
    rocshmem_tile_put(local_tensor, remote_buffer, start, boundary,
                     remote_pe, 0);
  }
}

/**
 * @brief GPU kernel demonstrating tile_get
 *
 * This kernel uses the rocSHMEM Tile API to fetch a tile from a remote PE.
 * It demonstrates tensor-aware RMA with proper stride handling.
 *
 * @param local_buffer Local buffer to receive the tile
 * @param remote_tensor Source tensor on remote PE
 * @param tile_row Starting row of the tile
 * @param tile_col Starting column of the tile
 * @param tile_rows Number of rows in the tile
 * @param tile_cols Number of columns in the tile
 * @param remote_pe PE ID to fetch from
 */
__global__ void tile_get_kernel(Tensor2D<int> local_buffer,
                                Tensor2D<int> remote_tensor,
                                int tile_row, int tile_col,
                                int tile_rows, int tile_cols,
                                int remote_pe) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Define source tile on remote PE
    Coord2D start(tile_row, tile_col);
    Coord2D boundary(tile_row + tile_rows, tile_col + tile_cols);

    // Fetch tile using Tile API
    // Note: tile_get uses (dst, src) parameter order
    rocshmem_tile_get(local_buffer, remote_tensor, start, boundary,
                     remote_pe, 0);
  }
}

/******************************************************************************
 * HOST CODE
 *****************************************************************************/

/**
 * @brief Initialize matrix with test pattern (integers for easy verification)
 * @param mat Matrix to initialize
 * @param rows Number of rows
 * @param cols Number of columns
 * @param row_offset Global row offset (for distributed matrices)
 * @param pe_id PE identifier to make each PE's data unique
 */
void init_matrix(std::vector<int>& mat, int rows, int cols,
                 int row_offset, int pe_id) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      // Use global row index and PE ID to make each PE's data unique
      // Format: (global_row * 1000) + col + (PE_ID * 100000)
      int global_row = row_offset + i;
      mat[i * cols + j] = (global_row * 1000) + j + (pe_id * 100000);
    }
  }
}

/**
 * @brief Verify distributed GEMM result
 */
bool verify_result(const std::vector<int>& C_local,
                  const std::vector<int>& A_local,
                  const std::vector<int>& B,
                  int M_local, int K, int N) {
  int errors = 0;
  for (int i = 0; i < M_local; i++) {
    for (int j = 0; j < N; j++) {
      long long expected = 0;
      for (int k = 0; k < K; k++) {
        expected += (long long)A_local[i * K + k] * B[k * N + j];
      }
      int actual = C_local[i * N + j];
      if (expected != actual) {
        if (errors < 5) {  // Only print first 5 errors
          std::cerr << "Mismatch at (" << i << "," << j << "): "
                    << "expected " << expected << ", got " << actual << std::endl;
        }
        errors++;
      }
    }
  }
  if (errors > 5) {
    std::cerr << "... and " << (errors - 5) << " more errors" << std::endl;
  }
  return (errors == 0);
}

int main(int argc, char* argv[]) {
  // Set GPU device based on local rank (ensures each PE uses a different GPU)
  // This MUST be done before rocshmem_init()
  CHECK_HIP(hipSetDevice(get_launcher_local_rank()));

  // Initialize rocSHMEM
  rocshmem_init();

  int my_pe = rocshmem_my_pe();
  int n_pes = rocshmem_n_pes();

  // Matrix dimensions (can be customized via command line)
  int M = (argc > 1) ? atoi(argv[1]) : 128;  // Total rows of A
  int K = (argc > 2) ? atoi(argv[2]) : 64;   // Columns of A / Rows of B
  int N = (argc > 3) ? atoi(argv[3]) : 96;   // Columns of B

  // Ensure M is divisible by number of PEs
  if (M % n_pes != 0) {
    if (my_pe == 0) {
      std::cerr << "M (" << M << ") must be divisible by n_pes ("
                << n_pes << ")" << std::endl;
    }
    rocshmem_finalize();
    return 1;
  }

  int M_local = M / n_pes;  // Rows per PE

  if (my_pe == 0) {
    std::cout << "Distributed GEMM: C(" << M << "×" << N << ") = "
              << "A(" << M << "×" << K << ") × B(" << K << "×" << N << ")\n";
    std::cout << "Using " << n_pes << " PEs, "
              << M_local << " rows per PE\n" << std::endl;
  }

  // Allocate host memory
  std::vector<int> h_A_local(M_local * K);
  std::vector<int> h_B(K * N);
  std::vector<int> h_C_local(M_local * N);

  // Initialize matrices with unique values per PE
  // A_local: each PE gets different row values based on its global row offset
  int global_row_offset = my_pe * M_local;
  init_matrix(h_A_local, M_local, K, global_row_offset, my_pe);
  // B: all PEs initialize the same B matrix (replicated)
  init_matrix(h_B, K, N, 0, 0);

  if (my_pe == 0) {
    std::cout << "Sample values from PE 0's A matrix:" << std::endl;
    std::cout << "  A[0,0] = " << h_A_local[0] << std::endl;
    std::cout << "  A[0,1] = " << h_A_local[1] << std::endl;
    std::cout << "  A[1,0] = " << h_A_local[K] << std::endl;
  }

  // Allocate symmetric heap for matrices
  // WORKAROUND: Allocate a dummy buffer first to avoid the first-allocation issue
  int* dummy = (int*)rocshmem_malloc(32);  // 32 bytes dummy allocation
  int* d_A_local = (int*)rocshmem_malloc(M_local * K * sizeof(int));
  int* d_B = (int*)rocshmem_malloc(K * N * sizeof(int));
  int* d_C_local = (int*)rocshmem_malloc(M_local * N * sizeof(int));

  if (!d_A_local || !d_B || !d_C_local) {
    std::cerr << "PE " << my_pe << ": rocshmem_malloc failed" << std::endl;
    rocshmem_finalize();
    return 1;
  }

  // Copy data to device
  hipMemcpy(d_A_local, h_A_local.data(), M_local * K * sizeof(int),
           hipMemcpyHostToDevice);
  hipMemcpy(d_B, h_B.data(), K * N * sizeof(int), hipMemcpyHostToDevice);

  // Ensure all PEs have initialized their data
  rocshmem_barrier_all();

  // Create tensor descriptors for GEMM
  Tensor2D<int> A_tensor(d_A_local, M_local, K);  // Local rows of A
  Tensor2D<int> B_tensor(d_B, K, N);              // Full B matrix (replicated)
  Tensor2D<int> C_tensor(d_C_local, M_local, N);  // Local rows of C

  // Launch GEMM kernel with tensor objects
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M_local + block.y - 1) / block.y);

  hipLaunchKernelGGL(gemm_kernel, grid, block, 0, 0,
                    A_tensor, B_tensor, C_tensor);

  hipDeviceSynchronize();

  // Synchronize all PEs
  rocshmem_barrier_all();

  // Copy result back to host
  hipMemcpy(h_C_local.data(), d_C_local, M_local * N * sizeof(int),
           hipMemcpyDeviceToHost);

  // Verify result
  bool success = verify_result(h_C_local, h_A_local, h_B, M_local, K, N);

  if (my_pe == 0) {
    if (success) {
      std::cout << "\n✓ Verification PASSED" << std::endl;
    } else {
      std::cout << "\n✗ Verification FAILED" << std::endl;
    }
  }

  // Example: Demonstrate Tile API for sending data to remote PE
  // (Not needed for this GEMM, but shows usage)
  if (n_pes > 1) {
    // Allocate buffer for tile on both PEs
    int* d_tile_buffer = (int*)rocshmem_malloc(16 * sizeof(int));

    if (my_pe == 1) {
      // PE 1 sends a 4×4 tile to PE 0
      std::cout << "\n--- Tile API Demo ---" << std::endl;
      std::cout << "PE 1 sending a 4×4 tile to PE 0's buffer..." << std::endl;
      std::cout << "PE 1's d_A_local = " << d_A_local << std::endl;
      std::cout << "PE 1's d_B = " << d_B << std::endl;
      std::cout << "PE 1's d_C_local = " << d_C_local << std::endl;

      if (M_local >= 4 && K >= 4) {
        // Create tensor descriptors
        Tensor2D<int> local_A(d_A_local, M_local, K);  // PE 1's A matrix
        Tensor2D<int> tile_buf(d_tile_buffer, 4, 4);   // 4×4 tile buffer on PE 0

        // Send tile to PE 0 (rows 0-3, cols 0-3 of local matrix)
        hipLaunchKernelGGL(tile_exchange_kernel, 1, 1, 0, 0,
                          local_A, tile_buf, 0, 0, 4, 4, 0);
        hipDeviceSynchronize();
      }
    }

    rocshmem_barrier_all();

    if (my_pe == 0) {
      std::cout << "PE 0 receiving tile from PE 1..." << std::endl;

      // Copy tile back to host for verification
      std::vector<int> h_tile_buffer(16);
      hipMemcpy(h_tile_buffer.data(), d_tile_buffer, 16 * sizeof(int),
               hipMemcpyDeviceToHost);

      // Verify: the tile should match PE 1's A matrix values
      // PE 1's global row offset is M_local
      int pe1_row_offset = 1 * M_local;
      int pe1_id = 1;

      bool tile_correct = true;
      std::cout << "\nExpected vs Actual values for 4×4 tile:" << std::endl;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          // Expected value from PE 1's initialization
          // Using same formula as init_matrix: (global_row * 1000) + col + (PE_ID * 100000)
          int global_row = pe1_row_offset + i;
          int expected = (global_row * 1000) + j + (pe1_id * 100000);
          int actual = h_tile_buffer[i * 4 + j];

          std::cout << "  [" << i << "," << j << "] expected=" << expected
                    << " actual=" << actual;
          if (expected != actual) {
            std::cout << " ✗ MISMATCH";
            tile_correct = false;
          } else {
            std::cout << " ✓";
          }
          std::cout << std::endl;
        }
      }

      if (tile_correct) {
        std::cout << "✓ Tile data verification PASSED" << std::endl;
        std::cout << "  Successfully received correct 4×4 region from PE 1" << std::endl;
      } else {
        std::cout << "✗ Tile data verification FAILED" << std::endl;
      }
    }

    rocshmem_free(d_tile_buffer);
  }

  // Example 2: Demonstrate tile_get API
  if (n_pes > 1 && my_pe == 0) {
    std::cout << "\n--- Tile API Demo (tile_get) ---" << std::endl;
    std::cout << "PE 0 fetching a 5×5 tile from PE 1's matrix A..." << std::endl;
    std::cout << "PE 0's d_A_local = " << d_A_local << std::endl;
    std::cout << "PE 0's d_B = " << d_B << std::endl;
    std::cout << "PE 0's d_C_local = " << d_C_local << std::endl;

    // Allocate buffer for fetched tile (5×5 = 25 elements)
    int* d_tile_buffer_get = (int*)rocshmem_malloc(25 * sizeof(int));
    std::vector<int> h_tile_buffer_get(25);

    if (M_local >= 5 && K >= 5) {
      // Create tensor descriptors
      // Now test with d_A_local after dummy allocation
      Tensor2D<int> remote_A(d_A_local, M_local, K);  // PE 1's A matrix (symmetric heap)
      Tensor2D<int> tile_buf(d_tile_buffer_get, 5, 5);   // 5×5 tile buffer

      // Fetch tile from PE 1 (rows 0-4, cols 0-4) from A matrix
      hipLaunchKernelGGL(tile_get_kernel, 1, 1, 0, 0,
                        tile_buf, remote_A, 0, 0, 5, 5, 1);
      hipDeviceSynchronize();

      // Copy tile back to host for verification
      hipMemcpy(h_tile_buffer_get.data(), d_tile_buffer_get, 25 * sizeof(int),
               hipMemcpyDeviceToHost);

      // Verify: the tile should match PE 1's A matrix values
      int pe1_row_offset = 1 * M_local;
      int pe1_id = 1;

      bool tile_correct = true;
      std::cout << "\nExpected vs Actual values for 5×5 tile (tile_get from A matrix [0-4, 0-4] with dummy workaround):" << std::endl;
      for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
          int global_row = pe1_row_offset + i;
          int expected = (global_row * 1000) + j + (pe1_id * 100000);
          int actual = h_tile_buffer_get[i * 5 + j];

          std::cout << "  [" << i << "," << j << "] expected=" << expected
                    << " actual=" << actual;
          if (expected != actual) {
            std::cout << " ✗ MISMATCH";
            tile_correct = false;
          } else {
            std::cout << " ✓";
          }
          std::cout << std::endl;
        }
      }

      if (tile_correct) {
        std::cout << "✓ Tile_get verification PASSED" << std::endl;
        std::cout << "  Successfully fetched correct 4×4 region from PE 1" << std::endl;
      } else {
        std::cout << "✗ Tile_get verification FAILED" << std::endl;
      }
    }

    rocshmem_free(d_tile_buffer_get);
  }

  // Cleanup
  rocshmem_free(d_A_local);
  rocshmem_free(d_B);
  rocshmem_free(d_C_local);

  rocshmem_finalize();

  return success ? 0 : 1;
}
