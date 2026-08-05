/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
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
#ifdef __HIP_DEVICE_COMPILE__
#include <rocshmem/rocshmem_device.hpp>
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
    // Note: tile_put uses (dst, src) parameter order
    rocshmem_tile_put(remote_buffer, local_tensor, start, boundary,
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
                                Tensor2D<int> tensor,
                                int tile_row, int tile_col,
                                int tile_rows, int tile_cols,
                                int remote_pe) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Define source tile on remote PE
    Coord2D start(tile_row, tile_col);
    Coord2D boundary(tile_row + tile_rows, tile_col + tile_cols);

    // Fetch tile using Tile API
    // Note: tile_get uses (dst, src) parameter order
    rocshmem_tile_get(local_buffer, tensor, start, boundary,
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
      // Small values to avoid integer overflow in GEMM
      // Format: (row % 10) + (col % 10) + 1
      // This keeps values in range [1, 19] for easy verification
      mat[i * cols + j] = ((row_offset + i) % 10) + (j % 10) + 1;
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
  int* d_A_local = (int*)rocshmem_malloc(M_local * K * sizeof(int));
  int* d_B = (int*)rocshmem_malloc(K * N * sizeof(int));
  int* d_C_local = (int*)rocshmem_malloc(M_local * N * sizeof(int));

  if (!d_A_local || !d_B || !d_C_local) {
    std::cerr << "PE " << my_pe << ": rocshmem_malloc failed" << std::endl;
    rocshmem_finalize();
    return 1;
  }

  // Copy data to device
  CHECK_HIP(hipMemcpy(d_A_local, h_A_local.data(), M_local * K * sizeof(int),
                      hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(d_B, h_B.data(), K * N * sizeof(int), hipMemcpyHostToDevice));
  // CHECK_HIP(hipStreamSynchronize(0));

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

  CHECK_HIP(hipDeviceSynchronize());

  // Synchronize all PEs
  rocshmem_barrier_all();

  // Copy result back to host
  CHECK_HIP(hipMemcpy(h_C_local.data(), d_C_local, M_local * N * sizeof(int),
                      hipMemcpyDeviceToHost));
  CHECK_HIP(hipStreamSynchronize(0));  // Ensure data is visible before verification

  // Verify result
  bool success = verify_result(h_C_local, h_A_local, h_B, M_local, K, N);

  if (my_pe == 0) {
    if (success) {
      std::cout << "\n✓ Verification PASSED" << std::endl;
    } else {
      std::cout << "\n✗ Verification FAILED" << std::endl;
    }
  }

  // Demonstrate tile_get API
  // Fetch directly into A_tensor instead of using a separate buffer
  if (n_pes > 1 && my_pe == 0) {
    std::cout << "\n--- Tile API Demo (tile_get) ---" << std::endl;
    std::cout << "PE 0 fetching 4×4 tile from PE 1's matrix A directly into A_tensor..." << std::endl;
    std::cout << "PE 0's d_A_local = " << d_A_local << std::endl;
    std::cout << "PE 0's d_B = " << d_B << std::endl;
    std::cout << "PE 0's d_C_local = " << d_C_local << std::endl;

    if (M_local >= 4 && K >= 4) {
      std::cout << "\nFetching 4×4 tile from PE 1's A[0:4, 0:4] into PE 0's A_tensor[0:4, 0:4]" << std::endl;

      // Fetch 4×4 tile from PE 1 directly into A_tensor at position [0,0]
      // This will overwrite PE 0's original A values in that region
      hipLaunchKernelGGL(tile_get_kernel, 1, 1, 0, 0,
                        A_tensor, A_tensor, 0, 0, 4, 4, 1);
      CHECK_HIP(hipStreamSynchronize(0));

      // Copy the modified A_tensor region back to host for verification
      std::vector<int> h_A_local(M_local * K);
      CHECK_HIP(hipMemcpy(h_A_local.data(), d_A_local, M_local * K * sizeof(int),
                          hipMemcpyDeviceToHost));

      // Verify: A_tensor[0:4, 0:4] should now contain PE 1's A matrix values
      int pe1_row_offset = 1 * M_local;

      bool tile_correct = true;
      std::cout << "\nExpected vs Actual values for 4×4 tile (fetched into A_tensor):" << std::endl;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          int global_row = pe1_row_offset + i;
          int expected = (global_row % 10) + (j % 10) + 1;
          int actual = h_A_local[i * K + j];  // A_tensor is row-major with stride K

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
        std::cout << "  Successfully fetched 4×4 region from PE 1 directly into A_tensor" << std::endl;
      } else {
        std::cout << "✗ Tile_get verification FAILED" << std::endl;
      }
    }
  }

  rocshmem_barrier_all();
  rocshmem_barrier_all();

  // Cleanup - all collective operations
  rocshmem_free(d_A_local);
  rocshmem_free(d_B);
  rocshmem_free(d_C_local);

  rocshmem_finalize();
  return success ? 0 : 1;
}
