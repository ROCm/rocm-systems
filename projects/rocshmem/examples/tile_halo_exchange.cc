/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

/**
 * 2D Halo Exchange using rocSHMEM Tile API
 *
 * This example demonstrates a 2D halo exchange pattern common in stencil
 * computations. PEs are arranged in a 2D grid topology, and each PE exchanges
 * boundary data with its neighbors using tile_put_wg.
 *
 * Requirements:
 * - Number of PEs must be a perfect square (4, 9, 16, 25, ...)
 * - Each PE has a local domain with a 1-element halo region on each side
 * - Communication uses workgroup-collective tile_put_wg operations
 */

#include <rocshmem/rocshmem.hpp>
#ifdef __HIP_DEVICE_COMPILE__
#include <rocshmem/rocshmem_device.hpp>
#endif

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "util.h"

using namespace rocshmem;

/******************************************************************************
 * COORDINATE TYPES FOR TILE API
 *****************************************************************************/

struct Coord2D {
  int row;
  int col;

  __host__ __device__ Coord2D(int r, int c) : row(r), col(c) {}
  __host__ __device__ int get(int dim) const { return (dim == 0) ? row : col; }
};

/******************************************************************************
 * TENSOR TYPE
 *****************************************************************************/

template <typename T>
struct Tensor2D {
  using element_type = T;
  static constexpr int ndim = 2;

  T* data;
  int rows;
  int cols;
  int row_stride;
  int col_stride;

  __host__ __device__ Tensor2D(T* data_, int rows_, int cols_,
                                int row_stride_ = -1, int col_stride_ = 1)
      : data(data_), rows(rows_), cols(cols_), col_stride(col_stride_) {
    row_stride = (row_stride_ == -1) ? cols : row_stride_;
  }

  __host__ __device__ T* data_handle() const { return data; }
  __host__ __device__ int stride(int dim) const {
    return (dim == 0) ? row_stride : col_stride;
  }
};

/******************************************************************************
 * HALO EXCHANGE KERNEL
 *****************************************************************************/

/**
 * @brief Workgroup-collective kernel to perform halo exchange
 *
 * Each workgroup sends one boundary to a neighbor using tile_put_wg
 *
 * @param local_tensor Local domain tensor (includes halo regions)
 * @param inner_size Size of inner domain (excluding halo)
 * @param north_pe PE ID of north neighbor (-1 if none)
 * @param south_pe PE ID of south neighbor (-1 if none)
 * @param east_pe PE ID of east neighbor (-1 if none)
 * @param west_pe PE ID of west neighbor (-1 if none)
 */
__global__ void halo_exchange_kernel(Tensor2D<float> local_tensor,
                                     int inner_size,
                                     int north_pe, int south_pe,
                                     int east_pe, int west_pe) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_ctx_create(ROCSHMEM_CTX_WG_PRIVATE, &ctx);

  int wg_id = blockIdx.x;
  int halo_width = 1;
  int total_size = inner_size + 2 * halo_width;

  // Each workgroup handles one direction
  // WG 0: North, WG 1: South, WG 2: East, WG 3: West

  if (wg_id == 0 && north_pe >= 0) {
    // Send north boundary (row 1, cols 1-8) to north neighbor's south halo (last row, cols 1-8)
    // Source: row 1, cols 1-8 of local tensor
    // Dest: row 9 (last), cols 1-8 of north neighbor's tensor
    float* src_ptr = local_tensor.data + halo_width * total_size + halo_width;
    float* dst_ptr = local_tensor.data + (total_size - halo_width) * total_size + halo_width;
    Tensor2D<float> src_sub(src_ptr, 1, inner_size, total_size);
    Tensor2D<float> dst_sub(dst_ptr, 1, inner_size, total_size);

    Coord2D start(0, 0);
    Coord2D boundary(1, inner_size);
    rocshmem_ctx_tile_put_wg(ctx, dst_sub, src_sub, start, boundary, north_pe, 0);
  }
  else if (wg_id == 1 && south_pe >= 0) {
    // Send south boundary (row 8, cols 1-8) to south neighbor's north halo (row 0, cols 1-8)
    float* src_ptr = local_tensor.data + (total_size - halo_width - 1) * total_size + halo_width;
    float* dst_ptr = local_tensor.data + 0 * total_size + halo_width;
    Tensor2D<float> src_sub(src_ptr, 1, inner_size, total_size);
    Tensor2D<float> dst_sub(dst_ptr, 1, inner_size, total_size);

    Coord2D start(0, 0);
    Coord2D boundary(1, inner_size);
    rocshmem_ctx_tile_put_wg(ctx, dst_sub, src_sub, start, boundary, south_pe, 0);
  }
  else if (wg_id == 2 && east_pe >= 0) {
    // Send east boundary (rows 1-8, col 8) to east neighbor's west halo (rows 1-8, col 0)
    float* src_ptr = local_tensor.data + halo_width * total_size + (total_size - halo_width - 1);
    float* dst_ptr = local_tensor.data + halo_width * total_size + 0;
    Tensor2D<float> src_sub(src_ptr, inner_size, 1, total_size);
    Tensor2D<float> dst_sub(dst_ptr, inner_size, 1, total_size);

    Coord2D start(0, 0);
    Coord2D boundary(inner_size, 1);
    rocshmem_ctx_tile_put_wg(ctx, dst_sub, src_sub, start, boundary, east_pe, 0);
  }
  else if (wg_id == 3 && west_pe >= 0) {
    // Send west boundary (rows 1-8, col 1) to west neighbor's east halo (rows 1-8, col 9/last)
    float* src_ptr = local_tensor.data + halo_width * total_size + halo_width;
    float* dst_ptr = local_tensor.data + halo_width * total_size + (total_size - halo_width);
    Tensor2D<float> src_sub(src_ptr, inner_size, 1, total_size);
    Tensor2D<float> dst_sub(dst_ptr, inner_size, 1, total_size);

    Coord2D start(0, 0);
    Coord2D boundary(inner_size, 1);
    rocshmem_ctx_tile_put_wg(ctx, dst_sub, src_sub, start, boundary, west_pe, 0);
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HELPER FUNCTIONS
 *****************************************************************************/

void print_domain(const std::vector<float>& data, int size, int my_pe,
                  int grid_row, int grid_col) {
  std::cout << "\nPE " << my_pe << " (grid pos [" << grid_row << "," << grid_col
            << "]) domain (" << size << "×" << size << "):" << std::endl;

  for (int i = 0; i < size; i++) {
    std::cout << "  ";
    for (int j = 0; j < size; j++) {
      float val = data[i * size + j];
      if (val < 0) {
        std::cout << " -- ";  // Uninitialized halo
      } else {
        std::cout << std::setw(3) << std::setfill(' ') << (int)val << " ";
      }
    }
    std::cout << std::endl;
  }
}

/******************************************************************************
 * MAIN
 *****************************************************************************/

int main(int argc, char** argv) {
  rocshmem_init();

  int my_pe = rocshmem_my_pe();
  int n_pes = rocshmem_n_pes();

  // Check that we have a perfect square number of PEs
  int grid_size = (int)std::sqrt(n_pes);
  if (grid_size * grid_size != n_pes) {
    if (my_pe == 0) {
      std::cerr << "Error: Number of PEs (" << n_pes
                << ") must be a perfect square (4, 9, 16, 25, ...)" << std::endl;
    }
    rocshmem_finalize();
    return 1;
  }

  // Determine my position in the 2D grid
  int my_grid_row = my_pe / grid_size;
  int my_grid_col = my_pe % grid_size;

  // Determine neighbor PEs (-1 if at boundary)
  int north_pe = (my_grid_row > 0) ? (my_grid_row - 1) * grid_size + my_grid_col : -1;
  int south_pe = (my_grid_row < grid_size - 1) ? (my_grid_row + 1) * grid_size + my_grid_col : -1;
  int west_pe = (my_grid_col > 0) ? my_grid_row * grid_size + (my_grid_col - 1) : -1;
  int east_pe = (my_grid_col < grid_size - 1) ? my_grid_row * grid_size + (my_grid_col + 1) : -1;

  if (my_pe == 0) {
    std::cout << "=== 2D Halo Exchange using Tile API ===" << std::endl;
    std::cout << "Grid topology: " << grid_size << "×" << grid_size
              << " (" << n_pes << " PEs)" << std::endl;
  }

  // Configuration
  const int INNER_SIZE = 8;  // Inner domain size (excluding halo)
  const int HALO_WIDTH = 1;
  const int TOTAL_SIZE = INNER_SIZE + 2 * HALO_WIDTH;  // 10×10 including halo

  // Allocate symmetric memory for the domain (including halo)
  float* d_domain = (float*)rocshmem_malloc(TOTAL_SIZE * TOTAL_SIZE * sizeof(float));

  // Initialize domain on host
  std::vector<float> h_domain(TOTAL_SIZE * TOTAL_SIZE, -1.0f);  // -1 for halo regions

  // Initialize inner domain with unique values per PE: PE_ID * 100 + row * 10 + col
  for (int i = HALO_WIDTH; i < INNER_SIZE + HALO_WIDTH; i++) {
    for (int j = HALO_WIDTH; j < INNER_SIZE + HALO_WIDTH; j++) {
      int inner_row = i - HALO_WIDTH;
      int inner_col = j - HALO_WIDTH;
      h_domain[i * TOTAL_SIZE + j] = my_pe * 100.0f + inner_row * 10.0f + inner_col;
    }
  }

  // Copy to device
  CHECK_HIP(hipMemcpy(d_domain, h_domain.data(),
                      TOTAL_SIZE * TOTAL_SIZE * sizeof(float),
                      hipMemcpyHostToDevice));

  if (my_pe == 0) {
    std::cout << "\n--- Before Halo Exchange ---" << std::endl;
  }

  rocshmem_barrier_all();

  // Print initial state
  CHECK_HIP(hipMemcpy(h_domain.data(), d_domain,
                      TOTAL_SIZE * TOTAL_SIZE * sizeof(float),
                      hipMemcpyDeviceToHost));
  print_domain(h_domain, TOTAL_SIZE, my_pe, my_grid_row, my_grid_col);

  rocshmem_barrier_all();

  // Perform halo exchange
  // Launch 4 workgroups (one per direction), each with 256 threads
  Tensor2D<float> domain_tensor(d_domain, TOTAL_SIZE, TOTAL_SIZE);

  hipLaunchKernelGGL(halo_exchange_kernel, 4, 256, 0, 0,
                     domain_tensor, INNER_SIZE,
                     north_pe, south_pe, east_pe, west_pe);
  CHECK_HIP(hipStreamSynchronize(0));

  rocshmem_barrier_all();

  if (my_pe == 0) {
    std::cout << "\n--- After Halo Exchange ---" << std::endl;
  }

  rocshmem_barrier_all();

  // Print final state
  CHECK_HIP(hipMemcpy(h_domain.data(), d_domain,
                      TOTAL_SIZE * TOTAL_SIZE * sizeof(float),
                      hipMemcpyDeviceToHost));
  print_domain(h_domain, TOTAL_SIZE, my_pe, my_grid_row, my_grid_col);

  // Verify halo values
  bool success = true;

  // Check north halo (row 0)
  if (north_pe >= 0) {
    int north_grid_row = north_pe / grid_size;
    int north_grid_col = north_pe % grid_size;
    for (int j = HALO_WIDTH; j < INNER_SIZE + HALO_WIDTH; j++) {
      int expected_inner_row = INNER_SIZE - 1;  // Last row of north neighbor's inner domain
      int expected_inner_col = j - HALO_WIDTH;
      float expected = north_pe * 100.0f + expected_inner_row * 10.0f + expected_inner_col;
      float actual = h_domain[0 * TOTAL_SIZE + j];

      if (std::abs(actual - expected) > 0.001f) {
        std::cout << "PE " << my_pe << " north halo error at [0," << j
                  << "]: expected " << expected << ", got " << actual << std::endl;
        success = false;
      }
    }
  }

  // Check south halo (last row)
  if (south_pe >= 0) {
    for (int j = HALO_WIDTH; j < INNER_SIZE + HALO_WIDTH; j++) {
      int expected_inner_row = 0;  // First row of south neighbor's inner domain
      int expected_inner_col = j - HALO_WIDTH;
      float expected = south_pe * 100.0f + expected_inner_row * 10.0f + expected_inner_col;
      float actual = h_domain[(TOTAL_SIZE - 1) * TOTAL_SIZE + j];

      if (std::abs(actual - expected) > 0.001f) {
        std::cout << "PE " << my_pe << " south halo error at [" << (TOTAL_SIZE - 1)
                  << "," << j << "]: expected " << expected << ", got " << actual << std::endl;
        success = false;
      }
    }
  }

  // Check west halo (column 0)
  if (west_pe >= 0) {
    for (int i = HALO_WIDTH; i < INNER_SIZE + HALO_WIDTH; i++) {
      int expected_inner_row = i - HALO_WIDTH;
      int expected_inner_col = INNER_SIZE - 1;  // Last column of west neighbor's inner domain
      float expected = west_pe * 100.0f + expected_inner_row * 10.0f + expected_inner_col;
      float actual = h_domain[i * TOTAL_SIZE + 0];

      if (std::abs(actual - expected) > 0.001f) {
        std::cout << "PE " << my_pe << " west halo error at [" << i
                  << ",0]: expected " << expected << ", got " << actual << std::endl;
        success = false;
      }
    }
  }

  // Check east halo (last column)
  if (east_pe >= 0) {
    for (int i = HALO_WIDTH; i < INNER_SIZE + HALO_WIDTH; i++) {
      int expected_inner_row = i - HALO_WIDTH;
      int expected_inner_col = 0;  // First column of east neighbor's inner domain
      float expected = east_pe * 100.0f + expected_inner_row * 10.0f + expected_inner_col;
      float actual = h_domain[i * TOTAL_SIZE + (TOTAL_SIZE - 1)];

      if (std::abs(actual - expected) > 0.001f) {
        std::cout << "PE " << my_pe << " east halo error at [" << i
                  << "," << (TOTAL_SIZE - 1) << "]: expected " << expected
                  << ", got " << actual << std::endl;
        success = false;
      }
    }
  }

  rocshmem_barrier_all();

  if (my_pe == 0) {
    std::cout << "\n";
    if (success) {
      std::cout << "✓ Halo exchange verification PASSED" << std::endl;
    } else {
      std::cout << "✗ Halo exchange verification FAILED" << std::endl;
    }
  }

  // Cleanup
  rocshmem_free(d_domain);
  rocshmem_finalize();

  return success ? 0 : 1;
}
