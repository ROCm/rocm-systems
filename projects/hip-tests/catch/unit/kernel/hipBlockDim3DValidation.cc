/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * Each thread writes its globally-unique linear index into the output buffer.
 * The global linear index is computed as:
 *   global_x = blockIdx.x * blockDim.x + threadIdx.x
 *   global_y = blockIdx.y * blockDim.y + threadIdx.y
 *   global_z = blockIdx.z * blockDim.z + threadIdx.z
 *   flat = global_z * (gridDim.y * blockDim.y * gridDim.x * blockDim.x)
 *        + global_y * (gridDim.x * blockDim.x)
 *        + global_x
 */
__global__ void blockDim3DValidationKernel(int* out, int total_x, int total_y) {
  int global_x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int global_y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  int global_z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  int flat = global_z * (total_y * total_x) + global_y * total_x + global_x;
  out[flat] = flat;
}

/**
 * Test Description
 * ------------------------
 *  - Launches a 3D kernel over several block shapes and validates that every
 *    thread wrote the correct globally-unique linear index into the output
 *    buffer. Test aims to verify launch API correctly uses block dimensions.
 *    Block shapes tested (all with product <= 1024):
 *      dim3(1024,1,1), dim3(32,32,1), dim3(8,8,16), dim3(4,4,4)
 *  - Grid is fixed at dim3(2,2,2).
 *  - Shapes that exceed device limits are skipped gracefully.
 * Test source
 * ------------------------
 *  - catch/unit/kernel/hipBlockDim3DValidation.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_kernel_BlockDim3D_Validation_Functional) {
  // Query device limits once.
  int max_threads_per_block = 0;
  int max_block_dim_x = 0;
  int max_block_dim_y = 0;
  int max_block_dim_z = 0;
  HIP_CHECK(hipDeviceGetAttribute(&max_threads_per_block, hipDeviceAttributeMaxThreadsPerBlock, 0));
  HIP_CHECK(hipDeviceGetAttribute(&max_block_dim_x, hipDeviceAttributeMaxBlockDimX, 0));
  HIP_CHECK(hipDeviceGetAttribute(&max_block_dim_y, hipDeviceAttributeMaxBlockDimY, 0));
  HIP_CHECK(hipDeviceGetAttribute(&max_block_dim_z, hipDeviceAttributeMaxBlockDimZ, 0));

  // Generate over several 3D block shapes, each with product <= 1024.
  auto block = GENERATE(dim3(1024, 1, 1), dim3(32, 32, 1), dim3(8, 8, 16), dim3(4, 4, 4));

  // Skip if the shape exceeds any device limit.
  int block_product =
      static_cast<int>(block.x) * static_cast<int>(block.y) * static_cast<int>(block.z);
  if (block_product > max_threads_per_block || static_cast<int>(block.x) > max_block_dim_x ||
      static_cast<int>(block.y) > max_block_dim_y || static_cast<int>(block.z) > max_block_dim_z) {
    return;
  }

  const dim3 grid(2, 2, 2);

  int total_x = static_cast<int>(grid.x * block.x);
  int total_y = static_cast<int>(grid.y * block.y);
  int total_z = static_cast<int>(grid.z * block.z);
  int total_threads = total_x * total_y * total_z;

  size_t nbytes = static_cast<size_t>(total_threads) * sizeof(int);

  int* out_d = nullptr;
  HIP_CHECK(hipMalloc(&out_d, nbytes));
  HIP_CHECK(hipMemset(out_d, 0xFF, nbytes));  // fill with sentinel so unwritten entries show up

  hipLaunchKernelGGL(blockDim3DValidationKernel, grid, block, 0, 0, out_d, total_x, total_y);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  int* out_h = reinterpret_cast<int*>(malloc(nbytes));
  HIP_CHECK(hipMemcpy(out_h, out_d, nbytes, hipMemcpyDeviceToHost));

  // Validate: every element [flat] must equal flat.
  for (int iz = 0; iz < total_z; ++iz) {
    for (int iy = 0; iy < total_y; ++iy) {
      for (int ix = 0; ix < total_x; ++ix) {
        int expected = iz * (total_y * total_x) + iy * total_x + ix;
        REQUIRE(out_h[expected] == expected);
      }
    }
  }

  free(out_h);
  HIP_CHECK(hipFree(out_d));
}

/**
 * End doxygen group KernelTest.
 * @}
 */
