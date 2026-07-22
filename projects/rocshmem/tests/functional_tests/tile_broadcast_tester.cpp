/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "tile_broadcast_tester.hpp"

#include <rocshmem/rocshmem.hpp>

// Include internal context types before the tile API implementations
#include "../../src/context_incl.hpp"

// Include tile API template implementations
#include <rocshmem/rocshmem_TILE_impl.hpp>

using namespace rocshmem;

/******************************************************************************
 * TENSOR HELPERS (from tile_rma_tester.cpp)
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

// Simple tuple for coordinates
struct Tuple2D {
  int x, y;
  __device__ Tuple2D(int x_, int y_) : x(x_), y(y_) {}
  __device__ int get(int dim) const { return (dim == 0) ? x : y; }
};

/******************************************************************************
 * DEVICE TEST KERNELS
 *****************************************************************************/

// Thread-level broadcast test - single WG, single wave
__global__ void TileBroadcastThreadTest(rocshmem_team_t team,
                                        float *source, float *dest,
                                        int tile_extent_0, int tile_extent_1,
                                        int my_world_pe, int pe_root,
                                        ShmemContextType ctx_type,
                                        int *error_flag) {
  __shared__ rocshmem_ctx_t ctx;

  // Create context from team (single WG)
  rocshmem_wg_team_create_ctx(team, ctx_type, &ctx);

  // Single tile, no offset needed
  Tensor2D<float> src_tensor(source, tile_extent_0, tile_extent_1);
  Tensor2D<float> dst_tensor(dest, tile_extent_0, tile_extent_1);
  Tuple2D start(0, 0);
  Tuple2D boundary(tile_extent_0, tile_extent_1);

  // Only thread 0 performs broadcast
  if (threadIdx.x == 0) {
    rocshmem_ctx_tile_broadcast(ctx, team, dst_tensor, src_tensor,
                                 start, boundary, pe_root, 0);
  }
  __syncthreads();

  // Verify results on non-root PEs
  if (my_world_pe != pe_root && threadIdx.x == 0) {
    int matrix_size = tile_extent_0 * tile_extent_1;
    for (int idx = 0; idx < matrix_size; idx++) {
      float expected = pe_root * 100.0f + idx;
      float actual = dest[idx];
      if (actual != expected) {
        printf("Thread-level: PE %d verification failed at [%d]: got %f, expected %f\n",
               my_world_pe, idx, actual, expected);
        *error_flag = 1;
      }
    }
  }

  __syncthreads();
  rocshmem_wg_ctx_destroy(&ctx);
}

// Wave-level broadcast test - single WG, single wave
__global__ void TileBroadcastWaveTest(rocshmem_team_t team,
                                      float *source, float *dest,
                                      int tile_extent_0, int tile_extent_1,
                                      int my_world_pe, int pe_root,
                                      ShmemContextType ctx_type,
                                      int wf_size,
                                      int *error_flag) {
  __shared__ rocshmem_ctx_t ctx;

  // Create context from team
  rocshmem_wg_team_create_ctx(team, ctx_type, &ctx);

  Tensor2D<float> src_tensor(source, tile_extent_0, tile_extent_1);
  Tensor2D<float> dst_tensor(dest, tile_extent_0, tile_extent_1);
  Tuple2D start(0, 0);
  Tuple2D boundary(tile_extent_0, tile_extent_1);

  // Only first wave performs broadcast
  if (threadIdx.x < wf_size) {
    rocshmem_ctx_tile_broadcast_wave(ctx, team, dst_tensor, src_tensor,
                                      start, boundary, pe_root, 0);
  }
  __syncthreads();

  // Verify on non-root PEs
  if (my_world_pe != pe_root && threadIdx.x == 0) {
    int matrix_size = tile_extent_0 * tile_extent_1;
    for (int idx = 0; idx < matrix_size; idx++) {
      float expected = pe_root * 100.0f + idx;
      float actual = dest[idx];
      if (actual != expected) {
        printf("Wave-level: PE %d verification failed at [%d]: got %f, expected %f\n",
               my_world_pe, idx, actual, expected);
        *error_flag = 1;
      }
    }
  }

  __syncthreads();
  rocshmem_wg_ctx_destroy(&ctx);
}

// Workgroup-level broadcast test - multiple WGs with different teams
__global__ void TileBroadcastTest(rocshmem_team_t *teams, int num_teams,
                                  float *source, float *dest,
                                  int tile_extent_0, int tile_extent_1,
                                  int my_world_pe, int pe_root,
                                  ShmemContextType ctx_type,
                                  int *error_flag) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();

  // Each workgroup uses a DIFFERENT team
  rocshmem_team_t my_team = teams[wg_id % num_teams];

  // Create context from this workgroup's specific team
  rocshmem_wg_team_create_ctx(my_team, ctx_type, &ctx);

  // Calculate offset for this workgroup's tile
  int matrix_size = tile_extent_0 * tile_extent_1;
  int offset = matrix_size * wg_id;

  // Create tensors for source and destination
  Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1);
  Tensor2D<float> dst_tensor(dest + offset, tile_extent_0, tile_extent_1);
  Tuple2D start(0, 0);
  Tuple2D boundary(tile_extent_0, tile_extent_1);

  // Perform tile broadcast - root PE broadcasts its tile to all other PEs on the team
  rocshmem_ctx_tile_broadcast_wg(ctx, my_team, dst_tensor, src_tensor,
                                  start, boundary, pe_root, 0);

  __syncthreads();

  // Verify results on non-root PEs
  if (my_world_pe != pe_root) {
    if (threadIdx.x == 0) {
      // Check that we received the root PE's data
      // Root PE's data is pe_root * 100 + element index
      for (int i = 0; i < tile_extent_0; i++) {
        for (int j = 0; j < tile_extent_1; j++) {
          int idx = i * tile_extent_1 + j;
          float expected = pe_root * 100.0f + idx;
          float actual = dest[offset + idx];
          if (actual != expected) {
            printf("WG %d: PE %d verification failed at [%d,%d]: got %f, expected %f\n",
                   wg_id, my_world_pe, i, j, actual, expected);
            *error_flag = 1;
          }
        }
      }
    }
  }

  __syncthreads();

  // Team barrier - each workgroup synchronizes on its own team
  rocshmem_ctx_sync_wg(ctx, my_team);

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
TileBroadcastTester::TileBroadcastTester(TesterArguments args)
    : Tester(args) {
  num_teams = 4;  // Default to 4 teams

  // Allocate teams using hipHostMalloc
  CHECK_HIP(hipHostMalloc(&teams, num_teams * sizeof(rocshmem_team_t)));

  // Initialize all team handles to ROCSHMEM_TEAM_INVALID
  for (int i = 0; i < num_teams; i++) {
    teams[i] = ROCSHMEM_TEAM_INVALID;
  }

  // Allocate tile data
  tile_extent_0 = 8;  // Default tile size
  tile_extent_1 = 8;
  int tile_size = tile_extent_0 * tile_extent_1;
  int total_size = tile_size * num_teams;  // One tile per workgroup

  source = (float *)rocshmem_malloc(total_size * sizeof(float));
  dest = (float *)rocshmem_malloc(total_size * sizeof(float));

  if (!source || !dest) {
    fprintf(stderr, "Failed to allocate symmetric memory for tiles\n");
    exit(EXIT_FAILURE);
  }

  // Allocate error flag
  CHECK_HIP(hipMalloc(&error_flag, sizeof(int)));
}

TileBroadcastTester::~TileBroadcastTester() {
  rocshmem_free(source);
  rocshmem_free(dest);
  CHECK_HIP(hipFree(error_flag));

  // Destroy teams
  for (int i = 0; i < num_teams; i++) {
    if (teams[i] != ROCSHMEM_TEAM_INVALID) {
      rocshmem_team_destroy(teams[i]);
    }
  }
  CHECK_HIP(hipHostFree(teams));
}

void TileBroadcastTester::resetBuffers([[maybe_unused]] size_t size) {
  int tile_size = tile_extent_0 * tile_extent_1;
  int total_size = tile_size * num_teams;

  // Initialize source data: each PE's data is pe * 100 + element index
  for (int i = 0; i < total_size; i++) {
    source[i] = args.myid * 100.0f + (i % tile_size);
    dest[i] = -1.0f;  // Initialize dest to invalid value
  }

  // Reset error flag
  int zero = 0;
  CHECK_HIP(hipMemcpy(error_flag, &zero, sizeof(int), hipMemcpyHostToDevice));
}

void TileBroadcastTester::preLaunchKernel() {
  int n_pes = rocshmem_n_pes();

  if (_type == TileBroadcastWGTestType) {
    // Multi-team case: Create multiple duplicate teams (all PEs are members of all teams)
    for (int i = 0; i < num_teams; i++) {
      teams[i] = ROCSHMEM_TEAM_INVALID;
      rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0, &teams[i]);

      if (teams[i] == ROCSHMEM_TEAM_INVALID) {
        printf("PE %d: Failed to create team %d\n", args.myid, i);
        rocshmem_global_exit(1);
      }
    }
  } else {
    // Single-team case: Create just one team for thread/wave level tests
    teams[0] = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0, &teams[0]);

    if (teams[0] == ROCSHMEM_TEAM_INVALID) {
      printf("PE %d: Failed to create team\n", args.myid);
      rocshmem_global_exit(1);
    }
  }
}

void TileBroadcastTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                       [[maybe_unused]] int loop,
                                       [[maybe_unused]] size_t size) {
  int pe_root = 0;  // Root PE for broadcast

  switch (_type) {
    case TileBroadcastTestType:
      // Thread-level test: single WG, single wave
      hipLaunchKernelGGL(TileBroadcastThreadTest, dim3(1), blockSize, 0, stream,
                         teams[0], source, dest, tile_extent_0, tile_extent_1,
                         args.myid, pe_root, _shmem_context, error_flag);
      break;

    case TileBroadcastWaveTestType:
      // Wave-level test: single WG, single wave
      hipLaunchKernelGGL(TileBroadcastWaveTest, dim3(1), blockSize, 0, stream,
                         teams[0], source, dest, tile_extent_0, tile_extent_1,
                         args.myid, pe_root, _shmem_context, wf_size, error_flag);
      break;

    case TileBroadcastWGTestType:
      // Workgroup-level test: multiple WGs with different teams
      hipLaunchKernelGGL(TileBroadcastTest, dim3(num_teams), blockSize, 0, stream,
                         teams, num_teams, source, dest, tile_extent_0, tile_extent_1,
                         args.myid, pe_root, _shmem_context, error_flag);
      break;

    default:
      fprintf(stderr, "Unknown TileBroadcast test type\n");
      exit(EXIT_FAILURE);
  }
}

void TileBroadcastTester::postLaunchKernel() {
  // Destroy teams after each iteration to prevent resource exhaustion
  if (_type == TileBroadcastWGTestType) {
    // Multi-team case: destroy all teams
    for (int i = 0; i < num_teams; i++) {
      if (teams[i] != ROCSHMEM_TEAM_INVALID) {
        rocshmem_team_destroy(teams[i]);
        teams[i] = ROCSHMEM_TEAM_INVALID;
      }
    }
  } else {
    // Single-team case: destroy the one team
    if (teams[0] != ROCSHMEM_TEAM_INVALID) {
      rocshmem_team_destroy(teams[0]);
      teams[0] = ROCSHMEM_TEAM_INVALID;
    }
  }
}

void TileBroadcastTester::verifyResults([[maybe_unused]] size_t size) {
  // Check error flag
  int h_error_flag;
  CHECK_HIP(hipMemcpy(&h_error_flag, error_flag, sizeof(int), hipMemcpyDeviceToHost));

  if (h_error_flag) {
    fprintf(stderr, "PE %d: Tile broadcast verification FAILED\n", args.myid);
    exit(EXIT_FAILURE);
  }

  if (args.myid == 0) {
    printf("PE %d: Tile broadcast verification PASSED\n", args.myid);
  }
}
