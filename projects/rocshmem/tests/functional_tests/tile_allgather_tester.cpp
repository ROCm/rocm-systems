/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "tile_allgather_tester.hpp"

#include <rocshmem/rocshmem.hpp>

// Include internal context types before the tile API implementations
#include "../../src/context_incl.hpp"

// Include tile API template implementations
#include <rocshmem/rocshmem_TILE_impl.hpp>

using namespace rocshmem;

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

// Simple tuple for coordinates
struct Tuple2D {
  int x, y;
  __device__ Tuple2D(int x_, int y_) : x(x_), y(y_) {}
  __device__ int get(int dim) const { return (dim == 0) ? x : y; }
};

/******************************************************************************
 * DEVICE TEST KERNELS
 *****************************************************************************/

// Thread-level allgather test - single WG, single wave
__global__ void TileAllgatherThreadTest(rocshmem_team_t team,
                                        float *source, float *dest,
                                        int tile_extent_0, int tile_extent_1,
                                        int my_world_pe, int n_pes,
                                        ShmemContextType ctx_type,
                                        int loop, int skip,
                                        long long int *start_time,
                                        long long int *end_time,
                                        int *error_flag) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();

  rocshmem_wg_team_create_ctx(team, ctx_type, &ctx);

  int tile_size = tile_extent_0 * tile_extent_1;

  Tensor2D<float> src_tensor(source, tile_extent_0, tile_extent_1);
  Tensor2D<float> dst_tensor(dest, tile_extent_0 * n_pes, tile_extent_1);
  Tuple2D start_coord(0, 0);
  Tuple2D boundary(tile_extent_0, tile_extent_1);

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && threadIdx.x == 0) {
      start_time[wg_id] = wall_clock64();
    }
    if (threadIdx.x == 0) {
      rocshmem_ctx_tile_allgather(ctx, team, dst_tensor, src_tensor,
                                  start_coord, boundary, 0);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    end_time[wg_id] = wall_clock64();
  }

  if (threadIdx.x == 0) {
    for (int pe = 0; pe < n_pes; pe++) {
      int offset = pe * tile_size;
      for (int idx = 0; idx < tile_size; idx++) {
        float expected = pe * 100.0f + idx;
        float actual = dest[offset + idx];
        if (actual != expected) {
          printf("Thread-level: PE %d verification failed for PE %d's data at [%d]: got %f, expected %f\n",
                 my_world_pe, pe, idx, actual, expected);
          *error_flag = 1;
        }
      }
    }
  }

  __syncthreads();
  rocshmem_wg_ctx_destroy(&ctx);
}

// Wave-level allgather test - each wave uses its own team and context
__global__ void TileAllgatherWaveTest(rocshmem_team_t *teams,
                                      float *source, float *dest,
                                      int tile_extent_0, int tile_extent_1,
                                      int my_world_pe, int n_pes,
                                      ShmemContextType ctx_type,
                                      int wf_size, int num_waves_per_wg,
                                      int loop, int skip,
                                      long long int *start_time,
                                      long long int *end_time,
                                      int *error_flag) {
  extern __shared__ rocshmem_ctx_t ctx_array[];

  int t_id       = get_flat_block_id();
  int wg_id      = get_flat_grid_id();
  int wf_id      = t_id / wf_size;
  int wg_offset  = wg_id * num_waves_per_wg;
  int flat_wf_id = wg_offset + wf_id;

  for (int wf_i = 0; wf_i < num_waves_per_wg; wf_i++) {
    rocshmem_wg_team_create_ctx(teams[wg_offset + wf_i], ctx_type,
                                &ctx_array[wf_i]);
    __syncthreads();
  }

  int tile_size = tile_extent_0 * tile_extent_1;
  float *my_source = source + flat_wf_id * tile_size;
  float *my_dest   = dest   + flat_wf_id * tile_size * n_pes;

  Tensor2D<float> src_tensor(my_source, tile_extent_0, tile_extent_1);
  Tensor2D<float> dst_tensor(my_dest, tile_extent_0 * n_pes, tile_extent_1);
  Tuple2D start_coord(0, 0);
  Tuple2D boundary(tile_extent_0, tile_extent_1);

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && t_id % wf_size == 0) {
      start_time[flat_wf_id] = wall_clock64();
    }
    rocshmem_ctx_tile_allgather_wave(ctx_array[wf_id], teams[flat_wf_id],
                                     dst_tensor, src_tensor, start_coord, boundary, 0);
  }
  __syncthreads();

  if (t_id % wf_size == 0) {
    end_time[flat_wf_id] = wall_clock64();
  }

  if (t_id % wf_size == 0) {
    for (int pe = 0; pe < n_pes; pe++) {
      int offset = pe * tile_size;
      for (int idx = 0; idx < tile_size; idx++) {
        float expected = pe * 100.0f + flat_wf_id * 1000.0f + idx;
        float actual = my_dest[offset + idx];
        if (actual != expected) {
          printf("Wave-level: PE %d wave %d verification failed for PE %d's data at [%d]: got %f, expected %f\n",
                 my_world_pe, flat_wf_id, pe, idx, actual, expected);
          *error_flag = 1;
        }
      }
    }
  }

  __syncthreads();

  for (int wf_i = 0; wf_i < num_waves_per_wg; wf_i++) {
    rocshmem_wg_ctx_destroy(&ctx_array[wf_i]);
    __syncthreads();
  }
}

// Workgroup-level allgather test - multiple WGs with different teams
__global__ void TileAllgatherTest(rocshmem_team_t *teams, int num_teams,
                                   float *source, float *dest,
                                   int tile_extent_0, int tile_extent_1,
                                   int my_world_pe, int n_pes,
                                   ShmemContextType ctx_type,
                                   int loop, int skip,
                                   long long int *start_time,
                                   long long int *end_time,
                                   int *error_flag) {
  extern __shared__ rocshmem_ctx_t ctx_array[];
  int wg_id = get_flat_grid_id();

  rocshmem_team_t my_team = teams[wg_id % num_teams];
  rocshmem_wg_team_create_ctx(my_team, ctx_type, &ctx_array[0]);

  int tile_size  = tile_extent_0 * tile_extent_1;
  int src_offset = tile_size * wg_id;
  int dst_offset = tile_size * n_pes * wg_id;

  Tensor2D<float> src_tensor(source + src_offset, tile_extent_0, tile_extent_1);
  Tensor2D<float> dst_tensor(dest + dst_offset, tile_extent_0 * n_pes, tile_extent_1);
  Tuple2D start_coord(0, 0);
  Tuple2D boundary(tile_extent_0, tile_extent_1);

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && threadIdx.x == 0) {
      start_time[wg_id] = wall_clock64();
    }
    rocshmem_ctx_tile_allgather_wg(ctx_array[0], my_team, dst_tensor, src_tensor,
                                    start_coord, boundary, 0);
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    end_time[wg_id] = wall_clock64();
  }

  if (threadIdx.x == 0) {
    for (int pe = 0; pe < n_pes; pe++) {
      int pe_offset = pe * tile_size;
      for (int i = 0; i < tile_extent_0; i++) {
        for (int j = 0; j < tile_extent_1; j++) {
          int idx = i * tile_extent_1 + j;
          float expected = pe * 100.0f + wg_id * 1000.0f + idx;
          float actual = dest[dst_offset + pe_offset + idx];
          if (actual != expected) {
            printf("WG %d: PE %d verification failed for PE %d's data at [%d,%d]: got %f, expected %f\n",
                   wg_id, my_world_pe, pe, i, j, actual, expected);
            *error_flag = 1;
          }
        }
      }
    }
  }

  __syncthreads();
  rocshmem_ctx_sync_wg(ctx_array[0], my_team);
  rocshmem_wg_ctx_destroy(&ctx_array[0]);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
TileAllgatherTester::TileAllgatherTester(TesterArguments args)
    : Tester(args) {
  // Wave test needs one team per wave (num_wgs * num_warps); WG test needs
  // one team per WG (num_wgs).  Allocate the larger so one array covers both.
  num_teams = args.num_wgs * num_warps;

  // Allocate teams using hipHostMalloc
  CHECK_HIP(hipHostMalloc(&teams, num_teams * sizeof(rocshmem_team_t)));

  // Initialize all team handles to ROCSHMEM_TEAM_INVALID
  for (int i = 0; i < num_teams; i++) {
    teams[i] = ROCSHMEM_TEAM_INVALID;
  }

  // Derive tile dimensions from max_msg_size if provided.
  // max_msg_size = tile_extent_0 * tile_extent_1 * sizeof(float) (source tile).
  tile_extent_0 = 8;
  tile_extent_1 = 8;
  if (args.max_msg_size_set) {
    int derived = static_cast<int>(args.max_msg_size / (tile_extent_0 * sizeof(float)));
    if (derived >= 1) tile_extent_1 = derived;
  }
  // Sweep from one column to the full tile.
  size_t tile_size_bytes = static_cast<size_t>(tile_extent_0) * tile_extent_1 * sizeof(float);
  size_t row_bytes       = static_cast<size_t>(tile_extent_0) * sizeof(float);
  this->args.min_msg_size = row_bytes;
  max_msg_size            = tile_size_bytes;

  // Fix num_timers for the wave variant.
  if (_type == TileAllgatherWaveTestType) {
    num_timers = args.num_wgs * num_warps;
  }

  int tile_size = tile_extent_0 * tile_extent_1;
  int n_pes = rocshmem_n_pes();

  // Allocate for max tile size; smaller sweep steps use a subset.
  int total_src_size = tile_size * num_teams;
  int total_dst_size = tile_size * n_pes * num_teams;

  source = (float *)rocshmem_malloc(total_src_size * sizeof(float));
  dest = (float *)rocshmem_malloc(total_dst_size * sizeof(float));

  if (!source || !dest) {
    fprintf(stderr, "Failed to allocate symmetric memory for tiles\n");
    exit(EXIT_FAILURE);
  }

  // Allocate error flag
  CHECK_HIP(hipMalloc(&error_flag, sizeof(int)));
}

TileAllgatherTester::~TileAllgatherTester() {
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

void TileAllgatherTester::resetBuffers(size_t size) {
  int t1 = tile_extent_1;
  if (size > 0) {
    int derived = static_cast<int>(size / (tile_extent_0 * sizeof(float)));
    if (derived >= 1 && derived <= tile_extent_1) t1 = derived;
  }
  int tile_size     = tile_extent_0 * t1;
  int max_tile_size = tile_extent_0 * tile_extent_1;
  int n_pes = rocshmem_n_pes();
  int total_dst_size = max_tile_size * n_pes * num_teams;

  for (int i = 0; i < total_dst_size; i++) dest[i] = -1.0f;

  // Source stride matches the kernel's per-slot offset (tile_size per slot).
  for (int slot = 0; slot < num_teams; slot++) {
    int src_offset = slot * tile_size;
    for (int i = 0; i < tile_size; i++) {
      source[src_offset + i] = args.myid * 100.0f + slot * 1000.0f + i;
    }
  }

  int zero = 0;
  CHECK_HIP(hipMemcpy(error_flag, &zero, sizeof(int), hipMemcpyHostToDevice));
}

void TileAllgatherTester::preLaunchKernel() {
  int n_pes = rocshmem_n_pes();
  int teams_needed;

  if (_type == TileAllgatherWaveTestType) {
    // One team per wave (num_wgs * num_warps)
    teams_needed = args.num_wgs * num_warps;
  } else if (_type == TileAllgatherWGTestType) {
    // One team per WG
    teams_needed = args.num_wgs;
  } else {
    // Thread-level: single team
    teams_needed = 1;
  }

  for (int i = 0; i < teams_needed; i++) {
    teams[i] = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                &teams[i]);
    if (teams[i] == ROCSHMEM_TEAM_INVALID) {
      printf("PE %d: Failed to create team %d\n", args.myid, i);
      rocshmem_global_exit(1);
    }
  }
}

void TileAllgatherTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                       int loop, size_t size) {
  int n_pes = rocshmem_n_pes();

  int ke_t1 = tile_extent_1;
  if (size > 0) {
    int derived = static_cast<int>(size / (tile_extent_0 * sizeof(float)));
    if (derived >= 1 && derived <= tile_extent_1) ke_t1 = derived;
  }

  switch (_type) {
    case TileAllgatherTestType:
      hipLaunchKernelGGL(TileAllgatherThreadTest, dim3(1), blockSize, 0, stream,
                         teams[0], source, dest, tile_extent_0, ke_t1,
                         args.myid, n_pes, _shmem_context,
                         loop, args.skip, start_time, end_time, error_flag);
      break;

    case TileAllgatherWaveTestType: {
      size_t wave_shared = num_warps * sizeof(rocshmem_ctx_t);
      hipLaunchKernelGGL(TileAllgatherWaveTest, gridSize, blockSize,
                         wave_shared, stream,
                         teams, source, dest, tile_extent_0, ke_t1,
                         args.myid, n_pes, _shmem_context,
                         wf_size, num_warps,
                         loop, args.skip, start_time, end_time, error_flag);
      break;
    }

    case TileAllgatherWGTestType: {
      size_t wg_shared = sizeof(rocshmem_ctx_t);
      hipLaunchKernelGGL(TileAllgatherTest, dim3(args.num_wgs), blockSize,
                         wg_shared, stream,
                         teams, args.num_wgs, source, dest,
                         tile_extent_0, ke_t1,
                         args.myid, n_pes, _shmem_context,
                         loop, args.skip, start_time, end_time, error_flag);
      break;
    }

    default:
      fprintf(stderr, "Unknown TileAllgather test type\n");
      exit(EXIT_FAILURE);
  }

  size_t tiles_per_loop;
  if (_type == TileAllgatherWaveTestType) {
    tiles_per_loop = gridSize.x * num_warps;
  } else {
    tiles_per_loop = gridSize.x;
  }
  num_msgs       = (loop + args.skip) * tiles_per_loop;
  num_timed_msgs = loop * tiles_per_loop;
}

void TileAllgatherTester::postLaunchKernel() {
  int teams_to_destroy;
  if (_type == TileAllgatherWaveTestType) {
    teams_to_destroy = args.num_wgs * num_warps;
  } else if (_type == TileAllgatherWGTestType) {
    teams_to_destroy = args.num_wgs;
  } else {
    teams_to_destroy = 1;
  }

  for (int i = 0; i < teams_to_destroy; i++) {
    if (teams[i] != ROCSHMEM_TEAM_INVALID) {
      rocshmem_team_destroy(teams[i]);
      teams[i] = ROCSHMEM_TEAM_INVALID;
    }
  }
}

void TileAllgatherTester::verifyResults([[maybe_unused]] size_t size) {
  // Check error flag
  int h_error_flag;
  CHECK_HIP(hipMemcpy(&h_error_flag, error_flag, sizeof(int), hipMemcpyDeviceToHost));

  if (h_error_flag) {
    fprintf(stderr, "PE %d: Tile allgather verification FAILED\n", args.myid);
    exit(EXIT_FAILURE);
  }

  if (args.myid == 0 && size == max_msg_size) {
    printf("PE %d: Tile allgather verification PASSED\n", args.myid);
  }
}
