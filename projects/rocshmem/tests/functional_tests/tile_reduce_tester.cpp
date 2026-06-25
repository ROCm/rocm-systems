/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "tile_reduce_tester.hpp"

#include <rocshmem/rocshmem.hpp>

#include "../../src/context_incl.hpp"

#include <rocshmem/rocshmem_TILE_impl.hpp>

using namespace rocshmem;

template <typename T>
struct Tensor2D {
  using element_type = T;
  static constexpr int ndim = 2;

  T *data;
  int rows;
  int cols;
  int row_stride;
  int col_stride;

  __device__ Tensor2D(T *data_, int rows_, int cols_, int row_stride_ = -1,
                      int col_stride_ = 1)
      : data(data_), rows(rows_), cols(cols_), col_stride(col_stride_) {
    row_stride = (row_stride_ == -1) ? cols : row_stride_;
  }

  __device__ T *data_handle() const { return data; }
  __device__ int stride(int dim) const {
    return (dim == 0) ? row_stride : col_stride;
  }
};

struct Tuple2D {
  int x, y;
  __device__ Tuple2D(int x_, int y_) : x(x_), y(y_) {}
  __device__ int get(int dim) const { return (dim == 0) ? x : y; }
};

__device__ float expected_sum_value(int idx, int n_pes) {
  return 100.0f * (n_pes * (n_pes - 1) / 2) + n_pes * idx;
}

__device__ float expected_max_value(int idx, int n_pes) {
  return (n_pes - 1) * 100.0f + idx;
}

__device__ float expected_min_value(int idx) { return static_cast<float>(idx); }

__device__ void verify_reduce_results(const char *label, float *sum_dest,
                                      float *max_dest, float *min_dest,
                                      int tile_extent_0, int tile_extent_1,
                                      int my_world_pe, int n_pes,
                                      int *error_flag) {
  int matrix_size = tile_extent_0 * tile_extent_1;
  for (int idx = 0; idx < matrix_size; idx++) {
    float expected_sum = expected_sum_value(idx, n_pes);
    float expected_max = expected_max_value(idx, n_pes);
    float expected_min = expected_min_value(idx);

    if (sum_dest[idx] != expected_sum || max_dest[idx] != expected_max ||
        min_dest[idx] != expected_min) {
      printf("%s: PE %d verification failed at [%d]: "
             "sum got %f expected %f, max got %f expected %f, "
             "min got %f expected %f\n",
             label, my_world_pe, idx, sum_dest[idx], expected_sum,
             max_dest[idx], expected_max, min_dest[idx], expected_min);
      *error_flag = 1;
      return;
    }
  }
}

__global__ void TileReduceThreadTest(rocshmem_team_t team, float *source,
                                     float *sum_dest, float *max_dest,
                                     float *min_dest, int tile_extent_0,
                                     int tile_extent_1, int my_world_pe,
                                     int n_pes, int root,
                                     ShmemContextType ctx_type,
                                     int *error_flag) {
  __shared__ rocshmem_ctx_t ctx;

  rocshmem_wg_team_create_ctx(team, ctx_type, &ctx);

  Tensor2D<float> src_tensor(source, tile_extent_0, tile_extent_1);
  Tensor2D<float> sum_tensor(sum_dest, tile_extent_0, tile_extent_1);
  Tensor2D<float> max_tensor(max_dest, tile_extent_0, tile_extent_1);
  Tensor2D<float> min_tensor(min_dest, tile_extent_0, tile_extent_1);
  Tuple2D start(0, 0);
  Tuple2D boundary(tile_extent_0, tile_extent_1);

  if (threadIdx.x == 0) {
    if (rocshmem_ctx_tile_sum_reduce(ctx, team, sum_tensor, src_tensor, start,
                                     boundary, root, 0) != ROCSHMEM_SUCCESS ||
        rocshmem_ctx_tile_max_reduce(ctx, team, max_tensor, src_tensor, start,
                                     boundary, root, 0) != ROCSHMEM_SUCCESS ||
        rocshmem_ctx_tile_min_reduce(ctx, team, min_tensor, src_tensor, start,
                                     boundary, root, 0) != ROCSHMEM_SUCCESS) {
      *error_flag = 1;
    }
  }
  __syncthreads();

  if (my_world_pe == root && threadIdx.x == 0) {
    verify_reduce_results("Thread-level", sum_dest, max_dest, min_dest,
                          tile_extent_0, tile_extent_1, my_world_pe, n_pes,
                          error_flag);
  }

  __syncthreads();
  rocshmem_wg_ctx_destroy(&ctx);
}

__global__ void TileReduceWaveTest(rocshmem_team_t team, float *source,
                                   float *sum_dest, float *max_dest,
                                   float *min_dest, int tile_extent_0,
                                   int tile_extent_1, int my_world_pe,
                                   int n_pes, int root,
                                   ShmemContextType ctx_type, int wf_size,
                                   int *error_flag) {
  __shared__ rocshmem_ctx_t ctx;

  rocshmem_wg_team_create_ctx(team, ctx_type, &ctx);

  Tensor2D<float> src_tensor(source, tile_extent_0, tile_extent_1);
  Tensor2D<float> sum_tensor(sum_dest, tile_extent_0, tile_extent_1);
  Tensor2D<float> max_tensor(max_dest, tile_extent_0, tile_extent_1);
  Tensor2D<float> min_tensor(min_dest, tile_extent_0, tile_extent_1);
  Tuple2D start(0, 0);
  Tuple2D boundary(tile_extent_0, tile_extent_1);

  if (threadIdx.x < wf_size) {
    if (rocshmem_ctx_tile_sum_reduce_wave(ctx, team, sum_tensor, src_tensor,
                                          start, boundary, root, 0) !=
            ROCSHMEM_SUCCESS ||
        rocshmem_ctx_tile_max_reduce_wave(ctx, team, max_tensor, src_tensor,
                                          start, boundary, root, 0) !=
            ROCSHMEM_SUCCESS ||
        rocshmem_ctx_tile_min_reduce_wave(ctx, team, min_tensor, src_tensor,
                                          start, boundary, root, 0) !=
            ROCSHMEM_SUCCESS) {
      *error_flag = 1;
    }
  }
  __syncthreads();

  if (my_world_pe == root && threadIdx.x == 0) {
    verify_reduce_results("Wave-level", sum_dest, max_dest, min_dest,
                          tile_extent_0, tile_extent_1, my_world_pe, n_pes,
                          error_flag);
  }

  __syncthreads();
  rocshmem_wg_ctx_destroy(&ctx);
}

__global__ void TileReduceTest(rocshmem_team_t *teams, int num_teams,
                               float *source, float *sum_dest, float *max_dest,
                               float *min_dest, int tile_extent_0,
                               int tile_extent_1, int my_world_pe, int n_pes,
                               int root, ShmemContextType ctx_type,
                               int *error_flag) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  rocshmem_team_t my_team = teams[wg_id % num_teams];

  rocshmem_wg_team_create_ctx(my_team, ctx_type, &ctx);

  int matrix_size = tile_extent_0 * tile_extent_1;
  int offset = matrix_size * wg_id;

  Tensor2D<float> src_tensor(source + offset, tile_extent_0, tile_extent_1);
  Tensor2D<float> sum_tensor(sum_dest + offset, tile_extent_0, tile_extent_1);
  Tensor2D<float> max_tensor(max_dest + offset, tile_extent_0, tile_extent_1);
  Tensor2D<float> min_tensor(min_dest + offset, tile_extent_0, tile_extent_1);
  Tuple2D start(0, 0);
  Tuple2D boundary(tile_extent_0, tile_extent_1);

  if (rocshmem_ctx_tile_sum_reduce_wg(ctx, my_team, sum_tensor, src_tensor,
                                      start, boundary, root, 0) !=
          ROCSHMEM_SUCCESS ||
      rocshmem_ctx_tile_max_reduce_wg(ctx, my_team, max_tensor, src_tensor,
                                      start, boundary, root, 0) !=
          ROCSHMEM_SUCCESS ||
      rocshmem_ctx_tile_min_reduce_wg(ctx, my_team, min_tensor, src_tensor,
                                      start, boundary, root, 0) !=
          ROCSHMEM_SUCCESS) {
    if (threadIdx.x == 0) {
      *error_flag = 1;
    }
  }
  __syncthreads();

  if (my_world_pe == root && threadIdx.x == 0) {
    verify_reduce_results("WG-level", sum_dest + offset, max_dest + offset,
                          min_dest + offset, tile_extent_0, tile_extent_1,
                          my_world_pe, n_pes, error_flag);
  }

  __syncthreads();
  rocshmem_ctx_sync_wg(ctx, my_team);
  rocshmem_wg_ctx_destroy(&ctx);
}

TileReduceTester::TileReduceTester(TesterArguments args) : Tester(args) {
  num_teams = (_type == TileReduceWGTestType) ? args.num_wgs : 1;

  CHECK_HIP(hipHostMalloc(&teams, num_teams * sizeof(rocshmem_team_t)));
  for (int i = 0; i < num_teams; i++) {
    teams[i] = ROCSHMEM_TEAM_INVALID;
  }

  tile_extent_0 = 8;
  tile_extent_1 = 8;
  int tile_size = tile_extent_0 * tile_extent_1;
  int total_size = tile_size * num_teams;

  source = (float *)rocshmem_malloc(total_size * sizeof(float));
  sum_dest = (float *)rocshmem_malloc(total_size * sizeof(float));
  max_dest = (float *)rocshmem_malloc(total_size * sizeof(float));
  min_dest = (float *)rocshmem_malloc(total_size * sizeof(float));

  if (!source || !sum_dest || !max_dest || !min_dest) {
    fprintf(stderr, "Failed to allocate symmetric memory for tile reductions\n");
    exit(EXIT_FAILURE);
  }

  CHECK_HIP(hipMalloc(&error_flag, sizeof(int)));
}

TileReduceTester::~TileReduceTester() {
  rocshmem_free(source);
  rocshmem_free(sum_dest);
  rocshmem_free(max_dest);
  rocshmem_free(min_dest);
  CHECK_HIP(hipFree(error_flag));

  for (int i = 0; i < num_teams; i++) {
    if (teams[i] != ROCSHMEM_TEAM_INVALID) {
      rocshmem_team_destroy(teams[i]);
    }
  }
  CHECK_HIP(hipHostFree(teams));
}

void TileReduceTester::resetBuffers([[maybe_unused]] size_t size) {
  int tile_size = tile_extent_0 * tile_extent_1;
  int total_size = tile_size * num_teams;

  for (int i = 0; i < total_size; i++) {
    source[i] = args.myid * 100.0f + (i % tile_size);
    sum_dest[i] = -1.0f;
    max_dest[i] = -1.0f;
    min_dest[i] = -1.0f;
  }

  int zero = 0;
  CHECK_HIP(hipMemcpy(error_flag, &zero, sizeof(int), hipMemcpyHostToDevice));
}

void TileReduceTester::preLaunchKernel() {
  int n_pes = rocshmem_n_pes();

  if (_type == TileReduceWGTestType) {
    for (int i = 0; i < num_teams; i++) {
      teams[i] = ROCSHMEM_TEAM_INVALID;
      rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                  &teams[i]);

      if (teams[i] == ROCSHMEM_TEAM_INVALID) {
        printf("PE %d: Failed to create team %d\n", args.myid, i);
        rocshmem_global_exit(1);
      }
    }
  } else {
    teams[0] = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                &teams[0]);

    if (teams[0] == ROCSHMEM_TEAM_INVALID) {
      printf("PE %d: Failed to create team\n", args.myid);
      rocshmem_global_exit(1);
    }
  }
}

void TileReduceTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                    [[maybe_unused]] int loop,
                                    [[maybe_unused]] size_t size) {
  int root = 0;
  int n_pes = rocshmem_n_pes();

  switch (_type) {
    case TileReduceTestType:
      hipLaunchKernelGGL(TileReduceThreadTest, dim3(1), blockSize, 0, stream,
                         teams[0], source, sum_dest, max_dest, min_dest,
                         tile_extent_0, tile_extent_1, args.myid, n_pes, root,
                         _shmem_context, error_flag);
      break;

    case TileReduceWaveTestType:
      hipLaunchKernelGGL(TileReduceWaveTest, dim3(1), blockSize, 0, stream,
                         teams[0], source, sum_dest, max_dest, min_dest,
                         tile_extent_0, tile_extent_1, args.myid, n_pes, root,
                         _shmem_context, wf_size, error_flag);
      break;

    case TileReduceWGTestType:
      hipLaunchKernelGGL(TileReduceTest, dim3(num_teams), blockSize, 0, stream,
                         teams, num_teams, source, sum_dest, max_dest, min_dest,
                         tile_extent_0, tile_extent_1, args.myid, n_pes, root,
                         _shmem_context, error_flag);
      break;

    default:
      fprintf(stderr, "Unknown TileReduce test type\n");
      exit(EXIT_FAILURE);
  }
}

void TileReduceTester::postLaunchKernel() {
  if (_type == TileReduceWGTestType) {
    for (int i = 0; i < num_teams; i++) {
      if (teams[i] != ROCSHMEM_TEAM_INVALID) {
        rocshmem_team_destroy(teams[i]);
        teams[i] = ROCSHMEM_TEAM_INVALID;
      }
    }
  } else if (teams[0] != ROCSHMEM_TEAM_INVALID) {
    rocshmem_team_destroy(teams[0]);
    teams[0] = ROCSHMEM_TEAM_INVALID;
  }
}

void TileReduceTester::verifyResults([[maybe_unused]] size_t size) {
  int h_error_flag;
  CHECK_HIP(
      hipMemcpy(&h_error_flag, error_flag, sizeof(int), hipMemcpyDeviceToHost));

  if (h_error_flag) {
    fprintf(stderr, "PE %d: Tile reduce verification FAILED\n", args.myid);
    exit(EXIT_FAILURE);
  }

  if (args.myid == 0) {
    printf("PE %d: Tile reduce verification PASSED\n", args.myid);
  }
}
