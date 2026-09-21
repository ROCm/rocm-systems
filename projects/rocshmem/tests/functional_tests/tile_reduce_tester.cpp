/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

// This file is included by tile_reduce_tester.hpp and must not be compiled
// directly — all rocshmem headers are brought in by the including translation unit.
#ifndef TILE_REDUCE_TESTER_HPP
#error "Include tile_reduce_tester.hpp, not tile_reduce_tester.cpp directly"
#endif

using namespace rocshmem;

/******************************************************************************
 * TENSOR / COORDINATE HELPERS
 * Use the same Tensor2D / Tuple2D structs as the other tile testers so that
 * the instantiated symbols match those already compiled into the library.
 ******************************************************************************/

template <typename T>
struct Tensor2D {
  using element_type = T;
  static constexpr int ndim = 2;

  T *data;
  int rows, cols, row_stride, col_stride;

  __device__ Tensor2D(T *d, int r, int c, int rs = -1, int cs = 1)
      : data(d), rows(r), cols(c), col_stride(cs) {
    row_stride = (rs == -1) ? c : rs;
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

/******************************************************************************
 * TILE REDUCE API DISPATCH
 *
 * Maps <T, Op> -> the appropriate rocshmem_ctx_tile_*_reduce_{wave/wg/thread}
 * call.  Three scopes (Lane, Wave, Wg) each get their own dispatch template.
 ******************************************************************************/

// ── Generic stubs (intentionally unimplemented — only specializations fire) ─
template <typename T, ROCSHMEM_OP Op>
__device__ int tile_reduce_lane(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                Tensor2D<T> dst,
                                Tensor2D<T> src,
                                Tuple2D start,
                                Tuple2D boundary, int root) {
  return ROCSHMEM_SUCCESS;
}

template <typename T, ROCSHMEM_OP Op>
__device__ int tile_reduce_wave(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                Tensor2D<T> dst,
                                Tensor2D<T> src,
                                Tuple2D start,
                                Tuple2D boundary, int root) {
  return ROCSHMEM_SUCCESS;
}

template <typename T, ROCSHMEM_OP Op>
__device__ int tile_reduce_wg(rocshmem_ctx_t ctx, rocshmem_team_t team,
                              Tensor2D<T> dst,
                              Tensor2D<T> src,
                              Tuple2D start,
                              Tuple2D boundary, int root) {
  return ROCSHMEM_SUCCESS;
}

// ── Specialisation macro ─────────────────────────────────────────────────────
#define TILE_REDUCE_DEF_GEN(T, TNAME, OP_NAME, OP)                            \
  template <>                                                                  \
  __device__ int tile_reduce_lane<T, OP>(                                      \
      rocshmem_ctx_t ctx, rocshmem_team_t team,                                \
      Tensor2D<T> dst, Tensor2D<T> src,                    \
      Tuple2D start, Tuple2D boundary, int root) {         \
    return rocshmem_ctx_tile_##OP_NAME##_reduce(ctx, team, dst, src,           \
                                                start, boundary, root, 0);     \
  }                                                                            \
  template <>                                                                  \
  __device__ int tile_reduce_wave<T, OP>(                                      \
      rocshmem_ctx_t ctx, rocshmem_team_t team,                                \
      Tensor2D<T> dst, Tensor2D<T> src,                    \
      Tuple2D start, Tuple2D boundary, int root) {         \
    return rocshmem_ctx_tile_##OP_NAME##_reduce_wave(ctx, team, dst, src,      \
                                                     start, boundary, root, 0);\
  }                                                                            \
  template <>                                                                  \
  __device__ int tile_reduce_wg<T, OP>(                                        \
      rocshmem_ctx_t ctx, rocshmem_team_t team,                                \
      Tensor2D<T> dst, Tensor2D<T> src,                    \
      Tuple2D start, Tuple2D boundary, int root) {         \
    return rocshmem_ctx_tile_##OP_NAME##_reduce_wg(ctx, team, dst, src,        \
                                                   start, boundary, root, 0);  \
  }

TILE_REDUCE_DEF_GEN(float, float, sum, ROCSHMEM_SUM)
TILE_REDUCE_DEF_GEN(float, float, max, ROCSHMEM_MAX)
TILE_REDUCE_DEF_GEN(float, float, min, ROCSHMEM_MIN)
TILE_REDUCE_DEF_GEN(short, short, sum, ROCSHMEM_SUM)
TILE_REDUCE_DEF_GEN(short, short, max, ROCSHMEM_MAX)
TILE_REDUCE_DEF_GEN(short, short, min, ROCSHMEM_MIN)
TILE_REDUCE_DEF_GEN(int,   int,   sum, ROCSHMEM_SUM)
TILE_REDUCE_DEF_GEN(int,   int,   max, ROCSHMEM_MAX)
TILE_REDUCE_DEF_GEN(int,   int,   min, ROCSHMEM_MIN)
TILE_REDUCE_DEF_GEN(long,  long,  sum, ROCSHMEM_SUM)
TILE_REDUCE_DEF_GEN(long,  long,  max, ROCSHMEM_MAX)
TILE_REDUCE_DEF_GEN(long,  long,  min, ROCSHMEM_MIN)

/******************************************************************************
 * DEVICE KERNELS
 ******************************************************************************/

template <typename T, ROCSHMEM_OP Op>
__global__ void TileReduceThreadTest(rocshmem_team_t team,
                                     T *s_buf, T *r_buf,
                                     int tile_extent_0, int tile_extent_1,
                                     int root, ShmemContextType ctx_type,
                                     int loop, int skip,
                                     long long int *start_time,
                                     long long int *end_time,
                                     int *error_flag) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();

  rocshmem_wg_team_create_ctx(team, ctx_type, &ctx);

  int matrix_size = tile_extent_0 * tile_extent_1;
  Tensor2D<T> src(s_buf, tile_extent_0, tile_extent_1);
  Tensor2D<T> dst(r_buf, tile_extent_0, tile_extent_1);
  Tuple2D start_coord(0, 0);
  Tuple2D boundary(tile_extent_0, tile_extent_1);

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && threadIdx.x == 0) {
      start_time[wg_id] = wall_clock64();
    }
    if (threadIdx.x == 0) {
      tile_reduce_lane<T, Op>(ctx, team, dst, src, start_coord, boundary, root);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    end_time[wg_id] = wall_clock64();
  }

  // Verify on root PE
  if (threadIdx.x == 0) {
    for (int idx = 0; idx < matrix_size; idx++) {
      T actual = r_buf[idx];
      // Verification done host-side via error_flag; kernel just signals error.
      (void)actual;
    }
  }

  __syncthreads();
  rocshmem_wg_ctx_destroy(&ctx);
}

template <typename T, ROCSHMEM_OP Op>
__global__ void TileReduceWaveTest(rocshmem_team_t *teams,
                                   T *s_buf, T *r_buf,
                                   int tile_extent_0, int tile_extent_1,
                                   int root, ShmemContextType ctx_type,
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

  int matrix_size = tile_extent_0 * tile_extent_1;
  int offset = flat_wf_id * matrix_size;

  Tensor2D<T> src(s_buf + offset, tile_extent_0, tile_extent_1);
  Tensor2D<T> dst(r_buf + offset, tile_extent_0, tile_extent_1);
  Tuple2D start_coord(0, 0);
  Tuple2D boundary(tile_extent_0, tile_extent_1);

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && t_id % wf_size == 0) {
      start_time[flat_wf_id] = wall_clock64();
    }
    tile_reduce_wave<T, Op>(ctx_array[wf_id], teams[flat_wf_id],
                            dst, src, start_coord, boundary, root);
  }
  __syncthreads();

  if (t_id % wf_size == 0) {
    end_time[flat_wf_id] = wall_clock64();
  }

  __syncthreads();

  for (int wf_i = 0; wf_i < num_waves_per_wg; wf_i++) {
    rocshmem_wg_ctx_destroy(&ctx_array[wf_i]);
    __syncthreads();
  }
}

template <typename T, ROCSHMEM_OP Op>
__global__ void TileReduceWGTest(rocshmem_team_t *teams, int num_teams,
                                 T *s_buf, T *r_buf,
                                 int tile_extent_0, int tile_extent_1,
                                 int root, ShmemContextType ctx_type,
                                 int loop, int skip,
                                 long long int *start_time,
                                 long long int *end_time,
                                 int *error_flag) {
  extern __shared__ rocshmem_ctx_t ctx_array[];
  int wg_id = get_flat_grid_id();

  rocshmem_team_t my_team = teams[wg_id % num_teams];
  rocshmem_wg_team_create_ctx(my_team, ctx_type, &ctx_array[0]);

  int matrix_size = tile_extent_0 * tile_extent_1;
  int offset = matrix_size * wg_id;

  Tensor2D<T> src(s_buf + offset, tile_extent_0, tile_extent_1);
  Tensor2D<T> dst(r_buf + offset, tile_extent_0, tile_extent_1);
  Tuple2D start_coord(0, 0);
  Tuple2D boundary(tile_extent_0, tile_extent_1);

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && threadIdx.x == 0) {
      start_time[wg_id] = wall_clock64();
    }
    tile_reduce_wg<T, Op>(ctx_array[0], my_team,
                          dst, src, start_coord, boundary, root);
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    end_time[wg_id] = wall_clock64();
  }

  __syncthreads();
  rocshmem_ctx_sync_wg(ctx_array[0], my_team);
  rocshmem_wg_ctx_destroy(&ctx_array[0]);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 ******************************************************************************/

template <typename T, ROCSHMEM_OP Op>
TileReduceTester<T, Op>::TileReduceTester(
    TesterArguments args,
    std::function<void(T &, T &)> f1,
    std::function<bool(T, int, int)> f2)
    : Tester(args), init_buf(f1), verify_buf(f2) {

  // When --num-wf is used, args.wg_size is updated after the base constructor
  // sets num_warps, so we must recompute the actual waves-per-WG here.
  int actual_warps = (args.num_wf > 0) ? args.num_wf : num_warps;

  if (_type == TileReduceWaveTestType) {
    num_teams  = args.num_wgs * actual_warps;
    num_timers = args.num_wgs * actual_warps;
    num_warps  = actual_warps;  // keep consistent for launchKernel
  } else if (_type == TileReduceWGTestType) {
    num_teams = args.num_wgs;
  } else {
    num_teams = 1;
  }

  CHECK_HIP(hipHostMalloc(&teams, num_teams * sizeof(rocshmem_team_t)));
  for (int i = 0; i < num_teams; i++) teams[i] = ROCSHMEM_TEAM_INVALID;

  // Derive tile dimensions from max_msg_size if provided.
  tile_extent_0 = 8;
  tile_extent_1 = 8;
  if (args.max_msg_size_set) {
    int derived = static_cast<int>(args.max_msg_size / (tile_extent_0 * sizeof(T)));
    if (derived >= 1) tile_extent_1 = derived;
  }
  size_t tile_size_bytes = static_cast<size_t>(tile_extent_0) * tile_extent_1 * sizeof(T);
  size_t row_bytes       = static_cast<size_t>(tile_extent_0) * sizeof(T);
  this->args.min_msg_size = row_bytes;
  max_msg_size            = tile_size_bytes;

  int tile_size  = tile_extent_0 * tile_extent_1;
  int total_size = tile_size * num_teams;

  s_buf = (T *)rocshmem_malloc(total_size * sizeof(T));
  r_buf = (T *)rocshmem_malloc(total_size * sizeof(T));

  if (!s_buf || !r_buf) {
    fprintf(stderr, "TileReduceTester: failed to allocate symmetric memory\n");
    exit(EXIT_FAILURE);
  }

  CHECK_HIP(hipMalloc(&error_flag, sizeof(int)));
}

template <typename T, ROCSHMEM_OP Op>
TileReduceTester<T, Op>::~TileReduceTester() {
  rocshmem_free(s_buf);
  rocshmem_free(r_buf);
  CHECK_HIP(hipFree(error_flag));
  for (int i = 0; i < num_teams; i++) {
    if (teams[i] != ROCSHMEM_TEAM_INVALID) {
      rocshmem_team_destroy(teams[i]);
    }
  }
  CHECK_HIP(hipHostFree(teams));
}

template <typename T, ROCSHMEM_OP Op>
void TileReduceTester<T, Op>::resetBuffers(size_t size) {
  int t1 = tile_extent_1;
  if (size > 0) {
    int derived = static_cast<int>(size / (tile_extent_0 * sizeof(T)));
    if (derived >= 1 && derived <= tile_extent_1) t1 = derived;
  }
  int tile_size     = tile_extent_0 * t1;
  int max_tile_size = tile_extent_0 * tile_extent_1;
  int total_size    = max_tile_size * num_teams;

  for (int i = 0; i < total_size; i++) {
    T s_val{}, r_val{};
    init_buf(s_val, r_val);
    s_buf[i] = s_val;
    r_buf[i] = r_val;
  }

  int zero = 0;
  CHECK_HIP(hipMemcpy(error_flag, &zero, sizeof(int), hipMemcpyHostToDevice));
}

template <typename T, ROCSHMEM_OP Op>
void TileReduceTester<T, Op>::preLaunchKernel() {
  int n_pes = rocshmem_n_pes();

  for (int i = 0; i < num_teams; i++) {
    teams[i] = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                &teams[i]);
    if (teams[i] == ROCSHMEM_TEAM_INVALID) {
      printf("PE %d: TileReduceTester failed to create team %d\n",
             args.myid, i);
      rocshmem_global_exit(1);
    }
  }
}

template <typename T, ROCSHMEM_OP Op>
void TileReduceTester<T, Op>::launchKernel(dim3 gridSize, dim3 blockSize,
                                           int loop, size_t size) {
  int root = 0;

  int ke_t1 = tile_extent_1;
  if (size > 0) {
    int derived = static_cast<int>(size / (tile_extent_0 * sizeof(T)));
    if (derived >= 1 && derived <= tile_extent_1) ke_t1 = derived;
  }

  switch (_type) {
    case TileReduceTestType:
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(TileReduceThreadTest<T, Op>),
          dim3(1), blockSize, 0, stream,
          teams[0], s_buf, r_buf, tile_extent_0, ke_t1, root,
          _shmem_context, loop, args.skip, start_time, end_time, error_flag);
      break;

    case TileReduceWaveTestType: {
      size_t wave_shared = num_warps * sizeof(rocshmem_ctx_t);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(TileReduceWaveTest<T, Op>),
          gridSize, blockSize, wave_shared, stream,
          teams, s_buf, r_buf, tile_extent_0, ke_t1, root,
          _shmem_context, wf_size, num_warps,
          loop, args.skip, start_time, end_time, error_flag);
      break;
    }

    case TileReduceWGTestType: {
      size_t wg_shared = sizeof(rocshmem_ctx_t);
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(TileReduceWGTest<T, Op>),
          dim3(args.num_wgs), blockSize, wg_shared, stream,
          teams, args.num_wgs, s_buf, r_buf, tile_extent_0, ke_t1, root,
          _shmem_context, loop, args.skip, start_time, end_time, error_flag);
      break;
    }

    default:
      fprintf(stderr, "TileReduceTester: unknown test type\n");
      exit(EXIT_FAILURE);
  }

  size_t tiles_per_loop;
  if (_type == TileReduceWaveTestType) {
    tiles_per_loop = gridSize.x * num_warps;
  } else {
    tiles_per_loop = gridSize.x;
  }
  num_msgs       = (loop + args.skip) * tiles_per_loop;
  num_timed_msgs = loop * tiles_per_loop;
}

template <typename T, ROCSHMEM_OP Op>
void TileReduceTester<T, Op>::postLaunchKernel() {
  for (int i = 0; i < num_teams; i++) {
    if (teams[i] != ROCSHMEM_TEAM_INVALID) {
      rocshmem_team_destroy(teams[i]);
      teams[i] = ROCSHMEM_TEAM_INVALID;
    }
  }
}

template <typename T, ROCSHMEM_OP Op>
void TileReduceTester<T, Op>::verifyResults(size_t size) {
  int t1 = tile_extent_1;
  if (size > 0) {
    int derived = static_cast<int>(size / (tile_extent_0 * sizeof(T)));
    if (derived >= 1 && derived <= tile_extent_1) t1 = derived;
  }
  int tile_size = tile_extent_0 * t1;
  int n_pes     = rocshmem_n_pes();

  // Only root PE (PE 0) holds the reduction result.
  if (args.myid != 0) return;

  // Verify each slot's result tile.
  size_t tiles_per_loop;
  if (_type == TileReduceWaveTestType) {
    tiles_per_loop = static_cast<size_t>(args.num_wgs) * num_warps;
  } else {
    tiles_per_loop = args.num_wgs;
  }

  bool failed = false;
  for (size_t slot = 0; slot < tiles_per_loop; slot++) {
    int base = static_cast<int>(slot) * tile_size;
    for (int idx = 0; idx < tile_size; idx++) {
      T actual = r_buf[base + idx];
      if (!verify_buf(actual, n_pes, idx)) {
        fprintf(stderr,
                "PE 0: Tile reduce verification failed at slot %zu idx %d: "
                "got %lld\n",
                slot, idx, static_cast<long long>(actual));
        failed = true;
      }
    }
  }

  if (failed) {
    exit(EXIT_FAILURE);
  }

  if (size == max_msg_size) {
    printf("PE 0: Tile reduce verification PASSED\n");
  }
}
