/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_cooperative_groups.h>
#include <hip_test_common.hh>

static __global__ void wg_split_barrier(float *out, float *in) {
  namespace cg = cooperative_groups;

  __shared__ float mid[32];
  size_t i = threadIdx.x;
  auto tb = cg::this_thread_block();

  // Must precede arrive: only pre-arrive stores are visible to peers after
  // their wait.
  if (i == 0) {
    for (size_t j = 0; j < 32; j++) {
      mid[j] = in[j];
    }
  }

  auto tok = tb.barrier_arrive();

  // Thread-local, so safe in the arrive/wait gap.
  out[i] = in[i] * 2.0f;

  tb.barrier_wait(std::move(tok));

  out[i] += mid[i];
}

HIP_TEST_CASE(Unit_coop_thread_block_split_barrier) {
  constexpr size_t size = 32;
  float *d_out, *d_in;

  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));

  std::vector<float> in(size, 0.0f), out = in;
  for (size_t i = 0; i < size; i++) {
    in[i] = i + 1;
  }

  HIP_CHECK(hipMemset(d_out, 0, sizeof(float) * size));
  HIP_CHECK(
      hipMemcpy(d_in, in.data(), sizeof(float) * size, hipMemcpyHostToDevice));
  wg_split_barrier<<<1, size>>>(d_out, d_in);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * size,
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_in));

  for (size_t i = 0; i < size; i++) {
    INFO("Index: " << i << " in: " << in[i] << " out: " << out[i]);
    REQUIRE((in[i] * 3.0f) == Catch::Approx(out[i]));
  }
}

static __global__ void grid_split_barrier(int *data, int *result, int N) {
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  // Must precede arrive: thread 0 reads every other block's write after the
  // wait, so writing in the gap would be a cross-block race.
  if (gid < N) {
    data[gid] = gid + 1;
  }

  auto tok = grid.barrier_arrive();
  grid.barrier_wait(std::move(tok));

  if (grid.thread_rank() == 0) {
    int sum = 0;
    for (int i = 0; i < N; i++)
      sum += data[i];
    *result = sum;
  }
}

HIP_TEST_CASE(Unit_coop_grids_split_barrier) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  if (prop.cooperativeLaunch == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
  }

  int N = 1024;
  const int threads = 128;
  const int blocks = (N + threads - 1) / threads;

  int *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, N * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_out, sizeof(int)));

  void *args[] = {&d_in, &d_out, &N};

  dim3 grid(blocks);
  dim3 block(threads);

  HIP_CHECK(hipLaunchCooperativeKernel((void *)grid_split_barrier, grid, block,
                                       args, 0, 0));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  int out = 0;
  HIP_CHECK(hipMemcpy(&out, d_out, sizeof(int), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  REQUIRE(out == ((N * (N + 1)) / 2));
}

// Multiple sequential arrive/wait pairs in one kernel: each phase produces
// data that the next phase consumes via shared memory, so any token mix-up
// or missed wait would corrupt the result.
static __global__ void wg_split_barrier_multi(float* out, const float* in,
                                              int phases) {
  namespace cg = cooperative_groups;
  extern __shared__ float scratch[];
  auto tb = cg::this_thread_block();
  size_t i = threadIdx.x;

  scratch[i] = in[i];
  for (int p = 0; p < phases; ++p) {
    auto tok = tb.barrier_arrive();
    // independent per-thread work between arrive and wait.
    float local = scratch[i] + 1.0f;
    tb.barrier_wait(std::move(tok));

    // Read a neighbour (round-robin): requires the previous phase's writes
    // to be visible to all threads — i.e. the wait actually waited.
    size_t neighbour = (i + 1) % blockDim.x;
    float nb = scratch[neighbour];

    // WAR barrier: nobody may overwrite scratch[i] until every thread has read
    // its neighbour. The sum itself is register-only, so it belongs in the gap.
    auto tok2 = tb.barrier_arrive();
    const float next = local + nb;
    tb.barrier_wait(std::move(tok2));

    scratch[i] = next;
  }
  out[i] = scratch[i];
}

HIP_TEST_CASE(Unit_coop_thread_block_split_barrier_Multiple) {
  for (const int phases : {1, 2, 4, 8}) {
    constexpr size_t size = 64;
    float *d_out, *d_in;
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));

    std::vector<float> in(size), out(size, 0.0f), expected(size);
    for (size_t i = 0; i < size; i++) in[i] = static_cast<float>(i + 1);
    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * size,
                        hipMemcpyHostToDevice));

    // Mirror device computation on host.
    expected = in;
    for (int p = 0; p < phases; ++p) {
      std::vector<float> snapshot = expected;
      for (size_t i = 0; i < size; i++) {
        float local = snapshot[i] + 1.0f;
        float nb = snapshot[(i + 1) % size];
        expected[i] = local + nb;
      }
    }

    INFO("phases: " << phases);
    wg_split_barrier_multi<<<1, size, sizeof(float) * size>>>(d_out, d_in,
                                                              phases);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_in));

    for (size_t i = 0; i < size; i++) {
      INFO("phases " << phases << " idx " << i);
      REQUIRE(expected[i] == Catch::Approx(out[i]));
    }
  }
}

// arrive+wait must be observationally equivalent to sync(): the same kernel
// run twice (once with sync, once with arrive/wait) must give identical
// results.
static __global__ void wg_sync_kernel(int* out, const int* in, int rounds) {
  extern __shared__ int s_int[];
  auto tb = cooperative_groups::this_thread_block();
  size_t i = threadIdx.x;
  s_int[i] = in[i];
  for (int r = 0; r < rounds; ++r) {
    tb.sync();
    int v = s_int[(i + 1) % blockDim.x] + r;
    tb.sync();
    s_int[i] = v;
  }
  out[i] = s_int[i];
}

static __global__ void wg_split_kernel(int* out, const int* in, int rounds) {
  extern __shared__ int s_int[];
  auto tb = cooperative_groups::this_thread_block();
  size_t i = threadIdx.x;
  s_int[i] = in[i];
  for (int r = 0; r < rounds; ++r) {
    auto t1 = tb.barrier_arrive();
    tb.barrier_wait(std::move(t1));
    int v = s_int[(i + 1) % blockDim.x] + r;
    auto t2 = tb.barrier_arrive();
    tb.barrier_wait(std::move(t2));
    s_int[i] = v;
  }
  out[i] = s_int[i];
}

HIP_TEST_CASE(Unit_coop_thread_block_split_barrier_EquivalentToSync) {
  for (const size_t size : {32u, 64u, 128u, 256u}) {
    const int rounds = 4;
    int *d_a, *d_b, *d_in;
    HIP_CHECK(hipMalloc(&d_a, sizeof(int) * size));
    HIP_CHECK(hipMalloc(&d_b, sizeof(int) * size));
    HIP_CHECK(hipMalloc(&d_in, sizeof(int) * size));

    std::vector<int> in(size), a(size), b(size);
    for (size_t i = 0; i < size; i++) in[i] = static_cast<int>(i * 3 + 7);
    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(int) * size,
                        hipMemcpyHostToDevice));

    wg_sync_kernel<<<1, size, sizeof(int) * size>>>(d_a, d_in, rounds);
    HIP_CHECK(hipGetLastError());
    wg_split_kernel<<<1, size, sizeof(int) * size>>>(d_b, d_in, rounds);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(a.data(), d_a, sizeof(int) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(b.data(), d_b, sizeof(int) * size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_in));

    for (size_t i = 0; i < size; i++) {
      INFO("size " << size << " idx " << i);
      REQUIRE(a[i] == b[i]);
    }
  }
}

// Producer/consumer pattern: half the threads write, the other half read
// across the barrier. The producer write precedes barrier_arrive(), so
// barrier_wait() is what makes it visible to the consumers.
static __global__ void wg_split_barrier_producer_consumer(int* out,
                                                          const int* in) {
  namespace cg = cooperative_groups;
  extern __shared__ int sh[];
  auto tb = cg::this_thread_block();
  size_t i = threadIdx.x;
  size_t half = blockDim.x / 2;

  // Producers (i < half) prepare data; consumers (i >= half) will read it.
  if (i < half) {
    sh[i] = in[i] * 10;
  }
  auto tok = tb.barrier_arrive();
  // Some independent local work between arrive and wait. This must NOT
  // prevent the producer's stores from being observed by the consumers.
  int local_acc = static_cast<int>(i);
  for (int k = 0; k < 4; ++k) local_acc = local_acc * 3 + k;
  tb.barrier_wait(std::move(tok));

  if (i >= half) {
    // Consumers read data the producers wrote.
    out[i] = sh[i - half] + local_acc;
  } else {
    out[i] = local_acc;
  }
}

HIP_TEST_CASE(Unit_coop_thread_block_split_barrier_ProducerConsumer) {
  constexpr size_t size = 128;
  int *d_out, *d_in;
  HIP_CHECK(hipMalloc(&d_out, sizeof(int) * size));
  HIP_CHECK(hipMalloc(&d_in, sizeof(int) * size));

  std::vector<int> in(size), out(size, -1), expected(size);
  for (size_t i = 0; i < size; i++) in[i] = static_cast<int>(i + 1);
  HIP_CHECK(
      hipMemcpy(d_in, in.data(), sizeof(int) * size, hipMemcpyHostToDevice));

  size_t half = size / 2;
  for (size_t i = 0; i < size; i++) {
    int local = static_cast<int>(i);
    for (int k = 0; k < 4; ++k) local = local * 3 + k;
    if (i >= half) {
      expected[i] = in[i - half] * 10 + local;
    } else {
      expected[i] = local;
    }
  }

  wg_split_barrier_producer_consumer<<<1, size, sizeof(int) * size>>>(d_out,
                                                                      d_in);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(int) * size,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_in));

  for (size_t i = 0; i < size; i++) {
    INFO("idx " << i);
    REQUIRE(out[i] == expected[i]);
  }
}

// Cooperative grid: multiple sequential grid-wide arrive/wait pairs.
static __global__ void grid_split_barrier_multi(int* data, int* result, int N,
                                                int phases) {
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) data[gid] = gid + 1;

  for (int p = 0; p < phases; ++p) {
    auto tok = grid.barrier_arrive();
    grid.barrier_wait(std::move(tok));
    if (gid < N) data[gid] += 1;
  }

  auto tok = grid.barrier_arrive();
  grid.barrier_wait(std::move(tok));

  if (grid.thread_rank() == 0) {
    int sum = 0;
    for (int i = 0; i < N; i++) sum += data[i];
    *result = sum;
  }
}

HIP_TEST_CASE(Unit_coop_grids_split_barrier_Multiple) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  if (prop.cooperativeLaunch == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
  }

  for (const int phases : {1, 2, 5}) {
    int N = 1024;
    const int threads = 128;
    const int blocks = (N + threads - 1) / threads;

    int *d_in, *d_out, p = phases;
    HIP_CHECK(hipMalloc(&d_in, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));
    HIP_CHECK(hipMemset(d_out, 0, sizeof(int)));

    void* args[] = {&d_in, &d_out, &N, &p};
    dim3 grid(blocks);
    dim3 block(threads);

    HIP_CHECK(hipLaunchCooperativeKernel((void*)grid_split_barrier_multi, grid,
                                         block, args, 0, 0));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    int out = 0;
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    // Expected: sum_{i=1..N}(i + phases) = N(N+1)/2 + N*phases.
    const int expected = (N * (N + 1)) / 2 + N * phases;
    INFO("phases " << phases);
    REQUIRE(out == expected);
  }
}

static __global__ void wg_split_barrier_probe(int* out, const int* in) {
  namespace cg = cooperative_groups;
  extern __shared__ int sb_probe[];
  auto tb = cg::this_thread_block();
  size_t i = threadIdx.x;

  sb_probe[i] = in[i];  // publish before arrive
  auto tok = tb.barrier_arrive();
  int local = sb_probe[i] * 2;  // independent, thread-local gap work
  tb.barrier_wait(std::move(tok));
  // Neighbour's pre-arrive write, only correct if the wait synchronised every
  // wave in the block.
  int nb = sb_probe[(i + 1) % blockDim.x];
  out[i] = local + nb;
}

HIP_TEST_CASE(Unit_coop_thread_block_split_barrier_Sanity) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  // Multi-wave blocks, so arrive/wait is not a single-wave no-op.
  for (const unsigned size : {256u, 512u, 1024u}) {
    if (size > static_cast<unsigned>(prop.maxThreadsPerBlock)) continue;

    int *d_out, *d_in;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int) * size));
    HIP_CHECK(hipMalloc(&d_in, sizeof(int) * size));

    std::vector<int> in(size), out(size, 0), expected(size);
    for (unsigned i = 0; i < size; i++) in[i] = static_cast<int>(i + 1);
    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(int) * size,
                        hipMemcpyHostToDevice));

    wg_split_barrier_probe<<<1, size, sizeof(int) * size>>>(d_out, d_in);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(int) * size,
                        hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_in));

    for (unsigned i = 0; i < size; i++) {
      expected[i] = in[i] * 2 + in[(i + 1) % size];
      INFO("arch " << prop.gcnArchName << " size " << size << " idx " << i);
      REQUIRE(out[i] == expected[i]);
    }
  }
}

// Thread 0 does far more gap work than its peers, so the light threads arrive
// long before they can be released.
static __global__ void wg_split_barrier_imbalanced(int* out, const int* in,
                                                   int heavy_iters) {
  namespace cg = cooperative_groups;
  extern __shared__ int sh_imb[];
  auto tb = cg::this_thread_block();
  size_t i = threadIdx.x;

  sh_imb[i] = in[i];  // publish before arrive
  auto tok = tb.barrier_arrive();

  const int iters = (i == 0) ? heavy_iters : 8;
  int acc = 0;
  for (int k = 0; k < iters; ++k) acc += in[i] + k;

  tb.barrier_wait(std::move(tok));

  int nb = sh_imb[(i + 1) % blockDim.x];  // consume neighbour after wait
  out[i] = nb + acc;
}

HIP_TEST_CASE(Unit_coop_thread_block_split_barrier_ImbalancedWorkload) {
  constexpr unsigned size = 256;
  const int heavy = 4096;

  int *d_out, *d_in;
  HIP_CHECK(hipMalloc(&d_out, sizeof(int) * size));
  HIP_CHECK(hipMalloc(&d_in, sizeof(int) * size));

  std::vector<int> in(size), out(size, 0), expected(size);
  for (unsigned i = 0; i < size; i++) in[i] = static_cast<int>(i + 1);
  HIP_CHECK(
      hipMemcpy(d_in, in.data(), sizeof(int) * size, hipMemcpyHostToDevice));

  for (unsigned i = 0; i < size; i++) {
    const int iters = (i == 0) ? heavy : 8;
    int acc = 0;
    for (int k = 0; k < iters; ++k) acc += in[i] + k;
    expected[i] = acc + in[(i + 1) % size];
  }

  wg_split_barrier_imbalanced<<<1, size, sizeof(int) * size>>>(d_out, d_in,
                                                               heavy);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(int) * size,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_in));

  for (unsigned i = 0; i < size; i++) {
    INFO("idx " << i);
    REQUIRE(out[i] == expected[i]);
  }
}

// The token must stay valid when arrive and wait happen in different scopes.
static __device__ cooperative_groups::thread_block::arrival_token
arrive_in_helper(const cooperative_groups::thread_block& tb) {
  return tb.barrier_arrive();
}

static __global__ void wg_split_barrier_token_lifetime(int* out,
                                                       const int* in) {
  namespace cg = cooperative_groups;
  extern __shared__ int sh_tok[];
  auto tb = cg::this_thread_block();
  size_t i = threadIdx.x;

  sh_tok[i] = in[i] + 1;  // publish before arrive
  auto tok = arrive_in_helper(tb);
  int local = in[i] * 3;  // independent gap work
  {
    tb.barrier_wait(std::move(tok));
  }
  out[i] = sh_tok[(i + 1) % blockDim.x] + local;
}

HIP_TEST_CASE(Unit_coop_thread_block_split_barrier_TokenLifetime) {
  constexpr unsigned size = 256;

  int *d_out, *d_in;
  HIP_CHECK(hipMalloc(&d_out, sizeof(int) * size));
  HIP_CHECK(hipMalloc(&d_in, sizeof(int) * size));

  std::vector<int> in(size), out(size, 0), expected(size);
  for (unsigned i = 0; i < size; i++) in[i] = static_cast<int>(i + 1);
  HIP_CHECK(
      hipMemcpy(d_in, in.data(), sizeof(int) * size, hipMemcpyHostToDevice));

  for (unsigned i = 0; i < size; i++) {
    expected[i] = (in[(i + 1) % size] + 1) + in[i] * 3;
  }

  wg_split_barrier_token_lifetime<<<1, size, sizeof(int) * size>>>(d_out, d_in);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(int) * size,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_in));

  for (unsigned i = 0; i < size; i++) {
    INFO("idx " << i);
    REQUIRE(out[i] == expected[i]);
  }
}
