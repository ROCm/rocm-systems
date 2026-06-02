// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_rccl_test.cpp
/// @brief RCCL collective communication tests on simulated multi-process GPU.
///
/// Each process is one rank in a collective. Rank 0 generates an ncclUniqueId
/// and writes it to a shared file; all ranks read the ID and call
/// ncclCommInitRank. Each test validates the RCCL result against a
/// single-threaded host-computed golden reference using random fuzz inputs.
///
/// Compiled with hipcc, linked with -lrccl. Requires LD_PRELOAD=librocjitsu_kmd.so.
/// Arguments: --rank=N --world-size=M --shared-dir=DIR [--seed=S]

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

static int g_rank = 0;
static int g_world_size = 2;
static std::string g_shared_dir;
static uint32_t g_base_seed = 42;
static constexpr int kFuzzIters = 3;
static constexpr float kTolerance = 1e-4f;

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

#define NCCL_ASSERT(call)                                                                          \
  do {                                                                                             \
    ncclResult_t res = (call);                                                                     \
    ASSERT_EQ(res, ncclSuccess) << "NCCL error: " << ncclGetErrorString(res);                      \
  } while (0)

__global__ void copy_kernel(float *dst, const float *src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = src[i];
}

static std::vector<float> random_floats(uint32_t seed, int n) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  std::vector<float> v(n);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

static int fuzz_size(uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(64, 1024);
  int n = dist(rng);
  return (n / g_world_size) * g_world_size;
}

static void write_file(const std::string &path, const void *data, size_t size) {
  std::ofstream ofs(path, std::ios::binary);
  ofs.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
}

static void read_file_blocking(const std::string &path, void *data, size_t size) {
  using namespace std::chrono_literals;
  for (int i = 0; i < 200; ++i) {
    if (std::filesystem::exists(path) && std::filesystem::file_size(path) >= size) {
      std::ifstream ifs(path, std::ios::binary);
      ifs.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
      if (ifs.good())
        return;
    }
    std::this_thread::sleep_for(50ms);
  }
  FAIL() << "Timeout waiting for " << path;
}

static void barrier(const std::string &name) {
  std::string ready = g_shared_dir + "/" + name + ".ready." + std::to_string(g_rank);
  {
    std::ofstream(ready) << "1";
  }
  for (int r = 0; r < g_world_size; ++r) {
    std::string other = g_shared_dir + "/" + name + ".ready." + std::to_string(r);
    while (!std::filesystem::exists(other))
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

static std::vector<float> all_ranks_input(uint32_t iter_seed, int n) {
  std::vector<float> all(n * g_world_size);
  for (int r = 0; r < g_world_size; ++r) {
    auto v = random_floats(iter_seed + r, n);
    std::copy(v.begin(), v.end(), all.begin() + r * n);
  }
  return all;
}

class RcclTest : public ::testing::Test {
protected:
  void SetUp() override {
    ncclUniqueId id{};
    if (g_rank == 0) {
      NCCL_ASSERT(ncclGetUniqueId(&id));
      write_file(g_shared_dir + "/nccl_id", &id, sizeof(id));
    }
    barrier("nccl_init");
    if (g_rank != 0)
      read_file_blocking(g_shared_dir + "/nccl_id", &id, sizeof(id));
    NCCL_ASSERT(ncclCommInitRank(&comm_, g_world_size, id, g_rank));
    HIP_ASSERT(hipStreamCreate(&stream_));
  }

  void TearDown() override {
    if (stream_) {
      (void)hipStreamDestroy(stream_);
      stream_ = nullptr;
    }
    if (comm_) {
      (void)ncclCommDestroy(comm_);
      comm_ = nullptr;
    }
  }

  ncclComm_t comm_ = nullptr;
  hipStream_t stream_ = nullptr;
};

TEST_F(RcclTest, AllReduce) {
  for (int iter = 0; iter < kFuzzIters; ++iter) {
    uint32_t size_seed = g_base_seed + 100 * iter;
    int n = fuzz_size(size_seed);
    uint32_t iter_seed = g_base_seed + 1000 * iter;
    auto local = random_floats(iter_seed + g_rank, n);

    float *d_send = nullptr, *d_recv = nullptr;
    HIP_ASSERT(hipMalloc(&d_send, n * sizeof(float)));
    HIP_ASSERT(hipMalloc(&d_recv, n * sizeof(float)));
    HIP_ASSERT(hipMemcpy(d_send, local.data(), n * sizeof(float), hipMemcpyHostToDevice));

    NCCL_ASSERT(ncclAllReduce(d_send, d_recv, n, ncclFloat, ncclSum, comm_, stream_));
    HIP_ASSERT(hipStreamSynchronize(stream_));

    std::vector<float> result(n);
    HIP_ASSERT(hipMemcpy(result.data(), d_recv, n * sizeof(float), hipMemcpyDeviceToHost));

    auto all = all_ranks_input(iter_seed, n);
    std::vector<float> ref(n, 0.0f);
    for (int r = 0; r < g_world_size; ++r)
      for (int i = 0; i < n; ++i)
        ref[i] += all[r * n + i];

    for (int i = 0; i < n; ++i)
      EXPECT_NEAR(result[i], ref[i], kTolerance)
          << "iter=" << iter << " rank=" << g_rank << " i=" << i;

    (void)hipFree(d_send);
    (void)hipFree(d_recv);
  }
}

TEST_F(RcclTest, Broadcast) {
  for (int iter = 0; iter < kFuzzIters; ++iter) {
    uint32_t size_seed = g_base_seed + 100 * iter + 10;
    int n = fuzz_size(size_seed);
    uint32_t iter_seed = g_base_seed + 1000 * iter + 10;
    auto root_data = random_floats(iter_seed, n);

    float *d_buf = nullptr;
    HIP_ASSERT(hipMalloc(&d_buf, n * sizeof(float)));
    if (g_rank == 0)
      HIP_ASSERT(hipMemcpy(d_buf, root_data.data(), n * sizeof(float), hipMemcpyHostToDevice));

    NCCL_ASSERT(ncclBroadcast(d_buf, d_buf, n, ncclFloat, 0, comm_, stream_));
    HIP_ASSERT(hipStreamSynchronize(stream_));

    std::vector<float> result(n);
    HIP_ASSERT(hipMemcpy(result.data(), d_buf, n * sizeof(float), hipMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i)
      EXPECT_NEAR(result[i], root_data[i], kTolerance)
          << "iter=" << iter << " rank=" << g_rank << " i=" << i;

    (void)hipFree(d_buf);
  }
}

TEST_F(RcclTest, AllGather) {
  for (int iter = 0; iter < kFuzzIters; ++iter) {
    uint32_t size_seed = g_base_seed + 100 * iter + 20;
    int total = fuzz_size(size_seed);
    int chunk = total / g_world_size;
    uint32_t iter_seed = g_base_seed + 1000 * iter + 20;
    auto local = random_floats(iter_seed + g_rank, chunk);

    float *d_send = nullptr, *d_recv = nullptr;
    HIP_ASSERT(hipMalloc(&d_send, chunk * sizeof(float)));
    HIP_ASSERT(hipMalloc(&d_recv, total * sizeof(float)));
    HIP_ASSERT(hipMemcpy(d_send, local.data(), chunk * sizeof(float), hipMemcpyHostToDevice));

    NCCL_ASSERT(ncclAllGather(d_send, d_recv, chunk, ncclFloat, comm_, stream_));
    HIP_ASSERT(hipStreamSynchronize(stream_));

    std::vector<float> result(total);
    HIP_ASSERT(hipMemcpy(result.data(), d_recv, total * sizeof(float), hipMemcpyDeviceToHost));

    for (int r = 0; r < g_world_size; ++r) {
      auto ref_chunk = random_floats(iter_seed + r, chunk);
      for (int i = 0; i < chunk; ++i)
        EXPECT_NEAR(result[r * chunk + i], ref_chunk[i], kTolerance)
            << "iter=" << iter << " rank=" << g_rank << " src_rank=" << r << " i=" << i;
    }

    (void)hipFree(d_send);
    (void)hipFree(d_recv);
  }
}

TEST_F(RcclTest, ReduceScatter) {
  for (int iter = 0; iter < kFuzzIters; ++iter) {
    uint32_t size_seed = g_base_seed + 100 * iter + 30;
    int total = fuzz_size(size_seed);
    int chunk = total / g_world_size;
    uint32_t iter_seed = g_base_seed + 1000 * iter + 30;
    auto local = random_floats(iter_seed + g_rank, total);

    float *d_send = nullptr, *d_recv = nullptr;
    HIP_ASSERT(hipMalloc(&d_send, total * sizeof(float)));
    HIP_ASSERT(hipMalloc(&d_recv, chunk * sizeof(float)));
    HIP_ASSERT(hipMemcpy(d_send, local.data(), total * sizeof(float), hipMemcpyHostToDevice));

    NCCL_ASSERT(ncclReduceScatter(d_send, d_recv, chunk, ncclFloat, ncclSum, comm_, stream_));
    HIP_ASSERT(hipStreamSynchronize(stream_));

    std::vector<float> result(chunk);
    HIP_ASSERT(hipMemcpy(result.data(), d_recv, chunk * sizeof(float), hipMemcpyDeviceToHost));

    auto all = all_ranks_input(iter_seed, total);
    int start = g_rank * chunk;
    for (int i = 0; i < chunk; ++i) {
      float ref = 0.0f;
      for (int r = 0; r < g_world_size; ++r)
        ref += all[r * total + start + i];
      EXPECT_NEAR(result[i], ref, kTolerance)
          << "iter=" << iter << " rank=" << g_rank << " i=" << i;
    }

    (void)hipFree(d_send);
    (void)hipFree(d_recv);
  }
}

TEST_F(RcclTest, SendRecv) {
  for (int iter = 0; iter < kFuzzIters; ++iter) {
    uint32_t size_seed = g_base_seed + 100 * iter + 40;
    int n = fuzz_size(size_seed);
    uint32_t iter_seed = g_base_seed + 1000 * iter + 40;
    auto send_data = random_floats(iter_seed, n);

    float *d_buf = nullptr;
    HIP_ASSERT(hipMalloc(&d_buf, n * sizeof(float)));

    if (g_rank == 0) {
      HIP_ASSERT(hipMemcpy(d_buf, send_data.data(), n * sizeof(float), hipMemcpyHostToDevice));
      NCCL_ASSERT(ncclGroupStart());
      NCCL_ASSERT(ncclSend(d_buf, n, ncclFloat, 1, comm_, stream_));
      NCCL_ASSERT(ncclGroupEnd());
    } else {
      NCCL_ASSERT(ncclGroupStart());
      NCCL_ASSERT(ncclRecv(d_buf, n, ncclFloat, 0, comm_, stream_));
      NCCL_ASSERT(ncclGroupEnd());
    }
    HIP_ASSERT(hipStreamSynchronize(stream_));

    if (g_rank == 1) {
      std::vector<float> result(n);
      HIP_ASSERT(hipMemcpy(result.data(), d_buf, n * sizeof(float), hipMemcpyDeviceToHost));
      for (int i = 0; i < n; ++i)
        EXPECT_NEAR(result[i], send_data[i], kTolerance) << "iter=" << iter << " i=" << i;
    }

    (void)hipFree(d_buf);
  }
}

int main(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--rank=", 7) == 0)
      g_rank = std::atoi(argv[i] + 7);
    else if (std::strncmp(argv[i], "--world-size=", 13) == 0)
      g_world_size = std::atoi(argv[i] + 13);
    else if (std::strncmp(argv[i], "--shared-dir=", 13) == 0)
      g_shared_dir = argv[i] + 13;
    else if (std::strncmp(argv[i], "--seed=", 7) == 0)
      g_base_seed = static_cast<uint32_t>(std::atoi(argv[i] + 7));
  }

  if (g_shared_dir.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " --rank=N --world-size=M --shared-dir=DIR [--seed=S] [gtest args]\n";
    return 1;
  }

  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}
