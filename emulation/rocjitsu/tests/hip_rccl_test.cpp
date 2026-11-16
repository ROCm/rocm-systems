// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_rccl_test.cpp
/// @brief RCCL collective communication tests on simulated multi-process GPU.
///
/// Each process is one rank in a collective. Rank 0 generates an ncclUniqueId
/// and exchanges it through a shared directory or a bounded TCP session before
/// ncclCommInitRank. Each test validates the RCCL result against a
/// single-threaded host-computed golden reference using random fuzz inputs.
///
/// Compiled with hipcc, linked with -lrccl. Requires LD_PRELOAD=librocjitsu.so.
/// Arguments: --rank=N --world-size=M (--shared-dir=DIR |
///            --bootstrap-listen=IPv4:PORT | --bootstrap-connect=IPv4:PORT)
///            [--seed=S] [--expected-devices=N] [--payload-elements=N]
///            [--collective-timeout-ms=N]
///            [--collective-timeout-marker=PATH]
/// TCP mode additionally requires --bootstrap-run-id=HEX32 and
/// --bootstrap-timeout-ms=N.

#include "rccl_tcp_bootstrap.h"

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <vector>

namespace tcp_bootstrap = rocjitsu::test::rccl_tcp_bootstrap;

static int g_rank = 0;
static int g_world_size = 2;
static std::string g_shared_dir;
static uint32_t g_base_seed = 42;
static int g_expected_devices = -1;
static int g_payload_elements = 0;
static int64_t g_collective_timeout_ms = 0;
static std::string g_collective_timeout_marker;
static std::unique_ptr<tcp_bootstrap::Session> g_tcp_session;
static uint32_t g_case_ordinal = 0;
static constexpr int kFuzzIters = 3;
static constexpr float kTolerance = 1e-4f;

static float reduction_tolerance(float absolute_input_sum) {
  // RCCL may add ranks in a different order from the serial host reference.
  // gamma_(world_size-1) bounds normal FP32 accumulation roundoff; retain the
  // original absolute floor for values whose exact sum is close to zero.
  const float steps = static_cast<float>(g_world_size - 1);
  const float error = steps * std::numeric_limits<float>::epsilon();
  return kTolerance + (error / (1.0f - error)) * absolute_input_sum;
}

// Opt-in fail-stop deadline for a rank stuck in submission or stream
// synchronization. The optional marker lets a separate guest supervisor
// power off even when rank teardown stalls after _Exit.
class CollectiveDeadline {
public:
  CollectiveDeadline(const char *name, int iter) : name_(name), iter_(iter) {
    if (g_collective_timeout_ms == 0)
      return;
    std::cout << "RCCL_COLLECTIVE_BEGIN case=" << name_ << " rank=" << g_rank << " iter=" << iter_
              << std::endl;
    worker_ = std::thread([this] {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::milliseconds(g_collective_timeout_ms);
      while (!stopped_.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) {
          if (stopped_.load(std::memory_order_acquire))
            return;
          char message[192];
          const int length =
              std::snprintf(message, sizeof(message),
                            "RCCL_COLLECTIVE_TIMEOUT case=%s rank=%d iter=%d deadline_ms=%lld\n",
                            name_, g_rank, iter_, static_cast<long long>(g_collective_timeout_ms));
          if (length > 0) {
            const size_t size = std::min(static_cast<size_t>(length), sizeof(message) - 1);
            (void)::write(STDERR_FILENO, message, size);
            if (!g_collective_timeout_marker.empty()) {
              const int fd = ::open(g_collective_timeout_marker.c_str(),
                                    O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
              if (fd >= 0) {
                (void)::write(fd, message, size);
                (void)::close(fd);
              }
            }
          }
          std::_Exit(124);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    });
  }

  CollectiveDeadline(const CollectiveDeadline &) = delete;
  CollectiveDeadline &operator=(const CollectiveDeadline &) = delete;

  ~CollectiveDeadline() {
    stopped_.store(true, std::memory_order_release);
    if (worker_.joinable())
      worker_.join();
  }

private:
  const char *name_;
  int iter_;
  std::atomic<bool> stopped_{false};
  std::thread worker_;
};

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
  if (g_payload_elements != 0)
    return g_payload_elements;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(64, 1024);
  int n = dist(rng);
  return (n / g_world_size) * g_world_size;
}

static void log_payload(int iter, int elements) {
  const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::cout << "RCCL payload case=" << test_info->name() << " rank=" << g_rank << " iter=" << iter
            << " elements=" << elements << '\n';
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
  { std::ofstream(ready) << "1"; }
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
    std::ranges::copy(v, all.begin() + r * n);
  }
  return all;
}

class RcclTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (g_expected_devices >= 0) {
      int device_count = 0;
      HIP_ASSERT(hipGetDeviceCount(&device_count));
      ASSERT_EQ(device_count, g_expected_devices)
          << "rank=" << g_rank << " expected node-local GPU discovery";
    }

    const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    ASSERT_NE(test_info, nullptr);
    // A process may run several cases in one shared directory. Give each
    // communicator its own files so the next case cannot reuse stale markers.
    const std::string rendezvous = test_info->name();
    ncclUniqueId id{};
    if (g_tcp_session) {
      if (g_rank == 0)
        NCCL_ASSERT(ncclGetUniqueId(&id));
      const std::string full_name = std::string(test_info->test_suite_name()) + "." + rendezvous;
      try {
        g_tcp_session->ExchangeCase(full_name, g_case_ordinal++, &id);
      } catch (const std::exception &error) {
        FAIL() << "TCP bootstrap failed for " << full_name << ": " << error.what();
      }
    } else {
      const std::string id_path = g_shared_dir + "/nccl_id." + rendezvous;
      if (g_rank == 0) {
        NCCL_ASSERT(ncclGetUniqueId(&id));
        write_file(id_path, &id, sizeof(id));
      }
      barrier("nccl_init." + rendezvous);
      if (g_rank != 0)
        read_file_blocking(id_path, &id, sizeof(id));
    }
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
    log_payload(iter, n);
    uint32_t iter_seed = g_base_seed + 1000 * iter;
    auto local = random_floats(iter_seed + g_rank, n);

    float *d_send = nullptr, *d_recv = nullptr;
    HIP_ASSERT(hipMalloc(&d_send, n * sizeof(float)));
    HIP_ASSERT(hipMalloc(&d_recv, n * sizeof(float)));
    GTEST_LOG_(INFO) << "probe: rank=" << g_rank << " stage=alloc iter=" << iter
                     << " send_va=" << static_cast<void *>(d_send)
                     << " recv_va=" << static_cast<void *>(d_recv);
    HIP_ASSERT(hipMemcpy(d_send, local.data(), n * sizeof(float), hipMemcpyHostToDevice));

    if (iter != 0) {
      std::vector<float> input_readback(n);
      HIP_ASSERT(
          hipMemcpy(input_readback.data(), d_send, n * sizeof(float), hipMemcpyDeviceToHost));
      int input_mismatches = 0;
      for (int i = 0; i < n; ++i)
        input_mismatches += input_readback[i] != local[i];
      GTEST_LOG_(INFO) << "probe: rank=" << g_rank << " stage=input iter=" << iter << " n=" << n
                       << " mismatches=" << input_mismatches;
      ASSERT_EQ(input_mismatches, 0);

      float *d_copy = nullptr;
      HIP_ASSERT(hipMalloc(&d_copy, n * sizeof(float)));
      copy_kernel<<<(n + 255) / 256, 256, 0, stream_>>>(d_copy, d_send, n);
      HIP_ASSERT(hipGetLastError());
      HIP_ASSERT(hipStreamSynchronize(stream_));
      std::vector<float> copied(n);
      HIP_ASSERT(hipMemcpy(copied.data(), d_copy, n * sizeof(float), hipMemcpyDeviceToHost));
      int gpu_input_mismatches = 0;
      for (int i = 0; i < n; ++i)
        gpu_input_mismatches += copied[i] != local[i];
      GTEST_LOG_(INFO) << "probe: rank=" << g_rank << " stage=gpu-input iter=" << iter << " n=" << n
                       << " mismatches=" << gpu_input_mismatches;
      ASSERT_EQ(gpu_input_mismatches, 0);
      (void)hipFree(d_copy);
    }

    if (iter == 0) {
      float *d_copy = nullptr;
      HIP_ASSERT(hipMalloc(&d_copy, n * sizeof(float)));
      std::vector<float> h2d(n), copied(n), before(n);
      HIP_ASSERT(hipMemcpy(h2d.data(), d_send, n * sizeof(float), hipMemcpyDeviceToHost));
      copy_kernel<<<(n + 255) / 256, 256, 0, stream_>>>(d_copy, d_send, n);
      HIP_ASSERT(hipGetLastError());
      HIP_ASSERT(hipStreamSynchronize(stream_));
      HIP_ASSERT(hipMemcpy(copied.data(), d_copy, n * sizeof(float), hipMemcpyDeviceToHost));

      const float sentinel = 37.25f + static_cast<float>(g_rank);
      const std::vector<float> sentinels(n, sentinel);
      HIP_ASSERT(hipMemcpy(d_recv, sentinels.data(), n * sizeof(float), hipMemcpyHostToDevice));
      HIP_ASSERT(hipMemcpy(before.data(), d_recv, n * sizeof(float), hipMemcpyDeviceToHost));
      int h2d_mismatches = 0;
      int copy_mismatches = 0;
      int sentinel_mismatches = 0;
      for (int i = 0; i < n; ++i) {
        h2d_mismatches += h2d[i] != local[i];
        copy_mismatches += copied[i] != local[i];
        sentinel_mismatches += before[i] != sentinel;
      }
      GTEST_LOG_(INFO) << "probe: rank=" << g_rank << " stage=pre n=" << n
                       << " h2d_mismatches=" << h2d_mismatches
                       << " copy_mismatches=" << copy_mismatches
                       << " sentinel_mismatches=" << sentinel_mismatches;
      ASSERT_EQ(h2d_mismatches, 0);
      ASSERT_EQ(copy_mismatches, 0);
      ASSERT_EQ(sentinel_mismatches, 0);
      (void)hipFree(d_copy);
    }

    {
      CollectiveDeadline deadline("AllReduce", iter);
      NCCL_ASSERT(ncclAllReduce(d_send, d_recv, n, ncclFloat, ncclSum, comm_, stream_));
      HIP_ASSERT(hipStreamSynchronize(stream_));
    }

    std::vector<float> result(n);
    HIP_ASSERT(hipMemcpy(result.data(), d_recv, n * sizeof(float), hipMemcpyDeviceToHost));

    auto all = all_ranks_input(iter_seed, n);
    std::vector<float> ref(n, 0.0f);
    for (int r = 0; r < g_world_size; ++r)
      for (int i = 0; i < n; ++i)
        ref[i] += all[r * n + i];

    int mismatches = 0;
    int first_mismatch = -1;
    float first_result = 0.0f;
    float first_reference = 0.0f;
    float first_tolerance = 0.0f;
    for (int i = 0; i < n; ++i) {
      float absolute_input_sum = 0.0f;
      for (int r = 0; r < g_world_size; ++r)
        absolute_input_sum += std::fabs(all[r * n + i]);
      const float tolerance = reduction_tolerance(absolute_input_sum);
      if (!std::isfinite(result[i]) || std::fabs(result[i] - ref[i]) > tolerance) {
        ++mismatches;
        if (first_mismatch < 0) {
          first_mismatch = i;
          first_result = result[i];
          first_reference = ref[i];
          first_tolerance = tolerance;
        }
      }
    }
    GTEST_LOG_(INFO) << "probe: rank=" << g_rank << " stage=post iter=" << iter << " n=" << n
                     << " mismatches=" << mismatches;
    EXPECT_EQ(mismatches, 0) << "iter=" << iter << " rank=" << g_rank
                             << " first_i=" << first_mismatch << " result=" << first_result
                             << " reference=" << first_reference
                             << " tolerance=" << first_tolerance;

    (void)hipFree(d_send);
    (void)hipFree(d_recv);
  }
}

TEST_F(RcclTest, Broadcast) {
  for (int iter = 0; iter < kFuzzIters; ++iter) {
    uint32_t size_seed = g_base_seed + 100 * iter + 10;
    int n = fuzz_size(size_seed);
    log_payload(iter, n);
    uint32_t iter_seed = g_base_seed + 1000 * iter + 10;
    auto root_data = random_floats(iter_seed, n);

    float *d_buf = nullptr;
    HIP_ASSERT(hipMalloc(&d_buf, n * sizeof(float)));
    GTEST_LOG_(INFO) << "probe: rank=" << g_rank << " stage=broadcast-alloc iter=" << iter
                     << " n=" << n << " buf_va=" << static_cast<void *>(d_buf);
    if (g_rank == 0)
      HIP_ASSERT(hipMemcpy(d_buf, root_data.data(), n * sizeof(float), hipMemcpyHostToDevice));

    {
      CollectiveDeadline deadline("Broadcast", iter);
      NCCL_ASSERT(ncclBroadcast(d_buf, d_buf, n, ncclFloat, 0, comm_, stream_));
      HIP_ASSERT(hipStreamSynchronize(stream_));
    }

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
    log_payload(iter, total);
    int chunk = total / g_world_size;
    uint32_t iter_seed = g_base_seed + 1000 * iter + 20;
    auto local = random_floats(iter_seed + g_rank, chunk);

    float *d_send = nullptr, *d_recv = nullptr;
    HIP_ASSERT(hipMalloc(&d_send, chunk * sizeof(float)));
    HIP_ASSERT(hipMalloc(&d_recv, total * sizeof(float)));
    HIP_ASSERT(hipMemcpy(d_send, local.data(), chunk * sizeof(float), hipMemcpyHostToDevice));

    {
      CollectiveDeadline deadline("AllGather", iter);
      NCCL_ASSERT(ncclAllGather(d_send, d_recv, chunk, ncclFloat, comm_, stream_));
      HIP_ASSERT(hipStreamSynchronize(stream_));
    }

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
    log_payload(iter, total);
    int chunk = total / g_world_size;
    uint32_t iter_seed = g_base_seed + 1000 * iter + 30;
    auto local = random_floats(iter_seed + g_rank, total);

    float *d_send = nullptr, *d_recv = nullptr;
    HIP_ASSERT(hipMalloc(&d_send, total * sizeof(float)));
    HIP_ASSERT(hipMalloc(&d_recv, chunk * sizeof(float)));
    HIP_ASSERT(hipMemcpy(d_send, local.data(), total * sizeof(float), hipMemcpyHostToDevice));

    {
      CollectiveDeadline deadline("ReduceScatter", iter);
      NCCL_ASSERT(ncclReduceScatter(d_send, d_recv, chunk, ncclFloat, ncclSum, comm_, stream_));
      HIP_ASSERT(hipStreamSynchronize(stream_));
    }

    std::vector<float> result(chunk);
    HIP_ASSERT(hipMemcpy(result.data(), d_recv, chunk * sizeof(float), hipMemcpyDeviceToHost));

    auto all = all_ranks_input(iter_seed, total);
    int start = g_rank * chunk;
    for (int i = 0; i < chunk; ++i) {
      float ref = 0.0f;
      float absolute_input_sum = 0.0f;
      for (int r = 0; r < g_world_size; ++r) {
        ref += all[r * total + start + i];
        absolute_input_sum += std::fabs(all[r * total + start + i]);
      }
      EXPECT_NEAR(result[i], ref, reduction_tolerance(absolute_input_sum))
          << "iter=" << iter << " rank=" << g_rank << " i=" << i;
    }

    (void)hipFree(d_send);
    (void)hipFree(d_recv);
  }
}

TEST_F(RcclTest, SendRecv) {
  ASSERT_GE(g_world_size, 2);
  for (int iter = 0; iter < kFuzzIters; ++iter) {
    uint32_t size_seed = g_base_seed + 100 * iter + 40;
    int n = fuzz_size(size_seed);
    log_payload(iter, n);
    uint32_t iter_seed = g_base_seed + 1000 * iter + 40;
    const int source = (g_rank + g_world_size - 1) % g_world_size;
    const int destination = (g_rank + 1) % g_world_size;
    auto send_data = random_floats(iter_seed + g_rank, n);
    auto expected = random_floats(iter_seed + source, n);

    float *d_send = nullptr, *d_recv = nullptr;
    HIP_ASSERT(hipMalloc(&d_send, n * sizeof(float)));
    HIP_ASSERT(hipMalloc(&d_recv, n * sizeof(float)));
    HIP_ASSERT(hipMemcpy(d_send, send_data.data(), n * sizeof(float), hipMemcpyHostToDevice));

    // Every rank posts one matching send and receive in the same group.
    {
      CollectiveDeadline deadline("SendRecv", iter);
      NCCL_ASSERT(ncclGroupStart());
      NCCL_ASSERT(ncclSend(d_send, n, ncclFloat, destination, comm_, stream_));
      NCCL_ASSERT(ncclRecv(d_recv, n, ncclFloat, source, comm_, stream_));
      NCCL_ASSERT(ncclGroupEnd());
      HIP_ASSERT(hipStreamSynchronize(stream_));
    }

    std::vector<float> result(n);
    HIP_ASSERT(hipMemcpy(result.data(), d_recv, n * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
      EXPECT_NEAR(result[i], expected[i], kTolerance)
          << "iter=" << iter << " rank=" << g_rank << " i=" << i;

    (void)hipFree(d_send);
    (void)hipFree(d_recv);
  }
}

int main(int argc, char **argv) {
  std::optional<tcp_bootstrap::Role> role;
  std::string endpoint;
  std::string run_id;
  int64_t timeout_ms = 0;
  bool saw_rank = false;
  bool saw_world_size = false;
  bool saw_shared_dir = false;
  bool saw_run_id = false;
  bool saw_timeout = false;
  bool saw_payload_elements = false;
  bool saw_collective_timeout = false;
  bool saw_collective_marker = false;

  const auto parse_uint = [](std::string_view value, uint64_t maximum,
                             std::string_view option) -> uint64_t {
    uint64_t result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (value.empty() || error != std::errc{} || end != value.data() + value.size() ||
        result > maximum)
      throw std::invalid_argument(std::string(option) + " requires an unsigned integer in range");
    return result;
  };

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string_view arg(argv[i]);
      if (arg.starts_with("--rank=")) {
        if (saw_rank)
          throw std::invalid_argument("duplicate --rank");
        g_rank = static_cast<int>(parse_uint(arg.substr(7), INT_MAX, "--rank"));
        saw_rank = true;
      } else if (arg.starts_with("--world-size=")) {
        if (saw_world_size)
          throw std::invalid_argument("duplicate --world-size");
        g_world_size = static_cast<int>(parse_uint(arg.substr(13), INT_MAX, "--world-size"));
        saw_world_size = true;
      } else if (arg.starts_with("--shared-dir=")) {
        if (saw_shared_dir)
          throw std::invalid_argument("duplicate --shared-dir");
        g_shared_dir = arg.substr(13);
        saw_shared_dir = true;
      } else if (arg.starts_with("--seed=")) {
        g_base_seed = static_cast<uint32_t>(
            parse_uint(arg.substr(7), std::numeric_limits<uint32_t>::max(), "--seed"));
      } else if (arg.starts_with("--expected-devices=")) {
        g_expected_devices =
            static_cast<int>(parse_uint(arg.substr(19), INT_MAX, "--expected-devices"));
      } else if (arg.starts_with("--payload-elements=")) {
        if (saw_payload_elements)
          throw std::invalid_argument("duplicate --payload-elements");
        g_payload_elements =
            static_cast<int>(parse_uint(arg.substr(19), 1 << 20, "--payload-elements"));
        if (g_payload_elements == 0)
          throw std::invalid_argument("--payload-elements must be positive");
        saw_payload_elements = true;
      } else if (arg.starts_with("--collective-timeout-ms=")) {
        if (saw_collective_timeout)
          throw std::invalid_argument("duplicate --collective-timeout-ms");
        constexpr std::string_view option = "--collective-timeout-ms=";
        g_collective_timeout_ms =
            static_cast<int64_t>(parse_uint(arg.substr(option.size()), 600000, option));
        if (g_collective_timeout_ms == 0)
          throw std::invalid_argument("--collective-timeout-ms must be positive");
        saw_collective_timeout = true;
      } else if (arg.starts_with("--collective-timeout-marker=")) {
        if (saw_collective_marker)
          throw std::invalid_argument("duplicate --collective-timeout-marker");
        constexpr std::string_view option = "--collective-timeout-marker=";
        g_collective_timeout_marker = arg.substr(option.size());
        if (g_collective_timeout_marker.empty())
          throw std::invalid_argument("--collective-timeout-marker needs a path");
        saw_collective_marker = true;
      } else if (arg.starts_with("--bootstrap-listen=")) {
        if (role)
          throw std::invalid_argument("choose exactly one TCP bootstrap role");
        role = tcp_bootstrap::Role::Listen;
        endpoint = arg.substr(19);
      } else if (arg.starts_with("--bootstrap-connect=")) {
        if (role)
          throw std::invalid_argument("choose exactly one TCP bootstrap role");
        role = tcp_bootstrap::Role::Connect;
        endpoint = arg.substr(20);
      } else if (arg.starts_with("--bootstrap-run-id=")) {
        if (saw_run_id)
          throw std::invalid_argument("duplicate --bootstrap-run-id");
        run_id = arg.substr(19);
        saw_run_id = true;
      } else if (arg.starts_with("--bootstrap-timeout-ms=")) {
        if (saw_timeout)
          throw std::invalid_argument("duplicate --bootstrap-timeout-ms");
        timeout_ms =
            static_cast<int64_t>(parse_uint(arg.substr(23), 600000, "--bootstrap-timeout-ms"));
        saw_timeout = true;
      } else if (arg.starts_with("--bootstrap-")) {
        throw std::invalid_argument("unknown TCP bootstrap option");
      }
    }

    if (g_world_size == 0)
      throw std::invalid_argument("--world-size must be positive");
    if (saw_payload_elements && g_payload_elements % g_world_size != 0)
      throw std::invalid_argument("--payload-elements must be divisible by --world-size");
    if (saw_collective_marker && !saw_collective_timeout)
      throw std::invalid_argument("--collective-timeout-marker requires a deadline");

    if (role || saw_run_id || saw_timeout) {
      if (saw_shared_dir || !role || !saw_run_id || !saw_timeout || !saw_rank || !saw_world_size)
        throw std::invalid_argument("TCP bootstrap requires rank, world size, role, run ID, "
                                    "and timeout, without --shared-dir");
      tcp_bootstrap::Config config{*role,
                                   endpoint,
                                   run_id,
                                   g_rank,
                                   g_world_size,
                                   g_base_seed,
                                   sizeof(ncclUniqueId),
                                   std::chrono::milliseconds(timeout_ms)};
      g_tcp_session =
          std::make_unique<tcp_bootstrap::Session>(tcp_bootstrap::Session::Open(std::move(config)));
    } else if (g_shared_dir.empty()) {
      throw std::invalid_argument("--shared-dir is required in file bootstrap mode");
    }
  } catch (const std::exception &error) {
    std::cerr << "RCCL bootstrap: " << error.what() << '\n'
              << "Usage: " << argv[0]
              << " --rank=N --world-size=M [--seed=S] [--expected-devices=N]"
                 " [--payload-elements=N] [--collective-timeout-ms=N]"
                 " [--collective-timeout-marker=PATH]"
                 " (--shared-dir=DIR | --bootstrap-listen=IPv4:PORT |"
                 " --bootstrap-connect=IPv4:PORT) [--bootstrap-run-id=HEX32"
                 " --bootstrap-timeout-ms=N] [gtest args]\n";
    return 1;
  }

  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}
