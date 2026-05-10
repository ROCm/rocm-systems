/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
/*
This testfile verifies the following scenarios of all hipMemcpy API
1. Multi thread
2. Multi size
*/

static auto Available_Gpus{0};
static constexpr auto MAX_GPU{256};

enum apiToTest {
  TEST_MEMCPY,
  TEST_MEMCPYH2D,
  TEST_MEMCPYD2H,
  TEST_MEMCPYD2D,
  TEST_MEMCPYASYNC,
  TEST_MEMCPYH2DASYNC,
  TEST_MEMCPYD2HASYNC,
  TEST_MEMCPYD2DASYNC
};

namespace {

/** Uniform random int in [0, n); n must be > 0. */
inline int RandomDeviceIndex(int n, std::mt19937& gen) {
  return static_cast<int>(std::uniform_int_distribution<int>(0, n - 1)(gen));
}

/** Pick j != intra with hipDeviceCanAccessPeer(intra, j); random trials, or -1. */
inline int RandomPeerDst(int intra_dev, int n_gpu, std::mt19937& gen) {
  if (n_gpu <= 1) {
    return -1;
  }
  constexpr int kMaxTries = 64;
  for (int t = 0; t < kMaxTries; ++t) {
    const int j = RandomDeviceIndex(n_gpu, gen);
    if (j == intra_dev) {
      continue;
    }
    int p = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&p, intra_dev, j));
    if (p) {
      return j;
    }
  }
  return -1;
}

const char* ApiToString(apiToTest api) {
  switch (api) {
    case TEST_MEMCPY:
      return "hipMemcpy";
    case TEST_MEMCPYH2D:
      return "hipMemcpyHtoD";
    case TEST_MEMCPYD2H:
      return "hipMemcpyDtoH";
    case TEST_MEMCPYD2D:
      return "hipMemcpyDtoD";
    case TEST_MEMCPYASYNC:
      return "hipMemcpyAsync";
    case TEST_MEMCPYH2DASYNC:
      return "hipMemcpyHtoDAsync";
    case TEST_MEMCPYD2HASYNC:
      return "hipMemcpyDtoHAsync";
    case TEST_MEMCPYD2DASYNC:
      return "hipMemcpyDtoDAsync";
  }
  return "unknown";
}

}  // namespace

template <typename TestType> void Memcpy_And_verify(size_t NUM_ELM, std::mt19937& gen) {
  // level_0 (quick level): skip H2D/D2H/H2DAsync/D2HAsync entirely,
  // and cap async APIs (MemcpyAsync/DtoDAsync) at 64 KiB.
  // All other levels run every API at every size.
  // Each API uses one randomly chosen GPU (and a second for P2P when applicable); device memory is
  // allocated only for GPUs used in that iteration.
  constexpr size_t kQuickAsyncMaxBytes = 64u * 1024u;
  const size_t requested_bytes = NUM_ELM * sizeof(TestType);

  TestType *A_h, *B_h;
  for (apiToTest api = TEST_MEMCPY; api <= TEST_MEMCPYD2DASYNC; api = apiToTest(api + 1)) {
    if (isQuickLevel() && (api == TEST_MEMCPYH2D || api == TEST_MEMCPYD2H || api == TEST_MEMCPYH2DASYNC ||
                           api == TEST_MEMCPYD2HASYNC)) {
      continue;
    }
    const bool async_api = (api == TEST_MEMCPYASYNC || api == TEST_MEMCPYH2DASYNC ||
                            api == TEST_MEMCPYD2HASYNC || api == TEST_MEMCPYD2DASYNC);
    if (isQuickLevel() && async_api && requested_bytes > kQuickAsyncMaxBytes) {
      continue;
    }

    std::cout << "[Stress_hipMemcpy_multiDevice_AllAPIs] api=" << ApiToString(api)
              << " NUM_ELM=" << NUM_ELM << " bytes=" << requested_bytes << std::endl;

    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr, &A_h, &B_h, nullptr, NUM_ELM);
    HIP_CHECK(hipGetDeviceCount(&Available_Gpus));
    if (Available_Gpus <= 0) {
      HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr, A_h, B_h, nullptr, false);
      continue;
    }

    TestType* A_d[MAX_GPU]{};
    hipStream_t stream[MAX_GPU]{};
    bool stream_ok[MAX_GPU]{};

    switch (api) {
      case TEST_MEMCPY: {
        // One random GPU; P2P is covered in TEST_MEMCPYD2D / ASYNC.
        const int d = RandomDeviceIndex(Available_Gpus, gen);
        HIP_CHECK(hipSetDevice(d));
        HIP_CHECK(hipMalloc(&A_d[d], NUM_ELM * sizeof(TestType)));
        HIP_CHECK(hipMemcpy(A_d[d], A_h, NUM_ELM * sizeof(TestType), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(B_h, A_d[d], NUM_ELM * sizeof(TestType), hipMemcpyDeviceToHost));
        HipTest::checkTest(A_h, B_h, NUM_ELM);
        break;
      }
      case TEST_MEMCPYH2D:  // To test hipMemcpyHtoD()
      {
        const int d = RandomDeviceIndex(Available_Gpus, gen);
        HIP_CHECK(hipSetDevice(d));
        HIP_CHECK(hipMalloc(&A_d[d], NUM_ELM * sizeof(TestType)));
        HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d[d]), A_h, NUM_ELM * sizeof(TestType)));
        HIP_CHECK(hipMemcpy(B_h, A_d[d], NUM_ELM * sizeof(TestType), hipMemcpyDeviceToHost));
        HipTest::checkTest(A_h, B_h, NUM_ELM);
        break;
      }
      case TEST_MEMCPYD2H:  // To test hipMemcpyDtoH()--done
      {
        const int d = RandomDeviceIndex(Available_Gpus, gen);
        HIP_CHECK(hipSetDevice(d));
        HIP_CHECK(hipMalloc(&A_d[d], NUM_ELM * sizeof(TestType)));
        HIP_CHECK(hipMemcpy(A_d[d], A_h, NUM_ELM * sizeof(TestType), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpyDtoH(B_h, hipDeviceptr_t(A_d[d]), NUM_ELM * sizeof(TestType)));
        HipTest::checkTest(A_h, B_h, NUM_ELM);
        break;
      }
      case TEST_MEMCPYD2D:  // To test hipMemcpyDtoD()
      {
        // Random intra GPU; random peer dst (no full-GPU scan); P2P intra→peer (reuse A_d[intra], no H2D).
        const int intra_dev = RandomDeviceIndex(Available_Gpus, gen);
        const int peer_dst = RandomPeerDst(intra_dev, Available_Gpus, gen);

        HIP_CHECK(hipSetDevice(intra_dev));
        HIP_CHECK(hipMalloc(&A_d[intra_dev], NUM_ELM * sizeof(TestType)));
        if (peer_dst >= 0) {
          HIP_CHECK(hipSetDevice(peer_dst));
          HIP_CHECK(hipMalloc(&A_d[peer_dst], NUM_ELM * sizeof(TestType)));
        }

        TestType* scratch_d[MAX_GPU]{};
        HIP_CHECK(hipSetDevice(intra_dev));
        HIP_CHECK(hipMalloc(&scratch_d[intra_dev], NUM_ELM * sizeof(TestType)));

        {
          const int i = intra_dev;
          std::cout << "[Stress_hipMemcpy_multiDevice_AllAPIs] hipMemcpyDtoD intra dev=" << i
                    << " bytes=" << (NUM_ELM * sizeof(TestType)) << std::endl;
          HIP_CHECK(hipSetDevice(i));
          HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d[i]), A_h, NUM_ELM * sizeof(TestType)));
          HIP_CHECK(hipMemcpyDtoD(hipDeviceptr_t(scratch_d[i]), hipDeviceptr_t(A_d[i]),
                                  NUM_ELM * sizeof(TestType)));
          HIP_CHECK(hipMemcpyDtoD(hipDeviceptr_t(A_d[i]), hipDeviceptr_t(scratch_d[i]),
                                  NUM_ELM * sizeof(TestType)));
          HIP_CHECK(hipMemcpy(B_h, A_d[i], NUM_ELM * sizeof(TestType), hipMemcpyDeviceToHost));
          HipTest::checkTest(A_h, B_h, NUM_ELM);
        }

        if (peer_dst >= 0) {
          std::cout << "[Stress_hipMemcpy_multiDevice_AllAPIs] hipMemcpyDtoD peer " << intra_dev << "->"
                    << peer_dst << " bytes=" << (NUM_ELM * sizeof(TestType)) << std::endl;
          HIP_CHECK(hipSetDevice(peer_dst));
          HIP_CHECK(hipMemcpyDtoD(hipDeviceptr_t(A_d[peer_dst]), hipDeviceptr_t(A_d[intra_dev]),
                                  NUM_ELM * sizeof(TestType)));
          HIP_CHECK(hipMemcpy(B_h, A_d[peer_dst], NUM_ELM * sizeof(TestType), hipMemcpyDeviceToHost));
          HipTest::checkTest(A_h, B_h, NUM_ELM);
        }

        HIP_CHECK(hipSetDevice(intra_dev));
        HIP_CHECK(hipFree(scratch_d[intra_dev]));
        break;
      }
      case TEST_MEMCPYASYNC: {
        // One random GPU; P2P is covered in TEST_MEMCPYD2DASYNC.
        const int d = RandomDeviceIndex(Available_Gpus, gen);
        HIP_CHECK(hipSetDevice(d));
        HIP_CHECK(hipMalloc(&A_d[d], NUM_ELM * sizeof(TestType)));
        HIP_CHECK(hipStreamCreate(&stream[d]));
        stream_ok[d] = true;
        HIP_CHECK(hipMemcpyAsync(A_d[d], A_h, NUM_ELM * sizeof(TestType), hipMemcpyHostToDevice,
                                 stream[d]));
        HIP_CHECK(hipMemcpyAsync(B_h, A_d[d], NUM_ELM * sizeof(TestType), hipMemcpyDeviceToHost,
                                 stream[d]));
        HIP_CHECK(hipStreamSynchronize(stream[d]));
        HipTest::checkTest(A_h, B_h, NUM_ELM);
        break;
      }
      case TEST_MEMCPYH2DASYNC:  // To test hipMemcpyHtoDAsync()
      {
        const int d = RandomDeviceIndex(Available_Gpus, gen);
        HIP_CHECK(hipSetDevice(d));
        HIP_CHECK(hipMalloc(&A_d[d], NUM_ELM * sizeof(TestType)));
        HIP_CHECK(hipStreamCreate(&stream[d]));
        stream_ok[d] = true;
        HIP_CHECK(hipMemcpyHtoDAsync(hipDeviceptr_t(A_d[d]), A_h, NUM_ELM * sizeof(TestType),
                                     stream[d]));
        HIP_CHECK(hipStreamSynchronize(stream[d]));
        HIP_CHECK(hipMemcpy(B_h, A_d[d], NUM_ELM * sizeof(TestType), hipMemcpyDeviceToHost));
        HipTest::checkTest(A_h, B_h, NUM_ELM);
        break;
      }
      case TEST_MEMCPYD2HASYNC:  // To test hipMemcpyDtoHAsync()
      {
        const int d = RandomDeviceIndex(Available_Gpus, gen);
        HIP_CHECK(hipSetDevice(d));
        HIP_CHECK(hipMalloc(&A_d[d], NUM_ELM * sizeof(TestType)));
        HIP_CHECK(hipStreamCreate(&stream[d]));
        stream_ok[d] = true;
        HIP_CHECK(hipMemcpy(A_d[d], A_h, NUM_ELM * sizeof(TestType), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpyDtoHAsync(B_h, hipDeviceptr_t(A_d[d]), NUM_ELM * sizeof(TestType),
                                     stream[d]));
        HIP_CHECK(hipStreamSynchronize(stream[d]));
        HipTest::checkTest(A_h, B_h, NUM_ELM);
        break;
      }
      case TEST_MEMCPYD2DASYNC:  // To test hipMemcpyDtoDAsync()
      {
        // Same as sync DtoD: random intra / random peer; allocate only those GPUs.
        const int intra_dev = RandomDeviceIndex(Available_Gpus, gen);
        const int peer_dst = RandomPeerDst(intra_dev, Available_Gpus, gen);

        HIP_CHECK(hipSetDevice(intra_dev));
        HIP_CHECK(hipMalloc(&A_d[intra_dev], NUM_ELM * sizeof(TestType)));
        HIP_CHECK(hipStreamCreate(&stream[intra_dev]));
        stream_ok[intra_dev] = true;
        if (peer_dst >= 0) {
          HIP_CHECK(hipSetDevice(peer_dst));
          HIP_CHECK(hipMalloc(&A_d[peer_dst], NUM_ELM * sizeof(TestType)));
        }

        TestType* scratch_d[MAX_GPU]{};
        HIP_CHECK(hipSetDevice(intra_dev));
        HIP_CHECK(hipMalloc(&scratch_d[intra_dev], NUM_ELM * sizeof(TestType)));

        {
          const int i = intra_dev;
          std::cout << "[Stress_hipMemcpy_multiDevice_AllAPIs] hipMemcpyDtoDAsync intra dev=" << i
                    << " bytes=" << (NUM_ELM * sizeof(TestType)) << std::endl;
          HIP_CHECK(hipSetDevice(i));
          HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d[i]), A_h, NUM_ELM * sizeof(TestType)));
          HIP_CHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(scratch_d[i]), hipDeviceptr_t(A_d[i]),
                                       NUM_ELM * sizeof(TestType), stream[i]));
          HIP_CHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(A_d[i]), hipDeviceptr_t(scratch_d[i]),
                                       NUM_ELM * sizeof(TestType), stream[i]));
          HIP_CHECK(hipStreamSynchronize(stream[i]));
          HIP_CHECK(hipMemcpy(B_h, A_d[i], NUM_ELM * sizeof(TestType), hipMemcpyDeviceToHost));
          HipTest::checkTest(A_h, B_h, NUM_ELM);
        }

        if (peer_dst >= 0) {
          std::cout << "[Stress_hipMemcpy_multiDevice_AllAPIs] hipMemcpyDtoDAsync peer " << intra_dev
                    << "->" << peer_dst << " bytes=" << (NUM_ELM * sizeof(TestType)) << std::endl;
          HIP_CHECK(hipSetDevice(peer_dst));
          HIP_CHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(A_d[peer_dst]), hipDeviceptr_t(A_d[intra_dev]),
                                       NUM_ELM * sizeof(TestType), stream[intra_dev]));
          HIP_CHECK(hipStreamSynchronize(stream[intra_dev]));
          HIP_CHECK(hipMemcpy(B_h, A_d[peer_dst], NUM_ELM * sizeof(TestType), hipMemcpyDeviceToHost));
          HipTest::checkTest(A_h, B_h, NUM_ELM);
        }

        HIP_CHECK(hipSetDevice(intra_dev));
        HIP_CHECK(hipFree(scratch_d[intra_dev]));
        break;
      }
    }
    for (int i = 0; i < Available_Gpus; ++i) {
      if (A_d[i]) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipFree(A_d[i]));
      }
      if (stream_ok[i]) {
        HIP_CHECK(hipStreamDestroy(stream[i]));
      }
    }
    HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr, A_h, B_h, nullptr, false);
  }
}

HIP_TEST_CASE(Stress_hipMemcpy_multiDevice_AllAPIs) {
  constexpr size_t kQuickMaxBytes = 32u * 1024u * 1024u;
  // Memcpy_And_verify (quick level): MemcpyAsync / DtoDAsync run only for transfer <= 64 KiB.
  static constexpr int kNumElmStepsQuick[] = {
      1,  3,  7,  15, 31,  33,  63,  65,  255, 257,  1023, 1025,  4095, 4097,
      8191, 8193,  32767, 32769,  65535, 65537,
      10 * 1024,
      100 * 1024,
      3 * 1024 * 1024,
      7 * 1024 * 1024,
      10 * 1024 * 1024,
  };
  static constexpr int kNumElmSteps[] = {
      1,    3,     5,     7,     10,     15,    17,    31,    33,    48,
      63,   65,    96,    97,    100,    101,   127,   129,   192,   255,
      257,  511,   513,   768,   1009,   1023,  1025,  1536,  2047,  2049,
      3072, 4095,  4097,  6144,  8191,   8193,  12288, 16383, 16385, 24576,
      32767, 32769, 65535, 65537, 
      10 * 1024,
      100 * 1024,
      3 * 1024 * 1024,
      7 * 1024 * 1024,
      10 * 1024 * 1024,
      48 * 1024 * 1024,
      100 * 1024 * 1024,
  };
  // Inclusive exponent range for 2^p elements. char: end=32 → 2^32 B transfer. Wider TestType: end=30 to cap bytes.
  constexpr int kPow2StartPower = 0;
  constexpr int kPow2EndPower = 32;

  size_t free = 0, total = 0;
  HIP_CHECK(hipMemGetInfo(&free, &total));

  const int* const extra_elm_steps = isQuickLevel() ? kNumElmStepsQuick : kNumElmSteps;
  const size_t extra_elm_count =
      isQuickLevel() ? sizeof(kNumElmStepsQuick) / sizeof(kNumElmStepsQuick[0])
                     : sizeof(kNumElmSteps) / sizeof(kNumElmSteps[0]);

  std::vector<size_t> all_sizes;
  all_sizes.reserve((kPow2EndPower - kPow2StartPower + 1) + extra_elm_count);
  for (int p = kPow2StartPower; p <= kPow2EndPower; ++p) {
    all_sizes.push_back(1ull << p);
  }
  for (size_t e = 0; e < extra_elm_count; ++e) {
    all_sizes.push_back(static_cast<size_t>(extra_elm_steps[e]));
  }
  std::sort(all_sizes.begin(), all_sizes.end());
  all_sizes.erase(std::unique(all_sizes.begin(), all_sizes.end()), all_sizes.end());

  auto use_size = [&](size_t NUM_ELM) -> bool {
    NUM_ELM = std::max(NUM_ELM, size_t{1});
    const size_t requested_bytes = NUM_ELM * sizeof(char);
    // level_0 (quick level): cap all scheduled sizes at 32 MiB (Memcpy_And_verify uses a subset of APIs).
    if (isQuickLevel() && requested_bytes > kQuickMaxBytes) {
      return false;
    }
    return true;
  };

  int total_steps = 0;
  for (size_t NUM_ELM : all_sizes) {
    if (use_size(NUM_ELM)) {
      ++total_steps;
    }
  }

  std::random_device rd;
  unsigned int seed = rd();
  const char* seed_env = std::getenv("HIP_TEST_SEED");
  if (seed_env && seed_env[0] != '\0') {
    seed = static_cast<unsigned int>(std::atol(seed_env));
  }
  std::mt19937 gen(seed);
  std::cout << "[Stress_hipMemcpy_multiDevice_AllAPIs] seed=" << seed << std::endl;

  int step = 0;
  for (size_t NUM_ELM : all_sizes) {
    NUM_ELM = std::max(NUM_ELM, size_t{1});
    if (!use_size(NUM_ELM)) {
      continue;
    }
    ++step;
    const size_t requested_bytes = NUM_ELM * sizeof(char);
    std::cout << "[Stress_hipMemcpy_multiDevice_AllAPIs] " << step << "/" << total_steps
              << " NUM_ELM=" << NUM_ELM << " bytes=" << requested_bytes << std::endl;
    if (requested_bytes <= free) {
      Memcpy_And_verify<char>(NUM_ELM, gen);
      HIP_CHECK(hipDeviceSynchronize());
    } else {
      std::cout << "[Stress_hipMemcpy_multiDevice_AllAPIs] skip (need " << requested_bytes
                << " B, free " << free << " B)" << std::endl;
    }
  }
}
