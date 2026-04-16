/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_params.hh>
#include <algorithm>
#include <cstdint>
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

// level_2, transfer > 256 MiB, multi-GPU: DtoD uses one random device for intra-DtoD only; chosen on first
// qualifying DtoD call and reused for later sizes (sync + async share this index).
int g_level2_d2d_focus_device = -1;

}  // namespace

template <typename TestType> void Memcpy_And_verify(size_t NUM_ELM) {
  // MEMCPYH2D / DtoH / H2DAsync / DtoHAsync: only run when transfer <= 64 KiB (level_2+). Not run at all for
  // level_0 smoke. Primary four (MEMCPY, DtoD, ASYNC, DtoDAsync) have no per-API cap beyond smoke 32 MiB.
  // level_0 smoke: MemcpyAsync / DtoDAsync only when transfer <= 1 MiB and ~10% of NUM_ELM (deterministic).
  // level_2 + multi-GPU + DtoD: transfer > 256 MiB uses one random GPU for intra-DtoD only (see cases).
  constexpr size_t kNonPrimaryApiMaxBytes = 64u * 1024u;
  constexpr size_t kSmokeAsyncMaxBytes = 1024u * 1024u;
  const size_t requested_bytes = NUM_ELM * sizeof(TestType);
  const bool smoke_level =
      (TestParameterStore::instance().currentTestLevel == "level_0");

  TestType *A_h, *B_h;
  for (apiToTest api = TEST_MEMCPY; api <= TEST_MEMCPYD2DASYNC; api = apiToTest(api + 1)) {
    if (smoke_level && (api == TEST_MEMCPYH2D || api == TEST_MEMCPYD2H || api == TEST_MEMCPYH2DASYNC ||
                        api == TEST_MEMCPYD2HASYNC)) {
      continue;
    }
    const bool primary_api = (api == TEST_MEMCPY || api == TEST_MEMCPYD2D || api == TEST_MEMCPYASYNC ||
                              api == TEST_MEMCPYD2DASYNC);
    if (!primary_api && requested_bytes > kNonPrimaryApiMaxBytes) {
      continue;
    }
    const bool async_api = (api == TEST_MEMCPYASYNC || api == TEST_MEMCPYH2DASYNC ||
                            api == TEST_MEMCPYD2HASYNC || api == TEST_MEMCPYD2DASYNC);
    if (smoke_level && async_api) {
      if (requested_bytes > kSmokeAsyncMaxBytes) {
        continue;
      }
      // ~10% of sizes: mix NUM_ELM into one bucket (stable across runs for a given build).
      const uint64_t mix =
          static_cast<uint64_t>(NUM_ELM) * UINT64_C(0x9E3779B97F4A7C15) ^ static_cast<uint64_t>(NUM_ELM >> 32);
      if ((mix % 10) != 0) {
        continue;
      }
    }

    std::cout << "[Stress_hipMemcpy_multiDevice_AllAPIs] api=" << ApiToString(api)
              << " NUM_ELM=" << NUM_ELM << " bytes=" << requested_bytes << std::endl;

    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr, &A_h, &B_h, nullptr, NUM_ELM);
    HIP_CHECK(hipGetDeviceCount(&Available_Gpus));
    TestType* A_d[MAX_GPU];
    hipStream_t stream[MAX_GPU];
    for (int i = 0; i < Available_Gpus; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipMalloc(&A_d[i], NUM_ELM * sizeof(TestType)));
      if (api >= TEST_MEMCPYD2D) {
        HIP_CHECK(hipStreamCreate(&stream[i]));
      }
    }
    HIP_CHECK(hipSetDevice(0));
    int canAccessPeer = 0;
    switch (api) {
      case TEST_MEMCPY: {
        // To test hipMemcpy()
        // Copying data from host to individual devices followed by copying
        // back to host and verifying the data consistency.
        for (int i = 0; i < Available_Gpus; ++i) {
          HIP_CHECK(hipMemcpy(A_d[i], A_h, NUM_ELM * sizeof(TestType), hipMemcpyHostToDevice));
          HIP_CHECK(hipMemcpy(B_h, A_d[i], NUM_ELM * sizeof(TestType), hipMemcpyDeviceToHost));
          HipTest::checkTest(A_h, B_h, NUM_ELM);
        }
        // Device to Device copying for all combinations
        for (int i = 0; i < Available_Gpus; ++i) {
          for (int j = i + 1; j < Available_Gpus; ++j) {
            canAccessPeer = 0;
            HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, i, j));
            if (canAccessPeer) {
              HIP_CHECK(hipMemcpy(A_d[j], A_d[i], NUM_ELM * sizeof(TestType), hipMemcpyDefault));
              // Copying in reverse dir of above to check if bidirectional
              // access is happening without any error
              HIP_CHECK(hipMemcpy(A_d[i], A_d[j], NUM_ELM * sizeof(TestType), hipMemcpyDefault));
              // Copying data to host to verify the content
              HIP_CHECK(hipMemcpy(B_h, A_d[j], NUM_ELM * sizeof(TestType), hipMemcpyDefault));
              HipTest::checkTest(A_h, B_h, NUM_ELM);
            }
          }
        }
        break;
      }
      case TEST_MEMCPYH2D:  // To test hipMemcpyHtoD()
      {
        for (int i = 0; i < Available_Gpus; ++i) {
          HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d[i]), A_h, NUM_ELM * sizeof(TestType)));
          // Copying data from device to host to check data consistency
          HIP_CHECK(hipMemcpy(B_h, A_d[i], NUM_ELM * sizeof(TestType), hipMemcpyDeviceToHost));
          HipTest::checkTest(A_h, B_h, NUM_ELM);
        }
        break;
      }
      case TEST_MEMCPYD2H:  // To test hipMemcpyDtoH()--done
      {
        for (int i = 0; i < Available_Gpus; ++i) {
          HIP_CHECK(hipMemcpy(A_d[i], A_h, NUM_ELM * sizeof(TestType), hipMemcpyHostToDevice));
          HIP_CHECK(hipMemcpyDtoH(B_h, hipDeviceptr_t(A_d[i]), NUM_ELM * sizeof(TestType)));
          HipTest::checkTest(A_h, B_h, NUM_ELM);
        }
        break;
      }
      case TEST_MEMCPYD2D:  // To test hipMemcpyDtoD()
      {
        // Peer i!=j: one picked (i→j) pair always runs at full NUM_ELM. Other peer pairs run only when
        // NUM_ELM <= cap (64 KiB level_0, 64 MiB otherwise); if larger, those transfers are skipped.
        // Intra-device uses full NUM_ELM for all devices.
        // level_2 + transfer > 256 MiB + multi-GPU: only intra-DtoD on one random GPU (no peer matrix).
        constexpr size_t kLevel2D2dMultiGpuBytes = 256ull * 1024u * 1024u;
        const bool level_2 = (TestParameterStore::instance().currentTestLevel == "level_2");
        const bool use_single_dev_d2d =
            level_2 && (requested_bytes > kLevel2D2dMultiGpuBytes) && (Available_Gpus > 1);
        if (use_single_dev_d2d && g_level2_d2d_focus_device < 0) {
          std::random_device rd;
          g_level2_d2d_focus_device =
              static_cast<int>(rd() % static_cast<unsigned>(Available_Gpus));
        }
        const int focus_dev = use_single_dev_d2d ? g_level2_d2d_focus_device : -1;

        const bool smoke_level =
            (TestParameterStore::instance().currentTestLevel == "level_0");
        constexpr size_t kPeerOtherMaxBytesSmoke = 64u * 1024u;
        constexpr size_t kPeerOtherMaxBytesFull = 64u * 1024u * 1024u;
        const size_t peer_other_cap_bytes =
            smoke_level ? kPeerOtherMaxBytesSmoke : kPeerOtherMaxBytesFull;
        const size_t peer_other_cap_elm =
            std::max(size_t{1}, peer_other_cap_bytes / sizeof(TestType));

        std::vector<std::pair<int, int>> peer_pairs;
        int chosen_peer_i = -1;
        int chosen_peer_j = -1;
        if (!use_single_dev_d2d) {
          peer_pairs.reserve(static_cast<size_t>(Available_Gpus * Available_Gpus));
          for (int ii = 0; ii < Available_Gpus; ++ii) {
            for (int jj = 0; jj < Available_Gpus; ++jj) {
              if (ii == jj) {
                continue;
              }
              int p = 0;
              HIP_CHECK(hipDeviceCanAccessPeer(&p, ii, jj));
              if (p) {
                peer_pairs.emplace_back(ii, jj);
              }
            }
          }
          if (!peer_pairs.empty()) {
            const size_t pick =
                (NUM_ELM + static_cast<size_t>(Available_Gpus)) % peer_pairs.size();
            chosen_peer_i = peer_pairs[pick].first;
            chosen_peer_j = peer_pairs[pick].second;
          }
        }

        TestType* scratch_d[MAX_GPU]{};
        if (use_single_dev_d2d) {
          HIP_CHECK(hipSetDevice(focus_dev));
          HIP_CHECK(hipMalloc(&scratch_d[focus_dev], NUM_ELM * sizeof(TestType)));
        } else {
          for (int d = 0; d < Available_Gpus; ++d) {
            HIP_CHECK(hipSetDevice(d));
            HIP_CHECK(hipMalloc(&scratch_d[d], NUM_ELM * sizeof(TestType)));
          }
        }
        // j runs 0..N-1: diagonal (i==j) is intra-GPU (A_d[i] <-> scratch_d[i]); off-diagonal is one DtoD i→j.
        int peer_ij = 0;
        for (int i = 0; i < Available_Gpus; ++i) {
          for (int j = 0; j < Available_Gpus; ++j) {
            if (use_single_dev_d2d && (i != focus_dev || j != focus_dev)) {
              continue;
            }
            if (i == j) {
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
            } else {
              HIP_CHECK(hipDeviceCanAccessPeer(&peer_ij, i, j));
              if (peer_ij) {
                const bool full_peer_pair = (i == chosen_peer_i && j == chosen_peer_j);
                if (full_peer_pair || NUM_ELM <= peer_other_cap_elm) {
                  std::cout << "[Stress_hipMemcpy_multiDevice_AllAPIs] hipMemcpyDtoD peer " << i << "->"
                            << j << " bytes=" << (NUM_ELM * sizeof(TestType)) << std::endl;
                  HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d[i]), A_h, NUM_ELM * sizeof(TestType)));
                  HIP_CHECK(hipMemcpyDtoD(hipDeviceptr_t(A_d[j]), hipDeviceptr_t(A_d[i]),
                                          NUM_ELM * sizeof(TestType)));
                  HIP_CHECK(
                      hipMemcpy(B_h, A_d[j], NUM_ELM * sizeof(TestType), hipMemcpyDeviceToHost));
                  HipTest::checkTest(A_h, B_h, NUM_ELM);
                }
              }
            }
          }
        }
        if (use_single_dev_d2d) {
          HIP_CHECK(hipSetDevice(focus_dev));
          HIP_CHECK(hipFree(scratch_d[focus_dev]));
        } else {
          for (int d = 0; d < Available_Gpus; ++d) {
            HIP_CHECK(hipSetDevice(d));
            HIP_CHECK(hipFree(scratch_d[d]));
          }
        }
        break;
      }
      case TEST_MEMCPYASYNC: {
        // To test hipMemcpyAsync()
        // Copying data from host to individual devices followed by copying
        // back to host and verifying the data consistency.
        for (int i = 0; i < Available_Gpus; ++i) {
          HIP_CHECK(hipMemcpyAsync(A_d[i], A_h, NUM_ELM * sizeof(TestType), hipMemcpyHostToDevice,
                                   stream[i]));
          HIP_CHECK(hipMemcpyAsync(B_h, A_d[i], NUM_ELM * sizeof(TestType), hipMemcpyDeviceToHost,
                                   stream[i]));
          HIP_CHECK(hipStreamSynchronize(stream[i]));
          HipTest::checkTest(A_h, B_h, NUM_ELM);
        }
        // Device to Device copying for all combinations
        for (int i = 0; i < Available_Gpus; ++i) {
          for (int j = i + 1; j < Available_Gpus; ++j) {
            canAccessPeer = 0;
            HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, i, j));
            if (canAccessPeer) {
              HIP_CHECK(hipMemcpyAsync(A_d[j], A_d[i], NUM_ELM * sizeof(TestType), hipMemcpyDefault,
                                       stream[i]));
              // Copying in direction reverse of above to
              // check if bidirectional
              // access is happening without any error
              HIP_CHECK(hipMemcpyAsync(A_d[i], A_d[j], NUM_ELM * sizeof(TestType), hipMemcpyDefault,
                                       stream[i]));
              HIP_CHECK(hipStreamSynchronize(stream[i]));
              HIP_CHECK(hipMemcpy(B_h, A_d[j], NUM_ELM * sizeof(TestType), hipMemcpyDefault));
              HipTest::checkTest(A_h, B_h, NUM_ELM);
            }
          }
        }
        break;
      }
      case TEST_MEMCPYH2DASYNC:  // To test hipMemcpyHtoDAsync()
      {
        for (int i = 0; i < Available_Gpus; ++i) {
          HIP_CHECK(hipMemcpyHtoDAsync(hipDeviceptr_t(A_d[i]), A_h, NUM_ELM * sizeof(TestType),
                                       stream[i]));
          HIP_CHECK(hipStreamSynchronize(stream[i]));
          // Copying data from device to host to check data consistency
          HIP_CHECK(hipMemcpy(B_h, A_d[i], NUM_ELM * sizeof(TestType), hipMemcpyDeviceToHost));
          HipTest::checkTest(A_h, B_h, NUM_ELM);
        }
        break;
      }
      case TEST_MEMCPYD2HASYNC:  // To test hipMemcpyDtoHAsync()
      {
        for (int i = 0; i < Available_Gpus; ++i) {
          HIP_CHECK(hipMemcpy(A_d[i], A_h, NUM_ELM * sizeof(TestType), hipMemcpyHostToDevice));
          HIP_CHECK(hipMemcpyDtoHAsync(B_h, hipDeviceptr_t(A_d[i]), NUM_ELM * sizeof(TestType),
                                       stream[i]));
          HIP_CHECK(hipStreamSynchronize(stream[i]));
          HipTest::checkTest(A_h, B_h, NUM_ELM);
        }
        break;
      }
      case TEST_MEMCPYD2DASYNC:  // To test hipMemcpyDtoDAsync()
      {
        // Same peer policy as TEST_MEMCPYD2D: full size only on picked pair; other peers skipped if
        // NUM_ELM exceeds cap. Same level_2 >256 MiB multi-GPU single-device focus as sync DtoD.
        constexpr size_t kLevel2D2dMultiGpuBytes = 256ull * 1024u * 1024u;
        const bool level_2 = (TestParameterStore::instance().currentTestLevel == "level_2");
        const bool use_single_dev_d2d =
            level_2 && (requested_bytes > kLevel2D2dMultiGpuBytes) && (Available_Gpus > 1);
        if (use_single_dev_d2d && g_level2_d2d_focus_device < 0) {
          std::random_device rd;
          g_level2_d2d_focus_device =
              static_cast<int>(rd() % static_cast<unsigned>(Available_Gpus));
        }
        const int focus_dev = use_single_dev_d2d ? g_level2_d2d_focus_device : -1;

        const bool smoke_level =
            (TestParameterStore::instance().currentTestLevel == "level_0");
        constexpr size_t kPeerOtherMaxBytesSmoke = 64u * 1024u;
        constexpr size_t kPeerOtherMaxBytesFull = 64u * 1024u * 1024u;
        const size_t peer_other_cap_bytes =
            smoke_level ? kPeerOtherMaxBytesSmoke : kPeerOtherMaxBytesFull;
        const size_t peer_other_cap_elm =
            std::max(size_t{1}, peer_other_cap_bytes / sizeof(TestType));

        std::vector<std::pair<int, int>> peer_pairs;
        int chosen_peer_i = -1;
        int chosen_peer_j = -1;
        if (!use_single_dev_d2d) {
          peer_pairs.reserve(static_cast<size_t>(Available_Gpus * Available_Gpus));
          for (int ii = 0; ii < Available_Gpus; ++ii) {
            for (int jj = 0; jj < Available_Gpus; ++jj) {
              if (ii == jj) {
                continue;
              }
              int p = 0;
              HIP_CHECK(hipDeviceCanAccessPeer(&p, ii, jj));
              if (p) {
                peer_pairs.emplace_back(ii, jj);
              }
            }
          }
          if (!peer_pairs.empty()) {
            // Different mix from DtoD so sync vs async "full" pair can differ for the same NUM_ELM.
            const size_t pick =
                (NUM_ELM * 2 + static_cast<size_t>(Available_Gpus)) % peer_pairs.size();
            chosen_peer_i = peer_pairs[pick].first;
            chosen_peer_j = peer_pairs[pick].second;
          }
        }

        TestType* scratch_d[MAX_GPU]{};
        if (use_single_dev_d2d) {
          HIP_CHECK(hipSetDevice(focus_dev));
          HIP_CHECK(hipMalloc(&scratch_d[focus_dev], NUM_ELM * sizeof(TestType)));
        } else {
          for (int d = 0; d < Available_Gpus; ++d) {
            HIP_CHECK(hipSetDevice(d));
            HIP_CHECK(hipMalloc(&scratch_d[d], NUM_ELM * sizeof(TestType)));
          }
        }
        for (int i = 0; i < Available_Gpus; ++i) {
          for (int j = 0; j < Available_Gpus; ++j) {
            if (use_single_dev_d2d && (i != focus_dev || j != focus_dev)) {
              continue;
            }
            if (i == j) {
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
            } else {
              canAccessPeer = 0;
              HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, i, j));
              if (canAccessPeer) {
                const bool full_peer_pair = (i == chosen_peer_i && j == chosen_peer_j);
                if (full_peer_pair || NUM_ELM <= peer_other_cap_elm) {
                  std::cout << "[Stress_hipMemcpy_multiDevice_AllAPIs] hipMemcpyDtoDAsync peer " << i
                            << "->" << j << " bytes=" << (NUM_ELM * sizeof(TestType)) << std::endl;
                  HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d[i]), A_h, NUM_ELM * sizeof(TestType)));
                  HIP_CHECK(hipSetDevice(j));
                  HIP_CHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(A_d[j]), hipDeviceptr_t(A_d[i]),
                                               NUM_ELM * sizeof(TestType), stream[i]));
                  HIP_CHECK(hipStreamSynchronize(stream[i]));
                  HIP_CHECK(
                      hipMemcpy(B_h, A_d[j], NUM_ELM * sizeof(TestType), hipMemcpyDeviceToHost));
                  HipTest::checkTest(A_h, B_h, NUM_ELM);
                }
              }
            }
          }
        }
        if (use_single_dev_d2d) {
          HIP_CHECK(hipSetDevice(focus_dev));
          HIP_CHECK(hipFree(scratch_d[focus_dev]));
        } else {
          for (int d = 0; d < Available_Gpus; ++d) {
            HIP_CHECK(hipSetDevice(d));
            HIP_CHECK(hipFree(scratch_d[d]));
          }
        }
        break;
      }
    }
    for (int i = 0; i < Available_Gpus; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipFree((A_d[i])));
      if (api >= TEST_MEMCPYD2D) {
        HIP_CHECK(hipStreamDestroy(stream[i]));
      }
    }
    HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr, A_h, B_h, nullptr, false);
  }
}

HIP_TEMPLATE_TEST_CASE(Stress_hipMemcpy_multiDevice_AllAPIs, char) {
  constexpr size_t kSmokeMaxBytes = 32u * 1024u * 1024u;
  // kNumElmSteps / kNumElmStepsSmoke: extra element counts (edges, non-pow2, etc.). Pure powers of two 2^p
  // are merged separately, sorted, deduplicated. Smoke uses the shorter kNumElmStepsSmoke; level_2+ uses
  // kNumElmSteps.
  // level_0: transfer size <= kSmokeMaxBytes (32 MiB). hipMemcpyHtoD / hipMemcpyDtoH / *HtoDAsync / *DtoHAsync
  // are not run. MemcpyAsync / DtoDAsync when transfer <= 1 MiB and ~10% of NUM_ELM.
  // level_2+: same 64 KiB rule for those four APIs; primary four use full size sweep. For DtoD only,
  // multi-GPU runs use all devices up to 256 MiB; above that, intra-DtoD runs on one random GPU only.
  static constexpr int kNumElmStepsSmoke[] = {
      1,  3,  7,  15, 31,  33,  63,  65,  255, 257,  1023, 1025,  4095, 4097,
      8191, 8193,  32767, 32769,  65535, 65537,
      10 * 1024,
      100 * 1024,
      3 * 1024 * 1024,
      7 * 1024 * 1024,
      10 * 1024 * 1024,
  };
  static constexpr int kNumElmSteps[] = {
      1,
      3,
      5,
      7,
      10,
      15,
      17,
      31,
      33,
      48,
      63,
      65,
      96,
      97,
      100,
      101,
      127,
      129,
      192,
      255,
      257,
      511,
      513,
      768,
      1009,
      1023,
      1025,
      1536,
      2047,
      2049,
      3072,
      4095,
      4097,
      6144,
      8191,
      8193,
      12288,
      16383,
      16385,
      24576,
      32767,
      32769,
      65535,
      65537,
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
  constexpr int kPow2EndPower = std::is_same<TestType, char>::value ? 32 : 30;

  const bool smoke = (TestParameterStore::instance().currentTestLevel == "level_0");

  size_t free = 0, total = 0;
  HIP_CHECK(hipMemGetInfo(&free, &total));
  (void)total;

  const int* const extra_elm_steps = smoke ? kNumElmStepsSmoke : kNumElmSteps;
  const size_t extra_elm_count =
      smoke ? sizeof(kNumElmStepsSmoke) / sizeof(kNumElmStepsSmoke[0])
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
    const size_t requested_bytes = NUM_ELM * sizeof(TestType);
    // level_0 smoke: cap all scheduled sizes at 32 MiB (Memcpy_And_verify uses a subset of APIs).
    if (smoke && requested_bytes > kSmokeMaxBytes) {
      return false;
    }
    return true;
  };

  int total_steps = 0;
  for (size_t NUM_ELM : all_sizes) {
    NUM_ELM = std::max(NUM_ELM, size_t{1});
    if (use_size(NUM_ELM)) {
      ++total_steps;
    }
  }

  int step = 0;
  for (size_t NUM_ELM : all_sizes) {
    NUM_ELM = std::max(NUM_ELM, size_t{1});
    if (!use_size(NUM_ELM)) {
      continue;
    }
    ++step;
    const size_t requested_bytes = NUM_ELM * sizeof(TestType);
    std::cout << "[Stress_hipMemcpy_multiDevice_AllAPIs] " << step << "/" << total_steps
              << " NUM_ELM=" << NUM_ELM << " bytes=" << requested_bytes << std::endl;
    if (requested_bytes <= free) {
      Memcpy_And_verify<TestType>(NUM_ELM);
      HIP_CHECK(hipDeviceSynchronize());
    } else {
      std::cout << "[Stress_hipMemcpy_multiDevice_AllAPIs] skip (need " << requested_bytes
                << " B, free " << free << " B)" << std::endl;
    }
  }
}
