/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR pinned host snapshots
 * @{
 * @ingroup HRRTest
 * A kernel reads pinned host memory that the host refills with plain stores
 * between launches, which no HIP call records. Capture snapshots it before
 * each launch and replay writes it back, so the replayed D2H checks match.
 */

#include "hrr_test_common.hh"

#include <string>
#include <vector>

namespace {
constexpr int kPinnedN     = 256;
constexpr int kPinnedIters = 4;
// The input changes on every second launch: two snapshot records, four checks.
int pinned_value(int it, int i) { return (it / 2) * 1000 + i; }
}  // namespace

__global__ void hrr_pinned_triple(int* out, const int* in, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * 3 + 1;
}

static void hrr_pinned_workload(int* in) {
  int* dev_in = nullptr;
  HRR_HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dev_in), in, 0));
  int* dout = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dout, kPinnedN * sizeof(int)));
  std::vector<int> hout(kPinnedN);
  for (int it = 0; it < kPinnedIters; ++it) {
    for (int i = 0; i < kPinnedN; ++i) in[i] = pinned_value(it, i);  // uncaptured stores
    hipLaunchKernelGGL(hrr_pinned_triple, dim3(1), dim3(kPinnedN), 0, nullptr,
                       dout, dev_in, kPinnedN);
    HRR_HIP_CHECK(hipGetLastError());
    HRR_HIP_CHECK(hipMemcpy(hout.data(), dout, kPinnedN * sizeof(int), hipMemcpyDeviceToHost));
    for (int i = 0; i < kPinnedN; ++i) REQUIRE(hout[i] == pinned_value(it, i) * 3 + 1);
  }
  HRR_HIP_CHECK(hipFree(dout));
}

TEST_CASE("Unit_HRR_PinnedHostMalloc_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* in = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&in), kPinnedN * sizeof(int), 0));
  hrr_pinned_workload(in);
  HRR_HIP_CHECK(hipHostFree(in));
}

TEST_CASE("Unit_HRR_PinnedHostRegister_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  std::vector<int> in(kPinnedN);
  HRR_HIP_CHECK(hipHostRegister(in.data(), kPinnedN * sizeof(int), hipHostRegisterDefault));
  hrr_pinned_workload(in.data());
  HRR_HIP_CHECK(hipHostUnregister(in.data()));
}

#if defined(HRR_PLAYBACK_EXE) && defined(HRR_TEST_EXE)
// Replay matches the recording only if it wrote each launch's input back; its
// own buffer never sees the host's stores. With HIP_HRR_HOST_SNAPSHOTS=0 there
// is nothing to write back and the checks fail.
HRR_TEST_CASE(Unit_HRR_PinnedHostRoundtrip) {
  for (const char* direct : {"Unit_HRR_PinnedHostMalloc_Direct",
                             "Unit_HRR_PinnedHostRegister_Direct"}) {
    for (const char* snapshots : {"1", "0"}) {
      INFO(direct << " HIP_HRR_HOST_SNAPSHOTS=" << snapshots);
      ScopedDir cap{fs::temp_directory_path() / "hrr_pinned_host"};
      hrr_capture_direct(direct, cap.path, 5, {{"HIP_HRR_HOST_SNAPSHOTS", snapshots}});
      auto [ret, out] = hrr_playback_env(cap.path, {{"HIP_HRR_D2H_EXACT", "1"}});
      INFO("Playback exit " << ret << ", stdout:\n" << out);
      int pass = 0, fail = 0;
      REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
      if (std::string(snapshots) == "1") {
        CHECK(ret == 0);
        CHECK(pass == kPinnedIters);
        CHECK(fail == 0);
        CHECK(out.find("Pinned host snapshots: 2 restored, 0 refused") != std::string::npos);
      } else {
        CHECK(ret != 0);
        CHECK(fail > 0);
      }
    }
  }
}
#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
