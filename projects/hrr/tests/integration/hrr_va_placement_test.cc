/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR capture-address placement
 * @{
 * @ingroup HRRTest
 * A device pointer the program stored in device memory with an H2D copy is
 * right at replay only if its allocation lands at its recorded address
 * (vLLM's block table, ROCM-31827). The kernel also gets the buffer as an
 * argument, which replay translates, and reads through the stored pointer only
 * when the two agree, so a stale pointer fails a D2H check instead of faulting.
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

#include <string>
#include <vector>

#if defined(HRR_PLAYBACK_EXE) && defined(HRR_TEST_EXE)

namespace {
constexpr int    kElems = 1024;
constexpr size_t kBytes = kElems * sizeof(int);
}  // namespace

__global__ void hrr_place_deref(int* out, int* const* cell, const int* expect, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int* p = *cell;
  out[i] = (p == expect) ? p[i] * 2 : -1;
}

namespace {
// Allocate a buffer (hipMallocAsync on `s` when given), store its address in a
// cell with hipMemcpy, read the buffer through the cell, free both.
void hrr_place_round(int* out, int seed, hipStream_t s) {
  int* buf = nullptr;
  if (s) {
    HRR_HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&buf), kBytes, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));
  } else {
    HRR_HIP_CHECK(hipMalloc(&buf, kBytes));
  }
  std::vector<int> host(kElems);
  for (int i = 0; i < kElems; ++i) host[i] = seed + i;
  HRR_HIP_CHECK(hipMemcpy(buf, host.data(), kBytes, hipMemcpyHostToDevice));
  int** cell = nullptr;
  HRR_HIP_CHECK(hipMalloc(&cell, sizeof(int*)));
  HRR_HIP_CHECK(hipMemcpy(cell, &buf, sizeof(int*), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(hrr_place_deref, dim3(kElems / 256), dim3(256), 0, nullptr, out, cell,
                     buf, kElems);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());
  HRR_HIP_CHECK(hipMemcpy(host.data(), out, kBytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < kElems; ++i) REQUIRE(host[i] == (seed + i) * 2);

  HRR_HIP_CHECK(hipFree(cell));
  if (s) {
    HRR_HIP_CHECK(hipFreeAsync(buf, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));
  } else {
    HRR_HIP_CHECK(hipFree(buf));
  }
}
}  // namespace

TEST_CASE("Unit_HRR_VaPlacement_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipFree(nullptr));  // ROCM-30200: not the allocation first
  HRR_HIP_CHECK(hipSetDevice(0));
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, kBytes));
  hrr_place_round(out, 1, nullptr);
  hipStream_t s = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&s));
  hrr_place_round(out, 1000, s);
  HRR_HIP_CHECK(hipStreamDestroy(s));
  // `out` stays live: the --kernel-filter warm-up must release it.
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_StoredPointer) {
#ifdef _WIN32
  HRR_SKIP("placement is Linux only");
#endif
  int vmm = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(
      &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
  if (!vmm) HRR_SKIP("placement needs virtual memory management");

  ScopedDir cap(fs::temp_directory_path() / "hrr_va_placement.hrr");
  hrr_capture_direct("Unit_HRR_VaPlacement_Direct", cap.path);
  const fs::path archive = hrr_single_process_archive(cap.path);

  {
    auto [rc, out] = hrr_playback_merged(archive);
    INFO("Replay:\n" << out);
    int pass = 0, fail = 0;
    REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
    CHECK(pass == 2);
    CHECK(fail == 0);
    CHECK(rc == 0);
    CHECK(out.find("placed at capture address, 0 fell back") != std::string::npos);
  }
  {
    auto [rc, out] = hrr_playback_merged(archive, "--kernel-filter hrr_place_deref");
    INFO("Replay --kernel-filter:\n" << out);
    CHECK(out.find("placed at capture address, 0 fell back") != std::string::npos);
  }
  {
    // Without placement both rounds read through a stale pointer.
    auto [rc, out] = hrr_playback_merged(archive, "--no-placement");
    INFO("Replay --no-placement:\n" << out);
    int pass = 0, fail = 0;
    REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
    CHECK(pass == 0);
    CHECK(fail == 2);
    CHECK(rc != 0);
    CHECK(out.find("Placement") == std::string::npos);
  }
}

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
