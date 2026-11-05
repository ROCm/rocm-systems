/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR host-backed allocation budget
 * @{
 * @ingroup HRRTest
 * hipMallocManaged and a hipMemCreate with a host location can be backed by
 * host memory, so replay charges them to --max-host-bytes like hipHostMalloc,
 * refuses a record that would pass the cap, and gives the charge back on hipFree
 * and hipMemRelease. Each workload allocates 2 MiB three times in a row,
 * freeing the buffer in between, so a replay whose cap holds one buffer (3 MiB)
 * only gets through if the free returns the charge, and a cap below one buffer
 * (1 MiB) refuses the first.
 */

#include "hrr_test_common.hh"

#include <string>

namespace {
constexpr size_t kBudgetBuf = 2u << 20;
constexpr int kBudgetRounds = 3;

// Capture is not capped; only replay is.
void hrr_budget_replay(const char* direct_case, const char* dir, const char* cap_bytes,
                       const char* api) {
  ScopedDir cap{hrr_temp_dir(dir)};
  hrr_capture_direct(direct_case, cap.path, 5);
  for (const bool over : {false, true}) {
    const std::string cap_arg =
        std::string("--max-host-bytes ") + (over ? "1048576" : cap_bytes);
    auto [ret, out] = hrr_playback_merged(cap.path, cap_arg);
    INFO(cap_arg << ": playback exit " << ret << ", output:\n" << out);
    if (over) {
      CHECK(ret != 0);
      CHECK(out.find(std::string("[HRR] Fatal: ") + api + " needs") != std::string::npos);
      CHECK(out.find("--max-host-bytes") != std::string::npos);
    } else {
      CHECK(ret == 0);
      CHECK(out.find("[HRR] Fatal") == std::string::npos);
    }
  }
}
}  // namespace

TEST_CASE("Unit_HRR_HostBudgetManaged_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  for (int i = 0; i < kBudgetRounds; ++i) {
    void* p = nullptr;
    HRR_HIP_CHECK(hipMallocManaged(&p, kBudgetBuf));
    HRR_HIP_CHECK(hipFree(p));
  }
}

TEST_CASE("Unit_HRR_HostBudgetMemCreate_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  hipMemAllocationProp prop{};
  prop.type          = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeHost;
  prop.location.id   = 0;
  size_t gran = 0;
  if (hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum) !=
          hipSuccess || gran == 0)
    HRR_SKIP("no host-located VMM granularity on this device");
  const size_t size = (kBudgetBuf + gran - 1) / gran * gran;
  if (size > 3u << 20) HRR_SKIP("host-located granularity " << gran << " is above the 3 MiB cap");
  // Support is decided once, here; a failure in a later round is a real one.
  {
    hipMemGenericAllocationHandle_t probe{};
    if (hipMemCreate(&probe, size, &prop, 0) != hipSuccess)
      HRR_SKIP("host-located hipMemCreate is not supported on this device");
    HRR_HIP_CHECK(hipMemRelease(probe));
  }
  for (int i = 0; i < kBudgetRounds; ++i) {
    hipMemGenericAllocationHandle_t h{};
    HRR_HIP_CHECK(hipMemCreate(&h, size, &prop, 0));
    HRR_HIP_CHECK(hipMemRelease(h));
  }
}

// Within a one-buffer cap the free returns the charge; under the buffer, the
// first allocation is refused.
HRR_TEST_CASE(Unit_HRR_HostBudgetManagedRoundtrip) {
  hrr_budget_replay("Unit_HRR_HostBudgetManaged_Direct", "hrr_budget_managed", "3145728",
                    "hipMallocManaged");
}

HRR_TEST_CASE(Unit_HRR_HostBudgetMemCreateRoundtrip) {
  hrr_budget_replay("Unit_HRR_HostBudgetMemCreate_Direct", "hrr_budget_memcreate", "3145728",
                    "hipMemCreate (host location)");
}

// A handle that hipMemMap still holds keeps its charge after hipMemRelease, until
// hipMemUnmap. The workload maps h1, releases it, then creates h2 before the
// unmap: with the hold the 3 MiB replay is refused at h2 (h1's 2 MiB is still
// charged); without it the release would have returned the charge and h2 fits.
// A 5 MiB cap takes h1 + h2 plus the replay's cached blobs, so the replay
// completes.
TEST_CASE("Unit_HRR_HostBudgetMemMapHold_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  hipMemAllocationProp prop{};
  prop.type          = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeHost;
  prop.location.id   = 0;
  size_t gran = 0;
  if (hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum) !=
          hipSuccess || gran == 0)
    HRR_SKIP("no host-located VMM granularity on this device");
  const size_t size = (kBudgetBuf + gran - 1) / gran * gran;
  if (size != kBudgetBuf) HRR_SKIP("host-located granularity " << gran << " does not fit 2 MiB");

  void* va = nullptr;
  HRR_HIP_CHECK(hipMemAddressReserve(&va, size, 0, nullptr, 0));
  hipMemGenericAllocationHandle_t h1{}, h2{};
  // Support is decided here; a failure after it is a real one.
  if (hipMemCreate(&h1, size, &prop, 0) != hipSuccess) {
    (void)hipMemAddressFree(va, size);
    HRR_SKIP("host-located hipMemCreate is not supported on this device");
  }
  if (hipMemMap(va, size, 0, h1, 0) != hipSuccess) {
    (void)hipMemRelease(h1);
    (void)hipMemAddressFree(va, size);
    HRR_SKIP("a host-located handle cannot be mapped on this device");
  }
  HRR_HIP_CHECK(hipMemRelease(h1));            // mapping keeps the memory
  HRR_HIP_CHECK(hipMemCreate(&h2, size, &prop, 0));
  HRR_HIP_CHECK(hipMemRelease(h2));
  HRR_HIP_CHECK(hipMemUnmap(va, size));        // the last reference to h1
  HRR_HIP_CHECK(hipMemAddressFree(va, size));
}

HRR_TEST_CASE(Unit_HRR_HostBudgetMemMapHoldRoundtrip) {
  ScopedDir cap{hrr_temp_dir("hrr_budget_memmap_hold")};
  hrr_capture_direct("Unit_HRR_HostBudgetMemMapHold_Direct", cap.path, 5);
  for (const bool refuse : {false, true}) {
    const std::string cap_arg =
        std::string("--max-host-bytes ") + (refuse ? "3145728" : "5242880");
    auto [ret, out] = hrr_playback_merged(cap.path, cap_arg);
    INFO(cap_arg << ": playback exit " << ret << ", output:\n" << out);
    if (refuse) {
      CHECK(ret != 0);
      CHECK(out.find("[HRR] Fatal: hipMemCreate (host location) needs") != std::string::npos);
    } else {
      CHECK(ret == 0);
      CHECK(out.find("[HRR] Fatal") == std::string::npos);
    }
  }
}

// A managed buffer the workload never frees stays live through the --kernel-filter
// warm-up pass. The timed pass replays the same allocation and replaces it, which
// must return the old charge: the cap holds one buffer (3 MiB), so a charge that
// is kept makes the second pass refuse.
TEST_CASE("Unit_HRR_HostBudgetManagedLeak_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  void* p = nullptr;
  HRR_HIP_CHECK(hipMallocManaged(&p, kBudgetBuf));  // deliberately never freed
}

HRR_TEST_CASE(Unit_HRR_HostBudgetManagedLeakRoundtrip) {
  ScopedDir cap{hrr_temp_dir("hrr_budget_managed_leak")};
  hrr_capture_direct("Unit_HRR_HostBudgetManagedLeak_Direct", cap.path, 5);
  auto [ret, out] = hrr_playback_merged(
      cap.path, "--kernel-filter hrr_no_such_kernel --max-host-bytes 3145728");
  INFO("playback exit " << ret << ", output:\n" << out);
  CHECK(ret == 0);
  CHECK(out.find("[HRR] Fatal") == std::string::npos);
}

/**
 * @}
 */
