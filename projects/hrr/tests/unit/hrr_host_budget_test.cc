/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Host Budget
 * @{
 * @ingroup HRRTest
 * Handler-level tests for the host-memory budget (--max-host-bytes): the
 * budgeted blob and code-object cache (PlaybackContext::cache_file, behind
 * load_blob and load_code_object) and the host-allocation handlers' refusal
 * paths. None reaches a HIP call, so no GPU is needed.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr_handler_test_util.h"
#include "hrr_reader.h"
#include "hrr/hrr_api_args.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

// Defined in hip_playback.cpp; not declared in the public header.
hipError_t playback_hipHostMalloc(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipMallocHost(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipHostRegister(PlaybackContext& ctx, const uint8_t* pl);

namespace fs = std::filesystem;

namespace {

// A throwaway archive directory with blobs/ and code_objects/.
struct ScratchArchive {
  fs::path root;
  explicit ScratchArchive(const std::string& name) {
    root = fs::temp_directory_path() / ("hrr_budget_" + name);
    fs::remove_all(root);
    fs::create_directories(root / "blobs");
    fs::create_directories(root / "code_objects");
  }
  ~ScratchArchive() { fs::remove_all(root); }

  void add_blob(uint64_t lo, uint64_t hi, size_t n) const {
    const std::string hex = hrr::hash_hex(lo, hi);
    fs::create_directories(root / "blobs" / hex.substr(0, 2));
    std::ofstream f(root / "blobs" / hex.substr(0, 2) / (hex + ".blob"), std::ios::binary);
    f << std::string(n, 'b');
  }
  void add_code_object(uint64_t lo, uint64_t hi, size_t n) const {
    const std::string hex = hrr::hash_hex(lo, hi);
    std::ofstream f(root / "code_objects" / (hex + ".hsaco"), std::ios::binary);
    f << std::string(n, 'c');
  }
};

}  // namespace

/**
 * Test Description
 * ----------------
 *   - load_blob charges a blob's size to the host budget before reading it, and
 *     a second load of the same blob returns the cached bytes without
 *     charging again.
 */
TEST_CASE("Unit_HRR_HostBudget_BlobCacheChargesOnce", "[hrr]") {
  ScratchArchive arc("once");
  arc.add_blob(1, 2, 64);
  PlaybackContext ctx;
  ctx.archive_dir = arc.root.string();
  ctx.host_budget.set_cap(100);

  size_t sz = 0;
  const void* p1 = ctx.load_blob(1, 2, &sz);
  REQUIRE(p1 != nullptr);
  CHECK(sz == 64);
  CHECK(ctx.host_budget.used() == 64);

  const void* p2 = ctx.load_blob(1, 2, &sz);
  CHECK(p2 == p1);
  CHECK(ctx.host_budget.used() == 64);
  CHECK_FALSE(ctx.fatal_error.load());
}

/**
 * Test Description
 * ----------------
 *   - A blob that does not fit the remaining budget is refused: load_blob
 *     returns null, the budget is unchanged, the replay is ended, and the file
 *     is not read (the charge comes first). A missing blob returns null without
 *     touching the budget or ending the replay.
 */
TEST_CASE("Unit_HRR_HostBudget_BlobCacheRefusesOverBudgetAndMissing", "[hrr]") {
  ScratchArchive arc("refuse");
  arc.add_blob(1, 2, 64);
  arc.add_blob(3, 4, 64);
  PlaybackContext ctx;
  ctx.archive_dir = arc.root.string();
  ctx.host_budget.set_cap(100);
  REQUIRE(ctx.load_blob(1, 2, nullptr) != nullptr);

  // A missing blob: null, nothing charged, replay not ended.
  const std::string missing_err = hrr_test::capture_stderr(
      [&] { CHECK(ctx.load_blob(9, 9, nullptr) == nullptr); });
  CHECK(ctx.host_budget.used() == 64);
  CHECK_FALSE(ctx.fatal_error.load());

  // 64 + 64 > 100.
  const std::string err = hrr_test::capture_stderr(
      [&] { CHECK(ctx.load_blob(3, 4, nullptr) == nullptr); });
  CHECK(ctx.host_budget.used() == 64);
  CHECK(ctx.fatal_error.load());
  CHECK(err.find("blob cache") != std::string::npos);
  CHECK(err.find("--max-host-bytes") != std::string::npos);
}

/**
 * Test Description
 * ----------------
 *   - Threads that miss on the same blob at once end up with exactly one
 *     charge: the loser of the insert gives its charge back.
 */
TEST_CASE("Unit_HRR_HostBudget_ConcurrentLoadsOfOneBlobChargeOnce", "[hrr]") {
  ScratchArchive arc("concurrent");
  arc.add_blob(5, 6, 50);
  PlaybackContext ctx;
  ctx.archive_dir = arc.root.string();
  ctx.host_budget.set_cap(1000);

  std::vector<std::thread> ts;
  std::atomic<int> ok{0};
  for (int t = 0; t < 8; ++t)
    ts.emplace_back([&] {
      if (ctx.load_blob(5, 6, nullptr)) ok.fetch_add(1);
    });
  for (auto& t : ts) t.join();
  CHECK(ok.load() == 8);
  CHECK(ctx.host_budget.used() == 50);
}

/**
 * Test Description
 * ----------------
 *   - The code-object cache goes through the same budget, under its own key:
 *     a blob and a code object with the same hash are two entries, two
 *     charges.
 */
TEST_CASE("Unit_HRR_HostBudget_CodeObjectCacheIsChargedSeparately", "[hrr]") {
  ScratchArchive arc("co");
  arc.add_blob(7, 8, 40);
  arc.add_code_object(7, 8, 30);
  PlaybackContext ctx;
  ctx.archive_dir = arc.root.string();
  ctx.host_budget.set_cap(100);

  size_t sz = 0;
  REQUIRE(ctx.load_blob(7, 8, &sz) != nullptr);
  CHECK(sz == 40);
  REQUIRE(ctx.load_code_object(7, 8, &sz) != nullptr);
  CHECK(sz == 30);
  CHECK(ctx.host_budget.used() == 70);

  // Another 40 would pass the cap.
  arc.add_code_object(9, 9, 40);
  (void)hrr_test::capture_stderr([&] { CHECK(ctx.load_code_object(9, 9, &sz) == nullptr); });
  CHECK(ctx.host_budget.used() == 70);
  CHECK(ctx.fatal_error.load());
}

/**
 * Test Description
 * ----------------
 *   - hipHostMalloc, hipMallocHost and hipHostRegister refuse a recorded size
 *     the budget cannot hold before they call HIP: hipErrorMemoryAllocation,
 *     the replay ended, nothing charged, no mapping recorded.
 */
TEST_CASE("Unit_HRR_HostBudget_HostAllocationHandlersRefuseOverBudget", "[hrr]") {
  {
    PlaybackContext ctx;
    ctx.host_budget.set_cap(100);
    hrr_test::ExactRecord<hrr_args_hipHostMalloc> rec;
    rec.get()->ptr = 0x1000;
    rec.get()->size = 101;
    hipError_t r = hipSuccess;
    (void)hrr_test::capture_stderr([&] { r = playback_hipHostMalloc(ctx, rec.bytes()); });
    CHECK(r == hipErrorMemoryAllocation);
    CHECK(ctx.fatal_error.load());
    CHECK(ctx.host_budget.used() == 0);
    CHECK_FALSE(ctx.has_alloc(0x1000));
  }
  {
    PlaybackContext ctx;
    ctx.host_budget.set_cap(100);
    hrr_test::ExactRecord<hrr_args_hipMallocHost> rec;
    rec.get()->ptr = 0x2000;
    rec.get()->size = UINT64_MAX;  // a size that would wrap any naive sum
    hipError_t r = hipSuccess;
    (void)hrr_test::capture_stderr([&] { r = playback_hipMallocHost(ctx, rec.bytes()); });
    CHECK(r == hipErrorMemoryAllocation);
    CHECK(ctx.fatal_error.load());
    CHECK(ctx.host_budget.used() == 0);
  }
  {
    PlaybackContext ctx;
    ctx.host_budget.set_cap(100);
    hrr_test::ExactRecord<hrr_args_hipHostRegister> rec;
    rec.get()->hostPtr = 0x3000;
    rec.get()->sizeBytes = 1u << 20;
    hipError_t r = hipSuccess;
    (void)hrr_test::capture_stderr([&] { r = playback_hipHostRegister(ctx, rec.bytes()); });
    CHECK(r == hipErrorMemoryAllocation);
    CHECK(ctx.fatal_error.load());
    CHECK(ctx.host_budget.used() == 0);
    CHECK_FALSE(ctx.has_alloc(0x3000));
  }
}

/**
 * Test Description
 * ----------------
 *   - host_landing_buffer charges only a buffer's growth, hands back the same
 *     buffer for a smaller request, and on a refusal returns null, ends the
 *     replay and leaves the buffer as it was.
 */
TEST_CASE("Unit_HRR_HostBudget_LandingBufferChargesGrowthOnly", "[hrr]") {
  PlaybackContext ctx;
  ctx.host_budget.set_cap(100);

  void* a = ctx.host_landing_buffer(0x10, 64);
  REQUIRE(a != nullptr);
  CHECK(ctx.host_budget.used() == 64);

  CHECK(ctx.host_landing_buffer(0x10, 32) == a);  // smaller: same buffer, no charge
  CHECK(ctx.host_budget.used() == 64);

  void* b = ctx.host_landing_buffer(0x10, 96);  // growth of 32: fits
  REQUIRE(b != nullptr);
  CHECK(ctx.host_budget.used() == 96);

  (void)hrr_test::capture_stderr([&] { CHECK(ctx.host_landing_buffer(0x10, 200) == nullptr); });
  CHECK(ctx.host_budget.used() == 96);
  CHECK(ctx.fatal_error.load());

  CHECK(ctx.host_landing_buffer(0, 8) == nullptr);  // a null recorded key stays null
}

/** @} */
