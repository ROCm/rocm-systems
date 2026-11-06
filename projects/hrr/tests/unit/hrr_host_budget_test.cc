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
 * paths. No case needs a GPU; the few that reach a HIP call expect its failure
 * without one.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr_handler_test_util.h"
#include "hrr_reader.h"
#include "hrr/hrr_api_args.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
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
hipError_t playback_hipFreeHost(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipMallocManaged(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipMemCreate(PlaybackContext& ctx, const uint8_t* pl);

// AddressSanitizer aborts on a failed allocation unless told to return null,
// which the allocation-failure case below needs.
extern "C" const char* __asan_default_options() { return "allocator_may_return_null=1"; }

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
    rec.get()->blob_hash_lo = 1;  // capture records the registered bytes
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
 *   - The replay's dispatch table (what dispatch_event calls) reaches the
 *     charging handlers for hipHostMalloc, hipMallocHost and hipFreeHost, not a
 *     generated one that skips the budget (or, for hipMallocHost/hipFreeHost, a
 *     NOOP). Calling a dispatch entry over budget refuses the allocation.
 */
TEST_CASE("Unit_HRR_HostBudget_DispatchTableUsesTheChargingHandlers", "[hrr]") {
  CHECK(hrr_playback_dispatch[HRR_API_HIPHOSTMALLOC] == &playback_hipHostMalloc);
  CHECK(hrr_playback_dispatch[HRR_API_HIPMALLOCHOST] == &playback_hipMallocHost);
  CHECK(hrr_playback_dispatch[HRR_API_HIPFREEHOST] == &playback_hipFreeHost);

  PlaybackContext ctx;
  ctx.host_budget.set_cap(100);
  hrr_test::ExactRecord<hrr_args_hipMallocHost> rec;
  rec.get()->ptr = 0x4000;
  rec.get()->size = 101;
  hipError_t r = hipSuccess;
  (void)hrr_test::capture_stderr(
      [&] { r = hrr_playback_dispatch[HRR_API_HIPMALLOCHOST](ctx, rec.bytes()); });
  CHECK(r == hipErrorMemoryAllocation);
  CHECK(ctx.fatal_error.load());
}

/**
 * Test Description
 * ----------------
 *   - Recording a pinned allocation again at an address that still holds one,
 *     through the dispatch entries replay calls, drops the old entry. The new
 *     allocation is over budget here so no HIP allocation is made: only the
 *     replaced entry's release is observed.
 *   - The old charge goes back only if the runtime let go of the buffer. A
 *     hipHostMalloc entry's buffer is freed (a null live pointer frees
 *     cleanly where a runtime is up, and fails without one, which the charge
 *     must then follow); a failing hipHostFree keeps the charge, and the
 *     buffer is left alone. A hipHostRegister entry's charge stays either way,
 *     and a failing hipHostUnregister leaves its buffer alone too. The seeds
 *     are not pinned, so the failures need no GPU.
 */
TEST_CASE("Unit_HRR_HostBudget_DispatchFreesReplacedHostAllocation", "[hrr]") {
  auto run = [](uint32_t api, auto& rec, AllocKind kind, void* live, size_t expect_used) {
    PlaybackContext ctx;
    ctx.host_budget.set_cap(100);
    REQUIRE(ctx.charge_host(60, "seed"));
    ctx.record_alloc(0x5000, live, 60, kind);
    if (kind == AllocKind::HostRegister) ctx.host_reg_bufs[0x5000] = live;
    REQUIRE(ctx.host_budget.used() == 60);
    rec.get()->ptr = 0x5000;
    rec.get()->size = 101;
    (void)hrr_test::capture_stderr([&] { (void)hrr_playback_dispatch[api](ctx, rec.bytes()); });
    CHECK_FALSE(ctx.has_alloc(0x5000));
    CHECK(ctx.host_budget.used() == expect_used);
  };

  // hipHostFree(nullptr) succeeds with a runtime up and fails without one.
  const size_t freed_used = hipHostFree(nullptr) == hipSuccess ? 0 : 60;
  hrr_test::ExactRecord<hrr_args_hipHostMalloc> hm;
  run(HRR_API_HIPHOSTMALLOC, hm, AllocKind::HostMalloc, nullptr, freed_used);
  hrr_test::ExactRecord<hrr_args_hipMallocHost> mh;
  run(HRR_API_HIPMALLOCHOST, mh, AllocKind::HostMalloc, nullptr, freed_used);

  // hipHostFree refuses a pointer the runtime does not own: the charge stays
  // and the buffer is not touched.
  void* plain = std::malloc(64);
  REQUIRE(plain != nullptr);
  std::memset(plain, 0x5A, 64);
  hrr_test::ExactRecord<hrr_args_hipHostMalloc> hm_plain;
  run(HRR_API_HIPHOSTMALLOC, hm_plain, AllocKind::HostMalloc, plain, 60);
  // Under AddressSanitizer a freed buffer fails this read.
  CHECK(static_cast<unsigned char*>(plain)[0] == 0x5A);
  std::free(plain);

  // hipHostUnregister refuses a buffer that was never registered: same result.
  void* reg = std::malloc(64);
  REQUIRE(reg != nullptr);
  std::memset(reg, 0x5A, 64);
  hrr_test::ExactRecord<hrr_args_hipHostMalloc> hm_reg;
  run(HRR_API_HIPHOSTMALLOC, hm_reg, AllocKind::HostRegister, reg, 60);
  CHECK(static_cast<unsigned char*>(reg)[0] == 0x5A);
  std::free(reg);
}

/**
 * Test Description
 * ----------------
 *   - hipMallocManaged refuses a recorded size the budget cannot hold before it
 *     calls HIP (managed memory can be backed from host memory): the
 *     hipHostMalloc refusal, so hipErrorMemoryAllocation, the replay ended,
 *     the message names --max-host-bytes, nothing charged, no mapping recorded.
 *     It does so through the dispatch table too, and a size that would wrap a
 *     naive sum over an existing charge is refused the same way, the earlier
 *     charge left as it was.
 */
TEST_CASE("Unit_HRR_HostBudget_ManagedAllocationRefusedOverBudget", "[hrr]") {
  CHECK(hrr_playback_dispatch[HRR_API_HIPMALLOCMANAGED] == &playback_hipMallocManaged);
  struct Case {
    uint64_t cap, seed, size;
  };
  for (const Case c : {Case{100, 0, 101}, Case{100, 0, uint64_t{1} << 32},
                       Case{UINT64_MAX, 8192, UINT64_MAX - 4096}}) {
    PlaybackContext ctx;
    ctx.host_budget.set_cap(c.cap);
    if (c.seed) REQUIRE(ctx.charge_host(c.seed, "seed"));
    const uint64_t size = c.size;
    hrr_test::ExactRecord<hrr_args_hipMalloc> rec;
    rec.get()->ptr = 0x6000;
    rec.get()->size = size;
    hipError_t r = hipSuccess;
    const std::string err = hrr_test::capture_stderr(
        [&] { r = hrr_playback_dispatch[HRR_API_HIPMALLOCMANAGED](ctx, rec.bytes()); });
    CHECK(r == hipErrorMemoryAllocation);
    CHECK(ctx.fatal_error.load());
    CHECK(ctx.host_budget.used() == c.seed);
    CHECK_FALSE(ctx.has_alloc(0x6000));
    CHECK(err.find("hipMallocManaged") != std::string::npos);
    CHECK(err.find("--max-host-bytes") != std::string::npos);
  }
}

/**
 * Test Description
 * ----------------
 *   - hipMemCreate with a host location (Host, HostNuma or HostNumaCurrent)
 *     refuses a size the budget cannot hold before it calls HIP: the same
 *     refusal as hipHostMalloc, nothing charged, no handle recorded. This holds
 *     through the dispatch table too.
 */
TEST_CASE("Unit_HRR_HostBudget_HostMemCreateRefusedOverBudget", "[hrr]") {
  CHECK(hrr_playback_dispatch[HRR_API_HIPMEMCREATE] == &playback_hipMemCreate);
  for (hipMemLocationType loc : {hipMemLocationTypeHost, hipMemLocationTypeHostNuma,
                                 hipMemLocationTypeHostNumaCurrent}) {
    PlaybackContext ctx;
    ctx.host_budget.set_cap(100);
    hrr_test::ExactRecord<hrr_args_hipMemCreate> rec;
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = loc;
    prop.location.id = 0;
    rec.get()->handle = 0x7000;
    rec.get()->size = uint64_t{4} << 30;
    rec.get()->prop_present = 1;
    std::memcpy(rec.get()->prop_bytes, &prop, sizeof(prop));
    hipError_t r = hipSuccess;
    const std::string err = hrr_test::capture_stderr(
        [&] { r = hrr_playback_dispatch[HRR_API_HIPMEMCREATE](ctx, rec.bytes()); });
    CHECK(r == hipErrorMemoryAllocation);
    CHECK(ctx.fatal_error.load());
    CHECK(ctx.host_budget.used() == 0);
    CHECK(ctx.translate_vmm_handle(0x7000) == hipMemGenericAllocationHandle_t{});
    CHECK(err.find("hipMemCreate") != std::string::npos);
    CHECK(err.find("--max-host-bytes") != std::string::npos);
  }
}

/**
 * Test Description
 * ----------------
 *   - hipMemCreate with a device location is not charged to the host budget:
 *     a size far over the cap does not end the replay and leaves nothing
 *     charged, whatever the HIP call returns.
 */
TEST_CASE("Unit_HRR_HostBudget_DeviceMemCreateNotCharged", "[hrr]") {
  // Without a device hipMemCreate fails at the ordinal check, before the
  // charge decision, so the case would pass even if device memory were charged.
  if (!hrr_test::have_device())
    SKIP("needs a GPU: without one the call fails before the charge decision");
  PlaybackContext ctx;
  ctx.host_budget.set_cap(100);
  hrr_test::ExactRecord<hrr_args_hipMemCreate> rec;
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;
  rec.get()->handle = 0x7100;
  rec.get()->size = uint64_t{1} << 50;  // no device holds this: HIP refuses it
  rec.get()->prop_present = 1;
  std::memcpy(rec.get()->prop_bytes, &prop, sizeof(prop));
  const std::string err = hrr_test::capture_stderr(
      [&] { (void)hrr_playback_dispatch[HRR_API_HIPMEMCREATE](ctx, rec.bytes()); });
  CHECK_FALSE(ctx.fatal_error.load());
  CHECK(ctx.host_budget.used() == 0);
  CHECK(err.find("--max-host-bytes") == std::string::npos);
}

/**
 * Test Description
 * ----------------
 *   - A managed entry holds its charge until it is removed: remove_alloc (what
 *     hipFree calls) returns it, a device entry returns nothing, and removing
 *     the same address twice does not give back more than was charged.
 *   - A managed allocation recorded again at an address that still holds one
 *     frees the old entry and gives its charge back before the new one is
 *     charged. The new allocation is over budget here so no HIP allocation is
 *     made: only the replaced entry's release is observed.
 */
TEST_CASE("Unit_HRR_HostBudget_ManagedAllocationReturnsChargeOnFree", "[hrr]") {
  {
    PlaybackContext ctx;
    ctx.host_budget.set_cap(100);
    REQUIRE(ctx.charge_host(60, "seed managed"));
    ctx.record_alloc(0x8000, nullptr, 60, AllocKind::Managed);
    REQUIRE(ctx.charge_host(30, "seed device-side"));  // unrelated charge stays
    ctx.record_alloc(0x9000, nullptr, 30, AllocKind::Device);
    CHECK(ctx.host_budget.used() == 90);

    ctx.remove_alloc(0x9000);  // a device entry was never charged here
    CHECK(ctx.host_budget.used() == 90);
    ctx.remove_alloc(0x8000);
    CHECK(ctx.host_budget.used() == 30);
    ctx.remove_alloc(0x8000);  // already gone
    CHECK(ctx.host_budget.used() == 30);
  }
  {
    // The replaced entry is freed with hipFree(nullptr), which succeeds with a
    // runtime up and fails without one; a failed free keeps the charge.
    const size_t freed_used = hipFree(nullptr) == hipSuccess ? 0 : 60;
    PlaybackContext ctx;
    ctx.host_budget.set_cap(100);
    REQUIRE(ctx.charge_host(60, "seed"));
    ctx.record_alloc(0xA000, nullptr, 60, AllocKind::Managed);
    hrr_test::ExactRecord<hrr_args_hipMalloc> rec;
    rec.get()->ptr = 0xA000;
    rec.get()->size = 101;
    (void)hrr_test::capture_stderr(
        [&] { (void)hrr_playback_dispatch[HRR_API_HIPMALLOCMANAGED](ctx, rec.bytes()); });
    CHECK_FALSE(ctx.has_alloc(0xA000));
    CHECK(ctx.host_budget.used() == freed_used);
  }
}

/**
 * Test Description
 * ----------------
 *   - host_landing_buffer hands back the same buffer for a smaller request,
 *     charges a larger one whole (the outgrown block stays resident for the
 *     nodes holding it), and on a refusal returns null, ends the replay and
 *     leaves the buffer as it was.
 */
TEST_CASE("Unit_HRR_HostBudget_LandingBufferChargesWholeBlock", "[hrr]") {
  PlaybackContext ctx;
  ctx.host_budget.set_cap(200);

  void* a = ctx.host_landing_buffer(0x10, 64);
  REQUIRE(a != nullptr);
  CHECK(ctx.host_budget.used() == 64);

  CHECK(ctx.host_landing_buffer(0x10, 32) == a);  // smaller: same buffer, no charge
  CHECK(ctx.host_budget.used() == 64);

  void* b = ctx.host_landing_buffer(0x10, 96);  // a new 96-byte block; the 64 stays: 160 fits
  REQUIRE(b != nullptr);
  CHECK(b != a);
  CHECK(ctx.host_budget.used() == 160);

  (void)hrr_test::capture_stderr([&] { CHECK(ctx.host_landing_buffer(0x10, 200) == nullptr); });
  CHECK(ctx.host_budget.used() == 160);
  CHECK(ctx.fatal_error.load());
  CHECK(ctx.host_landing_buffer(0x10, 96) == b);  // the refusal left the buffer as it was

  CHECK(ctx.host_landing_buffer(0, 8) == nullptr);  // a null recorded key stays null
}

/**
 * Test Description
 * ----------------
 *   - A graph node keeps the host pointer it was given, so growing the buffer
 *     for the same recorded key must not free it: the pointer from before the
 *     growth stays writable (AddressSanitizer aborts on a write through a freed
 *     block).
 */
TEST_CASE("Unit_HRR_HostBudget_LandingBufferPointerSurvivesGrowth", "[hrr]") {
  PlaybackContext ctx;

  auto* n1 = static_cast<uint8_t*>(ctx.host_landing_buffer(0x20, 16));
  REQUIRE(n1 != nullptr);
  REQUIRE(ctx.host_landing_buffer(0x20, 4096) != nullptr);  // grows the same key

  // Launching graph 1 now writes through the pointer node 1 was given.
  std::memset(n1, 0x44, 16);
  CHECK(n1[15] == 0x44);
}

/**
 * Test Description
 * ----------------
 *   - A host-located hipMemCreate handle keeps its charge while a hipMemMap
 *     holds it: hipMemRelease alone does not give the bytes back (the runtime
 *     keeps the memory until the last hipMemUnmap), the unmap does. A handle
 *     mapped twice is returned only by the second unmap, and one never mapped is
 *     returned by the release. Each step checks the exact figure, so a charge
 *     returned twice shows too.
 */
TEST_CASE("Unit_HRR_HostBudget_MemCreateChargeHeldWhileMapped", "[hrr]") {
  auto* h1 = reinterpret_cast<hipMemGenericAllocationHandle_t>(uintptr_t{0x1000});
  auto* h2 = reinterpret_cast<hipMemGenericAllocationHandle_t>(uintptr_t{0x2000});
  auto* va = reinterpret_cast<void*>(uintptr_t{0x100000});
  auto* va2 = reinterpret_cast<void*>(uintptr_t{0x200000});
  PlaybackContext ctx;
  ctx.host_budget.set_cap(1000);

  // Create, map, release: the charge stays, then the unmap returns it.
  REQUIRE(ctx.charge_host(300, "seed"));
  ctx.vmm_note_create(0x7000, h1, 300);
  ctx.vmm_note_map(va, 300, h1);
  ctx.vmm_note_release(h1);
  CHECK(ctx.host_budget.used() == 300);
  ctx.vmm_note_unmap(va, 300);
  CHECK(ctx.host_budget.used() == 0);

  // Two mappings: only the last unmap returns it.
  REQUIRE(ctx.charge_host(200, "seed"));
  ctx.vmm_note_create(0x7100, h2, 200);
  ctx.vmm_note_map(va, 100, h2);
  ctx.vmm_note_map(va2, 100, h2);
  ctx.vmm_note_release(h2);
  ctx.vmm_note_unmap(va, 100);
  CHECK(ctx.host_budget.used() == 200);
  ctx.vmm_note_unmap(va2, 100);
  CHECK(ctx.host_budget.used() == 0);
  ctx.vmm_note_unmap(va2, 100);  // nothing mapped any more
  CHECK(ctx.host_budget.used() == 0);

  // Never mapped: the release returns it at once.
  REQUIRE(ctx.charge_host(50, "seed"));
  ctx.vmm_note_create(0x7200, h1, 50);
  ctx.vmm_note_release(h1);
  CHECK(ctx.host_budget.used() == 0);
}

/**
 * Test Description
 * ----------------
 *   - Host blocks start on a 64-byte cache line, so two small blocks never
 *     share one (hipHostRegister's alignment advice).
 */
TEST_CASE("Unit_HRR_HostBudget_HostBlocksStartOnCacheLine", "[hrr]") {
  PlaybackContext ctx;
  for (uint64_t rec = 1; rec <= 8; ++rec) {
    void* p = ctx.host_landing_buffer(rec, 8);
    REQUIRE(p != nullptr);
    CHECK(reinterpret_cast<uintptr_t>(p) % 64 == 0);
  }
}

/**
 * Test Description
 * ----------------
 *   - Releasing a region's stand-in buffer removes the alloc_map entry only if
 *     that entry is the stand-in. When the archive has since allocated managed
 *     memory at the region base, region DEL leaves the managed entry and its
 *     charge in place.
 */
TEST_CASE("Unit_HRR_HostBudget_RegionReleaseKeepsReplacingAllocation", "[hrr]") {
  auto* managed = reinterpret_cast<void*>(uintptr_t{0xB0000});
  PlaybackContext ctx;
  ctx.host_budget.set_cap(100);
  REQUIRE(ctx.charge_host(60, "seed managed"));
  ctx.record_alloc(0xB000, managed, 60, AllocKind::Managed);

  hrr_release_region(ctx, 0xB000, nullptr);  // the stand-in was some other buffer
  CHECK(ctx.has_alloc(0xB000));
  CHECK(ctx.host_budget.used() == 60);

  hrr_release_region(ctx, 0xB000, managed);  // now it is the entry's own buffer
  CHECK_FALSE(ctx.has_alloc(0xB000));
}

/**
 * Test Description
 * ----------------
 *   - A hipHostRegister the budget allows but the allocator cannot back is
 *     refused with hipErrorMemoryAllocation, and its charge is given back.
 */
TEST_CASE("Unit_HRR_HostBudget_HostRegisterAllocationFailureRefuses", "[hrr]") {
  constexpr uint64_t kHuge = uint64_t{1} << 62;  // more than any address space
  PlaybackContext ctx;
  ctx.host_budget.set_cap(kHuge);

  hrr_test::ExactRecord<hrr_args_hipHostRegister> reg;
  reg.get()->hostPtr = 0x6000;
  reg.get()->sizeBytes = kHuge;
  reg.get()->blob_hash_lo = 1;
  CHECK(playback_hipHostRegister(ctx, reg.bytes()) == hipErrorMemoryAllocation);
  CHECK(ctx.host_budget.used() == 0);
}

/** @} */
