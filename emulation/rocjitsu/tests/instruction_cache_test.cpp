// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/instruction_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace rocjitsu::amdgpu {
class GpuMemoryFetchabilityTestPeer {
public:
  static std::unique_lock<std::shared_mutex> lock_registry(GpuMemory &memory) {
    return std::unique_lock(memory.vmid_mutex_);
  }

  static void unregister_and_pause(GpuMemory &memory, uint32_t vmid, std::promise<void> &removed,
                                   const std::shared_future<void> &resume) {
    memory.update_vmid_registration(vmid, [&](auto it) {
      memory.vmid_table_.erase(it);
      removed.set_value();
      resume.wait();
      return true;
    });
  }
};
} // namespace rocjitsu::amdgpu

namespace {

using rocjitsu::amdgpu::GpuMemory;
using rocjitsu::amdgpu::InstructionCache;
namespace amdgpu = rocjitsu::amdgpu;

constexpr uint64_t kCodeBase = 0x200000;
constexpr uint32_t kMappedVmid = 7;

void register_fetchable_process(GpuMemory &memory, rocjitsu::KfdProcess &process) {
  memory.register_process(process.process_id(), &process.page_table_, &process.page_table_mutex_,
                          process.page_table_generation(), process.page_table_request_mutex(),
                          process.page_table_fetchability_epoch());
}

TEST(FetchabilityCacheTest, PositiveHitDoesNotTakeRegistryOrPageTableLocks) {
  GpuMemory memory("memory");
  rocjitsu::KfdProcess process(kMappedVmid);
  std::array<uint8_t, InstructionCache::kLineSize> code{};
  process.map_pages(kCodeBase, code.data(), code.size());
  register_fetchable_process(memory, process);
  GpuMemory::FetchabilityCache cache;
  ASSERT_TRUE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));

  auto registry_lock = amdgpu::GpuMemoryFetchabilityTestPeer::lock_registry(memory);
  std::unique_lock lock(process.page_table_mutex_);
  auto hit = std::async(std::launch::async,
                        [&] { return memory.is_fetchable(kCodeBase + 4, kMappedVmid, cache); });
  const auto status = hit.wait_for(std::chrono::seconds(2));
  lock.unlock();
  registry_lock.unlock();
  EXPECT_EQ(status, std::future_status::ready);
  EXPECT_TRUE(hit.get());
}

TEST(FetchabilityCacheTest, MappingMutationsAndPageKeysInvalidateThePositive) {
  GpuMemory memory("memory");
  rocjitsu::KfdProcess process(kMappedVmid);
  std::array<uint8_t, InstructionCache::kLineSize> code{};
  register_fetchable_process(memory, process);
  GpuMemory::FetchabilityCache cache;
  EXPECT_FALSE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));

  process.map_pages(kCodeBase, code.data(), code.size());
  ASSERT_TRUE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));
  EXPECT_FALSE(memory.is_fetchable(kCodeBase + GpuMemory::PAGE_SIZE, kMappedVmid, cache));
  ASSERT_TRUE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));
  EXPECT_FALSE(memory.is_fetchable(kCodeBase, kMappedVmid + 1, cache));
  ASSERT_TRUE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));

  // Page presence, including disjoint sub-page mappings, matches is_mapped().
  process.map_pages(kCodeBase + 128, code.data(), code.size());
  process.unmap_pages(kCodeBase, code.size());
  EXPECT_TRUE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));
  process.unmap_pages(kCodeBase + 128, code.size());
  EXPECT_FALSE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));
  process.map_pages(kCodeBase, code.data(), code.size());
  EXPECT_TRUE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));
}

TEST(FetchabilityCacheTest, InvalidatedPositiveRequiresAFreshLockedSnapshot) {
  GpuMemory memory("memory");
  rocjitsu::KfdProcess process(kMappedVmid);
  std::array<uint8_t, InstructionCache::kLineSize> code{};
  process.map_pages(kCodeBase, code.data(), code.size());
  register_fetchable_process(memory, process);
  GpuMemory::FetchabilityCache cache;
  ASSERT_TRUE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));
  process.map_pages(kCodeBase, code.data(), code.size());

  auto registry_lock = amdgpu::GpuMemoryFetchabilityTestPeer::lock_registry(memory);
  std::unique_lock lock(process.page_table_mutex_);
  std::promise<void> started;
  auto started_future = started.get_future();
  auto refill = std::async(std::launch::async, [&] {
    started.set_value();
    return memory.is_fetchable(kCodeBase, kMappedVmid, cache);
  });
  started_future.wait();
  const auto status = refill.wait_for(std::chrono::milliseconds(100));
  lock.unlock();
  registry_lock.unlock();
  EXPECT_EQ(status, std::future_status::timeout);
  EXPECT_TRUE(refill.get());
}

TEST(FetchabilityCacheTest, ReplacingARegistrationInvalidatesItsOldToken) {
  GpuMemory memory("memory");
  rocjitsu::KfdProcess first(kMappedVmid);
  rocjitsu::KfdProcess replacement(kMappedVmid);
  std::array<uint8_t, InstructionCache::kLineSize> code{};
  first.map_pages(kCodeBase, code.data(), code.size());
  register_fetchable_process(memory, first);
  GpuMemory::FetchabilityCache cache;
  ASSERT_TRUE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));

  register_fetchable_process(memory, replacement);
  EXPECT_FALSE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));
}

TEST(FetchabilityCacheTest, InProgressUnregistrationInvalidatesCachedChecks) {
  GpuMemory memory("memory");
  rocjitsu::KfdProcess process(kMappedVmid);
  std::array<uint8_t, InstructionCache::kLineSize> code{};
  process.map_pages(kCodeBase, code.data(), code.size());
  register_fetchable_process(memory, process);
  GpuMemory::FetchabilityCache cache;
  ASSERT_TRUE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));

  std::promise<void> removed, resume;
  auto removed_future = removed.get_future();
  auto resume_future = resume.get_future().share();
  auto unregister = std::async(std::launch::async, [&] {
    amdgpu::GpuMemoryFetchabilityTestPeer::unregister_and_pause(memory, kMappedVmid, removed,
                                                                resume_future);
  });
  removed_future.wait();

  // The binding is gone, but the registry update still holds its lock. A
  // cached check must refill after that update instead of accepting the old
  // positive while waiting for the generation to be published.
  std::promise<void> started;
  auto started_future = started.get_future();
  auto check = std::async(std::launch::async, [&] {
    started.set_value();
    return memory.is_fetchable(kCodeBase, kMappedVmid, cache);
  });
  started_future.wait();
  const auto status = check.wait_for(std::chrono::milliseconds(100));
  resume.set_value();
  unregister.get();
  EXPECT_EQ(status, std::future_status::timeout);
  EXPECT_FALSE(check.get());
}

TEST(FetchabilityCacheTest, RetainedTokenSurvivesUnregistrationAndProcessDestruction) {
  GpuMemory memory("memory");
  auto process = std::make_unique<rocjitsu::KfdProcess>(kMappedVmid);
  std::array<uint8_t, InstructionCache::kLineSize> code{};
  process->map_pages(kCodeBase, code.data(), code.size());
  register_fetchable_process(memory, *process);
  std::weak_ptr<const std::atomic<uint64_t>> epoch = process->page_table_fetchability_epoch();
  GpuMemory::FetchabilityCache cache;
  ASSERT_TRUE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));

  memory.unregister_process(kMappedVmid);
  process.reset();
  EXPECT_FALSE(epoch.expired());
  EXPECT_FALSE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));
  EXPECT_TRUE(epoch.expired());
}

TEST(FetchabilityCacheTest, ReconstructedMemoryDoesNotReuseAPositiveSnapshot) {
  std::optional<GpuMemory> memory(std::in_place, "first");
  rocjitsu::KfdProcess first(kMappedVmid);
  rocjitsu::KfdProcess replacement(kMappedVmid);
  std::array<uint8_t, InstructionCache::kLineSize> code{};
  first.map_pages(kCodeBase, code.data(), code.size());
  register_fetchable_process(*memory, first);
  GpuMemory::FetchabilityCache cache;
  ASSERT_TRUE(memory->is_fetchable(kCodeBase, kMappedVmid, cache));

  // The second memory uses the same address and the same registry generation.
  memory.reset();
  memory.emplace("replacement");
  register_fetchable_process(*memory, replacement);
  EXPECT_FALSE(memory->is_fetchable(kCodeBase, kMappedVmid, cache));
}

TEST(FetchabilityCacheTest, LegacyRegistrationWithoutATokenIsUncached) {
  GpuMemory memory("memory");
  rocjitsu::KfdProcess process(kMappedVmid);
  std::array<uint8_t, InstructionCache::kLineSize> code{};
  process.map_pages(kCodeBase, code.data(), code.size());
  memory.register_process(kMappedVmid, &process.page_table_, &process.page_table_mutex_,
                          process.page_table_generation());
  GpuMemory::FetchabilityCache cache;
  ASSERT_TRUE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));
  {
    std::unique_lock lock(process.page_table_mutex_);
    // Legacy embedders need not publish either kind of generation.
    process.page_table_.clear();
  }
  EXPECT_FALSE(memory.is_fetchable(kCodeBase, kMappedVmid, cache));
}

TEST(FetchabilityCacheTest, ClientMemoryFallbackIsNotCached) {
  GpuMemory memory("memory");
  rocjitsu::KfdProcess process(kMappedVmid);
  register_fetchable_process(memory, process);
  const int mem_fd = ::open("/proc/self/mem", O_RDONLY | O_CLOEXEC);
  ASSERT_GE(mem_fd, 0);
  memory.set_process_mem_fd(kMappedVmid, mem_fd);
  ::close(mem_fd);
  uint8_t byte = 42;
  const uint64_t address = reinterpret_cast<uint64_t>(&byte);
  GpuMemory::FetchabilityCache cache;
  ASSERT_TRUE(memory.is_fetchable(address, kMappedVmid, cache));

  memory.set_process_mem_fd(kMappedVmid, -1);
  EXPECT_FALSE(memory.is_fetchable(address, kMappedVmid, cache));
}

/// @brief Fill @p bytes of code memory at kCodeBase with a per-byte pattern.
std::vector<uint8_t> fill_code(GpuMemory &memory, size_t bytes, uint8_t salt, uint32_t vmid = 0) {
  std::vector<uint8_t> expected(bytes);
  for (size_t i = 0; i < bytes; ++i)
    expected[i] = static_cast<uint8_t>((i * 7) ^ salt);
  memory.write_block(kCodeBase, std::span<const uint8_t>(expected), vmid);
  return expected;
}

std::array<uint8_t, InstructionCache::kFetchBytes>
fetch_at(InstructionCache &icache, const GpuMemory &memory, uint64_t pc, uint32_t vmid = 0) {
  std::array<uint8_t, InstructionCache::kFetchBytes> got{};
  icache.fetch(memory, pc, vmid, got.data());
  return got;
}

// Every four-byte-aligned PC in a two-line window, including the offsets whose
// fetch window runs off the end of a line, must return the backing bytes.
TEST(InstructionCacheTest, FetchMatchesBackingMemoryAtEveryAlignedOffset) {
  GpuMemory memory("memory");
  InstructionCache icache;
  const size_t span = InstructionCache::kLineSize * 3;
  const std::vector<uint8_t> expected = fill_code(memory, span, 0x5a);

  for (uint32_t off = 0; off + InstructionCache::kFetchBytes <= span; off += 4) {
    const auto got = fetch_at(icache, memory, kCodeBase + off);
    EXPECT_TRUE(std::equal(got.begin(), got.end(), expected.begin() + off))
        << "mismatch at offset " << off;
  }
}

// The straddling offsets are the interesting ones: assert they are actually
// exercised above, so the loop cannot silently stop covering them.
TEST(InstructionCacheTest, FetchWindowStraddlesALineBoundary) {
  static_assert(InstructionCache::kLineSize % InstructionCache::kFetchBytes == 0);
  GpuMemory memory("memory");
  InstructionCache icache;
  const std::vector<uint8_t> expected = fill_code(memory, InstructionCache::kLineSize * 2, 0x3c);

  // Offset 60 puts 4 bytes in one line and 12 in the next.
  constexpr uint32_t kStraddle = InstructionCache::kLineSize - 4;
  ASSERT_GT(kStraddle + InstructionCache::kFetchBytes, InstructionCache::kLineSize);

  const auto got = fetch_at(icache, memory, kCodeBase + kStraddle);
  EXPECT_TRUE(std::equal(got.begin(), got.end(), expected.begin() + kStraddle));
}

// The I$ is deliberately not coherent with data writes, matching hardware: a
// write to code memory is invisible until something issues s_icache_inv.
TEST(InstructionCacheTest, CachedLineSurvivesABackingWriteUntilInvalidated) {
  GpuMemory memory("memory");
  InstructionCache icache;
  const std::vector<uint8_t> first = fill_code(memory, InstructionCache::kLineSize, 0x11);

  const auto before = fetch_at(icache, memory, kCodeBase);
  EXPECT_TRUE(std::equal(before.begin(), before.end(), first.begin()));

  const std::vector<uint8_t> second = fill_code(memory, InstructionCache::kLineSize, 0x22);
  ASSERT_NE(first, second);

  const auto stale = fetch_at(icache, memory, kCodeBase);
  EXPECT_TRUE(std::equal(stale.begin(), stale.end(), first.begin()))
      << "the I$ must not observe a data write on its own";

  icache.invalidate_all();
  const auto after = fetch_at(icache, memory, kCodeBase);
  EXPECT_TRUE(std::equal(after.begin(), after.end(), second.begin()));
}

// Lines are tagged by vmid, so the same address in two address spaces must not
// alias even though it selects the same line. vmid 1 has no mapping and no
// client process, so it reaches the same sparse backing as vmid 0 -- rewriting
// that backing between the two fetches is what makes the miss observable at
// all. Without it, dropping the vmid check from line_for() would still pass.
TEST(InstructionCacheTest, LinesDoNotAliasAcrossVmids) {
  GpuMemory memory("memory");
  InstructionCache icache;
  const std::vector<uint8_t> vm0 = fill_code(memory, InstructionCache::kLineSize, 0x01, 0);

  const auto got0 = fetch_at(icache, memory, kCodeBase, 0);
  EXPECT_TRUE(std::equal(got0.begin(), got0.end(), vm0.begin()));

  // A vmid 1 fetch of the same address must miss and refill, so it sees the
  // rewritten bytes rather than the line vmid 0 just installed.
  const std::vector<uint8_t> vm1 = fill_code(memory, InstructionCache::kLineSize, 0x02, 0);
  ASSERT_NE(vm0, vm1);
  const auto got1 = fetch_at(icache, memory, kCodeBase, 1);
  EXPECT_TRUE(std::equal(got1.begin(), got1.end(), vm1.begin()))
      << "the vmid 1 lookup returned vmid 0's line";

  // ...and it must have installed a line under its own tag, not bypassed the
  // cache: a second rewrite is invisible to the vmid 1 fetch that follows it.
  const std::vector<uint8_t> vm2 = fill_code(memory, InstructionCache::kLineSize, 0x03, 0);
  ASSERT_NE(vm1, vm2);
  const auto again1 = fetch_at(icache, memory, kCodeBase, 1);
  EXPECT_TRUE(std::equal(again1.begin(), again1.end(), vm1.begin()))
      << "the vmid 1 fetch did not cache its line";
}

// A working set larger than the cache must still read correctly once lines
// start evicting each other.
TEST(InstructionCacheTest, FetchIsCorrectWhenTheWorkingSetExceedsTheCache) {
  GpuMemory memory("memory");
  InstructionCache icache;
  const size_t span = InstructionCache::kCacheBytes * 2;
  const std::vector<uint8_t> expected = fill_code(memory, span, 0x7e);

  for (int pass = 0; pass < 2; ++pass) {
    for (uint32_t off = 0; off + InstructionCache::kFetchBytes <= span;
         off += InstructionCache::kLineSize) {
      const auto got = fetch_at(icache, memory, kCodeBase + off);
      EXPECT_TRUE(std::equal(got.begin(), got.end(), expected.begin() + off))
          << "pass " << pass << " offset " << off;
    }
  }
}

// ---------------------------------------------------------------------------
// CU-level coherence: the points at which something actually invalidates the
// I$ during a run, exercised through the CU rather than by calling
// invalidate_all() directly.
// ---------------------------------------------------------------------------

// CDNA4 encodings. s_mov_b32 is SOP1 with sdst in [22:16], op in [15:8] and
// ssrc0 in [7:0]; inline constant N is encoded as 128 + N.
constexpr uint32_t kSNop = 0xBF800000u;
constexpr uint32_t kSEndpgm = 0xBF810000u;
constexpr uint32_t kSIcacheInv = 0xBF930000u;
constexpr uint32_t s_mov_b32_s0_imm(uint32_t imm) { return 0xBE800000u | (128u + imm); }

/// @brief One CDNA4 CU running a wave from a program at @ref kCodeBase.
///
/// @details Instructions are written straight into GPU memory and the wave is
/// advanced one at a time, so a test can rewrite code between two issues the
/// way self-modifying code or a debugger would.
class CuFixture {
public:
  explicit CuFixture(const std::string &name, uint32_t wf_slots = 1)
      : memory_(name + "_memory"), l2_(name + "_l2") {
    config_.arch = ROCJITSU_CODE_ARCH_CDNA4;
    config_.num_wf_slots = wf_slots;
    config_.sgprs_per_wf = 106;
    config_.vgprs_per_wf = 256;
    config_.lds_size_kb = 64;
    l2_.set_backing_memory(&memory_);
    cu_ = amdgpu::ComputeUnitCore::create(name, config_, &memory_, &l2_);
  }

  void write_program(std::span<const uint32_t> words, uint64_t base = kCodeBase) {
    for (size_t i = 0; i < words.size(); ++i)
      memory_.write32(base + i * sizeof(uint32_t), words[i]);
  }

  amdgpu::Wavefront *launch(uint32_t dispatch_id, uint32_t wg_id, uint64_t pc = kCodeBase) {
    cu_->begin_workgroup(dispatch_id, wg_id, 1);
    return cu_->dispatch_wf(wg_id, pc, config_.sgprs_per_wf, config_.vgprs_per_wf);
  }

  /// @brief Bytes the CU's I$ currently returns for @p pc, without refilling.
  std::array<uint8_t, InstructionCache::kFetchBytes> peek(uint64_t pc = kCodeBase) {
    std::array<uint8_t, InstructionCache::kFetchBytes> got{};
    cu_->instruction_cache().fetch(memory_, pc, 0, got.data());
    return got;
  }

  uint32_t read_s0(const amdgpu::Wavefront &wf) const {
    return cu_->read_sgpr(wf.sgpr_alloc().base);
  }

  amdgpu::ComputeUnitCore *cu() { return cu_.get(); }
  GpuMemory &memory() { return memory_; }

private:
  GpuMemory memory_;
  amdgpu::L2Cache l2_;
  amdgpu::ComputeUnitCore::Config config_{};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
};

TEST(InstructionCacheCuTest, UnmappingCodeStopsAWaveWithAWarmInstructionCache) {
  CuFixture fixture("unmap");
  rocjitsu::KfdProcess process(kMappedVmid);
  std::array<uint32_t, GpuMemory::PAGE_SIZE / sizeof(uint32_t)> code{};
  code[0] = s_mov_b32_s0_imm(1);
  code[1] = s_mov_b32_s0_imm(2);
  code[2] = kSEndpgm;
  process.map_pages(kCodeBase, code.data(), sizeof(code));
  register_fetchable_process(fixture.memory(), process);
  auto *wf = fixture.launch(1, 0);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(kMappedVmid);
  fixture.cu()->step();
  ASSERT_EQ(fixture.read_s0(*wf), 1u);
  ASSERT_FALSE(wf->is_halted());

  process.unmap_pages(kCodeBase, sizeof(code));
  fixture.cu()->step();
  // The cached s_mov would leave the wave running. Halting frees its registers,
  // so inspect the wave state instead of reading s0 after its allocation is freed.
  EXPECT_TRUE(wf->is_halted());
}

TEST(InstructionCacheCuTest, DebuggerWritesRemainVisibleWithAWarmMappedFetchabilityCache) {
  CuFixture fixture("mapped_debug");
  rocjitsu::KfdProcess process(kMappedVmid);
  std::array<uint32_t, GpuMemory::PAGE_SIZE / sizeof(uint32_t)> code{};
  code[0] = s_mov_b32_s0_imm(1);
  code[1] = s_mov_b32_s0_imm(2);
  code[2] = s_mov_b32_s0_imm(3);
  code[3] = kSEndpgm;
  process.map_pages(kCodeBase, code.data(), sizeof(code));
  register_fetchable_process(fixture.memory(), process);
  auto *wf = fixture.launch(1, 0);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(kMappedVmid);
  fixture.cu()->step();
  ASSERT_EQ(fixture.read_s0(*wf), 1u);

  fixture.cu()->set_debug_active(true);
  code[1] = s_mov_b32_s0_imm(22);
  fixture.cu()->step();
  EXPECT_EQ(fixture.read_s0(*wf), 22u);
  code[2] = s_mov_b32_s0_imm(33);
  fixture.cu()->set_debug_active(false);
  fixture.cu()->step();
  EXPECT_EQ(fixture.read_s0(*wf), 33u);
}

// Self-modifying code: the rewritten instruction only becomes visible to the
// fetcher when the wave retires s_icache_inv. This runs the generated
// execute_s_icache_inv_sopp body, which is the only thing standing between the
// architectural invalidation and a wave that keeps executing stale bytes.
TEST(InstructionCacheCuTest, SIcacheInvExecutedByAWaveExposesRewrittenCode) {
  for (const bool invalidate : {false, true}) {
    SCOPED_TRACE(invalidate ? "s_icache_inv" : "s_nop (control)");
    CuFixture fixture(invalidate ? "icache_inv_cu" : "icache_inv_control_cu");
    const std::array<uint32_t, 4> program = {
        kSNop,                            // 0x00: fills the line under the PC
        invalidate ? kSIcacheInv : kSNop, // 0x04
        s_mov_b32_s0_imm(1),              // 0x08: rewritten below, before it issues
        kSEndpgm,                         // 0x0c
    };
    fixture.write_program(program);

    auto *wf = fixture.launch(1, 0);
    ASSERT_NE(wf, nullptr);
    fixture.cu()->step();
    ASSERT_EQ(wf->pc, kCodeBase + 4) << "the first instruction did not retire";

    // The whole program is one 64-byte line, so it is already cached.
    fixture.memory().write32(kCodeBase + 8, s_mov_b32_s0_imm(2));

    fixture.cu()->step(); // s_icache_inv, or s_nop in the control
    fixture.cu()->step(); // the rewritten s_mov_b32
    EXPECT_EQ(fixture.read_s0(*wf), invalidate ? 2u : 1u)
        << "a rewritten instruction became visible " << (invalidate ? "too late" : "too early");
  }
}

// A debugger writes breakpoints straight into code memory, so the I$ has to be
// dropped when a session ends. It can attach, plant the breakpoint on a wave
// that is already stopped, and detach without that wave ever issuing -- so the
// invalidation cannot be driven from the issue path's own bypass.
TEST(InstructionCacheCuTest, DebugSessionInvalidatesEvenWithNoIssueWhileAttached) {
  CuFixture fixture("icache_debug_cu");
  const std::array<uint32_t, 3> program = {
      kSNop,               // 0x00
      s_mov_b32_s0_imm(1), // 0x04: the debugger overwrites this
      kSEndpgm,            // 0x08
  };
  fixture.write_program(program);

  auto *wf = fixture.launch(1, 0);
  ASSERT_NE(wf, nullptr);
  fixture.cu()->step();
  ASSERT_EQ(wf->pc, kCodeBase + 4);

  // Attach, write, detach -- with no instruction issued in between, which is
  // what leaves the CU thread nothing to notice unless the transition itself
  // published the invalidation.
  fixture.cu()->set_debug_active(true);
  fixture.memory().write32(kCodeBase + 4, s_mov_b32_s0_imm(2));
  fixture.cu()->set_debug_active(false);

  fixture.cu()->step();
  EXPECT_EQ(fixture.read_s0(*wf), 2u) << "the wave resumed on pre-attach code bytes";
}

// The launch invalidation belongs to the dispatch, not to each wave placed for
// it: sibling waves of one dispatch share the lines they fill, and only a new
// dispatch starts cold.
TEST(InstructionCacheCuTest, LaunchInvalidationIsOncePerDispatch) {
  CuFixture fixture("icache_dispatch_cu", /*wf_slots=*/4);
  fixture.write_program(std::array<uint32_t, 2>{kSNop, kSEndpgm});

  auto *first = fixture.launch(7, 0);
  ASSERT_NE(first, nullptr);
  fixture.cu()->step();
  const auto launched = fixture.peek();

  // Rewrite the code page. Nothing has issued s_icache_inv, so only a launch
  // invalidation can make this visible.
  const std::vector<uint8_t> rewritten =
      fill_code(fixture.memory(), InstructionCache::kLineSize, 0x6b);

  // Another workgroup of the same dispatch: no second invalidation, so the
  // lines its siblings filled are still there.
  ASSERT_NE(fixture.launch(7, 1), nullptr);
  const auto same_dispatch = fixture.peek();
  EXPECT_EQ(same_dispatch, launched) << "a sibling wave cold-started the I$";

  // A new dispatch may have loaded a different kernel at the same VA, so its
  // launch drops everything.
  ASSERT_NE(fixture.launch(8, 0), nullptr);
  const auto next_dispatch = fixture.peek();
  EXPECT_TRUE(std::equal(next_dispatch.begin(), next_dispatch.end(), rewritten.begin()))
      << "a new dispatch reused code bytes cached by the previous one";
}

} // namespace
