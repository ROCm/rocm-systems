// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/decoded_instruction_cache.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/instruction_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using rocjitsu::amdgpu::GpuMemory;
using rocjitsu::amdgpu::GpuVm;
using rocjitsu::amdgpu::InstructionCache;
namespace amdgpu = rocjitsu::amdgpu;

constexpr uint64_t kCodeBase = 0x200000;

/// @brief Fill @p bytes of code memory at kCodeBase with a per-byte pattern.
std::vector<uint8_t> fill_code(GpuMemory &memory, size_t bytes, uint8_t salt) {
  std::vector<uint8_t> expected(bytes);
  for (size_t i = 0; i < bytes; ++i)
    expected[i] = static_cast<uint8_t>((i * 7) ^ salt);
  memory.write_block(kCodeBase, std::span<const uint8_t>(expected));
  return expected;
}

std::array<uint8_t, InstructionCache::kFetchBytes> fetch_at(InstructionCache &icache,
                                                            const GpuMemory &memory, uint64_t pc) {
  std::array<uint8_t, InstructionCache::kFetchBytes> got{};
  icache.fetch(memory, pc, got.data());
  return got;
}

class ExecutableAddressSpace final : public amdgpu::AddressSpaceTranslator,
                                     public amdgpu::PhysicalMemoryAccess {
public:
  explicit ExecutableAddressSpace(uint8_t value, size_t size = InstructionCache::kLineSize * 2)
      : bytes_(size, static_cast<std::byte>(value)) {}

  void fill(uint8_t value) { std::ranges::fill(bytes_, static_cast<std::byte>(value)); }

  void write_program(std::span<const uint32_t> words) {
    ASSERT_LE(words.size_bytes(), bytes_.size());
    std::memcpy(bytes_.data(), words.data(), words.size_bytes());
  }

  amdgpu::VmTranslationResult translate(uint64_t address, std::size_t size,
                                        amdgpu::VmAccessKind access) const override {
    if (access != amdgpu::VmAccessKind::Read && access != amdgpu::VmAccessKind::Execute)
      return {.outcome = amdgpu::VmAccessOutcome::Faulted, .translation = {}};
    if (address < kCodeBase || size == 0 || address - kCodeBase > bytes_.size() ||
        size > bytes_.size() - (address - kCodeBase)) {
      return {.outcome = amdgpu::VmAccessOutcome::Faulted, .translation = {}};
    }
    return {
        .outcome = amdgpu::VmAccessOutcome::Complete,
        .translation = {.domain = amdgpu::VmMemoryDomain::System,
                        .address = address - kCodeBase,
                        .contiguous_bytes = bytes_.size() - (address - kCodeBase),
                        .mtype = amdgpu::Mtype::RW,
                        .permissions = {.readable = true, .writable = false, .executable = true}}};
  }

  amdgpu::VmAccessOutcome read(amdgpu::VmMemoryDomain domain, uint64_t address,
                               std::span<std::byte> bytes) override {
    if (domain != amdgpu::VmMemoryDomain::System || address > bytes_.size() ||
        bytes.size() > bytes_.size() - address) {
      return amdgpu::VmAccessOutcome::Faulted;
    }
    std::ranges::copy_n(bytes_.begin() + static_cast<ptrdiff_t>(address), bytes.size(),
                        bytes.begin());
    return amdgpu::VmAccessOutcome::Complete;
  }

  amdgpu::VmAccessOutcome write(amdgpu::VmMemoryDomain, uint64_t,
                                std::span<const std::byte>) override {
    return amdgpu::VmAccessOutcome::Faulted;
  }

private:
  std::vector<std::byte> bytes_;
};

std::array<uint8_t, InstructionCache::kFetchBytes>
fetch_at(InstructionCache &icache, const amdgpu::GpuVmAccess &access, uint64_t pc = kCodeBase) {
  std::array<uint8_t, InstructionCache::kFetchBytes> got{};
  EXPECT_EQ(icache.fetch(access, pc, got.data()), amdgpu::VmAccessOutcome::Complete);
  return got;
}

// Every four-byte-aligned PC in a two-line window, including the offsets whose
// fetch window runs off the end of a line, must return the backing bytes.
TEST(InstructionCacheTest, PeekOnlyReturnsPresentAlignedWordsForTheOwningVmid) {
  GpuMemory memory("peek_memory");
  InstructionCache cache;
  constexpr uint64_t pc = 0x4000;
  constexpr uint32_t value = 0x12345678;
  uint32_t word = 0;
  memory.write32(pc + 60, value);
  EXPECT_FALSE(cache.peek_word(pc + 60, 0, word));
  uint8_t fetched[InstructionCache::kFetchBytes];
  cache.fetch(memory, pc, fetched);
  ASSERT_TRUE(cache.peek_word(pc + 60, 0, word));
  EXPECT_EQ(word, value);
  EXPECT_FALSE(cache.peek_word(pc + 61, 0, word));
  EXPECT_FALSE(cache.peek_word(pc + 60, 1, word));
  EXPECT_FALSE(cache.peek_word(pc + InstructionCache::kCacheBytes + 60, 0, word));
  cache.invalidate_all();
  EXPECT_FALSE(cache.peek_word(pc + 60, 0, word));
}

TEST(InstructionCacheTest, FetchMatchesBackingMemoryAtEveryAlignedOffset) {
  GpuMemory memory("memory");
  InstructionCache icache;
  const size_t span = InstructionCache::kLineSize * 3;
  const std::vector<uint8_t> expected = fill_code(memory, span, 0x5a);

  for (uint32_t off = 0; off + InstructionCache::kFetchBytes <= span; off += 4) {
    const auto got = fetch_at(icache, memory, kCodeBase + off);
    EXPECT_TRUE(std::ranges::equal(got, std::span(expected).subspan(off, got.size())))
        << "mismatch at offset " << off;
  }
}

TEST(InstructionCacheTest, RepeatedMaintenanceAdvancesEpochAndInvalidatesRefills) {
  GpuMemory memory("memory");
  InstructionCache cache;
  uint64_t epoch = cache.epoch();
  const std::vector<uint8_t> first = fill_code(memory, InstructionCache::kCacheBytes, 0x31);
  for (unsigned cycle = 0; cycle != 3; ++cycle) {
    for (unsigned repeat = 0; repeat != 4; ++repeat) {
      cache.invalidate_all();
      EXPECT_EQ(cache.epoch(), ++epoch);
      uint32_t word = 0;
      EXPECT_FALSE(cache.peek_word(kCodeBase, 0, word));
    }
    for (uint32_t offset = 0; offset != InstructionCache::kCacheBytes;
         offset += InstructionCache::kLineSize) {
      const auto bytes = fetch_at(cache, memory, kCodeBase + offset);
      EXPECT_TRUE(std::ranges::equal(bytes, std::span(first).subspan(offset, bytes.size())));
    }
  }
  const std::vector<uint8_t> second = fill_code(memory, InstructionCache::kCacheBytes, 0x72);
  cache.invalidate_all();
  EXPECT_EQ(cache.epoch(), ++epoch);
  for (uint32_t offset = 0; offset != InstructionCache::kCacheBytes;
       offset += InstructionCache::kLineSize) {
    const auto bytes = fetch_at(cache, memory, kCodeBase + offset);
    EXPECT_TRUE(std::ranges::equal(bytes, std::span(second).subspan(offset, bytes.size())));
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
  EXPECT_TRUE(std::ranges::equal(got, std::span(expected).subspan(kStraddle, got.size())));
}

// The I$ is deliberately not coherent with data writes, matching hardware: a
// write to code memory is invisible until something issues s_icache_inv.
TEST(InstructionCacheTest, CachedLineSurvivesABackingWriteUntilInvalidated) {
  GpuMemory memory("memory");
  InstructionCache icache;
  const std::vector<uint8_t> first = fill_code(memory, InstructionCache::kLineSize, 0x11);

  const auto before = fetch_at(icache, memory, kCodeBase);
  EXPECT_TRUE(std::ranges::equal(before, std::span(first).first(before.size())));

  const std::vector<uint8_t> second = fill_code(memory, InstructionCache::kLineSize, 0x22);
  ASSERT_NE(first, second);

  const auto stale = fetch_at(icache, memory, kCodeBase);
  EXPECT_TRUE(std::ranges::equal(stale, std::span(first).first(stale.size())))
      << "the I$ must not observe a data write on its own";

  icache.invalidate_all();
  const auto after = fetch_at(icache, memory, kCodeBase);
  EXPECT_TRUE(std::ranges::equal(after, std::span(second).first(after.size())));
}

TEST(InstructionCacheTest, DeviceMaintenanceInvalidatesLazilyOnOwningThread) {
  GpuMemory memory("memory");
  InstructionCache instruction_cache;
  const std::vector<uint8_t> first = fill_code(memory, InstructionCache::kLineSize, 0x31);
  const std::array<uint8_t, InstructionCache::kFetchBytes> before =
      fetch_at(instruction_cache, memory, kCodeBase);
  ASSERT_TRUE(std::ranges::equal(before, std::span(first).first(before.size())));

  const uint64_t epoch_before = instruction_cache.coherence_domain()->current_instruction_epoch();
  std::vector<uint8_t> second(InstructionCache::kLineSize, 0x72);
  {
    [[maybe_unused]] amdgpu::DeviceCacheMaintenanceLease maintenance =
        instruction_cache.coherence_domain()->acquire_cache_maintenance(
            amdgpu::DeviceCacheOperation::WritebackInvalidate);
    EXPECT_EQ(instruction_cache.coherence_domain()->current_instruction_epoch(), epoch_before);
    memory.write_block(kCodeBase, std::span<const uint8_t>(second));
  }
  EXPECT_GT(instruction_cache.coherence_domain()->current_instruction_epoch(), epoch_before);

  const std::array<uint8_t, InstructionCache::kFetchBytes> after =
      fetch_at(instruction_cache, memory, kCodeBase);
  EXPECT_TRUE(std::ranges::equal(after, std::span(second).first(after.size())));
}

// Lines are tagged by address-space identity, so the same virtual address in
// two address spaces must not alias even though it selects the same line.
TEST(InstructionCacheTest, LinesDoNotAliasAcrossVmids) {
  GpuVm gpu_vm;
  InstructionCache icache;
  auto vm0 = std::make_shared<ExecutableAddressSpace>(0x01);
  auto vm1 = std::make_shared<ExecutableAddressSpace>(0x02);
  const auto handle0 = gpu_vm.register_translated(0, vm0, vm0);
  const auto handle1 = gpu_vm.register_translated(1, vm1, vm1);
  ASSERT_TRUE(handle0);
  ASSERT_TRUE(handle1);
  const auto access0 = gpu_vm.snapshot(handle0);
  const auto access1 = gpu_vm.snapshot(handle1);
  ASSERT_TRUE(access0);
  ASSERT_TRUE(access1);

  const auto got0 = fetch_at(icache, *access0);
  EXPECT_TRUE(std::ranges::all_of(got0, [](uint8_t byte) { return byte == 0x01; }));
  uint32_t legacy_word = 0;
  EXPECT_FALSE(icache.peek_word(kCodeBase, 0, legacy_word))
      << "translated code must not satisfy legacy-address-space lookahead";

  const auto got1 = fetch_at(icache, *access1);
  EXPECT_TRUE(std::ranges::all_of(got1, [](uint8_t byte) { return byte == 0x02; }))
      << "the vmid 1 lookup returned vmid 0's line";

  const auto again1 = fetch_at(icache, *access1);
  EXPECT_EQ(again1, got1) << "the vmid 1 fetch did not cache its line";
}

TEST(InstructionCacheTest, RepeatedMaintenanceInvalidatesTranslatedRefills) {
  GpuVm gpu_vm;
  InstructionCache cache;
  auto backing = std::make_shared<ExecutableAddressSpace>(0x11);
  const auto handle = gpu_vm.register_translated(7, backing, backing);
  ASSERT_TRUE(handle);
  const auto access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(access);
  auto cached = fetch_at(cache, *access);
  for (uint8_t value : {0x22, 0x33, 0x44}) {
    backing->fill(value);
    EXPECT_EQ(fetch_at(cache, *access), cached);
    const uint64_t epoch = cache.epoch();
    cache.invalidate_all();
    cache.invalidate_all();
    EXPECT_EQ(cache.epoch(), epoch + 2);
    cached = fetch_at(cache, *access);
    EXPECT_TRUE(std::ranges::all_of(cached, [value](uint8_t byte) { return byte == value; }));
  }
}

TEST(InstructionCacheTest, TranslatedLinesDoNotAliasAcrossRootReplacement) {
  GpuVm gpu_vm;
  InstructionCache icache;
  auto first = std::make_shared<ExecutableAddressSpace>(0x11);
  const amdgpu::AddressSpaceHandle handle = gpu_vm.register_translated(7, first, first);
  ASSERT_TRUE(handle);
  const std::optional<amdgpu::GpuVmAccess> first_access = gpu_vm.snapshot_pinned(handle);
  ASSERT_TRUE(first_access);

  const auto first_fetch = fetch_at(icache, *first_access);
  EXPECT_TRUE(std::ranges::all_of(first_fetch, [](uint8_t byte) { return byte == 0x11; }));

  auto replacement = std::make_shared<ExecutableAddressSpace>(0x22);
  ASSERT_TRUE(gpu_vm.replace_translated(handle, replacement, replacement));
  const std::optional<amdgpu::GpuVmAccess> replacement_access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(replacement_access);
  EXPECT_NE(first_access->cache_namespace(), replacement_access->cache_namespace());

  const auto replacement_fetch = fetch_at(icache, *replacement_access);
  EXPECT_TRUE(std::ranges::all_of(replacement_fetch, [](uint8_t byte) { return byte == 0x22; }));
  const auto retained_old_fetch = fetch_at(icache, *first_access);
  EXPECT_TRUE(std::ranges::all_of(retained_old_fetch, [](uint8_t byte) { return byte == 0x11; }));
}

TEST(InstructionCacheTest, FetchBypassesAnIncompleteCacheLine) {
  GpuVm gpu_vm;
  InstructionCache icache;
  auto address_space =
      std::make_shared<ExecutableAddressSpace>(0x11, InstructionCache::kFetchBytes);
  const amdgpu::AddressSpaceHandle handle =
      gpu_vm.register_translated(7, address_space, address_space);
  ASSERT_TRUE(handle);
  const std::optional<amdgpu::GpuVmAccess> access = gpu_vm.snapshot(handle);
  ASSERT_TRUE(access);

  const auto first = fetch_at(icache, *access);
  EXPECT_TRUE(std::ranges::all_of(first, [](uint8_t byte) { return byte == 0x11; }));

  address_space->fill(0x22);
  const auto second = fetch_at(icache, *access);
  EXPECT_TRUE(std::ranges::all_of(second, [](uint8_t byte) { return byte == 0x22; }))
      << "a partial executable extent must not leave a cached full line";

  std::array<uint8_t, InstructionCache::kFetchBytes> invalid{};
  std::ranges::fill(invalid, uint8_t{0xcc});
  EXPECT_EQ(icache.fetch(*access, kCodeBase + sizeof(uint32_t), invalid.data()),
            amdgpu::VmAccessOutcome::Faulted);
  EXPECT_TRUE(std::ranges::all_of(invalid, [](uint8_t byte) { return byte == 0xcc; }));
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
      EXPECT_TRUE(std::ranges::equal(got, std::span(expected).subspan(off, got.size())))
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
    cu_->instruction_cache().fetch(memory_, pc, got.data());
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

class CountingDecoder : public rocjitsu::Decoder {
public:
  explicit CountingDecoder(rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA4)
      : decoder_(rocjitsu::Decoder::create(arch)) {}

  rocjitsu::DecodeResult decode(const rj_code_binary_inst_t *words,
                                const rocjitsu::DecodeErrorEmitter &emit_error) override {
    ++calls;
    return decoder_->decode(words, emit_error);
  }
  size_t max_instruction_words() const override { return decoder_->max_instruction_words(); }

  unsigned calls = 0;

private:
  std::unique_ptr<rocjitsu::Decoder> decoder_;
};

TEST(DecodedInstructionCacheCuTest, ConstructionAndDecoderReplacementPreserveForeignPool) {
  rocjitsu::Instruction::ScopedHeapAllocation restore_allocator;
  CountingDecoder foreign_decoder;
  foreign_decoder.enable_pool();
  const auto alloc_fn = rocjitsu::Instruction::alloc_fn_;
  const auto dealloc_fn = rocjitsu::Instruction::dealloc_fn_;
  const auto pool = rocjitsu::Instruction::alloc_pool_;
  ASSERT_NE(pool, nullptr);
  const auto expect_foreign_pool = [&] {
    EXPECT_EQ(rocjitsu::Instruction::alloc_fn_, alloc_fn);
    EXPECT_EQ(rocjitsu::Instruction::dealloc_fn_, dealloc_fn);
    EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, pool);
  };
  {
    CuFixture fixture("decoded_foreign_pool");
    expect_foreign_pool();
    fixture.cu()->replace_decoder_for_test(std::make_unique<CountingDecoder>());
    expect_foreign_pool();
    fixture.write_program(std::array<uint32_t, 2>{s_mov_b32_s0_imm(17), kSEndpgm});
    auto *wf = fixture.launch(1, 0);
    ASSERT_NE(wf, nullptr);
    fixture.cu()->step();
    EXPECT_EQ(fixture.read_s0(*wf), 17u);
    expect_foreign_pool();
    fixture.cu()->step();
    expect_foreign_pool();
  }
  expect_foreign_pool();
}

TEST(DecodedInstructionCacheCuTest, LoopDecodesOnceAndReadsCurrentRegisters) {
  CuFixture fixture("decoded_loop");
  auto decoder = std::make_unique<CountingDecoder>();
  auto *counter = decoder.get();
  fixture.cu()->replace_decoder_for_test(std::move(decoder));
  // s_mov_b32 s0, s1; s_branch -2. The branch revisits both decoded objects.
  fixture.write_program(std::array<uint32_t, 2>{0xBE800001u, 0xBF82FFFEu});
  auto *wf = fixture.launch(1, 0);
  ASSERT_NE(wf, nullptr);
  for (uint32_t value = 1; value <= 10; ++value) {
    fixture.cu()->write_sgpr(wf->sgpr_alloc().base + 1, value * 19);
    fixture.cu()->step();
    EXPECT_EQ(fixture.read_s0(*wf), value * 19);
    fixture.cu()->step();
    ASSERT_EQ(wf->pc, kCodeBase);
  }
  EXPECT_EQ(counter->calls, 2u);
}

TEST(DecodedInstructionCacheCuTest, RewrittenLiteralAndDebuggerBypassUseFreshDecodes) {
  CuFixture fixture("decoded_literal");
  auto decoder = std::make_unique<CountingDecoder>();
  auto *counter = decoder.get();
  fixture.cu()->replace_decoder_for_test(std::move(decoder));
  fixture.write_program(std::array<uint32_t, 3>{0xBE8000FFu, 17u, kSEndpgm});
  auto *wf = fixture.launch(1, 0);
  ASSERT_NE(wf, nullptr);
  fixture.cu()->step();
  EXPECT_EQ(fixture.read_s0(*wf), 17u);

  fixture.memory().write32(kCodeBase + 4, 29u);
  wf->pc = kCodeBase;
  fixture.cu()->step();
  EXPECT_EQ(fixture.read_s0(*wf), 17u);
  EXPECT_EQ(counter->calls, 1u);

  fixture.cu()->instruction_cache().invalidate_all();
  wf->pc = kCodeBase;
  fixture.cu()->step();
  EXPECT_EQ(fixture.read_s0(*wf), 29u);
  EXPECT_EQ(counter->calls, 2u);

  fixture.cu()->set_debug_active(true);
  fixture.memory().write32(kCodeBase + 4, 43u);
  wf->pc = kCodeBase;
  fixture.cu()->step();
  EXPECT_EQ(fixture.read_s0(*wf), 43u);
  fixture.cu()->set_debug_active(false);
  wf->pc = kCodeBase;
  fixture.cu()->step();
  EXPECT_EQ(fixture.read_s0(*wf), 43u);
  EXPECT_EQ(counter->calls, 4u);
}

TEST(DecodedInstructionCacheCuTest, VectorReadsCurrentExecAcrossDispatches) {
  CuFixture fixture("decoded_vector", 2);
  auto decoder = std::make_unique<CountingDecoder>();
  auto *counter = decoder.get();
  fixture.cu()->replace_decoder_for_test(std::move(decoder));
  // v_mov_b32_e32 v0, v1; s_branch -2.
  fixture.write_program(std::array<uint32_t, 2>{0x7E000301u, 0xBF82FFFEu});
  for (unsigned dispatch = 1; dispatch <= 3; ++dispatch) {
    auto *wf = fixture.launch(dispatch, dispatch);
    ASSERT_NE(wf, nullptr);
    const auto base = wf->vgpr_alloc().base;
    for (uint64_t mask : {~uint64_t{0}, uint64_t{0}, uint64_t{0x3333333333333333}}) {
      wf->set_exec(mask);
      for (unsigned lane = 0; lane < wf->wf_size(); ++lane) {
        fixture.cu()->write_vgpr(base, lane, 0xDEAD0000u + lane);
        fixture.cu()->write_vgpr(base + 1, lane, dispatch * 1000 + lane);
      }
      fixture.cu()->step();
      ASSERT_EQ(wf->pc, kCodeBase + 4);
      for (unsigned lane = 0; lane < wf->wf_size(); ++lane)
        EXPECT_EQ(fixture.cu()->read_vgpr(base, lane),
                  ((mask >> lane) & 1) ? dispatch * 1000 + lane : 0xDEAD0000u + lane);
      fixture.cu()->step();
      ASSERT_EQ(wf->pc, kCodeBase);
    }
    wf->halt();
  }
  EXPECT_EQ(counter->calls, 2u);
}

TEST(DecodedInstructionCacheCuTest, DppDelegatesReadCurrentRegistersAndRestoreAcrossWaves) {
  CuFixture fixture("decoded_dpp", 2);
  auto *wf0 = fixture.launch(1, 0);
  auto *wf1 = fixture.launch(1, 1);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  ASSERT_NE(wf0->vgpr_alloc().base, wf1->vgpr_alloc().base);
  CountingDecoder decoder;
  amdgpu::DecodedInstructionCache cache;
  // v_mov_b32 v0, v1 with DPP quad permutation [1, 0, 3, 2].
  rocjitsu::cdna4::Vop1VopDppMachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.op = rocjitsu::cdna4::kVMovB32Vop1;
  raw.encoding = rocjitsu::cdna4::encoding::kVop1 >> 2;
  raw.vdst = 0;
  raw.vsrc0 = 1;
  raw.dpp_ctrl = 0xB1;
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;
  std::array<uint32_t, 4> words{};
  std::memcpy(words.data(), &raw, sizeof(raw));
  const rocjitsu::Instruction *identity = nullptr;
  for (unsigned repeat = 1; repeat <= 4; ++repeat)
    for (auto *wf : {wf0, wf1})
      for (uint64_t mask : {~uint64_t{0}, uint64_t{0}, uint64_t{0x3333333333333333}}) {
        const auto base = wf->vgpr_alloc().base;
        const auto value = repeat * 1000 + base;
        wf->set_exec(mask);
        for (unsigned lane = 0; lane < wf->wf_size(); ++lane) {
          fixture.cu()->write_vgpr(base, lane, 0xDEAD0000u + lane);
          fixture.cu()->write_vgpr(base + 1, lane, value + lane);
        }
        auto decoded = cache.decode(decoder, kCodeBase, 7, words, {});
        ASSERT_FALSE(decoded.failed());
        if (!identity)
          identity = decoded.value().get();
        EXPECT_EQ(decoded.value().get(), identity);
        amdgpu::DecodedInstructionCache::ScopedReturn returned(cache, decoded.value(), kCodeBase, 7,
                                                               words);
        ASSERT_TRUE(fixture.cu()->execute_instruction(decoded.value().get(), *wf).succeeded());
        EXPECT_EQ(decoded.value()->src_operand(0)->delegate(), nullptr);
        for (unsigned lane = 0; lane < wf->wf_size(); ++lane)
          EXPECT_EQ(fixture.cu()->read_vgpr(base, lane),
                    ((mask >> lane) & 1) ? value + (lane ^ 1u) : 0xDEAD0000u + lane);
      }
  EXPECT_EQ(decoder.calls, 1u);
}

TEST(DecodedInstructionCacheTest, InFlightOwnershipPreventsReuse) {
  CountingDecoder decoder;
  amdgpu::DecodedInstructionCache cache;
  std::array<uint32_t, 4> words{kSNop};
  std::unique_ptr<rocjitsu::Instruction> in_flight;
  {
    auto decoded = cache.decode(decoder, kCodeBase, 7, words, {});
    ASSERT_FALSE(decoded.failed());
    amdgpu::DecodedInstructionCache::ScopedReturn returned(cache, decoded.value(), kCodeBase, 7,
                                                           words);
    in_flight = std::move(decoded.value());
  }
  auto decoded = cache.decode(decoder, kCodeBase, 7, words, {});
  ASSERT_FALSE(decoded.failed());
  EXPECT_NE(decoded.value().get(), in_flight.get());
  EXPECT_EQ(decoder.calls, 2u);
}

TEST(DecodedInstructionCacheTest, FailedExecutionAndExceptionsDoNotRetainInstructions) {
  CountingDecoder decoder;
  amdgpu::DecodedInstructionCache cache;
  std::array<uint32_t, 4> words{kSNop};
  {
    auto decoded = cache.decode(decoder, kCodeBase, 7, words, {});
    ASSERT_FALSE(decoded.failed());
    amdgpu::DecodedInstructionCache::ScopedReturn returned(cache, decoded.value(), kCodeBase, 7,
                                                           words);
    returned.discard();
  }
  EXPECT_EQ(decoder.calls, 1u);
  try {
    auto decoded = cache.decode(decoder, kCodeBase, 7, words, {});
    ASSERT_FALSE(decoded.failed());
    amdgpu::DecodedInstructionCache::ScopedReturn returned(cache, decoded.value(), kCodeBase, 7,
                                                           words);
    throw std::runtime_error("execution callback failed");
  } catch (const std::runtime_error &) {
  }
  auto decoded = cache.decode(decoder, kCodeBase, 7, words, {});
  ASSERT_FALSE(decoded.failed());
  EXPECT_EQ(decoder.calls, 3u);
}

TEST(DecodedInstructionCacheTest, BorrowedVopdEncodingIsNotRetainedAcrossIssues) {
  for (const auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                          ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(arch);
    CountingDecoder decoder(arch);
    amdgpu::DecodedInstructionCache cache;
    // v_dual_cndmask_b32 v2, v1, v2 :: v_dual_mov_b32 v1, 0.
    std::array<uint32_t, 4> words{0xCA500501u, 0x02000080u};
    {
      auto decoded = cache.decode(decoder, kCodeBase, 7, words, {});
      ASSERT_FALSE(decoded.failed());
      ASSERT_EQ(decoded.value()->raw_encoding(), words.data());
      amdgpu::DecodedInstructionCache::ScopedReturn returned(cache, decoded.value(), kCodeBase, 7,
                                                             words);
    }
    // Use the same bytes from a different fetch buffer on the next issue.
    auto next_words = words;
    auto decoded = cache.decode(decoder, kCodeBase, 7, next_words, {});
    ASSERT_FALSE(decoded.failed());
    EXPECT_EQ(decoded.value()->raw_encoding(), next_words.data());
    EXPECT_EQ(decoder.calls, 2u);
  }
}

TEST(DecodedInstructionCacheTest, AddressSpacesCollisionsAndClearCauseMisses) {
  CountingDecoder decoder;
  amdgpu::DecodedInstructionCache cache;
  std::array<uint32_t, 4> words{kSNop};
  auto issue = [&](uint64_t pc, uint32_t vmid) {
    auto decoded = cache.decode(decoder, pc, vmid, words, {});
    ASSERT_FALSE(decoded.failed());
    amdgpu::DecodedInstructionCache::ScopedReturn returned(cache, decoded.value(), pc, vmid, words);
  };
  issue(kCodeBase, 7);
  issue(kCodeBase, 7);
  EXPECT_EQ(decoder.calls, 1u);
  issue(kCodeBase, 8);
  issue(kCodeBase, 7);
  EXPECT_EQ(decoder.calls, 3u);
  issue(kCodeBase + 4 * amdgpu::DecodedInstructionCache::kNumEntries, 7);
  issue(kCodeBase, 7);
  EXPECT_EQ(decoder.calls, 5u);
  cache.clear();
  issue(kCodeBase, 7);
  EXPECT_EQ(decoder.calls, 6u);
}

TEST(DecodedInstructionCacheTest, DynamicStateIsNotReusedAndPoolLifetimeIsIndependent) {
  amdgpu::DecodedInstructionCache cache;
  std::array<uint32_t, 4> words{kSNop};
  {
    CountingDecoder decoder;
    decoder.enable_pool();
    {
      auto decoded = cache.decode(decoder, kCodeBase, 7, words, {});
      ASSERT_FALSE(decoded.failed());
      amdgpu::DecodedInstructionCache::ScopedReturn returned(cache, decoded.value(), kCodeBase, 7,
                                                             words);
      decoded.value()->set_data(std::make_unique<rocjitsu::DynamicInstState>());
    }
    auto decoded = cache.decode(decoder, kCodeBase, 7, words, {});
    ASSERT_FALSE(decoded.failed());
    EXPECT_EQ(decoder.calls, 2u);
    EXPECT_EQ(decoded.value()->data(), nullptr);
    amdgpu::DecodedInstructionCache::ScopedReturn returned(cache, decoded.value(), kCodeBase, 7,
                                                           words);
  }
  // The decoder and its active allocator pool are gone before the cached
  // instruction is destroyed. Cache entries must have independent storage.
  cache.clear();
}

TEST(DecodedInstructionCacheTest, UncachedInstructionsAreHeapBackedWithAnActivePool) {
  rocjitsu::Instruction::ScopedHeapAllocation restore_allocator;
  CountingDecoder decoder;
  decoder.enable_pool();
  const auto pool = static_cast<rocjitsu::Decoder::Pool *>(rocjitsu::Instruction::alloc_pool_);
  ASSERT_NE(pool, nullptr);
  amdgpu::DecodedInstructionCache cache;
  std::array<uint32_t, 4> words{kSNop};
  for (unsigned mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    std::unique_ptr<rocjitsu::Instruction> in_flight;
    {
      const bool enabled = mode != 2;
      auto decoded = cache.decode(decoder, kCodeBase, 7, words, {}, enabled);
      ASSERT_FALSE(decoded.failed());
      EXPECT_FALSE(pool->owns(decoded.value().get()));
      EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, pool);
      amdgpu::DecodedInstructionCache::ScopedReturn returned(cache, decoded.value(), kCodeBase, 7,
                                                             words, enabled);
      if (mode == 0)
        returned.discard();
      else if (mode == 1)
        decoded.value()->set_data(std::make_unique<rocjitsu::DynamicInstState>());
      else if (mode == 3)
        in_flight = std::move(decoded.value());
    }
    // ArenaAlloc's non-owned-pointer fallback is the supported heap deletion
    // path even when an unrelated decoder's pool is ambient on this thread.
    in_flight.reset();
    EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, pool);
  }
  EXPECT_EQ(decoder.calls, 4u);
}

TEST(DecodedInstructionCacheTest, EntriesOutliveMultipleWorkerPools) {
  amdgpu::DecodedInstructionCache cache;
  std::array<uint32_t, 4> words{kSNop};
  const rocjitsu::Instruction *identity = nullptr;
  for (unsigned worker = 0; worker < 4; ++worker) {
    std::thread thread([&] {
      CountingDecoder decoder;
      decoder.enable_pool();
      auto decoded = cache.decode(decoder, kCodeBase, 7, words, {});
      ASSERT_FALSE(decoded.failed());
      EXPECT_FALSE(static_cast<rocjitsu::Decoder::Pool *>(rocjitsu::Instruction::alloc_pool_)
                       ->owns(decoded.value().get()));
      if (!identity)
        identity = decoded.value().get();
      EXPECT_EQ(decoded.value().get(), identity);
      EXPECT_EQ(decoder.calls, worker == 0 ? 1u : 0u);
      amdgpu::DecodedInstructionCache::ScopedReturn returned(cache, decoded.value(), kCodeBase, 7,
                                                             words);
    });
    thread.join();
  }
  cache.clear();
}

// A warm instruction-cache line must not hide revocation of the VM snapshot
// retained by the CU. Exercise both explicit handles and the VMID fallback.
TEST(InstructionCacheCuTest, VmInvalidationRefreshesAnAlreadyCachedInstruction) {
  for (const bool explicit_handle : {false, true}) {
    SCOPED_TRACE(explicit_handle);
    GpuVm gpu_vm;
    CuFixture fixture("vm_invalidation_cu");
    auto backing = std::make_shared<ExecutableAddressSpace>(0);
    backing->write_program(std::array<uint32_t, 3>{kSNop, s_mov_b32_s0_imm(1), kSEndpgm});
    const auto handle = gpu_vm.register_translated(7, backing, backing);
    ASSERT_TRUE(handle);
    fixture.cu()->set_gpu_vm(&gpu_vm);
    auto *wf = fixture.launch(1, 0);
    ASSERT_NE(wf, nullptr);
    wf->set_process_id(7);
    if (explicit_handle)
      wf->set_address_space(handle);
    fixture.cu()->step();
    ASSERT_EQ(wf->pc, kCodeBase + 4);

    backing->write_program(std::array<uint32_t, 3>{kSNop, s_mov_b32_s0_imm(2), kSEndpgm});
    ASSERT_TRUE(gpu_vm.invalidate(handle));
    fixture.cu()->step();
    EXPECT_EQ(fixture.read_s0(*wf), 2u);
  }
}

TEST(InstructionCacheCuTest, RootReplacementRefreshesAnAlreadyCachedInstruction) {
  GpuVm gpu_vm;
  CuFixture fixture("vm_replacement_cu");
  auto first = std::make_shared<ExecutableAddressSpace>(0);
  first->write_program(std::array<uint32_t, 3>{kSNop, s_mov_b32_s0_imm(1), kSEndpgm});
  const auto handle = gpu_vm.register_translated(7, first, first);
  ASSERT_TRUE(handle);
  fixture.cu()->set_gpu_vm(&gpu_vm);
  auto *wf = fixture.launch(1, 0);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(7);
  wf->set_address_space(handle);
  fixture.cu()->step();
  ASSERT_EQ(wf->pc, kCodeBase + 4);

  auto replacement = std::make_shared<ExecutableAddressSpace>(0);
  replacement->write_program(std::array<uint32_t, 3>{kSNop, s_mov_b32_s0_imm(2), kSEndpgm});
  ASSERT_TRUE(gpu_vm.replace_translated(handle, replacement, replacement));
  fixture.cu()->step();
  EXPECT_EQ(fixture.read_s0(*wf), 2u);
}

TEST(InstructionCacheCuTest, ReusedVmidCannotReviveAStaleExplicitHandle) {
  GpuVm gpu_vm;
  CuFixture fixture("vm_stale_handle_cu");
  auto backing = std::make_shared<ExecutableAddressSpace>(0);
  backing->write_program(std::array<uint32_t, 3>{kSNop, s_mov_b32_s0_imm(1), kSEndpgm});
  const auto handle = gpu_vm.register_translated(7, backing, backing);
  ASSERT_TRUE(handle);
  fixture.cu()->set_gpu_vm(&gpu_vm);
  auto *wf = fixture.launch(1, 0);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(7);
  wf->set_address_space(handle);
  fixture.cu()->step();
  ASSERT_EQ(wf->pc, kCodeBase + 4);

  ASSERT_TRUE(gpu_vm.unregister_address_space(handle));
  const auto replacement = gpu_vm.register_translated(7, backing, backing);
  ASSERT_TRUE(replacement);
  EXPECT_NE(handle, replacement);
  fixture.cu()->step();
  EXPECT_TRUE(wf->is_halted());
  EXPECT_FALSE(fixture.cu()->has_active_wfs())
      << "the stale address space must abort the dispatch before executing cached code";
}

TEST(InstructionCacheCuTest, ChangingAddressSpaceSelectsItsOwnInstruction) {
  for (const bool explicit_handle : {false, true}) {
    SCOPED_TRACE(explicit_handle);
    GpuVm gpu_vm;
    CuFixture fixture("vm_switch_cu");
    auto first = std::make_shared<ExecutableAddressSpace>(0);
    first->write_program(std::array<uint32_t, 3>{kSNop, s_mov_b32_s0_imm(1), kSEndpgm});
    const auto first_handle = gpu_vm.register_translated(7, first, first);
    auto second = std::make_shared<ExecutableAddressSpace>(0);
    second->write_program(std::array<uint32_t, 3>{kSNop, s_mov_b32_s0_imm(2), kSEndpgm});
    const auto second_handle = gpu_vm.register_translated(8, second, second);
    ASSERT_TRUE(first_handle);
    ASSERT_TRUE(second_handle);
    fixture.cu()->set_gpu_vm(&gpu_vm);
    auto *wf = fixture.launch(1, 0);
    ASSERT_NE(wf, nullptr);
    wf->set_process_id(7);
    if (explicit_handle)
      wf->set_address_space(first_handle);
    fixture.cu()->step();
    ASSERT_EQ(wf->pc, kCodeBase + 4);

    wf->set_process_id(8);
    if (explicit_handle)
      wf->set_address_space(second_handle);
    fixture.cu()->step();
    EXPECT_EQ(fixture.read_s0(*wf), 2u);
  }
}

TEST(InstructionCacheCuTest, SwitchingVmServicesDropsTheRetainedAccess) {
  GpuVm first_vm;
  GpuVm second_vm;
  CuFixture fixture("vm_service_switch_cu");
  auto first = std::make_shared<ExecutableAddressSpace>(0);
  first->write_program(std::array<uint32_t, 3>{kSNop, s_mov_b32_s0_imm(1), kSEndpgm});
  const auto first_handle = first_vm.register_translated(7, first, first);
  auto second = std::make_shared<ExecutableAddressSpace>(0);
  second->write_program(std::array<uint32_t, 3>{kSNop, s_mov_b32_s0_imm(2), kSEndpgm});
  const auto second_handle = second_vm.register_translated(7, second, second);
  ASSERT_TRUE(first_handle);
  ASSERT_EQ(first_handle, second_handle) << "exercise matching keys in different VM services";
  fixture.cu()->set_gpu_vm(&first_vm);
  auto *wf = fixture.launch(1, 0);
  ASSERT_NE(wf, nullptr);
  wf->set_process_id(7);
  wf->set_address_space(first_handle);
  fixture.cu()->step();
  ASSERT_EQ(wf->pc, kCodeBase + 4);

  fixture.cu()->set_gpu_vm(&second_vm);
  fixture.cu()->step();
  EXPECT_EQ(fixture.read_s0(*wf), 2u);
}

struct CountedReporter {
  std::shared_ptr<rocjitsu::KfdProcess> process;
  unsigned *copies;

  CountedReporter(std::shared_ptr<rocjitsu::KfdProcess> owner, unsigned &count)
      : process(std::move(owner)), copies(&count) {}
  CountedReporter(const CountedReporter &other) : process(other.process), copies(other.copies) {
    ++*copies;
  }
  void operator()(uint64_t, amdgpu::VmAccessKind) const {}
};

// Capture a real process as the KFD binding does. Releasing the registry binding
// must not leave that process owned by an idle CU.
TEST(InstructionCacheCuTest, CompletedWaveDoesNotRetainItsVmOwner) {
  GpuVm gpu_vm;
  CuFixture fixture("vm_completed_lifetime_cu");
  auto backing = std::make_shared<ExecutableAddressSpace>(0);
  backing->write_program(std::array<uint32_t, 2>{kSNop, kSEndpgm});
  auto owner = std::make_shared<rocjitsu::KfdProcess>(7);
  const std::weak_ptr<rocjitsu::KfdProcess> lifetime = owner;
  const auto handle = gpu_vm.register_address_space(
      7, backing, backing, [owner](uint64_t, amdgpu::VmAccessKind) { (void)owner; });
  ASSERT_TRUE(handle);
  owner.reset();
  fixture.cu()->set_gpu_vm(&gpu_vm);
  auto *wf = fixture.launch(1, 0);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(1);
  wf->set_process_id(7);
  wf->set_address_space(handle);
  fixture.cu()->step();
  ASSERT_EQ(wf->pc, kCodeBase + 4);
  fixture.cu()->step();
  ASSERT_TRUE(wf->is_halted());
  ASSERT_FALSE(fixture.cu()->has_active_wfs());

  ASSERT_TRUE(gpu_vm.unregister_address_space(handle));
  EXPECT_TRUE(lifetime.expired()) << "an idle CU retained the completed process binding";
}

TEST(InstructionCacheCuTest, AbortedWaveDoesNotRetainItsVmOwner) {
  GpuVm gpu_vm;
  CuFixture fixture("vm_aborted_lifetime_cu");
  auto backing = std::make_shared<ExecutableAddressSpace>(0);
  backing->write_program(std::array<uint32_t, 3>{kSNop, kSNop, kSEndpgm});
  auto owner = std::make_shared<rocjitsu::KfdProcess>(7);
  const std::weak_ptr<rocjitsu::KfdProcess> lifetime = owner;
  const auto handle = gpu_vm.register_address_space(
      7, backing, backing, [owner](uint64_t, amdgpu::VmAccessKind) { (void)owner; });
  ASSERT_TRUE(handle);
  owner.reset();
  fixture.cu()->set_gpu_vm(&gpu_vm);
  auto *wf = fixture.launch(1, 0);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(1);
  wf->set_process_id(7);
  wf->set_address_space(handle);
  fixture.cu()->step();
  ASSERT_EQ(wf->pc, kCodeBase + 4);
  fixture.cu()->abort_dispatch(1);
  ASSERT_TRUE(wf->is_halted());

  ASSERT_TRUE(gpu_vm.unregister_address_space(handle));
  EXPECT_TRUE(lifetime.expired()) << "an idle CU retained the cancelled process binding";
}

TEST(InstructionCacheCuTest, ReentrantFaultCancellationKeepsAccessUntilIssueReturns) {
  GpuVm gpu_vm;
  CuFixture fixture("vm_reentrant_lifetime_cu");
  auto backing = std::make_shared<ExecutableAddressSpace>(0);
  auto owner = std::make_shared<rocjitsu::KfdProcess>(7);
  const std::weak_ptr<rocjitsu::KfdProcess> lifetime = owner;
  unsigned faults = 0;
  const auto handle = gpu_vm.register_address_space(
      7, backing, backing, [owner, &fixture, &lifetime, &faults](uint64_t, amdgpu::VmAccessKind) {
        (void)owner;
        ++faults;
        fixture.cu()->abort_dispatch(1);
        // The registry and in-flight access share the generation's callback.
        // Cancellation must leave its owner alive while the callback executes.
        EXPECT_FALSE(lifetime.expired());
      });
  ASSERT_TRUE(handle);
  owner.reset();
  fixture.cu()->set_gpu_vm(&gpu_vm);
  auto *wf = fixture.launch(1, 0, kCodeBase + InstructionCache::kLineSize * 4);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(1);
  wf->set_process_id(7);
  wf->set_address_space(handle);
  fixture.cu()->step();
  EXPECT_GE(faults, 1u);
  EXPECT_TRUE(wf->is_halted());

  ASSERT_TRUE(gpu_vm.unregister_address_space(handle));
  EXPECT_TRUE(lifetime.expired()) << "fault handling left the binding retained after issue";
}

TEST(InstructionCacheCuTest, ExceptionalQuantumReleasesTheProcessBeforeCancellation) {
  GpuVm gpu_vm;
  CuFixture fixture("vm_exception_lifetime_cu");
  auto backing = std::make_shared<ExecutableAddressSpace>(0);
  auto owner = std::make_shared<rocjitsu::KfdProcess>(7);
  const std::weak_ptr<rocjitsu::KfdProcess> lifetime = owner;
  const auto handle =
      gpu_vm.register_address_space(7, backing, backing, [owner](uint64_t, amdgpu::VmAccessKind) {
        (void)owner;
        throw std::runtime_error("fault callback failed");
      });
  ASSERT_TRUE(handle);
  owner.reset();
  fixture.cu()->set_gpu_vm(&gpu_vm);
  auto *wf = fixture.launch(1, 0, kCodeBase + InstructionCache::kLineSize * 4);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(1);
  wf->set_process_id(7);
  wf->set_address_space(handle);
  EXPECT_THROW(fixture.cu()->run_quantum(), std::runtime_error);
  ASSERT_TRUE(fixture.cu()->has_active_wfs());
  ASSERT_TRUE(gpu_vm.unregister_address_space(handle));
  EXPECT_TRUE(lifetime.expired()) << "exception unwinding must release the quantum snapshot";
  fixture.cu()->abort_dispatch(1);
  EXPECT_TRUE(lifetime.expired());
}

TEST(InstructionCacheCuTest, ActiveWaveSharesReporterAcrossQuantaAndDirectSteps) {
  GpuVm gpu_vm;
  CuFixture fixture("vm_quantum_lifetime_cu");
  auto backing = std::make_shared<ExecutableAddressSpace>(0);
  backing->write_program(std::array<uint32_t, 3>{kSNop, kSNop, kSEndpgm});
  auto process = std::make_shared<rocjitsu::KfdProcess>(7);
  const std::weak_ptr<rocjitsu::KfdProcess> lifetime = process;
  unsigned copies = 0;
  const auto handle =
      gpu_vm.register_address_space(7, backing, backing, CountedReporter(process, copies));
  ASSERT_TRUE(handle);
  process.reset();
  fixture.cu()->set_gpu_vm(&gpu_vm);
  fixture.cu()->set_functional_quantum(1);
  auto *wf = fixture.launch(1, 0);
  ASSERT_NE(wf, nullptr);
  wf->set_dispatch_id(1);
  wf->set_process_id(7);
  wf->set_address_space(handle);
  copies = 0;
  fixture.cu()->run_quantum();
  ASSERT_EQ(wf->pc, kCodeBase + 4);
  EXPECT_EQ(copies, 0u);
  fixture.cu()->run_quantum();
  ASSERT_EQ(wf->pc, kCodeBase + 8);
  EXPECT_EQ(copies, 0u) << "new quantum snapshots must share the fault reporter";
  fixture.cu()->step();
  EXPECT_TRUE(wf->is_halted());
  EXPECT_EQ(copies, 0u) << "direct stepping must share the fault reporter";

  ASSERT_TRUE(gpu_vm.unregister_address_space(handle));
  EXPECT_TRUE(lifetime.expired()) << "the final wave retained its process after retiring";
}

TEST(InstructionCacheCuTest, EndingOneWaveKeepsTheOtherWavesVmAccess) {
  GpuVm gpu_vm;
  CuFixture fixture("vm_multiwave_lifetime_cu", /*wf_slots=*/2);
  auto backing = std::make_shared<ExecutableAddressSpace>(0);
  backing->write_program(std::array<uint32_t, 3>{kSNop, kSNop, kSEndpgm});
  auto process = std::make_shared<rocjitsu::KfdProcess>(7);
  const std::weak_ptr<rocjitsu::KfdProcess> lifetime = process;
  unsigned copies = 0;
  const auto handle =
      gpu_vm.register_address_space(7, backing, backing, CountedReporter(process, copies));
  ASSERT_TRUE(handle);
  process.reset();
  fixture.cu()->set_gpu_vm(&gpu_vm);
  auto *first = fixture.launch(1, 0, kCodeBase + 8);
  auto *second = fixture.launch(1, 1);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  for (auto *wf : {first, second}) {
    wf->set_dispatch_id(1);
    wf->set_process_id(7);
    wf->set_address_space(handle);
  }
  copies = 0;
  fixture.cu()->step();
  ASSERT_TRUE(first->is_halted());
  ASSERT_FALSE(second->is_halted());
  ASSERT_EQ(second->pc, kCodeBase + 4);
  EXPECT_EQ(copies, 0u) << "both waves must share the fault reporter";
  EXPECT_FALSE(lifetime.expired());
  fixture.cu()->step();
  ASSERT_EQ(second->pc, kCodeBase + 8);
  EXPECT_EQ(copies, 0u);
  fixture.cu()->step();
  EXPECT_TRUE(second->is_halted());
  EXPECT_FALSE(fixture.cu()->has_active_wfs());
  EXPECT_EQ(copies, 0u);
  ASSERT_TRUE(gpu_vm.unregister_address_space(handle));
  EXPECT_TRUE(lifetime.expired());
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
  EXPECT_TRUE(std::ranges::equal(next_dispatch, std::span(rewritten).first(next_dispatch.size())))
      << "a new dispatch reused code bytes cached by the previous one";
}

} // namespace
