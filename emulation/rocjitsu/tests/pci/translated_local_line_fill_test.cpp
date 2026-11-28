// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/pci/vram_store.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <latch>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <vector>

namespace {

using rocjitsu::amdgpu::AddressSpaceHandle;
using rocjitsu::amdgpu::AddressSpaceTranslator;
using rocjitsu::amdgpu::AtomicCompareExchangeResult;
using rocjitsu::amdgpu::AtomicLoadResult;
using rocjitsu::amdgpu::DeviceCacheCoherence;
using rocjitsu::amdgpu::GpuMemory;
using rocjitsu::amdgpu::GpuVm;
using rocjitsu::amdgpu::GpuVmAccess;
using rocjitsu::amdgpu::L2Cache;
using rocjitsu::amdgpu::Mtype;
using rocjitsu::amdgpu::PhysicalMemoryAccess;
using rocjitsu::amdgpu::VmAccessKind;
using rocjitsu::amdgpu::VmAccessOutcome;
using rocjitsu::amdgpu::VmMemoryDomain;
using rocjitsu::amdgpu::VmTranslationResult;

constexpr uint64_t kVramBytes = 0x2000;

/// @brief Maps the first page of an address space onto local memory at the
///        same offsets, for every kind of access.
class LocalPage final : public AddressSpaceTranslator {
public:
  VmTranslationResult translate(uint64_t address, std::size_t size, VmAccessKind) const override {
    if (address >= kPageBytes || size == 0 || size > kPageBytes - address) {
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    }
    return {.outcome = VmAccessOutcome::Complete,
            .translation = {.domain = VmMemoryDomain::Local,
                            .address = address,
                            .contiguous_bytes = kPageBytes - address,
                            .mtype = Mtype::RW,
                            .permissions = {.readable = true, .writable = true}}};
  }

private:
  static constexpr uint64_t kPageBytes = 0x1000;
};

/// @brief Local memory held in a real video memory store.
/// @details Stands in for the PCI device between the address space and the
/// store. The order in which a store is made and then published to the
/// coherence domain is repeated here from GpuPciDevice, not shared with it,
/// and has to be kept in step with it.
class VramBacking final : public PhysicalMemoryAccess {
public:
  explicit VramBacking(std::shared_ptr<DeviceCacheCoherence> domain) : domain_(std::move(domain)) {}

  [[nodiscard]] bool usable() const { return vram_.usable(); }
  bool supports_translated_local_cache() const override { return true; }

  VmAccessOutcome read(VmMemoryDomain, uint64_t address, std::span<std::byte> bytes) override {
    return vram_.read_all(bytes, address) ? VmAccessOutcome::Complete : VmAccessOutcome::Faulted;
  }
  VmAccessOutcome write(VmMemoryDomain, uint64_t address,
                        std::span<const std::byte> bytes) override {
    if (!vram_.write_all(bytes, address)) {
      return VmAccessOutcome::Faulted;
    }
    domain_->publish_local_backing_write(address, bytes.size());
    return VmAccessOutcome::Complete;
  }
  AtomicLoadResult atomic_load(VmMemoryDomain, uint64_t address, uint32_t width) override {
    const auto loaded = vram_.atomic_load(address, width);
    if (!loaded) {
      return {.outcome = VmAccessOutcome::Faulted};
    }
    return {.outcome = VmAccessOutcome::Complete, .value = loaded->value};
  }
  VmAccessOutcome atomic_store(VmMemoryDomain, uint64_t address, uint32_t width,
                               uint64_t value) override {
    if (!vram_.atomic_store(address, width, value)) {
      return VmAccessOutcome::Faulted;
    }
    domain_->publish_local_backing_write(address, width);
    return VmAccessOutcome::Complete;
  }
  AtomicCompareExchangeResult compare_exchange(VmMemoryDomain, uint64_t address, uint32_t width,
                                               uint64_t expected, uint64_t desired) override {
    const auto result = vram_.compare_exchange(address, width, expected, desired);
    if (!result) {
      return {.outcome = VmAccessOutcome::Faulted};
    }
    if (result->exchanged) {
      domain_->publish_local_backing_write(address, width);
    }
    return {.outcome = VmAccessOutcome::Complete,
            .observed = result->observed,
            .exchanged = result->exchanged};
  }

private:
  std::shared_ptr<DeviceCacheCoherence> domain_;
  rocjitsu::VramStore vram_{"line-fill-vram", kVramBytes, kVramBytes};
};

// Compute units fill a cache line from mapped video memory while others
// exchange, atomically store and plainly store single words of that line. The
// fill holds no lock in common with those stores: it loads each word once, and
// relies on the line's publication epoch, which every store advances after it
// is made, to keep a line that was filled across a store from being served
// afterwards. A plain copy on the mapped path races the stores here, which a
// ThreadSanitizer build reports. In any build the test checks what a compute
// unit is owed: a read that begins after a store has returned sees that store
// or a later one, whether the cache or memory answers it; the words nobody
// stores stay intact; and no store or exchange is lost. The stores go on past
// their minimum count until fills have been overtaken by them, for as long as
// the deadline allows, so that a run in which the two barely met is not taken
// for one in which they raced.
TEST(L2CacheVramLineFill, FillsRaceTranslatedExchangesAndStoresOfOtherWordsInTheLine) {
  constexpr uint64_t kLine = 0x400;
  constexpr std::size_t kExchangedQuadword = 0x18;
  constexpr std::size_t kAtomicallyStored = 0x48;
  constexpr std::size_t kPlainQuadword = 0x68;
  constexpr std::array<std::size_t, 3> kStored{kExchangedQuadword, kAtomicallyStored,
                                               kPlainQuadword};
  constexpr uint32_t kRounds = 20000;
  constexpr uint32_t kOvertakenFills = 64;
  constexpr std::chrono::seconds kRacingLimit{5};
  constexpr std::size_t kFillers = 2;
  // The round in the low half and its complement in the high half, so that a
  // value made of two different stores is told apart from either.
  const auto quadword_of = [](uint32_t round) { return (uint64_t{~round} << 32) | round; };
  const auto round_of = [](uint64_t quadword) -> std::optional<uint32_t> {
    const uint32_t round = static_cast<uint32_t>(quadword);
    if (static_cast<uint32_t>(quadword >> 32) != ~round) {
      return std::nullopt;
    }
    return round;
  };
  using Line = std::array<uint8_t, L2Cache::LINE_SIZE>;
  const auto line_at_rounds = [&](const std::array<uint32_t, kStored.size()> &rounds) {
    Line line;
    for (std::size_t index = 0; index < line.size(); ++index) {
      line[index] = static_cast<uint8_t>(index * 7 % 255 + 1);
    }
    for (std::size_t index = 0; index < kStored.size(); ++index) {
      const uint64_t value = quadword_of(rounds[index]);
      std::memcpy(line.data() + kStored[index], &value, sizeof(value));
    }
    return line;
  };

  auto coherence = std::make_shared<DeviceCacheCoherence>();
  auto backing = std::make_shared<VramBacking>(coherence);
  ASSERT_TRUE(backing->usable());
  GpuVm vm;
  const AddressSpaceHandle space =
      vm.register_unrouted_address_space(77, std::make_shared<LocalPage>(), backing);
  ASSERT_TRUE(space);
  GpuMemory legacy_backing("line-fill-legacy-maintenance");
  L2Cache l2("line-fill-l2", coherence);
  l2.set_gpu_vm(&vm);
  l2.set_legacy_maintenance_memory(&legacy_backing);
  l2.set_legacy_maintenance_vm(&vm);

  const auto fill = [&](const GpuVmAccess &access, Line &line) {
    const auto domain_access = coherence->acquire_local_cache_access();
    return l2.read_translated_local(access, kLine, line, Mtype::RW, domain_access);
  };

  const Line initial = line_at_rounds({});
  {
    const auto access = vm.snapshot(space);
    ASSERT_TRUE(access);
    ASSERT_EQ(access->write(kLine, std::as_bytes(std::span(initial))), VmAccessOutcome::Complete);
    Line line{};
    ASSERT_EQ(fill(*access, line), VmAccessOutcome::Complete);
    ASSERT_EQ(line, initial);
    const uint64_t fetched = l2.backing_read_transactions();
    ASSERT_EQ(fill(*access, line), VmAccessOutcome::Complete);
    ASSERT_EQ(line, initial);
    ASSERT_EQ(l2.backing_read_transactions(), fetched) << "the line is not being cached";
  }

  std::latch start{1};
  std::atomic<bool> stop{false};
  // The last round whose store of each word has returned to its compute unit.
  std::array<std::atomic<uint32_t>, kStored.size()> returned{};
  // Fills that a store was published during, which the cache turned away.
  std::atomic<uint32_t> overtaken{0};
  const auto deadline = std::chrono::steady_clock::now() + kRacingLimit;
  const auto racing = [&](uint32_t round) {
    return round <= kRounds || (overtaken.load(std::memory_order_relaxed) < kOvertakenFills &&
                                std::chrono::steady_clock::now() < deadline);
  };
  std::atomic<uint32_t> refused{0};
  std::atomic<uint32_t> lost{0};
  std::atomic<uint32_t> disturbed{0};
  std::atomic<uint32_t> torn{0};
  std::atomic<uint32_t> stale{0};
  const auto count_unless = [](std::atomic<uint32_t> &counter, bool ok) {
    if (!ok) {
      counter.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::vector<std::jthread> writers;
  writers.emplace_back([&] {
    const auto access = vm.snapshot(space);
    start.wait();
    for (uint32_t round = 1; access && racing(round); ++round) {
      const AtomicCompareExchangeResult exchanged = access->compare_exchange(
          kLine + kExchangedQuadword, sizeof(uint64_t), quadword_of(round - 1), quadword_of(round));
      count_unless(refused, exchanged.outcome == VmAccessOutcome::Complete);
      // Nothing else stores this word, so an exchange can only fail to find
      // its own last value if something put an older one back.
      count_unless(lost, exchanged.exchanged);
      returned[0].store(round);
    }
  });
  writers.emplace_back([&] {
    const auto access = vm.snapshot(space);
    start.wait();
    for (uint32_t round = 1; access && racing(round); ++round) {
      count_unless(refused, access->atomic_store(kLine + kAtomicallyStored, sizeof(uint64_t),
                                                 quadword_of(round)) == VmAccessOutcome::Complete);
      returned[1].store(round);
    }
  });
  writers.emplace_back([&] {
    const auto access = vm.snapshot(space);
    start.wait();
    for (uint32_t round = 1; access && racing(round); ++round) {
      const uint64_t quadword = quadword_of(round);
      count_unless(refused,
                   access->write(kLine + kPlainQuadword, std::as_bytes(std::span(&quadword, 1))) ==
                       VmAccessOutcome::Complete);
      returned[2].store(round);
    }
  });
  std::vector<std::jthread> fillers;
  for (std::size_t filler = 0; filler < kFillers; ++filler) {
    fillers.emplace_back([&] {
      const auto access = vm.snapshot(space);
      Line line{};
      start.wait();
      // Sampled before the fill, so that one more fill follows the last store.
      for (bool last = false; access && !last;) {
        last = stop.load();
        std::array<uint32_t, kStored.size()> returned_before{};
        for (std::size_t index = 0; index < kStored.size(); ++index) {
          returned_before[index] = returned[index].load();
        }
        // A fill that a store overtook is refused rather than cached, and is
        // simply made again.
        VmAccessOutcome outcome = VmAccessOutcome::Unavailable;
        while ((outcome = fill(*access, line)) == VmAccessOutcome::Unavailable) {
          overtaken.fetch_add(1, std::memory_order_relaxed);
        }
        if (outcome != VmAccessOutcome::Complete) {
          count_unless(refused, false);
          continue;
        }
        Line unstored = line;
        for (std::size_t index = 0; index < kStored.size(); ++index) {
          uint64_t value = 0;
          std::memcpy(&value, line.data() + kStored[index], sizeof(value));
          const std::optional<uint32_t> round = round_of(value);
          count_unless(torn, round.has_value());
          if (round) {
            count_unless(stale, *round >= returned_before[index]);
          }
          std::memcpy(unstored.data() + kStored[index], initial.data() + kStored[index],
                      sizeof(value));
        }
        count_unless(disturbed, unstored == initial);
      }
    });
  }
  start.count_down();
  writers.clear();
  stop.store(true);
  fillers.clear();

  EXPECT_EQ(refused.load(), 0u);
  EXPECT_EQ(lost.load(), 0u) << "an exchange did not find the value the one before it left";
  EXPECT_EQ(disturbed.load(), 0u) << "a fill returned a changed byte that no thread stores";
  EXPECT_EQ(torn.load(), 0u) << "a fill returned a word that no single store left";
  EXPECT_EQ(stale.load(), 0u) << "a fill returned a word older than a store that had returned";
  std::array<uint32_t, kStored.size()> final_rounds{};
  for (std::size_t index = 0; index < kStored.size(); ++index) {
    final_rounds[index] = returned[index].load();
    EXPECT_GE(final_rounds[index], kRounds);
  }
  RecordProperty("overtaken_fills", static_cast<int>(overtaken.load()));

  const auto access = vm.snapshot(space);
  ASSERT_TRUE(access);
  Line line{};
  ASSERT_EQ(fill(*access, line), VmAccessOutcome::Complete);
  EXPECT_EQ(line, line_at_rounds(final_rounds));
  Line in_memory{};
  ASSERT_EQ(access->read(kLine, std::as_writable_bytes(std::span(in_memory))),
            VmAccessOutcome::Complete);
  EXPECT_EQ(in_memory, line_at_rounds(final_rounds));
}

} // namespace
