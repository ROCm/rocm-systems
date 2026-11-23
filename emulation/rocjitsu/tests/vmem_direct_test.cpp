// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/vm_direct_backing.h"
#include "rocjitsu/vm/amdgpu/vmem_direct.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

constexpr uint64_t kBase = 0x8A0000;
constexpr uint64_t kExtentBytes = 4 * kLegacyPageSize;

// One extent of host bytes standing in for a binding's local memory.
class FakeBacking final : public VmDirectBacking {
public:
  FakeBacking() : host(kExtentBytes) { std::iota(host.begin(), host.end(), uint8_t{1}); }

  uint64_t generation() const override { return current; }
  std::optional<LegacyLocalExtent> extent(uint64_t address, uint64_t generation) const override {
    if (generation != current || address < kBase || address >= kBase + kExtentBytes)
      return std::nullopt;
    return LegacyLocalExtent{.begin = kBase,
                             .end = kBase + kExtentBytes,
                             .host_begin = const_cast<uint8_t *>(host.data()),
                             .mtype = mtype,
                             .owner = owner};
  }
  bool run_admitted(uint64_t generation, void (*fn)(void *), void *context) const override {
    if (!admit || generation != current)
      return false;
    fn(context);
    return true;
  }
  std::optional<VmRawBackingWriteGate::Permit> begin_write(uint64_t, std::size_t) const override {
    if (!writable)
      return std::nullopt;
    return VmRawBackingWriteGate::Permit{};
  }
  bool copy(std::span<const Copy> spans, bool into_memory, bool guarded) const override {
    last_guarded = guarded;
    for (const Copy &span : spans) {
      if (fault_at == copies++)
        return false;
      if (during_copy)
        during_copy();
      if (into_memory)
        std::memcpy(span.host, span.local, span.size);
      else
        std::memcpy(span.local, span.host, span.size);
    }
    return true;
  }

  std::vector<uint8_t> host;
  uint64_t current = 1;
  Mtype mtype = Mtype::RW;
  LegacyHostExtentOwner owner = LegacyHostExtentOwner::DriverSealedRam;
  bool admit = true;
  bool writable = true;
  mutable int fault_at = -1;
  mutable int copies = 0;
  mutable bool last_guarded = false;
  std::function<void()> during_copy;
};

class VmemDirectTest : public ::testing::Test {
protected:
  VmemDirectTest() { l2.set_backing_memory(&memory); }

  std::optional<VmAccessOutcome> access(bool is_load, uint64_t lanes, uint32_t elem_size = 4,
                                        uint32_t num_elems = 1) {
    const DirectVectorRequest request{.addrs = addrs.data(),
                                      .lanes = lanes,
                                      .wf_size = 32,
                                      .elem_size = elem_size,
                                      .num_elems = num_elems,
                                      .instruction_mtype = Mtype::RW,
                                      .transactions = l1.transaction_policy(),
                                      .data = data.data(),
                                      .is_load = is_load};
    return try_direct_vector_access(request, backing, l1, l2);
  }

  uint32_t host_word(uint64_t address) const {
    uint32_t value;
    std::memcpy(&value, backing.host.data() + (address - kBase), sizeof(value));
    return value;
  }
  uint32_t lane_word(uint32_t lane, uint32_t stride = 4) const {
    uint32_t value;
    std::memcpy(&value, data.data() + lane * stride, sizeof(value));
    return value;
  }

  GpuMemory memory{"memory"};
  L2Cache l2{"l2"};
  L1VectorCache l1{&l2};
  FakeBacking backing;
  std::array<uint64_t, 64> addrs{};
  std::array<uint8_t, 64 * 16> data{};
};

TEST_F(VmemDirectTest, LoadsContiguousAndReorderedLanes) {
  for (uint32_t lane = 0; lane < 32; ++lane)
    addrs[lane] = kBase + 0x100 + lane * 4;
  ASSERT_EQ(access(true, UINT32_MAX), VmAccessOutcome::Complete);
  for (uint32_t lane = 0; lane < 32; ++lane)
    EXPECT_EQ(lane_word(lane), host_word(addrs[lane])) << lane;

  data.fill(0);
  for (uint32_t lane = 0; lane < 32; ++lane)
    addrs[lane] = kBase + 0x2000 + (31 - lane) * 4;
  ASSERT_EQ(access(true, 0x0000ffffu), VmAccessOutcome::Complete);
  for (uint32_t lane = 0; lane < 32; ++lane)
    EXPECT_EQ(lane_word(lane), lane < 16 ? host_word(addrs[lane]) : 0u) << lane;
}

TEST_F(VmemDirectTest, StoresWholeElementsOfEveryActiveLane) {
  for (uint32_t lane = 0; lane < 32; ++lane) {
    addrs[lane] = kBase + 0x400 + lane * 16;
    for (uint32_t byte = 0; byte < 16; ++byte)
      data[lane * 16 + byte] = static_cast<uint8_t>(lane * 16 + byte);
  }
  const uint32_t untouched = host_word(addrs[3]);
  ASSERT_EQ(access(false, UINT32_MAX & ~(uint64_t{1} << 3), 4, 4), VmAccessOutcome::Complete);
  for (uint32_t lane = 0; lane < 32; ++lane) {
    const uint32_t expected = lane == 3 ? untouched : lane_word(lane, 16);
    EXPECT_EQ(host_word(addrs[lane]), expected) << lane;
  }
}

TEST_F(VmemDirectTest, DeclinesUncachedPagesAndExtentGapsWithoutEffect) {
  for (uint32_t lane = 0; lane < 32; ++lane)
    addrs[lane] = kBase + lane * 4;
  const std::vector<uint8_t> before = backing.host;
  for (const Mtype mtype : {Mtype::UC, Mtype::CC}) {
    backing.mtype = mtype;
    EXPECT_FALSE(access(false, UINT32_MAX));
  }
  backing.mtype = Mtype::RW;
  addrs[31] = kBase + kExtentBytes;
  EXPECT_FALSE(access(false, UINT32_MAX));
  addrs[31] = kBase + kExtentBytes - 2;
  EXPECT_FALSE(access(false, UINT32_MAX));
  EXPECT_EQ(backing.host, before);
  EXPECT_EQ(backing.copies, 0);
}

TEST_F(VmemDirectTest, StaleGenerationOrClosedWriteDeclinesWithoutEffect) {
  for (uint32_t lane = 0; lane < 32; ++lane)
    addrs[lane] = kBase + lane * 4;
  backing.admit = false;
  EXPECT_FALSE(access(false, UINT32_MAX));
  backing.admit = true;
  backing.writable = false;
  EXPECT_FALSE(access(false, UINT32_MAX));
  EXPECT_EQ(backing.copies, 0);
  EXPECT_EQ(access(true, UINT32_MAX), VmAccessOutcome::Complete);
}

TEST_F(VmemDirectTest, HostFaultDeclinesSoTheRoutedPathRepeatsTheRequest) {
  for (uint32_t lane = 0; lane < 32; ++lane)
    addrs[lane] = kBase + lane * 64;
  backing.owner = LegacyHostExtentOwner::Application;
  backing.fault_at = 2;
  EXPECT_FALSE(access(false, UINT32_MAX));
  EXPECT_TRUE(backing.last_guarded);
  backing.owner = LegacyHostExtentOwner::DriverSealedRam;
  backing.fault_at = -1;
  EXPECT_EQ(access(true, UINT32_MAX), VmAccessOutcome::Complete);
  EXPECT_FALSE(backing.last_guarded);
}

TEST_F(VmemDirectTest, StoreDeclinesWhileACacheMayHoldItsBytes) {
  const uint64_t resident = kBase + 0x800;
  const uint64_t one[1] = {resident};
  std::array<uint8_t, 4> sink{};
  ASSERT_EQ(l1.load(one, 1, 4, 1, sink.data(), Mtype::RW, false, false, 1),
            VmAccessOutcome::Complete);
  for (uint32_t lane = 0; lane < 32; ++lane)
    addrs[lane] = resident + lane * 4;
  // A resident copy would turn stale under a store that bypasses it; loads
  // read backing, which is never older than a write-through copy.
  EXPECT_FALSE(access(false, UINT32_MAX));
  EXPECT_EQ(access(true, UINT32_MAX), VmAccessOutcome::Complete);
  l1.invalidate_all();
  EXPECT_FALSE(access(false, UINT32_MAX));
  ASSERT_EQ(l2.flush_all(), VmAccessOutcome::Complete);
  EXPECT_EQ(access(false, UINT32_MAX), VmAccessOutcome::Complete);
}

TEST_F(VmemDirectTest, StoreDiscardsACopyAnotherCuFilledDuringTheWrite) {
  L1VectorCache other_cu(&l2);
  const uint64_t line = kBase + 0x1800;
  for (uint32_t lane = 0; lane < 32; ++lane)
    addrs[lane] = line + lane * 4;
  backing.during_copy = [&] {
    const uint64_t one[1] = {line};
    std::array<uint8_t, 4> sink{};
    ASSERT_EQ(other_cu.load(one, 1, 4, 1, sink.data(), Mtype::RW, false, false, 1),
              VmAccessOutcome::Complete);
  };
  ASSERT_EQ(access(false, UINT32_MAX), VmAccessOutcome::Complete);
  // The other CU's L1 may keep its copy until it invalidates, as on hardware;
  // the shared L2 must not serve the pre-store bytes to anyone afterwards.
  EXPECT_TRUE(l2.may_bypass(line, 128));
}

TEST_F(VmemDirectTest, RecheckDropsACopyFilledDuringTheBypassWrite) {
  const uint64_t line = kBase + 0x1000;
  const uint64_t one[1] = {line};
  std::array<uint8_t, 4> sink{};
  EXPECT_FALSE(l2.bypass_raced_fill(line, 4));
  ASSERT_EQ(l1.load(one, 1, 4, 1, sink.data(), Mtype::RW, false, false, 1),
            VmAccessOutcome::Complete);
  l1.invalidate_all();
  EXPECT_TRUE(l2.bypass_raced_fill(line, 4));
  l2.drop_bypassed_copies(line, 4);
  EXPECT_FALSE(l2.bypass_raced_fill(line, 4));
  EXPECT_TRUE(l2.may_bypass(line, L2Cache::LINE_SIZE));
}

} // namespace
} // namespace rocjitsu::amdgpu
