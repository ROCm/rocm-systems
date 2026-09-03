// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pm4_bootstrap_executor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

class FlatMemory final : public AddressSpaceTranslator, public PhysicalMemoryAccess {
public:
  FlatMemory() : bytes_(0x1000) {}

  VmTranslationResult translate(uint64_t address, std::size_t size, VmAccessKind) const override {
    if (size == 0 || address > bytes_.size() || size > bytes_.size() - address)
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation = {.domain = VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = bytes_.size() - address,
                        .mtype = Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = false}}};
  }

  VmAccessOutcome read(VmMemoryDomain, uint64_t address, std::span<std::byte> bytes) override {
    if (unavailable_read_ && address == *unavailable_read_)
      return VmAccessOutcome::Unavailable;
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(address), bytes.size(), bytes.begin());
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome write(VmMemoryDomain, uint64_t address,
                        std::span<const std::byte> bytes) override {
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::copy(bytes.begin(), bytes.end(), bytes_.begin() + static_cast<std::ptrdiff_t>(address));
    return VmAccessOutcome::Complete;
  }

  AtomicLoadResult atomic_load(VmMemoryDomain, uint64_t address, uint32_t width) override {
    if ((width != 4 && width != 8) || address % width != 0 || address > bytes_.size() ||
        width > bytes_.size() - address)
      return {.outcome = VmAccessOutcome::Malformed};
    uint64_t value = 0;
    std::memcpy(&value, bytes_.data() + address, width);
    return {.outcome = VmAccessOutcome::Complete, .value = value};
  }

  VmAccessOutcome atomic_store(VmMemoryDomain, uint64_t address, uint32_t width,
                               uint64_t value) override {
    if ((width != 4 && width != 8) || address % width != 0 || address > bytes_.size() ||
        width > bytes_.size() - address)
      return VmAccessOutcome::Malformed;
    std::memcpy(bytes_.data() + address, &value, width);
    return VmAccessOutcome::Complete;
  }

  AtomicCompareExchangeResult compare_exchange(VmMemoryDomain, uint64_t, uint32_t, uint64_t,
                                               uint64_t) override {
    return {.outcome = VmAccessOutcome::Malformed};
  }

  template <typename T> void store(uint64_t address, T value) {
    std::memcpy(bytes_.data() + address, &value, sizeof(value));
  }

  void make_read_unavailable(uint64_t address) { unavailable_read_ = address; }

private:
  std::vector<std::byte> bytes_;
  std::optional<uint64_t> unavailable_read_;
};

struct ExecutorFixture {
  ExecutorFixture() : memory(std::make_shared<FlatMemory>()), vm(nullptr) {
    handle = vm.register_translated(1, memory, memory);
    access = vm.snapshot(handle);
  }

  std::shared_ptr<FlatMemory> memory;
  GpuVm vm;
  AddressSpaceHandle handle;
  std::optional<GpuVmAccess> access;
};

TEST(Pm4BootstrapExecutorTest, ExecutesTheBoundedStartupSubsetAcrossRingWrap) {
  ExecutorFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kRing = 0x100;
  constexpr uint64_t kRingDwords = 4;
  fixture.memory->store<uint32_t>(kRing + 3 * sizeof(uint32_t), 0xc0017900);
  fixture.memory->store<uint32_t>(kRing + 0 * sizeof(uint32_t), 0x40);
  fixture.memory->store<uint32_t>(kRing + 1 * sizeof(uint32_t), 0xdeadbeef);
  fixture.memory->store<uint32_t>(kRing + 2 * sizeof(uint32_t), 0xffff1000);

  uint64_t register_dword = 0;
  uint32_t register_value = 0;
  Pm4BootstrapExecutor executor({.write_uconfig_register = [&](uint64_t reg, uint32_t value) {
    register_dword = reg;
    register_value = value;
    return true;
  }});
  const Pm4BootstrapResult result = executor.execute(*fixture.access, kRing, kRingDwords, 3, 7);

  EXPECT_EQ(result.outcome, Pm4BootstrapOutcome::Complete);
  EXPECT_EQ(result.read_pointer, 7u);
  EXPECT_EQ(register_dword, 0xc040u);
  EXPECT_EQ(register_value, 0xdeadbeefu);
}

TEST(Pm4BootstrapExecutorTest, LeavesTheFaultingPacketAtTheReadPointer) {
  ExecutorFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kRing = 0x100;
  fixture.memory->store<uint32_t>(kRing, 0xc0002000);

  Pm4BootstrapExecutor executor({});
  const Pm4BootstrapResult result = executor.execute(*fixture.access, kRing, 4, 0, 1);

  EXPECT_EQ(result.outcome, Pm4BootstrapOutcome::UnsupportedPacket);
  EXPECT_EQ(result.read_pointer, 0u);
  EXPECT_EQ(result.packet_header, 0xc0002000u);
}

TEST(Pm4BootstrapExecutorTest, PreservesUnavailableAsARetryableOutcome) {
  ExecutorFixture fixture;
  ASSERT_TRUE(fixture.access);
  constexpr uint64_t kRing = 0x100;
  fixture.memory->make_read_unavailable(kRing);

  Pm4BootstrapExecutor executor({});
  const Pm4BootstrapResult result = executor.execute(*fixture.access, kRing, 4, 0, 1);

  EXPECT_EQ(result.outcome, Pm4BootstrapOutcome::Unavailable);
  EXPECT_EQ(result.read_pointer, 0u);
}

TEST(Pm4BootstrapExecutorTest, RejectsARingWhoseAddressRangeWraps) {
  ExecutorFixture fixture;
  ASSERT_TRUE(fixture.access);

  Pm4BootstrapExecutor executor({});
  const Pm4BootstrapResult result = executor.execute(*fixture.access, UINT64_MAX - 7, 4, 0, 1);

  EXPECT_EQ(result.outcome, Pm4BootstrapOutcome::Malformed);
  EXPECT_EQ(result.read_pointer, 0u);
}

} // namespace
} // namespace rocjitsu::amdgpu
