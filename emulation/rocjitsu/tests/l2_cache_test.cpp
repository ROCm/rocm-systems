// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/hbm_controller.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_side_cache.h"
#include "rocjitsu/vm/amdgpu/request_mtype_resolver.h"
#include "rocjitsu/vm/soc.h"
#include "simdojo/sim/exec_mode.h"
#include "util/except.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <linux/memfd.h>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace rocjitsu::amdgpu {

class MemorySideCacheTestAccess {
public:
  static void store_dirty_line(MemorySideCache &cache, uint64_t line_address,
                               std::span<const uint8_t> data, uint32_t vmid) {
    assert(CacheStore::line_address(line_address) == line_address);
    assert(data.size() == MemorySideCache::LINE_SIZE);
    std::unique_lock access_lock(cache.access_gate_);
    std::lock_guard stripe_lock(cache.stripes_[cache.stripe_index(line_address)]);
    ASSERT_EQ(cache.ensure_line(line_address, vmid), VmAccessOutcome::Complete);
    cache.cache_.write_line(line_address, data.data(), 0, static_cast<uint32_t>(data.size()), vmid);
    simdojo::CacheTag *tag = nullptr;
    cache.cache_.lookup(line_address, &tag, vmid);
    assert(tag != nullptr);
    tag->dirty = true;
    tag->coherence = simdojo::CoherenceState::MODIFIED;
  }

  static bool line_is_dirty(MemorySideCache &cache, uint64_t line_address, uint32_t vmid) {
    std::unique_lock access_lock(cache.access_gate_);
    std::lock_guard stripe_lock(cache.stripes_[cache.stripe_index(line_address)]);
    simdojo::CacheTag *tag = nullptr;
    return cache.cache_.lookup(line_address, &tag, vmid) && tag->dirty;
  }

private:
  using CacheStore = MemorySideCache::CacheStore;
};

} // namespace rocjitsu::amdgpu

namespace {

using rocjitsu::amdgpu::GpuMemory;
using rocjitsu::amdgpu::HbmController;
using rocjitsu::amdgpu::L1ScalarCache;
using rocjitsu::amdgpu::L1VectorCache;
using rocjitsu::amdgpu::L2Cache;
using rocjitsu::amdgpu::MemorySideCache;
using rocjitsu::amdgpu::Mtype;
using rocjitsu::amdgpu::RequestMtypeResolver;

class IdentityAddressSpace final : public rocjitsu::amdgpu::AddressSpaceTranslator {
public:
  explicit IdentityAddressSpace(uint64_t size) : size_(size) {}

  rocjitsu::amdgpu::VmTranslationResult translate(uint64_t address, std::size_t size,
                                                  rocjitsu::amdgpu::VmAccessKind) const override {
    if (size == 0 || address > size_ || size > size_ - address)
      return {.outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted, .translation = {}};
    return {
        .outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete,
        .translation = {.domain = rocjitsu::amdgpu::VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = size_ - address,
                        .mtype = Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }

private:
  uint64_t size_;
};

class ConfigurablePhysicalMemory final : public rocjitsu::amdgpu::PhysicalMemoryAccess {
public:
  explicit ConfigurablePhysicalMemory(std::size_t size) : bytes_(size) {}

  rocjitsu::amdgpu::VmAccessOutcome read(rocjitsu::amdgpu::VmMemoryDomain domain, uint64_t address,
                                         std::span<std::byte> bytes) override {
    ++read_calls;
    if (read_outcome != rocjitsu::amdgpu::VmAccessOutcome::Complete)
      return read_outcome;
    if (domain != rocjitsu::amdgpu::VmMemoryDomain::System || !contains(address, bytes.size()))
      return rocjitsu::amdgpu::VmAccessOutcome::Faulted;
    std::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(address), bytes.size(), bytes.begin());
    return rocjitsu::amdgpu::VmAccessOutcome::Complete;
  }

  rocjitsu::amdgpu::VmAccessOutcome write(rocjitsu::amdgpu::VmMemoryDomain domain, uint64_t address,
                                          std::span<const std::byte> bytes) override {
    ++write_calls;
    if (write_outcome != rocjitsu::amdgpu::VmAccessOutcome::Complete)
      return write_outcome;
    if (domain != rocjitsu::amdgpu::VmMemoryDomain::System || !contains(address, bytes.size()))
      return rocjitsu::amdgpu::VmAccessOutcome::Faulted;
    std::copy(bytes.begin(), bytes.end(), bytes_.begin() + static_cast<std::ptrdiff_t>(address));
    return rocjitsu::amdgpu::VmAccessOutcome::Complete;
  }

  rocjitsu::amdgpu::VmAccessOutcome atomic_modify(rocjitsu::amdgpu::VmMemoryDomain domain,
                                                  uint64_t address, uint32_t width,
                                                  const AtomicMutation &mutation) override {
    ++atomic_calls;
    if (atomic_outcome != rocjitsu::amdgpu::VmAccessOutcome::Complete)
      return atomic_outcome;
    if (domain != rocjitsu::amdgpu::VmMemoryDomain::System || !contains(address, width) ||
        (width != sizeof(uint32_t) && width != sizeof(uint64_t)))
      return rocjitsu::amdgpu::VmAccessOutcome::Faulted;
    mutation(std::span<std::byte>(bytes_.data() + address, width));
    return rocjitsu::amdgpu::VmAccessOutcome::Complete;
  }

  template <typename T> void store(uint64_t address, T value) {
    ASSERT_TRUE(contains(address, sizeof(value)));
    if (contains(address, sizeof(value)))
      std::memcpy(bytes_.data() + address, &value, sizeof(value));
  }

  template <typename T> T load(uint64_t address) const {
    T value{};
    EXPECT_TRUE(contains(address, sizeof(value)));
    if (contains(address, sizeof(value)))
      std::memcpy(&value, bytes_.data() + address, sizeof(value));
    return value;
  }

  rocjitsu::amdgpu::VmAccessOutcome read_outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete;
  rocjitsu::amdgpu::VmAccessOutcome write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete;
  rocjitsu::amdgpu::VmAccessOutcome atomic_outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete;
  uint32_t read_calls = 0;
  uint32_t write_calls = 0;
  uint32_t atomic_calls = 0;

private:
  bool contains(uint64_t address, std::size_t size) const {
    return address <= bytes_.size() && size <= bytes_.size() - address;
  }

  std::vector<std::byte> bytes_;
};

class LinkedCacheHierarchy {
public:
  explicit LinkedCacheHierarchy(std::size_t size)
      : translator(std::make_shared<IdentityAddressSpace>(size)),
        physical(std::make_shared<ConfigurablePhysicalMemory>(size)), hbm("hbm", &memory, &gpu_vm),
        memory_side_cache("memory_side_cache", coherence, &memory), l2("l2", coherence),
        memory_side_port(memory_side_cache.create_cpl_port("l2")),
        memory_side_hbm_link(/*id=*/0, memory_side_cache.req_port(), hbm.cpl_port(), /*latency=*/0),
        l2_memory_side_link(/*id=*/1, l2.req_port(), memory_side_port, /*latency=*/0) {
    address_space = gpu_vm.register_translated(kVmid, translator, physical);
    memory_side_hbm_link.set_exec_mode(simdojo::ExecMode::FUNCTIONAL);
    l2_memory_side_link.set_exec_mode(simdojo::ExecMode::FUNCTIONAL);
    memory_side_cache.req_port()->set_link(&memory_side_hbm_link);
    hbm.cpl_port()->set_link(&memory_side_hbm_link);
    l2.req_port()->set_link(&l2_memory_side_link);
    memory_side_port->set_link(&l2_memory_side_link);
  }

  static constexpr uint32_t kVmid = 73;
  std::shared_ptr<IdentityAddressSpace> translator;
  std::shared_ptr<ConfigurablePhysicalMemory> physical;
  GpuMemory memory{"memory"};
  rocjitsu::amdgpu::GpuVm gpu_vm;
  rocjitsu::amdgpu::AddressSpaceHandle address_space;
  std::shared_ptr<rocjitsu::amdgpu::DeviceCacheCoherence> coherence =
      std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  HbmController hbm;
  MemorySideCache memory_side_cache;
  L2Cache l2;
  simdojo::Port *memory_side_port;
  simdojo::Link memory_side_hbm_link;
  simdojo::Link l2_memory_side_link;
};

TEST(L2CacheTest, LinkedUnavailableReadDoesNotFillCacheHierarchy) {
  constexpr uint64_t kAddress = 0x400;
  constexpr uint32_t kBackingValue = 0x1234abcd;
  constexpr uint32_t kSentinel = 0xfeedface;
  LinkedCacheHierarchy hierarchy(0x2000);
  ASSERT_TRUE(hierarchy.address_space);
  hierarchy.physical->store(kAddress, kBackingValue);
  hierarchy.physical->read_outcome = rocjitsu::amdgpu::VmAccessOutcome::Unavailable;

  uint32_t observed = kSentinel;
  EXPECT_EQ(hierarchy.l2.read(kAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed),
                              Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_EQ(observed, kSentinel);
  EXPECT_EQ(hierarchy.physical->read_calls, 1u);

  hierarchy.physical->read_outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete;
  EXPECT_EQ(hierarchy.l2.read(kAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed),
                              Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kBackingValue);
  EXPECT_EQ(hierarchy.physical->read_calls, 2u);
}

TEST(L2CacheTest, LinkedWriteFailureDoesNotMutateCachedOrBackingState) {
  constexpr uint64_t kAddress = 0x500;
  constexpr uint32_t kInitial = 0x01020304;
  constexpr uint32_t kUnavailableValue = 0x11112222;
  constexpr uint32_t kFaultedValue = 0x33334444;
  LinkedCacheHierarchy hierarchy(0x2000);
  ASSERT_TRUE(hierarchy.address_space);
  hierarchy.physical->store(kAddress, kInitial);

  uint32_t observed = 0;
  ASSERT_EQ(hierarchy.l2.read(kAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed),
                              Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(observed, kInitial);

  hierarchy.physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Unavailable;
  EXPECT_EQ(hierarchy.l2.write(kAddress, reinterpret_cast<const uint8_t *>(&kUnavailableValue),
                               sizeof(kUnavailableValue), Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_EQ(hierarchy.physical->load<uint32_t>(kAddress), kInitial);
  EXPECT_EQ(hierarchy.l2.read(kAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed),
                              Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kInitial);

  hierarchy.physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted;
  EXPECT_EQ(hierarchy.l2.write(kAddress, reinterpret_cast<const uint8_t *>(&kFaultedValue),
                               sizeof(kFaultedValue), Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(hierarchy.physical->load<uint32_t>(kAddress), kInitial);
  EXPECT_EQ(hierarchy.l2.read(kAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed),
                              Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kInitial);
}

TEST(L2CacheTest, LinkedMalformedReadPropagatesWithoutFillingCaches) {
  constexpr uint64_t kAddress = 0x580;
  constexpr uint32_t kBackingValue = 0x55667788;
  constexpr uint32_t kSentinel = 0xdeadbeef;
  LinkedCacheHierarchy hierarchy(0x2000);
  ASSERT_TRUE(hierarchy.address_space);
  hierarchy.physical->store(kAddress, kBackingValue);
  hierarchy.physical->read_outcome = rocjitsu::amdgpu::VmAccessOutcome::Malformed;

  uint32_t observed = kSentinel;
  EXPECT_EQ(hierarchy.l2.read(kAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed),
                              Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Malformed);
  EXPECT_EQ(observed, kSentinel);

  hierarchy.physical->read_outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete;
  EXPECT_EQ(hierarchy.l2.read(kAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed),
                              Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kBackingValue);
}

TEST(L2CacheTest, LinkedFaultedReadDoesNotCommitOrRunAtomicCallback) {
  constexpr uint64_t kAddress = 0x600;
  constexpr uint32_t kAtomicInitial = 17;
  constexpr uint32_t kSentinel = 0xa5a55a5a;
  LinkedCacheHierarchy hierarchy(0x2000);
  ASSERT_TRUE(hierarchy.address_space);
  hierarchy.physical->read_outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted;

  uint32_t observed = kSentinel;
  EXPECT_EQ(hierarchy.l2.read(kAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed),
                              Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(observed, kSentinel);

  bool callback_ran = false;
  hierarchy.physical->atomic_outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted;
  EXPECT_EQ(hierarchy.l2.atomic_rmw(
                kAddress, sizeof(uint32_t), [&](uint8_t *, uint32_t) { callback_ran = true; },
                LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Faulted);
  EXPECT_FALSE(callback_ran);
  EXPECT_EQ(hierarchy.physical->write_calls, 0u);
  EXPECT_EQ(hierarchy.physical->atomic_calls, 1u);

  hierarchy.physical->store(kAddress, kAtomicInitial);
  hierarchy.physical->atomic_outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete;
  EXPECT_EQ(hierarchy.l2.atomic_rmw(
                kAddress, sizeof(uint32_t),
                [&](uint8_t *target, uint32_t) {
                  callback_ran = true;
                  uint32_t value = 0;
                  std::memcpy(&value, target, sizeof(value));
                  ++value;
                  std::memcpy(target, &value, sizeof(value));
                },
                LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_TRUE(callback_ran);
  EXPECT_EQ(hierarchy.physical->load<uint32_t>(kAddress), kAtomicInitial + 1);
  EXPECT_EQ(hierarchy.physical->atomic_calls, 2u);
  EXPECT_EQ(hierarchy.physical->write_calls, 0u);
}

TEST(L2CacheTest, AtomicBoundaryFailureRetainsDirtyStateAndSuppressesMutation) {
  constexpr uint32_t kVmid = 76;
  constexpr uint64_t kDirtyAddress = 0x800;
  constexpr uint64_t kAtomicAddress = 0x1000;
  constexpr uint32_t kInitialDirty = 0x11112222;
  constexpr uint32_t kDirty = 0x33334444;
  constexpr uint32_t kInitialAtomic = 9;
  auto translator = std::make_shared<IdentityAddressSpace>(0x3000);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(0x3000);
  physical->store(kDirtyAddress, kInitialDirty);
  physical->store(kAtomicAddress, kInitialAtomic);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  ASSERT_TRUE(gpu_vm.register_translated(kVmid, translator, physical));
  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&gpu_vm);

  std::array<uint8_t, L2Cache::LINE_SIZE> dirty_line{};
  std::memcpy(dirty_line.data(), &kDirty, sizeof(kDirty));
  ASSERT_EQ(
      l2.writeback_line(kDirtyAddress, dirty_line.data(), 0, sizeof(kDirty), Mtype::RW, kVmid),
      rocjitsu::amdgpu::VmAccessOutcome::Complete);

  physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted;
  const uint64_t epoch_before_failure = coherence->current_epoch();
  bool callback_ran = false;
  auto increment = [&](uint8_t *target, uint32_t) {
    callback_ran = true;
    uint32_t value = 0;
    std::memcpy(&value, target, sizeof(value));
    ++value;
    std::memcpy(target, &value, sizeof(value));
  };
  EXPECT_EQ(l2.atomic_rmw(kAtomicAddress, sizeof(uint32_t), increment, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Faulted);
  EXPECT_FALSE(callback_ran);
  EXPECT_EQ(physical->atomic_calls, 0u);
  EXPECT_EQ(coherence->current_epoch(), epoch_before_failure);
  EXPECT_EQ(physical->load<uint32_t>(kDirtyAddress), kInitialDirty);
  EXPECT_EQ(physical->load<uint32_t>(kAtomicAddress), kInitialAtomic);

  physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete;
  EXPECT_EQ(l2.atomic_rmw(kAtomicAddress, sizeof(uint32_t), increment, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_TRUE(callback_ran);
  EXPECT_EQ(physical->load<uint32_t>(kDirtyAddress), kDirty);
  EXPECT_EQ(physical->load<uint32_t>(kAtomicAddress), kInitialAtomic + 1);
}

TEST(L2CacheTest, FailedDirectDirtyFlushRetainsLineForRetry) {
  constexpr uint32_t kVmid = 74;
  constexpr uint64_t kAddress = 0x800;
  constexpr uint32_t kInitial = 0x11112222;
  constexpr uint32_t kDirty = 0x33334444;
  auto translator = std::make_shared<IdentityAddressSpace>(0x2000);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  physical->store(kAddress, kInitial);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  ASSERT_TRUE(gpu_vm.register_translated(kVmid, translator, physical));
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&gpu_vm);

  std::array<uint8_t, L2Cache::LINE_SIZE> dirty_line{};
  std::memcpy(dirty_line.data(), &kDirty, sizeof(kDirty));
  ASSERT_EQ(l2.writeback_line(kAddress, dirty_line.data(), 0, sizeof(kDirty), Mtype::RW, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);

  physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted;
  EXPECT_EQ(l2.flush_line(kAddress, kVmid), rocjitsu::amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(physical->load<uint32_t>(kAddress), kInitial);

  physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete;
  EXPECT_EQ(l2.flush_line(kAddress, kVmid), rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(physical->load<uint32_t>(kAddress), kDirty);
}

TEST(L2CacheTest, FailedDirtyEvictionRestoresVictimForRetry) {
  constexpr uint32_t kVmid = 75;
  constexpr uint64_t kBase = 0x1000;
  constexpr uint64_t kSetStride = static_cast<uint64_t>(L2Cache::LINE_SIZE) * L2Cache::NUM_SETS;
  constexpr uint64_t kSize = kBase + (L2Cache::ASSOCIATIVITY + 1) * kSetStride;
  constexpr uint32_t kInitial = 0x13572468;
  constexpr uint32_t kDirty = 0x24681357;
  constexpr uint32_t kSentinel = 0xdeadbeef;
  auto translator = std::make_shared<IdentityAddressSpace>(kSize);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(kSize);
  physical->store(kBase, kInitial);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  ASSERT_TRUE(gpu_vm.register_translated(kVmid, translator, physical));
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&gpu_vm);

  std::array<uint8_t, L2Cache::LINE_SIZE> dirty_line{};
  std::memcpy(dirty_line.data(), &kDirty, sizeof(kDirty));
  ASSERT_EQ(l2.writeback_line(kBase, dirty_line.data(), 0, sizeof(kDirty), Mtype::RW, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);

  for (uint32_t way = 1; way < L2Cache::ASSOCIATIVITY; ++way) {
    uint32_t observed = 0;
    ASSERT_EQ(l2.read(kBase + static_cast<uint64_t>(way) * kSetStride,
                      reinterpret_cast<uint8_t *>(&observed), sizeof(observed), Mtype::RW, kVmid),
              rocjitsu::amdgpu::VmAccessOutcome::Complete);
  }

  physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted;
  uint32_t replacement = kSentinel;
  EXPECT_EQ(l2.read(kBase + static_cast<uint64_t>(L2Cache::ASSOCIATIVITY) * kSetStride,
                    reinterpret_cast<uint8_t *>(&replacement), sizeof(replacement), Mtype::RW,
                    kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(replacement, kSentinel);
  EXPECT_EQ(physical->load<uint32_t>(kBase), kInitial);

  const uint32_t reads_before_recovery = physical->read_calls;
  uint32_t recovered = 0;
  EXPECT_EQ(
      l2.read(kBase, reinterpret_cast<uint8_t *>(&recovered), sizeof(recovered), Mtype::RW, kVmid),
      rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(recovered, kDirty);
  EXPECT_EQ(physical->read_calls, reads_before_recovery);

  physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete;
  EXPECT_EQ(l2.flush_line(kBase, kVmid), rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(physical->load<uint32_t>(kBase), kDirty);
}

TEST(L2CacheTest, FailedLinkedDirtyEvictionRestoresVictimForRetry) {
  constexpr uint64_t kBase = 0x1000;
  constexpr uint64_t kSetStride = static_cast<uint64_t>(L2Cache::LINE_SIZE) * L2Cache::NUM_SETS;
  constexpr uint64_t kSize = kBase + (L2Cache::ASSOCIATIVITY + 1) * kSetStride;
  constexpr uint32_t kInitial = 0x10203040;
  constexpr uint32_t kDirty = 0x50607080;
  constexpr uint32_t kSentinel = 0xabcdef01;
  LinkedCacheHierarchy hierarchy(kSize);
  ASSERT_TRUE(hierarchy.address_space);
  hierarchy.physical->store(kBase, kInitial);

  std::array<uint8_t, L2Cache::LINE_SIZE> dirty_line{};
  std::memcpy(dirty_line.data(), &kDirty, sizeof(kDirty));
  ASSERT_EQ(hierarchy.l2.writeback_line(kBase, dirty_line.data(), 0, sizeof(kDirty), Mtype::RW,
                                        LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);

  for (uint32_t way = 1; way < L2Cache::ASSOCIATIVITY; ++way) {
    uint32_t observed = 0;
    ASSERT_EQ(hierarchy.l2.read(kBase + static_cast<uint64_t>(way) * kSetStride,
                                reinterpret_cast<uint8_t *>(&observed), sizeof(observed), Mtype::RW,
                                LinkedCacheHierarchy::kVmid),
              rocjitsu::amdgpu::VmAccessOutcome::Complete);
  }

  hierarchy.physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted;
  uint32_t replacement = kSentinel;
  EXPECT_EQ(hierarchy.l2.read(kBase + static_cast<uint64_t>(L2Cache::ASSOCIATIVITY) * kSetStride,
                              reinterpret_cast<uint8_t *>(&replacement), sizeof(replacement),
                              Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(replacement, kSentinel);
  EXPECT_EQ(hierarchy.physical->load<uint32_t>(kBase), kInitial);

  const uint32_t reads_before_recovery = hierarchy.physical->read_calls;
  uint32_t recovered = 0;
  EXPECT_EQ(hierarchy.l2.read(kBase, reinterpret_cast<uint8_t *>(&recovered), sizeof(recovered),
                              Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(recovered, kDirty);
  EXPECT_EQ(hierarchy.physical->read_calls, reads_before_recovery);

  hierarchy.physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete;
  EXPECT_EQ(hierarchy.l2.flush_line(kBase, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(hierarchy.physical->load<uint32_t>(kBase), kDirty);
}

TEST(DeviceCacheCoherenceTest, MaintenancePreservesWritebackVersusInvalidateSemantics) {
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_legacy_maintenance_memory(&memory);
  l2.set_legacy_maintenance_vm(&gpu_vm);
  constexpr uint64_t kDiscardedAddress = 0x9000;
  constexpr uint64_t kPublishedAddress = 0x9080;
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty_line{};
  dirty_line.fill(0x5a);

  l2.writeback_line(kDiscardedAddress, dirty_line.data(), Mtype::RW);
  {
    [[maybe_unused]] rocjitsu::amdgpu::DeviceCacheMaintenanceLease maintenance =
        l2.coherence_domain()->acquire_cache_maintenance(
            rocjitsu::amdgpu::DeviceCacheOperation::Invalidate);
  }
  EXPECT_EQ(memory.read8(kDiscardedAddress), 0u);

  l2.writeback_line(kPublishedAddress, dirty_line.data(), Mtype::RW);
  {
    [[maybe_unused]] rocjitsu::amdgpu::DeviceCacheMaintenanceLease maintenance =
        l2.coherence_domain()->acquire_cache_maintenance(
            rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
  }
  EXPECT_EQ(memory.read8(kPublishedAddress), 0x5au);
}

TEST(DeviceCacheCoherenceTest, MaintenanceRefreshesMemorySideCacheAfterDirectWrite) {
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  HbmController hbm("hbm", &memory);
  MemorySideCache memory_side_cache(
      "memory_side_cache", std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>(), &memory);
  memory_side_cache.set_legacy_maintenance_vm(&gpu_vm);
  simdojo::Link link(/*id=*/0, memory_side_cache.req_port(), hbm.cpl_port(), /*latency=*/0);
  link.set_exec_mode(simdojo::ExecMode::FUNCTIONAL);
  memory_side_cache.req_port()->set_link(&link);
  hbm.cpl_port()->set_link(&link);

  constexpr uint64_t kAddress = 0x9100;
  constexpr uint32_t kInitial = 7;
  constexpr uint32_t kReplacement = 19;
  memory.write32(kAddress, kInitial);
  uint32_t cached = 0;
  memory_side_cache.read(kAddress, reinterpret_cast<uint8_t *>(&cached), sizeof(cached));
  ASSERT_EQ(cached, kInitial);

  {
    [[maybe_unused]] rocjitsu::amdgpu::DeviceCacheMaintenanceLease maintenance =
        memory_side_cache.coherence_domain()->acquire_cache_maintenance(
            rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
    memory.write32(kAddress, kReplacement);
  }

  uint32_t refreshed = 0;
  memory_side_cache.read(kAddress, reinterpret_cast<uint8_t *>(&refreshed), sizeof(refreshed));
  EXPECT_EQ(refreshed, kReplacement);
}

TEST(DeviceCacheCoherenceTest, DomainsAreIsolated) {
  auto first_domain = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  auto second_domain = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  GpuMemory first_memory("first_memory");
  GpuMemory second_memory("second_memory");
  rocjitsu::amdgpu::GpuVm first_gpu_vm;
  rocjitsu::amdgpu::GpuVm second_gpu_vm;
  L2Cache first_l2("first_l2", first_domain);
  L2Cache second_l2("second_l2", second_domain);
  first_l2.set_backing_memory(&first_memory);
  first_l2.set_legacy_maintenance_memory(&first_memory);
  first_l2.set_legacy_maintenance_vm(&first_gpu_vm);
  second_l2.set_backing_memory(&second_memory);
  second_l2.set_legacy_maintenance_memory(&second_memory);
  second_l2.set_legacy_maintenance_vm(&second_gpu_vm);

  const uint64_t first_epoch = first_domain->current_epoch();
  const uint64_t second_epoch = second_domain->current_epoch();
  {
    [[maybe_unused]] auto maintenance = first_domain->acquire_cache_maintenance(
        rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
  }

  EXPECT_GT(first_domain->current_epoch(), first_epoch);
  EXPECT_EQ(second_domain->current_epoch(), second_epoch);
}

TEST(DeviceCacheCoherenceTest, SocsOwnDistinctDomains) {
  rocjitsu::SoC first("first");
  rocjitsu::SoC second("second");
  EXPECT_NE(first.cache_coherence(), second.cache_coherence());
}

TEST(DeviceCacheCoherenceTest, LegacyMaintenanceDoesNotUseL2RequesterPort) {
  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  L2Cache l2("l2", coherence);
  l2.set_backing_memory(&memory);
  l2.set_legacy_maintenance_memory(&memory);
  l2.set_legacy_maintenance_vm(&gpu_vm);

  constexpr uint64_t kAddress = 0x9200;
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty_line{};
  dirty_line.fill(0x6b);
  l2.writeback_line(kAddress, dirty_line.data(), Mtype::RW);

  // Removing the ordinary direct backing leaves the requester port unlinked.
  // Maintenance must still publish through its explicit legacy-only backing.
  l2.set_backing_memory(nullptr);
  {
    [[maybe_unused]] auto maintenance = coherence->acquire_cache_maintenance(
        rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
  }
  EXPECT_EQ(memory.read8(kAddress), 0x6bu);
}

TEST(DeviceCacheCoherenceTest, WritebackMaintenanceRejectsMissingLegacyBacking) {
  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto maintenance = coherence->acquire_cache_maintenance(
            rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
      },
      util::ConfigError);
}

void increment_u32(uint8_t *line, uint32_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, line + offset, sizeof(value));
  ++value;
  std::memcpy(line + offset, &value, sizeof(value));
}

void report_benchmark(std::string_view name, uint64_t operations,
                      std::chrono::steady_clock::duration elapsed) {
  const double total_ns = std::chrono::duration<double, std::nano>(elapsed).count();
  std::cout << "ROCJITSU_BENCHMARK" << " name=" << name << " operations=" << operations
            << " total_ns=" << static_cast<uint64_t>(total_ns) << " ns_per_op=" << std::fixed
            << std::setprecision(3) << total_ns / static_cast<double>(operations) << '\n';
}

void run_cross_l2_atomic_benchmark(std::string_view name, bool same_address) {
  GpuMemory memory("memory");
  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  L2Cache l2a("l2a", coherence);
  L2Cache l2b("l2b", coherence);
  l2a.set_backing_memory(&memory);
  l2b.set_backing_memory(&memory);

  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kIterations = 25'000;
  constexpr uint64_t kBase = 0x800000;
  constexpr uint64_t kOperations = static_cast<uint64_t>(kThreads) * kIterations;

  for (uint32_t tid = 0; tid < kThreads; ++tid)
    memory.write32(kBase + static_cast<uint64_t>(tid) * L2Cache::LINE_SIZE, 0);

  std::barrier ready(kThreads + 1);
  std::barrier start(kThreads + 1);
  std::barrier done(kThreads + 1);
  std::array<L2Cache *, 2> l2s = {&l2a, &l2b};
  std::atomic<uint64_t> failed_operations{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      L2Cache *l2 = l2s[tid % l2s.size()];
      const uint64_t target =
          kBase + (same_address ? 0 : static_cast<uint64_t>(tid) * L2Cache::LINE_SIZE);
      ready.arrive_and_wait();
      start.arrive_and_wait();
      for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
        if (l2->atomic_rmw(target, sizeof(uint32_t), increment_u32) !=
            rocjitsu::amdgpu::VmAccessOutcome::Complete)
          failed_operations.fetch_add(1, std::memory_order_relaxed);
      }
      done.arrive_and_wait();
    });
  }

  ready.arrive_and_wait();
  const auto begin = std::chrono::steady_clock::now();
  start.arrive_and_wait();
  done.arrive_and_wait();
  const auto end = std::chrono::steady_clock::now();
  for (auto &worker : workers)
    worker.join();
  EXPECT_EQ(failed_operations.load(std::memory_order_relaxed), 0u);

  if (same_address) {
    EXPECT_EQ(memory.read32(kBase), kOperations);
  } else {
    uint64_t observed_operations = 0;
    for (uint32_t tid = 0; tid < kThreads; ++tid) {
      const uint32_t observed =
          memory.read32(kBase + static_cast<uint64_t>(tid) * L2Cache::LINE_SIZE);
      EXPECT_EQ(observed, kIterations) << "thread=" << tid;
      observed_operations += observed;
    }
    EXPECT_EQ(observed_operations, kOperations);
  }
  report_benchmark(name, kOperations, end - begin);
}

void run_atomic_hierarchy_benchmark(std::string_view name, uint32_t hierarchy_count) {
  GpuMemory memory("memory");
  std::vector<std::unique_ptr<L2Cache>> l2s;
  std::vector<std::unique_ptr<L1ScalarCache>> scalar_l1s;
  std::vector<std::unique_ptr<L1VectorCache>> vector_l1s;
  for (uint32_t i = 0; i < hierarchy_count; ++i) {
    auto l2 = std::make_unique<L2Cache>("l2_" + std::to_string(i));
    l2->set_backing_memory(&memory);
    scalar_l1s.push_back(std::make_unique<L1ScalarCache>(l2.get()));
    vector_l1s.push_back(std::make_unique<L1VectorCache>(l2.get()));
    l2s.push_back(std::move(l2));
  }

  constexpr uint64_t kAddr = 0x880000;
  constexpr uint32_t kIterations = 100'000;
  memory.write32(kAddr, 0);
  const auto begin = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < kIterations; ++i)
    ASSERT_EQ(l2s.front()->atomic_rmw(kAddr, sizeof(uint32_t), increment_u32),
              rocjitsu::amdgpu::VmAccessOutcome::Complete);
  const auto end = std::chrono::steady_clock::now();

  EXPECT_EQ(memory.read32(kAddr), kIterations);
  report_benchmark(name, kIterations, end - begin);
}

void run_scalar_l1_hit_benchmark() {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  L1ScalarCache l1(&l2);
  constexpr uint64_t kAddr = 0x890000;
  constexpr uint32_t kIterations = 1'000'000;
  memory.write32(kAddr, 17);
  uint32_t value = 0;
  l1.load(kAddr, 1, &value);

  const auto begin = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < kIterations; ++i)
    l1.load(kAddr, 1, &value);
  const auto end = std::chrono::steady_clock::now();

  EXPECT_EQ(value, 17u);
  report_benchmark("scalar_l1_hit", kIterations, end - begin);
}

void run_vector_l1_hit_benchmark() {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  L1VectorCache l1(&l2);
  constexpr uint64_t kAddr = 0x8A0000;
  constexpr uint32_t kIterations = 500'000;
  memory.write32(kAddr, 23);
  const uint64_t addrs[1] = {kAddr};
  std::array<uint8_t, sizeof(uint32_t)> value{};
  l1.load(addrs, 1, sizeof(uint32_t), 1, value.data(), Mtype::RW, false, false, 1);

  const auto begin = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < kIterations; ++i)
    l1.load(addrs, 1, sizeof(uint32_t), 1, value.data(), Mtype::RW, false, false, 1);
  const auto end = std::chrono::steady_clock::now();

  uint32_t observed = 0;
  std::memcpy(&observed, value.data(), sizeof(observed));
  EXPECT_EQ(observed, 23u);
  report_benchmark("vector_l1_hit", kIterations, end - begin);
}

void run_invalidate_range_benchmark(std::string_view name, uint32_t lines, uint32_t iterations) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint64_t kBase = 0xa00000;
  std::array<uint8_t, L2Cache::LINE_SIZE> replacement{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  for (uint32_t line = 0; line < lines; ++line) {
    const uint64_t address = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
    replacement.fill(static_cast<uint8_t>(line + 0x40));
    memory.write_block(address, std::span<const uint8_t>(replacement));
  }

  std::chrono::steady_clock::duration elapsed{};
  uint64_t refill_checksum = 0;
  uint64_t expected_refill_checksum = 0;
  for (uint32_t line = 0; line < lines; ++line)
    expected_refill_checksum += static_cast<uint8_t>(line + 0x40);
  expected_refill_checksum *= iterations;

  for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
    // Refill outside the timed interval so every measured call invalidates
    // resident lines rather than repeatedly walking an empty cache.
    for (uint32_t line = 0; line < lines; ++line) {
      const uint64_t address = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
      l2.read(address, actual.data(), actual.size());
      refill_checksum += actual.front();
    }

    const auto begin = std::chrono::steady_clock::now();
    l2.invalidate_range(kBase, static_cast<uint64_t>(lines) * L2Cache::LINE_SIZE, 0);
    elapsed += std::chrono::steady_clock::now() - begin;
  }

  uint64_t checksum = 0;
  uint64_t expected_checksum = 0;
  for (uint32_t line = 0; line < lines; ++line) {
    const uint64_t address = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
    replacement.fill(static_cast<uint8_t>(line + 0x40));
    l2.read(address, actual.data(), actual.size());
    EXPECT_EQ(actual, replacement) << "line=" << line;
    checksum += actual.front();
    expected_checksum += replacement.front();
  }
  EXPECT_EQ(refill_checksum, expected_refill_checksum);
  EXPECT_EQ(checksum, expected_checksum);
  report_benchmark(name, iterations, elapsed);
}

class ScopedFd {
public:
  explicit ScopedFd(int fd) : fd_(fd) {}
  ScopedFd(const ScopedFd &) = delete;
  ScopedFd &operator=(const ScopedFd &) = delete;
  ~ScopedFd() {
    if (fd_ >= 0)
      close(fd_);
  }

  int get() const { return fd_; }

private:
  int fd_;
};

class ScopedMapping {
public:
  ScopedMapping(void *address, size_t size) : address_(address), size_(size) {}
  ScopedMapping(const ScopedMapping &) = delete;
  ScopedMapping &operator=(const ScopedMapping &) = delete;
  ~ScopedMapping() {
    if (address_ != MAP_FAILED)
      munmap(address_, size_);
  }

  uint8_t *data() const { return static_cast<uint8_t *>(address_); }

private:
  void *address_;
  size_t size_;
};

TEST(DeviceCacheCoherenceTest, FailedL2WritebackRetainsOnlyUnpublishedDirtyBytes) {
  constexpr uint32_t kVmid = 41;
  constexpr uint64_t kGpuAddress = 0x140000;
  constexpr uint32_t kFirstOffset = 0;
  constexpr uint32_t kSecondOffset = 64;
  constexpr uint32_t kFirstValue = 0x11223344;
  constexpr uint32_t kSecondValue = 0x55667788;
  constexpr uint32_t kReplacementFirstValue = 0xaabbccdd;

  void *first_raw = mmap(nullptr, GpuMemory::PAGE_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(first_raw, MAP_FAILED);
  ScopedMapping first_mapping(first_raw, GpuMemory::PAGE_SIZE);
  void *second_raw = mmap(nullptr, GpuMemory::PAGE_SIZE, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(second_raw, MAP_FAILED);
  ScopedMapping second_mapping(second_raw, GpuMemory::PAGE_SIZE);

  rocjitsu::KfdProcess process(kVmid);
  process.map_pages(kGpuAddress + kFirstOffset, first_mapping.data(), sizeof(uint32_t));
  process.map_pages(kGpuAddress + kSecondOffset, second_mapping.data(), sizeof(uint32_t));
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  rocjitsu::amdgpu::LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  const rocjitsu::amdgpu::AddressSpaceHandle address_space = legacy_vm.register_address_space(
      kVmid, &process.page_table_, &process.page_table_mutex_, process.page_table_generation());
  ASSERT_TRUE(address_space);
  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  l2.set_backing_memory(&memory);
  l2.set_legacy_maintenance_memory(&memory);
  l2.set_legacy_maintenance_vm(&gpu_vm);
  l2.set_gpu_vm(&gpu_vm);

  std::array<uint8_t, L2Cache::LINE_SIZE> dirty_line{};
  std::memcpy(dirty_line.data() + kFirstOffset, &kFirstValue, sizeof(kFirstValue));
  std::memcpy(dirty_line.data() + kSecondOffset, &kSecondValue, sizeof(kSecondValue));
  l2.writeback_line(kGpuAddress, dirty_line.data(), kFirstOffset, sizeof(kFirstValue), Mtype::RW,
                    kVmid);
  l2.writeback_line(kGpuAddress, dirty_line.data(), kSecondOffset, sizeof(kSecondValue), Mtype::RW,
                    kVmid);

  ASSERT_EQ(mprotect(second_mapping.data(), GpuMemory::PAGE_SIZE, PROT_READ), 0);
  const uint64_t epoch_before_failure = coherence->current_epoch();
  bool protected_operation_ran = false;
  EXPECT_THROW(
      {
        [[maybe_unused]] auto maintenance = coherence->acquire_cache_maintenance(
            rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
        protected_operation_ran = true;
      },
      util::Exception);
  EXPECT_FALSE(protected_operation_ran);
  EXPECT_EQ(coherence->current_epoch(), epoch_before_failure);

  uint32_t first_published = 0;
  uint32_t second_unpublished = 0;
  std::memcpy(&first_published, first_mapping.data(), sizeof(first_published));
  std::memcpy(&second_unpublished, second_mapping.data(), sizeof(second_unpublished));
  EXPECT_EQ(first_published, kFirstValue);
  EXPECT_EQ(second_unpublished, 0u);

  std::memcpy(first_mapping.data(), &kReplacementFirstValue, sizeof(kReplacementFirstValue));
  ASSERT_EQ(mprotect(second_mapping.data(), GpuMemory::PAGE_SIZE, PROT_READ | PROT_WRITE), 0);
  {
    [[maybe_unused]] auto maintenance = coherence->acquire_cache_maintenance(
        rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
  }

  uint32_t retained_first = 0;
  uint32_t retried_second = 0;
  std::memcpy(&retained_first, first_mapping.data(), sizeof(retained_first));
  std::memcpy(&retried_second, second_mapping.data(), sizeof(retried_second));
  EXPECT_EQ(retained_first, kReplacementFirstValue);
  EXPECT_EQ(retried_second, kSecondValue);
  EXPECT_TRUE(legacy_vm.unregister_vmid(kVmid));
}

TEST(DeviceCacheCoherenceTest, FailedMemorySideWritebackRetainsDirtyLineForRetry) {
  constexpr uint32_t kVmid = 42;
  constexpr uint64_t kGpuAddress = 0x150000;

  void *raw_mapping = mmap(nullptr, GpuMemory::PAGE_SIZE, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(raw_mapping, MAP_FAILED);
  ScopedMapping mapping(raw_mapping, GpuMemory::PAGE_SIZE);

  rocjitsu::KfdProcess process(kVmid);
  process.map_pages(kGpuAddress, mapping.data(), GpuMemory::PAGE_SIZE);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  rocjitsu::amdgpu::LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  const rocjitsu::amdgpu::AddressSpaceHandle address_space = legacy_vm.register_address_space(
      kVmid, &process.page_table_, &process.page_table_mutex_, process.page_table_generation());
  ASSERT_TRUE(address_space);
  HbmController hbm("hbm", &memory);
  hbm.set_gpu_vm(&gpu_vm);
  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  MemorySideCache memory_side_cache("memory_side_cache", coherence, &memory);
  memory_side_cache.set_legacy_maintenance_vm(&gpu_vm);
  simdojo::Link link(/*id=*/0, memory_side_cache.req_port(), hbm.cpl_port(), /*latency=*/0);
  link.set_exec_mode(simdojo::ExecMode::FUNCTIONAL);
  memory_side_cache.req_port()->set_link(&link);
  hbm.cpl_port()->set_link(&link);

  std::array<uint8_t, MemorySideCache::LINE_SIZE> dirty_line{};
  dirty_line.fill(0x6d);
  rocjitsu::amdgpu::MemorySideCacheTestAccess::store_dirty_line(memory_side_cache, kGpuAddress,
                                                                dirty_line, kVmid);
  ASSERT_EQ(mprotect(mapping.data(), GpuMemory::PAGE_SIZE, PROT_READ), 0);

  bool protected_operation_ran = false;
  EXPECT_THROW(
      {
        [[maybe_unused]] auto maintenance = coherence->acquire_cache_maintenance(
            rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
        protected_operation_ran = true;
      },
      util::Exception);
  EXPECT_FALSE(protected_operation_ran);
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_dirty(memory_side_cache,
                                                                         kGpuAddress, kVmid));

  ASSERT_EQ(mprotect(mapping.data(), GpuMemory::PAGE_SIZE, PROT_READ | PROT_WRITE), 0);
  {
    [[maybe_unused]] auto maintenance = coherence->acquire_cache_maintenance(
        rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
  }
  EXPECT_FALSE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_dirty(memory_side_cache,
                                                                          kGpuAddress, kVmid));
  EXPECT_TRUE(std::equal(dirty_line.begin(), dirty_line.end(), mapping.data()));
  EXPECT_TRUE(legacy_vm.unregister_vmid(kVmid));
}

TEST(DeviceCacheCoherenceTest, FailedMemorySideEvictionRestoresDirtyVictimForRetry) {
  constexpr uint32_t kFirstVmid = 100;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kInitial = 0x11223344;
  constexpr uint32_t kDirty = 0x55667788;
  constexpr uint32_t kSentinel = 0xa5a55a5a;
  auto translator = std::make_shared<IdentityAddressSpace>(0x2000);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  physical->store(kAddress, kInitial);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  for (uint32_t way = 0; way <= MemorySideCache::ASSOCIATIVITY; ++way)
    ASSERT_TRUE(gpu_vm.register_translated(kFirstVmid + way, translator, physical));

  HbmController hbm("hbm", &memory, &gpu_vm);
  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  MemorySideCache memory_side_cache("memory_side_cache", coherence, &memory);
  simdojo::Link link(/*id=*/0, memory_side_cache.req_port(), hbm.cpl_port(), /*latency=*/0);
  link.set_exec_mode(simdojo::ExecMode::FUNCTIONAL);
  memory_side_cache.req_port()->set_link(&link);
  hbm.cpl_port()->set_link(&link);

  std::array<uint8_t, MemorySideCache::LINE_SIZE> dirty_line{};
  std::memcpy(dirty_line.data(), &kDirty, sizeof(kDirty));
  rocjitsu::amdgpu::MemorySideCacheTestAccess::store_dirty_line(memory_side_cache, kAddress,
                                                                dirty_line, kFirstVmid);
  for (uint32_t way = 1; way < MemorySideCache::ASSOCIATIVITY; ++way) {
    uint32_t observed = 0;
    ASSERT_EQ(memory_side_cache.read(kAddress, reinterpret_cast<uint8_t *>(&observed),
                                     sizeof(observed), kFirstVmid + way),
              rocjitsu::amdgpu::VmAccessOutcome::Complete);
  }

  physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted;
  uint32_t replacement = kSentinel;
  EXPECT_EQ(memory_side_cache.read(kAddress, reinterpret_cast<uint8_t *>(&replacement),
                                   sizeof(replacement),
                                   kFirstVmid + MemorySideCache::ASSOCIATIVITY),
            rocjitsu::amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(replacement, kSentinel);
  EXPECT_EQ(physical->load<uint32_t>(kAddress), kInitial);
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_dirty(memory_side_cache,
                                                                         kAddress, kFirstVmid));

  const uint32_t reads_before_recovery = physical->read_calls;
  uint32_t recovered = 0;
  EXPECT_EQ(memory_side_cache.read(kAddress, reinterpret_cast<uint8_t *>(&recovered),
                                   sizeof(recovered), kFirstVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(recovered, kDirty);
  EXPECT_EQ(physical->read_calls, reads_before_recovery);

  physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete;
  EXPECT_EQ(memory_side_cache.flush_all(), rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(physical->load<uint32_t>(kAddress), kDirty);
}

class FunctionalMemoryPort : public simdojo::Component {
public:
  FunctionalMemoryPort(std::string name, uint64_t base, size_t size)
      : simdojo::Component(std::move(name)), base_(base), bytes_(size) {
    port_ = add_port(std::make_unique<simdojo::Port>("memory", 0, this, simdojo::PortDirection::IN,
                                                     simdojo::PortProtocol::MEMORY));
    port_->set_handler([this](simdojo::Tick, simdojo::Message *message) {
      assert(message != nullptr);
      const auto &header = message->header();
      assert(header.addr >= base_);
      const uint64_t offset = header.addr - base_;
      assert(offset + header.size_bytes <= bytes_.size());
      auto *payload = reinterpret_cast<uint8_t *>(message->payload());
      assert(payload != nullptr);
      if (header.op == simdojo::MessageOp::READ) {
        std::memcpy(payload, bytes_.data() + offset, header.size_bytes);
      } else if (header.op == simdojo::MessageOp::WRITE) {
        std::memcpy(bytes_.data() + offset, payload, header.size_bytes);
      } else {
        assert(header.op == simdojo::MessageOp::ATOMIC);
        auto *mutation = reinterpret_cast<simdojo::MemoryAtomicMutation *>(message->payload());
        (*mutation)(std::span<std::byte>(reinterpret_cast<std::byte *>(bytes_.data() + offset),
                                         header.size_bytes));
      }
    });
  }

  simdojo::Port *port() { return port_; }

  void write32(uint64_t addr, uint32_t value) {
    assert(addr >= base_ && addr - base_ + sizeof(value) <= bytes_.size());
    std::memcpy(bytes_.data() + (addr - base_), &value, sizeof(value));
  }

  uint32_t read32(uint64_t addr) const {
    assert(addr >= base_ && addr - base_ + sizeof(uint32_t) <= bytes_.size());
    uint32_t value = 0;
    std::memcpy(&value, bytes_.data() + (addr - base_), sizeof(value));
    return value;
  }

  uint8_t read8(uint64_t addr) const {
    assert(addr >= base_ && addr - base_ < bytes_.size());
    return bytes_[addr - base_];
  }

  void write8(uint64_t addr, uint8_t value) {
    assert(addr >= base_ && addr - base_ < bytes_.size());
    bytes_[addr - base_] = value;
  }

private:
  uint64_t base_;
  std::vector<uint8_t> bytes_;
  simdojo::Port *port_ = nullptr;
};

TEST(L2CacheTest, ClockedLinkedMemoryRequestIsRejectedSynchronously) {
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kSentinel = 0xcafef00d;
  L2Cache l2("l2");
  FunctionalMemoryPort backing("backing", kAddress, L2Cache::LINE_SIZE);
  simdojo::Link link(/*id=*/0, l2.req_port(), backing.port(), /*latency=*/1);
  l2.req_port()->set_link(&link);
  backing.port()->set_link(&link);

  uint32_t observed = kSentinel;
  EXPECT_EQ(l2.read(kAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed)),
            rocjitsu::amdgpu::VmAccessOutcome::Malformed);
  EXPECT_EQ(observed, kSentinel);
}

TEST(L2CacheTest, ClockedMemorySideRequestIsRejectedSynchronously) {
  constexpr uint64_t kAddress = 0x1100;
  constexpr uint32_t kSentinel = 0x0badc0de;
  MemorySideCache memory_side_cache("memory_side_cache");
  FunctionalMemoryPort backing("backing", kAddress, MemorySideCache::LINE_SIZE);
  simdojo::Link link(/*id=*/0, memory_side_cache.req_port(), backing.port(), /*latency=*/1);
  memory_side_cache.req_port()->set_link(&link);
  backing.port()->set_link(&link);

  uint32_t observed = kSentinel;
  EXPECT_EQ(
      memory_side_cache.read(kAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed)),
      rocjitsu::amdgpu::VmAccessOutcome::Malformed);
  EXPECT_EQ(observed, kSentinel);
}

class L1CacheMtypeTest : public testing::Test {
protected:
  static constexpr uint32_t kVmid = 7;
  static constexpr uint64_t kBase = 0x100000;
  static constexpr uint64_t kAddr = kBase + GpuMemory::PAGE_SIZE - sizeof(uint32_t);
  static constexpr uint32_t kFirst = 0x11112222;
  static constexpr uint32_t kSecond = 0x33334444;
  static constexpr uint32_t kFirstReplacement = 0x55556666;
  static constexpr uint32_t kSecondReplacement = 0x77778888;
  static constexpr std::array<uint32_t, 2> kValues = {kFirst, kSecond};

  void map_pages(Mtype first_mtype, Mtype second_mtype) {
    process_.map_pages(kBase, first_page_.data(), first_page_.size(), first_mtype);
    process_.map_pages(kBase + GpuMemory::PAGE_SIZE, second_page_.data(), second_page_.size(),
                       second_mtype);
    address_space_ = legacy_vm_.register_address_space(
        kVmid, &process_.page_table_, &process_.page_table_mutex_, process_.page_table_generation(),
        process_.page_table_request_mutex());
    ASSERT_TRUE(address_space_);
    l2_.set_backing_memory(&memory_);
    l2_.set_gpu_vm(&gpu_vm_);
  }

  void write_words(uint32_t first, uint32_t second) {
    std::memcpy(first_page_.data() + GpuMemory::PAGE_SIZE - sizeof(first), &first, sizeof(first));
    std::memcpy(second_page_.data(), &second, sizeof(second));
  }

  std::array<uint32_t, 2> read_words() const {
    std::array<uint32_t, 2> result{};
    std::memcpy(&result[0], first_page_.data() + GpuMemory::PAGE_SIZE - sizeof(result[0]),
                sizeof(result[0]));
    std::memcpy(&result[1], second_page_.data(), sizeof(result[1]));
    return result;
  }

  std::array<uint8_t, GpuMemory::PAGE_SIZE> first_page_{};
  std::array<uint8_t, GpuMemory::PAGE_SIZE> second_page_{};
  rocjitsu::KfdProcess process_{kVmid};
  GpuMemory memory_{"memory"};
  rocjitsu::amdgpu::GpuVm gpu_vm_;
  rocjitsu::amdgpu::LegacyGpuVmAdapter legacy_vm_{gpu_vm_, &memory_};
  rocjitsu::amdgpu::AddressSpaceHandle address_space_;
  L2Cache l2_{"l2"};
};

TEST_F(L1CacheMtypeTest, ScalarLoadKeepsPageSpecificMtypeAcrossBoundary) {
  write_words(kFirst, kSecond);
  map_pages(Mtype::RW, Mtype::UC);
  L1ScalarCache l1(&l2_);
  l1.set_gpu_vm(&gpu_vm_);

  std::array<uint32_t, 2> result{};
  l1.load(kAddr, result.size(), result.data(), kVmid);
  ASSERT_EQ(result, kValues);

  write_words(kFirstReplacement, kSecondReplacement);
  l1.load(kAddr, result.size(), result.data(), kVmid);

  EXPECT_EQ(result[0], kFirst);
  EXPECT_EQ(result[1], kSecondReplacement);
}

TEST_F(L1CacheMtypeTest, ScalarLoadBytesKeepsPageSpecificMtypeAcrossBoundary) {
  write_words(kFirst, kSecond);
  map_pages(Mtype::RW, Mtype::UC);
  L1ScalarCache l1(&l2_);
  l1.set_gpu_vm(&gpu_vm_);

  std::array<uint32_t, 2> result{};
  l1.load_bytes(kAddr, sizeof(result), reinterpret_cast<uint8_t *>(result.data()), kVmid);
  ASSERT_EQ(result, kValues);

  write_words(kFirstReplacement, kSecondReplacement);
  l1.load_bytes(kAddr, sizeof(result), reinterpret_cast<uint8_t *>(result.data()), kVmid);

  EXPECT_EQ(result[0], kFirst);
  EXPECT_EQ(result[1], kSecondReplacement);
}

TEST_F(L1CacheMtypeTest, PageMtypeMutationRefreshesLiveResolver) {
  write_words(kFirst, kSecond);
  map_pages(Mtype::RW, Mtype::UC);
  L1ScalarCache l1(&l2_);
  l1.set_gpu_vm(&gpu_vm_);

  uint32_t result = 0;
  l1.load(kAddr, 1, &result, kVmid);
  ASSERT_EQ(result, kFirst);

  RequestMtypeResolver request(&gpu_vm_, kVmid);
  ASSERT_EQ(request.at(kAddr), Mtype::RW);
  process_.set_page_mtype(kBase, GpuMemory::PAGE_SIZE, Mtype::CC);
  write_words(kFirstReplacement, kSecond);
  EXPECT_EQ(request.at(kAddr + 1), Mtype::CC);

  l1.load(kAddr, 1, &result, kVmid);
  EXPECT_EQ(result, kFirstReplacement);
}

TEST_F(L1CacheMtypeTest, VmidRebindingRefreshesNewResolverAndUsesNewPolicy) {
  write_words(kFirst, kSecond);
  map_pages(Mtype::RW, Mtype::RW);
  L1ScalarCache l1(&l2_);
  l1.set_gpu_vm(&gpu_vm_);

  uint32_t result = 0;
  l1.load(kAddr, 1, &result, kVmid);
  ASSERT_EQ(result, kFirst);

  rocjitsu::KfdProcess replacement_process(kVmid);
  std::array<uint8_t, GpuMemory::PAGE_SIZE> replacement_page{};
  std::memcpy(replacement_page.data() + GpuMemory::PAGE_SIZE - sizeof(kFirstReplacement),
              &kFirstReplacement, sizeof(kFirstReplacement));
  replacement_process.map_pages(kBase, replacement_page.data(), replacement_page.size(), Mtype::UC);

  {
    RequestMtypeResolver request(&gpu_vm_, kVmid);
    ASSERT_EQ(request.at(kAddr), Mtype::RW);
  }
  ASSERT_TRUE(legacy_vm_.unregister_vmid(kVmid));
  address_space_ = {};
  address_space_ = legacy_vm_.register_address_space(
      kVmid, &replacement_process.page_table_, &replacement_process.page_table_mutex_,
      replacement_process.page_table_generation(), replacement_process.page_table_request_mutex());
  ASSERT_TRUE(address_space_);
  RequestMtypeResolver replacement_request(&gpu_vm_, kVmid);
  EXPECT_EQ(replacement_request.at(kAddr + 1), Mtype::UC);

  l1.load(kAddr, 1, &result, kVmid);
  EXPECT_EQ(result, kFirstReplacement);
  EXPECT_TRUE(legacy_vm_.unregister_vmid(kVmid));
  address_space_ = {};
}

TEST_F(L1CacheMtypeTest, VmidUnregistrationPreservesLiveResolverSnapshot) {
  map_pages(Mtype::UC, Mtype::UC);

  RequestMtypeResolver request(&gpu_vm_, kVmid);
  ASSERT_EQ(request.at(kAddr), Mtype::UC);
  ASSERT_TRUE(legacy_vm_.unregister_vmid(kVmid));
  address_space_ = {};
  EXPECT_EQ(request.at(kAddr + 1), Mtype::UC);
  EXPECT_FALSE(gpu_vm_.snapshot_vmid(kVmid));

  RequestMtypeResolver new_request(&gpu_vm_, kVmid);
  EXPECT_EQ(new_request.at(kAddr + 1), Mtype::RW);
}

TEST_F(L1CacheMtypeTest, VectorLoadKeepsPageSpecificMtypeAcrossBoundary) {
  write_words(kFirst, kSecond);
  map_pages(Mtype::RW, Mtype::UC);
  L1VectorCache l1(&l2_);
  l1.set_gpu_vm(&gpu_vm_);

  const uint64_t addrs[] = {kAddr};
  std::array<uint32_t, 2> result{};
  l1.load(addrs, /*lane_mask=*/1, sizeof(uint32_t), result.size(),
          reinterpret_cast<uint8_t *>(result.data()), Mtype::RW, /*non_temporal=*/false,
          /*request_l1_bypass=*/false, /*wf_size=*/1, kVmid);
  ASSERT_EQ(result, kValues);

  write_words(kFirstReplacement, kSecondReplacement);
  l1.load(addrs, /*lane_mask=*/1, sizeof(uint32_t), result.size(),
          reinterpret_cast<uint8_t *>(result.data()), Mtype::RW, /*non_temporal=*/false,
          /*request_l1_bypass=*/false, /*wf_size=*/1, kVmid);

  EXPECT_EQ(result[0], kFirst);
  EXPECT_EQ(result[1], kSecondReplacement);
}

TEST_F(L1CacheMtypeTest, ScalarStoreKeepsPageSpecificMtypeAcrossBoundary) {
  map_pages(Mtype::RW, Mtype::UC);
  L1ScalarCache l1(&l2_);
  l1.set_gpu_vm(&gpu_vm_);

  l1.store(kAddr, kValues.size(), kValues.data(), kVmid);

  EXPECT_EQ(read_words(), kValues);
  EXPECT_EQ(l2_.backing_read_transactions(), 1u);
}

TEST_F(L1CacheMtypeTest, VectorStoreKeepsPageSpecificMtypeAcrossBoundary) {
  map_pages(Mtype::RW, Mtype::UC);
  L1VectorCache l1(&l2_);
  l1.set_gpu_vm(&gpu_vm_);

  const uint64_t addrs[] = {kAddr};
  l1.store(addrs, /*lane_mask=*/1, sizeof(uint32_t), kValues.size(),
           reinterpret_cast<const uint8_t *>(kValues.data()), Mtype::RW,
           /*non_temporal=*/false, /*wf_size=*/1, kVmid);

  EXPECT_EQ(read_words(), kValues);
  EXPECT_EQ(l2_.backing_read_transactions(), 1u);
}

TEST_F(L1CacheMtypeTest, VectorStoreKeepsUcThenRwMtypeAcrossBoundary) {
  map_pages(Mtype::UC, Mtype::RW);
  L1VectorCache l1(&l2_);
  l1.set_gpu_vm(&gpu_vm_);

  const uint64_t addrs[] = {kAddr};
  l1.store(addrs, /*lane_mask=*/1, sizeof(uint32_t), kValues.size(),
           reinterpret_cast<const uint8_t *>(kValues.data()), Mtype::RW,
           /*non_temporal=*/false, /*wf_size=*/1, kVmid);

  EXPECT_EQ(read_words(), kValues);
  EXPECT_EQ(l2_.backing_read_transactions(), 1u);
}

TEST(L2CacheThreadingTest, ConcurrentDifferentSetWritesArePreserved) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kLinesPerThread = 128;
  constexpr uint64_t kBase = 0x100000;

  std::barrier start(kThreads);
  std::vector<std::thread> workers;
  workers.reserve(kThreads);

  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      std::array<uint8_t, L2Cache::LINE_SIZE> line{};
      start.arrive_and_wait();
      for (uint32_t i = 0; i < kLinesPerThread; ++i) {
        const uint64_t addr =
            kBase + (static_cast<uint64_t>(i) * kThreads + tid) * L2Cache::LINE_SIZE;
        for (uint32_t b = 0; b < line.size(); ++b)
          line[b] = static_cast<uint8_t>((tid << 4) ^ i ^ b);
        l2.write(addr, line.data(), line.size());
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  std::array<uint8_t, L2Cache::LINE_SIZE> expected{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    for (uint32_t i = 0; i < kLinesPerThread; ++i) {
      const uint64_t addr =
          kBase + (static_cast<uint64_t>(i) * kThreads + tid) * L2Cache::LINE_SIZE;
      for (uint32_t b = 0; b < expected.size(); ++b)
        expected[b] = static_cast<uint8_t>((tid << 4) ^ i ^ b);
      l2.read(addr, actual.data(), actual.size());
      EXPECT_EQ(actual, expected) << "addr=0x" << std::hex << addr;
    }
  }
}

TEST(L2CacheThreadingTest, ConcurrentSameSetAccessesPreserveLines) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kIterations = 256;
  constexpr uint64_t kBase = 0x180000;
  constexpr uint64_t kSetStride = static_cast<uint64_t>(L2Cache::NUM_SETS) * L2Cache::LINE_SIZE;

  std::barrier start(kThreads);
  std::atomic<uint64_t> mismatches{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      const uint64_t addr = kBase + tid * kSetStride;
      std::array<uint8_t, L2Cache::LINE_SIZE> line{};
      std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
      start.arrive_and_wait();
      for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
        line.fill(static_cast<uint8_t>((tid << 4) ^ iteration));
        l2.write(addr, line.data(), line.size());
        l2.read(addr, actual.data(), actual.size());
        if (actual != line)
          mismatches.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(mismatches.load(std::memory_order_relaxed), 0u);
}

TEST(L2CacheThreadingTest, ConcurrentAtomicRmwSameLineIsSerialized) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kIterations = 1000;
  constexpr uint64_t kTarget = 0x200000;

  memory.write32(kTarget, 0);

  std::barrier start(kThreads);
  std::atomic<uint64_t> failed_operations{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);

  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&] {
      start.arrive_and_wait();
      for (uint32_t i = 0; i < kIterations; ++i) {
        if (l2.atomic_rmw(kTarget, sizeof(uint32_t), [](uint8_t *line, uint32_t offset) {
              uint32_t value = 0;
              std::memcpy(&value, line + offset, sizeof(value));
              ++value;
              std::memcpy(line + offset, &value, sizeof(value));
            }) != rocjitsu::amdgpu::VmAccessOutcome::Complete)
          failed_operations.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(failed_operations.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(memory.read32(kTarget), kThreads * kIterations);
}

TEST(L2CacheThreadingTest, CrossL2AtomicRmwSameAddressIsSerialized) {
  GpuMemory memory("memory");
  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  L2Cache l2a("l2a", coherence);
  L2Cache l2b("l2b", coherence);
  l2a.set_backing_memory(&memory);
  l2b.set_backing_memory(&memory);

  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kIterations = 1000;
  constexpr uint64_t kTarget = 0x280000;

  memory.write32(kTarget, 0);

  std::array<L2Cache *, 2> l2s = {&l2a, &l2b};
  std::barrier start(kThreads);
  std::atomic<uint64_t> failed_operations{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);

  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      auto *l2 = l2s[tid % l2s.size()];
      start.arrive_and_wait();
      for (uint32_t i = 0; i < kIterations; ++i) {
        if (l2->atomic_rmw(kTarget, sizeof(uint32_t), [](uint8_t *line, uint32_t offset) {
              uint32_t value = 0;
              std::memcpy(&value, line + offset, sizeof(value));
              ++value;
              std::memcpy(line + offset, &value, sizeof(value));
            }) != rocjitsu::amdgpu::VmAccessOutcome::Complete)
          failed_operations.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(failed_operations.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(memory.read32(kTarget), kThreads * kIterations);
}

TEST(L2CacheThreadingTest, CrossL2AtomicRmwAliasedVasIsSerialized) {
  GpuMemory memory("memory");
  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  L2Cache l2a("l2a", coherence);
  L2Cache l2b("l2b", coherence);
  l2a.set_backing_memory(&memory);
  l2b.set_backing_memory(&memory);

  constexpr uint32_t kVmidA = 7;
  constexpr uint32_t kVmidB = 8;
  constexpr uint64_t kVaA = 0x100000;
  constexpr uint64_t kVaB = 0x201000;
  constexpr uint64_t kOffset = 64;
  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kIterations = 1000;

  rocjitsu::KfdProcess process_a(kVmidA);
  rocjitsu::KfdProcess process_b(kVmidB);
  std::array<uint8_t, GpuMemory::PAGE_SIZE> backing{};
  process_a.map_pages(kVaA, backing.data(), backing.size());
  process_b.map_pages(kVaB, backing.data(), backing.size());
  rocjitsu::amdgpu::GpuVm gpu_vm;
  rocjitsu::amdgpu::LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  ASSERT_TRUE(legacy_vm.register_address_space(kVmidA, &process_a.page_table_,
                                               &process_a.page_table_mutex_,
                                               process_a.page_table_generation()));
  ASSERT_TRUE(legacy_vm.register_address_space(kVmidB, &process_b.page_table_,
                                               &process_b.page_table_mutex_,
                                               process_b.page_table_generation()));
  l2a.set_gpu_vm(&gpu_vm);
  l2b.set_gpu_vm(&gpu_vm);

  std::barrier start(kThreads);
  std::atomic<uint64_t> failed_operations{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      L2Cache &l2 = (tid & 1) ? l2b : l2a;
      const uint64_t addr = ((tid & 1) ? kVaB : kVaA) + kOffset;
      const uint32_t vmid = (tid & 1) ? kVmidB : kVmidA;
      start.arrive_and_wait();
      for (uint32_t i = 0; i < kIterations; ++i) {
        if (l2.atomic_rmw(
                addr, sizeof(uint32_t),
                [](uint8_t *line, uint32_t offset) {
                  uint32_t value = 0;
                  std::memcpy(&value, line + offset, sizeof(value));
                  ++value;
                  std::memcpy(line + offset, &value, sizeof(value));
                },
                vmid) != rocjitsu::amdgpu::VmAccessOutcome::Complete)
          failed_operations.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(failed_operations.load(std::memory_order_relaxed), 0u);
  uint32_t actual = 0;
  std::memcpy(&actual, backing.data() + kOffset, sizeof(actual));
  EXPECT_EQ(actual, kThreads * kIterations);
}

TEST(L2CacheThreadingTest, CrossL2AtomicRmwDistinctSharedMappingsIncludesDirtyWriteback) {
  constexpr size_t kMappingSize = GpuMemory::PAGE_SIZE;
  const int raw_fd = static_cast<int>(syscall(SYS_memfd_create, "l2_atomic_alias", MFD_CLOEXEC));
  ASSERT_GE(raw_fd, 0);
  ScopedFd fd(raw_fd);
  ASSERT_EQ(ftruncate(fd.get(), static_cast<off_t>(kMappingSize)), 0);

  void *raw_mapping_a =
      mmap(nullptr, kMappingSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
  ASSERT_NE(raw_mapping_a, MAP_FAILED);
  ScopedMapping mapping_a(raw_mapping_a, kMappingSize);
  void *raw_mapping_b =
      mmap(nullptr, kMappingSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
  ASSERT_NE(raw_mapping_b, MAP_FAILED);
  ScopedMapping mapping_b(raw_mapping_b, kMappingSize);
  ASSERT_NE(mapping_a.data(), mapping_b.data());

  constexpr uint32_t kVmidA = 17;
  constexpr uint32_t kVmidB = 18;
  constexpr uint64_t kVaA = 0x310000;
  constexpr uint64_t kVaB = 0x420000;
  constexpr uint32_t kOffset = 64;
  rocjitsu::KfdProcess process_a(kVmidA);
  rocjitsu::KfdProcess process_b(kVmidB);
  process_a.map_pages(kVaA, mapping_a.data(), kMappingSize, Mtype::CC);
  process_b.map_pages(kVaB, mapping_b.data(), kMappingSize, Mtype::CC);

  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  rocjitsu::amdgpu::LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  ASSERT_TRUE(legacy_vm.register_address_space(kVmidA, &process_a.page_table_,
                                               &process_a.page_table_mutex_,
                                               process_a.page_table_generation()));
  ASSERT_TRUE(legacy_vm.register_address_space(kVmidB, &process_b.page_table_,
                                               &process_b.page_table_mutex_,
                                               process_b.page_table_generation()));
  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  L2Cache l2a("l2a", coherence);
  L2Cache l2b("l2b", coherence);
  l2a.set_backing_memory(&memory);
  l2b.set_backing_memory(&memory);
  l2a.set_gpu_vm(&gpu_vm);
  l2b.set_gpu_vm(&gpu_vm);

  std::array<uint8_t, L2Cache::LINE_SIZE> dirty_line{};
  constexpr uint32_t kDirtyValue = 40;
  std::memcpy(dirty_line.data() + kOffset, &kDirtyValue, sizeof(kDirtyValue));
  l2a.writeback_line(kVaA, dirty_line.data(), Mtype::RW, kVmidA);

  auto increment = [](uint8_t *line, uint32_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, line + offset, sizeof(value));
    ++value;
    std::memcpy(line + offset, &value, sizeof(value));
  };

  // Both virtual addresses resolve to the same MAP_SHARED bytes. The first
  // atomic boundary to enter must publish l2a's dirty line before either RMW,
  // and both RMWs must remain serialized across the distinct L2 objects.
  std::barrier start(3);
  std::atomic<uint64_t> failed_operations{0};
  std::thread first_atomic([&] {
    start.arrive_and_wait();
    if (l2a.atomic_rmw(kVaA + kOffset, sizeof(uint32_t), increment, kVmidA) !=
        rocjitsu::amdgpu::VmAccessOutcome::Complete)
      failed_operations.fetch_add(1, std::memory_order_relaxed);
  });
  std::thread second_atomic([&] {
    start.arrive_and_wait();
    if (l2b.atomic_rmw(kVaB + kOffset, sizeof(uint32_t), increment, kVmidB) !=
        rocjitsu::amdgpu::VmAccessOutcome::Complete)
      failed_operations.fetch_add(1, std::memory_order_relaxed);
  });
  start.arrive_and_wait();

  first_atomic.join();
  second_atomic.join();

  EXPECT_EQ(failed_operations.load(std::memory_order_relaxed), 0u);
  uint32_t actual = 0;
  std::memcpy(&actual, mapping_a.data() + kOffset, sizeof(actual));
  EXPECT_EQ(actual, 42u);
}

TEST(DeviceCacheCoherenceTest, ScalarWriteThroughSurvivesAliasedRemoteAtomic) {
  constexpr size_t kMappingSize = GpuMemory::PAGE_SIZE;
  const int raw_fd =
      static_cast<int>(syscall(SYS_memfd_create, "scalar_atomic_alias", MFD_CLOEXEC));
  ASSERT_GE(raw_fd, 0);
  ScopedFd fd(raw_fd);
  ASSERT_EQ(ftruncate(fd.get(), static_cast<off_t>(kMappingSize)), 0);

  void *raw_scalar_mapping =
      mmap(nullptr, kMappingSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
  ASSERT_NE(raw_scalar_mapping, MAP_FAILED);
  ScopedMapping scalar_mapping(raw_scalar_mapping, kMappingSize);
  void *raw_atomic_mapping =
      mmap(nullptr, kMappingSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
  ASSERT_NE(raw_atomic_mapping, MAP_FAILED);
  ScopedMapping atomic_mapping(raw_atomic_mapping, kMappingSize);
  ASSERT_NE(scalar_mapping.data(), atomic_mapping.data());

  constexpr uint32_t kScalarVmid = 21;
  constexpr uint32_t kAtomicVmid = 22;
  constexpr uint64_t kScalarVa = 0x510000;
  constexpr uint64_t kAtomicVa = 0x620000;
  constexpr uint64_t kLineOffset = L2Cache::LINE_SIZE;
  constexpr uint64_t kAtomicAddr = kAtomicVa + kLineOffset;
  constexpr uint64_t kScalarAddr = kScalarVa + kLineOffset + sizeof(uint32_t);
  constexpr uint32_t kScalarValue = 0xA5A5A5A5;

  rocjitsu::KfdProcess scalar_process(kScalarVmid);
  rocjitsu::KfdProcess atomic_process(kAtomicVmid);
  scalar_process.map_pages(kScalarVa, scalar_mapping.data(), kMappingSize, Mtype::RW);
  atomic_process.map_pages(kAtomicVa, atomic_mapping.data(), kMappingSize, Mtype::RW);

  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  rocjitsu::amdgpu::LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  ASSERT_TRUE(legacy_vm.register_address_space(kScalarVmid, &scalar_process.page_table_,
                                               &scalar_process.page_table_mutex_,
                                               scalar_process.page_table_generation()));
  ASSERT_TRUE(legacy_vm.register_address_space(kAtomicVmid, &atomic_process.page_table_,
                                               &atomic_process.page_table_mutex_,
                                               atomic_process.page_table_generation()));
  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  L2Cache scalar_l2("scalar_l2", coherence);
  L2Cache atomic_l2("atomic_l2", coherence);
  rocjitsu::amdgpu::L1ScalarCache scalar_l1(&scalar_l2);
  scalar_l2.set_backing_memory(&memory);
  atomic_l2.set_backing_memory(&memory);
  scalar_l2.set_gpu_vm(&gpu_vm);
  atomic_l2.set_gpu_vm(&gpu_vm);
  scalar_l1.set_gpu_vm(&gpu_vm);

  scalar_l1.store(kScalarAddr, /*num_dwords=*/1, &kScalarValue, kScalarVmid);
  ASSERT_EQ(atomic_l2.atomic_rmw(kAtomicAddr, sizeof(uint32_t), increment_u32, kAtomicVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  scalar_l1.writeback_all(kScalarVmid);

  uint32_t atomic_value = 0;
  uint32_t scalar_value = 0;
  std::memcpy(&atomic_value, scalar_mapping.data() + kLineOffset, sizeof(atomic_value));
  std::memcpy(&scalar_value, scalar_mapping.data() + kLineOffset + sizeof(uint32_t),
              sizeof(scalar_value));
  EXPECT_EQ(atomic_value, 1u);
  EXPECT_EQ(scalar_value, kScalarValue);
}

TEST(DeviceCacheCoherenceTest, DisjointDirtyL2AliasesSurviveRemoteAtomic) {
  constexpr size_t kMappingSize = GpuMemory::PAGE_SIZE;
  const int raw_fd =
      static_cast<int>(syscall(SYS_memfd_create, "l2_dirty_atomic_alias", MFD_CLOEXEC));
  ASSERT_GE(raw_fd, 0);
  ScopedFd fd(raw_fd);
  ASSERT_EQ(ftruncate(fd.get(), static_cast<off_t>(kMappingSize)), 0);

  void *raw_mapping_a =
      mmap(nullptr, kMappingSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
  ASSERT_NE(raw_mapping_a, MAP_FAILED);
  ScopedMapping mapping_a(raw_mapping_a, kMappingSize);
  void *raw_mapping_b =
      mmap(nullptr, kMappingSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
  ASSERT_NE(raw_mapping_b, MAP_FAILED);
  ScopedMapping mapping_b(raw_mapping_b, kMappingSize);
  void *raw_mapping_atomic =
      mmap(nullptr, kMappingSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd.get(), 0);
  ASSERT_NE(raw_mapping_atomic, MAP_FAILED);
  ScopedMapping mapping_atomic(raw_mapping_atomic, kMappingSize);
  ASSERT_NE(mapping_a.data(), mapping_b.data());
  ASSERT_NE(mapping_a.data(), mapping_atomic.data());
  ASSERT_NE(mapping_b.data(), mapping_atomic.data());

  constexpr uint32_t kVmidA = 31;
  constexpr uint32_t kVmidB = 32;
  constexpr uint32_t kAtomicVmid = 33;
  constexpr uint64_t kVaA = 0x710000;
  constexpr uint64_t kVaB = 0x820000;
  constexpr uint64_t kAtomicVa = 0x930000;
  constexpr uint64_t kLineOffset = L2Cache::LINE_SIZE;
  constexpr uint32_t kValueA = 0x11112222;
  constexpr uint32_t kValueB = 0x33334444;
  constexpr uint32_t kOffsetA = sizeof(uint32_t);
  constexpr uint32_t kOffsetB = 2 * sizeof(uint32_t);

  rocjitsu::KfdProcess process_a(kVmidA);
  rocjitsu::KfdProcess process_b(kVmidB);
  rocjitsu::KfdProcess atomic_process(kAtomicVmid);
  process_a.map_pages(kVaA, mapping_a.data(), kMappingSize, Mtype::RW);
  process_b.map_pages(kVaB, mapping_b.data(), kMappingSize, Mtype::RW);
  atomic_process.map_pages(kAtomicVa, mapping_atomic.data(), kMappingSize, Mtype::RW);

  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  rocjitsu::amdgpu::LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  ASSERT_TRUE(legacy_vm.register_address_space(kVmidA, &process_a.page_table_,
                                               &process_a.page_table_mutex_,
                                               process_a.page_table_generation()));
  ASSERT_TRUE(legacy_vm.register_address_space(kVmidB, &process_b.page_table_,
                                               &process_b.page_table_mutex_,
                                               process_b.page_table_generation()));
  ASSERT_TRUE(legacy_vm.register_address_space(kAtomicVmid, &atomic_process.page_table_,
                                               &atomic_process.page_table_mutex_,
                                               atomic_process.page_table_generation()));
  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  L2Cache l2a("l2a", coherence);
  L2Cache l2b("l2b", coherence);
  L2Cache atomic_l2("atomic_l2", coherence);
  l2a.set_backing_memory(&memory);
  l2b.set_backing_memory(&memory);
  atomic_l2.set_backing_memory(&memory);
  l2a.set_gpu_vm(&gpu_vm);
  l2b.set_gpu_vm(&gpu_vm);
  atomic_l2.set_gpu_vm(&gpu_vm);

  std::array<uint8_t, L2Cache::LINE_SIZE> line_a{};
  std::array<uint8_t, L2Cache::LINE_SIZE> line_b{};
  std::memcpy(line_a.data() + kOffsetA, &kValueA, sizeof(kValueA));
  std::memcpy(line_b.data() + kOffsetB, &kValueB, sizeof(kValueB));
  l2a.writeback_line(kVaA + kLineOffset, line_a.data(), kOffsetA, sizeof(kValueA), Mtype::RW,
                     kVmidA);
  l2b.writeback_line(kVaB + kLineOffset, line_b.data(), kOffsetB, sizeof(kValueB), Mtype::RW,
                     kVmidB);

  ASSERT_EQ(
      atomic_l2.atomic_rmw(kAtomicVa + kLineOffset, sizeof(uint32_t), increment_u32, kAtomicVmid),
      rocjitsu::amdgpu::VmAccessOutcome::Complete);

  uint32_t atomic_value = 0;
  uint32_t value_a = 0;
  uint32_t value_b = 0;
  std::memcpy(&atomic_value, mapping_a.data() + kLineOffset, sizeof(atomic_value));
  std::memcpy(&value_a, mapping_a.data() + kLineOffset + kOffsetA, sizeof(value_a));
  std::memcpy(&value_b, mapping_a.data() + kLineOffset + kOffsetB, sizeof(value_b));
  EXPECT_EQ(atomic_value, 1u);
  EXPECT_EQ(value_a, kValueA);
  EXPECT_EQ(value_b, kValueB);
}

TEST(DeviceCacheCoherenceTest, ConcurrentScalarLoadsTrackRepeatedAtomicEpochs) {
  GpuMemory memory("memory");
  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  L2Cache scalar_l2("scalar_l2", coherence);
  L2Cache atomic_l2("atomic_l2", coherence);
  rocjitsu::amdgpu::L1ScalarCache scalar_l1(&scalar_l2);
  scalar_l2.set_backing_memory(&memory);
  atomic_l2.set_backing_memory(&memory);

  constexpr uint64_t kAddr = 0xA600;
  constexpr uint32_t kAtomicIterations = 256;
  constexpr uint32_t kLoadIterations = 2048;
  memory.write32(kAddr, 0);
  uint32_t initially_cached = 1;
  scalar_l1.load(kAddr, /*num_dwords=*/1, &initially_cached);
  ASSERT_EQ(initially_cached, 0u);

  std::mutex start_mutex;
  std::condition_variable start_cv;
  bool atomic_ready = false;
  bool scalar_ready = false;
  std::atomic<bool> synchronization_failed{false};
  std::atomic<bool> nonmonotonic_load{false};
  std::atomic<bool> atomic_failed{false};

  auto wait_for_peer = [&](bool &self_ready, const bool &peer_ready) {
    std::unique_lock lock(start_mutex);
    self_ready = true;
    start_cv.notify_all();
    if (!start_cv.wait_for(lock, std::chrono::seconds(1), [&] { return peer_ready; }))
      synchronization_failed.store(true, std::memory_order_relaxed);
  };

  std::thread atomic([&] {
    wait_for_peer(atomic_ready, scalar_ready);
    for (uint32_t iteration = 0; iteration < kAtomicIterations; ++iteration) {
      if (atomic_l2.atomic_rmw(kAddr, sizeof(uint32_t), increment_u32) !=
          rocjitsu::amdgpu::VmAccessOutcome::Complete)
        atomic_failed.store(true, std::memory_order_relaxed);
      std::this_thread::yield();
    }
  });
  std::thread scalar([&] {
    wait_for_peer(scalar_ready, atomic_ready);
    uint32_t previous = 0;
    for (uint32_t iteration = 0; iteration < kLoadIterations; ++iteration) {
      uint32_t observed = 0;
      scalar_l1.load(kAddr, /*num_dwords=*/1, &observed);
      if (observed < previous || observed > kAtomicIterations)
        nonmonotonic_load.store(true, std::memory_order_relaxed);
      previous = observed;
      std::this_thread::yield();
    }
  });
  atomic.join();
  scalar.join();

  EXPECT_FALSE(synchronization_failed.load(std::memory_order_relaxed));
  EXPECT_FALSE(nonmonotonic_load.load(std::memory_order_relaxed));
  EXPECT_FALSE(atomic_failed.load(std::memory_order_relaxed));
  uint32_t final_value = 0;
  scalar_l1.load(kAddr, /*num_dwords=*/1, &final_value);
  EXPECT_EQ(final_value, kAtomicIterations);
}

TEST(L2CacheTest, FunctionalLinkedPortAtomicRmwUpdatesBacking) {
  constexpr uint64_t kBase = 0xB00000;
  constexpr uint64_t kAddr = kBase + 20;
  L2Cache l2("l2");
  FunctionalMemoryPort backing("backing", kBase, 2 * L2Cache::LINE_SIZE);
  simdojo::Link link(/*id=*/0, l2.req_port(), backing.port(), /*latency=*/0);
  link.set_exec_mode(simdojo::ExecMode::FUNCTIONAL);
  l2.req_port()->set_link(&link);
  backing.port()->set_link(&link);

  backing.write8(kAddr - 1, 0xA5);
  backing.write32(kAddr, 41);
  backing.write8(kAddr + sizeof(uint32_t), 0x5A);
  uint32_t callback_old_value = 0;
  ASSERT_EQ(l2.atomic_rmw(kAddr, sizeof(uint32_t),
                          [&](uint8_t *target, uint32_t offset) {
                            EXPECT_EQ(offset, 0u);
                            std::memcpy(&callback_old_value, target, sizeof(callback_old_value));
                            const uint32_t replacement = callback_old_value + 1;
                            std::memcpy(target, &replacement, sizeof(replacement));
                          }),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);

  EXPECT_EQ(callback_old_value, 41u);
  EXPECT_EQ(backing.read32(kAddr), 42u);
  EXPECT_EQ(backing.read8(kAddr - 1), 0xA5);
  EXPECT_EQ(backing.read8(kAddr + sizeof(uint32_t)), 0x5A);
}

TEST(L2CacheTest, LinkedPortAtomicRmwRefreshesStaleMemorySideAlias) {
  constexpr uint32_t kVmidA = 41;
  constexpr uint32_t kVmidB = 42;
  constexpr uint64_t kVaA = 0xC10000;
  constexpr uint64_t kVaB = 0xD20000;
  constexpr uint32_t kOffset = 64;
  constexpr uint32_t kPublishedValue = 40;

  rocjitsu::KfdProcess process_a(kVmidA);
  rocjitsu::KfdProcess process_b(kVmidB);
  std::array<uint8_t, GpuMemory::PAGE_SIZE> backing{};
  process_a.map_pages(kVaA, backing.data(), backing.size(), Mtype::RW);
  process_b.map_pages(kVaB, backing.data(), backing.size(), Mtype::RW);

  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  rocjitsu::amdgpu::LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  ASSERT_TRUE(legacy_vm.register_address_space(kVmidA, &process_a.page_table_,
                                               &process_a.page_table_mutex_,
                                               process_a.page_table_generation()));
  ASSERT_TRUE(legacy_vm.register_address_space(kVmidB, &process_b.page_table_,
                                               &process_b.page_table_mutex_,
                                               process_b.page_table_generation()));

  HbmController hbm("hbm", &memory);
  hbm.set_gpu_vm(&gpu_vm);
  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  MemorySideCache msc("msc", coherence, &memory);
  L2Cache l2a("l2a", coherence);
  L2Cache l2b("l2b", coherence);

  simdojo::Port *l2a_msc_port = msc.create_cpl_port("l2a");
  simdojo::Port *l2b_msc_port = msc.create_cpl_port("l2b");
  simdojo::Link msc_hbm_link(/*id=*/0, msc.req_port(), hbm.cpl_port(), /*latency=*/0);
  simdojo::Link l2a_msc_link(/*id=*/1, l2a.req_port(), l2a_msc_port, /*latency=*/0);
  simdojo::Link l2b_msc_link(/*id=*/2, l2b.req_port(), l2b_msc_port, /*latency=*/0);
  for (simdojo::Link *link : {&msc_hbm_link, &l2a_msc_link, &l2b_msc_link})
    link->set_exec_mode(simdojo::ExecMode::FUNCTIONAL);
  msc.req_port()->set_link(&msc_hbm_link);
  hbm.cpl_port()->set_link(&msc_hbm_link);
  l2a.req_port()->set_link(&l2a_msc_link);
  l2a_msc_port->set_link(&l2a_msc_link);
  l2b.req_port()->set_link(&l2b_msc_link);
  l2b_msc_port->set_link(&l2b_msc_link);

  uint32_t stale_value = 1;
  l2b.read(kVaB + kOffset, reinterpret_cast<uint8_t *>(&stale_value), sizeof(stale_value),
           Mtype::RW, kVmidB);
  ASSERT_EQ(stale_value, 0u);

  std::array<uint8_t, L2Cache::LINE_SIZE> dirty_line{};
  std::memcpy(dirty_line.data() + kOffset, &kPublishedValue, sizeof(kPublishedValue));
  l2a.writeback_line(kVaA, dirty_line.data(), kOffset, sizeof(kPublishedValue), Mtype::RW, kVmidA);

  uint32_t callback_old_value = 0;
  ASSERT_EQ(l2b.atomic_rmw(
                kVaB + kOffset, sizeof(uint32_t),
                [&](uint8_t *target, uint32_t offset) {
                  std::memcpy(&callback_old_value, target + offset, sizeof(callback_old_value));
                  const uint32_t replacement = callback_old_value + 1;
                  std::memcpy(target + offset, &replacement, sizeof(replacement));
                },
                kVmidB),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);

  EXPECT_EQ(callback_old_value, kPublishedValue);
  uint32_t host_value = 0;
  std::memcpy(&host_value, backing.data() + kOffset, sizeof(host_value));
  EXPECT_EQ(host_value, kPublishedValue + 1);

  uint32_t alias_a_value = 0;
  uint32_t alias_b_value = 0;
  l2a.read(kVaA + kOffset, reinterpret_cast<uint8_t *>(&alias_a_value), sizeof(alias_a_value),
           Mtype::RW, kVmidA);
  l2b.read(kVaB + kOffset, reinterpret_cast<uint8_t *>(&alias_b_value), sizeof(alias_b_value),
           Mtype::RW, kVmidB);
  EXPECT_EQ(alias_a_value, kPublishedValue + 1);
  EXPECT_EQ(alias_b_value, kPublishedValue + 1);
}

TEST(L2CacheTest, AliasedVasRequireCoherenceBoundary) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kVmidA = 7;
  constexpr uint32_t kVmidB = 8;
  constexpr uint64_t kVaA = 0x100000;
  constexpr uint64_t kVaB = 0x201000;

  rocjitsu::KfdProcess process_a(kVmidA);
  rocjitsu::KfdProcess process_b(kVmidB);
  std::array<uint8_t, GpuMemory::PAGE_SIZE> backing{};
  process_a.map_pages(kVaA, backing.data(), backing.size());
  process_b.map_pages(kVaB, backing.data(), backing.size());
  rocjitsu::amdgpu::GpuVm gpu_vm;
  rocjitsu::amdgpu::LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  ASSERT_TRUE(legacy_vm.register_address_space(kVmidA, &process_a.page_table_,
                                               &process_a.page_table_mutex_,
                                               process_a.page_table_generation()));
  ASSERT_TRUE(legacy_vm.register_address_space(kVmidB, &process_b.page_table_,
                                               &process_b.page_table_mutex_,
                                               process_b.page_table_generation()));
  l2.set_gpu_vm(&gpu_vm);

  std::array<uint8_t, L2Cache::LINE_SIZE> initial{};
  std::array<uint8_t, L2Cache::LINE_SIZE> replacement{};
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  initial.fill(0x11);
  replacement.fill(0x22);
  dirty.fill(0x33);

  std::copy(initial.begin(), initial.end(), backing.begin());
  l2.read(kVaA, actual.data(), actual.size(), Mtype::RW, kVmidA);
  ASSERT_EQ(actual, initial);
  l2.read(kVaB, actual.data(), actual.size(), Mtype::RW, kVmidB);
  ASSERT_EQ(actual, initial);

  l2.write(kVaB, replacement.data(), replacement.size(), Mtype::RW, kVmidB);
  l2.read(kVaA, actual.data(), actual.size(), Mtype::RW, kVmidA);
  EXPECT_EQ(actual, initial);
  l2.read(kVaA, actual.data(), actual.size(), Mtype::CC, kVmidA);
  EXPECT_EQ(actual, replacement);

  l2.writeback_line(kVaA, dirty.data(), Mtype::RW, kVmidA);
  std::copy_n(backing.begin(), actual.size(), actual.begin());
  EXPECT_EQ(actual, replacement);
  l2.flush_line(kVaA, kVmidA);
  std::copy_n(backing.begin(), actual.size(), actual.begin());
  EXPECT_EQ(actual, dirty);

  l2.read(kVaB, actual.data(), actual.size(), Mtype::RW, kVmidB);
  EXPECT_EQ(actual, replacement);
  l2.read(kVaB, actual.data(), actual.size(), Mtype::CC, kVmidB);
  EXPECT_EQ(actual, dirty);
}

TEST(L2CacheThreadingTest, ConcurrentFlushAllPreservesDirtyWritebacks) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kWriterThreads = 4;
  constexpr uint32_t kLinesPerThread = 8;
  constexpr uint32_t kIterations = 64;
  constexpr uint64_t kBase = 0x300000;

  std::atomic<uint32_t> active_writers{0};
  std::barrier start(kWriterThreads + 1);
  std::vector<std::thread> workers;
  workers.reserve(kWriterThreads);

  for (uint32_t tid = 0; tid < kWriterThreads; ++tid) {
    workers.emplace_back([&, tid] {
      std::array<uint8_t, L2Cache::LINE_SIZE> line{};
      start.arrive_and_wait();
      active_writers.fetch_add(1, std::memory_order_release);
      for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
        for (uint32_t i = 0; i < kLinesPerThread; ++i) {
          const uint64_t addr =
              kBase + (static_cast<uint64_t>(i) * kWriterThreads + tid) * L2Cache::LINE_SIZE;
          for (uint32_t b = 0; b < line.size(); ++b)
            line[b] = static_cast<uint8_t>((tid << 5) ^ iteration ^ i ^ b);
          l2.writeback_line(addr, line.data());
        }
        std::this_thread::yield();
      }
    });
  }

  std::thread flusher([&] {
    start.arrive_and_wait();
    while (active_writers.load(std::memory_order_acquire) < kWriterThreads)
      std::this_thread::yield();
    for (uint32_t i = 0; i < 4; ++i) {
      l2.flush_all();
      std::this_thread::yield();
    }
  });

  for (auto &worker : workers)
    worker.join();
  flusher.join();

  l2.flush_all();

  std::array<uint8_t, L2Cache::LINE_SIZE> expected{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  for (uint32_t tid = 0; tid < kWriterThreads; ++tid) {
    for (uint32_t i = 0; i < kLinesPerThread; ++i) {
      const uint64_t addr =
          kBase + (static_cast<uint64_t>(i) * kWriterThreads + tid) * L2Cache::LINE_SIZE;
      for (uint32_t b = 0; b < expected.size(); ++b)
        expected[b] = static_cast<uint8_t>((tid << 5) ^ (kIterations - 1) ^ i ^ b);
      memory.read_block(addr, std::span<uint8_t>(actual));
      EXPECT_EQ(actual, expected) << "addr=0x" << std::hex << addr;
    }
  }
}

TEST(L2CacheTest, InvalidateRangeClampsAtAddressSpaceEnd) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint64_t kLineAddr = std::numeric_limits<uint64_t>::max() - (L2Cache::LINE_SIZE - 1);
  constexpr uint64_t kRangeAddr =
      std::numeric_limits<uint64_t>::max() - (L2Cache::LINE_SIZE / 2 - 1);
  std::array<uint8_t, L2Cache::LINE_SIZE> initial{};
  std::array<uint8_t, L2Cache::LINE_SIZE> replacement{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  initial.fill(0x11);
  replacement.fill(0x22);

  memory.write_block(kLineAddr, std::span<const uint8_t>(initial));
  l2.read(kLineAddr, actual.data(), actual.size());
  ASSERT_EQ(actual, initial);

  memory.write_block(kLineAddr, std::span<const uint8_t>(replacement));
  l2.invalidate_range(kRangeAddr, L2Cache::LINE_SIZE, 0);
  l2.read(kLineAddr, actual.data(), actual.size());

  EXPECT_EQ(actual, replacement);
}

TEST(L2CacheTest, InvalidateRangeRefreshesOnlyCoveredSetsAndZeroSizeIsNoOp) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint64_t kBase = 0x400000;
  constexpr uint32_t kLines = 5;
  std::array<std::array<uint8_t, L2Cache::LINE_SIZE>, kLines> initial{};
  std::array<std::array<uint8_t, L2Cache::LINE_SIZE>, kLines> replacement{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};

  for (uint32_t line = 0; line < kLines; ++line) {
    initial[line].fill(static_cast<uint8_t>(0x10 + line));
    replacement[line].fill(static_cast<uint8_t>(0x80 + line));
    const uint64_t addr = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
    memory.write_block(addr, std::span<const uint8_t>(initial[line]));
    l2.read(addr, actual.data(), actual.size());
    ASSERT_EQ(actual, initial[line]);
    memory.write_block(addr, std::span<const uint8_t>(replacement[line]));
  }

  l2.invalidate_range(kBase + L2Cache::LINE_SIZE + 32, 2 * L2Cache::LINE_SIZE, 0);

  for (uint32_t line = 0; line < kLines; ++line) {
    const uint64_t addr = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
    l2.read(addr, actual.data(), actual.size());
    const auto &expected = (line >= 1 && line <= 3) ? replacement[line] : initial[line];
    EXPECT_EQ(actual, expected) << "line=" << line;
  }

  l2.invalidate_range(kBase, 0, 0);
  l2.read(kBase, actual.data(), actual.size());
  EXPECT_EQ(actual, initial[0]);
}

TEST(L2CacheTest, InvalidateRangeOver128LinesUsesExclusiveMaintenance) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint64_t kBase = 0x700000;
  constexpr uint32_t kLines = 129;
  std::array<uint8_t, L2Cache::LINE_SIZE> initial{};
  std::array<uint8_t, L2Cache::LINE_SIZE> replacement{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};

  for (uint32_t line = 0; line < kLines; ++line) {
    const uint64_t addr = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
    initial.fill(static_cast<uint8_t>(line));
    replacement.fill(static_cast<uint8_t>(line + 0x40));
    memory.write_block(addr, std::span<const uint8_t>(initial));
    l2.read(addr, actual.data(), actual.size());
    ASSERT_EQ(actual, initial) << "line=" << line;
    memory.write_block(addr, std::span<const uint8_t>(replacement));
  }

  l2.invalidate_range(kBase, kLines * L2Cache::LINE_SIZE, 0);

  for (uint32_t line = 0; line < kLines; ++line) {
    const uint64_t addr = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
    replacement.fill(static_cast<uint8_t>(line + 0x40));
    l2.read(addr, actual.data(), actual.size());
    EXPECT_EQ(actual, replacement) << "line=" << line;
  }
}

TEST(L2CacheTest, InvalidateRangeOnlyAffectsRequestedVmid) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kVmidA = 7;
  constexpr uint32_t kVmidB = 8;
  constexpr uint64_t kAddr = 0x500000;
  rocjitsu::KfdProcess process_a(kVmidA);
  rocjitsu::KfdProcess process_b(kVmidB);
  std::array<uint8_t, GpuMemory::PAGE_SIZE> backing_a{};
  std::array<uint8_t, GpuMemory::PAGE_SIZE> backing_b{};
  process_a.map_pages(kAddr, backing_a.data(), backing_a.size());
  process_b.map_pages(kAddr, backing_b.data(), backing_b.size());
  rocjitsu::amdgpu::GpuVm gpu_vm;
  rocjitsu::amdgpu::LegacyGpuVmAdapter legacy_vm(gpu_vm, &memory);
  ASSERT_TRUE(legacy_vm.register_address_space(kVmidA, &process_a.page_table_,
                                               &process_a.page_table_mutex_,
                                               process_a.page_table_generation()));
  ASSERT_TRUE(legacy_vm.register_address_space(kVmidB, &process_b.page_table_,
                                               &process_b.page_table_mutex_,
                                               process_b.page_table_generation()));
  l2.set_gpu_vm(&gpu_vm);

  std::array<uint8_t, L2Cache::LINE_SIZE> initial_a{};
  std::array<uint8_t, L2Cache::LINE_SIZE> initial_b{};
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty_a{};
  std::array<uint8_t, L2Cache::LINE_SIZE> replacement_b{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  initial_a.fill(0x11);
  initial_b.fill(0x22);
  dirty_a.fill(0x33);
  replacement_b.fill(0x44);

  std::copy(initial_a.begin(), initial_a.end(), backing_a.begin());
  std::copy(initial_b.begin(), initial_b.end(), backing_b.begin());
  l2.read(kAddr, actual.data(), actual.size(), Mtype::RW, kVmidA);
  ASSERT_EQ(actual, initial_a);
  l2.read(kAddr, actual.data(), actual.size(), Mtype::RW, kVmidB);
  ASSERT_EQ(actual, initial_b);
  l2.writeback_line(kAddr, dirty_a.data(), Mtype::RW, kVmidA);

  std::copy(replacement_b.begin(), replacement_b.end(), backing_b.begin());
  l2.invalidate_range(kAddr, L2Cache::LINE_SIZE, kVmidB);
  l2.read(kAddr, actual.data(), actual.size(), Mtype::RW, kVmidB);
  EXPECT_EQ(actual, replacement_b);

  l2.flush_line(kAddr, kVmidA);
  std::copy_n(backing_a.begin(), actual.size(), actual.begin());
  EXPECT_EQ(actual, dirty_a);
}

TEST(L2CacheTest, InvalidateRangePreservesDirtyBytesOutsidePartialWrite) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint64_t kAddr = 0x600000;
  constexpr uint32_t kOffset = 32;
  std::array<uint8_t, L2Cache::LINE_SIZE> initial{};
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  std::array<uint8_t, 16> host_write{};
  std::array<uint8_t, L2Cache::LINE_SIZE> expected{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  initial.fill(0x11);
  dirty.fill(0x22);
  host_write.fill(0x33);
  expected = dirty;
  std::memcpy(expected.data() + kOffset, host_write.data(), host_write.size());

  memory.write_block(kAddr, std::span<const uint8_t>(initial));
  l2.read(kAddr, actual.data(), actual.size());
  ASSERT_EQ(actual, initial);
  l2.writeback_line(kAddr, dirty.data());

  memory.write_block(kAddr + kOffset, std::span<const uint8_t>(host_write));
  l2.invalidate_range(kAddr + kOffset, host_write.size(), 0);

  memory.read_block(kAddr, std::span<uint8_t>(actual));
  EXPECT_EQ(actual, expected);
  l2.read(kAddr, actual.data(), actual.size());
  EXPECT_EQ(actual, expected);
}

TEST(L2CacheBenchmark, CrossL2SameAddress) {
  run_cross_l2_atomic_benchmark("cross_l2_same_address", true);
}

TEST(L2CacheBenchmark, CrossL2IndependentAddresses) {
  run_cross_l2_atomic_benchmark("cross_l2_independent_addresses", false);
}

TEST(L2CacheBenchmark, AtomicOneHierarchy) {
  run_atomic_hierarchy_benchmark("atomic_hierarchies_1", 1);
}

TEST(L2CacheBenchmark, AtomicTwoHierarchies) {
  run_atomic_hierarchy_benchmark("atomic_hierarchies_2", 2);
}

TEST(L2CacheBenchmark, AtomicFourHierarchies) {
  run_atomic_hierarchy_benchmark("atomic_hierarchies_4", 4);
}

TEST(L2CacheBenchmark, AtomicEightHierarchies) {
  run_atomic_hierarchy_benchmark("atomic_hierarchies_8", 8);
}

TEST(L2CacheBenchmark, ScalarL1Hit) { run_scalar_l1_hit_benchmark(); }

TEST(L2CacheBenchmark, VectorL1Hit) { run_vector_l1_hit_benchmark(); }

TEST(L2CacheBenchmark, InvalidateThreeLines) {
  run_invalidate_range_benchmark("invalidate_3_lines", 3, 200'000);
}

TEST(L2CacheBenchmark, Invalidate129Lines) {
  run_invalidate_range_benchmark("invalidate_129_lines", 129, 5'000);
}

TEST(GpuMemoryTest, BlockAccessHandlesPageBoundaries) {
  GpuMemory memory("memory");

  constexpr uint64_t kAddr = 0x3ff0;
  std::array<uint8_t, 64> input{};
  std::array<uint8_t, 64> output{};
  for (uint32_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<uint8_t>(i * 3);

  memory.write_block(kAddr, std::span<const uint8_t>(input));
  memory.read_block(kAddr, std::span<uint8_t>(output));

  EXPECT_EQ(output, input);
}

} // namespace
