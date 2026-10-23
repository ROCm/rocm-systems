// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "legacy_gpu_memory_fixture.h"
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
  explicit IdentityAddressSpace(uint64_t size, Mtype mtype = Mtype::RW)
      : size_(size), mtype_(mtype) {}

  rocjitsu::amdgpu::VmTranslationResult translate(uint64_t address, std::size_t size,
                                                  rocjitsu::amdgpu::VmAccessKind) const override {
    if (size == 0 || address > size_ || size > size_ - address)
      return {.outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted, .translation = {}};
    return {
        .outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete,
        .translation = {.domain = rocjitsu::amdgpu::VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = size_ - address,
                        .mtype = mtype_,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }

private:
  uint64_t size_;
  Mtype mtype_;
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
    std::ranges::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(address), bytes.size(),
                        bytes.begin());
    return rocjitsu::amdgpu::VmAccessOutcome::Complete;
  }

  rocjitsu::amdgpu::VmAccessOutcome write(rocjitsu::amdgpu::VmMemoryDomain domain, uint64_t address,
                                          std::span<const std::byte> bytes) override {
    ++write_calls;
    if (write_outcome != rocjitsu::amdgpu::VmAccessOutcome::Complete)
      return write_outcome;
    if (domain != rocjitsu::amdgpu::VmMemoryDomain::System || !contains(address, bytes.size()))
      return rocjitsu::amdgpu::VmAccessOutcome::Faulted;
    std::ranges::copy(bytes, bytes_.begin() + static_cast<std::ptrdiff_t>(address));
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
    memory_side_cache.set_gpu_vm(&gpu_vm);
    l2.set_gpu_vm(&gpu_vm);
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

TEST(L2CacheTest, LinkedHierarchyCachesOnlyAccessibleBytesOfIncompleteLines) {
  constexpr uint64_t kValidAddress = 20;
  constexpr uint64_t kCrossingAddress = 22;
  constexpr uint32_t kInitial = 0x11223344;
  constexpr uint32_t kReplacement = 0xaabbccdd;
  constexpr uint32_t kInvalid = 0x55667788;
  LinkedCacheHierarchy hierarchy(24);
  ASSERT_TRUE(hierarchy.address_space);
  hierarchy.physical->store(kValidAddress, kInitial);

  uint32_t observed = 0;
  ASSERT_EQ(hierarchy.l2.read(kValidAddress, reinterpret_cast<uint8_t *>(&observed),
                              sizeof(observed), Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kInitial);
  EXPECT_EQ(hierarchy.physical->read_calls, 1u);

  hierarchy.physical->store(kValidAddress, kReplacement);
  ASSERT_EQ(hierarchy.l2.read(kValidAddress, reinterpret_cast<uint8_t *>(&observed),
                              sizeof(observed), Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kInitial);
  EXPECT_EQ(hierarchy.physical->read_calls, 1u)
      << "repeated valid accesses should hit the partial line at every cache level";

  {
    [[maybe_unused]] auto maintenance = hierarchy.coherence->acquire_cache_maintenance(
        rocjitsu::amdgpu::DeviceCacheOperation::Invalidate);
  }
  ASSERT_EQ(hierarchy.l2.read(kValidAddress, reinterpret_cast<uint8_t *>(&observed),
                              sizeof(observed), Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kReplacement);
  EXPECT_EQ(hierarchy.physical->read_calls, 2u);

  observed = kInvalid;
  EXPECT_EQ(hierarchy.l2.read(kCrossingAddress, reinterpret_cast<uint8_t *>(&observed),
                              sizeof(observed), Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(observed, kInvalid);

  EXPECT_EQ(hierarchy.l2.write(kCrossingAddress, reinterpret_cast<const uint8_t *>(&kInvalid),
                               sizeof(kInvalid), Mtype::RW, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(hierarchy.physical->load<uint32_t>(kValidAddress), kReplacement);
}

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

  void *initial_raw = mmap(nullptr, GpuMemory::PAGE_SIZE, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(initial_raw, MAP_FAILED);
  ScopedMapping initial_mapping(initial_raw, GpuMemory::PAGE_SIZE);
  void *first_raw = mmap(nullptr, GpuMemory::PAGE_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(first_raw, MAP_FAILED);
  ScopedMapping first_mapping(first_raw, GpuMemory::PAGE_SIZE);
  void *second_raw = mmap(nullptr, GpuMemory::PAGE_SIZE, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(second_raw, MAP_FAILED);
  ScopedMapping second_mapping(second_raw, GpuMemory::PAGE_SIZE);

  rocjitsu::KfdProcess process(kVmid);
  process.map_pages(kGpuAddress, initial_mapping.data(), GpuMemory::PAGE_SIZE);
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
  ASSERT_EQ(l2.writeback_line(kGpuAddress, dirty_line.data(), kFirstOffset, sizeof(kFirstValue),
                              Mtype::RW, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(l2.writeback_line(kGpuAddress, dirty_line.data(), kSecondOffset, sizeof(kSecondValue),
                              Mtype::RW, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);

  process.unmap_pages(kGpuAddress, GpuMemory::PAGE_SIZE);
  process.map_pages(kGpuAddress + kFirstOffset, first_mapping.data(), sizeof(uint32_t));
  process.map_pages(kGpuAddress + kSecondOffset, second_mapping.data(), sizeof(uint32_t));

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
  EXPECT_TRUE(std::ranges::equal(dirty_line, std::span(mapping.data(), dirty_line.size())));
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
        kVmid, {.page_table = &process_.page_table_,
                .page_table_mutex = &process_.page_table_mutex_,
                .page_table_generation = process_.page_table_generation(),
                .request_mutex = process_.page_table_request_mutex(),
                .mutation_epoch = process_.page_table_mutation_epoch()});
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
  rocjitsu::amdgpu::VmMtypeCache mtype_cache_;
};

class LegacySubPageCacheTest : public testing::Test {
protected:
  static constexpr uint32_t kVmid = 7;
  static constexpr uint64_t kBase = 0x200000;
  static constexpr size_t kBackingSize = 24;

  void SetUp() override {
    std::ranges::fill(backing_, uint8_t{0x5a});
    page_table_[kBase >> rocjitsu::KfdProcess::kPageShift] = {backing_.data(), Mtype::RW,
                                                              backing_.size(), 0};
    address_space_ = legacy_vm_.register_address_space(kVmid, &page_table_, &page_table_mutex_);
    ASSERT_TRUE(address_space_);
    l2_.set_backing_memory(&memory_);
    l2_.set_gpu_vm(&gpu_vm_);
  }

  std::array<uint8_t, kBackingSize> backing_{};
  rocjitsu::KfdProcess::PageTable page_table_;
  util::DistributedSharedMutex page_table_mutex_;
  GpuMemory memory_{"memory"};
  rocjitsu::amdgpu::GpuVm gpu_vm_;
  rocjitsu::amdgpu::LegacyGpuVmAdapter legacy_vm_{gpu_vm_, &memory_};
  rocjitsu::amdgpu::AddressSpaceHandle address_space_;
  L2Cache l2_{"l2"};
};

class ScalarRamBatchTest : public testing::Test, public rocjitsu::amdgpu::MemoryFaultReporter {
protected:
  using Owner = rocjitsu::amdgpu::LegacyHostExtentOwner;
  using Outcome = rocjitsu::amdgpu::VmAccessOutcome;
  static constexpr uint32_t kVmid = 29;
  static constexpr uint64_t kBase = 0x800000;
  static constexpr uint32_t kSentinel = 0xdeadbeef;

  void SetUp() override {
    for (size_t i = 0; i < backing_.size(); ++i)
      backing_[i] = static_cast<uint8_t>(i * 17 + 3);
    address_space_ = adapter_.register_address_space(
        kVmid, {.page_table = &process_.page_table_,
                .page_table_mutex = &process_.page_table_mutex_,
                .page_table_generation = process_.page_table_generation(),
                .request_mutex = process_.page_table_request_mutex(),
                .mutation_epoch = process_.page_table_mutation_epoch(),
                .page_table_cache_state = process_.page_table_cache_state(),
                .fault_reporter = this});
    ASSERT_TRUE(address_space_);
    l2_.set_backing_memory(&memory_);
    l2_.set_gpu_vm(&vm_);
    l1_.set_gpu_vm(&vm_);
  }

  void map(Owner owner = Owner::DriverSealedRam, Mtype mtype = Mtype::UC, size_t size = 8192) {
    process_.map_pages(kBase, backing_.data(), size, mtype, owner);
  }

  void report_memory_fault(uint32_t vmid, uint64_t address,
                           rocjitsu::amdgpu::MemoryFaultCause) override {
    EXPECT_EQ(vmid, kVmid);
    faults_.push_back(address);
  }

  void expect_words(const std::array<uint32_t, 16> &words, uint32_t count, size_t offset = 0) {
    EXPECT_EQ(std::memcmp(words.data(), backing_.data() + offset, count * sizeof(uint32_t)), 0);
    for (uint32_t i = count; i < words.size(); ++i)
      EXPECT_EQ(words[i], kSentinel);
  }

  std::array<uint8_t, 8192> backing_{};
  rocjitsu::KfdProcess process_{kVmid};
  GpuMemory memory_{"scalar_ram"};
  rocjitsu::amdgpu::GpuVm vm_;
  rocjitsu::amdgpu::LegacyGpuVmAdapter adapter_{vm_, &memory_};
  rocjitsu::amdgpu::AddressSpaceHandle address_space_;
  L2Cache l2_{"scalar_ram_l2"};
  L1ScalarCache l1_{&l2_};
  std::vector<uint64_t> faults_;
};

class VectorRamBatchTest : public ScalarRamBatchTest {
protected:
  using Store = rocjitsu::amdgpu::VmRamDwordStore;
  void SetUp() override {
    ScalarRamBatchTest::SetUp();
    vector_.set_gpu_vm(&vm_);
    for (size_t i = 0; i < values_.size(); ++i)
      values_[i] = 0x10203000u + static_cast<uint32_t>(i);
  }
  void reset_bytes() {
    vector_.invalidate_all();
    l2_.invalidate_all();
    backing_.fill(0x5a);
    faults_.clear();
  }
  std::array<Store, 3> stores() const {
    return {{{kBase + 4, reinterpret_cast<const uint8_t *>(&values_[0])},
             {kBase + 12, reinterpret_cast<const uint8_t *>(&values_[1])},
             {kBase + 4, reinterpret_cast<const uint8_t *>(&values_[2])}}};
  }
  L1VectorCache vector_{&l2_};
  std::array<uint32_t, 256> values_{};
};

TEST_F(VectorRamBatchTest, SwizzledMasksDuplicatesAndBoundariesMatchOrdinaryStores) {
  map(Owner::DriverSealedRam, Mtype::RW);
  for (const Mtype policy : {Mtype::UC, Mtype::RW, Mtype::CC, Mtype::NT}) {
    for (const uint32_t unit : {4u, 16u}) {
      for (const uint32_t offset : {0u, 120u, 4080u}) {
        for (const bool partial : {false, true}) {
          SCOPED_TRACE(static_cast<uint32_t>(policy));
          SCOPED_TRACE(unit);
          SCOPED_TRACE(offset);
          SCOPED_TRACE(partial);
          std::array<uint64_t, 64> addresses;
          for (size_t lane = 0; lane < addresses.size(); ++lane)
            addresses[lane] = kBase + offset + lane * unit;
          addresses[3] = addresses[2];
          const uint64_t lanes = partial ? 0x7fff00ffff00ffffull : UINT64_MAX;
          const std::array<uint64_t, 4> masks{lanes, lanes & 0x5555555555555555ull,
                                              lanes & 0x3333333333333333ull, lanes};
          std::span<const uint64_t> element_masks;
          if (partial)
            element_masks = masks;
          std::array<uint8_t, 8192> ordinary;
          std::array<uint8_t, sizeof(values_)> ordinary_read{};
          uint64_t ordinary_transactions = 0;
          for (const bool allow : {false, true}) {
            reset_bytes();
            const uint64_t before = l2_.backing_write_transactions();
            ASSERT_EQ(vector_.store(addresses.data(), lanes, 4, 4,
                                    reinterpret_cast<const uint8_t *>(values_.data()), policy,
                                    policy == Mtype::NT, 64, kVmid, unit * 32, offset % unit,
                                    element_masks, unit, allow),
                      Outcome::Complete);
            std::array<uint8_t, sizeof(values_)> read{};
            ASSERT_EQ(vector_.load(addresses.data(), lanes, 4, 4, read.data(), policy,
                                   policy == Mtype::NT, false, 64, kVmid, unit * 32, offset % unit,
                                   element_masks, unit),
                      Outcome::Complete);
            if (!allow) {
              ordinary = backing_;
              ordinary_read = read;
              ordinary_transactions = l2_.backing_write_transactions() - before;
            } else {
              EXPECT_EQ(backing_, ordinary);
              EXPECT_EQ(read, ordinary_read);
              EXPECT_EQ(l2_.backing_write_transactions() - before, ordinary_transactions);
            }
            EXPECT_TRUE(faults_.empty());
          }
        }
      }
    }
  }
}

TEST_F(VectorRamBatchTest, DirectBatchKeepsDuplicateOrderCountersAndAliasVisibility) {
  for (const Mtype policy : {Mtype::UC, Mtype::CC, Mtype::RW}) {
    map(Owner::DriverSealedRam, Mtype::RW);
    process_.map_pages(kBase + 8192, backing_.data(), backing_.size(), Mtype::RW,
                       Owner::DriverSealedRam);
    reset_bytes();
    uint32_t alias_before = 0;
    ASSERT_EQ(
        l2_.read(kBase + 8192 + 4, reinterpret_cast<uint8_t *>(&alias_before), 4, Mtype::RW, kVmid),
        Outcome::Complete);
    ASSERT_EQ(l2_.write(kBase, reinterpret_cast<const uint8_t *>(&values_[3]), 4, policy, kVmid),
              Outcome::Complete);
    const uint64_t transactions = l2_.backing_write_transactions();
    const uint64_t writes = l2_.write_count();
    ASSERT_TRUE(l2_.try_write_private_dwords(stores(), policy, policy, kVmid));
    EXPECT_EQ(l2_.backing_write_transactions() - transactions, 3u);
    EXPECT_EQ(l2_.write_count() - writes, policy == Mtype::UC ? 0u : 3u);
    uint32_t value = 0;
    std::memcpy(&value, backing_.data() + 4, 4);
    EXPECT_EQ(value, values_[2]);
    ASSERT_EQ(l2_.read(kBase + 4, reinterpret_cast<uint8_t *>(&value), 4, policy, kVmid),
              Outcome::Complete);
    EXPECT_EQ(value, values_[2]);
    ASSERT_EQ(l2_.read(kBase + 8192 + 4, reinterpret_cast<uint8_t *>(&value), 4, Mtype::RW, kVmid),
              Outcome::Complete);
    EXPECT_EQ(value, alias_before);
    ASSERT_EQ(l2_.read(kBase + 8192 + 4, reinterpret_cast<uint8_t *>(&value), 4, Mtype::UC, kVmid),
              Outcome::Complete);
    EXPECT_EQ(value, values_[2]);
  }
}

TEST_F(VectorRamBatchTest, OwnerPolicySourceAliasAndDirtyRefusalsHaveNoEffects) {
  auto access = vm_.snapshot(address_space_);
  ASSERT_TRUE(access);
  for (const Owner owner : {Owner::Application, Owner::Driver, Owner::DriverSealedRam}) {
    map(owner, Mtype::UC);
    reset_bytes();
    const auto before = backing_;
    EXPECT_EQ(access->try_write_private_dwords(stores(), Mtype::RW, Mtype::UC),
              owner == Owner::DriverSealedRam);
    if (owner != Owner::DriverSealedRam) {
      EXPECT_EQ(backing_, before);
    }
  }
  map(Owner::DriverSealedRam, Mtype::RW);
  reset_bytes();
  const auto before = backing_;
  EXPECT_FALSE(access->try_write_private_dwords(stores(), Mtype::RW, Mtype::UC));
  auto overlapping_sources = stores();
  overlapping_sources[0].source = backing_.data() + 4;
  EXPECT_FALSE(access->try_write_private_dwords(overlapping_sources, Mtype::RW, Mtype::RW));
  EXPECT_EQ(backing_, before);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty;
  dirty.fill(0xa5);
  ASSERT_EQ(l2_.writeback_line(kBase, dirty.data(), Mtype::RW, kVmid), Outcome::Complete);
  const uint64_t transactions = l2_.backing_write_transactions();
  EXPECT_FALSE(l2_.try_write_private_dwords(stores(), Mtype::RW, Mtype::RW, kVmid));
  EXPECT_FALSE(l2_.try_write_private_dwords(stores(), Mtype::UC, Mtype::UC, kVmid));
  EXPECT_EQ(l2_.backing_write_transactions(), transactions);
  EXPECT_EQ(backing_, before);
  ASSERT_EQ(l2_.flush_line(kBase, kVmid), Outcome::Complete);
  EXPECT_TRUE(std::ranges::equal(dirty, std::span(backing_).first(dirty.size())));
}

TEST_F(VectorRamBatchTest, FirstRequestKeepsDirtyPrefixAndLaterFlushContents) {
  map(Owner::DriverSealedRam, Mtype::RW);
  std::array<uint64_t, 8> addresses;
  for (size_t lane = 0; lane < addresses.size(); ++lane)
    addresses[lane] = kBase + lane * 4;
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty;
  dirty.fill(0xa5);
  for (const Mtype policy : {Mtype::UC, Mtype::RW}) {
    std::array<uint8_t, 8192> ordinary_before_flush, ordinary_after_flush;
    uint64_t ordinary_transactions = 0;
    for (const bool allow : {false, true}) {
      reset_bytes();
      ASSERT_EQ(l2_.writeback_line(kBase, dirty.data(), Mtype::RW, kVmid), Outcome::Complete);
      const uint64_t before = l2_.backing_write_transactions();
      ASSERT_EQ(vector_.store(addresses.data(), 255, 4, 1,
                              reinterpret_cast<const uint8_t *>(values_.data()), policy, false, 8,
                              kVmid, 32, 0, {}, 4, allow),
                Outcome::Complete);
      if (!allow)
        ordinary_before_flush = backing_;
      else
        EXPECT_EQ(backing_, ordinary_before_flush);
      ASSERT_EQ(l2_.flush_all(), Outcome::Complete);
      if (!allow) {
        ordinary_after_flush = backing_;
        ordinary_transactions = l2_.backing_write_transactions() - before;
      } else {
        EXPECT_EQ(backing_, ordinary_after_flush);
        EXPECT_EQ(l2_.backing_write_transactions() - before, ordinary_transactions);
      }
      EXPECT_EQ(std::memcmp(backing_.data(), values_.data(), 8 * sizeof(uint32_t)), 0);
      EXPECT_TRUE(std::ranges::all_of(std::span(backing_).subspan(32, 96),
                                      [](uint8_t value) { return value == 0xa5; }));
      EXPECT_TRUE(faults_.empty());
    }
  }
}

TEST_F(VectorRamBatchTest, HintsNeverRefreshOrAuthorizeChangedMappings) {
  auto access = vm_.snapshot(address_space_);
  ASSERT_TRUE(access);
  rocjitsu::amdgpu::VmMtypeCache cache;
  for (const Mtype policy : {Mtype::UC, Mtype::RW}) {
    map(Owner::DriverSealedRam, policy);
    ASSERT_EQ(access->query_mtype(kBase, cache), policy);
    ASSERT_EQ(access->cached_private_ram_mtype(kBase, cache), policy);
    map(Owner::Application, policy);
    EXPECT_FALSE(access->cached_private_ram_mtype(kBase, cache));
    const auto before = backing_;
    EXPECT_FALSE(access->try_write_private_dwords(stores(), Mtype::RW, policy));
    EXPECT_EQ(backing_, before);
    ASSERT_EQ(access->query_mtype(kBase, cache), policy);
    EXPECT_FALSE(access->cached_private_ram_mtype(kBase, cache));
  }
  map(Owner::DriverSealedRam, Mtype::UC);
  ASSERT_EQ(access->query_mtype(kBase, cache), Mtype::UC);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(256);
  ASSERT_TRUE(adapter_.unregister_address_space(address_space_));
  address_space_ = vm_.register_address_space(kVmid, std::make_shared<IdentityAddressSpace>(256),
                                              physical, {}, true);
  ASSERT_TRUE(address_space_);
  EXPECT_FALSE(access->try_write_private_dwords(stores(), Mtype::RW, Mtype::UC));
  auto replacement = vm_.snapshot(address_space_);
  ASSERT_TRUE(replacement);
  EXPECT_FALSE(replacement->cached_private_ram_mtype(kBase, cache));
  EXPECT_FALSE(replacement->try_write_private_dwords(stores(), Mtype::RW, Mtype::UC));
  EXPECT_EQ(physical->write_calls, 0u);
}

TEST_F(VectorRamBatchTest, StrictSubextentFailuresRetainExactWrittenPrefix) {
  for (const uint32_t kind : {0u, 1u, 2u}) {
    for (const bool allow : {false, true}) {
      map(Owner::DriverSealedRam, Mtype::UC, kind == 0 ? 10 : 8192);
      if (kind == 1)
        process_.page_table_[kBase >> rocjitsu::KfdProcess::kPageShift].host_extents = {
            {backing_.data(), 6, 0, Owner::DriverSealedRam},
            {backing_.data() + 6, 58, 6, Owner::DriverSealedRam}};
      if (kind == 2)
        process_.page_table_[kBase >> rocjitsu::KfdProcess::kPageShift].host_extents = {
            {backing_.data(), 64, 0, Owner::DriverSealedRam},
            {backing_.data() + 8, 8, 8, Owner::DriverSealedRam}};
      reset_bytes();
      std::array<uint64_t, 8> addresses;
      for (size_t i = 0; i < addresses.size(); ++i)
        addresses[i] = kBase + i * 4;
      const uint64_t before = l2_.backing_write_transactions();
      EXPECT_EQ(vector_.store(addresses.data(), 255, 4, 1,
                              reinterpret_cast<const uint8_t *>(values_.data()), Mtype::UC, false,
                              8, kVmid, 32, 0, {}, 4, allow),
                Outcome::Faulted);
      const size_t prefix = kind == 1 ? 4 : 8;
      EXPECT_EQ(std::memcmp(backing_.data(), values_.data(), prefix), 0);
      EXPECT_TRUE(std::ranges::all_of(std::span(backing_).subspan(prefix),
                                      [](uint8_t value) { return value == 0x5a; }));
      EXPECT_EQ(faults_, (std::vector<uint64_t>{kBase + prefix}));
      EXPECT_EQ(l2_.backing_write_transactions() - before, prefix / 4 + 1);
    }
  }
}

TEST_F(VectorRamBatchTest, AdmissionPinsPageTableUntilMappingGuardAllowsCommit) {
  map();
  auto access = vm_.snapshot(address_space_);
  ASSERT_TRUE(access);
  const auto before = backing_;
  const auto request_mutex = process_.page_table_request_mutex();
  std::barrier start(2);
  bool copied = false;
  std::thread worker([&] {
    start.arrive_and_wait();
    copied = access->try_write_private_dwords(stores(), Mtype::RW, Mtype::UC);
  });
  auto mapping = rocjitsu::host_mapping_lock().lock_exclusive();
  start.arrive_and_wait();
  bool pinned = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!process_.page_table_mutex_.try_lock()) {
      pinned = true;
      break;
    }
    process_.page_table_mutex_.unlock();
    std::this_thread::yield();
  }
  const bool writable = request_mutex->try_lock();
  if (writable)
    request_mutex->unlock();
  const bool unchanged = backing_ == before;
  mapping.unlock();
  worker.join();
  EXPECT_TRUE(pinned);
  EXPECT_FALSE(writable);
  EXPECT_TRUE(unchanged);
  EXPECT_TRUE(copied);
  EXPECT_TRUE(faults_.empty());
}

TEST(VectorRamBatch, CacheInspectionDoesNotChangeReplacementOrder) {
  simdojo::Cache<7, 1, 2> cache;
  cache.allocate(0);
  cache.allocate(128);
  ASSERT_NE(cache.peek(0), nullptr);
  EXPECT_EQ(cache.peek(256), nullptr);
  simdojo::CacheTag evicted;
  cache.allocate(256, 0, &evicted);
  EXPECT_EQ(evicted.tag, 0u);
  EXPECT_EQ(cache.peek(0), nullptr);
  EXPECT_NE(cache.peek(128), nullptr);
}

TEST(VectorRamBatch, UnknownBackendAddsNoOptionalQueriesOrRepeatedStoreEffects) {
  using namespace rocjitsu::amdgpu;
  class ObservedPolicy final : public AddressSpaceTranslator {
  public:
    VmTranslationResult translate(uint64_t address, size_t size,
                                  VmAccessKind access) const override {
      ++queries;
      return inner.translate(address, size, access);
    }
    bool try_write_private_dwords(PhysicalMemoryAccess &, std::span<const VmRamDwordStore>, Mtype,
                                  Mtype) const override {
      ++optional_attempts;
      return false;
    }
    mutable uint32_t queries = 0, optional_attempts = 0;
    IdentityAddressSpace inner{256, Mtype::UC};
  };
  class ObservedMemory final : public PhysicalMemoryAccess {
  public:
    VmAccessOutcome read(VmMemoryDomain, uint64_t, std::span<std::byte>) override {
      ADD_FAILURE() << "UC store must not read backing";
      return VmAccessOutcome::Malformed;
    }
    VmAccessOutcome write(VmMemoryDomain, uint64_t address,
                          std::span<const std::byte> bytes) override {
      stores.emplace_back(address, bytes.size());
      if (stores.size() == 5 && result != VmAccessOutcome::Complete)
        return result;
      std::copy(bytes.begin(), bytes.end(), backing.begin() + address);
      return VmAccessOutcome::Complete;
    }
    VmAccessOutcome result = VmAccessOutcome::Complete;
    std::array<std::byte, 256> backing{};
    std::vector<std::pair<uint64_t, size_t>> stores;
  };
  for (const auto outcome :
       {VmAccessOutcome::Complete, VmAccessOutcome::Faulted, VmAccessOutcome::Unavailable}) {
    uint32_t ordinary_queries = 0;
    for (const bool allow : {false, true}) {
      GpuVm vm;
      auto policy = std::make_shared<ObservedPolicy>();
      auto physical = std::make_shared<ObservedMemory>();
      physical->result = outcome;
      const auto handle = vm.register_address_space(7, policy, physical, {}, true);
      ASSERT_TRUE(handle);
      GpuMemory memory("store_custom");
      L2Cache l2("store_custom_l2");
      l2.set_backing_memory(&memory);
      l2.set_gpu_vm(&vm);
      L1VectorCache l1(&l2);
      l1.set_gpu_vm(&vm);
      std::array<uint64_t, 8> addresses;
      std::array<uint32_t, 8> values;
      for (uint32_t lane = 0; lane < 8; ++lane) {
        addresses[lane] = 64 + lane * 4;
        values[lane] = 0x76540000u + lane;
      }
      EXPECT_EQ(l1.store(addresses.data(), 255, 4, 1,
                         reinterpret_cast<const uint8_t *>(values.data()), Mtype::UC, false, 8, 7,
                         32, 0, {}, 4, allow),
                outcome);
      const size_t complete = outcome == VmAccessOutcome::Complete ? 8 : 4;
      const size_t attempts = outcome == VmAccessOutcome::Complete ? 8 : 5;
      ASSERT_EQ(physical->stores.size(), attempts);
      for (size_t index = 0; index < attempts; ++index)
        EXPECT_EQ(physical->stores[index], (std::pair<uint64_t, size_t>{64 + index * 4, 4}));
      EXPECT_EQ(std::memcmp(physical->backing.data() + 64, values.data(), complete * 4), 0);
      EXPECT_TRUE(std::ranges::all_of(std::span(physical->backing).subspan(64 + complete * 4),
                                      [](std::byte value) { return value == std::byte{0}; }));
      EXPECT_EQ(l2.backing_write_transactions(), attempts);
      EXPECT_EQ(policy->optional_attempts, 0u);
      if (!allow)
        ordinary_queries = policy->queries;
      else
        EXPECT_EQ(policy->queries, ordinary_queries);
    }
  }
}

TEST(VectorRamBatch, RawRegistryReplacementRetainsOnlyOriginalRequestOwner) {
  using namespace rocjitsu::amdgpu;
  GpuMemory memory("store_owner");
  GpuVm vm;
  LegacyGpuVmAdapter adapter(vm, &memory);
  std::array<uint8_t, 128> backing{};
  LegacyPageTable table;
  table[8] = {backing.data(), Mtype::UC, backing.size(), 0};
  table[8].host_extents.front().owner = LegacyHostExtentOwner::DriverSealedRam;
  util::DistributedSharedMutex page_mutex;
  const auto destroyed = std::make_shared<bool>(false);
  auto request = std::shared_ptr<util::DistributedSharedMutex>(new util::DistributedSharedMutex,
                                                               [destroyed](auto *mutex) {
                                                                 *destroyed = true;
                                                                 delete mutex;
                                                               });
  const auto handle = adapter.register_address_space(
      7, {.page_table = &table, .page_table_mutex = &page_mutex, .request_mutex = request});
  ASSERT_TRUE(handle);
  auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  std::array<std::byte, 8> words;
  words.fill(std::byte{0x5a});
  ASSERT_TRUE(access->try_read_uncached_ram(8 * GpuMemory::PAGE_SIZE, words));
  EXPECT_EQ(words, (std::array<std::byte, 8>{}));
  adapter.address_space(7)->register_process(7, &table, &page_mutex, nullptr,
                                             std::make_shared<util::DistributedSharedMutex>());
  request.reset();
  EXPECT_FALSE(*destroyed);
  const uint32_t word = 0x12345678;
  const std::array<VmRamDwordStore, 2> stores{
      {{8 * GpuMemory::PAGE_SIZE, reinterpret_cast<const uint8_t *>(&word)},
       {8 * GpuMemory::PAGE_SIZE + 4, reinterpret_cast<const uint8_t *>(&word)}}};
  EXPECT_FALSE(access->try_write_private_dwords(stores, Mtype::UC, Mtype::UC));
  words.fill(std::byte{0x5a});
  EXPECT_FALSE(access->try_read_uncached_ram(8 * GpuMemory::PAGE_SIZE, words));
  EXPECT_TRUE(std::ranges::all_of(words, [](std::byte value) { return value == std::byte{0x5a}; }));
  EXPECT_FALSE(*destroyed);
  EXPECT_TRUE(std::ranges::all_of(backing, [](uint8_t value) { return value == 0; }));
  ASSERT_TRUE(adapter.unregister_address_space(handle));
  access.reset();
  EXPECT_TRUE(*destroyed);
}

TEST_F(ScalarRamBatchTest, SealedUncachedWidthsCopyOnceWithoutPopulatingCache) {
  map();
  for (const uint32_t width : {1u, 4u, 8u, 16u}) {
    for (uint32_t repeat = 0; repeat < 2; ++repeat) {
      backing_[0] ^= 0x5a;
      std::array<uint32_t, 16> words;
      words.fill(kSentinel);
      const uint64_t before = l2_.backing_read_transactions();
      ASSERT_EQ(l1_.load(kBase, width, words.data(), kVmid, true), Outcome::Complete);
      expect_words(words, width);
      EXPECT_EQ(l2_.backing_read_transactions() - before, 1u);
    }
  }
  EXPECT_TRUE(faults_.empty());
}

TEST_F(ScalarRamBatchTest, PrivateCopyUsesPageBoundsAndDeclinesGuestAliases) {
  map();
  auto access = vm_.snapshot(address_space_);
  ASSERT_TRUE(access);
  std::array<std::byte, 4096> copied{};
  for (std::size_t size : {128u, 4096u}) {
    EXPECT_TRUE(access->try_read_uncached_ram(kBase, std::span(copied).first(size)));
    EXPECT_EQ(std::memcmp(copied.data(), backing_.data(), size), 0);
  }
  const auto original = backing_;
  EXPECT_FALSE(
      access->try_read_uncached_ram(kBase, std::as_writable_bytes(std::span(backing_).first(128))));
  EXPECT_EQ(backing_, original);
  process_.unmap_pages(kBase, backing_.size());
  map(Owner::DriverSealedRam, Mtype::UC, 128);
  copied.fill(std::byte{0x5a});
  EXPECT_FALSE(access->try_read_uncached_ram(kBase, std::span(copied).first(256)));
  EXPECT_TRUE(
      std::ranges::all_of(copied, [](std::byte value) { return value == std::byte{0x5a}; }));
}

TEST_F(ScalarRamBatchTest, DisabledBatchAndUnprovedOwnersKeepDwordTransactions) {
  for (const Owner owner : {Owner::Application, Owner::Driver, Owner::DriverSealedRam}) {
    map(owner);
    for (const bool allow : {false, true}) {
      std::array<uint32_t, 16> words;
      words.fill(kSentinel);
      const uint64_t before = l2_.backing_read_transactions();
      ASSERT_EQ(l1_.load(kBase, 8, words.data(), kVmid, allow), Outcome::Complete);
      expect_words(words, 8);
      EXPECT_EQ(l2_.backing_read_transactions() - before,
                allow && owner == Owner::DriverSealedRam ? 1u : 8u);
    }
  }
}

TEST_F(ScalarRamBatchTest, RefusesAbsentMappingAndClippedExtentBeforeOriginalFault) {
  map(Owner::DriverSealedRam, Mtype::UC, 22);
  for (const bool allow : {false, true}) {
    faults_.clear();
    std::array<uint32_t, 16> words;
    words.fill(kSentinel);
    const uint64_t before = l2_.backing_read_transactions();
    EXPECT_EQ(l1_.load(kBase, 8, words.data(), kVmid, allow), Outcome::Faulted);
    expect_words(words, 5);
    EXPECT_EQ(faults_, (std::vector<uint64_t>{kBase + 20}));
    EXPECT_EQ(l2_.backing_read_transactions() - before, 6u);
  }
  auto access = vm_.snapshot(address_space_);
  ASSERT_TRUE(access);
  std::array<std::byte, 32> bytes;
  bytes.fill(std::byte{0xcc});
  faults_.clear();
  EXPECT_FALSE(access->try_read_uncached_ram(kBase, bytes));
  EXPECT_FALSE(access->try_read_uncached_ram(kBase + GpuMemory::PAGE_SIZE, bytes));
  EXPECT_TRUE(std::ranges::all_of(bytes, [](std::byte b) { return b == std::byte{0xcc}; }));
  EXPECT_TRUE(faults_.empty());
}

TEST_F(ScalarRamBatchTest, AlignmentLineAndPageCrossingsKeepOriginalChunks) {
  map();
  for (const uint32_t offset : {1u, 60u, static_cast<uint32_t>(GpuMemory::PAGE_SIZE - 4)}) {
    uint64_t original_reads = 0;
    for (const bool allow : {false, true}) {
      std::array<uint32_t, 16> words;
      words.fill(kSentinel);
      const uint64_t before = l2_.backing_read_transactions();
      ASSERT_EQ(l1_.load(kBase + offset, 8, words.data(), kVmid, allow), Outcome::Complete);
      expect_words(words, 8, offset);
      if (!allow)
        original_reads = l2_.backing_read_transactions() - before;
      else
        EXPECT_EQ(l2_.backing_read_transactions() - before, original_reads);
    }
  }
}

TEST_F(ScalarRamBatchTest, AdjacentSealedExtentsRetainDwordFaultBoundaries) {
  for (const uint32_t split : {2u, 6u}) {
    map();
    // Populate before first access rather than using map_pages, which merges
    // contiguous extents. The translation interface also accepts split PTEs.
    process_.page_table_[kBase >> rocjitsu::KfdProcess::kPageShift].host_extents = {
        {backing_.data(), split, 0, Owner::DriverSealedRam},
        {backing_.data() + split, 64 - split, split, Owner::DriverSealedRam}};
    for (const bool allow : {false, true}) {
      faults_.clear();
      std::array<uint32_t, 16> words;
      words.fill(kSentinel);
      const uint64_t reads = l2_.backing_read_transactions();
      EXPECT_EQ(l1_.load(kBase, 8, words.data(), kVmid, allow), Outcome::Faulted);
      expect_words(words, split / 4);
      EXPECT_EQ(faults_, (std::vector<uint64_t>{kBase + (split / 4) * 4}));
      EXPECT_EQ(l2_.backing_read_transactions() - reads, split / 4 + 1);
    }
  }
}

TEST_F(ScalarRamBatchTest, OverlappingSealedExtentsKeepSuccessfulDwordPrefix) {
  map();
  process_.page_table_[kBase >> rocjitsu::KfdProcess::kPageShift].host_extents = {
      {backing_.data(), 32, 0, Owner::DriverSealedRam},
      {backing_.data() + 8, 8, 8, Owner::DriverSealedRam}};
  for (const bool allow : {false, true}) {
    faults_.clear();
    std::array<uint32_t, 16> words;
    words.fill(kSentinel);
    const uint64_t reads = l2_.backing_read_transactions();
    EXPECT_EQ(l1_.load(kBase, 8, words.data(), kVmid, allow), Outcome::Faulted);
    expect_words(words, 2);
    EXPECT_EQ(faults_, (std::vector<uint64_t>{kBase + 8}));
    EXPECT_EQ(l2_.backing_read_transactions() - reads, 3u);
  }
}

TEST_F(ScalarRamBatchTest, ResidentCleanAndDirtyLinesKeepFlushBeforeRead) {
  map();
  std::array<uint8_t, L2Cache::LINE_SIZE> cached{};
  ASSERT_EQ(l2_.read(kBase, cached.data(), cached.size(), Mtype::RW, kVmid), Outcome::Complete);
  for (const bool dirty : {false, true}) {
    if (dirty) {
      cached.fill(0xa5);
      ASSERT_EQ(l2_.writeback_line(kBase, cached.data(), Mtype::RW, kVmid), Outcome::Complete);
    }
    std::array<uint32_t, 16> words;
    words.fill(kSentinel);
    const uint64_t reads = l2_.backing_read_transactions();
    const uint64_t writes = l2_.backing_write_transactions();
    ASSERT_EQ(l1_.load(kBase, 8, words.data(), kVmid, true), Outcome::Complete);
    expect_words(words, 8);
    EXPECT_EQ(l2_.backing_read_transactions() - reads, 8u);
    EXPECT_EQ(l2_.backing_write_transactions() - writes, dirty ? 1u : 0u);
    if (dirty) {
      EXPECT_EQ(words[0], 0xa5a5a5a5u);
    }
  }
}

TEST_F(ScalarRamBatchTest, CacheablePolicyAndRetiredSnapshotsDeclinePrivateCopy) {
  map(Owner::DriverSealedRam, Mtype::RW);
  auto access = vm_.snapshot(address_space_);
  ASSERT_TRUE(access);
  std::array<std::byte, 32> bytes;
  bytes.fill(std::byte{0xcc});
  EXPECT_FALSE(access->try_read_uncached_ram(kBase, bytes));
  std::array<uint32_t, 16> words;
  words.fill(kSentinel);
  ASSERT_EQ(l1_.load(kBase, 8, words.data(), kVmid, true), Outcome::Complete);
  expect_words(words, 8);
  const uint64_t reads = l2_.backing_read_transactions();
  ASSERT_EQ(l1_.load(kBase, 8, words.data(), kVmid, true), Outcome::Complete);
  EXPECT_EQ(l2_.backing_read_transactions(), reads);
  EXPECT_TRUE(vm_.unregister_address_space(address_space_));
  EXPECT_FALSE(access->try_read_uncached_ram(kBase, bytes));
  EXPECT_TRUE(std::ranges::all_of(bytes, [](std::byte b) { return b == std::byte{0xcc}; }));
  EXPECT_TRUE(faults_.empty());
}

TEST_F(ScalarRamBatchTest, OwnershipHintsFollowMutationEpochWithinQuantum) {
  map(Owner::Application);
  auto access = vm_.snapshot(address_space_);
  ASSERT_TRUE(access);
  rocjitsu::amdgpu::VmMtypeCache cache;
  rocjitsu::amdgpu::GpuVmAccessBatchGuard quantum;
  for (const Owner owner : {Owner::Application, Owner::DriverSealedRam, Owner::Driver,
                            Owner::DriverSealedRam, Owner::Application}) {
    map(owner);
    ASSERT_EQ(access->query_mtype(kBase, cache), Mtype::UC);
    EXPECT_EQ(access->cached_private_uc_hint(kBase, cache), owner == Owner::DriverSealedRam);
    std::array<uint32_t, 16> words;
    words.fill(kSentinel);
    const uint64_t reads = l2_.backing_read_transactions();
    ASSERT_EQ(l1_.load(kBase, 8, words.data(), kVmid, true), Outcome::Complete);
    expect_words(words, 8);
    EXPECT_EQ(l2_.backing_read_transactions() - reads, owner == Owner::DriverSealedRam ? 1u : 8u);
  }
  EXPECT_TRUE(faults_.empty());
}

TEST_F(ScalarRamBatchTest, StalePositiveHintNeverAuthorizesChangedOwnerOrPolicy) {
  map();
  auto access = vm_.snapshot(address_space_);
  ASSERT_TRUE(access);
  rocjitsu::amdgpu::VmMtypeCache cache;
  for (const bool change_owner : {true, false}) {
    map();
    ASSERT_EQ(access->query_mtype(kBase, cache), Mtype::UC);
    ASSERT_TRUE(access->cached_private_uc_hint(kBase, cache));
    map(change_owner ? Owner::Application : Owner::DriverSealedRam,
        change_owner ? Mtype::UC : Mtype::RW);
    // Deliberately retain the positive hint across the mutation. The copy
    // helper must prove current policy independently before any guest effect.
    std::array<uint32_t, 16> words;
    words.fill(kSentinel);
    const uint64_t reads = l2_.backing_read_transactions();
    EXPECT_FALSE(l2_.try_read_scalar_ram(kBase, words.data(), 8, kVmid));
    expect_words(words, 0);
    EXPECT_EQ(l2_.backing_read_transactions(), reads);
    EXPECT_TRUE(faults_.empty());
    EXPECT_FALSE(access->cached_private_uc_hint(kBase, cache));
    EXPECT_EQ(access->query_mtype(kBase, cache), change_owner ? Mtype::UC : Mtype::RW);
    EXPECT_FALSE(access->cached_private_uc_hint(kBase, cache));
  }
}

TEST_F(ScalarRamBatchTest, BindingReplacementDiscardsCachedOwnershipHint) {
  map();
  auto old_access = vm_.snapshot(address_space_);
  ASSERT_TRUE(old_access);
  rocjitsu::amdgpu::VmMtypeCache cache;
  ASSERT_EQ(old_access->query_mtype(kBase, cache), Mtype::UC);
  ASSERT_TRUE(old_access->cached_private_uc_hint(kBase, cache));
  rocjitsu::amdgpu::GpuVmAccessBatchGuard quantum;
  std::array<uint32_t, 16> words{};
  ASSERT_EQ(l1_.load(kBase, 8, words.data(), kVmid, true), Outcome::Complete);
  ASSERT_TRUE(vm_.unregister_address_space(address_space_));
  auto translator = std::make_shared<IdentityAddressSpace>(kBase + 8192, Mtype::UC);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(kBase + 8192);
  auto replacement = vm_.register_address_space(kVmid, translator, physical, {}, true);
  ASSERT_TRUE(replacement);
  auto new_access = vm_.snapshot(replacement);
  ASSERT_TRUE(new_access);
  EXPECT_FALSE(new_access->cached_private_uc_hint(kBase, cache));
  EXPECT_FALSE(old_access->cached_private_uc_hint(kBase, cache));
  EXPECT_EQ(new_access->query_mtype(kBase, cache), Mtype::UC);
  EXPECT_FALSE(new_access->cached_private_uc_hint(kBase, cache));
  EXPECT_FALSE(old_access->query_mtype(kBase, cache));
  EXPECT_FALSE(old_access->cached_private_uc_hint(kBase, cache));
  const uint64_t reads = l2_.backing_read_transactions();
  ASSERT_EQ(l1_.load(kBase, 8, words.data(), kVmid, true), Outcome::Complete);
  EXPECT_EQ(l2_.backing_read_transactions() - reads, 8u);
  EXPECT_EQ(physical->read_calls, 8u);
  EXPECT_TRUE(faults_.empty());
}

TEST(ScalarRamBatch, UnknownEligibilityAddsNoPolicyOrBackingObservations) {
  using namespace rocjitsu::amdgpu;
  class CountingPolicy final : public AddressSpaceTranslator {
  public:
    VmTranslationResult translate(uint64_t address, size_t size,
                                  VmAccessKind access) const override {
      ++translations;
      return inner_.translate(address, size, access);
    }
    bool try_read_uncached_ram(PhysicalMemoryAccess &, uint64_t,
                               std::span<std::byte>) const override {
      ++optional_attempts;
      return false;
    }
    mutable uint64_t translations = 0;
    mutable uint64_t optional_attempts = 0;

  private:
    IdentityAddressSpace inner_{256, Mtype::UC};
  };
  uint64_t original_translations = 0;
  for (const bool allow : {false, true}) {
    GpuVm vm;
    auto translator = std::make_shared<CountingPolicy>();
    auto physical = std::make_shared<ConfigurablePhysicalMemory>(256);
    const auto handle = vm.register_address_space(7, translator, physical, {}, true);
    ASSERT_TRUE(handle);
    auto access = vm.snapshot(handle);
    ASSERT_TRUE(access);
    VmMtypeCache empty_cache;
    EXPECT_FALSE(access->cached_private_uc_hint(64, empty_cache));
    EXPECT_FALSE(access->cached_private_uc_hint(64, empty_cache));
    EXPECT_EQ(translator->translations, 0u);
    EXPECT_EQ(physical->read_calls, 0u);
    GpuMemory memory("unknown_hint");
    L2Cache l2("unknown_hint_l2");
    l2.set_backing_memory(&memory);
    l2.set_gpu_vm(&vm);
    L1ScalarCache l1(&l2);
    l1.set_gpu_vm(&vm);
    GpuVmAccessBatchGuard quantum;
    std::array<uint32_t, 8> words{};
    ASSERT_EQ(l1.load(64, words.size(), words.data(), 7, allow), VmAccessOutcome::Complete);
    EXPECT_EQ(physical->read_calls, 8u);
    EXPECT_EQ(translator->optional_attempts, 0u);
    if (!allow)
      original_translations = translator->translations;
    else
      EXPECT_EQ(translator->translations, original_translations);
  }
}

TEST(ScalarRamBatch, CustomTransportKeepsEachReadEffectAndIncompletePrefix) {
  using Outcome = rocjitsu::amdgpu::VmAccessOutcome;
  class ObservedMemory final : public rocjitsu::amdgpu::PhysicalMemoryAccess {
  public:
    Outcome read(rocjitsu::amdgpu::VmMemoryDomain, uint64_t address,
                 std::span<std::byte> bytes) override {
      reads.emplace_back(address, bytes.size());
      if (reads.size() == 5 && incomplete != Outcome::Complete)
        return incomplete;
      const uint32_t value = 0x12340000 + static_cast<uint32_t>(reads.size());
      EXPECT_EQ(bytes.size(), sizeof(value));
      if (bytes.size() != sizeof(value))
        return Outcome::Malformed;
      std::memcpy(bytes.data(), &value, bytes.size());
      return Outcome::Complete;
    }
    Outcome write(rocjitsu::amdgpu::VmMemoryDomain, uint64_t, std::span<const std::byte>) override {
      ADD_FAILURE() << "scalar read unexpectedly wrote backing";
      return Outcome::Malformed;
    }
    bool try_read_contiguous(rocjitsu::amdgpu::VmMemoryDomain, uint64_t,
                             std::span<std::byte>) override {
      ADD_FAILURE() << "generic optional transport must not be used as private RAM proof";
      return false;
    }
    Outcome incomplete = Outcome::Complete;
    std::vector<std::pair<uint64_t, size_t>> reads;
  };
  for (const auto outcome : {Outcome::Complete, Outcome::Faulted, Outcome::Unavailable}) {
    for (const bool allow : {false, true}) {
      rocjitsu::amdgpu::GpuVm vm;
      auto translator = std::make_shared<IdentityAddressSpace>(256, Mtype::UC);
      auto physical = std::make_shared<ObservedMemory>();
      physical->incomplete = outcome;
      ASSERT_TRUE(vm.register_address_space(7, translator, physical, {}, true));
      GpuMemory memory("observed");
      L2Cache l2("observed_l2");
      l2.set_backing_memory(&memory);
      l2.set_gpu_vm(&vm);
      L1ScalarCache l1(&l2);
      l1.set_gpu_vm(&vm);
      std::array<uint32_t, 8> words;
      words.fill(0xdeadbeef);
      EXPECT_EQ(l1.load(64, words.size(), words.data(), 7, allow), outcome);
      const size_t completed = outcome == Outcome::Complete ? 8 : 4;
      const size_t attempts = outcome == Outcome::Complete ? 8 : 5;
      ASSERT_EQ(physical->reads.size(), attempts);
      for (size_t i = 0; i < attempts; ++i)
        EXPECT_EQ(physical->reads[i], (std::pair<uint64_t, size_t>{64 + i * 4, 4}));
      for (size_t i = 0; i < words.size(); ++i)
        EXPECT_EQ(words[i], i < completed ? 0x12340001u + i : 0xdeadbeefu);
      EXPECT_EQ(l2.backing_read_transactions(), attempts);
    }
  }
}

TEST_F(LegacySubPageCacheTest, VectorAccessCachesOnlyAccessibleBytesOfIncompleteLine) {
  constexpr uint64_t kAddress = kBase + 8;
  constexpr size_t kAccessSize = kBackingSize - 8;
  const uint64_t addresses[] = {kAddress};
  const std::array<uint8_t, kAccessSize> stored = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                                                   0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f};
  L1VectorCache l1(&l2_);
  l1.set_gpu_vm(&gpu_vm_);

  ASSERT_EQ(l1.store(addresses, /*lane_mask=*/1, sizeof(uint32_t), kAccessSize / sizeof(uint32_t),
                     stored.data(), Mtype::RW,
                     /*non_temporal=*/false, /*wf_size=*/1, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_TRUE(std::ranges::equal(stored, std::span(backing_).subspan(8, stored.size())));

  std::array<uint8_t, kAccessSize> replacement{};
  std::ranges::fill(replacement, uint8_t{0xa5});
  std::ranges::copy(replacement, backing_.begin() + 8);
  std::array<uint8_t, kAccessSize> observed{};
  ASSERT_EQ(l1.load(addresses, /*lane_mask=*/1, sizeof(uint32_t), kAccessSize / sizeof(uint32_t),
                    observed.data(), Mtype::RW,
                    /*non_temporal=*/false, /*request_l1_bypass=*/false, /*wf_size=*/1, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, stored);
  EXPECT_EQ(l2_.backing_read_transactions(), 0u)
      << "the valid suffix should remain cached after the write-through store";

  {
    [[maybe_unused]] auto maintenance = l2_.coherence_domain()->acquire_cache_maintenance(
        rocjitsu::amdgpu::DeviceCacheOperation::Invalidate);
  }
  ASSERT_EQ(l1.load(addresses, /*lane_mask=*/1, sizeof(uint32_t), kAccessSize / sizeof(uint32_t),
                    observed.data(), Mtype::RW,
                    /*non_temporal=*/false, /*request_l1_bypass=*/false, /*wf_size=*/1, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, replacement);
  EXPECT_EQ(l2_.backing_read_transactions(), 1u);

  const uint64_t crossing_addresses[] = {kBase + 12};
  std::ranges::fill(observed, uint8_t{0xcc});
  EXPECT_EQ(l1.load(crossing_addresses, /*lane_mask=*/1, sizeof(uint32_t),
                    kAccessSize / sizeof(uint32_t), observed.data(), Mtype::RW,
                    /*non_temporal=*/false, /*request_l1_bypass=*/false, /*wf_size=*/1, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Faulted);
  EXPECT_TRUE(std::ranges::all_of(observed, [](uint8_t byte) { return byte == 0xcc; }));

  const std::array<uint8_t, kBackingSize> before_invalid_store = backing_;
  EXPECT_EQ(l1.store(crossing_addresses, /*lane_mask=*/1, sizeof(uint32_t),
                     kAccessSize / sizeof(uint32_t), stored.data(), Mtype::RW,
                     /*non_temporal=*/false, /*wf_size=*/1, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(backing_, before_invalid_store);
}

TEST_F(LegacySubPageCacheTest, ScalarAccessCachesOnlyAccessibleBytesOfIncompleteLine) {
  constexpr uint64_t kValidAddress = kBase + kBackingSize - sizeof(uint32_t);
  constexpr uint64_t kCrossingAddress = kBase + kBackingSize - 2;
  constexpr uint32_t kStored = 0x11223344;
  constexpr uint32_t kReplacement = 0xaabbccdd;
  L1ScalarCache l1(&l2_);
  l1.set_gpu_vm(&gpu_vm_);

  ASSERT_EQ(l1.store(kValidAddress, /*num_dwords=*/1, &kStored, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  uint32_t backing_value = 0;
  std::memcpy(&backing_value, backing_.data() + kBackingSize - sizeof(uint32_t),
              sizeof(backing_value));
  ASSERT_EQ(backing_value, kStored);

  std::memcpy(backing_.data() + kBackingSize - sizeof(uint32_t), &kReplacement,
              sizeof(kReplacement));
  uint32_t observed = 0;
  ASSERT_EQ(l1.load(kValidAddress, /*num_dwords=*/1, &observed, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kStored);
  EXPECT_EQ(l2_.backing_read_transactions(), 0u)
      << "the valid suffix should remain cached after the write-through store";

  {
    [[maybe_unused]] auto maintenance = l2_.coherence_domain()->acquire_cache_maintenance(
        rocjitsu::amdgpu::DeviceCacheOperation::Invalidate);
  }
  ASSERT_EQ(l1.load(kValidAddress, /*num_dwords=*/1, &observed, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kReplacement);
  EXPECT_EQ(l2_.backing_read_transactions(), 1u);

  std::array<uint8_t, sizeof(uint32_t)> invalid_read{};
  std::ranges::fill(invalid_read, uint8_t{0xcc});
  EXPECT_EQ(l1.load_bytes(kCrossingAddress, invalid_read.size(), invalid_read.data(), kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Faulted);
  EXPECT_TRUE(std::ranges::all_of(invalid_read, [](uint8_t byte) { return byte == 0xcc; }));

  const std::array<uint8_t, kBackingSize> before_invalid_store = backing_;
  EXPECT_EQ(l1.store(kCrossingAddress, /*num_dwords=*/1, &kStored, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Faulted);
  EXPECT_EQ(backing_, before_invalid_store);
}

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

  RequestMtypeResolver request(&gpu_vm_, kVmid, mtype_cache_);
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
    RequestMtypeResolver request(&gpu_vm_, kVmid, mtype_cache_);
    ASSERT_EQ(request.at(kAddr), Mtype::RW);
  }
  ASSERT_TRUE(legacy_vm_.unregister_vmid(kVmid));
  address_space_ = {};
  address_space_ = legacy_vm_.register_address_space(
      kVmid, &replacement_process.page_table_, &replacement_process.page_table_mutex_,
      replacement_process.page_table_generation(), replacement_process.page_table_request_mutex());
  ASSERT_TRUE(address_space_);
  RequestMtypeResolver replacement_request(&gpu_vm_, kVmid, mtype_cache_);
  EXPECT_EQ(replacement_request.at(kAddr + 1), Mtype::UC);

  l1.load(kAddr, 1, &result, kVmid);
  EXPECT_EQ(result, kFirstReplacement);
  EXPECT_TRUE(legacy_vm_.unregister_vmid(kVmid));
  address_space_ = {};
}

TEST_F(L1CacheMtypeTest, VmidUnregistrationRevokesLiveResolverSnapshot) {
  map_pages(Mtype::UC, Mtype::UC);

  RequestMtypeResolver request(&gpu_vm_, kVmid, mtype_cache_);
  ASSERT_EQ(request.at(kAddr), Mtype::UC);
  ASSERT_TRUE(legacy_vm_.unregister_vmid(kVmid));
  address_space_ = {};
  EXPECT_EQ(request.at(kAddr + 1), Mtype::RW);
  EXPECT_FALSE(gpu_vm_.snapshot_vmid(kVmid));

  RequestMtypeResolver new_request(&gpu_vm_, kVmid, mtype_cache_);
  EXPECT_EQ(new_request.at(kAddr + 1), Mtype::RW);
}

TEST_F(L1CacheMtypeTest, SamePageHitDoesNotTakePageTableLocks) {
  map_pages(Mtype::UC, Mtype::RW);
  RequestMtypeResolver request(&gpu_vm_, kVmid, mtype_cache_);
  ASSERT_EQ(request.at(kBase), Mtype::UC);
  std::unique_lock request_lock(*process_.page_table_request_mutex());
  std::unique_lock table_lock(process_.page_table_mutex_);
  auto hit = std::async(std::launch::async, [&] { return request.at(kBase + 4); });
  const auto status = hit.wait_for(std::chrono::seconds(2));
  table_lock.unlock();
  request_lock.unlock();
  EXPECT_EQ(status, std::future_status::ready);
  EXPECT_EQ(hit.get(), Mtype::UC);
}

TEST_F(L1CacheMtypeTest, NewRequestsReusePolicyWithoutPageTableLocks) {
  write_words(kFirst, kSecond);
  map_pages(Mtype::RW, Mtype::RW);
  for (bool vector : {false, true}) {
    SCOPED_TRACE(vector ? "vector" : "scalar");
    L1ScalarCache scalar_cache(&l2_);
    L1VectorCache vector_cache(&l2_);
    scalar_cache.set_gpu_vm(&gpu_vm_);
    vector_cache.set_gpu_vm(&gpu_vm_);
    uint32_t observed = 0;
    auto load = [&] {
      if (vector)
        return vector_cache.load(&kAddr, /*lane_mask=*/1, sizeof(observed), /*num_elems=*/1,
                                 reinterpret_cast<uint8_t *>(&observed), Mtype::RW,
                                 /*non_temporal=*/false, /*request_l1_bypass=*/false,
                                 /*wf_size=*/1, kVmid);
      return scalar_cache.load(kAddr, 1, &observed, kVmid);
    };
    ASSERT_EQ(load(), rocjitsu::amdgpu::VmAccessOutcome::Complete);
    ASSERT_EQ(observed, kFirst);
    observed = 0;

    std::unique_lock request_lock(*process_.page_table_request_mutex());
    std::unique_lock table_lock(process_.page_table_mutex_);
    auto hit = std::async(std::launch::async, load);
    const auto status = hit.wait_for(std::chrono::seconds(2));
    table_lock.unlock();
    request_lock.unlock();
    EXPECT_EQ(status, std::future_status::ready);
    EXPECT_EQ(hit.get(), rocjitsu::amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(observed, kFirst);
  }
}

TEST_F(L1CacheMtypeTest, CachedPagePolicyDoesNotRetainPreviousInstructionPolicy) {
  map_pages(Mtype::RW, Mtype::RW);
  EXPECT_EQ(RequestMtypeResolver(&gpu_vm_, kVmid, mtype_cache_, Mtype::NT).at(kBase), Mtype::NT);
  EXPECT_EQ(RequestMtypeResolver(&gpu_vm_, kVmid, mtype_cache_, Mtype::RW).at(kBase), Mtype::RW);
  EXPECT_EQ(RequestMtypeResolver(&gpu_vm_, kVmid, mtype_cache_).at(kBase), Mtype::RW);

  process_.set_page_mtype(kBase, GpuMemory::PAGE_SIZE, Mtype::UC);
  EXPECT_EQ(RequestMtypeResolver(&gpu_vm_, kVmid, mtype_cache_, Mtype::NT).at(kBase), Mtype::UC);
}

TEST_F(L1CacheMtypeTest, InstructionPolicyIsPreservedAcrossHitsAndMutation) {
  map_pages(Mtype::RW, Mtype::RW);
  RequestMtypeResolver request(&gpu_vm_, kVmid, mtype_cache_, Mtype::NT);
  ASSERT_EQ(request.at(kBase), Mtype::NT);
  EXPECT_EQ(request.at(kBase + 4), Mtype::NT);
  process_.set_page_mtype(kBase, GpuMemory::PAGE_SIZE, Mtype::UC);
  EXPECT_EQ(request.at(kBase + 4), Mtype::UC);
  process_.set_page_mtype(kBase, GpuMemory::PAGE_SIZE, Mtype::RW);
  EXPECT_EQ(request.at(kBase + 4), Mtype::NT);
}

TEST_F(L1CacheMtypeTest, LiveResolverObservesUnmapAndRemapOnSamePage) {
  map_pages(Mtype::UC, Mtype::RW);
  RequestMtypeResolver request(&gpu_vm_, kVmid, mtype_cache_);
  ASSERT_EQ(request.at(kAddr), Mtype::UC);
  EXPECT_EQ(request.at(kAddr + 1), Mtype::UC);

  process_.unmap_pages(kBase, GpuMemory::PAGE_SIZE);
  EXPECT_EQ(request.at(kAddr), Mtype::RW);
  process_.map_pages(kBase, first_page_.data(), first_page_.size(), Mtype::CC);
  EXPECT_EQ(request.at(kAddr + 1), Mtype::CC);
}

TEST_F(L1CacheMtypeTest, LiveResolverSurvivesRegisteredProcessDestruction) {
  auto process = std::make_unique<rocjitsu::KfdProcess>(kVmid);
  process->map_pages(kBase, first_page_.data(), first_page_.size(), Mtype::UC);
  address_space_ = legacy_vm_.register_address_space(
      kVmid, {.page_table = &process->page_table_,
              .page_table_mutex = &process->page_table_mutex_,
              .page_table_generation = process->page_table_generation(),
              .request_mutex = process->page_table_request_mutex(),
              .mutation_epoch = process->page_table_mutation_epoch()});
  ASSERT_TRUE(address_space_);
  RequestMtypeResolver request(&gpu_vm_, kVmid, mtype_cache_);
  ASSERT_EQ(request.at(kAddr), Mtype::UC);
  ASSERT_TRUE(legacy_vm_.unregister_vmid(kVmid));
  address_space_ = {};
  process.reset();
  EXPECT_EQ(request.at(kAddr), Mtype::RW);

  map_pages(Mtype::CC, Mtype::RW);
  // A retired snapshot must not adopt a new owner of the same numeric VMID.
  EXPECT_EQ(request.at(kAddr), Mtype::RW);
  RequestMtypeResolver replacement(&gpu_vm_, kVmid, mtype_cache_);
  EXPECT_EQ(replacement.at(kAddr), Mtype::CC);
}

TEST_F(L1CacheMtypeTest, LiveResolverWithoutMutationTokenStillRefreshes) {
  map_pages(Mtype::UC, Mtype::RW);
  ASSERT_TRUE(legacy_vm_.unregister_vmid(kVmid));
  address_space_ = legacy_vm_.register_address_space(
      kVmid, {.page_table = &process_.page_table_,
              .page_table_mutex = &process_.page_table_mutex_,
              .page_table_generation = process_.page_table_generation(),
              .request_mutex = process_.page_table_request_mutex(),
              .mutation_epoch = {}});
  ASSERT_TRUE(address_space_);
  RequestMtypeResolver request(&gpu_vm_, kVmid, mtype_cache_);
  ASSERT_EQ(request.at(kAddr), Mtype::UC);
  process_.set_page_mtype(kBase, GpuMemory::PAGE_SIZE, Mtype::CC);
  EXPECT_EQ(request.at(kAddr + 1), Mtype::CC);
}

TEST_F(L1CacheMtypeTest, LiveResolverWithoutGenerationDoesNotCacheMutationToken) {
  map_pages(Mtype::UC, Mtype::RW);
  ASSERT_TRUE(legacy_vm_.unregister_vmid(kVmid));
  address_space_ = legacy_vm_.register_address_space(
      kVmid, {.page_table = &process_.page_table_,
              .page_table_mutex = &process_.page_table_mutex_,
              .page_table_generation = nullptr,
              .request_mutex = process_.page_table_request_mutex(),
              .mutation_epoch = process_.page_table_mutation_epoch()});
  ASSERT_TRUE(address_space_);
  RequestMtypeResolver request(&gpu_vm_, kVmid, mtype_cache_);
  ASSERT_EQ(request.at(kAddr), Mtype::UC);
  {
    // Legacy registrations may omit generation publication; they must keep
    // observing the table through the locked path on every lookup.
    std::unique_lock request_lock(*process_.page_table_request_mutex());
    std::unique_lock table_lock(process_.page_table_mutex_);
    process_.page_table_.at(kBase >> GpuMemory::PAGE_SHIFT).mtype = Mtype::CC;
  }
  EXPECT_EQ(request.at(kAddr + 1), Mtype::CC);
}

TEST_F(L1CacheMtypeTest, MutationTokenRefreshesPolicyWithoutGenerationPublication) {
  map_pages(Mtype::UC, Mtype::RW);
  auto epoch = std::make_shared<std::atomic<uint64_t>>(1);
  ASSERT_TRUE(legacy_vm_.unregister_vmid(kVmid));
  address_space_ = legacy_vm_.register_address_space(
      kVmid, {.page_table = &process_.page_table_,
              .page_table_mutex = &process_.page_table_mutex_,
              .page_table_generation = process_.page_table_generation(),
              .request_mutex = process_.page_table_request_mutex(),
              .mutation_epoch = epoch});
  ASSERT_TRUE(address_space_);
  RequestMtypeResolver request(&gpu_vm_, kVmid, mtype_cache_);
  ASSERT_EQ(request.at(kAddr), Mtype::UC);
  {
    // Model a mutation that invalidates snapshots and changes a PTE, then
    // throws before publishing the ordinary page-table generation.
    std::unique_lock request_lock(*process_.page_table_request_mutex());
    std::unique_lock table_lock(process_.page_table_mutex_);
    epoch->fetch_add(1, std::memory_order_release);
    process_.page_table_.at(kBase >> GpuMemory::PAGE_SHIFT).mtype = Mtype::CC;
  }
  EXPECT_EQ(request.at(kAddr + 1), Mtype::CC);
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
  EXPECT_EQ(l2_.backing_read_transactions(), 0u);
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
  EXPECT_EQ(l2_.backing_read_transactions(), 0u);
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
  EXPECT_EQ(l2_.backing_read_transactions(), 0u);
}

TEST(L2CacheTest, FullLineStoresSkipBackingReads) {
  constexpr uint64_t kAddr = 0x180000;
  constexpr uint32_t kLanes = L2Cache::LINE_SIZE / sizeof(uint32_t);
  constexpr uint64_t kLaneMask = ~uint64_t{0} >> (64 - kLanes);
  for (Mtype mtype : {Mtype::RW, Mtype::CC, Mtype::WB, Mtype::UC}) {
    for (bool through_l1 : {false, true}) {
      SCOPED_TRACE(static_cast<int>(mtype));
      SCOPED_TRACE(through_l1);
      GpuMemory memory("memory");
      L2Cache l2("l2");
      l2.set_backing_memory(&memory);
      L1VectorCache l1(&l2);
      std::array<uint64_t, kLanes> addrs{};
      std::array<uint32_t, kLanes> values{};
      std::array<uint32_t, kLanes> actual{};
      for (uint32_t lane = 0; lane < kLanes; ++lane)
        addrs[lane] = kAddr + lane * sizeof(uint32_t);

      for (uint32_t iteration = 0; iteration < 2; ++iteration) {
        for (uint32_t lane = 0; lane < kLanes; ++lane)
          values[lane] = 0x12340000 + iteration * kLanes + lane;
        const auto *bytes = reinterpret_cast<const uint8_t *>(values.data());
        const uint64_t reads_before = l2.backing_read_transactions();
        if (through_l1)
          l1.store(addrs.data(), /*lane_mask=*/kLaneMask, sizeof(uint32_t), /*num_elems=*/1, bytes,
                   mtype, /*non_temporal=*/false, /*wf_size=*/kLanes);
        else
          l2.write(kAddr, bytes, sizeof(values), mtype);
        EXPECT_EQ(l2.backing_read_transactions(), reads_before);
        memory.read_block(
            kAddr, std::span<uint8_t>(reinterpret_cast<uint8_t *>(actual.data()), sizeof(actual)));
        EXPECT_EQ(actual, values);
        if (through_l1)
          l1.load(addrs.data(), /*lane_mask=*/kLaneMask, sizeof(uint32_t), /*num_elems=*/1,
                  reinterpret_cast<uint8_t *>(actual.data()), mtype,
                  /*non_temporal=*/false, /*request_l1_bypass=*/false, /*wf_size=*/kLanes);
        else
          l2.read(kAddr, reinterpret_cast<uint8_t *>(actual.data()), sizeof(actual), mtype);
        EXPECT_EQ(actual, values);
      }
    }
  }
}

TEST(L2CacheTest, DisjointPartialStoresAcrossL2sNeverReadAllocate) {
  GpuMemory memory("memory");
  L2Cache l2a("l2a");
  L2Cache l2b("l2b");
  l2a.set_backing_memory(&memory);
  l2b.set_backing_memory(&memory);

  constexpr uint64_t kAddr = 0x190000;
  constexpr uint32_t kOffsetA = 2 * sizeof(uint32_t);
  constexpr uint32_t kOffsetB = 3 * sizeof(uint32_t);
  constexpr uint32_t kValueA = 0x11223344;
  constexpr uint32_t kValueB = 0xaabbccdd;
  std::array<uint8_t, L2Cache::LINE_SIZE> initial{};
  initial.fill(0x5a);
  memory.write_block(kAddr, initial);

  ASSERT_EQ(
      l2a.write(kAddr + kOffsetA, reinterpret_cast<const uint8_t *>(&kValueA), sizeof(kValueA)),
      rocjitsu::amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(
      l2b.write(kAddr + kOffsetB, reinterpret_cast<const uint8_t *>(&kValueB), sizeof(kValueB)),
      rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(l2a.backing_read_transactions(), 0u);
  EXPECT_EQ(l2b.backing_read_transactions(), 0u);

  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  ASSERT_EQ(l2a.read(kAddr, actual.data(), actual.size()),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  uint32_t actual_a = 0;
  uint32_t actual_b = 0;
  std::memcpy(&actual_a, actual.data() + kOffsetA, sizeof(actual_a));
  std::memcpy(&actual_b, actual.data() + kOffsetB, sizeof(actual_b));
  EXPECT_EQ(actual_a, kValueA);
  EXPECT_EQ(actual_b, kValueB);
  EXPECT_EQ(l2a.backing_read_transactions(), 1u);
}

TEST(L2CacheTest, FullLineStorePublishesDirtyEviction) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  constexpr uint64_t kAddr = 0x180000;
  constexpr uint64_t kSetStride = L2Cache::LINE_SIZE * L2Cache::NUM_SETS;
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  std::array<uint8_t, L2Cache::LINE_SIZE> replacement{};
  dirty.fill(0x55);
  replacement.fill(0xaa);
  for (uint32_t way = 0; way < L2Cache::ASSOCIATIVITY; ++way)
    l2.writeback_line(kAddr + way * kSetStride, dirty.data());

  const uint64_t destination = kAddr + L2Cache::ASSOCIATIVITY * kSetStride;
  const uint64_t reads_before = l2.backing_read_transactions();
  l2.write(destination, replacement.data(), replacement.size());
  EXPECT_EQ(l2.backing_read_transactions(), reads_before);
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  memory.read_block(kAddr, actual);
  EXPECT_EQ(actual, dirty);
  memory.read_block(destination, actual);
  EXPECT_EQ(actual, replacement);
}

TEST(L2CacheTest, FullLineStoreEvictsDirtyLineUnderItsOwningVmid) {
  constexpr uint32_t kVmidA = 7;
  constexpr uint32_t kVmidB = 11;
  constexpr uint64_t kAddr = 0x180000;
  constexpr uint64_t kSetStride = L2Cache::LINE_SIZE * L2Cache::NUM_SETS;
  std::array<std::array<uint8_t, GpuMemory::PAGE_SIZE>, L2Cache::ASSOCIATIVITY> pages_a{};
  std::array<uint8_t, GpuMemory::PAGE_SIZE> page_b{};
  rocjitsu::KfdProcess process_a(kVmidA), process_b(kVmidB);
  rocjitsu::test::LegacyGpuMemoryFixture memory("memory");
  for (uint32_t way = 0; way < pages_a.size(); ++way)
    process_a.map_pages(kAddr + way * kSetStride, pages_a[way].data(), pages_a[way].size());
  process_b.map_pages(kAddr, page_b.data(), page_b.size());
  for (auto *process : {&process_a, &process_b})
    memory.register_process(process == &process_a ? kVmidA : kVmidB, &process->page_table_,
                            &process->page_table_mutex_, process->page_table_generation(),
                            process->page_table_request_mutex());
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&memory.gpu_vm());
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{}, replacement{}, actual{};
  dirty.fill(0x55);
  replacement.fill(0xaa);
  for (uint32_t way = 0; way < pages_a.size(); ++way)
    l2.writeback_line(kAddr + way * kSetStride, dirty.data(), Mtype::RW, kVmidA);
  EXPECT_EQ(pages_a[0][0], 0u);
  const uint64_t reads_before = l2.backing_read_transactions();
  // The same GPU VA misses because the store belongs to a different process.
  l2.write(kAddr, replacement.data(), replacement.size(), Mtype::RW, kVmidB);
  EXPECT_EQ(l2.backing_read_transactions(), reads_before);
  memory.read_block(kAddr, actual, kVmidA);
  EXPECT_EQ(actual, dirty);
  memory.read_block(kAddr, actual, kVmidB);
  EXPECT_EQ(actual, replacement);
  l2.read(kAddr, actual.data(), actual.size(), Mtype::RW, kVmidB);
  EXPECT_EQ(actual, replacement);
  l2.read(kAddr, actual.data(), actual.size(), Mtype::RW, kVmidA);
  EXPECT_EQ(actual, dirty);
}

TEST(L2CacheTest, FullLineStoreReplacesResidentDirtyBytes) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  constexpr uint64_t kAddr = 0x180000;
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{}, replacement{}, external{}, actual{};
  dirty.fill(0x55);
  replacement.fill(0xaa);
  external.fill(0x33);
  l2.writeback_line(kAddr, dirty.data());
  const uint64_t reads_before = l2.backing_read_transactions();
  l2.write(kAddr, replacement.data(), replacement.size());
  EXPECT_EQ(l2.backing_read_transactions(), reads_before);
  l2.read(kAddr, actual.data(), actual.size());
  EXPECT_EQ(actual, replacement);
  memory.read_block(kAddr, actual);
  EXPECT_EQ(actual, replacement);
  // A subsequent flush must not publish stale dirty bytes over a host update.
  memory.write_block(kAddr, external);
  l2.flush_line(kAddr);
  memory.read_block(kAddr, actual);
  EXPECT_EQ(actual, external);
}

TEST(L2CacheTest, UnalignedVectorStoreSkipsBackingReads) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  L1VectorCache l1(&l2);
  constexpr uint64_t kAddr = 0x180000;
  std::array<uint8_t, 3 * L2Cache::LINE_SIZE> initial{};
  initial.fill(0xa5);
  memory.write_block(kAddr, initial);
  std::array<uint8_t, 2 * L2Cache::LINE_SIZE> values{};
  for (uint32_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<uint8_t>(i);
  std::array<uint8_t, 3 * L2Cache::LINE_SIZE> expected = initial;
  std::ranges::copy(values, expected.begin() + sizeof(uint32_t));
  const uint64_t addrs[] = {kAddr + sizeof(uint32_t)};
  l1.store(addrs, /*lane_mask=*/1, sizeof(uint32_t), values.size() / sizeof(uint32_t),
           values.data(), Mtype::RW,
           /*non_temporal=*/false, /*wf_size=*/1);
  EXPECT_EQ(l2.backing_read_transactions(), 0u);
  std::array<uint8_t, initial.size()> actual{};
  memory.read_block(kAddr, actual);
  EXPECT_EQ(actual, expected);
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

TEST(L2CacheThreadingTest, BatchedAndIndividualAtomicsShareCoherence) {
  GpuMemory memory("memory");
  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  L2Cache l2a("l2a", coherence), l2b("l2b", coherence);
  l2a.set_backing_memory(&memory);
  l2b.set_backing_memory(&memory);
  constexpr uint64_t kTarget = 0x280000;
  constexpr uint32_t kBatches = 64, kLanes = 32;
  memory.write32(kTarget, 0);
  const auto increment = [](uint8_t *target, uint32_t offset) {
    uint32_t value;
    std::memcpy(&value, target + offset, sizeof(value));
    ++value;
    std::memcpy(target + offset, &value, sizeof(value));
  };
  std::barrier start(2);
  std::atomic<unsigned> failures = 0;
  std::thread batched([&] {
    start.arrive_and_wait();
    for (uint32_t batch = 0; batch < kBatches; ++batch) {
      auto boundary = coherence->acquire_atomic_boundary();
      for (uint32_t lane = 0; lane < kLanes; ++lane)
        if (l2a.atomic_rmw(boundary, kTarget, sizeof(uint32_t), increment, 0, std::nullopt) !=
            rocjitsu::amdgpu::VmAccessOutcome::Complete)
          ++failures;
    }
  });
  start.arrive_and_wait();
  for (uint32_t i = 0; i < kBatches * kLanes; ++i)
    if (l2b.atomic_rmw(kTarget, sizeof(uint32_t), increment) !=
        rocjitsu::amdgpu::VmAccessOutcome::Complete)
      ++failures;
  batched.join();
  EXPECT_EQ(failures.load(), 0u);
  EXPECT_EQ(memory.read32(kTarget), 2 * kBatches * kLanes);
}

TEST(L2CacheTest, AtomicBatchRejectsWrongDomainAndMovedBoundary) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  auto other = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  bool called = false;
  auto mutation = [&](uint8_t *, uint32_t) { called = true; };
  {
    auto wrong_boundary = other->acquire_atomic_boundary();
    EXPECT_EQ(l2.atomic_rmw(wrong_boundary, 0x1000, sizeof(uint32_t), mutation, 0, std::nullopt),
              rocjitsu::amdgpu::VmAccessOutcome::Malformed);
  }
  {
    auto boundary = l2.coherence_domain()->acquire_atomic_boundary();
    auto moved = std::move(boundary);
    EXPECT_EQ(l2.atomic_rmw(boundary, 0x1000, sizeof(uint32_t), mutation, 0, std::nullopt),
              rocjitsu::amdgpu::VmAccessOutcome::Malformed);
    EXPECT_FALSE(called);
    EXPECT_EQ(l2.atomic_rmw(moved, 0x1000, sizeof(uint32_t), mutation, 0, std::nullopt),
              rocjitsu::amdgpu::VmAccessOutcome::Complete);
    EXPECT_TRUE(called);
  }
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

  std::ranges::copy(initial, backing.begin());
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
  std::ranges::copy_n(backing.begin(), actual.size(), actual.begin());
  EXPECT_EQ(actual, replacement);
  l2.flush_line(kVaA, kVmidA);
  std::ranges::copy_n(backing.begin(), actual.size(), actual.begin());
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

  std::ranges::copy(initial_a, backing_a.begin());
  std::ranges::copy(initial_b, backing_b.begin());
  l2.read(kAddr, actual.data(), actual.size(), Mtype::RW, kVmidA);
  ASSERT_EQ(actual, initial_a);
  l2.read(kAddr, actual.data(), actual.size(), Mtype::RW, kVmidB);
  ASSERT_EQ(actual, initial_b);
  l2.writeback_line(kAddr, dirty_a.data(), Mtype::RW, kVmidA);

  std::ranges::copy(replacement_b, backing_b.begin());
  l2.invalidate_range(kAddr, L2Cache::LINE_SIZE, kVmidB);
  l2.read(kAddr, actual.data(), actual.size(), Mtype::RW, kVmidB);
  EXPECT_EQ(actual, replacement_b);

  l2.flush_line(kAddr, kVmidA);
  std::ranges::copy_n(backing_a.begin(), actual.size(), actual.begin());
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

TEST(L2DiagnosticCounters, JoinedTotalsCoverDistributedAndSharedLines) {
  using namespace rocjitsu::amdgpu;
  class StatelessMemory final : public PhysicalMemoryAccess {
  public:
    VmAccessOutcome read(VmMemoryDomain, uint64_t, std::span<std::byte> bytes) override {
      std::ranges::fill(bytes, std::byte{0});
      return VmAccessOutcome::Complete;
    }
    VmAccessOutcome write(VmMemoryDomain, uint64_t, std::span<const std::byte>) override {
      return VmAccessOutcome::Complete;
    }
  };
  constexpr uint32_t kThreads = 8, kOperations = 64;
  for (const bool distributed : {false, true}) {
    GpuVm vm;
    auto translator = std::make_shared<IdentityAddressSpace>(1u << 20);
    auto physical = std::make_shared<StatelessMemory>();
    ASSERT_TRUE(vm.register_address_space(7, translator, physical, {}, true));
    GpuMemory memory("diagnostic_memory");
    L2Cache l2("diagnostic_l2");
    l2.set_backing_memory(&memory);
    l2.set_gpu_vm(&vm);
    std::barrier start(kThreads);
    std::atomic<bool> succeeded{true};
    std::vector<std::thread> workers;
    for (uint32_t thread = 0; thread < kThreads; ++thread) {
      workers.emplace_back([&, thread] {
        const uint32_t value = thread;
        start.arrive_and_wait();
        for (uint32_t i = 0; i < kOperations; ++i) {
          const uint64_t address = distributed ? (thread + i * kThreads) * L2Cache::LINE_SIZE : 0;
          for (const auto policy : {Mtype::RW, Mtype::UC}) {
            if (l2.write(address, reinterpret_cast<const uint8_t *>(&value), sizeof(value), policy,
                         7) != VmAccessOutcome::Complete)
              succeeded.store(false, std::memory_order_relaxed);
          }
        }
      });
    }
    for (auto &worker : workers)
      worker.join();
    ASSERT_TRUE(succeeded.load(std::memory_order_relaxed));
    EXPECT_EQ(l2.write_count(), kThreads * kOperations);
    EXPECT_EQ(l2.backing_write_transactions(), 2u * kThreads * kOperations);
    EXPECT_EQ(l2.backing_read_transactions(), 0u);
  }
}

TEST(L2DiagnosticCounters, ReentrantQueriesKeepAttemptAndSuccessBoundaries) {
  using namespace rocjitsu::amdgpu;
  class ObservedMemory final : public PhysicalMemoryAccess {
  public:
    void observe() {
      snapshots.push_back({cache->backing_read_transactions(), cache->backing_write_transactions(),
                           cache->write_count()});
    }
    VmAccessOutcome read(VmMemoryDomain, uint64_t, std::span<std::byte> bytes) override {
      observe();
      std::ranges::fill(bytes, std::byte{0});
      return VmAccessOutcome::Complete;
    }
    VmAccessOutcome write(VmMemoryDomain, uint64_t, std::span<const std::byte>) override {
      observe();
      return write_outcome;
    }
    VmAccessOutcome atomic_modify(VmMemoryDomain, uint64_t, uint32_t width,
                                  const AtomicMutation &mutation) override {
      observe();
      std::array<std::byte, sizeof(uint32_t)> bytes{};
      if (width != bytes.size())
        return VmAccessOutcome::Malformed;
      mutation(bytes);
      return VmAccessOutcome::Complete;
    }
    L2Cache *cache = nullptr;
    VmAccessOutcome write_outcome = VmAccessOutcome::Complete;
    std::vector<std::array<uint64_t, 3>> snapshots;
  };
  GpuVm vm;
  auto translator = std::make_shared<IdentityAddressSpace>(1u << 20);
  auto physical = std::make_shared<ObservedMemory>();
  ASSERT_TRUE(vm.register_address_space(7, translator, physical, {}, true));
  GpuMemory memory("diagnostic_observed_memory");
  L2Cache l2("diagnostic_observed_l2");
  physical->cache = &l2;
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&vm);
  const uint32_t value = 0x12345678;
  const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
  ASSERT_EQ(l2.write(0x68000, bytes, sizeof(value), Mtype::RW, 7), VmAccessOutcome::Complete);
  physical->write_outcome = VmAccessOutcome::Faulted;
  EXPECT_EQ(l2.write(0xe0000, bytes, sizeof(value), Mtype::RW, 7), VmAccessOutcome::Faulted);
  physical->write_outcome = VmAccessOutcome::Complete;
  ASSERT_EQ(l2.write(0x48000, bytes, sizeof(value), Mtype::UC, 7), VmAccessOutcome::Complete);
  uint32_t loaded = 1;
  ASSERT_EQ(l2.read(0x50000, reinterpret_cast<uint8_t *>(&loaded), sizeof(loaded), Mtype::UC, 7),
            VmAccessOutcome::Complete);
  ASSERT_EQ(l2.atomic_rmw(
                0x58000, sizeof(value), [](uint8_t *, uint32_t) {}, 7),
            VmAccessOutcome::Complete);
  const std::vector<std::array<uint64_t, 3>> expected{
      {0, 1, 0}, {0, 2, 1}, {0, 3, 1}, {1, 3, 1}, {2, 4, 1}};
  EXPECT_EQ(physical->snapshots, expected);
  ASSERT_EQ(l2.flush_all(), VmAccessOutcome::Complete);
  l2.set_coherence_domain(std::make_shared<DeviceCacheCoherence>());
  EXPECT_EQ(l2.backing_read_transactions(), 2u);
  EXPECT_EQ(l2.backing_write_transactions(), 4u);
  EXPECT_EQ(l2.write_count(), 1u);
}

} // namespace
