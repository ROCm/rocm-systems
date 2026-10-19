// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "legacy_gpu_memory_fixture.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/completion_tracker.h"
#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/hbm_controller.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_side_cache.h"
#include "rocjitsu/vm/amdgpu/prepared_canonical_writeback.h"
#include "rocjitsu/vm/amdgpu/request_mtype_resolver.h"
#include "rocjitsu/vm/fabric_backing.h"
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
#include <format>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <linux/memfd.h>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {

class L2CacheTestAccess {
public:
  static void
  store_prepared_dirty_line(L2Cache &cache, uint64_t line_address, std::span<const uint8_t> data,
                            uint32_t vmid, uint64_t cache_identity, uint64_t generation,
                            std::span<const std::pair<uint32_t, uint32_t>> dirty_ranges) {
    assert(L2Cache::CacheStore::line_address(line_address) == line_address);
    assert(data.size() == L2Cache::LINE_SIZE);
    auto maintenance_lock = cache.acquire_cache_access();
    std::lock_guard set_lock(cache.set_mutex(line_address));
    simdojo::CacheTag *tag = cache.cache_.allocate(line_address, vmid);
    cache.cache_.write_line(line_address, data.data(), 0, L2Cache::LINE_SIZE, vmid);
    tag->cache_identity = cache_identity;
    tag->admission_generation = generation;
    tag->dirty = true;
    tag->coherence = simdojo::CoherenceState::MODIFIED;
    for (const auto &[offset, size] : dirty_ranges)
      cache.mark_dirty_bytes(line_address, offset, size, vmid);
  }

  static void store_prepared_clean_line(L2Cache &cache, uint64_t line_address,
                                        std::span<const uint8_t> data, uint32_t vmid,
                                        uint64_t cache_identity, uint64_t generation) {
    assert(L2Cache::CacheStore::line_address(line_address) == line_address);
    assert(data.size() == L2Cache::LINE_SIZE);
    auto maintenance_lock = cache.acquire_cache_access();
    std::lock_guard set_lock(cache.set_mutex(line_address));
    simdojo::CacheTag *tag = cache.cache_.allocate(line_address, vmid);
    cache.cache_.write_line(line_address, data.data(), 0, L2Cache::LINE_SIZE, vmid);
    tag->cache_identity = cache_identity;
    tag->admission_generation = generation;
    tag->coherence = simdojo::CoherenceState::SHARED;
  }

  static void rebind_prepared_dirty_line(L2Cache &cache, uint64_t line_address,
                                         std::span<const uint8_t> data, uint32_t vmid,
                                         uint64_t cache_identity, uint64_t generation) {
    assert(L2Cache::CacheStore::line_address(line_address) == line_address);
    assert(data.size() == L2Cache::LINE_SIZE);
    auto maintenance_lock = cache.acquire_cache_access();
    std::lock_guard set_lock(cache.set_mutex(line_address));
    simdojo::CacheTag *tag = nullptr;
    ASSERT_TRUE(cache.cache_.lookup(line_address, &tag, vmid));
    ASSERT_NE(tag, nullptr);
    cache.cache_.write_line(line_address, data.data(), 0, L2Cache::LINE_SIZE, vmid);
    tag->cache_identity = cache_identity;
    tag->admission_generation = generation;
  }

  static void replace_prepared_dirty_line_bytes(L2Cache &cache, uint64_t line_address,
                                                std::span<const uint8_t> data, uint32_t vmid) {
    assert(L2Cache::CacheStore::line_address(line_address) == line_address);
    assert(data.size() == L2Cache::LINE_SIZE);
    auto maintenance_lock = cache.acquire_cache_access();
    std::lock_guard set_lock(cache.set_mutex(line_address));
    simdojo::CacheTag *tag = nullptr;
    ASSERT_TRUE(cache.cache_.lookup(line_address, &tag, vmid));
    ASSERT_NE(tag, nullptr);
    const simdojo::CacheTag original = *tag;
    cache.cache_.write_line(line_address, data.data(), 0, L2Cache::LINE_SIZE, vmid);
    EXPECT_EQ(tag->cache_identity, original.cache_identity);
    EXPECT_EQ(tag->admission_generation, original.admission_generation);
    EXPECT_EQ(tag->coherence_epoch, original.coherence_epoch);
    EXPECT_EQ(tag->valid_bytes, original.valid_bytes);
    EXPECT_EQ(tag->coherence, original.coherence);
    EXPECT_EQ(tag->dirty, original.dirty);
  }

  static bool dirty_byte(L2Cache &cache, uint64_t line_address, uint32_t vmid, uint32_t offset) {
    auto maintenance_lock = cache.acquire_cache_access();
    std::lock_guard set_lock(cache.set_mutex(line_address));
    std::lock_guard mask_lock(cache.dirty_bytes_mutex_);
    const auto position = cache.dirty_bytes_.find({vmid, line_address});
    return position != cache.dirty_bytes_.end() &&
           (position->second[offset / 64] & (uint64_t{1} << (offset % 64))) != 0;
  }

  static bool dirty_byte_without_epoch_reconciliation(L2Cache &cache, uint64_t line_address,
                                                      uint32_t vmid, uint32_t offset) {
    std::shared_lock maintenance_lock(cache.maintenance_mutex_);
    std::lock_guard set_lock(cache.set_mutex(line_address));
    std::lock_guard mask_lock(cache.dirty_bytes_mutex_);
    const auto position = cache.dirty_bytes_.find({vmid, line_address});
    return position != cache.dirty_bytes_.end() &&
           (position->second[offset / 64] & (uint64_t{1} << (offset % 64))) != 0;
  }

  static bool resident(L2Cache &cache, uint64_t line_address, uint32_t vmid) {
    auto maintenance_lock = cache.acquire_cache_access();
    std::lock_guard set_lock(cache.set_mutex(line_address));
    return cache.cache_.peek(line_address, vmid) != nullptr;
  }

  static uint64_t recorded_epoch(L2Cache &cache) {
    std::shared_lock lock(cache.maintenance_mutex_);
    return cache.coherence_epoch_;
  }

  static bool cache_access_available(L2Cache &cache) {
    if (!cache.maintenance_mutex_.try_lock_shared())
      return false;
    cache.maintenance_mutex_.unlock_shared();
    return true;
  }

  static auto hold_cache_access(L2Cache &cache) { return cache.acquire_cache_access(); }
};

class L1ScalarCacheTestAccess {
public:
  static void store_clean_line(L1ScalarCache &cache, uint64_t address, uint32_t vmid,
                               uint64_t generation, uint64_t identity) {
    std::lock_guard lock(cache.maintenance_mutex_);
    simdojo::CacheTag *tag = cache.cache_.allocate(address, vmid);
    tag->admission_generation = generation;
    tag->cache_identity = identity;
  }

  static bool resident(L1ScalarCache &cache, uint64_t address, uint32_t vmid) {
    std::lock_guard lock(cache.maintenance_mutex_);
    return cache.cache_.peek(address, vmid) != nullptr;
  }

  static bool cache_access_available(L1ScalarCache &cache) {
    auto gate = DeviceCacheCoherence::try_acquire_l1_access(cache.registered_domain_.get());
    if (cache.registered_domain_ && !gate.owns_lock())
      return false;
    const bool available = cache.maintenance_mutex_.try_lock();
    if (available)
      cache.maintenance_mutex_.unlock();
    return available;
  }

  static std::shared_lock<util::DistributedSharedMutex>
  hold_l1_access(const DeviceCacheCoherence &domain) {
    return DeviceCacheCoherence::acquire_l1_access(&domain);
  }
};

class L1VectorCacheTestAccess {
public:
  static void store_clean_line(L1VectorCache &cache, uint64_t address, uint32_t vmid,
                               uint64_t generation, uint64_t identity) {
    std::lock_guard lock(cache.maintenance_mutex_);
    simdojo::CacheTag *tag = cache.cache_.allocate(address, vmid);
    tag->admission_generation = generation;
    tag->cache_identity = identity;
  }

  static bool resident(L1VectorCache &cache, uint64_t address, uint32_t vmid) {
    std::lock_guard lock(cache.maintenance_mutex_);
    return cache.cache_.peek(address, vmid) != nullptr;
  }

  static bool cache_access_available(L1VectorCache &cache) {
    auto gate = DeviceCacheCoherence::try_acquire_l1_access(cache.registered_domain_.get());
    if (cache.registered_domain_ && !gate.owns_lock())
      return false;
    const bool available = cache.maintenance_mutex_.try_lock();
    if (available)
      cache.maintenance_mutex_.unlock();
    return available;
  }
};

class PreparedBackingOwner final : public FabricBacking::Owner {
public:
  static constexpr uint64_t kBaseOffset = 0x2000;
  static constexpr uint64_t kBackingId = 57;

  [[nodiscard]] uint64_t base_offset() const noexcept override { return kBaseOffset; }
  [[nodiscard]] size_t extent() const noexcept override { return bytes_.size(); }
  [[nodiscard]] uint64_t backing_id() const noexcept override { return kBackingId; }

  [[nodiscard]] bool read(uint64_t offset, std::span<std::byte> result) noexcept override {
    if (!contains(offset, result.size()))
      return false;
    std::copy_n(bytes_.begin() + (offset - kBaseOffset), result.size(), result.begin());
    return true;
  }

  [[nodiscard]] bool write(uint64_t offset, std::span<const std::byte> data) noexcept override {
    ++write_attempts;
    if (!contains(offset, data.size()) || write_attempts == fail_on_attempt)
      return false;
    std::copy(data.begin(), data.end(), bytes_.begin() + (offset - kBaseOffset));
    return true;
  }

  std::array<std::byte, 256> bytes_{};
  size_t write_attempts = 0;
  size_t fail_on_attempt = 0;

private:
  [[nodiscard]] bool contains(uint64_t offset, size_t size) const noexcept {
    if (offset < kBaseOffset)
      return false;
    const uint64_t relative = offset - kBaseOffset;
    return relative <= bytes_.size() && size <= bytes_.size() - relative;
  }
};

class PreparedCanonicalWritebackTestAccess {
public:
  static PreparedCanonicalWriteback make(
      uint64_t address, size_t extent, uint32_t vmid, uint64_t cache_identity, uint64_t generation,
      FabricBacking backing,
      std::shared_ptr<VmTransferEffectGate> gate = std::make_shared<VmTransferEffectGate>(),
      std::function<void()> release = [] {}, std::span<std::byte> direct_bytes = {}) {
    std::vector<PreparedCanonicalWriteback::Entry> entries;
    entries.emplace_back(
        address, extent, vmid,
        VmCacheNamespace{.address_space = {.slot = 1, .generation = 1}, .translation_epoch = 1},
        cache_identity, generation, backing.backing_id(), backing.offset(), std::move(backing),
        direct_bytes);
    return PreparedCanonicalWriteback(std::move(entries), MemoryIngressPermit(std::move(release)),
                                      std::move(gate));
  }
};

class MemorySideCacheTestAccess {
public:
  static void store_prepared_dirty_line(MemorySideCache &cache, uint64_t line_address,
                                        std::span<const uint8_t> data, uint32_t vmid,
                                        uint64_t cache_identity, uint64_t generation) {
    assert(CacheStore::line_address(line_address) == line_address);
    assert(data.size() == MemorySideCache::LINE_SIZE);
    std::unique_lock access_lock(cache.access_gate_);
    std::lock_guard stripe_lock(cache.stripes_[cache.stripe_index(line_address)]);
    simdojo::CacheTag *tag = cache.cache_.allocate(line_address, vmid);
    cache.cache_.write_line(line_address, data.data(), 0, MemorySideCache::LINE_SIZE, vmid);
    tag->cache_identity = cache_identity;
    tag->admission_generation = generation;
    tag->coherence = simdojo::CoherenceState::MODIFIED;
    cache.mark_line_dirty(*tag);
  }

  static void set_byte_valid(MemorySideCache &cache, uint64_t line_address, uint32_t vmid,
                             uint32_t offset, bool valid) {
    std::unique_lock access_lock(cache.access_gate_);
    std::lock_guard stripe_lock(cache.stripes_[cache.stripe_index(line_address)]);
    simdojo::CacheTag *tag = nullptr;
    ASSERT_TRUE(cache.cache_.lookup(line_address, &tag, vmid));
    const uint64_t bit = uint64_t{1} << (offset % 64);
    if (valid)
      tag->valid_bytes[offset / 64] |= bit;
    else
      tag->valid_bytes[offset / 64] &= ~bit;
  }

  static void store_dirty_line(MemorySideCache &cache, uint64_t line_address,
                               std::span<const uint8_t> data, uint32_t vmid) {
    assert(CacheStore::line_address(line_address) == line_address);
    assert(data.size() == MemorySideCache::LINE_SIZE);
    std::unique_lock access_lock(cache.access_gate_);
    std::lock_guard stripe_lock(cache.stripes_[cache.stripe_index(line_address)]);
    const std::optional<GpuVmAccess> access = vmid != 0 && cache.cache_vm() != nullptr
                                                  ? cache.cache_vm()->snapshot_vmid(vmid)
                                                  : std::nullopt;
    if (vmid != 0) {
      ASSERT_TRUE(access);
    }
    const VmAccessAdmission admission =
        access ? access->admit_access(line_address, CacheStore::LINE_SIZE, VmAccessKind::Write,
                                      /*report_fault=*/false)
               : cache.admit_access(line_address, CacheStore::LINE_SIZE, vmid, VmAccessKind::Write,
                                    /*report_fault=*/false);
    ASSERT_TRUE(admission);
    ASSERT_EQ(
        cache.ensure_line(line_address, vmid, admission.generation, access ? &*access : nullptr),
        VmAccessOutcome::Complete);
    cache.cache_.write_line(line_address, data.data(), 0, static_cast<uint32_t>(data.size()), vmid);
    simdojo::CacheTag *tag = nullptr;
    cache.cache_.lookup(line_address, &tag, vmid);
    assert(tag != nullptr);
    tag->coherence = simdojo::CoherenceState::MODIFIED;
    cache.mark_line_dirty(*tag);
  }

  static void mark_resident_dirty_after_barrier(MemorySideCache &cache, uint64_t line_address,
                                                std::span<const uint8_t> data,
                                                std::barrier<> &ready, std::barrier<> &release) {
    std::shared_lock access_lock(cache.access_gate_);
    std::lock_guard stripe_lock(cache.stripes_[cache.stripe_index(line_address)]);
    simdojo::CacheTag *tag = nullptr;
    const bool resident = cache.cache_.lookup(line_address, &tag, 0);
    ready.arrive_and_wait();
    release.arrive_and_wait();
    if (!resident)
      return;
    cache.cache_.write_line(line_address, data.data(), 0, static_cast<uint32_t>(data.size()), 0);
    tag->coherence = simdojo::CoherenceState::MODIFIED;
    cache.mark_line_dirty(*tag);
  }

  static std::pair<uint64_t, uint64_t> scan_counts(MemorySideCache &cache) {
    std::unique_lock access_lock(cache.access_gate_);
    return {cache.full_dirty_scans_, cache.skipped_clean_scans_};
  }

  static bool may_have_dirty_lines(MemorySideCache &cache) {
    std::unique_lock access_lock(cache.access_gate_);
    return cache.may_have_dirty_lines_.load(std::memory_order_relaxed);
  }

  static bool exclusive_maintenance_queued(MemorySideCache &cache) {
    if (!cache.access_gate_.try_lock_shared())
      return true;
    cache.access_gate_.unlock_shared();
    return false;
  }

  static bool line_is_dirty(MemorySideCache &cache, uint64_t line_address, uint32_t vmid) {
    std::unique_lock access_lock(cache.access_gate_);
    std::lock_guard stripe_lock(cache.stripes_[cache.stripe_index(line_address)]);
    const simdojo::CacheTag *tag = cache.cache_.peek(line_address, vmid);
    return tag != nullptr && tag->dirty;
  }

  static bool line_is_resident(MemorySideCache &cache, uint64_t line_address, uint32_t vmid) {
    std::unique_lock access_lock(cache.access_gate_);
    return cache.cache_.peek(line_address, vmid) != nullptr;
  }

  static uint64_t line_cache_identity(MemorySideCache &cache, uint64_t line_address,
                                      uint32_t vmid) {
    std::unique_lock access_lock(cache.access_gate_);
    const simdojo::CacheTag *tag = cache.cache_.peek(line_address, vmid);
    return tag != nullptr ? tag->cache_identity : 0;
  }

private:
  using CacheStore = MemorySideCache::CacheStore;
};

class DeviceCacheCoherenceTestAccess {
public:
  static std::vector<DeviceCacheMaintenanceLease>
  acquire_scoped_alias_maintenance_group(std::span<const DeviceCacheDomain> domains,
                                         std::span<const DeviceCacheAliasScope> aliases,
                                         const PreparedWritebackResolver &resolve_writeback,
                                         const CanonicalLineWitnessResolver &resolve_witness,
                                         std::chrono::milliseconds admission_retry_budget) {
    return DeviceCacheCoherence::acquire_scoped_alias_maintenance_group_with_budget(
        domains, aliases, resolve_writeback, resolve_witness, admission_retry_budget);
  }
};

} // namespace rocjitsu::amdgpu

namespace {

using rocjitsu::amdgpu::DeviceCacheCoherenceTestAccess;
using rocjitsu::amdgpu::GpuMemory;
using rocjitsu::amdgpu::HbmController;
using rocjitsu::amdgpu::L1ScalarCache;
using rocjitsu::amdgpu::L1VectorCache;
using rocjitsu::amdgpu::L2Cache;
using rocjitsu::amdgpu::MemorySideCache;
using rocjitsu::amdgpu::Mtype;
using rocjitsu::amdgpu::RequestMtypeResolver;

template <typename Attempt> void expect_terminal_maintenance_failure(Attempt &&attempt) {
  try {
    std::forward<Attempt>(attempt)();
    FAIL() << "expected terminal cache-maintenance failure";
  } catch (const rocjitsu::amdgpu::CacheMaintenanceUnavailable &error) {
    FAIL() << "unexpected retryable cache-maintenance refusal: " << error.what();
  } catch (const util::Exception &) {
  }
}

class IdentityAddressSpace final : public rocjitsu::amdgpu::AddressSpaceTranslator {
public:
  explicit IdentityAddressSpace(uint64_t size, Mtype mtype = Mtype::RW,
                                uint64_t admission_generation = 0, uint64_t translation_span = 0)
      : size_(size), mtype_(mtype), admission_generation_(admission_generation),
        translation_span_(translation_span) {}

  rocjitsu::amdgpu::VmAccessAdmission admit_access(uint64_t, std::size_t,
                                                   rocjitsu::amdgpu::VmAccessKind) const override {
    return {.generation = admission_generation_,
            .direct_backing_write_allowed = direct_backing_write_allowed};
  }

  rocjitsu::amdgpu::VmTranslationResult translate(uint64_t address, std::size_t size,
                                                  rocjitsu::amdgpu::VmAccessKind) const override {
    if (size == 0 || address > size_ || size > size_ - address)
      return {.outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted, .translation = {}};
    const uint64_t contiguous_bytes =
        translation_span_ == 0
            ? size_ - address
            : std::min(size_ - address, translation_span_ - address % translation_span_);
    return {
        .outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete,
        .translation = {.domain = rocjitsu::amdgpu::VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = contiguous_bytes,
                        .mtype = mtype_,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }

  bool direct_backing_write_allowed = true;

private:
  uint64_t size_;
  Mtype mtype_;
  uint64_t admission_generation_;
  uint64_t translation_span_;
};

class DeferredSwitchAddressSpace final : public rocjitsu::amdgpu::AddressSpaceTranslator {
public:
  explicit DeferredSwitchAddressSpace(uint64_t size) : size_(size) {}

  rocjitsu::amdgpu::VmDataRoute classify_data_route(uint64_t, std::size_t,
                                                    rocjitsu::amdgpu::VmAccessKind) const override {
    return route.load();
  }

  rocjitsu::amdgpu::VmAccessAdmission admit_access(uint64_t, std::size_t,
                                                   rocjitsu::amdgpu::VmAccessKind) const override {
    return route.load() != rocjitsu::amdgpu::VmDataRoute::Local
               ? rocjitsu::amdgpu::VmAccessAdmission{.outcome =
                                                         rocjitsu::amdgpu::VmAccessOutcome::Faulted}
               : rocjitsu::amdgpu::VmAccessAdmission{};
  }

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

  std::atomic<rocjitsu::amdgpu::VmDataRoute> route{rocjitsu::amdgpu::VmDataRoute::Local};

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
    std::ranges::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(address), bytes.size(),
                        bytes.begin());
    return rocjitsu::amdgpu::VmAccessOutcome::Complete;
  }

  rocjitsu::amdgpu::VmAccessOutcome write(rocjitsu::amdgpu::VmMemoryDomain domain, uint64_t address,
                                          std::span<const std::byte> bytes) override {
    ++write_calls;
    if (before_write)
      before_write();
    if (write_calls == fail_on_write_call)
      return fail_write_outcome;
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
  rocjitsu::amdgpu::VmAccessOutcome fail_write_outcome =
      rocjitsu::amdgpu::VmAccessOutcome::Unavailable;
  rocjitsu::amdgpu::VmAccessOutcome atomic_outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete;
  std::function<void()> before_write;
  uint32_t read_calls = 0;
  uint32_t write_calls = 0;
  uint32_t fail_on_write_call = 0;
  uint32_t atomic_calls = 0;

private:
  bool contains(uint64_t address, std::size_t size) const {
    return address <= bytes_.size() && size <= bytes_.size() - address;
  }

  std::vector<std::byte> bytes_;
};

class LinkedCacheHierarchy {
public:
  explicit LinkedCacheHierarchy(std::size_t size, uint64_t admission_generation = 0)
      : translator(std::make_shared<IdentityAddressSpace>(size, Mtype::RW, admission_generation)),
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

TEST(MemorySideCacheTest, CleanFullScanSkipsThenNewDirtyLinePublishes) {
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  constexpr uint64_t kAddress = 0x1000;
  LinkedCacheHierarchy hierarchy(0x2000);
  EXPECT_EQ(hierarchy.memory_side_cache.flush_all(), rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(MemorySideCacheTestAccess::scan_counts(hierarchy.memory_side_cache),
            (std::pair<uint64_t, uint64_t>{0, 1}));

  std::array<uint8_t, MemorySideCache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  MemorySideCacheTestAccess::store_dirty_line(hierarchy.memory_side_cache, kAddress, dirty,
                                              LinkedCacheHierarchy::kVmid);
  EXPECT_TRUE(MemorySideCacheTestAccess::may_have_dirty_lines(hierarchy.memory_side_cache));
  EXPECT_EQ(hierarchy.physical->load<uint8_t>(kAddress), 0u);

  ASSERT_EQ(hierarchy.memory_side_cache.flush_all(), rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(hierarchy.physical->load<uint8_t>(kAddress), dirty[0]);
  EXPECT_FALSE(MemorySideCacheTestAccess::may_have_dirty_lines(hierarchy.memory_side_cache));
  EXPECT_EQ(MemorySideCacheTestAccess::scan_counts(hierarchy.memory_side_cache),
            (std::pair<uint64_t, uint64_t>{1, 1}));
  EXPECT_EQ(hierarchy.memory_side_cache.flush_all(), rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(MemorySideCacheTestAccess::scan_counts(hierarchy.memory_side_cache),
            (std::pair<uint64_t, uint64_t>{1, 2}));
}

TEST(MemorySideCacheTest, FailedFullScanRetainsDirtyHintForRetry) {
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  constexpr uint64_t kAddress = 0x1000;
  LinkedCacheHierarchy hierarchy(0x2000);
  std::array<uint8_t, MemorySideCache::LINE_SIZE> dirty{};
  dirty.fill(0x6d);
  MemorySideCacheTestAccess::store_dirty_line(hierarchy.memory_side_cache, kAddress, dirty,
                                              LinkedCacheHierarchy::kVmid);

  hierarchy.physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Unavailable;
  EXPECT_EQ(hierarchy.memory_side_cache.flush_all(),
            rocjitsu::amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_TRUE(MemorySideCacheTestAccess::may_have_dirty_lines(hierarchy.memory_side_cache));
  EXPECT_TRUE(MemorySideCacheTestAccess::line_is_dirty(hierarchy.memory_side_cache, kAddress,
                                                       LinkedCacheHierarchy::kVmid));
  EXPECT_EQ(MemorySideCacheTestAccess::scan_counts(hierarchy.memory_side_cache),
            (std::pair<uint64_t, uint64_t>{1, 0}));
  EXPECT_EQ(hierarchy.physical->load<uint8_t>(kAddress), 0u);

  hierarchy.physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete;
  ASSERT_EQ(hierarchy.memory_side_cache.flush_all(), rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(hierarchy.physical->load<uint8_t>(kAddress), dirty[0]);
  EXPECT_FALSE(MemorySideCacheTestAccess::may_have_dirty_lines(hierarchy.memory_side_cache));
  EXPECT_EQ(MemorySideCacheTestAccess::scan_counts(hierarchy.memory_side_cache),
            (std::pair<uint64_t, uint64_t>{2, 0}));
  ASSERT_EQ(hierarchy.memory_side_cache.flush_all(), rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(MemorySideCacheTestAccess::scan_counts(hierarchy.memory_side_cache),
            (std::pair<uint64_t, uint64_t>{2, 1}));
}

TEST(MemorySideCacheTest, PartialRangeMaintenanceKeepsOtherVmidDirtyForFullScan) {
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  constexpr uint32_t kOtherVmid = LinkedCacheHierarchy::kVmid + 1;
  constexpr uint64_t kTargetAddress = 0x1000;
  constexpr uint64_t kOtherAddress = 0x1100;
  constexpr uint32_t kExternalOffset = 32;
  constexpr uint32_t kExternalValue = 0x12345678;
  LinkedCacheHierarchy hierarchy(0x2000);
  ASSERT_TRUE(
      hierarchy.gpu_vm.register_translated(kOtherVmid, hierarchy.translator, hierarchy.physical));
  std::array<uint8_t, MemorySideCache::LINE_SIZE> target_dirty{};
  std::array<uint8_t, MemorySideCache::LINE_SIZE> other_dirty{};
  target_dirty.fill(0x5a);
  other_dirty.fill(0x6d);
  MemorySideCacheTestAccess::store_dirty_line(hierarchy.memory_side_cache, kTargetAddress,
                                              target_dirty, LinkedCacheHierarchy::kVmid);
  MemorySideCacheTestAccess::store_dirty_line(hierarchy.memory_side_cache, kOtherAddress,
                                              other_dirty, kOtherVmid);

  ASSERT_EQ(hierarchy.memory_side_cache.flush_range(kTargetAddress + kExternalOffset,
                                                    sizeof(kExternalValue),
                                                    LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  hierarchy.physical->store(kTargetAddress + kExternalOffset, kExternalValue);
  hierarchy.memory_side_cache.invalidate_range_after_external_write(
      kTargetAddress + kExternalOffset, sizeof(kExternalValue), LinkedCacheHierarchy::kVmid);
  EXPECT_TRUE(MemorySideCacheTestAccess::may_have_dirty_lines(hierarchy.memory_side_cache));
  EXPECT_EQ(MemorySideCacheTestAccess::scan_counts(hierarchy.memory_side_cache),
            (std::pair<uint64_t, uint64_t>{0, 0}));
  EXPECT_EQ(hierarchy.physical->load<uint8_t>(kTargetAddress), target_dirty[0]);
  EXPECT_EQ(hierarchy.physical->load<uint32_t>(kTargetAddress + kExternalOffset), kExternalValue);
  EXPECT_EQ(hierarchy.physical->load<uint8_t>(kOtherAddress), 0u);

  ASSERT_EQ(hierarchy.memory_side_cache.flush_all(), rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(hierarchy.physical->load<uint8_t>(kOtherAddress), other_dirty[0]);
  EXPECT_EQ(MemorySideCacheTestAccess::scan_counts(hierarchy.memory_side_cache),
            (std::pair<uint64_t, uint64_t>{1, 0}));
}

TEST(MemorySideCacheTest, FullScanDropsRetiredBindingAndPublishesReplacementDirtyLine) {
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kNewBackingValue = 0x11223344;
  LinkedCacheHierarchy hierarchy(0x2000);
  std::array<uint8_t, MemorySideCache::LINE_SIZE> old_dirty{};
  std::array<uint8_t, MemorySideCache::LINE_SIZE> new_dirty{};
  old_dirty.fill(0x5a);
  new_dirty.fill(0x6d);
  MemorySideCacheTestAccess::store_dirty_line(hierarchy.memory_side_cache, kAddress, old_dirty,
                                              LinkedCacheHierarchy::kVmid);
  auto new_backing = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  new_backing->store(kAddress, kNewBackingValue);
  ASSERT_TRUE(hierarchy.gpu_vm.replace_translated(hierarchy.address_space, hierarchy.translator,
                                                  new_backing));

  ASSERT_EQ(hierarchy.memory_side_cache.flush_all(), rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(new_backing->load<uint32_t>(kAddress), kNewBackingValue);
  EXPECT_EQ(hierarchy.physical->write_calls, 0u);
  EXPECT_EQ(new_backing->write_calls, 0u);
  EXPECT_FALSE(MemorySideCacheTestAccess::may_have_dirty_lines(hierarchy.memory_side_cache));
  EXPECT_EQ(MemorySideCacheTestAccess::scan_counts(hierarchy.memory_side_cache),
            (std::pair<uint64_t, uint64_t>{1, 0}));

  MemorySideCacheTestAccess::store_dirty_line(hierarchy.memory_side_cache, kAddress, new_dirty,
                                              LinkedCacheHierarchy::kVmid);
  ASSERT_EQ(hierarchy.memory_side_cache.flush_all(), rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(new_backing->load<uint8_t>(kAddress), new_dirty[0]);
  EXPECT_EQ(MemorySideCacheTestAccess::scan_counts(hierarchy.memory_side_cache),
            (std::pair<uint64_t, uint64_t>{2, 0}));
}

TEST(MemorySideCacheTest, ConcurrentSharedAdmissionsKeepDirtyHintSetUntilFullScan) {
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  constexpr uint64_t kFirstAddress = 0x1000;
  constexpr uint64_t kSecondAddress = kFirstAddress + MemorySideCache::LINE_SIZE;
  GpuMemory memory("memory");
  HbmController hbm("hbm", &memory);
  MemorySideCache cache("memory_side_cache");
  simdojo::Link link(/*id=*/0, cache.req_port(), hbm.cpl_port(), /*latency=*/0);
  link.set_exec_mode(simdojo::ExecMode::FUNCTIONAL);
  cache.req_port()->set_link(&link);
  hbm.cpl_port()->set_link(&link);

  for (uint64_t address : {kFirstAddress, kSecondAddress}) {
    uint32_t observed = 0;
    ASSERT_EQ(cache.read(address, reinterpret_cast<uint8_t *>(&observed), sizeof(observed)),
              rocjitsu::amdgpu::VmAccessOutcome::Complete);
  }
  std::array<uint8_t, MemorySideCache::LINE_SIZE> first_dirty{};
  std::array<uint8_t, MemorySideCache::LINE_SIZE> second_dirty{};
  first_dirty.fill(0x5a);
  second_dirty.fill(0x6d);
  std::barrier ready(3);
  std::barrier release(3);
  auto first = std::async(std::launch::async, [&] {
    MemorySideCacheTestAccess::mark_resident_dirty_after_barrier(cache, kFirstAddress, first_dirty,
                                                                 ready, release);
  });
  auto second = std::async(std::launch::async, [&] {
    MemorySideCacheTestAccess::mark_resident_dirty_after_barrier(cache, kSecondAddress,
                                                                 second_dirty, ready, release);
  });
  ready.arrive_and_wait();
  auto flush = std::async(std::launch::async, [&] { return cache.flush_all(); });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  bool flush_queued = false;
  while (std::chrono::steady_clock::now() < deadline) {
    if (MemorySideCacheTestAccess::exclusive_maintenance_queued(cache)) {
      flush_queued = true;
      break;
    }
    std::this_thread::yield();
  }
  release.arrive_and_wait();
  first.get();
  second.get();
  EXPECT_TRUE(flush_queued);
  EXPECT_EQ(flush.get(), rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(memory.read8(kFirstAddress), first_dirty[0]);
  EXPECT_EQ(memory.read8(kSecondAddress), second_dirty[0]);
  EXPECT_EQ(MemorySideCacheTestAccess::scan_counts(cache), (std::pair<uint64_t, uint64_t>{1, 0}));
  EXPECT_FALSE(MemorySideCacheTestAccess::may_have_dirty_lines(cache));
}

TEST(MemorySideCacheTest, FlushRangeInvalidatesOnlyOverlappingLinesInRequestedVmid) {
  constexpr uint32_t kOtherVmid = LinkedCacheHierarchy::kVmid + 1;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kOtherAddress = kAddress + MemorySideCache::LINE_SIZE;
  constexpr uint32_t kInitial = 0x11223344;
  constexpr uint32_t kReplacement = 0xaabbccdd;
  LinkedCacheHierarchy hierarchy(0x2000);
  ASSERT_TRUE(
      hierarchy.gpu_vm.register_translated(kOtherVmid, hierarchy.translator, hierarchy.physical));
  hierarchy.physical->store(kAddress, kInitial);
  hierarchy.physical->store(kOtherAddress, kInitial);

  uint32_t observed = 0;
  for (const auto &[address, vmid] :
       {std::pair{kAddress, LinkedCacheHierarchy::kVmid},
        std::pair{kOtherAddress, LinkedCacheHierarchy::kVmid}, std::pair{kAddress, kOtherVmid}}) {
    ASSERT_EQ(hierarchy.memory_side_cache.read(address, reinterpret_cast<uint8_t *>(&observed),
                                               sizeof(observed), vmid),
              rocjitsu::amdgpu::VmAccessOutcome::Complete);
    ASSERT_EQ(observed, kInitial);
  }
  ASSERT_EQ(hierarchy.physical->read_calls, 3u);

  ASSERT_EQ(hierarchy.memory_side_cache.flush_range(kAddress, sizeof(kReplacement),
                                                    LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_FALSE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kAddress, LinkedCacheHierarchy::kVmid));
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kOtherAddress, LinkedCacheHierarchy::kVmid));
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kAddress, kOtherVmid));

  hierarchy.physical->store(kAddress, kReplacement);
  hierarchy.physical->store(kOtherAddress, kReplacement);

  ASSERT_EQ(hierarchy.memory_side_cache.read(kAddress, reinterpret_cast<uint8_t *>(&observed),
                                             sizeof(observed), LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kReplacement);
  ASSERT_EQ(hierarchy.memory_side_cache.read(kOtherAddress, reinterpret_cast<uint8_t *>(&observed),
                                             sizeof(observed), LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kInitial);
  ASSERT_EQ(hierarchy.memory_side_cache.read(kAddress, reinterpret_cast<uint8_t *>(&observed),
                                             sizeof(observed), kOtherVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kInitial);
  EXPECT_EQ(hierarchy.physical->read_calls, 4u);

  ASSERT_EQ(hierarchy.memory_side_cache.flush_range(kAddress, 0, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kAddress, LinkedCacheHierarchy::kVmid));

  // The large-range path visits resident tags rather than the entire span.
  constexpr uint32_t kThirdValue = 0x55667788;
  ASSERT_EQ(hierarchy.memory_side_cache.flush_range(0, std::numeric_limits<uint32_t>::max(),
                                                    LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_FALSE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kAddress, LinkedCacheHierarchy::kVmid));
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kAddress, kOtherVmid));
  hierarchy.physical->store(kAddress, kThirdValue);
  ASSERT_EQ(hierarchy.memory_side_cache.read(kAddress, reinterpret_cast<uint8_t *>(&observed),
                                             sizeof(observed), LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kThirdValue);
}

TEST(MemorySideCacheTest, PreWriteFlushRangePreservesDirtySiblingBytes) {
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kExternalOffset = 32;
  constexpr uint32_t kExternalValue = 0xaabbccdd;
  LinkedCacheHierarchy hierarchy(0x2000);
  std::array<uint8_t, MemorySideCache::LINE_SIZE> dirty{};
  std::array<uint8_t, MemorySideCache::LINE_SIZE> observed{};
  dirty.fill(0x5a);
  rocjitsu::amdgpu::MemorySideCacheTestAccess::store_dirty_line(
      hierarchy.memory_side_cache, kAddress, dirty, LinkedCacheHierarchy::kVmid);

  // The external writer has not committed yet. Publish the complete dirty
  // line first, then let the external write replace only its target bytes.
  ASSERT_EQ(hierarchy.memory_side_cache.flush_range(
                kAddress + kExternalOffset, sizeof(kExternalValue), LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_FALSE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kAddress, LinkedCacheHierarchy::kVmid));
  EXPECT_EQ(hierarchy.physical->write_calls, 1u);
  hierarchy.physical->store(kAddress + kExternalOffset, kExternalValue);
  hierarchy.memory_side_cache.invalidate_range_after_external_write(
      kAddress + kExternalOffset, sizeof(kExternalValue), LinkedCacheHierarchy::kVmid);
  EXPECT_EQ(hierarchy.physical->write_calls, 1u);

  ASSERT_EQ(hierarchy.memory_side_cache.read(kAddress, observed.data(), observed.size(),
                                             LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  std::memcpy(dirty.data() + kExternalOffset, &kExternalValue, sizeof(kExternalValue));
  EXPECT_EQ(observed, dirty);
}

TEST(MemorySideCacheTest, PostWriteInvalidationPreservesExternalBytesWithoutBackingWrite) {
  constexpr uint64_t kDirtyAddress = 0x1000;
  constexpr uint64_t kCleanAddress = kDirtyAddress + MemorySideCache::LINE_SIZE;
  constexpr uint64_t kUnrelatedAddress = kCleanAddress + MemorySideCache::LINE_SIZE;
  constexpr uint32_t kOtherVmid = LinkedCacheHierarchy::kVmid + 1;
  constexpr uint32_t kFirstExternalValue = 0xaabbccdd;
  constexpr uint32_t kSecondExternalValue = 0x12345678;
  constexpr uint32_t kInitial = 0x11223344;
  LinkedCacheHierarchy hierarchy(0x2000);
  ASSERT_TRUE(
      hierarchy.gpu_vm.register_translated(kOtherVmid, hierarchy.translator, hierarchy.physical));
  std::array<uint8_t, MemorySideCache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  rocjitsu::amdgpu::MemorySideCacheTestAccess::store_dirty_line(
      hierarchy.memory_side_cache, kDirtyAddress, dirty, LinkedCacheHierarchy::kVmid);
  hierarchy.physical->store(kCleanAddress, kInitial);
  hierarchy.physical->store(kUnrelatedAddress, kInitial);
  uint32_t observed = 0;
  for (const auto &[address, vmid] : {std::pair{kCleanAddress, LinkedCacheHierarchy::kVmid},
                                      std::pair{kUnrelatedAddress, LinkedCacheHierarchy::kVmid},
                                      std::pair{kDirtyAddress, kOtherVmid}}) {
    ASSERT_EQ(hierarchy.memory_side_cache.read(address, reinterpret_cast<uint8_t *>(&observed),
                                               sizeof(observed), vmid),
              rocjitsu::amdgpu::VmAccessOutcome::Complete);
  }

  const uint64_t external_address = kCleanAddress - sizeof(kFirstExternalValue);
  hierarchy.memory_side_cache.invalidate_range_after_external_write(external_address, 0,
                                                                    LinkedCacheHierarchy::kVmid);
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_dirty(
      hierarchy.memory_side_cache, kDirtyAddress, LinkedCacheHierarchy::kVmid));
  const uint32_t reads_before_invalidation = hierarchy.physical->read_calls;
  hierarchy.physical->store(external_address, kFirstExternalValue);
  hierarchy.physical->store(kCleanAddress, kSecondExternalValue);
  hierarchy.memory_side_cache.invalidate_range_after_external_write(
      external_address, 2 * sizeof(uint32_t), LinkedCacheHierarchy::kVmid);

  EXPECT_EQ(hierarchy.physical->load<uint32_t>(external_address), kFirstExternalValue);
  EXPECT_EQ(hierarchy.physical->load<uint32_t>(kCleanAddress), kSecondExternalValue);
  EXPECT_EQ(hierarchy.physical->read_calls, reads_before_invalidation);
  EXPECT_EQ(hierarchy.physical->write_calls, 0u);
  EXPECT_FALSE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kDirtyAddress, LinkedCacheHierarchy::kVmid));
  EXPECT_FALSE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kCleanAddress, LinkedCacheHierarchy::kVmid));
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kUnrelatedAddress, LinkedCacheHierarchy::kVmid));
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kDirtyAddress, kOtherVmid));

  ASSERT_EQ(hierarchy.memory_side_cache.read(external_address,
                                             reinterpret_cast<uint8_t *>(&observed),
                                             sizeof(observed), LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kFirstExternalValue);
  ASSERT_EQ(hierarchy.memory_side_cache.read(kCleanAddress, reinterpret_cast<uint8_t *>(&observed),
                                             sizeof(observed), LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kSecondExternalValue);

  // The large-range walk is bounded by cache storage and still selects one VMID.
  hierarchy.memory_side_cache.invalidate_range_after_external_write(
      0, std::numeric_limits<uint32_t>::max(), LinkedCacheHierarchy::kVmid);
  EXPECT_FALSE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kUnrelatedAddress, LinkedCacheHierarchy::kVmid));
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kDirtyAddress, kOtherVmid));
  EXPECT_EQ(hierarchy.physical->write_calls, 0u);
}

TEST(MemorySideCacheTest, PostWriteInvalidationClampsOverflowingRange) {
  GpuMemory memory("memory");
  HbmController hbm("hbm", &memory);
  MemorySideCache cache("memory_side_cache");
  simdojo::Link link(/*id=*/0, cache.req_port(), hbm.cpl_port(), /*latency=*/0);
  link.set_exec_mode(simdojo::ExecMode::FUNCTIONAL);
  cache.req_port()->set_link(&link);
  hbm.cpl_port()->set_link(&link);

  constexpr uint64_t kLastLine =
      std::numeric_limits<uint64_t>::max() - (MemorySideCache::LINE_SIZE - 1);
  constexpr uint64_t kHighAddress = kLastLine + MemorySideCache::LINE_SIZE / 2;
  constexpr uint32_t kReplacement = 0xaabbccdd;
  std::array<uint8_t, MemorySideCache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  rocjitsu::amdgpu::MemorySideCacheTestAccess::store_dirty_line(cache, kLastLine, dirty, 0);
  uint32_t observed = 0;
  ASSERT_EQ(cache.read(0, reinterpret_cast<uint8_t *>(&observed), sizeof(observed)),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);

  cache.invalidate_range_after_external_write(kHighAddress, 0, 0);
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_dirty(cache, kLastLine, 0));
  memory.write32(kHighAddress, kReplacement);
  cache.invalidate_range_after_external_write(kHighAddress, MemorySideCache::LINE_SIZE, 0);
  EXPECT_FALSE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(cache, kLastLine, 0));
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(cache, 0, 0));
  ASSERT_EQ(cache.read(kHighAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed)),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kReplacement);
}

TEST(MemorySideCacheTest, FailedRangeWritebackRetainsDirtyLineForRetry) {
  constexpr uint64_t kAddress = 0x1000;
  LinkedCacheHierarchy hierarchy(0x2000);
  std::array<uint8_t, MemorySideCache::LINE_SIZE> dirty{};
  dirty.fill(0x6d);
  rocjitsu::amdgpu::MemorySideCacheTestAccess::store_dirty_line(
      hierarchy.memory_side_cache, kAddress, dirty, LinkedCacheHierarchy::kVmid);

  hierarchy.physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Unavailable;
  EXPECT_EQ(hierarchy.memory_side_cache.flush_range(kAddress + 1, 1, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Unavailable);
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_dirty(
      hierarchy.memory_side_cache, kAddress, LinkedCacheHierarchy::kVmid));
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kAddress, LinkedCacheHierarchy::kVmid));
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::may_have_dirty_lines(
      hierarchy.memory_side_cache));
  EXPECT_EQ(hierarchy.physical->load<uint8_t>(kAddress), 0u);

  hierarchy.physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete;
  ASSERT_EQ(hierarchy.memory_side_cache.flush_range(kAddress + 1, 1, LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_FALSE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kAddress, LinkedCacheHierarchy::kVmid));
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::may_have_dirty_lines(
      hierarchy.memory_side_cache));
  EXPECT_EQ(hierarchy.physical->load<uint8_t>(kAddress), dirty[0]);
  EXPECT_EQ(hierarchy.physical->write_calls, 2u);
  ASSERT_EQ(hierarchy.memory_side_cache.flush_all(), rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_FALSE(rocjitsu::amdgpu::MemorySideCacheTestAccess::may_have_dirty_lines(
      hierarchy.memory_side_cache));
  EXPECT_EQ(rocjitsu::amdgpu::MemorySideCacheTestAccess::scan_counts(hierarchy.memory_side_cache),
            (std::pair<uint64_t, uint64_t>{1, 0}));
}

TEST(MemorySideCacheTest, RangeMaintenanceDropsDirtyLineAfterAdmissionRevocation) {
  constexpr uint64_t kAddress = 0x1000;
  LinkedCacheHierarchy hierarchy(0x2000);
  std::array<uint8_t, MemorySideCache::LINE_SIZE> dirty{};
  dirty.fill(0x6d);
  rocjitsu::amdgpu::MemorySideCacheTestAccess::store_dirty_line(
      hierarchy.memory_side_cache, kAddress, dirty, LinkedCacheHierarchy::kVmid);

  ASSERT_TRUE(hierarchy.gpu_vm.unregister_address_space(hierarchy.address_space));
  ASSERT_EQ(
      hierarchy.memory_side_cache.flush_range(kAddress, dirty.size(), LinkedCacheHierarchy::kVmid),
      rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_FALSE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kAddress, LinkedCacheHierarchy::kVmid));
  EXPECT_EQ(hierarchy.physical->write_calls, 0u);
  EXPECT_EQ(hierarchy.physical->load<uint8_t>(kAddress), 0u);
}

TEST(MemorySideCacheTest, RangeMaintenanceDoesNotPublishOldDirtyLineToReusedVmid) {
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kNewBackingValue = 0x11223344;
  LinkedCacheHierarchy hierarchy(0x2000);
  std::array<uint8_t, MemorySideCache::LINE_SIZE> dirty{};
  dirty.fill(0x6d);
  rocjitsu::amdgpu::MemorySideCacheTestAccess::store_dirty_line(
      hierarchy.memory_side_cache, kAddress, dirty, LinkedCacheHierarchy::kVmid);
  const auto old_access = hierarchy.gpu_vm.snapshot_vmid(LinkedCacheHierarchy::kVmid);
  ASSERT_TRUE(old_access);
  ASSERT_EQ(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_cache_identity(
                hierarchy.memory_side_cache, kAddress, LinkedCacheHierarchy::kVmid),
            old_access->cache_identity());
  const auto admitted =
      old_access->admit_access(kAddress, dirty.size(), rocjitsu::amdgpu::VmAccessKind::Write);
  ASSERT_TRUE(admitted);

  ASSERT_TRUE(hierarchy.gpu_vm.unregister_address_space(hierarchy.address_space));
  auto new_backing = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  new_backing->store(kAddress, kNewBackingValue);
  const auto new_handle = hierarchy.gpu_vm.register_translated(LinkedCacheHierarchy::kVmid,
                                                               hierarchy.translator, new_backing);
  ASSERT_TRUE(new_handle);
  const auto new_access = hierarchy.gpu_vm.snapshot_vmid(LinkedCacheHierarchy::kVmid);
  ASSERT_TRUE(new_access);
  ASSERT_NE(old_access->cache_namespace(), new_access->cache_namespace());
  ASSERT_NE(old_access->cache_identity(), new_access->cache_identity());
  EXPECT_EQ(old_access->write(kAddress, std::as_bytes(std::span<const uint8_t>(dirty))),
            rocjitsu::amdgpu::VmAccessOutcome::Revoked);

  ASSERT_EQ(
      hierarchy.memory_side_cache.flush_range(kAddress, dirty.size(), LinkedCacheHierarchy::kVmid),
      rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_FALSE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kAddress, LinkedCacheHierarchy::kVmid));
  EXPECT_EQ(hierarchy.physical->write_calls, 0u);
  EXPECT_EQ(new_backing->write_calls, 0u);
  EXPECT_EQ(new_backing->load<uint32_t>(kAddress), kNewBackingValue);

  uint32_t observed = 0;
  ASSERT_EQ(hierarchy.memory_side_cache.read(kAddress, reinterpret_cast<uint8_t *>(&observed),
                                             sizeof(observed), LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kNewBackingValue);
  EXPECT_EQ(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_cache_identity(
                hierarchy.memory_side_cache, kAddress, LinkedCacheHierarchy::kVmid),
            new_access->cache_identity());
}

TEST(MemorySideCacheTest, RangeMaintenanceDoesNotPublishOldDirtyLineAfterRootReplacement) {
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kNewBackingValue = 0x11223344;
  LinkedCacheHierarchy hierarchy(0x2000);
  std::array<uint8_t, MemorySideCache::LINE_SIZE> dirty{};
  dirty.fill(0x6d);
  rocjitsu::amdgpu::MemorySideCacheTestAccess::store_dirty_line(
      hierarchy.memory_side_cache, kAddress, dirty, LinkedCacheHierarchy::kVmid);
  const auto old_access = hierarchy.gpu_vm.snapshot_vmid(LinkedCacheHierarchy::kVmid);
  ASSERT_TRUE(old_access);

  auto new_backing = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  new_backing->store(kAddress, kNewBackingValue);
  ASSERT_TRUE(hierarchy.gpu_vm.replace_translated(hierarchy.address_space, hierarchy.translator,
                                                  new_backing));
  const auto new_access = hierarchy.gpu_vm.snapshot_vmid(LinkedCacheHierarchy::kVmid);
  ASSERT_TRUE(new_access);
  ASSERT_NE(old_access->cache_namespace(), new_access->cache_namespace());
  ASSERT_NE(old_access->cache_identity(), new_access->cache_identity());

  ASSERT_EQ(
      hierarchy.memory_side_cache.flush_range(kAddress, dirty.size(), LinkedCacheHierarchy::kVmid),
      rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_FALSE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(
      hierarchy.memory_side_cache, kAddress, LinkedCacheHierarchy::kVmid));
  EXPECT_EQ(new_backing->write_calls, 0u);
  EXPECT_EQ(new_backing->load<uint32_t>(kAddress), kNewBackingValue);
}

TEST(MemorySideCacheTest, RangeMaintenanceRetainsBindingDuringConcurrentRootReplacement) {
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kNewBackingValue = 0xaabbccdd;
  LinkedCacheHierarchy hierarchy(0x2000);
  auto new_backing = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  new_backing->store(kAddress, kNewBackingValue);
  std::array<uint8_t, MemorySideCache::LINE_SIZE> dirty{};
  dirty.fill(0x6d);
  MemorySideCacheTestAccess::store_dirty_line(hierarchy.memory_side_cache, kAddress, dirty,
                                              LinkedCacheHierarchy::kVmid);
  MemorySideCacheTestAccess::store_dirty_line(hierarchy.memory_side_cache,
                                              kAddress + MemorySideCache::LINE_SIZE, dirty,
                                              LinkedCacheHierarchy::kVmid);

  std::mutex mutex;
  std::condition_variable condition;
  bool write_entered = false;
  bool release_write = false;
  hierarchy.physical->before_write = [&] {
    std::unique_lock lock(mutex);
    write_entered = true;
    condition.notify_all();
    condition.wait(lock, [&] { return release_write; });
  };
  auto flush = std::async(std::launch::async, [&] {
    return hierarchy.memory_side_cache.flush_range(kAddress, 2 * MemorySideCache::LINE_SIZE,
                                                   LinkedCacheHierarchy::kVmid);
  });
  bool entered = false;
  {
    std::unique_lock lock(mutex);
    entered = condition.wait_for(lock, std::chrono::seconds(2), [&] { return write_entered; });
  }
  std::promise<void> replacement_started;
  std::future<void> replacement_running = replacement_started.get_future();
  auto replacement = std::async(std::launch::async, [&] {
    replacement_started.set_value();
    return hierarchy.gpu_vm.replace_translated(hierarchy.address_space, hierarchy.translator,
                                               new_backing);
  });
  replacement_running.wait();
  EXPECT_EQ(replacement.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
  {
    std::lock_guard lock(mutex);
    release_write = true;
  }
  condition.notify_all();
  EXPECT_TRUE(entered);
  EXPECT_EQ(flush.get(), VmAccessOutcome::Complete);
  ASSERT_TRUE(replacement.get());
  hierarchy.physical->before_write = {};
  EXPECT_EQ(new_backing->write_calls, 0u);
  EXPECT_EQ(new_backing->load<uint32_t>(kAddress), kNewBackingValue);
}

TEST(MemorySideCacheTest, OverflowingRangeStopsAtLastAddressableLine) {
  GpuMemory memory("memory");
  HbmController hbm("hbm", &memory);
  MemorySideCache cache("memory_side_cache");
  simdojo::Link link(/*id=*/0, cache.req_port(), hbm.cpl_port(), /*latency=*/0);
  link.set_exec_mode(simdojo::ExecMode::FUNCTIONAL);
  cache.req_port()->set_link(&link);
  hbm.cpl_port()->set_link(&link);

  constexpr uint64_t kLastLine =
      std::numeric_limits<uint64_t>::max() - (MemorySideCache::LINE_SIZE - 1);
  constexpr uint64_t kHighAddress = kLastLine + MemorySideCache::LINE_SIZE / 2;
  constexpr uint32_t kInitial = 0x11223344;
  constexpr uint32_t kReplacement = 0xaabbccdd;
  memory.write32(kHighAddress, kInitial);
  memory.write32(0, kInitial);
  uint32_t observed = 0;
  ASSERT_EQ(cache.read(kHighAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed)),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(cache.read(0, reinterpret_cast<uint8_t *>(&observed), sizeof(observed)),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(cache.flush_range(kHighAddress, MemorySideCache::LINE_SIZE, 0),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_FALSE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(cache, kLastLine, 0));
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_resident(cache, 0, 0));
  memory.write32(kHighAddress, kReplacement);
  memory.write32(0, kReplacement);
  ASSERT_EQ(cache.read(kHighAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed)),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kReplacement);
  ASSERT_EQ(cache.read(0, reinterpret_cast<uint8_t *>(&observed), sizeof(observed)),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kInitial);
}

TEST(MemorySideCacheTest, WithoutRangeMaintenanceExternalWriteLeavesStaleLine) {
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kInitial = 0x11223344;
  constexpr uint32_t kReplacement = 0xaabbccdd;
  LinkedCacheHierarchy hierarchy(0x2000);
  hierarchy.physical->store(kAddress, kInitial);
  uint32_t observed = 0;
  ASSERT_EQ(hierarchy.memory_side_cache.read(kAddress, reinterpret_cast<uint8_t *>(&observed),
                                             sizeof(observed), LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(observed, kInitial);

  hierarchy.physical->store(kAddress, kReplacement);
  ASSERT_EQ(hierarchy.memory_side_cache.read(kAddress, reinterpret_cast<uint8_t *>(&observed),
                                             sizeof(observed), LinkedCacheHierarchy::kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kInitial);
  EXPECT_EQ(hierarchy.physical->read_calls, 1u);
}

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

TEST(DeviceCacheCoherenceTest, WritebackMaintenancePublishesMemorySideBeforeL2DirtyBytes) {
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;

  auto coherence = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  L2Cache l2("l2", coherence);
  MemorySideCache memory_side_cache("memory_side_cache", coherence, &memory);
  l2.set_backing_memory(&memory);
  l2.set_legacy_maintenance_memory(&memory);
  l2.set_legacy_maintenance_vm(&gpu_vm);
  memory_side_cache.set_legacy_maintenance_vm(&gpu_vm);

  constexpr uint64_t kAddress = 0x9180;
  constexpr uint32_t kDirtyOffset = 32;
  constexpr uint32_t kDirtyBytes = 4;
  std::array<uint8_t, MemorySideCache::LINE_SIZE> memory_side_line{};
  memory_side_line.fill(0x11);
  MemorySideCacheTestAccess::store_prepared_dirty_line(memory_side_cache, kAddress,
                                                       memory_side_line, 0, 0, 0);
  std::array<uint8_t, L2Cache::LINE_SIZE> l2_line{};
  l2_line.fill(0x22);
  ASSERT_EQ(l2.writeback_line(kAddress, l2_line.data(), kDirtyOffset, kDirtyBytes, Mtype::RW),
            VmAccessOutcome::Complete);

  {
    [[maybe_unused]] auto maintenance = coherence->acquire_cache_maintenance(
        rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
  }
  EXPECT_EQ(memory.read8(kAddress + kDirtyOffset - 1), 0x11u);
  for (uint32_t byte = 0; byte < kDirtyBytes; ++byte)
    EXPECT_EQ(memory.read8(kAddress + kDirtyOffset + byte), 0x22u);
  EXPECT_EQ(memory.read8(kAddress + kDirtyOffset + kDirtyBytes), 0x11u);
}

TEST(DeviceCacheCoherenceTest, GroupMaintenanceRejectsOverlappingDirtyDomains) {
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;

  auto first_domain = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  auto second_domain = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  GpuMemory memory("shared_memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  L2Cache first_l2("first_l2", first_domain);
  MemorySideCache second_msc("second_msc", second_domain, &memory);
  first_l2.set_backing_memory(&memory);
  first_l2.set_legacy_maintenance_memory(&memory);
  first_l2.set_legacy_maintenance_vm(&gpu_vm);
  second_msc.set_legacy_maintenance_vm(&gpu_vm);

  constexpr uint64_t kAddress = 0x9200;
  constexpr uint32_t kDirtyOffset = 32;
  constexpr uint32_t kDirtyBytes = 4;
  std::array<uint8_t, MemorySideCache::LINE_SIZE> old_line{};
  old_line.fill(0x11);
  MemorySideCacheTestAccess::store_prepared_dirty_line(second_msc, kAddress, old_line, 0, 0, 0);
  std::array<uint8_t, L2Cache::LINE_SIZE> newer_line{};
  newer_line.fill(0x22);
  ASSERT_EQ(
      first_l2.writeback_line(kAddress, newer_line.data(), kDirtyOffset, kDirtyBytes, Mtype::RW),
      VmAccessOutcome::Complete);
  ASSERT_EQ(memory.read8(kAddress + kDirtyOffset), 0u);

  const uint64_t first_epoch = first_domain->current_epoch();
  const uint64_t second_epoch = second_domain->current_epoch();
  const std::array domains{first_domain, second_domain};
  EXPECT_THROW(
      {
        [[maybe_unused]] auto maintenance =
            rocjitsu::amdgpu::DeviceCacheCoherence::acquire_cache_maintenance_group(
                domains, rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
      },
      util::Exception);
  EXPECT_EQ(memory.read8(kAddress + kDirtyOffset), 0u);
  EXPECT_EQ(first_domain->current_epoch(), first_epoch);
  EXPECT_EQ(second_domain->current_epoch(), second_epoch);
  EXPECT_TRUE(rocjitsu::amdgpu::L2CacheTestAccess::dirty_byte(first_l2, kAddress, 0, kDirtyOffset));
}

TEST(DeviceCacheCoherenceTest, GroupMaintenanceWritesDisjointDirtyDomains) {
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;

  auto first_domain = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  auto second_domain = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  GpuMemory memory("shared_memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  MemorySideCache first_msc("first_msc", first_domain, &memory);
  MemorySideCache second_msc("second_msc", second_domain, &memory);
  first_msc.set_legacy_maintenance_vm(&gpu_vm);
  second_msc.set_legacy_maintenance_vm(&gpu_vm);

  constexpr uint64_t kFirstAddress = 0x9200;
  constexpr uint64_t kSecondAddress = 0x9300;
  std::array<uint8_t, MemorySideCache::LINE_SIZE> first_dirty{};
  std::array<uint8_t, MemorySideCache::LINE_SIZE> second_dirty{};
  first_dirty.fill(0x33);
  second_dirty.fill(0x44);
  MemorySideCacheTestAccess::store_prepared_dirty_line(first_msc, kFirstAddress, first_dirty, 0, 0,
                                                       0);
  MemorySideCacheTestAccess::store_prepared_dirty_line(second_msc, kSecondAddress, second_dirty, 0,
                                                       0, 0);

  const std::array domains{first_domain, second_domain};
  {
    [[maybe_unused]] auto maintenance =
        rocjitsu::amdgpu::DeviceCacheCoherence::acquire_cache_maintenance_group(
            domains, rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
  }
  EXPECT_EQ(memory.read8(kFirstAddress), 0x33u);
  EXPECT_EQ(memory.read8(kSecondAddress), 0x44u);
}

TEST(DeviceCacheCoherenceTest, GroupMaintenanceDistinguishesSeparateBackingMemory) {
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;

  auto first_domain = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  auto second_domain = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  GpuMemory first_memory("first_memory");
  GpuMemory second_memory("second_memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  MemorySideCache first_msc("first_msc", first_domain, &first_memory);
  MemorySideCache second_msc("second_msc", second_domain, &second_memory);
  first_msc.set_legacy_maintenance_vm(&gpu_vm);
  second_msc.set_legacy_maintenance_vm(&gpu_vm);

  constexpr uint64_t kAddress = 0x9400;
  std::array<uint8_t, MemorySideCache::LINE_SIZE> first_dirty{};
  std::array<uint8_t, MemorySideCache::LINE_SIZE> second_dirty{};
  first_dirty.fill(0x55);
  second_dirty.fill(0x66);
  MemorySideCacheTestAccess::store_prepared_dirty_line(first_msc, kAddress, first_dirty, 0, 0, 0);
  MemorySideCacheTestAccess::store_prepared_dirty_line(second_msc, kAddress, second_dirty, 0, 0, 0);

  const std::array domains{first_domain, second_domain};
  {
    [[maybe_unused]] auto maintenance =
        rocjitsu::amdgpu::DeviceCacheCoherence::acquire_cache_maintenance_group(
            domains, rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
  }
  EXPECT_EQ(first_memory.read8(kAddress), 0x55u);
  EXPECT_EQ(second_memory.read8(kAddress), 0x66u);
}

TEST(DeviceCacheCoherenceTest, GroupPreflightRejectsDirtyNonzeroVmidBeforeAnyWrite) {
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;

  auto first_domain = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  auto second_domain = std::make_shared<rocjitsu::amdgpu::DeviceCacheCoherence>();
  GpuMemory memory("shared_memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  MemorySideCache first_msc("first_msc", first_domain, &memory);
  L2Cache second_l2("second_l2", second_domain);
  first_msc.set_legacy_maintenance_vm(&gpu_vm);
  second_l2.set_legacy_maintenance_memory(&memory);
  second_l2.set_legacy_maintenance_vm(&gpu_vm);

  constexpr uint64_t kFirstAddress = 0xa000;
  constexpr uint64_t kSecondAddress = 0xb000;
  constexpr uint32_t kNonzeroVmid = 42;
  std::array<uint8_t, MemorySideCache::LINE_SIZE> first_dirty{};
  first_dirty.fill(0x77);
  MemorySideCacheTestAccess::store_prepared_dirty_line(first_msc, kFirstAddress, first_dirty, 0, 0,
                                                       0);
  std::array<uint8_t, L2Cache::LINE_SIZE> second_dirty{};
  second_dirty.fill(0x99);
  const std::array dirty_range{std::pair<uint32_t, uint32_t>{0, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(second_l2, kSecondAddress, second_dirty,
                                               kNonzeroVmid, 0, 0, dirty_range);

  const uint64_t first_epoch = first_domain->current_epoch();
  const uint64_t second_epoch = second_domain->current_epoch();
  const std::array domains{first_domain, second_domain};
  EXPECT_THROW(
      {
        [[maybe_unused]] auto maintenance =
            rocjitsu::amdgpu::DeviceCacheCoherence::acquire_cache_maintenance_group(
                domains, rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
      },
      util::Exception);
  EXPECT_EQ(memory.read8(kFirstAddress), 0u);
  EXPECT_EQ(first_domain->current_epoch(), first_epoch);
  EXPECT_EQ(second_domain->current_epoch(), second_epoch);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(second_l2, kSecondAddress, kNonzeroVmid, 0));

  const std::array retry_domains{first_domain};
  {
    [[maybe_unused]] auto maintenance =
        rocjitsu::amdgpu::DeviceCacheCoherence::acquire_cache_maintenance_group(
            retry_domains, rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
  }
  EXPECT_EQ(memory.read8(kFirstAddress), 0x77u);
}

TEST(DeviceCacheCoherenceTest, GroupPreparedSingleL2RunCommitsBeforeLeaseEpochs) {
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kGpuId = 11;
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kIdentity = 101;
  constexpr uint64_t kGeneration = 23;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  const std::array dirty_ranges{std::pair<uint32_t, uint32_t>{4, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, dirty, kVmid, kIdentity, kGeneration,
                                               dirty_ranges);
  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{coherence, kGpuId}};
  size_t resolver_calls = 0;
  const uint64_t data_epoch = coherence->current_epoch();
  const uint64_t instruction_epoch = coherence->current_instruction_epoch();
  auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
      domains, DeviceCacheOperation::WritebackInvalidate,
      [&](uint32_t gpu_id, uint32_t vmid, uint64_t line_address, size_t line_size) {
        ++resolver_calls;
        EXPECT_EQ(gpu_id, kGpuId);
        EXPECT_EQ(vmid, kVmid);
        EXPECT_EQ(line_address, kAddress);
        EXPECT_EQ(line_size, L2Cache::LINE_SIZE);
        return PreparedCanonicalWritebackTestAccess::make(
            kAddress, dirty.size(), kVmid, kIdentity, kGeneration,
            rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, dirty.size()));
      });
  ASSERT_EQ(leases.size(), 1u);
  EXPECT_EQ(resolver_calls, 1u);
  EXPECT_EQ(owner->write_attempts, 1u);
  EXPECT_EQ(owner->bytes_[3], std::byte{0});
  EXPECT_EQ(owner->bytes_[4], std::byte{0x5a});
  EXPECT_EQ(owner->bytes_[7], std::byte{0x5a});
  EXPECT_EQ(owner->bytes_[8], std::byte{0});
  EXPECT_EQ(coherence->current_epoch(), data_epoch);
  EXPECT_EQ(coherence->current_instruction_epoch(), instruction_epoch);
  leases.clear();
  EXPECT_EQ(coherence->current_epoch(), data_epoch + 1);
  EXPECT_EQ(coherence->current_instruction_epoch(), instruction_epoch + 1);
  EXPECT_FALSE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 4));
}

TEST(DeviceCacheCoherenceTest, GroupPreparedCompleteGrantWithWrongIdentityIsTerminal) {
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kIdentity = 101;
  constexpr uint64_t kGeneration = 23;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  const std::array dirty_ranges{std::pair<uint32_t, uint32_t>{8, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, dirty, kVmid, kIdentity, kGeneration,
                                               dirty_ranges);
  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const uint64_t epoch = coherence->current_epoch();
  size_t resolver_calls = 0;
  expect_terminal_maintenance_failure([&] {
    [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
        domains, DeviceCacheOperation::WritebackInvalidate,
        [&](uint32_t, uint32_t, uint64_t address, size_t line_size) {
          ++resolver_calls;
          return PreparedCanonicalWritebackTestAccess::make(
              address, line_size, kVmid, kIdentity + 1, kGeneration,
              rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, line_size));
        });
  });
  EXPECT_EQ(resolver_calls, 1u);
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_EQ(owner->bytes_[8], std::byte{0});
  EXPECT_EQ(coherence->current_epoch(), epoch);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 8));
}

TEST(DeviceCacheCoherenceTest, GroupPreparedRevokedGateRetainsDirtyL2Run) {
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;
  using rocjitsu::amdgpu::VmTransferEffectGate;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kIdentity = 101;
  constexpr uint64_t kGeneration = 23;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  const std::array dirty_ranges{std::pair<uint32_t, uint32_t>{8, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, dirty, kVmid, kIdentity, kGeneration,
                                               dirty_ranges);
  auto owner = std::make_shared<PreparedBackingOwner>();
  auto gate = std::make_shared<VmTransferEffectGate>();
  gate->revoke();
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const uint64_t data_epoch = coherence->current_epoch();
  const uint64_t instruction_epoch = coherence->current_instruction_epoch();
  expect_terminal_maintenance_failure([&] {
    [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
        domains, DeviceCacheOperation::WritebackInvalidate,
        [&](uint32_t, uint32_t, uint64_t, size_t) {
          return PreparedCanonicalWritebackTestAccess::make(
              kAddress, dirty.size(), kVmid, kIdentity, kGeneration,
              rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, dirty.size()),
              gate);
        });
  });
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_EQ(owner->bytes_[8], std::byte{0});
  EXPECT_EQ(coherence->current_epoch(), data_epoch);
  EXPECT_EQ(coherence->current_instruction_epoch(), instruction_epoch);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 8));
}

TEST(DeviceCacheCoherenceTest, GroupPreparedFailedBackingRetainsDirtyL2Run) {
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kIdentity = 101;
  constexpr uint64_t kGeneration = 23;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  const std::array dirty_ranges{std::pair<uint32_t, uint32_t>{8, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, dirty, kVmid, kIdentity, kGeneration,
                                               dirty_ranges);
  auto owner = std::make_shared<PreparedBackingOwner>();
  owner->fail_on_attempt = 1;
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const uint64_t data_epoch = coherence->current_epoch();
  const uint64_t instruction_epoch = coherence->current_instruction_epoch();
  const auto resolve = [&](uint32_t, uint32_t, uint64_t, size_t) {
    return PreparedCanonicalWritebackTestAccess::make(
        kAddress, dirty.size(), kVmid, kIdentity, kGeneration,
        rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, dirty.size()));
  };
  expect_terminal_maintenance_failure([&] {
    [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
        domains, DeviceCacheOperation::WritebackInvalidate, resolve);
  });
  EXPECT_EQ(owner->write_attempts, 1u);
  EXPECT_EQ(owner->bytes_[8], std::byte{0});
  EXPECT_EQ(coherence->current_epoch(), data_epoch);
  EXPECT_EQ(coherence->current_instruction_epoch(), instruction_epoch);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 8));

  owner->fail_on_attempt = 0;
  {
    [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
        domains, DeviceCacheOperation::WritebackInvalidate, resolve);
  }
  EXPECT_EQ(owner->write_attempts, 2u);
  EXPECT_EQ(owner->bytes_[8], std::byte{0x5a});
  EXPECT_FALSE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 8));
}

TEST(DeviceCacheCoherenceTest, GroupPreparedDirectViewCommitsTwoDisjointL2Runs) {
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  const std::array<std::pair<uint32_t, uint32_t>, 2> dirty_ranges{{{4, 4}, {112, 4}}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, dirty, kVmid, 101, 23, dirty_ranges);
  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  size_t resolver_calls = 0;
  const uint64_t epoch = coherence->current_epoch();
  auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
      domains, DeviceCacheOperation::WritebackInvalidate,
      [&](uint32_t gpu_id, uint32_t vmid, uint64_t address, size_t line_size) {
        ++resolver_calls;
        EXPECT_EQ(gpu_id, 11u);
        EXPECT_EQ(vmid, kVmid);
        EXPECT_EQ(address, kAddress);
        EXPECT_EQ(line_size, L2Cache::LINE_SIZE);
        return PreparedCanonicalWritebackTestAccess::make(
            kAddress, dirty.size(), kVmid, 101, 23,
            rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, dirty.size()),
            std::make_shared<rocjitsu::amdgpu::VmTransferEffectGate>(), [] {},
            std::span<std::byte>(owner->bytes_).first(dirty.size()));
      });
  ASSERT_EQ(leases.size(), 1u);
  EXPECT_EQ(resolver_calls, 1u);
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_EQ(owner->bytes_[3], std::byte{0});
  EXPECT_EQ(owner->bytes_[4], std::byte{0x5a});
  EXPECT_EQ(owner->bytes_[7], std::byte{0x5a});
  EXPECT_EQ(owner->bytes_[8], std::byte{0});
  EXPECT_EQ(owner->bytes_[111], std::byte{0});
  EXPECT_EQ(owner->bytes_[112], std::byte{0x5a});
  EXPECT_EQ(owner->bytes_[115], std::byte{0x5a});
  EXPECT_EQ(owner->bytes_[116], std::byte{0});
  EXPECT_EQ(coherence->current_epoch(), epoch);
  leases.clear();
  EXPECT_EQ(coherence->current_epoch(), epoch + 1);
  EXPECT_FALSE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 4));
  EXPECT_FALSE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 112));
}

TEST(DeviceCacheCoherenceTest, GroupPreparedFallibleOwnerRejectsTwoRunsBeforeAnyWrite) {
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  const std::array<std::pair<uint32_t, uint32_t>, 2> dirty_ranges{{{4, 4}, {112, 4}}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, dirty, kVmid, 101, 23, dirty_ranges);
  auto owner = std::make_shared<PreparedBackingOwner>();
  owner->fail_on_attempt = 2;
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  size_t resolver_calls = 0;
  const uint64_t data_epoch = coherence->current_epoch();
  const uint64_t instruction_epoch = coherence->current_instruction_epoch();
  EXPECT_THROW(
      {
        [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
            domains, DeviceCacheOperation::WritebackInvalidate,
            [&](uint32_t, uint32_t, uint64_t, size_t) {
              ++resolver_calls;
              return PreparedCanonicalWritebackTestAccess::make(
                  kAddress, dirty.size(), kVmid, 101, 23,
                  rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset,
                                          dirty.size()));
            });
      },
      util::Exception);
  EXPECT_EQ(resolver_calls, 1u);
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_EQ(owner->bytes_[4], std::byte{0});
  EXPECT_EQ(owner->bytes_[112], std::byte{0});
  EXPECT_EQ(coherence->current_epoch(), data_epoch);
  EXPECT_EQ(coherence->current_instruction_epoch(), instruction_epoch);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 4));
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 112));
}

TEST(DeviceCacheCoherenceTest, GroupPreparedTwoLinesAcrossDomainsRetainEveryPermit) {
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kFirstAddress = 0x1000;
  constexpr uint64_t kSecondAddress = kFirstAddress + L2Cache::LINE_SIZE;
  constexpr uint32_t kFirstOffset = 8;
  constexpr uint32_t kSecondOffset = 16;
  auto first_domain = std::make_shared<DeviceCacheCoherence>();
  auto second_domain = std::make_shared<DeviceCacheCoherence>();
  L2Cache first_l2("first_l2", first_domain);
  L2Cache second_l2("second_l2", second_domain);
  std::array<uint8_t, L2Cache::LINE_SIZE> first_dirty{};
  std::array<uint8_t, L2Cache::LINE_SIZE> second_dirty{};
  first_dirty.fill(0x5a);
  second_dirty.fill(0x6b);
  const std::array first_range{std::pair<uint32_t, uint32_t>{kFirstOffset, 4}};
  const std::array second_range{std::pair<uint32_t, uint32_t>{kSecondOffset, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(first_l2, kFirstAddress, first_dirty, kVmid, 101, 23,
                                               first_range);
  L2CacheTestAccess::store_prepared_dirty_line(second_l2, kSecondAddress, second_dirty, kVmid, 102,
                                               24, second_range);

  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{first_domain, 11},
                           DeviceCacheDomain{second_domain, 12}};
  const uint64_t first_epoch = first_domain->current_epoch();
  const uint64_t second_epoch = second_domain->current_epoch();
  size_t resolver_calls = 0;
  size_t permit_releases = 0;
  bool caches_unlocked_at_release = true;
  auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
      domains, DeviceCacheOperation::WritebackInvalidate,
      [&](uint32_t gpu_id, uint32_t vmid, uint64_t address, size_t line_size) {
        ++resolver_calls;
        EXPECT_EQ(vmid, kVmid);
        EXPECT_EQ(line_size, L2Cache::LINE_SIZE);
        const bool first = address == kFirstAddress;
        EXPECT_EQ(gpu_id, first ? 11u : 12u);
        const size_t backing_offset = first ? 0 : L2Cache::LINE_SIZE;
        return PreparedCanonicalWritebackTestAccess::make(
            address, line_size, kVmid, first ? 101 : 102, first ? 23 : 24,
            rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset + backing_offset,
                                    line_size),
            std::make_shared<rocjitsu::amdgpu::VmTransferEffectGate>(),
            [&] {
              ++permit_releases;
              caches_unlocked_at_release &= L2CacheTestAccess::cache_access_available(first_l2) &&
                                            L2CacheTestAccess::cache_access_available(second_l2);
            },
            std::span<std::byte>(owner->bytes_).subspan(backing_offset, line_size));
      });
  ASSERT_EQ(leases.size(), 2u);
  EXPECT_EQ(resolver_calls, 2u);
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_EQ(owner->bytes_[kFirstOffset], std::byte{0x5a});
  EXPECT_EQ(owner->bytes_[L2Cache::LINE_SIZE + kSecondOffset], std::byte{0x6b});
  EXPECT_EQ(first_domain->current_epoch(), first_epoch);
  EXPECT_EQ(second_domain->current_epoch(), second_epoch);
  EXPECT_EQ(permit_releases, 0u);
  leases.pop_back();
  EXPECT_EQ(permit_releases, 0u);
  leases.clear();
  EXPECT_EQ(permit_releases, 2u);
  EXPECT_TRUE(caches_unlocked_at_release);
  EXPECT_EQ(first_domain->current_epoch(), first_epoch + 1);
  EXPECT_EQ(second_domain->current_epoch(), second_epoch + 1);
  EXPECT_FALSE(L2CacheTestAccess::dirty_byte(first_l2, kFirstAddress, kVmid, kFirstOffset));
  EXPECT_FALSE(L2CacheTestAccess::dirty_byte(second_l2, kSecondAddress, kVmid, kSecondOffset));
}

TEST(DeviceCacheCoherenceTest, GroupPreparedSameLineMemorySidePrecedesL2Overlay) {
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kL2Offset = 32;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  MemorySideCache memory_side_cache("msc", coherence);
  L2Cache l2("l2", coherence);
  std::array<uint8_t, MemorySideCache::LINE_SIZE> memory_side_dirty{};
  std::array<uint8_t, L2Cache::LINE_SIZE> l2_dirty{};
  memory_side_dirty.fill(0x11);
  l2_dirty.fill(0x22);
  MemorySideCacheTestAccess::store_prepared_dirty_line(memory_side_cache, kAddress,
                                                       memory_side_dirty, kVmid, 101, 23);
  const std::array l2_range{std::pair<uint32_t, uint32_t>{kL2Offset, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, l2_dirty, kVmid, 101, 23, l2_range);
  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  size_t resolver_calls = 0;
  const uint64_t epoch = coherence->current_epoch();
  auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
      domains, DeviceCacheOperation::WritebackInvalidate,
      [&](uint32_t gpu_id, uint32_t vmid, uint64_t address, size_t line_size) {
        ++resolver_calls;
        EXPECT_EQ(gpu_id, 11u);
        EXPECT_EQ(vmid, kVmid);
        EXPECT_EQ(address, kAddress);
        EXPECT_EQ(line_size, L2Cache::LINE_SIZE);
        return PreparedCanonicalWritebackTestAccess::make(
            kAddress, line_size, kVmid, 101, 23,
            rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, line_size),
            std::make_shared<rocjitsu::amdgpu::VmTransferEffectGate>(), [] {},
            std::span<std::byte>(owner->bytes_).first(line_size));
      });
  ASSERT_EQ(leases.size(), 1u);
  EXPECT_EQ(resolver_calls, 2u);
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_EQ(owner->bytes_[kL2Offset - 1], std::byte{0x11});
  EXPECT_EQ(owner->bytes_[kL2Offset], std::byte{0x22});
  EXPECT_EQ(owner->bytes_[kL2Offset + 3], std::byte{0x22});
  EXPECT_EQ(owner->bytes_[kL2Offset + 4], std::byte{0x11});
  EXPECT_EQ(coherence->current_epoch(), epoch);
  leases.clear();
  EXPECT_EQ(coherence->current_epoch(), epoch + 1);
  EXPECT_FALSE(MemorySideCacheTestAccess::line_is_dirty(memory_side_cache, kAddress, kVmid));
  EXPECT_FALSE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, kL2Offset));
}

void check_group_prepared_rejects_distinct_va_canonical_overlap(bool separate_domains) {
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kFirstAddress = 0x1000;
  constexpr uint64_t kSecondAddress = 0x2000;
  constexpr uint32_t kDirtyOffset = 8;
  {
    auto first_domain = std::make_shared<DeviceCacheCoherence>();
    std::shared_ptr<DeviceCacheCoherence> second_domain =
        separate_domains ? std::make_shared<DeviceCacheCoherence>() : first_domain;
    L2Cache first_l2("first_l2", first_domain);
    L2Cache second_l2("second_l2", second_domain);
    std::array<uint8_t, L2Cache::LINE_SIZE> first_dirty{};
    std::array<uint8_t, L2Cache::LINE_SIZE> second_dirty{};
    first_dirty.fill(0x5a);
    second_dirty.fill(0x6b);
    const std::array dirty_range{std::pair<uint32_t, uint32_t>{kDirtyOffset, 4}};
    L2CacheTestAccess::store_prepared_dirty_line(first_l2, kFirstAddress, first_dirty, kVmid, 101,
                                                 23, dirty_range);
    L2CacheTestAccess::store_prepared_dirty_line(second_l2, kSecondAddress, second_dirty, kVmid,
                                                 102, 24, dirty_range);
    auto owner = std::make_shared<PreparedBackingOwner>();
    std::vector<DeviceCacheDomain> domains{{first_domain, 11}};
    if (separate_domains)
      domains.push_back({second_domain, 12});
    const uint64_t first_epoch = first_domain->current_epoch();
    const uint64_t second_epoch = second_domain->current_epoch();
    size_t resolver_calls = 0;
    expect_terminal_maintenance_failure([&] {
      [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
          domains, DeviceCacheOperation::WritebackInvalidate,
          [&](uint32_t, uint32_t vmid, uint64_t address, size_t line_size) {
            ++resolver_calls;
            const bool first = address == kFirstAddress;
            EXPECT_EQ(vmid, kVmid);
            return PreparedCanonicalWritebackTestAccess::make(
                address, line_size, kVmid, first ? 101 : 102, first ? 23 : 24,
                rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, line_size),
                std::make_shared<rocjitsu::amdgpu::VmTransferEffectGate>(), [] {},
                std::span<std::byte>(owner->bytes_).first(line_size));
          });
    });
    EXPECT_EQ(resolver_calls, 2u);
    EXPECT_EQ(owner->write_attempts, 0u);
    EXPECT_EQ(owner->bytes_[kDirtyOffset], std::byte{0});
    EXPECT_EQ(first_domain->current_epoch(), first_epoch);
    EXPECT_EQ(second_domain->current_epoch(), second_epoch);
    EXPECT_TRUE(L2CacheTestAccess::dirty_byte(first_l2, kFirstAddress, kVmid, kDirtyOffset));
    EXPECT_TRUE(L2CacheTestAccess::dirty_byte(second_l2, kSecondAddress, kVmid, kDirtyOffset));
  }
}

TEST(DeviceCacheCoherenceTest, GroupPreparedRejectsDistinctVaCanonicalOverlapInOneDomain) {
  check_group_prepared_rejects_distinct_va_canonical_overlap(false);
}

TEST(DeviceCacheCoherenceTest, GroupPreparedRejectsDistinctVaCanonicalOverlapAcrossDomains) {
  check_group_prepared_rejects_distinct_va_canonical_overlap(true);
}

TEST(DeviceCacheCoherenceTest, ScopedInvalidatesSelectedCleanLineWithoutEpoch) {
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::L2CacheTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  std::array<uint8_t, L2Cache::LINE_SIZE> clean_bytes{};
  clean_bytes.fill(0x5a);
  L2CacheTestAccess::store_prepared_clean_line(l2, kAddress, clean_bytes, kVmid, 101, 23);
  ASSERT_TRUE(L2CacheTestAccess::resident(l2, kAddress, kVmid));
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const std::array scopes{
      DeviceCacheAliasScope{domains[0], kVmid, kAddress, L2Cache::LINE_SIZE, 23, 101}};
  const uint64_t data_epoch = coherence->current_epoch();
  const uint64_t instruction_epoch = coherence->current_instruction_epoch();
  auto leases = DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
      domains, scopes,
      [&](uint32_t, uint32_t, uint64_t, size_t) -> rocjitsu::amdgpu::PreparedCanonicalWriteback {
        ADD_FAILURE() << "clean scoped maintenance must not resolve a dirty writer";
        throw util::Exception(std::string_view("unexpected dirty writer"));
      });
  leases.clear();
  EXPECT_FALSE(L2CacheTestAccess::resident(l2, kAddress, kVmid));
  EXPECT_EQ(coherence->current_epoch(), data_epoch);
  EXPECT_EQ(coherence->current_instruction_epoch(), instruction_epoch);
}

TEST(DeviceCacheCoherenceTest, ScopedGroupKeepsL1GateUntilLastDomainLeaseLeaves) {
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;

  auto first_domain = std::make_shared<DeviceCacheCoherence>();
  auto second_domain = std::make_shared<DeviceCacheCoherence>();
  L2Cache first_l2("first_l2", first_domain);
  L2Cache second_l2("second_l2", second_domain);
  L1ScalarCache first_l1(&first_l2);
  L1ScalarCache second_l1(&second_l2);
  const std::array domains{DeviceCacheDomain{first_domain, 11},
                           DeviceCacheDomain{second_domain, 12}};
  const std::array scopes{
      DeviceCacheAliasScope{domains[0], 73, 0x1000, L2Cache::LINE_SIZE, 23, 101},
      DeviceCacheAliasScope{domains[1], 73, 0x1000, L2Cache::LINE_SIZE, 24, 102}};
  auto leases = DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
      domains, scopes,
      [&](uint32_t, uint32_t, uint64_t, size_t) -> rocjitsu::amdgpu::PreparedCanonicalWriteback {
        ADD_FAILURE() << "clean scoped maintenance must not resolve a dirty writer";
        throw util::Exception(std::string_view("unexpected dirty writer"));
      });
  ASSERT_EQ(leases.size(), 2u);
  leases.pop_back();

  std::promise<void> worker_started;
  auto started = worker_started.get_future();
  auto pending = std::async(std::launch::async, [&] {
    worker_started.set_value();
    first_l1.invalidate_all();
  });
  started.wait();
  EXPECT_EQ(pending.wait_for(std::chrono::milliseconds(25)), std::future_status::timeout);
  leases.clear();
  EXPECT_EQ(pending.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  pending.get();
}

TEST(DeviceCacheCoherenceTest, ScopedGroupRetainsEveryDomainUntilLastGateUnlock) {
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;

  std::weak_ptr<DeviceCacheCoherence> first_weak;
  std::weak_ptr<DeviceCacheCoherence> second_weak;
  auto leases = [&] {
    auto first = std::make_shared<DeviceCacheCoherence>();
    auto second = std::make_shared<DeviceCacheCoherence>();
    first_weak = first;
    second_weak = second;
    const std::array domains{DeviceCacheDomain{first, 11}, DeviceCacheDomain{second, 12}};
    const std::array scopes{
        DeviceCacheAliasScope{domains[0], 73, 0x1000, L2Cache::LINE_SIZE, 23, 101},
        DeviceCacheAliasScope{domains[1], 73, 0x2000, L2Cache::LINE_SIZE, 24, 102}};
    return DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
        domains, scopes,
        [&](uint32_t, uint32_t, uint64_t, size_t) -> rocjitsu::amdgpu::PreparedCanonicalWriteback {
          ADD_FAILURE() << "empty domains must not resolve a dirty writer";
          throw util::Exception(std::string_view("unexpected dirty writer"));
        });
  }();
  ASSERT_EQ(leases.size(), 2u);
  leases.pop_back();
  EXPECT_FALSE(first_weak.expired());
  EXPECT_FALSE(second_weak.expired());
  leases.clear();
  EXPECT_TRUE(first_weak.expired());
  EXPECT_TRUE(second_weak.expired());
}

TEST(DeviceCacheCoherenceTest, ScopedMaintenanceLeavesUnrelatedDomainL1Available) {
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::L1ScalarCacheTestAccess;

  auto selected = std::make_shared<DeviceCacheCoherence>();
  auto unrelated = std::make_shared<DeviceCacheCoherence>();
  L2Cache selected_l2("selected_l2", selected);
  L2Cache unrelated_l2("unrelated_l2", unrelated);
  L1ScalarCache selected_l1(&selected_l2);
  L1ScalarCache unrelated_l1(&unrelated_l2);
  const std::array domains{DeviceCacheDomain{selected, 11}};
  const std::array scopes{
      DeviceCacheAliasScope{domains[0], 73, 0x1000, L2Cache::LINE_SIZE, 23, 101}};
  auto leases = DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
      domains, scopes,
      [&](uint32_t, uint32_t, uint64_t, size_t) -> rocjitsu::amdgpu::PreparedCanonicalWriteback {
        ADD_FAILURE() << "clean scoped maintenance must not resolve a dirty writer";
        throw util::Exception(std::string_view("unexpected dirty writer"));
      });
  EXPECT_FALSE(L1ScalarCacheTestAccess::cache_access_available(selected_l1));
  EXPECT_TRUE(L1ScalarCacheTestAccess::cache_access_available(unrelated_l1));
  EXPECT_EQ(selected->scoped_maintenance_success_count(), 1u);
  EXPECT_EQ(selected->scoped_admission_timeout_count(), 0u);
  EXPECT_EQ(unrelated->scoped_maintenance_success_count(), 0u);
}

TEST(DeviceCacheCoherenceTest, ScopedAdmissionIsBoundedWithOverlappingL1Readers) {
  using rocjitsu::amdgpu::CacheMaintenanceUnavailable;
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::L1ScalarCacheTestAccess;

  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  L1ScalarCache l1(&l2);
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const std::array scopes{
      DeviceCacheAliasScope{domains[0], 73, 0x1000, L2Cache::LINE_SIZE, 23, 101}};
  const auto resolve = [&](uint32_t, uint32_t, uint64_t,
                           size_t) -> rocjitsu::amdgpu::PreparedCanonicalWriteback {
    ADD_FAILURE() << "clean scoped maintenance must not resolve a dirty writer";
    throw util::Exception(std::string_view("unexpected dirty writer"));
  };
  auto first_reader = L1ScalarCacheTestAccess::hold_l1_access(*coherence);
  std::promise<void> second_ready;
  std::promise<void> release_second;
  auto second_release = release_second.get_future();
  auto second_reader = std::async(std::launch::async, [&] {
    auto held = L1ScalarCacheTestAccess::hold_l1_access(*coherence);
    second_ready.set_value();
    second_release.wait();
  });
  second_ready.get_future().wait();
  auto pending = std::async(std::launch::async, [&] {
    try {
      auto leases = DeviceCacheCoherenceTestAccess::acquire_scoped_alias_maintenance_group(
          domains, scopes, resolve, {}, std::chrono::milliseconds(10));
      return false;
    } catch (const CacheMaintenanceUnavailable &) {
      return true;
    }
  });
  const auto status = pending.wait_for(std::chrono::seconds(1));
  first_reader.unlock();
  release_second.set_value();
  second_reader.get();
  EXPECT_EQ(status, std::future_status::ready);
  EXPECT_TRUE(pending.get());
  EXPECT_EQ(coherence->scoped_maintenance_success_count(), 0u);
  EXPECT_EQ(coherence->scoped_admission_timeout_count(), 1u);
}

TEST(DeviceCacheCoherenceTest, ScopedWaitsForCoordinatorWithoutHoldingL1Gate) {
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L1ScalarCacheTestAccess;

  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  L1ScalarCache l1(&l2);
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const std::array scopes{
      DeviceCacheAliasScope{domains[0], 73, 0x1000, L2Cache::LINE_SIZE, 23, 101}};
  std::optional<rocjitsu::amdgpu::DeviceCacheMaintenanceLease> full;
  full.emplace(coherence->acquire_cache_maintenance(DeviceCacheOperation::Invalidate));

  std::promise<void> worker_started;
  auto started = worker_started.get_future();
  auto pending = std::async(std::launch::async, [&] {
    worker_started.set_value();
    auto leases = DeviceCacheCoherenceTestAccess::acquire_scoped_alias_maintenance_group(
        domains, scopes,
        [&](uint32_t, uint32_t, uint64_t, size_t) -> rocjitsu::amdgpu::PreparedCanonicalWriteback {
          ADD_FAILURE() << "clean scoped maintenance must not resolve a dirty writer";
          throw util::Exception(std::string_view("unexpected dirty writer"));
        },
        {}, std::chrono::seconds(2));
    leases.clear();
  });
  started.wait();
  EXPECT_EQ(pending.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
  // A peer reader can hold a page-table lease before reaching L1. If scoped
  // maintenance held the L1 gate while waiting here, that reader could not
  // release the lease needed by the full-domain owner.
  EXPECT_TRUE(L1ScalarCacheTestAccess::cache_access_available(l1));
  full.reset();
  EXPECT_EQ(pending.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  pending.get();
}

TEST(DeviceCacheCoherenceTest, ScopedReleasesCoordinatorWhileL1ReaderIsActive) {
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L1ScalarCacheTestAccess;

  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  L1ScalarCache l1(&l2);
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const std::array scopes{
      DeviceCacheAliasScope{domains[0], 73, 0x1000, L2Cache::LINE_SIZE, 23, 101}};
  auto active_reader = L1ScalarCacheTestAccess::hold_l1_access(*coherence);

  std::promise<void> worker_started;
  auto started = worker_started.get_future();
  auto scoped = std::async(std::launch::async, [&] {
    worker_started.set_value();
    auto leases = DeviceCacheCoherenceTestAccess::acquire_scoped_alias_maintenance_group(
        domains, scopes,
        [&](uint32_t, uint32_t, uint64_t, size_t) -> rocjitsu::amdgpu::PreparedCanonicalWriteback {
          ADD_FAILURE() << "clean scoped maintenance must not resolve a dirty writer";
          throw util::Exception(std::string_view("unexpected dirty writer"));
        },
        {}, std::chrono::seconds(2));
    leases.clear();
  });
  started.wait();
  EXPECT_EQ(scoped.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  auto full = std::async(std::launch::async, [&] {
    auto lease = coherence->acquire_cache_maintenance(DeviceCacheOperation::Invalidate);
  });
  const auto full_status = full.wait_for(std::chrono::seconds(2));
  // Release the reader even after a failed expectation so both workers can
  // complete and the test cannot strand an async destructor.
  active_reader.unlock();
  EXPECT_EQ(full_status, std::future_status::ready);
  full.get();
  EXPECT_EQ(scoped.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  scoped.get();
}

TEST(DeviceCacheCoherenceTest, ScopedReleasesL1GateWhileL2ReaderIsActive) {
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::L1ScalarCacheTestAccess;
  using rocjitsu::amdgpu::L2CacheTestAccess;

  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  L1ScalarCache l1(&l2);
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const std::array scopes{
      DeviceCacheAliasScope{domains[0], 73, 0x1000, L2Cache::LINE_SIZE, 23, 101}};
  auto active_reader = L2CacheTestAccess::hold_cache_access(l2);

  std::promise<void> worker_started;
  auto started = worker_started.get_future();
  auto pending = std::async(std::launch::async, [&] {
    worker_started.set_value();
    auto leases = DeviceCacheCoherenceTestAccess::acquire_scoped_alias_maintenance_group(
        domains, scopes,
        [&](uint32_t, uint32_t, uint64_t, size_t) -> rocjitsu::amdgpu::PreparedCanonicalWriteback {
          ADD_FAILURE() << "clean scoped maintenance must not resolve a dirty writer";
          throw util::Exception(std::string_view("unexpected dirty writer"));
        },
        {}, std::chrono::seconds(2));
    leases.clear();
  });
  started.wait();
  EXPECT_EQ(pending.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
  std::promise<void> l1_started;
  auto l1_start = l1_started.get_future();
  auto l1_access = std::async(std::launch::async, [&] {
    l1_started.set_value();
    l1.invalidate_all();
  });
  l1_start.wait();
  const auto l1_status = l1_access.wait_for(std::chrono::seconds(2));
  // Release the L2 reader even on a failed expectation so both workers can
  // finish without leaving a blocked async destructor.
  active_reader.unlock();
  EXPECT_EQ(l1_status, std::future_status::ready);
  l1_access.get();
  EXPECT_EQ(pending.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  pending.get();
}

TEST(DeviceCacheCoherenceTest, ScopedInvalidatesPrivateLinesAndAcknowledgesExternalEffect) {
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::L1ScalarCacheTestAccess;
  using rocjitsu::amdgpu::L1VectorCacheTestAccess;
  using rocjitsu::amdgpu::VmAccessKind;
  using rocjitsu::amdgpu::VmAccessOutcome;

  constexpr uint64_t kSelected = 0x1000;
  constexpr uint64_t kOther = 0x1100;
  constexpr uint32_t kOld = 0x11223344;
  constexpr uint32_t kNew = 0x55667788;
  LinkedCacheHierarchy hierarchy(0x2000, 23);
  L1ScalarCache scalar(&hierarchy.l2);
  L1VectorCache vector(&hierarchy.l2);
  scalar.set_gpu_vm(&hierarchy.gpu_vm);
  vector.set_gpu_vm(&hierarchy.gpu_vm);
  hierarchy.physical->store(kSelected, kOld);
  hierarchy.physical->store(kOther, kOld);
  const uint64_t selected_address[] = {kSelected};
  const uint64_t other_address[] = {kOther};
  uint32_t observed = 0;
  ASSERT_EQ(scalar.load(kSelected, 1, &observed, LinkedCacheHierarchy::kVmid),
            VmAccessOutcome::Complete);
  ASSERT_EQ(observed, kOld);
  ASSERT_EQ(scalar.load(kOther, 1, &observed, LinkedCacheHierarchy::kVmid),
            VmAccessOutcome::Complete);
  ASSERT_EQ(vector.load(selected_address, 1, sizeof(observed), 1,
                        reinterpret_cast<uint8_t *>(&observed), Mtype::RW, false, false, 1,
                        LinkedCacheHierarchy::kVmid),
            VmAccessOutcome::Complete);
  ASSERT_EQ(observed, kOld);
  ASSERT_EQ(vector.load(other_address, 1, sizeof(observed), 1,
                        reinterpret_cast<uint8_t *>(&observed), Mtype::RW, false, false, 1,
                        LinkedCacheHierarchy::kVmid),
            VmAccessOutcome::Complete);
  ASSERT_EQ(observed, kOld);

  const auto [identity, generation] = hierarchy.gpu_vm.with_vmid_snapshot(
      LinkedCacheHierarchy::kVmid, [&](const rocjitsu::amdgpu::GpuVmAccess *access) {
        EXPECT_NE(access, nullptr);
        const auto admission =
            access->admit_access(kSelected, L2Cache::LINE_SIZE, VmAccessKind::Read);
        EXPECT_EQ(admission.outcome, VmAccessOutcome::Complete);
        return std::pair{access->cache_identity(), admission.generation};
      });
  ASSERT_NE(identity, 0u);
  ASSERT_NE(generation, 0u);
  const std::array domains{DeviceCacheDomain{hierarchy.coherence, 11}};
  const std::array scopes{DeviceCacheAliasScope{domains[0], LinkedCacheHierarchy::kVmid, kSelected,
                                                L2Cache::LINE_SIZE, generation, identity}};
  const uint64_t epoch = hierarchy.coherence->current_epoch();
  auto leases = DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
      domains, scopes,
      [&](uint32_t, uint32_t, uint64_t, size_t) -> rocjitsu::amdgpu::PreparedCanonicalWriteback {
        ADD_FAILURE() << "clean scoped maintenance must not resolve a dirty writer";
        throw util::Exception(std::string_view("unexpected dirty writer"));
      });
  EXPECT_FALSE(std::async(std::launch::async, [&] {
                 return L1ScalarCacheTestAccess::cache_access_available(scalar);
               }).get());
  EXPECT_FALSE(std::async(std::launch::async, [&] {
                 return L1VectorCacheTestAccess::cache_access_available(vector);
               }).get());
  hierarchy.physical->store(kSelected, kNew);
  std::promise<void> worker_started;
  auto started = worker_started.get_future();
  auto pending = std::async(std::launch::async, [&] {
    worker_started.set_value();
    uint32_t value = 0;
    const auto outcome =
        vector.load(selected_address, 1, sizeof(value), 1, reinterpret_cast<uint8_t *>(&value),
                    Mtype::RW, false, false, 1, LinkedCacheHierarchy::kVmid);
    return std::pair{outcome, value};
  });
  started.wait();
  EXPECT_EQ(pending.wait_for(std::chrono::milliseconds(25)), std::future_status::timeout);
  leases.clear();
  const auto [outcome, value] = pending.get();
  EXPECT_TRUE(L1ScalarCacheTestAccess::cache_access_available(scalar));
  EXPECT_TRUE(L1VectorCacheTestAccess::cache_access_available(vector));
  EXPECT_EQ(outcome, VmAccessOutcome::Complete);
  EXPECT_EQ(value, kNew);
  EXPECT_EQ(hierarchy.coherence->current_epoch(), epoch);
  EXPECT_FALSE(L1ScalarCacheTestAccess::resident(scalar, kSelected, LinkedCacheHierarchy::kVmid));
  EXPECT_TRUE(L1ScalarCacheTestAccess::resident(scalar, kOther, LinkedCacheHierarchy::kVmid));
  EXPECT_TRUE(L1VectorCacheTestAccess::resident(vector, kOther, LinkedCacheHierarchy::kVmid));
  ASSERT_EQ(scalar.load(kSelected, 1, &observed, LinkedCacheHierarchy::kVmid),
            VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kNew);
}

TEST(DeviceCacheCoherenceTest, ScopedPrivateSelectionMatchesVmidAddressGrantAndIdentity) {
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::L1ScalarCacheTestAccess;
  using rocjitsu::amdgpu::L1VectorCacheTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kGeneration = 23;
  constexpr uint64_t kIdentity = 101;
  struct Resident {
    uint64_t address;
    uint32_t vmid;
    uint64_t generation;
    uint64_t identity;
  };
  const std::array residents{
      Resident{kAddress, kVmid, kGeneration, kIdentity},
      Resident{kAddress + L2Cache::LINE_SIZE, kVmid, kGeneration, kIdentity},
      Resident{kAddress, kVmid + 1, kGeneration, kIdentity},
      Resident{kAddress, kVmid, kGeneration + 1, kIdentity},
      Resident{kAddress, kVmid, kGeneration, kIdentity + 1},
  };
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  std::vector<std::unique_ptr<L1ScalarCache>> scalars;
  std::vector<std::unique_ptr<L1VectorCache>> vectors;
  for (const Resident &resident : residents) {
    scalars.push_back(std::make_unique<L1ScalarCache>(&l2));
    vectors.push_back(std::make_unique<L1VectorCache>(&l2));
    L1ScalarCacheTestAccess::store_clean_line(*scalars.back(), resident.address, resident.vmid,
                                              resident.generation, resident.identity);
    L1VectorCacheTestAccess::store_clean_line(*vectors.back(), resident.address, resident.vmid,
                                              resident.generation, resident.identity);
  }
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const std::array scopes{DeviceCacheAliasScope{domains[0], kVmid, kAddress, L2Cache::LINE_SIZE,
                                                kGeneration, kIdentity}};
  auto leases = DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
      domains, scopes,
      [&](uint32_t, uint32_t, uint64_t, size_t) -> rocjitsu::amdgpu::PreparedCanonicalWriteback {
        ADD_FAILURE() << "clean scoped maintenance must not resolve a dirty writer";
        throw util::Exception(std::string_view("unexpected dirty writer"));
      });
  leases.clear();
  for (size_t index = 0; index < residents.size(); ++index) {
    const Resident &resident = residents[index];
    EXPECT_EQ(L1ScalarCacheTestAccess::resident(*scalars[index], resident.address, resident.vmid),
              index != 0);
    EXPECT_EQ(L1VectorCacheTestAccess::resident(*vectors[index], resident.address, resident.vmid),
              index != 0);
  }
}

TEST(DeviceCacheCoherenceTest, ScopedRefusesUnidentifiedPrivateLineAndReleasesLocks) {
  using rocjitsu::amdgpu::CacheMaintenanceUnavailable;
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::L1ScalarCacheTestAccess;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;

  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kValue = 0x11223344;
  LinkedCacheHierarchy hierarchy(0x2000, 23);
  L1ScalarCache scalar(&hierarchy.l2);
  // This synthetic client has no VM service, so its nonzero-VMID L1 tag
  // cannot identify the backing generation selected by scoped maintenance.
  hierarchy.physical->store(kAddress, kValue);
  uint32_t observed = 0;
  ASSERT_EQ(scalar.load(kAddress, 1, &observed, LinkedCacheHierarchy::kVmid),
            VmAccessOutcome::Complete);
  ASSERT_EQ(observed, kValue);
  const uint64_t identity = hierarchy.gpu_vm.with_vmid_snapshot(
      LinkedCacheHierarchy::kVmid, [](const rocjitsu::amdgpu::GpuVmAccess *access) {
        return access != nullptr ? access->cache_identity() : uint64_t{0};
      });
  ASSERT_NE(identity, 0u);
  const std::array domains{DeviceCacheDomain{hierarchy.coherence, 11}};
  const std::array scopes{DeviceCacheAliasScope{domains[0], LinkedCacheHierarchy::kVmid, kAddress,
                                                L2Cache::LINE_SIZE, 23, identity}};
  EXPECT_THROW(
      {
        [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
            domains, scopes,
            [&](uint32_t, uint32_t, uint64_t,
                size_t) -> rocjitsu::amdgpu::PreparedCanonicalWriteback {
              ADD_FAILURE() << "clean scoped maintenance must not resolve a dirty writer";
              throw util::Exception(std::string_view("unexpected dirty writer"));
            });
      },
      CacheMaintenanceUnavailable);
  EXPECT_TRUE(std::async(std::launch::async, [&] {
                return L1ScalarCacheTestAccess::cache_access_available(scalar);
              }).get());
  EXPECT_TRUE(L1ScalarCacheTestAccess::resident(scalar, kAddress, LinkedCacheHierarchy::kVmid));
  EXPECT_TRUE(L2CacheTestAccess::resident(hierarchy.l2, kAddress, LinkedCacheHierarchy::kVmid));
}

TEST(DeviceCacheCoherenceTest, ScopedWritebackPreservesOtherDirtyMemorySideDomain) {
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kSelectedAddress = 0x1000;
  constexpr uint64_t kOtherAddress = 0x2000;
  auto selected_domain = std::make_shared<DeviceCacheCoherence>();
  auto other_domain = std::make_shared<DeviceCacheCoherence>();
  L2Cache selected_l2("selected_l2", selected_domain);
  MemorySideCache other_msc("other_msc", other_domain);
  std::array<uint8_t, L2Cache::LINE_SIZE> selected_bytes{};
  std::array<uint8_t, MemorySideCache::LINE_SIZE> other_bytes{};
  selected_bytes.fill(0x5a);
  other_bytes.fill(0x6b);
  const std::array selected_range{std::pair<uint32_t, uint32_t>{8, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(selected_l2, kSelectedAddress, selected_bytes, kVmid,
                                               101, 23, selected_range);
  MemorySideCacheTestAccess::store_prepared_dirty_line(other_msc, kOtherAddress, other_bytes, kVmid,
                                                       102, 24);
  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{selected_domain, 11},
                           DeviceCacheDomain{other_domain, 12}};
  const std::array scopes{
      DeviceCacheAliasScope{domains[0], kVmid, kSelectedAddress, L2Cache::LINE_SIZE, 23, 101}};
  const uint64_t selected_epoch = selected_domain->current_epoch();
  const uint64_t other_epoch = other_domain->current_epoch();
  size_t permit_releases = 0;
  auto leases = DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
      domains, scopes, [&](uint32_t gpu_id, uint32_t vmid, uint64_t address, size_t size) {
        EXPECT_EQ(vmid, kVmid);
        EXPECT_EQ(size, L2Cache::LINE_SIZE);
        const bool selected = gpu_id == 11;
        EXPECT_EQ(address, selected ? kSelectedAddress : kOtherAddress);
        const size_t offset = selected ? 0 : L2Cache::LINE_SIZE;
        return PreparedCanonicalWritebackTestAccess::make(
            address, size, kVmid, selected ? 101 : 102, selected ? 23 : 24,
            rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset + offset, size),
            std::make_shared<rocjitsu::amdgpu::VmTransferEffectGate>(), [&] { ++permit_releases; });
      });
  ASSERT_EQ(leases.size(), 2u);
  EXPECT_EQ(owner->write_attempts, 1u);
  EXPECT_EQ(owner->bytes_[8], std::byte{0x5a});
  EXPECT_EQ(owner->bytes_[L2Cache::LINE_SIZE], std::byte{0});
  EXPECT_EQ(permit_releases, 0u);
  leases.clear();
  EXPECT_EQ(permit_releases, 2u);
  EXPECT_EQ(selected_domain->current_epoch(), selected_epoch);
  EXPECT_EQ(other_domain->current_epoch(), other_epoch);
  EXPECT_FALSE(L2CacheTestAccess::dirty_byte(selected_l2, kSelectedAddress, kVmid, 8));
  EXPECT_TRUE(MemorySideCacheTestAccess::line_is_dirty(other_msc, kOtherAddress, kVmid));
  EXPECT_TRUE(MemorySideCacheTestAccess::line_is_resident(other_msc, kOtherAddress, kVmid));
}

TEST(DeviceCacheCoherenceTest, ScopedRefusesUnresolvedOrphanDirtyWitnessBeforeWrite) {
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kSelectedAddress = 0x1000;
  constexpr uint64_t kOrphanAddress = 0x2000;
  auto selected_domain = std::make_shared<DeviceCacheCoherence>();
  auto orphan_domain = std::make_shared<DeviceCacheCoherence>();
  L2Cache selected_l2("selected_l2", selected_domain);
  L2Cache orphan_l2("orphan_l2", orphan_domain);
  std::array<uint8_t, L2Cache::LINE_SIZE> selected_bytes{};
  std::array<uint8_t, L2Cache::LINE_SIZE> orphan_bytes{};
  selected_bytes.fill(0x5a);
  orphan_bytes.fill(0x6b);
  const std::array dirty_range{std::pair<uint32_t, uint32_t>{8, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(selected_l2, kSelectedAddress, selected_bytes, kVmid,
                                               101, 23, dirty_range);
  L2CacheTestAccess::store_prepared_dirty_line(orphan_l2, kOrphanAddress, orphan_bytes, kVmid, 102,
                                               24, dirty_range);
  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{selected_domain, 11},
                           DeviceCacheDomain{orphan_domain, 12}};
  const std::array scopes{
      DeviceCacheAliasScope{domains[0], kVmid, kSelectedAddress, L2Cache::LINE_SIZE, 23, 101}};
  const uint64_t selected_epoch = selected_domain->current_epoch();
  const uint64_t orphan_epoch = orphan_domain->current_epoch();
  bool orphan_witness_seen = false;
  EXPECT_THROW(
      {
        [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
            domains, scopes, [&](uint32_t gpu_id, uint32_t vmid, uint64_t address, size_t size) {
              EXPECT_EQ(vmid, kVmid);
              EXPECT_EQ(size, L2Cache::LINE_SIZE);
              if (gpu_id == 12 && address == kOrphanAddress) {
                orphan_witness_seen = true;
                throw util::Exception(std::string_view("orphan dirty writer has no live grant"));
              }
              EXPECT_EQ(gpu_id, 11u);
              EXPECT_EQ(address, kSelectedAddress);
              return PreparedCanonicalWritebackTestAccess::make(
                  address, size, kVmid, 101, 23,
                  rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, size),
                  std::make_shared<rocjitsu::amdgpu::VmTransferEffectGate>(), [] {},
                  std::span<std::byte>(owner->bytes_).first(size));
            });
      },
      util::Exception);
  EXPECT_TRUE(orphan_witness_seen);
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_EQ(owner->bytes_[8], std::byte{0});
  EXPECT_EQ(selected_domain->current_epoch(), selected_epoch);
  EXPECT_EQ(orphan_domain->current_epoch(), orphan_epoch);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(selected_l2, kSelectedAddress, kVmid, 8));
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(orphan_l2, kOrphanAddress, kVmid, 8));
}

TEST(DeviceCacheCoherenceTest, ScopedRejectsOverlappingUnselectedDirtyAlias) {
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kSelectedAddress = 0x1000;
  constexpr uint64_t kOtherAddress = 0x2000;
  auto selected_domain = std::make_shared<DeviceCacheCoherence>();
  auto other_domain = std::make_shared<DeviceCacheCoherence>();
  L2Cache selected_l2("selected_l2", selected_domain);
  L2Cache other_l2("other_l2", other_domain);
  std::array<uint8_t, L2Cache::LINE_SIZE> selected_bytes{};
  std::array<uint8_t, L2Cache::LINE_SIZE> other_bytes{};
  selected_bytes.fill(0x5a);
  other_bytes.fill(0x6b);
  const std::array selected_range{std::pair<uint32_t, uint32_t>{8, 4}};
  const std::array other_range{std::pair<uint32_t, uint32_t>{80, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(selected_l2, kSelectedAddress, selected_bytes, kVmid,
                                               101, 23, selected_range);
  L2CacheTestAccess::store_prepared_dirty_line(other_l2, kOtherAddress, other_bytes, kVmid, 102, 24,
                                               other_range);
  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{selected_domain, 11},
                           DeviceCacheDomain{other_domain, 12}};
  const std::array scopes{
      DeviceCacheAliasScope{domains[0], kVmid, kSelectedAddress, L2Cache::LINE_SIZE, 23, 101}};
  const uint64_t selected_epoch = selected_domain->current_epoch();
  const uint64_t other_epoch = other_domain->current_epoch();
  EXPECT_THROW(
      {
        [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
            domains, scopes, [&](uint32_t, uint32_t, uint64_t address, size_t size) {
              const bool selected = address == kSelectedAddress;
              return PreparedCanonicalWritebackTestAccess::make(
                  address, size, kVmid, selected ? 101 : 102, selected ? 23 : 24,
                  rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, size),
                  std::make_shared<rocjitsu::amdgpu::VmTransferEffectGate>(), [] {},
                  std::span<std::byte>(owner->bytes_).first(size));
            });
      },
      util::Exception);
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_EQ(owner->bytes_[8], std::byte{0});
  EXPECT_EQ(owner->bytes_[80], std::byte{0});
  EXPECT_EQ(selected_domain->current_epoch(), selected_epoch);
  EXPECT_EQ(other_domain->current_epoch(), other_epoch);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(selected_l2, kSelectedAddress, kVmid, 8));
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(other_l2, kOtherAddress, kVmid, 80));
}

TEST(DeviceCacheCoherenceTest, ScopedWitnessRefusesMissingStaleOrUnretainedDirtyLine) {
  using rocjitsu::amdgpu::CacheMaintenanceUnavailable;
  using rocjitsu::amdgpu::CanonicalLineWitness;
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kSelectedAddress = 0x1000;
  constexpr uint64_t kOtherAddress = 0x2000;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("witness_l2", coherence);
  std::array<uint8_t, L2Cache::LINE_SIZE> bytes{};
  bytes[8] = 0x5a;
  const std::array dirty_range{std::pair<uint32_t, uint32_t>{8, 1}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kOtherAddress, bytes, kVmid, 102, 24,
                                               dirty_range);
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const std::array scopes{
      DeviceCacheAliasScope{domains[0], kVmid, kSelectedAddress, L2Cache::LINE_SIZE, 23, 101,
                            PreparedBackingOwner::kBackingId, PreparedBackingOwner::kBaseOffset}};
  const uint64_t epoch = coherence->current_epoch();
  for (int invalid = 0; invalid < 3; ++invalid) {
    bool witness_called = false;
    EXPECT_THROW(
        {
          [[maybe_unused]] auto leases =
              DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
                  domains, scopes,
                  [](uint32_t, uint32_t, uint64_t,
                     size_t) -> rocjitsu::amdgpu::PreparedCanonicalWriteback {
                    ADD_FAILURE() << "an absent selected line must not resolve a writeback";
                    throw util::Exception(std::string_view("unexpected selected dirty line"));
                  },
                  [&](uint32_t gpu_id, uint32_t vmid, uint64_t address, size_t size,
                      uint64_t identity,
                      uint64_t generation) -> std::optional<CanonicalLineWitness> {
                    witness_called = true;
                    EXPECT_EQ(gpu_id, 11u);
                    EXPECT_EQ(vmid, kVmid);
                    EXPECT_EQ(address, kOtherAddress);
                    EXPECT_EQ(size, L2Cache::LINE_SIZE);
                    EXPECT_EQ(identity, 102u);
                    EXPECT_EQ(generation, 24u);
                    if (invalid == 0)
                      return std::nullopt;
                    CanonicalLineWitness witness{
                        .gpu_id = gpu_id,
                        .vmid = vmid,
                        .line_address = address,
                        .line_size = size,
                        .cache_identity = identity,
                        .admission_generation = generation,
                        .backing_id = PreparedBackingOwner::kBackingId,
                        .backing_offset = PreparedBackingOwner::kBaseOffset + L2Cache::LINE_SIZE,
                        .retention = std::make_shared<int>(1)};
                    if (invalid == 1)
                      ++witness.admission_generation;
                    else
                      witness.retention.reset();
                    return witness;
                  });
        },
        CacheMaintenanceUnavailable);
    EXPECT_TRUE(witness_called);
    EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kOtherAddress, kVmid, 8));
    EXPECT_EQ(coherence->current_epoch(), epoch);
  }
}

TEST(DeviceCacheCoherenceTest, ScopedWitnessOutlivesCacheLocks) {
  using rocjitsu::amdgpu::CanonicalLineWitness;
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kSelectedAddress = 0x1000;
  constexpr uint64_t kOtherAddress = 0x2000;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("witness_l2", coherence);
  std::array<uint8_t, L2Cache::LINE_SIZE> bytes{};
  bytes[8] = 0x5a;
  L2CacheTestAccess::store_prepared_clean_line(l2, kSelectedAddress, bytes, kVmid, 101, 23);
  const std::array dirty_range{std::pair<uint32_t, uint32_t>{8, 1}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kOtherAddress, bytes, kVmid, 102, 24,
                                               dirty_range);
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const std::array scopes{
      DeviceCacheAliasScope{domains[0], kVmid, kSelectedAddress, L2Cache::LINE_SIZE, 23, 101,
                            PreparedBackingOwner::kBackingId, PreparedBackingOwner::kBaseOffset}};
  bool released = false;
  bool locks_released_first = false;
  auto leases = DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
      domains, scopes,
      [](uint32_t, uint32_t, uint64_t, size_t) -> rocjitsu::amdgpu::PreparedCanonicalWriteback {
        ADD_FAILURE() << "a clean selected line must not resolve a writeback";
        throw util::Exception(std::string_view("unexpected selected dirty line"));
      },
      [&](uint32_t gpu_id, uint32_t vmid, uint64_t address, size_t size, uint64_t identity,
          uint64_t generation) -> std::optional<CanonicalLineWitness> {
        auto retention = std::shared_ptr<int>(new int(1), [&](int *value) {
          locks_released_first = L2CacheTestAccess::cache_access_available(l2);
          released = true;
          delete value;
        });
        return CanonicalLineWitness{.gpu_id = gpu_id,
                                    .vmid = vmid,
                                    .line_address = address,
                                    .line_size = size,
                                    .cache_identity = identity,
                                    .admission_generation = generation,
                                    .backing_id = PreparedBackingOwner::kBackingId,
                                    .backing_offset =
                                        PreparedBackingOwner::kBaseOffset + L2Cache::LINE_SIZE,
                                    .retention = std::move(retention)};
      });
  ASSERT_EQ(leases.size(), 1u);
  EXPECT_FALSE(released);
  leases.clear();
  EXPECT_TRUE(released);
  EXPECT_TRUE(locks_released_first);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kOtherAddress, kVmid, 8));
}

TEST(DeviceCacheCoherenceTest, ScopedWitnessChecksSelectedBackingAndScopeConflicts) {
  using rocjitsu::amdgpu::CanonicalLineWitness;
  using rocjitsu::amdgpu::DeviceCacheAliasScope;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kSelectedAddress = 0x1000;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("selected_l2", coherence);
  std::array<uint8_t, L2Cache::LINE_SIZE> bytes{};
  bytes[8] = 0x5a;
  const std::array dirty_range{std::pair<uint32_t, uint32_t>{8, 1}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kSelectedAddress, bytes, kVmid, 101, 23,
                                               dirty_range);
  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const uint64_t epoch = coherence->current_epoch();
  for (int mismatch = 0; mismatch < 2; ++mismatch) {
    const std::array scopes{DeviceCacheAliasScope{
        domains[0], kVmid, kSelectedAddress, L2Cache::LINE_SIZE, 23, 101,
        mismatch == 0 ? PreparedBackingOwner::kBackingId + 1 : PreparedBackingOwner::kBackingId,
        PreparedBackingOwner::kBaseOffset + (mismatch == 0 ? 0 : L2Cache::LINE_SIZE)}};
    EXPECT_THROW(
        {
          [[maybe_unused]] auto leases =
              DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
                  domains, scopes,
                  [&](uint32_t, uint32_t, uint64_t address, size_t size) {
                    return PreparedCanonicalWritebackTestAccess::make(
                        address, size, kVmid, 101, 23,
                        rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, size),
                        std::make_shared<rocjitsu::amdgpu::VmTransferEffectGate>(), [] {},
                        std::span<std::byte>(owner->bytes_).first(size));
                  },
                  [](uint32_t, uint32_t, uint64_t, size_t, uint64_t,
                     uint64_t) -> std::optional<CanonicalLineWitness> {
                    ADD_FAILURE() << "a selected dirty line must not need an unselected witness";
                    return std::nullopt;
                  });
        },
        util::Exception);
    EXPECT_EQ(owner->write_attempts, 0u);
    EXPECT_EQ(owner->bytes_[8], std::byte{0});
    EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kSelectedAddress, kVmid, 8));
    EXPECT_EQ(coherence->current_epoch(), epoch);
  }
  const std::array conflicting_scopes{
      DeviceCacheAliasScope{domains[0], kVmid, kSelectedAddress, L2Cache::LINE_SIZE, 23, 101,
                            PreparedBackingOwner::kBackingId, PreparedBackingOwner::kBaseOffset},
      DeviceCacheAliasScope{domains[0], kVmid, kSelectedAddress, L2Cache::LINE_SIZE, 23, 101,
                            PreparedBackingOwner::kBackingId + 1,
                            PreparedBackingOwner::kBaseOffset}};
  EXPECT_THROW(
      {
        [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
            domains, conflicting_scopes,
            [&](uint32_t, uint32_t, uint64_t address, size_t size) {
              return PreparedCanonicalWritebackTestAccess::make(
                  address, size, kVmid, 101, 23,
                  rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, size),
                  std::make_shared<rocjitsu::amdgpu::VmTransferEffectGate>(), [] {},
                  std::span<std::byte>(owner->bytes_).first(size));
            },
            [](uint32_t, uint32_t, uint64_t, size_t, uint64_t,
               uint64_t) -> std::optional<CanonicalLineWitness> { return std::nullopt; });
      },
      util::ConfigError);
  EXPECT_EQ(owner->bytes_[8], std::byte{0});
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kSelectedAddress, kVmid, 8));
  const std::array equivalent_scopes{conflicting_scopes[0], conflicting_scopes[0]};
  auto leases = DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
      domains, equivalent_scopes,
      [&](uint32_t, uint32_t, uint64_t address, size_t size) {
        return PreparedCanonicalWritebackTestAccess::make(
            address, size, kVmid, 101, 23,
            rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, size),
            std::make_shared<rocjitsu::amdgpu::VmTransferEffectGate>(), [] {},
            std::span<std::byte>(owner->bytes_).first(size));
      },
      [](uint32_t, uint32_t, uint64_t, size_t, uint64_t,
         uint64_t) -> std::optional<CanonicalLineWitness> { return std::nullopt; });
  ASSERT_EQ(leases.size(), 1u);
  leases.clear();
  EXPECT_EQ(owner->bytes_[8], std::byte{0x5a});
  EXPECT_FALSE(L2CacheTestAccess::dirty_byte(l2, kSelectedAddress, kVmid, 8));
}

TEST(DeviceCacheCoherenceTest, GroupPreparedRevokedSecondGateRetainsDirtyBatch) {
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;
  using rocjitsu::amdgpu::VmTransferEffectGate;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kFirstAddress = 0x1000;
  constexpr uint64_t kSecondAddress = kFirstAddress + L2Cache::LINE_SIZE;
  constexpr uint32_t kDirtyOffset = 8;
  auto first_domain = std::make_shared<DeviceCacheCoherence>();
  auto second_domain = std::make_shared<DeviceCacheCoherence>();
  L2Cache first_l2("first_l2", first_domain);
  L2Cache second_l2("second_l2", second_domain);
  std::array<uint8_t, L2Cache::LINE_SIZE> first_dirty{};
  std::array<uint8_t, L2Cache::LINE_SIZE> second_dirty{};
  first_dirty.fill(0x5a);
  second_dirty.fill(0x6b);
  const std::array dirty_range{std::pair<uint32_t, uint32_t>{kDirtyOffset, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(first_l2, kFirstAddress, first_dirty, kVmid, 101, 23,
                                               dirty_range);
  L2CacheTestAccess::store_prepared_dirty_line(second_l2, kSecondAddress, second_dirty, kVmid, 102,
                                               24, dirty_range);
  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{first_domain, 11},
                           DeviceCacheDomain{second_domain, 12}};
  const uint64_t first_data_epoch = first_domain->current_epoch();
  const uint64_t first_instruction_epoch = first_domain->current_instruction_epoch();
  const uint64_t second_data_epoch = second_domain->current_epoch();
  const uint64_t second_instruction_epoch = second_domain->current_instruction_epoch();
  auto first_gate = std::make_shared<VmTransferEffectGate>();
  auto second_gate = std::make_shared<VmTransferEffectGate>();
  // The coordinator sorts gates by this comparator. Revoke the gate acquired
  // second so the first permit has already been taken before the refusal.
  if (std::less<std::shared_ptr<VmTransferEffectGate>>{}(first_gate, second_gate))
    second_gate->revoke();
  else
    first_gate->revoke();
  size_t resolver_calls = 0;
  const auto resolve = [&](uint32_t, uint32_t vmid, uint64_t address, size_t line_size) {
    ++resolver_calls;
    EXPECT_EQ(vmid, kVmid);
    const bool first = address == kFirstAddress;
    const size_t backing_offset = first ? 0 : L2Cache::LINE_SIZE;
    return PreparedCanonicalWritebackTestAccess::make(
        address, line_size, kVmid, first ? 101 : 102, first ? 23 : 24,
        rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset + backing_offset,
                                line_size),
        first ? first_gate : second_gate, [] {},
        std::span<std::byte>(owner->bytes_).subspan(backing_offset, line_size));
  };
  expect_terminal_maintenance_failure([&] {
    [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
        domains, DeviceCacheOperation::WritebackInvalidate, resolve);
  });
  EXPECT_EQ(resolver_calls, 2u);
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_EQ(owner->bytes_[kDirtyOffset], std::byte{0});
  EXPECT_EQ(owner->bytes_[L2Cache::LINE_SIZE + kDirtyOffset], std::byte{0});
  EXPECT_EQ(first_domain->current_epoch(), first_data_epoch);
  EXPECT_EQ(first_domain->current_instruction_epoch(), first_instruction_epoch);
  EXPECT_EQ(second_domain->current_epoch(), second_data_epoch);
  EXPECT_EQ(second_domain->current_instruction_epoch(), second_instruction_epoch);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(first_l2, kFirstAddress, kVmid, kDirtyOffset));
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(second_l2, kSecondAddress, kVmid, kDirtyOffset));

  resolver_calls = 0;
  {
    [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
        domains, DeviceCacheOperation::WritebackInvalidate,
        [&](uint32_t, uint32_t vmid, uint64_t address, size_t line_size) {
          ++resolver_calls;
          EXPECT_EQ(vmid, kVmid);
          const bool first = address == kFirstAddress;
          const size_t backing_offset = first ? 0 : L2Cache::LINE_SIZE;
          return PreparedCanonicalWritebackTestAccess::make(
              address, line_size, kVmid, first ? 101 : 102, first ? 23 : 24,
              rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset + backing_offset,
                                      line_size),
              std::make_shared<VmTransferEffectGate>(), [] {},
              std::span<std::byte>(owner->bytes_).subspan(backing_offset, line_size));
        });
  }
  EXPECT_EQ(resolver_calls, 2u);
  EXPECT_EQ(owner->bytes_[kDirtyOffset], std::byte{0x5a});
  EXPECT_EQ(owner->bytes_[L2Cache::LINE_SIZE + kDirtyOffset], std::byte{0x6b});
  EXPECT_EQ(first_domain->current_epoch(), first_data_epoch + 1);
  EXPECT_EQ(first_domain->current_instruction_epoch(), first_instruction_epoch + 1);
  EXPECT_EQ(second_domain->current_epoch(), second_data_epoch + 1);
  EXPECT_EQ(second_domain->current_instruction_epoch(), second_instruction_epoch + 1);
  EXPECT_FALSE(L2CacheTestAccess::dirty_byte(first_l2, kFirstAddress, kVmid, kDirtyOffset));
  EXPECT_FALSE(L2CacheTestAccess::dirty_byte(second_l2, kSecondAddress, kVmid, kDirtyOffset));
}

TEST(DeviceCacheCoherenceTest, GroupPreparedRejectsMixedVmidZeroWriterBeforeAnyWrite) {
  using rocjitsu::amdgpu::CacheMaintenanceUnavailable;
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kZeroAddress = 0x2000;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  GpuMemory memory("memory");
  L2Cache l2("l2", coherence);
  MemorySideCache memory_side_cache("msc", coherence, &memory);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  const std::array dirty_ranges{std::pair<uint32_t, uint32_t>{8, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, dirty, kVmid, 101, 23, dirty_ranges);
  std::array<uint8_t, MemorySideCache::LINE_SIZE> zero_dirty{};
  zero_dirty.fill(0x6b);
  MemorySideCacheTestAccess::store_prepared_dirty_line(memory_side_cache, kZeroAddress, zero_dirty,
                                                       0, 0, 0);
  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  size_t resolver_calls = 0;
  const uint64_t epoch = coherence->current_epoch();
  EXPECT_THROW(
      {
        [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
            domains, DeviceCacheOperation::WritebackInvalidate,
            [&](uint32_t, uint32_t, uint64_t, size_t) {
              ++resolver_calls;
              return PreparedCanonicalWritebackTestAccess::make(
                  kAddress, dirty.size(), kVmid, 101, 23,
                  rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset,
                                          dirty.size()));
            });
      },
      CacheMaintenanceUnavailable);
  EXPECT_EQ(resolver_calls, 0u);
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_EQ(owner->bytes_[8], std::byte{0});
  EXPECT_EQ(memory.read8(kZeroAddress), 0u);
  EXPECT_EQ(coherence->current_epoch(), epoch);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 8));
  EXPECT_TRUE(MemorySideCacheTestAccess::line_is_dirty(memory_side_cache, kZeroAddress, 0));
}

TEST(DeviceCacheCoherenceTest, GroupPreparedRejectsReboundDirtyTagDuringResolver) {
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kOldIdentity = 101;
  constexpr uint64_t kNewIdentity = 102;
  constexpr uint64_t kGeneration = 23;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  std::array<uint8_t, L2Cache::LINE_SIZE> old_dirty{};
  old_dirty.fill(0x5a);
  const std::array dirty_ranges{std::pair<uint32_t, uint32_t>{8, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, old_dirty, kVmid, kOldIdentity,
                                               kGeneration, dirty_ranges);
  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const uint64_t epoch = coherence->current_epoch();
  size_t resolver_calls = 0;
  EXPECT_THROW(
      {
        [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
            domains, DeviceCacheOperation::WritebackInvalidate,
            [&](uint32_t, uint32_t, uint64_t, size_t) {
              ++resolver_calls;
              std::array<uint8_t, L2Cache::LINE_SIZE> new_dirty{};
              new_dirty.fill(0x6b);
              L2CacheTestAccess::rebind_prepared_dirty_line(l2, kAddress, new_dirty, kVmid,
                                                            kNewIdentity, kGeneration);
              return PreparedCanonicalWritebackTestAccess::make(
                  kAddress, old_dirty.size(), kVmid, kOldIdentity, kGeneration,
                  rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset,
                                          old_dirty.size()));
            });
      },
      util::Exception);
  EXPECT_EQ(resolver_calls, 1u);
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_EQ(owner->bytes_[8], std::byte{0});
  EXPECT_EQ(coherence->current_epoch(), epoch);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 8));

  {
    [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
        domains, DeviceCacheOperation::WritebackInvalidate,
        [&](uint32_t, uint32_t, uint64_t, size_t) {
          return PreparedCanonicalWritebackTestAccess::make(
              kAddress, old_dirty.size(), kVmid, kNewIdentity, kGeneration,
              rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset,
                                      old_dirty.size()));
        });
  }
  EXPECT_EQ(owner->write_attempts, 1u);
  EXPECT_EQ(owner->bytes_[8], std::byte{0x6b});
  EXPECT_FALSE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 8));
}

TEST(DeviceCacheCoherenceTest, GroupPreparedRejectsOtherLineByteChangeDuringBatchPreparation) {
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kFirstAddress = 0x1000;
  constexpr uint64_t kSecondAddress = kFirstAddress + L2Cache::LINE_SIZE;
  constexpr uint32_t kDirtyOffset = 8;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache first_l2("first_l2", coherence);
  L2Cache second_l2("second_l2", coherence);
  std::array<uint8_t, L2Cache::LINE_SIZE> first_dirty{};
  std::array<uint8_t, L2Cache::LINE_SIZE> second_dirty{};
  first_dirty.fill(0x5a);
  second_dirty.fill(0x6b);
  const std::array dirty_range{std::pair<uint32_t, uint32_t>{kDirtyOffset, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(first_l2, kFirstAddress, first_dirty, kVmid, 101, 23,
                                               dirty_range);
  L2CacheTestAccess::store_prepared_dirty_line(second_l2, kSecondAddress, second_dirty, kVmid, 102,
                                               24, dirty_range);
  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const uint64_t data_epoch = coherence->current_epoch();
  const uint64_t instruction_epoch = coherence->current_instruction_epoch();
  size_t resolver_calls = 0;
  EXPECT_THROW(
      {
        [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
            domains, DeviceCacheOperation::WritebackInvalidate,
            [&](uint32_t gpu_id, uint32_t vmid, uint64_t address, size_t line_size) {
              ++resolver_calls;
              EXPECT_EQ(gpu_id, 11u);
              EXPECT_EQ(vmid, kVmid);
              const bool first = address == kFirstAddress;
              if (resolver_calls == 1) {
                std::array<uint8_t, L2Cache::LINE_SIZE> replacement{};
                replacement.fill(0x7c);
                if (first)
                  L2CacheTestAccess::replace_prepared_dirty_line_bytes(second_l2, kSecondAddress,
                                                                       replacement, kVmid);
                else
                  L2CacheTestAccess::replace_prepared_dirty_line_bytes(first_l2, kFirstAddress,
                                                                       replacement, kVmid);
              }
              const size_t backing_offset = first ? 0 : L2Cache::LINE_SIZE;
              return PreparedCanonicalWritebackTestAccess::make(
                  address, line_size, kVmid, first ? 101 : 102, first ? 23 : 24,
                  rocjitsu::FabricBacking(
                      owner, 1, PreparedBackingOwner::kBaseOffset + backing_offset, line_size),
                  std::make_shared<rocjitsu::amdgpu::VmTransferEffectGate>(), [] {},
                  std::span<std::byte>(owner->bytes_).subspan(backing_offset, line_size));
            });
      },
      util::Exception);
  EXPECT_EQ(resolver_calls, 2u);
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_EQ(owner->bytes_[kDirtyOffset], std::byte{0});
  EXPECT_EQ(owner->bytes_[L2Cache::LINE_SIZE + kDirtyOffset], std::byte{0});
  EXPECT_EQ(coherence->current_epoch(), data_epoch);
  EXPECT_EQ(coherence->current_instruction_epoch(), instruction_epoch);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(first_l2, kFirstAddress, kVmid, kDirtyOffset));
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(second_l2, kSecondAddress, kVmid, kDirtyOffset));
}

TEST(DeviceCacheCoherenceTest, GroupPreparedVmRebindDuringResolverRevokesTwoRunBatch) {
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  rocjitsu::amdgpu::GpuVm gpu_vm;
  auto translator = std::make_shared<IdentityAddressSpace>(0x3000);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(0x3000);
  const auto old_handle = gpu_vm.register_translated(kVmid, translator, physical);
  ASSERT_TRUE(old_handle);
  const auto old_access = gpu_vm.snapshot_vmid(kVmid);
  ASSERT_TRUE(old_access);

  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  const std::array<std::pair<uint32_t, uint32_t>, 2> dirty_ranges{{{4, 4}, {112, 4}}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, dirty, kVmid,
                                               old_access->cache_identity(), 23, dirty_ranges);
  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const uint64_t data_epoch = coherence->current_epoch();
  const uint64_t instruction_epoch = coherence->current_instruction_epoch();
  size_t resolver_calls = 0;
  EXPECT_THROW(
      {
        [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
            domains, DeviceCacheOperation::WritebackInvalidate,
            [&](uint32_t, uint32_t, uint64_t address, size_t line_size) {
              ++resolver_calls;
              EXPECT_EQ(address, kAddress);
              EXPECT_TRUE(gpu_vm.unregister_address_space(old_handle));
              const auto new_handle = gpu_vm.register_translated(kVmid, translator, physical);
              EXPECT_TRUE(new_handle);
              const auto new_access = gpu_vm.snapshot_vmid(kVmid);
              EXPECT_TRUE(new_access);
              if (new_access) {
                EXPECT_NE(new_access->cache_identity(), old_access->cache_identity());
              }
              return PreparedCanonicalWritebackTestAccess::make(
                  kAddress, line_size, kVmid, old_access->cache_identity(), 23,
                  rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, line_size),
                  old_access->retain_effect_gate(), [] {},
                  std::span<std::byte>(owner->bytes_).first(line_size));
            });
      },
      util::Exception);
  EXPECT_EQ(resolver_calls, 1u);
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_EQ(owner->bytes_[4], std::byte{0});
  EXPECT_EQ(owner->bytes_[112], std::byte{0});
  EXPECT_EQ(coherence->current_epoch(), data_epoch);
  EXPECT_EQ(coherence->current_instruction_epoch(), instruction_epoch);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 4));
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 112));
}

TEST(DeviceCacheCoherenceTest, GroupPreparedRejectsEpochChangeDuringResolver) {
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kIdentity = 101;
  constexpr uint64_t kGeneration = 23;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  const std::array dirty_ranges{std::pair<uint32_t, uint32_t>{8, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, dirty, kVmid, kIdentity, kGeneration,
                                               dirty_ranges);
  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{coherence, 11}};
  const uint64_t data_epoch = coherence->current_epoch();
  const uint64_t instruction_epoch = coherence->current_instruction_epoch();
  size_t resolver_calls = 0;
  EXPECT_THROW(
      {
        [[maybe_unused]] auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
            domains, DeviceCacheOperation::WritebackInvalidate,
            [&](uint32_t, uint32_t, uint64_t, size_t) {
              ++resolver_calls;
              {
                [[maybe_unused]] auto invalidate =
                    coherence->acquire_cache_maintenance(DeviceCacheOperation::Invalidate);
              }
              EXPECT_EQ(coherence->current_epoch(), data_epoch + 1);
              EXPECT_TRUE(L2CacheTestAccess::dirty_byte_without_epoch_reconciliation(l2, kAddress,
                                                                                     kVmid, 8));
              return PreparedCanonicalWritebackTestAccess::make(
                  kAddress, dirty.size(), kVmid, kIdentity, kGeneration,
                  rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset,
                                          dirty.size()));
            });
      },
      util::Exception);
  EXPECT_EQ(resolver_calls, 1u);
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_EQ(owner->bytes_[8], std::byte{0});
  EXPECT_EQ(coherence->current_epoch(), data_epoch + 1);
  EXPECT_EQ(coherence->current_instruction_epoch(), instruction_epoch + 1);
  // A normal L2 access would reconcile the new epoch and discard this tag.
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte_without_epoch_reconciliation(l2, kAddress, kVmid, 8));
}

TEST(DeviceCacheCoherenceTest, GroupPreparedPermitOutlivesAllDomainLeases) {
  using rocjitsu::amdgpu::DeviceCacheCoherence;
  using rocjitsu::amdgpu::DeviceCacheDomain;
  using rocjitsu::amdgpu::DeviceCacheOperation;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  auto first = std::make_shared<DeviceCacheCoherence>();
  auto second = std::make_shared<DeviceCacheCoherence>();
  L2Cache first_l2("first_l2", first);
  L2Cache second_l2("second_l2", second);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  const std::array dirty_ranges{std::pair<uint32_t, uint32_t>{8, 4}};
  L2CacheTestAccess::store_prepared_dirty_line(first_l2, kAddress, dirty, kVmid, 101, 23,
                                               dirty_ranges);
  auto owner = std::make_shared<PreparedBackingOwner>();
  const std::array domains{DeviceCacheDomain{first, 11}, DeviceCacheDomain{second, 12}};
  size_t permit_releases = 0;
  bool cache_locks_released_on_permit = false;
  auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
      domains, DeviceCacheOperation::WritebackInvalidate,
      [&](uint32_t, uint32_t, uint64_t, size_t) {
        return PreparedCanonicalWritebackTestAccess::make(
            kAddress, dirty.size(), kVmid, 101, 23,
            rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, dirty.size()),
            std::make_shared<rocjitsu::amdgpu::VmTransferEffectGate>(), [&] {
              ++permit_releases;
              cache_locks_released_on_permit =
                  L2CacheTestAccess::cache_access_available(first_l2) &&
                  L2CacheTestAccess::cache_access_available(second_l2);
            });
      });
  ASSERT_EQ(leases.size(), 2u);
  EXPECT_EQ(permit_releases, 0u);
  leases.pop_back();
  EXPECT_EQ(permit_releases, 0u);
  leases.clear();
  EXPECT_EQ(permit_releases, 1u);
  EXPECT_TRUE(cache_locks_released_on_permit);
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
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::may_have_dirty_lines(memory_side_cache));
  EXPECT_EQ(rocjitsu::amdgpu::MemorySideCacheTestAccess::scan_counts(memory_side_cache),
            (std::pair<uint64_t, uint64_t>{1, 0}));

  ASSERT_EQ(mprotect(mapping.data(), GpuMemory::PAGE_SIZE, PROT_READ | PROT_WRITE), 0);
  {
    [[maybe_unused]] auto maintenance = coherence->acquire_cache_maintenance(
        rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
  }
  EXPECT_FALSE(rocjitsu::amdgpu::MemorySideCacheTestAccess::line_is_dirty(memory_side_cache,
                                                                          kGpuAddress, kVmid));
  EXPECT_TRUE(std::ranges::equal(dirty_line, std::span(mapping.data(), dirty_line.size())));
  EXPECT_FALSE(
      rocjitsu::amdgpu::MemorySideCacheTestAccess::may_have_dirty_lines(memory_side_cache));
  EXPECT_EQ(rocjitsu::amdgpu::MemorySideCacheTestAccess::scan_counts(memory_side_cache),
            (std::pair<uint64_t, uint64_t>{2, 0}));
  {
    [[maybe_unused]] auto maintenance = coherence->acquire_cache_maintenance(
        rocjitsu::amdgpu::DeviceCacheOperation::WritebackInvalidate);
  }
  EXPECT_EQ(rocjitsu::amdgpu::MemorySideCacheTestAccess::scan_counts(memory_side_cache),
            (std::pair<uint64_t, uint64_t>{2, 1}));
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
  memory_side_cache.set_gpu_vm(&gpu_vm);
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
  EXPECT_TRUE(rocjitsu::amdgpu::MemorySideCacheTestAccess::may_have_dirty_lines(memory_side_cache));

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
  EXPECT_EQ(rocjitsu::amdgpu::MemorySideCacheTestAccess::scan_counts(memory_side_cache),
            (std::pair<uint64_t, uint64_t>{1, 0}));
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

TEST(L2CacheTest, TransactionPolicySplitsRequestsWithoutChangingCacheGeometry) {
  using Policy = rocjitsu::amdgpu::MemoryTransactionPolicy;
  ASSERT_EQ(L1VectorCache::LINE_SIZE, 128u);
  ASSERT_EQ(L2Cache::LINE_SIZE, 128u);
  for (auto policy : {Policy{}, Policy::unqualified_64()}) {
    for (uint32_t elements : {3u, 4u}) {
      for (uint32_t lanes : {4u, 8u, 16u}) {
        for (uint32_t offset : {0u, 4u, 16u, 64u}) {
          for (bool reverse : {false, true}) {
            const uint32_t width = elements * sizeof(uint32_t);
            const uint32_t bytes = lanes * width;
            SCOPED_TRACE(std::format("granule={} bytes={} offset={} reverse={}", policy.bytes(),
                                     bytes, offset, reverse));
            const uint64_t base = 0x180000 + offset;
            GpuMemory memory("memory", policy);
            L2Cache l2("l2");
            l2.set_backing_memory(&memory);
            L1VectorCache l1(&l2, policy);
            std::array<uint64_t, 64> addresses{};
            std::array<uint32_t, 64 * 4> source{}, result{};
            for (uint32_t lane = 0; lane < lanes; ++lane) {
              addresses[lane] = base + (reverse ? lanes - 1 - lane : lane) * width;
              for (uint32_t word = 0; word < elements; ++word)
                source[elements * lane + word] = 100 * lane + word;
            }
            const uint64_t mask = (uint64_t{1} << lanes) - 1;
            const uint64_t expected =
                (offset % policy.bytes() + bytes + policy.bytes() - 1) / policy.bytes();
            ASSERT_EQ(l1.store(addresses.data(), mask, sizeof(uint32_t), elements,
                               reinterpret_cast<const uint8_t *>(source.data()), Mtype::UC, false,
                               64),
                      rocjitsu::amdgpu::VmAccessOutcome::Complete);
            EXPECT_EQ(l2.backing_write_transactions(), expected);
            ASSERT_EQ(l1.load(addresses.data(), mask, sizeof(uint32_t), elements,
                              reinterpret_cast<uint8_t *>(result.data()), Mtype::UC, false, true,
                              64),
                      rocjitsu::amdgpu::VmAccessOutcome::Complete);
            EXPECT_EQ(l2.backing_read_transactions(), expected);
            EXPECT_EQ(result, source);
          }
        }
      }
    }
  }
}

TEST(L2CacheTest, PermutedLanesPublishAndReadOneContiguousTransaction) {
  constexpr uint64_t kAddress = 0x180000;
  for (uint32_t bytes : {64u, 128u}) {
    for (uint32_t width : {4u, 8u, 16u}) {
      const uint32_t lanes = bytes / width;
      for (uint32_t spacing : {1u, 2u}) {
        for (uint32_t permutation : {0u, 1u, lanes - 1}) {
          SCOPED_TRACE(std::format("bytes={} width={} spacing={} permutation={}", bytes, width,
                                   spacing, permutation));
          GpuMemory memory("memory");
          L2Cache l2("l2");
          l2.set_backing_memory(&memory);
          L1VectorCache l1(&l2);
          std::array<uint64_t, 64> addresses{};
          std::array<uint8_t, 64 * 16> source{};
          std::array<uint8_t, 64 * 16> result{};
          std::array<uint8_t, L2Cache::LINE_SIZE> expected{};
          std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
          result.fill(0xcd);
          uint64_t mask = 0;
          for (uint32_t index = 0; index < lanes; ++index) {
            const uint32_t lane = index * spacing;
            const uint32_t offset = (index ^ permutation) * width;
            addresses[lane] = kAddress + offset;
            mask |= uint64_t{1} << lane;
            for (uint32_t byte = 0; byte < width; ++byte)
              expected[offset + byte] = source[lane * width + byte] = lane + byte;
          }
          ASSERT_EQ(l1.store(addresses.data(), mask, sizeof(uint32_t), width / sizeof(uint32_t),
                             source.data(), Mtype::UC, false, 64),
                    rocjitsu::amdgpu::VmAccessOutcome::Complete);
          EXPECT_EQ(l2.backing_write_transactions(), 1u);
          EXPECT_EQ(l2.backing_read_transactions(), 0u);
          memory.read_block(kAddress, std::span<uint8_t>(actual.data(), bytes));
          EXPECT_EQ(actual, expected);
          ASSERT_EQ(l1.load(addresses.data(), mask, sizeof(uint32_t), width / sizeof(uint32_t),
                            result.data(), Mtype::UC, false, true, 64),
                    rocjitsu::amdgpu::VmAccessOutcome::Complete);
          EXPECT_EQ(l2.backing_read_transactions(), 1u);
          for (uint32_t lane = 0; lane < 64; ++lane)
            for (uint32_t byte = 0; byte < width; ++byte)
              EXPECT_EQ(result[lane * width + byte],
                        mask & (uint64_t{1} << lane) ? source[lane * width + byte] : 0xcd);
        }
      }
    }
  }
}

TEST(L2CacheTest, ReorderedTransactionsKeepGapsClippingAndDuplicateLaneOrder) {
  constexpr uint64_t kAddress = 0x180000;
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  L1VectorCache l1(&l2);
  std::array<uint32_t, 8> initial;
  initial.fill(0xcccccccc);
  memory.write_block(
      kAddress,
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(initial.data()), sizeof(initial)));
  // Two complete b64 lanes are permuted; only the first dword of lane two is valid.
  const uint64_t addresses[] = {kAddress + 8, kAddress, kAddress + 24};
  const uint64_t elements[] = {0x7, 0x3};
  const std::array<uint32_t, 6> values = {10, 11, 20, 21, 30, 31};
  ASSERT_EQ(l1.store(addresses, 0x7, sizeof(uint32_t), 2,
                     reinterpret_cast<const uint8_t *>(values.data()), Mtype::UC, false, 3, 0, 0, 0,
                     elements),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  std::array<uint32_t, 8> actual{};
  memory.read_block(kAddress,
                    std::span<uint8_t>(reinterpret_cast<uint8_t *>(actual.data()), sizeof(actual)));
  EXPECT_EQ(actual,
            (std::array<uint32_t, 8>{20, 21, 10, 11, 0xcccccccc, 0xcccccccc, 30, 0xcccccccc}));
  std::array<uint32_t, 6> loaded;
  loaded.fill(0xdddddddd);
  ASSERT_EQ(l1.load(addresses, 0x7, sizeof(uint32_t), 2, reinterpret_cast<uint8_t *>(loaded.data()),
                    Mtype::UC, false, true, 3, 0, 0, 0, elements),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(loaded, (std::array<uint32_t, 6>{10, 11, 20, 21, 30, 0xdddddddd}));

  const uint64_t aliases[] = {kAddress, kAddress, kAddress + 4};
  const std::array<uint32_t, 3> overlapping = {1, 2, 3};
  ASSERT_EQ(l1.store(aliases, 0x7, sizeof(uint32_t), 1,
                     reinterpret_cast<const uint8_t *>(overlapping.data()), Mtype::UC, false, 3),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(memory.read32(kAddress), 2u);
  EXPECT_EQ(memory.read32(kAddress + 4), 3u);
}

TEST(L2CacheTest, ReorderedTransactionStopsAtCacheLineBoundary) {
  constexpr uint64_t kAddress = 0x180040;
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  L1VectorCache l1(&l2);
  std::array<uint64_t, 32> addresses{};
  std::array<uint32_t, 32> source{};
  std::array<uint32_t, 32> result{};
  for (uint32_t lane = 0; lane < 32; ++lane) {
    addresses[lane] = kAddress + (31 - lane) * sizeof(uint32_t);
    source[lane] = lane;
  }
  ASSERT_EQ(l1.store(addresses.data(), 0xffffffff, sizeof(uint32_t), 1,
                     reinterpret_cast<const uint8_t *>(source.data()), Mtype::UC, false, 32),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(l2.backing_write_transactions(), 2u);
  ASSERT_EQ(l1.load(addresses.data(), 0xffffffff, sizeof(uint32_t), 1,
                    reinterpret_cast<uint8_t *>(result.data()), Mtype::UC, false, true, 32),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(l2.backing_read_transactions(), 2u);
  EXPECT_EQ(result, source);
}

TEST(L2CacheTest, SeededLaneTransfersMatchIndependentByteAndTransactionOracles) {
  using Policy = rocjitsu::amdgpu::MemoryTransactionPolicy;
  constexpr uint64_t kBase = 0x180000;
  constexpr size_t kBytes = 2048;
  constexpr uint32_t kSeed = 0x5e9d3147;
  std::mt19937 random(kSeed);
  for (auto policy : {Policy{}, Policy::unqualified_64()}) {
    for (uint32_t wave_size : {32u, 64u}) {
      for (uint32_t width : {1u, 2u, 4u, 8u, 12u, 16u}) {
        for (uint32_t trial = 0; trial < 32; ++trial) {
          SCOPED_TRACE(std::format("seed={} policy={} wave={} width={} trial={}", kSeed,
                                   policy.bytes(), wave_size, width, trial));
          const uint32_t offset = random() % 128;
          std::array<uint32_t, 64> offsets{}, order{};
          uint32_t cursor = offset;
          for (uint32_t lane = 0; lane < wave_size; ++lane) {
            offsets[lane] = cursor;
            order[lane] = lane;
            cursor += width + (trial % 4 == 0 ? 0 : random() % 8);
          }
          std::ranges::shuffle(std::span(order).first(wave_size), random);
          uint64_t mask = (uint64_t{random()} << 32) | random();
          if (wave_size == 32)
            mask &= 0xffffffff;
          if (trial % 8 == 0)
            mask = wave_size == 64 ? ~uint64_t{0} : uint64_t{0xffffffff};
          if (trial == 31)
            mask = 0;
          std::array<uint64_t, 64> addresses{};
          std::array<uint8_t, 64 * 16> source{}, loaded{}, expected_load{};
          std::array<uint8_t, kBytes> expected_store{}, actual{}, read_source{};
          std::array<bool, kBytes> requested{};
          expected_store.fill(0xa5);
          loaded.fill(0xcd);
          expected_load.fill(0xcd);
          for (size_t byte = 0; byte < read_source.size(); ++byte)
            read_source[byte] = static_cast<uint8_t>(random());
          // The reference visits individual requested bytes, independently of
          // lane grouping, gathering, scattering, and cache-line geometry.
          for (uint32_t lane = 0; lane < wave_size; ++lane) {
            addresses[lane] = kBase + offsets[order[lane]];
            for (uint32_t byte = 0; byte < width; ++byte) {
              source[lane * width + byte] = static_cast<uint8_t>(random());
              if (!(mask & (uint64_t{1} << lane)))
                continue;
              const size_t at = addresses[lane] - kBase + byte;
              ASSERT_LT(at, kBytes);
              requested[at] = true;
              expected_store[at] = source[lane * width + byte];
              expected_load[lane * width + byte] = read_source[at];
            }
          }
          uint64_t active_address = kBase;
          for (uint32_t lane = 0; lane < wave_size; ++lane)
            if (mask & (uint64_t{1} << lane)) {
              active_address = addresses[lane];
              break;
            }
          for (uint32_t lane = 0; lane < wave_size; ++lane)
            if (!(mask & (uint64_t{1} << lane)))
              addresses[lane] = trial % 2 ? active_address : UINT64_MAX;
          uint64_t expected_transactions = 0;
          for (size_t byte = 0; byte < requested.size(); ++byte)
            if (requested[byte] &&
                (byte == 0 || !requested[byte - 1] || (kBase + byte) % policy.bytes() == 0))
              ++expected_transactions;

          GpuMemory memory("memory", policy);
          L2Cache l2("l2");
          l2.set_backing_memory(&memory);
          L1VectorCache l1(&l2, policy);
          actual.fill(0xa5);
          memory.write_block(kBase, actual);
          const uint32_t element_size = std::min(width, 4u);
          ASSERT_EQ(l1.store(addresses.data(), mask, element_size, width / element_size,
                             source.data(), Mtype::UC, false, wave_size),
                    rocjitsu::amdgpu::VmAccessOutcome::Complete);
          EXPECT_EQ(l2.backing_write_transactions(), expected_transactions);
          memory.read_block(kBase, actual);
          EXPECT_EQ(actual, expected_store);

          // Load a different backing pattern so mutually inverse store/load
          // mistakes cannot cancel each other in a round-trip comparison.
          memory.write_block(kBase, read_source);
          ASSERT_EQ(l1.load(addresses.data(), mask, element_size, width / element_size,
                            loaded.data(), Mtype::UC, false, true, wave_size),
                    rocjitsu::amdgpu::VmAccessOutcome::Complete);
          EXPECT_EQ(l2.backing_read_transactions(), expected_transactions);
          EXPECT_EQ(loaded, expected_load);
        }
      }
    }
  }
}

TEST(L2CacheTest, PartialOverlapWithEarlierActiveLaneMatchesByteOrder) {
  using Policy = rocjitsu::amdgpu::MemoryTransactionPolicy;
  constexpr uint64_t kBase = 0x180000;
  for (auto policy : {Policy{}, Policy::unqualified_64()}) {
    for (uint32_t width : {1u, 2u, 4u, 8u, 12u, 16u}) {
      for (uint64_t mask : {uint64_t{0xff}, uint64_t{0xd7}}) {
        SCOPED_TRACE(std::format("policy={} width={} mask={}", policy.bytes(), width, mask));
        // The first descending operand overlaps an earlier active lane rather
        // than its immediate predecessor, including under the sparse mask.
        const std::array<uint32_t, 8> offsets{0,          3 * width,
                                              6 * width,  width / 2,
                                              9 * width,  3 * width + width / 2,
                                              12 * width, 6 * width + width / 2};
        std::array<uint64_t, 8> addresses{};
        std::array<uint8_t, 8 * 16> source{}, loaded{}, expected_load{};
        std::array<uint8_t, 256> actual{}, expected_store{}, read_source{};
        actual.fill(0xa5);
        expected_store = actual;
        loaded.fill(0xcd);
        expected_load = loaded;
        for (size_t byte = 0; byte < read_source.size(); ++byte)
          read_source[byte] = static_cast<uint8_t>(byte ^ 0x5a);
        for (uint32_t lane = 0; lane < addresses.size(); ++lane) {
          addresses[lane] = kBase + 5 + offsets[lane];
          for (uint32_t byte = 0; byte < width; ++byte) {
            source[lane * width + byte] = static_cast<uint8_t>(lane * 31 + byte);
            if (!(mask & (uint64_t{1} << lane)))
              continue;
            const size_t at = addresses[lane] - kBase + byte;
            ASSERT_LT(at, expected_store.size());
            expected_store[at] = source[lane * width + byte];
            expected_load[lane * width + byte] = read_source[at];
          }
        }
        GpuMemory memory("memory", policy);
        memory.write_block(kBase, actual);
        L2Cache l2("l2");
        l2.set_backing_memory(&memory);
        L1VectorCache l1(&l2, policy);
        const uint32_t element_size = std::min(width, 4u);
        ASSERT_EQ(l1.store(addresses.data(), mask, element_size, width / element_size,
                           source.data(), Mtype::UC, false, addresses.size()),
                  rocjitsu::amdgpu::VmAccessOutcome::Complete);
        memory.read_block(kBase, actual);
        EXPECT_EQ(actual, expected_store);
        memory.write_block(kBase, read_source);
        ASSERT_EQ(l1.load(addresses.data(), mask, element_size, width / element_size, loaded.data(),
                          Mtype::UC, false, true, addresses.size()),
                  rocjitsu::amdgpu::VmAccessOutcome::Complete);
        EXPECT_EQ(loaded, expected_load);
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

TEST(L2CacheTest, PostExternalWriteDropsDirtyLineWithoutPublishingOldBytes) {
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kExternalOffset = 32;
  constexpr uint32_t kExternalValue = 0xaabbccdd;
  auto translator = std::make_shared<IdentityAddressSpace>(0x2000);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  ASSERT_TRUE(gpu_vm.register_translated(kVmid, translator, physical));
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&gpu_vm);

  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  ASSERT_EQ(l2.writeback_line(kAddress, dirty.data(), Mtype::RW, kVmid),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  physical->store(kAddress + kExternalOffset, kExternalValue);
  physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Faulted;

  l2.invalidate_range_after_external_write(kAddress + kExternalOffset, sizeof(kExternalValue),
                                           kVmid);
  EXPECT_EQ(physical->write_calls, 0u);
  EXPECT_EQ(physical->load<uint32_t>(kAddress + kExternalOffset), kExternalValue);
  physical->write_outcome = rocjitsu::amdgpu::VmAccessOutcome::Complete;
  ASSERT_EQ(l2.flush_all(), rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(physical->write_calls, 0u);
}

TEST(L2CacheTest, PreExternalWritePreservesDirtySiblingsAcrossL2AndMemorySideCache) {
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kExternalOffset = 32;
  constexpr uint32_t kExternalValue = 0xaabbccdd;
  LinkedCacheHierarchy hierarchy(0x2000);
  std::array<uint8_t, L2Cache::LINE_SIZE> memory_side_dirty{};
  std::array<uint8_t, L2Cache::LINE_SIZE> l2_dirty{};
  std::array<uint8_t, L2Cache::LINE_SIZE> observed{};
  memory_side_dirty.fill(0x33);
  l2_dirty.fill(0x5a);
  MemorySideCacheTestAccess::store_dirty_line(hierarchy.memory_side_cache, kAddress,
                                              memory_side_dirty, LinkedCacheHierarchy::kVmid);
  ASSERT_EQ(hierarchy.l2.writeback_line(kAddress, l2_dirty.data(), 0, sizeof(uint32_t), Mtype::RW,
                                        LinkedCacheHierarchy::kVmid),
            VmAccessOutcome::Complete);
  ASSERT_EQ(hierarchy.l2.writeback_line(kAddress, l2_dirty.data(), 64, sizeof(uint32_t), Mtype::RW,
                                        LinkedCacheHierarchy::kVmid),
            VmAccessOutcome::Complete);

  // The L2's retained-binding publication goes directly to physical backing.
  // First retire older MSC dirty state so it cannot overwrite newer L2 bytes.
  ASSERT_EQ(hierarchy.memory_side_cache.flush_range(
                kAddress + kExternalOffset, sizeof(kExternalValue), LinkedCacheHierarchy::kVmid),
            VmAccessOutcome::Complete);
  ASSERT_EQ(hierarchy.l2.flush_range_before_external_write(
                kAddress + kExternalOffset, sizeof(kExternalValue), LinkedCacheHierarchy::kVmid),
            VmAccessOutcome::Complete);
  EXPECT_EQ(hierarchy.physical->write_calls, 3u);
  hierarchy.physical->store(kAddress + kExternalOffset, kExternalValue);
  hierarchy.l2.invalidate_range_after_external_write(
      kAddress + kExternalOffset, sizeof(kExternalValue), LinkedCacheHierarchy::kVmid);
  hierarchy.memory_side_cache.invalidate_range_after_external_write(
      kAddress + kExternalOffset, sizeof(kExternalValue), LinkedCacheHierarchy::kVmid);
  ASSERT_EQ(hierarchy.l2.read(kAddress, observed.data(), observed.size(), Mtype::RW,
                              LinkedCacheHierarchy::kVmid),
            VmAccessOutcome::Complete);
  std::array<uint8_t, L2Cache::LINE_SIZE> expected = memory_side_dirty;
  std::memcpy(expected.data(), l2_dirty.data(), sizeof(uint32_t));
  std::memcpy(expected.data() + 64, l2_dirty.data() + 64, sizeof(uint32_t));
  std::memcpy(expected.data() + kExternalOffset, &kExternalValue, sizeof(kExternalValue));
  EXPECT_EQ(observed, expected);
  EXPECT_EQ(hierarchy.physical->write_calls, 3u);
}

TEST(L2CacheTest, ExternalWriteRangesSelectVmidAndResidentLinesOnly) {
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint32_t kVmidA = 73;
  constexpr uint32_t kVmidB = 74;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kOtherAddress = kAddress + 2 * L2Cache::LINE_SIZE;
  constexpr uint32_t kInitial = 0x11223344;
  constexpr uint32_t kReplacement = 0xaabbccdd;
  auto translator = std::make_shared<IdentityAddressSpace>(0x2000);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  ASSERT_TRUE(gpu_vm.register_translated(kVmidA, translator, physical));
  ASSERT_TRUE(gpu_vm.register_translated(kVmidB, translator, physical));
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&gpu_vm);
  physical->store(kAddress, kInitial);
  physical->store(kOtherAddress, kInitial);
  uint32_t observed = 0;
  for (const auto &[address, vmid] : {std::pair{kAddress, kVmidA}, std::pair{kAddress, kVmidB},
                                      std::pair{kOtherAddress, kVmidA}}) {
    ASSERT_EQ(
        l2.read(address, reinterpret_cast<uint8_t *>(&observed), sizeof(observed), Mtype::RW, vmid),
        VmAccessOutcome::Complete);
    ASSERT_EQ(observed, kInitial);
  }
  const uint32_t reads_before = physical->read_calls;
  ASSERT_EQ(l2.flush_range_before_external_write(kAddress, 0, kVmidA), VmAccessOutcome::Complete);
  l2.invalidate_range_after_external_write(kAddress, 0, kVmidA);
  ASSERT_EQ(l2.flush_range_before_external_write(kAddress, sizeof(kReplacement), kVmidA),
            VmAccessOutcome::Complete);
  physical->store(kAddress, kReplacement);
  l2.invalidate_range_after_external_write(kAddress, sizeof(kReplacement), kVmidA);
  ASSERT_EQ(l2.read(kAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed), Mtype::RW,
                    kVmidA),
            VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kReplacement);
  ASSERT_EQ(l2.read(kAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed), Mtype::RW,
                    kVmidB),
            VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kInitial);
  ASSERT_EQ(l2.read(kOtherAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed),
                    Mtype::RW, kVmidA),
            VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kInitial);
  EXPECT_EQ(physical->read_calls, reads_before + 1);

  // This span covers millions of addresses but visits bounded cache storage.
  ASSERT_EQ(l2.flush_range_before_external_write(0, std::numeric_limits<uint32_t>::max(), kVmidA),
            VmAccessOutcome::Complete);
  l2.invalidate_range_after_external_write(0, std::numeric_limits<uint32_t>::max(), kVmidA);
  EXPECT_EQ(physical->write_calls, 0u);
}

TEST(L2CacheTest, ExternalWriteRangesClampAtLastAddressableLine) {
  constexpr uint64_t kLastLine = std::numeric_limits<uint64_t>::max() - (L2Cache::LINE_SIZE - 1);
  constexpr uint64_t kHighAddress = kLastLine + L2Cache::LINE_SIZE / 2;
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  ASSERT_EQ(l2.writeback_line(kLastLine, dirty.data()),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  ASSERT_EQ(l2.flush_range_before_external_write(kHighAddress, L2Cache::LINE_SIZE, 0),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(memory.read32(kLastLine), 0x5a5a5a5au);
  memory.write32(kHighAddress, 0xaabbccdd);
  l2.invalidate_range_after_external_write(kHighAddress, L2Cache::LINE_SIZE, 0);
  EXPECT_EQ(memory.read32(kHighAddress), 0xaabbccddu);
  EXPECT_EQ(memory.read32(0), 0u);
}

TEST(L2CacheTest, FailedExternalPrewriteRetainsDirtyBytesForRetry) {
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  auto translator = std::make_shared<IdentityAddressSpace>(0x2000);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  ASSERT_TRUE(gpu_vm.register_translated(kVmid, translator, physical));
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&gpu_vm);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  ASSERT_EQ(l2.writeback_line(kAddress, dirty.data(), Mtype::RW, kVmid), VmAccessOutcome::Complete);
  physical->write_outcome = VmAccessOutcome::Faulted;
  EXPECT_EQ(l2.flush_range_before_external_write(kAddress + 32, 4, kVmid),
            VmAccessOutcome::Faulted);
  EXPECT_EQ(physical->load<uint32_t>(kAddress), 0u);
  physical->write_outcome = VmAccessOutcome::Complete;
  ASSERT_EQ(l2.flush_range_before_external_write(kAddress + 32, 4, kVmid),
            VmAccessOutcome::Complete);
  EXPECT_EQ(physical->load<uint32_t>(kAddress), 0x5a5a5a5au);
  EXPECT_EQ(physical->write_calls, 2u);
}

TEST(L2CacheTest, NonlocalAliasPrewriteRetainsDirtyBytesForCoordinatedRetry) {
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  auto translator = std::make_shared<DeferredSwitchAddressSpace>(0x2000);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  std::atomic<uint32_t> fault_count{0};
  ASSERT_TRUE(gpu_vm.register_address_space(
      kVmid, translator, physical,
      [&](uint64_t, rocjitsu::amdgpu::VmAccessKind) { ++fault_count; }));
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&gpu_vm);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  ASSERT_EQ(l2.writeback_line(kAddress, dirty.data(), Mtype::RW, kVmid), VmAccessOutcome::Complete);

  translator->route = rocjitsu::amdgpu::VmDataRoute::Deferred;
  EXPECT_EQ(l2.flush_range_before_external_write(kAddress + 32, 4, kVmid),
            VmAccessOutcome::Unavailable);
  EXPECT_EQ(physical->write_calls, 0u);
  EXPECT_EQ(fault_count, 0u);
  translator->route = rocjitsu::amdgpu::VmDataRoute::Faulted;
  EXPECT_EQ(l2.flush_range_before_external_write(kAddress + 32, 4, kVmid),
            VmAccessOutcome::Faulted);
  EXPECT_EQ(physical->write_calls, 0u);
  EXPECT_EQ(fault_count, 0u);
  const auto architectural_access = gpu_vm.snapshot_vmid(kVmid);
  ASSERT_TRUE(architectural_access);
  EXPECT_EQ(architectural_access->classify_data_route(kAddress + 32, 1,
                                                      rocjitsu::amdgpu::VmAccessKind::Write),
            rocjitsu::amdgpu::VmDataRoute::Faulted);
  EXPECT_EQ(fault_count, 1u);
  translator->route = rocjitsu::amdgpu::VmDataRoute::Local;
  ASSERT_EQ(l2.flush_range_before_external_write(kAddress + 32, 4, kVmid),
            VmAccessOutcome::Complete);
  EXPECT_EQ(physical->load<uint32_t>(kAddress), 0x5a5a5a5au);
  EXPECT_EQ(physical->write_calls, 1u);
  EXPECT_EQ(fault_count, 1u);
}

TEST(L2CacheTest, CompletionFlushSelectsExactVmidAndCacheIdentity) {
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint32_t kSelectedVmid = 73;
  constexpr uint32_t kOtherVmid = 74;
  constexpr uint64_t kSelectedDirty = 0x1000;
  constexpr uint64_t kSelectedClean = 0x1100;
  constexpr uint64_t kStaleIdentity = 0x1200;
  constexpr uint64_t kOtherDirty = 0x1300;
  constexpr uint64_t kOtherClean = 0x1400;
  auto translator = std::make_shared<IdentityAddressSpace>(0x3000);
  auto selected_physical = std::make_shared<ConfigurablePhysicalMemory>(0x3000);
  auto other_physical = std::make_shared<ConfigurablePhysicalMemory>(0x3000);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm vm;
  ASSERT_TRUE(vm.register_translated(kSelectedVmid, translator, selected_physical));
  ASSERT_TRUE(vm.register_translated(kOtherVmid, translator, other_physical));
  const auto access = vm.snapshot_vmid(kSelectedVmid);
  ASSERT_TRUE(access);
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&vm);
  std::array<uint8_t, L2Cache::LINE_SIZE> bytes{};
  bytes.fill(0x5a);
  const std::array<std::pair<uint32_t, uint32_t>, 1> dirty_range{{{8, 4}}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kSelectedDirty, bytes, kSelectedVmid,
                                               access->cache_identity(), 0, dirty_range);
  L2CacheTestAccess::store_prepared_clean_line(l2, kSelectedClean, bytes, kSelectedVmid,
                                               access->cache_identity(), 0);
  L2CacheTestAccess::store_prepared_dirty_line(l2, kStaleIdentity, bytes, kSelectedVmid,
                                               access->cache_identity() + 1, 0, dirty_range);
  L2CacheTestAccess::store_prepared_dirty_line(l2, kOtherDirty, bytes, kOtherVmid,
                                               access->cache_identity(), 0, dirty_range);
  L2CacheTestAccess::store_prepared_clean_line(l2, kOtherClean, bytes, kOtherVmid,
                                               access->cache_identity(), 0);

  EXPECT_EQ(l2.flush_for_completion(*access), VmAccessOutcome::Complete);
  EXPECT_EQ(selected_physical->load<uint32_t>(kSelectedDirty + 8), 0x5a5a5a5au);
  EXPECT_EQ(selected_physical->load<uint32_t>(kStaleIdentity + 8), 0u);
  EXPECT_EQ(other_physical->write_calls, 0u);
  EXPECT_FALSE(L2CacheTestAccess::resident(l2, kSelectedDirty, kSelectedVmid));
  EXPECT_FALSE(L2CacheTestAccess::resident(l2, kSelectedClean, kSelectedVmid));
  EXPECT_TRUE(L2CacheTestAccess::resident(l2, kStaleIdentity, kSelectedVmid));
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kStaleIdentity, kSelectedVmid, 8));
  EXPECT_TRUE(L2CacheTestAccess::resident(l2, kOtherDirty, kOtherVmid));
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kOtherDirty, kOtherVmid, 8));
  EXPECT_TRUE(L2CacheTestAccess::resident(l2, kOtherClean, kOtherVmid));
}

TEST(L2CacheTest, CompletionRetiresOnlyTheSelectedRootsCuLines) {
  using rocjitsu::amdgpu::CompletionTracker;
  using rocjitsu::amdgpu::ComputeQueueRecord;
  using rocjitsu::amdgpu::ComputeUnitCore;
  using rocjitsu::amdgpu::DispatchEntry;
  using rocjitsu::amdgpu::DispatchPacketKind;
  using rocjitsu::amdgpu::GpuVmAccess;
  using rocjitsu::amdgpu::InstructionCache;
  using rocjitsu::amdgpu::L1ScalarCacheTestAccess;
  using rocjitsu::amdgpu::L1VectorCacheTestAccess;
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;

  constexpr uint32_t kSelectedVmid = 73;
  constexpr uint32_t kOtherVmid = 74;
  constexpr uint64_t kSelectedData = 0x1000;
  constexpr uint64_t kSelectedSecondGeneration = 0x1080;
  constexpr uint64_t kOldRootData = 0x1100;
  constexpr uint64_t kOtherData = 0x1200;
  constexpr uint64_t kOldCode = 0x600;
  constexpr uint64_t kSelectedCode = 0x640;
  constexpr uint64_t kOtherCode = 0x680;

  auto translator = std::make_shared<IdentityAddressSpace>(0x3000);
  auto old_physical = std::make_shared<ConfigurablePhysicalMemory>(0x3000);
  auto selected_physical = std::make_shared<ConfigurablePhysicalMemory>(0x3000);
  auto other_physical = std::make_shared<ConfigurablePhysicalMemory>(0x3000);
  rocjitsu::amdgpu::GpuVm vm;
  const rocjitsu::amdgpu::AddressSpaceHandle selected_handle =
      vm.register_translated(kSelectedVmid, translator, old_physical);
  ASSERT_TRUE(selected_handle);
  const std::optional<GpuVmAccess> old_ordinary = vm.snapshot(selected_handle);
  const std::optional<GpuVmAccess> old_access = vm.snapshot_pinned(selected_handle);
  ASSERT_TRUE(old_ordinary);
  ASSERT_TRUE(old_access);
  ASSERT_TRUE(vm.replace_translated(selected_handle, translator, selected_physical));
  const std::optional<GpuVmAccess> selected_ordinary = vm.snapshot(selected_handle);
  const std::optional<GpuVmAccess> selected_access = vm.snapshot_pinned(selected_handle);
  ASSERT_TRUE(selected_ordinary);
  ASSERT_TRUE(selected_access);
  const rocjitsu::amdgpu::AddressSpaceHandle other_handle =
      vm.register_translated(kOtherVmid, translator, other_physical);
  ASSERT_TRUE(other_handle);
  const std::optional<GpuVmAccess> other_access = vm.snapshot(other_handle);
  ASSERT_TRUE(other_access);
  ASSERT_NE(old_ordinary->cache_identity(), selected_ordinary->cache_identity());
  ASSERT_NE(old_access->cache_namespace(), selected_access->cache_namespace());

  GpuMemory memory("completion_l1_memory");
  L2Cache l2("completion_l1_l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&vm);
  ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_CDNA3;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 104;
  config.vgprs_per_wf = 256;
  config.lds_size_kb = 64;
  std::unique_ptr<ComputeUnitCore> cu =
      ComputeUnitCore::create("completion_l1_cu", config, &memory, &l2);
  ASSERT_NE(cu, nullptr);
  cu->set_gpu_vm(&vm);

  for (const uint64_t address : {kSelectedData, kSelectedSecondGeneration}) {
    const uint64_t generation = address == kSelectedData ? 1 : 2;
    L1ScalarCacheTestAccess::store_clean_line(cu->l1_scalar(), address, kSelectedVmid, generation,
                                              selected_ordinary->cache_identity());
    L1VectorCacheTestAccess::store_clean_line(cu->l1_vector(), address, kSelectedVmid, generation,
                                              selected_ordinary->cache_identity());
  }
  L1ScalarCacheTestAccess::store_clean_line(cu->l1_scalar(), kOldRootData, kSelectedVmid, 1,
                                            old_ordinary->cache_identity());
  L1VectorCacheTestAccess::store_clean_line(cu->l1_vector(), kOldRootData, kSelectedVmid, 1,
                                            old_ordinary->cache_identity());
  L1ScalarCacheTestAccess::store_clean_line(cu->l1_scalar(), kOtherData, kOtherVmid, 1,
                                            other_access->cache_identity());
  L1VectorCacheTestAccess::store_clean_line(cu->l1_vector(), kOtherData, kOtherVmid, 1,
                                            other_access->cache_identity());
  std::array<uint8_t, L2Cache::LINE_SIZE> line{};
  L2CacheTestAccess::store_prepared_clean_line(l2, kSelectedData, line, kSelectedVmid,
                                               selected_ordinary->cache_identity(), 1);
  L2CacheTestAccess::store_prepared_clean_line(l2, kOtherData, line, kOtherVmid,
                                               other_access->cache_identity(), 1);

  old_physical->store(kOldCode, uint32_t{0x11111111});
  selected_physical->store(kSelectedCode, uint32_t{0x22222222});
  other_physical->store(kOtherCode, uint32_t{0x33333333});
  const auto fetch_word = [&](const GpuVmAccess &access, uint64_t address) {
    std::array<uint8_t, InstructionCache::kFetchBytes> bytes{};
    EXPECT_EQ(cu->instruction_cache().fetch(access, address, bytes.data()),
              VmAccessOutcome::Complete);
    uint32_t word = 0;
    std::memcpy(&word, bytes.data(), sizeof(word));
    return word;
  };
  ASSERT_EQ(fetch_word(*old_access, kOldCode), 0x11111111u);
  ASSERT_EQ(fetch_word(*selected_access, kSelectedCode), 0x22222222u);
  ASSERT_EQ(fetch_word(*other_access, kOtherCode), 0x33333333u);
  old_physical->store(kOldCode, uint32_t{0xaaaaaaaa});
  selected_physical->store(kSelectedCode, uint32_t{0xbbbbbbbb});
  other_physical->store(kOtherCode, uint32_t{0xcccccccc});
  const uint64_t instruction_epoch = cu->instruction_cache().epoch();

  std::vector<ComputeUnitCore *> cus{cu.get()};
  std::vector<L2Cache *> l2_caches{&l2};
  CompletionTracker tracker(vm, cus, l2_caches);
  ComputeQueueRecord queue{};
  queue.fanout_replica = true;
  DispatchEntry entry{};
  entry.dispatch_id = 1;
  entry.address_space = selected_handle;
  entry.execution_access = std::make_shared<GpuVmAccess>(*selected_access);
  entry.execution_translation_epoch = selected_access->info().translation_epoch;
  entry.execution_cache_write_sequence = selected_access->cache_write_sequence();
  entry.process_id = kSelectedVmid;
  entry.total_wgs = entry.completed_wgs = 1;
  entry.kind = DispatchPacketKind::Kernel;
  queue.push_entry(std::move(entry));
  std::vector<ComputeQueueRecord> queues;
  queues.push_back(std::move(queue));

  const rocjitsu::amdgpu::CompletionDrainResult result = tracker.drain_completions(queues);
  ASSERT_FALSE(result.retry_pending);
  ASSERT_FALSE(result.terminal_fault);
  EXPECT_TRUE(queues.front().entries.empty());
  for (const uint64_t address : {kSelectedData, kSelectedSecondGeneration}) {
    EXPECT_FALSE(L1ScalarCacheTestAccess::resident(cu->l1_scalar(), address, kSelectedVmid));
    EXPECT_FALSE(L1VectorCacheTestAccess::resident(cu->l1_vector(), address, kSelectedVmid));
  }
  EXPECT_TRUE(L1ScalarCacheTestAccess::resident(cu->l1_scalar(), kOldRootData, kSelectedVmid));
  EXPECT_TRUE(L1VectorCacheTestAccess::resident(cu->l1_vector(), kOldRootData, kSelectedVmid));
  EXPECT_TRUE(L1ScalarCacheTestAccess::resident(cu->l1_scalar(), kOtherData, kOtherVmid));
  EXPECT_TRUE(L1VectorCacheTestAccess::resident(cu->l1_vector(), kOtherData, kOtherVmid));
  EXPECT_FALSE(L2CacheTestAccess::resident(l2, kSelectedData, kSelectedVmid));
  EXPECT_TRUE(L2CacheTestAccess::resident(l2, kOtherData, kOtherVmid));
  EXPECT_EQ(cu->instruction_cache().epoch(), instruction_epoch + 1);
  EXPECT_EQ(fetch_word(*selected_access, kSelectedCode), 0xbbbbbbbbu);
  EXPECT_EQ(fetch_word(*old_access, kOldCode), 0x11111111u);
  EXPECT_EQ(fetch_word(*other_access, kOtherCode), 0x33333333u);
}

TEST(L2CacheTest, CompletionFlushFaultsLinkedPathAndRetriesDirtyWrite) {
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  auto translator = std::make_shared<IdentityAddressSpace>(0x2000);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm vm;
  ASSERT_TRUE(vm.register_translated(kVmid, translator, physical));
  const auto access = vm.snapshot_vmid(kVmid);
  ASSERT_TRUE(access);
  L2Cache l2("l2");
  l2.set_gpu_vm(&vm);
  std::array<uint8_t, L2Cache::LINE_SIZE> bytes{};
  bytes.fill(0xa5);
  const std::array<std::pair<uint32_t, uint32_t>, 1> dirty_range{{{16, 4}}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, bytes, kVmid, access->cache_identity(),
                                               0, dirty_range);

  EXPECT_EQ(l2.flush_for_completion(*access), VmAccessOutcome::Malformed);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 16));
  EXPECT_EQ(physical->write_calls, 0u);

  l2.set_backing_memory(&memory);
  translator->direct_backing_write_allowed = false;
  EXPECT_EQ(l2.flush_for_completion(*access), VmAccessOutcome::Unavailable);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 16));
  EXPECT_EQ(physical->write_calls, 0u);

  translator->direct_backing_write_allowed = true;
  physical->write_outcome = VmAccessOutcome::Unavailable;
  EXPECT_EQ(l2.flush_for_completion(*access), VmAccessOutcome::Unavailable);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 16));
  EXPECT_TRUE(L2CacheTestAccess::resident(l2, kAddress, kVmid));
  EXPECT_EQ(physical->load<uint32_t>(kAddress + 16), 0u);

  physical->write_outcome = VmAccessOutcome::Complete;
  EXPECT_EQ(l2.flush_for_completion(*access), VmAccessOutcome::Complete);
  EXPECT_FALSE(L2CacheTestAccess::resident(l2, kAddress, kVmid));
  EXPECT_EQ(physical->load<uint32_t>(kAddress + 16), 0xa5a5a5a5u);
}

TEST(L2CacheTest, CompletionFlushRetiresOnlyCommittedPrefixAcrossTranslations) {
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  auto translator =
      std::make_shared<IdentityAddressSpace>(0x2000, Mtype::RW, 0, /*translation_span=*/64);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm vm;
  ASSERT_TRUE(vm.register_translated(kVmid, translator, physical));
  const auto access = vm.snapshot_vmid(kVmid);
  ASSERT_TRUE(access);
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&vm);
  std::array<uint8_t, L2Cache::LINE_SIZE> bytes{};
  bytes.fill(0x6b);
  const std::array<std::pair<uint32_t, uint32_t>, 1> dirty_range{{{48, 32}}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, bytes, kVmid, access->cache_identity(),
                                               0, dirty_range);
  physical->fail_on_write_call = 2;

  EXPECT_EQ(l2.flush_for_completion(*access), VmAccessOutcome::Unavailable);
  EXPECT_EQ(physical->write_calls, 2u);
  EXPECT_EQ(physical->load<uint32_t>(kAddress + 48), 0x6b6b6b6bu);
  EXPECT_EQ(physical->load<uint32_t>(kAddress + 64), 0u);
  EXPECT_FALSE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 48));
  EXPECT_FALSE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 63));
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 64));
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 79));
  EXPECT_TRUE(L2CacheTestAccess::resident(l2, kAddress, kVmid));

  physical->fail_on_write_call = 0;
  EXPECT_EQ(l2.flush_for_completion(*access), VmAccessOutcome::Complete);
  EXPECT_EQ(physical->write_calls, 3u);
  EXPECT_EQ(physical->load<uint32_t>(kAddress + 64), 0x6b6b6b6bu);
  EXPECT_FALSE(L2CacheTestAccess::resident(l2, kAddress, kVmid));
}

TEST(L2CacheTest, CompletionFlushRetainsDirtyBytesOnStaleGrantAndRevocation) {
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  auto translator = std::make_shared<IdentityAddressSpace>(0x2000, Mtype::RW,
                                                           /*admission_generation=*/24);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm vm;
  const auto handle = vm.register_translated(kVmid, translator, physical);
  ASSERT_TRUE(handle);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&vm);
  std::array<uint8_t, L2Cache::LINE_SIZE> bytes{};
  bytes.fill(0x6b);
  const std::array<std::pair<uint32_t, uint32_t>, 1> dirty_range{{{8, 4}}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, bytes, kVmid, access->cache_identity(),
                                               23, dirty_range);

  EXPECT_EQ(l2.flush_for_completion(*access), VmAccessOutcome::Faulted);
  EXPECT_EQ(physical->write_calls, 0u);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 8));
  ASSERT_TRUE(vm.unregister_address_space(handle));
  EXPECT_EQ(l2.flush_for_completion(*access), VmAccessOutcome::Revoked);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 8));
}

TEST(L2CacheTest, PreparedCanonicalPrewritePublishesOnlyCommittedDirtySiblings) {
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kIdentity = 101;
  constexpr uint64_t kGeneration = 23;
  L2Cache l2("l2");
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  const std::array<std::pair<uint32_t, uint32_t>, 2> dirty_ranges{{{4, 4}, {112, 4}}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, dirty, kVmid, kIdentity, kGeneration,
                                               dirty_ranges);

  auto owner = std::make_shared<PreparedBackingOwner>();
  owner->fail_on_attempt = 2;
  rocjitsu::FabricBacking backing(owner, 1, PreparedBackingOwner::kBaseOffset, dirty.size());
  const auto prepared = PreparedCanonicalWritebackTestAccess::make(kAddress, dirty.size(), kVmid,
                                                                   kIdentity, kGeneration, backing);
  ASSERT_TRUE(prepared);

  // The four-byte external write is elsewhere in this line. Both dirty runs
  // are siblings and must be published before that effect can begin.
  EXPECT_EQ(l2.flush_range_before_external_write(kAddress + 32, 4, kVmid, prepared),
            VmAccessOutcome::Unavailable);
  EXPECT_EQ(owner->write_attempts, 2u);
  EXPECT_EQ(owner->bytes_[4], std::byte{0x5a});
  EXPECT_EQ(owner->bytes_[112], std::byte{0});
  EXPECT_FALSE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 4));
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 112));
  EXPECT_TRUE(L2CacheTestAccess::resident(l2, kAddress, kVmid));

  owner->fail_on_attempt = 0;
  ASSERT_EQ(l2.flush_range_before_external_write(kAddress + 32, 4, kVmid, prepared),
            VmAccessOutcome::Complete);
  EXPECT_EQ(owner->write_attempts, 3u);
  EXPECT_EQ(owner->bytes_[112], std::byte{0x5a});
  EXPECT_EQ(owner->bytes_[32], std::byte{0});
  EXPECT_FALSE(L2CacheTestAccess::resident(l2, kAddress, kVmid));
}

TEST(L2CacheTest, MovedFromPreparedCanonicalWritebackCannotAuthorizeEmptyRange) {
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWriteback;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  L2Cache l2("l2");
  auto owner = std::make_shared<PreparedBackingOwner>();
  PreparedCanonicalWriteback prepared = PreparedCanonicalWritebackTestAccess::make(
      kAddress, L2Cache::LINE_SIZE, kVmid, 101, 23,
      rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, L2Cache::LINE_SIZE));
  ASSERT_TRUE(prepared);
  PreparedCanonicalWriteback retained = std::move(prepared);
  EXPECT_EQ(prepared.outcome(), VmAccessOutcome::Unavailable);
  EXPECT_FALSE(prepared);
  EXPECT_EQ(l2.flush_range_before_external_write(kAddress, 4, kVmid, prepared),
            VmAccessOutcome::Unavailable);
  EXPECT_EQ(l2.flush_range_before_external_write(kAddress, 4, kVmid, retained),
            VmAccessOutcome::Complete);
}

TEST(L2CacheTest, PreparedCanonicalPrewriteRefusedGateAndReusedVmidRetainDirtyBytes) {
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;
  using rocjitsu::amdgpu::VmTransferEffectGate;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kIdentity = 101;
  constexpr uint64_t kGeneration = 23;
  L2Cache l2("l2");
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  const std::array<std::pair<uint32_t, uint32_t>, 1> dirty_ranges{{{8, 4}}};
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, dirty, kVmid, kIdentity, kGeneration,
                                               dirty_ranges);

  auto old_owner = std::make_shared<PreparedBackingOwner>();
  auto old_gate = std::make_shared<VmTransferEffectGate>();
  const auto old_prepared = PreparedCanonicalWritebackTestAccess::make(
      kAddress, dirty.size(), kVmid, kIdentity, kGeneration,
      rocjitsu::FabricBacking(old_owner, 1, PreparedBackingOwner::kBaseOffset, dirty.size()),
      old_gate);
  ASSERT_TRUE(old_prepared);
  old_gate->revoke();
  EXPECT_EQ(l2.flush_range_before_external_write(kAddress + 32, 4, kVmid, old_prepared),
            VmAccessOutcome::Revoked);
  EXPECT_EQ(old_owner->write_attempts, 0u);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 8));

  auto new_owner = std::make_shared<PreparedBackingOwner>();
  const auto new_prepared = PreparedCanonicalWritebackTestAccess::make(
      kAddress, dirty.size(), kVmid, kIdentity + 1, kGeneration,
      rocjitsu::FabricBacking(new_owner, 1, PreparedBackingOwner::kBaseOffset, dirty.size()));
  ASSERT_TRUE(new_prepared);
  EXPECT_EQ(l2.flush_range_before_external_write(kAddress + 32, 4, kVmid, new_prepared),
            VmAccessOutcome::Unavailable);
  EXPECT_EQ(new_owner->write_attempts, 0u);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 8));
}

TEST(MemorySideCacheTest, PreparedCanonicalPrewriteRequiresWholeLineAndRetries) {
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  using rocjitsu::amdgpu::PreparedBackingOwner;
  using rocjitsu::amdgpu::PreparedCanonicalWritebackTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;

  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kIdentity = 101;
  constexpr uint64_t kGeneration = 23;
  MemorySideCache cache("msc");
  std::array<uint8_t, MemorySideCache::LINE_SIZE> dirty{};
  dirty.fill(0x6b);
  MemorySideCacheTestAccess::store_prepared_dirty_line(cache, kAddress, dirty, kVmid, kIdentity,
                                                       kGeneration);

  auto owner = std::make_shared<PreparedBackingOwner>();
  const auto partial = PreparedCanonicalWritebackTestAccess::make(
      kAddress, dirty.size() / 2, kVmid, kIdentity, kGeneration,
      rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, dirty.size() / 2));
  ASSERT_TRUE(partial);
  EXPECT_EQ(cache.flush_range(kAddress + 32, 4, kVmid, partial), VmAccessOutcome::Unavailable);
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_TRUE(MemorySideCacheTestAccess::line_is_dirty(cache, kAddress, kVmid));

  const auto complete = PreparedCanonicalWritebackTestAccess::make(
      kAddress, dirty.size(), kVmid, kIdentity, kGeneration,
      rocjitsu::FabricBacking(owner, 1, PreparedBackingOwner::kBaseOffset, dirty.size()));
  ASSERT_TRUE(complete);
  MemorySideCacheTestAccess::set_byte_valid(cache, kAddress, kVmid, 16, false);
  EXPECT_EQ(cache.flush_range(kAddress + 32, 4, kVmid, complete), VmAccessOutcome::Unavailable);
  EXPECT_EQ(owner->write_attempts, 0u);
  EXPECT_TRUE(MemorySideCacheTestAccess::line_is_dirty(cache, kAddress, kVmid));
  MemorySideCacheTestAccess::set_byte_valid(cache, kAddress, kVmid, 16, true);
  owner->fail_on_attempt = 1;
  EXPECT_EQ(cache.flush_range(kAddress + 32, 4, kVmid, complete), VmAccessOutcome::Unavailable);
  EXPECT_EQ(owner->bytes_[32], std::byte{0});
  EXPECT_TRUE(MemorySideCacheTestAccess::line_is_dirty(cache, kAddress, kVmid));

  owner->fail_on_attempt = 0;
  ASSERT_EQ(cache.flush_range(kAddress + 32, 4, kVmid, complete), VmAccessOutcome::Complete);
  EXPECT_EQ(owner->bytes_[32], std::byte{0x6b});
  EXPECT_FALSE(MemorySideCacheTestAccess::line_is_resident(cache, kAddress, kVmid));
}

TEST(L2CacheTest, ExternalPrewriteDropsRetiredVmidBindingWithoutTouchingReplacement) {
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kNewBackingValue = 0xaabbccdd;
  auto translator = std::make_shared<IdentityAddressSpace>(0x2000);
  auto old_backing = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  auto new_backing = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  new_backing->store(kAddress, kNewBackingValue);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  const auto address_space = gpu_vm.register_translated(kVmid, translator, old_backing);
  ASSERT_TRUE(address_space);
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&gpu_vm);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  ASSERT_EQ(l2.writeback_line(kAddress, dirty.data(), Mtype::RW, kVmid), VmAccessOutcome::Complete);
  ASSERT_TRUE(gpu_vm.replace_translated(address_space, translator, new_backing));

  ASSERT_EQ(l2.flush_range_before_external_write(kAddress, 4, kVmid), VmAccessOutcome::Complete);
  EXPECT_EQ(old_backing->write_calls, 0u);
  EXPECT_EQ(new_backing->write_calls, 0u);
  EXPECT_EQ(new_backing->load<uint32_t>(kAddress), kNewBackingValue);
  uint32_t observed = 0;
  ASSERT_EQ(
      l2.read(kAddress, reinterpret_cast<uint8_t *>(&observed), sizeof(observed), Mtype::RW, kVmid),
      VmAccessOutcome::Complete);
  EXPECT_EQ(observed, kNewBackingValue);
}

TEST(L2CacheTest, ExternalPrewriteRetainsBindingAcrossConcurrentRebind) {
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kNewBackingValue = 0xaabbccdd;
  auto translator = std::make_shared<IdentityAddressSpace>(0x2000);
  auto old_backing = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  auto new_backing = std::make_shared<ConfigurablePhysicalMemory>(0x2000);
  new_backing->store(kAddress, kNewBackingValue);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm gpu_vm;
  const auto address_space = gpu_vm.register_translated(kVmid, translator, old_backing);
  ASSERT_TRUE(address_space);
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&gpu_vm);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  ASSERT_EQ(l2.writeback_line(kAddress, dirty.data(), Mtype::RW, kVmid), VmAccessOutcome::Complete);
  ASSERT_EQ(l2.writeback_line(kAddress + L2Cache::LINE_SIZE, dirty.data(), Mtype::RW, kVmid),
            VmAccessOutcome::Complete);

  std::mutex mutex;
  std::condition_variable condition;
  bool write_entered = false;
  bool release_write = false;
  old_backing->before_write = [&] {
    std::unique_lock lock(mutex);
    write_entered = true;
    condition.notify_all();
    condition.wait(lock, [&] { return release_write; });
  };
  auto flush = std::async(std::launch::async, [&] {
    return l2.flush_range_before_external_write(kAddress, 2 * L2Cache::LINE_SIZE, kVmid);
  });
  bool entered = false;
  {
    std::unique_lock lock(mutex);
    entered = condition.wait_for(lock, std::chrono::seconds(2), [&] { return write_entered; });
  }
  std::promise<void> replacement_started;
  std::future<void> replacement_running = replacement_started.get_future();
  auto replacement = std::async(std::launch::async, [&] {
    replacement_started.set_value();
    return gpu_vm.replace_translated(address_space, translator, new_backing);
  });
  replacement_running.wait();
  EXPECT_EQ(replacement.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
  {
    std::lock_guard lock(mutex);
    release_write = true;
  }
  condition.notify_all();
  EXPECT_TRUE(entered);
  EXPECT_EQ(flush.get(), VmAccessOutcome::Complete);
  ASSERT_TRUE(replacement.get());
  old_backing->before_write = {};
  EXPECT_EQ(new_backing->write_calls, 0u);
  EXPECT_EQ(new_backing->load<uint32_t>(kAddress), kNewBackingValue);
  ASSERT_EQ(l2.flush_range_before_external_write(kAddress, 2 * L2Cache::LINE_SIZE, kVmid),
            VmAccessOutcome::Complete);
  EXPECT_EQ(new_backing->write_calls, 0u);
}

TEST(PeerCacheTest, DirectProbeAndFillRequireMatchingRawKeyAndValidBytes) {
  using namespace rocjitsu::amdgpu;
  constexpr uint32_t kVmid = 7;
  constexpr uint64_t kGeneration = 19;
  constexpr uint64_t kIdentity = 23;
  constexpr uint64_t kAddress = 0x4004;

  L2Cache l2("l2");
  L1ScalarCache scalar(&l2);
  L1VectorCache vector(&l2);
  MemorySideCache memory_side("msc");

  const auto exercise = [&](auto &cache) {
    std::array<uint8_t, 2> source{0x31, 0x72};
    std::array<uint8_t, 2> result{0, 0};
    EXPECT_FALSE(cache.try_read_peer_cached(kAddress, result, kVmid, kGeneration, kIdentity));
    EXPECT_FALSE(cache.try_fill_peer_cached(kAddress, std::span<const uint8_t>{}, kVmid,
                                            kGeneration, kIdentity));
    EXPECT_FALSE(cache.try_fill_peer_cached(kAddress, std::span<const uint8_t>(source), kVmid, 0,
                                            kIdentity));
    EXPECT_FALSE(cache.try_fill_peer_cached(0x407f, source, kVmid, kGeneration, kIdentity));
    ASSERT_TRUE(cache.try_fill_peer_cached(kAddress, std::span<const uint8_t>(source).first(1),
                                           kVmid, kGeneration, kIdentity));
    EXPECT_FALSE(cache.try_read_peer_cached(kAddress, result, kVmid, kGeneration, kIdentity));
    EXPECT_FALSE(cache.try_read_peer_cached(kAddress, std::span<uint8_t>(result).first(1), kVmid,
                                            kGeneration + 1, kIdentity));
    EXPECT_FALSE(cache.try_read_peer_cached(kAddress, std::span<uint8_t>(result).first(1), kVmid,
                                            kGeneration, kIdentity + 1));
    ASSERT_TRUE(cache.try_fill_peer_cached(kAddress + 1, std::span<const uint8_t>(source).last(1),
                                           kVmid, kGeneration, kIdentity));
    ASSERT_TRUE(cache.try_read_peer_cached(kAddress, result, kVmid, kGeneration, kIdentity));
    EXPECT_EQ(result, source);
  };

  exercise(scalar);
  exercise(vector);
  exercise(l2);
  exercise(memory_side);
  EXPECT_EQ(l2.backing_read_transactions(), 0u);
  EXPECT_EQ(l2.backing_write_transactions(), 0u);
}

TEST(PeerCacheTest, L2PeerProbesRecoverAfterMaintenanceWithoutOrdinaryAccess) {
  using namespace rocjitsu::amdgpu;
  constexpr uint32_t kVmid = 7;
  constexpr uint64_t kGeneration = 19;
  constexpr uint64_t kIdentity = 23;
  constexpr uint64_t kAddress = 0x4004;
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  L2Cache l2("l2", coherence);
  const std::array<uint8_t, 2> first{0x31, 0x72};
  const std::array<uint8_t, 2> second{0x53, 0x94};
  std::array<uint8_t, 2> observed{};

  ASSERT_TRUE(l2.try_fill_peer_cached(kAddress, first, kVmid, kGeneration, kIdentity));
  const uint64_t first_epoch = L2CacheTestAccess::recorded_epoch(l2);
  { auto lease = coherence->acquire_cache_maintenance(DeviceCacheOperation::Invalidate); }
  ASSERT_GT(coherence->current_epoch(), first_epoch);
  EXPECT_EQ(L2CacheTestAccess::recorded_epoch(l2), first_epoch);
  EXPECT_FALSE(l2.try_read_peer_cached(kAddress, observed, kVmid, kGeneration, kIdentity));
  EXPECT_EQ(L2CacheTestAccess::recorded_epoch(l2), coherence->current_epoch());
  ASSERT_TRUE(l2.try_fill_peer_cached(kAddress, first, kVmid, kGeneration, kIdentity));

  { auto lease = coherence->acquire_cache_maintenance(DeviceCacheOperation::Invalidate); }
  EXPECT_NE(L2CacheTestAccess::recorded_epoch(l2), coherence->current_epoch());
  ASSERT_TRUE(l2.try_fill_peer_cached(kAddress, second, kVmid, kGeneration, kIdentity));
  ASSERT_TRUE(l2.try_read_peer_cached(kAddress, observed, kVmid, kGeneration, kIdentity));
  EXPECT_EQ(observed, second);
  EXPECT_EQ(l2.backing_read_transactions(), 0u);
}

TEST(ControlledCacheBackingTest, DirectL2AndMemorySideSinksRetainDirtyBytesUntilRetirement) {
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint32_t kOffset = 16;
  auto translator = std::make_shared<IdentityAddressSpace>(0x3000);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(0x3000);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm vm;
  const auto handle = vm.register_translated(kVmid, translator, physical);
  ASSERT_TRUE(handle);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  ASSERT_TRUE(vm.protect_controlled_cache_backing(handle, kAddress, L2Cache::LINE_SIZE));

  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&vm);
  l2.set_legacy_maintenance_memory(&memory);
  l2.set_legacy_maintenance_vm(&vm);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0xa5);
  EXPECT_EQ(l2.write(kAddress + kOffset, dirty.data(), 4, Mtype::RW, kVmid),
            VmAccessOutcome::Unavailable);
  EXPECT_FALSE(L2CacheTestAccess::resident(l2, kAddress, kVmid));
  ASSERT_EQ(l2.writeback_line(kAddress, dirty.data(), kOffset, 4, Mtype::RW, kVmid),
            VmAccessOutcome::Complete);
  const uint64_t transactions = l2.backing_write_transactions();
  EXPECT_EQ(l2.flush_line(kAddress, kVmid), VmAccessOutcome::Unavailable);
  EXPECT_EQ(l2.invalidate_range(kAddress + kOffset, 4, kVmid), VmAccessOutcome::Unavailable);
  EXPECT_EQ(l2.flush_range_before_external_write(kAddress + kOffset, 4, kVmid),
            VmAccessOutcome::Unavailable);
  EXPECT_EQ(l2.flush_all(), VmAccessOutcome::Unavailable);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, kOffset));
  EXPECT_EQ(l2.backing_write_transactions(), transactions);
  EXPECT_EQ(physical->write_calls, 0u);

  MemorySideCache memory_side("msc");
  memory_side.set_gpu_vm(&vm);
  memory_side.set_legacy_maintenance_vm(&vm);
  memory_side.set_legacy_maintenance_memory(&memory);
  EXPECT_EQ(memory_side.write(kAddress + kOffset, dirty.data(), 4, kVmid),
            VmAccessOutcome::Unavailable);
  MemorySideCacheTestAccess::store_prepared_dirty_line(memory_side, kAddress, dirty, kVmid,
                                                       access->cache_identity(), 0);
  EXPECT_EQ(memory_side.flush_range(kAddress + kOffset, 4, kVmid), VmAccessOutcome::Unavailable);
  EXPECT_EQ(memory_side.flush_all(), VmAccessOutcome::Unavailable);
  EXPECT_TRUE(MemorySideCacheTestAccess::line_is_dirty(memory_side, kAddress, kVmid));
  EXPECT_EQ(physical->write_calls, 0u);

  vm.retire_controlled_cache_backing(handle, kAddress, L2Cache::LINE_SIZE);
  EXPECT_EQ(l2.flush_line(kAddress, kVmid), VmAccessOutcome::Complete);
  EXPECT_EQ(memory_side.flush_all(), VmAccessOutcome::Complete);
  EXPECT_EQ(physical->load<uint32_t>(kAddress + kOffset), 0xa5a5a5a5u);
}

TEST(ControlledCacheBackingTest, L2PreflightChecksDirtyBytesAtPartialRangeBoundary) {
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  auto translator = std::make_shared<IdentityAddressSpace>(0x3000);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(0x3000);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm vm;
  const auto handle = vm.register_translated(kVmid, translator, physical);
  ASSERT_TRUE(handle);
  ASSERT_TRUE(vm.protect_controlled_cache_backing(handle, kAddress + 64, 4));
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&vm);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  ASSERT_EQ(l2.writeback_line(kAddress, dirty.data(), 8, 4, Mtype::RW, kVmid),
            VmAccessOutcome::Complete);
  EXPECT_EQ(l2.flush_line(kAddress, kVmid), VmAccessOutcome::Complete);
  EXPECT_EQ(physical->load<uint32_t>(kAddress + 8), 0x5a5a5a5au);
  ASSERT_EQ(l2.writeback_line(kAddress, dirty.data(), 64, 4, Mtype::RW, kVmid),
            VmAccessOutcome::Complete);
  EXPECT_EQ(l2.invalidate_range(kAddress + 64, 4, kVmid), VmAccessOutcome::Unavailable);
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 64));
  EXPECT_EQ(physical->load<uint32_t>(kAddress + 64), 0u);
}

TEST(ControlledCacheBackingTest, UnregisterDropsOldVmidPolicyBeforeBindingReuse) {
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  auto translator = std::make_shared<IdentityAddressSpace>(0x3000);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(0x3000);
  rocjitsu::amdgpu::GpuVm vm;
  const auto old_handle = vm.register_translated(kVmid, translator, physical);
  ASSERT_TRUE(old_handle);
  ASSERT_TRUE(vm.protect_controlled_cache_backing(old_handle, kAddress, L2Cache::LINE_SIZE));
  ASSERT_TRUE(vm.controlled_cache_backing_overlaps(kVmid, kAddress, 4));
  ASSERT_TRUE(vm.unregister_address_space(old_handle));
  EXPECT_FALSE(vm.controlled_cache_backing_overlaps(kVmid, kAddress, 4));
  const auto new_handle = vm.register_translated(kVmid, translator, physical);
  ASSERT_TRUE(new_handle);
  EXPECT_NE(new_handle, old_handle);
  EXPECT_FALSE(vm.controlled_cache_backing_overlaps(kVmid, kAddress, 4));
}

TEST(ControlledCacheBackingTest, ConcurrentReadersSurvivePolicyRetirementAndVmidReuse) {
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  auto translator = std::make_shared<IdentityAddressSpace>(0x3000);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(0x3000);
  rocjitsu::amdgpu::GpuVm vm;
  auto handle = vm.register_translated(kVmid, translator, physical);
  ASSERT_TRUE(handle);
  std::atomic<bool> done{false};
  std::atomic<uint64_t> probes{0};
  std::thread reader([&] {
    while (!done.load(std::memory_order_acquire)) {
      (void)vm.controlled_cache_backing_overlaps(kVmid, kAddress, 4);
      probes.fetch_add(1, std::memory_order_relaxed);
    }
  });
  while (probes.load(std::memory_order_acquire) == 0)
    std::this_thread::yield();
  for (int attempt = 0; attempt < 100; ++attempt) {
    EXPECT_TRUE(vm.protect_controlled_cache_backing(handle, kAddress, 4));
    vm.retire_controlled_cache_backing(handle, kAddress, 4);
    EXPECT_TRUE(vm.unregister_address_space(handle));
    handle = vm.register_translated(kVmid, translator, physical);
    EXPECT_TRUE(handle);
  }
  done.store(true, std::memory_order_release);
  reader.join();
  EXPECT_GT(probes.load(std::memory_order_relaxed), 0u);
  EXPECT_FALSE(vm.controlled_cache_backing_overlaps(kVmid, kAddress, 4));
}

TEST(ControlledCacheBackingTest, DirtyVictimEvictionRetainsLineAndReplacementOrder) {
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kGeneration = 19;
  constexpr uint64_t kSetSpan = static_cast<uint64_t>(L2Cache::NUM_SETS) * L2Cache::LINE_SIZE;
  constexpr uint64_t kCandidate = kAddress + L2Cache::ASSOCIATIVITY * kSetSpan;
  auto translator = std::make_shared<IdentityAddressSpace>(kCandidate + L2Cache::LINE_SIZE,
                                                           Mtype::RW, kGeneration);
  auto physical = std::make_shared<ConfigurablePhysicalMemory>(kCandidate + L2Cache::LINE_SIZE);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm vm;
  const auto handle = vm.register_translated(kVmid, translator, physical);
  ASSERT_TRUE(handle);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  ASSERT_TRUE(vm.protect_controlled_cache_backing(handle, kAddress, L2Cache::LINE_SIZE));
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  l2.set_gpu_vm(&vm);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty[16] = 0xa5;
  ASSERT_EQ(l2.writeback_line(kAddress, dirty.data(), 16, 1, Mtype::RW, kVmid),
            VmAccessOutcome::Complete);
  const std::array<uint8_t, 1> clean{0x31};
  for (uint32_t way = 1; way < L2Cache::ASSOCIATIVITY; ++way)
    ASSERT_TRUE(l2.try_fill_peer_cached(kAddress + way * kSetSpan, clean, kVmid, kGeneration,
                                        access->cache_identity()));
  ASSERT_FALSE(
      l2.try_fill_peer_cached(kCandidate, clean, kVmid, kGeneration, access->cache_identity()));
  uint8_t observed = 0;
  for (int retry = 0; retry < 2; ++retry) {
    EXPECT_EQ(l2.read(kCandidate, &observed, 1, Mtype::RW, kVmid), VmAccessOutcome::Unavailable);
    EXPECT_TRUE(L2CacheTestAccess::resident(l2, kAddress, kVmid));
    EXPECT_FALSE(L2CacheTestAccess::resident(l2, kCandidate, kVmid));
    EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 16));
    EXPECT_EQ(physical->write_calls, 0u);
  }
  vm.retire_controlled_cache_backing(handle, kAddress, L2Cache::LINE_SIZE);
  EXPECT_EQ(l2.read(kCandidate, &observed, 1, Mtype::RW, kVmid), VmAccessOutcome::Complete);
  EXPECT_EQ(physical->load<uint8_t>(kAddress + 16), 0xa5);
}

TEST(ControlledCacheBackingTest, MemorySideDirtyVictimRefusalKeepsReplacementOrder) {
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kGeneration = 19;
  constexpr uint64_t kSetSpan =
      static_cast<uint64_t>(MemorySideCache::NUM_SETS) * MemorySideCache::LINE_SIZE;
  constexpr uint64_t kCandidate = kAddress + MemorySideCache::ASSOCIATIVITY * kSetSpan;
  auto translator = std::make_shared<IdentityAddressSpace>(kCandidate + MemorySideCache::LINE_SIZE,
                                                           Mtype::RW, kGeneration);
  auto physical =
      std::make_shared<ConfigurablePhysicalMemory>(kCandidate + MemorySideCache::LINE_SIZE);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm vm;
  HbmController hbm("hbm", &memory, &vm);
  MemorySideCache memory_side("msc");
  simdojo::Link link(/*id=*/0, memory_side.req_port(), hbm.cpl_port(), /*latency=*/0);
  link.set_exec_mode(simdojo::ExecMode::FUNCTIONAL);
  memory_side.req_port()->set_link(&link);
  hbm.cpl_port()->set_link(&link);
  const auto handle = vm.register_translated(kVmid, translator, physical);
  ASSERT_TRUE(handle);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  ASSERT_TRUE(vm.protect_controlled_cache_backing(handle, kAddress, MemorySideCache::LINE_SIZE));
  memory_side.set_gpu_vm(&vm);
  memory_side.set_legacy_maintenance_vm(&vm);
  memory_side.set_legacy_maintenance_memory(&memory);
  std::array<uint8_t, MemorySideCache::LINE_SIZE> dirty{};
  dirty.fill(0x6b);
  MemorySideCacheTestAccess::store_prepared_dirty_line(memory_side, kAddress, dirty, kVmid,
                                                       access->cache_identity(), kGeneration);
  const std::array<uint8_t, 1> clean{0x31};
  for (uint32_t way = 1; way < MemorySideCache::ASSOCIATIVITY; ++way)
    ASSERT_TRUE(memory_side.try_fill_peer_cached(kAddress + way * kSetSpan, clean, kVmid,
                                                 kGeneration, access->cache_identity()));
  ASSERT_FALSE(memory_side.try_fill_peer_cached(kCandidate, clean, kVmid, kGeneration,
                                                access->cache_identity()));
  uint8_t observed = 0;
  for (int retry = 0; retry < 2; ++retry) {
    EXPECT_EQ(memory_side.read(kCandidate, &observed, 1, kVmid), VmAccessOutcome::Unavailable);
    EXPECT_TRUE(MemorySideCacheTestAccess::line_is_resident(memory_side, kAddress, kVmid));
    EXPECT_FALSE(MemorySideCacheTestAccess::line_is_resident(memory_side, kCandidate, kVmid));
    EXPECT_EQ(physical->write_calls, 0u);
  }
  EXPECT_TRUE(MemorySideCacheTestAccess::line_is_dirty(memory_side, kAddress, kVmid));
  vm.retire_controlled_cache_backing(handle, kAddress, MemorySideCache::LINE_SIZE);
  EXPECT_EQ(memory_side.read(kCandidate, &observed, 1, kVmid), VmAccessOutcome::Complete);
  EXPECT_EQ(physical->load<uint8_t>(kAddress), 0x6b);
}

TEST(ControlledCacheBackingTest, StaleDirtyVictimsDoNotStallControlledBacking) {
  using rocjitsu::amdgpu::L2CacheTestAccess;
  using rocjitsu::amdgpu::MemorySideCacheTestAccess;
  using rocjitsu::amdgpu::VmAccessOutcome;
  constexpr uint32_t kVmid = 73;
  constexpr uint64_t kAddress = 0x1000;
  constexpr uint64_t kGeneration = 19;
  constexpr uint64_t kL2SetSpan = static_cast<uint64_t>(L2Cache::NUM_SETS) * L2Cache::LINE_SIZE;
  constexpr uint64_t kMscSetSpan =
      static_cast<uint64_t>(MemorySideCache::NUM_SETS) * MemorySideCache::LINE_SIZE;
  constexpr uint64_t kL2Candidate = kAddress + L2Cache::ASSOCIATIVITY * kL2SetSpan;
  constexpr uint64_t kMscCandidate = kAddress + MemorySideCache::ASSOCIATIVITY * kMscSetSpan;
  auto translator = std::make_shared<IdentityAddressSpace>(
      kMscCandidate + MemorySideCache::LINE_SIZE, Mtype::RW, kGeneration);
  auto physical =
      std::make_shared<ConfigurablePhysicalMemory>(kMscCandidate + MemorySideCache::LINE_SIZE);
  GpuMemory memory("memory");
  rocjitsu::amdgpu::GpuVm vm;
  const auto handle = vm.register_translated(kVmid, translator, physical);
  ASSERT_TRUE(handle);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  ASSERT_TRUE(vm.protect_controlled_cache_backing(handle, kAddress, L2Cache::LINE_SIZE));

  for (bool stale_identity : {true, false}) {
    const uint64_t identity =
        stale_identity ? access->cache_identity() + 1 : access->cache_identity();
    const uint64_t generation = stale_identity ? kGeneration : kGeneration - 1;
    L2Cache l2("l2");
    l2.set_backing_memory(&memory);
    l2.set_gpu_vm(&vm);
    std::array<uint8_t, L2Cache::LINE_SIZE> l2_dirty{};
    l2_dirty[16] = 0xa5;
    constexpr std::array<std::pair<uint32_t, uint32_t>, 1> kDirtyRanges{{{16, 1}}};
    L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, l2_dirty, kVmid, identity,
                                                 generation, kDirtyRanges);
    const std::array<uint8_t, 1> clean{0x31};
    for (uint32_t way = 1; way < L2Cache::ASSOCIATIVITY; ++way)
      ASSERT_TRUE(l2.try_fill_peer_cached(kAddress + way * kL2SetSpan, clean, kVmid, kGeneration,
                                          access->cache_identity()));
    uint8_t observed = 0;
    EXPECT_EQ(l2.read(kL2Candidate, &observed, 1, Mtype::RW, kVmid), VmAccessOutcome::Complete);
    EXPECT_FALSE(L2CacheTestAccess::resident(l2, kAddress, kVmid));

    HbmController hbm("hbm", &memory, &vm);
    MemorySideCache memory_side("msc");
    simdojo::Link link(/*id=*/0, memory_side.req_port(), hbm.cpl_port(), /*latency=*/0);
    link.set_exec_mode(simdojo::ExecMode::FUNCTIONAL);
    memory_side.req_port()->set_link(&link);
    hbm.cpl_port()->set_link(&link);
    memory_side.set_gpu_vm(&vm);
    std::array<uint8_t, MemorySideCache::LINE_SIZE> msc_dirty{};
    msc_dirty.fill(0x6b);
    MemorySideCacheTestAccess::store_prepared_dirty_line(memory_side, kAddress, msc_dirty, kVmid,
                                                         identity, generation);
    for (uint32_t way = 1; way < MemorySideCache::ASSOCIATIVITY; ++way)
      ASSERT_TRUE(memory_side.try_fill_peer_cached(kAddress + way * kMscSetSpan, clean, kVmid,
                                                   kGeneration, access->cache_identity()));
    EXPECT_EQ(memory_side.read(kMscCandidate, &observed, 1, kVmid), VmAccessOutcome::Complete);
    EXPECT_FALSE(MemorySideCacheTestAccess::line_is_resident(memory_side, kAddress, kVmid));
    EXPECT_EQ(physical->write_calls, 0u);
  }
  EXPECT_TRUE(vm.controlled_cache_backing_overlaps(kVmid, kAddress, 1));
}

TEST(PeerCacheTest, DirtyVictimRefusalPreservesTagAndReplacementOrder) {
  using namespace rocjitsu::amdgpu;
  constexpr uint32_t kVmid = 7;
  constexpr uint64_t kGeneration = 19;
  constexpr uint64_t kIdentity = 23;
  constexpr uint64_t kAddress = 0x4000;
  constexpr uint64_t kL2SetSpan = static_cast<uint64_t>(L2Cache::NUM_SETS) * L2Cache::LINE_SIZE;
  constexpr uint64_t kMemorySideSetSpan =
      static_cast<uint64_t>(MemorySideCache::NUM_SETS) * MemorySideCache::LINE_SIZE;
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0x5a);
  constexpr std::array<std::pair<uint32_t, uint32_t>, 1> kDirtyRanges{{{0, L2Cache::LINE_SIZE}}};
  const std::array<uint8_t, 1> clean{0x33};
  std::array<uint8_t, 1> result{};

  L2Cache l2("l2");
  L2CacheTestAccess::store_prepared_dirty_line(l2, kAddress, dirty, kVmid, kIdentity, kGeneration,
                                               kDirtyRanges);
  EXPECT_FALSE(l2.try_fill_peer_cached(kAddress, clean, kVmid, kGeneration, kIdentity));
  for (uint32_t way = 1; way < L2Cache::ASSOCIATIVITY; ++way)
    ASSERT_TRUE(
        l2.try_fill_peer_cached(kAddress + way * kL2SetSpan, clean, kVmid, kGeneration, kIdentity));
  const uint64_t l2_candidate = kAddress + L2Cache::ASSOCIATIVITY * kL2SetSpan;
  EXPECT_FALSE(l2.try_fill_peer_cached(l2_candidate, clean, kVmid, kGeneration, kIdentity));
  EXPECT_FALSE(l2.try_fill_peer_cached(l2_candidate, clean, kVmid, kGeneration, kIdentity));
  EXPECT_FALSE(L2CacheTestAccess::resident(l2, l2_candidate, kVmid));
  EXPECT_TRUE(L2CacheTestAccess::dirty_byte(l2, kAddress, kVmid, 0));
  ASSERT_TRUE(l2.try_read_peer_cached(kAddress, result, kVmid, kGeneration, kIdentity));
  EXPECT_EQ(result[0], dirty[0]);
  EXPECT_EQ(l2.backing_write_transactions(), 0u);

  MemorySideCache memory_side("msc");
  MemorySideCacheTestAccess::store_prepared_dirty_line(memory_side, kAddress, dirty, kVmid,
                                                       kIdentity, kGeneration);
  EXPECT_FALSE(memory_side.try_fill_peer_cached(kAddress, clean, kVmid, kGeneration, kIdentity));
  for (uint32_t way = 1; way < MemorySideCache::ASSOCIATIVITY; ++way)
    ASSERT_TRUE(memory_side.try_fill_peer_cached(kAddress + way * kMemorySideSetSpan, clean, kVmid,
                                                 kGeneration, kIdentity));
  const uint64_t memory_side_candidate =
      kAddress + MemorySideCache::ASSOCIATIVITY * kMemorySideSetSpan;
  EXPECT_FALSE(memory_side.try_fill_peer_cached(memory_side_candidate, clean, kVmid, kGeneration,
                                                kIdentity));
  EXPECT_FALSE(memory_side.try_fill_peer_cached(memory_side_candidate, clean, kVmid, kGeneration,
                                                kIdentity));
  EXPECT_FALSE(
      MemorySideCacheTestAccess::line_is_resident(memory_side, memory_side_candidate, kVmid));
  EXPECT_TRUE(MemorySideCacheTestAccess::line_is_dirty(memory_side, kAddress, kVmid));
  ASSERT_TRUE(memory_side.try_read_peer_cached(kAddress, result, kVmid, kGeneration, kIdentity));
  EXPECT_EQ(result[0], dirty[0]);
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

TEST(TranslatedLocalCache, ExactHandlePtePolicyAndFabricBoundaryRefetchOwnerData) {
  using namespace rocjitsu::amdgpu;
  class MutableLocalTranslator final : public AddressSpaceTranslator {
  public:
    VmTranslationResult translate(uint64_t address, std::size_t size,
                                  VmAccessKind access) const override {
      if (address >= 4096 || size == 0 || size > 4096 - address ||
          (access != VmAccessKind::Read && access != VmAccessKind::Write))
        return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
      return {.outcome = VmAccessOutcome::Complete,
              .translation = {.domain = VmMemoryDomain::Local,
                              .address = backing_base + address,
                              .contiguous_bytes = 4096 - address,
                              .mtype = Mtype::RW,
                              .gfx12_pte_mtype = raw_type,
                              .permissions = {.readable = true, .writable = true}}};
    }
    uint64_t backing_base = 0x1000;
    Gfx12PteMtype raw_type = Gfx12PteMtype::NC;
  };
  class LocalBacking final : public PhysicalMemoryAccess {
  public:
    explicit LocalBacking(std::shared_ptr<DeviceCacheCoherence> domain)
        : domain_(std::move(domain)) {}
    bool supports_translated_local_cache() const override { return true; }
    MemoryIngressAdmission admit_pt_rewalk() const override {
      return {.outcome = transport_open ? VmAccessOutcome::Complete : VmAccessOutcome::Revoked,
              .permit = {}};
    }
    VmAccessOutcome read(VmMemoryDomain domain, uint64_t address,
                         std::span<std::byte> result) override {
      if (domain != VmMemoryDomain::Local || address > bytes_.size() ||
          result.size() > bytes_.size() - address)
        return VmAccessOutcome::Faulted;
      ++reads;
      std::memcpy(result.data(), bytes_.data() + address, result.size());
      return VmAccessOutcome::Complete;
    }
    VmAccessOutcome write(VmMemoryDomain domain, uint64_t address,
                          std::span<const std::byte> source) override {
      if (domain != VmMemoryDomain::Local || address > bytes_.size() ||
          source.size() > bytes_.size() - address)
        return VmAccessOutcome::Faulted;
      std::memcpy(bytes_.data() + address, source.data(), source.size());
      domain_->publish_local_backing_write(address, source.size());
      return VmAccessOutcome::Complete;
    }
    void external_effect(uint64_t address, uint32_t value) {
      ASSERT_LE(address + sizeof(value), bytes_.size());
      std::memcpy(bytes_.data() + address, &value, sizeof(value));
    }
    size_t reads = 0;
    bool transport_open = true;

  private:
    std::shared_ptr<DeviceCacheCoherence> domain_;
    std::array<std::byte, 0x4000> bytes_{};
  };

  auto coherence = std::make_shared<DeviceCacheCoherence>();
  auto backing = std::make_shared<LocalBacking>(coherence);
  auto first_root = std::make_shared<MutableLocalTranslator>();
  auto second_root = std::make_shared<MutableLocalTranslator>();
  second_root->backing_base = 0x2000;
  constexpr uint64_t kVa = 0x80;
  constexpr uint32_t kInitial = 0x13579bdf;
  constexpr uint32_t kOther = 0x2468ace0;
  backing->external_effect(0x1000 + kVa, kInitial);
  backing->external_effect(0x2000 + kVa, kOther);

  GpuVm vm;
  const AddressSpaceHandle first = vm.register_unrouted_address_space(77, first_root, backing);
  const AddressSpaceHandle second = vm.register_unrouted_address_space(77, second_root, backing);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_FALSE(vm.find_vmid(77));
  auto first_access = vm.snapshot(first);
  auto second_access = vm.snapshot(second);
  ASSERT_TRUE(first_access);
  ASSERT_TRUE(second_access);

  GpuMemory legacy_backing("translated-local-legacy-maintenance");
  L2Cache l2("translated-local-l2", coherence);
  l2.set_gpu_vm(&vm);
  l2.set_legacy_maintenance_memory(&legacy_backing);
  l2.set_legacy_maintenance_vm(&vm);
  L1VectorCache vector(&l2);
  vector.set_gpu_vm(&vm);
  const auto load = [&](const GpuVmAccess &access) {
    uint32_t value = 0;
    const VmAccessOutcome outcome = vector.load_translated_local(
        access, kVa, {reinterpret_cast<uint8_t *>(&value), sizeof(value)}, Mtype::RW, false, false);
    EXPECT_EQ(outcome, VmAccessOutcome::Complete);
    return value;
  };

  EXPECT_EQ(load(*first_access), kInitial);
  EXPECT_EQ(load(*first_access), kInitial);
  EXPECT_EQ(backing->reads, 1u) << "the second owner read must hit the modeled V$";
  backing->transport_open = false;
  uint32_t revoked_result = 0;
  EXPECT_EQ(vector.load_translated_local(
                *first_access, kVa,
                {reinterpret_cast<uint8_t *>(&revoked_result), sizeof(revoked_result)}, Mtype::RW,
                false, false),
            VmAccessOutcome::Revoked);
  backing->transport_open = true;
  EXPECT_EQ(backing->reads, 1u) << "a revoked session cannot consume cached bytes";
  EXPECT_EQ(load(*second_access), kOther);
  EXPECT_EQ(load(*first_access), kInitial);
  EXPECT_EQ(backing->reads, 2u) << "equal numeric PASIDs must retain distinct handles";

  first_root->raw_type = Gfx12PteMtype::RW;
  EXPECT_EQ(load(*first_access), kInitial);
  EXPECT_EQ(backing->reads, 3u) << "NC and RW PTE encodings cannot share a cached hit";

  const uint32_t unrelated_write = 0xdecafbad;
  ASSERT_EQ(second_access->write(kVa, std::as_bytes(std::span(&unrelated_write, 1))),
            VmAccessOutcome::Complete);
  EXPECT_EQ(load(*first_access), kInitial);
  EXPECT_EQ(backing->reads, 3u)
      << "a write to another physical line must preserve the owner cache hit";
  EXPECT_EQ(load(*second_access), unrelated_write);
  EXPECT_EQ(backing->reads, 4u);

  const uint32_t local_write = 0x10203040;
  ASSERT_EQ(first_access->write(kVa, std::as_bytes(std::span(&local_write, 1))),
            VmAccessOutcome::Complete);
  EXPECT_EQ(load(*first_access), local_write);
  EXPECT_EQ(backing->reads, 5u);

  // Fabric maintenance holds both domains through the canonical owner effect;
  // its release retires the cached owner value before a later CU load.
  const std::array domains{coherence};
  {
    auto leases = DeviceCacheCoherence::acquire_cache_maintenance_group(
        domains, DeviceCacheOperation::WritebackInvalidate);
    ASSERT_EQ(leases.size(), 1u);
    backing->external_effect(0x1000 + kVa, 0x55667788);
  }
  EXPECT_EQ(load(*first_access), 0x55667788u);
  EXPECT_EQ(backing->reads, 6u);

  ASSERT_TRUE(vm.invalidate(first));
  auto refreshed = vm.snapshot(first);
  ASSERT_TRUE(refreshed);
  EXPECT_NE(refreshed->cache_namespace(), first_access->cache_namespace());
  EXPECT_EQ(load(*refreshed), 0x55667788u);
  EXPECT_EQ(backing->reads, 7u);

  // V$ completion alone exposes the retained L2 line. Both completion
  // operations must retire the translated key before a later backing read.
  backing->external_effect(0x1000 + kVa, 0x89abcdef);
  vector.invalidate_for_completion(refreshed->info().vmid, refreshed->cache_identity());
  EXPECT_EQ(load(*refreshed), 0x55667788u);
  EXPECT_EQ(backing->reads, 7u);
  ASSERT_EQ(l2.flush_for_completion(*refreshed), VmAccessOutcome::Complete);
  vector.invalidate_for_completion(refreshed->info().vmid, refreshed->cache_identity());
  EXPECT_EQ(load(*refreshed), 0x89abcdefu);
  EXPECT_EQ(backing->reads, 8u);
}

} // namespace
