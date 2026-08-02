// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_memory.h
/// @brief AMDGPU VRAM memory with per-process VMID-based page table resolution.

#ifndef ROCJITSU_VM_AMDGPU_GPU_MEMORY_H_
#define ROCJITSU_VM_AMDGPU_GPU_MEMORY_H_

#include "rocjitsu/kmd/linux/kfd_process.h"
#include "simdojo/components/sparse_memory.h"
#include "simdojo/sim/component.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <sys/uio.h>
#include <type_traits>
#include <unordered_map>
#include <utility>

#if defined(__SANITIZE_ADDRESS__)
#define RJ_GPU_MEMORY_WITH_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define RJ_GPU_MEMORY_WITH_ASAN 1
#endif
#endif

#if defined(RJ_GPU_MEMORY_WITH_ASAN)
#include <sanitizer/asan_interface.h>
#endif

namespace rocjitsu {
namespace amdgpu {

class GpuMemoryTestAccess;

/// @brief AMDGPU VRAM memory with VMID-based per-process page table resolution.
///
/// @details Mirrors the GFXHUB's VMID register file. Each process registers its
/// page table via register_process(). Every memory access carries an explicit
/// vmid parameter that selects the page table for VA-to-host translation,
/// matching real hardware where the VMID travels with each request from the
/// issuing wave through the memory hierarchy.
class GpuMemory : public simdojo::SparseMemory {
public:
  explicit GpuMemory(std::string name)
      : simdojo::SparseMemory(std::move(name)),
        // Function-static TLS translation caches may outlive a GpuMemory on a
        // long-lived host thread. A lifetime token prevents one of those caches
        // from matching an object later reconstructed at the same address.
        instance_id_(next_instance_id_.fetch_add(1, std::memory_order_relaxed)) {
    cpl_ = add_port(std::make_unique<simdojo::Port>("cpl", 0, this, simdojo::PortDirection::IN,
                                                    simdojo::PortProtocol::MEMORY));
    cpl_->recv_event()->set_handler([this](simdojo::Tick, simdojo::Message *msg) {
      auto &hdr = msg->header();
      auto *data = reinterpret_cast<uint8_t *>(msg->payload());
      if (hdr.op == simdojo::MessageOp::READ) {
        read_block(hdr.addr, std::span<uint8_t>(data, hdr.size_bytes), hdr.vmid);
      } else if (hdr.op == simdojo::MessageOp::WRITE) {
        write_block(hdr.addr, std::span<const uint8_t>(data, hdr.size_bytes), hdr.vmid);
      }
      hdr.op = simdojo::MessageOp::RESPONSE;
    });
  }

  simdojo::Port *cpl_port() { return cpl_; }

  /// @brief Register a process's page table in the VMID table.
  /// @param generation Optional mutation counter used by translation caches.
  ///        Omitting it disables the per-thread fast path for this page table.
  void register_process(uint32_t pid, KfdProcess::PageTable *pt, std::shared_mutex *mu,
                        const uint64_t *generation = nullptr) {
    util::Logger::cp("VMID_REG pid=", pid, " mem=0x", std::hex, reinterpret_cast<uintptr_t>(this),
                     std::dec, " pt_size=", pt->size());
    std::unique_lock lk(vmid_mutex_);
    vmid_table_[pid] = {
        .page_table = pt,
        .mutex = mu,
        .client_pid = 0,
        .generation = generation,
    };
    // The VMID may now select a different page table even though neither page
    // table changed. Invalidate TLS entries that cached the old registration.
    ++vmid_registry_generation_;
  }

  /// @brief Unregister a process from the VMID table.
  void unregister_process(uint32_t pid) {
    util::Logger::cp("VMID_UNREG pid=", pid, " mem=0x", std::hex, reinterpret_cast<uintptr_t>(this),
                     std::dec);
    std::unique_lock lk(vmid_mutex_);
    auto it = vmid_table_.find(pid);
    if (it == vmid_table_.end())
      return;
    vmid_table_.erase(it);
    ++vmid_registry_generation_;
  }

  void set_process_client_pid(uint32_t pid, pid_t client_pid) {
    std::unique_lock lk(vmid_mutex_);
    auto it = vmid_table_.find(pid);
    if (it != vmid_table_.end())
      it->second.client_pid = client_pid;
  }

  /// @brief Enable passthrough for unmapped addresses (local/user-mode only).
  /// @details When true, addresses not found in the page table are treated as
  /// host pointers (GPU VA == host VA). This mirrors QEMU user-mode's identity
  /// mapping and is only valid when simulator and target share an address space.
  void set_passthrough(bool v) { passthrough_ = v; }

  /// @brief Return whether the page containing an address has a known mapping.
  /// @details Unlike resolve_host_ptr(), this deliberately ignores the current
  /// host accessibility of the requested byte. Callers use it to distinguish a
  /// known mapping whose live extent is clipped from a page that may not have
  /// been installed yet.
  bool has_page_mapping(uint64_t addr, uint32_t vmid = 0) const {
    if (vmid == 0)
      return passthrough_ && addr < kUserSpaceLimit && addr != 0;

    std::shared_lock vmid_lock(vmid_mutex_);
    auto vmid_entry = vmid_table_.find(vmid);
    if (vmid_entry == vmid_table_.end())
      return passthrough_ && addr < kUserSpaceLimit && addr != 0;

    auto &entry = vmid_entry->second;
    std::shared_lock page_table_lock(*entry.mutex);
    if (entry.page_table->contains(addr >> PAGE_SHIFT))
      return true;
    return passthrough_ && addr < kUserSpaceLimit && addr != 0;
  }

  /// @brief Return whether every page touched by an address range is known.
  /// @details This checks page-table presence rather than live host extents, so
  /// callers can retry a not-yet-installed range while still allowing a mapped
  /// page's deliberately clipped extent to produce bounded partial accesses.
  bool has_range_mapping(uint64_t addr, size_t size, uint32_t vmid = 0) const {
    if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - addr)
      return false;
    bool mapped = true;
    for_each_page_chunk(addr, size, [&](uint64_t ea, size_t, size_t) {
      if (mapped && !has_page_mapping(ea, vmid))
        mapped = false;
    });
    return mapped;
  }

  /// @brief Resolve a GPU VA range to its first borrowed host byte.
  /// @details The returned pointer is only valid while page-table remapping and
  /// process teardown are quiesced. Normal memory operations use an internal
  /// callback that keeps the page-table shared lock held through the copy.
  uint8_t *resolve_host_ptr(uint64_t addr, uint32_t vmid = 0, size_t size = 1) const {
    if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - addr)
      return nullptr;
    uint8_t *first_host_ptr = nullptr;
    bool contiguous = true;
    for_each_page_chunk(addr, size, [&](uint64_t ea, size_t offset, size_t chunk) {
      if (!contiguous)
        return;
      auto *host_ptr = translate(ea, vmid, chunk);
      if (!host_ptr || (first_host_ptr && host_ptr != first_host_ptr + offset)) {
        contiguous = false;
        return;
      }
      if (!first_host_ptr)
        first_host_ptr = host_ptr;
    });
    return contiguous ? first_host_ptr : nullptr;
  }

  /// @brief Look up PTE MTYPE for a GPU VA in the given VMID's page table.
  Mtype pte_mtype(uint64_t addr, uint32_t vmid = 0) const {
    if (vmid == 0)
      return Mtype::RW;
    static thread_local PteCache cache;
    return cached_walk(addr, vmid, cache, [](const KfdProcess::PageTableEntry *pte) {
      return pte ? pte->mtype : Mtype::RW;
    });
  }

  uint32_t fetch32(uint64_t addr, uint32_t vmid = 0) const { return read32(addr, vmid); }

  /// @brief Read a contiguous range from simulated GPU memory.
  /// @details Handles each page through mapped host memory, client memory, or
  /// sparse backing memory. A mapped access clipped by a host extent remains
  /// zero-filled and emits a VM diagnostic so a future strict-fault mode can
  /// reuse the same boundary detection.
  void read_block(uint64_t addr, std::span<uint8_t> dst, uint32_t vmid = 0) const {
    for_each_page_chunk(addr, dst.size(), [&](uint64_t ea, size_t offset, size_t chunk) {
      auto out = dst.subspan(offset, chunk);
      if (read_mapped(ea, out.data(), chunk, vmid))
        return;
      if (vmid > 0 && read_client_memory(ea, out.data(), chunk, vmid))
        return;
      for (size_t i = 0; i < chunk; ++i)
        out[i] = simdojo::SparseMemory::read8(ea + i);
    });
  }

  /// @brief Write a contiguous range to simulated GPU memory.
  /// @details Handles each page through mapped host memory, client memory, or
  /// sparse backing memory. A mapped access clipped by a host extent is dropped
  /// for the missing bytes and emits a VM diagnostic.
  void write_block(uint64_t addr, std::span<const uint8_t> src, uint32_t vmid = 0) {
    for_each_page_chunk(addr, src.size(), [&](uint64_t ea, size_t offset, size_t chunk) {
      auto in = src.subspan(offset, chunk);
      if (write_mapped(ea, in.data(), chunk, vmid))
        return;
      if (vmid > 0 && write_client_memory(ea, in.data(), chunk, vmid))
        return;
      for (size_t i = 0; i < chunk; ++i)
        simdojo::SparseMemory::write8(ea + i, in[i]);
    });
  }

  /// @brief Perform an atomic read-modify-write on resolved backing storage.
  /// @details Storage classification and the page-table shared lock remain
  /// stable through the callback. Mapped aliases rendezvous on a process-wide
  /// host-address stripe; unmapped sparse/client accesses use an address-space
  /// stripe instead. Bytes outside a mapped host extent read as zero and discard
  /// callback writes, including accesses that straddle the extent boundary.
  /// Such clipping emits a VM diagnostic rather than remaining silent.
  /// @param addr GPU virtual address of the target.
  /// @param size Access size in bytes (4 or 8).
  /// @param fn Callback invoked with a pointer to the target bytes.
  /// @param vmid Process VMID used for address translation.
  template <typename F> void atomic_rmw(uint64_t addr, uint32_t size, F &&fn, uint32_t vmid = 0) {
    assert((size == 4 || size == 8) && (addr & PAGE_MASK) + size <= PAGE_SIZE);

    if (vmid == 0) {
      atomic_rmw_unmapped(addr, size, 0, fn);
      return;
    }

    std::shared_lock vmid_lock(vmid_mutex_);
    auto vmid_entry = vmid_table_.find(vmid);
    if (vmid_entry == vmid_table_.end()) {
      atomic_rmw_unmapped(addr, size, 0, fn);
      return;
    }

    auto &entry = vmid_entry->second;
    std::shared_lock page_table_lock(*entry.mutex);
    const uint64_t page_key = addr >> PAGE_SHIFT;
    auto pte = entry.page_table->find(page_key);
    if (pte != entry.page_table->end()) {
      if (!atomic_rmw_mapped_page(pte->second, addr & PAGE_MASK, size, fn))
        note_clipped_mapped_access("atomic", addr, size, vmid);
      return;
    }

    atomic_rmw_unmapped(addr, size, entry.client_pid, fn);
  }

  uint8_t *translate_debug(uint64_t addr, uint32_t vmid, size_t size = 1) const {
    return translate(addr, vmid, size);
  }

  /// @brief Find the contiguous host range containing a VMID-scoped GPU VA.
  /// @details KFD dispatches use per-process page tables. Kernel-symbol
  /// resolution needs a daemon-accessible host pointer range so it can scan
  /// backward from the kernel descriptor to the loaded ELF header.
  std::pair<uint64_t, uint64_t> find_host_range(uint64_t addr, uint32_t vmid) const {
    if (vmid == 0) {
      auto *host = translate(addr, vmid, 1);
      if (!host)
        return {0, 0};
      auto *page = host - (addr & PAGE_MASK);
      auto [range, size] = addressable_range_containing(page, PAGE_SIZE, host);
      return {reinterpret_cast<uint64_t>(range), size};
    }

    std::shared_lock vmid_lock(vmid_mutex_);
    auto vmid_entry = vmid_table_.find(vmid);
    if (vmid_entry == vmid_table_.end())
      return {0, 0};

    auto &entry = vmid_entry->second;
    std::shared_lock page_table_lock(*entry.mutex);
    const uint64_t page = addr >> PAGE_SHIFT;
    auto page_entry = entry.page_table->find(page);
    if (page_entry == entry.page_table->end())
      return {0, 0};

    const size_t page_offset = addr & PAGE_MASK;
    const auto *current_extent = host_extent_at(page_entry->second, page_offset);
    if (!current_extent)
      return {0, 0};

    uint64_t first_page = page;
    const auto *first_extent = current_extent;
    uint8_t *first_host_byte = first_extent->host_ptr;
    while (first_page > 0 && first_extent->gpu_page_offset == 0) {
      auto previous_page_entry = entry.page_table->find(first_page - 1);
      if (previous_page_entry == entry.page_table->end())
        break;
      const auto *previous_extent = host_extent_ending_at_page(previous_page_entry->second);
      if (!previous_extent ||
          previous_extent->host_ptr + previous_extent->host_backed_bytes != first_host_byte)
        break;
      --first_page;
      first_extent = previous_extent;
      first_host_byte = first_extent->host_ptr;
    }

    uint64_t last_page = page;
    const auto *last_extent = current_extent;
    while (last_extent->gpu_page_offset + last_extent->host_backed_bytes == PAGE_SIZE) {
      auto next_page_entry = entry.page_table->find(last_page + 1);
      if (next_page_entry == entry.page_table->end())
        break;
      const auto *next_extent = host_extent_starting_at_page(next_page_entry->second);
      if (!next_extent ||
          next_extent->host_ptr != last_extent->host_ptr + last_extent->host_backed_bytes)
        break;
      ++last_page;
      last_extent = next_extent;
    }

    const uintptr_t first_host_address = reinterpret_cast<uintptr_t>(first_host_byte);
    const uintptr_t last_host_address = reinterpret_cast<uintptr_t>(last_extent->host_ptr);
    const uint64_t declared_range_size =
        last_host_address - first_host_address + last_extent->host_backed_bytes;
    auto *host_byte = current_extent->host_ptr + (page_offset - current_extent->gpu_page_offset);
    auto [range, range_size] =
        addressable_range_containing(first_host_byte, declared_range_size, host_byte);
    return {reinterpret_cast<uint64_t>(range), range_size};
  }

  std::string debug_page_table_info(uint32_t vmid, uint64_t page_key) const {
    std::shared_lock lk(vmid_mutex_);
    auto it = vmid_table_.find(vmid);
    if (it == vmid_table_.end())
      return "vmid_not_found";
    auto &entry = it->second;
    std::shared_lock pt_lk(*entry.mutex);
    auto pt_it = entry.page_table->find(page_key);
    if (pt_it != entry.page_table->end())
      return "page_found";
    std::string result = "page_missing pt_size=" + std::to_string(entry.page_table->size());
    uint64_t lo = std::numeric_limits<uint64_t>::max(), hi = 0;
    for (auto &[k, v] : *entry.page_table) {
      if (k < lo)
        lo = k;
      if (k > hi)
        hi = k;
    }
    result += " range=[0x" + std::format("{:x}", lo) + ",0x" + std::format("{:x}", hi) + "]";
    return result;
  }

  uint8_t read8(uint64_t addr, uint32_t vmid = 0) const {
    uint8_t val = 0;
    if (read_mapped(addr, &val, sizeof(val), vmid))
      return val;
    if (vmid > 0 && read_client_memory(addr, &val, 1, vmid))
      return val;
    return SparseMemory::read8(addr);
  }

  uint16_t read16(uint64_t addr, uint32_t vmid = 0) const {
    uint16_t val = 0;
    if (read_mapped(addr, &val, sizeof(val), vmid))
      return val;
    if (vmid > 0 && read_client_memory(addr, &val, 2, vmid))
      return val;
    return SparseMemory::read16(addr);
  }

  uint32_t read32(uint64_t addr, uint32_t vmid = 0) const {
    uint32_t val = 0;
    if (read_mapped(addr, &val, sizeof(val), vmid))
      return val;
    if (vmid > 0 && read_client_memory(addr, &val, 4, vmid))
      return val;
    return SparseMemory::read32(addr);
  }

  uint64_t read64(uint64_t addr, uint32_t vmid = 0) const {
    uint64_t val = 0;
    if (read_mapped(addr, &val, sizeof(val), vmid))
      return val;
    if (vmid > 0 && read_client_memory(addr, &val, 8, vmid))
      return val;
    return SparseMemory::read64(addr);
  }

  void write8(uint64_t addr, uint8_t val, uint32_t vmid = 0) {
    if (write_mapped(addr, &val, sizeof(val), vmid))
      return;
    if (vmid > 0 && write_client_memory(addr, &val, 1, vmid))
      return;
    SparseMemory::write8(addr, val);
  }

  void write16(uint64_t addr, uint16_t val, uint32_t vmid = 0) {
    if (write_mapped(addr, &val, sizeof(val), vmid))
      return;
    if (vmid > 0 && write_client_memory(addr, &val, 2, vmid))
      return;
    SparseMemory::write16(addr, val);
  }

  void write32(uint64_t addr, uint32_t val, uint32_t vmid = 0) {
    if (write_mapped(addr, &val, sizeof(val), vmid))
      return;
    if (vmid > 0 && write_client_memory(addr, &val, 4, vmid))
      return;
    SparseMemory::write32(addr, val);
  }

  void write64(uint64_t addr, uint64_t val, uint32_t vmid = 0) {
    if (write_mapped(addr, &val, sizeof(val), vmid))
      return;
    if (vmid > 0 && write_client_memory(addr, &val, 8, vmid))
      return;
    SparseMemory::write64(addr, val);
  }

private:
  friend class GpuMemoryTestAccess;

  // The largest supported atomic is eight bytes, so discard the three byte
  // offset bits before choosing a lock stripe.
  static constexpr unsigned kBackingAtomicGranuleShift = 3;
  // Fold high address bits into the low stripe-index bits before masking.
  static constexpr unsigned kBackingAtomicHashFoldShift1 = 17;
  static constexpr unsigned kBackingAtomicHashFoldShift2 = 31;
  // Bound mutex storage while keeping collisions low for common GPU workloads.
  static constexpr size_t kBackingAtomicLockStripes = 4096;
  // The 64-bit golden-ratio hash constant scatters adjacent client PIDs.
  static constexpr uintptr_t kClientPidHashSalt = static_cast<uintptr_t>(0x9e3779b97f4a7c15ULL);
  static_assert((kBackingAtomicLockStripes & (kBackingAtomicLockStripes - 1)) == 0,
                "atomic lock stripe count must be a power of two");

  static auto &backing_atomic_mutexes() {
    static std::array<std::mutex, kBackingAtomicLockStripes> mutexes;
    return mutexes;
  }

  static size_t backing_atomic_mutex_index(uintptr_t key) {
    key >>= kBackingAtomicGranuleShift;
    key ^= key >> kBackingAtomicHashFoldShift1;
    key ^= key >> kBackingAtomicHashFoldShift2;
    return key & (kBackingAtomicLockStripes - 1);
  }

  static std::mutex &backing_atomic_mutex_at(size_t index) {
    return backing_atomic_mutexes()[index];
  }

  static std::mutex &backing_atomic_mutex(uintptr_t key) {
    return backing_atomic_mutex_at(backing_atomic_mutex_index(key));
  }

  template <typename F> static void atomic_rmw_mapped(uint8_t *target, F &fn) {
    std::lock_guard lock(backing_atomic_mutex(reinterpret_cast<uintptr_t>(target)));
    fn(target);
  }

  template <typename F>
  void atomic_rmw_unmapped(uint64_t addr, uint32_t size, pid_t client_pid, F &fn) {
    auto *target = reinterpret_cast<uint8_t *>(addr);
    if (passthrough_ && addr < kUserSpaceLimit && size <= kUserSpaceLimit - addr &&
        target != nullptr) {
      if (addressable_prefix(target, size) == size)
        atomic_rmw_mapped(target, fn);
      else
        atomic_rmw_discarded(fn);
      return;
    }
    atomic_rmw_fallback(addr, size, client_pid, fn);
  }

  template <typename F>
  void atomic_rmw_fallback(uint64_t addr, uint32_t size, pid_t client_pid, F &fn) {
    uintptr_t key = static_cast<uintptr_t>(addr ^ (addr >> 32));
    if (client_pid > 0)
      key ^= static_cast<uintptr_t>(client_pid) * kClientPidHashSalt;
    else
      key ^= reinterpret_cast<uintptr_t>(this);

    std::lock_guard lock(backing_atomic_mutex(key));
    std::array<uint8_t, sizeof(uint64_t)> value{};
    const bool client_storage =
        client_pid > 0 && read_client_memory_for_pid(addr, value.data(), size, client_pid);
    if (!client_storage) {
      for (uint32_t i = 0; i < size; ++i)
        value[i] = simdojo::SparseMemory::read8(addr + i);
    }

    fn(value.data());

    if (client_storage) {
      write_client_memory_for_pid(addr, value.data(), size, client_pid);
      return;
    }
    for (uint32_t i = 0; i < size; ++i)
      simdojo::SparseMemory::write8(addr + i, value[i]);
  }

  template <typename F>
  static bool atomic_rmw_mapped_page(const KfdProcess::PageTableEntry &pte, size_t page_offset,
                                     size_t size, F &fn) {
    const auto *extent = host_extent_at(pte, page_offset);
    if (extent && size <= extent->host_backed_bytes - (page_offset - extent->gpu_page_offset)) {
      auto *target = extent->host_ptr + (page_offset - extent->gpu_page_offset);
      if (addressable_prefix(target, size) == size) {
        atomic_rmw_mapped(target, fn);
        return true;
      }
    }

    struct AtomicSpan {
      size_t value_offset = 0;
      uint8_t *host_ptr = nullptr;
      size_t size = 0;
    };
    std::array<AtomicSpan, sizeof(uint64_t)> spans{};
    size_t span_count = 0;
    const size_t mapped_bytes = for_each_mapped_span(
        pte, page_offset, size, [&](size_t value_offset, uint8_t *host_ptr, size_t span_size) {
          assert(span_count < spans.size());
          spans[span_count++] = {value_offset, host_ptr, span_size};
        });
    if (span_count == 0) {
      atomic_rmw_discarded(fn);
      return false;
    }

    std::array<size_t, sizeof(uint64_t)> lock_indices{};
    lock_indices.fill(kBackingAtomicLockStripes);
    for (size_t i = 0; i < span_count; ++i) {
      for (size_t byte = 0; byte < spans[i].size; ++byte) {
        const size_t index =
            backing_atomic_mutex_index(reinterpret_cast<uintptr_t>(spans[i].host_ptr + byte));
        if (std::find(lock_indices.begin(), lock_indices.end(), index) != lock_indices.end())
          continue;
        auto free_slot =
            std::find(lock_indices.begin(), lock_indices.end(), kBackingAtomicLockStripes);
        if (free_slot == lock_indices.end()) {
          atomic_rmw_discarded(fn);
          return false;
        }
        *free_slot = index;
      }
    }
    std::sort(lock_indices.begin(), lock_indices.end());
    std::array<std::unique_lock<std::mutex>, sizeof(uint64_t)> locks;
    size_t lock_count = 0;
    while (lock_count < lock_indices.size() &&
           lock_indices[lock_count] != kBackingAtomicLockStripes) {
      const size_t i = lock_count++;
      locks[i] = std::unique_lock(backing_atomic_mutex_at(lock_indices[i]));
    }

    std::array<uint8_t, sizeof(uint64_t)> value{};
    for (size_t i = 0; i < span_count; ++i)
      std::memcpy(value.data() + spans[i].value_offset, spans[i].host_ptr, spans[i].size);
    fn(value.data());
    for (size_t i = 0; i < span_count; ++i)
      std::memcpy(spans[i].host_ptr, value.data() + spans[i].value_offset, spans[i].size);
    return mapped_bytes == size;
  }

  template <typename F> static void atomic_rmw_discarded(F &fn) {
    std::array<uint8_t, sizeof(uint64_t)> value{};
    fn(value.data());
  }

  template <typename F> static void for_each_page_chunk(uint64_t addr, size_t len, F &&fn) {
    size_t offset = 0;
    while (offset < len) {
      const uint64_t ea = addr + offset;
      const size_t chunk = std::min(len - offset, PAGE_SIZE - (ea & PAGE_MASK));
      fn(ea, offset, chunk);
      offset += chunk;
    }
  }

  static constexpr uint64_t kUserSpaceLimit = 0x800000000000ULL;

  static size_t addressable_prefix(const uint8_t *ptr, size_t len) {
    if (ptr == nullptr)
      return 0;
#if defined(RJ_GPU_MEMORY_WITH_ASAN)
    if (auto *poisoned = static_cast<const uint8_t *>(
            __asan_region_is_poisoned(const_cast<uint8_t *>(ptr), len)))
      return static_cast<size_t>(poisoned - ptr);
#endif
    return len;
  }

  static std::pair<uint8_t *, size_t> addressable_range_containing(uint8_t *base, size_t len,
                                                                   uint8_t *address) {
    if (base == nullptr || address < base || static_cast<size_t>(address - base) >= len)
      return {nullptr, 0};
#if defined(RJ_GPU_MEMORY_WITH_ASAN)
    len = heap_allocation_bounded_length(base, len);
    if (static_cast<size_t>(address - base) >= len)
      return {nullptr, 0};
    if (__asan_address_is_poisoned(address))
      return {nullptr, 0};
    auto *begin = address;
    constexpr size_t kBackwardProbeBytes = 4096;
    while (begin > base) {
      auto *chunk_begin = begin - std::min<size_t>(begin - base, kBackwardProbeBytes);
      if (__asan_region_is_poisoned(chunk_begin, begin - chunk_begin) == nullptr) {
        begin = chunk_begin;
        continue;
      }
      while (begin > chunk_begin && !__asan_address_is_poisoned(begin - 1))
        --begin;
      break;
    }
    auto *limit = base + len;
    auto *end = address + addressable_prefix(address, limit - address);
    return {begin, static_cast<size_t>(end - begin)};
#else
    return {base, len};
#endif
  }

  template <typename F>
  static void for_each_bounded_addressable_span(uint8_t *base, size_t len, F &&fn) {
    if (base == nullptr || len == 0)
      return;
#if defined(RJ_GPU_MEMORY_WITH_ASAN)
    size_t offset = 0;
    while (offset < len) {
      auto *poisoned =
          static_cast<uint8_t *>(__asan_region_is_poisoned(base + offset, len - offset));
      if (poisoned == nullptr) {
        fn(offset, len - offset);
        break;
      }
      const size_t poisoned_offset = poisoned - base;
      if (offset < poisoned_offset)
        fn(offset, poisoned_offset - offset);
      offset = poisoned_offset;
      while (offset < len && __asan_address_is_poisoned(base + offset))
        ++offset;
    }
#else
    fn(0, len);
#endif
  }

  template <typename F> static void for_each_addressable_span(uint8_t *base, size_t len, F &&fn) {
    if (base == nullptr || len == 0)
      return;
#if defined(RJ_GPU_MEMORY_WITH_ASAN)
    len = heap_allocation_bounded_length(base, len);
#endif
    for_each_bounded_addressable_span(base, len, std::forward<F>(fn));
  }

  static size_t heap_allocation_bounded_length([[maybe_unused]] uint8_t *base, size_t len) {
#if defined(RJ_GPU_MEMORY_WITH_ASAN)
    std::array<char, 1> name{};
    void *region_address = nullptr;
    size_t region_size = 0;
    const char *region_kind =
        __asan_locate_address(base, name.data(), name.size(), &region_address, &region_size);
    // GCC's ASan can report stack-variable metadata from the current thread for
    // an address on another thread's stack. Its global shadow remains accurate,
    // so only use allocator metadata to bound actual heap allocations.
    const auto base_address = reinterpret_cast<uintptr_t>(base);
    const auto region_begin = reinterpret_cast<uintptr_t>(region_address);
    if (region_kind != nullptr && std::strcmp(region_kind, "heap") == 0 &&
        region_address != nullptr && base_address >= region_begin &&
        base_address - region_begin < region_size)
      return std::min(len, region_size - (base_address - region_begin));
#endif
    return len;
  }

  struct VmidEntry {
    KfdProcess::PageTable *page_table = nullptr;
    std::shared_mutex *mutex = nullptr;
    pid_t client_pid = 0;
    const uint64_t *generation = nullptr;
  };

  struct PteCache {
    const GpuMemory *memory = nullptr;
    uint64_t memory_instance = 0;
    uint32_t vmid = 0;
    uint64_t registry_generation = 0;
    uint64_t page_key = 0;
    uint64_t generation = 0;
    bool found = false;
    KfdProcess::PageTableEntry pte;
    KfdProcess::PageTable *page_table = nullptr;
    std::shared_mutex *mutex = nullptr;
    const uint64_t *generation_ptr = nullptr;
  };

  static const KfdProcess::HostExtent *host_extent_at(const KfdProcess::PageTableEntry &pte,
                                                      size_t page_offset) {
    for (const auto &extent : pte.host_extents) {
      if (page_offset >= extent.gpu_page_offset &&
          page_offset - extent.gpu_page_offset < extent.host_backed_bytes)
        return &extent;
    }
    return nullptr;
  }

  static const KfdProcess::HostExtent *
  host_extent_starting_at_page(const KfdProcess::PageTableEntry &pte) {
    return !pte.host_extents.empty() && pte.host_extents.front().gpu_page_offset == 0
               ? &pte.host_extents.front()
               : nullptr;
  }

  static const KfdProcess::HostExtent *
  host_extent_ending_at_page(const KfdProcess::PageTableEntry &pte) {
    if (pte.host_extents.empty())
      return nullptr;
    const auto &extent = pte.host_extents.back();
    return extent.gpu_page_offset + extent.host_backed_bytes == PAGE_SIZE ? &extent : nullptr;
  }

  template <typename F>
  static size_t for_each_mapped_span(const KfdProcess::PageTableEntry &pte, size_t access_begin,
                                     size_t len, F &&fn) {
    const size_t access_end = access_begin + len;
    size_t mapped_bytes = 0;
    for (const auto &extent : pte.host_extents) {
      const size_t extent_begin = extent.gpu_page_offset;
      const size_t extent_end = extent_begin + extent.host_backed_bytes;
      const size_t overlap_begin = std::max(access_begin, extent_begin);
      const size_t overlap_end = std::min(access_end, extent_end);
      if (overlap_begin >= overlap_end)
        continue;
      auto *host_begin = extent.host_ptr + (overlap_begin - extent_begin);
      for_each_bounded_addressable_span(
          host_begin, overlap_end - overlap_begin, [&](size_t span_offset, size_t span_size) {
            mapped_bytes += span_size;
            fn(overlap_begin - access_begin + span_offset, host_begin + span_offset, span_size);
          });
    }
    return mapped_bytes;
  }

  void note_clipped_mapped_access(const char *operation, uint64_t addr, size_t size,
                                  uint32_t vmid) const {
    const uint64_t count = clipped_mapped_accesses_.fetch_add(1, std::memory_order_relaxed) + 1;
    util::Logger::vm("GPU memory ", operation, " clipped: addr=0x", std::hex, addr, std::dec,
                     " size=", size, " vmid=", vmid, " count=", count);
  }

  /// @brief Walk a VMID page table with a generation-keyed thread-local cache.
  /// @details The callback runs while both VMID registration and the selected
  /// page table are shared-locked. This keeps a cached host pointer alive for
  /// the whole copy and makes translate() and pte_mtype() share one invalidation
  /// protocol.
  template <typename F>
  auto cached_walk(uint64_t addr, uint32_t vmid, PteCache &cache,
                   F &&fn) const -> std::invoke_result_t<F, const KfdProcess::PageTableEntry *> {
    const uint64_t page_key = addr >> PAGE_SHIFT;
    std::shared_lock vmid_lock(vmid_mutex_);
    const uint64_t registry_generation = vmid_registry_generation_;

    const bool cached_table =
        cache.memory == this && cache.memory_instance == instance_id_ && cache.vmid == vmid &&
        cache.registry_generation == registry_generation && cache.page_table && cache.mutex;
    if (!cached_table) {
      auto it = vmid_table_.find(vmid);
      if (it == vmid_table_.end()) {
        cache = {};
        return fn(nullptr);
      }
      cache.memory = this;
      cache.memory_instance = instance_id_;
      cache.vmid = vmid;
      cache.registry_generation = registry_generation;
      cache.page_table = it->second.page_table;
      cache.mutex = it->second.mutex;
      cache.generation_ptr = it->second.generation;
      cache.found = false;
    }

    std::shared_lock page_table_lock(*cache.mutex);
    const uint64_t generation = cache.generation_ptr ? *cache.generation_ptr : 0;
    const bool cached_page = cached_table && cache.generation_ptr &&
                             cache.generation == generation && cache.page_key == page_key;
    if (!cached_page) {
      auto it = cache.page_table->find(page_key);
      cache.page_key = page_key;
      cache.generation = generation;
      cache.found = it != cache.page_table->end();
      if (cache.found)
        cache.pte = it->second;
#if defined(RJ_GPU_MEMORY_WITH_ASAN)
      if (cache.found) {
        for (auto &extent : cache.pte.host_extents)
          extent.host_backed_bytes =
              heap_allocation_bounded_length(extent.host_ptr, extent.host_backed_bytes);
      }
#endif
    }

    return fn(cache.found ? &cache.pte : nullptr);
  }

  template <typename F> bool with_page_mapping(uint64_t addr, uint32_t vmid, F &&fn) const {
    if (vmid == 0) {
      auto *page = reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK);
      if (!passthrough_ || addr >= kUserSpaceLimit || page == nullptr)
        return false;
      fn(nullptr, page);
      return true;
    }

    static thread_local PteCache cache;
    return cached_walk(addr, vmid, cache, [&](const KfdProcess::PageTableEntry *pte) {
      if (pte) {
        if (pte->host_extents.empty())
          return false;
        fn(pte, nullptr);
        return true;
      }
      if (passthrough_ && addr < kUserSpaceLimit) {
        auto *page = reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK);
        if (page == nullptr)
          return false;
        fn(nullptr, page);
        return true;
      }
      return false;
    });
  }

  bool read_mapped(uint64_t addr, void *dst, size_t len, uint32_t vmid) const {
    if ((addr & PAGE_MASK) + len > PAGE_SIZE)
      return false;
    std::memset(dst, 0, len);
    return with_page_mapping(
        addr, vmid, [&](const KfdProcess::PageTableEntry *pte, uint8_t *passthrough_page) {
          const size_t access_begin = addr & PAGE_MASK;
          if (pte) {
            const size_t mapped_bytes = for_each_mapped_span(
                *pte, access_begin, len,
                [&](size_t value_offset, uint8_t *host_ptr, size_t span_size) {
                  std::memcpy(static_cast<uint8_t *>(dst) + value_offset, host_ptr, span_size);
                });
            if (mapped_bytes != len)
              note_clipped_mapped_access("read", addr, len, vmid);
            return;
          }
          auto *host_ptr = passthrough_page + access_begin;
          for_each_addressable_span(host_ptr, len, [&](size_t value_offset, size_t span_size) {
            std::memcpy(static_cast<uint8_t *>(dst) + value_offset, host_ptr + value_offset,
                        span_size);
          });
        });
  }

  bool write_mapped(uint64_t addr, const void *src, size_t len, uint32_t vmid) {
    if ((addr & PAGE_MASK) + len > PAGE_SIZE)
      return false;
    return with_page_mapping(
        addr, vmid, [&](const KfdProcess::PageTableEntry *pte, uint8_t *passthrough_page) {
          const size_t access_begin = addr & PAGE_MASK;
          if (pte) {
            const size_t mapped_bytes = for_each_mapped_span(
                *pte, access_begin, len,
                [&](size_t value_offset, uint8_t *host_ptr, size_t span_size) {
                  std::memcpy(host_ptr, static_cast<const uint8_t *>(src) + value_offset,
                              span_size);
                });
            if (mapped_bytes != len)
              note_clipped_mapped_access("write", addr, len, vmid);
            return;
          }
          auto *host_ptr = passthrough_page + access_begin;
          for_each_addressable_span(host_ptr, len, [&](size_t value_offset, size_t span_size) {
            std::memcpy(host_ptr + value_offset, static_cast<const uint8_t *>(src) + value_offset,
                        span_size);
          });
        });
  }

  uint8_t *translate(uint64_t addr, uint32_t vmid, size_t size) const {
    if (size == 0 || (addr & PAGE_MASK) + size > PAGE_SIZE)
      return nullptr;
    uint8_t *host_ptr = nullptr;
    with_page_mapping(
        addr, vmid, [&](const KfdProcess::PageTableEntry *pte, uint8_t *passthrough_page) {
          const size_t page_offset = addr & PAGE_MASK;
          if (pte) {
            const auto *extent = host_extent_at(*pte, page_offset);
            if (!extent ||
                size > extent->host_backed_bytes - (page_offset - extent->gpu_page_offset))
              return;
            auto *candidate = extent->host_ptr + (page_offset - extent->gpu_page_offset);
            if (addressable_prefix(candidate, size) == size)
              host_ptr = candidate;
            return;
          }
          if (addr >= kUserSpaceLimit || size > kUserSpaceLimit - addr)
            return;
          auto *candidate = passthrough_page + page_offset;
          if (addressable_prefix(candidate, size) == size)
            host_ptr = candidate;
        });
    return host_ptr;
  }

  pid_t client_pid_for_vmid(uint32_t vmid) const {
    std::shared_lock lk(vmid_mutex_);
    auto it = vmid_table_.find(vmid);
    return (it != vmid_table_.end()) ? it->second.client_pid : 0;
  }

  bool read_client_memory(uint64_t addr, void *dst, size_t len, uint32_t vmid) const {
    return read_client_memory_for_pid(addr, dst, len, client_pid_for_vmid(vmid));
  }

  static bool read_client_memory_for_pid(uint64_t addr, void *dst, size_t len, pid_t pid) {
    if (pid <= 0)
      return false;
    iovec local{dst, len};
    iovec remote{reinterpret_cast<void *>(addr), len};
    ssize_t rc = process_vm_readv(pid, &local, 1, &remote, 1, 0);
    if (rc != static_cast<ssize_t>(len)) {
      util::Logger::warn("process_vm_readv failed: addr=0x", std::hex, addr, " pid=", std::dec, pid,
                         " rc=", rc, " errno=", errno);
      return false;
    }
    return true;
  }

  bool write_client_memory(uint64_t addr, const void *src, size_t len, uint32_t vmid) {
    return write_client_memory_for_pid(addr, src, len, client_pid_for_vmid(vmid));
  }

  static bool write_client_memory_for_pid(uint64_t addr, const void *src, size_t len, pid_t pid) {
    if (pid <= 0)
      return false;
    iovec local{const_cast<void *>(src), len};
    iovec remote{reinterpret_cast<void *>(addr), len};
    ssize_t rc = process_vm_writev(pid, &local, 1, &remote, 1, 0);
    if (rc != static_cast<ssize_t>(len)) {
      util::Logger::warn("process_vm_writev failed: addr=0x", std::hex, addr, " pid=", std::dec,
                         pid, " rc=", rc, " errno=", errno);
      return false;
    }
    return true;
  }

  simdojo::Port *cpl_ = nullptr;
  // Every object lifetime needs a distinct token because the function-static
  // TLS caches can survive destruction on long-lived host threads.
  inline static std::atomic<uint64_t> next_instance_id_{1};
  const uint64_t instance_id_;
  mutable std::shared_mutex vmid_mutex_;
  std::unordered_map<uint32_t, VmidEntry> vmid_table_;
  // Version of VMID-to-page-table bindings, accessed only under vmid_mutex_.
  uint64_t vmid_registry_generation_ = 1;
  mutable std::atomic<uint64_t> clipped_mapped_accesses_{0};
  bool passthrough_ = false;
};

} // namespace amdgpu
} // namespace rocjitsu

#undef RJ_GPU_MEMORY_WITH_ASAN

#endif // ROCJITSU_VM_AMDGPU_GPU_MEMORY_H_
