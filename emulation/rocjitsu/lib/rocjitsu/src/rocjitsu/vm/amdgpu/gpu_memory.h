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

  /// @brief Resolve a GPU VA range to its first borrowed host byte.
  /// @details The returned pointer is only valid while page-table remapping and
  /// process teardown are quiesced. Normal memory operations use an internal
  /// callback that keeps the page-table shared lock held through the copy.
  uint8_t *resolve_host_ptr(uint64_t addr, uint32_t vmid = 0, size_t size = 1) const {
    if (size == 0)
      return nullptr;
    if ((addr & PAGE_MASK) + size <= PAGE_SIZE)
      return translate(addr, vmid, size);

    auto *host_ptr = translate(addr, vmid, 1);
    if (!host_ptr)
      return nullptr;
    if (reinterpret_cast<uintptr_t>(host_ptr) == addr)
      return addr < kUserSpaceLimit && size <= kUserSpaceLimit - addr ? host_ptr : nullptr;

    auto [range_base, range_size] = find_host_range(addr, vmid);
    const uintptr_t host_address = reinterpret_cast<uintptr_t>(host_ptr);
    if (range_base == 0 || host_address < range_base || host_address - range_base >= range_size ||
        size > range_size - (host_address - range_base))
      return nullptr;
    return host_ptr;
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
  /// sparse backing memory.
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
  /// sparse backing memory.
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
      const size_t page_offset = addr & PAGE_MASK;
      const size_t host_backed_bytes =
          cached_host_backed_bytes(page_key, pte->second, entry.page_table, entry.generation);
      if (pte->second.host_ptr == nullptr) {
        atomic_rmw_fallback(addr, size, entry.client_pid, fn);
        return;
      }

      const size_t backed_begin = pte->second.gpu_page_offset;
      const size_t backed_end = backed_begin + host_backed_bytes;
      const size_t access_end = page_offset + size;
      const size_t overlap_begin = std::max(page_offset, backed_begin);
      const size_t overlap_end = std::min(access_end, backed_end);
      if (overlap_begin >= overlap_end) {
        atomic_rmw_discarded(fn);
        return;
      }

      auto *target = pte->second.host_ptr + (overlap_begin - backed_begin);
      const size_t value_offset = overlap_begin - page_offset;
      const size_t mapped_bytes = overlap_end - overlap_begin;
      if (value_offset == 0 && mapped_bytes == size)
        atomic_rmw_mapped(target, fn);
      else
        atomic_rmw_clipped(target, value_offset, mapped_bytes, fn);
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
      return {reinterpret_cast<uint64_t>(host) - (addr & PAGE_MASK), PAGE_SIZE};
    }

    std::shared_lock vmid_lock(vmid_mutex_);
    auto vmid_entry = vmid_table_.find(vmid);
    if (vmid_entry == vmid_table_.end())
      return {0, 0};

    auto &entry = vmid_entry->second;
    std::shared_lock page_table_lock(*entry.mutex);
    const uint64_t page = addr >> PAGE_SHIFT;
    auto page_entry = entry.page_table->find(page);
    if (page_entry == entry.page_table->end() || page_entry->second.host_ptr == nullptr)
      return {0, 0};

    const size_t page_offset = addr & PAGE_MASK;
    if (page_offset < page_entry->second.gpu_page_offset ||
        page_offset >= page_entry->second.gpu_page_offset + page_entry->second.host_backed_bytes)
      return {0, 0};

    uint64_t first_page = page;
    auto *first_entry = &page_entry->second;
    uint8_t *first_host_byte = first_entry->host_ptr;
    while (first_page > 0) {
      auto previous_page_entry = entry.page_table->find(first_page - 1);
      if (previous_page_entry == entry.page_table->end() ||
          previous_page_entry->second.host_ptr == nullptr || first_entry->gpu_page_offset != 0 ||
          previous_page_entry->second.gpu_page_offset +
                  previous_page_entry->second.host_backed_bytes !=
              PAGE_SIZE ||
          previous_page_entry->second.host_ptr + previous_page_entry->second.host_backed_bytes !=
              first_host_byte)
        break;
      --first_page;
      first_entry = &previous_page_entry->second;
      first_host_byte = first_entry->host_ptr;
    }

    uint64_t last_page = page;
    auto *last_entry = &page_entry->second;
    while (true) {
      if (last_entry->gpu_page_offset + last_entry->host_backed_bytes != PAGE_SIZE)
        break;
      auto next_page_entry = entry.page_table->find(last_page + 1);
      if (next_page_entry == entry.page_table->end() ||
          next_page_entry->second.host_ptr == nullptr ||
          next_page_entry->second.gpu_page_offset != 0 ||
          next_page_entry->second.host_ptr != last_entry->host_ptr + last_entry->host_backed_bytes)
        break;
      ++last_page;
      last_entry = &next_page_entry->second;
    }

    // A host allocation can only acquire an ASan redzone at its final byte, so
    // the contiguous PTE walk uses declared extents and probes that boundary
    // once. This keeps per-dispatch symbol lookup linear in page-table entries
    // without adding a shadow scan for every page.
    const size_t last_host_backed_bytes =
        cached_host_backed_bytes(last_page, *last_entry, entry.page_table, entry.generation);
    const uintptr_t first_host_address = reinterpret_cast<uintptr_t>(first_host_byte);
    const uintptr_t last_host_address = reinterpret_cast<uintptr_t>(last_entry->host_ptr);
    const uint64_t range_size = last_host_address - first_host_address + last_host_backed_bytes;
    auto *host_byte =
        page_entry->second.host_ptr + (page_offset - page_entry->second.gpu_page_offset);
    if (reinterpret_cast<uintptr_t>(host_byte) - first_host_address >= range_size)
      return {0, 0};
    return {reinterpret_cast<uint64_t>(first_host_byte), range_size};
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

  static std::mutex &backing_atomic_mutex(uintptr_t key) {
    static std::array<std::mutex, kBackingAtomicLockStripes> mutexes;
    key >>= kBackingAtomicGranuleShift;
    key ^= key >> kBackingAtomicHashFoldShift1;
    key ^= key >> kBackingAtomicHashFoldShift2;
    return mutexes[key & (mutexes.size() - 1)];
  }

  template <typename F> static void atomic_rmw_mapped(uint8_t *target, F &fn) {
    std::lock_guard lock(backing_atomic_mutex(reinterpret_cast<uintptr_t>(target)));
    fn(target);
  }

  template <typename F>
  void atomic_rmw_unmapped(uint64_t addr, uint32_t size, pid_t client_pid, F &fn) {
    auto *target = reinterpret_cast<uint8_t *>(addr);
    if (passthrough_ && addr < kUserSpaceLimit && target != nullptr)
      atomic_rmw_mapped(target, fn);
    else
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
  static void atomic_rmw_clipped(uint8_t *target, size_t value_offset, size_t mapped_bytes, F &fn) {
    std::lock_guard lock(backing_atomic_mutex(reinterpret_cast<uintptr_t>(target)));
    std::array<uint8_t, sizeof(uint64_t)> value{};
    std::memcpy(value.data() + value_offset, target, mapped_bytes);
    fn(value.data());
    std::memcpy(target, value.data() + value_offset, mapped_bytes);
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

  static size_t effective_host_backed_bytes(const KfdProcess::PageTableEntry &pte) {
    return addressable_prefix(pte.host_ptr, pte.host_backed_bytes);
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

  struct HostExtentCacheEntry {
    const GpuMemory *memory = nullptr;
    uint64_t memory_instance = 0;
    KfdProcess::PageTable *page_table = nullptr;
    const uint64_t *generation_ptr = nullptr;
    uint64_t generation = 0;
    uint64_t page_key = 0;
    uint8_t *host_ptr = nullptr;
    size_t declared_bytes = 0;
    size_t host_backed_bytes = 0;
  };

  static constexpr size_t kHostExtentCacheEntries = 256;
  static_assert((kHostExtentCacheEntries & (kHostExtentCacheEntries - 1)) == 0,
                "host extent cache size must be a power of two");

  size_t cached_host_backed_bytes(uint64_t page_key, const KfdProcess::PageTableEntry &pte,
                                  KfdProcess::PageTable *page_table,
                                  const uint64_t *generation_ptr) const {
#if defined(RJ_GPU_MEMORY_WITH_ASAN)
    if (generation_ptr == nullptr)
      return effective_host_backed_bytes(pte);

    // HIP can update ASan shadow state without changing the KFD page table.
    // Cache the prefix scan, but recheck its two transition bytes on every hit
    // so both growth and shrinkage become visible before mapped memory is used.
    static thread_local std::array<HostExtentCacheEntry, kHostExtentCacheEntries> cache;
    const uintptr_t table_key = reinterpret_cast<uintptr_t>(page_table) >> PAGE_SHIFT;
    HostExtentCacheEntry &cache_entry =
        cache[(page_key ^ table_key ^ instance_id_) & (cache.size() - 1)];
    const uint64_t generation = *generation_ptr;
    bool cache_hit =
        cache_entry.memory == this && cache_entry.memory_instance == instance_id_ &&
        cache_entry.page_table == page_table && cache_entry.generation_ptr == generation_ptr &&
        cache_entry.generation == generation && cache_entry.page_key == page_key &&
        cache_entry.host_ptr == pte.host_ptr && cache_entry.declared_bytes == pte.host_backed_bytes;
    if (cache_hit) {
      const size_t cached_bytes = cache_entry.host_backed_bytes;
      const bool cached_tail_is_addressable =
          cached_bytes == 0 || addressable_prefix(pte.host_ptr + cached_bytes - 1, 1) == 1;
      const bool next_byte_is_poisoned = cached_bytes == pte.host_backed_bytes ||
                                         addressable_prefix(pte.host_ptr + cached_bytes, 1) == 0;
      cache_hit = cached_tail_is_addressable && next_byte_is_poisoned;
    }
    if (!cache_hit) {
      cache_entry = {
          .memory = this,
          .memory_instance = instance_id_,
          .page_table = page_table,
          .generation_ptr = generation_ptr,
          .generation = generation,
          .page_key = page_key,
          .host_ptr = pte.host_ptr,
          .declared_bytes = pte.host_backed_bytes,
          .host_backed_bytes = effective_host_backed_bytes(pte),
      };
    }
    return cache_entry.host_backed_bytes;
#else
    (void)page_key;
    (void)page_table;
    (void)generation_ptr;
    return pte.host_backed_bytes;
#endif
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
    }

    return fn(cache.found ? &cache.pte : nullptr);
  }

  template <typename F> bool with_host_ptr(uint64_t addr, uint32_t vmid, F &&fn) const {
    if (vmid == 0) {
      auto *page = reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK);
      if (!passthrough_ || addr >= kUserSpaceLimit || page == nullptr)
        return false;
      fn(page, 0, PAGE_SIZE);
      return true;
    }

    static thread_local PteCache cache;
    return cached_walk(addr, vmid, cache, [&](const KfdProcess::PageTableEntry *pte) {
      if (pte) {
        if (pte->host_ptr == nullptr)
          return false;
        const size_t host_backed_bytes =
            cached_host_backed_bytes(cache.page_key, *pte, cache.page_table, cache.generation_ptr);
        fn(pte->host_ptr, pte->gpu_page_offset, host_backed_bytes);
        return true;
      }
      if (passthrough_ && addr < kUserSpaceLimit) {
        auto *page = reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK);
        if (page == nullptr)
          return false;
        fn(page, 0, PAGE_SIZE);
        return true;
      }
      return false;
    });
  }

  bool read_mapped(uint64_t addr, void *dst, size_t len, uint32_t vmid) const {
    if ((addr & PAGE_MASK) + len > PAGE_SIZE)
      return false;
    return with_host_ptr(
        addr, vmid, [&](const uint8_t *host_ptr, size_t backed_begin, size_t host_backed_bytes) {
          const size_t access_begin = addr & PAGE_MASK;
          const size_t access_end = access_begin + len;
          const size_t backed_end = backed_begin + host_backed_bytes;
          const size_t overlap_begin = std::max(access_begin, backed_begin);
          const size_t overlap_end = std::min(access_end, backed_end);
          std::memset(dst, 0, len);
          if (overlap_begin < overlap_end) {
            std::memcpy(static_cast<uint8_t *>(dst) + (overlap_begin - access_begin),
                        host_ptr + (overlap_begin - backed_begin), overlap_end - overlap_begin);
          }
        });
  }

  bool write_mapped(uint64_t addr, const void *src, size_t len, uint32_t vmid) {
    if ((addr & PAGE_MASK) + len > PAGE_SIZE)
      return false;
    return with_host_ptr(
        addr, vmid, [&](uint8_t *host_ptr, size_t backed_begin, size_t host_backed_bytes) {
          const size_t access_begin = addr & PAGE_MASK;
          const size_t access_end = access_begin + len;
          const size_t backed_end = backed_begin + host_backed_bytes;
          const size_t overlap_begin = std::max(access_begin, backed_begin);
          const size_t overlap_end = std::min(access_end, backed_end);
          if (overlap_begin < overlap_end) {
            std::memcpy(host_ptr + (overlap_begin - backed_begin),
                        static_cast<const uint8_t *>(src) + (overlap_begin - access_begin),
                        overlap_end - overlap_begin);
          }
        });
  }

  uint8_t *translate(uint64_t addr, uint32_t vmid, size_t size) const {
    if (size == 0 || (addr & PAGE_MASK) + size > PAGE_SIZE)
      return nullptr;
    uint8_t *host_ptr = nullptr;
    with_host_ptr(
        addr, vmid, [&](uint8_t *mapped_host_ptr, size_t backed_begin, size_t host_backed_bytes) {
          const size_t page_offset = addr & PAGE_MASK;
          if (page_offset >= backed_begin && page_offset - backed_begin < host_backed_bytes &&
              size <= host_backed_bytes - (page_offset - backed_begin)) {
            host_ptr = mapped_host_ptr + (page_offset - backed_begin);
          }
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
  bool passthrough_ = false;
};

} // namespace amdgpu
} // namespace rocjitsu

#undef RJ_GPU_MEMORY_WITH_ASAN

#endif // ROCJITSU_VM_AMDGPU_GPU_MEMORY_H_
