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
#include "util/unique_handle.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <format>
#include <functional>
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

static_assert(KfdProcess::kPageShift == simdojo::SparseMemory::PAGE_SHIFT,
              "KFD and sparse-memory page shifts must match");
static_assert(KfdProcess::kPageSize == simdojo::SparseMemory::PAGE_SIZE,
              "KFD and sparse-memory page sizes must match");

/// @brief AMDGPU VRAM memory with VMID-based per-process page table resolution.
///
/// @details Mirrors the GFXHUB's VMID register file. Each process registers its
/// page table via register_process(). A memory access's vmid parameter selects
/// the page table for VA-to-host translation. VMID zero remains the intentional
/// default for host, driver, and test callers. Wave-issued accesses must instead
/// bind the issuing wave's process ID through a wave-scoped memory API, matching
/// real hardware where the VMID travels with each request through the memory
/// hierarchy.
class GpuMemory : public simdojo::SparseMemory {
public:
  class PageTableRequestGuard {
  public:
    PageTableRequestGuard() = default;
    PageTableRequestGuard(PageTableRequestGuard &&) noexcept = default;
    PageTableRequestGuard &operator=(PageTableRequestGuard &&) noexcept = default;
    PageTableRequestGuard(const PageTableRequestGuard &) = delete;
    PageTableRequestGuard &operator=(const PageTableRequestGuard &) = delete;

    bool owns_lock() const { return lock_.owns_lock(); }
    bool cacheable() const { return owns_lock() && generation_ != nullptr; }
    uint64_t registry_generation() const {
      assert(owns_lock());
      return registry_generation_;
    }
    uint64_t page_table_generation() const {
      assert(cacheable());
      return *generation_;
    }
    void unlock() {
      if (owns_lock())
        lock_.unlock();
    }

  private:
    friend class GpuMemory;

    explicit PageTableRequestGuard(std::shared_ptr<std::shared_mutex> mutex)
        : mutex_(std::move(mutex)),
          lock_(mutex_ ? std::shared_lock(*mutex_) : std::shared_lock<std::shared_mutex>{}) {}

    void bind(KfdProcess::PageTable *page_table, std::shared_mutex *page_table_mutex,
              const uint64_t *generation, uint64_t registry_generation) {
      page_table_ = page_table;
      page_table_mutex_ = page_table_mutex;
      generation_ = generation;
      registry_generation_ = registry_generation;
    }

    std::shared_ptr<std::shared_mutex> mutex_;
    std::shared_lock<std::shared_mutex> lock_;
    KfdProcess::PageTable *page_table_ = nullptr;
    std::shared_mutex *page_table_mutex_ = nullptr;
    const uint64_t *generation_ = nullptr;
    uint64_t registry_generation_ = 0;
  };

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
  /// @param request_mutex Optional lease that stabilizes batched page-table
  ///        lookups. Omitting it disables cross-chunk MTYPE reuse.
  void register_process(uint32_t pid, KfdProcess::PageTable *pt, std::shared_mutex *mu,
                        const uint64_t *generation = nullptr,
                        std::shared_ptr<std::shared_mutex> request_mutex = {}) {
    util::Logger::cp("VMID_REG pid=", pid, " mem=0x", std::hex, reinterpret_cast<uintptr_t>(this),
                     std::dec, " pt_size=", pt->size());
    update_vmid_registration(pid, [&](auto) {
      vmid_table_[pid] = {
          .page_table = pt,
          .mutex = mu,
          .client_pid = 0,
          .client_mem_fd = {},
          .generation = generation,
          .request_mutex = std::move(request_mutex),
      };
      return true;
    });
  }

  /// @brief Stabilize a registered process's page table for one MTYPE lookup.
  /// @details Page-table mutations take the exclusive side of this lease before
  /// the ordinary page-table lock. Callers must release the lease before a
  /// backing-memory access, whose allocator metadata query may reenter KFD.
  PageTableRequestGuard acquire_page_table_request(uint32_t vmid) const {
    if (vmid == 0)
      return {};
    while (true) {
      std::shared_ptr<std::shared_mutex> request_mutex;
      {
        std::shared_lock lock(vmid_mutex_);
        auto it = vmid_table_.find(vmid);
        if (it == vmid_table_.end() || !it->second.request_mutex)
          return {};
        request_mutex = it->second.request_mutex;
      }

      PageTableRequestGuard guard(request_mutex);
      std::shared_lock lock(vmid_mutex_);
      auto it = vmid_table_.find(vmid);
      if (it != vmid_table_.end() && it->second.request_mutex == request_mutex) {
        guard.bind(it->second.page_table, it->second.mutex, it->second.generation,
                   vmid_registry_generation_);
        return guard;
      }
    }
  }

  /// @brief Reacquire a previously discovered request lease and revalidate its
  /// VMID binding.
  /// @details The cached mutex avoids the initial VMID-table lookup on the hot
  /// path. The binding is still checked after locking because replacement can
  /// install a different page table while the lease is released.
  bool reacquire_page_table_request(uint32_t vmid, PageTableRequestGuard &guard) const {
    if (vmid == 0)
      return false;
    if (guard.mutex_) {
      guard.lock_.lock();
      std::shared_lock lock(vmid_mutex_);
      auto it = vmid_table_.find(vmid);
      if (it != vmid_table_.end() && it->second.request_mutex == guard.mutex_) {
        guard.bind(it->second.page_table, it->second.mutex, it->second.generation,
                   vmid_registry_generation_);
        return true;
      }
      guard.lock_.unlock();
      guard = {};
    }
    guard = acquire_page_table_request(vmid);
    return guard.owns_lock();
  }

  /// @brief Unregister a process from the VMID table.
  void unregister_process(uint32_t pid) {
    util::Logger::cp("VMID_UNREG pid=", pid, " mem=0x", std::hex, reinterpret_cast<uintptr_t>(this),
                     std::dec);
    update_vmid_registration(pid, [&](auto it) {
      if (it == vmid_table_.end())
        return false;
      vmid_table_.erase(it);
      return true;
    });
  }

  void set_process_client_pid(uint32_t pid, pid_t client_pid) {
    std::unique_lock lk(vmid_mutex_);
    auto it = vmid_table_.find(pid);
    if (it != vmid_table_.end())
      it->second.client_pid = client_pid;
  }

  void set_process_mem_fd(uint32_t pid, int mem_fd) {
    std::unique_lock lk(vmid_mutex_);
    auto it = vmid_table_.find(pid);
    if (it == vmid_table_.end())
      return;
    const int duplicate = mem_fd >= 0 ? ::fcntl(mem_fd, F_DUPFD_CLOEXEC, 0) : -1;
    it->second.client_mem_fd.reset(duplicate);
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
    return every_page(addr, size,
                      [&](uint64_t page_addr) { return has_page_mapping(page_addr, vmid); });
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

  /// @brief Return whether a GPU VA has a VMID page-table mapping.
  bool is_mapped(uint64_t addr, uint32_t vmid = 0) const {
    if (vmid == 0)
      return passthrough_;
    std::shared_lock lk(vmid_mutex_);
    auto it = vmid_table_.find(vmid);
    if (it == vmid_table_.end())
      return false;
    auto &entry = it->second;
    std::shared_lock pt_lk(*entry.mutex);
    return entry.page_table->contains(addr >> PAGE_SHIFT);
  }

  /// @brief Return whether every page touched by a range has a VMID mapping.
  /// @details The range-aware form of is_mapped(), with the same deliberate
  /// strictness: passthrough is ignored for vmid != 0. Use this rather than
  /// is_mapped() on a base address when the access is wider than a byte -- an
  /// access starting near the end of a mapped page can still run off it.
  /// A zero size, or a range that wraps the address space, is not mapped.
  bool is_range_mapped(uint64_t addr, size_t size, uint32_t vmid = 0) const {
    return every_page(addr, size, [&](uint64_t page_addr) { return is_mapped(page_addr, vmid); });
  }

  /// @brief Look up PTE MTYPE for a GPU VA in the given VMID's page table.
  /// @details This reads only the page-wide policy. It deliberately avoids the
  /// host-extent copy, addressability checks, and external allocator queries
  /// needed by accesses that expose backing storage.
  Mtype pte_mtype(uint64_t addr, uint32_t vmid = 0) const {
    if (vmid == 0)
      return Mtype::RW;
    std::shared_lock vmid_lock(vmid_mutex_);
    auto vmid_entry = vmid_table_.find(vmid);
    if (vmid_entry == vmid_table_.end())
      return Mtype::RW;
    std::shared_lock page_table_lock(*vmid_entry->second.mutex);
    auto pte = vmid_entry->second.page_table->find(addr >> PAGE_SHIFT);
    return pte != vmid_entry->second.page_table->end() ? pte->second.mtype : Mtype::RW;
  }

  /// @brief Look up PTE MTYPE through an already validated request lease.
  Mtype pte_mtype(uint64_t addr, const PageTableRequestGuard &guard) const {
    assert(guard.owns_lock());
    assert(guard.page_table_ != nullptr && guard.page_table_mutex_ != nullptr);
    std::shared_lock page_table_lock(*guard.page_table_mutex_);
    auto pte = guard.page_table_->find(addr >> PAGE_SHIFT);
    return pte != guard.page_table_->end() ? pte->second.mtype : Mtype::RW;
  }

  uint32_t fetch32(uint64_t addr, uint32_t vmid = 0) const { return read32(addr, vmid); }

  bool is_fetchable(uint64_t addr, uint32_t vmid = 0) const {
    if (is_mapped(addr, vmid) || simdojo::SparseMemory::has_page(addr))
      return true;
    uint8_t byte = 0;
    if (util::UniqueHandle mem_fd = duplicate_client_mem_fd(vmid); mem_fd.get() >= 0)
      return pread(mem_fd.get(), &byte, sizeof(byte), static_cast<off_t>(addr)) == sizeof(byte);
    pid_t pid = client_pid_for_vmid(vmid);
    if (pid <= 0)
      return false;
    iovec local{&byte, sizeof(byte)};
    iovec remote{reinterpret_cast<void *>(addr), sizeof(byte)};
    return process_vm_readv(pid, &local, 1, &remote, 1, 0) == sizeof(byte);
  }

  /// @brief Read a contiguous range from simulated GPU memory.
  /// @details Handles each page through mapped host memory, client memory, or
  /// sparse backing memory. Every path resolves a whole page chunk at a time,
  /// so a read costs one page-table walk and one page-stripe lock per page it
  /// touches rather than one per byte -- which is what makes this cheap enough
  /// for the I$ to fill a line through. A mapped access clipped by a host
  /// extent remains zero-filled and emits a VM diagnostic so a future
  /// strict-fault mode can reuse the same boundary detection.
  void read_block(uint64_t addr, std::span<uint8_t> dst, uint32_t vmid = 0) const {
    for_each_page_chunk(addr, dst.size(), [&](uint64_t ea, size_t offset, size_t chunk) {
      auto out = dst.subspan(offset, chunk);
      if (read_mapped(ea, out.data(), chunk, vmid))
        return;
      if (vmid > 0 && read_client_memory(ea, out.data(), chunk, vmid))
        return;
      // The chunk is within one page, so this is a single sparse-page lock.
      simdojo::SparseMemory::read_block(ea, out);
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

  /// @brief Copy a contiguous range between two VMID-scoped addresses.
  /// @details Unlike read_block()/write_block(), this does not fall back to
  /// sparse memory: an SDMA packet must remain pending when either endpoint is
  /// inaccessible. In daemon mode, pageable host pointers are accessed through
  /// process_vm_readv/process_vm_writev while GPU allocations use their mapped
  /// daemon backing. Transfers are split at both source and destination page
  /// boundaries so each chunk resolves within one page, and each read and write
  /// runs inside the page-table mapping callback: no host pointer outlives the
  /// lock that keeps its allocation alive, so a concurrent process teardown
  /// cannot unmap the storage mid-copy.
  bool copy_block(uint64_t dst_addr, uint64_t src_addr, size_t len, uint32_t vmid = 0) {
    std::array<uint8_t, PAGE_SIZE> buffer{};
    size_t offset = 0;
    while (offset < len) {
      const uint64_t src_ea = src_addr + offset;
      const uint64_t dst_ea = dst_addr + offset;
      const size_t chunk = std::min(
          {len - offset, PAGE_SIZE - (src_ea & PAGE_MASK), PAGE_SIZE - (dst_ea & PAGE_MASK)});

      if (!copy_from_mapped(src_ea, buffer.data(), chunk, vmid) &&
          (vmid == 0 || !read_client_memory(src_ea, buffer.data(), chunk, vmid)))
        return false;

      if (!copy_to_mapped(dst_ea, buffer.data(), chunk, vmid) &&
          (vmid == 0 || !write_client_memory(dst_ea, buffer.data(), chunk, vmid)))
        return false;

      offset += chunk;
    }
    return true;
  }

  uint8_t *translate_debug(uint64_t addr, uint32_t vmid, size_t size = 1) const {
    return translate(addr, vmid, size);
  }

  /// @brief Find the contiguous host range containing a VMID-scoped GPU VA.
  /// @details KFD dispatches use per-process page tables. Kernel-symbol
  /// resolution needs a daemon-accessible host pointer range so it can scan
  /// backward from the kernel descriptor to the loaded ELF header. Sanitized
  /// builds release the VMID and page-table locks around allocator queries,
  /// then revalidate the registry generation and every contributing PTE.
  std::pair<uint64_t, uint64_t> find_host_range(uint64_t addr, uint32_t vmid) const {
    if (vmid == 0) {
      auto *host = translate(addr, vmid, 1);
      if (!host)
        return {0, 0};
      auto *page = host - (addr & PAGE_MASK);
      auto [range, size] = addressable_range_containing(page, PAGE_SIZE, host);
      return {reinterpret_cast<uint64_t>(range), size};
    }

#if defined(RJ_GPU_MEMORY_WITH_ASAN)
    size_t metadata_retries = 0;
#endif
    while (true) {
      std::shared_lock vmid_lock(vmid_mutex_);
#if defined(RJ_GPU_MEMORY_WITH_ASAN)
      const uint64_t registry_generation = vmid_registry_generation_;
#endif
      auto vmid_entry = vmid_table_.find(vmid);
      if (vmid_entry == vmid_table_.end())
        return {0, 0};

      auto &entry = vmid_entry->second;
      auto *page_table = entry.page_table;
      auto *page_table_mutex = entry.mutex;
#if defined(RJ_GPU_MEMORY_WITH_ASAN)
      const uint64_t *generation_ptr = entry.generation;
#endif
      std::shared_lock page_table_lock(*page_table_mutex);
      const uint64_t page = addr >> PAGE_SHIFT;
      auto page_entry = page_table->find(page);
      if (page_entry == page_table->end())
        return {0, 0};

      const size_t page_offset = addr & PAGE_MASK;
      const auto *current_extent = host_extent_at(page_entry->second, page_offset);
      if (!current_extent)
        return {0, 0};

      uint64_t first_page = page;
      const auto *first_extent = current_extent;
      uint8_t *first_host_byte = first_extent->host_ptr;
      while (first_page > 0 && first_extent->gpu_page_offset == 0) {
        auto previous_page_entry = page_table->find(first_page - 1);
        if (previous_page_entry == page_table->end())
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
        auto next_page_entry = page_table->find(last_page + 1);
        if (next_page_entry == page_table->end())
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
#if defined(RJ_GPU_MEMORY_WITH_ASAN)
      auto *host_byte = current_extent->host_ptr + (page_offset - current_extent->gpu_page_offset);
      std::vector<std::pair<uint64_t, KfdProcess::PageTableEntry>> pte_snapshot;
      pte_snapshot.reserve(last_page - first_page + 1);
      for (uint64_t snapshot_page = first_page;; ++snapshot_page) {
        pte_snapshot.emplace_back(snapshot_page, page_table->at(snapshot_page));
        if (snapshot_page == last_page)
          break;
      }

      page_table_lock.unlock();
      vmid_lock.unlock();
      run_asan_page_table_unlocked_hook();
      auto [range, range_size] =
          addressable_range_containing(first_host_byte, declared_range_size, host_byte);

      vmid_lock.lock();
      auto current_vmid_entry = vmid_table_.find(vmid);
      bool snapshot_valid = vmid_registry_generation_ == registry_generation &&
                            current_vmid_entry != vmid_table_.end() &&
                            current_vmid_entry->second.page_table == page_table &&
                            current_vmid_entry->second.mutex == page_table_mutex &&
                            current_vmid_entry->second.generation == generation_ptr;
      if (snapshot_valid) {
        page_table_lock.lock();
        for (const auto &[snapshot_page, snapshot_pte] : pte_snapshot) {
          auto current_pte = page_table->find(snapshot_page);
          if (current_pte == page_table->end() || current_pte->second != snapshot_pte) {
            snapshot_valid = false;
            break;
          }
        }
      }
      if (snapshot_valid)
        return {reinterpret_cast<uint64_t>(range), range_size};
      if (++metadata_retries >= kMaxMetadataRetries)
        return {0, 0};
#else
      auto [range, range_size] = addressable_range_containing(
          first_host_byte, declared_range_size,
          current_extent->host_ptr + (page_offset - current_extent->gpu_page_offset));
      return {reinterpret_cast<uint64_t>(range), range_size};
#endif
    }
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

  /// @brief Try to read a 64-bit location as ONE atomic acquire load.
  /// @details An AQL write pointer, a read pointer, a completion-signal value: the
  /// other side publishes these with a single atomic store. Copying them byte-wise
  /// can observe a half-updated value, and -- just as damaging -- leaves the reader
  /// with no happens-before edge to that store, so everything the store publishes
  /// (the packet a new write index makes visible) is read unsynchronized too.
  /// @returns false, writing nothing, when the eight bytes are not one aligned
  ///          mapped span; the caller keeps whatever slower path it already had,
  ///          since a split or unmapped range cannot be read atomically anyway.
  [[nodiscard]] bool try_read_u64_atomic(uint64_t addr, uint64_t *out, uint32_t vmid) const {
    constexpr size_t kLen = sizeof(uint64_t);
    if (addr % alignof(uint64_t) != 0 || (addr & PAGE_MASK) + kLen > PAGE_SIZE)
      return false;

    bool loaded = false;
    const bool mapped = with_page_mapping(
        addr, vmid, [&](const KfdProcess::PageTableEntry *pte, uint8_t *passthrough_page) {
          (void)passthrough_page; // Passthrough pages keep the addressability-checked copy.
          if (!pte)
            return;
          uint8_t *whole = nullptr;
          size_t spans = 0;
          const size_t mapped_bytes =
              for_each_mapped_span(*pte, addr & PAGE_MASK, kLen,
                                   [&](size_t value_offset, uint8_t *host_ptr, size_t span_size) {
                                     ++spans;
                                     if (value_offset == 0 && span_size == kLen)
                                       whole = host_ptr;
                                   });
          if (mapped_bytes != kLen || spans != 1 || whole == nullptr ||
              reinterpret_cast<uintptr_t>(whole) % alignof(uint64_t) != 0)
            return;
          *out = std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(whole))
                     .load(std::memory_order_acquire);
          loaded = true;
        });
    return mapped && loaded;
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
  // Bound repeated allocator queries when one page is continuously remapped.
  static constexpr size_t kMaxMetadataRetries = 8;
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

  /// @brief Whether @p pred holds for every page touched by [addr, addr+size).
  /// @details Shared spine of has_range_mapping() and is_range_mapped(), which
  /// differ only in their per-page predicate. Unlike for_each_page_chunk() this
  /// stops at the first page that fails, so a large unmapped range costs one
  /// page-table walk rather than one per 4 KiB. A zero size, or a range that
  /// wraps the address space, satisfies nothing.
  template <typename Pred> static bool every_page(uint64_t addr, size_t size, Pred &&pred) {
    if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - addr)
      return false;
    size_t offset = 0;
    while (offset < size) {
      const uint64_t ea = addr + offset;
      if (!pred(ea))
        return false;
      offset += std::min(size - offset, PAGE_SIZE - (ea & PAGE_MASK));
    }
    return true;
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
    // A fully addressable range cannot cross an ASan heap allocation boundary:
    // heap redzones are poisoned. Clean shadow also covers memory owned by
    // external allocators, where the declared mapping is the available bound.
    if (__asan_region_is_poisoned(base, len) == nullptr)
      return len;
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
    /// Debugger-authorized /proc/<target>/mem fd, or empty.
    util::UniqueHandle client_mem_fd;
    const uint64_t *generation = nullptr;
    std::shared_ptr<std::shared_mutex> request_mutex;
  };

  /// @brief Update a VMID binding while excluding an in-progress MTYPE lookup.
  /// @details The lease is acquired without holding vmid_mutex_, then the
  /// binding is revalidated under the exclusive VMID lock. This prevents a
  /// replacement or removal from overtaking an active lookup without
  /// introducing a lock-order cycle.
  template <typename F> void update_vmid_registration(uint32_t pid, F &&update) {
    while (true) {
      std::shared_ptr<std::shared_mutex> request_mutex;
      {
        std::shared_lock lock(vmid_mutex_);
        auto it = vmid_table_.find(pid);
        if (it != vmid_table_.end())
          request_mutex = it->second.request_mutex;
      }

      std::unique_lock<std::shared_mutex> request_lock;
      if (request_mutex)
        request_lock = std::unique_lock(*request_mutex);

      std::unique_lock lock(vmid_mutex_);
      auto it = vmid_table_.find(pid);
      const auto current_request_mutex =
          it != vmid_table_.end() ? it->second.request_mutex : nullptr;
      if (current_request_mutex != request_mutex)
        continue;

      if (update(it))
        ++vmid_registry_generation_;
      return;
    }
  }

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

#if defined(RJ_GPU_MEMORY_WITH_ASAN)
  /// @brief Test-only callback invoked while page-table and VMID locks are released.
  using AsanPageTableUnlockedHook = std::function<void()>;
#endif

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
  /// @details A mapped-PTE callback runs while both VMID registration and the
  /// selected page table are shared-locked. Miss callbacks and ASan allocator
  /// queries run without either lock. After an unlocked query, the VMID binding
  /// and exact PTE contents are revalidated before the bounded copy is published.
  /// Addressability checks in the mapped-span helpers remain the final guard
  /// against host-allocation reuse that preserves identical PTE contents.
  template <typename F>
  auto cached_walk(uint64_t addr, uint32_t vmid, PteCache &cache,
                   F &&fn) const -> std::invoke_result_t<F, const KfdProcess::PageTableEntry *> {
    const uint64_t page_key = addr >> PAGE_SHIFT;
#if defined(RJ_GPU_MEMORY_WITH_ASAN)
    size_t metadata_retries = 0;
#endif
    bool allow_cache_hit = true;
    while (true) {
      std::shared_lock vmid_lock(vmid_mutex_);
      const uint64_t registry_generation = vmid_registry_generation_;
      const bool cached_table =
          cache.memory == this && cache.memory_instance == instance_id_ && cache.vmid == vmid &&
          cache.registry_generation == registry_generation && cache.page_table && cache.mutex;
      KfdProcess::PageTable *page_table = cache.page_table;
      std::shared_mutex *page_table_mutex = cache.mutex;
      const uint64_t *generation_ptr = cache.generation_ptr;
      if (!cached_table) {
        auto vmid_entry = vmid_table_.find(vmid);
        if (vmid_entry == vmid_table_.end()) {
          cache = {};
          vmid_lock.unlock();
          return fn(nullptr);
        }
        page_table = vmid_entry->second.page_table;
        page_table_mutex = vmid_entry->second.mutex;
        generation_ptr = vmid_entry->second.generation;
      }

      std::shared_lock page_table_lock(*page_table_mutex);
      uint64_t generation = generation_ptr ? *generation_ptr : 0;
      auto publish_cache = [&](bool found, KfdProcess::PageTableEntry pte) {
        cache = {
            .memory = this,
            .memory_instance = instance_id_,
            .vmid = vmid,
            .registry_generation = registry_generation,
            .page_key = page_key,
            .generation = generation,
            .found = found,
            .pte = std::move(pte),
            .page_table = page_table,
            .mutex = page_table_mutex,
            .generation_ptr = generation_ptr,
        };
      };
      const bool cached_page = allow_cache_hit && cached_table && generation_ptr &&
                               cache.generation == generation && cache.page_key == page_key;
      if (cached_page) {
        if (cache.found)
          return fn(&cache.pte);
        page_table_lock.unlock();
        vmid_lock.unlock();
        return fn(nullptr);
      }
      allow_cache_hit = false;

      auto it = page_table->find(page_key);
      if (it == page_table->end()) {
        publish_cache(false, {});
        // A miss exposes no page-table storage. Release this lock before a
        // passthrough callback performs any ASan allocator query.
        page_table_lock.unlock();
        vmid_lock.unlock();
        return fn(nullptr);
      }
      const auto &[candidate_mtype, candidate_host_extents] = it->second;
      KfdProcess::PageTableEntry candidate;
      candidate.mtype = candidate_mtype;
      candidate.host_extents = candidate_host_extents;
#if defined(RJ_GPU_MEMORY_WITH_ASAN)
      const KfdProcess::PageTableEntry raw_pte = candidate;
      // AMD's ASan address lookup may consult ROCr for device allocations.
      // Drop the page-table lock around that external query, then validate the
      // copied PTE before exposing its host pointers to the callback.
      page_table_lock.unlock();
      vmid_lock.unlock();
      run_asan_page_table_unlocked_hook();
      for (auto &extent : candidate.host_extents)
        extent.host_backed_bytes =
            heap_allocation_bounded_length(extent.host_ptr, extent.host_backed_bytes);

      vmid_lock.lock();
      if (vmid_registry_generation_ != registry_generation) {
        cache = {};
        if (++metadata_retries < kMaxMetadataRetries)
          continue;

        // Unrelated VMID churn consumes the same bounded retry budget as a
        // target remap. Re-resolve the target registration under its locks so
        // this access remains fail-closed without turning a live mapping into
        // a passthrough miss.
        auto current_vmid_entry = vmid_table_.find(vmid);
        if (current_vmid_entry == vmid_table_.end()) {
          vmid_lock.unlock();
          return fn(nullptr);
        }
        auto *current_page_table = current_vmid_entry->second.page_table;
        std::shared_lock current_page_table_lock(*current_vmid_entry->second.mutex);
        auto current_pte = current_page_table->find(page_key);
        if (current_pte == current_page_table->end()) {
          current_page_table_lock.unlock();
          vmid_lock.unlock();
          return fn(nullptr);
        }
        candidate = current_pte->second;
        for (auto &extent : candidate.host_extents)
          extent.host_backed_bytes = 0;
        return fn(&candidate);
      }
      page_table_lock.lock();
      const bool generation_unchanged = generation_ptr && *generation_ptr == generation;
      bool mapping_changed = !generation_unchanged;
      if (mapping_changed) {
        it = page_table->find(page_key);
        mapping_changed = it == page_table->end() || it->second != raw_pte;
      }
      if (mapping_changed) {
        if (++metadata_retries < kMaxMetadataRetries)
          continue;

        // Preserve mapped-page identity while preventing access through bounds
        // that could not be validated under continuous remapping.
        if (it == page_table->end()) {
          publish_cache(false, {});
          page_table_lock.unlock();
          vmid_lock.unlock();
          return fn(nullptr);
        }
        candidate = it->second;
        for (auto &extent : candidate.host_extents)
          extent.host_backed_bytes = 0;
        // Use the fail-closed bounds for this access only. Publishing this
        // synthetic PTE would make later accesses reuse zero-length extents
        // after remapping has stopped.
        cache = {};
        return fn(&candidate);
      }
      if (generation_ptr)
        generation = *generation_ptr;
#endif
      publish_cache(true, std::move(candidate));
      return fn(&cache.pte);
    }
  }

#if defined(RJ_GPU_MEMORY_WITH_ASAN)
  void run_asan_page_table_unlocked_hook() const {
    if (auto *hook = asan_page_table_unlocked_hook_.load(std::memory_order_acquire))
      (*hook)();
  }
#endif

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

  /// @brief Run @p fn over a fully-backed, page-bounded host span with the
  /// page-table lock held for the duration.
  /// @details Applies exactly translate()'s resolution rules -- the range must
  /// be contiguous, host-backed and addressable -- but hands the pointer to a
  /// callback instead of returning it, so a concurrent unmap or VMID
  /// unregistration cannot free the storage between resolution and use.
  /// @return true if the span resolved and @p fn ran.
  template <typename F>
  bool with_translated_span(uint64_t addr, uint32_t vmid, size_t size, F &&fn) const {
    if (size == 0 || (addr & PAGE_MASK) + size > PAGE_SIZE)
      return false;
    bool copied = false;
    with_page_mapping(addr, vmid,
                      [&](const KfdProcess::PageTableEntry *pte, uint8_t *passthrough_page) {
                        const size_t page_offset = addr & PAGE_MASK;
                        uint8_t *candidate = nullptr;
                        if (pte) {
                          const auto *extent = host_extent_at(*pte, page_offset);
                          if (!extent || size > extent->host_backed_bytes -
                                                    (page_offset - extent->gpu_page_offset))
                            return;
                          candidate = extent->host_ptr + (page_offset - extent->gpu_page_offset);
                        } else {
                          if (addr >= kUserSpaceLimit || size > kUserSpaceLimit - addr)
                            return;
                          candidate = passthrough_page + page_offset;
                        }
                        if (addressable_prefix(candidate, size) != size)
                          return;
                        fn(candidate);
                        copied = true;
                      });
    return copied;
  }

  /// @brief Read a mapped span into @p dst without ever exposing a bare pointer.
  bool copy_from_mapped(uint64_t addr, void *dst, size_t size, uint32_t vmid) const {
    return with_translated_span(addr, vmid, size,
                                [&](const uint8_t *host_ptr) { std::memcpy(dst, host_ptr, size); });
  }

  /// @brief Write @p src into a mapped span without ever exposing a bare pointer.
  bool copy_to_mapped(uint64_t addr, const void *src, size_t size, uint32_t vmid) const {
    return with_translated_span(addr, vmid, size,
                                [&](uint8_t *host_ptr) { std::memcpy(host_ptr, src, size); });
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

  util::UniqueHandle duplicate_client_mem_fd(uint32_t vmid) const {
    std::shared_lock lk(vmid_mutex_);
    auto it = vmid_table_.find(vmid);
    if (it == vmid_table_.end() || it->second.client_mem_fd.get() < 0)
      return {};
    return util::UniqueHandle(::fcntl(it->second.client_mem_fd.get(), F_DUPFD_CLOEXEC, 0));
  }

  bool read_client_memory(uint64_t addr, void *dst, size_t len, uint32_t vmid) const {
    // Prefer the debugger-authorized /proc/<pid>/mem fd when the debug session
    // transferred one. The daemon is not the debuggee's ptrace parent, so the
    // process_vm_readv() fallback below is refused (EPERM) for a target it did
    // not itself attach to.
    if (util::UniqueHandle mem_fd = duplicate_client_mem_fd(vmid); mem_fd.get() >= 0) {
      const ssize_t rc = pread(mem_fd.get(), dst, len, static_cast<off_t>(addr));
      if (rc == static_cast<ssize_t>(len))
        return true;
    }
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
    // See read_client_memory(): the authorized fd is the only path that works
    // for a debuggee the daemon did not ptrace-attach to itself.
    if (util::UniqueHandle mem_fd = duplicate_client_mem_fd(vmid); mem_fd.get() >= 0) {
      const ssize_t rc = pwrite(mem_fd.get(), src, len, static_cast<off_t>(addr));
      if (rc == static_cast<ssize_t>(len))
        return true;
    }
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
#if defined(RJ_GPU_MEMORY_WITH_ASAN)
  /// @brief Private unit-test seam for deterministic unlocked-query coverage.
  /// @details The pointed-to callback must outlive every concurrent invocation.
  mutable std::atomic<AsanPageTableUnlockedHook *> asan_page_table_unlocked_hook_{nullptr};
#endif
  mutable std::atomic<uint64_t> clipped_mapped_accesses_{0};
  bool passthrough_ = false;
};

} // namespace amdgpu
} // namespace rocjitsu

#undef RJ_GPU_MEMORY_WITH_ASAN

#endif // ROCJITSU_VM_AMDGPU_GPU_MEMORY_H_
