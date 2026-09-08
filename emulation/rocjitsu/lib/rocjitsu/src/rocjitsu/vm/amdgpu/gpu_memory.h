// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_memory.h
/// @brief AMDGPU VRAM memory with per-process VMID-based page table resolution.

#ifndef ROCJITSU_VM_AMDGPU_GPU_MEMORY_H_
#define ROCJITSU_VM_AMDGPU_GPU_MEMORY_H_

#include "rocjitsu/kmd/linux/host_access_guard.h"
#include "rocjitsu/kmd/linux/host_mapping_lock.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "simdojo/components/sparse_memory.h"
#include "simdojo/sim/component.h"
#include "util/log.h"
#include "util/unique_handle.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdio>
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
#include <sys/syscall.h>
#include <sys/uio.h>
#include <type_traits>
#include <unistd.h>
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
/// @brief Sink for GPU memory violations detected while translating an address.
///
/// @details Hardware answers an access it cannot translate with a VM fault, and
/// the driver turns that into an event the runtime can see. GpuMemory detects
/// the same condition but has no way to reach a process, so it hands the
/// violation to whoever is emulating the driver. Keeping this an interface
/// rather than a direct call is what stops the memory model depending on the
/// KFD emulation it is used by.
/// @brief Why an access was refused, mapped onto what the driver reports.
/// @details The KFD ABI separates a page that is not there from one that is
/// there but refuses the write, and the runtime reads the two fields
/// separately, so collapsing them would misstate the cause.
enum class MemoryFaultCause : uint8_t {
  NotPresent, ///< No mapping at the address, or it is inaccessible outright.
  ReadOnly,   ///< The page is readable, but rejected the write.
  /// @brief The simulator could not establish the mapping's protection.
  /// @details Not a property of the address at all: procfs could not be opened
  /// or read through. The access still fails closed, but reporting it as a
  /// protection violation would blame the workload for the simulator running
  /// out of descriptors, so it is raised with no failure cause set -- an
  /// imprecise violation -- and named distinctly in the log.
  Indeterminate,
};

/// @brief What the kernel's record says about storing through a host page.
/// @details Read-only and absent are kept apart because the runtime reads the
/// two failure bits separately. Collapsing them reports a stale page-table
/// entry whose backing was unmapped, or one aimed into a PROT_NONE
/// reservation, as a protection violation on memory that is not there at all.
enum class PageWritability : uint8_t {
  Writable,      ///< The mapping exists and permits writes.
  ReadOnly,      ///< The mapping exists and is readable, but refuses writes.
  Inaccessible,  ///< Nothing is mapped there, or the mapping permits neither.
  Indeterminate, ///< The record could not be consulted; nothing was learned.
};

/// @brief A violation waiting for its caller's translation locks to release.
struct PendingFault {
  bool armed = false;
  uint64_t addr = 0;
  uint32_t vmid = 0;
  MemoryFaultCause cause = MemoryFaultCause::NotPresent;
};

class MemoryFaultReporter {
public:
  virtual ~MemoryFaultReporter() = default;

  /// @brief Report that @p addr could not be serviced for @p vmid.
  ///
  /// @details Called from simulation threads, so implementations must be
  /// thread-safe. Each GpuMemory binds its own reporter, so the implementation
  /// knows which device faulted without being told; a shared reporter would
  /// have to guess, and would name the wrong one.
  ///
  /// Calling back into GpuMemory is permitted, and is why delivery is deferred
  /// at all: this runs from ~FaultDispatch, after the access that recorded the
  /// fault has released the VMID and page-table locks, so a reporter is free to
  /// take the driver locks that reach GpuMemory from the other direction. A
  /// reporter that could not do that would be unable to reach the driver, which
  /// is the only thing it exists to do.
  ///
  /// The one thing it must not do is perform an access that itself faults. The
  /// pending slot is cleared before this call, so a nested fault arms a fresh
  /// one and the nested guard delivers it -- to this same reporter, without
  /// bound.
  virtual void report_memory_fault(uint32_t vmid, uint64_t addr, MemoryFaultCause cause) = 0;
};

/// @brief Why a block access ended, once a faulted address is distinguishable
/// from memory that is simply not backed yet.
///
/// @details Sparse backing is legitimate: GPU memory that has never been written
/// reads as zero and must keep doing so. A validated-inaccessible address is not
/// that, and conflating the two is what let an invalid transfer report success.
enum class AccessOutcome : uint8_t {
  Complete, ///< Every byte was serviced (mapped, client, or sparse backing).
  Faulted,  ///< At least one byte resolved to an address that does not exist.
};

/// @brief Why a copy between two GPU addresses ended.
enum class CopyOutcome : uint8_t {
  Complete,    ///< Every byte was copied.
  Unavailable, ///< An endpoint is not resolvable yet; the caller may retry.
  Faulted,     ///< An endpoint does not exist; retrying will never help.
};

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
    // Accesses through this object dereference memory the application can
    // revoke, so the handler that turns such a fault into a reported violation
    // has to exist before the first one. Idempotent, and cheap enough to do per
    // instance rather than asking every embedder to remember.
    rocjitsu::install_host_access_guard();
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
  /// @brief Observes whether any access faulted while it is alive.
  /// @details Thread-local because an access is serviced on the thread that
  /// issued it, and because threading an out-parameter through every accessor
  /// would put the reporting concern in signatures with no use for it. Callers
  /// that must tell "not resolvable yet" from "will never resolve" -- the SDMA
  /// control addresses, which otherwise retry a permanent fault forever -- wrap
  /// the access in one of these.
  class FaultScope {
  public:
    FaultScope() : start_(tls_identity_faults) {}
    [[nodiscard]] bool observed() const { return tls_identity_faults != start_; }

  private:
    uint64_t start_;
  };

  void set_passthrough(bool v) { passthrough_ = v; }

  /// @brief Route detected memory violations to @p reporter.
  /// @details Stored as a plain pointer load so the translation path pays
  /// nothing for it. The reporter must outlive every access; the driver owns it
  /// and clears it before teardown.
  void set_memory_fault_reporter(MemoryFaultReporter *reporter) {
    fault_reporter_.store(reporter, std::memory_order_release);
  }

  /// @brief Return whether the page containing an address has a known mapping.
  /// @details Unlike resolve_host_ptr(), this deliberately ignores the current
  /// host accessibility of the requested byte. Callers use it to distinguish a
  /// known mapping whose live extent is clipped from a page that may not have
  /// been installed yet.
  ///
  /// This and its siblings (has_range_mapping(), is_mapped(),
  /// is_range_mapped(), is_fetchable()) therefore still answer true for a
  /// passthrough address whose host page with_page_mapping() would reject, so a
  /// gate here can admit an access that then resolves to nothing. That is
  /// deliberate: these predicates screen whole SDMA ranges, and validating them
  /// would cost one syscall per page across transfers measured in megabytes.
  /// The accesses themselves are validated, which is what stops the host being
  /// corrupted; reconciling the gates needs a mechanism that does not scale with
  /// range size.
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
  AccessOutcome read_block(uint64_t addr, std::span<uint8_t> dst, uint32_t vmid = 0) const {
    const FaultDispatch fault_dispatch(*this);
    if (!range_within_address_space(addr, dst.size())) {
      note_rejected_identity_access(addr, vmid);
      std::ranges::fill(dst, uint8_t{0});
      return AccessOutcome::Faulted;
    }
    size_t stopped_at = dst.size();
    const bool completed =
        for_each_page_chunk_until(addr, dst.size(), [&](uint64_t ea, size_t offset, size_t chunk) {
          const FaultScope chunk_faults;
          auto out = dst.subspan(offset, chunk);
          if (read_mapped(ea, out.data(), chunk, vmid))
            return true;
          // A refused read has already zero-filled its own chunk. Substituting
          // sparse storage for it would hand back invented bytes as if they
          // were the address's contents, and reading on past the fault would
          // do the same for every page behind it.
          if (chunk_faults.observed()) {
            stopped_at = offset + chunk;
            return false;
          }
          // A registered client owns this address space, so its answer is the
          // only answer: substituting sparse storage for a transfer the kernel
          // refused hands back fabricated zeroes as if they were the address's
          // contents. Sparse remains the backing for GPU memory never written,
          // which is what an address with no client behind it is.
          if (vmid > 0 && has_client_backing(vmid)) {
            if (read_client_memory(ea, out.data(), chunk, vmid))
              return true;
            note_rejected_identity_access(ea, vmid);
            stopped_at = offset;
            return false;
          }
          // The chunk is within one page, so this is a single sparse-page lock.
          simdojo::SparseMemory::read_block(ea, out);
          return true;
        });
    if (completed)
      return AccessOutcome::Complete;
    // The caller may ignore the outcome, so the bytes past the fault must still
    // be the documented zero rather than whatever it handed in.
    std::ranges::fill(dst.subspan(stopped_at), uint8_t{0});
    return AccessOutcome::Faulted;
  }

  /// @brief Read a span, refusing to return anything less than all of it.
  ///
  /// @details read_block() is deliberately forgiving: bytes with no backing
  /// read as zero, because unwritten GPU memory legitimately reads as zero. A
  /// caller that is reading a *record* rather than data cannot use that. A
  /// signal's event id fabricated from a clipped extent is not a harmless zero:
  /// it either suppresses the wakeup its owner is parked on or names a
  /// different event entirely. Page-table entries carry sub-page and disjoint
  /// extents by design, so a record can straddle the end of its backing.
  [[nodiscard]] AccessOutcome read_block_exact(uint64_t addr, std::span<uint8_t> dst,
                                               uint32_t vmid = 0) const {
    const FaultDispatch fault_dispatch(*this);
    const uint64_t clipped_before = tls_clipped_accesses;
    const AccessOutcome outcome = read_block(addr, dst, vmid);
    if (outcome == AccessOutcome::Faulted || tls_clipped_accesses == clipped_before)
      return outcome;
    note_rejected_identity_access(addr, vmid);
    std::ranges::fill(dst, uint8_t{0});
    return AccessOutcome::Faulted;
  }

  /// @brief Write a contiguous range to simulated GPU memory.
  /// @details Handles each page through mapped host memory, client memory, or
  /// sparse backing memory. A mapped access clipped by a host extent is dropped
  /// for the missing bytes and emits a VM diagnostic.
  AccessOutcome write_block(uint64_t addr, std::span<const uint8_t> src, uint32_t vmid = 0) {
    const FaultDispatch fault_dispatch(*this);
    if (!range_within_address_space(addr, src.size())) {
      note_rejected_identity_access(addr, vmid);
      return AccessOutcome::Faulted;
    }
    const bool completed =
        for_each_page_chunk_until(addr, src.size(), [&](uint64_t ea, size_t offset, size_t chunk) {
          const FaultScope chunk_faults;
          auto in = src.subspan(offset, chunk);
          if (write_mapped(ea, in.data(), chunk, vmid))
            return true;
          // A refusal is not "nothing is mapped here, try elsewhere": the
          // address exists and may not be written. Falling through to the
          // client or to sparse would report a successful write of bytes
          // nobody can see, and continuing the walk would modify pages past
          // the one the engine should have stopped on.
          if (chunk_faults.observed())
            return false;
          // As in read_block(): a client-owned address that the kernel refused
          // is a fault, not an invitation to write somewhere the client will
          // never look.
          if (vmid > 0 && has_client_backing(vmid)) {
            if (write_client_memory(ea, in.data(), chunk, vmid))
              return true;
            note_rejected_identity_access(ea, vmid, MemoryFaultCause::Indeterminate);
            return false;
          }
          if (chunk_faults.observed())
            return false;
          for (size_t i = 0; i < chunk; ++i)
            simdojo::SparseMemory::write8(ea + i, in[i]);
          return true;
        });
    return completed ? AccessOutcome::Complete : AccessOutcome::Faulted;
  }

  /// @brief Perform an atomic read-modify-write on resolved backing storage.
  /// @details Storage classification and the page-table shared lock remain
  /// stable through the callback. Mapped aliases rendezvous on a process-wide
  /// host-address stripe; unmapped sparse/client accesses use an address-space
  /// stripe instead. An access whose range is not wholly backed is refused, not
  /// clipped: the callback sees zeroes, no byte of the target is modified, and
  /// the refusal is raised as a fault so a caller that publishes on completion
  /// does not publish a torn value.
  /// @param addr GPU virtual address of the target.
  /// @param size Access size in bytes (4 or 8).
  /// @param fn Callback invoked with a pointer to the target bytes.
  /// @param vmid Process VMID used for address translation.
  template <typename F> void atomic_rmw(uint64_t addr, uint32_t size, F &&fn, uint32_t vmid = 0) {
    assert((size == 4 || size == 8) && (addr & PAGE_MASK) + size <= PAGE_SIZE);
    const FaultDispatch fault_dispatch(*this);

    if (vmid == 0) {
      atomic_rmw_unmapped(addr, size, 0, vmid, fn);
      return;
    }

    std::shared_lock vmid_lock(vmid_mutex_);
    auto vmid_entry = vmid_table_.find(vmid);
    if (vmid_entry == vmid_table_.end()) {
      atomic_rmw_unmapped(addr, size, 0, vmid, fn);
      return;
    }

    auto &entry = vmid_entry->second;
    std::shared_lock page_table_lock(*entry.mutex);
    const uint64_t page_key = addr >> PAGE_SHIFT;
    auto pte = entry.page_table->find(page_key);
    if (pte != entry.page_table->end()) {
      if (atomic_rmw_mapped_page(pte->second, addr & PAGE_MASK, size, fn, addr, vmid) ==
          AtomicPageOutcome::Clipped) {
        // The access was refused rather than clipped: nothing was modified, and
        // a caller that publishes on Complete must not publish this one.
        note_clipped_mapped_access("atomic", addr, size, vmid);
        note_rejected_identity_access(addr, vmid);
      }
      return;
    }

    atomic_rmw_unmapped(addr, size, entry.client_pid, vmid, fn);
  }

  /// @brief Load @p size bytes atomically, holding the mapping still across it.
  ///
  /// @details The command processor used to read its control words -- SDMA wait
  /// operands, poll operands, semaphores -- through a pointer from translate(),
  /// which proves the page readable and then hands the pointer back. Between
  /// that proof and the load the application may munmap or mprotect the page,
  /// and the load faults the simulator on memory the check said was there.
  ///
  /// This keeps the acquire ordering the wait protocol needs -- the load is
  /// still a single atomic on the resolved storage -- while the mapping lease is
  /// held across both the validation and the load, so the answer cannot go stale
  /// between them.
  ///
  /// @param[in] addr GPU virtual address of the operand.
  /// @param[in] size 4 or 8; the operand must be naturally aligned.
  /// @param[out] value The loaded value, zero unless Complete is returned.
  /// @param[in] vmid Owning VMID.
  /// @retval Complete The value was read.
  /// @retval Unavailable Not resolvable yet; the caller may retry.
  /// @retval Faulted The address does not exist; retrying will never help.
  [[nodiscard]] CopyOutcome atomic_load(uint64_t addr, uint32_t size, uint64_t &value,
                                        uint32_t vmid) const {
    assert((size == sizeof(uint32_t) || size == sizeof(uint64_t)) &&
           "atomic control loads are 4 or 8 bytes");
    assert((addr & (size - 1)) == 0 && "atomic control loads must be naturally aligned");
    value = 0;
    const FaultScope faults;
    const FaultDispatch fault_dispatch(*this);
    if ((addr & PAGE_MASK) + size > PAGE_SIZE)
      return CopyOutcome::Unavailable; // Straddles a page, so not one atomic.
    const bool loaded = with_page_mapping(
        addr, vmid, [&](const KfdProcess::PageTableEntry *pte, IdentityPage page) {
          const size_t page_offset = addr & PAGE_MASK;
          // Taken before either branch validates, and held through the load.
          const auto mapping_lease = rocjitsu::host_mapping_lock().lock_shared();
          uint8_t *target = nullptr;
          if (pte) {
            const auto *extent = host_extent_at(*pte, page_offset);
            if (!extent ||
                size > extent->host_backed_bytes - (page_offset - extent->gpu_page_offset))
              return false;
            target = extent->host_ptr + (page_offset - extent->gpu_page_offset);
            if (addressable_prefix(target, size) != size)
              return false;
          } else {
            target = page.read_valid_pointer(page_offset, size);
            if (target == nullptr) {
              note_rejected_identity_access(addr, vmid);
              return false;
            }
          }
          value = size == sizeof(uint64_t)
                      ? std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(target))
                            .load(std::memory_order_acquire)
                      : std::atomic_ref<uint32_t>(*reinterpret_cast<uint32_t *>(target))
                            .load(std::memory_order_acquire);
          return true;
        });
    if (loaded)
      return CopyOutcome::Complete;
    // Same split the control paths already make: a fault is permanent and must
    // halt the queue, while an address that simply is not mapped yet is what a
    // wait exists to wait for.
    return faults.observed() ? CopyOutcome::Faulted : CopyOutcome::Unavailable;
  }

  /// @brief Store @p value atomically, reporting whether the address existed.
  ///
  /// @details The command processor used to publish fences, timestamps and
  /// queue pointers by storing through a pointer from translate(), which is
  /// proven readable and nothing more: a read-only destination crashed the
  /// process, and an unmap between the two redirected the store. Routing them
  /// here keeps the release ordering -- the store is still a single atomic on
  /// the resolved storage -- while the address is validated by the same path as
  /// every other access, and a bad one is reported rather than dereferenced.
  [[nodiscard]] AccessOutcome atomic_store(uint64_t addr, uint32_t size, uint64_t value,
                                           uint32_t vmid) {
    const FaultScope faults;
    atomic_rmw(
        addr, size,
        [&](uint8_t *bytes) {
          if (size == sizeof(uint64_t))
            std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(bytes))
                .store(value, std::memory_order_release);
          else
            std::atomic_ref<uint32_t>(*reinterpret_cast<uint32_t *>(bytes))
                .store(static_cast<uint32_t>(value), std::memory_order_release);
        },
        vmid);
    return faults.observed() ? AccessOutcome::Faulted : AccessOutcome::Complete;
  }

  /// @brief Subtract @p amount atomically, reporting whether the address existed.
  [[nodiscard]] AccessOutcome atomic_fetch_sub64(uint64_t addr, int64_t amount, uint32_t vmid) {
    const FaultScope faults;
    atomic_rmw(
        addr, sizeof(int64_t),
        [&](uint8_t *bytes) {
          std::atomic_ref<int64_t>(*reinterpret_cast<int64_t *>(bytes))
              .fetch_sub(amount, std::memory_order_release);
        },
        vmid);
    return faults.observed() ? AccessOutcome::Faulted : AccessOutcome::Complete;
  }

  /// @brief Add @p amount atomically, reporting whether the address existed.
  /// @details Unsigned because the operand is a raw 64-bit packet field: it may
  /// be any bit pattern, and reaching an addition by negating a signed value
  /// cannot express INT64_MIN -- negating it is undefined. Two's-complement
  /// wrap is the hardware behaviour anyway.
  [[nodiscard]] AccessOutcome atomic_fetch_add64(uint64_t addr, uint64_t amount, uint32_t vmid) {
    const FaultScope faults;
    atomic_rmw(
        addr, sizeof(uint64_t),
        [&](uint8_t *bytes) {
          std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(bytes))
              .fetch_add(amount, std::memory_order_release);
        },
        vmid);
    return faults.observed() ? AccessOutcome::Faulted : AccessOutcome::Complete;
  }

  /// @brief Report whether a range a caller will walk itself is expressible.
  ///
  /// @details The block and copy entry points reject a range that runs past the
  /// end of the address space and report it, but a caller that resolves and
  /// walks a range on its own -- the SDMA fill, which never presents the whole
  /// span to one call -- would otherwise reject it silently. Routing that
  /// preflight here keeps one definition of a malformed range and one place
  /// that tells the process about it, so a queue never halts without saying why.
  [[nodiscard]] AccessOutcome check_range(uint64_t addr, size_t size, uint32_t vmid) {
    const FaultDispatch fault_dispatch(*this);
    if (range_within_address_space(addr, size))
      return AccessOutcome::Complete;
    note_rejected_identity_access(addr, vmid);
    return AccessOutcome::Faulted;
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
  CopyOutcome copy_block(uint64_t dst_addr, uint64_t src_addr, size_t len, uint32_t vmid = 0) {
    // Declared before the scope so it destructs last: this is the only access
    // entry point that arms faults at its own level -- the client fallbacks
    // below -- rather than inside a helper that dispatches for itself, so
    // without this a refusal would be left armed for some later, unrelated
    // access to deliver against the wrong address.
    const FaultDispatch fault_dispatch(*this);
    const FaultScope faults;
    // Reported against whichever endpoint is malformed: naming the other one
    // sends the runtime to a buffer that is perfectly valid.
    if (!range_within_address_space(src_addr, len)) {
      note_rejected_identity_access(src_addr, vmid);
      return CopyOutcome::Faulted;
    }
    if (!range_within_address_space(dst_addr, len)) {
      note_rejected_identity_access(dst_addr, vmid);
      return CopyOutcome::Faulted;
    }
    std::array<uint8_t, PAGE_SIZE> buffer{};
    size_t offset = 0;
    while (offset < len) {
      const uint64_t src_ea = src_addr + offset;
      const uint64_t dst_ea = dst_addr + offset;
      const size_t chunk = std::min(
          {len - offset, PAGE_SIZE - (src_ea & PAGE_MASK), PAGE_SIZE - (dst_ea & PAGE_MASK)});

      // An endpoint that is merely not mapped yet is worth waiting for, so it
      // stays Unavailable and the packet is retried. An endpoint a registered
      // client owns and the kernel refused is not: it will not become readable
      // later, and retrying it re-runs the same packet on every doorbell while
      // the queue never drains. That is a fault.
      if (!copy_from_mapped(src_ea, buffer.data(), chunk, vmid)) {
        if (vmid == 0 || !has_client_backing(vmid))
          return faults.observed() ? CopyOutcome::Faulted : CopyOutcome::Unavailable;
        if (!read_client_memory(src_ea, buffer.data(), chunk, vmid)) {
          note_rejected_identity_access(src_ea, vmid);
          return CopyOutcome::Faulted;
        }
      }

      if (!copy_to_mapped(dst_ea, buffer.data(), chunk, vmid)) {
        if (vmid == 0 || !has_client_backing(vmid))
          return faults.observed() ? CopyOutcome::Faulted : CopyOutcome::Unavailable;
        if (!write_client_memory(dst_ea, buffer.data(), chunk, vmid)) {
          // Indeterminate, as in write_block(): process_vm_writev reports the
          // same failure for a page that is absent, one that is mapped
          // read-only, and a process that has exited, so naming this NotPresent
          // would put a cause on the fault that nothing here established.
          note_rejected_identity_access(dst_ea, vmid, MemoryFaultCause::Indeterminate);
          return CopyOutcome::Faulted;
        }
      }

      offset += chunk;
    }
    return faults.observed() ? CopyOutcome::Faulted : CopyOutcome::Complete;
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
    const bool mapped =
        with_page_mapping(addr, vmid, [&](const KfdProcess::PageTableEntry *pte, IdentityPage) {
          // Passthrough pages keep the addressability-checked copy.
          if (!pte)
            return false;
          uint8_t *whole = nullptr;
          size_t spans = 0;
          const size_t mapped_bytes =
              for_each_mapped_span(*pte, addr & PAGE_MASK, kLen,
                                   [&](size_t value_offset, uint8_t *host_ptr, size_t span_size,
                                       const KfdProcess::HostExtent &) {
                                     ++spans;
                                     if (value_offset == 0 && span_size == kLen)
                                       whole = host_ptr;
                                   });
          if (mapped_bytes != kLen || spans != 1 || whole == nullptr ||
              reinterpret_cast<uintptr_t>(whole) % alignof(uint64_t) != 0)
            return false;
          *out = std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(whole))
                     .load(std::memory_order_acquire);
          loaded = true;
          return true;
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
  void atomic_rmw_unmapped(uint64_t addr, uint32_t size, pid_t client_pid, uint32_t vmid, F &fn) {
    auto *target = reinterpret_cast<uint8_t *>(addr);
    if (passthrough_ && addr < kUserSpaceLimit && size <= kUserSpaceLimit - addr &&
        target != nullptr) {
      // The atomic is performed in place, on the real page, so it is a genuine
      // system-scope atomic: an application thread incrementing the same address
      // participates in it, which a read-modify-write split across two syscalls
      // could not offer -- both sides would read the same value and one update
      // would be lost. The vendored HSA contract requires that scope for
      // fine-grained system memory, so the alternative to doing it properly is
      // refusing to do it at all, not doing it approximately.
      //
      // Which means the pointer has to be one we may genuinely store through,
      // and readability does not establish that.
      auto *page = reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK);
      // Held across the check AND the modify, so the mapping cannot be
      // withdrawn or made read-only in between. The interposer takes the same
      // lock exclusively around the application's mapping calls.
      auto mapping_lock = rocjitsu::host_mapping_lock().lock_shared();
      const auto writability = host_page_writability(page);
      if (writability == PageWritability::Writable && addressable_prefix(target, size) == size) {
        atomic_rmw_mapped(target, fn);
        return;
      }
      mapping_lock.unlock();
      note_rejected_identity_access(addr, vmid, fault_cause_for(writability));
      atomic_rmw_discarded(fn);
      return;
    }
    atomic_rmw_fallback(addr, size, client_pid, vmid, fn);
  }

  /// @brief Perform an atomic with no host mapping to work against.
  ///
  /// @details Two unrelated situations reach here. A client process may own the
  /// bytes, in which case they are reached across the process boundary and the
  /// kernel says whether that worked. Or nothing owns them, in which case the
  /// simulator's own sparse store stands in -- a model of memory the GPU may
  /// scribble on that no one else observes.
  ///
  /// These must not be blended. Substituting sparse storage for a client access
  /// the kernel refused turns a fault into a successful atomic on invented
  /// memory: a queue read pointer or a completion signal appears to advance
  /// while the value the client actually reads never changes, which presents as
  /// a hang with no attribution. When a client owns the address, its answer is
  /// the only answer.
  template <typename F>
  void atomic_rmw_fallback(uint64_t addr, uint32_t size, pid_t client_pid, uint32_t vmid, F &fn) {
    uintptr_t key = static_cast<uintptr_t>(addr ^ (addr >> 32));
    if (client_pid > 0)
      key ^= static_cast<uintptr_t>(client_pid) * kClientPidHashSalt;
    else
      key ^= reinterpret_cast<uintptr_t>(this);

    std::lock_guard lock(backing_atomic_mutex(key));
    // Aligned for the widest atomic a callback may form over it: atomic_ref
    // requires its referent to meet required_alignment, which a byte array does
    // not promise even where the stack happens to supply it.
    alignas(uint64_t) std::array<uint8_t, sizeof(uint64_t)> value{};
    if (client_pid > 0) {
      // An atomic cannot be carried out on memory belonging to another
      // process. A read-modify-write split across two syscalls loses a
      // concurrent client update, and even a blind store is no better: the
      // release store lands on the local buffer above rather than on the
      // client's object, and process_vm_writev() is not documented to be
      // atomic, so the client can observe a torn value with none of the
      // ordering that publication depends on. The HSA contract puts device
      // atomics on fine-grained system memory at system scope, so approximating
      // one is reporting a completion the client cannot rely on. Servicing this
      // properly needs shared storage or a client-side atomic protocol.
      note_rejected_identity_access(addr, vmid, MemoryFaultCause::Indeterminate);
      atomic_rmw_discarded(fn);
      return;
    }

    for (uint32_t i = 0; i < size; ++i)
      value[i] = simdojo::SparseMemory::read8(addr + i);
    fn(value.data());
    for (uint32_t i = 0; i < size; ++i)
      simdojo::SparseMemory::write8(addr + i, value[i]);
  }

  /// @brief How an atomic against a page-table-backed page ended.
  enum class AtomicPageOutcome {
    Complete, ///< Every byte of the access landed on host storage.
    Clipped,  ///< Part of the access had no host backing and was discarded.
    Faulted,  ///< Host storage exists but may not be stored through.
  };

  /// @brief Perform an atomic against the host storage a PTE names.
  ///
  /// @details A page-table entry says where the bytes live, not what may be
  /// done to them. The host pages it names are the application's own mappings
  /// in local mode, so the application may have mapped or reprotected them
  /// read-only -- a queue read pointer mapped PROT_READ is the ordinary case --
  /// and an atomic stores in place by construction. Storing anyway is a host
  /// SIGSEGV inside the emulated command processor, attributed to nothing.
  ///
  /// So the same writability question the identity path asks is asked here,
  /// under the same lock and for the same span of time: the check and the
  /// modify must be one region, or the protection can be revoked between them.
  template <typename F>
  AtomicPageOutcome atomic_rmw_mapped_page(const KfdProcess::PageTableEntry &pte,
                                           size_t page_offset, size_t size, F &fn, uint64_t addr,
                                           uint32_t vmid) const {
    auto mapping_lock = rocjitsu::host_mapping_lock().lock_shared();
    const auto *extent = host_extent_at(pte, page_offset);
    if (extent && size <= extent->host_backed_bytes - (page_offset - extent->gpu_page_offset)) {
      auto *target = extent->host_ptr + (page_offset - extent->gpu_page_offset);
      auto cause = MemoryFaultCause::NotPresent;
      if (!extent_is_writable(*extent, target, size, cause)) {
        mapping_lock.unlock();
        note_rejected_identity_access(addr, vmid, cause);
        atomic_rmw_discarded(fn);
        return AtomicPageOutcome::Faulted;
      }
      if (addressable_prefix(target, size) == size) {
        atomic_rmw_mapped(target, fn);
        return AtomicPageOutcome::Complete;
      }
    }

    struct AtomicSpan {
      size_t value_offset = 0;
      uint8_t *host_ptr = nullptr;
      size_t size = 0;
    };
    std::array<AtomicSpan, sizeof(uint64_t)> spans{};
    size_t span_count = 0;
    const size_t mapped_bytes =
        for_each_mapped_span(pte, page_offset, size,
                             [&](size_t value_offset, uint8_t *host_ptr, size_t span_size,
                                 const KfdProcess::HostExtent &) {
                               assert(span_count < spans.size());
                               spans[span_count++] = {value_offset, host_ptr, span_size};
                             });
    // An atomic that covers bytes with no host backing is not an atomic over
    // its operand: applying it to the spans that happen to exist publishes a
    // partial fence, signal or queue pointer that the owner reads as whole.
    // Refuse the whole access and leave every span untouched.
    if (mapped_bytes != size) {
      atomic_rmw_discarded(fn);
      return AtomicPageOutcome::Clipped;
    }

    // A partially backed access is still a store into every span it does
    // cover, so each one has to be writable before any of them is touched.
    for (size_t i = 0; i < span_count; ++i) {
      const auto writability = host_range_writability(spans[i].host_ptr, spans[i].size);
      if (writability == PageWritability::Writable)
        continue;
      mapping_lock.unlock();
      note_rejected_identity_access(addr, vmid, fault_cause_for(writability));
      atomic_rmw_discarded(fn);
      return AtomicPageOutcome::Faulted;
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
          return AtomicPageOutcome::Clipped;
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

    // Aligned for the widest atomic a callback may form over it: atomic_ref
    // requires its referent to meet required_alignment, which a byte array does
    // not promise even where the stack happens to supply it.
    alignas(uint64_t) std::array<uint8_t, sizeof(uint64_t)> value{};
    for (size_t i = 0; i < span_count; ++i)
      std::memcpy(value.data() + spans[i].value_offset, spans[i].host_ptr, spans[i].size);
    fn(value.data());
    for (size_t i = 0; i < span_count; ++i)
      std::memcpy(spans[i].host_ptr, value.data() + spans[i].value_offset, spans[i].size);
    // Refused above unless the whole range is backed, so reaching here means
    // every byte landed.
    return AtomicPageOutcome::Complete;
  }

  template <typename F> static void atomic_rmw_discarded(F &fn) {
    // Aligned for the widest atomic a callback may form over it: atomic_ref
    // requires its referent to meet required_alignment, which a byte array does
    // not promise even where the stack happens to supply it.
    alignas(uint64_t) std::array<uint8_t, sizeof(uint64_t)> value{};
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

  /// @brief Whether [addr, addr+size) stays inside the address space.
  /// @details A range that wraps past the end is a malformed request, not one
  /// waiting on a mapping: the page walks below add offsets to @p addr without
  /// rechecking, so a wrapped range would resume at zero and modify unrelated
  /// low memory while reporting that it completed. An empty range trivially
  /// fits and stays a no-op.
  static bool range_within_address_space(uint64_t addr, size_t size) {
    return size == 0 || size - 1 <= std::numeric_limits<uint64_t>::max() - addr;
  }

  /// @brief Walk page chunks of [addr, addr+len), stopping when @p fn says so.
  /// @details Like for_each_page_chunk(), but the callback returns false to end
  /// the walk. A faulted access must not be followed by further accesses:
  /// hardware stops the engine at the fault, so a payload that spans a good
  /// page, a faulted one and another good one must leave the last page alone.
  /// @return True when every chunk was visited.
  template <typename F> static bool for_each_page_chunk_until(uint64_t addr, size_t len, F &&fn) {
    size_t offset = 0;
    while (offset < len) {
      const uint64_t ea = addr + offset;
      const size_t chunk = std::min(len - offset, PAGE_SIZE - (ea & PAGE_MASK));
      if (!fn(ea, offset, chunk))
        return false;
      offset += chunk;
    }
    return true;
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

  /// @brief A passthrough page, reachable only through a checked operation.
  ///
  /// @details The bare address of an identity page is the hazard this whole
  /// path exists to contain, so it is not handed out. A copy goes through the
  /// kernel, which validates and moves the bytes in the same call and therefore
  /// cannot be overtaken by an unmap between the two. Only a caller that must
  /// return a pointer to someone else asks for one, and pays a separate probe
  /// to get it -- that window is unavoidable once a raw pointer escapes, which
  /// is the reason to keep the set of callers that do so small and visible.
  class IdentityPage {
  public:
    explicit IdentityPage(uint8_t *page) : page_(page) {}

    [[nodiscard]] bool read(size_t offset, void *dst, size_t len) const {
      return transfer(offset, dst, len, /*to_page=*/false);
    }

    [[nodiscard]] bool write(size_t offset, const void *src, size_t len) const {
      return transfer(offset, const_cast<void *>(src), len, /*to_page=*/true);
    }

    /// @brief Return a pointer proven READABLE, or null.
    ///
    /// @details Readability is all the probe establishes, so storing through the
    /// result is not sound: a PROT_READ page satisfies it and then faults the
    /// host on the write. Every operation that modifies memory goes through
    /// read()/write() instead, where the kernel answers the permission question
    /// by performing the access, or -- for atomics, which must modify in place --
    /// through the writability check that precedes them. This exists only for
    /// translate(), whose callers need an address they can hold; those that then
    /// write through it, the direct SDMA stores and the completion signals,
    /// still carry that risk and want converting to the checked writes.
    [[nodiscard]] uint8_t *read_valid_pointer(size_t offset, size_t len) const {
      if (page_ == nullptr || !identity_page_is_accessible(page_))
        return nullptr;
      auto *candidate = page_ + offset;
      return addressable_prefix(candidate, len) == len ? candidate : nullptr;
    }

    [[nodiscard]] bool valid() const { return page_ != nullptr; }

    /// @brief Classify a failed write: readable pages refused it on protection.
    [[nodiscard]] MemoryFaultCause write_refusal_cause() const {
      return identity_page_is_accessible(page_) ? MemoryFaultCause::ReadOnly
                                                : MemoryFaultCause::NotPresent;
    }

  private:
    bool transfer(size_t offset, void *local_bytes, size_t len, bool to_page) const {
      if (page_ == nullptr)
        return false;
      if (addressable_prefix(page_ + offset, len) != len)
        return false; // Sanitized builds still veto poisoned bytes.
      iovec local{local_bytes, len};
      iovec remote{page_ + offset, len};
      const ssize_t moved = to_page ? process_vm_writev(getpid(), &local, 1, &remote, 1, 0)
                                    : process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
      return moved == static_cast<ssize_t>(len);
    }

    uint8_t *page_ = nullptr;
  };

  /// @brief Report whether the host page behind an identity translation exists.
  ///
  /// @details Passthrough answers a translation miss by reinterpreting the GPU
  /// address as a host address. That is sound only while the address really is
  /// one. An address invented by a defect elsewhere in the simulator is not, and
  /// dereferencing it costs a host SIGSEGV or -- worse -- silently lands on an
  /// unrelated live allocation. Neither outcome is attributable to the GPU
  /// access that caused it, which is the whole problem: this class turns other
  /// components' bugs into host crashes.
  ///
  /// Ask the kernel rather than trusting the address. A one-byte
  /// process_vm_readv() against this process reports failure for a hole, for a
  /// PROT_NONE reservation -- the shape the ROCm runtime leaves behind when it
  /// reserves a VA aperture, and the shape an unresolved GPU VA most often lands
  /// in -- and for an address never mapped at all, without ever touching the
  /// page. mincore() and msync() are cheaper but both report PROT_NONE as
  /// mapped, which is precisely the case worth catching.
  ///
  /// Page granularity is deliberate: a VMA never splits mid-page, so one probe
  /// settles the whole page, and every identity span this class hands out is
  /// bounded to a single page by its caller.
  static bool identity_page_is_accessible(const uint8_t *page) {
    if (page == nullptr)
      return false;
    uint8_t probe = 0;
    iovec local{&probe, sizeof(probe)};
    iovec remote{const_cast<uint8_t *>(page), sizeof(probe)};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == sizeof(probe);
  }

  /// @brief Report whether every host page under [ptr, ptr+size) is writable.
  /// @details An access never spans more than two pages here -- callers bound it
  /// to one GPU page -- but a host extent need not be page-aligned, so the last
  /// byte can sit in the next VMA, which may carry different protection.
  static PageWritability host_range_writability(const uint8_t *ptr, size_t size) {
    if (ptr == nullptr || size == 0)
      return PageWritability::Inaccessible;
    const auto first = reinterpret_cast<uintptr_t>(ptr) & ~static_cast<uintptr_t>(PAGE_MASK);
    const auto last =
        reinterpret_cast<uintptr_t>(ptr + size - 1) & ~static_cast<uintptr_t>(PAGE_MASK);
    // A definite refusal outranks an indeterminate one: knowing that any page
    // of the range may not be written settles the access regardless of what
    // could not be established about the rest.
    bool indeterminate = false;
    for (uintptr_t page = first; page <= last; page += PAGE_SIZE) {
      const auto writability = host_page_writability(reinterpret_cast<const uint8_t *>(page));
      if (writability == PageWritability::Writable)
        continue;
      // A definite refusal settles the access and carries the cause the
      // runtime will read, so the first one wins over a page nothing could be
      // established about.
      if (writability != PageWritability::Indeterminate)
        return writability;
      indeterminate = true;
    }
    return indeterminate ? PageWritability::Indeterminate : PageWritability::Writable;
  }

  /// @brief Whether a page-table extent may be dereferenced for @p writing.
  ///
  /// @details Driver extents are answered without a syscall. Their backing is a
  /// memfd the driver created, mapped read-write and still holds open, so no
  /// other party can change its protection or unmap it, and probing it would
  /// add a per-page cost to the path that moves the most bytes -- the SDMA
  /// block copies, measured in megabytes -- to re-derive a fact that is true by
  /// construction.
  ///
  /// Application extents are the caller's own pages, reached through USERPTR or
  /// by identity. The application may mprotect or munmap them while a transfer
  /// is in flight, so these are probed: readability by the same one-byte
  /// process_vm_readv the identity path uses, and writability by the protection
  /// the kernel reports, since there is no non-destructive write probe.
  ///
  /// @pre The caller holds rocjitsu::host_mapping_lock() shared across this and
  /// the access it authorises, so the answer cannot go stale in between.
  /// @param[out] cause Set only when the extent is refused.
  /// @brief Whether an in-place modification of @p extent may proceed.
  ///
  /// @details Only atomics need this. A block transfer lets the kernel perform
  /// the permission check as part of the move, which is both cheaper and free
  /// of any window between the two; an atomic has to modify the bytes where
  /// they are, so the question must be asked separately and answered before the
  /// modification. Driver extents are answered without asking, for the reason
  /// given on HostExtentOwner.
  ///
  /// @pre The caller holds rocjitsu::host_mapping_lock() shared across this and
  /// the modification it authorises.
  [[nodiscard]] static bool extent_is_writable(const KfdProcess::HostExtent &extent,
                                               const uint8_t *target, size_t size,
                                               MemoryFaultCause &cause) {
    if (extent.owner == KfdProcess::HostExtentOwner::Driver)
      return true;
    const auto writability = host_range_writability(target, size);
    if (writability == PageWritability::Writable)
      return true;
    cause = fault_cause_for(writability);
    return false;
  }

  /// @brief Classify a fault the access guard absorbed.
  /// @details Only reached when an access already faulted, so the cost of
  /// asking the kernel what the protection is here is paid once per violation
  /// rather than once per access.
  static MemoryFaultCause guarded_fault_cause() {
    const auto *address = static_cast<const uint8_t *>(rocjitsu::last_guarded_fault_address());
    if (address == nullptr)
      return MemoryFaultCause::NotPresent;
    return fault_cause_for(host_range_writability(address, 1));
  }

  /// @brief Translate a refused writability answer into the fault it reports.
  static MemoryFaultCause fault_cause_for(PageWritability writability) {
    switch (writability) {
    case PageWritability::ReadOnly:
      return MemoryFaultCause::ReadOnly;
    case PageWritability::Indeterminate:
      return MemoryFaultCause::Indeterminate;
    case PageWritability::Inaccessible:
    case PageWritability::Writable:
      break;
    }
    // Writable reaches here only when the store was refused for a reason the
    // protection did not explain -- a poisoned region under ASan -- which is
    // not a page the access may use either.
    return MemoryFaultCause::NotPresent;
  }

  /// @brief Report whether a host page is present and may be stored through.
  ///
  /// @details An atomic cannot be split into a read and a write without losing
  /// what makes it an atomic, so it needs a pointer it may genuinely store
  /// through -- and a read probe does not establish that. There is no
  /// non-destructive syscall that answers "is this writable", so ask the kernel
  /// for its own record of the mapping instead. That answer costs microseconds,
  /// which is why only the atomic path asks: ordinary reads and writes let the
  /// kernel answer by performing the access.
  ///
  /// The answer is deliberately not cached. Dating a cache to a counter the
  /// interposer bumps only covers mapping changes the interposer sees, and an
  /// address recycled by a change it missed reads back the old protection --
  /// which is a silent store through a read-only pointer, exactly the fault
  /// this exists to prevent, now with no diagnostic. A cache is only safe once
  /// the protection is metadata the driver owns rather than something the
  /// kernel is asked about after the fact.
  ///
  /// The caller must hold rocjitsu::host_mapping_lock() shared across this call and
  /// the store it authorises. Without that the application could revoke the
  /// protection in between, and the answer would describe a mapping that no
  /// longer exists.
  ///
  /// Every syscall here is issued raw. rocjitsu interposes open() and close(),
  /// and those hooks take the interposer's descriptor lock -- which a DRM
  /// GEM_VA ioctl already holds when it calls into the page table this runs
  /// under. Reaching them from here would close that cycle:
  ///   this path:  page table lock -> close() -> descriptor lock
  ///   GEM_VA:     descriptor lock -> map_pages() -> page table lock
  /// Nothing about reading procfs wants the hooks, so it does not call them.
  static PageWritability host_page_writability(const uint8_t *page) {
    if (page == nullptr)
      return PageWritability::Inaccessible;
    const auto address = reinterpret_cast<uintptr_t>(page);
    const long fd = ::syscall(SYS_openat, AT_FDCWD, "/proc/self/maps", O_RDONLY | O_CLOEXEC);
    if (fd < 0)
      return PageWritability::Indeterminate;

    // Parsed incrementally: the table can be large, and a line may straddle
    // reads, so keep any partial tail and prepend it to the next chunk.
    std::string pending;
    char chunk[8192];
    PageWritability answer = PageWritability::Indeterminate;
    for (bool reading = true; reading;) {
      const long got = ::syscall(SYS_read, static_cast<int>(fd), chunk, sizeof(chunk));
      if (got < 0) {
        // A signal arriving mid-read says nothing about the mapping.
        if (errno == EINTR)
          continue;
        break;
      }
      if (got == 0) {
        // Walked the whole table without covering the address: nothing is
        // mapped there, which is a definite answer rather than a failure.
        answer = PageWritability::Inaccessible;
        break;
      }
      pending.append(chunk, static_cast<size_t>(got));
      size_t line_begin = 0;
      for (size_t newline = pending.find('\n', line_begin); newline != std::string::npos;
           newline = pending.find('\n', line_begin)) {
        const std::string line(pending, line_begin, newline - line_begin);
        line_begin = newline + 1;
        uintptr_t begin = 0;
        uintptr_t end = 0;
        char permissions[5] = {};
        if (std::sscanf(line.c_str(), "%zx-%zx %4s", &begin, &end, permissions) != 3)
          continue;
        if (address < begin || address >= end)
          continue;
        // A mapping that permits neither read nor write -- a PROT_NONE
        // reservation, the shape the runtime leaves behind around an aperture
        // -- is absent as far as the GPU is concerned, not merely protected.
        answer = permissions[1] == 'w'   ? PageWritability::Writable
                 : permissions[0] == 'r' ? PageWritability::ReadOnly
                                         : PageWritability::Inaccessible;
        reading = false;
        break;
      }
      pending.erase(0, line_begin);
    }
    ::syscall(SYS_close, static_cast<int>(fd));
    return answer;
  }

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
      for_each_bounded_addressable_span(host_begin, overlap_end - overlap_begin,
                                        [&](size_t span_offset, size_t span_size) {
                                          mapped_bytes += span_size;
                                          fn(overlap_begin - access_begin + span_offset,
                                             host_begin + span_offset, span_size, extent);
                                        });
    }
    return mapped_bytes;
  }

  void note_clipped_mapped_access(const char *operation, uint64_t addr, size_t size,
                                  uint32_t vmid) const {
    ++tls_clipped_accesses;
    const uint64_t count = clipped_mapped_accesses_.fetch_add(1, std::memory_order_relaxed) + 1;
    util::Logger::vm("GPU memory ", operation, " clipped: addr=0x", std::hex, addr, std::dec,
                     " size=", size, " vmid=", vmid, " count=", count);
  }

  /// @brief Record a translation that resolved to an inaccessible identity page.
  /// @details Warn rather than trace: this always means the GPU address was
  /// never valid, and the access that follows reads zeros or is dropped. Left
  /// silent it would surface far away from its cause -- as wrong results, or as
  /// a wait on a completion signal that is never written.

  /// @brief Deliver any fault recorded during an access, after locks release.
  ///
  /// @details Reporting reaches the driver, which takes its process table lock;
  /// registering a process takes that lock and then this class's VMID lock. A
  /// translation miss discovers the fault while holding the VMID lock, so
  /// reporting from there would close the cycle -- a reopen racing an old
  /// process's faulting access would deadlock. Recording the fault and
  /// delivering it from a guard declared before the locks keeps the two orders
  /// from ever meeting.
  class FaultDispatch {
  public:
    explicit FaultDispatch(const GpuMemory &memory) : memory_(memory) {}
    FaultDispatch(const FaultDispatch &) = delete;
    FaultDispatch &operator=(const FaultDispatch &) = delete;
    ~FaultDispatch() { memory_.deliver_pending_fault(); }

  private:
    const GpuMemory &memory_;
  };

  void deliver_pending_fault() const {
    if (!tls_pending_fault.armed)
      return;
    const PendingFault pending = tls_pending_fault;
    tls_pending_fault.armed = false;
    if (pending.vmid == 0)
      return;
    if (auto *reporter = fault_reporter_.load(std::memory_order_acquire))
      reporter->report_memory_fault(pending.vmid, pending.addr, pending.cause);
  }

  void note_rejected_identity_access(uint64_t addr, uint32_t vmid,
                                     MemoryFaultCause cause = MemoryFaultCause::NotPresent) const {
    ++tls_identity_faults;
    const uint64_t count = rejected_identity_accesses_.fetch_add(1, std::memory_order_relaxed) + 1;
    // One bad address is rarely reached once: a wave re-executing the access
    // that produced it, across every lane, turns an unconditional warning into
    // millions of identical lines that bury the first one. Report at powers of
    // two so the opening occurrence is immediate, later ones stay visible, and
    // the total stays logarithmic in the damage.
    if ((count & (count - 1)) == 0)
      util::Logger::warn("GPU memory access rejected: address 0x", std::hex, addr, std::dec,
                         cause == MemoryFaultCause::ReadOnly ? " is not writable"
                         : cause == MemoryFaultCause::Indeterminate
                             ? " was refused for an undetermined reason"
                             : " has no host page",
                         " (vmid=", vmid, " count=", count, ")");

    // Raise it as a fault against the owning process, the way hardware would --
    // but not from here, which may hold translation locks the driver's own
    // lock order runs against. Arm it for the guard to deliver. Arming is
    // unthrottled where the log is not: the runtime coalesces repeats on one
    // event, and suppressing them here would instead hide a later, different
    // fault. VMID zero is the host/driver/test entry point and owns no process,
    // so there is nobody to fault.
    if (vmid == 0)
      return;
    tls_pending_fault = {true, addr, vmid, cause};
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

  /// @brief Resolve @p addr to either a page-table entry or an identity page.
  ///
  /// @details These two branches are the only places an identity host pointer is
  /// created, so validating here is what keeps every consumer honest --
  /// translate(), read_mapped(), write_mapped() and the span copies all inherit
  /// it, and find_host_range()'s VMID-zero range is
  /// exactly the page validated here. A page-table hit is left alone: those
  /// pointers address driver-owned memfd mappings and are valid by
  /// construction, so probing them would buy nothing and cost a syscall.
  template <typename F> bool with_page_mapping(uint64_t addr, uint32_t vmid, F &&fn) const {
    if (vmid == 0) {
      auto *page = reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK);
      if (!passthrough_ || addr >= kUserSpaceLimit || page == nullptr)
        return false;
      return fn(nullptr, IdentityPage(page));
    }

    static thread_local PteCache cache;
    return cached_walk(addr, vmid, cache, [&](const KfdProcess::PageTableEntry *pte) {
      if (pte) {
        if (pte->host_extents.empty())
          return false;
        return fn(pte, IdentityPage(nullptr));
      }
      if (passthrough_ && addr < kUserSpaceLimit) {
        auto *page = reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK);
        if (page == nullptr)
          return false;
        return fn(nullptr, IdentityPage(page));
      }
      return false;
    });
  }

  bool read_mapped(uint64_t addr, void *dst, size_t len, uint32_t vmid) const {
    const FaultDispatch fault_dispatch(*this);
    if ((addr & PAGE_MASK) + len > PAGE_SIZE)
      return false;
    std::memset(dst, 0, len);
    return with_page_mapping(
        addr, vmid, [&](const KfdProcess::PageTableEntry *pte, IdentityPage page) {
          const size_t access_begin = addr & PAGE_MASK;
          if (pte) {
            // Held across the span walk and the copies for the same reason
            // copy_mapped_span() holds it: a USERPTR-backed extent belongs to
            // the application, which can revoke it, and the interposer takes
            // this exclusively around the syscalls that do.
            const auto mapping_lease = rocjitsu::host_mapping_lock().lock_shared();
            size_t mapped_bytes = 0;
            // Attempted rather than pre-checked: an extent may describe USERPTR
            // memory the application can unmap or protect at any moment, and
            // proving otherwise before each access costs more than the access.
            // The guard turns the resulting host fault into this refusal.
            if (!rocjitsu::with_host_access_guard([&] {
                  mapped_bytes =
                      for_each_mapped_span(*pte, access_begin, len,
                                           [&](size_t value_offset, uint8_t *host_ptr,
                                               size_t span_size, const KfdProcess::HostExtent &) {
                                             std::memcpy(static_cast<uint8_t *>(dst) + value_offset,
                                                         host_ptr, span_size);
                                           });
                })) {
              // The destination was zeroed before the walk and whatever landed
              // before the fault is discarded with it: bytes read out of a page
              // that then vanished are not the memory the caller asked for.
              std::memset(dst, 0, len);
              note_rejected_identity_access(addr, vmid, guarded_fault_cause());
              return false;
            }
            if (mapped_bytes != len)
              note_clipped_mapped_access("read", addr, len, vmid);
            return true;
          }
          if (page.read(access_begin, dst, len))
            return true;
          note_rejected_identity_access(addr, vmid);
          return false;
        });
  }

  bool write_mapped(uint64_t addr, const void *src, size_t len, uint32_t vmid) {
    const FaultDispatch fault_dispatch(*this);
    if ((addr & PAGE_MASK) + len > PAGE_SIZE)
      return false;
    return with_page_mapping(
        addr, vmid, [&](const KfdProcess::PageTableEntry *pte, IdentityPage page) {
          const size_t access_begin = addr & PAGE_MASK;
          if (pte) {
            // Held across the span walk and the copies for the same reason
            // copy_mapped_span() holds it: a USERPTR-backed extent belongs to
            // the application, which can revoke it, and the interposer takes
            // this exclusively around the syscalls that do.
            const auto mapping_lease = rocjitsu::host_mapping_lock().lock_shared();
            size_t mapped_bytes = 0;
            if (!rocjitsu::with_host_access_guard([&] {
                  mapped_bytes = for_each_mapped_span(
                      *pte, access_begin, len,
                      [&](size_t value_offset, uint8_t *host_ptr, size_t span_size,
                          const KfdProcess::HostExtent &) {
                        std::memcpy(host_ptr, static_cast<const uint8_t *>(src) + value_offset,
                                    span_size);
                      });
                })) {
              // Whatever preceded the faulting byte has already been stored,
              // which is what hardware does when a transfer walks into a page
              // that is not there. The access is reported as faulted rather
              // than as the prefix that happened to land.
              note_rejected_identity_access(addr, vmid, guarded_fault_cause());
              return false;
            }
            if (mapped_bytes != len)
              note_clipped_mapped_access("write", addr, len, vmid);
            return true;
          }
          if (page.write(access_begin, src, len))
            return true;
          note_rejected_identity_access(addr, vmid, page.write_refusal_cause());
          return false;
        });
  }

  /// @brief Copy a page-bounded span in or out without exposing a bare pointer.
  ///
  /// @details A page-table span is memcpy'd from its extent. An identity span is
  /// moved by the kernel, so the check and the copy are the same operation and
  /// no unmap can slip between them.
  bool copy_mapped_span(uint64_t addr, void *bytes, size_t size, uint32_t vmid,
                        bool into_memory) const {
    const FaultDispatch fault_dispatch(*this);
    if (size == 0 || (addr & PAGE_MASK) + size > PAGE_SIZE)
      return false;
    return with_page_mapping(
        addr, vmid, [&](const KfdProcess::PageTableEntry *pte, IdentityPage page) {
          const size_t page_offset = addr & PAGE_MASK;
          if (pte) {
            // Held across the extent lookup, the check and the copy. A PTE may
            // describe USERPTR memory, which the application still owns and can
            // mprotect or munmap; the interposer takes this exclusively around
            // those syscalls, so the lease is what stops the extent from being
            // revoked between being validated here and being copied below.
            const auto mapping_lease = rocjitsu::host_mapping_lock().lock_shared();
            const auto *extent = host_extent_at(*pte, page_offset);
            if (!extent ||
                size > extent->host_backed_bytes - (page_offset - extent->gpu_page_offset))
              return false;
            auto *candidate = extent->host_ptr + (page_offset - extent->gpu_page_offset);
            if (addressable_prefix(candidate, size) != size)
              return false;
            if (!rocjitsu::with_host_access_guard([&] {
                  if (into_memory)
                    std::memcpy(candidate, bytes, size);
                  else
                    std::memcpy(bytes, candidate, size);
                })) {
              note_rejected_identity_access(addr, vmid, guarded_fault_cause());
              return false;
            }
            return true;
          }
          if (addr >= kUserSpaceLimit || size > kUserSpaceLimit - addr)
            return false;
          const bool moved = into_memory ? page.write(page_offset, bytes, size)
                                         : page.read(page_offset, bytes, size);
          if (!moved)
            note_rejected_identity_access(addr, vmid,
                                          into_memory ? page.write_refusal_cause()
                                                      : MemoryFaultCause::NotPresent);
          return moved;
        });
  }

  /// @brief Read a mapped span into @p dst without ever exposing a bare pointer.
  bool copy_from_mapped(uint64_t addr, void *dst, size_t size, uint32_t vmid) const {
    return copy_mapped_span(addr, dst, size, vmid, /*into_memory=*/false);
  }

  /// @brief Write @p src into a mapped span without ever exposing a bare pointer.
  bool copy_to_mapped(uint64_t addr, const void *src, size_t size, uint32_t vmid) const {
    return copy_mapped_span(addr, const_cast<void *>(src), size, vmid, /*into_memory=*/true);
  }

  uint8_t *translate(uint64_t addr, uint32_t vmid, size_t size) const {
    const FaultDispatch fault_dispatch(*this);
    if (size == 0 || (addr & PAGE_MASK) + size > PAGE_SIZE)
      return nullptr;
    uint8_t *host_ptr = nullptr;
    with_page_mapping(addr, vmid, [&](const KfdProcess::PageTableEntry *pte, IdentityPage page) {
      const size_t page_offset = addr & PAGE_MASK;
      if (pte) {
        const auto *extent = host_extent_at(*pte, page_offset);
        if (extent && size <= extent->host_backed_bytes - (page_offset - extent->gpu_page_offset)) {
          auto *candidate = extent->host_ptr + (page_offset - extent->gpu_page_offset);
          if (addressable_prefix(candidate, size) == size)
            host_ptr = candidate;
        }
        // A page-table entry that cannot cover the span is not a mapping that
        // has yet to appear -- the mapping is here and it does not reach.
        // Sub-page and disjoint extents are supported, so this is reachable for
        // a control operand wider than its backing, and a caller that reads it
        // as "not ready" waits for a mapping that already arrived.
        if (host_ptr == nullptr)
          note_rejected_identity_access(addr, vmid);
        return host_ptr != nullptr;
      }
      if (addr >= kUserSpaceLimit || size > kUserSpaceLimit - addr)
        return false;
      // The one path that must surrender a bare pointer, so the probe is
      // explicit here rather than folded into an operation.
      host_ptr = page.read_valid_pointer(page_offset, size);
      if (host_ptr == nullptr)
        note_rejected_identity_access(addr, vmid);
      return host_ptr != nullptr;
    });
    return host_ptr;
  }

  /// @brief Whether another process, not sparse storage, owns this VMID's memory.
  /// @details Either conduit counts: the debugger-authorized /proc/<pid>/mem
  /// descriptor and the process_vm_readv() path both reach memory this
  /// simulator does not own, and a refusal from either is a real failure rather
  /// than a cue to fall back to storage the owner cannot see.
  bool has_client_backing(uint32_t vmid) const {
    std::shared_lock lk(vmid_mutex_);
    auto it = vmid_table_.find(vmid);
    return it != vmid_table_.end() &&
           (it->second.client_pid > 0 || it->second.client_mem_fd.get() >= 0);
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
  mutable std::atomic<uint64_t> rejected_identity_accesses_{0};
  std::atomic<MemoryFaultReporter *> fault_reporter_{nullptr};
  inline static thread_local uint64_t tls_identity_faults = 0;
  inline static thread_local uint64_t tls_clipped_accesses = 0;

  inline static thread_local PendingFault tls_pending_fault{};

  bool passthrough_ = false;
};

} // namespace amdgpu
} // namespace rocjitsu

#undef RJ_GPU_MEMORY_WITH_ASAN

#endif // ROCJITSU_VM_AMDGPU_GPU_MEMORY_H_
