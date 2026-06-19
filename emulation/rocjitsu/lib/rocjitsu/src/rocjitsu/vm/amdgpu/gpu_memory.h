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
#include <atomic>
#include <cstring>
#include <format>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <sys/uio.h>
#include <unordered_map>
#include <utility>

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
        instance_id_(next_instance_id_.fetch_add(1, std::memory_order_relaxed)) {
    cpl_ = add_port(std::make_unique<simdojo::Port>("cpl", 0, this, simdojo::PortDirection::IN,
                                                    simdojo::PortProtocol::MEMORY));
    cpl_->recv_event()->set_handler([this](simdojo::Tick, simdojo::Message *msg) {
      auto &hdr = msg->header();
      auto *data = reinterpret_cast<uint8_t *>(msg->payload());
      if (hdr.op == simdojo::MessageOp::READ) {
        read_block(hdr.addr, data, hdr.size_bytes, hdr.vmid);
      } else if (hdr.op == simdojo::MessageOp::WRITE) {
        write_block(hdr.addr, data, hdr.size_bytes, hdr.vmid);
      }
      hdr.op = simdojo::MessageOp::RESPONSE;
    });
  }

  simdojo::Port *cpl_port() { return cpl_; }

  /// @brief Register a process's page table in the VMID table.
  void register_process(uint32_t pid, KfdProcess::PageTable *pt, std::shared_mutex *mu,
                        std::atomic<uint64_t> *generation) {
    util::Logger::cp("VMID_REG pid=", pid, " mem=0x", std::hex, reinterpret_cast<uintptr_t>(this),
                     std::dec, " pt_size=", pt->size());
    std::unique_lock lk(vmid_mutex_);
    vmid_table_[pid] = {pt, mu, 0, generation};
    vmid_table_generation_.fetch_add(1, std::memory_order_release);
  }

  /// @brief Unregister a process from the VMID table.
  void unregister_process(uint32_t pid) {
    util::Logger::cp("VMID_UNREG pid=", pid, " mem=0x", std::hex, reinterpret_cast<uintptr_t>(this),
                     std::dec);
    std::unique_lock lk(vmid_mutex_);
    auto it = vmid_table_.find(pid);
    if (it == vmid_table_.end())
      return;
    if (it->second.generation)
      it->second.generation->fetch_add(1, std::memory_order_release);
    vmid_table_.erase(it);
    vmid_table_generation_.fetch_add(1, std::memory_order_release);
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

  /// @brief Resolve a GPU VA to a host pointer via the given VMID's page table.
  uint8_t *resolve_host_ptr(uint64_t addr, uint32_t vmid = 0) const {
    return translate(addr, vmid);
  }

  /// @brief Look up PTE MTYPE for a GPU VA in the given VMID's page table.
  Mtype pte_mtype(uint64_t addr, uint32_t vmid = 0) const {
    if (vmid == 0)
      return Mtype::RW;
    const uint64_t page_key = addr >> PAGE_SHIFT;
    struct MtypeCache {
      const GpuMemory *memory = nullptr;
      uint64_t memory_instance = 0;
      uint32_t vmid = 0;
      uint64_t table_generation = 0;
      uint64_t page_key = 0;
      uint64_t generation = 0;
      Mtype mtype = Mtype::RW;
      KfdProcess::PageTable *page_table = nullptr;
      std::shared_mutex *mutex = nullptr;
      std::atomic<uint64_t> *generation_ptr = nullptr;
    };
    static thread_local MtypeCache cache;

    {
      std::shared_lock lk(vmid_mutex_);
      uint64_t table_generation = vmid_table_generation_.load(std::memory_order_acquire);
      if (cache.memory == this && cache.memory_instance == instance_id_ && cache.vmid == vmid &&
          cache.table_generation == table_generation && cache.generation_ptr) {
        uint64_t generation = cache.generation_ptr->load(std::memory_order_acquire);
        if (cache.generation == generation && cache.page_key == page_key)
          return cache.mtype;
        if (cache.generation == generation && cache.page_table && cache.mutex) {
          std::shared_lock pt_lk(*cache.mutex);
          auto pt_it = cache.page_table->find(page_key);
          cache.page_key = page_key;
          cache.mtype = pt_it != cache.page_table->end() ? pt_it->second.mtype : Mtype::RW;
          return cache.mtype;
        }
      }

      auto it = vmid_table_.find(vmid);
      if (it != vmid_table_.end()) {
        auto &entry = it->second;
        uint64_t generation =
            entry.generation ? entry.generation->load(std::memory_order_acquire) : 0;
        std::shared_lock pt_lk(*entry.mutex);
        auto pt_it = entry.page_table->find(page_key);
        Mtype mtype = pt_it != entry.page_table->end() ? pt_it->second.mtype : Mtype::RW;
        cache = {this,  instance_id_,     vmid,        table_generation, page_key, generation,
                 mtype, entry.page_table, entry.mutex, entry.generation};
        return mtype;
      }
    }
    return Mtype::RW;
  }

  uint32_t fetch32(uint64_t addr, uint32_t vmid = 0) const { return read32(addr, vmid); }

  uint8_t *translate_debug(uint64_t addr, uint32_t vmid) const { return translate(addr, vmid); }

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
    uint64_t lo = UINT64_MAX, hi = 0;
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
    if (auto *p = translate(addr, vmid))
      return p[addr & PAGE_MASK];
    uint8_t val = 0;
    if (vmid > 0 && read_client_memory(addr, &val, 1, vmid))
      return val;
    return SparseMemory::read8(addr);
  }

  void read_block(uint64_t addr, uint8_t *dst, uint32_t size, uint32_t vmid = 0) const {
    uint32_t copied = 0;
    while (copied < size) {
      const uint64_t ea = addr + copied;
      const size_t page_offset = ea & PAGE_MASK;
      const uint32_t chunk =
          std::min<uint32_t>(size - copied, static_cast<uint32_t>(PAGE_SIZE - page_offset));
      if (auto *p = translate(ea, vmid)) {
        std::memcpy(dst + copied, p + page_offset, chunk);
      } else {
        SparseMemory::read_block(ea, dst + copied, chunk);
      }
      copied += chunk;
    }
  }

  uint16_t read16(uint64_t addr, uint32_t vmid = 0) const {
    if (auto *p = translate(addr, vmid); p && (addr & PAGE_MASK) + 2 <= PAGE_SIZE) {
      uint16_t val;
      std::memcpy(&val, p + (addr & PAGE_MASK), 2);
      return val;
    }
    uint16_t val = 0;
    if (vmid > 0 && read_client_memory(addr, &val, 2, vmid))
      return val;
    return SparseMemory::read16(addr);
  }

  uint32_t read32(uint64_t addr, uint32_t vmid = 0) const {
    if (auto *p = translate(addr, vmid); p && (addr & PAGE_MASK) + 4 <= PAGE_SIZE) {
      uint32_t val;
      std::memcpy(&val, p + (addr & PAGE_MASK), 4);
      return val;
    }
    uint32_t val = 0;
    if (vmid > 0 && read_client_memory(addr, &val, 4, vmid))
      return val;
    return SparseMemory::read32(addr);
  }

  uint64_t read64(uint64_t addr, uint32_t vmid = 0) const {
    if (auto *p = translate(addr, vmid); p && (addr & PAGE_MASK) + 8 <= PAGE_SIZE) {
      uint64_t val;
      std::memcpy(&val, p + (addr & PAGE_MASK), 8);
      return val;
    }
    uint64_t val = 0;
    if (vmid > 0 && read_client_memory(addr, &val, 8, vmid))
      return val;
    return SparseMemory::read64(addr);
  }

  void write8(uint64_t addr, uint8_t val, uint32_t vmid = 0) {
    if (auto *p = translate(addr, vmid)) {
      p[addr & PAGE_MASK] = val;
      return;
    }
    if (vmid > 0 && write_client_memory(addr, &val, 1, vmid))
      return;
    SparseMemory::write8(addr, val);
  }

  void write_block(uint64_t addr, const uint8_t *src, uint32_t size, uint32_t vmid = 0) {
    uint32_t copied = 0;
    while (copied < size) {
      const uint64_t ea = addr + copied;
      const size_t page_offset = ea & PAGE_MASK;
      const uint32_t chunk =
          std::min<uint32_t>(size - copied, static_cast<uint32_t>(PAGE_SIZE - page_offset));
      if (auto *p = translate(ea, vmid)) {
        std::memcpy(p + page_offset, src + copied, chunk);
      } else {
        SparseMemory::write_block(ea, src + copied, chunk);
      }
      copied += chunk;
    }
  }

  void write16(uint64_t addr, uint16_t val, uint32_t vmid = 0) {
    if (auto *p = translate(addr, vmid); p && (addr & PAGE_MASK) + 2 <= PAGE_SIZE) {
      std::memcpy(p + (addr & PAGE_MASK), &val, 2);
      return;
    }
    if (vmid > 0 && write_client_memory(addr, &val, 2, vmid))
      return;
    SparseMemory::write16(addr, val);
  }

  void write32(uint64_t addr, uint32_t val, uint32_t vmid = 0) {
    if (auto *p = translate(addr, vmid); p && (addr & PAGE_MASK) + 4 <= PAGE_SIZE) {
      std::memcpy(p + (addr & PAGE_MASK), &val, 4);
      return;
    }
    if (vmid > 0 && write_client_memory(addr, &val, 4, vmid))
      return;
    SparseMemory::write32(addr, val);
  }

  void write64(uint64_t addr, uint64_t val, uint32_t vmid = 0) {
    if (auto *p = translate(addr, vmid); p && (addr & PAGE_MASK) + 8 <= PAGE_SIZE) {
      std::memcpy(p + (addr & PAGE_MASK), &val, 8);
      return;
    }
    if (vmid > 0 && write_client_memory(addr, &val, 8, vmid))
      return;
    SparseMemory::write64(addr, val);
  }

private:
  static constexpr uint64_t kUserSpaceLimit = 0x800000000000ULL;

  // Thread-local translation caches keep these pointers only while the simulated
  // process is active; driver teardown unregisters after GPU work is drained.
  struct VmidEntry {
    KfdProcess::PageTable *page_table = nullptr;
    std::shared_mutex *mutex = nullptr;
    pid_t client_pid = 0;
    std::atomic<uint64_t> *generation = nullptr;
  };

  uint8_t *translate(uint64_t addr, uint32_t vmid) const {
    if (vmid == 0)
      return passthrough_ ? reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK) : nullptr;
    const uint64_t page_key = addr >> PAGE_SHIFT;
    struct TranslationCache {
      const GpuMemory *memory = nullptr;
      uint64_t memory_instance = 0;
      uint32_t vmid = 0;
      uint64_t table_generation = 0;
      uint64_t page_key = 0;
      uint64_t generation = 0;
      uint8_t *host_ptr = nullptr;
      KfdProcess::PageTable *page_table = nullptr;
      std::shared_mutex *mutex = nullptr;
      std::atomic<uint64_t> *generation_ptr = nullptr;
    };
    static thread_local TranslationCache cache;

    {
      std::shared_lock lk(vmid_mutex_);
      uint64_t table_generation = vmid_table_generation_.load(std::memory_order_acquire);
      if (cache.memory == this && cache.memory_instance == instance_id_ && cache.vmid == vmid &&
          cache.table_generation == table_generation && cache.generation_ptr) {
        uint64_t generation = cache.generation_ptr->load(std::memory_order_acquire);
        if (cache.generation == generation && cache.page_key == page_key)
          return cache.host_ptr;
        if (cache.generation == generation && cache.page_table && cache.mutex) {
          std::shared_lock pt_lk(*cache.mutex);
          auto pt_it = cache.page_table->find(page_key);
          uint8_t *host_ptr = nullptr;
          if (pt_it != cache.page_table->end())
            host_ptr = pt_it->second.host_ptr;
          else if (passthrough_ && addr < kUserSpaceLimit)
            host_ptr = reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK);
          cache.page_key = page_key;
          cache.host_ptr = host_ptr;
          return host_ptr;
        }
      }

      auto it = vmid_table_.find(vmid);
      if (it != vmid_table_.end()) {
        auto &entry = it->second;
        uint64_t generation =
            entry.generation ? entry.generation->load(std::memory_order_acquire) : 0;
        std::shared_lock pt_lk(*entry.mutex);
        auto pt_it = entry.page_table->find(page_key);
        uint8_t *host_ptr = nullptr;
        if (pt_it != entry.page_table->end())
          host_ptr = pt_it->second.host_ptr;
        else if (passthrough_ && addr < kUserSpaceLimit)
          host_ptr = reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK);
        cache = {this,     instance_id_,     vmid,        table_generation, page_key, generation,
                 host_ptr, entry.page_table, entry.mutex, entry.generation};
        return host_ptr;
      }
    }
    if (passthrough_ && addr < kUserSpaceLimit)
      return reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK);
    return nullptr;
  }

  pid_t client_pid_for_vmid(uint32_t vmid) const {
    std::shared_lock lk(vmid_mutex_);
    auto it = vmid_table_.find(vmid);
    return (it != vmid_table_.end()) ? it->second.client_pid : 0;
  }

  bool read_client_memory(uint64_t addr, void *dst, size_t len, uint32_t vmid) const {
    pid_t pid = client_pid_for_vmid(vmid);
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
    pid_t pid = client_pid_for_vmid(vmid);
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
  inline static std::atomic<uint64_t> next_instance_id_{1};
  const uint64_t instance_id_;
  mutable std::shared_mutex vmid_mutex_;
  std::unordered_map<uint32_t, VmidEntry> vmid_table_;
  mutable std::atomic<uint64_t> vmid_table_generation_ = 1;
  bool passthrough_ = false;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_GPU_MEMORY_H_
