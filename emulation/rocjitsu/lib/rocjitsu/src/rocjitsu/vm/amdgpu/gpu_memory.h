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
  explicit GpuMemory(std::string name) : simdojo::SparseMemory(std::move(name)) {
    cpl_ = add_port(std::make_unique<simdojo::Port>("cpl", 0, this, simdojo::PortDirection::IN,
                                                    simdojo::PortProtocol::MEMORY));
    cpl_->recv_event()->set_handler([this](simdojo::Tick, simdojo::Message *msg) {
      auto &hdr = msg->header();
      auto *data = reinterpret_cast<uint8_t *>(msg->payload());
      if (hdr.op == simdojo::MessageOp::READ) {
        for (uint32_t i = 0; i < hdr.size_bytes; ++i)
          data[i] = read8(hdr.addr + i);
      } else if (hdr.op == simdojo::MessageOp::WRITE) {
        for (uint32_t i = 0; i < hdr.size_bytes; ++i)
          write8(hdr.addr + i, data[i]);
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
    vmid_table_[pid] = {pt, mu, generation};
  }

  /// @brief Unregister a process from the VMID table.
  void unregister_process(uint32_t pid) {
    util::Logger::cp("VMID_UNREG pid=", pid, " mem=0x", std::hex, reinterpret_cast<uintptr_t>(this),
                     std::dec);
    std::unique_lock lk(vmid_mutex_);
    vmid_table_.erase(pid);
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
      uint32_t vmid = 0;
      uint64_t page_key = 0;
      uint64_t generation = 0;
      Mtype mtype = Mtype::RW;
      KfdProcess::PageTable *page_table = nullptr;
      std::shared_mutex *mutex = nullptr;
      std::atomic<uint64_t> *generation_ptr = nullptr;
    };
    static thread_local MtypeCache cache;

    if (cache.memory == this && cache.vmid == vmid && cache.generation_ptr) {
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

    {
      std::shared_lock lk(vmid_mutex_);
      auto it = vmid_table_.find(vmid);
      if (it != vmid_table_.end()) {
        auto &entry = it->second;
        uint64_t generation =
            entry.generation ? entry.generation->load(std::memory_order_acquire) : 0;
        std::shared_lock pt_lk(*entry.mutex);
        auto pt_it = entry.page_table->find(page_key);
        Mtype mtype = pt_it != entry.page_table->end() ? pt_it->second.mtype : Mtype::RW;
        cache = {this,        vmid,
                 page_key,    generation,
                 mtype,       entry.page_table,
                 entry.mutex, entry.generation};
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
        for (uint32_t i = 0; i < chunk; ++i)
          dst[copied + i] = SparseMemory::read8(ea + i);
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
    return SparseMemory::read16(addr);
  }

  uint32_t read32(uint64_t addr, uint32_t vmid = 0) const {
    if (auto *p = translate(addr, vmid); p && (addr & PAGE_MASK) + 4 <= PAGE_SIZE) {
      uint32_t val;
      std::memcpy(&val, p + (addr & PAGE_MASK), 4);
      return val;
    }
    return SparseMemory::read32(addr);
  }

  uint64_t read64(uint64_t addr, uint32_t vmid = 0) const {
    if (auto *p = translate(addr, vmid); p && (addr & PAGE_MASK) + 8 <= PAGE_SIZE) {
      uint64_t val;
      std::memcpy(&val, p + (addr & PAGE_MASK), 8);
      return val;
    }
    return SparseMemory::read64(addr);
  }

  void write8(uint64_t addr, uint8_t val, uint32_t vmid = 0) {
    if (auto *p = translate(addr, vmid)) {
      p[addr & PAGE_MASK] = val;
      return;
    }
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
        for (uint32_t i = 0; i < chunk; ++i)
          SparseMemory::write8(ea + i, src[copied + i]);
      }
      copied += chunk;
    }
  }

  void write16(uint64_t addr, uint16_t val, uint32_t vmid = 0) {
    if (auto *p = translate(addr, vmid); p && (addr & PAGE_MASK) + 2 <= PAGE_SIZE) {
      std::memcpy(p + (addr & PAGE_MASK), &val, 2);
      return;
    }
    SparseMemory::write16(addr, val);
  }

  void write32(uint64_t addr, uint32_t val, uint32_t vmid = 0) {
    if (auto *p = translate(addr, vmid); p && (addr & PAGE_MASK) + 4 <= PAGE_SIZE) {
      std::memcpy(p + (addr & PAGE_MASK), &val, 4);
      return;
    }
    SparseMemory::write32(addr, val);
  }

  void write64(uint64_t addr, uint64_t val, uint32_t vmid = 0) {
    if (auto *p = translate(addr, vmid); p && (addr & PAGE_MASK) + 8 <= PAGE_SIZE) {
      std::memcpy(p + (addr & PAGE_MASK), &val, 8);
      return;
    }
    SparseMemory::write64(addr, val);
  }

private:
  // Thread-local translation caches keep these pointers only while the simulated
  // process is active; driver teardown unregisters after GPU work is drained.
  struct VmidEntry {
    KfdProcess::PageTable *page_table = nullptr;
    std::shared_mutex *mutex = nullptr;
    std::atomic<uint64_t> *generation = nullptr;
  };

  uint8_t *translate(uint64_t addr, uint32_t vmid) const {
    if (vmid == 0)
      return passthrough_ ? reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK) : nullptr;
    const uint64_t page_key = addr >> PAGE_SHIFT;
    struct TranslationCache {
      const GpuMemory *memory = nullptr;
      uint32_t vmid = 0;
      uint64_t page_key = 0;
      uint64_t generation = 0;
      uint8_t *host_ptr = nullptr;
      KfdProcess::PageTable *page_table = nullptr;
      std::shared_mutex *mutex = nullptr;
      std::atomic<uint64_t> *generation_ptr = nullptr;
    };
    static thread_local TranslationCache cache;

    if (cache.memory == this && cache.vmid == vmid && cache.generation_ptr) {
      uint64_t generation = cache.generation_ptr->load(std::memory_order_acquire);
      if (cache.generation == generation && cache.page_key == page_key)
        return cache.host_ptr;
      if (cache.generation == generation && cache.page_table && cache.mutex) {
        std::shared_lock pt_lk(*cache.mutex);
        auto pt_it = cache.page_table->find(page_key);
        uint8_t *host_ptr = nullptr;
        if (pt_it != cache.page_table->end())
          host_ptr = pt_it->second.host_ptr;
        else if (passthrough_)
          host_ptr = reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK);
        cache.page_key = page_key;
        cache.host_ptr = host_ptr;
        return host_ptr;
      }
    }

    {
      std::shared_lock lk(vmid_mutex_);
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
        else if (passthrough_)
          host_ptr = reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK);
        cache = {this,        vmid,
                 page_key,    generation,
                 host_ptr,    entry.page_table,
                 entry.mutex, entry.generation};
        return host_ptr;
      }
    }
    static constexpr uint64_t kUserSpaceLimit = 0x800000000000ULL;
    if (passthrough_ && addr < kUserSpaceLimit)
      return reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK);
    return nullptr;
  }

  simdojo::Port *cpl_ = nullptr;
  mutable std::shared_mutex vmid_mutex_;
  std::unordered_map<uint32_t, VmidEntry> vmid_table_;
  bool passthrough_ = false;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_GPU_MEMORY_H_
