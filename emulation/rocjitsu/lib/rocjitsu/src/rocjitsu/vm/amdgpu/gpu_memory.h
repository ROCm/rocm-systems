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
  explicit GpuMemory(std::string name) : simdojo::SparseMemory(std::move(name)) {
    cpl_ = add_port(std::make_unique<simdojo::Port>("cpl", 0, this, simdojo::PortDirection::IN,
                                                    simdojo::PortProtocol::MEMORY));
    cpl_->recv_event()->set_handler([this](simdojo::Tick, simdojo::Message *msg) {
      auto &hdr = msg->header();
      auto *data = reinterpret_cast<uint8_t *>(msg->payload());
      if (hdr.op == simdojo::MessageOp::READ) {
        for (uint32_t i = 0; i < hdr.size_bytes; ++i)
          data[i] = read8(hdr.addr + i, hdr.vmid);
      } else if (hdr.op == simdojo::MessageOp::WRITE) {
        for (uint32_t i = 0; i < hdr.size_bytes; ++i)
          write8(hdr.addr + i, data[i], hdr.vmid);
      }
      hdr.op = simdojo::MessageOp::RESPONSE;
    });
  }

  simdojo::Port *cpl_port() { return cpl_; }

  /// @brief Register a process's page table in the VMID table.
  void register_process(uint32_t pid, KfdProcess::PageTable *pt, std::shared_mutex *mu) {
    util::Logger::cp("VMID_REG pid=", pid, " mem=0x", std::hex, reinterpret_cast<uintptr_t>(this),
                     std::dec, " pt_size=", pt->size());
    std::unique_lock lk(vmid_mutex_);
    vmid_table_[pid] = {pt, mu};
  }

  /// @brief Unregister a process from the VMID table.
  void unregister_process(uint32_t pid) {
    util::Logger::cp("VMID_UNREG pid=", pid, " mem=0x", std::hex, reinterpret_cast<uintptr_t>(this),
                     std::dec);
    std::unique_lock lk(vmid_mutex_);
    vmid_table_.erase(pid);
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
    {
      std::shared_lock lk(vmid_mutex_);
      auto it = vmid_table_.find(vmid);
      if (it != vmid_table_.end()) {
        auto &entry = it->second;
        std::shared_lock pt_lk(*entry.mutex);
        auto pt_it = entry.page_table->find(addr >> PAGE_SHIFT);
        if (pt_it != entry.page_table->end())
          return pt_it->second.mtype;
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
  struct VmidEntry {
    KfdProcess::PageTable *page_table = nullptr;
    std::shared_mutex *mutex = nullptr;
    pid_t client_pid = 0;
  };

  uint8_t *translate(uint64_t addr, uint32_t vmid) const {
    if (vmid == 0)
      return passthrough_ ? reinterpret_cast<uint8_t *>(addr & ~PAGE_MASK) : nullptr;
    {
      std::shared_lock lk(vmid_mutex_);
      auto it = vmid_table_.find(vmid);
      if (it != vmid_table_.end()) {
        auto &entry = it->second;
        std::shared_lock pt_lk(*entry.mutex);
        auto pt_it = entry.page_table->find(addr >> PAGE_SHIFT);
        if (pt_it != entry.page_table->end())
          return pt_it->second.host_ptr;
      }
    }
    static constexpr uint64_t kUserSpaceLimit = 0x800000000000ULL;
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
  mutable std::shared_mutex vmid_mutex_;
  std::unordered_map<uint32_t, VmidEntry> vmid_table_;
  bool passthrough_ = false;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_GPU_MEMORY_H_
