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
#include <cstring>
#include <format>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
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
        read_block(hdr.addr, std::span<uint8_t>(data, hdr.size_bytes), hdr.vmid);
      } else if (hdr.op == simdojo::MessageOp::WRITE) {
        write_block(hdr.addr, std::span<const uint8_t>(data, hdr.size_bytes), hdr.vmid);
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

  /// @brief Read a contiguous range from simulated GPU memory.
  /// @details Handles each page through mapped host memory, client memory, or
  /// sparse backing memory.
  void read_block(uint64_t addr, std::span<uint8_t> dst, uint32_t vmid = 0) const {
    for_each_page_chunk(addr, dst.size(), [&](uint64_t ea, size_t offset, size_t chunk) {
      auto out = dst.subspan(offset, chunk);
      if (auto *p = translate(ea, vmid)) {
        std::memcpy(out.data(), p + (ea & PAGE_MASK), chunk);
        return;
      }
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
      if (auto *p = translate(ea, vmid)) {
        std::memcpy(p + (ea & PAGE_MASK), in.data(), chunk);
        return;
      }
      if (vmid > 0 && write_client_memory(ea, in.data(), chunk, vmid))
        return;
      for (size_t i = 0; i < chunk; ++i)
        simdojo::SparseMemory::write8(ea + i, in[i]);
    });
  }

  uint8_t *translate_debug(uint64_t addr, uint32_t vmid) const { return translate(addr, vmid); }

  /// @brief Find the contiguous host range containing a VMID-scoped GPU VA.
  /// @details KFD dispatches use per-process page tables. Kernel-symbol
  /// resolution needs a daemon-accessible host pointer range so it can scan
  /// backward from the kernel descriptor to the loaded ELF header.
  std::pair<uint64_t, uint64_t> find_host_range(uint64_t addr, uint32_t vmid) const {
    if (vmid == 0) {
      auto *host = translate(addr, vmid);
      if (!host)
        return {0, 0};
      return {reinterpret_cast<uint64_t>(host), PAGE_SIZE};
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

    uint64_t first_page = page;
    uint8_t *first_host_page = page_entry->second.host_ptr;
    while (first_page > 0) {
      auto previous_page_entry = entry.page_table->find(first_page - 1);
      if (previous_page_entry == entry.page_table->end() ||
          previous_page_entry->second.host_ptr + PAGE_SIZE != first_host_page)
        break;
      --first_page;
      first_host_page = previous_page_entry->second.host_ptr;
    }

    uint64_t last_page = page;
    uint8_t *last_host_page = page_entry->second.host_ptr;
    while (true) {
      auto next_page_entry = entry.page_table->find(last_page + 1);
      if (next_page_entry == entry.page_table->end() ||
          next_page_entry->second.host_ptr != last_host_page + PAGE_SIZE)
        break;
      ++last_page;
      last_host_page = next_page_entry->second.host_ptr;
    }

    const uint64_t range_size = ((last_page - first_page) + 1) << PAGE_SHIFT;
    return {reinterpret_cast<uint64_t>(first_host_page), range_size};
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
  template <typename F> static void for_each_page_chunk(uint64_t addr, size_t len, F &&fn) {
    size_t offset = 0;
    while (offset < len) {
      const uint64_t ea = addr + offset;
      const size_t chunk = std::min(len - offset, PAGE_SIZE - (ea & PAGE_MASK));
      fn(ea, offset, chunk);
      offset += chunk;
    }
  }

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
