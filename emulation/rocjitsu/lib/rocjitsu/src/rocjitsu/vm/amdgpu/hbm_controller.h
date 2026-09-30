// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "simdojo/sim/component.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// @brief High Bandwidth Memory (HBM) controller — the lowest level of the memory hierarchy.
///
/// @details Wraps GpuMemory (SparseMemory) and services memory requests
/// received on its completer port. In a timing model this would model HBM
/// channel latency, bandwidth, and bank conflicts. The current functional
/// implementation is synchronous and immediate.
class HbmController : public simdojo::Component {
public:
  explicit HbmController(GpuMemory *memory, GpuVm *gpu_vm = nullptr)
      : simdojo::Component("hbm"), memory_(memory), gpu_vm_(gpu_vm) {
    cpl_ = add_port(std::make_unique<simdojo::Port>("cpl", 0, this, simdojo::PortDirection::IN,
                                                    simdojo::PortProtocol::MEMORY));
    install_cpl_handler();
  }

  HbmController(std::string name, GpuMemory *memory, GpuVm *gpu_vm = nullptr)
      : simdojo::Component(std::move(name)), memory_(memory), gpu_vm_(gpu_vm) {
    cpl_ = add_port(std::make_unique<simdojo::Port>("cpl", 0, this, simdojo::PortDirection::IN,
                                                    simdojo::PortProtocol::MEMORY));
    install_cpl_handler();
  }

  /// @brief Return the default completer port (receives requests from IOD/MSC).
  /// @returns Pointer to the completer port.
  simdojo::Port *cpl_port() { return cpl_; }

  /// @brief Create an additional completer port (for multiple upstream connections).
  /// @param src_name Name suffix for the port.
  /// @returns Pointer to the newly created completer port.
  simdojo::Port *create_cpl_port(const std::string &src_name) {
    auto port_id = static_cast<simdojo::PortID>(cpl_ports_.size() + 1);
    auto port =
        std::make_unique<simdojo::Port>("cpl_" + src_name, port_id, this,
                                        simdojo::PortDirection::IN, simdojo::PortProtocol::MEMORY);
    auto *raw = add_port(std::move(port));
    install_handler(raw);
    cpl_ports_.push_back(raw);
    return raw;
  }

  VmAccessOutcome read(uint64_t addr, uint8_t *dst, uint32_t size, uint32_t vmid = 0) {
    if (vmid != 0) {
      const std::optional<GpuVmAccess> access =
          gpu_vm_ != nullptr ? gpu_vm_->snapshot_vmid(vmid) : std::nullopt;
      return access ? access->read(addr,
                                   std::span<std::byte>(reinterpret_cast<std::byte *>(dst), size))
                    : VmAccessOutcome::Faulted;
    }
    if (memory_ == nullptr)
      return VmAccessOutcome::Unavailable;
    memory_->read_block(addr, std::span<uint8_t>(dst, size));
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome write(uint64_t addr, const uint8_t *src, uint32_t size, uint32_t vmid = 0) {
    if (vmid != 0) {
      const std::optional<GpuVmAccess> access =
          gpu_vm_ != nullptr ? gpu_vm_->snapshot_vmid(vmid) : std::nullopt;
      return access ? access->write(addr, std::span<const std::byte>(
                                              reinterpret_cast<const std::byte *>(src), size))
                    : VmAccessOutcome::Faulted;
    }
    if (memory_ == nullptr)
      return VmAccessOutcome::Unavailable;
    memory_->write_block(addr, std::span<const uint8_t>(src, size));
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome atomic_modify(uint64_t addr, uint32_t size,
                                const simdojo::MemoryAtomicMutation &mutation, uint32_t vmid = 0) {
    if (vmid != 0) {
      const std::optional<GpuVmAccess> access =
          gpu_vm_ != nullptr ? gpu_vm_->snapshot_vmid(vmid) : std::nullopt;
      return access ? access->atomic_modify(addr, size, mutation) : VmAccessOutcome::Faulted;
    }
    if (memory_ == nullptr)
      return VmAccessOutcome::Unavailable;
    const bool modified = memory_->atomic_modify(addr, size, [&](uint8_t *target) {
      mutation(std::span<std::byte>(reinterpret_cast<std::byte *>(target), size));
    });
    return modified ? VmAccessOutcome::Complete : VmAccessOutcome::Malformed;
  }

  /// @brief Read a 32-bit dword (little-endian).
  uint32_t read32(uint64_t addr, uint32_t vmid = 0) {
    uint32_t value = 0;
    (void)read(addr, reinterpret_cast<uint8_t *>(&value), sizeof(value), vmid);
    return value;
  }

  /// @brief Write a 32-bit dword (little-endian).
  VmAccessOutcome write32(uint64_t addr, uint32_t val, uint32_t vmid = 0) {
    return write(addr, reinterpret_cast<const uint8_t *>(&val), sizeof(val), vmid);
  }

  /// @brief Direct access to the underlying GpuMemory.
  GpuMemory *memory() const { return memory_; }

  /// @brief Set (or replace) the underlying GpuMemory.
  ///
  /// Used by the config loader for deferred initialization.
  /// @param memory New GPU memory (not owned).
  void set_memory(GpuMemory *memory) { memory_ = memory; }
  void set_gpu_vm(GpuVm *gpu_vm) { gpu_vm_ = gpu_vm; }

private:
  static simdojo::MessageStatus message_status(VmAccessOutcome outcome) {
    switch (outcome) {
    case VmAccessOutcome::Complete:
      return simdojo::MessageStatus::Complete;
    case VmAccessOutcome::Unavailable:
      return simdojo::MessageStatus::Unavailable;
    case VmAccessOutcome::Faulted:
      return simdojo::MessageStatus::Faulted;
    case VmAccessOutcome::Malformed:
      return simdojo::MessageStatus::Malformed;
    }
    return simdojo::MessageStatus::Malformed;
  }

  void install_handler(simdojo::Port *port) {
    port->recv_event()->set_handler([this](simdojo::Tick, simdojo::Message *msg) {
      auto &hdr = msg->header();
      auto *data = reinterpret_cast<uint8_t *>(msg->payload());
      VmAccessOutcome outcome = VmAccessOutcome::Malformed;
      if (hdr.op == simdojo::MessageOp::READ)
        outcome = read(hdr.addr, data, hdr.size_bytes, hdr.vmid);
      else if (hdr.op == simdojo::MessageOp::WRITE)
        outcome = write(hdr.addr, data, hdr.size_bytes, hdr.vmid);
      else if (hdr.op == simdojo::MessageOp::ATOMIC) {
        auto *mutation = reinterpret_cast<simdojo::MemoryAtomicMutation *>(msg->payload());
        outcome = mutation != nullptr ? atomic_modify(hdr.addr, hdr.size_bytes, *mutation, hdr.vmid)
                                      : VmAccessOutcome::Malformed;
      }
      if (hdr.completion_status != nullptr)
        *hdr.completion_status = message_status(outcome);
      hdr.op = simdojo::MessageOp::RESPONSE;
    });
  }

  void install_cpl_handler() { install_handler(cpl_); }

  GpuMemory *memory_;
  GpuVm *gpu_vm_ = nullptr;
  simdojo::Port *cpl_ = nullptr;
  std::vector<simdojo::Port *> cpl_ports_;
};

} // namespace amdgpu
} // namespace rocjitsu
