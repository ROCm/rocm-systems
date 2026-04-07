// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_HBM_CONTROLLER_H_
#define ROCJITSU_VM_AMDGPU_HBM_CONTROLLER_H_

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "simdojo/sim/component.h"

#include <cstdint>
#include <cstring>
#include <memory>
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
  explicit HbmController(GpuMemory *memory) : simdojo::Component("hbm"), memory_(memory) {
    cpl_ = add_port(std::make_unique<simdojo::Port>("cpl", 0, this, simdojo::PortDirection::IN,
                                                    simdojo::PortProtocol::MEMORY));
    install_cpl_handler();
  }

  HbmController(std::string name, GpuMemory *memory)
      : simdojo::Component(std::move(name)), memory_(memory) {
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

  void read(uint64_t addr, uint8_t *dst, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i)
      dst[i] = memory_->read8(addr + i);
  }

  void write(uint64_t addr, const uint8_t *src, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i)
      memory_->write8(addr + i, src[i]);
  }

  /// @brief Read a 32-bit dword (little-endian).
  uint32_t read32(uint64_t addr) const { return memory_->read32(addr); }

  /// @brief Write a 32-bit dword (little-endian).
  void write32(uint64_t addr, uint32_t val) { memory_->write32(addr, val); }

  /// @brief Direct access to the underlying GpuMemory.
  GpuMemory *memory() const { return memory_; }

  /// @brief Set (or replace) the underlying GpuMemory.
  ///
  /// Used by the config loader for deferred initialization.
  /// @param memory New GPU memory (not owned).
  void set_memory(GpuMemory *memory) { memory_ = memory; }

private:
  void install_handler(simdojo::Port *port) {
    port->recv_event()->set_handler([this](simdojo::Tick, simdojo::Message *msg) {
      auto &hdr = msg->header();
      auto *data = reinterpret_cast<uint8_t *>(msg->payload());
      if (hdr.op == simdojo::MessageOp::READ)
        read(hdr.addr, data, hdr.size_bytes);
      else if (hdr.op == simdojo::MessageOp::WRITE)
        write(hdr.addr, data, hdr.size_bytes);
      hdr.op = simdojo::MessageOp::RESPONSE;
    });
  }

  void install_cpl_handler() { install_handler(cpl_); }

  GpuMemory *memory_;
  simdojo::Port *cpl_ = nullptr;
  std::vector<simdojo::Port *> cpl_ports_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_HBM_CONTROLLER_H_
