// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_memory.h
/// @brief AMDGPU VRAM memory as a passive simulation component.

#ifndef ROCJITSU_VM_AMDGPU_GPU_MEMORY_H_
#define ROCJITSU_VM_AMDGPU_GPU_MEMORY_H_

#include "simdojo/components/sparse_memory.h"
#include "simdojo/sim/component.h"

#include <memory>
#include <string>
#include <utility>

namespace rocjitsu {
namespace amdgpu {

/// @brief AMDGPU VRAM memory as a passive simulation component.
///
/// @details Inherits all read/write/load functionality from simdojo::SparseMemory.
/// Memory requests are serviced synchronously via direct read/write calls
/// from compute units - no step() or event-driven scheduling.
///
/// Has an IN port for structural topology visibility. In functional mode,
/// data flows via direct method calls; the port is unused. In clocked mode
/// (future), L2 miss traffic would arrive through this port.
class GpuMemory : public simdojo::SparseMemory {
public:
  /// @brief Construct a GPU memory component.
  /// @param name Human-readable name (e.g., "vram").
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

  /// @brief Return the completer port (receives memory requests).
  /// @returns Pointer to the completer port.
  simdojo::Port *cpl_port() { return cpl_; }

private:
  simdojo::Port *cpl_ = nullptr;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_GPU_MEMORY_H_
