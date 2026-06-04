// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"

#ifndef HSA_LARGE_MODEL
#define HSA_LARGE_MODEL 1
#endif

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/hsa.h"
RJ_DIAGNOSTIC_POP

#include <chrono>
#include <cstring>
#include <thread>

namespace rocjitsu::test {

/// Simulates what ROCR's user-mode AqlQueue does: manages a ring buffer,
/// writes AQL packets, and rings the doorbell. Does NOT load kernels --
/// that's the runtime's job. Tests write kernel descriptors and code
/// to GPU memory directly, following the AMDHSA ABI.
class AqlQueue {
public:
  static constexpr uint64_t DEFAULT_RING_ADDR = 0xF0000000ULL;
  static constexpr uint32_t DEFAULT_RING_SIZE = 4096; // 64 packets
  static constexpr uint64_t DEFAULT_READ_PTR_ADDR = 0xF0010000ULL;
  static constexpr uint64_t DEFAULT_WRITE_PTR_ADDR = 0xF0010008ULL;
  static constexpr uint64_t DEFAULT_DOORBELL_ADDR = 0xF0010010ULL;

  AqlQueue(amdgpu::GpuMemory *memory, amdgpu::CommandProcessor *cp,
           uint64_t ring_addr = DEFAULT_RING_ADDR, uint32_t ring_size = DEFAULT_RING_SIZE,
           uint64_t read_ptr_addr = DEFAULT_READ_PTR_ADDR,
           uint64_t write_ptr_addr = DEFAULT_WRITE_PTR_ADDR,
           uint64_t doorbell_addr = DEFAULT_DOORBELL_ADDR)
      : memory_(memory), cp_(cp), ring_addr_(ring_addr), ring_size_(ring_size),
        read_ptr_addr_(read_ptr_addr), write_ptr_addr_(write_ptr_addr),
        doorbell_addr_(doorbell_addr) {
    uint64_t zero = 0;
    memory_->load_image(reinterpret_cast<const uint8_t *>(&zero), 8, read_ptr_addr_);
    memory_->load_image(reinterpret_cast<const uint8_t *>(&zero), 8, write_ptr_addr_);
    memory_->load_image(reinterpret_cast<const uint8_t *>(&zero), 8, doorbell_addr_);

    amdgpu::HwQueue hw{};
    hw.queue_id = 1;
    hw.ring_base_va = ring_addr_;
    hw.ring_size = ring_size_;
    hw.read_ptr_va = read_ptr_addr_;
    hw.write_ptr_va = write_ptr_addr_;
    hw.doorbell_va = doorbell_addr_;
    cp_->register_queue(std::move(hw));
  }

  /// Write an AQL dispatch packet and ring the doorbell via GPU memory.
  void submit(const hsa_kernel_dispatch_packet_t &pkt) {
    uint32_t slot = static_cast<uint32_t>(write_idx_ % (ring_size_ / 64));
    uint64_t pkt_addr = ring_addr_ + slot * 64;
    memory_->load_image(reinterpret_cast<const uint8_t *>(&pkt), 64, pkt_addr);
    ++write_idx_;
    memory_->load_image(reinterpret_cast<const uint8_t *>(&write_idx_), 8, write_ptr_addr_);
    memory_->load_image(reinterpret_cast<const uint8_t *>(&write_idx_), 8, doorbell_addr_);
    // Inject a doorbell event so the engine processes it on the next drain.
    cp_->engine()->schedule_event_now(cp_->doorbell_event());
  }

  /// Build and submit a kernel dispatch packet.
  void dispatch(uint64_t kernel_object, uint32_t grid_size_x, uint16_t workgroup_size_x = 64,
                uint64_t kernarg_addr = 0) {
    hsa_kernel_dispatch_packet_t pkt{};
    pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    pkt.setup = 1;
    pkt.workgroup_size_x = workgroup_size_x;
    pkt.workgroup_size_y = 1;
    pkt.workgroup_size_z = 1;
    pkt.grid_size_x = grid_size_x;
    pkt.grid_size_y = 1;
    pkt.grid_size_z = 1;
    pkt.kernel_object = kernel_object;
    pkt.kernarg_address = reinterpret_cast<void *>(kernarg_addr);
    submit(pkt);
  }

private:
  amdgpu::GpuMemory *memory_;
  amdgpu::CommandProcessor *cp_;
  uint64_t ring_addr_;
  uint32_t ring_size_;
  uint64_t read_ptr_addr_;
  uint64_t write_ptr_addr_;
  uint64_t doorbell_addr_;
  uint64_t write_idx_ = 0;
};

} // namespace rocjitsu::test
