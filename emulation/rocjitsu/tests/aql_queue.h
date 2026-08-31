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
#include "hsa/amd_ext_aql_packet.h"
#include "hsa/hsa.h"
RJ_DIAGNOSTIC_POP

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
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

  /// @brief Register a queue with one command processor.
  /// @param memory Simulator GPU memory holding the ring and its pointers.
  /// @param cp Command processor to register with.
  /// @param ring_addr Ring buffer base address.
  /// @param ring_size Ring buffer size in bytes.
  /// @param read_ptr_addr Address of the read pointer.
  /// @param write_ptr_addr Address of the write pointer.
  /// @param doorbell_addr Address of the doorbell.
  /// @param xcd_fanout Spread each dispatch over every XCD, as the KFD create-queue
  /// ioctl now does for compute queues on a multi-XCD part. Off by default so a test
  /// that binds a queue to one command processor keeps its CUs to itself.
  /// @param queue_id Queue id. A fan-out queue is replicated onto every XCD and its
  /// shards are routed back by (queue_id, process_id), so a test creating more than
  /// one fan-out queue must give each a distinct id.
  AqlQueue(amdgpu::GpuMemory *memory, amdgpu::CommandProcessor *cp,
           uint64_t ring_addr = DEFAULT_RING_ADDR, uint32_t ring_size = DEFAULT_RING_SIZE,
           uint64_t read_ptr_addr = DEFAULT_READ_PTR_ADDR,
           uint64_t write_ptr_addr = DEFAULT_WRITE_PTR_ADDR,
           uint64_t doorbell_addr = DEFAULT_DOORBELL_ADDR, bool xcd_fanout = false,
           uint32_t queue_id = 1)
      : memory_(memory), cp_(cp), ring_addr_(ring_addr), ring_size_(ring_size),
        read_ptr_addr_(read_ptr_addr), write_ptr_addr_(write_ptr_addr),
        doorbell_addr_(doorbell_addr) {
    uint64_t zero = 0;
    memory_->load_image(reinterpret_cast<const uint8_t *>(&zero), 8, read_ptr_addr_);
    memory_->load_image(reinterpret_cast<const uint8_t *>(&zero), 8, write_ptr_addr_);
    memory_->load_image(reinterpret_cast<const uint8_t *>(&zero), 8, doorbell_addr_);

    amdgpu::HwQueue hw{};
    hw.queue_id = queue_id;
    hw.ring_base_va = ring_addr_;
    hw.ring_size = ring_size_;
    hw.read_ptr_va = read_ptr_addr_;
    hw.write_ptr_va = write_ptr_addr_;
    hw.doorbell_va = doorbell_addr_;
    hw.xcd_fanout = xcd_fanout;
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

  void submit(const amdgpu::AmdExtKernelDispatchPacket &pkt) {
    hsa_kernel_dispatch_packet_t raw{};
    std::memcpy(&raw, &pkt, sizeof(pkt));
    submit(raw);
  }

  void submit(const amdgpu::AmdBarrierValuePacket &pkt) {
    hsa_kernel_dispatch_packet_t raw{};
    std::memcpy(&raw, &pkt, sizeof(pkt));
    submit(raw);
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

  /// Same as dispatch(), with the AQL barrier bit set: the packet must not start
  /// until every preceding packet on this queue has completed.
  void dispatch_with_barrier(uint64_t kernel_object, uint32_t grid_size_x,
                             uint16_t workgroup_size_x = 64, uint64_t kernarg_addr = 0) {
    hsa_kernel_dispatch_packet_t pkt{};
    pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH | (1u << HSA_PACKET_HEADER_BARRIER);
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

  /// Submit a BarrierAND packet.
  ///
  /// @param dep_signal_va Address of a dependency signal object, or 0 for none.
  /// A barrier with no dependencies is satisfied as soon as it is read; one whose
  /// signal still holds a positive value stalls the queue's reader, which is what
  /// must keep a following dispatch from being handed to any XCD.
  void barrier_and(uint64_t dep_signal_va = 0) {
    hsa_kernel_dispatch_packet_t pkt{};
    pkt.header = HSA_PACKET_TYPE_BARRIER_AND;
    // Dependency signal handles start at byte 8 of the barrier packet.
    std::memcpy(reinterpret_cast<uint8_t *>(&pkt) + 8, &dep_signal_va, sizeof(dep_signal_va));
    submit(pkt);
  }

  /// Submit an AMD vendor PM4 indirect-buffer packet.
  ///
  /// @details Runs no shader, so it is one of the packets a fanned-out queue has
  /// to place on every XCD rather than split.
  /// @param completion_signal_va Signal to decrement when it retires, or 0.
  void pm4_ib(uint64_t completion_signal_va = 0) {
    hsa_kernel_dispatch_packet_t pkt{};
    pkt.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
    pkt.setup = amdgpu::kAmdAqlFormatPm4Ib;
    pkt.completion_signal.handle = completion_signal_va;
    submit(pkt);
  }

  /// Build and submit an AMD extended kernel dispatch packet with cluster shape.
  void dispatch_clustered(uint64_t kernel_object, uint32_t cluster_count_x, uint8_t cluster_size_x,
                          uint16_t workgroup_size_x = 64, uint64_t kernarg_addr = 0,
                          uint32_t group_segment_size = 0) {
    amdgpu::AmdExtKernelDispatchPacket pkt{};
    pkt.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
    pkt.amd_format = amdgpu::kHsaAmdPacketTypeExtKernelDispatch;
    pkt.setup = 1;
    pkt.workgroup_size_x = workgroup_size_x;
    pkt.workgroup_size_y = 1;
    pkt.workgroup_size_z = 1;
    pkt.cluster_count_x = cluster_count_x;
    pkt.cluster_count_y = 1;
    pkt.cluster_count_z = 1;
    pkt.cluster_size_x = cluster_size_x;
    pkt.cluster_size_y = 1;
    pkt.cluster_size_z = 1;
    pkt.group_segment_size = group_segment_size;
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

/// @brief Build a fan-out queue without spelling out the defaulted addresses.
/// @param memory Simulator GPU memory holding the ring and its pointers.
/// @param cp Command processor that will own the queue.
/// @param queue_id Must be distinct per fan-out queue; shards route by it.
/// @param ring_addr Ring buffer base; the pointers are placed relative to it.
/// @returns A registered fan-out queue.
inline std::unique_ptr<AqlQueue>
make_fanout_queue(amdgpu::GpuMemory *memory, amdgpu::CommandProcessor *cp, uint32_t queue_id = 1,
                  uint64_t ring_addr = AqlQueue::DEFAULT_RING_ADDR) {
  return std::make_unique<AqlQueue>(memory, cp, ring_addr, AqlQueue::DEFAULT_RING_SIZE,
                                    ring_addr + 0x10000, ring_addr + 0x10008, ring_addr + 0x10010,
                                    /*xcd_fanout=*/true, queue_id);
}

} // namespace rocjitsu::test
