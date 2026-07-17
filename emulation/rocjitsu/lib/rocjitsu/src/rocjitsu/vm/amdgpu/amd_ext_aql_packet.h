// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_AMD_EXT_AQL_PACKET_H_
#define ROCJITSU_VM_AMDGPU_AMD_EXT_AQL_PACKET_H_

#ifndef HSA_LARGE_MODEL
#define HSA_LARGE_MODEL 1
#endif

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/hsa.h"
RJ_DIAGNOSTIC_POP

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rocjitsu::amdgpu {

// AMD vendor-specific packet format 0 is unassigned; 200 is reserved and unreleased.

/// ROCR-internal PM4 indirect-buffer packet format.
constexpr uint8_t kAmdAqlFormatPm4Ib = 1;

/// AMD vendor-specific packet format selector for barrier-value packets.
constexpr uint8_t kHsaAmdPacketTypeBarrierValue = 2;

/// AMD vendor-specific packet format selector for extended kernel dispatch.
constexpr uint8_t kHsaAmdPacketTypeExtKernelDispatch = 3;

/// @brief AMD vendor-specific barrier-value packet layout.
///
/// @details Processing stops until `(signal_value & mask) cond value` is true.
/// This mirrors hsa_amd_barrier_value_packet_t without depending on the ROCr
/// extension-header version installed on the host.
struct AmdBarrierValuePacket {
  uint16_t header;
  uint8_t amd_format;
  uint8_t reserved_header;
  uint32_t reserved0;
  hsa_signal_t signal;
  int64_t value;
  int64_t mask;
  uint32_t condition;
  uint32_t reserved1;
  uint64_t reserved2;
  uint64_t reserved3;
  hsa_signal_t completion_signal;
};

static_assert(std::is_trivially_copyable_v<AmdBarrierValuePacket>);
static_assert(sizeof(AmdBarrierValuePacket) == 64);
static_assert(offsetof(AmdBarrierValuePacket, signal) == 8);
static_assert(offsetof(AmdBarrierValuePacket, value) == 16);
static_assert(offsetof(AmdBarrierValuePacket, mask) == 24);
static_assert(offsetof(AmdBarrierValuePacket, condition) == 32);
static_assert(offsetof(AmdBarrierValuePacket, completion_signal) == 56);

/// @brief AMD vendor-specific extended kernel dispatch packet layout.
///
/// @details This mirrors the 64-byte wire packet consumed from an AQL ring for
/// clustered dispatch. Keep the field order and size stable; tests and runtime
/// queue code copy this type directly into packet slots.
struct AmdExtKernelDispatchPacket {
  uint16_t header;
  uint8_t amd_format;
  uint8_t setup;
  uint16_t workgroup_size_x;
  uint16_t workgroup_size_y;
  uint16_t workgroup_size_z;
  uint16_t reserved0;
  uint32_t cluster_count_x;
  uint16_t cluster_count_y;
  uint16_t cluster_count_z;
  uint8_t cluster_size_x;
  uint8_t cluster_size_y;
  uint8_t cluster_size_z;
  uint8_t perf_hint;
  uint32_t private_segment_size;
  uint32_t group_segment_size;
  uint64_t kernel_object;
  void *kernarg_address;
  hsa_signal_t dep_signal;
  hsa_signal_t completion_signal;
};

static_assert(std::is_trivially_copyable_v<AmdExtKernelDispatchPacket>);
static_assert(sizeof(AmdExtKernelDispatchPacket) == 64);
static_assert(offsetof(AmdExtKernelDispatchPacket, cluster_count_x) == 12);
static_assert(offsetof(AmdExtKernelDispatchPacket, cluster_size_x) == 20);
static_assert(offsetof(AmdExtKernelDispatchPacket, private_segment_size) == 24);
static_assert(offsetof(AmdExtKernelDispatchPacket, group_segment_size) == 28);
static_assert(offsetof(AmdExtKernelDispatchPacket, kernel_object) == 32);
static_assert(offsetof(AmdExtKernelDispatchPacket, kernarg_address) == 40);
static_assert(offsetof(AmdExtKernelDispatchPacket, dep_signal) == 48);
static_assert(offsetof(AmdExtKernelDispatchPacket, completion_signal) == 56);

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_VM_AMDGPU_AMD_EXT_AQL_PACKET_H_
