// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file amd_ext_aql_packet.h
/// @brief Minimal AMD vendor-specific AQL packet ABI mirror.
///
/// @details The source of truth for the public packet layouts and format values is
/// `projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h`. Keep this mirror limited to the
/// packet ABI consumed by RocJITsu and update the layout assertions whenever that header changes.

#pragma once

#include "hsa/hsa.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rocjitsu::amdgpu {

/// ROCR-internal PM4 indirect-buffer packet format.
constexpr uint8_t kAmdAqlFormatPm4Ib = 1;

/// AMD vendor-specific packet format selector for barrier-value packets.
constexpr uint8_t kHsaAmdPacketTypeBarrierValue = 2;

/// AMD vendor-specific packet format selector for extended kernel dispatch.
constexpr uint8_t kHsaAmdPacketTypeExtKernelDispatch = 3;

/// Reserved, unreleased AMD vendor-specific packet format.
constexpr uint8_t kHsaAmdPacketTypeReserved200 = 200;

/// @brief AMD vendor-specific barrier-value packet layout.
///
/// @details Processing stops until `(signal_value & mask) cond value` is true. This mirrors
/// `hsa_amd_barrier_value_packet_t` without depending on an ROCr extension header installed on the
/// host.
struct AmdBarrierValuePacket {
  uint16_t header;
  uint8_t amd_format;
  uint8_t reserved_header;
  uint32_t reserved0;
  hsa_signal_t signal;
  hsa_signal_value_t value;
  hsa_signal_value_t mask;
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
/// @details This mirrors `hsa_amd_ext_kernel_dispatch_packet_t`, the 64-byte wire packet consumed
/// from an AQL ring for clustered dispatch. Keep the field order and size stable; tests and runtime
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
