// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "hsa/hsa_ext_amd.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rocjitsu::amdgpu {

// This ROCR-internal format is not declared in the public HSA extension API.
inline constexpr uint8_t kAmdAqlFormatPm4Ib = 1;

inline constexpr uint8_t kHsaAmdPacketTypeBarrierValue = HSA_AMD_PACKET_TYPE_BARRIER_VALUE;
inline constexpr uint8_t kHsaAmdPacketTypeExtKernelDispatch =
    HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH;
inline constexpr uint8_t kHsaAmdPacketTypeReserved200 = HSA_AMD_PACKET_TYPE_RESERVED200;

using AmdBarrierValuePacket = hsa_amd_barrier_value_packet_t;
using AmdExtKernelDispatchPacket = hsa_amd_ext_kernel_dispatch_packet_t;

static_assert(std::is_trivially_copyable_v<AmdBarrierValuePacket>);
static_assert(sizeof(AmdBarrierValuePacket) == 64);
static_assert(offsetof(AmdBarrierValuePacket, signal) == 8);
static_assert(offsetof(AmdBarrierValuePacket, value) == 16);
static_assert(offsetof(AmdBarrierValuePacket, mask) == 24);
static_assert(offsetof(AmdBarrierValuePacket, cond) == 32);
static_assert(offsetof(AmdBarrierValuePacket, completion_signal) == 56);

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
