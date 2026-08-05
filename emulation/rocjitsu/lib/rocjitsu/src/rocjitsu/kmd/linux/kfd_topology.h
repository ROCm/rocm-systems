// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kfd_topology.h
/// @brief Per-GFXIP KFD topology @c debug_prop values the synthetic topology
/// generator mirrors from the amdkfd driver.
///
/// @details These are the address-watch-mask low/high bit positions that
/// @c kfd_topology_set_capabilities() (drivers/gpu/drm/amd/amdkfd/kfd_topology.c)
/// writes into each node's @c debug_prop property and that libhsakmt and
/// rocdbgapi read back. They are driver-internal -- NOT part of the KFD UAPI --
/// so rocjitsu defines its own constants here instead of vendoring the private
/// amdkfd header. The values are placed within the UAPI-defined
/// @c HSA_DBG_WATCH_ADDR_MASK_{LO,HI}_BIT fields from linux/uapi/kfd_sysfs.h,
/// which is the ABI both sides agree on, so the high values are shifted by
/// @c HSA_DBG_WATCH_ADDR_MASK_HI_BIT_SHIFT to land in the high field.

#ifndef ROCJITSU_KMD_LINUX_KFD_TOPOLOGY_H_
#define ROCJITSU_KMD_LINUX_KFD_TOPOLOGY_H_

#include "rocjitsu/base/rj_compiler.h"

RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_sysfs.h"
RJ_DIAGNOSTIC_POP

#include <cstdint>

namespace rocjitsu::kmd {

/// @brief gfx9 (CDNA) low watch-address-mask bit
/// (upstream @c HSA_DBG_WATCH_ADDR_MASK_LO_BIT_GFX9).
inline constexpr uint64_t kWatchAddrMaskLoBitGfx9 = 6;

/// @brief gfx9.4.3/gfx9.4.4 low watch-address-mask bit
/// (upstream @c HSA_DBG_WATCH_ADDR_MASK_LO_BIT_GFX9_4_3).
inline constexpr uint64_t kWatchAddrMaskLoBitGfx943 = 7;

/// @brief gfx10+ (RDNA) low watch-address-mask bit
/// (upstream @c HSA_DBG_WATCH_ADDR_MASK_LO_BIT_GFX10).
inline constexpr uint64_t kWatchAddrMaskLoBitGfx10 = 7;

/// @brief Default high watch-address-mask bit
/// (upstream @c HSA_DBG_WATCH_ADDR_MASK_HI_BIT).
inline constexpr uint64_t kWatchAddrMaskHiBit = 29ull << HSA_DBG_WATCH_ADDR_MASK_HI_BIT_SHIFT;

/// @brief gfx9.4.3/gfx9.4.4 high watch-address-mask bit
/// (upstream @c HSA_DBG_WATCH_ADDR_MASK_HI_BIT_GFX9_4_3).
inline constexpr uint64_t kWatchAddrMaskHiBitGfx943 = 30ull << HSA_DBG_WATCH_ADDR_MASK_HI_BIT_SHIFT;

} // namespace rocjitsu::kmd

#endif // ROCJITSU_KMD_LINUX_KFD_TOPOLOGY_H_
