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

#include "rocjitsu/kmd/linux/amdgpu_properties.h"

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

/// @brief Debug-related topology bits derived from a GPU's GFXIP version.
///
/// @details Mirrors the per-node values the amdkfd driver programs in
/// kfd_topology_set_capabilities() (drivers/gpu/drm/amd/amdkfd/kfd_topology.c):
/// the trap-debugger capability flags, the capability2 flags, and the
/// debug_prop address-watch-mask range that libhsakmt and rocdbgapi read back.
struct DebugTopology {
  uint32_t capability = 0;
  uint32_t capability2 = 0;
  uint64_t debug_prop = 0;
};

/// @brief Reproduces kfd_topology_set_capabilities() for the simulated GPU
/// identified by @p gfx_target_version.
///
/// @details The driver keys every decision on the GC hardware IP version, which
/// is not the same number as gfx_target_version for CDNA parts (see
/// gc_ip_version_for_gfx_target_version), so we translate first and then apply
/// the driver's exact IP_VERSION thresholds. Shared by the sysfs topology
/// generator and the DBG_TRAP GET_DEVICE_SNAPSHOT path so both advertise the
/// same debugger-relevant capability/debug_prop.
///
/// \NPI sync this with the KFD driver code in drivers/gpu/drm/amd/amdkfd/kfd_topology.c
inline DebugTopology debug_topology_for(uint32_t gfx_target_version) {
  const uint32_t gc = gc_ip_version_for_gfx_target_version(gfx_target_version);

  DebugTopology topo;

  // Trap-based debugging is advertised for every debug-capable GPU.
  topo.capability = HSA_CAP_TRAP_DEBUG_SUPPORT |
                    HSA_CAP_TRAP_DEBUG_WAVE_LAUNCH_TRAP_OVERRIDE_SUPPORTED |
                    HSA_CAP_TRAP_DEBUG_WAVE_LAUNCH_MODE_SUPPORTED;

  // Advertise the four hardware address-watch registers (TCP_WATCH0..3) so
  // rocm-dbgapi will program data watchpoints. The KFD sets
  // num_of_watch_points = 4 for the supported debug ASICs
  // (amd/amdkfd/kfd_device.c) and packs WATCH_POINTS_SUPPORTED plus
  // ilog2(count) into HSA_CAP_WATCH_POINTS_TOTALBITS (kfd_topology.c). Without
  // it rocm-dbgapi reports zero watch registers and refuses to insert a
  // watchpoint ("too many hardware watchpoints").
  constexpr uint32_t kKfdNumWatchPoints = 4;    // amd/amdkfd/kfd_device.c
  constexpr uint32_t kWatchPointsTotalBits = 2; // ilog2(kKfdNumWatchPoints)
  static_assert((1u << kWatchPointsTotalBits) == kKfdNumWatchPoints,
                "TOTALBITS must be ilog2(num_watch_points)");
  topo.capability |= HSA_CAP_WATCH_POINTS_SUPPORTED |
                     ((kWatchPointsTotalBits << HSA_CAP_WATCH_POINTS_TOTALBITS_SHIFT) &
                      HSA_CAP_WATCH_POINTS_TOTALBITS_MASK);

  // kfd_dbg_has_ttmps_always_setup(): dispatch info (ttmps) is always valid
  // except on gfx9.4.2 (Aldebaran) below gfx11, and on gfx11 only with modern
  // MES firmware (sched_version >= 70), which the simulator always models.
  const bool ttmps_always_setup =
      (gc < make_gc_ip_version(11, 0, 0) && gc != make_gc_ip_version(9, 4, 2)) ||
      gc >= make_gc_ip_version(11, 0, 0);
  if (ttmps_always_setup)
    topo.debug_prop |= HSA_DBG_DISPATCH_INFO_ALWAYS_VALID;

  if (gc < make_gc_ip_version(10, 0, 0)) {
    // gfx9 (CDNA). The watch-address-mask range widens by one bit on the
    // gfx9.4.3/gfx9.4.4 parts (LO 6->7, HI 29->30).
    if (gc == make_gc_ip_version(9, 4, 3) || gc == make_gc_ip_version(9, 4, 4))
      topo.debug_prop |= kWatchAddrMaskLoBitGfx943 | kWatchAddrMaskHiBitGfx943;
    else
      topo.debug_prop |= kWatchAddrMaskLoBitGfx9 | kWatchAddrMaskHiBit;

    if (gc >= make_gc_ip_version(9, 4, 2))
      topo.capability |= HSA_CAP_TRAP_DEBUG_PRECISE_MEMORY_OPERATIONS_SUPPORTED;

    // Per-queue reset is withheld only from SR-IOV virtual functions, which the
    // simulator never models.
    topo.capability |= HSA_CAP_PER_QUEUE_RESET_SUPPORTED;
  } else {
    // gfx10+ (RDNA).
    topo.debug_prop |= kWatchAddrMaskLoBitGfx10 | kWatchAddrMaskHiBit;

    if (gc >= make_gc_ip_version(12, 0, 0))
      topo.capability |= HSA_CAP_TRAP_DEBUG_PRECISE_ALU_OPERATIONS_SUPPORTED;

    if (gc >= make_gc_ip_version(12, 1, 0)) {
      topo.capability |= HSA_CAP_TRAP_DEBUG_PRECISE_MEMORY_OPERATIONS_SUPPORTED |
                         HSA_CAP_PER_QUEUE_RESET_SUPPORTED;
      topo.capability2 |= HSA_CAP2_TRAP_DEBUG_LDS_OUT_OF_ADDR_RANGE_SUPPORTED;
    }
  }

  // Firmware-backed trap debugging (kfd_topology_set_dbg_firmware_support()).
  // The simulator always provides compatible "firmware", so advertise it.
  topo.capability |= HSA_CAP_TRAP_DEBUG_FIRMWARE_SUPPORTED;

  return topo;
}

} // namespace rocjitsu::kmd

#endif // ROCJITSU_KMD_LINUX_KFD_TOPOLOGY_H_
