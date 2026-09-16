// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kfd_topology.h
/// @brief Derivation of the debugger-visible KFD node properties -- @c capability,
/// @c capability2 and @c debug_prop -- from a GPU's GFXIP version, mirrored from
/// the amdkfd driver.
///
/// @details Shared by the synthetic sysfs topology generator and by DBG_TRAP
/// GET_DEVICE_SNAPSHOT, so both advertise the same values to libhsakmt and
/// rocdbgapi. Anything a debugger reads off a node belongs here rather than in
/// one of the two consumers, or the two views drift.
///
/// The address-watch-mask constants below are the low/high bit positions that
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
#include "rocjitsu/kmd/linux/cwsr.h"

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

/// @brief Number of hardware address-watch registers (TCP_WATCH0..3) the
/// simulated debug ASICs expose, matching the driver's @c num_of_watch_points
/// (drivers/gpu/drm/amd/amdkfd/kfd_device.c).
inline constexpr uint32_t kNumWatchPoints = 4;

/// @brief Value the driver packs into @c HSA_CAP_WATCH_POINTS_TOTALBITS.
///
/// @details The field holds ilog2(num_watch_points), not the count itself:
/// rocdbgapi recovers the register count as @c 1<<TOTALBITS (see
/// os_driver_kfd.cpp, kfd_topology_snapshot()). Advertising the count directly
/// would claim 16 registers and make rocdbgapi program watchpoints that do not
/// exist.
inline constexpr uint32_t kWatchPointsTotalBits = 2;
static_assert((1u << kWatchPointsTotalBits) == kNumWatchPoints,
              "TOTALBITS must be ilog2(kNumWatchPoints)");

/// @brief Non-debug capability bits advertised for simulated data-center GPUs.
///
/// @details The address-watch bits are deliberately not here: they are a
/// debugger-relevant capability, so debug_topology_for() contributes them
/// alongside the other bits rocdbgapi reads out of the DBG_TRAP device
/// snapshot. Note that both this word and the derived debug bits are dropped
/// when a config supplies its own @c capability (see effective_topology_for),
/// so an overriding config must carry the watch bits in its captured value.
inline constexpr uint32_t default_non_debug_capability() {
  return HSA_CAP_ATS_PRESENT | HSA_CAP_QUEUE_IDLE_EVENT |
         ((HSA_CAP_DOORBELL_TYPE_2_0 << HSA_CAP_DOORBELL_TYPE_TOTALBITS_SHIFT) &
          HSA_CAP_DOORBELL_TYPE_TOTALBITS_MASK) |
         HSA_CAP_AQL_QUEUE_DOUBLE_MAP | HSA_CAP_MEM_EDCSUPPORTED | HSA_CAP_RASEVENTNOTIFY |
         HSA_CAP_SRAM_EDCSUPPORTED | HSA_CAP_SVMAPI_SUPPORTED | HSA_CAP_FLAGS_COHERENTHOSTACCESS;
}

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

  // Trap-based debugging is advertised for every debug-capable GPU. The
  // simulator withholds the support bit again at the end of this function for
  // parts whose CWSR layout it cannot produce; derive it here so the driver
  // mirroring stays readable as one piece.
  topo.capability = HSA_CAP_TRAP_DEBUG_SUPPORT |
                    HSA_CAP_TRAP_DEBUG_WAVE_LAUNCH_TRAP_OVERRIDE_SUPPORTED |
                    HSA_CAP_TRAP_DEBUG_WAVE_LAUNCH_MODE_SUPPORTED;

  // Every debug-capable ASIC the simulator models carries the four hardware
  // address-watch registers, so advertise them here rather than alongside the
  // generic non-debug bits: rocdbgapi reads this out of the device snapshot to
  // decide how many data watchpoints it may insert. Without the bits it reports
  // zero watch registers and refuses to insert one ("too many hardware
  // watchpoints").
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

  // Everything above mirrors the driver. This does not, and is the one place the
  // simulator is deliberately less capable than the hardware it models: a
  // debugger can only be serviced on a part whose CWSR record layout the codec
  // reproduces, because rocm-dbgapi reads wave state straight out of the save
  // area rather than through an ioctl. Withhold the single bit it keys
  // agent_t::supports_debugging() on, so an unmodelled agent is declined at the
  // capability boundary -- at attach, with every other agent still usable --
  // instead of attaching and then having each individual wave stop declined.
  // The descriptive sub-capabilities stay as the driver derives them; they
  // qualify a support that is no longer claimed.
  if (!cwsr_layout_modelled_for_gc_ip_version(gc))
    topo.capability &= ~static_cast<uint32_t>(HSA_CAP_TRAP_DEBUG_SUPPORT);

  return topo;
}

/// @brief Resolve configured topology overrides and the ASIC revision bits.
inline DebugTopology effective_topology_for(uint32_t gfx_target_version, uint32_t capability,
                                            uint32_t capability2, uint64_t debug_prop,
                                            uint32_t revision_id) {
  const DebugTopology derived = debug_topology_for(gfx_target_version);
  DebugTopology effective{
      .capability =
          capability != 0 ? capability : default_non_debug_capability() | derived.capability,
      .capability2 = capability2 != 0 ? capability2 : derived.capability2,
      .debug_prop = debug_prop != 0 ? debug_prop : derived.debug_prop,
  };
  effective.capability =
      (effective.capability & ~HSA_CAP_ASIC_REVISION_MASK) |
      ((revision_id << HSA_CAP_ASIC_REVISION_SHIFT) & HSA_CAP_ASIC_REVISION_MASK);
  // Again after the override merge, not only in debug_topology_for(): a config
  // that captured a real device's capability word carries that device's
  // trap-debug bit, and the simulator must not inherit a claim it cannot honour.
  if (!cwsr_layout_modelled_for_gc_ip_version(
          gc_ip_version_for_gfx_target_version(gfx_target_version)))
    effective.capability &= ~static_cast<uint32_t>(HSA_CAP_TRAP_DEBUG_SUPPORT);
  return effective;
}

} // namespace rocjitsu::kmd

#endif // ROCJITSU_KMD_LINUX_KFD_TOPOLOGY_H_
