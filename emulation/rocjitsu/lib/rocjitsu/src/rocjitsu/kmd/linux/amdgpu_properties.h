// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_AMDGPU_PROPERTIES_H_
#define ROCJITSU_KMD_LINUX_AMDGPU_PROPERTIES_H_

#include "rocjitsu/code/rj_code.h"

#include "util/bit.h"

#include <cstdint>
#include <string>

namespace rocjitsu::kmd {

inline constexpr uint32_t kAmdgpuVramTypeHbm = 6;
inline constexpr uint32_t kAmdgpuVramTypeGddr6 = 9;

struct GfxIpVersion {
  uint32_t major = 0;
  uint32_t minor = 0;
  uint32_t stepping = 0;
};

constexpr GfxIpVersion decode_gfx_target_version(uint32_t gfx_target_version) {
  return {
      gfx_target_version / 10000u,
      (gfx_target_version / 100u) % 100u,
      gfx_target_version % 100u,
  };
}

/// @brief Packs a GC (graphics core) hardware IP version exactly like the
/// amdgpu @c IP_VERSION() macro: @c (major<<24)|(minor<<16)|(rev<<8).
///
/// @details Producing the same packed representation the kernel uses lets the
/// synthetic topology mirror the amdkfd driver's @c KFD_GC_VERSION comparisons
/// (drivers/gpu/drm/amd/amdkfd/kfd_topology.c) bit-for-bit.
constexpr uint32_t make_gc_ip_version(uint32_t major, uint32_t minor, uint32_t rev) {
  return (major << 24) | (minor << 16) | (rev << 8);
}

/// @brief Maps a @c gfx_target_version (e.g. 90402 for gfx942) to the GC
/// hardware IP version the amdkfd driver keys its capability logic on.
///
/// @details The two identifiers are not interchangeable for CDNA parts. The KFD
/// device table (drivers/gpu/drm/amd/amdkfd/kfd_device.c) reports, for example,
/// gfx90a (90010) as GC 9.4.2 and gfx942 (90402) as GC 9.4.3, so a plain
/// major.minor.stepping decode would land on the wrong side of the driver's
/// version thresholds (missing precise-memory support on gfx90a, or the
/// gfx9.4.3 watch-address-mask values on gfx942). Only the parts whose encoding
/// diverges from a direct decode are listed here; every other GPU -- all of
/// gfx10/gfx11 and gfx12.0 -- decodes directly, which is exact for the
/// thresholds the driver compares against.
///
/// \NPI add a case when a new GPU's gfx_target_version does not decode directly
/// to its GC hardware IP version (see the KFD device table in
/// drivers/gpu/drm/amd/amdkfd/kfd_device.c).
constexpr uint32_t gc_ip_version_for_gfx_target_version(uint32_t gfx_target_version) {
  switch (gfx_target_version) {
  case 90010: // Aldebaran / gfx90a  -> GC 9.4.2
    return make_gc_ip_version(9, 4, 2);
  case 90402: // MI300 / gfx942      -> GC 9.4.3
    return make_gc_ip_version(9, 4, 3);
  case 90500: // MI350 / gfx950      -> GC 9.5.0
    return make_gc_ip_version(9, 5, 0);
  case 120500: // gfx1250            -> GC 12.1.0
    return make_gc_ip_version(12, 1, 0);
  default: {
    const GfxIpVersion ip = decode_gfx_target_version(gfx_target_version);
    return make_gc_ip_version(ip.major, ip.minor, ip.stepping);
  }
  }
}

inline std::string gfx_target_name(uint32_t gfx_target_version) {
  auto ip = decode_gfx_target_version(gfx_target_version);
  constexpr char kHexDigits[] = "0123456789abcdef";
  if (ip.minor < 16 && ip.stepping < 16)
    return "gfx" + std::to_string(ip.major) + kHexDigits[ip.minor] + kHexDigits[ip.stepping];
  return "gfx" + std::to_string(ip.major) + std::to_string(ip.minor) + std::to_string(ip.stepping);
}

constexpr uint32_t external_rev_id_for_gfx_target_version(uint32_t gfx_target_version,
                                                          uint32_t revision_id) {
  auto ip = decode_gfx_target_version(gfx_target_version);
  if (ip.major == 12 && ip.minor == 0) {
    if (ip.stepping == 0)
      return revision_id + 0x40;
    if (ip.stepping == 1)
      return revision_id + 0x50;
  }

  if (ip.major == 11 && ip.minor == 0) {
    if (ip.stepping == 0 || ip.stepping == 1)
      return revision_id + 0x1;
    if (ip.stepping == 2)
      return revision_id + 0x10;
    if (ip.stepping == 3)
      return revision_id + 0x20;
    if (ip.stepping == 4)
      return revision_id + 0x80;
  }

  if (ip.major == 11 && ip.minor == 5) {
    if (ip.stepping == 0)
      return revision_id == 0 ? 0x1 : revision_id + 0x10;
    if (ip.stepping == 1)
      return revision_id + 0xc1;
    if (ip.stepping == 2)
      return revision_id + 0x40;
    if (ip.stepping == 3)
      return revision_id + 0x50;
    if (ip.stepping == 4)
      return revision_id + 0x1;
    if (ip.stepping == 6)
      return revision_id + 0xd0;
  }

  return revision_id;
}

constexpr uint32_t num_hw_gfx_contexts_for_gfx_target_version(uint32_t gfx_target_version) {
  auto ip = decode_gfx_target_version(gfx_target_version);
  return (ip.major >= 9 && ip.major <= 12) ? 8u : 1u;
}

// gb_addr_config is read by the kernel from the GB_ADDR_CONFIG register
// (gfx_v11_0/gfx_v12_0 get_gb_addr_config). These constants were captured
// from local W7900/R9700 hardware through AMDKFD_IOC_GET_TILE_CONFIG.
constexpr uint32_t gb_addr_config_for_arch(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA3:
    return 0x545;
  case ROCJITSU_CODE_ARCH_RDNA4:
    return 0x8200545;
  default:
    return 0;
  }
}

constexpr uint32_t gb_addr_config_for_gfx_target_version(uint32_t gfx_target_version) {
  auto ip = decode_gfx_target_version(gfx_target_version);
  if (ip.major == 12 && ip.minor == 0)
    return gb_addr_config_for_arch(ROCJITSU_CODE_ARCH_RDNA4);
  if (ip.major == 11 && ip.minor == 0)
    return gb_addr_config_for_arch(ROCJITSU_CODE_ARCH_RDNA3);
  return 0;
}

constexpr uint32_t drm_shader_engine_count(uint32_t kfd_array_count, uint32_t arrays_per_engine) {
  if (kfd_array_count == 0 || arrays_per_engine <= 1)
    return kfd_array_count;
  return util::ceil_div(kfd_array_count, arrays_per_engine);
}

constexpr uint32_t drm_quad_shader_pipe_count(uint32_t kfd_array_count) { return kfd_array_count; }

constexpr uint32_t drm_cu_active_number(uint32_t kfd_array_count, uint32_t cu_per_shader_array) {
  return kfd_array_count * cu_per_shader_array;
}

} // namespace rocjitsu::kmd

#endif // ROCJITSU_KMD_LINUX_AMDGPU_PROPERTIES_H_
