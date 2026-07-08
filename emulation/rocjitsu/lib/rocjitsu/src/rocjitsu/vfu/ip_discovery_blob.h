// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file ip_discovery_blob.h
/// @brief Builds a minimal but valid AMD IP discovery binary for GFX9.4.4 (MI350P).
///
/// The amdgpu driver reads this binary from the end of VRAM (at
/// vram_size - DISCOVERY_TMR_OFFSET) to identify the GPU's IP blocks and their
/// versions. The binary is placed there by Bar0Vram during initialisation.
///
/// Binary layout (from discovery.h):
///   binary_header      — magic signature, version, per-table offsets+checksums
///   ip_discovery_header — signature, num_dies, die_info array
///   die_header         — die_id, num_ips
///   ip[]               — one per IP block (hw_id, major, minor, rev, base_addresses)
///   harvest_info_header + harvest_table — empty (no harvested blocks)

#ifndef ROCJITSU_VFU_IP_DISCOVERY_BLOB_H_
#define ROCJITSU_VFU_IP_DISCOVERY_BLOB_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rocjitsu::vfu {

/// @brief Build a GFX9.4.4 (MI350P) IP discovery binary blob.
///
/// The returned vector is exactly DISCOVERY_TMR_SIZE (10240) bytes,
/// with all required fields, signatures, and checksums populated so the
/// driver's verify steps pass.
///
/// IP versions emitted match what amdgpu_discovery_set_ip_blocks() expects
/// to activate the gfx_v9_4_3_ip_block and related blocks for GFX9.4.4.
std::vector<uint8_t> build_gfx944_discovery_blob();

} // namespace rocjitsu::vfu

#endif // ROCJITSU_VFU_IP_DISCOVERY_BLOB_H_
