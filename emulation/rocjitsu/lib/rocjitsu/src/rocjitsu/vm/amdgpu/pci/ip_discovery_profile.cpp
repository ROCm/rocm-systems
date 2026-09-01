// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/ip_discovery_profile.h"

#include <cstddef>
#include <iterator>
#include <vector>

namespace rocjitsu {
namespace {

// Register bases are per-block *segments*, not one number. Every register the
// driver touches carries a `_BASE_IDX` naming the segment it lives in, and
// `SOC15_REG_OFFSET` resolves it as `reg_offset[ip][inst][BASE_IDX] + reg`. A
// block that publishes a single base therefore answers only for its
// `_BASE_IDX 0` registers; the rest resolve against whatever follows that base
// in the blob, which for a 64-bit base is its zero high dword. The access then
// lands at a raw offset with nothing reporting an error, so it reads as a
// broken register model rather than as a short table. gc_12_1_0_offset.h puts
// 3576 registers at index 1 against 2441 at index 0, so this is the common case
// and not a corner: gfx_v12_1 alone uses 58 of them, and gfxhub_v12_1 reads
// regGCMC_VM_FB_OFFSET during the GMC init this device has to survive.
//
// These lists are transcribed from a **shipping GFX12 part** reporting GC
// 12.0.1, read out of the table the driver publishes for any bound GPU at
// /sys/class/drm/card*/device/ip_discovery/die/0/<ip>/<inst>/base_addr. Reading
// them off a part that answers beats deriving them: the values are what the
// driver itself resolved, so misreading the format shows up as a mismatch
// rather than as a plausible wrong number. MMHUB and ATHUB match that part's
// versions exactly (4.1.0), so they are transcriptions rather than inferences;
// the others are the same family at a different revision, so treat their higher
// segments as unconfirmed until a table for this part can be read the same way.

/// @brief GC's four segments, which SDMA shares verbatim on this family.
constexpr uint64_t kGcBases[] = {0x00001260, 0x0000A000, 0x0001C000, 0x02402C00};

/// @brief The security and management processors, which publish one list between them.
constexpr uint64_t kMpBases[] = {0x00016000, 0x00016200, 0x0001CE00, 0x00DC0000, 0x00E00000,
                                 0x00E40000, 0x00E80000, 0x00EC0000, 0x00F00000, 0x02400400,
                                 0x0243FC00, 0x0244D400, 0x03200000, 0x03240000, 0x03280000};

constexpr uint64_t kOssSysBases[] = {0x000010A0, 0x0240A000};
constexpr uint64_t kNbifBases[] = {0x00000000, 0x00000014, 0x00000D20,
                                   0x00010400, 0x0241B000, 0x04040000};
constexpr uint64_t kHdpBases[] = {0x00000F20, 0x0240A400};
constexpr uint64_t kMmHubBases[] = {0x0001A000, 0x02408800};
constexpr uint64_t kAtHubBases[] = {0x00000C00, 0x02408C00};

template <std::size_t N> std::vector<uint64_t> bases(const uint64_t (&list)[N]) {
  return std::vector<uint64_t>(std::begin(list), std::end(list));
}

} // namespace

IpDiscoverySpec gfx1250_discovery_spec(const GpuDiscoveryTopology &topology) {
  IpDiscoverySpec spec;

  // Graphics and compute. Its instance count is what the driver turns into the
  // XCC mask, and a table with no graphics block at all is refused outright.
  for (uint8_t instance = 0; instance < topology.graphics_instances; ++instance) {
    spec.blocks.push_back({.hardware_id = IpHardwareId::Gc,
                           .instance = instance,
                           .major = 12,
                           .minor = 1,
                           .revision = 0,
                           .register_bases = bases(kGcBases)});
  }

  // The security processor. The driver has no arm for a device without one and
  // fails to add any PSP block, so it must be named whether or not it is used.
  spec.blocks.push_back({.hardware_id = IpHardwareId::Mp0,
                         .instance = 0,
                         .major = 15,
                         .minor = 0,
                         .revision = 8,
                         .register_bases = bases(kMpBases)});
  spec.blocks.push_back({.hardware_id = IpHardwareId::Mp1,
                         .instance = 0,
                         .major = 15,
                         .minor = 0,
                         .revision = 8,
                         .register_bases = bases(kMpBases)});
  spec.blocks.push_back({.hardware_id = IpHardwareId::OssSys,
                         .instance = 0,
                         .major = 7,
                         .minor = 1,
                         .revision = 0,
                         .register_bases = bases(kOssSysBases)});
  spec.blocks.push_back({.hardware_id = IpHardwareId::Nbif,
                         .instance = 0,
                         .major = 7,
                         .minor = 11,
                         .revision = 0,
                         .register_bases = bases(kNbifBases)});
  // 7.0.0 rather than 7.1.0, deliberately. The driver's HDP switch has an arm
  // only for 7.0.0; 7.1.0 falls through its default and leaves `hdp.funcs`
  // null, so there is no HDP block driver at all and the flush path quietly
  // does nothing. No hdp_v7_1 exists anywhere in the tree, and 7.0.0 is what
  // the real part these bases came from reports, so nothing is given up.
  spec.blocks.push_back({.hardware_id = IpHardwareId::Hdp,
                         .instance = 0,
                         .major = 7,
                         .minor = 0,
                         .revision = 0,
                         .register_bases = bases(kHdpBases)});
  spec.blocks.push_back({.hardware_id = IpHardwareId::MmHub,
                         .instance = 0,
                         .major = 4,
                         .minor = 1,
                         .revision = 0,
                         .register_bases = bases(kMmHubBases)});
  spec.blocks.push_back({.hardware_id = IpHardwareId::AtHub,
                         .instance = 0,
                         .major = 4,
                         .minor = 1,
                         .revision = 0,
                         .register_bases = bases(kAtHubBases)});
  // One engine, named so the driver instantiates an SDMA block at all; a second
  // instance would be a second record here, and this device models none. The
  // segments are GC's, which this family shares between the two.
  spec.blocks.push_back({.hardware_id = IpHardwareId::Sdma0,
                         .instance = 0,
                         .major = 7,
                         .minor = 1,
                         .revision = 0,
                         .register_bases = bases(kGcBases)});
  return spec;
}

} // namespace rocjitsu
