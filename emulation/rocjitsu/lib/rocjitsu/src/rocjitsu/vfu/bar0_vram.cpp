// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vfu/bar0_vram.h"
#include "rocjitsu/vfu/ip_discovery_blob.h"
#include "rocjitsu/vfu/minimal_vbios.h"

#include <cstdio>
#include <cstring>

namespace rocjitsu::vfu {

namespace {
// DISCOVERY_TMR_OFFSET: the amdgpu driver places the IP discovery table this
// many bytes before the end of VRAM (256 << 20 for a 256 MB aperture).
constexpr uint64_t kDiscoveryTmrOffset = 64ULL * 1024;
} // namespace

Bar0Vram::Bar0Vram(uint64_t size)
    // SealedMemfd: create, truncate, leave zeroed (no init byte), then seal.
    : memfd_("rocjitsu_vram", static_cast<size_t>(size)) {
  if (!memfd_.valid())
    return;

  // Write minimal atom VBIOS at VRAM byte 0.
  // amdgpu_read_bios_from_vram() ioremaps BAR0+0 and checks for 0x55 0xAA signature.
  // Placing a valid minimal VBIOS here lets adev->bios get populated so the
  // atom context is created, allowing smu_v13_0_get_vbios_bootup_values() to parse it.
  memfd_.with_mapping(0, sizeof(kMinimalVbios), PROT_WRITE,
                      [](void *p) {
                        std::memcpy(p, kMinimalVbios, sizeof(kMinimalVbios));
                      });
  std::fprintf(stderr, "[vfu/bar0] minimal VBIOS written at VRAM+0x0 (%zu bytes)\n",
               sizeof(kMinimalVbios));

  // Write the GFX9.4.4 IP discovery binary at vram_size - DISCOVERY_TMR_OFFSET.
  // amdgpu_discovery_read_binary_from_mem() reads mmRCC_CONFIG_MEMSIZE from
  // BAR5 (returns kVramSizeMb = 256), computes vram_bytes = 256 << 20, then
  // reads the table from BAR0 at that offset. Placing a valid binary there lets
  // ip_discovery_init() succeed.
  if (size > kDiscoveryTmrOffset) {
    auto blob = build_gfx944_discovery_blob();
    const uint64_t blob_offset = size - kDiscoveryTmrOffset;
    bool ok = memfd_.with_mapping(blob_offset, blob.size(), PROT_WRITE,
                                  [&](void *p) {
                                    std::memcpy(p, blob.data(), blob.size());
                                  });
    if (ok)
      std::fprintf(stderr,
                   "[vfu/bar0] IP discovery blob written at VRAM+0x%llx (%zu bytes)\n",
                   static_cast<unsigned long long>(blob_offset), blob.size());
    else
      std::perror("[vfu/bar0] mmap for IP discovery blob");
  }
}


} // namespace rocjitsu::vfu
