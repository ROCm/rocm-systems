// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vfu/bar0_vram.h"
#include "rocjitsu/vfu/ip_discovery_blob.h"
#include "rocjitsu/vfu/pci_config.h"

#include <libvfio-user.h>

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

int Bar0Vram::setup(vfu_ctx_t *ctx) {
  if (!memfd_.valid()) {
    std::fprintf(stderr, "[vfu/bar0] memfd not initialized\n");
    return -1;
  }

  // BAR0/1: 64-bit prefetchable VRAM aperture, fully mmap-able.
  // VFU_REGION_FLAG_64_BITS: locatable field 0b10 (64-bit), consumes BAR1 slot.
  // VFU_REGION_FLAG_PREFETCH: allocate from the PCIe pref64 window.
  iovec mmap_area{.iov_base = nullptr, .iov_len = memfd_.size()};
  if (vfu_setup_region(ctx,
                       VFU_PCI_DEV_BAR0_REGION_IDX,
                       memfd_.size(),
                       nullptr, // fully mmap-able; no per-access callback needed
                       VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM |
                         VFU_REGION_FLAG_64_BITS | VFU_REGION_FLAG_PREFETCH,
                       &mmap_area, 1,
                       memfd_.fd(), 0) != 0) {
    std::perror("vfu_setup_region BAR0");
    return -1;
  }

  return 0;
}

} // namespace rocjitsu::vfu
