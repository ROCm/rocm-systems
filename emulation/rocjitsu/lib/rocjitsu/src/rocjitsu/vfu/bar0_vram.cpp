// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vfu/bar0_vram.h"
#include "rocjitsu/vfu/ip_discovery_blob.h"
#include "rocjitsu/vfu/pci_config.h"

#include <libvfio-user.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
#ifndef MFD_ALLOW_SEALING
#define MFD_ALLOW_SEALING 0x0002U
#endif

namespace rocjitsu::vfu {

Bar0Vram::Bar0Vram(uint64_t size) : size_(size) {
  memfd_ = static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_vram",
                                     MFD_CLOEXEC | MFD_ALLOW_SEALING));
  if (memfd_ < 0) {
    std::perror("memfd_create (VRAM)");
    return;
  }

  if (ftruncate(memfd_, static_cast<off_t>(size_)) != 0) {
    std::perror("ftruncate (VRAM memfd)");
    ::close(memfd_);
    memfd_ = -1;
    return;
  }

  // Write the IP discovery binary at vram_size - DISCOVERY_TMR_OFFSET (64 KB).
  // amdgpu_discovery_read_binary_from_mem() reads mmRCC_CONFIG_MEMSIZE from
  // BAR5 (returns kVramSizeMb = 256), computes vram_bytes = 256 << 20, then
  // reads the discovery table starting at vram_bytes - 65536 via
  // amdgpu_device_vram_access() which goes through BAR0. Placing a valid
  // GFX9.4.4 binary there lets ip_discovery_init() succeed.
  {
    static constexpr uint64_t kDiscoveryTmrOffset = 64ULL * 1024; // DISCOVERY_TMR_OFFSET
    if (size_ > kDiscoveryTmrOffset) {
      auto blob = build_gfx944_discovery_blob();
      const uint64_t blob_offset = size_ - kDiscoveryTmrOffset;
      void *p = ::mmap(nullptr, blob.size(), PROT_WRITE, MAP_SHARED, memfd_,
                       static_cast<off_t>(blob_offset));
      if (p != MAP_FAILED) {
        std::memcpy(p, blob.data(), blob.size());
        ::munmap(p, blob.size());
        std::fprintf(stderr,
                     "[vfu/bar0] IP discovery blob written at VRAM+0x%llx (%zu bytes)\n",
                     static_cast<unsigned long long>(blob_offset), blob.size());
      } else {
        std::perror("[vfu/bar0] mmap for IP discovery blob");
      }
    }
  }

  // Seal after all writes are done so QEMU can create a dma-buf from the fd.
  // F_SEAL_SHRINK + F_SEAL_GROW prevent size changes, required by the kernel's
  // dma_buf_fd() path. Must have no open MAP_WRITE mappings at this point.
  if (fcntl(memfd_, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW) != 0)
    std::perror("F_ADD_SEALS (VRAM) - BAR0 DMA-buf may not work");
}

Bar0Vram::~Bar0Vram() {
  if (memfd_ >= 0)
    ::close(memfd_);
}

int Bar0Vram::setup(vfu_ctx_t *ctx) {
  if (memfd_ < 0) {
    std::fprintf(stderr, "[vfu/bar0] memfd not initialized\n");
    return -1;
  }

  // BAR0/1: 64-bit prefetchable VRAM aperture, fully mmap-able.
  // VFU_REGION_FLAG_64_BITS tells libvfio-user to set the locatable field in
  // the BAR register to 0b10 (64-bit) and consume the adjacent BAR1 slot for
  // the upper 32 bits of the base address -- matching real AMD GPU hardware.
  // VFU_REGION_FLAG_PREFETCH sets the prefetchable bit so the PCIe root port
  // allocates this BAR from its 64-bit prefetchable (pref64) window, which
  // is large enough for a 256 MB (or ReBAR-sized) aperture.
  iovec mmap_area{.iov_base = nullptr, .iov_len = size_};
  if (vfu_setup_region(ctx,
                       VFU_PCI_DEV_BAR0_REGION_IDX,
                       size_,
                       nullptr, // fully mmap-able; no per-access callback needed
                       VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM |
                         VFU_REGION_FLAG_64_BITS | VFU_REGION_FLAG_PREFETCH,
                       &mmap_area, 1,
                       memfd_, 0) != 0) {
    std::perror("vfu_setup_region BAR0");
    return -1;
  }

  return 0;
}

} // namespace rocjitsu::vfu
