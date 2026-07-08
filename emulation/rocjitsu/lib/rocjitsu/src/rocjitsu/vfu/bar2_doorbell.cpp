// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vfu/bar2_doorbell.h"
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

namespace rocjitsu::vfu {

Bar2Doorbell::Bar2Doorbell(uint64_t size, RingCallback cb)
    : size_(size), ring_cb_(std::move(cb)) {
  // Create a sealable backing memfd. The ALLOW_SEALING flag is required so
  // we can add F_SEAL_SHRINK|F_SEAL_GROW after truncation, which lets QEMU's
  // vfio-user client create a dma-buf from the fd.
  memfd_ = static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_doorbell",
                                     MFD_CLOEXEC | MFD_ALLOW_SEALING));
  if (memfd_ < 0) {
    std::perror("memfd_create (doorbell)");
    return;
  }
  if (ftruncate(memfd_, static_cast<off_t>(size_)) != 0) {
    std::perror("ftruncate (doorbell memfd)");
    ::close(memfd_);
    memfd_ = -1;
    return;
  }

  // Initialize all doorbell slots to the sentinel value BEFORE sealing.
  // rocjitsu's CP polling thread uses 0xFF…FF as the "no packet" sentinel;
  // without this the first read pointer update would be silently dropped.
  {
    void *p = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, memfd_, 0);
    if (p != MAP_FAILED) {
      ::memset(p, 0xFF, size_);
      ::munmap(p, size_);
    }
  }

  // Seal: no more truncations allowed. Must happen after the mmap above is
  // unmapped so no MAP_WRITE mapping exists at seal time.
  if (fcntl(memfd_, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW) != 0)
    std::perror("F_ADD_SEALS (doorbell) — BAR2 DMA-buf may not work");
}

Bar2Doorbell::~Bar2Doorbell() {
  if (memfd_ >= 0)
    ::close(memfd_);
}

int Bar2Doorbell::setup(vfu_ctx_t *ctx) {
  if (memfd_ < 0) {
    std::fprintf(stderr, "[vfu/bar2] doorbell memfd not initialized\n");
    return -1;
  }

  // BAR2/3: 64-bit prefetchable doorbell region.
  // Trap writes so we can dispatch into the CP, but also provide a mmap area
  // so the guest can signal doorbells without per-access socket round-trips.
  // 64-bit prefetchable matches real AMD GPU hardware (amdgpu_doorbell.h).
  iovec mmap_area{.iov_base = nullptr, .iov_len = size_};
  if (vfu_setup_region(ctx,
                       VFU_PCI_DEV_BAR2_REGION_IDX,
                       size_,
                       Bar2Doorbell::access_cb,
                       VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM |
                         VFU_REGION_FLAG_64_BITS | VFU_REGION_FLAG_PREFETCH,
                       &mmap_area, 1,
                       memfd_, 0) != 0) {
    std::perror("vfu_setup_region BAR2");
    return -1;
  }

  return 0;
}

ssize_t Bar2Doorbell::handle_access(char *buf, size_t count, long offset, bool is_write) {
  if (static_cast<uint64_t>(offset) + count > size_) {
    errno = ERANGE;
    return -1;
  }

  if (!is_write) {
    // Guest reads the doorbell page (unusual but harmless — return zeros).
    std::memset(buf, 0, count);
    return static_cast<ssize_t>(count);
  }

  // Guest wrote a doorbell slot. The write is either 4 or 8 bytes.
  // Doorbells are 64-bit slots; collect either size and dispatch.
  uint64_t value = 0;
  std::memcpy(&value, buf, count < 8 ? count : 8);

  // Also update the shadow memfd so a subsequent guest mmap read is coherent.
  void *p = ::mmap(nullptr, count, PROT_WRITE, MAP_SHARED, memfd_,
                   static_cast<off_t>(offset & ~(static_cast<long>(sysconf(_SC_PAGE_SIZE))-1)));
  if (p != MAP_FAILED) {
    std::memcpy(p, buf, count);
    ::munmap(p, count);
  }

  if (ring_cb_)
    ring_cb_(static_cast<uint32_t>(offset), value);

  return static_cast<ssize_t>(count);
}

ssize_t Bar2Doorbell::access_cb(vfu_ctx_t *ctx, char *buf, size_t count,
                                  long offset, bool is_write) {
  auto *db = reinterpret_cast<Bar2Doorbell *>(vfu_get_private(ctx));
  if (!db)
    return -1;
  return db->handle_access(buf, count, offset, is_write);
}

} // namespace rocjitsu::vfu
