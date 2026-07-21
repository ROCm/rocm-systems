// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vfu/gpu_pci_device.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace rocjitsu::vfu {

GpuPciDevice::GpuPciDevice(SimulatedKfd *kfd, uint32_t guest_pid,
                             uint64_t vram_bar_size)
    : kfd_(kfd),
      guest_pid_(guest_pid),
      bar0_(vram_bar_size),
      bar2_(BarSizes::kBar2Doorbell,
            [this](uint32_t offset, uint64_t val) {
              if (kfd_)
                kfd_->trigger_doorbell(guest_pid_, offset, val);
            }),
      bar5_(bar0_.fd(), bar0_.size()) {}

simdojo::PciId GpuPciDevice::pci_id() const {
  return {
      .vendor           = PciIdentity::kVendorId,
      .device           = PciIdentity::kDeviceId,
      .subsystem_vendor = PciIdentity::kSubsystemVendor,
      .subsystem_device = PciIdentity::kSubsystemDevice,
      .revision         = PciIdentity::kRevisionId,
      .class_code       = PciIdentity::kClassCode,
  };
}

std::vector<simdojo::BarSpec> GpuPciDevice::bars() const {
  return {
      {.index = 0, .size = bar0_.size(),
       .is_64bit = true,  .prefetchable = true},
      {.index = 2, .size = bar2_.size(),
       .is_64bit = true,  .prefetchable = true},
      {.index = 5, .size = BarSizes::kBar5Mmio,
       .is_64bit = false, .prefetchable = false},
  };
}

int GpuPciDevice::bar_fd(int bar_index) const {
  switch (bar_index) {
    case 0: return bar0_.fd();
    case 2: return bar2_.fd();
    default: return -1;
  }
}

ssize_t GpuPciDevice::bar_access(int bar_index, std::span<std::byte> buf,
                                   uint64_t offset, bool is_write) {
  char *cbuf = reinterpret_cast<char *>(buf.data());
  size_t count = buf.size();
  long loff = static_cast<long>(offset);

  switch (bar_index) {
    case 2:
      // Doorbell writes are trapped; reads return zeros.
      return bar2_.handle_access(cbuf, count, loff, is_write);

    case 5:
      return is_write ? bar5_.write(cbuf, count, loff)
                      : bar5_.read(cbuf, count, loff);

    case 0:
      // BAR0 is fully mmap-able (no callback registered); the transport should
      // not call bar_access for it. Log and fail gracefully.
      std::fprintf(stderr,
                   "[vfu/gpu] unexpected bar_access for mmap-only BAR0 "
                   "offset=0x%llx count=%zu\n",
                   static_cast<unsigned long long>(offset), count);
      errno = EINVAL;
      return -1;

    default:
      errno = EINVAL;
      return -1;
  }
}

void GpuPciDevice::dma_map(const simdojo::DmaRegion &region) {
  if (!kfd_ || !region.vaddr)
    return;
  kfd_->register_guest_dma(guest_pid_, region.iova, region.vaddr,
                            region.length, true);
}

void GpuPciDevice::dma_unmap(const simdojo::DmaRegion &region) {
  if (!kfd_)
    return;
  kfd_->register_guest_dma(guest_pid_, region.iova, nullptr,
                            region.length, false);
}

} // namespace rocjitsu::vfu
