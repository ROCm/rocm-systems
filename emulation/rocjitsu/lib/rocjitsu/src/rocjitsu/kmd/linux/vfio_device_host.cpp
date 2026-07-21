// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// THIS IS THE ONLY FILE IN THE rocjitsu vfio-user FRONT-END THAT MAY INCLUDE
// <libvfio-user.h>. All vfu_* calls must live here; device models and BAR
// handlers must remain transport-neutral.

#include "rocjitsu/kmd/linux/vfio_device_host.h"

#include <libvfio-user.h>
#include <vfio-user/pci_defs.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <span>

namespace rocjitsu {

namespace {

// ---------------------------------------------------------------------------
// PCI capability structs (only used inside this translation unit)
// ---------------------------------------------------------------------------

struct [[gnu::packed]] PcieCapability {
  uint8_t  cap_id       = 0x10;
  uint8_t  next_ptr     = 0x00;
  uint16_t pcie_cap     = 0x0002; // PCIe Endpoint, v2
  uint32_t dev_cap      = 0x00028000;
  uint16_t dev_ctrl     = 0x0000;
  uint16_t dev_status   = 0x0000;
  uint32_t link_cap     = 0x00040C05; // x16, Gen5
  uint16_t link_ctrl    = 0x0000;
  uint16_t link_status  = 0x0000;
  uint32_t slot_cap     = 0;
  uint16_t slot_ctrl    = 0;
  uint16_t slot_status  = 0;
  uint16_t root_ctrl    = 0;
  uint16_t root_cap     = 0;
  uint32_t root_status  = 0;
  uint32_t dev_cap2     = 0;
  uint16_t dev_ctrl2    = 0;
  uint16_t dev_status2  = 0;
  uint32_t link_cap2    = 0x0000003F;
  uint16_t link_ctrl2   = 0;
  uint16_t link_status2 = 0;
};

struct [[gnu::packed]] PmCapability {
  uint8_t  cap_id    = 0x01;
  uint8_t  next_ptr  = 0x00;
  uint16_t pmc       = 0x0003; // PME capable, v1.1
  uint16_t pmcsr     = 0x0000;
  uint8_t  pmcsr_bse = 0x00;
  uint8_t  data      = 0x00;
};

struct [[gnu::packed]] MsixCapability {
  uint8_t  cap_id   = 0x11;
  uint8_t  next_ptr = 0x00;
  uint16_t msg_ctrl;   // set from msix_vectors in setup
  uint32_t table_bir;  // BAR5, offset 0
  uint32_t pba_bir;    // BAR5, offset 0x2000
};

} // namespace

// ---------------------------------------------------------------------------
// VfioDeviceHost — transport wrapper
// ---------------------------------------------------------------------------

VfioDeviceHost::VfioDeviceHost(std::string socket_path,
                                simdojo::PciDevice *device)
    : socket_path_(std::move(socket_path)), device_(device) {}

VfioDeviceHost::~VfioDeviceHost() {
  if (ctx_)
    vfu_destroy_ctx(ctx_);
}

void VfioDeviceHost::stop() {
  stop_requested_.store(true, std::memory_order_relaxed);
}

void VfioDeviceHost::trigger(uint32_t vector) {
  if (ctx_)
    vfu_irq_trigger(ctx_, vector);
}

void VfioDeviceHost::map(const simdojo::DmaRegion &region) {
  device_->dma_map(region);
}

void VfioDeviceHost::unmap(const simdojo::DmaRegion &region) {
  device_->dma_unmap(region);
}

int VfioDeviceHost::setup_bars() {
  for (const auto &bar : device_->bars()) {
    int fd = device_->bar_fd(bar.index);
    uint32_t flags = VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM;
    if (bar.is_64bit)    flags |= VFU_REGION_FLAG_64_BITS;
    if (bar.prefetchable) flags |= VFU_REGION_FLAG_PREFETCH;

    if (fd >= 0) {
      // Memfd-backed BAR: fully mmap-able, no per-access callback.
      iovec mmap_area{.iov_base = nullptr, .iov_len = bar.size};
      if (vfu_setup_region(ctx_, bar.index, bar.size, nullptr,
                           flags, &mmap_area, 1, fd, 0) != 0) {
        std::perror("vfu_setup_region (mmap BAR)");
        return -1;
      }
    } else {
      // Register-trapped BAR: install a lambda trampoline.
      // The lambda captures bar.index so the device can dispatch by BAR.
      // vfu_get_private(ctx) → VfioDeviceHost* → device_->bar_access().
      int bar_index = bar.index;
      auto cb = [](vfu_ctx_t *c, char *buf, size_t cnt,
                   long off, bool wr) -> ssize_t {
        auto *host = reinterpret_cast<VfioDeviceHost *>(vfu_get_private(c));
        if (!host || !host->device_)
          return -1;
        // Recover bar_index from the region index via libvfio-user's opaque.
        // The callback is registered once per BAR; encode the index in the
        // closure via a second level of indirection using a helper struct
        // stored as a persistent object on the host.
        //
        // Simpler approach: since we have only one register-trapped BAR (BAR5),
        // route to bar_access(5, …) directly. If additional trapped BARs are
        // added later, store a per-BAR dispatch table in VfioDeviceHost.
        //
        // For now, bar_index is captured from the enclosing lambda — but C
        // function pointers can't capture. Use the vfu region index instead:
        // vfu passes the region index as part of the callback type but not in
        // the current vfu_region_access_cb_t signature. We rely on the fact
        // that BAR5 is the only trapped BAR and route directly.
        std::span<std::byte> span(reinterpret_cast<std::byte *>(buf), cnt);
        return host->device_->bar_access(5, span,
                                         static_cast<uint64_t>(off), wr);
      };
      (void)bar_index; // used in the comment above; silence warning

      if (vfu_setup_region(ctx_, bar.index, bar.size, cb,
                           flags, nullptr, 0, -1, 0) != 0) {
        std::perror("vfu_setup_region (trapped BAR)");
        return -1;
      }
    }
  }
  return 0;
}

int VfioDeviceHost::setup_pci_identity() {
  auto id = device_->pci_id();

  vfu_pci_set_id(ctx_, id.vendor, id.device,
                 id.subsystem_vendor, id.subsystem_device);
  vfu_pci_set_class(ctx_,
                    static_cast<uint8_t>((id.class_code >> 16) & 0xFF),
                    static_cast<uint8_t>((id.class_code >>  8) & 0xFF),
                    static_cast<uint8_t>( id.class_code        & 0xFF));
  auto *cs = vfu_pci_get_config_space(ctx_);
  if (cs)
    cs->hdr.rid = id.revision;

  // PCI capabilities
  PcieCapability pcie{};
  if (vfu_pci_add_capability(ctx_, 0, 0,
                              reinterpret_cast<char *>(&pcie)) < 0) {
    std::perror("vfu_pci_add_capability PCIe");
    return -1;
  }

  PmCapability pm{};
  if (vfu_pci_add_capability(ctx_, 0, 0,
                              reinterpret_cast<char *>(&pm)) < 0) {
    std::perror("vfu_pci_add_capability PM");
    return -1;
  }

  uint32_t nvec = device_->msix_vectors();
  if (nvec > 0) {
    MsixCapability msix{};
    msix.msg_ctrl  = static_cast<uint16_t>((nvec - 1) | 0x8000);
    msix.table_bir = 0x00000005; // BAR5, offset 0
    msix.pba_bir   = 0x00002005; // BAR5, offset 0x2000
    if (vfu_pci_add_capability(ctx_, 0, VFU_CAP_FLAG_READONLY,
                                reinterpret_cast<char *>(&msix)) < 0) {
      std::perror("vfu_pci_add_capability MSI-X");
      return -1;
    }
    if (vfu_setup_device_nr_irqs(ctx_, VFU_DEV_INTX_IRQ, 1) != 0) {
      std::perror("vfu_setup_device_nr_irqs INTx");
      return -1;
    }
    if (vfu_setup_device_nr_irqs(ctx_, VFU_DEV_MSIX_IRQ, nvec) != 0) {
      std::perror("vfu_setup_device_nr_irqs MSI-X");
      return -1;
    }
  }

  return 0;
}

int VfioDeviceHost::init() {
  ctx_ = vfu_create_ctx(VFU_TRANS_SOCK,
                        socket_path_.c_str(),
                        0,
                        this, // private data — retrieved via vfu_get_private()
                        VFU_DEV_TYPE_PCI);
  if (!ctx_) {
    std::perror("vfu_create_ctx");
    return -1;
  }

  if (vfu_pci_init(ctx_, VFU_PCI_TYPE_EXPRESS, PCI_HEADER_TYPE_NORMAL, 0) != 0) {
    std::perror("vfu_pci_init");
    return -1;
  }

  if (setup_bars() != 0)
    return -1;

  if (setup_pci_identity() != 0)
    return -1;

  // DMA map/unmap callbacks — route through VfioDeviceHost to DmaEngine.
  auto dma_reg = [](vfu_ctx_t *c, vfu_dma_info_t *info) {
    auto *host = reinterpret_cast<VfioDeviceHost *>(vfu_get_private(c));
    if (!host || !info)
      return;
    simdojo::DmaRegion r;
    r.iova   = reinterpret_cast<uint64_t>(info->iova.iov_base);
    r.length = info->iova.iov_len;
    r.vaddr  = info->vaddr;
    host->map(r);
  };
  auto dma_unreg = [](vfu_ctx_t *c, vfu_dma_info_t *info) {
    auto *host = reinterpret_cast<VfioDeviceHost *>(vfu_get_private(c));
    if (!host || !info)
      return;
    simdojo::DmaRegion r;
    r.iova   = reinterpret_cast<uint64_t>(info->iova.iov_base);
    r.length = info->iova.iov_len;
    r.vaddr  = info->vaddr;
    host->unmap(r);
  };

  if (vfu_setup_device_dma(ctx_, LIBVFIO_USER_MAX_DMA_REGIONS,
                             dma_reg, dma_unreg) != 0) {
    std::perror("vfu_setup_device_dma");
    return -1;
  }

  if (vfu_realize_ctx(ctx_) != 0) {
    std::perror("vfu_realize_ctx");
    return -1;
  }

  // Inject transport interfaces into the device so it can trigger interrupts
  // and receive DMA notifications without knowing about vfu_*.
  device_->set_irq_sink(this);
  device_->set_dma_engine(this);

  return 0;
}

int VfioDeviceHost::run() {
  if (init() != 0)
    return -1;

  std::fprintf(stderr, "[vfio-host] listening on %s\n", socket_path_.c_str());

  if (vfu_attach_ctx(ctx_) != 0) {
    std::perror("vfu_attach_ctx");
    return -1;
  }

  std::fprintf(stderr, "[vfio-host] client connected\n");

  int ret = 0;
  while (!stop_requested_.load(std::memory_order_relaxed)) {
    ret = vfu_run_ctx(ctx_);
    if (ret != 0) {
      if (errno == EAGAIN)
        continue;
      if (errno == ENOTCONN) {
        std::fprintf(stderr, "[vfio-host] client disconnected\n");
        ret = 0;
      } else {
        std::perror("vfu_run_ctx");
      }
      break;
    }
  }

  return ret;
}

} // namespace rocjitsu
