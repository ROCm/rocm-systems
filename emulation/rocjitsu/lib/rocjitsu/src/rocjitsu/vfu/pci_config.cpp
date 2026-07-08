// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vfu/pci_config.h"
#include "rocjitsu/vfu/bar5_mmio.h"
#include "rocjitsu/vfu/mmio_registers.h"

#include <libvfio-user.h>
#include <vfio-user/pci_defs.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace rocjitsu::vfu {

namespace {

// ---------------------------------------------------------------------------
// PCI Express Capability — matches the pci_caps/px.h layout expected by
// libvfio-user.  We use the raw struct from pci_caps/px.h when available;
// here we build a compatible byte blob.
// ---------------------------------------------------------------------------
struct [[gnu::packed]] PcieCapability {
  uint8_t  cap_id        = 0x10; ///< PCI_CAP_ID_EXP
  uint8_t  next_ptr      = 0x00;
  uint16_t pcie_cap      = 0x0002; ///< PCIe Endpoint, v2
  uint32_t dev_cap       = 0x00028000;
  uint16_t dev_ctrl      = 0x0000;
  uint16_t dev_status    = 0x0000;
  uint32_t link_cap      = 0x00040C05; ///< x16, Gen5
  uint16_t link_ctrl     = 0x0000;
  uint16_t link_status   = 0x0000;
  uint32_t slot_cap      = 0;
  uint16_t slot_ctrl     = 0;
  uint16_t slot_status   = 0;
  uint16_t root_ctrl     = 0;
  uint16_t root_cap      = 0;
  uint32_t root_status   = 0;
  uint32_t dev_cap2      = 0;
  uint16_t dev_ctrl2     = 0;
  uint16_t dev_status2   = 0;
  uint32_t link_cap2     = 0x0000003F;
  uint16_t link_ctrl2    = 0;
  uint16_t link_status2  = 0;
};

// ---------------------------------------------------------------------------
// Power Management Capability
// ---------------------------------------------------------------------------
struct [[gnu::packed]] PmCapability {
  uint8_t  cap_id    = 0x01;
  uint8_t  next_ptr  = 0x00;
  uint16_t pmc       = 0x0003; ///< PME capable, version 1.1
  uint16_t pmcsr     = 0x0000;
  uint8_t  pmcsr_bse = 0x00;
  uint8_t  data      = 0x00;
};

// ---------------------------------------------------------------------------
// MSI-X Capability
// ---------------------------------------------------------------------------
struct [[gnu::packed]] MsixCapability {
  uint8_t  cap_id    = 0x11;
  uint8_t  next_ptr  = 0x00;
  // msg_ctrl: enable bit (bit 15) + (N-1) in bits [10:0]
  uint16_t msg_ctrl  = static_cast<uint16_t>((kMsiXVectors - 1) | 0x8000);
  // Table in BAR5, offset 0
  uint32_t table_bir = 0x00000005;
  // PBA in BAR5, offset 0x2000
  uint32_t pba_bir   = 0x00002005;
};

} // namespace

int setup_pci_config(vfu_ctx_t *ctx, int /*vram_fd*/, uint64_t /*vram_size*/,
                     int /*doorbell_fd*/) {
  // -----------------------------------------------------------------------
  // 1. PCI identity (vendor, device, subsystem, class, revision)
  //    vfu_pci_set_id sets vendor+device+subvendor+subdevice in config space.
  // -----------------------------------------------------------------------
  // vfu_pci_set_id returns void.
  vfu_pci_set_id(ctx,
                 PciIdentity::kVendorId,
                 PciIdentity::kDeviceId,
                 PciIdentity::kSubsystemVendor,
                 PciIdentity::kSubsystemDevice);

  // Set class code: base=0x12 (processing accelerator), sub=0x00, pi=0x00.
  // PCI_CLASS_ACCELERATOR_PROCESSING is what real AMD Instinct GPUs advertise
  // and what the amdgpu CHIP_IP_DISCOVERY wildcard entry matches on.
  vfu_pci_set_class(ctx, 0x12, 0x00, 0x00);

  // Set revision via raw config space pointer.
  auto *cs = vfu_pci_get_config_space(ctx);
  if (cs)
    cs->hdr.rid = PciIdentity::kRevisionId;

  // -----------------------------------------------------------------------
  // 2. PCI capabilities
  // -----------------------------------------------------------------------
  PcieCapability pcie_cap{};
  if (vfu_pci_add_capability(ctx, 0, 0,
                             reinterpret_cast<char *>(&pcie_cap)) < 0) {
    std::perror("vfu_pci_add_capability PCIe");
    return -1;
  }

  PmCapability pm_cap{};
  if (vfu_pci_add_capability(ctx, 0, 0,
                             reinterpret_cast<char *>(&pm_cap)) < 0) {
    std::perror("vfu_pci_add_capability PM");
    return -1;
  }

  MsixCapability msix_cap{};
  if (vfu_pci_add_capability(ctx, 0, VFU_CAP_FLAG_READONLY,
                             reinterpret_cast<char *>(&msix_cap)) < 0) {
    std::perror("vfu_pci_add_capability MSI-X");
    return -1;
  }

  // -----------------------------------------------------------------------
  // 3. Interrupt setup
  // -----------------------------------------------------------------------
  if (vfu_setup_device_nr_irqs(ctx, VFU_DEV_INTX_IRQ, 1) != 0) {
    std::perror("vfu_setup_device_nr_irqs INTx");
    return -1;
  }
  if (vfu_setup_device_nr_irqs(ctx, VFU_DEV_MSIX_IRQ,
                                static_cast<uint32_t>(kMsiXVectors)) != 0) {
    std::perror("vfu_setup_device_nr_irqs MSI-X");
    return -1;
  }

  return 0;
}

} // namespace rocjitsu::vfu
