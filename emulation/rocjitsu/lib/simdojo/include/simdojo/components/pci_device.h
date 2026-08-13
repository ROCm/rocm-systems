// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file pci_device.h
/// @brief Transport-agnostic PCI device model for the simulation graph.
///
/// A simdojo::PciDevice is the neutral, transport-independent object that a
/// PCI-attached front-end backs. It knows nothing about libvfio-user, KVM,
/// sockets, or any specific VMM: it declares its bus shape (identity, BARs,
/// interrupt count) and reacts to guest-driven events (BAR access, DMA
/// map/unmap, reset).
///
/// A transport binds to it by injecting an IrqSink and a DmaEngine, then
/// translating its wire protocol into calls on the device's virtual hooks.
/// This inversion allows a single device model to be driven by more than one
/// transport (e.g. a libvfio-user host today, a KVM path later), and allows
/// device kinds other than GPUs (e.g. a NIC) to implement the same interface
/// with zero coupling to any one device family.
///
/// No libvfio-user or VMM headers are included here.

#ifndef SIMDOJO_COMPONENTS_PCI_DEVICE_H_
#define SIMDOJO_COMPONENTS_PCI_DEVICE_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <sys/types.h>  // ssize_t
#include <vector>

namespace simdojo {

// ---------------------------------------------------------------------------
// Bus-shape descriptors
// ---------------------------------------------------------------------------

/// @brief PCI identity triple plus subystem IDs and class code.
struct PciId {
  uint16_t vendor           = 0;
  uint16_t device           = 0;
  uint16_t subsystem_vendor = 0;
  uint16_t subsystem_device = 0;
  uint8_t  revision         = 0;
  /// @brief 24-bit class code: (class << 16) | (subclass << 8) | prog_if.
  uint32_t class_code       = 0;
};

/// @brief Describes a single BAR (Base Address Register).
///
/// A 64-bit BAR implicitly consumes two physical BAR indices (index and
/// index+1); the transport handles the split internally.
struct BarSpec {
  /// @brief Physical BAR number (0, 2, 4, or 5 on a standard function).
  int index = 0;

  /// @brief Total size of the BAR region in bytes.
  uint64_t size = 0;

  /// @brief True for a 64-bit prefetchable BAR (type = 0b10, prefetchable = 1).
  bool is_64bit = false;

  /// @brief True for a prefetchable BAR; implies is_64bit for all modern GPUs.
  bool prefetchable = false;
};

/// @brief Describes a guest DMA memory region registered by the VMM.
///
/// iova and length are always valid. vaddr is the host virtual address of the
/// region when the VMM has mapped it into the process; nullptr means the region
/// is registered as an IOVA but is not directly host-accessible (the device
/// model should record the mapping but not dereference vaddr).
struct DmaRegion {
  uint64_t iova   = 0;   ///< Guest physical (IOVA) base address.
  size_t   length = 0;   ///< Length of the region in bytes.
  void    *vaddr  = nullptr; ///< Host virtual base address (may be nullptr).
};

// ---------------------------------------------------------------------------
// Transport interfaces injected into the device
// ---------------------------------------------------------------------------

/// @brief Abstract sink for device-to-guest interrupts.
///
/// Implemented by the transport (e.g. VfioDeviceHost calls vfu_irq_trigger).
/// The device model calls trigger() to raise an MSI-X vector.
class IrqSink {
public:
  virtual ~IrqSink() = default;
  /// @brief Trigger MSI-X interrupt @p vector toward the guest.
  virtual void trigger(uint32_t vector) = 0;
};

/// @brief Abstract DMA engine exposed to the device model.
///
/// Implemented by the transport. The device model calls map/unmap when the
/// VMM registers or unregisters guest DMA memory regions.
class DmaEngine {
public:
  virtual ~DmaEngine() = default;
  virtual void map(const DmaRegion &region)   = 0;
  virtual void unmap(const DmaRegion &region) = 0;
};

// ---------------------------------------------------------------------------
// PciDevice — the core abstract class
// ---------------------------------------------------------------------------

/// @brief Abstract PCI device model.
///
/// Subclass this to implement a specific device (GPU, NIC, …). Override the
/// virtual hooks; call irq_ and dma_ (injected by the transport after
/// construction via set_irq_sink / set_dma_engine) to interact back with the
/// guest.
class PciDevice {
public:
  virtual ~PciDevice() = default;

  // --- Bus shape (called once during transport setup) ---

  /// @brief PCI identity (vendor/device/class/revision).
  virtual PciId pci_id() const = 0;

  /// @brief BAR descriptors. The transport iterates this to set up regions.
  virtual std::vector<BarSpec> bars() const = 0;

  /// @brief Number of MSI-X vectors this device advertises (0 = MSI-X absent).
  virtual uint32_t msix_vectors() const = 0;

  /// @brief Optional per-BAR backing file descriptor.
  ///
  /// The transport passes this fd to the VMM so the guest can mmap the BAR
  /// directly without per-access trapping. Return -1 for BARs that have no
  /// backing fd (register-trapped BARs use bar_access instead).
  virtual int bar_fd(int bar_index) const { (void)bar_index; return -1; }

  // --- Guest-driven events ---

  /// @brief Handle a read or write to a BAR from the guest.
  ///
  /// @param bar_index  Physical BAR number (matches BarSpec::index).
  /// @param buf        Buffer: filled on read, consumed on write.
  /// @param offset     Byte offset within the BAR.
  /// @param is_write   True for a write, false for a read.
  /// @returns Bytes transferred on success, -1 on error (set errno).
  virtual ssize_t bar_access(int bar_index, std::span<std::byte> buf,
                              uint64_t offset, bool is_write) = 0;

  /// @brief A guest DMA region has been registered by the VMM.
  virtual void dma_map(const DmaRegion &region) { (void)region; }

  /// @brief A guest DMA region has been unregistered by the VMM.
  virtual void dma_unmap(const DmaRegion &region) { (void)region; }

  /// @brief Optional warm reset from the guest.
  virtual void reset() {}

  /// @brief Peek a 32-bit shadow register from the device's MMIO model without
  /// side effects.  Used by the transport to read SDMA ring-base registers when
  /// servicing a doorbell write.  Default returns 0.
  virtual uint32_t mmio_peek(uint32_t byte_offset) const {
    (void)byte_offset; return 0;
  }

  // --- Injection (called by the transport before any bar_access calls) ---

  void set_irq_sink(IrqSink *irq)     { irq_ = irq; }
  void set_dma_engine(DmaEngine *dma) { dma_ = dma; }

protected:
  /// @brief Transport-injected interrupt sink. May be nullptr before setup.
  IrqSink   *irq_ = nullptr;
  /// @brief Transport-injected DMA engine. May be nullptr before setup.
  DmaEngine *dma_ = nullptr;
};

// ---------------------------------------------------------------------------
// PciDeviceProvider
// ---------------------------------------------------------------------------

/// @brief Factory that yields PciDevice instances for a given device family.
///
/// The transport iterates provider.pci_devices() and creates one transport
/// endpoint per device. Adding a second device type (NIC, crypto engine, …)
/// means registering an additional provider — zero GPU files touched.
class PciDeviceProvider {
public:
  virtual ~PciDeviceProvider() = default;
  /// @brief Returns pointers to all devices this provider owns.
  ///        The provider retains ownership; pointers are valid for its lifetime.
  virtual std::vector<PciDevice *> pci_devices() = 0;
};

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_PCI_DEVICE_H_
