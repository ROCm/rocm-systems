// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vfio_device_host.h
/// @brief libvfio-user transport host for a simdojo::PciDevice.
///
/// VfioDeviceHost is the sole owner of a vfu_ctx_t. It implements
/// simdojo::IrqSink (forwarding trigger() to vfu_irq_trigger) and
/// simdojo::DmaEngine (forwarding map/unmap to the device). All vfu_*
/// calls in the vfio-user front-end live here; the device model
/// (GpuPciDevice) contains zero libvfio-user references.
///
/// Litmus check:
///   <libvfio-user.h> must appear in exactly one .cpp file in this tree:
///   vfio_device_host.cpp. Zero other files include it.

#ifndef ROCJITSU_KMD_LINUX_VFIO_DEVICE_HOST_H_
#define ROCJITSU_KMD_LINUX_VFIO_DEVICE_HOST_H_

#include "simdojo/components/pci_device.h"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Forward-declare the opaque libvfio-user context to avoid pulling in the
// full header. The implementation file includes <libvfio-user.h>.
struct vfu_ctx;
typedef struct vfu_ctx vfu_ctx_t;

namespace rocjitsu {

/// @brief libvfio-user transport wrapper for a simdojo::PciDevice.
///
/// Owns the vfu_ctx_t and the poll thread. Implements IrqSink and DmaEngine
/// so the device model can trigger interrupts and receive DMA events through
/// transport-neutral interfaces.
class VfioDeviceHost : public simdojo::IrqSink,
                       public simdojo::DmaEngine {
public:
  /// @param socket_path  UNIX socket path the server listens on.
  /// @param device       Non-owning pointer; must outlive this object.
  VfioDeviceHost(std::string socket_path, simdojo::PciDevice *device);
  ~VfioDeviceHost() override;

  VfioDeviceHost(const VfioDeviceHost &) = delete;
  VfioDeviceHost &operator=(const VfioDeviceHost &) = delete;

  /// @brief Attach to the VMM and run the vfio-user event loop.
  ///        Blocks until the client disconnects or stop() is called.
  /// @returns 0 on clean exit, -1 on error.
  int run();

  /// @brief Request the event loop to exit (thread-safe).
  void stop();

  // --- simdojo::IrqSink ---
  void trigger(uint32_t vector) override;

  // --- simdojo::DmaEngine ---
  void map(const simdojo::DmaRegion &region) override;
  void unmap(const simdojo::DmaRegion &region) override;

  // DMA region entry — exposed so the BAR2 callback (a lambda) can read it.
  struct DmaEntry { uint64_t iova; void *vaddr; uint64_t length; };

private:
  int init();
  int setup_bars();
  int setup_pci_identity();

  // Fence/ring-test service thread.
  std::thread       fence_thread_;
  std::atomic<bool> fence_stop_{false};
  std::atomic<bool> sdma_needs_scan_{false};
  void fence_service_loop();

  // Registered DMA regions (system RAM) for GTT/SDMA ring access.
  std::vector<DmaEntry> dma_regions_;
  std::mutex            dma_mutex_;

  // Translate a guest IOVA to host virtual address.  Returns nullptr if not
  // mapped or if [iova, iova+len) spans multiple or no DMA regions.
  void *iova_to_hva(uint64_t iova, size_t len) const;

  std::string           socket_path_;
  simdojo::PciDevice   *device_;
  vfu_ctx_t            *ctx_ = nullptr;
  std::atomic<bool>     stop_requested_{false};
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_VFIO_DEVICE_HOST_H_
