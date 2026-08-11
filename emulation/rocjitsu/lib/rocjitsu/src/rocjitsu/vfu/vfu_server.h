// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vfu_server.h
/// @brief Top-level libvfio-user server for the rocjitsu GPU emulator.
///
/// VfuServer owns a libvfio-user context, a rocjitsu VirtualMachine, and the
/// three PCIe BAR handlers (BAR0 VRAM, BAR2 doorbell, BAR5 MMIO). It drives
/// the vfu_run_ctx event loop, fielding QEMU messages and forwarding them to
/// the appropriate subsystem.
///
/// Usage:
///   VfuServer server("/tmp/rocjitsu-vfu-0.sock", "configs/gfx950_mi350p_kmd.json");
///   server.run();  // blocks until QEMU disconnects or stop() is called

#ifndef ROCJITSU_VFU_VFU_SERVER_H_
#define ROCJITSU_VFU_VFU_SERVER_H_

#include "rocjitsu/vfu/bar0_vram.h"
#include "rocjitsu/vfu/bar2_doorbell.h"
#include "rocjitsu/vfu/bar5_mmio.h"
#include "rocjitsu/vfu/dma_mapper.h"
#include "rocjitsu/vfu/pci_config.h"
#include "rocjitsu/vm/rj_vm.h"

#include <atomic>
#include <memory>
#include <string>
#include <thread>

// Forward-declare libvfio-user types.
typedef struct vfu_ctx vfu_ctx_t;

namespace rocjitsu {
class SimulatedKfd;
} // namespace rocjitsu

namespace rocjitsu::vfu {

/// @brief Options for the VfuServer.
struct VfuServerOptions {
  /// UNIX socket path the server listens on (QEMU connects here).
  std::string socket_path;

  /// Path to the rocjitsu JSON topology config file (e.g. gfx950_mi350p_kmd.json).
  std::string config_path;

  /// VRAM BAR size in bytes. Use BarSizes::kBar0VramDefault (256 MB) without
  /// ReBAR, or BarSizes::kBar0VramFull (144 GB) for full ReBAR mode.
  uint64_t vram_bar_size = BarSizes::kBar0VramDefault;
};

/// @brief libvfio-user GPU server: exposes rocjitsu as a PCIe AMD GPU in a VM.
class VfuServer {
public:
  explicit VfuServer(VfuServerOptions opts);
  ~VfuServer();

  VfuServer(const VfuServer &) = delete;
  VfuServer &operator=(const VfuServer &) = delete;

  /// @brief Attach to QEMU and run the event loop (blocks until disconnect).
  /// @returns 0 on clean exit, -1 on error.
  int run();

  /// @brief Request the event loop to exit (thread-safe).
  void stop();

  DmaMapper *dma() const { return dma_.get(); }

private:
  int init();
  void on_doorbell_write(uint32_t doorbell_offset, uint64_t value);

  VfuServerOptions opts_;
  vfu_ctx_t *ctx_ = nullptr;

  /// @brief Background thread that services fence completions and ring test mocks.
  std::thread fence_thread_;
  std::atomic<bool> fence_stop_{false};
  void fence_service_loop(int vram_fd, uint64_t vram_size);

  /// @brief Find QEMU's HVA for the BAR5 (256 KB) region by scanning /proc/maps.
  /// Returns 0 if not found.
  static uintptr_t find_qemu_bar5_hva();

  rj_vm_t *vm_handle_ = nullptr;  ///< Owned VM handle (from rj_vm_create).
  rocjitsu::SimulatedKfd *driver_ = nullptr; ///< Non-owning pointer into vm_handle_.

  std::unique_ptr<Bar0Vram>    bar0_;
  std::unique_ptr<Bar2Doorbell> bar2_;
  std::unique_ptr<MmioModel>   bar5_;
  std::unique_ptr<DmaMapper>   dma_;

  uint32_t guest_process_id_ = 0;
  std::atomic<bool> stop_requested_{false};
};

} // namespace rocjitsu::vfu

#endif // ROCJITSU_VFU_VFU_SERVER_H_
