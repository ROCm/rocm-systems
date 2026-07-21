// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vfu/vfu_server.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/rj_vm_impl.h"

// Public C API
#include "rocjitsu/vm/rj_vm.h"

#include <libvfio-user.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace rocjitsu::vfu {

VfuServer::VfuServer(VfuServerOptions opts) : opts_(std::move(opts)) {}

VfuServer::~VfuServer() {
  if (vm_handle_) {
    rj_vm_device_close(vm_handle_, guest_process_id_);
    rj_vm_destroy(vm_handle_);
  }
  if (ctx_)
    vfu_destroy_ctx(ctx_);
}

void VfuServer::stop() {
  stop_requested_.store(true, std::memory_order_relaxed);
}

void VfuServer::on_doorbell_write(uint32_t doorbell_offset, uint64_t value) {
  if (!driver_)
    return;
  driver_->trigger_doorbell(guest_process_id_, doorbell_offset, value);
}

int VfuServer::init() {
  // -----------------------------------------------------------------------
  // 1. Create the rocjitsu VM in DAEMON mode via the public C API.
  //    DAEMON mode backs all GPU allocations with memfds for cross-process
  //    sharing (the guest DMA engine needs the same handles).
  // -----------------------------------------------------------------------
  rj_status_t st = rj_vm_create(opts_.config_path.c_str(), RJ_VM_MODE_DAEMON, &vm_handle_);
  if (st != ROCJITSU_STATUS_SUCCESS) {
    std::fprintf(stderr, "[vfu] rj_vm_create failed: %d (config: %s)\n",
                 static_cast<int>(st), opts_.config_path.c_str());
    return -1;
  }

  // Reach into the internal VM to get the SimulatedKfd pointer.
  // This is intentional: VfuServer is part of the rocjitsu library, not an
  // external consumer, so accessing rj_vm_impl is appropriate here.
  driver_ = vm_handle_->vm ? vm_handle_->vm->driver() : nullptr;
  if (!driver_) {
    std::fprintf(stderr, "[vfu] SimulatedKfd not initialized\n");
    return -1;
  }

  // Open a guest process (analogous to the guest kernel opening /dev/kfd).
  st = rj_vm_device_open(vm_handle_, 0, &guest_process_id_);
  if (st != ROCJITSU_STATUS_SUCCESS) {
    std::fprintf(stderr, "[vfu] rj_vm_device_open failed: %d\n", static_cast<int>(st));
    return -1;
  }

  // -----------------------------------------------------------------------
  // 2. Create the libvfio-user context bound to the UNIX socket.
  // -----------------------------------------------------------------------
  ctx_ = vfu_create_ctx(VFU_TRANS_SOCK,
                        opts_.socket_path.c_str(),
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

  // -----------------------------------------------------------------------
  // 3. Construct BAR handlers.
  // -----------------------------------------------------------------------
  bar0_ = std::make_unique<Bar0Vram>(opts_.vram_bar_size);
  if (bar0_->setup(ctx_) != 0)
    return -1;

  bar2_ = std::make_unique<Bar2Doorbell>(
      BarSizes::kBar2Doorbell,
      [this](uint32_t offset, uint64_t val) { on_doorbell_write(offset, val); });
  if (bar2_->valid() != 0) {
    std::fprintf(stderr, "[vfu] BAR2 doorbell memfd initialization failed\n");
    return -1;
  }

  // BAR2/3: 64-bit prefetchable doorbell region. Use a lambda trampoline
  // (same pattern as BAR5) so vfu_get_private(ctx) → VfuServer* → bar2_.
  auto bar2_cb = [](vfu_ctx_t *c, char *buf, size_t cnt,
                    long off, bool wr) -> ssize_t {
    auto *srv = reinterpret_cast<VfuServer *>(vfu_get_private(c));
    if (!srv || !srv->bar2_)
      return -1;
    return srv->bar2_->handle_access(buf, cnt, off, wr);
  };

  iovec bar2_mmap{.iov_base = nullptr, .iov_len = bar2_->size()};
  if (vfu_setup_region(ctx_,
                       VFU_PCI_DEV_BAR2_REGION_IDX,
                       bar2_->size(),
                       bar2_cb,
                       VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM |
                         VFU_REGION_FLAG_64_BITS | VFU_REGION_FLAG_PREFETCH,
                       &bar2_mmap, 1,
                       bar2_->fd(), 0) != 0) {
    std::perror("vfu_setup_region BAR2");
    return -1;
  }

  bar5_ = std::make_unique<MmioModel>(bar0_->fd(), bar0_->size());

  // -----------------------------------------------------------------------
  // 4. Register BAR5 with a lambda trampoline that routes through VfuServer.
  //    vfu_get_private(ctx) returns the VfuServer* (== this).
  //    BAR0 and BAR2 were registered inside bar0_->setup() and bar2_->setup().
  // -----------------------------------------------------------------------
  // vfu_region_access_cb_t uses loff_t (== long on Linux) for offset.
  auto bar5_cb = [](vfu_ctx_t *c, char *buf, size_t cnt,
                    long off, bool wr) -> ssize_t {
    auto *srv = reinterpret_cast<VfuServer *>(vfu_get_private(c));
    if (!srv || !srv->bar5_)
      return -1;
    return wr ? srv->bar5_->write(buf, cnt, off)
              : srv->bar5_->read(buf, cnt, off);
  };

  // BAR5: 32-bit non-prefetchable MMIO register window (256 KB).
  // VFU_REGION_FLAG_MEM declares it as a memory BAR (not I/O).
  // No 64_BITS or PREFETCH flags — amdgpu_device.c calls
  // pci_resource_start(pdev, 5) which is the 32-bit non-prefetchable BAR.
  if (vfu_setup_region(ctx_,
                       VFU_PCI_DEV_BAR5_REGION_IDX,
                       BarSizes::kBar5Mmio,
                       bar5_cb,
                       VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM,
                       nullptr, 0, -1, 0) != 0) {
    std::perror("vfu_setup_region BAR5");
    return -1;
  }

  // PCI identity, capabilities, and interrupt config.
  if (setup_pci_config(ctx_, bar0_->fd(), opts_.vram_bar_size, bar2_->fd()) != 0)
    return -1;

  // -----------------------------------------------------------------------
  // 5. DMA map/unmap callbacks (guest memory registration).
  // -----------------------------------------------------------------------
  dma_ = std::make_unique<DmaMapper>(*driver_, guest_process_id_);

  if (vfu_setup_device_dma(ctx_, LIBVFIO_USER_MAX_DMA_REGIONS, DmaMapper::dma_register, DmaMapper::dma_unregister) != 0) {
    std::perror("vfu_setup_device_dma");
    return -1;
  }

  // -----------------------------------------------------------------------
  // 6. Finalize context.
  // -----------------------------------------------------------------------
  if (vfu_realize_ctx(ctx_) != 0) {
    std::perror("vfu_realize_ctx");
    return -1;
  }

  return 0;
}

int VfuServer::run() {
  if (init() != 0)
    return -1;

  std::fprintf(stderr, "[vfu] listening on %s\n", opts_.socket_path.c_str());

  // Wait for QEMU to connect.
  if (vfu_attach_ctx(ctx_) != 0) {
    std::perror("vfu_attach_ctx");
    return -1;
  }

  std::fprintf(stderr, "[vfu] client connected\n");

  // Event loop: process one vfio-user message per iteration.
  int ret = 0;
  while (!stop_requested_.load(std::memory_order_relaxed)) {
    ret = vfu_run_ctx(ctx_);
    if (ret != 0) {
      if (errno == EAGAIN)
        continue;
      if (errno == ENOTCONN) {
        std::fprintf(stderr, "[vfu] client disconnected\n");
        ret = 0;
      } else {
        std::perror("vfu_run_ctx");
      }
      break;
    }
  }

  return ret;
}

} // namespace rocjitsu::vfu
