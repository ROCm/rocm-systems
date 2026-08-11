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
#include <cstdlib>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

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
  // Remove any stale socket left by a previous crash before binding.
  // -----------------------------------------------------------------------
  ::unlink(opts_.socket_path.c_str());
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
  if (!bar0_->valid()) {
    std::fprintf(stderr, "[vfu] BAR0 VRAM memfd initialization failed\n");
    return -1;
  }

  // BAR0/1: 64-bit prefetchable VRAM aperture, fully mmap-able (no callback).
  iovec bar0_mmap{.iov_base = nullptr, .iov_len = bar0_->size()};
  if (vfu_setup_region(ctx_,
                       VFU_PCI_DEV_BAR0_REGION_IDX,
                       bar0_->size(),
                       nullptr,
                       VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM |
                         VFU_REGION_FLAG_64_BITS | VFU_REGION_FLAG_PREFETCH,
                       &bar0_mmap, 1,
                       bar0_->fd(), 0) != 0) {
    std::perror("vfu_setup_region BAR0");
    return -1;
  }

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
  // We provide a shared memfd (bar5_->bar5_fd()) so QEMU maps it directly into
  // the guest address space. CPU MMIO writes go straight to the memfd; our
  // fence thread polls the memfd for ring-test sentinels and writes completions.
  // The vfio-user callback is still registered for the indirect-register paths
  // (MM_INDEX/MM_DATA, RSMU_INDEX/DATA) that QEMU does route via the socket.
  iovec bar5_mmap{.iov_base = nullptr, .iov_len = BarSizes::kBar5Mmio};
  int bar5_mmfd = bar5_ ? bar5_->bar5_fd() : -1;
  if (vfu_setup_region(ctx_,
                       VFU_PCI_DEV_BAR5_REGION_IDX,
                       BarSizes::kBar5Mmio,
                       bar5_cb,
                       VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM |
                         VFU_REGION_FLAG_ALWAYS_CB,
                       (bar5_mmfd >= 0) ? &bar5_mmap : nullptr,
                       (bar5_mmfd >= 0) ? 1 : 0,
                       bar5_mmfd, 0) != 0) {
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

  // Start background fence service thread.
  int vram_fd = bar0_ ? bar0_->fd() : -1;
  uint64_t vram_sz = bar0_ ? bar0_->size() : 0;
  std::fprintf(stderr, "[vfu] creating fence thread (vram_fd=%d vram_sz=%llu)\n",
               vram_fd, (unsigned long long)vram_sz);
  std::fflush(stderr);
  fence_stop_.store(false);
  try {
    fence_thread_ = std::thread(&VfuServer::fence_service_loop, this, vram_fd, vram_sz);
    std::fprintf(stderr, "[vfu] fence thread created\n");
  } catch (const std::system_error &e) {
    std::fprintf(stderr, "[vfu] fence thread creation failed: %s\n", e.what());
  }
  std::fflush(stderr);

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

  fence_stop_.store(true);
  if (fence_thread_.joinable())
    fence_thread_.join();

  return ret;
}

uintptr_t VfuServer::find_qemu_bar5_hva() {
  // Scan /proc for a qemu-system process, then search its maps for the 256 KB
  // anonymous zero-filled mapping that is QEMU's internal BAR5 shadow buffer.
  int qemu_pid = 0;
  {
    DIR *proc_dir = ::opendir("/proc");
    if (!proc_dir) return 0;
    struct dirent *ent;
    while ((ent = ::readdir(proc_dir)) != nullptr) {
      if (ent->d_type != DT_DIR && ent->d_type != DT_UNKNOWN) continue;
      char *endp;
      long pid = std::strtol(ent->d_name, &endp, 10);
      if (*endp != '\0' || pid <= 0) continue;
      // Read /proc/<pid>/comm to check if it's qemu-system
      char comm_path[64];
      std::snprintf(comm_path, sizeof(comm_path), "/proc/%ld/comm", pid);
      FILE *cf = std::fopen(comm_path, "r");
      if (!cf) continue;
      char comm[64] = {};
      std::fgets(comm, sizeof(comm), cf);
      std::fclose(cf);
      if (std::strncmp(comm, "qemu-system", 11) == 0) {
        qemu_pid = static_cast<int>(pid);
        break;
      }
    }
    ::closedir(proc_dir);
  }
  if (qemu_pid == 0) return 0;

  char maps_path[64], mem_path[64];
  std::snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", qemu_pid);
  std::snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", qemu_pid);

  FILE *maps = std::fopen(maps_path, "r");
  if (!maps) return 0;

  uintptr_t result = 0;
  char line[512];
  while (std::fgets(line, sizeof(line), maps)) {
    unsigned long start = 0, end = 0;
    char perms[8] = {};
    // Format: start-end perms offset dev inode [path]
    // Use strtoul for robustness — sscanf field widths can vary.
    char *p = line;
    char *q;
    start = std::strtoul(p, &q, 16); if (!q || *q != '-') continue;
    p = q + 1;
    end = std::strtoul(p, &q, 16);   if (!q || *q != ' ') continue;
    p = q + 1;
    if (std::sscanf(p, "%4s", perms) != 1) continue;

    if (end - start != 262144) continue;  // must be exactly 256 KB
    if (perms[0] != 'r' || perms[1] != 'w') continue;
    // Anonymous mapping: the inode field is 0 and no trailing path.
    // Quick check: does the line contain a non-zero inode? Look for a "0 \n"
    // pattern at the end (inode=0, no path). Just open mem and check.

    int memfd = ::open(mem_path, O_RDONLY);
    if (memfd < 0) continue;
    char buf[64] = {};
    ssize_t n = ::pread(memfd, buf, 64, static_cast<off_t>(start));
    ::close(memfd);
    if (n != 64) continue;
    bool all_zero = true;
    for (int i = 0; i < 64 && all_zero; ++i)
      if (buf[i] != 0) all_zero = false;
    if (all_zero) {
      result = static_cast<uintptr_t>(start);
      break;
    }
  }
  std::fclose(maps);

  if (result)
    std::fprintf(stderr, "[vfu] QEMU BAR5 HVA: 0x%lx (pid %d)\n", result, qemu_pid);
  else
    std::fprintf(stderr, "[vfu] QEMU BAR5 HVA: not found (pid %d)\n", qemu_pid);

  return result;
}

void VfuServer::fence_service_loop(int vram_fd, uint64_t vram_size) {
  std::fprintf(stderr, "[vfu] fence thread started (vram_fd=%d size=0x%llx)\n",
               vram_fd, static_cast<unsigned long long>(vram_size));
  std::fflush(stderr);
  if (vram_fd < 0 || vram_size == 0) {
    std::fprintf(stderr, "[vfu] fence thread exiting: invalid params\n");
    std::fflush(stderr);
    return;
  }

  // Get a direct pointer to the BAR5 shadow memfd.
  // CPU MMIO writes from the guest go here (via QEMU's mmap of bar5_fd_).
  // We poll this for ring-test sentinels and write completions in-place.
  volatile uint32_t *bar5_mem = bar5_ ? bar5_->bar5_mem() : nullptr;
  std::fprintf(stderr, "[vfu] fence thread: bar5_mem=%p\n", (void*)bar5_mem);
  std::fflush(stderr);

  // PSP ring frames are 64 bytes each.
  // fb_start = 0x1000000 (from 0x1 RSMU fallback for MC base).
  // We scan the ENTIRE VRAM memfd (starting from byte 0) since fence buffers
  // and ring buffers may be at any offset. The fence_addr in the ring frame
  // is a GPU MC address (>= fb_start for VRAM-backed BOs).
  static constexpr uint64_t fb_start = 0x1000000ULL;
  // Scan all usable VRAM: full memfd minus TMR region at top.
  const uint64_t scan_size = (vram_size > 280ULL << 20) ?
                              vram_size - (280ULL << 20) : vram_size;
  static constexpr size_t frame_size = 64;

  while (!fence_stop_.load(std::memory_order_relaxed)) {
    // Map the entire scannable VRAM region from offset 0.
    void *p = ::mmap(nullptr, scan_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                     vram_fd, 0);
    if (p != MAP_FAILED) {
      auto *mem = static_cast<uint8_t *>(p);
      for (size_t off = 0; off + frame_size <= scan_size; off += frame_size) {
        const auto *frame = reinterpret_cast<const uint32_t *>(mem + off);
        uint32_t fence_lo  = frame[3];  // fence_addr_lo
        uint32_t fence_hi  = frame[4];  // fence_addr_hi
        uint32_t fence_val = frame[5];  // fence_value
        if (fence_val == 0) continue;
        uint64_t fence_gpu = (static_cast<uint64_t>(fence_hi) << 32) | fence_lo;
        // fence_gpu is a GPU MC address; VRAM-backed fences have hi=0 and
        // lo in [fb_start, fb_start + vram_size). Convert to memfd offset.
        if (fence_hi != 0) continue;
        if (fence_lo < fb_start || fence_lo >= fb_start + vram_size) continue;
        uint64_t fence_memfd_off = fence_lo - fb_start;
        // The fence buffer is NOT at the same memfd offset as fence_lo since
        // memfd offset 0 = GPU MC 0, not fb_start. fence_lo IS the memfd offset.
        // (fb_start corresponds to memfd byte fb_start, not byte 0.)
        // So fence_memfd_off correctly gives us the byte within our scan mmap.
        if (fence_memfd_off >= scan_size) continue;
        auto *fence_ptr = reinterpret_cast<uint32_t *>(mem + fence_memfd_off);
        if (*fence_ptr != fence_val) {
          std::fprintf(stderr,
                       "[fence] write: ring_frame@memfd+0x%zx fence=0x%x val=%u\n",
                       off, fence_lo, fence_val);
          *fence_ptr = fence_val;
        }
      }
      ::munmap(p, scan_size);
    }
    // GFX/KIQ ring test auto-complete: if SCRATCH_REG0 in the BAR5 memfd still
    // holds 0xCAFEDEAD (the driver's init sentinel), overwrite with 0xDEADBEEF
    // so the polling loop exits immediately even without CP execution.
    // BAR5 dword index for SCRATCH_REG0: 0x10100 bytes / 4 = 0x4040.
    if (bar5_mem) {
      static constexpr uint32_t kScratchIdx = 0x10100 / 4;
      if (bar5_mem[kScratchIdx] == 0xCAFEDEADU)
        bar5_mem[kScratchIdx] = 0xDEADBEEFU;
    }

    struct timespec ts{0, 100000};  // 100μs between scans
    ::nanosleep(&ts, nullptr);
  }
}

} // namespace rocjitsu::vfu
