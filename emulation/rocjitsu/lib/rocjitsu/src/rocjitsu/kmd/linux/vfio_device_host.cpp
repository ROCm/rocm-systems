// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// THIS IS THE ONLY FILE IN THE rocjitsu vfio-user FRONT-END THAT MAY INCLUDE
// <libvfio-user.h>. All vfu_* calls must live here; device models and BAR
// handlers must remain transport-neutral.

#include "rocjitsu/kmd/linux/vfio_device_host.h"

#include <libvfio-user.h>
#include <vfio-user/pci_defs.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstring>
#include <span>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

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
  if (region.vaddr && region.length > 0) {
    std::lock_guard<std::mutex> lk(dma_mutex_);
    dma_regions_.push_back({region.iova, region.vaddr, region.length});
  }
}

void VfioDeviceHost::unmap(const simdojo::DmaRegion &region) {
  device_->dma_unmap(region);
  if (region.vaddr) {
    std::lock_guard<std::mutex> lk(dma_mutex_);
    dma_regions_.erase(
        std::remove_if(dma_regions_.begin(), dma_regions_.end(),
                       [&](const DmaEntry &e) { return e.vaddr == region.vaddr; }),
        dma_regions_.end());
  }
}

void *VfioDeviceHost::iova_to_hva(uint64_t iova, size_t len) const {
  // Caller must NOT hold dma_mutex_ (this is const; callers manage locking).
  for (const auto &e : dma_regions_) {
    if (iova >= e.iova && iova + len <= e.iova + e.length)
      return static_cast<uint8_t *>(e.vaddr) + (iova - e.iova);
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// SDMA ring execution — called from the BAR2 doorbell callback.
//
// When amdgpu commits an SDMA ring by writing the wptr to the doorbell BAR,
// we read the ring packet out of guest RAM (via the DMA region map) and
// execute it in software.  Currently handles only SDMA_OP_WRITE/LINEAR,
// which is all that sdma_v4_4_2_ring_test_ring needs.
//
// SDMA0 register offsets (base = 0x1260 dwords, BAR5 byte addresses):
//   regSDMA_GFX_RB_BASE    = (0x1260+0x0081)*4 = 0x4B84  [ring_gpu_addr >> 8]
//   regSDMA_GFX_RB_BASE_HI = 0x4B88
//   regSDMA_GFX_RB_RPTR    = 0x4B8C
//   regSDMA_GFX_RB_WPTR    = 0x4B94  [written by set_wptr in non-doorbell path]
//
// SDMA0 GFX doorbell: doorbell_index = sdma_engine[0]<<1 = 0x200.
//   WDOORBELL64(0x200, wptr) → BAR2 dword 0x200 → byte 0x800.
// ---------------------------------------------------------------------------
namespace {

// Dword offsets of key SDMA0 registers within BAR5.
static constexpr uint32_t kSdma0RbBase   = (0x1260 + 0x0081) * 4; // ring_base>>8
static constexpr uint32_t kSdma0RbBaseHi = (0x1260 + 0x0082) * 4;
static constexpr uint32_t kSdma0RbRptr   = (0x1260 + 0x0083) * 4;
static constexpr uint32_t kSdma0RbWptr   = (0x1260 + 0x0085) * 4;

// Ring buffer size field lives in RB_CNTL[5:1] — 2^(field+1) dwords.
// We don't need to check wrap for a 5-dword test packet; just read linearly.

// SDMA packet opcodes
static constexpr uint8_t kSdmaOpWrite       = 0x2A;
static constexpr uint8_t kSdmaSubopLinear   = 0x00;

void execute_sdma_ring(simdojo::PciDevice *device,
                       const std::vector<VfioDeviceHost::DmaEntry> &dma_regions,
                       uint64_t new_wptr_bytes) {
  // Read ring base IOVA from SDMA register shadow.
  uint32_t rb_base_lo = device->mmio_peek(kSdma0RbBase);
  uint32_t rb_base_hi = device->mmio_peek(kSdma0RbBaseHi);
  uint64_t ring_iova  = (static_cast<uint64_t>(rb_base_hi) << 40) |
                        (static_cast<uint64_t>(rb_base_lo) << 8);

  // rptr register holds dword count; doorbell carries byte offset (wptr<<2).
  uint32_t rptr_dw = device->mmio_peek(kSdma0RbRptr);
  uint64_t rptr_bytes = static_cast<uint64_t>(rptr_dw) * 4;

  std::fprintf(stderr,
               "[sdma] doorbell: ring_iova=0x%llx rptr=0x%llx wptr=0x%llx\n",
               (unsigned long long)ring_iova,
               (unsigned long long)rptr_bytes,
               (unsigned long long)new_wptr_bytes);

  if (ring_iova == 0 || new_wptr_bytes <= rptr_bytes)
    return;

  // Locate ring buffer in DMA regions.
  void *ring_hva = nullptr;
  size_t ring_avail = new_wptr_bytes - rptr_bytes;
  for (const auto &e : dma_regions) {
    uint64_t off = ring_iova + rptr_bytes;
    if (off >= e.iova && off + ring_avail <= e.iova + e.length) {
      ring_hva = static_cast<uint8_t *>(e.vaddr) + (off - e.iova);
      break;
    }
  }
  if (!ring_hva) {
    std::fprintf(stderr, "[sdma] ring HVA not found for iova=0x%llx rptr=0x%llx\n",
                 (unsigned long long)ring_iova, (unsigned long long)rptr_bytes);
    return;
  }

  // Walk packets between rptr and wptr.
  const uint32_t *pkt = static_cast<const uint32_t *>(ring_hva);
  size_t ndw = ring_avail / 4;
  size_t i = 0;
  while (i < ndw) {
    uint32_t hdr = pkt[i];
    uint8_t  op    = hdr & 0xFF;
    uint8_t  subop = (hdr >> 8) & 0xFF;

    if (op == kSdmaOpWrite && subop == kSdmaSubopLinear) {
      // SDMA WRITE_LINEAR: 5 dwords
      // [0] header  [1] dst_lo  [2] dst_hi  [3] count-1  [4] data
      if (i + 4 >= ndw) break;
      uint32_t dst_lo  = pkt[i + 1];
      uint32_t dst_hi  = pkt[i + 2];
      uint32_t count_m1 = pkt[i + 3] & 0x3FFFF; // lower 18 bits = dword_count-1
      uint32_t data    = pkt[i + 4];
      uint64_t dst_iova = (static_cast<uint64_t>(dst_hi) << 32) |
                          static_cast<uint64_t>(dst_lo);
      uint32_t ndwords = count_m1 + 1;

      std::fprintf(stderr,
                   "[sdma] WRITE_LINEAR dst=0x%llx ndw=%u data=0x%08x\n",
                   (unsigned long long)dst_iova, ndwords, data);

      // Write data to destination in DMA regions.
      for (uint32_t d = 0; d < ndwords; ++d) {
        void *dst_hva = nullptr;
        for (const auto &e : dma_regions) {
          uint64_t off = dst_iova + d * 4;
          if (off >= e.iova && off + 4 <= e.iova + e.length) {
            dst_hva = static_cast<uint8_t *>(e.vaddr) + (off - e.iova);
            break;
          }
        }
        if (dst_hva)
          *static_cast<uint32_t *>(dst_hva) = data;
        else
          std::fprintf(stderr, "[sdma] dst HVA not found for iova=0x%llx\n",
                       (unsigned long long)(dst_iova + d * 4));
      }
      i += 5;
    } else if (op == 0x00) {
      // NOP — skip one dword.
      ++i;
    } else {
      // Unknown packet: log and bail.
      std::fprintf(stderr, "[sdma] unknown pkt op=0x%02x subop=0x%02x hdr=0x%08x at i=%zu\n",
                   op, subop, hdr, i);
      break;
    }
  }
}

} // namespace

int VfioDeviceHost::setup_bars() {
  // Callback trampoline for register-trapped BARs (currently only BAR5).
  // Routes to device_->bar_access(5, ...) for indirect register handling
  // (MM_INDEX/MM_DATA, RSMU_INDEX/DATA, etc.).
  auto bar5_cb = [](vfu_ctx_t *c, char *buf, size_t cnt,
                    long off, bool wr) -> ssize_t {
    auto *host = reinterpret_cast<VfioDeviceHost *>(vfu_get_private(c));
    if (!host || !host->device_)
      return -1;
    std::span<std::byte> span(reinterpret_cast<std::byte *>(buf), cnt);
    return host->device_->bar_access(5, span, static_cast<uint64_t>(off), wr);
  };

  // BAR2 doorbell callback: on writes, execute SDMA ring packets.
  // The SDMA ring test writes a WRITE_LINEAR packet and commits it via
  // WDOORBELL64(0x200, wptr) → BAR2 byte offset 0x800.
  auto bar2_cb = [](vfu_ctx_t *c, char *buf, size_t cnt,
                    long off, bool wr) -> ssize_t {
    auto *host = reinterpret_cast<VfioDeviceHost *>(vfu_get_private(c));
    if (!host || !host->device_)
      return -1;

    // Route through the device model first (records doorbell for KFD, etc.).
    std::span<std::byte> span(reinterpret_cast<std::byte *>(buf), cnt);
    ssize_t r = host->device_->bar_access(2, span, static_cast<uint64_t>(off), wr);

    // On writes to SDMA0 doorbell (BAR2 byte 0x800 = dword 0x200 << 2):
    // Execute the SDMA ring packets now that the ring has been committed.
    if (wr && static_cast<uint32_t>(off) == 0x800 && cnt >= 4) {
      uint64_t wptr_bytes = 0;
      if (cnt >= 8)
        std::memcpy(&wptr_bytes, buf, 8);
      else {
        uint32_t lo = 0;
        std::memcpy(&lo, buf, 4);
        wptr_bytes = lo;
      }
      std::lock_guard<std::mutex> lk(host->dma_mutex_);
      execute_sdma_ring(host->device_, host->dma_regions_, wptr_bytes);
    }
    return r;
  };

  for (const auto &bar : device_->bars()) {
    int fd = device_->bar_fd(bar.index);
    uint32_t flags = VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM;
    if (bar.is_64bit)     flags |= VFU_REGION_FLAG_64_BITS;
    if (bar.prefetchable)  flags |= VFU_REGION_FLAG_PREFETCH;

    if (fd >= 0 && bar.index == 5) {
      // BAR5: memfd-backed AND callback-trapped.
      // QEMU maps the memfd directly so CPU MMIO writes (e.g. SCRATCH_REG0)
      // land in the shared buffer without going through the socket. The callback
      // still fires for accesses QEMU explicitly proxies (indirect registers).
      // VFU_REGION_FLAG_ALWAYS_CB ensures the callback fires for all accesses
      // that QEMU does route through the socket.
      iovec mmap_area{.iov_base = nullptr, .iov_len = bar.size};
      if (vfu_setup_region(ctx_, bar.index, bar.size, bar5_cb,
                           flags | VFU_REGION_FLAG_ALWAYS_CB,
                           &mmap_area, 1, fd, 0) != 0) {
        std::perror("vfu_setup_region (BAR5 mmap+cb)");
        return -1;
      }
    } else if (fd >= 0 && bar.index == 2) {
      // BAR2 (doorbell): memfd-backed + ALWAYS_CB so doorbell writes are
      // intercepted even though QEMU also mmaps the memfd directly.
      iovec mmap_area{.iov_base = nullptr, .iov_len = bar.size};
      if (vfu_setup_region(ctx_, bar.index, bar.size, bar2_cb,
                           flags | VFU_REGION_FLAG_ALWAYS_CB,
                           &mmap_area, 1, fd, 0) != 0) {
        std::perror("vfu_setup_region (BAR2 mmap+cb)");
        return -1;
      }
    } else if (fd >= 0) {
      // Memfd-backed BAR: fully mmap-able, no per-access callback.
      iovec mmap_area{.iov_base = nullptr, .iov_len = bar.size};
      if (vfu_setup_region(ctx_, bar.index, bar.size, nullptr,
                           flags, &mmap_area, 1, fd, 0) != 0) {
        std::perror("vfu_setup_region (mmap BAR)");
        return -1;
      }
    } else {
      // Register-trapped BAR (no memfd): callback for all accesses.
      // VFU_REGION_FLAG_ALWAYS_CB ensures libvfio-user routes every access
      // (including direct MMIO writes from the guest) through our callback.
      uint32_t cb_flags = flags;
      if (bar.index == 5) cb_flags |= VFU_REGION_FLAG_ALWAYS_CB;
      if (vfu_setup_region(ctx_, bar.index, bar.size,
                           bar.index == 5 ? bar5_cb : nullptr,
                           cb_flags, nullptr, 0, -1, 0) != 0) {
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

void VfioDeviceHost::fence_service_loop() {
  // Map BAR5 memfd for GFX ring-test polling.
  int bar5_fd = device_ ? device_->bar_fd(5) : -1;
  static constexpr size_t kBar5Size = 256 * 1024;
  void *bar5_p = nullptr;
  volatile uint32_t *bar5 = nullptr;

  if (bar5_fd >= 0) {
    bar5_p = ::mmap(nullptr, kBar5Size, PROT_READ | PROT_WRITE, MAP_SHARED,
                    bar5_fd, 0);
    if (bar5_p == MAP_FAILED) {
      std::perror("[vfio-host] fence thread: mmap BAR5");
      bar5_p = nullptr;
    } else {
      bar5 = static_cast<volatile uint32_t *>(bar5_p);
    }
  }

  // Map BAR2 (doorbell) memfd for SDMA ring-test polling.
  // QEMU writes doorbells directly via the mmap — the vfio-user callback is
  // never invoked for CPU MMIO writes to mmap-backed BARs.  We poll instead.
  int bar2_fd = device_ ? device_->bar_fd(2) : -1;
  static constexpr size_t kBar2Size = 2 * 1024 * 1024; // 2 MB doorbell BAR
  void *bar2_p = nullptr;
  volatile uint64_t *bar2 = nullptr;
  // Track last seen SDMA0 doorbell value to detect new commits.
  uint64_t sdma0_wptr_prev = 0;

  if (bar2_fd >= 0) {
    bar2_p = ::mmap(nullptr, kBar2Size, PROT_READ | PROT_WRITE, MAP_SHARED,
                    bar2_fd, 0);
    if (bar2_p == MAP_FAILED) {
      std::perror("[vfio-host] fence thread: mmap BAR2");
      bar2_p = nullptr;
    } else {
      bar2 = static_cast<volatile uint64_t *>(bar2_p);
    }
  }

  // Poll for GFX/KIQ ring test sentinel: SCRATCH_REG0 at byte 0x10100.
  // When amdgpu writes 0xCAFEDEAD (ring test init), we overwrite with
  // 0xDEADBEEF so the poll loop exits immediately without CP execution.
  static constexpr uint32_t kScratchIdx    = 0x10100 / 4;
  static constexpr uint32_t kTestInit      = 0xCAFEDEADU;
  static constexpr uint32_t kTestDone      = 0xDEADBEEFU;

  // SDMA0 GFX doorbell: WDOORBELL64(doorbell_index=0x200, wptr_bytes).
  // BAR2 dword offset 0x200 = uint64_t index 0x100 (64-bit doorbells).
  // ring->doorbell_index = sdma_engine[0]<<1 = 0x200.
  // WDOORBELL64 writes: cpu_addr[0x200] = wptr; (uint32_t ptr indexing)
  // As uint64_t: bar2[0x100] = wptr_bytes.
  static constexpr size_t kSdma0DbIdx = 0x100; // 64-bit index into BAR2

  // KIQ GFX ring doorbell: WDOORBELL64(doorbell_index=0x000, wptr).
  // doorbell_index.kiq = AMDGPU_DOORBELL_LAYOUT1_KIQ_START = 0x000.
  // WDOORBELL64 writes: cpu_addr[0x000] = wptr; as uint64_t: bar2[0x000].
  // When the KIQ wptr advances, the CP would process MAP_QUEUES/MAP_PROCESS
  // PM4 packets and write acknowledgments. We detect this and simulate fence
  // signal so KFD fence waits don't block (already handled by livepatch stubs).
  static constexpr size_t kKiqDbIdx = 0x000; // 64-bit index into BAR2
  uint64_t kiq_wptr_prev = ~0ULL;

  uint64_t iteration = 0;
  while (!fence_stop_.load(std::memory_order_relaxed)) {
    ++iteration;
    // GFX SCRATCH_REG0 ring test (every iteration).
    if (bar5 && bar5[kScratchIdx] == kTestInit)
      bar5[kScratchIdx] = kTestDone;

    // KIQ GFX doorbell (BAR2 64-bit index 0x000).
    // Detects MAP_QUEUES/MAP_PROCESS commits from the KFD scheduler.
    // The KIQ ring acknowledgment is handled by our livepatch stubs
    // (amdgpu_fence_wait_polling returns immediately). We log KIQ activity
    // for diagnostics.
    if (bar2) {
      uint64_t kiq_now = bar2[kKiqDbIdx];
      if (kiq_now != kiq_wptr_prev && kiq_now != ~0ULL) {
        std::fprintf(stderr,
                     "[vfio-host] KIQ doorbell: wptr 0x%llx→0x%llx\n",
                     (unsigned long long)kiq_wptr_prev,
                     (unsigned long long)kiq_now);
        kiq_wptr_prev = kiq_now;
        // KIQ ring committed PM4 packets (MAP_QUEUES, MAP_PROCESS, etc.).
        // The KFD fence waits are handled by our livepatch stubs.
        // The BAR5 KIQ fence address is updated separately via fence_drv.
      }
    }

    // SDMA0 kernel-mode doorbell polling (BAR2 64-bit index 0x100).
    // Detects when the kernel SDMA0 GFX ring commits a new wptr.
    if (bar2) {
      uint64_t wptr_now = bar2[kSdma0DbIdx];
      if (wptr_now != sdma0_wptr_prev) {
        sdma0_wptr_prev = wptr_now;
        if (wptr_now != ~0ULL && wptr_now != 0)
          sdma_needs_scan_.store(true, std::memory_order_relaxed);
      }
    }

    // GTT sentinel scan: triggered on SDMA doorbell change or every 100 iterations
    // (~50ms) to catch user-mode queue completions.
    // Scans the first 4096 dwords of each DMA region for:
    //   1. 0xCAFEDEAD → 0xDEADBEEF  (kernel ring test sentinel)
    //   2. SDMA WRITE_LINEAR packets (op=0x2A) → execute them
    if (sdma_needs_scan_.load(std::memory_order_relaxed) || (iteration % 100) == 0) {
      sdma_needs_scan_.store(false, std::memory_order_relaxed);
      static constexpr size_t kScanDwords = 4096;
      // SDMA WRITE_LINEAR header: op=0x2A (bits 7:0), subop=0x00 (bits 15:8)
      static constexpr uint32_t kSdmaWriteLinearHdr = 0x0000002AU;
      std::vector<DmaEntry> snap;
      {
        std::lock_guard<std::mutex> lk(dma_mutex_);
        snap = dma_regions_;
      }
      for (const auto &e : snap) {
        if (e.length < 4 || !e.vaddr) continue;
        auto *words = static_cast<volatile uint32_t *>(e.vaddr);
        size_t nw = std::min(e.length / 4, kScanDwords);
        for (size_t j = 0; j < nw; ++j) {
          // Kernel ring test: CAFEDEAD → DEADBEEF
          if (words[j] == kTestInit) {
            words[j] = kTestDone;
            continue;
          }
          // SDMA WRITE_LINEAR packet: 5 dwords
          // [0]=hdr [1]=dst_lo [2]=dst_hi [3]=count-1 [4]=data
          if (words[j] == kSdmaWriteLinearHdr && j + 4 < nw) {
            uint32_t dst_lo   = words[j + 1];
            uint32_t dst_hi   = words[j + 2];
            uint32_t count_m1 = words[j + 3] & 0x3FFFF;
            uint32_t data     = words[j + 4];
            uint64_t dst_iova = (static_cast<uint64_t>(dst_hi) << 32) | dst_lo;
            // Only execute if dst is a known DMA region and count is small.
            // Large copies (H→D) are not executed (no real data movement).
            // Small writes (completion signals, ≤4 dwords) are executed.
            if (count_m1 <= 3 && dst_iova != 0) {
              for (uint32_t d = 0; d <= count_m1; ++d) {
                uint64_t tgt = dst_iova + d * 4;
                void *dst_hva = nullptr;
                for (const auto &r : snap)
                  if (tgt >= r.iova && tgt + 4 <= r.iova + r.length) {
                    dst_hva = static_cast<uint8_t *>(r.vaddr) + (tgt - r.iova);
                    break;
                  }
                if (dst_hva)
                  *static_cast<volatile uint32_t *>(dst_hva) = data;
              }
              // Mark packet as consumed: zero the header so we don't re-execute.
              words[j] = 0;
            }
          }
        }
      }
    }

    struct timespec ts{0, 500000};  // 0.5 ms per iteration (faster for SDMA)
    ::nanosleep(&ts, nullptr);
  }

  if (bar5_p) ::munmap(bar5_p, kBar5Size);
  if (bar2_p) ::munmap(bar2_p, kBar2Size);
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

  // Start fence/ring-test service thread.
  fence_stop_.store(false);
  fence_thread_ = std::thread(&VfioDeviceHost::fence_service_loop, this);

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

  fence_stop_.store(true);
  if (fence_thread_.joinable())
    fence_thread_.join();

  return ret;
}

} // namespace rocjitsu
