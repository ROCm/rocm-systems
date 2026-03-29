/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*
 * Queue management for wddm_lite backend.
 * Creates and manages AQL compute queues through the WDDM driver's
 * escape interface, which handles MES-based queue activation on GFX12.
 */

#include "wddm_lite_internal.h"
#include "wddm_lite_device.h"
#include "gpu_init.h"
#include <string.h>
#include <stdlib.h>

extern struct WddmLiteDevice g_wddm_lite_dev;

/* Defined in memory.cpp — look up driver handle for a CPU address */
extern "C" ULONGLONG wddm_lite_lookup_alloc_handle(void *cpu_addr);

/* Queue resource tracking */
#define MAX_QUEUES 16
static ULONG s_next_queue_id = 1;

struct DmaBuffer {
    void       *cpu_addr;
    ULONGLONG   bus_addr;
    ULONGLONG   gpu_addr;   /* GART-mapped GPU address */
    PVOID       handle;     /* DMA alloc handle for freeing */
    ULONG       size;
};

struct QueueResources {
    BOOLEAN  in_use;
    ULONG    queue_id;
    ULONG    hqd_queue_idx;   /* HQD index for gpu_setup_compute_queue */
    /* DMA-allocated buffers with bus addresses for GART mapping */
    struct DmaBuffer rptr;
    struct DmaBuffer wptr;
    struct DmaBuffer eop;
    /* Ring buffer GART mapping (ring itself owned by ROCR) */
    ULONGLONG ring_gpu_addr;
    ULONG     ring_num_pages;
};

static struct QueueResources s_queues[MAX_QUEUES];
static ULONG s_next_hqd_idx = 0;
static BOOLEAN s_hqd_in_use[GPU_MAX_COMPUTE_QUEUES] = {};

/* Find a free HQD index, or return GPU_MAX_COMPUTE_QUEUES if none */
static ULONG alloc_hqd_index(void) {
    /* First try reusing a freed slot */
    for (ULONG i = 0; i < s_next_hqd_idx && i < GPU_MAX_COMPUTE_QUEUES; i++) {
        if (!s_hqd_in_use[i]) {
            s_hqd_in_use[i] = TRUE;
            return i;
        }
    }
    /* Then try the next unused slot */
    if (s_next_hqd_idx < GPU_MAX_COMPUTE_QUEUES) {
        ULONG idx = s_next_hqd_idx++;
        s_hqd_in_use[idx] = TRUE;
        return idx;
    }
    return GPU_MAX_COMPUTE_QUEUES;  /* No slots available */
}

static void free_hqd_index(ULONG idx) {
    if (idx < GPU_MAX_COMPUTE_QUEUES)
        s_hqd_in_use[idx] = FALSE;
}

/*
 * Allocate a DMA buffer (physically contiguous, bus address known).
 * Used for queue control structures that the GPU must access.
 */
static int alloc_dma_buffer(struct WddmLiteDevice *dev, struct DmaBuffer *buf,
                            ULONG size)
{
    memset(buf, 0, sizeof(*buf));

    AMDGPU_ESCAPE_ALLOC_DMA_DATA dma;
    memset(&dma, 0, sizeof(dma));
    dma.Header.Command = AMDGPU_ESCAPE_ALLOC_DMA;
    dma.Header.Size = sizeof(dma);
    dma.Size = size;

    if (wddm_lite_escape(dev, &dma, sizeof(dma)) != 0 || !dma.CpuAddress)
        return -1;

    buf->cpu_addr = dma.CpuAddress;
    buf->bus_addr = dma.BusAddress;
    buf->handle = dma.AllocationHandle;
    buf->size = size;

    memset(buf->cpu_addr, 0, size);

    /* Map into GART for GPU access */
    buf->gpu_addr = gpu_gart_map_contig(dev, buf->bus_addr, size);
    if (buf->gpu_addr == 0) {
        pr_err("alloc_dma_buffer: GART mapping failed for bus 0x%llx\n",
               (unsigned long long)buf->bus_addr);
        return -1;
    }

    pr_info("alloc_dma_buffer: cpu=%p bus=0x%llx gpu=0x%llx size=%u\n",
            buf->cpu_addr, (unsigned long long)buf->bus_addr,
            (unsigned long long)buf->gpu_addr, size);

    return 0;
}

static void free_dma_buffer(struct WddmLiteDevice *dev, struct DmaBuffer *buf)
{
    if (buf->gpu_addr)
        gpu_gart_unmap(dev, buf->gpu_addr, (buf->size + 4095) / 4096);
    if (buf->handle) {
        AMDGPU_ESCAPE_ALLOC_DMA_DATA free_dma;
        memset(&free_dma, 0, sizeof(free_dma));
        free_dma.Header.Command = AMDGPU_ESCAPE_FREE_DMA;
        free_dma.Header.Size = sizeof(free_dma);
        free_dma.AllocationHandle = buf->handle;
        wddm_lite_escape(dev, &free_dma, sizeof(free_dma));
    }
    memset(buf, 0, sizeof(*buf));
}

/*
 * Get physical pages for an allocation and map them through GART.
 * Returns GPU address, or 0 on failure.
 * The allocation must have been made via hsaKmtAllocMemory
 * (tracked by memory.cpp).
 */
static ULONGLONG gart_map_allocation(struct WddmLiteDevice *dev,
                                     void *cpu_addr, ULONGLONG size,
                                     ULONG *out_num_pages)
{
    ULONGLONG handle = wddm_lite_lookup_alloc_handle(cpu_addr);
    if (handle == (ULONGLONG)-1) {
        pr_err("gart_map_allocation: no handle for %p\n", cpu_addr);
        return 0;
    }

    ULONG total_pages = (ULONG)((size + 4095) / 4096);
    ULONGLONG *bus_addrs = NULL;
    ULONGLONG gpu_addr = 0;

    /* Allocate bus address array */
    bus_addrs = (ULONGLONG *)malloc(total_pages * sizeof(ULONGLONG));
    if (!bus_addrs) {
        pr_err("gart_map_allocation: malloc failed for %u pages\n", total_pages);
        return 0;
    }

    /* Fetch physical pages in batches */
    ULONG fetched = 0;
    while (fetched < total_pages) {
        AMDGPU_ESCAPE_GET_PHYS_PAGES_DATA phys;
        memset(&phys, 0, sizeof(phys));
        phys.Header.Command = AMDGPU_ESCAPE_GET_PHYS_PAGES;
        phys.Header.Size = sizeof(phys);
        phys.Handle = handle;
        phys.PageOffset = fetched;

        if (wddm_lite_escape(dev, &phys, sizeof(phys)) != 0 ||
            phys.Header.Status != 0 || phys.NumPages == 0) {
            pr_err("gart_map_allocation: GET_PHYS_PAGES failed at page %u\n",
                   fetched);
            free(bus_addrs);
            return 0;
        }

        for (ULONG i = 0; i < phys.NumPages && fetched < total_pages; i++)
            bus_addrs[fetched++] = phys.PhysAddrs[i];
    }

    /* Map through GART */
    gpu_addr = gpu_gart_map(dev, bus_addrs, total_pages);
    free(bus_addrs);

    if (gpu_addr == 0) {
        pr_err("gart_map_allocation: GART map failed for %u pages\n",
               total_pages);
        return 0;
    }

    if (out_num_pages)
        *out_num_pages = total_pages;

    pr_info("gart_map_allocation: %p (handle %llu) -> GPU 0x%llx (%u pages)\n",
            cpu_addr, (unsigned long long)handle,
            (unsigned long long)gpu_addr, total_pages);

    return gpu_addr;
}

/*
 * Allocate a small system memory buffer via escape (legacy, for non-GPU use).
 * Returns CPU address, or NULL on failure.
 */
static void *alloc_queue_buffer(struct WddmLiteDevice *dev, ULONG size)
{
    AMDGPU_ESCAPE_ALLOC_MEMORY_DATA alloc;
    memset(&alloc, 0, sizeof(alloc));
    alloc.Header.Command = AMDGPU_ESCAPE_ALLOC_MEMORY;
    alloc.Header.Size = sizeof(alloc);
    alloc.SizeInBytes = size;
    alloc.Flags = 0x0024;  /* SYSTEM | HOST_ACCESS */

    if (wddm_lite_escape(dev, &alloc, sizeof(alloc)) != 0)
        return NULL;
    if (alloc.Header.Status != 0 || !alloc.CpuAddress)
        return NULL;

    memset(alloc.CpuAddress, 0, size);
    return alloc.CpuAddress;
}

/*
 * Map the doorbell BAR into userspace (lazy, called once).
 * We map a 4KB page — enough for many queues (each uses 8 bytes).
 */
static int ensure_doorbell_mapped(struct WddmLiteDevice *dev)
{
    if (dev->doorbell_base)
        return 0;  /* Already mapped */

    /*
     * Find the doorbell BAR: it's the memory BAR that's neither the
     * largest (VRAM) nor the smallest (MMIO registers).
     */
    pr_info("ensure_doorbell_mapped: NumBars=%u VramBar=%u MmioBar=%u\n",
            dev->info.NumBars, dev->info.VramBarIndex, dev->info.MmioBarIndex);
    for (ULONG bi = 0; bi < dev->info.NumBars && bi < 8; bi++) {
        pr_info("  BAR%u: phys=0x%llx len=0x%llx isMem=%d\n",
                bi,
                (unsigned long long)dev->info.Bars[bi].PhysicalAddress.QuadPart,
                (unsigned long long)dev->info.Bars[bi].Length,
                dev->info.Bars[bi].IsMemory);
    }
    ULONG db_bar = 0;
    ULONGLONG db_size = 4096;  /* Map first page of doorbell BAR */
    {
        ULONG vram_bar = dev->info.VramBarIndex;
        ULONG mmio_bar = dev->info.MmioBarIndex;
        bool found = false;
        for (ULONG i = 0; i < dev->info.NumBars; i++) {
            if (!dev->info.Bars[i].IsMemory || dev->info.Bars[i].Length == 0)
                continue;
            if (i != vram_bar && i != mmio_bar) {
                db_bar = i;
                found = true;
                break;
            }
        }
        if (!found) {
            pr_err("ensure_doorbell_mapped: no doorbell BAR found\n");
            return -1;
        }
    }

    pr_info("ensure_doorbell_mapped: mapping BAR%u (phys=0x%llx, len=0x%llx)\n",
            db_bar,
            (unsigned long long)dev->info.Bars[db_bar].PhysicalAddress.QuadPart,
            (unsigned long long)dev->info.Bars[db_bar].Length);

    /*
     * Map the real doorbell BAR via MAP_BAR escape.
     * The kernel driver does MmMapIoSpace → MDL → MmMapLockedPages
     * to give us a user-mode pointer directly into the doorbell aperture.
     * When the GPU hardware detects a write to a doorbell offset,
     * it triggers the corresponding compute queue to fetch new commands.
     */
    AMDGPU_ESCAPE_MAP_BAR_DATA map;
    memset(&map, 0, sizeof(map));
    map.Header.Command = AMDGPU_ESCAPE_MAP_BAR;
    map.Header.Size = sizeof(map);
    map.BarIndex = db_bar;
    map.Offset = 0;
    map.Length = db_size;

    if (wddm_lite_escape(dev, &map, sizeof(map)) != 0 ||
        map.Header.Status != 0 || !map.MappedAddress) {
        pr_err("ensure_doorbell_mapped: MAP_BAR failed, status=0x%lx\n",
               (unsigned long)map.Header.Status);
        return -1;
    }

    dev->doorbell_base = map.MappedAddress;
    dev->doorbell_mapping_handle = map.MappingHandle;
    dev->doorbell_size = db_size;

    pr_info("ensure_doorbell_mapped: BAR%u mapped at %p (size 0x%llx)\n",
            db_bar, dev->doorbell_base, (unsigned long long)db_size);

    return 0;
}

/*
 * Map HSA_QUEUE_TYPE to AMDGPU_QUEUE_TYPE_*.
 */
static ULONG hsa_queue_type_to_escape(HSA_QUEUE_TYPE type)
{
    switch (type) {
    case HSA_QUEUE_COMPUTE:
    case HSA_QUEUE_COMPUTE_OS:
        return AMDGPU_QUEUE_TYPE_COMPUTE;
    case HSA_QUEUE_COMPUTE_AQL:
        return AMDGPU_QUEUE_TYPE_COMPUTE_AQL;
    case HSA_QUEUE_SDMA:
    case HSA_QUEUE_SDMA_OS:
    case HSA_QUEUE_SDMA_XGMI:
    case HSA_QUEUE_DMA_AQL:
    case HSA_QUEUE_DMA_AQL_XGMI:
        return AMDGPU_QUEUE_TYPE_SDMA;
    default:
        return AMDGPU_QUEUE_TYPE_COMPUTE_AQL;
    }
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtCreateQueue(HSAuint32 NodeId, HSA_QUEUE_TYPE Type,
                   HSAuint32 QueuePercentage,
                   HSA_QUEUE_PRIORITY Priority,
                   void *QueueAddress, HSAuint64 QueueSizeInBytes,
                   HsaEvent *Event, HsaQueueResource *QueueResource)
{
    CHECK_KFD_OPEN();

    if (!QueueAddress || !QueueResource || QueueSizeInBytes == 0)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    pr_err("hsaKmtCreateQueue: node=%u type=%d pct=%u ring=%p size=0x%llx\n",
           NodeId, (int)Type, QueuePercentage,
           QueueAddress, (unsigned long long)QueueSizeInBytes);
    fflush(stderr);

    /* Find a free queue slot */
    int slot = -1;
    for (int i = 0; i < MAX_QUEUES; i++) {
        if (!s_queues[i].in_use) { slot = i; break; }
    }
    if (slot < 0) {
        pr_err("hsaKmtCreateQueue: no free queue slots\n");
        return HSAKMT_STATUS_OUT_OF_RESOURCES;
    }

    /* Map doorbell BAR if not already done */
    if (ensure_doorbell_mapped(&g_wddm_lite_dev) != 0) {
        pr_err("hsaKmtCreateQueue: doorbell mapping failed\n");
        return HSAKMT_STATUS_ERROR;
    }

    /* Determine if this is an AQL queue */
    BOOLEAN aql = (Type == HSA_QUEUE_COMPUTE_AQL || Type == HSA_QUEUE_DMA_AQL ||
                   Type == HSA_QUEUE_DMA_AQL_XGMI);

    memset(&s_queues[slot], 0, sizeof(s_queues[slot]));

    /*
     * Allocate RPTR, WPTR, and EOP buffers via DMA (contiguous, bus addr known).
     * Map each through GART so the GPU can access them.
     */
    if (alloc_dma_buffer(&g_wddm_lite_dev, &s_queues[slot].rptr, 4096) != 0 ||
        alloc_dma_buffer(&g_wddm_lite_dev, &s_queues[slot].wptr, 4096) != 0 ||
        alloc_dma_buffer(&g_wddm_lite_dev, &s_queues[slot].eop, 4096) != 0) {
        pr_err("hsaKmtCreateQueue: DMA buffer allocation failed\n");
        free_dma_buffer(&g_wddm_lite_dev, &s_queues[slot].rptr);
        free_dma_buffer(&g_wddm_lite_dev, &s_queues[slot].wptr);
        free_dma_buffer(&g_wddm_lite_dev, &s_queues[slot].eop);
        return HSAKMT_STATUS_NO_MEMORY;
    }

    /*
     * Map the ring buffer (allocated by ROCR via hsaKmtAllocMemory) through GART.
     * GET_PHYS_PAGES returns per-page bus addresses even for non-contiguous
     * allocations, and GART makes them appear contiguous to the GPU.
     */
    ULONG ring_num_pages = 0;
    ULONGLONG ring_gpu_addr = gart_map_allocation(&g_wddm_lite_dev,
                                                   QueueAddress,
                                                   QueueSizeInBytes,
                                                   &ring_num_pages);
    if (ring_gpu_addr == 0) {
        pr_warn("hsaKmtCreateQueue: ring GART mapping failed, "
                "falling back to CPU address (dispatch will not work)\n");
        ring_gpu_addr = (ULONGLONG)(uintptr_t)QueueAddress;
    }

    s_queues[slot].ring_gpu_addr = ring_gpu_addr;
    s_queues[slot].ring_num_pages = ring_num_pages;

    /* Try direct HQD programming if GFX engine is initialized */
    ULONG hqd_idx = GPU_MAX_COMPUTE_QUEUES;
    if (g_wddm_lite_dev.hw.gfx_initialized) {
        hqd_idx = alloc_hqd_index();
    }
    if (hqd_idx < GPU_MAX_COMPUTE_QUEUES) {
        pr_info("hsaKmtCreateQueue: programming HQD %u with GART addresses\n",
                hqd_idx);
        pr_info("  ring_gpu=0x%llx rptr_gpu=0x%llx wptr_gpu=0x%llx eop_gpu=0x%llx\n",
                (unsigned long long)ring_gpu_addr,
                (unsigned long long)s_queues[slot].rptr.gpu_addr,
                (unsigned long long)s_queues[slot].wptr.gpu_addr,
                (unsigned long long)s_queues[slot].eop.gpu_addr);

        int ret = gpu_setup_compute_queue(&g_wddm_lite_dev, hqd_idx,
            ring_gpu_addr, (ULONG)QueueSizeInBytes,
            s_queues[slot].rptr.gpu_addr,
            s_queues[slot].wptr.gpu_addr,
            s_queues[slot].eop.gpu_addr,
            s_queues[slot].eop.size, aql);

        if (ret != 0) {
            pr_warn("hsaKmtCreateQueue: HQD programming failed, "
                    "queue will be non-functional\n");
            free_hqd_index(hqd_idx);
            hqd_idx = GPU_MAX_COMPUTE_QUEUES;
        } else {
            pr_info("hsaKmtCreateQueue: HQD %u activated, doorbell_index=%u (BAR offset=0x%x)\n",
                    hqd_idx, g_wddm_lite_dev.hw.queues[hqd_idx].doorbell_index,
                    g_wddm_lite_dev.hw.queues[hqd_idx].doorbell_index * 8);

            /* === NOP AQL packet test (VRAM-based, matching tinygrad) ===
             * Use VRAM ring at offset 7MB (within system aperture, VMID 0).
             * This bypasses GART entirely. If this works, the issue was
             * GART/VMID1 access. */
            if (hqd_idx == 0) {
                /* Use VRAM at offset 7MB for NOP test ring + control */
                ULONGLONG nop_vram_off = 7 * 1024 * 1024;
                ULONGLONG nop_ring_mc = g_wddm_lite_dev.hw.gmc.vram_start + nop_vram_off;
                ULONGLONG nop_rptr_mc = nop_ring_mc + 0x1000;
                ULONGLONG nop_wptr_mc = nop_ring_mc + 0x2000;
                ULONGLONG nop_eop_mc  = nop_ring_mc + 0x3000;
                ULONGLONG nop_mqd_mc  = nop_ring_mc + 0x4000;

                /* Map VRAM for NOP test */
                AMDGPU_ESCAPE_MAP_BAR_DATA nop_map;
                memset(&nop_map, 0, sizeof(nop_map));
                nop_map.Header.Command = AMDGPU_ESCAPE_MAP_BAR;
                nop_map.Header.Size = sizeof(nop_map);
                nop_map.BarIndex = 0; /* BAR0 = VRAM */
                nop_map.Offset = nop_vram_off;
                nop_map.Length = 0x5000;
                if (wddm_lite_escape(&g_wddm_lite_dev, &nop_map, sizeof(nop_map)) != 0 ||
                    !nop_map.MappedAddress) {
                    pr_warn("hsaKmtCreateQueue: NOP VRAM mapping failed\n");
                } else {
                UCHAR *nop_cpu = (UCHAR *)nop_map.MappedAddress;
                memset(nop_cpu, 0, 0x5000);

                /* Write NOP AQL barrier packet at ring offset 0 */
                volatile USHORT *nop_hdr = (volatile USHORT *)(nop_cpu);
                nop_hdr[0] = (4) | (1 << 8); /* type=BARRIER_AND, barrier=1 */

                /* Build MQD for VMID 0 VRAM-based ring */
                volatile ULONG *nop_mqd = (volatile ULONG *)(nop_cpu + 0x4000);
                /* MQD header (tinygrad: 0xC0310800) — MEC reads from memory */
                nop_mqd[0] = 0xC0310800;
                nop_mqd[128] = (ULONG)(nop_mqd_mc & 0xFFFFFFFC);
                nop_mqd[129] = (ULONG)(nop_mqd_mc >> 32);
                nop_mqd[130] = 0; /* active=0 (set later) */
                nop_mqd[131] = 0; /* vmid=0 */
                nop_mqd[132] = (0x55 << 8) | 1; /* persistent_state: preload_size=0x55, preload_req=1 (matching tinygrad) */
                nop_mqd[133] = 2; /* pipe_priority */
                nop_mqd[134] = 0xF; /* queue_priority */
                nop_mqd[135] = 0x111; /* cp_hqd_quantum (tinygrad value) */
                /* PQ base = ring_addr >> 8 */
                nop_mqd[136] = (ULONG)((nop_ring_mc >> 8) & 0xFFFFFFFF);
                nop_mqd[137] = (ULONG)(nop_ring_mc >> 40);
                nop_mqd[138] = 0; /* rptr */
                nop_mqd[139] = (ULONG)(nop_rptr_mc & 0xFFFFFFFF);
                nop_mqd[140] = (ULONG)(nop_rptr_mc >> 32);
                nop_mqd[141] = (ULONG)(nop_wptr_mc & 0xFFFFFFFF);
                nop_mqd[142] = (ULONG)((nop_wptr_mc >> 32) & 0xFFFF);
                /* doorbell: index 3, dword offset 6, shifted */
                nop_mqd[143] = (1 << 30) | (6 << 2); /* DOORBELL_EN | OFFSET=6<<2=24=0x18 */
                /* PQ_CONTROL matching tinygrad for AQL:
                 * queue_size=9, rptr_block_size=5, unord_dispatch=0,
                 * AQL: queue_full_en=1 (bit 26), slot_based_wptr=2 (bits [24:23]),
                 * priv_state=1 (bit 30), kmd_queue=1 (bit 31), no_update_rptr=1 (bit 28) */
                nop_mqd[145] = (9 << 0)       /* QUEUE_SIZE */
                             | (5 << 8)       /* RPTR_BLOCK_SIZE (tinygrad: 5) */
                             | (1 << 26)      /* QUEUE_FULL_EN (AQL) */
                             | (2 << 23)      /* SLOT_BASED_WPTR (AQL) */
                             | (1 << 30)      /* PRIV_STATE */
                             | (1u << 31)     /* KMD_QUEUE */
                             | (1 << 28);     /* NO_UPDATE_RPTR */
                /* EOP */
                nop_mqd[152] = (ULONG)((nop_eop_mc >> 8) & 0xFFFFFFFF);
                nop_mqd[153] = (ULONG)(nop_eop_mc >> 40);
                nop_mqd[154] = 9;
                /* IB control: min_ib_avail_size=3 (tinygrad) */
                nop_mqd[149] = (3 << 20); /* CP_HQD_IB_CONTROL */
                /* HQ_STATUS0 = 0x20004000 (tinygrad magic value) */
                nop_mqd[160] = 0x20004000; /* CP_HQD_HQ_STATUS0 */
                /* MQD_CONTROL: priv_state=1 (bit 8) */
                nop_mqd[162] = (1 << 8); /* CP_MQD_CONTROL */
                /* AQL control */
                nop_mqd[181] = 1;
                /* Static thread mgmt: all SEs enabled */
                for (int se = 0; se < 8; se++)
                    nop_mqd[163 + se] = 0xFFFFFFFF; /* compute_static_thread_mgmt_se0-7 (approximate offsets) */
                /* WPTR = 15 (1 AQL pkt - 1, matching tinygrad) */
                nop_mqd[182] = 15;
                nop_mqd[183] = 0;

                /* Program HQD via bulk MQD copy */
                extern void gpu_program_hqd_from_mqd(struct WddmLiteDevice *dev,
                    ULONG me, ULONG pipe, ULONG queue, volatile ULONG *mqd);
                gpu_program_hqd_from_mqd(&g_wddm_lite_dev, 1, 0, 0, nop_mqd);

                /* Set WPTR in poll address */
                volatile ULONG *nop_wptr = (volatile ULONG *)(nop_cpu + 0x2000);
                nop_wptr[0] = 15;
                MemoryBarrier();

                /* HDP flush + doorbell */
                gpu_hdp_flush(&g_wddm_lite_dev);
                ULONG db_off = g_wddm_lite_dev.hw.queues[hqd_idx].doorbell_index * 8;
                volatile ULONGLONG *db = (volatile ULONGLONG *)
                    ((UCHAR *)g_wddm_lite_dev.doorbell_base + db_off);
                pr_info("hsaKmtCreateQueue: VRAM NOP test — doorbell at 0x%x, val=15\n", db_off);
                *db = 15;
                MemoryBarrier();

                /* Wait for RPTR */
                volatile ULONG *nop_rptr = (volatile ULONG *)(nop_cpu + 0x1000);
                for (int poll = 0; poll < 100; poll++) {
                    Sleep(10);
                    if (nop_rptr[0] != 0) {
                        pr_info("hsaKmtCreateQueue: VRAM NOP test PASSED! "
                                "RPTR=%u after %d ms\n",
                                nop_rptr[0], (poll + 1) * 10);
                        break;
                    }
                    if (poll == 99) {
                        extern ULONG gpu_read_hqd_wptr(struct WddmLiteDevice *dev,
                            ULONG me, ULONG pipe, ULONG queue);
                        ULONG w = gpu_read_hqd_wptr(&g_wddm_lite_dev, 1, 0, 0);
                        /* Check GFXHUB fault */
                        extern ULONG gpu_check_gfxhub_fault(struct WddmLiteDevice *dev);
                        ULONG fault = gpu_check_gfxhub_fault(&g_wddm_lite_dev);
                        pr_warn("hsaKmtCreateQueue: VRAM NOP test FAILED — "
                                "RPTR=0 after 1000ms. HQD WPTR=%u FAULT=0x%08x\n", w, fault);
                    }
                }
                } /* end else (VRAM mapped OK) */
            }
        }
    } else {
        pr_warn("hsaKmtCreateQueue: GFX engine not initialized or no HQD slots, "
                "queue will be non-functional\n");
    }

    /* Also send escape to WDDM driver for bookkeeping */
    AMDGPU_ESCAPE_CREATE_QUEUE_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_CREATE_QUEUE;
    data.Header.Size = sizeof(data);
    data.GpuId = NodeId;
    data.QueueType = hsa_queue_type_to_escape(Type);
    data.QueuePercentage = QueuePercentage;
    data.Priority = (LONG)Priority;
    data.QueueAddress = (ULONGLONG)QueueAddress;
    data.QueueSizeInBytes = QueueSizeInBytes;

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0) {
        pr_warn("hsaKmtCreateQueue: escape failed (non-fatal)\n");
    }

    /* Fill in queue resource */
    ULONG queue_id = s_next_queue_id++;
    QueueResource->QueueId = queue_id;
    /*
     * Doorbell byte offset = doorbell_index * 8 (each AQL doorbell is 8 bytes).
     * For queues beyond GPU_MAX_COMPUTE_QUEUES, use doorbell slot 0 (no HQD backing).
     */
    ULONG db_byte_off = 0;
    if (hqd_idx < GPU_MAX_COMPUTE_QUEUES)
        db_byte_off = g_wddm_lite_dev.hw.queues[hqd_idx].doorbell_index * 8;
    QueueResource->Queue_DoorBell_aql = (HSAuint64 *)(
        (char *)g_wddm_lite_dev.doorbell_base + db_byte_off);
    QueueResource->Queue_read_ptr_aql = (HSAuint64 *)s_queues[slot].rptr.cpu_addr;
    QueueResource->Queue_write_ptr_aql = (HSAuint64 *)s_queues[slot].wptr.cpu_addr;

    /* Track resources */
    s_queues[slot].in_use = TRUE;
    s_queues[slot].queue_id = queue_id;
    s_queues[slot].hqd_queue_idx = hqd_idx;

    pr_info("hsaKmtCreateQueue: created queue %u, hqd=%u, "
            "doorbell=%p (byte_off=0x%x), rptr=%p, wptr=%p\n",
            queue_id, hqd_idx,
            (void *)QueueResource->Queue_DoorBell_aql,
            (unsigned)db_byte_off,
            (void *)QueueResource->Queue_read_ptr_aql,
            (void *)QueueResource->Queue_write_ptr_aql);

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtCreateQueueExt(HSAuint32 NodeId, HSA_QUEUE_TYPE Type,
                      HSAuint32 QueuePercentage, HSA_QUEUE_PRIORITY Priority,
                      HSAuint32 SdmaEngineId, void *QueueAddress,
                      HSAuint64 QueueSizeInBytes, HsaEvent *Event,
                      HsaQueueResource *QueueResource)
{
    /* Delegate to hsaKmtCreateQueue — SdmaEngineId is ignored for compute */
    return hsaKmtCreateQueue(NodeId, Type, QueuePercentage, Priority,
                             QueueAddress, QueueSizeInBytes, Event,
                             QueueResource);
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtUpdateQueue(HSA_QUEUEID QueueId, HSAuint32 QueuePercentage,
                   HSA_QUEUE_PRIORITY Priority,
                   void *QueueAddress, HSAuint64 QueueSizeInBytes,
                   HsaEvent *Event)
{
    CHECK_KFD_OPEN();

    AMDGPU_ESCAPE_UPDATE_QUEUE_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_UPDATE_QUEUE;
    data.Header.Size = sizeof(data);
    data.QueueId = QueueId;
    data.QueuePercentage = QueuePercentage;
    data.Priority = (LONG)Priority;
    data.QueueAddress = (ULONGLONG)QueueAddress;
    data.QueueSizeInBytes = QueueSizeInBytes;

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0)
        return HSAKMT_STATUS_ERROR;

    if (data.Header.Status != STATUS_SUCCESS)
        return HSAKMT_STATUS_ERROR;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtDestroyQueue(HSA_QUEUEID QueueId)
{
    CHECK_KFD_OPEN();

    /* Find and clean up tracked resources */
    for (int i = 0; i < MAX_QUEUES; i++) {
        if (s_queues[i].in_use && s_queues[i].queue_id == (ULONG)QueueId) {
            pr_info("hsaKmtDestroyQueue: destroying queue %u (hqd=%u)\n",
                    (ULONG)QueueId, s_queues[i].hqd_queue_idx);

            /* Deactivate HQD before freeing memory the GPU may reference */
            gpu_deactivate_compute_queue(&g_wddm_lite_dev,
                                          s_queues[i].hqd_queue_idx);

            /* Free DMA buffers and their GART mappings */
            free_dma_buffer(&g_wddm_lite_dev, &s_queues[i].rptr);
            free_dma_buffer(&g_wddm_lite_dev, &s_queues[i].wptr);
            free_dma_buffer(&g_wddm_lite_dev, &s_queues[i].eop);

            /* Unmap ring buffer from GART */
            if (s_queues[i].ring_gpu_addr && s_queues[i].ring_num_pages)
                gpu_gart_unmap(&g_wddm_lite_dev,
                               s_queues[i].ring_gpu_addr,
                               s_queues[i].ring_num_pages);

            /* Release HQD index for reuse */
            free_hqd_index(s_queues[i].hqd_queue_idx);

            s_queues[i].in_use = FALSE;
            break;
        }
    }

    /* Send escape to WDDM driver */
    AMDGPU_ESCAPE_DESTROY_QUEUE_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_DESTROY_QUEUE;
    data.Header.Size = sizeof(data);
    data.QueueId = QueueId;

    wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data));

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSetTrapHandler(HSAuint32 Node, void *TrapHandlerBaseAddress,
                      HSAuint64 TrapHandlerSizeInBytes,
                      void *TrapBufferBaseAddress,
                      HSAuint64 TrapBufferSizeInBytes)
{
    CHECK_KFD_OPEN();

    AMDGPU_ESCAPE_SET_TRAP_HANDLER_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_SET_TRAP_HANDLER;
    data.Header.Size = sizeof(data);
    data.GpuId = Node;
    data.TbaAddress = (ULONGLONG)TrapHandlerBaseAddress;
    data.TbaSize = TrapHandlerSizeInBytes;
    data.TmaAddress = (ULONGLONG)TrapBufferBaseAddress;
    data.TmaSize = TrapBufferSizeInBytes;

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0)
        return HSAKMT_STATUS_ERROR;

    if (data.Header.Status != STATUS_SUCCESS)
        return HSAKMT_STATUS_ERROR;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetTileConfig(HSAuint32 NodeId, HsaGpuTileConfig *config)
{
    CHECK_KFD_OPEN();

    if (!config)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    /* GFX12 doesn't use tile swizzling in the traditional sense.
     * Return minimal config. */
    memset(config, 0, sizeof(*config));
    config->NumTileConfigs = 0;
    config->NumMacroTileConfigs = 0;

    return HSAKMT_STATUS_SUCCESS;
}
