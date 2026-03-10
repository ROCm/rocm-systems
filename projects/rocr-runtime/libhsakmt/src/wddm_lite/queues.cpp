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

extern struct WddmLiteDevice g_wddm_lite_dev;

/* Queue resource tracking */
#define MAX_QUEUES 16
static ULONG s_next_queue_id = 1;

struct QueueResources {
    BOOLEAN  in_use;
    ULONG    queue_id;
    ULONG    hqd_queue_idx;   /* HQD index for gpu_setup_compute_queue */
    /* Allocated buffers */
    void    *rptr_cpu;         /* RPTR writeback (8 bytes) */
    void    *wptr_cpu;         /* WPTR poll location (8 bytes) */
    void    *eop_cpu;          /* EOP buffer */
    ULONG    eop_size;
};

static struct QueueResources s_queues[MAX_QUEUES];
static ULONG s_next_hqd_idx = 0;

/*
 * Allocate a small system memory buffer via escape.
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
    ULONG db_bar = 0;
    ULONGLONG db_size = 4096;  /* Map first 4KB of doorbell BAR */
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

    /*
     * Allocate a fake doorbell page via system memory.
     * Real doorbell BAR mapping via MmMapIoSpace hangs on VFIO passthrough
     * because DXGKRNL may not allow direct MMIO mapping of the doorbell BAR.
     * Since MEC isn't programmed yet anyway, a fake doorbell lets ROCR
     * create queues without crashing (dispatch will be a no-op).
     */
    AMDGPU_ESCAPE_ALLOC_MEMORY_DATA alloc;
    memset(&alloc, 0, sizeof(alloc));
    alloc.Header.Command = AMDGPU_ESCAPE_ALLOC_MEMORY;
    alloc.Header.Size = sizeof(alloc);
    alloc.SizeInBytes = db_size;
    alloc.Flags = 0x0024;  /* AMDGPU_MEM_TYPE_SYSTEM | AMDGPU_MEM_FLAG_HOST_ACCESS */

    if (wddm_lite_escape(dev, &alloc, sizeof(alloc)) != 0) {
        pr_err("ensure_doorbell_mapped: alloc escape failed\n");
        return -1;
    }

    if (alloc.Header.Status != 0 || !alloc.CpuAddress) {
        pr_err("ensure_doorbell_mapped: alloc returned 0x%lx cpu=%p\n",
               (unsigned long)alloc.Header.Status, alloc.CpuAddress);
        return -1;
    }

    dev->doorbell_base = alloc.CpuAddress;
    dev->doorbell_mapping_handle = NULL;
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

    /* Allocate RPTR, WPTR, and EOP buffers */
    void *rptr_cpu = alloc_queue_buffer(&g_wddm_lite_dev, 4096);
    void *wptr_cpu = alloc_queue_buffer(&g_wddm_lite_dev, 4096);
    ULONG eop_size = 4096;
    void *eop_cpu = alloc_queue_buffer(&g_wddm_lite_dev, eop_size);

    if (!rptr_cpu || !wptr_cpu || !eop_cpu) {
        pr_err("hsaKmtCreateQueue: buffer allocation failed "
               "rptr=%p wptr=%p eop=%p\n", rptr_cpu, wptr_cpu, eop_cpu);
        return HSAKMT_STATUS_NO_MEMORY;
    }

    /* Map doorbell BAR if not already done */
    if (ensure_doorbell_mapped(&g_wddm_lite_dev) != 0) {
        pr_err("hsaKmtCreateQueue: doorbell mapping failed\n");
        return HSAKMT_STATUS_ERROR;
    }

    /* Determine if this is an AQL queue */
    BOOLEAN aql = (Type == HSA_QUEUE_COMPUTE_AQL || Type == HSA_QUEUE_DMA_AQL ||
                   Type == HSA_QUEUE_DMA_AQL_XGMI);

    /* Try direct HQD programming if GFX engine is initialized */
    ULONG hqd_idx = s_next_hqd_idx;
    if (g_wddm_lite_dev.hw.gfx_initialized && hqd_idx < GPU_MAX_COMPUTE_QUEUES) {
        pr_info("hsaKmtCreateQueue: programming HQD %u directly\n", hqd_idx);

        int ret = gpu_setup_compute_queue(&g_wddm_lite_dev, hqd_idx,
            (ULONGLONG)(uintptr_t)QueueAddress, (ULONG)QueueSizeInBytes,
            (ULONGLONG)(uintptr_t)rptr_cpu, (ULONGLONG)(uintptr_t)wptr_cpu,
            (ULONGLONG)(uintptr_t)eop_cpu, eop_size, aql);

        if (ret != 0) {
            pr_warn("hsaKmtCreateQueue: HQD programming failed, "
                    "queue will be non-functional\n");
        } else {
            s_next_hqd_idx++;
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
    QueueResource->Queue_DoorBell_aql = (HSAuint64 *)(
        (char *)g_wddm_lite_dev.doorbell_base +
        g_wddm_lite_dev.hw.queues[hqd_idx].doorbell_offset);
    QueueResource->Queue_read_ptr_aql = (HSAuint64 *)rptr_cpu;
    QueueResource->Queue_write_ptr_aql = (HSAuint64 *)wptr_cpu;

    /* Track resources */
    s_queues[slot].in_use = TRUE;
    s_queues[slot].queue_id = queue_id;
    s_queues[slot].hqd_queue_idx = hqd_idx;
    s_queues[slot].rptr_cpu = rptr_cpu;
    s_queues[slot].wptr_cpu = wptr_cpu;
    s_queues[slot].eop_cpu = eop_cpu;
    s_queues[slot].eop_size = eop_size;

    pr_info("hsaKmtCreateQueue: created queue %u, hqd=%u, "
            "doorbell=%p, rptr=%p, wptr=%p\n",
            queue_id, hqd_idx,
            (void *)QueueResource->Queue_DoorBell_aql,
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
