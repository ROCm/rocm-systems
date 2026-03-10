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
#include <string.h>

extern struct WddmLiteDevice g_wddm_lite_dev;

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

    pr_err("hsaKmtCreateQueue: calling escape...\n");
    fflush(stderr);
    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0) {
        pr_err("hsaKmtCreateQueue: escape failed\n");
        return HSAKMT_STATUS_ERROR;
    }

    if (data.Header.Status != STATUS_SUCCESS) {
        pr_err("hsaKmtCreateQueue: driver returned 0x%lx\n",
               (unsigned long)data.Header.Status);
        return HSAKMT_STATUS_ERROR;
    }

    /* Map doorbell BAR if not already done */
    if (ensure_doorbell_mapped(&g_wddm_lite_dev) != 0) {
        pr_err("hsaKmtCreateQueue: doorbell mapping failed\n");
        return HSAKMT_STATUS_ERROR;
    }

    QueueResource->QueueId = data.QueueId;
    /* Return a pointer into the mapped doorbell BAR, not just an offset */
    QueueResource->Queue_DoorBell_aql = (HSAuint64 *)(
        (char *)g_wddm_lite_dev.doorbell_base + data.DoorbellOffset);

    pr_info("hsaKmtCreateQueue: created queue %llu, doorbell offset 0x%llx, "
            "doorbell ptr %p\n",
            (unsigned long long)data.QueueId,
            (unsigned long long)data.DoorbellOffset,
            (void *)QueueResource->Queue_DoorBell_aql);

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtCreateQueueExt(HSAuint32 NodeId, HSA_QUEUE_TYPE Type,
                      HSAuint32 QueuePercentage, HSA_QUEUE_PRIORITY Priority,
                      HSAuint32 SdmaEngineId, void *QueueAddress,
                      HSAuint64 QueueSizeInBytes, HsaEvent *Event,
                      HsaQueueResource *QueueResource)
{
    CHECK_KFD_OPEN();

    if (!QueueAddress || !QueueResource || QueueSizeInBytes == 0)
        return HSAKMT_STATUS_INVALID_PARAMETER;

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
    data.SdmaEngineId = SdmaEngineId;

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0)
        return HSAKMT_STATUS_ERROR;

    if (data.Header.Status != STATUS_SUCCESS)
        return HSAKMT_STATUS_ERROR;

    /* Map doorbell BAR if not already done */
    if (ensure_doorbell_mapped(&g_wddm_lite_dev) != 0)
        return HSAKMT_STATUS_ERROR;

    QueueResource->QueueId = data.QueueId;
    QueueResource->Queue_DoorBell_aql = (HSAuint64 *)(
        (char *)g_wddm_lite_dev.doorbell_base + data.DoorbellOffset);

    return HSAKMT_STATUS_SUCCESS;
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

    AMDGPU_ESCAPE_DESTROY_QUEUE_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_DESTROY_QUEUE;
    data.Header.Size = sizeof(data);
    data.QueueId = QueueId;

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0)
        return HSAKMT_STATUS_ERROR;

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
