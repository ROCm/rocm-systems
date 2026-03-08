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
        pr_err("hsaKmtCreateQueue: escape failed\n");
        return HSAKMT_STATUS_ERROR;
    }

    if (data.Header.Status != STATUS_SUCCESS) {
        pr_err("hsaKmtCreateQueue: driver returned 0x%lx\n",
               (unsigned long)data.Header.Status);
        return HSAKMT_STATUS_ERROR;
    }

    QueueResource->QueueId = data.QueueId;
    QueueResource->QueueDoorBell = data.DoorbellOffset;

    pr_info("hsaKmtCreateQueue: created queue %llu, doorbell offset 0x%llx\n",
            (unsigned long long)data.QueueId,
            (unsigned long long)data.DoorbellOffset);

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

    QueueResource->QueueId = data.QueueId;
    QueueResource->QueueDoorBell = data.DoorbellOffset;

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
