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
 * Event management for wddm_lite backend.
 * Creates, signals, and waits on completion events through the WDDM
 * driver's escape interface. Events are used by ROCR-Runtime for
 * signal/fence synchronization.
 */

#include "wddm_lite_internal.h"
#include "wddm_lite_device.h"
#include <stdlib.h>
#include <string.h>

extern struct WddmLiteDevice g_wddm_lite_dev;

HSAKMT_STATUS HSAKMTAPI
hsaKmtCreateEvent(HsaEventDescriptor *EventDesc, bool ManualReset,
                   bool IsSignaled, HsaEvent **Event)
{
    CHECK_KFD_OPEN();

    if (!EventDesc || !Event)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    AMDGPU_ESCAPE_CREATE_EVENT_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_CREATE_EVENT;
    data.Header.Size = sizeof(data);
    data.EventType = (ULONG)EventDesc->EventType;
    data.GpuId = EventDesc->NodeId;
    data.AutoReset = !ManualReset;

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0) {
        pr_err("hsaKmtCreateEvent: escape failed\n");
        return HSAKMT_STATUS_ERROR;
    }

    if (data.Header.Status != STATUS_SUCCESS) {
        pr_err("hsaKmtCreateEvent: driver returned 0x%lx\n",
               (unsigned long)data.Header.Status);
        return HSAKMT_STATUS_ERROR;
    }

    /* Allocate userspace HsaEvent structure */
    HsaEvent *evt = (HsaEvent *)calloc(1, sizeof(HsaEvent));
    if (!evt)
        return HSAKMT_STATUS_NO_MEMORY;

    evt->EventId = data.EventId;
    evt->EventData.EventType = EventDesc->EventType;

    /* Store the event page address and slot in HWData fields.
     * ROCR uses HWData2 as the signal address for doorbell-based signaling. */
    evt->EventData.HWData1 = 0; /* OS event handle - managed by driver */
    evt->EventData.HWData2 = data.EventPageAddress +
                              (data.EventSlotIndex * sizeof(uint64_t));
    evt->EventData.HWData3 = data.EventSlotIndex;

    /* Copy sync variable from descriptor */
    evt->EventData.EventData.SyncVar = EventDesc->SyncVar;

    if (IsSignaled) {
        /* Signal immediately if requested */
        AMDGPU_ESCAPE_SET_EVENT_DATA set_data;
        memset(&set_data, 0, sizeof(set_data));
        set_data.Header.Command = AMDGPU_ESCAPE_SET_EVENT;
        set_data.Header.Size = sizeof(set_data);
        set_data.EventId = data.EventId;
        wddm_lite_escape(&g_wddm_lite_dev, &set_data, sizeof(set_data));
    }

    *Event = evt;
    pr_info("hsaKmtCreateEvent: created event %u, page addr 0x%llx slot %u\n",
            data.EventId,
            (unsigned long long)data.EventPageAddress,
            data.EventSlotIndex);

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtDestroyEvent(HsaEvent *Event)
{
    CHECK_KFD_OPEN();

    if (!Event)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    AMDGPU_ESCAPE_DESTROY_EVENT_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_DESTROY_EVENT;
    data.Header.Size = sizeof(data);
    data.EventId = Event->EventId;

    wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data));

    free(Event);
    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtSetEvent(HsaEvent *Event)
{
    CHECK_KFD_OPEN();

    if (!Event)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    AMDGPU_ESCAPE_SET_EVENT_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_SET_EVENT;
    data.Header.Size = sizeof(data);
    data.EventId = Event->EventId;

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0)
        return HSAKMT_STATUS_ERROR;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtResetEvent(HsaEvent *Event)
{
    CHECK_KFD_OPEN();

    if (!Event)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    AMDGPU_ESCAPE_RESET_EVENT_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_RESET_EVENT;
    data.Header.Size = sizeof(data);
    data.EventId = Event->EventId;

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0)
        return HSAKMT_STATUS_ERROR;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtQueryEventState(HsaEvent *Event)
{
    CHECK_KFD_OPEN();

    if (!Event)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    /* Check event state without blocking.
     * Use WaitEvents with 0 timeout. */
    AMDGPU_ESCAPE_WAIT_EVENTS_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_WAIT_EVENTS;
    data.Header.Size = sizeof(data);
    data.NumEvents = 1;
    data.EventIds[0] = Event->EventId;
    data.WaitAll = FALSE;
    data.TimeoutMs = 0;

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0)
        return HSAKMT_STATUS_ERROR;

    /* STATUS_TIMEOUT means not signaled yet */
    if (data.Header.Status == (NTSTATUS)0x00000102L) /* STATUS_TIMEOUT */
        return HSAKMT_STATUS_WAIT_TIMEOUT;

    if (data.Header.Status != STATUS_SUCCESS)
        return HSAKMT_STATUS_ERROR;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtWaitOnEvent(HsaEvent *Event, HSAuint32 Milliseconds)
{
    CHECK_KFD_OPEN();

    if (!Event)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    AMDGPU_ESCAPE_WAIT_EVENTS_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_WAIT_EVENTS;
    data.Header.Size = sizeof(data);
    data.NumEvents = 1;
    data.EventIds[0] = Event->EventId;
    data.WaitAll = FALSE;
    data.TimeoutMs = Milliseconds;

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0)
        return HSAKMT_STATUS_ERROR;

    if (data.Header.Status == (NTSTATUS)0x00000102L)
        return HSAKMT_STATUS_WAIT_TIMEOUT;

    if (data.Header.Status != STATUS_SUCCESS)
        return HSAKMT_STATUS_ERROR;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtWaitOnEvent_Ext(HsaEvent *Event, HSAuint32 Milliseconds,
                       uint64_t *event_age)
{
    HSAKMT_STATUS status = hsaKmtWaitOnEvent(Event, Milliseconds);
    if (event_age)
        *event_age = 0;
    return status;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtWaitOnMultipleEvents(HsaEvent *Events[], HSAuint32 NumEvents,
                            bool WaitOnAll, HSAuint32 Milliseconds)
{
    CHECK_KFD_OPEN();

    if (!Events || NumEvents == 0)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    if (NumEvents > AMDGPU_MAX_WAIT_EVENTS)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    AMDGPU_ESCAPE_WAIT_EVENTS_DATA data;
    memset(&data, 0, sizeof(data));
    data.Header.Command = AMDGPU_ESCAPE_WAIT_EVENTS;
    data.Header.Size = sizeof(data);
    data.NumEvents = NumEvents;
    data.WaitAll = WaitOnAll ? TRUE : FALSE;
    data.TimeoutMs = Milliseconds;

    for (HSAuint32 i = 0; i < NumEvents; i++) {
        if (!Events[i])
            return HSAKMT_STATUS_INVALID_PARAMETER;
        data.EventIds[i] = Events[i]->EventId;
    }

    if (wddm_lite_escape(&g_wddm_lite_dev, &data, sizeof(data)) != 0)
        return HSAKMT_STATUS_ERROR;

    if (data.Header.Status == (NTSTATUS)0x00000102L)
        return HSAKMT_STATUS_WAIT_TIMEOUT;

    if (data.Header.Status != STATUS_SUCCESS)
        return HSAKMT_STATUS_ERROR;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtWaitOnMultipleEvents_Ext(HsaEvent *Events[], HSAuint32 NumEvents,
                                bool WaitOnAll, HSAuint32 Milliseconds,
                                uint64_t *event_age)
{
    HSAKMT_STATUS status = hsaKmtWaitOnMultipleEvents(Events, NumEvents,
                                                       WaitOnAll, Milliseconds);
    if (event_age)
        *event_age = 0;
    return status;
}
