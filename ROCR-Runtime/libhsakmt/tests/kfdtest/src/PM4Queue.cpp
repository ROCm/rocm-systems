/*
 * Copyright (C) 2014-2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "PM4Queue.hpp"
#include "pm4_pkt_struct_common.h"
#include "GoogleTestExtension.hpp"
#include "kfd_pm4_opcodes.h"


PM4Queue::PM4Queue(void) {
    CMD_NOP = CMD_NOP_TYPE_3;
}

PM4Queue::~PM4Queue(void) {
}

unsigned int PM4Queue::Wptr() {
    /* Write pointer in dwords. Simulate 32-bit wptr that wraps at
     * queue size even on Vega10 and later chips with 64-bit wptr.
     */
    return *m_Resources.Queue_write_ptr % (m_QueueBuf->Size() / 4);
}

unsigned int PM4Queue::Rptr() {
    /* CP read pointer in dwords. It's still 32-bit even on Vega10. */
    return *m_Resources.Queue_read_ptr;
}

unsigned int PM4Queue::RptrWhenConsumed() {
    /* On PM4 queues Rptr is always 32-bit in dword units and wraps at
     * queue size. The expected value when all packets are consumed is
     * exactly the value returned by Wptr().
     */
    return Wptr();
}

void PM4Queue::SubmitPacket() {
    // m_pending Wptr is in dwords
    if (m_FamilyId < FAMILY_AI) {
        // Pre-Vega10 uses 32-bit wptr and doorbell
        MemoryBarrier();
        *m_Resources.Queue_write_ptr = m_pendingWptr;
        MemoryBarrier();
        *(m_Resources.Queue_DoorBell) = m_pendingWptr;
    } else {
        // Vega10 and later uses 64-bit wptr and doorbell
        MemoryBarrier();
        *m_Resources.Queue_write_ptr_aql = m_pendingWptr64;
        MemoryBarrier();
        *(m_Resources.Queue_DoorBell_aql) = m_pendingWptr64;
    }
}

void PM4Queue::Wait4PacketConsumption(HsaEvent *event, unsigned int timeOut) {
    if (event) {
        PlaceAndSubmitPacket(PM4ReleaseMemoryPacket(m_FamilyId, 0,
                    event->EventData.HWData2,
                    event->EventId,
                    true));

        EXPECT_SUCCESS(hsaKmtWaitOnEvent(event, timeOut));
    } else {
        BaseQueue::Wait4PacketConsumption(NULL, timeOut);
    }
}
