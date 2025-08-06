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

#include "SDMAQueue.hpp"
#include "SDMAPacket.hpp"

SDMAQueue::SDMAQueue(void) {
     CMD_NOP = 0;
}

SDMAQueue::~SDMAQueue(void) {
}

unsigned int SDMAQueue::Wptr() {
    /* In SDMA queues write pointers are saved in bytes, convert the
     * wptr value to dword to fit the way BaseQueue works. On Vega10
     * the write ptr is 64-bit. We only read the low 32 bit (assuming
     * the queue buffer is smaller than 4GB) and modulo divide by the
     * queue size to simulate a 32-bit read pointer.
     */
    return (*m_Resources.Queue_write_ptr % m_QueueBuf->Size()) /
        sizeof(unsigned int);
}

unsigned int SDMAQueue::Rptr() {
    /* In SDMA queues read pointers are saved in bytes, convert the
     * read value to dword to fit the way BaseQueue works. On Vega10
     * the read ptr is 64-bit. We only read the low 32 bit (assuming
     * the queue buffer is smaller than 4GB) and modulo divide by the
     * queue size to simulate a 32-bit read pointer.
     */
    return (*m_Resources.Queue_read_ptr % m_QueueBuf->Size()) /
        sizeof(unsigned int);
}

unsigned int SDMAQueue::RptrWhenConsumed() {
    /* Rptr is same size and byte units as Wptr. Here we only care
     * about the low 32-bits. When all packets are consumed, read and
     * write pointers should have the same value.
     */
    return *m_Resources.Queue_write_ptr;
}

void SDMAQueue::SubmitPacket() {
    // m_pending Wptr is in dwords
    if (m_FamilyId < FAMILY_AI) {
        // Pre-Vega10 uses 32-bit wptr and doorbell
        unsigned int wPtrInBytes = m_pendingWptr * sizeof(unsigned int);
        MemoryBarrier();
        *m_Resources.Queue_write_ptr = wPtrInBytes;
        MemoryBarrier();
        *(m_Resources.Queue_DoorBell) = wPtrInBytes;
    } else {
        // Vega10 and later uses 64-bit wptr and doorbell
        HSAuint64 wPtrInBytes = m_pendingWptr64 * sizeof(unsigned int);
        MemoryBarrier();
        *m_Resources.Queue_write_ptr_aql = wPtrInBytes;
        MemoryBarrier();
        *(m_Resources.Queue_DoorBell_aql) = wPtrInBytes;
    }
}

void SDMAQueue::Wait4PacketConsumption(HsaEvent *event, unsigned int timeOut) {
    if (event) {
        PlacePacket(SDMAFencePacket(m_FamilyId, (void*)event->EventData.HWData2, event->EventId));

        PlaceAndSubmitPacket(SDMATrapPacket(event->EventId));

        EXPECT_SUCCESS(hsaKmtWaitOnEvent(event, timeOut));
    } else {
        BaseQueue::Wait4PacketConsumption(NULL, timeOut);
    }
}
