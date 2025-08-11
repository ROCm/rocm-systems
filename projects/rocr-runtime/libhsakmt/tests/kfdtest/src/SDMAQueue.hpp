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

#ifndef __KFD_SDMA_QUEUE__H__
#define __KFD_SDMA_QUEUE__H__

#include "BaseQueue.hpp"

class SDMAQueue : public BaseQueue {
 public:
    SDMAQueue(void);
    virtual ~SDMAQueue(void);

    // @brief Update queue write pointer and set the queue doorbell to the queue write pointer
    virtual void SubmitPacket();

    /** Wait for all the packets submitted to the queue to be consumed. (i.e. wait until RPTR=WPTR).
     *  Note that all packets being consumed is not the same as all packets being processed.
     *  If event is set, wait all packets being processed.
     *  And we can benefit from that as it has
     *  1) Less CPU usage (process can sleep, waiting for interrupt).
     *  2) Lower latency (GPU only updates RPTR in memory periodically).
     */
    virtual void Wait4PacketConsumption(HsaEvent *event = NULL, unsigned int timeOut = g_TestTimeOut);

 protected:
    // @ return Write pointer modulo queue size in dwords
    virtual unsigned int Wptr();
    // @ return Read pointer modulo queue size in dwords
    virtual unsigned int Rptr();
    // @ return Expected m_Resources.Queue_read_ptr when all packets are consumed
    virtual unsigned int RptrWhenConsumed();

    virtual PACKETTYPE PacketTypeSupported() { return PACKETTYPE_SDMA; }

    virtual _HSA_QUEUE_TYPE GetQueueType() { return HSA_QUEUE_SDMA; }
};

#endif  // __KFD_SDMA_QUEUE__H__
