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

#ifndef __KFD_AQL_QUEUE__H__
#define __KFD_AQL_QUEUE__H__

#include "BaseQueue.hpp"

class AqlQueue : public BaseQueue {
 public:
    AqlQueue();
    virtual ~AqlQueue();

    // @brief Updates queue write pointer and sets the queue doorbell to the queue write pointer
    virtual void SubmitPacket();

    // @return Read pointer in dwords
    virtual unsigned int Rptr();
    // @return Write pointer in dwords
    virtual unsigned int Wptr();
    // @return Expected m_Resources.Queue_read_ptr when all packets are consumed
    virtual unsigned int RptrWhenConsumed();

 protected:
    virtual PACKETTYPE PacketTypeSupported() { return PACKETTYPE_AQL; }

    virtual _HSA_QUEUE_TYPE GetQueueType() { return HSA_QUEUE_COMPUTE_AQL; }
};

#endif  // __KFD_AQL_QUEUE__H__
