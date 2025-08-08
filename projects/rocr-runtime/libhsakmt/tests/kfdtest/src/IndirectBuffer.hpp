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

#ifndef __INDIRECT_BUFFER__H__
#define __INDIRECT_BUFFER__H__

#include "BasePacket.hpp"
#include "KFDTestUtil.hpp"

/** @class IndirectBuffer
 *  When working with an indirect buffer, create IndirectBuffer, fill it with all the packets you want,
 *  create an indirect packet to point to it, and submit the packet to queue
 */
class IndirectBuffer {
 public:
    // @param[size] Queue max size in DWords
    // @param[type] Packet type allowed in queue
    IndirectBuffer(PACKETTYPE type, unsigned int sizeInDWords, unsigned int NodeId);
    ~IndirectBuffer(void);

    // @brief Add packet to queue, all validations are done with gtest ASSERT and EXPECT
    uint32_t *AddPacket(const BasePacket &packet);
    // @returns Actual size of the indirect queue in DWords, equivalent to write pointer
    unsigned int SizeInDWord() { return m_ActualSize; }
    // @returns Indirect queue address
    unsigned int *Addr() { return m_IndirectBuf->As<unsigned int*>(); }

 protected:
    // Number of packets in the queue
    unsigned int m_NumOfPackets;
    // Max size of queue in DWords
    unsigned int m_MaxSize;
    // Current size of queue in DWords
    unsigned int m_ActualSize;
    HsaMemoryBuffer *m_IndirectBuf;
    // What packets are supported in this queue
    PACKETTYPE m_PacketTypeAllowed;
};

#endif  //  __INDIRECT_BUFFER__H__
