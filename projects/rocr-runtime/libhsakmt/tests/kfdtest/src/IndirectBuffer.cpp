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

#include "IndirectBuffer.hpp"
#include "GoogleTestExtension.hpp"
#include "pm4_pkt_struct_common.h"
#include "PM4Packet.hpp"


IndirectBuffer::IndirectBuffer(PACKETTYPE type,  unsigned int sizeInDWords, unsigned int NodeId)
    :m_NumOfPackets(0), m_MaxSize(sizeInDWords), m_ActualSize(0), m_PacketTypeAllowed(type) {
    m_IndirectBuf = new HsaMemoryBuffer(sizeInDWords*sizeof(unsigned int), NodeId, true/*zero*/,
                                        false/*local*/, true/*exec*/, false/*isScratch*/,
                                        false/*isReadOnly*/, true/*isUncached*/);
}

IndirectBuffer::~IndirectBuffer(void) {
    delete m_IndirectBuf;
}

uint32_t *IndirectBuffer::AddPacket(const BasePacket &packet) {
    EXPECT_EQ(packet.PacketType(), m_PacketTypeAllowed) << "Cannot add a packet since packet type doesn't match queue";

    unsigned int writePtr = m_ActualSize;

    EXPECT_GE(m_MaxSize, packet.SizeInDWords() + writePtr) << "Cannot add a packet, not enough room";

    memcpy(m_IndirectBuf->As<unsigned int*>() + writePtr , packet.GetPacket(),  packet.SizeInBytes());
    m_ActualSize += packet.SizeInDWords();
    m_NumOfPackets++;

    return m_IndirectBuf->As<HSAuint32 *>() + writePtr;
}
