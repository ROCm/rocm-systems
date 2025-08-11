/*
 * Copyright (C) 2017-2018 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "BasePacket.hpp"
#include "KFDTestUtil.hpp"
#include "KFDBaseComponentTest.hpp"

BasePacket::BasePacket(void): m_packetAllocation(NULL) {
    m_FamilyId = g_baseTest->GetFamilyIdFromDefaultNode();
}

BasePacket::~BasePacket(void) {
    if (m_packetAllocation)
        free(m_packetAllocation);
}

void BasePacket::Dump() const {
    unsigned int size = SizeInDWords();
    const HSAuint32 *packet = (const HSAuint32 *)GetPacket();
    std::ostream &log = LOG();
    unsigned int i;

    log << "Packet dump:" << std::hex;
    for (i = 0; i < size; i++)
        log << " " << std::setw(8) << std::setfill('0') << packet[i];
    log << std::endl;
}

void *BasePacket::AllocPacket(void) {
    unsigned int size = SizeInBytes();

    EXPECT_NE(0, size);
    if (!size)
        return NULL;

    m_packetAllocation = calloc(1, size);
    EXPECT_NOTNULL(m_packetAllocation);

    return m_packetAllocation;
}
