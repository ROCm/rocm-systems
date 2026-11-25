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

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "SDMAPacket.hpp"
#include "KFDTestUtil.hpp"

/* Byte/dword count in many SDMA packets is 1-based in AI, meaning a
 * count of 1 is encoded as 0.
 */
#define SDMA_COUNT(c) (m_FamilyId < FAMILY_AI ? (c) : (c)-1)

SDMAWriteDataPacket::SDMAWriteDataPacket(unsigned int familyId, void* destAddr, unsigned int data,
                                         unsigned int packetSizeOffset):
    packetData(NULL) {
    m_FamilyId = familyId;
    if (familyId < FAMILY_GFX13)
        InitPacket(destAddr, 1, &data, packetSizeOffset);
    else
        InitPacket_v8(destAddr, 1, &data, packetSizeOffset);
}

SDMAWriteDataPacket::SDMAWriteDataPacket(unsigned int familyId, void* destAddr, unsigned int ndw,
                                         void *data):
    packetData(NULL) {
    m_FamilyId = familyId;
    if (familyId < FAMILY_GFX13)
        InitPacket(destAddr, ndw, data);
    else
        InitPacket_v8(destAddr, ndw, data);
}

void SDMAWriteDataPacket::InitPacket(void* destAddr, unsigned int ndw,
                                     void *data, unsigned int packetSizeOffset) {
    SDMA_PKT_WRITE_UNTILED *pSDMA;
    packetSize = sizeof(SDMA_PKT_WRITE_UNTILED) +
        (ndw - 1) * sizeof(unsigned int);
    packetSize -= packetSizeOffset;
    pSDMA = reinterpret_cast<SDMA_PKT_WRITE_UNTILED *>(AllocPacket());
    packetData = reinterpret_cast<void *>(pSDMA);

    pSDMA->HEADER_UNION.op = SDMA_OP_WRITE;
    pSDMA->HEADER_UNION.sub_op = SDMA_SUBOP_WRITE_LINEAR;

    SplitU64(reinterpret_cast<HSAuint64>(destAddr),
             pSDMA->DST_ADDR_LO_UNION.DW_1_DATA,  // dst_addr_31_0
             pSDMA->DST_ADDR_HI_UNION.DW_2_DATA);  // dst_addr_63_32

    pSDMA->DW_3_UNION.count = SDMA_COUNT(ndw);
    if (m_FamilyId >= FAMILY_GFX125X)
        pSDMA->DW_3_UNION.scope = SDMA_SCOPE_SYS;
    memcpy(&pSDMA->DATA0_UNION.DW_4_DATA, data, ndw*sizeof(unsigned int));
}

void SDMAWriteDataPacket::InitPacket_v8(void* destAddr, unsigned int ndw,
                                     void *data, unsigned int packetSizeOffset) {
    SDMA_V8_PKT_WRITE_UNTILED *pSDMA;
    packetSize = sizeof(SDMA_V8_PKT_WRITE_UNTILED) +
        (ndw - 1) * sizeof(unsigned int);
    packetSize -= packetSizeOffset;
    pSDMA = reinterpret_cast<SDMA_V8_PKT_WRITE_UNTILED *>(AllocPacket());
    packetData = reinterpret_cast<void *>(pSDMA);

    pSDMA->HEADER_UNION.op = SDMA_OP_WRITE;
    pSDMA->HEADER_UNION.sub_op = SDMA_SUBOP_WRITE_LINEAR;
    pSDMA->HEADER_UNION.dst_uc_sel = uc_sel_override_io_space;

    SplitU64(reinterpret_cast<HSAuint64>(destAddr),
             pSDMA->DST_ADDR_LO_UNION.DW_1_DATA,  // dst_addr_31_0
             pSDMA->DST_ADDR_HI_UNION.DW_2_DATA);  // dst_addr_63_32

    pSDMA->DW_3_UNION.count = SDMA_COUNT(ndw);
    pSDMA->DW_4_UNION.DW_4_DATA = 0;
    memcpy(&pSDMA->DATA0_UNION.DW_5_DATA, data, ndw*sizeof(unsigned int));
}

SDMACopyDataPacket::SDMACopyDataPacket(unsigned int familyId,
                        void *const dsts[], void *src, int n, unsigned int surfsize):
    packetData(NULL) {
    m_FamilyId = familyId;
    if (familyId < FAMILY_GFX13)
        InitPacket(dsts, src, n, surfsize);
    else
        InitPacket_v8(dsts, src, n, surfsize);
}

SDMACopyDataPacket::SDMACopyDataPacket(unsigned int familyId, void* dst, void *src, unsigned int surfsize) {
    new (this)SDMACopyDataPacket(familyId, &dst, src, 1, surfsize);
}

#define BITS (21)
#define TWO_MEG (1 << BITS)
void SDMACopyDataPacket::InitPacket(void *const dsts[], void *src, int n, unsigned int surfsize) {
    int32_t size = 0, i;
    void **dst = reinterpret_cast<void**>(malloc(sizeof(void*) * n));
    const int singlePacketSize = sizeof(SDMA_PKT_COPY_LINEAR) +
                        sizeof(SDMA_PKT_COPY_LINEAR::DST_ADDR[0]) * n;

    if (n > 2)
        WARN() << "SDMACopyDataPacket does not support more than 2 dst addresses!" << std::endl;

    memcpy(dst, dsts, sizeof(void*) * n);

    packetSize = ((surfsize + TWO_MEG - 1) >> BITS) * singlePacketSize;

    SDMA_PKT_COPY_LINEAR *pSDMA = reinterpret_cast<SDMA_PKT_COPY_LINEAR *>(AllocPacket());
    packetData = reinterpret_cast<void *>(pSDMA);

    while (surfsize > 0) {
        /* SDMA support maximum 0x3fffe0 byte in one copy, take 2M here */
        if (surfsize > TWO_MEG)
            size = TWO_MEG;
        else
            size = surfsize;

        memset(pSDMA, 0, singlePacketSize);
        pSDMA->HEADER_UNION.op           = SDMA_OP_COPY;
        pSDMA->HEADER_UNION.sub_op       = SDMA_SUBOP_COPY_LINEAR;
        pSDMA->HEADER_UNION.broadcast       = n > 1 ? 1 : 0;
        pSDMA->COUNT_UNION.count             = SDMA_COUNT(size);
        if (m_FamilyId >= FAMILY_GFX125X) {
            pSDMA->PARAMETER_UNION.src_scope = SDMA_SCOPE_SYS;
            pSDMA->PARAMETER_UNION.dst_scope = SDMA_SCOPE_SYS;
        }

        SplitU64(reinterpret_cast<HSAuint64>(src),
                 pSDMA->SRC_ADDR_LO_UNION.DW_3_DATA,  // src_addr_31_0
                 pSDMA->SRC_ADDR_HI_UNION.DW_4_DATA);  // src_addr_63_32

        for (i = 0; i < n; i++)
            SplitU64(reinterpret_cast<HSAuint64>(dst[i]),
                    pSDMA->DST_ADDR[i].DST_ADDR_LO_UNION.DW_5_DATA,  // dst_addr_31_0
                    pSDMA->DST_ADDR[i].DST_ADDR_HI_UNION.DW_6_DATA);  // dst_addr_63_32

        pSDMA = reinterpret_cast<SDMA_PKT_COPY_LINEAR *>(reinterpret_cast<char *>(pSDMA) + singlePacketSize);
        for (i = 0; i < n; i++)
            dst[i] = reinterpret_cast<char *>(dst[i]) + size;
        src = reinterpret_cast<char *>(src) + size;
        surfsize -= size;
    }
    free(dst);
}

void SDMACopyDataPacket::InitPacket_v8(void *const dsts[], void *src, int n, unsigned int surfsize) {
    int32_t size = 0, i;
    void **dst = reinterpret_cast<void**>(malloc(sizeof(void*) * n));
    const int singlePacketSize = sizeof(SDMA_V8_PKT_COPY_LINEAR) +
                        sizeof(SDMA_V8_PKT_COPY_LINEAR::DST_ADDR[0]) * n;

    if (n > 2)
        WARN() << "SDMACopyDataPacket does not support more than 2 dst addresses!" << std::endl;

    memcpy(dst, dsts, sizeof(void*) * n);

    packetSize = ((surfsize + TWO_MEG - 1) >> BITS) * singlePacketSize;

    SDMA_V8_PKT_COPY_LINEAR *pSDMA = reinterpret_cast<SDMA_V8_PKT_COPY_LINEAR *>(AllocPacket());
    packetData = reinterpret_cast<void *>(pSDMA);

    while (surfsize > 0) {
        /* SDMA support maximum 0x3fffe0 byte in one copy, take 2M here */
        if (surfsize > TWO_MEG)
            size = TWO_MEG;
        else
            size = surfsize;

        memset(pSDMA, 0, singlePacketSize);
        pSDMA->HEADER_UNION.op           = SDMA_OP_COPY;
        pSDMA->HEADER_UNION.sub_op       = SDMA_SUBOP_COPY_LINEAR;
        pSDMA->HEADER_UNION.broadcast       = n > 1 ? 1 : 0;
        pSDMA->COUNT_UNION.count             = SDMA_COUNT(size);
        pSDMA->PARAMETER_UNION.dst_uc_sel = uc_sel_override_io_space;
        pSDMA->PARAMETER_UNION.src_uc_sel = uc_sel_override_io_space;

        SplitU64(reinterpret_cast<HSAuint64>(src),
                 pSDMA->SRC_ADDR_LO_UNION.DW_4_DATA,  // src_addr_31_0
                 pSDMA->SRC_ADDR_HI_UNION.DW_5_DATA);  // src_addr_63_32

        for (i = 0; i < n; i++)
            SplitU64(reinterpret_cast<HSAuint64>(dst[i]),
                    pSDMA->DST_ADDR[i].DST_ADDR_LO_UNION.DW_6_DATA,  // dst_addr_31_0
                    pSDMA->DST_ADDR[i].DST_ADDR_HI_UNION.DW_7_DATA);  // dst_addr_63_32

        pSDMA = reinterpret_cast<SDMA_V8_PKT_COPY_LINEAR *>(reinterpret_cast<char *>(pSDMA) + singlePacketSize);
        for (i = 0; i < n; i++)
            dst[i] = reinterpret_cast<char *>(dst[i]) + size;
        src = reinterpret_cast<char *>(src) + size;
        surfsize -= size;
    }
    free(dst);
}

SDMAFillDataPacket::SDMAFillDataPacket(unsigned int familyId, void *dst, unsigned int data, unsigned int size) {
    unsigned int copy_size;
    SDMA_PKT_CONSTANT_FILL *pSDMA;

    m_FamilyId = familyId;
    /* SDMA support maximum 0x3fffe0 byte in one copy. Use 2M copy_size */
    m_PacketSize = ((size + TWO_MEG - 1) >> BITS) * (sizeof(SDMA_PKT_CONSTANT_FILL) +
                    (familyId < FAMILY_GFX13 ? 0 : 4));
    pSDMA = reinterpret_cast<SDMA_PKT_CONSTANT_FILL *>(AllocPacket());
    memset(pSDMA, 0, m_PacketSize);
    m_PacketData = pSDMA;

    while (size > 0) {
        if (size > TWO_MEG)
            copy_size = TWO_MEG;
        else
            copy_size = size;

        pSDMA->HEADER_UNION.op = SDMA_OP_CONST_FILL;
        pSDMA->HEADER_UNION.sub_op = 0;
        if (m_FamilyId >= FAMILY_GFX125X)
            pSDMA->HEADER_UNION.scope = SDMA_SCOPE_SYS;

        /* If both size and address are DW aligned, then use DW fill */
        if (!(copy_size & 0x3) && !((HSAuint64)dst & 0x3)) {
            pSDMA->HEADER_UNION.fillsize = 2; /* DW Fill */
            pSDMA->COUNT_UNION.count = (m_FamilyId >= FAMILY_GFX12) ?
                                        (copy_size - 4) : SDMA_COUNT(copy_size);
        } else {
            pSDMA->HEADER_UNION.fillsize = 0; /* Byte Fill */
            pSDMA->COUNT_UNION.count = SDMA_COUNT(copy_size);
        }

        if (familyId >= FAMILY_GFX13)
            /* override IO space mtype to uc */
            pSDMA->COUNT_UNION.reserved_0 = uc_sel_override_io_space;

        SplitU64(reinterpret_cast<HSAuint64>(dst),
            pSDMA->DST_ADDR_LO_UNION.DW_1_DATA, /*dst_addr_31_0*/
            pSDMA->DST_ADDR_HI_UNION.DW_2_DATA); /*dst_addr_63_32*/

        pSDMA->DATA_UNION.DW_3_DATA = data;
        pSDMA++;

        if (familyId >= FAMILY_GFX13)
            pSDMA = reinterpret_cast<SDMA_PKT_CONSTANT_FILL *>(reinterpret_cast<char *>(pSDMA) + 4);

        dst = reinterpret_cast<char *>(dst) + copy_size;
        size -= copy_size;
    }
}

SDMAFencePacket::SDMAFencePacket(void) {
}

SDMAFencePacket::SDMAFencePacket(unsigned int familyId, void* destAddr, unsigned int data) {
    m_FamilyId = familyId;
    if (m_FamilyId < FAMILY_NV)
        InitPacketCI(destAddr, data);
    else
        InitPacketNV(destAddr, data);
}

SDMAFencePacket::~SDMAFencePacket(void) {
}

void SDMAFencePacket::InitPacketCI(void* destAddr, unsigned int data) {
    memset(&packetData, 0, SizeInBytes());

    packetData.HEADER_UNION.op = SDMA_OP_FENCE;

    SplitU64(reinterpret_cast<HSAuint64>(destAddr),
             packetData.ADDR_LO_UNION.DW_1_DATA, /*dst_addr_31_0*/
             packetData.ADDR_HI_UNION.DW_2_DATA); /*dst_addr_63_32*/

    packetData.DATA_UNION.data = data;
}

void SDMAFencePacket::InitPacketNV(void * destAddr,unsigned int data) {
    memset(&packetData, 0, SizeInBytes());

    /* GPA=0 becaue we use virtual address
     * Snoop = 1 because we want the write be CPU coherent
     * System = 1 because the memory is system memory
     * mtype = uncached, for the purpose of CPU coherent, L2 policy doesn't matter in this case
     */
    packetData.HEADER_UNION.DW_0_DATA = (0 << 23) | (1 << 22) | (1 << 20) | (3 << 16) | SDMA_OP_FENCE;
    if (m_FamilyId >= FAMILY_GFX125X)
        packetData.HEADER_UNION.scope = SDMA_SCOPE_SYS;

    SplitU64(reinterpret_cast<unsigned long long>(destAddr),
             packetData.ADDR_LO_UNION.DW_1_DATA, /*dst_addr_31_0*/
             packetData.ADDR_HI_UNION.DW_2_DATA); /*dst_addr_63_32*/

    packetData.DATA_UNION.data = data;
}


SDMATrapPacket::SDMATrapPacket(unsigned int eventID) {
    InitPacket(eventID);
}

SDMATrapPacket::~SDMATrapPacket(void) {
}

void SDMATrapPacket::InitPacket(unsigned int eventID) {
    memset(&packetData, 0, SizeInBytes());

    packetData.HEADER_UNION.op = SDMA_OP_TRAP;
    packetData.INT_CONTEXT_UNION.int_context = eventID;
}

SDMAPollRegMemPacket::SDMAPollRegMemPacket(void *addr, int value) {
    InitPacket(addr, value);
}

SDMAPollRegMemPacket::~SDMAPollRegMemPacket(void) {
}

void SDMAPollRegMemPacket::InitPacket(void *addr, int value) {
    memset(&packetData, 0, SizeInBytes());

    packetData.HEADER_UNION.op = SDMA_OP_POLL_REGMEM;
    packetData.HEADER_UNION.mem_poll = 1;
    packetData.HEADER_UNION.func = 0x3; // IsEqual.
    SplitU64(reinterpret_cast<unsigned long long>(addr),
             packetData.ADDR_LO_UNION.DW_1_DATA,
             packetData.ADDR_HI_UNION.DW_2_DATA);
    packetData.VALUE_UNION.value = value;
    packetData.MASK_UNION.mask = 0xffffffff; // Compare the whole content.
    packetData.DW5_UNION.interval = 0x04;
    packetData.DW5_UNION.retry_count = 0xfff;
}

SDMATimePacket::SDMATimePacket(void *destaddr) {
    InitPacket(destaddr);
}

SDMATimePacket::~SDMATimePacket(void) {
}

void SDMATimePacket::InitPacket(void *destaddr) {
    memset(&packetData, 0, SizeInBytes());

    packetData.HEADER_UNION.op = SDMA_OP_TIMESTAMP;
    packetData.HEADER_UNION.sub_op = 1 << 1; /* Get Global GPU Timestamp*/

    if (reinterpret_cast<unsigned long long>(destaddr) & 0x1f)
        WARN() << "SDMATimePacket dst address must aligned to 32bytes boundary" << std::endl;

    SplitU64(reinterpret_cast<unsigned long long>(destaddr),
            packetData.ADDR_LO_UNION.DW_1_DATA, /*dst_addr_31_0*/
            packetData.ADDR_HI_UNION.DW_2_DATA); /*dst_addr_63_32*/
}

SDMANopPacket::SDMANopPacket(unsigned int count) {
    packetSize = count * sizeof(unsigned int);
    packetData = reinterpret_cast<SDMA_PKT_NOP *>(AllocPacket());

    packetData->HEADER_UNION.op = SDMA_OP_NOP;
    packetData->HEADER_UNION.sub_op = 0;
    packetData->HEADER_UNION.count = count - 1;
}
