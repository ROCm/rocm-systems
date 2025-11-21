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

#ifndef __KFD_PM4_PACKET__H__
#define __KFD_PM4_PACKET__H__

#include "BasePacket.hpp"
#include "kfd_pm4_opcodes.h"
#include "pm4_pkt_struct_common.h"
#include "pm4_pkt_struct_ci.h"
#include "pm4_pkt_struct_ai.h"
#include "pm4_pkt_struct_nv.h"
#include "IndirectBuffer.hpp"

// @class PM4Packet: Marks a group of all PM4 packets
class PM4Packet : public BasePacket {
 public:
        PM4Packet(void): m_HeaderCountOffset(0) {}
        virtual ~PM4Packet(void) {}

        virtual PACKETTYPE PacketType() const { return PACKETTYPE_PM4; }
        unsigned int CalcCountValue() const;

 protected:
        int m_HeaderCountOffset;
        void InitPM4Header(PM4_TYPE_3_HEADER &header, it_opcode_type opCode);
};

// @class PM4WriteDataPacket
class PM4WriteDataPacket : public PM4Packet {
 public:
    // Empty constructor, before using the packet call the init func
    PM4WriteDataPacket(void): m_ndw(0), m_pPacketData(NULL) {}
    // This contructor will also init the packet, no need for additional calls
    PM4WriteDataPacket(unsigned int *destBuf, unsigned int data1):
        m_ndw(1), m_pPacketData(NULL) {InitPacket(destBuf, &data1);}
    PM4WriteDataPacket(unsigned int *destBuf, unsigned int data1, unsigned int data2):
        m_ndw(2), m_pPacketData(NULL) {
        unsigned int data[2] = {data1, data2};
        InitPacket(destBuf, data);
    }

    virtual ~PM4WriteDataPacket(void) {}
    // @returns Packet size in bytes
    virtual unsigned int SizeInBytes() const;
    // @returns Pointer to the packet
    virtual const void *GetPacket() const { return m_pPacketData; }
    // @brief Initialise the packet
    void InitPacket(unsigned int *destBuf, unsigned int data1) {
        m_ndw = 1;
        InitPacket(destBuf, &data1);
    }
    void InitPacket(unsigned int *destBuf, unsigned int data1, unsigned int data2) {
        unsigned int data[2] = {data1, data2};
        m_ndw = 2;
        InitPacket(destBuf, data);
    }
    void InitPacket(unsigned int *destBuf, void *data);

 protected:
    unsigned int m_ndw;
    // PM4WRITE_DATA_CI struct contains all the packet's data
    PM4WRITE_DATA_CI  *m_pPacketData;
};

// @class PM4ReleaseMemoryPacket
class PM4ReleaseMemoryPacket : public PM4Packet {
 public:
    // Empty constructor, before using the packet call the init func
    PM4ReleaseMemoryPacket(void): m_pPacketData(NULL) {}
    // This contructor will also init the packet, no need for additional calls
    PM4ReleaseMemoryPacket(unsigned int familyId, bool isPolling, uint64_t address, uint64_t data,
                           bool is64bit = false, bool isTimeStamp = false, int headerCountOffset = 0);

    virtual ~PM4ReleaseMemoryPacket(void) {}
    // @returns Packet size in bytes
    virtual unsigned int SizeInBytes() const { return m_packetSize; }
    // @returns Pointer to the packet
    virtual const void *GetPacket() const { return m_pPacketData; }
    // @brief Initialise the packet

 private:
    void InitPacketCI(bool isPolling, uint64_t address, uint64_t data,
                 bool is64bit = false, bool isTimeStamp = false);
    void InitPacketAI(bool isPolling, uint64_t address, uint64_t data,
                 bool is64bit = false, bool isTimeStamp = false);
    void InitPacketNV(bool isPolling, uint64_t address, uint64_t data,
                 bool is64bit = false, bool isTimeStamp = false);

    void *m_pPacketData;
    unsigned int  m_packetSize;
};

// @class PM4IndirectBufPacket
class PM4IndirectBufPacket : public PM4Packet {
 public:
    // Empty constructor, before using the packet call the init func
    PM4IndirectBufPacket(void) {}
    // This contructor will also init the packet, no need for additional calls
    explicit PM4IndirectBufPacket(IndirectBuffer *pIb);

    virtual ~PM4IndirectBufPacket(void) {}
    // @returns Packet size in bytes
    virtual unsigned int SizeInBytes() const;
    // @returns Pointer to the packet
    virtual const void *GetPacket() const { return &m_packetData; }
    // @breif Initialise the packet
    void InitPacket(IndirectBuffer *pIb);

 private:
    // PM4MEC_INDIRECT_BUFFER struct contains all the packet's data
    PM4MEC_INDIRECT_BUFFER  m_packetData;
};

// @class PM4AcquireMemoryPacket
class PM4AcquireMemoryPacket : public PM4Packet {
 public:
    PM4AcquireMemoryPacket(unsigned int familyId);
    virtual ~PM4AcquireMemoryPacket(void) {}

    // @returns the packet size in bytes
    virtual unsigned int SizeInBytes() const { return m_packetSize; }
    // @returns Pointer to the packet
    virtual const void *GetPacket() const { return m_pPacketData; }

 private:
    void InitPacketAI(void);
    void InitPacketNV(void);
    void *m_pPacketData;
    unsigned int  m_packetSize;
};

// @class PM4SetShaderRegPacket Packet that writes to consecutive registers starting at baseOffset.
class PM4SetShaderRegPacket : public PM4Packet {
 public:
    PM4SetShaderRegPacket(void);

    PM4SetShaderRegPacket(unsigned int baseOffset, const unsigned int regValues[], unsigned int numRegs);

    virtual ~PM4SetShaderRegPacket(void) {}

    // @returns Packet size in bytes
    virtual unsigned int SizeInBytes() const { return m_packetSize; }
    // @returns Pointer to the packet
    virtual const void *GetPacket() const { return m_pPacketData; }

    void InitPacket(unsigned int baseOffset, const unsigned int regValues[], unsigned int numRegs);

 private:
    unsigned int m_packetSize;
    // PM4SET_SH_REG struct contains all the packet's data
    PM4SET_SH_REG  *m_pPacketData;
};

// @class PM4DispatchDirectPacket
class PM4DispatchDirectPacket : public PM4Packet {
 public:
    PM4DispatchDirectPacket(void) {}

    PM4DispatchDirectPacket(unsigned int dimX, unsigned int dimY, unsigned int dimZ, unsigned int dispatchInit);

    virtual ~PM4DispatchDirectPacket(void) {}

    // @returns Packet size in bytes
    virtual unsigned int SizeInBytes() const;
    // @returns Pointer to the packet
    virtual const void *GetPacket() const { return &m_packetData; }

    void InitPacket(unsigned int dimX, unsigned int dimY, unsigned int dimZ, unsigned int dispatchInit);

 private:
    // PM4DISPATCH_DIRECT struct contains all the packet's data
    PM4DISPATCH_DIRECT  m_packetData;
};

// @class PM4PartialFlushPacket
class PM4PartialFlushPacket : public PM4Packet {
 public:
    PM4PartialFlushPacket(void);
    virtual ~PM4PartialFlushPacket(void) {}

    // @returns Packet size in bytes
    virtual unsigned int SizeInBytes() const;
    // @returns Pointer to the packet
    virtual const void *GetPacket() const { return &m_packetData; }

 private:
    // PM4EVENT_WRITE struct contains all the packet's data
    PM4EVENT_WRITE  m_packetData;
};

// @class PM4NopPacket
class PM4NopPacket : public PM4Packet {
 public:
    PM4NopPacket(unsigned int count = 1);
    virtual ~PM4NopPacket(void) {}

    // @returns Packet size in bytes
    virtual unsigned int SizeInBytes() const { return m_packetSize; }
    // @returns Pointer to the packet
    virtual const void *GetPacket() const { return m_packetData; }

 private:
    unsigned int m_packetSize;
    PM4_TYPE_3_HEADER *m_packetData;
};

// @class PM4WaitRegMemPacket
class PM4WaitRegMemPacket : public PM4Packet {
 public:
    PM4WaitRegMemPacket(void) {}
    PM4WaitRegMemPacket(bool memory, uint64_t addr, uint32_t ref, uint16_t pollInterval);
    PM4WaitRegMemPacket(unsigned int function, unsigned int space, unsigned int operation,
                        uint64_t addr, uint32_t ref, uint32_t mask, uint16_t pollInterval);
    virtual ~PM4WaitRegMemPacket(void) {}

    // @returns Packet size in bytes
    virtual unsigned int SizeInBytes() const;
    // @returns Pointer to the packet
    virtual const void *GetPacket() const { return &m_packetData; }

    void InitPacket(unsigned int function, unsigned int space, unsigned int operation,
                    uint64_t addr, uint32_t ref, uint32_t mask, uint16_t pollInterval);

 private:
    PM4MEC_WAIT_REG_MEM m_packetData;
};

#endif  // __KFD_PM4_PACKET__H__
