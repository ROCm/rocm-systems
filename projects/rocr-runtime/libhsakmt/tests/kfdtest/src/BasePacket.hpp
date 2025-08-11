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

#ifndef __KFD_BASE_PACKET__H__
#define __KFD_BASE_PACKET__H__

/**
 * All packets profiles must be defined here
 * Every type defined here has sub-types
 */
enum PACKETTYPE {
    PACKETTYPE_PM4,
    PACKETTYPE_SDMA,
    PACKETTYPE_AQL
};

// @class BasePacket
class BasePacket {
 public:
    BasePacket(void);
    virtual ~BasePacket(void);

    // @returns Packet type
    virtual PACKETTYPE PacketType() const = 0;
    // @returns Pointer to the packet
    virtual const void *GetPacket() const = 0;
    // @returns Packet size in bytes
    virtual unsigned int SizeInBytes() const = 0;
    // @returns Packet size in dwordS
    unsigned int SizeInDWords() const { return SizeInBytes()/sizeof(unsigned int); }

    void Dump() const;

 protected:
    unsigned int m_FamilyId;
    void *m_packetAllocation;

    void *AllocPacket(void);
};

#endif  // __KFD_BASE_PACKET__H__
