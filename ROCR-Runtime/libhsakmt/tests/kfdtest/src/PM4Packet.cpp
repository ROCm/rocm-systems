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
#include "PM4Packet.hpp"
#include "hsakmt/hsakmttypes.h"
#include "KFDBaseComponentTest.hpp"

#include "asic_reg/gfx_7_2_enum.h"

unsigned int PM4Packet::CalcCountValue() const {
    return (SizeInDWords() - (sizeof(PM4_TYPE_3_HEADER) / sizeof(uint32_t)) - 1);
}

void PM4Packet::InitPM4Header(PM4_TYPE_3_HEADER &header, it_opcode_type opCode) {
    header.count                    = CalcCountValue() + m_HeaderCountOffset;
    header.opcode                 = opCode;
    header.type                      = PM4_TYPE_3;
    header.shaderType          = 1;  // compute
    header.predicate              = 0;
    header.reserved1             = 0;
}

unsigned int PM4WriteDataPacket::SizeInBytes() const {
    return (offsetof(PM4WRITE_DATA_CI, data) + m_ndw*sizeof(uint32_t));
}

void PM4WriteDataPacket::InitPacket(unsigned int *destBuf, void *data) {
    m_pPacketData = reinterpret_cast<PM4WRITE_DATA_CI *>(AllocPacket());

    InitPM4Header(m_pPacketData->header, IT_WRITE_DATA);

    m_pPacketData->bitfields2.dst_sel      = dst_sel_mec_write_data_MEMORY_5;  // memory-async
    m_pPacketData->bitfields2.addr_incr    = addr_incr_mec_write_data_INCREMENT_ADDR_0;  // increment addr
    m_pPacketData->bitfields2.wr_confirm   = wr_confirm_mec_write_data_WAIT_FOR_CONFIRMATION_1;
    m_pPacketData->bitfields2.atc          = hsakmt_is_dgpu() ?
        atc_write_data_NOT_USE_ATC_0 : atc_write_data_USE_ATC_1;
    m_pPacketData->bitfields2.cache_policy = cache_policy_mec_write_data_BYPASS_2;

    m_pPacketData->dst_addr_lo    = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(destBuf));  // byte addr
    m_pPacketData->dst_address_hi = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(destBuf) >> 32);

    memcpy(m_pPacketData->data, data, m_ndw * sizeof(uint32_t));
}

PM4ReleaseMemoryPacket::PM4ReleaseMemoryPacket(unsigned int familyId, bool isPolling,
                    uint64_t address, uint64_t data, bool is64bit, bool isTimeStamp,
                    int headerCountOffset):m_pPacketData(NULL) {
    m_FamilyId = familyId;
    m_HeaderCountOffset = headerCountOffset;
    if (familyId < FAMILY_AI)
        InitPacketCI(isPolling, address, data, is64bit, isTimeStamp);
    else if (familyId < FAMILY_NV)
        InitPacketAI(isPolling, address, data, is64bit, isTimeStamp);
    else
        InitPacketNV(isPolling, address, data, is64bit, isTimeStamp);
}

void PM4ReleaseMemoryPacket::InitPacketCI(bool isPolling, uint64_t address,
                                    uint64_t data, bool is64bit, bool isTimeStamp) {
    PM4_RELEASE_MEM_CI *pkt;

    m_packetSize = sizeof(PM4_RELEASE_MEM_CI);
    pkt = reinterpret_cast<PM4_RELEASE_MEM_CI *>(AllocPacket());
    m_pPacketData = pkt;

    InitPM4Header(pkt->header, IT_RELEASE_MEM);

    pkt->bitfields2.event_type       = 0x14;
    pkt->bitfields2.event_index      = event_index_mec_release_mem_EVENT_WRITE_EOP_5;
                        // Possible values:
                        // 0101(5): EVENT_WRITE_EOP event types
                        // 0110(6): Reserved for EVENT_WRITE_EOS packet.
                        // 0111(7): Reserved (previously) for EVENT_WRITE packet.
    pkt->bitfields2.l2_wb            = 1;
    pkt->bitfields2.l2_inv           = 1;
    pkt->bitfields2.cache_policy     = cache_policy_mec_release_mem_BYPASS_2;
    pkt->bitfields2.atc = hsakmt_is_dgpu() ?
                    atc_mec_release_mem_ci_NOT_USE_ATC_0 :
                    atc_mec_release_mem_ci_USE_ATC_1;  // ATC setting for fences and timestamps to the MC or TCL2.
    pkt->bitfields3.dst_sel          = dst_sel_mec_release_mem_MEMORY_CONTROLLER_0;
                        // Possible values:
                        // 0 - memory_controller.
                        // 1 - tc_l2.
    if (address) {
        pkt->bitfields3.int_sel      = (isPolling ?
                    int_sel_mec_release_mem_SEND_DATA_AFTER_WRITE_CONFIRM_3 :
                    int_sel_mec_release_mem_SEND_INTERRUPT_AFTER_WRITE_CONFIRM_2);
                // Possible values:
                // 0 - None (Do not send an interrupt).
                // 1 - Send Interrupt Only. Program DATA_SEL 0".
                // 2 - Send Interrupt when Write Confirm (WC) is received from the MC.
                // 3 - Wait for WC, but dont send interrupt (applicable to 7.3+) [g73_1]
                // 4 - Reserved for INTERRUPT packet
        if (isTimeStamp && is64bit)
            pkt->bitfields3.data_sel = data_sel_mec_release_mem_SEND_GPU_CLOCK_COUNTER_3;
        else
            pkt->bitfields3.data_sel     = is64bit ?
                        data_sel_mec_release_mem_SEND_64_BIT_DATA_2 :
                        data_sel_mec_release_mem_SEND_32_BIT_LOW_1;
                    // Possible values:
                    // 0 - None, i.e., Discard Data.
                    // 1 - Send 32-bit Data Low (Discard Data High).
                    // 2 - Send 64-bit Data.
                    // 3 - Send current value of the 64 bit global GPU clock counter.
                    // 4 - Send current value of the 64 bit system clock counter.
                    // 5 - Store GDS Data to memory.
                    // 6 - Reserved for use by the CP for Signal Semaphore.
                    // 7 - Reserved for use by the CP for Wait Semaphore.
    } else {
        pkt->bitfields3.int_sel      = (isPolling ?
                    int_sel_mec_release_mem_NONE_0 :
                    int_sel_mec_release_mem_SEND_INTERRUPT_ONLY_1);
        pkt->bitfields3.data_sel     = data_sel_mec_release_mem_NONE_0;
    }

    pkt->bitfields4a.address_lo_dword_aligned = static_cast<uint32_t>((address&0xffffffff) >> 2);
    pkt->addr_hi = static_cast<uint32_t>(address>>32);

    pkt->data_lo = static_cast<uint32_t>(data);
    pkt->data_hi = static_cast<uint32_t>(data >> 32);
}
void PM4ReleaseMemoryPacket::InitPacketAI(bool isPolling, uint64_t address,
                                        uint64_t data, bool is64bit, bool isTimeStamp) {
    PM4MEC_RELEASE_MEM_AI *pkt;

    m_packetSize = sizeof(PM4MEC_RELEASE_MEM_AI);
    pkt = reinterpret_cast<PM4MEC_RELEASE_MEM_AI *>(AllocPacket());
    m_pPacketData = pkt;

    InitPM4Header(pkt->header, IT_RELEASE_MEM);

    pkt->bitfields2.event_type       = 0x14;
    pkt->bitfields2.event_index      = event_index__mec_release_mem__end_of_pipe;
    pkt->bitfields2.tc_wb_action_ena = 1;
    pkt->bitfields2.tc_action_ena    = 1;
    pkt->bitfields2.cache_policy     = cache_policy__mec_release_mem__lru;

    pkt->bitfields3.dst_sel          = dst_sel__mec_release_mem__memory_controller;

    if (address) {
        pkt->bitfields3.int_sel  = (isPolling ?
                int_sel__mec_release_mem__send_data_after_write_confirm:
                int_sel__mec_release_mem__send_interrupt_after_write_confirm);

        if (isTimeStamp && is64bit)
            pkt->bitfields3.data_sel = data_sel__mec_release_mem__send_gpu_clock_counter;
        else
            pkt->bitfields3.data_sel     = is64bit ?
                    data_sel__mec_release_mem__send_64_bit_data :
                    data_sel__mec_release_mem__send_32_bit_low;
    } else {
        pkt->bitfields3.int_sel  = (isPolling ?
                int_sel__mec_release_mem__none:
                int_sel__mec_release_mem__send_interrupt_only);
        pkt->bitfields3.data_sel     = data_sel__mec_release_mem__none;
    }

    pkt->bitfields4a.address_lo_32b = static_cast<uint32_t>((address&0xffffffff) >> 2);
    pkt->address_hi = static_cast<uint32_t>(address>>32);

    pkt->data_lo = static_cast<uint32_t>(data);
    pkt->data_hi = static_cast<uint32_t>(data >> 32);

    pkt->int_ctxid = static_cast<uint32_t>(data);
}

void PM4ReleaseMemoryPacket::InitPacketNV(bool isPolling, uint64_t address,
                                uint64_t data, bool is64bit, bool isTimeStamp) {
    PM4MEC_RELEASE_MEM_NV *pkt;

    m_packetSize = sizeof(PM4_MEC_RELEASE_MEM_NV);
    pkt = reinterpret_cast<PM4_MEC_RELEASE_MEM_NV *>(AllocPacket());
    m_pPacketData = pkt;

    InitPM4Header(pkt->header, IT_RELEASE_MEM);

    pkt->bitfields2.event_type       = 0x14;
    pkt->bitfields2.event_index      = event_index__mec_release_mem__end_of_pipe;
    pkt->bitfields2.gcr_cntl         = (1<<10) | (1<<9) | (1<<8) | (1<<3) | (1<<2);
    pkt->bitfields2.cache_policy     = cache_policy__mec_release_mem__lru;

    pkt->bitfields3.dst_sel          = dst_sel__mec_release_mem__memory_controller;

    if (address) {
        pkt->bitfields3.int_sel  = (isPolling ?
                int_sel__mec_release_mem__send_data_after_write_confirm:
                int_sel__mec_release_mem__send_interrupt_after_write_confirm);

        if (isTimeStamp && is64bit)
            pkt->bitfields3.data_sel = data_sel__mec_release_mem__send_gpu_clock_counter;
        else
            pkt->bitfields3.data_sel     = is64bit ?
                    data_sel__mec_release_mem__send_64_bit_data :
                    data_sel__mec_release_mem__send_32_bit_low;
    } else {
        pkt->bitfields3.int_sel  = (isPolling ?
                int_sel__mec_release_mem__none:
                int_sel__mec_release_mem__send_interrupt_only);
        pkt->bitfields3.data_sel     = data_sel__mec_release_mem__none;
    }

    pkt->bitfields4a.address_lo_32b = static_cast<uint32_t>((address&0xffffffff) >> 2);
    pkt->address_hi = static_cast<uint32_t>(address>>32);

    pkt->data_lo = static_cast<uint32_t>(data);
    pkt->data_hi = static_cast<uint32_t>(data >> 32);

    pkt->int_ctxid = static_cast<uint32_t>(data);
}

PM4IndirectBufPacket::PM4IndirectBufPacket(IndirectBuffer *pIb) {
    InitPacket(pIb);
}

unsigned int PM4IndirectBufPacket::SizeInBytes() const {
    return sizeof(PM4MEC_INDIRECT_BUFFER);
}

void PM4IndirectBufPacket::InitPacket(IndirectBuffer *pIb) {
    memset(&m_packetData, 0, SizeInBytes());
    InitPM4Header(m_packetData.header,  IT_INDIRECT_BUFFER);

    m_packetData.bitfields2.ib_base_lo = static_cast<HSAuint32>((reinterpret_cast<HSAuint64>(pIb->Addr()))) >> 2;
    m_packetData.bitfields3.ib_base_hi = reinterpret_cast<HSAuint64>(pIb->Addr()) >> 32;
    m_packetData.bitfields4.ib_size          = pIb->SizeInDWord();
    m_packetData.bitfields4.chain            = 0;
    m_packetData.bitfields4.offload_polling  = 0;
    m_packetData.bitfields4.volatile_setting = 0;
    m_packetData.bitfields4.valid            = 1;
    m_packetData.bitfields4.vmid             = 0;  // in iommutest:  vmid = queueParams.VMID;
    m_packetData.bitfields4.cache_policy     = cache_policy_indirect_buffer_BYPASS_2;
}
PM4AcquireMemoryPacket::PM4AcquireMemoryPacket(unsigned int familyId):m_pPacketData(NULL)
{
    m_FamilyId = familyId;

    if (familyId < FAMILY_NV)
        InitPacketAI();
    else
        InitPacketNV();
}

void PM4AcquireMemoryPacket::InitPacketAI(void) {

    PM4ACQUIRE_MEM *pkt;
    m_packetSize = sizeof(PM4ACQUIRE_MEM);
    pkt = reinterpret_cast<PM4ACQUIRE_MEM*>(AllocPacket());
    m_pPacketData = pkt;

    InitPM4Header(pkt->header,  IT_ACQUIRE_MEM);
    pkt->bitfields2.coher_cntl     = 0x28c00000;  // copied from the way the HSART does this.
    pkt->bitfields2.engine         = engine_acquire_mem_PFP_0;
    pkt->coher_size                = 0xFFFFFFFF;
    pkt->bitfields3.coher_size_hi  = 0;
    pkt->coher_base_lo             = 0;
    pkt->bitfields4.coher_base_hi  = 0;
    pkt->bitfields5.poll_interval  = 4;  // copied from the way the HSART does this.
}
void PM4AcquireMemoryPacket::InitPacketNV(void) {
    PM4ACQUIRE_MEM_NV *pkt;
    m_packetSize = sizeof(PM4ACQUIRE_MEM_NV);
    pkt = reinterpret_cast<PM4ACQUIRE_MEM_NV*>(AllocPacket());
    m_pPacketData = pkt;

    InitPM4Header(pkt->header,  IT_ACQUIRE_MEM);
    pkt->coher_size                = 0xFFFFFFFF;
    pkt->bitfields3.coher_size_hi  = 0;
    pkt->coher_base_lo             = 0;
    pkt->bitfields4.coher_base_hi  = 0;
    pkt->bitfields5.poll_interval  = 4; //copied from the way the HSART does this.
    /* Invalidate gL2, gL1 with range base
          * Invalidate GLV, GLK (L0$)
          * Invalidate all Icache (GLI)
          */
    pkt->bitfields6.gcr_cntl = (1<<14|1<<9|1<<8|1<<7|1);
}

PM4SetShaderRegPacket::PM4SetShaderRegPacket(void) {
}

PM4SetShaderRegPacket::PM4SetShaderRegPacket(unsigned int baseOffset, const unsigned int regValues[],
                                             unsigned int numRegs) {
    InitPacket(baseOffset, regValues, numRegs);
}

void PM4SetShaderRegPacket::InitPacket(unsigned int baseOffset, const unsigned int regValues[],
                                       unsigned int numRegs) {
    // 1st register is a part of the packet struct.
    m_packetSize = sizeof(PM4SET_SH_REG) + (numRegs-1)*sizeof(uint32_t);

    /* Allocating the size of the packet, since the packet is assembled from a struct
     * followed by an additional dword data
     */
    m_pPacketData = reinterpret_cast<PM4SET_SH_REG *>(AllocPacket());

    memset(m_pPacketData, 0, m_packetSize);

    InitPM4Header(m_pPacketData->header,  IT_SET_SH_REG);

    m_pPacketData->bitfields2.reg_offset = baseOffset - PERSISTENT_SPACE_START;

    memcpy(m_pPacketData->reg_data, regValues, numRegs*sizeof(uint32_t));
}

PM4DispatchDirectPacket::PM4DispatchDirectPacket(unsigned int dimX, unsigned int dimY,
                                                 unsigned int dimZ, unsigned int dispatchInit) {
    InitPacket(dimX, dimY, dimZ, dispatchInit);
}

void PM4DispatchDirectPacket::InitPacket(unsigned int dimX, unsigned int dimY, unsigned int dimZ,
                                         unsigned int dispatchInit) {
    memset(&m_packetData, 0, SizeInBytes());
    InitPM4Header(m_packetData.header, IT_DISPATCH_DIRECT);

    m_packetData.dim_x = dimX;
    m_packetData.dim_y = dimY;
    m_packetData.dim_z = dimZ;
    m_packetData.dispatch_initiator = dispatchInit;
}

unsigned int PM4DispatchDirectPacket::SizeInBytes() const {
    return sizeof(PM4DISPATCH_DIRECT);
}

PM4PartialFlushPacket::PM4PartialFlushPacket(void) {
    memset(&m_packetData, 0, SizeInBytes());
    InitPM4Header(m_packetData.header, IT_EVENT_WRITE);

    m_packetData.bitfields2.event_index = event_index_event_write_CS_VS_PS_PARTIAL_FLUSH_4;
    m_packetData.bitfields2.event_type = CS_PARTIAL_FLUSH;
}

unsigned int PM4PartialFlushPacket::SizeInBytes() const {
    // For PARTIAL_FLUSH_CS packets, the last 2 dwordS don't exist.
    return sizeof(PM4EVENT_WRITE) - sizeof(uint32_t)*2;
}

PM4NopPacket::PM4NopPacket(unsigned int count): m_packetSize(count * 4) {
    m_packetData = reinterpret_cast<PM4_TYPE_3_HEADER *>(AllocPacket());
    InitPM4Header(*m_packetData, IT_NOP);
}

PM4WaitRegMemPacket::PM4WaitRegMemPacket(bool memory, uint64_t addr,
                                         uint32_t ref, uint16_t pollInterval) {
    InitPacket(function__mec_wait_reg_mem__equal_to_the_reference_value,
               memory ?
               mem_space__mec_wait_reg_mem__memory_space :
               mem_space__mec_wait_reg_mem__register_space,
               operation__mec_wait_reg_mem__wait_reg_mem,
               addr, ref, 0xffffffff, pollInterval);
}
PM4WaitRegMemPacket::PM4WaitRegMemPacket(unsigned int function,
                                         unsigned int space,
                                         unsigned int operation,
                                         uint64_t addr, uint32_t ref,
                                         uint32_t mask, uint16_t pollInterval) {
    InitPacket(function, space, operation, addr, ref, mask, pollInterval);
}

void PM4WaitRegMemPacket::InitPacket(unsigned int function,
                                     unsigned int space,
                                     unsigned int operation,
                                     uint64_t addr, uint32_t ref,
                                     uint32_t mask, uint16_t pollInterval) {
    memset(&m_packetData, 0, SizeInBytes());
    InitPM4Header(m_packetData.header, IT_WAIT_REG_MEM);

    m_packetData.bitfields2.function = (MEC_WAIT_REG_MEM_function_enum)function;
    m_packetData.bitfields2.mem_space = (MEC_WAIT_REG_MEM_mem_space_enum)space;
    m_packetData.bitfields2.operation = (MEC_WAIT_REG_MEM_operation_enum)operation;

    m_packetData.ordinal3 = addr;
    m_packetData.mem_poll_addr_hi = addr >> 32;

    m_packetData.reference = ref;
    m_packetData.mask = mask;

    m_packetData.bitfields7.poll_interval = pollInterval;
    m_packetData.bitfields7.optimize_ace_offload_mode = 1;
}

unsigned int PM4WaitRegMemPacket::SizeInBytes() const {
    return sizeof(m_packetData);
}
