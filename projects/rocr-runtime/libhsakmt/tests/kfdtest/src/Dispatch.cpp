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

#include "Dispatch.hpp"

#include "PM4Packet.hpp"

#include "asic_reg/gfx_7_2_d.h"
#include "asic_reg/gfx_7_2_sh_mask.h"

#include "KFDBaseComponentTest.hpp"

#define mmCOMPUTE_PGM_RSRC3                                                     0x2e2d

Dispatch::Dispatch(const HsaMemoryBuffer& isaBuf, const bool eventAutoReset)
    :m_IsaBuf(isaBuf), m_IndirectBuf(PACKETTYPE_PM4, PAGE_SIZE / sizeof(unsigned int), isaBuf.Node()),
    m_DimX(1), m_DimY(1), m_DimZ(1), m_pArg1(NULL), m_pArg2(NULL), m_pEop(NULL), m_ScratchEn(false),
    m_ComputeTmpringSize(0), m_scratch_base(0ll), m_SpiPriority(0) {
    HsaEventDescriptor eventDesc;
    eventDesc.EventType = HSA_EVENTTYPE_SIGNAL;
    eventDesc.NodeId = isaBuf.Node();
    eventDesc.SyncVar.SyncVar.UserData = NULL;
    eventDesc.SyncVar.SyncVarSize = 0;

    hsaKmtCreateEvent(&eventDesc, !eventAutoReset, false, &m_pEop);

    m_FamilyId  = g_baseTest->GetFamilyIdFromNodeId(isaBuf.Node());
    m_NeedCwsrWA = g_baseTest->NeedCwsrWA(isaBuf.Node());
}

Dispatch::~Dispatch() {
    if (m_pEop != NULL)
        hsaKmtDestroyEvent(m_pEop);
}

void Dispatch::SetArgs(void* pArg1, void* pArg2) {
    m_pArg1 = pArg1;
    m_pArg2 = pArg2;
}

void Dispatch::SetDim(unsigned int x, unsigned int y, unsigned int z) {
    m_DimX = x;
    m_DimY = y;
    m_DimZ = z;
}

void Dispatch::SetScratch(int numWaves, int waveSize, HSAuint64 scratch_base) {
    m_ComputeTmpringSize = ((waveSize << 12) | (numWaves));
    m_ScratchEn = true;
    m_scratch_base = scratch_base;
}

void Dispatch::SetSpiPriority(unsigned int priority) {
    m_SpiPriority = priority;
}

void Dispatch::SetPriv(bool priv) {
    m_NeedCwsrWA = priv;
}

void Dispatch::Submit(BaseQueue& queue) {
    ASSERT_NE(m_pEop, (void*)0);
    EXPECT_EQ(m_FamilyId, queue.GetFamilyId());

    BuildIb();

    queue.PlaceAndSubmitPacket(PM4IndirectBufPacket(&m_IndirectBuf));

    // Write data to SyncVar for synchronization purpose
    if (m_pEop->EventData.EventData.SyncVar.SyncVar.UserData != NULL) {
        queue.PlaceAndSubmitPacket(PM4WriteDataPacket((unsigned int*)m_pEop->
            EventData.EventData.SyncVar.SyncVar.UserData, m_pEop->EventId));
    }

    queue.PlaceAndSubmitPacket(PM4ReleaseMemoryPacket(m_FamilyId, false, m_pEop->EventData.HWData2, m_pEop->EventId));

    if (!queue.GetSkipWaitConsump())
        queue.Wait4PacketConsumption();
}

void Dispatch::Sync(unsigned int timeout) {
    ASSERT_SUCCESS(hsaKmtWaitOnEvent(m_pEop, timeout));
}

// Returning with status in order to allow actions to be performed before process termination
int Dispatch::SyncWithStatus(unsigned int timeout) {
    int stat;

    return ((stat = hsaKmtWaitOnEvent(m_pEop, timeout)) != HSAKMT_STATUS_SUCCESS);
}

void Dispatch::BuildIb() {
    HSAuint64 shiftedIsaAddr = m_IsaBuf.As<uint64_t>() >> 8;
    unsigned int arg0, arg1, arg2, arg3;
    SplitU64(reinterpret_cast<uint64_t>(m_pArg1), arg0, arg1);
    SplitU64(reinterpret_cast<uint64_t>(m_pArg2), arg2, arg3);

    // Starts at COMPUTE_START_X
    const unsigned int COMPUTE_DISPATCH_DIMS_VALUES[] = {
        0,      // START_X
        0,      // START_Y
        0,      // START_Z
        1,      // NUM_THREADS_X - this is actually the number of threads in a thread group
        1,      // NUM_THREADS_Y
        1,      // NUM_THREADS_Z
        0,      // COMPUTE_PIPELINESTAT_ENABLE
        0,      // COMPUTE_PERFCOUNT_ENABLE
    };

    /*
     * For some special asics in the list of DEGFX11_12113
     * COMPUTE_PGM_RSRC needs priv=1 to prevent hardware traps
     */
    const bool priv = m_NeedCwsrWA;

    unsigned int pgmRsrc1 =
        (0xc0 << COMPUTE_PGM_RSRC1__FLOAT_MODE__SHIFT) |
        ((m_SpiPriority & 3) << COMPUTE_PGM_RSRC1__PRIORITY__SHIFT) |
        (priv << COMPUTE_PGM_RSRC1__PRIV__SHIFT) |
        ((m_FamilyId < FAMILY_GFX12) ? (0x2 << COMPUTE_PGM_RSRC1__SGPRS__SHIFT) : 0) |
        (0x4 << COMPUTE_PGM_RSRC1__VGPRS__SHIFT);  // 4 * 8 = 32 VGPRs

    unsigned int pgmRsrc2 = 0;
    pgmRsrc2 |= (m_ScratchEn << COMPUTE_PGM_RSRC2__SCRATCH_EN__SHIFT)
            & COMPUTE_PGM_RSRC2__SCRATCH_EN_MASK;
    pgmRsrc2 |= ((m_scratch_base ? 6 : 4) << COMPUTE_PGM_RSRC2__USER_SGPR__SHIFT)
            & COMPUTE_PGM_RSRC2__USER_SGPR_MASK;

    if (m_FamilyId < FAMILY_GFX12) {
        pgmRsrc2 |= (1 << COMPUTE_PGM_RSRC2__TRAP_PRESENT__SHIFT)
            & COMPUTE_PGM_RSRC2__TRAP_PRESENT_MASK;
    }

    pgmRsrc2 |= (1 << COMPUTE_PGM_RSRC2__TGID_X_EN__SHIFT)
            & COMPUTE_PGM_RSRC2__TGID_X_EN_MASK;
    pgmRsrc2 |= (1 << COMPUTE_PGM_RSRC2__TIDIG_COMP_CNT__SHIFT)
            & COMPUTE_PGM_RSRC2__TIDIG_COMP_CNT_MASK;
    pgmRsrc2 |= (0 << COMPUTE_PGM_RSRC2__EXCP_EN__SHIFT)
            & COMPUTE_PGM_RSRC2__EXCP_EN_MASK;
    pgmRsrc2 |= (1 << COMPUTE_PGM_RSRC2__EXCP_EN_MSB__SHIFT)
            & COMPUTE_PGM_RSRC2__EXCP_EN_MSB_MASK;

    const unsigned int COMPUTE_PGM_RSRC[] = {
        pgmRsrc1,
        pgmRsrc2
    };

    // Starts at COMPUTE_PGM_LO
    const unsigned int COMPUTE_PGM_VALUES_GFX8[] = {
        static_cast<uint32_t>(shiftedIsaAddr),                  // PGM_LO
        static_cast<uint32_t>(shiftedIsaAddr >> 32)             // PGM_HI
            | (hsakmt_is_dgpu() ? 0 : (1<<8))                          // including PGM_ATC=?
    };

    // Starts at COMPUTE_PGM_LO
    const unsigned int COMPUTE_PGM_VALUES_GFX9[] = {
        static_cast<uint32_t>(shiftedIsaAddr),                  // PGM_LO
        static_cast<uint32_t>(shiftedIsaAddr >> 32)             // PGM_HI
            | (hsakmt_is_dgpu() ? 0 : (1<<8)),                         // including PGM_ATC=?
        0,
        0,
        static_cast<uint32_t>(m_scratch_base >> 8),              // compute_dispatch_scratch_base
        static_cast<uint32_t>(m_scratch_base >> 40)
    };

    // Starts at COMPUTE_RESOURCE_LIMITS
    const unsigned int COMPUTE_RESOURCE_LIMITS[] = {
        0,                      // COMPUTE_RESOURCE_LIMITS
    };

    // Starts at COMPUTE_TMPRING_SIZE
    const unsigned int COMPUTE_TMPRING_SIZE[] = {
        m_ComputeTmpringSize,   // COMPUTE_TMPRING_SIZE
    };

    // Starts at COMPUTE_RESTART_X
    const unsigned int COMPUTE_RESTART_VALUES[] = {
        0,                      // COMPUTE_RESTART_X
        0,                      // COMPUTE_RESTART_Y
        0,                      // COMPUTE_RESTART_Z
        0                       // COMPUTE_THREAD_TRACE_ENABLE
    };

    // Starts at COMPUTE_USER_DATA_0
    const unsigned int COMPUTE_USER_DATA_VALUES[] = {
                // Reg name             - use in KFDtest - use in ABI
        arg0,   // COMPUTE_USER_DATA_0  - arg0           - resource descriptor for the scratch buffer - 1st dword
        arg1,   // COMPUTE_USER_DATA_1  - arg1           - resource descriptor for the scratch buffer - 2nd dword
        arg2,   // COMPUTE_USER_DATA_2  - arg2           - resource descriptor for the scratch buffer - 3rd dword
        arg3,   // COMPUTE_USER_DATA_3  - arg3           - resource descriptor for the scratch buffer - 4th dword
        static_cast<uint32_t>(m_scratch_base),  // COMPUTE_USER_DATA_4  - flat_scratch_lo
        static_cast<uint32_t>(m_scratch_base >> 32),  // COMPUTE_USER_DATA_4  - flat_scratch_hi
        0,      // COMPUTE_USER_DATA_6  -                - AQL queue address, low part
        0,      // COMPUTE_USER_DATA_7  -                - AQL queue address, high part
        0,      // COMPUTE_USER_DATA_8  -                - kernel arguments block, low part
        0,      // COMPUTE_USER_DATA_9  -                - kernel arguments block, high part
        0,      // COMPUTE_USER_DATA_10 -                - unused
        0,      // COMPUTE_USER_DATA_11 -                - unused
        0,      // COMPUTE_USER_DATA_12 -                - unused
        0,      // COMPUTE_USER_DATA_13 -                - unused
        0,      // COMPUTE_USER_DATA_14 -                - unused
        0,      // COMPUTE_USER_DATA_15 -                - unused
    };

    const unsigned int DISPATCH_INIT_VALUE = 0x00000021 | (hsakmt_is_dgpu() ? 0 : 0x1000) |
                ((m_FamilyId >= FAMILY_NV) ? 0x8000 : 0);
    // {COMPUTE_SHADER_EN=1, PARTIAL_TG_EN=0, FORCE_START_AT_000=0, ORDERED_APPEND_ENBL=0,
    // ORDERED_APPEND_MODE=0, USE_THREAD_DIMENSIONS=1, ORDER_MODE=0, DISPATCH_CACHE_CNTL=0,
    // SCALAR_L1_INV_VOL=0, VECTOR_L1_INV_VOL=0, DATA_ATC=?, RESTORE=0}
    // Set CS_W32_EN for wave32 workloads for gfx10 since all the shaders used in KFDTest is 32 bit .

    m_IndirectBuf.AddPacket(PM4AcquireMemoryPacket(m_FamilyId));

    m_IndirectBuf.AddPacket(PM4SetShaderRegPacket(mmCOMPUTE_START_X, COMPUTE_DISPATCH_DIMS_VALUES,
                                                  ARRAY_SIZE(COMPUTE_DISPATCH_DIMS_VALUES)));

    m_IndirectBuf.AddPacket(PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO,
        (m_FamilyId >= FAMILY_AI) ? COMPUTE_PGM_VALUES_GFX9 : COMPUTE_PGM_VALUES_GFX8,
        (m_FamilyId >= FAMILY_AI) ? ARRAY_SIZE(COMPUTE_PGM_VALUES_GFX9) : ARRAY_SIZE(COMPUTE_PGM_VALUES_GFX8)));
    m_IndirectBuf.AddPacket(PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, COMPUTE_PGM_RSRC,
                                                  ARRAY_SIZE(COMPUTE_PGM_RSRC)));

    if (m_FamilyId == FAMILY_AL || m_FamilyId == FAMILY_AV) {
        const unsigned int COMPUTE_PGM_RSRC3[] = {9};
        m_IndirectBuf.AddPacket(PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC3, COMPUTE_PGM_RSRC3,
                                                      ARRAY_SIZE(COMPUTE_PGM_RSRC3)));
    }

    m_IndirectBuf.AddPacket(PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, COMPUTE_RESOURCE_LIMITS,
                                                  ARRAY_SIZE(COMPUTE_RESOURCE_LIMITS)));
    m_IndirectBuf.AddPacket(PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, COMPUTE_TMPRING_SIZE,
                                                  ARRAY_SIZE(COMPUTE_TMPRING_SIZE)));
    m_IndirectBuf.AddPacket(PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, COMPUTE_RESTART_VALUES,
                                                  ARRAY_SIZE(COMPUTE_RESTART_VALUES)));

    m_IndirectBuf.AddPacket(PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, COMPUTE_USER_DATA_VALUES,
                                                  ARRAY_SIZE(COMPUTE_USER_DATA_VALUES)));

    m_IndirectBuf.AddPacket(PM4DispatchDirectPacket(m_DimX, m_DimY, m_DimZ, DISPATCH_INIT_VALUE));

    // EVENT_WRITE.partial_flush causes problems with preemptions in
    // GWS testing. Since this is specific to this PM4 command and
    // doesn't affect AQL, it's easier to fix KFDTest than the
    // firmware.
    //
    // Replace PartialFlush with an ReleaseMem (with no interrupt) + WaitRegMem
    //
    // Original: m_IndirectBuf.AddPacket(PM4PartialFlushPacket());
    uint32_t *nop = m_IndirectBuf.AddPacket(PM4NopPacket(2)); // NOP packet with one dword payload for the release-mem fence
    m_IndirectBuf.AddPacket(PM4ReleaseMemoryPacket(m_FamilyId, true, (uint64_t)&nop[1], 0xdeadbeef));
    m_IndirectBuf.AddPacket(PM4WaitRegMemPacket(true, (uint64_t)&nop[1], 0xdeadbeef, 4));
}
