
// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef _GFX1250_BLOCKTABLE_H_
#define _GFX1250_BLOCKTABLE_H_

#include "gfxip/gfx12/gfx12_reg_info_macros.h"

namespace gfxip {
namespace gfx12 {
namespace gfx1250 {
// Counter register info - Auto-generated from chip_offset_byte.h, edit with extra caution
static const CounterRegInfo ChaCounterRegAddr[]           = {REG_INFO_4(CHA)};
static const CounterRegInfo ChcCounterRegAddr[]           = {REG_INFO_4(CHC)};
static const CounterRegInfo CpcCounterRegAddr[]           = {REG_INFO_2(CPC)};
static const CounterRegInfo CpfCounterRegAddr[]           = {REG_INFO_2(CPF)};
static const CounterRegInfo CpgCounterRegAddr[]           = {REG_INFO_2(CPG)};
static const CounterRegInfo GcmcVmL2CounterRegAddr[]      = {REG_INFO_WITH_CFG_16(GC, GCMC_VM_L2)};
static const CounterRegInfo GcrCounterRegAddr[]           = {REG_INFO_WITH_CTRL_2(GCR, REG_32B_ADDR(GC, 0, regGCR_GENERAL_CNTL))};
static const CounterRegInfo Gcutcl2CounterRegAddr[]         = {REG_INFO_WITH_CFG_4(GC, GCUTCL2)};
static const CounterRegInfo GcCaneCounterRegAddr[]          = {REG_INFO_1(GC_CANE)};
static const CounterRegInfo GcAtcl2CounterRegAddr[]         = {REG_INFO_WITH_CFG_16(GC, GC_ATC_L2)};
static const CounterRegInfo Gcutcl2FfbmCounterRegAddr[]     = {REG_INFO_WITH_CFG_16(GC, GCUTCL2_FFBM)};
static const CounterRegInfo GcL2tlbCounterRegAddr[]         = {REG_INFO_WITH_CFG_4(GC, GC_L2TLB)};
static const CounterRegInfo Gcutcl2NhttlbCounterRegAddr[] = {REG_INFO_WITH_CFG_16(GC, GCUTCL2_NHTTLB)};
static const CounterRegInfo GcEaCpwdCounterRegAddr[]      = {REG_INFO_2(GC_EA_CPWD)};
static const CounterRegInfo GcEaSeCounterRegAddr[]        = {REG_INFO_WITH_INST_2(GC_EA_SE, 8)};
static const CounterRegInfo Gl1aCounterRegAddr[]          = {REG_INFO_4(GL1A)};
static const CounterRegInfo Gl1cCounterRegAddr[]          = {REG_INFO_4(GL1C)};
static const CounterRegInfo Gl2aCounterRegAddr[]          = {REG_INFO_WITH_INST_4(GL2A, 8)};
static const CounterRegInfo Gl2cCounterRegAddr[]          = {REG_INFO_WITH_INST_4(GL2C, 8)};
static const CounterRegInfo GlarbaCounterRegAddr[]        = {REG_INFO_4(GLARBA)};
static const CounterRegInfo GlarbcCounterRegAddr[]        = {REG_INFO_4(GLARBC)};
static const CounterRegInfo GrbmCounterRegAddr[]          = {REG_INFO_2(GRBM)};
static const CounterRegInfo GrbmaCounterRegAddr[]         = {REG_INFO_WITH_INST_2(GRBMA, 8)};
static const CounterRegInfo GrbmhCounterRegAddr[]         = {REG_INFO_2(GRBMH)};
static const CounterRegInfo RlcCounterRegAddr[]           = {REG_INFO_2(RLC)};
static const CounterRegInfo SdmaCounterRegAddr[]          = {REG_INFO_6(SDMA0_SDMA), REG_INFO_6(SDMA1_SDMA)};
static const CounterRegInfo SpiCounterRegAddr[]           = {REG_INFO_6(SPI)};
static const CounterRegInfo SqgCounterRegAddr[]           = {REG_INFO_WITH_CTRL_8(SQG, REG_32B_ADDR(GC, 0, regSQG_PERFCOUNTER_CTRL))};
static const CounterRegInfo TaCounterRegAddr[]            = {REG_INFO_2(TA)};
static const CounterRegInfo TcpCounterRegAddr[]           = {REG_INFO_8(TCP)};
static const CounterRegInfo TdCounterRegAddr[]            = {REG_INFO_2(TD)};
static const CounterRegInfo Utcl1CounterRegAddr[]         = {REG_INFO_4(UTCL1)};

// Special handling of SQC:
//   SQC only supports 32bit PMC.
//   regSQ_PERFCOUNTER#even_number#_SELECT is used by PMC and SPM
//   regSQ_PERFCOUNTER#odd_number#_SELECT is used by SPM only
static const CounterRegInfo SqcCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER0_SELECT),  REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER0_LO), REG_32B_NULL, REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER2_SELECT),  REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER1_LO), REG_32B_NULL, REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER4_SELECT),  REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER2_LO), REG_32B_NULL, REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER6_SELECT),  REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER3_LO), REG_32B_NULL, REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER8_SELECT),  REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER4_LO), REG_32B_NULL, REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER10_SELECT), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER5_LO), REG_32B_NULL, REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER12_SELECT), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER6_LO), REG_32B_NULL, REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER14_SELECT), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER7_LO), REG_32B_NULL, REG_32B_NULL}};

// Global blocks: ATCL2 CHA CHC CPC CPF CPG EA FFBM GCR GL2A GL2C GRBM RLC SDMA VML2 UTCL2
//   (Grphics only - not supported in ROCm): GE1 GE2_DIST PH
//   (Grphics only): CPG is for graphics, but it is not physically removed for compute products
//   (Not enabled for gfx12): CHCG GDS GUS
// AIGC blocks: EA GL2A GL2C GRBMA UTCL2(GPUVM/ATCL2/FFBM)
// clang-format off
static const GpuBlockInfo GceaSeCounterBlockInfo = {
    "GCEA_SE",
    __BLOCK_ID(GCEA_SE),  // 51 = AQLPROFILE_BLOCK_NAME_GCEA_SE (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 17)
    GcEaSeCounterBlockNumInstances,
    GcEaSeCounterBlockMaxEvent,
    GcEaSeCounterBlockNumCounters,
    GcEaSeCounterRegAddr,
    Primitives::select_value,
    CounterBlockGrbmaAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo Gl2aCounterBlockInfo = {
    "GL2A",
    __BLOCK_ID_HSA(GL2A),  // 28 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A
    Gl2aCounterBlockNumInstances,
    Gl2aCounterBlockMaxEvent,
    Gl2aCounterBlockNumCounters,
    Gl2aCounterRegAddr,
    Primitives::select_value,
    CounterBlockGrbmaAttr|CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo Gl2cCounterBlockInfo = {
    "GL2C",
    __BLOCK_ID_HSA(GL2C),  // 29 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C
    Gl2cCounterBlockNumInstances,
    Gl2cCounterBlockMaxEvent,
    Gl2cCounterBlockNumCounters,
    Gl2cCounterRegAddr,
    Primitives::select_value,
    CounterBlockGrbmaAttr|CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo GrbmaCounterBlockInfo = {
    "GRBMA",
    __BLOCK_ID(GRBMA),  // 56 = AQLPROFILE_BLOCK_NAME_GRBMA (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 22)
    GrbmaCounterBlockNumInstances,
    GrbmaCounterBlockMaxEvent,
    GrbmaCounterBlockNumCounters,
    GrbmaCounterRegAddr,
    Primitives::select_value,
    CounterBlockGrbmaAttr|CounterBlockGRBMAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo Atcl2CounterBlockInfo = {
    "ATCL2",
    __BLOCK_ID_HSA(ATCL2),  // 22 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2
    GcAtcL2CounterBlockNumInstances,
    GcAtcl2CounterBlockMaxEvent,
    GcAtcL2CounterBlockNumCounters,
    GcAtcl2CounterRegAddr,
    Primitives::mc_select_value,
    CounterBlockUtcl2Attr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo Gcutcl2FfbmCounterBlockInfo = {
    "GC_FFBM",
    __BLOCK_ID(GC_FFBM),  // 45 = AQLPROFILE_BLOCK_NAME_GC_FFBM (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 11)
    Gcutcl2FfbmCounterBlockNumInstances,
    Gcutcl2FfbmCounterBlockMaxEvent,
    Gcutcl2FfbmCounterBlockNumCounters,
    Gcutcl2FfbmCounterRegAddr,
    Primitives::mc_select_value,
    CounterBlockUtcl2Attr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo GcL2tlbCounterBlockInfo = {
    "GC_L2TLB",
    __BLOCK_ID(GC_L2TLB),  // 46 = AQLPROFILE_BLOCK_NAME_GC_L2TLB (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 12)
    GcL2tlbCounterBlockNumInstances,
    GcL2tlbCounterBlockMaxEvent,
    GcL2tlbCounterBlockNumCounters,
    GcL2tlbCounterRegAddr,
    Primitives::mc_select_value,
    CounterBlockUtcl2Attr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo Gcutcl2NhttlbCounterBlockInfo = {
    "GC_NHTTLB",
    __BLOCK_ID(GC_NHTTLB),  // 57 = AQLPROFILE_BLOCK_NAME_GC_NHTTLB (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 23)
    Gcutcl2NhttlbCounterBlockNumInstances,
    Gcutcl2NhttlbCounterBlockMaxEvent,
    Gcutcl2NhttlbCounterBlockNumCounters,
    Gcutcl2NhttlbCounterRegAddr,
    Primitives::mc_select_value,
    CounterBlockUtcl2Attr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo GcUtcl2CounterBlockInfo = {
    "GC_UTCL2",
    __BLOCK_ID(GC_UTCL2),  // 48 = AQLPROFILE_BLOCK_NAME_GC_UTCL2 (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 14)
    Gcutcl2CounterBlockNumInstances,
    Gcutcl2CounterBlockMaxEvent,
    Gcutcl2CounterBlockNumCounters,
    Gcutcl2CounterRegAddr,
    Primitives::mc_select_value,
    CounterBlockUtcl2Attr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo GcVml2CounterBlockInfo = {
    "GC_VML2",
    __BLOCK_ID(GC_VML2),  // 49 = AQLPROFILE_BLOCK_NAME_GC_VML2 (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 15)
    GcmcVmL2CounterBlockNumInstances,
    GcmcVmL2CounterBlockMaxEvent,
    GcmcVmL2CounterBlockNumCounters,
    GcmcVmL2CounterRegAddr,
    Primitives::mc_select_value,
    CounterBlockUtcl2Attr,
    BLOCK_DELAY_NONE};
// Global blocks (gfx1250): GC_CANE GLARBA GLARBC
static const GpuBlockInfo GcCaneCounterBlockInfo = {
    "GC_CANE",
    __BLOCK_ID(GC_CANE),  // 44 = AQLPROFILE_BLOCK_NAME_GC_CANE (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 10)
    GcCaneCounterBlockNumInstances,
    GcCaneCounterBlockMaxEvent,
    GcCaneCounterBlockNumCounters,
    GcCaneCounterRegAddr,
    Primitives::select_value,
    CounterBlockDfltAttr|CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo GlarbaCounterBlockInfo = {
    "GLARBA",
    __BLOCK_ID(GLARBA),  // 54 = AQLPROFILE_BLOCK_NAME_GLARBA (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 20)
    GlarbaCounterBlockNumInstances,
    GlarbaCounterBlockMaxEvent,
    GlarbaCounterBlockNumCounters,
    GlarbaCounterRegAddr,
    Primitives::select_value,
    CounterBlockDfltAttr|CounterBlockTcAttr|CounterBlockGlarbAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo GlarbcCounterBlockInfo = {
    "GLARBC",
    __BLOCK_ID(GLARBC),  // 55 = AQLPROFILE_BLOCK_NAME_GLARBC (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 21)
    GlarbcCounterBlockNumInstances,
    GlarbcCounterBlockMaxEvent,
    GlarbcCounterBlockNumCounters,
    GlarbcCounterRegAddr,
    Primitives::select_value,
    CounterBlockDfltAttr|CounterBlockTcAttr|CounterBlockGlarbAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo ChaCounterBlockInfo = {
    "CHA",
    __BLOCK_ID(CHA),  // 42 = AQLPROFILE_BLOCK_NAME_CHA (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 8)
    ChaCounterBlockNumInstances,
    ChaCounterBlockMaxEvent,
    ChaCounterBlockNumCounters,
    ChaCounterRegAddr,
    Primitives::select_value,
    CounterBlockDfltAttr|CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo ChcCounterBlockInfo = {
    "CHC",
    __BLOCK_ID(CHC),  // 43 = AQLPROFILE_BLOCK_NAME_CHC (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 9)
    ChcCounterBlockNumInstances,
    ChcCounterBlockMaxEvent,
    ChcCounterBlockNumCounters,
    ChcCounterRegAddr,
    Primitives::select_value,
    CounterBlockDfltAttr|CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo CpcCounterBlockInfo = {
    "CPC",
    __BLOCK_ID_HSA(CPC),  // 0 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC
    CpcCounterBlockNumInstances,
    CpcCounterBlockMaxEvent,
    CpcCounterBlockNumCounters,
    CpcCounterRegAddr,
    Primitives::select_value,
    CounterBlockDfltAttr|CounterBlockSpmGlobalAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo CpfCounterBlockInfo = {
    "CPF",
    __BLOCK_ID_HSA(CPF),  // 1 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF
    CpfCounterBlockNumInstances,
    CpfCounterBlockMaxEvent,
    CpfCounterBlockNumCounters,
    CpfCounterRegAddr,
    Primitives::select_value,
    CounterBlockDfltAttr|CounterBlockSpmGlobalAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo CpgCounterBlockInfo = {
    "CPG",
    __BLOCK_ID(CPG),  // 40 = AQLPROFILE_BLOCK_NAME_CPG (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 6)
    CpgCounterBlockNumInstances,
    CpgCounterBlockMaxEvent,
    CpgCounterBlockNumCounters,
    CpgCounterRegAddr,
    Primitives::select_value,
    CounterBlockDfltAttr|CounterBlockSpmGlobalAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo GcrCounterBlockInfo = {
    "GCR",
    __BLOCK_ID_HSA(GCR),  // 30 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR
    GcrCounterBlockNumInstances,
    GcrCounterBlockMaxEvent,
    GcrCounterBlockNumCounters,
    GcrCounterRegAddr,
    Primitives::select_value,
    CounterBlockDfltAttr|CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo GceaCounterBlockInfo = {
    "GCEA",
    __BLOCK_ID_HSA(GCEA),  // 23 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA
    GcEaCpwdCounterBlockNumInstances,
    GcEaCpwdCounterBlockMaxEvent,
    GcEaCpwdCounterBlockNumCounters,
    GcEaCpwdCounterRegAddr,
    Primitives::select_value,
    CounterBlockDfltAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo GrbmCounterBlockInfo = {
    "GRBM",
    __BLOCK_ID_HSA(GRBM),  // 3 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM
    GrbmCounterBlockNumInstances,
    GrbmCounterBlockMaxEvent,
    GrbmCounterBlockNumCounters,
    GrbmCounterRegAddr,
    Primitives::select_value,
    CounterBlockDfltAttr|CounterBlockGRBMAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo RlcCounterBlockInfo = {
    "RLC",
    __BLOCK_ID(RLC),  // 41 = AQLPROFILE_BLOCK_NAME_RLC (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 7)
    RlcCounterBlockNumInstances,
    RlcCounterBlockMaxEvent,
    RlcCounterBlockNumCounters,
    RlcCounterRegAddr,
    Primitives::select_value,
    CounterBlockDfltAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo SdmaCounterBlockInfo = {
    "SDMA",
    __BLOCK_ID_HSA(SDMA),  // 25 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA
    SdmaCounterBlockNumInstances,
    SdmaCounterBlockMaxEvent,
    SdmaCounterBlockNumCounters,
    SdmaCounterRegAddr,
    Primitives::select_value,
    CounterBlockDfltAttr|CounterBlockExplInstAttr|CounterBlockSpmGlobalAttr,
    BLOCK_DELAY_NONE};
// SE blocks: GRBMH SPI SQG UTCL1
//   (Grphics only - not supported in ROCm): GE GL1XA GL1XC PA PC WGS
static const GpuBlockInfo GrbmhCounterBlockInfo = {
    "GRBMH",
    __BLOCK_ID(GRBMH),  // 52 = AQLPROFILE_BLOCK_NAME_GRBMH (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 18)
    GrbmhCounterBlockNumInstances,
    GrbmhCounterBlockMaxEvent,
    GrbmhCounterBlockNumCounters,
    GrbmhCounterRegAddr,
    Primitives::select_value,
    CounterBlockSeAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo SpiCounterBlockInfo = {
    "SPI",
    __BLOCK_ID_HSA(SPI),  // 5 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI
    SpiCounterBlockNumInstances,
    SpiCounterBlockMaxEvent,
    SpiCounterBlockNumCounters,
    SpiCounterRegAddr,
    Primitives::select_value,
    CounterBlockSeAttr|CounterBlockSPIAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo SqgCounterBlockInfo = {
    "SQG",
    __BLOCK_ID(SQG),  // 53 = AQLPROFILE_BLOCK_NAME_SQG (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 19)
    SqgCounterBlockNumInstances,
    SqgCounterBlockMaxEvent,
    SqgCounterBlockNumCounters,
    SqgCounterRegAddr,
    Primitives::sq_select_value,
    CounterBlockSeAttr|CounterBlockSqAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo GcUtcl1CounterBlockInfo = {
    "GC_UTCL1",
    __BLOCK_ID(GC_UTCL1),  // 47 = AQLPROFILE_BLOCK_NAME_GC_UTCL1 (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER + 13)
    Utcl1CounterBlockNumInstances,
    Utcl1CounterBlockMaxEvent,
    Utcl1CounterBlockNumCounters,
    Utcl1CounterRegAddr,
    Primitives::select_value,
    CounterBlockSeAttr,
    BLOCK_DELAY_NONE};
// SE blocks (gfx1250): GL1A GL1C (moved from SA)
static const GpuBlockInfo Gl1aCounterBlockInfo = {
    "GL1A",
    __BLOCK_ID_HSA(GL1A),  // 26 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A
    Gl1aCounterBlockNumInstances,
    Gl1aCounterBlockMaxEvent,
    Gl1aCounterBlockNumCounters,
    Gl1aCounterRegAddr,
    Primitives::select_value,
    CounterBlockSeAttr|CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo Gl1cCounterBlockInfo = {
    "GL1C",
    __BLOCK_ID_HSA(GL1C),  // 27 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C
    Gl1cCounterBlockNumInstances,
    Gl1cCounterBlockMaxEvent,
    Gl1cCounterBlockNumCounters,
    Gl1cCounterRegAddr,
    Primitives::select_value,
    CounterBlockSeAttr|CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
// WGP blocks: SQC TA TCP TD
static const GpuBlockInfo SqcCounterBlockInfo = {
    "SQ",
    __BLOCK_ID_HSA(SQ),  // 6 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ
    SqcCounterBlockNumInstances,
    SqcCounterBlockMaxEvent,
    SqcCounterBlockNumCounters,
    SqcCounterRegAddr,
    Primitives::sq_select_value,
    CounterBlockSeAttr|CounterBlockSaAttr|CounterBlockWgpAttr|CounterBlockSqAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo TaCounterBlockInfo = {
    "TA",
    __BLOCK_ID_HSA(TA),  // 10 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA
    TaCounterBlockNumInstances,
    TaCounterBlockMaxEvent,
    TaCounterBlockNumCounters,
    TaCounterRegAddr,
    Primitives::select_value,
    CounterBlockSeAttr|CounterBlockSaAttr|CounterBlockWgpAttr|CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo TdCounterBlockInfo = {
    "TD",
    __BLOCK_ID_HSA(TD),  // 14 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD
    TdCounterBlockNumInstances,
    TdCounterBlockMaxEvent,
    TdCounterBlockNumCounters,
    TdCounterRegAddr,
    Primitives::select_value,
    CounterBlockSeAttr|CounterBlockSaAttr|CounterBlockWgpAttr|CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo TcpCounterBlockInfo = {
    "TCP",
    __BLOCK_ID_HSA(TCP),  // 13 = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP
    TcpCounterBlockNumInstances,
    TcpCounterBlockMaxEvent,
    TcpCounterBlockNumCounters,
    TcpCounterRegAddr,
    Primitives::select_value,
    CounterBlockSeAttr|CounterBlockSaAttr|CounterBlockWgpAttr|CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
// clang-format on
}  // namespace gfx1250
}  // namespace gfx12
}  // namespace gfxip

#endif  // _GFX1250_BLOCKTABLE_H_
