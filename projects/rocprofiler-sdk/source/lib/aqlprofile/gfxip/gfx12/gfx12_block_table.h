
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

#ifndef _GFX12_BLOCKTABLE_H_
#define _GFX12_BLOCKTABLE_H_

#define REG_INFO_ENTRY_FULL(BLOCK, INST, CTRL, INDEX)                                              \
    {                                                                                              \
        REG_32B_ADDR(GC, INST, reg##BLOCK##_PERFCOUNTER##INDEX##_SELECT), CTRL,                    \
            REG_32B_ADDR(GC, INST, reg##BLOCK##_PERFCOUNTER##INDEX##_LO),                          \
            REG_32B_ADDR(GC, INST, reg##BLOCK##_PERFCOUNTER##INDEX##_HI), REG_32B_NULL             \
    }
#define REG_INFO_ENTRY(BLOCK, INDEX)                 REG_INFO_ENTRY_FULL(BLOCK, 0, REG_32B_NULL, INDEX)
#define REG_INFO_ENTRY_WITH_CTRL(BLOCK, CTRL, INDEX) REG_INFO_ENTRY_FULL(BLOCK, 0, CTRL, INDEX)
#define REG_INFO_ENTRY_WITH_INST(BLOCK, INST, INDEX)                                               \
    REG_INFO_ENTRY_FULL(BLOCK, INST, REG_32B_NULL, INDEX)

#define REG_INFO_1(BLOCK) REG_INFO_ENTRY(BLOCK, 0)
#define REG_INFO_2(BLOCK) REG_INFO_1(BLOCK), REG_INFO_ENTRY(BLOCK, 1)
#define REG_INFO_3(BLOCK) REG_INFO_2(BLOCK), REG_INFO_ENTRY(BLOCK, 2)
#define REG_INFO_4(BLOCK) REG_INFO_3(BLOCK), REG_INFO_ENTRY(BLOCK, 3)
#define REG_INFO_5(BLOCK) REG_INFO_4(BLOCK), REG_INFO_ENTRY(BLOCK, 4)
#define REG_INFO_6(BLOCK) REG_INFO_5(BLOCK), REG_INFO_ENTRY(BLOCK, 5)
#define REG_INFO_7(BLOCK) REG_INFO_6(BLOCK), REG_INFO_ENTRY(BLOCK, 6)
#define REG_INFO_8(BLOCK) REG_INFO_7(BLOCK), REG_INFO_ENTRY(BLOCK, 7)

#define REG_INFO_WITH_CTRL_1(BLOCK, CTRL) REG_INFO_ENTRY_WITH_CTRL(BLOCK, CTRL, 0)
#define REG_INFO_WITH_CTRL_2(BLOCK, CTRL)                                                          \
    REG_INFO_WITH_CTRL_1(BLOCK, CTRL), REG_INFO_ENTRY_WITH_CTRL(BLOCK, CTRL, 1)
#define REG_INFO_WITH_CTRL_3(BLOCK, CTRL)                                                          \
    REG_INFO_WITH_CTRL_2(BLOCK, CTRL), REG_INFO_ENTRY_WITH_CTRL(BLOCK, CTRL, 2)
#define REG_INFO_WITH_CTRL_4(BLOCK, CTRL)                                                          \
    REG_INFO_WITH_CTRL_3(BLOCK, CTRL), REG_INFO_ENTRY_WITH_CTRL(BLOCK, CTRL, 3)
#define REG_INFO_WITH_CTRL_5(BLOCK, CTRL)                                                          \
    REG_INFO_WITH_CTRL_4(BLOCK, CTRL), REG_INFO_ENTRY_WITH_CTRL(BLOCK, CTRL, 4)
#define REG_INFO_WITH_CTRL_6(BLOCK, CTRL)                                                          \
    REG_INFO_WITH_CTRL_5(BLOCK, CTRL), REG_INFO_ENTRY_WITH_CTRL(BLOCK, CTRL, 5)
#define REG_INFO_WITH_CTRL_7(BLOCK, CTRL)                                                          \
    REG_INFO_WITH_CTRL_6(BLOCK, CTRL), REG_INFO_ENTRY_WITH_CTRL(BLOCK, CTRL, 6)
#define REG_INFO_WITH_CTRL_8(BLOCK, CTRL)                                                          \
    REG_INFO_WITH_CTRL_7(BLOCK, CTRL), REG_INFO_ENTRY_WITH_CTRL(BLOCK, CTRL, 7)

#define REG_INFO_WITH_INST_1(BLOCK, INST) REG_INFO_ENTRY_WITH_INST(BLOCK, INST, 0)
#define REG_INFO_WITH_INST_2(BLOCK, INST)                                                          \
    REG_INFO_WITH_INST_1(BLOCK, INST), REG_INFO_ENTRY_WITH_INST(BLOCK, INST, 1)
#define REG_INFO_WITH_INST_3(BLOCK, INST)                                                          \
    REG_INFO_WITH_INST_2(BLOCK, INST), REG_INFO_ENTRY_WITH_INST(BLOCK, INST, 2)
#define REG_INFO_WITH_INST_4(BLOCK, INST)                                                          \
    REG_INFO_WITH_INST_3(BLOCK, INST), REG_INFO_ENTRY_WITH_INST(BLOCK, INST, 3)

#define REG_INFO_WITH_CFG(IP, BLOCK, INDEX)                                                        \
    {                                                                                              \
        REG_32B_ADDR(IP, 0, reg##BLOCK##_PERFCOUNTER##INDEX##_CFG),                                \
            REG_32B_ADDR(IP, 0, reg##BLOCK##_PERFCOUNTER_RSLT_CNTL),                               \
            REG_32B_ADDR(IP, 0, reg##BLOCK##_PERFCOUNTER_LO),                                      \
            REG_32B_ADDR(IP, 0, reg##BLOCK##_PERFCOUNTER_HI), REG_32B_NULL                         \
    }
#define REG_INFO_WITH_CFG_1(IP, BLOCK) REG_INFO_WITH_CFG(IP, BLOCK, 0)
#define REG_INFO_WITH_CFG_2(IP, BLOCK)                                                             \
    REG_INFO_WITH_CFG_1(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 1)
#define REG_INFO_WITH_CFG_3(IP, BLOCK)                                                             \
    REG_INFO_WITH_CFG_2(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 2)
#define REG_INFO_WITH_CFG_4(IP, BLOCK)                                                             \
    REG_INFO_WITH_CFG_3(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 3)
#define REG_INFO_WITH_CFG_5(IP, BLOCK)                                                             \
    REG_INFO_WITH_CFG_4(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 4)
#define REG_INFO_WITH_CFG_6(IP, BLOCK)                                                             \
    REG_INFO_WITH_CFG_5(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 5)
#define REG_INFO_WITH_CFG_7(IP, BLOCK)                                                             \
    REG_INFO_WITH_CFG_6(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 6)
#define REG_INFO_WITH_CFG_8(IP, BLOCK)                                                             \
    REG_INFO_WITH_CFG_7(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 7)
#define REG_INFO_WITH_CFG_9(IP, BLOCK)                                                             \
    REG_INFO_WITH_CFG_8(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 8)
#define REG_INFO_WITH_CFG_10(IP, BLOCK)                                                            \
    REG_INFO_WITH_CFG_9(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 9)
#define REG_INFO_WITH_CFG_11(IP, BLOCK)                                                            \
    REG_INFO_WITH_CFG_10(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 10)
#define REG_INFO_WITH_CFG_12(IP, BLOCK)                                                            \
    REG_INFO_WITH_CFG_11(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 11)
#define REG_INFO_WITH_CFG_13(IP, BLOCK)                                                            \
    REG_INFO_WITH_CFG_12(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 12)
#define REG_INFO_WITH_CFG_14(IP, BLOCK)                                                            \
    REG_INFO_WITH_CFG_13(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 13)
#define REG_INFO_WITH_CFG_15(IP, BLOCK)                                                            \
    REG_INFO_WITH_CFG_14(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 14)
#define REG_INFO_WITH_CFG_16(IP, BLOCK)                                                            \
    REG_INFO_WITH_CFG_15(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 15)

namespace gfxip
{
namespace gfx12
{
#if GFX12_VARIANT == GFX12_VARIANT_1250
namespace gfx1250
{
#else
namespace gfx1200
{
#endif
// Counter register info - Auto-generated from chip_offset_byte.h, edit with extra caution
static const CounterRegInfo ChaCounterRegAddr[] = {REG_INFO_4(CHA)};
static const CounterRegInfo ChcCounterRegAddr[] = {REG_INFO_4(CHC)};
static const CounterRegInfo CpcCounterRegAddr[] = {REG_INFO_2(CPC)};
static const CounterRegInfo CpfCounterRegAddr[] = {REG_INFO_2(CPF)};
static const CounterRegInfo CpgCounterRegAddr[] = {REG_INFO_2(CPG)};
#if GFX12_VARIANT == GFX12_VARIANT_1250
static const CounterRegInfo GcmcVmL2CounterRegAddr[] = {REG_INFO_WITH_CFG_16(GC, GCMC_VM_L2)};
#else
static const CounterRegInfo GcmcVmL2CounterRegAddr[]  = {REG_INFO_WITH_CFG_8(GC, GCMC_VM_L2)};
#endif
static const CounterRegInfo GcrCounterRegAddr[] = {
    REG_INFO_WITH_CTRL_2(GCR, REG_32B_ADDR(GC, 0, regGCR_GENERAL_CNTL))};
#if GFX12_VARIANT <= GFX12_VARIANT_1201
static const CounterRegInfo RpbCounterRegAddr[] = {REG_INFO_WITH_CFG_4(ATHUB, RPB)};
#endif
static const CounterRegInfo Gcutcl2CounterRegAddr[] = {REG_INFO_WITH_CFG_4(GC, GCUTCL2)};
// static const CounterRegInfo Gcvml2CounterRegAddr[] = {REG_INFO_2(GCVML2)};
#if GFX12_VARIANT == GFX12_VARIANT_1250
static const CounterRegInfo GcCaneCounterRegAddr[]      = {REG_INFO_1(GC_CANE)};
static const CounterRegInfo GcAtcl2CounterRegAddr[]     = {REG_INFO_WITH_CFG_16(GC, GC_ATC_L2)};
static const CounterRegInfo Gcutcl2FfbmCounterRegAddr[] = {REG_INFO_WITH_CFG_16(GC, GCUTCL2_FFBM)};
static const CounterRegInfo GcL2tlbCounterRegAddr[]     = {REG_INFO_WITH_CFG_4(GC, GC_L2TLB)};
static const CounterRegInfo Gcutcl2NhttlbCounterRegAddr[] = {
    REG_INFO_WITH_CFG_16(GC, GCUTCL2_NHTTLB)};
#endif
static const CounterRegInfo GcEaCpwdCounterRegAddr[] = {REG_INFO_2(GC_EA_CPWD)};
#if GFX12_VARIANT == GFX12_VARIANT_1250
static const CounterRegInfo GcEaSeCounterRegAddr[] = {REG_INFO_WITH_INST_2(GC_EA_SE, 8)};
#else
static const CounterRegInfo GcEaSeCounterRegAddr[]    = {REG_INFO_2(GC_EA_SE)};
#endif
static const CounterRegInfo Gl1aCounterRegAddr[] = {REG_INFO_4(GL1A)};
static const CounterRegInfo Gl1cCounterRegAddr[] = {REG_INFO_4(GL1C)};
#if GFX12_VARIANT == GFX12_VARIANT_1250
static const CounterRegInfo Gl2aCounterRegAddr[] = {REG_INFO_WITH_INST_4(GL2A, 8)};
static const CounterRegInfo Gl2cCounterRegAddr[] = {REG_INFO_WITH_INST_4(GL2C, 8)};
#else
static const CounterRegInfo Gl2aCounterRegAddr[]      = {REG_INFO_4(GL2A)};
static const CounterRegInfo Gl2cCounterRegAddr[]      = {REG_INFO_4(GL2C)};
#endif
#if GFX12_VARIANT == GFX12_VARIANT_1250
static const CounterRegInfo GlarbaCounterRegAddr[] = {REG_INFO_4(GLARBA)};
static const CounterRegInfo GlarbcCounterRegAddr[] = {REG_INFO_4(GLARBC)};
#endif
static const CounterRegInfo GrbmCounterRegAddr[] = {REG_INFO_2(GRBM)};
#if GFX12_VARIANT == GFX12_VARIANT_1250
static const CounterRegInfo GrbmaCounterRegAddr[] = {REG_INFO_WITH_INST_2(GRBMA, 8)};
#endif
static const CounterRegInfo GrbmhCounterRegAddr[] = {REG_INFO_2(GRBMH)};
static const CounterRegInfo RlcCounterRegAddr[]   = {REG_INFO_2(RLC)};
#if GFX12_VARIANT == GFX12_VARIANT_1250
static const CounterRegInfo SdmaCounterRegAddr[] = {REG_INFO_6(SDMA0_SDMA), REG_INFO_6(SDMA1_SDMA)};
#else
static const CounterRegInfo SdmaCounterRegAddr[]      = {REG_INFO_2(SDMA0), REG_INFO_2(SDMA1)};
#endif
static const CounterRegInfo SpiCounterRegAddr[] = {REG_INFO_6(SPI)};
// static const CounterRegInfo SqcCounterRegAddr[] = {REG_INFO_WITH_CTRL_16(SQ, REG_32B_ADDR(GC, 0,
// regSQ_PERFCOUNTER_CTRL))};
static const CounterRegInfo SqgCounterRegAddr[] = {
    REG_INFO_WITH_CTRL_8(SQG, REG_32B_ADDR(GC, 0, regSQG_PERFCOUNTER_CTRL))};
static const CounterRegInfo TaCounterRegAddr[] = {REG_INFO_2(TA)};
#if GFX12_VARIANT == GFX12_VARIANT_1250
static const CounterRegInfo TcpCounterRegAddr[] = {REG_INFO_8(TCP)};
#else
static const CounterRegInfo TcpCounterRegAddr[]       = {REG_INFO_4(TCP)};
#endif
static const CounterRegInfo TdCounterRegAddr[]    = {REG_INFO_2(TD)};
static const CounterRegInfo Utcl1CounterRegAddr[] = {REG_INFO_4(UTCL1)};

// Special handling of SQC:
//   SQC only supports 32bit PMC.
//   regSQ_PERFCOUNTER#even_number#_SELECT is used by PMC and SPM
//   regSQ_PERFCOUNTER#odd_number#_SELECT is used by SPM only
static const CounterRegInfo SqcCounterRegAddr[] = {{REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER0_SELECT),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER0_LO),
                                                    REG_32B_NULL,
                                                    REG_32B_NULL},
                                                   {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER2_SELECT),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER1_LO),
                                                    REG_32B_NULL,
                                                    REG_32B_NULL},
                                                   {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER4_SELECT),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER2_LO),
                                                    REG_32B_NULL,
                                                    REG_32B_NULL},
                                                   {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER6_SELECT),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER3_LO),
                                                    REG_32B_NULL,
                                                    REG_32B_NULL},
                                                   {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER8_SELECT),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER4_LO),
                                                    REG_32B_NULL,
                                                    REG_32B_NULL},
                                                   {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER10_SELECT),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER5_LO),
                                                    REG_32B_NULL,
                                                    REG_32B_NULL},
                                                   {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER12_SELECT),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER6_LO),
                                                    REG_32B_NULL,
                                                    REG_32B_NULL},
                                                   {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER14_SELECT),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
                                                    REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER7_LO),
                                                    REG_32B_NULL,
                                                    REG_32B_NULL}};

#if GFX12_VARIANT != GFX12_VARIANT_1250
// Special handling of GCVML2 (SPM only):
static const CounterRegInfo Gcvml2CounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regGCVML2_PERFCOUNTER2_0_SELECT),
     REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGCVML2_PERFCOUNTER2_0_LO),
     REG_32B_ADDR(GC, 0, regGCVML2_PERFCOUNTER2_0_HI),
     REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regGCVML2_PERFCOUNTER2_1_SELECT),
     REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGCVML2_PERFCOUNTER2_1_LO),
     REG_32B_ADDR(GC, 0, regGCVML2_PERFCOUNTER2_1_HI),
     REG_32B_NULL}};
#endif

// Global blocks: ATCL2 CHA CHC CPC CPF CPG EA FFBM GCR GL2A GL2C GRBM RLC SDMA VML2 UTCL2
//   (Grphics only - not supported in ROCm): GE1 GE2_DIST PH
//   (Grphics only): CPG is for graphics, but it is not physically removed for compute products
//   (Not enabled for gfx12): CHCG GDS GUS
#if GFX12_VARIANT == GFX12_VARIANT_1250
// AIGC blocks: EA GL2A GL2C GRBMA UTCL2(GPUVM/ATCL2/FFBM)
static const GpuBlockInfo GceaSeCounterBlockInfo        = {"GCEA_SE",
                                                    __BLOCK_ID(GCEA_SE),
                                                    GcEaSeCounterBlockNumInstances,
                                                    GcEaSeCounterBlockMaxEvent,
                                                    GcEaSeCounterBlockNumCounters,
                                                    GcEaSeCounterRegAddr,
                                                    gfx12_cntx_prim::select_value,
                                                    CounterBlockGrbmaAttr,
                                                    BLOCK_DELAY_NONE};
static const GpuBlockInfo Gl2aCounterBlockInfo          = {"GL2A",
                                                  __BLOCK_ID_HSA(GL2A),
                                                  Gl2aCounterBlockNumInstances,
                                                  Gl2aCounterBlockMaxEvent,
                                                  Gl2aCounterBlockNumCounters,
                                                  Gl2aCounterRegAddr,
                                                  gfx12_cntx_prim::select_value,
                                                  CounterBlockGrbmaAttr | CounterBlockTcAttr,
                                                  BLOCK_DELAY_NONE};
static const GpuBlockInfo Gl2cCounterBlockInfo          = {"GL2C",
                                                  __BLOCK_ID_HSA(GL2C),
                                                  Gl2cCounterBlockNumInstances,
                                                  Gl2cCounterBlockMaxEvent,
                                                  Gl2cCounterBlockNumCounters,
                                                  Gl2cCounterRegAddr,
                                                  gfx12_cntx_prim::select_value,
                                                  CounterBlockGrbmaAttr | CounterBlockTcAttr,
                                                  BLOCK_DELAY_NONE};
static const GpuBlockInfo GrbmaCounterBlockInfo         = {"GRBMA",
                                                   __BLOCK_ID(GRBMA),
                                                   GrbmaCounterBlockNumInstances,
                                                   GrbmaCounterBlockMaxEvent,
                                                   GrbmaCounterBlockNumCounters,
                                                   GrbmaCounterRegAddr,
                                                   gfx12_cntx_prim::select_value,
                                                   CounterBlockGrbmaAttr | CounterBlockGRBMAttr,
                                                   BLOCK_DELAY_NONE};
static const GpuBlockInfo Atcl2CounterBlockInfo         = {"ATCL2",
                                                   __BLOCK_ID_HSA(ATCL2),
                                                   GcAtcL2CounterBlockNumInstances,
                                                   GcAtcl2CounterBlockMaxEvent,
                                                   GcAtcL2CounterBlockNumCounters,
                                                   GcAtcl2CounterRegAddr,
                                                   gfx12_cntx_prim::mc_select_value,
                                                   CounterBlockUtcl2Attr,
                                                   BLOCK_DELAY_NONE};
static const GpuBlockInfo Gcutcl2FfbmCounterBlockInfo   = {"GC_FFBM",
                                                         __BLOCK_ID(GC_FFBM),
                                                         Gcutcl2FfbmCounterBlockNumInstances,
                                                         Gcutcl2FfbmCounterBlockMaxEvent,
                                                         Gcutcl2FfbmCounterBlockNumCounters,
                                                         Gcutcl2FfbmCounterRegAddr,
                                                         gfx12_cntx_prim::mc_select_value,
                                                         CounterBlockUtcl2Attr,
                                                         BLOCK_DELAY_NONE};
static const GpuBlockInfo GcL2tlbCounterBlockInfo       = {"GC_L2TLB",
                                                     __BLOCK_ID(GC_L2TLB),
                                                     GcL2tlbCounterBlockNumInstances,
                                                     GcL2tlbCounterBlockMaxEvent,
                                                     GcL2tlbCounterBlockNumCounters,
                                                     GcL2tlbCounterRegAddr,
                                                     gfx12_cntx_prim::mc_select_value,
                                                     CounterBlockUtcl2Attr,
                                                     BLOCK_DELAY_NONE};
static const GpuBlockInfo Gcutcl2NhttlbCounterBlockInfo = {"GC_NHTTLB",
                                                           __BLOCK_ID(GC_NHTTLB),
                                                           Gcutcl2NhttlbCounterBlockNumInstances,
                                                           Gcutcl2NhttlbCounterBlockMaxEvent,
                                                           Gcutcl2NhttlbCounterBlockNumCounters,
                                                           Gcutcl2NhttlbCounterRegAddr,
                                                           gfx12_cntx_prim::mc_select_value,
                                                           CounterBlockUtcl2Attr,
                                                           BLOCK_DELAY_NONE};
static const GpuBlockInfo GcUtcl2CounterBlockInfo       = {"GC_UTCL2",
                                                     __BLOCK_ID(GC_UTCL2),
                                                     Gcutcl2CounterBlockNumInstances,
                                                     Gcutcl2CounterBlockMaxEvent,
                                                     Gcutcl2CounterBlockNumCounters,
                                                     Gcutcl2CounterRegAddr,
                                                     gfx12_cntx_prim::mc_select_value,
                                                     CounterBlockUtcl2Attr,
                                                     BLOCK_DELAY_NONE};
static const GpuBlockInfo GcVml2CounterBlockInfo        = {"GC_VML2",
                                                    __BLOCK_ID(GC_VML2),
                                                    GcmcVmL2CounterBlockNumInstances,
                                                    GcmcVmL2CounterBlockMaxEvent,
                                                    GcmcVmL2CounterBlockNumCounters,
                                                    GcmcVmL2CounterRegAddr,
                                                    gfx12_cntx_prim::mc_select_value,
                                                    CounterBlockUtcl2Attr,
                                                    BLOCK_DELAY_NONE};
// Global blocks (gfx1250): GC_CANE GLARBA GLARBC
static const GpuBlockInfo GcCaneCounterBlockInfo = {"GC_CANE",
                                                    __BLOCK_ID(GC_CANE),
                                                    GcCaneCounterBlockNumInstances,
                                                    GcCaneCounterBlockMaxEvent,
                                                    GcCaneCounterBlockNumCounters,
                                                    GcCaneCounterRegAddr,
                                                    gfx12_cntx_prim::select_value,
                                                    CounterBlockDfltAttr | CounterBlockTcAttr,
                                                    BLOCK_DELAY_NONE};
static const GpuBlockInfo GlarbaCounterBlockInfo = {
    "GLARBA",
    __BLOCK_ID(GLARBA),
    GlarbaCounterBlockNumInstances,
    GlarbaCounterBlockMaxEvent,
    GlarbaCounterBlockNumCounters,
    GlarbaCounterRegAddr,
    gfx12_cntx_prim::select_value,
    CounterBlockDfltAttr | CounterBlockTcAttr | CounterBlockGlarbAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo GlarbcCounterBlockInfo = {
    "GLARBC",
    __BLOCK_ID(GLARBC),
    GlarbcCounterBlockNumInstances,
    GlarbcCounterBlockMaxEvent,
    GlarbcCounterBlockNumCounters,
    GlarbcCounterRegAddr,
    gfx12_cntx_prim::select_value,
    CounterBlockDfltAttr | CounterBlockTcAttr | CounterBlockGlarbAttr,
    BLOCK_DELAY_NONE};
#else
static const GpuBlockInfo   Gl2aCounterBlockInfo      = {"GL2A",
                                                  __BLOCK_ID_HSA(GL2A),
                                                  Gl2aCounterBlockNumInstances,
                                                  Gl2aCounterBlockMaxEvent,
                                                  Gl2aCounterBlockNumCounters,
                                                  Gl2aCounterRegAddr,
                                                  gfx12_cntx_prim::select_value,
                                                  CounterBlockDfltAttr | CounterBlockTcAttr,
                                                  BLOCK_DELAY_NONE};
static const GpuBlockInfo   Gl2cCounterBlockInfo      = {"GL2C",
                                                  __BLOCK_ID_HSA(GL2C),
                                                  Gl2cCounterBlockNumInstances,
                                                  Gl2cCounterBlockMaxEvent,
                                                  Gl2cCounterBlockNumCounters,
                                                  Gl2cCounterRegAddr,
                                                  gfx12_cntx_prim::select_value,
                                                  CounterBlockDfltAttr | CounterBlockTcAttr,
                                                  BLOCK_DELAY_NONE};
static const GpuBlockInfo   GcUtcl2CounterBlockInfo   = {"GC_UTCL2",
                                                     __BLOCK_ID(GC_UTCL2),
                                                     Gcutcl2CounterBlockNumInstances,
                                                     Gcutcl2CounterBlockMaxEvent,
                                                     Gcutcl2CounterBlockNumCounters,
                                                     Gcutcl2CounterRegAddr,
                                                     gfx12_cntx_prim::mc_select_value,
                                                     CounterBlockRpbAttr | CounterBlockAidAttr,
                                                     BLOCK_DELAY_NONE};
static const GpuBlockInfo   GcVml2CounterBlockInfo    = {"GC_VML2",
                                                    __BLOCK_ID(GC_VML2),
                                                    GcmcVmL2CounterBlockNumInstances,
                                                    GcmcVmL2CounterBlockMaxEvent,
                                                    GcmcVmL2CounterBlockNumCounters,
                                                    GcmcVmL2CounterRegAddr,
                                                    gfx12_cntx_prim::mc_select_value,
                                                    CounterBlockRpbAttr | CounterBlockAidAttr,
                                                    BLOCK_DELAY_NONE};
static const GpuBlockInfo   GcVml2SpmCounterBlockInfo = {"GC_VML2_SPM",
                                                       __BLOCK_ID(GC_VML2_SPM),
                                                       GcmcVmL2CounterBlockNumInstances,
                                                       Gcvml2CounterBlockMaxEvent,
                                                       Gcvml2CounterBlockNumCounters,
                                                       Gcvml2CounterRegAddr,
                                                       gfx12_cntx_prim::select_value,
                                                       CounterBlockDfltAttr,
                                                       BLOCK_DELAY_NONE};
#endif
static const GpuBlockInfo ChaCounterBlockInfo  = {"CHA",
                                                 __BLOCK_ID(CHA),
                                                 ChaCounterBlockNumInstances,
                                                 ChaCounterBlockMaxEvent,
                                                 ChaCounterBlockNumCounters,
                                                 ChaCounterRegAddr,
                                                 gfx12_cntx_prim::select_value,
                                                 CounterBlockDfltAttr | CounterBlockTcAttr,
                                                 BLOCK_DELAY_NONE};
static const GpuBlockInfo ChcCounterBlockInfo  = {"CHC",
                                                 __BLOCK_ID(CHC),
                                                 ChcCounterBlockNumInstances,
                                                 ChcCounterBlockMaxEvent,
                                                 ChcCounterBlockNumCounters,
                                                 ChcCounterRegAddr,
                                                 gfx12_cntx_prim::select_value,
                                                 CounterBlockDfltAttr | CounterBlockTcAttr,
                                                 BLOCK_DELAY_NONE};
static const GpuBlockInfo CpcCounterBlockInfo  = {"CPC",
                                                 __BLOCK_ID_HSA(CPC),
                                                 CpcCounterBlockNumInstances,
                                                 CpcCounterBlockMaxEvent,
                                                 CpcCounterBlockNumCounters,
                                                 CpcCounterRegAddr,
                                                 gfx12_cntx_prim::select_value,
                                                 CounterBlockDfltAttr | CounterBlockSpmGlobalAttr,
                                                 BLOCK_DELAY_NONE};
static const GpuBlockInfo CpfCounterBlockInfo  = {"CPF",
                                                 __BLOCK_ID_HSA(CPF),
                                                 CpfCounterBlockNumInstances,
                                                 CpfCounterBlockMaxEvent,
                                                 CpfCounterBlockNumCounters,
                                                 CpfCounterRegAddr,
                                                 gfx12_cntx_prim::select_value,
                                                 CounterBlockDfltAttr | CounterBlockSpmGlobalAttr,
                                                 BLOCK_DELAY_NONE};
static const GpuBlockInfo CpgCounterBlockInfo  = {"CPG",
                                                 __BLOCK_ID(CPG),
                                                 CpgCounterBlockNumInstances,
                                                 CpgCounterBlockMaxEvent,
                                                 CpgCounterBlockNumCounters,
                                                 CpgCounterRegAddr,
                                                 gfx12_cntx_prim::select_value,
                                                 CounterBlockDfltAttr | CounterBlockSpmGlobalAttr,
                                                 BLOCK_DELAY_NONE};
static const GpuBlockInfo GcrCounterBlockInfo  = {"GCR",
                                                 __BLOCK_ID_HSA(GCR),
                                                 GcrCounterBlockNumInstances,
                                                 GcrCounterBlockMaxEvent,
                                                 GcrCounterBlockNumCounters,
                                                 GcrCounterRegAddr,
                                                 gfx12_cntx_prim::select_value,
                                                 CounterBlockDfltAttr | CounterBlockTcAttr,
                                                 BLOCK_DELAY_NONE};
static const GpuBlockInfo GceaCounterBlockInfo = {"GCEA",
                                                  __BLOCK_ID_HSA(GCEA),
                                                  GcEaCpwdCounterBlockNumInstances,
                                                  GcEaCpwdCounterBlockMaxEvent,
                                                  GcEaCpwdCounterBlockNumCounters,
                                                  GcEaCpwdCounterRegAddr,
                                                  gfx12_cntx_prim::select_value,
                                                  CounterBlockDfltAttr,
                                                  BLOCK_DELAY_NONE};
static const GpuBlockInfo GrbmCounterBlockInfo = {"GRBM",
                                                  __BLOCK_ID_HSA(GRBM),
                                                  GrbmCounterBlockNumInstances,
                                                  GrbmCounterBlockMaxEvent,
                                                  GrbmCounterBlockNumCounters,
                                                  GrbmCounterRegAddr,
                                                  gfx12_cntx_prim::select_value,
                                                  CounterBlockDfltAttr | CounterBlockGRBMAttr,
                                                  BLOCK_DELAY_NONE};
static const GpuBlockInfo RlcCounterBlockInfo  = {"RLC",
                                                 __BLOCK_ID(RLC),
                                                 RlcCounterBlockNumInstances,
                                                 RlcCounterBlockMaxEvent,
                                                 RlcCounterBlockNumCounters,
                                                 RlcCounterRegAddr,
                                                 gfx12_cntx_prim::select_value,
                                                 CounterBlockDfltAttr,
                                                 BLOCK_DELAY_NONE};
static const GpuBlockInfo SdmaCounterBlockInfo = {
    "SDMA",
    __BLOCK_ID_HSA(SDMA),
    SdmaCounterBlockNumInstances,
    SdmaCounterBlockMaxEvent,
    SdmaCounterBlockNumCounters,
    SdmaCounterRegAddr,
    gfx12_cntx_prim::select_value,
    CounterBlockDfltAttr | CounterBlockExplInstAttr | CounterBlockSpmGlobalAttr,
    BLOCK_DELAY_NONE};
// SE blocks: EA_SE GL2A GL2C GRBMH SPI SQG UTCL1
//   (Grphics only - not supported in ROCm): GE GL1XA GL1XC PA PC WGS
#if GFX12_VARIANT != GFX12_VARIANT_1250
static const GpuBlockInfo GceaSeCounterBlockInfo = {"GCEA_SE",
                                                    __BLOCK_ID(GCEA_SE),
                                                    GcEaSeCounterBlockNumInstances,
                                                    GcEaSeCounterBlockMaxEvent,
                                                    GcEaSeCounterBlockNumCounters,
                                                    GcEaSeCounterRegAddr,
                                                    gfx12_cntx_prim::select_value,
                                                    CounterBlockSeAttr,
                                                    BLOCK_DELAY_NONE};
#endif
static const GpuBlockInfo GrbmhCounterBlockInfo   = {"GRBMH",
                                                   __BLOCK_ID(GRBMH),
                                                   GrbmhCounterBlockNumInstances,
                                                   GrbmhCounterBlockMaxEvent,
                                                   GrbmhCounterBlockNumCounters,
                                                   GrbmhCounterRegAddr,
                                                   gfx12_cntx_prim::select_value,
                                                   CounterBlockSeAttr,
                                                   BLOCK_DELAY_NONE};
static const GpuBlockInfo SpiCounterBlockInfo     = {"SPI",
                                                 __BLOCK_ID_HSA(SPI),
                                                 SpiCounterBlockNumInstances,
                                                 SpiCounterBlockMaxEvent,
                                                 SpiCounterBlockNumCounters,
                                                 SpiCounterRegAddr,
                                                 gfx12_cntx_prim::select_value,
                                                 CounterBlockSeAttr | CounterBlockSPIAttr,
                                                 BLOCK_DELAY_NONE};
static const GpuBlockInfo SqgCounterBlockInfo     = {"SQG",
                                                 __BLOCK_ID(SQG),
                                                 SqgCounterBlockNumInstances,
                                                 SqgCounterBlockMaxEvent,
                                                 SqgCounterBlockNumCounters,
                                                 SqgCounterRegAddr,
                                                 gfx12_cntx_prim::sq_select_value,
                                                 CounterBlockSeAttr | CounterBlockSqAttr,
                                                 BLOCK_DELAY_NONE};
static const GpuBlockInfo GcUtcl1CounterBlockInfo = {"GC_UTCL1",
                                                     __BLOCK_ID(GC_UTCL1),
                                                     Utcl1CounterBlockNumInstances,
                                                     Utcl1CounterBlockMaxEvent,
                                                     Utcl1CounterBlockNumCounters,
                                                     Utcl1CounterRegAddr,
                                                     gfx12_cntx_prim::select_value,
                                                     CounterBlockSeAttr,
                                                     BLOCK_DELAY_NONE};
#if GFX12_VARIANT == GFX12_VARIANT_1250
// SE blocks (gfx1250): GL1A GL1C (moved from SA)
static const GpuBlockInfo Gl1aCounterBlockInfo = {"GL1A",
                                                  __BLOCK_ID_HSA(GL1A),
                                                  Gl1aCounterBlockNumInstances,
                                                  Gl1aCounterBlockMaxEvent,
                                                  Gl1aCounterBlockNumCounters,
                                                  Gl1aCounterRegAddr,
                                                  gfx12_cntx_prim::select_value,
                                                  CounterBlockSeAttr | CounterBlockTcAttr,
                                                  BLOCK_DELAY_NONE};
static const GpuBlockInfo Gl1cCounterBlockInfo = {"GL1C",
                                                  __BLOCK_ID_HSA(GL1C),
                                                  Gl1cCounterBlockNumInstances,
                                                  Gl1cCounterBlockMaxEvent,
                                                  Gl1cCounterBlockNumCounters,
                                                  Gl1cCounterRegAddr,
                                                  gfx12_cntx_prim::select_value,
                                                  CounterBlockSeAttr | CounterBlockTcAttr,
                                                  BLOCK_DELAY_NONE};
#endif
// SA blocks: GL1A GL1C
//   (Grphics only - not supported in ROCm): CB DB SC SX
//   (Not enabled for gfx12): GL1CG
#if GFX12_VARIANT != GFX12_VARIANT_1250
static const GpuBlockInfo Gl1aCounterBlockInfo = {
    "GL1A",
    __BLOCK_ID_HSA(GL1A),
    Gl1aCounterBlockNumInstances,
    Gl1aCounterBlockMaxEvent,
    Gl1aCounterBlockNumCounters,
    Gl1aCounterRegAddr,
    gfx12_cntx_prim::select_value,
    CounterBlockSeAttr | CounterBlockSaAttr | CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo Gl1cCounterBlockInfo = {
    "GL1C",
    __BLOCK_ID_HSA(GL1C),
    Gl1cCounterBlockNumInstances,
    Gl1cCounterBlockMaxEvent,
    Gl1cCounterBlockNumCounters,
    Gl1cCounterRegAddr,
    gfx12_cntx_prim::select_value,
    CounterBlockSeAttr | CounterBlockSaAttr | CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
#endif
// WGP blocks: SQC TA TCP TD
static const GpuBlockInfo SqcCounterBlockInfo = {
    "SQ",
    __BLOCK_ID_HSA(SQ),
    SqcCounterBlockNumInstances,
    SqcCounterBlockMaxEvent,
    SqcCounterBlockNumCounters,
    SqcCounterRegAddr,
    gfx12_cntx_prim::sq_select_value,
    CounterBlockSeAttr | CounterBlockSaAttr | CounterBlockWgpAttr | CounterBlockSqAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo TaCounterBlockInfo = {
    "TA",
    __BLOCK_ID_HSA(TA),
    TaCounterBlockNumInstances,
    TaCounterBlockMaxEvent,
    TaCounterBlockNumCounters,
    TaCounterRegAddr,
    gfx12_cntx_prim::select_value,
    CounterBlockSeAttr | CounterBlockSaAttr | CounterBlockWgpAttr | CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo TdCounterBlockInfo = {
    "TD",
    __BLOCK_ID_HSA(TD),
    TdCounterBlockNumInstances,
    TdCounterBlockMaxEvent,
    TdCounterBlockNumCounters,
    TdCounterRegAddr,
    gfx12_cntx_prim::select_value,
    CounterBlockSeAttr | CounterBlockSaAttr | CounterBlockWgpAttr | CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
static const GpuBlockInfo TcpCounterBlockInfo = {
    "TCP",
    __BLOCK_ID_HSA(TCP),
    TcpCounterBlockNumInstances,
    TcpCounterBlockMaxEvent,
    TcpCounterBlockNumCounters,
    TcpCounterRegAddr,
    gfx12_cntx_prim::select_value,
    CounterBlockSeAttr | CounterBlockSaAttr | CounterBlockWgpAttr | CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
}  // namespace gfx1200/gfx1250

namespace gfx1201
{
static const GpuBlockInfo Gl2cCounterBlockInfo   = {"GL2C",
                                                  __BLOCK_ID_HSA(GL2C),
                                                  gfx1201::Gl2cCounterBlockNumInstances,
                                                  Gl2cCounterBlockMaxEvent,
                                                  Gl2cCounterBlockNumCounters,
                                                  Gl2cCounterRegAddr,
                                                  gfx12_cntx_prim::select_value,
                                                  CounterBlockDfltAttr | CounterBlockTcAttr,
                                                  BLOCK_DELAY_NONE};
static const GpuBlockInfo ChcCounterBlockInfo    = {"CHC",
                                                 __BLOCK_ID(CHC),
                                                 gfx1201::ChcCounterBlockNumInstances,
                                                 ChcCounterBlockMaxEvent,
                                                 ChcCounterBlockNumCounters,
                                                 ChcCounterRegAddr,
                                                 gfx12_cntx_prim::select_value,
                                                 CounterBlockDfltAttr | CounterBlockTcAttr,
                                                 BLOCK_DELAY_NONE};
static const GpuBlockInfo GceaCounterBlockInfo   = {"GCEA",
                                                  __BLOCK_ID_HSA(GCEA),
                                                  gfx1201::GcEaCpwdCounterBlockNumInstances,
                                                  GcEaCpwdCounterBlockMaxEvent,
                                                  GcEaCpwdCounterBlockNumCounters,
                                                  GcEaCpwdCounterRegAddr,
                                                  gfx12_cntx_prim::select_value,
                                                  CounterBlockDfltAttr,
                                                  BLOCK_DELAY_NONE};
static const GpuBlockInfo GceaSeCounterBlockInfo = {"GCEA_SE",
                                                    __BLOCK_ID(GCEA_SE),
                                                    gfx1201::GcEaSeCounterBlockNumInstances,
                                                    GcEaSeCounterBlockMaxEvent,
                                                    GcEaSeCounterBlockNumCounters,
                                                    GcEaSeCounterRegAddr,
                                                    gfx12_cntx_prim::select_value,
                                                    CounterBlockSeAttr,
                                                    BLOCK_DELAY_NONE};
}  // namespace gfx1201

#if GFX12_VARIANT <= GFX12_VARIANT_1201
static const GpuBlockInfo RpbCounterBlockInfo = {"RPB",
                                                 __BLOCK_ID_HSA(RPB),
                                                 1,
                                                 RpbCounterBlockMaxEvent,
                                                 RpbCounterBlockNumCounters,
                                                 RpbCounterRegAddr,
                                                 gfx12_cntx_prim::mc_select_value,
                                                 CounterBlockRpbAttr | CounterBlockAidAttr,
                                                 BLOCK_DELAY_NONE};
#endif

}  // namespace gfx12
}  // namespace gfxip

#endif  // _GFX12_BLOCKTABLE_H_
