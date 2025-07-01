
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

#define REG_INFO_WITH_CTRL(BLOCK, CTRL, INDEX) \
 {REG_32B_ADDR(GC, 0, reg##BLOCK##_PERFCOUNTER##INDEX##_SELECT), CTRL, REG_32B_ADDR(GC, 0, reg##BLOCK##_PERFCOUNTER##INDEX##_LO), REG_32B_ADDR(GC, 0, reg##BLOCK##_PERFCOUNTER##INDEX##_HI)}
#define REG_INFO_WITH_CTRL_1(BLOCK, CTRL) REG_INFO_WITH_CTRL(BLOCK, CTRL, 0)
#define REG_INFO_WITH_CTRL_2(BLOCK, CTRL) REG_INFO_WITH_CTRL_1(BLOCK, CTRL), REG_INFO_WITH_CTRL(BLOCK, CTRL, 1)
#define REG_INFO_WITH_CTRL_3(BLOCK, CTRL) REG_INFO_WITH_CTRL_2(BLOCK, CTRL), REG_INFO_WITH_CTRL(BLOCK, CTRL, 2)
#define REG_INFO_WITH_CTRL_4(BLOCK, CTRL) REG_INFO_WITH_CTRL_3(BLOCK, CTRL), REG_INFO_WITH_CTRL(BLOCK, CTRL, 3)
#define REG_INFO_WITH_CTRL_5(BLOCK, CTRL) REG_INFO_WITH_CTRL_4(BLOCK, CTRL), REG_INFO_WITH_CTRL(BLOCK, CTRL, 4)
#define REG_INFO_WITH_CTRL_6(BLOCK, CTRL) REG_INFO_WITH_CTRL_5(BLOCK, CTRL), REG_INFO_WITH_CTRL(BLOCK, CTRL, 5)
#define REG_INFO_WITH_CTRL_7(BLOCK, CTRL) REG_INFO_WITH_CTRL_6(BLOCK, CTRL), REG_INFO_WITH_CTRL(BLOCK, CTRL, 6)
#define REG_INFO_WITH_CTRL_8(BLOCK, CTRL) REG_INFO_WITH_CTRL_7(BLOCK, CTRL), REG_INFO_WITH_CTRL(BLOCK, CTRL, 7)
#define REG_INFO_1(BLOCK) REG_INFO_WITH_CTRL_1(BLOCK, REG_32B_NULL)
#define REG_INFO_2(BLOCK) REG_INFO_WITH_CTRL_2(BLOCK, REG_32B_NULL)
#define REG_INFO_3(BLOCK) REG_INFO_WITH_CTRL_3(BLOCK, REG_32B_NULL)
#define REG_INFO_4(BLOCK) REG_INFO_WITH_CTRL_4(BLOCK, REG_32B_NULL)
#define REG_INFO_5(BLOCK) REG_INFO_WITH_CTRL_5(BLOCK, REG_32B_NULL)
#define REG_INFO_6(BLOCK) REG_INFO_WITH_CTRL_6(BLOCK, REG_32B_NULL)
#define REG_INFO_7(BLOCK) REG_INFO_WITH_CTRL_7(BLOCK, REG_32B_NULL)
#define REG_INFO_8(BLOCK) REG_INFO_WITH_CTRL_8(BLOCK, REG_32B_NULL)

#define REG_INFO_WITH_CFG(IP, BLOCK, INDEX)				\
 {REG_32B_ADDR(IP, 0, reg##BLOCK##_PERFCOUNTER##INDEX##_CFG), REG_32B_ADDR(IP, 0, reg##BLOCK##_PERFCOUNTER_RSLT_CNTL), REG_32B_ADDR(IP, 0, reg##BLOCK##_PERFCOUNTER_LO), REG_32B_ADDR(IP, 0, reg##BLOCK##_PERFCOUNTER_HI)}
#define REG_INFO_WITH_CFG_1(IP, BLOCK) REG_INFO_WITH_CFG(IP, BLOCK, 0)
#define REG_INFO_WITH_CFG_2(IP, BLOCK) REG_INFO_WITH_CFG_1(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 1)
#define REG_INFO_WITH_CFG_3(IP, BLOCK) REG_INFO_WITH_CFG_2(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 2)
#define REG_INFO_WITH_CFG_4(IP, BLOCK) REG_INFO_WITH_CFG_3(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 3)
#define REG_INFO_WITH_CFG_5(IP, BLOCK) REG_INFO_WITH_CFG_4(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 4)
#define REG_INFO_WITH_CFG_6(IP, BLOCK) REG_INFO_WITH_CFG_5(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 5)
#define REG_INFO_WITH_CFG_7(IP, BLOCK) REG_INFO_WITH_CFG_6(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 6)
#define REG_INFO_WITH_CFG_8(IP, BLOCK) REG_INFO_WITH_CFG_7(IP, BLOCK), REG_INFO_WITH_CFG(IP, BLOCK, 7)

namespace gfxip {
namespace gfx12 {
namespace gfx1200 {
// Counter register info - Auto-generated from chip_offset_byte.h, edit with extra caution
static const CounterRegInfo ChaCounterRegAddr[] = {REG_INFO_4(CHA)};
static const CounterRegInfo ChcCounterRegAddr[] = {REG_INFO_4(CHC)};
static const CounterRegInfo CpcCounterRegAddr[] = {REG_INFO_2(CPC)};
static const CounterRegInfo CpfCounterRegAddr[] = {REG_INFO_2(CPF)};
static const CounterRegInfo CpgCounterRegAddr[] = {REG_INFO_2(CPG)};
static const CounterRegInfo GcmcVmL2CounterRegAddr[] = {REG_INFO_WITH_CFG_8(GC, GCMC_VM_L2)};
static const CounterRegInfo GcrCounterRegAddr[] = {REG_INFO_WITH_CTRL_2(GCR, REG_32B_ADDR(GC, 0, regGCR_GENERAL_CNTL))};
static const CounterRegInfo RpbCounterRegAddr[] = {REG_INFO_WITH_CFG_4(ATHUB, RPB)};
static const CounterRegInfo Gcutcl2CounterRegAddr[] = {REG_INFO_WITH_CFG_4(GC, GCUTCL2)};
// static const CounterRegInfo Gcvml2CounterRegAddr[] = {REG_INFO_2(GCVML2)};
static const CounterRegInfo GcEaCpwdCounterRegAddr[] = {REG_INFO_2(GC_EA_CPWD)};
static const CounterRegInfo GcEaSeCounterRegAddr[] = {REG_INFO_2(GC_EA_SE)};
static const CounterRegInfo Gl1aCounterRegAddr[] = {REG_INFO_4(GL1A)};
static const CounterRegInfo Gl1cCounterRegAddr[] = {REG_INFO_4(GL1C)};
static const CounterRegInfo Gl2aCounterRegAddr[] = {REG_INFO_4(GL2A)};
static const CounterRegInfo Gl2cCounterRegAddr[] = {REG_INFO_4(GL2C)};
static const CounterRegInfo GrbmCounterRegAddr[] = {REG_INFO_2(GRBM)};
static const CounterRegInfo GrbmhCounterRegAddr[] = {REG_INFO_2(GRBMH)};
static const CounterRegInfo RlcCounterRegAddr[] = {REG_INFO_2(RLC)};
static const CounterRegInfo SdmaCounterRegAddr[] = {REG_INFO_2(SDMA0), REG_INFO_2(SDMA1)};
static const CounterRegInfo SpiCounterRegAddr[] = {REG_INFO_6(SPI)};
//static const CounterRegInfo SqcCounterRegAddr[] = {REG_INFO_WITH_CTRL_16(SQ, REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL))};
static const CounterRegInfo SqgCounterRegAddr[] = {REG_INFO_WITH_CTRL_8(SQG, REG_32B_ADDR(GC, 0, regSQG_PERFCOUNTER_CTRL))};
static const CounterRegInfo TaCounterRegAddr[] = {REG_INFO_2(TA)};
static const CounterRegInfo TcpCounterRegAddr[] = {REG_INFO_4(TCP)};
static const CounterRegInfo TdCounterRegAddr[] = {REG_INFO_2(TD)};
static const CounterRegInfo Utcl1CounterRegAddr[] = {REG_INFO_4(UTCL1)};

// Special handling of SQC:
//   SQC only supports 32bit PMC.
//   regSQ_PERFCOUNTER#even_number#_SELECT is used by PMC and SPM
//   regSQ_PERFCOUNTER#odd_number#_SELECT is used by SPM only
static const CounterRegInfo SqcCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER0_SELECT),  REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER0_LO), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER2_SELECT),  REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER1_LO), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER4_SELECT),  REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER2_LO), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER6_SELECT),  REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER3_LO), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER8_SELECT),  REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER4_LO), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER10_SELECT), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER5_LO), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER12_SELECT), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER6_LO), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER14_SELECT), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER7_LO), REG_32B_NULL}};

// Special handling of GCVML2 (SPM only):
static const CounterRegInfo Gcvml2CounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regGCVML2_PERFCOUNTER2_0_SELECT), REG_32B_NULL, REG_32B_ADDR(GC, 0, regGCVML2_PERFCOUNTER2_0_LO), REG_32B_ADDR(GC, 0, regGCVML2_PERFCOUNTER2_0_HI)},
    {REG_32B_ADDR(GC, 0, regGCVML2_PERFCOUNTER2_1_SELECT), REG_32B_NULL, REG_32B_ADDR(GC, 0, regGCVML2_PERFCOUNTER2_1_LO), REG_32B_ADDR(GC, 0, regGCVML2_PERFCOUNTER2_1_HI)}};

// Global blocks: ATCL2 CHA CHC CPC CPF CPG EA FFBM GCR GL2A GL2C GRBM RLC SDMA VML2 UTCL2
//   (Grphics only - not supported in ROCm): GE1 GE2_DIST PH
//   (Grphics only): CPG is for graphics, but it is not physically removed for compute products
//   (Not enabled for gfx12): CHCG GDS GUS
static const GpuBlockInfo Gl2aCounterBlockInfo = {"GL2A", __BLOCK_ID_HSA(GL2A), Gl2aCounterBlockNumInstances, Gl2aCounterBlockMaxEvent, Gl2aCounterBlockNumCounters, Gl2aCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr|CounterBlockTcAttr};
static const GpuBlockInfo Gl2cCounterBlockInfo = {"GL2C", __BLOCK_ID_HSA(GL2C), Gl2cCounterBlockNumInstances, Gl2cCounterBlockMaxEvent, Gl2cCounterBlockNumCounters, Gl2cCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr|CounterBlockTcAttr};
static const GpuBlockInfo Atcl2CounterBlockInfo = {"ATCL2", __BLOCK_ID_HSA(ATCL2)}; // Placeholder now
static const GpuBlockInfo GcFfbmCounterBlockInfo = {"GC_FFBM", __BLOCK_ID(GC_FFBM)};  // Placeholder now
static const GpuBlockInfo RpbCounterBlockInfo = {"RPB", __BLOCK_ID_HSA(RPB), 1, RpbCounterBlockMaxEvent, RpbCounterBlockNumCounters, RpbCounterRegAddr, gfx12_cntx_prim::mc_select_value, CounterBlockRpbAttr|CounterBlockAidAttr};
static const GpuBlockInfo GcUtcl2CounterBlockInfo = {"GC_UTCL2", __BLOCK_ID(GC_UTCL2), 1, Gcutcl2CounterBlockMaxEvent, Gcutcl2CounterBlockNumCounters, Gcutcl2CounterRegAddr, gfx12_cntx_prim::mc_select_value, CounterBlockRpbAttr|CounterBlockAidAttr};
static const GpuBlockInfo GcVml2CounterBlockInfo = {"GC_VML2", __BLOCK_ID(GC_VML2), 1, GcmcVmL2CounterBlockMaxEvent, GcmcVmL2CounterBlockNumCounters, GcmcVmL2CounterRegAddr, gfx12_cntx_prim::mc_select_value, CounterBlockRpbAttr|CounterBlockAidAttr};
static const GpuBlockInfo GcVml2SpmCounterBlockInfo = {"GC_VML2_SPM", __BLOCK_ID(GC_VML2_SPM), 1, Gcvml2CounterBlockMaxEvent, Gcvml2CounterBlockNumCounters, Gcvml2CounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr};
static const GpuBlockInfo ChaCounterBlockInfo = {"CHA", __BLOCK_ID(CHA), ChaCounterBlockNumInstances, ChaCounterBlockMaxEvent, ChaCounterBlockNumCounters, ChaCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr|CounterBlockTcAttr};
static const GpuBlockInfo ChcCounterBlockInfo = {"CHC", __BLOCK_ID(CHC), ChcCounterBlockNumInstances, ChcCounterBlockMaxEvent, ChcCounterBlockNumCounters, ChcCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr|CounterBlockTcAttr};
static const GpuBlockInfo CpcCounterBlockInfo = {"CPC", __BLOCK_ID_HSA(CPC), CpcCounterBlockNumInstances, CpcCounterBlockMaxEvent, CpcCounterBlockNumCounters, CpcCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr|CounterBlockSpmGlobalAttr, NULL, SPM_GLOBAL_BLOCK_NAME_CPC};
static const GpuBlockInfo CpfCounterBlockInfo = {"CPF", __BLOCK_ID_HSA(CPF), CpfCounterBlockNumInstances, CpfCounterBlockMaxEvent, CpfCounterBlockNumCounters, CpfCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr|CounterBlockSpmGlobalAttr, NULL, SPM_GLOBAL_BLOCK_NAME_CPF};
static const GpuBlockInfo CpgCounterBlockInfo = {"CPG", __BLOCK_ID(CPG), CpgCounterBlockNumInstances, CpgCounterBlockMaxEvent, CpgCounterBlockNumCounters, CpgCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr|CounterBlockSpmGlobalAttr, NULL, SPM_GLOBAL_BLOCK_NAME_CPG};
static const GpuBlockInfo GcrCounterBlockInfo = {"GCR", __BLOCK_ID_HSA(GCR), GcrCounterBlockNumInstances, GcrCounterBlockMaxEvent, GcrCounterBlockNumCounters, GcrCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr|CounterBlockTcAttr};
static const GpuBlockInfo GceaCounterBlockInfo = {"GCEA", __BLOCK_ID_HSA(GCEA), GcEaCpwdCounterBlockNumInstances, GcEaCpwdCounterBlockMaxEvent, GcEaCpwdCounterBlockNumCounters, GcEaCpwdCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr};
static const GpuBlockInfo GrbmCounterBlockInfo = {"GRBM", __BLOCK_ID_HSA(GRBM), GrbmCounterBlockNumInstances, GrbmCounterBlockMaxEvent, GrbmCounterBlockNumCounters, GrbmCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr|CounterBlockGRBMAttr};
static const GpuBlockInfo RlcCounterBlockInfo = {"RLC", __BLOCK_ID(RLC), RlcCounterBlockNumInstances, RlcCounterBlockMaxEvent, RlcCounterBlockNumCounters, RlcCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr};
static const GpuBlockInfo SdmaCounterBlockInfo = {"SDMA", __BLOCK_ID_HSA(SDMA), SdmaCounterBlockNumInstances, SdmaCounterBlockMaxEvent, SdmaCounterBlockNumCounters, SdmaCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr|CounterBlockExplInstAttr|CounterBlockSpmGlobalAttr, NULL, SPM_GLOBAL_BLOCK_NAME_SDMA};
// SE blocks: EA_SE GL2A GL2C GRBMH SPI SQG UTCL1
//   (Grphics only - not supported in ROCm): GE GL1XA GL1XC PA PC WGS
static const GpuBlockInfo GceaSeCounterBlockInfo = {"GCEA_SE", __BLOCK_ID(GCEA_SE), GcEaSeCounterBlockNumInstances, GcEaSeCounterBlockMaxEvent, GcEaSeCounterBlockNumCounters, GcEaSeCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockSeAttr};
static const GpuBlockInfo GrbmhCounterBlockInfo = {"GRBMH", __BLOCK_ID(GRBMH), GrbmhCounterBlockNumInstances, GrbmhCounterBlockMaxEvent, GrbmhCounterBlockNumCounters, GrbmhCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockSeAttr};
static const GpuBlockInfo SpiCounterBlockInfo = {"SPI", __BLOCK_ID_HSA(SPI), SpiCounterBlockNumInstances, SpiCounterBlockMaxEvent, SpiCounterBlockNumCounters, SpiCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockSeAttr|CounterBlockSPIAttr, NULL, SPM_SE_BLOCK_NAME_SPI};
static const GpuBlockInfo SqgCounterBlockInfo = {"SQG", __BLOCK_ID(SQG), SqgCounterBlockNumInstances, SqgCounterBlockMaxEvent, SqgCounterBlockNumCounters, SqgCounterRegAddr, gfx12_cntx_prim::sq_select_value, CounterBlockSeAttr|CounterBlockSqAttr, NULL, SPM_SE_BLOCK_NAME_SQG};
static const GpuBlockInfo GcUtcl1CounterBlockInfo = {"GC_UTCL1", __BLOCK_ID(GC_UTCL1), Utcl1CounterBlockNumInstances, Utcl1CounterBlockMaxEvent, Utcl1CounterBlockNumCounters, Utcl1CounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockSeAttr, NULL, SPM_SE_BLOCK_NAME_UTCL1};
// SA blocks: GL1A GL1C
//   (Grphics only - not supported in ROCm): CB DB SC SX
//   (Not enabled for gfx12): GL1CG
static const GpuBlockInfo Gl1aCounterBlockInfo = {"GL1A", __BLOCK_ID_HSA(GL1A), Gl1aCounterBlockNumInstances, Gl1aCounterBlockMaxEvent, Gl1aCounterBlockNumCounters, Gl1aCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockSeAttr|CounterBlockSaAttr|CounterBlockTcAttr};
static const GpuBlockInfo Gl1cCounterBlockInfo = {"GL1C", __BLOCK_ID_HSA(GL1C), Gl1cCounterBlockNumInstances, Gl1cCounterBlockMaxEvent, Gl1cCounterBlockNumCounters, Gl1cCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockSeAttr|CounterBlockSaAttr|CounterBlockTcAttr};
// WGP blocks: SQC TA TCP TD
static const GpuBlockInfo SqcCounterBlockInfo = {"SQ", __BLOCK_ID_HSA(SQ), SqcCounterBlockNumInstances, SqcCounterBlockMaxEvent, SqcCounterBlockNumCounters, SqcCounterRegAddr, gfx12_cntx_prim::sq_select_value, CounterBlockSeAttr|CounterBlockSaAttr|CounterBlockWgpAttr|CounterBlockSqAttr, NULL, SPM_SE_BLOCK_NAME_SQC};
static const GpuBlockInfo TaCounterBlockInfo = {"TA", __BLOCK_ID_HSA(TA), TaCounterBlockNumInstances, TaCounterBlockMaxEvent, TaCounterBlockNumCounters, TaCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockSeAttr|CounterBlockSaAttr|CounterBlockWgpAttr|CounterBlockTcAttr, NULL/*TaBlockDelayInfo*/, SPM_SE_BLOCK_NAME_TA};
static const GpuBlockInfo TdCounterBlockInfo = {"TD", __BLOCK_ID_HSA(TD), TdCounterBlockNumInstances, TdCounterBlockMaxEvent, TdCounterBlockNumCounters, TdCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockSeAttr|CounterBlockSaAttr|CounterBlockWgpAttr|CounterBlockTcAttr, NULL/*TdBlockDelayInfo*/, SPM_SE_BLOCK_NAME_TD};
static const GpuBlockInfo TcpCounterBlockInfo = {"TCP", __BLOCK_ID_HSA(TCP), TcpCounterBlockNumInstances, TcpCounterBlockMaxEvent, TcpCounterBlockNumCounters, TcpCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockSeAttr|CounterBlockSaAttr|CounterBlockWgpAttr|CounterBlockTcAttr, NULL/*TdBlockDelayInfo*/, SPM_SE_BLOCK_NAME_TCP};
}  // namespace gfx1200

namespace gfx1201 {
static const GpuBlockInfo Gl2cCounterBlockInfo = {"GL2C", __BLOCK_ID_HSA(GL2C), gfx1201::Gl2cCounterBlockNumInstances, Gl2cCounterBlockMaxEvent, Gl2cCounterBlockNumCounters, Gl2cCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr|CounterBlockTcAttr};
static const GpuBlockInfo ChcCounterBlockInfo = {"CHC", __BLOCK_ID(CHC), gfx1201::ChcCounterBlockNumInstances, ChcCounterBlockMaxEvent, ChcCounterBlockNumCounters, ChcCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr|CounterBlockTcAttr};
static const GpuBlockInfo GceaCounterBlockInfo = {"GCEA", __BLOCK_ID_HSA(GCEA), gfx1201::GcEaCpwdCounterBlockNumInstances, GcEaCpwdCounterBlockMaxEvent, GcEaCpwdCounterBlockNumCounters, GcEaCpwdCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockDfltAttr};
static const GpuBlockInfo GceaSeCounterBlockInfo = {"GCEA_SE", __BLOCK_ID(GCEA_SE), gfx1201::GcEaSeCounterBlockNumInstances, GcEaSeCounterBlockMaxEvent, GcEaSeCounterBlockNumCounters, GcEaSeCounterRegAddr, gfx12_cntx_prim::select_value, CounterBlockSeAttr};
}  // namespace gfx1201

}  // namespace gfx12
}  // namespace gfxip

#endif  // _GFX12_BLOCKTABLE_H_
