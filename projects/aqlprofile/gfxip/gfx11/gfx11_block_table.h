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

#ifndef _GFX11_BLOCKTABLE_H_
#define _GFX11_BLOCKTABLE_H_

namespace gfxip {
namespace gfx11 {

/*
 * CPC    CORRECT
 */
static const CounterRegInfo CpcCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regCPC_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regCPC_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, regCPC_PERFCOUNTER0_HI)},
    {REG_32B_ADDR(GC, 0, regCPC_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regCPC_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, regCPC_PERFCOUNTER1_HI)}};

/*
 * CPF    CORRECT
 */
static const CounterRegInfo CpfCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regCPF_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regCPF_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, regCPF_PERFCOUNTER0_HI)},
    {REG_32B_ADDR(GC, 0, regCPF_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regCPF_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, regCPF_PERFCOUNTER1_HI)}};

/*
 * GDS     CORRECT
 */
static const CounterRegInfo GdsCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regGDS_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGDS_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, regGDS_PERFCOUNTER0_HI)},
    {REG_32B_ADDR(GC, 0, regGDS_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGDS_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, regGDS_PERFCOUNTER1_HI)},
    {REG_32B_ADDR(GC, 0, regGDS_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGDS_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, regGDS_PERFCOUNTER2_HI)},
    {REG_32B_ADDR(GC, 0, regGDS_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGDS_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, regGDS_PERFCOUNTER3_HI)}};
/*
 * GRBM     CORRECT
 */
static const CounterRegInfo GrbmCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regGRBM_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGRBM_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, regGRBM_PERFCOUNTER0_HI)},
    {REG_32B_ADDR(GC, 0, regGRBM_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGRBM_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, regGRBM_PERFCOUNTER1_HI)}};

/*
 * GRBM_SE    CORRECT
 */
static const CounterRegInfo GrbmSeCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regGRBM_SE0_PERFCOUNTER_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGRBM_SE0_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, regGRBM_SE0_PERFCOUNTER_HI)},
    {REG_32B_ADDR(GC, 0, regGRBM_SE1_PERFCOUNTER_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGRBM_SE1_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, regGRBM_SE1_PERFCOUNTER_HI)},
    {REG_32B_ADDR(GC, 0, regGRBM_SE2_PERFCOUNTER_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGRBM_SE2_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, regGRBM_SE2_PERFCOUNTER_HI)},
    {REG_32B_ADDR(GC, 0, regGRBM_SE3_PERFCOUNTER_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGRBM_SE3_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, regGRBM_SE3_PERFCOUNTER_HI)},
    {REG_32B_ADDR(GC, 0, regGRBM_SE4_PERFCOUNTER_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGRBM_SE4_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, regGRBM_SE4_PERFCOUNTER_HI)},
    {REG_32B_ADDR(GC, 0, regGRBM_SE5_PERFCOUNTER_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGRBM_SE5_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, regGRBM_SE5_PERFCOUNTER_HI)},
    {REG_32B_ADDR(GC, 0, regGRBM_SE6_PERFCOUNTER_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGRBM_SE6_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, regGRBM_SE6_PERFCOUNTER_HI)}};

/*
 * SPI        CORRECT
 */
static const CounterRegInfo SpiCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER0_HI)},
    {REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER1_HI)},
    {REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER2_HI)},
    {REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER3_HI)},
    {REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER4_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER4_LO), REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER4_HI)},
    {REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER5_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER5_LO), REG_32B_ADDR(GC, 0, regSPI_PERFCOUNTER5_HI)}};
/*
 * SQ        CORRECT
 */
static const CounterRegInfo SqCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER0_SELECT), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER0_LO), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER2_SELECT), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER1_LO), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER4_SELECT), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER2_LO), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER6_SELECT), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER3_LO), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER8_SELECT), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER4_LO), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER10_SELECT), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER5_LO), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER12_SELECT), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER6_LO), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER14_SELECT), REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, regSQ_PERFCOUNTER7_LO), REG_32B_NULL}};
/*
 * SX       CORRECT
 */
static const CounterRegInfo SxCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regSX_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regSX_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, regSX_PERFCOUNTER0_HI)},
    {REG_32B_ADDR(GC, 0, regSX_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regSX_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, regSX_PERFCOUNTER1_HI)},
    {REG_32B_ADDR(GC, 0, regSX_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regSX_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, regSX_PERFCOUNTER2_HI)},
    {REG_32B_ADDR(GC, 0, regSX_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regSX_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, regSX_PERFCOUNTER3_HI)}};

/*
 * GCEA
 */
static const CounterRegInfo GceaCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regGCEA_PERFCOUNTER0_CFG),
     REG_32B_ADDR(GC, 0, regGCEA_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(GC, 0, regGCEA_PERFCOUNTER_LO), REG_32B_ADDR(GC, 0, regGCEA_PERFCOUNTER_HI)},
    {REG_32B_ADDR(GC, 0, regGCEA_PERFCOUNTER1_CFG),
     REG_32B_ADDR(GC, 0, regGCEA_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(GC, 0, regGCEA_PERFCOUNTER_LO), REG_32B_ADDR(GC, 0, regGCEA_PERFCOUNTER_HI)}};

// Define GFX10 specific blocks table entries like GC caches blocks
/*
 * GCR    CORRECT
 */
static const CounterRegInfo GcrCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regGCR_PERFCOUNTER0_SELECT), REG_32B_ADDR(GC, 0, regGCR_GENERAL_CNTL),
     REG_32B_ADDR(GC, 0, regGCR_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, regGCR_PERFCOUNTER0_HI)},
    {REG_32B_ADDR(GC, 0, regGCR_PERFCOUNTER1_SELECT), REG_32B_ADDR(GC, 0, regGCR_GENERAL_CNTL),
     REG_32B_ADDR(GC, 0, regGCR_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, regGCR_PERFCOUNTER1_HI)}};

/*
 * TCP
 */
static const CounterRegInfo TcpCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regTCP_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regTCP_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, regTCP_PERFCOUNTER0_HI)},
    {REG_32B_ADDR(GC, 0, regTCP_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regTCP_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, regTCP_PERFCOUNTER1_HI)},
    {REG_32B_ADDR(GC, 0, regTCP_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regTCP_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, regTCP_PERFCOUNTER2_HI)},
    {REG_32B_ADDR(GC, 0, regTCP_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regTCP_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, regTCP_PERFCOUNTER3_HI)}};
/*
 * GL1A    CORRECT
 */
static const CounterRegInfo Gl1aCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regGL1A_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL1A_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, regGL1A_PERFCOUNTER0_HI)},
    {REG_32B_ADDR(GC, 0, regGL1A_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL1A_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, regGL1A_PERFCOUNTER1_HI)},
    {REG_32B_ADDR(GC, 0, regGL1A_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL1A_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, regGL1A_PERFCOUNTER2_HI)},
    {REG_32B_ADDR(GC, 0, regGL1A_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL1A_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, regGL1A_PERFCOUNTER3_HI)},
};

/*
 * GL1C    CORRECT
 */
static const CounterRegInfo Gl1cCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regGL1C_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL1C_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, regGL1C_PERFCOUNTER0_HI)},
    {REG_32B_ADDR(GC, 0, regGL1C_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL1C_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, regGL1C_PERFCOUNTER1_HI)},
    {REG_32B_ADDR(GC, 0, regGL1C_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL1C_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, regGL1C_PERFCOUNTER2_HI)},
    {REG_32B_ADDR(GC, 0, regGL1C_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL1C_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, regGL1C_PERFCOUNTER3_HI)},
};

/*
 * GL2A     CORRECT
 */
static const CounterRegInfo Gl2aCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regGL2A_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL2A_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, regGL2A_PERFCOUNTER0_HI)},
    {REG_32B_ADDR(GC, 0, regGL2A_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL2A_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, regGL2A_PERFCOUNTER1_HI)},
    {REG_32B_ADDR(GC, 0, regGL2A_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL2A_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, regGL2A_PERFCOUNTER2_HI)},
    {REG_32B_ADDR(GC, 0, regGL2A_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL2A_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, regGL2A_PERFCOUNTER3_HI)},
};

/*
 * GL2C     CORRECT
 */
static const CounterRegInfo Gl2cCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regGL2C_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL2C_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, regGL2C_PERFCOUNTER0_HI)},
    {REG_32B_ADDR(GC, 0, regGL2C_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL2C_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, regGL2C_PERFCOUNTER1_HI)},
    {REG_32B_ADDR(GC, 0, regGL2C_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL2C_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, regGL2C_PERFCOUNTER2_HI)},
    {REG_32B_ADDR(GC, 0, regGL2C_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regGL2C_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, regGL2C_PERFCOUNTER3_HI)},
};

/*
 * GUS        ????? need more investigations
 */
static const CounterRegInfo GusCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regGUS_PERFCOUNTER0_CFG),
     REG_32B_ADDR(GC, 0, regGUS_PERFCOUNTER_RSLT_CNTL), REG_32B_ADDR(GC, 0, regGUS_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, regGUS_PERFCOUNTER_HI)},
    {REG_32B_ADDR(GC, 0, regGUS_PERFCOUNTER1_CFG),
     REG_32B_ADDR(GC, 0, regGUS_PERFCOUNTER_RSLT_CNTL), REG_32B_ADDR(GC, 0, regGUS_PERFCOUNTER2_LO),
     REG_32B_ADDR(GC, 0, regGUS_PERFCOUNTER2_HI)},
};

/*
 * TA            CORRECT
 */
static const CounterRegInfo TaCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, regTA_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regTA_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, regTA_PERFCOUNTER0_HI)},
    {REG_32B_ADDR(GC, 0, regTA_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, regTA_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, regTA_PERFCOUNTER1_HI)}};

// Counter block CPC
static const GpuBlockInfo CpcCounterBlockInfo = {
    "CPC",
    CpcCounterBlockId,
    1,
    CpcCounterBlockMaxEvent,
    CpcCounterBlockNumCounters,
    CpcCounterRegAddr,
    gfx11_cntx_prim::select_value_CPC_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockSpmGlobalAttr,
    NULL /*CpcBlockDelayInfo*/,
    SPM_GLOBAL_BLOCK_NAME_CPC};
// Counter block CPF
static const GpuBlockInfo CpfCounterBlockInfo = {
    "CPF",
    CpfCounterBlockId,
    1,
    CpfCounterBlockMaxEvent,
    CpfCounterBlockNumCounters,
    CpfCounterRegAddr,
    gfx11_cntx_prim::select_value_CPF_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockSpmGlobalAttr,
    NULL /*CpfBlockDelayInfo*/,
    SPM_GLOBAL_BLOCK_NAME_CPF};
// Counter block GDS
static const GpuBlockInfo GdsCounterBlockInfo = {
    "GDS",
    GdsCounterBlockId,
    1,
    GdsCounterBlockMaxEvent,
    GdsCounterBlockNumCounters,
    GdsCounterRegAddr,
    gfx11_cntx_prim::select_value_GDS_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockSpmGlobalAttr,
    NULL /*GdsBlockDelayInfo*/,
    SPM_GLOBAL_BLOCK_NAME_GDS};
// Counter block GRBM
static const GpuBlockInfo GrbmCounterBlockInfo = {
    "GRBM",
    GrbmCounterBlockId,
    1,
    GrbmCounterBlockMaxEvent,
    GrbmCounterBlockNumCounters,
    GrbmCounterRegAddr,
    gfx11_cntx_prim::select_value_GRBM_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockGRBMAttr};
// Counter block GRBMSE
static const GpuBlockInfo GrbmSeCounterBlockInfo = {
    "GRBM_SE",
    GrbmSeCounterBlockId,
    1,
    GrbmSeCounterBlockMaxEvent,
    GrbmSeCounterBlockNumCounters,
    GrbmSeCounterRegAddr,
    gfx11_cntx_prim::select_value_GRBM_SE0_PERFCOUNTER_SELECT,
    CounterBlockDfltAttr};
// Counter block SPI
static const GpuBlockInfo SpiCounterBlockInfo = {
    "SPI",
    SpiCounterBlockId,
    1,
    SpiCounterBlockMaxEvent,
    SpiCounterBlockNumCounters,
    SpiCounterRegAddr,
    gfx11_cntx_prim::select_value_SPI_PERFCOUNTER0_SELECT,
    CounterBlockSeAttr | CounterBlockSPIAttr,
    NULL /*SpiBlockDelayInfo*/,
    SPM_SE_BLOCK_NAME_SPI};
// Counter block SQ
static const GpuBlockInfo SqCounterBlockInfo = {
    "SQ",
    SqCounterBlockId,
    1,
    SqCounterBlockMaxEvent,
    SqCounterBlockNumCounters,
    SqCounterRegAddr,
    gfx11_cntx_prim::sq_select_value,
    CounterBlockSeAttr | CounterBlockSqAttr | CounterBlockSaAttr,
    NULL,
    SPM_SE_BLOCK_NAME_SQG};
// Counter block SX
static const GpuBlockInfo SxCounterBlockInfo = {
    "SX",
    SxCounterBlockId,
    1,
    SxCounterBlockMaxEvent,
    SxCounterBlockNumCounters,
    SxCounterRegAddr,
    gfx11_cntx_prim::select_value_SX_PERFCOUNTER0_SELECT,
    CounterBlockSeAttr | CounterBlockCleanAttr,
    NULL /*SxBlockDelayInfo*/,
    SPM_SE_BLOCK_NAME_SX};
// Counter block GCEA
static const GpuBlockInfo GceaCounterBlockInfo = {
    "GCEA",
    GceaCounterBlockId,
    GceaCounterBlockNumInstances,
    GceaCounterBlockMaxEvent,
    GceaCounterBlockNumCounters,
    GceaCounterRegAddr,
    gfx11_cntx_prim::mc_select_value_GCEA_PERFCOUNTER0_CFG,
    CounterBlockMcAttr};
// Counter block TCP
static const GpuBlockInfo TcpCounterBlockInfo = {
    "TCP",
    TcpCounterBlockId,
    16,
    TcpCounterBlockMaxEvent,
    TcpCounterBlockNumCounters,
    TcpCounterRegAddr,
    gfx11_cntx_prim::select_value_TCP_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockSeAttr | CounterBlockSaAttr};
// Counter block GL1A
static const GpuBlockInfo Gl1aCounterBlockInfo = {
    "GL1A",
    Gl1aCounterBlockId,
    4,
    Gl1aCounterBlockMaxEvent,
    Gl1aCounterBlockNumCounters,
    Gl1aCounterRegAddr,
    gfx11_cntx_prim::select_value_TCP_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockSeAttr | CounterBlockSaAttr | CounterBlockTcAttr};
// Counter block GL1C
static const GpuBlockInfo Gl1cCounterBlockInfo = {
    "GL1C",
    Gl1cCounterBlockId,
    4,
    Gl1cCounterBlockMaxEvent,
    Gl1cCounterBlockNumCounters,
    Gl1cCounterRegAddr,
    gfx11_cntx_prim::select_value_TCP_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockSeAttr | CounterBlockSaAttr | CounterBlockTcAttr};
// Counter block GL2A
static const GpuBlockInfo Gl2aCounterBlockInfo = {
    "GL2A",
    Gl2aCounterBlockId,
    32,
    Gl2aCounterBlockMaxEvent,
    Gl2aCounterBlockNumCounters,
    Gl2aCounterRegAddr,
    gfx11_cntx_prim::select_value_TCP_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockTcAttr};
// Counter block GL2C
static const GpuBlockInfo Gl2cCounterBlockInfo = {
    "GL2C",
    Gl2cCounterBlockId,
    32,
    Gl2cCounterBlockMaxEvent,
    Gl2cCounterBlockNumCounters,
    Gl2cCounterRegAddr,
    gfx11_cntx_prim::select_value_TCP_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockTcAttr};
// Counter block GCR
static const GpuBlockInfo GcrCounterBlockInfo = {
    "GCR",
    GcrCounterBlockId,
    1,
    GcrCounterBlockMaxEvent,
    GcrCounterBlockNumCounters,
    GcrCounterRegAddr,
    gfx11_cntx_prim::select_value_TCP_PERFCOUNTER0_SELECT,
    CounterBlockTcAttr};
// Counter block GUS
static const GpuBlockInfo GusCounterBlockInfo = {
    "GUS",
    GusCounterBlockId,
    1,
    GusCounterBlockMaxEvent,
    GusCounterBlockNumCounters,
    GusCounterRegAddr,
    gfx11_cntx_prim::mc_select_value_RPB_PERFCOUNTER0_CFG,
    CounterBlockGusAttr};
// Counter block TA
static const GpuBlockInfo TaCounterBlockInfo = {
    "TA",
    TaCounterBlockId,
    TaCounterBlockNumInstances,
    235 /*TaCounterBlockMaxEvent*/,
    TaCounterBlockNumCounters,
    TaCounterRegAddr,
    gfx11_cntx_prim::select_value_TA_PERFCOUNTER0_SELECT,
    CounterBlockSeAttr | CounterBlockTcAttr,
    NULL /*TaBlockDelayInfo*/,
    SPM_SE_BLOCK_NAME_TA};

}  // namespace gfx11
}  // namespace gfxip

#endif  //  _GFX11_BLOCKTABLE_H_
