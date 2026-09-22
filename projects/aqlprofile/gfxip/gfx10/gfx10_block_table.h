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

#ifndef _GFX10_BLOCKTABLE_H_
#define _GFX10_BLOCKTABLE_H_

namespace gfxip {
namespace gfx10 {

/*
 * CPC
 */
static const CounterRegInfo CpcCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmCPC_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmCPC_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmCPC_PERFCOUNTER0_HI),
     REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmCPC_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmCPC_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmCPC_PERFCOUNTER1_HI),
     REG_32B_NULL}};

/*
 * CPF
 */
static const CounterRegInfo CpfCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmCPF_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmCPF_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmCPF_PERFCOUNTER0_HI),
     REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmCPF_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmCPF_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmCPF_PERFCOUNTER1_HI),
     REG_32B_NULL}};

/*
 * GDS
 */
static const CounterRegInfo GdsCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER0_HI),
     REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER1_HI),
     REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER2_HI),
     REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER3_HI),
     REG_32B_NULL}};

/*
 * GRBM
 */
static const CounterRegInfo GrbmCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmGRBM_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGRBM_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmGRBM_PERFCOUNTER0_HI),
     REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGRBM_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGRBM_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmGRBM_PERFCOUNTER1_HI),
     REG_32B_NULL}};

/*
 * GRBM_SE
 */
static const CounterRegInfo GrbmSeCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmGRBM_SE0_PERFCOUNTER_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGRBM_SE0_PERFCOUNTER_LO), REG_32B_ADDR(GC, 0, mmGRBM_SE0_PERFCOUNTER_HI),
     REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGRBM_SE1_PERFCOUNTER_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGRBM_SE1_PERFCOUNTER_LO), REG_32B_ADDR(GC, 0, mmGRBM_SE1_PERFCOUNTER_HI),
     REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGRBM_SE2_PERFCOUNTER_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGRBM_SE2_PERFCOUNTER_LO), REG_32B_ADDR(GC, 0, mmGRBM_SE2_PERFCOUNTER_HI),
     REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGRBM_SE3_PERFCOUNTER_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGRBM_SE3_PERFCOUNTER_LO), REG_32B_ADDR(GC, 0, mmGRBM_SE3_PERFCOUNTER_HI),
     REG_32B_NULL}};

/*
 * SPI
 */
static const CounterRegInfo SpiCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER0_HI),
     REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER1_HI),
     REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER2_HI),
     REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER3_HI),
     REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER4_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER4_LO), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER4_HI),
     REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER5_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER5_LO), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER5_HI),
     REG_32B_NULL}};

/*
 * SQ
 */
static const CounterRegInfo SqCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER0_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER0_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER1_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER1_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER2_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER3_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER3_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER4_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER4_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER4_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER5_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER5_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER5_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER6_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER6_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER6_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER7_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER7_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER7_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER8_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER8_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER8_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER9_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER9_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER9_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER10_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER10_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER10_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER11_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER11_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER11_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER12_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER12_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER12_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER13_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER13_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER13_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER14_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER14_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER14_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER15_SELECT), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER_CTRL),
     REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER15_LO), REG_32B_ADDR(GC, 0, mmSQ_PERFCOUNTER15_HI), REG_32B_NULL}};

/*
 * SX
 */
static const CounterRegInfo SxCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER0_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER1_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER3_HI), REG_32B_NULL}};

/*
 * GCEA
 */
static const CounterRegInfo GceaCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER0_CFG),
     REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER_RSLT_CNTL), REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER1_CFG),
     REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER_RSLT_CNTL), REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER_HI), REG_32B_NULL}};

// Define GFX10 specific blocks table entries like GC caches blocks
/*
 * GCR
 */
static const CounterRegInfo GcrCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmGCR_PERFCOUNTER0_SELECT), REG_32B_ADDR(GC, 0, mmGCR_GENERAL_CNTL),
     REG_32B_ADDR(GC, 0, mmGCR_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmGCR_PERFCOUNTER0_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGCR_PERFCOUNTER1_SELECT), REG_32B_ADDR(GC, 0, mmGCR_GENERAL_CNTL),
     REG_32B_ADDR(GC, 0, mmGCR_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmGCR_PERFCOUNTER1_HI), REG_32B_NULL}};

/*
 * GL1A
 */
static const CounterRegInfo Gl1aCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmGL1A_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL1A_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmGL1A_PERFCOUNTER0_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGL1A_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL1A_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmGL1A_PERFCOUNTER1_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGL1A_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL1A_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmGL1A_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGL1A_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL1A_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmGL1A_PERFCOUNTER3_HI), REG_32B_NULL}};

/*
 * GL1C
 */
static const CounterRegInfo Gl1cCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmGL1C_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL1C_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmGL1C_PERFCOUNTER0_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGL1C_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL1C_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmGL1C_PERFCOUNTER1_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGL1C_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL1C_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmGL1C_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGL1C_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL1C_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmGL1C_PERFCOUNTER3_HI), REG_32B_NULL},
};

/*
 * GL2A
 */
static const CounterRegInfo Gl2aCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmGL2A_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL2A_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmGL2A_PERFCOUNTER0_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGL2A_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL2A_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmGL2A_PERFCOUNTER1_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGL2A_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL2A_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmGL2A_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGL2A_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL2A_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmGL2A_PERFCOUNTER3_HI), REG_32B_NULL},
};

/*
 * GL2C
 */
static const CounterRegInfo Gl2cCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmGL2C_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL2C_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmGL2C_PERFCOUNTER0_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGL2C_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL2C_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmGL2C_PERFCOUNTER1_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGL2C_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL2C_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmGL2C_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGL2C_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGL2C_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmGL2C_PERFCOUNTER3_HI), REG_32B_NULL},
};

/*
 * GUS
 */
static const CounterRegInfo GusCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmGUS_PERFCOUNTER0_CFG), REG_32B_ADDR(GC, 0, mmGUS_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(GC, 0, mmGUS_PERFCOUNTER_LO), REG_32B_ADDR(GC, 0, mmGUS_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGUS_PERFCOUNTER1_CFG), REG_32B_ADDR(GC, 0, mmGUS_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(GC, 0, mmGUS_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmGUS_PERFCOUNTER2_HI), REG_32B_NULL},
};

/*
 * TA
 */
static const CounterRegInfo TaCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmTA_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTA_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmTA_PERFCOUNTER0_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmTA_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTA_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmTA_PERFCOUNTER1_HI), REG_32B_NULL}};

// Counter block CPC
static const GpuBlockInfo CpcCounterBlockInfo = {
    "CPC",
    CpcCounterBlockId,
    1,
    CpcCounterBlockMaxEvent,
    CpcCounterBlockNumCounters,
    CpcCounterRegAddr,
    gfx10_cntx_prim::select_value_CPC_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockSpmGlobalAttr,
    BLOCK_DELAY_NONE};
// Counter block CPF
static const GpuBlockInfo CpfCounterBlockInfo = {
    "CPF",
    CpfCounterBlockId,
    1,
    CpfCounterBlockMaxEvent,
    CpfCounterBlockNumCounters,
    CpfCounterRegAddr,
    gfx10_cntx_prim::select_value_CPF_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockSpmGlobalAttr,
    BLOCK_DELAY_NONE};
// Counter block GDS
static const GpuBlockInfo GdsCounterBlockInfo = {
    "GDS",
    GdsCounterBlockId,
    1,
    GdsCounterBlockMaxEvent,
    GdsCounterBlockNumCounters,
    GdsCounterRegAddr,
    gfx10_cntx_prim::select_value_GDS_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockSpmGlobalAttr,
    BLOCK_DELAY_NONE};
// Counter block GRBM
static const GpuBlockInfo GrbmCounterBlockInfo = {
    "GRBM",
    GrbmCounterBlockId,
    1,
    GrbmCounterBlockMaxEvent,
    GrbmCounterBlockNumCounters,
    GrbmCounterRegAddr,
    gfx10_cntx_prim::select_value_GRBM_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockGRBMAttr,
    BLOCK_DELAY_NONE};
// Counter block GRBMSE
static const GpuBlockInfo GrbmSeCounterBlockInfo = {
    "GRBM_SE",
    GrbmSeCounterBlockId,
    1,
    GrbmSeCounterBlockMaxEvent,
    GrbmSeCounterBlockNumCounters,
    GrbmSeCounterRegAddr,
    gfx10_cntx_prim::select_value_GRBM_SE0_PERFCOUNTER_SELECT,
    CounterBlockDfltAttr,
    BLOCK_DELAY_NONE};
// Counter block SPI
static const GpuBlockInfo SpiCounterBlockInfo = {
    "SPI",
    SpiCounterBlockId,
    1,
    SpiCounterBlockMaxEvent,
    SpiCounterBlockNumCounters,
    SpiCounterRegAddr,
    gfx10_cntx_prim::select_value_SPI_PERFCOUNTER0_SELECT,
    CounterBlockSeAttr | CounterBlockSPIAttr,
    BLOCK_DELAY_NONE};
// Counter block SQ
static const GpuBlockInfo SqCounterBlockInfo = {"SQ",
                                                SqCounterBlockId,
                                                1,
                                                SqCounterBlockMaxEvent,
                                                SqCounterBlockNumCounters,
                                                SqCounterRegAddr,
                                                gfx10_cntx_prim::sq_select_value,
                                                CounterBlockSeAttr | CounterBlockSqAttr,
                                                BLOCK_DELAY_NONE};
// Counter block SX
static const GpuBlockInfo SxCounterBlockInfo = {
    "SX",
    SxCounterBlockId,
    1,
    SxCounterBlockMaxEvent,
    SxCounterBlockNumCounters,
    SxCounterRegAddr,
    gfx10_cntx_prim::select_value_SX_PERFCOUNTER0_SELECT,
    CounterBlockSeAttr | CounterBlockCleanAttr,
    BLOCK_DELAY_NONE};
// Counter block GCEA
static const GpuBlockInfo GceaCounterBlockInfo = {
    "GCEA",
    GceaCounterBlockId,
    GceaCounterBlockNumInstances,
    GceaCounterBlockMaxEvent,
    GceaCounterBlockNumCounters,
    GceaCounterRegAddr,
    gfx10_cntx_prim::mc_select_value_GCEA_PERFCOUNTER0_CFG,
    CounterBlockMcAttr,
    BLOCK_DELAY_NONE};
// Counter block GL1A
static const GpuBlockInfo Gl1aCounterBlockInfo = {
    "GL1A",
    Gl1aCounterBlockId,
    8,
    Gl1aCounterBlockMaxEvent,
    Gl1aCounterBlockNumCounters,
    Gl1aCounterRegAddr,
    gfx10_cntx_prim::select_value_TCP_PERFCOUNTER0_SELECT,
    CounterBlockSeAttr | CounterBlockSaAttr | CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
// Counter block GL1C
static const GpuBlockInfo Gl1cCounterBlockInfo = {
    "GL1C",
    Gl1cCounterBlockId,
    8,
    Gl1cCounterBlockMaxEvent,
    Gl1cCounterBlockNumCounters,
    Gl1cCounterRegAddr,
    gfx10_cntx_prim::select_value_TCP_PERFCOUNTER0_SELECT,
    CounterBlockSeAttr | CounterBlockSaAttr | CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
// Counter block GL2A
static const GpuBlockInfo Gl2aCounterBlockInfo = {
    "GL2A",
    Gl2aCounterBlockId,
    32,
    Gl2aCounterBlockMaxEvent,
    Gl2aCounterBlockNumCounters,
    Gl2aCounterRegAddr,
    gfx10_cntx_prim::select_value_TCP_PERFCOUNTER0_SELECT,
    CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
// Counter block GL2C
static const GpuBlockInfo Gl2cCounterBlockInfo = {
    "GL2C",
    Gl2cCounterBlockId,
    32,
    Gl2cCounterBlockMaxEvent,
    Gl2cCounterBlockNumCounters,
    Gl2cCounterRegAddr,
    gfx10_cntx_prim::select_value_TCP_PERFCOUNTER0_SELECT,
    CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
// Counter block GCR
static const GpuBlockInfo GcrCounterBlockInfo = {
    "GCR",
    GcrCounterBlockId,
    1,
    GcrCounterBlockMaxEvent,
    GcrCounterBlockNumCounters,
    GcrCounterRegAddr,
    gfx10_cntx_prim::select_value_TCP_PERFCOUNTER0_SELECT,
    CounterBlockTcAttr,
    BLOCK_DELAY_NONE};
// Counter block GUS
static const GpuBlockInfo GusCounterBlockInfo = {
    "GUS",
    GusCounterBlockId,
    1,
    GusCounterBlockMaxEvent,
    GusCounterBlockNumCounters,
    GusCounterRegAddr,
    gfx10_cntx_prim::mc_select_value_RPB_PERFCOUNTER0_CFG,
    CounterBlockGusAttr,
    BLOCK_DELAY_NONE};
// Counter block TA
static const GpuBlockInfo TaCounterBlockInfo = {
    "TA",
    TaCounterBlockId,
    TaCounterBlockNumInstances,
    235 /*TaCounterBlockMaxEvent*/,
    TaCounterBlockNumCounters,
    TaCounterRegAddr,
    gfx10_cntx_prim::select_value_TA_PERFCOUNTER0_SELECT,
    CounterBlockSeAttr | CounterBlockTcAttr,
    BLOCK_DELAY_NONE};

}  // namespace gfx10
}  // namespace gfxip

#endif  //  _GFX10_BLOCKTABLE_H_
