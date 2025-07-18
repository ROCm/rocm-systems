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

#ifndef _GFX9_BLOCKTABLE_H_
#define _GFX9_BLOCKTABLE_H_

namespace gfxip {
namespace gfx9 {
/*
 * The following tables contain register addresses of the SQ counter registers
 */

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
 * GRBM
 */
static const CounterRegInfo GrbmCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmGRBM_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGRBM_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmGRBM_PERFCOUNTER0_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGRBM_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGRBM_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmGRBM_PERFCOUNTER1_HI), REG_32B_NULL}};

/*
 * GRBM_SE
 */
static const CounterRegInfo GrbmSeCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmGRBM_SE0_PERFCOUNTER_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGRBM_SE0_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmGRBM_SE0_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGRBM_SE1_PERFCOUNTER_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGRBM_SE1_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmGRBM_SE1_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGRBM_SE2_PERFCOUNTER_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGRBM_SE2_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmGRBM_SE2_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGRBM_SE3_PERFCOUNTER_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGRBM_SE3_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmGRBM_SE3_PERFCOUNTER_HI), REG_32B_NULL}};

/*
 * PA_SU
 */
static const CounterRegInfo PaSuCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmPA_SU_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmPA_SU_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmPA_SU_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmPA_SU_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmPA_SU_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmPA_SU_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmPA_SU_PERFCOUNTER1_HI), REG_32B_ADDR(GC, 0, mmPA_SU_PERFCOUNTER1_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmPA_SU_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmPA_SU_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmPA_SU_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmPA_SU_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmPA_SU_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmPA_SU_PERFCOUNTER3_HI), REG_32B_NULL}};

/*
 * PA_SC
 */
static const CounterRegInfo PaScCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER1_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER3_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER4_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER4_LO), REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER4_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER5_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER5_LO), REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER5_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER6_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER6_LO), REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER6_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER7_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER7_LO), REG_32B_ADDR(GC, 0, mmPA_SC_PERFCOUNTER7_HI), REG_32B_NULL}};

/*
 * SPI
 */
static const CounterRegInfo SpiCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER1_HI), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER1_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER2_HI), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER2_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER3_HI), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER3_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER4_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER4_LO), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER4_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER5_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER5_LO), REG_32B_ADDR(GC, 0, mmSPI_PERFCOUNTER5_HI), REG_32B_NULL}};

/*
 * TCA
 */
static const CounterRegInfo TcaCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmTCA_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTCA_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmTCA_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmTCA_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmTCA_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTCA_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmTCA_PERFCOUNTER1_HI), REG_32B_ADDR(GC, 0, mmTCA_PERFCOUNTER1_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmTCA_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTCA_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmTCA_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmTCA_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTCA_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmTCA_PERFCOUNTER3_HI), REG_32B_NULL}};

/*
 * TCC
 */
static const CounterRegInfo TccCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmTCC_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTCC_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmTCC_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmTCC_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmTCC_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTCC_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmTCC_PERFCOUNTER1_HI), REG_32B_ADDR(GC, 0, mmTCC_PERFCOUNTER1_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmTCC_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTCC_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmTCC_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmTCC_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTCC_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmTCC_PERFCOUNTER3_HI), REG_32B_NULL}};

/*
 * TCP
 */
static const CounterRegInfo TcpCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmTCP_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTCP_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmTCP_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmTCP_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmTCP_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTCP_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmTCP_PERFCOUNTER1_HI), REG_32B_ADDR(GC, 0, mmTCP_PERFCOUNTER1_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmTCP_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTCP_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmTCP_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmTCP_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTCP_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmTCP_PERFCOUNTER3_HI), REG_32B_NULL}};

/*
 * CB
 */
static const CounterRegInfo CbCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmCB_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmCB_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmCB_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmCB_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmCB_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmCB_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmCB_PERFCOUNTER1_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmCB_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmCB_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmCB_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmCB_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmCB_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmCB_PERFCOUNTER3_HI), REG_32B_NULL}};

/*
 * DB
 */
static const CounterRegInfo DbCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmDB_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmDB_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmDB_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmDB_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmDB_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmDB_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmDB_PERFCOUNTER1_HI), REG_32B_ADDR(GC, 0, mmDB_PERFCOUNTER1_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmDB_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmDB_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmDB_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmDB_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmDB_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmDB_PERFCOUNTER3_HI), REG_32B_NULL}};

/*
 * RLC
 */
static const CounterRegInfo RlcCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmRLC_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmRLC_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmRLC_PERFCOUNTER0_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmRLC_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmRLC_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmRLC_PERFCOUNTER1_HI), REG_32B_NULL}};

/*
 * SX
 */
static const CounterRegInfo SxCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER1_HI), REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER1_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmSX_PERFCOUNTER3_HI), REG_32B_NULL}};

/*
 * TA
 */
static const CounterRegInfo TaCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmTA_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTA_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmTA_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmTA_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmTA_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTA_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmTA_PERFCOUNTER1_HI), REG_32B_NULL}};

/*
 * TD
 */
static const CounterRegInfo TdCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmTD_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTD_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmTD_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmTD_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmTD_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmTD_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmTD_PERFCOUNTER1_HI), REG_32B_NULL}};

/*
 * GDS
 */
static const CounterRegInfo GdsCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER1_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmGDS_PERFCOUNTER3_HI), REG_32B_NULL}};

/*
 * VGT
 */
static const CounterRegInfo VgtCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmVGT_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmVGT_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmVGT_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmVGT_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmVGT_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmVGT_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmVGT_PERFCOUNTER1_HI), REG_32B_ADDR(GC, 0, mmVGT_PERFCOUNTER1_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmVGT_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmVGT_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmVGT_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmVGT_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmVGT_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmVGT_PERFCOUNTER3_HI), REG_32B_NULL}};

/*
 * IA
 */
static const CounterRegInfo IaCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmIA_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmIA_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmIA_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmIA_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmIA_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmIA_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmIA_PERFCOUNTER1_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmIA_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmIA_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmIA_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmIA_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmIA_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmIA_PERFCOUNTER3_HI), REG_32B_NULL}};

/*
 * WD
 */
static const CounterRegInfo WdCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmWD_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmWD_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmWD_PERFCOUNTER0_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmWD_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmWD_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmWD_PERFCOUNTER1_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmWD_PERFCOUNTER2_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmWD_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmWD_PERFCOUNTER2_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmWD_PERFCOUNTER3_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmWD_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmWD_PERFCOUNTER3_HI), REG_32B_NULL}};

/*
 * CPC
 */
static const CounterRegInfo CpcCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmCPC_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmCPC_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmCPC_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmCPC_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmCPC_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmCPC_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmCPC_PERFCOUNTER1_HI), REG_32B_NULL}};

/*
 * CPF
 */
static const CounterRegInfo CpfCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmCPF_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmCPF_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmCPF_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmCPF_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmCPF_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmCPF_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmCPF_PERFCOUNTER1_HI), REG_32B_NULL}};

/*
 * CPG
 */
static const CounterRegInfo CpgCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmCPG_PERFCOUNTER0_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmCPG_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmCPG_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmCPG_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmCPG_PERFCOUNTER1_SELECT), REG_32B_NULL,
     REG_32B_ADDR(GC, 0, mmCPG_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmCPG_PERFCOUNTER1_HI), REG_32B_NULL}};

// RMI
static const CounterRegInfo RmiCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmRMI_PERFCOUNTER0_SELECT), REG_32B_ADDR(GC, 0, mmRMI_PERF_COUNTER_CNTL),
     REG_32B_ADDR(GC, 0, mmRMI_PERFCOUNTER0_LO), REG_32B_ADDR(GC, 0, mmRMI_PERFCOUNTER0_HI), REG_32B_ADDR(GC, 0, mmRMI_PERFCOUNTER0_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmRMI_PERFCOUNTER1_SELECT), REG_32B_ADDR(GC, 0, mmRMI_PERF_COUNTER_CNTL),
     REG_32B_ADDR(GC, 0, mmRMI_PERFCOUNTER1_LO), REG_32B_ADDR(GC, 0, mmRMI_PERFCOUNTER1_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmRMI_PERFCOUNTER2_SELECT), REG_32B_ADDR(GC, 0, mmRMI_PERF_COUNTER_CNTL),
     REG_32B_ADDR(GC, 0, mmRMI_PERFCOUNTER2_LO), REG_32B_ADDR(GC, 0, mmRMI_PERFCOUNTER2_HI), REG_32B_ADDR(GC, 0, mmRMI_PERFCOUNTER2_SELECT1)},
    {REG_32B_ADDR(GC, 0, mmRMI_PERFCOUNTER3_SELECT), REG_32B_ADDR(GC, 0, mmRMI_PERF_COUNTER_CNTL),
     REG_32B_ADDR(GC, 0, mmRMI_PERFCOUNTER3_LO), REG_32B_ADDR(GC, 0, mmRMI_PERFCOUNTER3_HI), REG_32B_NULL}};

// GCEA
static const CounterRegInfo GceaCounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER0_CFG),
     REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER_RSLT_CNTL), REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER1_CFG),
     REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER_RSLT_CNTL), REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmGCEA_PERFCOUNTER_HI), REG_32B_NULL}};

// ATC
static const CounterRegInfo AtcCounterRegAddr[] = {
    {REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER0_CFG),
     REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER_LO), REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER1_CFG),
     REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER_LO), REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER2_CFG),
     REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER_LO), REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER3_CFG),
     REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER_LO), REG_32B_ADDR(ATHUB, 0, mmATC_PERFCOUNTER_HI), REG_32B_NULL}};

// ATC L2
static const CounterRegInfo AtcL2CounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmATC_L2_PERFCOUNTER0_CFG),
     REG_32B_ADDR(GC, 0, mmATC_L2_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(GC, 0, mmATC_L2_PERFCOUNTER_LO), REG_32B_ADDR(GC, 0, mmATC_L2_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmATC_L2_PERFCOUNTER1_CFG),
     REG_32B_ADDR(GC, 0, mmATC_L2_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(GC, 0, mmATC_L2_PERFCOUNTER_LO), REG_32B_ADDR(GC, 0, mmATC_L2_PERFCOUNTER_HI), REG_32B_NULL}};

// RPB
static const CounterRegInfo RpbCounterRegAddr[] = {
    {REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER0_CFG),
     REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER_LO), REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER1_CFG),
     REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER_LO), REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER2_CFG),
     REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER_LO), REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER3_CFG),
     REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER_LO), REG_32B_ADDR(ATHUB, 0, mmRPB_PERFCOUNTER_HI), REG_32B_NULL}};

// MC VM L2
static const CounterRegInfo McVmL2CounterRegAddr[] = {
    {REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER0_CFG),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER1_CFG),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER2_CFG),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER3_CFG),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER4_CFG),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER5_CFG),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER6_CFG),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_HI), REG_32B_NULL},
    {REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER7_CFG),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_RSLT_CNTL),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_LO),
     REG_32B_ADDR(GC, 0, mmMC_VM_L2_PERFCOUNTER_HI), REG_32B_NULL}};

// BlockDelayInfo for SPM
static const uint32_t SqBlockDelayValue[] = {0x3b, 0x38, 0x39, 0x36}; // Verified
static const uint32_t PaSuBlockDelayValue[] = {0x22, 0x22, 0x20, 0x20};
static const uint32_t PaScBlockDelayValue[] = {0x26, 0x26, 0x22, 0x22, 0x23, 0x23, 0x21, 0x21};
static const uint32_t SpiBlockDelayValue[] = {0x39, 0x39, 0x36, 0x36}; // Verified
static const uint32_t TcaBlockDelayValue[] = {0x18, 0x1c};
static const uint32_t TccBlockDelayValue[] = {
    0x0f, 0x15, 0x17, 0x1d, 0x1b, 0x17, 0x13, 0x0f, 0x14, 0x18, 0x1c, 0x20, 0x1d, 0x19, 0x15, 0x11};
static const uint32_t TcpBlockDelayValue[] = {
    0x30, 0x2e, 0x2c, 0x2a, 0x28, 0x26, 0x24, 0x22, 0x20, 0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12,  // se0
    0x2c, 0x2a, 0x28, 0x26, 0x24, 0x22, 0x20, 0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e,  // se1
    0x2c, 0x2a, 0x28, 0x26, 0x24, 0x22, 0x20, 0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e,  // se2
    0x2c, 0x28, 0x26, 0x24, 0x22, 0x20, 0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c}; // se3
static const uint32_t CbBlockDelayValue[] = {
    0x1d, 0x08, 0x14, 0x11, 0x19, 0x04, 0x10, 0x0d, 0x19, 0x05, 0x11, 0x0e, 0x15, 0x00, 0x0c, 0x09};
static const uint32_t DbBlockDelayValue[] = {
    0x1b, 0x0c, 0x17, 0x10, 0x17, 0x08, 0x13, 0x0c, 0x17, 0x09, 0x14, 0x0d, 0x13, 0x08, 0x0f, 0x08};
static const uint32_t SxBlockDelayValue[] = {0x06, 0x06, 0x08, 0x04};
static const uint32_t TaBlockDelayValue[] = {
    0x2c, 0x2a, 0x28, 0x26, 0x24, 0x22, 0x20, 0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e,
    0x28, 0x26, 0x25, 0x23, 0x21, 0x1f, 0x1d, 0x1b, 0x19, 0x17, 0x15, 0x13, 0x11, 0x0f, 0x0d, 0x0b,
    0x28, 0x26, 0x24, 0x22, 0x20, 0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a,
    0x27, 0x25, 0x23, 0x21, 0x1f, 0x1d, 0x1b, 0x19, 0x17, 0x15, 0x13, 0x11, 0x0f, 0x0d, 0x0b, 0x09};
static const uint32_t TdBlockDelayValue[] = {
    0x2c, 0x2a, 0x28, 0x26, 0x24, 0x22, 0x20, 0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e,
    0x28, 0x26, 0x25, 0x23, 0x21, 0x1f, 0x1d, 0x1b, 0x19, 0x17, 0x15, 0x13, 0x11, 0x0f, 0x0d, 0x0b,
    0x28, 0x27, 0x25, 0x23, 0x21, 0x1f, 0x1d, 0x1b, 0x19, 0x17, 0x15, 0x13, 0x11, 0x0f, 0x0d, 0x0b,
    0x27, 0x25, 0x23, 0x21, 0x1f, 0x1d, 0x1b, 0x19, 0x17, 0x15, 0x13, 0x11, 0x0f, 0x0d, 0x0b, 0x09};
static const uint32_t GdsBlockDelayValue[] = {0x2d};
static const uint32_t VgtBlockDelayValue[] = {0x27, 0x27, 0x23, 0x24};
static const uint32_t IaBlockDelayValue[] = {0x32};
static const uint32_t CpcBlockDelayValue[] = {0x3c};
static const uint32_t CpfBlockDelayValue[] = {0x32};
static const uint32_t CpgBlockDelayValue[] = {0x30}; // Verified

static const BlockDelayInfo SqBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_SQG_PERFMON_SAMPLE_DELAY), SqBlockDelayValue};
static const BlockDelayInfo PaSuBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_PA_PERFMON_SAMPLE_DELAY), PaSuBlockDelayValue};
static const BlockDelayInfo PaScBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_SC_PERFMON_SAMPLE_DELAY), PaScBlockDelayValue};
static const BlockDelayInfo SpiBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_SPI_PERFMON_SAMPLE_DELAY), SpiBlockDelayValue};
static const BlockDelayInfo TcaBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_TCA_PERFMON_SAMPLE_DELAY), TcaBlockDelayValue};
static const BlockDelayInfo TccBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_TCC_PERFMON_SAMPLE_DELAY), TccBlockDelayValue};
static const BlockDelayInfo TcpBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_TCP_PERFMON_SAMPLE_DELAY), TcpBlockDelayValue};
static const BlockDelayInfo CbBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_CB_PERFMON_SAMPLE_DELAY), CbBlockDelayValue};
static const BlockDelayInfo DbBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_DB_PERFMON_SAMPLE_DELAY), DbBlockDelayValue};
static const BlockDelayInfo SxBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_SX_PERFMON_SAMPLE_DELAY), SxBlockDelayValue};
static const BlockDelayInfo TaBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_TA_PERFMON_SAMPLE_DELAY), TaBlockDelayValue};
static const BlockDelayInfo TdBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_TD_PERFMON_SAMPLE_DELAY), TdBlockDelayValue};
static const BlockDelayInfo GdsBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_GDS_PERFMON_SAMPLE_DELAY), GdsBlockDelayValue};
static const BlockDelayInfo VgtBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_VGT_PERFMON_SAMPLE_DELAY), VgtBlockDelayValue};
static const BlockDelayInfo IaBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_IA_PERFMON_SAMPLE_DELAY), IaBlockDelayValue};
static const BlockDelayInfo CpcBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_CPC_PERFMON_SAMPLE_DELAY), CpcBlockDelayValue};
static const BlockDelayInfo CpfBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_CPF_PERFMON_SAMPLE_DELAY), CpfBlockDelayValue};
static const BlockDelayInfo CpgBlockDelayInfo = {REG_32B_ADDR(GC, 0, mmRLC_SPM_CPG_PERFMON_SAMPLE_DELAY), CpgBlockDelayValue};

// Counter block info table
// SPM global blocks: CPG, CPC, CPF, GDS, TCC, TCA, IA, TCS
// SPM shader engine blocks: CB, DB, SC, SX, TA, TD, TCP, VGT, SQG, SPI, PA
// Counter block CB
static const GpuBlockInfo CbCounterBlockInfo = {"CB",
                                                CbCounterBlockId,
                                                CbCounterBlockNumInstances,
                                                CbCounterBlockMaxEvent,
                                                CbCounterBlockNumCounters,
                                                CbCounterRegAddr,
                                                gfx9_cntx_prim::select_value_CB_PERFCOUNTER0_SELECT,
                                                CounterBlockSeAttr | CounterBlockCleanAttr,
                                                CbBlockDelayInfo,
                                                SPM_SE_BLOCK_NAME_CB};
// Counter block DB
static const GpuBlockInfo DbCounterBlockInfo = {"DB",
                                                DbCounterBlockId,
                                                DbCounterBlockNumInstances,
                                                DbCounterBlockMaxEvent,
                                                DbCounterBlockNumCounters,
                                                DbCounterRegAddr,
                                                gfx9_cntx_prim::select_value_DB_PERFCOUNTER0_SELECT,
                                                CounterBlockSeAttr | CounterBlockCleanAttr,
                                                DbBlockDelayInfo,
                                                SPM_SE_BLOCK_NAME_DB};
// Counter block GRBM
static const GpuBlockInfo GrbmCounterBlockInfo = {
    "GRBM",
    GrbmCounterBlockId,
    1,
    GrbmCounterBlockMaxEvent,
    GrbmCounterBlockNumCounters,
    GrbmCounterRegAddr,
    gfx9_cntx_prim::select_value_GRBM_PERFCOUNTER0_SELECT,
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
    gfx9_cntx_prim::select_value_GRBM_SE0_PERFCOUNTER_SELECT,
    CounterBlockDfltAttr,
    BLOCK_DELAY_NONE};
// Counter block PA_SU
static const GpuBlockInfo PaSuCounterBlockInfo = {
    "PA_SU",
    PaSuCounterBlockId,
    1,
    PaSuCounterBlockMaxEvent,
    PaSuCounterBlockNumCounters,
    PaSuCounterRegAddr,
    gfx9_cntx_prim::select_value_PA_SU_PERFCOUNTER0_SELECT,
    CounterBlockSeAttr,
    PaSuBlockDelayInfo,
    SPM_SE_BLOCK_NAME_PA};
// Counter block PA_SC
static const GpuBlockInfo PaScCounterBlockInfo = {
    "PA_SC",
    PaScCounterBlockId,
    1,
    PaScCounterBlockMaxEvent,
    PaScCounterBlockNumCounters,
    PaScCounterRegAddr,
    gfx9_cntx_prim::select_value_PA_SC_PERFCOUNTER0_SELECT,
    CounterBlockSeAttr,
    PaScBlockDelayInfo,
    SPM_SE_BLOCK_NAME_SC};
// Counter block SPI
static const GpuBlockInfo SpiCounterBlockInfo = {
    "SPI",
    SpiCounterBlockId,
    1,
    SpiCounterBlockMaxEvent,
    SpiCounterBlockNumCounters,
    SpiCounterRegAddr,
    gfx9_cntx_prim::select_value_SPI_PERFCOUNTER0_SELECT,
    CounterBlockSeAttr | CounterBlockSPIAttr,
    SpiBlockDelayInfo,
    SPM_SE_BLOCK_NAME_SPI};
// Counter block SQ
static const GpuBlockInfo SqCounterBlockInfo = {"SQ",
                                                SqCounterBlockId,
                                                1,
                                                SqCounterBlockMaxEvent,
                                                SqCounterBlockNumCounters,
                                                SqCounterRegAddr,
                                                gfx9_cntx_prim::sq_select_value,
                                                CounterBlockSeAttr | CounterBlockSqAttr,
                                                SqBlockDelayInfo,
                                                SPM_SE_BLOCK_NAME_SQG};
static const GpuBlockInfo SqGsCounterBlockInfo = {"SQ_GS",
                                                  SqCounterBlockId,
                                                  1,
                                                  SqCounterBlockMaxEvent,
                                                  SqCounterBlockNumCounters,
                                                  SqCounterRegAddr,
                                                  gfx9_cntx_prim::sq_select_value,
                                                  CounterBlockSeAttr | CounterBlockSqAttr,
                                                  BLOCK_DELAY_NONE};
static const GpuBlockInfo SqVsCounterBlockInfo = {"SQ_VS",
                                                  SqCounterBlockId,
                                                  1,
                                                  SqCounterBlockMaxEvent,
                                                  SqCounterBlockNumCounters,
                                                  SqCounterRegAddr,
                                                  gfx9_cntx_prim::sq_select_value,
                                                  CounterBlockSeAttr | CounterBlockSqAttr,
                                                  BLOCK_DELAY_NONE};
static const GpuBlockInfo SqPsCounterBlockInfo = {"SQ_PS",
                                                  SqCounterBlockId,
                                                  1,
                                                  SqCounterBlockMaxEvent,
                                                  SqCounterBlockNumCounters,
                                                  SqCounterRegAddr,
                                                  gfx9_cntx_prim::sq_select_value,
                                                  CounterBlockSeAttr | CounterBlockSqAttr,
                                                  BLOCK_DELAY_NONE};
static const GpuBlockInfo SqHsCounterBlockInfo = {"SQ_HS",
                                                  SqCounterBlockId,
                                                  1,
                                                  SqCounterBlockMaxEvent,
                                                  SqCounterBlockNumCounters,
                                                  SqCounterRegAddr,
                                                  gfx9_cntx_prim::sq_select_value,
                                                  CounterBlockSeAttr | CounterBlockSqAttr,
                                                  BLOCK_DELAY_NONE};
static const GpuBlockInfo SqCsCounterBlockInfo = {"SQ_CS",
                                                  SqCounterBlockId,
                                                  1,
                                                  SqCounterBlockMaxEvent,
                                                  SqCounterBlockNumCounters,
                                                  SqCounterRegAddr,
                                                  gfx9_cntx_prim::sq_select_value,
                                                  CounterBlockSeAttr | CounterBlockSqAttr,
                                                  BLOCK_DELAY_NONE};
// Counter block SX
static const GpuBlockInfo SxCounterBlockInfo = {"SX",
                                                SxCounterBlockId,
                                                1,
                                                SxCounterBlockMaxEvent,
                                                SxCounterBlockNumCounters,
                                                SxCounterRegAddr,
                                                gfx9_cntx_prim::select_value_SX_PERFCOUNTER0_SELECT,
                                                CounterBlockSeAttr | CounterBlockCleanAttr,
                                                SxBlockDelayInfo,
                                                SPM_SE_BLOCK_NAME_SX};
// Counter block TA
static const GpuBlockInfo TaCounterBlockInfo = {"TA",
                                                TaCounterBlockId,
                                                TaCounterBlockNumInstances,
                                                TaCounterBlockMaxEvent,
                                                TaCounterBlockNumCounters,
                                                TaCounterRegAddr,
                                                gfx9_cntx_prim::select_value_TA_PERFCOUNTER0_SELECT,
                                                CounterBlockSeAttr | CounterBlockTcAttr,
                                                TaBlockDelayInfo,
                                                SPM_SE_BLOCK_NAME_TA};
// Counter block TCA
static const GpuBlockInfo TcaCounterBlockInfo = {
    "TCA",
    TcaCounterBlockId,
    TcaCounterBlockNumInstances,
    TcaCounterBlockMaxEvent,
    TcaCounterBlockNumCounters,
    TcaCounterRegAddr,
    gfx9_cntx_prim::select_value_TCA_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockTcAttr | CounterBlockSpmGlobalAttr,
    TcaBlockDelayInfo,
    SPM_GLOBAL_BLOCK_NAME_TCA};
// Counter block TCC
static const GpuBlockInfo TccCounterBlockInfo = {
    "TCC",
    TccCounterBlockId,
    TccCounterBlockNumInstances,
    TccCounterBlockMaxEvent,
    TccCounterBlockNumCounters,
    TccCounterRegAddr,
    gfx9_cntx_prim::select_value_TCC_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockTcAttr | CounterBlockSpmGlobalAttr,
    TccBlockDelayInfo,
    SPM_GLOBAL_BLOCK_NAME_TCC};
// Counter block TD
static const GpuBlockInfo TdCounterBlockInfo = {"TD",
                                                TdCounterBlockId,
                                                TdCounterBlockNumInstances,
                                                TdCounterBlockMaxEvent,
                                                TdCounterBlockNumCounters,
                                                TdCounterRegAddr,
                                                gfx9_cntx_prim::select_value_TD_PERFCOUNTER0_SELECT,
                                                CounterBlockSeAttr | CounterBlockTcAttr,
                                                TdBlockDelayInfo,
                                                SPM_SE_BLOCK_NAME_TD};
// Counter block TCP
static const GpuBlockInfo TcpCounterBlockInfo = {
    "TCP",
    TcpCounterBlockId,
    TcpCounterBlockNumInstances,
    TcpCounterBlockMaxEvent,
    TcpCounterBlockNumCounters,
    TcpCounterRegAddr,
    gfx9_cntx_prim::select_value_TCP_PERFCOUNTER0_SELECT,
    CounterBlockSeAttr | CounterBlockTcAttr,
    TcpBlockDelayInfo,
    SPM_SE_BLOCK_NAME_TCP};
// Counter block GDS
static const GpuBlockInfo GdsCounterBlockInfo = {
    "GDS",
    GdsCounterBlockId,
    1,
    GdsCounterBlockMaxEvent,
    GdsCounterBlockNumCounters,
    GdsCounterRegAddr,
    gfx9_cntx_prim::select_value_GDS_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockSpmGlobalAttr,
    GdsBlockDelayInfo,
    SPM_GLOBAL_BLOCK_NAME_GDS};
// Counter block VGT
static const GpuBlockInfo VgtCounterBlockInfo = {
    "VGT",
    VgtCounterBlockId,
    1,
    VgtCounterBlockMaxEvent,
    VgtCounterBlockNumCounters,
    VgtCounterRegAddr,
    gfx9_cntx_prim::select_value_VGT_PERFCOUNTER0_SELECT,
    CounterBlockSeAttr,
    VgtBlockDelayInfo,
    SPM_SE_BLOCK_NAME_VGT};
// Counter block IA
static const GpuBlockInfo IaCounterBlockInfo = {"IA",
                                                IaCounterBlockId,
                                                1,
                                                IaCounterBlockMaxEvent,
                                                IaCounterBlockNumCounters,
                                                IaCounterRegAddr,
                                                gfx9_cntx_prim::select_value_IA_PERFCOUNTER0_SELECT,
                                                CounterBlockSeAttr | CounterBlockSpmGlobalAttr,
                                                IaBlockDelayInfo,
                                                SPM_GLOBAL_BLOCK_NAME_IA};
// Counter block WD
static const GpuBlockInfo WdCounterBlockInfo = {"WD",
                                                WdCounterBlockId,
                                                1,
                                                WdCounterBlockMaxEvent,
                                                WdCounterBlockNumCounters,
                                                WdCounterRegAddr,
                                                gfx9_cntx_prim::select_value_WD_PERFCOUNTER0_SELECT,
                                                CounterBlockDfltAttr,
                                                BLOCK_DELAY_NONE};
// Counter block CPC
static const GpuBlockInfo CpcCounterBlockInfo = {
    "CPC",
    CpcCounterBlockId,
    1,
    CpcCounterBlockMaxEvent,
    CpcCounterBlockNumCounters,
    CpcCounterRegAddr,
    gfx9_cntx_prim::select_value_CPC_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockSpmGlobalAttr,
    CpcBlockDelayInfo,
    SPM_GLOBAL_BLOCK_NAME_CPC};
// Counter block CPF
static const GpuBlockInfo CpfCounterBlockInfo = {
    "CPF",
    CpfCounterBlockId,
    1,
    CpfCounterBlockMaxEvent,
    CpfCounterBlockNumCounters,
    CpfCounterRegAddr,
    gfx9_cntx_prim::select_value_CPF_PERFCOUNTER0_SELECT,
    CounterBlockDfltAttr | CounterBlockSpmGlobalAttr,
    CpfBlockDelayInfo,
    SPM_GLOBAL_BLOCK_NAME_CPF};
// Counter block MCVM L2
static const GpuBlockInfo McVmL2CounterBlockInfo = {
    "MCVML2",
    McVmL2CounterBlockId,
    1,
    McVmL2CounterBlockMaxEvent,
    McVmL2CounterBlockNumCounters,
    McVmL2CounterRegAddr,
    gfx9_cntx_prim::mc_select_value_MC_VM_L2_PERFCOUNTER0_CFG,
    CounterBlockMcAttr,
    BLOCK_DELAY_NONE};
// Counter block ATC L2
static const GpuBlockInfo AtcL2CounterBlockInfo = {
    "ATCL2",
    AtcL2CounterBlockId,
    1,
    AtcL2CounterBlockMaxEvent,
    AtcL2CounterBlockNumCounters,
    AtcL2CounterRegAddr,
    gfx9_cntx_prim::mc_select_value_ATC_L2_PERFCOUNTER0_CFG,
    CounterBlockMcAttr,
    BLOCK_DELAY_NONE};
// Counter block ATC
static const GpuBlockInfo AtcCounterBlockInfo = {
    "ATC",
    AtcCounterBlockId,
    1,
    AtcCounterBlockMaxEvent,
    AtcCounterBlockNumCounters,
    AtcCounterRegAddr,
    gfx9_cntx_prim::mc_select_value_ATC_PERFCOUNTER0_CFG,
    CounterBlockAtcAttr | CounterBlockAidAttr,
    BLOCK_DELAY_NONE};
// Counter block GCEA
static const GpuBlockInfo GceaCounterBlockInfo = {
    "GCEA",
    GceaCounterBlockId,
    GceaCounterBlockNumInstances,
    GceaCounterBlockMaxEvent,
    GceaCounterBlockNumCounters,
    GceaCounterRegAddr,
    gfx9_cntx_prim::mc_select_value_GCEA_PERFCOUNTER0_CFG,
    CounterBlockMcAttr,
    BLOCK_DELAY_NONE};
// Counter block RPB
static const GpuBlockInfo RpbCounterBlockInfo = {
    "RPB",
    RpbCounterBlockId,
    1,
    RpbCounterBlockMaxEvent,
    RpbCounterBlockNumCounters,
    RpbCounterRegAddr,
    gfx9_cntx_prim::mc_select_value_RPB_PERFCOUNTER0_CFG,
    CounterBlockRpbAttr | CounterBlockAidAttr,
    BLOCK_DELAY_NONE};

}  // namespace gfx9
}  // namespace gfxip

#endif  //  _GFX9_BLOCKTABLE_H_
