////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_AMD_GPU_PM4_H_
#define HSA_RUNTIME_CORE_INC_AMD_GPU_PM4_H_

 // clang-format off

#define PM4_HDR_IT_OPCODE_NOP                             0x10U
#define PM4_HDR_IT_OPCODE_INDIRECT_BUFFER                 0x3FU
#define PM4_HDR_IT_OPCODE_RELEASE_MEM                     0x49U
#define PM4_HDR_IT_OPCODE_ACQUIRE_MEM                     0x58U

#define PM4_HDR_IT_OPCODE_ATOMIC_MEM                      0x1EU
#define PM4_HDR_IT_OPCODE_PRED_EXEC                       0x23U
#define PM4_HDR_IT_OPCODE_WRITE_DATA                      0x37U
#define PM4_HDR_IT_OPCODE_WAIT_REG_MEM                    0x3CU
#define PM4_HDR_IT_OPCODE_COPY_DATA                       0x40U
#define PM4_HDR_IT_OPCODE_DMA_DATA                        0x50U

#define PM4_HDR_SHADER_TYPE(x)                            (((x) & 0x1U) << 1)
#define PM4_HDR_IT_OPCODE(x)                              (((x) & 0xFFU) << 8)
#define PM4_HDR_COUNT(x)                                  (((x) & 0x3FFFU) << 16)
#define PM4_HDR_TYPE(x)                                   (((x) & 0x3U) << 30)

#define PM4_HDR(it_opcode, pkt_size_dw, gfxip_ver) (  \
  PM4_HDR_SHADER_TYPE((gfxip_ver) == 7 ? 1 : 0)    |  \
  PM4_HDR_IT_OPCODE(it_opcode)                     |  \
  PM4_HDR_COUNT(pkt_size_dw - 2)                   |  \
  PM4_HDR_TYPE(3)                                     \
)

#define PM4_INDIRECT_BUFFER_DW1_IB_BASE_LO(x)              (((x) & 0x3FFFFFFFU) << 2)
#define PM4_INDIRECT_BUFFER_DW2_IB_BASE_HI(x)              (((x) & 0xFFFFU) << 0)
#define PM4_INDIRECT_BUFFER_DW3_IB_SIZE(x)                 (((x) & 0xFFFFFU) << 0)
#define PM4_INDIRECT_BUFFER_DW3_IB_VALID(x)                (((x) & 0x1U) << 23)

#define PM4_ACQUIRE_MEM_DW1_COHER_CNTL(x)                  (((x) & 0x7FFFFFFFU) << 0)
#  define PM4_ACQUIRE_MEM_COHER_CNTL_TC_WB_ACTION_ENA      (1U << 18)
#  define PM4_ACQUIRE_MEM_COHER_CNTL_TC_ACTION_ENA         (1U << 23)
#  define PM4_ACQUIRE_MEM_COHER_CNTL_SH_KCACHE_ACTION_ENA  (1U << 27)
#  define PM4_ACQUIRE_MEM_COHER_CNTL_SH_ICACHE_ACTION_ENA  (1U << 29)
#define PM4_ACQUIRE_MEM_DW2_COHER_SIZE(x)                  (((x) & 0xFFFFFFFFU) << 0)
#define PM4_ACQUIRE_MEM_DW3_COHER_SIZE_HI(x)               (((x) & 0xFFU) << 0)
#define PM4_ACQUIRE_MEM_DW4_COHER_BASE(x)                  ((x >> 8) & 0xFFFFFFFFU)
#define PM4_ACQUIRE_MEM_DW4_COHER_BASE_HI(x)               ((x >> 40) & 0xFFFFFFU)
#define PM4_ACQUIRE_MEM_DW7_GCR_CNTL(x)                    (((x) & 0x7FFFFU) << 0)
#  define PM4_ACQUIRE_MEM_GCR_CNTL_GLI_INV(x)              (((x) & 0x3U) << 0)
#  define PM4_ACQUIRE_MEM_GCR_CNTL_GLK_INV                 (1U << 7)
#  define PM4_ACQUIRE_MEM_GCR_CNTL_GLV_INV                 (1U << 8)
#  define PM4_ACQUIRE_MEM_GCR_CNTL_GL1_INV                 (1U << 9)
#  define PM4_ACQUIRE_MEM_GCR_CNTL_GL2_INV                 (1U << 14)
#  define PM4_ACQUIRE_MEM_GCR_CNTL_GL2_WB                  (1U << 15)
#define PM4_RELEASE_MEM_DW1_EVENT_INDEX(x)                 (((x) & 0xFU) << 8)
#  define PM4_RELEASE_MEM_EVENT_INDEX_AQL                  0x7U

#define PM4_ATOMIC_MEM_DW1_ATOMIC(x)                       (((x) & 0x7FU) << 0)
#  define PM4_ATOMIC_MEM_GL2_OP_ATOMIC_SWAP_RTN_64         (39U << 0)
#define PM4_ATOMIC_MEM_DW2_ADDR_LO(x)                      (((x) & 0xFFFFFFF8U) << 0)
#define PM4_ATOMIC_MEM_DW3_ADDR_HI(x)                      (((x) & 0xFFFFFFFFU) << 0)
#define PM4_ATOMIC_MEM_DW4_SRC_DATA_LO(x)                  (((x) & 0xFFFFFFFFU) << 0)
#define PM4_ATOMIC_MEM_DW5_SRC_DATA_HI(x)                  (((x) & 0xFFFFFFFFU) << 0)

#define PM4_PRED_EXEC_DW1_HEADER(x)                        (((x) & 0xFFFFFFFFU) << 0)
#define PM4_PRED_EXEC_DW2_EXEC_COUNT(x)                    (((x) & 0x3FFFU) << 0)
#define PM4_PRED_EXEC_DW2_VIRTUALXCCID_SELECT(x)           (((x) & 0xFFU) << 24)

#define PM4_COPY_DATA_DW1(x)                               (((x) & 0xFFFFFFFFU) << 0)
#  define PM4_COPY_DATA_SRC_SEL_ATOMIC_RETURN_DATA         (6U << 0)
#  define PM4_COPY_DATA_DST_SEL_TC_12                      (2U << 8)
#  define PM4_COPY_DATA_COUNT_SEL                          (1U << 16)
#  define PM4_COPY_DATA_WR_CONFIRM                         (1U << 20)
#define PM4_COPY_DATA_DW4_DST_ADDR_LO(x)                   (((x) & 0xFFFFFFF8U) << 0)
#define PM4_COPY_DATA_DW5_DST_ADDR_HI(x)                   (((x) & 0xFFFFFFFFU) << 0)

#define PM4_WAIT_REG_MEM_DW1(x)                            (((x) & 0xFFFFFFFFU) << 0)
#  define PM4_WAIT_REG_MEM_FUNCTION_EQUAL_TO_REFERENCE     (3U << 0)
#  define PM4_WAIT_REG_MEM_MEM_SPACE_MEMORY_SPACE          (1U << 4)
#  define PM4_WAIT_REG_MEM_OPERATION_WAIT_REG_MEM          (0U << 6)
#define PM4_WAIT_REG_MEM_DW2_MEM_POLL_ADDR_LO(x)           (((x) & 0xFFFFFFFCU) << 0)
#define PM4_WAIT_REG_MEM_DW3_MEM_POLL_ADDR_HI(x)           (((x) & 0xFFFFFFFFU) << 0)
#define PM4_WAIT_REG_MEM_DW4_REFERENCE(x)                  (((x) & 0xFFFFFFFFU) << 0)
#define PM4_WAIT_REG_MEM_DW6(x)                            (((x) & 0x8000FFFFU) << 0)
#  define PM4_WAIT_REG_MEM_POLL_INTERVAL(x)                (((x) & 0xFFFFU) << 0)
#  define PM4_WAIT_REG_MEM_OPTIMIZE_ACE_OFFLOAD_MODE       (1U << 31)

#define PM4_DMA_DATA_DW1(x)                            (((x) & 0xFFFFFFFFU) << 0)
#  define PM4_DMA_DATA_DST_SEL_DST_ADDR_USING_L2       (3U << 20)
#  define PM4_DMA_DATA_SRC_SEL_SRC_ADDR_USING_L2       (3U << 29)
#define PM4_DMA_DATA_DW2_SRC_ADDR_LO(x)                (((x) & 0xFFFFFFFFU) << 0)
#define PM4_DMA_DATA_DW3_SRC_ADDR_HI(x)                (((x) & 0xFFFFFFFFU) << 0)
#define PM4_DMA_DATA_DW4_DST_ADDR_LO(x)                (((x) & 0xFFFFFFFFU) << 0)
#define PM4_DMA_DATA_DW5_DST_ADDR_HI(x)                (((x) & 0xFFFFFFFFU) << 0)
#define PM4_DMA_DATA_DW6(x)                            (((x) & 0xFFFFFFFFU) << 0)
#  define PM4_DMA_DATA_BYTE_COUNT(x)                   (((x) & 0x3FFFFFFU) << 0)
#  define PM4_DMA_DATA_DIS_WC                          (1U << 31)
#  define PM4_DMA_DATA_DIS_WC_LAST                     (0U << 31)

#define PM4_WRITE_DATA_DW1(x)                          (((x) & 0xFFFFFF00U) << 0)
#  define PM4_WRITE_DATA_DST_SEL_TC_L2                 (2U << 8)
#  define PM4_WRITE_DATA_WR_CONFIRM_WAIT_CONFIRMATION  (1U << 20)
#define PM4_WRITE_DATA_DW2_DST_MEM_ADDR_LO(x)          (((x) & 0xFFFFFFFCU) << 0)
#define PM4_WRITE_DATA_DW3_DST_MEM_ADDR_HI(x)          (((x) & 0xFFFFFFFFU) << 0)
#define PM4_WRITE_DATA_DW4_DATA(x)                     (((x) & 0xFFFFFFFFU) << 0)

// clang-format on

#endif  // header guard
