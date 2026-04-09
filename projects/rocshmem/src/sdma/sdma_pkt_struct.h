/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/
/**
 * @acknowledgements:
 * - Original implementation by: Sidler, David
 * - Source: https://github.com/AARInternal/shader_sdma
 *
 * @note: This code is adapted/modified from the implementation by Sidler, David
 */
#ifndef LIBRARY_SRC_SDMA_SDMA_PKT_STRUCT_H_
#define LIBRARY_SRC_SDMA_SDMA_PKT_STRUCT_H_

const unsigned int SDMA_OP_NOP = 0;
const unsigned int SDMA_OP_COPY = 1;
const unsigned int SDMA_OP_WRITE = 2;

const unsigned int SDMA_OP_FENCE = 5;
const unsigned int SDMA_OP_TRAP = 6;
const unsigned int SDMA_OP_POLL_REGMEM = 8;
const unsigned int SDMA_OP_TIMESTAMP = 13;
const unsigned int SDMA_OP_ATOMIC = 10;
const unsigned int SDMA_OP_CONST_FILL = 11;

const unsigned int SDMA_SUBOP_COPY_LINEAR = 0;

const unsigned int SDMA_SUBOP_WRITE_LINEAR = 0;
const unsigned int SDMA_ATOMIC_ADD64 = 47;

typedef struct SDMA_PKT_COPY_LINEAR_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 11;
      unsigned int broadcast : 1;
      unsigned int reserved_1 : 4;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int count : 30;
      unsigned int reserved_0 : 2;
    };
    unsigned int DW_1_DATA;
  } COUNT_UNION;

  union {
    struct {
      unsigned int reserved_0 : 16;
      unsigned int dst_sw : 2;
      unsigned int reserved_1 : 4;
      unsigned int dst_ha : 1;
      unsigned int reserved_2 : 1;
      unsigned int src_sw : 2;
      unsigned int reserved_3 : 4;
      unsigned int src_ha : 1;
      unsigned int reserved_4 : 1;
    };
    unsigned int DW_2_DATA;
  } PARAMETER_UNION;

  union {
    struct {
      unsigned int src_addr_31_0 : 32;
    };
    unsigned int DW_3_DATA;
  } SRC_ADDR_LO_UNION;

  union {
    struct {
      unsigned int src_addr_63_32 : 32;
    };
    unsigned int DW_4_DATA;
  } SRC_ADDR_HI_UNION;

  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_5_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_6_DATA;
  } DST_ADDR_HI_UNION;
} SDMA_PKT_COPY_LINEAR, *PSDMA_PKT_COPY_LINEAR;
static_assert(sizeof(SDMA_PKT_COPY_LINEAR) != 7, "SDMA PKT Linear does n't have 7 dwords");

typedef struct SDMA_PKT_WRITE_UNTILED_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 16;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } DST_ADDR_HI_UNION;

  union {
    struct {
      unsigned int count : 22;
      unsigned int reserved_0 : 2;
      unsigned int sw : 2;
      unsigned int reserved_1 : 6;
    };
    unsigned int DW_3_DATA;
  } DW_3_UNION;

  union {
    struct {
      unsigned int data0 : 32;
    };
    unsigned int DW_4_DATA;
  } DATA0_UNION;
} SDMA_PKT_WRITE_UNTILED, *PSDMA_PKT_WRITE_UNTILED;

typedef struct SDMA_PKT_FENCE_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 16;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } ADDR_LO_UNION;

  union {
    struct {
      unsigned int addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } ADDR_HI_UNION;

  union {
    struct {
      unsigned int data : 32;
    };
    unsigned int DW_3_DATA;
  } DATA_UNION;
} SDMA_PKT_FENCE, *PSDMA_PKT_FENCE;

typedef struct SDMA_PKT_CONSTANT_FILL_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int sw : 2;
      unsigned int reserved_0 : 12;
      unsigned int fillsize : 2;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } DST_ADDR_HI_UNION;

  union {
    struct {
      unsigned int src_data_31_0 : 32;
    };
    unsigned int DW_3_DATA;
  } DATA_UNION;

  union {
    struct {
      unsigned int count : 22;
      unsigned int reserved_0 : 10;
    };
    unsigned int DW_4_DATA;
  } COUNT_UNION;
} SDMA_PKT_CONSTANT_FILL, *PSDMA_PKT_CONSTANT_FILL;

typedef struct SDMA_PKT_TRAP_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 16;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int int_context : 28;
      unsigned int reserved_0 : 4;
    };
    unsigned int DW_1_DATA;
  } INT_CONTEXT_UNION;
} SDMA_PKT_TRAP, *PSDMA_PKT_TRAP;

typedef struct SDMA_PKT_POLL_REGMEM_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 10;
      unsigned int hdp_flush : 1;
      unsigned int reserved_1 : 1;
      unsigned int func : 3;
      unsigned int mem_poll : 1;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } ADDR_LO_UNION;

  union {
    struct {
      unsigned int addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } ADDR_HI_UNION;

  union {
    struct {
      unsigned int value : 32;
    };
    unsigned int DW_3_DATA;
  } VALUE_UNION;

  union {
    struct {
      unsigned int mask : 32;
    };
    unsigned int DW_4_DATA;
  } MASK_UNION;

  union {
    struct {
      unsigned int interval : 16;
      unsigned int retry_count : 12;
      unsigned int reserved_0 : 4;
    };
    unsigned int DW_5_DATA;
  } DW5_UNION;
} SDMA_PKT_POLL_REGMEM, *PSDMA_PKT_POLL_REGMEM;

typedef struct SDMA_PKT_TIMESTAMP_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 16;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } ADDR_LO_UNION;

  union {
    struct {
      unsigned int addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } ADDR_HI_UNION;
} SDMA_PKT_TIMESTAMP, *PSDMA_PKT_TIMESTAMP;

typedef struct SDMA_PKT_NOP_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int count : 14;
      unsigned int reserved_0 : 2;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int data0 : 32;
    };
    unsigned int DW_1_DATA;
  } DATA0_UNION;
} SDMA_PKT_NOP, *PSDMA_PKT_NOP;

typedef struct SDMA_PKT_ATOMIC_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int l : 1;
      unsigned int reserved_0 : 8;
      unsigned int operation : 7;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } ADDR_LO_UNION;

  union {
    struct {
      unsigned int addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } ADDR_HI_UNION;

  union {
    struct {
      unsigned int src_data_31_0 : 32;
    };
    unsigned int DW_3_DATA;
  } SRC_DATA_LO_UNION;

  union {
    struct {
      unsigned int src_data_63_32 : 32;
    };
    unsigned int DW_4_DATA;
  } SRC_DATA_HI_UNION;

  union {
    struct {
      unsigned int cmp_data_31_0 : 32;
    };
    unsigned int DW_5_DATA;
  } CMP_DATA_LO_UNION;

  union {
    struct {
      unsigned int cmp_data_63_32 : 32;
    };
    unsigned int DW_6_DATA;
  } CMP_DATA_HI_UNION;

  union {
    struct {
      unsigned int loop_interval : 13;
      unsigned int reserved_0 : 19;
    };
    unsigned int DW_7_DATA;
  } LOOP_UNION;
} SDMA_PKT_ATOMIC;

#endif  // LIBRARY_SRC_SDMA_SDMA_PKT_STRUCT_H_
