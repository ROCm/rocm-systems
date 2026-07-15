////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HSA_RUNTIME_CORE_INC_SDMA_REGISTERS_H_
#define HSA_RUNTIME_CORE_INC_SDMA_REGISTERS_H_

#include <stddef.h>
#include <stdint.h>

namespace rocr {
namespace AMD {

// SDMA packet for VI device.
// Reference: http://people.freedesktop.org/~agd5f/dma_packets.txt

const unsigned int SDMA_OP_COPY = 1;
const unsigned int SDMA_OP_FENCE = 5;
const unsigned int SDMA_OP_TRAP = 6;
const unsigned int SDMA_OP_POLL_REGMEM = 8;
const unsigned int SDMA_OP_ATOMIC = 10;
const unsigned int SDMA_OP_CONST_FILL = 11;
const unsigned int SDMA_OP_TIMESTAMP = 13;
const unsigned int SDMA_OP_GCR = 17;
const unsigned int SDMA_SUBOP_COPY_LINEAR = 0;
// Broadcast linear copy uses the linear sub-op with the broadcast packet format.
const unsigned int SDMA_SUBOP_COPY_LINEAR_BROADCAST = SDMA_SUBOP_COPY_LINEAR;
// Indirect linear copy reuses the linear sub-op; the engine selects the indirect
// packet format from the INDIRECT_SRC/INDIRECT_DST bits in the header DW.
const unsigned int SDMA_SUBOP_COPY_INDIRECT = SDMA_SUBOP_COPY_LINEAR;
const unsigned int SDMA_SUBOP_COPY_LINEAR_RECT = 4;
const unsigned int SDMA_SUBOP_COPY_SWAP = 9;
const unsigned int SDMA_SUBOP_COPY_MULTICAST = 10;
const unsigned int SDMA_SUBOP_FENCE_64B = 2;
const unsigned int SDMA_SUBOP_POLL_MEM_64B = 5;
const unsigned int SDMA_SUBOP_TIMESTAMP_GET_GLOBAL = 2;
const unsigned int SDMA_SUBOP_USER_GCR = 1;
const unsigned int SDMA_ATOMIC_ADD64 = 47;

const unsigned int SDMA_MEMORY_SCOPE_CU = 0;  /* workgroup scope */
const unsigned int SDMA_MEMORY_SCOPE_SE = 1;  /* super-group/group-of-group */
const unsigned int SDMA_MEMORY_SCOPE_DEV = 2; /* device scope */
const unsigned int SDMA_MEMORY_SCOPE_SYS = 3; /* system scope */

// clang-format off
typedef struct SDMA_PKT_COPY_LINEAR_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 12;
      unsigned int npd : 1;
      unsigned int reserved_1 : 3;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int count : 22;
      unsigned int reserved_0 : 10;
    } count;
    struct {
      unsigned int count : 30;
      unsigned int reserved_0 : 2;
    } count_ext;
    unsigned int DW_1_DATA;
  } COUNT_UNION;

  union {
    struct {
      unsigned int reserved_0 : 18;
      unsigned int dst_scope : 2;
      unsigned int dst_temporal_hint : 3;
      unsigned int reserved_1 : 3;
      unsigned int src_scope : 2;
      unsigned int src_temporal_hint : 3;
      unsigned int reserved_2 : 1;
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

  static const size_t kMaxSize_ = 0x3fffe0;
} SDMA_PKT_COPY_LINEAR;

// linear copy (broadcast) packet (SDMA5.2+)
typedef struct SDMA_PKT_COPY_LINEAR_BROADCAST_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int enc : 1;
      unsigned int reserved_0 : 1;
      unsigned int tmz : 1;
      unsigned int reserved_1 : 8;
      unsigned int broadcast : 1;
      unsigned int alg : 2;
      unsigned int aes : 1;
      unsigned int reserved_2 : 1;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int count : 22;
      unsigned int reserved_0 : 10;
    } count;
    struct {
      unsigned int count : 30;
      unsigned int reserved_0 : 2;
    } count_ext;
    unsigned int DW_1_DATA;
  } COUNT_UNION;

  union {
    struct {
      unsigned int reserved_0 : 8;
      unsigned int dst2_swap : 2;
      unsigned int dst2_cache_policy : 3;
      unsigned int reserved_1 : 3;
      unsigned int dst1_swap : 2;
      unsigned int dst1_cache_policy : 3;
      unsigned int reserved_2 : 3;
      unsigned int src_swap : 2;
      unsigned int src_cache_policy : 3;
      unsigned int reserved_3 : 3;
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

  union {
    struct {
      unsigned int dst2_addr_31_0 : 32;
    };
    unsigned int DW_7_DATA;
  } DST2_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst2_addr_63_32 : 32;
    };
    unsigned int DW_8_DATA;
  } DST2_ADDR_HI_UNION;

  static const size_t kMaxSize_ = 0x3fffe0;
  static const size_t kDstAlignMask_ = 0x1F;
} SDMA_PKT_COPY_LINEAR_BROADCAST;

// linear copy (swap) packet (SDMA5.2+)
// Atomically swaps data between Address A and Address B.
// Addresses must be 64-byte aligned for gfx94/gfx95X.
typedef struct SDMA_PKT_COPY_LINEAR_SWAP_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int extra_info : 16;
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
      unsigned int dst_cache_policy : 3;
      unsigned int reserved_1 : 3;
      unsigned int src_sw : 2;
      unsigned int src_cache_policy : 3;
      unsigned int reserved_2 : 3;
    };
    unsigned int DW_2_DATA;
  } PARAMETER_UNION;

  union {
    struct {
      unsigned int reserved_0 : 6;
      unsigned int addr_a_31_6 : 26;
    };
    unsigned int DW_3_DATA;
  } ADDR_A_LO_UNION;

  union {
    struct {
      unsigned int addr_a_63_32 : 32;
    };
    unsigned int DW_4_DATA;
  } ADDR_A_HI_UNION;

  union {
    struct {
      unsigned int reserved_0 : 6;
      unsigned int addr_b_31_6 : 26;
    };
    unsigned int DW_5_DATA;
  } ADDR_B_LO_UNION;

  union {
    struct {
      unsigned int addr_b_63_32 : 32;
    };
    unsigned int DW_6_DATA;
  } ADDR_B_HI_UNION;

  static const size_t kMaxSize_ = 0x3fffffe0;
  static const size_t kAlignment_ = 64;
} SDMA_PKT_COPY_LINEAR_SWAP;

// linear sub-window (pre-GFX12)
typedef struct SDMA_PKT_COPY_LINEAR_RECT_TAG {
  static const unsigned int pitch_bits = 19;
  static const unsigned int slice_bits = 28;
  static const unsigned int rect_xy_bits = 14;
  static const unsigned int rect_z_bits = 11;

  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved : 13;
      unsigned int element : 3;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int src_addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } SRC_ADDR_LO_UNION;

  union {
    struct {
      unsigned int src_addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } SRC_ADDR_HI_UNION;

  union {
    struct {
      unsigned int src_offset_x : 14;
      unsigned int reserved_1 : 2;
      unsigned int src_offset_y : 14;
      unsigned int reserved_2 : 2;
    };
    unsigned int DW_3_DATA;
  } SRC_PARAMETER_1_UNION;

  union {
    struct {
      unsigned int src_offset_z : 11;
      unsigned int reserved_1 : 2;
      unsigned int src_pitch : pitch_bits;
    };
    unsigned int DW_4_DATA;
  } SRC_PARAMETER_2_UNION;

  union {
    struct {
      unsigned int src_slice_pitch : slice_bits;
      unsigned int reserved_1 : 4;
    };
    unsigned int DW_5_DATA;
  } SRC_PARAMETER_3_UNION;

  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_6_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_7_DATA;
  } DST_ADDR_HI_UNION;

  union {
    struct {
      unsigned int dst_offset_x : 14;
      unsigned int reserved_1 : 2;
      unsigned int dst_offset_y : 14;
      unsigned int reserved_2 : 2;
    };
    unsigned int DW_8_DATA;
  } DST_PARAMETER_1_UNION;

  union {
    struct {
      unsigned int dst_offset_z : 11;
      unsigned int reserved_1 : 2;
      unsigned int dst_pitch : pitch_bits;
    };
    unsigned int DW_9_DATA;
  } DST_PARAMETER_2_UNION;

  union {
    struct {
      unsigned int dst_slice_pitch : slice_bits;
      unsigned int reserved_1 : 4;
    };
    unsigned int DW_10_DATA;
  } DST_PARAMETER_3_UNION;

  union {
    struct {
      unsigned int rect_x : rect_xy_bits;
      unsigned int reserved_1 : 2;
      unsigned int rect_y : rect_xy_bits;
      unsigned int reserved_2 : 2;
    };
    unsigned int DW_11_DATA;
  } RECT_PARAMETER_1_UNION;

  union {
    struct {
      unsigned int rect_z : rect_z_bits;
      unsigned int reserved_1 : 5;
      unsigned int dst_swap : 2;
      unsigned int reserved_2 : 6;
      unsigned int src_swap : 2;
      unsigned int reserved_3 : 6;
    };
    unsigned int DW_12_DATA;
  } RECT_PARAMETER_2_UNION;

} SDMA_PKT_COPY_LINEAR_RECT;

// linear sub-window (GFX12)
typedef struct SDMA_PKT_COPY_LINEAR_RECT_TAG_GFX12 {
  static const unsigned int pitch_bits   = 16;
  static const unsigned int slice_bits   = 32;
  static const unsigned int rect_xy_bits = 16;
  static const unsigned int rect_z_bits  = 14;

  union {
    struct {
      unsigned int op       :  8;
      unsigned int sub_op   :  8;
      unsigned int reserved : 12;
      unsigned int npd      :  1;
      unsigned int element  :  3;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int src_addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } SRC_ADDR_LO_UNION;

  union {
    struct {
      unsigned int src_addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } SRC_ADDR_HI_UNION;

  union {
    struct {
      unsigned int src_offset_x : 16;
      unsigned int src_offset_y : 16;
    };
    unsigned int DW_3_DATA;
  } SRC_PARAMETER_1_UNION;

  union {
    struct {
      unsigned int src_offset_z : 14;
      unsigned int reserved_1   : 2;
      unsigned int src_pitch    : pitch_bits;
    };
    unsigned int DW_4_DATA;
  } SRC_PARAMETER_2_UNION;

  union {
    struct {
      unsigned int src_slice_pitch : slice_bits;
    };
    unsigned int DW_5_DATA;
  } SRC_PARAMETER_3_UNION;

  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_6_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_7_DATA;
  } DST_ADDR_HI_UNION;

  union {
    struct {
      unsigned int dst_offset_x : 16;
      unsigned int dst_offset_y : 16;
    };
    unsigned int DW_8_DATA;
  } DST_PARAMETER_1_UNION;

  union {
    struct {
      unsigned int dst_offset_z : 14;
      unsigned int reserved_1   : 2;
      unsigned int dst_pitch    : pitch_bits;
    };
    unsigned int DW_9_DATA;
  } DST_PARAMETER_2_UNION;

  union {
    struct {
      unsigned int dst_slice_pitch : slice_bits;
    };
    unsigned int DW_10_DATA;
  } DST_PARAMETER_3_UNION;

  union {
    struct {
      unsigned int rect_x : rect_xy_bits;
      unsigned int rect_y : rect_xy_bits;
      };
    unsigned int DW_11_DATA;
  } RECT_PARAMETER_1_UNION;

  union {
    struct {
      unsigned int rect_z           : rect_z_bits;
      unsigned int reserved_1       : 6;
      unsigned int dst_cache_policy : 3;
      unsigned int reserved_2       : 5;
      unsigned int src_cache_policy : 3;
      unsigned int reserved_3       : 1;
    } gfx12;
    struct {
      unsigned int rect_z           : rect_z_bits;
      unsigned int reserved_1       : 4;
      unsigned int dst_scope        : 2;
      unsigned int dst_temporal_hint: 3;
      unsigned int reserved_2       : 3;
      unsigned int src_scope        : 2;
      unsigned int src_temporal_hint: 3;
      unsigned int reserved_3       : 1;
    } gfx1250;
    unsigned int DW_12_DATA;
  } RECT_PARAMETER_2_UNION;

} SDMA_PKT_COPY_LINEAR_RECT_GFX12;

typedef struct SDMA_PKT_CONSTANT_FILL_TAG {
  union {
    struct {
      unsigned int op           : 8;
      unsigned int sub_op       : 8;
      unsigned int mtype        : 2;  /* gfx1250*/
      unsigned int reserved_0   : 2;
      unsigned int sys          : 1;  /* gfx1250*/
      unsigned int reserved_1   : 1;
      unsigned int snp          : 1;  /* gfx1250*/
      unsigned int gpa          : 1;  /* gfx1250*/
      unsigned int scope        : 2;  /* gfx1250*/
      unsigned int temporal_hint: 3;  /* gfx1250*/
      unsigned int npd          : 1;  /* gfx1250*/
      unsigned int fillsize     : 2;
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

  static const size_t kMaxSize_ = 0x3fffe0;
} SDMA_PKT_CONSTANT_FILL;

typedef struct SDMA_PKT_FENCE_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int mtype : 3;
      unsigned int gcc : 1;
      unsigned int sys : 1;
      unsigned int pad1 : 1;
      unsigned int snp : 1;
      unsigned int gpa : 1;
      unsigned int l2_policy : 2;
      unsigned int reserved_0 : 6;
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
} SDMA_PKT_FENCE;

typedef struct SDMA_PKT_FENCE_TAG_GFX12 {
  union {
    struct {
      unsigned int op           : 8;
      unsigned int sub_op       : 8;
      unsigned int mtype        : 2;
      unsigned int pad1         : 2;
      unsigned int sys          : 1;
      unsigned int pad2         : 1;
      unsigned int snp          : 1;
      unsigned int gpa          : 1;
      unsigned int scope        : 2;
      unsigned int temporal_hint: 3;
      unsigned int reserved_0   : 3;
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
} SDMA_PKT_FENCE_GFX12;

// Fence 64b (GFX1250)
// MAS 3.4.5 - Writes 64-bit data to a QW-aligned address.
// OP=0x5, SUBOP=0x2, 5 DWs.
typedef struct SDMA_PKT_FENCE_64B_TAG_GFX1250 {
  union {
    struct {
      unsigned int op             :  8;
      unsigned int sub_op         :  8;
      unsigned int mtype          :  2;
      unsigned int reserved_0     :  2;
      unsigned int sys            :  1;
      unsigned int reserved_1     :  1;
      unsigned int snp            :  1;
      unsigned int gpa            :  1;
      unsigned int scope          :  2;
      unsigned int temporal_hint  :  3;
      unsigned int reserved_2     :  3;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int reserved_0  :  3;
      unsigned int addr_31_3   : 29;
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
      unsigned int data_31_0 : 32;
    };
    unsigned int DW_3_DATA;
  } DATA_LO_UNION;

  union {
    struct {
      unsigned int data_63_32 : 32;
    };
    unsigned int DW_4_DATA;
  } DATA_HI_UNION;
} SDMA_PKT_FENCE_64B_GFX1250;

typedef struct SDMA_PKT_POLL_REGMEM_TAG {
  union {
    struct {
      unsigned int op         : 8;
      unsigned int sub_op     : 8;
      unsigned int reserved_0 : 10;
      unsigned int hdp_flush  : 1;
      unsigned int reserved_1 : 1;
      unsigned int func       : 3;
      unsigned int mem_poll   : 1;
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
      unsigned int interval     : 16;
      unsigned int retry_count  : 12;
      unsigned int scope        : 2;
      unsigned int reserved_0   : 2;
    };
    unsigned int DW_5_DATA;
  } DW5_UNION;
} SDMA_PKT_POLL_REGMEM;

// Poll Memory 64b (GFX1250)
// MAS 3.4.9 - Polls a QW-aligned 64-bit memory location.
// Condition: FUNC(mem_data_64b & MASK_64b, REFERENCE_64b) == TRUE
// OP=0x8, SUBOP=0x5, 8 DWs.
typedef struct SDMA_PKT_POLL_MEM_64B_TAG_GFX1250 {
  union {
    struct {
      unsigned int op             :  8;
      unsigned int sub_op         :  8;
      unsigned int mtype          :  2;
      unsigned int reserved_0     :  2;
      unsigned int sys            :  1;
      unsigned int reserved_1     :  1;
      unsigned int snp            :  1;
      unsigned int gpa            :  1;
      unsigned int cache_policy   :  3;
      unsigned int reserved_2     :  1;
      unsigned int func           :  3;
      unsigned int reserved_3     :  1;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int reserved_0  :  3;
      unsigned int addr_31_3   : 29;
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
      unsigned int reference_31_0 : 32;
    };
    unsigned int DW_3_DATA;
  } REFERENCE_LO_UNION;

  union {
    struct {
      unsigned int reference_63_32 : 32;
    };
    unsigned int DW_4_DATA;
  } REFERENCE_HI_UNION;

  union {
    struct {
      unsigned int mask_31_0 : 32;
    };
    unsigned int DW_5_DATA;
  } MASK_LO_UNION;

  union {
    struct {
      unsigned int mask_63_32 : 32;
    };
    unsigned int DW_6_DATA;
  } MASK_HI_UNION;

  union {
    struct {
      unsigned int reserved_0   : 16;
      unsigned int retry_count  :  8;
      unsigned int reserved_1   :  4;
      unsigned int scope        :  2;
      unsigned int reserved_2   :  2;
    };
    unsigned int DW_7_DATA;
  } DW7_UNION;
} SDMA_PKT_POLL_MEM_64B_GFX1250;

typedef struct SDMA_PKT_ATOMIC_TAG {
  union {
    struct {
      unsigned int op           : 8;
      unsigned int sub_op       : 8;
      unsigned int l            : 1;
      unsigned int reserved_0   : 1;
      unsigned int tmz          : 1;
      unsigned int reserved_1   : 1;
      unsigned int scope        : 2;
      unsigned int temporal_hint: 3;
      unsigned int operation    : 7;
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
      unsigned int loop_interval  : 13;
      unsigned int reserved_0     : 19;
    };
    unsigned int DW_7_DATA;
  } LOOP_UNION;
} SDMA_PKT_ATOMIC;

typedef struct SDMA_PKT_TIMESTAMP_TAG {
  union {
    struct {
      unsigned int op           : 8;
      unsigned int sub_op       : 8;
      unsigned int reserved     : 8;
      unsigned int scope        : 2;
      unsigned int temporal_hint: 3;
      unsigned int reserved_0   : 3;
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

} SDMA_PKT_TIMESTAMP;

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
      unsigned int int_ctx : 28;
      unsigned int reserved_1 : 4;
    };
    unsigned int DW_1_DATA;
  } INT_CONTEXT_UNION;
} SDMA_PKT_TRAP;

// HDP flush packet, no parameters.
typedef struct SDMA_PKT_HDP_FLUSH_TAG {
  unsigned int DW_0_DATA;
  unsigned int DW_1_DATA;
  unsigned int DW_2_DATA;
  unsigned int DW_3_DATA;
  unsigned int DW_4_DATA;
  unsigned int DW_5_DATA;

  // Version of gfx9 sDMA microcode introducing SDMA_PKT_HDP_FLUSH
  static const uint16_t kMinVersion_ = 0x1A5;
} SDMA_PKT_HDP_FLUSH;
static const SDMA_PKT_HDP_FLUSH hdp_flush_cmd = {0x8, 0x0, 0x80000000, 0x0, 0x0, 0x0};

typedef struct SDMA_PKT_GCR_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int : 16;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int : 7;
      unsigned int BaseVA_LO : 25;
    };
    unsigned int DW_1_DATA;
  } WORD1_UNION;

  union {
    struct {
      unsigned int BaseVA_HI              : 16;
      unsigned int GCR_CONTROL_GLI_INV    : 2;
      unsigned int GCR_CONTROL_GL1_RANGE  : 2;
      unsigned int GCR_CONTROL_GLM_WB     : 1;
      unsigned int GCR_CONTROL_GLM_INV    : 1;
      unsigned int GCR_CONTROL_GLK_WB     : 1;
      unsigned int GCR_CONTROL_GLK_INV    : 1;
      unsigned int GCR_CONTROL_GLV_INV    : 1;
      unsigned int GCR_CONTROL_GL1_INV    : 1;
      unsigned int GCR_CONTROL_GL2_US     : 1;
      unsigned int GCR_CONTROL_GL2_RANGE  : 2;
      unsigned int GCR_CONTROL_GL2_DISCARD: 1;
      unsigned int GCR_CONTROL_GL2_INV    : 1;
      unsigned int GCR_CONTROL_GL2_WB     : 1;
    };
    unsigned int DW_2_DATA;
  } WORD2_UNION;

  union {
    struct {
      unsigned int GCR_CONTROL_RANGE_IS_PA  : 1;
      unsigned int GCR_CONTROL_SEQ          : 2;
      unsigned int                          : 4;
      unsigned int LimitVA_LO               : 25;
    };
    unsigned int DW_3_DATA;
  } WORD3_UNION;

  union {
    struct {
      unsigned int LimitVA_HI               : 16;
      unsigned int                          : 8;
      unsigned int VMID                     : 4;
      unsigned int                          : 4;
    };
    unsigned int DW_4_DATA;
  } WORD4_UNION;
} SDMA_PKT_GCR;

typedef struct SDMA_PKT_GCR_TAG_GFX1250 {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int : 16;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int : 7;
      unsigned int BaseVA_LO : 25;
    };
    unsigned int DW_1_DATA;
  } WORD1_UNION;

  union {
    struct {
      unsigned int BaseVA_HI                : 25;
      unsigned int                          : 7;
    };
    unsigned int DW_2_DATA;
  } WORD2_UNION;

  union {
    struct {
      unsigned int GCR_CONTROL_GLI_INV      : 2;
      unsigned int GCR_CONTROL_GL1_RANGE    : 2;
      unsigned int GCR_CONTROL_GL2_SCOPE    : 2;
      unsigned int GCR_CONTROL_GLV_WB       : 1;
      unsigned int GCR_CONTROL_GLK_INV      : 1;
      unsigned int GCR_CONTROL_GLV_INV      : 1;
      unsigned int                          : 1;
      unsigned int GCR_CONTROL_GL2_US       : 1;
      unsigned int GCR_CONTROL_GL2_RANGE    : 2;
      unsigned int GCR_CONTROL_GL2_DISCARD  : 1;
      unsigned int GCR_CONTROL_GL2_INV      : 1;
      unsigned int GCR_CONTROL_GL2_WB       : 1;
      unsigned int GCR_CONTROL_SEQ          : 2;
      unsigned int GCR_CONTROL_RANGE_IS_PA  : 1;
      unsigned int                          : 4;
      unsigned int LimitVA_LO               : 9;
    };
    unsigned int DW_3_DATA;
  } WORD3_UNION;

  union {
    struct {
      unsigned int LimitVA_MID              : 32;
    };
    unsigned int DW_4_DATA;
  } WORD4_UNION;

  union {
    struct {
      unsigned int LimitVA_HI               : 9;
      unsigned int                          : 17;
      unsigned int VMID                     : 4;
      unsigned int                          : 2;
    };
    unsigned int DW_5_DATA;
  } WORD5_UNION;
} SDMA_PKT_GCR_GFX1250;

// Linear Swap Copy (GFX1250)
// Atomically swaps data between Address A and Address B.
// Addresses must be 32-byte aligned.
typedef struct SDMA_PKT_COPY_LINEAR_SWAP_TAG_GFX1250 {
  union {
    struct {
      unsigned int op          :  8;
      unsigned int sub_op      :  8;
      unsigned int reserved_0  :  2;
      unsigned int tmz         :  1;
      unsigned int reserved_1  : 13;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int count       : 30;
      unsigned int reserved_0  :  2;
    };
    unsigned int DW_1_DATA;
  } COUNT_UNION;

  union {
    struct {
      unsigned int reserved_0       : 18;
      unsigned int scope_b          :  2;
      unsigned int temporal_hint_b  :  3;
      unsigned int reserved_1       :  3;
      unsigned int scope_a          :  2;
      unsigned int temporal_hint_a  :  3;
      unsigned int reserved_2       :  1;
    };
    unsigned int DW_2_DATA;
  } PARAMETER_UNION;

  union {
    struct {
      unsigned int reserved_0  :  5;
      unsigned int addr_a_31_5 : 27;
    };
    unsigned int DW_3_DATA;
  } ADDR_A_LO_UNION;

  union {
    struct {
      unsigned int addr_a_63_32 : 32;
    };
    unsigned int DW_4_DATA;
  } ADDR_A_HI_UNION;

  union {
    struct {
      unsigned int reserved_0  :  5;
      unsigned int addr_b_31_5 : 27;
    };
    unsigned int DW_5_DATA;
  } ADDR_B_LO_UNION;

  union {
    struct {
      unsigned int addr_b_63_32 : 32;
    };
    unsigned int DW_6_DATA;
  } ADDR_B_HI_UNION;

  static const size_t kMaxSize_ = 0x3fffffe0;
  static const size_t kAlignment_ = 32;
} SDMA_PKT_COPY_LINEAR_SWAP_GFX1250;

// Linear Multicast Copy (GFX1250)
// Copies from one source to multiple destinations.
// Destination address pairs (lo/hi) follow contiguously for num_of_destination entries.
typedef struct SDMA_PKT_COPY_LINEAR_MULTICAST_TAG_GFX1250 {
  union {
    struct {
      unsigned int op          :  8;
      unsigned int sub_op      :  8;
      unsigned int reserved_0  :  2;
      unsigned int tmz         :  1;
      unsigned int reserved_1  :  9;
      unsigned int npd         :  1;
      unsigned int reserved_2  :  3;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int count       : 30;
      unsigned int reserved_0  :  2;
    };
    unsigned int DW_1_DATA;
  } COUNT_UNION;

  union {
    struct {
      unsigned int num_of_destination :  10;
      unsigned int reserved_0        :   8;
      unsigned int dst_scope         :   2;
      unsigned int dst_temporal_hint :   3;
      unsigned int reserved_1        :   3;
      unsigned int src_scope         :   2;
      unsigned int src_temporal_hint :   3;
      unsigned int reserved_2        :   1;
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

} SDMA_PKT_COPY_LINEAR_MULTICAST_GFX1250;

// Linear Copy with Wait/Signal (GFX1250)
// Macro command combining optional wait, linear copy, and optional signal.
// DW layout below assumes both WAIT=1 and SIGNAL=1 (maximum packet size).
// When WAIT=0, DW1-DW7 are absent and copy DWs shift down.
// When SIGNAL=0, DW14-DW18 are absent.
typedef struct SDMA_PKT_COPY_LINEAR_WAITSIGNAL_TAG_GFX1250 {
  union {
    struct {
      unsigned int op          :  8;
      unsigned int sub_op      :  8;
      unsigned int reserved_0  :  2;
      unsigned int tmz         :  1;
      unsigned int reserved_1  :  9;
      unsigned int npd         :  1;
      unsigned int reserved_2  :  1;
      unsigned int wait        :  1;
      unsigned int signal      :  1;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int wait_function      :  3;
      unsigned int reserved_0         : 15;
      unsigned int wait_scope         :  2;
      unsigned int wait_temporal_hint :  3;
      unsigned int reserved_1         :  9;
    };
    unsigned int DW_1_DATA;
  } WAIT_FUNCTION_UNION;

  union {
    struct {
      unsigned int reserved_0     :  3;
      unsigned int wait_addr_31_3 : 29;
    };
    unsigned int DW_2_DATA;
  } WAIT_ADDR_LO_UNION;

  union {
    struct {
      unsigned int wait_addr_63_32 : 32;
    };
    unsigned int DW_3_DATA;
  } WAIT_ADDR_HI_UNION;

  union {
    struct {
      unsigned int wait_reference_31_0 : 32;
    };
    unsigned int DW_4_DATA;
  } WAIT_REFERENCE_LO_UNION;

  union {
    struct {
      unsigned int wait_reference_63_32 : 32;
    };
    unsigned int DW_5_DATA;
  } WAIT_REFERENCE_HI_UNION;

  union {
    struct {
      unsigned int wait_mask_31_0 : 32;
    };
    unsigned int DW_6_DATA;
  } WAIT_MASK_LO_UNION;

  union {
    struct {
      unsigned int wait_mask_63_32 : 32;
    };
    unsigned int DW_7_DATA;
  } WAIT_MASK_HI_UNION;

  union {
    struct {
      unsigned int copy_count  : 30;
      unsigned int reserved_0  :  2;
    };
    unsigned int DW_8_DATA;
  } COPY_COUNT_UNION;

  union {
    struct {
      unsigned int reserved_0       : 18;
      unsigned int dst_scope        :  2;
      unsigned int dst_temporal_hint:  3;
      unsigned int reserved_1       :  3;
      unsigned int src_scope        :  2;
      unsigned int src_temporal_hint:  3;
      unsigned int reserved_2       :  1;
    };
    unsigned int DW_9_DATA;
  } COPY_PARAMETER_UNION;

  union {
    struct {
      unsigned int src_addr_31_0 : 32;
    };
    unsigned int DW_10_DATA;
  } SRC_ADDR_LO_UNION;

  union {
    struct {
      unsigned int src_addr_63_32 : 32;
    };
    unsigned int DW_11_DATA;
  } SRC_ADDR_HI_UNION;

  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_12_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_13_DATA;
  } DST_ADDR_HI_UNION;

  union {
    struct {
      unsigned int signal_operation    :  7;
      unsigned int reserved_0          : 11;
      unsigned int signal_scope        :  2;
      unsigned int signal_temporal_hint:  3;
      unsigned int reserved_1          :  9;
    };
    unsigned int DW_14_DATA;
  } SIGNAL_OPERATION_UNION;

  union {
    struct {
      unsigned int reserved_0        :  3;
      unsigned int signal_addr_31_3  : 29;
    };
    unsigned int DW_15_DATA;
  } SIGNAL_ADDR_LO_UNION;

  union {
    struct {
      unsigned int signal_addr_63_32 : 32;
    };
    unsigned int DW_16_DATA;
  } SIGNAL_ADDR_HI_UNION;

  union {
    struct {
      unsigned int signal_data_31_0 : 32;
    };
    unsigned int DW_17_DATA;
  } SIGNAL_DATA_LO_UNION;

  union {
    struct {
      unsigned int signal_data_63_32 : 32;
    };
    unsigned int DW_18_DATA;
  } SIGNAL_DATA_HI_UNION;

} SDMA_PKT_COPY_LINEAR_WAITSIGNAL_GFX1250;

// Linear Copy with Wait/Signal - Indirect Memory Address (GFX1250)
// Like Linear Copy with Wait/Signal but source/destination addresses can be indirect pointers.
// DW layout below assumes both WAIT=1 and SIGNAL=1 (maximum packet size).
typedef struct SDMA_PKT_COPY_LINEAR_WAITSIGNAL_INDIRECT_TAG_GFX1250 {
  union {
    struct {
      unsigned int op           :  8;
      unsigned int sub_op       :  8;
      unsigned int reserved_0   :  2;
      unsigned int tmz          :  1;
      unsigned int reserved_1   :  1;
      unsigned int indirect_src :  1;
      unsigned int indirect_dst :  1;
      unsigned int reserved_2   :  6;
      unsigned int npd          :  1;
      unsigned int reserved_3   :  1;
      unsigned int wait         :  1;
      unsigned int signal       :  1;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int wait_function      :  3;
      unsigned int reserved_0         : 15;
      unsigned int wait_scope         :  2;
      unsigned int wait_temporal_hint :  3;
      unsigned int reserved_1         :  9;
    };
    unsigned int DW_1_DATA;
  } WAIT_FUNCTION_UNION;

  union {
    struct {
      unsigned int reserved_0     :  3;
      unsigned int wait_addr_31_3 : 29;
    };
    unsigned int DW_2_DATA;
  } WAIT_ADDR_LO_UNION;

  union {
    struct {
      unsigned int wait_addr_63_32 : 32;
    };
    unsigned int DW_3_DATA;
  } WAIT_ADDR_HI_UNION;

  union {
    struct {
      unsigned int wait_reference_31_0 : 32;
    };
    unsigned int DW_4_DATA;
  } WAIT_REFERENCE_LO_UNION;

  union {
    struct {
      unsigned int wait_reference_63_32 : 32;
    };
    unsigned int DW_5_DATA;
  } WAIT_REFERENCE_HI_UNION;

  union {
    struct {
      unsigned int wait_mask_31_0 : 32;
    };
    unsigned int DW_6_DATA;
  } WAIT_MASK_LO_UNION;

  union {
    struct {
      unsigned int wait_mask_63_32 : 32;
    };
    unsigned int DW_7_DATA;
  } WAIT_MASK_HI_UNION;

  union {
    struct {
      unsigned int copy_count  : 30;
      unsigned int reserved_0  :  2;
    };
    unsigned int DW_8_DATA;
  } COPY_COUNT_UNION;

  union {
    struct {
      unsigned int reserved_0              : 10;
      unsigned int indirect_addr_scope     :  2;
      unsigned int indirect_temporal_hint  :  3;
      unsigned int reserved_1              :  3;
      unsigned int copy_dst_scope          :  2;
      unsigned int copy_dst_temporal_hint  :  3;
      unsigned int reserved_2              :  3;
      unsigned int copy_src_scope          :  2;
      unsigned int copy_src_temporal_hint  :  3;
      unsigned int reserved_3              :  1;
    };
    unsigned int DW_9_DATA;
  } COPY_PARAMETER_UNION;

  union {
    struct {
      unsigned int copy_src_addr_31_0 : 32;
    };
    unsigned int DW_10_DATA;
  } SRC_ADDR_LO_UNION;

  union {
    struct {
      unsigned int copy_src_addr_63_32 : 32;
    };
    unsigned int DW_11_DATA;
  } SRC_ADDR_HI_UNION;

  union {
    struct {
      unsigned int copy_dst_addr_31_0 : 32;
    };
    unsigned int DW_12_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int copy_dst_addr_63_32 : 32;
    };
    unsigned int DW_13_DATA;
  } DST_ADDR_HI_UNION;

  union {
    struct {
      unsigned int signal_operation    :  7;
      unsigned int reserved_0          : 11;
      unsigned int signal_scope        :  2;
      unsigned int signal_temporal_hint:  3;
      unsigned int reserved_1          :  9;
    };
    unsigned int DW_14_DATA;
  } SIGNAL_OPERATION_UNION;

  union {
    struct {
      unsigned int reserved_0        :  3;
      unsigned int signal_addr_31_3  : 29;
    };
    unsigned int DW_15_DATA;
  } SIGNAL_ADDR_LO_UNION;

  union {
    struct {
      unsigned int signal_addr_63_32 : 32;
    };
    unsigned int DW_16_DATA;
  } SIGNAL_ADDR_HI_UNION;

  union {
    struct {
      unsigned int signal_data_31_0 : 32;
    };
    unsigned int DW_17_DATA;
  } SIGNAL_DATA_LO_UNION;

  union {
    struct {
      unsigned int signal_data_63_32 : 32;
    };
    unsigned int DW_18_DATA;
  } SIGNAL_DATA_HI_UNION;

} SDMA_PKT_COPY_LINEAR_WAITSIGNAL_INDIRECT_GFX1250;

// Linear Multicast Copy with Wait/Signal (GFX1250)
// Macro command combining optional wait, multicast copy, and optional signal.
// DW layout below assumes both WAIT=1 and SIGNAL=1 (maximum packet size).
// Additional destination address pairs follow DST_ADDR for num_of_destination entries.
typedef struct SDMA_PKT_COPY_LINEAR_MULTICAST_WAITSIGNAL_TAG_GFX1250 {
  union {
    struct {
      unsigned int op          :  8;
      unsigned int sub_op      :  8;
      unsigned int reserved_0  :  2;
      unsigned int tmz         :  1;
      unsigned int reserved_1  :  9;
      unsigned int npd         :  1;
      unsigned int reserved_2  :  1;
      unsigned int wait        :  1;
      unsigned int signal      :  1;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int wait_function      :  3;
      unsigned int reserved_0         : 15;
      unsigned int wait_scope         :  2;
      unsigned int wait_temporal_hint :  3;
      unsigned int reserved_1         :  9;
    };
    unsigned int DW_1_DATA;
  } WAIT_FUNCTION_UNION;

  union {
    struct {
      unsigned int reserved_0     :  3;
      unsigned int wait_addr_31_3 : 29;
    };
    unsigned int DW_2_DATA;
  } WAIT_ADDR_LO_UNION;

  union {
    struct {
      unsigned int wait_addr_63_32 : 32;
    };
    unsigned int DW_3_DATA;
  } WAIT_ADDR_HI_UNION;

  union {
    struct {
      unsigned int wait_reference_31_0 : 32;
    };
    unsigned int DW_4_DATA;
  } WAIT_REFERENCE_LO_UNION;

  union {
    struct {
      unsigned int wait_reference_63_32 : 32;
    };
    unsigned int DW_5_DATA;
  } WAIT_REFERENCE_HI_UNION;

  union {
    struct {
      unsigned int wait_mask_31_0 : 32;
    };
    unsigned int DW_6_DATA;
  } WAIT_MASK_LO_UNION;

  union {
    struct {
      unsigned int wait_mask_63_32 : 32;
    };
    unsigned int DW_7_DATA;
  } WAIT_MASK_HI_UNION;

  union {
    struct {
      unsigned int count       : 30;
      unsigned int reserved_0  :  2;
    };
    unsigned int DW_8_DATA;
  } COUNT_UNION;

  union {
    struct {
      unsigned int num_of_destination :  10;
      unsigned int reserved_0        :   8;
      unsigned int dst_scope         :   2;
      unsigned int dst_temporal_hint :   3;
      unsigned int reserved_1        :   3;
      unsigned int src_scope         :   2;
      unsigned int src_temporal_hint :   3;
      unsigned int reserved_2        :   1;
    };
    unsigned int DW_9_DATA;
  } COPY_PARAMETER_UNION;

  union {
    struct {
      unsigned int src_addr_31_0 : 32;
    };
    unsigned int DW_10_DATA;
  } SRC_ADDR_LO_UNION;

  union {
    struct {
      unsigned int src_addr_63_32 : 32;
    };
    unsigned int DW_11_DATA;
  } SRC_ADDR_HI_UNION;

  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_12_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_13_DATA;
  } DST_ADDR_HI_UNION;

  union {
    struct {
      unsigned int signal_operation    :  7;
      unsigned int reserved_0          : 11;
      unsigned int signal_scope        :  2;
      unsigned int signal_temporal_hint:  3;
      unsigned int reserved_1          :  9;
    };
    unsigned int DW_14_DATA;
  } SIGNAL_OPERATION_UNION;

  union {
    struct {
      unsigned int reserved_0        :  3;
      unsigned int signal_addr_31_3  : 29;
    };
    unsigned int DW_15_DATA;
  } SIGNAL_ADDR_LO_UNION;

  union {
    struct {
      unsigned int signal_addr_63_32 : 32;
    };
    unsigned int DW_16_DATA;
  } SIGNAL_ADDR_HI_UNION;

  union {
    struct {
      unsigned int signal_data_31_0 : 32;
    };
    unsigned int DW_17_DATA;
  } SIGNAL_DATA_LO_UNION;

  union {
    struct {
      unsigned int signal_data_63_32 : 32;
    };
    unsigned int DW_18_DATA;
  } SIGNAL_DATA_HI_UNION;

} SDMA_PKT_COPY_LINEAR_MULTICAST_WAITSIGNAL_GFX1250;

// Linear Swap Copy with Wait/Signal (GFX1250)
// Macro command combining optional wait, swap copy, and optional signal.
// DW layout below assumes both WAIT=1 and SIGNAL=1 (maximum packet size).
// Swap addresses must be 32-byte aligned.
typedef struct SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_TAG_GFX1250 {
  union {
    struct {
      unsigned int op          :  8;
      unsigned int sub_op      :  8;
      unsigned int reserved_0  :  2;
      unsigned int tmz         :  1;
      unsigned int reserved_1  : 11;
      unsigned int wait        :  1;
      unsigned int signal      :  1;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int wait_function      :  3;
      unsigned int reserved_0         : 15;
      unsigned int wait_scope         :  2;
      unsigned int wait_temporal_hint :  3;
      unsigned int reserved_1         :  9;
    };
    unsigned int DW_1_DATA;
  } WAIT_FUNCTION_UNION;

  union {
    struct {
      unsigned int reserved_0     :  3;
      unsigned int wait_addr_31_3 : 29;
    };
    unsigned int DW_2_DATA;
  } WAIT_ADDR_LO_UNION;

  union {
    struct {
      unsigned int wait_addr_63_32 : 32;
    };
    unsigned int DW_3_DATA;
  } WAIT_ADDR_HI_UNION;

  union {
    struct {
      unsigned int wait_reference_31_0 : 32;
    };
    unsigned int DW_4_DATA;
  } WAIT_REFERENCE_LO_UNION;

  union {
    struct {
      unsigned int wait_reference_63_32 : 32;
    };
    unsigned int DW_5_DATA;
  } WAIT_REFERENCE_HI_UNION;

  union {
    struct {
      unsigned int wait_mask_31_0 : 32;
    };
    unsigned int DW_6_DATA;
  } WAIT_MASK_LO_UNION;

  union {
    struct {
      unsigned int wait_mask_63_32 : 32;
    };
    unsigned int DW_7_DATA;
  } WAIT_MASK_HI_UNION;

  union {
    struct {
      unsigned int count       : 30;
      unsigned int reserved_0  :  2;
    };
    unsigned int DW_8_DATA;
  } COUNT_UNION;

  union {
    struct {
      unsigned int reserved_0       : 18;
      unsigned int scope_b          :  2;
      unsigned int temporal_hint_b  :  3;
      unsigned int reserved_1       :  3;
      unsigned int scope_a          :  2;
      unsigned int temporal_hint_a  :  3;
      unsigned int reserved_2       :  1;
    };
    unsigned int DW_9_DATA;
  } COPY_PARAMETER_UNION;

  union {
    struct {
      unsigned int addr_a_31_0 : 32;
    };
    unsigned int DW_10_DATA;
  } ADDR_A_LO_UNION;

  union {
    struct {
      unsigned int addr_a_63_32 : 32;
    };
    unsigned int DW_11_DATA;
  } ADDR_A_HI_UNION;

  union {
    struct {
      unsigned int addr_b_31_0 : 32;
    };
    unsigned int DW_12_DATA;
  } ADDR_B_LO_UNION;

  union {
    struct {
      unsigned int addr_b_63_32 : 32;
    };
    unsigned int DW_13_DATA;
  } ADDR_B_HI_UNION;

  union {
    struct {
      unsigned int signal_operation    :  7;
      unsigned int reserved_0          : 11;
      unsigned int signal_scope        :  2;
      unsigned int signal_temporal_hint:  3;
      unsigned int reserved_1          :  9;
    };
    unsigned int DW_14_DATA;
  } SIGNAL_OPERATION_UNION;

  union {
    struct {
      unsigned int reserved_0        :  3;
      unsigned int signal_addr_31_3  : 29;
    };
    unsigned int DW_15_DATA;
  } SIGNAL_ADDR_LO_UNION;

  union {
    struct {
      unsigned int signal_addr_63_32 : 32;
    };
    unsigned int DW_16_DATA;
  } SIGNAL_ADDR_HI_UNION;

  union {
    struct {
      unsigned int signal_data_31_0 : 32;
    };
    unsigned int DW_17_DATA;
  } SIGNAL_DATA_LO_UNION;

  union {
    struct {
      unsigned int signal_data_63_32 : 32;
    };
    unsigned int DW_18_DATA;
  } SIGNAL_DATA_HI_UNION;

  static const size_t kMaxSize_ = 0x3fffffe0;
  static const size_t kAlignment_ = 32;
} SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX1250;

// clang-format on

}  // namespace amd
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_SDMA_REGISTERS_H_
