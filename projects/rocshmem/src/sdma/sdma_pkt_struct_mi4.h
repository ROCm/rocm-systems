/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * OSS7.0 SDMA packet structures (CDNA4 / MI350X and later).
 *
 * Struct definitions derived from the field macros in sdma-packet.h
 * (auto-generated from OSS_70-sDMA_MAS.md).  Each struct mirrors
 * the coding style in sdma_pkt_struct.h: one tagged typedef per
 * packet, union-per-DWORD with named bitfields + raw DW_N_DATA
 * accessor, and a static_assert on the total size.
 *
 * These types are always defined; use SDMA_IS_OSS7 (from sdma_opcodes.h)
 * to guard code paths that emit or decode OSS7 packets.
 */

#pragma once

#include "sdma_opcodes.h"

/* ---------------------------------------------------------------
 * SDMA_PKT_COPY_LINEAR_PHY_MI4   (OP=0x1  SUB_OP=0x8)
 * Physical Linear Copy -- 8 DWORDs
 * --------------------------------------------------------------- */
typedef struct SDMA_PKT_COPY_LINEAR_PHY_MI4_TAG {
  union {
    struct {
      unsigned int op_code : 8;
      unsigned int sub_op_code : 8;
      unsigned int reserved_0 : 2;
      unsigned int tmz : 1;
      unsigned int reserved_1 : 9;
      unsigned int nsd : 1;
      unsigned int reserved_2 : 3;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int count : 22;
      unsigned int reserved_0 : 2;
      unsigned int addr_pair : 8;
    };
    unsigned int DW_1_DATA;
  } COUNT_UNION;

  union {
    struct {
      unsigned int reserved_0 : 4;
      unsigned int dst_mtype : 2;
      unsigned int dst_scope : 2;
      unsigned int dst_temporal_hint : 3;
      unsigned int reserved_1 : 1;
      unsigned int src_mtype : 2;
      unsigned int src_scope : 2;
      unsigned int src_temporal_hint : 3;
      unsigned int gcc : 1;
      unsigned int dst_sys : 1;
      unsigned int reserved_2 : 1;
      unsigned int dst_snp : 1;
      unsigned int dst_gpa : 1;
      unsigned int reserved_3 : 4;
      unsigned int src_sys : 1;
      unsigned int reserved_4 : 1;
      unsigned int src_snp : 1;
      unsigned int src_gpa : 1;
    };
    unsigned int DW_2_DATA;
  } PARAMETER_UNION;

  union {
    struct {
      unsigned int data_format : 6;
      unsigned int reserved_0 : 3;
      unsigned int num_type : 3;
      unsigned int reserved_1 : 4;
      unsigned int rd_cm : 2;
      unsigned int wr_cm : 2;
      unsigned int reserved_2 : 4;
      unsigned int max_com : 2;
      unsigned int max_ucom : 1;
      unsigned int reserved_3 : 4;
      unsigned int dcc : 1;
    };
    unsigned int DW_3_DATA;
  } DW3_UNION;

  union {
    struct {
      unsigned int src_address_lo : 32;
    };
    unsigned int DW_4_DATA;
  } SRC_ADDR_LO_UNION;

  union {
    struct {
      unsigned int src_address_hi : 32;
    };
    unsigned int DW_5_DATA;
  } SRC_ADDR_HI_UNION;

  union {
    struct {
      unsigned int dst_address_lo : 32;
    };
    unsigned int DW_6_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_address_hi : 32;
    };
    unsigned int DW_7_DATA;
  } DST_ADDR_HI_UNION;
} SDMA_PKT_COPY_LINEAR_PHY_MI4;
static_assert(sizeof(SDMA_PKT_COPY_LINEAR_PHY_MI4) == 8 * sizeof(unsigned int),
              "SDMA_PKT_COPY_LINEAR_PHY_MI4 must be 8 DWORDs");

/* ---------------------------------------------------------------
 * SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4   (OP=0x1  SUB_OP=0x0)
 * Linear Copy with Wait / Signal -- 19 DWORDs
 *
 * Combines copy + poll-wait + atomic-signal into a single packet,
 * replacing the MI300X two-packet COPY_LINEAR + ATOMIC chain.
 * --------------------------------------------------------------- */
typedef struct SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_TAG {
  /* DW 0 */
  union {
    struct {
      unsigned int op : 8;
      unsigned int subop : 8;
      unsigned int reserved_0 : 2;
      unsigned int tmz : 1;
      unsigned int reserved_1 : 9;
      unsigned int npd : 1;
      unsigned int reserved_2 : 1;
      unsigned int wait : 1;
      unsigned int signal : 1;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  /* DW 1 */
  union {
    struct {
      unsigned int wait_function : 3;
      unsigned int reserved_0 : 15;
      unsigned int wait_scope : 2;
      unsigned int wait_temporal_hint : 3;
      unsigned int reserved_1 : 9;
    };
    unsigned int DW_1_DATA;
  } WAIT_CTRL_UNION;

  /* DW 2 -- bits [2:0] reserved (8-byte aligned addr) */
  union {
    struct {
      unsigned int reserved_0 : 3;
      unsigned int wait_addr_31_3 : 29;
    };
    unsigned int DW_2_DATA;
  } WAIT_ADDR_LO_UNION;

  /* DW 3 */
  union {
    struct {
      unsigned int wait_addr_63_32 : 32;
    };
    unsigned int DW_3_DATA;
  } WAIT_ADDR_HI_UNION;

  /* DW 4 */
  union {
    struct {
      unsigned int wait_reference_31_0 : 32;
    };
    unsigned int DW_4_DATA;
  } WAIT_REF_LO_UNION;

  /* DW 5 */
  union {
    struct {
      unsigned int wait_reference_63_32 : 32;
    };
    unsigned int DW_5_DATA;
  } WAIT_REF_HI_UNION;

  /* DW 6 */
  union {
    struct {
      unsigned int wait_mask_31_0 : 32;
    };
    unsigned int DW_6_DATA;
  } WAIT_MASK_LO_UNION;

  /* DW 7 */
  union {
    struct {
      unsigned int wait_mask_63_32 : 32;
    };
    unsigned int DW_7_DATA;
  } WAIT_MASK_HI_UNION;

  /* DW 8 */
  union {
    struct {
      unsigned int copy_count : 30;
      unsigned int reserved_0 : 2;
    };
    unsigned int DW_8_DATA;
  } COPY_COUNT_UNION;

  /* DW 9 */
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
    unsigned int DW_9_DATA;
  } COPY_PARAM_UNION;

  /* DW 10 */
  union {
    struct {
      unsigned int src_addr_31_0 : 32;
    };
    unsigned int DW_10_DATA;
  } SRC_ADDR_LO_UNION;

  /* DW 11 */
  union {
    struct {
      unsigned int src_addr_63_32 : 32;
    };
    unsigned int DW_11_DATA;
  } SRC_ADDR_HI_UNION;

  /* DW 12 */
  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_12_DATA;
  } DST_ADDR_LO_UNION;

  /* DW 13 */
  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_13_DATA;
  } DST_ADDR_HI_UNION;

  /* DW 14 */
  union {
    struct {
      unsigned int signal_operation : 7;
      unsigned int reserved_0 : 11;
      unsigned int signal_scope : 2;
      unsigned int signal_temporal_hint : 3;
      unsigned int reserved_1 : 9;
    };
    unsigned int DW_14_DATA;
  } SIGNAL_CTRL_UNION;

  /* DW 15 -- bits [2:0] reserved (8-byte aligned addr) */
  union {
    struct {
      unsigned int reserved_0 : 3;
      unsigned int signal_addr_31_3 : 29;
    };
    unsigned int DW_15_DATA;
  } SIGNAL_ADDR_LO_UNION;

  /* DW 16 */
  union {
    struct {
      unsigned int signal_addr_63_32 : 32;
    };
    unsigned int DW_16_DATA;
  } SIGNAL_ADDR_HI_UNION;

  /* DW 17 */
  union {
    struct {
      unsigned int signal_data_31_0 : 32;
    };
    unsigned int DW_17_DATA;
  } SIGNAL_DATA_LO_UNION;

  /* DW 18 */
  union {
    struct {
      unsigned int signal_data_63_32 : 32;
    };
    unsigned int DW_18_DATA;
  } SIGNAL_DATA_HI_UNION;
} SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4;
static_assert(sizeof(SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4) == 19 * sizeof(unsigned int),
              "SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4 must be 19 DWORDs");

/* ---------------------------------------------------------------
 * SDMA_PKT_FENCE_MI4   (OP=0x5  SUB_OP=0x0)
 * Fence -- 4 DWORDs
 * --------------------------------------------------------------- */
typedef struct SDMA_PKT_FENCE_MI4_TAG {
  union {
    struct {
      unsigned int op_code : 8;
      unsigned int sub_op_code : 8;
      unsigned int mtype : 2;
      unsigned int reserved_0 : 2;
      unsigned int s : 1;
      unsigned int reserved_1 : 1;
      unsigned int snp : 1;
      unsigned int gpa : 1;
      unsigned int scope : 2;
      unsigned int temporal_hint : 3;
      unsigned int reserved_2 : 3;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int fence_addr_lo : 32;
    };
    unsigned int DW_1_DATA;
  } ADDR_LO_UNION;

  union {
    struct {
      unsigned int fence_addr_hi : 32;
    };
    unsigned int DW_2_DATA;
  } ADDR_HI_UNION;

  union {
    struct {
      unsigned int data : 32;
    };
    unsigned int DW_3_DATA;
  } DATA_UNION;
} SDMA_PKT_FENCE_MI4;
static_assert(sizeof(SDMA_PKT_FENCE_MI4) == 4 * sizeof(unsigned int),
              "SDMA_PKT_FENCE_MI4 must be 4 DWORDs");

/* ---------------------------------------------------------------
 * SDMA_PKT_FENCE_64B_MI4   (OP=0x5  SUB_OP=0x2)
 * 64-bit Fence -- 5 DWORDs
 *
 * Addresses are 8-byte aligned; bits [2:0] of addr_lo are reserved.
 * --------------------------------------------------------------- */
typedef struct SDMA_PKT_FENCE_64B_MI4_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int subop : 8;
      unsigned int mtype : 2;
      unsigned int reserved_0 : 2;
      unsigned int sys : 1;
      unsigned int reserved_1 : 1;
      unsigned int snp : 1;
      unsigned int gpa : 1;
      unsigned int scope : 2;
      unsigned int temporal_hint : 3;
      unsigned int reserved_2 : 3;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int reserved_0 : 3;
      unsigned int addr_31_3 : 29;
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
} SDMA_PKT_FENCE_64B_MI4;
static_assert(sizeof(SDMA_PKT_FENCE_64B_MI4) == 5 * sizeof(unsigned int),
              "SDMA_PKT_FENCE_64B_MI4 must be 5 DWORDs");
