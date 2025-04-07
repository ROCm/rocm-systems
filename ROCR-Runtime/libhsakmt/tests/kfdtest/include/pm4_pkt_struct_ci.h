/*
 * Copyright (C) 2012-2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef __PM4_PKT_STRUCT_CI_H__
#define __PM4_PKT_STRUCT_CI_H__


enum WRITE_DATA_CI_atc_enum { atc_write_data_NOT_USE_ATC_0 = 0, atc_write_data_USE_ATC_1 = 1 };
enum WRITE_DATA_CI_engine_sel { engine_sel_write_data_ci_MICRO_ENGINE_0 = 0, engine_sel_write_data_ci_PREFETCH_PARSER_1 = 1, engine_sel_write_data_ci_CONST_ENG_2 = 2 };

typedef struct _PM4WRITE_DATA_CI {
    union {
        PM4_TYPE_3_HEADER   header;
        unsigned int        ordinal1;
    };

    union {
        struct {
            unsigned int reserved1:8;
            MEC_WRITE_DATA_dst_sel_enum dst_sel:4;
            unsigned int reserved2:4;
            MEC_WRITE_DATA_addr_incr_enum addr_incr:1;
            unsigned int reserved3:3;
            MEC_WRITE_DATA_wr_confirm_enum wr_confirm:1;
            unsigned int reserved4:3;
            WRITE_DATA_CI_atc_enum atc:1;
            MEC_WRITE_DATA_cache_policy_enum cache_policy:2;
            unsigned int volatile_setting:1;
            unsigned int reserved5:2;
            WRITE_DATA_CI_engine_sel engine_sel:2;
        } bitfields2;
        unsigned int ordinal2;
    };

    unsigned int dst_addr_lo;

    unsigned int dst_address_hi;

    unsigned int data[1];    // 1..N of these fields
}  PM4WRITE_DATA_CI, *PPM4WRITE_DATA_CI;


enum MEC_RELEASE_MEM_CI_atc_enum { atc_mec_release_mem_ci_NOT_USE_ATC_0 = 0, atc_mec_release_mem_ci_USE_ATC_1 = 1 };

typedef struct _PM4_RELEASE_MEM_CI {
    union {
        PM4_TYPE_3_HEADER   header;
        unsigned int        ordinal1;
    };

    union {
        struct {
            unsigned int event_type:6;
            unsigned int reserved1:2;
            MEC_RELEASE_MEM_event_index_enum event_index:4;
            unsigned int l1_vol:1;
            unsigned int l2_vol:1;
            unsigned int reserved:1;
            unsigned int l2_wb:1;
            unsigned int l1_inv:1;
            unsigned int l2_inv:1;
            unsigned int reserved2:6;
            MEC_RELEASE_MEM_CI_atc_enum atc:1;
            MEC_RELEASE_MEM_cache_policy_enum cache_policy:2;
            unsigned int volatile_setting:1;
            unsigned int reserved3:4;
        } bitfields2;
        unsigned int ordinal2;
    };

    union {
        struct {
            unsigned int reserved4:16;
            MEC_RELEASE_MEM_dst_sel_enum dst_sel:2;
            unsigned int reserved5:6;
            MEC_RELEASE_MEM_int_sel_enum int_sel:3;
            unsigned int reserved6:2;
            MEC_RELEASE_MEM_data_sel_enum data_sel:3;
        } bitfields3;
        unsigned int ordinal3;
    };

    union {
        struct {
            unsigned int reserved7:2;
            unsigned int address_lo_dword_aligned:30;
        } bitfields4a;
        struct {
            unsigned int reserved8:3;
            unsigned int address_lo_qword_aligned:29;
        } bitfields4b;
        unsigned int ordinal4;
    };

    unsigned int addr_hi;

    union {
        unsigned int data_lo;
        struct {
            unsigned int offset:16;
            unsigned int num_dwords:16;
        } bitfields5b;
        unsigned int ordinal6;
    };

    unsigned int data_hi;
}  PM4_RELEASE_MEM_CI, *PPM4_RELEASE_MEM_CI;

#endif  // __PM4_PKT_STRUCT_CI_H__
