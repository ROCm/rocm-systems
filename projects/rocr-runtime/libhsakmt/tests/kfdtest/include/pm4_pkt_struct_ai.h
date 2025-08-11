/*
 * Copyright (C) 2016-2018 Advanced Micro Devices, Inc. All Rights Reserved.
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

#ifndef __PM4_PKT_STRUCT_AI_H__
#define __PM4_PKT_STRUCT_AI_H__

#ifndef PM4_MEC_RELEASE_MEM_AI_DEFINED
#define PM4_MEC_RELEASE_MEM_AI_DEFINED

enum AI_MEC_RELEASE_MEM_event_index_enum {
     event_index__mec_release_mem__end_of_pipe = 5,
     event_index__mec_release_mem__shader_done = 6 };

enum AI_MEC_RELEASE_MEM_cache_policy_enum {
     cache_policy__mec_release_mem__lru = 0,
     cache_policy__mec_release_mem__stream = 1 };

enum AI_MEC_RELEASE_MEM_pq_exe_status_enum {
     pq_exe_status__mec_release_mem__default = 0,
     pq_exe_status__mec_release_mem__phase_update = 1 };

enum AI_MEC_RELEASE_MEM_dst_sel_enum {
     dst_sel__mec_release_mem__memory_controller = 0,
     dst_sel__mec_release_mem__tc_l2 = 1,
     dst_sel__mec_release_mem__queue_write_pointer_register = 2,
     dst_sel__mec_release_mem__queue_write_pointer_poll_mask_bit = 3 };

enum AI_MEC_RELEASE_MEM_int_sel_enum {
     int_sel__mec_release_mem__none = 0,
     int_sel__mec_release_mem__send_interrupt_only = 1,
     int_sel__mec_release_mem__send_interrupt_after_write_confirm = 2,
     int_sel__mec_release_mem__send_data_after_write_confirm = 3,
     int_sel__mec_release_mem__unconditionally_send_int_ctxid = 4,
     int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_32_bit_compare = 5,
     int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_64_bit_compare = 6 };

enum AI_MEC_RELEASE_MEM_data_sel_enum {
     data_sel__mec_release_mem__none = 0,
     data_sel__mec_release_mem__send_32_bit_low = 1,
     data_sel__mec_release_mem__send_64_bit_data = 2,
     data_sel__mec_release_mem__send_gpu_clock_counter = 3,
     data_sel__mec_release_mem__send_cp_perfcounter_hi_lo = 4,
     data_sel__mec_release_mem__store_gds_data_to_memory = 5 };


typedef struct PM4_MEC_RELEASE_MEM_AI {
    union {
        PM4_TYPE_3_HEADER   header;
        unsigned int        ordinal1;
    };

    union {
        struct {
            unsigned int event_type:6;
            unsigned int reserved1:2;
            AI_MEC_RELEASE_MEM_event_index_enum event_index:4;
            unsigned int tcl1_vol_action_ena:1;
            unsigned int tc_vol_action_ena:1;
            unsigned int reserved2:1;
            unsigned int tc_wb_action_ena:1;
            unsigned int tcl1_action_ena:1;
            unsigned int tc_action_ena:1;
            unsigned int reserved3:1;
            unsigned int tc_nc_action_ena:1;
            unsigned int tc_wc_action_ena:1;
            unsigned int tc_md_action_ena:1;
            unsigned int reserved4:3;
            AI_MEC_RELEASE_MEM_cache_policy_enum cache_policy:2;
            unsigned int reserved5:2;
            AI_MEC_RELEASE_MEM_pq_exe_status_enum pq_exe_status:1;
            unsigned int reserved6:2;
        } bitfields2;
        unsigned int ordinal2;
    };

    union {
        struct {
            unsigned int reserved7:16;
            AI_MEC_RELEASE_MEM_dst_sel_enum dst_sel:2;
            unsigned int reserved8:6;
            AI_MEC_RELEASE_MEM_int_sel_enum int_sel:3;
            unsigned int reserved9:2;
            AI_MEC_RELEASE_MEM_data_sel_enum data_sel:3;
        } bitfields3;
        unsigned int ordinal3;
    };

    union {
        struct {
            unsigned int reserved10:2;
            unsigned int address_lo_32b:30;
        } bitfields4a;
        struct {
            unsigned int reserved11:3;
            unsigned int address_lo_64b:29;
        } bitfields4b;
        unsigned int reserved12;

        unsigned int ordinal4;
    };

    union {
        unsigned int address_hi;

        unsigned int reserved13;

        unsigned int ordinal5;
    };

    union {
        unsigned int data_lo;

        unsigned int cmp_data_lo;

        struct {
            unsigned int dw_offset:16;
            unsigned int num_dwords:16;
        } bitfields6c;
        unsigned int reserved14;

        unsigned int ordinal6;
    };

    union {
        unsigned int data_hi;

        unsigned int cmp_data_hi;

        unsigned int reserved15;

        unsigned int reserved16;

        unsigned int ordinal7;
    };

    unsigned int int_ctxid;
} PM4MEC_RELEASE_MEM_AI, *PPM4MEC_RELEASE_MEM_AI;

#endif  // PM4_MEC_RELEASE_MEM_AI_DEFINED
#endif  // __PM4_PKT_STRUCT_AI_H__
