/*
 * Copyright 2018 Advanced Micro Devices, Inc.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef __PM4__PKT__STRUCT__NV__HPP__
#define __PM4__PKT__STRUCT__NV__HPP__

#include "pm4_pkt_struct_ai.h"

typedef struct _PM4_ACQUIRE_MEM_NV
{
    union
    {
        PM4_TYPE_3_HEADER   header;            ///header
        unsigned int        ordinal1;
    };

    unsigned int reserved;

    unsigned int coher_size;


    union
    {
        struct
        {
            unsigned int coher_size_hi:8;
            unsigned int reserved1:24;
        } bitfields3;
        unsigned int ordinal4;
    };

    unsigned int coher_base_lo;


    union
    {
        struct
        {
            unsigned int coher_base_hi:24;
            unsigned int reserved2:8;
        } bitfields4;
        unsigned int ordinal6;
    };

    union
    {
        struct
        {
            unsigned int poll_interval:16;
            unsigned int reserved3:16;
        } bitfields5;
        unsigned int ordinal7;
    };

    union
    {
        struct
        {
            unsigned int gcr_cntl:18;
            unsigned int reserved4:14;
        } bitfields6;
        unsigned int ordinal8;
    };


}  PM4ACQUIRE_MEM_NV, *PPM4ACQUIRE_MEM_NV;

typedef struct PM4_MEC_RELEASE_MEM_NV {
    union {
        PM4_TYPE_3_HEADER   header;
        unsigned int        ordinal1;
    };

    union {
        struct {
            unsigned int event_type:6;
            unsigned int reserved1:2;
            AI_MEC_RELEASE_MEM_event_index_enum event_index:4;
            unsigned int gcr_cntl:12;
            unsigned int reserved4:1;
            AI_MEC_RELEASE_MEM_cache_policy_enum cache_policy:2;
            unsigned int reserved5:1;
            AI_MEC_RELEASE_MEM_pq_exe_status_enum pq_exe_status:1;
            unsigned int reserved6:3;
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
} PM4MEC_RELEASE_MEM_NV, *PPM4MEC_RELEASE_MEM_NV;


#endif // __PM4__PKT__STRUCT__NV__HPP__
