/*
 * Copyright (C) 2014-2018 Advanced Micro Devices, Inc. All Rights Reserved.
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

#ifndef __SDMA_PKT_STRUCT_H__
#define __SDMA_PKT_STRUCT_H__


const unsigned int SDMA_OP_NOP = 0;
const unsigned int SDMA_OP_COPY = 1;
const unsigned int SDMA_OP_WRITE = 2;

const unsigned int SDMA_OP_FENCE = 5;
const unsigned int SDMA_OP_TRAP = 6;
const unsigned int SDMA_OP_POLL_REGMEM = 8;
const unsigned int SDMA_OP_TIMESTAMP = 13;

const unsigned int SDMA_OP_CONST_FILL = 11;

const unsigned int SDMA_SUBOP_COPY_LINEAR = 0;

const unsigned int SDMA_SUBOP_WRITE_LINEAR = 0;

/*
** Definitions for SDMA_PKT_COPY_LINEAR packet
*/

typedef struct SDMA_PKT_COPY_LINEAR_TAG
{

    union
    {
        struct
        {
            unsigned int op:8;
            unsigned int sub_op:8;
            unsigned int reserved_0:11;
            unsigned int broadcast:1;
            unsigned int reserved_1:4;
        };
        unsigned int DW_0_DATA;
    } HEADER_UNION;

    union
    {
        struct
        {
            unsigned int count:22;
            unsigned int reserved_0:10;
        };
        unsigned int DW_1_DATA;
    } COUNT_UNION;

    union
    {
        struct
        {
            unsigned int reserved_0:16;
            unsigned int dst_sw:2;
            unsigned int reserved_1:4;
            unsigned int dst_ha:1;
            unsigned int reserved_2:1;
            unsigned int src_sw:2;
            unsigned int reserved_3:4;
            unsigned int src_ha:1;
            unsigned int reserved_4:1;
        };
        unsigned int DW_2_DATA;
    } PARAMETER_UNION;

    union
    {
        struct
        {
            unsigned int src_addr_31_0:32;
        };
        unsigned int DW_3_DATA;
    } SRC_ADDR_LO_UNION;

    union
    {
        struct
        {
            unsigned int src_addr_63_32:32;
        };
        unsigned int DW_4_DATA;
    } SRC_ADDR_HI_UNION;

    struct
    {
        union
        {
            struct
            {
                unsigned int dst_addr_31_0:32;
            };
            unsigned int DW_5_DATA;
        } DST_ADDR_LO_UNION;

        union
        {
            struct
            {
                unsigned int dst_addr_63_32:32;
            };
            unsigned int DW_6_DATA;
        } DST_ADDR_HI_UNION;
    } DST_ADDR[0];
} SDMA_PKT_COPY_LINEAR, *PSDMA_PKT_COPY_LINEAR;

/*
** Definitions for SDMA_PKT_WRITE_UNTILED packet
*/

typedef struct SDMA_PKT_WRITE_UNTILED_TAG
{

    union
    {
        struct
        {
            unsigned int op:8;
            unsigned int sub_op:8;
            unsigned int reserved_0:16;
        };
        unsigned int DW_0_DATA;
    } HEADER_UNION;

    union
    {
        struct
        {
            unsigned int dst_addr_31_0:32;
        };
        unsigned int DW_1_DATA;
    } DST_ADDR_LO_UNION;

    union
    {
        struct
        {
            unsigned int dst_addr_63_32:32;
        };
        unsigned int DW_2_DATA;
    } DST_ADDR_HI_UNION;

    union
    {
        struct
        {
            unsigned int count:22;
            unsigned int reserved_0:2;
            unsigned int sw:2;
            unsigned int reserved_1:6;
        };
        unsigned int DW_3_DATA;
    } DW_3_UNION;

    union
    {
        struct
        {
            unsigned int data0:32;
        };
        unsigned int DW_4_DATA;
    } DATA0_UNION;
} SDMA_PKT_WRITE_UNTILED, *PSDMA_PKT_WRITE_UNTILED;

/*
** Definitions for SDMA_PKT_FENCE packet
*/

typedef struct SDMA_PKT_FENCE_TAG
{

    union
    {
        struct
        {
            unsigned int op:8;
            unsigned int sub_op:8;
            unsigned int reserved_0:16;
        };
        unsigned int DW_0_DATA;
    } HEADER_UNION;

    union
    {
        struct
        {
            unsigned int addr_31_0:32;
        };
        unsigned int DW_1_DATA;
    } ADDR_LO_UNION;

    union
    {
        struct
        {
            unsigned int addr_63_32:32;
        };
        unsigned int DW_2_DATA;
    } ADDR_HI_UNION;

    union
    {
        struct
        {
            unsigned int data:32;
        };
        unsigned int DW_3_DATA;
    } DATA_UNION;
} SDMA_PKT_FENCE, *PSDMA_PKT_FENCE;

/*
** Definitions for SDMA_PKT_CONSTANT_FILL packet
*/

typedef struct SDMA_PKT_CONSTANT_FILL_TAG
{

    union
    {
        struct
        {
            unsigned int op:8;
            unsigned int sub_op:8;
            unsigned int sw:2;
            unsigned int reserved_0:12;
            unsigned int fillsize:2;
        };
        unsigned int DW_0_DATA;
    } HEADER_UNION;

    union
    {
        struct
        {
            unsigned int dst_addr_31_0:32;
        };
        unsigned int DW_1_DATA;
    } DST_ADDR_LO_UNION;

    union
    {
        struct
        {
            unsigned int dst_addr_63_32:32;
        };
        unsigned int DW_2_DATA;
    } DST_ADDR_HI_UNION;

    union
    {
        struct
        {
            unsigned int src_data_31_0:32;
        };
        unsigned int DW_3_DATA;
    } DATA_UNION;

    union
    {
        struct
        {
            unsigned int count:22;
            unsigned int reserved_0:10;
        };
        unsigned int DW_4_DATA;
    } COUNT_UNION;
} SDMA_PKT_CONSTANT_FILL, *PSDMA_PKT_CONSTANT_FILL;

/*
** Definitions for SDMA_PKT_TRAP packet
*/

typedef struct SDMA_PKT_TRAP_TAG
{

    union
    {
        struct
        {
            unsigned int op:8;
            unsigned int sub_op:8;
            unsigned int reserved_0:16;
        };
        unsigned int DW_0_DATA;
    } HEADER_UNION;

    union
    {
        struct
        {
            unsigned int int_context:28;
            unsigned int reserved_0:4;
        };
        unsigned int DW_1_DATA;
    } INT_CONTEXT_UNION;
} SDMA_PKT_TRAP, *PSDMA_PKT_TRAP;

/*
** Definitions for SDMA_PKT_POLL_REGMEM_TAG packet
*/

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

/*
** Definitions for SDMA_PKT_TIMESTAMP packet
*/

typedef struct SDMA_PKT_TIMESTAMP_TAG
{

    union
    {
        struct
        {
            unsigned int op:8;
            unsigned int sub_op:8;
            unsigned int reserved_0:16;
        };
        unsigned int DW_0_DATA;
    } HEADER_UNION;

    union
    {
        struct
        {
            unsigned int addr_31_0:32;
        };
        unsigned int DW_1_DATA;
    } ADDR_LO_UNION;

    union
    {
        struct
        {
            unsigned int addr_63_32:32;
        };
        unsigned int DW_2_DATA;
    } ADDR_HI_UNION;
} SDMA_PKT_TIMESTAMP, *PSDMA_PKT_TIMESTAMP;


/*
** Definitions for SDMA_PKT_NOP packet
*/

typedef struct SDMA_PKT_NOP_TAG
{
    union
    {
        struct
        {
            unsigned int op:8;
            unsigned int sub_op:8;
            unsigned int count:14;
            unsigned int reserved_0:2;
        };
        unsigned int DW_0_DATA;
    } HEADER_UNION;

    union
    {
        struct
        {
            unsigned int data0:32;
        };
        unsigned int DW_1_DATA;
    } DATA0_UNION;
} SDMA_PKT_NOP, *PSDMA_PKT_NOP;

#endif // __SDMA_PKT_STRUCT_H__
