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

#ifndef __PM4_PKT_STRUCT_COMMON_H__
#define __PM4_PKT_STRUCT_COMMON_H__

#ifndef PM4_HEADER_DEFINED
#define PM4_HEADER_DEFINED
typedef union PM4_TYPE_3_HEADER
{
    struct
    {
        unsigned int predicate : 1; ///< predicated version of packet when set
        unsigned int shaderType: 1; ///< 0: Graphics, 1: Compute Shader
        unsigned int reserved1 : 6; ///< reserved
        unsigned int opcode    : 8; ///< IT opcode
        unsigned int count     : 14;///< number of DWORDs - 1 in the information body.
        unsigned int type      : 2; ///< packet identifier. It should be 3 for type 3 packets
    };
    unsigned int u32All;
} PM4_TYPE_3_HEADER;
#endif // PM4_HEADER_DEFINED

//--------------------DISPATCH_DIRECT--------------------


typedef struct _PM4_DISPATCH_DIRECT
{
    union
    {
        PM4_TYPE_3_HEADER   header;            ///header
        unsigned int        ordinal1;
    };

    unsigned int dim_x;


    unsigned int dim_y;


    unsigned int dim_z;


    unsigned int dispatch_initiator;


}  PM4DISPATCH_DIRECT, *PPM4DISPATCH_DIRECT;

//--------------------INDIRECT_BUFFER--------------------

enum INDIRECT_BUFFER_cache_policy_enum { cache_policy_indirect_buffer_LRU_0 = 0, cache_policy_indirect_buffer_STREAM_1 = 1, cache_policy_indirect_buffer_BYPASS_2 = 2 };


//--------------------EVENT_WRITE--------------------

enum EVENT_WRITE_event_index_enum { event_index_event_write_OTHER_0 = 0, event_index_event_write_ZPASS_DONE_1 = 1, event_index_event_write_SAMPLE_PIPELINESTAT_2 = 2, event_index_event_write_SAMPLE_STREAMOUTSTAT_3 = 3, event_index_event_write_CS_VS_PS_PARTIAL_FLUSH_4 = 4, event_index_event_write_RESERVED_EOP_5 = 5, event_index_event_write_RESERVED_EOS_6 = 6, event_index_event_write_CACHE_FLUSH_7 = 7 };

typedef struct _PM4_EVENT_WRITE
{
    union
    {
        PM4_TYPE_3_HEADER   header;            ///header
        unsigned int        ordinal1;
    };

    union
    {
        struct
        {
            unsigned int event_type:6;
            unsigned int reserved1:2;
            EVENT_WRITE_event_index_enum event_index:4;
            unsigned int reserved2:20;
        } bitfields2;
        unsigned int ordinal2;
    };

    union
    {
        struct
        {
            unsigned int reserved3:3;
            unsigned int address_lo:29;
        } bitfields3;
        unsigned int ordinal3;
    };

    union
    {
        struct
        {
            unsigned int address_hi:16;
            unsigned int reserved4:16;
        } bitfields4;
        unsigned int ordinal4;
    };

}  PM4EVENT_WRITE, *PPM4EVENT_WRITE;


//--------------------SET_SH_REG--------------------


typedef struct _PM4_SET_SH_REG
{
    union
    {
        PM4_TYPE_3_HEADER   header;            ///header
        unsigned int        ordinal1;
    };

    union
    {
        struct
        {
            unsigned int reg_offset:16;
            unsigned int reserved1:16;
        } bitfields2;
        unsigned int ordinal2;
    };

    unsigned int reg_data[1];    //1..N of these fields


}  PM4SET_SH_REG, *PPM4SET_SH_REG;


//--------------------ACQUIRE_MEM--------------------

enum ACQUIRE_MEM_engine_enum { engine_acquire_mem_PFP_0 = 0, engine_acquire_mem_ME_1 = 1 };


typedef struct _PM4_ACQUIRE_MEM
{
    union
    {
        PM4_TYPE_3_HEADER   header;            ///header
        unsigned int        ordinal1;
    };

    union
    {
        struct
        {
            unsigned int coher_cntl:31;
            ACQUIRE_MEM_engine_enum engine:1;
        } bitfields2;
        unsigned int ordinal2;
    };

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
            unsigned int coher_base_hi:25;
            unsigned int reserved2:7;
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

}  PM4ACQUIRE_MEM, *PPM4ACQUIRE_MEM;


//--------------------MEC_INDIRECT_BUFFER--------------------

typedef struct _PM4_MEC_INDIRECT_BUFFER
{
    union
    {
        PM4_TYPE_3_HEADER   header;            ///header
        unsigned int        ordinal1;
    };

    union
    {
        struct
        {
            unsigned int swap_function:2;
            unsigned int ib_base_lo:30;
        } bitfields2;
        unsigned int ordinal2;
    };

    union
    {
        struct
        {
            unsigned int ib_base_hi:16;
            unsigned int reserved1:16;
        } bitfields3;
        unsigned int ordinal3;
    };

    union
    {
        struct
        {
            unsigned int ib_size:20;
            unsigned int chain:1;
            unsigned int offload_polling:1;
            unsigned int volatile_setting:1;
            unsigned int valid:1;
            unsigned int vmid:4;
            INDIRECT_BUFFER_cache_policy_enum cache_policy:2;
            unsigned int reserved4:2;
        } bitfields4;
        unsigned int ordinal4;
    };

}  PM4MEC_INDIRECT_BUFFER, *PPM4MEC_INDIRECT_BUFFER;

//--------------------MEC_WAIT_REG_MEM--------------------

enum MEC_WAIT_REG_MEM_function_enum {
     function__mec_wait_reg_mem__always_pass = 0,
     function__mec_wait_reg_mem__less_than_ref_value = 1,
     function__mec_wait_reg_mem__less_than_equal_to_the_ref_value = 2,
     function__mec_wait_reg_mem__equal_to_the_reference_value = 3,
     function__mec_wait_reg_mem__not_equal_reference_value = 4,
     function__mec_wait_reg_mem__greater_than_or_equal_reference_value = 5,
     function__mec_wait_reg_mem__greater_than_reference_value = 6 };

enum MEC_WAIT_REG_MEM_mem_space_enum {
     mem_space__mec_wait_reg_mem__register_space = 0,
     mem_space__mec_wait_reg_mem__memory_space = 1 };

enum MEC_WAIT_REG_MEM_operation_enum {
     operation__mec_wait_reg_mem__wait_reg_mem = 0,
     operation__mec_wait_reg_mem__wr_wait_wr_reg = 1,
     operation__mec_wait_reg_mem__wait_mem_preemptable = 3 };


typedef struct PM4_MEC_WAIT_REG_MEM
{
    union
    {
        PM4_TYPE_3_HEADER   header;            ///header
        uint32_t            ordinal1;
    };

    union
    {
        struct
        {
            MEC_WAIT_REG_MEM_function_enum function:3;
            uint32_t reserved1:1;
            MEC_WAIT_REG_MEM_mem_space_enum mem_space:2;
            MEC_WAIT_REG_MEM_operation_enum operation:2;
            uint32_t reserved2:24;
        } bitfields2;
        uint32_t ordinal2;
    };

    union
    {
        struct
        {
            uint32_t reserved3:2;
            uint32_t mem_poll_addr_lo:30;
        } bitfields3a;
        struct
        {
            uint32_t reg_poll_addr:18;
            uint32_t reserved4:14;
        } bitfields3b;
        struct
        {
            uint32_t reg_write_addr1:18;
            uint32_t reserved5:14;
        } bitfields3c;
        uint32_t ordinal3;
    };

    union
    {
        uint32_t mem_poll_addr_hi;

        struct
        {
            uint32_t reg_write_addr2:18;
            uint32_t reserved6:14;
        } bitfields4b;
        uint32_t ordinal4;
    };

    uint32_t reference;

    uint32_t mask;

    union
    {
        struct
        {
            uint32_t poll_interval:16;
            uint32_t reserved7:15;
            uint32_t optimize_ace_offload_mode:1;
        } bitfields7;
        uint32_t ordinal7;
    };

} PM4MEC_WAIT_REG_MEM, *PPM4MEC_WAIT_REG_MEM;

//--------------------MEC_WRITE_DATA--------------------

enum MEC_WRITE_DATA_dst_sel_enum { dst_sel_mec_write_data_MEM_MAPPED_REGISTER_0 = 0, dst_sel_mec_write_data_TC_L2_2 = 2, dst_sel_mec_write_data_GDS_3 = 3, dst_sel_mec_write_data_MEMORY_5 = 5 };
enum MEC_WRITE_DATA_addr_incr_enum { addr_incr_mec_write_data_INCREMENT_ADDR_0 = 0, addr_incr_mec_write_data_DO_NOT_INCREMENT_ADDR_1 = 1 };
enum MEC_WRITE_DATA_wr_confirm_enum { wr_confirm_mec_write_data_DO_NOT_WAIT_FOR_CONFIRMATION_0 = 0, wr_confirm_mec_write_data_WAIT_FOR_CONFIRMATION_1 = 1 };
enum MEC_WRITE_DATA_cache_policy_enum { cache_policy_mec_write_data_LRU_0 = 0, cache_policy_mec_write_data_STREAM_1 = 1, cache_policy_mec_write_data_BYPASS_2 = 2 };

//--------------------MEC_RELEASE_MEM--------------------

enum MEC_RELEASE_MEM_event_index_enum { event_index_mec_release_mem_EVENT_WRITE_EOP_5 = 5, event_index_mec_release_mem_CS_Done_6 = 6 };
enum MEC_RELEASE_MEM_cache_policy_enum { cache_policy_mec_release_mem_LRU_0 = 0, cache_policy_mec_release_mem_STREAM_1 = 1, cache_policy_mec_release_mem_BYPASS_2 = 2 };
enum MEC_RELEASE_MEM_dst_sel_enum { dst_sel_mec_release_mem_MEMORY_CONTROLLER_0 = 0, dst_sel_mec_release_mem_TC_L2_1 = 1 };
enum MEC_RELEASE_MEM_int_sel_enum { int_sel_mec_release_mem_NONE_0 = 0, int_sel_mec_release_mem_SEND_INTERRUPT_ONLY_1 = 1, int_sel_mec_release_mem_SEND_INTERRUPT_AFTER_WRITE_CONFIRM_2 = 2, int_sel_mec_release_mem_SEND_DATA_AFTER_WRITE_CONFIRM_3 = 3 };
enum MEC_RELEASE_MEM_data_sel_enum { data_sel_mec_release_mem_NONE_0 = 0, data_sel_mec_release_mem_SEND_32_BIT_LOW_1 = 1, data_sel_mec_release_mem_SEND_64_BIT_DATA_2 = 2, data_sel_mec_release_mem_SEND_GPU_CLOCK_COUNTER_3 = 3, data_sel_mec_release_mem_SEND_CP_PERFCOUNTER_HI_LO_4 = 4, data_sel_mec_release_mem_STORE_GDS_DATA_TO_MEMORY_5 = 5 };


#endif

