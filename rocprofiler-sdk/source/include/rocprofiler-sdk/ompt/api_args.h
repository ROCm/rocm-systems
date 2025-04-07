// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/defines.h>

#include <rocprofiler-sdk/ompt/omp-tools.h>

#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

// all the available callback interface runtime entry points
typedef struct rocprofiler_ompt_callback_functions_t
{
    ompt_enumerate_states_t         ompt_enumerate_states;
    ompt_enumerate_mutex_impls_t    ompt_enumerate_mutex_impls;
    ompt_get_thread_data_t          ompt_get_thread_data;
    ompt_get_num_places_t           ompt_get_num_places;
    ompt_get_place_proc_ids_t       ompt_get_place_proc_ids;
    ompt_get_place_num_t            ompt_get_place_num;
    ompt_get_partition_place_nums_t ompt_get_partition_place_nums;
    ompt_get_proc_id_t              ompt_get_proc_id;
    ompt_get_state_t                ompt_get_state;
    ompt_get_parallel_info_t        ompt_get_parallel_info;
    ompt_get_task_info_t            ompt_get_task_info;
    ompt_get_task_memory_t          ompt_get_task_memory;
    ompt_get_num_devices_t          ompt_get_num_devices;
    ompt_get_num_procs_t            ompt_get_num_procs;
    ompt_get_target_info_t          ompt_get_target_info;
    ompt_get_unique_id_t            ompt_get_unique_id;
} rocprofiler_ompt_callback_functions_t;

// Empty struct has a size of 0 in C but size of 1 in C++.
// This struct is added to the union members which represent
// functions with no arguments to ensure ABI compatibility
typedef struct rocprofiler_ompt_no_args
{
    char empty;
} rocprofiler_ompt_no_args;

typedef union rocprofiler_ompt_args_t
{
    // The ompt_data_t* values passed to the client tool are proxies.
    // This allows the client tool to use them as it would in their own
    // OMPT tool.
    // We keepa a map from the address of the ompt_data_t passed to the SDK's
    // callback to the proxy object and keep it in sync when a callback is done
    // to the client tool.
    struct
    {
        ompt_thread_t thread_type;
        ompt_data_t*  thread_data;
    } thread_begin;

    struct
    {
        ompt_data_t* thread_data;
    } thread_end;

    struct
    {
        ompt_data_t*        encountering_task_data;
        const ompt_frame_t* encountering_task_frame;
        ompt_data_t*        parallel_data;
        unsigned int        requested_parallelism;
        int                 flags;
        const void*         codeptr_ra;
    } parallel_begin;

    struct
    {
        ompt_data_t* parallel_data;
        ompt_data_t* encountering_task_data;
        int          flags;
        const void*  codeptr_ra;
    } parallel_end;

    struct
    {
        ompt_data_t*        encountering_task_data;
        const ompt_frame_t* encountering_task_frame;
        ompt_data_t*        new_task_data;
        int                 flags;
        int                 has_dependences;
        const void*         codeptr_ra;
    } task_create;

    struct
    {
        ompt_data_t*       prior_task_data;
        ompt_task_status_t prior_task_status;
        ompt_data_t*       next_task_data;
    } task_schedule;

    struct
    {
        ompt_scope_endpoint_t endpoint;
        ompt_data_t*          parallel_data;
        ompt_data_t*          task_data;
        unsigned int          actual_parallelism;
        unsigned int          index;
        int                   flags;
    } implicit_task;

    struct
    {
        int                    device_num;
        const char*            type;
        ompt_device_t*         device;
        ompt_function_lookup_t lookup;
        const char*            documentation;
    } device_initialize;

    struct
    {
        int device_num;
    } device_finalize;

    struct
    {
        int         device_num;
        const char* filename;
        int64_t     offset_in_file;
        void*       vma_in_file;
        size_t      bytes;
        void*       host_addr;
        void*       device_addr;
        uint64_t    module_id;
    } device_load;

    // struct
    // {
    //     int      device_num;
    //     uint64_t module_id;
    // } device_unload;

    struct
    {
        ompt_sync_region_t    kind;
        ompt_scope_endpoint_t endpoint;
        ompt_data_t*          parallel_data;
        ompt_data_t*          task_data;
        const void*           codeptr_ra;
    } sync_region_wait;

    struct
    {
        ompt_mutex_t   kind;
        ompt_wait_id_t wait_id;
        const void*    codeptr_ra;
    } mutex_released;

    struct
    {
        ompt_data_t*             task_data;
        const ompt_dependence_t* deps;
        int                      ndeps;
    } dependences;

    struct
    {
        ompt_data_t* src_task_data;
        ompt_data_t* sink_task_data;
    } task_dependence;

    struct
    {
        ompt_work_t           work_type;
        ompt_scope_endpoint_t endpoint;
        ompt_data_t*          parallel_data;
        ompt_data_t*          task_data;
        uint64_t              count;
        const void*           codeptr_ra;
    } work;

    struct
    {
        ompt_scope_endpoint_t endpoint;
        ompt_data_t*          parallel_data;
        ompt_data_t*          task_data;
        const void*           codeptr_ra;
    } masked;

    struct
    {
        ompt_sync_region_t    kind;
        ompt_scope_endpoint_t endpoint;
        ompt_data_t*          parallel_data;
        ompt_data_t*          task_data;
        const void*           codeptr_ra;
    } sync_region;

    struct
    {
        ompt_mutex_t   kind;
        unsigned int   hint;
        unsigned int   impl;
        ompt_wait_id_t wait_id;
        const void*    codeptr_ra;
    } lock_init;

    struct
    {
        ompt_mutex_t   kind;
        ompt_wait_id_t wait_id;
        const void*    codeptr_ra;
    } lock_destroy;

    struct
    {
        ompt_mutex_t   kind;
        unsigned int   hint;
        unsigned int   impl;
        ompt_wait_id_t wait_id;
        const void*    codeptr_ra;
    } mutex_acquire;

    struct
    {
        ompt_mutex_t   kind;
        ompt_wait_id_t wait_id;
        const void*    codeptr_ra;
    } mutex_acquired;

    struct
    {
        ompt_scope_endpoint_t endpoint;
        ompt_wait_id_t        wait_id;
        const void*           codeptr_ra;
    } nest_lock;

    struct
    {
        ompt_data_t* thread_data;
        const void*  codeptr_ra;
    } flush;

    struct
    {
        ompt_data_t* task_data;
        int          flags;
        const void*  codeptr_ra;
    } cancel;

    struct
    {
        ompt_sync_region_t    kind;
        ompt_scope_endpoint_t endpoint;
        ompt_data_t*          parallel_data;
        ompt_data_t*          task_data;
        const void*           codeptr_ra;
    } reduction;

    struct
    {
        ompt_data_t*    parallel_data;
        ompt_data_t*    task_data;
        ompt_dispatch_t kind;
        ompt_data_t     instance;
    } dispatch;

    struct
    {
        ompt_target_t         kind;
        ompt_scope_endpoint_t endpoint;
        int                   device_num;
        ompt_data_t*          task_data;
        ompt_data_t*          target_task_data;
        ompt_data_t*          target_data;
        const void*           codeptr_ra;
    } target_emi;

    struct
    {
        ompt_scope_endpoint_t endpoint;
        ompt_data_t*          target_task_data;
        ompt_data_t*          target_data;
        ompt_data_t*          host_op_id;
        ompt_target_data_op_t optype;
        void*                 src_address;
        int                   src_device_num;
        void*                 dst_address;
        int                   dst_device_num;
        size_t                bytes;
        const void*           codeptr_ra;
    } target_data_op_emi;

    struct
    {
        ompt_scope_endpoint_t endpoint;
        ompt_data_t*          target_data;
        ompt_data_t*          host_op_id;
        unsigned int          requested_num_teams;
    } target_submit_emi;

    // struct
    // {
    //     unsigned int  nitems;
    //     void**        host_addr;
    //     void**        device_addr;
    //     size_t*       bytes;
    //     unsigned int* mapping_flags;
    //     const void*   codeptr_ra;
    // } target_map_emi;

    struct
    {
        ompt_severity_t severity;
        const char*     message;
        size_t          length;
        const void*     codeptr_ra;
    } error;

    rocprofiler_ompt_callback_functions_t callback_functions;

} rocprofiler_ompt_args_t;

ROCPROFILER_EXTERN_C_FINI
