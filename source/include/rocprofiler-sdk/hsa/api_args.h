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
#include <rocprofiler-sdk/ext_version.h>
#include <rocprofiler-sdk/hsa/api_trace_version.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ext_finalize.h>
#include <hsa/hsa_ext_image.h>

ROCPROFILER_EXTERN_C_INIT

// Empty struct has a size of 0 in C but size of 1 in C++.
// This struct is added to the union members which represent
// functions with no arguments to ensure ABI compatibility
typedef struct rocprofiler_hsa_api_no_args
{
    char empty;
} rocprofiler_hsa_api_no_args;

typedef union rocprofiler_hsa_api_retval_t
{
    uint64_t           uint64_t_retval;
    uint32_t           uint32_t_retval;
    hsa_signal_value_t hsa_signal_value_t_retval;
    hsa_status_t       hsa_status_t_retval;
} rocprofiler_hsa_api_retval_t;

// the following hsa_* typedefs are only in hsa/hsa_api_trace.h but we cannot include that file here
// because it is not C-compatible
typedef hsa_status_t (*hsa_ext_program_iterate_modules_cb_t)(hsa_ext_program_t program,
                                                             hsa_ext_module_t  module,
                                                             void*             data);

typedef void (*hsa_amd_queue_intercept_packet_writer)(const void* pkts, uint64_t pkt_count);

typedef void (*hsa_amd_queue_intercept_handler)(const void* pkts,
                                                uint64_t    pkt_count,
                                                uint64_t    user_pkt_index,
                                                void*       data,
                                                hsa_amd_queue_intercept_packet_writer writer);

typedef void (*hsa_amd_runtime_queue_notifier)(const hsa_queue_t* queue,
                                               hsa_agent_t        agent,
                                               void*              data);

typedef union rocprofiler_hsa_api_args_t
{
    // block: CoreApi API
    struct
    {
        // Empty struct has a size of 0 in C but size of 1 in C++.
        // Add the rocprofiler_hsa_api_no_args struct to fix this
        rocprofiler_hsa_api_no_args no_args;
    } hsa_init;
    struct
    {
        // Empty struct has a size of 0 in C but size of 1 in C++.
        // Add the rocprofiler_hsa_api_no_args struct to fix this
        rocprofiler_hsa_api_no_args no_args;
    } hsa_shut_down;
    struct
    {
        hsa_system_info_t attribute;
        void*             value;
    } hsa_system_get_info;
    struct
    {
        uint16_t extension;
        uint16_t version_major;
        uint16_t version_minor;
        bool*    result;
    } hsa_system_extension_supported;
    struct
    {
        uint16_t extension;
        uint16_t version_major;
        uint16_t version_minor;
        void*    table;
    } hsa_system_get_extension_table;
    struct
    {
        hsa_status_t (*callback)(hsa_agent_t agent, void* data);
        void* data;
    } hsa_iterate_agents;
    struct
    {
        hsa_agent_t      agent;
        hsa_agent_info_t attribute;
        void*            value;
    } hsa_agent_get_info;
    struct
    {
        hsa_agent_t        agent;
        uint32_t           size;
        hsa_queue_type32_t type;
        void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data);
        void*         data;
        uint32_t      private_segment_size;
        uint32_t      group_segment_size;
        hsa_queue_t** queue;
    } hsa_queue_create;
    struct
    {
        hsa_region_t       region;
        uint32_t           size;
        hsa_queue_type32_t type;
        uint32_t           features;
        hsa_signal_t       doorbell_signal;
        hsa_queue_t**      queue;
    } hsa_soft_queue_create;
    struct
    {
        hsa_queue_t* queue;
    } hsa_queue_destroy;
    struct
    {
        hsa_queue_t* queue;
    } hsa_queue_inactivate;
    struct
    {
        const hsa_queue_t* queue;
    } hsa_queue_load_read_index_scacquire;
    struct
    {
        const hsa_queue_t* queue;
    } hsa_queue_load_read_index_relaxed;
    struct
    {
        const hsa_queue_t* queue;
    } hsa_queue_load_write_index_scacquire;
    struct
    {
        const hsa_queue_t* queue;
    } hsa_queue_load_write_index_relaxed;
    struct
    {
        const hsa_queue_t* queue;
        uint64_t           value;
    } hsa_queue_store_write_index_relaxed;
    struct
    {
        const hsa_queue_t* queue;
        uint64_t           value;
    } hsa_queue_store_write_index_screlease;
    struct
    {
        const hsa_queue_t* queue;
        uint64_t           expected;
        uint64_t           value;
    } hsa_queue_cas_write_index_scacq_screl;
    struct
    {
        const hsa_queue_t* queue;
        uint64_t           expected;
        uint64_t           value;
    } hsa_queue_cas_write_index_scacquire;
    struct
    {
        const hsa_queue_t* queue;
        uint64_t           expected;
        uint64_t           value;
    } hsa_queue_cas_write_index_relaxed;
    struct
    {
        const hsa_queue_t* queue;
        uint64_t           expected;
        uint64_t           value;
    } hsa_queue_cas_write_index_screlease;
    struct
    {
        const hsa_queue_t* queue;
        uint64_t           value;
    } hsa_queue_add_write_index_scacq_screl;
    struct
    {
        const hsa_queue_t* queue;
        uint64_t           value;
    } hsa_queue_add_write_index_scacquire;
    struct
    {
        const hsa_queue_t* queue;
        uint64_t           value;
    } hsa_queue_add_write_index_relaxed;
    struct
    {
        const hsa_queue_t* queue;
        uint64_t           value;
    } hsa_queue_add_write_index_screlease;
    struct
    {
        const hsa_queue_t* queue;
        uint64_t           value;
    } hsa_queue_store_read_index_relaxed;
    struct
    {
        const hsa_queue_t* queue;
        uint64_t           value;
    } hsa_queue_store_read_index_screlease;
    struct
    {
        hsa_agent_t agent;
        hsa_status_t (*callback)(hsa_region_t region, void* data);
        void* data;
    } hsa_agent_iterate_regions;
    struct
    {
        hsa_region_t      region;
        hsa_region_info_t attribute;
        void*             value;
    } hsa_region_get_info;
    struct
    {
        hsa_agent_t   agent;
        hsa_profile_t profile;
        uint16_t*     mask;
    } hsa_agent_get_exception_policies;
    struct
    {
        uint16_t    extension;
        hsa_agent_t agent;
        uint16_t    version_major;
        uint16_t    version_minor;
        bool*       result;
    } hsa_agent_extension_supported;
    struct
    {
        void*  ptr;
        size_t size;
    } hsa_memory_register;
    struct
    {
        void*  ptr;
        size_t size;
    } hsa_memory_deregister;
    struct
    {
        hsa_region_t region;
        size_t       size;
        void**       ptr;
    } hsa_memory_allocate;
    struct
    {
        void* ptr;
    } hsa_memory_free;
    struct
    {
        void*       dst;
        const void* src;
        size_t      size;
    } hsa_memory_copy;
    struct
    {
        void*                   ptr;
        hsa_agent_t             agent;
        hsa_access_permission_t access;
    } hsa_memory_assign_agent;
    struct
    {
        hsa_signal_value_t initial_value;
        uint32_t           num_consumers;
        const hsa_agent_t* consumers;
        hsa_signal_t*      signal;
    } hsa_signal_create;
    struct
    {
        hsa_signal_t signal;
    } hsa_signal_destroy;
    struct
    {
        hsa_signal_t signal;
    } hsa_signal_load_relaxed;
    struct
    {
        hsa_signal_t signal;
    } hsa_signal_load_scacquire;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_store_relaxed;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_store_screlease;
    struct
    {
        hsa_signal_t           signal;
        hsa_signal_condition_t condition;
        hsa_signal_value_t     compare_value;
        uint64_t               timeout_hint;
        hsa_wait_state_t       wait_state_hint;
    } hsa_signal_wait_relaxed;
    struct
    {
        hsa_signal_t           signal;
        hsa_signal_condition_t condition;
        hsa_signal_value_t     compare_value;
        uint64_t               timeout_hint;
        hsa_wait_state_t       wait_state_hint;
    } hsa_signal_wait_scacquire;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_and_relaxed;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_and_scacquire;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_and_screlease;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_and_scacq_screl;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_or_relaxed;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_or_scacquire;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_or_screlease;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_or_scacq_screl;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_xor_relaxed;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_xor_scacquire;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_xor_screlease;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_xor_scacq_screl;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_exchange_relaxed;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_exchange_scacquire;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_exchange_screlease;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_exchange_scacq_screl;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_add_relaxed;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_add_scacquire;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_add_screlease;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_add_scacq_screl;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_subtract_relaxed;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_subtract_scacquire;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_subtract_screlease;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_subtract_scacq_screl;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t expected;
        hsa_signal_value_t value;
    } hsa_signal_cas_relaxed;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t expected;
        hsa_signal_value_t value;
    } hsa_signal_cas_scacquire;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t expected;
        hsa_signal_value_t value;
    } hsa_signal_cas_screlease;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t expected;
        hsa_signal_value_t value;
    } hsa_signal_cas_scacq_screl;
    struct
    {
        const char* name;
        hsa_isa_t*  isa;
    } hsa_isa_from_name;
    struct
    {
        hsa_isa_t      isa;
        hsa_isa_info_t attribute;
        uint32_t       index;
        void*          value;
    } hsa_isa_get_info;
    struct
    {
        hsa_isa_t code_object_isa;
        hsa_isa_t agent_isa;
        bool*     result;
    } hsa_isa_compatible;
    struct
    {
        hsa_code_object_t code_object;
        hsa_status_t (*alloc_callback)(size_t size, hsa_callback_data_t data, void** address);
        hsa_callback_data_t callback_data;
        const char*         options;
        void**              serialized_code_object;
        size_t*             serialized_code_object_size;
    } hsa_code_object_serialize;
    struct
    {
        void*              serialized_code_object;
        size_t             serialized_code_object_size;
        const char*        options;
        hsa_code_object_t* code_object;
    } hsa_code_object_deserialize;
    struct
    {
        hsa_code_object_t code_object;
    } hsa_code_object_destroy;
    struct
    {
        hsa_code_object_t      code_object;
        hsa_code_object_info_t attribute;
        void*                  value;
    } hsa_code_object_get_info;
    struct
    {
        hsa_code_object_t  code_object;
        const char*        symbol_name;
        hsa_code_symbol_t* symbol;
    } hsa_code_object_get_symbol;
    struct
    {
        hsa_code_symbol_t      code_symbol;
        hsa_code_symbol_info_t attribute;
        void*                  value;
    } hsa_code_symbol_get_info;
    struct
    {
        hsa_code_object_t code_object;
        hsa_status_t (*callback)(hsa_code_object_t code_object,
                                 hsa_code_symbol_t symbol,
                                 void*             data);
        void* data;
    } hsa_code_object_iterate_symbols;
    struct
    {
        hsa_profile_t          profile;
        hsa_executable_state_t executable_state;
        const char*            options;
        hsa_executable_t*      executable;
    } hsa_executable_create;
    struct
    {
        hsa_executable_t executable;
    } hsa_executable_destroy;
    struct
    {
        hsa_executable_t  executable;
        hsa_agent_t       agent;
        hsa_code_object_t code_object;
        const char*       options;
    } hsa_executable_load_code_object;
    struct
    {
        hsa_executable_t executable;
        const char*      options;
    } hsa_executable_freeze;
    struct
    {
        hsa_executable_t      executable;
        hsa_executable_info_t attribute;
        void*                 value;
    } hsa_executable_get_info;
    struct
    {
        hsa_executable_t executable;
        const char*      variable_name;
        void*            address;
    } hsa_executable_global_variable_define;
    struct
    {
        hsa_executable_t executable;
        hsa_agent_t      agent;
        const char*      variable_name;
        void*            address;
    } hsa_executable_agent_global_variable_define;
    struct
    {
        hsa_executable_t executable;
        hsa_agent_t      agent;
        const char*      variable_name;
        void*            address;
    } hsa_executable_readonly_variable_define;
    struct
    {
        hsa_executable_t executable;
        uint32_t*        result;
    } hsa_executable_validate;
    struct
    {
        hsa_executable_t         executable;
        const char*              module_name;
        const char*              symbol_name;
        hsa_agent_t              agent;
        int32_t                  call_convention;
        hsa_executable_symbol_t* symbol;
    } hsa_executable_get_symbol;
    struct
    {
        hsa_executable_symbol_t      executable_symbol;
        hsa_executable_symbol_info_t attribute;
        void*                        value;
    } hsa_executable_symbol_get_info;
    struct
    {
        hsa_executable_t executable;
        hsa_status_t (*callback)(hsa_executable_t exec, hsa_executable_symbol_t symbol, void* data);
        void* data;
    } hsa_executable_iterate_symbols;
    struct
    {
        hsa_status_t status;
        const char** status_string;
    } hsa_status_string;
    struct
    {
        uint16_t     extension;
        const char** name;
    } hsa_extension_get_name;
    struct
    {
        uint16_t  extension;
        uint16_t  version_major;
        uint16_t* version_minor;
        bool*     result;
    } hsa_system_major_extension_supported;
    struct
    {
        uint16_t extension;
        uint16_t version_major;
        size_t   table_length;
        void*    table;
    } hsa_system_get_major_extension_table;
    struct
    {
        uint16_t    extension;
        hsa_agent_t agent;
        uint16_t    version_major;
        uint16_t*   version_minor;
        bool*       result;
    } hsa_agent_major_extension_supported;
    struct
    {
        hsa_cache_t      cache;
        hsa_cache_info_t attribute;
        void*            value;
    } hsa_cache_get_info;
    struct
    {
        hsa_agent_t agent;
        hsa_status_t (*callback)(hsa_cache_t cache, void* data);
        void* data;
    } hsa_agent_iterate_caches;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_silent_store_relaxed;
    struct
    {
        hsa_signal_t       signal;
        hsa_signal_value_t value;
    } hsa_signal_silent_store_screlease;
    struct
    {
        uint32_t            num_signals;
        const hsa_signal_t* signals;
        uint32_t            num_consumers;
        const hsa_agent_t*  consumers;
        hsa_signal_group_t* signal_group;
    } hsa_signal_group_create;
    struct
    {
        hsa_signal_group_t signal_group;
    } hsa_signal_group_destroy;
    struct
    {
        hsa_signal_group_t            signal_group;
        const hsa_signal_condition_t* conditions;
        const hsa_signal_value_t*     compare_values;
        hsa_wait_state_t              wait_state_hint;
        hsa_signal_t*                 signal;
        hsa_signal_value_t*           value;
    } hsa_signal_group_wait_any_scacquire;
    struct
    {
        hsa_signal_group_t            signal_group;
        const hsa_signal_condition_t* conditions;
        const hsa_signal_value_t*     compare_values;
        hsa_wait_state_t              wait_state_hint;
        hsa_signal_t*                 signal;
        hsa_signal_value_t*           value;
    } hsa_signal_group_wait_any_relaxed;
    struct
    {
        hsa_agent_t agent;
        hsa_status_t (*callback)(hsa_isa_t isa, void* data);
        void* data;
    } hsa_agent_iterate_isas;
    struct
    {
        hsa_isa_t      isa;
        hsa_isa_info_t attribute;
        void*          value;
    } hsa_isa_get_info_alt;
    struct
    {
        hsa_isa_t     isa;
        hsa_profile_t profile;
        uint16_t*     mask;
    } hsa_isa_get_exception_policies;
    struct
    {
        hsa_isa_t           isa;
        hsa_fp_type_t       fp_type;
        hsa_flush_mode_t    flush_mode;
        hsa_round_method_t* round_method;
    } hsa_isa_get_round_method;
    struct
    {
        hsa_wavefront_t      wavefront;
        hsa_wavefront_info_t attribute;
        void*                value;
    } hsa_wavefront_get_info;
    struct
    {
        hsa_isa_t isa;
        hsa_status_t (*callback)(hsa_wavefront_t wavefront, void* data);
        void* data;
    } hsa_isa_iterate_wavefronts;
    struct
    {
        hsa_code_object_t  code_object;
        const char*        module_name;
        const char*        symbol_name;
        hsa_code_symbol_t* symbol;
    } hsa_code_object_get_symbol_from_name;
    struct
    {
        hsa_file_t                file;
        hsa_code_object_reader_t* code_object_reader;
    } hsa_code_object_reader_create_from_file;
    struct
    {
        const void*               code_object;
        size_t                    size;
        hsa_code_object_reader_t* code_object_reader;
    } hsa_code_object_reader_create_from_memory;
    struct
    {
        hsa_code_object_reader_t code_object_reader;
    } hsa_code_object_reader_destroy;
    struct
    {
        hsa_profile_t                     profile;
        hsa_default_float_rounding_mode_t default_float_rounding_mode;
        const char*                       options;
        hsa_executable_t*                 executable;
    } hsa_executable_create_alt;
    struct
    {
        hsa_executable_t          executable;
        hsa_code_object_reader_t  code_object_reader;
        const char*               options;
        hsa_loaded_code_object_t* loaded_code_object;
    } hsa_executable_load_program_code_object;
    struct
    {
        hsa_executable_t          executable;
        hsa_agent_t               agent;
        hsa_code_object_reader_t  code_object_reader;
        const char*               options;
        hsa_loaded_code_object_t* loaded_code_object;
    } hsa_executable_load_agent_code_object;
    struct
    {
        hsa_executable_t executable;
        const char*      options;
        uint32_t*        result;
    } hsa_executable_validate_alt;
    struct
    {
        hsa_executable_t         executable;
        const char*              symbol_name;
        const hsa_agent_t*       agent;
        hsa_executable_symbol_t* symbol;
    } hsa_executable_get_symbol_by_name;
    struct
    {
        hsa_executable_t executable;
        hsa_agent_t      agent;
        hsa_status_t (*callback)(hsa_executable_t        exec,
                                 hsa_agent_t             agent,
                                 hsa_executable_symbol_t symbol,
                                 void*                   data);
        void* data;
    } hsa_executable_iterate_agent_symbols;
    struct
    {
        hsa_executable_t executable;
        hsa_status_t (*callback)(hsa_executable_t exec, hsa_executable_symbol_t symbol, void* data);
        void* data;
    } hsa_executable_iterate_program_symbols;

    // block: AmdExt API
    struct
    {
        hsa_agent_t               agent;
        hsa_amd_coherency_type_t* type;
    } hsa_amd_coherency_get_type;
    struct
    {
        hsa_agent_t              agent;
        hsa_amd_coherency_type_t type;
    } hsa_amd_coherency_set_type;
    struct
    {
        hsa_queue_t* queue;
        int          enable;
    } hsa_amd_profiling_set_profiler_enabled;
    struct
    {
        bool enable;
    } hsa_amd_profiling_async_copy_enable;
    struct
    {
        hsa_agent_t                        agent;
        hsa_signal_t                       signal;
        hsa_amd_profiling_dispatch_time_t* time;
    } hsa_amd_profiling_get_dispatch_time;
    struct
    {
        hsa_signal_t                         signal;
        hsa_amd_profiling_async_copy_time_t* time;
    } hsa_amd_profiling_get_async_copy_time;
    struct
    {
        hsa_agent_t agent;
        uint64_t    agent_tick;
        uint64_t*   system_tick;
    } hsa_amd_profiling_convert_tick_to_system_domain;
    struct
    {
        hsa_signal_t           signal;
        hsa_signal_condition_t cond;
        hsa_signal_value_t     value;
        hsa_amd_signal_handler handler;
        void*                  arg;
    } hsa_amd_signal_async_handler;
    struct
    {
        void (*callback)(void* arg);
        void* arg;
    } hsa_amd_async_function;
    struct
    {
        uint32_t                signal_count;
        hsa_signal_t*           signals;
        hsa_signal_condition_t* conds;
        hsa_signal_value_t*     values;
        uint64_t                timeout_hint;
        hsa_wait_state_t        wait_hint;
        hsa_signal_value_t*     satisfying_value;
    } hsa_amd_signal_wait_any;
    struct
    {
        const hsa_queue_t* queue;
        uint32_t           num_cu_mask_count;
        const uint32_t*    cu_mask;
    } hsa_amd_queue_cu_set_mask;
    struct
    {
        hsa_amd_memory_pool_t      memory_pool;
        hsa_amd_memory_pool_info_t attribute;
        void*                      value;
    } hsa_amd_memory_pool_get_info;
    struct
    {
        hsa_agent_t agent;
        hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data);
        void* data;
    } hsa_amd_agent_iterate_memory_pools;
    struct
    {
        hsa_amd_memory_pool_t memory_pool;
        size_t                size;
        uint32_t              flags;
        void**                ptr;
    } hsa_amd_memory_pool_allocate;
    struct
    {
        void* ptr;
    } hsa_amd_memory_pool_free;
    struct
    {
        void*               dst;
        hsa_agent_t         dst_agent;
        const void*         src;
        hsa_agent_t         src_agent;
        size_t              size;
        uint32_t            num_dep_signals;
        const hsa_signal_t* dep_signals;
        hsa_signal_t        completion_signal;
    } hsa_amd_memory_async_copy;
    struct
    {
        void*                    dst;
        hsa_agent_t              dst_agent;
        const void*              src;
        hsa_agent_t              src_agent;
        size_t                   size;
        uint32_t                 num_dep_signals;
        const hsa_signal_t*      dep_signals;
        hsa_signal_t             completion_signal;
        hsa_amd_sdma_engine_id_t engine_id;
        bool                     force_copy_on_sdma;
    } hsa_amd_memory_async_copy_on_engine;
    struct
    {
        hsa_agent_t dst_agent;
        hsa_agent_t src_agent;
        uint32_t*   engine_ids_mask;
    } hsa_amd_memory_copy_engine_status;
    struct
    {
        hsa_agent_t                      agent;
        hsa_amd_memory_pool_t            memory_pool;
        hsa_amd_agent_memory_pool_info_t attribute;
        void*                            value;
    } hsa_amd_agent_memory_pool_get_info;
    struct
    {
        uint32_t           num_agents;
        const hsa_agent_t* agents;
        const uint32_t*    flags;
        const void*        ptr;
    } hsa_amd_agents_allow_access;
    struct
    {
        hsa_amd_memory_pool_t src_memory_pool;
        hsa_amd_memory_pool_t dst_memory_pool;
        bool*                 result;
    } hsa_amd_memory_pool_can_migrate;
    struct
    {
        const void*           ptr;
        hsa_amd_memory_pool_t memory_pool;
        uint32_t              flags;
    } hsa_amd_memory_migrate;
    struct
    {
        void*        host_ptr;
        size_t       size;
        hsa_agent_t* agents;
        int          num_agent;
        void**       agent_ptr;
    } hsa_amd_memory_lock;
    struct
    {
        void* host_ptr;
    } hsa_amd_memory_unlock;
    struct
    {
        void*    ptr;
        uint32_t value;
        size_t   count;
    } hsa_amd_memory_fill;
    struct
    {
        uint32_t     num_agents;
        hsa_agent_t* agents;
        int          interop_handle;
        uint32_t     flags;
        size_t*      size;
        void**       ptr;
        size_t*      metadata_size;
        const void** metadata;
    } hsa_amd_interop_map_buffer;
    struct
    {
        void* ptr;
    } hsa_amd_interop_unmap_buffer;
    struct
    {
        hsa_agent_t                       agent;
        const hsa_ext_image_descriptor_t* image_descriptor;
        const hsa_amd_image_descriptor_t* image_layout;
        const void*                       image_data;
        hsa_access_permission_t           access_permission;
        hsa_ext_image_t*                  image;
    } hsa_amd_image_create;
    struct
    {
        const void*             ptr;
        hsa_amd_pointer_info_t* info;
        void* (*alloc)(size_t);
        uint32_t*     num_agents_accessible;
        hsa_agent_t** accessible;
    } hsa_amd_pointer_info;
    struct
    {
        const void* ptr;
        void*       userdata;
    } hsa_amd_pointer_info_set_userdata;
    struct
    {
        void*                 ptr;
        size_t                len;
        hsa_amd_ipc_memory_t* handle;
    } hsa_amd_ipc_memory_create;
    struct
    {
        const hsa_amd_ipc_memory_t* handle;
        size_t                      len;
        uint32_t                    num_agents;
        const hsa_agent_t*          mapping_agents;
        void**                      mapped_ptr;
    } hsa_amd_ipc_memory_attach;
    struct
    {
        void* mapped_ptr;
    } hsa_amd_ipc_memory_detach;
    struct
    {
        hsa_signal_value_t initial_value;
        uint32_t           num_consumers;
        const hsa_agent_t* consumers;
        uint64_t           attributes;
        hsa_signal_t*      signal;
    } hsa_amd_signal_create;
    struct
    {
        hsa_signal_t          signal;
        hsa_amd_ipc_signal_t* handle;
    } hsa_amd_ipc_signal_create;
    struct
    {
        const hsa_amd_ipc_signal_t* handle;
        hsa_signal_t*               signal;
    } hsa_amd_ipc_signal_attach;
    struct
    {
        hsa_amd_system_event_callback_t callback;
        void*                           data;
    } hsa_amd_register_system_event_handler;
    struct
    {
        hsa_agent_t        agent_handle;
        uint32_t           size;
        hsa_queue_type32_t type;
        void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data);
        void*         data;
        uint32_t      private_segment_size;
        uint32_t      group_segment_size;
        hsa_queue_t** queue;
    } hsa_amd_queue_intercept_create;
    struct
    {
        hsa_queue_t*                    queue;
        hsa_amd_queue_intercept_handler callback;
        void*                           user_data;
    } hsa_amd_queue_intercept_register;
    struct
    {
        hsa_queue_t*             queue;
        hsa_amd_queue_priority_t priority;
    } hsa_amd_queue_set_priority;
    struct
    {
        const hsa_pitched_ptr_t* dst;
        const hsa_dim3_t*        dst_offset;
        const hsa_pitched_ptr_t* src;
        const hsa_dim3_t*        src_offset;
        const hsa_dim3_t*        range;
        hsa_agent_t              copy_agent;
        hsa_amd_copy_direction_t dir;
        uint32_t                 num_dep_signals;
        const hsa_signal_t*      dep_signals;
        hsa_signal_t             completion_signal;
    } hsa_amd_memory_async_copy_rect;
    struct
    {
        hsa_amd_runtime_queue_notifier callback;
        void*                          user_data;
    } hsa_amd_runtime_queue_create_register;
    struct
    {
        void*                 host_ptr;
        size_t                size;
        hsa_agent_t*          agents;
        int                   num_agent;
        hsa_amd_memory_pool_t pool;
        uint32_t              flags;
        void**                agent_ptr;
    } hsa_amd_memory_lock_to_pool;
    struct
    {
        void*                           ptr;
        hsa_amd_deallocation_callback_t callback;
        void*                           user_data;
    } hsa_amd_register_deallocation_callback;
    struct
    {
        void*                           ptr;
        hsa_amd_deallocation_callback_t callback;
    } hsa_amd_deregister_deallocation_callback;
    struct
    {
        hsa_signal_t                  signal;
        volatile hsa_signal_value_t** value_ptr;
    } hsa_amd_signal_value_pointer;
    struct
    {
        void*                         ptr;
        size_t                        size;
        hsa_amd_svm_attribute_pair_t* attribute_list;
        size_t                        attribute_count;
    } hsa_amd_svm_attributes_set;
    struct
    {
        void*                         ptr;
        size_t                        size;
        hsa_amd_svm_attribute_pair_t* attribute_list;
        size_t                        attribute_count;
    } hsa_amd_svm_attributes_get;
    struct
    {
        void*               ptr;
        size_t              size;
        hsa_agent_t         agent;
        uint32_t            num_dep_signals;
        const hsa_signal_t* dep_signals;
        hsa_signal_t        completion_signal;
    } hsa_amd_svm_prefetch_async;
    struct
    {
        hsa_agent_t preferred_agent;
    } hsa_amd_spm_acquire;
    struct
    {
        hsa_agent_t preferred_agent;
    } hsa_amd_spm_release;
    struct
    {
        hsa_agent_t preferred_agent;
        size_t      size_in_bytes;
        uint32_t*   timeout;
        uint32_t*   size_copied;
        void*       dest;
        bool*       is_data_loss;
    } hsa_amd_spm_set_dest_buffer;
    struct
    {
        const hsa_queue_t* queue;
        uint32_t           num_cu_mask_count;
        uint32_t*          cu_mask;
    } hsa_amd_queue_cu_get_mask;
    struct
    {
        const void* ptr;
        size_t      size;
        int*        dmabuf;
        uint64_t*   offset;
    } hsa_amd_portable_export_dmabuf;
    struct
    {
        int dmabuf;
    } hsa_amd_portable_close_dmabuf;

    // block: ImageExt API
    struct
    {
        hsa_agent_t                   agent;
        hsa_ext_image_geometry_t      geometry;
        const hsa_ext_image_format_t* image_format;
        uint32_t*                     capability_mask;
    } hsa_ext_image_get_capability;
    struct
    {
        hsa_agent_t                       agent;
        const hsa_ext_image_descriptor_t* image_descriptor;
        hsa_access_permission_t           access_permission;
        hsa_ext_image_data_info_t*        image_data_info;
    } hsa_ext_image_data_get_info;
    struct
    {
        hsa_agent_t                       agent;
        const hsa_ext_image_descriptor_t* image_descriptor;
        const void*                       image_data;
        hsa_access_permission_t           access_permission;
        hsa_ext_image_t*                  image;
    } hsa_ext_image_create;
    struct
    {
        hsa_agent_t                   agent;
        const void*                   src_memory;
        size_t                        src_row_pitch;
        size_t                        src_slice_pitch;
        hsa_ext_image_t               dst_image;
        const hsa_ext_image_region_t* image_region;
    } hsa_ext_image_import;
    struct
    {
        hsa_agent_t                   agent;
        hsa_ext_image_t               src_image;
        void*                         dst_memory;
        size_t                        dst_row_pitch;
        size_t                        dst_slice_pitch;
        const hsa_ext_image_region_t* image_region;
    } hsa_ext_image_export;
    struct
    {
        hsa_agent_t       agent;
        hsa_ext_image_t   src_image;
        const hsa_dim3_t* src_offset;
        hsa_ext_image_t   dst_image;
        const hsa_dim3_t* dst_offset;
        const hsa_dim3_t* range;
    } hsa_ext_image_copy;
    struct
    {
        hsa_agent_t                   agent;
        hsa_ext_image_t               image;
        const void*                   data;
        const hsa_ext_image_region_t* image_region;
    } hsa_ext_image_clear;
    struct
    {
        hsa_agent_t     agent;
        hsa_ext_image_t image;
    } hsa_ext_image_destroy;
    struct
    {
        hsa_agent_t                         agent;
        const hsa_ext_sampler_descriptor_t* sampler_descriptor;
        hsa_ext_sampler_t*                  sampler;
    } hsa_ext_sampler_create;
    struct
    {
        hsa_agent_t       agent;
        hsa_ext_sampler_t sampler;
    } hsa_ext_sampler_destroy;
    struct
    {
        hsa_agent_t                   agent;
        hsa_ext_image_geometry_t      geometry;
        const hsa_ext_image_format_t* image_format;
        hsa_ext_image_data_layout_t   image_data_layout;
        uint32_t*                     capability_mask;
    } hsa_ext_image_get_capability_with_layout;
    struct
    {
        hsa_agent_t                       agent;
        const hsa_ext_image_descriptor_t* image_descriptor;
        hsa_access_permission_t           access_permission;
        hsa_ext_image_data_layout_t       image_data_layout;
        size_t                            image_data_row_pitch;
        size_t                            image_data_slice_pitch;
        hsa_ext_image_data_info_t*        image_data_info;
    } hsa_ext_image_data_get_info_with_layout;
    struct
    {
        hsa_agent_t                       agent;
        const hsa_ext_image_descriptor_t* image_descriptor;
        const void*                       image_data;
        hsa_access_permission_t           access_permission;
        hsa_ext_image_data_layout_t       image_data_layout;
        size_t                            image_data_row_pitch;
        size_t                            image_data_slice_pitch;
        hsa_ext_image_t*                  image;
    } hsa_ext_image_create_with_layout;
    // finalize ext
    struct
    {
        hsa_machine_model_t               machine_model;
        hsa_profile_t                     profile;
        hsa_default_float_rounding_mode_t default_float_rounding_mode;
        const char*                       options;
        hsa_ext_program_t*                program;
    } hsa_ext_program_create;
    struct
    {
        hsa_ext_program_t program;
    } hsa_ext_program_destroy;
    struct
    {
        hsa_ext_program_t program;
        hsa_ext_module_t  module;
    } hsa_ext_program_add_module;
    struct
    {
        hsa_ext_program_t                    program;
        hsa_ext_program_iterate_modules_cb_t callback;
        void*                                data;
    } hsa_ext_program_iterate_modules;
    struct
    {
        hsa_ext_program_t      program;
        hsa_ext_program_info_t attribute;
        void*                  value;
    } hsa_ext_program_get_info;
    struct
    {
        hsa_ext_program_t            program;
        hsa_isa_t                    isa;
        int32_t                      call_convention;
        hsa_ext_control_directives_t control_directives;
        const char*                  options;
        hsa_code_object_type_t       code_object_type;
        hsa_code_object_t*           code_object;
    } hsa_ext_program_finalize;
    //
#if HSA_AMD_EXT_API_TABLE_MAJOR_VERSION >= 0x02
    struct
    {
        void**   ptr;
        size_t   size;
        uint64_t address;
        uint64_t flags;
    } hsa_amd_vmem_address_reserve;
    struct
    {
        void*  ptr;
        size_t size;
    } hsa_amd_vmem_address_free;
    struct
    {
        hsa_amd_memory_pool_t        pool;
        size_t                       size;
        hsa_amd_memory_type_t        type;
        uint64_t                     flags;
        hsa_amd_vmem_alloc_handle_t* memory_handle;
    } hsa_amd_vmem_handle_create;
    struct
    {
        hsa_amd_vmem_alloc_handle_t memory_handle;
    } hsa_amd_vmem_handle_release;
    struct
    {
        void*                       va;
        size_t                      size;
        size_t                      in_offset;
        hsa_amd_vmem_alloc_handle_t memory_handle;
        uint64_t                    flags;
    } hsa_amd_vmem_map;
    struct
    {
        void*  va;
        size_t size;
    } hsa_amd_vmem_unmap;
    struct
    {
        void*                               va;
        size_t                              size;
        const hsa_amd_memory_access_desc_t* desc;
        size_t                              desc_cnt;
    } hsa_amd_vmem_set_access;
    struct
    {
        void*                    va;
        hsa_access_permission_t* perms;
        hsa_agent_t              agent_handle;
    } hsa_amd_vmem_get_access;
    struct
    {
        int*                        dmabuf_fd;
        hsa_amd_vmem_alloc_handle_t handle;
        uint64_t                    flags;
    } hsa_amd_vmem_export_shareable_handle;
    struct
    {
        int                          dmabuf_fd;
        hsa_amd_vmem_alloc_handle_t* handle;
    } hsa_amd_vmem_import_shareable_handle;
    struct
    {
        hsa_amd_vmem_alloc_handle_t* handle;
        void*                        addr;
    } hsa_amd_vmem_retain_alloc_handle;
    struct
    {
        hsa_amd_vmem_alloc_handle_t alloc_handle;
        hsa_amd_memory_pool_t*      pool;
        hsa_amd_memory_type_t*      type;
    } hsa_amd_vmem_get_alloc_properties_from_handle;
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x01
    struct
    {
        hsa_agent_t agent;
        size_t      threshold;
    } hsa_amd_agent_set_async_scratch_limit;
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x02
    struct
    {
        hsa_queue_t*               queue;
        hsa_queue_info_attribute_t attribute;
        void*                      value;
    } hsa_amd_queue_get_info;
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x03
    struct
    {
        void**   ptr;
        size_t   size;
        uint64_t address;
        uint64_t alignment;
        uint64_t flags;
    } hsa_amd_vmem_address_reserve_align;
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x04
    struct
    {
        uint8_t* flags;
        void*    file;
    } hsa_amd_enable_logging;
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x05
    struct
    {
        uint32_t                signal_count;
        hsa_signal_t*           signals;
        hsa_signal_condition_t* conds;
        hsa_signal_value_t*     values;
        uint64_t                timeout_hint;
        hsa_wait_state_t        wait_hint;
        hsa_signal_value_t*     satisfying_values;
    } hsa_amd_signal_wait_all;
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x06
    struct
    {
        hsa_agent_t dst_agent;
        hsa_agent_t src_agent;
        uint32_t*   recommended_ids_mask;
    } hsa_amd_memory_get_preferred_copy_engine;
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x07
    struct
    {
        const void* ptr;
        size_t      size;
        int*        dmabuf;
        uint64_t*   offset;
        uint64_t    flags;
    } hsa_amd_portable_export_dmabuf_v2;
#    endif
#endif
} rocprofiler_hsa_api_args_t;

ROCPROFILER_EXTERN_C_FINI
