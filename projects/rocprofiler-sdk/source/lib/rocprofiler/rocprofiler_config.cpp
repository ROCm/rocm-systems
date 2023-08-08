
#include <rocprofiler/rocprofiler.h>
#include <rocprofiler/config.h>

#include "config_helpers.hpp"
#include "config_internal.hpp"

#include <atomic>
#include <cstddef>
#include <roctracer/roctx.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <mutex>
#include <iostream>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ext_image.h>
#include <hsa/hsa_api_trace.h>

typedef enum
{
    ACTIVITY_API_PHASE_ENTER = 0,
    ACTIVITY_API_PHASE_EXIT  = 1
} activity_api_phase_t;

typedef struct roctx_api_data_s
{
    union
    {
        struct
        {
            const char*      message;
            roctx_range_id_t id;
        };
        struct
        {
            const char* message;
        } roctxMarkA;
        struct
        {
            const char* message;
        } roctxRangePushA;
        struct
        {
            const char* message;
        } roctxRangePop;
        struct
        {
            const char*      message;
            roctx_range_id_t id;
        } roctxRangeStartA;
        struct
        {
            const char*      message;
            roctx_range_id_t id;
        } roctxRangeStop;
    } args;
} roctx_api_data_t;

typedef struct hsa_api_data_s
{
    uint64_t correlation_id;
    uint32_t phase;
    union
    {
        uint32_t           uint32_t_retval;
        hsa_signal_value_t hsa_signal_value_t_retval;
        uint64_t           uint64_t_retval;
        hsa_status_t       hsa_status_t_retval;
    };
    union
    {
        /* block: CoreApi API */
        struct
        {
        } hsa_init;
        struct
        {
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
            hsa_status_t (*callback)(hsa_executable_t        exec,
                                     hsa_executable_symbol_t symbol,
                                     void*                   data);
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
            hsa_status_t (*callback)(hsa_executable_t        exec,
                                     hsa_executable_symbol_t symbol,
                                     void*                   data);
            void* data;
        } hsa_executable_iterate_program_symbols;

        /* block: AmdExt API */
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
            hsa_dim3_t               range__val;
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

        /* block: ImageExt API */
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
    } args;
    uint64_t* phase_data;
} hsa_api_data_t;

// HIP API callbacks data structures
typedef struct hip_api_data_s
{
    uint64_t correlation_id;
    uint32_t phase;
    union
    {
        struct
        {
            dim3*        gridDim;
            dim3         gridDim__val;
            dim3*        blockDim;
            dim3         blockDim__val;
            size_t*      sharedMem;
            size_t       sharedMem__val;
            hipStream_t* stream;
            hipStream_t  stream__val;
        } __hipPopCallConfiguration;
        struct
        {
            dim3        gridDim;
            dim3        blockDim;
            size_t      sharedMem;
            hipStream_t stream;
        } __hipPushCallConfiguration;
        struct
        {
            hipArray**                    array;
            hipArray*                     array__val;
            const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray;
            HIP_ARRAY3D_DESCRIPTOR        pAllocateArray__val;
        } hipArray3DCreate;
        struct
        {
            HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor;
            HIP_ARRAY3D_DESCRIPTOR  pArrayDescriptor__val;
            hipArray*               array;
            hipArray                array__val;
        } hipArray3DGetDescriptor;
        struct
        {
            hipArray**                  pHandle;
            hipArray*                   pHandle__val;
            const HIP_ARRAY_DESCRIPTOR* pAllocateArray;
            HIP_ARRAY_DESCRIPTOR        pAllocateArray__val;
        } hipArrayCreate;
        struct
        {
            hipArray* array;
            hipArray  array__val;
        } hipArrayDestroy;
        struct
        {
            HIP_ARRAY_DESCRIPTOR* pArrayDescriptor;
            HIP_ARRAY_DESCRIPTOR  pArrayDescriptor__val;
            hipArray*             array;
            hipArray              array__val;
        } hipArrayGetDescriptor;
        struct
        {
            hipChannelFormatDesc* desc;
            hipChannelFormatDesc  desc__val;
            hipExtent*            extent;
            hipExtent             extent__val;
            unsigned int*         flags;
            unsigned int          flags__val;
            hipArray*             array;
            hipArray              array__val;
        } hipArrayGetInfo;
        struct
        {
            int*                   device;
            int                    device__val;
            const hipDeviceProp_t* prop;
            hipDeviceProp_t        prop__val;
        } hipChooseDevice;
        struct
        {
            dim3        gridDim;
            dim3        blockDim;
            size_t      sharedMem;
            hipStream_t stream;
        } hipConfigureCall;
        struct
        {
            hipSurfaceObject_t*    pSurfObject;
            hipSurfaceObject_t     pSurfObject__val;
            const hipResourceDesc* pResDesc;
            hipResourceDesc        pResDesc__val;
        } hipCreateSurfaceObject;
        struct
        {
            hipCtx_t*    ctx;
            hipCtx_t     ctx__val;
            unsigned int flags;
            hipDevice_t  device;
        } hipCtxCreate;
        struct
        {
            hipCtx_t ctx;
        } hipCtxDestroy;
        struct
        {
            hipCtx_t peerCtx;
        } hipCtxDisablePeerAccess;
        struct
        {
            hipCtx_t     peerCtx;
            unsigned int flags;
        } hipCtxEnablePeerAccess;
        struct
        {
            hipCtx_t ctx;
            int*     apiVersion;
            int      apiVersion__val;
        } hipCtxGetApiVersion;
        struct
        {
            hipFuncCache_t* cacheConfig;
            hipFuncCache_t  cacheConfig__val;
        } hipCtxGetCacheConfig;
        struct
        {
            hipCtx_t* ctx;
            hipCtx_t  ctx__val;
        } hipCtxGetCurrent;
        struct
        {
            hipDevice_t* device;
            hipDevice_t  device__val;
        } hipCtxGetDevice;
        struct
        {
            unsigned int* flags;
            unsigned int  flags__val;
        } hipCtxGetFlags;
        struct
        {
            hipSharedMemConfig* pConfig;
            hipSharedMemConfig  pConfig__val;
        } hipCtxGetSharedMemConfig;
        struct
        {
            hipCtx_t* ctx;
            hipCtx_t  ctx__val;
        } hipCtxPopCurrent;
        struct
        {
            hipCtx_t ctx;
        } hipCtxPushCurrent;
        struct
        {
            hipFuncCache_t cacheConfig;
        } hipCtxSetCacheConfig;
        struct
        {
            hipCtx_t ctx;
        } hipCtxSetCurrent;
        struct
        {
            hipSharedMemConfig config;
        } hipCtxSetSharedMemConfig;
        struct
        {
            hipExternalMemory_t extMem;
        } hipDestroyExternalMemory;
        struct
        {
            hipExternalSemaphore_t extSem;
        } hipDestroyExternalSemaphore;
        struct
        {
            hipSurfaceObject_t surfaceObject;
        } hipDestroySurfaceObject;
        struct
        {
            int* canAccessPeer;
            int  canAccessPeer__val;
            int  deviceId;
            int  peerDeviceId;
        } hipDeviceCanAccessPeer;
        struct
        {
            int*        major;
            int         major__val;
            int*        minor;
            int         minor__val;
            hipDevice_t device;
        } hipDeviceComputeCapability;
        struct
        {
            int peerDeviceId;
        } hipDeviceDisablePeerAccess;
        struct
        {
            int          peerDeviceId;
            unsigned int flags;
        } hipDeviceEnablePeerAccess;
        struct
        {
            hipDevice_t* device;
            hipDevice_t  device__val;
            int          ordinal;
        } hipDeviceGet;
        struct
        {
            int*                 pi;
            int                  pi__val;
            hipDeviceAttribute_t attr;
            int                  deviceId;
        } hipDeviceGetAttribute;
        struct
        {
            int*        device;
            int         device__val;
            const char* pciBusId;
            char        pciBusId__val;
        } hipDeviceGetByPCIBusId;
        struct
        {
            hipFuncCache_t* cacheConfig;
            hipFuncCache_t  cacheConfig__val;
        } hipDeviceGetCacheConfig;
        struct
        {
            hipMemPool_t* mem_pool;
            hipMemPool_t  mem_pool__val;
            int           device;
        } hipDeviceGetDefaultMemPool;
        struct
        {
            int                      device;
            hipGraphMemAttributeType attr;
            void*                    value;
        } hipDeviceGetGraphMemAttribute;
        struct
        {
            size_t*         pValue;
            size_t          pValue__val;
            enum hipLimit_t limit;
        } hipDeviceGetLimit;
        struct
        {
            hipMemPool_t* mem_pool;
            hipMemPool_t  mem_pool__val;
            int           device;
        } hipDeviceGetMemPool;
        struct
        {
            char*       name;
            char        name__val;
            int         len;
            hipDevice_t device;
        } hipDeviceGetName;
        struct
        {
            int*             value;
            int              value__val;
            hipDeviceP2PAttr attr;
            int              srcDevice;
            int              dstDevice;
        } hipDeviceGetP2PAttribute;
        struct
        {
            char* pciBusId;
            char  pciBusId__val;
            int   len;
            int   device;
        } hipDeviceGetPCIBusId;
        struct
        {
            hipSharedMemConfig* pConfig;
            hipSharedMemConfig  pConfig__val;
        } hipDeviceGetSharedMemConfig;
        struct
        {
            int* leastPriority;
            int  leastPriority__val;
            int* greatestPriority;
            int  greatestPriority__val;
        } hipDeviceGetStreamPriorityRange;
        struct
        {
            hipUUID*    uuid;
            hipUUID     uuid__val;
            hipDevice_t device;
        } hipDeviceGetUuid;
        struct
        {
            int device;
        } hipDeviceGraphMemTrim;
        struct
        {
            hipDevice_t   dev;
            unsigned int* flags;
            unsigned int  flags__val;
            int*          active;
            int           active__val;
        } hipDevicePrimaryCtxGetState;
        struct
        {
            hipDevice_t dev;
        } hipDevicePrimaryCtxRelease;
        struct
        {
            hipDevice_t dev;
        } hipDevicePrimaryCtxReset;
        struct
        {
            hipCtx_t*   pctx;
            hipCtx_t    pctx__val;
            hipDevice_t dev;
        } hipDevicePrimaryCtxRetain;
        struct
        {
            hipDevice_t  dev;
            unsigned int flags;
        } hipDevicePrimaryCtxSetFlags;
        struct
        {
            hipFuncCache_t cacheConfig;
        } hipDeviceSetCacheConfig;
        struct
        {
            int                      device;
            hipGraphMemAttributeType attr;
            void*                    value;
        } hipDeviceSetGraphMemAttribute;
        struct
        {
            enum hipLimit_t limit;
            size_t          value;
        } hipDeviceSetLimit;
        struct
        {
            int          device;
            hipMemPool_t mem_pool;
        } hipDeviceSetMemPool;
        struct
        {
            hipSharedMemConfig config;
        } hipDeviceSetSharedMemConfig;
        struct
        {
            size_t*     bytes;
            size_t      bytes__val;
            hipDevice_t device;
        } hipDeviceTotalMem;
        struct
        {
            int* driverVersion;
            int  driverVersion__val;
        } hipDriverGetVersion;
        struct
        {
            const hip_Memcpy2D* pCopy;
            hip_Memcpy2D        pCopy__val;
        } hipDrvMemcpy2DUnaligned;
        struct
        {
            const HIP_MEMCPY3D* pCopy;
            HIP_MEMCPY3D        pCopy__val;
        } hipDrvMemcpy3D;
        struct
        {
            const HIP_MEMCPY3D* pCopy;
            HIP_MEMCPY3D        pCopy__val;
            hipStream_t         stream;
        } hipDrvMemcpy3DAsync;
        struct
        {
            unsigned int          numAttributes;
            hipPointer_attribute* attributes;
            hipPointer_attribute  attributes__val;
            void**                data;
            void*                 data__val;
            hipDeviceptr_t        ptr;
        } hipDrvPointerGetAttributes;
        struct
        {
            hipEvent_t* event;
            hipEvent_t  event__val;
        } hipEventCreate;
        struct
        {
            hipEvent_t*  event;
            hipEvent_t   event__val;
            unsigned int flags;
        } hipEventCreateWithFlags;
        struct
        {
            hipEvent_t event;
        } hipEventDestroy;
        struct
        {
            float*     ms;
            float      ms__val;
            hipEvent_t start;
            hipEvent_t stop;
        } hipEventElapsedTime;
        struct
        {
            hipEvent_t event;
        } hipEventQuery;
        struct
        {
            hipEvent_t  event;
            hipStream_t stream;
        } hipEventRecord;
        struct
        {
            hipEvent_t event;
        } hipEventSynchronize;
        struct
        {
            int           device1;
            int           device2;
            unsigned int* linktype;
            unsigned int  linktype__val;
            unsigned int* hopcount;
            unsigned int  hopcount__val;
        } hipExtGetLinkTypeAndHopCount;
        struct
        {
            const void* function_address;
            dim3        numBlocks;
            dim3        dimBlocks;
            void**      args;
            void*       args__val;
            size_t      sharedMemBytes;
            hipStream_t stream;
            hipEvent_t  startEvent;
            hipEvent_t  stopEvent;
            int         flags;
        } hipExtLaunchKernel;
        struct
        {
            hipLaunchParams* launchParamsList;
            hipLaunchParams  launchParamsList__val;
            int              numDevices;
            unsigned int     flags;
        } hipExtLaunchMultiKernelMultiDevice;
        struct
        {
            void**       ptr;
            void*        ptr__val;
            size_t       sizeBytes;
            unsigned int flags;
        } hipExtMallocWithFlags;
        struct
        {
            hipFunction_t f;
            unsigned int  globalWorkSizeX;
            unsigned int  globalWorkSizeY;
            unsigned int  globalWorkSizeZ;
            unsigned int  localWorkSizeX;
            unsigned int  localWorkSizeY;
            unsigned int  localWorkSizeZ;
            size_t        sharedMemBytes;
            hipStream_t   hStream;
            void**        kernelParams;
            void*         kernelParams__val;
            void**        extra;
            void*         extra__val;
            hipEvent_t    startEvent;
            hipEvent_t    stopEvent;
            unsigned int  flags;
        } hipExtModuleLaunchKernel;
        struct
        {
            hipStream_t*        stream;
            hipStream_t         stream__val;
            unsigned int        cuMaskSize;
            const unsigned int* cuMask;
            unsigned int        cuMask__val;
        } hipExtStreamCreateWithCUMask;
        struct
        {
            hipStream_t   stream;
            unsigned int  cuMaskSize;
            unsigned int* cuMask;
            unsigned int  cuMask__val;
        } hipExtStreamGetCUMask;
        struct
        {
            void**                             devPtr;
            void*                              devPtr__val;
            hipExternalMemory_t                extMem;
            const hipExternalMemoryBufferDesc* bufferDesc;
            hipExternalMemoryBufferDesc        bufferDesc__val;
        } hipExternalMemoryGetMappedBuffer;
        struct
        {
            void* ptr;
        } hipFree;
        struct
        {
            hipArray* array;
            hipArray  array__val;
        } hipFreeArray;
        struct
        {
            void*       dev_ptr;
            hipStream_t stream;
        } hipFreeAsync;
        struct
        {
            void* ptr;
        } hipFreeHost;
        struct
        {
            hipMipmappedArray_t mipmappedArray;
        } hipFreeMipmappedArray;
        struct
        {
            int*                  value;
            int                   value__val;
            hipFunction_attribute attrib;
            hipFunction_t         hfunc;
        } hipFuncGetAttribute;
        struct
        {
            hipFuncAttributes* attr;
            hipFuncAttributes  attr__val;
            const void*        func;
        } hipFuncGetAttributes;
        struct
        {
            const void*      func;
            hipFuncAttribute attr;
            int              value;
        } hipFuncSetAttribute;
        struct
        {
            const void*    func;
            hipFuncCache_t config;
        } hipFuncSetCacheConfig;
        struct
        {
            const void*        func;
            hipSharedMemConfig config;
        } hipFuncSetSharedMemConfig;
        struct
        {
            unsigned int*   pHipDeviceCount;
            unsigned int    pHipDeviceCount__val;
            int*            pHipDevices;
            int             pHipDevices__val;
            unsigned int    hipDeviceCount;
            hipGLDeviceList deviceList;
        } hipGLGetDevices;
        struct
        {
            hipChannelFormatDesc* desc;
            hipChannelFormatDesc  desc__val;
            hipArray_const_t      array;
        } hipGetChannelDesc;
        struct
        {
            int* deviceId;
            int  deviceId__val;
        } hipGetDevice;
        struct
        {
            int* count;
            int  count__val;
        } hipGetDeviceCount;
        struct
        {
            unsigned int* flags;
            unsigned int  flags__val;
        } hipGetDeviceFlags;
        struct
        {
            hipDeviceProp_t* props;
            hipDeviceProp_t  props__val;
            hipDevice_t      device;
        } hipGetDeviceProperties;
        struct
        {
            hipArray_t*               levelArray;
            hipArray_t                levelArray__val;
            hipMipmappedArray_const_t mipmappedArray;
            unsigned int              level;
        } hipGetMipmappedArrayLevel;
        struct
        {
            void**      devPtr;
            void*       devPtr__val;
            const void* symbol;
        } hipGetSymbolAddress;
        struct
        {
            size_t*     size;
            size_t      size__val;
            const void* symbol;
        } hipGetSymbolSize;
        struct
        {
            hipGraphNode_t*       pGraphNode;
            hipGraphNode_t        pGraphNode__val;
            hipGraph_t            graph;
            const hipGraphNode_t* pDependencies;
            hipGraphNode_t        pDependencies__val;
            size_t                numDependencies;
            hipGraph_t            childGraph;
        } hipGraphAddChildGraphNode;
        struct
        {
            hipGraph_t            graph;
            const hipGraphNode_t* from;
            hipGraphNode_t        from__val;
            const hipGraphNode_t* to;
            hipGraphNode_t        to__val;
            size_t                numDependencies;
        } hipGraphAddDependencies;
        struct
        {
            hipGraphNode_t*       pGraphNode;
            hipGraphNode_t        pGraphNode__val;
            hipGraph_t            graph;
            const hipGraphNode_t* pDependencies;
            hipGraphNode_t        pDependencies__val;
            size_t                numDependencies;
        } hipGraphAddEmptyNode;
        struct
        {
            hipGraphNode_t*       pGraphNode;
            hipGraphNode_t        pGraphNode__val;
            hipGraph_t            graph;
            const hipGraphNode_t* pDependencies;
            hipGraphNode_t        pDependencies__val;
            size_t                numDependencies;
            hipEvent_t            event;
        } hipGraphAddEventRecordNode;
        struct
        {
            hipGraphNode_t*       pGraphNode;
            hipGraphNode_t        pGraphNode__val;
            hipGraph_t            graph;
            const hipGraphNode_t* pDependencies;
            hipGraphNode_t        pDependencies__val;
            size_t                numDependencies;
            hipEvent_t            event;
        } hipGraphAddEventWaitNode;
        struct
        {
            hipGraphNode_t*          pGraphNode;
            hipGraphNode_t           pGraphNode__val;
            hipGraph_t               graph;
            const hipGraphNode_t*    pDependencies;
            hipGraphNode_t           pDependencies__val;
            size_t                   numDependencies;
            const hipHostNodeParams* pNodeParams;
            hipHostNodeParams        pNodeParams__val;
        } hipGraphAddHostNode;
        struct
        {
            hipGraphNode_t*            pGraphNode;
            hipGraphNode_t             pGraphNode__val;
            hipGraph_t                 graph;
            const hipGraphNode_t*      pDependencies;
            hipGraphNode_t             pDependencies__val;
            size_t                     numDependencies;
            const hipKernelNodeParams* pNodeParams;
            hipKernelNodeParams        pNodeParams__val;
        } hipGraphAddKernelNode;
        struct
        {
            hipGraphNode_t*        pGraphNode;
            hipGraphNode_t         pGraphNode__val;
            hipGraph_t             graph;
            const hipGraphNode_t*  pDependencies;
            hipGraphNode_t         pDependencies__val;
            size_t                 numDependencies;
            hipMemAllocNodeParams* pNodeParams;
            hipMemAllocNodeParams  pNodeParams__val;
        } hipGraphAddMemAllocNode;
        struct
        {
            hipGraphNode_t*       pGraphNode;
            hipGraphNode_t        pGraphNode__val;
            hipGraph_t            graph;
            const hipGraphNode_t* pDependencies;
            hipGraphNode_t        pDependencies__val;
            size_t                numDependencies;
            void*                 dev_ptr;
        } hipGraphAddMemFreeNode;
        struct
        {
            hipGraphNode_t*         pGraphNode;
            hipGraphNode_t          pGraphNode__val;
            hipGraph_t              graph;
            const hipGraphNode_t*   pDependencies;
            hipGraphNode_t          pDependencies__val;
            size_t                  numDependencies;
            const hipMemcpy3DParms* pCopyParams;
            hipMemcpy3DParms        pCopyParams__val;
        } hipGraphAddMemcpyNode;
        struct
        {
            hipGraphNode_t*       pGraphNode;
            hipGraphNode_t        pGraphNode__val;
            hipGraph_t            graph;
            const hipGraphNode_t* pDependencies;
            hipGraphNode_t        pDependencies__val;
            size_t                numDependencies;
            void*                 dst;
            const void*           src;
            size_t                count;
            hipMemcpyKind         kind;
        } hipGraphAddMemcpyNode1D;
        struct
        {
            hipGraphNode_t*       pGraphNode;
            hipGraphNode_t        pGraphNode__val;
            hipGraph_t            graph;
            const hipGraphNode_t* pDependencies;
            hipGraphNode_t        pDependencies__val;
            size_t                numDependencies;
            void*                 dst;
            const void*           symbol;
            size_t                count;
            size_t                offset;
            hipMemcpyKind         kind;
        } hipGraphAddMemcpyNodeFromSymbol;
        struct
        {
            hipGraphNode_t*       pGraphNode;
            hipGraphNode_t        pGraphNode__val;
            hipGraph_t            graph;
            const hipGraphNode_t* pDependencies;
            hipGraphNode_t        pDependencies__val;
            size_t                numDependencies;
            const void*           symbol;
            const void*           src;
            size_t                count;
            size_t                offset;
            hipMemcpyKind         kind;
        } hipGraphAddMemcpyNodeToSymbol;
        struct
        {
            hipGraphNode_t*        pGraphNode;
            hipGraphNode_t         pGraphNode__val;
            hipGraph_t             graph;
            const hipGraphNode_t*  pDependencies;
            hipGraphNode_t         pDependencies__val;
            size_t                 numDependencies;
            const hipMemsetParams* pMemsetParams;
            hipMemsetParams        pMemsetParams__val;
        } hipGraphAddMemsetNode;
        struct
        {
            hipGraphNode_t node;
            hipGraph_t*    pGraph;
            hipGraph_t     pGraph__val;
        } hipGraphChildGraphNodeGetGraph;
        struct
        {
            hipGraph_t* pGraphClone;
            hipGraph_t  pGraphClone__val;
            hipGraph_t  originalGraph;
        } hipGraphClone;
        struct
        {
            hipGraph_t*  pGraph;
            hipGraph_t   pGraph__val;
            unsigned int flags;
        } hipGraphCreate;
        struct
        {
            hipGraph_t   graph;
            const char*  path;
            char         path__val;
            unsigned int flags;
        } hipGraphDebugDotPrint;
        struct
        {
            hipGraph_t graph;
        } hipGraphDestroy;
        struct
        {
            hipGraphNode_t node;
        } hipGraphDestroyNode;
        struct
        {
            hipGraphNode_t node;
            hipEvent_t*    event_out;
            hipEvent_t     event_out__val;
        } hipGraphEventRecordNodeGetEvent;
        struct
        {
            hipGraphNode_t node;
            hipEvent_t     event;
        } hipGraphEventRecordNodeSetEvent;
        struct
        {
            hipGraphNode_t node;
            hipEvent_t*    event_out;
            hipEvent_t     event_out__val;
        } hipGraphEventWaitNodeGetEvent;
        struct
        {
            hipGraphNode_t node;
            hipEvent_t     event;
        } hipGraphEventWaitNodeSetEvent;
        struct
        {
            hipGraphExec_t hGraphExec;
            hipGraphNode_t node;
            hipGraph_t     childGraph;
        } hipGraphExecChildGraphNodeSetParams;
        struct
        {
            hipGraphExec_t graphExec;
        } hipGraphExecDestroy;
        struct
        {
            hipGraphExec_t hGraphExec;
            hipGraphNode_t hNode;
            hipEvent_t     event;
        } hipGraphExecEventRecordNodeSetEvent;
        struct
        {
            hipGraphExec_t hGraphExec;
            hipGraphNode_t hNode;
            hipEvent_t     event;
        } hipGraphExecEventWaitNodeSetEvent;
        struct
        {
            hipGraphExec_t           hGraphExec;
            hipGraphNode_t           node;
            const hipHostNodeParams* pNodeParams;
            hipHostNodeParams        pNodeParams__val;
        } hipGraphExecHostNodeSetParams;
        struct
        {
            hipGraphExec_t             hGraphExec;
            hipGraphNode_t             node;
            const hipKernelNodeParams* pNodeParams;
            hipKernelNodeParams        pNodeParams__val;
        } hipGraphExecKernelNodeSetParams;
        struct
        {
            hipGraphExec_t    hGraphExec;
            hipGraphNode_t    node;
            hipMemcpy3DParms* pNodeParams;
            hipMemcpy3DParms  pNodeParams__val;
        } hipGraphExecMemcpyNodeSetParams;
        struct
        {
            hipGraphExec_t hGraphExec;
            hipGraphNode_t node;
            void*          dst;
            const void*    src;
            size_t         count;
            hipMemcpyKind  kind;
        } hipGraphExecMemcpyNodeSetParams1D;
        struct
        {
            hipGraphExec_t hGraphExec;
            hipGraphNode_t node;
            void*          dst;
            const void*    symbol;
            size_t         count;
            size_t         offset;
            hipMemcpyKind  kind;
        } hipGraphExecMemcpyNodeSetParamsFromSymbol;
        struct
        {
            hipGraphExec_t hGraphExec;
            hipGraphNode_t node;
            const void*    symbol;
            const void*    src;
            size_t         count;
            size_t         offset;
            hipMemcpyKind  kind;
        } hipGraphExecMemcpyNodeSetParamsToSymbol;
        struct
        {
            hipGraphExec_t         hGraphExec;
            hipGraphNode_t         node;
            const hipMemsetParams* pNodeParams;
            hipMemsetParams        pNodeParams__val;
        } hipGraphExecMemsetNodeSetParams;
        struct
        {
            hipGraphExec_t            hGraphExec;
            hipGraph_t                hGraph;
            hipGraphNode_t*           hErrorNode_out;
            hipGraphNode_t            hErrorNode_out__val;
            hipGraphExecUpdateResult* updateResult_out;
            hipGraphExecUpdateResult  updateResult_out__val;
        } hipGraphExecUpdate;
        struct
        {
            hipGraph_t      graph;
            hipGraphNode_t* from;
            hipGraphNode_t  from__val;
            hipGraphNode_t* to;
            hipGraphNode_t  to__val;
            size_t*         numEdges;
            size_t          numEdges__val;
        } hipGraphGetEdges;
        struct
        {
            hipGraph_t      graph;
            hipGraphNode_t* nodes;
            hipGraphNode_t  nodes__val;
            size_t*         numNodes;
            size_t          numNodes__val;
        } hipGraphGetNodes;
        struct
        {
            hipGraph_t      graph;
            hipGraphNode_t* pRootNodes;
            hipGraphNode_t  pRootNodes__val;
            size_t*         pNumRootNodes;
            size_t          pNumRootNodes__val;
        } hipGraphGetRootNodes;
        struct
        {
            hipGraphNode_t     node;
            hipHostNodeParams* pNodeParams;
            hipHostNodeParams  pNodeParams__val;
        } hipGraphHostNodeGetParams;
        struct
        {
            hipGraphNode_t           node;
            const hipHostNodeParams* pNodeParams;
            hipHostNodeParams        pNodeParams__val;
        } hipGraphHostNodeSetParams;
        struct
        {
            hipGraphExec_t* pGraphExec;
            hipGraphExec_t  pGraphExec__val;
            hipGraph_t      graph;
            hipGraphNode_t* pErrorNode;
            hipGraphNode_t  pErrorNode__val;
            char*           pLogBuffer;
            char            pLogBuffer__val;
            size_t          bufferSize;
        } hipGraphInstantiate;
        struct
        {
            hipGraphExec_t*    pGraphExec;
            hipGraphExec_t     pGraphExec__val;
            hipGraph_t         graph;
            unsigned long long flags;
        } hipGraphInstantiateWithFlags;
        struct
        {
            hipGraphNode_t hSrc;
            hipGraphNode_t hDst;
        } hipGraphKernelNodeCopyAttributes;
        struct
        {
            hipGraphNode_t          hNode;
            hipKernelNodeAttrID     attr;
            hipKernelNodeAttrValue* value;
            hipKernelNodeAttrValue  value__val;
        } hipGraphKernelNodeGetAttribute;
        struct
        {
            hipGraphNode_t       node;
            hipKernelNodeParams* pNodeParams;
            hipKernelNodeParams  pNodeParams__val;
        } hipGraphKernelNodeGetParams;
        struct
        {
            hipGraphNode_t                hNode;
            hipKernelNodeAttrID           attr;
            const hipKernelNodeAttrValue* value;
            hipKernelNodeAttrValue        value__val;
        } hipGraphKernelNodeSetAttribute;
        struct
        {
            hipGraphNode_t             node;
            const hipKernelNodeParams* pNodeParams;
            hipKernelNodeParams        pNodeParams__val;
        } hipGraphKernelNodeSetParams;
        struct
        {
            hipGraphExec_t graphExec;
            hipStream_t    stream;
        } hipGraphLaunch;
        struct
        {
            hipGraphNode_t         node;
            hipMemAllocNodeParams* pNodeParams;
            hipMemAllocNodeParams  pNodeParams__val;
        } hipGraphMemAllocNodeGetParams;
        struct
        {
            hipGraphNode_t node;
            void*          dev_ptr;
        } hipGraphMemFreeNodeGetParams;
        struct
        {
            hipGraphNode_t    node;
            hipMemcpy3DParms* pNodeParams;
            hipMemcpy3DParms  pNodeParams__val;
        } hipGraphMemcpyNodeGetParams;
        struct
        {
            hipGraphNode_t          node;
            const hipMemcpy3DParms* pNodeParams;
            hipMemcpy3DParms        pNodeParams__val;
        } hipGraphMemcpyNodeSetParams;
        struct
        {
            hipGraphNode_t node;
            void*          dst;
            const void*    src;
            size_t         count;
            hipMemcpyKind  kind;
        } hipGraphMemcpyNodeSetParams1D;
        struct
        {
            hipGraphNode_t node;
            void*          dst;
            const void*    symbol;
            size_t         count;
            size_t         offset;
            hipMemcpyKind  kind;
        } hipGraphMemcpyNodeSetParamsFromSymbol;
        struct
        {
            hipGraphNode_t node;
            const void*    symbol;
            const void*    src;
            size_t         count;
            size_t         offset;
            hipMemcpyKind  kind;
        } hipGraphMemcpyNodeSetParamsToSymbol;
        struct
        {
            hipGraphNode_t   node;
            hipMemsetParams* pNodeParams;
            hipMemsetParams  pNodeParams__val;
        } hipGraphMemsetNodeGetParams;
        struct
        {
            hipGraphNode_t         node;
            const hipMemsetParams* pNodeParams;
            hipMemsetParams        pNodeParams__val;
        } hipGraphMemsetNodeSetParams;
        struct
        {
            hipGraphNode_t* pNode;
            hipGraphNode_t  pNode__val;
            hipGraphNode_t  originalNode;
            hipGraph_t      clonedGraph;
        } hipGraphNodeFindInClone;
        struct
        {
            hipGraphNode_t  node;
            hipGraphNode_t* pDependencies;
            hipGraphNode_t  pDependencies__val;
            size_t*         pNumDependencies;
            size_t          pNumDependencies__val;
        } hipGraphNodeGetDependencies;
        struct
        {
            hipGraphNode_t  node;
            hipGraphNode_t* pDependentNodes;
            hipGraphNode_t  pDependentNodes__val;
            size_t*         pNumDependentNodes;
            size_t          pNumDependentNodes__val;
        } hipGraphNodeGetDependentNodes;
        struct
        {
            hipGraphExec_t hGraphExec;
            hipGraphNode_t hNode;
            unsigned int*  isEnabled;
            unsigned int   isEnabled__val;
        } hipGraphNodeGetEnabled;
        struct
        {
            hipGraphNode_t    node;
            hipGraphNodeType* pType;
            hipGraphNodeType  pType__val;
        } hipGraphNodeGetType;
        struct
        {
            hipGraphExec_t hGraphExec;
            hipGraphNode_t hNode;
            unsigned int   isEnabled;
        } hipGraphNodeSetEnabled;
        struct
        {
            hipGraph_t      graph;
            hipUserObject_t object;
            unsigned int    count;
        } hipGraphReleaseUserObject;
        struct
        {
            hipGraph_t            graph;
            const hipGraphNode_t* from;
            hipGraphNode_t        from__val;
            const hipGraphNode_t* to;
            hipGraphNode_t        to__val;
            size_t                numDependencies;
        } hipGraphRemoveDependencies;
        struct
        {
            hipGraph_t      graph;
            hipUserObject_t object;
            unsigned int    count;
            unsigned int    flags;
        } hipGraphRetainUserObject;
        struct
        {
            hipGraphExec_t graphExec;
            hipStream_t    stream;
        } hipGraphUpload;
        struct
        {
            hipGraphicsResource** resource;
            hipGraphicsResource*  resource__val;
            GLuint                buffer;
            unsigned int          flags;
        } hipGraphicsGLRegisterBuffer;
        struct
        {
            hipGraphicsResource** resource;
            hipGraphicsResource*  resource__val;
            GLuint                image;
            GLenum                target;
            unsigned int          flags;
        } hipGraphicsGLRegisterImage;
        struct
        {
            int                    count;
            hipGraphicsResource_t* resources;
            hipGraphicsResource_t  resources__val;
            hipStream_t            stream;
        } hipGraphicsMapResources;
        struct
        {
            void**                devPtr;
            void*                 devPtr__val;
            size_t*               size;
            size_t                size__val;
            hipGraphicsResource_t resource;
        } hipGraphicsResourceGetMappedPointer;
        struct
        {
            hipArray_t*           array;
            hipArray_t            array__val;
            hipGraphicsResource_t resource;
            unsigned int          arrayIndex;
            unsigned int          mipLevel;
        } hipGraphicsSubResourceGetMappedArray;
        struct
        {
            int                    count;
            hipGraphicsResource_t* resources;
            hipGraphicsResource_t  resources__val;
            hipStream_t            stream;
        } hipGraphicsUnmapResources;
        struct
        {
            hipGraphicsResource_t resource;
        } hipGraphicsUnregisterResource;
        struct
        {
            hipFunction_t f;
            unsigned int  globalWorkSizeX;
            unsigned int  globalWorkSizeY;
            unsigned int  globalWorkSizeZ;
            unsigned int  blockDimX;
            unsigned int  blockDimY;
            unsigned int  blockDimZ;
            size_t        sharedMemBytes;
            hipStream_t   hStream;
            void**        kernelParams;
            void*         kernelParams__val;
            void**        extra;
            void*         extra__val;
            hipEvent_t    startEvent;
            hipEvent_t    stopEvent;
        } hipHccModuleLaunchKernel;
        struct
        {
            void**       ptr;
            void*        ptr__val;
            size_t       size;
            unsigned int flags;
        } hipHostAlloc;
        struct
        {
            void* ptr;
        } hipHostFree;
        struct
        {
            void**       devPtr;
            void*        devPtr__val;
            void*        hstPtr;
            unsigned int flags;
        } hipHostGetDevicePointer;
        struct
        {
            unsigned int* flagsPtr;
            unsigned int  flagsPtr__val;
            void*         hostPtr;
        } hipHostGetFlags;
        struct
        {
            void**       ptr;
            void*        ptr__val;
            size_t       size;
            unsigned int flags;
        } hipHostMalloc;
        struct
        {
            void*        hostPtr;
            size_t       sizeBytes;
            unsigned int flags;
        } hipHostRegister;
        struct
        {
            void* hostPtr;
        } hipHostUnregister;
        struct
        {
            hipExternalMemory_t*               extMem_out;
            hipExternalMemory_t                extMem_out__val;
            const hipExternalMemoryHandleDesc* memHandleDesc;
            hipExternalMemoryHandleDesc        memHandleDesc__val;
        } hipImportExternalMemory;
        struct
        {
            hipExternalSemaphore_t*               extSem_out;
            hipExternalSemaphore_t                extSem_out__val;
            const hipExternalSemaphoreHandleDesc* semHandleDesc;
            hipExternalSemaphoreHandleDesc        semHandleDesc__val;
        } hipImportExternalSemaphore;
        struct
        {
            unsigned int flags;
        } hipInit;
        struct
        {
            void* devPtr;
        } hipIpcCloseMemHandle;
        struct
        {
            hipIpcEventHandle_t* handle;
            hipIpcEventHandle_t  handle__val;
            hipEvent_t           event;
        } hipIpcGetEventHandle;
        struct
        {
            hipIpcMemHandle_t* handle;
            hipIpcMemHandle_t  handle__val;
            void*              devPtr;
        } hipIpcGetMemHandle;
        struct
        {
            hipEvent_t*         event;
            hipEvent_t          event__val;
            hipIpcEventHandle_t handle;
        } hipIpcOpenEventHandle;
        struct
        {
            void**            devPtr;
            void*             devPtr__val;
            hipIpcMemHandle_t handle;
            unsigned int      flags;
        } hipIpcOpenMemHandle;
        struct
        {
            const void* hostFunction;
        } hipLaunchByPtr;
        struct
        {
            const void*  f;
            dim3         gridDim;
            dim3         blockDimX;
            void**       kernelParams;
            void*        kernelParams__val;
            unsigned int sharedMemBytes;
            hipStream_t  stream;
        } hipLaunchCooperativeKernel;
        struct
        {
            hipLaunchParams* launchParamsList;
            hipLaunchParams  launchParamsList__val;
            int              numDevices;
            unsigned int     flags;
        } hipLaunchCooperativeKernelMultiDevice;
        struct
        {
            hipStream_t stream;
            hipHostFn_t fn;
            void*       userData;
        } hipLaunchHostFunc;
        struct
        {
            const void* function_address;
            dim3        numBlocks;
            dim3        dimBlocks;
            void**      args;
            void*       args__val;
            size_t      sharedMemBytes;
            hipStream_t stream;
        } hipLaunchKernel;
        struct
        {
            void** ptr;
            void*  ptr__val;
            size_t size;
        } hipMalloc;
        struct
        {
            hipPitchedPtr* pitchedDevPtr;
            hipPitchedPtr  pitchedDevPtr__val;
            hipExtent      extent;
        } hipMalloc3D;
        struct
        {
            hipArray_t*                 array;
            hipArray_t                  array__val;
            const hipChannelFormatDesc* desc;
            hipChannelFormatDesc        desc__val;
            hipExtent                   extent;
            unsigned int                flags;
        } hipMalloc3DArray;
        struct
        {
            hipArray**                  array;
            hipArray*                   array__val;
            const hipChannelFormatDesc* desc;
            hipChannelFormatDesc        desc__val;
            size_t                      width;
            size_t                      height;
            unsigned int                flags;
        } hipMallocArray;
        struct
        {
            void**      dev_ptr;
            void*       dev_ptr__val;
            size_t      size;
            hipStream_t stream;
        } hipMallocAsync;
        struct
        {
            void**       dev_ptr;
            void*        dev_ptr__val;
            size_t       size;
            hipMemPool_t mem_pool;
            hipStream_t  stream;
        } hipMallocFromPoolAsync;
        struct
        {
            void** ptr;
            void*  ptr__val;
            size_t size;
        } hipMallocHost;
        struct
        {
            void**       dev_ptr;
            void*        dev_ptr__val;
            size_t       size;
            unsigned int flags;
        } hipMallocManaged;
        struct
        {
            hipMipmappedArray_t*        mipmappedArray;
            hipMipmappedArray_t         mipmappedArray__val;
            const hipChannelFormatDesc* desc;
            hipChannelFormatDesc        desc__val;
            hipExtent                   extent;
            unsigned int                numLevels;
            unsigned int                flags;
        } hipMallocMipmappedArray;
        struct
        {
            void**  ptr;
            void*   ptr__val;
            size_t* pitch;
            size_t  pitch__val;
            size_t  width;
            size_t  height;
        } hipMallocPitch;
        struct
        {
            void*  devPtr;
            size_t size;
        } hipMemAddressFree;
        struct
        {
            void**             ptr;
            void*              ptr__val;
            size_t             size;
            size_t             alignment;
            void*              addr;
            unsigned long long flags;
        } hipMemAddressReserve;
        struct
        {
            const void*     dev_ptr;
            size_t          count;
            hipMemoryAdvise advice;
            int             device;
        } hipMemAdvise;
        struct
        {
            void** ptr;
            void*  ptr__val;
            size_t size;
        } hipMemAllocHost;
        struct
        {
            hipDeviceptr_t* dptr;
            hipDeviceptr_t  dptr__val;
            size_t*         pitch;
            size_t          pitch__val;
            size_t          widthInBytes;
            size_t          height;
            unsigned int    elementSizeBytes;
        } hipMemAllocPitch;
        struct
        {
            hipMemGenericAllocationHandle_t* handle;
            hipMemGenericAllocationHandle_t  handle__val;
            size_t                           size;
            const hipMemAllocationProp*      prop;
            hipMemAllocationProp             prop__val;
            unsigned long long               flags;
        } hipMemCreate;
        struct
        {
            void*                           shareableHandle;
            hipMemGenericAllocationHandle_t handle;
            hipMemAllocationHandleType      handleType;
            unsigned long long              flags;
        } hipMemExportToShareableHandle;
        struct
        {
            unsigned long long*   flags;
            unsigned long long    flags__val;
            const hipMemLocation* location;
            hipMemLocation        location__val;
            void*                 ptr;
        } hipMemGetAccess;
        struct
        {
            hipDeviceptr_t* pbase;
            hipDeviceptr_t  pbase__val;
            size_t*         psize;
            size_t          psize__val;
            hipDeviceptr_t  dptr;
        } hipMemGetAddressRange;
        struct
        {
            size_t*                           granularity;
            size_t                            granularity__val;
            const hipMemAllocationProp*       prop;
            hipMemAllocationProp              prop__val;
            hipMemAllocationGranularity_flags option;
        } hipMemGetAllocationGranularity;
        struct
        {
            hipMemAllocationProp*           prop;
            hipMemAllocationProp            prop__val;
            hipMemGenericAllocationHandle_t handle;
        } hipMemGetAllocationPropertiesFromHandle;
        struct
        {
            size_t* free;
            size_t  free__val;
            size_t* total;
            size_t  total__val;
        } hipMemGetInfo;
        struct
        {
            hipMemGenericAllocationHandle_t* handle;
            hipMemGenericAllocationHandle_t  handle__val;
            void*                            osHandle;
            hipMemAllocationHandleType       shHandleType;
        } hipMemImportFromShareableHandle;
        struct
        {
            void*                           ptr;
            size_t                          size;
            size_t                          offset;
            hipMemGenericAllocationHandle_t handle;
            unsigned long long              flags;
        } hipMemMap;
        struct
        {
            hipArrayMapInfo* mapInfoList;
            hipArrayMapInfo  mapInfoList__val;
            unsigned int     count;
            hipStream_t      stream;
        } hipMemMapArrayAsync;
        struct
        {
            hipMemPool_t*          mem_pool;
            hipMemPool_t           mem_pool__val;
            const hipMemPoolProps* pool_props;
            hipMemPoolProps        pool_props__val;
        } hipMemPoolCreate;
        struct
        {
            hipMemPool_t mem_pool;
        } hipMemPoolDestroy;
        struct
        {
            hipMemPoolPtrExportData* export_data;
            hipMemPoolPtrExportData  export_data__val;
            void*                    dev_ptr;
        } hipMemPoolExportPointer;
        struct
        {
            void*                      shared_handle;
            hipMemPool_t               mem_pool;
            hipMemAllocationHandleType handle_type;
            unsigned int               flags;
        } hipMemPoolExportToShareableHandle;
        struct
        {
            hipMemAccessFlags* flags;
            hipMemAccessFlags  flags__val;
            hipMemPool_t       mem_pool;
            hipMemLocation*    location;
            hipMemLocation     location__val;
        } hipMemPoolGetAccess;
        struct
        {
            hipMemPool_t   mem_pool;
            hipMemPoolAttr attr;
            void*          value;
        } hipMemPoolGetAttribute;
        struct
        {
            hipMemPool_t*              mem_pool;
            hipMemPool_t               mem_pool__val;
            void*                      shared_handle;
            hipMemAllocationHandleType handle_type;
            unsigned int               flags;
        } hipMemPoolImportFromShareableHandle;
        struct
        {
            void**                   dev_ptr;
            void*                    dev_ptr__val;
            hipMemPool_t             mem_pool;
            hipMemPoolPtrExportData* export_data;
            hipMemPoolPtrExportData  export_data__val;
        } hipMemPoolImportPointer;
        struct
        {
            hipMemPool_t            mem_pool;
            const hipMemAccessDesc* desc_list;
            hipMemAccessDesc        desc_list__val;
            size_t                  count;
        } hipMemPoolSetAccess;
        struct
        {
            hipMemPool_t   mem_pool;
            hipMemPoolAttr attr;
            void*          value;
        } hipMemPoolSetAttribute;
        struct
        {
            hipMemPool_t mem_pool;
            size_t       min_bytes_to_hold;
        } hipMemPoolTrimTo;
        struct
        {
            const void* dev_ptr;
            size_t      count;
            int         device;
            hipStream_t stream;
        } hipMemPrefetchAsync;
        struct
        {
            void*   ptr;
            size_t* size;
            size_t  size__val;
        } hipMemPtrGetInfo;
        struct
        {
            void*                data;
            size_t               data_size;
            hipMemRangeAttribute attribute;
            const void*          dev_ptr;
            size_t               count;
        } hipMemRangeGetAttribute;
        struct
        {
            void**                data;
            void*                 data__val;
            size_t*               data_sizes;
            size_t                data_sizes__val;
            hipMemRangeAttribute* attributes;
            hipMemRangeAttribute  attributes__val;
            size_t                num_attributes;
            const void*           dev_ptr;
            size_t                count;
        } hipMemRangeGetAttributes;
        struct
        {
            hipMemGenericAllocationHandle_t handle;
        } hipMemRelease;
        struct
        {
            hipMemGenericAllocationHandle_t* handle;
            hipMemGenericAllocationHandle_t  handle__val;
            void*                            addr;
        } hipMemRetainAllocationHandle;
        struct
        {
            void*                   ptr;
            size_t                  size;
            const hipMemAccessDesc* desc;
            hipMemAccessDesc        desc__val;
            size_t                  count;
        } hipMemSetAccess;
        struct
        {
            void*  ptr;
            size_t size;
        } hipMemUnmap;
        struct
        {
            void*         dst;
            const void*   src;
            size_t        sizeBytes;
            hipMemcpyKind kind;
        } hipMemcpy;
        struct
        {
            void*         dst;
            size_t        dpitch;
            const void*   src;
            size_t        spitch;
            size_t        width;
            size_t        height;
            hipMemcpyKind kind;
        } hipMemcpy2D;
        struct
        {
            void*         dst;
            size_t        dpitch;
            const void*   src;
            size_t        spitch;
            size_t        width;
            size_t        height;
            hipMemcpyKind kind;
            hipStream_t   stream;
        } hipMemcpy2DAsync;
        struct
        {
            void*            dst;
            size_t           dpitch;
            hipArray_const_t src;
            size_t           wOffset;
            size_t           hOffset;
            size_t           width;
            size_t           height;
            hipMemcpyKind    kind;
        } hipMemcpy2DFromArray;
        struct
        {
            void*            dst;
            size_t           dpitch;
            hipArray_const_t src;
            size_t           wOffset;
            size_t           hOffset;
            size_t           width;
            size_t           height;
            hipMemcpyKind    kind;
            hipStream_t      stream;
        } hipMemcpy2DFromArrayAsync;
        struct
        {
            hipArray*     dst;
            hipArray      dst__val;
            size_t        wOffset;
            size_t        hOffset;
            const void*   src;
            size_t        spitch;
            size_t        width;
            size_t        height;
            hipMemcpyKind kind;
        } hipMemcpy2DToArray;
        struct
        {
            hipArray*     dst;
            hipArray      dst__val;
            size_t        wOffset;
            size_t        hOffset;
            const void*   src;
            size_t        spitch;
            size_t        width;
            size_t        height;
            hipMemcpyKind kind;
            hipStream_t   stream;
        } hipMemcpy2DToArrayAsync;
        struct
        {
            const hipMemcpy3DParms* p;
            hipMemcpy3DParms        p__val;
        } hipMemcpy3D;
        struct
        {
            const hipMemcpy3DParms* p;
            hipMemcpy3DParms        p__val;
            hipStream_t             stream;
        } hipMemcpy3DAsync;
        struct
        {
            void*         dst;
            const void*   src;
            size_t        sizeBytes;
            hipMemcpyKind kind;
            hipStream_t   stream;
        } hipMemcpyAsync;
        struct
        {
            void*     dst;
            hipArray* srcArray;
            hipArray  srcArray__val;
            size_t    srcOffset;
            size_t    count;
        } hipMemcpyAtoH;
        struct
        {
            hipDeviceptr_t dst;
            hipDeviceptr_t src;
            size_t         sizeBytes;
        } hipMemcpyDtoD;
        struct
        {
            hipDeviceptr_t dst;
            hipDeviceptr_t src;
            size_t         sizeBytes;
            hipStream_t    stream;
        } hipMemcpyDtoDAsync;
        struct
        {
            void*          dst;
            hipDeviceptr_t src;
            size_t         sizeBytes;
        } hipMemcpyDtoH;
        struct
        {
            void*          dst;
            hipDeviceptr_t src;
            size_t         sizeBytes;
            hipStream_t    stream;
        } hipMemcpyDtoHAsync;
        struct
        {
            void*            dst;
            hipArray_const_t srcArray;
            size_t           wOffset;
            size_t           hOffset;
            size_t           count;
            hipMemcpyKind    kind;
        } hipMemcpyFromArray;
        struct
        {
            void*         dst;
            const void*   symbol;
            size_t        sizeBytes;
            size_t        offset;
            hipMemcpyKind kind;
        } hipMemcpyFromSymbol;
        struct
        {
            void*         dst;
            const void*   symbol;
            size_t        sizeBytes;
            size_t        offset;
            hipMemcpyKind kind;
            hipStream_t   stream;
        } hipMemcpyFromSymbolAsync;
        struct
        {
            hipArray*   dstArray;
            hipArray    dstArray__val;
            size_t      dstOffset;
            const void* srcHost;
            size_t      count;
        } hipMemcpyHtoA;
        struct
        {
            hipDeviceptr_t dst;
            void*          src;
            size_t         sizeBytes;
        } hipMemcpyHtoD;
        struct
        {
            hipDeviceptr_t dst;
            void*          src;
            size_t         sizeBytes;
            hipStream_t    stream;
        } hipMemcpyHtoDAsync;
        struct
        {
            const hip_Memcpy2D* pCopy;
            hip_Memcpy2D        pCopy__val;
        } hipMemcpyParam2D;
        struct
        {
            const hip_Memcpy2D* pCopy;
            hip_Memcpy2D        pCopy__val;
            hipStream_t         stream;
        } hipMemcpyParam2DAsync;
        struct
        {
            void*       dst;
            int         dstDeviceId;
            const void* src;
            int         srcDeviceId;
            size_t      sizeBytes;
        } hipMemcpyPeer;
        struct
        {
            void*       dst;
            int         dstDeviceId;
            const void* src;
            int         srcDevice;
            size_t      sizeBytes;
            hipStream_t stream;
        } hipMemcpyPeerAsync;
        struct
        {
            hipArray*     dst;
            hipArray      dst__val;
            size_t        wOffset;
            size_t        hOffset;
            const void*   src;
            size_t        count;
            hipMemcpyKind kind;
        } hipMemcpyToArray;
        struct
        {
            const void*   symbol;
            const void*   src;
            size_t        sizeBytes;
            size_t        offset;
            hipMemcpyKind kind;
        } hipMemcpyToSymbol;
        struct
        {
            const void*   symbol;
            const void*   src;
            size_t        sizeBytes;
            size_t        offset;
            hipMemcpyKind kind;
            hipStream_t   stream;
        } hipMemcpyToSymbolAsync;
        struct
        {
            void*         dst;
            const void*   src;
            size_t        sizeBytes;
            hipMemcpyKind kind;
            hipStream_t   stream;
        } hipMemcpyWithStream;
        struct
        {
            void*  dst;
            int    value;
            size_t sizeBytes;
        } hipMemset;
        struct
        {
            void*  dst;
            size_t pitch;
            int    value;
            size_t width;
            size_t height;
        } hipMemset2D;
        struct
        {
            void*       dst;
            size_t      pitch;
            int         value;
            size_t      width;
            size_t      height;
            hipStream_t stream;
        } hipMemset2DAsync;
        struct
        {
            hipPitchedPtr pitchedDevPtr;
            int           value;
            hipExtent     extent;
        } hipMemset3D;
        struct
        {
            hipPitchedPtr pitchedDevPtr;
            int           value;
            hipExtent     extent;
            hipStream_t   stream;
        } hipMemset3DAsync;
        struct
        {
            void*       dst;
            int         value;
            size_t      sizeBytes;
            hipStream_t stream;
        } hipMemsetAsync;
        struct
        {
            hipDeviceptr_t dest;
            unsigned short value;
            size_t         count;
        } hipMemsetD16;
        struct
        {
            hipDeviceptr_t dest;
            unsigned short value;
            size_t         count;
            hipStream_t    stream;
        } hipMemsetD16Async;
        struct
        {
            hipDeviceptr_t dest;
            int            value;
            size_t         count;
        } hipMemsetD32;
        struct
        {
            hipDeviceptr_t dst;
            int            value;
            size_t         count;
            hipStream_t    stream;
        } hipMemsetD32Async;
        struct
        {
            hipDeviceptr_t dest;
            unsigned char  value;
            size_t         count;
        } hipMemsetD8;
        struct
        {
            hipDeviceptr_t dest;
            unsigned char  value;
            size_t         count;
            hipStream_t    stream;
        } hipMemsetD8Async;
        struct
        {
            hipMipmappedArray_t*    pHandle;
            hipMipmappedArray_t     pHandle__val;
            HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc;
            HIP_ARRAY3D_DESCRIPTOR  pMipmappedArrayDesc__val;
            unsigned int            numMipmapLevels;
        } hipMipmappedArrayCreate;
        struct
        {
            hipMipmappedArray_t hMipmappedArray;
        } hipMipmappedArrayDestroy;
        struct
        {
            hipArray_t*         pLevelArray;
            hipArray_t          pLevelArray__val;
            hipMipmappedArray_t hMipMappedArray;
            unsigned int        level;
        } hipMipmappedArrayGetLevel;
        struct
        {
            hipFunction_t* function;
            hipFunction_t  function__val;
            hipModule_t    module;
            const char*    kname;
            char           kname__val;
        } hipModuleGetFunction;
        struct
        {
            hipDeviceptr_t* dptr;
            hipDeviceptr_t  dptr__val;
            size_t*         bytes;
            size_t          bytes__val;
            hipModule_t     hmod;
            const char*     name;
            char            name__val;
        } hipModuleGetGlobal;
        struct
        {
            textureReference** texRef;
            textureReference*  texRef__val;
            hipModule_t        hmod;
            const char*        name;
            char               name__val;
        } hipModuleGetTexRef;
        struct
        {
            hipFunction_t f;
            unsigned int  gridDimX;
            unsigned int  gridDimY;
            unsigned int  gridDimZ;
            unsigned int  blockDimX;
            unsigned int  blockDimY;
            unsigned int  blockDimZ;
            unsigned int  sharedMemBytes;
            hipStream_t   stream;
            void**        kernelParams;
            void*         kernelParams__val;
        } hipModuleLaunchCooperativeKernel;
        struct
        {
            hipFunctionLaunchParams* launchParamsList;
            hipFunctionLaunchParams  launchParamsList__val;
            unsigned int             numDevices;
            unsigned int             flags;
        } hipModuleLaunchCooperativeKernelMultiDevice;
        struct
        {
            hipFunction_t f;
            unsigned int  gridDimX;
            unsigned int  gridDimY;
            unsigned int  gridDimZ;
            unsigned int  blockDimX;
            unsigned int  blockDimY;
            unsigned int  blockDimZ;
            unsigned int  sharedMemBytes;
            hipStream_t   stream;
            void**        kernelParams;
            void*         kernelParams__val;
            void**        extra;
            void*         extra__val;
        } hipModuleLaunchKernel;
        struct
        {
            hipModule_t* module;
            hipModule_t  module__val;
            const char*  fname;
            char         fname__val;
        } hipModuleLoad;
        struct
        {
            hipModule_t* module;
            hipModule_t  module__val;
            const void*  image;
        } hipModuleLoadData;
        struct
        {
            hipModule_t*  module;
            hipModule_t   module__val;
            const void*   image;
            unsigned int  numOptions;
            hipJitOption* options;
            hipJitOption  options__val;
            void**        optionsValues;
            void*         optionsValues__val;
        } hipModuleLoadDataEx;
        struct
        {
            int*          numBlocks;
            int           numBlocks__val;
            hipFunction_t f;
            int           blockSize;
            size_t        dynSharedMemPerBlk;
        } hipModuleOccupancyMaxActiveBlocksPerMultiprocessor;
        struct
        {
            int*          numBlocks;
            int           numBlocks__val;
            hipFunction_t f;
            int           blockSize;
            size_t        dynSharedMemPerBlk;
            unsigned int  flags;
        } hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
        struct
        {
            int*          gridSize;
            int           gridSize__val;
            int*          blockSize;
            int           blockSize__val;
            hipFunction_t f;
            size_t        dynSharedMemPerBlk;
            int           blockSizeLimit;
        } hipModuleOccupancyMaxPotentialBlockSize;
        struct
        {
            int*          gridSize;
            int           gridSize__val;
            int*          blockSize;
            int           blockSize__val;
            hipFunction_t f;
            size_t        dynSharedMemPerBlk;
            int           blockSizeLimit;
            unsigned int  flags;
        } hipModuleOccupancyMaxPotentialBlockSizeWithFlags;
        struct
        {
            hipModule_t module;
        } hipModuleUnload;
        struct
        {
            int*        numBlocks;
            int         numBlocks__val;
            const void* f;
            int         blockSize;
            size_t      dynamicSMemSize;
        } hipOccupancyMaxActiveBlocksPerMultiprocessor;
        struct
        {
            int*         numBlocks;
            int          numBlocks__val;
            const void*  f;
            int          blockSize;
            size_t       dynamicSMemSize;
            unsigned int flags;
        } hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
        struct
        {
            int*        gridSize;
            int         gridSize__val;
            int*        blockSize;
            int         blockSize__val;
            const void* f;
            size_t      dynSharedMemPerBlk;
            int         blockSizeLimit;
        } hipOccupancyMaxPotentialBlockSize;
        struct
        {
            void*                data;
            hipPointer_attribute attribute;
            hipDeviceptr_t       ptr;
        } hipPointerGetAttribute;
        struct
        {
            hipPointerAttribute_t* attributes;
            hipPointerAttribute_t  attributes__val;
            const void*            ptr;
        } hipPointerGetAttributes;
        struct
        {
            const void*          value;
            hipPointer_attribute attribute;
            hipDeviceptr_t       ptr;
        } hipPointerSetAttribute;
        struct
        {
            int* runtimeVersion;
            int  runtimeVersion__val;
        } hipRuntimeGetVersion;
        struct
        {
            int deviceId;
        } hipSetDevice;
        struct
        {
            unsigned int flags;
        } hipSetDeviceFlags;
        struct
        {
            const void* arg;
            size_t      size;
            size_t      offset;
        } hipSetupArgument;
        struct
        {
            const hipExternalSemaphore_t*           extSemArray;
            hipExternalSemaphore_t                  extSemArray__val;
            const hipExternalSemaphoreSignalParams* paramsArray;
            hipExternalSemaphoreSignalParams        paramsArray__val;
            unsigned int                            numExtSems;
            hipStream_t                             stream;
        } hipSignalExternalSemaphoresAsync;
        struct
        {
            hipStream_t         stream;
            hipStreamCallback_t callback;
            void*               userData;
            unsigned int        flags;
        } hipStreamAddCallback;
        struct
        {
            hipStream_t  stream;
            void*        dev_ptr;
            size_t       length;
            unsigned int flags;
        } hipStreamAttachMemAsync;
        struct
        {
            hipStream_t          stream;
            hipStreamCaptureMode mode;
        } hipStreamBeginCapture;
        struct
        {
            hipStream_t* stream;
            hipStream_t  stream__val;
        } hipStreamCreate;
        struct
        {
            hipStream_t* stream;
            hipStream_t  stream__val;
            unsigned int flags;
        } hipStreamCreateWithFlags;
        struct
        {
            hipStream_t* stream;
            hipStream_t  stream__val;
            unsigned int flags;
            int          priority;
        } hipStreamCreateWithPriority;
        struct
        {
            hipStream_t stream;
        } hipStreamDestroy;
        struct
        {
            hipStream_t stream;
            hipGraph_t* pGraph;
            hipGraph_t  pGraph__val;
        } hipStreamEndCapture;
        struct
        {
            hipStream_t             stream;
            hipStreamCaptureStatus* pCaptureStatus;
            hipStreamCaptureStatus  pCaptureStatus__val;
            unsigned long long*     pId;
            unsigned long long      pId__val;
        } hipStreamGetCaptureInfo;
        struct
        {
            hipStream_t             stream;
            hipStreamCaptureStatus* captureStatus_out;
            hipStreamCaptureStatus  captureStatus_out__val;
            unsigned long long*     id_out;
            unsigned long long      id_out__val;
            hipGraph_t*             graph_out;
            hipGraph_t              graph_out__val;
            const hipGraphNode_t**  dependencies_out;
            const hipGraphNode_t*   dependencies_out__val;
            size_t*                 numDependencies_out;
            size_t                  numDependencies_out__val;
        } hipStreamGetCaptureInfo_v2;
        struct
        {
            hipStream_t  stream;
            hipDevice_t* device;
            hipDevice_t  device__val;
        } hipStreamGetDevice;
        struct
        {
            hipStream_t   stream;
            unsigned int* flags;
            unsigned int  flags__val;
        } hipStreamGetFlags;
        struct
        {
            hipStream_t stream;
            int*        priority;
            int         priority__val;
        } hipStreamGetPriority;
        struct
        {
            hipStream_t             stream;
            hipStreamCaptureStatus* pCaptureStatus;
            hipStreamCaptureStatus  pCaptureStatus__val;
        } hipStreamIsCapturing;
        struct
        {
            hipStream_t stream;
        } hipStreamQuery;
        struct
        {
            hipStream_t stream;
        } hipStreamSynchronize;
        struct
        {
            hipStream_t     stream;
            hipGraphNode_t* dependencies;
            hipGraphNode_t  dependencies__val;
            size_t          numDependencies;
            unsigned int    flags;
        } hipStreamUpdateCaptureDependencies;
        struct
        {
            hipStream_t  stream;
            hipEvent_t   event;
            unsigned int flags;
        } hipStreamWaitEvent;
        struct
        {
            hipStream_t  stream;
            void*        ptr;
            unsigned int value;
            unsigned int flags;
            unsigned int mask;
        } hipStreamWaitValue32;
        struct
        {
            hipStream_t  stream;
            void*        ptr;
            uint64_t     value;
            unsigned int flags;
            uint64_t     mask;
        } hipStreamWaitValue64;
        struct
        {
            hipStream_t  stream;
            void*        ptr;
            unsigned int value;
            unsigned int flags;
        } hipStreamWriteValue32;
        struct
        {
            hipStream_t  stream;
            void*        ptr;
            uint64_t     value;
            unsigned int flags;
        } hipStreamWriteValue64;
        struct
        {
            hipDeviceptr_t*         dev_ptr;
            hipDeviceptr_t          dev_ptr__val;
            const textureReference* texRef;
            textureReference        texRef__val;
        } hipTexRefGetAddress;
        struct
        {
            unsigned int*           pFlags;
            unsigned int            pFlags__val;
            const textureReference* texRef;
            textureReference        texRef__val;
        } hipTexRefGetFlags;
        struct
        {
            hipArray_Format*        pFormat;
            hipArray_Format         pFormat__val;
            int*                    pNumChannels;
            int                     pNumChannels__val;
            const textureReference* texRef;
            textureReference        texRef__val;
        } hipTexRefGetFormat;
        struct
        {
            int*                    pmaxAnsio;
            int                     pmaxAnsio__val;
            const textureReference* texRef;
            textureReference        texRef__val;
        } hipTexRefGetMaxAnisotropy;
        struct
        {
            hipMipmappedArray_t*    pArray;
            hipMipmappedArray_t     pArray__val;
            const textureReference* texRef;
            textureReference        texRef__val;
        } hipTexRefGetMipMappedArray;
        struct
        {
            float*                  pbias;
            float                   pbias__val;
            const textureReference* texRef;
            textureReference        texRef__val;
        } hipTexRefGetMipmapLevelBias;
        struct
        {
            float*                  pminMipmapLevelClamp;
            float                   pminMipmapLevelClamp__val;
            float*                  pmaxMipmapLevelClamp;
            float                   pmaxMipmapLevelClamp__val;
            const textureReference* texRef;
            textureReference        texRef__val;
        } hipTexRefGetMipmapLevelClamp;
        struct
        {
            size_t*           ByteOffset;
            size_t            ByteOffset__val;
            textureReference* texRef;
            textureReference  texRef__val;
            hipDeviceptr_t    dptr;
            size_t            bytes;
        } hipTexRefSetAddress;
        struct
        {
            textureReference*           texRef;
            textureReference            texRef__val;
            const HIP_ARRAY_DESCRIPTOR* desc;
            HIP_ARRAY_DESCRIPTOR        desc__val;
            hipDeviceptr_t              dptr;
            size_t                      Pitch;
        } hipTexRefSetAddress2D;
        struct
        {
            textureReference* tex;
            textureReference  tex__val;
            hipArray_const_t  array;
            unsigned int      flags;
        } hipTexRefSetArray;
        struct
        {
            textureReference* texRef;
            textureReference  texRef__val;
            float*            pBorderColor;
            float             pBorderColor__val;
        } hipTexRefSetBorderColor;
        struct
        {
            textureReference* texRef;
            textureReference  texRef__val;
            unsigned int      Flags;
        } hipTexRefSetFlags;
        struct
        {
            textureReference* texRef;
            textureReference  texRef__val;
            hipArray_Format   fmt;
            int               NumPackedComponents;
        } hipTexRefSetFormat;
        struct
        {
            textureReference* texRef;
            textureReference  texRef__val;
            unsigned int      maxAniso;
        } hipTexRefSetMaxAnisotropy;
        struct
        {
            textureReference* texRef;
            textureReference  texRef__val;
            float             bias;
        } hipTexRefSetMipmapLevelBias;
        struct
        {
            textureReference* texRef;
            textureReference  texRef__val;
            float             minMipMapLevelClamp;
            float             maxMipMapLevelClamp;
        } hipTexRefSetMipmapLevelClamp;
        struct
        {
            textureReference*  texRef;
            textureReference   texRef__val;
            hipMipmappedArray* mipmappedArray;
            hipMipmappedArray  mipmappedArray__val;
            unsigned int       Flags;
        } hipTexRefSetMipmappedArray;
        struct
        {
            hipStreamCaptureMode* mode;
            hipStreamCaptureMode  mode__val;
        } hipThreadExchangeStreamCaptureMode;
        struct
        {
            hipUserObject_t* object_out;
            hipUserObject_t  object_out__val;
            void*            ptr;
            hipHostFn_t      destroy;
            unsigned int     initialRefcount;
            unsigned int     flags;
        } hipUserObjectCreate;
        struct
        {
            hipUserObject_t object;
            unsigned int    count;
        } hipUserObjectRelease;
        struct
        {
            hipUserObject_t object;
            unsigned int    count;
        } hipUserObjectRetain;
        struct
        {
            const hipExternalSemaphore_t*         extSemArray;
            hipExternalSemaphore_t                extSemArray__val;
            const hipExternalSemaphoreWaitParams* paramsArray;
            hipExternalSemaphoreWaitParams        paramsArray__val;
            unsigned int                          numExtSems;
            hipStream_t                           stream;
        } hipWaitExternalSemaphoresAsync;
    } args;
    uint64_t* phase_data;
} hip_api_data_t;

// helper macros ensuring C and C++ structs adhere to specific naming convention
#define ROCP_PUBLIC_CONFIG(TYPE)  ::rocprofiler_##TYPE
#define ROCP_PRIVATE_CONFIG(TYPE) ::rocprofiler::internal::TYPE

// Below asserts at compile time that the external C object has the same size as internal
// C++ object, e.g.,
//      sizeof(rocprofiler_domain_config) == sizeof(rocprofiler::internal::domain_config)
#define ROCP_ASSERT_CONFIG_ABI(TYPE)                                                               \
    static_assert(sizeof(ROCP_PUBLIC_CONFIG(TYPE)) == sizeof(ROCP_PRIVATE_CONFIG(TYPE)),           \
                  "Error! rocprofiler_" #TYPE " ABI error");

// Below asserts at compile time that the external C struct members has the same offset as
// internal C++ struct members
#define ROCP_ASSERT_CONFIG_OFFSET_ABI(TYPE, PUB_FIELD, PRIV_FIELD)                                 \
    static_assert(offsetof(ROCP_PUBLIC_CONFIG(TYPE), PUB_FIELD) ==                                 \
                      offsetof(ROCP_PRIVATE_CONFIG(TYPE), PRIV_FIELD),                             \
                  "Error! rocprofiler_" #TYPE "." #PUB_FIELD " ABI offset error");                 \
    static_assert(sizeof(ROCP_PUBLIC_CONFIG(TYPE)::PUB_FIELD) ==                                   \
                      sizeof(ROCP_PRIVATE_CONFIG(TYPE)::PRIV_FIELD),                               \
                  "Error! rocprofiler_" #TYPE "." #PUB_FIELD " ABI size error");

// this defines a template specialization for ensuring that the reinterpret_cast is only
// applied between public C structs and private C++ structs which are compatible.
#define ROCP_DEFINE_API_CAST_IMPL(INPUT_TYPE, OUTPUT_TYPE)                                         \
    namespace traits                                                                               \
    {                                                                                              \
    template <>                                                                                    \
    struct api_cast<INPUT_TYPE>                                                                    \
    {                                                                                              \
        using input_type  = INPUT_TYPE;                                                            \
        using output_type = OUTPUT_TYPE;                                                           \
                                                                                                   \
        output_type* operator()(input_type* _v) const                                              \
        {                                                                                          \
            return reinterpret_cast<output_type*>(_v);                                             \
        }                                                                                          \
                                                                                                   \
        const output_type* operator()(const input_type* _v) const                                  \
        {                                                                                          \
            return reinterpret_cast<const output_type*>(_v);                                       \
        }                                                                                          \
    };                                                                                             \
    }

// define C -> C++ and C++ -> C casting rules
#define ROCP_DEFINE_API_CAST_D(TYPE)                                                               \
    ROCP_DEFINE_API_CAST_IMPL(ROCP_PUBLIC_CONFIG(TYPE), ROCP_PRIVATE_CONFIG(TYPE))                 \
    ROCP_DEFINE_API_CAST_IMPL(ROCP_PRIVATE_CONFIG(TYPE), ROCP_PUBLIC_CONFIG(TYPE))

// use only when C++ struct is just an alias for C struct
#define ROCP_DEFINE_API_CAST_S(TYPE)                                                               \
    ROCP_DEFINE_API_CAST_IMPL(ROCP_PUBLIC_CONFIG(TYPE), ROCP_PRIVATE_CONFIG(TYPE))

namespace
{
namespace traits
{
// left undefined to ensure template specialization
template <typename PublicT>
struct api_cast;

// ensure api_cast<decltype(a)> where decltype(a) is const Tp equates to api_cast<Tp>
template <typename PublicT>
struct api_cast<const PublicT> : api_cast<PublicT>
{};

// ensure api_cast<decltype(a)> where decltype(a) is Tp& equates to api_cast<Tp>
template <typename PublicT>
struct api_cast<PublicT&> : api_cast<PublicT>
{};

// ensure api_cast<decltype(a)> where decltype(a) is Tp* equates to api_cast<Tp>
template <typename PublicT>
struct api_cast<PublicT*> : api_cast<PublicT>
{};
}  // namespace traits

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
//
//                          SEE BELOW! VERY IMPORTANT!
//
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
//
//  EVERY NEW CONFIG AND ALL OF ITS MEMBER FIELDS NEED TO HAVE THESE COMPILE TIME CHECKS!
//
//  these checks verify the two structs have the same size and that each
//  member field has the same size and offset into the struct
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

ROCP_ASSERT_CONFIG_ABI(config)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, size, size)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, compat_version, compat_version)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, api_version, api_version)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, reserved0, session_idx)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, user_data, user_data)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, buffer, buffer)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, domain, domain)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, filter, filter)

ROCP_ASSERT_CONFIG_ABI(domain_config)
ROCP_ASSERT_CONFIG_OFFSET_ABI(domain_config, callback, user_sync_callback)
ROCP_ASSERT_CONFIG_OFFSET_ABI(domain_config, reserved0, domains)
ROCP_ASSERT_CONFIG_OFFSET_ABI(domain_config, reserved1, opcodes)

ROCP_ASSERT_CONFIG_ABI(buffer_config)
ROCP_ASSERT_CONFIG_OFFSET_ABI(buffer_config, callback, callback)
ROCP_ASSERT_CONFIG_OFFSET_ABI(buffer_config, buffer_size, buffer_size)
// ROCP_ASSERT_CONFIG_OFFSET_ABI(buffer_config, reserved0, buffer)
ROCP_ASSERT_CONFIG_OFFSET_ABI(buffer_config, reserved1, buffer_idx)

ROCP_DEFINE_API_CAST_D(config)
ROCP_DEFINE_API_CAST_D(domain_config)
ROCP_DEFINE_API_CAST_D(buffer_config)
ROCP_DEFINE_API_CAST_S(filter_config)

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
//
//                          SEE ABOVE! VERY IMPORTANT!
//
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

/// use this to ensure that reinterpret_cast from public C struct to internal C++ struct
/// is valid, e.g. guard against accidentally casting to wrong type
template <typename Tp>
auto
rocp_cast(Tp* _val)
{
    return traits::api_cast<Tp>{}(_val);
}

/// helper function for making copies of the fields in rocprofiler_config. If the config
/// field needs to be copied in some special way, use a template specialization of the
/// "construct" function in the allocator to handle this, e.g.:
///
///     using special_config = ::rocprofiler::internal::special_config;
///
///     template <>
///     void
///     allocator<special_config, 8>::construct(special_config* const _p,
///                                             const special_config& _v) const
///     {
///         auto _tmp = special_config{};
///         // ... special copy of fields from _v into _tmp
///
///         // placement new of _tmp into _p
///         _p = new(_p) special_config{ _tmp };
///     }
///
///     template <>
///     void
///     allocator<special_config, 8>::construct(special_config* const _p,
///                                             special_config&& _v) const
///     {
///         auto _tmp = std::move(_v);
///         // ... perform special needs
///
///         // placement new of _tmp into _p
///         _p = new(_p) special_config{ std::move(_tmp) };
///     }
///
template <typename Tp, typename Up>
Tp*&
copy_config_field(Tp*& _dst, Up* _src_v)
{
    static auto _allocator = allocator<Tp>{};

    if constexpr(!std::is_same<Tp, Up>::value)
    {
        using PrivateT = typename traits::api_cast<Up>::output_type;
        static_assert(std::is_same<PrivateT, Tp>::value, "Error incorrect field copy");

        auto _src = rocp_cast(_src_v);
        if(_src)
        {
            _dst = _allocator.allocate(1);
            _allocator.construct(_dst, *_src);
        }
        return _dst;
    }
    else
    {
        if(_src_v)
        {
            _dst = _allocator.allocate(1);
            _allocator.construct(_dst, *_src_v);
        }
        return _dst;
    }
}

auto&
get_configs_buffer()
{
    static char
        _v[::rocprofiler::internal::max_configs_count * sizeof(rocprofiler::internal::config)];
    return _v;
}

auto&
get_configs_mutex()
{
    static auto _v = std::mutex{};
    return _v;
}

inline uint32_t
get_tid()
{
    return syscall(__NR_gettid);
}

constexpr auto rocp_max_configs = ::rocprofiler::internal::max_configs_count;
}  // namespace

namespace rocprofiler
{
namespace internal
{
std::array<rocprofiler::internal::config*, max_configs_count>&
get_registered_configs()
{
    static auto _v = std::array<rocprofiler::internal::config*, max_configs_count>{};
    return _v;
}

std::array<std::atomic<rocprofiler::internal::config*>, max_configs_count>&
get_active_configs()
{
    static auto _v = std::array<std::atomic<rocprofiler::internal::config*>, max_configs_count>{};
    return _v;
}
}  // namespace internal
}  // namespace rocprofiler

extern "C" {

rocprofiler_status_t
rocprofiler_allocate_config(rocprofiler_config* _inp_cfg)
{
    // perform checks that rocprofiler can be activated

    ::memset(_inp_cfg, 0, sizeof(rocprofiler_config));

    auto* _cfg = rocp_cast(_inp_cfg);

    _cfg->size           = sizeof(::rocprofiler_config);
    _cfg->compat_version = 0;
    _cfg->api_version    = ROCPROFILER_API_VERSION_ID;
    _cfg->session_idx    = std::numeric_limits<decltype(_cfg->session_idx)>::max();

    // initial value checks
    assert(_cfg->size == sizeof(rocprofiler::internal::config));
    assert(_cfg->compat_version == 0);
    assert(_cfg->api_version == ROCPROFILER_API_VERSION_ID);
    assert(_cfg->buffer == nullptr);
    assert(_cfg->domain == nullptr);
    assert(_cfg->filter == nullptr);
    assert(_cfg->session_idx ==
           std::numeric_limits<decltype(rocprofiler::internal::config::session_idx)>::max());

    // ... allocate any internal space needed to handle another config ...
    {
        auto _lk = std::unique_lock<std::mutex>{get_configs_mutex()};
        // ...
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_validate_config(const rocprofiler_config* cfg_v)
{
    const auto* cfg = rocp_cast(cfg_v);

    if(cfg->buffer == nullptr) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;

    if(cfg->filter == nullptr) return ROCPROFILER_STATUS_ERROR_FILTER_NOT_FOUND;

    if(cfg->domain == nullptr || cfg->domain->domains == 0)
        return ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN;

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_start_config(rocprofiler_config* cfg_v, rocprofiler_session_id_t* session_id)
{
    if(rocprofiler_validate_config(cfg_v) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "rocprofiler_start_config() provided an invalid configuration. tool "
                     "should use rocprofiler_validate_config() to check whether the "
                     "config is valid and adapt accordingly to issues before trying to "
                     "start the configuration."
                  << std::endl;
        abort();
    }

    auto* cfg = rocp_cast(cfg_v);

    uint64_t idx = rocp_max_configs;
    {
        auto _lk = std::unique_lock<std::mutex>{get_configs_mutex()};
        for(size_t i = 0; i < rocp_max_configs; ++i)
        {
            if(rocprofiler::internal::get_registered_configs().at(i) == nullptr)
            {
                idx = i;
                break;
            }
        }
    }

    // too many configs already registered
    if(idx == rocp_max_configs) return ROCPROFILER_STATUS_ERROR_SESSION_NOT_ACTIVE;

    cfg->session_idx   = idx;
    session_id->handle = idx;

    // using the session id, compute the location in the buffer of configs
    auto* _offset = get_configs_buffer() + (idx * sizeof(rocprofiler::internal::config));

    // placement new into the buffer
    auto* _copy_cfg = new(_offset) rocprofiler::internal::config{*cfg};

    // make copies of non-null config fields
    copy_config_field(_copy_cfg->buffer, cfg->buffer);
    copy_config_field(_copy_cfg->domain, cfg->domain);
    copy_config_field(_copy_cfg->filter, cfg->filter);

    // store until "deallocation"
    rocprofiler::internal::get_registered_configs().at(idx) = _copy_cfg;

    using config_t = rocprofiler::internal::config;
    // atomic swap the pointer into the "active" array used internally
    config_t* _expected = nullptr;
    bool      success = rocprofiler::internal::get_active_configs().at(idx).compare_exchange_strong(
        _expected, rocprofiler::internal::get_registered_configs().at(idx));

    if(!success) return ROCPROFILER_STATUS_ERROR_HAS_ACTIVE_SESSION;  // need relevant enum

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_stop_config(rocprofiler_session_id_t idx)
{
    // atomically assign the config pointer to NULL so that it is skipped in future
    // callbacks
    auto* _expected =
        rocprofiler::internal::get_active_configs().at(idx.handle).load(std::memory_order_relaxed);
    bool success = rocprofiler::internal::get_active_configs()
                       .at(idx.handle)
                       .compare_exchange_strong(_expected, nullptr);

    if(!success)
        return ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND;  // compare exchange strong
                                                            // failed

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_domain_add_domain(struct rocprofiler_domain_config*    _inp_cfg,
                              rocprofiler_tracer_activity_domain_t _domain)
{
    auto* _cfg = rocp_cast(_inp_cfg);
    if(_domain <= ACTIVITY_DOMAIN_NONE || _domain >= ACTIVITY_DOMAIN_NUMBER)
        return ROCPROFILER_STATUS_ERROR_INVALID_DOMAIN_ID;

    _cfg->domains |= (1 << _domain);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_domain_add_domains(struct rocprofiler_domain_config*     _inp_cfg,
                               rocprofiler_tracer_activity_domain_t* _domains,
                               size_t                                _ndomains)
{
    for(size_t i = 0; i < _ndomains; ++i)
    {
        auto _status = rocprofiler_domain_add_domain(_inp_cfg, _domains[i]);
        if(_status != ROCPROFILER_STATUS_SUCCESS) return _status;
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_domain_add_op(struct rocprofiler_domain_config*    _inp_cfg,
                          rocprofiler_tracer_activity_domain_t _domain,
                          uint32_t                             _op)
{
    auto* _cfg = rocp_cast(_inp_cfg);
    if(_domain <= ACTIVITY_DOMAIN_NONE || _domain >= ACTIVITY_DOMAIN_NUMBER)
        return ROCPROFILER_STATUS_ERROR_INVALID_DOMAIN_ID;

    if(_op >= get_domain_max_op(_domain)) return ROCPROFILER_STATUS_ERROR_INVALID_OPERATION_ID;

    auto _offset = (_domain * rocprofiler::internal::domain_ops_offset);
    _cfg->opcodes.set(_offset + _op, true);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_domain_add_ops(struct rocprofiler_domain_config*    _inp_cfg,
                           rocprofiler_tracer_activity_domain_t _domain,
                           uint32_t*                            _ops,
                           size_t                               _nops)
{
    for(size_t i = 0; i < _nops; ++i)
    {
        auto _status = rocprofiler_domain_add_op(_inp_cfg, _domain, _ops[i]);
        if(_status != ROCPROFILER_STATUS_SUCCESS) return _status;
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

// ------------------------------------------------------------------------------------ //
//
//                  demo of internal implementation
//
// ------------------------------------------------------------------------------------ //

void
api_callback(rocprofiler_tracer_activity_domain_t domain,
             uint32_t                             cid,
             const void*                          callback_data,
             void*)
{
    for(const auto& aitr : rocprofiler::internal::get_active_configs())
    {
        auto* itr = aitr.load();
        if(!itr) continue;

        // below should be valid so this might need to raise error
        if(!itr->domain) continue;

        // if the given domain + op is not enabled, skip this config
        if(!(*itr->domain)(domain, cid)) continue;

        if(itr->filter)
        {
            if(domain == ACTIVITY_DOMAIN_ROCTX)
            {}
            else if(domain == ACTIVITY_DOMAIN_HSA_API)
            {
                if(itr->filter->hsa_function_id && itr->filter->hsa_function_id(cid) == 0) continue;
            }
            else if(domain == ACTIVITY_DOMAIN_HIP_API)
            {
                if(itr->filter->hip_function_id && itr->filter->hip_function_id(cid) == 0) continue;
            }
        }

        auto& _domain      = (*itr->domain);
        auto& _correlation = (*itr->correlation_id);

        auto _correlation_id = rocprofiler::internal::correlation_config::get_unique_record_id();
        if(_correlation.external_id_callback)
            _correlation.external_id =
                _correlation.external_id_callback(domain, cid, _correlation_id);

        auto timestamp_ns = []() -> uint64_t {
            return std::chrono::steady_clock::now().time_since_epoch().count();
        };

        auto _header        = rocprofiler_record_header_t{ROCPROFILER_TRACER_RECORD,
                                                   rocprofiler_record_id_t{_correlation_id}};
        auto _op_id         = rocprofiler_tracer_operation_id_t{cid};
        auto _agent_id      = rocprofiler_agent_id_t{0};
        auto _queue_id      = rocprofiler_queue_id_t{0};
        auto _thread_id     = rocprofiler_thread_id_t{get_tid()};
        auto _session       = rocprofiler_session_id_t{itr->session_idx};
        auto _timestamp_raw = rocprofiler_timestamp_t{timestamp_ns()};
        auto _timestamp     = rocprofiler_record_header_timestamp_t{_timestamp_raw, _timestamp_raw};

        if(domain == ACTIVITY_DOMAIN_ROCTX)
        {
            auto                    _api_data = rocprofiler_tracer_api_data_t{};
            const roctx_api_data_t* _data =
                reinterpret_cast<const roctx_api_data_t*>(callback_data);

            if(itr->filter && itr->filter->name && itr->filter->name(_data->args.message) == 0)
                continue;

            _api_data.roctx = _data;

            auto _phase = rocprofiler_api_tracing_phase_t{ROCPROFILER_PHASE_ENTER};
            _timestamp  = {_timestamp_raw, _timestamp_raw};

            auto _external_cid = rocprofiler_tracer_external_id_t{_data ? _data->args.id : 0};
            auto _activity_cid = rocprofiler_tracer_activity_correlation_id_t{0};
            const char* _name  = _data->args.message;

            _domain.user_sync_callback(rocprofiler_record_tracer_t{_header,
                                                                   _external_cid,
                                                                   ACTIVITY_DOMAIN_ROCTX,
                                                                   _op_id,
                                                                   _api_data,
                                                                   _activity_cid,
                                                                   _timestamp,
                                                                   _agent_id,
                                                                   _queue_id,
                                                                   _thread_id,
                                                                   _phase,
                                                                   _name},
                                       _session);
        }
        else if(domain == ACTIVITY_DOMAIN_HSA_API)
        {
            auto                  _api_data = rocprofiler_tracer_api_data_t{};
            const hsa_api_data_t* _data = reinterpret_cast<const hsa_api_data_t*>(callback_data);
            _api_data.hsa               = _data;

            auto _phase = rocprofiler_api_tracing_phase_t{(_data->phase == ACTIVITY_API_PHASE_ENTER)
                                                              ? ROCPROFILER_PHASE_ENTER
                                                              : ROCPROFILER_PHASE_EXIT};

            if(_phase == ROCPROFILER_PHASE_ENTER)
                _timestamp.begin = _timestamp_raw;
            else
                _timestamp.end = _timestamp_raw;

            auto _external_cid = rocprofiler_tracer_external_id_t{0};
            auto _activity_cid =
                rocprofiler_tracer_activity_correlation_id_t{_data->correlation_id};
            const char* _name = nullptr;

            _domain.user_sync_callback(rocprofiler_record_tracer_t{_header,
                                                                   _external_cid,
                                                                   ACTIVITY_DOMAIN_HSA_API,
                                                                   _op_id,
                                                                   _api_data,
                                                                   _activity_cid,
                                                                   _timestamp,
                                                                   _agent_id,
                                                                   _queue_id,
                                                                   _thread_id,
                                                                   _phase,
                                                                   _name},
                                       _session);
        }
        else if(domain == ACTIVITY_DOMAIN_HIP_API)
        {
            auto                  _api_data = rocprofiler_tracer_api_data_t{};
            const hip_api_data_t* _data = reinterpret_cast<const hip_api_data_t*>(callback_data);
            _api_data.hip               = _data;

            auto _phase = rocprofiler_api_tracing_phase_t{(_data->phase == ACTIVITY_API_PHASE_ENTER)
                                                              ? ROCPROFILER_PHASE_ENTER
                                                              : ROCPROFILER_PHASE_EXIT};

            if(_phase == ROCPROFILER_PHASE_ENTER)
                _timestamp.begin = _timestamp_raw;
            else
                _timestamp.end = _timestamp_raw;

            auto _external_cid = rocprofiler_tracer_external_id_t{0};
            auto _activity_cid =
                rocprofiler_tracer_activity_correlation_id_t{_data->correlation_id};
            const char* _name = nullptr;

            _domain.user_sync_callback(rocprofiler_record_tracer_t{_header,
                                                                   _external_cid,
                                                                   ACTIVITY_DOMAIN_HIP_API,
                                                                   _op_id,
                                                                   _api_data,
                                                                   _activity_cid,
                                                                   _timestamp,
                                                                   _agent_id,
                                                                   _queue_id,
                                                                   _thread_id,
                                                                   _phase,
                                                                   _name},
                                       _session);
        }
    }
}

void
InitRoctracer()
{
    for(const auto& itr : rocprofiler::internal::get_registered_configs())
    {
        if(!itr) continue;

        // below should be valid so this might need to raise error
        if(!itr->domain) continue;

        for(auto ditr : {ACTIVITY_DOMAIN_ROCTX, ACTIVITY_DOMAIN_HSA_API, ACTIVITY_DOMAIN_HIP_API})
        {
            if((*itr->domain)(ditr))
            {
                if(itr->domain->user_sync_callback)
                {
                    // ...
                }
                else
                {
                    // ...
                }
            }
        }

        for(auto ditr : {ACTIVITY_DOMAIN_HSA_OPS, ACTIVITY_DOMAIN_HIP_OPS})
        {
            if((*itr->domain)(ditr))
            {
                if(itr->domain->opcodes.none())
                {
                    // ...
                }
                else
                {
                    for(size_t i = 0; i < itr->domain->opcodes.size(); ++i)
                    {
                        if((*itr->domain)(ditr, i))
                        {
                            // ...
                        }
                    }
                }
            }
        }
    }
}
}
