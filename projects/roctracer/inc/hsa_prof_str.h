// automatically generated

/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/////////////////////////////////////////////////////////////////////////////
//
// HSA API tracing primitives
//
// 'CoreApiTable', header 'hsa.h', 125 funcs
// 'AmdExtTable', header 'hsa_ext_amd.h', 40 funcs
// 'ImageExtTable', header 'hsa_ext_image.h', 13 funcs
// 'AmdExtTable', header 'hsa_api_trace.h', 40 funcs
//
/////////////////////////////////////////////////////////////////////////////

#ifndef INC_HSA_PROF_STR_H_
#define INC_HSA_PROF_STR_H_

// section: API ID enumeration

enum hsa_api_id_t {
  // block: CoreApiTable API
  HSA_API_ID_hsa_init = 0,
  HSA_API_ID_hsa_shut_down = 1,
  HSA_API_ID_hsa_system_get_info = 2,
  HSA_API_ID_hsa_system_extension_supported = 3,
  HSA_API_ID_hsa_system_get_extension_table = 4,
  HSA_API_ID_hsa_iterate_agents = 5,
  HSA_API_ID_hsa_agent_get_info = 6,
  HSA_API_ID_hsa_queue_create = 7,
  HSA_API_ID_hsa_soft_queue_create = 8,
  HSA_API_ID_hsa_queue_destroy = 9,
  HSA_API_ID_hsa_queue_inactivate = 10,
  HSA_API_ID_hsa_queue_load_read_index_scacquire = 11,
  HSA_API_ID_hsa_queue_load_read_index_relaxed = 12,
  HSA_API_ID_hsa_queue_load_write_index_scacquire = 13,
  HSA_API_ID_hsa_queue_load_write_index_relaxed = 14,
  HSA_API_ID_hsa_queue_store_write_index_relaxed = 15,
  HSA_API_ID_hsa_queue_store_write_index_screlease = 16,
  HSA_API_ID_hsa_queue_cas_write_index_scacq_screl = 17,
  HSA_API_ID_hsa_queue_cas_write_index_scacquire = 18,
  HSA_API_ID_hsa_queue_cas_write_index_relaxed = 19,
  HSA_API_ID_hsa_queue_cas_write_index_screlease = 20,
  HSA_API_ID_hsa_queue_add_write_index_scacq_screl = 21,
  HSA_API_ID_hsa_queue_add_write_index_scacquire = 22,
  HSA_API_ID_hsa_queue_add_write_index_relaxed = 23,
  HSA_API_ID_hsa_queue_add_write_index_screlease = 24,
  HSA_API_ID_hsa_queue_store_read_index_relaxed = 25,
  HSA_API_ID_hsa_queue_store_read_index_screlease = 26,
  HSA_API_ID_hsa_agent_iterate_regions = 27,
  HSA_API_ID_hsa_region_get_info = 28,
  HSA_API_ID_hsa_agent_get_exception_policies = 29,
  HSA_API_ID_hsa_agent_extension_supported = 30,
  HSA_API_ID_hsa_memory_register = 31,
  HSA_API_ID_hsa_memory_deregister = 32,
  HSA_API_ID_hsa_memory_allocate = 33,
  HSA_API_ID_hsa_memory_free = 34,
  HSA_API_ID_hsa_memory_copy = 35,
  HSA_API_ID_hsa_memory_assign_agent = 36,
  HSA_API_ID_hsa_signal_create = 37,
  HSA_API_ID_hsa_signal_destroy = 38,
  HSA_API_ID_hsa_signal_load_relaxed = 39,
  HSA_API_ID_hsa_signal_load_scacquire = 40,
  HSA_API_ID_hsa_signal_store_relaxed = 41,
  HSA_API_ID_hsa_signal_store_screlease = 42,
  HSA_API_ID_hsa_signal_wait_relaxed = 43,
  HSA_API_ID_hsa_signal_wait_scacquire = 44,
  HSA_API_ID_hsa_signal_and_relaxed = 45,
  HSA_API_ID_hsa_signal_and_scacquire = 46,
  HSA_API_ID_hsa_signal_and_screlease = 47,
  HSA_API_ID_hsa_signal_and_scacq_screl = 48,
  HSA_API_ID_hsa_signal_or_relaxed = 49,
  HSA_API_ID_hsa_signal_or_scacquire = 50,
  HSA_API_ID_hsa_signal_or_screlease = 51,
  HSA_API_ID_hsa_signal_or_scacq_screl = 52,
  HSA_API_ID_hsa_signal_xor_relaxed = 53,
  HSA_API_ID_hsa_signal_xor_scacquire = 54,
  HSA_API_ID_hsa_signal_xor_screlease = 55,
  HSA_API_ID_hsa_signal_xor_scacq_screl = 56,
  HSA_API_ID_hsa_signal_exchange_relaxed = 57,
  HSA_API_ID_hsa_signal_exchange_scacquire = 58,
  HSA_API_ID_hsa_signal_exchange_screlease = 59,
  HSA_API_ID_hsa_signal_exchange_scacq_screl = 60,
  HSA_API_ID_hsa_signal_add_relaxed = 61,
  HSA_API_ID_hsa_signal_add_scacquire = 62,
  HSA_API_ID_hsa_signal_add_screlease = 63,
  HSA_API_ID_hsa_signal_add_scacq_screl = 64,
  HSA_API_ID_hsa_signal_subtract_relaxed = 65,
  HSA_API_ID_hsa_signal_subtract_scacquire = 66,
  HSA_API_ID_hsa_signal_subtract_screlease = 67,
  HSA_API_ID_hsa_signal_subtract_scacq_screl = 68,
  HSA_API_ID_hsa_signal_cas_relaxed = 69,
  HSA_API_ID_hsa_signal_cas_scacquire = 70,
  HSA_API_ID_hsa_signal_cas_screlease = 71,
  HSA_API_ID_hsa_signal_cas_scacq_screl = 72,
  HSA_API_ID_hsa_isa_from_name = 73,
  HSA_API_ID_hsa_isa_get_info = 74,
  HSA_API_ID_hsa_isa_compatible = 75,
  HSA_API_ID_hsa_code_object_serialize = 76,
  HSA_API_ID_hsa_code_object_deserialize = 77,
  HSA_API_ID_hsa_code_object_destroy = 78,
  HSA_API_ID_hsa_code_object_get_info = 79,
  HSA_API_ID_hsa_code_object_get_symbol = 80,
  HSA_API_ID_hsa_code_symbol_get_info = 81,
  HSA_API_ID_hsa_code_object_iterate_symbols = 82,
  HSA_API_ID_hsa_executable_create = 83,
  HSA_API_ID_hsa_executable_destroy = 84,
  HSA_API_ID_hsa_executable_load_code_object = 85,
  HSA_API_ID_hsa_executable_freeze = 86,
  HSA_API_ID_hsa_executable_get_info = 87,
  HSA_API_ID_hsa_executable_global_variable_define = 88,
  HSA_API_ID_hsa_executable_agent_global_variable_define = 89,
  HSA_API_ID_hsa_executable_readonly_variable_define = 90,
  HSA_API_ID_hsa_executable_validate = 91,
  HSA_API_ID_hsa_executable_get_symbol = 92,
  HSA_API_ID_hsa_executable_symbol_get_info = 93,
  HSA_API_ID_hsa_executable_iterate_symbols = 94,
  HSA_API_ID_hsa_status_string = 95,
  HSA_API_ID_hsa_extension_get_name = 96,
  HSA_API_ID_hsa_system_major_extension_supported = 97,
  HSA_API_ID_hsa_system_get_major_extension_table = 98,
  HSA_API_ID_hsa_agent_major_extension_supported = 99,
  HSA_API_ID_hsa_cache_get_info = 100,
  HSA_API_ID_hsa_agent_iterate_caches = 101,
  HSA_API_ID_hsa_signal_silent_store_relaxed = 102,
  HSA_API_ID_hsa_signal_silent_store_screlease = 103,
  HSA_API_ID_hsa_signal_group_create = 104,
  HSA_API_ID_hsa_signal_group_destroy = 105,
  HSA_API_ID_hsa_signal_group_wait_any_scacquire = 106,
  HSA_API_ID_hsa_signal_group_wait_any_relaxed = 107,
  HSA_API_ID_hsa_agent_iterate_isas = 108,
  HSA_API_ID_hsa_isa_get_info_alt = 109,
  HSA_API_ID_hsa_isa_get_exception_policies = 110,
  HSA_API_ID_hsa_isa_get_round_method = 111,
  HSA_API_ID_hsa_wavefront_get_info = 112,
  HSA_API_ID_hsa_isa_iterate_wavefronts = 113,
  HSA_API_ID_hsa_code_object_get_symbol_from_name = 114,
  HSA_API_ID_hsa_code_object_reader_create_from_file = 115,
  HSA_API_ID_hsa_code_object_reader_create_from_memory = 116,
  HSA_API_ID_hsa_code_object_reader_destroy = 117,
  HSA_API_ID_hsa_executable_create_alt = 118,
  HSA_API_ID_hsa_executable_load_program_code_object = 119,
  HSA_API_ID_hsa_executable_load_agent_code_object = 120,
  HSA_API_ID_hsa_executable_validate_alt = 121,
  HSA_API_ID_hsa_executable_get_symbol_by_name = 122,
  HSA_API_ID_hsa_executable_iterate_agent_symbols = 123,
  HSA_API_ID_hsa_executable_iterate_program_symbols = 124,

  // block: AmdExtTable API
  HSA_API_ID_hsa_amd_coherency_get_type = 125,
  HSA_API_ID_hsa_amd_coherency_set_type = 126,
  HSA_API_ID_hsa_amd_profiling_set_profiler_enabled = 127,
  HSA_API_ID_hsa_amd_profiling_async_copy_enable = 128,
  HSA_API_ID_hsa_amd_profiling_get_dispatch_time = 129,
  HSA_API_ID_hsa_amd_profiling_get_async_copy_time = 130,
  HSA_API_ID_hsa_amd_profiling_convert_tick_to_system_domain = 131,
  HSA_API_ID_hsa_amd_signal_async_handler = 132,
  HSA_API_ID_hsa_amd_async_function = 133,
  HSA_API_ID_hsa_amd_signal_wait_any = 134,
  HSA_API_ID_hsa_amd_queue_cu_set_mask = 135,
  HSA_API_ID_hsa_amd_memory_pool_get_info = 136,
  HSA_API_ID_hsa_amd_agent_iterate_memory_pools = 137,
  HSA_API_ID_hsa_amd_memory_pool_allocate = 138,
  HSA_API_ID_hsa_amd_memory_pool_free = 139,
  HSA_API_ID_hsa_amd_memory_async_copy = 140,
  HSA_API_ID_hsa_amd_agent_memory_pool_get_info = 141,
  HSA_API_ID_hsa_amd_agents_allow_access = 142,
  HSA_API_ID_hsa_amd_memory_pool_can_migrate = 143,
  HSA_API_ID_hsa_amd_memory_migrate = 144,
  HSA_API_ID_hsa_amd_memory_lock = 145,
  HSA_API_ID_hsa_amd_memory_unlock = 146,
  HSA_API_ID_hsa_amd_memory_fill = 147,
  HSA_API_ID_hsa_amd_interop_map_buffer = 148,
  HSA_API_ID_hsa_amd_interop_unmap_buffer = 149,
  HSA_API_ID_hsa_amd_image_create = 150,
  HSA_API_ID_hsa_amd_pointer_info = 151,
  HSA_API_ID_hsa_amd_pointer_info_set_userdata = 152,
  HSA_API_ID_hsa_amd_ipc_memory_create = 153,
  HSA_API_ID_hsa_amd_ipc_memory_attach = 154,
  HSA_API_ID_hsa_amd_ipc_memory_detach = 155,
  HSA_API_ID_hsa_amd_signal_create = 156,
  HSA_API_ID_hsa_amd_ipc_signal_create = 157,
  HSA_API_ID_hsa_amd_ipc_signal_attach = 158,
  HSA_API_ID_hsa_amd_register_system_event_handler = 159,
  HSA_API_ID_hsa_amd_queue_intercept_create = 160,
  HSA_API_ID_hsa_amd_queue_intercept_register = 161,
  HSA_API_ID_hsa_amd_queue_set_priority = 162,
  HSA_API_ID_hsa_amd_memory_async_copy_rect = 163,
  HSA_API_ID_hsa_amd_runtime_queue_create_register = 164,

  // block: ImageExtTable API
  HSA_API_ID_hsa_ext_image_get_capability = 165,
  HSA_API_ID_hsa_ext_image_data_get_info = 166,
  HSA_API_ID_hsa_ext_image_create = 167,
  HSA_API_ID_hsa_ext_image_import = 168,
  HSA_API_ID_hsa_ext_image_export = 169,
  HSA_API_ID_hsa_ext_image_copy = 170,
  HSA_API_ID_hsa_ext_image_clear = 171,
  HSA_API_ID_hsa_ext_image_destroy = 172,
  HSA_API_ID_hsa_ext_sampler_create = 173,
  HSA_API_ID_hsa_ext_sampler_destroy = 174,
  HSA_API_ID_hsa_ext_image_get_capability_with_layout = 175,
  HSA_API_ID_hsa_ext_image_data_get_info_with_layout = 176,
  HSA_API_ID_hsa_ext_image_create_with_layout = 177,

  HSA_API_ID_NUMBER = 178,
  HSA_API_ID_ANY = 179,
};

// section: API arg structure

struct hsa_api_data_t {
  uint64_t correlation_id;
  uint32_t phase;
  union {
    hsa_status_t hsa_status_t_retval;
    uint64_t uint64_t_retval;
    hsa_signal_value_t hsa_signal_value_t_retval;
    uint32_t uint32_t_retval;
  };
  union {
    // block: CoreApiTable API
    struct {
    } hsa_init;
    struct {
    } hsa_shut_down;
    struct {
      hsa_system_info_t attribute;
      void* value;
    } hsa_system_get_info;
    struct {
      uint16_t version_minor;
      bool* result;
      uint16_t extension;
      uint16_t version_major;
    } hsa_system_extension_supported;
    struct {
      void* table;
      uint16_t version_minor;
      uint16_t extension;
      uint16_t version_major;
    } hsa_system_get_extension_table;
    struct {
      hsa_status_t (* callback)(hsa_agent_t agent,void* data);
      void* data;
    } hsa_iterate_agents;
    struct {
      hsa_agent_info_t attribute;
      void* value;
      hsa_agent_t agent;
    } hsa_agent_get_info;
    struct {
      uint32_t private_segment_size;
      void* data;
      hsa_agent_t agent;
      hsa_queue_t** queue;
      void (* callback)(hsa_status_t status,hsa_queue_t* source,void* data);
      uint32_t group_segment_size;
      hsa_queue_type32_t type;
      uint32_t size;
    } hsa_queue_create;
    struct {
      uint32_t features;
      hsa_region_t region;
      hsa_queue_t** queue;
      hsa_signal_t doorbell_signal;
      hsa_queue_type32_t type;
      uint32_t size;
    } hsa_soft_queue_create;
    struct {
      hsa_queue_t* queue;
    } hsa_queue_destroy;
    struct {
      hsa_queue_t* queue;
    } hsa_queue_inactivate;
    struct {
      const hsa_queue_t* queue;
    } hsa_queue_load_read_index_scacquire;
    struct {
      const hsa_queue_t* queue;
    } hsa_queue_load_read_index_relaxed;
    struct {
      const hsa_queue_t* queue;
    } hsa_queue_load_write_index_scacquire;
    struct {
      const hsa_queue_t* queue;
    } hsa_queue_load_write_index_relaxed;
    struct {
      const hsa_queue_t* queue;
      uint64_t value;
    } hsa_queue_store_write_index_relaxed;
    struct {
      const hsa_queue_t* queue;
      uint64_t value;
    } hsa_queue_store_write_index_screlease;
    struct {
      const hsa_queue_t* queue;
      uint64_t expected;
      uint64_t value;
    } hsa_queue_cas_write_index_scacq_screl;
    struct {
      const hsa_queue_t* queue;
      uint64_t expected;
      uint64_t value;
    } hsa_queue_cas_write_index_scacquire;
    struct {
      const hsa_queue_t* queue;
      uint64_t expected;
      uint64_t value;
    } hsa_queue_cas_write_index_relaxed;
    struct {
      const hsa_queue_t* queue;
      uint64_t expected;
      uint64_t value;
    } hsa_queue_cas_write_index_screlease;
    struct {
      const hsa_queue_t* queue;
      uint64_t value;
    } hsa_queue_add_write_index_scacq_screl;
    struct {
      const hsa_queue_t* queue;
      uint64_t value;
    } hsa_queue_add_write_index_scacquire;
    struct {
      const hsa_queue_t* queue;
      uint64_t value;
    } hsa_queue_add_write_index_relaxed;
    struct {
      const hsa_queue_t* queue;
      uint64_t value;
    } hsa_queue_add_write_index_screlease;
    struct {
      const hsa_queue_t* queue;
      uint64_t value;
    } hsa_queue_store_read_index_relaxed;
    struct {
      const hsa_queue_t* queue;
      uint64_t value;
    } hsa_queue_store_read_index_screlease;
    struct {
      hsa_status_t (* callback)(hsa_region_t region,void* data);
      void* data;
      hsa_agent_t agent;
    } hsa_agent_iterate_regions;
    struct {
      hsa_region_info_t attribute;
      hsa_region_t region;
      void* value;
    } hsa_region_get_info;
    struct {
      hsa_profile_t profile;
      uint16_t* mask;
      hsa_agent_t agent;
    } hsa_agent_get_exception_policies;
    struct {
      bool* result;
      uint16_t version_minor;
      hsa_agent_t agent;
      uint16_t extension;
      uint16_t version_major;
    } hsa_agent_extension_supported;
    struct {
      void* ptr;
      size_t size;
    } hsa_memory_register;
    struct {
      void* ptr;
      size_t size;
    } hsa_memory_deregister;
    struct {
      hsa_region_t region;
      void** ptr;
      size_t size;
    } hsa_memory_allocate;
    struct {
      void* ptr;
    } hsa_memory_free;
    struct {
      const void* src;
      void* dst;
      size_t size;
    } hsa_memory_copy;
    struct {
      hsa_access_permission_t access;
      void* ptr;
      hsa_agent_t agent;
    } hsa_memory_assign_agent;
    struct {
      hsa_signal_t* signal;
      uint32_t num_consumers;
      const hsa_agent_t* consumers;
      hsa_signal_value_t initial_value;
    } hsa_signal_create;
    struct {
      hsa_signal_t signal;
    } hsa_signal_destroy;
    struct {
      hsa_signal_t signal;
    } hsa_signal_load_relaxed;
    struct {
      hsa_signal_t signal;
    } hsa_signal_load_scacquire;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_store_relaxed;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_store_screlease;
    struct {
      hsa_signal_t signal;
      uint64_t timeout_hint;
      hsa_wait_state_t wait_state_hint;
      hsa_signal_value_t compare_value;
      hsa_signal_condition_t condition;
    } hsa_signal_wait_relaxed;
    struct {
      hsa_signal_t signal;
      uint64_t timeout_hint;
      hsa_wait_state_t wait_state_hint;
      hsa_signal_value_t compare_value;
      hsa_signal_condition_t condition;
    } hsa_signal_wait_scacquire;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_and_relaxed;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_and_scacquire;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_and_screlease;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_and_scacq_screl;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_or_relaxed;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_or_scacquire;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_or_screlease;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_or_scacq_screl;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_xor_relaxed;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_xor_scacquire;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_xor_screlease;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_xor_scacq_screl;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_exchange_relaxed;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_exchange_scacquire;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_exchange_screlease;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_exchange_scacq_screl;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_add_relaxed;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_add_scacquire;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_add_screlease;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_add_scacq_screl;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_subtract_relaxed;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_subtract_scacquire;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_subtract_screlease;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_subtract_scacq_screl;
    struct {
      hsa_signal_value_t expected;
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_cas_relaxed;
    struct {
      hsa_signal_value_t expected;
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_cas_scacquire;
    struct {
      hsa_signal_value_t expected;
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_cas_screlease;
    struct {
      hsa_signal_value_t expected;
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_cas_scacq_screl;
    struct {
      hsa_isa_t* isa;
      const char* name;
    } hsa_isa_from_name;
    struct {
      hsa_isa_info_t attribute;
      hsa_isa_t isa;
      void* value;
      uint32_t index;
    } hsa_isa_get_info;
    struct {
      hsa_isa_t code_object_isa;
      bool* result;
      hsa_isa_t agent_isa;
    } hsa_isa_compatible;
    struct {
      hsa_callback_data_t callback_data;
      hsa_code_object_t code_object;
      void** serialized_code_object;
      size_t* serialized_code_object_size;
      hsa_status_t (* alloc_callback)(size_t size,hsa_callback_data_t data,void** address);
      const char* options;
    } hsa_code_object_serialize;
    struct {
      void* serialized_code_object;
      size_t serialized_code_object_size;
      hsa_code_object_t* code_object;
      const char* options;
    } hsa_code_object_deserialize;
    struct {
      hsa_code_object_t code_object;
    } hsa_code_object_destroy;
    struct {
      hsa_code_object_info_t attribute;
      hsa_code_object_t code_object;
      void* value;
    } hsa_code_object_get_info;
    struct {
      hsa_code_symbol_t* symbol;
      hsa_code_object_t code_object;
      const char* symbol_name;
    } hsa_code_object_get_symbol;
    struct {
      hsa_code_symbol_t code_symbol;
      void* value;
      hsa_code_symbol_info_t attribute;
    } hsa_code_symbol_get_info;
    struct {
      hsa_status_t (* callback)(hsa_code_object_t code_object,hsa_code_symbol_t symbol,void* data);
      hsa_code_object_t code_object;
      void* data;
    } hsa_code_object_iterate_symbols;
    struct {
      hsa_profile_t profile;
      hsa_executable_state_t executable_state;
      hsa_executable_t* executable;
      const char* options;
    } hsa_executable_create;
    struct {
      hsa_executable_t executable;
    } hsa_executable_destroy;
    struct {
      hsa_executable_t executable;
      hsa_code_object_t code_object;
      const char* options;
      hsa_agent_t agent;
    } hsa_executable_load_code_object;
    struct {
      hsa_executable_t executable;
      const char* options;
    } hsa_executable_freeze;
    struct {
      hsa_executable_info_t attribute;
      hsa_executable_t executable;
      void* value;
    } hsa_executable_get_info;
    struct {
      hsa_executable_t executable;
      void* address;
      const char* variable_name;
    } hsa_executable_global_variable_define;
    struct {
      hsa_executable_t executable;
      void* address;
      hsa_agent_t agent;
      const char* variable_name;
    } hsa_executable_agent_global_variable_define;
    struct {
      hsa_executable_t executable;
      void* address;
      hsa_agent_t agent;
      const char* variable_name;
    } hsa_executable_readonly_variable_define;
    struct {
      hsa_executable_t executable;
      uint32_t* result;
    } hsa_executable_validate;
    struct {
      hsa_executable_t executable;
      const char* symbol_name;
      hsa_executable_symbol_t* symbol;
      hsa_agent_t agent;
      int32_t call_convention;
      const char* module_name;
    } hsa_executable_get_symbol;
    struct {
      hsa_executable_symbol_info_t attribute;
      hsa_executable_symbol_t executable_symbol;
      void* value;
    } hsa_executable_symbol_get_info;
    struct {
      hsa_status_t (* callback)(hsa_executable_t exec,hsa_executable_symbol_t symbol,void* data);
      hsa_executable_t executable;
      void* data;
    } hsa_executable_iterate_symbols;
    struct {
      hsa_status_t status;
      const char** status_string;
    } hsa_status_string;
    struct {
      const char** name;
      uint16_t extension;
    } hsa_extension_get_name;
    struct {
      uint16_t* version_minor;
      bool* result;
      uint16_t extension;
      uint16_t version_major;
    } hsa_system_major_extension_supported;
    struct {
      void* table;
      size_t table_length;
      uint16_t extension;
      uint16_t version_major;
    } hsa_system_get_major_extension_table;
    struct {
      bool* result;
      uint16_t* version_minor;
      hsa_agent_t agent;
      uint16_t extension;
      uint16_t version_major;
    } hsa_agent_major_extension_supported;
    struct {
      hsa_cache_info_t attribute;
      hsa_cache_t cache;
      void* value;
    } hsa_cache_get_info;
    struct {
      hsa_status_t (* callback)(hsa_cache_t cache,void* data);
      void* data;
      hsa_agent_t agent;
    } hsa_agent_iterate_caches;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_silent_store_relaxed;
    struct {
      hsa_signal_t signal;
      hsa_signal_value_t value;
    } hsa_signal_silent_store_screlease;
    struct {
      const hsa_signal_t* signals;
      uint32_t num_signals;
      hsa_signal_group_t* signal_group;
      const hsa_agent_t* consumers;
      uint32_t num_consumers;
    } hsa_signal_group_create;
    struct {
      hsa_signal_group_t signal_group;
    } hsa_signal_group_destroy;
    struct {
      hsa_signal_group_t signal_group;
      hsa_wait_state_t wait_state_hint;
      hsa_signal_t* signal;
      const hsa_signal_value_t* compare_values;
      hsa_signal_value_t* value;
      const hsa_signal_condition_t* conditions;
    } hsa_signal_group_wait_any_scacquire;
    struct {
      hsa_signal_group_t signal_group;
      hsa_wait_state_t wait_state_hint;
      hsa_signal_t* signal;
      const hsa_signal_value_t* compare_values;
      hsa_signal_value_t* value;
      const hsa_signal_condition_t* conditions;
    } hsa_signal_group_wait_any_relaxed;
    struct {
      hsa_status_t (* callback)(hsa_isa_t isa,void* data);
      void* data;
      hsa_agent_t agent;
    } hsa_agent_iterate_isas;
    struct {
      hsa_isa_info_t attribute;
      hsa_isa_t isa;
      void* value;
    } hsa_isa_get_info_alt;
    struct {
      hsa_profile_t profile;
      hsa_isa_t isa;
      uint16_t* mask;
    } hsa_isa_get_exception_policies;
    struct {
      hsa_isa_t isa;
      hsa_round_method_t* round_method;
      hsa_flush_mode_t flush_mode;
      hsa_fp_type_t fp_type;
    } hsa_isa_get_round_method;
    struct {
      hsa_wavefront_info_t attribute;
      hsa_wavefront_t wavefront;
      void* value;
    } hsa_wavefront_get_info;
    struct {
      hsa_status_t (* callback)(hsa_wavefront_t wavefront,void* data);
      hsa_isa_t isa;
      void* data;
    } hsa_isa_iterate_wavefronts;
    struct {
      const char* module_name;
      hsa_code_symbol_t* symbol;
      hsa_code_object_t code_object;
      const char* symbol_name;
    } hsa_code_object_get_symbol_from_name;
    struct {
      hsa_code_object_reader_t* code_object_reader;
      hsa_file_t file;
    } hsa_code_object_reader_create_from_file;
    struct {
      hsa_code_object_reader_t* code_object_reader;
      const void* code_object;
      size_t size;
    } hsa_code_object_reader_create_from_memory;
    struct {
      hsa_code_object_reader_t code_object_reader;
    } hsa_code_object_reader_destroy;
    struct {
      hsa_profile_t profile;
      hsa_default_float_rounding_mode_t default_float_rounding_mode;
      hsa_executable_t* executable;
      const char* options;
    } hsa_executable_create_alt;
    struct {
      hsa_code_object_reader_t code_object_reader;
      hsa_executable_t executable;
      hsa_loaded_code_object_t* loaded_code_object;
      const char* options;
    } hsa_executable_load_program_code_object;
    struct {
      hsa_code_object_reader_t code_object_reader;
      hsa_executable_t executable;
      hsa_loaded_code_object_t* loaded_code_object;
      const char* options;
      hsa_agent_t agent;
    } hsa_executable_load_agent_code_object;
    struct {
      hsa_executable_t executable;
      const char* options;
      uint32_t* result;
    } hsa_executable_validate_alt;
    struct {
      hsa_executable_symbol_t* symbol;
      hsa_executable_t executable;
      const char* symbol_name;
      const hsa_agent_t* agent;
    } hsa_executable_get_symbol_by_name;
    struct {
      hsa_status_t (* callback)(hsa_executable_t exec,hsa_agent_t agent,hsa_executable_symbol_t symbol,void* data);
      hsa_executable_t executable;
      void* data;
      hsa_agent_t agent;
    } hsa_executable_iterate_agent_symbols;
    struct {
      hsa_status_t (* callback)(hsa_executable_t exec,hsa_executable_symbol_t symbol,void* data);
      hsa_executable_t executable;
      void* data;
    } hsa_executable_iterate_program_symbols;

    // block: AmdExtTable API
    struct {
      hsa_amd_coherency_type_t* type;
      hsa_agent_t agent;
    } hsa_amd_coherency_get_type;
    struct {
      hsa_amd_coherency_type_t type;
      hsa_agent_t agent;
    } hsa_amd_coherency_set_type;
    struct {
      hsa_queue_t* queue;
      int enable;
    } hsa_amd_profiling_set_profiler_enabled;
    struct {
      bool enable;
    } hsa_amd_profiling_async_copy_enable;
    struct {
      hsa_signal_t signal;
      hsa_agent_t agent;
      hsa_amd_profiling_dispatch_time_t* time;
    } hsa_amd_profiling_get_dispatch_time;
    struct {
      hsa_signal_t signal;
      hsa_amd_profiling_async_copy_time_t* time;
    } hsa_amd_profiling_get_async_copy_time;
    struct {
      uint64_t* system_tick;
      hsa_agent_t agent;
      uint64_t agent_tick;
    } hsa_amd_profiling_convert_tick_to_system_domain;
    struct {
      hsa_signal_t signal;
      hsa_amd_signal_handler handler;
      hsa_signal_condition_t cond;
      hsa_signal_value_t value;
      void* arg;
    } hsa_amd_signal_async_handler;
    struct {
      void (* callback)(void* arg);
      void* arg;
    } hsa_amd_async_function;
    struct {
      uint64_t timeout_hint;
      uint32_t signal_count;
      hsa_signal_condition_t* conds;
      hsa_signal_t* signals;
      hsa_signal_value_t* values;
      hsa_signal_value_t* satisfying_value;
      hsa_wait_state_t wait_hint;
    } hsa_amd_signal_wait_any;
    struct {
      const hsa_queue_t* queue;
      const uint32_t* cu_mask;
      uint32_t num_cu_mask_count;
    } hsa_amd_queue_cu_set_mask;
    struct {
      hsa_amd_memory_pool_info_t attribute;
      hsa_amd_memory_pool_t memory_pool;
      void* value;
    } hsa_amd_memory_pool_get_info;
    struct {
      hsa_status_t (* callback)(hsa_amd_memory_pool_t memory_pool,void* data);
      void* data;
      hsa_agent_t agent;
    } hsa_amd_agent_iterate_memory_pools;
    struct {
      void** ptr;
      uint32_t flags;
      hsa_amd_memory_pool_t memory_pool;
      size_t size;
    } hsa_amd_memory_pool_allocate;
    struct {
      void* ptr;
    } hsa_amd_memory_pool_free;
    struct {
      hsa_signal_t completion_signal;
      const void* src;
      void* dst;
      uint32_t num_dep_signals;
      hsa_agent_t src_agent;
      const hsa_signal_t* dep_signals;
      hsa_agent_t dst_agent;
      size_t size;
    } hsa_amd_memory_async_copy;
    struct {
      hsa_amd_agent_memory_pool_info_t attribute;
      void* value;
      hsa_amd_memory_pool_t memory_pool;
      hsa_agent_t agent;
    } hsa_amd_agent_memory_pool_get_info;
    struct {
      const uint32_t* flags;
      const hsa_agent_t* agents;
      const void* ptr;
      uint32_t num_agents;
    } hsa_amd_agents_allow_access;
    struct {
      hsa_amd_memory_pool_t src_memory_pool;
      hsa_amd_memory_pool_t dst_memory_pool;
      bool* result;
    } hsa_amd_memory_pool_can_migrate;
    struct {
      uint32_t flags;
      const void* ptr;
      hsa_amd_memory_pool_t memory_pool;
    } hsa_amd_memory_migrate;
    struct {
      void* host_ptr;
      int num_agent;
      hsa_agent_t* agents;
      void** agent_ptr;
      size_t size;
    } hsa_amd_memory_lock;
    struct {
      void* host_ptr;
    } hsa_amd_memory_unlock;
    struct {
      size_t count;
      void* ptr;
      uint32_t value;
    } hsa_amd_memory_fill;
    struct {
      uint32_t num_agents;
      size_t* metadata_size;
      uint32_t flags;
      hsa_agent_t* agents;
      const void** metadata;
      void** ptr;
      int interop_handle;
      size_t* size;
    } hsa_amd_interop_map_buffer;
    struct {
      void* ptr;
    } hsa_amd_interop_unmap_buffer;
    struct {
      const hsa_ext_image_descriptor_t* image_descriptor;
      hsa_ext_image_t* image;
      hsa_agent_t agent;
      hsa_access_permission_t access_permission;
      const void* image_data;
      const hsa_amd_image_descriptor_t* image_layout;
    } hsa_amd_image_create;
    struct {
      hsa_amd_pointer_info_t* info;
      hsa_agent_t** accessible;
      void* (* alloc)(size_t);
      void* ptr;
      uint32_t* num_agents_accessible;
    } hsa_amd_pointer_info;
    struct {
      void* userdata;
      void* ptr;
    } hsa_amd_pointer_info_set_userdata;
    struct {
      hsa_amd_ipc_memory_t* handle;
      void* ptr;
      size_t len;
    } hsa_amd_ipc_memory_create;
    struct {
      void** mapped_ptr;
      const hsa_amd_ipc_memory_t* handle;
      const hsa_agent_t* mapping_agents;
      size_t len;
      uint32_t num_agents;
    } hsa_amd_ipc_memory_attach;
    struct {
      void* mapped_ptr;
    } hsa_amd_ipc_memory_detach;
    struct {
      uint64_t attributes;
      hsa_signal_t* signal;
      uint32_t num_consumers;
      const hsa_agent_t* consumers;
      hsa_signal_value_t initial_value;
    } hsa_amd_signal_create;
    struct {
      hsa_signal_t signal;
      hsa_amd_ipc_signal_t* handle;
    } hsa_amd_ipc_signal_create;
    struct {
      hsa_signal_t* signal;
      const hsa_amd_ipc_signal_t* handle;
    } hsa_amd_ipc_signal_attach;
    struct {
      hsa_amd_system_event_callback_t callback;
      void* data;
    } hsa_amd_register_system_event_handler;
    struct {
      hsa_agent_t agent_handle;
      uint32_t private_segment_size;
      void* data;
      hsa_queue_t** queue;
      void (* callback)(hsa_status_t status,hsa_queue_t* source,void* data);
      uint32_t group_segment_size;
      hsa_queue_type32_t type;
      uint32_t size;
    } hsa_amd_queue_intercept_create;
    struct {
      hsa_queue_t* queue;
      hsa_amd_queue_intercept_handler callback;
      void* user_data;
    } hsa_amd_queue_intercept_register;
    struct {
      hsa_queue_t* queue;
      hsa_amd_queue_priority_t priority;
    } hsa_amd_queue_set_priority;
    struct {
      hsa_signal_t completion_signal;
      const hsa_pitched_ptr_t* src;
      const hsa_dim3_t* src_offset;
      const hsa_dim3_t* dst_offset;
      const hsa_pitched_ptr_t* dst;
      const hsa_signal_t* dep_signals;
      uint32_t num_dep_signals;
      const hsa_dim3_t* range;
      hsa_agent_t copy_agent;
      hsa_amd_copy_direction_t dir;
    } hsa_amd_memory_async_copy_rect;
    struct {
      hsa_amd_runtime_queue_notifier callback;
      void* user_data;
    } hsa_amd_runtime_queue_create_register;

    // block: ImageExtTable API
    struct {
      hsa_ext_image_geometry_t geometry;
      uint32_t* capability_mask;
      const hsa_ext_image_format_t* image_format;
      hsa_agent_t agent;
    } hsa_ext_image_get_capability;
    struct {
      const hsa_ext_image_descriptor_t* image_descriptor;
      hsa_ext_image_data_info_t* image_data_info;
      hsa_agent_t agent;
      hsa_access_permission_t access_permission;
    } hsa_ext_image_data_get_info;
    struct {
      const void* image_data;
      const hsa_ext_image_descriptor_t* image_descriptor;
      hsa_ext_image_t* image;
      hsa_agent_t agent;
      hsa_access_permission_t access_permission;
    } hsa_ext_image_create;
    struct {
      size_t src_row_pitch;
      const hsa_ext_image_region_t* image_region;
      hsa_agent_t agent;
      size_t src_slice_pitch;
      const void* src_memory;
      hsa_ext_image_t dst_image;
    } hsa_ext_image_import;
    struct {
      size_t dst_slice_pitch;
      const hsa_ext_image_region_t* image_region;
      hsa_agent_t agent;
      size_t dst_row_pitch;
      hsa_ext_image_t src_image;
      void* dst_memory;
    } hsa_ext_image_export;
    struct {
      const hsa_dim3_t* src_offset;
      const hsa_dim3_t* dst_offset;
      hsa_agent_t agent;
      const hsa_dim3_t* range;
      hsa_ext_image_t src_image;
      hsa_ext_image_t dst_image;
    } hsa_ext_image_copy;
    struct {
      hsa_ext_image_t image;
      const void* data;
      const hsa_ext_image_region_t* image_region;
      hsa_agent_t agent;
    } hsa_ext_image_clear;
    struct {
      hsa_ext_image_t image;
      hsa_agent_t agent;
    } hsa_ext_image_destroy;
    struct {
      const hsa_ext_sampler_descriptor_t* sampler_descriptor;
      hsa_agent_t agent;
      hsa_ext_sampler_t* sampler;
    } hsa_ext_sampler_create;
    struct {
      hsa_agent_t agent;
      hsa_ext_sampler_t sampler;
    } hsa_ext_sampler_destroy;
    struct {
      hsa_ext_image_geometry_t geometry;
      hsa_ext_image_data_layout_t image_data_layout;
      const hsa_ext_image_format_t* image_format;
      uint32_t* capability_mask;
      hsa_agent_t agent;
    } hsa_ext_image_get_capability_with_layout;
    struct {
      const hsa_ext_image_descriptor_t* image_descriptor;
      hsa_ext_image_data_layout_t image_data_layout;
      size_t image_data_row_pitch;
      hsa_agent_t agent;
      hsa_access_permission_t access_permission;
      size_t image_data_slice_pitch;
      hsa_ext_image_data_info_t* image_data_info;
    } hsa_ext_image_data_get_info_with_layout;
    struct {
      const hsa_ext_image_descriptor_t* image_descriptor;
      hsa_ext_image_data_layout_t image_data_layout;
      size_t image_data_row_pitch;
      hsa_agent_t agent;
      size_t image_data_slice_pitch;
      hsa_access_permission_t access_permission;
      const void* image_data;
      hsa_ext_image_t* image;
    } hsa_ext_image_create_with_layout;
  } args;
};

#if PROF_API_IMPL
namespace roctracer {
namespace hsa_support {

// section: API callback functions

typedef CbTable<HSA_API_ID_NUMBER> cb_table_t;
extern cb_table_t cb_table;

// block: CoreApiTable API
static hsa_status_t hsa_init_callback() {
  hsa_api_data_t api_data{};
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_init, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_init, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_init_fn();
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_init, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_shut_down_callback() {
  hsa_api_data_t api_data{};
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_shut_down, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_shut_down, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_shut_down_fn();
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_shut_down, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_system_get_info_callback(hsa_system_info_t attribute, void* value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_system_get_info.attribute = attribute;
  api_data.args.hsa_system_get_info.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_system_get_info, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_system_get_info, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_system_get_info_fn(attribute, value);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_system_get_info, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_system_extension_supported_callback(uint16_t extension, uint16_t version_major, uint16_t version_minor, bool* result) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_system_extension_supported.extension = extension;
  api_data.args.hsa_system_extension_supported.version_major = version_major;
  api_data.args.hsa_system_extension_supported.version_minor = version_minor;
  api_data.args.hsa_system_extension_supported.result = result;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_system_extension_supported, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_system_extension_supported, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_system_extension_supported_fn(extension, version_major, version_minor, result);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_system_extension_supported, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_system_get_extension_table_callback(uint16_t extension, uint16_t version_major, uint16_t version_minor, void* table) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_system_get_extension_table.extension = extension;
  api_data.args.hsa_system_get_extension_table.version_major = version_major;
  api_data.args.hsa_system_get_extension_table.version_minor = version_minor;
  api_data.args.hsa_system_get_extension_table.table = table;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_system_get_extension_table, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_system_get_extension_table, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_system_get_extension_table_fn(extension, version_major, version_minor, table);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_system_get_extension_table, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_iterate_agents_callback(hsa_status_t (* callback)(hsa_agent_t agent, void* data), void* data) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_iterate_agents.callback = callback;
  api_data.args.hsa_iterate_agents.data = data;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_iterate_agents, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_iterate_agents, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_iterate_agents_fn(callback, data);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_iterate_agents, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_agent_get_info_callback(hsa_agent_t agent, hsa_agent_info_t attribute, void* value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_agent_get_info.agent = agent;
  api_data.args.hsa_agent_get_info.attribute = attribute;
  api_data.args.hsa_agent_get_info.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_agent_get_info, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_agent_get_info, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_agent_get_info_fn(agent, attribute, value);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_agent_get_info, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_queue_create_callback(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type, void (* callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data, uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_create.agent = agent;
  api_data.args.hsa_queue_create.size = size;
  api_data.args.hsa_queue_create.type = type;
  api_data.args.hsa_queue_create.callback = callback;
  api_data.args.hsa_queue_create.data = data;
  api_data.args.hsa_queue_create.private_segment_size = private_segment_size;
  api_data.args.hsa_queue_create.group_segment_size = group_segment_size;
  api_data.args.hsa_queue_create.queue = queue;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_create, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_create, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_queue_create_fn(agent, size, type, callback, data, private_segment_size, group_segment_size, queue);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_create, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_soft_queue_create_callback(hsa_region_t region, uint32_t size, hsa_queue_type32_t type, uint32_t features, hsa_signal_t doorbell_signal, hsa_queue_t** queue) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_soft_queue_create.region = region;
  api_data.args.hsa_soft_queue_create.size = size;
  api_data.args.hsa_soft_queue_create.type = type;
  api_data.args.hsa_soft_queue_create.features = features;
  api_data.args.hsa_soft_queue_create.doorbell_signal = doorbell_signal;
  api_data.args.hsa_soft_queue_create.queue = queue;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_soft_queue_create, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_soft_queue_create, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_soft_queue_create_fn(region, size, type, features, doorbell_signal, queue);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_soft_queue_create, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_queue_destroy_callback(hsa_queue_t* queue) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_destroy.queue = queue;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_destroy, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_destroy, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_queue_destroy_fn(queue);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_destroy, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_queue_inactivate_callback(hsa_queue_t* queue) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_inactivate.queue = queue;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_inactivate, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_inactivate, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_queue_inactivate_fn(queue);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_inactivate, &api_data, api_callback_arg);
  return ret;
}
static uint64_t hsa_queue_load_read_index_scacquire_callback(const hsa_queue_t* queue) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_load_read_index_scacquire.queue = queue;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_load_read_index_scacquire, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_load_read_index_scacquire, &api_data, api_callback_arg);
  uint64_t ret =  CoreApiTable_saved.hsa_queue_load_read_index_scacquire_fn(queue);
  api_data.uint64_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_load_read_index_scacquire, &api_data, api_callback_arg);
  return ret;
}
static uint64_t hsa_queue_load_read_index_relaxed_callback(const hsa_queue_t* queue) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_load_read_index_relaxed.queue = queue;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_load_read_index_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_load_read_index_relaxed, &api_data, api_callback_arg);
  uint64_t ret =  CoreApiTable_saved.hsa_queue_load_read_index_relaxed_fn(queue);
  api_data.uint64_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_load_read_index_relaxed, &api_data, api_callback_arg);
  return ret;
}
static uint64_t hsa_queue_load_write_index_scacquire_callback(const hsa_queue_t* queue) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_load_write_index_scacquire.queue = queue;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_load_write_index_scacquire, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_load_write_index_scacquire, &api_data, api_callback_arg);
  uint64_t ret =  CoreApiTable_saved.hsa_queue_load_write_index_scacquire_fn(queue);
  api_data.uint64_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_load_write_index_scacquire, &api_data, api_callback_arg);
  return ret;
}
static uint64_t hsa_queue_load_write_index_relaxed_callback(const hsa_queue_t* queue) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_load_write_index_relaxed.queue = queue;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_load_write_index_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_load_write_index_relaxed, &api_data, api_callback_arg);
  uint64_t ret =  CoreApiTable_saved.hsa_queue_load_write_index_relaxed_fn(queue);
  api_data.uint64_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_load_write_index_relaxed, &api_data, api_callback_arg);
  return ret;
}
static void hsa_queue_store_write_index_relaxed_callback(const hsa_queue_t* queue, uint64_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_store_write_index_relaxed.queue = queue;
  api_data.args.hsa_queue_store_write_index_relaxed.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_store_write_index_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_store_write_index_relaxed, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_queue_store_write_index_relaxed_fn(queue, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_store_write_index_relaxed, &api_data, api_callback_arg);
}
static void hsa_queue_store_write_index_screlease_callback(const hsa_queue_t* queue, uint64_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_store_write_index_screlease.queue = queue;
  api_data.args.hsa_queue_store_write_index_screlease.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_store_write_index_screlease, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_store_write_index_screlease, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_queue_store_write_index_screlease_fn(queue, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_store_write_index_screlease, &api_data, api_callback_arg);
}
static uint64_t hsa_queue_cas_write_index_scacq_screl_callback(const hsa_queue_t* queue, uint64_t expected, uint64_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_cas_write_index_scacq_screl.queue = queue;
  api_data.args.hsa_queue_cas_write_index_scacq_screl.expected = expected;
  api_data.args.hsa_queue_cas_write_index_scacq_screl.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_cas_write_index_scacq_screl, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_cas_write_index_scacq_screl, &api_data, api_callback_arg);
  uint64_t ret =  CoreApiTable_saved.hsa_queue_cas_write_index_scacq_screl_fn(queue, expected, value);
  api_data.uint64_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_cas_write_index_scacq_screl, &api_data, api_callback_arg);
  return ret;
}
static uint64_t hsa_queue_cas_write_index_scacquire_callback(const hsa_queue_t* queue, uint64_t expected, uint64_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_cas_write_index_scacquire.queue = queue;
  api_data.args.hsa_queue_cas_write_index_scacquire.expected = expected;
  api_data.args.hsa_queue_cas_write_index_scacquire.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_cas_write_index_scacquire, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_cas_write_index_scacquire, &api_data, api_callback_arg);
  uint64_t ret =  CoreApiTable_saved.hsa_queue_cas_write_index_scacquire_fn(queue, expected, value);
  api_data.uint64_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_cas_write_index_scacquire, &api_data, api_callback_arg);
  return ret;
}
static uint64_t hsa_queue_cas_write_index_relaxed_callback(const hsa_queue_t* queue, uint64_t expected, uint64_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_cas_write_index_relaxed.queue = queue;
  api_data.args.hsa_queue_cas_write_index_relaxed.expected = expected;
  api_data.args.hsa_queue_cas_write_index_relaxed.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_cas_write_index_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_cas_write_index_relaxed, &api_data, api_callback_arg);
  uint64_t ret =  CoreApiTable_saved.hsa_queue_cas_write_index_relaxed_fn(queue, expected, value);
  api_data.uint64_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_cas_write_index_relaxed, &api_data, api_callback_arg);
  return ret;
}
static uint64_t hsa_queue_cas_write_index_screlease_callback(const hsa_queue_t* queue, uint64_t expected, uint64_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_cas_write_index_screlease.queue = queue;
  api_data.args.hsa_queue_cas_write_index_screlease.expected = expected;
  api_data.args.hsa_queue_cas_write_index_screlease.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_cas_write_index_screlease, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_cas_write_index_screlease, &api_data, api_callback_arg);
  uint64_t ret =  CoreApiTable_saved.hsa_queue_cas_write_index_screlease_fn(queue, expected, value);
  api_data.uint64_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_cas_write_index_screlease, &api_data, api_callback_arg);
  return ret;
}
static uint64_t hsa_queue_add_write_index_scacq_screl_callback(const hsa_queue_t* queue, uint64_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_add_write_index_scacq_screl.queue = queue;
  api_data.args.hsa_queue_add_write_index_scacq_screl.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_add_write_index_scacq_screl, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_add_write_index_scacq_screl, &api_data, api_callback_arg);
  uint64_t ret =  CoreApiTable_saved.hsa_queue_add_write_index_scacq_screl_fn(queue, value);
  api_data.uint64_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_add_write_index_scacq_screl, &api_data, api_callback_arg);
  return ret;
}
static uint64_t hsa_queue_add_write_index_scacquire_callback(const hsa_queue_t* queue, uint64_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_add_write_index_scacquire.queue = queue;
  api_data.args.hsa_queue_add_write_index_scacquire.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_add_write_index_scacquire, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_add_write_index_scacquire, &api_data, api_callback_arg);
  uint64_t ret =  CoreApiTable_saved.hsa_queue_add_write_index_scacquire_fn(queue, value);
  api_data.uint64_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_add_write_index_scacquire, &api_data, api_callback_arg);
  return ret;
}
static uint64_t hsa_queue_add_write_index_relaxed_callback(const hsa_queue_t* queue, uint64_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_add_write_index_relaxed.queue = queue;
  api_data.args.hsa_queue_add_write_index_relaxed.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_add_write_index_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_add_write_index_relaxed, &api_data, api_callback_arg);
  uint64_t ret =  CoreApiTable_saved.hsa_queue_add_write_index_relaxed_fn(queue, value);
  api_data.uint64_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_add_write_index_relaxed, &api_data, api_callback_arg);
  return ret;
}
static uint64_t hsa_queue_add_write_index_screlease_callback(const hsa_queue_t* queue, uint64_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_add_write_index_screlease.queue = queue;
  api_data.args.hsa_queue_add_write_index_screlease.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_add_write_index_screlease, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_add_write_index_screlease, &api_data, api_callback_arg);
  uint64_t ret =  CoreApiTable_saved.hsa_queue_add_write_index_screlease_fn(queue, value);
  api_data.uint64_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_add_write_index_screlease, &api_data, api_callback_arg);
  return ret;
}
static void hsa_queue_store_read_index_relaxed_callback(const hsa_queue_t* queue, uint64_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_store_read_index_relaxed.queue = queue;
  api_data.args.hsa_queue_store_read_index_relaxed.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_store_read_index_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_store_read_index_relaxed, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_queue_store_read_index_relaxed_fn(queue, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_store_read_index_relaxed, &api_data, api_callback_arg);
}
static void hsa_queue_store_read_index_screlease_callback(const hsa_queue_t* queue, uint64_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_queue_store_read_index_screlease.queue = queue;
  api_data.args.hsa_queue_store_read_index_screlease.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_queue_store_read_index_screlease, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_store_read_index_screlease, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_queue_store_read_index_screlease_fn(queue, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_queue_store_read_index_screlease, &api_data, api_callback_arg);
}
static hsa_status_t hsa_agent_iterate_regions_callback(hsa_agent_t agent, hsa_status_t (* callback)(hsa_region_t region, void* data), void* data) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_agent_iterate_regions.agent = agent;
  api_data.args.hsa_agent_iterate_regions.callback = callback;
  api_data.args.hsa_agent_iterate_regions.data = data;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_agent_iterate_regions, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_agent_iterate_regions, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_agent_iterate_regions_fn(agent, callback, data);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_agent_iterate_regions, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_region_get_info_callback(hsa_region_t region, hsa_region_info_t attribute, void* value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_region_get_info.region = region;
  api_data.args.hsa_region_get_info.attribute = attribute;
  api_data.args.hsa_region_get_info.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_region_get_info, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_region_get_info, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_region_get_info_fn(region, attribute, value);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_region_get_info, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_agent_get_exception_policies_callback(hsa_agent_t agent, hsa_profile_t profile, uint16_t* mask) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_agent_get_exception_policies.agent = agent;
  api_data.args.hsa_agent_get_exception_policies.profile = profile;
  api_data.args.hsa_agent_get_exception_policies.mask = mask;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_agent_get_exception_policies, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_agent_get_exception_policies, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_agent_get_exception_policies_fn(agent, profile, mask);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_agent_get_exception_policies, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_agent_extension_supported_callback(uint16_t extension, hsa_agent_t agent, uint16_t version_major, uint16_t version_minor, bool* result) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_agent_extension_supported.extension = extension;
  api_data.args.hsa_agent_extension_supported.agent = agent;
  api_data.args.hsa_agent_extension_supported.version_major = version_major;
  api_data.args.hsa_agent_extension_supported.version_minor = version_minor;
  api_data.args.hsa_agent_extension_supported.result = result;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_agent_extension_supported, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_agent_extension_supported, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_agent_extension_supported_fn(extension, agent, version_major, version_minor, result);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_agent_extension_supported, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_memory_register_callback(void* ptr, size_t size) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_memory_register.ptr = ptr;
  api_data.args.hsa_memory_register.size = size;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_memory_register, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_memory_register, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_memory_register_fn(ptr, size);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_memory_register, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_memory_deregister_callback(void* ptr, size_t size) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_memory_deregister.ptr = ptr;
  api_data.args.hsa_memory_deregister.size = size;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_memory_deregister, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_memory_deregister, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_memory_deregister_fn(ptr, size);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_memory_deregister, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_memory_allocate_callback(hsa_region_t region, size_t size, void** ptr) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_memory_allocate.region = region;
  api_data.args.hsa_memory_allocate.size = size;
  api_data.args.hsa_memory_allocate.ptr = ptr;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_memory_allocate, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_memory_allocate, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_memory_allocate_fn(region, size, ptr);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_memory_allocate, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_memory_free_callback(void* ptr) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_memory_free.ptr = ptr;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_memory_free, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_memory_free, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_memory_free_fn(ptr);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_memory_free, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_memory_copy_callback(void* dst, const void* src, size_t size) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_memory_copy.dst = dst;
  api_data.args.hsa_memory_copy.src = src;
  api_data.args.hsa_memory_copy.size = size;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_memory_copy, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_memory_copy, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_memory_copy_fn(dst, src, size);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_memory_copy, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_memory_assign_agent_callback(void* ptr, hsa_agent_t agent, hsa_access_permission_t access) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_memory_assign_agent.ptr = ptr;
  api_data.args.hsa_memory_assign_agent.agent = agent;
  api_data.args.hsa_memory_assign_agent.access = access;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_memory_assign_agent, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_memory_assign_agent, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_memory_assign_agent_fn(ptr, agent, access);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_memory_assign_agent, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_signal_create_callback(hsa_signal_value_t initial_value, uint32_t num_consumers, const hsa_agent_t* consumers, hsa_signal_t* signal) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_create.initial_value = initial_value;
  api_data.args.hsa_signal_create.num_consumers = num_consumers;
  api_data.args.hsa_signal_create.consumers = consumers;
  api_data.args.hsa_signal_create.signal = signal;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_create, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_create, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_signal_create_fn(initial_value, num_consumers, consumers, signal);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_create, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_signal_destroy_callback(hsa_signal_t signal) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_destroy.signal = signal;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_destroy, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_destroy, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_signal_destroy_fn(signal);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_destroy, &api_data, api_callback_arg);
  return ret;
}
static hsa_signal_value_t hsa_signal_load_relaxed_callback(hsa_signal_t signal) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_load_relaxed.signal = signal;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_load_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_load_relaxed, &api_data, api_callback_arg);
  hsa_signal_value_t ret =  CoreApiTable_saved.hsa_signal_load_relaxed_fn(signal);
  api_data.hsa_signal_value_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_load_relaxed, &api_data, api_callback_arg);
  return ret;
}
static hsa_signal_value_t hsa_signal_load_scacquire_callback(hsa_signal_t signal) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_load_scacquire.signal = signal;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_load_scacquire, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_load_scacquire, &api_data, api_callback_arg);
  hsa_signal_value_t ret =  CoreApiTable_saved.hsa_signal_load_scacquire_fn(signal);
  api_data.hsa_signal_value_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_load_scacquire, &api_data, api_callback_arg);
  return ret;
}
static void hsa_signal_store_relaxed_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_store_relaxed.signal = signal;
  api_data.args.hsa_signal_store_relaxed.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_store_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_store_relaxed, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_store_relaxed_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_store_relaxed, &api_data, api_callback_arg);
}
static void hsa_signal_store_screlease_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_store_screlease.signal = signal;
  api_data.args.hsa_signal_store_screlease.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_store_screlease, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_store_screlease, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_store_screlease_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_store_screlease, &api_data, api_callback_arg);
}
static hsa_signal_value_t hsa_signal_wait_relaxed_callback(hsa_signal_t signal, hsa_signal_condition_t condition, hsa_signal_value_t compare_value, uint64_t timeout_hint, hsa_wait_state_t wait_state_hint) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_wait_relaxed.signal = signal;
  api_data.args.hsa_signal_wait_relaxed.condition = condition;
  api_data.args.hsa_signal_wait_relaxed.compare_value = compare_value;
  api_data.args.hsa_signal_wait_relaxed.timeout_hint = timeout_hint;
  api_data.args.hsa_signal_wait_relaxed.wait_state_hint = wait_state_hint;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_wait_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_wait_relaxed, &api_data, api_callback_arg);
  hsa_signal_value_t ret =  CoreApiTable_saved.hsa_signal_wait_relaxed_fn(signal, condition, compare_value, timeout_hint, wait_state_hint);
  api_data.hsa_signal_value_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_wait_relaxed, &api_data, api_callback_arg);
  return ret;
}
static hsa_signal_value_t hsa_signal_wait_scacquire_callback(hsa_signal_t signal, hsa_signal_condition_t condition, hsa_signal_value_t compare_value, uint64_t timeout_hint, hsa_wait_state_t wait_state_hint) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_wait_scacquire.signal = signal;
  api_data.args.hsa_signal_wait_scacquire.condition = condition;
  api_data.args.hsa_signal_wait_scacquire.compare_value = compare_value;
  api_data.args.hsa_signal_wait_scacquire.timeout_hint = timeout_hint;
  api_data.args.hsa_signal_wait_scacquire.wait_state_hint = wait_state_hint;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_wait_scacquire, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_wait_scacquire, &api_data, api_callback_arg);
  hsa_signal_value_t ret =  CoreApiTable_saved.hsa_signal_wait_scacquire_fn(signal, condition, compare_value, timeout_hint, wait_state_hint);
  api_data.hsa_signal_value_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_wait_scacquire, &api_data, api_callback_arg);
  return ret;
}
static void hsa_signal_and_relaxed_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_and_relaxed.signal = signal;
  api_data.args.hsa_signal_and_relaxed.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_and_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_and_relaxed, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_and_relaxed_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_and_relaxed, &api_data, api_callback_arg);
}
static void hsa_signal_and_scacquire_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_and_scacquire.signal = signal;
  api_data.args.hsa_signal_and_scacquire.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_and_scacquire, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_and_scacquire, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_and_scacquire_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_and_scacquire, &api_data, api_callback_arg);
}
static void hsa_signal_and_screlease_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_and_screlease.signal = signal;
  api_data.args.hsa_signal_and_screlease.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_and_screlease, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_and_screlease, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_and_screlease_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_and_screlease, &api_data, api_callback_arg);
}
static void hsa_signal_and_scacq_screl_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_and_scacq_screl.signal = signal;
  api_data.args.hsa_signal_and_scacq_screl.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_and_scacq_screl, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_and_scacq_screl, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_and_scacq_screl_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_and_scacq_screl, &api_data, api_callback_arg);
}
static void hsa_signal_or_relaxed_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_or_relaxed.signal = signal;
  api_data.args.hsa_signal_or_relaxed.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_or_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_or_relaxed, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_or_relaxed_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_or_relaxed, &api_data, api_callback_arg);
}
static void hsa_signal_or_scacquire_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_or_scacquire.signal = signal;
  api_data.args.hsa_signal_or_scacquire.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_or_scacquire, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_or_scacquire, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_or_scacquire_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_or_scacquire, &api_data, api_callback_arg);
}
static void hsa_signal_or_screlease_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_or_screlease.signal = signal;
  api_data.args.hsa_signal_or_screlease.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_or_screlease, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_or_screlease, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_or_screlease_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_or_screlease, &api_data, api_callback_arg);
}
static void hsa_signal_or_scacq_screl_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_or_scacq_screl.signal = signal;
  api_data.args.hsa_signal_or_scacq_screl.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_or_scacq_screl, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_or_scacq_screl, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_or_scacq_screl_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_or_scacq_screl, &api_data, api_callback_arg);
}
static void hsa_signal_xor_relaxed_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_xor_relaxed.signal = signal;
  api_data.args.hsa_signal_xor_relaxed.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_xor_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_xor_relaxed, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_xor_relaxed_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_xor_relaxed, &api_data, api_callback_arg);
}
static void hsa_signal_xor_scacquire_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_xor_scacquire.signal = signal;
  api_data.args.hsa_signal_xor_scacquire.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_xor_scacquire, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_xor_scacquire, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_xor_scacquire_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_xor_scacquire, &api_data, api_callback_arg);
}
static void hsa_signal_xor_screlease_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_xor_screlease.signal = signal;
  api_data.args.hsa_signal_xor_screlease.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_xor_screlease, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_xor_screlease, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_xor_screlease_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_xor_screlease, &api_data, api_callback_arg);
}
static void hsa_signal_xor_scacq_screl_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_xor_scacq_screl.signal = signal;
  api_data.args.hsa_signal_xor_scacq_screl.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_xor_scacq_screl, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_xor_scacq_screl, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_xor_scacq_screl_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_xor_scacq_screl, &api_data, api_callback_arg);
}
static hsa_signal_value_t hsa_signal_exchange_relaxed_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_exchange_relaxed.signal = signal;
  api_data.args.hsa_signal_exchange_relaxed.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_exchange_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_exchange_relaxed, &api_data, api_callback_arg);
  hsa_signal_value_t ret =  CoreApiTable_saved.hsa_signal_exchange_relaxed_fn(signal, value);
  api_data.hsa_signal_value_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_exchange_relaxed, &api_data, api_callback_arg);
  return ret;
}
static hsa_signal_value_t hsa_signal_exchange_scacquire_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_exchange_scacquire.signal = signal;
  api_data.args.hsa_signal_exchange_scacquire.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_exchange_scacquire, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_exchange_scacquire, &api_data, api_callback_arg);
  hsa_signal_value_t ret =  CoreApiTable_saved.hsa_signal_exchange_scacquire_fn(signal, value);
  api_data.hsa_signal_value_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_exchange_scacquire, &api_data, api_callback_arg);
  return ret;
}
static hsa_signal_value_t hsa_signal_exchange_screlease_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_exchange_screlease.signal = signal;
  api_data.args.hsa_signal_exchange_screlease.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_exchange_screlease, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_exchange_screlease, &api_data, api_callback_arg);
  hsa_signal_value_t ret =  CoreApiTable_saved.hsa_signal_exchange_screlease_fn(signal, value);
  api_data.hsa_signal_value_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_exchange_screlease, &api_data, api_callback_arg);
  return ret;
}
static hsa_signal_value_t hsa_signal_exchange_scacq_screl_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_exchange_scacq_screl.signal = signal;
  api_data.args.hsa_signal_exchange_scacq_screl.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_exchange_scacq_screl, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_exchange_scacq_screl, &api_data, api_callback_arg);
  hsa_signal_value_t ret =  CoreApiTable_saved.hsa_signal_exchange_scacq_screl_fn(signal, value);
  api_data.hsa_signal_value_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_exchange_scacq_screl, &api_data, api_callback_arg);
  return ret;
}
static void hsa_signal_add_relaxed_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_add_relaxed.signal = signal;
  api_data.args.hsa_signal_add_relaxed.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_add_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_add_relaxed, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_add_relaxed_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_add_relaxed, &api_data, api_callback_arg);
}
static void hsa_signal_add_scacquire_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_add_scacquire.signal = signal;
  api_data.args.hsa_signal_add_scacquire.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_add_scacquire, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_add_scacquire, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_add_scacquire_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_add_scacquire, &api_data, api_callback_arg);
}
static void hsa_signal_add_screlease_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_add_screlease.signal = signal;
  api_data.args.hsa_signal_add_screlease.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_add_screlease, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_add_screlease, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_add_screlease_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_add_screlease, &api_data, api_callback_arg);
}
static void hsa_signal_add_scacq_screl_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_add_scacq_screl.signal = signal;
  api_data.args.hsa_signal_add_scacq_screl.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_add_scacq_screl, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_add_scacq_screl, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_add_scacq_screl_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_add_scacq_screl, &api_data, api_callback_arg);
}
static void hsa_signal_subtract_relaxed_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_subtract_relaxed.signal = signal;
  api_data.args.hsa_signal_subtract_relaxed.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_subtract_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_subtract_relaxed, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_subtract_relaxed_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_subtract_relaxed, &api_data, api_callback_arg);
}
static void hsa_signal_subtract_scacquire_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_subtract_scacquire.signal = signal;
  api_data.args.hsa_signal_subtract_scacquire.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_subtract_scacquire, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_subtract_scacquire, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_subtract_scacquire_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_subtract_scacquire, &api_data, api_callback_arg);
}
static void hsa_signal_subtract_screlease_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_subtract_screlease.signal = signal;
  api_data.args.hsa_signal_subtract_screlease.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_subtract_screlease, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_subtract_screlease, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_subtract_screlease_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_subtract_screlease, &api_data, api_callback_arg);
}
static void hsa_signal_subtract_scacq_screl_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_subtract_scacq_screl.signal = signal;
  api_data.args.hsa_signal_subtract_scacq_screl.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_subtract_scacq_screl, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_subtract_scacq_screl, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_subtract_scacq_screl_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_subtract_scacq_screl, &api_data, api_callback_arg);
}
static hsa_signal_value_t hsa_signal_cas_relaxed_callback(hsa_signal_t signal, hsa_signal_value_t expected, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_cas_relaxed.signal = signal;
  api_data.args.hsa_signal_cas_relaxed.expected = expected;
  api_data.args.hsa_signal_cas_relaxed.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_cas_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_cas_relaxed, &api_data, api_callback_arg);
  hsa_signal_value_t ret =  CoreApiTable_saved.hsa_signal_cas_relaxed_fn(signal, expected, value);
  api_data.hsa_signal_value_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_cas_relaxed, &api_data, api_callback_arg);
  return ret;
}
static hsa_signal_value_t hsa_signal_cas_scacquire_callback(hsa_signal_t signal, hsa_signal_value_t expected, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_cas_scacquire.signal = signal;
  api_data.args.hsa_signal_cas_scacquire.expected = expected;
  api_data.args.hsa_signal_cas_scacquire.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_cas_scacquire, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_cas_scacquire, &api_data, api_callback_arg);
  hsa_signal_value_t ret =  CoreApiTable_saved.hsa_signal_cas_scacquire_fn(signal, expected, value);
  api_data.hsa_signal_value_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_cas_scacquire, &api_data, api_callback_arg);
  return ret;
}
static hsa_signal_value_t hsa_signal_cas_screlease_callback(hsa_signal_t signal, hsa_signal_value_t expected, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_cas_screlease.signal = signal;
  api_data.args.hsa_signal_cas_screlease.expected = expected;
  api_data.args.hsa_signal_cas_screlease.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_cas_screlease, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_cas_screlease, &api_data, api_callback_arg);
  hsa_signal_value_t ret =  CoreApiTable_saved.hsa_signal_cas_screlease_fn(signal, expected, value);
  api_data.hsa_signal_value_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_cas_screlease, &api_data, api_callback_arg);
  return ret;
}
static hsa_signal_value_t hsa_signal_cas_scacq_screl_callback(hsa_signal_t signal, hsa_signal_value_t expected, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_cas_scacq_screl.signal = signal;
  api_data.args.hsa_signal_cas_scacq_screl.expected = expected;
  api_data.args.hsa_signal_cas_scacq_screl.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_cas_scacq_screl, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_cas_scacq_screl, &api_data, api_callback_arg);
  hsa_signal_value_t ret =  CoreApiTable_saved.hsa_signal_cas_scacq_screl_fn(signal, expected, value);
  api_data.hsa_signal_value_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_cas_scacq_screl, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_isa_from_name_callback(const char* name, hsa_isa_t* isa) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_isa_from_name.name = name;
  api_data.args.hsa_isa_from_name.isa = isa;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_isa_from_name, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_isa_from_name, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_isa_from_name_fn(name, isa);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_isa_from_name, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_isa_get_info_callback(hsa_isa_t isa, hsa_isa_info_t attribute, uint32_t index, void* value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_isa_get_info.isa = isa;
  api_data.args.hsa_isa_get_info.attribute = attribute;
  api_data.args.hsa_isa_get_info.index = index;
  api_data.args.hsa_isa_get_info.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_isa_get_info, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_isa_get_info, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_isa_get_info_fn(isa, attribute, index, value);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_isa_get_info, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_isa_compatible_callback(hsa_isa_t code_object_isa, hsa_isa_t agent_isa, bool* result) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_isa_compatible.code_object_isa = code_object_isa;
  api_data.args.hsa_isa_compatible.agent_isa = agent_isa;
  api_data.args.hsa_isa_compatible.result = result;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_isa_compatible, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_isa_compatible, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_isa_compatible_fn(code_object_isa, agent_isa, result);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_isa_compatible, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_code_object_serialize_callback(hsa_code_object_t code_object, hsa_status_t (* alloc_callback)(size_t size, hsa_callback_data_t data, void** address), hsa_callback_data_t callback_data, const char* options, void** serialized_code_object, size_t* serialized_code_object_size) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_code_object_serialize.code_object = code_object;
  api_data.args.hsa_code_object_serialize.alloc_callback = alloc_callback;
  api_data.args.hsa_code_object_serialize.callback_data = callback_data;
  api_data.args.hsa_code_object_serialize.options = options;
  api_data.args.hsa_code_object_serialize.serialized_code_object = serialized_code_object;
  api_data.args.hsa_code_object_serialize.serialized_code_object_size = serialized_code_object_size;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_code_object_serialize, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_serialize, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_code_object_serialize_fn(code_object, alloc_callback, callback_data, options, serialized_code_object, serialized_code_object_size);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_serialize, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_code_object_deserialize_callback(void* serialized_code_object, size_t serialized_code_object_size, const char* options, hsa_code_object_t* code_object) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_code_object_deserialize.serialized_code_object = serialized_code_object;
  api_data.args.hsa_code_object_deserialize.serialized_code_object_size = serialized_code_object_size;
  api_data.args.hsa_code_object_deserialize.options = options;
  api_data.args.hsa_code_object_deserialize.code_object = code_object;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_code_object_deserialize, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_deserialize, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_code_object_deserialize_fn(serialized_code_object, serialized_code_object_size, options, code_object);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_deserialize, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_code_object_destroy_callback(hsa_code_object_t code_object) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_code_object_destroy.code_object = code_object;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_code_object_destroy, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_destroy, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_code_object_destroy_fn(code_object);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_destroy, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_code_object_get_info_callback(hsa_code_object_t code_object, hsa_code_object_info_t attribute, void* value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_code_object_get_info.code_object = code_object;
  api_data.args.hsa_code_object_get_info.attribute = attribute;
  api_data.args.hsa_code_object_get_info.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_code_object_get_info, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_get_info, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_code_object_get_info_fn(code_object, attribute, value);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_get_info, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_code_object_get_symbol_callback(hsa_code_object_t code_object, const char* symbol_name, hsa_code_symbol_t* symbol) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_code_object_get_symbol.code_object = code_object;
  api_data.args.hsa_code_object_get_symbol.symbol_name = symbol_name;
  api_data.args.hsa_code_object_get_symbol.symbol = symbol;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_code_object_get_symbol, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_get_symbol, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_code_object_get_symbol_fn(code_object, symbol_name, symbol);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_get_symbol, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_code_symbol_get_info_callback(hsa_code_symbol_t code_symbol, hsa_code_symbol_info_t attribute, void* value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_code_symbol_get_info.code_symbol = code_symbol;
  api_data.args.hsa_code_symbol_get_info.attribute = attribute;
  api_data.args.hsa_code_symbol_get_info.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_code_symbol_get_info, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_symbol_get_info, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_code_symbol_get_info_fn(code_symbol, attribute, value);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_symbol_get_info, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_code_object_iterate_symbols_callback(hsa_code_object_t code_object, hsa_status_t (* callback)(hsa_code_object_t code_object, hsa_code_symbol_t symbol, void* data), void* data) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_code_object_iterate_symbols.code_object = code_object;
  api_data.args.hsa_code_object_iterate_symbols.callback = callback;
  api_data.args.hsa_code_object_iterate_symbols.data = data;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_code_object_iterate_symbols, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_iterate_symbols, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_code_object_iterate_symbols_fn(code_object, callback, data);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_iterate_symbols, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_create_callback(hsa_profile_t profile, hsa_executable_state_t executable_state, const char* options, hsa_executable_t* executable) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_create.profile = profile;
  api_data.args.hsa_executable_create.executable_state = executable_state;
  api_data.args.hsa_executable_create.options = options;
  api_data.args.hsa_executable_create.executable = executable;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_create, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_create, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_create_fn(profile, executable_state, options, executable);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_create, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_destroy_callback(hsa_executable_t executable) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_destroy.executable = executable;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_destroy, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_destroy, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_destroy_fn(executable);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_destroy, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_load_code_object_callback(hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_t code_object, const char* options) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_load_code_object.executable = executable;
  api_data.args.hsa_executable_load_code_object.agent = agent;
  api_data.args.hsa_executable_load_code_object.code_object = code_object;
  api_data.args.hsa_executable_load_code_object.options = options;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_load_code_object, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_load_code_object, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_load_code_object_fn(executable, agent, code_object, options);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_load_code_object, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_freeze_callback(hsa_executable_t executable, const char* options) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_freeze.executable = executable;
  api_data.args.hsa_executable_freeze.options = options;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_freeze, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_freeze, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_freeze_fn(executable, options);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_freeze, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_get_info_callback(hsa_executable_t executable, hsa_executable_info_t attribute, void* value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_get_info.executable = executable;
  api_data.args.hsa_executable_get_info.attribute = attribute;
  api_data.args.hsa_executable_get_info.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_get_info, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_get_info, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_get_info_fn(executable, attribute, value);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_get_info, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_global_variable_define_callback(hsa_executable_t executable, const char* variable_name, void* address) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_global_variable_define.executable = executable;
  api_data.args.hsa_executable_global_variable_define.variable_name = variable_name;
  api_data.args.hsa_executable_global_variable_define.address = address;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_global_variable_define, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_global_variable_define, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_global_variable_define_fn(executable, variable_name, address);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_global_variable_define, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_agent_global_variable_define_callback(hsa_executable_t executable, hsa_agent_t agent, const char* variable_name, void* address) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_agent_global_variable_define.executable = executable;
  api_data.args.hsa_executable_agent_global_variable_define.agent = agent;
  api_data.args.hsa_executable_agent_global_variable_define.variable_name = variable_name;
  api_data.args.hsa_executable_agent_global_variable_define.address = address;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_agent_global_variable_define, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_agent_global_variable_define, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_agent_global_variable_define_fn(executable, agent, variable_name, address);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_agent_global_variable_define, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_readonly_variable_define_callback(hsa_executable_t executable, hsa_agent_t agent, const char* variable_name, void* address) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_readonly_variable_define.executable = executable;
  api_data.args.hsa_executable_readonly_variable_define.agent = agent;
  api_data.args.hsa_executable_readonly_variable_define.variable_name = variable_name;
  api_data.args.hsa_executable_readonly_variable_define.address = address;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_readonly_variable_define, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_readonly_variable_define, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_readonly_variable_define_fn(executable, agent, variable_name, address);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_readonly_variable_define, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_validate_callback(hsa_executable_t executable, uint32_t* result) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_validate.executable = executable;
  api_data.args.hsa_executable_validate.result = result;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_validate, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_validate, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_validate_fn(executable, result);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_validate, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_get_symbol_callback(hsa_executable_t executable, const char* module_name, const char* symbol_name, hsa_agent_t agent, int32_t call_convention, hsa_executable_symbol_t* symbol) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_get_symbol.executable = executable;
  api_data.args.hsa_executable_get_symbol.module_name = module_name;
  api_data.args.hsa_executable_get_symbol.symbol_name = symbol_name;
  api_data.args.hsa_executable_get_symbol.agent = agent;
  api_data.args.hsa_executable_get_symbol.call_convention = call_convention;
  api_data.args.hsa_executable_get_symbol.symbol = symbol;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_get_symbol, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_get_symbol, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_get_symbol_fn(executable, module_name, symbol_name, agent, call_convention, symbol);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_get_symbol, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_symbol_get_info_callback(hsa_executable_symbol_t executable_symbol, hsa_executable_symbol_info_t attribute, void* value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_symbol_get_info.executable_symbol = executable_symbol;
  api_data.args.hsa_executable_symbol_get_info.attribute = attribute;
  api_data.args.hsa_executable_symbol_get_info.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_symbol_get_info, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_symbol_get_info, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_symbol_get_info_fn(executable_symbol, attribute, value);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_symbol_get_info, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_iterate_symbols_callback(hsa_executable_t executable, hsa_status_t (* callback)(hsa_executable_t exec, hsa_executable_symbol_t symbol, void* data), void* data) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_iterate_symbols.executable = executable;
  api_data.args.hsa_executable_iterate_symbols.callback = callback;
  api_data.args.hsa_executable_iterate_symbols.data = data;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_iterate_symbols, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_iterate_symbols, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_iterate_symbols_fn(executable, callback, data);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_iterate_symbols, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_status_string_callback(hsa_status_t status, const char** status_string) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_status_string.status = status;
  api_data.args.hsa_status_string.status_string = status_string;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_status_string, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_status_string, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_status_string_fn(status, status_string);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_status_string, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_extension_get_name_callback(uint16_t extension, const char** name) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_extension_get_name.extension = extension;
  api_data.args.hsa_extension_get_name.name = name;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_extension_get_name, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_extension_get_name, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_extension_get_name_fn(extension, name);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_extension_get_name, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_system_major_extension_supported_callback(uint16_t extension, uint16_t version_major, uint16_t* version_minor, bool* result) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_system_major_extension_supported.extension = extension;
  api_data.args.hsa_system_major_extension_supported.version_major = version_major;
  api_data.args.hsa_system_major_extension_supported.version_minor = version_minor;
  api_data.args.hsa_system_major_extension_supported.result = result;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_system_major_extension_supported, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_system_major_extension_supported, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_system_major_extension_supported_fn(extension, version_major, version_minor, result);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_system_major_extension_supported, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_system_get_major_extension_table_callback(uint16_t extension, uint16_t version_major, size_t table_length, void* table) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_system_get_major_extension_table.extension = extension;
  api_data.args.hsa_system_get_major_extension_table.version_major = version_major;
  api_data.args.hsa_system_get_major_extension_table.table_length = table_length;
  api_data.args.hsa_system_get_major_extension_table.table = table;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_system_get_major_extension_table, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_system_get_major_extension_table, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_system_get_major_extension_table_fn(extension, version_major, table_length, table);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_system_get_major_extension_table, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_agent_major_extension_supported_callback(uint16_t extension, hsa_agent_t agent, uint16_t version_major, uint16_t* version_minor, bool* result) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_agent_major_extension_supported.extension = extension;
  api_data.args.hsa_agent_major_extension_supported.agent = agent;
  api_data.args.hsa_agent_major_extension_supported.version_major = version_major;
  api_data.args.hsa_agent_major_extension_supported.version_minor = version_minor;
  api_data.args.hsa_agent_major_extension_supported.result = result;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_agent_major_extension_supported, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_agent_major_extension_supported, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_agent_major_extension_supported_fn(extension, agent, version_major, version_minor, result);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_agent_major_extension_supported, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_cache_get_info_callback(hsa_cache_t cache, hsa_cache_info_t attribute, void* value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_cache_get_info.cache = cache;
  api_data.args.hsa_cache_get_info.attribute = attribute;
  api_data.args.hsa_cache_get_info.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_cache_get_info, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_cache_get_info, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_cache_get_info_fn(cache, attribute, value);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_cache_get_info, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_agent_iterate_caches_callback(hsa_agent_t agent, hsa_status_t (* callback)(hsa_cache_t cache, void* data), void* data) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_agent_iterate_caches.agent = agent;
  api_data.args.hsa_agent_iterate_caches.callback = callback;
  api_data.args.hsa_agent_iterate_caches.data = data;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_agent_iterate_caches, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_agent_iterate_caches, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_agent_iterate_caches_fn(agent, callback, data);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_agent_iterate_caches, &api_data, api_callback_arg);
  return ret;
}
static void hsa_signal_silent_store_relaxed_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_silent_store_relaxed.signal = signal;
  api_data.args.hsa_signal_silent_store_relaxed.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_silent_store_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_silent_store_relaxed, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_silent_store_relaxed_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_silent_store_relaxed, &api_data, api_callback_arg);
}
static void hsa_signal_silent_store_screlease_callback(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_silent_store_screlease.signal = signal;
  api_data.args.hsa_signal_silent_store_screlease.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_silent_store_screlease, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_silent_store_screlease, &api_data, api_callback_arg);
  CoreApiTable_saved.hsa_signal_silent_store_screlease_fn(signal, value);
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_silent_store_screlease, &api_data, api_callback_arg);
}
static hsa_status_t hsa_signal_group_create_callback(uint32_t num_signals, const hsa_signal_t* signals, uint32_t num_consumers, const hsa_agent_t* consumers, hsa_signal_group_t* signal_group) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_group_create.num_signals = num_signals;
  api_data.args.hsa_signal_group_create.signals = signals;
  api_data.args.hsa_signal_group_create.num_consumers = num_consumers;
  api_data.args.hsa_signal_group_create.consumers = consumers;
  api_data.args.hsa_signal_group_create.signal_group = signal_group;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_group_create, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_group_create, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_signal_group_create_fn(num_signals, signals, num_consumers, consumers, signal_group);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_group_create, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_signal_group_destroy_callback(hsa_signal_group_t signal_group) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_group_destroy.signal_group = signal_group;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_group_destroy, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_group_destroy, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_signal_group_destroy_fn(signal_group);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_group_destroy, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_signal_group_wait_any_scacquire_callback(hsa_signal_group_t signal_group, const hsa_signal_condition_t* conditions, const hsa_signal_value_t* compare_values, hsa_wait_state_t wait_state_hint, hsa_signal_t* signal, hsa_signal_value_t* value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_group_wait_any_scacquire.signal_group = signal_group;
  api_data.args.hsa_signal_group_wait_any_scacquire.conditions = conditions;
  api_data.args.hsa_signal_group_wait_any_scacquire.compare_values = compare_values;
  api_data.args.hsa_signal_group_wait_any_scacquire.wait_state_hint = wait_state_hint;
  api_data.args.hsa_signal_group_wait_any_scacquire.signal = signal;
  api_data.args.hsa_signal_group_wait_any_scacquire.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_group_wait_any_scacquire, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_group_wait_any_scacquire, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_signal_group_wait_any_scacquire_fn(signal_group, conditions, compare_values, wait_state_hint, signal, value);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_group_wait_any_scacquire, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_signal_group_wait_any_relaxed_callback(hsa_signal_group_t signal_group, const hsa_signal_condition_t* conditions, const hsa_signal_value_t* compare_values, hsa_wait_state_t wait_state_hint, hsa_signal_t* signal, hsa_signal_value_t* value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_signal_group_wait_any_relaxed.signal_group = signal_group;
  api_data.args.hsa_signal_group_wait_any_relaxed.conditions = conditions;
  api_data.args.hsa_signal_group_wait_any_relaxed.compare_values = compare_values;
  api_data.args.hsa_signal_group_wait_any_relaxed.wait_state_hint = wait_state_hint;
  api_data.args.hsa_signal_group_wait_any_relaxed.signal = signal;
  api_data.args.hsa_signal_group_wait_any_relaxed.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_signal_group_wait_any_relaxed, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_group_wait_any_relaxed, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_signal_group_wait_any_relaxed_fn(signal_group, conditions, compare_values, wait_state_hint, signal, value);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_signal_group_wait_any_relaxed, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_agent_iterate_isas_callback(hsa_agent_t agent, hsa_status_t (* callback)(hsa_isa_t isa, void* data), void* data) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_agent_iterate_isas.agent = agent;
  api_data.args.hsa_agent_iterate_isas.callback = callback;
  api_data.args.hsa_agent_iterate_isas.data = data;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_agent_iterate_isas, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_agent_iterate_isas, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_agent_iterate_isas_fn(agent, callback, data);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_agent_iterate_isas, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_isa_get_info_alt_callback(hsa_isa_t isa, hsa_isa_info_t attribute, void* value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_isa_get_info_alt.isa = isa;
  api_data.args.hsa_isa_get_info_alt.attribute = attribute;
  api_data.args.hsa_isa_get_info_alt.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_isa_get_info_alt, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_isa_get_info_alt, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_isa_get_info_alt_fn(isa, attribute, value);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_isa_get_info_alt, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_isa_get_exception_policies_callback(hsa_isa_t isa, hsa_profile_t profile, uint16_t* mask) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_isa_get_exception_policies.isa = isa;
  api_data.args.hsa_isa_get_exception_policies.profile = profile;
  api_data.args.hsa_isa_get_exception_policies.mask = mask;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_isa_get_exception_policies, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_isa_get_exception_policies, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_isa_get_exception_policies_fn(isa, profile, mask);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_isa_get_exception_policies, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_isa_get_round_method_callback(hsa_isa_t isa, hsa_fp_type_t fp_type, hsa_flush_mode_t flush_mode, hsa_round_method_t* round_method) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_isa_get_round_method.isa = isa;
  api_data.args.hsa_isa_get_round_method.fp_type = fp_type;
  api_data.args.hsa_isa_get_round_method.flush_mode = flush_mode;
  api_data.args.hsa_isa_get_round_method.round_method = round_method;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_isa_get_round_method, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_isa_get_round_method, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_isa_get_round_method_fn(isa, fp_type, flush_mode, round_method);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_isa_get_round_method, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_wavefront_get_info_callback(hsa_wavefront_t wavefront, hsa_wavefront_info_t attribute, void* value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_wavefront_get_info.wavefront = wavefront;
  api_data.args.hsa_wavefront_get_info.attribute = attribute;
  api_data.args.hsa_wavefront_get_info.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_wavefront_get_info, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_wavefront_get_info, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_wavefront_get_info_fn(wavefront, attribute, value);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_wavefront_get_info, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_isa_iterate_wavefronts_callback(hsa_isa_t isa, hsa_status_t (* callback)(hsa_wavefront_t wavefront, void* data), void* data) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_isa_iterate_wavefronts.isa = isa;
  api_data.args.hsa_isa_iterate_wavefronts.callback = callback;
  api_data.args.hsa_isa_iterate_wavefronts.data = data;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_isa_iterate_wavefronts, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_isa_iterate_wavefronts, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_isa_iterate_wavefronts_fn(isa, callback, data);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_isa_iterate_wavefronts, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_code_object_get_symbol_from_name_callback(hsa_code_object_t code_object, const char* module_name, const char* symbol_name, hsa_code_symbol_t* symbol) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_code_object_get_symbol_from_name.code_object = code_object;
  api_data.args.hsa_code_object_get_symbol_from_name.module_name = module_name;
  api_data.args.hsa_code_object_get_symbol_from_name.symbol_name = symbol_name;
  api_data.args.hsa_code_object_get_symbol_from_name.symbol = symbol;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_code_object_get_symbol_from_name, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_get_symbol_from_name, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_code_object_get_symbol_from_name_fn(code_object, module_name, symbol_name, symbol);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_get_symbol_from_name, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_code_object_reader_create_from_file_callback(hsa_file_t file, hsa_code_object_reader_t* code_object_reader) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_code_object_reader_create_from_file.file = file;
  api_data.args.hsa_code_object_reader_create_from_file.code_object_reader = code_object_reader;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_code_object_reader_create_from_file, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_reader_create_from_file, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_code_object_reader_create_from_file_fn(file, code_object_reader);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_reader_create_from_file, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_code_object_reader_create_from_memory_callback(const void* code_object, size_t size, hsa_code_object_reader_t* code_object_reader) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_code_object_reader_create_from_memory.code_object = code_object;
  api_data.args.hsa_code_object_reader_create_from_memory.size = size;
  api_data.args.hsa_code_object_reader_create_from_memory.code_object_reader = code_object_reader;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_code_object_reader_create_from_memory, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_reader_create_from_memory, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_code_object_reader_create_from_memory_fn(code_object, size, code_object_reader);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_reader_create_from_memory, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_code_object_reader_destroy_callback(hsa_code_object_reader_t code_object_reader) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_code_object_reader_destroy.code_object_reader = code_object_reader;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_code_object_reader_destroy, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_reader_destroy, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_code_object_reader_destroy_fn(code_object_reader);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_code_object_reader_destroy, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_create_alt_callback(hsa_profile_t profile, hsa_default_float_rounding_mode_t default_float_rounding_mode, const char* options, hsa_executable_t* executable) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_create_alt.profile = profile;
  api_data.args.hsa_executable_create_alt.default_float_rounding_mode = default_float_rounding_mode;
  api_data.args.hsa_executable_create_alt.options = options;
  api_data.args.hsa_executable_create_alt.executable = executable;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_create_alt, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_create_alt, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_create_alt_fn(profile, default_float_rounding_mode, options, executable);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_create_alt, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_load_program_code_object_callback(hsa_executable_t executable, hsa_code_object_reader_t code_object_reader, const char* options, hsa_loaded_code_object_t* loaded_code_object) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_load_program_code_object.executable = executable;
  api_data.args.hsa_executable_load_program_code_object.code_object_reader = code_object_reader;
  api_data.args.hsa_executable_load_program_code_object.options = options;
  api_data.args.hsa_executable_load_program_code_object.loaded_code_object = loaded_code_object;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_load_program_code_object, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_load_program_code_object, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_load_program_code_object_fn(executable, code_object_reader, options, loaded_code_object);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_load_program_code_object, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_load_agent_code_object_callback(hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader, const char* options, hsa_loaded_code_object_t* loaded_code_object) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_load_agent_code_object.executable = executable;
  api_data.args.hsa_executable_load_agent_code_object.agent = agent;
  api_data.args.hsa_executable_load_agent_code_object.code_object_reader = code_object_reader;
  api_data.args.hsa_executable_load_agent_code_object.options = options;
  api_data.args.hsa_executable_load_agent_code_object.loaded_code_object = loaded_code_object;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_load_agent_code_object, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_load_agent_code_object, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_load_agent_code_object_fn(executable, agent, code_object_reader, options, loaded_code_object);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_load_agent_code_object, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_validate_alt_callback(hsa_executable_t executable, const char* options, uint32_t* result) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_validate_alt.executable = executable;
  api_data.args.hsa_executable_validate_alt.options = options;
  api_data.args.hsa_executable_validate_alt.result = result;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_validate_alt, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_validate_alt, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_validate_alt_fn(executable, options, result);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_validate_alt, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_get_symbol_by_name_callback(hsa_executable_t executable, const char* symbol_name, const hsa_agent_t* agent, hsa_executable_symbol_t* symbol) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_get_symbol_by_name.executable = executable;
  api_data.args.hsa_executable_get_symbol_by_name.symbol_name = symbol_name;
  api_data.args.hsa_executable_get_symbol_by_name.agent = agent;
  api_data.args.hsa_executable_get_symbol_by_name.symbol = symbol;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_get_symbol_by_name, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_get_symbol_by_name, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_get_symbol_by_name_fn(executable, symbol_name, agent, symbol);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_get_symbol_by_name, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_iterate_agent_symbols_callback(hsa_executable_t executable, hsa_agent_t agent, hsa_status_t (* callback)(hsa_executable_t exec, hsa_agent_t agent, hsa_executable_symbol_t symbol, void* data), void* data) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_iterate_agent_symbols.executable = executable;
  api_data.args.hsa_executable_iterate_agent_symbols.agent = agent;
  api_data.args.hsa_executable_iterate_agent_symbols.callback = callback;
  api_data.args.hsa_executable_iterate_agent_symbols.data = data;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_iterate_agent_symbols, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_iterate_agent_symbols, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_iterate_agent_symbols_fn(executable, agent, callback, data);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_iterate_agent_symbols, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_executable_iterate_program_symbols_callback(hsa_executable_t executable, hsa_status_t (* callback)(hsa_executable_t exec, hsa_executable_symbol_t symbol, void* data), void* data) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_executable_iterate_program_symbols.executable = executable;
  api_data.args.hsa_executable_iterate_program_symbols.callback = callback;
  api_data.args.hsa_executable_iterate_program_symbols.data = data;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_executable_iterate_program_symbols, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_iterate_program_symbols, &api_data, api_callback_arg);
  hsa_status_t ret =  CoreApiTable_saved.hsa_executable_iterate_program_symbols_fn(executable, callback, data);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_executable_iterate_program_symbols, &api_data, api_callback_arg);
  return ret;
}

// block: AmdExtTable API
static hsa_status_t hsa_amd_coherency_get_type_callback(hsa_agent_t agent, hsa_amd_coherency_type_t* type) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_coherency_get_type.agent = agent;
  api_data.args.hsa_amd_coherency_get_type.type = type;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_coherency_get_type, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_coherency_get_type, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_coherency_get_type_fn(agent, type);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_coherency_get_type, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_coherency_set_type_callback(hsa_agent_t agent, hsa_amd_coherency_type_t type) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_coherency_set_type.agent = agent;
  api_data.args.hsa_amd_coherency_set_type.type = type;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_coherency_set_type, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_coherency_set_type, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_coherency_set_type_fn(agent, type);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_coherency_set_type, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_profiling_set_profiler_enabled_callback(hsa_queue_t* queue, int enable) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_profiling_set_profiler_enabled.queue = queue;
  api_data.args.hsa_amd_profiling_set_profiler_enabled.enable = enable;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_profiling_set_profiler_enabled, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_profiling_set_profiler_enabled, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_profiling_set_profiler_enabled_fn(queue, enable);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_profiling_set_profiler_enabled, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_profiling_async_copy_enable_callback(bool enable) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_profiling_async_copy_enable.enable = enable;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_profiling_async_copy_enable, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_profiling_async_copy_enable, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_profiling_async_copy_enable_fn(enable);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_profiling_async_copy_enable, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_profiling_get_dispatch_time_callback(hsa_agent_t agent, hsa_signal_t signal, hsa_amd_profiling_dispatch_time_t* time) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_profiling_get_dispatch_time.agent = agent;
  api_data.args.hsa_amd_profiling_get_dispatch_time.signal = signal;
  api_data.args.hsa_amd_profiling_get_dispatch_time.time = time;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_profiling_get_dispatch_time, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_profiling_get_dispatch_time, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_profiling_get_dispatch_time_fn(agent, signal, time);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_profiling_get_dispatch_time, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_profiling_get_async_copy_time_callback(hsa_signal_t signal, hsa_amd_profiling_async_copy_time_t* time) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_profiling_get_async_copy_time.signal = signal;
  api_data.args.hsa_amd_profiling_get_async_copy_time.time = time;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_profiling_get_async_copy_time, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_profiling_get_async_copy_time, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_profiling_get_async_copy_time_fn(signal, time);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_profiling_get_async_copy_time, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_profiling_convert_tick_to_system_domain_callback(hsa_agent_t agent, uint64_t agent_tick, uint64_t* system_tick) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_profiling_convert_tick_to_system_domain.agent = agent;
  api_data.args.hsa_amd_profiling_convert_tick_to_system_domain.agent_tick = agent_tick;
  api_data.args.hsa_amd_profiling_convert_tick_to_system_domain.system_tick = system_tick;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_profiling_convert_tick_to_system_domain, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_profiling_convert_tick_to_system_domain, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_profiling_convert_tick_to_system_domain_fn(agent, agent_tick, system_tick);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_profiling_convert_tick_to_system_domain, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_signal_async_handler_callback(hsa_signal_t signal, hsa_signal_condition_t cond, hsa_signal_value_t value, hsa_amd_signal_handler handler, void* arg) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_signal_async_handler.signal = signal;
  api_data.args.hsa_amd_signal_async_handler.cond = cond;
  api_data.args.hsa_amd_signal_async_handler.value = value;
  api_data.args.hsa_amd_signal_async_handler.handler = handler;
  api_data.args.hsa_amd_signal_async_handler.arg = arg;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_signal_async_handler, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_signal_async_handler, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_signal_async_handler_fn(signal, cond, value, handler, arg);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_signal_async_handler, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_async_function_callback(void (* callback)(void* arg), void* arg) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_async_function.callback = callback;
  api_data.args.hsa_amd_async_function.arg = arg;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_async_function, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_async_function, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_async_function_fn(callback, arg);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_async_function, &api_data, api_callback_arg);
  return ret;
}
static uint32_t hsa_amd_signal_wait_any_callback(uint32_t signal_count, hsa_signal_t* signals, hsa_signal_condition_t* conds, hsa_signal_value_t* values, uint64_t timeout_hint, hsa_wait_state_t wait_hint, hsa_signal_value_t* satisfying_value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_signal_wait_any.signal_count = signal_count;
  api_data.args.hsa_amd_signal_wait_any.signals = signals;
  api_data.args.hsa_amd_signal_wait_any.conds = conds;
  api_data.args.hsa_amd_signal_wait_any.values = values;
  api_data.args.hsa_amd_signal_wait_any.timeout_hint = timeout_hint;
  api_data.args.hsa_amd_signal_wait_any.wait_hint = wait_hint;
  api_data.args.hsa_amd_signal_wait_any.satisfying_value = satisfying_value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_signal_wait_any, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_signal_wait_any, &api_data, api_callback_arg);
  uint32_t ret =  AmdExtTable_saved.hsa_amd_signal_wait_any_fn(signal_count, signals, conds, values, timeout_hint, wait_hint, satisfying_value);
  api_data.uint32_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_signal_wait_any, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_queue_cu_set_mask_callback(const hsa_queue_t* queue, uint32_t num_cu_mask_count, const uint32_t* cu_mask) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_queue_cu_set_mask.queue = queue;
  api_data.args.hsa_amd_queue_cu_set_mask.num_cu_mask_count = num_cu_mask_count;
  api_data.args.hsa_amd_queue_cu_set_mask.cu_mask = cu_mask;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_queue_cu_set_mask, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_queue_cu_set_mask, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_queue_cu_set_mask_fn(queue, num_cu_mask_count, cu_mask);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_queue_cu_set_mask, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_memory_pool_get_info_callback(hsa_amd_memory_pool_t memory_pool, hsa_amd_memory_pool_info_t attribute, void* value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_memory_pool_get_info.memory_pool = memory_pool;
  api_data.args.hsa_amd_memory_pool_get_info.attribute = attribute;
  api_data.args.hsa_amd_memory_pool_get_info.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_memory_pool_get_info, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_pool_get_info, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_memory_pool_get_info_fn(memory_pool, attribute, value);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_pool_get_info, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_agent_iterate_memory_pools_callback(hsa_agent_t agent, hsa_status_t (* callback)(hsa_amd_memory_pool_t memory_pool, void* data), void* data) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_agent_iterate_memory_pools.agent = agent;
  api_data.args.hsa_amd_agent_iterate_memory_pools.callback = callback;
  api_data.args.hsa_amd_agent_iterate_memory_pools.data = data;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_agent_iterate_memory_pools, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_agent_iterate_memory_pools, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_agent_iterate_memory_pools_fn(agent, callback, data);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_agent_iterate_memory_pools, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_memory_pool_allocate_callback(hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags, void** ptr) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_memory_pool_allocate.memory_pool = memory_pool;
  api_data.args.hsa_amd_memory_pool_allocate.size = size;
  api_data.args.hsa_amd_memory_pool_allocate.flags = flags;
  api_data.args.hsa_amd_memory_pool_allocate.ptr = ptr;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_memory_pool_allocate, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_pool_allocate, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_memory_pool_allocate_fn(memory_pool, size, flags, ptr);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_pool_allocate, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_memory_pool_free_callback(void* ptr) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_memory_pool_free.ptr = ptr;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_memory_pool_free, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_pool_free, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_memory_pool_free_fn(ptr);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_pool_free, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_memory_async_copy_callback(void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent, size_t size, uint32_t num_dep_signals, const hsa_signal_t* dep_signals, hsa_signal_t completion_signal) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_memory_async_copy.dst = dst;
  api_data.args.hsa_amd_memory_async_copy.dst_agent = dst_agent;
  api_data.args.hsa_amd_memory_async_copy.src = src;
  api_data.args.hsa_amd_memory_async_copy.src_agent = src_agent;
  api_data.args.hsa_amd_memory_async_copy.size = size;
  api_data.args.hsa_amd_memory_async_copy.num_dep_signals = num_dep_signals;
  api_data.args.hsa_amd_memory_async_copy.dep_signals = dep_signals;
  api_data.args.hsa_amd_memory_async_copy.completion_signal = completion_signal;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_memory_async_copy, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_async_copy, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_memory_async_copy_fn(dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, completion_signal);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_async_copy, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_agent_memory_pool_get_info_callback(hsa_agent_t agent, hsa_amd_memory_pool_t memory_pool, hsa_amd_agent_memory_pool_info_t attribute, void* value) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_agent_memory_pool_get_info.agent = agent;
  api_data.args.hsa_amd_agent_memory_pool_get_info.memory_pool = memory_pool;
  api_data.args.hsa_amd_agent_memory_pool_get_info.attribute = attribute;
  api_data.args.hsa_amd_agent_memory_pool_get_info.value = value;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_agent_memory_pool_get_info, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_agent_memory_pool_get_info, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_agent_memory_pool_get_info_fn(agent, memory_pool, attribute, value);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_agent_memory_pool_get_info, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_agents_allow_access_callback(uint32_t num_agents, const hsa_agent_t* agents, const uint32_t* flags, const void* ptr) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_agents_allow_access.num_agents = num_agents;
  api_data.args.hsa_amd_agents_allow_access.agents = agents;
  api_data.args.hsa_amd_agents_allow_access.flags = flags;
  api_data.args.hsa_amd_agents_allow_access.ptr = ptr;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_agents_allow_access, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_agents_allow_access, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_agents_allow_access_fn(num_agents, agents, flags, ptr);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_agents_allow_access, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_memory_pool_can_migrate_callback(hsa_amd_memory_pool_t src_memory_pool, hsa_amd_memory_pool_t dst_memory_pool, bool* result) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_memory_pool_can_migrate.src_memory_pool = src_memory_pool;
  api_data.args.hsa_amd_memory_pool_can_migrate.dst_memory_pool = dst_memory_pool;
  api_data.args.hsa_amd_memory_pool_can_migrate.result = result;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_memory_pool_can_migrate, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_pool_can_migrate, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_memory_pool_can_migrate_fn(src_memory_pool, dst_memory_pool, result);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_pool_can_migrate, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_memory_migrate_callback(const void* ptr, hsa_amd_memory_pool_t memory_pool, uint32_t flags) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_memory_migrate.ptr = ptr;
  api_data.args.hsa_amd_memory_migrate.memory_pool = memory_pool;
  api_data.args.hsa_amd_memory_migrate.flags = flags;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_memory_migrate, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_migrate, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_memory_migrate_fn(ptr, memory_pool, flags);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_migrate, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_memory_lock_callback(void* host_ptr, size_t size, hsa_agent_t* agents, int num_agent, void** agent_ptr) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_memory_lock.host_ptr = host_ptr;
  api_data.args.hsa_amd_memory_lock.size = size;
  api_data.args.hsa_amd_memory_lock.agents = agents;
  api_data.args.hsa_amd_memory_lock.num_agent = num_agent;
  api_data.args.hsa_amd_memory_lock.agent_ptr = agent_ptr;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_memory_lock, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_lock, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_memory_lock_fn(host_ptr, size, agents, num_agent, agent_ptr);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_lock, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_memory_unlock_callback(void* host_ptr) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_memory_unlock.host_ptr = host_ptr;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_memory_unlock, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_unlock, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_memory_unlock_fn(host_ptr);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_unlock, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_memory_fill_callback(void* ptr, uint32_t value, size_t count) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_memory_fill.ptr = ptr;
  api_data.args.hsa_amd_memory_fill.value = value;
  api_data.args.hsa_amd_memory_fill.count = count;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_memory_fill, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_fill, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_memory_fill_fn(ptr, value, count);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_fill, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_interop_map_buffer_callback(uint32_t num_agents, hsa_agent_t* agents, int interop_handle, uint32_t flags, size_t* size, void** ptr, size_t* metadata_size, const void** metadata) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_interop_map_buffer.num_agents = num_agents;
  api_data.args.hsa_amd_interop_map_buffer.agents = agents;
  api_data.args.hsa_amd_interop_map_buffer.interop_handle = interop_handle;
  api_data.args.hsa_amd_interop_map_buffer.flags = flags;
  api_data.args.hsa_amd_interop_map_buffer.size = size;
  api_data.args.hsa_amd_interop_map_buffer.ptr = ptr;
  api_data.args.hsa_amd_interop_map_buffer.metadata_size = metadata_size;
  api_data.args.hsa_amd_interop_map_buffer.metadata = metadata;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_interop_map_buffer, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_interop_map_buffer, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_interop_map_buffer_fn(num_agents, agents, interop_handle, flags, size, ptr, metadata_size, metadata);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_interop_map_buffer, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_interop_unmap_buffer_callback(void* ptr) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_interop_unmap_buffer.ptr = ptr;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_interop_unmap_buffer, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_interop_unmap_buffer, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_interop_unmap_buffer_fn(ptr);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_interop_unmap_buffer, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_image_create_callback(hsa_agent_t agent, const hsa_ext_image_descriptor_t* image_descriptor, const hsa_amd_image_descriptor_t* image_layout, const void* image_data, hsa_access_permission_t access_permission, hsa_ext_image_t* image) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_image_create.agent = agent;
  api_data.args.hsa_amd_image_create.image_descriptor = image_descriptor;
  api_data.args.hsa_amd_image_create.image_layout = image_layout;
  api_data.args.hsa_amd_image_create.image_data = image_data;
  api_data.args.hsa_amd_image_create.access_permission = access_permission;
  api_data.args.hsa_amd_image_create.image = image;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_image_create, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_image_create, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_image_create_fn(agent, image_descriptor, image_layout, image_data, access_permission, image);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_image_create, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_pointer_info_callback(void* ptr, hsa_amd_pointer_info_t* info, void* (* alloc)(size_t), uint32_t* num_agents_accessible, hsa_agent_t** accessible) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_pointer_info.ptr = ptr;
  api_data.args.hsa_amd_pointer_info.info = info;
  api_data.args.hsa_amd_pointer_info.alloc = alloc;
  api_data.args.hsa_amd_pointer_info.num_agents_accessible = num_agents_accessible;
  api_data.args.hsa_amd_pointer_info.accessible = accessible;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_pointer_info, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_pointer_info, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_pointer_info_fn(ptr, info, alloc, num_agents_accessible, accessible);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_pointer_info, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_pointer_info_set_userdata_callback(void* ptr, void* userdata) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_pointer_info_set_userdata.ptr = ptr;
  api_data.args.hsa_amd_pointer_info_set_userdata.userdata = userdata;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_pointer_info_set_userdata, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_pointer_info_set_userdata, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_pointer_info_set_userdata_fn(ptr, userdata);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_pointer_info_set_userdata, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_ipc_memory_create_callback(void* ptr, size_t len, hsa_amd_ipc_memory_t* handle) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_ipc_memory_create.ptr = ptr;
  api_data.args.hsa_amd_ipc_memory_create.len = len;
  api_data.args.hsa_amd_ipc_memory_create.handle = handle;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_ipc_memory_create, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_ipc_memory_create, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_ipc_memory_create_fn(ptr, len, handle);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_ipc_memory_create, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_ipc_memory_attach_callback(const hsa_amd_ipc_memory_t* handle, size_t len, uint32_t num_agents, const hsa_agent_t* mapping_agents, void** mapped_ptr) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_ipc_memory_attach.handle = handle;
  api_data.args.hsa_amd_ipc_memory_attach.len = len;
  api_data.args.hsa_amd_ipc_memory_attach.num_agents = num_agents;
  api_data.args.hsa_amd_ipc_memory_attach.mapping_agents = mapping_agents;
  api_data.args.hsa_amd_ipc_memory_attach.mapped_ptr = mapped_ptr;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_ipc_memory_attach, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_ipc_memory_attach, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_ipc_memory_attach_fn(handle, len, num_agents, mapping_agents, mapped_ptr);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_ipc_memory_attach, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_ipc_memory_detach_callback(void* mapped_ptr) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_ipc_memory_detach.mapped_ptr = mapped_ptr;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_ipc_memory_detach, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_ipc_memory_detach, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_ipc_memory_detach_fn(mapped_ptr);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_ipc_memory_detach, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_signal_create_callback(hsa_signal_value_t initial_value, uint32_t num_consumers, const hsa_agent_t* consumers, uint64_t attributes, hsa_signal_t* signal) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_signal_create.initial_value = initial_value;
  api_data.args.hsa_amd_signal_create.num_consumers = num_consumers;
  api_data.args.hsa_amd_signal_create.consumers = consumers;
  api_data.args.hsa_amd_signal_create.attributes = attributes;
  api_data.args.hsa_amd_signal_create.signal = signal;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_signal_create, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_signal_create, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_signal_create_fn(initial_value, num_consumers, consumers, attributes, signal);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_signal_create, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_ipc_signal_create_callback(hsa_signal_t signal, hsa_amd_ipc_signal_t* handle) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_ipc_signal_create.signal = signal;
  api_data.args.hsa_amd_ipc_signal_create.handle = handle;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_ipc_signal_create, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_ipc_signal_create, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_ipc_signal_create_fn(signal, handle);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_ipc_signal_create, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_ipc_signal_attach_callback(const hsa_amd_ipc_signal_t* handle, hsa_signal_t* signal) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_ipc_signal_attach.handle = handle;
  api_data.args.hsa_amd_ipc_signal_attach.signal = signal;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_ipc_signal_attach, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_ipc_signal_attach, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_ipc_signal_attach_fn(handle, signal);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_ipc_signal_attach, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_register_system_event_handler_callback(hsa_amd_system_event_callback_t callback, void* data) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_register_system_event_handler.callback = callback;
  api_data.args.hsa_amd_register_system_event_handler.data = data;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_register_system_event_handler, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_register_system_event_handler, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_register_system_event_handler_fn(callback, data);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_register_system_event_handler, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_queue_intercept_create_callback(hsa_agent_t agent_handle, uint32_t size, hsa_queue_type32_t type, void (* callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data, uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_queue_intercept_create.agent_handle = agent_handle;
  api_data.args.hsa_amd_queue_intercept_create.size = size;
  api_data.args.hsa_amd_queue_intercept_create.type = type;
  api_data.args.hsa_amd_queue_intercept_create.callback = callback;
  api_data.args.hsa_amd_queue_intercept_create.data = data;
  api_data.args.hsa_amd_queue_intercept_create.private_segment_size = private_segment_size;
  api_data.args.hsa_amd_queue_intercept_create.group_segment_size = group_segment_size;
  api_data.args.hsa_amd_queue_intercept_create.queue = queue;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_queue_intercept_create, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_queue_intercept_create, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_queue_intercept_create_fn(agent_handle, size, type, callback, data, private_segment_size, group_segment_size, queue);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_queue_intercept_create, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_queue_intercept_register_callback(hsa_queue_t* queue, hsa_amd_queue_intercept_handler callback, void* user_data) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_queue_intercept_register.queue = queue;
  api_data.args.hsa_amd_queue_intercept_register.callback = callback;
  api_data.args.hsa_amd_queue_intercept_register.user_data = user_data;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_queue_intercept_register, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_queue_intercept_register, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_queue_intercept_register_fn(queue, callback, user_data);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_queue_intercept_register, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_queue_set_priority_callback(hsa_queue_t* queue, hsa_amd_queue_priority_t priority) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_queue_set_priority.queue = queue;
  api_data.args.hsa_amd_queue_set_priority.priority = priority;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_queue_set_priority, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_queue_set_priority, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_queue_set_priority_fn(queue, priority);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_queue_set_priority, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_memory_async_copy_rect_callback(const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset, const hsa_pitched_ptr_t* src, const hsa_dim3_t* src_offset, const hsa_dim3_t* range, hsa_agent_t copy_agent, hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t* dep_signals, hsa_signal_t completion_signal) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_memory_async_copy_rect.dst = dst;
  api_data.args.hsa_amd_memory_async_copy_rect.dst_offset = dst_offset;
  api_data.args.hsa_amd_memory_async_copy_rect.src = src;
  api_data.args.hsa_amd_memory_async_copy_rect.src_offset = src_offset;
  api_data.args.hsa_amd_memory_async_copy_rect.range = range;
  api_data.args.hsa_amd_memory_async_copy_rect.copy_agent = copy_agent;
  api_data.args.hsa_amd_memory_async_copy_rect.dir = dir;
  api_data.args.hsa_amd_memory_async_copy_rect.num_dep_signals = num_dep_signals;
  api_data.args.hsa_amd_memory_async_copy_rect.dep_signals = dep_signals;
  api_data.args.hsa_amd_memory_async_copy_rect.completion_signal = completion_signal;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_memory_async_copy_rect, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_async_copy_rect, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_memory_async_copy_rect_fn(dst, dst_offset, src, src_offset, range, copy_agent, dir, num_dep_signals, dep_signals, completion_signal);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_memory_async_copy_rect, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_amd_runtime_queue_create_register_callback(hsa_amd_runtime_queue_notifier callback, void* user_data) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_amd_runtime_queue_create_register.callback = callback;
  api_data.args.hsa_amd_runtime_queue_create_register.user_data = user_data;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_amd_runtime_queue_create_register, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_runtime_queue_create_register, &api_data, api_callback_arg);
  hsa_status_t ret =  AmdExtTable_saved.hsa_amd_runtime_queue_create_register_fn(callback, user_data);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_amd_runtime_queue_create_register, &api_data, api_callback_arg);
  return ret;
}

// block: ImageExtTable API
static hsa_status_t hsa_ext_image_get_capability_callback(hsa_agent_t agent, hsa_ext_image_geometry_t geometry, const hsa_ext_image_format_t* image_format, uint32_t* capability_mask) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_ext_image_get_capability.agent = agent;
  api_data.args.hsa_ext_image_get_capability.geometry = geometry;
  api_data.args.hsa_ext_image_get_capability.image_format = image_format;
  api_data.args.hsa_ext_image_get_capability.capability_mask = capability_mask;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_ext_image_get_capability, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_get_capability, &api_data, api_callback_arg);
  hsa_status_t ret =  ImageExtTable_saved.hsa_ext_image_get_capability_fn(agent, geometry, image_format, capability_mask);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_get_capability, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_ext_image_data_get_info_callback(hsa_agent_t agent, const hsa_ext_image_descriptor_t* image_descriptor, hsa_access_permission_t access_permission, hsa_ext_image_data_info_t* image_data_info) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_ext_image_data_get_info.agent = agent;
  api_data.args.hsa_ext_image_data_get_info.image_descriptor = image_descriptor;
  api_data.args.hsa_ext_image_data_get_info.access_permission = access_permission;
  api_data.args.hsa_ext_image_data_get_info.image_data_info = image_data_info;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_ext_image_data_get_info, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_data_get_info, &api_data, api_callback_arg);
  hsa_status_t ret =  ImageExtTable_saved.hsa_ext_image_data_get_info_fn(agent, image_descriptor, access_permission, image_data_info);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_data_get_info, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_ext_image_create_callback(hsa_agent_t agent, const hsa_ext_image_descriptor_t* image_descriptor, const void* image_data, hsa_access_permission_t access_permission, hsa_ext_image_t* image) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_ext_image_create.agent = agent;
  api_data.args.hsa_ext_image_create.image_descriptor = image_descriptor;
  api_data.args.hsa_ext_image_create.image_data = image_data;
  api_data.args.hsa_ext_image_create.access_permission = access_permission;
  api_data.args.hsa_ext_image_create.image = image;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_ext_image_create, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_create, &api_data, api_callback_arg);
  hsa_status_t ret =  ImageExtTable_saved.hsa_ext_image_create_fn(agent, image_descriptor, image_data, access_permission, image);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_create, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_ext_image_import_callback(hsa_agent_t agent, const void* src_memory, size_t src_row_pitch, size_t src_slice_pitch, hsa_ext_image_t dst_image, const hsa_ext_image_region_t* image_region) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_ext_image_import.agent = agent;
  api_data.args.hsa_ext_image_import.src_memory = src_memory;
  api_data.args.hsa_ext_image_import.src_row_pitch = src_row_pitch;
  api_data.args.hsa_ext_image_import.src_slice_pitch = src_slice_pitch;
  api_data.args.hsa_ext_image_import.dst_image = dst_image;
  api_data.args.hsa_ext_image_import.image_region = image_region;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_ext_image_import, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_import, &api_data, api_callback_arg);
  hsa_status_t ret =  ImageExtTable_saved.hsa_ext_image_import_fn(agent, src_memory, src_row_pitch, src_slice_pitch, dst_image, image_region);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_import, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_ext_image_export_callback(hsa_agent_t agent, hsa_ext_image_t src_image, void* dst_memory, size_t dst_row_pitch, size_t dst_slice_pitch, const hsa_ext_image_region_t* image_region) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_ext_image_export.agent = agent;
  api_data.args.hsa_ext_image_export.src_image = src_image;
  api_data.args.hsa_ext_image_export.dst_memory = dst_memory;
  api_data.args.hsa_ext_image_export.dst_row_pitch = dst_row_pitch;
  api_data.args.hsa_ext_image_export.dst_slice_pitch = dst_slice_pitch;
  api_data.args.hsa_ext_image_export.image_region = image_region;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_ext_image_export, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_export, &api_data, api_callback_arg);
  hsa_status_t ret =  ImageExtTable_saved.hsa_ext_image_export_fn(agent, src_image, dst_memory, dst_row_pitch, dst_slice_pitch, image_region);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_export, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_ext_image_copy_callback(hsa_agent_t agent, hsa_ext_image_t src_image, const hsa_dim3_t* src_offset, hsa_ext_image_t dst_image, const hsa_dim3_t* dst_offset, const hsa_dim3_t* range) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_ext_image_copy.agent = agent;
  api_data.args.hsa_ext_image_copy.src_image = src_image;
  api_data.args.hsa_ext_image_copy.src_offset = src_offset;
  api_data.args.hsa_ext_image_copy.dst_image = dst_image;
  api_data.args.hsa_ext_image_copy.dst_offset = dst_offset;
  api_data.args.hsa_ext_image_copy.range = range;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_ext_image_copy, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_copy, &api_data, api_callback_arg);
  hsa_status_t ret =  ImageExtTable_saved.hsa_ext_image_copy_fn(agent, src_image, src_offset, dst_image, dst_offset, range);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_copy, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_ext_image_clear_callback(hsa_agent_t agent, hsa_ext_image_t image, const void* data, const hsa_ext_image_region_t* image_region) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_ext_image_clear.agent = agent;
  api_data.args.hsa_ext_image_clear.image = image;
  api_data.args.hsa_ext_image_clear.data = data;
  api_data.args.hsa_ext_image_clear.image_region = image_region;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_ext_image_clear, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_clear, &api_data, api_callback_arg);
  hsa_status_t ret =  ImageExtTable_saved.hsa_ext_image_clear_fn(agent, image, data, image_region);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_clear, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_ext_image_destroy_callback(hsa_agent_t agent, hsa_ext_image_t image) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_ext_image_destroy.agent = agent;
  api_data.args.hsa_ext_image_destroy.image = image;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_ext_image_destroy, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_destroy, &api_data, api_callback_arg);
  hsa_status_t ret =  ImageExtTable_saved.hsa_ext_image_destroy_fn(agent, image);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_destroy, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_ext_sampler_create_callback(hsa_agent_t agent, const hsa_ext_sampler_descriptor_t* sampler_descriptor, hsa_ext_sampler_t* sampler) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_ext_sampler_create.agent = agent;
  api_data.args.hsa_ext_sampler_create.sampler_descriptor = sampler_descriptor;
  api_data.args.hsa_ext_sampler_create.sampler = sampler;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_ext_sampler_create, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_sampler_create, &api_data, api_callback_arg);
  hsa_status_t ret =  ImageExtTable_saved.hsa_ext_sampler_create_fn(agent, sampler_descriptor, sampler);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_sampler_create, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_ext_sampler_destroy_callback(hsa_agent_t agent, hsa_ext_sampler_t sampler) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_ext_sampler_destroy.agent = agent;
  api_data.args.hsa_ext_sampler_destroy.sampler = sampler;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_ext_sampler_destroy, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_sampler_destroy, &api_data, api_callback_arg);
  hsa_status_t ret =  ImageExtTable_saved.hsa_ext_sampler_destroy_fn(agent, sampler);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_sampler_destroy, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_ext_image_get_capability_with_layout_callback(hsa_agent_t agent, hsa_ext_image_geometry_t geometry, const hsa_ext_image_format_t* image_format, hsa_ext_image_data_layout_t image_data_layout, uint32_t* capability_mask) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_ext_image_get_capability_with_layout.agent = agent;
  api_data.args.hsa_ext_image_get_capability_with_layout.geometry = geometry;
  api_data.args.hsa_ext_image_get_capability_with_layout.image_format = image_format;
  api_data.args.hsa_ext_image_get_capability_with_layout.image_data_layout = image_data_layout;
  api_data.args.hsa_ext_image_get_capability_with_layout.capability_mask = capability_mask;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_ext_image_get_capability_with_layout, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_get_capability_with_layout, &api_data, api_callback_arg);
  hsa_status_t ret =  ImageExtTable_saved.hsa_ext_image_get_capability_with_layout_fn(agent, geometry, image_format, image_data_layout, capability_mask);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_get_capability_with_layout, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_ext_image_data_get_info_with_layout_callback(hsa_agent_t agent, const hsa_ext_image_descriptor_t* image_descriptor, hsa_access_permission_t access_permission, hsa_ext_image_data_layout_t image_data_layout, size_t image_data_row_pitch, size_t image_data_slice_pitch, hsa_ext_image_data_info_t* image_data_info) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_ext_image_data_get_info_with_layout.agent = agent;
  api_data.args.hsa_ext_image_data_get_info_with_layout.image_descriptor = image_descriptor;
  api_data.args.hsa_ext_image_data_get_info_with_layout.access_permission = access_permission;
  api_data.args.hsa_ext_image_data_get_info_with_layout.image_data_layout = image_data_layout;
  api_data.args.hsa_ext_image_data_get_info_with_layout.image_data_row_pitch = image_data_row_pitch;
  api_data.args.hsa_ext_image_data_get_info_with_layout.image_data_slice_pitch = image_data_slice_pitch;
  api_data.args.hsa_ext_image_data_get_info_with_layout.image_data_info = image_data_info;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_ext_image_data_get_info_with_layout, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_data_get_info_with_layout, &api_data, api_callback_arg);
  hsa_status_t ret =  ImageExtTable_saved.hsa_ext_image_data_get_info_with_layout_fn(agent, image_descriptor, access_permission, image_data_layout, image_data_row_pitch, image_data_slice_pitch, image_data_info);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_data_get_info_with_layout, &api_data, api_callback_arg);
  return ret;
}
static hsa_status_t hsa_ext_image_create_with_layout_callback(hsa_agent_t agent, const hsa_ext_image_descriptor_t* image_descriptor, const void* image_data, hsa_access_permission_t access_permission, hsa_ext_image_data_layout_t image_data_layout, size_t image_data_row_pitch, size_t image_data_slice_pitch, hsa_ext_image_t* image) {
  hsa_api_data_t api_data{};
  api_data.args.hsa_ext_image_create_with_layout.agent = agent;
  api_data.args.hsa_ext_image_create_with_layout.image_descriptor = image_descriptor;
  api_data.args.hsa_ext_image_create_with_layout.image_data = image_data;
  api_data.args.hsa_ext_image_create_with_layout.access_permission = access_permission;
  api_data.args.hsa_ext_image_create_with_layout.image_data_layout = image_data_layout;
  api_data.args.hsa_ext_image_create_with_layout.image_data_row_pitch = image_data_row_pitch;
  api_data.args.hsa_ext_image_create_with_layout.image_data_slice_pitch = image_data_slice_pitch;
  api_data.args.hsa_ext_image_create_with_layout.image = image;
  activity_rtapi_callback_t api_callback_fun = NULL;
  void* api_callback_arg = NULL;
  cb_table.get(HSA_API_ID_hsa_ext_image_create_with_layout, &api_callback_fun, &api_callback_arg);
  api_data.phase = 0;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_create_with_layout, &api_data, api_callback_arg);
  hsa_status_t ret =  ImageExtTable_saved.hsa_ext_image_create_with_layout_fn(agent, image_descriptor, image_data, access_permission, image_data_layout, image_data_row_pitch, image_data_slice_pitch, image);
  api_data.hsa_status_t_retval = ret;
  api_data.phase = 1;
  if (api_callback_fun) api_callback_fun(ACTIVITY_DOMAIN_HSA_API, HSA_API_ID_hsa_ext_image_create_with_layout, &api_data, api_callback_arg);
  return ret;
}

// section: API intercepting code

// block: CoreApiTable API
static void intercept_CoreApiTable(CoreApiTable* table) {
  CoreApiTable_saved = *table;
  table->hsa_init_fn = hsa_init_callback;
  table->hsa_shut_down_fn = hsa_shut_down_callback;
  table->hsa_system_get_info_fn = hsa_system_get_info_callback;
  table->hsa_system_extension_supported_fn = hsa_system_extension_supported_callback;
  table->hsa_system_get_extension_table_fn = hsa_system_get_extension_table_callback;
  table->hsa_iterate_agents_fn = hsa_iterate_agents_callback;
  table->hsa_agent_get_info_fn = hsa_agent_get_info_callback;
  table->hsa_queue_create_fn = hsa_queue_create_callback;
  table->hsa_soft_queue_create_fn = hsa_soft_queue_create_callback;
  table->hsa_queue_destroy_fn = hsa_queue_destroy_callback;
  table->hsa_queue_inactivate_fn = hsa_queue_inactivate_callback;
  table->hsa_queue_load_read_index_scacquire_fn = hsa_queue_load_read_index_scacquire_callback;
  table->hsa_queue_load_read_index_relaxed_fn = hsa_queue_load_read_index_relaxed_callback;
  table->hsa_queue_load_write_index_scacquire_fn = hsa_queue_load_write_index_scacquire_callback;
  table->hsa_queue_load_write_index_relaxed_fn = hsa_queue_load_write_index_relaxed_callback;
  table->hsa_queue_store_write_index_relaxed_fn = hsa_queue_store_write_index_relaxed_callback;
  table->hsa_queue_store_write_index_screlease_fn = hsa_queue_store_write_index_screlease_callback;
  table->hsa_queue_cas_write_index_scacq_screl_fn = hsa_queue_cas_write_index_scacq_screl_callback;
  table->hsa_queue_cas_write_index_scacquire_fn = hsa_queue_cas_write_index_scacquire_callback;
  table->hsa_queue_cas_write_index_relaxed_fn = hsa_queue_cas_write_index_relaxed_callback;
  table->hsa_queue_cas_write_index_screlease_fn = hsa_queue_cas_write_index_screlease_callback;
  table->hsa_queue_add_write_index_scacq_screl_fn = hsa_queue_add_write_index_scacq_screl_callback;
  table->hsa_queue_add_write_index_scacquire_fn = hsa_queue_add_write_index_scacquire_callback;
  table->hsa_queue_add_write_index_relaxed_fn = hsa_queue_add_write_index_relaxed_callback;
  table->hsa_queue_add_write_index_screlease_fn = hsa_queue_add_write_index_screlease_callback;
  table->hsa_queue_store_read_index_relaxed_fn = hsa_queue_store_read_index_relaxed_callback;
  table->hsa_queue_store_read_index_screlease_fn = hsa_queue_store_read_index_screlease_callback;
  table->hsa_agent_iterate_regions_fn = hsa_agent_iterate_regions_callback;
  table->hsa_region_get_info_fn = hsa_region_get_info_callback;
  table->hsa_agent_get_exception_policies_fn = hsa_agent_get_exception_policies_callback;
  table->hsa_agent_extension_supported_fn = hsa_agent_extension_supported_callback;
  table->hsa_memory_register_fn = hsa_memory_register_callback;
  table->hsa_memory_deregister_fn = hsa_memory_deregister_callback;
  table->hsa_memory_allocate_fn = hsa_memory_allocate_callback;
  table->hsa_memory_free_fn = hsa_memory_free_callback;
  table->hsa_memory_copy_fn = hsa_memory_copy_callback;
  table->hsa_memory_assign_agent_fn = hsa_memory_assign_agent_callback;
  table->hsa_signal_create_fn = hsa_signal_create_callback;
  table->hsa_signal_destroy_fn = hsa_signal_destroy_callback;
  table->hsa_signal_load_relaxed_fn = hsa_signal_load_relaxed_callback;
  table->hsa_signal_load_scacquire_fn = hsa_signal_load_scacquire_callback;
  table->hsa_signal_store_relaxed_fn = hsa_signal_store_relaxed_callback;
  table->hsa_signal_store_screlease_fn = hsa_signal_store_screlease_callback;
  table->hsa_signal_wait_relaxed_fn = hsa_signal_wait_relaxed_callback;
  table->hsa_signal_wait_scacquire_fn = hsa_signal_wait_scacquire_callback;
  table->hsa_signal_and_relaxed_fn = hsa_signal_and_relaxed_callback;
  table->hsa_signal_and_scacquire_fn = hsa_signal_and_scacquire_callback;
  table->hsa_signal_and_screlease_fn = hsa_signal_and_screlease_callback;
  table->hsa_signal_and_scacq_screl_fn = hsa_signal_and_scacq_screl_callback;
  table->hsa_signal_or_relaxed_fn = hsa_signal_or_relaxed_callback;
  table->hsa_signal_or_scacquire_fn = hsa_signal_or_scacquire_callback;
  table->hsa_signal_or_screlease_fn = hsa_signal_or_screlease_callback;
  table->hsa_signal_or_scacq_screl_fn = hsa_signal_or_scacq_screl_callback;
  table->hsa_signal_xor_relaxed_fn = hsa_signal_xor_relaxed_callback;
  table->hsa_signal_xor_scacquire_fn = hsa_signal_xor_scacquire_callback;
  table->hsa_signal_xor_screlease_fn = hsa_signal_xor_screlease_callback;
  table->hsa_signal_xor_scacq_screl_fn = hsa_signal_xor_scacq_screl_callback;
  table->hsa_signal_exchange_relaxed_fn = hsa_signal_exchange_relaxed_callback;
  table->hsa_signal_exchange_scacquire_fn = hsa_signal_exchange_scacquire_callback;
  table->hsa_signal_exchange_screlease_fn = hsa_signal_exchange_screlease_callback;
  table->hsa_signal_exchange_scacq_screl_fn = hsa_signal_exchange_scacq_screl_callback;
  table->hsa_signal_add_relaxed_fn = hsa_signal_add_relaxed_callback;
  table->hsa_signal_add_scacquire_fn = hsa_signal_add_scacquire_callback;
  table->hsa_signal_add_screlease_fn = hsa_signal_add_screlease_callback;
  table->hsa_signal_add_scacq_screl_fn = hsa_signal_add_scacq_screl_callback;
  table->hsa_signal_subtract_relaxed_fn = hsa_signal_subtract_relaxed_callback;
  table->hsa_signal_subtract_scacquire_fn = hsa_signal_subtract_scacquire_callback;
  table->hsa_signal_subtract_screlease_fn = hsa_signal_subtract_screlease_callback;
  table->hsa_signal_subtract_scacq_screl_fn = hsa_signal_subtract_scacq_screl_callback;
  table->hsa_signal_cas_relaxed_fn = hsa_signal_cas_relaxed_callback;
  table->hsa_signal_cas_scacquire_fn = hsa_signal_cas_scacquire_callback;
  table->hsa_signal_cas_screlease_fn = hsa_signal_cas_screlease_callback;
  table->hsa_signal_cas_scacq_screl_fn = hsa_signal_cas_scacq_screl_callback;
  table->hsa_isa_from_name_fn = hsa_isa_from_name_callback;
  table->hsa_isa_get_info_fn = hsa_isa_get_info_callback;
  table->hsa_isa_compatible_fn = hsa_isa_compatible_callback;
  table->hsa_code_object_serialize_fn = hsa_code_object_serialize_callback;
  table->hsa_code_object_deserialize_fn = hsa_code_object_deserialize_callback;
  table->hsa_code_object_destroy_fn = hsa_code_object_destroy_callback;
  table->hsa_code_object_get_info_fn = hsa_code_object_get_info_callback;
  table->hsa_code_object_get_symbol_fn = hsa_code_object_get_symbol_callback;
  table->hsa_code_symbol_get_info_fn = hsa_code_symbol_get_info_callback;
  table->hsa_code_object_iterate_symbols_fn = hsa_code_object_iterate_symbols_callback;
  table->hsa_executable_create_fn = hsa_executable_create_callback;
  table->hsa_executable_destroy_fn = hsa_executable_destroy_callback;
  table->hsa_executable_load_code_object_fn = hsa_executable_load_code_object_callback;
  table->hsa_executable_freeze_fn = hsa_executable_freeze_callback;
  table->hsa_executable_get_info_fn = hsa_executable_get_info_callback;
  table->hsa_executable_global_variable_define_fn = hsa_executable_global_variable_define_callback;
  table->hsa_executable_agent_global_variable_define_fn = hsa_executable_agent_global_variable_define_callback;
  table->hsa_executable_readonly_variable_define_fn = hsa_executable_readonly_variable_define_callback;
  table->hsa_executable_validate_fn = hsa_executable_validate_callback;
  table->hsa_executable_get_symbol_fn = hsa_executable_get_symbol_callback;
  table->hsa_executable_symbol_get_info_fn = hsa_executable_symbol_get_info_callback;
  table->hsa_executable_iterate_symbols_fn = hsa_executable_iterate_symbols_callback;
  table->hsa_status_string_fn = hsa_status_string_callback;
  table->hsa_extension_get_name_fn = hsa_extension_get_name_callback;
  table->hsa_system_major_extension_supported_fn = hsa_system_major_extension_supported_callback;
  table->hsa_system_get_major_extension_table_fn = hsa_system_get_major_extension_table_callback;
  table->hsa_agent_major_extension_supported_fn = hsa_agent_major_extension_supported_callback;
  table->hsa_cache_get_info_fn = hsa_cache_get_info_callback;
  table->hsa_agent_iterate_caches_fn = hsa_agent_iterate_caches_callback;
  table->hsa_signal_silent_store_relaxed_fn = hsa_signal_silent_store_relaxed_callback;
  table->hsa_signal_silent_store_screlease_fn = hsa_signal_silent_store_screlease_callback;
  table->hsa_signal_group_create_fn = hsa_signal_group_create_callback;
  table->hsa_signal_group_destroy_fn = hsa_signal_group_destroy_callback;
  table->hsa_signal_group_wait_any_scacquire_fn = hsa_signal_group_wait_any_scacquire_callback;
  table->hsa_signal_group_wait_any_relaxed_fn = hsa_signal_group_wait_any_relaxed_callback;
  table->hsa_agent_iterate_isas_fn = hsa_agent_iterate_isas_callback;
  table->hsa_isa_get_info_alt_fn = hsa_isa_get_info_alt_callback;
  table->hsa_isa_get_exception_policies_fn = hsa_isa_get_exception_policies_callback;
  table->hsa_isa_get_round_method_fn = hsa_isa_get_round_method_callback;
  table->hsa_wavefront_get_info_fn = hsa_wavefront_get_info_callback;
  table->hsa_isa_iterate_wavefronts_fn = hsa_isa_iterate_wavefronts_callback;
  table->hsa_code_object_get_symbol_from_name_fn = hsa_code_object_get_symbol_from_name_callback;
  table->hsa_code_object_reader_create_from_file_fn = hsa_code_object_reader_create_from_file_callback;
  table->hsa_code_object_reader_create_from_memory_fn = hsa_code_object_reader_create_from_memory_callback;
  table->hsa_code_object_reader_destroy_fn = hsa_code_object_reader_destroy_callback;
  table->hsa_executable_create_alt_fn = hsa_executable_create_alt_callback;
  table->hsa_executable_load_program_code_object_fn = hsa_executable_load_program_code_object_callback;
  table->hsa_executable_load_agent_code_object_fn = hsa_executable_load_agent_code_object_callback;
  table->hsa_executable_validate_alt_fn = hsa_executable_validate_alt_callback;
  table->hsa_executable_get_symbol_by_name_fn = hsa_executable_get_symbol_by_name_callback;
  table->hsa_executable_iterate_agent_symbols_fn = hsa_executable_iterate_agent_symbols_callback;
  table->hsa_executable_iterate_program_symbols_fn = hsa_executable_iterate_program_symbols_callback;
};
static void intercept_AmdExtTable(AmdExtTable* table) {
  AmdExtTable_saved = *table;

// block: AmdExtTable API
  table->hsa_amd_coherency_get_type_fn = hsa_amd_coherency_get_type_callback;
  table->hsa_amd_coherency_set_type_fn = hsa_amd_coherency_set_type_callback;
  table->hsa_amd_profiling_set_profiler_enabled_fn = hsa_amd_profiling_set_profiler_enabled_callback;
  table->hsa_amd_profiling_async_copy_enable_fn = hsa_amd_profiling_async_copy_enable_callback;
  table->hsa_amd_profiling_get_dispatch_time_fn = hsa_amd_profiling_get_dispatch_time_callback;
  table->hsa_amd_profiling_get_async_copy_time_fn = hsa_amd_profiling_get_async_copy_time_callback;
  table->hsa_amd_profiling_convert_tick_to_system_domain_fn = hsa_amd_profiling_convert_tick_to_system_domain_callback;
  table->hsa_amd_signal_async_handler_fn = hsa_amd_signal_async_handler_callback;
  table->hsa_amd_async_function_fn = hsa_amd_async_function_callback;
  table->hsa_amd_signal_wait_any_fn = hsa_amd_signal_wait_any_callback;
  table->hsa_amd_queue_cu_set_mask_fn = hsa_amd_queue_cu_set_mask_callback;
  table->hsa_amd_memory_pool_get_info_fn = hsa_amd_memory_pool_get_info_callback;
  table->hsa_amd_agent_iterate_memory_pools_fn = hsa_amd_agent_iterate_memory_pools_callback;
  table->hsa_amd_memory_pool_allocate_fn = hsa_amd_memory_pool_allocate_callback;
  table->hsa_amd_memory_pool_free_fn = hsa_amd_memory_pool_free_callback;
  table->hsa_amd_memory_async_copy_fn = hsa_amd_memory_async_copy_callback;
  table->hsa_amd_agent_memory_pool_get_info_fn = hsa_amd_agent_memory_pool_get_info_callback;
  table->hsa_amd_agents_allow_access_fn = hsa_amd_agents_allow_access_callback;
  table->hsa_amd_memory_pool_can_migrate_fn = hsa_amd_memory_pool_can_migrate_callback;
  table->hsa_amd_memory_migrate_fn = hsa_amd_memory_migrate_callback;
  table->hsa_amd_memory_lock_fn = hsa_amd_memory_lock_callback;
  table->hsa_amd_memory_unlock_fn = hsa_amd_memory_unlock_callback;
  table->hsa_amd_memory_fill_fn = hsa_amd_memory_fill_callback;
  table->hsa_amd_interop_map_buffer_fn = hsa_amd_interop_map_buffer_callback;
  table->hsa_amd_interop_unmap_buffer_fn = hsa_amd_interop_unmap_buffer_callback;
  table->hsa_amd_image_create_fn = hsa_amd_image_create_callback;
  table->hsa_amd_pointer_info_fn = hsa_amd_pointer_info_callback;
  table->hsa_amd_pointer_info_set_userdata_fn = hsa_amd_pointer_info_set_userdata_callback;
  table->hsa_amd_ipc_memory_create_fn = hsa_amd_ipc_memory_create_callback;
  table->hsa_amd_ipc_memory_attach_fn = hsa_amd_ipc_memory_attach_callback;
  table->hsa_amd_ipc_memory_detach_fn = hsa_amd_ipc_memory_detach_callback;
  table->hsa_amd_signal_create_fn = hsa_amd_signal_create_callback;
  table->hsa_amd_ipc_signal_create_fn = hsa_amd_ipc_signal_create_callback;
  table->hsa_amd_ipc_signal_attach_fn = hsa_amd_ipc_signal_attach_callback;
  table->hsa_amd_register_system_event_handler_fn = hsa_amd_register_system_event_handler_callback;
  table->hsa_amd_queue_intercept_create_fn = hsa_amd_queue_intercept_create_callback;
  table->hsa_amd_queue_intercept_register_fn = hsa_amd_queue_intercept_register_callback;
  table->hsa_amd_queue_set_priority_fn = hsa_amd_queue_set_priority_callback;
  table->hsa_amd_memory_async_copy_rect_fn = hsa_amd_memory_async_copy_rect_callback;
  table->hsa_amd_runtime_queue_create_register_fn = hsa_amd_runtime_queue_create_register_callback;
};
static void intercept_ImageExtTable(ImageExtTable* table) {
  ImageExtTable_saved = *table;

// block: ImageExtTable API
  table->hsa_ext_image_get_capability_fn = hsa_ext_image_get_capability_callback;
  table->hsa_ext_image_data_get_info_fn = hsa_ext_image_data_get_info_callback;
  table->hsa_ext_image_create_fn = hsa_ext_image_create_callback;
  table->hsa_ext_image_import_fn = hsa_ext_image_import_callback;
  table->hsa_ext_image_export_fn = hsa_ext_image_export_callback;
  table->hsa_ext_image_copy_fn = hsa_ext_image_copy_callback;
  table->hsa_ext_image_clear_fn = hsa_ext_image_clear_callback;
  table->hsa_ext_image_destroy_fn = hsa_ext_image_destroy_callback;
  table->hsa_ext_sampler_create_fn = hsa_ext_sampler_create_callback;
  table->hsa_ext_sampler_destroy_fn = hsa_ext_sampler_destroy_callback;
  table->hsa_ext_image_get_capability_with_layout_fn = hsa_ext_image_get_capability_with_layout_callback;
  table->hsa_ext_image_data_get_info_with_layout_fn = hsa_ext_image_data_get_info_with_layout_callback;
  table->hsa_ext_image_create_with_layout_fn = hsa_ext_image_create_with_layout_callback;
};

// section: API get_name function

static const char* GetApiName(const uint32_t& id) {
  switch (id) {
    // block: CoreApiTable API
    case HSA_API_ID_hsa_init: return "hsa_init";
    case HSA_API_ID_hsa_shut_down: return "hsa_shut_down";
    case HSA_API_ID_hsa_system_get_info: return "hsa_system_get_info";
    case HSA_API_ID_hsa_system_extension_supported: return "hsa_system_extension_supported";
    case HSA_API_ID_hsa_system_get_extension_table: return "hsa_system_get_extension_table";
    case HSA_API_ID_hsa_iterate_agents: return "hsa_iterate_agents";
    case HSA_API_ID_hsa_agent_get_info: return "hsa_agent_get_info";
    case HSA_API_ID_hsa_queue_create: return "hsa_queue_create";
    case HSA_API_ID_hsa_soft_queue_create: return "hsa_soft_queue_create";
    case HSA_API_ID_hsa_queue_destroy: return "hsa_queue_destroy";
    case HSA_API_ID_hsa_queue_inactivate: return "hsa_queue_inactivate";
    case HSA_API_ID_hsa_queue_load_read_index_scacquire: return "hsa_queue_load_read_index_scacquire";
    case HSA_API_ID_hsa_queue_load_read_index_relaxed: return "hsa_queue_load_read_index_relaxed";
    case HSA_API_ID_hsa_queue_load_write_index_scacquire: return "hsa_queue_load_write_index_scacquire";
    case HSA_API_ID_hsa_queue_load_write_index_relaxed: return "hsa_queue_load_write_index_relaxed";
    case HSA_API_ID_hsa_queue_store_write_index_relaxed: return "hsa_queue_store_write_index_relaxed";
    case HSA_API_ID_hsa_queue_store_write_index_screlease: return "hsa_queue_store_write_index_screlease";
    case HSA_API_ID_hsa_queue_cas_write_index_scacq_screl: return "hsa_queue_cas_write_index_scacq_screl";
    case HSA_API_ID_hsa_queue_cas_write_index_scacquire: return "hsa_queue_cas_write_index_scacquire";
    case HSA_API_ID_hsa_queue_cas_write_index_relaxed: return "hsa_queue_cas_write_index_relaxed";
    case HSA_API_ID_hsa_queue_cas_write_index_screlease: return "hsa_queue_cas_write_index_screlease";
    case HSA_API_ID_hsa_queue_add_write_index_scacq_screl: return "hsa_queue_add_write_index_scacq_screl";
    case HSA_API_ID_hsa_queue_add_write_index_scacquire: return "hsa_queue_add_write_index_scacquire";
    case HSA_API_ID_hsa_queue_add_write_index_relaxed: return "hsa_queue_add_write_index_relaxed";
    case HSA_API_ID_hsa_queue_add_write_index_screlease: return "hsa_queue_add_write_index_screlease";
    case HSA_API_ID_hsa_queue_store_read_index_relaxed: return "hsa_queue_store_read_index_relaxed";
    case HSA_API_ID_hsa_queue_store_read_index_screlease: return "hsa_queue_store_read_index_screlease";
    case HSA_API_ID_hsa_agent_iterate_regions: return "hsa_agent_iterate_regions";
    case HSA_API_ID_hsa_region_get_info: return "hsa_region_get_info";
    case HSA_API_ID_hsa_agent_get_exception_policies: return "hsa_agent_get_exception_policies";
    case HSA_API_ID_hsa_agent_extension_supported: return "hsa_agent_extension_supported";
    case HSA_API_ID_hsa_memory_register: return "hsa_memory_register";
    case HSA_API_ID_hsa_memory_deregister: return "hsa_memory_deregister";
    case HSA_API_ID_hsa_memory_allocate: return "hsa_memory_allocate";
    case HSA_API_ID_hsa_memory_free: return "hsa_memory_free";
    case HSA_API_ID_hsa_memory_copy: return "hsa_memory_copy";
    case HSA_API_ID_hsa_memory_assign_agent: return "hsa_memory_assign_agent";
    case HSA_API_ID_hsa_signal_create: return "hsa_signal_create";
    case HSA_API_ID_hsa_signal_destroy: return "hsa_signal_destroy";
    case HSA_API_ID_hsa_signal_load_relaxed: return "hsa_signal_load_relaxed";
    case HSA_API_ID_hsa_signal_load_scacquire: return "hsa_signal_load_scacquire";
    case HSA_API_ID_hsa_signal_store_relaxed: return "hsa_signal_store_relaxed";
    case HSA_API_ID_hsa_signal_store_screlease: return "hsa_signal_store_screlease";
    case HSA_API_ID_hsa_signal_wait_relaxed: return "hsa_signal_wait_relaxed";
    case HSA_API_ID_hsa_signal_wait_scacquire: return "hsa_signal_wait_scacquire";
    case HSA_API_ID_hsa_signal_and_relaxed: return "hsa_signal_and_relaxed";
    case HSA_API_ID_hsa_signal_and_scacquire: return "hsa_signal_and_scacquire";
    case HSA_API_ID_hsa_signal_and_screlease: return "hsa_signal_and_screlease";
    case HSA_API_ID_hsa_signal_and_scacq_screl: return "hsa_signal_and_scacq_screl";
    case HSA_API_ID_hsa_signal_or_relaxed: return "hsa_signal_or_relaxed";
    case HSA_API_ID_hsa_signal_or_scacquire: return "hsa_signal_or_scacquire";
    case HSA_API_ID_hsa_signal_or_screlease: return "hsa_signal_or_screlease";
    case HSA_API_ID_hsa_signal_or_scacq_screl: return "hsa_signal_or_scacq_screl";
    case HSA_API_ID_hsa_signal_xor_relaxed: return "hsa_signal_xor_relaxed";
    case HSA_API_ID_hsa_signal_xor_scacquire: return "hsa_signal_xor_scacquire";
    case HSA_API_ID_hsa_signal_xor_screlease: return "hsa_signal_xor_screlease";
    case HSA_API_ID_hsa_signal_xor_scacq_screl: return "hsa_signal_xor_scacq_screl";
    case HSA_API_ID_hsa_signal_exchange_relaxed: return "hsa_signal_exchange_relaxed";
    case HSA_API_ID_hsa_signal_exchange_scacquire: return "hsa_signal_exchange_scacquire";
    case HSA_API_ID_hsa_signal_exchange_screlease: return "hsa_signal_exchange_screlease";
    case HSA_API_ID_hsa_signal_exchange_scacq_screl: return "hsa_signal_exchange_scacq_screl";
    case HSA_API_ID_hsa_signal_add_relaxed: return "hsa_signal_add_relaxed";
    case HSA_API_ID_hsa_signal_add_scacquire: return "hsa_signal_add_scacquire";
    case HSA_API_ID_hsa_signal_add_screlease: return "hsa_signal_add_screlease";
    case HSA_API_ID_hsa_signal_add_scacq_screl: return "hsa_signal_add_scacq_screl";
    case HSA_API_ID_hsa_signal_subtract_relaxed: return "hsa_signal_subtract_relaxed";
    case HSA_API_ID_hsa_signal_subtract_scacquire: return "hsa_signal_subtract_scacquire";
    case HSA_API_ID_hsa_signal_subtract_screlease: return "hsa_signal_subtract_screlease";
    case HSA_API_ID_hsa_signal_subtract_scacq_screl: return "hsa_signal_subtract_scacq_screl";
    case HSA_API_ID_hsa_signal_cas_relaxed: return "hsa_signal_cas_relaxed";
    case HSA_API_ID_hsa_signal_cas_scacquire: return "hsa_signal_cas_scacquire";
    case HSA_API_ID_hsa_signal_cas_screlease: return "hsa_signal_cas_screlease";
    case HSA_API_ID_hsa_signal_cas_scacq_screl: return "hsa_signal_cas_scacq_screl";
    case HSA_API_ID_hsa_isa_from_name: return "hsa_isa_from_name";
    case HSA_API_ID_hsa_isa_get_info: return "hsa_isa_get_info";
    case HSA_API_ID_hsa_isa_compatible: return "hsa_isa_compatible";
    case HSA_API_ID_hsa_code_object_serialize: return "hsa_code_object_serialize";
    case HSA_API_ID_hsa_code_object_deserialize: return "hsa_code_object_deserialize";
    case HSA_API_ID_hsa_code_object_destroy: return "hsa_code_object_destroy";
    case HSA_API_ID_hsa_code_object_get_info: return "hsa_code_object_get_info";
    case HSA_API_ID_hsa_code_object_get_symbol: return "hsa_code_object_get_symbol";
    case HSA_API_ID_hsa_code_symbol_get_info: return "hsa_code_symbol_get_info";
    case HSA_API_ID_hsa_code_object_iterate_symbols: return "hsa_code_object_iterate_symbols";
    case HSA_API_ID_hsa_executable_create: return "hsa_executable_create";
    case HSA_API_ID_hsa_executable_destroy: return "hsa_executable_destroy";
    case HSA_API_ID_hsa_executable_load_code_object: return "hsa_executable_load_code_object";
    case HSA_API_ID_hsa_executable_freeze: return "hsa_executable_freeze";
    case HSA_API_ID_hsa_executable_get_info: return "hsa_executable_get_info";
    case HSA_API_ID_hsa_executable_global_variable_define: return "hsa_executable_global_variable_define";
    case HSA_API_ID_hsa_executable_agent_global_variable_define: return "hsa_executable_agent_global_variable_define";
    case HSA_API_ID_hsa_executable_readonly_variable_define: return "hsa_executable_readonly_variable_define";
    case HSA_API_ID_hsa_executable_validate: return "hsa_executable_validate";
    case HSA_API_ID_hsa_executable_get_symbol: return "hsa_executable_get_symbol";
    case HSA_API_ID_hsa_executable_symbol_get_info: return "hsa_executable_symbol_get_info";
    case HSA_API_ID_hsa_executable_iterate_symbols: return "hsa_executable_iterate_symbols";
    case HSA_API_ID_hsa_status_string: return "hsa_status_string";
    case HSA_API_ID_hsa_extension_get_name: return "hsa_extension_get_name";
    case HSA_API_ID_hsa_system_major_extension_supported: return "hsa_system_major_extension_supported";
    case HSA_API_ID_hsa_system_get_major_extension_table: return "hsa_system_get_major_extension_table";
    case HSA_API_ID_hsa_agent_major_extension_supported: return "hsa_agent_major_extension_supported";
    case HSA_API_ID_hsa_cache_get_info: return "hsa_cache_get_info";
    case HSA_API_ID_hsa_agent_iterate_caches: return "hsa_agent_iterate_caches";
    case HSA_API_ID_hsa_signal_silent_store_relaxed: return "hsa_signal_silent_store_relaxed";
    case HSA_API_ID_hsa_signal_silent_store_screlease: return "hsa_signal_silent_store_screlease";
    case HSA_API_ID_hsa_signal_group_create: return "hsa_signal_group_create";
    case HSA_API_ID_hsa_signal_group_destroy: return "hsa_signal_group_destroy";
    case HSA_API_ID_hsa_signal_group_wait_any_scacquire: return "hsa_signal_group_wait_any_scacquire";
    case HSA_API_ID_hsa_signal_group_wait_any_relaxed: return "hsa_signal_group_wait_any_relaxed";
    case HSA_API_ID_hsa_agent_iterate_isas: return "hsa_agent_iterate_isas";
    case HSA_API_ID_hsa_isa_get_info_alt: return "hsa_isa_get_info_alt";
    case HSA_API_ID_hsa_isa_get_exception_policies: return "hsa_isa_get_exception_policies";
    case HSA_API_ID_hsa_isa_get_round_method: return "hsa_isa_get_round_method";
    case HSA_API_ID_hsa_wavefront_get_info: return "hsa_wavefront_get_info";
    case HSA_API_ID_hsa_isa_iterate_wavefronts: return "hsa_isa_iterate_wavefronts";
    case HSA_API_ID_hsa_code_object_get_symbol_from_name: return "hsa_code_object_get_symbol_from_name";
    case HSA_API_ID_hsa_code_object_reader_create_from_file: return "hsa_code_object_reader_create_from_file";
    case HSA_API_ID_hsa_code_object_reader_create_from_memory: return "hsa_code_object_reader_create_from_memory";
    case HSA_API_ID_hsa_code_object_reader_destroy: return "hsa_code_object_reader_destroy";
    case HSA_API_ID_hsa_executable_create_alt: return "hsa_executable_create_alt";
    case HSA_API_ID_hsa_executable_load_program_code_object: return "hsa_executable_load_program_code_object";
    case HSA_API_ID_hsa_executable_load_agent_code_object: return "hsa_executable_load_agent_code_object";
    case HSA_API_ID_hsa_executable_validate_alt: return "hsa_executable_validate_alt";
    case HSA_API_ID_hsa_executable_get_symbol_by_name: return "hsa_executable_get_symbol_by_name";
    case HSA_API_ID_hsa_executable_iterate_agent_symbols: return "hsa_executable_iterate_agent_symbols";
    case HSA_API_ID_hsa_executable_iterate_program_symbols: return "hsa_executable_iterate_program_symbols";

    // block: AmdExtTable API
    case HSA_API_ID_hsa_amd_coherency_get_type: return "hsa_amd_coherency_get_type";
    case HSA_API_ID_hsa_amd_coherency_set_type: return "hsa_amd_coherency_set_type";
    case HSA_API_ID_hsa_amd_profiling_set_profiler_enabled: return "hsa_amd_profiling_set_profiler_enabled";
    case HSA_API_ID_hsa_amd_profiling_async_copy_enable: return "hsa_amd_profiling_async_copy_enable";
    case HSA_API_ID_hsa_amd_profiling_get_dispatch_time: return "hsa_amd_profiling_get_dispatch_time";
    case HSA_API_ID_hsa_amd_profiling_get_async_copy_time: return "hsa_amd_profiling_get_async_copy_time";
    case HSA_API_ID_hsa_amd_profiling_convert_tick_to_system_domain: return "hsa_amd_profiling_convert_tick_to_system_domain";
    case HSA_API_ID_hsa_amd_signal_async_handler: return "hsa_amd_signal_async_handler";
    case HSA_API_ID_hsa_amd_async_function: return "hsa_amd_async_function";
    case HSA_API_ID_hsa_amd_signal_wait_any: return "hsa_amd_signal_wait_any";
    case HSA_API_ID_hsa_amd_queue_cu_set_mask: return "hsa_amd_queue_cu_set_mask";
    case HSA_API_ID_hsa_amd_memory_pool_get_info: return "hsa_amd_memory_pool_get_info";
    case HSA_API_ID_hsa_amd_agent_iterate_memory_pools: return "hsa_amd_agent_iterate_memory_pools";
    case HSA_API_ID_hsa_amd_memory_pool_allocate: return "hsa_amd_memory_pool_allocate";
    case HSA_API_ID_hsa_amd_memory_pool_free: return "hsa_amd_memory_pool_free";
    case HSA_API_ID_hsa_amd_memory_async_copy: return "hsa_amd_memory_async_copy";
    case HSA_API_ID_hsa_amd_agent_memory_pool_get_info: return "hsa_amd_agent_memory_pool_get_info";
    case HSA_API_ID_hsa_amd_agents_allow_access: return "hsa_amd_agents_allow_access";
    case HSA_API_ID_hsa_amd_memory_pool_can_migrate: return "hsa_amd_memory_pool_can_migrate";
    case HSA_API_ID_hsa_amd_memory_migrate: return "hsa_amd_memory_migrate";
    case HSA_API_ID_hsa_amd_memory_lock: return "hsa_amd_memory_lock";
    case HSA_API_ID_hsa_amd_memory_unlock: return "hsa_amd_memory_unlock";
    case HSA_API_ID_hsa_amd_memory_fill: return "hsa_amd_memory_fill";
    case HSA_API_ID_hsa_amd_interop_map_buffer: return "hsa_amd_interop_map_buffer";
    case HSA_API_ID_hsa_amd_interop_unmap_buffer: return "hsa_amd_interop_unmap_buffer";
    case HSA_API_ID_hsa_amd_image_create: return "hsa_amd_image_create";
    case HSA_API_ID_hsa_amd_pointer_info: return "hsa_amd_pointer_info";
    case HSA_API_ID_hsa_amd_pointer_info_set_userdata: return "hsa_amd_pointer_info_set_userdata";
    case HSA_API_ID_hsa_amd_ipc_memory_create: return "hsa_amd_ipc_memory_create";
    case HSA_API_ID_hsa_amd_ipc_memory_attach: return "hsa_amd_ipc_memory_attach";
    case HSA_API_ID_hsa_amd_ipc_memory_detach: return "hsa_amd_ipc_memory_detach";
    case HSA_API_ID_hsa_amd_signal_create: return "hsa_amd_signal_create";
    case HSA_API_ID_hsa_amd_ipc_signal_create: return "hsa_amd_ipc_signal_create";
    case HSA_API_ID_hsa_amd_ipc_signal_attach: return "hsa_amd_ipc_signal_attach";
    case HSA_API_ID_hsa_amd_register_system_event_handler: return "hsa_amd_register_system_event_handler";
    case HSA_API_ID_hsa_amd_queue_intercept_create: return "hsa_amd_queue_intercept_create";
    case HSA_API_ID_hsa_amd_queue_intercept_register: return "hsa_amd_queue_intercept_register";
    case HSA_API_ID_hsa_amd_queue_set_priority: return "hsa_amd_queue_set_priority";
    case HSA_API_ID_hsa_amd_memory_async_copy_rect: return "hsa_amd_memory_async_copy_rect";
    case HSA_API_ID_hsa_amd_runtime_queue_create_register: return "hsa_amd_runtime_queue_create_register";

    // block: ImageExtTable API
    case HSA_API_ID_hsa_ext_image_get_capability: return "hsa_ext_image_get_capability";
    case HSA_API_ID_hsa_ext_image_data_get_info: return "hsa_ext_image_data_get_info";
    case HSA_API_ID_hsa_ext_image_create: return "hsa_ext_image_create";
    case HSA_API_ID_hsa_ext_image_import: return "hsa_ext_image_import";
    case HSA_API_ID_hsa_ext_image_export: return "hsa_ext_image_export";
    case HSA_API_ID_hsa_ext_image_copy: return "hsa_ext_image_copy";
    case HSA_API_ID_hsa_ext_image_clear: return "hsa_ext_image_clear";
    case HSA_API_ID_hsa_ext_image_destroy: return "hsa_ext_image_destroy";
    case HSA_API_ID_hsa_ext_sampler_create: return "hsa_ext_sampler_create";
    case HSA_API_ID_hsa_ext_sampler_destroy: return "hsa_ext_sampler_destroy";
    case HSA_API_ID_hsa_ext_image_get_capability_with_layout: return "hsa_ext_image_get_capability_with_layout";
    case HSA_API_ID_hsa_ext_image_data_get_info_with_layout: return "hsa_ext_image_data_get_info_with_layout";
    case HSA_API_ID_hsa_ext_image_create_with_layout: return "hsa_ext_image_create_with_layout";
  }
  return "unknown";
}

};};
#endif // PROF_API_IMPL

// section: API output stream

typedef std::pair<uint32_t, hsa_api_data_t> hsa_api_data_pair_t;
inline std::ostream& operator<< (std::ostream& out, const hsa_api_data_pair_t& data_pair) {
  const uint32_t cid = data_pair.first;
  const hsa_api_data_t& api_data = data_pair.second;
  switch(cid) {
    // block: CoreApiTable API
    case HSA_API_ID_hsa_init: {
      out << "hsa_init(";
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_shut_down: {
      out << "hsa_shut_down(";
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_system_get_info: {
      out << "hsa_system_get_info(";
      typedef decltype(api_data.args.hsa_system_get_info.attribute) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_system_get_info.attribute) << ", ";
      typedef decltype(api_data.args.hsa_system_get_info.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_system_get_info.value);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_system_extension_supported: {
      out << "hsa_system_extension_supported(";
      typedef decltype(api_data.args.hsa_system_extension_supported.extension) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_system_extension_supported.extension) << ", ";
      typedef decltype(api_data.args.hsa_system_extension_supported.version_major) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_system_extension_supported.version_major) << ", ";
      typedef decltype(api_data.args.hsa_system_extension_supported.version_minor) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_system_extension_supported.version_minor) << ", ";
      typedef decltype(api_data.args.hsa_system_extension_supported.result) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_system_extension_supported.result);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_system_get_extension_table: {
      out << "hsa_system_get_extension_table(";
      typedef decltype(api_data.args.hsa_system_get_extension_table.extension) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_system_get_extension_table.extension) << ", ";
      typedef decltype(api_data.args.hsa_system_get_extension_table.version_major) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_system_get_extension_table.version_major) << ", ";
      typedef decltype(api_data.args.hsa_system_get_extension_table.version_minor) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_system_get_extension_table.version_minor) << ", ";
      typedef decltype(api_data.args.hsa_system_get_extension_table.table) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_system_get_extension_table.table);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_iterate_agents: {
      out << "hsa_iterate_agents(";
      typedef decltype(api_data.args.hsa_iterate_agents.callback) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_iterate_agents.callback) << ", ";
      typedef decltype(api_data.args.hsa_iterate_agents.data) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_iterate_agents.data);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_agent_get_info: {
      out << "hsa_agent_get_info(";
      typedef decltype(api_data.args.hsa_agent_get_info.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_agent_get_info.agent) << ", ";
      typedef decltype(api_data.args.hsa_agent_get_info.attribute) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_agent_get_info.attribute) << ", ";
      typedef decltype(api_data.args.hsa_agent_get_info.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_agent_get_info.value);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_create: {
      out << "hsa_queue_create(";
      typedef decltype(api_data.args.hsa_queue_create.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_create.agent) << ", ";
      typedef decltype(api_data.args.hsa_queue_create.size) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_queue_create.size) << ", ";
      typedef decltype(api_data.args.hsa_queue_create.type) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_queue_create.type) << ", ";
      typedef decltype(api_data.args.hsa_queue_create.callback) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_queue_create.callback) << ", ";
      typedef decltype(api_data.args.hsa_queue_create.data) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_queue_create.data) << ", ";
      typedef decltype(api_data.args.hsa_queue_create.private_segment_size) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_queue_create.private_segment_size) << ", ";
      typedef decltype(api_data.args.hsa_queue_create.group_segment_size) arg_val_type_t6;
      roctracer::hsa_support::output_streamer<arg_val_type_t6>::put(out, api_data.args.hsa_queue_create.group_segment_size) << ", ";
      typedef decltype(api_data.args.hsa_queue_create.queue) arg_val_type_t7;
      roctracer::hsa_support::output_streamer<arg_val_type_t7>::put(out, api_data.args.hsa_queue_create.queue);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_soft_queue_create: {
      out << "hsa_soft_queue_create(";
      typedef decltype(api_data.args.hsa_soft_queue_create.region) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_soft_queue_create.region) << ", ";
      typedef decltype(api_data.args.hsa_soft_queue_create.size) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_soft_queue_create.size) << ", ";
      typedef decltype(api_data.args.hsa_soft_queue_create.type) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_soft_queue_create.type) << ", ";
      typedef decltype(api_data.args.hsa_soft_queue_create.features) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_soft_queue_create.features) << ", ";
      typedef decltype(api_data.args.hsa_soft_queue_create.doorbell_signal) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_soft_queue_create.doorbell_signal) << ", ";
      typedef decltype(api_data.args.hsa_soft_queue_create.queue) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_soft_queue_create.queue);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_destroy: {
      out << "hsa_queue_destroy(";
      typedef decltype(api_data.args.hsa_queue_destroy.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_destroy.queue);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_inactivate: {
      out << "hsa_queue_inactivate(";
      typedef decltype(api_data.args.hsa_queue_inactivate.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_inactivate.queue);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_load_read_index_scacquire: {
      out << "hsa_queue_load_read_index_scacquire(";
      typedef decltype(api_data.args.hsa_queue_load_read_index_scacquire.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_load_read_index_scacquire.queue);
      out << ") = " << api_data.uint64_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_load_read_index_relaxed: {
      out << "hsa_queue_load_read_index_relaxed(";
      typedef decltype(api_data.args.hsa_queue_load_read_index_relaxed.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_load_read_index_relaxed.queue);
      out << ") = " << api_data.uint64_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_load_write_index_scacquire: {
      out << "hsa_queue_load_write_index_scacquire(";
      typedef decltype(api_data.args.hsa_queue_load_write_index_scacquire.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_load_write_index_scacquire.queue);
      out << ") = " << api_data.uint64_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_load_write_index_relaxed: {
      out << "hsa_queue_load_write_index_relaxed(";
      typedef decltype(api_data.args.hsa_queue_load_write_index_relaxed.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_load_write_index_relaxed.queue);
      out << ") = " << api_data.uint64_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_store_write_index_relaxed: {
      out << "hsa_queue_store_write_index_relaxed(";
      typedef decltype(api_data.args.hsa_queue_store_write_index_relaxed.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_store_write_index_relaxed.queue) << ", ";
      typedef decltype(api_data.args.hsa_queue_store_write_index_relaxed.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_queue_store_write_index_relaxed.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_queue_store_write_index_screlease: {
      out << "hsa_queue_store_write_index_screlease(";
      typedef decltype(api_data.args.hsa_queue_store_write_index_screlease.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_store_write_index_screlease.queue) << ", ";
      typedef decltype(api_data.args.hsa_queue_store_write_index_screlease.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_queue_store_write_index_screlease.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_queue_cas_write_index_scacq_screl: {
      out << "hsa_queue_cas_write_index_scacq_screl(";
      typedef decltype(api_data.args.hsa_queue_cas_write_index_scacq_screl.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_cas_write_index_scacq_screl.queue) << ", ";
      typedef decltype(api_data.args.hsa_queue_cas_write_index_scacq_screl.expected) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_queue_cas_write_index_scacq_screl.expected) << ", ";
      typedef decltype(api_data.args.hsa_queue_cas_write_index_scacq_screl.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_queue_cas_write_index_scacq_screl.value);
      out << ") = " << api_data.uint64_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_cas_write_index_scacquire: {
      out << "hsa_queue_cas_write_index_scacquire(";
      typedef decltype(api_data.args.hsa_queue_cas_write_index_scacquire.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_cas_write_index_scacquire.queue) << ", ";
      typedef decltype(api_data.args.hsa_queue_cas_write_index_scacquire.expected) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_queue_cas_write_index_scacquire.expected) << ", ";
      typedef decltype(api_data.args.hsa_queue_cas_write_index_scacquire.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_queue_cas_write_index_scacquire.value);
      out << ") = " << api_data.uint64_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_cas_write_index_relaxed: {
      out << "hsa_queue_cas_write_index_relaxed(";
      typedef decltype(api_data.args.hsa_queue_cas_write_index_relaxed.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_cas_write_index_relaxed.queue) << ", ";
      typedef decltype(api_data.args.hsa_queue_cas_write_index_relaxed.expected) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_queue_cas_write_index_relaxed.expected) << ", ";
      typedef decltype(api_data.args.hsa_queue_cas_write_index_relaxed.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_queue_cas_write_index_relaxed.value);
      out << ") = " << api_data.uint64_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_cas_write_index_screlease: {
      out << "hsa_queue_cas_write_index_screlease(";
      typedef decltype(api_data.args.hsa_queue_cas_write_index_screlease.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_cas_write_index_screlease.queue) << ", ";
      typedef decltype(api_data.args.hsa_queue_cas_write_index_screlease.expected) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_queue_cas_write_index_screlease.expected) << ", ";
      typedef decltype(api_data.args.hsa_queue_cas_write_index_screlease.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_queue_cas_write_index_screlease.value);
      out << ") = " << api_data.uint64_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_add_write_index_scacq_screl: {
      out << "hsa_queue_add_write_index_scacq_screl(";
      typedef decltype(api_data.args.hsa_queue_add_write_index_scacq_screl.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_add_write_index_scacq_screl.queue) << ", ";
      typedef decltype(api_data.args.hsa_queue_add_write_index_scacq_screl.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_queue_add_write_index_scacq_screl.value);
      out << ") = " << api_data.uint64_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_add_write_index_scacquire: {
      out << "hsa_queue_add_write_index_scacquire(";
      typedef decltype(api_data.args.hsa_queue_add_write_index_scacquire.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_add_write_index_scacquire.queue) << ", ";
      typedef decltype(api_data.args.hsa_queue_add_write_index_scacquire.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_queue_add_write_index_scacquire.value);
      out << ") = " << api_data.uint64_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_add_write_index_relaxed: {
      out << "hsa_queue_add_write_index_relaxed(";
      typedef decltype(api_data.args.hsa_queue_add_write_index_relaxed.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_add_write_index_relaxed.queue) << ", ";
      typedef decltype(api_data.args.hsa_queue_add_write_index_relaxed.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_queue_add_write_index_relaxed.value);
      out << ") = " << api_data.uint64_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_add_write_index_screlease: {
      out << "hsa_queue_add_write_index_screlease(";
      typedef decltype(api_data.args.hsa_queue_add_write_index_screlease.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_add_write_index_screlease.queue) << ", ";
      typedef decltype(api_data.args.hsa_queue_add_write_index_screlease.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_queue_add_write_index_screlease.value);
      out << ") = " << api_data.uint64_t_retval;
      break;
    }
    case HSA_API_ID_hsa_queue_store_read_index_relaxed: {
      out << "hsa_queue_store_read_index_relaxed(";
      typedef decltype(api_data.args.hsa_queue_store_read_index_relaxed.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_store_read_index_relaxed.queue) << ", ";
      typedef decltype(api_data.args.hsa_queue_store_read_index_relaxed.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_queue_store_read_index_relaxed.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_queue_store_read_index_screlease: {
      out << "hsa_queue_store_read_index_screlease(";
      typedef decltype(api_data.args.hsa_queue_store_read_index_screlease.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_queue_store_read_index_screlease.queue) << ", ";
      typedef decltype(api_data.args.hsa_queue_store_read_index_screlease.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_queue_store_read_index_screlease.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_agent_iterate_regions: {
      out << "hsa_agent_iterate_regions(";
      typedef decltype(api_data.args.hsa_agent_iterate_regions.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_agent_iterate_regions.agent) << ", ";
      typedef decltype(api_data.args.hsa_agent_iterate_regions.callback) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_agent_iterate_regions.callback) << ", ";
      typedef decltype(api_data.args.hsa_agent_iterate_regions.data) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_agent_iterate_regions.data);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_region_get_info: {
      out << "hsa_region_get_info(";
      typedef decltype(api_data.args.hsa_region_get_info.region) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_region_get_info.region) << ", ";
      typedef decltype(api_data.args.hsa_region_get_info.attribute) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_region_get_info.attribute) << ", ";
      typedef decltype(api_data.args.hsa_region_get_info.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_region_get_info.value);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_agent_get_exception_policies: {
      out << "hsa_agent_get_exception_policies(";
      typedef decltype(api_data.args.hsa_agent_get_exception_policies.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_agent_get_exception_policies.agent) << ", ";
      typedef decltype(api_data.args.hsa_agent_get_exception_policies.profile) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_agent_get_exception_policies.profile) << ", ";
      typedef decltype(api_data.args.hsa_agent_get_exception_policies.mask) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_agent_get_exception_policies.mask);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_agent_extension_supported: {
      out << "hsa_agent_extension_supported(";
      typedef decltype(api_data.args.hsa_agent_extension_supported.extension) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_agent_extension_supported.extension) << ", ";
      typedef decltype(api_data.args.hsa_agent_extension_supported.agent) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_agent_extension_supported.agent) << ", ";
      typedef decltype(api_data.args.hsa_agent_extension_supported.version_major) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_agent_extension_supported.version_major) << ", ";
      typedef decltype(api_data.args.hsa_agent_extension_supported.version_minor) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_agent_extension_supported.version_minor) << ", ";
      typedef decltype(api_data.args.hsa_agent_extension_supported.result) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_agent_extension_supported.result);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_memory_register: {
      out << "hsa_memory_register(";
      typedef decltype(api_data.args.hsa_memory_register.ptr) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_memory_register.ptr) << ", ";
      typedef decltype(api_data.args.hsa_memory_register.size) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_memory_register.size);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_memory_deregister: {
      out << "hsa_memory_deregister(";
      typedef decltype(api_data.args.hsa_memory_deregister.ptr) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_memory_deregister.ptr) << ", ";
      typedef decltype(api_data.args.hsa_memory_deregister.size) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_memory_deregister.size);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_memory_allocate: {
      out << "hsa_memory_allocate(";
      typedef decltype(api_data.args.hsa_memory_allocate.region) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_memory_allocate.region) << ", ";
      typedef decltype(api_data.args.hsa_memory_allocate.size) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_memory_allocate.size) << ", ";
      typedef decltype(api_data.args.hsa_memory_allocate.ptr) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_memory_allocate.ptr);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_memory_free: {
      out << "hsa_memory_free(";
      typedef decltype(api_data.args.hsa_memory_free.ptr) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_memory_free.ptr);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_memory_copy: {
      out << "hsa_memory_copy(";
      typedef decltype(api_data.args.hsa_memory_copy.dst) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_memory_copy.dst) << ", ";
      typedef decltype(api_data.args.hsa_memory_copy.src) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_memory_copy.src) << ", ";
      typedef decltype(api_data.args.hsa_memory_copy.size) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_memory_copy.size);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_memory_assign_agent: {
      out << "hsa_memory_assign_agent(";
      typedef decltype(api_data.args.hsa_memory_assign_agent.ptr) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_memory_assign_agent.ptr) << ", ";
      typedef decltype(api_data.args.hsa_memory_assign_agent.agent) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_memory_assign_agent.agent) << ", ";
      typedef decltype(api_data.args.hsa_memory_assign_agent.access) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_memory_assign_agent.access);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_create: {
      out << "hsa_signal_create(";
      typedef decltype(api_data.args.hsa_signal_create.initial_value) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_create.initial_value) << ", ";
      typedef decltype(api_data.args.hsa_signal_create.num_consumers) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_create.num_consumers) << ", ";
      typedef decltype(api_data.args.hsa_signal_create.consumers) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_signal_create.consumers) << ", ";
      typedef decltype(api_data.args.hsa_signal_create.signal) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_signal_create.signal);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_destroy: {
      out << "hsa_signal_destroy(";
      typedef decltype(api_data.args.hsa_signal_destroy.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_destroy.signal);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_load_relaxed: {
      out << "hsa_signal_load_relaxed(";
      typedef decltype(api_data.args.hsa_signal_load_relaxed.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_load_relaxed.signal);
      out << ") = " << api_data.hsa_signal_value_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_load_scacquire: {
      out << "hsa_signal_load_scacquire(";
      typedef decltype(api_data.args.hsa_signal_load_scacquire.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_load_scacquire.signal);
      out << ") = " << api_data.hsa_signal_value_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_store_relaxed: {
      out << "hsa_signal_store_relaxed(";
      typedef decltype(api_data.args.hsa_signal_store_relaxed.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_store_relaxed.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_store_relaxed.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_store_relaxed.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_store_screlease: {
      out << "hsa_signal_store_screlease(";
      typedef decltype(api_data.args.hsa_signal_store_screlease.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_store_screlease.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_store_screlease.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_store_screlease.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_wait_relaxed: {
      out << "hsa_signal_wait_relaxed(";
      typedef decltype(api_data.args.hsa_signal_wait_relaxed.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_wait_relaxed.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_wait_relaxed.condition) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_wait_relaxed.condition) << ", ";
      typedef decltype(api_data.args.hsa_signal_wait_relaxed.compare_value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_signal_wait_relaxed.compare_value) << ", ";
      typedef decltype(api_data.args.hsa_signal_wait_relaxed.timeout_hint) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_signal_wait_relaxed.timeout_hint) << ", ";
      typedef decltype(api_data.args.hsa_signal_wait_relaxed.wait_state_hint) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_signal_wait_relaxed.wait_state_hint);
      out << ") = " << api_data.hsa_signal_value_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_wait_scacquire: {
      out << "hsa_signal_wait_scacquire(";
      typedef decltype(api_data.args.hsa_signal_wait_scacquire.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_wait_scacquire.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_wait_scacquire.condition) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_wait_scacquire.condition) << ", ";
      typedef decltype(api_data.args.hsa_signal_wait_scacquire.compare_value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_signal_wait_scacquire.compare_value) << ", ";
      typedef decltype(api_data.args.hsa_signal_wait_scacquire.timeout_hint) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_signal_wait_scacquire.timeout_hint) << ", ";
      typedef decltype(api_data.args.hsa_signal_wait_scacquire.wait_state_hint) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_signal_wait_scacquire.wait_state_hint);
      out << ") = " << api_data.hsa_signal_value_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_and_relaxed: {
      out << "hsa_signal_and_relaxed(";
      typedef decltype(api_data.args.hsa_signal_and_relaxed.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_and_relaxed.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_and_relaxed.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_and_relaxed.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_and_scacquire: {
      out << "hsa_signal_and_scacquire(";
      typedef decltype(api_data.args.hsa_signal_and_scacquire.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_and_scacquire.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_and_scacquire.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_and_scacquire.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_and_screlease: {
      out << "hsa_signal_and_screlease(";
      typedef decltype(api_data.args.hsa_signal_and_screlease.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_and_screlease.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_and_screlease.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_and_screlease.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_and_scacq_screl: {
      out << "hsa_signal_and_scacq_screl(";
      typedef decltype(api_data.args.hsa_signal_and_scacq_screl.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_and_scacq_screl.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_and_scacq_screl.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_and_scacq_screl.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_or_relaxed: {
      out << "hsa_signal_or_relaxed(";
      typedef decltype(api_data.args.hsa_signal_or_relaxed.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_or_relaxed.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_or_relaxed.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_or_relaxed.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_or_scacquire: {
      out << "hsa_signal_or_scacquire(";
      typedef decltype(api_data.args.hsa_signal_or_scacquire.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_or_scacquire.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_or_scacquire.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_or_scacquire.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_or_screlease: {
      out << "hsa_signal_or_screlease(";
      typedef decltype(api_data.args.hsa_signal_or_screlease.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_or_screlease.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_or_screlease.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_or_screlease.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_or_scacq_screl: {
      out << "hsa_signal_or_scacq_screl(";
      typedef decltype(api_data.args.hsa_signal_or_scacq_screl.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_or_scacq_screl.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_or_scacq_screl.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_or_scacq_screl.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_xor_relaxed: {
      out << "hsa_signal_xor_relaxed(";
      typedef decltype(api_data.args.hsa_signal_xor_relaxed.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_xor_relaxed.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_xor_relaxed.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_xor_relaxed.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_xor_scacquire: {
      out << "hsa_signal_xor_scacquire(";
      typedef decltype(api_data.args.hsa_signal_xor_scacquire.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_xor_scacquire.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_xor_scacquire.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_xor_scacquire.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_xor_screlease: {
      out << "hsa_signal_xor_screlease(";
      typedef decltype(api_data.args.hsa_signal_xor_screlease.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_xor_screlease.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_xor_screlease.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_xor_screlease.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_xor_scacq_screl: {
      out << "hsa_signal_xor_scacq_screl(";
      typedef decltype(api_data.args.hsa_signal_xor_scacq_screl.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_xor_scacq_screl.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_xor_scacq_screl.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_xor_scacq_screl.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_exchange_relaxed: {
      out << "hsa_signal_exchange_relaxed(";
      typedef decltype(api_data.args.hsa_signal_exchange_relaxed.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_exchange_relaxed.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_exchange_relaxed.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_exchange_relaxed.value);
      out << ") = " << api_data.hsa_signal_value_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_exchange_scacquire: {
      out << "hsa_signal_exchange_scacquire(";
      typedef decltype(api_data.args.hsa_signal_exchange_scacquire.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_exchange_scacquire.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_exchange_scacquire.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_exchange_scacquire.value);
      out << ") = " << api_data.hsa_signal_value_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_exchange_screlease: {
      out << "hsa_signal_exchange_screlease(";
      typedef decltype(api_data.args.hsa_signal_exchange_screlease.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_exchange_screlease.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_exchange_screlease.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_exchange_screlease.value);
      out << ") = " << api_data.hsa_signal_value_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_exchange_scacq_screl: {
      out << "hsa_signal_exchange_scacq_screl(";
      typedef decltype(api_data.args.hsa_signal_exchange_scacq_screl.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_exchange_scacq_screl.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_exchange_scacq_screl.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_exchange_scacq_screl.value);
      out << ") = " << api_data.hsa_signal_value_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_add_relaxed: {
      out << "hsa_signal_add_relaxed(";
      typedef decltype(api_data.args.hsa_signal_add_relaxed.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_add_relaxed.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_add_relaxed.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_add_relaxed.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_add_scacquire: {
      out << "hsa_signal_add_scacquire(";
      typedef decltype(api_data.args.hsa_signal_add_scacquire.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_add_scacquire.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_add_scacquire.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_add_scacquire.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_add_screlease: {
      out << "hsa_signal_add_screlease(";
      typedef decltype(api_data.args.hsa_signal_add_screlease.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_add_screlease.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_add_screlease.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_add_screlease.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_add_scacq_screl: {
      out << "hsa_signal_add_scacq_screl(";
      typedef decltype(api_data.args.hsa_signal_add_scacq_screl.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_add_scacq_screl.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_add_scacq_screl.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_add_scacq_screl.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_subtract_relaxed: {
      out << "hsa_signal_subtract_relaxed(";
      typedef decltype(api_data.args.hsa_signal_subtract_relaxed.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_subtract_relaxed.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_subtract_relaxed.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_subtract_relaxed.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_subtract_scacquire: {
      out << "hsa_signal_subtract_scacquire(";
      typedef decltype(api_data.args.hsa_signal_subtract_scacquire.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_subtract_scacquire.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_subtract_scacquire.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_subtract_scacquire.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_subtract_screlease: {
      out << "hsa_signal_subtract_screlease(";
      typedef decltype(api_data.args.hsa_signal_subtract_screlease.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_subtract_screlease.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_subtract_screlease.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_subtract_screlease.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_subtract_scacq_screl: {
      out << "hsa_signal_subtract_scacq_screl(";
      typedef decltype(api_data.args.hsa_signal_subtract_scacq_screl.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_subtract_scacq_screl.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_subtract_scacq_screl.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_subtract_scacq_screl.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_cas_relaxed: {
      out << "hsa_signal_cas_relaxed(";
      typedef decltype(api_data.args.hsa_signal_cas_relaxed.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_cas_relaxed.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_cas_relaxed.expected) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_cas_relaxed.expected) << ", ";
      typedef decltype(api_data.args.hsa_signal_cas_relaxed.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_signal_cas_relaxed.value);
      out << ") = " << api_data.hsa_signal_value_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_cas_scacquire: {
      out << "hsa_signal_cas_scacquire(";
      typedef decltype(api_data.args.hsa_signal_cas_scacquire.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_cas_scacquire.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_cas_scacquire.expected) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_cas_scacquire.expected) << ", ";
      typedef decltype(api_data.args.hsa_signal_cas_scacquire.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_signal_cas_scacquire.value);
      out << ") = " << api_data.hsa_signal_value_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_cas_screlease: {
      out << "hsa_signal_cas_screlease(";
      typedef decltype(api_data.args.hsa_signal_cas_screlease.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_cas_screlease.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_cas_screlease.expected) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_cas_screlease.expected) << ", ";
      typedef decltype(api_data.args.hsa_signal_cas_screlease.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_signal_cas_screlease.value);
      out << ") = " << api_data.hsa_signal_value_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_cas_scacq_screl: {
      out << "hsa_signal_cas_scacq_screl(";
      typedef decltype(api_data.args.hsa_signal_cas_scacq_screl.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_cas_scacq_screl.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_cas_scacq_screl.expected) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_cas_scacq_screl.expected) << ", ";
      typedef decltype(api_data.args.hsa_signal_cas_scacq_screl.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_signal_cas_scacq_screl.value);
      out << ") = " << api_data.hsa_signal_value_t_retval;
      break;
    }
    case HSA_API_ID_hsa_isa_from_name: {
      out << "hsa_isa_from_name(";
      typedef decltype(api_data.args.hsa_isa_from_name.name) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_isa_from_name.name) << ", ";
      typedef decltype(api_data.args.hsa_isa_from_name.isa) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_isa_from_name.isa);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_isa_get_info: {
      out << "hsa_isa_get_info(";
      typedef decltype(api_data.args.hsa_isa_get_info.isa) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_isa_get_info.isa) << ", ";
      typedef decltype(api_data.args.hsa_isa_get_info.attribute) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_isa_get_info.attribute) << ", ";
      typedef decltype(api_data.args.hsa_isa_get_info.index) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_isa_get_info.index) << ", ";
      typedef decltype(api_data.args.hsa_isa_get_info.value) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_isa_get_info.value);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_isa_compatible: {
      out << "hsa_isa_compatible(";
      typedef decltype(api_data.args.hsa_isa_compatible.code_object_isa) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_isa_compatible.code_object_isa) << ", ";
      typedef decltype(api_data.args.hsa_isa_compatible.agent_isa) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_isa_compatible.agent_isa) << ", ";
      typedef decltype(api_data.args.hsa_isa_compatible.result) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_isa_compatible.result);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_code_object_serialize: {
      out << "hsa_code_object_serialize(";
      typedef decltype(api_data.args.hsa_code_object_serialize.code_object) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_code_object_serialize.code_object) << ", ";
      typedef decltype(api_data.args.hsa_code_object_serialize.alloc_callback) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_code_object_serialize.alloc_callback) << ", ";
      typedef decltype(api_data.args.hsa_code_object_serialize.callback_data) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_code_object_serialize.callback_data) << ", ";
      typedef decltype(api_data.args.hsa_code_object_serialize.options) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_code_object_serialize.options) << ", ";
      typedef decltype(api_data.args.hsa_code_object_serialize.serialized_code_object) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_code_object_serialize.serialized_code_object) << ", ";
      typedef decltype(api_data.args.hsa_code_object_serialize.serialized_code_object_size) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_code_object_serialize.serialized_code_object_size);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_code_object_deserialize: {
      out << "hsa_code_object_deserialize(";
      typedef decltype(api_data.args.hsa_code_object_deserialize.serialized_code_object) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_code_object_deserialize.serialized_code_object) << ", ";
      typedef decltype(api_data.args.hsa_code_object_deserialize.serialized_code_object_size) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_code_object_deserialize.serialized_code_object_size) << ", ";
      typedef decltype(api_data.args.hsa_code_object_deserialize.options) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_code_object_deserialize.options) << ", ";
      typedef decltype(api_data.args.hsa_code_object_deserialize.code_object) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_code_object_deserialize.code_object);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_code_object_destroy: {
      out << "hsa_code_object_destroy(";
      typedef decltype(api_data.args.hsa_code_object_destroy.code_object) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_code_object_destroy.code_object);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_code_object_get_info: {
      out << "hsa_code_object_get_info(";
      typedef decltype(api_data.args.hsa_code_object_get_info.code_object) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_code_object_get_info.code_object) << ", ";
      typedef decltype(api_data.args.hsa_code_object_get_info.attribute) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_code_object_get_info.attribute) << ", ";
      typedef decltype(api_data.args.hsa_code_object_get_info.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_code_object_get_info.value);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_code_object_get_symbol: {
      out << "hsa_code_object_get_symbol(";
      typedef decltype(api_data.args.hsa_code_object_get_symbol.code_object) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_code_object_get_symbol.code_object) << ", ";
      typedef decltype(api_data.args.hsa_code_object_get_symbol.symbol_name) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_code_object_get_symbol.symbol_name) << ", ";
      typedef decltype(api_data.args.hsa_code_object_get_symbol.symbol) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_code_object_get_symbol.symbol);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_code_symbol_get_info: {
      out << "hsa_code_symbol_get_info(";
      typedef decltype(api_data.args.hsa_code_symbol_get_info.code_symbol) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_code_symbol_get_info.code_symbol) << ", ";
      typedef decltype(api_data.args.hsa_code_symbol_get_info.attribute) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_code_symbol_get_info.attribute) << ", ";
      typedef decltype(api_data.args.hsa_code_symbol_get_info.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_code_symbol_get_info.value);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_code_object_iterate_symbols: {
      out << "hsa_code_object_iterate_symbols(";
      typedef decltype(api_data.args.hsa_code_object_iterate_symbols.code_object) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_code_object_iterate_symbols.code_object) << ", ";
      typedef decltype(api_data.args.hsa_code_object_iterate_symbols.callback) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_code_object_iterate_symbols.callback) << ", ";
      typedef decltype(api_data.args.hsa_code_object_iterate_symbols.data) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_code_object_iterate_symbols.data);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_create: {
      out << "hsa_executable_create(";
      typedef decltype(api_data.args.hsa_executable_create.profile) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_create.profile) << ", ";
      typedef decltype(api_data.args.hsa_executable_create.executable_state) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_create.executable_state) << ", ";
      typedef decltype(api_data.args.hsa_executable_create.options) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_create.options) << ", ";
      typedef decltype(api_data.args.hsa_executable_create.executable) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_executable_create.executable);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_destroy: {
      out << "hsa_executable_destroy(";
      typedef decltype(api_data.args.hsa_executable_destroy.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_destroy.executable);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_load_code_object: {
      out << "hsa_executable_load_code_object(";
      typedef decltype(api_data.args.hsa_executable_load_code_object.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_load_code_object.executable) << ", ";
      typedef decltype(api_data.args.hsa_executable_load_code_object.agent) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_load_code_object.agent) << ", ";
      typedef decltype(api_data.args.hsa_executable_load_code_object.code_object) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_load_code_object.code_object) << ", ";
      typedef decltype(api_data.args.hsa_executable_load_code_object.options) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_executable_load_code_object.options);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_freeze: {
      out << "hsa_executable_freeze(";
      typedef decltype(api_data.args.hsa_executable_freeze.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_freeze.executable) << ", ";
      typedef decltype(api_data.args.hsa_executable_freeze.options) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_freeze.options);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_get_info: {
      out << "hsa_executable_get_info(";
      typedef decltype(api_data.args.hsa_executable_get_info.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_get_info.executable) << ", ";
      typedef decltype(api_data.args.hsa_executable_get_info.attribute) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_get_info.attribute) << ", ";
      typedef decltype(api_data.args.hsa_executable_get_info.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_get_info.value);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_global_variable_define: {
      out << "hsa_executable_global_variable_define(";
      typedef decltype(api_data.args.hsa_executable_global_variable_define.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_global_variable_define.executable) << ", ";
      typedef decltype(api_data.args.hsa_executable_global_variable_define.variable_name) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_global_variable_define.variable_name) << ", ";
      typedef decltype(api_data.args.hsa_executable_global_variable_define.address) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_global_variable_define.address);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_agent_global_variable_define: {
      out << "hsa_executable_agent_global_variable_define(";
      typedef decltype(api_data.args.hsa_executable_agent_global_variable_define.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_agent_global_variable_define.executable) << ", ";
      typedef decltype(api_data.args.hsa_executable_agent_global_variable_define.agent) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_agent_global_variable_define.agent) << ", ";
      typedef decltype(api_data.args.hsa_executable_agent_global_variable_define.variable_name) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_agent_global_variable_define.variable_name) << ", ";
      typedef decltype(api_data.args.hsa_executable_agent_global_variable_define.address) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_executable_agent_global_variable_define.address);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_readonly_variable_define: {
      out << "hsa_executable_readonly_variable_define(";
      typedef decltype(api_data.args.hsa_executable_readonly_variable_define.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_readonly_variable_define.executable) << ", ";
      typedef decltype(api_data.args.hsa_executable_readonly_variable_define.agent) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_readonly_variable_define.agent) << ", ";
      typedef decltype(api_data.args.hsa_executable_readonly_variable_define.variable_name) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_readonly_variable_define.variable_name) << ", ";
      typedef decltype(api_data.args.hsa_executable_readonly_variable_define.address) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_executable_readonly_variable_define.address);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_validate: {
      out << "hsa_executable_validate(";
      typedef decltype(api_data.args.hsa_executable_validate.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_validate.executable) << ", ";
      typedef decltype(api_data.args.hsa_executable_validate.result) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_validate.result);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_get_symbol: {
      out << "hsa_executable_get_symbol(";
      typedef decltype(api_data.args.hsa_executable_get_symbol.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_get_symbol.executable) << ", ";
      typedef decltype(api_data.args.hsa_executable_get_symbol.module_name) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_get_symbol.module_name) << ", ";
      typedef decltype(api_data.args.hsa_executable_get_symbol.symbol_name) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_get_symbol.symbol_name) << ", ";
      typedef decltype(api_data.args.hsa_executable_get_symbol.agent) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_executable_get_symbol.agent) << ", ";
      typedef decltype(api_data.args.hsa_executable_get_symbol.call_convention) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_executable_get_symbol.call_convention) << ", ";
      typedef decltype(api_data.args.hsa_executable_get_symbol.symbol) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_executable_get_symbol.symbol);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_symbol_get_info: {
      out << "hsa_executable_symbol_get_info(";
      typedef decltype(api_data.args.hsa_executable_symbol_get_info.executable_symbol) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_symbol_get_info.executable_symbol) << ", ";
      typedef decltype(api_data.args.hsa_executable_symbol_get_info.attribute) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_symbol_get_info.attribute) << ", ";
      typedef decltype(api_data.args.hsa_executable_symbol_get_info.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_symbol_get_info.value);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_iterate_symbols: {
      out << "hsa_executable_iterate_symbols(";
      typedef decltype(api_data.args.hsa_executable_iterate_symbols.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_iterate_symbols.executable) << ", ";
      typedef decltype(api_data.args.hsa_executable_iterate_symbols.callback) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_iterate_symbols.callback) << ", ";
      typedef decltype(api_data.args.hsa_executable_iterate_symbols.data) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_iterate_symbols.data);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_status_string: {
      out << "hsa_status_string(";
      typedef decltype(api_data.args.hsa_status_string.status) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_status_string.status) << ", ";
      typedef decltype(api_data.args.hsa_status_string.status_string) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_status_string.status_string);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_extension_get_name: {
      out << "hsa_extension_get_name(";
      typedef decltype(api_data.args.hsa_extension_get_name.extension) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_extension_get_name.extension) << ", ";
      typedef decltype(api_data.args.hsa_extension_get_name.name) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_extension_get_name.name);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_system_major_extension_supported: {
      out << "hsa_system_major_extension_supported(";
      typedef decltype(api_data.args.hsa_system_major_extension_supported.extension) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_system_major_extension_supported.extension) << ", ";
      typedef decltype(api_data.args.hsa_system_major_extension_supported.version_major) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_system_major_extension_supported.version_major) << ", ";
      typedef decltype(api_data.args.hsa_system_major_extension_supported.version_minor) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_system_major_extension_supported.version_minor) << ", ";
      typedef decltype(api_data.args.hsa_system_major_extension_supported.result) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_system_major_extension_supported.result);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_system_get_major_extension_table: {
      out << "hsa_system_get_major_extension_table(";
      typedef decltype(api_data.args.hsa_system_get_major_extension_table.extension) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_system_get_major_extension_table.extension) << ", ";
      typedef decltype(api_data.args.hsa_system_get_major_extension_table.version_major) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_system_get_major_extension_table.version_major) << ", ";
      typedef decltype(api_data.args.hsa_system_get_major_extension_table.table_length) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_system_get_major_extension_table.table_length) << ", ";
      typedef decltype(api_data.args.hsa_system_get_major_extension_table.table) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_system_get_major_extension_table.table);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_agent_major_extension_supported: {
      out << "hsa_agent_major_extension_supported(";
      typedef decltype(api_data.args.hsa_agent_major_extension_supported.extension) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_agent_major_extension_supported.extension) << ", ";
      typedef decltype(api_data.args.hsa_agent_major_extension_supported.agent) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_agent_major_extension_supported.agent) << ", ";
      typedef decltype(api_data.args.hsa_agent_major_extension_supported.version_major) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_agent_major_extension_supported.version_major) << ", ";
      typedef decltype(api_data.args.hsa_agent_major_extension_supported.version_minor) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_agent_major_extension_supported.version_minor) << ", ";
      typedef decltype(api_data.args.hsa_agent_major_extension_supported.result) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_agent_major_extension_supported.result);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_cache_get_info: {
      out << "hsa_cache_get_info(";
      typedef decltype(api_data.args.hsa_cache_get_info.cache) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_cache_get_info.cache) << ", ";
      typedef decltype(api_data.args.hsa_cache_get_info.attribute) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_cache_get_info.attribute) << ", ";
      typedef decltype(api_data.args.hsa_cache_get_info.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_cache_get_info.value);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_agent_iterate_caches: {
      out << "hsa_agent_iterate_caches(";
      typedef decltype(api_data.args.hsa_agent_iterate_caches.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_agent_iterate_caches.agent) << ", ";
      typedef decltype(api_data.args.hsa_agent_iterate_caches.callback) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_agent_iterate_caches.callback) << ", ";
      typedef decltype(api_data.args.hsa_agent_iterate_caches.data) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_agent_iterate_caches.data);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_silent_store_relaxed: {
      out << "hsa_signal_silent_store_relaxed(";
      typedef decltype(api_data.args.hsa_signal_silent_store_relaxed.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_silent_store_relaxed.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_silent_store_relaxed.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_silent_store_relaxed.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_silent_store_screlease: {
      out << "hsa_signal_silent_store_screlease(";
      typedef decltype(api_data.args.hsa_signal_silent_store_screlease.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_silent_store_screlease.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_silent_store_screlease.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_silent_store_screlease.value);
      out << ") = void";
      break;
    }
    case HSA_API_ID_hsa_signal_group_create: {
      out << "hsa_signal_group_create(";
      typedef decltype(api_data.args.hsa_signal_group_create.num_signals) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_group_create.num_signals) << ", ";
      typedef decltype(api_data.args.hsa_signal_group_create.signals) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_group_create.signals) << ", ";
      typedef decltype(api_data.args.hsa_signal_group_create.num_consumers) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_signal_group_create.num_consumers) << ", ";
      typedef decltype(api_data.args.hsa_signal_group_create.consumers) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_signal_group_create.consumers) << ", ";
      typedef decltype(api_data.args.hsa_signal_group_create.signal_group) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_signal_group_create.signal_group);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_group_destroy: {
      out << "hsa_signal_group_destroy(";
      typedef decltype(api_data.args.hsa_signal_group_destroy.signal_group) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_group_destroy.signal_group);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_group_wait_any_scacquire: {
      out << "hsa_signal_group_wait_any_scacquire(";
      typedef decltype(api_data.args.hsa_signal_group_wait_any_scacquire.signal_group) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_group_wait_any_scacquire.signal_group) << ", ";
      typedef decltype(api_data.args.hsa_signal_group_wait_any_scacquire.conditions) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_group_wait_any_scacquire.conditions) << ", ";
      typedef decltype(api_data.args.hsa_signal_group_wait_any_scacquire.compare_values) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_signal_group_wait_any_scacquire.compare_values) << ", ";
      typedef decltype(api_data.args.hsa_signal_group_wait_any_scacquire.wait_state_hint) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_signal_group_wait_any_scacquire.wait_state_hint) << ", ";
      typedef decltype(api_data.args.hsa_signal_group_wait_any_scacquire.signal) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_signal_group_wait_any_scacquire.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_group_wait_any_scacquire.value) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_signal_group_wait_any_scacquire.value);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_signal_group_wait_any_relaxed: {
      out << "hsa_signal_group_wait_any_relaxed(";
      typedef decltype(api_data.args.hsa_signal_group_wait_any_relaxed.signal_group) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_signal_group_wait_any_relaxed.signal_group) << ", ";
      typedef decltype(api_data.args.hsa_signal_group_wait_any_relaxed.conditions) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_signal_group_wait_any_relaxed.conditions) << ", ";
      typedef decltype(api_data.args.hsa_signal_group_wait_any_relaxed.compare_values) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_signal_group_wait_any_relaxed.compare_values) << ", ";
      typedef decltype(api_data.args.hsa_signal_group_wait_any_relaxed.wait_state_hint) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_signal_group_wait_any_relaxed.wait_state_hint) << ", ";
      typedef decltype(api_data.args.hsa_signal_group_wait_any_relaxed.signal) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_signal_group_wait_any_relaxed.signal) << ", ";
      typedef decltype(api_data.args.hsa_signal_group_wait_any_relaxed.value) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_signal_group_wait_any_relaxed.value);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_agent_iterate_isas: {
      out << "hsa_agent_iterate_isas(";
      typedef decltype(api_data.args.hsa_agent_iterate_isas.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_agent_iterate_isas.agent) << ", ";
      typedef decltype(api_data.args.hsa_agent_iterate_isas.callback) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_agent_iterate_isas.callback) << ", ";
      typedef decltype(api_data.args.hsa_agent_iterate_isas.data) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_agent_iterate_isas.data);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_isa_get_info_alt: {
      out << "hsa_isa_get_info_alt(";
      typedef decltype(api_data.args.hsa_isa_get_info_alt.isa) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_isa_get_info_alt.isa) << ", ";
      typedef decltype(api_data.args.hsa_isa_get_info_alt.attribute) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_isa_get_info_alt.attribute) << ", ";
      typedef decltype(api_data.args.hsa_isa_get_info_alt.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_isa_get_info_alt.value);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_isa_get_exception_policies: {
      out << "hsa_isa_get_exception_policies(";
      typedef decltype(api_data.args.hsa_isa_get_exception_policies.isa) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_isa_get_exception_policies.isa) << ", ";
      typedef decltype(api_data.args.hsa_isa_get_exception_policies.profile) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_isa_get_exception_policies.profile) << ", ";
      typedef decltype(api_data.args.hsa_isa_get_exception_policies.mask) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_isa_get_exception_policies.mask);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_isa_get_round_method: {
      out << "hsa_isa_get_round_method(";
      typedef decltype(api_data.args.hsa_isa_get_round_method.isa) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_isa_get_round_method.isa) << ", ";
      typedef decltype(api_data.args.hsa_isa_get_round_method.fp_type) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_isa_get_round_method.fp_type) << ", ";
      typedef decltype(api_data.args.hsa_isa_get_round_method.flush_mode) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_isa_get_round_method.flush_mode) << ", ";
      typedef decltype(api_data.args.hsa_isa_get_round_method.round_method) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_isa_get_round_method.round_method);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_wavefront_get_info: {
      out << "hsa_wavefront_get_info(";
      typedef decltype(api_data.args.hsa_wavefront_get_info.wavefront) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_wavefront_get_info.wavefront) << ", ";
      typedef decltype(api_data.args.hsa_wavefront_get_info.attribute) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_wavefront_get_info.attribute) << ", ";
      typedef decltype(api_data.args.hsa_wavefront_get_info.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_wavefront_get_info.value);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_isa_iterate_wavefronts: {
      out << "hsa_isa_iterate_wavefronts(";
      typedef decltype(api_data.args.hsa_isa_iterate_wavefronts.isa) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_isa_iterate_wavefronts.isa) << ", ";
      typedef decltype(api_data.args.hsa_isa_iterate_wavefronts.callback) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_isa_iterate_wavefronts.callback) << ", ";
      typedef decltype(api_data.args.hsa_isa_iterate_wavefronts.data) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_isa_iterate_wavefronts.data);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_code_object_get_symbol_from_name: {
      out << "hsa_code_object_get_symbol_from_name(";
      typedef decltype(api_data.args.hsa_code_object_get_symbol_from_name.code_object) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_code_object_get_symbol_from_name.code_object) << ", ";
      typedef decltype(api_data.args.hsa_code_object_get_symbol_from_name.module_name) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_code_object_get_symbol_from_name.module_name) << ", ";
      typedef decltype(api_data.args.hsa_code_object_get_symbol_from_name.symbol_name) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_code_object_get_symbol_from_name.symbol_name) << ", ";
      typedef decltype(api_data.args.hsa_code_object_get_symbol_from_name.symbol) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_code_object_get_symbol_from_name.symbol);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_code_object_reader_create_from_file: {
      out << "hsa_code_object_reader_create_from_file(";
      typedef decltype(api_data.args.hsa_code_object_reader_create_from_file.file) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_code_object_reader_create_from_file.file) << ", ";
      typedef decltype(api_data.args.hsa_code_object_reader_create_from_file.code_object_reader) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_code_object_reader_create_from_file.code_object_reader);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_code_object_reader_create_from_memory: {
      out << "hsa_code_object_reader_create_from_memory(";
      typedef decltype(api_data.args.hsa_code_object_reader_create_from_memory.code_object) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_code_object_reader_create_from_memory.code_object) << ", ";
      typedef decltype(api_data.args.hsa_code_object_reader_create_from_memory.size) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_code_object_reader_create_from_memory.size) << ", ";
      typedef decltype(api_data.args.hsa_code_object_reader_create_from_memory.code_object_reader) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_code_object_reader_create_from_memory.code_object_reader);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_code_object_reader_destroy: {
      out << "hsa_code_object_reader_destroy(";
      typedef decltype(api_data.args.hsa_code_object_reader_destroy.code_object_reader) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_code_object_reader_destroy.code_object_reader);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_create_alt: {
      out << "hsa_executable_create_alt(";
      typedef decltype(api_data.args.hsa_executable_create_alt.profile) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_create_alt.profile) << ", ";
      typedef decltype(api_data.args.hsa_executable_create_alt.default_float_rounding_mode) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_create_alt.default_float_rounding_mode) << ", ";
      typedef decltype(api_data.args.hsa_executable_create_alt.options) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_create_alt.options) << ", ";
      typedef decltype(api_data.args.hsa_executable_create_alt.executable) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_executable_create_alt.executable);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_load_program_code_object: {
      out << "hsa_executable_load_program_code_object(";
      typedef decltype(api_data.args.hsa_executable_load_program_code_object.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_load_program_code_object.executable) << ", ";
      typedef decltype(api_data.args.hsa_executable_load_program_code_object.code_object_reader) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_load_program_code_object.code_object_reader) << ", ";
      typedef decltype(api_data.args.hsa_executable_load_program_code_object.options) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_load_program_code_object.options) << ", ";
      typedef decltype(api_data.args.hsa_executable_load_program_code_object.loaded_code_object) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_executable_load_program_code_object.loaded_code_object);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_load_agent_code_object: {
      out << "hsa_executable_load_agent_code_object(";
      typedef decltype(api_data.args.hsa_executable_load_agent_code_object.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_load_agent_code_object.executable) << ", ";
      typedef decltype(api_data.args.hsa_executable_load_agent_code_object.agent) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_load_agent_code_object.agent) << ", ";
      typedef decltype(api_data.args.hsa_executable_load_agent_code_object.code_object_reader) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_load_agent_code_object.code_object_reader) << ", ";
      typedef decltype(api_data.args.hsa_executable_load_agent_code_object.options) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_executable_load_agent_code_object.options) << ", ";
      typedef decltype(api_data.args.hsa_executable_load_agent_code_object.loaded_code_object) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_executable_load_agent_code_object.loaded_code_object);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_validate_alt: {
      out << "hsa_executable_validate_alt(";
      typedef decltype(api_data.args.hsa_executable_validate_alt.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_validate_alt.executable) << ", ";
      typedef decltype(api_data.args.hsa_executable_validate_alt.options) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_validate_alt.options) << ", ";
      typedef decltype(api_data.args.hsa_executable_validate_alt.result) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_validate_alt.result);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_get_symbol_by_name: {
      out << "hsa_executable_get_symbol_by_name(";
      typedef decltype(api_data.args.hsa_executable_get_symbol_by_name.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_get_symbol_by_name.executable) << ", ";
      typedef decltype(api_data.args.hsa_executable_get_symbol_by_name.symbol_name) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_get_symbol_by_name.symbol_name) << ", ";
      typedef decltype(api_data.args.hsa_executable_get_symbol_by_name.agent) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_get_symbol_by_name.agent) << ", ";
      typedef decltype(api_data.args.hsa_executable_get_symbol_by_name.symbol) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_executable_get_symbol_by_name.symbol);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_iterate_agent_symbols: {
      out << "hsa_executable_iterate_agent_symbols(";
      typedef decltype(api_data.args.hsa_executable_iterate_agent_symbols.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_iterate_agent_symbols.executable) << ", ";
      typedef decltype(api_data.args.hsa_executable_iterate_agent_symbols.agent) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_iterate_agent_symbols.agent) << ", ";
      typedef decltype(api_data.args.hsa_executable_iterate_agent_symbols.callback) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_iterate_agent_symbols.callback) << ", ";
      typedef decltype(api_data.args.hsa_executable_iterate_agent_symbols.data) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_executable_iterate_agent_symbols.data);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_executable_iterate_program_symbols: {
      out << "hsa_executable_iterate_program_symbols(";
      typedef decltype(api_data.args.hsa_executable_iterate_program_symbols.executable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_executable_iterate_program_symbols.executable) << ", ";
      typedef decltype(api_data.args.hsa_executable_iterate_program_symbols.callback) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_executable_iterate_program_symbols.callback) << ", ";
      typedef decltype(api_data.args.hsa_executable_iterate_program_symbols.data) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_executable_iterate_program_symbols.data);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }

    // block: AmdExtTable API
    case HSA_API_ID_hsa_amd_coherency_get_type: {
      out << "hsa_amd_coherency_get_type(";
      typedef decltype(api_data.args.hsa_amd_coherency_get_type.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_coherency_get_type.agent) << ", ";
      typedef decltype(api_data.args.hsa_amd_coherency_get_type.type) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_coherency_get_type.type);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_coherency_set_type: {
      out << "hsa_amd_coherency_set_type(";
      typedef decltype(api_data.args.hsa_amd_coherency_set_type.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_coherency_set_type.agent) << ", ";
      typedef decltype(api_data.args.hsa_amd_coherency_set_type.type) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_coherency_set_type.type);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_profiling_set_profiler_enabled: {
      out << "hsa_amd_profiling_set_profiler_enabled(";
      typedef decltype(api_data.args.hsa_amd_profiling_set_profiler_enabled.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_profiling_set_profiler_enabled.queue) << ", ";
      typedef decltype(api_data.args.hsa_amd_profiling_set_profiler_enabled.enable) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_profiling_set_profiler_enabled.enable);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_profiling_async_copy_enable: {
      out << "hsa_amd_profiling_async_copy_enable(";
      typedef decltype(api_data.args.hsa_amd_profiling_async_copy_enable.enable) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_profiling_async_copy_enable.enable);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_profiling_get_dispatch_time: {
      out << "hsa_amd_profiling_get_dispatch_time(";
      typedef decltype(api_data.args.hsa_amd_profiling_get_dispatch_time.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_profiling_get_dispatch_time.agent) << ", ";
      typedef decltype(api_data.args.hsa_amd_profiling_get_dispatch_time.signal) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_profiling_get_dispatch_time.signal) << ", ";
      typedef decltype(api_data.args.hsa_amd_profiling_get_dispatch_time.time) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_profiling_get_dispatch_time.time);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_profiling_get_async_copy_time: {
      out << "hsa_amd_profiling_get_async_copy_time(";
      typedef decltype(api_data.args.hsa_amd_profiling_get_async_copy_time.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_profiling_get_async_copy_time.signal) << ", ";
      typedef decltype(api_data.args.hsa_amd_profiling_get_async_copy_time.time) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_profiling_get_async_copy_time.time);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_profiling_convert_tick_to_system_domain: {
      out << "hsa_amd_profiling_convert_tick_to_system_domain(";
      typedef decltype(api_data.args.hsa_amd_profiling_convert_tick_to_system_domain.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_profiling_convert_tick_to_system_domain.agent) << ", ";
      typedef decltype(api_data.args.hsa_amd_profiling_convert_tick_to_system_domain.agent_tick) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_profiling_convert_tick_to_system_domain.agent_tick) << ", ";
      typedef decltype(api_data.args.hsa_amd_profiling_convert_tick_to_system_domain.system_tick) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_profiling_convert_tick_to_system_domain.system_tick);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_signal_async_handler: {
      out << "hsa_amd_signal_async_handler(";
      typedef decltype(api_data.args.hsa_amd_signal_async_handler.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_signal_async_handler.signal) << ", ";
      typedef decltype(api_data.args.hsa_amd_signal_async_handler.cond) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_signal_async_handler.cond) << ", ";
      typedef decltype(api_data.args.hsa_amd_signal_async_handler.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_signal_async_handler.value) << ", ";
      typedef decltype(api_data.args.hsa_amd_signal_async_handler.handler) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_amd_signal_async_handler.handler) << ", ";
      typedef decltype(api_data.args.hsa_amd_signal_async_handler.arg) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_amd_signal_async_handler.arg);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_async_function: {
      out << "hsa_amd_async_function(";
      typedef decltype(api_data.args.hsa_amd_async_function.callback) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_async_function.callback) << ", ";
      typedef decltype(api_data.args.hsa_amd_async_function.arg) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_async_function.arg);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_signal_wait_any: {
      out << "hsa_amd_signal_wait_any(";
      typedef decltype(api_data.args.hsa_amd_signal_wait_any.signal_count) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_signal_wait_any.signal_count) << ", ";
      typedef decltype(api_data.args.hsa_amd_signal_wait_any.signals) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_signal_wait_any.signals) << ", ";
      typedef decltype(api_data.args.hsa_amd_signal_wait_any.conds) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_signal_wait_any.conds) << ", ";
      typedef decltype(api_data.args.hsa_amd_signal_wait_any.values) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_amd_signal_wait_any.values) << ", ";
      typedef decltype(api_data.args.hsa_amd_signal_wait_any.timeout_hint) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_amd_signal_wait_any.timeout_hint) << ", ";
      typedef decltype(api_data.args.hsa_amd_signal_wait_any.wait_hint) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_amd_signal_wait_any.wait_hint) << ", ";
      typedef decltype(api_data.args.hsa_amd_signal_wait_any.satisfying_value) arg_val_type_t6;
      roctracer::hsa_support::output_streamer<arg_val_type_t6>::put(out, api_data.args.hsa_amd_signal_wait_any.satisfying_value);
      out << ") = " << api_data.uint32_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_queue_cu_set_mask: {
      out << "hsa_amd_queue_cu_set_mask(";
      typedef decltype(api_data.args.hsa_amd_queue_cu_set_mask.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_queue_cu_set_mask.queue) << ", ";
      typedef decltype(api_data.args.hsa_amd_queue_cu_set_mask.num_cu_mask_count) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_queue_cu_set_mask.num_cu_mask_count) << ", ";
      typedef decltype(api_data.args.hsa_amd_queue_cu_set_mask.cu_mask) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_queue_cu_set_mask.cu_mask);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_memory_pool_get_info: {
      out << "hsa_amd_memory_pool_get_info(";
      typedef decltype(api_data.args.hsa_amd_memory_pool_get_info.memory_pool) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_memory_pool_get_info.memory_pool) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_pool_get_info.attribute) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_memory_pool_get_info.attribute) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_pool_get_info.value) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_memory_pool_get_info.value);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_agent_iterate_memory_pools: {
      out << "hsa_amd_agent_iterate_memory_pools(";
      typedef decltype(api_data.args.hsa_amd_agent_iterate_memory_pools.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_agent_iterate_memory_pools.agent) << ", ";
      typedef decltype(api_data.args.hsa_amd_agent_iterate_memory_pools.callback) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_agent_iterate_memory_pools.callback) << ", ";
      typedef decltype(api_data.args.hsa_amd_agent_iterate_memory_pools.data) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_agent_iterate_memory_pools.data);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_memory_pool_allocate: {
      out << "hsa_amd_memory_pool_allocate(";
      typedef decltype(api_data.args.hsa_amd_memory_pool_allocate.memory_pool) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_memory_pool_allocate.memory_pool) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_pool_allocate.size) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_memory_pool_allocate.size) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_pool_allocate.flags) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_memory_pool_allocate.flags) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_pool_allocate.ptr) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_amd_memory_pool_allocate.ptr);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_memory_pool_free: {
      out << "hsa_amd_memory_pool_free(";
      typedef decltype(api_data.args.hsa_amd_memory_pool_free.ptr) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_memory_pool_free.ptr);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_memory_async_copy: {
      out << "hsa_amd_memory_async_copy(";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy.dst) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_memory_async_copy.dst) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy.dst_agent) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_memory_async_copy.dst_agent) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy.src) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_memory_async_copy.src) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy.src_agent) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_amd_memory_async_copy.src_agent) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy.size) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_amd_memory_async_copy.size) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy.num_dep_signals) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_amd_memory_async_copy.num_dep_signals) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy.dep_signals) arg_val_type_t6;
      roctracer::hsa_support::output_streamer<arg_val_type_t6>::put(out, api_data.args.hsa_amd_memory_async_copy.dep_signals) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy.completion_signal) arg_val_type_t7;
      roctracer::hsa_support::output_streamer<arg_val_type_t7>::put(out, api_data.args.hsa_amd_memory_async_copy.completion_signal);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_agent_memory_pool_get_info: {
      out << "hsa_amd_agent_memory_pool_get_info(";
      typedef decltype(api_data.args.hsa_amd_agent_memory_pool_get_info.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_agent_memory_pool_get_info.agent) << ", ";
      typedef decltype(api_data.args.hsa_amd_agent_memory_pool_get_info.memory_pool) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_agent_memory_pool_get_info.memory_pool) << ", ";
      typedef decltype(api_data.args.hsa_amd_agent_memory_pool_get_info.attribute) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_agent_memory_pool_get_info.attribute) << ", ";
      typedef decltype(api_data.args.hsa_amd_agent_memory_pool_get_info.value) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_amd_agent_memory_pool_get_info.value);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_agents_allow_access: {
      out << "hsa_amd_agents_allow_access(";
      typedef decltype(api_data.args.hsa_amd_agents_allow_access.num_agents) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_agents_allow_access.num_agents) << ", ";
      typedef decltype(api_data.args.hsa_amd_agents_allow_access.agents) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_agents_allow_access.agents) << ", ";
      typedef decltype(api_data.args.hsa_amd_agents_allow_access.flags) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_agents_allow_access.flags) << ", ";
      typedef decltype(api_data.args.hsa_amd_agents_allow_access.ptr) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_amd_agents_allow_access.ptr);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_memory_pool_can_migrate: {
      out << "hsa_amd_memory_pool_can_migrate(";
      typedef decltype(api_data.args.hsa_amd_memory_pool_can_migrate.src_memory_pool) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_memory_pool_can_migrate.src_memory_pool) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_pool_can_migrate.dst_memory_pool) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_memory_pool_can_migrate.dst_memory_pool) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_pool_can_migrate.result) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_memory_pool_can_migrate.result);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_memory_migrate: {
      out << "hsa_amd_memory_migrate(";
      typedef decltype(api_data.args.hsa_amd_memory_migrate.ptr) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_memory_migrate.ptr) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_migrate.memory_pool) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_memory_migrate.memory_pool) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_migrate.flags) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_memory_migrate.flags);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_memory_lock: {
      out << "hsa_amd_memory_lock(";
      typedef decltype(api_data.args.hsa_amd_memory_lock.host_ptr) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_memory_lock.host_ptr) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_lock.size) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_memory_lock.size) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_lock.agents) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_memory_lock.agents) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_lock.num_agent) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_amd_memory_lock.num_agent) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_lock.agent_ptr) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_amd_memory_lock.agent_ptr);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_memory_unlock: {
      out << "hsa_amd_memory_unlock(";
      typedef decltype(api_data.args.hsa_amd_memory_unlock.host_ptr) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_memory_unlock.host_ptr);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_memory_fill: {
      out << "hsa_amd_memory_fill(";
      typedef decltype(api_data.args.hsa_amd_memory_fill.ptr) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_memory_fill.ptr) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_fill.value) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_memory_fill.value) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_fill.count) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_memory_fill.count);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_interop_map_buffer: {
      out << "hsa_amd_interop_map_buffer(";
      typedef decltype(api_data.args.hsa_amd_interop_map_buffer.num_agents) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_interop_map_buffer.num_agents) << ", ";
      typedef decltype(api_data.args.hsa_amd_interop_map_buffer.agents) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_interop_map_buffer.agents) << ", ";
      typedef decltype(api_data.args.hsa_amd_interop_map_buffer.interop_handle) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_interop_map_buffer.interop_handle) << ", ";
      typedef decltype(api_data.args.hsa_amd_interop_map_buffer.flags) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_amd_interop_map_buffer.flags) << ", ";
      typedef decltype(api_data.args.hsa_amd_interop_map_buffer.size) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_amd_interop_map_buffer.size) << ", ";
      typedef decltype(api_data.args.hsa_amd_interop_map_buffer.ptr) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_amd_interop_map_buffer.ptr) << ", ";
      typedef decltype(api_data.args.hsa_amd_interop_map_buffer.metadata_size) arg_val_type_t6;
      roctracer::hsa_support::output_streamer<arg_val_type_t6>::put(out, api_data.args.hsa_amd_interop_map_buffer.metadata_size) << ", ";
      typedef decltype(api_data.args.hsa_amd_interop_map_buffer.metadata) arg_val_type_t7;
      roctracer::hsa_support::output_streamer<arg_val_type_t7>::put(out, api_data.args.hsa_amd_interop_map_buffer.metadata);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_interop_unmap_buffer: {
      out << "hsa_amd_interop_unmap_buffer(";
      typedef decltype(api_data.args.hsa_amd_interop_unmap_buffer.ptr) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_interop_unmap_buffer.ptr);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_image_create: {
      out << "hsa_amd_image_create(";
      typedef decltype(api_data.args.hsa_amd_image_create.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_image_create.agent) << ", ";
      typedef decltype(api_data.args.hsa_amd_image_create.image_descriptor) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_image_create.image_descriptor) << ", ";
      typedef decltype(api_data.args.hsa_amd_image_create.image_layout) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_image_create.image_layout) << ", ";
      typedef decltype(api_data.args.hsa_amd_image_create.image_data) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_amd_image_create.image_data) << ", ";
      typedef decltype(api_data.args.hsa_amd_image_create.access_permission) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_amd_image_create.access_permission) << ", ";
      typedef decltype(api_data.args.hsa_amd_image_create.image) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_amd_image_create.image);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_pointer_info: {
      out << "hsa_amd_pointer_info(";
      typedef decltype(api_data.args.hsa_amd_pointer_info.ptr) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_pointer_info.ptr) << ", ";
      typedef decltype(api_data.args.hsa_amd_pointer_info.info) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_pointer_info.info) << ", ";
      typedef decltype(api_data.args.hsa_amd_pointer_info.alloc) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_pointer_info.alloc) << ", ";
      typedef decltype(api_data.args.hsa_amd_pointer_info.num_agents_accessible) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_amd_pointer_info.num_agents_accessible) << ", ";
      typedef decltype(api_data.args.hsa_amd_pointer_info.accessible) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_amd_pointer_info.accessible);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_pointer_info_set_userdata: {
      out << "hsa_amd_pointer_info_set_userdata(";
      typedef decltype(api_data.args.hsa_amd_pointer_info_set_userdata.ptr) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_pointer_info_set_userdata.ptr) << ", ";
      typedef decltype(api_data.args.hsa_amd_pointer_info_set_userdata.userdata) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_pointer_info_set_userdata.userdata);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_ipc_memory_create: {
      out << "hsa_amd_ipc_memory_create(";
      typedef decltype(api_data.args.hsa_amd_ipc_memory_create.ptr) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_ipc_memory_create.ptr) << ", ";
      typedef decltype(api_data.args.hsa_amd_ipc_memory_create.len) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_ipc_memory_create.len) << ", ";
      typedef decltype(api_data.args.hsa_amd_ipc_memory_create.handle) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_ipc_memory_create.handle);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_ipc_memory_attach: {
      out << "hsa_amd_ipc_memory_attach(";
      typedef decltype(api_data.args.hsa_amd_ipc_memory_attach.handle) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_ipc_memory_attach.handle) << ", ";
      typedef decltype(api_data.args.hsa_amd_ipc_memory_attach.len) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_ipc_memory_attach.len) << ", ";
      typedef decltype(api_data.args.hsa_amd_ipc_memory_attach.num_agents) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_ipc_memory_attach.num_agents) << ", ";
      typedef decltype(api_data.args.hsa_amd_ipc_memory_attach.mapping_agents) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_amd_ipc_memory_attach.mapping_agents) << ", ";
      typedef decltype(api_data.args.hsa_amd_ipc_memory_attach.mapped_ptr) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_amd_ipc_memory_attach.mapped_ptr);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_ipc_memory_detach: {
      out << "hsa_amd_ipc_memory_detach(";
      typedef decltype(api_data.args.hsa_amd_ipc_memory_detach.mapped_ptr) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_ipc_memory_detach.mapped_ptr);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_signal_create: {
      out << "hsa_amd_signal_create(";
      typedef decltype(api_data.args.hsa_amd_signal_create.initial_value) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_signal_create.initial_value) << ", ";
      typedef decltype(api_data.args.hsa_amd_signal_create.num_consumers) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_signal_create.num_consumers) << ", ";
      typedef decltype(api_data.args.hsa_amd_signal_create.consumers) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_signal_create.consumers) << ", ";
      typedef decltype(api_data.args.hsa_amd_signal_create.attributes) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_amd_signal_create.attributes) << ", ";
      typedef decltype(api_data.args.hsa_amd_signal_create.signal) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_amd_signal_create.signal);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_ipc_signal_create: {
      out << "hsa_amd_ipc_signal_create(";
      typedef decltype(api_data.args.hsa_amd_ipc_signal_create.signal) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_ipc_signal_create.signal) << ", ";
      typedef decltype(api_data.args.hsa_amd_ipc_signal_create.handle) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_ipc_signal_create.handle);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_ipc_signal_attach: {
      out << "hsa_amd_ipc_signal_attach(";
      typedef decltype(api_data.args.hsa_amd_ipc_signal_attach.handle) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_ipc_signal_attach.handle) << ", ";
      typedef decltype(api_data.args.hsa_amd_ipc_signal_attach.signal) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_ipc_signal_attach.signal);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_register_system_event_handler: {
      out << "hsa_amd_register_system_event_handler(";
      typedef decltype(api_data.args.hsa_amd_register_system_event_handler.callback) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_register_system_event_handler.callback) << ", ";
      typedef decltype(api_data.args.hsa_amd_register_system_event_handler.data) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_register_system_event_handler.data);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_queue_intercept_create: {
      out << "hsa_amd_queue_intercept_create(";
      typedef decltype(api_data.args.hsa_amd_queue_intercept_create.agent_handle) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_queue_intercept_create.agent_handle) << ", ";
      typedef decltype(api_data.args.hsa_amd_queue_intercept_create.size) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_queue_intercept_create.size) << ", ";
      typedef decltype(api_data.args.hsa_amd_queue_intercept_create.type) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_queue_intercept_create.type) << ", ";
      typedef decltype(api_data.args.hsa_amd_queue_intercept_create.callback) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_amd_queue_intercept_create.callback) << ", ";
      typedef decltype(api_data.args.hsa_amd_queue_intercept_create.data) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_amd_queue_intercept_create.data) << ", ";
      typedef decltype(api_data.args.hsa_amd_queue_intercept_create.private_segment_size) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_amd_queue_intercept_create.private_segment_size) << ", ";
      typedef decltype(api_data.args.hsa_amd_queue_intercept_create.group_segment_size) arg_val_type_t6;
      roctracer::hsa_support::output_streamer<arg_val_type_t6>::put(out, api_data.args.hsa_amd_queue_intercept_create.group_segment_size) << ", ";
      typedef decltype(api_data.args.hsa_amd_queue_intercept_create.queue) arg_val_type_t7;
      roctracer::hsa_support::output_streamer<arg_val_type_t7>::put(out, api_data.args.hsa_amd_queue_intercept_create.queue);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_queue_intercept_register: {
      out << "hsa_amd_queue_intercept_register(";
      typedef decltype(api_data.args.hsa_amd_queue_intercept_register.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_queue_intercept_register.queue) << ", ";
      typedef decltype(api_data.args.hsa_amd_queue_intercept_register.callback) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_queue_intercept_register.callback) << ", ";
      typedef decltype(api_data.args.hsa_amd_queue_intercept_register.user_data) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_queue_intercept_register.user_data);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_queue_set_priority: {
      out << "hsa_amd_queue_set_priority(";
      typedef decltype(api_data.args.hsa_amd_queue_set_priority.queue) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_queue_set_priority.queue) << ", ";
      typedef decltype(api_data.args.hsa_amd_queue_set_priority.priority) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_queue_set_priority.priority);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_memory_async_copy_rect: {
      out << "hsa_amd_memory_async_copy_rect(";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy_rect.dst) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_memory_async_copy_rect.dst) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy_rect.dst_offset) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_memory_async_copy_rect.dst_offset) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy_rect.src) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_amd_memory_async_copy_rect.src) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy_rect.src_offset) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_amd_memory_async_copy_rect.src_offset) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy_rect.range) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_amd_memory_async_copy_rect.range) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy_rect.copy_agent) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_amd_memory_async_copy_rect.copy_agent) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy_rect.dir) arg_val_type_t6;
      roctracer::hsa_support::output_streamer<arg_val_type_t6>::put(out, api_data.args.hsa_amd_memory_async_copy_rect.dir) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy_rect.num_dep_signals) arg_val_type_t7;
      roctracer::hsa_support::output_streamer<arg_val_type_t7>::put(out, api_data.args.hsa_amd_memory_async_copy_rect.num_dep_signals) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy_rect.dep_signals) arg_val_type_t8;
      roctracer::hsa_support::output_streamer<arg_val_type_t8>::put(out, api_data.args.hsa_amd_memory_async_copy_rect.dep_signals) << ", ";
      typedef decltype(api_data.args.hsa_amd_memory_async_copy_rect.completion_signal) arg_val_type_t9;
      roctracer::hsa_support::output_streamer<arg_val_type_t9>::put(out, api_data.args.hsa_amd_memory_async_copy_rect.completion_signal);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_amd_runtime_queue_create_register: {
      out << "hsa_amd_runtime_queue_create_register(";
      typedef decltype(api_data.args.hsa_amd_runtime_queue_create_register.callback) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_amd_runtime_queue_create_register.callback) << ", ";
      typedef decltype(api_data.args.hsa_amd_runtime_queue_create_register.user_data) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_amd_runtime_queue_create_register.user_data);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }

    // block: ImageExtTable API
    case HSA_API_ID_hsa_ext_image_get_capability: {
      out << "hsa_ext_image_get_capability(";
      typedef decltype(api_data.args.hsa_ext_image_get_capability.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_ext_image_get_capability.agent) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_get_capability.geometry) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_ext_image_get_capability.geometry) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_get_capability.image_format) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_ext_image_get_capability.image_format) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_get_capability.capability_mask) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_ext_image_get_capability.capability_mask);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_ext_image_data_get_info: {
      out << "hsa_ext_image_data_get_info(";
      typedef decltype(api_data.args.hsa_ext_image_data_get_info.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_ext_image_data_get_info.agent) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_data_get_info.image_descriptor) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_ext_image_data_get_info.image_descriptor) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_data_get_info.access_permission) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_ext_image_data_get_info.access_permission) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_data_get_info.image_data_info) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_ext_image_data_get_info.image_data_info);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_ext_image_create: {
      out << "hsa_ext_image_create(";
      typedef decltype(api_data.args.hsa_ext_image_create.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_ext_image_create.agent) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_create.image_descriptor) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_ext_image_create.image_descriptor) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_create.image_data) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_ext_image_create.image_data) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_create.access_permission) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_ext_image_create.access_permission) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_create.image) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_ext_image_create.image);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_ext_image_import: {
      out << "hsa_ext_image_import(";
      typedef decltype(api_data.args.hsa_ext_image_import.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_ext_image_import.agent) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_import.src_memory) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_ext_image_import.src_memory) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_import.src_row_pitch) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_ext_image_import.src_row_pitch) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_import.src_slice_pitch) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_ext_image_import.src_slice_pitch) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_import.dst_image) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_ext_image_import.dst_image) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_import.image_region) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_ext_image_import.image_region);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_ext_image_export: {
      out << "hsa_ext_image_export(";
      typedef decltype(api_data.args.hsa_ext_image_export.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_ext_image_export.agent) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_export.src_image) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_ext_image_export.src_image) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_export.dst_memory) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_ext_image_export.dst_memory) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_export.dst_row_pitch) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_ext_image_export.dst_row_pitch) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_export.dst_slice_pitch) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_ext_image_export.dst_slice_pitch) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_export.image_region) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_ext_image_export.image_region);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_ext_image_copy: {
      out << "hsa_ext_image_copy(";
      typedef decltype(api_data.args.hsa_ext_image_copy.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_ext_image_copy.agent) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_copy.src_image) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_ext_image_copy.src_image) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_copy.src_offset) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_ext_image_copy.src_offset) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_copy.dst_image) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_ext_image_copy.dst_image) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_copy.dst_offset) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_ext_image_copy.dst_offset) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_copy.range) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_ext_image_copy.range);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_ext_image_clear: {
      out << "hsa_ext_image_clear(";
      typedef decltype(api_data.args.hsa_ext_image_clear.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_ext_image_clear.agent) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_clear.image) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_ext_image_clear.image) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_clear.data) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_ext_image_clear.data) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_clear.image_region) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_ext_image_clear.image_region);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_ext_image_destroy: {
      out << "hsa_ext_image_destroy(";
      typedef decltype(api_data.args.hsa_ext_image_destroy.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_ext_image_destroy.agent) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_destroy.image) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_ext_image_destroy.image);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_ext_sampler_create: {
      out << "hsa_ext_sampler_create(";
      typedef decltype(api_data.args.hsa_ext_sampler_create.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_ext_sampler_create.agent) << ", ";
      typedef decltype(api_data.args.hsa_ext_sampler_create.sampler_descriptor) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_ext_sampler_create.sampler_descriptor) << ", ";
      typedef decltype(api_data.args.hsa_ext_sampler_create.sampler) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_ext_sampler_create.sampler);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_ext_sampler_destroy: {
      out << "hsa_ext_sampler_destroy(";
      typedef decltype(api_data.args.hsa_ext_sampler_destroy.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_ext_sampler_destroy.agent) << ", ";
      typedef decltype(api_data.args.hsa_ext_sampler_destroy.sampler) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_ext_sampler_destroy.sampler);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_ext_image_get_capability_with_layout: {
      out << "hsa_ext_image_get_capability_with_layout(";
      typedef decltype(api_data.args.hsa_ext_image_get_capability_with_layout.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_ext_image_get_capability_with_layout.agent) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_get_capability_with_layout.geometry) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_ext_image_get_capability_with_layout.geometry) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_get_capability_with_layout.image_format) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_ext_image_get_capability_with_layout.image_format) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_get_capability_with_layout.image_data_layout) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_ext_image_get_capability_with_layout.image_data_layout) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_get_capability_with_layout.capability_mask) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_ext_image_get_capability_with_layout.capability_mask);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_ext_image_data_get_info_with_layout: {
      out << "hsa_ext_image_data_get_info_with_layout(";
      typedef decltype(api_data.args.hsa_ext_image_data_get_info_with_layout.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_ext_image_data_get_info_with_layout.agent) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_data_get_info_with_layout.image_descriptor) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_ext_image_data_get_info_with_layout.image_descriptor) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_data_get_info_with_layout.access_permission) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_ext_image_data_get_info_with_layout.access_permission) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_data_get_info_with_layout.image_data_layout) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_ext_image_data_get_info_with_layout.image_data_layout) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_data_get_info_with_layout.image_data_row_pitch) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_ext_image_data_get_info_with_layout.image_data_row_pitch) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_data_get_info_with_layout.image_data_slice_pitch) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_ext_image_data_get_info_with_layout.image_data_slice_pitch) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_data_get_info_with_layout.image_data_info) arg_val_type_t6;
      roctracer::hsa_support::output_streamer<arg_val_type_t6>::put(out, api_data.args.hsa_ext_image_data_get_info_with_layout.image_data_info);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    case HSA_API_ID_hsa_ext_image_create_with_layout: {
      out << "hsa_ext_image_create_with_layout(";
      typedef decltype(api_data.args.hsa_ext_image_create_with_layout.agent) arg_val_type_t0;
      roctracer::hsa_support::output_streamer<arg_val_type_t0>::put(out, api_data.args.hsa_ext_image_create_with_layout.agent) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_create_with_layout.image_descriptor) arg_val_type_t1;
      roctracer::hsa_support::output_streamer<arg_val_type_t1>::put(out, api_data.args.hsa_ext_image_create_with_layout.image_descriptor) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_create_with_layout.image_data) arg_val_type_t2;
      roctracer::hsa_support::output_streamer<arg_val_type_t2>::put(out, api_data.args.hsa_ext_image_create_with_layout.image_data) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_create_with_layout.access_permission) arg_val_type_t3;
      roctracer::hsa_support::output_streamer<arg_val_type_t3>::put(out, api_data.args.hsa_ext_image_create_with_layout.access_permission) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_create_with_layout.image_data_layout) arg_val_type_t4;
      roctracer::hsa_support::output_streamer<arg_val_type_t4>::put(out, api_data.args.hsa_ext_image_create_with_layout.image_data_layout) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_create_with_layout.image_data_row_pitch) arg_val_type_t5;
      roctracer::hsa_support::output_streamer<arg_val_type_t5>::put(out, api_data.args.hsa_ext_image_create_with_layout.image_data_row_pitch) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_create_with_layout.image_data_slice_pitch) arg_val_type_t6;
      roctracer::hsa_support::output_streamer<arg_val_type_t6>::put(out, api_data.args.hsa_ext_image_create_with_layout.image_data_slice_pitch) << ", ";
      typedef decltype(api_data.args.hsa_ext_image_create_with_layout.image) arg_val_type_t7;
      roctracer::hsa_support::output_streamer<arg_val_type_t7>::put(out, api_data.args.hsa_ext_image_create_with_layout.image);
      out << ") = " << api_data.hsa_status_t_retval;
      break;
    }
    default:
      out << "ERROR: unknown API";
      abort();
  }
  return out;
}

#endif // INC_HSA_PROF_STR_H