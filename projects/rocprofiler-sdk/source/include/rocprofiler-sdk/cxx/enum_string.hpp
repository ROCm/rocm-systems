// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#pragma once

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>

#define ROCPROFILER_ENUM_LABEL(VALUE)                                                              \
    static_assert(VALUE >= 0, #VALUE " is < 0. Do not use ROCPROFILER_ENUM_LABEL for this value"); \
    template <>                                                                                    \
    struct rocprofiler_enum_label<decltype(VALUE), VALUE>                                          \
    {                                                                                              \
        using type                        = decltype(VALUE);                                       \
        static constexpr auto   supported = true;                                                  \
        static constexpr auto   name      = #VALUE;                                                \
        static constexpr size_t value     = VALUE;                                                 \
    };

#define ROCPROFILER_ENUM_INFO(ENUM_T, BEG_VALUE, END_VALUE, IS_BITSET)                             \
    template <>                                                                                    \
    struct rocprofiler_enum_info<ENUM_T>                                                           \
    {                                                                                              \
        using type                          = ENUM_T;                                              \
        static constexpr size_t begin       = BEG_VALUE;                                           \
        static constexpr size_t end         = END_VALUE;                                           \
        static constexpr auto   is_bitset   = IS_BITSET;                                           \
        static constexpr bool   supported   = true;                                                \
        static constexpr auto   unsupported = "unsupported_" #ENUM_T;                              \
        static constexpr auto   compute(size_t Idx)                                                \
        {                                                                                          \
            if constexpr(is_bitset)                                                                \
                return (1 << Idx);                                                                 \
            else                                                                                   \
                return Idx;                                                                        \
        }                                                                                          \
    };

namespace rocprofiler
{
namespace sdk
{
namespace details
{
template <typename EnumT, size_t Value>
struct rocprofiler_enum_label
{
    static constexpr bool supported = false;
};

template <typename EnumT>
struct rocprofiler_enum_info
{
    static constexpr bool supported = false;
};

template <size_t Idx, size_t N = 0>
constexpr size_t
compute_bitset_sequence_range()
{
    static_assert(N <= 64, "recursive limit for 64-bit bitset");
    if constexpr(Idx == 0)
    {
        return N;
    }
    else
    {
        constexpr auto NextIdx = (Idx >> 1);
        return compute_bitset_sequence_range<NextIdx, N + 1>();
    }
}

template <typename EnumT, size_t Idx, size_t... IdxTail>
constexpr auto
get_enum_label(EnumT val, std::index_sequence<Idx, IdxTail...>)
{
    using info  = rocprofiler_enum_info<EnumT>;
    using label = rocprofiler_enum_label<EnumT, info::is_bitset ? info::compute(Idx) : Idx>;

    if constexpr(label::supported)
    {
        if(val == label::value) return label::name;
    }

    if constexpr(sizeof...(IdxTail) > 0)
        return get_enum_label(val, std::index_sequence<IdxTail...>{});

    return info::unsupported;
    (void) val;  // suppress unused-but-set-parameter warning
}

// Table ID
ROCPROFILER_ENUM_INFO(rocprofiler_hsa_table_id_t, 0, ROCPROFILER_HSA_TABLE_ID_LAST, false)
ROCPROFILER_ENUM_INFO(rocprofiler_hip_table_id_t, 0, ROCPROFILER_HIP_TABLE_ID_LAST, false)
ROCPROFILER_ENUM_INFO(rocprofiler_marker_table_id_t, 0, ROCPROFILER_MARKER_TABLE_ID_LAST, false)
ROCPROFILER_ENUM_INFO(rocprofiler_rccl_table_id_t, 0, ROCPROFILER_RCCL_TABLE_ID_LAST, false)
ROCPROFILER_ENUM_INFO(rocprofiler_rocdecode_table_id_t,
                      0,
                      ROCPROFILER_ROCDECODE_TABLE_ID_LAST,
                      false)
ROCPROFILER_ENUM_INFO(rocprofiler_rocjpeg_table_id_t, 0, ROCPROFILER_ROCJPEG_TABLE_ID_LAST, false)

// table enums
// rocprofiler_hsa_core_api_id_t
ROCPROFILER_ENUM_INFO(rocprofiler_hsa_core_api_id_t, 0, ROCPROFILER_HSA_CORE_API_ID_LAST, false)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_init);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_shut_down);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_system_get_info);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_system_extension_supported);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_system_get_extension_table);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_iterate_agents);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_agent_get_info);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_create);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_soft_queue_create);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_destroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_inactivate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_load_read_index_scacquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_load_read_index_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_load_write_index_scacquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_load_write_index_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_store_write_index_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_store_write_index_screlease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_cas_write_index_scacq_screl);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_cas_write_index_scacquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_cas_write_index_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_cas_write_index_screlease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_add_write_index_scacq_screl);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_add_write_index_scacquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_add_write_index_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_add_write_index_screlease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_store_read_index_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_queue_store_read_index_screlease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_agent_iterate_regions);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_region_get_info);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_agent_get_exception_policies);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_agent_extension_supported);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_memory_register);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_memory_deregister);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_memory_allocate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_memory_free);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_memory_copy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_memory_assign_agent);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_create);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_destroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_load_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_load_scacquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_store_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_store_screlease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_wait_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_wait_scacquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_and_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_and_scacquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_and_screlease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_and_scacq_screl);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_or_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_or_scacquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_or_screlease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_or_scacq_screl);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_xor_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_xor_scacquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_xor_screlease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_xor_scacq_screl);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_exchange_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_exchange_scacquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_exchange_screlease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_exchange_scacq_screl);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_add_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_add_scacquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_add_screlease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_add_scacq_screl);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_subtract_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_subtract_scacquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_subtract_screlease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_subtract_scacq_screl);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_cas_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_cas_scacquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_cas_screlease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_cas_scacq_screl);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_isa_from_name);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_isa_get_info);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_isa_compatible);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_code_object_serialize);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_code_object_deserialize);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_code_object_destroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_code_object_get_info);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_code_object_get_symbol);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_code_symbol_get_info);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_code_object_iterate_symbols);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_create);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_destroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_load_code_object);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_freeze);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_get_info);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_global_variable_define);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_agent_global_variable_define);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_readonly_variable_define);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_validate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_get_symbol);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_symbol_get_info);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_iterate_symbols);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_status_string);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_extension_get_name);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_system_major_extension_supported);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_system_get_major_extension_table);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_agent_major_extension_supported);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_cache_get_info);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_agent_iterate_caches);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_silent_store_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_silent_store_screlease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_group_create);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_group_destroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_group_wait_any_scacquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_signal_group_wait_any_relaxed);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_agent_iterate_isas);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_isa_get_info_alt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_isa_get_exception_policies);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_isa_get_round_method);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_wavefront_get_info);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_isa_iterate_wavefronts);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_code_object_get_symbol_from_name);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_code_object_reader_create_from_file);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_code_object_reader_create_from_memory);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_code_object_reader_destroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_create_alt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_load_program_code_object);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_load_agent_code_object);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_validate_alt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_get_symbol_by_name);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_iterate_agent_symbols);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_CORE_API_ID_hsa_executable_iterate_program_symbols);
static_assert(ROCPROFILER_HSA_CORE_API_ID_LAST == 125);

// rocprofiler_hsa_amd_ext_api_id_t
ROCPROFILER_ENUM_INFO(rocprofiler_hsa_amd_ext_api_id_t,
                      0,
                      ROCPROFILER_HSA_AMD_EXT_API_ID_LAST,
                      false)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_coherency_get_type);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_coherency_set_type);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_profiling_set_profiler_enabled);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_profiling_async_copy_enable);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_profiling_get_dispatch_time);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_profiling_get_async_copy_time);
ROCPROFILER_ENUM_LABEL(
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_profiling_convert_tick_to_system_domain);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_signal_async_handler);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_async_function);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_signal_wait_any);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_queue_cu_set_mask);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_pool_get_info);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_agent_iterate_memory_pools);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_pool_allocate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_pool_free);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_async_copy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_async_copy_on_engine);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_copy_engine_status);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_agent_memory_pool_get_info);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_agents_allow_access);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_pool_can_migrate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_migrate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_lock);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_unlock);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_fill);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_interop_map_buffer);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_interop_unmap_buffer);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_image_create);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_pointer_info);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_pointer_info_set_userdata);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_ipc_memory_create);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_ipc_memory_attach);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_ipc_memory_detach);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_signal_create);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_ipc_signal_create);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_ipc_signal_attach);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_register_system_event_handler);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_queue_intercept_create);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_queue_intercept_register);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_queue_set_priority);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_async_copy_rect);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_runtime_queue_create_register);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_lock_to_pool);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_register_deallocation_callback);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_deregister_deallocation_callback);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_signal_value_pointer);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_svm_attributes_set);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_svm_attributes_get);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_svm_prefetch_async);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_spm_acquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_spm_release);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_spm_set_dest_buffer);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_queue_cu_get_mask);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_portable_export_dmabuf);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_portable_close_dmabuf);

#if HSA_AMD_EXT_API_TABLE_MAJOR_VERSION >= 0x02
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_address_reserve);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_address_free);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_handle_create);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_handle_release);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_map);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_unmap);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_set_access);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_get_access);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_export_shareable_handle);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_import_shareable_handle);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_retain_alloc_handle);
ROCPROFILER_ENUM_LABEL(
    ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_get_alloc_properties_from_handle);
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x01
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_agent_set_async_scratch_limit);
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x02
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_queue_get_info);
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x03
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_vmem_address_reserve_align);
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x04
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_enable_logging);
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x05
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_signal_wait_all);
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x06
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_memory_get_preferred_copy_engine);
#    endif
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x07
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_portable_export_dmabuf_v2);
#    endif
#endif

#if HSA_AMD_EXT_API_TABLE_MAJOR_VERSION == 0x01
static_assert(ROCPROFILER_HSA_AMD_EXT_API_ID_LAST == 55);
#elif HSA_AMD_EXT_API_TABLE_MAJOR_VERSION == 0x02
#    if HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x00
static_assert(ROCPROFILER_HSA_AMD_EXT_API_ID_LAST == 67);
#    elif HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x01
static_assert(ROCPROFILER_HSA_AMD_EXT_API_ID_LAST == 68);
#    elif HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x02
static_assert(ROCPROFILER_HSA_AMD_EXT_API_ID_LAST == 69);
#    elif HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x03
static_assert(ROCPROFILER_HSA_AMD_EXT_API_ID_LAST == 70);
#    elif HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x04
static_assert(ROCPROFILER_HSA_AMD_EXT_API_ID_LAST == 71);
#    elif HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x05
static_assert(ROCPROFILER_HSA_AMD_EXT_API_ID_LAST == 72);
#    elif HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x06
static_assert(ROCPROFILER_HSA_AMD_EXT_API_ID_LAST == 73);
#    elif HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x07
static_assert(ROCPROFILER_HSA_AMD_EXT_API_ID_LAST == 74);
#    else
#        if !defined(ROCPROFILER_UNSAFE_NO_VERSION_CHECK) &&                                       \
            (defined(ROCPROFILER_CI) && ROCPROFILER_CI > 0)
static_assert(false, "Support for new HSA_AMD_EXT_API_TABLE_STEP_VERSION enumerations is required");
#        endif
#    endif
#else
#    if !defined(ROCPROFILER_UNSAFE_NO_VERSION_CHECK) &&                                           \
        (defined(ROCPROFILER_CI) && ROCPROFILER_CI > 0)
static_assert(false, "Support for HSA_AMD_EXT_API_TABLE_MAJOR_VERSION is required");
#    endif
#endif

// rocprofiler_hsa_image_ext_api_id_t
ROCPROFILER_ENUM_INFO(rocprofiler_hsa_image_ext_api_id_t,
                      0,
                      ROCPROFILER_HSA_IMAGE_EXT_API_ID_LAST,
                      false)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_get_capability);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_data_get_info);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_create);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_import);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_export);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_copy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_clear);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_destroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_sampler_create);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_sampler_destroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_get_capability_with_layout);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_data_get_info_with_layout);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_create_with_layout);
static_assert(ROCPROFILER_HSA_IMAGE_EXT_API_ID_LAST == 13);

// rocprofiler_hsa_finalize_ext_api_id_t
ROCPROFILER_ENUM_INFO(rocprofiler_hsa_finalize_ext_api_id_t,
                      0,
                      ROCPROFILER_HSA_FINALIZE_EXT_API_ID_LAST,
                      false)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_FINALIZE_EXT_API_ID_hsa_ext_program_create);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_FINALIZE_EXT_API_ID_hsa_ext_program_destroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_FINALIZE_EXT_API_ID_hsa_ext_program_add_module);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_FINALIZE_EXT_API_ID_hsa_ext_program_iterate_modules);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_FINALIZE_EXT_API_ID_hsa_ext_program_get_info);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_FINALIZE_EXT_API_ID_hsa_ext_program_finalize);
static_assert(ROCPROFILER_HSA_FINALIZE_EXT_API_ID_LAST == 6);

// rocprofiler_hip_compiler_api_id_t
ROCPROFILER_ENUM_INFO(rocprofiler_hip_compiler_api_id_t,
                      0,
                      ROCPROFILER_HIP_COMPILER_API_ID_LAST,
                      false)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_COMPILER_API_ID___hipPopCallConfiguration);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_COMPILER_API_ID___hipPushCallConfiguration);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_COMPILER_API_ID___hipRegisterFatBinary);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_COMPILER_API_ID___hipRegisterFunction);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_COMPILER_API_ID___hipRegisterManagedVar);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_COMPILER_API_ID___hipRegisterSurface);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_COMPILER_API_ID___hipRegisterTexture);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_COMPILER_API_ID___hipRegisterVar);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_COMPILER_API_ID___hipUnregisterFatBinary);
static_assert(ROCPROFILER_HIP_COMPILER_API_ID_LAST == 9);

// rocprofiler_hip_runtime_api_id_t
ROCPROFILER_ENUM_INFO(rocprofiler_hip_runtime_api_id_t,
                      0,
                      ROCPROFILER_HIP_RUNTIME_API_ID_LAST,
                      false)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipApiName);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipArray3DCreate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipArray3DGetDescriptor);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipArrayCreate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipArrayDestroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipArrayGetDescriptor);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipArrayGetInfo);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipBindTexture);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipBindTexture2D);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipBindTextureToArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipBindTextureToMipmappedArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipChooseDevice);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipChooseDeviceR0000);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipConfigureCall);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCreateSurfaceObject);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCreateTextureObject);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxCreate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxDestroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxDisablePeerAccess);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxEnablePeerAccess);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxGetApiVersion);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxGetCacheConfig);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxGetCurrent);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxGetDevice);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxGetFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxGetSharedMemConfig);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxPopCurrent);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxPushCurrent);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxSetCacheConfig);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxSetCurrent);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxSetSharedMemConfig);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxSynchronize);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDestroyExternalMemory);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDestroyExternalSemaphore);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDestroySurfaceObject);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDestroyTextureObject);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceCanAccessPeer);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceComputeCapability);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceDisablePeerAccess);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceEnablePeerAccess);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGet);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGetAttribute);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGetByPCIBusId);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGetCacheConfig);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGetDefaultMemPool);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGetGraphMemAttribute);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGetLimit);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGetMemPool);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGetName);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGetP2PAttribute);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGetPCIBusId);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGetSharedMemConfig);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGetStreamPriorityRange);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGetUuid);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGraphMemTrim);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDevicePrimaryCtxGetState);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDevicePrimaryCtxRelease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDevicePrimaryCtxReset);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDevicePrimaryCtxRetain);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDevicePrimaryCtxSetFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceReset);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceSetCacheConfig);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceSetGraphMemAttribute);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceSetLimit);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceSetMemPool);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceSetSharedMemConfig);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceSynchronize);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceTotalMem);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDriverGetVersion);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvGetErrorName);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvGetErrorString);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvGraphAddMemcpyNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvMemcpy2DUnaligned);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvMemcpy3D);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvMemcpy3DAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvPointerGetAttributes);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipEventCreate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipEventCreateWithFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipEventDestroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipEventElapsedTime);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipEventQuery);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecord);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipEventSynchronize);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipExtGetLinkTypeAndHopCount);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchMultiKernelMultiDevice);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipExtMallocWithFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipExtStreamCreateWithCUMask);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipExtStreamGetCUMask);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipExternalMemoryGetMappedBuffer);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipFree);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipFreeArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipFreeAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipFreeHost);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipFreeMipmappedArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipFuncGetAttribute);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipFuncGetAttributes);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipFuncSetAttribute);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipFuncSetCacheConfig);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipFuncSetSharedMemConfig);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGLGetDevices);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetChannelDesc);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetDevice);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetDeviceCount);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetDeviceFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetDevicePropertiesR0600);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetDevicePropertiesR0000);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetErrorName);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetErrorString);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetLastError);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetMipmappedArrayLevel);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetSymbolAddress);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetSymbolSize);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetTextureAlignmentOffset);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetTextureObjectResourceDesc);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetTextureObjectResourceViewDesc);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetTextureObjectTextureDesc);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetTextureReference);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddChildGraphNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddDependencies);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddEmptyNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddEventRecordNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddEventWaitNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddHostNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddKernelNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddMemAllocNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddMemFreeNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddMemcpyNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddMemcpyNode1D);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddMemcpyNodeFromSymbol);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddMemcpyNodeToSymbol);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddMemsetNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphChildGraphNodeGetGraph);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphClone);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphCreate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphDebugDotPrint);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphDestroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphDestroyNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphEventRecordNodeGetEvent);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphEventRecordNodeSetEvent);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphEventWaitNodeGetEvent);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphEventWaitNodeSetEvent);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecChildGraphNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecDestroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecEventRecordNodeSetEvent);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecEventWaitNodeSetEvent);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecHostNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecKernelNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecMemcpyNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecMemcpyNodeSetParams1D);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecMemcpyNodeSetParamsFromSymbol);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecMemcpyNodeSetParamsToSymbol);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecMemsetNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecUpdate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphGetEdges);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphGetNodes);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphGetRootNodes);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphHostNodeGetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphHostNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphKernelNodeCopyAttributes);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphKernelNodeGetAttribute);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphKernelNodeGetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphKernelNodeSetAttribute);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphKernelNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphMemAllocNodeGetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphMemFreeNodeGetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphMemcpyNodeGetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphMemcpyNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphMemcpyNodeSetParams1D);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphMemcpyNodeSetParamsFromSymbol);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphMemcpyNodeSetParamsToSymbol);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphMemsetNodeGetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphMemsetNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphNodeFindInClone);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphNodeGetDependencies);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphNodeGetDependentNodes);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphNodeGetEnabled);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphNodeGetType);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphNodeSetEnabled);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphReleaseUserObject);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphRemoveDependencies);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphRetainUserObject);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphUpload);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphicsGLRegisterBuffer);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphicsGLRegisterImage);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphicsMapResources);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphicsResourceGetMappedPointer);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphicsSubResourceGetMappedArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphicsUnmapResources);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphicsUnregisterResource);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipHostAlloc);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipHostFree);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipHostGetDevicePointer);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipHostGetFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipHostMalloc);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipHostRegister);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipHostUnregister);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipImportExternalMemory);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipImportExternalSemaphore);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipInit);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipIpcCloseMemHandle);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipIpcGetEventHandle);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipIpcGetMemHandle);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipIpcOpenEventHandle);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipIpcOpenMemHandle);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipKernelNameRef);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipKernelNameRefByPtr);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchByPtr);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernelMultiDevice);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchHostFunc);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc3D);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc3DArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMallocArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMallocAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMallocFromPoolAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMallocHost);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMallocManaged);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMallocMipmappedArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMallocPitch);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemAddressFree);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemAddressReserve);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemAdvise);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemAllocHost);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemAllocPitch);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemCreate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemExportToShareableHandle);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemGetAccess);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemGetAddressRange);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemGetAllocationGranularity);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemGetAllocationPropertiesFromHandle);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemGetInfo);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemImportFromShareableHandle);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemMap);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemMapArrayAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemPoolCreate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemPoolDestroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemPoolExportPointer);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemPoolExportToShareableHandle);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemPoolGetAccess);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemPoolGetAttribute);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemPoolImportFromShareableHandle);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemPoolImportPointer);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemPoolSetAccess);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemPoolSetAttribute);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemPoolTrimTo);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemPrefetchAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemPtrGetInfo);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemRangeGetAttribute);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemRangeGetAttributes);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemRelease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemRetainAllocationHandle);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemSetAccess);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemUnmap);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2D);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArrayAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArrayAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3D);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3DAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAtoH);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoD);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoH);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoHAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbol);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbolAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoA);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoD);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoDAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyParam2D);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyParam2DAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeer);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeerAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbol);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbolAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyWithStream);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset2D);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset2DAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset3D);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset3DAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD16);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD16Async);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD32);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD32Async);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD8);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD8Async);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMipmappedArrayCreate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMipmappedArrayDestroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMipmappedArrayGetLevel);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleGetFunction);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleGetGlobal);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleGetTexRef);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernelMultiDevice);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLoad);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLoadData);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLoadDataEx);
ROCPROFILER_ENUM_LABEL(
    ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor);
ROCPROFILER_ENUM_LABEL(
    ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleOccupancyMaxPotentialBlockSize);
ROCPROFILER_ENUM_LABEL(
    ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleOccupancyMaxPotentialBlockSizeWithFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleUnload);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessor);
ROCPROFILER_ENUM_LABEL(
    ROCPROFILER_HIP_RUNTIME_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipOccupancyMaxPotentialBlockSize);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipPeekAtLastError);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipPointerGetAttribute);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipPointerGetAttributes);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipPointerSetAttribute);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipProfilerStart);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipProfilerStop);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipRuntimeGetVersion);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipSetDevice);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipSetDeviceFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipSetupArgument);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipSignalExternalSemaphoresAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamAddCallback);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamAttachMemAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamCreate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamCreateWithFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamCreateWithPriority);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamDestroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamGetCaptureInfo);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamGetCaptureInfo_v2);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamGetDevice);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamGetFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamGetPriority);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamIsCapturing);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamQuery);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamSynchronize);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamUpdateCaptureDependencies);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitEvent);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitValue32);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitValue64);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWriteValue32);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWriteValue64);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexObjectCreate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexObjectDestroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexObjectGetResourceDesc);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexObjectGetResourceViewDesc);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexObjectGetTextureDesc);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefGetAddress);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefGetAddressMode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefGetFilterMode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefGetFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefGetFormat);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefGetMaxAnisotropy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefGetMipMappedArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefGetMipmapFilterMode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefGetMipmapLevelBias);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefGetMipmapLevelClamp);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefSetAddress);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefSetAddress2D);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefSetAddressMode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefSetArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefSetBorderColor);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefSetFilterMode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefSetFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefSetFormat);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefSetMaxAnisotropy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefSetMipmapFilterMode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefSetMipmapLevelBias);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefSetMipmapLevelClamp);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefSetMipmappedArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipThreadExchangeStreamCaptureMode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipUnbindTexture);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipUserObjectCreate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipUserObjectRelease);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipUserObjectRetain);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipWaitExternalSemaphoresAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipCreateChannelDesc);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbol_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbol_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2D_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArray_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3D_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetAsync_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset2D_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset2DAsync_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset3DAsync_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset3D_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3DAsync_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DAsync_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbolAsync_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbolAsync_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromArray_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArray_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArrayAsync_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArrayAsync_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamQuery_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamSynchronize_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamGetPriority_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitEvent_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamGetFlags_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamAddCallback_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecord_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamIsCapturing_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamGetCaptureInfo_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamGetCaptureInfo_v2_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchHostFunc_spt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetStreamDeviceId);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvGraphAddMemsetNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddExternalSemaphoresWaitNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddExternalSemaphoresSignalNode);
ROCPROFILER_ENUM_LABEL(
    ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExternalSemaphoresSignalNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExternalSemaphoresWaitNodeSetParams);
ROCPROFILER_ENUM_LABEL(
    ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExternalSemaphoresSignalNodeGetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExternalSemaphoresWaitNodeGetParams);
ROCPROFILER_ENUM_LABEL(
    ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecExternalSemaphoresSignalNodeSetParams);
ROCPROFILER_ENUM_LABEL(
    ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecExternalSemaphoresWaitNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipExtGetLastError);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefGetBorderColor);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipTexRefGetArray);
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 1
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetProcAddress);
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 2
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCaptureToGraph);
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 3
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGetFuncBySymbol);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipSetValidDevices);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAtoD);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoA);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAtoA);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAtoHAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoAAsync);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DArrayToArray);
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 4
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvGraphAddMemFreeNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvGraphExecMemcpyNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvGraphExecMemsetNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecGetFlags);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipExternalMemoryGetMappedMipmappedArray);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvGraphMemcpyNodeGetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvGraphMemcpyNodeSetParams);
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 5
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipExtHostAlloc);
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 6
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceGetTexture1DLinearMaxWidth);
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 7
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBatchMemOp);
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 8
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddBatchMemOpNode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphBatchMemOpNodeGetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphBatchMemOpNodeSetParams);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecBatchMemOpNodeSetParams);
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 9
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipLinkAddData)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipLinkAddFile)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipLinkComplete)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipLinkCreate)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipLinkDestroy)
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 10
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecordWithFlags)
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 11
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernelExC)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvLaunchKernelEx)
#endif
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 12
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_API_ID_hipMemGetHandleForAddressRange)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION == 0
static_assert(ROCPROFILER_HIP_RUNTIME_API_ID_LAST == 442);
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 1
static_assert(ROCPROFILER_HIP_RUNTIME_API_ID_LAST == 443);
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 2
static_assert(ROCPROFILER_HIP_RUNTIME_API_ID_LAST == 444);
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 3
static_assert(ROCPROFILER_HIP_RUNTIME_API_ID_LAST == 452);
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 4
static_assert(ROCPROFILER_HIP_RUNTIME_API_ID_LAST == 461);
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 5
static_assert(ROCPROFILER_HIP_RUNTIME_API_ID_LAST == 462);
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 6
static_assert(ROCPROFILER_HIP_RUNTIME_API_ID_LAST == 463);
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 7
static_assert(ROCPROFILER_HIP_RUNTIME_API_ID_LAST == 464);
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 8
static_assert(ROCPROFILER_HIP_RUNTIME_API_ID_LAST == 468);
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 9
static_assert(ROCPROFILER_HIP_RUNTIME_API_ID_LAST == 473);
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 10
static_assert(ROCPROFILER_HIP_RUNTIME_API_ID_LAST == 474);
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 11
static_assert(ROCPROFILER_HIP_RUNTIME_API_ID_LAST == 476);
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 12
static_assert(ROCPROFILER_HIP_RUNTIME_API_ID_LAST == 477);
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 13
static_assert(ROCPROFILER_HIP_RUNTIME_API_ID_LAST == 477);
#else
#    if !defined(ROCPROFILER_UNSAFE_NO_VERSION_CHECK) &&                                           \
        (defined(ROCPROFILER_CI) && ROCPROFILER_CI > 0)
static_assert(false, "Support for new HIP_RUNTIME_API_TABLE_STEP_VERSION enumerations is required");
#    endif
#endif

// rocprofiler_marker_core_api_id_t
ROCPROFILER_ENUM_INFO(rocprofiler_marker_core_api_id_t,
                      0,
                      ROCPROFILER_MARKER_CORE_API_ID_LAST,
                      false)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_CORE_API_ID_roctxGetThreadId);
static_assert(ROCPROFILER_MARKER_CORE_API_ID_LAST == 6);

// rocprofiler_marker_control_api_id_t
ROCPROFILER_ENUM_INFO(rocprofiler_marker_control_api_id_t,
                      0,
                      ROCPROFILER_MARKER_CONTROL_API_ID_LAST,
                      false)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume);
static_assert(ROCPROFILER_MARKER_CONTROL_API_ID_LAST == 2);

// rocprofiler_marker_name_api_id_t
ROCPROFILER_ENUM_INFO(rocprofiler_marker_name_api_id_t,
                      0,
                      ROCPROFILER_MARKER_NAME_API_ID_LAST,
                      false)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_NAME_API_ID_roctxNameOsThread);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_NAME_API_ID_roctxNameHsaAgent);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_NAME_API_ID_roctxNameHipDevice);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_NAME_API_ID_roctxNameHipStream);
static_assert(ROCPROFILER_MARKER_NAME_API_ID_LAST == 4);

ROCPROFILER_ENUM_INFO(rocprofiler_marker_core_range_api_id_t,
                      0,
                      ROCPROFILER_MARKER_CORE_RANGE_API_ID_LAST,
                      false)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxMarkA);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxThreadRangeA);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxProcessRangeA);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxGetThreadId);
static_assert(ROCPROFILER_MARKER_CORE_RANGE_API_ID_LAST == 4);

// rocprofiler_ompt_operation_t
ROCPROFILER_ENUM_INFO(rocprofiler_ompt_operation_t, 0, ROCPROFILER_OMPT_ID_LAST, false);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_thread_begin);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_thread_end);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_parallel_begin);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_parallel_end);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_task_create);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_task_schedule);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_implicit_task);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_device_initialize);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_device_finalize);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_device_load);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_sync_region_wait);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_mutex_released);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_dependences);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_task_dependence);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_work);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_masked);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_sync_region);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_lock_init);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_lock_destroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_mutex_acquire);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_mutex_acquired);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_nest_lock);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_flush);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_cancel);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_reduction);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_dispatch);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_target_emi);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_target_data_op_emi);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_target_submit_emi);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_error);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_OMPT_ID_callback_functions);
static_assert(ROCPROFILER_OMPT_ID_LAST == 31);

// rocprofiler_rccl_api_id_t
ROCPROFILER_ENUM_INFO(rocprofiler_rccl_api_id_t, 0, ROCPROFILER_RCCL_API_ID_LAST, false)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclAllGather);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclAllReduce);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclAllToAll);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclAllToAllv);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclBroadcast);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclGather);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclReduce);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclReduceScatter);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclScatter);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclSend);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclRecv);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclRedOpCreatePreMulSum);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclRedOpDestroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclGroupStart);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclGroupEnd);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclGetVersion);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclGetUniqueId);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclCommInitRank);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclCommInitAll);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclCommInitRankConfig);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclCommFinalize);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclCommDestroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclCommAbort);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclCommSplit);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclGetErrorString);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclGetLastError);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclCommGetAsyncError);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclCommCount);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclCommCuDevice);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclCommUserRank);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclMemAlloc);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclMemFree);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_mscclLoadAlgo);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_mscclRunAlgo);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_mscclUnloadAlgo);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclCommRegister);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_API_ID_ncclCommDeregister);
static_assert(ROCPROFILER_RCCL_API_ID_LAST == 37);

// rocprofiler_rocdecode_api_id_t
ROCPROFILER_ENUM_INFO(rocprofiler_rocdecode_api_id_t, 0, ROCPROFILER_ROCDECODE_API_ID_LAST, false)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecCreateVideoParser);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecParseVideoData);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecDestroyVideoParser);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecCreateDecoder);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecDestroyDecoder);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecGetDecoderCaps);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecDecodeFrame);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecGetDecodeStatus);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecReconfigureDecoder);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecGetVideoFrame);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecGetErrorName);
#if ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION >= 1
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecCreateBitstreamReader);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecGetBitstreamCodecType);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecGetBitstreamBitDepth);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecGetBitstreamPicData);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_API_ID_rocDecDestroyBitstreamReader);
#endif

#if ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION == 0
static_assert(ROCPROFILER_ROCDECODE_API_ID_LAST == 11);
#elif ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION == 1
static_assert(ROCPROFILER_ROCDECODE_API_ID_LAST == 16);
#else
#    if !defined(ROCPROFILER_UNSAFE_NO_VERSION_CHECK) &&                                           \
        (defined(ROCPROFILER_CI) && ROCPROFILER_CI > 0)
static_assert(false,
              "Support for new ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION enumerations is required");
#    endif
#endif

// rocprofiler_rocjpeg_api_id_t
ROCPROFILER_ENUM_INFO(rocprofiler_rocjpeg_api_id_t, 0, ROCPROFILER_ROCJPEG_API_ID_LAST, false);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCJPEG_API_ID_rocJpegStreamCreate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCJPEG_API_ID_rocJpegStreamParse);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCJPEG_API_ID_rocJpegStreamDestroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCJPEG_API_ID_rocJpegCreate);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCJPEG_API_ID_rocJpegDestroy);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCJPEG_API_ID_rocJpegGetImageInfo);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCJPEG_API_ID_rocJpegDecode);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCJPEG_API_ID_rocJpegDecodeBatched);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCJPEG_API_ID_rocJpegGetErrorName);
static_assert(ROCPROFILER_ROCJPEG_API_ID_LAST == 9);

// fwd.h
ROCPROFILER_ENUM_INFO(rocprofiler_status_t,
                      ROCPROFILER_STATUS_SUCCESS,
                      ROCPROFILER_STATUS_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_buffer_category_t,
                      ROCPROFILER_BUFFER_CATEGORY_NONE,
                      ROCPROFILER_BUFFER_CATEGORY_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_agent_type_t,
                      ROCPROFILER_AGENT_TYPE_NONE,
                      ROCPROFILER_AGENT_TYPE_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_callback_phase_t,
                      ROCPROFILER_CALLBACK_PHASE_NONE,
                      ROCPROFILER_CALLBACK_PHASE_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_callback_tracing_kind_t,
                      ROCPROFILER_CALLBACK_TRACING_NONE,
                      ROCPROFILER_CALLBACK_TRACING_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_buffer_tracing_kind_t,
                      ROCPROFILER_BUFFER_TRACING_NONE,
                      ROCPROFILER_BUFFER_TRACING_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_code_object_operation_t,
                      ROCPROFILER_CODE_OBJECT_NONE,
                      ROCPROFILER_CODE_OBJECT_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_memory_copy_operation_t,
                      ROCPROFILER_MEMORY_COPY_NONE,
                      ROCPROFILER_MEMORY_COPY_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_memory_allocation_operation_t,
                      ROCPROFILER_MEMORY_ALLOCATION_NONE,
                      ROCPROFILER_MEMORY_ALLOCATION_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_kernel_dispatch_operation_t,
                      ROCPROFILER_KERNEL_DISPATCH_NONE,
                      ROCPROFILER_KERNEL_DISPATCH_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_pc_sampling_method_t,
                      ROCPROFILER_PC_SAMPLING_METHOD_NONE,
                      ROCPROFILER_PC_SAMPLING_METHOD_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_pc_sampling_unit_t,
                      ROCPROFILER_PC_SAMPLING_UNIT_NONE,
                      ROCPROFILER_PC_SAMPLING_UNIT_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_buffer_policy_t,
                      ROCPROFILER_BUFFER_POLICY_NONE,
                      ROCPROFILER_BUFFER_POLICY_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_scratch_memory_operation_t,
                      ROCPROFILER_SCRATCH_MEMORY_NONE,
                      ROCPROFILER_SCRATCH_MEMORY_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_scratch_alloc_flag_t,
                      ROCPROFILER_SCRATCH_ALLOC_FLAG_NONE,
                      HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_ALT + 1,
                      true);
ROCPROFILER_ENUM_INFO(rocprofiler_runtime_initialization_operation_t,
                      ROCPROFILER_RUNTIME_INITIALIZATION_NONE,
                      ROCPROFILER_RUNTIME_INITIALIZATION_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_counter_info_version_id_t,
                      ROCPROFILER_COUNTER_INFO_VERSION_NONE,
                      ROCPROFILER_COUNTER_INFO_VERSION_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_counter_record_kind_t,
                      ROCPROFILER_COUNTER_RECORD_NONE,
                      ROCPROFILER_COUNTER_RECORD_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_counter_flag_t,
                      ROCPROFILER_COUNTER_FLAG_NONE,
                      ROCPROFILER_COUNTER_FLAG_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_code_object_storage_type_t,
                      ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_NONE,
                      ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_LAST,
                      false);

ROCPROFILER_ENUM_INFO(rocprofiler_runtime_library_t,
                      ROCPROFILER_LIBRARY,
                      details::compute_bitset_sequence_range<ROCPROFILER_LIBRARY_LAST>(),
                      true);
ROCPROFILER_ENUM_INFO(rocprofiler_intercept_table_t,
                      ROCPROFILER_HSA_TABLE,
                      details::compute_bitset_sequence_range<ROCPROFILER_TABLE_LAST>(),
                      true);

// callback_tracing.h
ROCPROFILER_ENUM_INFO(rocprofiler_pc_sampling_record_kind_t,
                      ROCPROFILER_PC_SAMPLING_RECORD_NONE,
                      ROCPROFILER_PC_SAMPLING_RECORD_LAST,
                      false);

// kfd/kfd_id.h
ROCPROFILER_ENUM_INFO(rocprofiler_kfd_event_page_migrate_operation_t,
                      ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_NONE,
                      ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_kfd_event_page_fault_operation_t,
                      ROCPROFILER_KFD_EVENT_PAGE_FAULT_NONE,
                      ROCPROFILER_KFD_EVENT_PAGE_FAULT_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_kfd_event_queue_operation_t,
                      ROCPROFILER_KFD_EVENT_QUEUE_NONE,
                      ROCPROFILER_KFD_EVENT_QUEUE_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_kfd_event_unmap_from_gpu_operation_t,
                      ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_NONE,
                      ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_kfd_event_dropped_events_operation_t,
                      ROCPROFILER_KFD_EVENT_DROPPED_EVENTS_NONE,
                      ROCPROFILER_KFD_EVENT_DROPPED_EVENTS_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_kfd_page_migrate_operation_t,
                      ROCPROFILER_KFD_PAGE_MIGRATE_NONE,
                      ROCPROFILER_KFD_PAGE_MIGRATE_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_kfd_page_fault_operation_t,
                      ROCPROFILER_KFD_PAGE_FAULT_NONE,
                      ROCPROFILER_KFD_PAGE_FAULT_LAST,
                      false);
ROCPROFILER_ENUM_INFO(rocprofiler_kfd_queue_operation_t,
                      ROCPROFILER_KFD_QUEUE_NONE,
                      ROCPROFILER_KFD_QUEUE_LAST,
                      false);

ROCPROFILER_ENUM_INFO(rocprofiler_external_correlation_id_request_kind_t,
                      ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_NONE,
                      ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_LAST,
                      false);

ROCPROFILER_ENUM_INFO(rocprofiler_thread_trace_parameter_type_t,
                      ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU,
                      ROCPROFILER_THREAD_TRACE_PARAMETER_LAST,
                      false);

ROCPROFILER_ENUM_INFO(rocprofiler_agent_version_t,
                      ROCPROFILER_AGENT_INFO_VERSION_NONE,
                      ROCPROFILER_AGENT_INFO_VERSION_LAST,
                      false);

// begin fwd.h
// rocprofiler_hsa_table_id_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_TABLE_ID_Core);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_TABLE_ID_AmdExt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_TABLE_ID_ImageExt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_TABLE_ID_FinalizeExt);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_TABLE_ID_AmdTool);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_TABLE_ID_PcSamplingExt);
static_assert(ROCPROFILER_HSA_TABLE_ID_LAST == 6);

// rocprofiler_hip_table_id_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_TABLE_ID_Compiler);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_TABLE_ID_Runtime);
static_assert(ROCPROFILER_HIP_TABLE_ID_LAST == 2);

// rocprofiler_marker_table_id_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_TABLE_ID_RoctxCore);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_TABLE_ID_RoctxControl);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_TABLE_ID_RoctxName);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_TABLE_ID_RoctxCoreRange);
static_assert(ROCPROFILER_MARKER_TABLE_ID_LAST == 4);

// rocprofiler_rccl_table_id_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_TABLE_ID);
static_assert(ROCPROFILER_RCCL_TABLE_ID_LAST == 1);

// rocprofiler_rocdecode_table_id_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_TABLE_ID_CORE);
static_assert(ROCPROFILER_ROCDECODE_TABLE_ID_LAST == 1);

// rocprofiler_rocjpeg_table_id_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCJPEG_TABLE_ID_CORE);
static_assert(ROCPROFILER_ROCJPEG_TABLE_ID_LAST == 1);

// rocprofiler_status_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_SUCCESS);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_THREAD_NOT_FOUND);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_CONTEXT_ID_NOT_ZERO);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_BUFFER_BUSY);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_METRIC_NOT_VALID_FOR_AGENT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_FINALIZED);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_DIM_NOT_FOUND);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_PROFILE_COUNTER_NOT_FOUND);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_AST_GENERATION_FAILED);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_AST_NOT_FOUND);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_AQL_NO_EVENT_COORD);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_PROFILE_NOT_FOUND);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_AGENT_DISPATCH_CONFLICT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_INTERNAL_NO_AGENT_CONTEXT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_SAMPLE_RATE_EXCEEDED);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_NO_PROFILE_QUEUE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_NO_HARDWARE_COUNTERS);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_AGENT_MISMATCH);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_STATUS_ERROR_PERMISSION_DENIED);
static_assert(ROCPROFILER_STATUS_LAST == 41);

// rocprofiler_buffer_category_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_CATEGORY_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_CATEGORY_TRACING);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_CATEGORY_COUNTERS);
static_assert(ROCPROFILER_BUFFER_CATEGORY_LAST == 4);

// rocprofiler_agent_type_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_AGENT_TYPE_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_AGENT_TYPE_CPU);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_AGENT_TYPE_GPU);
static_assert(ROCPROFILER_AGENT_TYPE_LAST == 3);

// rocprofiler_callback_phase_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_PHASE_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_PHASE_ENTER);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_PHASE_EXIT);
static_assert(ROCPROFILER_CALLBACK_PHASE_LAST == 3);

// rocprofiler_callback_tracing_kind_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_RANGE_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_SCRATCH_MEMORY);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_RCCL_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_OMPT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CALLBACK_TRACING_HIP_STREAM);
static_assert(ROCPROFILER_CALLBACK_TRACING_LAST == 22);

// rocprofiler_buffer_tracing_kind_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_HSA_CORE_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_RCCL_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_OMPT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_ROCDECODE_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_ROCJPEG_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_HIP_STREAM);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_ROCDECODE_API_EXT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_TRACING_KFD_QUEUE);
static_assert(ROCPROFILER_BUFFER_TRACING_LAST == 33);

// rocprofiler_code_object_operation_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CODE_OBJECT_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CODE_OBJECT_LOAD);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_CODE_OBJECT_HOST_KERNEL_SYMBOL_REGISTER);
static_assert(ROCPROFILER_CODE_OBJECT_LAST == 4);

// rocprofiler_memory_copy_operation_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MEMORY_COPY_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MEMORY_COPY_HOST_TO_HOST);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE);
static_assert(ROCPROFILER_MEMORY_COPY_LAST == 5);

// rocprofiler_memory_allocation_operation_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MEMORY_ALLOCATION_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MEMORY_ALLOCATION_VMEM_ALLOCATE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MEMORY_ALLOCATION_FREE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MEMORY_ALLOCATION_VMEM_FREE);
static_assert(ROCPROFILER_MEMORY_ALLOCATION_LAST == 5);

// rocprofiler_kernel_dispatch_operation_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KERNEL_DISPATCH_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KERNEL_DISPATCH_ENQUEUE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KERNEL_DISPATCH_COMPLETE);
static_assert(ROCPROFILER_KERNEL_DISPATCH_LAST == 3);

// rocprofiler_pc_sampling_method_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_PC_SAMPLING_METHOD_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP);
static_assert(ROCPROFILER_PC_SAMPLING_METHOD_LAST == 3);

// rocprofiler_pc_sampling_unit_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_PC_SAMPLING_UNIT_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_PC_SAMPLING_UNIT_CYCLES);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_PC_SAMPLING_UNIT_TIME);
static_assert(ROCPROFILER_PC_SAMPLING_UNIT_LAST == 4);

// rocprofiler_buffer_policy_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_POLICY_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_POLICY_DISCARD);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_BUFFER_POLICY_LOSSLESS);
static_assert(ROCPROFILER_BUFFER_POLICY_LAST == 3);

// rocprofiler_scratch_memory_operation_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_SCRATCH_MEMORY_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_SCRATCH_MEMORY_ALLOC);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_SCRATCH_MEMORY_FREE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_SCRATCH_MEMORY_ASYNC_RECLAIM);
static_assert(ROCPROFILER_SCRATCH_MEMORY_LAST == 4);

ROCPROFILER_ENUM_LABEL(ROCPROFILER_SCRATCH_ALLOC_FLAG_USE_ONCE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_SCRATCH_ALLOC_FLAG_ALT);

// rocprofiler_runtime_library_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_LIBRARY)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_LIBRARY)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_LIBRARY)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_LIBRARY)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_LIBRARY)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_LIBRARY)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCJPEG_LIBRARY)
static_assert(ROCPROFILER_LIBRARY_LAST == ROCPROFILER_ROCJPEG_LIBRARY);

// rocprofiler_intercept_table_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HSA_TABLE)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_RUNTIME_TABLE)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_HIP_COMPILER_TABLE)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_CORE_TABLE)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_CONTROL_TABLE)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_MARKER_NAME_TABLE)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RCCL_TABLE)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCDECODE_TABLE)
ROCPROFILER_ENUM_LABEL(ROCPROFILER_ROCJPEG_TABLE)
static_assert(ROCPROFILER_TABLE_LAST == ROCPROFILER_ROCJPEG_TABLE);

// rocprofiler_runtime_initialization_operation_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RUNTIME_INITIALIZATION_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RUNTIME_INITIALIZATION_HSA);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RUNTIME_INITIALIZATION_HIP);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RUNTIME_INITIALIZATION_MARKER);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RUNTIME_INITIALIZATION_RCCL);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RUNTIME_INITIALIZATION_ROCDECODE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_RUNTIME_INITIALIZATION_ROCJPEG);
static_assert(ROCPROFILER_RUNTIME_INITIALIZATION_LAST == 7);

// rocprofiler_counter_info_version_id_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_COUNTER_INFO_VERSION_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_COUNTER_INFO_VERSION_0);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_COUNTER_INFO_VERSION_1);
static_assert(ROCPROFILER_COUNTER_INFO_VERSION_LAST == 3);

// rocprofiler_counter_record_kind_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_COUNTER_RECORD_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_COUNTER_RECORD_VALUE);
static_assert(ROCPROFILER_COUNTER_RECORD_LAST == 3);

// rocprofiler_counter_flag_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_COUNTER_FLAG_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_COUNTER_FLAG_ASYNC);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_COUNTER_FLAG_APPEND_DEFINITION);
static_assert(ROCPROFILER_COUNTER_FLAG_LAST == 3);

// rocprofiler_pc_sampling_record_kind_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_PC_SAMPLING_RECORD_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE);
static_assert(ROCPROFILER_PC_SAMPLING_RECORD_LAST == 4);

// kfd events
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PREFETCH);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PAGEFAULT_GPU);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PAGEFAULT_CPU);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_TTM_EVICTION);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_END);
static_assert(ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_LAST == 5);

// rocprofiler_kfd_event_page_fault_operation_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_PAGE_FAULT_START);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_PAGE_FAULT_START_READ_FAULT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_PAGE_FAULT_START_WRITE_FAULT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_MIGRATED);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_UPDATED);
static_assert(ROCPROFILER_KFD_EVENT_PAGE_FAULT_LAST == 5);

// rocprofiler_kfd_event_queue_operation_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SVM);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_QUEUE_EVICT_USERPTR);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_QUEUE_EVICT_TTM);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SUSPEND);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_CHECKPOINT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_RESTORE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_QUEUE_RESTORE);
static_assert(ROCPROFILER_KFD_EVENT_QUEUE_LAST == 8);

// rocprofiler_kfd_event_unmap_from_gpu_operation_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY_MIGRATE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_UNMAP_FROM_CPU);
static_assert(ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_LAST == 3);

// rocprofiler_kfd_event_dropped_events_operation_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_EVENT_DROPPED_EVENTS);
static_assert(ROCPROFILER_KFD_EVENT_DROPPED_EVENTS_LAST == 1);

// rocprofiler_kfd_page_migrate_operation_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_PAGE_MIGRATE_PREFETCH);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_PAGE_MIGRATE_PAGEFAULT_GPU);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_PAGE_MIGRATE_PAGEFAULT_CPU);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_PAGE_MIGRATE_TTM_EVICTION);
static_assert(ROCPROFILER_KFD_PAGE_MIGRATE_LAST == 4);

// rocprofiler_kfd_page_fault_operation_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_PAGE_FAULT_READ_FAULT_MIGRATED);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_PAGE_FAULT_READ_FAULT_UPDATED);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_PAGE_FAULT_WRITE_FAULT_MIGRATED);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_PAGE_FAULT_WRITE_FAULT_UPDATED);
static_assert(ROCPROFILER_KFD_PAGE_FAULT_LAST == 4);

// rocprofiler_kfd_queue_operation_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_QUEUE_EVICT_SVM);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_QUEUE_EVICT_USERPTR);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_QUEUE_EVICT_TTM);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_QUEUE_EVICT_SUSPEND);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_QUEUE_EVICT_CRIU_CHECKPOINT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_KFD_QUEUE_EVICT_CRIU_RESTORE);
static_assert(ROCPROFILER_KFD_QUEUE_LAST == 6);

// rocprofiler_external_correlation_id_request_kind_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HSA_CORE_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HSA_AMD_EXT_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HSA_IMAGE_EXT_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HSA_FINALIZE_EXT_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_RUNTIME_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_COMPILER_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MARKER_CORE_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MARKER_CONTROL_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MARKER_NAME_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_SCRATCH_MEMORY);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_RCCL_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_OMPT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_ROCDECODE_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_ROCJPEG_API);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MARKER_CORE_RANGE_API);
static_assert(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_LAST == 19);

// rocprofiler_thread_trace_parameter_type_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTERS_CTRL);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTER);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_THREAD_TRACE_PARAMETER_SERIALIZE_ALL);
static_assert(ROCPROFILER_THREAD_TRACE_PARAMETER_LAST == 7);

ROCPROFILER_ENUM_LABEL(ROCPROFILER_THREAD_TRACE_CONTROL_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP);

// rocprofiler_agent_version_t
ROCPROFILER_ENUM_LABEL(ROCPROFILER_AGENT_INFO_VERSION_NONE);
ROCPROFILER_ENUM_LABEL(ROCPROFILER_AGENT_INFO_VERSION_0);
static_assert(ROCPROFILER_AGENT_INFO_VERSION_LAST == 2);

}  // namespace details

template <typename EnumT>
// constexpr auto
constexpr std::string_view
get_enum_label(EnumT val)
{
    if constexpr(details::rocprofiler_enum_info<EnumT>::supported)
    {
        return details::get_enum_label(val,
                                       std::make_index_sequence<static_cast<size_t>(
                                           details::rocprofiler_enum_info<EnumT>::end)>{});
    }
    else
    {
        static_assert(sizeof(EnumT) < 0, "Unsupported enum type");
    }

    return "unsupported";
}
}  // namespace sdk
}  // namespace rocprofiler
