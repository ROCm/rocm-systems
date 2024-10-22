// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "domain_type.hpp"
#include "lib/common/container/ring_buffer.hpp"
#include "lib/common/container/small_vector.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/demangle.hpp"
#include "lib/common/filesystem.hpp"
#include "output_file.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <cstdint>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/cxx/serialization.hpp>

#include <amd_comgr/amd_comgr.h>
#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <glog/logging.h>

#include <cxxabi.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <map>
#include <ostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = result;                 \
        if(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) != ROCPROFILER_STATUS_SUCCESS)              \
        {                                                                                          \
            std::string status_msg =                                                               \
                rocprofiler_get_status_string(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));        \
            ROCP_FATAL << " :: [" << __FILE__ << ":" << __LINE__ << "]\n\t" << #result << "\n\n"   \
                       << msg << " failed with error code "                                        \
                       << ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) << ": " << status_msg;       \
        }                                                                                          \
    }

constexpr size_t BUFFER_SIZE_BYTES = 4096;
constexpr size_t WATERMARK         = (BUFFER_SIZE_BYTES / 2);

using rocprofiler_tool_buffer_kind_names_t =
    std::unordered_map<rocprofiler_buffer_tracing_kind_t, std::string>;
using rocprofiler_tool_buffer_kind_operation_names_t =
    std::unordered_map<rocprofiler_buffer_tracing_kind_t,
                       std::unordered_map<uint32_t, std::string>>;

using marker_message_map_t = std::unordered_map<uint64_t, std::string>;
using rocprofiler_kernel_symbol_data_t =
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

namespace common = ::rocprofiler::common;
namespace tool   = ::rocprofiler::tool;

struct kernel_symbol_data : rocprofiler_kernel_symbol_data_t
{
    using base_type = rocprofiler_kernel_symbol_data_t;

    kernel_symbol_data(const base_type& _base)
    : base_type{_base}
    , formatted_kernel_name{tool::format_name(CHECK_NOTNULL(_base.kernel_name))}
    , demangled_kernel_name{common::cxx_demangle(CHECK_NOTNULL(_base.kernel_name))}
    , truncated_kernel_name{common::truncate_name(demangled_kernel_name)}
    {}

    kernel_symbol_data();
    ~kernel_symbol_data()                             = default;
    kernel_symbol_data(const kernel_symbol_data&)     = default;
    kernel_symbol_data(kernel_symbol_data&&) noexcept = default;
    kernel_symbol_data& operator=(const kernel_symbol_data&) = default;
    kernel_symbol_data& operator=(kernel_symbol_data&&) noexcept = default;

    std::string formatted_kernel_name = {};
    std::string demangled_kernel_name = {};
    std::string truncated_kernel_name = {};
};

inline kernel_symbol_data::kernel_symbol_data()
: base_type{0, 0, 0, "", 0, 0, 0, 0, 0, 0, 0, 0}
{}

using kernel_symbol_data_map_t = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data>;

struct rocprofiler_tool_counter_info_t : rocprofiler_counter_info_v0_t
{
    using parent_type          = rocprofiler_counter_info_v0_t;
    using dimension_id_vec_t   = std::vector<rocprofiler_counter_dimension_id_t>;
    using dimension_info_vec_t = std::vector<rocprofiler_record_dimension_info_t>;

    rocprofiler_tool_counter_info_t(rocprofiler_agent_id_t _agent_id,
                                    parent_type            _info,
                                    dimension_id_vec_t&&   _dim_ids,
                                    dimension_info_vec_t&& _dim_info)
    : parent_type{_info}
    , agent_id{_agent_id}
    , dimension_ids{std::move(_dim_ids)}
    , dimension_info{std::move(_dim_info)}
    {}

    ~rocprofiler_tool_counter_info_t()                                          = default;
    rocprofiler_tool_counter_info_t(const rocprofiler_tool_counter_info_t&)     = default;
    rocprofiler_tool_counter_info_t(rocprofiler_tool_counter_info_t&&) noexcept = default;
    rocprofiler_tool_counter_info_t& operator=(const rocprofiler_tool_counter_info_t&) = default;
    rocprofiler_tool_counter_info_t& operator=(rocprofiler_tool_counter_info_t&&) noexcept =
        default;

    rocprofiler_agent_id_t                           agent_id       = {};
    std::vector<rocprofiler_counter_dimension_id_t>  dimension_ids  = {};
    std::vector<rocprofiler_record_dimension_info_t> dimension_info = {};
};

rocprofiler::sdk::buffer_name_info_t<std::string_view>
get_buffer_id_names();

::rocprofiler::sdk::callback_name_info_t<std::string_view>
get_callback_id_names();

std::map<uint64_t, std::string>
get_callback_roctx_msg();

std::vector<kernel_symbol_data>
get_kernel_symbol_data();

std::vector<rocprofiler_callback_tracing_code_object_load_data_t>
get_code_object_data();

std::vector<rocprofiler_tool_counter_info_t>
get_tool_counter_info();

std::vector<rocprofiler_record_dimension_info_t>
get_tool_counter_dimension_info();

enum tracing_marker_kind
{
    MARKER_API_CORE = 0,
    MARKER_API_CONTROL,
    MARKER_API_NAME,
    MARKER_API_LAST,
};

template <size_t CommonV>
struct marker_tracing_kind_conversion;

#define MAP_TRACING_KIND_CONVERSION(COMMON, CALLBACK, BUFFERED)                                    \
    template <>                                                                                    \
    struct marker_tracing_kind_conversion<COMMON>                                                  \
    {                                                                                              \
        static constexpr auto callback_value = CALLBACK;                                           \
        static constexpr auto buffered_value = BUFFERED;                                           \
                                                                                                   \
        bool operator==(rocprofiler_callback_tracing_kind_t val) const                             \
        {                                                                                          \
            return (callback_value == val);                                                        \
        }                                                                                          \
                                                                                                   \
        bool operator==(rocprofiler_buffer_tracing_kind_t val) const                               \
        {                                                                                          \
            return (buffered_value == val);                                                        \
        }                                                                                          \
                                                                                                   \
        auto convert(rocprofiler_callback_tracing_kind_t) const { return buffered_value; }         \
        auto convert(rocprofiler_buffer_tracing_kind_t) const { return callback_value; }           \
    };

MAP_TRACING_KIND_CONVERSION(MARKER_API_CORE,
                            ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                            ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API)
MAP_TRACING_KIND_CONVERSION(MARKER_API_CONTROL,
                            ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                            ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API)
MAP_TRACING_KIND_CONVERSION(MARKER_API_NAME,
                            ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API,
                            ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API)
MAP_TRACING_KIND_CONVERSION(MARKER_API_LAST,
                            ROCPROFILER_CALLBACK_TRACING_LAST,
                            ROCPROFILER_BUFFER_TRACING_LAST)

template <typename TracingKindT, size_t Idx, size_t... Tail>
auto
convert_marker_tracing_kind(TracingKindT val, std::index_sequence<Idx, Tail...>)
{
    if(marker_tracing_kind_conversion<Idx>{} == val)
    {
        return marker_tracing_kind_conversion<Idx>{}.convert(val);
    }

    if constexpr(sizeof...(Tail) > 0)
        return convert_marker_tracing_kind(val, std::index_sequence<Tail...>{});

    return marker_tracing_kind_conversion<MARKER_API_LAST>{}.convert(val);
}

template <typename TracingKindT>
auto
convert_marker_tracing_kind(TracingKindT val)
{
    return convert_marker_tracing_kind(val, std::make_index_sequence<MARKER_API_LAST>{});
}

struct rocprofiler_tool_dimension_pos_t
{
    uint64_t dimension_id;
    size_t   instance;

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("dimension_id", dimension_id));
        ar(cereal::make_nvp("instance", instance));
    }
};

struct rocprofiler_tool_record_counter_t
{
    rocprofiler_counter_id_t     counter_id     = {};
    rocprofiler_record_counter_t record_counter = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("counter_id", counter_id));
        ar(cereal::make_nvp("value", record_counter.counter_value));
    }
};

struct rocprofiler_tool_counter_collection_record_t
{
    rocprofiler_dispatch_counting_service_data_t       dispatch_data    = {};
    std::array<rocprofiler_tool_record_counter_t, 512> records          = {};
    uint64_t                                           thread_id        = 0;
    uint64_t                                           arch_vgpr_count  = 0;
    uint64_t                                           sgpr_count       = 0;
    uint64_t                                           lds_block_size_v = 0;
    uint64_t                                           counter_count    = 0;

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("dispatch_data", dispatch_data));
        // should be removed when moving to buffered tracing
        std::vector<rocprofiler_tool_record_counter_t> tmp{records.begin(),
                                                           records.begin() + counter_count};
        ar(cereal::make_nvp("records", tmp));
        ar(cereal::make_nvp("thread_id", thread_id));
        ar(cereal::make_nvp("arch_vgpr_count", arch_vgpr_count));
        ar(cereal::make_nvp("sgpr_count", sgpr_count));
        ar(cereal::make_nvp("lds_block_size_v", lds_block_size_v));
    }
};

struct timestamps_t
{
    rocprofiler_timestamp_t app_start_time;
    rocprofiler_timestamp_t app_end_time;
};

namespace rocprofiler
{
namespace tool
{
template <typename Tp, domain_type DomainT>
struct buffered_output;
}
}  // namespace rocprofiler

using hip_buffered_output_t =
    ::rocprofiler::tool::buffered_output<rocprofiler_buffer_tracing_hip_api_record_t,
                                         domain_type::HIP>;
using hsa_buffered_output_t =
    ::rocprofiler::tool::buffered_output<rocprofiler_buffer_tracing_hsa_api_record_t,
                                         domain_type::HSA>;
using kernel_dispatch_buffered_output_t =
    ::rocprofiler::tool::buffered_output<rocprofiler_buffer_tracing_kernel_dispatch_record_t,
                                         domain_type::KERNEL_DISPATCH>;
using memory_copy_buffered_output_t =
    ::rocprofiler::tool::buffered_output<rocprofiler_buffer_tracing_memory_copy_record_t,
                                         domain_type::MEMORY_COPY>;
using marker_buffered_output_t =
    ::rocprofiler::tool::buffered_output<rocprofiler_buffer_tracing_marker_api_record_t,
                                         domain_type::MARKER>;
using rccl_buffered_output_t =
    ::rocprofiler::tool::buffered_output<rocprofiler_buffer_tracing_rccl_api_record_t,
                                         domain_type::RCCL>;
using counter_collection_buffered_output_t =
    ::rocprofiler::tool::buffered_output<rocprofiler_tool_counter_collection_record_t,
                                         domain_type::COUNTER_COLLECTION>;
using scratch_memory_buffered_output_t =
    ::rocprofiler::tool::buffered_output<rocprofiler_buffer_tracing_scratch_memory_record_t,
                                         domain_type::SCRATCH_MEMORY>;

using tool_get_agent_node_id_fn_t      = uint64_t (*)(rocprofiler_agent_id_t);
using tool_get_app_timestamps_fn_t     = timestamps_t* (*) ();
using tool_get_kernel_name_fn_t        = std::string_view (*)(uint64_t, uint64_t);
using tool_get_domain_name_fn_t        = std::string_view (*)(rocprofiler_buffer_tracing_kind_t);
using tool_get_operation_name_fn_t     = std::string_view (*)(rocprofiler_buffer_tracing_kind_t,
                                                          rocprofiler_tracing_operation_t);
using tool_get_callback_kind_name_fn_t = std::string_view (*)(rocprofiler_callback_tracing_kind_t);
using tool_get_callback_op_name_fn_t   = std::string_view (*)(rocprofiler_callback_tracing_kind_t,
                                                            uint32_t);
using tool_get_roctx_msg_fn_t          = std::string_view (*)(uint64_t);
using tool_get_counter_info_name_fn_t  = std::string (*)(uint64_t);

struct tool_table
{
    // node id
    tool_get_agent_node_id_fn_t tool_get_agent_node_id_fn = nullptr;
    // timestamps
    tool_get_app_timestamps_fn_t tool_get_app_timestamps_fn = nullptr;
    // names and messages
    tool_get_kernel_name_fn_t        tool_get_kernel_name_fn       = nullptr;
    tool_get_domain_name_fn_t        tool_get_domain_name_fn       = nullptr;
    tool_get_operation_name_fn_t     tool_get_operation_name_fn    = nullptr;
    tool_get_counter_info_name_fn_t  tool_get_counter_info_name_fn = nullptr;
    tool_get_callback_kind_name_fn_t tool_get_callback_kind_fn     = nullptr;
    tool_get_callback_op_name_fn_t   tool_get_callback_op_name_fn  = nullptr;
    tool_get_roctx_msg_fn_t          tool_get_roctx_msg_fn         = nullptr;
};

/// converts a container of ring buffers of element Tp into a single container of elements
template <typename Tp, template <typename, typename...> class ContainerT, typename... ParamsT>
ContainerT<Tp>
get_buffer_elements(ContainerT<rocprofiler::common::container::ring_buffer<Tp>, ParamsT...>&& data)
{
    auto ret = ContainerT<Tp>{};
    for(auto& buf : data)
    {
        Tp* record = nullptr;
        do
        {
            record = buf.retrieve();
            if(record) ret.emplace_back(*record);
        } while(record != nullptr);
    }

    return ret;
}

namespace cereal
{
#define SAVE_DATA_FIELD(FIELD) ar(make_nvp(#FIELD, data.FIELD))

template <typename ArchiveT>
void
save(ArchiveT& ar, const kernel_symbol_data& data)
{
    cereal::save(ar, static_cast<const rocprofiler_kernel_symbol_data_t&>(data));
    SAVE_DATA_FIELD(formatted_kernel_name);
    SAVE_DATA_FIELD(demangled_kernel_name);
    SAVE_DATA_FIELD(truncated_kernel_name);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const rocprofiler_tool_counter_info_t& data)
{
    SAVE_DATA_FIELD(agent_id);
    cereal::save(ar, static_cast<const rocprofiler_counter_info_v0_t&>(data));
    SAVE_DATA_FIELD(dimension_ids);
}

#undef SAVE_DATA_FIELD
}  // namespace cereal
