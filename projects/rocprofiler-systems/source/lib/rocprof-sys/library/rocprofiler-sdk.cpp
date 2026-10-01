// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk.hpp"
#include "api.hpp"
#include "backends/rocprofiler_sdk/backend.hpp"
#include "backends/rocprofiler_sdk/wrapper.hpp"
#include "binary/analysis.hpp"
#include "common/delimit.hpp"
#include "common/env_vars.hpp"
#include "common/path.hpp"
#include "common/synchronized.hpp"
#include "core/agent.hpp"
#include "core/agent_manager.hpp"
#include "core/common.hpp"
#include "core/common_types.hpp"
#include "core/config.hpp"
#include "core/containers/stable_vector.hpp"
#include "core/control/session.hpp"
#include "core/demangler.hpp"
#include "core/gpu.hpp"
#include "core/output_file_registry.hpp"
#include "core/perfetto.hpp"
#include "core/perfetto_fwd.hpp"
#include "core/sdk/tracing-config-deps.hpp"
#include "core/sdk/tracing-config.hpp"
#include "core/state.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/sampler.hpp"
#include "library/process_sampler.hpp"
#include "library/rocprofiler-sdk/counters.hpp"
#include "library/rocprofiler-sdk/domain_selection.hpp"
#include "library/rocprofiler-sdk/domain_service.hpp"
#include "library/rocprofiler-sdk/fwd.hpp"
#include "library/rocprofiler-sdk/stream_stack_service.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"
#include "rocprofiler-sdk.hpp"
#include "rocprofiler-sdk/roctx_client.hpp"

#include <algorithm>
#include <timemory/components/timing/wall_clock.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/unwind/processed_entry.hpp>
#include <timemory/utility/backtrace.hpp>
#include <timemory/variadic/lightweight_tuple.hpp>

#include <cstddef>
#include <exception>
#include <iterator>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

#include <fmt/format.h>
#include <nlohmann/json_fwd.hpp>

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <rocprofiler-sdk/version.h>

#if __has_include(<rocprofiler-sdk/experimental/registration.h>)
#    include <rocprofiler-sdk/experimental/registration.h>
#else
#    include <rocprofiler-sdk/registration.h>
#endif

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <timemory/process/threading.hpp>
#include <timemory/utility/types.hpp>

#include <fmt/ranges.h>
#include <nlohmann/json.hpp>

#include "logger/debug.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocprofsys::rocprofiler_sdk
{
namespace
{

auto
get_backtrace(std::optional<std::vector<tim::unwind::processed_entry>>& bt_data)
{
    auto backtrace = nlohmann::json::object();

    if(bt_data && !bt_data->empty())
    {
        const std::string unk    = "??";
        size_t            bt_cnt = 0;
        for(const auto& itr : *bt_data)
        {
            auto        linfo        = itr.lineinfo.get();
            const auto* func         = itr.name.empty() ? &unk : &itr.name;
            const auto* fallback_loc = itr.location.empty() ? &unk : &itr.location;
            const auto* loc =
                (linfo && !linfo.location.empty()) ? &linfo.location : fallback_loc;
            const auto fallback_line =
                (itr.lineno == 0) ? std::string{ "?" } : fmt::format("{}", itr.lineno);
            const auto line =
                (linfo && linfo.line > 0) ? fmt::format("{}", linfo.line) : fallback_line;

            auto const entry =
                fmt::format("{} @ {}:{}", rocprofsys::utility::demangle(*func),
                            path::filename(*loc), line);
            backtrace[fmt::format("frame#{}", bt_cnt++)] = entry;
        }
    }
    return backtrace;
}

// NOLINTBEGIN(readability-function-size)
// Implementation of rocprofiler_callback_tracing_operation_args_cb_t
int
save_args(rocprofiler_callback_tracing_kind_t /*kind*/, std::int32_t /*operation*/,
          std::uint32_t /*arg_number*/, const void* const /*arg_value_addr*/,
          std::int32_t /*arg_indirection_count*/, const char* /*arg_type*/,
          const char* arg_name, const char*             arg_value_str,
          std::int32_t /*arg_dereference_count*/, void* data)
{
    auto* argvec = static_cast<callback_arg_array_t*>(data);
    argvec->emplace_back(arg_name, arg_value_str);
    return 0;
}

// Additional implementation of rocprofiler_callback_tracing_operation_args_cb_t
// for iterating through arguments in a callback for rocpd_arg table in database
int
iterate_args_callback(rocprofiler_callback_tracing_kind_t /*kind*/,
                      std::int32_t /*operation*/, std::uint32_t arg_number,
                      const void* const /*arg_value_addr*/,
                      std::int32_t /*arg_indirection_count*/, const char* arg_type,
                      const char* arg_name, const char*             arg_value_str,
                      std::int32_t /*arg_dereference_count*/, void* data)
{
    auto* func_args = static_cast<function_args_t*>(data);
    if(arg_type && arg_name && arg_value_str)
    {
        func_args->emplace_back(
            argument_info{ .arg_number = arg_number,
                           .arg_type   = rocprofsys::utility::demangle(arg_type),
                           .arg_name   = arg_name,
                           .arg_value  = arg_value_str });
    }
    return 0;
}
// NOLINTEND(readability-function-size)

// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
client_data* g_tool_data = new client_data{};

using rocprofiler_sdk::default_externals;
using rocprofiler_sdk::tracing_config;
using rocprofiler_sdk::wrapper;

using production_backend = backends::rocprofiler_sdk::backend<rocprofiler_sdk::wrapper>;
using production_stream_stack_service = stream_stack_service<production_backend>;

struct external_dependencies
{
    using agent_t         = ::rocprofsys::agent;
    using agent_type_t    = ::rocprofsys::agent_type;
    using agent_manager_t = ::rocprofsys::agent_manager;
    using track_t         = trace_cache::info::track;
    using thread_info_t   = trace_cache::info::thread;
    using pmc_info_t      = trace_cache::info::pmc;
    using kfd_sample_t    = trace_cache::kfd_sample;

    static constexpr agent_type_t k_agent_type_gpu = agent_type_t::gpu;
    static constexpr agent_type_t k_agent_type_cpu = agent_type_t::cpu;

    static agent_manager_t& get_agent_manager()
    {
        return ::rocprofsys::get_agent_manager_instance();
    }

    static std::int32_t get_pid() { return static_cast<std::int32_t>(::getpid()); }
    static std::int32_t get_ppid() { return static_cast<std::int32_t>(::getppid()); }

    // ─── Members required by domains::callback::hip::{runtime,compiler}_api ────────
    using rocm_hip_api_category = category::rocm_hip_api;
    using region_sample         = trace_cache::region_sample;

    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr std::string_view rocm_hip_api_category_name =
        trait::name<category::rocm_hip_api>::value;

    // ─── Members required by domains::callback::hsa::{core,amd_ext,image_ext,
    // finalize_ext}_api ─────────────────────────────────────────────────────────
    using rocm_hsa_api_category = category::rocm_hsa_api;

    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr std::string_view rocm_hsa_api_category_name =
        trait::name<category::rocm_hsa_api>::value;

    // ─── Members required by domains::callback::{rocjpeg,rocdecode,rocshmem,
    // hipfile}_api ──────────────────────────────────────────────────────────────
    using rocm_rocjpeg_api_category = category::rocm_rocjpeg_api;

    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr std::string_view rocm_rocjpeg_api_category_name =
        trait::name<category::rocm_rocjpeg_api>::value;

    using rocm_rocdecode_api_category = category::rocm_rocdecode_api;

    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr std::string_view rocm_rocdecode_api_category_name =
        trait::name<category::rocm_rocdecode_api>::value;

    using rocm_rocshmem_api_category = category::rocm_rocshmem_api;

    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr std::string_view rocm_rocshmem_api_category_name =
        trait::name<category::rocm_rocshmem_api>::value;

    using rocm_hipfile_api_category = category::rocm_hipfile_api;

    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr std::string_view rocm_hipfile_api_category_name =
        trait::name<category::rocm_hipfile_api>::value;

    static bool is_active()
    {
        return ::rocprofsys::state::process::get() ==
               ::rocprofsys::state::process::Active;
    }

    static bool get_use_timemory() { return ::rocprofsys::get_use_timemory(); }

    template <typename CategoryT>
    static void tracing_push_timemory(CategoryT, std::string_view name)
    {
        tracing::push_timemory(CategoryT{}, name);
    }

    template <typename CategoryT>
    static void tracing_pop_timemory(CategoryT, std::string_view name)
    {
        tracing::pop_timemory(CategoryT{}, name);
    }

    // ─── Members required by domains::callback::k_rccl ──────────────────────────────
    using rocm_rccl_api_category = category::rocm_rccl_api;
    using pmc_event_with_sample  = trace_cache::pmc_event_with_sample;
    using metadata_registry_t    = trace_cache::metadata_registry;
    using buffer_storage_t       = trace_cache::buffer_storage_t;

    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr std::string_view rocm_rccl_api_category_name =
        trait::name<category::rocm_rccl_api>::value;

    // Single source of truth for these strings is core/categories.hpp's
    // trait::name<category::comm_data>; rccl.hpp itself never includes
    // categories.hpp, so the values are surfaced here instead.
    static constexpr std::string_view comm_data_name =
        trait::name<category::comm_data>::value;
    static constexpr std::string_view comm_data_description =
        trait::name<category::comm_data>::description;
    static constexpr std::size_t comm_data_enum_value =
        static_cast<std::size_t>(category_enum_id<category::comm_data>::value);

    static constexpr std::string_view rccl_send_label      = "RCCL Comm Send";
    static constexpr std::string_view rccl_recv_label      = "RCCL Comm Recv";
    static constexpr std::string_view rccl_send_track_name = rccl_send_label;
    static constexpr std::string_view rccl_recv_track_name = rccl_recv_label;

    // ─── Members required by domains::callback::ompt::k_ompt_api
    // ─────────────────────────
    using rocm_ompt_api_category = category::rocm_ompt_api;

    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr std::string_view rocm_ompt_api_category_name =
        trait::name<category::rocm_ompt_api>::value;

    static metadata_registry_t& get_metadata_registry()
    {
        return trace_cache::get_metadata_registry();
    }

    static buffer_storage_t& get_buffer_storage()
    {
        return trace_cache::get_buffer_storage();
    }

    static void* dlsym(const char* symbol_name)
    {
        return ::dlsym(RTLD_DEFAULT, symbol_name);
    }

    static const char* dlerror() { return ::dlerror(); }

    static bool check_backtrace_operations(rocprofiler_callback_tracing_kind_t kind,
                                           rocprofiler_tracing_operation_t     operation)
    {
        return g_tool_data->backtrace_operations.at(kind).contains(operation);
    }

    static auto get_backtrace_data(bool are_operations_available)
    {
        constexpr size_t k_backtrace_stack_depth       = 16;
        constexpr size_t k_backtrace_ignore_depth      = 3;
        constexpr bool   k_backtrace_with_signal_frame = true;
        auto const       use_perfetto =
            (config::get_use_perfetto() && config::get_perfetto_annotations());
        auto const use_rocpd = config::get_use_rocpd();

        const auto should_we_generate_backtrace =
            (use_perfetto || use_rocpd) && are_operations_available;

        auto result = std::optional<std::vector<tim::unwind::processed_entry>>{};

        if(!should_we_generate_backtrace)
        {
            return result;
        }

        auto const backtrace_stack =
            tim::get_unw_stack<k_backtrace_stack_depth, k_backtrace_ignore_depth,
                               k_backtrace_with_signal_frame>();
        result = std::vector<tim::unwind::processed_entry>{};
        result->reserve(backtrace_stack.size());

        for(const auto& itr : backtrace_stack)
        {
            if(!itr)
            {
                continue;
            }

            auto value = binary::lookup_ipaddr_entry<false>(itr->address());
            if(!value)
            {
                continue;
            }

            result->emplace_back(std::move(*value));
        }

        return result;
    }

    static auto get_backtrace_json(
        std::optional<std::vector<tim::unwind::processed_entry>>& bt_data)
    {
        return get_backtrace(bt_data);
    }

    // ─── kernel_dispatch buffered-domain dependencies ────────────────────────────
    using kernel_dispatch_sample_t = trace_cache::kernel_dispatch_sample;

    static constexpr std::string_view k_kernel_dispatch_category_name =
        trait::name<category::rocm_kernel_dispatch>::value;

    // ─── memory_copy buffered-domain dependencies ────────────────────────────────
    using memory_copy_sample_t = trace_cache::memory_copy_sample;

    static constexpr std::string_view k_memory_copy_category_name =
        trait::name<category::rocm_memory_copy>::value;

    // ─── memory_allocation buffered-domain dependencies ──────────────────────────
    using memory_allocation_sample_t = trace_cache::memory_allocate_sample;

    static constexpr std::string_view k_memory_allocation_category_name =
        trait::name<category::rocm_memory_allocate>::value;

    // ─── scratch_memory buffered-domain dependencies ─────────────────────────────
    using scratch_memory_sample_t = trace_cache::scratch_memory_sample;

    static constexpr std::string_view k_scratch_memory_category_name =
        trait::name<category::rocm_scratch_memory>::value;

    static std::string_view get_kernel_symbol_name(std::uint64_t kernel_id)
    {
        auto symbol = get_metadata_registry().get_kernel_symbol(kernel_id);
        return symbol.has_value() ? std::string_view{ symbol->kernel_name }
                                  : std::string_view{};
    }

    static std::optional<std::uint64_t> get_thread_info_sequent_tid(std::uint64_t tid)
    {
        const auto& thread_info_data =
            thread_info::get(static_cast<std::int64_t>(tid), SystemTID);
        if(!thread_info_data.has_value() || !thread_info_data->index_data.has_value())
        {
            return std::nullopt;
        }

        return thread_info_data->index_data->sequent_value;
    }

    static void write_timemory_bundle(std::string_view name, std::uint64_t tid,
                                      std::uint64_t elapsed_ns)
    {
        using kernel_dispatch_bundle_t =
            tim::lightweight_tuple<tim::component::wall_clock>;

        auto kernel_dispatch_bundle_data = kernel_dispatch_bundle_t{ name };
        kernel_dispatch_bundle_data.push(static_cast<std::int64_t>(tid)).start().stop();
        kernel_dispatch_bundle_data.get(
            [elapsed_ns](tim::component::wall_clock* wall_clock_data) {
                wall_clock_data->set_value(static_cast<std::int64_t>(elapsed_ns));
                wall_clock_data->set_accum(static_cast<std::int64_t>(elapsed_ns));
            });

        kernel_dispatch_bundle_data.pop();
    }

    using state_thread = state::thread;

    // Single source of truth is core/trace_cache/cacheable.hpp's ABSOLUTE constant;
    // kfd_events.hpp never includes that header, so the value is surfaced here.
    static constexpr std::string_view k_pmc_value_type_absolute = trace_cache::ABSOLUTE;

    // Single source of truth for these strings is core/categories.hpp's
    // trait::name<category::X>; kfd_events.hpp itself never includes
    // categories.hpp, so the values are surfaced here instead.
    static constexpr std::string_view k_kfd_page_fault_category_name =
        trait::name<category::rocm_kfd_page_fault>::value;
    static constexpr std::string_view k_kfd_page_fault_category_description =
        trait::name<category::rocm_kfd_page_fault>::description;
    static constexpr std::string_view k_kfd_page_migrate_category_name =
        trait::name<category::rocm_kfd_page_migrate>::value;
    static constexpr std::string_view k_kfd_page_migrate_category_description =
        trait::name<category::rocm_kfd_page_migrate>::description;
    static constexpr std::string_view k_kfd_event_page_fault_category_name =
        trait::name<category::rocm_kfd_event_page_fault>::value;
    static constexpr std::string_view k_kfd_event_page_fault_category_description =
        trait::name<category::rocm_kfd_event_page_fault>::description;
    static constexpr std::string_view k_kfd_event_page_migrate_category_name =
        trait::name<category::rocm_kfd_event_page_migrate>::value;
    static constexpr std::string_view k_kfd_event_page_migrate_category_description =
        trait::name<category::rocm_kfd_event_page_migrate>::description;
    static constexpr std::string_view k_kfd_queue_category_name =
        trait::name<category::rocm_kfd_queue>::value;
    static constexpr std::string_view k_kfd_queue_category_description =
        trait::name<category::rocm_kfd_queue>::description;
    static constexpr std::string_view k_kfd_event_queue_category_name =
        trait::name<category::rocm_kfd_event_queue>::value;
    static constexpr std::string_view k_kfd_event_queue_category_description =
        trait::name<category::rocm_kfd_event_queue>::description;
    static constexpr std::string_view k_kfd_event_unmap_from_gpu_category_name =
        trait::name<category::rocm_kfd_event_unmap_from_gpu>::value;
    static constexpr std::string_view k_kfd_event_unmap_from_gpu_category_description =
        trait::name<category::rocm_kfd_event_unmap_from_gpu>::description;
    static constexpr std::string_view k_kfd_event_dropped_events_category_name =
        trait::name<category::rocm_kfd_event_dropped_events>::value;
    static constexpr std::string_view k_kfd_event_dropped_events_category_description =
        trait::name<category::rocm_kfd_event_dropped_events>::description;
};

std::shared_ptr<domain_service<production_backend, external_dependencies>>
    g_domain_service;

using tool_agent_vec_t = std::vector<tool_agent>;
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
std::shared_ptr<roctx_client<>>   g_roctx_client = {};
std::shared_ptr<control::session> g_session      = {};

std::atomic<bool> tool_fini_done{ false };
std::atomic<bool> tool_init_done{ false };

void
thread_precreate(rocprofiler_runtime_library_t /*lib*/, void* /*tool_data*/)
{
    state::thread::push(state::thread::Internal);
}

void
thread_postcreate(rocprofiler_runtime_library_t /*lib*/, void* /*tool_data*/)
{
    state::thread::pop();
}

#if(ROCPROFILER_VERSION < 700)
/**
 * @brief Stream ID.
 */
typedef struct rocprofiler_stream_id_t
{
    std::uint64_t handle;
} rocprofiler_stream_id_t;

#endif

// this function creates a rocprofiler profile config on the first entry
std::vector<rocprofiler_counter_id_t>
create_agent_profile(rocprofiler_agent_id_t          agent_id,
                     const std::vector<std::string>& counters,
                     //  const tool_agent_vec_t&         gpu_agents,
                     //  const agent_counter_info_map_t& counters_info,
                     //  agent_counter_profile_map_t&    data)
                     client_data* data = g_tool_data)
{
    using counter_vec_t = std::vector<rocprofiler_counter_id_t>;

    // check if already created
    if(data->agent_counter_profiles.contains(agent_id))
    {
        return counter_vec_t{};
    }

    auto        profile      = std::optional<rocprofiler_profile_config_id_t>{};
    auto        expected_v   = counters.size();
    auto        found_v      = std::vector<std::string_view>{};
    auto        counters_v   = counter_vec_t{};
    const auto* tool_agent_v = data->get_gpu_tool_agent(agent_id);

    // Check if agent info is available (may not be for unsupported architectures)
    auto const agent_info_it = data->agent_counter_info.find(agent_id);
    if(agent_info_it == data->agent_counter_info.end())
    {
        LOG_WARNING("Skipping GPU agent {} (device {}) due to unsupported "
                    "architecture or missing counter info",
                    agent_id.handle, tool_agent_v->device_id);

        data->agent_counter_profiles.emplace(agent_id, profile);
        return counter_vec_t{};
    }

    constexpr auto device_qualifier = std::string_view{ ":device=" };
    for(const auto& itr : counters)
    {
        auto name_v = itr;
        if(auto pos = std::string::npos;
           (pos = itr.find(device_qualifier)) != std::string::npos)
        {
            name_v              = itr.substr(0, pos);
            auto const dev_id_s = itr.substr(pos + device_qualifier.length());

            if(dev_id_s.empty() ||
               dev_id_s.find_first_not_of("0123456789") != std::string::npos)
            {
                LOG_CRITICAL("invalid device qualifier format (':device=N) "
                             "where N is the GPU id: {}",
                             itr);
                ::rocprofsys::state::process::set(
                    ::rocprofsys::state::process::Finalized);
                std::abort();
            }

            auto dev_id_v = std::stoul(dev_id_s);

            LOG_DEBUG("tool agent device id={}, name={}, device_id={}",
                      tool_agent_v->device_id, name_v, dev_id_v);

            // skip this counter if the counter is for a specific device id (which
            // doesn't this agent's device id)
            if(dev_id_v != tool_agent_v->device_id)
            {
                --expected_v;  // is not expected
                continue;
            }
        }

        // Removes any numeric index enclosed in square brackets at the end of the
        // string. For example, "example[123]" will be converted to "example".
        auto _old_name_v = name_v;
        name_v =
            std::regex_replace(name_v, std::regex{ "^(.*)(\\[)([0-9]+)(\\])$" }, "$1");

        if(name_v != _old_name_v)
        {
            LOG_DEBUG("tool agent device id={}, old_name={}, name={}",
                      tool_agent_v->device_id, _old_name_v, name_v);
        }
        else if(name_v == itr)
        {
            LOG_DEBUG("tool agent device id={}, name={}", tool_agent_v->device_id,
                      name_v);
        }

        // search the gpu agent counter info for a counter with a matching name
        for(const auto& citr : agent_info_it->second)
        {
            if(name_v == std::string_view{ citr.name })
            {
                counters_v.emplace_back(citr.id);
                found_v.emplace_back(itr);
            }
        }
    }

    if(counters_v.size() != expected_v)
    {
        auto requested_counters = fmt::format("{}", fmt::join(counters, ", "));
        auto found_counters     = fmt::format("{}", fmt::join(found_v, ", "));

        // Determine which counters were not found
        auto missing_counters = std::vector<std::string>{};
        for(const auto& counter : counters)
        {
            if(std::ranges::find(found_v, counter) == found_v.end())
            {
                missing_counters.emplace_back(counter);
            }
        }
        auto missing_counters_str = fmt::format("{}", fmt::join(missing_counters, ", "));

        LOG_WARNING("Unable to find all counters for agent {} (gpu-{}, {}). "
                    "Requested: {}. Found: {}. Missing: {}. Continuing with "
                    "available counters.",
                    tool_agent_v->agent->node_id, tool_agent_v->device_id,
                    tool_agent_v->agent->name, requested_counters, found_counters,
                    missing_counters_str);
    }

    if(!counters_v.empty())
    {
        auto profile_v = rocprofiler_profile_config_id_t{};
        ROCPROFILER_CALL(rocprofiler_create_profile_config(
            agent_id, counters_v.data(), counters_v.size(), &profile_v));
        profile = profile_v;
    }

    data->agent_counter_profiles.emplace(agent_id, profile);

    return counters_v;
}

const kernel_symbol_data_t*
get_kernel_symbol_info(std::uint64_t _kernel_id)
{
    return g_tool_data->get_kernel_symbol_info(_kernel_id);
}

const rocprofiler_callback_tracing_code_object_load_data_t*
get_code_object_info(std::uint64_t _code_object_id)
{
    return g_tool_data->get_code_object_info(_code_object_id);
}

// template <typename Tp, typename... Args>
// Tp*
// as_pointer(Args&&... _args)
// {
//     return new Tp{ std::forward<Args>(_args)... };
// }

// template <typename... Tp>
// void
// consume_args(Tp&&...)
// {}

template <typename CorrelationIdType>
std::uint64_t
get_parent_stack_id([[maybe_unused]] const CorrelationIdType& correlation_id)
{
#if(ROCPROFILER_VERSION >= 700)
    if constexpr(std::is_same_v<rocprofiler_correlation_id_t, CorrelationIdType>)
    {
        return correlation_id.ancestor;
    }
    else
    {
        return 0;
    }
#else
    return 0;
#endif
}

// struct scope_destructor
// {
//     /// \fn scope_destructor(FuncT&& _fini, InitT&& _init)
//     /// \tparam FuncT "std::function<void()> or void (*)()"
//     /// \tparam InitT "std::function<void()> or void (*)()"
//     /// \param _fini Function to execute when object is destroyed
//     /// \param _init Function to execute when object is created (optional)
//     ///
//     /// \brief Provides a utility to perform an operation when exiting a scope.
//     template <typename FuncT, typename InitT = void (*)()>
//     scope_destructor(FuncT&& _fini, InitT&& _init = []() {});

//     ~scope_destructor() { m_functor(); }

//     // delete copy operations
//     scope_destructor(const scope_destructor&)            = delete;
//     scope_destructor& operator=(const scope_destructor&) = delete;

//     // allow move operations
//     scope_destructor(scope_destructor&& rhs) noexcept;
//     scope_destructor& operator=(scope_destructor&& rhs) noexcept;

// private:
//     std::function<void()> m_functor = []() {};
// };

// template <typename FuncT, typename InitT>
// scope_destructor::scope_destructor(FuncT&& _fini, InitT&& _init)
// : m_functor{ std::forward<FuncT>(_fini) }
// {
//     _init();
// }

// inline scope_destructor::scope_destructor(scope_destructor&& rhs) noexcept
// : m_functor{ std::move(rhs.m_functor) }
// {
//     rhs.m_functor = []() {};
// }

// inline scope_destructor&
// scope_destructor::operator=(scope_destructor&& rhs) noexcept
// {
//     if(this != &rhs)
//     {
//         m_functor     = std::move(rhs.m_functor);
//         rhs.m_functor = []() {};
//     }
//     return *this;
// }

// using kernel_rename_stack_t = std::stack<std::uint64_t>;

// thread_local auto thread_dispatch_rename      = as_pointer<kernel_rename_stack_t>();
// thread_local auto thread_dispatch_rename_dtor = scope_destructor{ []() {
//     delete thread_dispatch_rename;
//     thread_dispatch_rename = nullptr;
// } };

template <typename Category>
void
cache_category()
{
    trace_cache::get_metadata_registry().add_string(trait::name<Category>::value);
}

void
cache_add_thread_info(std::uint64_t tid)
{
    trace_cache::get_metadata_registry().add_thread_info({ .parent_process_id = getppid(),
                                                           .process_id        = getpid(),
                                                           .thread_id         = tid,
                                                           .start             = 0,
                                                           .end               = 0,
                                                           .extdata           = "{}" });
}

size_t
get_mem_copy_dst_address(
    [[maybe_unused]] const rocprofiler_buffer_tracing_memory_copy_record_t& record)
{
#if(ROCPROFILER_VERSION >= 700)
    return record.dst_address.value;
#else
    return 0;
#endif
}

size_t
get_mem_copy_src_address(
    [[maybe_unused]] const rocprofiler_buffer_tracing_memory_copy_record_t& record)
{
#if(ROCPROFILER_VERSION >= 700)
    return record.src_address.value;
#else
    return 0;
#endif
}

#if(ROCPROFILER_VERSION >= 600)
size_t
get_mem_alloc_address(
    [[maybe_unused]] const rocprofiler_buffer_tracing_memory_allocation_record_t& record)
{
#    if(ROCPROFILER_VERSION >= 700)
    return record.address.value;
#    else
    return static_cast<size_t>(record.address.handle);
#    endif
}
#endif

std::uint64_t
get_scratch_mem_alloc_size(
    [[maybe_unused]] const rocprofiler_buffer_tracing_scratch_memory_record_t& record)
{
// The version of rocprofiler_buffer_tracing_scratch_memory_record_t from ROCm < 7.1 does
// not have the allocation_size field. ROCPROFILER_VERSION for both ROCm 7.0 and 7.1
// is 1.0.0, so we need to check the ROCm version.
#if ROCPROFSYS_ROCM_VERSION >= 70100
    return record.allocation_size;
#else
    return 0;
#endif
}

void
cache_kernel_dispatch(rocprofiler_buffer_tracing_kernel_dispatch_record_t* record,
                      std::uint64_t                                        stream_handle)
{
    auto queue_handle = record->dispatch_info.queue_id.handle;

    trace_cache::get_metadata_registry().add_queue(queue_handle);
    trace_cache::get_metadata_registry().add_stream(stream_handle);

    trace_cache::get_buffer_storage().store(trace_cache::kernel_dispatch_sample{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->dispatch_info.agent_id.handle, record->dispatch_info.kernel_id,
        record->dispatch_info.dispatch_id, record->dispatch_info.queue_id.handle,
        record->correlation_id.internal, get_parent_stack_id(record->correlation_id),
        record->dispatch_info.private_segment_size,
        record->dispatch_info.group_segment_size, record->dispatch_info.workgroup_size.x,
        record->dispatch_info.workgroup_size.y, record->dispatch_info.workgroup_size.z,
        record->dispatch_info.grid_size.x, record->dispatch_info.grid_size.y,
        record->dispatch_info.grid_size.z, stream_handle });
}

// The cached samples carry the SDK operation name, so every name the SDK can
// report must already be in the metadata string table.
void
register_operation_name_strings()
{
    auto& metadata = trace_cache::get_metadata_registry();

    for(const auto& buffer_info : production_backend::get_buffer_tracing_names())
    {
        for(const auto& item : buffer_info.items())
        {
            metadata.add_string(*item.second);
        }
    }

    for(const auto& callback_info : production_backend::get_callback_tracing_names())
    {
        for(const auto& item : callback_info.items())
        {
            metadata.add_string(*item.second);
        }
    }
}

void
cache_scratch_memory(rocprofiler_buffer_tracing_scratch_memory_record_t* record,
                     std::uint64_t                                       stream_handle)
{
    trace_cache::get_metadata_registry().add_queue(record->queue_id.handle);
    trace_cache::get_metadata_registry().add_stream(stream_handle);
    trace_cache::get_buffer_storage().store(trace_cache::scratch_memory_sample{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->agent_id.handle, record->queue_id.handle,
        production_backend::get_buffer_tracing_names().at(record->kind,
                                                          record->operation),
        static_cast<std::int32_t>(record->operation),
        static_cast<std::int32_t>(record->flags), get_scratch_mem_alloc_size(*record),
        record->correlation_id.internal, get_parent_stack_id(record->correlation_id),
        stream_handle });
}

void
cache_memory_copy(rocprofiler_buffer_tracing_memory_copy_record_t* record,
                  std::uint64_t                                    stream_handle)
{
    trace_cache::get_metadata_registry().add_stream(stream_handle);
    trace_cache::get_buffer_storage().store(trace_cache::memory_copy_sample{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->dst_agent_id.handle, record->src_agent_id.handle,
        production_backend::get_buffer_tracing_names().at(record->kind,
                                                          record->operation),
        record->bytes, record->correlation_id.internal,
        get_parent_stack_id(record->correlation_id), get_mem_copy_dst_address(*record),
        get_mem_copy_src_address(*record), stream_handle });
}

#if(ROCPROFILER_VERSION >= 600)
void
cache_memory_allocation(rocprofiler_buffer_tracing_memory_allocation_record_t* record,
                        std::uint64_t stream_handle)
{
    trace_cache::get_metadata_registry().add_stream(stream_handle);
    trace_cache::get_buffer_storage().store(trace_cache::memory_allocate_sample{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->agent_id.handle,
        production_backend::get_buffer_tracing_names().at(record->kind,
                                                          record->operation),
        static_cast<std::int32_t>(record->operation), record->allocation_size,
        record->correlation_id.internal, get_parent_stack_id(record->correlation_id),
        get_mem_alloc_address(*record), stream_handle });
}
#endif

void
tool_code_object_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t* /*user_data*/, void* /*callback_data*/)
{
    auto ts = rocprofiler_timestamp_t{};
    ROCPROFILER_CALL(rocprofiler_get_timestamp(&ts));

    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            if(record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
            {
                auto data_v =
                    *static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(
                        record.payload);
                g_tool_data->code_object_records.wlock(
                    [ts, &record, &data_v](auto& data) {
                        data.emplace_back(code_object_callback_record_t{
                            .timestamp = ts, .record = record, .payload = data_v });
                    });
                trace_cache::get_metadata_registry().add_code_object(data_v);
            }
            else if(record.operation ==
                    ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
            {
                auto data_v = *static_cast<kernel_symbol_data_t*>(record.payload);
                g_tool_data->kernel_symbol_records.wlock(
                    [ts, &record, &data_v](auto& _data) {
                        _data.emplace_back(kernel_symbol_callback_record_t{
                            .timestamp = ts, .record = record, .payload = data_v });
                    });
                trace_cache::get_metadata_registry().add_kernel_symbol(data_v);
            }
        }
        return;
    }
}

auto&
get_kernel_dispatch_timestamps()
{
    static auto _v = std::unordered_map<rocprofiler_dispatch_id_t, timing_interval>{};
    return _v;
}

void
tool_tracing_callback(rocprofiler_callback_tracing_record_t     record,
                      [[maybe_unused]] rocprofiler_user_data_t* user_data,
                      void* /*callback_data*/)
{
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_NONE)
    {
        switch(record.kind)
        {
            case ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH:
            {
                if(record.operation == ROCPROFILER_KERNEL_DISPATCH_COMPLETE)
                {
                    auto const* _data =
                        static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(
                            record.payload);

                    // save for post-processing
                    get_kernel_dispatch_timestamps().emplace(
                        _data->dispatch_info.dispatch_id,
                        timing_interval{ .start = _data->start_timestamp,
                                         .end   = _data->end_timestamp });
                }
            }
            break;
            default:
            {
                LOG_WARNING("tool_tracing_callback: unhandled PHASE_NONE "
                            "callback record");
            }
            break;
        }
    }
}

auto&
get_counter_dispatch_data()
{
    static auto _v =
        container::stable_vector<rocprofiler_dispatch_counting_service_data_t>{};
    return _v;
}

auto&
get_counter_dispatch_records()
{
    static auto _v = std::vector<counter_dispatch_record>{};
    return _v;
}

using counter_storage_map_t =
    std::unordered_map<rocprofiler_counter_id_t, counter_storage>;
using agent_counter_storage_map_t =
    std::unordered_map<rocprofiler_agent_id_t, counter_storage_map_t>;

auto*&
get_counter_storage()
{
    static auto* _v = new agent_counter_storage_map_t{};
    return _v;
}

void
flush_counter_storage_outputs()
{
    auto const* _agent_counter_storage = get_counter_storage();
    if(!_agent_counter_storage)
    {
        return;
    }

    auto _cleanup_keys = std::vector<std::pair<std::string, const counter_storage*>>{};
    for(const auto& [agent_id, counter_map] : *_agent_counter_storage)
    {
        static_cast<void>(agent_id);
        for(const auto& [counter_id, storage] : counter_map)
        {
            static_cast<void>(counter_id);
            _cleanup_keys.emplace_back(storage.storage_name + "cleanup", &storage);
        }
    }

    std::ranges::sort(_cleanup_keys, [](const auto& lhs, const auto& rhs) {
        return *lhs.second < *rhs.second;
    });

    for(const auto& [cleanup_key, storage] : _cleanup_keys)
    {
        if(!storage || !storage->storage)
        {
            continue;
        }
        if(storage->manager)
        {
            storage->manager->cleanup(cleanup_key);
        }
        else
        {
            // No tim::manager at construction (add_cleanup was skipped); still flush
            // timemory storage once at shutdown.
            counter_storage::write(storage->storage.get(), storage->metric_name,
                                   storage->metric_description);
        }
    }
}

void
// NOLINTNEXTLINE(misc-const-correctness) - signature must match
// rocprofiler_dispatch_counting_record_cb_t exactly
counter_record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                        rocprofiler_record_counter_t* record_data, size_t record_count,
                        rocprofiler_user_data_t /*user_data*/,
                        void* /*callback_data_arg*/)
{
    auto* _agent_counter_storage = get_counter_storage();
    if(!_agent_counter_storage)
    {
        return;
    }

    static auto _mtx = std::mutex{};
    auto const  _lk  = std::unique_lock<std::mutex>{ _mtx };

    auto const _dispatch_id = dispatch_data.dispatch_info.dispatch_id;
    auto       _agent_id    = dispatch_data.dispatch_info.agent_id;
    auto const _scope       = scope::get_default();
    auto       _interval    = timing_interval{};
    auto       _aggregate =
        std::unordered_map<rocprofiler_counter_id_t, rocprofiler_record_counter_t>{};
    for(size_t i = 0; i < record_count; ++i)
    {
        auto _counter_id = rocprofiler_counter_id_t{};
        ROCPROFILER_CALL(
            rocprofiler_query_record_counter_id(record_data[i].id, &_counter_id));

        if(!_aggregate.emplace(_counter_id, record_data[i]).second)
        {
            _aggregate[_counter_id].counter_value += record_data[i].counter_value;
        }
    }

    if(!_agent_counter_storage->contains(_agent_id))
    {
        _agent_counter_storage->emplace(_agent_id, counter_storage_map_t{});
    }

    if(get_kernel_dispatch_timestamps().contains(_dispatch_id))
    {
        _interval = get_kernel_dispatch_timestamps().at(_dispatch_id);
        get_kernel_dispatch_timestamps().erase(_dispatch_id);
    }

    for(const auto& itr : _aggregate)
    {
        if(!_agent_counter_storage->at(_agent_id).contains(itr.first))
        {
            const auto* agent = g_tool_data->get_gpu_tool_agent(_agent_id);
            const auto* info  = g_tool_data->get_tool_counter_info(_agent_id, itr.first);

            if(!agent)
            {
                LOG_CRITICAL("unable to find tool agent for agent (id={})",
                             _agent_id.handle);
                ::rocprofsys::state::process::set(
                    ::rocprofsys::state::process::Finalized);
                ::std::abort();
            }
            if(!info)
            {
                LOG_CRITICAL("unable to find counter info for counter (id={}) on "
                             "agent (id={})",
                             itr.first.handle, _agent_id.handle);
                ::rocprofsys::state::process::set(
                    ::rocprofsys::state::process::Finalized);
                ::std::abort();
            }

            auto        _dev_id = static_cast<std::uint32_t>(agent->device_id);
            const auto& _agent_mgr_entry =
                get_agent_manager_instance().get_agent_by_handle(_agent_id.handle);
            auto _dev_type_index =
                static_cast<std::uint32_t>(_agent_mgr_entry.device_type_index);

            _agent_counter_storage->at(_agent_id).emplace(
                itr.first,
                counter_storage{ g_tool_data, _dev_id, _dev_type_index, 0, info->name });
        }

        auto const _event =
            counter_event{ counter_dispatch_record{ .dispatch_data  = &dispatch_data,
                                                    .dispatch_id    = _dispatch_id,
                                                    .counter_id     = itr.first,
                                                    .record_counter = itr.second } };

        _agent_counter_storage->at(_agent_id).at(itr.first)(_event, _interval, _scope);
    }
}

void
dispatch_counting_service_callback(
    rocprofiler_dispatch_counting_service_data_t dispatch_data,
    rocprofiler_profile_config_id_t* config, rocprofiler_user_data_t* /*user_data*/,
    void*                            callback_data_arg)
{
    auto* _data = as_client_data(callback_data_arg);
    if(!_data || !config)
    {
        return;
    }

    if(auto const itr =
           _data->agent_counter_profiles.find(dispatch_data.dispatch_info.agent_id);
       itr != _data->agent_counter_profiles.end() && itr->second)
    {
        *config = *itr->second;
    }
}

bool
is_initialized(rocprofiler_context_id_t ctx)
{
    return (ctx.handle > 0);
}

bool
is_active(rocprofiler_context_id_t ctx)
{
    int        status = 0;
    auto const errc   = rocprofiler_context_is_active(ctx, &status);
    return (errc == ROCPROFILER_STATUS_SUCCESS && status > 0);
}

bool
is_valid(rocprofiler_context_id_t ctx)
{
    int        status = 0;
    auto const errc   = rocprofiler_context_is_valid(ctx, &status);
    return (errc == ROCPROFILER_STATUS_SUCCESS && status > 0);
}

void
start_context(rocprofiler_context_id_t ctx)
{
    if(is_initialized(ctx) && !is_active(ctx))
    {
        ROCPROFILER_CALL(rocprofiler_start_context(ctx));
    }
}

void
stop_context(rocprofiler_context_id_t ctx)
{
    if(is_initialized(ctx) && is_active(ctx))
    {
        ROCPROFILER_CALL(rocprofiler_stop_context(ctx));
    }
}

void
start_context(const client_data::context_id_vec_t& ctxs)
{
    std::ranges::for_each(ctxs, [](const auto& ctx) { start_context(ctx); });
}

void
stop_context(const client_data::context_id_vec_t& ctxs)
{
    std::ranges::for_each(ctxs, [](const auto& ctx) { stop_context(ctx); });
}

void
flush()
{
    if(!g_tool_data)
    {
        return;
    }

    for(const auto& itr : g_tool_data->get_buffers())
    {
        if(itr.handle > 0)
        {
            auto const status = rocprofiler_flush_buffer(itr);
            if(status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
            {
                ROCPROFILER_CALL(status);
            }
        }
    }

    if(g_domain_service)
    {
        g_domain_service->flush();
    }
}

// True when tool_init must skip starting the main (primary/counter) contexts,
// leaving them for the "rocm" subscriber's on_resume to start once the session
// goes active.
bool
should_defer_main_contexts()
{
    return !g_session->is_active(control::scope::global);
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* user_data)
{
    // Only initialize once per session
    if(tool_init_done.exchange(true))
    {
        return 0;
    }

    auto const domains = settings::instance()->at(std::string{ env_vars::ROCM_DOMAINS });

    std::stringstream _domains_ss;
    for(const auto& itr : domains->get_choices())
    {
        _domains_ss << "- " << itr << "\n";
    }
    LOG_DEBUG("Available ROCm Domains: \n {}", _domains_ss.str());

    using sdk_backend_t    = backends::rocprofiler_sdk::backend<wrapper>;
    using tracing_config_t = tracing_config<sdk_backend_t, default_externals>;

    sdk_backend_t::check_version_compatibility();

    auto const _callback_domains = tracing_config_t::get_callback_domains();
    auto const _buffered_domain  = tracing_config_t::get_buffered_domains();
    auto const _counter_events   = config::get_rocm_counter_events();
    auto const _version          = tracing_config_t::get_version();
    if(_version.formatted() == 0)
    {
        LOG_WARNING("rocprofiler-sdk version not initialized");
    }

    auto* _data        = as_client_data(user_data);
    _data->client_fini = fini_func;

    _data->initialize();
    if(!_counter_events.empty())
    {
        _data->initialize_event_info();
    }

    ROCPROFILER_CALL(rocprofiler_create_context(&_data->primary_ctx));

    // Code object callback on its own always-on context so kernel names are captured
    // even when primary_ctx is stopped (region filter mode)
    ROCPROFILER_CALL(rocprofiler_create_context(&_data->code_object_ctx));
    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
        _data->code_object_ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
        tool_code_object_callback, _data));

    // Control context for marker-based region filtering and pause/resume (always-on)
    ROCPROFILER_CALL(rocprofiler_create_context(&_data->control_ctx));

    // Insert the default stream and queue info to ensure that the default entry exists
    {
        trace_cache::get_metadata_registry().add_stream(0);
        trace_cache::get_metadata_registry().add_queue(0);
    }

    register_operation_name_strings();

    // MARKER_CORE_API is handled by roctx_client on control_ctx

    g_domain_service =
        std::make_shared<domain_service<production_backend, external_dependencies>>();

    std::vector<domain_selection> domain_selection_list;

    if(_buffered_domain.contains(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH))
    {
        domain_selection selection;
        selection.name = "kernel_dispatch";
        domain_selection_list.push_back(selection);
    }

    if(_buffered_domain.contains(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY))
    {
        domain_selection selection;
        selection.name = "memory_copy";
        domain_selection_list.push_back(selection);
    }

#if(ROCPROFILER_VERSION >= 700)
    if(_buffered_domain.contains(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) ||
       _buffered_domain.contains(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY))
    {
        domain_selection selection;
        selection.name = "hip_stream";
        domain_selection_list.push_back(selection);
    }
#endif

#if(ROCPROFILER_VERSION >= 600)
    if(_buffered_domain.contains(ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION))
    {
        domain_selection selection;
        selection.name = "memory_allocation";
        domain_selection_list.push_back(selection);
    }
#endif

    if(_buffered_domain.contains(ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY))
    {
        domain_selection selection;
        selection.name = "scratch_memory";
        domain_selection_list.push_back(selection);
    }

#if(ROCPROFILER_VERSION >= 10202)
    if(_buffered_domain.contains(ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT))
    {
        domain_selection selection;
        selection.name = "kfd_page_fault";
        domain_selection_list.push_back(selection);
    }

    if(_buffered_domain.contains(ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE))
    {
        domain_selection selection;
        selection.name = "kfd_page_migrate";
        domain_selection_list.push_back(selection);
    }

    if(_buffered_domain.contains(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT))
    {
        domain_selection selection;
        selection.name = "kfd_event_page_fault";
        domain_selection_list.push_back(selection);
    }

    if(_buffered_domain.contains(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE))
    {
        domain_selection selection;
        selection.name = "kfd_event_page_migrate";
        domain_selection_list.push_back(selection);
    }

    if(_buffered_domain.contains(ROCPROFILER_BUFFER_TRACING_KFD_QUEUE))
    {
        domain_selection selection;
        selection.name = "kfd_queue";
        domain_selection_list.push_back(selection);
    }

    if(_buffered_domain.contains(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE))
    {
        domain_selection selection;
        selection.name       = "kfd_event_queue";
        selection.operations = { "ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED" };

        domain_selection_list.push_back(selection);
    }

    if(_buffered_domain.contains(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU))
    {
        domain_selection selection;
        selection.name = "kfd_event_unmap_from_gpu";

        domain_selection_list.push_back(selection);
    }

    if(_buffered_domain.contains(ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS))
    {
        domain_selection selection;
        selection.name = "kfd_event_dropped_events";

        domain_selection_list.push_back(selection);
    }
#endif

    // A std::nullopt tells domain_service::configure() to trace every operation of
    // the domain, so only build an explicit name list when the resolved selection is
    // narrower than the full operation set for the kind.
    auto const get_operation_names = [](sdk_backend_t::callback_tracing_kind_t kind)
        -> std::optional<std::vector<std::string>> {
        auto const selected_operations = tracing_config_t::get_operations(kind);

        std::size_t total_operations = 0;
        for(const auto& entry : sdk_backend_t::get_callback_tracing_names())
        {
            if(static_cast<sdk_backend_t::callback_tracing_kind_t>(entry.value) == kind)
            {
                total_operations = std::ranges::size(entry.operations);
                break;
            }
        }

        if(selected_operations.size() == total_operations)
        {
            return std::nullopt;
        }

        std::vector<std::string> names;
        names.reserve(selected_operations.size());
        for(auto const operation : selected_operations)
        {
            names.emplace_back(
                sdk_backend_t::get_callback_tracing_names().at(kind, operation));
        }
        return names;
    };

    if(_callback_domains.contains(ROCPROFILER_CALLBACK_TRACING_RCCL_API))
    {
        _data->backtrace_operations.emplace(ROCPROFILER_CALLBACK_TRACING_RCCL_API,
                                            tracing_config_t::get_backtrace_operations(
                                                ROCPROFILER_CALLBACK_TRACING_RCCL_API));

        domain_selection selection;
        selection.name       = "rccl_api";
        selection.operations = get_operation_names(ROCPROFILER_CALLBACK_TRACING_RCCL_API);
        domain_selection_list.push_back(selection);
    }

    if(_callback_domains.contains(ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API))
    {
        _data->backtrace_operations.emplace(
            ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API,
            tracing_config_t::get_backtrace_operations(
                ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API));

        domain_selection selection;
        selection.name = "hip_compiler_api";
        selection.operations =
            get_operation_names(ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API);
        domain_selection_list.push_back(selection);
    }

    if(_callback_domains.contains(ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API))
    {
        _data->backtrace_operations.emplace(
            ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
            tracing_config_t::get_backtrace_operations(
                ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API));

        domain_selection selection;
        selection.name = "hip_runtime_api";
        selection.operations =
            get_operation_names(ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API);
        domain_selection_list.push_back(selection);
    }

    if(_callback_domains.contains(ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API))
    {
        _data->backtrace_operations.emplace(
            ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API,
            tracing_config_t::get_backtrace_operations(
                ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API));

        domain_selection selection;
        selection.name = "hsa_core_api";
        selection.operations =
            get_operation_names(ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API);
        domain_selection_list.push_back(selection);
    }

    if(_callback_domains.contains(ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API))
    {
        _data->backtrace_operations.emplace(
            ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
            tracing_config_t::get_backtrace_operations(
                ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API));

        domain_selection selection;
        selection.name = "hsa_amd_ext_api";
        selection.operations =
            get_operation_names(ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API);
        domain_selection_list.push_back(selection);
    }

    if(_callback_domains.contains(ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API))
    {
        _data->backtrace_operations.emplace(
            ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API,
            tracing_config_t::get_backtrace_operations(
                ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API));

        domain_selection selection;
        selection.name = "hsa_image_ext_api";
        selection.operations =
            get_operation_names(ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API);
        domain_selection_list.push_back(selection);
    }

    if(_callback_domains.contains(ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API))
    {
        _data->backtrace_operations.emplace(
            ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
            tracing_config_t::get_backtrace_operations(
                ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API));

        domain_selection selection;
        selection.name = "hsa_finalize_ext_api";
        selection.operations =
            get_operation_names(ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API);
        domain_selection_list.push_back(selection);
    }

#if(ROCPROFILER_VERSION >= 600)
    if(_callback_domains.contains(ROCPROFILER_CALLBACK_TRACING_OMPT))
    {
        _data->backtrace_operations.emplace(ROCPROFILER_CALLBACK_TRACING_OMPT,
                                            tracing_config_t::get_backtrace_operations(
                                                ROCPROFILER_CALLBACK_TRACING_OMPT));

        domain_selection selection;
        selection.name       = "ompt";
        selection.operations = get_operation_names(ROCPROFILER_CALLBACK_TRACING_OMPT);
        domain_selection_list.push_back(selection);
    }
#endif

#if(ROCPROFILER_VERSION >= 600)
    if(_callback_domains.contains(ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API))
    {
        _data->backtrace_operations.emplace(
            ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API,
            tracing_config_t::get_backtrace_operations(
                ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API));

        domain_selection selection;
        selection.name = "rocdecode_api";
        selection.operations =
            get_operation_names(ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API);
        domain_selection_list.push_back(selection);
    }
#endif

#if(ROCPROFILER_VERSION >= 700)
    if(_callback_domains.contains(ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API))
    {
        _data->backtrace_operations.emplace(
            ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API,
            tracing_config_t::get_backtrace_operations(
                ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API));

        domain_selection selection;
        selection.name = "rocjpeg_api";
        selection.operations =
            get_operation_names(ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API);
        domain_selection_list.push_back(selection);
    }
#endif

#if(ROCPROFILER_VERSION >= 10304)
    if(_callback_domains.contains(ROCPROFILER_CALLBACK_TRACING_ROCSHMEM_API))
    {
        _data->backtrace_operations.emplace(
            ROCPROFILER_CALLBACK_TRACING_ROCSHMEM_API,
            tracing_config_t::get_backtrace_operations(
                ROCPROFILER_CALLBACK_TRACING_ROCSHMEM_API));

        domain_selection selection;
        selection.name = "rocshmem_api";
        selection.operations =
            get_operation_names(ROCPROFILER_CALLBACK_TRACING_ROCSHMEM_API);
        domain_selection_list.push_back(selection);
    }
#endif

#if(ROCPROFILER_VERSION >= 10305)
    if(_callback_domains.contains(ROCPROFILER_CALLBACK_TRACING_HIPFILE_API))
    {
        _data->backtrace_operations.emplace(
            ROCPROFILER_CALLBACK_TRACING_HIPFILE_API,
            tracing_config_t::get_backtrace_operations(
                ROCPROFILER_CALLBACK_TRACING_HIPFILE_API));

        domain_selection selection;
        selection.name = "hipfile_api";
        selection.operations =
            get_operation_names(ROCPROFILER_CALLBACK_TRACING_HIPFILE_API);
        domain_selection_list.push_back(selection);
    }
#endif

    try
    {
        if(g_domain_service)
        {
            g_domain_service->configure(domain_selection_list);
        }
    } catch(const std::exception& e)
    {
        LOG_CRITICAL("domain_service::configure failed: {}", e.what());
        throw;
    }

    if(!_counter_events.empty())
    {
        // Resolve counter names to counter IDs per agent
        for(const auto& itr : _data->gpu_agents)
        {
            const auto& _agent_id = rocprofiler_agent_id_t{ itr.agent->handle };
            _data->agent_events.emplace(
                _agent_id, create_agent_profile(_agent_id, _counter_events, _data));
        }

        // --- Dispatch-mode kernel counters ---
        ROCPROFILER_CALL(rocprofiler_create_context(&_data->counter_ctx));

        auto _operations = std::array<rocprofiler_tracing_operation_t, 1>{
            ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
        };

        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
            _data->counter_ctx, ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
            _operations.data(), _operations.size(), tool_tracing_callback, _data));

        ROCPROFILER_CALL(rocprofiler_configure_callback_dispatch_counting_service(
            _data->counter_ctx, dispatch_counting_service_callback, _data,
            counter_record_callback, _data));
    }

#if ROCPROFILER_VERSION >= 600
    const auto gpu_perf_counters_setting = get_gpu_perf_counters();
    if(!gpu_perf_counters_setting.empty() && !_data->gpu_agents.empty())
    {
        pmc::register_gpu_perf_counter_source(
            get_agent_manager_instance().get_agents_by_type(agent_type::gpu));
    }
#endif

    for(const auto& itr : _data->get_buffers())
    {
        if(itr.handle > 0)
        {
            auto client_thread = rocprofiler_callback_thread_t{};
            ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread));
            ROCPROFILER_CALL(rocprofiler_assign_callback_thread(itr, client_thread));
        }
    }

    if(!is_valid(_data->primary_ctx))
    {
        // notify rocprofiler that initialization failed and all the contexts,
        // buffers, etc. created should be ignored
        return -1;
    }

    gpu::add_device_metadata();

    if(config::get_use_process_sampling())
    {
        LOG_DEBUG("Setting PMC sampler state to active...");
        pmc::set_state(state::process::Active);
    }

    assert(g_session);
    create_roctx_client();

    if(g_roctx_client)
    {
        g_roctx_client->configure_services(_data->get_control_context());
    }

    if(should_defer_main_contexts())
    {
        start_context(_data->get_code_obj_context());
        start_context(_data->get_control_context());
    }
    else
    {
        g_domain_service->start();
        start();
    }
    // no errors
    return 0;
}

void
finalize_sdk_common()
{
    if(g_domain_service)
    {
        g_domain_service->finalize();
    }

    flush();
    stop();

    if(get_counter_storage())
    {
        flush_counter_storage_outputs();
        get_counter_storage()->clear();
        delete get_counter_storage();
        get_counter_storage() = nullptr;
    }
}

void
tool_fini(void* callback_data)
{
    if(tool_fini_done.exchange(true))
    {
        return;
    }

    flush();
    stop();
    finalize_sdk_common();

    // Destroy buffers to drain in-flight async flush callbacks before
    // deleting tool_data, which those callbacks may still be accessing.
    // rocprofiler_destroy_buffer returns BUFFER_BUSY if a flush is still
    // in progress, so retry until it succeeds or buffer is not found.
    for(auto const itr : g_tool_data->get_buffers())
    {
        while(itr.handle > 0 &&
              rocprofiler_destroy_buffer(itr) == ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
        {
            std::this_thread::yield();
        }
    }

    auto* _data        = as_client_data(callback_data);
    _data->client_id   = nullptr;
    _data->client_fini = nullptr;
    delete g_tool_data;
    g_tool_data = nullptr;
}

void
flush_counter_tracks_to_zero(rocprofiler_timestamp_t timestamp)
{
    if(timestamp == 0)
    {
        ROCPROFILER_CALL(rocprofiler_get_timestamp(&timestamp));
    }

    auto* storage = get_counter_storage();
    if(!storage)
    {
        return;
    }

    for(auto& [agent_id, counter_map] : *storage)
    {
        for(auto& [counter_id, cs] : counter_map)
        {
            cs.write_zero(timestamp);
        }
    }
}

}  // namespace

void
set_session(std::shared_ptr<control::session> sess)
{
    g_session = std::move(sess);
}

void
create_roctx_client()
{
    if(g_roctx_client || !g_session)
    {
        return;
    }

    const auto domains = rocprofsys::delimit(
        config::get_setting_value<std::string>(std::string{ env_vars::ROCM_DOMAINS })
            .value_or(std::string{}),
        " ,;:\t\n");
    const auto has_marker_domain =
        (std::ranges::find(domains, "marker_api") != domains.end() ||
         std::ranges::find(domains, "roctx") != domains.end());
    const auto roctx_traced_regions = config::get_trace_region();
    const auto has_trace_regions    = !roctx_traced_regions.empty();

    if(!has_marker_domain && !has_trace_regions)
    {
        return;
    }

    const auto roctx_config = roctx_client_config{
        .pause_resume_enabled   = has_marker_domain,
        .use_perfetto           = config::get_use_perfetto(),
        .use_timemory           = config::get_use_timemory(),
        .perfetto_annotations   = config::get_perfetto_annotations(),
        .selected_trace_regions = roctx_traced_regions,
    };
    g_roctx_client = std::make_shared<roctx_client<>>(g_session, roctx_config);
}

void
reset_sdk_session_guards()
{
    tool_fini_done.store(false);
    tool_init_done.store(false);
}

void
setup()
{}

void
shutdown()
{
    if(g_tool_data && g_tool_data->client_id && g_tool_data->client_fini)
    {
        g_tool_data->client_fini(*g_tool_data->client_id);
    }

    g_roctx_client.reset();
}

void
config()
{}

void
post_process()
{}

void
sample()
{}

void
start()
{
    if(!g_tool_data)
    {
        return;
    }

    start_context(g_tool_data->get_all_contexts());
}

void
stop()
{
    if(!g_tool_data)
    {
        return;
    }

    stop_context(g_tool_data->get_all_contexts());
}

void
resume()
{
    flush_counter_tracks_to_zero(0);

    if(!g_tool_data)
    {
        return;
    }
    start_context(g_tool_data->get_main_contexts());

    if(g_domain_service != nullptr)
    {
        g_domain_service->start();
    }
}

void
pause()
{
    if(!g_tool_data)
    {
        return;
    }
    stop_context(g_tool_data->get_main_contexts());

    if(g_domain_service != nullptr)
    {
        g_domain_service->pause();
    }

    flush_counter_tracks_to_zero(0);
}

std::vector<hardware_counter_info>
get_rocm_events_info()
{
    if(!g_tool_data)
    {
        auto tool_data_v = client_data{};
        tool_data_v.initialize_event_info();
        return tool_data_v.events_info;
    }

    if(g_tool_data->events_info.empty())
    {
        g_tool_data->initialize_event_info();
    }

    return g_tool_data->events_info;
}

#if ROCPROFILER_VERSION >= 10200

void
tool_attach_fini(void* /* tool_data */)
{
    // Stop and flush SDK contexts/buffers so that buffer callbacks
    // write their Perfetto events before Perfetto post-processing.
    ::rocprofsys::rocprofiler_sdk::stop();
    ::rocprofsys::rocprofiler_sdk::flush();
    finalize_sdk_common();

    // Flush any pending region cache entries
    rocprofsys_flush_pending_region_cache_hidden();

    // Write Perfetto trace output
    if(get_use_perfetto())
    {
        bool                             _perfetto_output_error = false;
        rocprofsys::output_file_registry _output_registry{};
        ::rocprofsys::perfetto::post_process(nullptr, _perfetto_output_error,
                                             _output_registry);
        if(_perfetto_output_error)
        {
            LOG_ERROR("Perfetto output error occurred during attach finalization");
        }
    }

    rocprofsys_finalize_hidden();
}

int
// NOLINTNEXTLINE(misc-const-correctness) - signature must match rocprofiler_tool_attach_t
// exactly
tool_attach_init([[maybe_unused]] rocprofiler_client_detach_t detach_func,
                 rocprofiler_context_id_t* context_ids, std::uint64_t context_ids_length,
                 [[maybe_unused]] void* tool_attach_data)
{
    static std::atomic<int> attach_count{ 0 };
    auto                    current_count = attach_count.fetch_add(1);

    if(current_count > 0)
    {
        LOG_INFO("Re-attaching to process {} (session {})", getpid(), current_count);
        rocprofsys_reset_for_reattach_hidden();
        reset_sdk_session_guards();

        // Restart Perfetto for a new tracing session
        if(get_use_perfetto())
        {
            ::rocprofsys::perfetto::start();
        }

        trace_cache::get_buffer_storage().start(getpid());

        // Restart process sampler (AMD SMI, CPU freq polling thread)
        if(config::get_use_process_sampling())
        {
            ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
            ::rocprofsys::process_sampler::setup();
        }

        ::rocprofsys::state::process::set(::rocprofsys::state::process::Active);
    }

    // Start all contexts provided by the SDK
    for(std::uint64_t i = 0; i < context_ids_length; ++i)
    {
        ROCPROFILER_CALL(rocprofiler_start_context(context_ids[i]));
    }

    ::rocprofsys::rocprofiler_sdk::start();

    return 0;
}
#endif

}  // namespace rocprofsys::rocprofiler_sdk

namespace
{
std::atomic<bool> sdk_configured{ false };

/**
 * Initialize rocprofiler-sdk tool configuration.
 *
 * Performs common setup for both standard and attach configure paths:
 * initializes tooling, validates ROCm usage, sets up client data,
 * logs version info, and registers thread callbacks.
 *
 * @param version The rocprofiler-sdk version number.
 * @param runtime_version The runtime version string.
 * @param id The client identifier to initialize.
 * @return true if initialization succeeded, false otherwise.
 */
bool
sdk_tool_configure(std::uint32_t version, const char* runtime_version,
                   rocprofiler_client_id_t* id)
{
    // Only configure once per attach session
    if(sdk_configured.exchange(true))
    {
        return true;
    }

    // Ensure tooling is initialized and state is Active
    if(!rocprofsys::config::settings_are_configured() ||
       rocprofsys::state::process::get() < rocprofsys::state::process::Active)
    {
        rocprofsys_init_tooling_hidden();
    }

    // set the client name
    id->name = "rocprofsys";

    if(!rocprofsys::rocprofiler_sdk::g_tool_data)
    {
        rocprofsys::rocprofiler_sdk::g_tool_data =
            new rocprofsys::rocprofiler_sdk::client_data{};
    }

    // store client info
    rocprofsys::rocprofiler_sdk::g_tool_data->client_id = id;

    // compute major/minor/patch version info
    std::uint32_t major = version / 10000;
    std::uint32_t minor = (version % 10000) / 100;
    std::uint32_t patch = version % 100;

    LOG_INFO("{} is using rocprofiler-sdk v{}.{}.{} ({})", id->name, major, minor, patch,
             runtime_version);

    ROCPROFILER_CALL(rocprofiler_at_internal_thread_create(
        rocprofsys::rocprofiler_sdk::thread_precreate,
        rocprofsys::rocprofiler_sdk::thread_postcreate,
        ROCPROFILER_LIBRARY | ROCPROFILER_HSA_LIBRARY | ROCPROFILER_HIP_LIBRARY |
            ROCPROFILER_MARKER_LIBRARY,
        nullptr));

    return true;
}

}  // namespace

extern "C"
{
    rocprofiler_tool_configure_result_t* rocprofiler_configure(
        std::uint32_t version, const char* runtime_version,
        [[maybe_unused]] std::uint32_t priority, rocprofiler_client_id_t* id)
    {
        // only activate once
        {
            static std::atomic<bool> _first{ true };
            if(!_first.exchange(false))
            {
                return nullptr;
            }
        }

        if(!rocprofsys::get_env(rocprofsys::env_vars::INIT_TOOLING, true))
        {
            return nullptr;
        }
        if(!tim::settings::enabled())
        {
            return nullptr;
        }

        if(!sdk_tool_configure(version, runtime_version, id))
        {
            return nullptr;
        }

        static auto cfg = rocprofiler_tool_configure_result_t{
            .size       = sizeof(rocprofiler_tool_configure_result_t),
            .initialize = &::rocprofsys::rocprofiler_sdk::tool_init,
            .finalize   = &::rocprofsys::rocprofiler_sdk::tool_fini,
            .tool_data  = rocprofsys::rocprofiler_sdk::g_tool_data
        };
        return &cfg;
    }

#if ROCPROFILER_VERSION >= 10200

    rocprofiler_tool_configure_attach_result_t* rocprofiler_configure_attach(
        std::uint32_t version, const char* runtime_version,
        [[maybe_unused]] std::uint32_t priority, rocprofiler_client_id_t* id)
    {
        if(!sdk_tool_configure(version, runtime_version, id))
        {
            return nullptr;
        }

        static auto cfg = rocprofiler_tool_configure_attach_result_t{
            .size        = sizeof(rocprofiler_tool_configure_attach_result_t),
            .tool_attach = &rocprofsys::rocprofiler_sdk::tool_attach_init,
            .tool_detach = &rocprofsys::rocprofiler_sdk::tool_attach_fini,
            .tool_data   = rocprofsys::rocprofiler_sdk::g_tool_data
        };
        return &cfg;
    }
#endif  // ROCPROFILER_VERSION >= 10200
}
