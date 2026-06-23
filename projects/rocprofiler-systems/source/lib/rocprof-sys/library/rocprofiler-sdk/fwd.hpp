// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/wrapper.hpp"
#include "common/synchronized.hpp"
#include "core/agent_manager.hpp"
#include "core/perfetto.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"
#include "logger/debug.hpp"

#include <spdlog/fmt/ranges.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <source_location>
#include <utility>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{

/// Log-and-continue helper for SDK calls not yet migrated to sdk_check<Wrapper>.
/// Uses Wrapper::STATUS_SUCCESS and Wrapper::get_status_string — no raw SDK symbols.
template <typename Wrapper>
inline void
rocprofiler_call(typename Wrapper::status_t status, const char* expr,
                 std::source_location loc = std::source_location::current())
{
    if(status != Wrapper::STATUS_SUCCESS)
    {
        LOG_WARNING(
            "[{}][{}:{}] rocprofiler-sdk call [{}] failed with error code {} :: {}", expr,
            loc.file_name(), loc.line(), expr, static_cast<int>(status),
            Wrapper::get_status_string(status));
    }
}

using hardware_counter_info = ::tim::hardware_counters::info;
using callback_arg_array_t  = std::vector<std::pair<std::string, std::string>>;
using rocprofsys_agent_t    = agent;

// ─── Template type aliases ────────────────────────────────────────────────────

template <typename Wrapper>
using kernel_symbol_data_t = typename Wrapper::kernel_symbol_data;

template <typename Wrapper>
using kernel_symbol_map_t =
    std::unordered_map<typename Wrapper::kernel_id_t, kernel_symbol_data_t<Wrapper>>;

// ─── code_object_callback_record_t<Wrapper> ──────────────────────────────────

template <typename Wrapper>
struct code_object_callback_record_t
{
    std::uint64_t                             timestamp = 0;
    typename Wrapper::callback_tracing_record record    = {};
    typename Wrapper::code_object_load_data   payload   = {};
};

// ─── kernel_symbol_callback_record_t<Wrapper> ────────────────────────────────

template <typename Wrapper>
struct kernel_symbol_callback_record_t
{
    std::uint64_t                             timestamp = 0;
    typename Wrapper::callback_tracing_record record    = {};
    kernel_symbol_data_t<Wrapper>             payload   = {};
};

// ─── timing_interval<Wrapper> ────────────────────────────────────────────────

template <typename Wrapper>
struct timing_interval
{
    typename Wrapper::timestamp_t start = 0;
    typename Wrapper::timestamp_t end   = 0;
};

// ─── tool_counter_info_t<Wrapper> ────────────────────────────────────────────

template <typename Wrapper>
struct tool_counter_info_t : Wrapper::counter_info_v0_t
{
    using this_type            = tool_counter_info_t<Wrapper>;
    using parent_type          = typename Wrapper::counter_info_v0_t;
    using dimension_info_vec_t = std::vector<typename Wrapper::dimension_info_t>;

    tool_counter_info_t(typename Wrapper::agent_id _agent_id, parent_type _info,
                        dimension_info_vec_t&& _dim_info)
    : parent_type{ _info }
    , agent_id{ _agent_id }
    , dimension_info{ std::move(_dim_info) }
    {}

    tool_counter_info_t()                                          = default;
    ~tool_counter_info_t()                                         = default;
    tool_counter_info_t(const tool_counter_info_t&)                = default;
    tool_counter_info_t(tool_counter_info_t&&) noexcept            = default;
    tool_counter_info_t& operator=(const tool_counter_info_t&)     = default;
    tool_counter_info_t& operator=(tool_counter_info_t&&) noexcept = default;

    typename Wrapper::agent_id                      agent_id       = {};
    std::vector<typename Wrapper::dimension_info_t> dimension_info = {};
};

struct tool_agent
{
    std::uint64_t             device_id = 0;
    const rocprofsys_agent_t* agent     = nullptr;
};

// ─── Aggregated map aliases ───────────────────────────────────────────────────

template <typename Wrapper>
using agent_counter_info_map_t =
    std::unordered_map<typename Wrapper::agent_id,
                       std::vector<tool_counter_info_t<Wrapper>>>;

template <typename Wrapper>
using agent_counter_profile_map_t =
    std::unordered_map<typename Wrapper::agent_id,
                       std::optional<typename Wrapper::counter_config_id>>;

template <typename Wrapper>
using counter_id_vec_t = std::vector<typename Wrapper::counter_id>;

template <typename Wrapper>
using agent_counter_id_map_t =
    std::unordered_map<typename Wrapper::agent_id, counter_id_vec_t<Wrapper>>;

template <typename Wrapper>
using backtrace_operation_map_t =
    std::unordered_map<typename Wrapper::callback_tracing_kind,
                       std::unordered_set<typename Wrapper::tracing_operation>>;

// ─── client_data<Wrapper> ────────────────────────────────────────────────────

template <typename Wrapper>
struct client_data
{
    static constexpr size_t num_buffers  = 11;
    static constexpr size_t num_contexts = 4;

    using buffer_name_info_t   = typename Wrapper::buffer_name_info_t;
    using callback_name_info_t = typename Wrapper::callback_name_info_t;
    using kernel_symbol_vec_t  = std::vector<kernel_symbol_callback_record_t<Wrapper>>;
    using code_object_vec_t    = std::vector<code_object_callback_record_t<Wrapper>>;
    using buffer_id_vec_t      = std::array<typename Wrapper::buffer_id, num_buffers>;
    using context_id_vec_t     = std::array<typename Wrapper::context_id, num_contexts>;
    using agent_vec_t          = std::vector<typename Wrapper::agent_t>;

    typename Wrapper::client_id_t*            client_id                 = nullptr;
    typename Wrapper::client_finalize_t       client_fini               = nullptr;
    typename Wrapper::context_id              primary_ctx               = { 0 };
    typename Wrapper::context_id              counter_ctx               = { 0 };
    typename Wrapper::context_id              code_object_ctx           = { 0 };
    typename Wrapper::context_id              control_ctx               = { 0 };
    typename Wrapper::buffer_id               kernel_dispatch_buffer    = { 0 };
    typename Wrapper::buffer_id               scratch_memory_buffer     = { 0 };
    typename Wrapper::buffer_id               memory_copy_buffer        = { 0 };
    typename Wrapper::buffer_id               memory_alloc_buffer       = { 0 };
    typename Wrapper::buffer_id               counter_collection_buffer = { 0 };
    typename Wrapper::buffer_id               kfd_page_fault_buffer     = { 0 };
    typename Wrapper::buffer_id               kfd_page_migrate_buffer   = { 0 };
    typename Wrapper::buffer_id               kfd_queue_buffer          = { 0 };
    typename Wrapper::buffer_id               kfd_event_queue_buffer    = { 0 };
    typename Wrapper::buffer_id               kfd_event_unmap_buffer    = { 0 };
    typename Wrapper::buffer_id               kfd_event_dropped_buffer  = { 0 };
    std::vector<tool_agent>                   cpu_agents                = {};
    std::vector<tool_agent>                   gpu_agents                = {};
    std::vector<hardware_counter_info>        events_info               = {};
    agent_counter_id_map_t<Wrapper>           agent_events              = {};
    agent_counter_info_map_t<Wrapper>         agent_counter_info        = {};
    agent_counter_profile_map_t<Wrapper>      agent_counter_profiles    = {};
    common::synchronized<code_object_vec_t>   code_object_records       = {};
    common::synchronized<kernel_symbol_vec_t> kernel_symbol_records     = {};
    buffer_name_info_t                        buffered_tracing_info     = {};
    callback_name_info_t                      callback_tracing_info     = {};
    backtrace_operation_map_t<Wrapper>        backtrace_operations      = {};

    void initialize();
    void initialize_event_info();
    void set_agents();

    context_id_vec_t             get_all_contexts() const;
    context_id_vec_t             get_main_contexts() const;
    typename Wrapper::context_id get_control_context() const;
    typename Wrapper::context_id get_code_obj_context() const;
    buffer_id_vec_t              get_buffers() const;

    const rocprofsys_agent_t* get_agent(typename Wrapper::agent_id _id) const;
    const tool_agent*         get_gpu_tool_agent(typename Wrapper::agent_id id) const;

    const kernel_symbol_data_t<Wrapper>* get_kernel_symbol_info(
        std::uint64_t _kernel_id) const;

    const tool_counter_info_t<Wrapper>* get_tool_counter_info(
        typename Wrapper::agent_id   _agent_id,
        typename Wrapper::counter_id _counter_id) const;

    const typename Wrapper::code_object_load_data* get_code_object_info(
        std::uint64_t code_object_id) const;

private:
    // ─── Counter query callbacks ──────────────────────────────────────────────

    static typename Wrapper::status_t dimensions_info_callback(
        typename Wrapper::counter_id /*id*/,
        const typename Wrapper::dimension_info_t* dim_info, long unsigned int num_dims,
        void* user_data)
    {
        auto* dimensions_info =
            static_cast<std::vector<typename Wrapper::dimension_info_t>*>(user_data);
        dimensions_info->reserve(num_dims);
        for(size_t j = 0; j < num_dims; j++)
            dimensions_info->emplace_back(dim_info[j]);
        return Wrapper::STATUS_SUCCESS;
    }

    static typename Wrapper::status_t counters_supported_callback(
        typename Wrapper::agent_id agent_id, typename Wrapper::counter_id* counters,
        size_t num_counters, void* user_data)
    {
        using value_type = typename agent_counter_info_map_t<Wrapper>::mapped_type;

        auto* data_v = static_cast<agent_counter_info_map_t<Wrapper>*>(user_data);
        data_v->emplace(agent_id, value_type{});
        for(size_t i = 0; i < num_counters; ++i)
        {
            auto _info     = typename Wrapper::counter_info_v0_t{};
            auto _dim_info = std::vector<typename Wrapper::dimension_info_t>{};

            rocprofiler_call<Wrapper>(
                Wrapper::query_counter_info(counters[i], Wrapper::COUNTER_INFO_VERSION_0,
                                            &_info),
                "Wrapper::query_counter_info");
            rocprofiler_call<Wrapper>(
                Wrapper::iterate_counter_dimensions(counters[i], dimensions_info_callback,
                                                    &_dim_info),
                "Wrapper::iterate_counter_dimensions");

            if(!_info.is_constant)
                data_v->at(agent_id).emplace_back(agent_id, _info, std::move(_dim_info));
        }
        return Wrapper::STATUS_SUCCESS;
    }

    static agent_counter_info_map_t<Wrapper> get_agent_counter_info(
        const std::vector<tool_agent>& _agents)
    {
        auto _data = agent_counter_info_map_t<Wrapper>{};
        for(const auto& itr : _agents)
        {
            const auto& _agent_id = typename Wrapper::agent_id{ itr.agent->handle };

            auto status = Wrapper::iterate_agent_supported_counters(
                _agent_id, counters_supported_callback, &_data);

            if(status != Wrapper::STATUS_SUCCESS)
            {
                LOG_WARNING("iterate_agent_supported_counters failed for agent {} "
                            "with status {} (Agent HW architecture may not be supported)",
                            _agent_id.handle, static_cast<int>(status));
                continue;
            }

            auto agent_it = _data.find(_agent_id);
            if(agent_it != _data.end())
            {
                std::sort(agent_it->second.begin(), agent_it->second.end(),
                          [](const auto& lhs, const auto& rhs) {
                              return (lhs.id.handle < rhs.id.handle);
                          });
                for(auto& citr : agent_it->second)
                {
                    std::sort(citr.dimension_info.begin(), citr.dimension_info.end(),
                              [](const auto& lhs, const auto& rhs) {
                                  return (lhs.id < rhs.id);
                              });
                }
            }
        }
        return _data;
    }
};

// ─── client_data method implementations ──────────────────────────────────────

template <typename Wrapper>
typename client_data<Wrapper>::context_id_vec_t
client_data<Wrapper>::get_all_contexts() const
{
    return context_id_vec_t{ primary_ctx, counter_ctx, code_object_ctx, control_ctx };
}

template <typename Wrapper>
typename client_data<Wrapper>::context_id_vec_t
client_data<Wrapper>::get_main_contexts() const
{
    return context_id_vec_t{ primary_ctx, counter_ctx };
}

template <typename Wrapper>
typename Wrapper::context_id
client_data<Wrapper>::get_control_context() const
{
    return control_ctx;
}

template <typename Wrapper>
typename Wrapper::context_id
client_data<Wrapper>::get_code_obj_context() const
{
    return code_object_ctx;
}

template <typename Wrapper>
typename client_data<Wrapper>::buffer_id_vec_t
client_data<Wrapper>::get_buffers() const
{
    return buffer_id_vec_t{ kernel_dispatch_buffer,    scratch_memory_buffer,
                            memory_copy_buffer,        memory_alloc_buffer,
                            counter_collection_buffer, kfd_page_fault_buffer,
                            kfd_page_migrate_buffer,   kfd_queue_buffer,
                            kfd_event_queue_buffer,    kfd_event_unmap_buffer,
                            kfd_event_dropped_buffer };
}

template <typename Wrapper>
const rocprofsys_agent_t*
client_data<Wrapper>::get_agent(typename Wrapper::agent_id _id) const
{
    const auto& agent_ref = get_agent_manager_instance().get_agent_by_handle(_id.handle);
    return &agent_ref;
}

template <typename Wrapper>
const tool_agent*
client_data<Wrapper>::get_gpu_tool_agent(typename Wrapper::agent_id id) const
{
    for(const auto& itr : gpu_agents)
        if(id.handle == itr.agent->handle) return &itr;
    return nullptr;
}

template <typename Wrapper>
const kernel_symbol_data_t<Wrapper>*
client_data<Wrapper>::get_kernel_symbol_info(std::uint64_t _kernel_id) const
{
    return kernel_symbol_records.rlock(
        [_kernel_id](const auto& _data) -> const kernel_symbol_data_t<Wrapper>* {
            for(const auto& itr : _data)
            {
                if(_kernel_id == itr.payload.kernel_id)
                {
                    return &itr.payload;
                    break;
                }
            }
            return nullptr;
        });
}

template <typename Wrapper>
const tool_counter_info_t<Wrapper>*
client_data<Wrapper>::get_tool_counter_info(
    typename Wrapper::agent_id _agent_id, typename Wrapper::counter_id _counter_id) const
{
    for(const auto& itr : agent_counter_info.at(_agent_id))
    {
        if(itr.id == _counter_id) return &itr;
    }
    return nullptr;
}

template <typename Wrapper>
const typename Wrapper::code_object_load_data*
client_data<Wrapper>::get_code_object_info(std::uint64_t code_object_id) const
{
    using load_data_t = typename Wrapper::code_object_load_data;
    return code_object_records.rlock(
        [code_object_id](const auto& _data) -> const load_data_t* {
            for(const auto& itr : _data)
            {
                if(code_object_id == itr.payload.code_object_id)
                {
                    return &itr.payload;
                    break;
                }
            }
            return nullptr;
        });
}

template <typename Wrapper>
void
client_data<Wrapper>::initialize()
{
    buffered_tracing_info = Wrapper::get_buffer_tracing_names();
    callback_tracing_info = Wrapper::get_callback_tracing_names();

    set_agents();
}

template <typename Wrapper>
void
client_data<Wrapper>::initialize_event_info()
{
    auto& agent_mngr = get_agent_manager_instance();

    if(agent_mngr.get_agents().empty())
    {
        initialize();
    }
    else if(gpu_agents.empty() && cpu_agents.empty())
    {
        set_agents();
    }

    if(agent_counter_info.size() != gpu_agents.size())
        agent_counter_info = get_agent_counter_info(gpu_agents);

    try
    {
        using qualifier_t     = tim::hardware_counters::qualifier;
        using qualifier_vec_t = std::vector<qualifier_t>;

        for(const auto& aitr : gpu_agents)
        {
            auto        _dev_index = aitr.device_id;
            const auto& _agent_id  = typename Wrapper::agent_id{ aitr.agent->handle };
            auto        _device_qualifier_sym = fmt::format(":device={}", _dev_index);
            auto        _device_qualifier =
                tim::hardware_counters::qualifier{ true, static_cast<int>(_dev_index),
                                                   _device_qualifier_sym,
                                                   fmt::format("Device {}", _dev_index) };

            // Check if agent info is available ( i.e., counters are supported)
            auto agent_info_it = agent_counter_info.find(_agent_id);
            if(agent_info_it == agent_counter_info.end())
            {
                LOG_WARNING("Skipping GPU device {} ({}, handle=0x{:X}) due to "
                            "counter not found for the specified architecture",
                            _dev_index, aitr.agent->name, aitr.agent->handle);
                continue;
            }

            auto _counter_info = agent_info_it->second;
            std::sort(_counter_info.begin(), _counter_info.end(),
                      [](const tool_counter_info_t<Wrapper>& lhs,
                         const tool_counter_info_t<Wrapper>& rhs) {
                          if(lhs.is_constant && rhs.is_constant)
                              return lhs.id < rhs.id;
                          else if(lhs.is_constant)
                              return true;
                          else if(rhs.is_constant)
                              return false;

                          if(!lhs.is_derived && !rhs.is_derived)
                              return lhs.id < rhs.id;
                          else if(!lhs.is_derived)
                              return true;
                          else if(!rhs.is_derived)
                              return false;

                          return lhs.id < rhs.id;
                      });

            for(const auto& ditr : _counter_info)
            {
                auto _long_desc = std::string{ ditr.description };
                auto _units     = std::string{};
                auto _pysym     = std::string{};

                if(ditr.is_constant)
                {
                    continue;
                }
                else if(ditr.is_derived)
                {
                    auto _sym = fmt::format("{}:device={}", ditr.name, _dev_index);
                    auto _short_desc =
                        fmt::format("Derived counter: {}", ditr.expression);
                    events_info.emplace_back(hardware_counter_info(
                        true, tim::hardware_counters::api::rocm, events_info.size(), 0,
                        _sym, _pysym, _short_desc, _long_desc, _units,
                        qualifier_vec_t{ _device_qualifier }));
                }
                else
                {
                    auto _dim_info = std::vector<std::string>{};

                    for(const auto& itr : ditr.dimension_info)
                    {
                        auto _info =
                            (itr.instance_size > 1)
                                ? fmt::format("{}[0:{}]", itr.name, itr.instance_size - 1)
                                : std::string{};
                        if(!_info.empty()) _dim_info.emplace_back(_info);
                    }

                    auto _sym = fmt::format("{}:device={}", ditr.name, _dev_index);
                    auto _short_desc =
                        fmt::format("{} on device {}", ditr.name, _dev_index);
                    if(!_dim_info.empty())
                    {
                        _short_desc += fmt::format("{}", fmt::join(_dim_info, ". "));
                    }
                    events_info.emplace_back(hardware_counter_info(
                        true, tim::hardware_counters::api::rocm, events_info.size(), 0,
                        _sym, _pysym, _short_desc, _long_desc, _units,
                        qualifier_vec_t{ _device_qualifier }));
                }
            }
        }
    } catch(std::exception& _e)
    {
        LOG_WARNING("Constructing ROCm event info failed: {}", _e.what());
    }
}

template <typename Wrapper>
void
client_data<Wrapper>::set_agents()
{
    auto& agent_mngr = get_agent_manager_instance();

    auto fill_agents = [&](agent_type type, std::vector<tool_agent>& out) {
        const auto& _agents = agent_mngr.get_agents_by_type(type);
        for(const auto& agent_ptr : _agents)
        {
            out.emplace_back(tool_agent{ agent_ptr->device_type_index, agent_ptr.get() });
        }
    };

    fill_agents(agent_type::GPU, gpu_agents);
    fill_agents(agent_type::CPU, cpu_agents);
}

// ─── as_client_data helper ────────────────────────────────────────────────────

template <typename Wrapper>
constexpr client_data<Wrapper>*
as_client_data(void* _ptr)
{
    return static_cast<client_data<Wrapper>*>(_ptr);
}

// ─── Production type aliases (for non-generic subdirectory code) ──────────────
// kfd_events, counters, rccl etc. are not templated; they use these fixed aliases.

using client_data_t     = client_data<backend>;
using timing_interval_t = timing_interval<backend>;

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
