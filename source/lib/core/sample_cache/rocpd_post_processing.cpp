// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "sample_cache/rocpd_post_processing.hpp"
#include "agent_manager.hpp"
#include "common.hpp"
#include "config.hpp"
#include "library/thread_info.hpp"
#include "node_info.hpp"
#include "rocpd/data_processor.hpp"
#include "sample_cache/cache_storage_parser.hpp"
#include "sample_cache/metadata_storage.hpp"
#include "sample_cache/sample_type.hpp"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <timemory/utility/demangle.hpp>
#if ROCPROFSYS_USE_ROCM > 0
#    include "library/rocprofiler-sdk/fwd.hpp"
#    include <rocprofiler-sdk/context.h>
#    include <rocprofiler-sdk/version.h>
#endif

namespace rocprofsys
{
namespace sample_cache
{
namespace
{
rocpd::data_processor&
get_data_processor()
{
    return rocpd::data_processor::get_instance();
}

#if ROCPROFSYS_USE_ROCM > 0
auto
get_handle_from_code_object(
    const rocprofiler_callback_tracing_code_object_load_data_t& code_object)
{
#    if(ROCPROFILER_VERSION >= 600)
    return code_object.agent_id.handle;
#    else
    return code_object.rocp_agent.handle;
#    endif
}
#endif
}  // namespace

postprocessing_callback
rocpd_post_processing::get_kernel_dispatch_callback() const
{
    return [&]([[maybe_unused]] const storage_parsed_type_base& parsed) {
#if ROCPROFSYS_USE_ROCM > 0
        auto _kds = static_cast<const struct kernel_dispatch_sample&>(parsed);

        auto& data_processor = get_data_processor();
        auto& agent_manager  = agent_manager::get_instance();
        auto& n_info         = node_info::get_instance();
        auto  process        = m_metadata.get_process_info();
        auto  agent_primary_key =
            agent_manager.get_agent_by_handle(_kds.agent_id_handle).base_id;

        auto thread_primary_key =
            data_processor.map_thread_id_to_primary_key(_kds.thread_id);

        auto category_id = data_processor.insert_string(
            trait::name<category::rocm_kernel_dispatch>::value);

        auto kernel_symbol = m_metadata.get_kernel_symbol(_kds.kernel_id);

        if(!kernel_symbol.has_value())
        {
            throw std::runtime_error("Kernel symbol is missing for kernel dispatch");
            return;
        }

        auto region_name_primary_key = data_processor.insert_string(
            tim::demangle(kernel_symbol->kernel_name).c_str());

        auto stack_id        = _kds.correlation_id_internal;
        auto parent_stack_id = _kds.correlation_id_ancestor;
        auto correlation_id  = 0;

        auto event_id = data_processor.insert_event(category_id, stack_id,
                                                    parent_stack_id, correlation_id);

        data_processor.insert_kernel_dispatch(
            n_info.id, process.pid, thread_primary_key, agent_primary_key, _kds.kernel_id,
            _kds.dispatch_id, _kds.queue_id_handle, _kds.stream_handle,
            _kds.start_timestamp, _kds.end_timestamp, _kds.private_segment_size,
            _kds.group_segment_size, _kds.workgroup_size_x, _kds.workgroup_size_y,
            _kds.workgroup_size_z, _kds.grid_size_x, _kds.grid_size_y, _kds.grid_size_z,
            region_name_primary_key, event_id);
#endif
    };
}

postprocessing_callback
rocpd_post_processing::get_memory_copy_callback() const
{
    return [&]([[maybe_unused]] const storage_parsed_type_base& parsed) {
#if ROCPROFSYS_USE_ROCM > 0
        auto _mcs = static_cast<const struct memory_copy_sample&>(parsed);

        auto& data_processor = get_data_processor();
        auto& agent_manager  = agent_manager::get_instance();
        auto& n_info         = node_info::get_instance();
        auto  process        = m_metadata.get_process_info();

        auto _name            = std::string{ m_metadata.get_buffer_name_info().at(
            static_cast<rocprofiler_buffer_tracing_kind_t>(_mcs.kind),
            static_cast<rocprofiler_tracing_operation_t>(_mcs.operation)) };
        auto name_primary_key = data_processor.insert_string(_name.c_str());

        auto category_primary_key =
            data_processor.insert_string(trait::name<category::rocm_memory_copy>::value);

        auto thread_primary_key =
            data_processor.map_thread_id_to_primary_key(_mcs.thread_id);

        auto dst_agent_primary_key =
            agent_manager.get_agent_by_handle(_mcs.dst_agent_id_handle).base_id;
        auto src_agent_primary_key =
            agent_manager.get_agent_by_handle(_mcs.src_agent_id_handle).base_id;

        auto stack_id        = _mcs.correlation_id_internal;
        auto parent_stack_id = _mcs.correlation_id_ancestor;
        auto correlation_id  = 0;
        auto queue_id        = 0;

        auto event_primary_key = data_processor.insert_event(
            category_primary_key, stack_id, parent_stack_id, correlation_id);

        data_processor.insert_memory_copy(
            n_info.id, process.pid, thread_primary_key, _mcs.start_timestamp,
            _mcs.end_timestamp, name_primary_key, dst_agent_primary_key,
            _mcs.dst_address_value, src_agent_primary_key, _mcs.src_address_value,
            _mcs.bytes, queue_id, _mcs.stream_handle, name_primary_key,
            event_primary_key);
#endif
    };
}

#if(ROCPROFSYS_USE_ROCM > 0 && ROCPROFILER_VERSION >= 600)
postprocessing_callback
rocpd_post_processing::get_memory_allocate_callback() const
{
#    if ROCPROFSYS_USE_ROCM > 0
    auto memtype_to_db =
        [](std::string_view memory_type) -> std::pair<std::string, std::string> {
        constexpr auto MEMORY_PREFIX  = std::string_view{ "MEMORY_ALLOCATION_" };
        constexpr auto SCRATCH_PREFIX = std::string_view{ "SCRATCH_MEMORY_" };
        constexpr auto VMEM_PREFIX    = std::string_view{ "VMEM_" };
        constexpr auto ASYNC_PREFIX   = std::string_view{ "ASYNC_" };

        std::string _type;
        std::string _level;
        if(memory_type.find(MEMORY_PREFIX) == 0)
        {
            _type = memory_type.substr(MEMORY_PREFIX.length());
            if(_type.find(VMEM_PREFIX) == 0)
            {
                _type  = _type.substr(VMEM_PREFIX.length());
                _level = "VIRTUAL";
            }
            else
            {
                _level = "REAL";
            }
        }
        else if(memory_type.find(SCRATCH_PREFIX) == 0)
        {
            _type  = memory_type.substr(SCRATCH_PREFIX.length());
            _level = "SCRATCH";
            if(memory_type.find(ASYNC_PREFIX) == 0)
            {
                _type = memory_type.substr(ASYNC_PREFIX.length());  // RECLAIM
            }
        }

        if(_type == "ALLOCATE")
        {
            _type = "ALLOC";
        }

        return std::make_pair(_type, _level);
    };
#    endif

    return [&]([[maybe_unused]] const storage_parsed_type_base& parsed) {
#    if ROCPROFSYS_USE_ROCM > 0
        auto  _mas           = static_cast<const struct memory_allocate_sample&>(parsed);
        auto& data_processor = get_data_processor();
        auto& agent_manager  = agent_manager::get_instance();
        auto& n_info         = node_info::get_instance();
        auto  process        = m_metadata.get_process_info();
        auto  thread_primary_key =
            data_processor.map_thread_id_to_primary_key(_mas.thread_id);
        auto agent_primary_key = std::optional<uint64_t>{};

        const auto invalid_context = ROCPROFILER_CONTEXT_NONE;
        if(_mas.agent_id_handle != invalid_context.handle)
        {
            {
                agent_primary_key =
                    agent_manager.get_agent_by_handle(_mas.agent_id_handle).base_id;
            }
            const auto* _name = m_metadata.get_buffer_name_info().at(
                static_cast<rocprofiler_buffer_tracing_kind_t>(_mas.kind),
                static_cast<rocprofiler_tracing_operation_t>(_mas.operation));

            auto [type, level] = memtype_to_db(_name);

            auto stack_id        = _mas.correlation_id_internal;
            auto parent_stack_id = _mas.correlation_id_ancestor;
            auto correlation_id  = 0;
            auto queue_id        = 0;

            auto category_primary_key = data_processor.insert_string(
                trait::name<category::rocm_memory_allocate>::value);

            auto event_primary_key = data_processor.insert_event(
                category_primary_key, stack_id, parent_stack_id, correlation_id);

            data_processor.insert_memory_alloc(
                n_info.id, process.pid, thread_primary_key, agent_primary_key,
                type.c_str(), level.c_str(), _mas.start_timestamp, _mas.end_timestamp,
                _mas.address_value, _mas.allocation_size, queue_id, _mas.stream_handle,
                event_primary_key);
#    endif
        };
    };
}
#endif

postprocessing_callback
rocpd_post_processing::get_region_callback() const
{
    [[maybe_unused]] auto parse_args = []([[maybe_unused]] const std::string& arg_str) {
#if ROCPROFSYS_USE_ROCM > 0
        rocprofiler_sdk::function_args_t args;
        const std::string                delimiter = ";;";

        auto split = [](const std::string& str, const std::string& _delimiter) {
            std::vector<std::string> tokens;
            size_t                   start = 0;
            size_t                   end   = str.find(_delimiter);

            while(end != std::string::npos)
            {
                tokens.push_back(str.substr(start, end - start));
                start = end + _delimiter.length();
                end   = str.find(_delimiter, start);
            }

            return tokens;
        };

        auto tokens = split(arg_str, delimiter);

        // Ensure the number of tokens is a multiple of 4
        if(tokens.size() % 4 != 0)
        {
            throw std::invalid_argument("Malformed argument string.");
        }

        for(auto it = tokens.begin(); it != tokens.end(); it += 4)
        {
            rocprofiler_sdk::argument_info arg = { static_cast<uint32_t>(std::stoi(*it)),
                                                   *(it + 1), *(it + 2), *(it + 3) };
            args.push_back(arg);
        }

        return args;
#endif
    };

    return [&]([[maybe_unused]] const storage_parsed_type_base& parsed) {
#if ROCPROFSYS_USE_ROCM > 0
        auto  _rs            = static_cast<const struct region_sample&>(parsed);
        auto& data_processor = get_data_processor();
        auto& n_info         = node_info::get_instance();
        auto  process        = m_metadata.get_process_info();
        auto  thread_primary_key =
            data_processor.map_thread_id_to_primary_key(_rs.thread_id);

        auto callback_tracing_info = m_metadata.get_callback_tracing_info();
        auto _name                 = std::string{ callback_tracing_info.at(
            static_cast<rocprofiler_callback_tracing_kind_t>(_rs.kind),
            static_cast<rocprofiler_tracing_operation_t>(_rs.operation)) };
        auto name_primary_key      = data_processor.insert_string(_name.c_str());

        auto category_primary_key = data_processor.insert_string(_rs.category.c_str());

        size_t stack_id        = _rs.correlation_id_internal;
        size_t parent_stack_id = _rs.correlation_id_ancestor;
        size_t correlation_id  = 0;

        auto event_primary_key =
            data_processor.insert_event(category_primary_key, stack_id, parent_stack_id,
                                        correlation_id, _rs.call_stack.c_str());

        auto args = parse_args(_rs.args_str);
        for(const auto& arg : args)
        {
            data_processor.insert_args(event_primary_key, arg.arg_number,
                                       arg.arg_type.c_str(), arg.arg_name.c_str(),
                                       arg.arg_value.c_str());
        }

        data_processor.insert_region(n_info.id, process.pid, thread_primary_key,
                                     _rs.start_timestamp, _rs.end_timestamp,
                                     name_primary_key, event_primary_key);
#endif
    };
}

postprocessing_callback
rocpd_post_processing::get_in_time_sample_callback() const
{
    return [&](const storage_parsed_type_base& parsed) {
        auto  _its              = static_cast<const struct in_time_sample&>(parsed);
        auto& data_processor    = get_data_processor();
        auto  track_primary_key = data_processor.insert_string(_its.track_name.c_str());

        auto event_id = data_processor.insert_event(
            track_primary_key, _its.stack_id, _its.parent_stack_id, _its.correlation_id,
            _its.call_stack.c_str(), _its.line_info.c_str(), _its.event_metadata.c_str());
        data_processor.insert_sample(_its.track_name.c_str(), _its.timestamp_ns, event_id,
                                     "{}");
    };
}
postprocessing_callback
rocpd_post_processing::get_pmc_event_with_sample_callback() const
{
    return [&](const storage_parsed_type_base& parsed) {
        auto  _pmc           = static_cast<const struct pmc_event_with_sample&>(parsed);
        auto& data_processor = get_data_processor();
        auto  track_primary_key = data_processor.insert_string(_pmc.track_name.c_str());

        auto& agent_manager = agent_manager::get_instance();
        auto  agent_primary_key =
            agent_manager.get_agent_by_handle(_pmc.agent_handle).base_id;

        auto event_id = data_processor.insert_event(
            track_primary_key, _pmc.stack_id, _pmc.parent_stack_id, _pmc.correlation_id,
            _pmc.call_stack.c_str(), _pmc.line_info.c_str(), _pmc.event_metadata.c_str());
        data_processor.insert_sample(_pmc.track_name.c_str(), _pmc.timestamp_ns, event_id,
                                     "{}");

        data_processor.insert_pmc_event(event_id, agent_primary_key,
                                        _pmc.pmc_info_name.c_str(), _pmc.value);
    };
}

rocpd_post_processing::rocpd_post_processing(metadata& md)
: m_metadata(md)
{}

void
rocpd_post_processing::register_parser_callback([[maybe_unused]] storage_parser& parser)
{
#if ROCPROFSYS_USE_ROCM > 0
    if(!get_use_rocpd())
    {
        return;
    }
    parser.register_type_callback(entry_type::region, get_region_callback());
    parser.register_type_callback(entry_type::kernel_dispatch,
                                  get_kernel_dispatch_callback());
    parser.register_type_callback(entry_type::memory_copy, get_memory_copy_callback());
#    if(ROCPROFILER_VERSION >= 600)
    parser.register_type_callback(entry_type::memory_alloc,
                                  get_memory_allocate_callback());
#    endif
    parser.register_type_callback(entry_type::in_time_sample,
                                  get_in_time_sample_callback());
    parser.register_type_callback(entry_type::pmc_event_with_sample,
                                  get_pmc_event_with_sample_callback());
#endif
}

void
rocpd_post_processing::post_process_metadata()
{
#if ROCPROFSYS_USE_ROCM > 0
    if(!get_use_rocpd())
    {
        return;
    }
    auto& data_processor = get_data_processor();
    auto& agent_mngr     = agent_manager::get_instance();
    auto  n_info         = node_info::get_instance();

    data_processor.insert_node_info(n_info.id, n_info.hash, n_info.machine_id.c_str(),
                                    n_info.system_name.c_str(), n_info.node_name.c_str(),
                                    n_info.release.c_str(), n_info.version.c_str(),
                                    n_info.machine.c_str(), n_info.domain_name.c_str());

    auto process_info = m_metadata.get_process_info();
    data_processor.insert_process_info(n_info.id, process_info.ppid, process_info.pid, 0,
                                       0, 0, 0, process_info.command.c_str(), "{}");

    const auto& agents  = agent_mngr.get_agents();
    int         counter = 0;
    for(const auto& rocpd_agent : agents)
    {
        auto _base_id = rocpd::data_processor::get_instance().insert_agent(
            n_info.id, process_info.pid,
            ((rocpd_agent->type == agent_type::GPU) ? "GPU" : "CPU"), counter++,
            rocpd_agent->logical_node_id, rocpd_agent->logical_node_type_id,
            rocpd_agent->device_id, rocpd_agent->name.c_str(),
            rocpd_agent->model_name.c_str(), rocpd_agent->vendor_name.c_str(),
            rocpd_agent->product_name.c_str(), "");
        rocpd_agent->base_id = _base_id;
    }
    auto _string_list = m_metadata.get_string_list();
    for(auto& _string : _string_list)
    {
        data_processor.insert_string(std::string(_string).c_str());
    }

    auto _thread_info_list = m_metadata.get_thread_info_list();
    for(auto& t_info : _thread_info_list)
    {
        rocpd_insert_thread_id(t_info, n_info, process_info);
    }

    auto _track_info_list = m_metadata.get_track_info_list();
    for(auto& track : _track_info_list)
    {
        auto thread_id =
            track.thread_id.has_value()
                ? std::make_optional<size_t>(data_processor.map_thread_id_to_primary_key(
                      track.thread_id.value()))
                : std::nullopt;
        data_processor.insert_track(track.track_name.c_str(), n_info.id, process_info.pid,
                                    thread_id);
    }

    auto _code_object_list = m_metadata.get_code_object_list();
    for(const auto& code_object : _code_object_list)
    {
        auto dev_id =
            agent_mngr.get_agent_by_handle(get_handle_from_code_object(code_object))
                .base_id;

        const char* strg_type = "UNKNOWN";
        switch(code_object.storage_type)
        {
            case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE: strg_type = "FILE"; break;
            case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY: strg_type = "MEMORY"; break;
            default: break;
        }
        data_processor.insert_code_object(code_object.code_object_id, n_info.id,
                                          process_info.pid, dev_id, code_object.uri,
                                          code_object.load_base, code_object.load_size,
                                          code_object.load_delta, strg_type);
    }

    auto _kernel_symbols_list = m_metadata.get_kernel_symbol_list();
    for(const auto& kernel_symbol : _kernel_symbols_list)
    {
        auto kernel_name = tim::demangle(kernel_symbol.kernel_name);
        data_processor.insert_kernel_symbol(
            kernel_symbol.kernel_id, n_info.id, process_info.pid,
            kernel_symbol.code_object_id, kernel_symbol.kernel_name, kernel_name.c_str(),
            kernel_symbol.kernel_object, kernel_symbol.kernarg_segment_size,
            kernel_symbol.kernarg_segment_alignment, kernel_symbol.group_segment_size,
            kernel_symbol.private_segment_size, kernel_symbol.sgpr_count,
            kernel_symbol.arch_vgpr_count, kernel_symbol.accum_vgpr_count);

        data_processor.insert_string(kernel_name.c_str());
    }

    auto _queue_list = m_metadata.get_queue_list();
    for(const auto& queue_handle : _queue_list)
    {
        std::stringstream ss;
        ss << "Queue " << queue_handle;
        data_processor.insert_queue_info(queue_handle, n_info.id, process_info.pid,
                                         ss.str().c_str());
    }

    auto _stream_list = m_metadata.get_stream_list();
    for(const auto& stream_handle : _stream_list)
    {
        std::stringstream ss;
        ss << "Stream " << stream_handle;
        data_processor.insert_stream_info(stream_handle, n_info.id, process_info.pid,
                                          ss.str().c_str());
    }

    auto buffer_info_list = m_metadata.get_buffer_name_info();
    for(const auto& buffer_info : buffer_info_list)
    {
        for(const auto& item : buffer_info.items())
        {
            data_processor.insert_string(*item.second);
        }
    }

    auto callback_info_list = m_metadata.get_callback_tracing_info();
    for(const auto& cb_info : callback_info_list)
    {
        for(const auto& item : cb_info.items())
        {
            data_processor.insert_string(*item.second);
        }
    }

    auto pmc_info_list = m_metadata.get_pmc_info_list();
    for(const auto& pmc_info : pmc_info_list)
    {
        const auto agent_primary_key =
            agent_mngr.get_agent_by_handle(pmc_info.agent_handle).base_id;

        data_processor.insert_pmc_description(
            n_info.id, process_info.pid, agent_primary_key, pmc_info.target_arch.c_str(),
            pmc_info.event_code, pmc_info.instance_id, pmc_info.name.c_str(),
            pmc_info.symbol.c_str(), pmc_info.description.c_str(),
            pmc_info.long_description.c_str(), pmc_info.component.c_str(),
            pmc_info.units.c_str(), pmc_info.value_type.c_str(), pmc_info.block.c_str(),
            pmc_info.expression.c_str(), pmc_info.is_constant, pmc_info.is_derived);
    }
#endif
}

inline void
rocpd_post_processing::rocpd_insert_thread_id(info::thread&        t_info,
                                              const node_info&     n_info,
                                              const info::process& process_info) const
{
    const auto& extended_info = thread_info::get(t_info.thread_id, SequentTID);
    if(extended_info.has_value())
    {
        t_info.start = extended_info->get_start();
        t_info.end   = extended_info->get_stop();
    }

    std::stringstream ss;
    ss << "Thread " << t_info.thread_id;
    get_data_processor().insert_thread_info(n_info.id, process_info.ppid,
                                            process_info.pid, t_info.thread_id,
                                            ss.str().c_str(), t_info.start, t_info.end);
}

}  // namespace sample_cache
}  // namespace rocprofsys
