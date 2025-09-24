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

#include "metadata_registry.hpp"
#include "agent_manager.hpp"
#include "core/debug.hpp"
#include <algorithm>
#include <cstdint>

#include <fstream>

#include <nlohmann/json.hpp>

namespace rocprofsys
{
namespace trace_cache
{

namespace
{

template <typename ReturnType, typename DataType, typename Filter>
std::optional<ReturnType>
get_type_info(const DataType& data, const Filter& filter)
{
    std::optional<ReturnType> result = std::nullopt;
    data.rlock([&filter, &result](const auto& _data) {
        auto it = std::find_if(_data.begin(), _data.end(), filter);
        result  = it == _data.end() ? std::nullopt : std::optional<ReturnType>(*it);
    });
    return result;
}

template <typename T>
auto
assign_set_to_vector(T& result)
{
    return [&result](const auto& _data) { result.assign(_data.cbegin(), _data.cend()); };
}

nlohmann::json
to_json(const info::process& process)
{
    nlohmann::json result;
    result["pid"]     = process.pid;
    result["ppid"]    = process.ppid;
    result["command"] = process.command;
    result["start"]   = process.start;
    result["end"]     = process.end;
    return result;
}

info::process
from_json_process(const nlohmann::json& _json)
{
    info::process p;
    p.pid     = _json["pid"].get<pid_t>();
    p.ppid    = _json["ppid"].get<pid_t>();
    p.command = _json["command"].get<std::string>();
    p.start   = _json["start"].get<int32_t>();
    p.end     = _json["end"].get<int32_t>();
    return p;
}

nlohmann::json
to_json(const info::pmc& pmc)
{
    nlohmann::json result;
    result["type"]             = static_cast<int32_t>(pmc.type);
    result["agent_type_index"] = static_cast<int>(pmc.agent_type_index);
    result["target_arch"]      = pmc.target_arch;
    result["event_code"]       = static_cast<int>(pmc.event_code);
    result["instance_id"]      = static_cast<int>(pmc.instance_id);
    result["name"]             = pmc.name;
    result["symbol"]           = pmc.symbol;
    result["description"]      = pmc.description;
    result["long_description"] = pmc.long_description;
    result["component"]        = pmc.component;
    result["units"]            = pmc.units;
    result["value_type"]       = pmc.value_type;
    result["block"]            = pmc.block;
    result["expression"]       = pmc.expression;
    result["is_constant"]      = pmc.is_constant;
    result["is_derived"]       = pmc.is_derived;
    result["extdata"]          = pmc.extdata;
    return result;
}

info::pmc
from_json_pmc(const nlohmann::json& _json)
{
    info::pmc p;
    p.type             = static_cast<agent_type>(_json["type"].get<int32_t>());
    p.agent_type_index = _json["agent_type_index"].get<int32_t>();
    p.target_arch      = _json["target_arch"].get<std::string>();
    p.event_code       = _json["event_code"].get<int32_t>();
    p.instance_id      = _json["instance_id"].get<int32_t>();
    p.name             = _json["name"].get<std::string>();
    p.symbol           = _json["symbol"].get<std::string>();
    p.description      = _json["description"].get<std::string>();
    p.long_description = _json["long_description"].get<std::string>();
    p.component        = _json["component"].get<std::string>();
    p.units            = _json["units"].get<std::string>();
    p.value_type       = _json["value_type"].get<std::string>();
    p.block            = _json["block"].get<std::string>();
    p.expression       = _json["expression"].get<std::string>();
    p.is_constant      = _json["is_constant"].get<int32_t>();
    p.is_derived       = _json["is_derived"].get<int32_t>();
    p.extdata          = _json["extdata"].get<std::string>();
    return p;
}

nlohmann::json
to_json(const info::thread& thread)
{
    nlohmann::json result;
    result["parent_process_id"] = thread.parent_process_id;
    result["process_id"]        = thread.process_id;
    result["thread_id"]         = static_cast<int32_t>(thread.thread_id);
    result["start"]             = thread.start;
    result["end"]               = thread.end;
    result["extdata"]           = thread.extdata;
    return result;
}

info::thread
from_json_thread(const nlohmann::json& _json)
{
    info::thread t;
    t.parent_process_id = _json["parent_process_id"].get<int32_t>();
    t.process_id        = _json["process_id"].get<int32_t>();
    t.thread_id         = _json["thread_id"].get<int32_t>();
    t.start             = _json["start"].get<int32_t>();
    t.end               = _json["end"].get<int32_t>();
    t.extdata           = _json["extdata"].get<std::string>();
    return t;
}

nlohmann::json
to_json(const info::track& track)
{
    nlohmann::json result;
    result["track_name"] = track.track_name;
    if(track.thread_id.has_value())
    {
        result["thread_id"] = static_cast<int32_t>(track.thread_id.value());
    }
    else
    {
        result["thread_id"] = nullptr;
    }
    result["extdata"] = track.extdata;
    return result;
}

info::track
from_json_track(const nlohmann::json& _json)
{
    info::track t;
    t.track_name = _json["track_name"].get<std::string>();
    if(_json["thread_id"].is_null())
    {
        t.thread_id = std::nullopt;
    }
    else
    {
        t.thread_id = _json["thread_id"].get<int32_t>();
    }
    t.extdata = _json["extdata"].get<std::string>();
    return t;
}

#if ROCPROFSYS_USE_ROCM

nlohmann::json
to_json(const rocprofiler_callback_tracing_code_object_load_data_t& code_object)
{
    nlohmann::json result;
    result["code_object_id"] = static_cast<long long>(code_object.code_object_id);
    result["uri"]            = std::string(code_object.uri);
    result["load_base"]      = static_cast<long long>(code_object.load_base);
    result["load_size"]      = static_cast<long long>(code_object.load_size);
    result["load_delta"]     = static_cast<long long>(code_object.load_delta);
    result["storage_type"]   = static_cast<int>(code_object.storage_type);
#    if(ROCPROFILER_VERSION >= 600)
    result["agent_id_handle"] = static_cast<long long>(code_object.agent_id.handle);
#    else
    result["agent_id_handle"] = static_cast<long long>(code_object.rocp_agent.handle);
#    endif
    return result;
}

rocprofiler_callback_tracing_code_object_load_data_t
from_json_code_object(const nlohmann::json& _json)
{
    rocprofiler_callback_tracing_code_object_load_data_t co = {};
    co.code_object_id = _json["code_object_id"].get<long long>();
    auto uri_str      = _json["uri"].get<std::string>();
    co.uri            = new char[uri_str.size() + 1];
    strncpy(const_cast<char*>(co.uri), uri_str.c_str(), uri_str.size() + 1);
    co.load_base    = _json["load_base"].get<long long>();
    co.load_size    = _json["load_size"].get<long long>();
    co.load_delta   = _json["load_delta"].get<long long>();
    co.storage_type = static_cast<rocprofiler_code_object_storage_type_t>(
        _json["storage_type"].get<int>());
    auto handle = _json["agent_id_handle"].get<long long>();
#    if(ROCPROFILER_VERSION >= 600)
    co.agent_id.handle = handle;
#    else
    co.rocp_agent.handle = handle;
#    endif
    return co;
}

nlohmann::json
to_json(const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t&
            kernel_symbol)
{
    nlohmann::json result;
    result["kernel_id"]            = static_cast<long long>(kernel_symbol.kernel_id);
    result["code_object_id"]       = static_cast<long long>(kernel_symbol.code_object_id);
    result["kernel_name"]          = std::string(kernel_symbol.kernel_name);
    result["kernel_object"]        = static_cast<long long>(kernel_symbol.kernel_object);
    result["kernarg_segment_size"] = static_cast<int>(kernel_symbol.kernarg_segment_size);
    result["kernarg_segment_alignment"] =
        static_cast<int>(kernel_symbol.kernarg_segment_alignment);
    result["group_segment_size"]   = static_cast<int>(kernel_symbol.group_segment_size);
    result["private_segment_size"] = static_cast<int>(kernel_symbol.private_segment_size);
    result["sgpr_count"]           = static_cast<int>(kernel_symbol.sgpr_count);
    result["arch_vgpr_count"]      = static_cast<int>(kernel_symbol.arch_vgpr_count);
    result["accum_vgpr_count"]     = static_cast<int>(kernel_symbol.accum_vgpr_count);
    return result;
}

rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
from_json_kernel_symbol(const nlohmann::json& _json)
{
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t ks = {};
    ks.kernel_id         = _json["kernel_id"].get<long long>();
    ks.code_object_id    = _json["code_object_id"].get<long long>();
    auto kernel_name_str = _json["kernel_name"].get<std::string>();
    ks.kernel_name       = new char[kernel_name_str.size() + 1];
    strncpy(const_cast<char*>(ks.kernel_name), kernel_name_str.c_str(),
            sizeof(ks.kernel_name) + 1);
    ks.kernel_object             = _json["kernel_object"].get<long long>();
    ks.kernarg_segment_size      = _json["kernarg_segment_size"].get<int>();
    ks.kernarg_segment_alignment = _json["kernarg_segment_alignment"].get<int>();
    ks.group_segment_size        = _json["group_segment_size"].get<int>();
    ks.private_segment_size      = _json["private_segment_size"].get<int>();
    ks.sgpr_count                = _json["sgpr_count"].get<int>();
    ks.arch_vgpr_count           = _json["arch_vgpr_count"].get<int>();
    ks.accum_vgpr_count          = _json["accum_vgpr_count"].get<int>();
    return ks;
}
#endif

nlohmann::json
to_json(const agent& _agent)
{
    nlohmann::json result;
    result["type"]                 = _agent.type;
    result["handle"]               = _agent.handle;
    result["device_id"]            = _agent.device_id;
    result["node_id"]              = _agent.node_id;
    result["logical_node_id"]      = _agent.logical_node_id;
    result["logical_node_type_id"] = _agent.logical_node_type_id;
    result["name"]                 = _agent.name;
    result["model_name"]           = _agent.model_name;
    result["vendor_name"]          = _agent.vendor_name;
    result["product_name"]         = _agent.product_name;
    result["device_type_index"]    = _agent.device_type_index;
    return result;
}

std::shared_ptr<agent>
from_json_agent(const nlohmann::json& _json)
{
    auto a                  = std::make_shared<agent>();
    a->type                 = _json["type"].get<agent_type>();
    a->handle               = _json["handle"].get<uint64_t>();
    a->device_id            = _json["device_id"].get<int32_t>();
    a->node_id              = _json["node_id"].get<int32_t>();
    a->logical_node_id      = _json["logical_node_id"].get<int32_t>();
    a->logical_node_type_id = _json["logical_node_type_id"].get<int32_t>();
    a->name                 = _json["name"].get<std::string>();
    a->model_name           = _json["model_name"].get<std::string>();
    a->vendor_name          = _json["vendor_name"].get<std::string>();
    a->product_name         = _json["product_name"].get<std::string>();
    a->device_type_index    = _json["device_type_index"].get<int32_t>();
    return a;
}

nlohmann::json
to_json(const metadata_registry&                   _registry,
        const std::vector<std::shared_ptr<agent>>& _agents)
{
    nlohmann::json result;

    auto process_info = _registry.get_process_info();
    result["process"] = to_json(process_info);

    auto           pmc_list  = _registry.get_pmc_info_list();
    nlohmann::json pmc_array = nlohmann::json::array();
    for(const auto& pmc : pmc_list)
    {
        pmc_array.push_back(to_json(pmc));
    }
    result["pmc_infos"] = pmc_array;

    auto           thread_list  = _registry.get_thread_info_list();
    nlohmann::json thread_array = nlohmann::json::array();
    for(const auto& thread : thread_list)
    {
        thread_array.push_back(to_json(thread));
    }
    result["threads"] = thread_array;

    auto           track_list  = _registry.get_track_info_list();
    nlohmann::json track_array = nlohmann::json::array();
    for(const auto& track : track_list)
    {
        track_array.push_back(to_json(track));
    }
    result["tracks"] = track_array;

    auto queue_list = _registry.get_queue_list();
    for(const auto& queue : queue_list)
    {
        result["queues"].push_back(static_cast<long long>(queue));
    }

    auto stream_list = _registry.get_stream_list();
    for(const auto& stream : stream_list)
    {
        result["streams"].push_back(static_cast<long long>(stream));
    }

    auto string_list = _registry.get_string_list();
    for(const auto& str : string_list)
    {
        result["strings"].push_back(str);
    }

#if ROCPROFSYS_USE_ROCM
    auto           code_object_list  = _registry.get_code_object_list();
    nlohmann::json code_object_array = nlohmann::json::array();
    for(const auto& code_object : code_object_list)
    {
        code_object_array.push_back(to_json(code_object));
    }
    result["code_objects"] = code_object_array;

    auto           kernel_symbol_list  = _registry.get_kernel_symbol_list();
    nlohmann::json kernel_symbol_array = nlohmann::json::array();
    for(const auto& kernel_symbol : kernel_symbol_list)
    {
        kernel_symbol_array.push_back(to_json(kernel_symbol));
    }
    result["kernel_symbols"] = kernel_symbol_array;
#endif

    for(const auto& agent : _agents)
    {
        if(agent == nullptr)
        {
            continue;
        }
        result["agents"].push_back(to_json(*agent));
    }

    return result;
}

void
from_json(metadata_registry& _registry, std::vector<std::shared_ptr<agent>>& _agents,
          const nlohmann::json& _json)
{
    const auto& process_json = _json["process"];
    auto        process      = from_json_process(process_json);
    _registry.set_process(process);

    const auto& pmc_array = _json["pmc_infos"];
    for(const auto& pmc_json : pmc_array)
    {
        auto pmc = from_json_pmc(pmc_json);
        _registry.add_pmc_info(pmc);
    }

    const auto& thread_array = _json["threads"];
    for(const auto& thread_json : thread_array)
    {
        auto thread = from_json_thread(thread_json);
        _registry.add_thread_info(thread);
    }

    const auto& track_array = _json["tracks"];
    for(const auto& track_json : track_array)
    {
        auto track = from_json_track(track_json);
        _registry.add_track(track);
    }

    const auto& queue_array = _json["queues"];
    for(const auto& queue_json : queue_array)
    {
        auto handle = queue_json.get<long long>();
        _registry.add_queue(static_cast<uint64_t>(handle));
    }

    const auto& stream_array = _json["streams"];
    for(const auto& stream_json : stream_array)
    {
        auto handle = stream_json.get<long long>();
        _registry.add_stream(static_cast<uint64_t>(handle));
    }

    const auto& string_array = _json["strings"];
    for(const auto& string_json : string_array)
    {
        auto str = string_json.get<std::string>();
        _registry.add_string(str);
    }

#if ROCPROFSYS_USE_ROCM
    if(_json.contains("code_objects"))
    {
        const auto& code_object_array = _json["code_objects"];
        for(const auto& code_object_json : code_object_array)
        {
            auto code_object = from_json_code_object(code_object_json);
            _registry.add_code_object(code_object);
        }
    }

    if(_json.contains("kernel_symbols"))
    {
        const auto& kernel_symbol_array = _json["kernel_symbols"];
        for(const auto& kernel_symbol_json : kernel_symbol_array)
        {
            auto kernel_symbol = from_json_kernel_symbol(kernel_symbol_json);
            _registry.add_kernel_symbol(kernel_symbol);
        }
    }
#endif

    if(!_agents.empty())
    {
        ROCPROFSYS_WARNING(0, "Given agents vector is not empty. Clearing it..");
        _agents.clear();
    }

    if(_json.contains("agents"))
    {
        const auto& agents_array = _json["agents"];
        for(const auto& agent_json : agents_array)
        {
            _agents.push_back(from_json_agent(agent_json));
        }
    }
}

}  // namespace

void
metadata_registry::set_process(const info::process& process)
{
    m_process.wlock([&process](auto& _process) { _process = process; });
}

void
metadata_registry::add_pmc_info(const info::pmc& pmc_info)
{
    m_pmc_infos.wlock([&pmc_info](auto& _data) {
        if(_data.count(pmc_info) > 0)
        {
            return;
        }
        _data.emplace(pmc_info);
    });
}

void
metadata_registry::add_thread_info(const info::thread& thread_info)
{
    m_threads.wlock([&thread_info](auto& _data) {
        if(_data.count(thread_info) > 0)
        {
            return;
        }
        _data.emplace(thread_info);
    });
}

void
metadata_registry::add_track(const info::track& track_info)
{
    m_tracks.wlock([&track_info](auto& _data) {
        if(_data.count(track_info) > 0)
        {
            return;
        }
        _data.emplace(track_info);
    });
}

void
metadata_registry::add_queue(const uint64_t& queue_handle)
{
    m_queues.wlock([&queue_handle](auto& _data) {
        if(_data.count(queue_handle) > 0)
        {
            return;
        }
        _data.emplace(queue_handle);
    });
}

void
metadata_registry::add_stream(const uint64_t& stream_handle)
{
    m_streams.wlock([&stream_handle](auto& _data) {
        if(_data.count(stream_handle) > 0)
        {
            return;
        }
        _data.emplace(stream_handle);
    });
}

void
metadata_registry::add_string(const std::string_view& string_value)
{
    m_strings.wlock([&string_value](auto& _data) {
        if(_data.count(string_value) > 0)
        {
            return;
        }
        _data.emplace(string_value);
    });
}

info::process
metadata_registry::get_process_info() const
{
    info::process result;
    m_process.rlock([&result](const auto& _process) { result = _process; });
    return result;
}

std::optional<info::pmc>
metadata_registry::get_pmc_info(const std::string_view& unique_name) const
{
    return get_type_info<info::pmc>(m_pmc_infos, [&unique_name](const info::pmc& val) {
        return val.name == unique_name;
    });
}

std::optional<info::thread>
metadata_registry::get_thread_info(const uint32_t& thread_id) const
{
    return get_type_info<info::thread>(m_threads, [&thread_id](const info::thread& val) {
        return val.thread_id == thread_id;
    });
}

std::optional<info::track>
metadata_registry::get_track_info(const std::string_view& track_name) const
{
    return get_type_info<info::track>(m_tracks, [&track_name](const info::track& val) {
        return val.track_name == track_name;
    });
}

std::vector<info::pmc>
metadata_registry::get_pmc_info_list() const
{
    std::vector<info::pmc> result;
    m_pmc_infos.rlock(assign_set_to_vector(result));
    return result;
}

std::vector<info::thread>
metadata_registry::get_thread_info_list() const
{
    std::vector<info::thread> result;
    m_threads.rlock(assign_set_to_vector(result));
    return result;
}

std::vector<info::track>
metadata_registry::get_track_info_list() const
{
    std::vector<info::track> result;
    m_tracks.rlock(assign_set_to_vector(result));
    return result;
}

std::vector<uint64_t>
metadata_registry::get_queue_list() const
{
    std::vector<uint64_t> result;
    m_queues.rlock(assign_set_to_vector(result));
    return result;
}

std::vector<uint64_t>
metadata_registry::get_stream_list() const
{
    std::vector<uint64_t> result;
    m_streams.rlock(assign_set_to_vector(result));
    return result;
}

std::vector<std::string_view>
metadata_registry::get_string_list() const
{
    std::vector<std::string_view> result;
    m_strings.rlock(assign_set_to_vector(result));
    return result;
}

#if ROCPROFSYS_USE_ROCM > 0

void
metadata_registry::add_code_object(
    const rocprofiler_callback_tracing_code_object_load_data_t& code_object)
{
    m_code_objects.wlock([&code_object](auto& _data) {
        if(_data.count(code_object) > 0)
        {
            return;
        }
        _data.emplace(code_object);
    });
}

void
metadata_registry::add_kernel_symbol(
    const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t&
        kernel_symbol)
{
    m_kernel_symbols.wlock([&kernel_symbol](auto& _data) {
        if(_data.count(kernel_symbol) > 0)
        {
            return;
        }
        _data.emplace(kernel_symbol);
    });
}

std::optional<rocprofiler_callback_tracing_code_object_load_data_t>
metadata_registry::get_code_object(uint64_t code_object_id) const
{
    return get_type_info<rocprofiler_callback_tracing_code_object_load_data_t>(
        m_code_objects,
        [&code_object_id](
            const rocprofiler_callback_tracing_code_object_load_data_t& val) {
            return val.code_object_id == code_object_id;
        });
}

std::optional<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t>
metadata_registry::get_kernel_symbol(uint64_t kernel_id) const
{
    return get_type_info<
        rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t>(
        m_kernel_symbols,
        [&kernel_id](
            const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t&
                val) { return val.kernel_id == kernel_id; });
}

std::vector<rocprofiler_callback_tracing_code_object_load_data_t>
metadata_registry::get_code_object_list() const
{
    std::vector<rocprofiler_callback_tracing_code_object_load_data_t> result;
    m_code_objects.rlock(assign_set_to_vector(result));
    return result;
}

std::vector<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t>
metadata_registry::get_kernel_symbol_list() const
{
    std::vector<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t>
        result;
    m_kernel_symbols.rlock(assign_set_to_vector(result));
    return result;
}

// As the underlying implementation of callback_name_info_t resizes the category
// storage during emplace, this special method is required
void
metadata_registry::overwrite_callback_names(
    std::initializer_list<
        std::pair<rocprofiler_callback_tracing_kind_t, callback_rename_map_t>>
        rename_table)
{
    if(rename_table.size() == 0) return;

    using callback_kind_t   = rocprofiler_callback_tracing_kind_t;
    using operation_names_t = std::vector<std::string_view>;

    auto category_names = std::vector<std::string_view>{};
    auto modified_ops   = std::map<callback_kind_t, operation_names_t>{};

    auto extract_operations = [&](callback_kind_t cat) -> operation_names_t {
        auto        items           = m_callback_tracing_info.items();
        const auto* target_category = items[static_cast<size_t>(cat)];

        auto              operations_data = target_category->items();
        operation_names_t operation_names;
        operation_names.reserve(operations_data.size());

        for(const auto& [op_idx, op_name] : operations_data)
            operation_names.push_back(*op_name);

        return operation_names;
    };

    // Store category names
    category_names.resize(ROCPROFILER_CALLBACK_TRACING_LAST);
    for(callback_kind_t i = ROCPROFILER_CALLBACK_TRACING_NONE;
        i < ROCPROFILER_CALLBACK_TRACING_LAST;
        i = static_cast<callback_kind_t>(static_cast<int>(i) + 1))
    {
        category_names[i] = m_callback_tracing_info.at(i);
    }

    // Process list
    for(const auto& category_info : rename_table)
    {
        auto callback_kind = category_info.first;
        // Store operations of all following categories
        //  as they will be deleted
        for(callback_kind_t i =
                static_cast<callback_kind_t>(static_cast<int>(callback_kind) + 1);
            i < ROCPROFILER_CALLBACK_TRACING_LAST;
            i = static_cast<callback_kind_t>(static_cast<int>(i) + 1))
        {
            if(modified_ops.find(i) != modified_ops.end()) break;
            modified_ops[i] = extract_operations(i);
        }

        ROCPROFSYS_CI_THROW(modified_ops.find(callback_kind) != modified_ops.end(),
                            "Overwriting a previously overwritten entry is forbidden");

        ROCPROFSYS_CI_THROW(!modified_ops.empty() &&
                                callback_kind >= modified_ops.begin()->first,
                            "Category must have a larger enum value than all previously "
                            "modified_ops categories");

        // Overwrite desired category
        auto operation_names = extract_operations(callback_kind);
        for(const auto& [index, new_value] : category_info.second)
        {
            ROCPROFSYS_CI_THROW(index < 0 ||
                                    static_cast<size_t>(index) >= operation_names.size(),
                                "Index is invalid");
            operation_names[index] = new_value;
        }
        modified_ops[callback_kind] = std::move(operation_names);
    }
    if(modified_ops.empty()) return;

    // Emplace the changed category operations
    for(callback_kind_t i = modified_ops.begin()->first;
        i < ROCPROFILER_CALLBACK_TRACING_LAST;
        i = static_cast<callback_kind_t>(static_cast<int>(i) + 1))
    {
        auto renaming_entry = modified_ops.find(i);

        ROCPROFSYS_CI_THROW(renaming_entry == modified_ops.end(),
                            "A category that needs to be emplaced is missing");

        const auto& operations_vec = renaming_entry->second;
        m_callback_tracing_info.emplace(i, category_names.at(i).data());
        for(size_t op_idx = 0; op_idx < operations_vec.size(); ++op_idx)
        {
            m_callback_tracing_info.emplace(
                i, static_cast<rocprofiler_tracing_operation_t>(op_idx),
                operations_vec[op_idx].data());
        }
    }
}

rocprofiler::sdk::buffer_name_info_t<const char*>
metadata_registry::get_buffer_name_info() const
{
    return m_buffered_tracing_info;
}

rocprofiler::sdk::callback_name_info_t<const char*>
metadata_registry::get_callback_tracing_info() const
{
    return m_callback_tracing_info;
}

#endif

metadata_registry::metadata_registry()
{
#if ROCPROFSYS_USE_ROCM > 0
    overwrite_callback_names({
#    if(ROCPROFILER_VERSION >= 600)
        { ROCPROFILER_CALLBACK_TRACING_OMPT,
          { { ROCPROFILER_OMPT_ID_thread_begin, "omp_thread" },
            { ROCPROFILER_OMPT_ID_thread_end, "omp_thread" },
            { ROCPROFILER_OMPT_ID_parallel_begin, "omp_parallel" },
            { ROCPROFILER_OMPT_ID_parallel_end, "omp_parallel" } } }
#    endif
    });
#endif
}

bool
metadata_registry::save_to_file(const std::string&                         filepath,
                                const std::vector<std::shared_ptr<agent>>& _agents) const
{
    try
    {
        auto json        = to_json(*this, _agents);
        auto json_string = json.dump();

        std::ofstream file(filepath);
        if(!file.is_open())
        {
            return false;
        }

        file << json_string;
        file.close();
        return true;
    } catch(const std::exception& e)
    {
        return false;
    }
}

bool
metadata_registry::load_from_file(const std::string&                   filepath,
                                  std::vector<std::shared_ptr<agent>>& _agents)
{
    try
    {
        std::ifstream file(filepath);
        if(!file.is_open())
        {
            return false;
        }

        nlohmann::json json;
        file >> json;
        file.close();

        rocprofsys::trace_cache::from_json(*this, _agents, json);
        return true;
    } catch(const std::exception& e)
    {
        return false;
    }
}

}  // namespace trace_cache
}  // namespace rocprofsys
