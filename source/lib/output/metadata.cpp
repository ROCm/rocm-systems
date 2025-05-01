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

#include "metadata.hpp"

#include "lib/att-tool/att_lib_wrapper.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/output/agent_info.hpp"
#include "lib/output/host_symbol_info.hpp"
#include "lib/output/kernel_symbol_info.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

#include <fmt/core.h>
#include <fmt/format.h>

#include <unistd.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace rocprofiler
{
namespace tool
{
namespace fs = common::filesystem;
namespace
{
rocprofiler_status_t
query_pc_sampling_configuration(const rocprofiler_pc_sampling_configuration_t* configs,
                                long unsigned int                              num_config,
                                void*                                          user_data)
{
    auto* avail_configs =
        static_cast<std::vector<rocprofiler_pc_sampling_configuration_t>*>(user_data);
    for(size_t i = 0; i < num_config; i++)
    {
        avail_configs->emplace_back(configs[i]);
    }
    return ROCPROFILER_STATUS_SUCCESS;
}
}  // namespace

kernel_symbol_info::kernel_symbol_info()
: base_type{0, 0, 0, "", 0, 0, 0, 0, 0, 0, 0, 0, 0, {.handle = 0}}
{}

constexpr auto null_address_v = rocprofiler_address_t{.handle = 0};
constexpr auto null_dim3_v    = rocprofiler_dim3_t{.x = 0, .y = 0, .z = 0};

host_function_info::host_function_info()
: base_type{0,
            0,
            0,
            0,
            null_address_v,
            null_address_v,
            "",
            0,
            null_dim3_v,
            null_dim3_v,
            null_dim3_v,
            null_dim3_v,
            0}
{}

metadata::metadata(inprocess)
: buffer_names{sdk::get_buffer_tracing_names()}
, callback_names{sdk::get_callback_tracing_names()}
{
    ROCPROFILER_CHECK(rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0,
        [](rocprofiler_agent_version_t, const void** _agents, size_t _num_agents, void* _data) {
            auto* _agents_v = static_cast<agent_info_vec_t*>(_data);
            _agents_v->reserve(_num_agents);
            for(size_t i = 0; i < _num_agents; ++i)
            {
                auto* agent = static_cast<const rocprofiler_agent_v0_t*>(_agents[i]);
                _agents_v->emplace_back(*agent);
            }
            return ROCPROFILER_STATUS_SUCCESS;
        },
        sizeof(rocprofiler_agent_v0_t),
        &agents));

    {
        auto _gpu_agents = std::vector<agent_info*>{};

        _gpu_agents.reserve(agents.size());
        for(auto& itr : agents)
        {
            if(itr.type == ROCPROFILER_AGENT_TYPE_GPU)
            {
                _gpu_agents.emplace_back(&itr);
                auto pc_configs = std::vector<rocprofiler_pc_sampling_configuration_t>{};
                rocprofiler_query_pc_sampling_agent_configurations(
                    itr.id, query_pc_sampling_configuration, &pc_configs);
                agent_pc_sample_config_info.emplace(itr.id, pc_configs);
            }
        }

        // make sure they are sorted by node id
        std::sort(_gpu_agents.begin(), _gpu_agents.end(), [](const auto& lhs, const auto& rhs) {
            return CHECK_NOTNULL(lhs)->node_id < CHECK_NOTNULL(rhs)->node_id;
        });

        int64_t _dev_id = 0;
        for(auto& itr : _gpu_agents)
            itr->gpu_index = _dev_id++;
    }

    for(auto itr : agents)
        agents_map.emplace(itr.id, itr);

    // Add kernel ID of zero
    kernel_symbol_info info{};
    info.kernel_id             = 0;
    info.formatted_kernel_name = "0";
    info.demangled_kernel_name = "0";
    info.truncated_kernel_name = "0";
    add_kernel_symbol(std::move(info));
}

void metadata::init(inprocess)
{
    if(inprocess_init) return;

    inprocess_init = true;
    for(auto itr : agents)
    {
        if(itr.type == ROCPROFILER_AGENT_TYPE_CPU) continue;

        auto status = rocprofiler_iterate_agent_supported_counters(
            itr.id,
            [](rocprofiler_agent_id_t    id,
               rocprofiler_counter_id_t* counters,
               size_t                    num_counters,
               void*                     user_data) {
                auto* data_v = static_cast<agent_counter_info_map_t*>(user_data);
                data_v->emplace(id, counter_info_vec_t{});

                for(size_t i = 0; i < num_counters; ++i)
                {
                    auto _info     = rocprofiler_counter_info_v1_t{};
                    auto _dim_ids  = std::vector<rocprofiler_counter_dimension_id_t>{};
                    auto _dim_info = std::vector<rocprofiler_record_dimension_info_t>{};

                    ROCPROFILER_CHECK(rocprofiler_query_counter_info(
                        counters[i],
                        ROCPROFILER_COUNTER_INFO_VERSION_1,
                        &static_cast<rocprofiler_counter_info_v1_t&>(_info)));

                    for(uint64_t j = 0; j < _info.dimensions_count; ++j)
                        _dim_ids.emplace_back(_info.dimensions[j].id);

                    data_v->at(id).emplace_back(
                        id, _info, std::move(_dim_ids), std::move(_dim_info));
                }
                return ROCPROFILER_STATUS_SUCCESS;
            },
            &agent_counter_info);

        ROCP_WARNING_IF(status != ROCPROFILER_STATUS_SUCCESS) << fmt::format(
            "rocprofiler_iterate_agent_supported_counters returned {} for agent {} ({}) :: {}",
            rocprofiler_get_status_name(status),
            itr.node_id,
            itr.name,
            rocprofiler_get_status_string(status));
    }
}

const agent_info*
metadata::get_agent(rocprofiler_agent_id_t _val) const
{
    for(const auto& itr : agents)
    {
        if(itr.id == _val) return &itr;
    }
    return nullptr;
}

const code_object_info*
metadata::get_code_object(uint64_t code_obj_id) const
{
    return code_objects.rlock([code_obj_id](const auto& _data) -> const code_object_info* {
        return &_data.at(code_obj_id);
    });
}

code_object_load_info_vec_t
metadata::get_code_object_load_info() const
{
    auto _data = code_object_load.rlock([](const auto& _data_v) {
        auto _info = std::vector<rocprofiler::att_wrapper::CodeobjLoadInfo>{};
        _info.reserve(_data_v.size());
        for(const auto& itr : _data_v)
            _info.emplace_back(itr);
        return _info;
    });

    uint64_t _sz = 0;
    for(const auto& itr : _data)
        _sz = std::max(_sz, itr.id);

    ROCP_WARNING_IF((_sz + 1) - _data.size() > 1000) << fmt::format(
        "Spares index detected for code object load info: {} < {}", _data.size(), _sz);

    auto _code_obj_data = std::vector<rocprofiler::att_wrapper::CodeobjLoadInfo>{};
    _code_obj_data.resize(_sz + 1, rocprofiler::att_wrapper::CodeobjLoadInfo{});
    // index by the code object id
    for(auto& itr : _data)
        _code_obj_data.at(itr.id) = itr;

    return _code_obj_data;
}

std::vector<std::string>
metadata::get_att_filenames() const
{
    auto data = std::vector<std::string>{};
    for(const auto& filenames : att_filenames)
    {
        for(const auto& file : filenames.second.second)
        {
            data.emplace_back(fs::path{file}.filename());
        }
    }
    return data;
}

const kernel_symbol_info*
metadata::get_kernel_symbol(uint64_t kernel_id) const
{
    return kernel_symbols.rlock([kernel_id](const auto& _data) -> const kernel_symbol_info* {
        return &_data.at(kernel_id);
    });
}

const host_function_info*
metadata::get_host_function(uint64_t host_function_id) const
{
    return host_functions.rlock([host_function_id](const auto& _data) -> const host_function_info* {
        return &_data.at(host_function_id);
    });
}

const tool_counter_info*
metadata::get_counter_info(uint64_t instance_id) const
{
    auto _counter_id = rocprofiler_counter_id_t{.handle = 0};
    ROCPROFILER_CHECK(rocprofiler_query_record_counter_id(instance_id, &_counter_id));
    return get_counter_info(_counter_id);
}

const tool_counter_info*
metadata::get_counter_info(rocprofiler_counter_id_t id) const
{
    for(const auto& itr : agent_counter_info)
    {
        for(const auto& aitr : itr.second)
        {
            if(aitr.id == id) return &aitr;
        }
    }
    return nullptr;
}

const counter_dimension_info_vec_t*
metadata::get_counter_dimension_info(uint64_t instance_id) const
{
    return &CHECK_NOTNULL(get_counter_info(instance_id))->dimensions;
}

code_object_data_vec_t
metadata::get_code_objects() const
{
    auto _data = code_objects.rlock([](const auto& _data_v) {
        auto _info = std::vector<code_object_info>{};
        _info.reserve(_data_v.size());
        for(const auto& itr : _data_v)
            _info.emplace_back(itr.second);
        return _info;
    });

    uint64_t _sz = 0;
    for(const auto& itr : _data)
        _sz = std::max(_sz, itr.code_object_id);

    auto _code_obj_data = std::vector<code_object_info>{};
    _code_obj_data.resize(_sz + 1, code_object_info{});
    // index by the code object id
    for(auto& itr : _data)
        _code_obj_data.at(itr.code_object_id) = itr;

    return _code_obj_data;
}

kernel_symbol_data_vec_t
metadata::get_kernel_symbols() const
{
    auto _data = kernel_symbols.rlock([](const auto& _data_v) {
        auto _info = std::vector<kernel_symbol_info>{};
        _info.reserve(_data_v.size());
        for(const auto& itr : _data_v)
            _info.emplace_back(itr.second);
        return _info;
    });

    uint64_t kernel_data_size = 0;
    for(const auto& itr : _data)
        kernel_data_size = std::max(kernel_data_size, itr.kernel_id);

    auto _symbol_data = std::vector<kernel_symbol_info>{};
    _symbol_data.resize(kernel_data_size + 1, kernel_symbol_info{});
    // index by the kernel id
    for(auto& itr : _data)
        _symbol_data.at(itr.kernel_id) = std::move(itr);

    return _symbol_data;
}

host_function_data_vec_t
metadata::get_host_symbols() const
{
    return host_functions.rlock([](const auto& _data_v) {
        auto _info = std::vector<host_function_info>{};
        _info.resize(_data_v.size() + 1, host_function_info{});
        for(const auto& itr : _data_v)
            _info.at(itr.first) = itr.second;
        return _info;
    });
}

metadata::agent_info_ptr_vec_t
metadata::get_gpu_agents() const
{
    auto _data = metadata::agent_info_ptr_vec_t{};
    for(const auto& itr : agents)
    {
        if(itr.type == ROCPROFILER_AGENT_TYPE_GPU) _data.emplace_back(&itr);
    }
    return _data;
}

pc_sample_config_vec_t
metadata::get_pc_sample_config_info(rocprofiler_agent_id_t _val) const
{
    auto _ret             = pc_sample_config_vec_t{};
    auto pc_sample_config = agent_pc_sample_config_info.at(_val);
    for(const auto& itr : pc_sample_config)
        _ret.emplace_back(itr);

    return _ret;
}

counter_info_vec_t
metadata::get_counter_info() const
{
    auto _ret = std::vector<tool_counter_info>{};
    for(const auto& itr : agent_counter_info)
    {
        for(const auto& iitr : itr.second)
            _ret.emplace_back(iitr);
    }
    return _ret;
}

counter_dimension_vec_t
metadata::get_counter_dimension_info() const
{
    auto _ret = counter_dimension_vec_t{};
    for(const auto& itr : agent_counter_info)
    {
        for(const auto& iitr : itr.second)
            for(const auto& ditr : iitr.dimensions)
                _ret.emplace_back(ditr);
    }

    auto _sorter = [](const rocprofiler_record_dimension_info_t& lhs,
                      const rocprofiler_record_dimension_info_t& rhs) {
        return std::tie(lhs.id, lhs.instance_size) < std::tie(rhs.id, rhs.instance_size);
    };
    auto _equiv = [](const rocprofiler_record_dimension_info_t& lhs,
                     const rocprofiler_record_dimension_info_t& rhs) {
        return std::tie(lhs.id, lhs.instance_size) == std::tie(rhs.id, rhs.instance_size);
    };

    std::sort(_ret.begin(), _ret.end(), _sorter);
    _ret.erase(std::unique(_ret.begin(), _ret.end(), _equiv), _ret.end());

    return _ret;
}

bool
metadata::add_marker_message(uint64_t corr_id, std::string&& msg)
{
    return marker_messages.wlock(
        [](auto& _data, uint64_t _cid_v, std::string&& _msg) -> bool {
            return _data.emplace(_cid_v, std::move(_msg)).second;
        },
        corr_id,
        std::move(msg));
}

bool
metadata::add_code_object(code_object_info obj)
{
    return code_objects.wlock(
        [](code_object_data_map_t& _data_v, code_object_info _obj_v) -> bool {
            return _data_v.emplace(_obj_v.code_object_id, _obj_v).second;
        },
        obj);
}

bool
metadata::add_kernel_symbol(kernel_symbol_info&& sym)
{
    return kernel_symbols.wlock(
        [](kernel_symbol_data_map_t& _data_v, kernel_symbol_info&& _sym_v) -> bool {
            return _data_v.emplace(_sym_v.kernel_id, std::move(_sym_v)).second;
        },
        std::move(sym));
}

bool
metadata::add_host_function(host_function_info&& func)
{
    return host_functions.wlock(
        [](host_function_info_map_t& _data_v, host_function_info&& _func_v) -> bool {
            return _data_v.emplace(_func_v.host_function_id, std::move(_func_v)).second;
        },
        std::move(func));
}

bool
metadata::add_string_entry(size_t key, std::string_view str)
{
    return string_entries.ulock(
        [](const auto& _data, size_t _key, std::string_view) { return (_data.count(_key) > 0); },
        [](auto& _data, size_t _key, std::string_view _str) {
            _data.emplace(_key, new std::string{_str});
            return true;
        },
        key,
        str);
}

bool
metadata::add_external_correlation_id(uint64_t val)
{
    return external_corr_ids.wlock(
        [](auto& _data, uint64_t _val) { return _data.emplace(_val).second; }, val);
}

bool
metadata::add_runtime_initialization(rocprofiler_runtime_initialization_operation_t runtime_op)
{
    return runtime_initialization_set.wlock(
        [](auto& _data, rocprofiler_runtime_initialization_operation_t _runtime_op) {
            return _data.emplace(_runtime_op).second;
        },
        runtime_op);
}

bool
metadata::is_runtime_initialized(rocprofiler_runtime_initialization_operation_t runtime_op) const
{
    return runtime_initialization_set.rlock(
        [](const auto& _data, rocprofiler_runtime_initialization_operation_t _runtime_op) {
            return _data.count(_runtime_op) > 0;
        },
        runtime_op);
}

std::string_view
metadata::get_marker_message(uint64_t corr_id) const
{
    return marker_messages.rlock(
        [](const auto& _data, uint64_t _corr_id_v) -> std::string_view {
            return _data.at(_corr_id_v);
        },
        corr_id);
}

std::string_view
metadata::get_kernel_name(uint64_t kernel_id, uint64_t rename_id) const
{
    if(rename_id > 0)
    {
        if(const auto* _name = common::get_string_entry(rename_id)) return std::string_view{*_name};
    }

    const auto* _kernel_data = get_kernel_symbol(kernel_id);
    return CHECK_NOTNULL(_kernel_data)->formatted_kernel_name;
}

std::string_view
metadata::get_kind_name(rocprofiler_callback_tracing_kind_t kind) const
{
    return callback_names.at(kind);
}

std::string_view
metadata::get_kind_name(rocprofiler_buffer_tracing_kind_t kind) const
{
    return buffer_names.at(kind);
}

std::string_view
metadata::get_operation_name(rocprofiler_callback_tracing_kind_t kind,
                             rocprofiler_tracing_operation_t     op) const
{
    return callback_names.at(kind, op);
}

std::string_view
metadata::get_operation_name(rocprofiler_buffer_tracing_kind_t kind,
                             rocprofiler_tracing_operation_t   op) const
{
    return buffer_names.at(kind, op);
}

agent_index
metadata::get_agent_index(rocprofiler_agent_id_t id, agent_indexing index) const
{
    const auto* _agent = get_agent(id);
    ROCP_FATAL_IF(!_agent) << "Information of the agent with handle: " << id.handle
                           << " is not present";

    // stringify agent type
    auto get_type = [_agent]() -> std::string_view {
        switch(_agent->type)
        {
            case ROCPROFILER_AGENT_TYPE_CPU: return "CPU";
            case ROCPROFILER_AGENT_TYPE_GPU: return "GPU";
            case ROCPROFILER_AGENT_TYPE_NONE:
            case ROCPROFILER_AGENT_TYPE_LAST: break;
        }
        return "UNK";
    };

    switch(index)
    {
        case agent_indexing::node: return agent_index{"Agent", _agent->node_id, get_type()};
        case agent_indexing::logical_node_type:
            return agent_index{
                get_type(), static_cast<uint32_t>(_agent->logical_node_type_id), get_type()};

        case agent_indexing::logical_node:
        default:
            return agent_index{"Agent", static_cast<uint32_t>(_agent->logical_node_id), get_type()};
    }
}

const std::string*
metadata::get_string_entry(size_t key) const
{
    const auto* ret = string_entries.rlock(
        [](const auto& _data, size_t _key) -> const std::string* {
            if(_data.count(_key) > 0) return _data.at(_key).get();
            return nullptr;
        },
        key);

    if(!ret) ret = common::get_string_entry(key);

    return ret;
}

int64_t
metadata::get_instruction_index(rocprofiler_pc_t record)
{
    inst_t ins;
    ins.code_object_id     = record.code_object_id;
    ins.code_object_offset = record.code_object_offset;
    auto itr               = indexes.find(ins);
    if(itr != indexes.end()) return itr->second;
    auto idx            = instruction_decoder.size();
    auto pc_instruction = decode_instruction(record);
    instruction_decoder.emplace_back(pc_instruction->inst);
    instruction_comment.emplace_back(pc_instruction->comment);
    indexes.emplace(ins, idx);
    return idx;
}

void
metadata::add_decoder(rocprofiler_code_object_info_t* obj_data)
{
    if(obj_data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
    {
        decoder.wlock(
            [](auto& _decoder, rocprofiler_code_object_info_t* obj_data_v) {
                _decoder.addDecoder(obj_data_v->uri,
                                    obj_data_v->code_object_id,
                                    obj_data_v->load_delta,
                                    obj_data_v->load_size);
            },
            obj_data);
    }
    else
    {
        decoder.wlock(
            [](auto& _decoder, rocprofiler_code_object_info_t* obj_data_v) {
                _decoder.addDecoder(
                    // NOLINTBEGIN(performance-no-int-to-ptr)
                    reinterpret_cast<const void*>(obj_data_v->memory_base),
                    // NOLINTEND(performance-no-int-to-ptr)
                    obj_data_v->memory_size,
                    obj_data_v->code_object_id,
                    obj_data_v->load_delta,
                    obj_data_v->load_size);
            },
            obj_data);
    }
}

std::unique_ptr<instruction_t>
metadata::decode_instruction(rocprofiler_pc_t pc)
{
    return decoder.wlock(
        [](auto& _decoder, uint64_t id, uint64_t addr) { return _decoder.get(id, addr); },
        pc.code_object_id,
        pc.code_object_offset);
}

}  // namespace tool
}  // namespace rocprofiler
