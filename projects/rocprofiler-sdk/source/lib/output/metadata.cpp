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

#include "metadata.hpp"

#include "lib/common/string_entry.hpp"
#include "lib/output/agent_info.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <memory>

namespace rocprofiler
{
namespace tool
{
namespace
{
rocprofiler_status_t
dimensions_info_callback(rocprofiler_counter_id_t /*id*/,
                         const rocprofiler_record_dimension_info_t* dim_info,
                         long unsigned int                          num_dims,
                         void*                                      user_data)
{
    auto* dimensions_info = static_cast<counter_dimension_info_vec_t*>(user_data);
    dimensions_info->reserve(num_dims);
    for(size_t j = 0; j < num_dims; j++)
        dimensions_info->emplace_back(dim_info[j]);
    return ROCPROFILER_STATUS_SUCCESS;
}
}  // namespace

kernel_symbol_info::kernel_symbol_info()
: base_type{0, 0, 0, "", 0, 0, 0, 0, 0, 0, 0, 0}
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
            if(itr.type == ROCPROFILER_AGENT_TYPE_GPU) _gpu_agents.emplace_back(&itr);
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
}

void metadata::init(inprocess)
{
    if(inprocess_init) return;

    inprocess_init = true;
    for(auto itr : agents)
    {
        if(itr.type == ROCPROFILER_AGENT_TYPE_CPU) continue;

        ROCPROFILER_CHECK(rocprofiler_iterate_agent_supported_counters(
            itr.id,
            [](rocprofiler_agent_id_t    id,
               rocprofiler_counter_id_t* counters,
               size_t                    num_counters,
               void*                     user_data) {
                auto* data_v = static_cast<agent_counter_info_map_t*>(user_data);
                data_v->emplace(id, counter_info_vec_t{});
                for(size_t i = 0; i < num_counters; ++i)
                {
                    auto _info     = rocprofiler_counter_info_v0_t{};
                    auto _dim_ids  = std::vector<rocprofiler_counter_dimension_id_t>{};
                    auto _dim_info = std::vector<rocprofiler_record_dimension_info_t>{};

                    ROCPROFILER_CHECK(rocprofiler_query_counter_info(
                        counters[i],
                        ROCPROFILER_COUNTER_INFO_VERSION_0,
                        &static_cast<rocprofiler_counter_info_v0_t&>(_info)));

                    ROCPROFILER_CHECK(rocprofiler_iterate_counter_dimensions(
                        counters[i], dimensions_info_callback, &_dim_info));

                    _dim_ids.reserve(_dim_info.size());
                    for(auto ditr : _dim_info)
                        _dim_ids.emplace_back(ditr.id);

                    data_v->at(id).emplace_back(
                        id, _info, std::move(_dim_ids), std::move(_dim_info));
                }
                return ROCPROFILER_STATUS_SUCCESS;
            },
            &agent_counter_info));
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

const kernel_symbol_info*
metadata::get_kernel_symbol(uint64_t kernel_id) const
{
    return kernel_symbols.rlock([kernel_id](const auto& _data) -> const kernel_symbol_info* {
        return &_data.at(kernel_id);
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

uint64_t
metadata::get_node_id(rocprofiler_agent_id_t _val) const
{
    return CHECK_NOTNULL(get_agent(_val))->logical_node_id;
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
}  // namespace tool
}  // namespace rocprofiler
