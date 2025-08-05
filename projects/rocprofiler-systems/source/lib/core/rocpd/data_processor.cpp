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

#include "data_processor.hpp"
#include "core/rocpd/data_storage/database.hpp"
#include "core/rocpd/data_storage/table_insert_query.hpp"
#include "debug.hpp"

namespace rocprofsys
{
namespace rocpd
{
data_processor::data_processor()
{
    data_storage::database::get_instance().initialize_schema();
    _upid = data_storage::database::get_instance().get_upid();

    // Initialize event statement
    initialize_event_stmt();
    initialize_pmc_event_stmt();
    initialize_sample_stmt();
    initialize_region_stmt();
    initialize_kernel_dispatch_stmt();
    initialize_memory_copy_stmt();
    initialize_code_object_stmt();
    initialize_kernel_symbol_stmt();
    initialize_metadata();
    initialize_args_stmt();
    initialize_memory_alloc_stmt();
}

data_processor&
data_processor::get_instance()
{
    static data_processor _instance;
    return _instance;
}

void
data_processor::initialize_metadata()
{
    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_metadata_" + _upid)
            .set_columns("tag", "value")
            .set_values("upid", _upid)
            .get_query_string());
}

size_t
data_processor::insert_string(const char* str)
{
    std::lock_guard<std::mutex> lock(_data_mutex);
    auto                        it = _string_map.find(str);
    if(it != _string_map.end()) return _string_map.at(str);

    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_string_" + _upid)
            .set_columns("guid", "string")
            .set_values(_upid, str)
            .get_query_string());

    const auto string_id = data_storage::database::get_instance().get_last_insert_id();
    _string_map.emplace(str, string_id);
    return string_id;
}

void
data_processor::insert_node_info(size_t node_id, size_t hash, const char* machine_id,
                                 const char* system_name, const char* hostname,
                                 const char* release, const char* version,
                                 const char* hardware_name, const char* domain_name)
{
    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_info_node_" + _upid)
            .set_columns("id", "guid", "hash", "machine_id", "system_name", "hostname",
                         "release", "version", "hardware_name", "domain_name")
            .set_values(node_id, _upid, hash, machine_id, system_name, hostname, release,
                        version, hardware_name, domain_name)
            .get_query_string());
}

void
data_processor::insert_process_info(size_t nid, size_t ppid, size_t pid, size_t init,
                                    size_t fini, size_t start, size_t end,
                                    const char* command, const char* environment,
                                    const char* extdata)
{
    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_info_process_" + _upid)
            .set_columns("id", "guid", "nid", "ppid", "pid", "init", "fini", "start",
                         "end", "command", "environment", "extdata")
            .set_values(pid, _upid, nid, ppid, pid, init, fini, start, end, command,
                        environment, extdata)
            .get_query_string());
}

size_t
data_processor::insert_agent(size_t node_id, size_t pid, const char* agent_type,
                             size_t absolute_index, size_t logical_index,
                             size_t type_index, uint64_t uuid, const char* name,
                             const char* model_name, const char* vendor_name,
                             const char* product_name, const char* user_name,
                             const char* extdata)
{
    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_info_agent_" + _upid)
            .set_columns("guid", "nid", "pid", "type", "absolute_index", "logical_index",
                         "type_index", "uuid", "name", "model_name", "vendor_name",
                         "product_name", "user_name", "extdata")
            .set_values(_upid, node_id, pid, agent_type, absolute_index, logical_index,
                        type_index, uuid, name, model_name, vendor_name, product_name,
                        user_name, extdata)
            .get_query_string());

    return data_storage::database::get_instance().get_last_insert_id();
}

void
data_processor::insert_track(const char* track_name, size_t node_id, size_t process_id,
                             std::optional<size_t> thread_id, const char* extdata)
{
    if(_tracks.find(track_name) != _tracks.end())
    {
        ROCPROFSYS_WARNING(2, "Fail to add track %s, already exist!\n", track_name);
        return;
    }

    auto name_id = insert_string(track_name);

    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_track_" + _upid)
            .set_columns("guid", "nid", "pid", "tid", "name_id", "extdata")
            .set_values(_upid, node_id, process_id, thread_id, name_id, extdata)
            .get_query_string());

    auto track_id       = data_storage::database::get_instance().get_last_insert_id();
    _tracks[track_name] = track_name_map{ track_id, name_id };
}

void
data_processor::insert_pmc_description(
    size_t node_id, size_t process_id, size_t agent_id, const char* target_arch,
    size_t event_code, size_t instance_id, const char* name, const char* symbol,
    const char* description, const char* long_description, const char* component,
    const char* units, const char* value_type, const char* block, const char* expression,
    uint32_t is_constant, uint32_t is_derived, const char* extdata)
{
    auto it = _pmc_descriptor_map.find({ agent_id, name });
    if(it != _pmc_descriptor_map.end())
    {
        ROCPROFSYS_WARNING(0,
                           "Insert PMC description failed! Error: PMC descriptor "
                           "(name:%s) (ID:%lu) already exist!\n",
                           name, agent_id);
        return;
    }
    data_storage::queries::table_insert_query query_builder;

    auto query =
        query_builder.set_table_name("rocpd_info_pmc_" + _upid)
            .set_columns("guid", "nid", "pid", "agent_id", "target_arch", "event_code",
                         "instance_id", "name", "symbol", "description",
                         "long_description", "component", "units", "value_type", "block",
                         "expression", "is_constant", "is_derived", "extdata")
            .set_values(_upid, node_id, process_id, agent_id, target_arch, event_code,
                        instance_id, name, symbol, description, long_description,
                        component, units, value_type, block, expression, is_constant,
                        is_derived, extdata)
            .get_query_string();
    data_storage::database::get_instance().execute_query(query);

    auto pmc_id = data_storage::database::get_instance().get_last_insert_id();
    _pmc_descriptor_map.emplace(
        std::pair<pmc_identifier, size_t>{ { agent_id, name }, pmc_id });
}

void
data_processor::insert_pmc_event(size_t event_id, size_t agent_id, const char* pmc_name,
                                 double value, const char* extdata)
{
    ROCPROFSYS_VERBOSE(2,
                       "Insert PMC event: id %ld, agent id: %ld, pmc name: %s, value: "
                       "%lf, extdata: %s\n",
                       event_id, agent_id, pmc_name, value, extdata);
    auto it = _pmc_descriptor_map.find({ agent_id, pmc_name });
    if(it == _pmc_descriptor_map.end())
    {
        ROCPROFSYS_WARNING(0,
                           "Insert PMC event failed! Error: non-existing PMC description "
                           "agent id: %ld, pmc name: %s !\n",
                           agent_id, pmc_name);
        return;
    }

    const auto pmc_description_id = it->second;
    _insert_pmc_event_statement(_upid.c_str(), event_id, pmc_description_id, value,
                                extdata);
}

void
data_processor::insert_sample(const char* track, uint64_t timestamp, size_t event_id,
                              const char* extdata)
{
    ROCPROFSYS_VERBOSE(
        3, "Insert sample: track: %s, timestamp: %lu, event id: %ld, extdata: %s\n",
        track, timestamp, event_id, extdata);
    auto it = _tracks.find(track);
    if(it == _tracks.end())
    {
        ROCPROFSYS_WARNING(0, "Insert sample failed! Error: Unexisting track %s!\n",
                           track);
        return;
    }
    auto track_info = it->second;
    _insert_sample_statement(_upid.c_str(), track_info.track_id, timestamp, event_id,
                             extdata);
}

size_t
data_processor::insert_event(size_t category_id, size_t stack_id, size_t parent_stack_id,
                             size_t correlation_id, const char* call_stack,
                             const char* line_info, const char* extdata)
{
    std::lock_guard<std::mutex> lock(_data_mutex);
    auto                        it = _category_map.find(category_id);
    if(it == _category_map.end())
    {
        std::ostringstream oss;
        oss << "Insert event failed! Error: Unknown category id: " << category_id
            << " for UPID: " << _upid;
        throw std::runtime_error(oss.str());
    }

    ROCPROFSYS_VERBOSE(3, "Insert event category id: %ld, string id: %ld\n", category_id,
                       it->second);

    _insert_event_statement(_upid.c_str(), it->second, stack_id, parent_stack_id,
                            correlation_id, call_stack, line_info, extdata);
    return data_storage::database::get_instance().get_last_insert_id();
}

void
data_processor::initialize_event_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_event_" + _upid)
                     .set_columns("guid", "category_id", "stack_id", "parent_stack_id",
                                  "correlation_id", "call_stack", "line_info", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?', '?')
                     .get_query_string();
    _insert_event_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, size_t, size_t, size_t,
                                       const char*, const char*, const char*>(query);
}

void
data_processor::initialize_pmc_event_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_pmc_event_" + _upid)
                     .set_columns("guid", "event_id", "pmc_id", "value", "extdata")
                     .set_values('?', '?', '?', '?', '?')
                     .get_query_string();
    _insert_pmc_event_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, size_t, double, const char*>(
                query);
}

void
data_processor::initialize_sample_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_sample_" + _upid)
                     .set_columns("guid", "track_id", "timestamp", "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?')
                     .get_query_string();
    _insert_sample_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, uint64_t, size_t,
                                       const char*>(query);
}

void
data_processor::initialize_region_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_region_" + _upid)
                     .set_columns("guid", "nid", "pid", "tid", "start", "end", "name_id",
                                  "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?')
                     .get_query_string();
    _insert_region_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, size_t, size_t, uint64_t,
                                       uint64_t, size_t, size_t, const char*>(query);
}

void
data_processor::initialize_kernel_dispatch_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_kernel_dispatch_" + _upid)
                     .set_columns("guid", "nid", "pid", "tid", "agent_id", "kernel_id",
                                  "dispatch_id", "queue_id", "stream_id", "start", "end",
                                  "private_segment_size", "group_segment_size",
                                  "workgroup_size_x", "workgroup_size_y",
                                  "workgroup_size_z", "grid_size_x", "grid_size_y",
                                  "grid_size_z", "region_name_id", "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
                                 '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
                     .get_query_string();
    _insert_kernel_dispatch_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, size_t, size_t, size_t,
                                       size_t, size_t, size_t, size_t, uint64_t, uint64_t,
                                       size_t, size_t, size_t, size_t, size_t, size_t,
                                       size_t, size_t, size_t, size_t, const char*>(
                query);
}

void
data_processor::initialize_memory_copy_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_memory_copy_" + _upid)
                     .set_columns("guid", "nid", "pid", "tid", "start", "end", "name_id",
                                  "dst_agent_id", "dst_address", "src_agent_id",
                                  "src_address", "size", "queue_id", "stream_id",
                                  "region_name_id", "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
                                 '?', '?', '?', '?', '?', '?')
                     .get_query_string();
    _insert_memory_copy_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, size_t, size_t, uint64_t,
                                       uint64_t, size_t, size_t, size_t, size_t, size_t,
                                       size_t, size_t, size_t, size_t, size_t,
                                       const char*>(query);
}

void
data_processor::initialize_kernel_symbol_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto                                      query =
        query_builder.set_table_name("rocpd_info_kernel_symbol_" + _upid)
            .set_columns("id", "guid", "nid", "pid", "code_object_id", "kernel_name",
                         "display_name", "kernel_object", "kernarg_segment_size",
                         "kernarg_segment_alignment", "group_segment_size",
                         "private_segment_size", "sgpr_count", "arch_vgpr_count",
                         "accum_vgpr_count", "extdata")
            .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
                        '?', '?', '?')
            .get_query_string();
    _insert_kernel_symbol_statement =
        data_storage::database::get_instance()
            .create_statement_executor<size_t, const char*, size_t, size_t, uint64_t,
                                       const char*, const char*, uint64_t, uint32_t,
                                       uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                       uint32_t, const char*>(query);
}

void
data_processor::initialize_code_object_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto                                      query =
        query_builder.set_table_name("rocpd_info_code_object_" + _upid)
            .set_columns("id", "guid", "nid", "pid", "agent_id", "uri", "load_base",
                         "load_size", "load_delta", "storage_type", "extdata")
            .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?')
            .get_query_string();
    _insert_code_object_statement =
        data_storage::database::get_instance()
            .create_statement_executor<size_t, const char*, size_t, size_t, size_t,
                                       const char*, uint64_t, uint64_t, uint64_t,
                                       const char*, const char*>(query);
}

void
data_processor::initialize_args_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_arg_" + _upid)
                     .set_columns("guid", "event_id", "position", "type", "name", "value",
                                  "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?')
                     .get_query_string();
    _insert_args_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, size_t, const char*,
                                       const char*, const char*, const char*>(query);
}

void
data_processor::initialize_memory_alloc_stmt()
{
    data_storage::queries::table_insert_query query_builder;
    auto query = query_builder.set_table_name("rocpd_memory_allocate_" + _upid)
                     .set_columns("guid", "nid", "pid", "tid", "agent_id", "type",
                                  "level", "start", "end", "address", "size", "queue_id",
                                  "stream_id", "event_id", "extdata")
                     .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
                                 '?', '?', '?', '?')
                     .get_query_string();
    _insert_memory_alloc_statement =
        data_storage::database::get_instance()
            .create_statement_executor<
                const char*, size_t, size_t, size_t, size_t, const char*, const char*,
                uint64_t, uint64_t, size_t, size_t, size_t, size_t, size_t, const char*>(
                query);

    // Statement without agent_id
    query = query_builder.set_table_name("rocpd_memory_allocate_" + _upid)
                .set_columns("guid", "nid", "pid", "tid", "type", "level", "start", "end",
                             "address", "size", "queue_id", "stream_id", "event_id",
                             "extdata")
                .set_values('?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?',
                            '?', '?')
                .get_query_string();
    _insert_memory_alloc_no_agent_statement =
        data_storage::database::get_instance()
            .create_statement_executor<const char*, size_t, size_t, size_t, const char*,
                                       const char*, uint64_t, uint64_t, size_t, size_t,
                                       size_t, size_t, size_t, const char*>(query);
}

void
data_processor::insert_args(size_t event_id, size_t position, const char* type,
                            const char* name, const char* value, const char* extdata)
{
    std::lock_guard<std::mutex> lock(_data_mutex);
    _insert_args_statement(_upid.c_str(), event_id, position, type, name, value, extdata);
}

void
data_processor::insert_stream_info(size_t stream_id, size_t node_id, size_t process_id,
                                   const char* name, const char* extdata)
{
    if(_stream_ids.count(stream_id) > 0)
    {
        // ROCPROFSYS_WARNING(
        //     1, "Insert stream info failed! Error: Stream ID %ld already exists!\n",
        //     stream_id);
        return;
    }
    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_info_stream_" + _upid)
            .set_columns("id", "guid", "nid", "pid", "name", "extdata")
            .set_values(stream_id, _upid, node_id, process_id, name, extdata)
            .get_query_string());
    _stream_ids.insert(stream_id);
}

void
data_processor::insert_queue_info(size_t queue_id, size_t node_id, size_t process_id,
                                  const char* name, const char* extdata)
{
    if(_queue_ids.count(queue_id) > 0)
    {
        // ROCPROFSYS_WARNING(
        //     1, "Insert queue info failed! Error: Queue ID %ld already exists!\n",
        //     queue_id);
        return;
    }
    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_info_queue_" + _upid)
            .set_columns("id", "guid", "nid", "pid", "name", "extdata")
            .set_values(queue_id, _upid, node_id, process_id, name, extdata)
            .get_query_string());
    _queue_ids.insert(queue_id);
}

void
data_processor::insert_code_object(size_t id, size_t node_id, size_t process_id,
                                   size_t agent_id, const char* uri, uint64_t ld_base,
                                   uint64_t ld_size, uint64_t ld_delta,
                                   const char* storage_type, const char* extdata)
{
    if(_code_object_ids.count(id) > 0)
    {
        // ROCPROFSYS_WARNING(
        //     1,
        //     "Insert code object info failed! Error: Code object ID %ld already
        //     exists!\n", id);
        return;
    }
    ROCPROFSYS_VERBOSE(2, "Insert code object with ID: %ld\n", id);
    std::lock_guard<std::mutex> lock(_data_mutex);
    _insert_code_object_statement(id, _upid.c_str(), node_id, process_id, agent_id, uri,
                                  ld_base, ld_size, ld_delta, storage_type, extdata);

    _code_object_ids.insert(id);
}

void
data_processor::insert_kernel_symbol(
    size_t id, size_t node_id, size_t process_id, uint64_t code_obj_id, const char* name,
    const char* display_name, uint32_t kernel_obj, uint32_t kernarg_segmnt_size,
    uint32_t kernarg_segment_alignment, uint32_t group_segment_size,
    uint32_t private_segment_size, uint32_t sgrp_count, uint32_t arch_vgrp_count,
    uint32_t accum_vgrp_count, const char* extdata)
{
    if(_kernel_sym_ids.count(id) > 0)
    {
        // ROCPROFSYS_WARNING(
        //     1,
        //     "Insert kernel symbol failed! Error: Kernel symbol ID %ld already
        //     exists!\n", id);
        return;
    }

    ROCPROFSYS_VERBOSE(2, "Insert kernel symbol: %s with ID: %ld\n", name, id);
    std::lock_guard<std::mutex> lock(_data_mutex);
    _insert_kernel_symbol_statement(
        id, _upid.c_str(), node_id, process_id, code_obj_id, name, display_name,
        kernel_obj, kernarg_segmnt_size, kernarg_segment_alignment, group_segment_size,
        private_segment_size, sgrp_count, arch_vgrp_count, accum_vgrp_count, extdata);

    _kernel_sym_ids.insert(id);
}

void
data_processor::insert_category(size_t category_id, const char* name)
{
    auto it = _category_map.find(category_id);
    if(it != _category_map.end())
    {
        // ROCPROFSYS_WARNING(
        //     1, "Insert category failed! Error: Category %s already exist!\n", name);
        return;
    }
    auto                        name_id = insert_string(name);
    std::lock_guard<std::mutex> lock(_data_mutex);
    ROCPROFSYS_VERBOSE(2, "Insert category: name: %s, id: %ld, name id: %ld\n", name,
                       category_id, name_id);
    _category_map.emplace(category_id, name_id);
}

void
data_processor::insert_region(size_t node_id, size_t process_id, size_t thread_id,
                              uint64_t start, uint64_t end, size_t name_id,
                              size_t event_id, const char* extdata)
{
    std::lock_guard<std::mutex> lock(_data_mutex);
    ROCPROFSYS_VERBOSE(2, "Insert region for event id: %ld\n", event_id);

    _insert_region_statement(_upid.c_str(), node_id, process_id, thread_id, start, end,
                             name_id, event_id, extdata);
}

void
data_processor::insert_kernel_dispatch(
    size_t node_id, size_t process_id, size_t thread_id, size_t agent_id,
    size_t kernel_id, size_t dispatch_id, size_t queue_id, size_t stream_id,
    uint64_t start, uint64_t end, size_t private_segment_size, size_t group_segment_size,
    size_t workgroup_size_x, size_t workgroup_size_y, size_t workgroup_size_z,
    size_t grid_size_x, size_t grid_size_y, size_t grid_size_z, size_t region_name_id,
    size_t event_id, const char* extdata)
{
    std::lock_guard<std::mutex> lock(_data_mutex);

    ROCPROFSYS_VERBOSE(2, "Insert kernel dispatch for event id: %ld\n", event_id);

    _insert_kernel_dispatch_statement(
        _upid.c_str(), node_id, process_id, thread_id, agent_id, kernel_id, dispatch_id,
        queue_id, stream_id, start, end, private_segment_size, group_segment_size,
        workgroup_size_x, workgroup_size_y, workgroup_size_z, grid_size_x, grid_size_y,
        grid_size_z, region_name_id, event_id, extdata);
}

void
data_processor::insert_memory_copy(size_t node_id, size_t process_id, size_t thread_id,
                                   uint64_t start, uint64_t end, size_t name_id,
                                   size_t dst_agent_id, size_t dst_addr,
                                   size_t src_agent_id, size_t src_addr, size_t size,
                                   size_t queue_id, size_t stream_id,
                                   size_t region_name_id, size_t event_id,
                                   const char* extdata)
{
    std::lock_guard<std::mutex> lock(_data_mutex);

    _insert_memory_copy_statement(_upid.c_str(), node_id, process_id, thread_id, start,
                                  end, name_id, dst_agent_id, dst_addr, src_agent_id,
                                  src_addr, size, queue_id, stream_id, region_name_id,
                                  event_id, extdata);
}

void
data_processor::insert_memory_alloc(size_t node_id, size_t process_id, size_t thread_id,
                                    std::optional<size_t> agent_id, const char* type,
                                    const char* level, uint64_t start, uint64_t end,
                                    size_t address, size_t size, size_t queue_id,
                                    size_t stream_id, size_t event_id,
                                    const char* extdata)
{
    if(agent_id.has_value())
    {
        _insert_memory_alloc_statement(_upid.c_str(), node_id, process_id, thread_id,
                                       agent_id.value(), type, level, start, end, address,
                                       size, queue_id, stream_id, event_id, extdata);
    }
    else
    {
        _insert_memory_alloc_no_agent_statement(
            _upid.c_str(), node_id, process_id, thread_id, type, level, start, end,
            address, size, queue_id, stream_id, event_id, extdata);
    }
}

size_t
data_processor::insert_thread_info(size_t node_id, size_t parent_process_id,
                                   size_t process_id, size_t thread_id, const char* name,
                                   uint64_t start, uint64_t end, const char* extdata)
{
    auto it = _thread_id_map.find(thread_id);

    if(it != _thread_id_map.end())
    {
        return _thread_id_map.at(thread_id);
    }

    data_storage::queries::table_insert_query query;
    data_storage::database::get_instance().execute_query(
        query.set_table_name("rocpd_info_thread_" + _upid)
            .set_columns("guid", "nid", "ppid", "pid", "tid", "name", "start", "end",
                         "extdata")
            .set_values(_upid.c_str(), node_id, parent_process_id, process_id, thread_id,
                        name, start, end, extdata)
            .get_query_string());

    auto thread_idx = data_storage::database::get_instance().get_last_insert_id();
    _thread_id_map.emplace(thread_id, thread_idx);
    return thread_idx;
}

void
data_processor::flush()
{
    // Flush all pending data to the database
    data_storage::database::get_instance().flush();
}

}  // namespace rocpd
}  // namespace rocprofsys
