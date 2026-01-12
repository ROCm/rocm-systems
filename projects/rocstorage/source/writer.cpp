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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocstorage/writer.hpp>

#include "data_storage/database.hpp"
#include "data_storage/insert_statements.hpp"
#include "data_storage/table_insert_query.hpp"

#include "debug.hpp"

#include <stdexcept>
#include <unordered_map>

namespace rocstorage {

namespace {
void initialize_metadata(
    const std::shared_ptr<data_storage::database> &database,
    const std::string &uuid) {
  data_storage::queries::table_insert_query query;
  database->execute_query(query.set_table_name("rocpd_metadata_" + uuid)
                              .set_columns("tag", "value")
                              .set_values("upid", uuid)
                              .get_query_string());
}

struct track_name_value {
  size_t track_id;
  size_t name_id;
};

struct pmc_identifier_key {
  size_t agent_id;
  std::string name;
};

struct pmc_identifier_hash {
  std::size_t operator()(const pmc_identifier_key &pmc) const noexcept {
    std::size_t h1 = std::hash<size_t>{}(pmc.agent_id);
    std::size_t h2 = std::hash<std::string>{}(pmc.name);
    return h1 ^ (h2 << 1);
  }
};

struct pmc_identifier_equal {
  bool operator()(const pmc_identifier_key &lhs,
                  const pmc_identifier_key &rhs) const noexcept {
    return lhs.agent_id == rhs.agent_id && lhs.name == rhs.name;
  }
};

} // namespace

struct writer::impl {
  struct data_identifiers {
    using track_name_key = std::string;
    using thread_id = size_t;
    using thread_primary_key = size_t;
    using pmc_info_primary_key = size_t;
    using string_key = std::string;
    using string_primary_key = size_t;

    ~data_identifiers() = default;

    std::unordered_map<track_name_key, track_name_value> m_tracks{};
    std::unordered_map<pmc_identifier_key, pmc_info_primary_key,
                       pmc_identifier_hash, pmc_identifier_equal>
        m_pmc_descriptor_map{};
    std::unordered_map<thread_id, thread_primary_key> m_thread_id_map{};
    std::unordered_map<string_key, string_primary_key> m_string_map{};
  };

  explicit impl(std::shared_ptr<data_storage::database> database,
                std::string uuid)
      : m_database(std::move(database)), m_uuid(std::move(uuid)),
        m_uuid_cstr(m_uuid.c_str()),
        m_data_identifiers(std::make_unique<data_identifiers>()) {
    if (!m_database) {
      throw std::invalid_argument(
          "Provided pointer to a non-existing database!");
    }

    if (m_uuid.empty()) {
      throw std::invalid_argument("Empty UUID provided!");
    }

    m_database->initialize_schema();
    initialize_metadata(m_database, m_uuid);
    m_insert_statements =
        std::make_unique<data_storage::insert_statements>(m_database, m_uuid);
  }

  size_t insert_string(const char *str) {
    auto it = m_data_identifiers->m_string_map.find(str);
    if (it != m_data_identifiers->m_string_map.end())
      return it->second;

    m_insert_statements->m_insert_string_statement(m_uuid_cstr, str);

    const auto string_id = m_database->get_last_insert_id();
    m_data_identifiers->m_string_map.emplace(str, string_id);
    return string_id;
  }

  void insert_node_info(size_t node_id, size_t hash, const char *machine_id,
                        const char *system_name, const char *hostname,
                        const char *release, const char *version,
                        const char *hardware_name, const char *domain_name) {
    data_storage::queries::table_insert_query query;
    m_database->execute_query(
        query.set_table_name("rocpd_info_node_" + m_uuid)
            .set_columns("id", "guid", "hash", "machine_id", "system_name",
                         "hostname", "release", "version", "hardware_name",
                         "domain_name")
            .set_values(node_id, m_uuid, hash, machine_id, system_name,
                        hostname, release, version, hardware_name, domain_name)
            .get_query_string());
  }

  void insert_process_info(size_t node_id, size_t ppid, size_t pid, size_t init,
                           size_t fini, size_t start, size_t end,
                           const char *command, const char *environment,
                           const char *extdata) {
    data_storage::queries::table_insert_query query;
    m_database->execute_query(
        query.set_table_name("rocpd_info_process_" + m_uuid)
            .set_columns("id", "guid", "nid", "ppid", "pid", "init", "fini",
                         "start", "end", "command", "environment", "extdata")
            .set_values(pid, m_uuid, node_id, ppid, pid, init, fini, start, end,
                        command, environment, extdata)
            .get_query_string());
  }

  size_t insert_agent(size_t node_id, size_t pid, const char *agent_type,
                      size_t absolute_index, size_t logical_index,
                      size_t type_index, size_t uuid, const char *name,
                      const char *model_name, const char *vendor_name,
                      const char *product_name, const char *user_name,
                      const char *extdata) {
    data_storage::queries::table_insert_query query;
    m_database->execute_query(
        query.set_table_name("rocpd_info_agent_" + m_uuid)
            .set_columns("guid", "nid", "pid", "type", "absolute_index",
                         "logical_index", "type_index", "uuid", "name",
                         "model_name", "vendor_name", "product_name",
                         "user_name", "extdata")
            .set_values(m_uuid, node_id, pid, agent_type, absolute_index,
                        logical_index, type_index, uuid, name, model_name,
                        vendor_name, product_name, user_name, extdata)
            .get_query_string());

    return m_database->get_last_insert_id();
  }

  void insert_track(const char *track_name, size_t node_id, size_t process_id,
                    std::optional<size_t> thread_id, const char *extdata) {
    if (m_data_identifiers->m_tracks.find(track_name) !=
        m_data_identifiers->m_tracks.end()) {
      LOG_ERROR("Failed to add track '{}': already exists", track_name);
      return;
    }

    auto name_id = insert_string(track_name);

    data_storage::queries::table_insert_query query;
    m_database->execute_query(
        query.set_table_name("rocpd_track_" + m_uuid)
            .set_columns("guid", "nid", "pid", "tid", "name_id", "extdata")
            .set_values(m_uuid, node_id, process_id, thread_id, name_id,
                        extdata)
            .get_query_string());

    auto track_id = m_database->get_last_insert_id();
    m_data_identifiers->m_tracks[track_name] =
        track_name_value{track_id, name_id};
  }

  size_t insert_event(size_t string_primary_key, size_t stack_id,
                      size_t parent_stack_id, size_t correlation_id,
                      const char *call_stack, const char *line_info,
                      const char *extdata) {
    m_insert_statements->m_insert_event_statement(
        m_uuid_cstr, string_primary_key, stack_id, parent_stack_id,
        correlation_id, call_stack, line_info, extdata);
    return m_database->get_last_insert_id();
  }

  void insert_pmc_event(size_t event_id, size_t agent_id,
                        const char *pmc_descriptor, double value,
                        const char *extdata) {
    auto it = m_data_identifiers->m_pmc_descriptor_map.find(
        {agent_id, pmc_descriptor});
    if (it == m_data_identifiers->m_pmc_descriptor_map.end()) {
      LOG_ERROR("Insert PMC event failed: non-existing PMC description "
                "(agent_id: {}, pmc_name: {})",
                agent_id, pmc_descriptor);
      return;
    }

    const auto pmc_description_id = it->second;
    m_insert_statements->m_insert_pmc_event_statement(
        m_uuid_cstr, event_id, pmc_description_id, value, extdata);
  }

  void insert_pmc_description(
      size_t node_id, size_t process_id, size_t agent_id,
      const char *target_arch, size_t event_code, size_t instance_id,
      const char *name, const char *symbol, const char *description,
      const char *long_description, const char *component, const char *units,
      const char *value_type, const char *block, const char *expression,
      uint32_t is_constant, uint32_t is_derived, const char *extdata) {
    auto it = m_data_identifiers->m_pmc_descriptor_map.find({agent_id, name});
    if (it != m_data_identifiers->m_pmc_descriptor_map.end()) {
      LOG_ERROR("Insert PMC description failed: PMC descriptor already "
                "exists (name: {}, agent_id: {})",
                name, agent_id);
      return;
    }
    data_storage::queries::table_insert_query query_builder;

    auto query =
        query_builder.set_table_name("rocpd_info_pmc_" + m_uuid)
            .set_columns("guid", "nid", "pid", "agent_id", "target_arch",
                         "event_code", "instance_id", "name", "symbol",
                         "description", "long_description", "component",
                         "units", "value_type", "block", "expression",
                         "is_constant", "is_derived", "extdata")
            .set_values(m_uuid, node_id, process_id, agent_id, target_arch,
                        event_code, instance_id, name, symbol, description,
                        long_description, component, units, value_type, block,
                        expression, is_constant, is_derived, extdata)
            .get_query_string();
    m_database->execute_query(query);

    auto pmc_id = m_database->get_last_insert_id();
    m_data_identifiers->m_pmc_descriptor_map.emplace(
        std::pair<pmc_identifier_key, size_t>{{agent_id, name}, pmc_id});
  }

  void insert_sample(const char *track, size_t timestamp, size_t event_id,
                     const char *extdata) {
    auto it = m_data_identifiers->m_tracks.find(track);
    if (it == m_data_identifiers->m_tracks.end()) {
      LOG_ERROR("Insert sample failed: track '{}' does not exist", track);
      return;
    }
    auto track_info = it->second;
    m_insert_statements->m_insert_sample_statement(
        m_uuid_cstr, track_info.track_id, timestamp, event_id, extdata);
  }

  void insert_region(size_t node_id, size_t process_id, size_t thread_id,
                     size_t start, size_t end, size_t name_id, size_t event_id,
                     const char *extdata) {
    m_insert_statements->m_insert_region_statement(
        m_uuid_cstr, node_id, process_id, thread_id, start, end, name_id,
        event_id, extdata);
  }

  size_t insert_thread_info(size_t node_id, size_t parent_process_id,
                            size_t process_id, size_t thread_id,
                            const char *name, size_t start, size_t end,
                            const char *extdata) {
    auto it = m_data_identifiers->m_thread_id_map.find(thread_id);

    if (it != m_data_identifiers->m_thread_id_map.end()) {
      return m_data_identifiers->m_thread_id_map.at(thread_id);
    }

    data_storage::queries::table_insert_query query;
    m_database->execute_query(
        query.set_table_name("rocpd_info_thread_" + m_uuid)
            .set_columns("guid", "nid", "ppid", "pid", "tid", "name", "start",
                         "end", "extdata")
            .set_values(m_uuid_cstr, node_id, parent_process_id, process_id,
                        thread_id, name, start, end, extdata)
            .get_query_string());

    auto thread_idx = m_database->get_last_insert_id();
    m_data_identifiers->m_thread_id_map.emplace(thread_id, thread_idx);
    return thread_idx;
  }

  void insert_stream_info(size_t stream_id, size_t node_id, size_t process_id,
                          const char *name, const char *extdata) {
    data_storage::queries::table_insert_query query;
    m_database->execute_query(
        query.set_table_name("rocpd_info_stream_" + m_uuid)
            .set_columns("id", "guid", "nid", "pid", "name", "extdata")
            .set_values(stream_id, m_uuid, node_id, process_id, name, extdata)
            .get_query_string());
  }

  void insert_queue_info(size_t queue_id, size_t node_id, size_t process_id,
                         const char *name, const char *extdata) {
    data_storage::queries::table_insert_query query;
    m_database->execute_query(
        query.set_table_name("rocpd_info_queue_" + m_uuid)
            .set_columns("id", "guid", "nid", "pid", "name", "extdata")
            .set_values(queue_id, m_uuid, node_id, process_id, name, extdata)
            .get_query_string());
  }

  void insert_kernel_dispatch(
      size_t node_id, size_t process_id, size_t thread_id, size_t agent_id,
      size_t kernel_id, size_t dispatch_id, size_t queue_id, size_t stream_id,
      size_t start, size_t end, size_t private_segment_size,
      size_t group_segment_size, size_t workgroup_size_x,
      size_t workgroup_size_y, size_t workgroup_size_z, size_t grid_size_x,
      size_t grid_size_y, size_t grid_size_z, size_t region_name_id,
      size_t event_id, const char *extdata) {
    m_insert_statements->m_insert_kernel_dispatch_statement(
        m_uuid_cstr, node_id, process_id, thread_id, agent_id, kernel_id,
        dispatch_id, queue_id, stream_id, start, end, private_segment_size,
        group_segment_size, workgroup_size_x, workgroup_size_y,
        workgroup_size_z, grid_size_x, grid_size_y, grid_size_z, region_name_id,
        event_id, extdata);
  }

  void insert_memory_copy(size_t node_id, size_t process_id, size_t thread_id,
                          size_t start, size_t end, size_t name_id,
                          size_t dst_agent_id, size_t dst_addr,
                          size_t src_agent_id, size_t src_addr, size_t size,
                          size_t queue_id, size_t stream_id,
                          size_t region_name_id, size_t event_id,
                          const char *extdata) {
    m_insert_statements->m_insert_memory_copy_statement(
        m_uuid_cstr, node_id, process_id, thread_id, start, end, name_id,
        dst_agent_id, dst_addr, src_agent_id, src_addr, size, queue_id,
        stream_id, region_name_id, event_id, extdata);
  }

  void insert_kernel_symbol(size_t id, size_t node_id, size_t process_id,
                            size_t code_obj_id, const char *name,
                            const char *display_name, uint32_t kernel_obj,
                            uint32_t kernarg_segmnt_size,
                            uint32_t kernarg_segment_alignment,
                            uint32_t group_segment_size,
                            uint32_t private_segment_size, uint32_t sgrp_count,
                            uint32_t arch_vgrp_count, uint32_t accum_vgrp_count,
                            const char *extdata) {
    m_insert_statements->m_insert_kernel_symbol_statement(
        id, m_uuid_cstr, node_id, process_id, code_obj_id, name, display_name,
        kernel_obj, kernarg_segmnt_size, kernarg_segment_alignment,
        group_segment_size, private_segment_size, sgrp_count, arch_vgrp_count,
        accum_vgrp_count, extdata);
  }

  void insert_code_object(size_t id, size_t node_id, size_t process_id,
                          size_t agent_id, const char *uri, size_t ld_base,
                          size_t ld_size, size_t ld_delta,
                          const char *storage_type, const char *extdata) {
    m_insert_statements->m_insert_code_object_statement(
        id, m_uuid_cstr, node_id, process_id, agent_id, uri, ld_base, ld_size,
        ld_delta, storage_type, extdata);
  }

  void insert_args(size_t event_id, size_t position, const char *type,
                   const char *name, const char *value, const char *extdata) {
    m_insert_statements->m_insert_args_statement(
        m_uuid_cstr, event_id, position, type, name, value, extdata);
  }

  void insert_memory_alloc(size_t node_id, size_t process_id, size_t thread_id,
                           std::optional<size_t> agent_id, const char *type,
                           const char *level, size_t start, size_t end,
                           size_t address, size_t size, size_t queue_id,
                           size_t stream_id, size_t event_id,
                           const char *extdata) {
    if (agent_id.has_value()) {
      m_insert_statements->m_insert_memory_alloc_statement(
          m_uuid_cstr, node_id, process_id, thread_id, agent_id.value(), type,
          level, start, end, address, size, queue_id, stream_id, event_id,
          extdata);
    } else {
      m_insert_statements->m_insert_memory_alloc_no_agent_statement(
          m_uuid_cstr, node_id, process_id, thread_id, type, level, start, end,
          address, size, queue_id, stream_id, event_id, extdata);
    }
  }

  size_t map_thread_id_to_primary_key(size_t thread_id) {
    auto it = m_data_identifiers->m_thread_id_map.find(thread_id);

    if (it == m_data_identifiers->m_thread_id_map.end()) {
      throw std::invalid_argument("Given thread id don't exist");
    }
    return m_data_identifiers->m_thread_id_map.at(thread_id);
  }

  void flush() { m_database->flush(); }

  std::shared_ptr<data_storage::database> m_database;
  std::string m_uuid;
  const char *m_uuid_cstr;
  std::unique_ptr<data_storage::insert_statements> m_insert_statements;
  std::unique_ptr<data_identifiers> m_data_identifiers;
};

// ----------------------- PUBLIC API -------------------------

writer::writer(std::shared_ptr<data_storage::database> database,
               std::string uuid)
    : m_impl(std::make_unique<impl>(std::move(database), std::move(uuid))) {}

writer::~writer() = default;

size_t writer::insert_string(const char *str) {
  return m_impl->insert_string(str);
}

void writer::insert_node_info(size_t node_id, size_t hash,
                              const char *machine_id, const char *system_name,
                              const char *hostname, const char *release,
                              const char *version, const char *hardware_name,
                              const char *domain_name) {
  m_impl->insert_node_info(node_id, hash, machine_id, system_name, hostname,
                           release, version, hardware_name, domain_name);
}

void writer::insert_process_info(size_t node_id, size_t ppid, size_t pid,
                                 size_t init, size_t fini, size_t start,
                                 size_t end, const char *command,
                                 const char *environment, const char *extdata) {
  m_impl->insert_process_info(node_id, ppid, pid, init, fini, start, end,
                              command, environment, extdata);
}

size_t writer::insert_agent(size_t node_id, size_t pid, const char *agent_type,
                            size_t absolute_index, size_t logical_index,
                            size_t type_index, uint64_t uuid, const char *name,
                            const char *model_name, const char *vendor_name,
                            const char *product_name, const char *user_name,
                            const char *extdata) {
  return m_impl->insert_agent(node_id, pid, agent_type, absolute_index,
                              logical_index, type_index, uuid, name, model_name,
                              vendor_name, product_name, user_name, extdata);
}

void writer::insert_track(const char *track_name, size_t node_id,
                          size_t process_id, std::optional<size_t> thread_id,
                          const char *extdata) {
  m_impl->insert_track(track_name, node_id, process_id, thread_id, extdata);
}

void writer::insert_pmc_description(
    size_t node_id, size_t process_id, size_t agent_id, const char *target_arch,
    size_t event_code, size_t instance_id, const char *name, const char *symbol,
    const char *description, const char *long_description,
    const char *component, const char *units, const char *value_type,
    const char *block, const char *expression, uint32_t is_constant,
    uint32_t is_derived, const char *extdata) {
  m_impl->insert_pmc_description(
      node_id, process_id, agent_id, target_arch, event_code, instance_id, name,
      symbol, description, long_description, component, units, value_type,
      block, expression, is_constant, is_derived, extdata);
}

void writer::insert_pmc_event(size_t event_id, size_t agent_id,
                              const char *pmc_name, double value,
                              const char *extdata) {
  m_impl->insert_pmc_event(event_id, agent_id, pmc_name, value, extdata);
}

void writer::insert_sample(const char *track, uint64_t timestamp,
                           size_t event_id, const char *extdata) {
  m_impl->insert_sample(track, timestamp, event_id, extdata);
}

size_t writer::insert_event(size_t string_primary_key, size_t stack_id,
                            size_t parent_stack_id, size_t correlation_id,
                            const char *call_stack, const char *line_info,
                            const char *extdata) {
  return m_impl->insert_event(string_primary_key, stack_id, parent_stack_id,
                              correlation_id, call_stack, line_info, extdata);
}

void writer::insert_args(size_t event_id, size_t position, const char *type,
                         const char *name, const char *value,
                         const char *extdata) {
  m_impl->insert_args(event_id, position, type, name, value, extdata);
}

void writer::insert_stream_info(size_t stream_id, size_t node_id,
                                size_t process_id, const char *name,
                                const char *extdata) {
  m_impl->insert_stream_info(stream_id, node_id, process_id, name, extdata);
}

void writer::insert_queue_info(size_t queue_id, size_t node_id,
                               size_t process_id, const char *name,
                               const char *extdata) {
  m_impl->insert_queue_info(queue_id, node_id, process_id, name, extdata);
}

void writer::insert_code_object(size_t id, size_t node_id, size_t process_id,
                                size_t agent_id, const char *uri,
                                uint64_t ld_base, uint64_t ld_size,
                                uint64_t ld_delta, const char *storage_type,
                                const char *extdata) {
  m_impl->insert_code_object(id, node_id, process_id, agent_id, uri, ld_base,
                             ld_size, ld_delta, storage_type, extdata);
}

void writer::insert_kernel_symbol(
    size_t id, size_t node_id, size_t process_id, uint64_t code_obj_id,
    const char *name, const char *display_name, uint32_t kernel_obj,
    uint32_t kernarg_segmnt_size, uint32_t kernarg_segment_alignment,
    uint32_t group_segment_size, uint32_t private_segment_size,
    uint32_t sgrp_count, uint32_t arch_vgrp_count, uint32_t accum_vgrp_count,
    const char *extdata) {
  m_impl->insert_kernel_symbol(id, node_id, process_id, code_obj_id, name,
                               display_name, kernel_obj, kernarg_segmnt_size,
                               kernarg_segment_alignment, group_segment_size,
                               private_segment_size, sgrp_count,
                               arch_vgrp_count, accum_vgrp_count, extdata);
}

void writer::insert_region(size_t node_id, size_t process_id, size_t thread_id,
                           uint64_t start, uint64_t end, size_t name_id,
                           size_t event_id, const char *extdata) {
  m_impl->insert_region(node_id, process_id, thread_id, start, end, name_id,
                        event_id, extdata);
}

void writer::insert_kernel_dispatch(
    size_t node_id, size_t process_id, size_t thread_id, size_t agent_id,
    size_t kernel_id, size_t dispatch_id, size_t queue_id, size_t stream_id,
    uint64_t start, uint64_t end, size_t private_segment_size,
    size_t group_segment_size, size_t workgroup_size_x, size_t workgroup_size_y,
    size_t workgroup_size_z, size_t grid_size_x, size_t grid_size_y,
    size_t grid_size_z, size_t region_name_id, size_t event_id,
    const char *extdata) {
  m_impl->insert_kernel_dispatch(
      node_id, process_id, thread_id, agent_id, kernel_id, dispatch_id,
      queue_id, stream_id, start, end, private_segment_size, group_segment_size,
      workgroup_size_x, workgroup_size_y, workgroup_size_z, grid_size_x,
      grid_size_y, grid_size_z, region_name_id, event_id, extdata);
}

void writer::insert_memory_copy(size_t node_id, size_t process_id,
                                size_t thread_id, uint64_t start, uint64_t end,
                                size_t name_id, size_t dst_agent_id,
                                size_t dst_addr, size_t src_agent_id,
                                size_t src_addr, size_t size, size_t queue_id,
                                size_t stream_id, size_t region_name_id,
                                size_t event_id, const char *extdata) {
  m_impl->insert_memory_copy(node_id, process_id, thread_id, start, end,
                             name_id, dst_agent_id, dst_addr, src_agent_id,
                             src_addr, size, queue_id, stream_id,
                             region_name_id, event_id, extdata);
}

void writer::insert_memory_alloc(size_t node_id, size_t process_id,
                                 size_t thread_id,
                                 std::optional<size_t> agent_id,
                                 const char *type, const char *level,
                                 uint64_t start, uint64_t end, size_t address,
                                 size_t size, size_t queue_id, size_t stream_id,
                                 size_t event_id, const char *extdata) {
  m_impl->insert_memory_alloc(node_id, process_id, thread_id, agent_id, type,
                              level, start, end, address, size, queue_id,
                              stream_id, event_id, extdata);
}

size_t writer::insert_thread_info(size_t node_id, size_t parent_process_id,
                                  size_t process_id, size_t thread_id,
                                  const char *name, uint64_t start,
                                  uint64_t end, const char *extdata) {
  return m_impl->insert_thread_info(node_id, parent_process_id, process_id,
                                    thread_id, name, start, end, extdata);
}

size_t writer::map_thread_id_to_primary_key(size_t thread_id) {
  return m_impl->map_thread_id_to_primary_key(thread_id);
}

void writer::flush() { m_impl->flush(); }

} // namespace rocstorage
