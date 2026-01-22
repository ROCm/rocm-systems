// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <rocstorage/writer.hpp>

#include "data_storage/database.hpp"
#include "data_storage/insert_statements.hpp"
#include "data_storage/table_insert_query.hpp"

#include "debug.hpp"

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace rocstorage
{

namespace
{
void
initialize_metadata(const std::shared_ptr<data_storage::database>& database,
                    const std::string&                             uuid)
{
    data_storage::queries::table_insert_query query;
    database->execute_query(query.set_table_name("rocpd_metadata_" + uuid)
                                .set_columns("tag", "value")
                                .set_values("upid", uuid)
                                .get_query_string());
}

using primary_key = size_t;

template <typename PrimaryKey = primary_key>
struct autoincrementer
{
    explicit autoincrementer(const char* label)
    : m_label(label)
    {}

    auto get_primary_key_value() noexcept { return m_primary_key_value.fetch_add(1); }

private:
    std::atomic<PrimaryKey> m_primary_key_value{};
    const char*             m_label;
};

struct track_name_value
{
    size_t track_id;
    size_t name_id;
};

struct agent_unique_id_hash
{
    std::size_t operator()(const writer_api::agent_unique_id_t& agent) const noexcept
    {
        return std::hash<std::string>{}(agent.agent_type) ^
               std::hash<size_t>{}(agent.type_index);
    }
};

struct pmc_unique_id_hash
{
    std::size_t operator()(const writer_api::pmc_info_unique_id_t& pmc) const noexcept
    {
        return agent_unique_id_hash{}(*pmc.agent_id) ^ std::hash<std::string>{}(pmc.name);
    }
};
struct track_info_hash
{
    std::size_t operator()(const writer_api::track_info_t& track_info) const noexcept
    {
        std::string track_name_value =
            track_info.name.has_value() ? track_info.name.value() : "";
        size_t process_id_value =
            track_info.process_id.has_value() ? track_info.process_id.value() : 0;
        size_t thread_id_value =
            track_info.thread_id.has_value() ? track_info.thread_id.value() : 0;

        return std::hash<size_t>{}(track_info.node_id) ^
               std::hash<std::string>{}(track_name_value) ^

               std::hash<size_t>{}(process_id_value) ^
               std::hash<size_t>{}(thread_id_value);
    }
};
}  // namespace

struct writer::impl
{
    struct data_identifiers
    {
        using primary_key = size_t;

        ~data_identifiers() = default;

        // ---------------------------- Node Info ----------------------------

        std::unordered_set<writer_api::node_id_t> m_node_info_id_set{};

        // ---------------------------- Process Info ----------------------------

        autoincrementer<primary_key> m_process_info_primary_key_provider{
            "process_info"
        };
        std::unordered_map<writer_api::process_id_t, primary_key> m_process_info_id_map{};

        // ---------------------------- Agent Info ----------------------------

        autoincrementer<primary_key> m_agent_info_primary_key_provider{ "agent_info" };
        std::unordered_map<writer_api::agent_unique_id_t,
                           primary_key,
                           agent_unique_id_hash>
            m_agent_id_map{};

        // ---------------------------- PMC Info ----------------------------

        autoincrementer<primary_key> m_pmc_info_primary_key_provider{ "pmc_info" };
        std::unordered_map<writer_api::pmc_info_unique_id_t,
                           primary_key,
                           pmc_unique_id_hash>
            m_pmc_info_id_map{};

        // ---------------------------- Thread Info ----------------------------

        autoincrementer<primary_key> m_thread_info_primary_key_provider{ "thread_info" };
        std::unordered_map<writer_api::thread_id_t, primary_key> m_thread_info_id_map{};

        // ---------------------------- Stream Info ----------------------------

        autoincrementer<primary_key> m_stream_info_primary_key_provider{ "stream_info" };
        std::unordered_map<writer_api::stream_id_t, primary_key> m_stream_info_id_map{};

        // ---------------------------- Queue Info ----------------------------

        autoincrementer<primary_key> m_queue_info_primary_key_provider{ "queue_info" };
        std::unordered_map<writer_api::queue_id_t, primary_key> m_queue_info_id_map{};

        // ---------------------------- Code Object Info ----------------------------

        std::unordered_set<writer_api::code_object_id_t> m_code_object_info_id_set{};

        // ---------------------------- Kernel Symbol Info ----------------------------

        std::unordered_set<writer_api::kernel_symbol_id_t> m_kernel_symbol_info_id_set{};

        // ---------------------------- Track Info ----------------------------

        autoincrementer<primary_key> m_track_info_primary_key_provider{ "track_info" };
        std::unordered_map<writer_api::track_info_t, primary_key, track_info_hash>
            m_track_info_id_map{};

        // ---------------------------- String Info ----------------------------

        autoincrementer<primary_key> m_string_info_primary_key_provider{ "string_info" };
        std::unordered_map<std::string, primary_key> m_string_info_id_map{};

        // ---------------------------- Event Info ----------------------------

        autoincrementer<primary_key> m_event_info_primary_key_provider{ "event_data" };

        // ---------------------------- Region Info ----------------------------

        autoincrementer<primary_key> m_region_info_primary_key_provider{ "region_data" };

        // ---------------------------- Arg ----------------------------

        autoincrementer<primary_key> m_arg_primary_key_provider{ "arg" };

        // ---------------------------- PMC Event Info ----------------------------

        autoincrementer<primary_key> m_pmc_event_info_primary_key_provider{
            "pmc_event_data"
        };

        // -------------------------- Kernel Dispatch Info ------------------------

        autoincrementer<primary_key> m_kernel_dispatch_info_primary_key_provider{
            "kernel_dispatch_data"
        };

        // -------------------------- Memory Copy Info ------------------------

        autoincrementer<primary_key> m_memory_copy_info_primary_key_provider{
            "memory_copy_data"
        };

        // -------------------------- Memory Alloc Info ------------------------

        autoincrementer<primary_key> m_memory_alloc_info_primary_key_provider{
            "memory_alloc_data"
        };
    };

private:
    inline bool is_node_registered(const writer_api::node_id_t& node_id) const noexcept
    {
        return m_data_identifiers->m_node_info_id_set.count(node_id) > 0;
    }
    inline bool is_process_registered(
        const writer_api::process_id_t& process_id) const noexcept
    {
        return m_data_identifiers->m_process_info_id_map.count(process_id) > 0;
    }

    inline std::optional<size_t> get_agent_primary_key(
        const std::optional<writer_api::agent_unique_id_t>& agent_unique_id) const
    {
        if(agent_unique_id.has_value())
        {
            auto it = m_data_identifiers->m_agent_id_map.find(*agent_unique_id);
            if(it == m_data_identifiers->m_agent_id_map.end())
            {
                throw std::runtime_error(
                    fmt::format("Agent not registered: agent_type: {}, agent_index: {}",
                                agent_unique_id->agent_type,
                                agent_unique_id->type_index));
            }
            return it->second;
        }
        return std::nullopt;
    }

    inline bool is_agent_registered(
        const writer_api::agent_unique_id_t& agent_unique_id) const noexcept
    {
        return m_data_identifiers->m_agent_id_map.count(agent_unique_id) > 0;
    }

    inline bool is_pmc_info_registered(
        const writer_api::pmc_info_unique_id_t& pmc_unique_id) const noexcept
    {
        return m_data_identifiers->m_pmc_info_id_map.count(pmc_unique_id) > 0;
    }

    inline bool is_thread_info_registered(
        const writer_api::thread_id_t& thread_id) const noexcept
    {
        return m_data_identifiers->m_thread_info_id_map.count(thread_id) > 0;
    }

    inline bool is_stream_info_registered(
        const writer_api::stream_id_t& stream_id) const noexcept
    {
        return m_data_identifiers->m_stream_info_id_map.count(stream_id) > 0;
    }

    inline bool is_queue_info_registered(
        const writer_api::queue_id_t& queue_id) const noexcept
    {
        return m_data_identifiers->m_queue_info_id_map.count(queue_id) > 0;
    }

    inline bool is_code_object_info_registered(
        const writer_api::code_object_id_t& code_obj_id) const noexcept
    {
        return m_data_identifiers->m_code_object_info_id_set.count(code_obj_id) > 0;
    }

    inline bool is_kernel_symbol_info_registered(
        const writer_api::kernel_symbol_id_t& kernel_symbol_id) const noexcept
    {
        return m_data_identifiers->m_kernel_symbol_info_id_set.count(kernel_symbol_id) >
               0;
    }

    inline bool is_track_info_registered(
        const writer_api::track_info_t& track_info) const noexcept
    {
        return m_data_identifiers->m_track_info_id_map.count(track_info) > 0;
    }

    inline bool is_string_registered(const std::string& str) const noexcept
    {
        return m_data_identifiers->m_string_info_id_map.count(str) > 0;
    }

public:
    explicit impl(std::shared_ptr<data_storage::database> database, std::string uuid)
    : m_database(std::move(database))
    , m_uuid(std::move(uuid))
    , m_data_identifiers(std::make_unique<data_identifiers>())
    {
        if(!m_database)
        {
            throw std::invalid_argument("Provided pointer to a non-existing database!");
        }

        if(m_uuid.empty())
        {
            throw std::invalid_argument("Empty UUID provided!");
        }

        m_database->initialize_schema();
        initialize_metadata(m_database, m_uuid);
        m_insert_statements =
            std::make_unique<data_storage::schema_v3::insert_statements>(m_database,
                                                                         m_uuid);
    }

    void register_node_info(const writer_api::node_info_t& node_info)
    {
        if(is_node_registered(node_info.node_id))
        {
            LOG_WARNING("Node already registered: node_id: {}", node_info.node_id);
            return;
        }

        m_insert_statements->node_info_statement()(node_info.node_id,
                                                   node_info.hash,
                                                   node_info.machine_id,
                                                   node_info.system_name,
                                                   node_info.hostname,
                                                   node_info.release,
                                                   node_info.version,
                                                   node_info.hardware_name,
                                                   node_info.domain_name);

        m_data_identifiers->m_node_info_id_set.insert(node_info.node_id);
    }

    void register_process_info(const writer_api::process_info_t& process_info)
    {
        if(is_process_registered(process_info.pid))
        {
            LOG_WARNING("Process already registered: pid: {}", process_info.pid);
            return;
        }

        if(!is_node_registered(process_info.node_id))
        {
            throw std::runtime_error(
                fmt::format("Node not registered for Process Info: node_id: {}",
                            process_info.node_id));
        }

        const auto primary_key = m_data_identifiers->m_process_info_primary_key_provider
                                     .get_primary_key_value();
        m_insert_statements->process_info_statement()(primary_key,
                                                      process_info.node_id,
                                                      process_info.ppid,
                                                      process_info.pid,
                                                      process_info.init,
                                                      process_info.fini,
                                                      process_info.start,
                                                      process_info.end,
                                                      process_info.command,
                                                      process_info.environment,
                                                      process_info.extdata);

        m_data_identifiers->m_process_info_id_map.emplace(process_info.pid, primary_key);
    }

    void register_agent_info(const writer_api::agent_info_t& agent_info)
    {
        if(is_agent_registered(agent_info.unique_id))
        {
            LOG_WARNING("Agent already registered: type: {}, index: {}, name: {}",
                        agent_info.unique_id.agent_type,
                        agent_info.unique_id.type_index,
                        agent_info.name);
            return;
        }

        if(!is_node_registered(agent_info.node_id))
        {
            throw std::runtime_error(fmt::format(
                "Node not registered for Agent Info: node_id: {}", agent_info.node_id));
        }

        if(!is_process_registered(agent_info.process_id))
        {
            throw std::runtime_error(fmt::format(
                "Process not registered for Agent Info: pid: {}", agent_info.process_id));
        }

        const auto process_primary_key =
            m_data_identifiers->m_process_info_id_map.at(agent_info.process_id);

        const std::string_view agent_type{ agent_info.unique_id.agent_type };

        if(agent_type != "CPU" && agent_type != "GPU")
        {
            throw std::runtime_error(fmt::format(
                "Invalid agent type: {}. Type can be NULL, CPU, or GPU.", agent_type));
        }

        const auto primary_key =
            m_data_identifiers->m_agent_info_primary_key_provider.get_primary_key_value();
        m_insert_statements->agent_info_statement()(primary_key,
                                                    agent_info.node_id,
                                                    process_primary_key,
                                                    agent_info.unique_id.agent_type,
                                                    agent_info.absolute_index,
                                                    agent_info.logical_index,
                                                    agent_info.unique_id.type_index,
                                                    agent_info.uuid,
                                                    agent_info.name,
                                                    agent_info.model_name,
                                                    agent_info.vendor_name,
                                                    agent_info.product_name,
                                                    agent_info.user_name,
                                                    agent_info.extdata);

        m_data_identifiers->m_agent_id_map.emplace(agent_info.unique_id, primary_key);
    }

    void register_pmc_info(const writer_api::pmc_info_t& pmc_info)
    {
        if(is_pmc_info_registered(pmc_info.unique_id))
        {
            LOG_WARNING("PMC already registered: name: {}, agent_id: {}",
                        pmc_info.unique_id.name,
                        pmc_info.unique_id.agent_id->agent_type);
            return;
        }

        if(!is_node_registered(pmc_info.node_id))
        {
            throw std::runtime_error(fmt::format(
                "Node not registered for PMC Info: node_id: {}", pmc_info.node_id));
        }

        if(!is_process_registered(pmc_info.process_id))
        {
            throw std::runtime_error(fmt::format(
                "Process not registered for PMC Info: pid: {}", pmc_info.process_id));
        }

        const auto process_primary_key =
            m_data_identifiers->m_process_info_id_map.at(pmc_info.process_id);

        const auto agent_foreign_key = get_agent_primary_key(pmc_info.unique_id.agent_id);

        const auto primary_key =
            m_data_identifiers->m_pmc_info_primary_key_provider.get_primary_key_value();
        m_insert_statements->pmc_info_statement()(primary_key,
                                                  pmc_info.node_id,
                                                  process_primary_key,
                                                  agent_foreign_key,
                                                  pmc_info.target_arch,
                                                  pmc_info.event_code,
                                                  pmc_info.instance_id,
                                                  pmc_info.unique_id.name,
                                                  pmc_info.symbol,
                                                  pmc_info.description,
                                                  pmc_info.long_description,
                                                  pmc_info.component,
                                                  pmc_info.units,
                                                  pmc_info.value_type,
                                                  pmc_info.block,
                                                  pmc_info.expression,
                                                  pmc_info.is_constant,
                                                  pmc_info.is_derived,
                                                  pmc_info.extdata);

        m_data_identifiers->m_pmc_info_id_map.emplace(pmc_info.unique_id, primary_key);
    }

    // ---------------------------------------------------------------------------------------------

    void register_thread_info(const writer_api::thread_info_t& thread_info)
    {
        if(is_thread_info_registered(thread_info.thread_id))
        {
            LOG_WARNING("Thread already registered: thread_id: {}",
                        thread_info.thread_id);
            return;
        }

        if(!is_node_registered(thread_info.node_id))
        {
            throw std::runtime_error(fmt::format(
                "Node not registered for Thread Info: node_id: {}", thread_info.node_id));
        }

        if(!is_process_registered(thread_info.process_id))
        {
            throw std::runtime_error(
                fmt::format("Process not registered for Thread Info: pid: {}",
                            thread_info.process_id));
        }

        const auto process_primary_key =
            m_data_identifiers->m_process_info_id_map.at(thread_info.process_id);

        const auto primary_key = m_data_identifiers->m_thread_info_primary_key_provider
                                     .get_primary_key_value();

        m_insert_statements->thread_info_statement()(primary_key,
                                                     thread_info.node_id,
                                                     thread_info.parent_process_id,
                                                     process_primary_key,
                                                     thread_info.thread_id,
                                                     thread_info.name,
                                                     thread_info.start,
                                                     thread_info.end,
                                                     thread_info.extdata);

        m_data_identifiers->m_thread_info_id_map.emplace(thread_info.thread_id,
                                                         primary_key);
    }

    void register_stream_info(const writer_api::stream_info_t& stream_info)
    {
        if(is_stream_info_registered(stream_info.stream_id))
        {
            LOG_WARNING("Stream already registered: stream_id: {}",
                        stream_info.stream_id);
            return;
        }

        if(!is_node_registered(stream_info.node_id))
        {
            throw std::runtime_error(fmt::format(
                "Node not registered for Stream Info: node_id: {}", stream_info.node_id));
        }

        if(!is_process_registered(stream_info.process_id))
        {
            throw std::runtime_error(
                fmt::format("Process not registered for Stream Info: pid: {}",
                            stream_info.process_id));
        }

        const auto process_primary_key =
            m_data_identifiers->m_process_info_id_map.at(stream_info.process_id);

        const auto primary_key = m_data_identifiers->m_stream_info_primary_key_provider
                                     .get_primary_key_value();
        m_insert_statements->stream_info_statement()(primary_key,
                                                     stream_info.node_id,
                                                     process_primary_key,
                                                     stream_info.name,
                                                     stream_info.extdata);

        m_data_identifiers->m_stream_info_id_map.emplace(stream_info.stream_id,
                                                         primary_key);
    }

    void register_queue_info(const writer_api::queue_info_t& queue_info)
    {
        if(is_queue_info_registered(queue_info.queue_id))
        {
            LOG_WARNING("Queue already registered: queue_id: {}", queue_info.queue_id);
            return;
        }

        if(!is_node_registered(queue_info.node_id))
        {
            throw std::runtime_error(fmt::format(
                "Node not registered for Queue Info: node_id: {}", queue_info.node_id));
        }

        if(!is_process_registered(queue_info.process_id))
        {
            throw std::runtime_error(fmt::format(
                "Process not registered for Queue Info: pid: {}", queue_info.process_id));
        }

        const auto process_primary_key =
            m_data_identifiers->m_process_info_id_map.at(queue_info.process_id);

        const auto primary_key =
            m_data_identifiers->m_queue_info_primary_key_provider.get_primary_key_value();

        m_insert_statements->queue_info_statement()(primary_key,
                                                    queue_info.node_id,
                                                    process_primary_key,
                                                    queue_info.name,
                                                    queue_info.extdata);

        m_data_identifiers->m_queue_info_id_map.emplace(queue_info.queue_id, primary_key);
    }

    void regsiter_code_object_info(const writer_api::code_object_info_t& code_object_info)
    {
        if(is_code_object_info_registered(code_object_info.id))
        {
            LOG_WARNING("Code object already registered: id: {}", code_object_info.id);
            return;
        }

        if(!is_node_registered(code_object_info.node_id))
        {
            throw std::runtime_error(
                fmt::format("Node not registered for Code Object Info: node_id: {}",
                            code_object_info.node_id));
        }

        if(!is_process_registered(code_object_info.process_id))
        {
            throw std::runtime_error(
                fmt::format("Process not registered for Code Object Info: pid: {}",
                            code_object_info.process_id));
        }

        const auto process_primary_key =
            m_data_identifiers->m_process_info_id_map.at(code_object_info.process_id);

        if(code_object_info.agent_id.has_value() &&
           !is_agent_registered(code_object_info.agent_id.value()))
        {
            throw std::runtime_error(
                fmt::format("Agent not registered for Code Object Info: agent_id "
                            "[agent_type={}, type_index={}]",
                            code_object_info.agent_id->agent_type,
                            code_object_info.agent_id->type_index));
        }

        std::optional<primary_key> agent_foreign_key = std::nullopt;
        if(code_object_info.agent_id.has_value())
        {
            agent_foreign_key =
                m_data_identifiers->m_agent_id_map.at(code_object_info.agent_id.value());
        }

        m_insert_statements->code_object_info_statement()(code_object_info.id,
                                                          code_object_info.node_id,
                                                          process_primary_key,
                                                          agent_foreign_key,
                                                          code_object_info.uri,
                                                          code_object_info.ld_base,
                                                          code_object_info.ld_size,
                                                          code_object_info.ld_delta,
                                                          code_object_info.storage_type,
                                                          code_object_info.extdata);

        m_data_identifiers->m_code_object_info_id_set.insert(code_object_info.id);
    }

    void register_kernel_symbol_info(
        const writer_api::kernel_symbol_info_t& kernel_symbol_info)
    {
        if(is_kernel_symbol_info_registered(kernel_symbol_info.id))
        {
            LOG_WARNING("Kernel symbol already registered: id: {}",
                        kernel_symbol_info.id);
            return;
        }

        if(!is_node_registered(kernel_symbol_info.node_id))
        {
            throw std::runtime_error(
                fmt::format("Node not registered for Kernel Symbol Info: node_id: {}",
                            kernel_symbol_info.node_id));
        }

        if(!is_process_registered(kernel_symbol_info.process_id))
        {
            throw std::runtime_error(
                fmt::format("Process not registered for Kernel Symbol Info: pid: {}",
                            kernel_symbol_info.process_id));
        }

        const auto process_primary_key =
            m_data_identifiers->m_process_info_id_map.at(kernel_symbol_info.process_id);

        if(!is_code_object_info_registered(kernel_symbol_info.code_obj_id))
        {
            throw std::runtime_error(fmt::format(
                "Code object not registered for Kernel Symbol Info: code_obj_id: {}",
                kernel_symbol_info.code_obj_id));
        }

        m_insert_statements->kernel_symbol_info_statement()(
            kernel_symbol_info.id,
            kernel_symbol_info.node_id,
            process_primary_key,
            kernel_symbol_info.code_obj_id,
            kernel_symbol_info.name,
            kernel_symbol_info.display_name,
            kernel_symbol_info.kernel_obj,
            kernel_symbol_info.kernarg_segmnt_size,
            kernel_symbol_info.kernarg_segment_alignment,
            kernel_symbol_info.group_segment_size,
            kernel_symbol_info.private_segment_size,
            kernel_symbol_info.sgrp_count,
            kernel_symbol_info.arch_vgrp_count,
            kernel_symbol_info.accum_vgrp_count,
            kernel_symbol_info.extdata);

        m_data_identifiers->m_kernel_symbol_info_id_set.emplace(kernel_symbol_info.id);
    }

    void register_track_info(const writer_api::track_info_t& track)
    {
        if(is_track_info_registered(track))
        {
            constexpr auto empty = "NULL";

            const auto process_print_value =
                track.process_id.has_value() ? std::to_string(track.process_id.value())
                                             : empty;
            const auto thread_print_value = track.thread_id.has_value()
                                                ? std::to_string(track.thread_id.value())
                                                : empty;
            const auto name_print_value =
                track.name.has_value() ? track.name.value() : empty;

            LOG_WARNING("Track already registered: node_id: {}, process_id: {}, "
                        "thread_id: {}, name: {}",
                        track.node_id,
                        process_print_value,
                        thread_print_value,
                        name_print_value);
            return;
        }

        if(!is_node_registered(track.node_id))
        {
            throw std::runtime_error(fmt::format(
                "Node not registered for Track Info: node_id: {}", track.node_id));
        }

        if(track.process_id.has_value() &&
           !is_process_registered(track.process_id.value()))
        {
            throw std::runtime_error(
                fmt::format("Process not registered for Track Info: pid: {}",
                            track.process_id.value()));
        }

        std::optional<primary_key> process_primary_key = std::nullopt;
        if(track.process_id.has_value())
        {
            process_primary_key =
                m_data_identifiers->m_process_info_id_map.at(track.process_id.value());
        }

        if(track.thread_id.has_value() &&
           !is_thread_info_registered(track.thread_id.value()))
        {
            throw std::runtime_error(
                fmt::format("Thread not registered for Track Info: thread_id: {}",
                            track.thread_id.value()));
        }

        std::optional<primary_key> thread_primary_key = std::nullopt;
        if(track.thread_id.has_value())
        {
            thread_primary_key =
                m_data_identifiers->m_thread_info_id_map.at(track.thread_id.value());
        }

        std::optional<primary_key> string_primary_key = std::nullopt;
        if(track.name.has_value())
        {
            if(!is_string_registered(track.name.value()))
            {
                register_string(track.name.value());
            }

            string_primary_key =
                m_data_identifiers->m_string_info_id_map.at(track.name.value());
        }

        const auto primary_key =
            m_data_identifiers->m_track_info_primary_key_provider.get_primary_key_value();

        m_insert_statements->track_info_statement()(primary_key,
                                                    track.node_id,
                                                    process_primary_key,
                                                    thread_primary_key,
                                                    string_primary_key,
                                                    track.extdata);

        m_data_identifiers->m_track_info_id_map.emplace(track, primary_key);
    }

    void register_string(const char* str)
    {
        if(str == nullptr)
        {
            throw std::runtime_error("Trying to register string that is null");
        }

        if(is_string_registered(str))
        {
            LOG_WARNING("String already registered: str: {}", str);
            return;
        }

        const auto primary_key = m_data_identifiers->m_string_info_primary_key_provider
                                     .get_primary_key_value();
        m_insert_statements->string_statement()(primary_key, str);
        m_data_identifiers->m_string_info_id_map.emplace(str, primary_key);
    }

    // --------------------- Data Tables ---------------------

private:
    primary_key insert_event(const writer_api::event_data_t& event_data)
    {
        auto json_call_stack_serializer = [](const writer_api::call_stack_t& call_stack) {
            return "{}";
        };

        auto json_line_info_serializer =
            [](const writer_api::source_context_list_t& line_info_list) { return "{}"; };

        if(!is_string_registered(event_data.event_category))
        {
            register_string(event_data.event_category);
        }

        const auto event_category_primary_key =
            m_data_identifiers->m_string_info_id_map.at(event_data.event_category);
        const auto primary_key =
            m_data_identifiers->m_event_info_primary_key_provider.get_primary_key_value();

        m_insert_statements->event_statement()(
            primary_key,
            event_category_primary_key,
            event_data.stack_id,
            event_data.parent_stack_id,
            event_data.correlation_id,
            json_call_stack_serializer(event_data.call_stack),
            json_line_info_serializer(event_data.line_info_list),
            event_data.extdata);

        return primary_key;
    }

    void insert_sample(const writer_api::sample_data_t& sample_data) {}

    inline void insert_arg(const writer_api::arg_data_t& arg_data, primary_key event_id)
    {
        if(arg_data.type == nullptr || arg_data.name == nullptr)
        {
            throw std::runtime_error(
                fmt::format("Type or name is null for Arg Data: type: {}, name: {}",
                            arg_data.type,
                            arg_data.name));
        }

        if(!is_string_registered(arg_data.type))
        {
            register_string(arg_data.type);
            auto primary_key =
                m_data_identifiers->m_arg_primary_key_provider.get_primary_key_value();
            m_insert_statements->arg_statement()(primary_key,
                                                 event_id,
                                                 arg_data.position,
                                                 arg_data.type,
                                                 arg_data.name,
                                                 arg_data.value,
                                                 arg_data.extdata);
        }
    }

public:
    void insert_region_data(const writer_api::region_data_t&       region_data,
                            const writer_api::trace_environment_t& trace_environment)

    {
        if(!trace_environment.node_id.has_value() ||
           !is_node_registered(trace_environment.node_id.value()))
        {
            const std::string node_id_str =
                trace_environment.node_id.has_value()
                    ? std::to_string(trace_environment.node_id.value())
                    : "[NULL]";
            throw std::runtime_error(fmt::format(
                "Node not registered for Region Data: node_id: {}", node_id_str));
        }

        if(!trace_environment.process_id.has_value() ||
           !is_process_registered(trace_environment.process_id.value()))
        {
            const std::string process_id_str =
                trace_environment.process_id.has_value()
                    ? std::to_string(trace_environment.process_id.value())
                    : "[NULL]";
            throw std::runtime_error(fmt::format(
                "Process not registered for Region Data: pid: {}", process_id_str));
        }

        const auto process_primary_key = m_data_identifiers->m_process_info_id_map.at(
            trace_environment.process_id.value());

        if(!trace_environment.thread_id.has_value() ||
           !is_thread_info_registered(trace_environment.thread_id.value()))
        {
            const std::string thread_id_str =
                trace_environment.thread_id.has_value()
                    ? std::to_string(trace_environment.thread_id.value())
                    : "[NULL]";
            throw std::runtime_error(fmt::format(
                "Thread not registered for Region Data: thread_id: {}", thread_id_str));
        }

        const auto thread_primary_key = m_data_identifiers->m_thread_info_id_map.at(
            trace_environment.thread_id.value());

        if(region_data.event.has_value() && !region_data.args.empty())
        {
            throw std::runtime_error(fmt::format(
                "Writing args require providing event data for correlation: name: {}",
                region_data.name));
        }

        // ----------------------------------------------------------------

        if(!is_string_registered(region_data.name))
        {
            register_string(region_data.name);
        }

        const auto region_name_primary_key =
            m_data_identifiers->m_string_info_id_map.at(region_data.name);

        std::optional<primary_key> event_primary_key = std::nullopt;
        if(region_data.event.has_value())
        {
            event_primary_key = insert_event(region_data.event.value());
        }

        const auto primary_key = m_data_identifiers->m_region_info_primary_key_provider
                                     .get_primary_key_value();
        m_insert_statements->region_statement()(primary_key,
                                                trace_environment.node_id.value(),
                                                process_primary_key,
                                                thread_primary_key,
                                                region_data.start_timestamp,
                                                region_data.end_timestamp,
                                                region_name_primary_key,
                                                event_primary_key,
                                                region_data.extdata);

        // Event should not be empty when writing args
        for(const auto& arg : region_data.args)
        {
            insert_arg(arg, event_primary_key.value());
        }

        if(trace_environment.track_name.has_value())
        {
            const writer_api::track_info_t track_info = {
                trace_environment.track_name.value(),
                nullptr,
                trace_environment.node_id.value(),
                trace_environment.process_id.value(),
                trace_environment.thread_id.value()
            };

            if(is_track_info_registered(track_info))
            {
                const writer_api::sample_data_t sample_data = {
                    region_data.start_timestamp, track_info, nullptr
                };
                insert_sample(sample_data);
            }
        }
    }

    void insert_pmc_event_data(const writer_api::pmc_event_data_t&     pmc_event_data,
                               const writer_api::pmc_info_unique_id_t& pmc_unique_id)
    {
        if(!is_pmc_info_registered(pmc_unique_id))
        {
            throw std::runtime_error(
                fmt::format("PMC Info not registered for PMC Event Data: pmc_name: {}",
                            pmc_unique_id.name));
        }

        const auto pmc_info_primary_key =
            m_data_identifiers->m_pmc_info_id_map.at(pmc_unique_id);

        std::optional<primary_key> event_primary_key = std::nullopt;
        if(pmc_event_data.event.has_value())
        {
            event_primary_key = insert_event(pmc_event_data.event.value());
        }

        const auto primary_key = m_data_identifiers->m_pmc_event_info_primary_key_provider
                                     .get_primary_key_value();
        m_insert_statements->pmc_event_statement()(primary_key,
                                                   event_primary_key,
                                                   pmc_info_primary_key,
                                                   pmc_event_data.value,
                                                   pmc_event_data.extdata);
    }

    void insert_kernel_dispatch_data(
        const writer_api::kernel_dispatch_data_t& kernel_dispatch_data,
        const writer_api::trace_environment_t&    trace_environment)
    {
        if(!trace_environment.node_id.has_value() ||
           !is_node_registered(trace_environment.node_id.value()))
        {
            const std::string node_id_str =
                trace_environment.node_id.has_value()
                    ? std::to_string(trace_environment.node_id.value())
                    : "[NULL]";
            throw std::runtime_error(
                fmt::format("Node not registered for Kernel Dispatch Data: node_id: {}",
                            node_id_str));
        }

        if(!trace_environment.process_id.has_value() ||
           !is_process_registered(trace_environment.process_id.value()))
        {
            const std::string process_id_str =
                trace_environment.process_id.has_value()
                    ? std::to_string(trace_environment.process_id.value())
                    : "[NULL]";
            throw std::runtime_error(
                fmt::format("Process not registered for Kernel Dispatch Data: pid: {}",
                            process_id_str));
        }

        const auto process_primary_key = m_data_identifiers->m_process_info_id_map.at(
            trace_environment.process_id.value());

        if(!trace_environment.thread_id.has_value() ||
           !is_thread_info_registered(trace_environment.thread_id.value()))
        {
            const std::string thread_id_str =
                trace_environment.thread_id.has_value()
                    ? std::to_string(trace_environment.thread_id.value())
                    : "[NULL]";
            throw std::runtime_error(fmt::format(
                "Thread not registered for Kernel Dispatch Data: thread_id: {}",
                thread_id_str));
        }

        std::optional<primary_key> thread_primary_key{ std::nullopt };
        if(trace_environment.thread_id.has_value())
        {
            thread_primary_key = m_data_identifiers->m_thread_info_id_map.at(
                trace_environment.thread_id.value());
        }

        if(!trace_environment.agent_id.has_value() ||
           !is_agent_registered(trace_environment.agent_id.value()))
        {
            const std::string agent_id_str =
                trace_environment.agent_id.has_value()
                    ? fmt::format("[agent_type={}, type_index={}]",
                                  trace_environment.agent_id->agent_type,
                                  trace_environment.agent_id->type_index)
                    : "[NULL]";

            throw std::runtime_error(
                fmt::format("Agent not registered for Kernel Dispatch Data: agent_id: {}",
                            agent_id_str));
        }

        const auto agent_primary_key =
            m_data_identifiers->m_agent_id_map.at(trace_environment.agent_id.value());

        if(!trace_environment.queue_id.has_value() ||
           !is_queue_info_registered(trace_environment.queue_id.value()))
        {
            const std::string queue_id_str =
                trace_environment.queue_id.has_value()
                    ? std::to_string(trace_environment.queue_id.value())
                    : "[NULL]";
            throw std::runtime_error(
                fmt::format("Queue not registered for Kernel Dispatch Data: queue_id: {}",
                            queue_id_str));
        }

        const auto queue_primary_key = m_data_identifiers->m_queue_info_id_map.at(
            trace_environment.queue_id.value());

        if(!trace_environment.stream_id.has_value() ||
           !is_stream_info_registered(trace_environment.stream_id.value()))
        {
            const std::string stream_id_str =
                trace_environment.stream_id.has_value()
                    ? std::to_string(trace_environment.stream_id.value())
                    : "[NULL]";
            throw std::runtime_error(fmt::format(
                "Stream not registered for Kernel Dispatch Data: stream_id: {}",
                stream_id_str));
        }

        const auto stream_primary_key = m_data_identifiers->m_stream_info_id_map.at(
            trace_environment.stream_id.value());

        if(!is_kernel_symbol_info_registered(kernel_dispatch_data.kernel_symbol_id))
        {
            throw std::runtime_error(
                fmt::format("Kernel symbol not registered for Kernel Dispatch Data: "
                            "kernel_id: {}",
                            kernel_dispatch_data.kernel_symbol_id));
        }

        // ----------------------------------------------------------------

        if(!is_string_registered(kernel_dispatch_data.name))
        {
            register_string(kernel_dispatch_data.name);
        }

        const auto kernel_name_primary_key =
            m_data_identifiers->m_string_info_id_map.at(kernel_dispatch_data.name);

        std::optional<primary_key> event_primary_key = std::nullopt;
        if(kernel_dispatch_data.event.has_value())
        {
            event_primary_key = insert_event(kernel_dispatch_data.event.value());
        }

        const auto primary_key =
            m_data_identifiers->m_kernel_dispatch_info_primary_key_provider
                .get_primary_key_value();

        m_insert_statements->kernel_dispatch_statement()(
            primary_key,
            trace_environment.node_id.value(),
            process_primary_key,
            thread_primary_key,
            agent_primary_key,
            kernel_dispatch_data.kernel_symbol_id,
            kernel_dispatch_data.dispatch_id,
            queue_primary_key,
            stream_primary_key,
            kernel_dispatch_data.start_timestamp,
            kernel_dispatch_data.end_timestamp,
            kernel_dispatch_data.private_segment_size,
            kernel_dispatch_data.group_segment_size,
            kernel_dispatch_data.workgroup_size_x,
            kernel_dispatch_data.workgroup_size_y,
            kernel_dispatch_data.workgroup_size_z,
            kernel_dispatch_data.grid_size_x,
            kernel_dispatch_data.grid_size_y,
            kernel_dispatch_data.grid_size_z,
            kernel_name_primary_key,
            event_primary_key,
            kernel_dispatch_data.extdata);
    }

    void insert_memory_copy_data(const writer_api::memory_copy_data_t&  memory_copy_data,
                                 const writer_api::trace_environment_t& trace_environment)
    {
        // TODO: Clean up if/else statements for all inserts

        // Node
        if(!trace_environment.node_id.has_value() ||
           !is_node_registered(trace_environment.node_id.value()))
        {
            const std::string node_id_str =
                trace_environment.node_id.has_value()
                    ? std::to_string(trace_environment.node_id.value())
                    : "[NULL]";
            throw std::runtime_error(fmt::format(
                "Node not provided or not registered for Memory Copy Data: node_id: {}",
                node_id_str));
        }

        // process
        if(!trace_environment.process_id.has_value() ||
           !is_process_registered(trace_environment.process_id.value()))
        {
            const std::string process_id_str =
                trace_environment.process_id.has_value()
                    ? std::to_string(trace_environment.process_id.value())
                    : "[NULL]";
            throw std::runtime_error(fmt::format(
                "Process not provided or not registered for Memory Copy Data: pid: {}",
                process_id_str));
        }

        const auto process_primary_key = m_data_identifiers->m_process_info_id_map.at(
            trace_environment.process_id.value());

        // thread
        if(trace_environment.thread_id.has_value() &&
           !is_thread_info_registered(trace_environment.thread_id.value()))
        {
            throw std::runtime_error(
                fmt::format("Thread not registered for Memory Copy Data: thread_id: {}",
                            std::to_string(trace_environment.thread_id.value())));
        }

        std::optional<primary_key> thread_primary_key = std::nullopt;
        if(trace_environment.thread_id.has_value())
        {
            thread_primary_key = m_data_identifiers->m_thread_info_id_map.at(
                trace_environment.thread_id.value());
        }

        // src agent
        if(memory_copy_data.src_agent_id.has_value() &&
           !is_agent_registered(memory_copy_data.src_agent_id.value()))
        {
            throw std::runtime_error(
                fmt::format("Source agent not registered for Memory Copy Data: agent_id "
                            "[agent_type={}, type_index={}]",
                            memory_copy_data.src_agent_id->agent_type,
                            memory_copy_data.src_agent_id->type_index));
        }

        std::optional<primary_key> src_agent_primary_key = std::nullopt;
        if(memory_copy_data.src_agent_id.has_value())
        {
            src_agent_primary_key = m_data_identifiers->m_agent_id_map.at(
                memory_copy_data.src_agent_id.value());
        }

        // dst agent
        if(memory_copy_data.dst_agent_id.has_value() &&
           !is_agent_registered(memory_copy_data.dst_agent_id.value()))
        {
            throw std::runtime_error(
                fmt::format("Destination agent not registered for Memory Copy Data: "
                            "agent_id [agent_type={}, type_index={}]",
                            memory_copy_data.dst_agent_id->agent_type,
                            memory_copy_data.dst_agent_id->type_index));
        }

        std::optional<primary_key> dst_agent_primary_key = std::nullopt;
        if(memory_copy_data.dst_agent_id.has_value())
        {
            dst_agent_primary_key = m_data_identifiers->m_agent_id_map.at(
                memory_copy_data.dst_agent_id.value());
        }

        // queue
        if(trace_environment.queue_id.has_value() &&
           !is_queue_info_registered(trace_environment.queue_id.value()))
        {
            throw std::runtime_error(
                fmt::format("Queue not registered for Memory Copy Data: queue_id: {}",
                            trace_environment.queue_id.value()));
        }

        std::optional<primary_key> queue_primary_key = std::nullopt;
        if(trace_environment.queue_id.has_value())
        {
            queue_primary_key = m_data_identifiers->m_queue_info_id_map.at(
                trace_environment.queue_id.value());
        }

        // stream
        if(trace_environment.stream_id.has_value() &&
           !is_stream_info_registered(trace_environment.stream_id.value()))
        {
            throw std::runtime_error(
                fmt::format("Stream not registered for Memory Copy Data: stream_id: {}",
                            trace_environment.stream_id.value()));
        }

        std::optional<primary_key> stream_primary_key = std::nullopt;
        if(trace_environment.stream_id.has_value())
        {
            stream_primary_key = m_data_identifiers->m_stream_info_id_map.at(
                trace_environment.stream_id.value());
        }

        // ----------------------------------------------------------------

        if(!is_string_registered(memory_copy_data.name))
        {
            register_string(memory_copy_data.name);
        }

        const auto name_primary_key =
            m_data_identifiers->m_string_info_id_map.at(memory_copy_data.name);

        // event
        std::optional<primary_key> event_primary_key = std::nullopt;
        if(memory_copy_data.event.has_value())
        {
            event_primary_key = insert_event(memory_copy_data.event.value());
        }

        std::optional<primary_key> region_name_primary_key = std::nullopt;
        if(memory_copy_data.region_name != nullptr)
        {
            if(!is_string_registered(memory_copy_data.region_name))
            {
                register_string(memory_copy_data.region_name);
            }
            region_name_primary_key =
                m_data_identifiers->m_string_info_id_map.at(memory_copy_data.region_name);
        }

        const auto primary_key =
            m_data_identifiers->m_memory_copy_info_primary_key_provider
                .get_primary_key_value();

        m_insert_statements->memory_copy_statement()(primary_key,
                                                     trace_environment.node_id.value(),
                                                     process_primary_key,
                                                     thread_primary_key,
                                                     memory_copy_data.start_timestamp,
                                                     memory_copy_data.end_timestamp,
                                                     name_primary_key,
                                                     dst_agent_primary_key,
                                                     memory_copy_data.dst_address,
                                                     src_agent_primary_key,
                                                     memory_copy_data.src_address,
                                                     memory_copy_data.size,
                                                     queue_primary_key,
                                                     stream_primary_key,
                                                     region_name_primary_key,
                                                     event_primary_key,
                                                     memory_copy_data.extdata);
    }

    void insert_memory_alloc_data(
        const writer_api::memory_alloc_data_t& memory_alloc_data,
        const writer_api::trace_environment_t& trace_environment)
    {
        // Node
        if(!trace_environment.node_id.has_value() ||
           !is_node_registered(trace_environment.node_id.value()))
        {
            const std::string node_id_str =
                trace_environment.node_id.has_value()
                    ? std::to_string(trace_environment.node_id.value())
                    : "[NULL]";
            throw std::runtime_error(fmt::format(
                "Node not provided or not registered for Memory Alloc Data: node_id: {}",
                node_id_str));
        }

        // process
        if(!trace_environment.process_id.has_value() ||
           !is_process_registered(trace_environment.process_id.value()))
        {
            const std::string process_id_str =
                trace_environment.process_id.has_value()
                    ? std::to_string(trace_environment.process_id.value())
                    : "[NULL]";
            throw std::runtime_error(fmt::format(
                "Process not provided or not registered for Memory Alloc Data: pid: {}",
                process_id_str));
        }

        const auto process_primary_key = m_data_identifiers->m_process_info_id_map.at(
            trace_environment.process_id.value());

        // thread
        if(trace_environment.thread_id.has_value() &&
           !is_thread_info_registered(trace_environment.thread_id.value()))
        {
            throw std::runtime_error(
                fmt::format("Thread not registered for Memory Alloc Data: thread_id: {}",
                            std::to_string(trace_environment.thread_id.value())));
        }

        std::optional<primary_key> thread_primary_key = std::nullopt;
        if(trace_environment.thread_id.has_value())
        {
            thread_primary_key = m_data_identifiers->m_thread_info_id_map.at(
                trace_environment.thread_id.value());
        }

        // agent
        if(trace_environment.agent_id.has_value() &&
           !is_agent_registered(trace_environment.agent_id.value()))
        {
            throw std::runtime_error(
                fmt::format("Agent not registered for Memory Alloc Data: agent_id "
                            "[agent_type={}, type_index={}]",
                            trace_environment.agent_id->agent_type,
                            trace_environment.agent_id->type_index));
        }

        std::optional<primary_key> agent_primary_key = std::nullopt;
        if(trace_environment.agent_id.has_value())
        {
            agent_primary_key =
                m_data_identifiers->m_agent_id_map.at(trace_environment.agent_id.value());
        }

        // queue
        if(trace_environment.queue_id.has_value() &&
           !is_queue_info_registered(trace_environment.queue_id.value()))
        {
            throw std::runtime_error(
                fmt::format("Queue not registered for Memory Alloc Data: queue_id: {}",
                            trace_environment.queue_id.value()));
        }

        std::optional<primary_key> queue_primary_key = std::nullopt;
        if(trace_environment.queue_id.has_value())
        {
            queue_primary_key = m_data_identifiers->m_queue_info_id_map.at(
                trace_environment.queue_id.value());
        }

        // stream
        if(trace_environment.stream_id.has_value() &&
           !is_stream_info_registered(trace_environment.stream_id.value()))
        {
            throw std::runtime_error(
                fmt::format("Stream not registered for Memory Alloc Data: stream_id: {}",
                            trace_environment.stream_id.value()));
        }

        std::optional<primary_key> stream_primary_key = std::nullopt;
        if(trace_environment.stream_id.has_value())
        {
            stream_primary_key = m_data_identifiers->m_stream_info_id_map.at(
                trace_environment.stream_id.value());
        }

        // -----------------------------------------------------------------------

        std::optional<primary_key> event_primary_key = std::nullopt;
        if(memory_alloc_data.event.has_value())
        {
            event_primary_key = insert_event(memory_alloc_data.event.value());
        }

        if(memory_alloc_data.type != nullptr)
        {
            const std::string_view                    type_value = memory_alloc_data.type;
            constexpr std::array<std::string_view, 4> allowed_type_values = {
                "ALLOC", "FREE", "REALLOC", "RECLAIM"
            };

            const auto is_value_allowed =
                std::find(allowed_type_values.begin(),
                          allowed_type_values.end(),
                          type_value) != allowed_type_values.end();

            if(!is_value_allowed)
            {
                throw std::runtime_error(
                    fmt::format("Invalid type value for Memory Alloc Data: type: {}. "
                                "Allowed values: {}",
                                type_value,
                                allowed_type_values));
            }
        }

        if(memory_alloc_data.level != nullptr)
        {
            const std::string_view level_value = memory_alloc_data.level;
            constexpr std::array<std::string_view, 3> allowed_level_values = {
                "REAL", "VIRTUAL", "SCRATCH"
            };
            const auto is_value_allowed =
                std::find(allowed_level_values.begin(),
                          allowed_level_values.end(),
                          level_value) != allowed_level_values.end();

            if(!is_value_allowed)
            {
                throw std::runtime_error(
                    fmt::format("Invalid level value for Memory Alloc Data: level: {}. "
                                "Allowed values: {}",
                                level_value,
                                allowed_level_values));
            }
        }

        const auto primary_key =
            m_data_identifiers->m_memory_alloc_info_primary_key_provider
                .get_primary_key_value();

        m_insert_statements->memory_alloc_statement()(primary_key,
                                                      trace_environment.node_id.value(),
                                                      process_primary_key,
                                                      thread_primary_key,
                                                      agent_primary_key,
                                                      memory_alloc_data.type,
                                                      memory_alloc_data.level,
                                                      memory_alloc_data.start_timestamp,
                                                      memory_alloc_data.end_timestamp,
                                                      memory_alloc_data.address,
                                                      memory_alloc_data.size,
                                                      queue_primary_key,
                                                      stream_primary_key,
                                                      event_primary_key,
                                                      memory_alloc_data.extdata);
    }

    void flush_in_memory_data_to_disk() { m_database->flush(); }

    std::shared_ptr<data_storage::database>                     m_database;
    std::string                                                 m_uuid;
    std::unique_ptr<data_storage::schema_v3::insert_statements> m_insert_statements;
    std::unique_ptr<data_identifiers>                           m_data_identifiers;
};

// ----------------------- PUBLIC API -------------------------

writer::writer(std::shared_ptr<data_storage::database> database, std::string uuid)
: m_impl(std::make_unique<impl>(std::move(database), std::move(uuid)))
{}

writer::~writer() = default;

void
writer::register_node_info(const writer_api::node_info_t& node_info)
{
    m_impl->register_node_info(node_info);
}

void
writer::register_process_info(const writer_api::process_info_t& process_info)
{
    m_impl->register_process_info(process_info);
}

void
writer::register_agent_info(const writer_api::agent_info_t& agent)
{
    m_impl->register_agent_info(agent);
}

void
writer::register_pmc_info(const writer_api::pmc_info_t& pmc_info)
{
    m_impl->register_pmc_info(pmc_info);
}

void
writer::register_thread_info(const writer_api::thread_info_t& thread_info)
{
    m_impl->register_thread_info(thread_info);
}

void
writer::register_stream_info(const writer_api::stream_info_t& stream_info)
{
    m_impl->register_stream_info(stream_info);
}

void
writer::register_queue_info(const writer_api::queue_info_t& queue_info)
{
    m_impl->register_queue_info(queue_info);
}

void
writer::register_code_object_info(const writer_api::code_object_info_t& code_object)
{
    m_impl->regsiter_code_object_info(code_object);
}

void
writer::register_kernel_symbol_info(const writer_api::kernel_symbol_info_t& kernel_symbol)
{
    m_impl->register_kernel_symbol_info(kernel_symbol);
}

void
writer::register_track_info(const writer_api::track_info_t& track)
{
    m_impl->register_track_info(track);
}

void
writer::register_string(const char* str)
{
    m_impl->register_string(str);
}

void
writer::insert_region_data(const writer_api::region_data_t&       region_data,
                           const writer_api::trace_environment_t& trace_environment)
{
    m_impl->insert_region_data(region_data, trace_environment);
}

void
writer::insert_pmc_event_data(const writer_api::pmc_event_data_t&     pmc_event_data,
                              const writer_api::pmc_info_unique_id_t& pmc_unique_id)
{
    m_impl->insert_pmc_event_data(pmc_event_data, pmc_unique_id);
}

void
writer::insert_kernel_dispatch_data(
    const writer_api::kernel_dispatch_data_t& kernel_dispatch_data,
    const writer_api::trace_environment_t&    trace_environment)
{
    m_impl->insert_kernel_dispatch_data(kernel_dispatch_data, trace_environment);
}

void
writer::insert_memory_copy_data(const writer_api::memory_copy_data_t&  memory_copy_data,
                                const writer_api::trace_environment_t& trace_environment)
{
    m_impl->insert_memory_copy_data(memory_copy_data, trace_environment);
}

void
writer::insert_memory_alloc_data(const writer_api::memory_alloc_data_t& memory_alloc_data,
                                 const writer_api::trace_environment_t& trace_environment)
{
    m_impl->insert_memory_alloc_data(memory_alloc_data, trace_environment);
}

void
writer::flush_in_memory_data_to_disk()
{
    m_impl->flush_in_memory_data_to_disk();
}

}  // namespace rocstorage
