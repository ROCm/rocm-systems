// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <rocstorage/writer.hpp>

#include "data_storage/database.hpp"
#include "data_storage/insert_statements.hpp"
#include "data_storage/table_insert_query.hpp"

#include "debug.hpp"

#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace std
{
std::string
to_string(const rocstorage::writer_api::agent_unique_id_t& agent_unique_id)
{
    return fmt::format("[agent_type={}, type_index={}]",
                       agent_unique_id.agent_type,
                       agent_unique_id.type_index);
}
}  // namespace std

namespace rocstorage
{

namespace
{

template <typename T>
[[nodiscard]] std::string
to_error_string(const std::optional<T>& opt)
{
    return opt.has_value() ? std::to_string(opt.value()) : "[NULL]";
}

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

template <typename Utility, typename EntityId>
[[nodiscard]] std::optional<primary_key>
resolve_optional_key(Utility& utility, const std::optional<EntityId>& entity_id)
{
    if(!entity_id.has_value()) return std::nullopt;
    return utility.get_primary_key_value_for_entity(entity_id.value());
}

}  // namespace

template <typename T>
struct is_std_unordered_map : std::false_type
{};

template <typename K, typename V, typename Hash, typename KeyEqual, typename Alloc>
struct is_std_unordered_map<std::unordered_map<K, V, Hash, KeyEqual, Alloc>>
: std::true_type
{};

template <typename T>
constexpr bool is_unordered_map_v = is_std_unordered_map<T>::value;

template <typename EntityContainerType, typename PrimaryKey = primary_key>
class entity_utility
{
public:
    template <typename Entity>
    bool is_entry_registered(const Entity& entity) const noexcept
    {
        return m_entity_container.count(entity) > 0;
    }

    template <typename... Entity>
    void emplace_entity(Entity&&... entity)
    {
        m_entity_container.emplace(std::forward<Entity>(entity)...);
    }

    template <typename Entity>
    PrimaryKey get_primary_key_value_for_entity(const Entity& entity) const noexcept
    {
        if constexpr(is_unordered_map_v<EntityContainerType>)
        {
            return m_entity_container.at(entity);
        }
        else
        {
            static_assert(false, "EntityContainerType is not an unordered map");
        }
    }

private:
    EntityContainerType m_entity_container;
};

struct writer::impl
{
    struct data_identifiers
    {
        using primary_key = size_t;

        ~data_identifiers() = default;

    private:
        entity_utility<std::unordered_set<writer_api::node_id_t>> node_info_utility{};
        entity_utility<std::unordered_map<writer_api::process_id_t, primary_key>>
            process_info_utility{};
        entity_utility<std::unordered_map<writer_api::agent_unique_id_t,
                                          primary_key,
                                          agent_unique_id_hash>>
            agent_info_utility{};
        entity_utility<std::unordered_map<writer_api::pmc_info_unique_id_t,
                                          primary_key,
                                          pmc_unique_id_hash>>
            pmc_info_utility{};
        entity_utility<std::unordered_map<writer_api::thread_id_t, primary_key>>
            thread_info_utility{};
        entity_utility<std::unordered_map<writer_api::stream_id_t, primary_key>>
            stream_info_utility{};
        entity_utility<std::unordered_map<writer_api::queue_id_t, primary_key>>
            queue_info_utility{};
        entity_utility<std::unordered_set<writer_api::code_object_id_t>>
            code_object_info_utility{};
        entity_utility<std::unordered_set<writer_api::kernel_symbol_id_t>>
            kernel_symbol_info_utility{};
        entity_utility<
            std::unordered_map<writer_api::track_info_t, primary_key, track_info_hash>>
            track_info_utility{};
        entity_utility<std::unordered_map<std::string, primary_key>>
            string_info_utility{};

    public:
        [[nodiscard]] auto& node_info() { return node_info_utility; }
        [[nodiscard]] auto& process_info() { return process_info_utility; }
        [[nodiscard]] auto& agent_info() { return agent_info_utility; }
        [[nodiscard]] auto& pmc_info() { return pmc_info_utility; }
        [[nodiscard]] auto& thread_info() { return thread_info_utility; }
        [[nodiscard]] auto& stream_info() { return stream_info_utility; }
        [[nodiscard]] auto& queue_info() { return queue_info_utility; }
        [[nodiscard]] auto& code_object_info() { return code_object_info_utility; }
        [[nodiscard]] auto& kernel_symbol_info() { return kernel_symbol_info_utility; }
        [[nodiscard]] auto& track_info() { return track_info_utility; }
        [[nodiscard]] auto& string_info() { return string_info_utility; }

    private:
        autoincrementer<primary_key> m_process_info_primary_key_provider{
            "process_info"
        };

        autoincrementer<primary_key> m_agent_info_primary_key_provider{ "agent_info" };
        autoincrementer<primary_key> m_pmc_info_primary_key_provider{ "pmc_info" };

        autoincrementer<primary_key> m_thread_info_primary_key_provider{ "thread_info" };
        autoincrementer<primary_key> m_stream_info_primary_key_provider{ "stream_info" };
        autoincrementer<primary_key> m_queue_info_primary_key_provider{ "queue_info" };
        autoincrementer<primary_key> m_track_info_primary_key_provider{ "track_info" };
        autoincrementer<primary_key> m_string_info_primary_key_provider{ "string_info" };

        autoincrementer<primary_key> m_event_data_primary_key_provider{ "event_data" };
        autoincrementer<primary_key> m_sample_data_primary_key_provider{ "sample_data" };
        autoincrementer<primary_key> m_region_data_primary_key_provider{ "region_data" };
        autoincrementer<primary_key> m_arg_primary_key_provider{ "arg" };
        autoincrementer<primary_key> m_pmc_event_data_primary_key_provider{
            "pmc_event_data"
        };
        autoincrementer<primary_key> m_kernel_dispatch_data_primary_key_provider{
            "kernel_dispatch_data"
        };
        autoincrementer<primary_key> m_memory_copy_data_primary_key_provider{
            "memory_copy_data"
        };
        autoincrementer<primary_key> m_memory_alloc_data_primary_key_provider{
            "memory_alloc_data"
        };

    public:
        [[nodiscard]] auto& process_info_primary_key_provider()
        {
            return m_process_info_primary_key_provider;
        }
        [[nodiscard]] auto& agent_info_primary_key_provider()
        {
            return m_agent_info_primary_key_provider;
        }
        [[nodiscard]] auto& pmc_info_primary_key_provider()
        {
            return m_pmc_info_primary_key_provider;
        }
        [[nodiscard]] auto& thread_info_primary_key_provider()
        {
            return m_thread_info_primary_key_provider;
        }
        [[nodiscard]] auto& stream_info_primary_key_provider()
        {
            return m_stream_info_primary_key_provider;
        }
        [[nodiscard]] auto& queue_info_primary_key_provider()
        {
            return m_queue_info_primary_key_provider;
        }
        [[nodiscard]] auto& track_info_primary_key_provider()
        {
            return m_track_info_primary_key_provider;
        }
        [[nodiscard]] auto& string_info_primary_key_provider()
        {
            return m_string_info_primary_key_provider;
        }
        [[nodiscard]] auto& event_data_primary_key_provider()
        {
            return m_event_data_primary_key_provider;
        }
        [[nodiscard]] auto& sample_data_primary_key_provider()
        {
            return m_sample_data_primary_key_provider;
        }
        [[nodiscard]] auto& region_data_primary_key_provider()
        {
            return m_region_data_primary_key_provider;
        }
        [[nodiscard]] auto& arg_primary_key_provider()
        {
            return m_arg_primary_key_provider;
        }
        [[nodiscard]] auto& pmc_event_data_primary_key_provider()
        {
            return m_pmc_event_data_primary_key_provider;
        }
        [[nodiscard]] auto& kernel_dispatch_data_primary_key_provider()
        {
            return m_kernel_dispatch_data_primary_key_provider;
        }
        [[nodiscard]] auto& memory_copy_data_primary_key_provider()
        {
            return m_memory_copy_data_primary_key_provider;
        }
        [[nodiscard]] auto& memory_alloc_data_primary_key_provider()
        {
            return m_memory_alloc_data_primary_key_provider;
        }
    };

    class insert_validator
    {
    public:
        explicit insert_validator(const std::shared_ptr<data_identifiers>& identifiers)
        : m_identifiers(std::move(identifiers))
        {}

        insert_validator& require_node(
            const std::optional<writer_api::node_id_t>& node_id)
        {
            validate_required(m_identifiers->node_info(), node_id, "Node", "node_id");
            return *this;
        }

        insert_validator& require_node(writer_api::node_id_t node_id)
        {
            validate_direct(m_identifiers->node_info(), node_id, "Node", "node_id");
            return *this;
        }

        insert_validator& require_process(
            const std::optional<writer_api::process_id_t>& process_id)
        {
            validate_required(
                m_identifiers->process_info(), process_id, "Process", "pid");
            return *this;
        }

        insert_validator& require_process(writer_api::process_id_t process_id)
        {
            validate_direct(m_identifiers->process_info(), process_id, "Process", "pid");
            return *this;
        }

        insert_validator& require_thread(
            const std::optional<writer_api::thread_id_t>& thread_id)
        {
            validate_required(
                m_identifiers->thread_info(), thread_id, "Thread", "thread_id");
            return *this;
        }

        insert_validator& require_agent(
            const std::optional<writer_api::agent_unique_id_t>& agent_id)
        {
            validate_required(m_identifiers->agent_info(), agent_id, "Agent", "agent_id");
            return *this;
        }

        insert_validator& require_agent(const writer_api::agent_unique_id_t& agent_id)
        {
            validate_direct(m_identifiers->agent_info(), agent_id, "Agent", "agent_id");
            return *this;
        }

        insert_validator& require_queue(
            const std::optional<writer_api::queue_id_t>& queue_id)
        {
            validate_required(m_identifiers->queue_info(), queue_id, "Queue", "queue_id");
            return *this;
        }

        insert_validator& require_stream(
            const std::optional<writer_api::stream_id_t>& stream_id)
        {
            validate_required(
                m_identifiers->stream_info(), stream_id, "Stream", "stream_id");
            return *this;
        }

        insert_validator& require_kernel_symbol(
            writer_api::kernel_symbol_id_t kernel_symbol_id)
        {
            if(!m_identifiers->kernel_symbol_info().is_entry_registered(kernel_symbol_id))
            {
                throw std::runtime_error(fmt::format(
                    "Kernel symbol not registered: kernel_id: {}", kernel_symbol_id));
            }
            return *this;
        }

        insert_validator& require_code_object(writer_api::code_object_id_t code_object_id)
        {
            if(!m_identifiers->code_object_info().is_entry_registered(code_object_id))
            {
                throw std::runtime_error(fmt::format(
                    "Code object not registered: code_obj_id: {}", code_object_id));
            }
            return *this;
        }

        insert_validator& require_pmc(
            const writer_api::pmc_info_unique_id_t& pmc_unique_id)
        {
            if(!m_identifiers->pmc_info().is_entry_registered(pmc_unique_id))
            {
                throw std::runtime_error(fmt::format(
                    "PMC Info not registered: pmc_name: {}", pmc_unique_id.name));
            }
            return *this;
        }

        insert_validator& validate_optional_thread(
            const std::optional<writer_api::thread_id_t>& thread_id)
        {
            validate_optional(
                m_identifiers->thread_info(), thread_id, "Thread", "thread_id");
            return *this;
        }

        insert_validator& validate_optional_process(
            const std::optional<writer_api::process_id_t>& process_id)
        {
            validate_optional(
                m_identifiers->process_info(), process_id, "Process", "pid");
            return *this;
        }

        insert_validator& validate_optional_agent(
            const std::optional<writer_api::agent_unique_id_t>& agent_id,
            std::string_view                                    agent_role = "Agent")
        {
            validate_optional(
                m_identifiers->agent_info(), agent_id, agent_role, "agent_id");
            return *this;
        }

        insert_validator& validate_optional_queue(
            const std::optional<writer_api::queue_id_t>& queue_id)
        {
            validate_optional(m_identifiers->queue_info(), queue_id, "Queue", "queue_id");
            return *this;
        }

        insert_validator& validate_optional_stream(
            const std::optional<writer_api::stream_id_t>& stream_id)
        {
            validate_optional(
                m_identifiers->stream_info(), stream_id, "Stream", "stream_id");
            return *this;
        }

        [[nodiscard]] primary_key resolve_process_key(
            const std::optional<writer_api::process_id_t>& process_id) const
        {
            return m_identifiers->process_info().get_primary_key_value_for_entity(
                process_id.value());
        }

        [[nodiscard]] primary_key resolve_process_key(
            writer_api::process_id_t process_id) const
        {
            return m_identifiers->process_info().get_primary_key_value_for_entity(
                process_id);
        }

        [[nodiscard]] primary_key resolve_thread_key(
            const std::optional<writer_api::thread_id_t>& thread_id) const
        {
            return m_identifiers->thread_info().get_primary_key_value_for_entity(
                thread_id.value());
        }

        [[nodiscard]] primary_key resolve_agent_key(
            const std::optional<writer_api::agent_unique_id_t>& agent_id) const
        {
            return m_identifiers->agent_info().get_primary_key_value_for_entity(
                agent_id.value());
        }

        [[nodiscard]] primary_key resolve_agent_key(
            const writer_api::agent_unique_id_t& agent_id) const
        {
            return m_identifiers->agent_info().get_primary_key_value_for_entity(agent_id);
        }

        [[nodiscard]] primary_key resolve_queue_key(
            const std::optional<writer_api::queue_id_t>& queue_id) const
        {
            return m_identifiers->queue_info().get_primary_key_value_for_entity(
                queue_id.value());
        }

        [[nodiscard]] primary_key resolve_stream_key(
            const std::optional<writer_api::stream_id_t>& stream_id) const
        {
            return m_identifiers->stream_info().get_primary_key_value_for_entity(
                stream_id.value());
        }

        [[nodiscard]] primary_key resolve_pmc_key(
            const writer_api::pmc_info_unique_id_t& pmc_unique_id) const
        {
            return m_identifiers->pmc_info().get_primary_key_value_for_entity(
                pmc_unique_id);
        }

        [[nodiscard]] std::optional<primary_key> resolve_optional_process_key(
            const std::optional<writer_api::process_id_t>& process_id) const
        {
            return resolve_optional_key(m_identifiers->process_info(), process_id);
        }

        [[nodiscard]] std::optional<primary_key> resolve_optional_thread_key(
            const std::optional<writer_api::thread_id_t>& thread_id) const
        {
            return resolve_optional_key(m_identifiers->thread_info(), thread_id);
        }

        [[nodiscard]] std::optional<primary_key> resolve_optional_agent_key(
            const std::optional<writer_api::agent_unique_id_t>& agent_id) const
        {
            return resolve_optional_key(m_identifiers->agent_info(), agent_id);
        }

        [[nodiscard]] std::optional<primary_key> resolve_optional_queue_key(
            const std::optional<writer_api::queue_id_t>& queue_id) const
        {
            return resolve_optional_key(m_identifiers->queue_info(), queue_id);
        }

        [[nodiscard]] std::optional<primary_key> resolve_optional_stream_key(
            const std::optional<writer_api::stream_id_t>& stream_id) const
        {
            return resolve_optional_key(m_identifiers->stream_info(), stream_id);
        }

        [[nodiscard]] std::optional<primary_key> resolve_optional_string_key(
            const std::optional<std::string>& str) const
        {
            if(!str.has_value()) return std::nullopt;
            return m_identifiers->string_info().get_primary_key_value_for_entity(
                str.value());
        }

        [[nodiscard]] data_identifiers& identifiers() const { return *m_identifiers; }

    private:
        template <typename Utility, typename EntityId>
        void validate_required(Utility&                       utility,
                               const std::optional<EntityId>& entity_id,
                               std::string_view               entity_name,
                               std::string_view               field_name)
        {
            if(!entity_id.has_value() || !utility.is_entry_registered(entity_id.value()))
            {
                throw std::runtime_error(fmt::format("{} not registered: {}: {}",
                                                     entity_name,
                                                     field_name,
                                                     to_error_string(entity_id)));
            }
        }

        template <typename Utility, typename EntityId>
        void validate_direct(Utility&         utility,
                             const EntityId&  entity_id,
                             std::string_view entity_name,
                             std::string_view field_name)
        {
            if(!utility.is_entry_registered(entity_id))
            {
                throw std::runtime_error(fmt::format("{} not registered: {}: {}",
                                                     entity_name,
                                                     field_name,
                                                     std::to_string(entity_id)));
            }
        }

        template <typename Utility, typename EntityId>
        void validate_optional(Utility&                       utility,
                               const std::optional<EntityId>& entity_id,
                               std::string_view               entity_name,
                               std::string_view               field_name)
        {
            if(entity_id.has_value() && !utility.is_entry_registered(entity_id.value()))
            {
                throw std::runtime_error(fmt::format("{} not registered: {}: {}",
                                                     entity_name,
                                                     field_name,
                                                     std::to_string(entity_id.value())));
            }
        }

        std::shared_ptr<data_identifiers> m_identifiers;
    };

public:
    explicit impl(std::shared_ptr<data_storage::database> database, std::string uuid)
    : m_database(std::move(database))
    , m_uuid(std::move(uuid))
    , m_entity_identifiers(std::make_shared<data_identifiers>())
    , m_validator(std::make_shared<insert_validator>(m_entity_identifiers))
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
        auto&      node_info_utility = m_entity_identifiers->node_info();
        const bool is_node_registered =
            node_info_utility.is_entry_registered(node_info.node_id);

        if(is_node_registered)
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

        node_info_utility.emplace_entity(node_info.node_id);
    }

    void register_process_info(const writer_api::process_info_t& process_info)
    {
        auto& process_info_utility = m_entity_identifiers->process_info();
        if(process_info_utility.is_entry_registered(process_info.pid))
        {
            LOG_WARNING("Process already registered: pid: {}", process_info.pid);
            return;
        }

        m_validator->require_node(process_info.node_id);

        const auto primary_key = m_entity_identifiers->process_info_primary_key_provider()
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

        process_info_utility.emplace_entity(process_info.pid, primary_key);
    }

    void register_agent_info(const writer_api::agent_info_t& agent_info)
    {
        auto& agent_info_utility = m_entity_identifiers->agent_info();
        if(agent_info_utility.is_entry_registered(agent_info.unique_id))
        {
            LOG_WARNING("Agent already registered: type: {}, index: {}, name: {}",
                        agent_info.unique_id.agent_type,
                        agent_info.unique_id.type_index,
                        agent_info.name);
            return;
        }

        m_validator->require_node(agent_info.node_id)
            .require_process(agent_info.process_id);

        const std::string_view agent_type{ agent_info.unique_id.agent_type };
        if(agent_type != "CPU" && agent_type != "GPU")
        {
            throw std::runtime_error(fmt::format(
                "Invalid agent type: {}. Type can be NULL, CPU, or GPU.", agent_type));
        }

        const auto process_pk  = m_validator->resolve_process_key(agent_info.process_id);
        const auto primary_key = m_entity_identifiers->agent_info_primary_key_provider()
                                     .get_primary_key_value();

        m_insert_statements->agent_info_statement()(primary_key,
                                                    agent_info.node_id,
                                                    process_pk,
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

        agent_info_utility.emplace_entity(agent_info.unique_id, primary_key);
    }

    void register_pmc_info(const writer_api::pmc_info_t& pmc_info)
    {
        auto& pmc_info_utility = m_entity_identifiers->pmc_info();
        if(pmc_info_utility.is_entry_registered(pmc_info.unique_id))
        {
            LOG_WARNING("PMC already registered: name: {}, agent_id: {}",
                        pmc_info.unique_id.name,
                        pmc_info.unique_id.agent_id->agent_type);
            return;
        }

        m_validator->require_node(pmc_info.node_id)
            .require_process(pmc_info.process_id)
            .require_agent(*pmc_info.unique_id.agent_id);

        const auto process_pk = m_validator->resolve_process_key(pmc_info.process_id);
        const auto agent_pk =
            m_validator->resolve_agent_key(*pmc_info.unique_id.agent_id);
        const auto primary_key =
            m_entity_identifiers->pmc_info_primary_key_provider().get_primary_key_value();

        m_insert_statements->pmc_info_statement()(primary_key,
                                                  pmc_info.node_id,
                                                  process_pk,
                                                  agent_pk,
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

        pmc_info_utility.emplace_entity(pmc_info.unique_id, primary_key);
    }

    void register_thread_info(const writer_api::thread_info_t& thread_info)
    {
        auto& thread_info_utility = m_entity_identifiers->thread_info();
        if(thread_info_utility.is_entry_registered(thread_info.thread_id))
        {
            LOG_WARNING("Thread already registered: thread_id: {}",
                        thread_info.thread_id);
            return;
        }

        m_validator->require_node(thread_info.node_id)
            .require_process(thread_info.process_id);

        const auto process_pk  = m_validator->resolve_process_key(thread_info.process_id);
        const auto primary_key = m_entity_identifiers->thread_info_primary_key_provider()
                                     .get_primary_key_value();

        m_insert_statements->thread_info_statement()(primary_key,
                                                     thread_info.node_id,
                                                     thread_info.parent_process_id,
                                                     process_pk,
                                                     thread_info.thread_id,
                                                     thread_info.name,
                                                     thread_info.start,
                                                     thread_info.end,
                                                     thread_info.extdata);

        thread_info_utility.emplace_entity(thread_info.thread_id, primary_key);
    }

    void register_stream_info(const writer_api::stream_info_t& stream_info)
    {
        auto& stream_info_utility = m_entity_identifiers->stream_info();
        if(stream_info_utility.is_entry_registered(stream_info.stream_id))
        {
            LOG_WARNING("Stream already registered: stream_id: {}",
                        stream_info.stream_id);
            return;
        }

        m_validator->require_node(stream_info.node_id)
            .require_process(stream_info.process_id);

        const auto process_pk  = m_validator->resolve_process_key(stream_info.process_id);
        const auto primary_key = m_entity_identifiers->stream_info_primary_key_provider()
                                     .get_primary_key_value();

        m_insert_statements->stream_info_statement()(primary_key,
                                                     stream_info.node_id,
                                                     process_pk,
                                                     stream_info.name,
                                                     stream_info.extdata);

        stream_info_utility.emplace_entity(stream_info.stream_id, primary_key);
    }

    void register_queue_info(const writer_api::queue_info_t& queue_info)
    {
        auto& queue_info_utility = m_entity_identifiers->queue_info();
        if(queue_info_utility.is_entry_registered(queue_info.queue_id))
        {
            LOG_WARNING("Queue already registered: queue_id: {}", queue_info.queue_id);
            return;
        }

        m_validator->require_node(queue_info.node_id)
            .require_process(queue_info.process_id);

        const auto process_pk  = m_validator->resolve_process_key(queue_info.process_id);
        const auto primary_key = m_entity_identifiers->queue_info_primary_key_provider()
                                     .get_primary_key_value();

        m_insert_statements->queue_info_statement()(primary_key,
                                                    queue_info.node_id,
                                                    process_pk,
                                                    queue_info.name,
                                                    queue_info.extdata);

        queue_info_utility.emplace_entity(queue_info.queue_id, primary_key);
    }

    void register_code_object_info(const writer_api::code_object_info_t& code_object_info)
    {
        auto& code_object_info_utility = m_entity_identifiers->code_object_info();
        if(code_object_info_utility.is_entry_registered(code_object_info.id))
        {
            LOG_WARNING("Code object already registered: id: {}", code_object_info.id);
            return;
        }

        m_validator->require_node(code_object_info.node_id)
            .require_process(code_object_info.process_id)
            .validate_optional_agent(code_object_info.agent_id);

        const auto process_pk =
            m_validator->resolve_process_key(code_object_info.process_id);
        const auto agent_pk =
            m_validator->resolve_optional_agent_key(code_object_info.agent_id);

        m_insert_statements->code_object_info_statement()(code_object_info.id,
                                                          code_object_info.node_id,
                                                          process_pk,
                                                          agent_pk,
                                                          code_object_info.uri,
                                                          code_object_info.ld_base,
                                                          code_object_info.ld_size,
                                                          code_object_info.ld_delta,
                                                          code_object_info.storage_type,
                                                          code_object_info.extdata);

        code_object_info_utility.emplace_entity(code_object_info.id);
    }

    void register_kernel_symbol_info(
        const writer_api::kernel_symbol_info_t& kernel_symbol_info)
    {
        auto& kernel_symbol_info_utility = m_entity_identifiers->kernel_symbol_info();
        if(kernel_symbol_info_utility.is_entry_registered(kernel_symbol_info.id))
        {
            LOG_WARNING("Kernel symbol already registered: id: {}",
                        kernel_symbol_info.id);
            return;
        }

        m_validator->require_node(kernel_symbol_info.node_id)
            .require_process(kernel_symbol_info.process_id)
            .require_code_object(kernel_symbol_info.code_obj_id);

        const auto process_pk =
            m_validator->resolve_process_key(kernel_symbol_info.process_id);

        m_insert_statements->kernel_symbol_info_statement()(
            kernel_symbol_info.id,
            kernel_symbol_info.node_id,
            process_pk,
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

        kernel_symbol_info_utility.emplace_entity(kernel_symbol_info.id);
    }

    void register_track_info(const writer_api::track_info_t& track)
    {
        auto& track_info_utility = m_entity_identifiers->track_info();
        if(track_info_utility.is_entry_registered(track))
        {
            LOG_WARNING("Track already registered: node_id: {}, process_id: {}, "
                        "thread_id: {}, name: {}",
                        track.node_id,
                        to_error_string(track.process_id),
                        to_error_string(track.thread_id),
                        track.name.value_or("NULL"));
            return;
        }

        m_validator->require_node(track.node_id)
            .validate_optional_process(track.process_id)
            .validate_optional_thread(track.thread_id);

        if(track.name.has_value() &&
           !m_entity_identifiers->string_info().is_entry_registered(track.name.value()))
        {
            register_string(track.name.value());
        }

        const auto process_pk =
            m_validator->resolve_optional_process_key(track.process_id);
        const auto thread_pk = m_validator->resolve_optional_thread_key(track.thread_id);
        const auto string_pk = m_validator->resolve_optional_string_key(track.name);
        const auto primary_key = m_entity_identifiers->track_info_primary_key_provider()
                                     .get_primary_key_value();

        m_insert_statements->track_info_statement()(
            primary_key, track.node_id, process_pk, thread_pk, string_pk, track.extdata);

        track_info_utility.emplace_entity(track, primary_key);
    }

    void register_string(const char* str)
    {
        auto& string_info_utility = m_entity_identifiers->string_info();

        if(str == nullptr)
        {
            throw std::runtime_error("Trying to register string that is null");
        }

        const auto is_string_registered = string_info_utility.is_entry_registered(str);

        if(is_string_registered)
        {
            LOG_WARNING("String already registered: str: {}", str);
            return;
        }

        const auto primary_key = m_entity_identifiers->string_info_primary_key_provider()
                                     .get_primary_key_value();

        m_insert_statements->string_statement()(primary_key, str);

        string_info_utility.emplace_entity(str, primary_key);
    }

    // --------------------- Data Tables ---------------------

private:
    primary_key insert_event(const writer_api::event_data_t& event_data)
    {
        auto& string_info_utility = m_entity_identifiers->string_info();

        auto json_call_stack_serializer = [](const writer_api::call_stack_t& call_stack) {
            return "{}";
        };

        auto json_line_info_serializer =
            [](const writer_api::source_context_list_t& line_info_list) { return "{}"; };

        const auto is_string_registered =
            string_info_utility.is_entry_registered(event_data.event_category);

        if(!is_string_registered)
        {
            register_string(event_data.event_category);
        }

        const auto event_category_primary_key =
            string_info_utility.get_primary_key_value_for_entity(
                event_data.event_category);
        const auto primary_key = m_entity_identifiers->event_data_primary_key_provider()
                                     .get_primary_key_value();

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

    void insert_sample(const writer_api::sample_data_t& sample_data,
                       const primary_key&               event_primary_key)
    {
        auto& track_info_utility = m_entity_identifiers->track_info();

        if(!track_info_utility.is_entry_registered(sample_data.track))
        {
            const auto track_name_print_value = sample_data.track.name.has_value()
                                                    ? sample_data.track.name.value()
                                                    : "[NULL]";

            throw std::runtime_error(
                fmt::format("Track not registered for Sample Data: track_name: {}",
                            track_name_print_value));
        }

        const auto track_primary_key =
            track_info_utility.get_primary_key_value_for_entity(sample_data.track);

        const auto primary_key = m_entity_identifiers->sample_data_primary_key_provider()
                                     .get_primary_key_value();
        m_insert_statements->sample_statement()(primary_key,
                                                track_primary_key,
                                                sample_data.timestamp,
                                                event_primary_key,
                                                sample_data.extdata);
    }

    inline void insert_arg(const writer_api::arg_data_t& arg_data, primary_key event_id)
    {
        auto& string_info_utility = m_entity_identifiers->string_info();

        if(arg_data.type == nullptr || arg_data.name == nullptr)
        {
            throw std::runtime_error(
                fmt::format("Type or name is null for Arg Data: type: {}, name: {}",
                            arg_data.type,
                            arg_data.name));
        }

        const auto is_string_registered =
            string_info_utility.is_entry_registered(arg_data.type);

        if(!is_string_registered)
        {
            register_string(arg_data.type);
        }

        const auto primary_key =
            m_entity_identifiers->arg_primary_key_provider().get_primary_key_value();

        m_insert_statements->arg_statement()(primary_key,
                                             event_id,
                                             arg_data.position,
                                             arg_data.type,
                                             arg_data.name,
                                             arg_data.value,
                                             arg_data.extdata);
    }

public:
    void insert_region_data(const writer_api::region_data_t&       region_data,
                            const writer_api::trace_environment_t& trace_environment)
    {
        m_validator->require_node(trace_environment.node_id)
            .require_process(trace_environment.process_id)
            .require_thread(trace_environment.thread_id);

        if(!region_data.event.has_value() && !region_data.args.empty())
        {
            throw std::runtime_error(fmt::format(
                "Writing args require providing event data for correlation: name: {}",
                region_data.name));
        }

        auto& string_info_utility = m_entity_identifiers->string_info();
        if(!string_info_utility.is_entry_registered(region_data.name))
            register_string(region_data.name);

        const auto process_pk =
            m_validator->resolve_process_key(trace_environment.process_id);
        const auto thread_pk =
            m_validator->resolve_thread_key(trace_environment.thread_id);
        const auto name_pk =
            string_info_utility.get_primary_key_value_for_entity(region_data.name);

        std::optional<primary_key> event_pk = std::nullopt;
        if(region_data.event.has_value())
        {
            event_pk = insert_event(region_data.event.value());
        }

        const auto primary_key = m_entity_identifiers->region_data_primary_key_provider()
                                     .get_primary_key_value();

        m_insert_statements->region_statement()(primary_key,
                                                trace_environment.node_id.value(),
                                                process_pk,
                                                thread_pk,
                                                region_data.start_timestamp,
                                                region_data.end_timestamp,
                                                name_pk,
                                                event_pk,
                                                region_data.extdata);

        for(const auto& arg : region_data.args)
            insert_arg(arg, event_pk.value());

        if(trace_environment.track_name.has_value() && event_pk.has_value())
        {
            const writer_api::track_info_t track_info = {
                trace_environment.track_name.value(),
                nullptr,
                trace_environment.node_id.value(),
                trace_environment.process_id.value(),
                trace_environment.thread_id.value()
            };
            const writer_api::sample_data_t sample_data = { region_data.start_timestamp,
                                                            track_info,
                                                            "{}" };
            insert_sample(sample_data, event_pk.value());
        }
    }

    void insert_pmc_event_data(const writer_api::pmc_event_data_t&     pmc_event_data,
                               const writer_api::pmc_info_unique_id_t& pmc_unique_id)
    {
        m_validator->require_pmc(pmc_unique_id);

        const auto pmc_pk = m_validator->resolve_pmc_key(pmc_unique_id);

        std::optional<primary_key> event_pk = std::nullopt;
        if(pmc_event_data.event.has_value())
        {
            event_pk = insert_event(pmc_event_data.event.value());
        }

        const auto primary_key =
            m_entity_identifiers->pmc_event_data_primary_key_provider()
                .get_primary_key_value();

        m_insert_statements->pmc_event_statement()(
            primary_key, event_pk, pmc_pk, pmc_event_data.value, pmc_event_data.extdata);
    }

    void insert_kernel_dispatch_data(
        const writer_api::kernel_dispatch_data_t& kernel_dispatch_data,
        const writer_api::trace_environment_t&    trace_environment)
    {
        m_validator->require_node(trace_environment.node_id)
            .require_process(trace_environment.process_id)
            .require_thread(trace_environment.thread_id)
            .require_agent(trace_environment.agent_id)
            .require_queue(trace_environment.queue_id)
            .require_stream(trace_environment.stream_id)
            .require_kernel_symbol(kernel_dispatch_data.kernel_symbol_id);

        auto& string_info_utility = m_entity_identifiers->string_info();
        if(!string_info_utility.is_entry_registered(kernel_dispatch_data.name))
            register_string(kernel_dispatch_data.name);

        const auto process_pk =
            m_validator->resolve_process_key(trace_environment.process_id);
        const auto thread_pk =
            m_validator->resolve_optional_thread_key(trace_environment.thread_id);
        const auto agent_pk = m_validator->resolve_agent_key(trace_environment.agent_id);
        const auto queue_pk = m_validator->resolve_queue_key(trace_environment.queue_id);
        const auto stream_pk =
            m_validator->resolve_stream_key(trace_environment.stream_id);
        const auto name_pk = string_info_utility.get_primary_key_value_for_entity(
            kernel_dispatch_data.name);

        std::optional<primary_key> event_pk = std::nullopt;
        if(kernel_dispatch_data.event.has_value())
        {
            event_pk = insert_event(kernel_dispatch_data.event.value());
        }

        const auto primary_key =
            m_entity_identifiers->kernel_dispatch_data_primary_key_provider()
                .get_primary_key_value();

        m_insert_statements->kernel_dispatch_statement()(
            primary_key,
            trace_environment.node_id.value(),
            process_pk,
            thread_pk,
            agent_pk,
            kernel_dispatch_data.kernel_symbol_id,
            kernel_dispatch_data.dispatch_id,
            queue_pk,
            stream_pk,
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
            name_pk,
            event_pk,
            kernel_dispatch_data.extdata);

        if(trace_environment.track_name.has_value() && event_pk.has_value())
        {
            const writer_api::track_info_t track_info = {
                trace_environment.track_name.value(),
                nullptr,
                trace_environment.node_id.value(),
                trace_environment.process_id.value(),
                trace_environment.thread_id.value()
            };
            const writer_api::sample_data_t sample_data = {
                kernel_dispatch_data.start_timestamp, track_info, "{}"
            };
            insert_sample(sample_data, event_pk.value());
        }
    }

    void insert_memory_copy_data(const writer_api::memory_copy_data_t&  memory_copy_data,
                                 const writer_api::trace_environment_t& trace_environment)
    {
        m_validator->require_node(trace_environment.node_id)
            .require_process(trace_environment.process_id)
            .validate_optional_thread(trace_environment.thread_id)
            .validate_optional_agent(memory_copy_data.src_agent_id, "Source agent")
            .validate_optional_agent(memory_copy_data.dst_agent_id, "Destination agent")
            .validate_optional_queue(trace_environment.queue_id)
            .validate_optional_stream(trace_environment.stream_id);

        auto& string_info_utility = m_entity_identifiers->string_info();
        if(!string_info_utility.is_entry_registered(memory_copy_data.name))
            register_string(memory_copy_data.name);
        if(memory_copy_data.region_name != nullptr &&
           !string_info_utility.is_entry_registered(memory_copy_data.region_name))
            register_string(memory_copy_data.region_name);

        const auto process_pk =
            m_validator->resolve_process_key(trace_environment.process_id);
        const auto thread_pk =
            m_validator->resolve_optional_thread_key(trace_environment.thread_id);
        const auto src_agent_pk =
            m_validator->resolve_optional_agent_key(memory_copy_data.src_agent_id);
        const auto dst_agent_pk =
            m_validator->resolve_optional_agent_key(memory_copy_data.dst_agent_id);
        const auto queue_pk =
            m_validator->resolve_optional_queue_key(trace_environment.queue_id);
        const auto stream_pk =
            m_validator->resolve_optional_stream_key(trace_environment.stream_id);
        const auto name_pk =
            string_info_utility.get_primary_key_value_for_entity(memory_copy_data.name);

        std::optional<primary_key> event_pk = std::nullopt;
        if(memory_copy_data.event.has_value())
        {
            event_pk = insert_event(memory_copy_data.event.value());
        }

        std::optional<primary_key> region_name_pk = std::nullopt;
        if(memory_copy_data.region_name != nullptr)
            region_name_pk = string_info_utility.get_primary_key_value_for_entity(
                memory_copy_data.region_name);

        const auto primary_key =
            m_entity_identifiers->memory_copy_data_primary_key_provider()
                .get_primary_key_value();

        m_insert_statements->memory_copy_statement()(primary_key,
                                                     trace_environment.node_id.value(),
                                                     process_pk,
                                                     thread_pk,
                                                     memory_copy_data.start_timestamp,
                                                     memory_copy_data.end_timestamp,
                                                     name_pk,
                                                     dst_agent_pk,
                                                     memory_copy_data.dst_address,
                                                     src_agent_pk,
                                                     memory_copy_data.src_address,
                                                     memory_copy_data.size,
                                                     queue_pk,
                                                     stream_pk,
                                                     region_name_pk,
                                                     event_pk,
                                                     memory_copy_data.extdata);

        if(trace_environment.track_name.has_value() && event_pk.has_value())
        {
            const writer_api::track_info_t track_info = {
                trace_environment.track_name.value(),
                nullptr,
                trace_environment.node_id.value(),
                trace_environment.process_id.value(),
                trace_environment.thread_id.value()
            };
            const writer_api::sample_data_t sample_data = {
                memory_copy_data.start_timestamp, track_info, "{}"
            };
            insert_sample(sample_data, event_pk.value());
        }
    }

    void insert_memory_alloc_data(
        const writer_api::memory_alloc_data_t& memory_alloc_data,
        const writer_api::trace_environment_t& trace_environment)
    {
        m_validator->require_node(trace_environment.node_id)
            .require_process(trace_environment.process_id)
            .validate_optional_thread(trace_environment.thread_id)
            .validate_optional_agent(trace_environment.agent_id)
            .validate_optional_queue(trace_environment.queue_id)
            .validate_optional_stream(trace_environment.stream_id);

        if(memory_alloc_data.type != nullptr)
        {
            constexpr std::array<std::string_view, 4> allowed_types = {
                "ALLOC", "FREE", "REALLOC", "RECLAIM"
            };
            if(std::find(allowed_types.begin(),
                         allowed_types.end(),
                         memory_alloc_data.type) == allowed_types.end())
            {
                throw std::runtime_error(fmt::format(
                    "Invalid type value for Memory Alloc Data: type: {}. Allowed: {}",
                    memory_alloc_data.type,
                    allowed_types));
            }
        }

        if(memory_alloc_data.level != nullptr)
        {
            constexpr std::array<std::string_view, 3> allowed_levels = { "REAL",
                                                                         "VIRTUAL",
                                                                         "SCRATCH" };
            if(std::find(allowed_levels.begin(),
                         allowed_levels.end(),
                         memory_alloc_data.level) == allowed_levels.end())
            {
                throw std::runtime_error(fmt::format(
                    "Invalid level value for Memory Alloc Data: level: {}. Allowed: {}",
                    memory_alloc_data.level,
                    allowed_levels));
            }
        }

        const auto process_pk =
            m_validator->resolve_process_key(trace_environment.process_id);
        const auto thread_pk =
            m_validator->resolve_optional_thread_key(trace_environment.thread_id);
        const auto agent_pk =
            m_validator->resolve_optional_agent_key(trace_environment.agent_id);
        const auto queue_pk =
            m_validator->resolve_optional_queue_key(trace_environment.queue_id);
        const auto stream_pk =
            m_validator->resolve_optional_stream_key(trace_environment.stream_id);

        std::optional<primary_key> event_pk = std::nullopt;
        if(memory_alloc_data.event.has_value())
        {
            event_pk = insert_event(memory_alloc_data.event.value());
        }

        const auto primary_key =
            m_entity_identifiers->memory_alloc_data_primary_key_provider()
                .get_primary_key_value();

        m_insert_statements->memory_alloc_statement()(primary_key,
                                                      trace_environment.node_id.value(),
                                                      process_pk,
                                                      thread_pk,
                                                      agent_pk,
                                                      memory_alloc_data.type,
                                                      memory_alloc_data.level,
                                                      memory_alloc_data.start_timestamp,
                                                      memory_alloc_data.end_timestamp,
                                                      memory_alloc_data.address,
                                                      memory_alloc_data.size,
                                                      queue_pk,
                                                      stream_pk,
                                                      event_pk,
                                                      memory_alloc_data.extdata);

        if(trace_environment.track_name.has_value() && event_pk.has_value())
        {
            const writer_api::track_info_t track_info = {
                trace_environment.track_name.value(),
                nullptr,
                trace_environment.node_id.value(),
                trace_environment.process_id.value(),
                trace_environment.thread_id.value()
            };
            const writer_api::sample_data_t sample_data = {
                memory_alloc_data.start_timestamp, track_info, "{}"
            };
            insert_sample(sample_data, event_pk.value());
        }
    }

    void flush_in_memory_data_to_disk() { m_database->flush(); }

    std::shared_ptr<data_storage::database>                     m_database;
    std::string                                                 m_uuid;
    std::unique_ptr<data_storage::schema_v3::insert_statements> m_insert_statements;
    std::shared_ptr<data_identifiers>                           m_entity_identifiers;
    std::shared_ptr<insert_validator>                           m_validator;
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
    m_impl->register_code_object_info(code_object);
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
