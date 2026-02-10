// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "backends/sqlite_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace rocstorage::data_storage::schema_v3
{
using integer_primary_key_t = size_t;
using integer_foreign_key_t = size_t;

struct insert_statements
{
    explicit insert_statements(std::shared_ptr<sqlite_backend> backend, std::string uuid);
    insert_statements()                                    = delete;
    insert_statements(const insert_statements&)            = delete;
    insert_statements(insert_statements&&)                 = delete;
    insert_statements& operator=(const insert_statements&) = delete;
    insert_statements& operator=(insert_statements&&)      = delete;
    virtual ~insert_statements()                           = default;

    using string_statement_func_t = std::function<void(size_t id, const char* string)>;

    using node_info_statement_func_t = std::function<void(integer_primary_key_t id,
                                                          size_t                hash,
                                                          const char* machine_id,
                                                          const char* system_name,
                                                          const char* hostname,
                                                          const char* release,
                                                          const char* version,
                                                          const char* hardware_name,
                                                          const char* domain_name)>;

    using process_info_statement_func_t = std::function<void(integer_primary_key_t id,
                                                             integer_foreign_key_t nid,
                                                             std::optional<size_t> ppid,
                                                             size_t                pid,
                                                             std::optional<size_t> init,
                                                             std::optional<size_t> fini,
                                                             std::optional<size_t> start,
                                                             std::optional<size_t> end,
                                                             const char* command,
                                                             const char* environment,
                                                             const char* extdata)>;

    using thread_info_statement_func_t = std::function<void(integer_primary_key_t id,
                                                            integer_foreign_key_t nid,
                                                            std::optional<size_t> ppid,
                                                            integer_foreign_key_t pid,
                                                            size_t                tid,
                                                            const char*           name,
                                                            std::optional<size_t> start,
                                                            std::optional<size_t> end,
                                                            const char* extdata)>;

    // Set type to nullptr if type is not available
    using agent_info_statement_func_t =
        std::function<void(integer_primary_key_t id,
                           integer_foreign_key_t nid,
                           integer_foreign_key_t pid,
                           const char*           type,
                           std::optional<size_t> absolute_index,
                           std::optional<size_t> logical_index,
                           size_t                type_index,
                           std::optional<size_t> uuid,
                           const char*           name,
                           const char*           model_name,
                           const char*           vendor_name,
                           const char*           product_name,
                           const char*           user_name,
                           const char*           extdata)>;

    using queue_info_statement_func_t = std::function<void(integer_primary_key_t id,
                                                           integer_foreign_key_t nid,
                                                           integer_foreign_key_t pid,
                                                           const char*           name,
                                                           const char* extdata)>;

    using stream_info_statement_func_t = std::function<void(integer_primary_key_t id,
                                                            integer_foreign_key_t nid,
                                                            integer_foreign_key_t pid,
                                                            const char*           name,
                                                            const char* extdata)>;

    using pmc_info_statement_func_t =
        std::function<void(integer_primary_key_t                id,
                           integer_foreign_key_t                nid,
                           integer_foreign_key_t                pid,
                           std::optional<integer_foreign_key_t> agent_id,
                           const char*                          target_arch,
                           std::optional<size_t>                event_code,
                           std::optional<size_t>                instance_id,
                           const char*                          name,
                           const char*                          symbol,
                           const char*                          description,
                           const char*                          long_description,
                           const char*                          component,
                           const char*                          units,
                           const char*                          value_type,
                           const char*                          block,
                           const char*                          expression,
                           std::optional<size_t>                is_constant,
                           std::optional<size_t>                is_derived,
                           const char*                          extdata)>;

    using code_object_info_statement_func_t =
        std::function<void(integer_primary_key_t                id,
                           integer_foreign_key_t                nid,
                           integer_foreign_key_t                pid,
                           std::optional<integer_foreign_key_t> agent_id,
                           const char*                          uri,
                           std::optional<size_t>                load_base,
                           std::optional<size_t>                load_size,
                           std::optional<size_t>                load_delta,
                           const char*                          storage_type,
                           const char*                          extdata)>;

    using kernel_symbol_info_statement_func_t =
        std::function<void(integer_primary_key_t id,
                           integer_foreign_key_t nid,
                           integer_foreign_key_t pid,
                           integer_foreign_key_t code_object_id,
                           const char*           kernel_name,
                           const char*           display_name,
                           std::optional<size_t> kernel_object,
                           std::optional<size_t> kernarg_segment_size,
                           std::optional<size_t> kernarg_segment_alignment,
                           std::optional<size_t> group_segment_size,
                           std::optional<size_t> private_segment_size,
                           std::optional<size_t> sgpr_count,
                           std::optional<size_t> arch_vgpr_count,
                           std::optional<size_t> accum_vgpr_count,
                           const char*           extdata)>;

    using track_info_statement_func_t =
        std::function<void(integer_primary_key_t                id,
                           integer_foreign_key_t                nid,
                           std::optional<integer_foreign_key_t> pid,
                           std::optional<integer_foreign_key_t> tid,
                           std::optional<integer_foreign_key_t> name_id,
                           const char*                          extdata)>;

    using event_statement_func_t =
        std::function<void(integer_primary_key_t                id,
                           std::optional<integer_foreign_key_t> category_id,
                           std::optional<size_t>                stack_id,
                           std::optional<size_t>                parent_stack_id,
                           std::optional<size_t>                correlation_id,
                           const char*                          call_stack,
                           const char*                          line_info,
                           const char*                          extdata)>;

    using arg_statement_func_t = std::function<void(integer_primary_key_t id,
                                                    integer_foreign_key_t event_id,
                                                    size_t                position,
                                                    const char*           type,
                                                    const char*           name,
                                                    const char*           value,
                                                    const char*           extdata)>;

    using pmc_event_statement_func_t =
        std::function<void(integer_primary_key_t                id,
                           std::optional<integer_foreign_key_t> event_id,
                           integer_foreign_key_t                pmc_id,
                           double                               value,
                           const char*                          extdata)>;

    using region_statement_func_t =
        std::function<void(integer_primary_key_t                id,
                           integer_foreign_key_t                nid,
                           integer_foreign_key_t                pid,
                           integer_foreign_key_t                tid,
                           uint64_t                             start,
                           uint64_t                             end,
                           integer_foreign_key_t                name_id,
                           std::optional<integer_foreign_key_t> event_id,
                           const char*                          extdata)>;

    using sample_statement_func_t =
        std::function<void(integer_primary_key_t                id,
                           integer_foreign_key_t                track_id,
                           uint64_t                             timestamp,
                           std::optional<integer_foreign_key_t> event_id,
                           const char*                          extdata)>;

    using kernel_dispatch_statement_func_t =
        std::function<void(integer_primary_key_t                id,
                           integer_foreign_key_t                nid,
                           integer_foreign_key_t                pid,
                           std::optional<integer_foreign_key_t> tid,
                           integer_foreign_key_t                agent_id,
                           integer_foreign_key_t                kernel_id,
                           size_t                               dispatch_id,
                           integer_foreign_key_t                queue_id,
                           integer_foreign_key_t                stream_id,
                           uint64_t                             start,
                           uint64_t                             end,
                           std::optional<size_t>                private_segment_size,
                           std::optional<size_t>                group_segment_size,
                           size_t                               workgroup_size_x,
                           size_t                               workgroup_size_y,
                           size_t                               workgroup_size_z,
                           size_t                               grid_size_x,
                           size_t                               grid_size_y,
                           size_t                               grid_size_z,
                           std::optional<integer_foreign_key_t> region_name_id,
                           std::optional<integer_foreign_key_t> event_id,
                           const char*                          extdata)>;

    using memory_copy_statement_func_t =
        std::function<void(integer_primary_key_t                id,
                           integer_foreign_key_t                nid,
                           integer_foreign_key_t                pid,
                           std::optional<integer_foreign_key_t> tid,
                           uint64_t                             start,
                           uint64_t                             end,
                           integer_foreign_key_t                name_id,
                           std::optional<integer_foreign_key_t> dst_agent_id,
                           std::optional<size_t>                dst_address,
                           std::optional<integer_foreign_key_t> src_agent_id,
                           std::optional<size_t>                src_address,
                           size_t                               size,
                           std::optional<integer_foreign_key_t> queue_id,
                           std::optional<integer_foreign_key_t> stream_id,
                           std::optional<integer_foreign_key_t> region_name_id,
                           std::optional<integer_foreign_key_t> event_id,
                           const char*                          extdata)>;

    using memory_alloc_statement_func_t =
        std::function<void(integer_primary_key_t                id,
                           integer_foreign_key_t                nid,
                           integer_foreign_key_t                pid,
                           std::optional<integer_foreign_key_t> tid,
                           std::optional<integer_foreign_key_t> agent_id,
                           const char*                          type,
                           const char*                          level,
                           uint64_t                             start,
                           uint64_t                             end,
                           std::optional<size_t>                address,
                           size_t                               size,
                           std::optional<integer_foreign_key_t> queue_id,
                           std::optional<integer_foreign_key_t> stream_id,
                           std::optional<integer_foreign_key_t> event_id,
                           const char*                          extdata)>;

public:
    [[nodiscard]] const string_statement_func_t& string_statement() const
    {
        return m_string_statement;
    }

    [[nodiscard]] const node_info_statement_func_t& node_info_statement() const
    {
        return m_node_info_statement;
    }

    [[nodiscard]] const process_info_statement_func_t& process_info_statement() const
    {
        return m_process_info_statement;
    }

    [[nodiscard]] const agent_info_statement_func_t& agent_info_statement() const
    {
        return m_agent_info_statement;
    }

    [[nodiscard]] const pmc_info_statement_func_t& pmc_info_statement() const
    {
        return m_pmc_info_statement;
    }

    [[nodiscard]] const thread_info_statement_func_t& thread_info_statement() const
    {
        return m_thread_info_statement;
    }

    [[nodiscard]] const stream_info_statement_func_t& stream_info_statement() const
    {
        return m_stream_info_statement;
    }

    [[nodiscard]] const queue_info_statement_func_t& queue_info_statement() const
    {
        return m_queue_info_statement;
    }

    [[nodiscard]] const kernel_symbol_info_statement_func_t&
    kernel_symbol_info_statement() const
    {
        return m_kernel_symbol_info_statement;
    }

    [[nodiscard]] const code_object_info_statement_func_t& code_object_info_statement()
        const
    {
        return m_code_object_info_statement;
    }

    [[nodiscard]] const track_info_statement_func_t& track_info_statement() const
    {
        return m_track_info_statement;
    }

    [[nodiscard]] const event_statement_func_t& event_statement() const
    {
        return m_event_statement;
    }

    [[nodiscard]] const arg_statement_func_t& arg_statement() const
    {
        return m_arg_statement;
    }

    [[nodiscard]] const pmc_event_statement_func_t& pmc_event_statement() const
    {
        return m_pmc_event_statement;
    }

    [[nodiscard]] const region_statement_func_t& region_statement() const
    {
        return m_region_statement;
    }

    [[nodiscard]] const sample_statement_func_t& sample_statement() const
    {
        return m_sample_statement;
    }

    [[nodiscard]] const kernel_dispatch_statement_func_t& kernel_dispatch_statement()
        const
    {
        return m_kernel_dispatch_statement;
    }

    [[nodiscard]] const memory_copy_statement_func_t& memory_copy_statement() const
    {
        return m_memory_copy_statement;
    }

    [[nodiscard]] const memory_alloc_statement_func_t& memory_alloc_statement() const
    {
        return m_memory_alloc_statement;
    }

private:
    void initialize_string_statement();
    void initialize_node_info_statement();
    void initialize_process_info_statement();
    void initialize_agent_info_statement();
    void initialize_pmc_info_statement();
    void initialize_thread_info_statement();
    void initialize_stream_info_statement();
    void initialize_queue_info_statement();
    void initialize_kernel_symbol_info_statement();
    void initialize_code_object_info_statement();
    void initialize_track_info_statement();
    void initialize_event_statement();
    void initialize_arg_statement();
    void initialize_pmc_event_statement();
    void initialize_region_statement();
    void initialize_sample_statement();
    void initialize_kernel_dispatch_statement();
    void initialize_memory_copy_statement();
    void initialize_memory_alloc_statement();

    std::shared_ptr<sqlite_backend> m_backend;
    std::string                     m_uuid;

    string_statement_func_t             m_string_statement;
    node_info_statement_func_t          m_node_info_statement;
    process_info_statement_func_t       m_process_info_statement;
    agent_info_statement_func_t         m_agent_info_statement;
    pmc_info_statement_func_t           m_pmc_info_statement;
    thread_info_statement_func_t        m_thread_info_statement;
    stream_info_statement_func_t        m_stream_info_statement;
    queue_info_statement_func_t         m_queue_info_statement;
    kernel_symbol_info_statement_func_t m_kernel_symbol_info_statement;
    code_object_info_statement_func_t   m_code_object_info_statement;
    track_info_statement_func_t         m_track_info_statement;
    event_statement_func_t              m_event_statement;
    arg_statement_func_t                m_arg_statement;
    pmc_event_statement_func_t          m_pmc_event_statement;
    region_statement_func_t             m_region_statement;
    sample_statement_func_t             m_sample_statement;
    kernel_dispatch_statement_func_t    m_kernel_dispatch_statement;
    memory_copy_statement_func_t        m_memory_copy_statement;
    memory_alloc_statement_func_t       m_memory_alloc_statement;
};

}  // namespace rocstorage::data_storage::schema_v3
