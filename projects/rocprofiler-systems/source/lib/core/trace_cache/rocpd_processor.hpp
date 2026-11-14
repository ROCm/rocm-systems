// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "agent_manager.hpp"
#include "core/agent.hpp"
#include "core/collection/schema.hpp"
#include "core/node_info.hpp"
#include "core/trace_cache/counter_samples.hpp"
#include "core/trace_cache/counter_schema_table.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_processor_interface.hpp"

#include "library/pmc/collectors/hipfile/sample.hpp"
#include "trace_cache/sample_type.hpp"

#include <profiler-hub/storage.hpp>
#include <profiler-hub/writer.hpp>
#include <profiler-hub/writer_types.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <unordered_set>
#include <vector>

namespace rocprofsys::trace_cache
{

class rocpd_processor_t : public sample_processor_interface
{
public:
    rocpd_processor_t(const std::shared_ptr<metadata_registry>& metadata,
                      const std::shared_ptr<agent_manager>& agent_mngr, pid_t pid,
                      pid_t ppid);

    void prepare_for_processing() override;
    void finalize_processing() override;

    void handle(const kernel_dispatch_sample& sample) override;
    void handle(const scratch_memory_sample& sample) override;
    void handle(const memory_copy_sample& sample) override;
    void handle(const memory_allocate_sample& sample) override;
    void handle(const region_sample& sample) override;
    void handle(const in_time_sample& sample) override;
    void handle(const pmc_event_with_sample& sample) override;
    void handle(const backtrace_region_sample& sample) override;
    void handle(const gpu_pmc_sample& sample) override;
    void handle(const ainic_pmc_sample& sample) override;
    void handle(const cpu_pmc_sample& sample) override;
    void handle(const gpu_perf_counter_sample& sample) override;
    void handle(const hipfile_pmc_sample& sample) override;
    void handle(const kfd_sample& sample) override;
    void handle(const counter_batch& batch) override;
    void handle(const gap_marker& gap) override;

private:
    /** One rocpd counter of a schema column, resolved once before processing. The views
     * point into the schemas held by m_counters. */
    struct counter_slot
    {
        profiler_hub::writer_types::pmc_info_unique_id_t uid;
        profiler_hub::writer_types::track_info_t         track;
        const char*                                      category = nullptr;
        /** Agent not found; warned at build. */
        bool skip      = false;
        bool had_value = false;
    };

    void post_process_metadata();
    /** Registers the registry tracks and those of the counter schemas. */
    void register_tracks(const std::vector<collection::schema>&   counter_schemas,
                         profiler_hub::writer_types::process_id_t process_id);
    /** Registers the registry PMC info and that of the counter schemas; a PMC whose
     * agent is unknown is skipped with a warning. */
    void register_pmc_infos(const std::vector<collection::schema>&   counter_schemas,
                            profiler_hub::writer_types::process_id_t process_id);
    void build_counter_slots();
    [[nodiscard]] counter_slot make_counter_slot(
        const collection::schema& description, const collection::column_spec& column,
        profiler_hub::writer_types::process_id_t process_id) const;
    void insert_counter_value(const counter_slot& slot, std::uint64_t timestamp,
                              double value);

    /** The agent of a PMC: by type index, then by id.
     * @throws std::out_of_range when neither exists */
    [[nodiscard]] const agent& find_pmc_agent(agent_type type, std::size_t index) const;

    /**
     * Try to insert a PMC event into the writer.
     *
     * If the PMC info is not registered, a warning will be logged and the event will
     * be dropped. If the PMC info has already been warned about, the event will be
     * dropped without warning.
     *
     * @param event_data The PMC event data to insert.
     * @param unique_id The unique ID of the PMC.
     * @param context The context of the PMC event. Used as a warning prefix: ex., "CPU
     * PMC sample".
     */
    void try_insert_pmc_event(
        const profiler_hub::writer_types::pmc_event_data_t&     event_data,
        const profiler_hub::writer_types::pmc_info_unique_id_t& unique_id,
        std::string_view                                        context);

    std::shared_ptr<metadata_registry>      m_metadata;
    std::shared_ptr<agent_manager>          m_agent_manager;
    std::unique_ptr<profiler_hub::writer_t> m_writer;
    pid_t                                   m_pid;
    std::string                             m_db_output_path;

    // PMC keys that have already been warned about
    std::unordered_set<std::string> m_unregistered_pmcs_already_warned;
    std::size_t                     m_dropped_pmc_events_count = 0;

    counter_schema_table<counter_slot> m_counters;
};

}  // namespace rocprofsys::trace_cache
