// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "agent_manager.hpp"
#include "core/node_info.hpp"
#include "core/trace_cache/cpu_snap_decoder.hpp"
#include "core/trace_cache/data_types.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_processor_interface.hpp"

#include "library/pmc/collectors/hipfile/sample.hpp"
#include "trace_cache/sample_type.hpp"

#include <profiler-hub/cpp/storage.hpp>
#include <profiler-hub/cpp/writer.hpp>
#include <profiler-hub/cpp/writer_types.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocprofsys::trace_cache
{

class rocpd_processor_t : public sample_processor_interface
{
public:
    /**
     * @param processor_config                 The process whose cache file this reads.
     * @param cpu_snap_skipped_module_prefixes Passed to the cpu-snap decoder.
     */
    rocpd_processor_t(const data::processor_config_t& processor_config,
                      std::vector<std::string>        cpu_snap_skipped_module_prefixes);

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
    void handle(const cpu_snap_sample& sample) override;

private:
    void post_process_metadata();

    // Writes one slice of the experimental call-stack sampler on its thread's track. A
    // slice of a thread missing from the metadata is dropped and counted: profiler-hub
    // throws for an unregistered thread, which would lose the whole process file.
    void handle_cpu_snap_slice(const backtrace_region_sample& slice);

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

    // Declared after m_metadata, which its constructor reads.
    cpu_snap_decoder m_cpu_snap_decoder;
    // Threads registered from the metadata, by tid; true once the thread's cpu-snap
    // track is registered too.
    std::unordered_map<std::uint64_t, bool> m_cpu_snap_track_registered;
    std::unordered_set<std::uint64_t>       m_cpu_snap_dropped_threads_tids;
    std::size_t                             m_cpu_snap_dropped_slices_count = 0;
};

}  // namespace rocprofsys::trace_cache
