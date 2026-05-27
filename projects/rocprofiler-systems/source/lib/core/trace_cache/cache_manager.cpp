// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cache_manager.hpp"
#include <cstdint>

#include "core/agent_manager.hpp"
#include "core/config.hpp"
#include "core/output_file_registry.hpp"
#include "core/trace_cache/data_types.hpp"
#include "core/trace_cache/discovery.hpp"
#include "core/trace_cache/post_processor.hpp"
#include "library/runtime.hpp"
#include "logger/debug.hpp"

#include <memory>
#include <unistd.h>

namespace rocprofsys
{
namespace trace_cache
{

cache_manager&
cache_manager::get_instance()
{
    static cache_manager instance;
    return instance;
}

void
cache_manager::post_process_bulk(output_file_registry& _output_registry,
                                 progress::tracker&    _tracker)
{
    LOG_TRACE("Starting trace cache bulk post-processing");

    if(!is_root_process())
    {
        LOG_DEBUG("Not root process, skipping bulk post-processing");
        return;
    }

    if(m_storage.is_running())
    {
        LOG_WARNING("Post-processing called without previously shutting down cache "
                    "storage. Calling shutdown explicitly..");
        shutdown();
    }

    const auto root_pid = get_root_process_id();
    LOG_DEBUG("Root process ID: {}", root_pid);

    const auto temp_directory_content =
        discovery::list_dir_files(trace_cache::tmp_directory);
    LOG_TRACE("Found {} files in temp directory", temp_directory_content.size());

    const auto cache_files =
        discovery::find_cache_files(root_pid, temp_directory_content);
    LOG_DEBUG("Found {} cache file pairs to process", cache_files.size());

    if(config::output_filtering::is_file_output_enabled_for_current_mpi_rank())
    {
        const data::enabled_formats_t enabled_formats;
        enabled_formats.print();

        auto processor_configs = post_processor::make_configs(cache_files, root_pid);

        processor_configs.push_back(std::make_shared<data::processor_config_t>(
            getpid(), root_pid, m_metadata,
            std::make_shared<agent_manager>(get_agent_manager_instance().get_agents())));

        LOG_INFO("Processing {} trace cache configurations", processor_configs.size());
        post_processor processor{ _tracker, _output_registry };
        processor.process(processor_configs, enabled_formats);

        if(enabled_formats.is_perfetto_enabled() && get_merge_perfetto_files())
            discovery::merge_perfetto_files();
    }

    discovery::clear(cache_files);

    LOG_TRACE("Trace cache bulk post-processing completed");
}

void
cache_manager::shutdown()
{
    LOG_DEBUG("Shutting down cache manager storage");
    m_storage.shutdown();
    LOG_TRACE("Cache manager storage shutdown complete");
}

}  // namespace trace_cache
}  // namespace rocprofsys
