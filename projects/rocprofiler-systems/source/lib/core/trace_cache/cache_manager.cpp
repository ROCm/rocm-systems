// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cache_manager.hpp"
#include <cstdint>

#include "core/agent_manager.hpp"
#include "core/config.hpp"
#include "core/output_file_registry.hpp"
#include "core/perfetto/engine.hpp"
#include "core/perfetto/sinks.hpp"
#include "core/trace_cache/data_types.hpp"
#include "core/trace_cache/discovery.hpp"
#include "core/trace_cache/post_processor.hpp"
#include "core/track_registry.hpp"
#include "library/runtime.hpp"
#include "logger/debug.hpp"

#include <memory>
#include <unistd.h>

namespace rocprofsys
{
namespace trace_cache
{
namespace
{
// RAII orchestrator for the cached-perfetto post-processing pipeline:
// owns the perfetto_engine + trace_sink + track_registry trio, wires the
// engine into the active post_processor on construction, and drives the
// engine's stop()/teardown on destruction so the destructor catches any
// drain exception. Construction may throw if init_sdk fails; partial
// state is unwound by member destruction in declared-reverse order, so
// callers do not need an explicit "engine_started" flag.
class cached_perfetto_session
{
public:
    cached_perfetto_session(output_file_registry& registry, pid_t root_pid,
                            bool single_file_layout, const std::vector<int>& source_pids,
                            post_processor& processor)
    : m_engine{ std::make_unique<core::perfetto_engine>(
          core::build_engine_config_from_settings()) }
    {
        m_engine->init_sdk();
        m_tracks = std::make_unique<rocprofsys::track_registry>();
        if(single_file_layout)
        {
            m_sink =
                std::make_unique<core::trace_sink>(core::single_file_sink{ registry });
        }
        else
        {
            m_sink = std::make_unique<core::trace_sink>(
                core::per_pid_file_sink{ root_pid, registry });
        }
        m_engine->start(core::perfetto_engine::mode::cached_interceptor, *m_sink);
        m_engine->preregister_pids(source_pids);
        processor.set_perfetto_engine(m_engine.get(), m_tracks.get());
        m_started = true;
    }

    cached_perfetto_session(const cached_perfetto_session&)            = delete;
    cached_perfetto_session& operator=(const cached_perfetto_session&) = delete;
    cached_perfetto_session(cached_perfetto_session&&)                 = delete;
    cached_perfetto_session& operator=(cached_perfetto_session&&)      = delete;

    ~cached_perfetto_session()
    {
        if(m_started)
        {
            try
            {
                m_engine->stop();
            } catch(const std::exception& exp)
            {
                LOG_ERROR("Perfetto engine stop/drain raised: {}", exp.what());
            } catch(...)
            {
                // Destructor must not leak — a non-std::exception throw would
                // otherwise reach std::terminate via stack unwinding.
                LOG_ERROR("Perfetto engine stop/drain raised a non-std::exception");
            }
        }
        // Members destruct in reverse declaration order (m_tracks → m_sink →
        // m_engine), which is the inverse of what cached-mode teardown
        // requires: the engine's stop() above has already drained into the
        // sink, so the sink must outlive the engine until that drain
        // returns. Hold the inverted lifetime explicitly so a future
        // reorder of member declarations cannot silently break it.
        m_engine.reset();
        m_sink.reset();
        m_tracks.reset();
    }

private:
    std::unique_ptr<core::perfetto_engine>      m_engine;
    std::unique_ptr<core::trace_sink>           m_sink;
    std::unique_ptr<rocprofsys::track_registry> m_tracks;
    bool                                        m_started{ false };
};
}  // namespace

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

        const auto output_layout      = config::get_perfetto_output_layout();
        const bool single_file_layout = (output_layout == "single_file");

        std::unique_ptr<cached_perfetto_session> session;
        if(enabled_formats.is_perfetto_enabled())
        {
            std::vector<int> source_pids;
            source_pids.reserve(processor_configs.size());
            for(const auto& cfg : processor_configs)
                source_pids.push_back(static_cast<int>(cfg->_pid));

            try
            {
                session = std::make_unique<cached_perfetto_session>(
                    _output_registry, static_cast<pid_t>(root_pid), single_file_layout,
                    source_pids, processor);
            } catch(const std::exception& exp)
            {
                LOG_ERROR("Perfetto engine initialization failed: {}. Skipping "
                          "perfetto output; RocPD output unaffected.",
                          exp.what());
            }
        }

        processor.process(processor_configs, enabled_formats);

        // Triggers engine.stop() + drain via the session destructor before
        // the merge script (below) reads what the sink wrote.
        session.reset();

        // single_file layout already produced exactly one .proto; running the
        // merge script would create a redundant merged.proto in the output
        // directory.
        if(enabled_formats.is_perfetto_enabled() && get_merge_perfetto_files() &&
           !single_file_layout)
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
