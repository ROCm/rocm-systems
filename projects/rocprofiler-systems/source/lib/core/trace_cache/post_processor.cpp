// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/post_processor.hpp"

#include "core/agent_manager.hpp"
#include "core/config.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/perfetto_processor.hpp"
#include "core/trace_cache/rocpd_processor.hpp"
#include "core/trace_cache/sample_processor.hpp"
#include "core/trace_cache/storage_parser_alias.hpp"
#include "core/trace_cache/unified_memory_processor.hpp"
#include "library/runtime.hpp"
#include "logger/debug.hpp"

#include <sys/stat.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rocprofsys::trace_cache
{
namespace
{

std::uint64_t
file_size_or_zero(const std::string& path) noexcept
{
    struct stat st
    {};
    if(::stat(path.c_str(), &st) != 0) return 0;
    return static_cast<std::uint64_t>(st.st_size);
}

std::uint64_t
sum_storage_bytes(const std::vector<std::shared_ptr<data::processor_config_t>>& configs)
{
    std::uint64_t total = 0;
    for(const auto& cfg : configs)
        total += file_size_or_zero(
            utility::get_buffered_storage_filename(cfg->_ppid, cfg->_pid));
    return total;
}

[[nodiscard]] data::processor_storage_t
configure_processors(const std::shared_ptr<sample_processor_t>&       _coordinator,
                     const std::shared_ptr<data::processor_config_t>& _config,
                     const data::enabled_formats_t&                   _formats,
                     output_file_registry&                            _registry)
{
    data::processor_storage_t storage;
    if(_formats.is_rocpd_enabled())
    {
        storage.rocpd_processor = std::make_shared<rocpd_processor_t>(
            _config->_metadata_registry, _config->_agent_manager, _config->_pid,
            _config->_ppid, _registry);
        _coordinator->add_handler(*storage.rocpd_processor);
    }
    if(_formats.is_perfetto_enabled())
    {
        storage.perfetto_processor = std::make_shared<perfetto_processor_t>(
            _config->_metadata_registry, _config->_agent_manager, _config->_pid,
            _config->_ppid, _registry);
        _coordinator->add_handler(*storage.perfetto_processor);
    }

    if(_formats.is_unified_memory_enabled())
    {
        storage.unified_memory_processor = std::make_shared<unified_memory_processor_t>(
            _config->_agent_manager, _config->_pid, output_file_sink_view{ _registry });
        _coordinator->add_handler(*storage.unified_memory_processor);
        LOG_DEBUG("Unified memory processor enabled for PID {}", _config->_pid);
    }

    return storage;
}

void
process_buffered_storage(const std::shared_ptr<data::processor_config_t>& _config,
                         const std::string&             _storage_filename,
                         const data::enabled_formats_t& _formats,
                         output_file_registry&          _registry,
                         progress::progress_callback    _progress_cb)
{
    LOG_DEBUG("Processing buffered storage: {} for pid={}", _storage_filename,
              _config->_pid);

    auto _coordinator = std::make_shared<sample_processor_t>();
    // RAII lifetime guard: configure_processors registers raw references to the
    // returned processors as handlers on _coordinator. Holding _storage in scope
    // keeps those processors alive until the parse + finalize is done.
    [[maybe_unused]] auto _storage =
        configure_processors(_coordinator, _config, _formats, _registry);
    storage_parser_t _parser(_storage_filename);

    _coordinator->prepare_for_processing();
    try
    {
        _parser.load(_coordinator, std::move(_progress_cb));
        LOG_TRACE("Successfully loaded buffered storage: {}", _storage_filename);
    } catch(const std::runtime_error& exp)
    {
        LOG_WARNING("Error parsing buffered storage {}: {}", _storage_filename,
                    exp.what());
    }
    _coordinator->finalize_processing();

    LOG_DEBUG("Finished processing buffered storage: {}", _storage_filename);
}
}  // namespace

post_processor::post_processor(progress::tracker&    tracker,
                               output_file_registry& registry) noexcept
: m_tracker(tracker)
, m_registry(registry)
{}

void
post_processor::process(
    const std::vector<std::shared_ptr<data::processor_config_t>>& configs,
    const data::enabled_formats_t&                                formats)
{
    if(formats.has_sequential_formats())
        run_sequential(configs, formats.get_sequential_formats());
    if(formats.has_parallel_formats())
        run_multithreaded(configs, formats.get_parallel_formats());
}

void
post_processor::run_sequential(
    const std::vector<std::shared_ptr<data::processor_config_t>>& configs,
    const data::enabled_formats_t&                                formats)
{
    LOG_DEBUG("Starting sequential processing with {} configs", configs.size());
    for(const auto& cfg : configs)
    {
        LOG_TRACE("Processing config for pid={}", cfg->_pid);
        const auto _filename =
            utility::get_buffered_storage_filename(cfg->_ppid, cfg->_pid);
        auto _progress_cb = m_tracker.begin(
            fmt::format("Generating trace-cache output for process [{}]", cfg->_pid),
            file_size_or_zero(_filename));
        process_buffered_storage(cfg, _filename, formats, m_registry, _progress_cb);
    }
    LOG_DEBUG("Sequential processing completed");
}

void
post_processor::run_multithreaded(
    const std::vector<std::shared_ptr<data::processor_config_t>>& configs,
    const data::enabled_formats_t&                                formats)
{
    LOG_DEBUG("Starting multithreaded processing with {} configs", configs.size());
    ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    auto _progress_cb = m_tracker.begin("Generating ROCpd output database file",
                                        sum_storage_bytes(configs));

    std::vector<std::thread> processing_threads;
    processing_threads.reserve(configs.size());
    for(const auto& cfg : configs)
    {
        LOG_TRACE("Spawning processing thread for pid={}", cfg->_pid);
        // Capture the callback by value: it holds a shared_ptr to the bar,
        // and bar::on_advance is thread-safe, so concurrent calls are fine.
        processing_threads.emplace_back([this, cfg, &formats, _progress_cb] {
            const auto _filename =
                utility::get_buffered_storage_filename(cfg->_ppid, cfg->_pid);
            process_buffered_storage(cfg, _filename, formats, m_registry, _progress_cb);
        });
    }

    LOG_TRACE("Waiting for {} processing threads to complete", processing_threads.size());
    for(auto& thread : processing_threads)
        thread.join();
    LOG_DEBUG("Multithreaded processing completed");
}

std::vector<std::shared_ptr<data::processor_config_t>>
post_processor::make_configs(const data::mapped_cache_files_t& cache_files,
                             const pid_t&                      root_pid)
{
    constexpr size_t ROOT_PROCESS_INCREMENT = 1;

    std::vector<std::shared_ptr<data::processor_config_t>> configs;
    configs.reserve(cache_files.size() + ROOT_PROCESS_INCREMENT);

    for(const auto& [pid, files] : cache_files)
    {
        if(files.empty()) continue;

        std::vector<std::shared_ptr<agent>> _agents;
        auto _metadata = std::make_shared<metadata_registry>();
        _metadata->load_from_file(files.metadata, _agents);

        auto _agent_manager = std::make_shared<agent_manager>(_agents);

        configs.push_back(std::make_shared<data::processor_config_t>(
            pid, root_pid, _metadata, _agent_manager));
    }
    return configs;
}

}  // namespace rocprofsys::trace_cache
