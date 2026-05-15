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

#include "cache_manager.hpp"

#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/perfetto_processor.hpp"
#include "core/trace_cache/rocpd_processor.hpp"
#include "core/trace_cache/sample_processor.hpp"

#include "core/agent_manager.hpp"
#include "core/config.hpp"
#include "core/timemory.hpp"

#include "library/runtime.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace rocprofsys
{
namespace trace_cache
{

namespace data
{
struct cache_files_t
{
    std::string buff_storage;
    std::string metadata;
    inline bool empty() const { return buff_storage.empty() || metadata.empty(); }
};

struct format_t
{
    bool        process_parallel;
    bool        enabled;
    const char* name;
};

struct enabled_formats_t
{
    std::vector<format_t> formats = { { true, get_use_rocpd(), "rocpd" },
                                      { false, get_caching_perfetto(), "perfetto" } };

    void print() const
    {
        if(std::none_of(formats.begin(), formats.end(),
                        [](const auto& f) { return f.enabled; }))
            return;

        std::stringstream ss;
        bool              first = true;

        for(const auto& fmt : formats)
        {
            if(fmt.enabled)
            {
                if(!first) ss << ", ";
                ss << fmt.name;
                first = false;
            }
        }

        LOG_INFO(
            "Generating [{}] format(s) with collected data from trace cache. This may "
            "take a while..",
            ss.str().c_str());

        if(has_parallel_formats())
        {
            std::stringstream parallel_ss;
            bool              first_parallel = true;
            for(const auto& fmt : formats)
            {
                if(fmt.enabled && fmt.process_parallel)
                {
                    if(!first_parallel) parallel_ss << ", ";
                    parallel_ss << fmt.name;
                    first_parallel = false;
                }
            }
            LOG_INFO("  - Using parallel processing for: {}", parallel_ss.str());
        }

        if(has_sequential_formats())
        {
            std::stringstream sequential_ss;
            bool              first_sequential = true;
            for(const auto& fmt : formats)
            {
                if(fmt.enabled && !fmt.process_parallel)
                {
                    if(!first_sequential) sequential_ss << ", ";
                    sequential_ss << fmt.name;
                    first_sequential = false;
                }
            }
            LOG_INFO("  - Using sequential processing for: {}",
                     sequential_ss.str().c_str());
        }
    }

    bool has_parallel_formats() const
    {
        return std::any_of(formats.begin(), formats.end(),
                           [](const auto& f) { return f.enabled && f.process_parallel; });
    }

    bool has_sequential_formats() const
    {
        return std::any_of(formats.begin(), formats.end(), [](const auto& f) {
            return f.enabled && !f.process_parallel;
        });
    }

    enabled_formats_t get_parallel_formats() const
    {
        enabled_formats_t parallel_formats;
        parallel_formats.formats.clear();
        for(const auto& fmt : formats)
        {
            parallel_formats.formats.push_back(
                { true, fmt.enabled && fmt.process_parallel, fmt.name });
        }
        return parallel_formats;
    }

    enabled_formats_t get_sequential_formats() const
    {
        enabled_formats_t sequential_formats;
        sequential_formats.formats.clear();
        for(const auto& fmt : formats)
        {
            sequential_formats.formats.push_back(
                { false, fmt.enabled && !fmt.process_parallel, fmt.name });
        }
        return sequential_formats;
    }

    bool is_rocpd_enabled() const
    {
        auto it = std::find_if(formats.begin(), formats.end(), [](const auto& f) {
            return std::strcmp(f.name, "rocpd") == 0;
        });
        return it != formats.end() && it->enabled;
    }

    bool is_perfetto_enabled() const
    {
        auto it = std::find_if(formats.begin(), formats.end(), [](const auto& f) {
            return std::strcmp(f.name, "perfetto") == 0;
        });
        return it != formats.end() && it->enabled;
    }
};

struct processor_config_t
{
    processor_config_t(pid_t pid, pid_t ppid,
                       std::shared_ptr<metadata_registry> metadata_registry_ptr,
                       std::shared_ptr<agent_manager>     agent_manager_ptr)
    : _pid(pid)
    , _ppid(ppid)
    , _metadata_registry(std::move(metadata_registry_ptr))
    , _agent_manager(std::move(agent_manager_ptr))
    {}

    pid_t _pid;
    pid_t _ppid;

    std::shared_ptr<metadata_registry> _metadata_registry;
    std::shared_ptr<agent_manager>     _agent_manager;
};

struct processor_storage_t
{
    std::shared_ptr<rocpd_processor_t>    rocpd_processor{ nullptr };
    std::shared_ptr<perfetto_processor_t> perfetto_processor{ nullptr };
};

using directory_files_t    = std::vector<std::string>;
using mapped_cache_files_t = std::map<pid_t, cache_files_t>;
}  // namespace data

namespace filesystem_utils
{

void
remove_if_exists(const std::string& fname)
{
    if(fname.empty()) return;
    std::ifstream file(fname);
    if(file.is_open())
    {
        file.close();
        auto result = std::remove(fname.c_str());
        if(result == 0)
        {
            LOG_DEBUG("Removed file: {}", fname);
        }
        else if(errno == ENOENT)
        {
            LOG_DEBUG("File does not exist: {}", fname);
        }
        else
        {
            LOG_WARNING("Failed to remove file: {} (errno: {} - {})", fname, errno,
                        std::strerror(errno));
        }
    }
}

data::directory_files_t
list_dir_files(const std::string& _path)
{
    if(_path.empty())
    {
        return {};
    }

    auto dir_deleter = [](DIR* d) {
        if(d) closedir(d);
    };

    std::unique_ptr<DIR, decltype(dir_deleter)> dir(opendir(_path.c_str()), dir_deleter);

    if(!dir)
    {
        throw std::runtime_error(fmt::format("Error opening directory: {}", _path));
    }

    data::directory_files_t result{};
    dirent*                 entry;

    while((entry = readdir(dir.get())) != nullptr)
    {
        if(std::string(entry->d_name) != "." && std::string(entry->d_name) != "..")
        {
            result.emplace_back(entry->d_name);
        }
    }

    return result;
}

data::mapped_cache_files_t
get_cache_files(const pid_t&                   root_pid,
                const data::directory_files_t& _files_from_temp_directory)
{
    if(_files_from_temp_directory.empty())
    {
        return {};
    }

    data::mapped_cache_files_t cache_map{};

    auto parse_and_fill_cache = [&](const std::string& filename) {
        if(filename.empty())
        {
            return;
        }

        const std::regex buff_regex(R"(buffered_storage_(\d+)_(\d+)\.bin)");
        const std::regex meta_regex(R"(metadata_(\d+)_(\d+)\.json)");
        std::smatch      match;

        if(std::regex_match(filename, match, buff_regex))
        {
            int parent_pid = std::stoi(match[1]);
            int pid        = std::stoi(match[2]);
            if(parent_pid == root_pid)
            {
                cache_map[pid].buff_storage = trace_cache::tmp_directory + filename;
            }
        }
        else if(std::regex_match(filename, match, meta_regex))
        {
            int parent_pid = std::stoi(match[1]);
            int pid        = std::stoi(match[2]);
            if(parent_pid == root_pid)
            {
                cache_map[pid].metadata = trace_cache::tmp_directory + filename;
            }
        }
    };

    std::for_each(_files_from_temp_directory.begin(), _files_from_temp_directory.end(),
                  parse_and_fill_cache);
    return cache_map;
}

void
clear_cache_files(const data::mapped_cache_files_t& _cache_files)
{
    LOG_DEBUG("Removing cached temporary files...");
    for(const auto& [_, files] : _cache_files)
    {
        LOG_DEBUG("Removing cached temporary file: {}", files.buff_storage);
        filesystem_utils::remove_if_exists(files.buff_storage);

        LOG_DEBUG("Removing cached temporary file: {}", files.metadata);
        filesystem_utils::remove_if_exists(files.metadata);
    }
}

void
merge_perfetto_files(const std::vector<std::string>& perfetto_files,
                     const std::string&              _filename)
{
    if(perfetto_files.empty())
    {
        LOG_ERROR("Perfetto trace data is empty. File '{}' will not be written...",
                  _filename);
        return;
    }

    std::vector<char> trace_data;
    size_t            total_size = 0;

    // Calculate total size for reservation
    for(const auto& file : perfetto_files)
    {
        std::ifstream ifs(file, std::ios::binary | std::ios::ate);
        if(ifs)
        {
            total_size += ifs.tellg();
        }
    }

    trace_data.reserve(total_size);

    // Read and concatenate all files
    for(const auto& file : perfetto_files)
    {
        std::ifstream ifs(file, std::ios::binary);
        if(!ifs)
        {
            LOG_ERROR("Error opening '{}'...", file);
            continue;
        }

        ifs.seekg(0, std::ios::end);
        size_t file_size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        size_t current_size = trace_data.size();
        trace_data.resize(current_size + file_size);

        ifs.read(trace_data.data() + current_size, file_size);
    }

    if(!trace_data.empty())
    {
        operation::file_output_message<tim::project::rocprofsys> _fom{};
        // Write the trace into a file.
        if(config::get_verbose() >= 0)
            _fom(_filename, std::string{ "perfetto" },
                 " (%.2f KB / %.2f MB / %.2f GB)... ",
                 static_cast<double>(trace_data.size()) / units::KB,
                 static_cast<double>(trace_data.size()) / units::MB,
                 static_cast<double>(trace_data.size()) / units::GB);
        std::ofstream ofs{};
        if(!filepath::open(ofs, _filename, std::ios::out | std::ios::binary))
        {
            _fom.append("Error opening '%s'...", _filename.c_str());
        }
        else
        {
            // Write the trace into a file.
            ofs.write(trace_data.data(), trace_data.size());
            if(config::get_verbose() >= 0) _fom.append("%s", "Done");  // NOLINT
        }
        ofs.close();
    }
    else
    {
        LOG_ERROR("Perfetto trace data is empty. File '{}' will not be written...",
                  _filename);
    }
}

}  // namespace filesystem_utils

namespace processing_utils
{
[[nodiscard]] data::processor_storage_t
configure_processors(const std::shared_ptr<sample_processor_t>&       _type_processing,
                     const std::shared_ptr<data::processor_config_t>& _processor_config,
                     const data::enabled_formats_t&                   _enabled_formats)
{
    data::processor_storage_t processor_storage;
    if(_enabled_formats.is_rocpd_enabled())
    {
        processor_storage.rocpd_processor = std::make_shared<rocpd_processor_t>(
            _processor_config->_metadata_registry, _processor_config->_agent_manager,
            _processor_config->_pid, _processor_config->_ppid);
        _type_processing->add_handler(*processor_storage.rocpd_processor);
    }
    if(_enabled_formats.is_perfetto_enabled())
    {
        processor_storage.perfetto_processor = std::make_shared<perfetto_processor_t>(
            _processor_config->_metadata_registry, _processor_config->_agent_manager,
            _processor_config->_pid, _processor_config->_ppid);
        _type_processing->add_handler(*processor_storage.perfetto_processor);
    }
    return processor_storage;
}

void
process_buffered_storage(
    const std::shared_ptr<data::processor_config_t>& _processor_config,
    const std::string& _storage_filename, const data::enabled_formats_t& _enabled_formats)
{
    LOG_DEBUG("Processing buffered storage: {} for pid={}", _storage_filename,
              _processor_config->_pid);

    auto _processor_coordinator = std::make_shared<sample_processor_t>();
    auto processor_storage =
        configure_processors(_processor_coordinator, _processor_config, _enabled_formats);
    storage_parser_t _parser(_storage_filename);

    _processor_coordinator->prepare_for_processing();
    try
    {
        _parser.load(_processor_coordinator);
        LOG_TRACE("Successfully loaded buffered storage: {}", _storage_filename);
    } catch(const std::runtime_error& exp)
    {
        LOG_WARNING("Error parsing buffered storage {}: {}", _storage_filename,
                    exp.what());
    }
    _processor_coordinator->finalize_processing();

    LOG_DEBUG("Finished processing buffered storage: {}", _storage_filename);
}

std::vector<std::shared_ptr<data::processor_config_t>>
create_processor_configs(const data::mapped_cache_files_t& _cache_files,
                         const pid_t&                      _root_pid)
{
    constexpr size_t ROOT_PROCESS_INCREMENT{ 1 };

    std::vector<std::shared_ptr<data::processor_config_t>> processor_configs;
    processor_configs.reserve(_cache_files.size() + ROOT_PROCESS_INCREMENT);

    for(const auto& [pid, files] : _cache_files)
    {
        if(files.empty())
        {
            continue;
        }

        std::vector<std::shared_ptr<agent>> _agents;
        auto _metadata = std::make_shared<metadata_registry>();
        _metadata->load_from_file(files.metadata, _agents);

        auto _agent_manager = std::make_shared<agent_manager>(_agents);

        processor_configs.push_back(std::make_shared<data::processor_config_t>(
            pid, _root_pid, _metadata, _agent_manager));
    }
    return processor_configs;
}

void
multithreaded_processing(
    const std::vector<std::shared_ptr<data::processor_config_t>>& _processor_configs,
    const data::enabled_formats_t&                                _enabled_formats)
{
    LOG_DEBUG("Starting multithreaded processing with {} configs",
              _processor_configs.size());
    ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    std::vector<std::thread> processing_threads;
    processing_threads.reserve(_processor_configs.size());
    for(const auto& processor_config : _processor_configs)
    {
        LOG_TRACE("Spawning processing thread for pid={}", processor_config->_pid);
        processing_threads.emplace_back(
            process_buffered_storage, processor_config,
            utility::get_buffered_storage_filename(processor_config->_ppid,
                                                   processor_config->_pid),
            _enabled_formats);
    }

    LOG_TRACE("Waiting for {} processing threads to complete", processing_threads.size());
    for(auto& thread : processing_threads)
    {
        thread.join();
    }
    LOG_DEBUG("Multithreaded processing completed");
}

void
sequential_processing(
    const std::vector<std::shared_ptr<data::processor_config_t>>& _processor_configs,
    const data::enabled_formats_t&                                _enabled_formats)
{
    LOG_DEBUG("Starting sequential processing with {} configs",
              _processor_configs.size());
    for(const auto& processor_config : _processor_configs)
    {
        LOG_TRACE("Processing config for pid={}", processor_config->_pid);
        process_buffered_storage(processor_config,
                                 utility::get_buffered_storage_filename(
                                     processor_config->_ppid, processor_config->_pid),
                                 _enabled_formats);
    }
    LOG_DEBUG("Sequential processing completed");
}

void
dispatch_processing(
    const std::vector<std::shared_ptr<data::processor_config_t>>& _processor_configs,
    const data::enabled_formats_t&                                _enabled_formats)
{
    if(_enabled_formats.has_sequential_formats())
    {
        auto sequential_formats = _enabled_formats.get_sequential_formats();
        sequential_processing(_processor_configs, sequential_formats);
    }
    if(_enabled_formats.has_parallel_formats())
    {
        auto parallel_formats = _enabled_formats.get_parallel_formats();
        multithreaded_processing(_processor_configs, parallel_formats);
    }
}

}  // namespace processing_utils

cache_manager&
cache_manager::get_instance()
{
    static cache_manager instance;
    return instance;
}

void
cache_manager::post_process_bulk()
{
    LOG_TRACE("Starting trace cache bulk post-processing");

    if(!is_root_process())
    {
        LOG_DEBUG("Not root process, skipping bulk post-processing");
        return;
    }

    if(m_storage.is_running())
    {
        LOG_WARNING(
            "Post-processing called without previously shutting down cache storage"
            "cache storage. Calling shutdown explicitly..");
        shutdown();
    }

    const auto root_pid = get_root_process_id();
    LOG_DEBUG("Root process ID: {}", root_pid);

    const auto temp_directory_content =
        filesystem_utils::list_dir_files(trace_cache::tmp_directory);
    LOG_TRACE("Found {} files in temp directory", temp_directory_content.size());

    const auto cache_files =
        filesystem_utils::get_cache_files(root_pid, temp_directory_content);
    LOG_DEBUG("Found {} cache file pairs to process", cache_files.size());

    const data::enabled_formats_t enabled_formats;
    enabled_formats.print();

    auto processor_configs =
        processing_utils::create_processor_configs(cache_files, root_pid);

    processor_configs.push_back(std::make_shared<data::processor_config_t>(
        getpid(), root_pid, m_metadata,
        std::make_shared<agent_manager>(get_agent_manager_instance().get_agents())));

    LOG_INFO("Processing {} trace cache configurations", processor_configs.size());
    processing_utils::dispatch_processing(processor_configs, enabled_formats);

    // MPI merge: use merge script to combine traces from all MPI ranks
    // This matches the legacy behavior in perfetto.cpp
#if defined(ROCPROFSYS_USE_MPI) && ROCPROFSYS_USE_MPI > 0
    if(enabled_formats.is_perfetto_enabled() && config::get_perfetto_combined_traces() &&
       dmp::rank() == 0)
    {
        auto _filename      = config::get_perfetto_output_filename();
        auto _output_folder = tim::filepath::dirname(_filename);
        auto _script_path   = std::string{ "rocprof-sys-merge-output.sh" };
        auto _script_dir    = get_env("ROCPROFSYS_SCRIPT_PATH", std::string{}, false);

        if(!_script_dir.empty())
        {
            _script_path = fmt::format("{}/{}", _script_dir, _script_path);
        }

        if(!tim::filepath::exists(_script_path))
        {
            LOG_WARNING("Merge script not found: {}", _script_path);
        }
        else
        {
            auto _command = _script_path + " '" + _output_folder + "'";

            int result = system(_command.c_str());

            if(result != 0)
            {
                LOG_ERROR("Failed to execute merge script: {}", _command);
            }
            else
            {
                LOG_INFO("Successfully merged MPI perfetto traces");
            }
        }
    }
#endif

    filesystem_utils::clear_cache_files(cache_files);

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
