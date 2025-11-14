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
#include "core/trace_cache/rocpd_processor.hpp"
#include "core/trace_cache/sample_processor.hpp"

#include "core/agent_manager.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"

#include "library/runtime.hpp"

#include <algorithm>
#include <memory>
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

struct enabled_formats_t
{
    bool rocpd = get_use_rocpd();

    void print() const
    {
        constexpr std::pair<const char*, bool enabled_formats_t::*> formats[] = {
            { "rocpd", &enabled_formats_t::rocpd },
        };

        bool any_enabled = false;
        for(const auto& fmt : formats)
            any_enabled |= this->*(fmt.second);

        if(!any_enabled) return;

        bool              first = true;
        std::stringstream ss;

        for(const auto& fmt : formats)
        {
            if(this->*(fmt.second))
            {
                if(!first && sizeof(formats) > 1) ss << ", ";
                ss << fmt.first;
                first = false;
            }
        }
        ROCPROFSYS_PRINT(
            "Generating [%s] format(s) with collected data from trace cache. This may "
            "take a while..\n",
            ss.str().c_str());
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
    std::shared_ptr<rocpd_processor_t> rocpd_processor{ nullptr };
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
            ROCPROFSYS_DEBUG("Removed file: %s\n", fname.c_str());
        }
        else if(errno == ENOENT)
        {
            ROCPROFSYS_DEBUG("File does not exist: %s\n", fname.c_str());
        }
        else
        {
            ROCPROFSYS_WARNING(0, "Failed to remove file: %s (errno: %d - %s)\n",
                               fname.c_str(), errno, std::strerror(errno));
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
        ROCPROFSYS_THROW("Error opening directory: %s", _path.c_str());
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
    ROCPROFSYS_PRINT("Removing cached temporary files...\n");
    for(const auto& [_, files] : _cache_files)
    {
        ROCPROFSYS_DEBUG("Removing cached temporary file: %s\n",
                         files.buff_storage.c_str());
        filesystem_utils::remove_if_exists(files.buff_storage);

        ROCPROFSYS_DEBUG("Removing cached temporary file: %s\n", files.metadata.c_str());
        filesystem_utils::remove_if_exists(files.metadata);
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
    if(_enabled_formats.rocpd)
    {
        processor_storage.rocpd_processor = std::make_shared<rocpd_processor_t>(
            _processor_config->_metadata_registry, _processor_config->_agent_manager,
            _processor_config->_pid, _processor_config->_ppid);
        _type_processing->add_handler(*processor_storage.rocpd_processor);
    }
    return processor_storage;
}

void
process_buffered_storage(
    const std::shared_ptr<data::processor_config_t>& _processor_config,
    const std::string& _storage_filename, const data::enabled_formats_t& _enabled_formats)
{
    auto _processor_coordinator = std::make_shared<sample_processor_t>();
    auto processor_storage =
        configure_processors(_processor_coordinator, _processor_config, _enabled_formats);
    storage_parser_t _parser(_storage_filename);

    _processor_coordinator->prepare_for_processing();
    _parser.load(_processor_coordinator);
    _processor_coordinator->finalize_processing();
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
    ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    std::vector<std::thread> processing_threads;
    processing_threads.reserve(_processor_configs.size());
    for(const auto& processor_config : _processor_configs)
    {
        processing_threads.emplace_back(
            process_buffered_storage, processor_config,
            utility::get_buffered_storage_filename(processor_config->_ppid,
                                                   processor_config->_pid),
            _enabled_formats);
    }

    for(auto& thread : processing_threads)
    {
        thread.join();
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
    if(!is_root_process())
    {
        return;
    }

    if(m_storage.is_running())
    {
        ROCPROFSYS_WARNING(2, "Postprocessing called without previously shutting down "
                              "cache storage. Calling shutdown explicitly..\n");
        shutdown();
    }

    const auto root_pid = get_root_process_id();
    const auto temp_directory_content =
        filesystem_utils::list_dir_files(trace_cache::tmp_directory);

    const auto cache_files =
        filesystem_utils::get_cache_files(root_pid, temp_directory_content);
    const data::enabled_formats_t enabled_formats;
    enabled_formats.print();

    auto processor_configs =
        processing_utils::create_processor_configs(cache_files, root_pid);

    processor_configs.push_back(std::make_shared<data::processor_config_t>(
        getpid(), root_pid, m_metadata,
        std::make_shared<agent_manager>(get_agent_manager_instance().get_agents())));

    processing_utils::multithreaded_processing(processor_configs, enabled_formats);

    filesystem_utils::clear_cache_files(cache_files);
}

void
cache_manager::shutdown()
{
    m_storage.shutdown();
}

}  // namespace trace_cache
}  // namespace rocprofsys
