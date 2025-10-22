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
#include "agent_manager.hpp"
#include "core/config.hpp"
#include "core/trace_cache/storage_parser.hpp"
#include "debug.hpp"
#include "library/runtime.hpp"
#include "trace_cache/cache_utility.hpp"
#include "trace_cache/metadata_registry.hpp"
#include "trace_cache/rocpd_post_processing.hpp"
#include <algorithm>
#include <memory>
#include <vector>

namespace rocprofsys
{
namespace trace_cache
{
namespace
{
std::vector<std::string>
list_dir_files(const std::string& path)
{
    DIR* dir = opendir(path.c_str());
    if(dir == nullptr)
    {
        ROCPROFSYS_THROW("Error opening directory: %s", path.c_str());
    }

    std::vector<std::string> result{};
    dirent*                  entry;

    while((entry = readdir(dir)) != nullptr)
    {
        if(std::string(entry->d_name) != "." && std::string(entry->d_name) != "..")
        {
            result.emplace_back(entry->d_name);
        }
    }

    closedir(dir);
    return result;
}

struct cache_files
{
    std::string buff_storage;
    std::string metadata;
};

std::map<pid_t, cache_files>
get_cache_files()
{
    const auto root_pid  = get_root_process_id();
    const auto tmp_files = list_dir_files("/tmp/");

    std::map<int, cache_files> cache_map{};

    auto parse_and_fill_cache = [&](const std::string& filename) {
        const std::regex buff_regex(R"(buffered_storage_(\d+)_(\d+)\.bin)");
        const std::regex meta_regex(R"(metadata_(\d+)_(\d+)\.json)");
        std::smatch      match;

        if(std::regex_match(filename, match, buff_regex))
        {
            int parent_pid = std::stoi(match[1]);
            int pid        = std::stoi(match[2]);
            if(parent_pid == root_pid)
            {
                cache_map[pid].buff_storage = "/tmp/" + filename;
            }
        }
        else if(std::regex_match(filename, match, meta_regex))
        {
            int parent_pid = std::stoi(match[1]);
            int pid        = std::stoi(match[2]);
            if(parent_pid == root_pid)
            {
                cache_map[pid].metadata = "/tmp/" + filename;
            }
        }
    };

    std::for_each(tmp_files.begin(), tmp_files.end(), parse_and_fill_cache);
    return cache_map;
}
}  // namespace

cache_manager&
cache_manager::get_instance()
{
    static cache_manager instance;
    return instance;
}

void
cache_manager::post_process_bulk()
{
    if(is_root_process())
    {
        if(m_storage.is_running())
        {
            ROCPROFSYS_WARNING(2,
                               "Postprocessing called without previously shutting down "
                               "cache storage. Calling shutdown explicitly..\n");
            shutdown();
        }

        if(get_use_rocpd())
        {
            ROCPROFSYS_PRINT(
                "Generating rocpd with collected data. This may take a while..\n");

            auto _cache_files = get_cache_files();

            std::vector<std::thread> rocpd_threads;
            ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

            rocpd_threads.emplace_back([this]() {
                auto                  pid  = getpid();
                auto                  ppid = get_root_process_id();
                rocpd_post_processing _post_processing(
                    m_metadata, get_agent_manager_instance(), pid, ppid);
                storage_parser _parser(
                    get_buffered_storage_filename(get_root_process_id(), getpid()));
                _post_processing.register_parser_callback(_parser);
                _post_processing.post_process_metadata();
                _parser.consume_storage();
            });

            for(const auto& [pid, files] : _cache_files)
            {
                if(!files.buff_storage.empty() && !files.metadata.empty())
                {
                    rocpd_threads.emplace_back([pid = pid, files = files]() {
                        ROCPROFSYS_DEBUG(
                            "Creating database for [%d] from buffered storage "
                            "file: %s and from metadata file: %s\n",
                            pid, files.buff_storage.c_str(), files.metadata.c_str());

                        std::vector<std::shared_ptr<agent>> _agents;
                        metadata_registry                   _metadata;

                        auto res = _metadata.load_from_file(files.metadata, _agents);
                        if(!res)
                        {
                            ROCPROFSYS_WARNING(0,
                                               "Load from file for metadata failed: %s\n",
                                               files.metadata.c_str());
                            return;
                        }

                        agent_manager         _agent_manager{ _agents };
                        auto                  ppid = get_root_process_id();
                        rocpd_post_processing _post_processing(_metadata, _agent_manager,
                                                               pid, ppid);
                        storage_parser        _parser(files.buff_storage);
                        _post_processing.register_parser_callback(_parser);
                        _post_processing.post_process_metadata();
                        _parser.consume_storage();
                        std::remove(files.metadata.c_str());  // Remove metadata file
                    });
                }
            }

            for(auto& thread : rocpd_threads)
            {
                thread.join();
            }
        }
    }
}

void
cache_manager::shutdown()
{
    m_storage.shutdown();
}

}  // namespace trace_cache
}  // namespace rocprofsys
