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
#include <iterator>
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
        std::cerr << "Error opening directory: " << path << std::endl;
        return {};
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
}  // namespace

cache_manager&
cache_manager::get_instance()
{
    static cache_manager instance;
    return instance;
}

void
cache_manager::post_process()
{
    if(m_storage.is_running())
    {
        ROCPROFSYS_WARNING(2, "Postprocessing called without previously shutting down "
                              "cache storage. Calling shutdown explicitly..\n");
        shutdown();
    }

    if(get_use_rocpd())
    {
        ROCPROFSYS_PRINT(
            "Generating rocpd with collected data. This may take a while..\n");
    }
    // post_process_metadata();
}

void
cache_manager::post_process_bulk()
{
    struct cache_files
    {
        std::string buff_storage;
        std::string metadata;
    };

    if(is_root_process())
    {
        auto root_pid  = get_root_process_id();
        auto tmp_files = list_dir_files("/tmp/");

        std::map<int, cache_files> cache_map{};

        auto parse_and_fill_cache = [&](const std::string& filename) {
            if(filename.find("buffered_storage_") != std::string::npos)
            {
                // Extract parent PID and PID from buffered_storage_<parent_pid>_<pid>.bin
                auto first_underscore = filename.find("buffered_storage_");
                if(first_underscore != std::string::npos)
                {
                    auto start_pos =
                        first_underscore + std::string("buffered_storage_").length();
                    auto second_underscore = filename.find('_', start_pos);
                    if(second_underscore != std::string::npos)
                    {
                        auto dot_pos = filename.find('.', second_underscore);
                        if(dot_pos != std::string::npos)
                        {
                            int parent_pid = std::stoi(filename.substr(
                                start_pos, second_underscore - start_pos));
                            int pid        = std::stoi(filename.substr(
                                second_underscore + 1, dot_pos - second_underscore - 1));

                            if(parent_pid == root_pid)
                            {
                                cache_map[pid].buff_storage =
                                    std::string("/tmp/" + filename);
                            }
                        }
                    }
                }
            }
            else if(filename.find("metadata_") != std::string::npos)
            {
                // Extract parent PID and PID from metadata_<parent_pid>_<pid>.json
                auto first_underscore = filename.find("metadata_");
                if(first_underscore != std::string::npos)
                {
                    auto start_pos = first_underscore + std::string("metadata_").length();
                    auto second_underscore = filename.find('_', start_pos);
                    if(second_underscore != std::string::npos)
                    {
                        auto dot_pos = filename.find('.', second_underscore);
                        if(dot_pos != std::string::npos)
                        {
                            int parent_pid = std::stoi(filename.substr(
                                start_pos, second_underscore - start_pos));
                            int pid        = std::stoi(filename.substr(
                                second_underscore + 1, dot_pos - second_underscore - 1));

                            if(parent_pid == root_pid)
                            {
                                cache_map[pid].metadata = std::string("/tmp/" + filename);
                            }
                        }
                    }
                }
            }
        };

        std::for_each(tmp_files.begin(), tmp_files.end(), parse_and_fill_cache);

        std::vector<std::thread> rocpd_threads;
        ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

        rocpd_threads.emplace_back([this]() {
            rocpd_post_processing _post_processing(
                m_metadata, agent_manager::get_instance(), getpid());
            storage_parser _parser(getpid(), filename);
            _post_processing.register_parser_callback(_parser);
            _post_processing.post_process_metadata();
            _parser.consume_storage();
            _post_processing.get_data_processor()->flush();
        });

        for(const auto& [pid, files] : cache_map)
        {
            if(!files.buff_storage.empty() && !files.metadata.empty())
            {
                rocpd_threads.emplace_back([pid = pid, files = files]() {
                    std::cout << "\nCreating database for [" << pid
                              << "]: " << files.buff_storage << " " << files.metadata
                              << std::endl;
                    metadata_registry metadata;
                    auto              res = metadata.load_from_file(files.metadata);
                    if(!res)
                    {
                        std::cout << "LOAD FROM FILE FOR METADATA FAILED" << std::endl;
                        return;
                    }
                    agent_manager         agent_mngr(metadata.get_agents());
                    rocpd_post_processing _post_processing(metadata, agent_mngr, pid);
                    storage_parser        _parser(getpid(), files.buff_storage);
                    _post_processing.register_parser_callback(_parser);
                    _post_processing.post_process_metadata();
                    _parser.consume_storage();
                    _post_processing.get_data_processor()->flush();
                });
            }
        }

        for(auto& thread : rocpd_threads)
            thread.join();
    }
}

void
cache_manager::shutdown()
{
    m_storage.shutdown();
}

}  // namespace trace_cache
}  // namespace rocprofsys
