// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/discovery.hpp"

#include "core/config.hpp"
#include "core/timemory.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "logger/debug.hpp"

#include <dirent.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <variant>

namespace rocprofsys::trace_cache::discovery
{
data::directory_files_t
list_dir_files(const std::string& path)
{
    if(path.empty()) return {};

    auto dir_deleter = [](DIR* d) {
        if(d) closedir(d);
    };

    std::unique_ptr<DIR, decltype(dir_deleter)> dir(opendir(path.c_str()), dir_deleter);

    if(!dir) throw std::runtime_error(fmt::format("Error opening directory: {}", path));

    data::directory_files_t result{};
    dirent*                 entry;

    while((entry = readdir(dir.get())) != nullptr)
    {
        if(std::string(entry->d_name) != "." && std::string(entry->d_name) != "..")
            result.emplace_back(entry->d_name);
    }

    return result;
}

data::mapped_cache_files_t
find_cache_files(const pid_t& root_pid, const data::directory_files_t& dir_contents)
{
    if(dir_contents.empty()) return {};

    data::mapped_cache_files_t cache_map{};

    auto parse_and_fill_cache = [&](const std::string& filename) {
        if(filename.empty()) return;

        const std::regex buff_regex(R"(buffered_storage_(\d+)_(\d+)\.bin)");
        const std::regex meta_regex(R"(metadata_(\d+)_(\d+)\.json)");
        std::smatch      match;

        // Wrap stoi: regex's (\d+) can match digit runs that overflow int.
        // A pathological file in the temp dir must not abort discovery.
        try
        {
            if(std::regex_match(filename, match, buff_regex))
            {
                int parent_pid = std::stoi(match[1]);
                int pid        = std::stoi(match[2]);
                if(parent_pid == root_pid)
                    cache_map[pid].buff_storage = trace_cache::tmp_directory + filename;
            }
            else if(std::regex_match(filename, match, meta_regex))
            {
                int parent_pid = std::stoi(match[1]);
                int pid        = std::stoi(match[2]);
                if(parent_pid == root_pid)
                    cache_map[pid].metadata = trace_cache::tmp_directory + filename;
            }
        } catch(const std::exception& e)
        {
            LOG_WARNING("Skipping cache file with unparsable PID '{}': {}", filename,
                        e.what());
        }
    };

    std::for_each(dir_contents.begin(), dir_contents.end(), parse_and_fill_cache);
    return cache_map;
}

void
clear(const data::mapped_cache_files_t& cache_files)
{
    LOG_DEBUG("Removing cached temporary files...");
    for(const auto& [_, files] : cache_files)
    {
        for(const auto* fname : { &files.buff_storage, &files.metadata })
        {
            if(fname->empty()) continue;

            if(std::remove(fname->c_str()) == 0)
                LOG_DEBUG("Removed file: {}", *fname);
            else if(errno != ENOENT)
                LOG_WARNING("Failed to remove file: {}: {}", *fname,
                            std::strerror(errno));
        }
    }
}

void
merge_perfetto_files()
{
    // dmp::rank() returns 0 if MPI is not initialized/finalized. During shutdown
    // MPI may already be finalized, so we read the rank captured at init time
    // via settings::default_process_suffix().
    auto         suffix_variant  = settings::default_process_suffix();
    std::int32_t cached_mpi_rank = 0;

    if(std::holds_alternative<int>(suffix_variant))
    {
        cached_mpi_rank = std::get<int>(suffix_variant);
    }
    else if(std::holds_alternative<std::string>(suffix_variant))
    {
        try
        {
            cached_mpi_rank = std::stoi(std::get<std::string>(suffix_variant));
        } catch(...)
        {
            cached_mpi_rank = 0;
        }
    }

    LOG_DEBUG("Merging perfetto files: rank={} (from settings::default_process_suffix)",
              cached_mpi_rank);

    if(cached_mpi_rank != 0) return;

    auto _filename      = config::get_perfetto_output_filename();
    auto _output_folder = tim::filepath::dirname(_filename);
    auto _script_path   = std::string{ "rocprof-sys-merge-output.sh" };
    auto _script_dir    = get_env("ROCPROFSYS_SCRIPT_PATH", std::string{});

    if(!_script_dir.empty())
        _script_path = fmt::format("{}/{}", _script_dir, _script_path);

    if(!tim::filepath::exists(_script_path))
    {
        LOG_WARNING("Merge script not found: {}", _script_path);
        return;
    }

    auto _command = _script_path + " '" + _output_folder + "'";
    int  result   = system(_command.c_str());

    if(result != 0)
        LOG_ERROR("Failed to execute merge script: {}", _command);
    else
        LOG_INFO("Successfully executed: {}", _command);
}

}  // namespace rocprofsys::trace_cache::discovery
