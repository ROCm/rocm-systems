// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <memory>
#include <string>
#include <sys/types.h>
#include <vector>

namespace rocprofsys
{
class agent_manager;
}

namespace rocprofsys::trace_cache
{
class metadata_registry;
class rocpd_processor_t;
class perfetto_processor_t;
}  // namespace rocprofsys::trace_cache

namespace rocprofsys::trace_cache::data
{

struct cache_files_t
{
    std::string buff_storage;
    std::string metadata;

    [[nodiscard]] bool empty() const noexcept
    {
        return buff_storage.empty() || metadata.empty();
    }
};

enum class format_kind
{
    rocpd,
    perfetto
};

struct format_t
{
    format_kind kind;
    bool        process_parallel;
    bool        enabled;
    const char* name;
};

struct enabled_formats_t
{
    std::vector<format_t> formats;

    enabled_formats_t();
    explicit enabled_formats_t(std::vector<format_t> _formats) noexcept;

    void                            print() const;
    [[nodiscard]] bool              has_parallel_formats() const;
    [[nodiscard]] bool              has_sequential_formats() const;
    [[nodiscard]] enabled_formats_t get_parallel_formats() const;
    [[nodiscard]] enabled_formats_t get_sequential_formats() const;
    [[nodiscard]] bool              is_rocpd_enabled() const;
    [[nodiscard]] bool              is_perfetto_enabled() const;
    [[nodiscard]] std::string       names() const;
};

struct processor_config_t
{
    processor_config_t(pid_t pid, pid_t ppid,
                       std::shared_ptr<metadata_registry> metadata_registry_ptr,
                       std::shared_ptr<agent_manager>     agent_manager_ptr);

    pid_t _pid;
    pid_t _ppid;

    std::shared_ptr<metadata_registry> _metadata_registry;
    std::shared_ptr<agent_manager>     _agent_manager;
};

struct processor_storage_t
{
    std::shared_ptr<rocpd_processor_t>    rocpd_processor;
    std::shared_ptr<perfetto_processor_t> perfetto_processor;
};

using directory_files_t    = std::vector<std::string>;
using mapped_cache_files_t = std::map<pid_t, cache_files_t>;

}  // namespace rocprofsys::trace_cache::data
