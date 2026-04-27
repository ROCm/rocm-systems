// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output_file_registry.hpp"
#include "core/progress/tracker.hpp"
#include "core/trace_cache/data_types.hpp"

#include <memory>
#include <sys/types.h>
#include <vector>

namespace rocprofsys::trace_cache
{

class post_processor
{
public:
    post_processor(progress::tracker& tracker, output_file_registry& registry) noexcept;

    post_processor(const post_processor&)            = delete;
    post_processor& operator=(const post_processor&) = delete;
    post_processor(post_processor&&)                 = delete;
    post_processor& operator=(post_processor&&)      = delete;

    void process(const std::vector<std::shared_ptr<data::processor_config_t>>& configs,
                 const data::enabled_formats_t&                                formats);

    [[nodiscard]] static std::vector<std::shared_ptr<data::processor_config_t>>
    make_configs(const data::mapped_cache_files_t& cache_files, const pid_t& root_pid);

private:
    void run_sequential(
        const std::vector<std::shared_ptr<data::processor_config_t>>& configs,
        const data::enabled_formats_t&                                formats);

    void run_multithreaded(
        const std::vector<std::shared_ptr<data::processor_config_t>>& configs,
        const data::enabled_formats_t&                                formats);

    progress::tracker&    m_tracker;
    output_file_registry& m_registry;
};

}  // namespace rocprofsys::trace_cache
