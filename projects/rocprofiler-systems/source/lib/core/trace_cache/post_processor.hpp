// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/output_file_registry.hpp"
#include "core/perfetto/fwd.hpp"
#include "core/progress/tracker.hpp"
#include "core/trace_cache/data_types.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <sys/types.h>
#include <vector>

namespace rocprofsys
{
class track_registry;
}  // namespace rocprofsys

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

    // Wire the cached-mode perfetto engine + its track_registry into the
    // per-pid perfetto_processor_t instances built by process(). Both
    // pointers must outlive every per-pid processing run; cache_manager
    // owns them for the lifetime of post_process_bulk. Calling with
    // nullptrs (the default) restores the pre-engine wiring — useful
    // when perfetto is disabled and the build of perfetto_processor_t
    // is suppressed anyway.
    void set_cached_perfetto_context(core::cached_perfetto_engine& engine,
                                     rocprofsys::track_registry&   tracks) noexcept;

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

    progress::tracker&                                                  m_tracker;
    output_file_registry&                                               m_registry;
    std::optional<std::reference_wrapper<core::cached_perfetto_engine>> m_engine{};
    std::optional<std::reference_wrapper<rocprofsys::track_registry>>   m_tracks{};
};

}  // namespace rocprofsys::trace_cache
