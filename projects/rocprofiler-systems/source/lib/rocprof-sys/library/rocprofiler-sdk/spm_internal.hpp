// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/spm.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#if ROCPROFSYS_USE_SPM
#    include <rocprofiler-sdk/fwd.h>
#endif

namespace rocprofsys::rocprofiler_sdk::spm
{
/// Normalized SPM counter collection settings derived from user configuration.
struct configuration
{
    std::vector<std::string> counter_events;
    std::uint64_t            sample_interval = 0;

    /// Returns true when this configuration requests SPM counter collection.
    [[nodiscard]] bool requested() const noexcept;
};

/// SPM events from ROCPROFSYS_ROCM_SPM_EVENTS.
[[nodiscard]] std::vector<std::string>
get_events();

/// SPM sample interval from ROCPROFSYS_ROCM_SPM_SAMPLE_INTERVAL.
[[nodiscard]] std::uint64_t
get_sample_interval();

/// Validate SPM collection configuration constraints.
///
/// Returns true when SPM is not requested because there is no SPM configuration to
/// reject. If SPM is requested, validates the required sample interval settings and
/// mutual exclusion with ROCPROFSYS_GPU_PERF_COUNTERS. Compatibility with
/// ROCPROFSYS_ROCM_EVENTS is determined by rocprofiler-sdk on the shared context.
[[nodiscard]] bool
is_config_valid(const configuration&            requested_config,
                const std::vector<std::string>& dispatch_counter_events,
                const std::string&              gpu_perf_counter_events);

/// Inject configuration inputs for validation-focused unit tests.
///
/// Returns false for invalid user configuration or an SDK context conflict. Other
/// SDK/hardware/runtime SPM setup failures warn and allow tool initialization to
/// continue without SPM.
[[nodiscard]] bool
configure_runtime(client_data* data, const configuration& requested_config,
                  const std::vector<std::string>& dispatch_counter_events,
                  const std::string&              gpu_perf_counter_events);

namespace detail
{
// Internal parser helpers are exposed here so user-input parsing stays testable
// without compiling the SDK SPM runtime path.
struct requested_counter
{
    std::string                  name;
    std::optional<std::uint64_t> device_id = std::nullopt;
};

using requested_counter_vec_t = std::vector<requested_counter>;

enum class RuntimeConfigurationResult
{
    Configured,
    Unavailable,
    FatalError,
};

#if ROCPROFSYS_USE_SPM
/// Classify an SDK setup status for the SPM runtime fallback policy.
[[nodiscard]] RuntimeConfigurationResult
classify_runtime_configuration_status(rocprofiler_status_t status) noexcept;
#endif

[[nodiscard]] std::optional<std::uint64_t>
parse_device_id(std::string_view value);

[[nodiscard]] std::string
parse_counter_name(std::string_view event);

[[nodiscard]] requested_counter_vec_t
parse_requested_counters(const configuration& requested_config);

[[nodiscard]] requested_counter_vec_t
requested_counters_for_device(const requested_counter_vec_t& all_requested,
                              std::uint64_t                  device_id);

[[nodiscard]] std::unordered_set<std::string>
requested_counter_names(const requested_counter_vec_t& requested);
}  // namespace detail
}  // namespace rocprofsys::rocprofiler_sdk::spm
