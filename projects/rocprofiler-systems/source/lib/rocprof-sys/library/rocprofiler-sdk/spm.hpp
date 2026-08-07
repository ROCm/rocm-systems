// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
struct client_data;

namespace spm
{
/// Plain value describing an SPM counter collection request.
struct request
{
    std::vector<std::string> events          = {};
    std::uint64_t            sample_interval = 0;

    /// Returns true when the user requested SPM counter collection.
    [[nodiscard]] bool requested() const noexcept;
};

namespace detail
{
// Internal parser helpers are exposed here so user-input parsing stays testable
// without compiling the SDK SPM runtime path.
struct requested_counter
{
    std::string                  name      = {};
    std::optional<std::uint64_t> device_id = std::nullopt;
};

using requested_counter_vec_t = std::vector<requested_counter>;

[[nodiscard]] std::optional<std::uint64_t>
parse_device_id(std::string_view value);

[[nodiscard]] std::string
parse_counter_name(std::string_view event);

[[nodiscard]] requested_counter_vec_t
parse_requested_counters(const request& req);

[[nodiscard]] requested_counter_vec_t
requested_counters_for_device(const requested_counter_vec_t& all_requested,
                              std::uint64_t                  device_id);

[[nodiscard]] std::unordered_set<std::string>
requested_counter_names(const requested_counter_vec_t& requested);
}  // namespace detail

/// Validate SPM collection request constraints.
///
/// Returns true when SPM is not requested because there is no SPM configuration to
/// reject. If SPM is requested, validates the required sample interval settings and
/// mutual exclusion with ROCPROFSYS_ROCM_EVENTS and ROCPROFSYS_GPU_PERF_COUNTERS.
[[nodiscard]]
bool
is_config_valid(const request&                  req,
                const std::vector<std::string>& dispatch_counter_events,
                const std::string&              gpu_perf_counter_events);

/// Configure the SDK SPM runtime service on the dedicated Systems SPM context.
[[nodiscard]] bool
configure_runtime(client_data* data, const request& req);

/// Log accumulated SPM data-loss reports, if any, and reset the counter.
void
log_data_loss(client_data* data);
}  // namespace spm
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
