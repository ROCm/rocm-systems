// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/hipfile/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rocprofsys::pmc::collectors::hipfile::testing
{

/**
 * @brief Programmable stand-in for the hipFile backend session.
 *
 * Satisfies hipfile_backend_contract with no hipFile dependency, so the collector's
 * behaviour - fastpath vs fallback attribution, cumulative counters, wall-clock
 * bandwidth, the unavailable path - is verifiable on any machine, with or without
 * GPU-direct storage hardware.
 */
struct mock_backend
{
    stats_snapshot snapshot{};
    bool           available  = true;
    std::size_t    call_count = 0;

    /// Timestamps passed to get_stats, in order. Lets a test assert the backend saw
    /// exactly one query per interval.
    std::vector<std::uint64_t> queried_timestamps{};

    [[nodiscard]] const stats_snapshot& get_stats(std::uint64_t timestamp)
    {
        ++call_count;
        queried_timestamps.push_back(timestamp);
        return snapshot;
    }

    [[nodiscard]] bool is_available() const noexcept { return available; }

    void shutdown() noexcept {}

    /// @brief Convenience accessor for populating one GPU's slot.
    [[nodiscard]] gpu_stats& gpu(std::size_t ordinal)
    {
        return snapshot.per_gpu[ordinal];
    }
};

/**
 * @brief Device provider stand-in handing out devices over a shared mock backend.
 */
struct mock_provider
{
    using backend_t = mock_backend;

    std::shared_ptr<mock_backend> backend         = std::make_shared<mock_backend>();
    bool                          shutdown_called = false;

    template <typename Device>
    [[nodiscard]] std::vector<std::shared_ptr<Device>> get_devices(std::size_t gpu_count)
    {
        std::vector<std::shared_ptr<Device>> devices;
        devices.reserve(gpu_count);
        for(std::size_t ordinal = 0; ordinal < gpu_count; ++ordinal)
            devices.push_back(std::make_shared<Device>(backend, ordinal));
        return devices;
    }

    void init() {}
    void shutdown() { shutdown_called = true; }
};

}  // namespace rocprofsys::pmc::collectors::hipfile::testing
