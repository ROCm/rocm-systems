// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/gpu_perf_counter/device.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <memory>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

using ::rocprofsys::pmc::device_selection_mode;
using ::rocprofsys::pmc::device_type;

template <typename BackendProvider>
struct gpu_perf_counter_traits
{
    using enabled_metrics_t = pmc::collectors::gpu_perf_counter::enabled_metrics;
    using device_t          = device<typename BackendProvider::backend_t>;
    using device_ptr_t      = std::shared_ptr<device_t>;

    struct device_entry
    {
        device_ptr_t      device;
        enabled_metrics_t supported_metrics;
    };

    // Device filtering is the caller's responsibility: the agent list is already
    // filtered upstream, so the Settings parameter is unused here; it keeps the
    // signature of the other traits.
    template <typename Settings, typename Provider>
    [[nodiscard]] static std::vector<device_entry> enumerate_devices(
        std::shared_ptr<Provider> provider)
    {
        std::vector<device_entry> entries;

        auto devices = provider->template get_devices<device_t>(device_type::gpu);

        for(auto& dev : devices)
        {
            if(dev->is_supported())
            {
                entries.push_back(
                    device_entry{ std::move(dev), enabled_metrics_t{ {} } });
            }
        }

        return entries;
    }
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
