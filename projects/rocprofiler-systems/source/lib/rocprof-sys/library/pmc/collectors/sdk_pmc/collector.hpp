// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/base/collector.hpp"
#include "library/pmc/collectors/common/settings.hpp"
#include "library/pmc/collectors/sdk_pmc/cache_policy.hpp"
#include "library/pmc/collectors/sdk_pmc/perfetto_policy.hpp"
#include "library/pmc/collectors/sdk_pmc/sdk_pmc_traits.hpp"

namespace rocprofsys::pmc::collectors::sdk_pmc
{

/**
 * @brief Default configuration policy for SDK PMC collector.
 */
struct production_config
{
    using SettingsApi = collectors::settings_policy;
    using PerfettoApi = sdk_pmc::perfetto_policy;
    using CacheApi    = sdk_pmc::cache_policy;
};

/**
 * @brief SDK PMC metrics collector for GPU hardware performance counters.
 *
 * This collector specializes the base::collector template for rocprofiler-sdk
 * device_counting_service. All SDK PMC-specific behavior is defined in
 * sdk_pmc_traits.
 *
 * @tparam DeviceProvider Type providing device enumeration from the SDK provider.
 * @tparam Config Configuration policy providing settings and output policies.
 */
template <typename DeviceProvider, typename Config = production_config>
using collector = base::collector<sdk_pmc_traits<DeviceProvider>, DeviceProvider, Config>;

}  // namespace rocprofsys::pmc::collectors::sdk_pmc
