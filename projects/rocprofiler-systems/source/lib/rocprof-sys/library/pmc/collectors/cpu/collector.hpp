// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/collectors/base/collector.hpp"
#include "library/pmc/collectors/cpu/cpu_traits.hpp"

namespace rocprofsys::pmc::collectors::cpu
{

/**
 * @brief CPU metrics collector for performance monitoring.
 *
 * Specializes base::collector for CPU devices via procfs.
 * All CPU-specific behavior is defined in cpu_traits.
 *
 * @tparam DeviceProvider Type providing CPU enumeration and driver access
 * @tparam Config Configuration policy providing settings and output policies
 */
template <typename DeviceProvider, typename Config>
using collector = base::collector<cpu_traits<DeviceProvider>, DeviceProvider, Config>;

}  // namespace rocprofsys::pmc::collectors::cpu
