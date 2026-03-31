// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/collectors/base/collector.hpp"
#include "library/pmc/collectors/gpu/gpu_traits.hpp"

namespace rocprofsys::pmc::collectors::gpu
{

/**
 * @brief GPU metrics collector for performance monitoring.
 *
 * This collector specializes the base::collector template for GPU devices
 * using AMD SMI. All GPU-specific behavior is defined in gpu_traits.
 *
 * SDMA delta computation is handled internally by the device class to maintain
 * state across samples while keeping traits stateless.
 *
 * @tparam DeviceProvider Type providing GPU device enumeration and management
 * @tparam Config Configuration policy providing settings and output policies
 */
template <typename DeviceProvider, typename Config>
using collector = base::collector<gpu_traits<DeviceProvider>, DeviceProvider, Config>;

}  // namespace rocprofsys::pmc::collectors::gpu
