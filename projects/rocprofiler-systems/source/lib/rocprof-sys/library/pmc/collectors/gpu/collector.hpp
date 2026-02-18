// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/collectors/base/collector.hpp"
#include "library/pmc/collectors/gpu/gpu_traits.hpp"

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace gpu
{

#if ROCPROFSYS_USE_ROCM > 0

/**
 * @brief GPU metrics collector for performance monitoring.
 *
 * This collector manages the lifecycle of GPU performance monitoring, including
 * device enumeration, metric sampling, and data storage. It uses a policy-based design
 * pattern via template parameters to allow compile-time dependency injection.
 *
 * This is a type alias to the unified base::collector with GPU-specific traits.
 *
 * @tparam DeviceProvider Type providing GPU device enumeration and management
 * @tparam Config Configuration policy providing settings and output policies (Perfetto,
 * RocPD)
 */
template <typename DeviceProvider, typename Config>
using collector = base::collector<gpu_traits<typename DeviceProvider::driver_t>,
                                  DeviceProvider, Config>;

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace gpu
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
