// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/base/collector.hpp"
#include "library/pmc/collectors/hipfile/hipfile_traits.hpp"

namespace rocprofsys::pmc::collectors::hipfile
{

/**
 * @brief hipFile GPU-direct storage I/O collector.
 *
 * Specializes the base collector template for hipFile. All hipFile-specific behaviour
 * lives in hipfile_traits, so lifecycle, device disabling on repeated failure, sample
 * accounting, and the zeroed pause sample are shared with the GPU, NIC, and CPU
 * collectors rather than reimplemented here.
 *
 * @tparam DeviceProvider Type providing hipFile device enumeration and management
 * @tparam DeviceType     Concrete device type, parameterized on its backend
 * @tparam Config         Configuration policy providing settings and output policies
 */
template <typename DeviceProvider, typename DeviceType, typename Config>
using collector =
    base::collector<hipfile_traits<DeviceProvider, DeviceType>, DeviceProvider, Config>;

}  // namespace rocprofsys::pmc::collectors::hipfile
