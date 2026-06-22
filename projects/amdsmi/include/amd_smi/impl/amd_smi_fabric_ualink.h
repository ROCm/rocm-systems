/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_FABRIC_UALINK_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_FABRIC_UALINK_H_

#include <string>
#include <string_view>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"

namespace amd::smi {

namespace fabric_ualink {

/**
 *  Filesystem-seam cores for the fabric apply/query paths.
 *
 *  These take the resolved UALink @p ualink_root and a @p device_supports_ualink
 *  flag instead of a live AMDSmiGPUDevice, so the sysfs serialize/parse round-trip
 *  can be exercised against a temp directory without hardware. The device-taking
 *  free functions in the .cc wrap these; commit-verify diagnostics live in those
 *  wrappers, not here.
 */
auto apply_ppod_config_at(const std::string& ualink_root, bool device_supports_ualink,
                          const amdsmi_fabric_ppod_config_t& config) -> amdsmi_status_t;
auto apply_vpod_config_at(const std::string& ualink_root, bool device_supports_ualink,
                          const amdsmi_fabric_vpod_config_t& config) -> amdsmi_status_t;
auto apply_station_config_at(const std::string& ualink_root, bool device_supports_ualink,
                             const amdsmi_fabric_station_config_t& config) -> amdsmi_status_t;

auto query_ppod_config_at(const std::string& ualink_root, bool device_supports_ualink,
                          amdsmi_fabric_ppod_config_t& config,
                          std::string_view subdir = kUALOE_UALINK_SETUP_SUBDIR) -> amdsmi_status_t;
auto query_vpod_config_at(const std::string& ualink_root, bool device_supports_ualink,
                          amdsmi_fabric_vpod_config_t& config,
                          std::string_view subdir = kUALOE_UALINK_CONFIG_SUBDIR) -> amdsmi_status_t;
auto query_station_config_at(const std::string& ualink_root, bool device_supports_ualink,
                             amdsmi_fabric_station_config_t& config,
                             std::string_view subdir = kUALOE_UALINK_STATIONS_SUBDIR)
    -> amdsmi_status_t;

}  // namespace fabric_ualink

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_FABRIC_UALINK_H_
