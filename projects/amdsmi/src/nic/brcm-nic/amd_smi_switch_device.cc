/*
 * Copyright (c) Broadcom Inc All Rights Reserved.
 *
 *  Developed by:
 *            Broadcom Inc
 *
 *            www.broadcom.com
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

#include <functional>
#include "amd_smi/impl/nic/amd_smi_switch_device.h"
#include "rocm_smi/rocm_smi_utils.h"

namespace amd::smi {

uint32_t AMDSmiSWITCHDevice::get_switch_id() const {
    return switch_id_;
}

std::string& AMDSmiSWITCHDevice::get_switch_path() {
    return path_;
}

amdsmi_bdf_t AMDSmiSWITCHDevice::get_bdf() {
    return bdf_;
}

amdsmi_status_t AMDSmiSWITCHDevice::get_no_drm_data() {
    amdsmi_status_t ret;
    std::string path;
    amdsmi_bdf_t bdf;
    ret = nodrm_.get_device_path_by_index(switch_id_, &path);
    if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;
    ret = nodrm_.get_bdf_by_index(switch_id_, &bdf);
    if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;
    path_ = path;

    return AMDSMI_STATUS_SUCCESS;
}

pthread_mutex_t* AMDSmiSWITCHDevice::get_mutex() {
    return amd::smi::GetMutex(switch_id_);
}

amdsmi_status_t AMDSmiSWITCHDevice::amd_query_switch_link_info(amdsmi_brcm_switch_link_metric_t& info) const {
  amdsmi_status_t ret;
  std::string devicePath;
  ret = nodrm_.get_device_path_by_index(switch_id_, &devicePath);
  if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;

  return nodrm_.amd_query_switch_link(devicePath, info);
}

amdsmi_status_t AMDSmiSWITCHDevice::amd_query_switch_power_info(amdsmi_brcm_switch_power_metric_t& info) const {
  amdsmi_status_t ret;
  std::string devicePath; //sys/bus/pci/devices/0000:9b:00.0
  ret = nodrm_.get_device_path_by_index(switch_id_, &devicePath);
  if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;

  return nodrm_.amd_query_switch_power(devicePath, info);
}

amdsmi_status_t AMDSmiSWITCHDevice::amd_query_switch_device_info(amdsmi_brcm_switch_device_metric_t& info) const {
  amdsmi_status_t ret;
  std::string devicePath;
  ret = nodrm_.get_device_path_by_index(switch_id_, &devicePath);
  if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;

  return nodrm_.amd_query_switch_device(devicePath, info);
}

amdsmi_status_t AMDSmiSWITCHDevice::amd_query_switch_uuid(std::string& serial) const {
  amdsmi_status_t ret;
  amdsmi_bdf_t bdf = {};
  ret = nodrm_.get_bdf_by_index(switch_id_, &bdf);

  if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;

  char bdf_str[20];
  snprintf(bdf_str, sizeof(bdf_str)-1, "%04lx:%02x:%02x.%d", bdf.domain_number, bdf.bus_number,
          bdf.device_number, bdf.function_number);

  return nodrm_.amd_query_switch_uuid(std::string(bdf_str), serial);
}

amdsmi_status_t AMDSmiSWITCHDevice::amd_query_switch_numa_affinity(int32_t *numa_node) const {
  amdsmi_status_t ret;
  std::string devicePath;
  ret = nodrm_.get_device_path_by_index(switch_id_, &devicePath);
  if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;

  return nodrm_.amd_query_switch_numa_affinity(devicePath, numa_node);
}

amdsmi_status_t AMDSmiSWITCHDevice::amd_query_switch_cpu_affinity(std::string& cpu_affinity) const {
  char bdf_str[20];
  snprintf(bdf_str, sizeof(bdf_str)-1, "%04lx:%02x", bdf_.domain_number, bdf_.bus_number);
  std::stringstream domain_bus_sstream;
  domain_bus_sstream << "/sys/class/pci_bus/" << std::string(bdf_str);
  
  return nodrm_.amd_query_switch_cpu_affinity(domain_bus_sstream.str(), cpu_affinity);
}

}  // namespace amd::smi

