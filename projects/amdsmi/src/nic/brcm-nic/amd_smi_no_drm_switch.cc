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
 
#include <sys/types.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <regex>
#include "amd_smi/impl/nic/amd_smi_no_drm_switch.h"
#include "amd_smi/impl/nic/amd_smi_lspci_commands.h"
#include "amd_smi/impl/amd_smi_common.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_utils.h"

static const char *kPathPciDevices = "/sys/bus/pci/devices/";

namespace amd::smi {

amdsmi_status_t AMDSmiNoDrmSwitch::init() {

    amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();
    auto devices = smi.switch_devices();

    bool has_valid_fds = false;
    for (uint32_t i=0; i < devices.size(); i++) {
        auto rocm_smi_device = devices[i];
        const std::string switch_host_folder = "/sys/class/scsi_host/host" + std::to_string(rocm_smi_device->index());
        std::string switch_dev_folder = switch_host_folder;
        std::vector<char> buf(512);
        ssize_t len;

        do {
          buf.resize(buf.size() + 128);
          len = ::readlink(switch_dev_folder.c_str(), &(buf[0]), buf.size());
        } while (buf.size() == len);

        if (len > 0) {
          buf[len] = '\0';
          switch_dev_folder = std::string(&(buf[0]));
          std::string suffixDel = "host" + std::to_string(rocm_smi_device->index()) +
                                  "/scsi_host/" + "host" +
                                  std::to_string(rocm_smi_device->index()) + "/";
          switch_dev_folder.erase(switch_dev_folder.length() - suffixDel.length());

          auto first = switch_dev_folder.begin();
          auto end = switch_dev_folder.begin() + switch_dev_folder.length() - 12;  // 12 characters. For example: "0000:45:00.0"
          switch_dev_folder.erase(first, end);

          std::string prefixAdd = kPathPciDevices;
          switch_dev_folder = prefixAdd.append(switch_dev_folder);
        }

        std::ostringstream ss;
        std::string vend_path = switch_dev_folder + "/vendor";
        std::string ldev_path = switch_dev_folder + "/device";

        if (FileExists(vend_path.c_str()) && FileExists(ldev_path.c_str())) {
          std::ifstream vfs, dfs;
          vfs.open(vend_path);
          dfs.open(ldev_path);

          if (vfs.is_open() && dfs.is_open()) {
            uint32_t vendor_id;
            uint32_t dev_id;

            vfs >> std::hex >> vendor_id;
            dfs >> std::hex >> dev_id;

            vfs.close();
            dfs.close();

            if (vendor_id == 0x1000 && dev_id == 0x00b2) {
              device_paths_.push_back(switch_dev_folder);
              host_paths_.push_back(switch_host_folder);
              has_valid_fds = true;

              uint64_t bdfid = 0;
              rsmi_status_t ret = rsmi_switch_dev_pci_id_get(i, &bdfid);
              if (ret != RSMI_STATUS_SUCCESS) {
                continue;
              }
              amdsmi_bdf_t bdf = {};
              bdf.function_number = bdfid & 0x7;
              bdf.device_number = (bdfid >> 3) & 0x1f;
              bdf.bus_number = (bdfid >> 8) & 0xff;
              bdf.domain_number = (bdfid >> 32) & 0xffffffff;
              no_drm_bdfs_.push_back(bdf);
            }
          }
        }
    }

    // cannot find any valid fds.
    if (!has_valid_fds) {
        no_drm_bdfs_.clear();
        return AMDSMI_STATUS_INIT_ERROR;
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmSwitch::cleanup() {
    device_paths_.clear();
    host_paths_.clear();
    no_drm_bdfs_.clear();
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmSwitch::amd_query_switch_link( std::string devicePath,
  amdsmi_brcm_switch_link_metric_t &info) {
 
  std::string current_speed = "current_link_speed";
  std::string max_speed = "max_link_speed";
  std::string current_width = "current_link_width";
  std::string max_width = "max_link_width";

  snprintf(info.current_link_speed, sizeof(info.current_link_speed)-1, "%s", smi_brcm_get_value_string(devicePath, current_speed).c_str());
  snprintf(info.max_link_speed, sizeof(info.max_link_speed)-1, "%s", smi_brcm_get_value_string(devicePath, max_speed).c_str());
  snprintf(info.current_link_width, sizeof(info.current_link_width)-1, "%s", smi_brcm_get_value_string(devicePath, current_width).c_str());
  snprintf(info.max_link_width, sizeof(info.max_link_width)-1, "%s", smi_brcm_get_value_string(devicePath, max_width).c_str());

  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmSwitch::amd_query_switch_uuid(std::string bdfStr, std::string& serial) {

  get_lspci_device_data(bdfStr, "Device Serial Number ", serial);

  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmSwitch::amd_query_switch_numa_affinity(std::string devicePath, int32_t *numa_node) {
  std::string numaFile = "numa_node";
  uint32_t numa = smi_brcm_get_value_u32(devicePath, numaFile);
  *numa_node = numa;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmSwitch::amd_query_switch_cpu_affinity(std::string devicePath, std::string &cpu_affinity) {
  std::string cpuAffFile = "cpulistaffinity";
  cpu_affinity = smi_brcm_get_value_string(devicePath, cpuAffFile);
  
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmSwitch::amd_query_switch_device( std::string devicePath,
  amdsmi_brcm_switch_device_metric_t &info) {
 
  std::string brcm_device_aer_dev_correctable      = "aer_dev_correctable";
  std::string brcm_device_aer_dev_fatal            = "aer_dev_fatal";
  std::string brcm_device_aer_dev_nonfatal         = "aer_dev_nonfatal";
  std::string brcm_device_ari_enabled              = "ari_enabled";
  std::string brcm_device_broken_parity_status     = "broken_parity_status";
  std::string brcm_device_class                    = "class";
  std::string brcm_device_config                   = "config";
  std::string brcm_device_consistent_dma_mask_bits = "consistent_dma_mask_bits";
  std::string brcm_device_current_link_speed       = "current_link_speed";
  std::string brcm_device_current_link_width       = "current_link_width";
  std::string brcm_device_d3cold_allowed           = "d3cold_allowed";
  std::string brcm_device_device                   = "device";
  std::string brcm_device_dma_mask_bits            = "dma_mask_bits";
  std::string brcm_device_driver_override          = "driver_override";
  std::string brcm_device_enable                   = "enable";
  std::string brcm_device_irq                      = "irq";
  std::string brcm_device_local_cpulist            = "local_cpulist";
  std::string brcm_device_local_cpus               = "local_cpus";
  std::string brcm_device_max_link_speed           = "max_link_speed";
  std::string brcm_device_max_link_width           = "max_link_width";
  std::string brcm_device_modalias                 = "modalias";
  std::string brcm_device_msi_bus                  = "msi_bus";
  std::string brcm_device_numa_node                = "numa_node";
  std::string brcm_device_pools                    = "pools";
  std::string brcm_device_power_state              = "power_state";
  std::string brcm_device_remove                   = "remove";
  std::string brcm_device_rescan                   = "rescan";
  std::string brcm_device_reset                    = "reset";
  std::string brcm_device_reset_method             = "reset_method";
  std::string brcm_device_resource                 = "resource";
  std::string brcm_device_resource0                = "resource0";
  std::string brcm_device_resource0_wc             = "resource0_wc";
  std::string brcm_device_revision                 = "revision";
  std::string brcm_device_subsystem_device         = "subsystem_device";
  std::string brcm_device_subsystem_vendor         = "subsystem_vendor";
  std::string brcm_device_uevent                   = "uevent";
  std::string brcm_device_vendor                   = "vendor";
  
  snprintf(info.brcm_device_aer_dev_correctable, sizeof(info.brcm_device_aer_dev_correctable)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_aer_dev_correctable).c_str());
  snprintf(info.brcm_device_aer_dev_fatal, sizeof(info.brcm_device_aer_dev_fatal)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_aer_dev_fatal).c_str());
  snprintf(info.brcm_device_aer_dev_nonfatal, sizeof(info.brcm_device_aer_dev_nonfatal)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_aer_dev_nonfatal).c_str());
  snprintf(info.brcm_device_ari_enabled, sizeof(info.brcm_device_ari_enabled)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_ari_enabled).c_str());
  snprintf(info.brcm_device_broken_parity_status, sizeof(info.brcm_device_broken_parity_status)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_broken_parity_status).c_str());
  snprintf(info.brcm_device_class, sizeof(info.brcm_device_class)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_class).c_str());
  snprintf(info.brcm_device_config, sizeof(info.brcm_device_config)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_config).c_str());
  snprintf(info.brcm_device_consistent_dma_mask_bits, sizeof(info.brcm_device_consistent_dma_mask_bits)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_consistent_dma_mask_bits).c_str());
  snprintf(info.brcm_device_current_link_speed, sizeof(info.brcm_device_current_link_speed)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_current_link_speed).c_str());
  snprintf(info.brcm_device_current_link_width, sizeof(info.brcm_device_current_link_width)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_current_link_width).c_str());
  snprintf(info.brcm_device_d3cold_allowed, sizeof(info.brcm_device_d3cold_allowed)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_d3cold_allowed).c_str());
  snprintf(info.brcm_device_device, sizeof(info.brcm_device_device)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_device).c_str());
  snprintf(info.brcm_device_dma_mask_bits, sizeof(info.brcm_device_dma_mask_bits)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_dma_mask_bits).c_str());
  snprintf(info.brcm_device_driver_override, sizeof(info.brcm_device_driver_override)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_driver_override).c_str());
  snprintf(info.brcm_device_enable, sizeof(info.brcm_device_enable)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_enable).c_str());
  snprintf(info.brcm_device_irq, sizeof(info.brcm_device_irq)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_irq).c_str());
  snprintf(info.brcm_device_local_cpulist, sizeof(info.brcm_device_local_cpulist)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_local_cpulist).c_str());
  snprintf(info.brcm_device_local_cpus, sizeof(info.brcm_device_local_cpus)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_local_cpus).c_str());
  snprintf(info.brcm_device_max_link_speed, sizeof(info.brcm_device_max_link_speed)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_max_link_speed).c_str());
  snprintf(info.brcm_device_max_link_width, sizeof(info.brcm_device_max_link_width)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_max_link_width).c_str());
  snprintf(info.brcm_device_modalias, sizeof(info.brcm_device_modalias)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_modalias).c_str());
  snprintf(info.brcm_device_msi_bus, sizeof(info.brcm_device_msi_bus)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_msi_bus).c_str());
  snprintf(info.brcm_device_numa_node, sizeof(info.brcm_device_numa_node)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_numa_node).c_str());
  snprintf(info.brcm_device_pools, sizeof(info.brcm_device_pools)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_pools).c_str());
  snprintf(info.brcm_device_power_state, sizeof(info.brcm_device_power_state)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_power_state).c_str());
  snprintf(info.brcm_device_reset_method, sizeof(info.brcm_device_reset_method)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_reset_method).c_str());
  snprintf(info.brcm_device_resource, sizeof(info.brcm_device_resource)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_resource).c_str());
  snprintf(info.brcm_device_revision, sizeof(info.brcm_device_revision)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_revision).c_str());
  snprintf(info.brcm_device_subsystem_device, sizeof(info.brcm_device_subsystem_device)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_subsystem_device).c_str());
  snprintf(info.brcm_device_subsystem_vendor, sizeof(info.brcm_device_subsystem_vendor)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_subsystem_vendor).c_str());
  snprintf(info.brcm_device_uevent, sizeof(info.brcm_device_uevent)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_uevent).c_str());
  snprintf(info.brcm_device_vendor, sizeof(info.brcm_device_vendor)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_device_vendor).c_str());

  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmSwitch::amd_query_switch_power( std::string devicePath,
  amdsmi_brcm_switch_power_metric_t &info) {
  
  devicePath = devicePath+"/power";
  std::string brcm_power_async                  = "async";
  std::string brcm_power_control                = "control";
  std::string brcm_power_runtime_active_kids    = "runtime_active_kids";
  std::string brcm_power_runtime_active_time    = "runtime_active_time";
  std::string brcm_power_runtime_enabled        = "runtime_enabled";
  std::string brcm_power_runtime_status         = "runtime_status";
  std::string brcm_power_runtime_suspended_time = "runtime_suspended_time";
  std::string brcm_power_runtime_usage          = "runtime_usage";
  std::string brcm_power_wakeup                 = "wakeup";
  std::string brcm_power_wakeup_abort_count     = "wakeup_abort_count";
  std::string brcm_power_wakeup_active          = "wakeup_active";
  std::string brcm_power_wakeup_active_count    = "wakeup_active_count";
  std::string brcm_power_wakeup_count           = "wakeup_count";
  std::string brcm_power_wakeup_expire_count    = "wakeup_expire_count";
  std::string brcm_power_wakeup_last_time_ms    = "wakeup_last_time_ms";
  std::string brcm_power_wakeup_max_time_ms     = "wakeup_max_time_ms";
  std::string brcm_power_wakeup_total_time_ms   = "wakeup_total_time_ms";
  
  snprintf(info.brcm_power_async, sizeof(info.brcm_power_async)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_async).c_str());
  snprintf(info.brcm_power_control, sizeof(info.brcm_power_control)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_control).c_str());
  snprintf(info.brcm_power_runtime_active_kids, sizeof(info.brcm_power_runtime_active_kids)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_runtime_active_kids).c_str());
  snprintf(info.brcm_power_runtime_active_time, sizeof(info.brcm_power_runtime_active_time)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_runtime_active_time).c_str());
  snprintf(info.brcm_power_runtime_enabled, sizeof(info.brcm_power_runtime_enabled)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_runtime_enabled).c_str());
  snprintf(info.brcm_power_runtime_status, sizeof(info.brcm_power_runtime_status)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_runtime_status).c_str());
  snprintf(info.brcm_power_runtime_suspended_time, sizeof(info.brcm_power_runtime_suspended_time)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_runtime_suspended_time).c_str());
  snprintf(info.brcm_power_runtime_usage, sizeof(info.brcm_power_runtime_usage)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_runtime_usage).c_str());
  snprintf(info.brcm_power_wakeup, sizeof(info.brcm_power_wakeup)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_wakeup).c_str());
  snprintf(info.brcm_power_wakeup_abort_count, sizeof(info.brcm_power_wakeup_abort_count)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_wakeup_abort_count).c_str());
  snprintf(info.brcm_power_wakeup_active, sizeof(info.brcm_power_wakeup_active)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_wakeup_active).c_str());
  snprintf(info.brcm_power_wakeup_active_count, sizeof(info.brcm_power_wakeup_active_count)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_wakeup_active_count).c_str());
  snprintf(info.brcm_power_wakeup_count, sizeof(info.brcm_power_wakeup_count)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_wakeup_count).c_str());
  snprintf(info.brcm_power_wakeup_expire_count, sizeof(info.brcm_power_wakeup_expire_count)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_wakeup_expire_count).c_str());
  snprintf(info.brcm_power_wakeup_last_time_ms, sizeof(info.brcm_power_wakeup_last_time_ms)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_wakeup_last_time_ms).c_str());
  snprintf(info.brcm_power_wakeup_max_time_ms, sizeof(info.brcm_power_wakeup_max_time_ms)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_wakeup_max_time_ms).c_str());
  snprintf(info.brcm_power_wakeup_total_time_ms, sizeof(info.brcm_power_wakeup_total_time_ms)-1, "%s", smi_brcm_get_value_string(devicePath, brcm_power_wakeup_total_time_ms).c_str());
  
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmSwitch::get_bdf_by_index(uint32_t switch_index, amdsmi_bdf_t *bdf_info) const {
    if (switch_index + 1 > no_drm_bdfs_.size()) return AMDSMI_STATUS_NOT_SUPPORTED;
    *bdf_info = no_drm_bdfs_[switch_index];
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmSwitch::get_device_path_by_index(uint32_t switch_index, std::string *device_path) const {
    if (switch_index + 1 > device_paths_.size()) return AMDSMI_STATUS_NOT_SUPPORTED;
    *device_path = device_paths_[switch_index];
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmSwitch::get_hwmon_path_by_index(uint32_t switch_index, std::string *hwm_path) const {
    if (switch_index + 1 > host_paths_.size()) return AMDSMI_STATUS_NOT_SUPPORTED;
    *hwm_path = host_paths_[switch_index];
    return AMDSMI_STATUS_SUCCESS;
}

std::vector<std::string>& AMDSmiNoDrmSwitch::get_device_paths() { return device_paths_; }
std::vector<std::string> &AMDSmiNoDrmSwitch::get_hwmon_paths() { return host_paths_; }

bool AMDSmiNoDrmSwitch::check_if_no_drm_is_supported() { return true; }

std::vector<amdsmi_bdf_t> AMDSmiNoDrmSwitch::get_bdfs() {
    return no_drm_bdfs_;
}

}  // namespace amd::smi

