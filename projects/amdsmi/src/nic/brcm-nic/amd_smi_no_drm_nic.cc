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
#include "amd_smi/impl/nic/amd_smi_no_drm_nic.h"
#include "amd_smi/impl/nic/amd_smi_lspci_commands.h"
#include "amd_smi/impl/amd_smi_common.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_main.h"

namespace amd::smi {

amdsmi_status_t AMDSmiNoDrmNIC::init() {
    amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();
    auto devices = smi.nic_devices();

    bool has_valid_hw_mon = false;
    for (uint32_t i=0; i < devices.size(); i++) {
      auto rocm_smi_device = devices[i];
      uint64_t bdfid = rocm_smi_device->bdfid();
      amdsmi_bdf_t bdf = {};
      bdf.function_number = bdfid & 0x7;
      bdf.device_number = (bdfid >> 3) & 0x1f;
      bdf.bus_number = (bdfid >> 8) & 0xff;
      bdf.domain_number = (bdfid >> 32) & 0xffffffff;
      no_drm_bdfs_.push_back(bdf);

      // get interface name from the path
      std::string interface_name = rocm_smi_device->path();
      interface_name = interface_name.substr(interface_name.find_last_of('/') + 1);
      interfaces_.push_back(interface_name);

      const std::string nic_dev_folder = rocm_smi_device->path() + "/device";
      device_paths_.push_back(nic_dev_folder);
      auto nic_dev_dir = opendir(std::string((nic_dev_folder + "/hwmon")).c_str());

      if (nic_dev_dir != nullptr) {
        auto dentry = readdir(nic_dev_dir);
        while (dentry != nullptr) {
          if (memcmp(dentry->d_name, "hwmon", strlen("hwmon")) == 0) {
            if ((strcmp(dentry->d_name, ".") == 0) || (strcmp(dentry->d_name, "..") == 0)) continue;
              const std::string nic_hw_folder = nic_dev_folder + "/hwmon/" + std::string(dentry->d_name);
              hwmon_paths_.push_back(nic_hw_folder);
              has_valid_hw_mon = true;
              break;
          }
          dentry = readdir(nic_dev_dir);
        }
        closedir(nic_dev_dir);
      }

      // cannot find any valid fds.
      if (!has_valid_hw_mon) {
          hwmon_paths_.push_back("");
      }
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmNIC::cleanup() {
    // Clear the vectors that hold the information about the NICs
    device_paths_.clear();
    hwmon_paths_.clear();
    no_drm_bdfs_.clear();
    interfaces_.clear();
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmNIC::amd_query_nic_info(uint32_t nic_index, amdsmi_brcm_nic_info_t& info) {
    // Retrieve information about a specific NIC.
    //
    // Parameters:
    // - nic_index: The index of the NIC in the list of available NICs.
    // - info: A reference to an object of type amdsmi_brcm_nic_info_t,
    //   which will contain information about the NIC.

    amdsmi_status_t ret = AMDSMI_STATUS_SUCCESS;
    get_bdf_by_index(nic_index, &info.nic_bdf);
    std::string strInterfaceName;
    get_interface_name_by_index(nic_index, &strInterfaceName);
    snprintf(info.nic_device_name, sizeof(info.nic_device_name)-1, "%s", strInterfaceName.c_str());

    char bdf_str[20];
    snprintf(bdf_str, sizeof(bdf_str)-1, "%04lx:%02x:%02x.%d", info.nic_bdf.domain_number, info.nic_bdf.bus_number, info.nic_bdf.device_number,
            info.nic_bdf.function_number);

    std::string part_number, fw_version;
    try {
      get_lspci_device_data(std::string(bdf_str), "PN] Part number: ", part_number);
      get_lspci_device_data(std::string(bdf_str), "V3] Vendor specific: ", fw_version);

      snprintf(info.nic_part_number, sizeof(info.nic_part_number)-1, "%s", part_number.c_str());
      snprintf(info.nic_firmware_version, sizeof(info.nic_firmware_version)-1, "%s", fw_version.c_str());

    } catch (const std::invalid_argument &e) {
        std::cerr << "AMDSmiNoDrmNIC::amd_query_nic_info - Error: Invalid argument exception caught in std::stoi.\n"
                  << "Exception message: " << e.what() << std::endl;
    } catch (const std::out_of_range &e) {
        std::cerr << "AMDSmiNoDrmNIC::amd_query_nic_info - Error: Out of range exception caught in std::stoi.\n"
                  << "Exception message: " << e.what() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "AMDSmiNoDrmNIC::amd_query_nic_info - An error occurred: " << e.what()
                  << std::endl;
    }

    std::string devicePath;
    get_device_path_by_index(nic_index, &devicePath); 
    std::string netPath =  devicePath + "/net";
    auto net_node_dir = opendir(netPath.c_str());
    if (net_node_dir != nullptr) {
      auto dentry = readdir(net_node_dir);
      std::string macPath;
      while ((dentry = readdir(net_node_dir)) != nullptr) {
        if ((strcmp(dentry->d_name, ".") == 0) || (strcmp(dentry->d_name, "..") == 0)) {
          continue;
        }
        macPath = netPath + "/" + dentry->d_name;
        std::string macAddress = "address";
        std::string strUUID = smi_brcm_get_value_string(macPath, macAddress);
        snprintf(info.nic_uuid, sizeof(info.nic_uuid)-1, "%s", strUUID.c_str());
      }
      closedir(net_node_dir);
    }
    
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmNIC::amd_query_nic_temp(std::string hwmonPath,
      amdsmi_brcm_nic_temperature_metric_t &info) {
    // Get nic temperature info
    std::string crit_alarm = "temp1_crit_alarm";
    std::string emergency_alarm = "temp1_emergency_alarm";
    std::string shutdown_alarm = "temp1_shutdown_alarm";
    std::string max_alarm = "temp1_max_alarm";

    std::string nic_crit = "temp1_crit";
    std::string nic_emergency = "temp1_emergency";
    std::string nic_input = "temp1_input";
    std::string nic_max = "temp1_max";
    std::string nic_shutdown = "temp1_shutdown";
  
    try {
        info.nic_temp_crit_alarm = smi_brcm_get_value_u32(hwmonPath, crit_alarm);
        info.nic_temp_emergency_alarm = smi_brcm_get_value_u32(hwmonPath, emergency_alarm);
      info.nic_temp_shutdown_alarm = smi_brcm_get_value_u32(hwmonPath, shutdown_alarm);
      info.nic_temp_max_alarm = smi_brcm_get_value_u32(hwmonPath, max_alarm);
    
      info.nic_temp_crit = smi_brcm_get_value_u32(hwmonPath, nic_crit);
      info.nic_temp_emergency = smi_brcm_get_value_u32(hwmonPath, nic_emergency);
      info.nic_temp_input = smi_brcm_get_value_u32(hwmonPath, nic_input);
      info.nic_temp_max = smi_brcm_get_value_u32(hwmonPath, nic_max);
      info.nic_temp_shutdown = smi_brcm_get_value_u32(hwmonPath, nic_shutdown);
    } catch (const std::exception& e) {
        std::cerr << "AMDSmiNoDrmNIC::amd_query_nic_temp - An error occurred: " << e.what()
                  << std::endl;
    }
    
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmNIC::amd_query_nic_power(std::string hwmonPath, amdsmi_brcm_nic_hwmon_power_t &info) {
    // Get power metrics for a NIC
    try {
        hwmonPath = hwmonPath+"/power";
        std::string async = "async";
        std::string control = "control";
        std::string runtime_active_kids = "runtime_active_kids";
        std::string runtime_active_time = "runtime_active_time";
        std::string runtime_enabled = "runtime_enabled";
        std::string runtime_status = "runtime_status";
        std::string runtime_suspended_time = "runtime_suspended_time";
        std::string runtime_usage = "runtime_usage";

        snprintf(info.nic_power_async, sizeof(info.nic_power_async)-1, "%s", smi_brcm_get_value_string(hwmonPath, async).c_str());
        snprintf(info.nic_power_control, sizeof(info.nic_power_control)-1, "%s", smi_brcm_get_value_string(hwmonPath, control).c_str());
        info.nic_power_runtime_active_time = smi_brcm_get_value_u32(hwmonPath, runtime_active_time);
        snprintf(info.nic_power_runtime_status, sizeof(info.nic_power_runtime_status)-1, "%s", smi_brcm_get_value_string(hwmonPath, runtime_status).c_str());
        info.nic_power_runtime_usage = smi_brcm_get_value_u32(hwmonPath, runtime_usage);
        info.nic_power_runtime_active_kids = smi_brcm_get_value_u32(hwmonPath, runtime_active_kids);
        snprintf(info.nic_power_runtime_enabled, sizeof(info.nic_power_runtime_enabled)-1, "%s", smi_brcm_get_value_string(hwmonPath, runtime_enabled).c_str());
        info.nic_power_runtime_suspended_time = smi_brcm_get_value_u32(hwmonPath, runtime_suspended_time);

    } catch (const std::invalid_argument& e) {
        printf("AMDSmiNoDrmNIC::amd_query_nic_power - Invalid argument: %s\n", e.what());
    } catch (const std::out_of_range& e) {
        printf("Out of range error: %s\n", e.what());
    } catch (...) {
        printf("AMDSmiNoDrmNIC::amd_query_nic_power - Error: Exception caught during NIC power query.\n");
    }
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmNIC::amd_query_nic_device(std::string hwmonPath, amdsmi_brcm_nic_hwmon_device_t &info) {
  
    try {
        hwmonPath = hwmonPath+"/device";
        std::string aer_dev_correctable = "aer_dev_correctable";
        std::string aer_dev_fatal = "aer_dev_fatal";
        std::string aer_dev_nonfatal = "aer_dev_nonfatal";
        std::string ari_enabled = "ari_enabled";
        std::string broken_parity_status = "broken_parity_status";
        std::string device_class = "class";
        std::string config = "config";
        std::string consistent_dma_mask_bit = "consistent_dma_mask_bit";
        std::string current_link_speed = "current_link_speed";
        std::string current_link_width = "current_link_width";
        std::string d3cold_allowed = "d3cold_allowed";
        std::string device = "device";
        std::string dma_mask_bits = "dma_mask_bits";
        std::string driver_override = "driver_override";
        std::string enable = "enable";
        std::string irq = "irq";
        std::string local_cpulist = "local_cpulist";
        std::string local_cpus = "local_cpus";
        std::string max_link_speed = "max_link_speed";
        std::string max_link_width = "max_link_width";
        std::string modalias = "modalias";
        std::string msi_bus = "msi_bus";
        std::string numa_node = "numa_node";
        std::string pools = "pools";
        std::string power_state = "power_state";
        std::string reset_method = "reset_method";
        std::string resource = "resource";
        std::string revision = "revision";
        std::string sriov_drivers_autoprobe = "sriov_drivers_autoprobe";
        std::string sriov_numvfs = "sriov_numvfs";
        std::string sriov_offset = "sriov_offset";
        std::string sriov_stride = "sriov_stride";
        std::string sriov_totalvfs = "sriov_totalvfs";
        std::string sriov_vf_device = "sriov_vf_device";
        std::string sriov_vf_total_msix = "sriov_vf_total_msix";
        std::string subsystem_device = "subsystem_device";
        std::string subsystem_vendor = "subsystem_vendor";
        std::string uevent = "uevent";
        std::string vendor = "vendor";
        std::string vpd = "vpd";

        snprintf(info.nic_device_aer_dev_correctable, sizeof(info.nic_device_aer_dev_correctable)-1, "%s", smi_brcm_get_value_string(hwmonPath, aer_dev_correctable).c_str());
        snprintf(info.nic_device_aer_dev_fatal, sizeof(info.nic_device_aer_dev_fatal)-1, "%s", smi_brcm_get_value_string(hwmonPath, aer_dev_fatal).c_str());
        snprintf(info.nic_device_aer_dev_nonfatal, sizeof(info.nic_device_aer_dev_nonfatal)-1, "%s", smi_brcm_get_value_string(hwmonPath, aer_dev_nonfatal).c_str());
        info.nic_device_ari_enabled = smi_brcm_get_value_u32(hwmonPath, ari_enabled);
        info.nic_device_broken_parity_status = smi_brcm_get_value_u32(hwmonPath, broken_parity_status);
        snprintf(info.nic_device_class, sizeof(info.nic_device_class)-1, "%s", smi_brcm_get_value_string(hwmonPath, device_class).c_str());
        snprintf(info.nic_device_config, sizeof(info.nic_device_config)-1, "%s", smi_brcm_get_value_string(hwmonPath, config).c_str());
        info.nic_device_consistent_dma_mask_bits = smi_brcm_get_value_u32(hwmonPath, consistent_dma_mask_bit);
        snprintf(info.nic_device_current_link_speed, sizeof(info.nic_device_current_link_speed)-1, "%s", smi_brcm_get_value_string(hwmonPath, current_link_speed).c_str());
        info.nic_device_current_link_width = smi_brcm_get_value_u32(hwmonPath, current_link_width);
        info.nic_device_d3cold_allowed = smi_brcm_get_value_u32(hwmonPath, d3cold_allowed);
        snprintf(info.nic_device_device, sizeof(info.nic_device_device)-1, "%s", smi_brcm_get_value_string(hwmonPath, device).c_str());
        info.nic_device_dma_mask_bits = smi_brcm_get_value_u32(hwmonPath, dma_mask_bits);
        snprintf(info.nic_device_driver_override, sizeof(info.nic_device_driver_override)-1, "%s", smi_brcm_get_value_string(hwmonPath, driver_override).c_str());
        info.nic_device_enable = smi_brcm_get_value_u32(hwmonPath, enable);
        info.nic_device_irq = smi_brcm_get_value_u32(hwmonPath, irq);
        snprintf(info.nic_device_local_cpulist, sizeof(info.nic_device_local_cpulist)-1, "%s", smi_brcm_get_value_string(hwmonPath, local_cpulist).c_str());
        snprintf(info.nic_device_local_cpus, sizeof(info.nic_device_local_cpus)-1, "%s", smi_brcm_get_value_string(hwmonPath, local_cpus).c_str());
        snprintf(info.nic_device_max_link_speed, sizeof(info.nic_device_max_link_speed)-1, "%s", smi_brcm_get_value_string(hwmonPath, max_link_speed).c_str());
        info.nic_device_max_link_width = smi_brcm_get_value_u32(hwmonPath, max_link_width);
        snprintf(info.nic_device_modalias, sizeof(info.nic_device_modalias)-1, "%s", smi_brcm_get_value_string(hwmonPath, modalias).c_str());
        info.nic_device_msi_bus = smi_brcm_get_value_u32(hwmonPath, msi_bus);
        info.nic_device_numa_node = smi_brcm_get_value_u32(hwmonPath, numa_node);
        snprintf(info.nic_device_pools, sizeof(info.nic_device_pools)-1, "%s", smi_brcm_get_value_string(hwmonPath, pools).c_str());
        snprintf(info.nic_device_power_state, sizeof(info.nic_device_power_state)-1, "%s", smi_brcm_get_value_string(hwmonPath, power_state).c_str());
        snprintf(info.nic_device_reset_method, sizeof(info.nic_device_reset_method)-1, "%s", smi_brcm_get_value_string(hwmonPath, reset_method).c_str());
        snprintf(info.nic_device_resource, sizeof(info.nic_device_resource)-1, "%s", smi_brcm_get_value_string(hwmonPath, resource).c_str());
        snprintf(info.nic_device_revision, sizeof(info.nic_device_revision)-1, "%s", smi_brcm_get_value_string(hwmonPath, revision).c_str());
        info.nic_device_sriov_drivers_autoprobe = smi_brcm_get_value_u32(hwmonPath, sriov_drivers_autoprobe);
        info.nic_device_sriov_numvfs = smi_brcm_get_value_u32(hwmonPath, sriov_numvfs);
        info.nic_device_sriov_offset = smi_brcm_get_value_u32(hwmonPath, sriov_offset);
        info.nic_device_sriov_stride = smi_brcm_get_value_u32(hwmonPath, sriov_stride);
        info.nic_device_sriov_totalvfs = smi_brcm_get_value_u32(hwmonPath, sriov_totalvfs);
        info.nic_device_sriov_vf_device = smi_brcm_get_value_u32(hwmonPath, sriov_vf_device);
        info.nic_device_sriov_vf_total_msix = smi_brcm_get_value_u32(hwmonPath, sriov_vf_total_msix);
        snprintf(info.nic_device_subsystem_device, sizeof(info.nic_device_subsystem_device-1), "%s", smi_brcm_get_value_string(hwmonPath, subsystem_device).c_str());
        snprintf(info.nic_device_subsystem_vendor, sizeof(info.nic_device_subsystem_vendor-1), "%s", smi_brcm_get_value_string(hwmonPath, subsystem_vendor).c_str());
        snprintf(info.nic_device_uevent, sizeof(info.nic_device_uevent-1), "%s", smi_brcm_get_value_string(hwmonPath, uevent).c_str());
        snprintf(info.nic_device_vendor, sizeof(info.nic_device_vendor-1), "%s", smi_brcm_get_value_string(hwmonPath, vendor).c_str());
        snprintf(info.nic_device_vpd, sizeof(info.nic_device_vpd-1), "%s", smi_brcm_get_value_string(hwmonPath, vpd).c_str());

    } catch (const std::invalid_argument& e) {
        std::cerr << "AMDSmiNoDrmNIC::amd_query_nic_device - Error: Invalid argument exception caught in std::stoi.\n"
                  << "Exception message: " << e.what() << std::endl;
    } catch (const std::out_of_range& e) {
        std::cerr << "AMDSmiNoDrmNIC::amd_query_nic_device - Error: Out of range exception caught in std::stoi.\n"
                  << "Exception message: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "AMDSmiNoDrmNIC::amd_query_nic_device - An error occurred: " << e.what() << std::endl;
    }
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmNIC::amd_query_nic_fw_info(std::string bdfStr, 
  amdsmi_brcm_nic_firmware_t &info) {
    // Retrieve firmware version information from the NIC.
    //
    // Args:
    //    bdfStr (std::string): Bus-Device-Function value of the NIC.
    //    info (amdsmi_brcm_nic_firmware_t): Structure to hold the firmware
    //                                        version information.
    std::string fw_pkg_version, fw_efi_version, fw_version, fw_ncsi_version, fw_roce_version;
    try {
      get_lspci_device_data(bdfStr, "V0] Vendor specific: ", fw_pkg_version);
      get_lspci_device_data(bdfStr, "V1] Vendor specific: ", fw_efi_version);
      get_lspci_device_data(bdfStr, "V3] Vendor specific: ", fw_version);
      get_lspci_device_data(bdfStr, "V8] Vendor specific: ", fw_ncsi_version);
      get_lspci_device_data(bdfStr, "VA] Vendor specific: ", fw_roce_version);

      snprintf(info.nic_fw_pkg_version, sizeof(info.nic_fw_pkg_version)-1, "%s", fw_pkg_version.c_str());
      snprintf(info.nic_fw_efi_version, sizeof(info.nic_fw_efi_version)-1, "%s", fw_efi_version.c_str());
      snprintf(info.nic_fw_version, sizeof(info.nic_fw_version)-1, "%s", fw_version.c_str());
      snprintf(info.nic_fw_ncsi_version, sizeof(info.nic_fw_ncsi_version)-1, "%s", fw_ncsi_version.c_str());
      snprintf(info.nic_fw_roce_version, sizeof(info.nic_fw_roce_version)-1, "%s", fw_roce_version.c_str());

    } catch (const std::invalid_argument &e) {
        std::cerr << "AMDSmiNoDrmNIC::amd_query_nic_fw_info - Error: Invalid argument exception caught in std::stoi.\n"
                  << "Exception message: " << e.what() << std::endl;
    } catch (const std::out_of_range &e) {
        std::cerr << "AMDSmiNoDrmNIC::amd_query_nic_fw_info - Error: Out of range exception caught in std::stoi.\n"
                  << "Exception message: " << e.what() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "AMDSmiNoDrmNIC::amd_query_nic_fw_info - An error occurred: " << e.what()
                  << std::endl;
    }
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmNIC::get_interface_name_by_index(uint32_t nic_index, std::string* interface_name) const {
    // Retrieve the interface name for the given NIC index
    if (nic_index + 1 > interfaces_.size()) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << " | "
           << "Failed to get interface name for NIC #" << nic_index << ". Error " << AMDSMI_STATUS_NOT_SUPPORTED << ".";
        LOG_DEBUG(ss);
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    *interface_name = interfaces_[nic_index];
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmNIC::get_bdf_by_index(uint32_t nic_index, amdsmi_bdf_t *bdf_info) const {
    // Retrieve the BDF for the given NIC index
    if (nic_index + 1 > no_drm_bdfs_.size()) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << " | "
           << "Failed to get BDF for NIC #" << nic_index << ". Error " << AMDSMI_STATUS_NOT_SUPPORTED << ".";
        LOG_DEBUG(ss);
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }
    *bdf_info = no_drm_bdfs_[nic_index];
    return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t AMDSmiNoDrmNIC::get_device_path_by_index(uint32_t nic_index, std::string *device_path) const {
    // Retrieve the device path for the given NIC index
    if (nic_index + 1 > device_paths_.size()) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << " | "
           << "Failed to get device path for NIC #" << nic_index << ". Error " << AMDSMI_STATUS_NOT_SUPPORTED << ".";
        LOG_DEBUG(ss);
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }
    *device_path = device_paths_[nic_index];
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmNIC::get_hwmon_path_by_index(uint32_t nic_index, std::string *hwm_path) const {
    // Retrieve the hwmon path for the given NIC index
    if (nic_index + 1 > hwmon_paths_.size()) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << " | "
           << "Failed to get hwmon path for NIC #" << nic_index << ". Error " << AMDSMI_STATUS_NOT_SUPPORTED << ".";
        LOG_DEBUG(ss);
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }
    *hwm_path = hwmon_paths_[nic_index];
    return AMDSMI_STATUS_SUCCESS;
}

std::vector<std::string>& AMDSmiNoDrmNIC::get_device_paths() { 
    // Return reference to vector of device paths.
    return device_paths_; 
}
std::vector<std::string>& AMDSmiNoDrmNIC::get_hwmon_paths() { 
    // Return reference to vector of hwmon paths.
    return hwmon_paths_; 
}

bool AMDSmiNoDrmNIC::check_if_no_drm_is_supported() {
    // Return true if no-drm NIC is supported.
    return true;
}

std::vector<amdsmi_bdf_t> AMDSmiNoDrmNIC::get_bdfs() {
    // Return reference to vector of BDFs.
    return no_drm_bdfs_;
}


amdsmi_status_t AMDSmiNoDrmNIC::amd_query_nic_uuid(std::string devicePath, std::string &version) {
  // Get NIC MAC address
  std::string netPath = devicePath + "/net";
  auto net_node_dir = opendir(netPath.c_str());
    if (net_node_dir == nullptr) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << " | "
           << "Failed to open net node directory: " << netPath << ". Error " << AMDSMI_STATUS_FILE_ERROR << ".";
        LOG_DEBUG(ss);

    return AMDSMI_STATUS_FILE_ERROR;
  }
  auto dentry = readdir(net_node_dir);
  std::string macPath;
  while ((dentry = readdir(net_node_dir)) != nullptr) {
    // Skip "." and ".." directories
    if ((strcmp(dentry->d_name, ".") == 0) || (strcmp(dentry->d_name, "..") == 0)) {
      continue;
    }
    macPath = netPath + "/" + dentry->d_name;
    std::string macAddress = "address";
    version = smi_brcm_get_value_string(macPath, macAddress);
  }
  closedir(net_node_dir);

  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmNIC::amd_query_nic_numa_affinity(std::string devicePath, int32_t *numa_node) {
  // Get NIC NUMA affinity
  std::string numaFile = "numa_node";
  uint32_t numa = smi_brcm_get_value_u32(devicePath, numaFile);
  *numa_node = numa;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiNoDrmNIC::amd_query_nic_cpu_affinity(std::string devicePath, std::string &cpu_affinity) {
  // Get NIC CPU affinity
  std::string cpuAffFile = "cpulistaffinity";
  cpu_affinity = smi_brcm_get_value_string(devicePath, cpuAffFile);
  
  return AMDSMI_STATUS_SUCCESS;
}

}  // namespace amd::smi
