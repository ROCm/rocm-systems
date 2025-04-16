/*
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017-2023, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_kfd.h"
#include "rocm_smi/rocm_smi_logger.h"


static const char *kPathDRMRoot = "/sys/class/drm";
static const char *kPathHWMonRoot = "/sys/class/hwmon";
static const char *kPathPowerRoot = "/sys/kernel/debug/dri";

static const char *kAMDMonitorTypes[] = {"radeon", "amdgpu", ""};

namespace amd {
namespace smi {

static uint32_t GetDeviceIndex(const std::string s) {
  std::string t = s;
  size_t tmp = t.find_last_not_of("0123456789");
  t.erase(0, tmp+1);

  assert(stoi(t) >= 0);
  return static_cast<uint32_t>(stoi(t));
}

// Find the drm minor from from sysfs path "/sys/class/drm/cardX/device/drm".
// From the directory renderDN in that sysfs path, the drm minor can be
// computed for cardX.
// On success, return drm_minor which is >= 128 otherwise return 0
static uint32_t  GetDrmRenderMinor(const std::string s) {
  std::ostringstream ss;
  std::string drm_path = s;
  int drm_minor = 0;
  const std::string render_file_prefix = "renderD";
  const uint64_t prefix_size = render_file_prefix.size();
  drm_path += "/device/drm";

  auto drm_dir = opendir(drm_path.c_str());
  if (drm_dir == nullptr)
    return 0;

  auto dentry = readdir(drm_dir);

  while (dentry != nullptr) {
    std::string render_file = dentry->d_name;
    if (!render_file.compare(0, prefix_size, render_file_prefix)) {
      drm_minor = stoi(render_file.substr(prefix_size));
      if (drm_minor)
        break;
    }
    dentry = readdir(drm_dir);
  }

  if (closedir(drm_dir)) {
    return 0;
  }

  ss << __PRETTY_FUNCTION__ << " | Discovered drmRenderMinor = "
     << std::to_string(drm_minor) << " | For drm_path = " << drm_path << " | ";
  LOG_DEBUG(ss);
  return static_cast<uint32_t>(drm_minor);
}

// Find the drm minor from from sysfs path "/sys/class/drm/renderDX/device/drm".
// From the directory cardN in that sysfs path, the card number can be
// computed for renderDX.
// On success, return drm_minor which is >= 128 otherwise return 0xFFFFFFFF
static uint32_t  GetCard(const std::string s) {
  std::ostringstream ss;
  std::string drm_path = s;
  int card_num = -1;
  const std::string card_file_prefix = "card";
  const uint64_t prefix_size = card_file_prefix.size();
  drm_path += "/device/drm";

  auto card_dir = opendir(drm_path.c_str());
  if (card_dir == nullptr)
    return static_cast<uint32_t>(-1);

  auto dentry = readdir(card_dir);

  while (dentry != nullptr) {
    std::string card_file = dentry->d_name;
    if (!card_file.compare(0, prefix_size, card_file_prefix)) {
      card_num = stoi(card_file.substr(prefix_size));
      if (card_num)
        break;
    }
    dentry = readdir(card_dir);
  }

  if (closedir(card_dir)) {
    return static_cast<uint32_t>(-1);
  }

  ss << __PRETTY_FUNCTION__ << " | Discovered card = "
     << std::to_string(card_num) << " | For drm_path = " << drm_path << " | ";
  LOG_DEBUG(ss);
  return static_cast<uint32_t>(card_num);
}

//  Determine if provided string is a bdfid pci path directory of the form
//  XXXX:XX:XX.X,
//  domain:bus:device.function
//
//  where X is a hex integer (lower case is expected). If so, write the value
//  to bdfid
static bool bdfid_from_path(const std::string in_name, uint64_t *bdfid) {
  char *p = nullptr;
  char *name_start;
  char name[13] = {'\0'};
  uint64_t tmp;

  assert(bdfid != nullptr);

  if (in_name.size() != 12) {
    return false;
  }

  tmp = in_name.copy(name, 12);
  assert(tmp == 12);

  // BDFID = ((<DOMAIN> & 0xffff) << 32) | ((<BUS> & 0xff) << 8) |
            //                        ((device& 0x1f) <<3 ) | (function & 0x7)
  *bdfid = 0;
  name_start = name;
  p = name_start;

  // Match this: XXXX:xx:xx.x
  tmp = std::strtoul(p, &p, 16);
  if (*p != ':' || p - name_start != 4) {
    return false;
  }
  *bdfid |= tmp << 32;

  // Match this: xxxx:XX:xx.x
  p++;  // Skip past ':'
  tmp = std::strtoul(p, &p, 16);
  if (*p != ':' || p - name_start != 7) {
    return false;
  }
  *bdfid |= tmp << 8;

  // Match this: xxxx:xx:XX.x
  p++;  // Skip past ':'
  tmp = std::strtoul(p, &p, 16);
  if (*p != '.' || p - name_start != 10) {
    return false;
  }
  *bdfid |= tmp << 3;

  // Match this: xxxx:xx:xx.X
  p++;  // Skip past '.'
  tmp = std::strtoul(p, &p, 16);
  if (*p != '\0' || p - name_start != 12) {
    return false;
  }
  *bdfid |= tmp;

  return true;
}

// 0 = successful bdfid found
// 1 = not a good bdfid found
[[maybe_unused]] static uint32_t ConstructBDFID(std::string path, uint64_t *bdfid) {
  std::ostringstream ss;
  assert(bdfid != nullptr);
  const unsigned int MAX_BDF_LENGTH = 512;
  char tpath[MAX_BDF_LENGTH] = {'\0'};
  ssize_t ret;
  memset(tpath, 0, MAX_BDF_LENGTH);

  ret = readlink(path.c_str(), tpath, MAX_BDF_LENGTH);

  assert(ret > 0);
  assert(ret < MAX_BDF_LENGTH);

  if (ret <= 0 || ret >= MAX_BDF_LENGTH) {
    ss << __PRETTY_FUNCTION__ << " | readlink failed for path = "
       << path << " | ret = " << ret
       << " | errno = " << errno
       << " | error = " << strerror(errno);
    // std::cout << ss.str() << std::endl;
    LOG_ERROR(ss);
    return 1;
  }

  // We are looking for the last element in the path that has the form
  //  XXXX:XX:XX.X, where X is a hex integer (lower case is expected)
  std::size_t slash_i;
  std::size_t end_i;
  std::string tmp;

  std::string tpath_str(tpath);

  end_i = tpath_str.size() - 1;
  while (end_i > 0) {
    slash_i = tpath_str.find_last_of('/', end_i);
    tmp = tpath_str.substr(slash_i + 1, end_i - slash_i);

    if (bdfid_from_path(tmp, bdfid)) {
      ss << __PRETTY_FUNCTION__ << " | Found bdfid = "
         << print_int_as_hex(*bdfid, true, 8) << " | from path = "
         << path << " | tmp = " << tmp << std::endl;
      LOG_INFO(ss);
      return 0;
    }
    end_i = slash_i - 1;
  }
  ss << __PRETTY_FUNCTION__ << " | No valid bdfid found in path = "
     << path << " | tpath = " << tpath
     << " | errno = " << errno
     << " | error = " << strerror(errno) << std::endl;
  LOG_ERROR(ss);
  return 1;
}

void
RocmSMI::Initialize(uint64_t flags) {
  auto i = 0;
  uint32_t ret;
  int i_ret;
  std::ostringstream ss;



  assert(ref_count_ == 1);
  if (ref_count_ != 1) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
            "Unexpected: RocmSMI ref_count_ != 1");
  }

  init_options_ = flags;

  euid_ = geteuid();

  GetEnvVariables();
  // To help debug env variable issues
  // debugRSMIEnvVarInfo();

  if (ROCmLogging::Logger::getInstance()->isLoggerEnabled()) {
    ROCmLogging::Logger::getInstance()->enableAllLogLevels();
    LOG_ALWAYS("=============== ROCM SMI initialize ================");
    logSystemDetails();
  }
  // Leaving below to allow developers to check current log settings
  // std::string logSettings = ROCmLogging::Logger::getInstance()->getLogSettings();
  // std::cout << "Current log settings:\n" << logSettings << std::endl;

  while (!std::string(kAMDMonitorTypes[i]).empty()) {
      amd_monitor_types_.insert(kAMDMonitorTypes[i]);
      ++i;
  }

  // DiscoverAmdgpuDevices() will search for devices and monitors and update
  // internal data structures.
  ret = DiscoverAmdgpuDevices();
  if (ret != 0) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
            "DiscoverAmdgpuDevices() failed.");
  }

  ss << __PRETTY_FUNCTION__ << " | about to sort by BDF..." << std::endl;
  LOG_DEBUG(ss);

  std::shared_ptr<amd::smi::Device> dev;
  // Sort index based on the BDF, collect BDF id firstly.
  std::vector<std::pair<uint64_t, std::shared_ptr<amd::smi::Device>>> dv_to_id;
  dv_to_id.reserve(devices_.size());
  for (uint32_t dv_ind = 0; dv_ind < devices_.size(); ++dv_ind) {
      dev = devices_[dv_ind];
      uint64_t bdfid = dev->bdfid();
      bdfid = bdfid & 0xFFFFFFFF0FFFFFFF;  // clear out partition id in bdf
      // NOTE: partition_id is not part of bdf (but is part of pci_id)
      // which is why it is removed in sorting
      dv_to_id.push_back({bdfid, dev});
  }
  ss << __PRETTY_FUNCTION__ << " Sort index based on BDF.";
  LOG_DEBUG(ss);

  // Stable sort to keep the order if bdf is equal.
  std::stable_sort(dv_to_id.begin(), dv_to_id.end(), []
  (const std::pair<uint64_t, std::shared_ptr<amd::smi::Device>>& p1,
    const std::pair<uint64_t, std::shared_ptr<amd::smi::Device>>& p2) {
        return p1.first < p2.first;
  });
  devices_.clear();
  for (uint32_t dv_ind = 0; dv_ind < dv_to_id.size(); ++dv_ind) {
    devices_.push_back(dv_to_id[dv_ind].second);
  }

  std::map<uint64_t, std::shared_ptr<KFDNode>> tmp_map;
  i_ret = DiscoverKFDNodes(&tmp_map);
  if (i_ret != 0) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
                 "Failed to initialize rocm_smi library (KFD node discovery).");
  }

  std::map<std::pair<uint32_t, uint32_t>, std::shared_ptr<IOLink>>
    io_link_map_tmp;
  i_ret = DiscoverIOLinks(&io_link_map_tmp);
  if (i_ret != 0) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
                 "Failed to initialize rocm_smi library (IO Links discovery).");
  }
  std::map<std::pair<uint32_t, uint32_t>, std::shared_ptr<IOLink>>::iterator it;
  for (it = io_link_map_tmp.begin(); it != io_link_map_tmp.end(); it++)
    io_link_map_[it->first] = it->second;


  // Remove any drm nodes that don't have  a corresponding readable kfd node.
  // kfd nodes will not be added if their properties file is not readable.
  auto dev_iter = devices_.begin();
  while (dev_iter != devices_.end()) {
    uint64_t bdfid = (*dev_iter)->bdfid();
    if (tmp_map.find(bdfid) == tmp_map.end()) {
      ss << __PRETTY_FUNCTION__ << " | removing device = "
         << (*dev_iter)->path() << "; bdfid = " << std::to_string(bdfid);
      dev_iter = devices_.erase(dev_iter);
      LOG_DEBUG(ss);
      continue;
    }
    dev_iter++;
  }

  // 1. construct kfd_node_map_ with gpu_id as key and *Device as value
  // 2. for each kfd node, write the corresponding dv_ind
  // 3. for each amdgpu device, write the corresponding gpu_id
  // 4. for each amdgpu device, attempt to store it's boot partition
  for (uint32_t dv_ind = 0; dv_ind < devices_.size(); ++dv_ind) {
    dev = devices_[dv_ind];
    uint64_t bdfid = dev->bdfid();
    assert(tmp_map.find(bdfid) != tmp_map.end());
    if (tmp_map.find(bdfid) == tmp_map.end()) {
      throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
                   "amdgpu device bdfid has no KFD matching node");
    }

    tmp_map[bdfid]->set_amdgpu_dev_index(dv_ind);
    dev_ind_to_node_ind_map_[dv_ind] = tmp_map[bdfid]->node_index();
    uint64_t gpu_id = tmp_map[bdfid]->gpu_id();
    dev->set_kfd_gpu_id(gpu_id);
    kfd_node_map_[gpu_id] = tmp_map[bdfid];

    // store each device boot partition state, if file doesn't exist
    dev->storeDevicePartitions(dv_ind);
  }

  // Assists displaying GPU information after device enumeration
  // Otherwise GPU related info will not be discoverable
  if (ROCmLogging::Logger::getInstance()->isLoggerEnabled()) {
    logSystemDetails();
  }

  // Leaving below to help debug temp file issues
  // displayAppTmpFilesContent();
  std::string amdGPUDeviceList = displayAllDevicePaths(devices_);
  ss << __PRETTY_FUNCTION__ << " | current device paths = " << amdGPUDeviceList;
  // std::cout << ss.str() << std::endl;
  LOG_DEBUG(ss);
}

void
RocmSMI::Cleanup() {
  devices_.clear();
  monitors_.clear();

  if (kfd_notif_evt_fh() >= 0) {
    int ret = close(kfd_notif_evt_fh());
    if (ret < 0) {
      throw amd::smi::rsmi_exception(RSMI_STATUS_FILE_ERROR,
                 "Failed to close kfd file handle on shutdown.");
    }
  }
}

RocmSMI::RocmSMI(uint64_t flags) : init_options_(flags),
                          kfd_notif_evt_fh_(-1), kfd_notif_evt_fh_refcnt_(0) {
}

RocmSMI::~RocmSMI() = default;

RocmSMI& RocmSMI::getInstance(uint64_t flags) {
  // Assume c++11 or greater. static objects will be created by only 1 thread
  // and creation will be thread-safe.
  static RocmSMI singleton(flags);
  return singleton;
}

static uint32_t GetEnvVarUInteger(const char *ev_str) {
#ifndef DEBUG
  (void)ev_str;
#else
  ev_str = getenv(ev_str);

  if (ev_str) {
    int ret = atoi(ev_str);
    assert(ret >= 0);
    return static_cast<uint32_t>(ret);
  }
#endif
  return 0;
}

// provides a way to get env variable detail in both debug & release
// helps enable full logging
// RSMI_LOGGING = 1, output to logs only
// RSMI_LOGGING = 2, output to console only
// RSMI_LOGGING = 3, output to logs and console
static uint32_t getRSMIEnvVar_LoggingEnabled(const char *ev_str) {
  uint32_t ret = 0;
  ev_str = getenv(ev_str);
  if (ev_str != nullptr) {
    int ev_ret = atoi(ev_str);
    ret = static_cast<uint32_t>(ev_ret);
  }
  return ret;
}

static inline std::unordered_set<uint32_t> GetEnvVarUIntegerSets(
  const char *ev_str) {
  std::unordered_set<uint32_t> returnSet;
#ifndef DEBUG
  (void)ev_str;
#else
  ev_str = getenv(ev_str);
  if(ev_str == nullptr) { return returnSet; }
  std::string stringEnv = ev_str;

  if (!stringEnv.empty()) {
    // parse out values by commas
    std::string parsedVal;
    std::istringstream ev_str_ss(stringEnv);

    while (std::getline(ev_str_ss, parsedVal, ',')) {
      int parsedInt = std::stoi(parsedVal);
      assert(parsedInt >= 0);
      uint32_t parsedUInt = static_cast<uint32_t>(parsedInt);
      returnSet.insert(parsedUInt);
    }
  }
#endif
  return returnSet;
}

// Get and store env. variables in this method
void RocmSMI::GetEnvVariables(void) {
  env_vars_.logging_on = getRSMIEnvVar_LoggingEnabled("RSMI_LOGGING");
#ifndef DEBUG
  (void)GetEnvVarUInteger(nullptr);  // This is to quiet release build warning.
  env_vars_.debug_output_bitfield = 0;
  env_vars_.path_DRM_root_override = nullptr;
  env_vars_.path_HWMon_root_override = nullptr;
  env_vars_.path_power_root_override = nullptr;
  env_vars_.debug_inf_loop = 0;
  env_vars_.enum_overrides.clear();
#else
  env_vars_.debug_output_bitfield = GetEnvVarUInteger("RSMI_DEBUG_BITFIELD");
  env_vars_.path_DRM_root_override   = getenv("RSMI_DEBUG_DRM_ROOT_OVERRIDE");
  env_vars_.path_HWMon_root_override = getenv("RSMI_DEBUG_HWMON_ROOT_OVERRIDE");
  env_vars_.path_power_root_override = getenv("RSMI_DEBUG_PP_ROOT_OVERRIDE");
  env_vars_.debug_inf_loop = GetEnvVarUInteger("RSMI_DEBUG_INFINITE_LOOP");
  env_vars_.enum_overrides = GetEnvVarUIntegerSets("RSMI_DEBUG_ENUM_OVERRIDE");
#endif
}

const RocmSMI_env_vars& RocmSMI::getEnv(void) {
  return env_vars_;
}

bool RocmSMI::isLoggingOn(void) {
  bool isLoggingOn = false;
  GetEnvVariables();
  if (this->env_vars_.logging_on > 0
      && this->env_vars_.logging_on <= 3) {
    isLoggingOn = true;
  }
  return isLoggingOn;
}

uint32_t RocmSMI::getLogSetting() {
  return this->env_vars_.logging_on;
}

void RocmSMI::debugRSMIEnvVarInfo(void) {
  std::cout << __PRETTY_FUNCTION__
            << RocmSMI::getInstance().getRSMIEnvVarInfo();
}

std::string RocmSMI::getRSMIEnvVarInfo(void) {
  std::ostringstream ss;
  ss << "\n\tRSMI_DEBUG_BITFIELD = "
     << ((env_vars_.debug_output_bitfield == 0) ? "<undefined>"
          : std::to_string(env_vars_.debug_output_bitfield))
     << std::endl;
  ss << "\tRSMI_DEBUG_DRM_ROOT_OVERRIDE = "
     << ((env_vars_.path_DRM_root_override == nullptr)
         ? "<undefined>" : env_vars_.path_DRM_root_override)
     << std::endl;
  ss << "\tRSMI_DEBUG_HWMON_ROOT_OVERRIDE = "
     << ((env_vars_.path_HWMon_root_override == nullptr)
          ? "<undefined>" : env_vars_.path_HWMon_root_override)
     << std::endl;
  ss << "\tRSMI_DEBUG_PP_ROOT_OVERRIDE = "
     << ((env_vars_.path_power_root_override == nullptr)
          ? "<undefined>" : env_vars_.path_power_root_override)
     << std::endl;
  ss << "\tRSMI_DEBUG_INFINITE_LOOP = "
     << ((env_vars_.debug_inf_loop == 0) ? "<undefined>"
          : std::to_string(env_vars_.debug_inf_loop))
     << std::endl;
  ss << "\tRSMI_LOGGING = "
            << getLogSetting() << std::endl;
  bool isLoggingOn = RocmSMI::isLoggingOn() ? true : false;
  ss << "\tRSMI_LOGGING (are logs on) = "
            << (isLoggingOn ? "TRUE" : "FALSE") << std::endl;
  ss << "\tRSMI_DEBUG_ENUM_OVERRIDE = {";
  if (env_vars_.enum_overrides.empty()) {
    ss << "}" << std::endl;
    return ss.str();
  }
  for (auto it=env_vars_.enum_overrides.begin();
       it != env_vars_.enum_overrides.end(); ++it) {
    DevInfoTypes type = static_cast<DevInfoTypes>(*it);
    ss << (std::to_string(*it) + " (" + Device::get_type_string(type) + ")");
    auto temp_it = it;
    if (++temp_it != env_vars_.enum_overrides.end()) {
      ss << ", ";
    }
  }
  ss << "}" << std::endl;
  return ss.str();
}

std::shared_ptr<Monitor>
RocmSMI::FindMonitor(std::string monitor_path) {
  std::string tmp;
  std::string err_msg;
  std::string mon_name;
  std::shared_ptr<Monitor> m;

  if (!FileExists(monitor_path.c_str())) {
    return nullptr;
  }

  auto mon_dir = opendir(monitor_path.c_str());

  if (mon_dir == nullptr) {
    return nullptr;
  }
  auto dentry = readdir(mon_dir);

  while (dentry != nullptr) {
    if (dentry->d_name[0] == '.') {
      dentry = readdir(mon_dir);
      continue;
    }

    mon_name = monitor_path;
    mon_name += "/";
    mon_name += dentry->d_name;
    tmp = mon_name + "/name";

    if (FileExists(tmp.c_str())) {
      std::ifstream fs;
      fs.open(tmp);

      if (!fs.is_open()) {
        err_msg = "Failed to open monitor file ";
        err_msg += tmp;
        err_msg += ".";
        perror(err_msg.c_str());
        return nullptr;
      }
      std::string mon_type;
      fs >> mon_type;
      fs.close();

      if (amd_monitor_types_.find(mon_type) != amd_monitor_types_.end()) {
        m = std::make_shared<Monitor>(mon_name, &env_vars_);
        m->setTempSensorLabelMap();
        m->setVoltSensorLabelMap();
        break;
      }
    }
    dentry = readdir(mon_dir);
  }

  if (closedir(mon_dir)) {
    err_msg = "Failed to close monitor directory ";
    err_msg += kPathHWMonRoot;
    err_msg += ".";
    perror(err_msg.c_str());
    return nullptr;
  }

  return m;
}

void RocmSMI::AddToDeviceList(std::string dev_name, uint64_t bdfid) {
  static const int BYTE = 8;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto dev_path = std::string(kPathDRMRoot);
  dev_path += "/";
  dev_path += dev_name;

  auto dev = std::make_shared<Device>(dev_path, &env_vars_);

  std::shared_ptr<Monitor> m = FindMonitor(dev_path + "/device/hwmon");
  dev->set_monitor(m);

  const std::string& d_name = dev_name;
  uint32_t card_indx = GetDeviceIndex(d_name);
  uint32_t drmRenderMinor = GetDrmRenderMinor(dev_path);
  dev->set_drm_render_minor(drmRenderMinor);
  dev->set_card_index(card_indx);
  GetSupportedEventGroups(card_indx, dev->supported_event_groups());
  if (bdfid != 0) {
    dev->set_bdfid(bdfid);
  }

  devices_.push_back(dev);
  ss << __PRETTY_FUNCTION__
     << " | Adding to device list dev_name = " << dev_name << "\n"
     << " | path = " << dev_path << "\n"
     << " | dName = " << d_name << "\n"
     << " | bdfid = " << (bdfid == UINT64_MAX ?
      "N/A" : print_int_as_hex(bdfid, true, 2*BYTE)) << "\n"
     << " | card index = " << std::to_string(card_indx) << "\n"
     << " | drmRenderMinor = " << std::to_string(drmRenderMinor) << "\n"
     << " | supported_event_groups = " << dev->supported_event_groups() << "\n";
  // std::cout << ss.str() << std::endl;
  LOG_DEBUG(ss);
}

// AddToDeviceList2 is used to add a device to the device list.
// [precondition] a. Iterate through KFD to find all accessible devices.
// [precondition] b. Provide BDFID of the device & the device path (card or render path)
// 1. Provide to function:
//    [optional; Will populate] rsmi_device_enumeration_t->card_index
//    [optional; Will populate
//     if card or render path provided] rsmi_device_enumeration_t->dev_name
//    [optional; Will populate] rsmi_device_enumeration_t->drm_render_path
//    [optional; Will populate] rsmi_device_enumeration_t->drm_card_path
//    [optional; Will populate] rsmi_device_enumeration_t->drm_render_minor
//    [Required] rsmi_device_enumeration_t->bdfid
rsmi_status_t RocmSMI::AddToDeviceList2(RocmSMI::rsmi_device_enumeration_t device) {
  static const int BYTE = 8;
  std::ostringstream ss;

  ss << __PRETTY_FUNCTION__ << " | ======= start ======="
     << "\n | card index = [" << std::to_string(device.card_index) << "]\n"
     << " | dev_name = [" << device.dev_name << "]\n"
     << " | drm_render_path = [" << device.drm_render_path << "]\n"
     << " | drm_card_path = [" << device.drm_card_path << "]\n"
     << " | drm_render_minor = [" << std::to_string(device.drm_render_minor)
     << "]\n | bdfid (value) = [" << (device.bdfid == UINT64_MAX ?
       "N/A" : print_int_as_hex(device.bdfid, true, 4*BYTE)) << "]\n"
     << " | bdfid (str) = ["
     << std::hex << std::setfill('0') << std::setw(4)
     << ((device.bdfid >> 32) & static_cast<uint64_t>(0xFFFFFFFF)) << ":"
     << std::hex << std::setfill('0') << std::setw(2) << ((device.bdfid >> 8)
                                                          & static_cast<uint64_t>(0xFF)) << ":"
     << std::hex << std::setfill('0') << std::setw(2) << ((device.bdfid >> 3)
                                                          & static_cast<uint64_t>(0x1F)) << "."
     << std::hex << std::setfill('0') << std::setw(1) << +(device.bdfid
                                                          & static_cast<uint64_t>(0x7)) << "]\n";
  // std::cout << ss.str() << std::endl;
  LOG_TRACE(ss);
  auto dev_path = std::string(kPathDRMRoot);

  if (device.dev_name.empty()) {
    ss << __PRETTY_FUNCTION__ << " | dev_name is empty";
    // std::cout << ss.str() << std::endl;
    LOG_DEBUG(ss);

    dev_path += "/";
    dev_path += ("renderD" + std::to_string(device.drm_render_minor));
    uint32_t card_num = GetCard(dev_path);
    device.dev_name = "card" + std::to_string(card_num);
    device.drm_render_path = dev_path;
    device.drm_card_path = std::string(kPathDRMRoot) + "/card" +
      std::to_string(card_num);
    device.card_index = card_num;
  }

  auto dev = std::make_shared<Device>(dev_path, &env_vars_);

  std::shared_ptr<Monitor> m = FindMonitor(dev_path + "/device/hwmon");
  dev->set_monitor(m);

  const std::string& d_name = device.dev_name;
  uint32_t card_indx = GetDeviceIndex(d_name);
  uint32_t drmRenderMinor = GetDrmRenderMinor(dev_path);
  dev->set_drm_render_minor(drmRenderMinor);
  dev->set_card_index(card_indx);
  GetSupportedEventGroups(card_indx, dev->supported_event_groups());
  if (device.bdfid != 0) {
    dev->set_bdfid(device.bdfid);
  }

  devices_.push_back(dev);
  ss << __PRETTY_FUNCTION__
     << " | Adding to device list dev_name = " << device.dev_name << "\n"
     << " | path = " << dev_path << "\n"
     << " | dName = " << d_name << "\n"
     << " | bdfid = " << (device.bdfid == UINT64_MAX ?
      "N/A" : print_int_as_hex(device.bdfid, true, 8*BYTE)) << "\n"
     << " | card index = " << std::to_string(card_indx) << "\n"
     << " | drmRenderMinor = " << std::to_string(drmRenderMinor) << "\n"
     << " | supported_event_groups = " << dev->supported_event_groups() << "\n";
  ss << " | ======= rsmi_device_enumeration_t details =======\n"
     << " | card index = [" << std::to_string(device.card_index) << "]\n"
     << " | dev_name = [" << device.dev_name << "]\n"
     << " | drm_render_path = [" << device.drm_render_path << "]\n"
     << " | drm_card_path = [" << device.drm_card_path << "]\n"
     << " | drm_render_minor = [" << std::to_string(device.drm_render_minor)
     << "]\n | bdfid (value) = [" << (device.bdfid == UINT64_MAX ?
       "N/A" : print_int_as_hex(device.bdfid, true, 8*BYTE)) << "]\n"
     << " | bdfid (str) = ["
     << std::hex << std::setfill('0') << std::setw(4)
     << ((device.bdfid >> 32) & static_cast<uint64_t>(0xFFFFFFFF)) << ":"
     << std::hex << std::setfill('0') << std::setw(2) << ((device.bdfid >> 8)
                                                          & static_cast<uint64_t>(0xFF)) << ":"
     << std::hex << std::setfill('0') << std::setw(2) << ((device.bdfid >> 3)
                                                          & static_cast<uint64_t>(0x1F)) << "."
     << std::hex << std::setfill('0') << std::setw(1) << +(device.bdfid
                                                          & static_cast<uint64_t>(0x7)) << "]\n"
     << " | END";
  // std::cout << ss.str() << std::endl;
  LOG_DEBUG(ss);
  return RSMI_STATUS_SUCCESS;
}

static const uint32_t kAmdGpuId = 0x1002;

[[maybe_unused]] static bool isAMDGPU(std::string dev_path) {
  bool isAmdGpu = false;
  std::ostringstream ss;
  std::string vend_path = dev_path + "/device/vendor";
  if (!FileExists(vend_path.c_str())) {
    ss << __PRETTY_FUNCTION__ << " | device_path = " << dev_path
       << " is an amdgpu device - " << (isAmdGpu ? "TRUE": " FALSE");
    LOG_DEBUG(ss);
    return isAmdGpu;
  }

  std::ifstream fs;
  fs.open(vend_path);

  if (!fs.is_open()) {
    ss << __PRETTY_FUNCTION__ << " | device_path = " << dev_path
       << " is an amdgpu device - " << (isAmdGpu ? "TRUE": " FALSE");
    LOG_DEBUG(ss);
    return isAmdGpu;
  }

  uint32_t vendor_id;

  fs >> std::hex >> vendor_id;

  fs.close();

  if (vendor_id == kAmdGpuId) {
    isAmdGpu = true;
  }
  ss << __PRETTY_FUNCTION__ << " | device_path = " << dev_path
     << " is an amdgpu device - " << (isAmdGpu ? "TRUE": " FALSE");
  LOG_DEBUG(ss);
  return isAmdGpu;
}

uint32_t GetLargestNodeNumber(const std::string& path = "/sys/class/kfd/kfd/topology/nodes/") {
  std::ostringstream ss;
  uint32_t largest_node_number = 0;

  // Open the directory
  DIR* dir = opendir(path.c_str());
  if (!dir) {
      // Return UINT32_MAX on error
      ss << __PRETTY_FUNCTION__ << " | Failed to open directory: " << path
         << " | errno = " << errno
         << " | error = " << strerror(errno);
      // std::cout << ss.str() << std::endl;
      LOG_ERROR(ss);
      return UINT32_MAX;
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
      // Skip "." and ".."
      if (entry->d_name[0] == '.') {
          continue;
      }

      // Check if the directory name is a number
      std::string dir_name(entry->d_name);
      if (std::all_of(dir_name.begin(), dir_name.end(), ::isdigit)) {
          uint32_t node_number = static_cast<uint32_t>(std::stoul(dir_name));
          largest_node_number = std::max(largest_node_number, node_number);
      }
  }

  if (closedir(dir)) {
      // Return UINT32_MAX on error
      ss << __PRETTY_FUNCTION__ << " | Failed to close directory: " << path
         << " | errno = " << errno
         << " | error = " << strerror(errno);
      // std::cout << ss.str() << std::endl;
      LOG_ERROR(ss);
      return UINT32_MAX;
  }

  return largest_node_number;
}

uint32_t RocmSMI::DiscoverAmdgpuDevices(void) {
  std::string err_msg;
  std::ostringstream ss;

  // If this gets called more than once, clear previous findings.
  devices_.clear();
  monitors_.clear();

  uint32_t max_nodes = GetLargestNodeNumber();
  ss << __PRETTY_FUNCTION__ << " | Discovered a potential of "
     << std::to_string(max_nodes) << " kfd nodes";
  // std::cout << ss.str() << std::endl;
  LOG_DEBUG(ss);
  if (max_nodes == UINT32_MAX) {
    ss << __PRETTY_FUNCTION__ << " | Failed to get largest node number";
    // std::cout << ss.str() << std::endl;
    LOG_ERROR(ss);
    return 1;
  }
  // Iterate through all nodes
  // and read all properties
  // under /sys/class/kfd/kfd/topology/nodes/
  // and add to systemNodes vector

  struct systemNode {
    uint32_t s_node_id = 0;
    uint64_t s_gpu_id = 0;
    uint64_t s_unique_id = 0;
    uint64_t s_location_id = 0;
    uint64_t s_bdf = 0;
    uint64_t s_domain = 0;
    uint8_t  s_bus = 0;
    uint8_t  s_device = 0;
    uint8_t  s_function = 0;
    uint8_t  s_partition_id = 0;
    uint32_t s_drm_render_minor = 0;
    uint64_t padding = 0;  // padding added in case new changes in future
  };
  std::multimap<uint64_t, systemNode> allSystemNodes;
  std::set<uint32_t> gpuNodeIdsFound;
  std::vector<systemNode> systemNodes;
  uint32_t node_id = 0;
  static const int BYTE = 8;
  while (node_id <= max_nodes) {
    ss << __PRETTY_FUNCTION__ << " | node_id = " << std::to_string(node_id);
    // std::cout << ss.str() << std::endl;
    LOG_DEBUG(ss);
    uint64_t gpu_id = 0, unique_id = 0, location_id = 0, domain = 0, render_d = 0;
    int ret_gpu_id = get_gpu_id(node_id, &gpu_id);
    int ret_unique_id = read_node_properties(node_id, "unique_id", &unique_id);
    int ret_loc_id =
      read_node_properties(node_id, "location_id", &location_id);
    int ret_domain = read_node_properties(node_id, "domain", &domain);
    int ret_renderd = read_node_properties(node_id, "drm_render_minor", &render_d);
    bool isANode = (ret_gpu_id == 0 &&
      (ret_domain == 0 && ret_loc_id == 0 && ret_renderd == 0));
    ss << __PRETTY_FUNCTION__ << " | isAGpuNode: "
       << (isANode ? "TRUE" : "FALSE") << "; is_vm_guest(): "
       << (is_vm_guest() ? "TRUE" : "FALSE")
       << "\nret_gpu_id: " << ret_gpu_id
       << "; ret_domain: " << ret_domain
       << "; ret_loc_id: " << ret_loc_id
       << "; ret_unique_id: " << ret_unique_id
       << "\nret_renderd: " << ret_renderd
       << "\n[node_id = " << print_unsigned_hex_and_int(node_id) << "\n"
       << "; gpu_id = " << print_unsigned_hex_and_int(gpu_id) << "\n"
       << "; unique_id = " << print_unsigned_hex_and_int(unique_id) << "\n"
       << "; location_id = " << print_unsigned_hex_and_int(location_id) << "\n"
       << "; domain = " << print_unsigned_hex_and_int(domain) << "\n"
       << "; drm_render_minor = " << print_unsigned_hex_and_int(render_d)
       << "]\n";
    LOG_DEBUG(ss);
    if (isANode || (is_vm_guest() && ret_gpu_id == 0)) {
        // Do not try to build a node if one of these fields
        // do not exist in KFD (0 as values okay)
      systemNode myNode;
      myNode.s_node_id = node_id;
      myNode.s_gpu_id = gpu_id;
      myNode.s_unique_id = unique_id;
      myNode.s_location_id = location_id;
      myNode.s_domain = domain & 0xFFFFFFFF;
      myNode.s_bdf = (myNode.s_domain << 32) | (myNode.s_location_id);
      myNode.s_bus = ((myNode.s_location_id >> 8) & 0xFF);
      myNode.s_device = ((myNode.s_location_id >> 3) & 0x1F);
      myNode.s_function = myNode.s_location_id & 0x7;
      myNode.s_partition_id = ((myNode.s_location_id >> 28) & 0xF);
      myNode.s_drm_render_minor = static_cast<uint32_t>((ret_renderd == 0) ? render_d : 0);
      if (gpu_id != 0) {  // only add gpu nodes, 0 = CPU
        auto ret = gpuNodeIdsFound.insert(node_id);
        if (ret.second != false) {
          // only print out nodes which do not already exist
          ss << __PRETTY_FUNCTION__ << " | isAGpuNode: "
             << (isANode ? "TRUE" : "FALSE") << "; is_vm_guest(): "
             << (is_vm_guest() ? "TRUE" : "FALSE")
             << "\nret_gpu_id: " << ret_gpu_id
             << "; ret_domain: " << ret_domain
             << "; ret_loc_id: " << ret_loc_id
             << "; ret_unique_id: " << ret_unique_id
             << "\n[node_id = " << print_unsigned_hex_and_int(node_id) << "\n"
             << "; gpu_id = " << print_unsigned_hex_and_int(gpu_id) << "\n"
             << "; unique_id = " << print_unsigned_hex_and_int(unique_id) << "\n"
             << "; location_id = " << print_unsigned_hex_and_int(location_id) << "\n"
             << "; domain = " << print_unsigned_hex_and_int(domain) << "\n"
             << "; bus = " << print_unsigned_hex_and_int(myNode.s_bus) << "\n"
             << "; device = " << print_unsigned_hex_and_int(myNode.s_device) << "\n"
             << "; function = " << print_unsigned_hex_and_int(myNode.s_function) << "\n"
             << "; partition_id = " << print_unsigned_hex_and_int(myNode.s_partition_id) << "\n"
             << "; bdf = " << print_unsigned_hex_and_int(myNode.s_bdf) << "\n"
             << "; drm_render_minor = " << print_unsigned_hex_and_int(myNode.s_drm_render_minor)
             << "]\n";
          LOG_DEBUG(ss);
        }
        systemNodes.push_back(myNode);
      }
    }
    node_id++;
  }

  ss << __PRETTY_FUNCTION__ << " | Ordered system nodes found = {";

  for (auto i : systemNodes) {
    ss << "\n[node_id = " << std::to_string(i.s_node_id) << "\n"
       << "; gpu_id = " << std::to_string(i.s_gpu_id) << "\n"
       << "; unique_id = " << std::to_string(i.s_unique_id) << "\n"
       << "; location_id = " << std::to_string(i.s_location_id) << "\n"
       << "; bdf = " << print_int_as_hex(i.s_bdf) << "\n"
       << "; domain = " << print_int_as_hex(i.s_domain, true, 2*BYTE) << "\n"
       << "; bus = " << print_int_as_hex(i.s_bus, true, BYTE) << "\n"
       << "; device = " << print_int_as_hex(i.s_device, true, BYTE) << "\n"
       << "; function = " << std::to_string(i.s_function) << "\n"
       << "; partition_id = " << std::to_string(i.s_partition_id) << "\n"
       << "; drm_render_minor = " << std::to_string(i.s_drm_render_minor)
       << "], \n";
    rsmi_device_enumeration_t rsmi_device;
    rsmi_device.dev_name = "";
    rsmi_device.bdfid = i.s_bdf;
    rsmi_device.drm_render_minor = i.s_drm_render_minor;
    AddToDeviceList2(rsmi_device);
  }
  ss << "}";
  // std::cout << ss.str() << std::endl;
  LOG_DEBUG(ss);
  return 0;
}


// Since these sysfs files require sudo access, we won't discover them
// with rsmi_init() (and thus always require the user to use "sudo".
// Instead, we will discover() all the power monitors the first time
// they are needed and then check for previous discovery on each subsequent
// call.
int RocmSMI::DiscoverAMDPowerMonitors(bool force_update) {
  if (force_update) {
    power_mons_.clear();
  }

  if (!power_mons_.empty()) {
    return 0;
  }

  errno = 0;
  auto dri_dir = opendir(kPathPowerRoot);

  if (dri_dir == nullptr) {
    return errno;
  }
  auto dentry = readdir(dri_dir);

  std::string mon_name;
  std::string tmp;

  while (dentry != nullptr) {
    if (dentry->d_name[0] == '.') {
      dentry = readdir(dri_dir);
      continue;
    }

    mon_name = kPathPowerRoot;
    mon_name += "/";
    mon_name += dentry->d_name;
    tmp = mon_name + "/amdgpu_pm_info";

    if (FileExists(tmp.c_str())) {
      std::shared_ptr<PowerMon> mon =
                std::make_shared<PowerMon>(mon_name, &env_vars_);
      power_mons_.push_back(mon);
      mon->set_dev_index(GetDeviceIndex(dentry->d_name));
    }
    dentry = readdir(dri_dir);
  }

  errno = 0;
  if (closedir(dri_dir)) {
    power_mons_.clear();
    return errno;
  }

  for (const auto& m : power_mons_) {
    for (const auto& d : devices_) {
      if (m->dev_index() == d->index()) {
        d->set_power_monitor(m);
        break;
      }
    }
  }

  return 0;
}

uint32_t RocmSMI::IterateSMIDevices(
     std::function<uint32_t(std::shared_ptr<Device>&, void *)> func, void *p) {
  if (func == nullptr) {
    return 1;
  }

  auto d = devices_.begin();
  uint32_t ret;

  while (d != devices_.end()) {
    ret = func(*d, p);
    if (ret != 0) {
      return ret;
    }
    ++d;
  }
  return 0;
}

int RocmSMI::get_node_index(uint32_t dv_ind, uint32_t *node_ind) {
  if (dev_ind_to_node_ind_map_.find(dv_ind) == dev_ind_to_node_ind_map_.end()) {
    return EINVAL;
  }
  *node_ind = dev_ind_to_node_ind_map_[dv_ind];
  return 0;
}

int RocmSMI::get_io_link_weight(uint32_t node_from, uint32_t node_to,
                                uint64_t *weight) {
  assert(weight != nullptr);
  if (weight == nullptr) {
    return EINVAL;
  }
  if (io_link_map_.find(std::make_pair(node_from, node_to)) ==
      io_link_map_.end()) {
    return EINVAL;
  }
  *weight = io_link_map_[std::make_pair(node_from, node_to)]->weight();
  return 0;
}

}  // namespace smi
}  // namespace amd
