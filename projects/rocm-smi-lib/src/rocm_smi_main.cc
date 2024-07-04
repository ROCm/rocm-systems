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

static const char *kDeviceNamePrefix = "card";

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
static uint32_t ConstructBDFID(std::string path, uint64_t *bdfid) {
  assert(bdfid != nullptr);
  const unsigned int MAX_BDF_LENGTH = 512;
  char tpath[MAX_BDF_LENGTH] = {'\0'};
  ssize_t ret;
  memset(tpath,0,MAX_BDF_LENGTH);

  ret = readlink(path.c_str(), tpath, MAX_BDF_LENGTH);

  assert(ret > 0);
  assert(ret < MAX_BDF_LENGTH);

  if (ret <= 0 || ret >= MAX_BDF_LENGTH) {
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
      return 0;
    }
    end_i = slash_i - 1;
  }

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

  uint64_t bdfid;
  for (auto & device : devices_) {
    if (ConstructBDFID(device->path(), &bdfid) != 0) {
      std::cerr << "Failed to construct BDFID." << std::endl;
      ret = 1;
    } else if (device->bdfid() != UINT64_MAX && device->bdfid() != bdfid) {
      // handles secondary partitions - compute partition feature nodes
      ss << __PRETTY_FUNCTION__
         << " | [before] device->path() = " << device->path()
         << "\n | bdfid = " << bdfid
         << "\n | device->bdfid() = " << device->bdfid()
         << " (" << print_int_as_hex(device->bdfid()) << ")"
         << "\n | (xgmi node) setting to setting "
         << "device->set_bdfid(device->bdfid())";
      LOG_TRACE(ss);
      device->set_bdfid(device->bdfid());
    } else {
      // legacy & pcie card updates
      ss << __PRETTY_FUNCTION__
         << " | [before] device->path() = " << device->path()
         << "\n | bdfid = " << bdfid
         << "\n | device->bdfid() = " << device->bdfid()
         << " (" << print_int_as_hex(device->bdfid()) << ")"
         << "\n | (legacy/pcie card) setting device->set_bdfid(bdfid)";
      LOG_TRACE(ss);
      device->set_bdfid(bdfid);
    }
      ss << __PRETTY_FUNCTION__
         << " | [after] device->path() = " << device->path()
         << "\n | bdfid = " << bdfid
         << "\n | device->bdfid() = " << device->bdfid()
         << " (" << print_int_as_hex(device->bdfid()) << ")"
         << "\n | final update: device->bdfid() holds correct device bdf";
      LOG_TRACE(ss);
  }

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
    ss << (std::to_string(*it) + " (" + Device::devInfoTypesStrings.at(type) + ")");
    auto temp_it = it;
    if(++temp_it != env_vars_.enum_overrides.end()) {
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
  dev->set_drm_render_minor(GetDrmRenderMinor(dev_path));
  dev->set_card_index(card_indx);
  GetSupportedEventGroups(card_indx, dev->supported_event_groups());
  if (bdfid != 0) {
    dev->set_bdfid(bdfid);
  }

  devices_.push_back(dev);
  ss << __PRETTY_FUNCTION__
     << " | Adding to device list dev_name = " << dev_name
     << " | path = " << dev_path
     << " | bdfid = " << bdfid
     << " | card index = " << std::to_string(card_indx) << " | ";
  LOG_DEBUG(ss);
}

static const uint32_t kAmdGpuId = 0x1002;

static bool isAMDGPU(std::string dev_path) {
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

uint32_t RocmSMI::DiscoverAmdgpuDevices(void) {
  std::string err_msg;
  uint32_t count = 0;
  int32_t cardId = 0;
  int32_t max_cardId = -1;
  std::ostringstream ss;

  // If this gets called more than once, clear previous findings.
  devices_.clear();
  monitors_.clear();

  auto drm_dir = opendir(kPathDRMRoot);
  if (drm_dir == nullptr) {
    err_msg = "Failed to open drm root directory ";
    err_msg += kPathDRMRoot;
    err_msg += ".";
    perror(err_msg.c_str());
    return 1;
  }

  auto dentry = readdir(drm_dir);

  while (dentry != nullptr) {
    if (memcmp(dentry->d_name, kDeviceNamePrefix, strlen(kDeviceNamePrefix))
                                                                       == 0) {
        if ((strcmp(dentry->d_name, ".") == 0) ||
           (strcmp(dentry->d_name, "..") == 0))
           continue;
        sscanf(&dentry->d_name[strlen(kDeviceNamePrefix)], "%d", &cardId);
        if (cardId > max_cardId)
          max_cardId = cardId;
        count++;
    }
    dentry = readdir(drm_dir);
  }
  ss << __PRETTY_FUNCTION__ << " | Discovered a potential of "
     << std::to_string(count) << " cards" << " | ";
  LOG_DEBUG(ss);

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
    uint64_t padding = 0;  // padding added in case new changes in future
  };
  // allSystemNodes[key = unique_id] => {node_id, gpu_id, unique_id,
  //                                     location_id, bdf, domain, bus, device,
  //                                     partition_id}
  std::multimap<uint64_t, systemNode> allSystemNodes;
  uint32_t node_id = 0;
  static const int BYTE = 8;
  while (true) {
    uint64_t gpu_id = 0, unique_id = 0, location_id = 0, domain = 0;
    int ret_gpu_id = get_gpu_id(node_id, &gpu_id);
    int ret_unique_id = read_node_properties(node_id, "unique_id", &unique_id);
    int ret_loc_id =
      read_node_properties(node_id, "location_id", &location_id);
    int ret_domain =
      read_node_properties(node_id, "domain", &domain);
    if (ret_gpu_id == 0 &&
      ~(ret_unique_id != 0 || ret_loc_id != 0 || ret_unique_id != 0)) {
        // Do not try to build a node if one of these fields
        // do not exist in KFD (0 as values okay)
      systemNode myNode;
      myNode.s_node_id = node_id;
      myNode.s_gpu_id = gpu_id;
      myNode.s_unique_id = unique_id;
      myNode.s_location_id = location_id;
      myNode.s_domain = domain & 0xFFFFFFFF;
      myNode.s_bdf = (myNode.s_domain << 32) | (myNode.s_location_id);
      myNode.s_location_id = myNode.s_bdf;
      myNode.s_bdf |= ((domain & 0xFFFFFFFF) << 32);
      myNode.s_location_id = myNode.s_bdf;
      myNode.s_domain = myNode.s_location_id >> 32;
      myNode.s_bus = ((myNode.s_location_id >> 8) & 0xFF);
      myNode.s_device = ((myNode.s_location_id >> 3) & 0x1F);
      myNode.s_function = myNode.s_location_id & 0x7;
      myNode.s_partition_id = ((myNode.s_location_id >> 28) & 0xF);
      if (gpu_id != 0) {  // only add gpu nodes, 0 = CPU
        allSystemNodes.emplace(unique_id, myNode);
      }
    } else {
      break;
    }
    node_id++;
  }

  ss << __PRETTY_FUNCTION__ << " | Ordered system nodes found = {";
  for (auto i : allSystemNodes) {
    ss << "\n[node_id = " << std::to_string(i.second.s_node_id)
       << "; gpu_id = " << std::to_string(i.second.s_gpu_id)
       << "; unique_id = " << std::to_string(i.second.s_unique_id)
       << "; location_id = " << std::to_string(i.second.s_location_id)
       << "; bdf = " << print_int_as_hex(i.second.s_bdf)
       << "; domain = " << print_int_as_hex(i.second.s_domain, true, 2*BYTE)
       << "; bus = " << print_int_as_hex(i.second.s_bus, true, BYTE)
       << "; device = " << print_int_as_hex(i.second.s_device, true, BYTE)
       << "; function = " << std::to_string(i.second.s_function)
       << "; partition_id = " << std::to_string(i.second.s_partition_id)
       << "], ";
  }
  ss << "}";
  LOG_DEBUG(ss);

  uint32_t cardAdded = 0;
  // Discover all root cards & gpu partitions associated with each
  for (uint32_t cardId = 0; cardId <= max_cardId; cardId++) {
    std::string path = kPathDRMRoot;
    path += "/card";
    path += std::to_string(cardId);
    uint64_t primary_unique_id = 0;
    uint64_t device_uuid = 0;
    bool doesDeviceSupportPartitions = false;
    // get current partition
    int kSize = 256;
    char computePartition[kSize];
    std::string strCompPartition = "UNKNOWN";
    uint32_t numMonDevices = 0;
    rsmi_num_monitor_devices(&numMonDevices);

    // each identified gpu card node is a primary node for
    // potential matching unique ids
    if (isAMDGPU(path) ||
        (init_options_ & RSMI_INIT_FLAG_ALL_GPUS)) {
      std::string d_name = "card";
      d_name += std::to_string(cardId);
      uint32_t numMonDevices = 0;
      rsmi_num_monitor_devices(&numMonDevices);
      if (rsmi_dev_compute_partition_get(cardAdded, computePartition, kSize)
          == RSMI_STATUS_SUCCESS) {
        strCompPartition = computePartition;
        doesDeviceSupportPartitions = true;
      }
      rsmi_status_t ret_unique_id =
          rsmi_dev_unique_id_get(cardAdded, &device_uuid);
      auto temp_numb_nodes = allSystemNodes.count(device_uuid);
      auto primaryBdfId =
          allSystemNodes.lower_bound(device_uuid)->second.s_location_id;
      auto i = allSystemNodes.lower_bound(device_uuid);
      if (doesDeviceSupportPartitions && temp_numb_nodes > 1
          && ret_unique_id == RSMI_STATUS_SUCCESS) {
        // helps identify xgmi nodes (secondary nodes) easier
        ss << __PRETTY_FUNCTION__ << " | secondary node add ; "
           << " BDF = " << std::to_string(primaryBdfId)
           << " (" << print_int_as_hex(primaryBdfId) << ")";
        LOG_DEBUG(ss);
        if (doesDeviceSupportPartitions && strCompPartition != "SPX"
           && i->second.s_partition_id == 0) {
          i->second.s_partition_id = i->second.s_function;
          ss << __PRETTY_FUNCTION__ << " | (secondary node add) fall back - "
            << "detected !SPX && partition_id == 0"
            << "; function = " << std::to_string(i->second.s_function)
            << "; partition_id = " << std::to_string(i->second.s_partition_id);
          LOG_DEBUG(ss);
        }
        ss << __PRETTY_FUNCTION__
           << " | (secondary node add) B4 AddToDeviceList() -->"
           << "\n[node_id = " << std::to_string(i->second.s_node_id)
           << "; gpu_id = " << std::to_string(i->second.s_gpu_id)
           << "; unique_id = " << std::to_string(i->second.s_unique_id)
           << "; location_id = " << std::to_string(i->second.s_location_id)
           << "; bdf = " << print_int_as_hex(i->second.s_bdf)
           << "; domain = " << print_int_as_hex(i->second.s_domain, true, 2*BYTE)
           << "; bus = " << print_int_as_hex(i->second.s_bus, true, BYTE)
           << "; device = " << print_int_as_hex(i->second.s_device, true, BYTE)
           << "; function = " << std::to_string(i->second.s_function)
           << "; partition_id = " << std::to_string(i->second.s_partition_id)
           << "], ";
        LOG_DEBUG(ss);
        AddToDeviceList(d_name, primaryBdfId);
      } else {
        ss << __PRETTY_FUNCTION__ << " | primary node add ; "
           << " BDF = " << std::to_string(UINT64_MAX);
        if (doesDeviceSupportPartitions && strCompPartition != "SPX"
           && i->second.s_partition_id == 0) {
          i->second.s_partition_id = i->second.s_function;
          ss << __PRETTY_FUNCTION__ << " | (primary node add) fall back - "
            << "detected !SPX && partition_id == 0"
            << "; function = " << std::to_string(i->second.s_function)
            << "; partition_id = " << std::to_string(i->second.s_partition_id);
          LOG_DEBUG(ss);
        }
        LOG_DEBUG(ss);
        ss << __PRETTY_FUNCTION__
           << " | (primary node add) After AddToDeviceList() -->"
           << "\n[node_id = " << std::to_string(i->second.s_node_id)
           << "; gpu_id = " << std::to_string(i->second.s_gpu_id)
           << "; unique_id = " << std::to_string(i->second.s_unique_id)
           << "; location_id = " << std::to_string(i->second.s_location_id)
           << "; bdf = " << print_int_as_hex(i->second.s_bdf)
           << "; domain = " << print_int_as_hex(i->second.s_domain, true, 2*BYTE)
           << "; bus = " << print_int_as_hex(i->second.s_bus, true, BYTE)
           << "; device = " << print_int_as_hex(i->second.s_device, true, BYTE)
           << "; function = " << std::to_string(i->second.s_function)
           << "; partition_id = " << std::to_string(i->second.s_partition_id)
           << "], ";
        LOG_DEBUG(ss);
        AddToDeviceList(d_name, UINT64_MAX);
      }

      ss << __PRETTY_FUNCTION__
         << " | Ordered system nodes seen in lookup = {";
      for (auto i : allSystemNodes) {
        ss << "\n[node_id = " << std::to_string(i.second.s_node_id)
           << "; gpu_id = " << std::to_string(i.second.s_gpu_id)
           << "; unique_id = " << std::to_string(i.second.s_unique_id)
           << "; location_id = " << std::to_string(i.second.s_location_id)
           << "; bdf = " << print_int_as_hex(i.second.s_bdf)
           << "; domain = " << print_int_as_hex(i.second.s_domain, true, 2*BYTE)
           << "; bus = " << print_int_as_hex(i.second.s_bus, true, BYTE)
           << "; device = " << print_int_as_hex(i.second.s_device, true, BYTE)
           << "; function = " << std::to_string(i.second.s_function)
           << "; partition_id = " << std::to_string(i.second.s_partition_id)
           << "], ";
      }
      ss << "}";
      LOG_DEBUG(ss);

      uint64_t temp_primary_unique_id = 0;
      uint64_t primary_location_id = 0;
      if (allSystemNodes.empty()) {
        cardAdded++;
        ss << __PRETTY_FUNCTION__
           << " | allSystemNodes.empty() = true, continue...";
        LOG_DEBUG(ss);
        continue;
      }

      // get current partition
      rsmi_num_monitor_devices(&numMonDevices);
      if (rsmi_dev_compute_partition_get(cardAdded, computePartition, kSize)
          == RSMI_STATUS_SUCCESS) {
        strCompPartition = computePartition;
      }
      if (rsmi_dev_unique_id_get(cardAdded, &device_uuid)
          != RSMI_STATUS_SUCCESS) {
        cardAdded++;
        allSystemNodes.erase(device_uuid);
        ss << __PRETTY_FUNCTION__
           << " | rsmi_dev_unique_id_get(cardId, &device_uuid)"
           << " was not successful, continue.. ";
        LOG_DEBUG(ss);
        continue;
      }

      temp_primary_unique_id =
          allSystemNodes.find(device_uuid)->second.s_unique_id;
      temp_numb_nodes = allSystemNodes.count(temp_primary_unique_id);

      ss << __PRETTY_FUNCTION__
         << " | device/node id (cardId) = " << std::to_string(cardId)
         << " | card id (cardAdded) = " << std::to_string(cardAdded)
         << " | numMonDevices = " << std::to_string(numMonDevices)
         << " | compute partition = " << strCompPartition
         << " | temp_primary_unique_id = "
         << std::to_string(temp_primary_unique_id)
         << " | Num of nodes matching temp_primary_unique_id = "
         << temp_numb_nodes
         << " | device_uuid (hex/uint) = "
         << print_unsigned_hex_and_int(device_uuid)
         << " | device_uuid (uint64_t) = " << device_uuid;
      LOG_DEBUG(ss);

      if (temp_primary_unique_id != 0) {
        primary_unique_id = temp_primary_unique_id;
      } else {
        cardAdded++;
        // remove already added nodes associated with current card
        auto erasedNodes = allSystemNodes.erase(0);
        continue;
      }

      auto numb_nodes = allSystemNodes.count(primary_unique_id);
      ss << __PRETTY_FUNCTION__ << " | REFRESH - primary_unique_id = "
         << std::to_string(primary_unique_id) << " has "
         << std::to_string(numb_nodes) << " known gpu nodes";
      LOG_DEBUG(ss);
      while (numb_nodes > 1) {
        std::string secNode = "card";
        secNode += std::to_string(cardId);  // maps the primary node card to
                                            // secondary - allows get/sets
        auto it = allSystemNodes.lower_bound(device_uuid);
        auto it_end = allSystemNodes.upper_bound(device_uuid);
        if (numb_nodes == temp_numb_nodes) {
          auto removalNodeId = it->second.s_node_id;
          auto removalGpuId = it->second.s_gpu_id;
          auto removalUniqueId = it->second.s_unique_id;
          auto removalLocId = it->second.s_location_id;
          auto removaldomain = it->second.s_domain;
          auto nodesErased = 1;
          primary_location_id = removalLocId;
          allSystemNodes.erase(it++);
          ss << __PRETTY_FUNCTION__
             << "\nPRIMARY --> num_nodes == temp_numb_nodes; ERASING "
             << std::to_string(nodesErased) << " node -> [node_id = "
             << std::to_string(removalNodeId)
             << "; gpu_id = " << std::to_string(removalGpuId)
             << "; unique_id = " << std::to_string(removalUniqueId)
             << "; location_id = " << std::to_string(removalLocId)
             << "; removaldomain = " << std::to_string(removaldomain)
             << "]";
          LOG_DEBUG(ss);
        }
        if (it == it_end) {
          break;
        }
        auto myBdfId = it->second.s_location_id;
        ss << __PRETTY_FUNCTION__ << " | secondary node add #2; "
           << " BDF = " << std::to_string(myBdfId)
           << " (" << print_int_as_hex(myBdfId) << ")";
        LOG_DEBUG(ss);
        if (doesDeviceSupportPartitions && strCompPartition != "SPX"
           && it->second.s_partition_id == 0) {
          it->second.s_partition_id = it->second.s_function;
          ss << __PRETTY_FUNCTION__ << " | (secondary node add #2) fall back - "
            << "detected !SPX && partition_id == 0"
            << "; function = " << std::to_string(it->second.s_function)
            << "; partition_id = " << std::to_string(it->second.s_partition_id);
          LOG_DEBUG(ss);
        }
        ss << __PRETTY_FUNCTION__
           << " | (secondary node add #2) B4 AddToDeviceList() -->"
           << "\n[node_id = " << std::to_string(it->second.s_node_id)
           << "; gpu_id = " << std::to_string(it->second.s_gpu_id)
           << "; unique_id = " << std::to_string(it->second.s_unique_id)
           << "; location_id = " << std::to_string(it->second.s_location_id)
           << "; bdf = " << print_int_as_hex(it->second.s_bdf)
           << "; domain = " << print_int_as_hex(it->second.s_domain, true, 2*BYTE)
           << "; bus = " << print_int_as_hex(it->second.s_bus, true, BYTE)
           << "; device = " << print_int_as_hex(it->second.s_device, true, BYTE)
           << "; function = " << std::to_string(it->second.s_function)
           << "; partition_id = " << std::to_string(it->second.s_partition_id)
           << "], ";
        LOG_DEBUG(ss);
        AddToDeviceList(secNode, myBdfId);
        allSystemNodes.erase(it++);
        numb_nodes--;
        cardAdded++;
      }
      // remove any remaining nodes associated with current card
      auto erasedNodes = allSystemNodes.erase(primary_unique_id);
      ss << __PRETTY_FUNCTION__ << " | After finding primary_unique_id = "
         << std::to_string(primary_unique_id) << " erased "
         << std::to_string(erasedNodes) << " nodes";
      LOG_DEBUG(ss);
      cardAdded++;
    }
  }

  if (closedir(drm_dir)) {
    err_msg = "Failed to close drm root directory ";
    err_msg += kPathDRMRoot;
    err_msg += ".";
    perror(err_msg.c_str());
    return 1;
  }
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
