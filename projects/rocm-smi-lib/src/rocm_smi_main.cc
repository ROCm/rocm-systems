/*
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
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
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdlib.h>

#include <string>
#include <cstdint>
#include <memory>
#include <fstream>
#include <vector>
#include <set>
#include <utility>
#include <functional>
#include <cerrno>
#include <unordered_map>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_kfd.h"

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

  return stoi(t);
}

// Find the drm minor from from sysfs path "/sys/class/drm/cardX/device/drm".
// From the directory renderDN in that sysfs path, the drm minor can be
// computed for cardX.
// On success, return drm_minor which is >= 128 otherwise return 0
static uint32_t  GetDrmRenderMinor(const std::string s) {
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

  if (closedir(drm_dir))
    return 0;

  return drm_minor;
}

static int SameDevice(const std::string fileA, const std::string fileB) {
  return SameFile(fileA + "/device", fileB + "/device");
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

static uint32_t ConstructBDFID(std::string path, uint64_t *bdfid) {
  assert(bdfid != nullptr);
  char tpath[256];
  ssize_t ret;

  ret = readlink(path.c_str(), tpath, 256);

  assert(ret > 0);
  assert(ret < 256);

  if (ret <= 0 || ret >= 256) {
    return -1;
  }

  // We are looking for the last element in the path that has the form
  //  XXXX:XX:XX.X, where X is a hex integer (lower case is expected)
  std::size_t slash_i, end_i;
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
// Call-back function to append to a vector of Devices
static uint32_t GetMonitorDevices(const std::shared_ptr<amd::smi::Device> &d,
                                                                    void *p) {
  std::string val_str;
  uint64_t bdfid;

  assert(p != nullptr);

  std::vector<std::shared_ptr<amd::smi::Device>> *device_list =
    reinterpret_cast<std::vector<std::shared_ptr<amd::smi::Device>> *>(p);

  if (d->monitor() != nullptr) {
    // Calculate BDFID and set for this device
    if (ConstructBDFID(d->path(), &bdfid) != 0) {
      return -1;
    }
    d->set_bdfid(bdfid);
    device_list->push_back(d);
  }
  return 0;
}

void
RocmSMI::Initialize(uint64_t flags) {
  auto i = 0;
  uint32_t ret;

  assert(ref_count_ == 1);
  if (ref_count_ != 1) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
            "Unexpected: RocmSMI ref_count_ != 1");
  }

  init_options_ = flags;

  euid_ = geteuid();

  GetEnvVariables();

  while (std::string(kAMDMonitorTypes[i]) != "") {
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

  // IterateSMIDevices will iterate through all the known devices and apply
  // the provided call-back to each device found.
  ret = IterateSMIDevices(GetMonitorDevices,
                                  reinterpret_cast<void *>(&monitor_devices_));

  if (ret != 0) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
            "Failed to initialize rocm_smi library (amdgpu node discovery).");
  }

  std::map<uint64_t, std::shared_ptr<KFDNode>> tmp_map;
  ret = DiscoverKFDNodes(&tmp_map);
  if (ret != 0) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
                 "Failed to initialize rocm_smi library (KFD node discovery).");
  }

  std::map<std::pair<uint32_t, uint32_t>, std::shared_ptr<IOLink>>
    io_link_map_tmp;
  ret = DiscoverIOLinks(&io_link_map_tmp);
  if (ret != 0) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
                 "Failed to initialize rocm_smi library (IO Links discovery).");
  }
  std::map<std::pair<uint32_t, uint32_t>, std::shared_ptr<IOLink>>::iterator it;
  for (it = io_link_map_tmp.begin(); it != io_link_map_tmp.end(); it++)
    io_link_map_[it->first] = it->second;

  std::shared_ptr<amd::smi::Device> dev;

  // Remove any drm nodes that don't have  a corresponding readable kfd node.
  // kfd nodes will not be added if their properties file is not readable.
  auto dev_iter = monitor_devices_.begin();
  while (dev_iter != monitor_devices_.end()) {
    uint64_t bdfid = (*dev_iter)->bdfid();
    if (tmp_map.find(bdfid) == tmp_map.end()) {
      dev_iter = monitor_devices_.erase(dev_iter);
      continue;
    }
    dev_iter++;
  }

  // 1. construct kfd_node_map_ with gpu_id as key and *Device as value
  // 2. for each kfd node, write the corresponding dv_ind
  // 3. for each amdgpu device, write the corresponding gpu_id
  for (uint32_t dv_ind = 0; dv_ind < monitor_devices_.size(); ++dv_ind) {
    dev = monitor_devices_[dv_ind];
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
  }
}

void
RocmSMI::Cleanup() {
  monitor_devices_.clear();
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

RocmSMI::~RocmSMI() {
}

RocmSMI& RocmSMI::getInstance(uint64_t flags) {
  // Assume c++11 or greater. static objects will be created by only 1 thread
  // and creation will be thread-safe.
  static RocmSMI singleton(flags);
  return singleton;
}

static int GetEnvVarInteger(const char *ev_str) {
#ifdef NDEBUG
  (void)ev_str;
#else
  ev_str = getenv(ev_str);

  if (ev_str) {
    return atoi(ev_str);
  }
#endif
  return 0;
}

// Get and store env. variables in this method
void RocmSMI::GetEnvVariables(void) {
#ifdef NDEBUG
  (void)GetEnvVarInteger(nullptr);  // This is to quiet release build warning.
  env_vars_.debug_output_bitfield = 0;
  env_vars_.path_DRM_root_override = nullptr;
  env_vars_.path_HWMon_root_override = nullptr;
  env_vars_.path_power_root_override = nullptr;
  env_vars_.enum_override = 0;
#else
  env_vars_.debug_output_bitfield = GetEnvVarInteger("RSMI_DEBUG_BITFIELD");
  env_vars_.path_DRM_root_override   = getenv("RSMI_DEBUG_DRM_ROOT_OVERRIDE");
  env_vars_.path_HWMon_root_override = getenv("RSMI_DEBUG_HWMON_ROOT_OVERRIDE");
  env_vars_.path_power_root_override = getenv("RSMI_DEBUG_PP_ROOT_OVERRIDE");
  env_vars_.enum_override = GetEnvVarInteger("RSMI_DEBUG_ENUM_OVERRIDE");
#endif
}

void
RocmSMI::AddToDeviceList(std::string dev_name) {
  auto ret = 0;

  auto dev_path = std::string(kPathDRMRoot);
  dev_path += "/";
  dev_path += dev_name;

  auto dev = std::shared_ptr<Device>(new Device(dev_path, &env_vars_));

  auto m = monitors_.begin();

  while (m != monitors_.end()) {
      ret = SameDevice(dev->path(), (*m)->path());

      if (ret == 0) {
        dev->set_monitor(*m);
        m = monitors_.erase(m);
        break;
      } else {
        assert(ret == 1);
        ++m;
      }
  }

  std::string d_name = dev_name;
  uint32_t card_indx = GetDeviceIndex(d_name);
  dev->set_drm_render_minor(GetDrmRenderMinor(dev_path));
  dev->set_card_index(card_indx);
  GetSupportedEventGroups(card_indx, dev->supported_event_groups());

  devices_.push_back(dev);

  return;
}

static const uint32_t kAmdGpuId = 0x1002;

static bool isAMDGPU(std::string dev_path) {
  std::string vend_path = dev_path + "/device/vendor";
  if (!FileExists(vend_path.c_str())) {
    return false;
  }

  std::ifstream fs;
  fs.open(vend_path);

  if (!fs.is_open()) {
      return false;
  }

  uint32_t vendor_id;

  fs >> std::hex >> vendor_id;

  fs.close();

  if (vendor_id == kAmdGpuId) {
    return true;
  }
  return false;
}

uint32_t RocmSMI::DiscoverAmdgpuDevices(void) {
  auto ret = 0;

  // If this gets called more than once, clear previous findings.
  devices_.clear();
  monitors_.clear();

  ret = DiscoverAMDMonitors();

  if (ret) {
    return ret;
  }

  auto drm_dir = opendir(kPathDRMRoot);
  if (drm_dir == nullptr) {
    return 1;
  }

  auto dentry = readdir(drm_dir);

  while (dentry != nullptr) {
    if (memcmp(dentry->d_name, kDeviceNamePrefix, strlen(kDeviceNamePrefix))
                                                                       == 0) {
      std::string vend_str_path = kPathDRMRoot;
      vend_str_path += "/";
      vend_str_path += dentry->d_name;

      if (isAMDGPU(vend_str_path) ||
          (init_options_ & RSMI_INIT_FLAG_ALL_GPUS)) {
        AddToDeviceList(dentry->d_name);
      }
    }
    dentry = readdir(drm_dir);
  }

  if (closedir(drm_dir)) {
    return 1;
  }
  return 0;
}

uint32_t RocmSMI::DiscoverAMDMonitors(void) {
  auto mon_dir = opendir(kPathHWMonRoot);

  auto dentry = readdir(mon_dir);

  std::string mon_name;
  std::string tmp;

  while (dentry != nullptr) {
    if (dentry->d_name[0] == '.') {
      dentry = readdir(mon_dir);
      continue;
    }

    mon_name = kPathHWMonRoot;
    mon_name += "/";
    mon_name += dentry->d_name;
    tmp = mon_name + "/name";

    if (FileExists(tmp.c_str())) {
      std::ifstream fs;
      fs.open(tmp);

      if (!fs.is_open()) {
          return 1;
      }
      std::string mon_type;
      fs >> mon_type;
      fs.close();

      if (amd_monitor_types_.find(mon_type) != amd_monitor_types_.end()) {
        std::shared_ptr<Monitor> m =
                  std::shared_ptr<Monitor>(new Monitor(mon_name, &env_vars_));
        m->setTempSensorLabelMap();
        m->setVoltSensorLabelMap();
        monitors_.push_back(m);
      }
    }
    dentry = readdir(mon_dir);
  }

  if (closedir(mon_dir)) {
    return 1;
  }
  return 0;
}

// Since these sysfs files require sudo access, we won't discover them
// with rsmi_init() (and thus always require the user to use "sudo".
// Instead, we will discover() all the power monitors the first time
// they are needed and then check for previous discovery on each subsequent
// call.
uint32_t RocmSMI::DiscoverAMDPowerMonitors(bool force_update) {
  if (force_update) {
    power_mons_.clear();
  }

  if (power_mons_.size() != 0) {
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
                std::shared_ptr<PowerMon>(new PowerMon(mon_name, &env_vars_));
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

  for (auto m : power_mons_) {
    for (auto d : devices_) {
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
    return -1;
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
