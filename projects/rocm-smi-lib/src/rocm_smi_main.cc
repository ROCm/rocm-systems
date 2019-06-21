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
#include <sys/stat.h>
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

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_exception.h"

static const char *kPathDRMRoot = "/sys/class/drm";
static const char *kPathHWMonRoot = "/sys/class/hwmon";
static const char *kPathPowerRoot = "/sys/kernel/debug/dri";

static const char *kDeviceNamePrefix = "card";

static const char *kAMDMonitorTypes[] = {"radeon", "amdgpu", ""};

namespace amd {
namespace smi {

static bool FileExists(char const *filename) {
  struct stat buf;
  return (stat(filename, &buf) == 0);
}

static uint32_t GetDeviceIndex(const std::string s) {
  std::string t = s;
  size_t tmp = t.find_last_not_of("0123456789");
  t.erase(0, tmp+1);

  return stoi(t);
}

// Return 0 if same file, 1 if not, and -1 for error
static int SameFile(const std::string fileA, const std::string fileB) {
  struct stat aStat;
  struct stat bStat;
  int ret;

  ret = stat(fileA.c_str(), &aStat);
  if (ret) {
      return -1;
  }

  ret = stat(fileB.c_str(), &bStat);
  if (ret) {
      return -1;
  }

  if (aStat.st_dev != bStat.st_dev) {
      return 1;
  }

  if (aStat.st_ino != bStat.st_ino) {
      return 1;
  }

  return 0;
}

static int SameDevice(const std::string fileA, const std::string fileB) {
  return SameFile(fileA + "/device", fileB + "/device");
}

//  Determine if provided string is a bdfid pci path directory of the form
//  XXXX:XX:XX.X,
//  domain:bus:device.function
//
//  where X is a hex integer (lower case is expected)
static bool is_bdfid_path_str(const std::string in_name, uint64_t *bdfid) {
  char *p = nullptr;
  char *name_start;
  char name[13] = {'\0'};
  uint32_t tmp;

  assert(bdfid != nullptr);

  if (in_name.size() != 12) {
    return false;
  }

  tmp = in_name.copy(name, 12);
  assert(tmp == 12);

  // BDFID = ((<DOMAIN> & 0xffff) << 13) | ((<BUS> & 0x1f) << 8) |
            //                        ((device& 0x1f) <<3 ) | (function & 0x7)
  *bdfid = 0;
  name_start = name;
  p = name_start;

  // Match this: XXXX:xx:xx.x
  tmp = std::strtoul(p, &p, 16);
  if (*p != ':' || p - name_start != 4) {
    return false;
  }
  *bdfid |= tmp << 13;

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

    if (is_bdfid_path_str(tmp, bdfid)) {
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

std::vector<std::shared_ptr<amd::smi::Device>> RocmSMI::s_monitor_devices;

void
RocmSMI::Initialize(uint64_t flags) {
  auto i = 0;
  uint32_t ret;

  init_options_ = flags;

  euid_ = geteuid();

  GetEnvVariables();

  while (std::string(kAMDMonitorTypes[i]) != "") {
      amd_monitor_types_.insert(kAMDMonitorTypes[i]);
      ++i;
  }

  // DiscoverDevices() will seach for devices and monitors and update internal
  // data structures.
  DiscoverDevices();

  // IterateSMIDevices will iterate through all the known devices and apply
  // the provided call-back to each device found.
  ret = IterateSMIDevices(GetMonitorDevices,
                                  reinterpret_cast<void *>(&s_monitor_devices));

  if (ret != 0) {
    throw amd::smi::rsmi_exception(RSMI_INITIALIZATION_ERROR,
                                      "Failed to initialize rocm_smi library.");
  }
}

void
RocmSMI::Cleanup() {
  s_monitor_devices.clear();
  devices_.clear();
  monitors_.clear();
}

RocmSMI::RocmSMI(uint64_t flags) : init_options_(flags) {
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
  ev_str = getenv(ev_str);

  if (ev_str) {
    return atoi(ev_str);
  }
  return 0;
}

// Get and store env. variables in this method
void RocmSMI::GetEnvVariables(void) {
  env_vars_.debug_output_bitfield = GetEnvVarInteger("RSMI_DEBUG_BITFIELD");
  env_vars_.path_DRM_root_override   = getenv("RSMI_DEBUG_DRM_ROOT_OVERRIDE");
  env_vars_.path_HWMon_root_override = getenv("RSMI_DEBUG_HWMON_ROOT_OVERRIDE");
  env_vars_.path_power_root_override = getenv("RSMI_DEBUG_PP_ROOT_OVERRIDE");
  env_vars_.enum_override = GetEnvVarInteger("RSMI_DEBUG_ENUM_OVERRIDE");
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
  uint32_t d_index = GetDeviceIndex(d_name);
  dev->set_index(d_index);

  GetSupportedEventGroups(d_index, dev->supported_event_groups());
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

uint32_t RocmSMI::DiscoverDevices(void) {
  auto ret = 0;

  // If this gets called more than once, clear previous findings.
  devices_.clear();
  monitors_.clear();

  ret = DiscoverAMDMonitors();

  if (ret) {
    return ret;
  }

  auto drm_dir = opendir(kPathDRMRoot);
  assert(drm_dir != nullptr);

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
        monitors_.push_back(std::shared_ptr<Monitor>(
                                          new Monitor(mon_name, &env_vars_)));
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

}  // namespace smi
}  // namespace amd
