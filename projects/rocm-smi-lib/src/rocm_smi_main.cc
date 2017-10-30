/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
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

#include <string>
#include <cstdint>
#include <memory>
#include <fstream>
#include <vector>
#include <set>
#include <utility>
#include <functional>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_main.h"

static const char *kPathDRMRoot = "/sys/class/drm";
static const char *kPathHWMonRoot = "/sys/class/hwmon";
static const char *kDeviceNamePrefix = "card";

static const char *kAMDMonitorTypes[] = {"radeon", "amdgpu", ""};

namespace amd {
namespace smi {

static bool FileExists(char const *filename) {
  struct stat buf;
  return (stat(filename, &buf) == 0);
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

void ShowAllTemperatures();

RocmSMI::RocmSMI() {
  auto i = 0;

  while (std::string(kAMDMonitorTypes[i]) != "") {
      amd_monitor_types_.insert(kAMDMonitorTypes[i]);
      ++i;
  }
}

RocmSMI::~RocmSMI() {
  devices_.clear();
  monitors_.clear();
}

void
RocmSMI::AddToDeviceList(std::string dev_name) {
  auto ret = 0;

  auto dev_path = std::string(kPathDRMRoot);
  dev_path += "/";
  dev_path += dev_name;

  auto dev = std::shared_ptr<Device>(new Device(dev_path));

  auto m = monitors_.begin();

  while (m != monitors_.end()) {
      ret = SameDevice(dev->path(), (*m)->path());

      if (ret == 0) {
        dev->set_monitor(*m);

        m = monitors_.erase(m);
      } else {
        assert(ret == 1);
        ++m;
      }
  }

  devices_.push_back(dev);

  return;
}

uint32_t RocmSMI::DiscoverDevices(void) {
  auto ret = 0;

  ret = DiscoverAMDMonitors();

  if (ret) {
    return ret;
  }

  auto drm_dir = opendir(kPathDRMRoot);
  auto dentry = readdir(drm_dir);

  while (dentry != nullptr) {
    if (memcmp(dentry->d_name, kDeviceNamePrefix, strlen(kDeviceNamePrefix))
                                                                       == 0) {
      AddToDeviceList(dentry->d_name);
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
        monitors_.push_back(std::shared_ptr<Monitor>(new Monitor(mon_name)));
      }
    }
    dentry = readdir(mon_dir);
  }

  if (closedir(mon_dir)) {
    return 1;
  }
  return 0;
}

void RocmSMI::IterateSMIDevices(
        std::function<bool(std::shared_ptr<Device>&, void *)> func, void *p) {

  if (func == nullptr) {
    return;
  }

  auto d = devices_.begin();

  while (d != devices_.end()) {
    if (func(*d, p)) {
      return;
    }
    ++d;
  }
}

}  // namespace smi
}  // namespace amd
