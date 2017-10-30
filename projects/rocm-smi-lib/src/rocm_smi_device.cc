/*   ROC Runtime Conformance Release License
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

#include <assert.h>
#include <sys/stat.h>

#include <string>
#include <map>
#include <fstream>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>

#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_device.h"

namespace amd {
namespace smi {

static const char *kDevPerfLevelFName = "power_dpm_force_performance_level";
static const char *kDevDevIDFName = "device";
static const char *kDevOverDriveLevelFName = "pp_sclk_od";
static const char *kDevGPUSClkFName = "pp_dpm_sclk";
static const char *kDevGPUMClkFName = "pp_dpm_mclk";

static const std::map<DevInfoTypes, const char *> kDevAttribNameMap = {
    {kDevPerfLevel, kDevPerfLevelFName},
    {kDevOverDriveLevel, kDevOverDriveLevelFName},
    {kDevDevID, kDevDevIDFName},
    {kDevGPUMClk, kDevGPUMClkFName},
    {kDevGPUSClk, kDevGPUSClkFName},
};

static bool isRegularFile(std::string fname) {
  struct stat file_stat;
  stat(fname.c_str(), &file_stat);
  return S_ISREG(file_stat.st_mode);
}

Device::Device(std::string p) : path_(p) {
  monitor_ = nullptr;
}

Device:: ~Device() {
}

// TODO(cfreehil): cache values that are constant
int Device::readDevInfoStr(DevInfoTypes type, std::string *retStr) {
  auto tempPath = path_;

  assert(retStr != nullptr);

  tempPath += "/device/";
  tempPath += kDevAttribNameMap.at(type);

  std::ifstream fs;
  fs.open(tempPath);

  if (!fs.is_open() || !isRegularFile(tempPath)) {
      return -1;
  }
  fs >> *retStr;
  fs.close();

  return 0;
}

int Device::readDevInfoMultiLineStr(DevInfoTypes type,
                                           std::vector<std::string> *retVec) {
  auto tempPath = path_;
  std::string line;

  assert(retVec != nullptr);

  tempPath += "/device/";
  tempPath += kDevAttribNameMap.at(type);

  std::ifstream fs(tempPath);
  std::stringstream buffer;


  if (!isRegularFile(tempPath)) {
    return -1;
  }

  while (std::getline(fs, line)) {
    retVec->push_back(line);
  }
  return 0;
}

int Device::readDevInfo(DevInfoTypes type, uint32_t *val) {
  assert(val != nullptr);

  std::string tempStr;

  switch (type) {
    case kDevDevID:
      if (readDevInfoStr(type, &tempStr)) {
        return -1;
      }
      *val = std::stoi(tempStr, 0, 16);
      break;

    case kDevOverDriveLevel:
      if (readDevInfoStr(type, &tempStr)) {
        return -1;
      }
      *val = std::stoi(tempStr, 0);
      break;

    default:
      return -1;
  }
  return 0;
}

int Device::readDevInfo(DevInfoTypes type, std::vector<std::string> *val) {
  assert(val != nullptr);

  switch (type) {
    case kDevGPUMClk:
    case kDevGPUSClk:
      if (readDevInfoMultiLineStr(type, val)) {
        return -1;
      }
      break;

    default:
      return -1;
  }

  return 0;
}

int Device::readDevInfo(DevInfoTypes type, std::string *val) {
  assert(val != nullptr);

  switch (type) {
    case kDevPerfLevel:
    case kDevOverDriveLevel:
    case kDevDevID:
      if (readDevInfoStr(type, val)) {
        return -1;
      }
      break;

    default:
      return -1;
  }
  return 0;
}

}  // namespace smi
}  // namespace amd
