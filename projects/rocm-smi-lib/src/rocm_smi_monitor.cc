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

#include <assert.h>

#include <fstream>
#include <string>
#include <cstdint>
#include <map>
#include <iostream>

#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_monitor.h"

namespace amd {
namespace smi {

struct MonitorNameEntry {
    MonitorTypes type;
    const char *name;
};


static const char *kMonTempFName = "temp1_input";
static const char *kMonFanSpeedFName = "pwm1";
static const char *kMonMaxFanSpeedFName = "pwm1_max";
static const char *kMonNameFName = "name";

static const std::map<MonitorTypes, const char *> kMonitorNameMap = {
    {kMonName, kMonNameFName},
    {kMonTemp, kMonTempFName},
    {kMonFanSpeed, kMonFanSpeedFName},
    {kMonMaxFanSpeed, kMonMaxFanSpeedFName}
};

Monitor::Monitor(std::string path) : path_(path) {
}
Monitor::~Monitor(void) {
}

int Monitor::readMonitorStr(MonitorTypes type, std::string *retStr) {
  auto tempPath = path_;

  assert(retStr != nullptr);

  tempPath += "/";
  tempPath += kMonitorNameMap.at(type);

  std::ifstream fs;
  fs.open(tempPath);

  if (!fs.is_open()) {
      return -1;
  }
  fs >> *retStr;
  fs.close();

  return 0;
}

int Monitor::readMonitor(MonitorTypes type, uint32_t *val) {
  assert(val != nullptr);

  std::string tempStr;

  switch (type)  {
    case kMonTemp:     // Temperature in millidegrees
    case kMonFanSpeed:
    case kMonMaxFanSpeed:
      if (readMonitorStr(type, &tempStr)) {
        return -1;
      }
      *val = std::stoi(tempStr);
      return 0;

    default:
      return -1;
  }
}

// This string version should work for all valid monitor types
int Monitor::readMonitor(MonitorTypes type, std::string *val) {
  assert(val != nullptr);

  if (readMonitorStr(type, val)) {
    return -1;
  }

  return 0;
}


}  // namespace smi
}  // namespace amd
