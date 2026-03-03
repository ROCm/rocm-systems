/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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

#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_common.h"
#include "rocm_smi/rocm_smi_exception.h"

namespace amd::smi {


static const char *kPowerMonPMName = "amdgpu_pm_info";

// Using this map in case we add other files from dri directory to parse.
static const std::map<PowerMonTypes, const char *> kMonitorNameMap = {
    {kPowerMaxGPUPower, kPowerMonPMName},
};

PowerMon::PowerMon(std::string path, RocmSMI_env_vars const *e) :
                                                        path_(path), env_(e) {
}
PowerMon::~PowerMon(void) = default;

static int parse_power_str(std::string s, PowerMonTypes type, uint64_t *val) {
  std::stringstream ss(s);
  std::string ln;
  std::string search_str;

  assert(val != nullptr);

  switch (type) {
    case kPowerMaxGPUPower:
      search_str = "(max GPU)";
      break;

    default:
      assert(false);  // Invalid search Power type requested
      return EINVAL;
  }

  bool found = false;
  while (std::getline(ss, ln)) {
    if (ln.rfind(search_str) != std::string::npos) {
      found = true;
      break;
    }
  }

  if (!found) {
    return EPERM;
  }

  ss.clear();
  std::stringstream l_ss;

  l_ss << ln;

  double num_units;
  std::string sz;

  switch (type) {
    case kPowerMaxGPUPower:
      l_ss >> num_units;
      l_ss >> sz;
      assert(sz == "W");  // We only expect Watts at this time
      if (sz != "W") {
        throw amd::smi::rsmi_exception(RSMI_STATUS_UNEXPECTED_DATA,
                                                                __FUNCTION__);
      }

      if (num_units > static_cast<long double>(0xFFFFFFFFFFFFFFFF)/1000) {
        throw amd::smi::rsmi_exception(RSMI_STATUS_UNEXPECTED_DATA,
                                                                __FUNCTION__);
      }
      *val = static_cast<uint64_t>(num_units * 1000);  // Convert W to mW
      break;

    default:
      assert(false);  // Invalid search Power type requested
      return EINVAL;
  }
  ss.clear();
  return 0;
}

int PowerMon::readPowerValue(PowerMonTypes type, uint64_t *power) {
  auto tempPath = path_;
  std::string fstr;

  assert(power != nullptr);

  tempPath += "/";
  tempPath += kMonitorNameMap.at(type);

  DBG_FILE_ERROR(tempPath, (std::string *)nullptr)
  int ret = ReadSysfsStr(tempPath, &fstr);

  if (ret) {
    return ret;
  }

  return parse_power_str(fstr, type, power);
}

} // namespace amd::smi
