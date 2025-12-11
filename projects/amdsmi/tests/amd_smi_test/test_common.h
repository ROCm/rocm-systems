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

#ifndef TESTS_AMD_SMI_TEST_TEST_COMMON_H_
#define TESTS_AMD_SMI_TEST_TEST_COMMON_H_

#include <memory>
#include <vector>
#include <string>

#include "amd_smi/amdsmi.h"

struct AMDSMITstGlobals {
  uint32_t verbosity;
  uint32_t monitor_verbosity;
  uint32_t num_iterations;
  uint64_t init_options;
  bool dont_fail;
};

uint32_t ProcessCmdline(AMDSMITstGlobals* test, int arg_cnt, char** arg_list);

void PrintTestHeader(uint32_t dv_ind);
const char *GetPerfLevelStr(amdsmi_dev_perf_level_t lvl);
const char *GetBlockNameStr(amdsmi_gpu_block_t id);
const char *GetErrStateNameStr(amdsmi_ras_err_state_t st);
const char *FreqEnumToStr(amdsmi_clk_type_t amdsmi_clk);
const std::string GetVoltSensorNameStr(amdsmi_voltage_type_t st);

#if ENABLE_SMI
void DumpMonitorInfo(const TestBase *test);
#endif

#define DISPLAY_AMDSMI_ERR(RET) { \
  if ((RET) != AMDSMI_STATUS_SUCCESS) { \
    const char *err_str; \
    std::cout << "\t===> ERROR: AMDSMI call returned " << (RET) << std::endl; \
    amdsmi_status_code_to_string((RET), &err_str); \
    std::cout << "\t===> (" << err_str << ")" << std::endl; \
    std::cout << "\t===> at " << __FILE__ << ":" << std::dec << __LINE__ << \
                                                                  std::endl; \
  } \
}

#define CHK_ERR_RET(RET) { \
  DISPLAY_AMDSMI_ERR(RET) \
  if ((RET) != AMDSMI_STATUS_SUCCESS) { \
    return (RET); \
  } \
}
#define CHK_AMDSMI_PERM_ERR(RET) { \
    if ((RET) == AMDSMI_STATUS_NO_PERM) { \
      std::cout << "This command requires root access." << std::endl; \
    } else { \
      DISPLAY_AMDSMI_ERR(RET) \
    } \
}

#endif  // TESTS_AMD_SMI_TEST_TEST_COMMON_H_
