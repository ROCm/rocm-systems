/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2018, Advanced Micro Devices, Inc.
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

#ifndef TESTS_ROCM_SMI_TEST_TEST_COMMON_H_
#define TESTS_ROCM_SMI_TEST_TEST_COMMON_H_

#include <memory>
#include <vector>
#include <string>

#include "rocm_smi/rocm_smi.h"

struct RSMITstGlobals {
  uint32_t verbosity;
  uint32_t monitor_verbosity;
  uint32_t num_iterations;
  uint64_t init_options;
  bool dont_fail;
};

uint32_t ProcessCmdline(RSMITstGlobals* test, int arg_cnt, char** arg_list);

void PrintTestHeader(uint32_t dv_ind);
const char *GetPerfLevelStr(rsmi_dev_perf_level_t lvl);
const char *GetBlockNameStr(rsmi_gpu_block_t id);
const char *GetErrStateNameStr(rsmi_ras_err_state_t st);
const char *FreqEnumToStr(rsmi_clk_type rsmi_clk);
const std::string GetVoltSensorNameStr(rsmi_voltage_type_t st);

#if ENABLE_SMI
void DumpMonitorInfo(const TestBase *test);
#endif

#define DISPLAY_RSMI_ERR(RET) { \
  if ((RET) != RSMI_STATUS_SUCCESS) { \
    const char *err_str; \
    std::cout << "\t===> ERROR: RSMI call returned " << (RET) << std::endl; \
    rsmi_status_string((RET), &err_str); \
    std::cout << "\t===> (" << err_str << ")" << std::endl; \
    std::cout << "\t===> at " << __FILE__ << ":" << std::dec << __LINE__ << \
                                                                  std::endl; \
  } \
}

#define CHK_ERR_RET(RET) { \
  DISPLAY_RSMI_ERR(RET) \
  if ((RET) != RSMI_STATUS_SUCCESS) { \
    return (RET); \
  } \
}
#define CHK_RSMI_PERM_ERR(RET) { \
    if ((RET) == RSMI_STATUS_PERMISSION) { \
      std::cout << "This command requires root access." << std::endl; \
    } else { \
      DISPLAY_RSMI_ERR(RET) \
    } \
}

void printRSMIError(rsmi_status_t err);

#endif  // TESTS_ROCM_SMI_TEST_TEST_COMMON_H_
