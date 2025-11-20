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
#include <iomanip>

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

#define DISPLAY_AMDSMI_API(FUNC_NAME, STR) { \
  std::cout << "\t### " << (FUNC_NAME) << "(" << (STR) << ")" << std::endl; \
}
static amdsmi_status_t NotSupportedErrorCodes[] = {
    AMDSMI_STATUS_NOT_SUPPORTED,
    AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
    AMDSMI_STATUS_NO_HSMP_MSG_SUP
};
#define DISPLAY_AMDSMI_STATUS(RET, ...) { \
  amdsmi_status_t retExpected[] = {__VA_ARGS__}; \
  int numRetExpected = sizeof(retExpected) / sizeof(retExpected[0]); \
  amdsmi_status_t RET_EXPECTED = retExpected[0]; \
  auto it = std::find(std::begin(retExpected), std::end(retExpected), (RET)); \
  if (it != std::end(retExpected)) { \
    RET_EXPECTED = (RET); \
  } \
  if ((RET) != RET_EXPECTED) { \
    const char *_err; \
    std::string err_str; \
    size_t pos; \
    amdsmi_status_code_to_string((RET), &_err); \
    err_str = std::string(_err); \
    std::string status_str = !err_str.empty() ? err_str : "Unknown"; \
    pos = status_str.find(":"); \
    if (pos != std::string::npos) status_str = status_str.substr(0, pos); \
    amdsmi_status_code_to_string(RET_EXPECTED, &_err); \
    err_str = std::string(_err); \
    std::string status_expected_str = !err_str.empty() ? err_str : "Unknown"; \
    pos = status_expected_str.find(":"); \
    if (pos != std::string::npos) status_expected_str = status_expected_str.substr(0, pos); \
    auto it = std::find(std::begin(NotSupportedErrorCodes),std::end(NotSupportedErrorCodes), (RET)); \
    if (it != std::end(NotSupportedErrorCodes)) { \
      std::cout << "\t===> AMDSMI API Returned " << (RET) << ", " << status_str << std::endl; \
    } else { \
      std::string start_dir = std::string(__FILE__); \
      pos = start_dir.find("tests/amd_smi_test"); \
      if (pos != std::string::npos) start_dir = start_dir.substr(pos); \
      std::cout << "\t===> TEST FAILURE." << std::endl; \
      std::cout << "\t===> ERROR: AMDSMI API Returned " << std::setfill(' ') << std::setw(2) << (RET) << ", " << status_str << std::endl; \
      if (numRetExpected == 1) \
      std::cout << "\t===>                   Expected " << std::setfill(' ') << std::setw(2) << RET_EXPECTED << ", " << status_expected_str << std::endl; \
      else { \
        std::cout << "\t===>                   Expected One of"; \
        for (int i=0; i<numRetExpected; ++i) std::cout << " " << retExpected[i]; \
        std::cout << std::endl; \
      } \
      std::cout << "\t===> " << start_dir << ":" << std::dec << __LINE__ << std::endl; \
    } \
  } \
}

#endif  // TESTS_AMD_SMI_TEST_TEST_COMMON_H_
