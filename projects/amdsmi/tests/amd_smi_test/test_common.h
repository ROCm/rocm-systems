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

#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "test_base.h"

struct AMDSMITstGlobals {
  uint32_t verbosity;
  uint32_t monitor_verbosity;
  uint32_t num_iterations;
  uint64_t init_options;
  bool dont_fail;
};

uint32_t ProcessCmdline(AMDSMITstGlobals* test, int arg_cnt, char** arg_list);

// Returns the global verbosity level set from the command line.
// Can be used in plain TEST() cases that don't inherit from TestBase.
uint32_t GetTestVerbosity();
void SetTestVerbosity(uint32_t verbosity);

void PrintTestHeader(uint32_t dv_ind);
const char* GetPerfLevelStr(amdsmi_dev_perf_level_t lvl);
const char* GetBlockNameStr(amdsmi_gpu_block_t id);
const char* GetErrStateNameStr(amdsmi_ras_err_state_t st);
const char* FreqEnumToStr(amdsmi_clk_type_t amdsmi_clk);
const std::string GetVoltSensorNameStr(amdsmi_voltage_type_t st);

#if ENABLE_SMI
void DumpMonitorInfo(const TestBase* test);
#endif

inline constexpr amdsmi_status_t NotSupportedErrorCodes[] = {
    AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_HSMP_MSG_SUP};

inline void DISPLAY_AMDSMI_API(std::string_view func_name, std::string_view desc, bool isVerbose) {
  if (isVerbose) std::cout << "\t### " << (func_name) << "(" << (desc) << ")" << std::endl;
  return;
}

template <typename... Args>
inline void DISPLAY_AMDSMI_STATUS(bool isVerbose, std::string_view fileName,
                                  long unsigned int lineNum, amdsmi_status_t returnCode,
                                  Args... args) {
  // Input:
  //     isVerbose  : Toggle for outputing to std_out
  //                  True : Allow printing
  //                  False: No printing
  //     fileName   : Name of file calling this routine
  //     lineNum    : Line number of function in file
  //     returnCode : API return code
  //     args       : API Expected return code(s)
  //                  Must have at least 1, can be multiple
  //
  // Output:
  //     isVerbose(true)  : Print results
  //     isVerbose(false) : No output
  //
  // Notes:
  //     1. The API returnCode is checked against all args, expected return codes.
  //        If API returnCode matches any expected return codes, test passes with
  //            TEST SUCCESS, AMDSMI API Returned 0, AMDSMI_STATUS_SUCCESS
  //     2. If the API returnCode is one of the Not Supported return codes
  //        the test will pass with
  //            TEST SUCCESS, AMDSMI API Returned 2, AMDSMI_STATUS_NOT_SUPPORTED
  //     3. If the API returnCode is not what was expected or not supported,
  //        the test will fail with
  //            TEST FAILURE, AMDSMI API Returned X1, AMDSMI_STATUS_XX1
  //                                     Expected X2, AMDSMI_STATUS_XX2
  //            where:
  //                 X1 API error code
  //                XX1 API error code string
  //                 X2 API exptected error code
  //                XX2 API exptected error code string
  //
  // TODO(amdsmi_team):
  //     1. Use this function to verify expected return codes and report failures
  //        to testing framework
  //     2. Upon failures, alter test flow
  //     3. For not supported API's, allow function to mark as failures where applicable

  int i;
  amdsmi_status_t retExpected[] = {args...};
  int numRetExpected = sizeof(retExpected) / sizeof(retExpected[0]);
  std::string status = smi_amdgpu_get_status_string(returnCode, false);
  amdsmi_status_t retExpectedStr = retExpected[0];

  // Check for successful (expected) return code
  for (i = 0; i < numRetExpected; ++i) {
    if (returnCode == retExpected[i]) {
      if (isVerbose)
        std::cout << "\t===> TEST SUCCESS, AMDSMI API Returned " << returnCode << ", " << status
                  << std::endl;
      return;
    }
  }

  //
  // Return code is not what was expected
  // Find and report error code
  //

  // Check if return code is in the not supported list
  int numNotSupportedErrorCodes =
      sizeof(NotSupportedErrorCodes) / sizeof(NotSupportedErrorCodes[0]);
  for (i = 0; i < numNotSupportedErrorCodes; ++i) {
    if (returnCode == NotSupportedErrorCodes[i]) {
      if (isVerbose)
        std::cout << "\t===> TEST SUCCESS, AMDSMI API Returned " << returnCode << ", " << status
                  << std::endl;
      return;
    }
  }

  //
  // Return code is not successful, print failure results
  //
  if (isVerbose) {
    std::string expectedStatus;
    std::cout << "\t===> TEST FAILURE, AMDSMI API Returned " << std::setfill(' ') << std::setw(2)
              << returnCode << ", " << status << std::endl;
    std::cout << "\t===>                          Expected ";
    if (numRetExpected == 1) {
      expectedStatus = smi_amdgpu_get_status_string(retExpectedStr, false);
      std::cout << std::setfill(' ') << std::setw(2) << std::right << retExpectedStr << ", "
                << expectedStatus << std::endl;
    } else {
      for (int i = 0; i < numRetExpected; ++i) {
        expectedStatus = smi_amdgpu_get_status_string(retExpected[i], false);
        if (i != 0) std::cout << "\t===>                                or ";
        std::cout << std::setfill(' ') << std::setw(2) << std::right << retExpected[i] << ", "
                  << expectedStatus << std::endl;
      }
      std::cout << std::endl;
    }
  }
  // Display file path starting from root directory
  if (isVerbose) {
    std::string start_dir = std::string(fileName);
    size_t pos = start_dir.find("tests/amd_smi_test");
    if (pos != std::string::npos) start_dir = start_dir.substr(pos);
    std::cout << "\t===> " << start_dir << ":" << std::dec << lineNum << std::endl;
  }
  return;
}

#endif  // TESTS_AMD_SMI_TEST_TEST_COMMON_H_
