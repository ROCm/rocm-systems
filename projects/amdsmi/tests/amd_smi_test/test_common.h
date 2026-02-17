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
const char* GetPerfLevelStr(amdsmi_dev_perf_level_t lvl);
const char* GetBlockNameStr(amdsmi_gpu_block_t id);
const char* GetErrStateNameStr(amdsmi_ras_err_state_t st);
const char* FreqEnumToStr(amdsmi_clk_type_t amdsmi_clk);
const std::string GetVoltSensorNameStr(amdsmi_voltage_type_t st);

#if ENABLE_SMI
void DumpMonitorInfo(const TestBase* test);
#endif

static amdsmi_status_t NotSupportedErrorCodes[] = {
    AMDSMI_STATUS_NOT_SUPPORTED,
    AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
    AMDSMI_STATUS_NO_HSMP_MSG_SUP
};

inline void DISPLAY_AMDSMI_API(std::string func_name, std::string desc) {
    std::cout << "\t### " << (func_name) << "(" << (desc) << ")" << std::endl;
    return;
}

inline std::string GetErrorCode(amdsmi_status_t returnCode) {
    size_t pos;
    const char *_err;
    std::string err_str;

    // Gets error code with code description
    amdsmi_status_code_to_string(returnCode, &_err);
    err_str = std::string(_err);
    std::string status = !err_str.empty() ? err_str : "Unknown";

    // Just want error code, remove error code description
    pos = status.find(":");
    if (pos != std::string::npos)
        status = status.substr(0, pos);

    return (status);
}

template<typename T, typename... Args>
inline void DISPLAY_AMDSMI_STATUS(T returnCode, Args... args) {
    // Input:
    //     RET: API return code
    //     ...: Expected API return code(s)
    // Output:
    //     print results

    int i;
    amdsmi_status_t retExpected[] = {args...};
    int numRetExpected = sizeof(retExpected) / sizeof(retExpected[0]);

    std::string status = GetErrorCode(returnCode);
    amdsmi_status_t retExpectedStr = retExpected[0];

    // Check for successful (expected) return code
    for (i=0; i<numRetExpected; ++i) {
        if (returnCode == retExpected[i]) {
            std::cout << "\t===> TEST SUCCESS, AMDSMI API Returned " << returnCode << ", " << status << std::endl;
            return;
        }
    }

    //
    // Return code is not what was expected
    // Find and report error code
    //

    // Check if return code is in the not supported list
    int numNotSupportedErrorCodes = sizeof(NotSupportedErrorCodes) / sizeof(NotSupportedErrorCodes[0]);
    for (i=0; i<numNotSupportedErrorCodes ; ++i) {
        if (returnCode == NotSupportedErrorCodes[i]) {
            std::cout << "\t===> TEST SUCCESS, AMDSMI API Returned " << returnCode << ", " << status << std::endl;
            return;
        }
    }

    //
    // Return code is not successful, print failure results
    //
    std::cout << "\t===> TEST FAILURE, AMDSMI API Returned " << std::setfill(' ') << std::setw(2) << returnCode << ", " << status << std::endl;
    std::cout << "\t===>                          Expected ";
    if (numRetExpected == 1) {
        std::string expectedStatus = GetErrorCode(retExpectedStr);
        std::cout << std::setfill(' ') << std::setw(2) << retExpectedStr << ", " << expectedStatus << std::endl;
    }
    else {
        std::cout << "One of";
        for (int i=0; i<numRetExpected; ++i)
            std::cout << " " << retExpected[i];
        std::cout << std::endl;
    }
    // Display file path starting from root directory
    std::string start_dir = std::string(__FILE__);
    int pos = start_dir.find("tests/amd_smi_test");
    if (pos != std::string::npos)
        start_dir = start_dir.substr(pos);
    std::cout << "\t===> " << start_dir << ":" << std::dec << __LINE__ << std::endl;

    return;
}

#endif  // TESTS_AMD_SMI_TEST_TEST_COMMON_H_

