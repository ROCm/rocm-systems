/*
Copyright (c) 2019 - Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef TESTS_RDC_TESTS_TEST_COMMON_H_
#define TESTS_RDC_TESTS_TEST_COMMON_H_

#include <memory>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"

struct RDCTstGlobals {
  uint32_t verbosity;
  uint32_t monitor_verbosity;
  std::string rdcd_path;
  std::string monitor_server_port;
  std::string monitor_server_ip;
  uint32_t num_iterations;
  bool dont_fail;
  bool secure;
  bool standalone;
  bool batch_mode;
};

uint32_t ProcessCmdline(RDCTstGlobals* test, int arg_cnt, char** arg_list);

void PrintTestHeader(uint32_t dv_ind);
const char* GetBlockNameStr(amdsmi_gpu_block_t id);
const char* GetErrStateNameStr(amdsmi_ras_err_state_t st);
// const char *GetGRPCChanStateStr(grpc_connectivity_state st);
const char* FreqEnumToStr(amdsmi_clk_type_t rsmi_clk);

#if ENABLE_SMI
void DumpMonitorInfo(const TestBase* test);
#endif

#define DISPLAY_RDC_ERR(RET)                                                             \
  {                                                                                      \
    if (RET != RDC_STATUS_SUCCESS) {                                                     \
      const char* err_str;                                                               \
      std::cout << "\t===> ERROR: RDC call returned " << (RET) << std::endl;             \
      rdc_status_string((RET), &err_str);                                                \
      std::cout << "\t===> (" << err_str << ")" << std::endl;                            \
      std::cout << "\t===> at " << __FILE__ << ":" << std::dec << __LINE__ << std::endl; \
    }                                                                                    \
  }

#define CHK_ERR_RET(RET)               \
  {                                    \
    DISPLAY_RDC_ERR(RET)               \
    if ((RET) != RDC_STATUS_SUCCESS) { \
      return (RET);                    \
    }                                  \
  }
#define CHK_RDC_PERM_ERR(RET)                                         \
  {                                                                   \
    if (RET == RDC_STATUS_PERMISSION) {                               \
      std::cout << "This command requires root access." << std::endl; \
    } else {                                                          \
      DISPLAY_RDC_ERR(RET)                                            \
    }                                                                 \
  }

#endif  // TESTS_RDC_TESTS_TEST_COMMON_H_
