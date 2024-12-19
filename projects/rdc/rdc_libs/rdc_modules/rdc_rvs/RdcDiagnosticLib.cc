/*
Copyright (c) 2023 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include <string.h>

#include "rdc/rdc.h"
#include "rdc_lib/RdcDiagnosticLibInterface.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rvs/RvsBase.h"

rdc_status_t rdc_diag_init(uint64_t) { return RDC_ST_OK; }

rdc_status_t rdc_diag_destroy() { return RDC_ST_OK; }

rdc_status_t rdc_diag_test_cases_query(rdc_diag_test_cases_t test_cases[MAX_TEST_CASES],
                                       uint32_t* test_case_count) {
  if (test_case_count == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  *test_case_count = 1;
  test_cases[0] = RDC_DIAG_RVS_TEST;

  return RDC_ST_OK;
}

rdc_status_t rdc_diag_test_case_run(rdc_diag_test_cases_t test_case,
                                    // TODO: use gpu_index
                                    uint32_t gpu_index[RDC_MAX_NUM_DEVICES], uint32_t gpu_count,
                                    const char* config, size_t config_size,
                                    rdc_diag_test_result_t* result, rdc_diag_callback_t* callback) {
  rvs_status_t rvs_status = RVS_STATUS_SUCCESS;
  if (result == nullptr || gpu_count == 0) {
    return RDC_ST_BAD_PARAMETER;
  }

  if (test_case != RDC_DIAG_RVS_TEST) {
    return RDC_ST_BAD_PARAMETER;
  }

  amd::rdc::RdcRVSBase rvs_base;

  // init the return data
  *result = {};
  result->test_case = test_case;
  result->status = RDC_DIAG_RESULT_PASS;
  result->per_gpu_result_count = 0;

  if (callback != nullptr && callback->callback != nullptr && callback->cookie != nullptr) {
    std::string str = "RVS test";
    callback->callback(callback->cookie, str.data());
  }
  switch (test_case) {
    case RDC_DIAG_RVS_TEST:
      strncpy_with_null(result->info, "Finished running RDC_DIAG_RVS_TEST", MAX_DIAG_MSG_LENGTH);
      rvs_status = rvs_base.run_rvs_app(config, config_size, callback);
      break;
    default:
      result->status = RDC_DIAG_RESULT_SKIP;
      strncpy_with_null(result->info, "Not supported yet", MAX_DIAG_MSG_LENGTH);
  }

  if (rvs_status != RVS_STATUS_SUCCESS) {
    result->status = RDC_DIAG_RESULT_FAIL;
  }

  return RDC_ST_OK;
}
