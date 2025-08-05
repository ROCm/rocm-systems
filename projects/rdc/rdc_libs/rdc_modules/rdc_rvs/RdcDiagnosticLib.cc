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

#include <algorithm>
#include <memory>

#include "rdc/rdc.h"
#include "rdc_lib/RdcDiagnosticLibInterface.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rvs/RvsBase.h"

std::unique_ptr<amd::rdc::RdcRVSBase> rvs_p;

bool is_rvs_disabled() {
  const char* value = std::getenv("RDC_DISABLE_RVS");
  if (value == nullptr) return false;

  std::string value_str = value;
  std::transform(value_str.begin(), value_str.end(), value_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::vector<const char*> positive_list = {"yes", "true", "1", "on", "y", "t"};

  return std::any_of(positive_list.begin(), positive_list.end(),
                     [&value_str](const char* val) { return value_str == val; });
}

rdc_status_t rdc_diag_init(uint64_t) {
  if (is_rvs_disabled()) {
    return RDC_ST_DISABLED_MODULE;
  }
  rvs_p = std::unique_ptr<amd::rdc::RdcRVSBase>(new amd::rdc::RdcRVSBase);
  return RDC_ST_OK;
}

rdc_status_t rdc_diag_destroy() {
  rvs_p.reset();
  return RDC_ST_OK;
}

rdc_status_t rdc_diag_test_cases_query(rdc_diag_test_cases_t test_cases[MAX_TEST_CASES],
                                       uint32_t* test_case_count) {
  if (test_case_count == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  auto test_to_conf = rvs_p->get_test_to_conf();
  *test_case_count = test_to_conf.size();
  for (auto& [key, value] : test_to_conf) {
    *test_cases++ = key;
  }

  return RDC_ST_OK;
}

rdc_status_t rdc_diag_test_case_run(rdc_diag_test_cases_t test_case,
                                    // TODO: use gpu_index
                                    uint32_t gpu_index[RDC_MAX_NUM_DEVICES], uint32_t gpu_count,
                                    const char* config, size_t config_size,
                                    rdc_diag_test_result_t* result, rdc_diag_callback_t* callback) {
  const bool is_custom = config != nullptr && config_size != 0;

  rvs_status_t rvs_status = RVS_STATUS_SUCCESS;
  if (result == nullptr || gpu_count == 0) {
    return RDC_ST_BAD_PARAMETER;
  }

  if (rvs_p == nullptr) {
    RDC_LOG(RDC_ERROR, "rvs_p is not set!");
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  // get test_to_conf
  auto test_to_conf = rvs_p->get_test_to_conf();

  // init the return data
  *result = {};
  result->test_case = test_case;
  result->status = RDC_DIAG_RESULT_PASS;
  result->per_gpu_result_count = 0;

  if (callback != nullptr && callback->callback != nullptr && callback->cookie != nullptr) {
    std::string str = "RVS test [" + test_to_name.at(test_case) + "]";
    callback->callback(callback->cookie, str.data());
  }

  // if config is given - only run one test and return
  // do not care about test_case
  if (is_custom) {
    rvs_status = rvs_p->run_rvs_app(config, config_size + 1, callback);
    if (rvs_status != RVS_STATUS_SUCCESS) {
      result->status = RDC_DIAG_RESULT_FAIL;
    }
    return RDC_ST_OK;
  }

  switch (test_case) {
    case RDC_DIAG_RVS_GST_TEST:
    case RDC_DIAG_RVS_MEMBW_TEST:
    case RDC_DIAG_RVS_H2DD2H_TEST:
    case RDC_DIAG_RVS_IET_TEST:
    case RDC_DIAG_RVS_GST_LONG_TEST:
    case RDC_DIAG_RVS_MEMBW_LONG_TEST:
    case RDC_DIAG_RVS_H2DD2H_LONG_TEST:
    case RDC_DIAG_RVS_IET_LONG_TEST: {
      const std::string test_name = "Finished running " + test_to_name.at(test_case);
      if (test_to_conf.find(test_case) == test_to_conf.end()) {
        RDC_LOG(RDC_ERROR, "cannot find test " << test_to_name.at(test_case));
        return RDC_ST_NOT_FOUND;
      }
      const std::string predefined_config = test_to_conf.at(test_case);
      //  +1 to copy null
      strncpy_with_null(result->info, test_name.c_str(), test_name.length() + 1);
      rvs_status =
          rvs_p->run_rvs_app(predefined_config.c_str(), predefined_config.length() + 1, callback);
      break;
    }
    case RDC_DIAG_RVS_CUSTOM:
      RDC_LOG(RDC_ERROR, "custom config cannot be bundled with other tests!");
      result->status = RDC_DIAG_RESULT_SKIP;
      return RDC_ST_BAD_PARAMETER;
      break;
    default:
      result->status = RDC_DIAG_RESULT_SKIP;
      strncpy_with_null(result->info, "Not supported yet", MAX_DIAG_MSG_LENGTH);
      return RDC_ST_BAD_PARAMETER;
  }

  if (rvs_status != RVS_STATUS_SUCCESS) {
    result->status = RDC_DIAG_RESULT_FAIL;
  }

  return RDC_ST_OK;
}
