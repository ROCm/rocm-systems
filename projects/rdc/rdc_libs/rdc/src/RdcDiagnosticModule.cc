/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include "rdc_lib/impl/RdcDiagnosticModule.h"

#include <map>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

rdc_status_t RdcDiagnosticModule::rdc_diag_test_cases_query(
    rdc_diag_test_cases_t test_cases[MAX_TEST_CASES], uint32_t* test_case_count) {
  if (test_case_count == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  auto ite = diagnostic_modules_.begin();
  *test_case_count = 0;
  for (; ite != diagnostic_modules_.end(); ite++) {
    uint32_t count = 0;
    rdc_status_t status =
        (*ite)->rdc_diag_test_cases_query(&(test_cases[*test_case_count]), &count);
    if (status == RDC_ST_OK) {
      *test_case_count += count;
    }
  }
  return RDC_ST_OK;
}

rdc_status_t RdcDiagnosticModule::rdc_test_case_run(rdc_diag_test_cases_t test_case,
                                                    uint32_t gpu_index[RDC_MAX_NUM_DEVICES],
                                                    uint32_t gpu_count, const char* config,
                                                    size_t config_size,
                                                    rdc_diag_test_result_t* result,
                                                    rdc_diag_callback_t* callback) {
  if (result == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  // Init test status
  auto ite = testcases_to_module_.find(test_case);
  if (ite == testcases_to_module_.end()) {
    result->status = RDC_DIAG_RESULT_SKIP;
    strncpy_with_null(result->info, "Not implemented", MAX_DIAG_MSG_LENGTH);
    return RDC_ST_NOT_SUPPORTED;
  }

  rdc_status_t status = ite->second->rdc_test_case_run(test_case, gpu_index, gpu_count, config,
                                                       config_size, result, callback);
  return status;
}

rdc_status_t RdcDiagnosticModule::rdc_diagnostic_run(const rdc_group_info_t& gpus,
                                                     rdc_diag_level_t level, const char* config,
                                                     size_t config_size,
                                                     rdc_diag_response_t* response,
                                                     rdc_diag_callback_t* callback) {
  const bool is_custom = config != nullptr && config_size != 0;

  if (response == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  std::vector<rdc_diag_test_cases_t> tests_to_search_for;
  if (level >= RDC_DIAG_LVL_SHORT) {  // Short run and above
    tests_to_search_for.push_back(RDC_DIAG_COMPUTE_PROCESS);
    tests_to_search_for.push_back(RDC_DIAG_NODE_TOPOLOGY);
    tests_to_search_for.push_back(RDC_DIAG_GPU_PARAMETERS);
    tests_to_search_for.push_back(RDC_DIAG_COMPUTE_QUEUE);
    tests_to_search_for.push_back(RDC_DIAG_SYS_MEM_CHECK);
  }

  if (level >= RDC_DIAG_LVL_MED) {  // Medium run and above
    tests_to_search_for.push_back(RDC_DIAG_RVS_GST_TEST);
    tests_to_search_for.push_back(RDC_DIAG_RVS_MEMBW_TEST);
    tests_to_search_for.push_back(RDC_DIAG_RVS_H2DD2H_TEST);
    tests_to_search_for.push_back(RDC_DIAG_RVS_IET_TEST);
  }

  if (level >= RDC_DIAG_LVL_LONG) {  // Medium run and above
    tests_to_search_for.push_back(RDC_DIAG_RVS_GST_LONG_TEST);
    tests_to_search_for.push_back(RDC_DIAG_RVS_MEMBW_LONG_TEST);
    tests_to_search_for.push_back(RDC_DIAG_RVS_H2DD2H_LONG_TEST);
    tests_to_search_for.push_back(RDC_DIAG_RVS_IET_LONG_TEST);
  }

  std::vector<rdc_diag_test_cases_t> tests_to_run;
  if (is_custom) {
    // respect custom config
    tests_to_run.push_back(RDC_DIAG_RVS_CUSTOM);
  } else {
    // respect level
    for (auto& test : tests_to_search_for) {
      if (testcases_to_module_.find(test) != testcases_to_module_.end()) {
        tests_to_run.push_back(test);
      } else {
        RDC_LOG(RDC_DEBUG, "test not found: " << test);
      }
    }
  }

  if (callback != nullptr && callback->callback != nullptr && callback->cookie != nullptr) {
    std::string log = "DiagnosticRun started";
    callback->callback(callback->cookie, log.data());
  }

  unsigned int i = 0;
  response->results_count = 0;
  for (i = 0; i < tests_to_run.size(); i++) {
    if (callback != nullptr && callback->callback != nullptr && callback->cookie != nullptr) {
      std::string log =
          "Test " + std::to_string(i + 1) + " / " + std::to_string(tests_to_run.size());
      callback->callback(callback->cookie, log.data());
    }
    response->diag_info[i].test_case = tests_to_run[i];
    // NOTE: rdc_test_case_run reuses the diagnostic_run callback
    rdc_test_case_run(tests_to_run[i], const_cast<uint32_t*>(gpus.entity_ids), gpus.count, config,
                      config_size, &(response->diag_info[i]), callback);
    response->results_count++;
  }

  if (callback != nullptr && callback->callback != nullptr && callback->cookie != nullptr) {
    std::string log = "DiagnosticRun finished";
    callback->callback(callback->cookie, log.data());
  }

  return RDC_ST_OK;
}

rdc_status_t RdcDiagnosticModule::rdc_diag_init(uint64_t flag) {
  auto ite = diagnostic_modules_.begin();
  for (; ite != diagnostic_modules_.end(); ite++) {
    (*ite)->rdc_diag_init(flag);
  }
  return RDC_ST_OK;
}

rdc_status_t RdcDiagnosticModule::RdcDiagnosticModule::rdc_diag_destroy() {
  auto ite = diagnostic_modules_.begin();
  for (; ite != diagnostic_modules_.end(); ite++) {
    (*ite)->rdc_diag_destroy();
  }
  return RDC_ST_OK;
}

RdcDiagnosticModule::RdcDiagnosticModule(std::list<RdcDiagnosticPtr> diagnostic_modules)
    : diagnostic_modules_(diagnostic_modules) {
  // find test cases
  for (auto& ite : diagnostic_modules_) {
    rdc_diag_test_cases_t test_cases[MAX_TEST_CASES];
    uint32_t test_count = 0;
    rdc_status_t status = ite->rdc_diag_test_cases_query(test_cases, &test_count);
    if (status == RDC_ST_OK) {
      for (uint32_t index = 0; index < test_count; index++) {
        testcases_to_module_.insert({test_cases[index], ite});
      }
    }
  }
}

}  // namespace rdc
}  // namespace amd
