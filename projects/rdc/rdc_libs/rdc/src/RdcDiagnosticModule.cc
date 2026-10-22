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

#include <algorithm>
#include <array>
#include <iterator>
#include <map>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {
namespace {

// The tests of a short run, and of every longer one.
constexpr std::array<rdc_diag_test_cases_t, 5> kShortRunTests = {
    RDC_DIAG_COMPUTE_PROCESS, RDC_DIAG_NODE_TOPOLOGY, RDC_DIAG_GPU_PARAMETERS,
    RDC_DIAG_COMPUTE_QUEUE, RDC_DIAG_SYS_MEM_CHECK};
static_assert(kShortRunTests.size() <= static_cast<size_t>(MAX_TEST_CASES),
              "a run's results must fit in rdc_diag_response_t::diag_info");

// A module's test-case array starts out filled with RDC_DIAG_TEST_LAST, which no test case has, so
// an entry that the module reports but does not write is not taken for a test case.
bool is_test_case(rdc_diag_test_cases_t test_case) {
  const int value = static_cast<int>(test_case);
  return value >= RDC_DIAG_TEST_FIRST && value < RDC_DIAG_TEST_LAST;
}

}  // namespace

rdc_status_t RdcDiagnosticModule::rdc_diag_test_cases_query(
    rdc_diag_test_cases_t test_cases[MAX_TEST_CASES], uint32_t* test_case_count) {
  if (test_case_count == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  auto ite = diagnostic_modules_.begin();
  *test_case_count = 0;
  for (; ite != diagnostic_modules_.end(); ite++) {
    // Each module writes into an array of its own, as its parameter promises.
    rdc_diag_test_cases_t module_cases[MAX_TEST_CASES];
    std::fill(std::begin(module_cases), std::end(module_cases), RDC_DIAG_TEST_LAST);
    uint32_t count = 0;
    rdc_status_t status = (*ite)->rdc_diag_test_cases_query(module_cases, &count);
    if (status != RDC_ST_OK) {
      continue;
    }
    count = std::min<uint32_t>(count, MAX_TEST_CASES);
    for (uint32_t i = 0; i < count; i++) {
      if (!is_test_case(module_cases[i])) {
        continue;
      }
      if (*test_case_count < MAX_TEST_CASES) {
        test_cases[(*test_case_count)++] = module_cases[i];
      } else {
        RDC_LOG(RDC_ERROR, "More than " << MAX_TEST_CASES << " test cases, dropped test case "
                                        << module_cases[i]);
      }
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

  // Init test status: a module that returns without writing the result leaves it skipped.
  *result = {};
  result->test_case = test_case;
  result->status = RDC_DIAG_RESULT_SKIP;
  auto ite = testcases_to_module_.find(test_case);
  if (ite == testcases_to_module_.end()) {
    strncpy_with_null(result->info, "Not implemented", MAX_DIAG_MSG_LENGTH);
    // The Skip result says so; rdcd sends no result for any other status.
    return RDC_ST_OK;
  }

  // gpu_index holds a group's entities, at most RDC_GROUP_MAX_ENTITIES.
  gpu_count = std::min<uint32_t>(gpu_count, RDC_GROUP_MAX_ENTITIES);
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

  std::vector<rdc_diag_test_cases_t> tests_to_run;
  if (is_custom) {
    RDC_LOG(RDC_DEBUG, "Custom diagnostic run not supported yet");
  } else if (level >= RDC_DIAG_LVL_SHORT) {  // Short run and above
    // respect level
    for (auto& test : kShortRunTests) {
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
    // NOTE: rdc_test_case_run reuses the diagnostic_run callback; it also clamps the count. Its
    // status is not the run's: each test's outcome is in its result, and rdcd sends no results
    // for a run that does not return RDC_ST_OK.
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
    std::fill(std::begin(test_cases), std::end(test_cases), RDC_DIAG_TEST_LAST);
    uint32_t test_count = 0;
    rdc_status_t status = ite->rdc_diag_test_cases_query(test_cases, &test_count);
    if (status == RDC_ST_OK) {
      test_count = std::min<uint32_t>(test_count, MAX_TEST_CASES);
      for (uint32_t index = 0; index < test_count; index++) {
        if (is_test_case(test_cases[index])) {
          testcases_to_module_.insert({test_cases[index], ite});
        }
      }
    } else {
      // The only line that explains the tests missing from `rdci diag`: the module's own loader
      // may have logged nothing, and the query made for each client request stays silent. The
      // dispatcher is built once, so one failing module logs one line.
      RDC_LOG(RDC_ERROR, "A diagnostic module's test list could not be read (status "
                             << status << "); its tests are not run.");
    }
  }
}

}  // namespace rdc
}  // namespace amd
