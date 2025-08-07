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
#include "rdc_lib/impl/RdcRocrLib.h"

#include <functional>

#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdcRocrLib::RdcRocrLib()
    : test_case_run_(nullptr),
      diag_test_cases_query_(nullptr),
      diag_init_(nullptr),
      diag_destroy_(nullptr) {
  rdc_status_t status = lib_loader_.load("librdc_rocr.so");
  if (status != RDC_ST_OK) {
    RDC_LOG(RDC_ERROR, "Rocr related function will not work.");
    return;
  }

  status = lib_loader_.load_symbol(&diag_init_, "rdc_diag_init");
  if (status != RDC_ST_OK) {
    diag_init_ = nullptr;
    return;
  }

  status = diag_init_(0);
  if (status != RDC_ST_OK) {
    RDC_LOG(RDC_ERROR, "Fail to init librdc_rocr.so:" << rdc_status_string(status)
                                                      << ". Rocr related function will not work.");
    return;
  }

  status = lib_loader_.load_symbol(&diag_destroy_, "rdc_diag_destroy");
  if (status != RDC_ST_OK) {
    diag_destroy_ = nullptr;
  }

  status = lib_loader_.load_symbol(&test_case_run_, "rdc_diag_test_case_run");
  if (status != RDC_ST_OK) {
    test_case_run_ = nullptr;
  }
  status = lib_loader_.load_symbol(&diag_test_cases_query_, "rdc_diag_test_cases_query");
  if (status != RDC_ST_OK) {
    diag_test_cases_query_ = nullptr;
  }
}

RdcRocrLib::~RdcRocrLib() {
  if (diag_destroy_) {
    diag_destroy_();
  }
}

rdc_status_t RdcRocrLib::rdc_diag_test_cases_query(rdc_diag_test_cases_t test_cases[MAX_TEST_CASES],
                                                   uint32_t* test_case_count) {
  if (test_case_count == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  if (!diag_test_cases_query_) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  rdc_status_t status = diag_test_cases_query_(test_cases, test_case_count);
  RDC_LOG(RDC_DEBUG,
          "Query " << *test_case_count << " test cases from Rocr: " << rdc_status_string(status));
  return status;
}

// Run a specific test case
rdc_status_t RdcRocrLib::rdc_test_case_run(rdc_diag_test_cases_t test_case,
                                           uint32_t gpu_index[RDC_MAX_NUM_DEVICES],
                                           uint32_t gpu_count, const char* config,
                                           size_t config_size, rdc_diag_test_result_t* result, rdc_diag_callback_t* callback) {
  if (result == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  if (!test_case_run_) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  rdc_status_t status =
      test_case_run_(test_case, gpu_index, gpu_count, config, config_size, result, callback);
  RDC_LOG(RDC_DEBUG, "Run " << test_case << " test case from Rocr: " << rdc_status_string(status));
  return status;
}

rdc_status_t RdcRocrLib::rdc_diagnostic_run(const rdc_group_info_t& gpus, rdc_diag_level_t level,
                                            const char* config, size_t config_size,
                                            rdc_diag_response_t* response, rdc_diag_callback_t* callback) {
  (void)gpus;
  (void)level;
  (void)config;
  (void)config_size;
  (void)response;
  (void)callback;
  return RDC_ST_NOT_SUPPORTED;
}

rdc_status_t RdcRocrLib::rdc_diag_init(uint64_t flags) {
  if (!diag_init_) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  return diag_init_(flags);
}
rdc_status_t RdcRocrLib::rdc_diag_destroy() {
  if (!diag_destroy_) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  return diag_destroy_();
}

}  // namespace rdc
}  // namespace amd
