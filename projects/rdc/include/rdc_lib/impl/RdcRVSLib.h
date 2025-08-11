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
#ifndef INCLUDE_RDC_LIB_IMPL_RDCRVSLIB_H_
#define INCLUDE_RDC_LIB_IMPL_RDCRVSLIB_H_

#include <memory>

#include "rdc/rdc.h"
#include "rdc_lib/RdcDiagnostic.h"
#include "rdc_lib/RdcLibraryLoader.h"

namespace amd {
namespace rdc {

class RdcRVSLib : public RdcDiagnostic {
 public:
  rdc_status_t rdc_diag_test_cases_query(rdc_diag_test_cases_t test_cases[MAX_TEST_CASES],
                                         uint32_t* test_case_count) override;

  // Run a specific test case
  rdc_status_t rdc_test_case_run(rdc_diag_test_cases_t test_case,
                                 uint32_t gpu_index[RDC_MAX_NUM_DEVICES], uint32_t gpu_count,
                                 const char* config, size_t config_size,
                                 rdc_diag_test_result_t* result, rdc_diag_callback_t* callback) override;

  rdc_status_t rdc_diagnostic_run(const rdc_group_info_t& gpus, rdc_diag_level_t level,
                                  const char* config, size_t config_size,
                                  rdc_diag_response_t* response, rdc_diag_callback_t* callback) override;

  rdc_status_t rdc_diag_init(uint64_t flags) override;
  rdc_status_t rdc_diag_destroy() override;

  RdcRVSLib();
  ~RdcRVSLib() override;

 private:
  RdcLibraryLoader lib_loader_;
  rdc_status_t (*test_case_run_)(rdc_diag_test_cases_t, uint32_t[RDC_MAX_NUM_DEVICES], uint32_t,
                                 const char* config, size_t config_size, rdc_diag_test_result_t*,
                                 rdc_diag_callback_t*);
  rdc_status_t (*diag_test_cases_query_)(rdc_diag_test_cases_t[MAX_TEST_CASES], uint32_t*);
  rdc_status_t (*diag_init_)(uint64_t);
  rdc_status_t (*diag_destroy_)();
};

typedef std::shared_ptr<RdcRVSLib> RdcRVSLibPtr;

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RDCRVSLIB_H_
