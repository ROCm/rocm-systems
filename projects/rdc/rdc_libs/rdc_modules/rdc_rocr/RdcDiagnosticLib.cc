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

#include <string>

#include "rdc_lib/RdcDiagnosticLibInterface.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rocr/ComputeQueueTest.h"
#include "rdc_modules/rdc_rocr/MemoryAccess.h"
#include "rdc_modules/rdc_rocr/MemoryTest.h"
#include "rdc_modules/rdc_rocr/common.h"

rdc_status_t rdc_diag_init(uint64_t) { return RDC_ST_OK; }

rdc_status_t rdc_diag_destroy() { return RDC_ST_OK; }

rdc_status_t rdc_diag_test_cases_query(rdc_diag_test_cases_t test_cases[MAX_TEST_CASES],
                                       uint32_t* test_case_count) {
  if (test_case_count == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  *test_case_count = 2;
  test_cases[0] = RDC_DIAG_COMPUTE_QUEUE;
  test_cases[1] = RDC_DIAG_SYS_MEM_CHECK;

  return RDC_ST_OK;
}

// Helper function to run the memory test on GPU
static rdc_status_t run_memory_test(uint32_t gpu_index, rdc_diag_test_result_t* result) {
  std::string info = result->info;
  std::string per_gpu_info = result->gpu_results[gpu_index].gpu_result.msg;

  try {
    amd::rdc::MemoryTest test(gpu_index);
    test.MaxSingleAllocationTest();

    info += test.get_gpu_info();
    per_gpu_info += test.get_per_gpu_info();
  } catch (const amd::rdc::SkipException& e) {
    result->status = RDC_DIAG_RESULT_SKIP;
    per_gpu_info += "MaxSingleAllocationTest is skipped: ";
    per_gpu_info += e.what();
    info += "GPU ";
    info += std::to_string(gpu_index);
    info += " MaxSingleAllocationTest is skipped: ";
    info += e.what();
    info += ".";
  } catch (const std::exception& e) {
    result->status = RDC_DIAG_RESULT_FAIL;
    per_gpu_info += "MaxSingleAllocationTest returns with error ";
    per_gpu_info += e.what();
    info += "GPU ";
    info += std::to_string(gpu_index);
    info += " MaxSingleAllocationTest returns with error ";
    info += e.what();
    info += ".";
  }

  try {
    amd::rdc::MemoryAccessTest test(gpu_index);
    test.CPUAccessToGPUMemoryTest();
    test.GPUAccessToCPUMemoryTest();
    info += test.get_gpu_info();
    per_gpu_info += test.get_per_gpu_info();
  } catch (const amd::rdc::SkipException& e) {
    result->status = RDC_DIAG_RESULT_SKIP;
    per_gpu_info += "Memory Access is skipped: ";
    per_gpu_info += e.what();
    info += "GPU ";
    info += std::to_string(gpu_index);
    info += " Memory Access is skipped: ";
    info += e.what();
    info += ".";
  } catch (const std::exception& e) {
    result->status = RDC_DIAG_RESULT_FAIL;
    per_gpu_info += "Memory Access returns with error ";
    per_gpu_info += e.what();
    info += "GPU ";
    info += std::to_string(gpu_index);
    info += " Memory Access returns with error ";
    info += e.what();
    info += ".";
  }

  strncpy_with_null(result->info, info.c_str(), MAX_DIAG_MSG_LENGTH);
  strncpy_with_null(result->gpu_results[gpu_index].gpu_result.msg, per_gpu_info.c_str(),
                    MAX_DIAG_MSG_LENGTH);

  return RDC_ST_OK;
}

static rdc_status_t run_compute_queue_test(uint32_t gpu_index, rdc_diag_test_result_t* result) {
  std::string info = result->info;
  std::string per_gpu_info = result->gpu_results[gpu_index].gpu_result.msg;

  try {
    amd::rdc::ComputeQueueTest test(gpu_index);
    test.RunBinarySearchTest();
    info += test.get_gpu_info();
    per_gpu_info += test.get_per_gpu_info();
  } catch (const amd::rdc::SkipException& e) {
    result->status = RDC_DIAG_RESULT_SKIP;
    per_gpu_info += "Compute Queue test is skipped: ";
    per_gpu_info += e.what();
    info += "GPU ";
    info += std::to_string(gpu_index);
    info += " Compute Queue test is skipped: ";
    info += e.what();
    info += ".";
  } catch (const std::exception& e) {
    result->status = RDC_DIAG_RESULT_FAIL;
    per_gpu_info += "Compute Queue test returns with error ";
    per_gpu_info += e.what();
    info += "GPU ";
    info += std::to_string(gpu_index);
    info += " Compute Queue test returns with error ";
    info += e.what();
    info += ".";
  }

  strncpy_with_null(result->info, info.c_str(), MAX_DIAG_MSG_LENGTH);
  strncpy_with_null(result->gpu_results[gpu_index].gpu_result.msg, per_gpu_info.c_str(),
                    MAX_DIAG_MSG_LENGTH);

  return RDC_ST_OK;
}

rdc_status_t rdc_diag_test_case_run(rdc_diag_test_cases_t test_case,
                                    uint32_t gpu_index[RDC_MAX_NUM_DEVICES], uint32_t gpu_count,
                                    const char* /*config*/, size_t /*config_size*/,
                                    rdc_diag_test_result_t* result, rdc_diag_callback_t* callback) {
  if (result == nullptr || gpu_count == 0) {
    return RDC_ST_BAD_PARAMETER;
  }

  if (test_case != RDC_DIAG_COMPUTE_QUEUE && test_case != RDC_DIAG_SYS_MEM_CHECK) {
    return RDC_ST_OK;
  }

  // init the return data
  *result = {};
  result->test_case = test_case;
  result->status = RDC_DIAG_RESULT_PASS;
  result->per_gpu_result_count = 0;

  // Run test for each GPU. It will continue even
  // if one GPU test is fail.
  for (uint32_t i = 0; i < gpu_count; i++) {
    if (callback != nullptr && callback->callback != nullptr && callback->cookie != nullptr) {
      std::string str = "ROCR test on GPU " + std::to_string(i);
      callback->callback(callback->cookie, str.data());
    }
    switch (test_case) {
      case RDC_DIAG_SYS_MEM_CHECK:
        run_memory_test(gpu_index[i], result);
        break;
      case RDC_DIAG_COMPUTE_QUEUE:
        run_compute_queue_test(gpu_index[i], result);
        break;
      default:
        result->status = RDC_DIAG_RESULT_SKIP;
        strncpy_with_null(result->info, "Not supported yet", MAX_DIAG_MSG_LENGTH);
    }
  }

  return RDC_ST_OK;
}
