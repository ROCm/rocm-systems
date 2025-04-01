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

#include <string.h>
#include <unistd.h>

#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#include "rdc/rdc.h"

static std::string get_test_name(rdc_diag_test_cases_t test_case) {
  const std::map<rdc_diag_test_cases_t, std::string> test_desc = {
      {RDC_DIAG_COMPUTE_PROCESS, "No compute process"},
      {RDC_DIAG_COMPUTE_QUEUE, "Compute Queue ready"},
      {RDC_DIAG_SYS_MEM_CHECK, "System memory check"},
      {RDC_DIAG_NODE_TOPOLOGY, "Node topology check"},
      {RDC_DIAG_RVS_GST_TEST, "RVS check"},
      {RDC_DIAG_GPU_PARAMETERS, "GPU parameters check"},
      {RDC_DIAG_TEST_LAST, "Unknown"}};

  auto test_name = test_desc.find(test_case);
  if (test_name == test_desc.end()) {
    return "Unknown Test";
  }
  return test_name->second;
}

int main(int, char**) {
  rdc_status_t result;
  rdc_handle_t rdc_handle;
  bool standalone = false;
  char hostIpAddress[] = {"127.0.0.1:50051"};
  char group_name[] = {"diag_group"};

  // Select the embedded mode and standalone mode dynamically.
  std::cout << "Start rdci in: \n";
  std::cout << "0 - Embedded mode \n";
  std::cout << "1 - Standalone mode \n";
  while (!(std::cin >> standalone)) {
    std::cout << "Invalid input.\n";
    std::cin.clear();
    std::cin.ignore();
  }
  std::cout << std::endl;
  std::cout << (standalone ? "Standalone mode selected.\n" : "Embedded mode selected.\n");

  // Init the rdc
  result = rdc_init(0);

  if (result != RDC_ST_OK) {
    std::cout << "Error initializing RDC. Return: " << rdc_status_string(result) << std::endl;
    goto cleanup;
  } else {
    std::cout << "RDC Initialized.\n";
  }

  if (standalone) {  // standalone
    result = rdc_connect(hostIpAddress, &rdc_handle, nullptr, nullptr, nullptr);
    if (result != RDC_ST_OK) {
      std::cout << "Error connecting to remote rdcd. Return: " << rdc_status_string(result)
                << std::endl;
      goto cleanup;
    }
  } else {  // embedded
    result = rdc_start_embedded(RDC_OPERATION_MODE_AUTO, &rdc_handle);
    if (result != RDC_ST_OK) {
      std::cout << "Error starting embedded RDC engine. Return: " << rdc_status_string(result)
                << std::endl;
      goto cleanup;
    }
  }

  // Now we can use the same API for both standalone and embedded
  // (1) create group for all GPUs
  rdc_gpu_group_t group_id;
  result = rdc_group_gpu_create(rdc_handle, RDC_GROUP_DEFAULT, group_name, &group_id);
  if (result != RDC_ST_OK) {
    std::cout << "Error creating group. Return: " << rdc_status_string(result);
    goto cleanup;
  }

  // (2) start to run short diagnostic.
  rdc_diag_response_t response;
  result = rdc_diagnostic_run(rdc_handle, group_id, RDC_DIAG_LVL_SHORT, nullptr, 0, &response, nullptr);

  if (result != RDC_ST_OK) {
    std::cout << "Error run RDC_DIAG_LVL_SHORT diagnostic. Return: " << rdc_status_string(result);
    goto cleanup;
  }

  // (3) Check diagnostic results
  for (uint32_t i = 0; i < response.results_count; i++) {
    const rdc_diag_test_result_t& test_result = response.diag_info[i];
    std::cout << std::setw(22) << std::left << get_test_name(test_result.test_case) + ":"
              << rdc_diagnostic_result_string(test_result.status) << "\n";
  }

  // (4) diagnostic detail information
  std::cout << " =============== Diagnostic Details ==================\n";
  for (uint32_t i = 0; i < response.results_count; i++) {
    const rdc_diag_test_result_t& test_result = response.diag_info[i];
    if (test_result.info[0] != '\0') {
      std::cout << std::setw(22) << std::left << get_test_name(test_result.test_case) + ":"
                << test_result.info << "\n";
    }
    for (uint32_t j = 0; j < test_result.per_gpu_result_count; j++) {
      const rdc_diag_per_gpu_result_t& gpu_result = test_result.gpu_results[j];
      if (strlen(gpu_result.gpu_result.msg) > 0) {
        std::cout << " GPU " << gpu_result.gpu_index << " " << gpu_result.gpu_result.msg << "\n";
      }
    }
  }

  // (5) run one test case
  std::cout << " ============== Run individual diagnostic test ===========\n";
  rdc_diag_test_result_t test_result;
  result =
      rdc_test_case_run(rdc_handle, group_id, RDC_DIAG_COMPUTE_PROCESS, nullptr, 0, &test_result, nullptr);

  if (result != RDC_ST_OK) {
    std::cout << "Error run RDC_DIAG_COMPUTE_PROCESS diagnostic. Return: "
              << rdc_status_string(result);
    goto cleanup;
  }

  std::cout << std::setw(22) << std::left << get_test_name(RDC_DIAG_COMPUTE_PROCESS) + ":"
            << test_result.info << "\n";

// Cleanup consists of shutting down RDC.
cleanup:
  std::cout << "Cleaning up.\n";
  if (standalone)
    rdc_disconnect(rdc_handle);
  else
    rdc_stop_embedded(rdc_handle);
  rdc_shutdown();
  return result;
}
