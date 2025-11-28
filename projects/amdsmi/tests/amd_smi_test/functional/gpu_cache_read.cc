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

#include "gpu_cache_read.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>

#include "../test_common.h"
#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "gpu_metrics_read.h"
#include "rocm_smi/rocm_smi_utils.h"

TestGPUCacheRead::TestGPUCacheRead() : TestBase() {
  set_title("GPU Cache Read Test");
  set_description(
      "This test verifies the GPU cache "
      "read metrics using the AMD SMI library.");
}

TestGPUCacheRead::~TestGPUCacheRead(void) {
  // Cleanup if necessary
}

void TestGPUCacheRead::SetUp() {
  TestBase::SetUp();
  return;
}

void TestGPUCacheRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestGPUCacheRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestGPUCacheRead::Close() {
  /**
   * @brief Closes the TestGPUCacheRead test case and performs necessary cleanup.
   *
   * This function overrides the Close method from the TestBase class.
   * It is responsible for executing any cleanup operations required after
   * running the GPU cache read test. The function calls the base class's
   * Close method to ensure all inherited cleanup procedures are executed.
   */
  TestBase::Close();
  return;
}

void TestGPUCacheRead::Run() {
  /**
   * @brief Runs the GPU cache read test.
   *
   * This function overrides the Run method from the TestBase class.
   * It is responsible for executing the GPU cache read test using the
   * AMD SMI library. The function retrieves the GPU cache read metrics
   * and displays them.
   */
  amdsmi_status_t err;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);
    std::cout << "Device #" << std::to_string(i) << "\n";

    IF_VERB(STANDARD) {
      std::cout << "\n\n";
      std::cout << "\t**GPU CACHE INFO: Using static struct (Backwards Compatibility):\n";
    }
    amdsmi_gpu_cache_info_t res = {};
    err = amdsmi_get_gpu_cache_info(processor_handles_[i], &res);
    const char *status_string;
    amdsmi_status_code_to_string(err, &status_string);
    std::cout << "\t\t** amdsmi_get_gpu_cache_info(): " << status_string << "\n";
    CHK_ERR_ASRT(err);
    std::cout << "\t\tnum_cache_types: " << res.num_cache_types << "\n";
    for (unsigned int j = 0; j < res.num_cache_types; j++) {
      std::cout << "\t\tCache Type " << j << ":\n";
      std::cout << "\t\t\tcache_level: " << res.cache[j].cache_level << "\n";
      std::cout << "\t\t\tcache_properties: (0x" << std::hex << res.cache[j].cache_properties
                << std::dec << ") ";

      // Example string representation (adjust according to actual bit definitions)
      std::string props_str;
      uint32_t props = res.cache[j].cache_properties;
      if (props & AMDSMI_CACHE_PROPERTY_DATA_CACHE) props_str += "Data Cache, ";
      if (props & AMDSMI_CACHE_PROPERTY_INST_CACHE) props_str += "Instruction Cache, ";
      if (props & AMDSMI_CACHE_PROPERTY_CPU_CACHE) props_str += "CPU Cache, ";
      if (props & AMDSMI_CACHE_PROPERTY_SIMD_CACHE) props_str += "SIMD Cache, ";
      if (!props_str.empty())
        props_str.erase(props_str.size() - 2);  // Remove trailing comma and space
      else
        props_str = "None";
      std::cout << props_str << "\n";
      std::cout << "\t\t\tcache_size: " << res.cache[j].cache_size << " KB\n";
      std::cout << "\t\t\tmax_num_cu_shared: " << res.cache[j].max_num_cu_shared << "\n";
      std::cout << "\t\t\tnum_cache_instance: " << res.cache[j].num_cache_instance << "\n";
    }
  }
}