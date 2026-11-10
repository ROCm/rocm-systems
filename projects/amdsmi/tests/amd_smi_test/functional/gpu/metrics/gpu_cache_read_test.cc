// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
#include <tuple>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "gpu_metrics_read.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "test_common.h"

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
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);
    std::cout << "Device #" << std::to_string(i) << "\n";

    IF_VERB(STANDARD) { std::cout << "\t**GPU CACHE INFO: Using caller-sized buffer:\n"; }
    uint32_t num_caches = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_cache_entries", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_cache_entries(processor_handles_[i], &num_caches, nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(err);
    ASSERT_GT(num_caches, 0u);

    std::vector<amdsmi_gpu_cache_entry_t> caches(num_caches);
    err = amdsmi_get_gpu_cache_entries(processor_handles_[i], &num_caches, caches.data());
    ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
    ASSERT_EQ(num_caches, caches.size());
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> entry_instances;
    for (uint32_t j = 0; j < num_caches; j++) {
      SCOPED_TRACE("cache entry " + std::to_string(j));
      std::cout << "\t\tCache Type " << j << ": level " << caches[j].cache_level << ", "
                << caches[j].cache_size << " KB, line " << caches[j].cache_line_size
                << " B, CUs shared " << caches[j].min_num_cu_shared << "-"
                << caches[j].max_num_cu_shared << ", instances " << caches[j].num_cache_instance
                << "\n";
      entry_instances[{caches[j].cache_level, caches[j].cache_properties, caches[j].cache_size}] +=
          caches[j].num_cache_instance;
      EXPECT_GT(caches[j].num_cache_instance, 0u);
      EXPECT_LE(caches[j].min_num_cu_shared, caches[j].max_num_cu_shared);
    }

    // A buffer one entry short gets the entries that fit, the full count, and
    // INSUFFICIENT_SIZE rather than a silently truncated success.
    if (num_caches > 1) {
      std::vector<amdsmi_gpu_cache_entry_t> short_caches(num_caches - 1);
      uint32_t short_count = num_caches - 1;
      err = amdsmi_get_gpu_cache_entries(processor_handles_[i], &short_count, short_caches.data());
      EXPECT_EQ(err, AMDSMI_STATUS_INSUFFICIENT_SIZE);
      EXPECT_EQ(short_count, num_caches);
      EXPECT_EQ(short_caches.back().cache_size, caches[num_caches - 2].cache_size);
    }

    err = amdsmi_get_gpu_cache_entries(processor_handles_[i], nullptr, caches.data());
    EXPECT_EQ(err, AMDSMI_STATUS_INVAL);

    IF_VERB(STANDARD) {
      std::cout << "\n\n";
      std::cout << "\t**GPU CACHE INFO: Using static struct (Backwards Compatibility):\n";
    }
    amdsmi_gpu_cache_info_t res = {};
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_cache_info", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_cache_info(processor_handles_[i], &res);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
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

    // Both queries count every cache instance: per level, properties and size,
    // the instance totals match.
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> legacy_instances;
    for (uint32_t j = 0; j < res.num_cache_types; j++) {
      legacy_instances[{res.cache[j].cache_level, res.cache[j].cache_properties,
                        res.cache[j].cache_size}] += res.cache[j].num_cache_instance;
    }
    EXPECT_EQ(entry_instances, legacy_instances);
  }
}
