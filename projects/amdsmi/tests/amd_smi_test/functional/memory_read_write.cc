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

#include "memory_read_write.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"

/**
 * RAII helper to set and automatically restore an environment variable
 */
class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value) : name_(name) {
    const char* old_val = std::getenv(name);
    if (old_val) {
      old_value_ = old_val;
      has_old_value_ = true;
    } else {
      has_old_value_ = false;
    }
    setenv(name, value, 1);
  }
  ~ScopedEnvVar() {
    if (has_old_value_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::string old_value_;
  bool has_old_value_;
};

TestMemoryReadWrite::TestMemoryReadWrite() : TestBase() {
  set_title("AMDSMI Memory Configuration Read/Write Test");
  set_description(
      "The Memory Configuration tests verify that "
      "UMA carveout and TTM configuration can be read and written properly.");
}

TestMemoryReadWrite::~TestMemoryReadWrite(void) {}

void TestMemoryReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestMemoryReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestMemoryReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestMemoryReadWrite::Close() {
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestMemoryReadWrite::Run(void) {
  amdsmi_status_t err;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  // Test UMA Carveout (per-GPU)
  IF_VERB(STANDARD) { std::cout << "\n=== Testing UMA Carveout (VRAM) ===" << std::endl; }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);

    amdsmi_uma_carveout_info_t uma_info;
    memset(&uma_info, 0, sizeof(uma_info));

    err = amdsmi_get_gpu_uma_carveout_info(processor_handles_[i], &uma_info);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**UMA Carveout not supported on this device." << std::endl;
      }
      ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
      continue;
    }

    CHK_ERR_ASRT(err)

    IF_VERB(STANDARD) {
      std::cout << "\t**Current UMA Carveout Index: " << uma_info.current_index << std::endl;
      std::cout << "\t**Number of Options: " << uma_info.num_options << std::endl;

      for (uint32_t j = 0; j < uma_info.num_options; ++j) {
        std::cout << "\t  Option " << uma_info.options[j].index << ": "
                  << uma_info.options[j].description << std::endl;
      }
    }

    // Validate that current_index is within valid range
    ASSERT_LT(uma_info.current_index, uma_info.num_options);
    ASSERT_GT(strlen(uma_info.options[uma_info.current_index].description), 0u);

    // Validate that num_options is reasonable (should be > 0 if supported)
    ASSERT_GT(uma_info.num_options, 0u);
    ASSERT_LE(uma_info.num_options, 16u);  // Max array size

    // Count and validate descriptions for valid options
    uint32_t valid_count = 0;
    for (uint32_t j = 0; j < uma_info.num_options; ++j) {
      if (strlen(uma_info.options[j].description) > 0) {
        valid_count++;
        // If it's valid, index must match position (as per implementation)
        ASSERT_EQ(uma_info.options[j].index, j);
      }
    }
    ASSERT_GT(valid_count, 0u);

    // Test setting UMA carveout in DRY_RUN mode
    IF_VERB(STANDARD) {
      std::cout << "\t**Testing set UMA carveout in DRY_RUN mode..." << std::endl;
    }

    {
      // Enable DRY_RUN mode for testing (RAII)
      ScopedEnvVar dry_run("AMDSMI_DRY_RUN", "1");

      // Test setting to current value (should succeed)
      err = amdsmi_set_gpu_uma_carveout(processor_handles_[i], uma_info.current_index);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t  Set to current index succeeded (DRY_RUN)" << std::endl;
      }

      // Test setting to a different valid index if available
      if (valid_count > 1) {
        for (uint32_t j = 0; j < uma_info.num_options; ++j) {
          if (j != uma_info.current_index && strlen(uma_info.options[j].description) > 0) {
            err = amdsmi_set_gpu_uma_carveout(processor_handles_[i], j);
            CHK_ERR_ASRT(err)
            IF_VERB(STANDARD) {
              std::cout << "\t  Set to different index " << j << " succeeded (DRY_RUN)"
                        << std::endl;
            }
            break;
          }
        }
      }

      // Test setting to invalid index (should fail with AMDSMI_STATUS_INVAL)
      err = amdsmi_set_gpu_uma_carveout(processor_handles_[i], uma_info.num_options + 10);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
      IF_VERB(STANDARD) {
        std::cout << "\t  Invalid index correctly rejected (DRY_RUN)" << std::endl;
      }
    }
  }

  // Test TTM Configuration (system-wide)
  IF_VERB(STANDARD) {
    std::cout << "\n=== Testing TTM Configuration (GTT/Shared Memory) ===" << std::endl;
  }

  amdsmi_ttm_info_t ttm_info;
  memset(&ttm_info, 0, sizeof(ttm_info));

  err = amdsmi_get_ttm_info(&ttm_info);
  if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
    IF_VERB(STANDARD) {
      std::cout << "\t**TTM pages_limit not supported on this system." << std::endl;
    }
    ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
  } else {
    CHK_ERR_ASRT(err)

    IF_VERB(STANDARD) {
      std::cout << "\t**Current TTM Pages Limit: " << ttm_info.current_pages << " pages"
                << std::endl;

      // Convert to GB for display
      const long page_size = sysconf(_SC_PAGESIZE);
      double gb =
          (static_cast<double>(ttm_info.current_pages) * page_size) / (1024.0 * 1024.0 * 1024.0);
      std::cout << "\t**Current TTM Size: " << gb << " GB" << std::endl;
    }

    // Validate that pages value is reasonable
    ASSERT_GT(ttm_info.current_pages, 0u);

    // Test TTM write operations in DRY_RUN mode
    IF_VERB(STANDARD) {
      std::cout << "\t**Testing TTM write operations in DRY_RUN mode..." << std::endl;
    }

    {
      // Enable DRY_RUN mode for testing (RAII)
      ScopedEnvVar dry_run("AMDSMI_DRY_RUN", "1");

      // Test setting TTM pages limit to current value
      err = amdsmi_set_ttm_pages_limit(ttm_info.current_pages);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t  Set TTM to current value succeeded (DRY_RUN)" << std::endl;
      }

      // Test setting TTM to a different value
      uint64_t test_pages = ttm_info.current_pages / 2;
      if (test_pages > 0) {
        err = amdsmi_set_ttm_pages_limit(test_pages);
        CHK_ERR_ASRT(err)
        IF_VERB(STANDARD) {
          std::cout << "\t  Set TTM to different value succeeded (DRY_RUN)" << std::endl;
        }
      }

      // Test setting TTM to 0 (should fail with AMDSMI_STATUS_INVAL)
      err = amdsmi_set_ttm_pages_limit(0);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
      IF_VERB(STANDARD) {
        std::cout << "\t  Invalid pages value (0) correctly rejected (DRY_RUN)" << std::endl;
      }

      // Test resetting TTM pages limit
      err = amdsmi_reset_ttm_pages_limit();
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) { std::cout << "\t  Reset TTM succeeded (DRY_RUN)" << std::endl; }
    }
  }

  IF_VERB(STANDARD) { std::cout << "\n=== Memory Configuration Tests Completed ===" << std::endl; }
}
