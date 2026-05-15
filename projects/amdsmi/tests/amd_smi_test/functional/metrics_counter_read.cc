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

#include <cstdint>

#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include "amd_smi/amdsmi.h"
#include "metrics_counter_read.h"


TestMetricsCounterRead::TestMetricsCounterRead() : TestBase() {
  set_title("AMDSMI GPU Metrics Counter Read Test");
  set_description("The GPU Metrics Counter tests verifies that "
                  "the gpu metrics counter info can be read properly.");
}

TestMetricsCounterRead::~TestMetricsCounterRead(void) {
}

void TestMetricsCounterRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestMetricsCounterRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestMetricsCounterRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestMetricsCounterRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestMetricsCounterRead::Run(void) {
  amdsmi_status_t err;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);

    IF_VERB(STANDARD) {
        std::cout << "\t**GPU METRICS ENERGY COUNTER:\n";
    }

    uint64_t energy_accumulator;
    uint64_t timestamp;
    float counter_resolution;
    err = amdsmi_get_energy_count(processor_handles_[i], &energy_accumulator, &counter_resolution, &timestamp);
    if (err != AMDSMI_STATUS_SUCCESS) {
      if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" <<
          "Not supported on this machine" << std::endl;
          return;
        }
      }
    } else {
      CHK_ERR_ASRT(err);
      IF_VERB(STANDARD) {
          std::cout << std::dec << "energy_accumulator counter="
          << energy_accumulator << '\n';
          std::cout << "energy_accumulator in uJ="
          << static_cast<double>((static_cast<double>(energy_accumulator) * counter_resolution)) << '\n';
          std::cout << std::dec << "timestamp="
          << timestamp << '\n';
      }
    }

    // Verify api support checking functionality is working
    err = amdsmi_get_energy_count(processor_handles_[i], nullptr, nullptr, nullptr);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    // Coarse Grain counters
    constexpr uint32_t kUTILIZATION_COUNTERS(3);
    amdsmi_utilization_counter_t utilization_counters[kUTILIZATION_COUNTERS];
    utilization_counters[0].type = AMDSMI_COARSE_GRAIN_GFX_ACTIVITY;
    utilization_counters[1].type = AMDSMI_COARSE_GRAIN_MEM_ACTIVITY;
    utilization_counters[2].type = AMDSMI_COARSE_DECODER_ACTIVITY;

    err = amdsmi_get_utilization_count(processor_handles_[i], utilization_counters,
                    kUTILIZATION_COUNTERS, &timestamp);
    if (err != AMDSMI_STATUS_SUCCESS) {
      if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" <<
          "amdsmi_get_utilization_count(): Not supported on this machine" << std::endl;
          return;
        }
      }
    } else {
      CHK_ERR_ASRT(err);
      IF_VERB(STANDARD) {
          std::cout << "\n\namdsmi_get_utilization_count() : COARSE GRAIN ACTIVITIES" << "\n";
          for (auto idx = uint32_t(0); idx < kUTILIZATION_COUNTERS; ++idx) {
              switch (utilization_counters[idx].type) {
                  case AMDSMI_COARSE_GRAIN_GFX_ACTIVITY:
                      std::cout << "-> gfx_activity: [" << utilization_counters[idx].fine_value_count << "]" << "\n";
                      break;

                  case AMDSMI_COARSE_GRAIN_MEM_ACTIVITY:
                      std::cout << "-> mem_activity: [" << utilization_counters[idx].fine_value_count << "]" << "\n";;
                      break;

                  case AMDSMI_COARSE_DECODER_ACTIVITY:
                      std::cout << "-> decoder_activity: [" << utilization_counters[idx].fine_value_count << "]" << "\n";
                      break;

                  default:
                      break;
              }

              for (auto val_idx = uint16_t(0); val_idx < utilization_counters[idx].fine_value_count; ++val_idx) {
                  std::cout << "\t" << std::dec << utilization_counters[idx].value << "\n";
              }
          }

          std::cout << std::dec << "timestamp=" << timestamp << '\n';
      }
    }

    // Fine Grain counters
    utilization_counters[0].type = AMDSMI_FINE_GRAIN_GFX_ACTIVITY;
    utilization_counters[1].type = AMDSMI_FINE_GRAIN_MEM_ACTIVITY;
    utilization_counters[2].type = AMDSMI_FINE_DECODER_ACTIVITY;
    err = amdsmi_get_utilization_count(processor_handles_[i], utilization_counters,
                    kUTILIZATION_COUNTERS, &timestamp);
    if (err != AMDSMI_STATUS_SUCCESS) {
      if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" <<
          "amdsmi_get_utilization_count(): Not supported on this machine" << std::endl;
          return;
        }
      }
    } else {
      CHK_ERR_ASRT(err);
      IF_VERB(STANDARD) {
          std::cout << "\n\namdsmi_get_utilization_count() : FINE GRAIN ACTIVITIES" << "\n";
          for (auto idx = uint32_t(0); idx < kUTILIZATION_COUNTERS; ++idx) {
              switch (utilization_counters[idx].type) {
                  case AMDSMI_FINE_GRAIN_GFX_ACTIVITY:
                      std::cout << "-> gfx_activity: [" << utilization_counters[idx].fine_value_count << "]" << "\n";
                      break;

                  case AMDSMI_FINE_GRAIN_MEM_ACTIVITY:
                      std::cout << "-> mem_activity: [" << utilization_counters[idx].fine_value_count << "]" << "\n";;
                      break;

                  case AMDSMI_FINE_DECODER_ACTIVITY:
                      std::cout << "-> decoder_activity: [" << utilization_counters[idx].fine_value_count << "]" << "\n";
                      break;

                  default:
                      break;
              }

              for (auto val_idx = uint16_t(0); val_idx < utilization_counters[idx].fine_value_count; ++val_idx) {
                  std::cout << "\t" << std::dec << utilization_counters[idx].fine_value[val_idx] << "\n";
              }
          }

          std::cout << std::dec << "timestamp=" << timestamp << '\n';
      }
    }

    // Verify api support checking functionality is working
    err = amdsmi_get_utilization_count(processor_handles_[i], nullptr,
                    1 , nullptr);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
  }  // end for
}
