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
#include "power_read_write.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

#include "../test_common.h"
#include "amd_smi/amdsmi.h"

TestPowerReadWrite::TestPowerReadWrite() : TestBase() {
  set_title("AMDSMI Power Profiles Read/Write Test");
  set_description(
      "The Power Profiles tests verify that the power profile "
      "settings can be read and controlled properly.");
}

TestPowerReadWrite::~TestPowerReadWrite(void) {}

void TestPowerReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPowerReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestPowerReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPowerReadWrite::Close() {
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const char* power_profile_string(amdsmi_power_profile_preset_masks_t profile) {
  switch (profile) {
    case AMDSMI_PWR_PROF_PRST_CUSTOM_MASK:
      return "CUSTOM";
    case AMDSMI_PWR_PROF_PRST_VIDEO_MASK:
      return "VIDEO";
    case AMDSMI_PWR_PROF_PRST_POWER_SAVING_MASK:
      return "POWER SAVING";
    case AMDSMI_PWR_PROF_PRST_COMPUTE_MASK:
      return "COMPUTE";
    case AMDSMI_PWR_PROF_PRST_VR_MASK:
      return "VR";
    case AMDSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK:
      return "3D FULL SCREEN";
    case AMDSMI_PWR_PROF_PRST_BOOTUP_DEFAULT:
      return "BOOTUP DEFAULT";
    default:
      return "UNKNOWN";
  }
}

// Helper: Find a profile mask NOT in the available_profiles bitmask.
// Used for negative testing to validate error handling of unavailable profiles.
static amdsmi_power_profile_preset_masks_t find_unavailable_profile(
    uint64_t available_profiles) {
  uint64_t mask = 1;
  while (mask <= AMDSMI_PWR_PROF_PRST_LAST) {
    if (!(available_profiles & mask)) {
      return (amdsmi_power_profile_preset_masks_t)mask;
    }
    mask <<= 1;
  }
  return AMDSMI_PWR_PROF_PRST_INVALID;
}

// Helper: Extract all available profile masks into a vector.
// Iterates through the available_profiles bitmask and collects each set bit.
static std::vector<amdsmi_power_profile_preset_masks_t> get_available_profiles(
    const amdsmi_power_profile_status_t& status) {
  std::vector<amdsmi_power_profile_preset_masks_t> profiles;
  uint64_t mask = 1;
  while (mask <= AMDSMI_PWR_PROF_PRST_LAST) {
    if (status.available_profiles & mask) {
      profiles.push_back((amdsmi_power_profile_preset_masks_t)mask);
    }
    mask <<= 1;
  }
  return profiles;
}

void TestPowerReadWrite::Run(void) {
  amdsmi_status_t ret;
  amdsmi_power_profile_status_t status;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_power_profile_presets", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_gpu_power_profile_presets(processor_handles_[dv_ind], 0, &status);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_power_profile_presets", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_get_gpu_power_profile_presets(processor_handles_[dv_ind], 0, nullptr);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
      ASSERT_EQ(ret, AMDSMI_STATUS_NOT_SUPPORTED);
      continue;
    }
    CHK_ERR_ASRT(ret)

    // Verify api support checking functionality is working
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_power_profile_presets", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_gpu_power_profile_presets(processor_handles_[dv_ind], 0, nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);

    IF_VERB(STANDARD) {
      std::cout << "The available power profiles are:" << std::endl;
      uint64_t tmp = 1;
      while (tmp <= AMDSMI_PWR_PROF_PRST_LAST) {
        if ((tmp & status.available_profiles) == tmp) {
          std::cout << "\t" << power_profile_string((amdsmi_power_profile_preset_masks_t)tmp)
                    << std::endl;
        }
        tmp = tmp << 1;
      }
      std::cout << "The current power profile is: " << power_profile_string(status.current)
                << std::endl;
    }

    amdsmi_power_profile_preset_masks_t orig_profile = status.current;

    // Try setting the profile to a different power profile
    amdsmi_bit_field_t diff_profiles;
    amdsmi_power_profile_preset_masks_t new_prof;
    diff_profiles = status.available_profiles & (~status.current);

    if (diff_profiles & AMDSMI_PWR_PROF_PRST_COMPUTE_MASK) {
      new_prof = AMDSMI_PWR_PROF_PRST_COMPUTE_MASK;
    } else if (diff_profiles & AMDSMI_PWR_PROF_PRST_VIDEO_MASK) {
      new_prof = AMDSMI_PWR_PROF_PRST_VIDEO_MASK;
    } else if (diff_profiles & AMDSMI_PWR_PROF_PRST_VR_MASK) {
      new_prof = AMDSMI_PWR_PROF_PRST_VR_MASK;
    } else if (diff_profiles & AMDSMI_PWR_PROF_PRST_POWER_SAVING_MASK) {
      new_prof = AMDSMI_PWR_PROF_PRST_POWER_SAVING_MASK;
    } else if (diff_profiles & AMDSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK) {
      new_prof = AMDSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK;
    } else {
      std::cout << "No other non-custom power profiles to set to. Exiting." << std::endl;
      return;
    }

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_power_profile", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_set_gpu_power_profile(processor_handles_[dv_ind], 0, new_prof);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret)

    amdsmi_dev_perf_level_t pfl;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_gpu_perf_level(processor_handles_[dv_ind], &pfl);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret)
    ASSERT_EQ(pfl, AMDSMI_DEV_PERF_LEVEL_MANUAL);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_power_profile_presets", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_gpu_power_profile_presets(processor_handles_[dv_ind], 0, &status);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret)

    ASSERT_EQ(status.current, new_prof);

    // Reset the state of perf level and power profile after testing
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_set_gpu_perf_level(processor_handles_[dv_ind], AMDSMI_DEV_PERF_LEVEL_AUTO);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret);

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_power_profile", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_set_gpu_power_profile(processor_handles_[dv_ind], 0, orig_profile);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret);
  }
}

// TEST CASE 1: Test All Available Profiles
void TestPowerReadWrite::TestAllAvailableProfiles() {
  amdsmi_status_t ret;
  amdsmi_power_profile_status_t status;

  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  IF_VERB(STANDARD) {
    std::cout << "\n=== TEST 1: Testing All Available Profiles ===" << std::endl;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_power_profile_presets", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_gpu_power_profile_presets(processor_handles_[dv_ind], 0, &status);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "Power profiles not supported on this GPU, skipping" << std::endl;
      }
      continue;
    }
    CHK_ERR_ASRT(ret);

    amdsmi_power_profile_preset_masks_t orig_profile = status.current;
    std::vector<amdsmi_power_profile_preset_masks_t> profiles_to_test =
        get_available_profiles(status);

    ASSERT_FALSE(profiles_to_test.empty())
        << "At least one profile should be available";

    IF_VERB(STANDARD) {
      std::cout << "\n>>> Testing " << profiles_to_test.size()
                << " available profiles (including CUSTOM if available)" << std::endl;
    }

    // Test switching to EACH profile
    for (size_t i = 0; i < profiles_to_test.size(); ++i) {
      auto profile = profiles_to_test[i];

      IF_VERB(STANDARD) {
        std::cout << "\n[" << (i + 1) << "/" << profiles_to_test.size() << "] "
                  << "Testing profile: " << power_profile_string(profile) << std::endl;
      }

      // Set profile
      DISPLAY_AMDSMI_API("amdsmi_set_gpu_power_profile",
                         "gpu=" + std::to_string(dv_ind) + ", profile=" + power_profile_string(profile),
                         VERB(STANDARD));
      ret = amdsmi_set_gpu_power_profile(processor_handles_[dv_ind], 0, profile);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);

      // Skip CUSTOM profile if not supported (Navi 3x quirk: appears in available list but can't be set)
      if (ret == AMDSMI_STATUS_NOT_SUPPORTED && profile == AMDSMI_PWR_PROF_PRST_CUSTOM_MASK) {
        IF_VERB(STANDARD) {
          std::cout << "  CUSTOM profile not supported for setting, skipping" << std::endl;
        }
        continue;
      }

      ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS)
          << "Failed to set " << power_profile_string(profile);

      // Verify it switched
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_power_profile_presets", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_get_gpu_power_profile_presets(processor_handles_[dv_ind], 0, &status);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret);
      ASSERT_EQ(status.current, profile)
          << "Profile didn't switch to " << power_profile_string(profile);

      IF_VERB(STANDARD) {
        std::cout << "  Profile switched successfully" << std::endl;
      }

      // Verify perf level is MANUAL
      amdsmi_dev_perf_level_t pfl;
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_get_gpu_perf_level(processor_handles_[dv_ind], &pfl);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret);
      ASSERT_EQ(pfl, AMDSMI_DEV_PERF_LEVEL_MANUAL);

      IF_VERB(STANDARD) {
        std::cout << "  Performance level is MANUAL" << std::endl;
      }

      usleep(100000);  // 100ms - Allow profile to stabilize
    }

    IF_VERB(STANDARD) {
      std::cout << "\n>>> Successfully tested all " << profiles_to_test.size()
                << " profiles!" << std::endl;
    }

    // Restore original state
    IF_VERB(STANDARD) {
      std::cout << "\nRestoring original state..." << std::endl;
    }
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level", "gpu=" + std::to_string(dv_ind) + ", level=AUTO",
                       VERB(STANDARD));
    ret = amdsmi_set_gpu_perf_level(processor_handles_[dv_ind],
                                    AMDSMI_DEV_PERF_LEVEL_AUTO);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret);

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_power_profile",
                       "gpu=" + std::to_string(dv_ind) + ", profile=" + power_profile_string(orig_profile),
                       VERB(STANDARD));
    ret = amdsmi_set_gpu_power_profile(processor_handles_[dv_ind], 0, orig_profile);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret);

    IF_VERB(STANDARD) {
      std::cout << "Restored to: " << power_profile_string(orig_profile) << std::endl;
    }
  }
}

// TEST CASE 2: Sequential Profile Switching
void TestPowerReadWrite::TestSequentialProfileSwitching() {
  amdsmi_status_t ret;
  amdsmi_power_profile_status_t status;

  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  IF_VERB(STANDARD) {
    std::cout << "\n=== TEST 2: Sequential Profile Switching ===" << std::endl;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_power_profile_presets", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_gpu_power_profile_presets(processor_handles_[dv_ind], 0, &status);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "Power profiles not supported, skipping" << std::endl;
      }
      continue;
    }
    CHK_ERR_ASRT(ret);

    amdsmi_power_profile_preset_masks_t orig_profile = status.current;
    const std::vector<amdsmi_power_profile_preset_masks_t> profiles =
        get_available_profiles(status);

    if (profiles.size() < 2) {
      IF_VERB(STANDARD) {
        std::cout << "Only 1 profile available, skipping" << std::endl;
      }
      continue;
    }

    const int num_cycles = 3;
    IF_VERB(STANDARD) {
      std::cout << "\n>>> Cycling through " << profiles.size()
                << " profiles " << num_cycles << " times" << std::endl;
    }

    for (int cycle = 0; cycle < num_cycles; ++cycle) {
      IF_VERB(STANDARD) {
        std::cout << "\n=== Cycle " << (cycle + 1) << "/" << num_cycles
                  << " ===" << std::endl;
      }

      for (size_t i = 0; i < profiles.size(); ++i) {
        auto profile = profiles[i];

        IF_VERB(STANDARD) {
          std::cout << "  [" << (i + 1) << "/" << profiles.size() << "] "
                    << "Setting: " << power_profile_string(profile) << std::flush;
        }

        DISPLAY_AMDSMI_API("amdsmi_set_gpu_power_profile",
                           "gpu=" + std::to_string(dv_ind) + ", profile=" + power_profile_string(profile),
                           VERB(STANDARD));
        ret = amdsmi_set_gpu_power_profile(processor_handles_[dv_ind], 0, profile);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);

        // Skip CUSTOM profile if not supported (Navi 3x quirk)
        if (ret == AMDSMI_STATUS_NOT_SUPPORTED && profile == AMDSMI_PWR_PROF_PRST_CUSTOM_MASK) {
          IF_VERB(STANDARD) {
            std::cout << " (skipped - not supported)" << std::endl;
          }
          continue;
        }

        ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);

        DISPLAY_AMDSMI_API("amdsmi_get_gpu_power_profile_presets", "gpu=" + std::to_string(dv_ind),
                           VERB(STANDARD));
        ret = amdsmi_get_gpu_power_profile_presets(processor_handles_[dv_ind], 0, &status);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
        CHK_ERR_ASRT(ret);
        ASSERT_EQ(status.current, profile);

        IF_VERB(STANDARD) {
          std::cout << std::endl;
        }
        usleep(50000);  // 50ms - Prevent rapid switching
      }
    }

    IF_VERB(STANDARD) {
      std::cout << "\n>>> Successfully completed " << num_cycles << " cycles!" << std::endl;
    }

    // Restore
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level", "gpu=" + std::to_string(dv_ind) + ", level=AUTO",
                       VERB(STANDARD));
    ret = amdsmi_set_gpu_perf_level(processor_handles_[dv_ind],
                                    AMDSMI_DEV_PERF_LEVEL_AUTO);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret);

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_power_profile",
                       "gpu=" + std::to_string(dv_ind) + ", profile=" + power_profile_string(orig_profile),
                       VERB(STANDARD));
    ret = amdsmi_set_gpu_power_profile(processor_handles_[dv_ind], 0, orig_profile);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret);
  }
}

// TEST CASE 3: Invalid Profile Handling
void TestPowerReadWrite::TestInvalidProfileHandling() {
  amdsmi_status_t ret;
  amdsmi_power_profile_status_t status;

  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  IF_VERB(STANDARD) {
    std::cout << "\n=== TEST 3: Invalid Profile Handling ===" << std::endl;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_power_profile_presets", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_gpu_power_profile_presets(processor_handles_[dv_ind], 0, &status);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "Power profiles not supported, skipping" << std::endl;
      }
      continue;
    }
    CHK_ERR_ASRT(ret);

    amdsmi_power_profile_preset_masks_t orig_profile = status.current;

    // Test 1: Invalid profile mask
    IF_VERB(STANDARD) {
      std::cout << "\n[Test 1] Invalid profile mask (0xFFFFFFFFFFFFFFFF)" << std::endl;
    }
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_power_profile",
                       "gpu=" + std::to_string(dv_ind) + ", profile=INVALID",
                       VERB(STANDARD));
    ret = amdsmi_set_gpu_power_profile(processor_handles_[dv_ind], 0,
                                       AMDSMI_PWR_PROF_PRST_INVALID);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
    ASSERT_NE(ret, AMDSMI_STATUS_SUCCESS)
        << "Should reject invalid profile mask";
    IF_VERB(STANDARD) {
      std::cout << "  Rejected with status: " << ret << std::endl;
    }

    // Test 2: Zero mask
    IF_VERB(STANDARD) {
      std::cout << "\n[Test 2] Zero profile mask" << std::endl;
    }
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_power_profile",
                       "gpu=" + std::to_string(dv_ind) + ", profile=0",
                       VERB(STANDARD));
    ret = amdsmi_set_gpu_power_profile(processor_handles_[dv_ind], 0,
                                       (amdsmi_power_profile_preset_masks_t)0);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
    ASSERT_NE(ret, AMDSMI_STATUS_SUCCESS);
    IF_VERB(STANDARD) {
      std::cout << "  Rejected with status: " << ret << std::endl;
    }

    // Test 3: Multiple bits set
    if ((status.available_profiles & AMDSMI_PWR_PROF_PRST_COMPUTE_MASK) &&
        (status.available_profiles & AMDSMI_PWR_PROF_PRST_VIDEO_MASK)) {

      IF_VERB(STANDARD) {
        std::cout << "\n[Test 3] Multiple profiles (COMPUTE | VIDEO)" << std::endl;
      }
      amdsmi_power_profile_preset_masks_t multi =
          (amdsmi_power_profile_preset_masks_t)(
              AMDSMI_PWR_PROF_PRST_COMPUTE_MASK | AMDSMI_PWR_PROF_PRST_VIDEO_MASK);
      DISPLAY_AMDSMI_API("amdsmi_set_gpu_power_profile",
                         "gpu=" + std::to_string(dv_ind) + ", profile=COMPUTE|VIDEO",
                         VERB(STANDARD));
      ret = amdsmi_set_gpu_power_profile(processor_handles_[dv_ind], 0, multi);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
      ASSERT_NE(ret, AMDSMI_STATUS_SUCCESS);
      IF_VERB(STANDARD) {
        std::cout << "  Rejected with status: " << ret << std::endl;
      }
    }

    // Test 4: Unavailable profile
    amdsmi_power_profile_preset_masks_t unavailable =
        find_unavailable_profile(status.available_profiles);

    if (unavailable != AMDSMI_PWR_PROF_PRST_INVALID) {
      IF_VERB(STANDARD) {
        std::cout << "\n[Test 4] Unavailable profile" << std::endl;
      }
      DISPLAY_AMDSMI_API("amdsmi_set_gpu_power_profile",
                         "gpu=" + std::to_string(dv_ind) + ", profile=unavailable",
                         VERB(STANDARD));
      ret = amdsmi_set_gpu_power_profile(processor_handles_[dv_ind], 0, unavailable);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_NOT_SUPPORTED);
      ASSERT_NE(ret, AMDSMI_STATUS_SUCCESS);
      IF_VERB(STANDARD) {
        std::cout << "  Rejected with status: " << ret << std::endl;
      }
    }

    // Test 5: Verify profile unchanged
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_power_profile_presets", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_gpu_power_profile_presets(processor_handles_[dv_ind], 0, &status);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret);
    ASSERT_EQ(status.current, orig_profile);
    IF_VERB(STANDARD) {
      std::cout << "\n  Profile unchanged after invalid attempts" << std::endl;
    }
  }
}

