/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <iostream>
#include <vector>
#include "suites/functional/signal_allocation_validation.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

// Target number of signals to allocate for stress testing
static const uint32_t TARGET_SIGNAL_COUNT = 100000;

SignalAllocationValidationTest::SignalAllocationValidationTest()
    : TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("RocR Signal Allocation Validation Test");
  set_description("This test verifies that hsa_signal_create returns valid signal handles when HSA_STATUS_SUCCESS is returned, even under resource pressure");
}

SignalAllocationValidationTest::~SignalAllocationValidationTest(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void SignalAllocationValidationTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  return;
}

void SignalAllocationValidationTest::Run(void) {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void SignalAllocationValidationTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void SignalAllocationValidationTest::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void SignalAllocationValidationTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

/*
 * Test Name: TestSignalAllocationValidation
 * Scope: Conformance
 *
 * Purpose: Verifies that hsa_signal_create never returns HSA_STATUS_SUCCESS
 * with an invalid (zero) signal handle, even when allocating a very large
 * number of signals.
 *
 * Test Description:
 * 1) Allocate a very large number of signals (approximately 100,000)
 * 2) For each allocation attempt:
 *    a) If hsa_signal_create returns HSA_STATUS_SUCCESS, verify that
 *       the returned signal handle is valid (non-zero)
 *    b) If hsa_signal_create returns HSA_STATUS_ERROR_OUT_OF_RESOURCES,
 *       this is expected behavior when resources are exhausted - stop allocating
 *    c) Track all successfully allocated signals for cleanup
 * 3) Report any cases where HSA_STATUS_SUCCESS was returned with an invalid handle
 * 4) Destroy all successfully allocated signals
 *
 * Expected Results: If hsa_signal_create returns HSA_STATUS_SUCCESS, the signal
 * handle must be non-zero.
 */
void SignalAllocationValidationTest::TestSignalAllocationValidation(void) {
  hsa_status_t status;
  std::vector<hsa_signal_t> allocated_signals;
  uint32_t invalid_handle_count = 0;
  uint32_t success_count = 0;
  uint32_t failure_count = 0;
  uint32_t i;

  std::cout << "Attempting to allocate " << TARGET_SIGNAL_COUNT << " signals..." << std::endl;

  // Reserve space to avoid reallocations during the test
  allocated_signals.reserve(TARGET_SIGNAL_COUNT);

  // Attempt to allocate TARGET_SIGNAL_COUNT signals
  for (i = 0; i < TARGET_SIGNAL_COUNT; ++i) {
    hsa_signal_t signal;
    status = hsa_signal_create(1, 0, NULL, &signal);

    if (status == HSA_STATUS_SUCCESS) {
      success_count++;

      // This is the critical check: if HSA_STATUS_SUCCESS is returned,
      // the signal handle MUST be valid (non-zero)
      if (signal.handle == 0) {
        invalid_handle_count++;
        std::cout << "ERROR: hsa_signal_create returned HSA_STATUS_SUCCESS but signal handle is 0 at allocation "
                  << i << std::endl;
        // This is the bug we're testing for - should never happen after the fix
        FAIL() << "Signal allocation returned success with invalid (zero) handle";
      } else {
        // Valid signal, add to our tracking list
        allocated_signals.push_back(signal);
      }
    } else if (status == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
      // This is expected when we run out of signal resources
      failure_count++;
      std::cout << "Signal allocation failed with HSA_STATUS_ERROR_OUT_OF_RESOURCES after "
                << success_count << " successful allocations (expected under resource pressure)" << std::endl;
      break;
    } else {
      // Unexpected error code
      failure_count++;
      std::cout << "ERROR: Unexpected status code " << status << " at allocation " << i << std::endl;
      FAIL() << "hsa_signal_create returned unexpected status code: " << status;
    }

    // Print progress every 10,000 signals
    if ((i + 1) % 10000 == 0) {
      if(verbosity() > 0) {
        std::cout << "Allocated " << (i + 1) << " signals..." << std::endl;
      }
    }
  }

  std::cout << "\nAllocation Summary:" << std::endl;
  std::cout << "  Successful allocations: " << success_count << std::endl;
  std::cout << "  Failed allocations: " << failure_count << std::endl;
  std::cout << "  Invalid handles on success: " << invalid_handle_count << std::endl;

  // The main assertion: we should never have invalid handles when success is returned
  ASSERT_EQ(invalid_handle_count, 0) << "Found " << invalid_handle_count
                                      << " cases where HSA_STATUS_SUCCESS was returned with invalid handle";

  // Cleanup: destroy all successfully allocated signals
  std::cout << "Cleaning up " << allocated_signals.size() << " signals..." << std::endl;
  for (i = 0; i < allocated_signals.size(); ++i) {
    status = hsa_signal_destroy(allocated_signals[i]);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Failed to destroy signal " << i;

    // Print cleanup progress every 10,000 signals
    if ((i + 1) % 10000 == 0) {
      if(verbosity() > 0) {
        std::cout << "Destroyed " << (i + 1) << " signals..." << std::endl;
      }
    }
  }

  
  // Second pass: re-allocate signals to verify resource reclamation.
  // After destroying all signals from the first pass, we should be able
  // to allocate a similar number of signals again. This validates that
  // the runtime properly reclaims signal resources on destruction.
  const uint32_t first_pass_success_count = success_count;
  allocated_signals.clear();
  uint32_t second_pass_success_count = 0;
  uint32_t second_pass_failure_count = 0;
  uint32_t second_pass_invalid_handle_count = 0;

  std::cout << "\nSecond pass: re-allocating signals to test resource reclamation..." << std::endl;

  for (i = 0; i < TARGET_SIGNAL_COUNT; ++i) {
    hsa_signal_t signal;
    status = hsa_signal_create(1, 0, NULL, &signal);

    if (status == HSA_STATUS_SUCCESS) {
      second_pass_success_count++;

      if (signal.handle == 0) {
        second_pass_invalid_handle_count++;
        std::cout << "ERROR: hsa_signal_create returned HSA_STATUS_SUCCESS but signal handle is 0 at allocation "
                  << i << " (second pass)" << std::endl;
        FAIL() << "Signal allocation returned success with invalid (zero) handle (second pass)";
      } else {
        allocated_signals.push_back(signal);
      }
    } else if (status == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
      second_pass_failure_count++;
      std::cout << "Signal allocation failed with HSA_STATUS_ERROR_OUT_OF_RESOURCES after "
                << second_pass_success_count << " successful allocations on second pass" << std::endl;
      break;
    } else {
      second_pass_failure_count++;
      std::cout << "ERROR: Unexpected status code " << status << " at allocation " << i
                << " (second pass)" << std::endl;
      FAIL() << "hsa_signal_create returned unexpected status code on second pass: " << status;
    }

    if ((i + 1) % 10000 == 0) {
      if(verbosity() > 0) {
        std::cout << "Allocated " << (i + 1) << " signals (second pass)..." << std::endl;
      }
    }
  }

  std::cout << "\nSecond Pass Allocation Summary:" << std::endl;
  std::cout << "  Successful allocations: " << second_pass_success_count << std::endl;
  std::cout << "  Failed allocations: " << second_pass_failure_count << std::endl;
  std::cout << "  Invalid handles on success: " << second_pass_invalid_handle_count << std::endl;

  ASSERT_EQ(second_pass_invalid_handle_count, 0u)
      << "Found invalid handles on second pass";

  // Verify that resource reclamation worked: the second pass should be able
  // to allocate at least as many signals as the first pass (allowing for some
  // small variation due to other runtime activity).
  EXPECT_GE(second_pass_success_count, first_pass_success_count)
      << "Second pass allocated fewer signals than first pass ("
      << second_pass_success_count << " vs " << first_pass_success_count
      << "), indicating possible resource leak or incomplete reclamation";

  // Cleanup: destroy all signals allocated in the second pass
  std::cout << "Cleaning up " << allocated_signals.size()
            << " signals from second pass..." << std::endl;
  for (i = 0; i < allocated_signals.size(); ++i) {
    status = hsa_signal_destroy(allocated_signals[i]);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status)
        << "Failed to destroy signal " << i << " (second pass)";

    if ((i + 1) % 10000 == 0) {
      if(verbosity() > 0) {
        std::cout << "Destroyed " << (i + 1) << " signals (second pass)..." << std::endl;
      }
    }
  }
  
  std::cout << "Test completed successfully - all signal allocations returned valid handles" << std::endl;
}
