/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_SIGNAL_WAIT_MULTI_H_
#define ROCRTST_SUITES_FUNCTIONAL_SIGNAL_WAIT_MULTI_H_

#include <vector>

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

class SignalWaitMultiTest : public TestBase {
 public:
    SignalWaitMultiTest();

    // @Brief: Destructor for the SignalWaitMultiTest class
    virtual ~SignalWaitMultiTest();

    // @Brief: Setup the environment for measurement
    virtual void SetUp();

    // @Brief: Core measurement execution
    virtual void Run();

    // @Brief: Clean up and retrieve the resource
    virtual void Close();

    // @Brief: Display results
    virtual void DisplayResults() const;

    // @Brief: Display information about what this test does
    virtual void DisplayTestInfo(void);

    // @Brief: Verify hsa_amd_signal_wait_any reports a nonzero satisfying index
    void TestWaitAnyNonzeroSatisfyingIndex(void);

    // @Brief: Verify hsa_amd_signal_wait_any compacts conds/values with signals
    void TestWaitAnyCompactsConditionsAndValues(void);

    // @Brief: Verify hsa_amd_signal_wait_all reports satisfying values
    void TestWaitAllReportsSatisfyingValues(void);

 protected:
    // @Brief: Timeout hint passed to the wait APIs, in timestamp ticks
    uint64_t timeout_hint_ = 0;

    // @Brief: Signals created during a test, tracked for cleanup in Close()
    std::vector<hsa_signal_t> signals_;

    // @Brief: Create a signal and track it for cleanup on success
    hsa_status_t CreateSignal(hsa_signal_value_t initial_value, hsa_signal_t* signal);
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_SIGNAL_WAIT_MULTI_H_
