/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_SIGNAL_ALLOCATION_VALIDATION_H_
#define ROCRTST_SUITES_FUNCTIONAL_SIGNAL_ALLOCATION_VALIDATION_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

class SignalAllocationValidationTest : public TestBase {
 public:
    SignalAllocationValidationTest();

    // @Brief: Destructor for the SignalAllocationValidationTest class
    virtual ~SignalAllocationValidationTest();

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

    // @Brief: Test that verifies signal handles are valid when HSA_STATUS_SUCCESS is returned
    void TestSignalAllocationValidation(void);
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_SIGNAL_ALLOCATION_VALIDATION_H_
