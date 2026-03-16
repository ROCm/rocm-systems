/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_CUID_H
#define ROCRTST_SUITES_FUNCTIONAL_CUID_H

#include "suites/test_common/test_base.h"
#include "common/base_rocr_utils.h"

class CuidTest : public TestBase {
public:
    explicit CuidTest();

    // @brief: Destructor for test case of CuidTest
    virtual ~CuidTest();

    // @Brief: Setup the environment for measurement
    virtual void SetUp();

    // @Brief: Core measurement execution
    virtual void Run();

    // @Brief: Clean up and retrive the resource
    virtual void Close();

    // @Brief: Display  results
    virtual void DisplayResults() const;

    // @Brief: Display information about what this test does
    virtual void DisplayTestInfo(void);

    // @brief Test to read in all GPU cuids from file
    void ValidateGpuCuidTest();
};

#endif // ROCRTST_SUITES_FUNCTIONAL_CUID_H